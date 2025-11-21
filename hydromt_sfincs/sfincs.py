"""
SfincsModel class
"""

# %% Import packages
from __future__ import annotations

import logging
import os
from os.path import dirname, join
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
from pyproj import CRS
import xarray as xr
import xugrid as xu

# %% Import model components
from hydromt.model import Model

from hydromt_sfincs import DATADIR, plots, utils

# Input component
from hydromt_sfincs.components.config import SfincsConfig

# Regular Grid components
from hydromt_sfincs.components.grid import (
    SfincsElevation,
    SfincsGrid,
    SfincsInfiltration,
    SfincsInitialConditions,
    SfincsMask,
    SfincsRoughness,
    SfincsStorageVolume,
    SfincsSubgridTable,
)

# Quadtree components
from hydromt_sfincs.components.quadtree import (
    SfincsQuadtreeGrid,
    SfincsQuadtreeElevation,
    SfincsQuadtreeInitialConditions,
    SfincsQuadtreeInfiltration,
    SfincsQuadtreeMask,
    SfincsQuadtreeStorageVolume,
    SfincsQuadtreeSubgridTable,
    SnapWaveQuadtreeMask,
)

# Boundary conditions / forcing components
from hydromt_sfincs.components.forcing import (
    SfincsDischargePoints,
    SfincsPrecipitation,
    SfincsPressure,
    SfincsRivers,
    SfincsWind,
    SfincsWaterLevel,
    SnapWaveBoundaryConditions,
)

# Geomatries/structures components
from hydromt_sfincs.components.geometries import (
    SfincsCrossSections,
    SfincsDrainageStructures,
    SfincsObservationPoints,
    SfincsThinDams,
    SfincsWaveMakers,
    SfincsWeirs,
)

# output / visualization types:
from hydromt_sfincs.components.output import SfincsOutput

__all__ = ["SfincsModel"]
__hydromt_eps__ = ["SfincsModel"]  # core entrypoints
logger = logging.getLogger(f"hydromt.{__name__}")


# %% SfincsModel class - in V1 style:
class SfincsModel(Model):
    """SFINCS model class."""

    name: str = "sfincs"

    # Grouped component definitions
    _CONFIG_COMPONENTS = {"config": SfincsConfig}
    _GRID_COMPONENTS = {
        "grid": SfincsGrid,
        "elevation": SfincsElevation,
        "mask": SfincsMask,
        "infiltration": SfincsInfiltration,
        "roughness": SfincsRoughness,
        "storage_volume": SfincsStorageVolume,
        "subgrid": SfincsSubgridTable,
        "initial_conditions": SfincsInitialConditions,
    }
    _QUADTREE_COMPONENTS = {
        "quadtree_grid": SfincsQuadtreeGrid,
        "quadtree_elevation": SfincsQuadtreeElevation,
        "quadtree_mask": SfincsQuadtreeMask,
        "quadtree_infiltration": SfincsQuadtreeInfiltration,
        "quadtree_storage_volume": SfincsQuadtreeStorageVolume,
        "quadtree_initial_conditions": SfincsQuadtreeInitialConditions,
        "quadtree_subgrid": SfincsQuadtreeSubgridTable,
        "quadtree_snapwave_mask": SnapWaveQuadtreeMask,
    }
    _GEOMETRY_COMPONENTS = {
        "observation_points": SfincsObservationPoints,
        "cross_sections": SfincsCrossSections,
        "thin_dams": SfincsThinDams,
        "weirs": SfincsWeirs,
        "wave_makers": SfincsWaveMakers,
        "drainage_structures": SfincsDrainageStructures,
    }
    _FORCING_COMPONENTS = {
        "rivers": SfincsRivers,
        "water_level": SfincsWaterLevel,
        "discharge_points": SfincsDischargePoints,
        "snapwave_boundary_conditions": SnapWaveBoundaryConditions,
        "precipitation": SfincsPrecipitation,
        "pressure": SfincsPressure,
        "wind": SfincsWind,
    }
    _OUTPUT_COMPONENTS = {"output": SfincsOutput}

    # Combine all component dictionaries
    _ALL_COMPONENTS = {
        **_CONFIG_COMPONENTS,
        **_GRID_COMPONENTS,
        **_QUADTREE_COMPONENTS,
        **_GEOMETRY_COMPONENTS,
        **_FORCING_COMPONENTS,
        **_OUTPUT_COMPONENTS,
    }

    # Precompute sets of component names for checking later ...
    _REGULAR_GRID_NAMES = set(_GRID_COMPONENTS.keys())
    _QUADTREE_GRID_NAMES = set(_QUADTREE_COMPONENTS.keys())

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        write_gis: bool = True,
        data_libs: Union[List[str], str] = None,
        **catalog_keys,
    ):
        """
        The SFINCS model class (SfincsModel) contains methods to read, write, setup and edit
        `SFINCS <https://sfincs.readthedocs.io/en/latest/>`_ models.

        Parameters
        ----------
        root: str, Path, optional
            Path to model folder
        mode: {'w', 'r+', 'r'}
            Open model in write, append or reading mode, by default 'w'
        write_gis: bool
            Write model files additionally to geotiff and geojson, by default True
        data_libs: List, str
            List of data catalog yaml files, by default None
        **catalog_keys:
            Additional keyword arguments to be passed down to the DataCatalog.
        """

        # define some default model properties
        self.grid_type = "regular"
        self.write_gis = write_gis

        super().__init__(
            root=root,
            mode=mode,
            data_libs=data_libs,
            **catalog_keys,
        )

        # Register all components and create properties dynamically
        for name, cls in self._ALL_COMPONENTS.items():
            instance = cls(self)
            self.add_component(name, instance)

    def __del__(self):
        """Close the model and remove the logger file handler."""
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and "hydromt.log" in handler.baseFilename
            ):
                handler.close()
                logger.removeHandler(handler)

    ## Real properties of the model ##
    @property
    def crs(self) -> CRS | None:
        """Returns the model crs"""
        if self.grid_type == "regular":
            return self.grid.crs
        elif self.grid_type == "quadtree":
            return self.quadtree_grid.crs

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the active model cells."""
        # NOTE overwrites property in GridModel
        region = gpd.GeoDataFrame()
        if self.grid_type == "regular":
            region = self.grid.region
        elif self.grid_type == "quadtree":
            region = self.quadtree_grid.exterior
        return region

    @property
    def bounds(self) -> List[float]:
        """Returns the bounding box of the model grid."""
        if self.grid_type == "regular":
            return self.grid.empty_mask.raster.bounds
        elif self.grid_type == "quadtree":
            # By are we getting the total bounds of the mask and not the grid?
            return self.quadtree_grid.empty_mask.ugrid.total_bounds

    @property
    def bbox(self) -> tuple:
        """Returns the bounding box in WGS 84 of the model grid."""
        if self.grid_type == "regular":
            return self.grid.empty_mask.raster.transform_bounds(4326)
        elif self.grid_type == "quadtree":
            return self.quadtree_grid.empty_mask.ugrid.to_crs(4326).ugrid.total_bounds

    ## I/O

    def read(self) -> None:
        """Read SfincsModel from disk.

        This methods determines the grid type from the configuration file (sfincs.inp),
        and reads all relevant components that are described in the config accordingly.

        For more information, see specific component read methods.
        """

        # always read config first
        self.config.read()

        for name, comp in self.components.items():
            if name == "config":
                continue  # skip config
            elif self.grid_type == "regular" and name in self._QUADTREE_GRID_NAMES:
                # skip reading quadtree components if grid_type is regular
                continue
            elif self.grid_type == "quadtree" and name in self._REGULAR_GRID_NAMES:
                # skip reading regular grid components if grid_type is quadtree
                continue
            try:
                comp.read()
            except Exception as e:
                logger.warning(f"Could not read component {name}: {e}")
                continue

    def write(self):
        """Write SfincsModel to disk.

        This methods writes all components that actually contain data to the specified
        model root folder. Finally, the configuration file (sfincs.inp) is written.

        For more information, see specific component write methods.
        """

        # TODO make sure that all components are in the config (in their individual write functions?)
        for name, comp in self.components.items():
            if name == "config":
                continue
            elif self.grid_type == "regular" and name in self._QUADTREE_GRID_NAMES:
                continue
            elif self.grid_type == "quadtree" and name in self._REGULAR_GRID_NAMES:
                continue
            comp.write()

        # Write config last, since individual write methods might update config settings
        self.config.write()

        # Write region geometry
        if self.write_gis:
            utils.write_vector(
                self.region,
                name="region",
                root=join(self.root.path, "gis"),
                logger=logger,
            )

    def clear_spatial_components(self):
        """Clear all spatial components."""
        # TODO if we want this, all components should have a clear method
        return
        # Do something like this
        for name, comp in self.components.items():
            if hasattr(comp, "clear"):
                comp.clear()

    ## Plotting
    def plot_forcing(self, fn_out=None, forcings="all", **kwargs):
        """Plot model timeseries forcing.

        For distributed forcing a spatial avarage, minimum or maximum is plotted.

        Parameters
        ----------
        fn_out: str
            Path to output figure file.
            If a basename is given it is saved to <model_root>/figs/<fn_out>
            If None, no file is saved.
        forcings : str
            List of forcings to plot, by default 'all'.
            If 'all', all available forcings are plotted.
            See :py:attr:`~hydromt_sfincs.SfincsModel.forcing.keys()`
            for available forcings.
        **kwargs : dict
            Additional keyword arguments passed to
            :py:func:`hydromt.plotting.plot_forcing`.

        Returns
        -------
        fig, axes
            Model fig and ax objects
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        _FORCING = {
            "water_level": {
                "bzs": {"standard_name": "waterlevel", "unit": "m+ref"},
                "bzi": {"standard_name": "wave height", "unit": "m"},
            },
            "discharge_points": {
                "dis": {"standard_name": "discharge", "unit": "m3.s-1"},
            },
            "precipitation": {
                "precip": {"standard_name": "precipitation", "unit": "mm.hr-1"},
                "precip_2d": {"standard_name": "precipitation", "unit": "mm.hr-1"},
            },
            "pressure": {
                "press_2d": {"standard_name": "barometric pressure", "unit": "Pa"},
            },
            "wind": {
                "wind": {"standard_name": "wind", "unit": "m/s"},
                "wind10_u": {"standard_name": "eastward wind", "unit": "m/s"},
                "wind10_v": {"standard_name": "northward wind", "unit": "m/s"},
            },
            "snapwave_boundary_conditions": {
                "hs": {},
                "tp": {},
                "dir": {},
                "ds": {},
            },
        }

        forcing = {}
        for component, vars_dict in _FORCING.items():
            if self.components.get(component) is not None:
                comp = self.components.get(component)
                if comp is None or not hasattr(comp, "data"):
                    continue
                data = comp.data
                if isinstance(data, xr.Dataset):
                    for name, attrs in vars_dict.items():
                        if name in data:
                            arr = data[name]
                            # check that the DataArray has data
                            if arr.size > 0 and not arr.isnull().all():
                                arr = arr.copy()
                                arr.attrs.update(**attrs)
                                forcing[name] = arr

                elif isinstance(data, xr.DataArray):
                    arr = data
                    if arr.size > 0 and not arr.isnull().all():
                        # assign to the one variable key if known
                        if len(vars_dict) == 1:
                            name, attrs = next(iter(vars_dict.items()))
                            arr = arr.copy()
                            arr.attrs.update(**attrs)
                            forcing[name] = arr
                        elif arr.name in vars_dict:
                            attrs = vars_dict[arr.name]
                            arr = arr.copy()
                            arr.attrs.update(**attrs)
                            forcing[arr.name] = arr

        if len(forcing) > 0:
            fig, axes = plots.plot_forcing(forcing, **kwargs)
            # set xlim to model tstart - tend
            tstart, tstop = self.get_model_time()
            axes[-1].set_xlim(mdates.date2num([tstart, tstop]))

            # save figure
            if fn_out is not None:
                if not os.path.isabs(fn_out):
                    fn_out = join(self.root.path, "figs", fn_out)
                if not os.path.isdir(dirname(fn_out)):
                    os.makedirs(dirname(fn_out))
                plt.savefig(fn_out, dpi=225, bbox_inches="tight")
            return fig, axes
        else:
            raise ValueError("No forcing found in model.")

    def plot_basemap(
        self,
        fn_out: str = None,
        variable: Union[str, xr.DataArray] = "elevation",
        shaded: bool = False,
        plot_bounds: bool = True,
        plot_region: bool = False,
        plot_geoms: bool = True,
        bmap: str = None,
        zoomlevel: int = "auto",
        figsize: Tuple[int] = None,
        geom_names: List[str] = None,
        geom_kwargs: Dict = {},
        legend_kwargs: Dict = {},
        **kwargs,
    ):
        """Create basemap plot.

        Parameters
        ----------
        fn_out: str, optional
            Path to output figure file, by default None.
            If a basename is given it is saved to <model_root>/figs/<fn_out>
            If None, no file is saved.
        variable : str, xr.DataArray, optional
            Map of variable in ds to plot, by default 'dep'
            Alternatively, provide a xr.DataArray
        shaded : bool, optional
            Add shade to variable (only for variable = 'dep' and non-rotated grids),
            by default False
        plot_bounds : bool, optional
            Add waterlevel (mask=2) and open (mask=3) boundary conditions to plot.
        plot_region : bool, optional
            If True, plot region outline.
        plot_geoms : bool, optional
            If True, plot available geoms.
        bmap : str, optional
            background map souce name, by default None.
            Default image tiles "sat", and "osm" are fetched from cartopy image tiles.
            If contextily is installed, xyzproviders tiles can be used as well.
        zoomlevel : int, optional
            zoomlevel, by default 'auto'
        figsize : Tuple[int], optional
            figure size, by default None
        geom_names : List[str], optional
            list of model geometries to plot, by default all model geometries.
        geom_kwargs : Dict of Dict, optional
            Model geometry styling per geometry, passed to geopandas.GeoDataFrame.plot method.
            For instance: {'src': {'markersize': 30}}.
        legend_kwargs : Dict, optional
            Legend kwargs, passed to ax.legend method.

        Returns
        -------
        fig, axes
            Model fig and ax objects
        """
        import matplotlib.pyplot as plt

        _GEOMS = {
            "observation_points": "obs",
            "cross_sections": "crs",
            "weirs": "weir",
            "thin_dams": "thd",
            "drainage_structures": "drn",
            "rivers": "rivers",
            "discharge_points": "src",
            "water_level": "bnd",
        }  # parsed to dict of geopandas.GeoDataFrame

        # combine geoms and forcing locations
        sg = {}
        for component, name in _GEOMS.items():
            # check if component exists and has data
            if self.components.get(component) is not None:
                comp = self.components[component]
                # vector geometries have data as gpd.GeoDataFrame
                if isinstance(comp.data, gpd.GeoDataFrame):
                    gdf = comp.data
                    if not gdf.empty:
                        sg.update({name: gdf})
                # forcing components have data as GeoDataArray(xr.Dataset)
                elif isinstance(comp.data, xr.Dataset):
                    # the property gdf returns a gpd.GeoDataFrame
                    gdf = comp.gdf
                    if not gdf.empty:
                        sg.update({name: gdf})

        if plot_region:  # and "region" not in self.geoms:
            sg.update({"region": self.region})

        # make sure grid are set
        if isinstance(variable, xr.DataArray):
            ds = variable.to_dataset()
            variable = variable.name
        elif isinstance(variable, xu.UgridDataArray):
            ds = variable.to_dataset()
            variable = variable.name
        elif variable.startswith("subgrid.") and self.subgrid.data is not None:
            ds = self.subgrid.data.copy()
            variable = variable.replace("subgrid.", "")
        else:
            if self.grid_type == "regular":
                ds = self.grid.data.copy()
                if "mask" not in ds:
                    ds["mask"] = self.grid.mask
                if variable == "elevation" and "elevation" not in ds:
                    variable = "dep"
            elif self.grid_type == "quadtree":
                ds = self.quadtree_grid.data.copy()
                if "mask" not in ds:
                    ds["mask"] = self.quadtree_grid.mask
                if variable == "elevation" and "elevation" not in ds:
                    variable = "z"

        fig, ax = plots.plot_basemap(
            ds,
            sg,
            variable=variable,
            shaded=shaded,
            plot_bounds=plot_bounds,
            plot_region=plot_region,
            plot_geoms=plot_geoms,
            bmap=bmap,
            zoomlevel=zoomlevel,
            figsize=figsize,
            geom_names=geom_names,
            geom_kwargs=geom_kwargs,
            legend_kwargs=legend_kwargs,
            logger=logger,
            **kwargs,
        )

        if fn_out is not None:
            if not os.path.isabs(fn_out):
                fn_out = join(self.root.path, "figs", fn_out)
            if not os.path.isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))
            plt.savefig(fn_out, dpi=225, bbox_inches="tight")

        return fig, ax

    def get_model_time(self):
        """Return (tstart, tstop) tuple with parsed model start and end time"""
        tstart = utils.parse_datetime(self.config.get("tstart"))
        tstop = utils.parse_datetime(self.config.get("tstop"))
        return tstart, tstop

    # ---------------
    # Helper Methods
    # ---------------

    def _parse_datasets_elevation(self, elevation_list, res):
        """Parse filenames or paths of Datasets in list of dictionaries elevation_list
        into xr.DataArray and gdf.GeoDataFrames:

        * "elevation" is parsed into da (xr.DataArray)
        * "offset" is parsed into da_offset (xr.DataArray)
        * "mask" is parsed into gdf (gpd.GeoDataFrame)

        Parameters
        ----------
        elevation_list : List[dict]
            List of dictionaries with topography and bathymetry data, each containing a dataset name or
            Path (dep) and optional merge arguments.
        res : float
            Resolution of the model grid in meters. Used to obtain the correct zoom
            level of the depth datasets.
        """
        parse_keys = ["elevation", "offset", "mask", "da"]
        copy_keys = ["zmin", "zmax", "reproj_method", "merge_method", "offset"]

        datasets_out = []
        for dataset in elevation_list:
            dd = {}
            # read in depth datasets; replace dep (source name; filename or xr.DataArray)
            if "elevation" in dataset or "da" in dataset:
                try:
                    da_elv = self.data_catalog.get_rasterdataset(
                        dataset.get("elevation", dataset.get("da")),
                        bbox=self.bbox,
                        buffer=10,
                        variables=["elevtn"],  # NOTE this is still hydromt convention
                        zoom=(res, "meter"),
                    )
                    # rename elevtn to elevation if present
                    da_elv.name = "elevation"
                # TODO remove ValueError after fix in hydromt core
                except (IndexError, ValueError):
                    data_name = dataset.get("elevation")
                    logger.warning(f"No data in domain for {data_name}, skipped.")
                    continue
                dd.update({"da": da_elv})
            else:
                raise ValueError(
                    "No 'elevation' (topobathy) dataset provided in elevation_list."
                )

            # read offset filenames
            # NOTE offsets can be xr.DataArrays and floats
            if "offset" in dataset and not isinstance(dataset["offset"], (float, int)):
                da_offset = self.data_catalog.get_rasterdataset(
                    dataset.get("offset"),
                    bbox=self.bbox,
                    buffer=10,
                )
                dd.update({"offset": da_offset})

            # read geodataframes describing valid areas
            if "mask" in dataset:
                gdf_valid = self.data_catalog.get_geodataframe(
                    dataset.get("mask"),
                    bbox=self.bbox,
                )
                dd.update({"gdf_valid": gdf_valid})

            # copy remaining keys
            for key, value in dataset.items():
                if key in copy_keys and key not in dd:
                    dd.update({key: value})
                elif key not in copy_keys + parse_keys:
                    logger.warning(f"Unknown key {key} in elevation_list. Ignoring.")
            datasets_out.append(dd)

        return datasets_out

    def _parse_roughness_list(self, roughness_list):
        """Parse filenames or paths of Datasets in list of dictionaries roughness_list
        into xr.DataArrays and gdf.GeoDataFrames:

        * "manning" is parsed into da (xr.DataArray)
        * "lulc" is parsed into da (xr.DataArray) using reclass table in "reclass_table"
        * "mask" is parsed into gdf_valid (gpd.GeoDataFrame)

        Parameters
        ----------
        roughness_list : List[dict], optional
            List of dictionaries with Manning's n datasets. Each dictionary should at
            least contain one of the following:
            * (1) manning: filename (or Path) of gridded data with manning values
            * (2) lulc (and reclass_table): a combination of a filename of gridded
                  landuse/landcover and a reclassify table.
            In additon, optional merge arguments can be provided e.g.: merge_method, mask
        """
        parse_keys = ["manning", "lulc", "reclass_table", "mask", "da"]
        copy_keys = ["reproj_method", "merge_method"]

        datasets_out = []
        for dataset in roughness_list:
            dd = {}

            if "manning" in dataset or "da" in dataset:
                da_man = self.data_catalog.get_rasterdataset(
                    dataset.get("manning", dataset.get("da")),
                    bbox=self.bbox,
                    buffer=10,
                )
                dd.update({"da": da_man})
            elif "lulc" in dataset:
                # landuse/landcover should always be combined with mapping
                lulc = dataset.get("lulc")
                reclass_table = dataset.get("reclass_table", None)
                if reclass_table is None and isinstance(lulc, str):
                    reclass_table = join(DATADIR, "lulc", f"{lulc}_mapping.csv")
                if reclass_table is None:
                    raise IOError(
                        "Manning roughness 'reclass_table' csv file must be provided"
                    )
                da_lulc = self.data_catalog.get_rasterdataset(
                    lulc,
                    bbox=self.bbox,
                    buffer=10,
                    variables=["lulc"],
                )
                df_map = self.data_catalog.get_dataframe(
                    reclass_table,
                    source_kwargs={
                        "driver": {"name": "pandas", "options": {"index_col": 0}}
                    },
                )
                # reclassify
                da_man = da_lulc.raster.reclassify(df_map[["N"]])["N"]
                dd.update({"da": da_man})
            else:
                raise ValueError("No 'manning' dataset provided in roughness_list.")

            # read geodataframes describing valid areas
            if "mask" in dataset:
                gdf_valid = self.data_catalog.get_geodataframe(
                    dataset.get("mask"),
                    bbox=self.bbox,
                )
                dd.update({"gdf_valid": gdf_valid})

            # copy remaining keys
            for key, value in dataset.items():
                if key in copy_keys and key not in dd:
                    dd.update({key: value})
                elif key not in copy_keys + parse_keys:
                    logger.warning(f"Unknown key {key} in roughness sets. Ignoring.")
            datasets_out.append(dd)

        return datasets_out

    def _parse_river_list(self, river_list):
        """Parse filenames or paths of Datasets in list of dictionaries
        river_list into xr.DataArrays and gdf.GeoDataFrames:

        see SfincsModel.setup_subgrid for details
        """
        # option 1: rectangular river cross-sections based on river centerline
        # depth/bedlevel, manning attributes are specified on the river centerline
        # TODO: make this work with LineStringZ geometries for bedlevel
        # the width is either specified on the river centerline or river mask
        # option 2: (TODO): irregular river cross-sections
        # cross-sections are specified as a series of points (river_crosssections)
        parse_keys = [
            "centerlines",
            "mask",
            "gdf_riv",
            "gdf_riv_mask",
            "gdf_zb",
            "point_zb",
        ]
        copy_keys = []
        attrs = ["rivwth", "rivdph", "rivbed", "manning"]

        datasets_out = []
        for dataset in river_list:
            dd = {}

            # parse rivers
            if "centerlines" in dataset:
                rivers = dataset.get("centerlines")
                if isinstance(rivers, str) and rivers in self.geoms:
                    gdf_riv = self.geoms[rivers].copy()
                else:
                    gdf_riv = self.data_catalog.get_geodataframe(
                        rivers,
                        bbox=self.bbox,
                        buffer=1e3,  # 1km
                    ).to_crs(self.crs)
                # update missing attributes based on global values
                for key in attrs:
                    if key in dataset:
                        value = dataset.pop(key)
                        if key not in gdf_riv.columns:  # update all
                            gdf_riv[key] = value
                        elif np.any(np.isnan(gdf_riv[key])):  # fill na
                            gdf_riv[key] = gdf_riv[key].fillna(value)
                dd.update({"gdf_riv": gdf_riv})

            # parse bed_level on points
            if "point_zb" in dataset:
                gdf_zb = self.data_catalog.get_geodataframe(
                    dataset.get("point_zb"),
                    bbox=self.bbox,
                )
                dd.update({"gdf_zb": gdf_zb})

            if "gdf_riv" in dd:
                if (
                    not gdf_riv.columns.isin(["rivbed", "rivdph"]).any()
                    and "gdf_zb" not in dd
                ):
                    raise ValueError("No 'rivbed' or 'rivdph' attribute found.")
            else:
                raise ValueError("No 'centerlines' dataset provided.")

            # parse mask
            if "mask" in dataset:
                gdf_riv_mask = self.data_catalog.get_geodataframe(
                    dataset.get("mask"),
                    bbox=self.bbox,
                )
                dd.update({"gdf_riv_mask": gdf_riv_mask})
            elif "rivwth" not in gdf_riv:
                raise ValueError(
                    "Either mask must be provided or centerlines "
                    "should contain a 'rivwth' attribute."
                )
            # copy remaining keys
            for key, value in dataset.items():
                if key in copy_keys and key not in dd:
                    dd.update({key: value})
                elif key not in copy_keys + parse_keys:
                    logger.warning(f"Unknown key {key} in river_list. Ignoring.")
            datasets_out.append(dd)

        return datasets_out

    # ---------------------------------------
    # Component properties (for IDE & Sphinx)
    # ---------------------------------------
    # To generate the properties below automatically, run the code commented out
    # here in a separate script or notebook.
    #
    # from hydromt_sfincs import SfincsModel
    #
    # for name, cls in SfincsModel._ALL_COMPONENTS.items():
    #     # Simplify module path for Sphinx
    #     mod_parts = cls.__module__.split('.')
    #     # remove hydromt_sfincs.components and script itself from name
    #     mod_parts_stripped = mod_parts[2]
    #     mod_path = '.'.join(mod_parts)

    #     print(f"""@property
    # def {name}(self) -> {cls.__name__}:
    #     \"\"\"Instance of :py:class:`~{'.'.join(mod_parts)}.{cls.__name__}`.\"\"\"
    #     return self.components['{name}']
    # """)

    @property
    def config(self) -> SfincsConfig:
        """Instance of :py:class:`~hydromt_sfincs.components.config.config.SfincsConfig`."""
        return self.components["config"]

    @property
    def grid(self) -> SfincsGrid:
        """Instance of :py:class:`~hydromt_sfincs.components.grid.regulargrid.SfincsGrid`."""
        return self.components["grid"]

    @property
    def elevation(self) -> SfincsElevation:
        """Instance of :py:class:`~hydromt_sfincs.components.grid.elevation.SfincsElevation`."""
        return self.components["elevation"]

    @property
    def mask(self) -> SfincsMask:
        """Instance of :py:class:`~hydromt_sfincs.components.grid.mask.SfincsMask`."""
        return self.components["mask"]

    @property
    def infiltration(self) -> SfincsInfiltration:
        """Instance of :py:class:`~hydromt_sfincs.components.grid.infiltration.SfincsInfiltration`."""
        return self.components["infiltration"]

    @property
    def roughness(self) -> SfincsRoughness:
        """Instance of :py:class:`~hydromt_sfincs.components.grid.roughness.SfincsRoughness`."""
        return self.components["roughness"]

    @property
    def storage_volume(self) -> SfincsStorageVolume:
        """Instance of :py:class:`~hydromt_sfincs.components.grid.storage_volume.SfincsStorageVolume`."""
        return self.components["storage_volume"]

    @property
    def subgrid(self) -> SfincsSubgridTable:
        """Instance of :py:class:`~hydromt_sfincs.components.grid.subgrid.SfincsSubgridTable`."""
        return self.components["subgrid"]

    @property
    def initial_conditions(self) -> SfincsInitialConditions:
        """Instance of :py:class:`~hydromt_sfincs.components.grid.initial_conditions.SfincsInitialConditions`."""
        return self.components["initial_conditions"]

    @property
    def quadtree_grid(self) -> SfincsQuadtreeGrid:
        """Instance of :py:class:`~hydromt_sfincs.components.quadtree.quadtree.SfincsQuadtreeGrid`."""
        return self.components["quadtree_grid"]

    @property
    def quadtree_elevation(self) -> SfincsQuadtreeElevation:
        """Instance of :py:class:`~hydromt_sfincs.components.quadtree.quadtree_elevation.SfincsQuadtreeElevation`."""
        return self.components["quadtree_elevation"]

    @property
    def quadtree_mask(self) -> SfincsQuadtreeMask:
        """Instance of :py:class:`~hydromt_sfincs.components.quadtree.quadtree_mask.SfincsQuadtreeMask`."""
        return self.components["quadtree_mask"]

    @property
    def quadtree_infiltration(self) -> SfincsQuadtreeInfiltration:
        """Instance of :py:class:`~hydromt_sfincs.components.quadtree.quadtree_infiltration.SfincsQuadtreeInfiltration`."""
        return self.components["quadtree_infiltration"]

    @property
    def quadtree_storage_volume(self) -> SfincsQuadtreeStorageVolume:
        """Instance of :py:class:`~hydromt_sfincs.components.quadtree.quadtree_storage_volume.SfincsQuadtreeStorageVolume`."""
        return self.components["quadtree_storage_volume"]

    @property
    def quadtree_initial_conditions(self) -> SfincsQuadtreeInitialConditions:
        """Instance of :py:class:`~hydromt_sfincs.components.quadtree.quadtree_initial_conditions.SfincsQuadtreeInitialConditions`."""
        return self.components["quadtree_initial_conditions"]

    @property
    def quadtree_subgrid(self) -> SfincsQuadtreeSubgridTable:
        """Instance of :py:class:`~hydromt_sfincs.components.quadtree.quadtree_subgrid.SfincsQuadtreeSubgridTable`."""
        return self.components["quadtree_subgrid"]

    @property
    def quadtree_snapwave_mask(self) -> SnapWaveQuadtreeMask:
        """Instance of :py:class:`~hydromt_sfincs.components.quadtree.snapwave_quadtree_mask.SnapWaveQuadtreeMask`."""
        return self.components["quadtree_snapwave_mask"]

    @property
    def observation_points(self) -> SfincsObservationPoints:
        """Instance of :py:class:`~hydromt_sfincs.components.geometries.observation_points.SfincsObservationPoints`."""
        return self.components["observation_points"]

    @property
    def cross_sections(self) -> SfincsCrossSections:
        """Instance of :py:class:`~hydromt_sfincs.components.geometries.cross_sections.SfincsCrossSections`."""
        return self.components["cross_sections"]

    @property
    def thin_dams(self) -> SfincsThinDams:
        """Instance of :py:class:`~hydromt_sfincs.components.geometries.thin_dams.SfincsThinDams`."""
        return self.components["thin_dams"]

    @property
    def weirs(self) -> SfincsWeirs:
        """Instance of :py:class:`~hydromt_sfincs.components.geometries.weirs.SfincsWeirs`."""
        return self.components["weirs"]

    @property
    def wave_makers(self) -> SfincsWaveMakers:
        """Instance of :py:class:`~hydromt_sfincs.components.geometries.wave_makers.SfincsWaveMakers`."""
        return self.components["wave_makers"]

    @property
    def drainage_structures(self) -> SfincsDrainageStructures:
        """Instance of :py:class:`~hydromt_sfincs.components.geometries.drainage_structures.SfincsDrainageStructures`."""
        return self.components["drainage_structures"]

    @property
    def rivers(self) -> SfincsRivers:
        """Instance of :py:class:`~hydromt_sfincs.components.forcing.rivers.SfincsRivers`."""
        return self.components["rivers"]

    @property
    def water_level(self) -> SfincsWaterLevel:
        """Instance of :py:class:`~hydromt_sfincs.components.forcing.water_level.SfincsWaterLevel`."""
        return self.components["water_level"]

    @property
    def discharge_points(self) -> SfincsDischargePoints:
        """Instance of :py:class:`~hydromt_sfincs.components.forcing.discharge_points.SfincsDischargePoints`."""
        return self.components["discharge_points"]

    @property
    def snapwave_boundary_conditions(self) -> SnapWaveBoundaryConditions:
        """Instance of :py:class:`~hydromt_sfincs.components.forcing.snapwave_boundary_conditions.SnapWaveBoundaryConditions`."""
        return self.components["snapwave_boundary_conditions"]

    @property
    def precipitation(self) -> SfincsPrecipitation:
        """Instance of :py:class:`~hydromt_sfincs.components.forcing.meteo.SfincsPrecipitation`."""
        return self.components["precipitation"]

    @property
    def pressure(self) -> SfincsPressure:
        """Instance of :py:class:`~hydromt_sfincs.components.forcing.meteo.SfincsPressure`."""
        return self.components["pressure"]

    @property
    def wind(self) -> SfincsWind:
        """Instance of :py:class:`~hydromt_sfincs.components.forcing.meteo.SfincsWind`."""
        return self.components["wind"]

    @property
    def output(self) -> SfincsOutput:
        """Instance of :py:class:`~hydromt_sfincs.components.output.SfincsOutput`."""
        return self.components["output"]
