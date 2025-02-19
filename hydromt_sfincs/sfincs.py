"""
SfincsModel class
"""

# %% Import packages
from __future__ import annotations

import glob
import logging
import os
from os.path import abspath, basename, dirname, isabs, isfile, join
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import hydromt
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
from hydromt.model import Model
from hydromt.gis.vector import GeoDataArray, GeoDataset
from hydromt.model.processes.meteo import da_to_timedelta
from pyproj import CRS
from shapely.geometry import LineString, box
from xugrid.core.wrap import UgridDataArray

from hydromt_sfincs import DATADIR, plots, utils, workflows

# %% Import model components
from hydromt.model import Model

# input types:
from hydromt_sfincs.config import SfincsInput

# grid types:
from hydromt_sfincs.quadtree import QuadtreeGrid
from hydromt_sfincs.regulargrid import RegularGrid
from hydromt_sfincs.subgrid import SubgridTableRegular

# map types:
# from hydromt_sfincs.mask import SfincsMask
# from hydromt_sfincs.bathymetry import SfincsBathymetry
from hydromt_sfincs.infiltration import SfincsInfiltration
from hydromt_sfincs.manning_roughness import SfincsManningRoughness
from hydromt_sfincs.initial_conditions import SfincsInitialConditions
from hydromt_sfincs.storage_volume import SfincsStorageVolume

# geoms types:
from hydromt_sfincs.observation_points import SfincsObservationPoints
from hydromt_sfincs.cross_sections import SfincsCrossSections
from hydromt_sfincs.weirs import SfincsWeirs
from hydromt_sfincs.thin_dams import SfincsThinDams
from hydromt_sfincs.wave_makers import SfincsWaveMakers
from hydromt_sfincs.drainage_structures import SfincsDrainageStructures
from hydromt_sfincs.rivers import SfincsRivers

# forcing types:
from hydromt_sfincs.discharge_points import SfincsDischargePoints

# from hydromt_sfincs.boundary_conditions import SfincsBoundaryConditions #/
from hydromt_sfincs.waterlevel_conditions import SfincsWaterlevelConditions
from hydromt_sfincs.snapwave_conditions import SfincsSnapWaveConditions

# from hydromt_sfincs.meteo import SfincsMeteo
from hydromt_sfincs.meteo import SfincsPrecipitation, SfincsPressure, SfincsWind

# output / visualization types:
from hydromt_sfincs.output import SfincsOutput

# from hydromt_sfincs.plots import SfincsPlots

__all__ = ["SfincsModel"]

__hydromt_eps__ = ["SfincsModel"]  # core entrypoints


logger = logging.getLogger(__name__)


# %% SfincsModel class - in V1 style:
class SfincsModel(Model):
    """SFINCS model class."""

    _FOLDERS = []

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        write_gis: bool = True,
        data_libs: Union[List[str], str] = None,
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

        """

        # model folders
        self._write_gis = write_gis
        if write_gis and "gis" not in self._FOLDERS:
            self._FOLDERS.append("gis")

        super().__init__(
            root=root,
            mode=mode,
            data_libs=data_libs,
        )
        # Initialize model components:

        self.grid_type = "regular"

        self.add_component("config", SfincsInput(self))
        # grid types:
        self.add_component("grid", RegularGrid(self))
        self.add_component("quadtree", QuadtreeGrid(self))
        # self.add_component("subgrid", SubgridTableRegular(self))
        # self.add_component("subgrid", SubgridTableRegular(self))

        # map types:
        # self.add_component("mask", SfincsMask(self))
        # self.add_component("bathymetry", SfincsBathymetry(self))
        # self.add_component("infiltration", SfincsInfiltration(self))
        # self.add_component("manning_roughness", SfincsManningRoughness(self))
        # self.add_component("initial_conditions", SfincsInitialConditions(self))
        # self.add_component("storage_volume", SfincsStorageVolume(self))

        # geoms types:
        # self.add_component("observation_points", SfincsObservationPoints(self))
        # self.add_component("cross_sections", SfincsCrossSections(self))
        # self.add_component("weirs", SfincsWeirs(self))
        # self.add_component("thin_dams", SfincsThinDams(self))
        # self.add_component("wave_makers", SfincsWaveMakers(self))
        # self.add_component("drainage_structures", SfincsDrainageStructures(self))
        # self.add_component("rivers", SfincsRivers(self))

        # forcing types:
        # self.add_component("discharge_points", SfincsDischargePoints(self))
        # self.add_component("waterlevel_conditions", SfincsWaterlevelConditions(self))
        # self.add_component("snapwave_conditions", SfincsSnapWaveConditions(self))
        # self.add_component("meteo", SfincsMeteo(self))
        # self.add_component("precipitation", SfincsPrecipitation(self))
        # self.add_component("pressure", SfincsPressure(self))
        # self.add_component("wind", SfincsWind(self))
        # self.add_component("forcing", SfincsForcing(self))

        # output / visualization types:
        # self.add_component("output", SfincsOutput(self))
        # self.add_component("plots", SfincsPlots(self))

    # def __del__(self):
    #     """Close the model and remove the logger file handler."""
    #     for handler in self.logger.handlers:
    #         if (
    #             isinstance(handler, logging.FileHandler)
    #             and "hydromt.log" in handler.baseFilename
    #         ):
    #             handler.close()
    #             self.logger.removeHandler(handler)

    @property
    def config(self) -> SfincsInput:
        """Returns the config object."""
        return self.components["config"]

    @property
    def grid(self) -> RegularGrid:
        """Returns the grid object."""
        return self.components["grid"]

    @property
    def quadtree(self) -> QuadtreeGrid:
        """Returns the quadtree object."""
        return self.components["quadtree"]

    @property
    def mask(self) -> xr.DataArray | None:
        """Returns model mask"""
        if self.grid_type == "regular":
            if "msk" in self.grid.data:
                return self.grid.data["msk"]
            elif self.grid is not None:
                return self.grid.empty_mask
        elif self.grid_type == "quadtree":
            if "msk" in self.quadtree.data:
                return self.quadtree.data["msk"]
            elif self.quadtree is not None:
                return self.quadtree.empty_mask

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the active model cells."""
        # NOTE overwrites property in GridModel
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif self.grid_type == "regular":
            if "msk" in self.grid and np.any(self.grid["msk"] > 0):
                da = xr.where(self.mask > 0, 1, 0).astype(np.int16)
                da.raster.set_nodata(0)
                region = da.raster.vectorize().dissolve()
            elif self.grid is not None:
                region = self.grid.empty_mask.raster.box
        elif self.grid_type == "quadtree":
            region = self.quadtree.exterior
        return region

    @property
    def bounds(self) -> List[float]:
        """Returns the bounding box of the model grid."""
        if self.grid_type == "regular":
            return self.mask.raster.bounds
        elif self.grid_type == "quadtree":
            return self.mask.ugrid.total_bounds

    @property
    def bbox(self) -> tuple:
        """Returns the bounding box in WGS 84 of the model grid."""
        if self.grid_type == "regular":
            return self.mask.raster.transform_bounds(4326)
        elif self.grid_type == "quadtree":
            return self.mask.ugrid.to_crs(4326).ugrid.total_bounds

    @property
    def crs(self) -> CRS | None:
        """Returns the model crs"""
        if self.grid_type == "regular":
            return self.grid.crs
        elif self.grid_type == "quadtree":
            return self.quadtree.data.grid.crs

    ## I/O
    def read(self, filename: str = None) -> None:
        """Read model components from config file and initialize model grid.

        Parameters
        ----------
        filename : str, optional
            Path to config file, by default None
        """
        # always read config first
        if filename is None:
            filename = join(self.root.path, "sfincs.inp")
        self.config.read(filename=filename)

        # Determine grid type based on configuration
        self.grid_type = "quadtree" if self.config.get("qtrfile") else "regular"

        # Remove the grid component if it doesn't match the current grid type
        if self.grid_type == "regular":
            self.components.pop("quadtree", None)
        elif self.grid_type == "quadtree":
            self.components.pop("grid", None)

        # loop over all components (except config) and read
        self.grid.read()

    def write(self):
        self.config.write()

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

        if self.forcing:
            forcing = {}
            if forcings == "all":
                forcings = list(self.forcing.keys())
            elif isinstance(forcings, str):
                forcings = [forcings]
            for name in forcings:
                if name not in self.forcing:
                    self.logger.warning(f'No forcing named "{name}" found in model.')
                    continue
                if isinstance(self.forcing[name], xr.Dataset):
                    self.logger.warning(
                        f'Skipping forcing "{name}" as it is a dataset.'
                    )
                    continue
                # plot only dataarrays
                forcing[name] = self.forcing[name].copy()
                # update missing attributes for plot labels
                forcing[name].attrs.update(**self._ATTRS.get(name, {}))
            if len(forcing) > 0:
                fig, axes = plots.plot_forcing(forcing, **kwargs)
                # set xlim to model tstart - tend
                tstart, tstop = self.get_model_time()
                axes[-1].set_xlim(mdates.date2num([tstart, tstop]))

                # save figure
                if fn_out is not None:
                    if not os.path.isabs(fn_out):
                        fn_out = join(self.root, "figs", fn_out)
                    if not os.path.isdir(dirname(fn_out)):
                        os.makedirs(dirname(fn_out))
                    plt.savefig(fn_out, dpi=225, bbox_inches="tight")
                return fig, axes
        else:
            raise ValueError("No forcing found in model.")

    def plot_basemap(
        self,
        fn_out: str = None,
        variable: Union[str, xr.DataArray] = "dep",
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
            Add waterlevel (msk=2) and open (msk=3) boundary conditions to plot.
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

        # combine geoms and forcing locations
        sg = self.geoms.copy()
        for fname, gname in self._FORCING_1D.values():
            if fname[0] in self.forcing and gname is not None:
                try:
                    sg.update({gname: self.forcing[fname[0]].vector.to_gdf()})
                except ValueError:
                    self.logger.debug(f'unable to plot forcing location: "{fname}"')
        if plot_region and "region" not in self.geoms:
            sg.update({"region": self.region})

        # make sure grid are set
        if isinstance(variable, xr.DataArray):
            ds = variable.to_dataset()
            variable = variable.name
        elif isinstance(variable, xu.UgridDataArray):
            ds = variable.to_dataset()
            variable = variable.name
        elif variable.startswith("subgrid.") and self.subgrid is not None:
            ds = self.subgrid.copy()
            variable = variable.replace("subgrid.", "")
        else:
            if self.grid_type == "regular":
                ds = self.grid.copy()
            elif self.grid_type == "quadtree":
                ds = self.quadtree.data.copy()
            if "msk" not in ds:
                ds["msk"] = self.mask

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
            logger=self.logger,
            **kwargs,
        )

        if fn_out is not None:
            if not os.path.isabs(fn_out):
                fn_out = join(self.root, "figs", fn_out)
            if not os.path.isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))
            plt.savefig(fn_out, dpi=225, bbox_inches="tight")

        return fig, ax

    def get_model_time(self):
        """Return (tstart, tstop) tuple with parsed model start and end time"""
        tstart = utils.parse_datetime(self.config["tstart"])
        tstop = utils.parse_datetime(self.config["tstop"])
        return tstart, tstop

    ## helper method
    def _parse_datasets_dep(self, datasets_dep, res):
        """Parse filenames or paths of Datasets in list of dictionaries datasets_dep
        into xr.DataArray and gdf.GeoDataFrames:

        * "elevtn" is parsed into da (xr.DataArray)
        * "offset" is parsed into da_offset (xr.DataArray)
        * "mask" is parsed into gdf (gpd.GeoDataFrame)

        Parameters
        ----------
        datasets_dep : List[dict]
            List of dictionaries with topobathy data, each containing a dataset name or
            Path (dep) and optional merge arguments.
        res : float
            Resolution of the model grid in meters. Used to obtain the correct zoom
            level of the depth datasets.
        """
        parse_keys = ["elevtn", "offset", "mask", "da"]
        copy_keys = ["zmin", "zmax", "reproj_method", "merge_method", "offset"]

        datasets_out = []
        for dataset in datasets_dep:
            dd = {}
            # read in depth datasets; replace dep (source name; filename or xr.DataArray)
            if "elevtn" in dataset or "da" in dataset:
                try:
                    da_elv = self.data_catalog.get_rasterdataset(
                        dataset.get("elevtn", dataset.get("da")),
                        bbox=self.bbox,
                        buffer=10,
                        variables=["elevtn"],
                        zoom_level=(res, "meter"),
                    )
                # TODO remove ValueError after fix in hydromt core
                except (IndexError, ValueError):
                    data_name = dataset.get("elevtn")
                    self.logger.warning(f"No data in domain for {data_name}, skipped.")
                    continue
                dd.update({"da": da_elv})
            else:
                raise ValueError(
                    "No 'elevtn' (topobathy) dataset provided in datasets_dep."
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
                    self.logger.warning(f"Unknown key {key} in datasets_dep. Ignoring.")
            datasets_out.append(dd)

        return datasets_out

    def _parse_datasets_rgh(self, datasets_rgh):
        """Parse filenames or paths of Datasets in list of dictionaries datasets_rgh
        into xr.DataArrays and gdf.GeoDataFrames:

        * "manning" is parsed into da (xr.DataArray)
        * "lulc" is parsed into da (xr.DataArray) using reclass table in "reclass_table"
        * "mask" is parsed into gdf_valid (gpd.GeoDataFrame)

        Parameters
        ----------
        datasets_rgh : List[dict], optional
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
        for dataset in datasets_rgh:
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
                df_map = self.data_catalog.get_dataframe(reclass_table, index_col=0)
                # reclassify
                da_man = da_lulc.raster.reclassify(df_map[["N"]])["N"]
                dd.update({"da": da_man})
            else:
                raise ValueError("No 'manning' dataset provided in datasets_rgh.")

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
                    self.logger.warning(f"Unknown key {key} in datasets_rgh. Ignoring.")
            datasets_out.append(dd)

        return datasets_out

    def _parse_datasets_riv(self, datasets_riv):
        """Parse filenames or paths of Datasets in list of dictionaries
        datasets_riv into xr.DataArrays and gdf.GeoDataFrames:

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
        for dataset in datasets_riv:
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
                    self.logger.warning(f"Unknown key {key} in datasets_riv. Ignoring.")
            datasets_out.append(dd)

        return datasets_out
