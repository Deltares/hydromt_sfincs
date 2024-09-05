"""
SfincsModel class
"""
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
from hydromt.models.model_grid import GridModel
from hydromt.vector import GeoDataArray, GeoDataset
from hydromt.workflows.forcing import da_to_timedelta
from pyproj import CRS
from shapely.geometry import LineString, box

from . import DATADIR, plots, utils, workflows
from .regulargrid import RegularGrid
from .subgrid import SubgridTableRegular
from .sfincs_input import SfincsInput

__all__ = ["SfincsModel"]

logger = logging.getLogger(__name__)


class SfincsModel(GridModel):
    # GLOBAL Static class variables that can be used by all methods within
    # SfincsModel class. Typically list of variables (e.g. _MAPS) or
    # dict with varname - filename pairs (e.g. thin_dams : thd)
    _NAME = "sfincs"
    _GEOMS = {
        "observation_points": "obs",
        "observation_lines": "crs",
        "weirs": "weir",
        "thin_dams": "thd",
        "drainage_structures": "drn",
    }  # parsed to dict of geopandas.GeoDataFrame
    _FORCING_1D = {
        # timeseries (can be multiple), locations tuple
        "waterlevel": (["bzs"], "bnd"),
        "waves": (["bzi"], "bnd"),
        "discharge": (["dis"], "src"),
        "precip": (["precip"], None),
        "wind": (["wnd"], None),
        "wavespectra": (["bhs", "btp", "bwd", "bds"], "bwv"),
        "wavemaker": (["whi", "wti", "wst"], "wvp"),  # TODO check names and test
    }
    _FORCING_NET = {
        # 2D forcing sfincs name, rename tuple
        "waterlevel": ("netbndbzsbzi", {"zs": "bzs", "zi": "bzi"}),
        "discharge": ("netsrcdis", {"discharge": "dis"}),
        "precip_2d": ("netampr", {"Precipitation": "precip_2d"}),
        "press_2d": ("netamp", {"barometric_pressure": "press_2d"}),
        "wind_2d": (
            "netamuamv",
            {"eastward_wind": "wind10_u", "northward_wind": "wind10_v"},
        ),
    }
    _FORCING_SPW = {"spiderweb": "spw"}  # TODO add read and write functions
    _MAPS = ["msk", "dep", "scs", "manning", "qinf", "smax", "seff", "ks", "vol"]
    _STATES = ["rst", "ini"]
    _FOLDERS = []
    _CLI_ARGS = {"region": "setup_grid_from_region", "res": "setup_grid_from_region"}
    _CONF = "sfincs.inp"
    _DATADIR = DATADIR
    _ATTRS = {
        "dep": {"standard_name": "elevation", "unit": "m+ref"},
        "msk": {"standard_name": "mask", "unit": "-"},
        "scs": {
            "standard_name": "potential maximum soil moisture retention",
            "unit": "in",
        },
        "qinf": {"standard_name": "infiltration rate", "unit": "mm.hr-1"},
        "manning": {"standard_name": "manning roughness", "unit": "s.m-1/3"},
        "vol": {"standard_name": "storage volume", "unit": "m3"},
        "bzs": {"standard_name": "waterlevel", "unit": "m+ref"},
        "bzi": {"standard_name": "wave height", "unit": "m"},
        "dis": {"standard_name": "discharge", "unit": "m3.s-1"},
        "precip": {"standard_name": "precipitation", "unit": "mm.hr-1"},
        "precip_2d": {"standard_name": "precipitation", "unit": "mm.hr-1"},
        "press_2d": {"standard_name": "barometric pressure", "unit": "Pa"},
        "wind10_u": {"standard_name": "eastward wind", "unit": "m/s"},
        "wind10_v": {"standard_name": "northward wind", "unit": "m/s"},
        "wnd": {"standard_name": "wind", "unit": "m/s"},
    }

    def __init__(
        self,
        root: str = None,
        mode: str = "w",
        config_fn: str = "sfincs.inp",
        write_gis: bool = True,
        data_libs: Union[List[str], str] = None,
        logger=logger,
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
        config_fn: str, Path, optional
            Filename of model config file, by default "sfincs.inp"
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
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

        # placeholder grid classes
        self.grid_type = None
        self.reggrid = None
        self.quadtree = None
        self.subgrid = xr.Dataset()

    @property
    def mask(self) -> xr.DataArray | None:
        """Returns model mask"""
        if self.grid_type == "regular":
            if "msk" in self.grid:
                return self.grid["msk"]
            elif self.reggrid is not None:
                return self.reggrid.empty_mask

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Returns the geometry of the active model cells."""
        # NOTE overwrites property in GridModel
        region = gpd.GeoDataFrame()
        if "region" in self.geoms:
            region = self.geoms["region"]
        elif "msk" in self.grid and np.any(self.grid["msk"] > 0):
            da = xr.where(self.mask > 0, 1, 0).astype(np.int16)
            da.raster.set_nodata(0)
            region = da.raster.vectorize().dissolve()
        elif self.reggrid is not None:
            region = self.reggrid.empty_mask.raster.box
        return region

    @property
    def crs(self) -> CRS | None:
        """Returns the model crs"""
        if self.grid_type == "regular":
            return self.reggrid.crs
        elif self.grid_type == "quadtree":
            return self.quadtree.crs

    def set_crs(self, crs: Any) -> None:
        """Sets the model crs"""
        if self.grid_type == "regular":
            self.reggrid.crs = CRS.from_user_input(crs)
            self.grid.raster.set_crs(self.reggrid.crs)
        elif self.grid_type == "quadtree":
            self.quadtree.crs = CRS.from_user_input(crs)

    def setup_grid(
        self,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nmax: int,
        mmax: int,
        rotation: float,
        epsg: int,
    ):
        """Setup a regular or quadtree grid.

        Parameters
        ----------
        x0, y0 : float
            x,y coordinates of the origin of the grid
        dx, dy : float
            grid cell size in x and y direction
        mmax, nmax : int
            number of grid cells in x and y direction
        rotation : float, optional
            rotation of grid [degree angle], by default None
        epsg : int, optional
            epsg-code of the coordinate reference system, by default None
        """
        # TODO gdf_refinement for quadtree

        self.config.update(
            x0=x0,
            y0=y0,
            dx=dx,
            dy=dy,
            nmax=nmax,
            mmax=mmax,
            rotation=rotation,
            epsg=epsg,
        )
        self.update_grid_from_config()

    def setup_grid_from_region(
        self,
        region: dict,
        res: float = 100,
        crs: Union[str, int] = "utm",
        rotated: bool = False,
        hydrography_fn: str = None,
        basin_index_fn: str = None,
        align: bool = False,
        dec_origin: int = 0,
        dec_rotation: int = 3,
    ):
        """Setup a regular or quadtree grid from a region.

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:

            * {'bbox': [xmin, ymin, xmax, ymax]}
            * {'geom': 'path/to/polygon_geometry'}

            Note: For the 'bbox' option the coordinates need to be provided in WG84/EPSG:4326.

            For a complete overview of all region options,
            see :py:func:`hydromt.workflows.basin_mask.parse_region`
        res : float, optional
            grid resolution, by default 100 m
        crs : Union[str, int], optional
            coordinate reference system of the grid
            if "utm" (default) the best UTM zone is selected
            else a pyproj crs string or epsg code (int) can be provided
        grid_type : str, optional
            grid type, "regular" (default) or "quadtree"
        rotated : bool, optional
            if True, a minimum rotated rectangular grid is fitted around the region, by default False
        hydrography_fn : str
            Name of data source for hydrography data.
        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.
        align : bool, optional
            If True (default), align target transform to resolution.
            Note that this has only been implemented for non-rotated grids.
        dec_origin : int, optional
            number of decimals to round the origin coordinates, by default 0
        dec_rotation : int, optional
            number of decimals to round the rotation angle, by default 3

        See Also
        --------
        hydromt.workflows.basin_mask.parse_region
        """
        # setup `region` of interest of the model.
        self.setup_region(
            region=region,
            hydrography_fn=hydrography_fn,
            basin_index_fn=basin_index_fn,
        )
        # get pyproj crs of best UTM zone if crs=utm
        pyproj_crs = hydromt.gis_utils.parse_crs(
            crs, self.region.to_crs(4326).total_bounds
        )
        if self.geoms["region"].crs != pyproj_crs:
            self.geoms["region"] = self.geoms["region"].to_crs(pyproj_crs)

        # update config for geographic coordinates
        if pyproj_crs.is_geographic:
            self.set_config("crsgeo", 1)

        # create grid from region
        # NOTE keyword rotated is added to still have the possibility to create unrotated grids if needed (e.g. for FEWS?)
        if rotated:
            geom = self.geoms["region"].union_all()
            x0, y0, mmax, nmax, rot = utils.rotated_grid(
                geom, res, dec_origin=dec_origin, dec_rotation=dec_rotation
            )
        else:
            x0, y0, x1, y1 = self.geoms["region"].total_bounds
            if align:
                x0 = round(x0 / res) * res
                y0 = round(y0 / res) * res
            else:
                x0, y0 = round(x0, dec_origin), round(y0, dec_origin)
            mmax = int(np.ceil((x1 - x0) / res))
            nmax = int(np.ceil((y1 - y0) / res))
            rot = 0
        self.setup_grid(
            x0=x0,
            y0=y0,
            dx=res,
            dy=res,
            nmax=nmax,
            mmax=mmax,
            rotation=rot,
            epsg=pyproj_crs.to_epsg(),
        )

    def setup_dep(
        self,
        datasets_dep: List[dict],
        buffer_cells: int = 0,  # not in list
        interp_method: str = "linear",  # used for buffer cells only
    ):
        """Interpolate topobathy (dep) data to the model grid.

        Adds model grid layers:

        * **dep**: combined elevation/bathymetry [m+ref]

        Parameters
        ----------
        datasets_dep : List[dict]
            List of dictionaries with topobathy data, each containing a dataset name or Path (elevtn) and optional merge arguments e.g.:
            [{'elevtn': merit_hydro, 'zmin': 0.01}, {'elevtn': gebco, 'offset': 0, 'merge_method': 'first', 'reproj_method': 'bilinear'}]
            For a complete overview of all merge options, see :py:func:`hydromt.workflows.merge_multi_dataarrays`
        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels, by default 0
        interp_method : str, optional
            Interpolation method used to fill the buffer cells , by default "linear"
        """

        # retrieve model resolution to determine zoom level for xyz-datasets
        # TODO fix for quadtree
        if not self.mask.raster.crs.is_geographic:
            res = np.abs(self.mask.raster.res[0])
        else:
            res = np.abs(self.mask.raster.res[0]) * 111111.0

        datasets_dep = self._parse_datasets_dep(datasets_dep, res=res)

        if self.grid_type == "regular":
            da_dep = workflows.merge_multi_dataarrays(
                da_list=datasets_dep,
                da_like=self.mask,
                buffer_cells=buffer_cells,
                interp_method=interp_method,
                logger=self.logger,
            )

            # check if no nan data is present in the bed levels
            nmissing = int(np.sum(np.isnan(da_dep.values)))
            if nmissing > 0:
                self.logger.warning(f"Interpolate elevation at {nmissing} cells")
                da_dep = da_dep.raster.interpolate_na(
                    method="rio_idw", extrapolate=True
                )

            self.set_grid(da_dep, name="dep")
            # FIXME this shouldn't be necessary, since da_dep should already have a crs
            if self.crs is not None and self.grid.raster.crs is None:
                self.grid.set_crs(self.crs)

            if "depfile" not in self.config:
                self.config.update({"depfile": "sfincs.dep"})
        elif self.grid_type == "quadtree":
            raise NotImplementedError(
                "Create dep not yet implemented for quadtree grids."
            )

    def setup_mask_active(
        self,
        mask: Union[str, Path, gpd.GeoDataFrame] = None,
        include_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        mask_buffer: int = 0,
        zmin: float = None,
        zmax: float = None,
        fill_area: float = 10.0,
        drop_area: float = 0.0,
        connectivity: int = 8,
        all_touched: bool = True,
        reset_mask: bool = True,
    ):
        """Setup active model cells.

        The SFINCS model mask defines inactive (msk=0), active (msk=1), and waterlevel boundary (msk=2)
        and outflow boundary (msk=3) cells. This method sets the active and inactive cells.

        Active model cells are based on a region and cells with valid elevation (i.e. not nodata),
        optionally bounded by areas inside the include geomtries, outside the exclude geomtries,
        larger or equal than a minimum elevation threshhold and smaller or equal than a
        maximum elevation threshhold.
        All conditions are combined using a logical AND operation.

        Sets model layers:

        * **msk** map: model mask [-]

        Parameters
        ----------
        mask: str, Path, gpd.GeoDataFrame, optional
            Path or data source name of polygons to initiliaze active mask with; proceding arguments can be used to include/exclude cells
            If not given, existing mask (if present) used, else mask is initialized empty.
        include_mask, exclude_mask: str, Path, gpd.GeoDataFrame, optional
            Path or data source name of polygons to include/exclude from the active model domain.
            Note that include (second last) and exclude (last) areas are processed after other critera,
            i.e. `zmin`, `zmax` and `drop_area`, and thus overrule these criteria for active model cells.
        mask_buffer: float, optional
            If larger than zero, extend the `include_mask` geometry with a buffer [m],
            by default 0.
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for active model cells.
        fill_area : float, optional
            Maximum area [km2] of contiguous cells below `zmin` or above `zmax` but surrounded
            by cells within the valid elevation range to be kept as active cells, by default 10 km2.
        drop_area : float, optional
            Maximum area [km2] of contiguous cells to be set as inactive cells, by default 0 km2.
        connectivity, {4, 8}:
            The connectivity used to define contiguous cells, if 4 only horizontal and vertical
            connections are used, if 8 (default) also diagonal connections.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.
        reset_mask: bool, optional
            If True  (default), reset existing mask layer. If False updating existing mask.
        """
        # read geometries
        gdf_mask, gdf_include, gdf_exclude = None, None, None
        bbox = self.region.to_crs(4326).total_bounds
        if mask is not None:
            if not isinstance(mask, gpd.GeoDataFrame) and str(mask).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_mask = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=mask), crs=self.region.crs
                )
            else:
                gdf_mask = self.data_catalog.get_geodataframe(mask, bbox=bbox)
            if mask_buffer > 0:  # NOTE assumes model in projected CRS!
                gdf_mask["geometry"] = gdf_mask.to_crs(self.crs).buffer(mask_buffer)
        if include_mask is not None:
            if not isinstance(include_mask, gpd.GeoDataFrame) and str(
                include_mask
            ).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_include = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=include_mask), crs=self.region.crs
                )
            else:
                gdf_include = self.data_catalog.get_geodataframe(
                    include_mask, bbox=bbox
                )
        if exclude_mask is not None:
            if not isinstance(exclude_mask, gpd.GeoDataFrame) and str(
                exclude_mask
            ).endswith(".pol"):
                gdf_exclude = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=exclude_mask), crs=self.region.crs
                )
            else:
                gdf_exclude = self.data_catalog.get_geodataframe(
                    exclude_mask, bbox=bbox
                )

        # get mask
        if self.grid_type == "regular":
            da_mask = self.reggrid.create_mask_active(
                da_mask=self.grid["msk"] if "msk" in self.grid else None,
                da_dep=self.grid["dep"] if "dep" in self.grid else None,
                gdf_mask=gdf_mask,
                gdf_include=gdf_include,
                gdf_exclude=gdf_exclude,
                zmin=zmin,
                zmax=zmax,
                fill_area=fill_area,
                drop_area=drop_area,
                connectivity=connectivity,
                all_touched=all_touched,
                reset_mask=reset_mask,
                logger=self.logger,
            )
            self.set_grid(da_mask, name="msk")
            # update config
            if "mskfile" not in self.config:
                self.config.update({"mskfile": "sfincs.msk"})
            if "indexfile" not in self.config:
                self.config.update({"indexfile": "sfincs.ind"})
            # update region
            if np.any(da_mask >= 1):
                self.logger.info("Derive region geometry based on active cells.")
                # make mask with ones and zeros only -> vectorize ones
                region = da_mask.where(da_mask <= 1, 1).raster.vectorize()
                if region.empty:
                    raise ValueError("No region found.")
                self.set_geoms(region, "region")
            else:
                self.logger.warning("No active cells found.")

    def setup_mask_bounds(
        self,
        btype: str = "waterlevel",
        include_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        include_mask_buffer: int = 0,
        zmin: float = None,
        zmax: float = None,
        connectivity: int = 8,
        all_touched: bool = False,
        reset_bounds: bool = False,
    ):
        """Set boundary cells in the model mask.

        The SFINCS model mask defines inactive (msk=0), active (msk=1), and waterlevel boundary (msk=2)
        and outflow boundary (msk=3) cells. Active cells set using the `setup_mask` method,
        while this method sets both types of boundary cells, see `btype` argument.

        Boundary cells at the edge of the active model domain,
        optionally bounded by areas inside the include geomtries, outside the exclude geomtries,
        larger or equal than a minimum elevation threshhold and smaller or equal than a
        maximum elevation threshhold.
        All conditions are combined using a logical AND operation.

        Updates model layers:

        * **msk** map: model mask [-]

        Parameters
        ----------
        btype: {'waterlevel', 'outflow'}
            Boundary type
        include_mask, exclude_mask: str, Path, gpd.GeoDataFrame, optional
            Path or data source name for geometries with areas to include/exclude from
            the model boundary.
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for boundary cells.
            Note that when include and exclude areas are used, the elevation range is
            only applied on cells within the include area and outside the exclude area.
        reset_bounds: bool, optional
            If True, reset existing boundary cells of the selected boundary
            type (`btype`) before setting new boundary cells, by default False.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.
        connectivity, {4, 8}:
            The connectivity used to detect the model edge, if 4 only horizontal and vertical
            connections are used, if 8 (default) also diagonal connections.
        """

        # get include / exclude geometries
        gdf_include, gdf_exclude = None, None
        bbox = self.mask.raster.transform_bounds(4326)
        if include_mask is not None:
            if not isinstance(include_mask, gpd.GeoDataFrame) and str(
                include_mask
            ).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_include = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=include_mask), crs=self.region.crs
                )
            else:
                gdf_include = self.data_catalog.get_geodataframe(
                    include_mask, bbox=bbox
                )
            if include_mask_buffer > 0:
                if self.crs.is_geographic:
                    include_mask_buffer = include_mask_buffer / 111111.0
                gdf_include["geometry"] = gdf_include.to_crs(self.crs).buffer(
                    include_mask_buffer
                )
        if exclude_mask is not None:
            if not isinstance(exclude_mask, gpd.GeoDataFrame) and str(
                exclude_mask
            ).endswith(".pol"):
                gdf_exclude = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=exclude_mask), crs=self.region.crs
                )
            else:
                gdf_exclude = self.data_catalog.get_geodataframe(
                    exclude_mask, bbox=bbox
                )

        # mask values
        if self.grid_type == "regular":
            da_mask = self.reggrid.create_mask_bounds(
                da_mask=self.grid["msk"],
                btype=btype,
                gdf_include=gdf_include,
                gdf_exclude=gdf_exclude,
                da_dep=self.grid["dep"] if "dep" in self.grid else None,
                zmin=zmin,
                zmax=zmax,
                connectivity=connectivity,
                all_touched=all_touched,
                reset_bounds=reset_bounds,
                logger=self.logger,
            )
            self.set_grid(da_mask, name="msk")

    def setup_subgrid(
        self,
        datasets_dep: List[dict],
        datasets_rgh: List[dict] = [],
        datasets_riv: List[dict] = [],
        buffer_cells: int = 0,
        nlevels: int = 10,
        nbins: int = None,
        nr_subgrid_pixels: int = 20,
        nrmax: int = 2000,  # blocksize
        max_gradient: float = 99999.0,
        z_minimum: float = -99999.0,
        huthresh: float = 0.01,
        q_table_option: int = 2,
        manning_land: float = 0.04,
        manning_sea: float = 0.02,
        rgh_lev_land: float = 0.0,
        write_dep_tif: bool = False,
        write_man_tif: bool = False,
    ):
        """Setup method for subgrid tables based on a list of
        elevation and Manning's roughness datasets.

        These datasets are used to derive relations between the water level
        and the volume in a cell to do the continuity update,
        and a representative water depth used to calculate momentum fluxes.

        This allows that one can compute on a coarser computational grid,
        while still accounting for the local topography and roughness.

        Parameters
        ----------
        datasets_dep : List[dict]
            List of dictionaries with topobathy data.
            Each should minimally contain a data catalog source name, data file path,
            or xarray raster object ('elevtn').
            Optional merge arguments include: 'zmin', 'zmax', 'mask', 'offset', 'reproj_method',
            and 'merge_method', see example below. For a complete overview of all merge options,
            see :py:func:`hydromt.workflows.merge_multi_dataarrays`

            ::

                [
                    {'elevtn': 'merit_hydro', 'zmin': 0.01},
                    {'elevtn': 'gebco', 'offset': 0, 'merge_method': 'first', reproj_method: 'bilinear'}
                ]

        datasets_rgh : List[dict], optional
            List of dictionaries with Manning's n datasets. Each dictionary should at
            least contain one of the following:

            * manning: filename (or Path) of gridded data with manning values
            * lulc (and reclass_table): a combination of a filename of gridded
              landuse/landcover and a mapping table.

            In additon, optional merge arguments can be provided, e.g.:

            ::

                [
                    {'manning': 'manning_data'},
                    {'lulc': 'esa_worlcover', 'reclass_table': 'esa_worlcover_mapping'}
                ]

        datasets_riv : List[dict], optional
            List of dictionaries with river datasets. Each dictionary should at least
            contain a river centerline data and optionally a river mask:

            * centerlines: filename (or Path) of river centerline with attributes
              rivwth (river width [m]; required if not river mask provided),
              rivdph or rivbed (river depth [m]; river bedlevel [m+REF]),
              manning (Manning's n [s/m^(1/3)]; optional)
            * mask (optional): filename (or Path) of river mask
            * point_zb (optional): filename (or Path) of river points with bed (z) values
            * river attributes (optional): "rivdph", "rivbed", "rivwth", "manning"
              to fill missing values
            * arguments to the river burn method (optional):
              segment_length [m] (default 500m) and riv_bank_q [0-1] (default 0.5)
              which used to estimate the river bank height in case river depth is provided.

            For more info see :py:func:`hydromt.workflows.bathymetry.burn_river_rect`

           ::

                [{'centerlines': 'river_lines', 'mask': 'river_mask', 'manning': 0.035}]

        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels,
            by default 0
        nbins : int, optional
            Number of bins in which hypsometry is subdivided, by default 10
            Note that this keyword is deprecated and will be removed in future versions.
        nlevels: int, optional
            Number of levels to describe hypsometry, by default 10
        nr_subgrid_pixels : int, optional
            Number of subgrid pixels per computational cell, by default 20
        nrmax : int, optional
            Maximum number of cells per subgrid-block, by default 2000
            These blocks are used to prevent memory issues while working with large datasets
        max_gradient : float, optional
            If slope in hypsometry exceeds this value, then smoothing is applied,
            to prevent numerical stability problems, by default 5.0
        z_minimum : float, optional
            Minimum depth in the subgrid tables, by default -99999.0
        huthresh : float, optional
            Threshold depth in SFINCS model, by default 0.01 m
        q_table_option : int, optional
            Option for the computation of the representative roughness and conveyance depth at u/v points, by default 2.
            1: "old" weighting method, compliant with SFINCS < v2.1.1, taking the avarage of the adjacent cells
            2: "improved" weighting method, recommended for SFINCS >= v2.1.1, that takes into account the wet fractions of the adjacent cells
        manning_land, manning_sea : float, optional
            Constant manning roughness values for land and sea, by default 0.04 and 0.02 s.m-1/3
            Note that these values are only used when no Manning's n datasets are provided,
            or to fill the nodata values
        rgh_lev_land : float, optional
            Elevation level to distinguish land and sea roughness
            (when using manning_land and manning_sea), by default 0.0
        write_dep_tif, write_man_tif : bool, optional
            Write geotiff of the merged topobathy / roughness on the subgrid resolution.
            These files are not used by SFINCS, but can be used for visualisation and
            downscaling of the floodmaps. Unlinke the SFINCS files it is written
            to disk at execution of this method. By default False
        """
        # retrieve model resolution
        # TODO fix for quadtree
        if not self.mask.raster.crs.is_geographic:
            res = np.abs(self.mask.raster.res[0]) / nr_subgrid_pixels
        else:
            res = np.abs(self.mask.raster.res[0]) * 111111.0 / nr_subgrid_pixels

        datasets_dep = self._parse_datasets_dep(datasets_dep, res=res)

        if len(datasets_rgh) > 0:
            # NOTE conversion from landuse/landcover to manning happens here
            datasets_rgh = self._parse_datasets_rgh(datasets_rgh)

        if len(datasets_riv) > 0:
            datasets_riv = self._parse_datasets_riv(datasets_riv)

        # folder where high-resolution topobathy and manning geotiffs are stored
        if write_dep_tif or write_man_tif:
            highres_dir = os.path.join(self.root, "subgrid")
            if not os.path.isdir(highres_dir):
                os.makedirs(highres_dir)
        else:
            highres_dir = None

        if nbins is not None:
            logger.warning(
                "Keyword nbins is deprecated and will be removed in future versions. Please use nlevels instead."
            )
            nlevels = nbins

        if q_table_option == 1 and max_gradient > 20.0:
            raise ValueError(
                "For the old q_table_option, a max_gradient of 5.0 is recommended to improve numerical stability"
            )

        if self.grid_type == "regular":
            self.reggrid.subgrid.build(
                da_mask=self.mask,
                datasets_dep=datasets_dep,
                datasets_rgh=datasets_rgh,
                datasets_riv=datasets_riv,
                buffer_cells=buffer_cells,
                nlevels=nlevels,
                nr_subgrid_pixels=nr_subgrid_pixels,
                nrmax=nrmax,
                max_gradient=max_gradient,
                z_minimum=z_minimum,
                huthresh=huthresh,
                q_table_option=q_table_option,
                manning_land=manning_land,
                manning_sea=manning_sea,
                rgh_lev_land=rgh_lev_land,
                write_dep_tif=write_dep_tif,
                write_man_tif=write_man_tif,
                highres_dir=highres_dir,
                logger=self.logger,
            )
            self.subgrid = self.reggrid.subgrid.to_xarray(
                dims=self.mask.raster.dims, coords=self.mask.raster.coords
            )
        elif self.grid_type == "quadtree":
            pass

        # when building a new subgrid table, always update config
        # NOTE from now onwards, netcdf subgrid tables are used
        self.config.update({"sbgfile": "sfincs_subgrid.nc"})
        # if "sbgfile" not in self.config:  # only add sbgfile if not already present
        #     self.config.update({"sbgfile": "sfincs.sbg"})
        # subgrid is used so no depfile or manningfile needed
        if "depfile" in self.config:
            self.config.pop("depfile")  # remove depfile from config
        if "manningfile" in self.config:
            self.config.pop("manningfile")  # remove manningfile from config

    def setup_river_inflow(
        self,
        rivers: Union[str, Path, gpd.GeoDataFrame] = None,
        hydrography: Union[str, Path, xr.Dataset] = None,
        buffer: float = 200,
        river_upa: float = 10.0,
        river_len: float = 1e3,
        river_width: float = 500,
        merge: bool = False,
        first_index: int = 1,
        keep_rivers_geom: bool = False,
        reverse_river_geom: bool = False,
        src_type: str = "inflow",
    ):
        """Setup discharge (src) points where a river enters the model domain.

        If `rivers` is not provided, river centerlines are extracted from the
        `hydrography` dataset based on the `river_upa` threshold.

        Waterlevel or outflow boundary cells intersecting with the river
        are removed from the model mask.

        Discharge is set to zero at these points, but can be updated
        using the `setup_discharge_forcing` or `setup_discharge_forcing_from_grid` methods.

        Note: this method assumes the rivers are directed from up- to downstream. Use
        `reverse_river_geom=True` if the rivers are directed from downstream to upstream.

        Adds model layers:

        * **dis** forcing: discharge forcing
        * **mask** map: SFINCS mask layer (only if `river_width` > 0)
        * **rivers_inflow** geoms: river centerline (if `keep_rivers_geom`; not used by SFINCS)

        Parameters
        ----------
        rivers : str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for river centerline data.
            If present, the 'uparea' and 'rivlen' attributes are used.
        hydrography: str, Path, xr.Dataset optional
            Path, data source name, or a xarray raster object for hydrography data.

            * Required layers: ['uparea', 'flwdir'].
        buffer: float, optional
            Buffer around the model region boundary to define in/outflow points [m],
            by default 200 m. We suggest to use a buffer of at least twice the hydrography
            resolution. Inflow points are moved to a downstreawm confluence if within the buffer.
        river_upa : float, optional
            Minimum upstream area threshold for rivers [km2], by default 10.0
        river_len: float, optional
            Mimimum river length within the model domain threshhold [m], by default 1 km.
        river_width: float, optional
            Estimated constant width [m] of the inflowing river. Boundary cells within
            half the width are forced to be closed (mask = 1) to avoid instabilities with
            nearby open or waterlevel boundary cells, by default 500 m.
        merge: bool, optional
            If True, merge rivers source points with existing points, by default False.
        first_index: int, optional
            First index for the river source points, by default 1.
        keep_rivers_geom: bool, optional
            If True, keep a geometry of the rivers "rivers_inflow" in geoms. By default False.
        reverse_river_geom: bool, optional
            If True, assume that segments in 'rivers' are drawn from downstream to upstream.
            Only used if 'rivers' is not None, By default False
        src_type: {'inflow', 'headwater'}, optional
            Source type, by default 'inflow'
            If 'inflow', return points where the river flows into the model domain.
            If 'headwater', return all headwater (including inflow) points within the model domain.

        See Also
        --------
        setup_discharge_forcing
        setup_discharge_forcing_from_grid
        """
        # get hydrography data
        da_uparea = None
        if hydrography is not None:
            ds = self.data_catalog.get_rasterdataset(
                hydrography,
                bbox=self.mask.raster.transform_bounds(4326),
                variables=["uparea", "flwdir"],
                buffer=5,
            )
            da_uparea = ds["uparea"]  # reused in river_source_points

        # get river centerlines
        if (
            isinstance(rivers, str)
            and rivers == "rivers_outflow"
            and rivers in self.geoms
        ):
            # reuse rivers from setup_river_in/outflow
            gdf_riv = self.geoms[rivers]
        elif rivers is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                rivers, geom=self.region
            ).to_crs(self.crs)
        elif hydrography is not None:
            gdf_riv = workflows.river_centerline_from_hydrography(
                da_flwdir=ds["flwdir"],
                da_uparea=da_uparea,
                river_upa=river_upa,
                river_len=river_len,
                gdf_mask=self.region,
            )
        elif hydrography is None:
            raise ValueError("Either hydrography or rivers must be provided.")

        # get river inflow / headwater source points
        gdf_src = workflows.river_source_points(
            gdf_riv=gdf_riv,
            gdf_mask=self.region,
            src_type=src_type,
            buffer=buffer,
            river_upa=river_upa,
            river_len=river_len,
            da_uparea=da_uparea,
            reverse_river_geom=reverse_river_geom,
            logger=self.logger,
        )
        if gdf_src.empty:
            return

        # set forcing src pnts
        gdf_src.index = gdf_src.index + first_index
        self.set_forcing_1d(gdf_locs=gdf_src.copy(), name="dis", merge=merge)

        # set river
        if keep_rivers_geom:
            self.set_geoms(gdf_riv, name="rivers_inflow")

        # update mask if river_width > 0
        if "rivwth" in gdf_src.columns:
            river_width = gdf_src["rivwth"].fillna(river_width)
        if np.any(river_width > 0) and np.any(self.mask > 1):
            # apply buffer
            gdf_src["geometry"] = gdf_src.buffer(river_width / 2)
            # find intersect of buffer and model grid
            tmp_msk = self.reggrid.create_mask_bounds(
                xr.where(self.mask > 0, 1, 0).astype(np.uint8), gdf_include=gdf_src
            )
            reset_msk = np.logical_and(tmp_msk > 1, self.mask > 1)
            # update model mask
            n = int(np.sum(reset_msk))
            if n > 0:
                da_mask = self.mask.where(~reset_msk, np.uint8(1))
                self.set_grid(da_mask, "msk")
                self.logger.info(f"Boundary cells (n={n}) updated around src points.")

    def setup_river_outflow(
        self,
        rivers: Union[str, Path, gpd.GeoDataFrame] = None,
        hydrography: Union[str, Path, xr.Dataset] = None,
        river_upa: float = 10.0,
        river_len: float = 1e3,
        river_width: float = 500,
        keep_rivers_geom: bool = False,
        reset_bounds: bool = False,
        btype: str = "outflow",
        reverse_river_geom: bool = False,
    ):
        """Setup open boundary cells (mask=3) where a river flows
        out of the model domain.

        If `rivers` is not provided, river centerlines are extracted from the
        `hydrography` dataset based on the `river_upa` threshold.

        River outflows that intersect with discharge source point or waterlevel
        boundary cells are omitted.

        Note: this method assumes the rivers are directed from up- to downstream.

        Adds / edits model layers:

        * **msk** map: edited by adding outflow points (msk=3)
        * **rivers_outflow** geoms: river centerline (if `keep_rivers_geom`; not used by SFINCS)

        Parameters
        ----------
        rivers : str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for river centerline data.
            If present, the 'uparea' and 'rivlen' attributes are used.
        hydrography: str, Path, xr.Dataset optional
            Path, data source name, or a xarray raster object for hydrography data.

            * Required layers: ['uparea', 'flwdir'].
        river_upa : float, optional
            Minimum upstream area threshold for rivers [km2], by default 10.0
        river_len: float, optional
            Mimimum river length within the model domain threshhold [m], by default 1000 m.
        river_width: int, optional
            The width [m] of the open boundary cells in the SFINCS msk file.
            By default 500m, i.e.: 250m to each side of the outflow location.
        append_bounds: bool, optional
            If True, write new outflow boundary cells on top of existing. If False (default),
            first reset existing outflow boundary cells to normal active cells.
        keep_rivers_geom: bool, optional
            If True, keep a geometry of the rivers "rivers_outflow" in geoms. By default False.
        reset_bounds: bool, optional
            If True, reset existing outlfow boundary cells before setting new boundary cells,
            by default False.
        btype: {'waterlevel', 'outflow'}
            Boundary type
        reverse_river_geom: bool, optional
            If True, assume that segments in 'rivers' are drawn from downstream to upstream.
            Only used if rivers is not None, By default False

        See Also
        --------
        setup_mask_bounds
        """
        # get hydrography data
        da_uparea = None
        if hydrography is not None:
            ds = self.data_catalog.get_rasterdataset(
                hydrography,
                bbox=self.mask.raster.transform_bounds(4326),
                variables=["uparea", "flwdir"],
                buffer=5,
            )
            da_uparea = ds["uparea"]  # reused in river_source_points

        # get river centerlines
        if (
            isinstance(rivers, str)
            and rivers == "rivers_inflow"
            and rivers in self.geoms
        ):
            # reuse rivers from setup_river_in/outflow
            gdf_riv = self.geoms[rivers]
        elif rivers is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                rivers, geom=self.region
            ).to_crs(self.crs)
        elif hydrography is not None:
            gdf_riv = workflows.river_centerline_from_hydrography(
                da_flwdir=ds["flwdir"],
                da_uparea=da_uparea,
                river_upa=river_upa,
                river_len=river_len,
                gdf_mask=self.region,
            )
        else:
            raise ValueError("Either hydrography or rivers must be provided.")

        # estimate buffer based on model resolution
        buffer = self.reggrid.dx
        if self.crs.is_geographic:
            buffer = buffer * 111111.0

        # get river inflow / headwater source points
        gdf_out = workflows.river_source_points(
            gdf_riv=gdf_riv,
            gdf_mask=self.region,
            src_type="outflow",
            buffer=buffer,
            river_upa=river_upa,
            river_len=river_len,
            da_uparea=da_uparea,
            reverse_river_geom=reverse_river_geom,
            logger=self.logger,
        )
        if gdf_out.empty:
            return

        if len(gdf_out) > 0:
            if "rivwth" in gdf_out.columns:
                river_width = gdf_out["rivwth"].fillna(river_width)
            gdf_out["geometry"] = gdf_out.buffer(river_width / 2)
            # remove points near waterlevel boundary cells
            if np.any(self.mask == 2) and btype == "outflow":
                gdf_msk2 = utils.get_bounds_vector(self.mask)
                # NOTE: this should be a single geom
                geom = gdf_msk2[gdf_msk2["value"] == 2].union_all()
                gdf_out = gdf_out[~gdf_out.intersects(geom)]
            # remove outflow points near source points
            if "dis" in self.forcing and len(gdf_out) > 0:
                geom = self.forcing["dis"].vector.to_gdf().union_all()
                gdf_out = gdf_out[~gdf_out.intersects(geom)]

        # update mask
        n = len(gdf_out.index)
        self.logger.info(f"Found {n} valid river outflow points.")
        if n > 0:
            self.setup_mask_bounds(
                btype=btype, include_mask=gdf_out, reset_bounds=reset_bounds
            )
        elif reset_bounds:
            self.setup_mask_bounds(btype=btype, reset_bounds=reset_bounds)

        # keep river centerlines
        if keep_rivers_geom and len(gdf_riv) > 0:
            self.set_geoms(gdf_riv, name="rivers_outflow")

    # Function to create constant spatially varying infiltration
    def setup_constant_infiltration(
        self,
        qinf=None,
        lulc=None,
        reclass_table=None,
        reproj_method="average",
    ):
        """Setup spatially varying constant infiltration rate (qinffile).

        Adds model layers:

        * **qinf** map: constant infiltration rate [mm/hr]

        Parameters
        ----------
        qinf : str, Path, or RasterDataset
            Spatially varying infiltration rates [mm/hr]
        lulc: str, Path, or RasterDataset
            Landuse/landcover data set
        reclass_table: str, Path, or pd.DataFrame
            Reclassification table to convert landuse/landcover to infiltration rates [mm/hr]
        reproj_method : str, optional
            Resampling method for reprojecting the infiltration data to the model grid.
            By default 'average'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        """

        # get infiltration data
        if qinf is not None:
            da_inf = self.data_catalog.get_rasterdataset(
                qinf,
                bbox=self.mask.raster.transform_bounds(4326),
                buffer=10,
            )
        elif lulc is not None:
            # landuse/landcover should always be combined with mapping
            if reclass_table is None:
                raise IOError(
                    f"Infiltration mapping file should be provided for {lulc}"
                )
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc,
                bbox=self.mask.raster.transform_bounds(4326),
                buffer=10,
                variables=["lulc"],
            )
            df_map = self.data_catalog.get_dataframe(
                reclass_table,
                variables=["qinf"],
                index_col=0,  # driver kwargs
            )
            # reclassify
            da_inf = da_lulc.raster.reclassify(df_map)["qinf"]
        else:
            raise ValueError(
                "Either qinf or lulc must be provided when setting up constant infiltration."
            )

        # reproject infiltration data to model grid
        da_inf = da_inf.raster.mask_nodata()  # set nodata to nan
        da_inf = da_inf.raster.reproject_like(self.mask, method=reproj_method)

        # check on nan values
        if np.logical_and(np.isnan(da_inf), self.mask >= 1).any():
            self.logger.warning("NaN values found in infiltration data; filled with 0")
            da_inf = da_inf.fillna(0)
        da_inf.raster.set_nodata(-9999.0)

        # set grid
        mname = "qinf"
        da_inf.attrs.update(**self._ATTRS.get(mname, {}))
        self.set_grid(da_inf, name=mname)

        # update config: remove default inf and set qinf map
        self.set_config(f"{mname}file", f"sfincs.{mname}")
        self.config.pop("qinf", None)

    # Function to create curve number for SFINCS
    def setup_cn_infiltration(self, cn, antecedent_moisture="avg", reproj_method="med"):
        """Setup model potential maximum soil moisture retention map (scsfile)
        from gridded curve number map.

        Adds model layers:

        * **scs** map: potential maximum soil moisture retention [inch]

        Parameters
        ---------
        cn: str, Path, or RasterDataset
            Name of gridded curve number map.

            * Required layers without antecedent runoff conditions: ['cn']
            * Required layers with antecedent runoff conditions: ['cn_dry', 'cn_avg', 'cn_wet']
        antecedent_moisture: {'dry', 'avg', 'wet'}, optional
            Antecedent runoff conditions.
            None if data has no antecedent runoff conditions.
            By default `avg`
        reproj_method : str, optional
            Resampling method for reprojecting the curve number data to the model grid.
            By default 'med'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        """
        # get data
        da_org = self.data_catalog.get_rasterdataset(
            cn, bbox=self.mask.raster.transform_bounds(4326), buffer=10
        )
        # read variable
        v = "cn"
        if antecedent_moisture:
            v = f"cn_{antecedent_moisture}"
        if isinstance(da_org, xr.Dataset) and v in da_org.data_vars:
            da_org = da_org[v]
        elif not isinstance(da_org, xr.DataArray):
            raise ValueError(f"Could not find variable {v} in {cn}")

        # reproject using median
        da_cn = da_org.raster.reproject_like(self.grid, method=reproj_method)

        # convert to potential maximum soil moisture retention S (1000/CN - 10) [inch]
        da_scs = workflows.cn_to_s(da_cn, self.mask > 0).round(3)

        # set grid
        mname = "scs"
        da_scs.attrs.update(**self._ATTRS.get(mname, {}))
        self.set_grid(da_scs, name=mname)
        # update config: remove default infiltration values and set scs map
        self.config.pop("qinf", None)
        self.set_config(f"{mname}file", f"sfincs.{mname}")

    # Function to create curve number for SFINCS including recovery via saturated hydraulic conductivity [mm/hr]
    def setup_cn_infiltration_with_ks(
        self, lulc, hsg, ksat, reclass_table, effective, block_size=2000
    ):
        """Setup model the Soil Conservation Service (SCS) Curve Number (CN) files for SFINCS
        including recovery term based on the soil saturation

        Parameters
        ---------
        lulc : str, Path, or RasterDataset
            Landuse/landcover data set
        hsg : str, Path, or RasterDataset
            HSG (Hydrological Similarity Group) in integers
        ksat : str, Path, or RasterDataset
            Ksat (saturated hydraulic conductivity) [mm/hr]
        reclass_table : str, Path, or RasterDataset
            reclass table to relate landcover with soiltype
        effective : float
            estimate of percentage effective soil, e.g. 0.50 for 50%
        block_size : float
            maximum block size - use larger values will get more data in memory but can be faster, default=2000
        """

        # Read the datafiles
        da_landuse = self.data_catalog.get_rasterdataset(
            lulc, bbox=self.mask.raster.transform_bounds(4326), buffer=10
        )
        da_HSG = self.data_catalog.get_rasterdataset(
            hsg, bbox=self.mask.raster.transform_bounds(4326), buffer=10
        )
        da_Ksat = self.data_catalog.get_rasterdataset(
            ksat, bbox=self.mask.raster.transform_bounds(4326), buffer=10
        )
        df_map = self.data_catalog.get_dataframe(reclass_table, index_col=0)

        # Define outputs
        da_smax = xr.full_like(self.mask, -9999, dtype=np.float32)
        da_ks = xr.full_like(self.mask, -9999, dtype=np.float32)

        # Compute resolution land use (we are assuming that is the finest)
        resolution_landuse = np.mean(
            [abs(da_landuse.raster.res[0]), abs(da_landuse.raster.res[1])]
        )
        if da_landuse.raster.crs.is_geographic:
            resolution_landuse = (
                resolution_landuse * 111111.0
            )  # assume 1 degree is 111km

        # Define the blocks
        nrmax = block_size
        nmax = np.shape(self.mask)[0]
        mmax = np.shape(self.mask)[1]
        refi = self.config["dx"] / resolution_landuse  # finest resolution of landuse
        nrcb = int(np.floor(nrmax / refi))  # nr of regular cells in a block
        nrbn = int(np.ceil(nmax / nrcb))  # nr of blocks in n direction
        nrbm = int(np.ceil(mmax / nrcb))  # nr of blocks in m direction
        x_dim, y_dim = self.mask.raster.x_dim, self.mask.raster.y_dim

        # avoid blocks with width or height of 1
        merge_last_col = False
        merge_last_row = False
        if mmax % nrcb == 1:
            nrbm -= 1
            merge_last_col = True
        if nmax % nrcb == 1:
            nrbn -= 1
            merge_last_row = True

        ## Loop through blocks
        ib = -1
        for ii in range(nrbm):
            bm0 = ii * nrcb  # Index of first m in block
            bm1 = min(bm0 + nrcb, mmax)  # last m in block
            if merge_last_col and ii == (nrbm - 1):
                bm1 += 1

            for jj in range(nrbn):
                bn0 = jj * nrcb  # Index of first n in block
                bn1 = min(bn0 + nrcb, nmax)  # last n in block
                if merge_last_row and jj == (nrbn - 1):
                    bn1 += 1

                # Count
                ib += 1
                self.logger.debug(
                    f"\nblock {ib + 1}/{nrbn * nrbm} -- "
                    f"col {bm0}:{bm1-1} | row {bn0}:{bn1-1}"
                )

                # calculate transform and shape of block at cell and subgrid level
                da_mask_block = self.mask.isel(
                    {x_dim: slice(bm0, bm1), y_dim: slice(bn0, bn1)}
                ).load()

                # Call workflow
                (
                    da_smax_block,
                    da_ks_block,
                ) = workflows.curvenumber.scs_recovery_determination(
                    da_landuse, da_HSG, da_Ksat, df_map, da_mask_block
                )

                # New place in the overall matrix
                sn, sm = slice(bn0, bn1), slice(bm0, bm1)
                da_smax[sn, sm] = da_smax_block
                da_ks[sn, sm] = da_ks_block

        # Done
        self.logger.info(f"Done with determination of values (in blocks).")

        # Specify the effective soil retention (seff)
        da_seff = da_smax
        da_seff = da_seff * effective
        da_seff.raster.set_nodata(da_smax.raster.nodata)

        # set grids for seff, smax and ks (saturated hydraulic conductivity)
        names = ["smax", "seff", "ks"]
        data = [da_smax, da_seff, da_ks]
        for name, da in zip(names, data):
            # Give metadata to the layer and set grid
            da.attrs.update(**self._ATTRS.get(name, {}))
            self.set_grid(da, name=name)

            # update config: set maps
            self.set_config(f"{name}file", f"sfincs.{name}")  # give it to SFINCS

        # Remove qinf variable in sfincs
        self.config.pop("qinf", None)

    # Roughness
    def setup_manning_roughness(
        self,
        datasets_rgh: List[dict] = [],
        manning_land=0.04,
        manning_sea=0.02,
        rgh_lev_land=0,
    ):
        """Setup model manning roughness map (manningfile) from gridded manning data or a combinataion of gridded
        land-use/land-cover map and manning roughness mapping table.

        Adds model layers:

        * **man** map: manning roughness coefficient [s.m-1/3]

        Parameters
        ---------
        datasets_rgh : List[dict], optional
            List of dictionaries with Manning's n datasets. Each dictionary should at least contain one of the following:
            * (1) manning: filename (or Path) of gridded data with manning values
            * (2) lulc (and reclass_table) :a combination of a filename of gridded landuse/landcover and a mapping table.
            In additon, optional merge arguments can be provided e.g.: merge_method, gdf_valid_fn
        manning_land, manning_sea : float, optional
            Constant manning roughness values for land and sea, by default 0.04 and 0.02 s.m-1/3
            Note that these values are only used when no Manning's n datasets are provided, or to fill the nodata values
        rgh_lev_land : float, optional
            Elevation level to distinguish land and sea roughness (when using manning_land and manning_sea), by default 0.0
        """

        if len(datasets_rgh) > 0:
            datasets_rgh = self._parse_datasets_rgh(datasets_rgh)
        else:
            datasets_rgh = []

        # fromdep keeps track of whether any manning values should be based on the depth or not
        fromdep = len(datasets_rgh) == 0
        if self.grid_type == "regular":
            if len(datasets_rgh) > 0:
                da_man = workflows.merge_multi_dataarrays(
                    da_list=datasets_rgh,
                    da_like=self.mask,
                    interp_method="linear",
                    logger=self.logger,
                )
                fromdep = np.isnan(da_man).where(self.mask > 0, False).any()
            if "dep" in self.grid and fromdep:
                da_man0 = xr.where(
                    self.grid["dep"] >= rgh_lev_land, manning_land, manning_sea
                )
            elif fromdep:
                da_man0 = xr.full_like(self.mask, manning_land, dtype=np.float32)

            if len(datasets_rgh) > 0 and fromdep:
                self.logger.warning("nan values in manning roughness array")
                da_man = da_man.where(~np.isnan(da_man), da_man0)
            elif fromdep:
                da_man = da_man0
            da_man.raster.set_nodata(-9999.0)

            # set grid
            mname = "manning"
            da_man.attrs.update(**self._ATTRS.get(mname, {}))
            self.set_grid(da_man, name=mname)
            # update config: remove default manning values and set maning map
            for v in ["manning_land", "manning_sea", "rgh_lev_land"]:
                self.config.pop(v, None)
            self.set_config(f"{mname}file", f"sfincs.{mname[:3]}")

    def setup_observation_points(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Setup model observation point locations.

        Adds model layers:

        * **obs** geom: observation point locations

        Parameters
        ---------
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for observation point locations.
        merge: bool, optional
            If True, merge the new observation points with the existing ones. By default True.
        """
        name = self._GEOMS["observation_points"]

        # FIXME ensure the catalog is loaded before adding any new entries
        self.data_catalog.sources

        gdf_obs = self.data_catalog.get_geodataframe(
            locations, geom=self.region, assert_gtype="Point", **kwargs
        ).to_crs(self.crs)

        if not gdf_obs.geometry.type.isin(["Point"]).all():
            raise ValueError("Observation points must be of type Point.")

        if merge and name in self.geoms:
            gdf0 = self._geoms.pop(name)
            gdf_obs = gpd.GeoDataFrame(pd.concat([gdf_obs, gdf0], ignore_index=True))
            self.logger.info(f"Adding new observation points to existing ones.")

        self.set_geoms(gdf_obs, name)
        self.set_config(f"{name}file", f"sfincs.{name}")

    def setup_observation_lines(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Setup model observation lines (cross-sections) to monitor discharges.

        Adds model layers:

        * **crs** geom: observation lines (cross-sections)

        Parameters
        ---------
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for observation lines (cross-sections).
        merge: bool, optional
            If True, merge the new observation lines with the existing ones. By default True.
        """
        name = self._GEOMS["observation_lines"]

        # FIXME ensure the catalog is loaded before adding any new entries
        self.data_catalog.sources

        # FIXME assert_gtype="LineString" does not work for MultiLineString and default seems to be Point (??)
        gdf_obs = self.data_catalog.get_geodataframe(
            locations, geom=self.region, assert_gtype=None, **kwargs
        ).to_crs(self.crs)

        # make sure MultiLineString are converted to LineString
        gdf_obs = gdf_obs.explode(index_parts=True).reset_index(drop=True)

        if not gdf_obs.geometry.type.isin(["LineString"]).all():
            raise ValueError("Observation lines must be of type LineString.")

        if merge and name in self.geoms:
            gdf0 = self._geoms.pop(name)
            gdf_obs = gpd.GeoDataFrame(pd.concat([gdf_obs, gdf0], ignore_index=True))
            self.logger.info(f"Adding new observation lines to existing ones.")

        self.set_geoms(gdf_obs, name)
        self.set_config(f"{name}file", f"sfincs.{name}")

    def setup_structures(
        self,
        structures: Union[str, Path, gpd.GeoDataFrame],
        stype: str,
        dep: Union[str, Path, xr.DataArray] = None,
        buffer: float = None,
        dz: float = None,
        merge: bool = True,
        **kwargs,
    ):
        """Setup thin dam or weir structures.

        Adds model layer (depending on `stype`):

        * **thd** geom: thin dam
        * **weir** geom: weir / levee

        Parameters
        ----------
        structures : str, Path
            Path, data source name, or geopandas object to structure line geometry file.
            The "name" (for thd and weir), "z" and "par1" (for weir only) variables are optional.
            For weirs: `dz` must be provided if gdf has no "z" column or ZLineString;
            "par1" defaults to 0.6 if gdf has no "par1" column.
        stype : {'thd', 'weir'}
            Structure type.
        dep : str, Path, xr.DataArray, optional
            Path, data source name, or xarray raster object ('elevtn') describing the depth in an
            alternative resolution which is used for sampling the weir.
        buffer : float, optional
            If provided, describes the distance from the centerline to the foot of the structure.
            This distance is supplied to the raster.sample as the window (wdw).
        merge : bool, optional
            If True, merge with existing'stype' structures, by default True.
        dz: float, optional
            If provided, for weir structures the z value is calculated from
            the model elevation (dep) plus dz.
        """

        # read, clip and reproject
        gdf_structures = self.data_catalog.get_geodataframe(
            structures, geom=self.region, **kwargs
        ).to_crs(self.crs)

        cols = {
            "thd": ["name", "geometry"],
            "weir": ["name", "z", "par1", "geometry"],
        }
        assert stype in cols, f"stype must be one of {list(cols.keys())}"
        gdf = gdf_structures[
            [c for c in cols[stype] if c in gdf_structures.columns]
        ]  # keep relevant cols

        structs = utils.gdf2linestring(gdf)  # check if it parsed correct
        # sample zb values from dep file and set z = zb + dz
        if stype == "weir" and (dep is not None or dz is not None):
            if dep is None or dep == "dep":
                assert "dep" in self.grid, "dep layer not found"
                elv = self.grid["dep"]
            else:
                elv = self.data_catalog.get_rasterdataset(
                    dep, geom=self.region, buffer=5, variables=["elevtn"]
                )

            # calculate window size from buffer
            if buffer is not None:
                res = abs(elv.raster.res[0])
                if elv.raster.crs.is_geographic:
                    res = res * 111111.0
                window_size = int(np.ceil(buffer / res))
            else:
                window_size = 0
            self.logger.debug(f"Sampling elevation with window size {window_size}")

            structs_out = []
            for s in structs:
                pnts = gpd.points_from_xy(x=s["x"], y=s["y"])
                zb = elv.raster.sample(
                    gpd.GeoDataFrame(geometry=pnts, crs=self.crs), wdw=window_size
                )
                if zb.ndim > 1:
                    zb = zb.max(axis=1)

                s["z"] = zb.values
                if dz is not None:
                    s["z"] += float(dz)
                structs_out.append(s)
            gdf = utils.linestring2gdf(structs_out, crs=self.crs)
        # Else function if you define elevation of weir
        elif stype == "weir" and np.any(["z" not in s for s in structs]):
            raise ValueError("Weir structure requires z values.")
        # combine with existing structures if present
        if merge and stype in self.geoms:
            gdf0 = self._geoms.pop(stype)
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf0], ignore_index=True))
            self.logger.info(f"Adding {stype} structures to existing structures.")

        # set structures
        self.set_geoms(gdf, stype)
        self.set_config(f"{stype}file", f"sfincs.{stype}")

    def setup_drainage_structures(
        self,
        structures: Union[str, Path, gpd.GeoDataFrame],
        stype: str = "pump",
        discharge: float = 0.0,
        merge: bool = True,
        **kwargs,
    ):
        """Setup drainage structures.

        Adds model layer:
        * **drn** geom: drainage pump or culvert

        Parameters
        ----------
        structures : str, Path
            Path, data source name, or geopandas object to structure line geometry file.
            The line should consist of only 2 points (else first and last points are used), ordered from up to downstream.
            The "type" (1 for pump and 2 for culvert), "par1" ("discharge" also accepted) variables are optional.
            If "type" or "par1" are not provided, they are based on stype or discharge arguments.
        stype : {'pump', 'culvert'}, optional
            Structure type, by default "pump". stype is converted to integer "type" to match with SFINCS expectations.
        discharge : float, optional
            Discharge of the structure, by default 0.0. For culverts, this is the maximum discharge,
            since actual discharge depends on waterlevel gradient
        merge : bool, optional
            If True, merge with existing drainage structures, by default True.
        """

        stype = stype.lower()
        svalues = {"pump": 1, "culvert": 2}
        if stype not in svalues:
            raise ValueError('stype must be one of "pump", "culvert"')
        svalue = svalues[stype]

        # read, clip and reproject
        gdf_structures = self.data_catalog.get_geodataframe(
            structures, geom=self.region, **kwargs
        ).to_crs(self.crs)

        # check if type (int) is present in gdf, else overwrite from args
        # TODO also add check if type is interger?
        if "type" not in gdf_structures:
            gdf_structures["type"] = svalue
        # if discharge is provided, rename to par1
        if "discharge" in gdf_structures:
            gdf_structures = gdf_structures.rename(columns={"discharge": "par1"})

        # add par1, par2, par3, par4, par5 if not present
        # NOTE only par1 is used in the model
        if "par1" not in gdf_structures:
            gdf_structures["par1"] = discharge
        if "par2" not in gdf_structures:
            gdf_structures["par2"] = 0
        if "par3" not in gdf_structures:
            gdf_structures["par3"] = 0
        if "par4" not in gdf_structures:
            gdf_structures["par4"] = 0
        if "par5" not in gdf_structures:
            gdf_structures["par5"] = 0

        # multi to single lines
        lines = gdf_structures.explode(column="geometry").reset_index(drop=True)
        # get start [0] and end [1] points
        endpoints = lines.boundary.explode(index_parts=True).unstack()
        # merge start and end points into a single linestring
        gdf_structures["geometry"] = endpoints.apply(
            lambda x: LineString(x.values.tolist()), axis=1
        )

        # combine with existing structures if present
        if merge and "drn" in self.geoms:
            gdf0 = self._geoms.pop("drn")
            gdf_structures = gpd.GeoDataFrame(
                pd.concat([gdf_structures, gdf0], ignore_index=True)
            )
            self.logger.info(f"Adding {stype} structures to existing structures.")

        # set structures
        self.set_geoms(gdf_structures, "drn")
        self.set_config("drnfile", "sfincs.drn")

    def setup_storage_volume(
        self,
        storage_locs: Union[str, Path, gpd.GeoDataFrame],
        volume: Union[float, List[float]] = None,
        height: Union[float, List[float]] = None,
        merge: bool = True,
    ):
        """Setup storage volume.

        Adds model layer:
        * **vol** map: storage volume for green infrastructure

        Parameters
        ----------
        storage_locs : str, Path
            Path, data source name, or geopandas object to storage location polygon or point geometry file.
            Optional "volume" or "height" attributes can be provided to set the storage volume.
        volume : float, optional
            Storage volume [m3], by default None
        height : float, optional
            Storage height [m], by default None
        merge : bool, optional
            If True, merge with existing storage volumes, by default True.

        """

        # read, clip and reproject
        gdf = self.data_catalog.get_geodataframe(
            storage_locs,
            geom=self.region,
            buffer=10,
        ).to_crs(self.crs)

        if self.grid_type == "regular":
            # if merge, add new storage volumes to existing ones
            if merge and "vol" in self.grid:
                da_vol = self.grid["vol"]
            else:
                da_vol = xr.full_like(self.mask, 0, dtype=np.float64)

            # add storage volumes form gdf to da_vol
            da_vol = workflows.add_storage_volume(
                da_vol,
                gdf,
                volume=volume,
                height=height,
                logger=self.logger,
            )

            # set grid
            mname = "vol"
            da_vol.attrs.update(**self._ATTRS.get(mname, {}))
            self.set_grid(da_vol, name=mname)
            # update config
            self.set_config(f"{mname}file", f"sfincs.{mname[:3]}")

    ### FORCING
    def set_forcing_1d(
        self,
        df_ts: pd.DataFrame = None,
        gdf_locs: gpd.GeoDataFrame = None,
        name: str = "bzs",
        merge: bool = True,
    ):
        """Set 1D forcing time series for 'bzs' or 'dis' boundary conditions.

        1D forcing exists of point location `gdf_locs` and associated timeseries `df_ts`.
        If `gdf_locs` is None, the currently set locations are used.

        If merge is True, time series in `df_ts` with the same index will
        overwrite existing data. Time series with new indices are added to
        the existing forcing.

        In case the forcing time series have a numeric index, the index is converted to
        a datetime index assuming the index is in seconds since `tref`.

        Parameters
        ----------
        df_ts : pd.DataFrame, optional
            1D forcing time series data. If None, dummy forcing data is added.
        gdf_locs : gpd.GeoDataFrame, optional
            Location of waterlevel boundary points. If None, the currently set locations are used.
        name : str, optional
            Name of the waterlevel boundary time series file, by default 'bzs'.
        merge : bool, optional
            If True, merge with existing forcing data, by default True.
        """
        # check dtypes
        if gdf_locs is not None:
            if not isinstance(gdf_locs, gpd.GeoDataFrame):
                raise ValueError("gdf_locs must be a gpd.GeoDataFrame")
            if not gdf_locs.index.is_integer() and gdf_locs.index.is_unique:
                raise ValueError("gdf_locs index must be unique integer values")
            if not gdf_locs.geometry.type.isin(["Point"]).all():
                raise ValueError("gdf_locs geometry must be Point")
            if gdf_locs.crs != self.crs:
                gdf_locs = gdf_locs.to_crs(self.crs)
        elif name in self.forcing:
            gdf_locs = self.forcing[name].vector.to_gdf()
        if df_ts is not None:
            if not isinstance(df_ts, pd.DataFrame):
                raise ValueError("df_ts must be a pd.DataFrame")
            if not df_ts.columns.is_integer() and df_ts.columns.is_unique:
                raise ValueError("df_ts column names must be unique integer values")
        # parse datetime index
        if df_ts is not None and df_ts.index.is_numeric():
            if "tref" not in self.config:
                raise ValueError(
                    "tref must be set in config to convert numeric index to datetime index"
                )
            tref = utils.parse_datetime(self.config["tref"])
            df_ts.index = tref + pd.to_timedelta(df_ts.index, unit="sec")
        # parse location index
        if (
            gdf_locs is not None
            and df_ts is not None
            and gdf_locs.index.size == df_ts.columns.size
            and not set(gdf_locs.index) == set(df_ts.columns)
        ):
            # loop over integer columns and find matching index
            for col in gdf_locs.select_dtypes(include=np.integer).columns:
                if set(gdf_locs[col]) == set(df_ts.columns):
                    gdf_locs = gdf_locs.set_index(col)
                    self.logger.info(f"Setting gdf_locs index to {col}")
                    break
            if not set(gdf_locs.index) == set(df_ts.columns):
                gdf_locs = gdf_locs.set_index(df_ts.columns)
                self.logger.info(
                    f"No matching index column found in gdf_locs; assuming the order is correct"
                )
        # merge with existing data
        if name in self.forcing and merge:
            # read existing data
            da = self.forcing[name]
            gdf0 = da.vector.to_gdf()
            df0 = da.transpose(..., da.vector.index_dim).to_pandas()
            if set(gdf0.index) != set(gdf_locs.index):
                # merge locations; overwrite existing locations with the same name
                gdf0 = gdf0.drop(gdf_locs.index, errors="ignore")
                gdf_locs = pd.concat([gdf0, gdf_locs], axis=0).sort_index()
                # gdf_locs = gpd.GeoDataFrame(gdf_locs, crs=gdf0.crs)
                df0 = df0.reindex(gdf_locs.index, axis=1, fill_value=0)
            if df_ts is None:
                df_ts = df0
            elif set(df0.columns) != set(df_ts.columns):
                # merge timeseries; overwrite existing timeseries with the same name
                df0 = df0.drop(columns=df_ts.columns, errors="ignore")
                df_ts = pd.concat([df0, df_ts], axis=1).sort_index()
                # use linear interpolation and backfill to fill in missing values
                df_ts = df_ts.sort_index()
                df_ts = df_ts.interpolate(method="linear").bfill().fillna(0)
        # location data is required
        if gdf_locs is None:
            raise ValueError(
                f"gdf_locs must be provided if not merged with existing {name} forcing data"
            )
        # fill in missing timeseries
        if df_ts is None:
            df_ts = pd.DataFrame(
                index=pd.date_range(*self.get_model_time(), periods=2),
                data=0,
                columns=gdf_locs.index,
            )
        # set forcing with consistent names
        if not set(gdf_locs.index) == set(df_ts.columns):
            raise ValueError("The gdf_locs index and df_ts columns must be the same")
        gdf_locs.index.name = "index"
        df_ts.columns.name = "index"
        df_ts.index.name = "time"
        da = GeoDataArray.from_gdf(gdf_locs.to_crs(self.crs), data=df_ts, name=name)
        self.set_forcing(da.transpose("time", "index"))

    def setup_waterlevel_forcing(
        self,
        geodataset: Union[str, Path, xr.Dataset] = None,
        timeseries: Union[str, Path, pd.DataFrame] = None,
        locations: Union[str, Path, gpd.GeoDataFrame] = None,
        offset: Union[str, Path, xr.Dataset] = None,
        buffer: float = 5e3,
        merge: bool = True,
    ):
        """Setup waterlevel forcing.

        Waterlevel boundary conditions are read from a `geodataset` (geospatial point timeseries)
        or a tabular `timeseries` dataframe. At least one of these must be provided.

        The tabular timeseries data is combined with `locations` if provided,
        or with existing 'bnd' locations if previously set.

        Adds model forcing layers:

        * **bzs** forcing: waterlevel time series [m+ref]

        Parameters
        ----------
        geodataset: str, Path, xr.Dataset, optional
            Path, data source name, or xarray data object for geospatial point timeseries.
        timeseries: str, Path, pd.DataFrame, optional
            Path, data source name, or pandas data object for tabular timeseries.
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for bnd point locations.
            It should contain a 'index' column matching the column names in `timeseries`.
        offset: str, Path, xr.Dataset, float, optional
            Path, data source name, constant value or xarray raster data for gridded offset
            between vertical reference of elevation and waterlevel data,
            The offset is added to the waterlevel data.
        buffer: float, optional
            Buffer [m] around model water level boundary cells to select waterlevel gauges,
            by default 5 km.
        merge : bool, optional
            If True, merge with existing forcing data, by default True.

        See Also
        --------
        set_forcing_1d
        """
        gdf_locs, df_ts = None, None
        tstart, tstop = self.get_model_time()  # model time
        # buffer around msk==2 values
        if np.any(self.mask == 2):
            region = self.mask.where(self.mask == 2, 0).raster.vectorize()
        else:
            region = self.region
        # read waterlevel data from geodataset or geodataframe
        if geodataset is not None:
            # read and clip data in time & space
            da = self.data_catalog.get_geodataset(
                geodataset,
                geom=region,
                buffer=buffer,
                variables=["waterlevel"],
                time_tuple=(tstart, tstop),
                crs=self.crs,
            )
            df_ts = da.transpose(..., da.vector.index_dim).to_pandas()
            gdf_locs = da.vector.to_gdf()
        elif timeseries is not None:
            df_ts = self.data_catalog.get_dataframe(
                timeseries,
                time_tuple=(tstart, tstop),
                # kwargs below only applied if timeseries not in data catalog
                parse_dates=True,
                index_col=0,
            )
            df_ts.columns = df_ts.columns.map(int)  # parse column names to integers

        # read location data (if not already read from geodataset)
        if gdf_locs is None and locations is not None:
            gdf_locs = self.data_catalog.get_geodataframe(
                locations, geom=region, buffer=buffer, crs=self.crs
            ).to_crs(self.crs)
            if "index" in gdf_locs.columns:
                gdf_locs = gdf_locs.set_index("index")
            # filter df_ts timeseries based on gdf_locs index
            # this allows to use a subset of the locations in the timeseries
            if df_ts is not None and np.isin(gdf_locs.index, df_ts.columns).all():
                df_ts = df_ts.reindex(gdf_locs.index, axis=1, fill_value=0)
        elif gdf_locs is None and "bzs" in self.forcing:
            gdf_locs = self.forcing["bzs"].vector.to_gdf()
        elif gdf_locs is None:
            raise ValueError("No waterlevel boundary (bnd) points provided.")

        # optionally read offset data and correct df_ts
        if offset is not None and gdf_locs is not None:
            if isinstance(offset, (float, int)):
                df_ts += offset
            else:
                da_offset = self.data_catalog.get_rasterdataset(
                    offset,
                    bbox=self.mask.raster.transform_bounds(4326),
                    buffer=5,
                )
                offset_pnts = da_offset.raster.sample(gdf_locs)
                df_offset = offset_pnts.to_pandas().reindex(df_ts.columns).fillna(0)
                df_ts = df_ts + df_offset
                offset = offset_pnts.mean().values
            self.logger.debug(
                f"waterlevel forcing: applied offset (avg: {offset:+.2f})"
            )

        # set/ update forcing
        self.set_forcing_1d(df_ts=df_ts, gdf_locs=gdf_locs, name="bzs", merge=merge)

    def setup_waterlevel_bnd_from_mask(
        self,
        distance: float = 1e4,
        merge: bool = True,
    ):
        """Setup waterlevel boundary (bnd) points along model waterlevel boundary (msk=2).

        The waterlevel boundary (msk=2) should be set before calling this method,
        e.g.: with `setup_mask_bounds`

        Waterlevels (bzs) are set to zero at these points, but can be updated
        with `setup_waterlevel_forcing`.

        Parameters
        ----------
        distance: float, optional
            Distance [m] between waterlevel boundary points,
            by default 10 km.
        merge : bool, optional
            If True, merge with existing forcing data, by default True.

        See Also
        --------
        setup_waterlevel_forcing
        setup_mask_bounds
        """
        # get waterlevel boundary vector based on mask
        gdf_msk = utils.get_bounds_vector(self.mask)
        gdf_msk2 = gdf_msk[gdf_msk["value"] == 2]

        # convert to meters if crs is geographic
        if self.mask.raster.crs.is_geographic:
            distance = distance / 111111.0

        # create points along boundary
        points = []
        for _, row in gdf_msk2.iterrows():
            distances = np.arange(0, row.geometry.length, distance)
            for d in distances:
                point = row.geometry.interpolate(d)
                points.append((point.x, point.y))

        # create geodataframe with points
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*points)), crs=self.crs)

        # set waterlevel boundary
        self.set_forcing_1d(gdf_locs=gdf, name="bzs", merge=merge)

    def setup_discharge_forcing(
        self,
        geodataset=None,
        timeseries=None,
        locations=None,
        merge=True,
        buffer: float = None,
    ):
        """Setup discharge forcing.

        Discharge timeseries are read from a `geodataset` (geospatial point timeseries)
        or a tabular `timeseries` dataframe. At least one of these must be provided.

        The tabular timeseries data is combined with `locations` if provided,
        or with existing 'src' locations if previously set, e.g., with the
        `setup_river_inflow` method.

        Adds model layers:

        * **dis** forcing: discharge time series [m3/s]

        Parameters
        ----------
        geodataset: str, Path, xr.Dataset, optional
            Path, data source name, or xarray data object for geospatial point timeseries.
        timeseries: str, Path, pd.DataFrame, optional
            Path, data source name, or pandas data object for tabular timeseries.
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for bnd point locations.
            It should contain a 'index' column matching the column names in `timeseries`.
        merge : bool, optional
            If True, merge with existing forcing data, by default True.
        buffer: float, optional
            Buffer [m] around model boundary within the model region
            select discharge gauges, by default None.

        See Also
        --------
        setup_river_inflow
        """
        gdf_locs, df_ts = None, None
        tstart, tstop = self.get_model_time()  # model time
        # buffer
        region = self.region
        if buffer is not None:  # TODO this assumes the model crs is projected
            region = region.boundary.buffer(buffer).clip(self.region)
        # read waterlevel data from geodataset or geodataframe
        if geodataset is not None:
            # read and clip data in time & space
            da = self.data_catalog.get_geodataset(
                geodataset,
                geom=region,
                variables=["discharge"],
                time_tuple=(tstart, tstop),
                crs=self.crs,
            )
            df_ts = da.transpose(..., da.vector.index_dim).to_pandas()
            gdf_locs = da.vector.to_gdf()
        elif timeseries is not None:
            df_ts = self.data_catalog.get_dataframe(
                timeseries,
                time_tuple=(tstart, tstop),
                # kwargs below only applied if timeseries not in data catalog
                parse_dates=True,
                index_col=0,
            )
            df_ts.columns = df_ts.columns.map(int)  # parse column names to integers

        # read location data (if not already read from geodataset)
        if gdf_locs is None and locations is not None:
            gdf_locs = self.data_catalog.get_geodataframe(
                locations, geom=region, crs=self.crs
            ).to_crs(self.crs)
            if "index" in gdf_locs.columns:
                gdf_locs = gdf_locs.set_index("index")
            # filter df_ts timeseries based on gdf_locs index
            # this allows to use a subset of the locations in the timeseries
            if df_ts is not None and np.isin(gdf_locs.index, df_ts.columns).all():
                df_ts = df_ts.reindex(gdf_locs.index, axis=1, fill_value=0)
        elif gdf_locs is None and "dis" in self.forcing:
            gdf_locs = self.forcing["dis"].vector.to_gdf()
        elif gdf_locs is None:
            raise ValueError("No discharge boundary (src) points provided.")

        # set/ update forcing
        self.set_forcing_1d(df_ts=df_ts, gdf_locs=gdf_locs, name="dis", merge=merge)

    def setup_discharge_forcing_from_grid(
        self,
        discharge,
        locations=None,
        uparea=None,
        wdw=1,
        rel_error=0.05,
        abs_error=50,
    ):
        """Setup discharge forcing based on a gridded discharge dataset.

        Discharge boundary timesereis are read from the `discharge` dataset
        with gridded discharge time series data.

        The `locations` are snapped to the `uparea` grid if provided based their
        uparea attribute. If not provided, the nearest grid cell is used.

        Adds model layers:

        * **dis** forcing: discharge time series [m3/s]

        Adds meta layer (not used by SFINCS):

        * **src_snapped** geom: snapped gauge location on discharge grid

        Parameters
        ----------
        discharge: str, Path, xr.DataArray optional
            Path,  data source name or xarray data object for gridded discharge timeseries dataset.

            * Required variables: ['discharge' (m3/s)]
            * Required coordinates: ['time', 'y', 'x']
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas data object for point location dataset.
            Not required if point location have previously been set, e.g. using the
            :py:meth:`~hydromt_sfincs.SfincsModel.setup_river_inflow` method.

            * Required variables: ['uparea' (km2)]
        uparea: str, Path, optional
            Path, data source name, or xarray data object for upstream area grid.

            * Required variables: ['uparea' (km2)]
        wdw: int, optional
            Window size in number of cells around discharge boundary locations
            to snap to, only used if ``uparea`` is provided. By default 1.
        rel_error, abs_error: float, optional
            Maximum relative error (default 0.05) and absolute error (default 50 km2)
            between the discharge boundary location upstream area and the upstream area of
            the best fit grid cell, only used if "discharge" geoms has a "uparea" column.

        See Also
        --------
        setup_river_inflow
        """
        if locations is not None:
            gdf = self.data_catalog.get_geodataframe(
                locations, geom=self.region, assert_gtype="Point"
            ).to_crs(self.crs)
        elif "dis" in self.forcing:
            gdf = self.forcing["dis"].vector.to_gdf()
        else:
            raise ValueError("No discharge boundary (src) points provided.")

        # read data
        ds = self.data_catalog.get_rasterdataset(
            discharge,
            bbox=self.mask.raster.transform_bounds(4326),
            buffer=2,
            time_tuple=self.get_model_time(),  # model time
            variables=["discharge"],
            single_var_as_array=False,
        )
        if uparea is not None and "uparea" in gdf.columns:
            da_upa = self.data_catalog.get_rasterdataset(
                uparea,
                bbox=self.mask.raster.transform_bounds(4326),
                buffer=2,
                variables=["uparea"],
            )
            # make sure ds and da_upa align
            ds["uparea"] = da_upa.raster.reproject_like(ds, method="nearest")
        elif "uparea" not in gdf.columns:
            self.logger.warning('No "uparea" column found in location data.')

        # TODO use hydromt core method
        ds_snapped = workflows.snap_discharge(
            ds=ds,
            gdf=gdf,
            wdw=wdw,
            rel_error=rel_error,
            abs_error=abs_error,
            uparea_name="uparea",
            discharge_name="discharge",
            logger=self.logger,
        )
        # set zeros for src points without matching discharge
        da_q = ds_snapped["discharge"].reindex(index=gdf.index, fill_value=0).fillna(0)
        df_q = da_q.transpose("time", ...).to_pandas()
        # update forcing
        self.set_forcing_1d(df_ts=df_q, gdf_locs=gdf, name="dis")
        # keep snapped locations
        self.set_geoms(ds_snapped.vector.to_gdf(), "src_snapped")

    def setup_precip_forcing_from_grid(
        self, precip, dst_res=None, aggregate=False, **kwargs
    ):
        """Setup precipitation forcing from a gridded spatially varying data source.

        If aggregate is True, spatially uniform precipitation forcing is added to
        the model based on the mean precipitation over the model domain.
        If aggregate is False, distributed precipitation is added to the model as netcdf file.
        The data is reprojected to the model CRS (and destination resolution `dst_res` if provided).

        Adds one of these model layer:

        * **netamprfile** forcing: distributed precipitation [mm/hr]
        * **precipfile** forcing: uniform precipitation [mm/hr]

        Parameters
        ----------
        precip, str, Path
            Path to precipitation rasterdataset netcdf file.

            * Required variables: ['precip' (mm)]
            * Required coordinates: ['time', 'y', 'x']

        dst_res: float
            output resolution (m), by default None and computed from source data.
            Only used in combination with aggregate=False
        aggregate: bool, {'mean', 'median'}, optional
            Method to aggregate distributed input precipitation data. If True, mean
            aggregation is used, if False (default) the data is not aggregated and
            spatially distributed precipitation is returned.
        """
        # get data for model domain and config time range
        precip = self.data_catalog.get_rasterdataset(
            precip,
            bbox=self.mask.raster.transform_bounds(4326),
            buffer=2,
            time_tuple=self.get_model_time(),
            variables=["precip"],
        )

        # aggregate or reproject in space
        if aggregate:
            stat = aggregate if isinstance(aggregate, str) else "mean"
            self.logger.debug(f"Aggregate precip using {stat}.")
            zone = self.region.dissolve()  # make sure we have a single (multi)polygon
            precip_out = precip.raster.zonal_stats(zone, stats=stat)[f"precip_{stat}"]
            df_ts = precip_out.where(precip_out >= 0, 0).fillna(0).squeeze().to_pandas()
            self.setup_precip_forcing(df_ts.to_frame())
        else:
            # reproject to model utm crs
            # NOTE: currently SFINCS errors (stack overflow) on large files,
            # downscaling to model grid is not recommended
            kwargs0 = dict(align=dst_res is not None, method="nearest_index")
            kwargs0.update(kwargs)
            meth = kwargs0["method"]
            self.logger.debug(f"Resample precip using {meth}.")
            precip_out = precip.raster.reproject(
                dst_crs=self.crs, dst_res=dst_res, **kwargs
            ).fillna(0)

            # only resample in time if freq < 1H, else keep input values
            if da_to_timedelta(precip_out) < pd.to_timedelta("1H"):
                precip_out = hydromt.workflows.resample_time(
                    precip_out,
                    freq=pd.to_timedelta("1H"),
                    conserve_mass=True,
                    upsampling="bfill",
                    downsampling="sum",
                    logger=self.logger,
                )
            precip_out = precip_out.rename("precip_2d")

            # add to forcing
            self.set_forcing(precip_out, name="precip_2d")

    def setup_precip_forcing(self, timeseries=None, magnitude=None):
        """Setup spatially uniform precipitation forcing (precip).

        Adds model layers:

        * **precipfile** forcing: uniform precipitation [mm/hr]

        Parameters
        ----------
        timeseries: str, Path
            Path to tabulated timeseries csv file with time index in first column
            and location IDs in the first row,
            see :py:meth:`hydromt.open_timeseries_from_table`, for details.
            Note: tabulated timeseries files cannot yet be set through the data_catalog yml file.
        magnitude: float
            Precipitation magnitude [mm/hr] to use if no timeseries is provided.
        """
        tstart, tstop = self.get_model_time()
        if timeseries is not None:
            df_ts = self.data_catalog.get_dataframe(
                timeseries,
                time_tuple=(tstart, tstop),
                # kwargs below only applied if timeseries not in data catalog
                parse_dates=True,
                index_col=0,
            )
        elif magnitude is not None:
            times = pd.date_range(*self.get_model_time(), freq="10T")
            df_ts = pd.DataFrame(
                index=times, data=np.full((len(times), 1), magnitude, dtype=float)
            )
        else:
            raise ValueError("Either timeseries or magnitude must be provided")

        if isinstance(df_ts, pd.DataFrame):
            df_ts = df_ts.squeeze()
        if not isinstance(df_ts, pd.Series):
            raise ValueError("df_ts must be a pandas.Series")
        df_ts.name = "precip"
        df_ts.index.name = "time"
        self.set_forcing(df_ts.to_xarray(), name="precip")

    def setup_pressure_forcing_from_grid(
        self, press, dst_res=None, fill_value=101325, **kwargs
    ):
        """Setup pressure forcing from a gridded spatially varying data source.

        Adds one model layer:

        * **netampfile** forcing: distributed barometric pressure [Pa]

        Parameters
        ----------
        press, str, Path, xr.Dataset, xr.DataArray
            Path to pressure rasterdataset netcdf file or xarray dataset.

            * Required variables: ['press_msl' (Pa)]
            * Required coordinates: ['time', 'y', 'x']

        dst_res: float
            output resolution (m), by default None and computed from source data.

        fill_value: float
            value to use when no data is available.
            Standard atmospheric pressure (101325 Pa) is used if no value is given.
        """
        # get data for model domain and config time range
        press = self.data_catalog.get_rasterdataset(
            press,
            geom=self.region,
            buffer=2,
            time_tuple=self.get_model_time(),
            variables=["press_msl"],
        )

        # reproject to model utm crs
        # NOTE: currently SFINCS errors (stack overflow) on large files,
        # downscaling to model grid is not recommended
        kwargs0 = dict(align=dst_res is not None, method="nearest_index")
        kwargs0.update(kwargs)
        meth = kwargs0["method"]
        self.logger.debug(f"Resample pressure using {meth}.")
        press_out = press.raster.reproject(
            dst_crs=self.crs, dst_res=dst_res, **kwargs
        ).fillna(fill_value)

        # only resample in time if freq < 1H, else keep input values
        if da_to_timedelta(press_out) < pd.to_timedelta("1H"):
            press_out = hydromt.workflows.resample_time(
                press_out,
                freq=pd.to_timedelta("1H"),
                conserve_mass=False,
                upsampling="interpolate",
                downsampling="interpolate",
                logger=self.logger,
            )

        press_out = press_out.rename("press_2d")

        # add to forcing
        self.set_forcing(press_out, name="press_2d")

    def setup_wind_forcing_from_grid(self, wind, dst_res=None, **kwargs):
        """Setup pressure forcing from a gridded spatially varying data source.

        Adds one model layer:

        * **netamuamv** forcing: distributed wind [m/s]

        Parameters
        ----------
        wind, str, Path, xr.Dataset
            Path to wind rasterdataset (including eastward and northward components) netcdf file or xarray dataset.

            * Required variables: ['wind10_u' (m/s), 'wind10_v' (m/s)]
            * Required coordinates: ['time', 'y', 'x']

        dst_res: float
            output resolution (m), by default None and computed from source data.
        """
        # get data for model domain and config time range
        wind = self.data_catalog.get_rasterdataset(
            wind,
            geom=self.region,
            buffer=2,
            time_tuple=self.get_model_time(),
            variables=["wind10_u", "wind10_v"],
        )

        # reproject to model utm crs
        # NOTE: currently SFINCS errors (stack overflow) on large files,
        # downscaling to model grid is not recommended
        kwargs0 = dict(align=dst_res is not None, method="nearest_index")
        kwargs0.update(kwargs)
        meth = kwargs0["method"]
        self.logger.debug(f"Resample wind using {meth}.")

        wind = wind.raster.reproject(
            dst_crs=self.crs, dst_res=dst_res, **kwargs
        ).fillna(0)

        # only resample in time if freq < 1H, else keep input values
        if da_to_timedelta(wind) < pd.to_timedelta("1H"):
            wind_out = xr.Dataset()
            # resample in time
            for var in wind.data_vars:
                wind_out[var] = hydromt.workflows.resample_time(
                    wind[var],
                    freq=pd.to_timedelta("1H"),
                    conserve_mass=False,
                    upsampling="interpolate",
                    downsampling="interpolate",
                    logger=self.logger,
                )
        else:
            wind_out = wind

        # add to forcing
        self.set_forcing(wind_out, name="wind_2d")

    def setup_wind_forcing(self, timeseries=None, magnitude=None, direction=None):
        """Setup spatially uniform wind forcing (wind).

        Adds model layers:

        * **windfile** forcing: uniform wind magnitude [m/s] and direction [deg]

        Parameters
        ----------
        timeseries, str, Path
            Path to tabulated timeseries csv file with time index in first column,
            magnitude in second column and direction in third column
            see :py:meth:`hydromt.open_timeseries_from_table`, for details.
            Note: tabulated timeseries files cannot yet be set through the data_catalog yml file.
        magnitude: float
            Magnitude of the wind [m/s]
        direction: float
            Direction where the wind is coming from [deg], e.g. 0 is north, 90 is east, etc.
        """
        tstart, tstop = self.get_model_time()
        if timeseries is not None:
            df_ts = self.data_catalog.get_dataframe(
                timeseries,
                time_tuple=(tstart, tstop),
                # kwargs below only applied if timeseries not in data catalog
                parse_dates=True,
                index_col=0,
            )
        elif magnitude is not None and direction is not None:
            df_ts = pd.DataFrame(
                index=pd.date_range(*self.get_model_time(), periods=2),
                data=np.array([[magnitude, direction], [magnitude, direction]]),
                columns=["mag", "dir"],
            )
        else:
            raise ValueError(
                "Either timeseries or magnitude and direction must be provided"
            )

        df_ts.name = "wnd"
        df_ts.index.name = "time"
        df_ts.columns.name = "index"
        da = xr.DataArray(
            df_ts.values,
            dims=("time", "index"),
            coords={"time": df_ts.index, "index": ["mag", "dir"]},
        )
        self.set_forcing(da, name="wnd")

    def setup_tiles(
        self,
        path: Union[str, Path] = None,
        region: dict = None,
        datasets_dep: List[dict] = [],
        zoom_range: Union[int, List[int]] = [0, 13],
        z_range: List[int] = [-20000.0, 20000.0],
        create_index_tiles: bool = True,
        create_topobathy_tiles: bool = True,
        fmt: str = "bin",
    ):
        """Create both index and topobathy tiles in webmercator format.

        Parameters
        ----------
        path : Union[str, Path]
            Directory in which to store the index tiles, if None, the model root + tiles is used.
        region : dict
            Dictionary describing region of interest, e.g.:
            * {'bbox': [xmin, ymin, xmax, ymax]}. Note bbox should be provided in WGS 84
            * {'geom': 'path/to/polygon_geometry'}
            If None, the model region is used.
        datasets_dep : List[dict]
            List of dictionaries with topobathy data, each containing a dataset name or Path (elevtn) and optional merge arguments e.g.:
            [{'elevtn': merit_hydro, 'zmin': 0.01}, {'elevtn': gebco, 'offset': 0, 'merge_method': 'first', reproj_method: 'bilinear'}]
            For a complete overview of all merge options, see :py:func:`~hydromt.workflows.merge_multi_dataarrays`
            Note that subgrid/dep_subgrid.tif is automatically used if present and datasets_dep is left empty.
        zoom_range : Union[int, List[int]], optional
            Range of zoom levels for which tiles are created, by default [0,13]
        z_range : List[int], optional
            Range of valid elevations that are included in the topobathy tiles, by default [-20000.0, 20000.0]
        create_index_tiles : bool, optional
            If True, index tiles are created, by default True
        create_topobathy_tiles : bool, optional
            If True, topobathy tiles are created, by default True.
        fmt : str, optional
            Format of the tiles: "bin" (binary, default), or "png".
        """
        # use model root if path not provided
        if path is None:
            path = os.path.join(self.root, "tiles")

        # use model region if region not provided
        if region is None:
            region = self.region
        else:
            _kind, _region = hydromt.workflows.parse_region(region=region)
            if "bbox" in _region:
                bbox = _region["bbox"]
                region = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
            elif "geom" in _region:
                region = _region["geom"]
                if region.crs is None:
                    raise ValueError('Model region "geom" has no CRS')

        # if only one zoom level is specified, create tiles up to that zoom level (inclusive)
        if isinstance(zoom_range, int):
            zoom_range = [0, zoom_range]

        # create index tiles
        if create_index_tiles:
            # only binary and png are supported for index tiles so set to binary if tif
            fmt_ind = "bin" if fmt == "tif" else fmt

            if self.grid_type == "regular":
                self.reggrid.create_index_tiles(
                    region=region,
                    root=path,
                    zoom_range=zoom_range,
                    fmt=fmt_ind,
                    logger=self.logger,
                )
            elif self.grid_type == "quadtree":
                raise NotImplementedError(
                    "Index tiles not yet implemented for quadtree grids."
                )

        # create topobathy tiles
        if create_topobathy_tiles:
            # compute resolution of highest zoom level
            # resolution of zoom level 0  on equator: 156543.03392804097
            res = 156543.03392804097 / 2 ** zoom_range[1]
            datasets_dep = self._parse_datasets_dep(datasets_dep, res=res)

            # if no datasets provided, check if high-res subgrid geotiff is there
            if len(datasets_dep) == 0:
                if os.path.exists(os.path.join(self.root, "subgrid")):
                    # check if there is a dep_subgrid.tif
                    dep = os.path.join(self.root, "subgrid", "dep_subgrid.tif")
                    if os.path.exists(dep):
                        da = self.data_catalog.get_rasterdataset(dep)
                        datasets_dep.append({"da": da})
                    else:
                        raise ValueError("No topobathy datasets provided.")

            # create topobathy tiles
            workflows.tiling.create_topobathy_tiles(
                root=path,
                region=region,
                datasets_dep=datasets_dep,
                index_path=os.path.join(path, "indices"),
                zoom_range=zoom_range,
                z_range=z_range,
                fmt=fmt,
            )

    # Plotting
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
        elif variable.startswith("subgrid.") and self.subgrid is not None:
            ds = self.subgrid.copy()
            variable = variable.replace("subgrid.", "")
        else:
            ds = self.grid.copy()
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
            **kwargs,
        )

        if fn_out is not None:
            if not os.path.isabs(fn_out):
                fn_out = join(self.root, "figs", fn_out)
            if not os.path.isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))
            plt.savefig(fn_out, dpi=225, bbox_inches="tight")

        return fig, ax

    # I/O
    def read(self, epsg: int = None):
        """Read the complete model schematization and configuration from file."""
        self.read_config(epsg=epsg)
        if epsg is None and "epsg" not in self.config:
            raise ValueError("Please specify epsg to read this model")
        self.read_grid()
        self.read_subgrid()
        self.read_geoms()
        self.read_forcing()
        self.logger.info("Model read")

    def write(self):
        """Write the complete model schematization and configuration to file."""
        self.logger.info(f"Writing model data to {self.root}")
        # TODO - add check for subgrid & quadtree > give flags to self.write_grid() and self.write_config()
        self.write_grid()
        self.write_subgrid()
        self.write_geoms()
        self.write_forcing()
        self.write_states()
        # config last; might be udpated when writing maps, states or forcing
        self.write_config()
        # write data catalog with used data sources
        self.write_data_catalog()  # new in hydromt v0.4.4

    def read_grid(self, data_vars: Union[List, str] = None) -> None:
        """Read SFINCS binary grid files and save to `grid` attribute.
        Filenames are taken from the `config` attribute (i.e. input file).

        Parameters
        ----------
        data_vars : Union[List, str], optional
            List of data variables to read, by default None (all)
        """
        if self._grid is None:
            self._grid = xr.Dataset()  # avoid reading grid twice

        da_lst = []
        if data_vars is None:
            data_vars = self._MAPS
        elif isinstance(data_vars, str):
            data_vars = list(data_vars)

        # read index file
        ind_fn = self.get_config("indexfile", fallback="sfincs.ind", abs_path=True)
        if not isfile(ind_fn):
            raise IOError(f".ind path {ind_fn} does not exist")

        dtypes = {"msk": "u1"}
        mvs = {"msk": 0}
        if self.reggrid is not None:
            ind = self.reggrid.read_ind(ind_fn=ind_fn)

            for name in data_vars:
                if f"{name}file" in self.config:
                    fn = self.get_config(
                        f"{name}file", fallback=f"sfincs.{name}", abs_path=True
                    )
                    if not isfile(fn):
                        self.logger.warning(f"{name}file not found at {fn}")
                        continue
                    dtype = dtypes.get(name, "f4")
                    mv = mvs.get(name, -9999.0)
                    da = self.reggrid.read_map(fn, ind, dtype, mv, name=name)
                    da_lst.append(da)
            ds = xr.merge(da_lst)
            epsg = self.config.get("epsg", None)
            if epsg is not None:
                ds.raster.set_crs(epsg)
            self.set_grid(ds)

            # keep some metadata maps from gis directory
            fns = glob.glob(join(self.root, "gis", "*.tif"))
            fns = [
                fn
                for fn in fns
                if basename(fn).split(".")[0] not in self.grid.data_vars
            ]
            if fns:
                ds = hydromt.open_mfraster(fns).load()
                self.set_grid(ds)
                ds.close()

    def write_grid(self, data_vars: Union[List, str] = None):
        """Write SFINCS grid to binary files including map index file.
        Filenames are taken from the `config` attribute (i.e. input file).

        If `write_gis` property is True, all grid variables are written to geotiff
        files in a "gis" subfolder.

        Parameters
        ----------
        data_vars : Union[List, str], optional
            List of data variables to write, by default None (all)
        """
        self._assert_write_mode

        dtypes = {"msk": "u1"}  # default to f4
        if self.reggrid and len(self.grid.data_vars) > 0 and "msk" in self.grid:
            # make sure orientation is S->N
            ds_out = self.grid
            if ds_out.raster.res[1] < 0:
                ds_out = ds_out.raster.flipud()
            mask = ds_out["msk"].values

            self.logger.debug("Write binary map indices based on mask.")
            ind_fn = self.get_config("indexfile", abs_path=True)
            self.reggrid.write_ind(ind_fn=ind_fn, mask=mask)

            if data_vars is None:  # write all maps
                data_vars = [v for v in self._MAPS if v in ds_out]
            elif isinstance(data_vars, str):
                data_vars = list(data_vars)
            self.logger.debug(f"Write binary map files: {data_vars}.")
            for name in data_vars:
                if f"{name}file" not in self.config:
                    self.set_config(f"{name}file", f"sfincs.{name}")
                # do not write depfile if subgrid is used
                if (name == "dep" or name == "manning") and self.subgrid:
                    continue
                self.reggrid.write_map(
                    map_fn=self.get_config(f"{name}file", abs_path=True),
                    data=ds_out[name].values,
                    mask=mask,
                    dtype=dtypes.get(name, "f4"),
                )

        if self._write_gis:
            self.write_raster("grid")

    def read_subgrid(self):
        """Read SFINCS subgrid file and add to `subgrid` attribute.
        Filename is taken from the `config` attribute (i.e. input file)."""

        self._assert_read_mode

        if "sbgfile" in self.config:
            fn = self.get_config("sbgfile", abs_path=True)
            if not isfile(fn):
                self.logger.warning(f"sbgfile not found at {fn}")
                return

            # re-initialize subgrid (different variables for old/new version)
            # TODO: come up with a better way to handle this
            self.reggrid.subgrid = SubgridTableRegular()
            self.subgrid = xr.Dataset()

            # read subgrid file
            if fn.parts[-1].endswith(".sbg"):  # read binary file
                self.reggrid.subgrid.read_binary(file_name=fn, mask=self.mask)
            else:  # read netcdf file
                self.reggrid.subgrid.read(file_name=fn, mask=self.mask)
            self.subgrid = self.reggrid.subgrid.to_xarray(
                dims=self.mask.raster.dims, coords=self.mask.raster.coords
            )

    def write_subgrid(self):
        """Write SFINCS subgrid file."""
        self._assert_write_mode

        if self.subgrid:
            if "sbgfile" not in self.config:
                # apparently no subgrid was read, so set default filename
                self.set_config("sbgfile", "sfincs_subgrid.nc")

            fn = self.get_config("sbgfile", abs_path=True)
            if fn.parts[-1].endswith(".sbg"):
                # write binary file
                self.reggrid.subgrid.write_binary(file_name=fn, mask=self.mask)
            else:
                # write netcdf file
                self.reggrid.subgrid.write(file_name=fn, mask=self.mask)

    def read_geoms(self):
        """Read geometry files and save to `geoms` attribute.
        Known geometry files mentioned in the sfincs.inp configuration file are read,
        including: bnd/src/obs xy(n) files, thd/weir structure files and drn drainage structure files.

        If other geojson files are present in a "gis" subfolder folder, those are read as well.
        """
        self._assert_read_mode
        if self._geoms is None:
            self._geoms = {}  # avoid reading geoms twice

        # read _GEOMS model files
        for gname in self._GEOMS.values():
            if f"{gname}file" in self.config:
                fn = self.get_config(f"{gname}file", abs_path=True)
                if fn is None:
                    continue
                elif not isfile(fn):
                    self.logger.warning(f"{gname}file not found at {fn}")
                    continue
                if gname in ["thd", "weir", "crs"]:
                    struct = utils.read_geoms(fn)
                    gdf = utils.linestring2gdf(struct, crs=self.crs)
                elif gname == "obs":
                    gdf = utils.read_xyn(fn, crs=self.crs)
                elif gname == "drn":
                    gdf = utils.read_drn(fn, crs=self.crs)
                else:
                    gdf = utils.read_xy(fn, crs=self.crs)
                # this seems to be required for new pandas versions
                gdf.set_geometry("geometry", inplace=True)
                self.set_geoms(gdf, name=gname)
        # read additional geojson files from gis directory
        for fn in glob.glob(join(self.root, "gis", "*.geojson")):
            name = basename(fn).replace(".geojson", "")
            gnames = [f[1] for f in self._FORCING_1D.values() if f[1] is not None]
            skip = gnames + list(self._GEOMS.values())
            if name in skip:
                continue
            gdf = hydromt.open_vector(fn, crs=self.crs)
            self.set_geoms(gdf, name=name)

    def write_geoms(self, data_vars: Union[List, str] = None):
        """Write geoms to bnd/src/obs xy files and thd/weir structure files.
        Filenames are based on the `config` attribute.

        If `write_gis` property is True, all geoms are written to geojson
        files in a "gis" subfolder.

        Parameters
        ----------
        data_vars : list of str, optional
            List of data variables to write, by default None (all)

        """
        self._assert_write_mode

        # change precision of coordinates according to crs
        if self.crs.is_geographic:
            fmt = "%.6f"
        else:
            fmt = "%.1f"

        if self.geoms:
            dvars = self._GEOMS.values()
            if data_vars is not None:
                dvars = [name for name in data_vars if name in self._GEOMS.values()]
            self.logger.info("Write geom files")
            for gname, gdf in self.geoms.items():
                if gname in dvars:
                    if f"{gname}file" not in self.config:
                        self.set_config(f"{gname}file", f"sfincs.{gname}")
                    fn = self.get_config(f"{gname}file", abs_path=True)
                    if gname in ["thd", "weir", "crs"]:
                        struct = utils.gdf2linestring(gdf)
                        utils.write_geoms(fn, struct, stype=gname, fmt=fmt)
                    elif gname == "obs":
                        utils.write_xyn(fn, gdf, fmt=fmt)
                    elif gname == "drn":
                        utils.write_drn(fn, gdf, fmt=fmt)
                    else:
                        hydromt.io.write_xy(fn, gdf, fmt="%8.2f")

            # NOTE: all geoms are written to geojson files in a "gis" subfolder
            if self._write_gis:
                self.write_vector(variables=["geoms"])

    def read_forcing(self, data_vars: List = None):
        """Read forcing files and save to `forcing` attribute.
        Known forcing files mentioned in the sfincs.inp configuration file are read,
        including: bzs/dis/precip ascii files and the netampr netcdf file.

        Parameters
        ----------
        data_vars : list of str, optional
            List of data variables to read, by default None (all)
        """
        self._assert_read_mode
        if self._forcing is None:
            self._forcing = {}  # avoid reading forcing twice
        if isinstance(data_vars, str):
            data_vars = list(data_vars)

        # 1D
        dvars_1d = self._FORCING_1D
        if data_vars is not None:
            dvars_1d = [name for name in data_vars if name in dvars_1d]
        tref = utils.parse_datetime(self.config["tref"])
        for name in dvars_1d:
            ts_names, xy_name = self._FORCING_1D[name]
            # read time series
            da_lst = []
            for ts_name in ts_names:
                ts_fn = self.get_config(f"{ts_name}file", abs_path=True)
                if ts_fn is None or not isfile(ts_fn):
                    if ts_fn is not None:
                        self.logger.warning(f"{ts_name}file not found at {ts_fn}")
                    continue
                df = utils.read_timeseries(ts_fn, tref)
                df.index.name = "time"
                if xy_name is not None:
                    df.columns.name = "index"
                    da = xr.DataArray(df, dims=("time", "index"), name=ts_name)
                else:  # spatially uniform forcing
                    da = xr.DataArray(df[df.columns[0]], dims=("time"), name=ts_name)
                da_lst.append(da)
            ds = xr.merge(da_lst[:])
            # read xy
            if xy_name is not None:
                xy_fn = self.get_config(f"{xy_name}file", abs_path=True)
                if xy_fn is None or not isfile(xy_fn):
                    if xy_fn is not None:
                        self.logger.warning(f"{xy_name}file not found at {xy_fn}")
                else:
                    gdf = utils.read_xy(xy_fn, crs=self.crs)
                    # read attribute data from gis files; merge based on index
                    gis_fn = join(self.root, "gis", f"{xy_name}.geojson")
                    if isfile(gis_fn):
                        gdf1 = gpd.read_file(gis_fn)
                        if "index" in gdf1.columns:
                            gdf1 = gdf1.set_index("index")
                        if not np.all(np.isin(gdf.index, gdf1.index)):
                            self.logger.warning(
                                f"Index in {xy_name}file does not match {gis_fn}"
                            )
                        else:
                            for col in gdf1.columns:
                                if col not in gdf.columns:
                                    gdf[col] = gdf1.loc[gdf.index, col]
                    # set locations and attributes as coordinates of dataset
                    ds = ds.assign_coords(index=gdf.index.values)
                    ds = GeoDataset.from_gdf(gdf, ds, index_dim="index")
            # save in self.forcing
            if len(ds) > 1:
                # keep wave forcing together
                self.set_forcing(ds, name=name, split_dataset=False)
            elif len(ds) > 0:
                self.set_forcing(ds, split_dataset=True)

        # 2D NETCDF format
        dvars_2d = self._FORCING_NET
        if data_vars is not None:
            dvars_2d = [name for name in data_vars if name in dvars_2d]
        for name in dvars_2d:
            fname, rename = self._FORCING_NET[name]
            fn = self.get_config(f"{fname}file", abs_path=True)
            if fn is None or not isfile(fn):
                if fn is not None:
                    self.logger.warning(f"{name}file not found at {fn}")
                continue
            elif name in ["netbndbzsbzi", "netsrcdis"]:
                ds = GeoDataset.from_netcdf(fn, crs=self.crs, chunks="auto")
            else:
                ds = xr.open_dataset(fn, chunks="auto")
            rename = {k: v for k, v in rename.items() if k in ds}
            if len(rename) > 0:
                ds = ds.rename(rename).squeeze(drop=True)[list(rename.values())]
                self.set_forcing(ds, split_dataset=True)
            else:
                logger.warning(f"No forcing variables found in {fname}file")

    def write_forcing(self, data_vars: Union[List, str] = None, fmt: str = "%7.2f"):
        """Write forcing to ascii or netcdf (netampr) files.
        Filenames are based on the `config` attribute.

        Parameters
        ----------
        data_vars : list of str, optional
            List of data variables to write, by default None (all)
        fmt : str, optional
            Format string for timeseries data, by default "%7.2f".
        """
        self._assert_write_mode

        # change precision of coordinates according to crs
        if self.crs.is_geographic:
            fmt_xy = "%.6f"
        else:
            fmt_xy = "%.1f"

        if self.forcing:
            self.logger.info("Write forcing files")

            tref = utils.parse_datetime(self.config["tref"])
            # for nc files -> time in minutes since tref
            tref_str = tref.strftime("%Y-%m-%d %H:%M:%S")

            # 1D timeseries + location text files
            dvars_1d = self._FORCING_1D
            if data_vars is not None:
                dvars_1d = [name for name in data_vars if name in self._FORCING_1D]
            for name in dvars_1d:
                ts_names, xy_name = self._FORCING_1D[name]
                if (
                    name in self._FORCING_NET
                    and f"{self._FORCING_NET[name][0]}file" in self.config
                ):
                    continue  # write NC file instead of text files
                # work with wavespectra dataset and bzs/dis dataarray
                if name in self.forcing and isinstance(self.forcing[name], xr.Dataset):
                    ds = self.forcing[name]
                else:
                    ds = self.forcing  # dict
                # write timeseries
                da = None
                for ts_name in ts_names:
                    if ts_name not in ds or ds[ts_name].ndim > 2:
                        continue
                    # parse data to dataframe
                    da = ds[ts_name].transpose("time", ...)
                    df = da.to_pandas()
                    # get filenames from config
                    if f"{ts_name}file" not in self.config:
                        self.set_config(f"{ts_name}file", f"sfincs.{ts_name}")
                    fn = self.get_config(f"{ts_name}file", abs_path=True)
                    # write timeseries
                    utils.write_timeseries(fn, df, tref, fmt=fmt)
                # write xy
                if xy_name and da is not None:
                    # parse data to geodataframe
                    try:
                        gdf = da.vector.to_gdf()
                    except Exception:
                        raise ValueError(f"Locations missing for {name} forcing")
                    # get filenames from config
                    if f"{xy_name}file" not in self.config:
                        self.set_config(f"{xy_name}file", f"sfincs.{xy_name}")
                    fn_xy = self.get_config(f"{xy_name}file", abs_path=True)
                    # write xy
                    hydromt.io.write_xy(fn_xy, gdf, fmt=fmt_xy)
                    if self._write_gis:  # write geojson file to gis folder
                        self.write_vector(variables=f"forcing.{ts_names[0]}")

            # netcdf forcing
            encoding = dict(
                time={"units": f"minutes since {tref_str}", "dtype": "float64"}
            )
            dvars_2d = self._FORCING_NET
            if data_vars is not None:
                dvars_2d = [name for name in data_vars if name in self._FORCING_NET]
            for name in dvars_2d:
                if (
                    name in self._FORCING_1D
                    and f"{self._FORCING_1D[name][1]}file" in self.config
                ):
                    continue  # timeseries + xy file already written
                fname, rename = self._FORCING_NET[name]
                # combine variables and rename to output names
                rename = {v: k for k, v in rename.items() if v in self.forcing}
                if len(rename) == 0:
                    continue
                ds = xr.merge([self.forcing[v] for v in rename.keys()]).rename(rename)
                # get filename from config
                if f"{fname}file" not in self.config:
                    self.set_config(f"{fname}file", f"{name}.nc")
                fn = self.get_config(f"{fname}file", abs_path=True)
                # write 1D timeseries
                if fname in ["netbndbzsbzi", "netsrcdis"]:
                    ds.vector.to_xy().to_netcdf(fn, encoding=encoding)
                    if self._write_gis:  # write geojson file to gis folder
                        self.write_vector(variables=f"forcing.{list(rename.keys())[0]}")
                # write 2D gridded timeseries
                else:
                    ds.to_netcdf(fn, encoding=encoding)

    def read_states(self):
        """Read waterlevel state (zsini) from binary file and save to `states` attribute.
        The inifile if mentioned in the sfincs.inp configuration file is read.

        """
        self._assert_read_mode

        # read index file
        # TODO make reggrid a property where we trigger the initialization of reggrid
        if self.reggrid is None:
            self.update_grid_from_config()
        if self.reggrid is not None:
            ind_fn = self.get_config("indexfile", fallback="sfincs.ind", abs_path=True)
            if "msk" in self.grid:  # triggers reading grid if empty and in read mode
                ind = self.reggrid.ind(self.grid["msk"].values)
            elif isfile(ind_fn):
                ind = self.reggrid.read_ind(ind_fn=ind_fn)
            else:
                raise IOError(f"indexfile {ind_fn} does not exist")
            if "inifile" in self.config:
                fn = self.get_config("inifile", abs_path=True)
                if not isfile(fn):
                    self.logger.warning("inifile not found at {fn}")
                    return
                zsini = self.reggrid.read_map(
                    fn, ind, dtype="f4", mv=-9999.0, name="zsini"
                )

                if self.crs is not None:
                    zsini.raster.set_crs(self.crs)
                self.set_states(zsini, "zsini")

    def write_states(self):
        """Write waterlevel state (zsini) to binary map file.
        The filenames is based on the `config` attribute.
        """
        self._assert_write_mode

        name = "zsini"

        if name not in self.states:
            self.logger.warning(f"{name} not in states, skipping")
            return

        if self.reggrid and "msk" in self.grid:
            # make sure orientation is S->N
            ds_out = self.grid
            if ds_out.raster.res[1] < 0:
                ds_out = ds_out.raster.flipud()
            mask = ds_out["msk"].values

            self.logger.debug("Write binary map indices based on mask.")
            # write index file
            ind_fn = self.get_config("indexfile", abs_path=True)
            self.reggrid.write_ind(ind_fn=ind_fn, mask=mask)

            if "inifile" not in self.config:
                self.set_config("inifile", f"sfincs.{name}")
            fn = self.get_config("inifile", abs_path=True)
            da = self.states[name]
            if da.raster.res[1] < 0:
                da = da.raster.flipud()

            self.logger.debug("Write binary water level state inifile")
            self.reggrid.write_map(
                map_fn=fn,
                data=da.values,
                mask=mask,
                dtype="f4",
            )

        if self._write_gis:
            self.write_raster("states")

    def read_results(
        self,
        chunksize=100,
        drop=["crs", "sfincsgrid"],
        fn_map="sfincs_map.nc",
        fn_his="sfincs_his.nc",
        **kwargs,
    ):
        """Read results from sfincs_map.nc and sfincs_his.nc and save to the `results` attribute.
        The staggered nc file format is translated into hydromt.RasterDataArray formats.
        Additionally, hmax is computed from zsmax and zb if present.

        Parameters
        ----------
        chunksize: int, optional
            chunk size along time dimension, by default 100
        drop: list, optional
            list of variables to drop, by default ["crs", "sfincsgrid"]
        fn_map: str, optional
            filename of sfincs_map.nc, by default "sfincs_map.nc"
        fn_his: str, optional
            filename of sfincs_his.nc, by default "sfincs_his.nc"
        """
        if not isabs(fn_map):
            fn_map = join(self.root, fn_map)
        if isfile(fn_map):
            ds_face, ds_edge = utils.read_sfincs_map_results(
                fn_map,
                ds_like=self.grid,  # TODO: fix for quadtree
                drop=drop,
                logger=self.logger,
                **kwargs,
            )
            # save as dict of DataArray
            self.set_results(ds_face, split_dataset=True)
            self.set_results(ds_edge, split_dataset=True)

        if not isabs(fn_his):
            fn_his = join(self.root, fn_his)
        if isfile(fn_his):
            ds_his = utils.read_sfincs_his_results(
                fn_his, crs=self.crs, chunksize=chunksize
            )
            # drop double vars (map files has priority)
            drop_vars = [v for v in ds_his.data_vars if v in self.results or v in drop]
            ds_his = ds_his.drop_vars(drop_vars)
            self.set_results(ds_his, split_dataset=True)

    def write_raster(
        self,
        variables=["grid", "states", "results.hmax"],
        root=None,
        driver="GTiff",
        compress="deflate",
        **kwargs,
    ):
        """Write model 2D raster variables to geotiff files.

        NOTE: these files are not used by the model by just saved for visualization/
        analysis purposes.

        Parameters
        ----------
        variables: str, list, optional
            Model variables are a combination of attribute and layer (optional) using <attribute>.<layer> syntax.
            Known ratster attributes are ["grid", "states", "results"].
            Different variables can be combined in a list.
            By default, variables is ["grid", "states", "results.hmax"]
        root: Path, str, optional
            The output folder path. If None it defaults to the <model_root>/gis folder (Default)
        kwargs:
            Key-word arguments passed to hydromt.RasterDataset.to_raster(driver='GTiff', compress='lzw').
        """

        # check variables
        if isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list):
            raise ValueError(f'"variables" should be a list, not {type(list)}.')
        # check root
        if root is None:
            root = join(self.root, "gis")
        if not os.path.isdir(root):
            os.makedirs(root)
        # save to file
        for var in variables:
            vsplit = var.split(".")
            attr = vsplit[0]
            obj = getattr(self, f"_{attr}")
            if obj is None or len(obj) == 0:
                continue  # empty
            self.logger.info(f"Write raster file(s) for {var} to 'gis' subfolder")
            layers = vsplit[1:] if len(vsplit) >= 2 else list(obj.keys())
            for layer in layers:
                if layer not in obj:
                    self.logger.warning(f"Variable {attr}.{layer} not found: skipping.")
                    continue
                da = obj[layer]
                if len(da.dims) != 2:
                    # try to reduce to 2D by taking maximum over time dimension
                    if "time" in da.dims:
                        da = da.max("time")
                    elif "timemax" in da.dims:
                        da = da.max("timemax")
                    # if still not 2D, skip
                    if len(da.dims) != 2:
                        self.logger.warning(
                            f"Variable {attr}.{layer} has more than 2 dimensions: skipping."
                        )
                        continue
                # If the raster type is float, set nodata to np.nan
                if da.dtype == "float32" or da.dtype == "float64":
                    da.raster.set_nodata(np.nan)
                # only write active cells to gis files
                da = da.where(self.mask > 0, da.raster.nodata).raster.mask_nodata()
                if da.raster.res[1] > 0:  # make sure orientation is N->S
                    da = da.raster.flipud()
                da.raster.to_raster(
                    join(root, f"{layer}.tif"),
                    driver=driver,
                    compress=compress,
                    **kwargs,
                )

    def write_vector(
        self,
        variables=["geoms", "forcing.bzs", "forcing.dis"],
        root=None,
        gdf=None,
        **kwargs,
    ):
        """Write model vector (geoms) variables to geojson files.

        NOTE: these files are not used by the model by just saved for visualization/
        analysis purposes.

        Parameters
        ----------
        variables: str, list, optional
            geoms variables. By default all geoms are saved.
        root: Path, str, optional
            The output folder path. If None it defaults to the <model_root>/gis folder (Default)
        kwargs:
            Key-word arguments passed to geopandas.GeoDataFrame.to_file(driver='GeoJSON').
        """
        kwargs.update(driver="GeoJSON")  # fixed
        # check variables
        if isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list):
            raise ValueError(f'"variables" should be a list, not {type(list)}.')
        # check root
        if root is None:
            root = join(self.root, "gis")
        if not os.path.isdir(root):
            os.makedirs(root)
        # save to file
        for var in variables:
            vsplit = var.split(".")
            attr = vsplit[0]
            obj = getattr(self, f"_{attr}")
            if obj is None or len(obj) == 0:
                continue  # empty
            self.logger.info(f"Write vector file(s) for {var} to 'gis' subfolder")
            names = vsplit[1:] if len(vsplit) >= 2 else list(obj.keys())
            for name in names:
                if name not in obj:
                    self.logger.warning(f"Variable {attr}.{name} not found: skipping.")
                    continue
                if isinstance(obj[name], gpd.GeoDataFrame):
                    gdf = obj[name]
                else:
                    try:
                        gdf = obj[name].vector.to_gdf()
                        # xy name -> difficult!
                        name = [
                            v[-1] for v in self._FORCING_1D.values() if name in v[0]
                        ][0]
                    except:
                        self.logger.debug(
                            f"Variable {attr}.{name} could not be written to vector file."
                        )
                        pass
                gdf.to_file(join(root, f"{name}.geojson"), **kwargs)

    ## model configuration

    def read_config(self, config_fn: str = None, epsg: int = None) -> None:
        """Parse config from SFINCS input file.
        If in write-only mode the config is initialized with default settings
        unless a path to a template config file is provided.

        Parameters
        ----------
        config_fn: str
            Filename of config file, by default None.
        epsg: int
            EPSG code of the model CRS. Only used if missing in the SFINCS input file,
            by default None.
        """
        inp = SfincsInput()  # initialize with defaults
        if config_fn is not None or self._read:
            if config_fn is None:  # read from default location
                config_fn = self._config_fn
            if not isabs(config_fn) and self._root:  # read from model root
                config_fn = abspath(join(self.root, config_fn))
            if not isfile(config_fn):
                raise IOError(f"SFINCS input file not found {config_fn}")
            # read inp file
            inp.read(inp_fn=config_fn)
        # overwrite / initialize config attribute
        self._config = inp.to_dict()
        if epsg is not None and "epsg" not in self.config:
            self.set_config("epsg", int(epsg))
        # update grid properties based on sfincs.inp
        self.update_grid_from_config()

    def write_config(self, config_fn: str = "sfincs.inp"):
        """Write config to <root/config_fn>"""
        self._assert_write_mode
        if not isabs(config_fn) and self._root:
            config_fn = join(self.root, config_fn)

        inp = SfincsInput.from_dict(self.config)
        inp.write(inp_fn=abspath(config_fn))

    def update_spatial_attrs(self):
        """Update geospatial `config` (sfincs.inp) attributes based on grid"""
        dx, dy = self.res
        # TODO check self.bounds with rotation!! origin not necessary equal to total_bounds
        west, south, _, _ = self.bounds
        if self.crs is not None:
            self.set_config("epsg", self.crs.to_epsg())
        self.set_config("mmax", self.width)
        self.set_config("nmax", self.height)
        self.set_config("dx", dx)
        self.set_config("dy", abs(dy))  # dy is always positive (orientation is S -> N)
        self.set_config("x0", west)
        self.set_config("y0", south)

    def update_grid_from_config(self):
        """Update grid properties based on `config` (sfincs.inp) attributes"""
        self.grid_type = (
            "quadtree" if self.config.get("qtrfile") is not None else "regular"
        )
        if self.grid_type == "regular":
            self.reggrid = RegularGrid(
                x0=self.config.get("x0"),
                y0=self.config.get("y0"),
                dx=self.config.get("dx"),
                dy=self.config.get("dy"),
                nmax=self.config.get("nmax"),
                mmax=self.config.get("mmax"),
                rotation=self.config.get("rotation", 0),
                epsg=self.config.get("epsg"),
            )
        else:
            raise not NotImplementedError("Quadtree grid not implemented yet")
            # self.quadtree = QuadtreeGrid()

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
                        bbox=self.mask.raster.transform_bounds(4326),
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
                    bbox=self.mask.raster.transform_bounds(4326),
                    buffer=10,
                )
                dd.update({"offset": da_offset})

            # read geodataframes describing valid areas
            if "mask" in dataset:
                gdf_valid = self.data_catalog.get_geodataframe(
                    dataset.get("mask"),
                    bbox=self.mask.raster.transform_bounds(4326),
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
                    bbox=self.mask.raster.transform_bounds(4326),
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
                        f"Manning roughness 'reclass_table' csv file must be provided"
                    )
                da_lulc = self.data_catalog.get_rasterdataset(
                    lulc,
                    bbox=self.mask.raster.transform_bounds(4326),
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
                    bbox=self.mask.raster.transform_bounds(4326),
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
                        geom=self.mask.raster.box,
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
                    geom=self.mask.raster.box,
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
                    geom=self.mask.raster.box,
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
