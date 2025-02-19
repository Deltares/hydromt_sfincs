"""RegularGrid class for SFINCS."""

import logging
import math
import glob
import os
from os.path import abspath, basename, dirname, isabs, isfile, join
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from pyflwdir.regions import region_area
from pyproj import CRS, Transformer
from scipy import ndimage
from shapely.geometry import LineString

from hydromt.model.components import GridComponent
from hydromt.model.processes.grid import create_grid_from_region

from hydromt_sfincs import workflows
from hydromt_sfincs.subgrid import SubgridTableRegular
from hydromt_sfincs.workflows.tiling import int2png, tile_window

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(__name__)

_MAPS = ["msk", "dep", "scs", "manning", "qinf", "smax", "seff", "ks", "vol"]


class RegularGrid(GridComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        super().__init__(
            model=model,
            filename="sfincs.nc",
            region_filename="region.geojson",
        )

        # set spatial attributes
        self.update_grid_from_config()

    @property
    def transform(self):
        """Return the affine transform of the regular grid."""
        transform = (
            Affine.translation(self.x0, self.y0)
            * Affine.rotation(self.rotation)
            * Affine.scale(self.dx, self.dy)
        )
        return transform

    @property
    def coordinates(self, x_dim="x", y_dim="y"):
        """Return the coordinates of the cell-centers the regular grid."""
        if self.transform.b == 0:
            x_coords, _ = self.transform * (
                np.arange(self.mmax) + 0.5,
                np.zeros(self.mmax) + 0.5,
            )
            _, y_coords = self.transform * (
                np.zeros(self.nmax) + 0.5,
                np.arange(self.nmax) + 0.5,
            )
            coords = {
                y_dim: (y_dim, y_coords),
                x_dim: (x_dim, x_coords),
            }
        else:
            x_coords, y_coords = (
                self.transform
                * self.transform.translation(0.5, 0.5)
                * np.meshgrid(np.arange(self.mmax), np.arange(self.nmax))
            )
            coords = {
                "yc": ((y_dim, x_dim), y_coords),
                "xc": ((y_dim, x_dim), x_coords),
            }
        return coords

    @property
    def edges(self):
        """Return the coordinates of the cell-edges the regular grid."""
        x_edges, y_edges = (
            self.transform
            * self.transform.translation(0, 0)
            * np.meshgrid(np.arange(self.mmax + 1), np.arange(self.nmax + 1))
        )
        return x_edges, y_edges

    @property
    def empty_mask(self) -> xr.DataArray:
        """Return mask with only inactive cells"""
        da_mask = xr.DataArray(
            name="msk",
            data=np.zeros((self.nmax, self.mmax), dtype=np.uint8),
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": 0},
        )
        da_mask.raster.set_crs(self.crs)
        return da_mask

    @property
    def crs(self) -> CRS:
        """Return the coordinate reference system of the regular grid."""
        if self.epsg is not None:
            return CRS.from_epsg(self.epsg)
        elif self.data.raster.crs is not None:
            return self.data.raster.crs
        else:
            raise ValueError("No CRS defined for the regular grid.")

    def read(self, data_vars: Union[List, str] = None) -> None:
        """Read SFINCS binary grid files and save to `data` attribute.
        Filenames are taken from the `model.config` attribute (i.e. input file).

        Parameters
        ----------
        data_vars : Union[List, str], optional
            List of data variables to read, by default None (all)
        """
        # check if in read mode and initialize grid
        self.root._assert_read_mode()
        self._initialize_grid(skip_read=True)

        # first update grid from config
        self.update_grid_from_config()

        # now read in the actual files
        da_lst = []
        if data_vars is None:
            data_vars = _MAPS
        elif isinstance(data_vars, str):
            data_vars = list(data_vars)

        # read index file
        ind_fn = self.model.config.get(
            "indexfile", fallback="sfincs.ind", abs_path=True
        )
        if not isfile(ind_fn):
            raise IOError(f".ind path {ind_fn} does not exist")

        dtypes = {"msk": "u1"}
        mvs = {"msk": 0}
        ind = self.read_ind(ind_fn=ind_fn)

        for name in data_vars:
            if f"{name}file" in self.model.config:
                fn = self.model.config.get(
                    f"{name}file", fallback=f"sfincs.{name}", abs_path=True
                )
                if not isfile(fn):
                    logger.warning(f"{name}file not found at {fn}")
                    continue
                dtype = dtypes.get(name, "f4")
                mv = mvs.get(name, -9999.0)
                da = self.read_map(fn, ind, dtype, mv, name=name)
                da_lst.append(da)
        ds = xr.merge(da_lst)
        epsg = self.model.config.get("epsg", None)
        if epsg is not None:
            ds.raster.set_crs(epsg)
        self.set(ds)

        # # TODO - fix this properly; but to create overlays in GUIs,
        # # we always convert regular grids to a UgridDataArray
        # self.quadtree = QuadtreeGrid(logger=logger)
        # if self.config.get("rotation", 0) != 0:  # This is a rotated regular grid
        #     self.quadtree.data = UgridDataArray.from_structured(
        #         self.mask, "xc", "yc"
        #     )
        # else:
        #     self.quadtree.data = UgridDataArray.from_structured(self.mask)
        # self.quadtree.data.grid.set_crs(self.crs)

        # keep some metadata maps from gis directory

        # fns = glob.glob(join(self.root, "gis", "*.tif"))
        # fns = [
        #     fn
        #     for fn in fns
        #     if basename(fn).split(".")[0] not in self.grid.data_vars
        # ]
        # if fns:
        #     ds = hydromt.open_mfraster(fns).load()
        #     self.set_grid(ds)
        #     ds.close()

    def write(self, data_vars: Union[List, str] = None):
        """Write SFINCS grid to binary files including map index file.
        Filenames are taken from the `config` attribute (i.e. input file).

        If `write_gis` property is True, all grid variables are written to geotiff
        files in a "gis" subfolder.

        Parameters
        ----------
        data_vars : Union[List, str], optional
            List of data variables to write, by default None (all)
        """
        self.root._assert_write_mode

        dtypes = {"msk": "u1"}  # default to f4
        if len(self.data.data_vars) > 0 and "msk" in self.data:
            # make sure orientation is S->N
            ds_out = self.data
            if ds_out.raster.res[1] < 0:
                ds_out = ds_out.raster.flipud()
            mask = ds_out["msk"].values

            logger.debug("Write binary map indices based on mask.")
            if self.model.config.get("indexfile") is None:
                self.model.config.set("indexfile", "sfincs.ind")
            self.write_ind(
                ind_fn=self.model.config.get("indexfile", abs_path=True), mask=mask
            )

            if data_vars is None:  # write all maps
                data_vars = [v for v in self._MAPS if v in ds_out]
            elif isinstance(data_vars, str):
                data_vars = list(data_vars)
            logger.debug(f"Write binary map files: {data_vars}.")
            for name in data_vars:
                if self.model.config.get(f"{name}file") is None:
                    self.config.set(f"{name}file", f"sfincs.{name}")
                # do not write depfile if subgrid is used
                # if (name == "dep" or name == "manning") and self.subgrid:
                #     continue
                self.write_map(
                    map_fn=self.model.config.get(f"{name}file", abs_path=True),
                    data=ds_out[name].values,
                    mask=mask,
                    dtype=dtypes.get(name, "f4"),
                )

        # if self._write_gis:
        #     self.write_raster("grid")

    def create(
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
            epsg-code of the coordinate reference system
        """

        # update the grid attributes in the model config
        self.model.config.update(
            {
                "x0": x0,
                "y0": y0,
                "dx": dx,
                "dy": dy,
                "nmax": nmax,
                "mmax": mmax,
                "rotation": rotation,
                "epsg": epsg,
            }
        )
        self.update_grid_from_config()

        # set an empty mask to data
        self.set(self.empty_mask)

    def create_from_region(
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
        hydromt.model.processes.create_grid_from_region
        """

        da = create_grid_from_region(
            region=region,
            data_catalog=self.model.data_catalog,
            res=res,
            crs=crs,
            region_crs=4326,
            rotated=rotated,
            hydrography_path=hydrography_fn,
            basin_index_path=basin_index_fn,
            add_mask=False,
            align=align,
            dec_origin=dec_origin,
            dec_rotation=dec_rotation,
        )

        # add the grid to the model
        self.set(da)
        # update the grid attributes in the model config
        self.update_config_from_grid()

    # %%   Original HydroMT-SFINCS setup_ functions:
    # setup_grid
    # setup_grid_from_region
    #
    # setup_dep
    #
    # setup_mask_active
    # setup_mask_bounds

    # %% core HydroMT-SFINCS functions:
    # _initialize
    #
    # GRID:
    #   read_grid
    #   write_grid
    #   create
    #       - create_grid (model.grid.create?)
    #       - create_grid_from_region (model.grid.create_from_region)
    #
    # DEP:
    #   read_dep
    #   write_dep
    #   create_dep
    #
    # MASK:
    #   read_msk
    #   write_msk
    #   create_msk
    #   create_msk_bounds
    #
    # supporting HydroMT-SFINCS functions:
    # - read_ind
    # - read_map
    # - write_ind
    # - write_map
    # - ind
    # - to_vector_lines

    # %% GRID:
    def read_grid(
        self,
    ):
        # uses read_ind()

        return

    def write_grid(
        self,
    ):
        # uses write_ind()

        return

    def create_grid(
        self,
    ):
        return

    def create_grid_from_region(
        self,
    ):
        return

    # %% DEP:
    def read_dep(
        self,
    ):
        # uses read_map()

        return

    def write_dep(
        self,
    ):
        # uses write_map()

        return

    def create_dep(
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
        if not self.mask.raster.crs.is_geographic:
            res = np.abs(self.mask.raster.res[0])
        else:
            res = np.abs(self.mask.raster.res[0]) * 111111.0

        datasets_dep = self._parse_datasets_dep(datasets_dep, res=res)

        da_dep = workflows.merge_multi_dataarrays(
            da_list=datasets_dep,
            da_like=self.mask,
            buffer_cells=buffer_cells,
            interp_method=interp_method,
            logger=logger,
        )

        # check if no nan data is present in the bed levels
        nmissing = int(np.sum(np.isnan(da_dep.values)))
        if nmissing > 0:
            logger.warning(f"Interpolate elevation at {nmissing} cells")
            da_dep = da_dep.raster.interpolate_na(method="rio_idw", extrapolate=True)

        self.set_grid(da_dep, name="dep")
        # FIXME this shouldn't be necessary, since da_dep should already have a crs
        if self.crs is not None and self.grid.raster.crs is None:
            self.grid.set_crs(self.crs)

        if "depfile" not in self.config:
            self.config.update({"depfile": "sfincs.dep"})

    ## MASK

    def read_msk(
        self,
    ):
        # uses read_map()

        return

    def write_msk(
        self,
    ):
        # uses write_map()

        return

    def create_mask_active(
        # def create_mask(
        self,
        da_mask: xr.DataArray = None,
        da_dep: xr.DataArray = None,
        gdf_mask: gpd.GeoDataFrame = None,
        gdf_include: gpd.GeoDataFrame = None,
        gdf_exclude: gpd.GeoDataFrame = None,
        zmin: float = None,
        zmax: float = None,
        fill_area: float = 10,
        drop_area: float = 0,
        connectivity: int = 8,
        all_touched: bool = True,
        reset_mask: bool = True,
        logger: logging.Logger = logger,
    ) -> xr.DataArray:
        """Create an integer mask with inactive (msk=0) and active (msk=1) cells, optionally bounded
        by several criteria.

        Parameters
        ----------
        da_mask: xarray.DataArray, optional
            Mask with 0) Inactive and 1) active cells to initialize with.
            If not provided, mask is initialized empty.
        da_dep: xarray.DataArray, optional
            Elevation data to use for active mask.
        gdf_mask: geopandas.GeoDataFrame, optional
            Geometry with area to initiliaze active mask with; proceding arguments can be used to include/exclude cells
            If not given, existing mask (if present) is used, else mask is initialized empty.
        gdf_include, gdf_exclude: geopandas.GeoDataFrame, optional
            Geometries with areas to include/exclude from the active model cells.
            Note that include (second last) and exclude (last) areas are processed after other critera,
            i.e. `zmin`, `zmax` and `drop_area`, and thus overrule these criteria for active model cells.
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for active model cells.
        fill_area : float, optional
            Maximum area [km2] of contiguous cells below `zmin` or above `zmax` but surrounded
            by cells within the valid elevation range to be kept as active cells, by default 10 km2.
        drop_area : float, optional
            Maximum area [km2] of contiguous cells to be set as inactive cells, by default 0 km2.
        connectivity: {4, 8}
            The connectivity used to define contiguous cells, if 4 only horizontal and vertical
            connections are used, if 8 (default) also diagonal connections.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.
        reset_mask: bool, optional
            If True (default), reset existing mask layer. If False updating existing mask.

        Returns
        -------
        da_mask: xr.DataArray
            Integer SFINCS model mask with inactive (msk=0), active (msk=1) cells
        """

        da_mask0 = None
        if not reset_mask and da_mask is not None:
            # use current active mask
            da_mask0 = da_mask > 0
        elif gdf_mask is not None:
            # start with active mask within provided region
            da_mask0 = (
                self.empty_mask.raster.geometry_mask(gdf_mask, all_touched=all_touched)
                > 0
            )
        # always intiliaze an inactive mask
        da_mask = self.empty_mask > 0

        latlon = self.crs.is_geographic

        if da_dep is None and (zmin is not None or zmax is not None):
            raise ValueError("da_dep required in combination with zmin / zmax")
        elif da_dep is not None and not da_dep.raster.identical_grid(da_mask):
            raise ValueError("da_dep does not match regular grid")

        s = None if connectivity == 4 else np.ones((3, 3), int)
        if zmin is not None or zmax is not None:
            _msk = da_dep != da_dep.raster.nodata
            if zmin is not None:
                _msk = np.logical_and(_msk, da_dep >= zmin)
            if zmax is not None:
                _msk = np.logical_and(_msk, da_dep <= zmax)
            if da_mask0 is not None:
                # if mask was provided; keep active mask only within valid elevations
                da_mask = np.logical_and(da_mask0, _msk)
            else:
                # no mask provided; set mask to valid elevations
                da_mask = _msk
        elif zmin is None and zmax is None and da_mask0 is not None:
            # in case a mask/region was provided, but you didn't want to update the mask based on elevation
            # just continue with the provided mask
            da_mask = da_mask0

        if fill_area > 0:
            _msk1 = np.logical_xor(
                da_mask, ndimage.binary_fill_holes(da_mask, structure=s)
            )
            regions = ndimage.measurements.label(_msk1, structure=s)[0]
            # TODO check if region_area works for rotated grids!
            lbls, areas = region_area(regions, self.transform, latlon)
            n = int(sum(areas / 1e6 < fill_area))
            logger.info(f"{n} gaps outside valid elevation range < {fill_area} km2.")
            da_mask = np.logical_or(
                da_mask, np.isin(regions, lbls[areas / 1e6 < fill_area])
            )
        if drop_area > 0:
            regions = ndimage.measurements.label(da_mask.values, structure=s)[0]
            lbls, areas = region_area(regions, self.transform, latlon)
            _msk = np.isin(regions, lbls[areas / 1e6 >= drop_area])
            n = int(sum(areas / 1e6 < drop_area))
            logger.info(f"{n} regions < {drop_area} km2 dropped.")
            da_mask = np.logical_and(da_mask, _msk)

        if gdf_include is not None:
            try:
                _msk = da_mask.raster.geometry_mask(
                    gdf_include, all_touched=all_touched
                )
                da_mask = np.logical_or(da_mask, _msk)  # NOTE logical OR statement
            except:
                logger.debug(f"No mask cells found within include polygon!")
        if gdf_exclude is not None:
            try:
                _msk = da_mask.raster.geometry_mask(
                    gdf_exclude, all_touched=all_touched
                )
                da_mask = np.logical_and(da_mask, ~_msk)
            except:
                logger.debug(f"No mask cells found within exclude polygon!")

        # update sfincs mask name, nodata value and crs
        da_mask = da_mask.where(da_mask, 0).astype(np.uint8).rename("mask")
        da_mask.raster.set_nodata(0)
        da_mask.raster.set_crs(self.crs)

        return da_mask

    def create_mask_bounds(
        self,
        da_mask: xr.DataArray,
        btype: str = "waterlevel",
        gdf_include: Optional[gpd.GeoDataFrame] = None,
        gdf_exclude: Optional[gpd.GeoDataFrame] = None,
        da_dep: xr.DataArray = None,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        connectivity: int = 8,
        all_touched=False,
        reset_bounds=False,
        logger: logging.Logger = logger,
    ) -> xr.DataArray:
        """Returns an integer SFINCS model mask with inactive (msk=0), active (msk=1), and waterlevel boundary (msk=2)
            and outflow boundary (msk=3) cells.  Boundary cells are defined by cells at the edge of active model domain.

        Parameters
        ----------
        da_mask: xarray.DataArray
            SFINCS model mask with inactive (msk=0) active (msk>0) cells.
        btype: {'waterlevel', 'outflow'}
            Boundary type
        gdf_include, gdf_exclude: geopandas.GeoDataFrame
            Geometries with areas to include/exclude from the model boundary.
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for boundary cells.
            Note that when include and exclude areas are used, the elevation range is only applied
            on cells within the include area and outside the exclude area.
        connectivity: {4, 8}
            The connectivity used to detect the model edge, if 4 only horizontal and vertical
            connections are used, if 8 (default) also diagonal connections.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.
        reset_bounds: bool, optional
            If True, reset existing boundary cells of the selected boundary
            type (`btype`) before setting new boundary cells, by default False.

        Returns
        -------
        da_mask: xr.DataArray
            Integer SFINCS model mask with inactive (msk=0), active (msk=1), and waterlevel boundary (msk=2)
            and outflow boundary (msk=3) cells

        """
        if not da_mask.raster.identical_grid(self.empty_mask):
            raise ValueError("da_mask does not match regular grid")
        latlon = self.crs.is_geographic

        if da_dep is None and (zmin is not None or zmax is not None):
            raise ValueError("da_dep required in combination with zmin / zmax")
        elif da_dep is not None and not da_dep.raster.identical_grid(da_mask):
            raise ValueError("da_dep does not match regular grid")

        btype = btype.lower()
        bvalues = {"waterlevel": 2, "outflow": 3}
        if btype not in bvalues:
            raise ValueError('btype must be one of "waterlevel", "outflow"')
        bvalue = bvalues[btype]

        if reset_bounds:  # reset existing boundary cells
            logger.debug(f"{btype} (mask={bvalue:d}) boundary cells reset.")
            da_mask = da_mask.where(da_mask != np.uint8(bvalue), np.uint8(1))
            if (
                zmin is None
                and zmax is None
                and gdf_include is None
                and gdf_exclude is None
            ):
                return da_mask

        s = None if connectivity == 4 else np.ones((3, 3), int)
        bounds0 = np.logical_xor(
            da_mask > 0, ndimage.binary_erosion(da_mask > 0, structure=s)
        )
        bounds = bounds0.copy()

        if zmin is not None:
            bounds = np.logical_and(bounds, da_dep >= zmin)
        if zmax is not None:
            bounds = np.logical_and(bounds, da_dep <= zmax)
        if gdf_include is not None:
            da_include = da_mask.raster.geometry_mask(
                gdf_include, all_touched=all_touched
            )
            # bounds = np.logical_or(bounds, np.logical_and(bounds0, da_include))
            bounds = np.logical_and(bounds, da_include)
        if gdf_exclude is not None:
            da_exclude = da_mask.raster.geometry_mask(
                gdf_exclude, all_touched=all_touched
            )
            bounds = np.logical_and(bounds, ~da_exclude)

        # avoid any msk3 cells neighboring msk2 cells
        if bvalue == 3 and np.any(da_mask == 2):
            msk2_dilated = ndimage.binary_dilation(
                (da_mask == 2).values,
                structure=np.ones((3, 3)),
                iterations=1,  # minimal one cell distance between msk2 and msk3 cells
            )
            bounds = bounds.where(~msk2_dilated, False)

        ncells = np.count_nonzero(bounds.values)
        if ncells > 0:
            da_mask = da_mask.where(~bounds, np.uint8(bvalue))

        return da_mask

    # %% supporting HydroMT-SFINCS functions:
    # other:
    # - ind
    # - read_ind
    # - read_map
    # - write_ind
    # - write_map
    # - to_vector_lines

    def ind(self, mask: np.ndarray) -> np.ndarray:
        """Return indices of active cells in mask."""
        assert mask.shape == (self.nmax, self.mmax)
        ind = np.where(mask.ravel(order="F"))[0]
        return ind

    def read_ind(
        self,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> np.ndarray:
        """Read indices of active cells in mask from binary file."""
        _ind = np.fromfile(ind_fn, dtype="u4")
        ind = _ind[1:] - 1  # convert to zero based index
        assert _ind[0] == ind.size

        return ind

    def read_map(
        self,
        map_fn: Union[str, Path],
        ind: np.ndarray,
        dtype: Union[str, np.dtype] = "f4",
        mv: float = -9999.0,
        name: str = None,
    ) -> xr.DataArray:
        """Read one of the grid variables of the SFINCS model map from a binary file."""

        data = np.full((self.mmax, self.nmax), mv, dtype=dtype)
        data.flat[ind] = np.fromfile(map_fn, dtype=dtype)
        data = data.transpose()

        da = xr.DataArray(
            name=map_fn.split(".")[-1] if name is None else name,
            data=data,
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": mv},
        )
        return da

    def write_ind(
        self,
        mask: np.ndarray,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> None:
        """Write indices of active cells in mask to binary file."""
        assert mask.shape == (self.nmax, self.mmax)
        # Add 1 because indices in SFINCS start with 1, not 0
        ind = self.ind(mask)
        indices_ = np.array(np.hstack([np.array(len(ind)), ind + 1]), dtype="u4")
        indices_.tofile(ind_fn)

    def write_map(
        self,
        map_fn: Union[str, Path],
        data: np.ndarray,
        mask: np.ndarray,
        dtype: Union[str, np.dtype] = "f4",
    ) -> None:
        """Write one of the grid variables of the SFINCS model map to a binary file."""

        data_out = np.asarray(data.transpose()[mask.transpose() > 0], dtype=dtype)
        data_out.tofile(map_fn)

    def update_grid_from_config(self):
        """Update grid properties based on `config` (sfincs.inp) attributes"""

        # assert model.config exists
        if not hasattr(self.model, "config"):
            raise AttributeError("Model has no config attribute")

        self.x0 = self.model.config.get("x0")
        self.y0 = self.model.config.get("y0")
        self.dx = self.model.config.get("dx")
        self.dy = self.model.config.get("dy")
        self.nmax = self.model.config.get("nmax")
        self.mmax = self.model.config.get("mmax")
        self.rotation = self.model.config.get("rotation", 0)
        self.epsg = self.model.config.get("epsg", None)

    def update_config_from_grid(self):
        """Update `config` (sfincs.inp) attributes based on grid properties"""

        # derive grid properties from grid
        self.nmax, self.mmax = self.data.raster.shape
        self.dx, self.dy = self.data.raster.res
        self.x0, self.y0 = self.data.raster.origin
        self.rotation = self.data.raster.rotation
        self.epsg = self.crs.to_epsg()

        # update the grid properties in the config
        self.model.config.update(
            {
                "x0": self.x0,
                "y0": self.y0,
                "dx": self.dx,
                "dy": self.dy,
                "nmax": self.nmax,
                "mmax": self.mmax,
                "rotation": self.rotation,
                "epsg": self.epsg,
            }
        )

    def to_vector_lines(self):
        """Return a geopandas GeoDataFrame with a geometry for each grid line."""

        x, y = self.edges

        # create vertical lines
        vertical_lines = []
        for i in range(self.nmax + 1):
            line = LineString([(x[i, 0], y[i, 0]), (x[i, -1], y[i, -1])])
            vertical_lines.append(line)

        # create horizontal lines
        horizontal_lines = []
        for j in range(self.mmax + 1):
            line = LineString([(x[0, j], y[0, j]), (x[-1, j], y[-1, j])])
            horizontal_lines.append(line)

        # combine lines into a single list
        grid_lines = vertical_lines + horizontal_lines

        return gpd.GeoDataFrame(geometry=grid_lines, crs=self.crs)

    # %% DDB GUI focused additional functions:
    # create_index_tiles > FIXME - TL: still needed?
    # map_overlay
    # snap_to_grid
    # _get_datashader_dataframe

    # TODO - missing as in cht_sfincs:
    # Many...

    def create_index_tiles(
        self,
        root: Union[str, Path],
        region: gpd.GeoDataFrame,
        zoom_range: Union[int, List[int]] = [0, 13],
        fmt: str = "bin",
        logger: logging.Logger = logger,
    ):
        """Create index tiles for a region. Index tiles are used to quickly map webmercator tiles to the corresponding SFINCS cell.

        Parameters
        ----------
        region : gpd.GeoDataFrame
            GeoDataFrame containing the region of interest
        root : Union[str, Path]
            Directory where index tiles are stored
        zoom_range : Union[int, List[int]], optional
            Range of zoom levels for which tiles are created, by default [0,13]
        fmt : str, optional
            Format of index tiles, either "bin" (binary, default) or "png"
        """

        index_path = os.path.join(root, "indices")
        npix = 256

        # for binary format, use .dat extension
        if fmt == "bin":
            extension = "dat"
        # for net, tif and png extension and format are the same
        else:
            extension = fmt

        # if only one zoom level is specified, create tiles up to that zoom level (inclusive)
        if isinstance(zoom_range, int):
            zoom_range = [0, zoom_range]

        # get bounding box of sfincs model
        minx, miny, maxx, maxy = region.total_bounds
        transformer = Transformer.from_crs(region.crs.to_epsg(), 3857)

        # rotation of the model
        cosrot = math.cos(-self.rotation * math.pi / 180)
        sinrot = math.sin(-self.rotation * math.pi / 180)

        # axis order is different for geographic and projected CRS
        if region.crs.is_geographic:
            minx, miny = map(
                max, zip(transformer.transform(miny, minx), [-20037508.34] * 2)
            )
            maxx, maxy = map(
                min, zip(transformer.transform(maxy, maxx), [20037508.34] * 2)
            )
        else:
            minx, miny = map(
                max, zip(transformer.transform(minx, miny), [-20037508.34] * 2)
            )
            maxx, maxy = map(
                min, zip(transformer.transform(maxx, maxy), [20037508.34] * 2)
            )

        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            logger.debug("Processing zoom level " + str(izoom))

            zoom_path = os.path.join(index_path, str(izoom))

            for transform, col, row in tile_window(izoom, minx, miny, maxx, maxy):
                # transform is a rasterio Affine object
                # col, row are the tile indices
                file_name = os.path.join(
                    zoom_path, str(col), str(row) + "." + extension
                )

                # get the coordinates of the tile in webmercator projection
                x = np.arange(0, npix) + 0.5
                y = np.arange(0, npix) + 0.5
                x3857, y3857 = transform * (x, y)
                x3857, y3857 = np.meshgrid(x3857, y3857)

                # convert to SFINCS coordinates
                x, y = transformer.transform(x3857, y3857, direction="INVERSE")

                # Now rotate around origin of SFINCS model
                x00 = x - self.x0
                y00 = y - self.y0
                xg = x00 * cosrot - y00 * sinrot
                yg = x00 * sinrot + y00 * cosrot

                # determine the SFINCS cell indices
                iind = np.floor(xg / self.dx).astype(int)
                jind = np.floor(yg / self.dy).astype(int)
                ind = iind * self.nmax + jind
                ind[iind < 0] = -999
                ind[jind < 0] = -999
                ind[iind >= self.mmax] = -999
                ind[jind >= self.nmax] = -999

                # only write tiles that link to at least one SFINCS cell
                if np.any(ind >= 0):
                    if not os.path.exists(os.path.join(zoom_path, str(col))):
                        os.makedirs(os.path.join(zoom_path, str(col)))
                    # And write indices to file
                    if fmt == "bin":
                        fid = open(file_name, "wb")
                        fid.write(ind)
                        fid.close()
                    elif fmt == "png":
                        # for png, change nodata -999 nodata into 0
                        ind[ind == -999] = 0
                        int2png(ind, file_name)
