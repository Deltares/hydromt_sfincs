"""RegularGrid class for SFINCS."""
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from pyflwdir.regions import region_area
from pyproj import CRS, Transformer
from scipy import ndimage
from shapely.geometry import LineString

from .subgrid import SubgridTableRegular
from .workflows.tiling import int2png, tile_window

logger = logging.getLogger(__name__)


class RegularGrid:
    def __init__(self, x0, y0, dx, dy, nmax, mmax, epsg=None, rotation=0):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax  # height
        self.mmax = mmax  # width
        self.rotation = rotation
        self.crs = None
        if epsg is not None:
            self.crs = CRS.from_user_input(epsg)
        self.subgrid = SubgridTableRegular()
        # self.data = xr.Dataset()

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
    def edges(self, x_dim="xg", y_dim="yg"):
        """Return the coordinates of the cell-edges the regular grid."""
        x_edges, y_edges = (
            self.transform
            * self.transform.translation(0, 0)
            * np.meshgrid(np.arange(self.mmax + 1), np.arange(self.nmax + 1))
        )
        # edges = {
        #     "yg": ((y_dim, x_dim), y_edges),
        #     "xg": ((y_dim, x_dim), x_edges),
        # }
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

    def ind(self, mask: np.ndarray) -> np.ndarray:
        """Return indices of active cells in mask."""
        assert mask.shape == (self.nmax, self.mmax)
        ind = np.where(mask.ravel(order="F"))[0]
        return ind

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

    def read_ind(
        self,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> np.ndarray:
        """Read indices of active cells in mask from binary file."""
        _ind = np.fromfile(ind_fn, dtype="u4")
        ind = _ind[1:] - 1  # convert to zero based index
        assert _ind[0] == ind.size

        return ind

    def create_mask_active(
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
