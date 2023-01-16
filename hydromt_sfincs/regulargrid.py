from affine import Affine
import geopandas as gpd
import numpy as np
import math
from pathlib import Path
from pyproj import CRS
from typing import Union, Optional, List, Dict
from scipy import ndimage
import xarray as xr
import logging

from pyflwdir.regions import region_area
from .sfincs_input import SfincsInput
from .workflows import merge_multi_dataarrays

logger = logging.getLogger(__name__)


class RegularGrid:
    def __init__(self, x0, y0, dx, dy, nmax, mmax, crs, rotation=0):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax  # height
        self.mmax = mmax  # width
        self.rotation = rotation
        self.crs = CRS.from_user_input(crs)
        # self.data = xr.Dataset()

        # cosrot = math.cos(rotation * math.pi / 180)
        # sinrot = math.sin(rotation * math.pi / 180)

        # xx = np.linspace(
        #     0.5 * self.dx, self.mmax * self.dx - 0.5 * self.dx, num=self.mmax
        # )
        # yy = np.linspace(
        #     0.5 * self.dy, self.nmax * self.dy - 0.5 * self.dy, num=self.nmax
        # )

        # xg0, yg0 = np.meshgrid(xx, yy)
        # xg = self.x0 + xg0 * cosrot - yg0 * sinrot
        # yg = self.y0 + xg0 * sinrot + yg0 * cosrot
        # self.xz = xg
        # self.yz = yg

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
    def empty_mask(self) -> xr.DataArray:
        """Return mask with only inactive cells"""
        da_mask = xr.DataArray(
            name="msk",
            data=np.zeros((self.nmax, self.mmax), dtype=np.int8),
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": 0},
        )
        da_mask.raster.set_crs(self.crs)
        return da_mask

    def ind(self, mask: np.ndarray) -> np.ndarray:
        assert mask.shape == (self.nmax, self.mmax)
        iok = np.where(np.transpose(mask) > 0)
        iok = (iok[1], iok[0])
        ind = np.ravel_multi_index(iok, (self.nmax, self.mmax), order="F")
        return ind

    def write_ind(
        self,
        mask: np.ndarray,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> None:
        assert mask.shape == (self.nmax, self.mmax)
        # Add 1 because indices in SFINCS start with 1, not 0
        ind = self.ind(mask)
        indices_ = np.array(np.hstack([np.array(len(ind)), ind + 1]), dtype="u4")
        indices_.tofile(ind_fn)

    def read_ind(
        self,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> np.ndarray:

        _ind = np.fromfile(ind_fn, dtype="u4")
        ind = _ind[1:] - 1  # convert to zero based index
        assert _ind[0] == ind.size

        return ind

    def create_dep(
        self,
        da_list: List[xr.DataArray],
        merge_kwargs: Union[Dict, List[Dict]] = {},
        reproj_kwargs: dict = {},
        merge_method: str = "first",
        reproj_method: str = "bilinear",  # #TODO different method for up- and downscaling?
        interp_method: str = "linear",
        logger=logger,
    ) -> xr.DataArray:

        da_dep = merge_multi_dataarrays(
            da_list=da_list,
            merge_kwargs=merge_kwargs,
            reproj_kwargs=reproj_kwargs,
            merge_method=merge_method,
            reproj_method=reproj_method,
            interp_method=interp_method,
            logger=logger,
        ).raster.reproject_like(self.empty_mask, method=reproj_method)

        return da_dep

    def create_mask_active(
        self,
        da_dep: xr.DataArray = None,
        gdf_include: gpd.GeoDataFrame = None,
        gdf_exclude: gpd.GeoDataFrame = None,
        elv_min: float = None,
        elv_max: float = None,
        fill_area: float = 10,
        drop_area: float = 0,
        connectivity: int = 8,
        all_touched=True,
    ) -> xr.DataArray:
        """Returns a boolean mask of valid (non nondata) elevation cells, optionally bounded
        by several criteria.

        Parameters
        ----------
        gdf_include, gdf_exclude: geopandas.GeoDataFrame, optional
            Geometries with areas to include/exclude from the active model cells.
            Note that include (second last) and exclude (last) and areas are processed after other critera,
            i.e. `elv_min`, `elv_max` and `drop_area`, and thus overrule these criteria for active model cells.
        elv_min, elv_max : float, optional
            Minimum and maximum elevation thresholds for active model cells.
        fill_area : float, optional
            Maximum area [km2] of contiguous cells below `elv_min` or above `elv_max` but surrounded
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

        Returns
        -------
        xr.DataArray
            model elevation mask
        """
        da_mask = self.empty_mask
        latlon = self.crs.is_geographic

        if da_dep is None and (elv_min is not None or elv_max is not None):
            raise ValueError("da_dep required in combination with elv_min / elv_max")
        elif da_dep is not None and not da_dep.raster.identical_grid(da_mask):
            raise ValueError("da_dep does not match regular grid")
        elif da_dep is not None:
            da_mask = da_mask.where(da_dep != da_dep.raster.nodata, 0)

        s = None if connectivity == 4 else np.ones((3, 3), int)
        if elv_min is not None or elv_max is not None:
            _msk = da_dep != da_dep.raster.nodata
            if elv_min is not None:
                _msk = np.logical_and(_msk, da_dep >= elv_min)
            if elv_max is not None:
                _msk = np.logical_and(_msk, da_dep <= elv_max)
            if fill_area > 0:
                _msk1 = np.logical_xor(
                    _msk, ndimage.binary_fill_holes(_msk, structure=s)
                )
                regions = ndimage.measurements.label(_msk1, structure=s)[0]
                # TODO check if region_area works for rotated grids!
                lbls, areas = region_area(regions, self.transform, latlon)
                _msk = np.logical_or(
                    _msk, np.isin(regions, lbls[areas / 1e6 < fill_area])
                )
                n = int(sum(areas / 1e6 < fill_area))
                print(f"{n} gaps outside valid elevation range < {fill_area} km2.")
            da_mask = np.logical_and(da_mask, _msk)
            if drop_area > 0:
                regions = ndimage.measurements.label(da_mask.values, structure=s)[0]
                lbls, areas = region_area(regions, self.transform, latlon)
                _msk = np.isin(regions, lbls[areas / 1e6 >= drop_area])
                n = int(sum(areas / 1e6 < drop_area))
                print(f"{n} regions < {drop_area} km2 dropped.")
                da_mask = np.logical_and(da_mask, _msk)

        if gdf_include is not None:
            try:
                _msk = da_mask.raster.geometry_mask(
                    gdf_include, all_touched=all_touched
                )
                da_mask = np.logical_or(da_mask, _msk)  # NOTE logical OR statement
            except:
                print(f"No mask cells found within include polygon!")
        if gdf_exclude is not None:
            try:
                _msk = da_mask.raster.geometry_mask(
                    gdf_exclude, all_touched=all_touched
                )
                da_mask = np.logical_and(da_mask, ~_msk)
            except:
                print(f"No mask cells found within exclude polygon!")

        # update sfincs mask name, nodata value and crs
        da_mask = da_mask.where(da_mask, 0).rename("mask")
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
        elv_min: Optional[float] = None,
        elv_max: Optional[float] = None,
        connectivity: int = 8,
        all_touched=False,
    ) -> xr.DataArray:
        """Returns a boolean mask model boundary cells, optionally bounded by several
        criteria. Boundary cells are defined by cells at the edge of active model domain.

        Parameters
        ----------
        btype: {'waterlevel', 'outflow'}
            Boundary type
        gdf_include, gdf_exclude: geopandas.GeoDataFrame
            Geometries with areas to include/exclude from the model boundary.
            Note that exclude (second last) and include (last) areas are processed after other critera,
            i.e. `elv_min`, `elv_max`, and thus overrule these criteria for model boundary cells.
        elv_min, elv_max : float, optional
            Minimum and maximum elevation thresholds for boundary cells.
        connectivity: {4, 8}
            The connectivity used to detect the model edge, if 4 only horizontal and vertical
            connections are used, if 8 (default) also diagonal connections.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.

        Returns
        -------
        bounds: xr.DataArray
            Boolean mask of model boundary cells.
        """
        if not da_mask.raster.identical_grid(self.empty_mask):
            raise ValueError("da_mask does not match regular grid")
        latlon = self.crs.is_geographic

        if da_dep is None and (elv_min is not None or elv_max is not None):
            raise ValueError("da_dep required in combination with elv_min / elv_max")
        elif da_dep is not None and not da_dep.raster.identical_grid(da_mask):
            raise ValueError("da_dep does not match regular grid")

        btype = btype.lower()
        bvalues = {"waterlevel": 2, "outflow": 3}
        if btype not in bvalues:
            raise ValueError('btype must be one of "waterlevel", "outflow"')
        bvalue = bvalues[btype]

        s = None if connectivity == 4 else np.ones((3, 3), int)
        bounds0 = np.logical_xor(
            da_mask > 0, ndimage.binary_erosion(da_mask > 0, structure=s)
        )
        bounds = bounds0.copy()

        if elv_min is not None:
            bounds = np.logical_and(bounds, da_dep >= elv_min)
        if elv_max is not None:
            bounds = np.logical_and(bounds, da_dep <= elv_max)
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
