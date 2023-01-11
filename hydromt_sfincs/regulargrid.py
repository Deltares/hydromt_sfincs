import numpy as np
import math
import xarray as xr
import geopandas as gpd
from affine import Affine
from typing import Union, Optional, List
from pathlib import Path
from pyproj import CRS
from .sfincs_input import SfincsInput
from scipy import ndimage
from pyflwdir.regions import region_area


class RegularGrid:
    def __init__(self, x0, y0, dx, dy, nmax, mmax, rotation, crs=None):

        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax  # height
        self.mmax = mmax  # width
        self.rotation = rotation
        self.shape = (nmax, mmax)
        self.crs = None
        if crs is not None:
            self.crs = CRS.from_user_input(crs)
        self.data = xr.Dataset()

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
    def mask(self):
        if "msk" in self.data:
            return self.data["msk"]

    @staticmethod
    def from_inp(inp: SfincsInput) -> None:
        return RegularGrid(
            inp.x0, inp.y0, inp.dx, inp.dy, inp.nmax, inp.mmax, inp.rotation
        )

    @property
    def affine(self):
        return Affine(self.dx, 0, self.x0, 0, self.dy, self.y0) * Affine.rotation(
            self.rotation
        )

    @property
    def coordinates(self):
        # TODO fix for ratated grids
        transform = self.affine * Affine.translation(0.5, 0.5)
        if self.affine.is_rectilinear:
            x_coords, _ = transform * (np.arange(self.mmax), np.zeros(self.mmax))
            _, y_coords = transform * (np.zeros(self.nmax), np.arange(self.nmax))
        else:
            x_coords, y_coords = transform * np.meshgrid(
                np.arange(self.mmax),
                np.arange(self.nmax),
            )
        return {"y": y_coords, "x": x_coords}

    @property
    def ind(self) -> np.ndarray:
        # assert ind.max() <= np.multiply(*shape)
        iok = np.where(np.transpose(self.mask.values) > 0)
        iok = (iok[1], iok[0])
        ind = np.ravel_multi_index(iok, (self.nmax, self.mmax), order="F")
        return ind

    def write_mask(
        self,
        msk_fn: Union[str, Path] = "sfincs.msk",
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> None:

        # Add 1 because indices in SFINCS start with 1, not 0
        indices_ = np.array(
            np.hstack([np.array(len(self.ind)), self.ind + 1]), dtype="u4"
        )
        indices_.tofile(ind_fn)

        self.write_map(map_fn=msk_fn, data=self.mask.values, dtype="u1")

    def read_mask(
        self,
        msk_fn: Union[str, Path] = "sfincs.msk",
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> xr.DataArray:

        _ind = np.fromfile(ind_fn, dtype="u4")
        ind = _ind[1:] - 1  # convert to zero based index
        assert _ind[0] == ind.size

        # mask = utils.read_binary_map()
        nrow, ncol = self.shape
        mask = np.full((ncol, nrow), 0, dtype="u1")
        mask.flat[ind] = np.fromfile(msk_fn, dtype=mask.dtype)
        mask = mask.transpose()

        da_mask = xr.DataArray(
            name="msk",
            data=mask,
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": 0},
        )
        if len(self.data.data_vars) == 0:
            # overwrite data property if empty
            self.data = da_mask.to_dataset()
            self.data.raster.set_crs(self.crs)
        else:
            self.data.update(da_mask.to_dataset())

        return da_mask

    def create_dep(
        self,
        bathymetry_sets: List[xr.DataArray],
    ) -> xr.DataArray:

        region = self.mask.raster.box

        # TODO convert list of xarrays into one merged xd.DataArray
        # For now, we simply take the first
        da_dep = bathymetry_sets[0]

        # reproject to destination CRS
        check_crs = self.crs is not None and da_dep.raster.crs != self.crs
        check_res = abs(da_dep.raster.res[0]) != self.dx
        if check_crs or check_res:
            da_dep = da_dep.raster.reproject(
                dst_res=self.dx, dst_crs=self.crs, align=True, method="bilinear"
            )
        # clip & mask
        da_dep = (
            da_dep.raster.clip_geom(geom=region, mask=True).drop('mask')
            .raster.mask_nodata() #set nodata to nan
            .fillna(-9999)  # force nodata value to be -9999
            .round(2)  # cm precision
        )
        da_dep.raster.set_nodata(-9999)

        # assign to SFINCS model
        if len(self.data.data_vars) == 0:
            # overwrite data property if empty
            self.data = da_dep.to_dataset()
            self.data.raster.set_crs(self.crs)
        else:
            da_dep.name = "dep"            
            self.data.update(da_dep.to_dataset())

    def create_mask_active(
        self,
        gdf_include: Optional[gpd.GeoDataFrame] = None,
        gdf_exclude: Optional[gpd.GeoDataFrame] = None,
        elv_min: Optional[float] = None,
        elv_max: Optional[float] = None,
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
        assert (
            elv_min is None and elv_max is None
        ) or "dep" in self.data, "elevation data required"

        if "dep" in self.data:
            # initiate array where no-elevation data part of domain is inactive
            da_mask = self.data["dep"] != self.data["dep"].raster.nodata
            transform, latlon = (
                self.data["dep"].raster.transform,
                self.data["dep"].raster.crs.is_geographic,
            )
        else:
            # initiate array where enitre domain is inactive
            da_mask = xr.DataArray(
                name="msk",
                data=0,
                coords=self.coordinates,
                dims=("y", "x"),
                attrs={"_FillValue": 0},
            )

        s = None if connectivity == 4 else np.ones((3, 3), int)
        if elv_min is not None or elv_max is not None:
            _msk = self.data["dep"] != self.data["dep"].raster.nodata
            if elv_min is not None:
                _msk = np.logical_and(_msk, self.data["dep"] >= elv_min)
            if elv_max is not None:
                _msk = np.logical_and(_msk, self.data["dep"] <= elv_max)
            if fill_area > 0:
                _msk1 = np.logical_xor(
                    _msk, ndimage.binary_fill_holes(_msk, structure=s)
                )
                regions = ndimage.measurements.label(_msk1, structure=s)[0]
                lbls, areas = region_area(regions, transform, latlon)
                _msk = np.logical_or(
                    _msk, np.isin(regions, lbls[areas / 1e6 < fill_area])
                )
                n = int(sum(areas / 1e6 < fill_area))
                print(f"{n} gaps outside valid elevation range < {fill_area} km2.")
            da_mask = np.logical_and(da_mask, _msk)
            if drop_area > 0:
                regions = ndimage.measurements.label(da_mask.values, structure=s)[0]
                lbls, areas = region_area(regions, transform, latlon)
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

        # update sfincs mask with boolean mask to conserve mask values
        da_mask = da_mask.where(da_mask, np.uint8(0))  # unint8 dtype!

        if len(self.data.data_vars) == 0:
            # overwrite data property if empty
            self.data = da_mask.to_dataset()
            self.data.raster.set_crs(self.crs)
        else:
            da_mask.name = "msk"
            da_mask.attrs = {"_FillValue": 0}
            self.data.update(da_mask.to_dataset())

        return da_mask

    def create_mask_bounds(
        self,
        btype: str = "waterlevel",
        gdf_include: Optional[gpd.GeoDataFrame] = None,
        gdf_exclude: Optional[gpd.GeoDataFrame] = None,
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
        assert (
            elv_min is None and elv_max is None
        ) or "dep" in self.data, "elevation data required"

        assert "msk" in self.data, "mask required"

        btype = btype.lower()
        bvalues = {"waterlevel": 2, "outflow": 3}
        if btype not in bvalues:
            raise ValueError('btype must be one of "waterlevel", "outflow"')
        bvalue = bvalues[btype]

        s = None if connectivity == 4 else np.ones((3, 3), int)
        da_mask = self.data["msk"] != self.data["msk"].raster.nodata
        bounds0 = np.logical_xor(da_mask, ndimage.binary_erosion(da_mask, structure=s))
        bounds = bounds0.copy()

        if elv_min is not None:
            bounds = np.logical_and(bounds, self.data["dep"] >= elv_min)
        if elv_max is not None:
            bounds = np.logical_and(bounds, self.data["dep"] <= elv_max)
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
        if bvalue == 3 and np.any(self.data["msk"] == 2):
            msk2_dilated = ndimage.binary_dilation(
                (da_mask == 2).values,
                structure=np.ones((3, 3)),
                iterations=1,  # minimal one cell distance between msk2 and msk3 cells
            )
            bounds = bounds.where(~msk2_dilated, False)

        ncells = np.count_nonzero(bounds.values)
        if ncells > 0:
            self.data["msk"] = self.data["msk"].where(~bounds, np.uint8(bvalue))

        return da_mask

    def write_map(
        self,
        map_fn: Union[str, Path],
        data: Union[xr.DataArray, np.ndarray],
        dtype: Union[str, np.dtype] = "f4",
    ) -> None:

        if isinstance(data, xr.DataArray):
            data = data.values

        data_out = np.asarray(
            data.transpose()[self.mask.values.transpose() > 0], dtype=dtype
        )
        data_out.tofile(map_fn)

    def read_map(
        self,
        map_fn: Union[str, Path],
        dtype: Union[str, np.dtype] = "f4",
        mv: float = -9999.0,
        name: str = None,
    ) -> xr.DataArray:
        nrow, ncol = self.shape
        data = np.full((ncol, nrow), mv, dtype=dtype)
        data.flat[self.ind] = np.fromfile(map_fn, dtype=dtype)
        data = data.transpose()

        da = xr.DataArray(
            name=map_fn.split(".")[-1] if name is None else name,
            data=data,
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": mv},
        )
        self.data.update(da.to_dataset())
        # da.raster.write_crs(self.crs)
        # TODO da = ...
        return da
