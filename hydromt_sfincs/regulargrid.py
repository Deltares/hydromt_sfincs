from affine import Affine
import geopandas as gpd
import numpy as np
import math
import os
from pathlib import Path
from pyproj import CRS
from typing import Union, Optional, List, Dict, Tuple
from scipy import ndimage
import xarray as xr
import logging

from pyflwdir.regions import region_area
from .sfincs_input import SfincsInput
from . import workflows

logger = logging.getLogger(__name__)

class RegularGrid:
    def __init__(self, x0, y0, dx, dy, nmax, mmax, crs=None, rotation=0):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax  # height
        self.mmax = mmax  # width
        self.rotation = rotation
        self.crs = None
        if crs is not None:
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
            data=np.zeros((self.nmax, self.mmax), dtype=np.uint8),
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

    def create_mask_active(
        self,
        da_mask: xr.DataArray = None,
        da_dep: xr.DataArray = None,
        gdf_include: gpd.GeoDataFrame = None,
        gdf_exclude: gpd.GeoDataFrame = None,
        elv_min: float = None,
        elv_max: float = None,
        fill_area: float = 10,
        drop_area: float = 0,
        connectivity: int = 8,
        all_touched: bool = True,
        reset_mask: bool = False,
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
        reset_mask: bool, optional
            If True, reset existing mask layer. If False (default) updating existing mask.

        Returns
        -------
        xr.DataArray
            model elevation mask
        """

        if not reset_mask and da_mask is not None:
            # use current active mask
            da_mask = da_mask > 0
        elif da_dep is not None:
            # start with active mask where dep available
            da_mask = da_dep != da_dep.raster.nodata
        else:
            # no dep info provided, start with inactive mask
            # Only include and exclude polygons are used
            da_mask = self.empty_mask > 0

        latlon = self.crs.is_geographic

        if da_dep is None and (elv_min is not None or elv_max is not None):
            raise ValueError("da_dep required in combination with elv_min / elv_max")
        elif da_dep is not None and not da_dep.raster.identical_grid(da_mask):
            raise ValueError("da_dep does not match regular grid")

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
        elv_min: Optional[float] = None,
        elv_max: Optional[float] = None,
        connectivity: int = 8,
        all_touched=False,
        reset_bounds=False,
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
        reset_bounds: bool, optional
            If True, reset existing boundary cells of the selected boundary
            type (`btype`) before setting new boundary cells, by default False.
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

        if reset_bounds:  # reset existing boundary cells
            # self.logger.debug(f"{btype} (mask={bvalue:d}) boundary cells reset.")
            da_mask = da_mask.where(da_mask != np.uint8(bvalue), np.uint8(1))
            if (
                elv_min is None
                and elv_max is None
                and gdf_include is None
                and gdf_exclude is None
            ):
                return da_mask

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

    def create_subgrid(
        self,
        da_mask: xr.DataArray,
        da_dep_lst: List[dict],
        da_manning_lst: List[dict] = [],
        nbins: int = 10,
        nr_subgrid_pixels: int = 20,
        nrmax: int = 2000,  # blocksize
        max_gradient: float = 5.0,
        zmin: float = -99999.0,
        manning_land: float = 0.04,
        manning_sea: float = 0.02,
        rgh_lev_land: float = 0.0,
        highres_dir=None,
    ) -> xr.Dataset:
        #TODO add buffer cells
        refi = nr_subgrid_pixels
        # create empty subgrid dataset based on mask dataarray and nbins
        height, width = da_mask.raster.shape
        x_dim, y_dim = da_mask.raster.x_dim, da_mask.raster.y_dim
        ds_sbg = xr.Dataset(coords={"bins": np.arange(nbins), **da_mask.raster.coords})
        # 2D arrays
        for name in [
            "z_zmin",
            "z_zmax",
            "z_zmin",
            "z_zmean",
            "z_volmax",
            "u_zmin",
            "u_zmax",
            "v_zmin",
            "v_zmax",
        ]:
            ds_sbg[name] = xr.Variable(
                (y_dim, x_dim), np.empty((height, width), dtype=np.float32)
            )
        # 3D arrays
        for name in ["z_depth", "u_hrep", "u_navg", "v_hrep", "v_navg"]:
            ds_sbg[name] = xr.Variable(
                ("bins", y_dim, x_dim), np.empty((nbins, height, width), dtype=np.float32)
            )

        dx, dy = da_mask.raster.res  # cell size
        dxp = dx / refi  # size of subgrid pixel
        dyp = dy / refi  # size of subgrid pixel

        # Compute pixel size in metres
        # if da_mask.raster.crs.is_geographic:
        #     ygc = yg[nn : nn + refi, mm : mm + refi] # FIXME
        #     mean_lat =np.abs(np.mean(ygc))
        #     dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
        #     dypm = dyp*111111.0
        # else:
        dxpm = dxp
        dypm = dyp

        nrcb = int(np.floor(nrmax / refi))  # nr of regular cells in a block
        nrbn = int(np.ceil(height / nrcb))  # nr of blocks in n direction
        nrbm = int(np.ceil(width / nrcb))  # nr of blocks in m direction
        ## Loop through blocks
        ib = 0
        for ii in range(nrbm):  # col
            for jj in range(nrbn):  # row
                ib += 1
                print(f"tile {ib}/{nrbn*nrbm}")
                # slices of first block (cell level)
                yslice = slice(jj * nrcb, (jj + 1) * nrcb)
                xslice = slice(ii * nrcb, (ii + 1) * nrcb)
                da_mask_block = da_mask.isel({x_dim: xslice, y_dim: yslice})
                ds_sbg_block = ds_sbg.isel({x_dim: xslice, y_dim: yslice})

                # calculate transform and shape of block at cell and subgrid level
                transform = da_mask_block.raster.transform
                reproj_kwargs = dict(
                    dst_crs=da_mask.raster.crs,
                    dst_transform=transform * transform.scale(1 / refi),
                    dst_width=(da_mask_block.raster.width + 1) * refi,  # add 1 cell overlap
                    dst_height=(da_mask_block.raster.height + 1) * refi,
                )

                # get subgrid bathymetry tile
                da_dep = workflows.merge_multi_dataarrays(
                    da_list=da_dep_lst,
                    reproj_kwargs=reproj_kwargs,
                    merge_method="first",
                    interp_method="linear",
                )
                # TODO what to do with remaining cell with nan values
                # da_dep = da_dep.fillna(value)
                assert np.all(~np.isnan(da_dep))

                # get subgrid manning roughness tile
                if len(da_manning_lst) > 0:
                    da_man = workflows.merge_multi_dataarrays(
                        da_list=da_manning_lst,
                        reproj_kwargs=reproj_kwargs,
                        merge_method="first",
                        interp_method="linear",
                    )

                else:
                    da_man = xr.where(da_dep >= rgh_lev_land, manning_land, manning_sea)
                assert np.all(~np.isnan(da_man))

                # optional write tile to file
                # TODO also write manning tiles?
                # NOTE tiles have overlap!
                if highres_dir:
                    fn_dep_tile = os.path.join(highres_dir, f"dep{ib:05d}.tif")
                    da_dep.raster.to_raster(fn_dep_tile, compress="deflate")

                # compute subgrid tables for tile and update subgrid dataset
                assert da_dep.shape == da_man.shape
                subgrid_tile(
                    ds_sbg=ds_sbg_block,  # updates ds_sbg in place
                    da_dep=da_dep,
                    da_man=da_man,
                    da_mask=da_mask_block,
                    res=(dxpm, dypm),
                    refi=refi,
                    zmin=zmin,
                    max_gradient=max_gradient,
                )

        return ds_sbg

def subgrid_tile(
    ds_sbg: xr.Dataset,  # updated inplace!
    da_dep: xr.DataArray,  # subgrid 2050x2050
    da_man: xr.DataArray,  # subgrid 2050x2050
    da_mask: xr.DataArray,  # computational grid 20x20
    res: Tuple[float] = (1.0, 1.0),  # subgrid resolution [m]
    refi: int = 100,  # subgrid pixels per computational cell
    zmin: float = -20.0,  #
    max_gradient: float = 5.0,  #
):
    # TODO make docstring
    nbins = ds_sbg["bins"].size
    dxpm, dypm = res

    count = 0
    for n in range(da_mask.shape[0]):  # row
        for m in range(da_mask.shape[1]):  # col
            print(count)
            count += 1
            if da_mask[n, m] < 1:  # Not an active point
                continue

            # First the volumes in the cells
            nslice = slice(n * refi, (n + 1) * refi)
            mslice = slice(m * refi, (m + 1) * refi)
            zv = da_dep[nslice, mslice].values.flatten()
            if not np.all(~np.isnan(zv)):
                print(zv)
            z, v, zmin, zmax, zmean = workflows.subgrid_v_table(
                zv, dxpm, dypm, nbins, zmin, max_gradient
            )
            ds_sbg["z_zmin"][n, m] = zmin
            ds_sbg["z_zmax"][n, m] = zmax
            ds_sbg["z_zmean"][n, m] = zmean
            ds_sbg["z_volmax"][n, m] = v[-1]
            ds_sbg["z_depth"][:, n, m] = z[1:]

            # Now the U/V points
            # U
            nslice_u = slice(n * refi, (n + 1) * refi)
            mslice_u = slice(int((m + 0.5) * refi), int((m + 1.5) * refi))
            zv = da_dep[nslice_u, mslice_u].values.transpose().flatten()
            rv = da_man[nslice_u, mslice_u].values.transpose().flatten()
            assert rv.size == zv.size
            if not (np.all(~np.isnan(zv)) and np.all(~np.isnan(rv))):
                print(zv)
            zmin, zmax, hrep, navg, zz = workflows.subgrid_q_table(zv, rv, nbins)
            ds_sbg["u_zmin"][n, m] = zmin
            ds_sbg["u_zmax"][n, m] = zmax
            ds_sbg["u_hrep"][:, n, m] = hrep
            ds_sbg["u_navg"][:, n, m] = navg

            # V
            nslice_v = slice(int((n + 0.5) * refi), int((n + 1.5) * refi))
            mslice_v = slice(m * refi, (m + 1) * refi)
            zv = da_dep[nslice_v, mslice_v].values.flatten()
            rv = da_man[nslice_v, mslice_v].values.flatten()
            assert rv.size == zv.size
            if not (np.all(~np.isnan(zv)) and np.all(~np.isnan(rv))):
                print(zv)
            zmin, zmax, hrep, navg, zz = workflows.subgrid_q_table(zv, rv, nbins)
            ds_sbg["v_zmin"][n, m] = zmin
            ds_sbg["v_zmax"][n, m] = zmax
            ds_sbg["v_hrep"][:, n, m] = hrep
            ds_sbg["v_navg"][:, n, m] = navg

    return ds_sbg        