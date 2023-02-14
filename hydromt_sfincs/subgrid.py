# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:25:23 2022

@author: ormondt
"""
import numpy as np
import os
from scipy import interpolate
import xarray as xr
from . import workflows
from hydromt import gis_utils


class SubgridTableRegular:
    def __init__(self, version=0):
        # A regular subgrid table contains only for cells with msk>0
        self.version = version

    def load(self, file_name, mask):

        if isinstance(mask, xr.DataArray):
            mask = mask.values

        iok = np.where(np.transpose(mask) > 0)
        iok = (iok[1], iok[0])

        nmax = np.shape(mask)[0]
        mmax = np.shape(mask)[1]

        grid_dim = (nmax, mmax)

        file = open(file_name, "rb")

        # File version
        # self.version = np.fromfile(file, dtype="i4", count=1)[0]
        self.nr_cells = np.fromfile(file, dtype="i4", count=1)[0]
        self.nr_uv_points = np.fromfile(file, dtype="i4", count=1)[0]
        self.nbins = np.fromfile(file, dtype="i4", count=1)[0]

        # Z points
        self.z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.z_zmean = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.z_depth = np.full((self.nbins, *grid_dim), fill_value=np.nan, dtype=float)

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.u_hrep = np.full((self.nbins, *grid_dim), fill_value=np.nan, dtype=float)
        self.u_navg = np.full((self.nbins, *grid_dim), fill_value=np.nan, dtype=float)

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.v_hrep = np.full((self.nbins, *grid_dim), fill_value=np.nan, dtype=float)
        self.v_navg = np.full((self.nbins, *grid_dim), fill_value=np.nan, dtype=float)

        self.z_zmin[iok[0], iok[1]] = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_zmax[iok[0], iok[1]] = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.z_zmean[iok[0], iok[1]] = np.fromfile(
            file, dtype="f4", count=self.nr_cells
        )
        self.z_volmax[iok[0], iok[1]] = np.fromfile(
            file, dtype="f4", count=self.nr_cells
        )
        for ibin in range(self.nbins):
            self.z_depth[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype="f4", count=self.nr_cells
            )

        self.u_zmin[iok[0], iok[1]] = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.u_zmax[iok[0], iok[1]] = np.fromfile(file, dtype="f4", count=self.nr_cells)
        for ibin in range(self.nbins):
            self.u_hrep[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype="f4", count=self.nr_cells
            )
        for ibin in range(self.nbins):
            self.u_navg[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype="f4", count=self.nr_cells
            )

        self.v_zmin[iok[0], iok[1]] = np.fromfile(file, dtype="f4", count=self.nr_cells)
        self.v_zmax[iok[0], iok[1]] = np.fromfile(file, dtype="f4", count=self.nr_cells)
        for ibin in range(self.nbins):
            self.v_hrep[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype="f4", count=self.nr_cells
            )
        for ibin in range(self.nbins):
            self.v_navg[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype="f4", count=self.nr_cells
            )

        file.close()

    def save(self, file_name, mask):

        if isinstance(mask, xr.DataArray):
            mask = mask.values

        iok = np.where(np.transpose(mask) > 0)
        iok = (iok[1], iok[0])

        nmax = np.shape(self.z_zmin)[0]
        mmax = np.shape(self.z_zmin)[1]

        # Add 1 because indices in SFINCS start with 1, not 0
        ind = np.ravel_multi_index(iok, (nmax, mmax), order="F") + 1

        file = open(file_name, "wb")
        # file.write(np.int32(self.version))  # version
        file.write(np.int32(np.size(ind)))  # Nr of active points
        file.write(np.int32(1))  # min
        file.write(np.int32(self.nbins))

        # Z
        v = self.z_zmin[iok]
        file.write(np.float32(v))
        v = self.z_zmax[iok]
        file.write(np.float32(v))
        v = self.z_volmax[iok]
        file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.z_depth[ibin, :, :])[iok]
            file.write(np.float32(v))

        # U
        v = self.u_zmin[iok]
        file.write(np.float32(v))
        v = self.u_zmax[iok]
        file.write(np.float32(v))
        dhdz = np.full(np.shape(v), 1.0)
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ibin in range(self.nbins):
            v = np.squeeze(self.u_hrep[ibin, :, :])[iok]
            file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.u_navg[ibin, :, :])[iok]
            file.write(np.float32(v))

        # V
        v = self.v_zmin[iok]
        file.write(np.float32(v))
        v = self.v_zmax[iok]
        file.write(np.float32(v))
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ibin in range(self.nbins):
            v = np.squeeze(self.v_hrep[ibin, :, :])[iok]
            file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.v_navg[ibin, :, :])[iok]
            file.write(np.float32(v))

        file.close()

    def build(
        self,
        da_mask: xr.DataArray,
        da_dep_lst: list[dict],
        da_manning_lst: list[dict] = [],
        nbins=10,
        nr_subgrid_pixels=20,
        nrmax=2000,
        max_gradient=5.0,
        z_minimum=-99999.0,  # unused
        manning_land: float = 0.04,
        manning_sea: float = 0.02,
        rgh_lev_land: float = 0.0,
        highres_dep_dir: str = None,
        highres_manning_dir: str = None,
        quiet=False,  # TODO replace by logger
    ):

        if highres_dep_dir:
            # Write the raster paths to a text file
            highres_dir = os.path.abspath(os.path.join(highres_dep_dir, os.pardir))
            filelist_dep = open(f"{highres_dir}\\filelist_dep.txt", "w")
        if highres_manning_dir:
            # Write the raster paths to a text file
            highres_dir = os.path.abspath(os.path.join(highres_manning_dir, os.pardir))
            filelist_man = open(f"{highres_dir}\\filelist_man.txt", "w")

        refi = nr_subgrid_pixels
        self.nbins = nbins
        grid_dim = da_mask.raster.shape
        x_dim, y_dim = da_mask.raster.x_dim, da_mask.raster.y_dim

        # Z points
        self.z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.z_zmean = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.z_depth = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=float)

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.u_hrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=float)
        self.u_navg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=float)

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=float)
        self.v_hrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=float)
        self.v_navg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=float)

        dx, dy = da_mask.raster.res
        dxp = dx / refi  # size of subgrid pixel
        dyp = dy / refi  # size of subgrid pixel

        n1, m1 = grid_dim
        nrcb = int(np.floor(nrmax / refi))  # nr of regular cells in a block
        nrbn = int(np.ceil(n1 / nrcb))  # nr of blocks in n direction
        nrbm = int(np.ceil(m1 / nrcb))  # nr of blocks in m direction

        # avoid blocks with width or height of 1
        merge_last_col = False
        merge_last_row = False
        if m1 % nrcb == 1:
            nrbm -= 1
            merge_last_col = True
        if n1 % nrcb == 1:
            nrbn -= 1
            merge_last_row = True

        # TODO add to logger
        if not quiet:
            print("Number of regular cells in a block : " + str(nrcb))
            print("Number of blocks in n direction    : " + str(nrbn))
            print("Number of blocks in m direction    : " + str(nrbm))

        if not quiet:
            print(f"Grid size of flux grid            : dx={dx}, dy={dy}")
            print(f"Grid size of subgrid pixels       : dx={dxp}, dy={dyp}")

        ## Loop through blocks
        ib = -1
        for ii in range(nrbm):
            bm0 = ii * nrcb  # Index of first m in block
            bm1 = min(bm0 + nrcb, m1)  # last m in block
            if merge_last_col and ii == (nrbm - 1):
                bm1 += 1

            for jj in range(nrbn):
                bn0 = jj * nrcb  # Index of first n in block
                bn1 = min(bn0 + nrcb, n1)  # last n in block
                if merge_last_row and jj == (nrbn - 1):
                    bn1 += 1

                # Count
                ib += 1
                if not quiet:
                    print(
                        f"\nblock {ib + 1}/{nrbn * nrbm} -- "
                        f"col {bm0}:{bm1-1} | row {bn0}:{bn1-1}"
                    )

                # calculate transform and shape of block at cell and subgrid level
                da_mask_block = da_mask.isel(
                    {x_dim: slice(bm0, bm1), y_dim: slice(bn0, bn1)}
                )
                assert np.all(
                    [s > 1 for s in da_mask_block.shape]
                ), "this shouldn't happen"
                if np.all(da_mask_block == 0):  # not active cells in block
                    print("Skip block - No active cells")
                    continue
                print(
                    f"Processing block with {np.sum(da_mask_block):d} active cells .."
                )

                transform = da_mask_block.raster.transform
                # add refi cells overlap in both dimensions for u and v in last row / col
                reproj_kwargs = dict(
                    dst_crs=da_mask.raster.crs,
                    dst_transform=transform * transform.scale(1 / refi),
                    dst_width=(da_mask_block.raster.width + 1) * refi,
                    dst_height=(da_mask_block.raster.height + 1) * refi,
                )

                # get subgrid bathymetry tile
                da_dep = workflows.merge_multi_dataarrays(
                    da_list=da_dep_lst,
                    reproj_kwargs=reproj_kwargs,
                    merge_method="first",
                    interp_method="linear",
                ).load()

                # set minimum depth
                da_dep = np.maximum(da_dep, z_minimum)
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
                    ).load()

                else:
                    da_man = xr.where(da_dep >= rgh_lev_land, manning_land, manning_sea)
                assert np.all(~np.isnan(da_man))

                # optional write tile to file
                # NOTE tiles have overlap! da_dep[:-refi,:-refi]
                # TODO properly fix the flipud issue in raster.to_raster
                if highres_dep_dir:
                    fn_dep_tile = os.path.join(
                        highres_dep_dir, f"merged_dep_m{ii:05d}_n{jj:05d}.tif"
                    )
                    if da_dep.raster.res[1] > 0:
                        da_dep.raster.flipud().raster.to_raster(
                            fn_dep_tile, compress="deflate"
                        )
                    else:
                        da_dep.raster.to_raster(fn_dep_tile, compress="deflate")
                    # add to filelist
                    filelist_dep.write(f"{fn_dep_tile}\n")
                if highres_manning_dir:
                    fn_man_tile = os.path.join(
                        highres_manning_dir, f"merged_man_m{ii:05d}_n{jj:05d}.tif"
                    )
                    if da_man.raster.res[1] > 0:
                        da_man.raster.flipud().raster.to_raster(
                            fn_man_tile, compress="deflate"
                        )
                    else:
                        da_man.raster.to_raster(fn_man_tile, compress="deflate")
                    # add to filelist
                    filelist_man.write(f"{fn_man_tile}\n")

                zg = da_dep.values
                manning_grid = da_man.values
                yg = da_dep.raster.ycoords.values

                # Now compute subgrid properties
                # Loop through all active cells in this block
                for m in range(bm0, bm1):
                    for n in range(bn0, bn1):

                        if da_mask.values[n, m] < 1:
                            # Not an active point
                            continue

                        # # Compute pixel size in metres
                        if da_mask.raster.crs.is_geographic:
                            ygc = yg[nn : nn + refi, mm : mm + refi]
                            mean_lat = np.abs(np.mean(ygc))
                            dxpm = dxp * 111111.0 * np.cos(np.pi * mean_lat / 180.0)
                            dypm = dyp * 111111.0
                        else:
                            dxpm = dxp
                            dypm = dyp

                        # First the volumes in the cells
                        nn = (n - bn0) * refi
                        mm = (m - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]
                        zv = zgc.flatten()
                        zvmin = -20.0
                        z, v, zmin, zmax, zmean = subgrid_v_table(
                            zv, dxpm, dypm, nbins, zvmin, max_gradient
                        )
                        self.z_zmin[n, m] = zmin
                        self.z_zmax[n, m] = zmax
                        self.z_zmean[n, m] = zmean
                        self.z_volmax[n, m] = v[-1]
                        self.z_depth[:, n, m] = z[1:]

                        # Now the U/V points
                        # U
                        nn = (n - bn0) * refi
                        mm = (m - bm0) * refi + int(0.5 * refi)
                        zgu = zg[nn : nn + refi, mm : mm + refi]
                        zgu = np.transpose(zgu)
                        zv = zgu.flatten()
                        manning = manning_grid[nn : nn + refi, mm : mm + refi]
                        manning = np.transpose(manning)
                        manning = manning.flatten()
                        zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                        self.u_zmin[n, m] = zmin
                        self.u_zmax[n, m] = zmax
                        self.u_hrep[:, n, m] = hrep
                        self.u_navg[:, n, m] = navg

                        # V
                        nn = (n - bn0) * refi + int(0.5 * refi)
                        mm = (m - bm0) * refi
                        zgu = zg[nn : nn + refi, mm : mm + refi]
                        zv = zgu.flatten()
                        manning = manning_grid[nn : nn + refi, mm : mm + refi]
                        manning = manning.flatten()
                        zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, manning, nbins)
                        self.v_zmin[n, m] = zmin
                        self.v_zmax[n, m] = zmax
                        self.v_hrep[:, n, m] = hrep
                        self.v_navg[:, n, m] = navg

        # write VRT file with all tiles at the end of the loop
        if highres_dep_dir:
            filelist_dep.close()
            # Create a vrt using GDAL
            gis_utils.create_vrt(vrt_path=f"{highres_dir}//dep.vrt",file_list_path=f"{highres_dir}//filelist_dep.txt")
        if highres_manning_dir:
            filelist_man.close()
            # Create a vrt using GDAL
            gis_utils.create_vrt(vrt_path=f"{highres_dir}//manning.vrt",file_list_path=f"{highres_dir}//filelist_man.txt")

    def to_xarray(self, dims, coords):
        ds_sbg = xr.Dataset(coords={"bins": np.arange(self.nbins), **coords})
        ds_sbg.attrs.update({"_FillValue": np.nan})

        zlst2 = ["z_zmin", "z_zmax", "z_zmin", "z_zmean", "z_volmax"]
        uvlst2 = ["u_zmin", "u_zmax", "v_zmin", "v_zmax"]
        lst3 = ["z_depth", "u_hrep", "u_navg", "v_hrep", "v_navg"]
        # 2D arrays
        for name in zlst2 + uvlst2:
            if hasattr(self, name):
                ds_sbg[name] = xr.Variable(dims, getattr(self, name))
        # 3D arrays
        for name in lst3:
            if hasattr(self, name):
                ds_sbg[name] = xr.Variable(("bins", *dims), getattr(self, name))
        return ds_sbg

    def from_xarray(self, ds_sbg):
        for name in ds_sbg.data_vars:
            setattr(self, name, ds_sbg[name].values)


# @njit
def subgrid_v_table(elevation, dx, dy, nbins, zvolmin, max_gradient):
    """
    map vector of elevation values into a hypsometric volume - depth relationship for one grid cell
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    dx: float, x-directional cell size (typically not known at this level) [m]
    dy: float, y-directional cell size (typically not known at this level) [m]
    Return
    ------
    ele_sort : np.ndarray (1D flattened from elevation) with sorted and flattened elevation values
    volume : np.ndarray (1D flattened from elevation) containing volumes (lowest value zero) per sorted elevation value
    """

    def get_dzdh(z, V, a):
        # change in level per unit of volume (m/m)
        dz = np.diff(z)
        # change in volume (normalized to meters)
        dh = np.maximum(np.diff(V) / a, 0.001)
        return dz / dh

    # Cell area
    a = np.size(elevation) * dx * dy

    # Set minimum elevation to -20 (needed with single precision), and sort
    ele_sort = np.sort(np.maximum(elevation, zvolmin).flatten())

    # Make sure each consecutive point is larger than previous
    for j in range(1, np.size(ele_sort)):
        if ele_sort[j] <= ele_sort[j - 1]:
            ele_sort[j] += 1.0e-6

    depth = ele_sort - ele_sort.min()

    volume = np.cumsum((np.diff(depth) * dx * dy) * np.arange(len(depth))[1:])
    # add trailing zero for first value
    volume = np.concatenate([np.array([0]), volume])

    # Resample volumes to discrete bins
    steps = np.arange(nbins + 1) / nbins
    V = steps * volume.max()
    dvol = volume.max() / nbins
    z = interpolate.interp1d(volume, ele_sort)(V)
    dzdh = get_dzdh(z, V, a)
    n = 0
    while (
        dzdh.max() > max_gradient and not (np.isclose(dzdh.max(), max_gradient))
    ) and n < nbins:
        # reshape until gradient is satisfactory
        idx = np.where(dzdh == dzdh.max())[0]
        z[idx + 1] = z[idx] + max_gradient * (dvol / a)
        dzdh = get_dzdh(z, V, a)
        n += 1
    return z, V, elevation.min(), z.max(), ele_sort.mean()


def subgrid_q_table(elevation, manning, nbins):
    """
    map vector of elevation values into a hypsometric hydraulic radius - depth relationship for one grid cell
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    manning : np.ndarray (nr of pixels in one cell) containing subgrid manning roughness values for one grid cell [s m^(-1/3)]
    dx : float, x-directional cell size (typically not known at this level) [m]
    dy : float, y-directional cell size (typically not known at this level) [m]
    Returns
    -------
    ele_sort, R : np.ndarray of sorted elevation values, np.ndarray of sorted hydraulic radii that belong with depth
    """

    hrep = np.zeros(nbins)
    navg = np.zeros(nbins)
    zz = np.zeros(nbins)

    n = int(np.size(elevation))  # Nr of pixels in grid cell
    n05 = int(n / 2)

    zmin_a = np.min(elevation[0:n05])
    zmax_a = np.max(elevation[0:n05])

    zmin_b = np.min(elevation[n05:])
    zmax_b = np.max(elevation[n05:])

    zmin = max(zmin_a, zmin_b)
    zmax = max(zmax_a, zmax_b)

    # Make sure zmax is a bit higher than zmin
    if zmax < zmin + 0.01:
        zmax += 0.01

    # Determine bin size
    dbin = (zmax - zmin) / nbins

    # Loop through bins
    for ibin in range(nbins):

        # Top of bin
        zbin = zmin + (ibin + 1) * dbin
        zz[ibin] = zbin

        ibelow = np.where(elevation <= zbin)  # index of pixels below bin level
        h = np.maximum(
            zbin - np.maximum(elevation, zmin), 0.0
        )  # water depth in each pixel
        qi = h ** (5.0 / 3.0) / manning  # unit discharge in each pixel
        q = np.sum(qi) / n  # combined unit discharge for cell

        if not np.any(manning[ibelow]):
            print("NaNs found?!")
        navg[ibin] = manning[ibelow].mean()  # mean manning's n
        hrep[ibin] = (q * navg[ibin]) ** (3.0 / 5.0)  # conveyance depth

    return zmin, zmax, hrep, navg, zz
