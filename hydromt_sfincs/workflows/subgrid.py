from hydromt import raster
import numpy as np
import os
from pyproj import CRS
from scipy import interpolate
from typing import List, Tuple, Dict, Union
import xarray as xr

from .merge import merge_multi_dataarrays

__all__ = ["subgrid_reggrid"]


def subgrid_reggrid(
    da_mask: xr.DataArray,
    bathymetry_set: List[xr.DataArray],
    bathymetry_merge_kwargs: Union[Dict, List[Dict]] = {},
    manning_set: List[xr.DataArray] = [],
    manning_merge_kwargs: Union[Dict, List[Dict]] = {},
    nbins: int = 10,
    nr_subgrid_pixels: int = 20,
    nrmax: int = 2000,  # blocksize
    max_gradient: float = 5.0,
    zmin: float = -99999.0,
    manning_land: float = 0.04,
    manning_sea: float = 0.02,
    rgh_lev_land: float = 0.0,
    highres_dir=None,
):
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
            da_dep = merge_multi_dataarrays(
                bathymetry_set,
                merge_kwargs=bathymetry_merge_kwargs,
                reproj_kwargs=reproj_kwargs,
                merge_method="first",
                reproj_method="bilinear",
                interp_method="linear",
            )
            # TODO what to do with remaining cell with nan values
            # da_dep = da_dep.fillna(value)
            assert np.all(~np.isnan(da_dep))

            # get subgrid manning roughness tile
            if len(manning_set) > 0:
                da_man = merge_multi_dataarrays(
                    manning_set,
                    merge_kwargs=manning_merge_kwargs,
                    reproj_kwargs=reproj_kwargs,
                    merge_method="first",
                    reproj_method="bilinear",
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
            subgrid_reggrid_tile(
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


def subgrid_reggrid_tile(
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

    for n in range(da_mask.shape[0]):  # row
        for m in range(da_mask.shape[1]):  # col
            if da_mask[n, m] < 1:  # Not an active point
                continue

            # First the volumes in the cells
            nslice = slice(n * refi, (n + 1) * refi)
            mslice = slice(m * refi, (m + 1) * refi)
            zv = da_dep[nslice, mslice].values.flatten()
            if not np.all(~np.isnan(zv)):
                print(zv)
            z, v, zmin, zmax, zmean = subgrid_v_table(
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
            zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, rv, nbins)
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
            zmin, zmax, hrep, navg, zz = subgrid_q_table(zv, rv, nbins)
            ds_sbg["v_zmin"][n, m] = zmin
            ds_sbg["v_zmax"][n, m] = zmax
            ds_sbg["v_hrep"][:, n, m] = hrep
            ds_sbg["v_navg"][:, n, m] = navg

    return ds_sbg


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
    # TODO update docstring & add annotation
    def get_dhdz(z, V, dx, dy):
        # change in level per unit of volume (m/m)
        dz = np.diff(z)
        # change in volume (normalized to meters)
        dh = np.diff(V) / (dx * dy)
        return dh / dz

    def get_dzdh(z, V, a):
        # change in level per unit of volume (m/m)
        dz = np.diff(z)
        # change in volume (normalized to meters)
        dh = np.maximum(np.diff(V) / a, 0.001)
        return dz / dh

    # Cell area
    a = elevation.size * dx * dy

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
    #    dhdz = get_dhdz(z, V, dx, dy)
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
    # while ((dhdz.min() < max_gradient and not(np.isclose(dhdz.min(), max_gradient))) and n < nbins):
    #     # reshape until gradient is satisfactory
    #     idx = np.where(dhdz == dhdz.min())[0]
    #     z[idx + 1] = z[idx] + (np.diff(V)[idx]/(dy*dx))/max_gradient
    #     dhdz = get_dhdz(z, V, dx, dy)
    #     n += 1
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
    # TODO update docstring & add annotation

    hrep = np.zeros(nbins)
    navg = np.zeros(nbins)
    zz = np.zeros(nbins)

    n = int(np.size(elevation) / 2)  # Nr of pixels in a half grid cell

    # Side A
    elevation_a = elevation[0:n]
    manning_a = manning[0:n]
    idx = np.argsort(elevation_a)
    try:
        z_a = elevation_a[idx]
        manning_a = manning_a[idx]
    except IndexError:
        print(manning_a.size, elevation_a.size)
    zmin_a = z_a[0]
    zmax_a = z_a[-1]

    # Side B
    elevation_b = elevation[n:]
    manning_b = manning[n:]
    idx = np.argsort(elevation_b)
    z_b = elevation_b[idx]
    manning_b = manning_b[idx]
    zmin_b = z_b[0]
    zmax_b = z_b[-1]

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

        # Side A
        ibelow = np.where(z_a <= zbin)  # index of pixels below bin level
        h = np.maximum(zbin - z_a, 0.0)  # water depth in each pixel
        qi = h ** (5.0 / 3.0) / manning_a  # unit discharge in each pixel
        q = np.sum(qi) / n  # combined unit discharge for cell

        navg_a = manning_a[ibelow].mean()  # mean manning's n
        hrep_a = (q * navg_a) ** (3.0 / 5.0)  # conveyance depth

        # Side B
        ibelow = np.where(z_b <= zbin)  # index of pixels below bin level
        h = np.maximum(zbin - z_b, 0.0)  # water depth in each pixel
        qi = h ** (5.0 / 3.0) / manning_b  # unit discharge in each pixel
        q = np.sum(qi) / n  # combined unit discharge for cell
        navg_b = manning_b[ibelow].mean()  # mean manning's n
        hrep_b = (q * navg_b) ** (3.0 / 5.0)  # conveyance depth

        # Now take minimum value of cells A and B
        if hrep_a <= hrep_b:
            hrep[ibin] = hrep_a
            navg[ibin] = navg_a
        else:
            hrep[ibin] = hrep_b
            navg[ibin] = navg_b

    return zmin, zmax, hrep, navg, zz
