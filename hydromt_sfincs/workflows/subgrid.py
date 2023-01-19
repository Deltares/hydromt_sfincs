from hydromt import raster
import numpy as np
import os
from pyproj import CRS
from scipy import interpolate
from typing import List, Tuple, Dict, Union
import xarray as xr

from .merge import merge_multi_dataarrays

__all__ = ["subgrid_v_table", "subgrid_q_table"]

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
