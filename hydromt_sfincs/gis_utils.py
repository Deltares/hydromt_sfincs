import pygeos
import geopandas as gpd
from typing import Tuple
import numpy as np
import logging
from affine import identity as IDENTITY
from numba import njit
import xarray as xr
from pyflwdir.gis_utils import degree_metres_x, degree_metres_y


logger = logging.getLogger(__name__)


__all__ = ["nearest", "spread2d", "flipud"]

# TODO: move to hydromt.raster
def flipud(ds):
    yrev = list(reversed(ds.raster.ycoords))
    return ds.reindex({ds.raster.y_dim: yrev})


def nearest(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame):
    """Return the index of and distance [m] to the nearest geometry
    in `gdf` for each (centroid) point of `gdf1`."""
    if np.all(gdf1.type == "Point"):
        pnts = gdf1.geometry.copy()
    elif np.all(np.isin(gdf1.type, ["LineString", "MultiLineString"])):
        pnts = gdf1.geometry.interpolate(0.5, normalized=True)  # mid point
    elif np.all(np.isin(gdf1.type, ["Polygon", "MultiPolygon"])):
        pnts = gdf1.geometry.representative_point()  # inside polygon
    else:
        raise NotImplementedError("Mixed geometry dataframes are not yet supported.")
    if gdf1.crs != gdf2.crs:
        pnts = pnts.to_crs(gdf2.crs)
    # find nearest using pygeos
    idx = gdf2.sindex.nearest(pygeos.from_shapely(pnts.geometry.values))[1]
    # get distance in meters
    gdf2_nearest = gdf2.iloc[idx]
    if gdf2_nearest.crs.is_geographic:
        pnts = pnts.to_crs(32736)  # web mercator
        gdf2_nearest = gdf2_nearest.to_crs(32736)
    dst = gdf2_nearest.distance(pnts, align=False).values
    return gdf2.index.values[idx], dst


def spread2d(da_obs, da_mask=None, da_friction=None):
    msk, frc = None, None
    if da_mask is not None:
        assert da_obs.raster.identical_grid(da_mask)
        msk = da_mask.values
    if da_friction is not None:
        assert da_obs.raster.identical_grid(da_friction)
        frc = da_friction.values
    out, src, dst = _spread2d(
        obs=da_obs.values,
        msk=msk,
        frc=frc,
        nodata=da_obs.raster.nodata,
        latlon=da_obs.raster.crs.is_geographic,
        transform=da_obs.raster.transform,
    )
    # combine and return dataset
    dims = da_obs.raster.dims
    coords = da_obs.raster.coords
    name = da_obs.name if da_obs.name else "obs"
    da_out = xr.DataArray(dims=dims, coords=coords, data=out, name=name)
    da_out.raster.set_nodata(da_obs.raster.nodata)
    da_src = xr.DataArray(dims=dims, coords=coords, data=src, name="source_idx")
    da_src.raster.set_nodata(-1)
    da_dst = xr.DataArray(dims=dims, coords=coords, data=dst, name="source_dst")
    da_dst.raster.set_nodata(-1)
    da_dst.attrs.update(unit="m")
    ds_out = xr.merge([da_out, da_src, da_dst])
    ds_out.raster.set_crs(da_obs.raster.crs)
    return ds_out


@njit(parallel=True)
def _spread2d(obs, msk=None, nodata=0, frc=None, latlon=False, transform=IDENTITY):
    """Returns filled array with nearest observations, origin cells and friction distance to origin.
    The friction distance is measured through valid cells in the mask and has a uniform value of 1. by default.
    The diagonal distance is taken as the hypot of the vertical and horizontal distances.


    Parameters
    ----------
    osb: 2D array
        Initial array with observations.
    msk: 2D array of bool, optional
        Mask of valid cells to consider for filling.
    nodata: int, float
        Missing data value in obs. Cells with this value and where mask equals True are filled, by default 0.
    frc: 2D array of float
        Friction values, by default a uniform value of 1 is used.
    latlon: bool
        If True the transform in units are assumed to be degrees and converted to metric distances.
    transform: affine.Transform
        Geospatial transform.

    Returns
    -------
    out: 2D array of obs.dtype
        Output observations array where nodata values are filled with the nearest observation.
    src: 2D array of int32
        Linear index of origin cell.
    dst: 2D array of float32
        Distance to origin cell.
    """
    nrow, ncol = obs.shape
    xres, yres, north = transform[0], transform[4], transform[5]
    dx, dy = xres, yres

    out = obs.copy().ravel()
    msk1 = msk != 0 if msk is not None else obs == nodata  # 2D
    src = np.full(obs.size, -1, dtype=np.int32)  # linear index of source
    dst = np.full(obs.size, -1, dtype=np.float32)  # distance from source
    idxs = np.full(obs.size, -1, dtype=np.uint32)
    nxt = out != nodata

    i1 = 0
    for idx in range(out.size):
        if out[idx] != nodata:
            idxs[i1] = idx
            src[idx] = idx
            dst[idx] = 0
            i1 += 1

    i0 = 0
    while True:
        idx = idxs[i0]
        if idx == np.uint32(-1):
            break
        idxs[i0] = np.uint32(-1)
        nxt[idx] = False
        idx0 = src[idx]
        d0 = dst[idx]
        f0 = 1.0 if frc is None else frc[idx]
        r = idx // ncol
        c = idx % ncol
        if latlon:
            lat = north + (r + 0.5) * yres
            dy = degree_metres_y(lat) * yres
            dx = degree_metres_x(lat) * xres
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                r1, c1 = r + dr, c + dc
                if r1 < 0 or r1 >= nrow or c1 < 0 or c1 >= ncol or not msk1[r1, c1]:
                    continue
                d = d0 + np.hypot(dr * dx, dc * dy) * f0
                idx1 = r1 * ncol + c1
                if src[idx1] == -1 or d < dst[idx1]:
                    src[idx1] = idx0
                    dst[idx1] = d
                    out[idx1] = out[idx0]
                    if not nxt[idx1]:
                        idxs[i1] = idx1
                        nxt[idx1] = True
                        i1 += 1
                        if i1 == idxs.size:
                            i1 = 0
        i0 += 1
        if i0 == idxs.size:
            i0 = 0
    return out.reshape(obs.shape), src.reshape(obs.shape), dst.reshape(obs.shape)


# def get_area_grid(ds):
#     """Returns a xarray.DataArray containing the area in [m2] of the reference grid ds.

#     Parameters
#     ----------
#     ds : xarray.DataArray or xarray.DataSet
#         xarray.DataArray or xarray.DataSet containing the reference grid(s).

#     Returns
#     -------
#     da_area : xarray.DataArray
#         xarray.DataArray containing the area in [m2] of the reference grid.
#     """
#     if ds.raster.crs.is_geographic:
#         area = hydromt.gis_utils.reggrid_area(
#             ds.raster.ycoords.values, ds.raster.xcoords.values
#         )
#         da_area = xr.DataArray(
#             data=area.astype("float32"), coords=ds.raster.coords, dims=ds.raster.dims
#         )

#     elif ds.raster.crs.is_projected:
#         da = ds[list(ds.data_vars)[0]] if isinstance(ds, xr.Dataset) else ds
#         xres = abs(da.raster.res[0]) * da.raster.crs.linear_units_factor[1]
#         yres = abs(da.raster.res[1]) * da.raster.crs.linear_units_factor[1]
#         da_area = xr.full_like(da, fill_value=1, dtype=np.float32) * xres * yres

#     da_area.raster.set_nodata(0)
#     da_area.raster.set_crs(ds.raster.crs)
#     da_area.attrs.update(unit="m2")

#     return da_area.rename("area")
