import pygeos
import geopandas as gpd
from typing import Tuple
import numpy as np
import logging
import hydromt
import xarray as xr

logger = logging.getLogger(__name__)


__all__ = ["nearest"]


def nearest(
    gdf_points: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the index of and distance [m] to the nearest geometry
    in `gdf` for each point in `gdf_points`.
    """
    assert np.all(gdf_points.geometry.type == "Point")
    pnts = gdf_points[["geometry"]].copy()
    if gdf_points.crs != gdf2.crs:
        pnts = pnts.to_crs(gdf2.crs)
    # find nearest using pygeos
    pnts_array = pygeos.points([g.coords[:][0] for g in pnts.geometry])
    idx = gdf2.sindex.nearest(pnts_array)[1]
    # get distance in meters
    gdf2_nearest = gdf2.iloc[idx]
    if gdf2_nearest.crs.is_geographic:
        pnts = gdf_points[["geometry"]].copy().to_crs(32736)  # web mercator
        gdf2_nearest = gdf2_nearest.to_crs(32736)
    dst = gdf2_nearest.distance(pnts, align=False).values
    return gdf2.index.values[idx], dst


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
