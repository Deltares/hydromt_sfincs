import pygeos
import geopandas as gpd
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


__all__ = ["nearest"]


def nearest(
    gdf_points: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the index of and distance [m] to the nearest geometry
    in `gdf` for each point in `gdf_points`.
    """
    assert np.all(gdf_points.geomtry.type == "Points")
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
