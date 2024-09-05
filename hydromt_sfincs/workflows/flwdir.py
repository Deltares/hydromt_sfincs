"""Flow direction and river network workflows for SFINCS models."""

import logging
from typing import Tuple

import geopandas as gpd
import hydromt
import numpy as np
import pyflwdir
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "river_source_points",
    "river_centerline_from_hydrography",
]


def river_centerline_from_hydrography(
    da_flwdir: xr.DataArray,
    da_uparea: xr.DataArray,
    river_upa: float = 10,
    river_len: float = 1e3,
    gdf_mask: gpd.GeoDataFrame = None,
) -> gpd.GeoDataFrame:
    """Returns the centerline of rivers based on a flow direction
    raster data (`da_flwdir`).

    Parameters
    ----------
    da_flwdir: xarray.DataArray
        D8 flow direction raster data.
    da_uparea: xarray.DataArray, optional
        River mask raster data. Used to mask da_flwdir, by default None.
    river_upa : float, optional
        Minimum upstream area threshold for rivers [km2], by default 10.0
    river_len: float, optional
        Mimimum river length [m] within the model domain to define river cells,
        by default 1000 m.
    gdf_mask: geopandas.GeoDataFrame, optional
        Polygon to clip river center lines before calculating the river length,
        by default None.

    Returns
    -------
    gdf_riv: geopandas.GeoDataFrame
        River line vector data.
    """
    # get river network from hydrography based on upstream area mask
    riv_mask = da_uparea >= river_upa
    if not riv_mask.any():
        return gpd.GeoDataFrame()
    flwdir = hydromt.flw.flwdir_from_da(da_flwdir, mask=riv_mask)
    feats = flwdir.streams(uparea=da_uparea.values)
    gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=da_flwdir.raster.crs)
    # clip to mask and remove empty geometries
    if gdf_mask is not None and isinstance(gdf_mask, gpd.GeoDataFrame):
        gdf_riv = gdf_riv.to_crs(gdf_mask.crs).clip(gdf_mask.union_all())
    gdf_riv = gdf_riv[~gdf_riv.is_empty]
    # create river network from gdf to get distance from outlet 'rivlen'
    # length of river segments
    if gdf_riv.crs.is_geographic:
        gdf_riv["seglen"] = gdf_riv.to_crs("epsg:3857").geometry.length
    else:
        gdf_riv["seglen"] = gdf_riv.geometry.length
    gdf_riv = gdf_riv[gdf_riv["seglen"] > 0]
    if gdf_riv.empty or river_len == 0:
        return gdf_riv
    # accumulate to get river length from outlet
    flwdir = pyflwdir.from_dataframe(gdf_riv.set_index("idx"), ds_col="idx_ds")
    gdf_riv["rivdst"] = flwdir.accuflux(gdf_riv["seglen"].values, direction="down")
    # get maximum river length from outlet (at headwater segments) for each river segment
    gdf_riv["rivlen"] = flwdir.fillnodata(
        np.where(flwdir.n_upstream == 0, gdf_riv["rivdst"], 0), 0
    )
    # filter river network based on total length
    gdf_riv = gdf_riv[gdf_riv["rivlen"] >= river_len]
    return gdf_riv


def river_source_points(
    gdf_riv: gpd.GeoDataFrame,
    gdf_mask: gpd.GeoDataFrame,
    src_type: str = "inflow",
    buffer: float = 200,
    river_upa: float = 10,
    river_len: float = 1e3,
    da_uparea: xr.DataArray = None,
    reverse_river_geom: bool = False,
    logger: logging.Logger = logger,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Returns the locations where a river flows in (`inflow=True`)
    or out (`inflow=False`) of the model gdf_mask.

    Rivers are based on either a river network vector data (`gdf_riv`) or
    a flow direction raster data (`da_flwdir`).

    Parameters
    ----------
    gdf_riv: geopandas.GeoDataFrame
        River network vector data, by default None.
        Requires 'uparea' and 'rivlen' attributes to
        check for river length and upstream area thresholds.
    gdf_mask: geopandas.GeoDataFrame
        Polygon of model gdf_mask of interest.
    src_type: ['inflow', 'outflow', 'headwater'], optional
        Type of river source points to return, by default 'inflow'.
        If 'inflow', return points where the river flows into the model domain.
        If 'outflow', return points where the river flows out of the model domain.
        If 'headwater', return all headwater (including inflow) points within the model domain.
    buffer: float, optional
        Buffer around gdf_mask to select river source points, by default 200 m.
        Inflow points are moved to a downstream confluence if within the buffer.
    river_upa : float, optional
        Minimum upstream area threshold for rivers [km2], by default 10.0
    river_len: float, optional
        Mimimum river length [m] within the model domain to define river cells, by default 1000 m.
    da_uparea: xarray.DataArray, optional
        River upstream area raster data, by default None.
    reverse_river_geom: bool, optional
        If True, assume that segments in 'rivers' are drawn from downstream to upstream.
        Only used if 'rivers' is not None, By default False

    Returns
    -------
    gdf_pnt: geopandas.GeoDataFrame
        Source points
    """
    # data checks
    if not (
        isinstance(gdf_mask, (gpd.GeoDataFrame, gpd.GeoSeries))
        and np.all(np.isin(gdf_mask.geometry.type, ["Polygon", "MultiPolygon"]))
    ):
        raise TypeError("gdf_mask must be a GeoDataFrame of Polygons.")
    if not (
        isinstance(gdf_riv, (gpd.GeoDataFrame, gpd.GeoSeries))
        and np.all(np.isin(gdf_riv.geometry.type, ["LineString", "MultiLineString"]))
    ):
        raise TypeError("gdf_riv must be a GeoDataFrame of LineStrings.")
    if src_type not in ["inflow", "outflow", "headwater"]:
        raise ValueError("src_type must be either 'inflow', 'outflow', or 'headwater'.")
    if gdf_mask.crs.is_geographic:  # to pseudo mercator
        gdf_mask = gdf_mask.to_crs("epsg:3857")

    # clip river to model gdf_mask
    gdf_riv = gdf_riv.to_crs(gdf_mask.crs).clip(gdf_mask.union_all())
    # keep only lines
    gdf_riv = gdf_riv[
        [t.endswith("LineString") for t in gdf_riv.geom_type]
    ].reset_index(drop=True)

    # filter river network based on uparea and length
    if "uparea" in gdf_riv.columns:
        gdf_riv = gdf_riv[gdf_riv["uparea"] >= river_upa]
    if "rivlen" in gdf_riv.columns:
        gdf_riv = gdf_riv[gdf_riv["rivlen"] > river_len]
    if gdf_riv.empty:
        logger.warning(
            "No rivers matching the uparea and rivlen thresholds found in gdf_riv."
        )
        return gpd.GeoDataFrame()

    # remove lines that fully are within the buffer of the mask boundary
    bnd = gdf_mask.boundary.buffer(buffer).union_all()
    gdf_riv = gdf_riv[~gdf_riv.within(bnd)]

    # get source points 1m before the start/end of the river
    # a positive dx results in a point near the start of the line (inflow)
    # a negative dx results in a point near the end of the line (outflow)
    dx = -1 if reverse_river_geom else 1
    gdf_up = gdf_riv.interpolate(dx).to_frame("geometry")
    gdf_ds = gdf_riv.interpolate(-dx).to_frame("geometry")

    # get points that do not intersect with up/downstream end of other river segments
    # use a small buffer of 5m around these points to account for dx and avoid issues with imprecise river geometries
    if src_type in ["inflow", "headwater"]:
        pnts_ds = gdf_ds.buffer(5).union_all()
        gdf_pnt = gdf_up[~gdf_up.intersects(pnts_ds)].reset_index(drop=True)
    elif src_type == "outflow":
        pnts_up = gdf_up.buffer(5).union_all()
        gdf_pnt = gdf_ds[~gdf_ds.intersects(pnts_up)].reset_index(drop=True)

    # get buffer around gdf_mask, in- and outflow points should be within this buffer
    if src_type in ["inflow", "outflow"]:
        gdf_pnt = gdf_pnt[gdf_pnt.intersects(bnd)].reset_index(drop=True)

    # log numer of source points
    logger.info(f"Found {gdf_pnt.index.size} {src_type} points.")

    # add uparea attribute if da_uparea is provided
    if da_uparea is not None:
        gdf_pnt["uparea"] = da_uparea.raster.sample(gdf_pnt).values
        gdf_pnt = gdf_pnt.sort_values("uparea", ascending=False).reset_index(drop=True)

    return gdf_pnt
