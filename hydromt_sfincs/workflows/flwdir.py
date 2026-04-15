"""Flow direction and river network workflows for SFINCS models."""

import logging
from typing import Tuple

import geopandas as gpd
import pandas as pd
import hydromt
import numpy as np
import pyflwdir
import xarray as xr
from shapely.geometry import LineString

from shapely.ops import linemerge
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

__all__ = [
    "river_source_points",
    "river_outflow_points",
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
    flwdir = hydromt.gis.flw.flwdir_from_da(da_flwdir, mask=riv_mask)
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
    if gdf_riv.index.size > 1:
        # accumulate to get river length from outlet
        flwdir = pyflwdir.from_dataframe(gdf_riv.set_index("idx"), ds_col="idx_ds")
        gdf_riv["rivdst"] = flwdir.accuflux(gdf_riv["seglen"].values, direction="down")
        # get maximum river length from outlet (at headwater segments) for each river segment
        gdf_riv["rivlen"] = flwdir.fillnodata(
            np.where(flwdir.n_upstream == 0, gdf_riv["rivdst"], 0), 0
        )
    else:
        gdf_riv["rivlen"] = gdf_riv["seglen"]
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
    internal_distance: float = 1000,
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
    _buffer = buffer if buffer is not None else 0
    bnd = gdf_mask.boundary.buffer(_buffer).union_all()
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

        # add uparea attribute if da_uparea is provided
    if da_uparea is not None:
        gdf_pnt["uparea"] = da_uparea.raster.sample(gdf_pnt).values
        gdf_pnt = gdf_pnt.sort_values("uparea", ascending=False).reset_index(drop=True)

        # get buffer around gdf_mask, in- and outflow points should be within this buffer
    if src_type in ["inflow", "outflow"]:
        gdf_pnt = gdf_pnt[gdf_pnt.intersects(bnd)].reset_index(drop=True)

    logger.info(f"Found {gdf_pnt.index.size} {src_type} points.")
    return gdf_pnt

def river_outflow_points(
    gdf_riv: gpd.GeoDataFrame,
    gdf_mask: gpd.GeoDataFrame,
    src_type: str = "inflow",
    buffer: float = 200,
    river_upa: float = 10,
    river_len: float = 1e3,
    internal_distance: float = 1000,
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
    _buffer = buffer if buffer is not None else 0
    bnd = gdf_mask.boundary.buffer(_buffer).union_all()
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

        # add uparea attribute if da_uparea is provided
    if da_uparea is not None:
        gdf_pnt["uparea"] = da_uparea.raster.sample(gdf_pnt).values
        gdf_pnt = gdf_pnt.sort_values("uparea", ascending=False).reset_index(drop=True)

        # get buffer around gdf_mask, in- and outflow points should be within this buffer
    if src_type in ["inflow", "outflow"]:
        gdf_pnt = gdf_pnt[gdf_pnt.intersects(bnd)].reset_index(drop=True)
    # for outflow points, find the all other points that are within the buffer of the outflow point and chose an internal point
    # to do so, do somethin similar as for the inflow points, but now with a buffer around the outflow points and check which points are within this buffer.

    if src_type == "outflow":
        gdf_pnt = river_downstream_boundary_points(
            gdf_pnt,
            gdf_riv,
            gdf_mask,
            src_type=src_type,
            internal_distance=internal_distance,
            logger=logger,
        )
        logger.info(f"Found {gdf_pnt.index.size} {src_type} points after processing downstream boundaries.")
        return gdf_pnt
    
    else:
        logger.info(f"Found {gdf_pnt.index.size} {src_type} points.")
        return gdf_pnt


def river_downstream_boundary_points(
    gdf_pnt: gpd.GeoDataFrame,
    gdf_riv: gpd.GeoDataFrame,
    gdf_mask: gpd.GeoDataFrame,
    src_type: str = "outflow",
    internal_distance: float = 1000,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:

    # collect lines in a list (faster & safer than repeated pd.concat)
    out_lines = []

    # -------------------------------
    # build upstream adjacency ONCE
    # -------------------------------
    has_graph = all(c in gdf_riv.columns for c in ["uparea", "idx", "idx_ds"])
    up_adj = None

    if has_graph:
        # optional: basic sanity checks
        if not gdf_riv["idx"].is_unique:
            logger.warning(
                "gdf_riv['idx'] is not unique; adjacency graph may be invalid."
            )
        up_adj = defaultdict(list)
        for seg_idx, seg_idx_ds in (
            gdf_riv[["idx", "idx_ds"]].dropna().itertuples(index=False)
        ):
            up_adj[int(seg_idx_ds)].append(int(seg_idx))

        logger.info("Built upstream adjacency list once for all outflow points.")

    for out_i, row in gdf_pnt.iterrows():
        p = row.geometry

        pnt_buffer = p.buffer(internal_distance + 10)
        pnt_buffer_near = p.buffer(100)

        gdf_riv_in = gdf_riv[gdf_riv.intersects(pnt_buffer)]
        gdf_riv_in = gdf_riv_in[~gdf_riv_in.within(pnt_buffer)].reset_index(
            drop=True
        )

        if gdf_riv_in.empty:
            logger.warning(
                f"No river segments found within {internal_distance} m of outflow point {out_i}."
            )
            continue

        if len(gdf_riv_in) > 1:
            if "uparea" in gdf_riv_in.columns:
                gdf_riv_in = (
                    gdf_riv_in.sort_values("uparea", ascending=False)
                    .head(1)
                    .reset_index(drop=True)
                )
            else:
                raise ValueError(
                    f"gdf_riv missing required columns: 'uparea' for disambiguating multiple segments near outflow point {out_i} (found {len(gdf_riv_in)} segments)."
                )
                continue

        # target segment idx (single-row gdf_riv_in)
        target_idx = None
        if "idx" in gdf_riv_in.columns:
            target_idx = int(gdf_riv_in.loc[0, "idx"])

        # -------------------------------
        # choose full_line via BFS path if possible, else fallback merge
        # -------------------------------
        full_line = None

        if has_graph and up_adj is not None and target_idx is not None:
            # find start segment near outflow point
            buf10 = p.buffer(10)
            cand = gdf_riv[gdf_riv.intersects(buf10)]
            start_row = (
                cand.loc[cand.distance(p).idxmin()]
                if len(cand)
                else gdf_riv.loc[gdf_riv.distance(p).idxmin()]
            )
            start_idx = int(start_row["idx"])

            if target_idx == start_idx:
                full_line = gdf_riv.loc[
                    gdf_riv["idx"] == start_idx, "geometry"
                ].values[0]
            else:
                queue = deque([start_idx])
                parent = {start_idx: None}
                found = False

                while queue:
                    cur = queue.popleft()
                    for nxt in up_adj.get(cur, []):
                        if nxt in parent:
                            continue
                        parent[nxt] = cur
                        if nxt == target_idx:
                            found = True
                            queue.clear()
                            break
                        queue.append(nxt)

                if not found:
                    logger.warning(
                        f"No upstream path found from outflow point {out_i} start_idx={start_idx} to target_idx={target_idx}. Falling back to merge."
                    )
                else:
                    # reconstruct path
                    idx_path = [target_idx]
                    while parent[idx_path[-1]] is not None:
                        idx_path.append(parent[idx_path[-1]])
                    idx_path = idx_path[::-1]

                    path_gdf = gdf_riv[gdf_riv["idx"].isin(idx_path)].copy()
                    order = {v: i for i, v in enumerate(idx_path)}
                    path_gdf["__ord"] = path_gdf["idx"].map(order)
                    path_gdf = path_gdf.sort_values("__ord").drop(columns="__ord")

                    full_line = linemerge(list(path_gdf.geometry.values))

        if full_line is None:
            # fallback: merge all segments within buffer
            try:
                gdf_merge = gdf_riv[gdf_riv.intersects(pnt_buffer)]
                full_line = linemerge(list(gdf_merge.geometry.values))
            except Exception as e:
                logger.warning(
                    f"Failed to merge segments for outflow point {out_i}. Error: {e}"
                )
                continue

        # -------------------------------
        # ensure we have a LineString to project onto
        # -------------------------------
        if full_line.geom_type == "MultiLineString":
            # pick the component closest to the outflow point
            full_line = min(full_line.geoms, key=lambda ls: ls.distance(p))

        # snap p to the line (optional but helps)
        p_on = full_line.interpolate(full_line.project(p))

        s0 = full_line.project(p_on)
        s_in = max(0.0, s0 - internal_distance)  # clamp
        p_in = full_line.interpolate(s_in)

        out_lines.append(LineString([p_on, p_in]))

    gdf_lines = gpd.GeoDataFrame(geometry=out_lines, crs=gdf_mask.crs)
    logger.info(
        f"Found {len(gdf_lines)} internal outflow lines (from {gdf_pnt.index.size} outflow points)."
    )
    return gdf_lines

