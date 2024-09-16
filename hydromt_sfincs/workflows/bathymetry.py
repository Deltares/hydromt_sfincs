"""Workflows, to estimate river bathymetry and burn these in a DEM."""
import logging

import geopandas as gpd
import numpy as np
import xarray as xr
from hydromt.gis_utils import nearest, parse_crs
from scipy import ndimage
from scipy.interpolate import interp1d
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.ops import linemerge, snap, split, unary_union

logger = logging.getLogger(__name__)

__all__ = [
    "burn_river_rect",
]


def _split_line_by_point(
    line: LineString, point: Point, tolerance: float = 1.0e-5
) -> MultiLineString:
    """Split a single line with a point.

    Parameters
    ----------
    line : LineString
        line
    point : Point
        point
    tolerance : float, optional
        tolerance to snap the point to the line, by default 1.0e-5

    Returns
    -------
    MultiLineString
        splitted line
    """
    return split(snap(line, point, tolerance), point)


def _line_to_points(line: LineString, dist: float = None, n: int = None) -> MultiPoint:
    """Get points along line based on a distance `dist` or number of points `n`.

    Parameters
    ----------
    line : LineString
        line
    dist : float
        distance between points
    n: integer
        numer of points

    Returns
    -------
    MultiPoint
        points
    """
    if dist is not None:
        distances = np.arange(0, line.length, dist)
    elif n is not None:
        distances = np.linspace(0, line.length, n)
    else:
        ValueError('Either "dist" or "n" should be provided')
    points = unary_union(line.interpolate(distances))
    return points


def _split_line_equal(
    line: LineString, approx_length: float, tolerance: float = 1.0e-5
) -> MultiLineString:
    """Split line into segments with equal length.

    Parameters
    ----------
    line : LineString
        line to split
    approx_length : float
        Based in this approximate length the number of line segments is determined.
    tolerance : float, optional
        tolerance to snap the point to the line, by default 1.0e-5

    Returns
    -------
    MultiLineString
        line splitted in segments of equal length
    """
    n = int(np.floor(line.length / approx_length))
    if n <= 1:
        return line
    else:
        split_points = _line_to_points(line, n=n)
        return _split_line_by_point(line, split_points, tolerance=tolerance)


def split_line_equal(gdf: gpd.GeoDataFrame, dist: float) -> gpd.GeoDataFrame:
    """Split lines in `gdf` into segments with equal length `dist`.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        lines
    dist : float
        approximate length of splitted line segments

    Returns
    -------
    gpd.GeoDataFrame
        splitted lines
    """

    def _split_geom(x: gpd.GeoSeries, dist: float = dist) -> MultiLineString:
        return _split_line_equal(x.geometry, dist)

    gdf_splitted = gdf.assign(geometry=gdf.apply(_split_geom, axis=1)).explode(
        index_parts=True
    )
    return gdf_splitted


def interp_along_line_to_grid(
    da_mask: xr.DataArray,
    gdf_lines: gpd.GeoDataFrame,
    gdf_zb: gpd.GeoDataFrame,
    column_names: list = ["z"],
    logger=logger,
) -> xr.DataArray:
    """Interpolate values from `gdf_zb` along
    lines from `gdf_lines` to grid cells from `da_mask`.

    Parameters
    ----------
    da_mask : xr.DataArray, optional
        Boolean mask. Only cells with True values are interpolated.
    gdf_lines : gpd.GeoDataFrame
        center lines
    gdf_zb : gpd.GeoDataFrame
        point locations with alues to interpolate
    column_names : list, optional
        column names to interpolate, by default ["z"]

    Returns
    -------
    xr.Dataset
        interpolated values
    """
    if not all([c in gdf_zb.columns for c in column_names]):
        missing = [c for c in column_names if c not in gdf_zb.columns]
        raise ValueError(f"Missing columns in gdf_zb: {missing}")
    # get cell centers of cells to interpolate
    xs, ys = da_mask.raster.xy(*np.where(da_mask.values))
    cc = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=da_mask.raster.crs)

    # find nearest line and calculate relative distance along line for all z points
    gdf_zb = gdf_zb[["geometry"] + column_names].copy()
    gdf_zb["idx0"], gdf_zb["dist"] = nearest(gdf_zb, gdf_lines)
    nearest_lines = gdf_lines.loc[gdf_zb["idx0"], "geometry"].values
    gdf_zb["x"] = nearest_lines.project(gdf_zb["geometry"].values)
    gdf_zb.set_index("idx0", inplace=True)
    # keep only lines with associated points
    gdf_lines = gdf_lines.loc[np.unique(gdf_zb.index.values)]

    # find nearest line and calculate relative distance along line for all cell centers
    cc["idx0"], cc["dist"] = nearest(cc, gdf_lines)
    nearest_lines = gdf_lines.loc[cc["idx0"], "geometry"].values
    cc["x"] = nearest_lines.project(cc["geometry"].to_crs(gdf_lines.crs).values)

    # interpolate z values per line
    def _interp(cc0, gdf_zb=gdf_zb, column_names=column_names):
        kwargs = dict(kind="linear", fill_value="extrapolate")
        idx0 = cc0["idx0"].values[0]  # iterpolate per line with ID idx0
        for name in column_names:
            x0 = np.atleast_1d(gdf_zb.loc[idx0, "x"])
            z0 = np.atleast_1d(gdf_zb.loc[idx0, name]).astype(np.float32)
            valid = np.isfinite(z0)
            x0, z0 = x0[valid], z0[valid]
            if x0.size == 0:
                cc0[name] = np.nan
                logger.warning(f"River segment {idx0} has no valid values for {name}.")
            elif x0.size == 1:
                cc0[name] = z0[0]
            else:
                x1 = cc0["x"].values
                cc0[name] = interp1d(x0, z0, **kwargs)(x1)
        return cc0

    cc = cc.groupby("idx0").apply(_interp)[["geometry"] + column_names]

    # rasterize interpolated z values
    ds_out = xr.Dataset()
    for name in column_names:
        da0 = da_mask.raster.rasterize(cc, name, nodata=np.nan).where(da_mask)
        ds_out = ds_out.assign(**{name: da0})

    return ds_out


def burn_river_rect(
    da_elv: xr.DataArray,
    gdf_riv: gpd.GeoDataFrame,
    da_man: xr.DataArray = None,
    gdf_zb: gpd.GeoDataFrame = None,
    gdf_riv_mask: gpd.GeoDataFrame = None,
    segment_length: float = 500,
    riv_bank_q: float = 0.5,  # TODO add default value in docstring of setup_subgrid
    rivwth_name: str = "rivwth",
    rivdph_name: str = "rivdph",
    rivbed_name: str = "rivbed",
    manning_name: str = "manning",
    logger=logger,
):
    """Burn rivers with a rectangular cross profile into a DEM.

    Parameters
    ----------
    da_elv, da_man : xr.DataArray
        DEM and manning raster to burn river depth and manning values into
    gdf_riv : gpd.GeoDataFrame
        River center lines.
    gdf_zb : gpd.GeoDataFrame, optional
        Point locations with a 'rivbed' river bed level [m+REF] column, by defualt None
    gdf_riv_mask : gpd.GeoDataFrame, optional
        Mask in which to interpolate z values, by default None.
        If provided, 'rivwth' column is not required in `gdf_riv`.
    segment_length : float, optional
        Approximate river segment length [m], by default 500
    riv_bank_q : float, optional
        quantile [0-1] for river bank estimation, by default 0.25
    rivwth_name, rivdph_name, rivbed_name, manning_name: str, optional
        river width [m], depth [m], bed level [m+REF], & manning [s.m-1/3] column names
        in gdf_riv, by default "rivwth", "rivdph", "rivbed", and "manning"


    """
    # clip
    gdf_riv = gdf_riv.clip(da_elv.raster.box.to_crs(gdf_riv.crs))
    if gdf_riv.index.size == 0:  # no rivers in domain
        return da_elv, da_man

    # reproject to utm if geographic
    dst_crs = gdf_riv.crs
    if dst_crs.is_geographic:
        dst_crs = parse_crs("utm", gdf_riv.to_crs(4326).total_bounds)
        gdf_riv = gdf_riv.to_crs(dst_crs)

    # check river mask
    # create gdf_riv_mask based on buffered river center line only
    # make sure the river is at least one cell wide
    res = abs(da_elv.raster.res[0])
    if rivwth_name in gdf_riv.columns:
        gdf_riv["buf"] = np.maximum(gdf_riv[rivwth_name].fillna(2), res) / 2
    else:
        gdf_riv["buf"] = res / 2
    if gdf_riv_mask is None:
        gdf_riv_mask = gdf_riv.assign(geometry=gdf_riv.buffer(gdf_riv["buf"]))
    else:
        # get gdf_riv outside of mask and buffer these lines
        # then merge with gdf_riv_mask to get the full river mask
        gdf_mask = gpd.GeoDataFrame(
            geometry=[gdf_riv_mask.buffer(0).union_all()],
            crs=gdf_riv_mask.crs,
        )  # create single polygon to clip
        gdf_riv_clip = gdf_riv.overlay(gdf_mask, how="difference")
        gdf_riv_mask1 = gdf_riv_clip.assign(
            geometry=gdf_riv_clip.buffer(gdf_riv_clip["buf"])
        )
        gdf_riv_mask = gpd.overlay(gdf_riv_mask, gdf_riv_mask1, how="union")
    da_riv_mask = da_elv.raster.geometry_mask(gdf_riv_mask)

    if gdf_zb is None and rivbed_name not in gdf_riv.columns:
        # calculate river bedlevel based on river depth per segment
        gdf_riv_seg = split_line_equal(gdf_riv, segment_length).reset_index(drop=True)
        # create mask or river bank cells adjacent to river mask -> numpy array
        riv_mask_closed = ndimage.binary_closing(da_riv_mask)  # remove islands
        riv_bank = np.logical_xor(
            riv_mask_closed, ndimage.binary_dilation(riv_mask_closed)
        )
        # make sure river bank cells are not nodata
        riv_bank = np.logical_and(
            riv_bank, np.isfinite(da_elv.raster.mask_nodata().values)
        )
        # sample elevation at river bank cells
        rows, cols = np.where(riv_bank)
        riv_bank_cc = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*da_elv.raster.xy(rows, cols)),
            data={"z": da_elv.values[rows, cols]},
            crs=da_elv.raster.crs,
        )
        # find nearest river center line for each river bank cell
        riv_bank_cc["idx0"], _ = nearest(riv_bank_cc, gdf_riv_seg)
        # calculate segment river bank elevation as percentile of river bank cells
        gdf_riv_seg["z"] = (
            riv_bank_cc[["idx0", "z"]].groupby("idx0").quantile(q=riv_bank_q)
        )
        # calculate river bed elevation per segment
        gdf_riv_seg[rivbed_name] = gdf_riv_seg["z"] - gdf_riv_seg[rivdph_name]
        # get zb points at center of line segments
        points = gdf_riv_seg.geometry.interpolate(0.5, normalized=True)
        gdf_zb = gdf_riv_seg.assign(geometry=points)
    elif gdf_zb is None:
        # get zb points at center of line segments
        points = gdf_riv.geometry.interpolate(0.5, normalized=True)
        gdf_zb = gdf_riv.assign(geometry=points)
    elif gdf_zb is not None:
        if rivbed_name not in gdf_zb.columns:
            raise ValueError(f"Missing {rivbed_name} attribute in gdf_zb")
        # fill missing manning values based on centerlines
        # TODO manning always defined on centerline?
        if manning_name not in gdf_zb.columns and manning_name in gdf_riv.columns:
            gdf_zb[manning_name] = np.nan
        if np.any(np.isnan(gdf_zb[manning_name])):
            gdf_zb["idx0"], _ = nearest(gdf_zb, gdf_riv)
            man_nearest = gdf_riv.loc[gdf_zb["idx0"], manning_name]
            man_nearest.index = gdf_zb.index
            gdf_zb[manning_name] = gdf_zb[manning_name].fillna(man_nearest)
    elif rivbed_name not in gdf_zb.columns:
        raise ValueError(f"Missing {rivbed_name} or {rivdph_name} attributes")

    # merge river lines > z points are interpolated along merged line
    if gdf_riv.index.size > 1:
        gdf_riv_merged = gpd.GeoDataFrame(
            geometry=[linemerge(gdf_riv.union_all())], crs=gdf_riv.crs
        )
        gdf_riv_merged = gdf_riv_merged.explode(index_parts=True).reset_index(drop=True)
    else:
        gdf_riv_merged = gdf_riv

    # interpolate river depth and manning along river center line
    # TODO nearest interpolation for manning?
    column_names = [rivbed_name]
    if manning_name in gdf_zb.columns:
        column_names += [manning_name]
    for name in column_names:
        gdf_zb = gdf_zb[np.isfinite(gdf_zb[name])]
    ds = interp_along_line_to_grid(
        da_mask=da_riv_mask,
        gdf_lines=gdf_riv_merged,
        gdf_zb=gdf_zb,
        column_names=column_names,
        logger=logger,
    )

    # update elevation with river bottom elevations
    # river bed elevation must be lower than original elevation
    da_elv1 = da_elv.where(
        np.logical_or(np.isnan(ds[rivbed_name]), da_elv < ds[rivbed_name]),
        ds[rivbed_name],
    )

    # update manning:
    da_man1 = da_man
    if manning_name in ds and da_man is not None:
        da_man1 = da_man1.where(np.isnan(ds[manning_name]), ds[manning_name])

    return da_elv1, da_man1
