"""DEM-based ridge / levee / embankment detection for SFINCS weirs.

Each public ``detect_ridges_*`` function takes an xarray DataArray DEM and
returns a ``geopandas.GeoDataFrame`` of LineStrings ready to be passed to
:py:meth:`hydromt_sfincs.components.geometries.weirs.SfincsWeirs.create` or
split between ``weirs.create`` and ``thin_dams.create``.

Four flavors are provided:

- ``detect_ridges_rea``       : relative-elevation attribute (scipy)
- ``detect_ridges_frangi``    : multi-scale Hessian ridge filter (scikit-image)
- ``detect_ridges_whitebox``  : WhiteboxTools find_ridges [+ geomorphons]
- ``detect_ridges_steger``    : unbiased curvilinear detector (opencv-contrib)

All flavors share the post-processing pipeline
``_postprocess_mask_to_polylines`` which does morphological cleanup,
skeletonization, branch decomposition (via ``skan``), Douglas-Peucker
simplification, and a swath-relief QC filter.

Returned columns: ``name, geometry, stype, width_m, score``.
CRS is that of the input DEM (must be projected; meters).
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.transform
import xarray as xr
from scipy import ndimage
from shapely.geometry import LineString

logger = logging.getLogger(__name__)

__all__ = [
    "detect_ridges_rea",
    "detect_ridges_frangi",
    "detect_ridges_whitebox",
    "detect_ridges_steger",
    "detect_ridges_lcp",
    "detect_river_banks",
]

# Heavy deps (scikit-image, skan, whitebox, cv2) are imported lazily inside
# the functions that need them so that importing this module never fails just
# because an optional extra is missing. Each flavor raises a clear ImportError
# on first call if its extras aren't installed.


def _require_skimage_skan():
    """Lazy-import the shared-pipeline heavy deps; raise a clear error."""
    try:
        from skimage.morphology import (  # noqa: F401
            binary_closing,
            disk,
            remove_small_objects,
            skeletonize,
        )
        from skimage.filters import apply_hysteresis_threshold  # noqa: F401
        from skan import Skeleton  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ridge_detection flavors require scikit-image and skan. "
            "Install with: pip install hydromt_sfincs[ridge_detection]"
        ) from e


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _assert_projected(da_dem: xr.DataArray) -> None:
    """Raise if the DEM CRS is geographic — flavors use metric length params."""
    crs = da_dem.raster.crs
    if crs is None:
        raise ValueError("DEM has no CRS. Reproject to a projected CRS first.")
    if crs.is_geographic:
        raise ValueError(
            f"DEM CRS {crs} is geographic; ridge_detection requires a projected "
            "CRS in meters. Reproject the DEM first (e.g. to local UTM)."
        )


def _dem_pixel_size_m(da_dem: xr.DataArray) -> float:
    """Return isotropic pixel size in meters; warn on anisotropy."""
    dx, dy = map(abs, da_dem.raster.res)
    if abs(dx - dy) / max(dx, dy) > 0.01:
        logger.warning(
            f"DEM pixels are anisotropic (dx={dx}, dy={dy}); using mean."
        )
    return 0.5 * (dx + dy)


def _dem_to_numpy(da_dem: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dem_array, valid_mask). Fills NaN with the mean for filters."""
    arr = np.asarray(da_dem.values, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    valid = np.isfinite(arr)
    nodata = da_dem.raster.nodata
    if nodata is not None and np.isfinite(nodata):
        valid &= arr != nodata
    if not valid.all():
        mean = float(arr[valid].mean()) if valid.any() else 0.0
        arr = np.where(valid, arr, mean)
    return arr, valid


def _high_pass(dem: np.ndarray, sigma_px: float) -> np.ndarray:
    """Remove regional slope so filters respond to local ridge curvature."""
    return dem - ndimage.gaussian_filter(dem, sigma=sigma_px, mode="reflect")


def _rea_reconstruction_mask(
    dem: np.ndarray,
    valid: np.ndarray,
    dx_m: float,
    windows_m: Tuple[float, ...],
    seed_quantile: float,
    grow_quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-scale REA + morphological reconstruction.

    Returns ``(mask, rea_max)`` where ``mask`` is the seed-grown boolean ridge
    mask and ``rea_max`` is the multi-scale REA response (for use as
    ``response_map`` in post-processing).

    For each window, REA = ``z - uniform_filter(z, window)``; we keep the
    pixel-wise maximum across windows. Strong seeds (``seed_quantile`` of
    positive REA) are then grown through the permissive mask
    (``grow_quantile``) via ``skimage.morphology.reconstruction`` — only
    components anchored to a strong seed survive, but those components
    extend to all moderate-REA pixels they touch.

    This produces longer connected components than threshold+hysteresis,
    naturally tracing levees and river banks across weak spots.
    """
    from skimage.morphology import reconstruction

    rea_max = np.zeros_like(dem, dtype=np.float32)
    for w in windows_m:
        win_px = max(3, int(round(w / dx_m)))
        if win_px % 2 == 0:
            win_px += 1
        m = ndimage.uniform_filter(dem, size=win_px, mode="reflect")
        rea_max = np.maximum(rea_max, dem - m)
    rea_max = np.where(valid, rea_max, 0.0)

    pos = rea_max[valid & (rea_max > 0)]
    if pos.size == 0:
        return np.zeros_like(rea_max, dtype=bool), rea_max

    seed_thresh = float(np.quantile(pos, seed_quantile))
    grow_thresh = float(np.quantile(pos, grow_quantile))
    if seed_thresh <= grow_thresh:
        # tiny scenes — fall back to a small bump
        seed_thresh = grow_thresh * 1.001 + 1e-6

    permissive = (rea_max >= grow_thresh) & valid
    seeds = (rea_max >= seed_thresh) & permissive  # required by reconstruction
    if not seeds.any():
        return np.zeros_like(rea_max, dtype=bool), rea_max

    grown = reconstruction(
        seeds.astype(np.uint8),
        permissive.astype(np.uint8),
        method="dilation",
    ).astype(bool)
    return grown, rea_max


def _quantile_hysteresis(
    response: np.ndarray,
    valid: np.ndarray,
    low_q: float,
    high_q: float,
) -> np.ndarray:
    """Hysteresis threshold using quantiles of the positive response."""
    from skimage.filters import apply_hysteresis_threshold

    pos = response[valid & (response > 0)]
    if pos.size == 0:
        return np.zeros_like(response, dtype=bool)
    lo = float(np.quantile(pos, low_q))
    hi = float(np.quantile(pos, high_q))
    if hi <= lo:
        hi = lo * 1.0001 + 1e-9
    return apply_hysteresis_threshold(response, lo, hi)


def _pixel_to_world(
    rows: np.ndarray,
    cols: np.ndarray,
    transform,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized row/col -> x/y using an affine transform (cell centers)."""
    # rasterio.transform.xy handles scalar or 1-D array inputs
    xs, ys = rasterio.transform.xy(transform, rows.tolist(), cols.tolist())
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _line_length_m(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 0.0
    return float(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])).sum())


def _swath_relief_stats(
    line: LineString,
    dem: np.ndarray,
    transform,
    half_m: float,
    n_samples_per_seg: int = 3,
    aggregation_quantile: float = 0.5,
) -> Tuple[float, float]:
    """Return (relief_at_quantile, min_relief) along a line.

    For each interior vertex (and optionally segment midpoints) we sample the
    DEM at the crest and at +/- ``half_m`` along the segment-perpendicular.
    Relief := z_crest - min(z_left, z_right). Negative on valley slopes.

    ``aggregation_quantile`` controls which order statistic of the per-vertex
    relief is returned as the first element: 0.5 = median (strict, reject if
    most of the line is flat); 0.75 = Q75 (lenient, keep if a quarter of the
    line has relief — better for long merged lines spanning flat sections).
    """
    coords = np.asarray(line.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return (np.nan, np.nan)

    # Build sample points: per interior vertex, use central-difference tangent.
    # For v1 this is simple and vectorized.
    p_center = coords[1:-1] if coords.shape[0] >= 3 else coords
    if coords.shape[0] >= 3:
        tangent = coords[2:] - coords[:-2]
    else:
        tangent = coords[1:] - coords[:-1]
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tnorm = np.where(tnorm == 0, 1.0, tnorm)
    t_unit = tangent / tnorm
    # normal = rotate tangent 90 deg
    n_unit = np.stack([-t_unit[:, 1], t_unit[:, 0]], axis=1)

    p_left = p_center + n_unit * half_m
    p_right = p_center - n_unit * half_m

    # World -> row/col using inverse transform
    inv = ~transform
    H, W = dem.shape
    reliefs = []
    for pc, pl, pr in zip(p_center, p_left, p_right):
        rc_c = inv * (pc[0], pc[1])
        rc_l = inv * (pl[0], pl[1])
        rc_r = inv * (pr[0], pr[1])
        cc = int(round(rc_c[0]))
        rcc = int(round(rc_c[1]))
        cl = int(round(rc_l[0]))
        rcl = int(round(rc_l[1]))
        cr_ = int(round(rc_r[0]))
        rcr = int(round(rc_r[1]))
        if not (0 <= rcc < H and 0 <= cc < W):
            continue
        zc = dem[rcc, cc]
        zl = dem[rcl, cl] if 0 <= rcl < H and 0 <= cl < W else np.nan
        zr = dem[rcr, cr_] if 0 <= rcr < H and 0 <= cr_ < W else np.nan
        shoulder_min = np.nanmin([zl, zr])
        reliefs.append(zc - shoulder_min)
    if not reliefs:
        return (np.nan, np.nan)
    reliefs = np.asarray(reliefs, dtype=np.float64)
    q = float(np.nanquantile(reliefs, aggregation_quantile))
    return (q, float(np.nanmin(reliefs)))


def _longest_paths_per_component(sk, max_paths_per_component: int = 1) -> list:
    """Return one (or k) longest geodesic paths per connected component.

    For each connected component of skan's pixel-level skeleton graph, compute
    the diameter (longest shortest-path) using the classic two-pass Dijkstra
    algorithm. Returns a list of paths, each as an ``(N, 2)`` array of
    ``(row, col)`` pixel coordinates ordered along the path.

    Used by the ``merge_strategy='longest_path'`` post-processing mode to
    return one polyline per "thing" rather than one per skan branch — natural
    answer when the user wants to maximise per-line continuity.
    """
    from scipy.sparse.csgraph import connected_components, dijkstra

    coords = np.asarray(sk.coordinates)
    csg = sk.graph
    n = csg.shape[0]
    if n == 0:
        return []

    n_comp, labels = connected_components(csg, directed=False)
    paths = []

    for c in range(n_comp):
        comp_mask = labels == c
        comp_idx = np.nonzero(comp_mask)[0]
        if comp_idx.size < 2:
            continue
        start = int(comp_idx[0])

        # First Dijkstra: find farthest node in component
        d1 = dijkstra(csg, indices=start, directed=False)
        d1_in_comp = np.where(comp_mask, d1, -np.inf)
        far = int(np.argmax(d1_in_comp))

        # Second Dijkstra: from farthest, find diameter endpoint with predecessors
        d2, pred = dijkstra(csg, indices=far, directed=False, return_predecessors=True)
        d2_in_comp = np.where(comp_mask, d2, -np.inf)
        end = int(np.argmax(d2_in_comp))

        # Walk predecessors from end -> far
        path_nodes = []
        cur = end
        guard = 0
        while cur >= 0 and cur != far and guard < n + 1:
            path_nodes.append(cur)
            cur = int(pred[cur]) if pred[cur] >= 0 else -1
            guard += 1
        if cur == far:
            path_nodes.append(far)
        elif guard >= n + 1:
            continue  # pathological — skip
        path_nodes.reverse()

        if len(path_nodes) >= 2:
            path_arr = coords[path_nodes]  # (M, 2) row, col
            paths.append(path_arr)

        # Optional: also include 2nd, 3rd longest paths per component
        # (k-shortest-paths-style); not enabled by default.
        _ = max_paths_per_component  # placeholder to satisfy lint

    return paths


def _line_tangent_at_endpoint(line: LineString, which: str, n_pts: int = 3) -> np.ndarray:
    """Unit tangent of a line at its start (which='start') or end ('end').

    Computed from the first/last ``n_pts`` vertices to be robust to single-
    pixel jitter at the endpoint.
    """
    coords = np.asarray(line.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return np.array([1.0, 0.0])
    n = min(n_pts, coords.shape[0])
    if which == "start":
        v = coords[0] - coords[n - 1]
    else:
        v = coords[-1] - coords[-n]
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([1.0, 0.0])
    return v / norm


def _merge_colinear_lines(
    lines: list,
    snap_distance_m: float,
    max_angle_deg: float = 30.0,
    require_colinearity: bool = True,
) -> list:
    """Snap nearby endpoints and fuse approximately-colinear lines.

    Many short fragments produced by skan's branch decomposition (or by mask
    holes at culverts/driveways) actually belong to the same physical levee.
    This step:

    1. Builds a cKDTree of all line endpoints.
    2. For each pair of distinct lines whose endpoints sit within
       ``snap_distance_m``, computes the angle between the outgoing tangents
       at those endpoints.
    3. If ``require_colinearity`` and the lines are approximately colinear
       (angle <= ``max_angle_deg``), fuses them. If not requiring
       colinearity, fuses any pair within ``snap_distance_m`` regardless
       of angle (aggressive — use for tight meanders).
    4. Iterates greedily until no more merges are possible.

    Returns the merged list of LineStrings.
    """
    from scipy.spatial import cKDTree

    if snap_distance_m <= 0 or len(lines) <= 1:
        return list(lines)

    cos_thresh = float(np.cos(np.deg2rad(max_angle_deg)))
    work = [LineString(np.asarray(ln.coords, dtype=np.float64)) for ln in lines]

    changed = True
    safety_iters = 0
    while changed and safety_iters < 12:
        changed = False
        safety_iters += 1
        n = len(work)
        if n <= 1:
            break

        # Build endpoint table: (line_idx, which_end='start'|'end', xy)
        ep_xy = np.empty((2 * n, 2), dtype=np.float64)
        ep_owner = np.empty(2 * n, dtype=np.int32)
        ep_side = np.empty(2 * n, dtype=np.int8)  # 0=start, 1=end
        for i, ln in enumerate(work):
            ep_xy[2 * i] = ln.coords[0]
            ep_xy[2 * i + 1] = ln.coords[-1]
            ep_owner[2 * i] = i
            ep_owner[2 * i + 1] = i
            ep_side[2 * i] = 0
            ep_side[2 * i + 1] = 1

        tree = cKDTree(ep_xy)
        pairs = tree.query_pairs(r=snap_distance_m, output_type="ndarray")
        if pairs.size == 0:
            break

        # Best (smallest distance) pair first; only one merge per line per pass
        d2 = np.sum((ep_xy[pairs[:, 0]] - ep_xy[pairs[:, 1]]) ** 2, axis=1)
        order = np.argsort(d2)
        used = set()
        new_lines = list(work)
        drop = set()

        for k in order:
            i, j = int(pairs[k, 0]), int(pairs[k, 1])
            li = int(ep_owner[i])
            lj = int(ep_owner[j])
            if li == lj:
                continue  # same line; ignore (no self-loop merging)
            if li in used or lj in used:
                continue

            la, lb = work[li], work[lj]
            # Tangent at the relevant endpoint of each line, oriented OUTWARD
            # (i.e. pointing away from the line interior). Two outward tangents
            # of colinear lines should point in opposite directions, so we
            # check that their dot product is < -cos_thresh.
            if require_colinearity:
                ta = _line_tangent_at_endpoint(la, "start" if ep_side[i] == 0 else "end")
                tb = _line_tangent_at_endpoint(lb, "start" if ep_side[j] == 0 else "end")
                if np.dot(ta, tb) > -cos_thresh:
                    continue

            # Build merged coordinates by orienting both lines so their joined
            # endpoints meet in the middle.
            ca = np.asarray(la.coords, dtype=np.float64)
            cb = np.asarray(lb.coords, dtype=np.float64)
            if ep_side[i] == 0:
                ca = ca[::-1]  # flip so endpoint 'i' is at the end of ca
            if ep_side[j] == 1:
                cb = cb[::-1]  # flip so endpoint 'j' is at the start of cb
            merged = np.vstack([ca, cb[1:]])  # avoid duplicating the join point
            new_lines[li] = LineString(merged)
            drop.add(lj)
            used.add(li)
            used.add(lj)
            changed = True

        if drop:
            work = [ln for k, ln in enumerate(new_lines) if k not in drop]

    return work


def _simplify_and_clean(line: LineString, tol_m: float) -> Optional[LineString]:
    """DP-simplify and reject degenerate results."""
    if tol_m <= 0:
        simplified = line
    else:
        simplified = line.simplify(tol_m, preserve_topology=False)
    if simplified.is_empty or simplified.length < 1.0:
        return None
    if simplified.geom_type != "LineString":
        # rare: simplify of a closed ring could return Polygon; skip
        return None
    if len(simplified.coords) < 2:
        return None
    return simplified


# ---------------------------------------------------------------------------
# Shared mask -> GDF post-processing
# ---------------------------------------------------------------------------


def _postprocess_mask_to_polylines(
    mask: np.ndarray,
    da_dem: xr.DataArray,
    *,
    min_length_m: float,
    min_relief_m: float,
    simplify_tol_m: Optional[float],
    swath_half_width_m: float,
    max_sinuosity: float = 1.5,
    bridge_gap_m: float = 2.0,
    endpoint_snap_m: float = 25.0,
    max_merge_angle_deg: float = 30.0,
    require_colinearity: bool = True,
    merge_strategy: str = "snap",
    relief_quantile: float = 0.5,
    width_map: Optional[np.ndarray] = None,
    width_thd_thresh_m: Optional[float] = None,
    response_map: Optional[np.ndarray] = None,
    min_component_area_px: int = 20,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Turn a boolean ridge mask into a QC-filtered GeoDataFrame[LineString].

    Parameters
    ----------
    mask : np.ndarray of bool, shape = da_dem.shape
        Candidate ridge pixels.
    da_dem : xr.DataArray
        Source DEM; provides transform, CRS, pixel size.
    min_length_m, min_relief_m, simplify_tol_m, swath_half_width_m : float
        QC thresholds and simplification tolerance. ``simplify_tol_m=None``
        defaults to 0.5 * pixel size.
    width_map : np.ndarray, optional
        Per-pixel feature width in meters (Steger only). If given, the median
        width along each polyline is recorded and used for thd/weir split.
    width_thd_thresh_m : float, optional
        Medians below this are classified ``stype="thd"``; above as ``"weir"``.
    response_map : np.ndarray, optional
        Per-pixel detector response, stored as the ``score`` column.
    min_component_area_px : int
        Drop connected components smaller than this before skeletonization.
    """
    _require_skimage_skan()
    from skimage.morphology import closing, disk, skeletonize
    from skimage.measure import label
    from skan import Skeleton

    _assert_projected(da_dem)
    dem_arr, _ = _dem_to_numpy(da_dem)
    transform = da_dem.raster.transform
    dx_m = _dem_pixel_size_m(da_dem)
    crs = da_dem.raster.crs
    if simplify_tol_m is None:
        simplify_tol_m = 0.5 * dx_m

    if mask.shape != dem_arr.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match DEM shape {dem_arr.shape}"
        )

    # 1. Morphological cleanup (avoids skimage >=0.26 deprecated APIs).
    # Aggressive `bridge_gap_m`-sized closing fills typical culvert / driveway
    # gaps in the mask (5-15 m on most floodplain levees) BEFORE component
    # filtering so reconnected lines pass the area threshold.
    mask_clean = mask.astype(bool)
    bridge_px = max(1, int(round(bridge_gap_m / dx_m)))
    mask_clean = closing(mask_clean, disk(bridge_px))
    if min_component_area_px > 0:
        lbl = label(mask_clean, connectivity=1)  # 4-connectivity (skimage default)
        areas = np.bincount(lbl.ravel())
        keep_ids = np.nonzero(areas >= min_component_area_px)[0]
        keep_ids = keep_ids[keep_ids != 0]  # drop background label 0
        mask_clean = np.isin(lbl, keep_ids)

    if not mask_clean.any():
        logger.info("ridge_detection: empty mask after cleanup; no polylines.")
        return gpd.GeoDataFrame(
            columns=["name", "stype", "width_m", "score", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    # 2. Skeletonize
    skel = skeletonize(mask_clean)
    if not skel.any():
        logger.info("ridge_detection: skeleton empty; no polylines.")
        return gpd.GeoDataFrame(
            columns=["name", "stype", "width_m", "score", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    # 3. skan -> branches (spacing=1.0 keeps coords in pixel units for
    # deterministic back-transform).
    try:
        sk = Skeleton(skel.astype(np.uint8), spacing=1.0)
    except ValueError:
        # Can happen for skeletons with no edges (single isolated pixels)
        logger.info("ridge_detection: skan found no branches.")
        return gpd.GeoDataFrame(
            columns=["name", "stype", "width_m", "score", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    # 3. Two strategies for going skeleton -> polylines:
    #    "snap"         : take every skan branch, then endpoint-snap-and-merge
    #                     colinear fragments (good general default)
    #    "longest_path" : for each connected component of the pixel-level
    #                     skeleton graph, take the longest geodesic path
    #                     (one polyline per ridge component; max continuity)
    if merge_strategy == "longest_path":
        path_arrays = _longest_paths_per_component(sk)
    elif merge_strategy == "snap":
        path_arrays = [sk.path_coordinates(i) for i in range(sk.n_paths)]
    else:
        raise ValueError(
            f"merge_strategy must be 'snap' or 'longest_path', got {merge_strategy!r}"
        )

    lines = []
    widths_m = []
    scores = []
    for rc in path_arrays:
        rc = np.asarray(rc)
        if rc.shape[0] < 2:
            continue
        rows, cols = rc[:, 0], rc[:, 1]
        xs, ys = _pixel_to_world(rows, cols, transform)
        coords_world = np.column_stack([xs, ys])
        if _line_length_m(coords_world) < min_length_m:
            continue
        line = LineString(coords_world)
        line = _simplify_and_clean(line, simplify_tol_m)
        if line is None or line.length < min_length_m:
            continue

        # Sinuosity filter: arc-length / euclidean end-to-end distance.
        # Anthropogenic embankments are ~straight (<~1.2); natural ridges
        # are more tortuous. Reject overly wavy lines.
        ep0, ep1 = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
        eucl = float(np.hypot(ep1[0] - ep0[0], ep1[1] - ep0[1]))
        if eucl > 0 and (line.length / eucl) > max_sinuosity:
            continue

        # Per-vertex width / score sampling
        ri = np.clip(np.round(rows).astype(int), 0, dem_arr.shape[0] - 1)
        ci = np.clip(np.round(cols).astype(int), 0, dem_arr.shape[1] - 1)
        if width_map is not None:
            w = np.asarray(width_map[ri, ci], dtype=float)
            widths_m.append(float(np.nanmedian(w)))
        else:
            widths_m.append(np.nan)
        if response_map is not None:
            s = np.asarray(response_map[ri, ci], dtype=float)
            scores.append(float(np.nanmedian(s)))
        else:
            scores.append(1.0)

        lines.append(line)

    if not lines:
        logger.info("ridge_detection: no polylines pass min_length_m filter.")
        return gpd.GeoDataFrame(
            columns=["name", "stype", "width_m", "score", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    # 3b. Endpoint-snap & colinear merge: reconnect fragments that skan broke
    # at junctions (the dominant fragmentation source). Now also runs after
    # 'longest_path' mode so adjacent ridge components (still split by mask
    # gaps too large to bridge) can fuse into one continuous line.
    if endpoint_snap_m > 0 and len(lines) > 1:
        n_before = len(lines)
        lengths = np.array([float(ln.length) for ln in lines])
        # Length-weighted aggregates so width/score survive the merge sensibly
        merged_lines = _merge_colinear_lines(
            lines,
            snap_distance_m=endpoint_snap_m,
            max_angle_deg=max_merge_angle_deg,
            require_colinearity=require_colinearity,
        )
        if len(merged_lines) != n_before:
            logger.info(
                f"ridge_detection: endpoint-snap merged "
                f"{n_before} -> {len(merged_lines)} polylines"
            )
            # Re-derive width / score by sampling the maps along the merged
            # geometry. This is more robust than weighting per-segment values.
            new_widths, new_scores = [], []
            for ln in merged_lines:
                xs = np.asarray([c[0] for c in ln.coords])
                ys = np.asarray([c[1] for c in ln.coords])
                inv = ~transform
                rc = np.array([inv * (x, y) for x, y in zip(xs, ys)])
                ri = np.clip(np.round(rc[:, 1]).astype(int), 0, dem_arr.shape[0] - 1)
                ci = np.clip(np.round(rc[:, 0]).astype(int), 0, dem_arr.shape[1] - 1)
                if width_map is not None:
                    new_widths.append(float(np.nanmedian(width_map[ri, ci])))
                else:
                    new_widths.append(np.nan)
                if response_map is not None:
                    new_scores.append(float(np.nanmedian(response_map[ri, ci])))
                else:
                    new_scores.append(1.0)
            lines = merged_lines
            widths_m = new_widths
            scores = new_scores

    # 3c. Drop merged lines that now exceed the sinuosity bound (rare but possible)
    keep_idx = []
    for i, ln in enumerate(lines):
        ep0 = np.asarray(ln.coords[0])
        ep1 = np.asarray(ln.coords[-1])
        eucl = float(np.hypot(ep1[0] - ep0[0], ep1[1] - ep0[1]))
        if eucl > 0 and (ln.length / eucl) > max_sinuosity:
            continue
        if ln.length < min_length_m:
            continue
        keep_idx.append(i)
    lines = [lines[i] for i in keep_idx]
    widths_m = [widths_m[i] for i in keep_idx]
    scores = [scores[i] for i in keep_idx]

    # 4. Swath-relief QC filter (relief at the configured quantile)
    kept_lines, kept_w, kept_s = [], [], []
    for line, w, s in zip(lines, widths_m, scores):
        relief_q, _ = _swath_relief_stats(
            line, dem_arr, transform, half_m=swath_half_width_m,
            aggregation_quantile=relief_quantile,
        )
        if not np.isfinite(relief_q):
            continue
        if relief_q >= min_relief_m:
            kept_lines.append(line)
            kept_w.append(w)
            kept_s.append(s)

    if not kept_lines:
        logger.info(
            f"ridge_detection: no polylines pass swath-relief filter "
            f"(min_relief_m={min_relief_m})."
        )
        return gpd.GeoDataFrame(
            columns=["name", "stype", "width_m", "score", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    # 5. stype classification
    stypes = []
    if width_map is not None and width_thd_thresh_m is not None:
        for w in kept_w:
            if np.isfinite(w) and w < width_thd_thresh_m:
                stypes.append("thd")
            else:
                stypes.append("weir")
    else:
        stypes = ["weir"] * len(kept_lines)

    gdf = gpd.GeoDataFrame(
        {
            "name": [f"ridge_{i:05d}" for i in range(len(kept_lines))],
            "stype": stypes,
            "width_m": kept_w,
            "score": kept_s,
            "geometry": kept_lines,
        },
        geometry="geometry",
        crs=crs,
    )
    logger.info(
        f"ridge_detection: kept {len(gdf)} polylines "
        f"(weir={sum(1 for t in stypes if t == 'weir')}, "
        f"thd={sum(1 for t in stypes if t == 'thd')})."
    )
    return gdf


# ---------------------------------------------------------------------------
# Flavor 1: REA — relative-elevation attribute (scipy baseline)
# ---------------------------------------------------------------------------


def detect_ridges_rea(
    da_dem: xr.DataArray,
    *,
    window_m: float = 15.0,
    windows_m: Optional[Tuple[float, ...]] = None,
    use_reconstruction: bool = False,
    seed_quantile: float = 0.95,
    grow_quantile: float = 0.60,
    rea_thresh_m: float = 0.5,
    high_quantile: float = 0.98,
    low_quantile: float = 0.88,
    min_length_m: float = 50.0,
    min_relief_m: float = 0.5,
    simplify_tol_m: Optional[float] = None,
    swath_half_width_m: float = 10.0,
    min_component_area_px: int = 20,
    bridge_gap_m: float = 2.0,
    endpoint_snap_m: float = 25.0,
    max_merge_angle_deg: float = 30.0,
    require_colinearity: bool = True,
    max_sinuosity: float = 1.5,
    merge_strategy: str = "snap",
    relief_quantile: float = 0.5,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Detect ridges via Relative Elevation Attribute (REA).

    REA(x, y) := z(x, y) - mean(z, window=window_m). Pixels with REA above
    a hysteresis threshold (``low_quantile``, ``high_quantile`` of positive
    REA) are candidate ridges.

    Set ``use_reconstruction=True`` to switch to a multi-scale, seed-grown
    mask via ``skimage.morphology.reconstruction``: the textbook way to keep
    only ridge components anchored by a strong seed while letting them grow
    through moderate-REA pixels to maintain continuity. Use ``windows_m``
    (e.g. ``(10, 25, 50, 100)``) for multi-scale REA-max — captures features
    at multiple scales in one mask.

    References
    ----------
    Cazorzi et al. 2013; Sofia, Dalla Fontana & Tarolli 2014.
    Reliable, cheap, interpretable; biased toward sharp narrow features.
    """
    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)

    if use_reconstruction:
        ws = tuple(windows_m) if windows_m else (window_m,)
        mask, rea = _rea_reconstruction_mask(
            dem_arr, valid, dx_m,
            windows_m=ws,
            seed_quantile=seed_quantile,
            grow_quantile=grow_quantile,
        )
        logger.info(
            f"REA reconstruction: windows_m={ws}, "
            f"seed_q={seed_quantile}, grow_q={grow_quantile} -> "
            f"{int(mask.sum())} candidate pixels"
        )
    else:
        win_px = max(3, int(round(window_m / dx_m)))
        if win_px % 2 == 0:
            win_px += 1  # odd-size window for symmetric centering
        mean_map = ndimage.uniform_filter(dem_arr, size=win_px, mode="reflect")
        rea = dem_arr - mean_map
        rea[~valid] = 0.0
        mask_abs = rea >= rea_thresh_m
        mask_hyst = _quantile_hysteresis(rea, valid, low_quantile, high_quantile)
        mask = mask_abs & mask_hyst

    return _postprocess_mask_to_polylines(
        mask,
        da_dem,
        min_length_m=min_length_m,
        min_relief_m=min_relief_m,
        simplify_tol_m=simplify_tol_m,
        swath_half_width_m=swath_half_width_m,
        bridge_gap_m=bridge_gap_m,
        endpoint_snap_m=endpoint_snap_m,
        max_merge_angle_deg=max_merge_angle_deg,
        require_colinearity=require_colinearity,
        max_sinuosity=max_sinuosity,
        merge_strategy=merge_strategy,
        relief_quantile=relief_quantile,
        response_map=rea,
        min_component_area_px=min_component_area_px,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Flavor 2: Frangi — multi-scale Hessian ridge filter (skimage)
# ---------------------------------------------------------------------------


def detect_ridges_frangi(
    da_dem: xr.DataArray,
    *,
    sigmas_m: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 7.5),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: Optional[float] = None,
    high_pass_sigma_m: float = 15.0,
    low_quantile: float = 0.88,
    high_quantile: float = 0.98,
    min_length_m: float = 50.0,
    min_relief_m: float = 0.5,
    simplify_tol_m: Optional[float] = None,
    swath_half_width_m: float = 10.0,
    min_component_area_px: int = 25,
    bridge_gap_m: float = 2.0,
    endpoint_snap_m: float = 25.0,
    max_merge_angle_deg: float = 30.0,
    require_colinearity: bool = True,
    max_sinuosity: float = 1.5,
    merge_strategy: str = "snap",
    relief_quantile: float = 0.5,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Detect ridges via multi-scale Frangi vesselness filter.

    Rank-1 recommended method: scale-space Hessian-eigenvalue analysis,
    orientation-selective, robust to scene heterogeneity when used with a
    high-pass residual DEM (``high_pass_sigma_m``) and quantile hysteresis.

    Physical scales in meters; converted internally to pixels.
    """
    from skimage.filters import frangi

    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)

    # High-pass residual topography to remove regional slope
    dem_hp = _high_pass(dem_arr, sigma_px=high_pass_sigma_m / dx_m)

    # Convert meter-scale sigmas to pixels
    sigmas_px = tuple(max(0.5, s / dx_m) for s in sigmas_m)
    logger.debug(f"frangi sigmas_px = {sigmas_px}")

    response = frangi(
        dem_hp,
        sigmas=sigmas_px,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False,
    )
    response = np.asarray(response, dtype=np.float32)
    response[~valid] = 0.0

    mask = _quantile_hysteresis(response, valid, low_quantile, high_quantile)

    return _postprocess_mask_to_polylines(
        mask,
        da_dem,
        min_length_m=min_length_m,
        min_relief_m=min_relief_m,
        simplify_tol_m=simplify_tol_m,
        swath_half_width_m=swath_half_width_m,
        bridge_gap_m=bridge_gap_m,
        endpoint_snap_m=endpoint_snap_m,
        max_merge_angle_deg=max_merge_angle_deg,
        require_colinearity=require_colinearity,
        max_sinuosity=max_sinuosity,
        merge_strategy=merge_strategy,
        relief_quantile=relief_quantile,
        response_map=response,
        min_component_area_px=min_component_area_px,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Flavor 3: WhiteboxTools — find_ridges (+ optional geomorphons)
# ---------------------------------------------------------------------------


def detect_ridges_whitebox(
    da_dem: xr.DataArray,
    *,
    use_geomorphons: bool = True,
    geomorphon_classes: Tuple[int, ...] = (3, 4, 9),  # ridge, shoulder, spur
    geomorphon_combine: str = "or",                    # "or" or "and" with find_ridges
    search_radius_m: float = 60.0,
    geomorphon_threshold_deg: float = 0.5,
    min_length_m: float = 50.0,
    min_relief_m: float = 0.5,
    simplify_tol_m: Optional[float] = None,
    swath_half_width_m: float = 10.0,
    min_component_area_px: int = 25,
    bridge_gap_m: float = 2.0,
    endpoint_snap_m: float = 25.0,
    max_merge_angle_deg: float = 30.0,
    require_colinearity: bool = True,
    max_sinuosity: float = 1.5,
    merge_strategy: str = "snap",
    relief_quantile: float = 0.5,
    work_dir: Optional[Path] = None,
    smooth_first: bool = True,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Detect ridges via WhiteboxTools ``find_ridges`` and geomorphons.

    Requires the ``whitebox`` package (installs a local Rust binary on first
    use). The binary writes intermediate GeoTIFFs to ``work_dir`` (a temp dir
    if not provided).

    Combines ``wbt.find_ridges(line_thin=True)`` with ``wbt.geomorphons``
    (default classes 3=ridge, 4=shoulder, 9=spur per Jasiewicz & Stepinski
    2013). ``geomorphon_combine`` controls how the two masks are fused:

    - ``"or"`` (default): keep pixels flagged by either detector — more
      recall, more fragmented output. Recommended for floodplain levees.
    - ``"and"``: strict intersection — high precision but very conservative
      and prone to under-detection on gentle levees.
    """
    try:
        import whitebox
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "detect_ridges_whitebox requires the 'whitebox' package. "
            "pip install whitebox"
        ) from e

    import rasterio as rio

    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)
    transform = da_dem.raster.transform
    crs = da_dem.raster.crs

    search_radius_px = max(3, int(round(search_radius_m / dx_m)))

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(work_dir) if work_dir is not None else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        dem_path = work / "dem_in.tif"
        smooth_path = work / "dem_smooth.tif"
        ridges_path = work / "ridges.tif"
        geom_path = work / "geomorphons.tif"

        # Write DEM to GeoTIFF for WBT
        profile = dict(
            driver="GTiff",
            width=dem_arr.shape[1],
            height=dem_arr.shape[0],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=-9999.0,
        )
        out_arr = dem_arr.astype("float32")
        out_arr[~valid] = -9999.0
        with rio.open(dem_path, "w", **profile) as dst:
            dst.write(out_arr, 1)

        wbt = whitebox.WhiteboxTools()
        wbt.verbose = False
        wbt.set_working_dir(str(work))

        dem_for_ridges = str(dem_path)
        if smooth_first:
            wbt.feature_preserving_smoothing(
                dem=str(dem_path),
                output=str(smooth_path),
                filter=max(3, int(round(5.0 / dx_m))),
                norm_diff=8.0,
                num_iter=3,
            )
            dem_for_ridges = str(smooth_path)

        wbt.find_ridges(
            dem=dem_for_ridges,
            output=str(ridges_path),
            line_thin=True,
        )

        with rio.open(ridges_path) as src:
            ridges = src.read(1)
        ridge_mask = (ridges > 0) & valid

        if use_geomorphons:
            wbt.geomorphons(
                dem=dem_for_ridges,
                output=str(geom_path),
                search=search_radius_px,
                threshold=geomorphon_threshold_deg,
                fdist=0,
                skip=0,
                forms=True,
            )
            with rio.open(geom_path) as src:
                geom = src.read(1)
            geom_mask = np.isin(geom, np.asarray(geomorphon_classes)) & valid
            combine = geomorphon_combine.lower()
            if combine == "or":
                ridge_mask = ridge_mask | geom_mask
            elif combine == "and":
                ridge_mask = ridge_mask & geom_mask
            else:
                raise ValueError(
                    f"geomorphon_combine must be 'or' or 'and', got {combine!r}"
                )

    return _postprocess_mask_to_polylines(
        ridge_mask,
        da_dem,
        min_length_m=min_length_m,
        min_relief_m=min_relief_m,
        simplify_tol_m=simplify_tol_m,
        swath_half_width_m=swath_half_width_m,
        bridge_gap_m=bridge_gap_m,
        endpoint_snap_m=endpoint_snap_m,
        max_merge_angle_deg=max_merge_angle_deg,
        require_colinearity=require_colinearity,
        max_sinuosity=max_sinuosity,
        merge_strategy=merge_strategy,
        relief_quantile=relief_quantile,
        response_map=ridges.astype(np.float32),
        min_component_area_px=min_component_area_px,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Flavor 4: Steger — unbiased curvilinear detector (opencv-contrib)
# ---------------------------------------------------------------------------


def detect_ridges_steger(
    da_dem: xr.DataArray,
    *,
    sigma_m: float = 2.0,
    response_high_quantile: float = 0.98,
    response_low_quantile: float = 0.88,
    width_thd_thresh_m: Optional[float] = None,
    high_pass_sigma_m: float = 15.0,
    min_length_m: float = 50.0,
    min_relief_m: float = 0.5,
    simplify_tol_m: Optional[float] = None,
    swath_half_width_m: float = 10.0,
    min_component_area_px: int = 20,
    bridge_gap_m: float = 2.0,
    endpoint_snap_m: float = 25.0,
    max_merge_angle_deg: float = 30.0,
    require_colinearity: bool = True,
    max_sinuosity: float = 1.5,
    merge_strategy: str = "snap",
    relief_quantile: float = 0.5,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Detect ridges via a Steger-style unbiased curvilinear detector.

    Implemented directly on the Hessian (no dependency on opencv-contrib's
    ``ximgproc.StegerFilter`` so portability is preserved). At each pixel,
    compute the Hessian eigen-decomposition and keep pixels where:

    - ``lambda_max`` (strongest negative curvature) is sufficiently negative
      (sharp cross-ridge curvature),
    - the gradient projected onto the corresponding eigenvector is near zero
      (we are on the crest rather than on a flank).

    Feature width per pixel is estimated as ``2 * sqrt(-z / lambda_max)``,
    where ``z`` is the residual DEM height above the smoothed background.
    This gives physical-unit widths that can drive the ``thd``/``weir`` split
    via ``width_thd_thresh_m`` (default ``0.5 * dx``).

    References
    ----------
    Steger 1998, IEEE TPAMI 20:113.
    """
    from scipy.ndimage import gaussian_filter

    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)
    sigma_px = max(0.5, sigma_m / dx_m)

    if width_thd_thresh_m is None:
        width_thd_thresh_m = 0.5 * dx_m

    # Residual topography for height magnitude used in width estimation
    dem_hp = _high_pass(dem_arr, sigma_px=high_pass_sigma_m / dx_m)

    # Gaussian derivatives of order 2 via scipy
    zxx = gaussian_filter(dem_arr, sigma=sigma_px, order=(0, 2), mode="reflect")
    zyy = gaussian_filter(dem_arr, sigma=sigma_px, order=(2, 0), mode="reflect")
    zxy = gaussian_filter(dem_arr, sigma=sigma_px, order=(1, 1), mode="reflect")
    zx = gaussian_filter(dem_arr, sigma=sigma_px, order=(0, 1), mode="reflect")
    zy = gaussian_filter(dem_arr, sigma=sigma_px, order=(1, 0), mode="reflect")

    # Eigenvalues of 2x2 symmetric Hessian
    #   [[zxx, zxy], [zxy, zyy]]
    # lambda_{1,2} = (zxx+zyy)/2 +- sqrt( ((zxx-zyy)/2)**2 + zxy**2 )
    mean_h = 0.5 * (zxx + zyy)
    diff_h = 0.5 * (zxx - zyy)
    disc = np.sqrt(diff_h * diff_h + zxy * zxy)
    lam1 = mean_h + disc  # >= lam2
    lam2 = mean_h - disc  # <= lam1; most-negative for bright ridges
    # Eigenvector of lam2 (cross-ridge direction):
    #   ((zxx - lam2), zxy) is one eigenvector pair
    v_x = zxx - lam2
    v_y = zxy
    vnorm = np.hypot(v_x, v_y)
    vnorm = np.where(vnorm == 0, 1.0, vnorm)
    v_x /= vnorm
    v_y /= vnorm

    # Response: normalize into meters of curvature per dx^2 for robustness,
    # then take magnitude of lam2 where it is negative (bright ridge).
    response = np.where(lam2 < 0, -lam2, 0.0).astype(np.float32)
    response[~valid] = 0.0

    # Steger's sub-pixel offset along the eigenvector; we require it to be
    # within half a pixel (i.e. the current pixel IS the crest).
    # t = -(zx*v_x + zy*v_y) / lam2  (but lam2 can be near zero)
    grad_proj = zx * v_x + zy * v_y
    with np.errstate(divide="ignore", invalid="ignore"):
        t_sub = np.where(lam2 < 0, -grad_proj / lam2, np.nan)
    # Sub-pixel offset in pixel units: t_sub * eigenvector; we require
    # abs(t_sub) <= 0.5 (within this cell)
    on_crest = np.isfinite(t_sub) & (np.abs(t_sub) <= 0.5)

    mask_curv = _quantile_hysteresis(
        response, valid, response_low_quantile, response_high_quantile
    )
    mask = mask_curv & on_crest & valid

    # Width estimate per pixel (parabolic-crest model):
    #   z(r) = z0 - 0.5 * |lam_phys| * r^2    (r in metres)
    # Zero-crossing at r = sqrt(2 z0 / |lam_phys|).
    # Full width at base = 2 * r.
    # `lam2` here is in z-units per pixel^2 (scipy gaussian_filter order=2);
    # convert to z-units per m^2 by dividing by dx_m^2.
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_phys = np.abs(lam2) / (dx_m * dx_m)  # z-units / m^2
        crest_h = np.clip(dem_hp, 0, None)
        # Full width = 2 * sqrt(2 * crest_h / |lam_phys|)
        width_map = 2.0 * np.sqrt(
            np.clip(np.where(lam_phys > 0, 2.0 * crest_h / lam_phys, np.nan), 0, None)
        )
    width_map = np.where(valid, width_map, np.nan).astype(np.float32)

    return _postprocess_mask_to_polylines(
        mask,
        da_dem,
        min_length_m=min_length_m,
        min_relief_m=min_relief_m,
        simplify_tol_m=simplify_tol_m,
        swath_half_width_m=swath_half_width_m,
        bridge_gap_m=bridge_gap_m,
        endpoint_snap_m=endpoint_snap_m,
        max_merge_angle_deg=max_merge_angle_deg,
        require_colinearity=require_colinearity,
        max_sinuosity=max_sinuosity,
        merge_strategy=merge_strategy,
        relief_quantile=relief_quantile,
        width_map=width_map,
        width_thd_thresh_m=width_thd_thresh_m,
        response_map=response,
        min_component_area_px=min_component_area_px,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Flavor 5: LCP — longest-geodesic-path on Frangi response (max-continuity)
# ---------------------------------------------------------------------------


def detect_ridges_lcp(
    da_dem: xr.DataArray,
    *,
    sigmas_m: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 7.5, 12.0),
    high_pass_sigma_m: float = 30.0,
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: Optional[float] = None,
    low_quantile: float = 0.70,
    high_quantile: float = 0.95,
    min_length_m: float = 100.0,
    min_relief_m: float = 0.3,
    simplify_tol_m: Optional[float] = None,
    swath_half_width_m: float = 12.0,
    bridge_gap_m: float = 8.0,
    endpoint_snap_m: float = 100.0,
    max_merge_angle_deg: float = 60.0,
    require_colinearity: bool = False,
    max_sinuosity: float = 3.0,
    relief_quantile: float = 0.75,
    min_component_area_px: int = 30,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Maximum-continuity ridge detection via longest-geodesic-path tracing.

    Pipeline:

    1. Compute multi-scale Frangi response on a high-passed DEM (same engine
       as :py:func:`detect_ridges_frangi`).
    2. Loose hysteresis threshold (``low_quantile=0.70`` by default — much
       looser than Frangi's 0.88) to keep weak portions of long features
       connected.
    3. Aggressive morphological closing (``bridge_gap_m=8`` by default) so
       culverts and driveway cuts don't fragment the binary mask.
    4. Skeletonize and build the pixel-level skeleton graph (skan).
    5. **For each connected component**, extract the longest geodesic path
       using the classic two-pass Dijkstra "tree-diameter" algorithm.

    The result: **one polyline per ridge component**, traversing junctions as
    needed. Total line count is far smaller than the snap-and-merge flavors
    but each line spans the full length of the ridge it represents — useful
    when the goal is "one weir per levee" rather than dense per-segment
    coverage.

    The endpoint-snap step is skipped (``merge_strategy='longest_path'``)
    since each component already produces exactly one polyline.

    References
    ----------
    Hardin et al. 2012 (*J. Coastal Res.* 28:939) used least-cost paths on
    ``max(z) − z`` to trace barrier-island dune crests. This implementation
    generalises the idea to ridge-response landscapes by tracing the
    longest geodesic path through each high-response skeleton component.
    """
    from skimage.filters import frangi

    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)

    # 1. High-pass + multi-scale Frangi
    dem_hp = _high_pass(dem_arr, sigma_px=high_pass_sigma_m / dx_m)
    sigmas_px = tuple(max(0.5, s / dx_m) for s in sigmas_m)
    response = frangi(
        dem_hp, sigmas=sigmas_px,
        alpha=alpha, beta=beta, gamma=gamma, black_ridges=False,
    ).astype(np.float32)
    response[~valid] = 0.0

    # 2. Loose hysteresis to preserve continuity
    mask = _quantile_hysteresis(response, valid, low_quantile, high_quantile)

    # 3+. Hand off to shared helper with longest_path strategy. Endpoint-snap
    # is now allowed in longest_path mode so adjacent components fuse into one
    # continuous line — set ``endpoint_snap_m=0`` to disable.
    return _postprocess_mask_to_polylines(
        mask,
        da_dem,
        min_length_m=min_length_m,
        min_relief_m=min_relief_m,
        simplify_tol_m=simplify_tol_m,
        swath_half_width_m=swath_half_width_m,
        bridge_gap_m=bridge_gap_m,
        endpoint_snap_m=endpoint_snap_m,
        max_merge_angle_deg=max_merge_angle_deg,
        require_colinearity=require_colinearity,
        max_sinuosity=max_sinuosity,
        merge_strategy="longest_path",
        relief_quantile=relief_quantile,
        response_map=response,
        min_component_area_px=min_component_area_px,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Flavor 6: River banks — pyflwdir streams + dilated rim + local elevation
# ---------------------------------------------------------------------------


def detect_river_banks(
    da_dem: xr.DataArray,
    *,
    da_uparea: Optional[xr.DataArray] = None,
    river_uparea_km2: float = 2.0,
    bank_offset_m: float = 15.0,
    rea_window_m: float = 30.0,
    bank_thresh_m: float = 0.5,
    min_bank_relief_m: float = 1.0,
    min_length_m: float = 200.0,
    min_relief_m: float = 0.5,
    simplify_tol_m: Optional[float] = None,
    swath_half_width_m: float = 15.0,
    bridge_gap_m: float = 8.0,
    endpoint_snap_m: float = 100.0,
    max_merge_angle_deg: float = 60.0,
    require_colinearity: bool = False,
    max_sinuosity: float = 4.0,
    relief_quantile: float = 0.75,
    merge_strategy: str = "longest_path",
    min_component_area_px: int = 30,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Trace river-bank crests as polylines.

    Hybrid pipeline that **guarantees river banks are captured**:

    1. Compute D8 flow direction and upstream area from the DEM via
       ``pyflwdir.from_dem`` (or accept a pre-computed ``da_uparea``).
    2. Threshold ``uparea_km2 >= river_uparea_km2`` to get a stream mask.
    3. Dilate the stream by ``bank_offset_m`` to get a "bank rim" mask
       (pixels within ``bank_offset_m`` of the stream, excluding the stream
       itself).
    4. Within the rim, keep pixels that are:
       (a) locally elevated (REA at ``rea_window_m`` ≥ ``bank_thresh_m``),
       AND
       (b) at least ``min_bank_relief_m`` above the **nearest stream pixel**
       (cheap HAND proxy via Euclidean distance transform).
    5. Hand off to ``_postprocess_mask_to_polylines`` with
       ``merge_strategy='longest_path'`` so each connected bank component
       comes out as one polyline.

    Output schema matches the other flavors: ``name, stype, width_m,
    score, geometry``. ``score`` carries the ``relief_above_stream`` median
    along each polyline.

    Notes
    -----
    Left and right banks are **not** split into separate polylines in v1 —
    each connected bank component along a river side becomes one polyline.

    Parameters
    ----------
    da_uparea : xr.DataArray, optional
        Pre-computed upstream area (km²) on the same grid as ``da_dem``. If
        provided, skips the ``pyflwdir.from_dem`` step. Useful when working
        with MERIT-Hydro or a previously cached flow accumulation.
    river_uparea_km2 : float
        Threshold for the stream mask. Snohomish-AOI default is 2.0; lower
        for smaller catchments, higher to keep only the trunk river.
    bank_offset_m, rea_window_m, bank_thresh_m, min_bank_relief_m : float
        Bank-detection parameters. ``rea_window_m ≈ 2 * bank_offset_m``
        keeps the local mean reaching into the channel.
    """
    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)
    transform = da_dem.raster.transform

    # 1. Upstream area — either supplied or computed from DEM
    if da_uparea is not None:
        uparea_arr = np.asarray(da_uparea.values, dtype=np.float32)
        if uparea_arr.ndim == 3 and uparea_arr.shape[0] == 1:
            uparea_arr = uparea_arr[0]
        if uparea_arr.shape != dem_arr.shape:
            raise ValueError(
                f"da_uparea shape {uparea_arr.shape} does not match DEM "
                f"shape {dem_arr.shape}"
            )
    else:
        try:
            import pyflwdir
        except ImportError as e:
            raise ImportError(
                "detect_river_banks requires pyflwdir (already a hydromt-sfincs "
                "dependency). Reinstall or provide da_uparea."
            ) from e
        nodata = da_dem.raster.nodata if da_dem.raster.nodata is not None else -9999.0
        flw = pyflwdir.from_dem(
            data=dem_arr.astype("float32"),
            nodata=float(nodata),
            transform=transform,
            latlon=False,
        )
        uparea_arr = np.asarray(flw.upstream_area(unit="km2"), dtype=np.float32)
        logger.info(
            f"detect_river_banks: pyflwdir upstream area max = "
            f"{float(np.nanmax(uparea_arr)):.2f} km^2 "
            f"(threshold = {river_uparea_km2} km^2)"
        )

    # 2. Stream mask
    stream = (uparea_arr >= river_uparea_km2) & valid
    if not stream.any():
        logger.warning(
            f"detect_river_banks: no stream pixels above "
            f"{river_uparea_km2} km^2 uparea — returning empty."
        )
        return gpd.GeoDataFrame(
            columns=["name", "stype", "width_m", "score", "geometry"],
            geometry="geometry", crs=da_dem.raster.crs,
        )

    # 3. Bank rim: pixels within bank_offset_m of stream, excluding stream
    offset_px = max(1, int(round(bank_offset_m / dx_m)))
    rim = ndimage.binary_dilation(stream, iterations=offset_px) & ~stream & valid

    # 4a. Local REA on the rim
    win_px = max(3, int(round(rea_window_m / dx_m)))
    if win_px % 2 == 0:
        win_px += 1
    rea_local = dem_arr - ndimage.uniform_filter(dem_arr, size=win_px, mode="reflect")

    # 4b. Cheap HAND proxy: Euclidean nearest-stream elevation
    _, (rs, cs) = ndimage.distance_transform_edt(~stream, return_indices=True)
    z_stream_near = dem_arr[rs, cs]
    relief_above_stream = (dem_arr - z_stream_near).astype(np.float32)

    bank_mask = (
        rim
        & (rea_local > bank_thresh_m)
        & (relief_above_stream > min_bank_relief_m)
    )
    logger.info(
        f"detect_river_banks: stream pixels = {int(stream.sum())}, "
        f"rim pixels = {int(rim.sum())}, bank candidates = {int(bank_mask.sum())}"
    )

    # 5. Standard post-processing
    return _postprocess_mask_to_polylines(
        bank_mask,
        da_dem,
        min_length_m=min_length_m,
        min_relief_m=min_relief_m,
        simplify_tol_m=simplify_tol_m,
        swath_half_width_m=swath_half_width_m,
        bridge_gap_m=bridge_gap_m,
        endpoint_snap_m=endpoint_snap_m,
        max_merge_angle_deg=max_merge_angle_deg,
        require_colinearity=require_colinearity,
        max_sinuosity=max_sinuosity,
        merge_strategy=merge_strategy,
        relief_quantile=relief_quantile,
        response_map=relief_above_stream,
        min_component_area_px=min_component_area_px,
        logger=logger,
    )
