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
    "detect_levees_breach",
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


def _erode_valid_buffer(
    valid: np.ndarray,
    sigma_px: float,
    *,
    sigma_factor: float = 4.0,
) -> np.ndarray:
    """Erode the valid mask by ~`sigma_factor`·σ pixels via Euclidean distance.

    Used to suppress phantom signal in a Hessian/convolution response
    near NoData boundaries: ``_dem_to_numpy`` mean-fills NoData cells, and
    ``gaussian_filter(mode='reflect')`` smears that fill into a `sigma_factor`·σ
    boundary buffer. By eroding the valid mask before applying it as the
    final filter, we discard any cells whose convolution kernel touched a
    fill cell.

    Implementation: `distance_transform_edt(valid)` then threshold. This
    is O(N) and uses ~24 N bytes (vs disk-SE erosion which allocates a
    radius²-cell SE — would MemoryError on production-scale AOIs with
    σ ≥ 30 px). Default `sigma_factor=4.0` matches scipy's
    `gaussian_filter(truncate=4.0)` so the buffer covers the actual
    kernel reach (3·σ leaves ~0.27% kernel weight outside).

    Returns the eroded mask.
    """
    if not (~valid).any():
        return valid  # nothing to erode against
    radius = max(1.0, sigma_factor * float(sigma_px))
    # Distance from each cell to the nearest INVALID cell (= 0 for invalid cells).
    dist = ndimage.distance_transform_edt(valid)
    return dist >= radius


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


def _frangi_multiscale(
    image: np.ndarray,
    valid: np.ndarray,
    sigmas_px: Tuple[float, ...],
    *,
    beta: float = 0.5,
    gamma: Optional[float] = None,
    gamma_percentile: float = 99.0,
    black_ridges: bool = False,
) -> np.ndarray:
    """Multi-scale Frangi vesselness with Lindeberg σ² normalization.

    Workaround for scikit-image issue #7711: ``skimage.filters.frangi`` (as
    of v0.20+) drops the σ² scale-normalization, causing the multi-scale
    response to collapse to the smallest σ (large ridges get massively
    underweighted). This function computes the vesselness explicitly with
    proper γ-normalized Gaussian second derivatives (Lindeberg 1998), so
    levees of widely different widths get comparable response magnitudes.

    For each σ the (γ=2)-normalized Hessian is computed as
    ``H_σ(x) = σ² · ∂²G_σ ∗ I``, eigenvalues |λ₁| ≤ |λ₂| are taken in 2D, and
    the Frangi vesselness measure is evaluated:

        v(σ) = exp(-(λ₁/λ₂)² / (2β²)) · (1 − exp(-S² / (2γ²)))   if λ₂<0
             = 0                                                  if λ₂≥0
        S    = √(λ₁² + λ₂²)

    For *bright* ridges on dark background (``black_ridges=False``) we keep
    pixels where λ₂ < 0; otherwise λ₂ > 0. γ is set per-scale (Frangi 1998
    Eq. 16) to half of the maximum Hessian Frobenius norm S; we use a
    ``gamma_percentile`` of 99 by default as a robust max-proxy
    (canonical max/2 corresponds to gamma_percentile=100). Lower values
    (≈50, the median) collapse γ and saturate the structureness factor —
    avoid.

    Returns the pointwise maximum response across all sigmas.
    """
    from scipy.ndimage import gaussian_filter

    response = np.zeros_like(image, dtype=np.float64)

    img = -image if black_ridges else image  # work in "bright ridge" convention

    for sigma in sigmas_px:
        if sigma <= 0:
            continue
        # Lindeberg σ²-normalized second derivatives
        zxx = sigma * sigma * gaussian_filter(img, sigma=sigma, order=(0, 2),
                                              mode="reflect")
        zyy = sigma * sigma * gaussian_filter(img, sigma=sigma, order=(2, 0),
                                              mode="reflect")
        zxy = sigma * sigma * gaussian_filter(img, sigma=sigma, order=(1, 1),
                                              mode="reflect")
        # Eigenvalues of 2x2 symmetric Hessian (analytic)
        mean_h = 0.5 * (zxx + zyy)
        diff_h = 0.5 * (zxx - zyy)
        disc = np.sqrt(diff_h * diff_h + zxy * zxy)
        lam_a = mean_h + disc  # >= lam_b
        lam_b = mean_h - disc  # <= lam_a
        # Sort by ABS magnitude: |λ₁| ≤ |λ₂|
        lam1 = np.where(np.abs(lam_a) <= np.abs(lam_b), lam_a, lam_b)
        lam2 = np.where(np.abs(lam_a) >  np.abs(lam_b), lam_a, lam_b)

        # Sign-gate: bright ridge requires λ2 < 0 (working image is "bright")
        ridge_sign = lam2 < 0
        # S = Frobenius norm of Hessian
        S = np.sqrt(lam1 * lam1 + lam2 * lam2)

        # γ per scale (Frangi 1998 Eq. 16): paper says max(S)/2. We use a
        # robust max-proxy via percentile (default 99) divided by 2 so that
        # occasional high-curvature spikes don't blow γ up. Median (50)
        # would collapse γ and saturate structureness everywhere — do not
        # set gamma_percentile below ~95.
        # Calibrate on the WHOLE-scene curvature distribution (`valid`),
        # not just on cells that pass the ridge sign-gate. Otherwise on a
        # valley-dominated DEM γ adapts to whatever survived sign-gating,
        # making valley walls saturate to the same peak as bright ridges
        # would (loss of bright/dark discrimination).
        if gamma is None:
            S_valid = S[valid]
            if S_valid.size == 0:
                continue
            gamma_s = 0.5 * float(np.percentile(S_valid, gamma_percentile))
            if gamma_s <= 0:
                gamma_s = 1e-6
        else:
            gamma_s = float(gamma)

        # Frangi 2D vesselness (Eq. 15)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_b = np.where(np.abs(lam2) > 1e-12, np.abs(lam1) / np.abs(lam2), 0.0)
        v_blob = np.exp(-(r_b * r_b) / (2.0 * beta * beta))
        v_struct = 1.0 - np.exp(-(S * S) / (2.0 * gamma_s * gamma_s))
        v = v_blob * v_struct
        v = np.where(ridge_sign & valid, v, 0.0)

        response = np.maximum(response, v)

    return response.astype(np.float32)


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

    Samples the DEM at ``n_samples_per_seg`` interior points along EACH
    segment of the line (NOT just at vertices), at the crest and at ±
    ``half_m`` along the segment-perpendicular. This is critical for
    DP-simplified 2-vertex polylines: vertex-only sampling would only
    check the endpoints (often near array boundaries → out-of-bounds NaN).
    Relief := z_crest - min(z_left, z_right). Negative on valley slopes.

    ``aggregation_quantile`` controls which order statistic of the per-sample
    relief is returned as the first element: 0.5 = median (strict, reject if
    most of the line is flat); 0.75 = Q75 (lenient, keep if a quarter of the
    line has relief — better for long merged lines spanning flat sections).
    """
    coords = np.asarray(line.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return (np.nan, np.nan)

    # Per-segment interior sampling: place ``n_samples_per_seg`` points
    # uniformly inside each segment (excluding the endpoints to avoid the
    # array-corner out-of-bounds issue). The per-segment tangent gives the
    # local perpendicular direction for swath sampling.
    n_per_seg = max(1, int(n_samples_per_seg))
    # Fractional positions along each segment (e.g., n=3 → 0.25, 0.5, 0.75)
    fracs = (np.arange(n_per_seg, dtype=np.float64) + 1.0) / (n_per_seg + 1.0)
    seg_starts = coords[:-1]
    seg_ends = coords[1:]
    seg_vec = seg_ends - seg_starts
    seg_norms = np.linalg.norm(seg_vec, axis=1, keepdims=True)
    seg_norms = np.where(seg_norms == 0, 1.0, seg_norms)
    t_unit_per_seg = seg_vec / seg_norms  # one unit-tangent per segment
    n_unit_per_seg = np.stack(
        [-t_unit_per_seg[:, 1], t_unit_per_seg[:, 0]], axis=1
    )  # rotate 90°

    # Build sample points: (n_segs * n_per_seg, 2)
    p_center_list = []
    n_unit_list = []
    for f in fracs:
        p_center_list.append(seg_starts + f * seg_vec)
        n_unit_list.append(n_unit_per_seg)
    p_center = np.concatenate(p_center_list, axis=0)
    n_unit = np.concatenate(n_unit_list, axis=0)

    p_left = p_center + n_unit * half_m
    p_right = p_center - n_unit * half_m

    # World -> row/col using inverse transform. Use floor (not round) — for
    # cell-centered world coords every fractional part is 0.5 and Python's
    # banker's-rounding picks the wrong cell asymmetrically.
    inv = ~transform
    H, W = dem.shape
    reliefs = []
    for pc, pl, pr in zip(p_center, p_left, p_right):
        rc_c = inv * (pc[0], pc[1])
        rc_l = inv * (pl[0], pl[1])
        rc_r = inv * (pr[0], pr[1])
        cc = int(np.floor(rc_c[0]))
        rcc = int(np.floor(rc_c[1]))
        cl = int(np.floor(rc_l[0]))
        rcl = int(np.floor(rc_l[1]))
        cr_ = int(np.floor(rc_r[0]))
        rcr = int(np.floor(rc_r[1]))
        if not (0 <= rcc < H and 0 <= cc < W):
            continue
        zc = dem[rcc, cc]
        zl = dem[rcl, cl] if 0 <= rcl < H and 0 <= cl < W else np.nan
        zr = dem[rcr, cr_] if 0 <= rcr < H and 0 <= cr_ < W else np.nan
        # Suppress "All-NaN axis encountered" RuntimeWarning when both
        # shoulders are out-of-bounds; relief becomes NaN, filtered
        # downstream by `np.isfinite(relief_q)`.
        if np.isnan(zl) and np.isnan(zr):
            reliefs.append(np.nan)
        else:
            with np.errstate(invalid="ignore"):
                shoulder_min = np.nanmin([zl, zr])
            reliefs.append(zc - shoulder_min)
    if not reliefs:
        return (np.nan, np.nan)
    reliefs = np.asarray(reliefs, dtype=np.float64)
    if np.all(np.isnan(reliefs)):
        return (np.nan, np.nan)
    with np.errstate(invalid="ignore"):
        q = float(np.nanquantile(reliefs, aggregation_quantile))
        mn = float(np.nanmin(reliefs))
    return (q, mn)


def _longest_paths_per_component(sk) -> list:
    """Extract polylines from skan's skeleton graph, one or more per component.

    Behaviour by component topology:
    - **Trees** (E = V−1): emits a single polyline = the longest geodesic
      via the classic two-pass Dijkstra "diameter" algorithm. Side-branches
      (spurs/teeth) are dropped by design (max-continuity output).
    - **Lollipops** (≥1 leaves + cycle): primary path via leaf-Dijkstra
      (longest leaf-to-any-vertex). Restart-BFS from any unvisited cycle
      vertex emits additional polylines until the component is covered.
    - **Pure cycles / multi-cycle (figure-8, theta)**: BFS ring-walk from
      ``comp_idx[0]`` for the primary; restart-BFS from any unvisited
      vertex emits additional rings. Each cycle is closed via direct
      adjacency or shared-neighbour bridge so closed-loop polylines
      (eucl=0) survive the sinuosity filter as polders.

    Returns a list of paths, each as an ``(N, 2)`` array of ``(row, col)``
    pixel coordinates ordered along the path.
    """
    from scipy.sparse.csgraph import connected_components, dijkstra

    coords = np.asarray(sk.coordinates)
    csg = sk.graph
    n = csg.shape[0]
    if n == 0:
        return []

    # Vertex degree in the skeleton graph (undirected edge count)
    csg_csr = csg.tocsr() if not hasattr(csg, "indptr") else csg
    degrees = np.diff(csg_csr.indptr)

    n_comp, labels = connected_components(csg, directed=False)
    paths = []

    def _walk_pred(end, far, pred, n):
        """Walk predecessor array from end back to far. Returns ordered list."""
        out = []
        cur = end
        guard = 0
        while cur >= 0 and cur != far and guard < n + 1:
            out.append(cur)
            cur = int(pred[cur]) if pred[cur] >= 0 else -1
            guard += 1
        if cur == far:
            out.append(far)
            out.reverse()
            return out
        return None  # pathological

    for c in range(n_comp):
        comp_mask = labels == c
        comp_idx = np.nonzero(comp_mask)[0]
        if comp_idx.size < 2:
            continue

        # Edges within this component (E_comp). Exact undirected edge count
        # = (sum of in-component degrees) / 2 — only counts edges where both
        # endpoints are in this component (which is automatic for connected
        # components).
        E_comp = int(degrees[comp_idx].sum() // 2)
        V_comp = int(comp_idx.size)
        is_tree = E_comp == V_comp - 1

        if is_tree:
            # Classic two-pass Dijkstra (correct on trees)
            start = int(comp_idx[0])
            d1 = dijkstra(csg, indices=start, directed=False)
            d1_in_comp = np.where(comp_mask, d1, -np.inf)
            far = int(np.argmax(d1_in_comp))
            d2, pred = dijkstra(csg, indices=far, directed=False,
                                return_predecessors=True)
            d2_in_comp = np.where(comp_mask, d2, -np.inf)
            end = int(np.argmax(d2_in_comp))
            path_nodes = _walk_pred(end, far, pred, n)
        else:
            # Graph has cycle(s). Emit a primary "longest-feasible" path,
            # then RESTART from any vertex still uncovered to emit more
            # polylines until every component vertex is visited at least
            # once. This handles:
            #   - simple lollipop (1 leaf + 1 cycle): primary leaf-Dijkstra
            #     gives stem+half-cycle; restart on the unvisited half
            #     emits the other half as a second polyline.
            #   - pure single cycle: BFS ring-walk emits the closed ring.
            #   - figure-8 / theta / handcuff (no leaves, multi-cycle):
            #     successive BFS-from-unvisited emits one polyline per
            #     cycle/face.
            leaves = comp_idx[degrees[comp_idx] == 1]
            csg_csr2 = csg.tocsr() if not hasattr(csg, "indptr") else csg
            visited_global = set()

            def _bfs_walk(start_v: int) -> list:
                """Greedy walk picking any unvisited neighbour. Returns
                ordered list of visited vertices (>=1)."""
                visited_global.add(start_v)
                walk = [start_v]
                cur_v = start_v
                while True:
                    nbrs_local = csg_csr2.indices[
                        csg_csr2.indptr[cur_v]:csg_csr2.indptr[cur_v + 1]
                    ]
                    next_v = -1
                    for nb in nbrs_local:
                        if int(nb) not in visited_global:
                            next_v = int(nb)
                            break
                    if next_v < 0:
                        break
                    visited_global.add(next_v)
                    walk.append(next_v)
                    cur_v = next_v
                return walk

            # Primary path
            primary: Optional[list] = None
            if leaves.size >= 1:
                # Lollipop / leaf-bearing cycle: Dijkstra from each leaf;
                # keep longest leaf-to-any-vertex path.
                leaf_iter = leaves
                if leaf_iter.size > 64:
                    leaf_iter = leaf_iter[::max(1, leaf_iter.size // 64)]
                best_len = -np.inf
                for lf in leaf_iter:
                    d, pred = dijkstra(csg, indices=int(lf), directed=False,
                                       return_predecessors=True)
                    d_in_comp = np.where(comp_mask, d, -np.inf)
                    end = int(np.argmax(d_in_comp))
                    L = float(d[end])
                    if not np.isfinite(L) or L <= best_len:
                        continue
                    candidate = _walk_pred(end, int(lf), pred, n)
                    if candidate is not None and len(candidate) >= 2:
                        primary = candidate
                        best_len = L
            else:
                # Pure / multi-cycle: BFS ring-walk from comp_idx[0]
                primary = _bfs_walk(int(comp_idx[0]))
                # Close the ring if traversal made a single loop
                if (len(primary) > 2
                        and int(comp_idx[0]) in csg_csr2.indices[
                            csg_csr2.indptr[primary[-1]]
                            :csg_csr2.indptr[primary[-1] + 1]
                        ]):
                    primary.append(int(comp_idx[0]))

            if primary is not None and len(primary) >= 2:
                paths.append(coords[primary])
                visited_global.update(primary)

            # Cover remaining vertices with additional BFS walks. For
            # closed-cycle topologies (figure-8, theta, adjacent rings
            # sharing edges), close the ring whenever start and end are
            # graph-adjacent OR share a common visited neighbour. The
            # latter handles cases where a junction vertex is consumed
            # by the primary walk: e.g. figure-8 with 1 shared corner —
            # the secondary cycle's end-cell is adjacent to the shared
            # corner, which is also adjacent to start. Closing via the
            # shared corner makes the polyline a true ring (eucl=0) so
            # the sinuosity filter doesn't kill it.
            comp_set = set(int(v) for v in comp_idx)
            unvisited = comp_set - visited_global
            while unvisited:
                start_v = next(iter(unvisited))
                walk = _bfs_walk(start_v)
                if len(walk) > 2:
                    end_v = walk[-1]
                    end_nbrs = set(int(x) for x in csg_csr2.indices[
                        csg_csr2.indptr[end_v]:csg_csr2.indptr[end_v + 1]
                    ])
                    start_nbrs = set(int(x) for x in csg_csr2.indices[
                        csg_csr2.indptr[start_v]:csg_csr2.indptr[start_v + 1]
                    ])
                    if start_v in end_nbrs:
                        # Direct cycle: end is graph-adjacent to start
                        walk.append(start_v)
                    else:
                        # Near-cycle: close via a shared-neighbour bridge
                        # (typically the junction vertex consumed by primary)
                        bridge = end_nbrs & start_nbrs
                        if bridge:
                            walk.append(int(next(iter(bridge))))
                            walk.append(start_v)
                if len(walk) >= 2:
                    paths.append(coords[walk])
                else:
                    # Single-vertex walks (orphan vertex with all
                    # neighbours already visited): emit a 2-vertex stub
                    # to preserve coverage. Filtered downstream by
                    # min_length_m if too short.
                    nbrs_local = csg_csr2.indices[
                        csg_csr2.indptr[start_v]
                        :csg_csr2.indptr[start_v + 1]
                    ]
                    if len(nbrs_local) > 0:
                        paths.append(coords[[start_v, int(nbrs_local[0])]])
                unvisited = comp_set - visited_global

            path_nodes = None  # already appended; skip the outer paths.append

        if path_nodes is not None and len(path_nodes) >= 2:
            paths.append(coords[path_nodes])  # (M, 2) row, col

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
            # Skip pairs involving closed rings — merging would destroy
            # their topological completeness (a closed ring has start==end
            # so its endpoints coincide; merging an open line onto either
            # endpoint produces an open polyline that no longer encloses
            # the polder).
            if la.is_closed or lb.is_closed:
                continue
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
    from skimage.morphology import disk, skeletonize
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

    # 1. Morphological cleanup. Closing with disk(bridge_px) fills gaps up
    # to ``bridge_gap_m`` wide in the mask before component filtering, so
    # reconnected lines pass the min-area threshold. Cap bridge_px at 25
    # cells to bound the worst-case `closing` cost (cubic in radius);
    # values above 25 are unusual for levee detection and risk fusing
    # unrelated features. Use scipy.ndimage.binary_closing (~2x faster
    # than skimage.morphology.closing on large arrays).
    mask_clean = mask.astype(bool)
    bridge_px = max(0, int(round(bridge_gap_m / dx_m)))
    if bridge_px > 25:
        logger.warning(
            f"bridge_gap_m={bridge_gap_m} -> {bridge_px} px closing radius "
            f"is unusually large; capping at 25 px to bound runtime."
        )
        bridge_px = 25
    if bridge_px > 0:
        # Use scipy default `border_value=0` (conservative under-closure
        # within bridge_px of the AOI boundary). An earlier revision
        # tried `border_value=1` to "match skimage.closing" — but
        # empirically scipy(border_value=1) FABRICATES cells in the
        # boundary band from nothing (24 spurious cells in col 0 on a
        # synthetic test), which silently propagated into a
        # `depression_label=0` bug in detect_levees_breach for
        # edge-adjacent basins. Conservative under-closure is the safe
        # choice; it loses at most a 25-cell band of correct closing at
        # the array edge.
        mask_clean = ndimage.binary_closing(
            mask_clean, structure=disk(bridge_px)
        )
    if min_component_area_px > 0:
        # 8-connectivity so single-pixel-wide diagonal masks (e.g. clean
        # Steger output) don't fragment into area-1 components.
        lbl = label(mask_clean, connectivity=2)
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
        # Closed rings (eucl=0): sinuosity is undefined. Accept if long
        # enough (likely polder/closed levee), reject if short (likely
        # skeletonization artifact, caught by min_length filter anyway).
        ep0, ep1 = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
        eucl = float(np.hypot(ep1[0] - ep0[0], ep1[1] - ep0[1]))
        if eucl > 0 and (line.length / eucl) > max_sinuosity:
            continue
        # eucl==0 falls through; line.length is filtered separately

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
                # Floor (not round) — banker's-rounding picks the wrong
                # cell on cell-centered .5 fractional coords.
                ri = np.clip(np.floor(rc[:, 1]).astype(int), 0, dem_arr.shape[0] - 1)
                ci = np.clip(np.floor(rc[:, 0]).astype(int), 0, dem_arr.shape[1] - 1)
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

    # 3c. Drop merged lines that now exceed the sinuosity bound (rare but possible).
    # Closed loops (eucl=0): keep if long (polder), filter via min_length only.
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
    window_m: float = 100.0,
    windows_m: Optional[Tuple[float, ...]] = None,
    use_reconstruction: bool = False,
    seed_quantile: float = 0.95,
    grow_quantile: float = 0.60,
    rea_thresh_m: float = 0.5,
    threshold_method: str = "std",
    k_std: float = 1.0,
    sign: str = "positive",
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

    ``REA(x, y) = z(x, y) − mean(z, window=window_m)``. Positive REA picks
    out topographic highs (levees, embankments); negative REA picks out
    ditches.

    Threshold methods (``threshold_method``):

    - ``"std"`` (Cazorzi et al. 2013, canonical): ``REA > k_std · σ_REA``,
      where ``σ_REA`` is estimated via MAD (median absolute deviation,
      robust to slope/heterogeneity) over valid pixels. ``k_std=1.0`` is
      the original recommendation (range 1.0–1.5). The ``rea_thresh_m``
      hard floor prevents pure-noise DEMs from flagging 16% of pixels.
    - ``"quantile"`` (extension): hysteresis on ``low_quantile``/
      ``high_quantile`` of positive REA, plus a hard floor ``rea_thresh_m``.

    Sign convention (``sign``):
    - ``"positive"`` (default): pick out levees / topographic highs.
    - ``"negative"``: pick out ditches / depressions.
    - ``"both"``: pick out both (separate masks unioned).

    Set ``use_reconstruction=True`` to switch to a multi-scale, seed-grown
    mask via ``skimage.morphology.reconstruction`` (Sofia 2014-style
    extension; not in Cazorzi 2013): strong seeds at ``seed_quantile`` grow
    through the permissive ``grow_quantile`` mask to produce long connected
    components from sparse evidence.

    Window-size guidance: the moving-mean window must be larger than the
    expected levee footprint (crest + flanks) so the window does not ride
    up onto the levee crest itself, but small enough to remain local.
    Sofia et al. 2014 tested windows from 3 to 55 m on agrarian floodplains
    with feature widths 10–50 m, and recommended ~23 m rectangular (or
    ~29 m circular). For larger anthropogenic levees (10–20 m wide on a
    floodplain), ``window_m`` ≈ 100 m is a safe default; use 200–300 m on
    very broad floodplains.

    References
    ----------
    Cazorzi, F., Dalla Fontana, G., Da Ros, D., Marchi, L., Sofia, G.,
    Tarolli, P. (2013). Drainage network detection and assessment of
    network storage capacity in agrarian landscape. Hydrol. Process. 27,
    3270-3282. doi:10.1002/hyp.9224
    Sofia, G., Dalla Fontana, G., Tarolli, P. (2014). High-resolution
    topographic data and anthropogenic feature extraction. Hydrol. Process.
    28, 2046-2061.
    """
    _assert_projected(da_dem)

    # Validate sign and dispatch on bidirectional cases. Ditches are
    # detected by negating the DEM and re-running with sign='positive'
    # — REA(-z) = -REA(z), and downstream relief filtering also works
    # correctly on the negated DEM (peaks become valleys and vice versa).
    sign_mode = (sign or "positive").lower()
    if sign_mode not in ("positive", "negative", "both"):
        raise ValueError(
            f"sign must be 'positive', 'negative', or 'both', got {sign!r}"
        )
    # Recurse for sign='negative' / sign='both' regardless of
    # `use_reconstruction`. The single recursion code-path below handles
    # both std/quantile and reconstruction branches transparently
    # (the negated DEM with sign="positive" hits the appropriate inner
    # branch via the same dispatch).
    if sign_mode == "negative":
        # Multiplication preserves rio CRS/transform automatically; only
        # nodata needs an explicit re-write because its sign flips.
        da_neg = (-1.0) * da_dem
        if da_dem.rio.nodata is not None:
            da_neg.rio.write_nodata(-float(da_dem.rio.nodata), inplace=True)
        return detect_ridges_rea(
            da_neg, window_m=window_m, windows_m=windows_m,
            use_reconstruction=use_reconstruction,
            seed_quantile=seed_quantile, grow_quantile=grow_quantile,
            rea_thresh_m=rea_thresh_m, threshold_method=threshold_method,
            k_std=k_std, sign="positive",
            high_quantile=high_quantile, low_quantile=low_quantile,
            min_length_m=min_length_m, min_relief_m=min_relief_m,
            simplify_tol_m=simplify_tol_m,
            swath_half_width_m=swath_half_width_m,
            min_component_area_px=min_component_area_px,
            bridge_gap_m=bridge_gap_m, endpoint_snap_m=endpoint_snap_m,
            max_merge_angle_deg=max_merge_angle_deg,
            require_colinearity=require_colinearity,
            max_sinuosity=max_sinuosity, merge_strategy=merge_strategy,
            relief_quantile=relief_quantile, logger=logger,
        )
    if sign_mode == "both":
        kwargs = dict(
            window_m=window_m, windows_m=windows_m,
            use_reconstruction=use_reconstruction,
            seed_quantile=seed_quantile, grow_quantile=grow_quantile,
            rea_thresh_m=rea_thresh_m, threshold_method=threshold_method,
            k_std=k_std, high_quantile=high_quantile,
            low_quantile=low_quantile, min_length_m=min_length_m,
            min_relief_m=min_relief_m, simplify_tol_m=simplify_tol_m,
            swath_half_width_m=swath_half_width_m,
            min_component_area_px=min_component_area_px,
            bridge_gap_m=bridge_gap_m, endpoint_snap_m=endpoint_snap_m,
            max_merge_angle_deg=max_merge_angle_deg,
            require_colinearity=require_colinearity,
            max_sinuosity=max_sinuosity, merge_strategy=merge_strategy,
            relief_quantile=relief_quantile, logger=logger,
        )
        gdf_pos = detect_ridges_rea(da_dem, sign="positive", **kwargs)
        gdf_neg = detect_ridges_rea(da_dem, sign="negative", **kwargs)
        if not gdf_neg.empty:
            gdf_neg["stype"] = "thd"
        gdf_combined = gpd.GeoDataFrame(
            pd.concat([gdf_pos, gdf_neg], ignore_index=True),
            crs=da_dem.raster.crs,
        )
        # Re-number polylines so positive-sign and negative-sign outputs
        # have unique names after concat (each call independently produces
        # ridge_00000, ridge_00001, ...).
        if "name" in gdf_combined.columns and not gdf_combined.empty:
            gdf_combined["name"] = [
                f"ridge_{i:05d}" for i in range(len(gdf_combined))
            ]
        return gdf_combined

    # Below: sign_mode == "positive". Single forward path.
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)

    if use_reconstruction:
        ws = tuple(windows_m) if windows_m else (window_m,)
        # Apply edge-buffer erosion to suppress moving-mean spillover
        # from mean-filled NoData and reflected-edge cells. Use the
        # LARGEST window's half-reach as the buffer radius (worst case
        # over the multi-scale REA-max). Without this, empirical 165×
        # spurious response near NaN holes with default windows_m=100.
        max_win_px = max(max(int(round(w / dx_m)) for w in ws), 3)
        valid_eff = _erode_valid_buffer(valid, sigma_px=max_win_px / 2.0,
                                        sigma_factor=1.0)
        mask, rea = _rea_reconstruction_mask(
            dem_arr, valid_eff, dx_m,
            windows_m=ws,
            seed_quantile=seed_quantile,
            grow_quantile=grow_quantile,
        )
        logger.info(
            f"REA reconstruction: windows_m={ws}, "
            f"seed_q={seed_quantile}, grow_q={grow_quantile} -> "
            f"{int(mask.sum())} candidate pixels"
        )
        valid = valid_eff  # so downstream postprocess uses the eroded mask
    else:
        win_px = max(3, int(round(window_m / dx_m)))
        if win_px % 2 == 0:
            win_px += 1  # odd-size window for symmetric centering
        mean_map = ndimage.uniform_filter(dem_arr, size=win_px, mode="reflect")
        rea = dem_arr - mean_map
        rea[~valid] = 0.0

        # Erode valid mask by ~win_px/2 (the uniform_filter half-kernel
        # reach) so cells whose moving-mean window touched mean-fill or
        # reflected-edge data are excluded from threshold/mask. This is
        # the convolution-equivalent of `_erode_valid_buffer` used by
        # Hessian flavors.
        valid = _erode_valid_buffer(valid, sigma_px=win_px / 2.0,
                                    sigma_factor=1.0)

        method = threshold_method.lower()
        if method == "std":
            # Robust scale via MAD: 1.4826·median(|REA - median(REA)|).
            # On heterogeneous DEMs (mountain+floodplain) plain std is
            # inflated 100× by the slope tail and real levees get missed.
            # MAD over the entire valid region gives a background-only
            # scale estimate consistent with Cazorzi 2013's intent.
            if valid.any():
                rea_v = rea[valid]
                med = float(np.median(rea_v))
                mad = float(np.median(np.abs(rea_v - med)))
                sigma_rea = 1.4826 * mad
                if sigma_rea <= 0:  # MAD=0 on a perfectly flat DEM
                    sigma_rea = float(np.std(rea_v)) or 1.0
            else:
                sigma_rea = 1.0
            # Hard floor on rea_thresh_m so pure-noise DEMs don't flag
            # 16% of pixels at k=1 (Gaussian one-tail); paper note in
            # Cazorzi 2013 §3.2.
            thresh = max(float(k_std) * sigma_rea, float(rea_thresh_m))
            mask = (rea > thresh) & valid
            logger.info(
                f"REA std threshold: sigma_REA(MAD)={sigma_rea:.3f} m, "
                f"k={k_std}, hard_floor={rea_thresh_m} m -> tau={thresh:.3f} m, "
                f"-> {int(mask.sum())} candidate pixels"
            )
        elif method == "quantile":
            mask_abs = rea >= rea_thresh_m
            mask_hyst = _quantile_hysteresis(rea, valid, low_quantile, high_quantile)
            mask = mask_abs & mask_hyst & valid
            logger.info(
                f"REA quantile-hysteresis threshold: low={low_quantile}, "
                f"high={high_quantile}, hard_floor={rea_thresh_m} m -> "
                f"{int(mask.sum())} candidate pixels"
            )
        else:
            raise ValueError(
                f"threshold_method must be 'std' or 'quantile', got {method!r}"
            )

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
    gamma_percentile: float = 99.0,
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

    Notes
    -----
    Uses our own σ²-normalized multi-scale Frangi (``_frangi_multiscale``)
    rather than ``skimage.filters.frangi``, which dropped the σ²
    normalization in v0.20+ and collapses the multi-scale max to the
    smallest σ (scikit-image issue #7711). ``alpha`` is unused in 2D
    (Frangi 1998 Eq. 15 — α is a 3D-only plate-vs-line parameter).
    """
    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)

    # High-pass residual topography to remove regional slope
    dem_hp = _high_pass(dem_arr, sigma_px=high_pass_sigma_m / dx_m)

    # Convert meter-scale sigmas to pixels
    sigmas_px = tuple(max(0.5, s / dx_m) for s in sigmas_m)
    logger.debug(f"frangi sigmas_px = {sigmas_px}")
    _ = alpha  # 2D no-op; kept for API parity

    # Erode valid mask by ~3·max(σ) to discard cells whose convolution
    # kernel touched the mean-filled NoData buffer (cross-cutting fix
    # X-1). For Frangi we additionally include the high-pass σ.
    sigma_buffer = max(max(sigmas_px), high_pass_sigma_m / dx_m)
    valid_eff = _erode_valid_buffer(valid, sigma_buffer)
    response = _frangi_multiscale(
        dem_hp, valid_eff,
        sigmas_px=sigmas_px,
        beta=beta,
        gamma=gamma,
        gamma_percentile=gamma_percentile,
        black_ridges=False,
    )
    response[~valid_eff] = 0.0

    mask = _quantile_hysteresis(response, valid_eff, low_quantile, high_quantile)

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
    geomorphon_classes: Tuple[int, ...] = (3, 4, 5),   # ridge, shoulder, spur
    geomorphon_combine: str = "and",                   # "and" intersection (high precision); "or" for high recall
    search_radius_m: float = 60.0,
    geomorphon_threshold_deg: float = 1.0,             # Jasiewicz & Stepinski 2013 default
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
    smooth_filter_m: float = 5.0,
    smooth_norm_diff_deg: float = 8.0,
    smooth_num_iter: int = 3,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Detect ridges via WhiteboxTools ``find_ridges`` and geomorphons.

    Requires the ``whitebox`` package (installs a local Rust binary on first
    use). The binary writes intermediate GeoTIFFs to ``work_dir`` (a temp dir
    if not provided).

    Pre-smoothing parameters (when ``smooth_first=True``):
    ``smooth_filter_m`` (filter window in metres; converted to cells),
    ``smooth_norm_diff_deg`` (normal-vector tolerance; lower = more
    feature-preserving — WBT default 15° relaxed to 8° here for narrow
    levee crests), ``smooth_num_iter`` (iterations).

    Combines ``wbt.find_ridges(line_thin=True)`` with ``wbt.geomorphons``.
    Geomorphon classes (Jasiewicz & Stepinski 2013, also WhiteboxTools /
    GRASS r.geomorphon / ArcGIS Pro): 1=Flat, 2=Peak, 3=Ridge, 4=Shoulder,
    5=Spur, 6=Slope, 7=Hollow, 8=Footslope, 9=Valley, 10=Pit. Default
    ``(3, 4, 5)`` keeps Ridge + Shoulder + Spur (the levee-relevant
    classes). ``geomorphon_combine`` controls how the two masks are fused:

    - ``"and"`` (default, high precision): strict intersection of
      ``find_ridges`` and the kept geomorphon classes — restricts to
      cells confirmed by BOTH detectors. Synthetic precision ≈ 0.56 vs
      0.03 with ``"or"`` on a sloping plane + Gaussian crest test.
    - ``"or"`` (high recall): keep pixels flagged by EITHER detector —
      more recall, more fragmented output. Use for gentle levees on
      floodplains where geomorphons alone are too noisy.
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

        def _check_wbt_output(path: Path, tool: str) -> None:
            """Whitebox panics on some path-edge cases (e.g. '=' in
            work_dir) yet the Python wrapper returns rc=0. Guard by
            checking the output GeoTIFF actually exists and has nonzero
            size; raise a clean exception if not."""
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(
                    f"WhiteboxTools `{tool}` produced no output at {path}. "
                    f"Check that work_dir contains no shell-special chars "
                    f"and the input DEM is valid."
                )

        # Pre-emptively delete any stale outputs from a prior run sharing
        # the same `work_dir`. Otherwise `_check_wbt_output` would pass on
        # a stale file even when the current WBT call silently failed,
        # masking the failed step and producing misleading errors one
        # step later.
        for _stale_path in (smooth_path, ridges_path, geom_path):
            if _stale_path.exists():
                _stale_path.unlink()

        dem_for_ridges = str(dem_path)
        if smooth_first:
            wbt.feature_preserving_smoothing(
                dem=str(dem_path),
                output=str(smooth_path),
                filter=max(3, int(round(smooth_filter_m / dx_m))),
                norm_diff=float(smooth_norm_diff_deg),
                # Use round + clamp >= 1: bare int() would truncate
                # `2.99 -> 2` silently and accept negative values.
                num_iter=max(1, int(round(smooth_num_iter))),
            )
            _check_wbt_output(smooth_path, "feature_preserving_smoothing")
            dem_for_ridges = str(smooth_path)

        wbt.find_ridges(
            dem=dem_for_ridges,
            output=str(ridges_path),
            line_thin=True,
        )
        _check_wbt_output(ridges_path, "find_ridges")

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
            _check_wbt_output(geom_path, "geomorphons")
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

    # Filter -9999 (WBT find_ridges nodata sentinel) and any non-finite
    # values out of the response map before passing to post-processing —
    # otherwise nanmedian along a polyline mixes legitimate scores with
    # the nodata sentinel and yields a meaningless score.
    ridges_score = np.where(
        (ridges == -9999.0) | ~np.isfinite(ridges), 0.0, ridges
    ).astype(np.float32)

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
        response_map=ridges_score,
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
    black_ridges: bool = False,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Detect ridges via a Steger-style unbiased curvilinear detector.

    Implemented directly on the Hessian (no dependency on opencv-contrib's
    ``ximgproc.StegerFilter`` so portability is preserved). At each pixel,
    compute the Hessian eigen-decomposition and keep pixels where:

    - ``lambda_max`` (strongest negative curvature for bright ridges; or
      strongest positive curvature for dark ridges/ravines when
      ``black_ridges=True``) is sufficiently extreme,
    - the gradient projected onto the corresponding eigenvector is near zero
      (we are on the crest rather than on a flank).

    Feature width per pixel is estimated from the parabolic-crest model
    ``2 * sqrt(2 * z / |lambda_max|)``, with σ²-normalized ``lambda_max``
    (Lindeberg γ=2) so the width is scale-invariant.
    Drives the ``thd``/``weir`` split via ``width_thd_thresh_m``
    (default ``0.5 * dx``).

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

    # Sign convention: for `black_ridges=True` (ravines/ditches), negate
    # the input so the dark-line case maps to the bright-line algorithm.
    # All downstream eigenvalue/eigenvector logic + relief filtering then
    # operates as for bright ridges.
    dem_for_hess = (-dem_arr) if black_ridges else dem_arr

    # Residual topography for height magnitude used in width estimation
    dem_hp = _high_pass(dem_for_hess, sigma_px=high_pass_sigma_m / dx_m)

    # Gaussian derivatives of order 2 via scipy
    zxx = gaussian_filter(dem_for_hess, sigma=sigma_px, order=(0, 2), mode="reflect")
    zyy = gaussian_filter(dem_for_hess, sigma=sigma_px, order=(2, 0), mode="reflect")
    zxy = gaussian_filter(dem_for_hess, sigma=sigma_px, order=(1, 1), mode="reflect")
    zx = gaussian_filter(dem_for_hess, sigma=sigma_px, order=(0, 1), mode="reflect")
    zy = gaussian_filter(dem_for_hess, sigma=sigma_px, order=(1, 0), mode="reflect")

    # Eigenvalues of 2x2 symmetric Hessian
    #   [[zxx, zxy], [zxy, zyy]]
    # lambda_{1,2} = (zxx+zyy)/2 +- sqrt( ((zxx-zyy)/2)**2 + zxy**2 )
    mean_h = 0.5 * (zxx + zyy)
    diff_h = 0.5 * (zxx - zyy)
    disc = np.sqrt(diff_h * diff_h + zxy * zxy)
    lam1 = mean_h + disc  # >= lam2
    lam2 = mean_h - disc  # <= lam1; most-negative for bright ridges
    # Eigenvector of lam2 (cross-ridge direction). For a 2x2 symmetric Hessian
    # H=[[a,b],[b,c]] with eigenvalue lam, an eigenvector is (lam-c, b) — i.e.
    # (lam2 - zyy, zxy). The alternate form (b, lam-a) = (zxy, lam2-zxx) is
    # also valid; we use the former because it stays non-zero when zxy=0 and
    # zxx != zyy (axis-aligned ridges).
    v_x = lam2 - zyy
    v_y = zxy
    vnorm = np.hypot(v_x, v_y)
    # Numerical degeneracy: lam2≈zyy AND zxy≈0 (axis-aligned ridge along x);
    # fall back to the alternate eigenvector form (zxy, lam2-zxx).
    degen = vnorm < 1e-12
    if degen.any():
        v_x = np.where(degen, zxy, v_x)
        v_y = np.where(degen, lam2 - zxx, v_y)
        vnorm = np.hypot(v_x, v_y)
    vnorm = np.where(vnorm == 0, 1.0, vnorm)
    v_x /= vnorm
    v_y /= vnorm

    # Response: normalize into meters of curvature per dx^2 for robustness,
    # then take magnitude of lam2 where it is negative (bright ridge).
    response = np.where(lam2 < 0, -lam2, 0.0).astype(np.float32)
    response[~valid] = 0.0

    # Steger's sub-pixel offset along the eigenvector; we require the offset
    # projected onto each axis to be within half a pixel (Steger 1998 Eq. 18:
    # |t·n_x| <= 0.5 AND |t·n_y| <= 0.5). This per-axis check IS Steger's
    # NMS — do not add a separate NMS step.
    # t = -(zx*v_x + zy*v_y) / lam2  (lam2 can be near zero — guard).
    grad_proj = zx * v_x + zy * v_y
    with np.errstate(divide="ignore", invalid="ignore"):
        t_sub = np.where(lam2 < 0, -grad_proj / lam2, np.nan)
    on_crest = (
        np.isfinite(t_sub)
        & (np.abs(t_sub * v_x) <= 0.5)
        & (np.abs(t_sub * v_y) <= 0.5)
    )

    # Erode valid mask by ~3·σ to discard cells whose Hessian kernel
    # touched the mean-filled NoData buffer (cross-cutting X-1).
    sigma_buffer = max(sigma_px, high_pass_sigma_m / dx_m)
    valid_eff = _erode_valid_buffer(valid, sigma_buffer)
    mask_curv = _quantile_hysteresis(
        response, valid_eff, response_low_quantile, response_high_quantile
    )
    mask = mask_curv & on_crest & valid_eff

    # Width estimate per pixel (parabolic-crest model):
    #   z(r) = z0 - 0.5 * |lam_phys| * r^2    (r in metres)
    # Zero-crossing at r = sqrt(2·z0 / |lam_phys|).  Full width = 2·r =
    # 2·sqrt(2·z0/|lam_phys|).
    # `lam2` from scipy.gaussian_filter(order=2) is in z-units / pixel²;
    # divide by dx_m² to convert to z-units / m² (physical curvature).
    # Note: applying a σ²-normalization here (Lindeberg γ=2) is wrong —
    # γ-normalization is for scale-normalized RESPONSE scoring (cross-
    # scale comparison in multi-scale Frangi), not a unit conversion to
    # physical curvature. The parabolic-crest model has an inherent
    # +30-50% bias on Gaussian profiles (because the smoothed curvature
    # at the crest equals A·σ_g²/(σ_g²+σ²)^(3/2), which gives a wider
    # parabola than the true profile). Document this as a known model
    # limitation; the σ²-correction tried in an earlier revision under-
    # corrected by -50%, which was worse.
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_phys = np.abs(lam2) / (dx_m * dx_m)  # z-units / m²
        crest_h = np.clip(dem_hp, 0, None)
        # Full width = 2 * sqrt(2 * crest_h / |lam_phys|)
        width_map = 2.0 * np.sqrt(
            np.clip(np.where(lam_phys > 0, 2.0 * crest_h / lam_phys, np.nan), 0, None)
        )
    width_map = np.where(valid, width_map, np.nan).astype(np.float32)

    # When black_ridges=True the Hessian saw a negated DEM. The relief
    # filter inside _postprocess_mask_to_polylines uses the DEM directly;
    # passing the negated DataArray makes ravine "depths" appear as
    # positive reliefs there, so the standard min_relief_m filter works.
    # `(-1.0) * da_dem` preserves the rio CRS/transform/spatial_ref coord
    # via standard xarray broadcasting — no explicit re-write needed.
    da_for_postproc = (-1.0) * da_dem if black_ridges else da_dem

    return _postprocess_mask_to_polylines(
        mask,
        da_for_postproc,
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
    high_pass_sigma_m: float = 40.0,
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: Optional[float] = None,
    gamma_percentile: float = 99.0,
    low_quantile: float = 0.70,
    high_quantile: float = 0.95,
    min_length_m: float = 100.0,
    min_relief_m: float = 0.3,
    simplify_tol_m: Optional[float] = None,
    swath_half_width_m: float = 12.0,
    bridge_gap_m: float = 8.0,
    endpoint_snap_m: float = 25.0,
    max_merge_angle_deg: float = 30.0,
    require_colinearity: bool = True,
    max_sinuosity: float = 3.0,
    relief_quantile: float = 0.5,
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

    The endpoint-snap step DOES still run after `longest_path` extraction so
    adjacent components (still separated by mask gaps that the closing
    didn't bridge) can fuse into one continuous line. Set
    ``endpoint_snap_m=0`` to disable. With ``require_colinearity=False``
    (default) and large ``endpoint_snap_m``, perpendicular ridges can fuse
    into zigzags — set ``require_colinearity=True`` if this is a concern.

    References
    ----------
    Hardin et al. 2012 (*J. Coastal Res.* 28:939) used least-cost paths on
    ``max(z) − z`` to trace barrier-island dune crests. This implementation
    generalises the idea to ridge-response landscapes by tracing the
    longest geodesic path through each high-response skeleton component.

    Notes
    -----
    Uses ``_frangi_multiscale`` (σ²-normalized) rather than
    ``skimage.filters.frangi`` — see notes on ``detect_ridges_frangi``.
    """
    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)

    # 1. High-pass + multi-scale Frangi (σ²-normalized)
    dem_hp = _high_pass(dem_arr, sigma_px=high_pass_sigma_m / dx_m)
    sigmas_px = tuple(max(0.5, s / dx_m) for s in sigmas_m)
    _ = alpha  # 2D no-op
    sigma_buffer = max(max(sigmas_px), high_pass_sigma_m / dx_m)
    valid_eff = _erode_valid_buffer(valid, sigma_buffer)
    response = _frangi_multiscale(
        dem_hp, valid_eff,
        sigmas_px=sigmas_px,
        beta=beta, gamma=gamma, gamma_percentile=gamma_percentile,
        black_ridges=False,
    )
    response[~valid_eff] = 0.0

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
    3. Mark cells within ``bank_offset_m`` of the stream as the "bank rim"
       (Euclidean distance via ``distance_transform_edt`` — gives a true
       Chebyshev-square rim; scipy's binary_dilation default is 4-conn
       which under-extends diagonal rims).
    4. Within the rim, keep pixels that are:
       (a) locally elevated (REA at ``rea_window_m`` ≥ ``bank_thresh_m``),
       AND
       (b) at least ``min_bank_relief_m`` above the **nearest downstream
       drainage** via D8-traced HAND (Nobre et al. 2011/2016 — true height
       above nearest drain, not the Euclidean-nearest-stream proxy used
       in earlier versions).
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
        # Pass a NaN-version of the DEM (NOT the mean-filled `dem_arr`).
        # `_dem_to_numpy` fills NaN with the global mean; pyflwdir tests
        # `elevtn == nodata`, and our mean ≠ -9999 ≠ NaN — so without
        # explicit re-masking, pyflwdir routes flow over mean-filled
        # cells, inflating uparea ~5× near NoData regions and producing
        # phantom drainage in ocean / lake / void cells. Silent on a
        # clean Snohomish AOI; fatal on Arctic AOIs with sensor gaps.
        nodata = da_dem.raster.nodata if da_dem.raster.nodata is not None else -9999.0
        nodata_f = float(nodata)
        if not np.isfinite(nodata_f):
            nodata_f = -9999.0
        dem_for_pyflwdir = np.where(valid, dem_arr, nodata_f).astype("float32")
        flw = pyflwdir.from_dem(
            data=dem_for_pyflwdir,
            nodata=nodata_f,
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

    # 3. Bank rim: pixels within bank_offset_m of stream, excluding stream.
    # Use Euclidean distance transform (Chebyshev-square rim) — scipy's
    # default binary_dilation is 4-connected, which produces a Manhattan
    # diamond and under-extends the rim by ~30% on diagonal rivers.
    rim_radius_cells = bank_offset_m / dx_m
    buffer_radius_cells = float(np.ceil(bridge_gap_m / dx_m) + 1)
    band_width_cells = rim_radius_cells - buffer_radius_cells
    if band_width_cells <= 1.0:
        logger.warning(
            f"detect_river_banks: rim radius ({rim_radius_cells:.1f} px = "
            f"{bank_offset_m} m / {dx_m} m) is too small relative to "
            f"stream-buffer ({buffer_radius_cells:.0f} px). Effective "
            f"bank-candidate band width = {band_width_cells:.1f} px ≤ 1; "
            f"detection likely returns empty. Increase bank_offset_m or "
            f"decrease bridge_gap_m, or use a finer DEM (current dx={dx_m} m)."
        )
    if rim_radius_cells < 1.0:
        logger.warning(
            f"detect_river_banks: bank_offset_m ({bank_offset_m} m) is "
            f"smaller than DEM dx ({dx_m} m); rim is sub-pixel and "
            f"will be empty. Use a finer DEM or larger bank_offset_m."
        )
    dist_to_stream = ndimage.distance_transform_edt(~stream)
    rim = (dist_to_stream > 0) & (dist_to_stream <= max(1.0, rim_radius_cells)) & valid

    # 4a. Local REA on the rim
    win_px = max(3, int(round(rea_window_m / dx_m)))
    if win_px % 2 == 0:
        win_px += 1
    rea_local = dem_arr - ndimage.uniform_filter(dem_arr, size=win_px, mode="reflect")

    # 4b. True HAND via D8 flow paths (Nobre et al. 2011/2016). Needs
    # the flow-direction object; if user supplied a precomputed uparea
    # without a flw object, fall back to Euclidean distance approximation.
    relief_above_stream: np.ndarray
    if da_uparea is None:
        # `flw` is in scope from above
        relief_above_stream = np.asarray(
            flw.hand(drain=stream, elevtn=dem_arr.astype("float32")),
            dtype=np.float32,
        )
        # pyflwdir.hand returns -9999.0 for cells outside the routed flow
        # network (pits, off-network islands, NaN cells). Replace with NaN
        # so np.nanmedian along polylines doesn't include the sentinel
        # (would otherwise collapse score columns to ≈ -4999 with 50% bad).
        relief_above_stream = np.where(
            relief_above_stream <= -9000.0, np.nan, relief_above_stream
        ).astype(np.float32)
        logger.info(
            "detect_river_banks: using D8-traced HAND (Nobre 2011/2016)."
        )
    else:
        # Euclidean nearest-stream approximation (faster but biased at
        # confluences) — used only when the user supplied uparea without
        # the flow-direction object.
        _, (rs, cs) = ndimage.distance_transform_edt(~stream, return_indices=True)
        relief_above_stream = (dem_arr - dem_arr[rs, cs]).astype(np.float32)
        logger.info(
            "detect_river_banks: using Euclidean nearest-stream HAND-proxy "
            "(da_uparea was supplied without flw — pass dem only for true HAND)."
        )

    bank_mask = (
        rim
        & (rea_local > bank_thresh_m)
        & (relief_above_stream > min_bank_relief_m)
    )
    # Exclude a small buffer around the stream itself from the bank mask
    # so the morphological closing in post-processing cannot bridge left
    # and right banks across the channel (which would put the skeleton on
    # the channel centerline). Reuses `dist_to_stream` from above.
    stream_buffer = dist_to_stream <= buffer_radius_cells
    bank_mask = bank_mask & ~stream_buffer
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


# ---------------------------------------------------------------------------
# Flavor 7: Breach — depression / watershed boundary (Pronk et al. 2026)
# ---------------------------------------------------------------------------


def _priority_flood_breach_python(
    dem: np.ndarray,
    valid: np.ndarray,
    connectivity: int = 8,
):
    """Pure-Python adapted Improved Priority-Flood (Pronk et al. 2026).

    Faithful Python port of ``Breach.jl/sbreach``. Two priority queues
    (``open`` for cells on the rising side; ``pit`` for cells inside a
    depression). Critical detail: the descent test uses the *filled*
    DEM (``fdem``), but minimum / volume / nesting tests use the
    *original* DEM. Cells with ``fdem[n] < fdem[c]`` are descents
    (strict ``<``, not ``≤``).

    Returns ``(labels, breach, min_elev, parent, spill,
    cell_count, volume_sum)``. Per-label arrays are 0-indexed (k <-> label k+1).
    """
    import heapq

    H, W = dem.shape
    INF = np.float64(np.inf)
    dem_orig = dem.astype(np.float64)
    fdem = dem_orig.copy()
    closed = np.zeros((H, W), dtype=bool)
    labels = np.zeros((H, W), dtype=np.int32)
    breach = np.zeros((H, W), dtype=bool)

    # Per-label storage (1-indexed in `labels`, 0-indexed here)
    min_elev = []
    parent = []
    spill = []
    cell_count = []
    volume_sum = []

    open_q: list = []
    pit_q: list = []
    counter = 0

    def push_open(rr, cc):
        nonlocal counter
        heapq.heappush(open_q, (float(dem_orig[rr, cc]), counter, rr, cc))
        counter += 1

    # Seed Open with valid edge cells AND valid neighbours of nodata cells
    for r in range(H):
        for c in range(W):
            if not valid[r, c]:
                continue
            is_edge = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
            has_nod = False
            if not is_edge:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and not valid[rr, cc]:
                            has_nod = True
                            break
                    if has_nod:
                        break
            if (is_edge or has_nod) and not closed[r, c]:
                push_open(r, c)
                closed[r, c] = True

    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_q or pit_q:
        if pit_q:
            _, _, r, c = heapq.heappop(pit_q)
        else:
            _, _, r, c = heapq.heappop(open_q)

        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            if not valid[rr, cc] or closed[rr, cc]:
                continue
            closed[rr, cc] = True
            f_c = fdem[r, c]
            f_n = fdem[rr, cc]

            if f_n < f_c:
                # Descent (strict)
                lbl_c = labels[r, c]
                breach_c = breach[r, c]
                if lbl_c == 0 and not breach_c:
                    # New depression; current cell c is the breach
                    min_elev.append(INF)
                    parent.append(0)
                    spill.append(float(dem_orig[r, c]))
                    cell_count.append(0)
                    volume_sum.append(0.0)
                    new_lbl = len(min_elev)
                    breach[r, c] = True
                    labels[rr, cc] = new_lbl
                elif breach_c:
                    # Continuing expansion from a known breach: inherit last label
                    labels[rr, cc] = len(min_elev)
                else:
                    # Already inside a depression; check nesting
                    if (dem_orig[r, c] > min_elev[lbl_c - 1]
                            and dem_orig[rr, cc] < dem_orig[r, c]):
                        # Nested: c was ascending (dem[c] > parent's min), n descends again
                        min_elev.append(INF)
                        parent.append(int(lbl_c))
                        spill.append(float(dem_orig[r, c]))
                        cell_count.append(0)
                        volume_sum.append(0.0)
                        new_lbl = len(min_elev)
                        breach[r, c] = True
                        labels[rr, cc] = new_lbl
                    elif dem_orig[rr, cc] < dem_orig[r, c]:
                        labels[rr, cc] = lbl_c
                    elif dem_orig[rr, cc] >= spill[lbl_c - 1]:
                        # Exiting nested depression -> back to parent
                        labels[rr, cc] = parent[lbl_c - 1]
                    else:
                        labels[rr, cc] = lbl_c

                # Fill to spill height and push to Pit (priority = original elev)
                fdem[rr, cc] = f_c
                heapq.heappush(pit_q, (float(dem_orig[rr, cc]), counter, rr, cc))
                counter += 1
            else:
                # Climbing
                heapq.heappush(open_q, (float(dem_orig[rr, cc]), counter, rr, cc))
                counter += 1

        # Update per-depression aggregates for the popped cell c
        lc = labels[r, c]
        if lc > 0:
            d = dem_orig[r, c]
            sp = spill[lc - 1]
            cell_count[lc - 1] += 1
            # Clamp non-negative; see comment in numba core.
            if d < sp:
                volume_sum[lc - 1] += (sp - d)
            if d < min_elev[lc - 1]:
                min_elev[lc - 1] = d

    return (
        labels,
        breach,
        np.asarray(min_elev, dtype=np.float64),
        np.asarray(parent, dtype=np.int32),
        np.asarray(spill, dtype=np.float64),
        np.asarray(cell_count, dtype=np.int64),
        np.asarray(volume_sum, dtype=np.float64),
    )


def _priority_flood_breach_numba_factory():
    """Build numba-jitted version lazily (so import never fails)."""
    try:
        import numba
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True)
    def _heap_push(elev, idx, heap_e, heap_i, n):
        # binary heap (min) on elev with secondary tiebreak via idx
        heap_e[n] = elev
        heap_i[n] = idx
        i = n
        n2 = n + 1
        while i > 0:
            p = (i - 1) // 2
            if (heap_e[p] > heap_e[i]) or (
                heap_e[p] == heap_e[i] and heap_i[p] > heap_i[i]
            ):
                tmpe = heap_e[p]; heap_e[p] = heap_e[i]; heap_e[i] = tmpe
                tmpi = heap_i[p]; heap_i[p] = heap_i[i]; heap_i[i] = tmpi
                i = p
            else:
                break
        return n2

    @njit(cache=True)
    def _heap_pop(heap_e, heap_i, n):
        top_e = heap_e[0]
        top_i = heap_i[0]
        n2 = n - 1
        heap_e[0] = heap_e[n2]
        heap_i[0] = heap_i[n2]
        i = 0
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            best = i
            if l < n2 and (
                (heap_e[l] < heap_e[best]) or
                (heap_e[l] == heap_e[best] and heap_i[l] < heap_i[best])
            ):
                best = l
            if r < n2 and (
                (heap_e[r] < heap_e[best]) or
                (heap_e[r] == heap_e[best] and heap_i[r] < heap_i[best])
            ):
                best = r
            if best == i:
                break
            tmpe = heap_e[i]; heap_e[i] = heap_e[best]; heap_e[best] = tmpe
            tmpi = heap_i[i]; heap_i[i] = heap_i[best]; heap_i[best] = tmpi
            i = best
        return top_e, top_i, n2

    @njit(cache=True)
    def _grow(arr_f, arr_i_parent, arr_f_spill, arr_i_count, arr_f_vol, used):
        cap = arr_f.shape[0]
        if used < cap:
            return arr_f, arr_i_parent, arr_f_spill, arr_i_count, arr_f_vol
        new_cap = cap * 2
        new_min = np.zeros(new_cap, dtype=np.float64)
        new_par = np.zeros(new_cap, dtype=np.int32)
        new_sp = np.zeros(new_cap, dtype=np.float64)
        new_cn = np.zeros(new_cap, dtype=np.int64)
        new_vl = np.zeros(new_cap, dtype=np.float64)
        for i in range(cap):
            new_min[i] = arr_f[i]
            new_par[i] = arr_i_parent[i]
            new_sp[i] = arr_f_spill[i]
            new_cn[i] = arr_i_count[i]
            new_vl[i] = arr_f_vol[i]
        return new_min, new_par, new_sp, new_cn, new_vl

    @njit(cache=True)
    def _core(fdem, dem_orig, valid, connectivity):
        H, W = fdem.shape
        closed = np.zeros((H, W), dtype=np.bool_)
        labels = np.zeros((H, W), dtype=np.int32)
        breach = np.zeros((H, W), dtype=np.bool_)

        N = H * W
        open_e = np.empty(N + 1, dtype=np.float64)
        open_i = np.empty(N + 1, dtype=np.int64)
        pit_e = np.empty(N + 1, dtype=np.float64)
        pit_i = np.empty(N + 1, dtype=np.int64)
        idx_rc = np.empty((N + 1, 2), dtype=np.int32)
        nopen = 0
        npit = 0
        counter = 0

        cap = 1024
        min_elev = np.full(cap, np.inf, dtype=np.float64)
        parent = np.zeros(cap, dtype=np.int32)
        spill = np.zeros(cap, dtype=np.float64)
        cell_count = np.zeros(cap, dtype=np.int64)
        volume_sum = np.zeros(cap, dtype=np.float64)
        n_lbl = 0

        for r in range(H):
            for c in range(W):
                if not valid[r, c]:
                    continue
                is_edge = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
                if not is_edge:
                    has_nod = False
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if dr == 0 and dc == 0:
                                continue
                            rr = r + dr; cc = c + dc
                            if 0 <= rr < H and 0 <= cc < W:
                                if not valid[rr, cc]:
                                    has_nod = True
                                    break
                        if has_nod:
                            break
                    if not has_nod:
                        continue
                if closed[r, c]:
                    continue
                idx = counter
                idx_rc[idx, 0] = r
                idx_rc[idx, 1] = c
                nopen = _heap_push(dem_orig[r, c], idx, open_e, open_i, nopen)
                counter += 1
                closed[r, c] = True

        if connectivity == 8:
            n_off = 8
        else:
            n_off = 4
        off_r = np.empty(n_off, dtype=np.int32)
        off_c = np.empty(n_off, dtype=np.int32)
        if connectivity == 8:
            off_r[0] = -1; off_c[0] = -1
            off_r[1] = -1; off_c[1] = 0
            off_r[2] = -1; off_c[2] = 1
            off_r[3] = 0;  off_c[3] = -1
            off_r[4] = 0;  off_c[4] = 1
            off_r[5] = 1;  off_c[5] = -1
            off_r[6] = 1;  off_c[6] = 0
            off_r[7] = 1;  off_c[7] = 1
        else:
            off_r[0] = -1; off_c[0] = 0
            off_r[1] = 1;  off_c[1] = 0
            off_r[2] = 0;  off_c[2] = -1
            off_r[3] = 0;  off_c[3] = 1

        while nopen > 0 or npit > 0:
            if npit > 0:
                _, idx, npit = _heap_pop(pit_e, pit_i, npit)
            else:
                _, idx, nopen = _heap_pop(open_e, open_i, nopen)
            r = idx_rc[idx, 0]
            c = idx_rc[idx, 1]

            for k in range(n_off):
                rr = r + off_r[k]
                cc = c + off_c[k]
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                if not valid[rr, cc] or closed[rr, cc]:
                    continue
                closed[rr, cc] = True
                f_c = fdem[r, c]
                f_n = fdem[rr, cc]

                if f_n < f_c:
                    lbl_c = labels[r, c]
                    breach_c = breach[r, c]

                    if lbl_c == 0 and not breach_c:
                        if n_lbl >= cap:
                            new_cap = cap * 2
                            tmp = np.full(new_cap, np.inf, dtype=np.float64)
                            for i in range(cap):
                                tmp[i] = min_elev[i]
                            min_elev = tmp
                            tmp2 = np.zeros(new_cap, dtype=np.int32)
                            for i in range(cap):
                                tmp2[i] = parent[i]
                            parent = tmp2
                            tmp3 = np.zeros(new_cap, dtype=np.float64)
                            for i in range(cap):
                                tmp3[i] = spill[i]
                            spill = tmp3
                            tmp4 = np.zeros(new_cap, dtype=np.int64)
                            for i in range(cap):
                                tmp4[i] = cell_count[i]
                            cell_count = tmp4
                            tmp5 = np.zeros(new_cap, dtype=np.float64)
                            for i in range(cap):
                                tmp5[i] = volume_sum[i]
                            volume_sum = tmp5
                            cap = new_cap
                        min_elev[n_lbl] = np.inf
                        parent[n_lbl] = 0
                        spill[n_lbl] = dem_orig[r, c]
                        cell_count[n_lbl] = 0
                        volume_sum[n_lbl] = 0.0
                        n_lbl += 1
                        breach[r, c] = True
                        labels[rr, cc] = n_lbl
                    elif breach_c:
                        labels[rr, cc] = n_lbl  # last created label
                    else:
                        if (dem_orig[r, c] > min_elev[lbl_c - 1]
                                and dem_orig[rr, cc] < dem_orig[r, c]):
                            if n_lbl >= cap:
                                new_cap = cap * 2
                                tmp = np.full(new_cap, np.inf, dtype=np.float64)
                                for i in range(cap):
                                    tmp[i] = min_elev[i]
                                min_elev = tmp
                                tmp2 = np.zeros(new_cap, dtype=np.int32)
                                for i in range(cap):
                                    tmp2[i] = parent[i]
                                parent = tmp2
                                tmp3 = np.zeros(new_cap, dtype=np.float64)
                                for i in range(cap):
                                    tmp3[i] = spill[i]
                                spill = tmp3
                                tmp4 = np.zeros(new_cap, dtype=np.int64)
                                for i in range(cap):
                                    tmp4[i] = cell_count[i]
                                cell_count = tmp4
                                tmp5 = np.zeros(new_cap, dtype=np.float64)
                                for i in range(cap):
                                    tmp5[i] = volume_sum[i]
                                volume_sum = tmp5
                                cap = new_cap
                            min_elev[n_lbl] = np.inf
                            parent[n_lbl] = lbl_c
                            spill[n_lbl] = dem_orig[r, c]
                            cell_count[n_lbl] = 0
                            volume_sum[n_lbl] = 0.0
                            n_lbl += 1
                            breach[r, c] = True
                            labels[rr, cc] = n_lbl
                        elif dem_orig[rr, cc] < dem_orig[r, c]:
                            labels[rr, cc] = lbl_c
                        elif dem_orig[rr, cc] >= spill[lbl_c - 1]:
                            labels[rr, cc] = parent[lbl_c - 1]
                        else:
                            labels[rr, cc] = lbl_c

                    fdem[rr, cc] = f_c
                    idx_new = counter
                    idx_rc[idx_new, 0] = rr
                    idx_rc[idx_new, 1] = cc
                    npit = _heap_push(dem_orig[rr, cc], idx_new, pit_e, pit_i, npit)
                    counter += 1
                else:
                    idx_new = counter
                    idx_rc[idx_new, 0] = rr
                    idx_rc[idx_new, 1] = cc
                    nopen = _heap_push(dem_orig[rr, cc], idx_new, open_e, open_i, nopen)
                    counter += 1

            # Update aggregates for popped cell c
            lc = labels[r, c]
            if lc > 0:
                d = dem_orig[r, c]
                sp = spill[lc - 1]
                cell_count[lc - 1] += 1
                # Clamp to non-negative: geophysical depression volume is
                # water held below the spill point. Cells assigned to a
                # nested child but at elevation > child's spill (rare,
                # happens at the child/parent boundary) would otherwise
                # contribute negative volume.
                if d < sp:
                    volume_sum[lc - 1] += (sp - d)
                if d < min_elev[lc - 1]:
                    min_elev[lc - 1] = d

        return (
            labels,
            breach,
            min_elev[:n_lbl].copy(),
            parent[:n_lbl].copy(),
            spill[:n_lbl].copy(),
            cell_count[:n_lbl].copy(),
            volume_sum[:n_lbl].copy(),
        )

    return _core


_NUMBA_BREACH_CORE = None


def _priority_flood_breach(dem, valid, connectivity=8, use_numba=True):
    global _NUMBA_BREACH_CORE
    if use_numba:
        if _NUMBA_BREACH_CORE is None:
            _NUMBA_BREACH_CORE = _priority_flood_breach_numba_factory()
        if _NUMBA_BREACH_CORE is not None:
            dem_orig = dem.astype(np.float64)
            fdem = dem_orig.copy()  # mutated in-place by core
            return _NUMBA_BREACH_CORE(fdem, dem_orig, valid.astype(np.bool_),
                                      np.int32(connectivity))
    return _priority_flood_breach_python(dem, valid, connectivity=connectivity)


def detect_levees_breach(
    da_dem: xr.DataArray,
    *,
    connectivity: int = 8,
    min_volume_m3: float = 50_000.0,
    min_area_m2: float = 0.0,
    max_area_m2: Optional[float] = None,
    vertical_tol_m: Optional[float] = None,
    keep_nested: bool = True,
    min_length_m: float = 100.0,
    min_relief_m: float = 0.3,
    simplify_tol_m: Optional[float] = None,
    swath_half_width_m: float = 10.0,
    bridge_gap_m: float = 5.0,
    endpoint_snap_m: float = 50.0,
    max_merge_angle_deg: float = 70.0,
    require_colinearity: bool = False,
    max_sinuosity: float = 4.0,
    relief_quantile: float = 0.75,
    merge_strategy: str = "longest_path",
    min_component_area_px: int = 5,
    use_numba: bool = True,
    return_attrs_table: bool = False,
    logger: logging.Logger = logger,
) -> gpd.GeoDataFrame:
    """Detect levees as watershed boundaries of DEM depressions.

    Implements the adapted Improved Priority-Flood algorithm of
    Pronk et al. (2026) — a Python port of the Julia ``Breach.jl`` package.
    The method finds depressions in the DEM in a single sweep (Barnes 2014
    + nested-depression extension after Wu et al. 2019), and identifies the
    levee crest as the watershed boundary of each depression, with the
    spill point as the lowest cell on that boundary.

    Filter variants from the paper, exposed as kwargs:

    - **full**:     ``min_volume_m3=0``, ``vertical_tol_m=None`` (baseline).
    - **filtered**: ``min_volume_m3=50000``, ``vertical_tol_m=None``
      (paper default; drops natural depressions).
    - **partial**:  ``min_volume_m3=50000``, ``vertical_tol_m=3.0`` (only
      breach-cells within 3 m vertical of the spill point are kept; this
      tries to isolate the crest from the rest of the ring).

    Parameters
    ----------
    da_dem : xr.DataArray
        DEM in a projected CRS (meters).
    connectivity : {4, 8}
        Pixel connectivity for the priority flood. 8 (default) matches the
        paper / Breach.jl.
    min_volume_m3 : float
        Drop depressions whose ``volume = sum( max(spill - dem, 0) ) * dx^2``
        is below this. Paper default: 50 000.
    min_area_m2, max_area_m2 : float, optional
        Additional area-based filters on the depression footprint.
    vertical_tol_m : float, optional
        If set, only keep breach-mask cells whose original DEM elevation is
        within ``vertical_tol_m`` of the spill height. Paper "partial"
        variant: 3.0 m.
    keep_nested : bool
        If False, drop depressions whose ``parent_label != 0`` (only top-level
        rings survive).
    use_numba : bool
        Use numba-jitted core if available (~50x faster). Falls back to
        pure Python heapq.
    return_attrs_table : bool
        If True, return ``(gdf, df_depressions)`` where ``df_depressions`` is
        a DataFrame of all per-label attributes (regardless of filtering).

    Returns
    -------
    gpd.GeoDataFrame
        Columns: ``name, geometry, stype, width_m, score,
        volume_m3, area_m2, mean_depth_m, spill_height_m, min_elev_m,
        parent_label``. CRS = ``da_dem.raster.crs``.

    References
    ----------
    Pronk, M., Gawehn, M., Eleveld, M., Ledoux, H. (2026). Automated Levee
    Detection in Digital Elevation Models. EarthArXiv (preprint).
    Reference Julia implementation: https://github.com/evetion/Breach.jl
    Barnes, R., Lehman, C., Mulla, D. (2014). Priority-Flood. Computers &
    Geosciences 62, 117-127.
    Wu, Q. et al. (2019). Efficient Delineation of Nested Depression
    Hierarchy in DEMs. JAWRA 55, 354-368.
    """
    _assert_projected(da_dem)
    dem_arr, valid = _dem_to_numpy(da_dem)
    dx_m = _dem_pixel_size_m(da_dem)
    pixel_area = dx_m * dx_m
    crs = da_dem.raster.crs

    def _empty_result(n_lbl_=0, attrs=None):
        empty = gpd.GeoDataFrame(
            columns=["name", "stype", "width_m", "score", "geometry"],
            geometry="geometry", crs=crs,
        )
        if return_attrs_table:
            if attrs is None:
                attrs = pd.DataFrame(columns=[
                    "label", "min_elev_m", "spill_height_m", "parent_label",
                    "area_m2", "volume_m3", "mean_depth_m"])
            return empty, attrs
        return empty

    if not valid.any():
        logger.warning("detect_levees_breach: DEM is entirely nodata; empty result.")
        return _empty_result()

    logger.info(
        f"detect_levees_breach: running priority-flood on "
        f"{dem_arr.shape[0]}x{dem_arr.shape[1]} DEM "
        f"({'numba' if use_numba else 'python'} backend)..."
    )
    import time as _time
    t0 = _time.time()
    labels, breach, min_elev, parent, spill, cell_count, volume_sum = (
        _priority_flood_breach(dem_arr, valid, connectivity=connectivity,
                               use_numba=use_numba)
    )
    logger.info(
        f"detect_levees_breach: priority-flood done in "
        f"{_time.time()-t0:.1f}s; {len(min_elev)} depressions found, "
        f"{int(breach.sum())} breach cells."
    )

    n_lbl = len(min_elev)
    if n_lbl == 0:
        return _empty_result()

    # Per-depression attributes — first compute raw per-label aggregates,
    # then apply the Breach.jl post-loop "parent merge" step which lifts
    # each child's volume to its parent's spill reference and accumulates
    # area / min into the parent. Without this, parent volumes for nested
    # systems (polder-in-polder) under-report by (parent.spill − child.spill)
    # × child.area for each child.
    area_m2 = cell_count.astype(np.float64) * pixel_area
    volume_m3 = (volume_sum * pixel_area).copy()
    spill_height_m = spill.astype(np.float64).copy()
    parent_label = parent.astype(np.int32).copy()
    min_elev_merged = min_elev.astype(np.float64).copy()

    # Parents are always created before children in the priority-flood, so
    # parent_label[k-1] < k for all k. Iterating labels from highest to
    # lowest index thus processes every child before its parent.
    if n_lbl > 1:
        for k in range(n_lbl, 1, -1):  # 1-indexed labels
            p = int(parent_label[k - 1])
            if p == 0:
                continue
            child_vol = volume_m3[k - 1]
            child_area = area_m2[k - 1]
            child_spill = spill_height_m[k - 1]
            child_min = min_elev_merged[k - 1]
            parent_spill = spill_height_m[p - 1]
            # Lift child volume from child.spill up to parent.spill (the
            # extra water column over child's area), then add child volume
            # and area to parent (Breach.jl merge formula).
            volume_m3[p - 1] += child_vol + (parent_spill - child_spill) * child_area
            area_m2[p - 1] += child_area
            if child_min < min_elev_merged[p - 1]:
                min_elev_merged[p - 1] = child_min

    mean_depth_m = np.where(area_m2 > 0, volume_m3 / area_m2, 0.0)
    # `min_elev` and `spill` arrays returned to the caller still reference
    # the post-merge values for parents and the original values for children.
    min_elev = min_elev_merged

    # Build list of labels that pass the depression-level filters
    keep_lbl_mask = np.ones(n_lbl, dtype=bool)
    keep_lbl_mask &= (volume_m3 >= float(min_volume_m3))
    keep_lbl_mask &= (area_m2 >= float(min_area_m2))
    if max_area_m2 is not None:
        keep_lbl_mask &= (area_m2 <= float(max_area_m2))
    if not keep_nested:
        keep_lbl_mask &= (parent_label == 0)
    keep_labels = np.nonzero(keep_lbl_mask)[0] + 1  # convert to 1-indexed
    logger.info(
        f"detect_levees_breach: {len(keep_labels)}/{n_lbl} depressions "
        f"pass volume/area filters."
    )

    df_attrs = pd.DataFrame({
        "label": np.arange(1, n_lbl + 1),
        "min_elev_m": min_elev,
        "spill_height_m": spill_height_m,
        "parent_label": parent_label,
        "area_m2": area_m2,
        "volume_m3": volume_m3,
        "mean_depth_m": mean_depth_m,
    })

    if len(keep_labels) == 0:
        return _empty_result(attrs=df_attrs)

    # Build the watershed boundary mask for retained depressions.
    # CAVEAT (Pronk 2026 mismatch): Breach.jl traces the watershed via flow
    # directions; we approximate that with morphological neighbours. The
    # priority-flood labels INSIDE-basin cells (those descended INTO via
    # f_n<f_c). The breach/spill cell itself remains label-0 (or carries
    # the parent's label for nested cases). A pure inner-edge boundary
    # would land at basin-floor elevation, NOT on the levee crest. To
    # approximate the crest we take the UNION of:
    #   - inner edge: inside-cells with any 8-neighbour outside (these
    #     are at-or-near the basin floor adjacent to the rim);
    #   - outer edge: cells one step OUTSIDE the inside-mask (these
    #     are the rim/crest cells that the algorithm did not label).
    # The outer-edge cells include the actual levee crest, while the
    # inner-edge cells help anchor the polyline near the basin's
    # spill-saddle. Skeletonization downstream produces a single-cell
    # ring through these.
    label_root = np.zeros(n_lbl + 1, dtype=np.int32)  # 0 = not retained
    keep_set = set(int(k) for k in keep_labels)
    for lbl in range(1, n_lbl + 1):
        cur = lbl
        seen = []
        while cur != 0:
            if cur in keep_set:
                for s in seen + [cur]:
                    label_root[s] = cur
                break
            seen.append(cur)
            cur = int(parent_label[cur - 1])

    # Build the per-LABEL boundary mask via VECTORIZED grey morphology.
    # We build a `label_kept` field: cells with a retained label keep
    # their integer label, all other cells are 0. Then a 3×3
    # grey_erosion gives the local-min label, grey_dilation gives the
    # local-max label. A cell is on a label's boundary iff:
    #   - inner_k: cell has label k AND any 3x3 neighbour has a DIFFERENT
    #     label (either 0 or another retained label), captured by
    #     mn != label OR mx != label.
    #   - outer_k: cell has label 0 AND any 3x3 neighbour has a retained
    #     label (mx > 0).
    # This is mathematically equivalent to the per-label loop
    # (binary_erosion + binary_dilation per label) but runs in a single
    # pass instead of O(n_labels). Empirical: on Snohomish 1m DEM with
    # 319 retained labels, the per-label loop takes ~100 s; this
    # vectorized version takes ~0.6 s with 0-cell difference (160×).
    from scipy.ndimage import grey_erosion, grey_dilation
    keep_set_arr = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for k in keep_labels:
        keep_set_arr[int(k)] = int(k)
    label_kept = keep_set_arr[np.clip(labels, 0, None)]
    mn = grey_erosion(label_kept, size=(3, 3), mode="constant", cval=0)
    mx = grey_dilation(label_kept, size=(3, 3), mode="constant", cval=0)
    inner = (label_kept > 0) & ((mn != label_kept) | (mx != label_kept))
    outer = (label_kept == 0) & (mx > 0)
    boundary_mask = (inner | outer) & valid

    # Compute nearest-non-zero-label EDT once (shared by vertical_tol_m
    # filter AND the score_map below). This attributes outer-edge cells
    # (labels==0) to their physically-nearest depression so their spill
    # height / volume / etc. lookups don't return 0.
    if (labels == 0).any() and (labels > 0).any():
        _, (nz_r, nz_c) = ndimage.distance_transform_edt(
            labels == 0, return_indices=True
        )
        nearest_label = np.where(labels > 0, labels, labels[nz_r, nz_c])
    else:
        nearest_label = labels.copy()

    if vertical_tol_m is not None:
        # Partial variant: keep only boundary cells within ±vertical_tol_m
        # of the spill height of their root depression.
        spill_per_root = np.concatenate([[np.nan], spill_height_m])
        nearest_root = label_root[nearest_label]
        spill_at_pixel = np.where(nearest_root > 0,
                                  spill_per_root[nearest_root],
                                  np.nan)
        with np.errstate(invalid="ignore"):
            within_tol = np.abs(dem_arr - spill_at_pixel) <= float(vertical_tol_m)
        boundary_mask &= within_tol
        logger.info(
            f"detect_levees_breach: vertical_tol_m={vertical_tol_m} m -> "
            f"{int(boundary_mask.sum())} boundary cells retained."
        )

    logger.info(
        f"detect_levees_breach: watershed-boundary mask has "
        f"{int(boundary_mask.sum())} cells across {len(keep_labels)} depressions."
    )

    if not boundary_mask.any():
        logger.warning("detect_levees_breach: no boundary cells after filtering.")
        return _empty_result(attrs=df_attrs)

    breach_mask = boundary_mask

    # Standard postprocessing -> LineStrings.
    # Score = spill height of the nearest retained depression. Use the
    # `nearest_label` field computed above (EDT-attributed) so outer-edge
    # cells (labels==0) inherit the spill of their physically-nearest
    # depression instead of 0. Without this, every crest-side polyline
    # would have score=0 (since outer-edge cells dominate the polyline
    # vertices).
    spill_lookup = np.concatenate([[0.0], spill_height_m])
    score_map = spill_lookup[nearest_label.astype(np.int32)].astype(np.float32)

    gdf = _postprocess_mask_to_polylines(
        breach_mask,
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
        response_map=score_map,
        min_component_area_px=min_component_area_px,
        logger=logger,
    )

    # Attach per-line depression attributes by majority-vote of *root*
    # labels along each polyline (root = nearest retained ancestor).
    if not gdf.empty:
        transform = da_dem.raster.transform
        inv = ~transform
        per_line_root = np.zeros(len(gdf), dtype=np.int32)
        for i, geom in enumerate(gdf.geometry.values):
            xs = np.asarray([c[0] for c in geom.coords])
            ys = np.asarray([c[1] for c in geom.coords])
            rc = np.array([inv * (x, y) for x, y in zip(xs, ys)])
            # Floor (not round) — banker's-rounding picks the wrong cell.
            ri = np.clip(np.floor(rc[:, 1]).astype(int), 0, dem_arr.shape[0] - 1)
            ci = np.clip(np.floor(rc[:, 0]).astype(int), 0, dem_arr.shape[1] - 1)
            # Use the EDT-attributed `nearest_label` field (computed once
            # above for vertical_tol_m + score_map). A vertex on the
            # outer-edge or skeleton-pushed-outward (raw `labels`==0) gets
            # its physically-nearest depression's label, so the
            # majority-vote across line vertices reliably picks the right
            # root depression. The previous 3×3 block-vote on raw
            # `labels` returned empty when the polyline was >1 px from
            # any labeled cell (e.g. small/edge-adjacent basins),
            # cascading to depression_label=0 + per-line attrs all 0/NaN.
            verts_label = nearest_label[ri, ci]
            roots = label_root[verts_label]
            roots = roots[roots > 0]
            if roots.size > 0:
                counts = np.bincount(roots)
                per_line_root[i] = int(np.argmax(counts))
        idx = np.clip(per_line_root - 1, 0, n_lbl - 1)
        has_root = per_line_root > 0
        gdf["depression_label"] = per_line_root
        gdf["volume_m3"] = np.where(has_root, volume_m3[idx], 0.0)
        gdf["area_m2"] = np.where(has_root, area_m2[idx], 0.0)
        gdf["mean_depth_m"] = np.where(has_root, mean_depth_m[idx], 0.0)
        gdf["spill_height_m"] = np.where(has_root, spill_height_m[idx], np.nan)
        gdf["min_elev_m"] = np.where(has_root, min_elev[idx], np.nan)
        gdf["parent_label"] = np.where(has_root, parent_label[idx], 0)
        gdf["name"] = [f"levee_{i:05d}" for i in range(len(gdf))]

    if return_attrs_table:
        return gdf, df_attrs
    return gdf
