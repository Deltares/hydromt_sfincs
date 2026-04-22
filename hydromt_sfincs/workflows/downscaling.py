"""Downscaling workflows: high-resolution floodmap generation, WSE
dilation, velocity-head correction, and index-COG builder for
SFINCS quadtree/regular subgrid outputs.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import xugrid as xu
from rasterio.windows import Window
from pyproj import Transformer

import hydromt
from hydromt.data_catalog.drivers import RasterioDriver
from hydromt.gis.gis_utils import zoom_to_overview_level

from hydromt_sfincs.utils import build_overviews, read_xy


if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel


__all__ = [
    "downscale_floodmap",
    "dilate_zsmax",
    "apply_energy_head",
    "compute_flow_connected_mask",
    "remove_disconnected_flooding",
    "make_index_cog",
]


logger = logging.getLogger(f"hydromt.{__name__}")


# =============================================================================
#  Pre-step helpers (method-agnostic): WSE dilation, velocity-head correction
# =============================================================================


def dilate_zsmax(
    zsmax: Union[xu.UgridDataArray, xr.DataArray],
    factor: float,
) -> Union[xu.UgridDataArray, xr.DataArray]:
    """Cell-space WSE dilation — works on both quadtree and regular grids.

    For each *already-wet* cell/pixel, raise its zsmax to the max of (own,
    wet-neighbour values) within a radius of ``(0.5 + factor)`` cell widths.
    Dry cells stay dry — the wet-cell set is preserved (key safeguard:
    no new cells are flooded).

    Typical use is to close 1 m-DEM connectivity gaps behind coarse-cell
    levees, where the parent cell's single WSE sits below the levee crest
    on the fine DEM.  Expanding each cell's WSE plateau by a modest
    fraction of its own size lets the flood cross the crest continuously
    without introducing new wet cells elsewhere.

    Parameters
    ----------
    zsmax : xu.UgridDataArray or xr.DataArray
        Maximum water level (m).  NaN where dry.  Quadtree grids are
        dispatched to a cKDTree-based implementation on the cell centres;
        regular grids to a ``scipy.ndimage.maximum_filter`` with a disk
        footprint in pixel units.
    factor : float
        Fraction of cell size.  ``factor=0`` picks up only the cell itself
        on a uniform grid; ``factor=0.5`` reaches the 4 edge-neighbours;
        ``factor=1.0`` reaches ~1.5 cell widths (full 3×3 stencil on a
        uniform grid).  Must be ``>= 0``.  Returned unchanged when
        ``factor <= 0``.

    Returns
    -------
    xu.UgridDataArray or xr.DataArray
        Dilated zsmax on the same grid and with the same wet-cell set as
        the input.  ``dilated >= zsmax`` on every wet cell.
    """
    if isinstance(zsmax, xu.UgridDataArray):
        return _dilate_zsmax_quadtree(zsmax, factor)
    if isinstance(zsmax, xr.DataArray):
        return _dilate_zsmax_regular(zsmax, factor)
    raise TypeError(
        "zsmax must be xu.UgridDataArray or xr.DataArray; "
        f"got {type(zsmax).__name__}."
    )


def _dilate_zsmax_quadtree(
    zsmax: xu.UgridDataArray,
    factor: float,
) -> xu.UgridDataArray:
    """Quadtree-grid dilation via cKDTree (see :func:`dilate_zsmax`)."""
    from scipy.spatial import cKDTree

    grid = zsmax.ugrid.grid
    face_x, face_y = grid.face_coordinates.T
    fb = grid.face_bounds                     # (n, 4): xmin, ymin, xmax, ymax
    dx_cell = fb[:, 2] - fb[:, 0]
    dy_cell = fb[:, 3] - fb[:, 1]
    dcell = np.maximum(dx_cell, dy_cell)

    vals = zsmax.values.astype(np.float64, copy=True)
    wet_before = ~np.isnan(vals)

    if factor <= 0.0:
        out = zsmax.copy()
        out.values = vals.astype(zsmax.dtype)
        return out

    tree = cKDTree(np.column_stack([face_x, face_y]))
    radii = dcell * (0.5 + factor)

    nbr_lists = tree.query_ball_point(
        np.column_stack([face_x, face_y]), r=radii,
    )

    dilated = vals.copy()
    for i, nbrs in enumerate(nbr_lists):
        if not nbrs:
            continue
        nbr_vals = vals[nbrs]
        wet_nbrs = nbr_vals[~np.isnan(nbr_vals)]
        if wet_nbrs.size == 0:
            continue
        nbr_max = float(wet_nbrs.max())
        if np.isnan(dilated[i]):
            dilated[i] = nbr_max
        else:
            dilated[i] = max(dilated[i], nbr_max)

    # Enforce the no-new-wet-cells constraint
    dilated[~wet_before] = np.nan

    _check_dilation_invariants(vals, dilated, wet_before)

    out = zsmax.copy()
    out.values = dilated.astype(zsmax.dtype)
    return out


def _dilate_zsmax_regular(
    zsmax: xr.DataArray,
    factor: float,
) -> xr.DataArray:
    """Regular-grid dilation via a disk footprint max-filter.

    Footprint radius is ``(0.5 + factor)`` pixels (Euclidean), matching
    the quadtree convention: ``factor=0.5`` reaches edge neighbours,
    ``factor=1.0`` reaches diagonal (full 3×3 stencil).
    """
    from scipy.ndimage import maximum_filter

    vals = zsmax.values.astype(np.float64, copy=True)
    wet_before = ~np.isnan(vals)

    if factor <= 0.0:
        out = zsmax.copy()
        out.values = vals.astype(zsmax.dtype)
        return out

    # Disk footprint in pixel units (Euclidean radius 0.5 + factor)
    R = 0.5 + float(factor)
    r = int(np.ceil(R))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    footprint = (xx * xx + yy * yy) <= (R * R)

    # Fill NaN with -inf so the max ignores dry cells
    filled = np.where(wet_before, vals, -np.inf)
    dilated = maximum_filter(filled, footprint=footprint, mode="constant", cval=-np.inf)

    # Enforce no-new-wet-cells: dry cells stay dry
    dilated = np.where(wet_before, dilated, np.nan)

    # A wet cell with no wet neighbour in-range still has its own value
    # in the footprint centre, so dilated >= vals on every wet cell.
    _check_dilation_invariants(vals, dilated, wet_before)

    out = zsmax.copy()
    out.values = dilated.astype(zsmax.dtype)
    return out


def _check_dilation_invariants(vals, dilated, wet_before):
    """Assert wet-set preservation + monotonic lift for dilation helpers."""
    wet_after = ~np.isnan(dilated)
    if not np.array_equal(wet_before, wet_after):
        raise RuntimeError("dilation changed the wet-cell set")
    raised = np.where(
        wet_before,
        dilated - np.where(wet_before, vals, 0.0),
        0.0,
    )
    if not np.all(raised >= -1e-9):
        raise RuntimeError("dilation lowered zsmax on some cell")
def apply_energy_head(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    qmax: Union[xr.DataArray, xu.UgridDataArray],
    zb: Optional[Union[xr.DataArray, xu.UgridDataArray]] = None,
    hmin: float = 0.05,
    q_threshold: float = 0.01,
) -> Union[xr.DataArray, xu.UgridDataArray]:
    """Add the velocity head v²/(2g) to zsmax (Bernoulli correction).

    Lifts the water level on wet cells where the unit discharge exceeds
    ``q_threshold``, converting zsmax to the total-energy head
    ``H = zsmax + v² / (2g)``.  The wet-cell set is preserved: NaN cells
    stay NaN.

    This is a **method-agnostic pre-step** — the returned DataArray can be
    consumed by any downscaling method (constant, bilinear, raw, etc.).
    Works on both SFINCS quadtree grids (``xu.UgridDataArray``) and
    regular grids (``xr.DataArray``); ``zsmax``, ``qmax``, and ``zb`` must
    all share the same grid.

    Parameters
    ----------
    zsmax : xu.UgridDataArray or xr.DataArray
        Maximum water level (m) on a SFINCS grid — quadtree
        (``xu.UgridDataArray``) or regular (``xr.DataArray``).  NaN where
        dry.
    qmax : xu.UgridDataArray or xr.DataArray
        Maximum unit discharge magnitude (m²/s), **cell-centred** — one
        value per cell, with the same shape and grid as ``zsmax``.  This is
        the convention SFINCS writes to ``sfincs_map.nc`` (variable
        ``qmax``) when ``storefluxmax=1``; no face-to-centre reduction is
        needed.  The sign is ignored (``|qmax|`` is used internally).
        Formula: ``vel_head = q² / (h² · 2g)`` with
        ``h = max(zsmax - zb, hmin)``.
    zb : xu.UgridDataArray or xr.DataArray, optional
        Bed elevation (m) at cell centres, used to estimate depth.  If
        omitted, a constant depth of ``hmin`` is assumed (conservative —
        overestimates velocity and therefore the head correction).
    hmin : float, optional
        Minimum depth (m) for velocity estimation, by default 0.05.
    q_threshold : float, optional
        Minimum unit discharge magnitude (m²/s) to apply the correction,
        by default 0.01.  Cells below this threshold keep their original
        zsmax.

    Returns
    -------
    xu.UgridDataArray or xr.DataArray
        zsmax with the velocity head added on qualifying cells.  Same grid
        and same wet-cell set as the input.  ``result >= zsmax`` on every
        wet cell (velocity head is always non-negative).
    """
    GRAVITY = 9.81

    zs_vals = zsmax.values.astype(np.float64, copy=True)
    q_vals = np.abs(qmax.values.astype(np.float64, copy=False))
    wet_before = ~np.isnan(zs_vals)

    if zb is not None:
        zb_vals = zb.values.astype(np.float64, copy=False)
        h = np.where(wet_before, np.maximum(zs_vals - zb_vals, hmin), hmin)
    else:
        h = np.full_like(zs_vals, hmin)

    v = np.where(h > 0, q_vals / h, 0.0)
    dH = np.where(
        np.isfinite(q_vals) & (q_vals > q_threshold),
        0.5 * v * v / GRAVITY,
        0.0,
    )

    zs_new = zs_vals + np.where(wet_before, dH, 0.0)
    zs_new[~wet_before] = np.nan

    # Invariants
    wet_after = ~np.isnan(zs_new)
    if not np.array_equal(wet_before, wet_after):
        raise RuntimeError("energy-head correction changed the wet-cell set")
    raised = np.where(
        wet_before,
        zs_new - np.where(wet_before, zs_vals, 0.0),
        0.0,
    )
    if not np.all(raised >= -1e-9):
        raise RuntimeError("energy-head correction lowered zsmax on some cell")

    out = zsmax.copy()
    out.values = zs_new.astype(zsmax.dtype)
    return out

def downscale_floodmap(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: Union[Path, str, xr.DataArray],
    method: str = "constant",
    indices: Union[Path, str, xr.DataArray] = None,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    floodmap_fn: Union[Path, str] = None,
    zsmap_fn: Union[Path, str] = None,
    dilation: Optional[float] = None,
    energy_flux: bool = False,
    qmax: Union[xr.DataArray, xu.UgridDataArray] = None,
    zb: Union[xr.DataArray, xu.UgridDataArray] = None,
    q_threshold: float = 0.01,
    q_scale: float = 0.5,
    reproj_method: str = "nearest",
    zoom_level: Optional[Union[int, tuple]] = None,
    nrmax: int = 2000,
    logger=logger,
    **kwargs,
):
    """Create a downscaled floodmap for (model) region.

    Supports multiple downscaling methods via the *method* parameter:

    * ``"raw"`` -- Paint each DEM pixel with its parent cell's WSE
      (nearest-neighbor).  No DEM subtraction, no wet/dry masking.
      Only produces a water-level raster (*zsmap_fn*).
    * ``"constant"`` -- Assign each DEM pixel the WSE of its parent cell
      (optionally via a pre-computed index COG), then subtract the DEM.
      This is the classic "bathtub" approach.
    * ``"bilinear"`` -- Bilinearly interpolate WSE from surrounding cell
      centers (Sanders & Schubert 2019), then subtract the DEM.

    Parameters
    ----------
    zsmax : xr.DataArray or xu.UgridDataArray
        Maximum water level (m).  When multiple timesteps are present the
        maximum over all timesteps is used.
    dep : Path, str, or xr.DataArray
        High-resolution DEM (m) of the model region.
    method : str, optional
        Downscaling method, by default ``"constant"``.
    indices : Path, str, or xr.DataArray, optional
        Pre-computed cell-index raster (only used by ``"constant"``).
    hmin : float, optional
        Minimum water depth (m) to be considered flooded, by default 0.05.
        Ignored by ``"raw"``.
    gdf_mask : gpd.GeoDataFrame, optional
        Polygons to mask the output (area outside is set to NaN).
    floodmap_fn : Path or str, optional
        Output flood-depth GeoTIFF.  Required for all methods except ``"raw"``.
    zsmap_fn : Path or str, optional
        Output water-level GeoTIFF.
    dilation : float, optional
        Cell-space WSE dilation factor.  When ``> 0``, each wet cell's
        zsmax is raised to the maximum of its wet neighbours within a
        radius of ``(0.5 + dilation)`` cell widths.  Dry cells stay dry
        (the wet-cell set is preserved).  Works on both quadtree
        (``xu.UgridDataArray``) and regular (``xr.DataArray``) ``zsmax``
        via :func:`dilate_zsmax`.  Default ``None`` (no dilation).
    energy_flux : bool, optional
        Enable the Bernoulli velocity-head correction ``H = zsmax + v²/(2g)``
        on cells with ``|qmax| > q_threshold``.  Requires ``qmax``.  Routing
        depends on *method*:

        * ``method="bilinear"`` — ``qmax`` is passed into
          :func:`_downscale_bilinear`, which applies the per-cell Bernoulli
          lift *and* propagates upstream energy across wet edges, blended by
          ``q_scale``.
        * any other method — :func:`apply_energy_head` runs as a pre-step
          (pure per-cell Bernoulli; no upstream propagation, ``q_scale``
          unused).

        When ``False`` (default), ``qmax`` is ignored and no velocity-head
        correction is applied.
    qmax : xu.UgridDataArray or xr.DataArray, optional
        Maximum unit discharge magnitude (m²/s), cell-centred (same shape
        as ``zsmax``).  Matches the ``qmax`` variable SFINCS writes to
        ``sfincs_map.nc`` when ``storefluxmax=1``.  Only used when
        ``energy_flux=True``.
    zb : xu.UgridDataArray or xr.DataArray, optional
        Bed elevation at cell centres (m).  Used with *qmax* to compute
        water depth for velocity estimation.  If omitted, *hmin* is used as
        the minimum depth (conservative: overestimates velocity).
    q_threshold : float, optional
        Minimum unit discharge (m²/s) at which the velocity-head correction
        becomes active (below it, cells keep their original ``zsmax``), by
        default 0.01.  Used by both the pre-step and the bilinear path.
    q_scale : float, optional
        Upstream-energy-propagation blend scale (m²/s), only used by
        ``method="bilinear"`` when ``energy_flux=True``.  At each wet edge,
        the blend weight is ``min(1, |qmax| / q_scale)``; so ``q_scale``
        sets the unit discharge at which full upstream propagation kicks
        in.  Default 0.5.  Ignored for all other methods (they apply the
        pure per-cell Bernoulli pre-step instead).
    reproj_method : str, optional
        Reprojection method for ``"constant"`` downscaling, by default
        ``"nearest"``.
    zoom_level : int or tuple, optional
        Overview level of the raster dataset (only for ``"constant"``).
    nrmax : int, optional
        Block size in pixels, by default 2000.
    logger : logging.Logger, optional
        Logger instance.
    kwargs : dict, optional
        Extra keyword arguments forwarded to ``RasterDataArray.to_raster``
        (only for the in-memory ``"constant"`` path).

    Returns
    -------
    hmax : xr.DataArray or None
        Returned only when *dep* is an ``xr.DataArray`` and
        ``method="constant"``.  Otherwise results are written to disk.
    """
    _VALID_METHODS = {"raw", "constant", "bilinear"}
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}.  Choose from {sorted(_VALID_METHODS)}."
        )

    # --- Reduce time dimension -----------------------------------------------
    if isinstance(zsmax, xu.UgridDataArray):
        timedim = set(zsmax.dims) - set(zsmax.ugrid.grid.dims)
    else:
        timedim = set(zsmax.dims) - set(zsmax.raster.dims)
    if timedim:
        logger.info(f"Taking maximum water level over {timedim} dimension(s).")
        zsmax = zsmax.max(timedim)

    if qmax is not None:
        if isinstance(qmax, xu.UgridDataArray):
            q_timedim = set(qmax.dims) - set(qmax.ugrid.grid.dims)
        else:
            q_timedim = set(qmax.dims) - set(qmax.raster.dims)
        if q_timedim:
            qmax = qmax.max(q_timedim)

    # --- Pre-step 1: cell-space WSE dilation (quadtree or regular grid) ------
    if dilation is not None and dilation > 0.0:
        logger.info(f"Applying WSE dilation with factor={dilation:g}.")
        zsmax = dilate_zsmax(zsmax, factor=float(dilation))

    # --- Pre-step 2: Bernoulli velocity-head correction ----------------------
    # Route depends on the method:
    #   * bilinear     → qmax flows into _downscale_bilinear, which does
    #                    per-cell Bernoulli *and* upstream propagation.
    #   * other method → pure per-cell Bernoulli pre-step here; drop qmax so
    #                    downstream code doesn't see it.
    if energy_flux:
        if qmax is None:
            raise ValueError("energy_flux=True requires qmax.")
        if method == "bilinear":
            logger.info("Applying velocity-head + upstream propagation (bilinear).")
        else:
            logger.info("Applying velocity-head pre-step.")
            zsmax = apply_energy_head(
                zsmax, qmax=qmax, zb=zb, hmin=hmin, q_threshold=q_threshold,
            )
            qmax = None
    else:
        qmax = None  # ignored when the switch is off

    # --- In-memory path (xr.DataArray dep) -- only for "constant" ------------
    if isinstance(dep, xr.DataArray):
        if method != "constant":
            raise ValueError(
                "Only method='constant' supports xr.DataArray dep.  "
                "Use a file path for other methods."
            )
        if isinstance(floodmap_fn, Path):
            floodmap_fn = str(floodmap_fn)
        if indices is not None:
            if isinstance(indices, (str, Path)) and not isinstance(dep, (str, Path)):
                raise ValueError("index should be xr.DataArray when dep is xr.DataArray.")
            elif isinstance(indices, xr.DataArray) and not isinstance(dep, xr.DataArray):
                raise ValueError("index should be str/Path when dep is str/Path.")
        hmax = _downscale_floodmap_da(
            zsmax=zsmax, dep=dep, indices=indices, hmin=hmin,
            gdf_mask=gdf_mask, reproj_method=reproj_method,
        )
        if floodmap_fn is not None:
            if not kwargs:
                kwargs = dict(
                    driver="GTiff", tiled=True, blockxsize=256, blockysize=256,
                    compress="deflate", predictor=2, profile="COG",
                )
            hmax.raster.to_raster(floodmap_fn, **kwargs)
            build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)
        hmax.name = "hmax"
        hmax.attrs.update({"long_name": "Maximum flood depth", "units": "m"})
        return hmax

    # --- File-based path (dep is str/Path) -----------------------------------
    if method == "raw":
        if zsmap_fn is None:
            raise ValueError("zsmap_fn is required for method='raw'.")
    else:
        if floodmap_fn is None:
            raise ValueError("floodmap_fn is required when dep is a file path.")

    # Dispatch to the appropriate file-based implementation
    if method == "constant":
        _downscale_constant(
            zsmax=zsmax, dep=dep, indices=indices, hmin=hmin, gdf_mask=gdf_mask,
            floodmap_fn=floodmap_fn, zsmap_fn=zsmap_fn, reproj_method=reproj_method,
            zoom_level=zoom_level, nrmax=nrmax, logger=logger,
        )
    elif method == "raw":
        _downscale_raw(
            zsmax=zsmax, dep=dep, zsmap_fn=zsmap_fn, gdf_mask=gdf_mask,
            nrmax=nrmax, logger=logger, indices=indices,
        )
    elif method == "bilinear":
        _downscale_bilinear(
            zsmax=zsmax, dep=dep, hmin=hmin, gdf_mask=gdf_mask,
            floodmap_fn=floodmap_fn, zsmap_fn=zsmap_fn, nrmax=nrmax, logger=logger,
            indices=indices, qmax=qmax, zb=zb, q_threshold=q_threshold, q_scale=q_scale,
        )


# =============================================================================
# =============================================================================

def _open_dem_geometry(dep):
    """Read only grid geometry (no elevation values) from a DEM GeoTIFF."""
    with rasterio.open(str(dep)) as src:
        return dict(
            transform=src.transform,
            width=src.width,
            height=src.height,
            crs=src.crs,
            dx=src.transform[0],
            dy=src.transform[4],
        )


def _make_output_profile(geo):
    """Standard COG profile for float32 output rasters."""
    return dict(
        driver="GTiff", width=geo["width"], height=geo["height"],
        count=1, dtype=np.float32, crs=geo["crs"], transform=geo["transform"],
        tiled=True, blockxsize=256, blockysize=256,
        compress="deflate", predictor=2, profile="COG",
        nodata=np.nan, BIGTIFF="YES",
    )


def _create_output_rasters(profile, floodmap_fn=None, zsmap_fn=None):
    """Create empty output GeoTIFF(s)."""
    if floodmap_fn is not None:
        with rasterio.open(str(floodmap_fn), "w", **profile):
            pass
    if zsmap_fn is not None:
        with rasterio.open(str(zsmap_fn), "w", **profile):
            pass


def _apply_mask_and_overviews(
    floodmap_fn, zsmap_fn, gdf_mask, geo, logger,
):
    """Apply polygon mask and build overviews on output raster(s)."""
    if gdf_mask is not None:
        logger.info("Applying polygon mask...")
        from rasterio.features import geometry_mask
        mask = geometry_mask(
            gdf_mask.geometry,
            out_shape=(geo["height"], geo["width"]),
            transform=geo["transform"],
            invert=True, all_touched=True,
        )
        for fn in [floodmap_fn, zsmap_fn]:
            if fn is None:
                continue
            with rasterio.open(str(fn), "r+") as dst:
                data = dst.read(1)
                data[~mask] = np.nan
                dst.write(data, indexes=1)

    for fn in [floodmap_fn, zsmap_fn]:
        if fn is not None:
            build_overviews(fn=str(fn), resample_method="nearest", logger=logger)


# =============================================================================
#  Method: raw  (nearest-neighbor WSE, no DEM subtraction)

def _downscale_raw(zsmax, dep, zsmap_fn, gdf_mask, nrmax, logger, indices=None):
    vals = zsmax.values
    wet = ~np.isnan(vals)
    if np.sum(wet) < 1:
        logger.warning("No wet cells found."); return

    geo = _open_dem_geometry(dep)
    profile = _make_output_profile(geo)
    _create_output_rasters(profile, zsmap_fn=zsmap_fn)

    nrcb = nrmax
    nrbn = int(np.ceil(geo["height"] / nrcb))
    nrbm = int(np.ceil(geo["width"] / nrcb))
    total = nrbn * nrbm
    done = 0

    if indices is not None:
        # ----- Index-COG path: exact cell containment, no interpolation -----
        logger.info(f"Raw quadtree (index-COG): {np.sum(wet)} wet cells")
        nodata_idx = 2147483647
        indices_src = rasterio.open(str(indices))

        for ii in range(nrbm):
            bm0 = ii * nrcb; bm1 = min(bm0 + nrcb, geo["width"])
            for jj in range(nrbn):
                bn0 = jj * nrcb; bn1 = min(bn0 + nrcb, geo["height"])
                window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

                idx_block = indices_src.read(1, window=window)
                zs_block = np.full(idx_block.shape, np.nan, dtype=np.float32)
                valid = idx_block != nodata_idx
                zs_block[valid] = vals[idx_block[valid]]

                with rasterio.open(str(zsmap_fn), "r+") as dst:
                    dst.write(zs_block, window=window, indexes=1)

                done += 1
                if done % 25 == 0 or done == total:
                    logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

        indices_src.close()
    else:
        # ----- Fallback: NearestNDInterpolator (no index COG supplied) ------
        from scipy.interpolate import NearestNDInterpolator

        grid = zsmax.ugrid.grid
        face_x, face_y = grid.face_coordinates.T
        interpolator = NearestNDInterpolator(
            np.column_stack([face_x[wet], face_y[wet]]), vals[wet],
        )
        logger.info(f"Raw quadtree (nearest-interp fallback): {np.sum(wet)} wet cells")

        for ii in range(nrbm):
            bm0 = ii * nrcb; bm1 = min(bm0 + nrcb, geo["width"])
            for jj in range(nrbn):
                bn0 = jj * nrcb; bn1 = min(bn0 + nrcb, geo["height"])
                window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

                xx, yy = np.meshgrid(
                    geo["transform"][2] + (np.arange(bm0, bm1) + 0.5) * geo["dx"],
                    geo["transform"][5] + (np.arange(bn0, bn1) + 0.5) * geo["dy"],
                )
                zs_block = interpolator(
                    np.column_stack([xx.ravel(), yy.ravel()])
                ).reshape(xx.shape).astype(np.float32)

                with rasterio.open(str(zsmap_fn), "r+") as dst:
                    dst.write(zs_block, window=window, indexes=1)

                done += 1
                if done % 25 == 0 or done == total:
                    logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

    _apply_mask_and_overviews(None, zsmap_fn, gdf_mask, geo, logger)
    logger.info(f"Raw quadtree water level map saved to: {zsmap_fn}")


# =============================================================================
#  Method: constant  (index-COG based, file path)
# =============================================================================
def _downscale_constant(
    zsmax, dep, indices, hmin, gdf_mask,
    floodmap_fn, zsmap_fn, reproj_method, zoom_level, nrmax, logger,
):
    """File-based constant (bathtub) downscaling via _downscale_floodmap_da."""
    if isinstance(floodmap_fn, Path):
        floodmap_fn = str(floodmap_fn)

    # indices validation
    if indices is not None:
        if not isinstance(indices, (str, Path)):
            raise ValueError("indices should be str/Path when dep is str/Path.")

    if zoom_level is not None:
        zls_dict, crs = RasterioDriver._get_zoom_levels_and_crs(dep)
        overview_level = zoom_to_overview_level(
            zoom=zoom_level, zls_dict=zls_dict, source_crs=crs,
        )
        if overview_level:
            overview_level -= 1
        else:
            overview_level = None
    else:
        overview_level = None

    _open_kwargs = {"overview_level": overview_level} if overview_level is not None else {}
    with rasterio.open(dep, **_open_kwargs) as src:
        if indices is not None:
            indices_src = rasterio.open(indices, **_open_kwargs)

        n1, m1 = src.shape
        nrcb = nrmax
        nrbn = int(np.ceil(n1 / nrcb))
        nrbm = int(np.ceil(m1 / nrcb))

        merge_last_col = m1 % nrcb == 1
        merge_last_row = n1 % nrcb == 1
        if merge_last_col:
            nrbm -= 1
        if merge_last_row:
            nrbn -= 1

        profile = dict(
            driver="GTiff", width=src.width, height=src.height,
            count=1, dtype=np.float32, crs=src.crs, transform=src.transform,
            tiled=True, blockxsize=256, blockysize=256,
            compress="deflate", predictor=2, profile="COG",
            nodata=np.nan, BIGTIFF="YES",
        )
        with rasterio.open(floodmap_fn, "w", **profile):
            pass
        if zsmap_fn is not None:
            with rasterio.open(zsmap_fn, "w", **profile):
                pass

        total = nrbm * nrbn
        done = 0
        skipped = 0
        logger.info(
            f"Constant WSE: {total} blocks to process "
            f"({m1}x{n1} pixels, block size {nrcb})"
        )

        for ii in range(nrbm):
            bm0 = ii * nrcb
            bm1 = min(bm0 + nrcb, m1)
            if merge_last_col and ii == (nrbm - 1):
                bm1 += 1
            for jj in range(nrbn):
                bn0 = jj * nrcb
                bn1 = min(bn0 + nrcb, n1)
                if merge_last_row and jj == (nrbn - 1):
                    bn1 += 1

                window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

                # Read indices first — skip block early if no SFINCS cells
                if indices is not None:
                    block_idx = indices_src.read(window=window)
                    if np.all(block_idx == indices_src.nodata):
                        done += 1
                        skipped += 1
                        continue

                block_data = src.read(window=window)
                if np.all(np.isnan(block_data)):
                    done += 1
                    skipped += 1
                    continue

                if src.transform[1] == 0 and src.transform[3] == 0:
                    x_coords = src.transform[2] + (np.arange(bm0, bm1) + 0.5) * src.transform[0]
                    y_coords = src.transform[5] + (np.arange(bn0, bn1) + 0.5) * src.transform[4]
                    block_dep = xr.DataArray(
                        block_data.squeeze(), dims=("y", "x"),
                        coords={"y": ("y", y_coords), "x": ("x", x_coords)},
                    )
                    if indices is not None:
                        block_idx = xr.DataArray(
                            block_idx.squeeze(), dims=("y", "x"),
                            coords={"y": ("y", y_coords), "x": ("x", x_coords)},
                        )
                else:
                    cols, rows = np.meshgrid(np.arange(bm0, bm1), np.arange(bn0, bn1))
                    xc, yc = src.transform * (cols + 0.5, rows + 0.5)
                    block_dep = xr.DataArray(
                        block_data.squeeze(), dims=("y", "x"),
                        coords={"yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
                    )
                    if indices is not None:
                        block_idx = xr.DataArray(
                            block_idx.squeeze(), dims=("y", "x"),
                            coords={"yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
                        )

                block_dep.raster.set_crs(src.crs.to_epsg())
                if indices is not None:
                    block_idx.raster.set_nodata(int(indices_src.nodata))
                    block_idx.raster.set_crs(indices_src.crs.to_epsg())

                block_hmax = _downscale_floodmap_da(
                    zsmax=zsmax, dep=block_dep,
                    indices=block_idx if indices is not None else None,
                    hmin=hmin, gdf_mask=gdf_mask, reproj_method=reproj_method,
                )

                with rasterio.open(floodmap_fn, "r+") as fm:
                    fm.write(block_hmax.values, window=window, indexes=1)
                if zsmap_fn is not None:
                    block_zs = (block_hmax + block_dep).astype(np.float32)
                    block_zs = block_zs.where(~np.isnan(block_hmax))
                    with rasterio.open(zsmap_fn, "r+") as zs:
                        zs.write(block_zs.values, window=window, indexes=1)

                done += 1
                if done % 25 == 0 or done == total:
                    logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

        if skipped:
            logger.info(f"  Skipped {skipped}/{total} empty blocks")

    build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)
    if zsmap_fn is not None:
        build_overviews(fn=zsmap_fn, resample_method="nearest", logger=logger)


# =============================================================================
#  Method: bilinear  (LinearNDInterpolator, block-based)
# =============================================================================
def _downscale_bilinear(zsmax, dep, hmin, gdf_mask, floodmap_fn, zsmap_fn, nrmax, logger,
                         indices=None, qmax=None, zb=None, q_threshold=0.01, q_scale=0.5):
    from scipy.interpolate import LinearNDInterpolator

    grid = zsmax.ugrid.grid
    face_x, face_y = grid.face_coordinates.T
    vals = zsmax.values
    if np.sum(~np.isnan(vals)) < 3:
        logger.warning("Fewer than 3 wet cells; cannot interpolate."); return

    if qmax is not None:
        # qmax is face-based (one value per cell, same shape as zsmax)
        q_cell_max = np.abs(qmax.values).astype(np.float64)  # (n_faces,)

        # Step 2 — local Bernoulli: H_local = zsmax + (q/h)²/2g
        h_cell = np.maximum(vals - zb.values, hmin) if zb is not None else np.full(len(vals), hmin)
        vel_head = q_cell_max**2 / (h_cell**2 * 2.0 * 9.81)
        H_local = np.where(~np.isnan(vals), vals + vel_head, np.nan)

        # Step 3 — upstream energy propagation via face-face adjacency
        # edge_face_connectivity may raise IndexError on malformed quadtree grids
        # where the internal invert_dense call fails; fall back to KD-tree in that case.
        _ef_ok = False
        try:
            ef = grid.edge_face_connectivity
            _ef_ok = ef.ndim == 2 and ef.shape[1] >= 2
        except (IndexError, ValueError):
            pass
        if _ef_ok:
            both_valid = (ef[:, 0] >= 0) & (ef[:, 1] >= 0)
            f0_safe = np.where(ef[:, 0] >= 0, ef[:, 0], 0)
            f1_safe = np.where(ef[:, 1] >= 0, ef[:, 1], 0)
            edge_q = np.where(both_valid, np.maximum(q_cell_max[f0_safe], q_cell_max[f1_safe]), 0.0)
            active = (edge_q > q_threshold) & both_valid
            ef_f0 = ef[active, 0]
            ef_f1 = ef[active, 1]
        else:
            # Malformed connectivity — rebuild face pairs spatially via KD-tree
            from scipy.spatial import cKDTree as _cKDTree
            fb = grid.face_bounds          # (n_face, 4): xmin, ymin, xmax, ymax
            _dx = fb[:, 2] - fb[:, 0]
            _dy = fb[:, 3] - fb[:, 1]
            _n_face = len(face_x)
            _max_cell = max(_dx.max(), _dy.max())
            _tree = _cKDTree(np.column_stack([face_x, face_y]))
            _offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
            _probes = np.vstack([
                np.column_stack([face_x + _dx * _offsets[d, 0],
                                 face_y + _dy * _offsets[d, 1]])
                for d in range(4)
            ])
            _dists, _idxs = _tree.query(_probes, k=1, distance_upper_bound=_max_cell + 1.0)
            _src = np.tile(np.arange(_n_face), 4)
            _valid = (_dists < _max_cell + 0.5) & (_src != _idxs)
            _src, _idxs = _src[_valid], _idxs[_valid]
            _keep = _src < _idxs
            _pairs = np.unique(np.column_stack([_src[_keep], _idxs[_keep]]), axis=0)
            _f0_all, _f1_all = _pairs[:, 0], _pairs[:, 1]
            _edge_q_all = np.maximum(q_cell_max[_f0_all], q_cell_max[_f1_all])
            _active = _edge_q_all > q_threshold
            ef_f0 = _f0_all[_active]
            ef_f1 = _f1_all[_active]
            active = _active  # for .any() check below

        upstream_H = np.full(len(vals), np.nan)
        if active.any():
            f0, f1 = ef_f0, ef_f1
            H0, H1 = H_local[f0], H_local[f1]
            fwd = ~np.isnan(H0) & (np.isnan(H1) | (H0 > H1))
            rev = ~np.isnan(H1) & (np.isnan(H0) | (H1 > H0))
            np.fmax.at(upstream_H, f1[fwd], H0[fwd])
            np.fmax.at(upstream_H, f0[rev], H1[rev])

        # Step 4 — blend: H_eff = H_local + blend × max(0, upstream_H − H_local)
        blend = np.minimum(1.0, q_cell_max / q_scale)
        H_eff = H_local.copy()
        boosted = ~np.isnan(upstream_H) & ~np.isnan(H_local)
        H_eff[boosted] += blend[boosted] * np.maximum(0.0, upstream_H[boosted] - H_local[boosted])

        # Also extend to truly dry cells that receive upstream energy (transitional cells)
        dry_with_upstream = np.isnan(vals) & ~np.isnan(upstream_H)
        H_eff[dry_with_upstream] = upstream_H[dry_with_upstream] * blend[dry_with_upstream]

        n_boost = int(np.sum(boosted & (upstream_H > H_local)))
        n_trans = int(np.sum(dry_with_upstream))
        logger.info(f"Bilinear WSE: Bernoulli + upstream energy applied ({n_boost} boosted, {n_trans} transitional cells)")
    else:
        H_eff = vals.copy()

    wet_ext = ~np.isnan(H_eff)
    interpolator = LinearNDInterpolator(
        np.column_stack([face_x[wet_ext], face_y[wet_ext]]), H_eff[wet_ext],
    )
    logger.info(f"Bilinear WSE: interpolant from {np.sum(wet_ext)} cells")

    # Open index COG to mask pixels not belonging to a wet SFINCS cell.
    # Without this, LinearNDInterpolator fills across dry-cell gaps inside
    # the convex hull of wet cell centres.
    indices_src = None
    idx_nodata = None
    if indices is not None:
        indices_src = rasterio.open(str(indices))
        idx_nodata = indices_src.nodata
        logger.info("  Using index COG to mask dry-cell gaps")

    geo = _open_dem_geometry(dep)
    profile = _make_output_profile(geo)
    _create_output_rasters(profile, floodmap_fn, zsmap_fn)

    nrcb = nrmax
    nrbn = int(np.ceil(geo["height"] / nrcb))
    nrbm = int(np.ceil(geo["width"] / nrcb))
    total = nrbn * nrbm
    done = 0

    for ii in range(nrbm):
        bm0 = ii * nrcb; bm1 = min(bm0 + nrcb, geo["width"])
        for jj in range(nrbn):
            bn0 = jj * nrcb; bn1 = min(bn0 + nrcb, geo["height"])
            window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

            with rasterio.open(str(dep)) as src:
                dem_block = src.read(1, window=window).astype(np.float64)

            xx, yy = np.meshgrid(
                geo["transform"][2] + (np.arange(bm0, bm1) + 0.5) * geo["dx"],
                geo["transform"][5] + (np.arange(bn0, bn1) + 0.5) * geo["dy"],
            )
            zs_interp = interpolator(
                np.column_stack([xx.ravel(), yy.ravel()])
            ).reshape(dem_block.shape).astype(np.float32)

            # Mask pixels outside any wet SFINCS cell
            if indices_src is not None:
                idx_block = indices_src.read(1, window=window)
                outside = (idx_block == idx_nodata)
                # Also mask pixels whose parent cell is dry (NaN zsmax)
                inside = ~outside
                if inside.any():
                    pidx = idx_block[inside].astype(int)
                    parent_zs = vals[pidx]
                    parent_H  = H_eff[pidx]
                    dry_parent = np.isnan(parent_zs) & np.isnan(parent_H)
                    mask_arr = np.zeros_like(outside)
                    mask_arr[inside] = dry_parent
                    outside |= mask_arr
                zs_interp[outside] = np.nan

            hmax_block = (zs_interp - dem_block).astype(np.float32)
            hmax_block[np.isnan(hmax_block)] = np.nan
            hmax_block[hmax_block <= hmin] = np.nan
            hmax_block[np.isnan(dem_block)] = np.nan
            zs_block = zs_interp.copy()
            zs_block[np.isnan(hmax_block)] = np.nan

            if np.any(~np.isnan(hmax_block)):
                with rasterio.open(str(floodmap_fn), "r+") as dst:
                    dst.write(hmax_block, window=window, indexes=1)
                if zsmap_fn is not None:
                    with rasterio.open(str(zsmap_fn), "r+") as dst:
                        dst.write(zs_block, window=window, indexes=1)

            done += 1
            if done % 25 == 0 or done == total:
                logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

    if indices_src is not None:
        indices_src.close()

    _apply_mask_and_overviews(floodmap_fn, zsmap_fn, gdf_mask, geo, logger)
    logger.info(f"Bilinear WSE floodmap saved to: {floodmap_fn}")


def _cell_to_pixel_window(cxmin, cxmax, cymin, cymax, transform, width, height):
    """Convert cell bounding box to DEM pixel window.

    Returns pixel indices such that only pixels whose **center** falls within
    the cell are included.  This avoids counting an extra ring of pixels along
    cell edges (which would inflate volumes by ~20% for typical quadtree cells).

    Parameters
    ----------
    cxmin, cxmax, cymin, cymax : float
        Cell bounding box in map coordinates.
    transform : affine.Affine
        DEM affine transform.
    width, height : int
        DEM dimensions in pixels.

    Returns
    -------
    col0, col1, row0, row1 : int
        Pixel index window [col0:col1, row0:row1).  Returns None when the
        window is empty.
    """
    dx = transform[0]   # positive
    dy = transform[4]   # negative (north-up)

    # Pixel whose center is at: x = origin_x + (col + 0.5) * dx
    # Include pixel if center_x >= cxmin  =>  col >= (cxmin - origin_x) / dx - 0.5
    # Exclude pixel if center_x >= cxmax  =>  col >= (cxmax - origin_x) / dx - 0.5
    col0 = int(np.ceil((cxmin - transform[2]) / dx - 0.5))
    col1 = int(np.ceil((cxmax - transform[2]) / dx - 0.5))

    # y-axis: dy is negative, so row increases as y decreases
    # Pixel center_y = origin_y + (row + 0.5) * dy
    # Include pixel if center_y <= cymax  =>  row >= (cymax - origin_y) / dy - 0.5
    # Exclude pixel if center_y <= cymin  =>  row >= (cymin - origin_y) / dy - 0.5
    row0 = int(np.ceil((cymax - transform[5]) / dy - 0.5))
    row1 = int(np.ceil((cymin - transform[5]) / dy - 0.5))

    # Clip to DEM extent
    col0 = max(0, col0)
    col1 = min(width, col1)
    row0 = max(0, row0)
    row1 = min(height, row1)

    if col1 <= col0 or row1 <= row0:
        return None
    return col0, col1, row0, row1

def _downscale_floodmap_da(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: xr.DataArray,
    indices: xr.DataArray = None,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    reproj_method: str = "nearest",
) -> xr.DataArray:
    """Create a downscaled floodmap for (model) region.

    Parameters
    ----------
    zsmax : xr.DataArray
        Maximum water level (m). When multiple timesteps provided, maximum over all timesteps is used.
    dep : Path, str, xr.DataArray
        High-resolution DEM (m) of model region:
    hmin : float, optional
        Minimum water depth (m) to be considered as "flooded", by default 0.05
    gdf_mask : gpd.GeoDataFrame, optional
        Geodataframe with polygons to mask floodmap, example containing the landarea, by default None
        Note that the area outside the polygons is set to nodata.
    """

    if indices is None:
        # interpolate zsmax to dep grid
        if isinstance(zsmax, xr.DataArray):
            zsmax = zsmax.raster.reproject_like(dep, method=reproj_method)
        elif isinstance(zsmax, xu.UgridDataArray):
            # if non-rotated grid, use xugrid rasterize_like
            if dep.raster.transform[1] == 0 and dep.raster.transform[3] == 0:
                zsmax = zsmax.ugrid.rasterize_like(dep)
            # if rotated grid, use xugrid regridder
            else:
                # need to convert dep to unstructured to enable xugrid regridder
                uda_dep = xu.UgridDataArray.from_structured(dep, "xc", "yc")
                regridder = xu.CentroidLocatorRegridder(source=zsmax, target=uda_dep)
                result = regridder.regrid(zsmax)
                # map back to structured
                zsmax = dep.copy(data=result.values.reshape(dep.shape))

        zsmax = zsmax.raster.mask_nodata()  # make sure nodata is nan

        # get flood depth
        hmax = (zsmax - dep).astype("float32")
        hmax.raster.set_nodata(np.nan)
    else:
        # make sure index is same shape as dep
        if indices.shape != dep.shape:
            raise ValueError(
                "Indices shape {} does not match dep shape {}.".format(
                    indices.shape, dep.shape
                )
            )

        # Get the no_data value from the indices array
        nan_val_indices = indices.raster.nodata  # indices.attrs["_FillValue"]
        # Set the no_data mask
        no_data_mask = indices == nan_val_indices

        # Turn indices into numpy array and set no_data values to 0
        indices = np.squeeze(indices.values[:])
        indices[np.where(indices == nan_val_indices)] = 0

        zsmax = zsmax.raster.mask_nodata()  # make sure nodata is nan

        # Compute water depth
        zs_numpy = zsmax.values[:].flatten()
        h = zs_numpy[indices] - dep.values[:]

        # Set water depth to NaN where indices are no data
        h[no_data_mask] = np.nan

        # Turn h into a DataArray with the same dimensions as zb
        # ds = xr.Dataset()
        hmax = xr.DataArray(h, dims=["y", "x"], coords={"y": dep.y, "x": dep.x})
        hmax.raster.set_nodata(np.nan)
        hmax.raster.set_crs(dep.raster.crs)

    # mask floodmap
    hmax = hmax.where(hmax > hmin)

    if gdf_mask is not None:
        mask = hmax.raster.geometry_mask(gdf_mask, all_touched=True)
        hmax = hmax.where(mask)

    return hmax


def compute_flow_connected_mask(
    zsmax: xu.UgridDataArray,
    sfincs_nc: Union[Path, str],
    hmin: float = 0.05,
    zs_tol: float = 0.01,
    strict_downhill: bool = False,
    logger=logger,
):
    """Compute a boolean mask of cells reachable from boundary/source cells
    via the quadtree neighbor connectivity.

    Uses the neighbor connectivity arrays (mu1/md1/nu1/nd1) from sfincs.nc.
    Starting from boundary cells (mask == 2 or 3), a BFS flood fill propagates
    to neighboring wet cells.

    Two modes are supported:

    * **strict_downhill=False** (default): any wet neighbor is considered
      reachable.  This removes truly isolated wet cells that have no
      wet-neighbor path to the boundary — the most common artifact from
      bilinear interpolation.

    * **strict_downhill=True**: only propagate to neighbors whose WSE is at
      most ``zs_tol`` higher than the current cell.  This enforces a
      physical flow-direction constraint but may be too strict when using
      ``zsmax`` (maximum over time), because different cells may have
      peaked at different times.

    Parameters
    ----------
    zsmax : xu.UgridDataArray
        Maximum water level (m) on the quadtree mesh.
    sfincs_nc : Path or str
        Path to the SFINCS model definition file (sfincs.nc) containing
        neighbor connectivity and mask arrays.
    hmin : float, optional
        Minimum water depth threshold, by default 0.05.
    zs_tol : float, optional
        Tolerance (m) for the downhill criterion.  A neighbor with
        ``zs[nb] <= zs[ci] + zs_tol`` is considered downstream.
        Only used when *strict_downhill* is True.  Default 0.01 m.
    strict_downhill : bool, optional
        If True, only propagate to neighbors with lower (or near-equal) WSE.
        If False, propagate to any wet neighbor.  Default False.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    reachable : np.ndarray of bool, shape (n_faces,)
        True for cells that are connected to boundary sources.
    """
    from collections import deque

    # --- Reduce time dimension ---
    timedim = set(zsmax.dims) - set(zsmax.ugrid.grid.dims)
    if timedim:
        zsmax = zsmax.max(timedim)
    zs = zsmax.values
    n_faces = len(zs)

    # --- Read neighbor connectivity from sfincs.nc ---
    uds = xu.open_dataset(str(sfincs_nc))
    mask_arr = uds["mask"].values if "mask" in uds else np.ones(n_faces, dtype=int)

    # Neighbor arrays: 1-based indices, 0 = no neighbor
    nb_keys = []
    neighbors = {}
    for name in ["mu1", "md1", "nu1", "nd1", "mu2", "md2", "nu2", "nd2"]:
        if name in uds:
            arr = uds[name].values.astype(np.int64) - 1  # 0-based; original 0 -> -1
            neighbors[name] = arr
            nb_keys.append(name)

    # --- Identify source cells ---
    wet = ~np.isnan(zs) & (zs > -999)
    is_boundary = (mask_arr == 2) | (mask_arr == 3)
    sources = np.where(wet & is_boundary)[0]

    if len(sources) == 0:
        # Fallback: cells with no upstream wet neighbor
        logger.info("No boundary cells found; using local-maximum WSE cells.")
        is_source = np.zeros(n_faces, dtype=bool)
        for i in range(n_faces):
            if not wet[i]:
                continue
            has_upstream = False
            for key in nb_keys:
                nb = neighbors[key][i]
                if 0 <= nb < n_faces and wet[nb] and zs[nb] > zs[i] + zs_tol:
                    has_upstream = True
                    break
            if not has_upstream:
                is_source[i] = True
        sources = np.where(is_source)[0]

    logger.info(
        f"Flow connectivity: {len(sources)} source cells, "
        f"{np.sum(wet)} wet cells, strict_downhill={strict_downhill}"
    )

    # --- BFS from sources ---
    reachable = np.zeros(n_faces, dtype=bool)
    queue = deque()
    for s in sources:
        reachable[s] = True
        queue.append(s)

    while queue:
        ci = queue.popleft()
        for key in nb_keys:
            nb = neighbors[key][ci]
            if nb < 0 or nb >= n_faces:
                continue
            if reachable[nb] or not wet[nb]:
                continue
            if strict_downhill and zs[nb] > zs[ci] + zs_tol:
                continue  # skip uphill neighbors beyond tolerance
            reachable[nb] = True
            queue.append(nb)

    n_reachable = np.sum(reachable)
    n_disconnected = np.sum(wet) - n_reachable
    logger.info(
        f"  {n_reachable} cells reachable, "
        f"{n_disconnected} wet cells disconnected "
        f"({100*n_disconnected/max(1, np.sum(wet)):.1f}%)"
    )

    return reachable

def remove_disconnected_flooding(
    depth_fn: Union[Path, str],
    bnd_fn: Union[Path, str],
    hmin: float = 0.02,
    connection_fn: Union[Path, str] = None,
    output_fns: dict = None,
    logger=logger,
):
    """Remove disconnected flooding from a downscaled depth raster.

    Identifies wet pixels reachable from SFINCS boundary points via BFS
    flood-fill (8-connectivity), then masks pixels that are wet but
    unreachable.  Removes the need for a manually drawn source polygon.

    Uses a vectorised level-set BFS instead of ``scipy.ndimage.label``
    to avoid allocating a full int64 label array, which can exceed
    available memory for large high-resolution domains.

    Parameters
    ----------
    depth_fn : Path or str
        Path to the downscaled flood-depth GeoTIFF.
    bnd_fn : Path or str
        Path to the SFINCS boundary point file (``sfincs.bnd``).
        Each point marks a location where water enters the domain.
    hmin : float, optional
        Minimum water depth (m) to be considered wet, by default 0.02.
    connection_fn : Path or str, optional
        If provided, a connection-mask GeoTIFF is written here with values:
        0 = dry, 1 = connected to boundary, 2 = disconnected (wet but
        unreachable from any boundary point).
    output_fns : dict, optional
        Dictionary of ``{input_fn: output_fn}`` pairs.  For each pair the
        input raster is read, pixels where ``connection != 1`` are set to
        NaN, and the result is written to *output_fn*.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    None
    """
    # --- 1. Get raster metadata without loading the full array ---------------
    with rasterio.open(str(depth_fn)) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    # --- 2. Build wet mask tile-by-tile (avoids holding full depth) ----------
    wet = np.empty((height, width), dtype=np.bool_)
    with rasterio.open(str(depth_fn)) as src:
        for _, window in src.block_windows(1):
            r0, c0 = window.row_off, window.col_off
            block = src.read(1, window=window)
            wet[r0 : r0 + window.height, c0 : c0 + window.width] = block > hmin

    n_wet = int(np.sum(wet))
    logger.info(
        f"Disconnected-flooding removal: {n_wet} wet pixels "
        f"(hmin={hmin} m)"
    )

    # --- 3. Read boundary points and map to pixel coordinates ----------------
    bnd_gdf = read_xy(str(bnd_fn), crs=crs)
    bnd_x = np.array([p.x for p in bnd_gdf.geometry])
    bnd_y = np.array([p.y for p in bnd_gdf.geometry])

    inv_transform = ~transform
    bnd_col, bnd_row = inv_transform * (bnd_x, bnd_y)
    bnd_row = np.round(bnd_row).astype(int)
    bnd_col = np.round(bnd_col).astype(int)

    # --- 4. BFS flood-fill from boundary points (8-connectivity) -------------
    connected = np.zeros((height, width), dtype=np.bool_)
    seed_indices = []
    n_bnd_wet = 0

    for r, c in zip(bnd_row, bnd_col):
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    if wet[rr, cc] and not connected[rr, cc]:
                        connected[rr, cc] = True
                        seed_indices.append(rr * width + cc)
        if 0 <= r < height and 0 <= c < width and wet[r, c]:
            n_bnd_wet += 1

    logger.info(
        f"  {len(bnd_gdf)} boundary points, {n_bnd_wet} on wet pixels, "
        f"{len(seed_indices)} seed pixels for BFS"
    )

    # Vectorised level-set BFS: expand frontier one ring at a time using numpy
    _neighbors = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    wet_flat = wet.ravel()
    conn_flat = connected.ravel()
    frontier = np.array(seed_indices, dtype=np.intp)

    iteration = 0
    total_processed = len(frontier)
    while len(frontier) > 0:
        iteration += 1

        fr = frontier // width
        fc = frontier % width

        new_seeds = []
        for dr, dc in _neighbors:
            nr = fr + dr
            nc = fc + dc
            valid = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
            flat_idx = nr[valid] * width + nc[valid]
            mask = wet_flat[flat_idx] & ~conn_flat[flat_idx]
            new_pixels = flat_idx[mask]
            if len(new_pixels) > 0:
                conn_flat[new_pixels] = True
                new_seeds.append(new_pixels)

        if new_seeds:
            frontier = np.unique(np.concatenate(new_seeds))
            total_processed += len(frontier)
        else:
            frontier = np.array([], dtype=np.intp)

        if iteration % 500 == 0:
            logger.info(
                f"  BFS iteration {iteration}: {total_processed:,} pixels "
                f"reached, frontier {len(frontier):,}"
            )

    n_connected = int(np.sum(connected))
    n_disconnected = n_wet - n_connected
    logger.info(
        f"  {n_connected} connected pixels, "
        f"{n_disconnected} disconnected pixels removed "
        f"({100 * n_disconnected / max(1, n_wet):.1f}%)"
    )

    # Free wet mask — connected mask is all we need from here
    del wet, wet_flat

    # --- 5. Write connection mask raster (optional) --------------------------
    if connection_fn is not None:
        conn_profile = dict(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="int32",
            crs=crs,
            transform=transform,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            nodata=0,
        )
        with rasterio.open(str(depth_fn)) as src_dep:
            with rasterio.open(str(connection_fn), "w", **conn_profile) as dst:
                for _, window in dst.block_windows(1):
                    r0 = window.row_off
                    c0 = window.col_off
                    h_blk = window.height
                    w_blk = window.width
                    dep_blk = src_dep.read(1, window=window)
                    wet_blk = dep_blk > hmin
                    conn_blk = connected[r0 : r0 + h_blk, c0 : c0 + w_blk]
                    blk = np.zeros((h_blk, w_blk), dtype=np.int32)
                    blk[wet_blk] = 2
                    blk[conn_blk & wet_blk] = 1
                    dst.write(blk, 1, window=window)
        logger.info(f"  Connection mask written: {connection_fn}")

    # --- 6. Mask additional rasters (optional) -------------------------------
    if output_fns:
        for input_fn, output_fn in output_fns.items():
            with rasterio.open(str(input_fn)) as src_var:
                out_meta = src_var.meta.copy()
                out_meta.update(
                    dtype="float32",
                    nodata=np.nan,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    compress="deflate",
                    predictor=2,
                )
                with rasterio.open(str(output_fn), "w", **out_meta) as dst:
                    for _, window in dst.block_windows(1):
                        r0 = window.row_off
                        c0 = window.col_off
                        var_block = src_var.read(1, window=window)
                        conn_blk = connected[
                            r0 : r0 + window.height, c0 : c0 + window.width
                        ]
                        masked_block = np.where(
                            conn_blk, var_block, np.nan
                        ).astype(np.float32)
                        dst.write(masked_block, 1, window=window)
            logger.info(f"  Masked raster written: {output_fn}")

    return None




# =============================================================================
#  Index COG builder (topobathy → SFINCS cell-index raster)
# =============================================================================


def make_index_cog(
    model: "SfincsModel",
    indices_fn: Union[str, Path],
    topobathy_fn: Union[str, Path],
    nrmax: int = 2000,
    nodata: int = 2147483647,
):
    """Make a Cloud Optimzied Geotiff (COG) file with the correspodning indices of the SFINCS
    grid cells to the high-resolution DEM COG.

    Parameters
    ----------
    model : SfincsModel
        The SfincsModel instance containing the grid information.
    indices_fn : Union[str, Path]
        The filename for the output COG file containing the indices. Note that this file only works
        for this SFINCS model and the topobathy file provided.
    topobathy_fn : Union[str, Path]
        The filename of the topobathy file from which to read the coordinates.
    nrmax : int, optional
        The maximum number of cells in a block, by default 2000.
    nodata : int, optional
        The nodata value to use in the output COG file, by default 2147483647
        (which is the maximum value for a 32-bit unsigned integer).

    See also:
    ----------
    hydromt_sfincs.utils.build_overviews : Function to build overviews for the COG file.
    hydromt_sfincs.workflows.downscaling.downscale_floodmap : Workflow to downscale flood maps
    """

    # Read coordinates from topobathy file
    with rasterio.open(topobathy_fn) as src:
        # Get the CRS of the grid
        dem_crs = src.crs
        # Get the transform of the grid
        dem_transform = src.transform
        # Get the width and height of the grid
        width = src.width
        height = src.height

        n1, m1 = src.shape
        nrcb = nrmax  # nr of cells in a block
        nrbn = int(np.ceil(n1 / nrcb))  # nr of blocks in n direction
        nrbm = int(np.ceil(m1 / nrcb))  # nr of blocks in m direction

        # avoid blocks with width or height of 1
        merge_last_col = False
        merge_last_row = False
        if m1 % nrcb == 1:
            nrbm -= 1
            merge_last_col = True
        if n1 % nrcb == 1:
            nrbn -= 1
            merge_last_row = True

        profile = dict(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.uint32,
            crs=dem_crs,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            transform=dem_transform,
            nodata=nodata,
            predictor=2,
            profile="COG",
            BIGTIFF="YES",  # Add the BIGTIFF option here
        )

    with rasterio.open(indices_fn, "w", **profile):
        pass

    # Get the computational grid component of the model
    grid_comp = model.quadtree_grid if model.grid_type == "quadtree" else model.grid

    ## Loop through blocks
    for ii in range(nrbm):
        bm0 = ii * nrcb  # Index of first m in block
        bm1 = min(bm0 + nrcb, m1)  # last m in block
        if merge_last_col and ii == (nrbm - 1):
            bm1 += 1

        for jj in range(nrbn):
            bn0 = jj * nrcb  # Index of first n in block
            bn1 = min(bn0 + nrcb, n1)  # last n in block

            if merge_last_row and jj == (nrbn - 1):
                bn1 += 1

            # Define a window to read a block of data
            window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

            # Calculate the coordinates of the center of each pixel in the block
            x_coords = dem_transform[2] + (np.arange(bm0, bm1) + 0.5) * dem_transform[0]
            y_coords = dem_transform[5] + (np.arange(bn0, bn1) + 0.5) * dem_transform[4]
            xx, yy = np.meshgrid(x_coords, y_coords)

            # Transform the coordinates to the model's CRS and get the corresponding indices
            proj = Transformer.from_crs(dem_crs, model.crs, always_xy=True)
            xx, yy = proj.transform(xx, yy)
            indices = grid_comp.get_indices_at_points(xx, yy)
            indices[np.where(indices == -999)] = nodata

            # Fill the array with indices
            ii = np.empty((bn1 - bn0, bm1 - bm0), dtype=np.uint32)
            ii[:, :] = indices

            with rasterio.open(indices_fn, "r+") as fm_tif:
                fm_tif.write(
                    ii,
                    window=window,
                    indexes=1,
                )
        # add overviews
        build_overviews(fn=indices_fn, resample_method="nearest")
