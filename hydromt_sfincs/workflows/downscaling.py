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
from pyproj import CRS, Transformer

import hydromt
from hydromt.data_catalog.drivers import RasterioDriver
from hydromt.gis.gis_utils import zoom_to_overview_level

from hydromt_sfincs.utils import build_overviews
from hydromt_sfincs.readers import read_xy

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel


__all__ = [
    # ordered by pipeline step (see MODULE MAP below)
    "make_index_cog",  # Step 0  — build the cell-index COG
    "adjust_zsmax_dilation",  # Step 1a — pre-step
    "adjust_zsmax_energyhead",  # Step 1b — pre-step
    "downscale_floodmap",  # Step 2  — downscale
    "remove_disconnected_flooding",  # Step 3a — post-process
    "compute_flow_connected_mask",  # Step 3b — post-process helper
]


logger = logging.getLogger(f"hydromt.{__name__}")


# =============================================================================
#  MODULE MAP — the downscaling pipeline runs as ordered steps:
#
#    Step 0  Setup     make_index_cog             build the cell-index COG that
#                                                 the constant/raw methods consume
#    Step 1  Pre-steps (method-agnostic, optional; adjust zsmax before downscaling)
#              1a. adjust_zsmax_dilation          cell-space WSE dilation
#              1b. adjust_zsmax_energyhead        per-cell Bernoulli velocity head
#    Step 2  Downscale
#              2a. downscale_floodmap             public entry + per-method dispatch
#              2b. helpers                        block streaming + output plumbing
#              2c. engines                        raw / constant / bilinear + core
#    Step 3  Post-process the downscaled raster (optional)
#              3a. remove_disconnected_flooding   drop disconnected wet pools
#              3b. compute_flow_connected_mask    cell-level connectivity helper
#
#  Private helpers are prefixed with `_` and grouped under their step.
# =============================================================================


# =============================================================================
#  STEP 0 — Setup: build the SFINCS cell-index COG (consumed by constant/raw)
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


# =============================================================================
#  STEP 1 — Pre-steps (method-agnostic): adjust zsmax before downscaling
# =============================================================================


def _reduce_extra_dims(da):
    """Collapse any non-spatial dims (e.g. ``time``/``timemax``) to their max.

    The pre-steps operate on a single 2-D (regular) or 1-D (quadtree) field;
    a caller that passes a time-dimensioned ``zsmax``/``qmax`` would otherwise
    get a per-timestep result.  Reducing to the temporal maximum here mirrors
    the reduction the downscaler used to do internally and keeps the wet-cell
    invariants meaningful.  Returns ``None`` unchanged.
    """
    if da is None:
        return None
    if isinstance(da, xu.UgridDataArray):
        spatial = {da.ugrid.grid.face_dimension}
    else:
        spatial = {da.raster.y_dim, da.raster.x_dim}
    extra = [d for d in da.dims if d not in spatial]
    if extra:
        da = da.max(dim=extra)
    return da


# ---- 1a. adjust_zsmax_dilation : cell-space WSE dilation --------------------
def adjust_zsmax_dilation(
    zsmax: Union[xu.UgridDataArray, xr.DataArray],
    factor: float,
) -> Union[xu.UgridDataArray, xr.DataArray]:
    """Cell-space WSE dilation pre-step — works on both quadtree and regular grids.

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
    zsmax = _reduce_extra_dims(zsmax)
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
    """Quadtree-grid dilation via cKDTree (see :func:`adjust_zsmax_dilation`)."""
    from scipy.spatial import cKDTree

    grid = zsmax.ugrid.grid
    face_x, face_y = grid.face_coordinates.T
    fb = grid.face_bounds  # (n, 4): xmin, ymin, xmax, ymax
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
        np.column_stack([face_x, face_y]),
        r=radii,
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
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
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


# ---- 1b. adjust_zsmax_energyhead : per-cell Bernoulli velocity head ---------
def adjust_zsmax_energyhead(
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

    zsmax = _reduce_extra_dims(zsmax)
    qmax = _reduce_extra_dims(qmax)
    zb = _reduce_extra_dims(zb)

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


# =============================================================================
#  STEP 2 — Downscale: high-resolution floodmap from a SFINCS zsmax field
# =============================================================================


# ---- 2a. downscale_floodmap : public entry + per-method dispatch ------------
def downscale_floodmap(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: Union[Path, str, xr.DataArray],
    reproj_method: str = "nearest",
    subtract_dem: bool = True,
    indices: Union[Path, str, xr.DataArray] = None,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    floodmap_fn: Union[Path, str] = None,
    zsmap_fn: Union[Path, str] = None,
    zoom_level: Optional[Union[int, tuple]] = None,
    nrmax: int = 2000,
    logger=logger,
    **kwargs,
):
    """Create a downscaled floodmap for (model) region.

    The downscaler assigns every high-resolution DEM pixel a water-surface
    elevation (WSE) interpolated from the SFINCS ``zsmax`` field and — by
    default — subtracts the DEM to obtain flood depth.  Its behaviour is set by
    two orthogonal knobs:

    * ``reproj_method`` -- how the WSE is interpolated onto the DEM grid:
      ``"nearest"`` (each pixel takes the WSE of the SFINCS cell that contains
      it — the classic bathtub) or ``"bilinear"`` (interpolate between
      surrounding cell centres, Sanders & Schubert 2019).
    * ``subtract_dem`` -- when ``True`` (default) the DEM is subtracted to
      return flood depth ``hmax``; when ``False`` the raw interpolated water
      level is returned (no DEM subtraction, no ``hmin`` masking).

    WSE pre-adjustments (cell-space dilation, Bernoulli velocity head) are
    *not* part of this function — call :func:`adjust_zsmax_dilation` /
    :func:`adjust_zsmax_energyhead` on ``zsmax`` beforehand if needed.

    Parameters
    ----------
    zsmax : xr.DataArray or xu.UgridDataArray
        Maximum water level (m).  When multiple timesteps are present the
        maximum over all timesteps is used.
    dep : Path, str, or xr.DataArray
        High-resolution DEM (m) of the model region.
    reproj_method : {"nearest", "bilinear"}, optional
        WSE interpolation onto the DEM grid, by default ``"nearest"``.  On a
        regular grid ``"bilinear"`` uses the reproject engine; on a quadtree it
        uses a scattered interpolator over the cell centres.
    subtract_dem : bool, optional
        Subtract the DEM to return flood depth (``True``, default) or return
        the raw interpolated water level (``False``).
    indices : Path, str, or xr.DataArray, optional
        Pre-computed cell-index raster (exact containment; used for
        ``reproj_method="nearest"`` and to mask pixels outside the domain).
    hmin : float, optional
        Minimum water depth (m) to be considered flooded, by default 0.05.
        Ignored when ``subtract_dem`` is ``False``.
    gdf_mask : gpd.GeoDataFrame, optional
        Polygons to mask the output (area outside is set to NaN).
    floodmap_fn : Path or str, optional
        Output flood-depth GeoTIFF.  Required (for file input) when
        ``subtract_dem`` is ``True``.
    zsmap_fn : Path or str, optional
        Output water-level GeoTIFF.  Required (for file input) when
        ``subtract_dem`` is ``False``.
    zoom_level : int or tuple, optional
        Overview level of the raster dataset (regular-grid nearest only).
    nrmax : int, optional
        Block size in pixels, by default 2000.
    logger : logging.Logger, optional
        Logger instance.
    kwargs : dict, optional
        Extra keyword arguments forwarded to ``RasterDataArray.to_raster``
        (only for the in-memory path).

    Returns
    -------
    xr.DataArray
        The downscaled product: flood depth (``hmax``) when ``subtract_dem`` is
        ``True``, otherwise the water level.  Dry pixels are NaN (a domain with
        no flooding yields an all-NaN array).  File-based calls also write it to
        *floodmap_fn* / *zsmap_fn* and return a lazy (dask-backed) view re-opened
        from that file; the in-memory path returns the array directly.
    """
    if reproj_method not in ("nearest", "bilinear"):
        raise ValueError(
            f"Unknown reproj_method {reproj_method!r}. Choose 'nearest' or 'bilinear'."
        )

    # --- Reduce time dimension -----------------------------------------------
    if isinstance(zsmax, xu.UgridDataArray):
        timedim = set(zsmax.dims) - set(zsmax.ugrid.grid.dims)
    else:
        timedim = set(zsmax.dims) - set(zsmax.raster.dims)
    if timedim:
        logger.info(f"Taking maximum water level over {timedim} dimension(s).")
        zsmax = zsmax.max(timedim)

    # --- In-memory path (xr.DataArray dep) -----------------------------------
    if isinstance(dep, xr.DataArray):
        if isinstance(floodmap_fn, Path):
            floodmap_fn = str(floodmap_fn)
        if indices is not None and isinstance(indices, (str, Path)):
            raise ValueError("index should be xr.DataArray when dep is xr.DataArray.")
        out = _downscale_floodmap_da(
            zsmax=zsmax,
            dep=dep,
            indices=indices,
            hmin=hmin,
            gdf_mask=gdf_mask,
            reproj_method=reproj_method,
            subtract_dem=subtract_dem,
        )
        if subtract_dem and floodmap_fn is not None:
            if not kwargs:
                kwargs = dict(
                    driver="GTiff",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    compress="deflate",
                    predictor=2,
                    profile="COG",
                )
            out.raster.to_raster(floodmap_fn, **kwargs)
            build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)
        out.name = "hmax" if subtract_dem else "zsmax"
        out.attrs.update(
            {
                "long_name": (
                    "Maximum flood depth" if subtract_dem else "Maximum water level"
                ),
                "units": "m",
            }
        )
        return out

    # --- File-based path (dep is str/Path) -----------------------------------
    if subtract_dem:
        if floodmap_fn is None:
            raise ValueError("floodmap_fn is required when dep is a file path.")
    elif zsmap_fn is None:
        raise ValueError(
            "zsmap_fn is required for a raw water-level map (subtract_dem=False)."
        )

    # One streamer handles every case; it selects exact containment (index COG),
    # a reproject (regular grid) or a scattered interpolant (quadtree) per block.
    _downscale_floodmap_file(
        zsmax=zsmax,
        dep=dep,
        reproj_method=reproj_method,
        subtract_dem=subtract_dem,
        indices=indices,
        hmin=hmin,
        gdf_mask=gdf_mask,
        floodmap_fn=floodmap_fn,
        zsmap_fn=zsmap_fn,
        zoom_level=zoom_level,
        nrmax=nrmax,
        logger=logger,
    )

    # Re-open the written product so the call returns a DataArray (write-then-
    # read keeps the streamer memory-bounded).  With subtract_dem=False the
    # water level (zsmap) is the product; otherwise it is the flood depth.
    out_fn = floodmap_fn if subtract_dem else zsmap_fn
    if out_fn is None or not Path(str(out_fn)).exists():
        return None
    out_name = "hmax" if subtract_dem else "zsmax"
    long_name = "Maximum flood depth" if subtract_dem else "Maximum water level"
    da = _open_result_da(out_fn, out_name)
    da.attrs.update({"long_name": long_name, "units": "m"})
    return da


# ---- 2b. helpers : block streaming + output-raster plumbing -----------------
def _open_result_da(fn, name):
    """Re-open a written GeoTIFF as a (lazy) DataArray for the return value.

    Uses dask-backed chunks so the array is not eagerly loaded — large domains
    stay memory-bounded and the caller computes only what it needs.
    """
    import rioxarray

    da = rioxarray.open_rasterio(
        str(fn), masked=True, chunks={"band": 1, "y": 2048, "x": 2048}
    )
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    da.name = name
    return da


def _stream_blocks(width, height, nrmax, merge_singletons=False):
    """Yield ``(window, bm0, bm1, bn0, bn1)`` tiles over a raster of size
    ``width`` x ``height`` in blocks of ``nrmax`` pixels.

    ``bm`` indexes columns (x / width), ``bn`` indexes rows (y / height).
    When ``merge_singletons`` is True a trailing 1-pixel column/row is merged
    into the previous block (avoids degenerate 1-px tiles) — matching the
    constant-method behaviour.
    """
    nrbm = int(np.ceil(width / nrmax))
    nrbn = int(np.ceil(height / nrmax))
    merge_last_col = merge_singletons and (width % nrmax == 1)
    merge_last_row = merge_singletons and (height % nrmax == 1)
    if merge_last_col:
        nrbm -= 1
    if merge_last_row:
        nrbn -= 1
    for ii in range(nrbm):
        bm0 = ii * nrmax
        bm1 = min(bm0 + nrmax, width)
        if merge_last_col and ii == (nrbm - 1):
            bm1 += 1
        for jj in range(nrbn):
            bn0 = jj * nrmax
            bn1 = min(bn0 + nrmax, height)
            if merge_last_row and jj == (nrbn - 1):
                bn1 += 1
            yield Window(bm0, bn0, bm1 - bm0, bn1 - bn0), bm0, bm1, bn0, bn1


def _block_pixel_centres(transform, bm0, bm1, bn0, bn1):
    """Map coordinates of pixel centres for a block window, rotation-aware.

    Applies the full affine ``transform`` so the rotation terms
    (``transform[1]``, ``transform[3]``) are honoured; for an axis-aligned
    DEM this reduces to the usual ``origin + (col + 0.5) * dx`` grid.  Returns
    2-D ``(xx, yy)`` arrays of shape ``(bn1 - bn0, bm1 - bm0)``.
    """
    cols, rows = np.meshgrid(np.arange(bm0, bm1), np.arange(bn0, bn1))
    xx, yy = transform * (cols + 0.5, rows + 0.5)
    return xx, yy


def _canonical_cellfield(zsmax):
    """Reorder a regular ``zsmax`` to SFINCS-canonical ``(y, x)`` south-up layout.

    The index COG stores, per pixel, the flat cell index ``iind*nmax + jind``
    (Fortran order over the ``(nmax, mmax)`` cell grid, with ``iind`` the
    column/x-index and ``jind`` the row/y-index *increasing in the +y
    direction* — see :meth:`RegularGrid.get_indices_at_points`).  A consumer
    that flattens ``zsmax.values`` in Fortran order must therefore hold the
    field as ``(y, x)`` with ``y`` ascending (south-up).  This normalises a
    transposed or north-up regular field to that layout so the flat lookup
    cannot silently scramble.  Quadtree (1-D) fields are returned unchanged.
    """
    if isinstance(zsmax, xu.UgridDataArray):
        return zsmax
    if zsmax.ndim != 2:
        # Pre-step reductions are expected to leave a 2-D field; bail rather
        # than guess an order for higher-rank input.
        return zsmax
    y_dim, x_dim = zsmax.raster.y_dim, zsmax.raster.x_dim
    zsmax = zsmax.transpose(y_dim, x_dim)
    y = np.asarray(zsmax[y_dim].values)
    if y.size > 1 and y[0] > y[-1]:
        zsmax = zsmax.isel({y_dim: slice(None, None, -1)})
    return zsmax


# ---- 2c. engine : single file-based streamer (shared core per block) --------
def _downscale_floodmap_file(
    zsmax,
    dep,
    reproj_method,
    subtract_dem,
    indices,
    hmin,
    gdf_mask,
    floodmap_fn,
    zsmap_fn,
    zoom_level,
    nrmax,
    logger,
):
    """Stream a DEM in blocks, downscaling each block through the shared core.

    Every variant (nearest/bilinear, depth/raw) uses this one block loop; they
    differ only in how :func:`_downscale_floodmap_da` interpolates the water
    level.  A quadtree scattered interpolant is built once and reused across
    blocks.  Flood depth is written to ``floodmap_fn`` (when ``subtract_dem``)
    and the interpolated water level to ``zsmap_fn`` (when given).
    """
    if isinstance(floodmap_fn, Path):
        floodmap_fn = str(floodmap_fn)
    if isinstance(zsmap_fn, Path):
        zsmap_fn = str(zsmap_fn)
    if indices is not None and not isinstance(indices, (str, Path)):
        raise ValueError("indices should be str/Path when dep is str/Path.")

    is_quadtree = isinstance(zsmax, xu.UgridDataArray)
    write_floodmap = subtract_dem and floodmap_fn is not None
    write_zsmap = zsmap_fn is not None

    # Build the scattered quadtree interpolant once and reuse it across blocks.
    # Only bilinear needs it; quadtree nearest uses exact cell containment in the
    # core (index-COG lookup, or rasterize_like when no index).  Regular grids
    # reproject per block, so they need no pre-built interpolant.
    interpolator = None
    if is_quadtree and reproj_method == "bilinear":
        wet = int(np.sum(~np.isnan(zsmax.values)))
        if wet < 3:
            logger.warning(
                "Fewer than 3 wet cells; cannot interpolate. Writing empty maps."
            )
        interpolator = _build_scatter_interpolator(zsmax, "bilinear")

    if zoom_level is not None:
        zls_dict, crs = RasterioDriver._get_zoom_levels_and_crs(dep)
        overview_level = zoom_to_overview_level(
            zoom=zoom_level, zls_dict=zls_dict, source_crs=crs
        )
        overview_level = overview_level - 1 if overview_level else None
    else:
        overview_level = None
    _open_kwargs = (
        {"overview_level": overview_level} if overview_level is not None else {}
    )

    with rasterio.open(dep, **_open_kwargs) as src:
        indices_src = (
            rasterio.open(indices, **_open_kwargs) if indices is not None else None
        )
        n1, m1 = src.shape  # rows (height), cols (width); used in the log below

        profile = dict(
            driver="GTiff",
            width=src.width,
            height=src.height,
            count=1,
            dtype=np.float32,
            crs=src.crs,
            transform=src.transform,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            predictor=2,
            profile="COG",
            nodata=np.nan,
            BIGTIFF="YES",
        )
        if write_floodmap:
            with rasterio.open(floodmap_fn, "w", **profile):
                pass
        if write_zsmap:
            with rasterio.open(zsmap_fn, "w", **profile):
                pass

        windows = list(
            _stream_blocks(src.width, src.height, nrmax, merge_singletons=True)
        )
        total = len(windows)
        done = 0
        skipped = 0
        logger.info(
            f"Downscaling ({reproj_method}, subtract_dem={subtract_dem}): "
            f"{total} blocks ({m1}x{n1} pixels, block size {nrmax})"
        )

        for window, bm0, bm1, bn0, bn1 in windows:
            # Read indices first — skip block early if no SFINCS cells.
            if indices_src is not None:
                block_idx = indices_src.read(window=window)
                if np.all(block_idx == indices_src.nodata):
                    done += 1
                    skipped += 1
                    continue

            block_data = src.read(window=window).astype(np.float32)
            # Mask the DEM's own nodata sentinel (e.g. -9999) to NaN so it is
            # never treated as bathymetry (h = zsmax - (-9999) would explode).
            if src.nodata is not None and not np.isnan(src.nodata):
                block_data[block_data == src.nodata] = np.nan
            # Depth needs a DEM; a raw water-level map paints regardless.
            if subtract_dem and np.all(np.isnan(block_data)):
                done += 1
                skipped += 1
                continue

            if src.transform[1] == 0 and src.transform[3] == 0:
                x_coords = (
                    src.transform[2] + (np.arange(bm0, bm1) + 0.5) * src.transform[0]
                )
                y_coords = (
                    src.transform[5] + (np.arange(bn0, bn1) + 0.5) * src.transform[4]
                )
                block_dep = xr.DataArray(
                    block_data.squeeze(),
                    dims=("y", "x"),
                    coords={"y": ("y", y_coords), "x": ("x", x_coords)},
                )
                if indices_src is not None:
                    block_idx = xr.DataArray(
                        block_idx.squeeze(),
                        dims=("y", "x"),
                        coords={"y": ("y", y_coords), "x": ("x", x_coords)},
                    )
            else:
                cols, rows = np.meshgrid(np.arange(bm0, bm1), np.arange(bn0, bn1))
                xc, yc = src.transform * (cols + 0.5, rows + 0.5)
                block_dep = xr.DataArray(
                    block_data.squeeze(),
                    dims=("y", "x"),
                    coords={"yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
                )
                if indices_src is not None:
                    block_idx = xr.DataArray(
                        block_idx.squeeze(),
                        dims=("y", "x"),
                        coords={"yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
                    )

            block_dep.raster.set_crs(src.crs.to_wkt())
            if indices_src is not None:
                block_idx.raster.set_nodata(int(indices_src.nodata))
                block_idx.raster.set_crs(indices_src.crs.to_wkt())

            block_out, block_zs = _downscale_floodmap_da(
                zsmax=zsmax,
                dep=block_dep,
                indices=block_idx if indices_src is not None else None,
                hmin=hmin,
                gdf_mask=gdf_mask,
                reproj_method=reproj_method,
                subtract_dem=subtract_dem,
                return_zs=True,
                interpolator=interpolator,
            )

            if write_floodmap:
                with rasterio.open(floodmap_fn, "r+") as fm:
                    fm.write(
                        block_out.values.astype(np.float32), window=window, indexes=1
                    )
            if write_zsmap:
                with rasterio.open(zsmap_fn, "r+") as zs:
                    zs.write(
                        block_zs.values.astype(np.float32), window=window, indexes=1
                    )

            done += 1
            if done % 25 == 0 or done == total:
                logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

        if skipped:
            logger.info(f"  Skipped {skipped}/{total} empty blocks")
        if indices_src is not None:
            indices_src.close()

    if write_floodmap:
        build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)
    if write_zsmap:
        build_overviews(fn=zsmap_fn, resample_method="nearest", logger=logger)


#  core helpers — interpolate a SFINCS zsmax field onto a DEM grid ------------
def _build_scatter_interpolator(zsmax: xu.UgridDataArray, reproj_method: str):
    """Scattered interpolant over the wet quadtree cell centres.

    ``"bilinear"`` -> linear barycentric interpolation (Sanders & Schubert
    2019); anything else -> nearest cell centre.  Built once over the whole
    mesh so the file-based streamer can reuse it across every block.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    grid = zsmax.ugrid.grid
    face_x, face_y = grid.face_coordinates.T
    vals = np.asarray(zsmax.values)
    wet = ~np.isnan(vals)
    points = np.column_stack([face_x[wet], face_y[wet]])
    if reproj_method == "bilinear":
        return LinearNDInterpolator(points, vals[wet])
    return NearestNDInterpolator(points, vals[wet])


def _scatter_zs_on_dep(interpolator, dep: xr.DataArray, model_crs) -> xr.DataArray:
    """Query a scattered interpolant at every DEM pixel centre.

    Pixel centres are computed in the DEM's CRS and transformed to the model
    CRS before querying — otherwise a DEM and model in different projections
    silently return all-NaN (the query lands outside the interpolant's hull).
    """
    x_dim, y_dim = dep.raster.x_dim, dep.raster.y_dim
    width, height = dep[x_dim].size, dep[y_dim].size
    xx, yy = _block_pixel_centres(dep.raster.transform, 0, width, 0, height)
    dep_crs = dep.raster.crs
    if model_crs is not None and dep_crs is not None:
        try:
            same_crs = CRS.from_user_input(dep_crs) == CRS.from_user_input(model_crs)
        except Exception:
            same_crs = False
        if not same_crs:
            transformer = Transformer.from_crs(dep_crs, model_crs, always_xy=True)
            xx, yy = transformer.transform(xx, yy)
    vals = (
        interpolator(np.column_stack([np.ravel(xx), np.ravel(yy)]))
        .reshape(np.shape(xx))
        .astype("float32")
    )
    zs = dep.copy(data=vals)
    zs.raster.set_nodata(np.nan)
    return zs


def _interp_zs_on_dep(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: xr.DataArray,
    reproj_method: str,
    interpolator=None,
) -> xr.DataArray:
    """Interpolate ``zsmax`` (regular or quadtree) onto the DEM grid.

    * a pre-built ``interpolator`` (file-based streamer) -> scattered query;
    * regular grid -> reproject engine (nearest/bilinear, CRS-aware);
    * quadtree + bilinear -> scattered linear interpolant over cell centres;
    * quadtree + nearest -> exact containment via xugrid rasterize/regrid.
    """
    if interpolator is not None:
        model_crs = (
            zsmax.ugrid.grid.crs
            if isinstance(zsmax, xu.UgridDataArray)
            else zsmax.raster.crs
        )
        return _scatter_zs_on_dep(interpolator, dep, model_crs)

    if isinstance(zsmax, xr.DataArray):
        zs_on_dep = zsmax.raster.reproject_like(dep, method=reproj_method)
        return zs_on_dep.raster.mask_nodata()

    # quadtree (xu.UgridDataArray)
    if reproj_method == "bilinear":
        interp = _build_scatter_interpolator(zsmax, "bilinear")
        return _scatter_zs_on_dep(interp, dep, zsmax.ugrid.grid.crs)

    # nearest: exact cell containment
    if dep.raster.transform[1] == 0 and dep.raster.transform[3] == 0:
        zs_on_dep = zsmax.ugrid.rasterize_like(dep)
    else:
        uda_dep = xu.UgridDataArray.from_structured2d(dep, "xc", "yc")
        regridder = xu.CentroidLocatorRegridder(source=zsmax, target=uda_dep)
        zs_on_dep = dep.copy(data=regridder.regrid(zsmax).values.reshape(dep.shape))
    return zs_on_dep.raster.mask_nodata()


#  core — per-block zs->depth/WSE engine (single interpolation, shared by all paths)
def _downscale_floodmap_da(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: xr.DataArray,
    indices: xr.DataArray = None,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    reproj_method: str = "nearest",
    subtract_dem: bool = True,
    return_zs: bool = False,
    interpolator=None,
) -> xr.DataArray:
    """Downscale a SFINCS ``zsmax`` field onto a DEM grid (in-memory / per-block).

    This is the single interpolation core shared by the in-memory and
    file-based paths.  It interpolates the water level onto the DEM grid
    (``reproj_method``), optionally subtracts the DEM to get flood depth
    (``subtract_dem``), and masks the result.

    Parameters
    ----------
    zsmax : xr.DataArray or xu.UgridDataArray
        Maximum water level (m); regular or quadtree.
    dep : xr.DataArray
        High-resolution DEM (m) of the model region (a single block).
    indices : xr.DataArray, optional
        Cell-index raster for exact containment (``reproj_method="nearest"``)
        and/or to mask pixels outside the domain.
    hmin : float, optional
        Minimum flood depth (m), by default 0.05.  Only applied when
        ``subtract_dem`` is ``True``.
    gdf_mask : gpd.GeoDataFrame, optional
        Polygons to mask the output (area outside set to nodata).
    reproj_method : {"nearest", "bilinear"}, optional
        WSE interpolation onto the DEM grid.
    subtract_dem : bool, optional
        Subtract the DEM to return flood depth (``True``, default) or return
        the raw interpolated water level (``False``).
    return_zs : bool, optional
        Also return the interpolated water level, by default False.
    interpolator : callable, optional
        Pre-built scattered interpolant over the quadtree cell centres (reused
        by the file-based streamer across blocks).

    Returns
    -------
    xr.DataArray or tuple[xr.DataArray, xr.DataArray]
        The downscaled product (flood depth or water level); when ``return_zs``
        is ``True`` a ``(product, water_level)`` tuple is returned instead.
    """
    idx_outside = None

    # Fast path: exact containing-cell lookup via the index COG.  Only valid
    # for nearest sampling — bilinear needs neighbouring cell centres, so it
    # falls through to the interpolation path and uses the index only to mask.
    if indices is not None and reproj_method != "bilinear":
        # Squeeze a possible band dim (rioxarray hands back (1, ny, nx)).
        idx_arr = np.squeeze(np.asarray(indices.values))
        if idx_arr.shape != dep.shape:
            raise ValueError(
                "Indices shape {} does not match dep shape {}.".format(
                    idx_arr.shape, dep.shape
                )
            )
        nan_val_indices = indices.raster.nodata
        if nan_val_indices is None:
            nan_val_indices = 2147483647
        no_data_mask = idx_arr == nan_val_indices

        idx = idx_arr.copy()
        idx[no_data_mask] = 0  # placeholder; masked back to NaN below

        # Normalise to canonical (y, x) south-up order BEFORE mask_nodata (so a
        # quadtree field keeps its Ugrid type), then flatten in Fortran order to
        # match the SFINCS index convention (get_indices_at_points returns
        # iind*nmax + jind; quadtree zsmax is already 1-D so both are a no-op).
        zs_cell = _canonical_cellfield(zsmax).raster.mask_nodata()
        zs_numpy = np.asarray(zs_cell.values).flatten(order="F")
        zs_vals = zs_numpy[idx]
        zs_vals[no_data_mask] = np.nan

        zs_on_dep = xr.DataArray(
            zs_vals, dims=["y", "x"], coords={"y": dep.y, "x": dep.x}
        )
        zs_on_dep.raster.set_nodata(np.nan)
        zs_on_dep.raster.set_crs(dep.raster.crs)
    else:
        zs_on_dep = _interp_zs_on_dep(zsmax, dep, reproj_method, interpolator)
        # An index COG (if supplied) masks pixels outside any cell, keeping a
        # bilinear interpolant from bleeding past the domain edge.
        if indices is not None:
            idx_vals = np.squeeze(np.asarray(indices.values))
            if idx_vals.shape != dep.shape:
                raise ValueError(
                    f"Indices shape {idx_vals.shape} does not match "
                    f"dep shape {dep.shape}."
                )
            idx_nodata = indices.raster.nodata
            if idx_nodata is None:
                idx_nodata = 2147483647
            idx_outside = idx_vals == idx_nodata
            # Also drop pixels whose parent cell is dry (zsmax NaN) so a
            # scattered interpolant cannot bleed across dry cells in-domain.
            inside = ~idx_outside
            if inside.any():
                zs_flat = np.asarray(
                    _canonical_cellfield(zsmax).raster.mask_nodata().values
                ).flatten(order="F")
                dry = np.zeros(idx_vals.shape, dtype=bool)
                dry[inside] = np.isnan(zs_flat[idx_vals[inside].astype(int)])
                idx_outside = idx_outside | dry

    # Subtract the DEM for flood depth, or keep the raw water level.
    if subtract_dem:
        out = (zs_on_dep - dep).astype("float32")
    else:
        out = zs_on_dep.astype("float32")
    out.raster.set_nodata(np.nan)

    if idx_outside is not None:
        out = out.where(~idx_outside)
    if subtract_dem:
        out = out.where(out > hmin)  # never want new wet cells below hmin
    if gdf_mask is not None:
        mask = out.raster.geometry_mask(gdf_mask, all_touched=True)
        out = out.where(mask)

    if return_zs:
        # water level masked to the same wet set as the product
        return out, zs_on_dep.where(~np.isnan(out))
    return out


# =============================================================================
#  STEP 3 — Post-process the downscaled raster (optional)
# =============================================================================


# ---- 3a. remove_disconnected_flooding : drop disconnected wet pools ---------
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
        f"Disconnected-flooding removal: {n_wet} wet pixels " f"(hmin={hmin} m)"
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
    _neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

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
                        masked_block = np.where(conn_blk, var_block, np.nan).astype(
                            np.float32
                        )
                        dst.write(masked_block, 1, window=window)
            logger.info(f"  Masked raster written: {output_fn}")

    return None


# ---- 3b. compute_flow_connected_mask : cell-level connectivity helper -------
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
