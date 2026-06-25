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
    method: str = "constant",
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

    Supports multiple downscaling methods via the *method* parameter:

    * ``"raw"`` -- Paint each DEM pixel with the water level of the SFINCS
      cell that *contains* it (the value straight from the SFINCS computation,
      via exact containment in the index COG).  On a regular grid containment
      is the nearest cell; on a quadtree it is not.  No DEM subtraction, no
      wet/dry masking.  Produces a water-level raster.
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
        Downscaling method (``"raw"`` / ``"constant"`` / ``"bilinear"``), by
        default ``"constant"``.  The interpolation follows from *method*: on a
        regular grid ``"constant"`` uses nearest resampling (bathtub) and
        ``"bilinear"`` uses bilinear resampling (reproject engine); on a
        quadtree ``"bilinear"`` uses a scattered interpolator.  WSE adjustments
        (dilation, velocity head) are applied beforehand by calling
        :func:`adjust_zsmax_dilation` / :func:`adjust_zsmax_energyhead` on
        ``zsmax`` directly.
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
    xr.DataArray
        The downscaled product: flood depth (``hmax``) for ``"constant"`` /
        ``"bilinear"``, or water level for ``"raw"``.  Dry pixels are NaN (a
        domain with no flooding yields an all-NaN array).  File-based methods
        also write it to *floodmap_fn* / *zsmap_fn* and return a lazy
        (dask-backed) view re-opened from that file; the in-memory
        ``"constant"`` path returns the array directly.
    """
    _VALID_METHODS = {"raw", "constant", "bilinear"}
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}.  Choose from {sorted(_VALID_METHODS)}."
        )

    # On a regular grid, bilinear reuses the reproject engine
    # (_downscale_constant / _downscale_floodmap_da derive the resampling from
    # `method`); the bespoke scattered _downscale_bilinear is reserved for
    # quadtree (unstructured) grids.
    is_quadtree = isinstance(zsmax, xu.UgridDataArray)

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
        if method == "raw" or (method == "bilinear" and is_quadtree):
            raise ValueError(
                "In-memory (xr.DataArray) dep supports method='constant' and "
                "regular-grid method='bilinear'; use a file path for "
                "method='raw' and quadtree method='bilinear'."
            )
        if isinstance(floodmap_fn, Path):
            floodmap_fn = str(floodmap_fn)
        if indices is not None:
            if isinstance(indices, (str, Path)) and not isinstance(dep, (str, Path)):
                raise ValueError(
                    "index should be xr.DataArray when dep is xr.DataArray."
                )
            elif isinstance(indices, xr.DataArray) and not isinstance(
                dep, xr.DataArray
            ):
                raise ValueError("index should be str/Path when dep is str/Path.")
        hmax = _downscale_floodmap_da(
            zsmax=zsmax,
            dep=dep,
            indices=indices,
            hmin=hmin,
            gdf_mask=gdf_mask,
            method=method,
        )
        if floodmap_fn is not None:
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

    # Dispatch to the appropriate engine.  Regular-grid bilinear reuses the
    # reproject engine (_downscale_constant, which derives the resampling from
    # `method`); quadtree bilinear uses the scattered interpolator.
    if method == "raw":
        _downscale_raw(
            zsmax=zsmax,
            dep=dep,
            zsmap_fn=zsmap_fn,
            gdf_mask=gdf_mask,
            nrmax=nrmax,
            logger=logger,
            indices=indices,
        )
    elif method == "bilinear" and is_quadtree:
        _downscale_bilinear(
            zsmax=zsmax,
            dep=dep,
            hmin=hmin,
            gdf_mask=gdf_mask,
            floodmap_fn=floodmap_fn,
            zsmap_fn=zsmap_fn,
            nrmax=nrmax,
            logger=logger,
            indices=indices,
        )
    else:  # "constant" (regular or quadtree) or regular-grid "bilinear"
        _downscale_constant(
            zsmax=zsmax,
            dep=dep,
            indices=indices,
            hmin=hmin,
            gdf_mask=gdf_mask,
            floodmap_fn=floodmap_fn,
            zsmap_fn=zsmap_fn,
            method=method,
            zoom_level=zoom_level,
            nrmax=nrmax,
            logger=logger,
        )

    # Re-open the written product so every method returns a DataArray
    # (write-then-read keeps the streaming engines memory-bounded).  ``raw`` has
    # no depth, so it returns the water level (zsmap) instead of hmax.
    out_fn = zsmap_fn if method == "raw" else floodmap_fn
    if out_fn is None or not Path(str(out_fn)).exists():
        return None
    out_name = "zsmax" if method == "raw" else "hmax"
    long_name = "Maximum water level" if method == "raw" else "Maximum flood depth"
    da = _open_result_da(out_fn, out_name)
    da.attrs.update({"long_name": long_name, "units": "m"})
    return da


# ---- 2b. helpers : block streaming + output-raster plumbing -----------------
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


def _make_output_profile(geo):
    """Standard COG profile for float32 output rasters."""
    return dict(
        driver="GTiff",
        width=geo["width"],
        height=geo["height"],
        count=1,
        dtype=np.float32,
        crs=geo["crs"],
        transform=geo["transform"],
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=2,
        profile="COG",
        nodata=np.nan,
        BIGTIFF="YES",
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
    floodmap_fn,
    zsmap_fn,
    gdf_mask,
    geo,
    logger,
):
    """Apply polygon mask and build overviews on output raster(s)."""
    if gdf_mask is not None:
        logger.info("Applying polygon mask...")
        from rasterio.features import geometry_mask

        # CRS-aware masking: reproject the mask polygons to the output CRS
        # (matches the in-memory path's raster.geometry_mask, which is CRS-aware).
        if gdf_mask.crs is not None and geo["crs"] is not None:
            gdf_mask = gdf_mask.to_crs(geo["crs"].to_wkt())

        mask = geometry_mask(
            gdf_mask.geometry,
            out_shape=(geo["height"], geo["width"]),
            transform=geo["transform"],
            invert=True,
            all_touched=True,
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


# ---- 2c. engines : per-method file-based implementations --------------------
#  raw — paint the containing SFINCS cell's WSE onto the DEM (no DEM subtraction)
def _downscale_raw(zsmax, dep, zsmap_fn, gdf_mask, nrmax, logger, indices=None):
    vals = zsmax.values
    wet = ~np.isnan(vals)

    geo = _open_dem_geometry(dep)
    profile = _make_output_profile(geo)
    _create_output_rasters(profile, zsmap_fn=zsmap_fn)

    if np.sum(wet) < 1:
        logger.warning("No wet cells found; writing an empty water-level map.")
        return

    windows = list(_stream_blocks(geo["width"], geo["height"], nrmax))
    total = len(windows)
    done = 0

    if indices is not None:
        # ----- Index-COG path: exact cell containment, no interpolation -----
        # Normalise to canonical (y, x) south-up order, then flatten in Fortran
        # order to match the SFINCS index convention (get_indices_at_points
        # returns iind*nmax + jind; quadtree zsmax is already 1-D so both are a
        # no-op there).
        vals_flat = np.asarray(_canonical_cellfield(zsmax).values).ravel(order="F")
        logger.info(f"Raw (index-COG): {int(np.sum(wet))} wet cells")
        indices_src = rasterio.open(str(indices))
        nodata_idx = indices_src.nodata
        nodata_idx = int(nodata_idx) if nodata_idx is not None else 2147483647

        for window, bm0, bm1, bn0, bn1 in windows:
            idx_block = indices_src.read(1, window=window)
            zs_block = np.full(idx_block.shape, np.nan, dtype=np.float32)
            valid = idx_block != nodata_idx
            zs_block[valid] = vals_flat[idx_block[valid]]

            with rasterio.open(str(zsmap_fn), "r+") as dst:
                dst.write(zs_block, window=window, indexes=1)

            done += 1
            if done % 25 == 0 or done == total:
                logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

        indices_src.close()
    elif isinstance(zsmax, xu.UgridDataArray):
        # ----- Quadtree fallback: NearestNDInterpolator over face centres ---
        from scipy.interpolate import NearestNDInterpolator

        grid = zsmax.ugrid.grid
        face_x, face_y = grid.face_coordinates.T
        interpolator = NearestNDInterpolator(
            np.column_stack([face_x[wet], face_y[wet]]),
            vals[wet],
        )
        logger.warning(
            "Raw quadtree without an index COG: approximating cell containment "
            f"with the nearest face centre ({int(np.sum(wet))} wet cells). "
            "Pass an index COG (make_index_cog) for exact containment."
        )

        for window, bm0, bm1, bn0, bn1 in windows:
            xx, yy = _block_pixel_centres(geo["transform"], bm0, bm1, bn0, bn1)
            zs_block = (
                interpolator(np.column_stack([xx.ravel(), yy.ravel()]))
                .reshape(xx.shape)
                .astype(np.float32)
            )

            with rasterio.open(str(zsmap_fn), "r+") as dst:
                dst.write(zs_block, window=window, indexes=1)

            done += 1
            if done % 25 == 0 or done == total:
                logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")
    else:
        # ----- Regular fallback: nearest reproject of zsmax onto the DEM ----
        logger.info(f"Raw regular (nearest reproject): {int(np.sum(wet))} wet cells")

        rotated = not (geo["transform"][1] == 0 and geo["transform"][3] == 0)
        for window, bm0, bm1, bn0, bn1 in windows:
            if rotated:
                xx, yy = _block_pixel_centres(geo["transform"], bm0, bm1, bn0, bn1)
                block_dep = xr.DataArray(
                    np.zeros((bn1 - bn0, bm1 - bm0), dtype=np.float32),
                    dims=("y", "x"),
                    coords={"yc": (("y", "x"), yy), "xc": (("y", "x"), xx)},
                )
            else:
                x_coords = geo["transform"][2] + (np.arange(bm0, bm1) + 0.5) * geo["dx"]
                y_coords = geo["transform"][5] + (np.arange(bn0, bn1) + 0.5) * geo["dy"]
                block_dep = xr.DataArray(
                    np.zeros((bn1 - bn0, bm1 - bm0), dtype=np.float32),
                    dims=("y", "x"),
                    coords={"y": ("y", y_coords), "x": ("x", x_coords)},
                )
            block_dep.raster.set_crs(geo["crs"].to_wkt())
            zs_block = (
                zsmax.raster.reproject_like(block_dep, method="nearest")
                .raster.mask_nodata()
                .values.astype(np.float32)
            )

            with rasterio.open(str(zsmap_fn), "r+") as dst:
                dst.write(zs_block, window=window, indexes=1)

            done += 1
            if done % 25 == 0 or done == total:
                logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

    _apply_mask_and_overviews(None, zsmap_fn, gdf_mask, geo, logger)
    logger.info(f"Raw quadtree water level map saved to: {zsmap_fn}")


#  constant — index-COG lookup / reproject per block (bathtub)
def _downscale_constant(
    zsmax,
    dep,
    indices,
    hmin,
    gdf_mask,
    floodmap_fn,
    zsmap_fn,
    method,
    zoom_level,
    nrmax,
    logger,
):
    """File-based constant/bilinear downscaling via _downscale_floodmap_da.

    ``method`` is ``"constant"`` (nearest) or ``"bilinear"`` (regular-grid
    bilinear resampling); it is forwarded per block to _downscale_floodmap_da.
    """
    if isinstance(floodmap_fn, Path):
        floodmap_fn = str(floodmap_fn)

    # indices validation
    if indices is not None:
        if not isinstance(indices, (str, Path)):
            raise ValueError("indices should be str/Path when dep is str/Path.")

    if zoom_level is not None:
        zls_dict, crs = RasterioDriver._get_zoom_levels_and_crs(dep)
        overview_level = zoom_to_overview_level(
            zoom=zoom_level,
            zls_dict=zls_dict,
            source_crs=crs,
        )
        if overview_level:
            overview_level -= 1
        else:
            overview_level = None
    else:
        overview_level = None

    _open_kwargs = (
        {"overview_level": overview_level} if overview_level is not None else {}
    )
    with rasterio.open(dep, **_open_kwargs) as src:
        if indices is not None:
            indices_src = rasterio.open(indices, **_open_kwargs)

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
        with rasterio.open(floodmap_fn, "w", **profile):
            pass
        if zsmap_fn is not None:
            with rasterio.open(zsmap_fn, "w", **profile):
                pass

        windows = list(
            _stream_blocks(src.width, src.height, nrmax, merge_singletons=True)
        )
        total = len(windows)
        done = 0
        skipped = 0
        logger.info(
            f"Constant WSE: {total} blocks to process "
            f"({m1}x{n1} pixels, block size {nrmax})"
        )

        for window, bm0, bm1, bn0, bn1 in windows:
            # Read indices first — skip block early if no SFINCS cells
            if indices is not None:
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
            if np.all(np.isnan(block_data)):
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
                if indices is not None:
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
                if indices is not None:
                    block_idx = xr.DataArray(
                        block_idx.squeeze(),
                        dims=("y", "x"),
                        coords={"yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
                    )

            block_dep.raster.set_crs(src.crs.to_wkt())
            if indices is not None:
                block_idx.raster.set_nodata(int(indices_src.nodata))
                block_idx.raster.set_crs(indices_src.crs.to_wkt())

            block_hmax = _downscale_floodmap_da(
                zsmax=zsmax,
                dep=block_dep,
                indices=block_idx if indices is not None else None,
                hmin=hmin,
                gdf_mask=gdf_mask,
                method=method,
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

        if indices is not None:
            indices_src.close()

    build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)
    if zsmap_fn is not None:
        build_overviews(fn=zsmap_fn, resample_method="nearest", logger=logger)


#  bilinear — scattered LinearNDInterpolator over cell centres (quadtree)
def _downscale_bilinear(
    zsmax,
    dep,
    hmin,
    gdf_mask,
    floodmap_fn,
    zsmap_fn,
    nrmax,
    logger,
    indices=None,
):
    from scipy.interpolate import LinearNDInterpolator

    grid = zsmax.ugrid.grid
    face_x, face_y = grid.face_coordinates.T
    vals = zsmax.values

    geo = _open_dem_geometry(dep)
    profile = _make_output_profile(geo)
    _create_output_rasters(profile, floodmap_fn, zsmap_fn)

    if np.sum(~np.isnan(vals)) < 3:
        logger.warning(
            "Fewer than 3 wet cells; cannot interpolate. Writing an empty floodmap."
        )
        return

    H_eff = vals.copy()

    wet_ext = ~np.isnan(H_eff)
    interpolator = LinearNDInterpolator(
        np.column_stack([face_x[wet_ext], face_y[wet_ext]]),
        H_eff[wet_ext],
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

    windows = list(_stream_blocks(geo["width"], geo["height"], nrmax))
    total = len(windows)
    done = 0

    for window, bm0, bm1, bn0, bn1 in windows:
        # Skip blocks with no SFINCS cells (fast path when an index COG is given)
        idx_block = None
        if indices_src is not None:
            idx_block = indices_src.read(1, window=window)
            if np.all(idx_block == idx_nodata):
                done += 1
                continue

        with rasterio.open(str(dep)) as src:
            dem_block = src.read(1, window=window).astype(np.float64)
            # Mask the DEM's own nodata sentinel (e.g. -9999) to NaN.
            if src.nodata is not None and not np.isnan(src.nodata):
                dem_block[dem_block == src.nodata] = np.nan

        xx, yy = _block_pixel_centres(geo["transform"], bm0, bm1, bn0, bn1)
        zs_interp = (
            interpolator(np.column_stack([xx.ravel(), yy.ravel()]))
            .reshape(dem_block.shape)
            .astype(np.float32)
        )

        # Mask pixels outside any wet SFINCS cell
        if idx_block is not None:
            outside = idx_block == idx_nodata
            # Also mask pixels whose parent cell is dry (NaN zsmax)
            inside = ~outside
            if inside.any():
                pidx = idx_block[inside].astype(int)
                parent_zs = vals[pidx]
                parent_H = H_eff[pidx]
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


#  core — per-block zs->hmax engine (reproject or index-COG lookup)
def _downscale_floodmap_da(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: xr.DataArray,
    indices: xr.DataArray = None,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    method: str = "constant",
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

    # Fast path: exact containing-cell lookup via the index COG.  Only valid
    # for nearest sampling (constant) — bilinear needs neighbouring cell
    # centres, so it falls through to the reproject path below and uses the
    # index (if any) purely to mask.
    if indices is not None and method != "bilinear":
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
        zsmax = _canonical_cellfield(zsmax)
        zsmax = zsmax.raster.mask_nodata()  # make sure nodata is nan
        zs_numpy = np.asarray(zsmax.values).flatten(order="F")
        h = zs_numpy[idx] - dep.values[:]
        h[no_data_mask] = np.nan

        hmax = xr.DataArray(h, dims=["y", "x"], coords={"y": dep.y, "x": dep.x})
        hmax.raster.set_nodata(np.nan)
        hmax.raster.set_crs(dep.raster.crs)
    else:
        # Interpolate zsmax onto the dep grid, honouring `method` (bilinear vs
        # nearest).  Regular grids reproject; quadtree grids rasterize/regrid.
        resampling = "bilinear" if method == "bilinear" else "nearest"
        if isinstance(zsmax, xr.DataArray):
            zs_on_dep = zsmax.raster.reproject_like(dep, method=resampling)
        elif isinstance(zsmax, xu.UgridDataArray):
            # if non-rotated grid, use xugrid rasterize_like
            if dep.raster.transform[1] == 0 and dep.raster.transform[3] == 0:
                zs_on_dep = zsmax.ugrid.rasterize_like(dep)
            # if rotated grid, use xugrid regridder
            else:
                # need to convert dep to unstructured to enable xugrid regridder
                uda_dep = xu.UgridDataArray.from_structured2d(dep, "xc", "yc")
                regridder = xu.CentroidLocatorRegridder(source=zsmax, target=uda_dep)
                result = regridder.regrid(zsmax)
                # map back to structured
                zs_on_dep = dep.copy(data=result.values.reshape(dep.shape))

        zs_on_dep = zs_on_dep.raster.mask_nodata()  # make sure nodata is nan
        hmax = (zs_on_dep - dep).astype("float32")
        hmax.raster.set_nodata(np.nan)

        # If an index COG is supplied, use it to mask pixels outside any cell
        # (keeps a bilinear interpolant from bleeding past the domain edge).
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
            hmax = hmax.where(idx_vals != idx_nodata)

    # mask floodmap
    hmax = hmax.where(hmax > hmin)

    if gdf_mask is not None:
        mask = hmax.raster.geometry_mask(gdf_mask, all_touched=True)
        hmax = hmax.where(mask)

    return hmax


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
