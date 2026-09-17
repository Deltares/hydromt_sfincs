"""Cloud-Optimized GeoTIFF (COG) outputs for SFINCS quadtree grids."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform as _warp_transform

__all__ = ["make_quadtree_index_cog", "make_topobathy_cog"]

logger = logging.getLogger(__name__)



def transform_coords(src_crs, dst_crs, xx: np.ndarray, yy: np.ndarray):
    """Transform 2-D coordinate arrays from ``src_crs`` to ``dst_crs``."""
    shape = np.shape(xx)
    xt, yt = _warp_transform(src_crs, dst_crs, np.ravel(xx), np.ravel(yy))
    return np.reshape(xt, shape), np.reshape(yt, shape)

def make_topobathy_cog(
    quadtree_grid,
    filename: Union[str, Path],
    bathymetry_sets: List[dict],
    bathymetry_database: Optional[object] = None,
    dx: float = 10.0,
) -> None:
    """Write a COG raster sampling the model topobathy.

    The COG is written in the model CRS, so this currently only supports
    projected coordinate systems.

    Parameters
    ----------
    quadtree_grid : SfincsQuadtreeGrid
        Grid component providing ``bounds`` and ``model.crs``.
    filename : str or Path
        Output COG file path.
    bathymetry_sets : list of dict
        Dataset list passed through to
        ``bathymetry_database.get_bathymetry_on_points``.
    bathymetry_database : object, optional
        Backing bathymetry database providing
        ``get_bathymetry_on_points``. Required for this function to
        produce data.
    dx : float, optional
        Raster resolution in model CRS units, by default ``10.0``.
    """
    bounds = quadtree_grid.bounds

    x0, y0, x1, y1 = bounds[0], bounds[1], bounds[2], bounds[3]

    # Round out to nearest dx
    x0 = x0 - (x0 % dx)
    x1 = x1 + (dx - x1 % dx)
    y0 = y0 - (y0 % dx)
    y1 = y1 + (dx - y1 % dx)

    xx = np.arange(x0, x1, dx) + 0.5 * dx
    yy = np.arange(y1, y0, -dx) - 0.5 * dx
    xx, yy = np.meshgrid(xx, yy)

    zz = bathymetry_database.get_bathymetry_on_points(
        xx, yy, dx, quadtree_grid.model.crs, bathymetry_sets
    )

    with rasterio.open(
        filename,
        "w",
        driver="COG",
        height=zz.shape[0],
        width=zz.shape[1],
        count=1,
        dtype=zz.dtype,
        crs=quadtree_grid.model.crs,
        transform=from_origin(x0, y1, dx, dx),
        nodata=-999.0,
    ) as dst:
        dst.write(zz, 1)


def make_quadtree_index_cog(
    quadtree_grid,
    filename: Union[str, Path],
    filename_topobathy: Union[str, Path],
    structures=None,
) -> int:
    """Write a COG raster mapping each pixel to a quadtree cell index.

    The output raster matches the resolution and grid of
    ``filename_topobathy`` (typically produced by
    :py:func:`make_topobathy_cog`). Pixels that do not fall inside any
    active cell are filled with the ``uint32`` sentinel ``2147483647``.

    With ``structures`` (thin dams, weirs, flood walls) the index is made
    structure-aware in the same step: a pixel whose straight link to its own
    cell centre crosses a structure is handed to the nearest cell centre it
    can reach without crossing one, which is the rule SFINCS itself uses to
    block faces, so the flood map follows the real structure instead of the
    grid-snapped one. The file then has four bands: the index, and per pixel
    the ids of surrounding cells that lie across a structure, which the
    flood map's blend and trend surface modes read so they never
    interpolate across a structure. See
    :py:func:`hydromt_sfincs.workflows.flood_map.apply_structures_to_index_cog`.

    Parameters
    ----------
    quadtree_grid : SfincsQuadtreeGrid
        Grid component providing ``get_indices_at_points`` and
        ``model.crs``.
    filename : str or Path
        Output COG file path.
    filename_topobathy : str or Path
        Reference topobathy COG whose grid / CRS define the output.
    structures : iterable of shapely LineString / MultiLineString, optional
        Structures in the model CRS. ``None`` or empty writes a plain
        single-band index.

    Returns
    -------
    int
        Number of pixels reassigned across structures (0 without structures).
    """
    with rasterio.open(filename_topobathy) as src:
        bounds = src.bounds
        dx = src.res[0]
        transform = src.transform
        width = src.width
        height = src.height
        src_crs = src.crs

    x0, y0, x1, y1 = bounds.left, bounds.bottom, bounds.right, bounds.top

    xx = np.arange(x0, x1, dx) + 0.5 * dx
    yy = np.arange(y1, y0, -dx) - 0.5 * dx
    xx, yy = np.meshgrid(xx, yy)

    # The index lookup expects coordinates in the model CRS. The model CRS
    # is read-only, so transform the pixel centres instead of overwriting it.
    model_crs = quadtree_grid.model.crs
    if model_crs is not None and src_crs is not None and src_crs != model_crs:
        xx, yy = transform_coords(src_crs, model_crs, xx, yy)

    nodata = 2147483647
    indices = quadtree_grid.get_indices_at_points(xx, yy)
    indices[indices == -999] = nodata

    ii = np.empty((height, width), dtype=np.uint32)
    ii[:, :] = indices

    lines = [g for g in (structures or []) if g is not None and not g.is_empty]
    n_reassigned = 0
    blocked = None
    if lines:
        from hydromt_sfincs.workflows.flood_map import (
            STRUCTURES_TAG,
            blocked_cells_for_pixels,
            reassign_index_for_structures,
            structures_to_tag,
        )

        # Work in the raster CRS: cell centres and structures come in the
        # model CRS and are transformed when the topobathy uses another one.
        xy = quadtree_grid.data.grid.face_coordinates
        xc, yc = xy[:, 0].astype(float), xy[:, 1].astype(float)
        if model_crs is not None and src_crs is not None and src_crs != model_crs:
            from pyproj import Transformer
            from shapely.ops import transform as shp_transform

            tr = Transformer.from_crs(model_crs, src_crs, always_xy=True)
            xc, yc = tr.transform(xc, yc)
            lines = [shp_transform(tr.transform, g) for g in lines]
        ii, changed = reassign_index_for_structures(ii, transform, lines, xc, yc, nodata)
        n_reassigned = int(changed.sum())
        blocked = blocked_cells_for_pixels(ii, transform, lines, xc, yc, nodata)

    with rasterio.open(
        filename,
        "w",
        driver="COG",
        height=height,
        width=width,
        count=1 if blocked is None else 4,
        dtype=ii.dtype,
        crs=src_crs,
        transform=transform,
        nodata=nodata,
        overview_resampling=Resampling.nearest,
    ) as dst:
        dst.write(ii, 1)
        dst.set_band_description(1, "cell index")
        if blocked is not None:
            for j in range(3):
                dst.write(blocked[j], j + 2)
                dst.set_band_description(j + 2, f"blocked cell {j + 1}")
            dst.update_tags(**{STRUCTURES_TAG: structures_to_tag(lines)})
    return n_reassigned
