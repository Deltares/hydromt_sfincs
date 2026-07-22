"""Cloud-Optimized GeoTIFF (COG) outputs for SFINCS quadtree grids."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

__all__ = ["make_quadtree_index_cog", "make_topobathy_cog"]

logger = logging.getLogger(__name__)


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
) -> None:
    """Write a COG raster mapping each pixel to a quadtree cell index.

    The output raster matches the resolution and grid of
    ``filename_topobathy`` (typically produced by
    :py:func:`make_topobathy_cog`). Pixels that do not fall inside any
    active cell are filled with the ``uint32`` sentinel ``2147483647``.

    Parameters
    ----------
    quadtree_grid : SfincsQuadtreeGrid
        Grid component providing ``get_indices_at_points`` and
        ``model.crs``.
    filename : str or Path
        Output COG file path.
    filename_topobathy : str or Path
        Reference topobathy COG whose grid / CRS define the output.
    """
    with rasterio.open(filename_topobathy) as src:
        bounds = src.bounds
        dx = src.res[0]
        transform = src.transform
        width = src.width
        height = src.height
        quadtree_grid.model.crs = src.crs

    x0, y0, x1, y1 = bounds.left, bounds.bottom, bounds.right, bounds.top

    xx = np.arange(x0, x1, dx) + 0.5 * dx
    yy = np.arange(y1, y0, -dx) - 0.5 * dx
    xx, yy = np.meshgrid(xx, yy)

    nodata = 2147483647
    indices = quadtree_grid.get_indices_at_points(xx, yy)
    indices[indices == -999] = nodata

    ii = np.empty((height, width), dtype=np.uint32)
    ii[:, :] = indices

    with rasterio.open(
        filename,
        "w",
        driver="COG",
        height=height,
        width=width,
        count=1,
        dtype=ii.dtype,
        crs=quadtree_grid.model.crs,
        transform=transform,
        nodata=nodata,
        overview_resampling=Resampling.nearest,
    ) as dst:
        dst.write(ii, 1)
