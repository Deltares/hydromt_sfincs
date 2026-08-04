"""Datashader-based map-overlay rendering for SFINCS quadtree grids."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

__all__ = [
    "ElevationOverlay",
    "MaskOverlay",
    "MeshOverlay",
    "make_edge_dataframe",
    "make_elevation_overlay",
    "make_elevation_trimesh",
    "make_map_overlay",
    "make_mask_dataframe",
    "make_mask_overlay",
]

logger = logging.getLogger(__name__)

# optional dependency
try:
    import datashader.transfer_functions as tf
    from datashader import Canvas
    from datashader.utils import export_image

    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False


def make_edge_dataframe(ugrid, source_crs: CRS) -> pd.DataFrame:
    """Build a datashader-ready edge-coordinate DataFrame in webmercator.

    Produces one row per mesh edge with its two endpoints reprojected
    from ``source_crs`` to EPSG:3857. If the source CRS is geographic
    and any longitude exceeds 180, negative webmercator x values are
    shifted by one world width so the line doesn't wrap back across the
    dateline.

    Parameters
    ----------
    ugrid : xugrid.Ugrid2d
        Source mesh exposing ``edge_node_coordinates``.
    source_crs : pyproj.CRS
        CRS of the mesh coordinates.

    Returns
    -------
    pd.DataFrame
        Columns ``x1, y1, x2, y2`` in EPSG:3857.
    """
    x1 = ugrid.edge_node_coordinates[:, 0, 0]
    x2 = ugrid.edge_node_coordinates[:, 1, 0]
    y1 = ugrid.edge_node_coordinates[:, 0, 1]
    y2 = ugrid.edge_node_coordinates[:, 1, 1]

    cross_dateline = False
    if source_crs.is_geographic and (np.max(x1) > 180.0 or np.max(x2) > 180.0):
        cross_dateline = True

    transformer = Transformer.from_crs(source_crs, 3857, always_xy=True)
    x1, y1 = transformer.transform(x1, y1)
    x2, y2 = transformer.transform(x2, y2)
    if cross_dateline:
        x1[x1 < 0] += 40075016.68557849
        x2[x2 < 0] += 40075016.68557849

    return pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))


def make_map_overlay(
    dataframe: pd.DataFrame,
    file_name: Union[str, Path],
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    color: str = "black",
    width: int = 800,
) -> bool:
    """Render a PNG map overlay of mesh edges using datashader.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Edge DataFrame as produced by :py:func:`make_edge_dataframe`
        (columns ``x1, y1, x2, y2`` in EPSG:3857).
    file_name : str or Path
        Output image path. The extension is stripped; datashader appends
        ``.png``.
    xlim : list of float, optional
        Longitude limits ``[xmin, xmax]`` in EPSG:4326.
    ylim : list of float, optional
        Latitude limits ``[ymin, ymax]`` in EPSG:4326.
    color : str, optional
        Line colour, by default ``"black"``.
    width : int, optional
        Output image width in pixels; height is derived from the aspect
        ratio of ``xlim`` / ``ylim``. Defaults to ``800``.

    Returns
    -------
    bool
        ``True`` if the overlay was written, ``False`` if datashader is
        unavailable, the dataframe is empty, or rendering raised an
        exception.
    """
    if not HAS_DATASHADER:
        logger.warning("Datashader is not available. Please install datashader.")
        return False

    if dataframe is None or dataframe.empty:
        return False

    try:
        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        xl0, yl0 = transformer.transform(xlim[0], ylim[0])
        xl1, yl1 = transformer.transform(xlim[1], ylim[1])
        if xl0 > xl1:
            xl1 += 40075016.68557849
        xlim = [xl0, xl1]
        ylim = [yl0, yl1]
        ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        height = int(width * ratio)

        cvs = Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)
        agg = cvs.line(dataframe, x=["x1", "x2"], y=["y1", "y2"], axis=1)
        img = tf.shade(agg, cmap=color)

        path_dir = os.path.dirname(file_name) or os.getcwd()
        name = os.path.splitext(os.path.basename(file_name))[0]
        export_image(img, name, export_path=path_dir)
        return True
    except Exception:
        return False


def _dilate_bool_agg(agg, px: int):
    """Spread True pixels of a boolean aggregation by ``px`` pixels.

    Drop-in replacement for ``tf.spread(agg, px=px)`` for boolean (``ds.any()``)
    aggregations, using a scipy binary dilation with datashader's circular
    footprint instead of datashader's numba-jitted spread kernel. The jitted
    kernel is compiled per marker size, which takes minutes inside a frozen
    (Nuitka) executable; the dilation is numba-free and operates on the small
    aggregated image, so its cost is independent of the number of points.
    """
    if not px or px <= 0:
        return agg
    import xarray as xr
    from scipy.ndimage import binary_dilation

    # Same circular footprint as datashader's tf.spread(shape="circle"),
    # which uses sqrt(x^2 + y^2) <= px + 0.5 (see datashader._circle_mask).
    yy, xx = np.ogrid[-px : px + 1, -px : px + 1]
    footprint = (xx * xx + yy * yy) <= (px + 0.5) ** 2

    values = np.nan_to_num(agg.values).astype(bool)
    dilated = binary_dilation(values, structure=footprint)
    return xr.DataArray(
        dilated, coords=agg.coords, dims=agg.dims, name=agg.name, attrs=agg.attrs
    )


def make_mask_dataframe(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    source_crs: CRS,
) -> pd.DataFrame:
    """Build a datashader-ready point DataFrame for mask rendering.

    Drops cells with ``mask <= 0`` and reprojects remaining cell centres
    from ``source_crs`` to EPSG:3857. Dateline wrapping is applied when
    the source CRS is geographic.

    Parameters
    ----------
    x, y : np.ndarray
        Cell-centre coordinates in the model CRS.
    mask : np.ndarray
        Integer mask values per cell.
    source_crs : pyproj.CRS
        CRS of ``x`` / ``y``.

    Returns
    -------
    pd.DataFrame
        Columns ``x, y, mask`` in EPSG:3857. Empty if no active cells.
    """
    iok = np.where(mask > 0)
    x = x[iok]
    y = y[iok]
    mask = mask[iok]
    if x.size == 0:
        return pd.DataFrame()

    cross_dateline = source_crs.is_geographic and np.max(x) > 180.0
    transformer = Transformer.from_crs(source_crs, 3857, always_xy=True)
    x, y = transformer.transform(x, y)
    if cross_dateline:
        x[x < 0] += 40075016.68557849

    return pd.DataFrame(dict(x=x, y=y, mask=mask))


def make_mask_overlay(
    dataframe: pd.DataFrame,
    file_name: Union[str, Path],
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    colors: Optional[dict] = None,
    px: int = 2,
    width: int = 800,
) -> bool:
    """Render a PNG mask-overlay image using datashader.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Point DataFrame as produced by :py:func:`make_mask_dataframe`
        (columns ``x, y, mask`` in EPSG:3857).
    file_name : str or Path
        Output image path. Datashader appends ``.png``.
    xlim, ylim : list of float, optional
        Lon/lat extent in EPSG:4326.
    colors : dict, optional
        Mapping of mask value → colour, e.g.
        ``{1: "yellow", 2: "red", 3: "green"}``.
    px : int, optional
        Marker radius in pixels, by default ``2``.
    width : int, optional
        Output image width in pixels, by default ``800``.

    Returns
    -------
    bool
        ``True`` if the image was written, ``False`` otherwise.
    """
    if colors is None:
        colors = {1: "yellow", 2: "red", 3: "green", 5: "purple", 6: "blue"}

    if not HAS_DATASHADER:
        logger.warning("Datashader is not available. Please install datashader.")
        return False

    if dataframe is None or dataframe.empty:
        return False

    try:
        import datashader as ds

        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        xl0, yl0 = transformer.transform(xlim[0], ylim[0])
        xl1, yl1 = transformer.transform(xlim[1], ylim[1])
        if xl0 > xl1:
            xl1 += 40075016.68557849
        xlim = [xl0, xl1]
        ylim = [yl0, yl1]
        ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        height = int(width * ratio)

        cvs = Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)

        images = []
        for mask_val, color in colors.items():
            df_sub = dataframe[dataframe["mask"] == mask_val]
            if len(df_sub) > 0:
                agg = cvs.points(df_sub, "x", "y", ds.any())
                # Marker spreading. NOTE: deliberately NOT tf.spread - that
                # numba-jit-compiles a kernel per marker size, which takes
                # minutes in a frozen (Nuitka) build. The aggregation is a
                # small boolean image here (height x width, independent of the
                # number of points), so a scipy binary dilation with the same
                # circular footprint is equivalent and instant.
                images.append(tf.shade(_dilate_bool_agg(agg, px), cmap=color))

        if not images:
            return False

        img = images[0]
        for im in images[1:]:
            img = tf.stack(img, im)

        path_dir = os.path.dirname(file_name) or os.getcwd()
        name = os.path.splitext(os.path.basename(file_name))[0]
        export_image(img, name, export_path=path_dir)
        return True
    except Exception:
        return False


class MaskOverlay:
    """Lazy, cached mask-overlay renderer.

    Holds the datashader DataFrame internally and rebuilds it on demand.
    Components that want to render a mask overlay just hold one instance
    and call :py:meth:`render` / :py:meth:`invalidate` — no need to expose
    a dataframe attribute or a build method.
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame = pd.DataFrame()

    def invalidate(self) -> None:
        """Drop the cached dataframe; it will be rebuilt on next render."""
        self._df = pd.DataFrame()

    def render(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        source_crs: CRS,
        file_name: Union[str, Path],
        xlim: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        colors: Optional[dict] = None,
        px: int = 2,
        width: int = 800,
    ) -> bool:
        """Render the mask overlay, rebuilding the cache if needed.

        Parameters mirror :py:func:`make_mask_dataframe` (for the cache
        build) and :py:func:`make_mask_overlay` (for the render).
        """
        if self._df.empty:
            self._df = make_mask_dataframe(x, y, mask, source_crs)
        return make_mask_overlay(
            self._df,
            file_name,
            xlim=xlim,
            ylim=ylim,
            colors=colors,
            px=px,
            width=width,
        )


class MeshOverlay:
    """Lazy, cached grid-edge overlay renderer.

    Holds the datashader DataFrame internally and rebuilds it on demand.
    Components that want to render a grid-edge overlay just hold one
    instance and call :py:meth:`render` / :py:meth:`invalidate` — no need
    to expose a dataframe attribute or a build method.
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame = pd.DataFrame()

    def invalidate(self) -> None:
        """Drop the cached dataframe; it will be rebuilt on next render."""
        self._df = pd.DataFrame()

    def render(
        self,
        ugrid,
        source_crs: CRS,
        file_name: Union[str, Path],
        xlim: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        color: str = "black",
        width: int = 800,
    ) -> bool:
        """Render the edge overlay, rebuilding the cache if needed.

        Parameters mirror :py:func:`make_edge_dataframe` (for the cache
        build) and :py:func:`make_map_overlay` (for the render).
        """
        if self._df.empty:
            self._df = make_edge_dataframe(ugrid, source_crs)
        return make_map_overlay(
            self._df,
            file_name,
            xlim=xlim,
            ylim=ylim,
            color=color,
            width=width,
        )


def make_elevation_trimesh(
    face_xy: np.ndarray,
    z: np.ndarray,
    level: np.ndarray,
    dx0: float,
    dy0: float,
    rotation: float,
    source_crs: CRS,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build datashader trimesh DataFrames from quadtree cell elevations.

    Each finite-z cell becomes 4 vertices + 2 triangles. Coordinates are
    reprojected from ``source_crs`` to EPSG:3857 (with dateline wrap).

    Returns
    -------
    (vertices, simplices) : two pd.DataFrame
        ``vertices`` has columns ``x, y, z``; ``simplices`` has ``v0, v1, v2``.
        Both empty if no finite z values are present.
    """
    valid = np.isfinite(z)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return pd.DataFrame(), pd.DataFrame()

    cx = face_xy[valid, 0]
    cy = face_xy[valid, 1]
    cz = z[valid]
    clevel = (level - 1)[valid]  # 0-based refinement level

    hdx = (dx0 / 2.0**clevel) / 2.0
    hdy = (dy0 / 2.0**clevel) / 2.0

    dx_offsets = np.array([-1, 1, 1, -1], dtype=np.float64)
    dy_offsets = np.array([-1, -1, 1, 1], dtype=np.float64)

    cosrot = np.cos(np.radians(rotation))
    sinrot = np.sin(np.radians(rotation))

    local_x = hdx[:, None] * dx_offsets[None, :]
    local_y = hdy[:, None] * dy_offsets[None, :]
    vx = cx[:, None] + cosrot * local_x - sinrot * local_y
    vy = cy[:, None] + sinrot * local_x + cosrot * local_y

    vx_flat = vx.ravel()
    vy_flat = vy.ravel()
    vz_flat = np.repeat(cz, 4)

    transformer = Transformer.from_crs(source_crs, 3857, always_xy=True)
    vx_3857, vy_3857 = transformer.transform(vx_flat, vy_flat)
    if source_crs.is_geographic and np.max(face_xy[:, 0]) > 180.0:
        vx_3857[vx_3857 < 0] += 40075016.68557849

    vertices = pd.DataFrame({"x": vx_3857, "y": vy_3857, "z": vz_flat})

    cell_idx = np.arange(n_valid, dtype=np.int32)
    v_base = cell_idx * 4
    t0 = np.column_stack([v_base, v_base + 1, v_base + 2])
    t1 = np.column_stack([v_base, v_base + 2, v_base + 3])
    tris = np.vstack([t0, t1])
    simplices = pd.DataFrame({"v0": tris[:, 0], "v1": tris[:, 1], "v2": tris[:, 2]})

    return vertices, simplices


def make_elevation_overlay(
    vertices: pd.DataFrame,
    simplices: pd.DataFrame,
    file_name: Union[str, Path],
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    cmap="gist_earth",
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    width: int = 800,
) -> bool:
    """Render an elevation trimesh overlay using datashader.

    Parameters mirror :py:func:`make_elevation_trimesh` (for the data)
    and add lon/lat extent, colormap and value-range controls.
    """
    if not HAS_DATASHADER:
        logger.warning("Datashader is not available. Please install datashader.")
        return False
    if vertices is None or vertices.empty:
        return False

    try:
        import datashader as ds

        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        xl0, yl0 = transformer.transform(xlim[0], ylim[0])
        xl1, yl1 = transformer.transform(xlim[1], ylim[1])
        if xl0 > xl1:
            xl1 += 40075016.68557849
        x_range = (xl0, xl1)
        y_range = (yl0, yl1)
        ratio = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
        height = int(width * ratio)

        cvs = Canvas(
            x_range=x_range,
            y_range=y_range,
            plot_width=width,
            plot_height=height,
        )
        mesh = ds.utils.mesh(vertices, simplices)
        agg = cvs.trimesh(
            vertices,
            simplices,
            mesh=mesh,
            agg=ds.mean("z"),
            interp=False,
        )

        if isinstance(cmap, str):
            from matplotlib import colormaps

            cmap = colormaps[cmap]
        span = (cmin, cmax) if (cmin is not None and cmax is not None) else None
        img = tf.shade(agg, cmap=cmap, span=span, how="linear")

        path_dir = os.path.dirname(file_name) or os.getcwd()
        name = os.path.splitext(os.path.basename(file_name))[0]
        export_image(img, name, export_path=path_dir)
        return True
    except Exception:
        return False


class ElevationOverlay:
    """Lazy, cached elevation trimesh overlay renderer."""

    def __init__(self) -> None:
        self._vertices: pd.DataFrame = pd.DataFrame()
        self._simplices: pd.DataFrame = pd.DataFrame()

    def invalidate(self) -> None:
        """Drop the cached trimesh; it will be rebuilt on next render."""
        self._vertices = pd.DataFrame()
        self._simplices = pd.DataFrame()

    def render(
        self,
        face_xy: np.ndarray,
        z: np.ndarray,
        level: np.ndarray,
        dx0: float,
        dy0: float,
        rotation: float,
        source_crs: CRS,
        file_name: Union[str, Path],
        xlim: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        cmap="gist_earth",
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        width: int = 800,
    ) -> bool:
        """Render the elevation overlay, rebuilding the trimesh if needed."""
        if self._vertices.empty:
            self._vertices, self._simplices = make_elevation_trimesh(
                face_xy,
                z,
                level,
                dx0,
                dy0,
                rotation,
                source_crs,
            )
        return make_elevation_overlay(
            self._vertices,
            self._simplices,
            file_name,
            xlim=xlim,
            ylim=ylim,
            cmap=cmap,
            cmin=cmin,
            cmax=cmax,
            width=width,
        )
