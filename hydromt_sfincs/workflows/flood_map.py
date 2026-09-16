"""Flood maps from SFINCS output on a high-resolution DEM, with rendering.

Provides the ``FloodMap`` class: it combines cell water levels (``zsmax``,
``zs``) or cell volumes (``zvolmax``, ``subgrid_volume``) with a topobathy COG
and a cell-index COG into a flood depth raster, writes it as GeoTIFF, NetCDF
or PNG, renders Web Mercator overlays for a map viewer and matplotlib plots.

Two methods are available. ``"level"`` projects the horizontal cell water
level onto the pixels. ``"slope"`` reconstructs a bed-parallel water surface
per cell from the subgrid tables (``z_zmean``, ``z_dzbdx``, ``z_dzbdy``,
``z_level_res``) and the cell volume, with per-cell, blended or bed-trend
surfaces, and rules that keep a horizontal surface in submerged, sea-touching,
non-planar and ponded cells.

Also provides structure-aware cell index rasters: pixels that a thin dam, weir
or flood wall separates from their own cell centre are handed to the nearest
cell on their side, and the cells across a structure are stored in extra bands
so the surface interpolation never crosses one (see
``apply_structures_to_index_cog``).
"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from PIL import Image
from pyproj import Transformer
from rasterio import features
from rasterio.transform import Affine
from rasterio.warp import Resampling
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, MultiLineString, mapping, shape


logger = logging.getLogger(f"hydromt.{__name__}")

__all__ = [
    "FloodMap",
    "apply_structures_to_index_cog",
    "reassign_index_for_structures",
    "blocked_cells_for_pixels",
    "read_structures_from_index",
    "get_rgb_data_array",
    "get_appropriate_overview_level",
    "reproject_bbox",
]

# Sentinel for "argument not given" where None is a meaningful value
_KEEP = object()


class FloodMap:
    """Compute and visualise flood depth maps from water level and topobathy data.

    Uses Cloud Optimized GeoTIFF (COG) files for topography and cell
    indices, and combines them with water level arrays to produce flood
    depth grids. Supports writing output as GeoTIFF/NetCDF, creating
    PNG map overlays, and matplotlib plotting.

    Parameters
    ----------
    topobathy_file : str | Path | None
        Path to the topobathy COG file.
    index_file : str | Path | None
        Path to the cell-index COG file.
    zbmin : float
        Minimum allowable topobathy value; below this is masked.
    zbmax : float
        Maximum allowable topobathy value; above this is masked.
    hmin : float
        Minimum water depth threshold; shallower areas are masked.
    max_pixel_size : float
        Maximum pixel size for overview level selection.
    data_array_name : str
        Name of the depth variable in the output dataset.
    cmap : str | None
        Matplotlib colormap name.
    cmin : float | None
        Minimum value for colormap normalization.
    cmax : float | None
        Maximum value for colormap normalization.
    color_values : list[dict] | None
        Discrete color definitions with ``lower_value``, ``upper_value``,
        and ``color``/``rgb`` keys.
    """

    def __init__(
        self,
        topobathy_file: str | Path | None = None,
        index_file: str | Path | None = None,
        zbmin: float = 0.0,
        zbmax: float = 99999.9,
        hmin: float = 0.1,
        max_pixel_size: float = 0.0,
        data_array_name: str = "water_depth",
        cmap: str | None = None,
        cmin: float | None = None,
        cmax: float | None = None,
        color_values: list[dict] | None = None,
    ) -> None:
        self.topobathy_file = None
        self.index_file = None
        self.zb = None
        self.indices = None
        self.zbmin = zbmin
        self.zbmax = zbmax
        self.hmin = hmin
        self.max_pixel_size = max_pixel_size
        self.data_array_name = data_array_name
        self.color_values = color_values if color_values is not None else "default"
        self.cmap = cmap if cmap is not None else "jet"
        self.cmin = cmin if cmin is not None else 0.0
        self.cmax = cmax if cmax is not None else 1.0
        self.discrete_colors = color_values is not None
        self.ds = xr.Dataset()

        # Water level (all methods) and cell volume / subgrid tables ("slope")
        self.zs = None
        self.volume = None
        self.method = "level"
        self.subgrid = None
        self.xc = None
        self.yc = None
        self.area = None
        # "slope" method options, see set_slope_options()
        self.volume_mode = "pixels"
        self.slope_zmin = None
        self.slope_max_residual_ratio = None
        self.slope_max_residual_relief = 10.0
        self.surface_mode = "cell"
        self.smoothing_max_jump = 1.0
        self.flat_level_rule = False
        self.flat_level_ratio = 0.2
        self.flat_level_min_drop = 0.25
        self.flat_level_min_neighbours = 2
        self._triangulation = None
        # structures (thin dams, weirs, walls) as segment arrays in raster CRS
        self._structure_a = None
        self._structure_b = None
        self._index_has_blocked = False
        self._blocked = None  # (3, n_valid) blocked cell ids of the current block

        if topobathy_file is not None:
            self.set_topobathy_file(topobathy_file)
        if index_file is not None:
            self.set_index_file(index_file)

        self.legend = {}
        self.legend["title"] = "Flood Depth (m)"
        self.legend["contour"] = []
        self.legend["contour"].append(
            {"color": "#FF0000", "lower_value": 2.0, "text": "2.0+ m"}
        )
        self.legend["contour"].append(
            {
                "color": "#FFA500",
                "lower_value": 1.0,
                "upper_value": 2.0,
                "text": "1.0--2.0 m",
            }
        )
        self.legend["contour"].append(
            {
                "color": "#FFFF00",
                "lower_value": 0.3,
                "upper_value": 1.0,
                "text": "0.3--1.0 m",
            }
        )
        self.legend["contour"].append(
            {
                "color": "#00FF00",
                "lower_value": 0.1,
                "upper_value": 0.3,
                "text": "0.1--0.3 m",
            }
        )

    def set_topobathy_file(self, topobathy_file: str | Path) -> None:
        """Set the topobathy file and open it with rasterio.

        Parameters
        ----------
        topobathy_file : str | Path
            Path to the topobathy COG file.
        """
        self.topobathy_file = topobathy_file
        self.zb = rasterio.open(self.topobathy_file)

    def set_index_file(self, index_file: str | Path) -> None:
        """Set the index file and open it with rasterio.

        Parameters
        ----------
        index_file : str | Path
            Path to the cell-index COG file.
        """
        self.index_file = index_file
        self.indices = rasterio.open(self.index_file)
        # A structure-aware index (see apply_structures_to_index_cog) carries, in
        # bands 2-4, the cells across a structure from each pixel; the blend
        # and trend surface modes read those, so no structures are needed at
        # map time. The stored geometry tag is kept only as a fallback.
        self._index_has_blocked = self.indices.count >= 4
        if not self._index_has_blocked:
            lines = structures_from_tag(self.indices.tags().get(STRUCTURES_TAG))
            self.set_structures(lines if lines else None)

    def close(self) -> None:
        """Close the topobathy, index, and dataset file handles."""
        if self.zb is not None:
            self.zb.close()
        if self.indices is not None:
            self.indices.close()
        self.ds.close()

    def read(self, tiffile: str | Path) -> None:
        """Read a GeoTIFF file with pre-computed flood depth data.

        Parameters
        ----------
        tiffile : str | Path
            Path to the GeoTIFF file.
        """
        self.ds = xr.Dataset()
        self.ds["water_depth"] = rioxarray.open_rasterio(tiffile, masked=True).squeeze()

    def set_water_level(self, zs: float | np.ndarray) -> None:
        """Set the water level data used for flood depth computation.

        Parameters
        ----------
        zs : float | np.ndarray
            A scalar or 1-D array of water levels indexed by cell index.
        """
        self.zs = zs

    def set_volume(self, volume: np.ndarray | None) -> None:
        """Set the subgrid cell volumes used by the ``"slope"`` method.

        Parameters
        ----------
        volume : np.ndarray | None
            1-D array of subgrid cell volumes (m3) indexed by cell index,
            typically ``zvolmax`` from the SFINCS map file. NaN marks dry
            cells. When ``None``, the volume is recovered from the water level
            by inverting the ``z_level`` table, which loses resolution on
            steep cells.
        """
        self.volume = None if volume is None else np.asarray(volume, dtype=np.float64)

    def set_method(self, method: str) -> None:
        """Select the flood depth method.

        Parameters
        ----------
        method : str
            ``"level"``: horizontal water surface per cell,
            ``h = zs[index] - zb`` (default). ``"slope"``: bed-parallel water
            surface reconstructed from the subgrid residual tables, see
            :meth:`set_subgrid`; a cell that is fully submerged under a
            horizontal surface keeps that horizontal surface.
        """
        if method not in ("level", "slope"):
            raise ValueError("method must be 'level' or 'slope'")
        self.method = method

    def set_subgrid(
        self,
        subgrid: str | Path | xr.Dataset,
        xc: np.ndarray,
        yc: np.ndarray,
        area: np.ndarray,
    ) -> None:
        """Set the subgrid tables and cell geometry for the ``"slope"`` method.

        The water surface in a cell is reconstructed as
        ``zs_i = eta + z_zmean + z_dzbdx (x_i - xc) + z_dzbdy (y_i - yc)``
        with ``eta`` the residual level at the cell volume from
        ``z_level_res``. Cells whose volume reaches ``z_volmax`` (fully
        submerged under a horizontal surface, e.g. sea and full ponds) keep
        a horizontal surface instead. Calling this method selects the
        ``"slope"`` method.

        Parameters
        ----------
        subgrid : str | Path | xr.Dataset
            SFINCS quadtree subgrid file (or opened dataset) with
            ``z_zmean``, ``z_dzbdx``, ``z_dzbdy``, ``z_level_res``,
            ``z_volmax_res``, ``z_level``, ``z_zmax`` and ``z_volmax``.
        xc, yc : np.ndarray
            Cell centre coordinates in the CRS of the topobathy raster,
            indexed by cell index.
        area : np.ndarray
            Cell area (m2), indexed by cell index.
        """
        if isinstance(subgrid, (str, Path)):
            with xr.open_dataset(subgrid) as ds:
                ds = ds.load()
        else:
            ds = subgrid
        needed = [
            "z_zmean",
            "z_dzbdx",
            "z_dzbdy",
            "z_level_res",
            "z_volmax_res",
            "z_level",
            "z_zmax",
            "z_volmax",
        ]
        missing = [v for v in needed if v not in ds]
        if missing:
            raise ValueError(f"Subgrid file lacks variables {missing}")
        self.subgrid = {v: ds[v].to_numpy().astype(np.float64) for v in needed}
        ncell = self.subgrid["z_zmean"].size
        self.xc = np.asarray(xc, dtype=np.float64).reshape(ncell)
        self.yc = np.asarray(yc, dtype=np.float64).reshape(ncell)
        self.area = np.asarray(area, dtype=np.float64).reshape(ncell)
        self.method = "slope"
        self._triangulation = None

    def set_structures(self, lines) -> None:
        """Set structures that water surfaces must not be interpolated across.

        In the ``"blend"`` and ``"trend"`` surface modes a neighbouring cell
        contributes to a pixel only if the straight link between them does
        not cross a structure, which is the same rule SFINCS uses to block a
        face (the link between two cell centres crossing the line). A
        structure-aware index raster made with
        :func:`apply_structures_to_index_cog` carries
        its structures, and :meth:`set_index_file` sets them automatically;
        call this only to override or clear them.

        Parameters
        ----------
        lines : iterable of shapely LineString / MultiLineString, or None
            Structures in the CRS of the topobathy raster. ``None`` clears.
        """
        if lines is None:
            self._structure_a = None
            self._structure_b = None
            return
        a, b = _line_segments(list(lines))
        self._structure_a = a if a.shape[0] else None
        self._structure_b = b if a.shape[0] else None

    def _structure_near_mask(self, zb: xr.DataArray, valid: np.ndarray) -> np.ndarray | None:
        """Valid pixels close enough to a structure to need the crossing test.

        Returns ``None`` when no structures are set. The radius is 1.5 times
        the largest spacing between neighbouring cell centres, which covers
        every pixel whose link to a candidate cell centre can cross a line.
        """
        if self._structure_a is None or self._blocked is not None:
            return None
        d, _ = cKDTree(np.column_stack([self.xc, self.yc])).query(
            np.column_stack([self.xc, self.yc]), k=2
        )
        radius = 1.5 * float(d[:, 1].max())
        geoms = [
            (LineString([self._structure_a[i], self._structure_b[i]]).buffer(radius), 1)
            for i in range(self._structure_a.shape[0])
        ]
        mask = features.rasterize(
            geoms,
            out_shape=zb.shape,
            transform=zb.rio.transform(),
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
        return mask[valid]

    def _drop_crossing(
        self,
        admissible: np.ndarray,
        p: np.ndarray,
        verts: np.ndarray,
        centres: np.ndarray,
        near: np.ndarray | None,
        blocked: np.ndarray | None = None,
    ) -> np.ndarray:
        """Clear ``admissible[i, j]`` where cell j lies across a structure from pixel i.

        With ``blocked`` (the pre-computed blocked cell ids from a
        structure-aware index, shape ``(3, n)``) no geometry is used;
        otherwise the link pixel-centre is tested against the structures
        set with :meth:`set_structures`.
        """
        if blocked is not None:
            for j in range(verts.shape[1]):
                hit = (verts[:, [j]] == blocked.T).any(axis=1)
                admissible[hit, j] = False
            return admissible
        if near is None or self._structure_a is None or not near.any():
            return admissible
        sel = np.flatnonzero(near)
        for j in range(verts.shape[1]):
            cj = centres[verts[sel, j]]
            cross = _segments_cross(p[sel], cj, self._structure_a, self._structure_b)
            admissible[sel[cross], j] = False
        return admissible

    def set_slope_options(
        self,
        volume_mode: str | None = None,
        slope_zmin: float | None | object = _KEEP,
        max_residual_ratio: float | None | object = _KEEP,
        max_residual_relief: float | None | object = _KEEP,
        surface_mode: str | None = None,
        smoothing_max_jump: float | None = None,
        surface_smoothing: bool | None = None,
        flat_level_rule: bool | None = None,
        flat_level_ratio: float | None = None,
        flat_level_min_drop: float | None = None,
        flat_level_min_neighbours: int | None = None,
    ) -> None:
        """Set options of the ``"slope"`` method.

        Arguments that are not given keep their current value.

        Parameters
        ----------
        flat_level_rule : bool | None
            When ``True``, a cell whose water level differs from that of at
            least ``flat_level_min_neighbours`` neighbours by less than
            ``flat_level_ratio`` times the drop in lowest pixel between them
            (for drops above ``flat_level_min_drop``) is treated as ponded
            water: it keeps a horizontal surface at its own level and does
            not influence its neighbours in the blend and trend modes. Sheet
            flow on a slope has a level difference close to the bed drop and
            is not affected. Default ``False``.
        flat_level_ratio : float | None
            Level difference over bed drop below which a pair counts as flat.
            Default 0.2.
        flat_level_min_drop : float | None
            Minimum drop in lowest pixel (m) for a pair to be tested, so
            cells side by side along a contour are not compared. Default 0.25.
        flat_level_min_neighbours : int | None
            Number of flat neighbours needed to flag a cell. Default 2.
        surface_mode : str | None
            How the water surface is built from the per-cell results.

            ``"cell"`` (default): each pixel uses the surface of its own
            cell, ``z_zmean + eta`` plus the cell's slope plane. Exact
            per-cell volume; the surface jumps at cell faces.

            ``"blend"``: barycentric blend of the surfaces of the three
            surrounding cell centres, each extrapolated along its own plane
            to the pixel. Vertices whose surface differs from the own-cell
            surface by more than ``smoothing_max_jump`` are dropped.

            ``"trend"``: a continuous bed trend is interpolated from the
            cell means between centres; residuals, the volume inversion and
            the sheet level ``eta`` are all taken against that trend, and
            ``eta`` is interpolated between centres. The surface is
            continuous everywhere; per-cell volume is approximate.

            In all modes dry cells stay dry and horizontal-rule cells keep
            their own horizontal level.
        smoothing_max_jump : float | None
            Step tolerance (m) for the ``"blend"`` and ``"trend"`` modes: a
            neighbour joins the interpolation only when its surface
            (``"blend"``) or sheet level (``"trend"``) lies within this
            distance of the pixel's own cell value, so real steps keep their
            jump. Default 1.0.
        surface_smoothing : bool | None
            Backwards-compatible switch: ``True`` selects ``"blend"``,
            ``False`` selects ``"cell"``.
        volume_mode : str | None
            ``"pixels"`` (default): the residual level of a cell is found by
            distributing its volume over the residual elevations of the
            topobathy pixels in the cell, so the map holds the cell volume
            exactly at the raster resolution in use. ``"table"``: the level
            is read from the ``z_level_res`` table, which was built on the
            subgrid pixels; faster, but with a raster of different resolution
            the wet pixel set no longer matches the volume for thin sheets.
        slope_zmin : float | None
            Cells whose lowest pixel (``z_zmin``) is below this level keep a
            horizontal surface, e.g. ``0.0`` to exclude cells touching the
            sea. Pass ``None`` to disable the test (the initial state).
        max_residual_ratio : float | None
            Cells whose residual relief exceeds this fraction of the raw
            relief keep a horizontal surface. Off by default (``None``): on
            gentle terrain an embankment or ditch already gives a high ratio
            although a tilted sheet is perfectly adequate there.
        max_residual_relief : float | None
            Cells whose residual relief (highest minus lowest residual)
            exceeds this many metres keep a horizontal surface. This is the
            cliff and gully test: such cells hold water at a level a plane
            cannot describe. Default 10.0; ``None`` disables it.
        """
        if volume_mode is not None:
            if volume_mode not in ("pixels", "table"):
                raise ValueError("volume_mode must be 'pixels' or 'table'")
            self.volume_mode = volume_mode
        if slope_zmin is not _KEEP:
            self.slope_zmin = slope_zmin
        if max_residual_ratio is not _KEEP:
            self.slope_max_residual_ratio = max_residual_ratio
        if max_residual_relief is not _KEEP:
            self.slope_max_residual_relief = max_residual_relief
        if surface_mode is not None:
            if surface_mode not in ("cell", "blend", "trend"):
                raise ValueError("surface_mode must be 'cell', 'blend' or 'trend'")
            self.surface_mode = surface_mode
        if surface_smoothing is not None:
            self.surface_mode = "blend" if surface_smoothing else "cell"
        if smoothing_max_jump is not None:
            self.smoothing_max_jump = float(smoothing_max_jump)
        if flat_level_rule is not None:
            self.flat_level_rule = bool(flat_level_rule)
        if flat_level_ratio is not None:
            self.flat_level_ratio = float(flat_level_ratio)
        if flat_level_min_drop is not None:
            self.flat_level_min_drop = float(flat_level_min_drop)
        if flat_level_min_neighbours is not None:
            self.flat_level_min_neighbours = int(flat_level_min_neighbours)

    def _blend_surfaces(
        self,
        px: np.ndarray,
        py: np.ndarray,
        own: np.ndarray,
        zs_own: np.ndarray,
        xc: np.ndarray,
        yc: np.ndarray,
        w_cell: np.ndarray,
        sx: np.ndarray,
        sy: np.ndarray,
        wet: np.ndarray,
        chunk: int = 2_000_000,
        near: np.ndarray | None = None,
    ) -> np.ndarray:
        """Blend the surfaces of the surrounding cells at each pixel.

        The cell centres are triangulated once (Delaunay, cached). For each
        pixel the three cells of its triangle contribute their surface
        ``W_j + sx_j (x - xc_j) + sy_j (y - yc_j)`` with barycentric weights.
        Dry cells, and cells whose surface at the pixel differs from the
        pixel's own-cell surface by more than ``smoothing_max_jump``, are
        dropped and the weights renormalised. Pixels outside the
        triangulation, or with no admissible vertex, keep their own surface.
        (Interpolating the centre levels alone instead of the planes was
        tried and fails on steep terrain: neighbouring centre levels differ
        by the bed drop between centres, metres on a slope.)

        Parameters
        ----------
        px, py : np.ndarray
            Pixel coordinates (same origin as ``xc``, ``yc``).
        own : np.ndarray
            Cell index of each pixel.
        zs_own : np.ndarray
            Surface of the pixel's own cell at the pixel (fallback).
        xc, yc, w_cell, sx, sy : np.ndarray
            Cell centres, centre water levels and surface slopes.
        wet : np.ndarray
            Per-cell wet flag.

        Returns
        -------
        np.ndarray
            Blended surface per pixel (float32).
        """
        tri = self._get_triangulation(xc, yc)
        out = zs_own.astype(np.float32).copy()
        npix = px.size
        for i0 in range(0, npix, chunk):
            i1 = min(i0 + chunk, npix)
            p = np.column_stack([px[i0:i1], py[i0:i1]]).astype(np.float64)
            simplex = tri.find_simplex(p)
            inside = simplex >= 0
            if not inside.any():
                continue
            s = simplex[inside]
            pi = p[inside]
            verts = tri.simplices[s]  # (n, 3)
            T = tri.transform[s]  # (n, 3, 2)
            b = np.einsum("nij,nj->ni", T[:, :2, :], pi - T[:, 2, :])
            bary = np.column_stack([b, 1.0 - b.sum(axis=1)])
            bary = np.clip(bary, 0.0, 1.0)
            zv = (
                w_cell[verts]
                + sx[verts] * (pi[:, 0:1] - xc[verts])
                + sy[verts] * (pi[:, 1:2] - yc[verts])
            )
            own_here = zs_own[i0:i1][inside].astype(np.float64)[:, None]
            admissible = wet[verts] & (np.abs(zv - own_here) <= self.smoothing_max_jump)
            if near is not None or self._blocked is not None:
                admissible = self._drop_crossing(
                    admissible,
                    pi,
                    verts,
                    np.column_stack([xc, yc]),
                    None if near is None else near[i0:i1][inside],
                    None if self._blocked is None else self._blocked[:, i0:i1][:, inside],
                )
            w = bary * admissible
            wsum = w.sum(axis=1)
            ok = wsum > 1.0e-6
            blend = (w * zv).sum(axis=1) / np.where(ok, wsum, 1.0)
            idx = np.arange(i0, i1)[inside]
            out[idx[ok]] = blend[ok].astype(np.float32)
        # pixels of cells that take no part in the blend keep their own surface
        out[~wet[own]] = zs_own[~wet[own]]
        return out

    def _cell_volume_from_level(self, zs: np.ndarray) -> np.ndarray:
        """Invert the ``z_level`` table: cell volume at water level ``zs``."""
        sg = self.subgrid
        zl = sg["z_level"]
        vmax = sg["z_volmax"]
        zmax = sg["z_zmax"]
        ncell, nlev = zl.shape
        zs = np.where(np.isfinite(zs), zs, -np.inf)
        j = np.clip((zl <= zs[:, None]).sum(axis=1) - 1, 0, nlev - 2)
        rows = np.arange(ncell)
        z0 = zl[rows, j]
        z1 = zl[rows, j + 1]
        w = np.clip((zs - z0) / np.maximum(z1 - z0, 1.0e-6), 0.0, 1.0)
        vol = (j + w) * vmax / (nlev - 1)
        vol = np.where(zs >= zmax, vmax + (zs - zmax) * self.area, vol)
        return np.where(zs <= zl[:, 0], 0.0, vol)

    def _residual_level(self, vol: np.ndarray) -> np.ndarray:
        """Residual water level ``eta`` at cell volume ``vol`` from ``z_level_res``.

        Inside the table the level is interpolated linearly in volume, as
        SFINCS does for ``z_level``; above it the surface rises linearly with
        the cell area, which also covers cells with a flat residual table.
        """
        sg = self.subgrid
        zl = sg["z_level_res"]
        vmax = sg["z_volmax_res"]
        ncell, nlev = zl.shape
        safe_vmax = np.where(vmax > 0.0, vmax, 1.0)
        frac = np.where(vmax > 0.0, vol / safe_vmax, np.inf)
        pos = np.clip(frac * (nlev - 1), 0.0, nlev - 1.0)
        j = np.minimum(np.floor(pos).astype(int), nlev - 2)
        w = pos - j
        rows = np.arange(ncell)
        eta_in = zl[rows, j] * (1.0 - w) + zl[rows, j + 1] * w
        eta_above = zl[:, -1] + (vol - vmax) / self.area
        eta = np.where(frac < 1.0, eta_in, eta_above)
        return np.where(vol > 0.0, eta, zl[:, 0])

    def _residual_level_pixels(
        self,
        k: np.ndarray,
        zres: np.ndarray,
        vol: np.ndarray,
        px_area: float,
    ) -> np.ndarray:
        """Residual level ``eta`` per cell by exact inversion on raster pixels.

        Solves ``sum_i max(eta - zres_i, 0) * px_area = vol`` per cell over
        the pixels of that cell, so the reconstructed depth field holds the
        cell volume exactly at the raster resolution in use.

        Parameters
        ----------
        k : np.ndarray
            1-D cell index of each valid pixel.
        zres : np.ndarray
            Residual elevation of each valid pixel.
        vol : np.ndarray
            Cell volume (m3) per cell.
        px_area : float
            Pixel area (m2).

        Returns
        -------
        np.ndarray
            ``eta`` per cell; NaN for cells without pixels.
        """
        ncell = vol.size
        order = np.lexsort((zres, k))
        ks = k[order]
        zs_ = zres[order].astype(np.float64)
        cnt = np.bincount(ks, minlength=ncell)
        off = np.zeros(ncell + 1, dtype=np.int64)
        off[1:] = np.cumsum(cnt)
        npix = ks.size
        pos = np.arange(npix) - off[ks]
        j = pos + 1  # wet pixel count when the surface lies between pixel pos and pos + 1
        cum = np.cumsum(zs_)
        s_cell = cum - (cum[off[ks]] - zs_[off[ks]])  # inclusive cumulative sum within the cell
        z_next = np.empty(npix)
        z_next[:-1] = zs_[1:]
        z_next[-1] = np.inf
        z_next[pos == cnt[ks] - 1] = np.inf
        with np.errstate(invalid="ignore"):
            v_next = px_area * (j * z_next - s_cell)  # volume when pixel pos + 1 starts to wet
        ok = vol[ks] <= v_next
        cand = np.where(ok, np.arange(npix), npix)
        has = cnt > 0
        first = np.full(ncell, npix, dtype=np.int64)
        first[has] = np.minimum.reduceat(cand, off[:-1][has])
        eta = np.full(ncell, np.nan)
        jj = j[first[has]].astype(np.float64)
        eta[has] = vol[has] / (px_area * jj) + s_cell[first[has]] / jj
        dry = has & ~(vol > 0.0)
        eta[dry] = zs_[off[:-1][dry]]
        return eta

    def _surface_slope(
        self, indices: np.ndarray, zb: xr.DataArray, valid: np.ndarray
    ) -> np.ndarray:
        """Per-pixel water surface for the ``"slope"`` method.

        Parameters
        ----------
        indices : np.ndarray
            2-D cell index per pixel (nodata already replaced by 0; the
            caller masks those pixels afterwards).
        zb : xr.DataArray
            Topobathy block with ``x`` and ``y`` pixel-centre coordinates.
        valid : np.ndarray
            2-D mask of pixels with a valid cell index and elevation.

        Returns
        -------
        np.ndarray
            Water surface elevation per pixel (float32).
        """
        if self.subgrid is None:
            raise ValueError("Call set_subgrid() before using method 'slope'.")
        sg = self.subgrid
        ncell = sg["z_zmean"].size

        # Cell volume: given, or recovered from the water level
        zs_cell = None
        if self.zs is not None and not isinstance(self.zs, float):
            zs_cell = np.asarray(self.zs, dtype=np.float64).reshape(ncell)
        if self.volume is not None:
            vol = self.volume.reshape(ncell)
        elif zs_cell is not None:
            vol = self._cell_volume_from_level(zs_cell)
        else:
            raise ValueError("Method 'slope' needs set_volume() or set_water_level().")
        vol = np.where(np.isfinite(vol), vol, 0.0)

        # Cells that keep a horizontal surface: fully submerged under a
        # horizontal surface, touching the sea, or not planar enough.
        horizontal = vol >= sg["z_volmax"]
        if self.slope_zmin is not None:
            horizontal |= sg["z_level"][:, 0] < self.slope_zmin
        res_relief = sg["z_level_res"][:, -1] - sg["z_level_res"][:, 0]
        if self.slope_max_residual_relief is not None:
            horizontal |= res_relief > self.slope_max_residual_relief
        if self.slope_max_residual_ratio is not None:
            raw_relief = np.maximum(sg["z_zmax"] - sg["z_level"][:, 0], 1.0e-3)
            horizontal |= res_relief > self.slope_max_residual_ratio * raw_relief
        if zs_cell is None:
            zs_cell = sg["z_zmax"] + (vol - sg["z_volmax"]) / self.area
        zs_cell = np.where(np.isfinite(zs_cell), zs_cell, -np.inf)

        # Pixel offsets from the cell centre, in float32 relative to an origin
        x0 = float(self.xc.min())
        y0 = float(self.yc.min())
        x = (zb.x.to_numpy() - x0).astype(np.float32)
        y = (zb.y.to_numpy() - y0).astype(np.float32)
        xx, yy = np.meshgrid(x, y)
        xc = (self.xc - x0).astype(np.float32)
        yc = (self.yc - y0).astype(np.float32)

        # Flat-level rule: a cell whose water level is nearly the same as a
        # neighbour's despite a real bed drop between them holds ponded
        # (coastal, riverine, backwater) water rather than a rain sheet.
        if self.flat_level_rule:
            horizontal |= self._flat_level_cells(
                zs_cell, vol, xc.astype(np.float64), yc.astype(np.float64)
            )

        k = indices
        plane = (
            sg["z_zmean"].astype(np.float32)[k]
            + sg["z_dzbdx"].astype(np.float32)[k] * (xx - xc[k])
            + sg["z_dzbdy"].astype(np.float32)[k] * (yy - yc[k])
        )

        px_area = float(abs(zb.rio.resolution()[0] * zb.rio.resolution()[1]))

        if self.surface_mode == "trend":
            return self._surface_trend(
                k, valid, xx, yy, xc, yc, plane, zb, px_area, vol, horizontal, zs_cell
            )

        if self.volume_mode == "pixels":
            zres = zb.to_numpy()[valid].astype(np.float32) - plane[valid]
            eta = self._residual_level_pixels(k[valid].ravel(), zres, vol, px_area)
            eta = np.where(np.isfinite(eta), eta, self._residual_level(vol))
        else:
            eta = self._residual_level(vol)

        zs_tilt = eta.astype(np.float32)[k] + plane
        zs_pix = np.where(horizontal[k], zs_cell.astype(np.float32)[k], zs_tilt)

        if self.surface_mode == "blend":
            # Per-cell surface: centre level and slope (zero for horizontal cells)
            w_cell = np.where(horizontal, zs_cell, eta + sg["z_zmean"])
            sx = np.where(horizontal, 0.0, sg["z_dzbdx"])
            sy = np.where(horizontal, 0.0, sg["z_dzbdy"])
            # Horizontal-rule cells (sea, ponds, submerged, non-planar) keep
            # their own level and do not influence their neighbours
            wet = (vol > 0.0) & np.isfinite(w_cell) & ~horizontal
            blended = self._blend_surfaces(
                xx[valid].astype(np.float64),
                yy[valid].astype(np.float64),
                k[valid].ravel(),
                zs_pix[valid],
                xc.astype(np.float64),
                yc.astype(np.float64),
                w_cell,
                sx,
                sy,
                wet,
                near=self._structure_near_mask(zb, valid),
            )
            zs_pix = zs_pix.copy()
            zs_pix[valid] = blended
            # dry cells stay dry
            zs_pix[~(vol > 0.0)[k]] = -np.inf
        return zs_pix

    def _flat_level_cells(
        self, zs_cell: np.ndarray, vol: np.ndarray, xc: np.ndarray, yc: np.ndarray
    ) -> np.ndarray:
        """Cells whose water level is flat towards neighbours with a real bed drop.

        For every pair of neighbouring cells (edges of the centre
        triangulation) with both cells wet and a difference in lowest pixel
        (``z_zmin``) larger than ``flat_level_min_drop``, the pair is flat
        when the water-level difference is smaller than ``flat_level_ratio``
        times that drop. The lowest pixel is used because the solver's level
        follows it: sheet flow gives a level difference close to the drop,
        ponded water gives a flat level. A cell is flagged when it is flat
        towards at least ``flat_level_min_neighbours`` neighbours, since
        ponds are contiguous while a single flat pair can be a chance
        alignment along a ditch. Pairs with no bed drop between them
        (neighbours along a contour) are not tested.

        Returns
        -------
        np.ndarray
            Boolean flag per cell.
        """
        sg = self.subgrid
        tri = self._get_triangulation(xc, yc)
        indptr, indices = tri.vertex_neighbor_vertices
        ncell = zs_cell.size
        i = np.repeat(np.arange(ncell), np.diff(indptr))
        j = indices
        keep = i < j  # each pair once
        i, j = i[keep], j[keep]
        zmin = sg["z_level"][:, 0]
        wet = (vol > 0.0) & np.isfinite(zs_cell)
        drop = np.abs(zmin[i] - zmin[j])
        dlevel = np.abs(zs_cell[i] - zs_cell[j])
        flat = (
            wet[i]
            & wet[j]
            & (drop > self.flat_level_min_drop)
            & (dlevel < self.flat_level_ratio * drop)
        )
        nflat = np.bincount(np.concatenate([i[flat], j[flat]]), minlength=ncell)
        return nflat >= max(int(self.flat_level_min_neighbours), 1)

    def _get_triangulation(self, xc: np.ndarray, yc: np.ndarray):
        """Delaunay triangulation of the cell centres (cached)."""
        if self._triangulation is None:
            self._triangulation = Delaunay(np.column_stack([xc, yc]))
        return self._triangulation

    def _barycentric(
        self, px: np.ndarray, py: np.ndarray, xc: np.ndarray, yc: np.ndarray, chunk: int = 2_000_000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simplex index and barycentric weights of pixels in the centre triangulation.

        Returns
        -------
        simplex : np.ndarray (npix,) int32
            Triangle index per pixel, -1 outside the triangulation.
        bary : np.ndarray (npix, 3) float32
            Barycentric weights of the triangle's three vertices (0 outside).
        """
        tri = self._get_triangulation(xc, yc)
        npix = px.size
        simplex = np.full(npix, -1, dtype=np.int32)
        bary = np.zeros((npix, 3), dtype=np.float32)
        for i0 in range(0, npix, chunk):
            i1 = min(i0 + chunk, npix)
            p = np.column_stack([px[i0:i1], py[i0:i1]]).astype(np.float64)
            s = tri.find_simplex(p)
            inside = s >= 0
            if not inside.any():
                continue
            T = tri.transform[s[inside]]
            b = np.einsum("nij,nj->ni", T[:, :2, :], p[inside] - T[:, 2, :])
            bb = np.clip(np.column_stack([b, 1.0 - b.sum(axis=1)]), 0.0, 1.0)
            simplex[i0:i1][inside] = s[inside]
            bary[i0:i1][inside] = bb.astype(np.float32)
        return simplex, bary

    def _interp_cells(
        self,
        values: np.ndarray,
        admissible: np.ndarray,
        simplex: np.ndarray,
        bary: np.ndarray,
        fallback: np.ndarray,
        ref: np.ndarray | None = None,
        max_jump: float | None = None,
        chunk: int = 2_000_000,
        pxy: np.ndarray | None = None,
        near: np.ndarray | None = None,
        centres: np.ndarray | None = None,
    ) -> np.ndarray:
        """Barycentric interpolation of a per-cell quantity at the pixels.

        With ``pxy`` (pixel coordinates), ``near`` (pixels close to a
        structure) and ``centres`` given, a vertex is dropped when the link
        from the pixel to that cell centre crosses a structure.

        Vertices that are not admissible, or whose value differs from
        ``ref`` by more than ``max_jump``, are dropped and the weights
        renormalised; pixels without an admissible vertex, or outside the
        triangulation, take ``fallback``.
        """
        tri = self._triangulation
        out = fallback.astype(np.float32).copy()
        npix = simplex.size
        for i0 in range(0, npix, chunk):
            i1 = min(i0 + chunk, npix)
            s = simplex[i0:i1]
            inside = s >= 0
            if not inside.any():
                continue
            verts = tri.simplices[s[inside]]
            bw = bary[i0:i1][inside].astype(np.float64)
            adm = admissible[verts]
            if (near is not None and pxy is not None) or self._blocked is not None:
                adm = self._drop_crossing(
                    adm.copy(),
                    None if pxy is None else pxy[i0:i1][inside],
                    verts,
                    centres,
                    None if near is None else near[i0:i1][inside],
                    None if self._blocked is None else self._blocked[:, i0:i1][:, inside],
                )
            w = bw * adm
            vals = values[verts]
            if max_jump is not None:
                if ref is not None:
                    r = ref[i0:i1][inside][:, None]
                else:
                    # symmetric reference: the nearest admissible vertex
                    r = vals[np.arange(vals.shape[0]), np.argmax(w, axis=1)][:, None]
                w = w * (np.abs(vals - r) <= max_jump)
            wsum = w.sum(axis=1)
            ok = wsum > 1.0e-6
            est = (w * vals).sum(axis=1) / np.where(ok, wsum, 1.0)
            idx = np.arange(i0, i1)[inside]
            out[idx[ok]] = est[ok].astype(np.float32)
        return out

    def _surface_trend(
        self,
        k: np.ndarray,
        valid: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        xc: np.ndarray,
        yc: np.ndarray,
        plane: np.ndarray,
        zb: xr.DataArray,
        px_area: float,
        vol: np.ndarray,
        horizontal: np.ndarray,
        zs_cell: np.ndarray,
    ) -> np.ndarray:
        """Per-pixel water surface for ``surface_mode="trend"``.

        A continuous bed trend ``T(x)`` is the barycentric interpolation of
        the cell means ``z_zmean`` between cell centres (own-cell plane
        outside the triangulation). Every pixel's residual is taken against
        that trend, the cell volume is inverted on those residuals to give a
        sheet level ``eta`` per cell, and ``eta`` is interpolated between
        centres in the same way. The surface ``T + eta`` is then continuous
        everywhere. Horizontal-rule cells keep their own level and are left
        out of the ``eta`` interpolation; dry cells stay dry.
        """
        sg = self.subgrid
        own = k[valid].ravel()
        px = xx[valid].astype(np.float64)
        py = yy[valid].astype(np.float64)
        simplex, bary = self._barycentric(px, py, xc.astype(np.float64), yc.astype(np.float64))

        # Continuous bed trend
        zmean = sg["z_zmean"]
        trend = self._interp_cells(
            zmean, np.isfinite(zmean), simplex, bary, fallback=plane[valid]
        )

        # Sheet level per cell from the volume, on residuals against the trend
        zres = zb.to_numpy()[valid].astype(np.float32) - trend
        eta = self._residual_level_pixels(own, zres, vol, px_area)
        eta = np.where(np.isfinite(eta), eta, 0.0)

        # Sheet level at the pixel, interpolated between wet tilted cells
        wet = (vol > 0.0) & ~horizontal
        near = self._structure_near_mask(zb, valid)
        eta_pix = self._interp_cells(
            eta,
            wet,
            simplex,
            bary,
            fallback=eta[own],
            ref=None,
            max_jump=self.smoothing_max_jump,
            pxy=np.column_stack([px, py]) if near is not None else None,
            near=near,
            centres=np.column_stack([xc, yc]).astype(np.float64),
        )
        zs_valid = trend + eta_pix
        zs_valid = np.where(horizontal[own], zs_cell.astype(np.float32)[own], zs_valid)
        zs_valid[~(vol[own] > 0.0)] = -np.inf

        zs_pix = np.full(k.shape, -np.inf, dtype=np.float32)
        zs_pix[valid] = zs_valid
        return zs_pix

    def make(
        self,
        max_pixel_size: float = 0.0,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> xr.Dataset:
        """Compute flood depth from water levels, topobathy, and cell indices.

        Reads topobathy and index COGs at the appropriate overview level,
        computes ``h = zs[index] - zb``, and masks areas that are too
        shallow or outside elevation bounds.

        Parameters
        ----------
        max_pixel_size : float
            Maximum pixel size in metres for overview selection. If 0.0,
            the native resolution is used.
        bbox : tuple[float, float, float, float] | None
            Bounding box ``(minx, miny, maxx, maxy)`` to clip the data.

        Returns
        -------
        xr.Dataset
            Dataset containing the computed flood depth array.
        """
        overview_level = 0

        if max_pixel_size > 0.0:
            overview_level = get_appropriate_overview_level(self.zb, max_pixel_size)

        if overview_level == 0:
            zb = rioxarray.open_rasterio(self.zb)
        else:
            zb = rioxarray.open_rasterio(self.zb, overview_level=overview_level)
        if "band" in zb.dims and zb.sizes["band"] == 1:
            zb = zb.squeeze(dim="band", drop=True)

        # The pixel-exact "slope" method distributes each cell's volume over
        # the pixels of that cell, so cells must not be cut by the clip box.
        # Pad the box by one coarse cell and crop the result afterwards.
        clip_bbox = bbox
        crop_back = False
        if (
            bbox is not None
            and self.method == "slope"
            and (self.volume_mode == "pixels" or self.surface_mode == "trend")
            and self.area is not None
        ):
            pad = float(np.sqrt(np.nanmax(self.area)))
            if zb.rio.crs is not None and zb.rio.crs.is_geographic:
                pad = pad / 111111.0
            clip_bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
            crop_back = True
        if clip_bbox is not None:
            zb = zb.rio.clip_box(
                minx=clip_bbox[0], miny=clip_bbox[1], maxx=clip_bbox[2], maxy=clip_bbox[3]
            )

        if overview_level == 0:
            indices = rioxarray.open_rasterio(self.indices)
        else:
            indices = rioxarray.open_rasterio(
                self.indices, overview_level=overview_level
            )
        if clip_bbox is not None:
            indices = indices.rio.clip_box(
                minx=clip_bbox[0], miny=clip_bbox[1], maxx=clip_bbox[2], maxy=clip_bbox[3]
            )
        # Structure-aware index: bands 2-4 hold the cells across a structure
        # from each pixel, used by the blend and trend surface modes.
        blocked_full = None
        if "band" in indices.dims and indices.sizes["band"] >= 4:
            blocked_full = indices.isel(band=slice(1, 4)).to_numpy()
            indices = indices.isel(band=0)
        elif "band" in indices.dims and indices.sizes["band"] == 1:
            indices = indices.squeeze(dim="band", drop=True)

        nan_val_indices = indices.attrs["_FillValue"]
        no_data_mask = indices == nan_val_indices
        indices = np.squeeze(indices.to_numpy()[:])
        indices[np.where(indices == nan_val_indices)] = 0

        if self.method == "slope":
            valid = ~no_data_mask.to_numpy() & np.isfinite(zb.to_numpy())
            if zb.rio.nodata is not None:
                valid &= zb.to_numpy() != zb.rio.nodata
            self._blocked = None
            if blocked_full is not None:
                bl = blocked_full[:, valid].astype(np.int64)
                bl[bl == int(nan_val_indices)] = -1
                self._blocked = bl if (bl >= 0).any() else None
            h = self._surface_slope(indices, zb, valid) - zb.to_numpy()[:]
            self._blocked = None
        elif isinstance(self.zs, float):
            h = np.full(zb.shape, self.zs) - zb.to_numpy()[:]
        else:
            h = self.zs[indices] - zb.to_numpy()[:]
        h[no_data_mask] = np.nan
        h[h < self.hmin] = np.nan
        h[zb.to_numpy()[:] < self.zbmin] = np.nan
        h[zb.to_numpy()[:] > self.zbmax] = np.nan

        self.ds = xr.Dataset()
        self.ds[self.data_array_name] = xr.DataArray(
            h, dims=["y", "x"], coords={"y": zb.y, "x": zb.x}
        )
        self.ds[self.data_array_name] = self.ds[self.data_array_name].rio.write_crs(
            zb.rio.crs, inplace=True
        )
        if crop_back:
            # Assigning into the existing Dataset would re-align to its larger
            # coordinates, so build a new Dataset from the cropped array.
            cropped = self.ds[self.data_array_name].rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3]
            )
            self.ds = cropped.to_dataset(name=self.data_array_name)

    def write(self, output_file: str | Path = "") -> None:
        """Write the flood map to a GeoTIFF or NetCDF file.

        Parameters
        ----------
        output_file : str | Path
            Output file path. Extension determines format: ``".tif"`` for
            COG GeoTIFF, ``".nc"`` for NetCDF, ``".png"`` for an RGBA image
            at the native raster resolution (no georeferencing).
        """
        output_file = str(output_file)
        if output_file.endswith(".nc"):
            self.ds.to_netcdf(output_file)

        elif output_file.endswith(".png"):
            rgb_da = get_rgb_data_array(
                self.ds[self.data_array_name],
                color_values=self.color_values,
                cmap=self.cmap,
                cmin=self.cmin,
                cmax=self.cmax,
                discrete_colors=self.discrete_colors,
            )
            rgba = np.moveaxis(rgb_da.to_numpy(), 0, -1)
            Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(output_file)

        elif output_file.endswith(".tif"):
            if self.cmap is not None:
                rgb_da = get_rgb_data_array(
                    self.ds[self.data_array_name],
                    color_values=self.color_values,
                    cmap=self.cmap,
                    cmin=self.cmin,
                    cmax=self.cmax,
                )

                rgb_da.rio.to_raster(
                    output_file,
                    driver="COG",
                    compress="deflate",
                    blocksize=512,
                    overview_resampling="nearest",
                )

            else:
                self.ds[self.data_array_name].rio.to_raster(
                    output_file,
                    driver="COG",
                    compress="deflate",
                    blocksize=512,
                    overview_resampling="nearest",
                )

    def map_overlay(
        self,
        file_name: str,
        xlim: list[float] | None = None,
        ylim: list[float] | None = None,
        width: int = 800,
    ) -> bool:
        """Create a PNG map overlay of the flood map in EPSG:3857.

        Parameters
        ----------
        file_name : str
            Output PNG file path.
        xlim : list[float] | None
            Longitude extent ``[lon_min, lon_max]``.
        ylim : list[float] | None
            Latitude extent ``[lat_min, lat_max]``.
        width : int
            Width in pixels for resolution calculation.

        Returns
        -------
        bool
            True on success, False on failure.
        """
        if self.ds is None:
            logger.error(
                "Dataset is not initialized. Call make() or read() before map_overlay()."
            )
            return False

        try:
            lon_min = xlim[0]
            lat_min = ylim[0]
            lon_max = xlim[1]
            lat_max = ylim[1]

            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x_min, y_min = transformer.transform(lon_min, lat_min)
            x_max, y_max = transformer.transform(lon_max, lat_max)

            dxy = (x_max - x_min) / width

            bbox = reproject_bbox(
                lon_min,
                lat_min,
                lon_max,
                lat_max,
                crs_src="EPSG:4326",
                crs_dst=self.zb.crs,
                buffer=0.05,
            )

            self.make(max_pixel_size=dxy, bbox=bbox)

            rgb_da = get_rgb_data_array(
                self.ds[self.data_array_name],
                cmap=self.cmap,
                cmin=self.cmin,
                cmax=self.cmax,
                discrete_colors=self.discrete_colors,
                color_values=self.color_values,
            )

            # Reproject straight onto the requested box: a grid anchored at its
            # top-left corner with exactly `width` columns, so the image covers
            # the extent the map stretches it over (no sub-pixel shift between
            # views). Nearest resampling keeps transparent pixels from bleeding
            # their black colour into the edges of the flooded areas.
            height = max(int(round((y_max - y_min) / dxy)), 1)
            dst_transform = Affine(dxy, 0.0, x_min, 0.0, -dxy, y_max)
            rgb_3857 = rgb_da.rio.reproject(
                "EPSG:3857",
                shape=(height, width),
                transform=dst_transform,
                resampling=Resampling.nearest,
                nodata=0,
            )

            rgba = rgb_3857.transpose("y", "x", "band").to_numpy().astype("uint8")

            Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(file_name)

            if self.discrete_colors:
                self.legend = {}
                self.legend["title"] = "Flood Depth (m)"
                self.legend["contour"] = []

                if isinstance(self.color_values, str):
                    color_values = []
                    color_values.append(
                        {"color": "lightgreen", "lower_value": 0.1, "upper_value": 0.3}
                    )
                    color_values.append(
                        {"color": "yellow", "lower_value": 0.3, "upper_value": 1.0}
                    )
                    color_values.append(
                        {"color": "#FFA500", "lower_value": 1.0, "upper_value": 2.0}
                    )
                    color_values.append({"color": "red", "lower_value": 2.0})
                else:
                    color_values = self.color_values

                for cv in color_values:
                    legend_item = {}
                    if "color" in cv:
                        legend_item["color"] = cv["color"]
                    elif "rgb" in cv:
                        r, g, b = cv["rgb"]
                        legend_item["color"] = f"#{r:02X}{g:02X}{b:02X}"
                    else:
                        raise ValueError(
                            f"Color definition must contain 'color' or 'rgb': {cv}"
                        )
                    if "upper_value" in cv and "lower_value" in cv:
                        legend_item["lower_value"] = cv["lower_value"]
                        legend_item["upper_value"] = cv["upper_value"]
                        legend_item["text"] = (
                            f"{cv['lower_value']}--{cv['upper_value']} m"
                        )
                    elif "upper_value" in cv:
                        legend_item["upper_value"] = cv["upper_value"]
                        legend_item["text"] = f"{cv['lower_value']}- m"
                    else:
                        legend_item["lower_value"] = cv["lower_value"]
                        legend_item["text"] = f"{cv['lower_value']}+ m"
                    self.legend["contour"].append(legend_item)

            else:
                self.legend = {}
                self.legend["title"] = "Flood Depth (m)"
                self.legend["cmin"] = self.cmin
                self.legend["cmax"] = self.cmax
                self.legend["cmap"] = self.cmap

            return True

        except Exception as e:
            logger.exception(e)
            return False

    def plot(
        self,
        pngfile: str,
        zoom: int | None = None,
        title: str = "Flood Depth (m)",
        color_values: list[dict] | None = None,
        cmap: str = "Blues",
        vmin: float = 0.0,
        vmax: float = 5.0,
        lon_lim: list[float] | None = None,
        lat_lim: list[float] | None = None,
        width: float = 10.0,
        background: str = "EsriWorldImagery",
    ) -> None:
        """Plot the flood map with a basemap and save to PNG.

        Parameters
        ----------
        pngfile : str
            Output PNG file path.
        zoom : int | None
            Basemap zoom level. If None, auto-detected.
        title : str
            Plot title.
        color_values : list[dict] | None
            Discrete color definitions. If a string is passed, a default
            flood depth color scheme is used.
        cmap : str
            Matplotlib colormap for continuous coloring.
        vmin : float
            Minimum value for color mapping.
        vmax : float
            Maximum value for color mapping.
        lon_lim : list[float] | None
            Longitude limits ``[lon_min, lon_max]``.
        lat_lim : list[float] | None
            Latitude limits ``[lat_min, lat_max]``.
        width : float
            Figure width in inches.
        background : str
            Basemap provider: ``"osm"`` or ``"EsriWorldImagery"``.
        """
        # plotting dependencies are optional for hydromt_sfincs
        import contextily as ctx
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch

        if lon_lim is None or lat_lim is None:
            lon_min = self.ds.x.min().to_numpy()
            lat_min = self.ds.y.min().to_numpy()
            lon_max = self.ds.x.max().to_numpy()
            lat_max = self.ds.y.max().to_numpy()
            crs = self.ds[self.data_array_name].rio.crs
            transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
            x_min, y_min = transformer.transform(lon_min, lat_min)
            x_max, y_max = transformer.transform(lon_max, lat_max)
        else:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x_min, y_min = transformer.transform(lon_lim[0], lat_lim[0])
            x_max, y_max = transformer.transform(lon_lim[1], lat_lim[1])

        da_3857 = self.ds[self.data_array_name].rio.reproject("EPSG:3857")

        if color_values is None:
            discrete_colors = False
        else:
            discrete_colors = True
            if isinstance(color_values, str):
                color_values = []
                color_values.append(
                    {"color": "lightgreen", "lower_value": 0.1, "upper_value": 0.3}
                )
                color_values.append(
                    {"color": "yellow", "lower_value": 0.3, "upper_value": 1.0}
                )
                color_values.append(
                    {"color": "#FFA500", "lower_value": 1.0, "upper_value": 2.0}
                )
                color_values.append({"color": "red", "lower_value": 2.0})

        aspect_ratio = (y_max - y_min) / (x_max - x_min)
        fig, ax = plt.subplots(figsize=(width, aspect_ratio * width))

        if discrete_colors:
            masked = da_3857.where(da_3857 >= color_values[0]["lower_value"])

            classified = xr.full_like(masked, np.nan)
            colors = []
            labels = []
            for icolor, color_value in enumerate(color_values):
                if "upper_value" in color_value:
                    lv = color_value["lower_value"]
                    uv = color_value["upper_value"]
                    classified = classified.where(
                        ~((masked > lv) & (masked <= uv)), icolor + 1
                    )
                    labels.append(f"{lv}--{uv} m")
                else:
                    lv = color_value["lower_value"]
                    classified = classified.where(~(masked > lv), icolor + 1)
                    labels.append(f">{lv} m")
                colors.append(color_value["color"])

            cmap = ListedColormap(colors)
            bounds = list(range(1, len(colors) + 2))

            norm = BoundaryNorm(bounds, cmap.N)

            classified.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

            legend_elements = []
            for i, color_value in enumerate(color_values):
                legend_elements.append(
                    Patch(facecolor=color_value["color"], label=labels[i])
                )
            plt.legend(handles=legend_elements, title="Flood Depth", loc="lower right")

        else:
            da_3857.plot(
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=True,
                cbar_kwargs={"label": "Flood Depth (m)"},
                alpha=0.75,
            )

        if background.lower() == "osm":
            if zoom is None:
                ctx.add_basemap(
                    ax, crs=da_3857.rio.crs, source=ctx.providers.OpenStreetMap.Mapnik
                )
            else:
                ctx.add_basemap(
                    ax,
                    crs=da_3857.rio.crs,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom=zoom,
                )
        else:
            if zoom is None:
                ctx.add_basemap(
                    ax, crs=da_3857.rio.crs, source=ctx.providers.Esri.WorldImagery
                )
            else:
                ctx.add_basemap(
                    ax,
                    crs=da_3857.rio.crs,
                    source=ctx.providers.Esri.WorldImagery,
                    zoom=zoom,
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_axis_off()
        plt.title(title)

        plt.tight_layout()
        plt.savefig(pngfile, dpi=300, bbox_inches="tight", pad_inches=0.1)


def get_appropriate_overview_level(
    src: rasterio.io.DatasetReader, max_pixel_size: float
) -> int:
    """Determine the appropriate rasterio overview level for a target resolution.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        An open rasterio dataset.
    max_pixel_size : float
        Maximum desired pixel size in metres.

    Returns
    -------
    int
        The overview level index (0 = native resolution).
    """
    original_resolution = src.res
    if src.crs.is_geographic:
        original_resolution = (
            original_resolution[0] * 111000,
            original_resolution[1] * 111000,
        )

    overview_levels = src.overviews(1)

    if not overview_levels:
        return 0

    resolutions = [
        (original_resolution[0] * factor, original_resolution[1] * factor)
        for factor in overview_levels
    ]

    selected_overview = 0
    for i, (x_res, y_res) in enumerate(resolutions):
        if x_res <= max_pixel_size and y_res <= max_pixel_size:
            selected_overview = i
        else:
            break

    return selected_overview


def get_rgb_data_array(
    da: xr.DataArray,
    cmap: str,
    cmin: float | None = None,
    cmax: float | None = None,
    color_values: list[dict] | None = None,
    discrete_colors: bool = False,
) -> xr.DataArray:
    """Convert an xarray DataArray to an RGBA DataArray using a colormap.

    Supports both continuous colormaps and discrete color value ranges.

    Parameters
    ----------
    da : xr.DataArray
        Input 2-D data array.
    cmap : str
        Matplotlib colormap name for continuous coloring.
    cmin : float | None
        Minimum value for normalization. Defaults to data minimum.
    cmax : float | None
        Maximum value for normalization. Defaults to data maximum.
    color_values : list[dict] | None
        Discrete color definitions with ``lower_value``, ``upper_value``,
        and ``rgb`` keys.
    discrete_colors : bool
        If True and ``color_values`` is provided, use discrete coloring
        via named color strings.

    Returns
    -------
    xr.DataArray
        RGBA DataArray with shape ``(4, height, width)`` and dtype uint8.
    """
    # matplotlib is optional for hydromt_sfincs; only needed for colouring
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    ny, nx = da.shape
    if color_values is not None:
        zz = da.to_numpy()

    if discrete_colors:
        if isinstance(color_values, str):
            color_values = []
            color_values.append(
                {"color": "lightgreen", "lower_value": 0.1, "upper_value": 0.3}
            )
            color_values.append(
                {"color": "yellow", "lower_value": 0.3, "upper_value": 1.0}
            )
            color_values.append(
                {"color": "#FFA500", "lower_value": 1.0, "upper_value": 2.0}
            )
            color_values.append({"color": "red", "lower_value": 2.0})

        rgba = np.zeros((ny, nx, 4), "uint8")
        for color_value in color_values:
            lower = color_value.get("lower_value", -np.inf)
            upper = color_value.get("upper_value", np.inf)
            inr = np.logical_and(zz >= lower, zz < upper)
            valid = np.logical_and(inr, ~np.isnan(zz))

            if "rgb" in color_value:
                rgba[valid, 0] = color_value["rgb"][0]
                rgba[valid, 1] = color_value["rgb"][1]
                rgba[valid, 2] = color_value["rgb"][2]
            elif "color" in color_value:
                color_rgba = mcolors.to_rgba(color_value["color"])
                rgba[valid, 0] = int(color_rgba[0] * 255)
                rgba[valid, 1] = int(color_rgba[1] * 255)
                rgba[valid, 2] = int(color_rgba[2] * 255)
            rgba[valid, 3] = 255

    else:
        if cmap is None:
            raise ValueError("Either color_values or cmap must be provided")

        if cmin is None:
            cmin = da.min()
        if cmax is None:
            cmax = da.max()

        if cmin == cmax:
            cmin = cmax - 1.0
            cmax = cmax + 1.0

        normed = (da - cmin) / (cmax - cmin)

        cmap_obj = plt.get_cmap(cmap)

        rgba = cmap_obj(normed)

        rgba = (rgba[:, :, :] * 255).astype("uint8")

    rgb_da = xr.DataArray(
        np.moveaxis(rgba, -1, 0),
        dims=("band", "y", "x"),
        coords={"band": [0, 1, 2, 3], "y": da.y, "x": da.x},
        attrs=da.attrs,
    )

    rgb_da.rio.write_crs(da.rio.crs, inplace=True)

    return rgb_da


def reproject_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    crs_src: str,
    crs_dst: str,
    buffer: float = 0.0,
) -> tuple[float, float, float, float]:
    """Reproject a bounding box between coordinate reference systems.

    Parameters
    ----------
    xmin : float
        Minimum x (or longitude).
    ymin : float
        Minimum y (or latitude).
    xmax : float
        Maximum x (or longitude).
    ymax : float
        Maximum y (or latitude).
    crs_src : str
        Source CRS string (e.g. ``"EPSG:4326"``).
    crs_dst : str
        Destination CRS string.
    buffer : float
        Fractional buffer to expand the bounding box before reprojection.

    Returns
    -------
    tuple[float, float, float, float]
        Reprojected bounding box ``(xmin, ymin, xmax, ymax)``.
    """
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    dx = (xmax - xmin) * buffer
    dy = (ymax - ymin) * buffer
    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy

    x0, y0 = transformer.transform(xmin, ymin)
    x1, y1 = transformer.transform(xmax, ymin)
    x2, y2 = transformer.transform(xmax, ymax)
    x3, y3 = transformer.transform(xmin, ymax)

    xs = [x0, x1, x2, x3]
    ys = [y0, y1, y2, y3]

    return min(xs), min(ys), max(xs), max(ys)


# =============================================================================
#  Structures (thin dams, weirs, flood walls): structure-aware index rasters
# =============================================================================
#
# SFINCS snaps a structure onto the grid: a face between two cells is blocked
# when the polyline crosses the straight link between the two cell centres.
# Water levels are computed with the structure on the faces, while the real
# structure cuts through cells, so a map that gives every pixel the level of
# its containing cell floods ground behind the structure and leaves ground in
# front of it dry. The functions below correct the pixel-to-cell index once,
# as a pre-processing step: a pixel whose link to its own cell centre crosses
# a structure is handed to the nearest centre it can reach without crossing
# one, and the cells across a structure from each pixel are stored in extra
# bands so the blend and trend surface modes need no geometry at map time.

#: GeoTIFF tag under which a structure-aware index COG stores its structures
STRUCTURES_TAG = "SFINCS_STRUCTURES_GEOJSON"


def structures_to_tag(lines) -> str:
    """Serialise (multi)linestrings to the GeoJSON string stored in the index COG."""
    return json.dumps(
        {"type": "GeometryCollection", "geometries": [mapping(g) for g in lines]}
    )


def structures_from_tag(text: str | None) -> list:
    """Deserialise the structures tag of an index COG (empty list if absent)."""
    if not text:
        return []
    geom = shape(json.loads(text))
    return list(getattr(geom, "geoms", [geom]))


def read_structures_from_index(index_file: str | Path) -> list:
    """Return the structures stored in a structure-aware index COG, if any."""
    with rasterio.open(index_file) as src:
        return structures_from_tag(src.tags().get(STRUCTURES_TAG))


def _segments_cross(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether each segment ``p1[i]-p2[i]`` crosses any segment ``a[j]-b[j]``.

    Proper crossings only (touching end points and collinear overlaps do
    not count), which is the same test SFINCS applies between cell centres.

    Parameters
    ----------
    p1, p2 : np.ndarray (n, 2)
    a, b : np.ndarray (m, 2)

    Returns
    -------
    np.ndarray (n,) bool
    """

    def orient(o, p, q):
        return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (
            q[..., 0] - o[..., 0]
        )

    P1 = p1[:, None, :]
    P2 = p2[:, None, :]
    A = a[None, :, :]
    B = b[None, :, :]
    d1 = orient(A, B, P1)
    d2 = orient(A, B, P2)
    d3 = orient(P1, P2, A)
    d4 = orient(P1, P2, B)
    cross = (d1 * d2 < 0) & (d3 * d4 < 0)
    return cross.any(axis=1)


def _line_segments(lines) -> tuple[np.ndarray, np.ndarray]:
    """Start and end points of all segments of a list of (multi)linestrings."""
    a, b = [], []
    for geom in lines:
        parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
        for part in parts:
            c = np.asarray(part.coords, dtype=np.float64)[:, :2]
            if len(c) < 2:
                continue
            a.append(c[:-1])
            b.append(c[1:])
    if not a:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.vstack(a), np.vstack(b)


def reassign_index_for_structures(
    indices: np.ndarray,
    transform,
    lines,
    xc: np.ndarray,
    yc: np.ndarray,
    nodata: int,
    search_radius: float | None = None,
    max_neighbours: int = 8,
    chunk: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reassign pixels that a structure separates from their own cell centre.

    Parameters
    ----------
    indices : np.ndarray (ny, nx)
        Cell index per pixel (0-based), ``nodata`` outside the model.
    transform : affine.Affine
        Raster transform of ``indices``.
    lines : iterable of shapely LineString / MultiLineString
        Structures in the raster CRS.
    xc, yc : np.ndarray (ncell,)
        Cell centre coordinates in the raster CRS.
    nodata : int
        Nodata value of ``indices``.
    search_radius : float, optional
        Only pixels within this distance of a structure are tested. Default:
        1.5 times the largest distance between neighbouring cell centres,
        which covers every pixel that can be in a crossed cell.
    max_neighbours : int
        Number of nearest cell centres tried when looking for a reachable
        cell.
    chunk : int
        Pixels per processing block.

    Returns
    -------
    new_indices : np.ndarray (ny, nx)
        Corrected index raster (copy).
    changed : np.ndarray (ny, nx) bool
        Pixels whose cell changed.
    """
    a, b = _line_segments(lines)
    out = indices.copy()
    changed = np.zeros(indices.shape, dtype=bool)
    if a.shape[0] == 0:
        return out, changed

    xc = np.asarray(xc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)
    centres = np.column_stack([xc, yc])
    tree = cKDTree(centres)
    if search_radius is None:
        d, _ = tree.query(centres, k=2)
        search_radius = 1.5 * float(d[:, 1].max())

    # Candidate pixels: within the search radius of any structure
    geoms = [LineString(np.vstack([a[i], b[i]])).buffer(search_radius) for i in range(a.shape[0])]
    cand = features.rasterize(
        [(g, 1) for g in geoms],
        out_shape=indices.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    cand &= indices != nodata
    rows, cols = np.nonzero(cand)
    if rows.size == 0:
        return out, changed

    px = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    py = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    own = indices[rows, cols].astype(np.int64)

    for i0 in range(0, rows.size, chunk):
        i1 = min(i0 + chunk, rows.size)
        p = np.column_stack([px[i0:i1], py[i0:i1]])
        c_own = centres[own[i0:i1]]
        cut = _segments_cross(p, c_own, a, b)
        if not cut.any():
            continue
        pc = p[cut]
        k = min(max_neighbours, centres.shape[0])
        _, nn = tree.query(pc, k=k)
        nn = np.atleast_2d(nn)
        new = own[i0:i1][cut].copy()
        done = np.zeros(pc.shape[0], dtype=bool)
        for j in range(k):
            cj = centres[nn[:, j]]
            ok = ~done & ~_segments_cross(pc, cj, a, b)
            new[ok] = nn[ok, j]
            done |= ok
            if done.all():
                break
        sel = np.arange(i0, i1)[cut]
        r, c = rows[sel], cols[sel]
        out[r, c] = new
        changed[r, c] = new != own[i0:i1][cut]
    return out, changed


def blocked_cells_for_pixels(
    indices: np.ndarray,
    transform,
    lines,
    xc: np.ndarray,
    yc: np.ndarray,
    nodata: int,
    search_radius: float | None = None,
    chunk: int = 500_000,
) -> np.ndarray:
    """Cells across a structure from each pixel, for the surface interpolation.

    The blend and trend surface modes of ``FloodMap`` interpolate between the
    three cell centres of the Delaunay triangle around a pixel. This returns,
    per pixel, the ids of those vertex cells whose link to the pixel crosses
    a structure, so the interpolation can leave them out without any
    geometry at map time.

    Returns
    -------
    np.ndarray (3, ny, nx) of the index dtype
        Blocked cell ids per pixel, ``nodata`` where none.
    """
    a, b = _line_segments(lines)
    out = np.full((3,) + indices.shape, nodata, dtype=indices.dtype)
    if a.shape[0] == 0:
        return out
    centres = np.column_stack([np.asarray(xc, float), np.asarray(yc, float)])
    tree = cKDTree(centres)
    if search_radius is None:
        d, _ = tree.query(centres, k=2)
        search_radius = 1.5 * float(d[:, 1].max())
    geoms = [LineString(np.vstack([a[i], b[i]])).buffer(search_radius) for i in range(a.shape[0])]
    cand = features.rasterize(
        [(g, 1) for g in geoms], out_shape=indices.shape, transform=transform, fill=0, dtype="uint8", all_touched=True
    ).astype(bool)
    cand &= indices != nodata
    rows, cols = np.nonzero(cand)
    if rows.size == 0:
        return out
    tri = Delaunay(centres)
    px = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    py = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    for i0 in range(0, rows.size, chunk):
        i1 = min(i0 + chunk, rows.size)
        p = np.column_stack([px[i0:i1], py[i0:i1]])
        s = tri.find_simplex(p)
        inside = s >= 0
        if not inside.any():
            continue
        verts = tri.simplices[s[inside]]
        pi = p[inside]
        r = rows[i0:i1][inside]
        c = cols[i0:i1][inside]
        for j in range(3):
            cross = _segments_cross(pi, centres[verts[:, j]], a, b)
            out[j, r[cross], c[cross]] = verts[cross, j]
    return out


def apply_structures_to_index_cog(
    index_file: str | Path,
    out_file: str | Path,
    lines,
    xc: np.ndarray,
    yc: np.ndarray,
    **kwargs,
) -> int:
    """Write a structure-aware copy of an index COG.

    The output has four bands: band 1 is the corrected cell index, bands 2-4
    hold, per pixel, the ids of surrounding cells that lie across a structure
    (``nodata`` where none), see :func:`blocked_cells_for_pixels`. A
    ``FloodMap`` that loads such an index uses both without being given the
    structures again: band 1 for the level of every pixel, bands 2-4 to keep
    the blend and trend surface modes from interpolating across a structure.
    The structures themselves are stored as a GeoJSON tag for reference.

    Parameters
    ----------
    index_file : str | Path
        Existing cell index COG (as made by ``make_index_cog``).
    out_file : str | Path
        Output COG.
    lines : iterable of shapely LineString / MultiLineString
        Structures in the raster CRS (thin dams, weirs, flood walls).
    xc, yc : np.ndarray
        Cell centre coordinates in the raster CRS, indexed by cell index.
    **kwargs
        Passed to :func:`reassign_index_for_structures`.

    Returns
    -------
    int
        Number of pixels reassigned.
    """
    lines = list(lines)
    with rasterio.open(index_file) as src:
        indices = src.read(1)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata if src.nodata is not None else 2147483647
    new, changed = reassign_index_for_structures(
        indices, transform, lines, xc, yc, int(nodata), **kwargs
    )
    blocked = blocked_cells_for_pixels(
        indices, transform, lines, xc, yc, int(nodata), search_radius=kwargs.get("search_radius")
    )
    profile.update(
        driver="COG",
        compress="deflate",
        blocksize=512,
        overview_resampling="nearest",
        nodata=nodata,
        count=4,
    )
    for key in ("blockxsize", "blockysize", "tiled", "interleave"):
        profile.pop(key, None)
    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(new, 1)
        for j in range(3):
            dst.write(blocked[j], j + 2)
        dst.set_band_description(1, "cell index")
        for j in range(3):
            dst.set_band_description(j + 2, f"blocked cell {j + 1}")
        dst.update_tags(**{STRUCTURES_TAG: structures_to_tag(lines)})
    return int(changed.sum())
