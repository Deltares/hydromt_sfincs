"""Unit tests for downscaling pre-step helpers (regular + quadtree grids).

Covers ``adjust_zsmax_dilation`` and ``adjust_zsmax_energyhead`` from
``hydromt_sfincs.workflows.downscaling`` — the method-agnostic pre-steps
that operate on a SFINCS zsmax field before any high-resolution
downscaling happens.

Invariants under test
---------------------
* wet-cell set is preserved (no new wet cells, no spurious drying)
* result is monotonically >= the input on every wet cell
* dry cells stay NaN
"""

import numpy as np
import xarray as xr
import xugrid as xu

from hydromt_sfincs.workflows.downscaling import (
    adjust_zsmax_energyhead,
    adjust_zsmax_dilation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regular_zsmax(shape=(8, 8), wet_value=1.0, dry_border=2):
    """Synthetic regular-grid zsmax: wet interior, NaN border."""
    vals = np.full(shape, np.nan, dtype=np.float32)
    vals[dry_border:-dry_border, dry_border:-dry_border] = wet_value
    return xr.DataArray(vals, dims=("y", "x"), name="zsmax")


# ---------------------------------------------------------------------------
# adjust_zsmax_dilation — regular grid
# ---------------------------------------------------------------------------


def test_adjust_zsmax_dilation_regular_factor_zero_is_noop():
    zs = _make_regular_zsmax()
    out = adjust_zsmax_dilation(zs, factor=0.0)
    # Identical values (NaN-aware)
    assert np.array_equal(np.isnan(out.values), np.isnan(zs.values))
    np.testing.assert_array_equal(
        out.values[~np.isnan(out.values)],
        zs.values[~np.isnan(zs.values)],
    )


def test_adjust_zsmax_dilation_regular_preserves_wet_set():
    """No new wet cells, no drying — the key safeguard."""
    zs = _make_regular_zsmax()
    out = adjust_zsmax_dilation(zs, factor=0.5)
    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(zs.values))


def test_adjust_zsmax_dilation_regular_lifts_low_neighbour():
    """A low wet cell next to a high wet cell should rise to the high value."""
    vals = np.full((5, 5), np.nan, dtype=np.float32)
    vals[2, 1] = 1.0  # low
    vals[2, 2] = 5.0  # high — edge-neighbour of the low cell
    zs = xr.DataArray(vals, dims=("y", "x"))

    out = adjust_zsmax_dilation(zs, factor=0.5)  # reaches edge neighbours

    assert out.values[2, 1] == 5.0
    assert out.values[2, 2] == 5.0
    # Diagonal/distant cells stay NaN
    assert np.isnan(out.values[0, 0])


# ---------------------------------------------------------------------------
# adjust_zsmax_dilation — quadtree grid
# ---------------------------------------------------------------------------


def test_adjust_zsmax_dilation_quadtree_preserves_wet_set(quadtree_model):
    """Same invariants on a real quadtree mesh from the test fixture."""
    grid = quadtree_model.quadtree_grid.data.ugrid.grid
    n_faces = grid.n_face

    # Synthetic zsmax: half the cells wet at 1 m, half dry (NaN)
    vals = np.full(n_faces, np.nan, dtype=np.float32)
    vals[: n_faces // 2] = 1.0
    zs = xu.UgridDataArray(
        xr.DataArray(vals, dims=(grid.face_dimension,), name="zsmax"),
        grid=grid,
    )

    out = adjust_zsmax_dilation(zs, factor=0.5)

    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(zs.values))
    # Monotone lift on wet cells
    wet = ~np.isnan(zs.values)
    assert np.all(out.values[wet] >= zs.values[wet] - 1e-9)


# ---------------------------------------------------------------------------
# adjust_zsmax_energyhead
# ---------------------------------------------------------------------------


def test_adjust_zsmax_energyhead_below_threshold_is_noop():
    """Cells with q < q_threshold keep their original zsmax."""
    zs_vals = np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float32)
    q_vals = np.full((2, 2), 0.001, dtype=np.float32)  # below default 0.01
    zs = xr.DataArray(zs_vals, dims=("y", "x"))
    q = xr.DataArray(q_vals, dims=("y", "x"))

    out = adjust_zsmax_energyhead(zs, q, hmin=0.05)

    wet = ~np.isnan(zs_vals)
    np.testing.assert_allclose(out.values[wet], zs_vals[wet])
    assert np.isnan(out.values[1, 0])


def test_adjust_zsmax_energyhead_lifts_with_velocity():
    """v²/(2g) lift = 0.5 * (q/h)² / g for q above threshold."""
    zs = xr.DataArray(np.array([[2.0]]), dims=("y", "x"))  # 2 m WSE
    zb = xr.DataArray(np.array([[0.0]]), dims=("y", "x"))  # 0 m bed → h=2 m
    q = xr.DataArray(np.array([[4.0]]), dims=("y", "x"))  # 4 m²/s

    out = adjust_zsmax_energyhead(zs, q, zb=zb, hmin=0.05, q_threshold=0.01)

    # v = q/h = 2 m/s ; vel_head = v²/(2g) = 4/(2*9.81) ≈ 0.2039 m
    expected = 2.0 + 0.5 * (4.0 / 2.0) ** 2 / 9.81
    np.testing.assert_allclose(out.values[0, 0], expected, rtol=1e-6)


def test_adjust_zsmax_energyhead_preserves_wet_set():
    """NaN cells stay NaN, no new wet cells appear."""
    zs_vals = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
    q_vals = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    zs = xr.DataArray(zs_vals, dims=("y", "x"))
    q = xr.DataArray(q_vals, dims=("y", "x"))

    out = adjust_zsmax_energyhead(zs, q, hmin=0.05)

    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(zs_vals))
    # Result >= input on wet cells (velocity head is non-negative)
    wet = ~np.isnan(zs_vals)
    assert np.all(out.values[wet] >= zs_vals[wet] - 1e-9)


def test_adjust_zsmax_energyhead_quadtree_preserves_wet_set(quadtree_model):
    """Energy-head pre-step on a real quadtree mesh: wet-set preserved, monotone
    lift, and a high-discharge cell is actually raised."""
    grid = quadtree_model.quadtree_grid.data.ugrid.grid
    n_faces = grid.n_face

    zs_vals = np.full(n_faces, np.nan, dtype=np.float32)
    zs_vals[: n_faces // 2] = 2.0  # wet half at 2 m WSE
    zb_vals = np.zeros(n_faces, dtype=np.float32)  # bed at 0 -> h = 2 m
    q_vals = np.zeros(n_faces, dtype=np.float32)
    q_vals[: n_faces // 4] = 4.0  # strong unit discharge on part of the wet cells

    def _uda(vals):
        return xu.UgridDataArray(
            xr.DataArray(vals, dims=(grid.face_dimension,)), grid=grid
        )

    zs, q, zb = _uda(zs_vals), _uda(q_vals), _uda(zb_vals)
    out = adjust_zsmax_energyhead(zs, q, zb=zb, hmin=0.05, q_threshold=0.01)

    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(zs_vals))
    wet = ~np.isnan(zs_vals)
    assert np.all(out.values[wet] >= zs_vals[wet] - 1e-9)  # monotone
    # cells with q=4, h=2 -> v=2 -> +v^2/2g ~ 0.204 m; must be strictly raised
    lifted = out.values[: n_faces // 4]
    np.testing.assert_allclose(lifted, 2.0 + 0.5 * (4.0 / 2.0) ** 2 / 9.81, rtol=1e-5)


# ---------------------------------------------------------------------------
# downscale_floodmap — regular-grid bilinear reuses the reproject engine
# ---------------------------------------------------------------------------

from hydromt_sfincs.workflows.downscaling import downscale_floodmap


def _regular_zs_and_dep():
    """Coarse 2x2 zsmax (WSE rising west->east) + fine 4x4 flat-bed DEM."""
    zs = xr.DataArray(
        np.array([[1.0, 3.0], [1.0, 3.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [150.0, 50.0], "x": [50.0, 150.0]},
        name="zsmax",
    )
    zs.raster.set_crs(32633)
    dep = xr.DataArray(
        np.zeros((4, 4), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [175.0, 125.0, 75.0, 25.0], "x": [25.0, 75.0, 125.0, 175.0]},
        name="dep",
    )
    dep.raster.set_crs(32633)
    return zs, dep


def test_downscale_floodmap_bilinear_regular_uses_reproject():
    """Regular bilinear interpolates between cell centres; nearest is blocky."""
    zs, dep = _regular_zs_and_dep()
    hmax_bil = downscale_floodmap(zs, dep, reproj_method="bilinear")
    hmax_near = downscale_floodmap(zs, dep, reproj_method="nearest")

    # bilinear must differ from nearest (otherwise it isn't interpolating)
    assert not np.allclose(hmax_bil.values, hmax_near.values, equal_nan=True)
    # interpolated depths (flat bed) stay within the source WSE range [1, 3]
    interior = hmax_bil.values[~np.isnan(hmax_bil.values)]
    assert interior.min() >= 1.0 - 1e-6
    assert interior.max() <= 3.0 + 1e-6


# ---------------------------------------------------------------------------
# downscale_floodmap — pre-step kwargs removed (pure downscaler)
# ---------------------------------------------------------------------------

import inspect


def test_downscale_floodmap_drops_prestep_kwargs():
    """The pre-step kwargs are gone — adjustments are standalone pre-steps now."""
    params = inspect.signature(downscale_floodmap).parameters
    for removed in (
        "dilation",
        "energy_flux",
        "qmax",
        "zb",
        "q_threshold",
        "q_scale",
        "method",  # the ambiguous method= param is gone (clean break)
    ):
        assert removed not in params, f"{removed} should no longer be a parameter"
    # the downscaler is parameterised by an interpolation + a subtract flag
    for present in ("reproj_method", "subtract_dem"):
        assert present in params, f"{present} should be a parameter"


# ---------------------------------------------------------------------------
# downscale_floodmap — method="raw" on a regular grid (no index COG)
# ---------------------------------------------------------------------------

import rasterio
from rasterio.transform import from_origin


def test_downscale_floodmap_raw_regular_no_index(tmp_path):
    """Regular 'raw' paints parent-cell WSE onto the DEM (no DEM subtraction)."""
    zs = xr.DataArray(
        np.full((2, 2), 2.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [150.0, 50.0], "x": [50.0, 150.0]},
        name="zsmax",
    )
    zs.raster.set_crs(32633)

    dep_fn = tmp_path / "dep.tif"
    with rasterio.open(
        dep_fn,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(0.0, 200.0, 50.0, 50.0),
        nodata=np.nan,
    ) as dst:
        dst.write(np.zeros((4, 4), dtype=np.float32), 1)

    zsmap_fn = tmp_path / "zsmap.tif"
    da = downscale_floodmap(
        zs,
        str(dep_fn),
        reproj_method="nearest",
        subtract_dem=False,
        zsmap_fn=str(zsmap_fn),
    )

    # the method returns the written product (raw -> water level)
    assert da is not None
    da_wet = da.values[~np.isnan(da.values)]
    assert da_wet.size > 0
    np.testing.assert_allclose(da_wet, 2.0)

    with rasterio.open(zsmap_fn) as src:
        out = src.read(1)
    wet = out[~np.isnan(out)]
    assert wet.size > 0
    np.testing.assert_allclose(wet, 2.0)


# ---------------------------------------------------------------------------
# _stream_blocks — shared block-window iterator
# ---------------------------------------------------------------------------

from hydromt_sfincs.workflows.downscaling import _stream_blocks


def test_stream_blocks_tiles_and_covers():
    """Every pixel is covered exactly once; windows fit inside the grid."""
    width, height, nrmax = 10, 7, 4
    covered = np.zeros((height, width), dtype=int)
    n = 0
    for window, bm0, bm1, bn0, bn1 in _stream_blocks(width, height, nrmax):
        assert 0 <= bm0 < bm1 <= width
        assert 0 <= bn0 < bn1 <= height
        covered[bn0:bn1, bm0:bm1] += 1
        n += 1
    assert n == 3 * 2  # ceil(10/4)=3 cols x ceil(7/4)=2 rows
    assert np.all(covered == 1)


def test_stream_blocks_merge_singletons():
    """A trailing 1-pixel column is merged into the previous block."""
    # width 9 with nrmax 4 -> 9 % 4 == 1 -> last 1-px col merged
    widths = [
        bm1 - bm0
        for _, bm0, bm1, _, _ in _stream_blocks(9, 4, 4, merge_singletons=True)
    ]
    assert 1 not in widths  # no degenerate 1-px tile
    # cols: [0:4], [4:9] -> widths 4 and 5
    assert sorted(set(widths)) == [4, 5]


def test_downscale_floodmap_constant_regular_file_blocks(tmp_path):
    """Regular 'constant' file path streams DEM blocks (with a merged 1-px col)."""
    zs = xr.DataArray(
        np.full((2, 2), 3.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [150.0, 50.0], "x": [50.0, 150.0]},
        name="zsmax",
    )
    zs.raster.set_crs(32633)

    # 5x5 DEM at 40 m, flat bed 0 -> hmax = 3.0 everywhere wet
    dep_fn = tmp_path / "dep.tif"
    with rasterio.open(
        dep_fn,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(0.0, 200.0, 40.0, 40.0),
        nodata=np.nan,
    ) as dst:
        dst.write(np.zeros((5, 5), dtype=np.float32), 1)

    floodmap_fn = tmp_path / "hmax.tif"
    # nrmax=2 over width 5 forces multiple blocks and a merged trailing column
    hmax = downscale_floodmap(
        zs, str(dep_fn), reproj_method="nearest", floodmap_fn=str(floodmap_fn), nrmax=2
    )

    # the method returns the written product (constant -> hmax)
    assert hmax is not None
    hmax_wet = hmax.values[~np.isnan(hmax.values)]
    assert hmax_wet.size > 0
    np.testing.assert_allclose(hmax_wet, 3.0)

    with rasterio.open(floodmap_fn) as src:
        out = src.read(1)
    wet = out[~np.isnan(out)]
    assert wet.size > 0
    np.testing.assert_allclose(wet, 3.0)


def test_downscale_floodmap_da_bilinear_ignores_index_lookup():
    """With an index COG, bilinear must still interpolate (not fall back to the
    nearest index lookup that constant/raw use)."""
    from hydromt_sfincs.workflows.downscaling import _downscale_floodmap_da

    zs, dep = _regular_zs_and_dep()  # 2x2 zsmax [[1,3],[1,3]], 4x4 flat-bed DEM
    # index COG: each fine pixel -> flat index of its containing 2x2 cell
    idx_vals = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int32
    )
    idx = xr.DataArray(
        idx_vals, dims=("y", "x"), coords={"y": dep.y, "x": dep.x}, name="indices"
    )
    idx.raster.set_nodata(-1)
    idx.raster.set_crs(32633)

    hmax_near = _downscale_floodmap_da(zs, dep, indices=idx, reproj_method="nearest")
    hmax_bil = _downscale_floodmap_da(zs, dep, indices=idx, reproj_method="bilinear")

    # nearest (index lookup) is blocky; bilinear must differ even WITH an index
    assert not np.allclose(hmax_near.values, hmax_bil.values, equal_nan=True)
    bil = hmax_bil.values[~np.isnan(hmax_bil.values)]
    assert bil.min() >= 1.0 - 1e-6 and bil.max() <= 3.0 + 1e-6


def test_downscale_floodmap_raw_regular_index_fortran_order(tmp_path):
    """raw + regular grid + index COG paints each pixel's CONTAINING cell value,
    using the SFINCS Fortran-order flat-index convention (iind*nmax + jind).
    A non-square grid would scramble under a C-order flatten."""
    # non-square 3x2 regular zsmax (nmax=3, mmax=2), distinct value per cell
    zs_vals = np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float32)
    zs = xr.DataArray(
        zs_vals,
        dims=("y", "x"),
        coords={"y": [250.0, 150.0, 50.0], "x": [50.0, 150.0]},
        name="zsmax",
    )
    zs.raster.set_crs(32633)

    transform = from_origin(0.0, 300.0, 100.0, 100.0)  # 3 rows x 2 cols, 100 m
    dep_fn = tmp_path / "dep.tif"
    with rasterio.open(
        dep_fn,
        "w",
        driver="GTiff",
        height=3,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(np.zeros((3, 2), dtype=np.float32), 1)

    # index COG matching RegularGrid.get_indices_at_points: iind = floor(x/dx)
    # (column), jind = floor(y/dy) (row, ASCENDING in +y), ind = iind*nmax+jind
    # (nmax=3).  DEM pixel rows run north->south (y = 250, 150, 50) so the
    # physical row index jind = 2, 1, 0 top-to-bottom:  ind[r,c] = c*3 + (2 - r).
    # A north-up array flatten (or C-order) would scramble this on a non-square
    # grid; _canonical_cellfield + Fortran-order flatten must reproduce it.
    idx_vals = np.array([[2, 5], [1, 4], [0, 3]], dtype=np.uint32)
    idx_fn = tmp_path / "idx.tif"
    with rasterio.open(
        idx_fn,
        "w",
        driver="GTiff",
        height=3,
        width=2,
        count=1,
        dtype="uint32",
        crs="EPSG:32633",
        transform=transform,
        nodata=2147483647,
    ) as dst:
        dst.write(idx_vals, 1)

    zsmap_fn = tmp_path / "zsmap.tif"
    da = downscale_floodmap(
        zs,
        str(dep_fn),
        reproj_method="nearest",
        subtract_dem=False,
        zsmap_fn=str(zsmap_fn),
        indices=str(idx_fn),
    )
    # each pixel must get its own cell's WSE; a wrong flatten order scrambles this
    np.testing.assert_array_equal(da.values, zs_vals)


def test_downscale_floodmap_da_bilinear_index_band_dim():
    """bilinear + a band-dimmed index (1, ny, nx) (rioxarray's shape) must be
    squeezed, not crash on hmax.where()."""
    from hydromt_sfincs.workflows.downscaling import _downscale_floodmap_da

    zs, dep = _regular_zs_and_dep()  # 2x2 zsmax, 4x4 dep
    idx2d = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int32
    )
    idx = xr.DataArray(
        idx2d[np.newaxis, ...],
        dims=("band", "y", "x"),
        coords={"band": [1], "y": dep.y, "x": dep.x},
        name="indices",
    )
    idx.raster.set_nodata(-1)
    idx.raster.set_crs(32633)

    hmax = _downscale_floodmap_da(zs, dep, indices=idx, reproj_method="bilinear")
    assert tuple(hmax.shape) == tuple(dep.shape)


# ---------------------------------------------------------------------------
# _downscale_floodmap_da — subtract_dem flag (raw water level vs flood depth)
# ---------------------------------------------------------------------------


def test_downscale_floodmap_da_subtract_dem_false_returns_wse():
    """subtract_dem=False returns the interpolated water level (no DEM subtraction,
    no hmin masking); subtract_dem=True returns depth = WSE - DEM."""
    from hydromt_sfincs.workflows.downscaling import _downscale_floodmap_da

    zs, dep = _regular_zs_and_dep()  # WSE [[1,3],[1,3]], flat bed 0
    dep = dep + 0.5  # non-zero bed so WSE (1..3) and depth (0.5..2.5) differ
    dep.raster.set_crs(32633)

    wse = _downscale_floodmap_da(zs, dep, reproj_method="nearest", subtract_dem=False)
    depth = _downscale_floodmap_da(zs, dep, reproj_method="nearest", subtract_dem=True)

    w = wse.values[~np.isnan(wse.values)]
    d = depth.values[~np.isnan(depth.values)]
    assert w.min() >= 1.0 - 1e-6 and w.max() <= 3.0 + 1e-6  # raw WSE range
    assert d.max() <= 2.5 + 1e-6  # depth = WSE - 0.5
    # water level is strictly above depth everywhere (bed is 0.5 m)
    assert np.allclose(w, d + 0.5, equal_nan=True)


def test_downscale_floodmap_da_return_zs_gives_both():
    """return_zs=True yields (product, interpolated_wse) so a caller can write
    both a depth and a water-level raster from a single interpolation."""
    from hydromt_sfincs.workflows.downscaling import _downscale_floodmap_da

    zs, dep = _regular_zs_and_dep()
    dep = dep + 0.5
    dep.raster.set_crs(32633)

    hmax, zs_on_dep = _downscale_floodmap_da(
        zs, dep, reproj_method="nearest", subtract_dem=True, return_zs=True
    )
    # zs_on_dep is the water level; hmax = zs_on_dep - dep on wet pixels
    wet = ~np.isnan(hmax.values)
    np.testing.assert_allclose(
        hmax.values[wet], (zs_on_dep.values - dep.values)[wet], rtol=1e-5
    )


# ---------------------------------------------------------------------------
# _downscale_floodmap_da — quadtree bilinear (scattered interpolation + CRS)
# ---------------------------------------------------------------------------


def _quadtree_zs_linear(grid):
    """WSE rising linearly west->east over the quadtree faces (0..1)."""
    fx, _ = grid.face_coordinates.T
    zs_vals = ((fx - fx.min()) / (fx.max() - fx.min())).astype("float32")
    return xu.UgridDataArray(
        xr.DataArray(zs_vals, dims=(grid.face_dimension,), name="zsmax"), grid=grid
    )


def _dep_over_grid(grid, n=24, bed=0.0, crs=None):
    """Flat-bed DEM covering the grid extent, axis-aligned in ``crs``."""
    from pyproj import Transformer

    fx, fy = grid.face_coordinates.T
    xmin, xmax = float(fx.min()), float(fx.max())
    ymin, ymax = float(fy.min()), float(fy.max())
    if crs is not None and crs != grid.crs:
        tf = Transformer.from_crs(grid.crs, crs, always_xy=True)
        (xmin, xmax), (ymin, ymax) = tf.transform([xmin, xmax], [ymin, ymax])
    dx, dy = (xmax - xmin) / n, (ymax - ymin) / n
    x = np.linspace(xmin + dx / 2, xmax - dx / 2, n)
    y = np.linspace(ymax - dy / 2, ymin + dy / 2, n)  # north-up (descending)
    dep = xr.DataArray(
        np.full((n, n), bed, dtype="float32"),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="dep",
    )
    dep.raster.set_crs(crs if crs is not None else grid.crs)
    return dep


def test_downscale_floodmap_da_quadtree_bilinear_interpolates(quadtree_model):
    """On a quadtree, bilinear scatters over cell centres (not blocky nearest)."""
    from hydromt_sfincs.workflows.downscaling import _downscale_floodmap_da

    grid = quadtree_model.quadtree_grid.data.ugrid.grid
    zs = _quadtree_zs_linear(grid)
    dep = _dep_over_grid(grid, n=24)

    hmax_bil = _downscale_floodmap_da(zs, dep, reproj_method="bilinear")
    hmax_near = _downscale_floodmap_da(zs, dep, reproj_method="nearest")

    assert np.isfinite(hmax_bil.values).any()
    both = np.isfinite(hmax_bil.values) & np.isfinite(hmax_near.values)
    assert both.any()
    # bilinear must interpolate, i.e. differ from the blocky nearest result
    assert not np.allclose(hmax_bil.values[both], hmax_near.values[both])
    # linear WSE 0..1 on a flat bed -> depths stay in range
    v = hmax_bil.values[np.isfinite(hmax_bil.values)]
    assert v.min() >= -1e-3 and v.max() <= 1.0 + 1e-3


def test_downscale_floodmap_da_quadtree_bilinear_crs_mismatch(quadtree_model):
    """DEM in a different CRS than the model: pixel centres must be transformed
    to the model CRS before querying the scattered interpolator (else silently
    all-NaN)."""
    from hydromt_sfincs.workflows.downscaling import _downscale_floodmap_da

    grid = quadtree_model.quadtree_grid.data.ugrid.grid  # EPSG:32633
    zs = _quadtree_zs_linear(grid)
    dep = _dep_over_grid(grid, n=20, crs=3857)  # DEM in web-mercator

    hmax = _downscale_floodmap_da(zs, dep, reproj_method="bilinear")

    # with the CRS transform the interior is wet; without it every query lands
    # far outside the interpolant's convex hull -> all NaN
    assert np.isfinite(hmax.values).any()
    v = hmax.values[np.isfinite(hmax.values)]
    assert v.min() >= -1e-3 and v.max() <= 1.0 + 1e-2


# ---------------------------------------------------------------------------
# downscale_floodmap — quadtree file-based streaming (bilinear + raw)
# ---------------------------------------------------------------------------


def _write_dep_tif(grid, path, n=40, bed=0.0):
    """Flat-bed DEM GeoTIFF covering the quadtree extent (model CRS)."""
    fx, fy = grid.face_coordinates.T
    xmin, xmax = float(fx.min()), float(fx.max())
    ymin, ymax = float(fy.min()), float(fy.max())
    dx, dy = (xmax - xmin) / n, (ymax - ymin) / n
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(xmin, ymax, dx, dy),
        nodata=np.nan,
    ) as dst:
        dst.write(np.full((n, n), bed, dtype="float32"), 1)


def test_downscale_floodmap_quadtree_file_bilinear(quadtree_model, tmp_path):
    """Quadtree bilinear via the file streamer (blocks + one shared interpolant)."""
    grid = quadtree_model.quadtree_grid.data.ugrid.grid
    zs = _quadtree_zs_linear(grid)
    dep_fn = tmp_path / "dep.tif"
    _write_dep_tif(grid, str(dep_fn), n=40)

    fm = tmp_path / "hmax.tif"
    # nrmax < n forces several blocks -> exercises the reused interpolant
    da = downscale_floodmap(
        zs, str(dep_fn), reproj_method="bilinear", floodmap_fn=str(fm), nrmax=16
    )
    assert da is not None
    with rasterio.open(fm) as src:
        out = src.read(1)
    wet = out[np.isfinite(out)]
    assert wet.size > 0
    assert wet.min() >= -1e-3 and wet.max() <= 1.0 + 1e-2


def test_downscale_floodmap_quadtree_file_raw(quadtree_model, tmp_path):
    """Quadtree raw water level via the file streamer (subtract_dem=False)."""
    grid = quadtree_model.quadtree_grid.data.ugrid.grid
    zs = _quadtree_zs_linear(grid)
    dep_fn = tmp_path / "dep.tif"
    _write_dep_tif(grid, str(dep_fn), n=40)

    zsmap = tmp_path / "zsmap.tif"
    da = downscale_floodmap(
        zs,
        str(dep_fn),
        reproj_method="nearest",
        subtract_dem=False,
        zsmap_fn=str(zsmap),
        nrmax=16,
    )
    assert da is not None
    with rasterio.open(zsmap) as src:
        out = src.read(1)
    wet = out[np.isfinite(out)]
    assert wet.size > 0
    assert wet.min() >= -1e-3 and wet.max() <= 1.0 + 1e-2
