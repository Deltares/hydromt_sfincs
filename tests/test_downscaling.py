"""Unit tests for downscaling pre-step helpers (regular + quadtree grids).

Covers ``dilate_zsmax`` and ``apply_energy_head`` from
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
import rasterio
import xarray as xr
import xugrid as xu
from rasterio.transform import from_origin

from hydromt_sfincs.workflows.downscaling import (
    apply_energy_head,
    dilate_zsmax,
    downscale_velocity,
    smooth_cell_field,
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
# dilate_zsmax — regular grid
# ---------------------------------------------------------------------------


def test_dilate_zsmax_regular_factor_zero_is_noop():
    zs = _make_regular_zsmax()
    out = dilate_zsmax(zs, factor=0.0)
    # Identical values (NaN-aware)
    assert np.array_equal(np.isnan(out.values), np.isnan(zs.values))
    np.testing.assert_array_equal(
        out.values[~np.isnan(out.values)],
        zs.values[~np.isnan(zs.values)],
    )


def test_dilate_zsmax_regular_preserves_wet_set():
    """No new wet cells, no drying — the key safeguard."""
    zs = _make_regular_zsmax()
    out = dilate_zsmax(zs, factor=0.5)
    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(zs.values))


def test_dilate_zsmax_regular_lifts_low_neighbour():
    """A low wet cell next to a high wet cell should rise to the high value."""
    vals = np.full((5, 5), np.nan, dtype=np.float32)
    vals[2, 1] = 1.0  # low
    vals[2, 2] = 5.0  # high — edge-neighbour of the low cell
    zs = xr.DataArray(vals, dims=("y", "x"))

    out = dilate_zsmax(zs, factor=0.5)  # reaches edge neighbours

    assert out.values[2, 1] == 5.0
    assert out.values[2, 2] == 5.0
    # Diagonal/distant cells stay NaN
    assert np.isnan(out.values[0, 0])


# ---------------------------------------------------------------------------
# dilate_zsmax — quadtree grid
# ---------------------------------------------------------------------------


def test_dilate_zsmax_quadtree_preserves_wet_set(quadtree_model):
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

    out = dilate_zsmax(zs, factor=0.5)

    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(zs.values))
    # Monotone lift on wet cells
    wet = ~np.isnan(zs.values)
    assert np.all(out.values[wet] >= zs.values[wet] - 1e-9)


# ---------------------------------------------------------------------------
# apply_energy_head
# ---------------------------------------------------------------------------


def test_apply_energy_head_below_threshold_is_noop():
    """Cells with q < q_threshold keep their original zsmax."""
    zs_vals = np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float32)
    q_vals = np.full((2, 2), 0.001, dtype=np.float32)  # below default 0.01
    zs = xr.DataArray(zs_vals, dims=("y", "x"))
    q = xr.DataArray(q_vals, dims=("y", "x"))

    out = apply_energy_head(zs, q, hmin=0.05)

    wet = ~np.isnan(zs_vals)
    np.testing.assert_allclose(out.values[wet], zs_vals[wet])
    assert np.isnan(out.values[1, 0])


def test_apply_energy_head_lifts_with_velocity():
    """v²/(2g) lift = 0.5 * (q/h)² / g for q above threshold."""
    zs = xr.DataArray(np.array([[2.0]]), dims=("y", "x"))  # 2 m WSE
    zb = xr.DataArray(np.array([[0.0]]), dims=("y", "x"))  # 0 m bed → h=2 m
    q = xr.DataArray(np.array([[4.0]]), dims=("y", "x"))  # 4 m²/s

    out = apply_energy_head(zs, q, zb=zb, hmin=0.05, q_threshold=0.01)

    # v = q/h = 2 m/s ; vel_head = v²/(2g) = 4/(2*9.81) ≈ 0.2039 m
    expected = 2.0 + 0.5 * (4.0 / 2.0) ** 2 / 9.81
    np.testing.assert_allclose(out.values[0, 0], expected, rtol=1e-6)


def test_apply_energy_head_preserves_wet_set():
    """NaN cells stay NaN, no new wet cells appear."""
    zs_vals = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
    q_vals = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    zs = xr.DataArray(zs_vals, dims=("y", "x"))
    q = xr.DataArray(q_vals, dims=("y", "x"))

    out = apply_energy_head(zs, q, hmin=0.05)

    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(zs_vals))
    # Result >= input on wet cells (velocity head is non-negative)
    wet = ~np.isnan(zs_vals)
    assert np.all(out.values[wet] >= zs_vals[wet] - 1e-9)


# ---------------------------------------------------------------------------
# downscale_velocity — helpers
# ---------------------------------------------------------------------------

_IDX_NODATA = 2147483647


def _grid(vals):
    """Fine-grid DataArray from a 2-D numpy array, with y/x coords."""
    vals = np.asarray(vals, dtype=np.float32)
    ny, nx = vals.shape
    return xr.DataArray(
        vals,
        dims=("y", "x"),
        coords={"y": np.arange(ny, 0, -1).astype(float), "x": np.arange(nx).astype(float)},
        name="hmax",
    )


def _idx(idx2d, like):
    """Index DataArray matching ``like``'s grid, with a nodata _FillValue."""
    da = xr.DataArray(
        np.asarray(idx2d, dtype=np.int64), dims=("y", "x"), coords=like.coords
    )
    da.attrs["_FillValue"] = _IDX_NODATA
    return da


def _cells(values):
    """Coarse cell field (1-D) as a plain DataArray."""
    return xr.DataArray(np.asarray(values, dtype=np.float64), dims=("cell",))


def _write_raster(path, arr, dtype, nodata):
    arr = np.asarray(arr)
    ny, nx = arr.shape
    profile = dict(
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype=dtype,
        crs="EPSG:32610",
        transform=from_origin(0, ny, 1, 1),
        nodata=nodata,
    )
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


# ---------------------------------------------------------------------------
# downscale_velocity — conveyance / continuity (in-memory)
# ---------------------------------------------------------------------------


def test_velocity_conveyance_conserves_cell_flux():
    """mean_i(h_i * v_i) over a cell equals that cell's qmax (the invariant)."""
    h = np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float32)
    hmax = _grid(h)
    idx = _idx(np.zeros((2, 2)), hmax)
    Q = 5.0
    out = downscale_velocity(
        hmax, _cells([Q]), idx, method="conveyance", froude_max=None, hmin=0.0
    )
    np.testing.assert_allclose(np.nanmean(h * out.values), Q, rtol=1e-5)


def test_velocity_continuity_is_q_over_h():
    h = np.array([[2.0, 5.0]], dtype=np.float32)
    hmax = _grid(h)
    idx = _idx(np.zeros((1, 2)), hmax)
    out = downscale_velocity(
        hmax, _cells([10.0]), idx, method="continuity", froude_max=None, hmin=0.0
    )
    np.testing.assert_allclose(out.values, 10.0 / h, rtol=1e-6)


def test_velocity_single_wet_pixel_reduces_to_continuity():
    """A cell with one wet pixel: conveyance collapses to v = q/h."""
    h = np.array([[3.0, 0.0]], dtype=np.float32)  # second pixel dry
    hmax = _grid(h)
    idx = _idx(np.zeros((1, 2)), hmax)
    out = downscale_velocity(
        hmax, _cells([6.0]), idx, method="conveyance", froude_max=None, hmin=0.05
    )
    np.testing.assert_allclose(out.values[0, 0], 6.0 / 3.0, rtol=1e-6)
    assert np.isnan(out.values[0, 1])


def test_velocity_zero_flux_gives_zero():
    h = np.array([[2.0, 3.0]], dtype=np.float32)
    hmax = _grid(h)
    idx = _idx(np.zeros((1, 2)), hmax)
    out = downscale_velocity(
        hmax, _cells([0.0]), idx, method="conveyance", froude_max=None, hmin=0.0
    )
    np.testing.assert_array_equal(out.values, np.zeros_like(h))


def test_velocity_all_dry_cell_is_nan():
    h = np.array([[0.0, 0.01], [0.02, 0.0]], dtype=np.float32)  # all <= hmin
    hmax = _grid(h)
    idx = _idx(np.zeros((2, 2)), hmax)
    out = downscale_velocity(
        hmax, _cells([5.0]), idx, method="conveyance", hmin=0.05
    )
    assert np.all(np.isnan(out.values))


def test_velocity_nodata_index_is_nan():
    h = np.array([[2.0, 3.0]], dtype=np.float32)
    hmax = _grid(h)
    idx = _idx(np.array([[0, _IDX_NODATA]]), hmax)
    out = downscale_velocity(
        hmax, _cells([5.0]), idx, method="conveyance", froude_max=None, hmin=0.0
    )
    assert np.isfinite(out.values[0, 0])
    assert np.isnan(out.values[0, 1])


def test_velocity_clip_caps_shallow_spike():
    """Shallow pixel with huge q is clipped to the Froude ceiling."""
    h = np.array([[0.1]], dtype=np.float32)
    hmax = _grid(h)
    idx = _idx(np.zeros((1, 1)), hmax)
    out = downscale_velocity(
        hmax, _cells([100.0]), idx, method="continuity", froude_max=1.0, hmin=0.0
    )
    ceiling = 1.0 * np.sqrt(9.81 * 0.1)  # Fr * sqrt(g h) ≈ 0.990
    np.testing.assert_allclose(out.values[0, 0], ceiling, rtol=1e-5)


def test_velocity_subcell_manning_shifts_distribution():
    """Equal-depth pixels: the smoother (lower-n) pixel flows faster.

    Conservation of cell-mean unit discharge still holds with roughness.
    """
    h = np.array([[2.0, 2.0]], dtype=np.float32)
    hmax = _grid(h)
    idx = _idx(np.zeros((1, 2)), hmax)
    manning = _grid(np.array([[0.02, 0.08]]))  # pixel 0 smoother
    Q = 4.0
    out = downscale_velocity(
        hmax, _cells([Q]), idx, method="conveyance",
        manning=manning, froude_max=None, hmin=0.0,
    )
    v = out.values[0]
    assert v[0] > v[1]  # smoother pixel is faster
    np.testing.assert_allclose(np.mean(h[0] * v), Q, rtol=1e-5)
    # A scalar manning must NOT change the result (it cancels in the ratio)
    out_const = downscale_velocity(
        hmax, _cells([Q]), idx, method="conveyance",
        manning=0.05, froude_max=None, hmin=0.0,
    )
    out_none = downscale_velocity(
        hmax, _cells([Q]), idx, method="conveyance",
        froude_max=None, hmin=0.0,
    )
    np.testing.assert_allclose(out_const.values, out_none.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# downscale_velocity — file-based path (two-pass, block boundaries)
# ---------------------------------------------------------------------------


def test_velocity_file_matches_inmemory_across_blocks(tmp_path):
    """A cell straddling two nrmax blocks must normalise over all its pixels."""
    h = np.array([[1.0, 4.0, 2.0, 3.0]], dtype=np.float32)  # 1x4, one cell
    idx = np.zeros((1, 4), dtype=np.uint32)
    Q = 5.0

    # In-memory reference (whole-domain normalisation)
    hmax_da = _grid(h)
    idx_da = _idx(idx, hmax_da)
    ref = downscale_velocity(
        hmax_da, _cells([Q]), idx_da, method="conveyance", froude_max=None, hmin=0.0
    )

    # File mode with nrmax=2 -> two column blocks -> cell 0 straddles them
    hmax_fn = tmp_path / "hmax.tif"
    idx_fn = tmp_path / "idx.tif"
    v_fn = tmp_path / "vel.tif"
    _write_raster(hmax_fn, h, "float32", np.nan)
    _write_raster(idx_fn, idx, "uint32", _IDX_NODATA)

    res = downscale_velocity(
        hmax_fn, _cells([Q]), idx_fn, method="conveyance",
        froude_max=None, hmin=0.0, velocity_fn=v_fn, nrmax=2,
    )
    assert res is None
    with rasterio.open(str(v_fn)) as src:
        v_file = src.read(1)
    np.testing.assert_allclose(v_file, ref.values, rtol=1e-5, equal_nan=True)


def test_velocity_file_manning_path_is_honoured(tmp_path):
    """A file-path manning raster must be used (not mistaken for a scalar)."""
    h = np.array([[2.0, 2.0]], dtype=np.float32)  # equal depth -> only n varies v
    idx = np.zeros((1, 2), dtype=np.uint32)
    n = np.array([[0.02, 0.08]], dtype=np.float32)  # sub-cell roughness
    Q = 4.0

    # In-memory reference using a DataArray manning of the same values
    hmax_da = _grid(h)
    ref = downscale_velocity(
        hmax_da, _cells([Q]), _idx(idx, hmax_da), method="conveyance",
        manning=_grid(n), froude_max=None, hmin=0.0,
    )

    hmax_fn = tmp_path / "hmax.tif"
    idx_fn = tmp_path / "idx.tif"
    man_fn = tmp_path / "manning.tif"
    v_fn = tmp_path / "vel.tif"
    _write_raster(hmax_fn, h, "float32", np.nan)
    _write_raster(idx_fn, idx, "uint32", _IDX_NODATA)
    _write_raster(man_fn, n, "float32", np.nan)

    downscale_velocity(
        hmax_fn, _cells([Q]), idx_fn, method="conveyance",
        manning=str(man_fn), froude_max=None, hmin=0.0, velocity_fn=v_fn, nrmax=64,
    )
    with rasterio.open(str(v_fn)) as src:
        v_file = src.read(1)
    np.testing.assert_allclose(v_file, ref.values, rtol=1e-5, equal_nan=True)
    assert v_file[0, 0] > v_file[0, 1]  # smoother pixel faster -> manning honoured


def test_velocity_northup_qmax_is_normalised_to_south_up():
    """A north-up (descending-y) qmax must be flipped to match the index order.

    Index COGs point into the SOUTH-UP flatten (make_index_cog convention);
    regression for the silent scramble when a field arrives north-up.
    """
    # two coarse cells stacked vertically; south-up flatten: 0=south, 1=north
    # fine 2x2: top (north) pixels -> cell 1, bottom (south) pixels -> cell 0
    h = np.full((2, 2), 2.0, dtype=np.float32)
    hmax = _grid(h)
    idx = _idx(np.array([[1, 1], [0, 0]]), hmax)

    q_south_up = xr.DataArray(  # row 0 = south cell (q=2), row 1 = north (q=4)
        np.array([[2.0], [4.0]]), dims=("y", "x"),
        coords={"y": [0.5, 1.5], "x": [0.5]},
    )
    q_north_up = xr.DataArray(  # same field, stored north-up
        np.array([[4.0], [2.0]]), dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5]},
    )

    kw = dict(method="continuity", froude_max=None, hmin=0.0)
    v_s = downscale_velocity(hmax, q_south_up, idx, **kw)
    v_n = downscale_velocity(hmax, q_north_up, idx, **kw)

    # v = q/h: north pixels 4/2=2, south pixels 2/2=1 — for BOTH orientations
    np.testing.assert_allclose(v_s.values, [[2.0, 2.0], [1.0, 1.0]], rtol=1e-6)
    np.testing.assert_allclose(v_n.values, v_s.values, rtol=1e-6)


# ---------------------------------------------------------------------------
# smooth_cell_field + downscale_velocity smoothing
# ---------------------------------------------------------------------------


def test_smooth_cell_field_nan_aware_mean():
    """Window mean uses wet cells only; dry cells stay NaN (wet set preserved)."""
    vals = np.array(
        [
            [1.0, 2.0, np.nan],
            [3.0, 4.0, 5.0],
            [np.nan, 6.0, 7.0],
        ],
        dtype=np.float32,
    )
    da = xr.DataArray(vals, dims=("y", "x"))
    out = smooth_cell_field(da, n=3)

    # wet set preserved
    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(vals))
    # centre cell: mean of the 7 wet cells in its 3x3 window
    np.testing.assert_allclose(out.values[1, 1], np.nanmean(vals), rtol=1e-6)
    # corner cell [0,0]: window covers (1,2,3,4) -> 2.5
    np.testing.assert_allclose(out.values[0, 0], 2.5, rtol=1e-6)


def test_smooth_cell_field_constant_and_noop():
    vals = np.full((4, 4), 3.7, dtype=np.float32)
    da = xr.DataArray(vals, dims=("y", "x"))
    # constant field is unchanged by smoothing
    np.testing.assert_allclose(smooth_cell_field(da, n=3).values, vals, rtol=1e-6)
    # n<=1 is a no-op; even n raises
    assert smooth_cell_field(da, n=1) is da
    import pytest

    with pytest.raises(ValueError, match="odd"):
        smooth_cell_field(da, n=4)


def test_smooth_cell_field_quadtree(quadtree_model):
    """Quadtree smoothing: wet set preserved, values within local range."""
    grid = quadtree_model.quadtree_grid.data.ugrid.grid
    n_faces = grid.n_face
    rng = np.random.default_rng(3)
    vals = rng.uniform(0.0, 2.0, n_faces).astype(np.float32)
    vals[: n_faces // 3] = np.nan  # some dry cells
    da = xu.UgridDataArray(
        xr.DataArray(vals, dims=(grid.face_dimension,), name="qmax"), grid=grid
    )
    out = smooth_cell_field(da, n=3)
    np.testing.assert_array_equal(np.isnan(out.values), np.isnan(vals))
    wet = ~np.isnan(vals)
    assert np.all(out.values[wet] >= np.nanmin(vals) - 1e-6)
    assert np.all(out.values[wet] <= np.nanmax(vals) + 1e-6)


def test_velocity_smooth_blends_cell_flux():
    """smooth=3 averages neighbouring cell fluxes before redistribution."""
    # two coarse cells side by side (q=2 and q=4), two fine pixels each
    h = np.full((1, 4), 1.0, dtype=np.float32)
    hmax = _grid(h)
    idx = _idx(np.array([[0, 0, 1, 1]]), hmax)
    q = xr.DataArray(
        np.array([[2.0, 4.0]]), dims=("y", "x"), coords={"y": [0.5], "x": [0.5, 1.5]}
    )

    kw = dict(method="continuity", froude_max=None, hmin=0.0)
    v_raw = downscale_velocity(hmax, q, idx, **kw)
    v_sm = downscale_velocity(hmax, q, idx, smooth=3, **kw)

    # unsmoothed: blocky 2 | 4; smoothed: both cells see mean(2,4)=3
    np.testing.assert_allclose(v_raw.values, [[2.0, 2.0, 4.0, 4.0]], rtol=1e-6)
    np.testing.assert_allclose(v_sm.values, [[3.0, 3.0, 3.0, 3.0]], rtol=1e-6)
