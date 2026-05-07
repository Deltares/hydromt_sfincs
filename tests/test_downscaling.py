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
import xarray as xr
import xugrid as xu

from hydromt_sfincs.workflows.downscaling import (
    apply_energy_head,
    dilate_zsmax,
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
