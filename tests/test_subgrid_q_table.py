"""Tests for the u/v conveyance table (hydromt_sfincs.workflows.subgrid.subgrid_q_table)."""

import numpy as np
import pytest

from hydromt_sfincs.workflows.subgrid import subgrid_q_table

NLEV = 11
HUTHRESH = 0.01


def _sliver_box():
    """20 x 20 pixel u/v box with a deep channel that lies almost entirely in side A
    (the first half of the flattened pixels), as happens where a sinuous channel
    crosses a face near a grid corner."""
    z = np.full((20, 20), 0.5)
    z[:10, 2:8] = -4.0  # channel in side A
    z[10:, 0:1] = -4.0  # one-pixel-wide sliver in side B
    return z.ravel(), np.full(400, 0.02)


def _whole_box_reference(elev, rgh, zz):
    havg = np.array([np.mean(np.maximum(z - elev, 0.0)) for z in zz])
    qall = np.array([np.mean(np.maximum(z - elev, 0.0) ** (5 / 3) / rgh) for z in zz])
    return havg, qall


def test_weight_option_all_is_whole_box_average():
    elev, rgh = _sliver_box()
    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(
        elev, rgh, NLEV, HUTHRESH, 2, -99999.0, -99999.0, "all", "manning"
    )
    havg_ref, qall_ref = _whole_box_reference(elev, rgh, zz)
    np.testing.assert_allclose(havg, havg_ref, rtol=1e-12)
    np.testing.assert_allclose(havg ** (5 / 3) / nrep, qall_ref, rtol=1e-12)
    np.testing.assert_allclose(
        pwet, [(z > elev).sum() / elev.size for z in zz], rtol=1e-12
    )


def test_weight_option_all_ignores_side_split_and_option():
    """'all' depends only on the pixel set: shuffling pixels across sides A and B, or
    switching q_table_option, must not change the table (except zmin, which is set
    by the side minima)."""
    elev, rgh = _sliver_box()
    ref = subgrid_q_table(
        elev, rgh, NLEV, HUTHRESH, 2, -99999.0, -99999.0, "all", "manning"
    )
    perm = np.random.default_rng(0).permutation(elev.size)
    shuffled = subgrid_q_table(
        elev[perm], rgh[perm], NLEV, HUTHRESH, 2, -99999.0, -99999.0, "all", "manning"
    )
    opt1 = subgrid_q_table(
        elev, rgh, NLEV, HUTHRESH, 1, -99999.0, -99999.0, "all", "manning"
    )
    # the shuffle leaves both sides with channel pixels, so zmin stays -4 + huthresh
    for a, b in zip(ref, shuffled):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    for a, b in zip(ref, opt1):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_min_throttles_sliver_face_and_all_does_not():
    elev, rgh = _sliver_box()
    out = {
        w: subgrid_q_table(
            elev, rgh, NLEV, HUTHRESH, 2, -99999.0, -99999.0, w, "manning"
        )
        for w in ("min", "mean", "all")
    }
    # unit conveyance at the level just below the floodplain (index NLEV - 2)
    k = {w: o[2][-2] ** (5 / 3) / o[3][-2] for w, o in out.items()}
    assert k["min"] < k["mean"] < k["all"]
    # 'min' is the side-B (sliver) conveyance
    zz = out["min"][7]
    hb = np.maximum(zz[-2] - elev[200:], 0.0)
    np.testing.assert_allclose(k["min"], np.mean(hb ** (5 / 3) / rgh[200:]), rtol=1e-12)


@pytest.mark.parametrize("weight_option", ["min", "mean", "all"])
def test_tables_are_finite(weight_option):
    elev, rgh = _sliver_box()
    out = subgrid_q_table(
        elev, rgh, NLEV, HUTHRESH, 2, -99999.0, -99999.0, weight_option, "manning"
    )
    for a in out:
        assert np.all(np.isfinite(a))
