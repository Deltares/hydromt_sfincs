"""Tests for the SfincsDrainageStructures component's TOML (``sfincs.toml.drn``) path.

Covers the ``obs_1``/``obs_2`` and gate ``flow_coef`` fields, which
``read_toml``/``write_toml`` used to silently drop (obs was never modelled at
all; flow_coef was only handled for culvert types, not gates).
"""

import tomllib

import tomli_w
import pytest

from .conftest import TESTDATADIR, TESTMODELDIR

# Two line locations already known to fall inside the sfincs_test model
# region (they are the src/dst points of the pump and culvert_simple
# entries in tests/data/sfincs_test/sfincs.drn).
SRC_1 = (322021.9, 5045547.7)
SRC_2 = (322378.7, 5045133.8)
SRC_1_B = (319160.2, 5045262.2)
SRC_2_B = (319167.4, 5044976.8)


def _gate_entry(
    name="GATE01",
    src_1=SRC_1,
    src_2=SRC_2,
    obs_1=None,
    obs_2=None,
    flow_coef=0.9,
    width=12.5,
    sill_elevation=-1.2,
    mannings_n=0.03,
    opening_duration=300.0,
    closing_duration=450.0,
    rules=(
        {"operation": "open", "when": "z1>0.5 & z1<2.0"},
        {"operation": "close", "when": "z1-z2<0.03048"},
    ),
):
    """Build a single ``[[src_structure]]`` gate entry dict.

    ``obs_1``/``obs_2``/``flow_coef`` are only added to the entry when not
    None, so a test can omit them to exercise the "file omits the key"
    path.
    """
    entry = {
        "type": "gate",
        "name": name,
        "src_1": list(src_1),
        "src_2": list(src_2),
        "width": width,
        "sill_elevation": sill_elevation,
        "mannings_n": mannings_n,
        "opening_duration": opening_duration,
        "closing_duration": closing_duration,
    }
    if flow_coef is not None:
        entry["flow_coef"] = flow_coef
    if obs_1 is not None:
        entry["obs_1"] = list(obs_1)
    if obs_2 is not None:
        entry["obs_2"] = list(obs_2)
    if rules:
        entry["rule"] = [dict(r) for r in rules]
    return entry


def _write_toml(path, entries):
    with open(path, "wb") as f:
        tomli_w.dump({"src_structure": list(entries)}, f)


def test_drainage_structures_gate_full_roundtrip(model_config, tmp_path):
    """A gate carrying the full TOML parameter set survives read -> write -> read.

    Covers src/obs/width/sill_elevation/flow_coef/mannings_n/durations and
    rule order (>= 2 rules).
    """
    obs_1 = SRC_1
    obs_2 = (SRC_2[0] + 1000.0, SRC_2[1])
    rules = (
        {"operation": "open", "when": "z1>0.5 & z1<2.0"},
        {"operation": "close", "when": "z1-z2<0.03048"},
        {"operation": "hold", "when": "z1<-999"},
    )
    entry = _gate_entry(src_1=SRC_1, src_2=SRC_2, obs_1=obs_1, obs_2=obs_2, rules=rules)

    toml_path = tmp_path / "sfincs.toml.drn"
    _write_toml(toml_path, [entry])

    model_config.root.set(tmp_path, mode="r+")
    model_config.drainage_structures.read_toml(toml_path)
    gdf0 = model_config.drainage_structures.data

    out_path = tmp_path / "roundtrip.toml.drn"
    model_config.drainage_structures.write_toml(out_path)

    model_config.drainage_structures.clear()
    model_config.drainage_structures.read_toml(out_path)
    gdf1 = model_config.drainage_structures.data

    for gdf in (gdf0, gdf1):
        assert len(gdf) == 1
        row = gdf.iloc[0]
        assert int(row["type"]) == 4
        assert row["width"] == pytest.approx(12.5)
        assert row["sill_elevation"] == pytest.approx(-1.2)
        assert row["flow_coef"] == pytest.approx(0.9)
        assert row["mannings_n"] == pytest.approx(0.03)
        assert row["opening_duration"] == pytest.approx(300.0)
        assert row["closing_duration"] == pytest.approx(450.0)
        assert row["obs_1_x"] == pytest.approx(obs_1[0])
        assert row["obs_1_y"] == pytest.approx(obs_1[1])
        assert row["obs_2_x"] == pytest.approx(obs_2[0])
        assert row["obs_2_y"] == pytest.approx(obs_2[1])
        assert row["rules"] == [dict(r) for r in rules]

        coords = list(row.geometry.coords)
        assert coords[0] == pytest.approx(SRC_1)
        assert coords[-1] == pytest.approx(SRC_2)


def test_drainage_structures_obs_defaults_to_src(model_config, tmp_path):
    """A gate with no ``obs_1``/``obs_2`` in the file gets the src points as obs."""
    entry = _gate_entry(src_1=SRC_1_B, src_2=SRC_2_B, obs_1=None, obs_2=None)

    toml_path = tmp_path / "sfincs.toml.drn"
    _write_toml(toml_path, [entry])

    model_config.root.set(tmp_path, mode="r+")
    model_config.drainage_structures.read_toml(toml_path)
    row = model_config.drainage_structures.data.iloc[0]

    assert row["obs_1_x"] == pytest.approx(SRC_1_B[0])
    assert row["obs_1_y"] == pytest.approx(SRC_1_B[1])
    assert row["obs_2_x"] == pytest.approx(SRC_2_B[0])
    assert row["obs_2_y"] == pytest.approx(SRC_2_B[1])


def test_drainage_structures_obs_equal_to_src_omitted_on_write(model_config, tmp_path):
    """obs == src round-trips through omission: no obs_1/obs_2 keys are written.

    SFINCS itself defaults obs to src when the keys are absent, so writing
    them when they equal src would add keys the source file never had. A
    gate whose obs is explicitly set equal to src (as well as one that
    never specified obs at all) must therefore write no obs_1/obs_2 keys,
    and reading that file back must still yield obs columns equal to src.
    """
    entry = _gate_entry(src_1=SRC_1_B, src_2=SRC_2_B, obs_1=SRC_1_B, obs_2=SRC_2_B)
    toml_path = tmp_path / "sfincs.toml.drn"
    _write_toml(toml_path, [entry])

    model_config.root.set(tmp_path, mode="r+")
    model_config.drainage_structures.read_toml(toml_path)

    out_path = tmp_path / "roundtrip.toml.drn"
    model_config.drainage_structures.write_toml(out_path)

    with open(out_path, "rb") as f:
        doc = tomllib.load(f)
    written = doc["src_structure"][0]
    assert "obs_1" not in written
    assert "obs_2" not in written

    model_config.drainage_structures.clear()
    model_config.drainage_structures.read_toml(out_path)
    row = model_config.drainage_structures.data.iloc[0]

    assert row["obs_1_x"] == pytest.approx(SRC_1_B[0])
    assert row["obs_1_y"] == pytest.approx(SRC_1_B[1])
    assert row["obs_2_x"] == pytest.approx(SRC_2_B[0])
    assert row["obs_2_y"] == pytest.approx(SRC_2_B[1])


def test_drainage_structures_obs_independent_of_src(model_config, tmp_path):
    """``obs_2`` 1000 m from ``src_2`` keeps that distance across a round trip.

    This is the case that failed before the fix: obs columns did not exist
    at all, so any explicit obs_1/obs_2 in the file was silently discarded
    on read.
    """
    obs_1 = SRC_1  # exactly on src_1, as in the reference model
    obs_2 = (SRC_2[0] + 1000.0, SRC_2[1])  # 1000 m east of src_2

    entry = _gate_entry(src_1=SRC_1, src_2=SRC_2, obs_1=obs_1, obs_2=obs_2)
    toml_path = tmp_path / "sfincs.toml.drn"
    _write_toml(toml_path, [entry])

    model_config.root.set(tmp_path, mode="r+")
    model_config.drainage_structures.read_toml(toml_path)

    out_path = tmp_path / "roundtrip.toml.drn"
    model_config.drainage_structures.write_toml(out_path)
    model_config.drainage_structures.clear()
    model_config.drainage_structures.read_toml(out_path)

    row = model_config.drainage_structures.data.iloc[0]
    src_2_after = list(row.geometry.coords)[-1]
    dist = (
        (row["obs_2_x"] - src_2_after[0]) ** 2 + (row["obs_2_y"] - src_2_after[1]) ** 2
    ) ** 0.5

    assert dist == pytest.approx(1000.0, abs=0.01)
    assert row["obs_1_x"] == pytest.approx(obs_1[0])
    assert row["obs_1_y"] == pytest.approx(obs_1[1])


def test_drainage_structures_gate_flow_coef(model_config, tmp_path):
    """A gate's ``flow_coef`` survives a round trip at a non-default value.

    The gate default (also what a file omitting the key falls back to) is
    1.0 -- distinct from the culvert default of 0.6 -- so this file sets
    0.9 to make sure a regression back to either default would be caught.
    """
    entry = _gate_entry(src_1=SRC_1_B, src_2=SRC_2_B, flow_coef=0.9)
    toml_path = tmp_path / "sfincs.toml.drn"
    _write_toml(toml_path, [entry])

    model_config.root.set(tmp_path, mode="r+")
    model_config.drainage_structures.read_toml(toml_path)
    row = model_config.drainage_structures.data.iloc[0]
    assert row["flow_coef"] == pytest.approx(0.9)

    out_path = tmp_path / "roundtrip.toml.drn"
    model_config.drainage_structures.write_toml(out_path)
    model_config.drainage_structures.clear()
    model_config.drainage_structures.read_toml(out_path)
    row2 = model_config.drainage_structures.data.iloc[0]
    assert row2["flow_coef"] == pytest.approx(0.9)


def test_drainage_structures_gate_flow_coef_defaults_to_one(model_config, tmp_path):
    """A gate with no ``flow_coef`` in the file gets the gate default, 1.0.

    Confirmed with the branch author: a gate's flow_coef should default to
    1.0, not the culvert value of 0.6 that hydromt used to stamp on every
    type via ``_DEFAULTS``.
    """
    entry_no_fc = _gate_entry(src_1=SRC_1_B, src_2=SRC_2_B, flow_coef=None)
    toml_path = tmp_path / "sfincs_no_flow_coef.toml.drn"
    _write_toml(toml_path, [entry_no_fc])

    model_config.root.set(tmp_path, mode="r+")
    model_config.drainage_structures.read_toml(toml_path)
    row = model_config.drainage_structures.data.iloc[0]
    assert row["flow_coef"] == pytest.approx(1.0)


def test_drainage_structures_culvert_flow_coef_default_unchanged(
    model_config, tmp_path
):
    """A culvert with no ``flow_coef`` still defaults to 0.6.

    Regression guard: the gate default of 1.0 must not leak into the
    culvert_simple/culvert types, which keep the pre-existing 0.6 default.
    """
    entry = {
        "type": "culvert_simple",
        "name": "CULV01",
        "src_1": list(SRC_1_B),
        "src_2": list(SRC_2_B),
        "direction": "both",
    }
    toml_path = tmp_path / "sfincs.toml.drn"
    _write_toml(toml_path, [entry])

    model_config.root.set(tmp_path, mode="r+")
    model_config.drainage_structures.read_toml(toml_path)
    row = model_config.drainage_structures.data.iloc[0]
    assert int(row["type"]) == 2
    assert row["flow_coef"] == pytest.approx(0.6)
