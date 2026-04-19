"""Tests for the SfincsUrbanDrainageAreas component."""

import tomllib
from os.path import isfile, join

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Fixtures — synthetic zones in the Charleston-like test region
# ---------------------------------------------------------------------------


def _square(x0: float, y0: float, size: float) -> Polygon:
    return Polygon(
        [
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
            (x0, y0),
        ]
    )


@pytest.fixture
def zones_gdf(model_config):
    """Two zones (one piped_drainage, one injection_well) inside the model region."""
    region = model_config.region
    minx, miny, maxx, maxy = region.total_bounds
    dx = (maxx - minx) / 10.0
    dy = (maxy - miny) / 10.0
    poly_a = _square(minx + 2 * dx, miny + 2 * dy, dx)
    poly_b = _square(minx + 5 * dx, miny + 5 * dy, dx)

    data = {
        "name": ["downtown", "north_well_field"],
        "type": ["piped_drainage", "injection_well"],
        "polygon_file": ["sfincs.pol", "sfincs.pol"],
        "outfall_x": [minx + 3 * dx, np.nan],
        "outfall_y": [miny + 3 * dy, np.nan],
        "design_precip": [20.0, np.nan],
        "max_outfall_rate": [np.nan, np.nan],
        "injection_rate": [np.nan, 0.5],
        "maximum_capacity": [np.nan, 5000.0],
        "h_threshold": [0.02, 0.0],
        "check_valve": [True, False],
        "include_outfall": [True, True],
        "dh_design_min": [0.1, 0.1],
    }
    return gpd.GeoDataFrame(data, geometry=[poly_a, poly_b], crs=model_config.crs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_urban_drainage_areas_io(model_config, zones_gdf, tmp_path):
    """Round-trip set → write → read → compare."""
    model_config.urban_drainage_areas.set(zones_gdf, merge=False)

    # Switch root to a writable tmp path.
    model_config.root.set(tmp_path, mode="r+")

    urb_file = tmp_path / "sfincs.urb"
    model_config.urban_drainage_areas.write(filename=urb_file)

    # Files on disk.
    assert urb_file.exists()
    assert (tmp_path / "sfincs.pol").exists()

    # urbfile made it into the config.
    assert model_config.config.get("urbfile") == "sfincs.urb"

    # TOML structure matches the schema.
    with open(urb_file, "rb") as f:
        doc = tomllib.load(f)
    zones = doc["urban_drainage_zone"]
    assert len(zones) == 2
    piped = next(z for z in zones if z["type"] == "piped_drainage")
    well = next(z for z in zones if z["type"] == "injection_well")
    # piped: full set of piped keys, and NOT the injection-well-only keys.
    # design_precip XOR max_outfall_rate — the user supplied design_precip.
    for k in (
        "outfall_x",
        "outfall_y",
        "design_precip",
        "dh_design_min",
        "include_outfall",
        "check_valve",
        "h_threshold",
    ):
        assert k in piped
    assert "max_outfall_rate" not in piped
    assert "injection_rate" not in piped
    assert "maximum_capacity" not in piped
    # well: injection-well keys only, no piped-only keys.
    for k in ("injection_rate", "maximum_capacity", "h_threshold"):
        assert k in well
    for k in (
        "design_precip",
        "max_outfall_rate",
        "outfall_x",
        "outfall_y",
        "check_valve",
        "include_outfall",
        "dh_design_min",
    ):
        assert k not in well

    # Read back and compare essential fields.
    model_config.urban_drainage_areas.clear()
    model_config.urban_drainage_areas.read(filename=urb_file)
    gdf1 = model_config.urban_drainage_areas.data

    assert len(gdf1) == 2
    assert set(gdf1["name"]) == {"downtown", "north_well_field"}
    row_piped = gdf1[gdf1["type"] == "piped_drainage"].iloc[0]
    assert row_piped["design_precip"] == pytest.approx(20.0)
    assert bool(row_piped["check_valve"]) is True
    row_well = gdf1[gdf1["type"] == "injection_well"].iloc[0]
    assert row_well["injection_rate"] == pytest.approx(0.5)
    assert row_well["maximum_capacity"] == pytest.approx(5000.0)


def test_urban_drainage_areas_validation(model_config, zones_gdf):
    """Invalid input should raise before anything is written."""
    bad = zones_gdf.copy()
    bad.loc[bad["type"] == "piped_drainage", "design_precip"] = np.nan
    bad.loc[bad["type"] == "piped_drainage", "max_outfall_rate"] = np.nan
    with pytest.raises(ValueError, match="design_precip"):
        model_config.urban_drainage_areas.set(bad, merge=False)

    both = zones_gdf.copy()
    both.loc[both["type"] == "piped_drainage", "max_outfall_rate"] = 6.0
    with pytest.raises(ValueError, match="only one"):
        model_config.urban_drainage_areas.set(both, merge=False)

    dup = zones_gdf.copy()
    dup.loc[:, "name"] = "downtown"
    with pytest.raises(ValueError, match="Duplicate"):
        model_config.urban_drainage_areas.set(dup, merge=False)

    wrong_type = zones_gdf.copy()
    wrong_type.loc[:, "type"] = "bogus"
    with pytest.raises(ValueError, match="Invalid type"):
        model_config.urban_drainage_areas.set(wrong_type, merge=False)


def test_urban_drainage_areas_shared_polygon_file(model_config, zones_gdf, tmp_path):
    """Zones sharing a polygon_file end up in the same polygon file."""
    model_config.urban_drainage_areas.set(zones_gdf, merge=False)
    model_config.root.set(tmp_path, mode="r+")
    model_config.urban_drainage_areas.write(filename=tmp_path / "sfincs.urb")

    pol = tmp_path / "sfincs.pol"
    text = pol.read_text()
    # Both zone names appear as polygon headers in the same file.
    assert "downtown" in text
    assert "north_well_field" in text
    # Only one polygon file (no per-zone files were generated by accident).
    pol_files = list(tmp_path.glob("*.pol"))
    assert len(pol_files) == 1


def test_urban_drainage_areas_delete_clear(model_config, zones_gdf):
    """delete() drops a row, clear() empties the component."""
    model_config.urban_drainage_areas.set(zones_gdf, merge=False)
    assert model_config.urban_drainage_areas.nr_areas == 2

    model_config.urban_drainage_areas.delete(index=[0])
    assert model_config.urban_drainage_areas.nr_areas == 1
    assert model_config.urban_drainage_areas.data["name"].iloc[0] == "north_well_field"

    with pytest.raises(ValueError):
        model_config.urban_drainage_areas.delete(index=[42])

    model_config.urban_drainage_areas.clear()
    assert model_config.urban_drainage_areas.data.empty
    assert model_config.config.get("urbfile") is None
