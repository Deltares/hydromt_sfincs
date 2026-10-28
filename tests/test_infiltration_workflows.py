import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path

from hydromt_sfincs import workflows, DATADIR


def _raster(values, name, dtype=np.float32):
    values = np.asarray(values, dtype=dtype)
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": np.arange(values.shape[0]), "x": np.arange(values.shape[1])},
        name=name,
    )
    da.raster.set_crs(4326)
    return da


def _modifier_table():
    """NLCD land-use classes mapped to infiltration modifier factors."""
    return pd.DataFrame(
        {
            "surface_factor": [
                0.05,
                0.70,
                0.45,
                0.45,
                0.20,
                0.85,
                1.15,
                1.15,
                1.15,
                1.00,
                1.00,
                0.90,
                0.90,
                0.55,
                0.55,
            ],
            "storage_factor": [
                0.20,
                0.75,
                0.60,
                0.60,
                0.40,
                0.80,
                1.30,
                1.30,
                1.30,
                1.00,
                1.00,
                1.10,
                1.10,
                0.70,
                0.70,
            ],
            "drainage_factor": [
                1.00,
                1.15,
                1.25,
                1.25,
                1.40,
                1.05,
                0.90,
                0.90,
                0.90,
                1.00,
                1.00,
                1.05,
                1.05,
                0.80,
                0.80,
            ],
        },
        index=[11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95],
    )


def test_normalize_hsg_codes_drained():
    da_hsg = _raster([[1, 2, 3, 4], [5, 6, 7, 8]], "hsg", dtype=np.int16)
    da_norm = workflows.normalize_hsg_codes(da_hsg, mode="drained")
    np.testing.assert_array_equal(
        da_norm.values,
        np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32),
    )


def test_normalize_hsg_codes_warns_on_unsupported_values(caplog):
    da_hsg = _raster([[1, 99]], "hsg", dtype=np.int16)

    with caplog.at_level(
        logging.WARNING, logger="hydromt_sfincs.workflows.infiltration"
    ):
        da_norm = workflows.normalize_hsg_codes(da_hsg, mode="drained")

    assert da_norm.values[0, 0] == 1.0
    assert np.isnan(da_norm.values[0, 1])
    assert len(caplog.records) == 1
    assert "Unsupported HSG codes [1-8] not recognized by the HSG mapping [99]" in (
        caplog.text
    )


def test_normalize_hsg_codes_logs_present_mappings(caplog):
    da_hsg = _raster([[5, 8]], "hsg", dtype=np.int16)

    with caplog.at_level(logging.INFO, logger="hydromt_sfincs.workflows.infiltration"):
        da_norm = workflows.normalize_hsg_codes(da_hsg, mode="drained")

    np.testing.assert_array_equal(da_norm.values, [[1.0, 4.0]])
    assert caplog.messages == [
        "Mapping HSG value 5 to drained value 1",
        "Mapping HSG value 8 to drained value 4",
    ]


def test_normalize_hsg_codes_native_and_invalid():
    da_hsg = _raster([[5, 6, 7, 8]], "hsg", dtype=np.int16)
    np.testing.assert_array_equal(
        workflows.normalize_hsg_codes(da_hsg, mode="native").values,
        np.array([[5, 6, 7, 8]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="dual_hsg"):
        workflows.normalize_hsg_codes(da_hsg, mode="bogus")
    with pytest.raises(ValueError, match="dual_hsg"):
        workflows.normalize_hsg_codes(da_hsg, mode=None)
    with pytest.raises(ValueError, match="dual_hsg"):
        workflows.normalize_hsg_codes(da_hsg, mode="none")


def test_mask_invalid_ksat_keeps_values_and_masks_nodata():
    da_ksat = _raster([[1.0, 2.5, np.nan, -9999.0]], "ksat")
    da_ks = workflows.mask_invalid_ksat(da_ksat)
    np.testing.assert_allclose(
        da_ks.values, [[1.0, 2.5, np.nan, np.nan]], rtol=1e-6, equal_nan=True
    )
    assert np.isnan(da_ks.raster.nodata)


def _projected(values, res, dtype=np.float32, nodata=None):
    """North-up raster in UTM with square cells of size ``res``."""
    values = np.asarray(values, dtype=dtype)
    ny, nx = values.shape
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "y": 1000.0 - res * (np.arange(ny) + 0.5),
            "x": 500.0 + res * (np.arange(nx) + 0.5),
        },
    )
    da.raster.set_crs(32617)
    if nodata is not None:
        da.raster.set_nodata(nodata)
    return da


def test_curve_number_open_water_area_weighted():
    # 4x4-pixel model cells: a quarter water, all water, all land, and half
    # land / half an unmapped class; land-use nodata is 255 as in NLCD
    lulc = np.full((8, 8), 41, dtype=np.uint8)
    lulc[:2, :2] = 11
    lulc[:4, 4:] = 11
    lulc[4:, 6:] = 99
    da_lulc = _projected(lulc, 1.0, dtype=np.uint8, nodata=255)
    da_hsg = _projected(np.full((8, 8), 2), 1.0, dtype=np.uint8, nodata=0)
    df_map = pd.DataFrame({"2": [0.0, 70.0]}, index=[11, 41])

    da_cn = workflows.curve_number_from_landuse_hsg(da_lulc, da_hsg, df_map)
    assert np.isnan(da_cn.raster.nodata)
    # the same steps as the CN-A components: S per pixel, then area average
    da_s = workflows.cn_to_s(da_cn, fill_cn=None)
    da_like = _projected(np.ones((2, 2)), 4.0)
    da_scs = da_s.raster.reproject_like(da_like, method="average")

    s_land = round(1000.0 / 70.0 - 10.0, 3)
    expected = [[0.75 * s_land, 0.0], [s_land, s_land]]
    np.testing.assert_allclose(da_scs.values, expected, rtol=1e-4)


def test_curve_number_with_recovery_ksat_cap_and_nodata():
    da_lulc = _projected(np.full((8, 8), 41), 1.0, dtype=np.uint8, nodata=255)
    da_hsg = _projected(np.full((8, 8), 1), 1.0, dtype=np.uint8, nodata=0)
    df_map = pd.DataFrame({"1": [70.0]}, index=[41])
    ksat = np.full((8, 8), 180.0)  # mm/hr
    ksat[:, 4:] = 1440.0  # mm/hr, above the default cap
    ksat[0, 0] = -9999.0  # undeclared nodata must not pull the average down
    da_ksat = _projected(ksat, 1.0)
    da_like = _projected(np.ones((2, 2)), 4.0)

    ds = workflows.curve_number_with_recovery(
        da_lulc, da_hsg, da_ksat, df_map, effective=0.5, da_mask=da_like
    )
    np.testing.assert_allclose(
        ds["ks"].values, [[180.0, 360.0], [180.0, 360.0]], rtol=1e-5
    )


def test_cn_to_s_units_and_invalid():
    da_cn = _raster([[0.0, 1.0, 80.0, 100.0]], "cn")
    da_s = workflows.cn_to_s(da_cn)
    np.testing.assert_allclose(da_s.values, [[990.0, 990.0, 2.5, 0.0]], rtol=1e-6)
    assert np.isnan(da_s.raster.nodata)
    np.testing.assert_allclose(
        workflows.cn_to_s(da_cn, output_unit="m").values,
        [[990.0 * 0.0254, 990.0 * 0.0254, 2.5 * 0.0254, 0.0]],
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="output_unit"):
        workflows.cn_to_s(da_cn, output_unit="mm")


def test_curve_number_from_landuse_hsg_expected_values():
    df_map = pd.DataFrame({1: [30.0, 77.0], 2: [55.0, 85.0]}, index=[41, 24])
    da_landuse = _raster([[41, 24, 41], [24, 41, 24]], "lulc", dtype=np.int16)
    da_hsg = _raster([[1, 2, 3], [1, 1, 2]], "hsg", dtype=np.int16)

    da_cn = workflows.curve_number_from_landuse_hsg(da_landuse, da_hsg, df_map)

    assert np.isclose(float(da_cn.values[0, 0]), 30.0)
    assert np.isclose(float(da_cn.values[0, 1]), 85.0)
    assert np.isclose(float(da_cn.values[1, 0]), 77.0)
    # HSG 3 is absent from the table, so the combination stays unmapped
    assert np.isnan(da_cn.values[0, 2])


def test_curve_number_from_landuse_hsg_water_and_missing_values():
    # CN 0 marks open water (impervious, CN 100); -9999 means no value
    df_map = pd.DataFrame({1: [0.0, 70.0], 2: [0.0, -9999.0]}, index=[11, 41])
    da_landuse = _raster([[11, 41], [11, 41]], "lulc", dtype=np.int16)
    da_hsg = _raster([[1, 1], [2, 2]], "hsg", dtype=np.int16)

    da_cn = workflows.curve_number_from_landuse_hsg(da_landuse, da_hsg, df_map)

    np.testing.assert_allclose(
        da_cn.values, [[100.0, 70.0], [100.0, np.nan]], equal_nan=True
    )


def test_adjust_curve_number_dry_wet_avg():
    da_cn = _raster([[70.0]], "cn")

    assert np.isclose(float(workflows.adjust_curve_number(da_cn).values[0, 0]), 70.0)
    dry = float(workflows.adjust_curve_number(da_cn, "dry").values[0, 0])
    wet = float(workflows.adjust_curve_number(da_cn, "wet").values[0, 0])

    assert np.isclose(dry, 70.0 / (2.281 - 0.01281 * 70.0), rtol=1e-5)
    assert np.isclose(wet, 70.0 / (0.427 + 0.00573 * 70.0), rtol=1e-5)
    assert dry < 70.0 < wet

    with pytest.raises(ValueError, match="antecedent_moisture"):
        workflows.adjust_curve_number(da_cn, "soaked")


def test_green_ampt_from_hsg_requires_ks_or_ksat():
    da_hsg = _raster([[1, 2]], "hsg", dtype=np.int16)
    df_map = pd.DataFrame({"psi": [90.0, 120.0], "sigma": [0.20, 0.25]}, index=[1, 2])

    with pytest.raises(ValueError, match="hydraulic conductivity"):
        workflows.green_ampt_from_hsg(da_hsg, df_map)

    da_ksat = _raster([[1.0, 2.0]], "ksat")
    ds = workflows.green_ampt_from_hsg(da_hsg, df_map, da_ksat=da_ksat)
    np.testing.assert_allclose(ds["psi"].values, [[90.0, 120.0]], rtol=1e-6)
    np.testing.assert_allclose(ds["ks"].values, [[1.0, 2.0]], rtol=1e-6)


def test_horton_from_hsg_derives_fc_and_f0():
    da_hsg = _raster([[1, 2]], "hsg", dtype=np.int16)
    da_ksat = _raster([[1.0, 1.0]], "ksat")
    df_map = pd.DataFrame(
        {"fc_scale": [0.4, 0.5], "f0_scale": [4.0, 5.0], "kd": [1.0, 2.0]},
        index=[1, 2],
    )

    ds = workflows.horton_from_hsg(da_hsg, df_map, da_ksat=da_ksat)

    # fc = ksat * fc_scale (ksat in mm/hr), f0 = fc * f0_scale
    np.testing.assert_allclose(ds["fc"].values, [[0.4, 0.5]], rtol=1e-5)
    np.testing.assert_allclose(ds["f0"].values, [[1.6, 2.5]], rtol=1e-5)
    np.testing.assert_allclose(ds["kd"].values, [[1.0, 2.0]], rtol=1e-6)

    with pytest.raises(ValueError, match="kd"):
        workflows.horton_from_hsg(da_hsg, df_map.drop(columns=["kd"]), da_ksat=da_ksat)


def test_reclassify_hsg_warns_when_recognized_code_missing_from_table(caplog):
    # Drained normalization maps 5 -> 1 and 6 -> 2; only class 1 is in this table.
    da_hsg = _raster([[5, 6]], "hsg", dtype=np.int16)
    df_map = pd.DataFrame({"psi": [90.0], "sigma": [0.2], "ks": [5.0]}, index=[1])

    with caplog.at_level(
        logging.WARNING, logger="hydromt_sfincs.workflows.infiltration"
    ):
        ds = workflows.green_ampt_from_hsg(da_hsg, df_map)

    assert len(caplog.records) == 1
    assert (
        "Recognized HSG codes missing from the reclassification table: [2.0]."
        in caplog.text
    )
    assert ds["psi"].values[0, 0] == 90.0
    assert np.isnan(ds["psi"].values[0, 1])


def test_soil_reclassification_accepts_declared_nodata(caplog):
    da_hsg = _raster([[1, 0]], "hsg", dtype=np.int16)
    da_hsg.raster.set_nodata(0)
    da_hsg = workflows.normalize_hsg_codes(da_hsg)
    da_ksat = _raster([[1.0, 1.0]], "ksat")
    ga_map = pd.DataFrame({"psi": [90.0], "sigma": [0.20]}, index=[1])
    horton_map = pd.DataFrame(
        {"fc_scale": [0.4], "f0_scale": [4.0], "kd": [1.0]}, index=[1]
    )

    with caplog.at_level(logging.WARNING):
        ds_ga = workflows.green_ampt_from_hsg(da_hsg, ga_map, da_ksat=da_ksat)
        ds_horton = workflows.horton_from_hsg(da_hsg, horton_map, da_ksat=da_ksat)

    assert "The nodata value" not in caplog.text
    assert np.isnan(ds_ga["psi"].values[0, 1])
    assert np.isnan(ds_ga["sigma"].values[0, 1])
    assert np.isnan(ds_horton["f0"].values[0, 1])
    assert np.isnan(ds_horton["fc"].values[0, 1])


def test_bucket_from_hsg_derives_storage_and_drainage():
    da_hsg = _raster([[1]], "hsg", dtype=np.int16)
    da_ksat = _raster([[1.0]], "ksat")
    df_map = pd.DataFrame(
        {
            "storage_depth_mm": [120.0],
            "effective_fraction": [0.4],
            "drain_factor": [1.0],
        },
        index=[1],
    )

    ds = workflows.bucket_from_hsg(da_hsg, df_map, da_ksat=da_ksat, bucket_loss=0.1)

    # smax = 120 * 0.4 = 48 mm; residence = 48 / 1.0 h, so k is its reciprocal
    np.testing.assert_allclose(ds["bucket_smax"].values, [[48.0]], rtol=1e-6)
    np.testing.assert_allclose(ds["bucket_k"].values, [[1.0 / 48.0]], rtol=1e-5)
    np.testing.assert_allclose(ds["bucket_loss"].values, [[0.1]], rtol=1e-6)


def test_bucket_from_hsg_uses_residence_time_without_ksat():
    da_hsg = _raster([[1]], "hsg", dtype=np.int16)
    df_map = pd.DataFrame(
        {
            "storage_depth_mm": [120.0],
            "effective_fraction": [0.4],
            "residence_time_hr": [4.0],
        },
        index=[1],
    )

    ds = workflows.bucket_from_hsg(da_hsg, df_map)

    np.testing.assert_allclose(ds["bucket_k"].values, [[0.25]], rtol=1e-6)
    np.testing.assert_allclose(ds["bucket_loss"].values, [[0.0]], rtol=1e-6)

    with pytest.raises(ValueError, match="bucket_k, residence_time_hr, or ksat"):
        workflows.bucket_from_hsg(da_hsg, df_map.drop(columns=["residence_time_hr"]))


def test_constant_infiltration_from_ksat_lulc_spread():
    da_ksat = _raster(
        [[0.02, 0.2, 1.0, 10.0], [0.02, 0.2, 1.0, 10.0]],
        "ksat",
    )
    da_lulc = _raster(
        [[24, 24, 41, 41], [11, 21, 81, 95]],
        "lulc",
        dtype=np.int16,
    )
    da_mask = _raster(np.ones((2, 4), dtype=np.int16), "mask", dtype=np.int16)

    da_qinf = workflows.constant_infiltration_from_ksat_lulc(
        da_ksat,
        da_lulc,
        _modifier_table(),
        da_mask=da_mask,
    )

    values = da_qinf.values
    assert float(np.nanmin(values)) >= 0.01 - 1e-6
    assert float(np.nanmax(values)) <= 19.9 + 1e-6
    assert len(np.unique(np.round(values[np.isfinite(values)], 4))) > 4
    assert float(np.nanmedian(values)) < 19.9
    assert float(da_qinf.values[0, 2]) > float(da_qinf.values[0, 1])
    assert float(da_qinf.values[0, 2]) > float(da_qinf.values[0, 0])


def test_green_ampt_horton_bucket_landuse_modifiers():
    da_hsg = _raster([[6, 6], [6, 6]], "hsg", dtype=np.int16)
    da_lulc = _raster([[41, 24], [81, 95]], "lulc", dtype=np.int16)
    da_ksat = _raster(np.full((2, 2), 1.0, dtype=np.float32), "ksat")

    ga_map = pd.DataFrame(
        {
            "psi": [90.0, 120.0, 150.0, 180.0],
            "sigma": [0.20, 0.25, 0.30, 0.35],
        },
        index=[1, 2, 3, 4],
    )
    horton_map = pd.DataFrame(
        {
            "fc_scale": [0.4, 0.5, 0.6, 0.7],
            "f0_scale": [4.0, 5.0, 6.0, 7.0],
            "kd": [1.0, 2.0, 3.0, 4.0],
        },
        index=[1, 2, 3, 4],
    )
    bucket_map = pd.DataFrame(
        {
            "storage_depth_mm": [120.0, 200.0, 280.0, 360.0],
            "effective_fraction": [0.4, 0.5, 0.6, 0.7],
            "drain_factor": [1.0, 1.5, 2.0, 2.5],
        },
        index=[1, 2, 3, 4],
    )
    modifiers = _modifier_table()

    ds_ga = workflows.green_ampt_from_hsg_landuse(
        da_hsg,
        da_lulc,
        ga_map,
        modifiers,
        da_ksat=da_ksat,
        dual_hsg="drained",
    )
    assert np.isclose(float(ds_ga["psi"].values[0, 0]), 120.0)
    assert np.isclose(float(ds_ga["psi"].values[0, 1]), 120.0)
    assert float(ds_ga["sigma"].values[0, 0]) > float(ds_ga["sigma"].values[0, 1])
    assert float(ds_ga["ks"].values[0, 0]) > float(ds_ga["ks"].values[0, 1])

    ds_horton = workflows.horton_from_hsg_landuse(
        da_hsg,
        da_lulc,
        horton_map,
        modifiers,
        da_ksat=da_ksat,
        dual_hsg="drained",
    )
    assert float(ds_horton["fc"].values[0, 0]) > float(ds_horton["fc"].values[0, 1])
    assert float(ds_horton["f0"].values[0, 0]) > float(ds_horton["f0"].values[0, 1])
    assert float(ds_horton["kd"].values[0, 1]) > float(ds_horton["kd"].values[0, 0])

    ds_bucket = workflows.bucket_from_hsg_landuse(
        da_hsg,
        da_lulc,
        bucket_map,
        modifiers,
        da_ksat=da_ksat,
        dual_hsg="drained",
    )
    assert float(ds_bucket["bucket_smax"].values[0, 0]) > float(
        ds_bucket["bucket_smax"].values[0, 1]
    )
    assert float(ds_bucket["bucket_k"].values[0, 1]) > float(
        ds_bucket["bucket_k"].values[0, 0]
    )
    assert np.allclose(
        ds_bucket["bucket_loss"].values[np.isfinite(ds_bucket["bucket_loss"].values)],
        0.10,
    )


@pytest.mark.parametrize(
    ("from_hsg", "from_hsg_landuse", "df_map", "extra_kwargs"),
    [
        (
            workflows.green_ampt_from_hsg,
            workflows.green_ampt_from_hsg_landuse,
            pd.DataFrame(
                {"psi": [120.0, 240.0], "sigma": [0.25, 0.40]},
                index=[2, 6],
            ),
            {},
        ),
        (
            workflows.horton_from_hsg,
            workflows.horton_from_hsg_landuse,
            pd.DataFrame(
                {
                    "fc_scale": [0.5, 0.8],
                    "f0_scale": [4.0, 6.0],
                    "kd": [2.0, 3.0],
                },
                index=[2, 6],
            ),
            {},
        ),
        (
            workflows.bucket_from_hsg,
            workflows.bucket_from_hsg_landuse,
            pd.DataFrame(
                {
                    "storage_depth_mm": [200.0, 300.0],
                    "effective_fraction": [0.4, 0.5],
                    "drain_factor": [1.5, 2.0],
                },
                index=[2, 6],
            ),
            {"bucket_loss": 0.1},
        ),
    ],
)
@pytest.mark.parametrize("dual_hsg", ["drained", "native"])
def test_dual_hsg_modes_are_consistent_with_and_without_landuse(
    from_hsg, from_hsg_landuse, df_map, extra_kwargs, dual_hsg
):
    da_hsg = _raster(np.full((2, 2), 6), "hsg", dtype=np.int16)
    da_ksat = _raster(np.ones((2, 2)), "ksat")
    da_lulc = _raster(np.full((2, 2), 10), "lulc", dtype=np.int16)
    modifiers = pd.DataFrame(
        {
            "surface_factor": [1.0],
            "storage_factor": [1.0],
            "drainage_factor": [1.0],
        },
        index=[10],
    )
    kwargs = {"da_ksat": da_ksat, "dual_hsg": dual_hsg, **extra_kwargs}

    ds_hsg = from_hsg(da_hsg, df_map, **kwargs)
    ds_landuse = from_hsg_landuse(da_hsg, da_lulc, df_map, modifiers, **kwargs)
    xr.testing.assert_allclose(ds_hsg, ds_landuse)

    if dual_hsg == "native":
        with pytest.raises(ValueError, match="Native dual-HSG values.*missing"):
            from_hsg(da_hsg, df_map.loc[[2]], **kwargs)


def test_modifier_table_supports_non_nlcd_landcover():
    da_hsg = _raster([[2, 2]], "hsg", dtype=np.int16)
    da_ksat = _raster(np.full((1, 2), 1.0, dtype=np.float32), "ksat")
    # ESA WorldCover classes: 10 = tree cover, 50 = built-up
    da_lulc = _raster([[10, 50]], "lulc", dtype=np.int16)
    modifiers = pd.DataFrame(
        {
            "surface_factor": [1.15, 0.20],
            "storage_factor": [1.30, 0.40],
            "drainage_factor": [0.90, 1.40],
        },
        index=[10, 50],
    )
    ga_map = pd.DataFrame({"psi": [90.0, 120.0], "sigma": [0.20, 0.25]}, index=[1, 2])

    ds = workflows.green_ampt_from_hsg_landuse(
        da_hsg, da_lulc, ga_map, modifiers, da_ksat=da_ksat, dual_hsg="drained"
    )
    assert float(ds["sigma"].values[0, 0]) > float(ds["sigma"].values[0, 1])
    assert float(ds["ks"].values[0, 0]) > float(ds["ks"].values[0, 1])


def test_modifier_table_warns_on_unmatched_landcover(caplog):
    da_hsg = _raster([[2, 2]], "hsg", dtype=np.int16)
    da_ksat = _raster(np.full((1, 2), 1.0, dtype=np.float32), "ksat")
    da_lulc = _raster([[10, 999]], "lulc", dtype=np.int16)
    modifiers = pd.DataFrame(
        {
            "surface_factor": [1.15],
            "storage_factor": [1.30],
            "drainage_factor": [0.90],
        },
        index=[10],
    )
    ga_map = pd.DataFrame({"psi": [90.0, 120.0], "sigma": [0.20, 0.25]}, index=[1, 2])

    with caplog.at_level(logging.WARNING):
        ds = workflows.green_ampt_from_hsg_landuse(
            da_hsg, da_lulc, ga_map, modifiers, da_ksat=da_ksat, dual_hsg="drained"
        )

    assert "absent from the modifier table" in caplog.text
    # the unmatched class keeps a neutral factor, so the base value is unchanged
    assert np.isclose(float(ds["sigma"].values[0, 1]), 0.25)


def test_modifier_table_does_not_warn_on_nodata(caplog):
    da_hsg = _raster([[2, 2, 2]], "hsg", dtype=np.int16)
    da_ksat = _raster(np.full((1, 3), 1.0, dtype=np.float32), "ksat")
    da_lulc = _raster([[10, 255, 999]], "lulc", dtype=np.int16)
    da_lulc.raster.set_nodata(255)
    modifiers = pd.DataFrame(
        {
            "surface_factor": [1.15],
            "storage_factor": [1.30],
            "drainage_factor": [0.90],
        },
        index=[10],
    )
    ga_map = pd.DataFrame({"psi": [90.0, 120.0], "sigma": [0.20, 0.25]}, index=[1, 2])

    with caplog.at_level(logging.WARNING):
        ds = workflows.green_ampt_from_hsg_landuse(
            da_hsg, da_lulc, ga_map, modifiers, da_ksat=da_ksat, dual_hsg="drained"
        )

    assert "1 cells hold land-use classes" in caplog.text
    assert "The nodata value" not in caplog.text
    assert np.isclose(float(ds["sigma"].values[0, 1]), 0.25)
    assert np.isclose(float(ds["sigma"].values[0, 2]), 0.25)


def test_bucket_loss_defaults_with_and_without_landuse(model):
    hsg = xr.where(model.grid.data["dep"] < -0.5, 4, 1)
    hsg.raster.set_crs(model.crs)
    ksat = xr.where(model.grid.data["dep"] < 0.0, 0.5, 5.0)
    ksat.raster.set_crs(model.crs)
    lulc = xr.where(model.grid.data["dep"] < -0.5, 24, 41)
    lulc.raster.set_crs(model.crs)

    reclass_table = Path(DATADIR) / "infiltration" / "hsg_bucket.csv"
    df_map = model.data_catalog.get_dataframe(
        reclass_table,
        source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
    )

    ds = workflows.bucket_from_hsg(da_hsg=hsg, df_map=df_map, da_ksat=ksat)
    assert np.isclose(float(ds["bucket_loss"].where(model.grid.mask > 0).mean()), 0.0)

    modifier_table_path = (
        Path(DATADIR) / "infiltration" / "nlcd_infiltration_modifiers.csv"
    )
    df_lulc_modifiers = model.data_catalog.get_dataframe(
        modifier_table_path,
        source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
    )

    ds = workflows.bucket_from_hsg_landuse(
        da_hsg=hsg,
        df_map=df_map,
        da_ksat=ksat,
        da_lulc=lulc,
        df_modifiers=df_lulc_modifiers,
    )
    assert np.isclose(
        float(ds["bucket_loss"].where(model.grid.mask > 0).mean()), 0.10, atol=1e-5
    )


def test_estimator_outputs_declare_nan_nodata_and_keep_missing_ksat():
    # older xarray drops attrs in arithmetic; outputs must still declare NaN
    # nodata so the final average skips cells without Ksat
    da_hsg = _raster([[2, 2], [2, 2]], "hsg", dtype=np.int16)
    da_lulc = _raster([[41, 41], [41, 41]], "lulc", dtype=np.int16)
    da_ksat = _raster([[1.0, 1.0], [1.0, -9999.0]], "ksat")
    horton_map = pd.DataFrame(
        {"fc_scale": [1.0, 1.0], "f0_scale": [4.0, 4.0], "kd": [2.0, 2.0]},
        index=[1, 2],
    )
    bucket_map = pd.DataFrame(
        {
            "storage_depth_mm": [200.0, 200.0],
            "effective_fraction": [0.5, 0.5],
            "drain_factor": [1.0, 1.0],
        },
        index=[1, 2],
    )
    with xr.set_options(keep_attrs=False):
        ds_hor = workflows.horton_from_hsg_landuse(
            da_hsg, da_lulc, horton_map, _modifier_table(), da_ksat=da_ksat
        )
        ds_bkt = workflows.bucket_from_hsg(da_hsg, bucket_map, da_ksat=da_ksat)
    for ds in (ds_hor, ds_bkt):
        for name in ds.data_vars:
            assert np.isnan(ds[name].raster.nodata), name
    assert np.isnan(ds_hor["fc"].values[1, 1])
    assert np.isnan(ds_bkt["bucket_k"].values[1, 1])
    assert float(ds_bkt["bucket_k"].values[0, 0]) > 0.0
