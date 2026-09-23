import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromt_sfincs import workflows


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


def test_normalize_hsg_codes_native_and_invalid():
    da_hsg = _raster([[5, 6, 7, 8]], "hsg", dtype=np.int16)
    np.testing.assert_array_equal(
        workflows.normalize_hsg_codes(da_hsg, mode="native").values,
        np.array([[5, 6, 7, 8]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="dual_hsg"):
        workflows.normalize_hsg_codes(da_hsg, mode="bogus")


def test_ksat_to_mmhr_converts_and_fills_nodata():
    da_ksat = _raster([[1.0, 2.5, np.nan]], "ksat")
    da_ks = workflows.ksat_to_mmhr(da_ksat, factor_ksat=3.6)
    np.testing.assert_allclose(da_ks.values, [[3.6, 9.0, 0.0]], rtol=1e-6)


def test_cn_to_s_units_and_invalid():
    da_cn = _raster([[80.0, 100.0]], "cn")
    np.testing.assert_allclose(workflows.cn_to_s(da_cn).values, [[2.5, 0.0]], rtol=1e-6)
    np.testing.assert_allclose(
        workflows.cn_to_s(da_cn, output_unit="m").values,
        [[2.5 * 0.0254, 0.0]],
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="output_unit"):
        workflows.cn_to_s(da_cn, output_unit="mm")


def test_curve_number_from_landuse_hsg_expected_values():
    df_map = pd.DataFrame({1: [30.0, 77.0], 2: [55.0, 85.0]}, index=[41, 24])
    da_landuse = _raster([[41, 24, 41]], "lulc", dtype=np.int16)
    da_hsg = _raster([[1, 2, 3]], "hsg", dtype=np.int16)

    da_cn = workflows.curve_number_from_landuse_hsg(da_landuse, da_hsg, df_map)

    assert np.isclose(float(da_cn.values[0, 0]), 30.0)
    assert np.isclose(float(da_cn.values[0, 1]), 85.0)
    # HSG 3 is absent from the table, so the combination stays unmapped
    assert np.isnan(da_cn.values[0, 2])


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


def test_green_ampt_from_soil_requires_ks_or_ksat():
    da_hsg = _raster([[1, 2]], "hsg", dtype=np.int16)
    df_map = pd.DataFrame({"psi": [90.0, 120.0], "sigma": [0.20, 0.25]}, index=[1, 2])

    with pytest.raises(ValueError, match="ks or ksat"):
        workflows.green_ampt_from_soil(da_hsg, df_map)

    da_ksat = _raster([[1.0, 2.0]], "ksat")
    ds = workflows.green_ampt_from_soil(da_hsg, df_map, da_ksat=da_ksat)
    np.testing.assert_allclose(ds["psi"].values, [[90.0, 120.0]], rtol=1e-6)
    np.testing.assert_allclose(ds["ks"].values, [[3.6, 7.2]], rtol=1e-6)


def test_horton_from_soil_derives_fc_and_f0():
    da_hsg = _raster([[1, 2]], "hsg", dtype=np.int16)
    da_ksat = _raster([[1.0, 1.0]], "ksat")
    df_map = pd.DataFrame(
        {"fc_scale": [0.4, 0.5], "f0_scale": [4.0, 5.0], "kd": [1.0, 2.0]},
        index=[1, 2],
    )

    ds = workflows.horton_from_soil(da_hsg, df_map, da_ksat=da_ksat)

    # fc = ksat * factor * fc_scale, f0 = fc * f0_scale
    np.testing.assert_allclose(ds["fc"].values, [[1.44, 1.8]], rtol=1e-5)
    np.testing.assert_allclose(ds["f0"].values, [[5.76, 9.0]], rtol=1e-5)
    np.testing.assert_allclose(ds["kd"].values, [[1.0, 2.0]], rtol=1e-6)

    with pytest.raises(ValueError, match="kd"):
        workflows.horton_from_soil(da_hsg, df_map.drop(columns=["kd"]), da_ksat=da_ksat)


def test_bucket_from_soil_derives_storage_and_drainage():
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

    ds = workflows.bucket_from_soil(da_hsg, df_map, da_ksat=da_ksat, bucket_loss=0.1)

    # smax = 120 * 0.4 = 48 mm; residence = 48 / 3.6 h, so k is its reciprocal
    np.testing.assert_allclose(ds["bucket_smax"].values, [[48.0]], rtol=1e-6)
    np.testing.assert_allclose(ds["bucket_k"].values, [[3.6 / 48.0]], rtol=1e-5)
    np.testing.assert_allclose(ds["bucket_loss"].values, [[0.1]], rtol=1e-6)


def test_bucket_from_soil_uses_residence_time_without_ksat():
    da_hsg = _raster([[1]], "hsg", dtype=np.int16)
    df_map = pd.DataFrame(
        {
            "storage_depth_mm": [120.0],
            "effective_fraction": [0.4],
            "residence_time_hr": [4.0],
        },
        index=[1],
    )

    ds = workflows.bucket_from_soil(da_hsg, df_map)

    np.testing.assert_allclose(ds["bucket_k"].values, [[0.25]], rtol=1e-6)
    np.testing.assert_allclose(ds["bucket_loss"].values, [[0.0]], rtol=1e-6)

    with pytest.raises(ValueError, match="bucket_k, residence_time_hr, or ksat"):
        workflows.bucket_from_soil(da_hsg, df_map.drop(columns=["residence_time_hr"]))


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
        factor_ksat=3.6,
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

    ds_ga = workflows.green_ampt_from_soil_landuse(
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

    ds_horton = workflows.horton_from_soil_landuse(
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

    ds_bucket = workflows.bucket_from_soil_landuse(
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

    ds = workflows.green_ampt_from_soil_landuse(
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
        ds = workflows.green_ampt_from_soil_landuse(
            da_hsg, da_lulc, ga_map, modifiers, da_ksat=da_ksat, dual_hsg="drained"
        )

    assert "absent from the modifier table" in caplog.text
    # the unmatched class keeps a neutral factor, so the base value is unchanged
    assert np.isclose(float(ds["sigma"].values[0, 1]), 0.25)


def test_bucket_loss_defaults_split_between_legacy_and_landuse(model):
    hsg = xr.where(model.grid.data["dep"] < -0.5, 4, 1)
    hsg.raster.set_crs(model.crs)
    ksat = xr.where(model.grid.data["dep"] < 0.0, 0.5, 5.0)
    ksat.raster.set_crs(model.crs)
    lulc = xr.where(model.grid.data["dep"] < -0.5, 24, 41)
    lulc.raster.set_crs(model.crs)

    model.infiltration.create_bucket_from_soil(hsg=hsg, ksat=ksat)
    assert np.isclose(
        model.grid.data["bucket_loss"].where(model.grid.mask > 0).mean(),
        0.0,
        atol=1e-6,
    )

    model.infiltration.create_bucket_from_soil(hsg=hsg, ksat=ksat, lulc=lulc)
    assert np.isclose(
        model.grid.data["bucket_loss"].where(model.grid.mask > 0).mean(),
        0.10,
        atol=1e-5,
    )
