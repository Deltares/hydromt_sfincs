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
    return pd.DataFrame(
        {
            "surface_factor": [0.05, 0.70, 0.45, 0.20, 0.85, 1.15, 1.00, 0.90, 0.55],
            "storage_factor": [0.20, 0.75, 0.60, 0.40, 0.80, 1.30, 1.00, 1.10, 0.70],
            "drainage_factor": [1.00, 1.15, 1.25, 1.40, 1.05, 0.90, 1.00, 1.05, 0.80],
        },
        index=[
            "water",
            "urban_low",
            "urban_med",
            "urban_high",
            "barren",
            "forest",
            "shrub_grass",
            "crops",
            "wetlands",
        ],
    )


def test_normalize_hsg_codes_drained():
    da_hsg = _raster([[1, 2, 3, 4], [5, 6, 7, 8]], "hsg", dtype=np.int16)
    da_norm = workflows.normalize_hsg_codes(da_hsg, mode="drained")
    np.testing.assert_array_equal(
        da_norm.values,
        np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32),
    )


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


def test_bucket_loss_defaults_split_between_legacy_and_landuse(model):
    hsg = xr.where(model.grid.data["dep"] < -0.5, 4, 1)
    hsg.raster.set_crs(model.crs)
    ksat = xr.where(model.grid.data["dep"] < 0.0, 0.5, 5.0)
    ksat.raster.set_crs(model.crs)
    lulc = xr.where(model.grid.data["dep"] < -0.5, 24, 41)
    lulc.raster.set_crs(model.crs)

    model.infiltration.create_bucket(hsg=hsg, ksat=ksat)
    assert np.isclose(
        model.grid.data["bucket_loss"].where(model.grid.mask > 0).mean(),
        0.0,
        atol=1e-6,
    )

    model.infiltration.create_bucket(hsg=hsg, ksat=ksat, lulc=lulc)
    assert np.isclose(
        model.grid.data["bucket_loss"].where(model.grid.mask > 0).mean(),
        0.10,
        atol=1e-5,
    )
