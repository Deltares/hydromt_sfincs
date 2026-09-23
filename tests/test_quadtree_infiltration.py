"""Test sfincs model class against hydromt.models.model_api"""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
import xarray as xr

# from hydromt.log import setuplog

from hydromt_sfincs.sfincs import SfincsModel


def test_quadtree_infiltration(model, quadtree_model):
    # set constant infiltration based on regular grid model elevation
    qinf = xr.where(model.grid.data["dep"] < -0.5, 0, 0.1)
    qinf.raster.set_nodata(np.nan)
    qinf.raster.set_crs(model.crs)

    # add to quadtree model
    quadtree_model.quadtree_infiltration.create_constant(qinf, reproj_method="nearest")
    assert quadtree_model.config.get("qinf") is None  # qinf removed from config
    assert quadtree_model.config.get("infiltrationfile") is not None  # qinf file set
    assert (
        quadtree_model.config.get("infiltrationtype") == "c2d"
    )  # infiltration type set to c2d
    assert "qinf" in quadtree_model.quadtree_grid.data
    assert np.isclose(
        float(
            quadtree_model.quadtree_grid.data["qinf"]
            .where(quadtree_model.quadtree_grid.mask > 0)
            .max()
        ),
        0.1,
    )

    # set cn infiltration based on regular grid model elevation
    cn = xr.where(model.grid.data["dep"] < -0.5, 100, 50)
    cn.raster.set_nodata(-1)
    cn.raster.set_crs(model.crs)
    quadtree_model.quadtree_infiltration.create_cn(cn, reproj_method="nearest")
    assert (
        quadtree_model.config.get("infiltrationtype") == "cna"
    )  # infiltration type set to cna
    assert "scs" in quadtree_model.quadtree_grid.data
    assert np.isclose(
        float(
            quadtree_model.quadtree_grid.data["scs"]
            .where(quadtree_model.quadtree_grid.mask > 0)
            .max()
        ),
        10,
    )

    # set cn infiltration with recovery based on regular grid model elevation
    lulc = xr.where(model.grid.data["dep"] < -0.5, 70, 30)
    lulc.raster.set_crs(model.crs)
    hsg = xr.where(model.grid.data["dep"] < 2, 1, 3)
    hsg.raster.set_crs(model.crs)
    ksat = xr.where(model.grid.data["dep"] < 1, 0.01, 0.2)
    ksat.raster.set_crs(model.crs)
    # create pandas reclass table for lulc and hsg to cn
    reclass_table = pd.DataFrame([[0, 35], [0, 56]], index=[70, 30], columns=[1, 3])
    effective = 0.5
    quadtree_model.quadtree_infiltration.create_cn_with_recovery(
        lulc=lulc, hsg=hsg, ksat=ksat, reclass_table=reclass_table, effective=effective
    )

    # Check if variables are there
    assert "smax" in quadtree_model.quadtree_grid.data
    assert "seff" in quadtree_model.quadtree_grid.data
    assert "ks" in quadtree_model.quadtree_grid.data
    assert (
        quadtree_model.config.get("infiltrationtype") == "cnb"
    )  # infiltration type set to cnb

    # Write model
    quadtree_model.quadtree_grid.write()
    quadtree_model.config.write()

    # read and check if identical
    mod1 = SfincsModel(root=quadtree_model.root.path, mode="r")
    mod1.config.read()
    mod1.quadtree_grid.read()

    # assure the sum of smax is close to earlier calculated value
    assert np.isclose(
        mod1.quadtree_grid.data["smax"].where(mod1.quadtree_grid.mask > 0).sum(),
        73.48772,
    )
    assert np.isclose(
        mod1.quadtree_grid.data["seff"].where(mod1.quadtree_grid.mask > 0).sum(),
        73.48772 * effective,
    )
    assert np.isclose(
        mod1.quadtree_grid.data["ks"].where(mod1.quadtree_grid.mask > 0).sum(),
        736.79039946,
    )


def test_cn_from_landuse_hsg_quadtree(model, quadtree_model):
    lulc = xr.where(model.grid.data["dep"] < -0.5, 70, 30)
    lulc.raster.set_crs(model.crs)
    hsg = xr.where(model.grid.data["dep"] < 2, 1, 3)
    hsg.raster.set_crs(model.crs)
    reclass_table = pd.DataFrame([[0, 35], [0, 56]], index=[70, 30], columns=[1, 3])

    # left at the default reproj_method so the default stays exercised
    quadtree_model.quadtree_infiltration.create_cn_from_landuse_hsg(
        lulc=lulc,
        hsg=hsg,
        reclass_table=reclass_table,
    )

    assert quadtree_model.config.get("infiltrationtype") == "cna"
    assert "scs" in quadtree_model.quadtree_grid.data
    assert np.isclose(
        float(
            quadtree_model.quadtree_grid.data["scs"]
            .where(quadtree_model.quadtree_grid.mask > 0)
            .max()
        ),
        7.857143,
        atol=1e-3,
    )


def test_infiltration_estimators_from_hsg_quadtree(model, quadtree_model):
    # HSG 4 in the deep cells and HSG 1 elsewhere; Ksat 0.5 below MSL, 5.0 above
    hsg = xr.where(model.grid.data["dep"] < -0.5, 4, 1)
    hsg.raster.set_crs(model.crs)
    ksat = xr.where(model.grid.data["dep"] < 0.0, 0.5, 5.0)
    ksat.raster.set_crs(model.crs)

    infil = quadtree_model.quadtree_infiltration
    grid = quadtree_model.quadtree_grid

    def _max(name):
        return float(grid.data[name].where(grid.mask > 0).max())

    # psi/sigma are reclassified from hsg_green_ampt.csv; ks is ksat * 3.6
    infil.create_green_ampt_from_soil(hsg=hsg, ksat=ksat)
    assert quadtree_model.config.get("infiltrationtype") == "gai"
    assert _max("psi") == pytest.approx(316.3, rel=1e-3)  # HSG 4
    assert _max("sigma") == pytest.approx(0.35, rel=1e-3)  # HSG 1
    assert _max("ks") == pytest.approx(18.0, rel=1e-3)  # 5.0 * 3.6

    # fc_scale is 1.0 for every HSG so fc == ks; f0 = fc * f0_scale (4.0 for HSG 1)
    infil.create_horton_from_soil(hsg=hsg, ksat=ksat)
    assert quadtree_model.config.get("infiltrationtype") == "hor"
    assert "psi" not in grid.data
    assert _max("fc") == pytest.approx(18.0, rel=1e-3)
    assert _max("f0") == pytest.approx(72.0, rel=1e-3)
    assert _max("kd") == pytest.approx(4.14, rel=1e-3)  # HSG 1

    # bucket_smax = storage_depth_mm * effective_fraction, 250 * 0.35 for HSG 1
    infil.create_bucket_from_soil(hsg=hsg, ksat=ksat)
    assert quadtree_model.config.get("bucketfile") is not None
    assert _max("bucket_smax") == pytest.approx(87.5, rel=1e-3)
    assert _max("bucket_k") == pytest.approx(18.0 / 87.5, rel=1e-3)
    assert _max("bucket_loss") == pytest.approx(0.0, abs=1e-6)


def test_get_vars_by_infiltration_type(quadtree_model):
    infil = quadtree_model.quadtree_infiltration

    fake_data = {
        "qinf": None,
        "scs": None,
        "smax": None,
        "seff": None,
        "ks": None,
    }

    # patch internal storage
    quadtree_model.quadtree_grid._data = fake_data

    write_vars, remove_vars = infil.get_vars_by_infiltration_type("cnb")

    assert set(write_vars) == {"smax", "seff", "ks"}
    assert set(remove_vars) == {"qinf", "scs"}
