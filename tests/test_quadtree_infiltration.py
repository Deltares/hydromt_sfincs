"""Test sfincs model class against hydromt.models.model_api"""

import numpy as np
import pandas as pd
import xarray as xr

# from hydromt.log import setuplog

from hydromt_sfincs.sfincs import SfincsModel


def test_quadtree_infiltration(model, quadtree_model):
    # set constant infiltration based on regular grid model elevation
    qinf = xr.where(model.grid.data["dep"] < -0.5, -9999, 0.1)
    qinf.raster.set_nodata(-9999.0)
    qinf.raster.set_crs(model.crs)

    # add to quadtree model
    quadtree_model.quadtree_infiltration.create_constant(qinf, reproj_method="nearest")
    assert quadtree_model.config.get("qinf") is None  # qinf removed from config
    assert quadtree_model.config.get("infiltrationfile") is not None  # qinf file set
    assert (
        quadtree_model.config.get("infiltrationtype") == "c2d"
    )  # infiltration type set to c2d
    assert "qinf" in quadtree_model.quadtree_grid.data
    assert (
        quadtree_model.quadtree_grid.data["qinf"]
        .where(quadtree_model.quadtree_grid.mask > 0)
        .max()
        == 0.1
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
    assert (
        quadtree_model.quadtree_grid.data["scs"].where(
            quadtree_model.quadtree_grid.mask > 0
        )
    ).max() == 10

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
        72.46283083,
    )
    assert np.isclose(
        mod1.quadtree_grid.data["seff"].where(mod1.quadtree_grid.mask > 0).sum(),
        72.46283083 * effective,
    )
    assert np.isclose(
        mod1.quadtree_grid.data["ks"].where(mod1.quadtree_grid.mask > 0).sum(),
        733.27316619,
    )
