"""Test sfincs model class against hydromt.models.model_api"""

import pytest
from os.path import join, dirname, abspath, isfile
import numpy as np
import xarray as xr
import pandas as pd

from hydromt.cli.cli_utils import parse_config
from hydromt.log import setuplog
from hydromt_sfincs.sfincs import SfincsModel

from .conftest import TESTDATADIR, TESTMODELDIR

_cases = {
    "test1": {
        "ini": "sfincs_test.yml",
        "example": "sfincs_test",
    },
}


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_model_class(case):
    # read model in examples folder
    root = join(TESTDATADIR, _cases[case]["example"])
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    # run test_model_api() method
    non_compliant_list = mod.test_model_api()
    assert len(non_compliant_list) == 0
    # pass


def test_states(tmpdir):
    root = TESTMODELDIR
    fn = "sfincs.zsini"
    mod = SfincsModel(root=root, mode="r+")
    mod.read()
    # create dummy state and set to states
    mask = mod.grid["dep"] < -0.5
    zsini = xr.where(mask, 0.5, -9999.0)
    zsini.raster.set_nodata(-9999.0)
    zsini.raster.set_crs(mod.crs)
    mod.set_states(zsini, "zsini")

    tmp_root = str(tmpdir.join("restart_test"))
    mod.set_root(tmp_root, mode="w")
    # write and check if isfile
    mod.write_states()
    mod.write_config()
    assert isfile(join(mod.root, fn))
    # read and check if identical
    mod1 = SfincsModel(root=tmp_root, mode="r")
    assert np.allclose(mod1.states["zsini"], mod.states["zsini"])


def test_infiltration(tmpdir):
    # FIXME: very shallow test, add more specific tests
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    mod.set_root(str(tmpdir.join("infiltration_test")), mode="w")

    # set constant infiltration
    qinf = xr.where(mod.grid["dep"] < -0.5, -9999, 0.1)
    qinf.raster.set_nodata(-9999.0)
    qinf.raster.set_crs(mod.crs)
    mod.setup_constant_infiltration(qinf, reproj_method="nearest")
    assert "qinf" not in mod.config  # qinf removed from config
    assert "qinffile" in mod.config
    assert "qinf" in mod.grid

    # set cn infiltration
    cn = xr.where(mod.grid["dep"] < -0.5, 0, 50)
    cn.raster.set_nodata(-1)
    cn.raster.set_crs(mod.crs)
    mod.setup_cn_infiltration(cn, reproj_method="nearest")
    assert "scsfile" in mod.config
    assert "scs" in mod.grid
    assert (mod.grid["scs"].where(mod.mask > 0)).min() == 10

    # set cn infiltration with recovery
    lulc = xr.where(mod.grid["dep"] < -0.5, 70, 30)
    hsg = xr.where(mod.grid["dep"] < 2, 1, 3)
    ksat = xr.where(mod.grid["dep"] < 1, 0.01, 0.2)
    # create pandas reclass table for lulc and hsg to cn
    reclass_table = pd.DataFrame([[0, 35], [0, 56]], index=[70, 30], columns=[1, 3])
    effective = 0.5
    mod.setup_cn_infiltration_with_kr(
        lulc=lulc, hsg=hsg, ksat=ksat, reclass_table=reclass_table, effective=effective
    )

    assert "smax" in mod.grid
    assert "seff" in mod.grid
    assert "kr" in mod.grid

    mod.write_grid()
    mod.write_config()

    # read and check if identical
    mod1 = SfincsModel(root=mod.root, mode="r")

    # assure the sum of smax is close to earlier calculated value
    assert np.isclose(mod1.grid["smax"].where(mod.mask > 0).sum(), 32.929287)
    assert np.isclose(
        mod1.grid["seff"].where(mod.mask > 0).sum(), 32.929287 * effective
    )
    assert np.isclose(mod1.grid["kr"].where(mod.mask > 0).sum(), 330.588)


def test_structs(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r+")
    # read
    mod.set_config("thdfile", "sfincs.thd")
    mod.read_grid()
    mod.read_geoms()
    assert "thd" in mod.geoms
    # write thd file only
    tmp_root = str(tmpdir.join("struct_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_geoms(data_vars=["thd"])
    assert isfile(join(mod.root, "sfincs.thd"))
    assert not isfile(join(mod.root, "sfincs.obs"))
    fn_thd_gis = join(mod.root, "gis", "thd.geojson")
    assert isfile(fn_thd_gis)
    # add second thd file
    mod.setup_structures(fn_thd_gis, stype="thd")
    assert len(mod.geoms["thd"].index) == 2
    # setup weir file from thd.geojson using dz option
    with pytest.raises(ValueError, match="Weir structure requires z"):
        mod.setup_structures(fn_thd_gis, stype="weir")
    mod.setup_structures(fn_thd_gis, stype="weir", dz=2)
    assert "weir" in mod.geoms
    assert "weirfile" in mod.config
    mod.write_geoms()
    assert isfile(join(mod.root, "sfincs.weir"))
    # test with buffer
    mod.setup_structures(fn_thd_gis, stype="weir", buffer=5, dep="dep", merge=False)
    assert len(mod.geoms["weir"].index) == 2


def test_drainage_structures(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r+")
    # read
    mod.set_config("drnfile", "sfincs.drn")
    mod.read_grid()
    mod.read_geoms()
    assert "drn" in mod.geoms
    nr_drainage_structures = len(mod.geoms["drn"].index)
    # write drn file only
    tmp_root = str(tmpdir.join("drainage_struct_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_geoms(data_vars=["drn"])
    assert isfile(join(mod.root, "sfincs.drn"))
    assert not isfile(join(mod.root, "sfincs.obs"))
    fn_drn_gis = join(mod.root, "gis", "drn.geojson")
    assert isfile(fn_drn_gis)
    # add more drainage structures
    mod.setup_drainage_structures(fn_drn_gis, merge=True)
    assert len(mod.geoms["drn"].index) == nr_drainage_structures * 2


def test_observations(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r+")
    # read
    mod.set_config("obsfile", "sfincs.obs")
    mod.read_grid()
    mod.read_geoms()

    # observation points
    assert "obs" in mod.geoms
    nr_observation_points = len(mod.geoms["obs"].index)
    # write obs file only
    tmp_root = str(tmpdir.join("observation_points_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_geoms(data_vars=["obs"])
    assert isfile(join(mod.root, "sfincs.obs"))
    assert not isfile(join(mod.root, "sfincs.crs"))
    fn_obs_gis = join(mod.root, "gis", "obs.geojson")
    assert isfile(fn_obs_gis)
    # add more observation points
    mod.setup_observation_points(fn_obs_gis, merge=True)
    assert len(mod.geoms["obs"].index) == nr_observation_points * 2

    # observation lines
    assert "crs" in mod.geoms
    nr_observation_lines = len(mod.geoms["crs"].index)
    # write crs file only
    tmp_root = str(tmpdir.join("observation_lines_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_geoms(data_vars=["crs"])
    assert isfile(join(mod.root, "sfincs.crs"))
    assert not isfile(join(mod.root, "sfincs.obs"))
    fn_crs_gis = join(mod.root, "gis", "crs.geojson")
    assert isfile(fn_crs_gis)
    # add more observation lines
    mod.setup_observation_lines(fn_crs_gis, merge=True)
    assert len(mod.geoms["crs"].index) == nr_observation_lines * 2


def test_results():
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r")
    assert all([v in mod.results for v in ["zs", "zsmax", "inp"]])


def test_plots(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    mod.set_root(str(tmpdir.join("plots_test")))
    mod.plot_forcing(fn_out="forcing.png")
    assert isfile(join(mod.root, "figs", "forcing.png"))
    mod.plot_basemap(fn_out="basemap.png")
    assert isfile(join(mod.root, "figs", "basemap.png"))


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_model_build(tmpdir, case):
    # compare results with model from examples folder
    root = str(tmpdir.join(case))
    root0 = TESTMODELDIR

    # Build model
    ini_fn = join(TESTDATADIR, _cases[case]["ini"])
    opt = parse_config(ini_fn)
    logger = setuplog(path=join(root, "hydromt.log"), log_level=10)
    mod1 = SfincsModel(root=root, mode="w", logger=logger, **opt.pop("global", {}))
    mod1.build(opt=opt)
    # Check if model is api compliant
    non_compliant_list = mod1.test_model_api()
    assert len(non_compliant_list) == 0

    # read and compare with model from examples folder
    mod0 = SfincsModel(root=root0, mode="r")
    mod0.read()
    mod1 = SfincsModel(root=root, mode="r")
    mod1.read()
    # check maps
    invalid_maps = []
    if len(mod0._staticmaps) > 0:
        assert np.all(mod0.crs == mod1.crs), f"map crs"
        for name in mod0.staticmaps.raster.vars:
            map0 = mod0.staticmaps[name]
            map1 = mod1.staticmaps[name]
            if not np.allclose(map0, map1):
                invalid_maps.append(name)
    invalid_map_str = ", ".join(invalid_maps)
    assert len(invalid_maps) == 0, f"invalid maps: {invalid_map_str}"
    # check geoms
    if mod0._staticgeoms:
        for name in mod0.staticgeoms:
            geom0 = mod0.staticgeoms[name]
            geom1 = mod1.staticgeoms[name]
            assert geom0.index.size == geom1.index.size and np.all(
                geom0.index == geom1.index
            ), f"geom index {name}"
            assert geom0.columns.size == geom1.columns.size and np.all(
                geom0.columns == geom1.columns
            ), f"geom columns {name}"
            assert geom0.crs == geom1.crs, f"geom crs {name}"
            assert np.all(geom0.geometry == geom1.geometry), f"geom {name}"
    # check forcing
    if mod0._forcing:
        for name in mod0.forcing:
            assert np.allclose(
                mod0.forcing[name], mod1.forcing[name]
            ), f"forcing {name}"
    # check config
    if mod0._config:
        # flatten
        assert mod0._config == mod1._config, f"config mismatch"
    # check forcing
    if mod0._forcing:
        for name in mod0.forcing:
            assert np.allclose(mod0.forcing[name], mod1.forcing[name])
