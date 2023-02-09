"""Test sfincs model class against hydromt.models.model_api"""

import pytest
from os.path import join, dirname, abspath, isfile
import numpy as np
import xarray as xr

from hydromt.cli.cli_utils import parse_config
from hydromt.log import setuplog
from hydromt_sfincs.sfincs import SfincsModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")

_cases = {
    "coastal": {
        "ini": "sfincs_coastal.ini",
        "region": {"bbox": [12.05, 45.30, 12.85, 45.65]},
        "res": 150,
        "example": "sfincs_coastal",
    },
    "riverine": {
        "ini": "sfincs_riverine.ini",
        "region": {"bbox": [11.97, 45.78, 12.28, 45.94]},
        "res": 50,
        "example": "sfincs_riverine",
    },
}


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_model_class(case):
    # read model in examples folder
    root = join(EXAMPLEDIR, _cases[case]["example"])
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    # run test_model_api() method
    non_compliant_list = mod.test_model_api()
    assert len(non_compliant_list) == 0
    # pass


def test_states(tmpdir):
    root = join(EXAMPLEDIR, _cases["riverine"]["example"])
    fn = "sfincs.restart"
    mod = SfincsModel(root=root, mode="r+")
    mod.set_config("inifile", fn)
    # read and check if DataArray
    assert isinstance(mod.states["zsini"], xr.DataArray)
    tmp_root = str(tmpdir.join("restart_test"))
    mod.set_root(tmp_root, mode="w")
    # write and check if isfile
    mod.write_states()
    mod.write_config()
    assert isfile(join(mod.root, fn))
    # read and check if identical
    mod1 = SfincsModel(root=tmp_root, mode="r")
    assert np.allclose(mod1.states["zsini"], mod.states["zsini"])


def test_structs(tmpdir):
    root = join(EXAMPLEDIR, _cases["riverine"]["example"])
    mod = SfincsModel(root=root, mode="r+")
    # read
    mod.set_config("thdfile", "sfincs.thd")
    mod.read_staticmaps()
    mod.read_staticgeoms()
    assert "thd" in mod.staticgeoms
    # write thd file only
    mod._staticgeoms = {"thd": mod.staticgeoms["thd"]}
    tmp_root = str(tmpdir.join("struct_test"))
    mod.set_root(tmp_root, mode="w")
    mod.write_staticgeoms()
    assert isfile(join(mod.root, "sfincs.thd"))
    fn_thd_gis = join(mod.root, "gis", "thd.geojson")
    assert isfile(fn_thd_gis)
    # add second thd file
    mod.setup_structures(fn_thd_gis, stype="thd")
    assert len(mod.staticgeoms["thd"].index) == 2
    # setup weir file from thd.geojson using dz option
    with pytest.raises(ValueError, match="Weir structure requires z"):
        mod.setup_structures(fn_thd_gis, stype="weir")
    mod.setup_structures(fn_thd_gis, stype="weir", dz=2)
    assert "weir" in mod.staticgeoms
    assert "weirfile" in mod.config
    mod.write_staticgeoms()
    assert isfile(join(mod.root, "sfincs.weir"))


def test_results():
    root = join(EXAMPLEDIR, _cases["riverine"]["example"])
    mod = SfincsModel(root=root, mode="r")
    assert np.all([v in mod.results for v in ["zs", "zsmax", "hmax", "inp"]])


def test_plots(tmpdir):
    root = join(EXAMPLEDIR, _cases["riverine"]["example"])
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    mod.set_root(str(tmpdir.join("plots_test")))
    mod.plot_forcing()
    assert isfile(join(mod.root, "figs", "forcing.png"))
    mod.plot_basemap()
    assert isfile(join(mod.root, "figs", "basemap.png"))


@pytest.mark.parametrize("case", list(_cases.keys()))
def test_model_build(tmpdir, case):
    # compare results with model from examples folder
    root = str(tmpdir.join(case))
    root0 = join(EXAMPLEDIR, _cases[case]["example"])

    # Build model
    ini_fn = join(EXAMPLEDIR, _cases[case]["ini"])
    region = _cases[case]["region"]
    res = _cases[case]["res"]
    opt = parse_config(ini_fn)
    logger = setuplog(path=join(root, "hydromt.log"), log_level=10)
    mod1 = SfincsModel(root=root, mode="w", logger=logger, **opt.pop("global", {}))
    mod1.build(region=region, res=res, opt=opt)
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
