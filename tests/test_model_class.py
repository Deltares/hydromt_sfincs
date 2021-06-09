"""Test sfincs model class against hydromt.models.model_api"""

from posixpath import basename
import pytest
from os.path import join, dirname, abspath, isfile
import numpy as np
import xarray as xr
import pdb

import hydromt
from hydromt.cli.cli_utils import parse_config
from hydromt_sfincs.sfincs import SfincsModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")

_cases = {
    "coastal": {
        "ini": "sfincs_coastal.ini",
        "region": {"bbox": [12.00, 45.35, 12.80, 45.65]},
        "res": 100,
        "example": "sfincs_coastal",
    },
    "riverine": {
        "ini": "sfincs_riverine.ini",
        "region": {"bbox": [11.975098, 45.786918, 12.271385, 45.938019]},
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
    assert isinstance(mod.states["zs"], xr.DataArray)
    tmp_root = str(tmpdir.join("restart_test"))
    mod.set_root(tmp_root, mode="w")
    # write and check if isfile
    mod.write_states()
    mod.write_config()
    assert isfile(join(mod.root, fn))
    # read and check if identical
    mod1 = SfincsModel(root=tmp_root, mode="r")
    assert np.allclose(mod1.states["zs"], mod.states["zs"])


def test_results():
    root = join(EXAMPLEDIR, _cases["riverine"]["example"])
    mod = SfincsModel(root=root, mode="r")
    assert np.all([v in mod.results for v in ["zs", "zsmax", "hmax", "inp"]])


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
    mod1 = SfincsModel(root=root, mode="w", **opt.pop("global", {}))
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
    # check config
    if mod0._config:
        # flatten
        assert mod0._config == mod1._config, f"config mismatch"
    # check forcing
    if mod0._forcing:
        for name in mod0.forcing:
            assert np.allclose(mod0.forcing[name], mod1.forcing[name])
