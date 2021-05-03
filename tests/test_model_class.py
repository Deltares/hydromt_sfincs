"""Test fiat model class against hydromt.models.model_api"""

import pytest
from os.path import join, dirname, abspath
import numpy as np
import pdb
from click.testing import CliRunner

import hydromt
from hydromt.models import MODELS
from hydromt.cli.cli_utils import parse_config
from hydromt.cli.main import main as hydromt_cli

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_model_class():
    # read model in examples folder
    root = join(EXAMPLEDIR, "fiat_test")
    mod = MODELS.get("fiat")(root=root, mode="r")
    mod.read()
    # run test_model_api() method
    non_compliant_list = mod.test_model_api()
    assert len(non_compliant_list) == 0
    # pass


def test_model_build(tmpdir):
    # test build method
    # compare results with model from examples folder
    model = "fiat"
    root = str(tmpdir.join(model))
    config = join(EXAMPLEDIR, "model_build.ini")
    region = "{'bbox': [11.70, 45.35, 12.95, 46.70]}"
    # Build model
    r = CliRunner().invoke(
        hydromt_cli, ["build", model, root, region, "-i", config, "-vv"]
    )
    assert r.exit_code == 0

    # Compare with model from examples folder
    root0 = join(EXAMPLEDIR, "fiat_test")
    mod0 = MODELS.get(model)(root=root0, mode="r")
    mod0.read()
    mod1 = MODELS.get(model)(root=root, mode="r")
    mod1.read()
    # check maps
    invalid_maps = []
    if len(mod0._staticmaps) > 0:
        maps = mod0.staticmaps.raster.vars
        assert np.all(mod0.crs == mod1.crs), f"map crs {name}"
        for name in maps:
            map0 = mod0.staticmaps[name].fillna(0)
            map1 = mod1.staticmaps[name].fillna(0)
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
