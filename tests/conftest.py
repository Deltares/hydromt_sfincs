"""add global fixtures"""

from os.path import abspath, dirname, join
import numpy as np
import tempfile
from pathlib import Path

import pytest
import numpy as np

from hydromt import DataCatalog
from hydromt_sfincs.sfincs import SfincsModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
TESTMODELDIR = join(TESTDATADIR, "sfincs_test")

local_data_yaml = join(TESTDATADIR, "local_data.yml")


@pytest.fixture()
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture
def data_catalog():
    return DataCatalog("artifact_data")


# initialize a model instance in write mode in a temporary directory
@pytest.fixture
def model_init(tmp_path):
    mod = SfincsModel(root=tmp_path, mode="w+", data_libs=["artifact_data"])
    return mod


# initialize a model instance with configuration read
@pytest.fixture
def model_config():
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r", data_libs=["artifact_data", local_data_yaml])
    mod.config.read()
    return mod


# read full model instance and set to write mode in a temporary directory
@pytest.fixture
def model(tmp_dir):
    root = join(TESTDATADIR, "sfincs_test")
    mod = SfincsModel(root=root, mode="r", data_libs=["artifact_data", local_data_yaml])
    mod.read()
    mod.root.set(tmp_dir, mode="r+")
    return mod


@pytest.fixture
def quadtree_model(tmp_dir):
    root = join(TESTDATADIR, "sfincs_test_quadtree")
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    mod.root.set(tmp_dir, mode="r+")
    return mod


# create a simple regular grid similar to sfincs_test
@pytest.fixture
def reggrid(model_config):
    grid_params = {
        "mmax": 36,
        "nmax": 84,
        "dx": 150,
        "dy": 150,
        "x0": 318650.0,
        "y0": 5040000.0,
        "rotation": 27.0,
        "epsg": 32633,
    }

    # create the grid (note this actually calls model.reggrid.create)
    model_config.grid.create(**grid_params)
    return model_config.grid


# create a random mask
@pytest.fixture
def mask(reggrid):
    # create a simple mask
    mask = np.zeros((reggrid.nmax, reggrid.mmax), dtype="u1")
    mask[2:, 3:-1] = 1
    return mask


@pytest.fixture
def weirs():
    feats = [
        {
            "name": "WEIR01",
            "x": [0, 10, 20],
            "y": [100, 100, 100],
            "z": 5.0,
            "par1": 0.6,
        },
        {
            "x": [100, 110, 120],
            "y": [100, 100, 100],
            "z": [5.0, 5.1, 5.0],
            "par1": 0.6,
        },
    ]
    return feats


@pytest.fixture
def hydrography(data_catalog):
    bbox = [12.64, 45.48, 12.82, 45.59]
    ds_hydro = data_catalog.get_rasterdataset(
        "merit_hydro", variables=["flwdir", "uparea", "basins"], bbox=bbox
    ).load()
    da_mask = (ds_hydro["basins"] == 210000039).astype(np.uint8)
    da_mask.raster.set_crs(ds_hydro.raster.crs)
    da_mask.raster.set_nodata(0)
    gdf_mask = da_mask.raster.vectorize()
    return ds_hydro["flwdir"], ds_hydro["uparea"], gdf_mask
