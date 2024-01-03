"""add global fixtures"""

from os.path import abspath, dirname, join

import pytest

from hydromt_sfincs.sfincs import SfincsModel
from hydromt import DataCatalog

TESTDATADIR = join(dirname(abspath(__file__)), "data")
TESTMODELDIR = join(TESTDATADIR, "sfincs_test")


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
def mod(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    mod.set_root(str(tmpdir), mode="r+")
    return mod

@pytest.fixture
def data_catalog():
    return DataCatalog('artifact_data')

@pytest.fixture
def hydrography(data_catalog):
    bbox = [12.64, 45.48, 12.82, 45.59]
    ds_hydro = data_catalog.get_rasterdataset("merit_hydro", variables=['flwdir', 'uparea', 'basins'], bbox=bbox).load()
    da_mask = (ds_hydro['basins'] == 210000039).astype(int)
    da_mask.raster.set_crs(ds_hydro.raster.crs)
    da_mask.raster.set_nodata(0)
    gdf_mask = da_mask.raster.vectorize()
    return ds_hydro['flwdir'], ds_hydro['uparea'], gdf_mask
