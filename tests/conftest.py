"""add global fixtures"""

from os.path import abspath, dirname, join

import pytest
import numpy as np

from hydromt_sfincs.sfincs import SfincsModel
from hydromt_sfincs.regulargrid import RegularGrid

TESTDATADIR = join(dirname(abspath(__file__)), "data")
TESTMODELDIR = join(TESTDATADIR, "sfincs_test")


@pytest.fixture
def reggrid():
    # create a simple regular grid
    grid = RegularGrid(
        x0=318650,
        y0=5040000,
        dx=150,
        dy=150,
        nmax=84,  # height
        mmax=36,  # width
    )
    return grid


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
def mod(tmpdir):
    root = TESTMODELDIR
    mod = SfincsModel(root=root, mode="r")
    mod.read()
    mod.set_root(str(tmpdir), mode="r+")
    return mod
