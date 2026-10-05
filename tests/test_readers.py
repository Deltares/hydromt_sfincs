from os.path import join
from pathlib import Path

import numpy as np

from hydromt_sfincs.readers import (
    read_binary_map,
    read_binary_map_index,
    read_geoms,
    read_xy,
)

from .conftest import TESTMODELDIR


def test_read_binary_map(model_config, tmp_dir):
    # get shape from config
    nmax = model_config.config.get("nmax")
    mmax = model_config.config.get("mmax")
    shape = (nmax, mmax)

    # read binary maps
    ind = read_binary_map_index(Path(TESTMODELDIR, "sfincs.ind"))
    msk = read_binary_map(
        Path(TESTMODELDIR, "sfincs.msk"), ind, shape=shape, dtype="u1", mv=0
    )
    assert [v in [0, 1, 2, 3] for v in np.unique(msk)]
    assert ind.max() == ind[-1]


def test_read_geoms():
    # Call the function
    g = read_geoms(Path(TESTMODELDIR, "sfincs.weir"))

    # Assert the data
    assert len(g) == 1
    assert g[0]["name"] == "weir01"
    assert 322500 < np.mean(g[0]["x"]) < 322750
    assert g[0]["elevation"] == [3.5] + [3.0] * 9
    assert g[0]["par1"] == [0.6] * 10


def test_read_xy_index_is_zero_based(tmp_dir):
    """read_xy returns a canonical 0-based index (bnd/src points)."""
    fn = join(tmp_dir, "sfincs.bnd")
    with open(fn, "w") as f:
        f.write("0.0 0.0\n10.0 0.0\n10.0 10.0\n")
    gdf = read_xy(fn, crs=32633)
    assert list(gdf.index) == [0, 1, 2]
