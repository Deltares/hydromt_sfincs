from pathlib import Path

import numpy as np
import pytest

from hydromt_sfincs.readers import (
    read_binary_map,
    read_binary_map_index,
    read_geoms,
)
from hydromt_sfincs.writers import (
    write_binary_map,
    write_binary_map_index,
    write_geoms,
)

from .conftest import TESTMODELDIR


def test_write_binary_map(model_config, tmp_dir):
    # Read the binary maps
    shape = (model_config.config.get("nmax"), model_config.config.get("mmax"))
    ind = read_binary_map_index(Path(TESTMODELDIR, "sfincs.ind"))
    msk = read_binary_map(
        Path(TESTMODELDIR, "sfincs.msk"), ind, shape=shape, dtype="u1", mv=0
    )
    # write binary maps
    fn_out = str(tmp_dir.joinpath("sfincs.ind"))
    write_binary_map_index(fn_out, msk)
    ind1 = read_binary_map_index(fn_out)
    assert np.all(ind == ind1)

    fn_out = str(tmp_dir.joinpath("sfincs.msk"))
    write_binary_map(fn_out, msk, msk, dtype="u1")
    msk1 = read_binary_map(fn_out, ind1, shape=shape, dtype="u1", mv=0)
    assert np.all(msk1 == msk1)


def test_write_geoms(tmp_dir):
    # Read the weirs
    g = read_geoms(Path(TESTMODELDIR, "sfincs.weir"))

    # Write it
    fn_out = str(tmp_dir.joinpath("test.weir"))
    write_geoms(fn_out, g, stype="WEIR")

    # Test the output, roundtrip
    g2 = read_geoms(fn_out)
    for i in range(len(g)):
        assert sorted(g2[i].items()) == sorted(g[i].items())


def test_write_geoms_errors():
    # Read the weirs
    g = read_geoms(Path(TESTMODELDIR, "sfincs.weir"))[0]
    g.pop("elevation")

    # Assert the error while writing
    with pytest.raises(ValueError, match='"elevation" value missing'):
        write_geoms("fail", [g], stype="weir")
