from pathlib import Path

import numpy as np
import pytest

from hydromt_sfincs.readers import (
    read_binary_map,
    read_binary_map_index,
    read_config,
    read_geoms,
)

from .conftest import TESTMODELDIR


def test_read_config(config_path: Path):
    # Call the function:
    inp = read_config(filename=config_path)

    # Assert the output\
    assert inp["mmax"] == 84
    assert inp["nmax"] == 36
    assert "depfile" in inp
    assert "inifile" not in inp
    assert int(inp["zsini"]) == 0


def test_read_config_errors(tmp_path: Path, config_path: Path):
    p = Path(tmp_path, "sfincs.inp")
    # With a nonsense path
    with pytest.raises(
        FileNotFoundError,
        match=f"SFINCS input file '{p.as_posix()}' does not exist.",
    ):
        _ = read_config(filename=p)

    # Read and alter the data
    with open(config_path, "r") as reader:
        data = reader.read()
    data = data.replace("20100201 000000", "foo")
    with open(p, "w") as writer:
        writer.write(data)
    # Read with a nonsense time value
    with pytest.raises(ValueError, match='"tref = foo" not understood.'):
        _ = read_config(filename=p)


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
