from pathlib import Path

import pytest

from hydromt_sfincs.readers import read_config


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
