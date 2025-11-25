from datetime import datetime
from pathlib import Path
import os
from os.path import abspath, join

import pytest
from pydantic import ValidationError

from hydromt_sfincs import SfincsModel


def test_config_get_set(model_init):
    config = model_init.config

    # check that a variable initiated as None is set correctly
    assert config.get("mmax") == None

    # set a new value and get it
    config.set("mmax", 20)
    assert config.get("mmax") == 20

    # check that another variable that has an preset variable is loaded correctly
    assert config.get("advection") == 1

    # set value out of bounds
    with pytest.raises(ValidationError):
        config.set("mmax", -1000)

    # now set a string with txt
    with pytest.raises(ValidationError):
        config.set("mmax", "text")

    # set a new values with type text
    config.set("outputformat", "ascii")
    assert config.get("outputformat") == "ascii"

    # set a non-existing key
    # Should raise KeyError for invalid attribute
    with pytest.raises(KeyError):
        config.set("invalid_key", 100)


def test_config_io(tmp_path):
    # Start with model initialized with default values
    model0 = SfincsModel(root=tmp_path, mode="w+")

    # update the configuration with new values
    inpdict = {
        "mmax": 84,
        "nmax": 36,
        "dx": 150,
        "dy": 150,
        "x0": 318650.0,
        "y0": 5034000.0,
        "rotation": 27.0,
        "epsg": 32633,
        "crsgeo": 0,
    }
    model0.config.update(inpdict)

    # check if the values are set correctly
    for key, value in inpdict.items():
        assert model0.config.get(key) == value

    # now test the read/write
    model0.config.write()

    # check if the file is written
    assert os.path.isfile(os.path.join(tmp_path, "sfincs.inp"))

    # now read the configuration again
    model1 = SfincsModel(root=tmp_path, mode="r")
    model1.config.read()

    d0 = model0.config.data.model_dump()
    d1 = model1.config.data.model_dump()

    diff = {
        k: (d0.get(k), d1.get(k))
        for k in d0.keys() | d1.keys()
        if d0.get(k) != d1.get(k)
    }

    assert not diff, f"Differences:\n{diff}"

    # write config including descriptions
    model1.config.write(filename="sfincs_with_description.inp", write_description=True)

    # read ascii file tmp_path/sfincs.inp
    with open(os.path.join(tmp_path, "sfincs.inp"), "r", encoding="ascii") as file:
        # Read the contents of the file
        contents = file.read()
    with open(
        os.path.join(tmp_path, "sfincs_with_description.inp"), "r", encoding="ascii"
    ) as file:
        # Read the contents of the file
        contents1 = file.read()

    # Files should differ because of descriptions
    assert contents != contents1


def test_config_datetime(model_init):
    config = model_init.config

    # assert tref corresponds to current year
    current_year = datetime.now().year

    assert isinstance(config.get("tref"), datetime)
    assert config.get("tref").year == current_year

    # now set a datestr instead of datetime
    datestr = "20100201 000000"  # YYYYMMDD HHMMSS
    config.set("tref", datestr)

    # check if it is converted to datetime
    assert isinstance(config.get("tref"), datetime)
    assert config.get("tref").year == 2010


def test_get_set_file_variable(model_config, tmp_dir):
    """Test get_set_file_variable with cross-platform paths."""

    config = model_config.config
    varname = "obsfile"

    # 1. Variable already in config ---
    obs0 = config.get(varname)  # e.g., "sfincs.obs"
    file_path = config.get_set_file_variable(
        key=varname, value=None, default="sfincs.obs"
    )

    obs1 = config.get(varname)
    assert obs0 == obs1

    # Path should include model root
    expected_path = Path(config.root.path) / obs1
    assert Path(file_path).resolve().as_posix() == expected_path.resolve().as_posix()

    # 2. Add obsfile as random absolute path ---
    random_location = str(tmp_dir / "sfincs.obs")
    config.set(varname, random_location)  # store string for Pydantic

    file_path = config.get_set_file_variable(
        key=varname, value=None, default="sfincs.obs"
    )
    assert (
        Path(file_path).resolve().as_posix()
        == Path(random_location).resolve().as_posix()
    )
    assert config.get(varname) == random_location

    # 3. Use default name if not yet in config ---
    config.set(varname, None)
    file_path = config.get_set_file_variable(
        key=varname, value=None, default="sfincs.obs"
    )
    obs3 = config.get(varname)
    assert obs3 == "sfincs.obs"

    expected_path = Path(config.root.path) / "sfincs.obs"
    assert Path(file_path).resolve().as_posix() == expected_path.resolve().as_posix()

    # 4. Input variable given as file name ---
    tmpvalue = "sfincs_test.obs"
    file_path = config.get_set_file_variable(
        key=varname, value=tmpvalue, default="sfincs.obs"
    )
    obs4 = config.get(varname)
    assert obs4 == tmpvalue

    expected_path = Path(config.root.path) / tmpvalue
    assert Path(file_path).resolve().as_posix() == expected_path.resolve().as_posix()

    # 5. Input variable given as full path inside root ---
    tmppath = Path(config.root.path) / "sfincs_test.obs"
    file_path = config.get_set_file_variable(
        key=varname, value=str(tmppath), default="sfincs.obs"
    )
    obs5 = config.get(varname)
    assert obs5 == tmpvalue  # config stores just the file name
    assert Path(file_path).resolve().as_posix() == tmppath.resolve().as_posix()

    # 6. Input variable given as random path outside root ---
    file_path = config.get_set_file_variable(
        key=varname, value=random_location, default="sfincs.obs"
    )
    obs6 = config.get(varname)

    obs6_path = Path(obs6).resolve().as_posix()
    random_location_path = Path(random_location).resolve().as_posix()

    assert obs6_path == random_location_path
    assert Path(file_path).resolve().as_posix() == random_location_path
