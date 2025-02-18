import pytest
from datetime import datetime
from hydromt.model import Model
from hydromt_sfincs.config import SfincsInput, SfincsInputVariables


def test_SfincsInput_initialization():
    model = Model()
    config = SfincsInput(model)

    assert isinstance(config, SfincsInput)
    assert isinstance(config.data, SfincsInputVariables)
    assert config.data.mmax == 10  # Default value check
    assert config.data.nmax == 10


def test_SfincsInput_get_set():
    model = Model()
    config = SfincsInput(model)

    # set a new value and get it
    config.set("mmax", 20)
    assert config.get("mmax") == 20

    # set a string with integer values
    config.set("mmax", "50")
    assert config.get("mmax") == 50

    # now set a string with txt
    with pytest.raises(TypeError):
        config.set("mmax", "text")

    # set a new values with type text
    config.set("outputformat", "ascii")
    assert config.get("outputformat") == "ascii"

    # set a non-existing key
    # Should raise KeyError for invalid attribute
    with pytest.raises(KeyError):
        config.set("invalid_key", 100)


def test_SfincsInput_io(tmpdir):
    # Initialize the configuration
    config0 = SfincsInput(Model)  # initialize with default values
    fn_out = str(tmpdir.join("sfincs.inp"))

    inpdict = {
        "mmax": 84,
        "nmax": 36,
        "dx": 150,
        "dy": 150,
        "x0": 318650.0,
        "y0": 5034000.0,
        "rotation": 27.0,
        "epsg": 32633,
    }
    config0.update(inpdict)

    # check if the values are set correctly
    for key, value in inpdict.items():
        assert config0.get(key) == value

    # now test the read/write
    config0.write(fn_out)

    config1 = SfincsInput(Model)
    config1.read(fn_out)
    assert config0.data == config1.data


def test_SfincsInput_default_datetime():
    model = Model()
    config = SfincsInput(model)

    assert isinstance(config.get("tref"), datetime)
    assert config.get("tref").year == 2010
    assert config.get("tref").month == 2
    assert config.get("tref").day == 1
