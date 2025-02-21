from datetime import datetime
import os
import pytest

from hydromt_sfincs import SfincsModel


def test_config_get_set(model_init):
    config = model_init.config

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
    assert model0.config.data == model1.config.data


def test_config_datetime(model_init):
    config = model_init.config

    assert isinstance(config.get("tref"), datetime)
    assert config.get("tref").year == 2010
    assert config.get("tref").month == 2
    assert config.get("tref").day == 1
