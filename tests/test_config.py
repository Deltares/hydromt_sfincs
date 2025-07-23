from datetime import datetime
from pathlib import Path
import os
from os.path import abspath, join

import pytest
from pydantic import ValidationError

from hydromt_sfincs import SfincsModel


def test_config_get_set(model_init):
    config = model_init.config

    # set a new value and get it
    config.set("mmax", 20)
    assert config.get("mmax") == 20

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

    # assert tref corresponds to current year
    current_year = datetime.now().year

    assert isinstance(config.get("tref"), datetime)
    assert config.get("tref").year == current_year

def test_get_set_file_variable(model_config):
    # test 3 situations of how function get_set_file_variable could be used

    # read existing obsfile in model.root
    config = model_config.config

    varname = "obsfile"

    # 2) variable 'key' already in config
    obs0 = config.get(varname)  # = sfincs.obs
    file_path = config.get_set_file_variable(
        key=varname, value=None, default="sfincs.obs"
    )

    # check if 'key' in config is unchanged
    obs1 = config.get(varname)  # = sfincs.obs
    assert obs0 == obs1

    # check if root added to returned file_path
    assert file_path == Path(abspath(join(config.root.path, obs1)))
    # assert file_path == join(config.root.path, obs1)

    # add obsfile as random full path
    random_location = "c:/random/file/location/sfincs.obs"
    config.set(varname, random_location)  # because of c:/ it is a 'plausible' one

    # see whether it is returned, in case it's already set in the config
    file_path = config.get_set_file_variable(
        key=varname, value=None, default="sfincs.obs"
    )
    assert file_path == Path(random_location)

    # and check whether it has been updated in the config
    obs2 = config.get(varname)  # = sfincs.obs
    assert random_location == obs2

    # 3) use default name and root if not yet in config
    # clear obsfile
    config.set(varname, None)

    # call without input
    file_path = config.get_set_file_variable(
        key=varname, value=None, default="sfincs.obs"
    )

    # check whether added to config
    obs3 = config.get(varname)  # = sfincs.obs
    assert obs3 == "sfincs.obs"

    # and whether path is correct
    assert file_path == Path(abspath(join(config.root.path, "sfincs.obs")))

    # 1) input file variable 'key' is given as input

    # first give in a file_name without path
    tmpvalue = "sfincs_test.obs"
    file_path = config.get_set_file_variable(
        key=varname, value=tmpvalue, default="sfincs.obs"
    )

    # check whether added to config
    obs4 = config.get(varname)
    assert obs4 == tmpvalue

    # and whether path is correct
    assert file_path == Path(abspath(join(config.root.path, tmpvalue)))

    # now give in a path, where the directory includes the model root
    tmppath = join(config.root.path, "sfincs_test.obs")
    file_path = config.get_set_file_variable(
        key=varname, value=tmppath, default="sfincs.obs"
    )

    # check whether added to config, without the full path
    obs5 = config.get(varname)
    assert obs5 == tmpvalue

    # and whether path is correct
    assert file_path == Path(abspath(tmppath))

    # and finally check giving in a random path, different than root
    file_path = config.get_set_file_variable(
        key=varname, value=random_location, default="sfincs.obs"
    )

    # and check whether it has been updated in the config
    obs6 = config.get(varname)
    # check whether output as string is without the double backslashes \\
    assert random_location == obs6

    # and whether path is correct
    assert file_path == Path(random_location)