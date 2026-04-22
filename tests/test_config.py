from datetime import datetime
from pathlib import Path
import logging
import os
from os.path import abspath, join

import pytest
from pydantic import ValidationError

from hydromt_sfincs import SfincsModel

TESTDATADIR = join(os.path.dirname(os.path.abspath(__file__)), "data")
TESTMODELDIR = join(TESTDATADIR, "sfincs_test")


def test_config_get_set(model_init, caplog):
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
    with caplog.at_level(logging.WARNING):
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


def test_read_root_locked_after_read(tmp_path):
    """_read_root and _filename are locked to the original root when read() is called."""
    original_root = Path(TESTMODELDIR)
    mod = SfincsModel(root=original_root, mode="r")
    mod.config.read()

    # _read_root should be set and match the original root
    assert hasattr(mod.config, "_read_root")
    assert mod.config._read_root == original_root.resolve()

    # _filename should be an absolute path under the original root
    assert Path(mod.config._filename).is_absolute()
    assert Path(mod.config._filename) == original_root.resolve() / "sfincs.inp"


def test_get_abs_path_uses_read_root_after_root_change(tmp_path):
    """get(abs_path=True) with a fallback resolves against _read_root, not the new root."""
    original_root = Path(TESTMODELDIR)
    mod = SfincsModel(root=original_root, mode="r")
    mod.config.read()

    # Verify the config has obsfile set (relative)
    obs_rel = mod.config.get("obsfile")
    assert obs_rel is not None

    # Before root change: resolves to original root
    obs_abs_before = mod.config.get("obsfile", fallback="sfincs.obs", abs_path=True)
    assert obs_abs_before == (original_root.resolve() / obs_rel)

    # Change root to a new (empty) location
    mod.root.set(tmp_path, mode="r+")

    # After root change: should still resolve against the original read root
    obs_abs_after = mod.config.get("obsfile", fallback="sfincs.obs", abs_path=True)
    assert obs_abs_after == (
        original_root.resolve() / obs_rel
    ), "get(abs_path=True) with fallback should use _read_root after root change"


def test_get_set_file_variable_uses_read_root_after_root_change(tmp_path):
    """get_set_file_variable with no default (read context) resolves against _read_root."""
    original_root = Path(TESTMODELDIR)
    mod = SfincsModel(root=original_root, mode="r")
    mod.config.read()

    obs_rel = mod.config.get("obsfile")
    assert obs_rel is not None

    # Change root to a new (empty) location
    mod.root.set(tmp_path, mode="r+")

    # Pure get (no default) — should resolve against original root
    obs_abs = mod.config.get_set_file_variable("obsfile")
    assert obs_abs == (
        original_root.resolve() / obs_rel
    ), "get_set_file_variable without default should use _read_root after root change"


def test_get_set_file_variable_write_uses_new_root_after_root_change(tmp_path):
    """get_set_file_variable with a default (write context) resolves against the new root."""
    original_root = Path(TESTMODELDIR)
    mod = SfincsModel(root=original_root, mode="r")
    mod.config.read()

    obs_rel = mod.config.get("obsfile")
    assert obs_rel is not None

    # Change root to a new (empty) location
    mod.root.set(tmp_path, mode="r+")

    # Write context (default provided) — should resolve against the NEW root
    obs_abs = mod.config.get_set_file_variable("obsfile", default="sfincs.obs")
    assert obs_abs == (
        tmp_path.resolve() / obs_rel
    ), "get_set_file_variable with default should use the new root after root change"


def test_get_abs_path_write_context_uses_new_root_after_root_change(tmp_path):
    """get(abs_path=True) without fallback (write context) resolves against the new root."""
    original_root = Path(TESTMODELDIR)
    mod = SfincsModel(root=original_root, mode="r")
    mod.config.read()

    # Change root to a new (empty) location
    mod.root.set(tmp_path, mode="r+")

    # No fallback → write context; should use new root
    obs_abs = mod.config.get("obsfile", abs_path=True)
    obs_rel = mod.config.get("obsfile")
    assert obs_abs == (
        tmp_path.resolve() / obs_rel
    ), "get(abs_path=True) without fallback should use the new root after root change"
