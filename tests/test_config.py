import logging
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from hydromt_sfincs import LATEST_SFINCS_VERSION, SfincsModel
from hydromt_sfincs.components.config.config_variables import (
    SfincsConfigVariables,
    _read_config,
)

TESTDATADIR = Path(__file__).resolve().parent / "data"
TESTMODELDIR = TESTDATADIR / "sfincs_test"


def test_write_default_config(tmp_path):
    data = SfincsConfigVariables()
    inpfile = tmp_path / "sfincs.inp"
    data.write(inpfile)
    assert inpfile.is_file()
    # check that the file contains expected keys
    keys = {line.split()[0] for line in inpfile.read_text().splitlines()}
    # all keys with always=True in the model fields should be present
    expected_keys = {
        k
        for k in type(data).model_fields
        if (type(data).model_fields[k].json_schema_extra or {}).get("always", False)
    }
    assert keys.issubset(expected_keys)


def test_config_get_set(caplog):
    data = SfincsConfigVariables()

    # check that a variable initiated as None is set correctly
    assert data.mmax is None

    # set a new value and get it
    data = data.set_value("mmax", 20)
    assert data.mmax == 20

    # check that another variable that has an preset variable is loaded correctly
    assert data.advection == 1

    # set value out of bounds
    with pytest.raises(ValidationError):
        data.set_value("mmax", -1000)

    # now set a string with txt
    with pytest.raises(ValidationError):
        data.set_value("mmax", "text")

    # set a new values with type text
    data = data.set_value("outputformat", "ascii")
    assert data.outputformat == "ascii"

    # set a non-existing key
    with caplog.at_level(logging.WARNING):
        data.set_value("invalid_key", 100)


def test_config_set_skip_validation():
    data = SfincsConfigVariables()

    data = data.set_value("mmax", -1000, skip_validation=True)

    assert data.mmax == -1000


def test_config_explicit_only(tmp_path):
    data = SfincsConfigVariables().set_values(
        {"theta": 1.0, "mmax": 10, "custom_key": "custom value"}
    )

    explicit_data = data.to_dict(explicit_only=True)
    assert set(explicit_data) == {"theta", "mmax", "custom_key"}
    assert explicit_data["theta"] == 1.0
    assert explicit_data["mmax"] == 10

    inpfile = tmp_path / "sfincs.inp"
    data.write(inpfile, explicit_only=True)
    keys = {line.split()[0] for line in inpfile.read_text().splitlines()}

    assert keys == {"theta", "mmax", "custom_key"}


@pytest.mark.parametrize(
    ("updates", "expected_key"),
    [
        ({"mmax": 10, "qtrfile": "sfincs.quadtree"}, "qtrfile"),
        ({"mmax": 10, "qtrfile": None}, "mmax"),
        ({"dtwave": 900.0, "snapwave": 0}, "snapwave"),
        ({"dtwave": 900.0, "snapwave": 1}, "dtwave"),
    ],
)
def test_config_condition_controls_written_fields(tmp_path, updates, expected_key):
    data = SfincsConfigVariables().set_values(updates)
    inpfile = tmp_path / "sfincs.inp"

    data.write(inpfile)
    keys = {line.split()[0] for line in inpfile.read_text().splitlines()}

    assert expected_key in keys


def test_config_reads_kernel_keywords_without_warning(tmp_path, caplog):
    inpfile = tmp_path / "sfincs.inp"
    values = {
        "urbfile": "sfincs.urb",
        "dkbfile": "sfincs.dkb",
        "vegfile": "sfincs.veg.nc",
        "storetzsmax": "1",
        "storezvolmax": "0",
        "store_river_discharge": "1",
        "wavemaker_nfreqs_inc": "50",
        "snapwave": "1",
        "snapwave_gamma": "0.8",
        "snapwave_Tinc2ig": "6.0",
    }
    inpfile.write_text("".join(f"{k} = {v}\n" for k, v in values.items()))

    with caplog.at_level(logging.WARNING):
        data = SfincsConfigVariables.read(inpfile)

    assert "Unrecognized" not in caplog.text
    assert data.urbfile == "sfincs.urb"
    assert data.storezvolmax == 0
    assert data.snapwave_gamma == 0.8

    outfile = tmp_path / "out" / "sfincs.inp"
    data.write(outfile)
    keys = {line.split()[0] for line in outfile.read_text().splitlines()}
    assert set(values) <= keys


def test_config_io(tmp_path):
    # Start with default values
    data0 = SfincsConfigVariables()

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
    data0 = data0.set_values(inpdict)

    # check if the values are set correctly
    for key, value in inpdict.items():
        assert getattr(data0, key) == value

    # now test the read/write
    inpfile = tmp_path / "sfincs.inp"
    data0.write(inpfile)

    # check if the file is written
    assert inpfile.is_file()

    # now read the configuration again
    data1 = SfincsConfigVariables.read(inpfile)

    d0 = data0.to_dict()
    d1 = data1.to_dict()

    diff = {
        k: (d0.get(k), d1.get(k))
        for k in d0.keys() | d1.keys()
        if d0.get(k) != d1.get(k)
    }

    assert not diff, f"Differences:\n{diff}"

    # write config including descriptions
    inpfile_desc = tmp_path / "sfincs_with_description.inp"
    data1.write(inpfile_desc, write_description=True)

    contents = inpfile.read_text(encoding="ascii")
    contents1 = inpfile_desc.read_text(encoding="ascii")

    # Files should differ because of descriptions
    assert contents != contents1


def test_config_read_write_roundtrip(tmp_path, caplog):
    # legacy (unversioned) sfincs.inp with keys equal to schema defaults that are
    # NOT flagged 'always' (fields flagged 'always' get force-written regardless of
    # input and are omitted here) plus the legacy 'dtout' keyword of 'dtmapout' (see
    # examples/missing_inp_values.py)
    inp_str = """rotation             = 0
epsg                 = 32617
latitude             = 0.0
crsgeo               = 0
dtout                = 3600.0
dtrstout             = 0.0
theta                = 1.0
dtmax                = 60.0
manning              = 0.04
manning_land         = 0.04
manning_sea          = 0.02
rgh_lev_land         = 0.0
gapres               = 101200.0
inputformat          = bin
outputformat         = net
"""
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(inp_str)

    data = SfincsConfigVariables.read(inpfile)

    test_inpfile = tmp_path / "sfincs.inp.test"
    data.write(test_inpfile)
    test_inp_str = test_inpfile.read_text()

    def _keys(text):
        return [
            line.split()[0]
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    # keywords (including legacy ones) are written as read
    original_keys = set(_keys(inp_str))
    original_fields = {SfincsConfigVariables.resolve_key(key) for key in original_keys}
    test_keys = set(_keys(test_inp_str))

    # fields flagged 'always' (with a non-None default) are force-written even
    # though absent from this legacy file, except those with a min_version, since
    # the SFINCS version of an unversioned file is unknown
    model_fields = SfincsConfigVariables.model_fields
    expected_extras = {
        key
        for key, field in model_fields.items()
        if key not in original_fields
        and (field.json_schema_extra or {}).get("always")
        and "min_version" not in (field.json_schema_extra or {})
        and field.default is not None
    }

    assert not original_keys - test_keys, "missing keys after round-trip"
    assert (
        test_keys - original_keys == expected_extras
    ), "unexpected extra keys after round-trip"

    # an unversioned file keeps its legacy keywords, and gets no sfincs_version
    assert "dtout" in test_keys
    assert "dtmapout" not in test_keys
    assert "sfincs_version" not in test_keys

    # an unrecognized key in the source file should trigger a warning, but be kept
    with open(inpfile, "a") as fid:
        fid.write("some_unknown_key   = 1\n")
    with caplog.at_level(logging.WARNING):
        data2 = SfincsConfigVariables.read(inpfile)
    assert "some_unknown_key" in caplog.text

    test_inpfile2 = tmp_path / "sfincs.inp.test2"
    data2.write(test_inpfile2)
    assert "some_unknown_key" in test_inpfile2.read_text()


def test_config_read_keeps_legacy_keywords_valid_for_version(tmp_path, caplog):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(
        "sfincs_version       = 2.4.0\n"
        "dtout                = 1800  # legacy map interval\n"
        "crs                  = 32633\n"
    )

    with caplog.at_level(logging.WARNING):
        data = SfincsConfigVariables.read(inpfile)
    assert data.dtmapout == 1800
    assert data.epsg == 32633
    # 'dtout' is read by SFINCS 2.4.0; 'crs' never was
    assert "'dtout'" not in caplog.text
    assert "'crs' is not valid for sfincs_version 2.4.0" in caplog.text

    output = tmp_path / "migrated.inp"
    data.write(output, write_comments=True, explicit_only=True)
    keys = {line.split()[0] for line in output.read_text().splitlines()}
    assert keys == {"sfincs_version", "dtout", "epsg"}
    assert "# legacy map interval" in output.read_text()


@pytest.mark.parametrize(
    ("sfincs_version", "expected_keys"),
    [
        (
            "2.5.0",
            {
                "wavemaker_wvmfile",
                "wavemaker_hmin",
                "global_model",
                "output_on_quadtree_mesh",
                "wavemaker_nfreqs_inc",
                "inffile",
                "inftype",
                "vegfile",
                "vegtype",
            },
        ),
        (
            "2.4.1",
            {
                "wvmfile",
                "wmhmin",
                "global",
                "regular_output_on_mesh",
                "storeqdrain",
                "nfreqsinc",
                "infiltration_file",
                "infiltration_type",
                "vegetationfile",
                "vegetationtype_toml",
            },
        ),
        (
            "2.3.2",
            {
                "wvmfile",
                "wmhmin",
                "global",
                "regular_output_on_mesh",
                "storeqdrain",
                "nfreqsinc",
                "infiltration_file",
                "infiltration_type",
                "vegetationfile",
                "vegetationtype_toml",
            },
        ),
    ],
)
def test_config_read_migrates_renamed_keywords(tmp_path, sfincs_version, expected_keys):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(
        f"sfincs_version       = {sfincs_version}\n"
        "wvmfile              = sfincs.wvm\n"
        "wmhmin               = 0.2\n"
        "global               = 1\n"
        "regular_output_on_mesh = 1\n"
        "storeqdrain          = 1\n"
        "nfreqsinc            = 50\n"
        "infiltration_file    = sfincs.inf.nc\n"
        "infiltration_type    = cna\n"
        "vegetationfile       = sfincs.veg.nc\n"
        "vegetationtype_toml  = vegetation.toml\n"
    )

    data = SfincsConfigVariables.read(inpfile)
    output = tmp_path / "migrated.inp"
    data.write(output, explicit_only=True)
    keys = {line.split()[0] for line in output.read_text().splitlines()}

    assert keys == expected_keys | {"sfincs_version"}


@pytest.mark.parametrize(
    ("sfincs_version", "expected_key"),
    [(None, "global"), ("2.5.0", "global_model"), ("2.4.1", "global")],
)
def test_config_legacy_alias_written_by_version(
    tmp_path, caplog, sfincs_version, expected_key
):
    inpfile = tmp_path / "sfincs.inp"
    version_line = f"sfincs_version = {sfincs_version}\n" if sfincs_version else ""
    inpfile.write_text(version_line + "global = 1\n")

    with caplog.at_level(logging.WARNING):
        data = SfincsConfigVariables.read(inpfile)
    assert "Unrecognized" not in caplog.text
    assert data.global_model == 1

    output = tmp_path / "out.inp"
    data.write(output)
    keys = {line.split()[0] for line in output.read_text().splitlines()}
    other_key = "global" if expected_key == "global_model" else "global_model"
    assert expected_key in keys
    assert other_key not in keys


def test_config_get_set_legacy_alias(tmp_path):
    data = SfincsConfigVariables().set_values({"wvmfile": "sfincs.wvm"})
    assert data.wavemaker_wvmfile == "sfincs.wvm"
    assert "wvmfile" not in (data.model_extra or {})
    assert SfincsConfigVariables.resolve_key("wvmfile") == "wavemaker_wvmfile"
    assert SfincsConfigVariables.resolve_key("precipfile") == "prcfile"
    assert SfincsConfigVariables.resolve_key("mmax") == "mmax"


def test_config_read_migration_prefers_new_name_by_version(tmp_path):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(
        "sfincs_version       = 2.4.0\n"
        "dtout                = 1800  # legacy interval\n"
        "dtmapout             = 3600  # current interval\n"
        "crs                  = 32633\n"
    )

    data = SfincsConfigVariables.read(inpfile)

    assert data.dtmapout == 3600
    assert data.epsg == 32633
    output = tmp_path / "migrated.inp"
    data.write(output, write_comments=True, explicit_only=True)
    output_text = output.read_text()
    assert "dtmapout             = 3600" in output_text
    assert "# current interval" in output_text
    assert "legacy interval" not in output_text
    assert "epsg" in output_text


def test_config_read_warns_and_removes_unsupported_fields(tmp_path, caplog):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(
        "sfincs_version       = 2.0.0\n"
        "urbfile              = sfincs.urb  # unsupported file\n"
        "storetzsmax          = 1\n"
        "custom_key           = keep\n"
    )

    with caplog.at_level(logging.WARNING):
        data = SfincsConfigVariables.read(inpfile)
    assert "below the minimum supported" in caplog.text

    assert data.urbfile is None
    assert data.storetzsmax == 0
    assert data.model_extra["custom_key"] == "keep"

    output = tmp_path / "filtered.inp"
    data.write(output, explicit_only=True)
    output_text = output.read_text()
    assert "urbfile" not in output_text
    assert "storetzsmax" not in output_text
    assert "custom_key" in output_text


def test_config_read_migrates_unsupported_alias(tmp_path, caplog):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(
        "sfincs_version       = 2.5.0\n"
        "dtout                = 1800  # legacy interval\n"
    )

    with caplog.at_level(logging.WARNING):
        data = SfincsConfigVariables.read(inpfile)
    assert "replaced by 'dtmapout'" in caplog.text

    assert data.dtmapout == 1800

    output = tmp_path / "migrated.inp"
    data.write(output, write_comments=True, explicit_only=True)
    output_text = output.read_text()
    assert "dtmapout             = 1800" in output_text
    assert "dtout" not in output_text
    assert "# legacy interval" in output_text


def test_config_read_migration_prefers_supported_current_name(tmp_path):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(
        "sfincs_version       = 2.4.0\n"
        "dtout                = 1800\n"
        "dtmapout             = 3600  # current interval\n"
    )

    data = SfincsConfigVariables.read(inpfile)

    assert data.dtmapout == 3600


def test_config_unversioned_keeps_legacy_keywords(tmp_path, caplog):
    # without sfincs_version, every keyword is written as read, except those that
    # predate the minimum supported SFINCS version ('crs')
    inp_str = (
        "dtout                = 1800.0\n"
        "crs                  = 32633\n"
        "precipfile           = sfincs.precip\n"
        "amrfile              = sfincs.ampr.nc\n"
        "wvmfile              = sfincs.wvm\n"
        "infiltrationfile     = sfincs.inf.nc\n"
        "waveforces_ratio     = 0.5\n"
        "snapwave             = 1\n"
    )
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(inp_str)

    with caplog.at_level(logging.WARNING):
        data = SfincsConfigVariables.read(inpfile)
    assert data.wavemaker_wvmfile == "sfincs.wvm"
    assert data.snapwave_waveforces_ratio == 0.5
    assert "'crs' predates the minimum supported" in caplog.text
    assert "'wvmfile'" not in caplog.text

    output = tmp_path / "out.inp"
    data.write(output, explicit_only=True)
    keys = [line.split()[0] for line in output.read_text().splitlines()]
    expected = {line.split()[0] for line in inp_str.splitlines()} - {"crs"}
    assert set(keys) == expected | {"epsg"}


@pytest.mark.parametrize(
    ("sfincs_version", "keyword", "expected_key"),
    [
        # each alias is written back as read when the version reads it ...
        (None, "snapwave_waveforces_factor", "snapwave_waveforces_factor"),
        (None, "waveforces_ratio", "waveforces_ratio"),
        ("2.3.3", "snapwave_waveforces_factor", "snapwave_waveforces_factor"),
        # ... and otherwise as the keyword of that version
        ("2.3.3", "waveforces_ratio", "snapwave_waveforces_factor"),
        ("2.4.1", "snapwave_waveforces_factor", "snapwave_waveforces_ratio"),
        ("2.4.1", "waveforces_ratio", "snapwave_waveforces_ratio"),
        ("2.4.1", "precipfile", "precipfile"),
    ],
)
def test_config_multiple_aliases_written_as_read(
    tmp_path, sfincs_version, keyword, expected_key
):
    inpfile = tmp_path / "sfincs.inp"
    version_line = f"sfincs_version = {sfincs_version}\n" if sfincs_version else ""
    value = "sfincs.prcp" if "file" in keyword else "0.5"
    inpfile.write_text(version_line + f"snapwave = 1\n{keyword} = {value}\n")

    data = SfincsConfigVariables.read(inpfile)
    output = tmp_path / "out.inp"
    data.write(output, explicit_only=True)
    keys = {line.split()[0] for line in output.read_text().splitlines()}
    assert keys - {"sfincs_version", "snapwave"} == {expected_key}


@pytest.mark.parametrize(
    ("used", "ignored"),
    [
        ("prcfile", "precipfile"),
        ("infiltrationfile", "infiltration_file"),
        ("infiltrationtype", "infiltration_type"),
        ("snapwave_waveforces_factor", "waveforces_ratio"),
    ],
)
def test_config_duplicate_aliases_follow_kernel_order(tmp_path, caplog, used, ignored):
    # with several aliases of one field in a file, the one the kernel looks up
    # first is used, regardless of the order in the file
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(f"snapwave = 1\n{ignored} = 0.3\n{used} = 0.5\n")

    with caplog.at_level(logging.WARNING):
        data = SfincsConfigVariables.read(inpfile)
    assert f"'{ignored}' is ignored" in caplog.text
    key = SfincsConfigVariables.resolve_key(used)
    assert data._explicit_keys[key] == used
    assert float(getattr(data, key)) == 0.5


def test_config_read_keeps_deprecated_field_unversioned(tmp_path, caplog):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text("storeqdrain = 1\n")

    # the targeted SFINCS version is unknown: kept without a warning
    with caplog.at_level(logging.WARNING):
        data = SfincsConfigVariables.read(inpfile)
    assert "storeqdrain" not in caplog.text

    output = tmp_path / "out.inp"
    data.write(output, explicit_only=True)
    assert "storeqdrain" in output.read_text()


def test_config_set_legacy_keyword_for_latest_version(caplog):
    data = SfincsConfigVariables().set_value("sfincs_version", LATEST_SFINCS_VERSION)

    with caplog.at_level(logging.WARNING):
        data = data.set_value("wvmfile", "sfincs.wvm")
    assert "replaced by 'wavemaker_wvmfile'" in caplog.text
    assert data._explicit_keys["wavemaker_wvmfile"] == "wavemaker_wvmfile"

    # without sfincs_version, the keyword is kept as set
    data = SfincsConfigVariables().set_value("wvmfile", "sfincs.wvm")
    assert data._explicit_keys["wavemaker_wvmfile"] == "wvmfile"


def test_config_write_update_to_latest(tmp_path):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(
        "dtout                = 1800.0  # map interval\n"
        "wvmfile              = sfincs.wvm\n"
        "global               = 1\n"
        "storeqdrain          = 1\n"
    )
    data = SfincsConfigVariables.read(inpfile)

    output = tmp_path / "latest.inp"
    data.write(output, write_comments=True, explicit_only=True, update_to_latest=True)
    text = output.read_text()
    keys = {line.split()[0] for line in text.splitlines()}
    assert keys == {"sfincs_version", "dtmapout", "wavemaker_wvmfile", "global_model"}
    assert f"sfincs_version       = {LATEST_SFINCS_VERSION}" in text
    assert "# map interval" in text

    # the configuration itself is not changed
    assert data.sfincs_version is None
    output = tmp_path / "as_read.inp"
    data.write(output, explicit_only=True)
    keys = {line.split()[0] for line in output.read_text().splitlines()}
    assert keys == {"dtout", "wvmfile", "global", "storeqdrain"}


def test_new_model_writes_sfincs_version(tmp_path):
    model = SfincsModel(root=tmp_path, mode="w")
    model.config.write()
    assert (
        f"sfincs_version       = {LATEST_SFINCS_VERSION}"
        in (tmp_path / "sfincs.inp").read_text()
    )


def test_read_config_raw(config_path: Path):
    # _read_config() only parses raw strings; type coercion is done by
    # SfincsConfigVariables.read() via pydantic
    inp, comments = _read_config(filename=config_path)

    assert inp["mmax"] == "84"
    assert inp["nmax"] == "36"
    assert "depfile" in inp
    assert "inifile" not in inp
    assert inp["zsini"] == "0.0"
    assert isinstance(comments, dict)


def test_read_config_raw_errors(tmp_path: Path):
    p = tmp_path / "sfincs.inp"
    with pytest.raises(
        FileNotFoundError,
        match=f"SFINCS input file '{p.as_posix()}' does not exist.",
    ):
        _read_config(filename=p)


def test_config_read_write_preserves_inline_comments(tmp_path):
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text(
        "mmax                 = 84  # custom comment for mmax\n"
        "nmax                 = 36\n"
    )

    data = SfincsConfigVariables.read(inpfile)

    test_inpfile = tmp_path / "sfincs.inp.test"
    data.write(test_inpfile, write_description=True, write_comments=True)
    test_inp_str = test_inpfile.read_text()

    # the original inline comment takes priority over the schema description
    assert "# custom comment for mmax" in test_inp_str
    assert "Number of grid cells in x-direction" not in test_inp_str
    # a field without an inline comment falls back to the schema description
    assert "Number of grid cells in y-direction" in test_inp_str

    # write_comments=False ignores the inline comment, even if present
    test_inpfile2 = tmp_path / "sfincs.inp.test2"
    data.write(test_inpfile2, write_description=True, write_comments=False)
    test_inp_str2 = test_inpfile2.read_text()
    assert "# custom comment for mmax" not in test_inp_str2
    assert "Number of grid cells in x-direction" in test_inp_str2

    # write_description=False, write_comments=False: no trailing comments at all
    test_inpfile3 = tmp_path / "sfincs.inp.test3"
    data.write(test_inpfile3)
    assert "#" not in test_inpfile3.read_text()


def test_config_read_invalid_datetime(tmp_path):
    # read_config() only parses raw strings; datetime validation happens in
    # SfincsConfigVariables.read() via pydantic
    inpfile = tmp_path / "sfincs.inp"
    inpfile.write_text("tref = foo\n")

    with pytest.raises(ValidationError):
        SfincsConfigVariables.read(inpfile)


def test_config_datetime():
    data = SfincsConfigVariables()

    # assert tref corresponds to current year
    current_year = datetime.now().year

    assert isinstance(data.tref, datetime)
    assert data.tref.year == current_year

    # now set a datestr instead of datetime
    datestr = "20100201 000000"  # YYYYMMDD HHMMSS
    data = data.set_value("tref", datestr)

    # check if it is converted to datetime
    assert isinstance(data.tref, datetime)
    assert data.tref.year == 2010


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


def test_config_manning_land_sea_io(tmp_path):
    # Start with default values
    data0 = SfincsConfigVariables()

    # By default, land/sea roughness is not set
    assert data0.manning_land is None
    assert data0.manning_sea is None
    assert data0.rgh_lev_land is None

    # Explicitly set land/sea roughness
    data0 = data0.set_values(
        {
            "manning_land": 0.04,
            "manning_sea": 0.02,
            "rgh_lev_land": 0.0,
        }
    )

    inpfile = tmp_path / "sfincs.inp"
    data0.write(inpfile)

    # Read the configuration back
    data1 = SfincsConfigVariables.read(inpfile)

    assert data1.manning_land == 0.04
    assert data1.manning_sea == 0.02
    assert data1.rgh_lev_land == 0.0
