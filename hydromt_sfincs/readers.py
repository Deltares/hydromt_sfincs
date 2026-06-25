"""Standalone reader functions for HydroMT-SFINCS."""

from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["read_config"]


def read_config(
    filename: Path | str,
) -> dict[str, Any]:
    """Read the SFINCS input file (sfincs.inp).

    Parameters
    ----------
    filename : Path | str
        The path to the input file containing the SFINCS model settings.

    Returns
    -------
    dict[str, Any]
        The model settings.
    """
    # Ensure typing
    filename = Path(filename)
    # Check if exists
    if not filename.exists():
        raise FileNotFoundError(
            f"SFINCS input file '{filename.as_posix()}' does not exist."
        )

    # Read the file line by line
    with open(filename, "r") as fid:
        lines = fid.readlines()

    inp_dict = {}
    for line in lines:
        # Check if first character is #
        if line.strip().startswith("#"):
            # Full line comment
            continue
        # Find last character before #
        comment_idx = line.find("#")
        if comment_idx >= 0:
            line = line[:comment_idx]
        line = [x.strip() for x in line.split("=")]
        if len(line) != 2:
            continue
        name, val = line
        if name in ["tref", "tstart", "tstop"]:
            try:
                val = datetime.strptime(val, "%Y%m%d %H%M%S")
            except ValueError:
                raise ValueError(f'"{name} = {val}" not understood.')
        elif name in ["cdwnd", "cdval"]:
            val = [float(x) for x in val.split()]
        elif name == "utmzone":
            val = str(val)
        else:
            try:
                val = literal_eval(val)
            except Exception:
                pass

        if name == "crs":
            name = "epsg"
        elif name == "dtout":
            name = "dtmapout"

        inp_dict[name] = val

    # Return the config values
    return inp_dict
