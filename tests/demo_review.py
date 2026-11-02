"""Demonstrate the #417 review points: read an sfincs.inp, show warnings and output."""

import logging
import tempfile
import warnings
from pathlib import Path

from hydromt_sfincs.components.config.config_variables import SfincsConfigVariables

logging.basicConfig(level=logging.WARNING, format="    LOG %(levelname)s: %(message)s")
warnings.simplefilter("always")
warnings.showwarning = lambda msg, cat, *a, **k: print(f"    {cat.__name__}: {msg}")
TMP = Path(tempfile.mkdtemp())


def demo(title, inp, **write_kwargs):
    print(f"\n=== {title}")
    print("  input:")
    print("".join(f"    {line}\n" for line in inp.strip().splitlines()))
    inpfile = TMP / "sfincs.inp"
    inpfile.write_text(inp)
    print("  read:")
    data = SfincsConfigVariables.read(inpfile)
    out = TMP / "out.inp"
    data.write(out, explicit_only=True, **write_kwargs)
    print(f"  written{' ' + str(write_kwargs) if write_kwargs else ''}:")
    print("".join(f"    {line}\n" for line in out.read_text().splitlines()))
    return data


old = """dtout        = 1800
crs          = 32633
precipfile   = sfincs.precip
wvmfile      = sfincs.wvm
global       = 1
storeqdrain  = 1
"""
demo("(1)+(2) unversioned old file: kept as read, only 'crs' (< 2.1.0) replaced", old)

demo(
    "(2) versioned file: keywords not read by SFINCS 2.5.0 are replaced, with warnings",
    "sfincs_version = 2.5.0\ndtout = 1800\nwvmfile = sfincs.wvm\nglobal = 1\n",
)
demo(
    "(2) versioned file: 2.4.1 still reads these, so they are kept",
    "sfincs_version = 2.4.1\ndtout = 1800\nwvmfile = sfincs.wvm\nglobal = 1\n",
)

for keyword in ["snapwave_waveforces_factor", "waveforces_ratio"]:
    demo(
        f"(3) multiple aliases: '{keyword}' is written back as read",
        f"snapwave = 1\n{keyword} = 0.5\n",
    )

demo(
    "(5) same old file, written with update_to_latest=True", old, update_to_latest=True
)

print(
    "\n=== (5) a new model gets sfincs_version; setting a legacy name stores the current one"
)
data = SfincsConfigVariables().set_value("sfincs_version", "2.5.0")
data = data.set_value("wvmfile", "sfincs.wvm")
out = TMP / "new.inp"
data.write(out, explicit_only=True)
print("".join(f"    {line}\n" for line in out.read_text().splitlines()))
