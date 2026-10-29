"""hydroMT plugin for sfincs models."""

from os.path import abspath, dirname, join

__version__ = "2.0.0-rc4"

DATADIR = join(dirname(abspath(__file__)), "data")

# Minimum supported SFINCS versions
MIN_SUPPORTED_SFINCS_VERSION = "2.1.0"
# Latest supported SFINCS version (new models will use this by default)
LATEST_SFINCS_VERSION = "2.5.0"

from .sfincs import *
