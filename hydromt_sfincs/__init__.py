"""hydroMT plugin for sfincs models."""

from os.path import abspath, dirname, join

__version__ = "2.0.0-rc4dev"

DATADIR = join(dirname(abspath(__file__)), "data")
MIN_SUPPORTED_SFINCS_VERSION = "2.1.0"

from .sfincs import *
