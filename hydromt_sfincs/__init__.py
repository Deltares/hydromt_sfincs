"""hydroMT plugin for sfincs models."""

from os.path import dirname, join, abspath


__version__ = "1.0.3"

DATADIR = join(dirname(abspath(__file__)), "data")

from .sfincs import *
