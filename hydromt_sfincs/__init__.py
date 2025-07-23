"""hydroMT plugin for sfincs models."""

from os.path import abspath, dirname, join

__version__ = "2.0.0.dev"

DATADIR = join(dirname(abspath(__file__)), "data")

from .sfincs import *
