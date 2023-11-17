"""hydroMT plugin for sfincs models."""

from os.path import dirname, join, abspath


__version__ = "1.0.2"

DATADIR = join(dirname(abspath(__file__)), "data")

# from .hydromt_sfincs import *
from .sfincs import *
