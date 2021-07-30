"""hydroMT plugin for sfincs models."""

from os.path import dirname, join, abspath


__version__ = "0.2.0"

DATADIR = join(dirname(abspath(__file__)), "data")

from .sfincs import *
from .utils import *
from . import gis_utils
