"""Test sfincs model class against hydromt.models.model_api"""

import pytest
from os.path import join, dirname, abspath
import numpy as np
import pdb
from click.testing import CliRunner

import hydromt
from hydromt.models import MODELS
from hydromt.cli.cli_utils import parse_config
from hydromt.cli.main import main as hydromt_cli

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")

# test build method
# compare results with model from examples folder
model = "sfincs"
root = r"d:\repos\hydromt_sfincs\examples\sfincs_riverine"
config = join(EXAMPLEDIR, "sfincs_riverine.ini")

region = "{'bbox': [11.975098,45.786918,12.271385,45.938019]}"

# Build model
r = CliRunner().invoke(
    hydromt_cli, ["build", model, root, region, "-i", config, "-r", 50, "-vv"]
)
