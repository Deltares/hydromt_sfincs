import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union

from hydromt.model import Model
from hydromt.model.components import SpatialDatasetsComponent
from hydromt_sfincs import utils


class SfincsInfiltration(SpatialDatasetsComponent):
    def __init__(
        self,
        model: Model,
    ):
        self._filename: None  # FIXME - depends on type of infiltration
        self._data: None  # FIXME - depends on type of infiltration
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> xr.Dataset:  # FIXME this is a copy of grid?
        """Infiltration data.

        Return FIXME
        """
        if self._data is None:
            self._initialize()
        return self._data

    # Original HydroMT-SFINCS setup_ functions:
    # setup_constant_infiltration
    # setup_cn_infiltration
    # setup_cn_infiltration_with_ks
    # To add: setup_green_ampt_infiltration
    # To add: setup_horton_infiltration


# %% core HydroMT-SFINCS functions:
# _initialize

# read
# write
# set
# create:
# create_constant_infiltration
# create_cn_infiltration
# create_cn_infiltration_with_ks
# create_green_ampt_infiltration
# create_horton_infiltration
# clear

# %% DDB GUI focused additional functions:
# - yet unsupported in DDB-
