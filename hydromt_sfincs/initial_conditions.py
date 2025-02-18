import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union

from hydromt.model.components import SpatialDatasetsComponent
from hydromt.model import Model
from hydromt_sfincs import utils


class SfincsInitialConditions(SpatialDatasetsComponent):
    def __init__(
        self,
        model: Model,
    ):
        self._filename: str = "sfincs_ini.nc"
        self._data: xr.Dataset = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> xr.Dataset:
        """Initial Conditions data.

        Return xr.Dataset
        """
        if self._data is None:
            self._initialize()
        return self._data

    # Original HydroMT-SFINCS setup_ functions:
    # not yet implemented


# %% core HydroMT-SFINCS functions:
# _initialize
# read
# write
# set
# create
# clear

# %% DDB GUI focused additional functions:
# interpolate
# interp2
