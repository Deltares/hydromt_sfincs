import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, List

from hydromt.model.components import ModelComponent
from hydromt.model import Model
from hydromt_sfincs import utils


class SfincsRivers(ModelComponent):
    def __init__(
        self,
        model: Model,
    ):
        # self._filename: str = "sfincs.src"
        self._data: gpd.GeoDataFrame = None  # FIXME - ?
        super().__init__(
            model=model,
        )

    # @property FIXME - has no own data? or would that be mask/centerlines etc?
    # def data(self) -> gpd.GeoDataFrame:
    #     """Discharge points data.

    #     Return gpd.GeoDataFrame
    #     """
    #     if self._data is None:
    #         self._initialize()
    #     return self._data

    # Original HydroMT-SFINCS setup_ functions:
    # setup_river_inflow
    # setup_river_outflow
    # FIXME - also functions like burn in river???
    # FIXME - also new functions to read/process/burn in river cross-section data???


# %% core HydroMT-SFINCS functions:
# _initialize
# create:
# create_inflow
# create_outflow
# clear

# %% DDB GUI focused additional functions:
# - yet unsupported in DDB-
