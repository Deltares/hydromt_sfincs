import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union

from hydromt.model.components import SpatialDatasetsComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class SfincsOutput(SpatialDatasetsComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs_map.nc"
        self._data: xr.Dataset = None #FIXME - how if xugrid needed?
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> xr.Dataset:
        """SFINCS output data.

        Return xr.Dataset
        """
        if self._data is None:
            self._initialize()
        return self._data
    
    # Original HydroMT-SFINCS setup_ functions:
    # read_results

#%% core HydroMT-SFINCS functions:
    # _initialize
    # read
    # set
    # clear

#%% DDB GUI focused additional functions:
    # read_his_file
    # read_zsmax
    # read_cumulative_precipitation