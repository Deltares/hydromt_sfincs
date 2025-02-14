import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union

from hydromt.model.components import ModelComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class SfincsStorageVolume(ModelComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs.vol"
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> pd.GeoDataFrame:
        """Storage volume data.

        Return geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data
    
#%% Original HydroMT-SFINCS setup_ functions:
    #   setup_storage_volume

#%% core HydroMT-SFINCS functions:
    # _initialize
    # read
    # write
    # set
    # create
    # add
    # delete
    # clear

#%% DDB GUI focused additional functions:
    # - yet unsupported in DDB-
