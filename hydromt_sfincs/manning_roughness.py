import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union

from hydromt.model.components import SpatialDatasetsComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class SfincsManningRoughness(SpatialDatasetsComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: None #FIXME - depends on type of infiltration
        self._data: None #FIXME - depends on type of infiltration
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> pd.GeoDataFrame: #FIXME
        """Manning Roughness data.

        Return FIXME
        """
        if self._data is None:
            self._initialize()
        return self._data
    
#%% Original HydroMT-SFINCS setup_ functions:
    # setup_manning_roughness > only used for non-subgrid model
    
    # FIXME - should the burning in river mask roughness intelligence of setup_subgrid be included in this class?
    # Or in rivers.py?

#%% core HydroMT-SFINCS functions:
    # _initialize
    # read
    # write
    # set
    # create
    # clear

#%% DDB GUI focused additional functions:
    # - yet unsupported in DDB-
