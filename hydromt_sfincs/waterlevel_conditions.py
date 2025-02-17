import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, List

from hydromt.model.components import ModelComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class SfincsWaterlevelConditions(ModelComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs.bnd" #FIXME - List(str = "sfincs.bnd" and str = "sfincs.bzs" or str = "sfincs_netbndbzsbzi.nc")
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> pd.GeoDataFrame:
        """Water level boundary conditions data.

        Return pd.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data
    
    # Original HydroMT-SFINCS setup_ functions:
    # setup_waterlevel_forcing
    # setup_waterlevel_bnd_from_mask
    #
    # To add: setup_tidal_forcing

#%% core HydroMT-SFINCS functions:
    # _initialize
    # read
        # read_waterlevel_points - bndfile
        # read_waterlevel_timeseries - bzsfile
        # read_waterlevel_astro - bcafile
        # read_waterlevel_netcdf - netbndbzsbzifile --> ds = GeoDataset.from_netcdf(fn, crs=self.crs, chunks="auto")
    # write
        # write_boundary_points - bndfile + write_boundary_conditions_timeseries - bzsfile
        # and/or: write_boundary_conditions - netbndbzsbzifile    
    # set
        # set_points - gpd.GeoDataFrame with points
        # set_timeseries - pd.DataFrame with timeseries (?)
    # create:
        # create_waterlevel_timeseries
        # create_astro_timeseries
    # add
    # delete
    # clear

#%% DDB GUI focused additional functions:
    # add_point (exception, call add for just one point)
    # delete_point (exception, call delete for just one point)
    # read_boundary_conditions_astro
    # write_boundary_conditions_astro
    # generate_bzs_from_bca
    # get_boundary_points_from_mask
    # set_timeseries
    # read_timeseries_file
    # to_fwf

