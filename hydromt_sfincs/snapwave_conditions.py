import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, List

from hydromt.model.components import ModelComponent
from hydromt.model import Model
from hydromt_sfincs import utils


class SfincsSnapWaveConditions(ModelComponent):
    def __init__(
        self,
        model: Model,
    ):
        self._filename: str = "snapwave.nc"
        self._data: xr.Dataset = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> xr.Dataset:
        """SnapWaveboundary conditions data.

        Return xr.Dataset
        """
        if self._data is None:
            self._initialize()
        return self._data

    # Original HydroMT-SFINCS setup_ functions:
    # setup_snapwave_forcing


# %% core HydroMT-SFINCS functions:
# _initialize
# read
# read_snapwave_netcdf - netsnapwavefile
# read_snapwave_points - snapwave_bndfile > FIXME don't support?
# read_snapwave_timeseries - bhsfile/btpfile/bwdfile/bdsfile> FIXME don't support?
# write
# write_snapwave_conditions - netsnapwavefile (or other default name)
# and/or: write_boundary_points - snapwave_bndfile + write_boundary_conditions_timeseries - bhsfile/btpfile/bwdfile/bdsfile> FIXME don't support?
# set
# set_points - gpd.GeoDataFrame with points
# set_timeseries - pd.DataFrame with timeseries (?)
# create:
# create_snapwave_timeseries
# add
# delete
# clear

# %% DDB GUI focused additional functions:
# add_point (exception, call add for just one point)
# delete_point (exception, call delete for just one point)

# to_gdf
# has_open_boundaries

# get_boundary_points_from_mask
# set_timeseries
# set_timeseries_uniform
# set_conditions_at_point
# read_timeseries_file
# to_fwf
# inpolygon
#
# class SnapWaveBoundaryEnclosure > we are not going to support this > directly in snapwave_mask
# class SnapWaveMask > will be part of Mask class - exactly same functionality as for SFINCS mask
