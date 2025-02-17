import os
from os.path import isfile
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, List

import hydromt
from hydromt.model.components import SpatialDatasetsComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils
from hydromt.workflows.forcing import da_to_timedelta
   
#%% Original HydroMT-SFINCS setup_ functions:
    # setup_precip_forcing
    # setup_precip_forcing_from_grid
    # setup_pressure_forcing_from_grid
    # setup_wind_forcing
    # setup_wind_forcing_from_grid

    # supported spiderweb TC files through sfincs.inp: cyclone.spw
    # TODO - now also as netcdf: cyclone.nc - FIXME take care with pressure_drop vs atmospheric_pressure

#%% core HydroMT-SFINCS functions:
    # _initialize

    # 1 class per type - precipitation, pressure, wind
    # 2 flavours for 2D, and uniform (either from time-series or aggregated from 2D)

    # read:
        # read
        # read_uniform
    # write
        # write
        # write_uniform
    # set
    # create:
        # model.precipitation.create
        # model.precipitation.create_uniform
        # model.precipitation.create_uniform_from_gridded

        # model.pressure.create

        # model.wind.create
        # model.wind.create_uniform
        # model.wind.create_uniform_from_gridded

    # clear

#%% Precipitation
class SfincsPrecipitation(SpatialDatasetsComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs_netampr.nc"
        self._data: xr.DataArray = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> xr.DataArray:
        """Precipitation data.

        Return xr.DataArray
        """
        if self._data is None:
            self._initialize()
        return self._data
    
    def _initialize(self, skip_read=False) -> None:
        """Initialize DataArray."""
        if self._data is None:
            self._data = xr.DataArray
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self):
        """Read in gridded precipitation data."""
        ds = xr.open_dataset(self._filename, chunks="auto")

        # rename variables if needed
        rename = {k: v for k, v in rename.items() if k in ds}
        if len(rename) > 0:
            ds = ds.rename(rename).squeeze(drop=True)[list(rename.values())]
            self.set_forcing(ds, split_dataset=True)
        else:
            logger.warning(f"No forcing variables found in {self._filename}")

        # Add to self._data            
        self.set(ds)

    def read_uniform(self):
        """Read in spatially uniform precipitation data."""
        # TL: parts are copied from sfincs.py - read_forcing()
        tref = utils.parse_datetime(self.config["tref"]) #FIXME  - or differently?
        
        if self._filename is None or not isfile(self._filename):
            if self._filename is not None:
                self.logger.warning(f"{self._filename} not found")

        df = utils.read_timeseries(self._filename, tref)
        df.index.name = "time"

        # spatially uniform forcing
        da = xr.DataArray(df[df.columns[0]], dims=("time"), name='precip')

        # Add to self._data            
        self.set(da)

    def write(self, filename=None): #TODO - TL: filename=None - still needed?
        """Write netamprfile."""

        #FIXME - correct for tref_str?
        encoding = dict(
            time={"units": f"minutes since {self.model.config.tref_str}", "dtype": "float64"} 
        )
        
        # combine variables and rename to output names
        rename = {v: k for k, v in rename.items() if v in self.forcing}
        ds = xr.merge([self.forcing[v] for v in rename.keys()]).rename(rename)        

        # write 2D gridded timeseries
        ds.to_netcdf(filename, encoding=encoding)

    def write_uniform(self, filename=None, fmt: str = "%7.2f"): #TODO - TL: filename=None - still needed?
        """Write precipfile."""

        tref = utils.parse_datetime(self.config["tref"]) #FIXME  - or differently?

        # parse data to dataframe
        da = self.data.transpose("time", ...)
        df = da.to_pandas()
        # get filenames from config - FIXME - needed?
        # if f"{ts_name}file" not in self.config:
        #     self.set_config(f"{ts_name}file", f"sfincs.{ts_name}")
        # fn = self.get_config(f"{ts_name}file", abs_path=True)

        # write timeseries
        utils.write_timeseries(filename, df, tref, fmt=fmt)

    def set(
            self,
            da: xr.DataArray,
        ):
        """Set 2D precipitation data.

        Arguments
        ---------
        da: xr.DataArray
            Set DataArray with precipitation data to self.data
        """       

        # FIXME - add check if wanted variables exist (?)
        # XXX

        self._data = da  # set da in self.data    

    # def set_uniform(): #FIXME - not needed?

    def create(
        self, 
        precip, 
        dst_res=None, 
        **kwargs,
        ):
        """Setup precipitation forcing from a gridded spatially varying data source.

        Distributed precipitation is added to the model as netcdf file.
        The data is reprojected to the model CRS (and destination resolution `dst_res` if provided).

        Adds model layer:

        * **netamprfile** forcing: distributed precipitation [mm/hr]

        Parameters
        ----------
        precip, str, Path
            Path to precipitation rasterdataset netcdf file.

            * Required variables: ['precip' (mm/hr)]   #FIXME > TL: why not mm/hr?
            * Required coordinates: ['time', 'y', 'x']

        dst_res: float
            output resolution (m), by default None and computed from source data.
            Only used in combination with aggregate=False
        """
        # get data for model domain and config time range
        precip = self.data_catalog.get_rasterdataset(
            precip,
            bbox=self.bbox,
            buffer=2,
            time_tuple=self.get_model_time(),
            variables=["precip"],
        )

        # reproject to model utm crs
        # NOTE: currently SFINCS errors (stack overflow) on large files, #FIXME - TL: old message?
        # downscaling to model grid is not recommended #FIXME - TL: though still not recommended to go finer than data source resolution!
        kwargs0 = dict(align=dst_res is not None, method="nearest_index")
        kwargs0.update(kwargs)
        meth = kwargs0["method"]
        self.logger.debug(f"Resample precip using {meth}.")
        precip_out = precip.raster.reproject(
            dst_crs=self.crs, dst_res=dst_res, **kwargs
        ).fillna(0)

        # only resample in time if freq < 1H, else keep input values        
        # FIXME - TL: make this user optional!!! 
        # TODO - and at least with clear warning
        
        if da_to_timedelta(precip_out) < pd.to_timedelta("1H"):
            precip_out = hydromt.workflows.resample_time(
                precip_out,
                freq=pd.to_timedelta("1H"),
                conserve_mass=True,
                upsampling="bfill",
                downsampling="sum",
                logger=self.logger,
            )
        precip_out = precip_out.rename("precip_2d") #FIXME - needed?

        # Add to self.data
        self.set(precip_out)

    def create_uniform(
        self, 
        timeseries=None, 
        magnitude=None
        ):
        """Setup spatially uniform precipitation forcing (precip).

        Adds model layers:

        * **precipfile** forcing: uniform precipitation [mm/hr]

        Parameters
        ----------
        timeseries: str, Path
            Path to tabulated timeseries csv file with time index in first column
            and spatially uniform precipitation rate in the first row,
            see :py:meth:`hydromt.open_timeseries_from_table`, for details.
            Note: tabulated timeseries files cannot yet be set through the data_catalog yml file.
        magnitude: float
            Precipitation magnitude [mm/hr] to use if no timeseries is provided.
        """
        tstart, tstop = self.get_model_time()

        if timeseries is not None:
            df_ts = self.data_catalog.get_dataframe(
                timeseries,
                time_tuple=(tstart, tstop),
                # kwargs below only applied if timeseries not in data catalog
                parse_dates=True,
                index_col=0,
            )
        elif magnitude is not None:
            times = pd.date_range(*self.get_model_time(), freq="10T")
            df_ts = pd.DataFrame(
                index=times, data=np.full((len(times), 1), magnitude, dtype=float)
            )
        else:
            raise ValueError("Either timeseries or magnitude must be provided")

        if isinstance(df_ts, pd.DataFrame):
            df_ts = df_ts.squeeze()
        if not isinstance(df_ts, pd.Series):
            raise ValueError("df_ts must be a pandas.Series")
        df_ts.name = "precip"
        df_ts.index.name = "time"

        # Add to self.data
        self.set(df_ts.to_xarray())

    def setup_precip_forcing_from_grid(
        self, 
        precip, 
        aggregate=False, 
        # **kwargs
        ):
        """Setup spatially uniform precipitation forcing from a gridded spatially varying data source.

        Spatially uniform precipitation forcing is added to
        the model based on the mean precipitation over the model domain.

        Adds one of these model layer:

        * **precipfile** forcing: uniform precipitation [mm/hr]

        Parameters
        ----------
        precip, str, Path
            Path to precipitation rasterdataset netcdf file.

            * Required variables: ['precip' (mm/hr)]
            * Required coordinates: ['time', 'y', 'x']

        aggregate: bool, {'mean', 'median'}, optional
            Method to aggregate distributed input precipitation data. If True, mean
            aggregation is used, if False (default) the data is not aggregated and
            spatially distributed precipitation is returned.
        """
        # get data for model domain and config time range
        precip = self.data_catalog.get_rasterdataset(
            precip,
            bbox=self.bbox,
            buffer=2,
            time_tuple=self.get_model_time(),
            variables=["precip"],
        )

        # aggregate in space
        stat = aggregate if isinstance(aggregate, str) else "mean"
        self.logger.debug(f"Aggregate precip using {stat}.")
        zone = self.region.dissolve()  # make sure we have a single (multi)polygon
        precip_out = precip.raster.zonal_stats(zone, stats=stat)[f"precip_{stat}"]
        df_ts = precip_out.where(precip_out >= 0, 0).fillna(0).squeeze().to_pandas()
        
        # call create_uniform
        self.create_uniform(df_ts.to_frame())

    def clear(self):
        """Clean DataArray with precipitation data."""
        self.data  = xr.DataArray()        

#%% Pressure
class SfincsPressure(SpatialDatasetsComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs_netamp.nc"
        self._data: xr.DataArray = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> xr.DataArray:
        """Pressure data.

        Return xr.DataArray
        """
        if self._data is None:
            self._initialize()
        return self._data
    
    def _initialize(self, skip_read=False) -> None:
        """Initialize DataArray."""
        if self._data is None:
            self._data = xr.DataArray
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    # TODO - add:
    # read:
        # read
    # write
        # write
    # set
    # create:
        # model.wind.create
    
    def clear(self):
        """Clean DataArray with atmospheric pressure data."""
        self.data  = xr.DataArray()      

#%% Wind
class SfincsWind(SpatialDatasetsComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs_netamuv.nc"
        self._data: xr.DataArray = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> xr.DataArray:
        """Wind data.

        Return xr.DataArray
        """
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize DataArray."""
        if self._data is None:
            self._data = xr.DataArray
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    # TODO - add:
    # read:
        # read
        # read_uniform
    # write
        # write
        # write_uniform
    # set
    # create:
        # model.wind.create
        # model.wind.create_uniform
        # model.wind.create_uniform_from_gridded
    
    def clear(self):
        """Clean DataArray with wind data."""
        self.data  = xr.DataArray()      

#%% DDB GUI focused additional functions:
    # - yet unsupported in DDB-
