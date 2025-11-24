import logging
from os.path import isfile
from pathlib import Path
from typing import List, Literal, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent
from hydromt.model.processes.meteo import da_to_timedelta

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

# dictionary of meteo variables and their corresponding SFINCS file names and renaming conventions
_METEO = {
    "precip": ("precip", None),
    "precip_2d": ("netampr", {"Precipitation": "precip_2d"}),
    "press_2d": ("netamp", {"barometric_pressure": "press_2d"}),
    "wind": ("wnd", None),
    "wind_2d": (
        "netamuamv",
        {"eastward_wind": "wind10_u", "northward_wind": "wind10_v"},
    ),
}


class SfincsMeteo(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = None  # set in subclasses
        self._data: Optional[Union[xr.DataArray, xr.Dataset]] = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> Union[xr.DataArray, xr.Dataset]:
        """Meteo data. Returns a xr.DataArray or xarray Dataset."""
        if self._data is None:
            self._initialize()

        assert self._data is not None
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._data is None:
            self._data = xr.Dataset()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(
        self,
        variable: str,
        filename: str | Path = None,
    ):
        """Read meteo data from file. Possible variables are 'precip', 'wind', 'press'."""
        # check that read mode is on
        self.root._assert_read_mode()

        assert variable in [
            "precip",
            "wind",
            "press",
        ], f"Variable {variable} not supported. Supported variables are 'precip', 'wind', 'press'."

        filtered_dict = {key: value for key, value in _METEO.items() if variable in key}

        # Check which type of meteo for this variable is present in the model
        for name in filtered_dict:
            fname, rename = _METEO[name]
            # get absolute file path and set it in config if obsfile is not None
            abs_file_path = self.model.config.get_set_file_variable(
                f"{fname}file", value=filename
            )
            # check if abs_file_path is None or does not exist
            if abs_file_path is None:
                continue  # skip if no file is set
            elif not abs_file_path.exists():
                raise FileNotFoundError(f"{fname}file not found: {abs_file_path}")

            # check whether we are reading gridded (always netcdf) or uniform data
            if abs_file_path.suffix == ".nc":
                self.read_gridded(filename=abs_file_path, rename=rename)
            else:
                self.read_uniform(filename=abs_file_path, variable=name)

            # also update the default filename (potentially used for writing)
            self._filename = abs_file_path.name

    def read_gridded(self, filename: str | Path = None, rename: Optional[dict] = None):
        """Read in gridded meteo data."""
        # open the netcdf file
        ds = xr.open_dataset(filename, chunks="auto")

        # check if variables need to be renamed
        rename = {k: v for k, v in rename.items() if k in ds}
        if len(rename) > 0:
            ds = ds.rename(rename).squeeze(drop=True)[list(rename.values())]

        # set the data
        self.set(ds)

    def read_uniform(self, variable: str, filename: str | Path = None):
        """Read in spatially uniform precipitation data."""
        tref = utils.parse_datetime(self.model.config.get("tref"))

        df = utils.read_timeseries(filename, tref)
        df.index.name = "time"

        # spatially uniform forcing
        if variable == "wind":
            # wind speed and direction
            da = xr.DataArray(
                df,
                dims=("time", "index"),
                coords={"time": df.index, "index": ["magnitude", "direction"]},
                name=variable,
            )
        else:
            da = xr.DataArray(df[df.columns[0]], dims=("time"), name=variable)

        # Add to self._data
        self.set(da)

    def write(self, variable: str, filename: str | Path = None, fmt: str = "%7.2f"):
        """Write meteo data to file. Possible variables are 'precip', 'wind', 'press'."""

        # check that write mode is on
        self.root._assert_write_mode()

        assert variable in [
            "precip",
            "wind",
            "press",
        ], f"Variable {variable} not supported. Supported variables are 'precip', 'wind', 'press'."
        filtered_dict = {key: value for key, value in _METEO.items() if variable in key}

        for name in filtered_dict:
            fname, rename = _METEO[name]
            # check if data present:
            if name == "wind_2d":
                if "wind10_u" and "wind10_v" not in self.data:
                    continue

            elif name not in self.data:
                continue

            # Set file name and get absolute path
            abs_file_path = self.model.config.get_set_file_variable(
                key=f"{fname}file", value=filename, default=self._filename
            )

            # Create parent directories if they do not exist
            abs_file_path.parent.mkdir(parents=True, exist_ok=True)

            if abs_file_path.suffix == ".nc":
                self.write_gridded(filename=abs_file_path, rename=rename)
            else:
                self.write_uniform(variable=name, filename=abs_file_path, fmt=fmt)

    def write_gridded(self, filename: str | Path = None, rename: Optional[dict] = None):
        """Write spatially varying meteo file as netcdf."""

        tref = self.model.config.get("tref")
        tref_str = tref.strftime("%Y-%m-%d %H:%M:%S")

        encoding = dict(time={"units": f"minutes since {tref_str}", "dtype": "float64"})

        # assign self.data to ds
        ds = self.data.load()

        # combine variables and rename to output names
        rename = {v: k for k, v in rename.items() if v in ds}
        if len(rename) > 0:
            ds = xr.merge([ds[v] for v in rename.keys()]).rename(rename)

        # write 2D gridded timeseries
        ds.to_netcdf(filename, encoding=encoding)

    def write_uniform(
        self, variable: str, filename: str | Path = None, fmt: str = "%7.2f"
    ):
        """Write uniform meteo file."""

        tref = self.model.config.get("tref")

        # parse data to dataframe
        da = self.data.transpose("time", ...)
        df = da[variable].to_pandas()

        # write timeseries
        utils.write_timeseries(filename, df, tref, fmt=fmt)

    def set(
        self,
        data: Union[xr.DataArray, xr.Dataset],
        name: Optional[str] = None,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        self._initialize()
        assert self._data is not None

        name_required = isinstance(data, np.ndarray) or (
            isinstance(data, xr.DataArray) and data.name is None
        )
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        if isinstance(data, xr.DataArray):
            if name is not None:
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"cannot set data of type {type(data).__name__}")

        # TODO: don't we always want to reset the data when setting new data?
        # that would mean that you can never have 1D and 2D data at the same time
        if len(self._data) == 0:  # empty grid
            self._data = data
        else:
            for dvar in data.data_vars:
                if dvar in self._data and self.root.is_reading_mode():
                    logger.warning(f"Replacing grid map: {dvar}")
                self._data[dvar] = data[dvar]

    def clear(self):
        """Clear the data attribute."""
        self._data = xr.Dataset()


class SfincsPrecipitation(SfincsMeteo):
    """
    SFINCS precipitation forcing. This component handles the creation and management of precipitation data for the SFINCS model.
    It supports both spatially distributed precipitation data (as a NetCDF file) and uniform precipitation data (as a time series).
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs_netampr.nc"
        super().__init__(model=model)

    def read(self, filename: str | Path = None):
        """Read precipitation data from file."""

        super().read(
            variable="precip",
            filename=filename,
        )

    def write(self, filename: str | Path = None, fmt: str = "%7.2f"):
        super().write(
            variable="precip",
            filename=filename,
            fmt=fmt,
        )

    @hydromt_step
    def create(
        self,
        precip,
        buffer: float = 5e3,
        dst_res: float = None,
        cumulative_input: bool = True,
        time_label: Literal["left", "right"] = "right",
        aggregate: bool = False,
        **kwargs,
    ):
        """Setup precipitation forcing from a gridded spatially varying data source.

        SFINCS requires the mean precipitation rate in mm/hr over the upcoming interval.
        It will use the rate at the start of this interval and keep it constant throughout.
        This method can use rainfall rates in [mm/hr] directly, or transform accumulated precip [mm] over any constant time interval to precipitation rates.

        Input precipitation can be specified as:

        * **Cumulative precipitation (mm)** over any constant time interval (e.g., 15/60/180 minutes).
          This will be converted to a rate (mm/hr) if ``cumulative_input=True`` (default).

        * **Precipitation rate (mm/hr)** at any time interval. Used as-is if
          ``cumulative_input=False``.

        If ``aggregate=True``, a spatially uniform precipitation forcing is applied based on
        the domain-wide mean. If ``aggregate=False``, distributed precipitation is applied
        using a NetCDF file. In that case, data is reprojected to the model CRS (and to
        ``dst_res`` if provided).

        One of the following model layers will be added:

        * **netamprfile**: for distributed precipitation rate [mm/hr].
        * **precipfile**: for uniform precipitation rate [mm/hr].

        .. note::

            SFINCS updates the meteo forcing every 1800 seconds (``dtwnd=1800`` by default).
            If your dataset has smaller intervals, ``dtwnd`` in ``sfincs.inp`` is adjusted automatically.

        .. note::

            To allow precipitation rates to vary linearly over the time interval
            (instead of being constant), set ``ampr_block = 0`` in ``sfincs.inp``.

        Parameters
        ----------
        precip: str or Path
            Path to precipitation rasterdataset netcdf file.

            * Required variables: ['precip' (mm) or 'precip' (mm/hr)]
            * Required coordinates: ['time', 'y', 'x']
        buffer: float, optional
            Buffer (m) around the model domain to read the precipitation data.
            This is useful to avoid edge effects when the precipitation data is clipped at the model domain.
        dst_res: float, optional
            Output resolution (m), by default None and computed from source data.
            Only used in combination with aggregate=False
        cumulative_input: bool, optional
            Option to indicate whether the input precipitation is cumulative in mm
            (True, default) or a precipitation rate in mm/hr (False). When cumulative,
            the data is converted to mm/hr by dividing by the time interval of the input dataset.
        time_label: literal, optional
            Label to prescribe whether the accumulation period of the precipitation
            is starting (left) or ending (right) at the validity date and time.
            This is only relevant if cumulative_input=True.
        aggregate: bool, {'mean', 'median'}, optional
            Method to aggregate distributed input precipitation data. If True, mean
            aggregation is used, if False (default) the data is not aggregated and
            spatially distributed precipitation is returned.
        """
        # get data for model domain and config time range
        precip = self.data_catalog.get_rasterdataset(
            precip,
            bbox=self.model.bbox,
            buffer=buffer,
            time_range=self.model.get_model_time(),
            variables=["precip"],
        )

        y_dim, x_dim = precip.raster.dims
        # check if x, y coordinates are 2D
        if precip.coords[x_dim].size < 2 or precip.coords[y_dim].size < 2:
            raise ValueError(
                "Precipitation does not have 2D coordinates after spatial clipping."
                "Check the input data and the model region or consider increasing the buffer."
            )

        # check time coordinates
        if precip.coords["time"].size < 2:
            raise ValueError(
                "Precipitation does not overlap with the model time range"
                "Check the input data and the model configuration."
            )

        # get the time interval of the input data in seconds
        time_interval = da_to_timedelta(precip).total_seconds()

        # check if time interval is set in the model config, else use default from SFINCS
        dtwnd = self.model.config.get("dtwnd", 1800)
        if dtwnd > time_interval:
            self.model.config.set("dtwnd", time_interval)
            logger.warning(
                f"dtwnd ({dtwnd}) was larger than the time interval of the precip data ({time_interval}) and therefore lowered."
            )

        # check if precip is cumulative or not to convert to mm/hr
        if cumulative_input:
            # convert to mm/hr by dividing by the time interval in seconds
            precip = precip / (time_interval / 3600)
            if time_label == "right":
                # typically cumulative precipation is accumulated over time interval ending at the validity time,
                # to match SFINCS conventions, we shift the time index to the left
                precip = precip.shift(time=-1, fill_value=0)

        # aggregate or reproject in space
        if aggregate:
            stat = aggregate if isinstance(aggregate, str) else "mean"
            logger.debug(f"Aggregate precip using {stat}.")
            zone = (
                self.model.region.dissolve()
            )  # make sure we have a single (multi)polygon
            precip_out = precip.raster.zonal_stats(zone, stats=stat)[f"precip_{stat}"]
            df_ts = precip_out.where(precip_out >= 0, 0).fillna(0).squeeze().to_pandas()
            self.create_uniform(timeseries=df_ts.to_frame())
        else:
            # reproject to model utm crs
            # downscaling to model grid is not recommended
            kwargs0 = dict(align=dst_res is not None, method="nearest_index")
            kwargs0.update(kwargs)
            meth = kwargs0["method"]
            logger.debug(f"Resample precip using {meth}.")
            precip_out = precip.raster.reproject(
                dst_crs=self.model.crs, dst_res=dst_res, **kwargs
            ).fillna(0)

            precip_out = precip_out.rename("precip_2d")

            # rename dimensions to match SFINCS conventions (always x and y)
            y_dim, x_dim = precip_out.raster.dims
            precip_out = precip_out.rename({y_dim: "y", x_dim: "x"})

            # add to data
            self.set(precip_out, name="precip_2d")
            # update the model config
            self.model.config.set("netamprfile", "sfincs_netampr.nc")
            if self.model.config.get("precipfile", None) is not None:
                logger.warning(
                    "precipfile was previously set, but is now replaced by netamprfile."
                )
                self.model.config.set("precipfile", None)
            # update the default filename
            self._filename = "sfincs.netampr.nc"

    @hydromt_step
    def create_uniform(self, timeseries=None, magnitude=None):
        """Setup spatially uniform precipitation forcing (precip).

        Adds model layers:

        * **precipfile** forcing: uniform precipitation [mm/hr]

        Parameters
        ----------
        timeseries: str, Path
            Path to tabulated timeseries csv file with time index in first column
            and location IDs in the first row,
            see :py:meth:`hydromt.open_timeseries_from_table`, for details.
            Note: tabulated timeseries files cannot yet be set through the data_catalog yml file.
        magnitude: float
            Precipitation magnitude [mm/hr] to use if no timeseries is provided.
        """
        tstart, tstop = self.model.get_model_time()
        if timeseries is not None:
            df_ts = self.data_catalog.get_dataframe(
                timeseries,
                time_range=(tstart, tstop),
                source_kwargs={
                    "driver": {
                        "name": "pandas",
                        "options": {"index_col": 0, "parse_dates": True},
                    }
                },
            )
        elif magnitude is not None:
            times = pd.date_range(*self.model.get_model_time(), freq="10T")
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
        self.set(df_ts.to_xarray(), name="precip")
        # update the model config
        self.model.config.set("precipfile", "sfincs.precip")
        if self.model.config.get("netamprfile", None) is not None:
            logger.warning(
                "netamprfile was previously set, but is now replaced by precipfile."
            )
            self.model.config.set("netamprfile", None)
        # update the default filename
        self._filename = "sfincs.precip"


# %% Pressure
class SfincsPressure(SfincsMeteo):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs_netamp.nc"
        self._data: xr.DataArray = None
        super().__init__(
            model=model,
        )

    def read(self, filename: str | Path = None):
        super().read(
            variable="press",
            filename=filename,
        )

    def write(self, filename: str | Path = None, fmt: str = "%7.2f"):
        super().write(
            variable="press",
            filename=filename,
            fmt=fmt,
        )

    @hydromt_step
    def create(
        self, press, buffer: float = 5e3, dst_res=None, fill_value=101325, **kwargs
    ):
        """Setup pressure forcing from a gridded spatially varying data source.

        Adds one model layer:

        * **netampfile** forcing: distributed barometric pressure [Pa]

        Parameters
        ----------
        press, str, Path, xr.Dataset, xr.DataArray
            Path to pressure rasterdataset netcdf file or xarray dataset.

            * Required variables: ['press_msl' (Pa)]
            * Required coordinates: ['time', 'y', 'x']
        buffer: float, optional
            Buffer (m) around the model domain to read the pressure data.
            This is useful to avoid edge effects when the pressure data is clipped at the model domain.
        dst_res: float
            output resolution (m), by default None and computed from source data.

        fill_value: float
            value to use when no data is available.
            Standard atmospheric pressure (101325 Pa) is used if no value is given.
        """
        # get data for model domain and config time range
        press = self.data_catalog.get_rasterdataset(
            press,
            bbox=self.model.bbox,
            buffer=buffer,
            time_range=self.model.get_model_time(),
            variables=["press_msl"],
        )

        y_dim, x_dim = press.raster.dims
        # check if x, y coordinates are 2D
        if press.coords[x_dim].size < 2 or press.coords[y_dim].size < 2:
            raise ValueError(
                "Pressure does not have 2D coordinates after spatial clipping."
                "Check the input data and the model region or consider increasing the buffer."
            )
        # check time coordinates
        if press.coords["time"].size < 2:
            raise ValueError(
                "Pressure does not overlap with the model time range"
                "Check the input data and the model configuration."
            )

        # get the time interval of the input data in seconds
        time_interval = da_to_timedelta(press).total_seconds()

        # check if time interval is set in the model config, else use default from SFINCS
        dtwnd = self.model.config.get("dtwnd", 1800)
        if dtwnd > time_interval:
            self.model.config.set("dtwnd", time_interval)
            logger.warning(
                f"dtwnd ({dtwnd}) was larger than the time interval of the pressure data ({time_interval}) and therefore lowered."
            )

        # reproject to model utm crs
        # downscaling to model grid is not recommended
        kwargs0 = dict(align=dst_res is not None, method="nearest_index")
        kwargs0.update(kwargs)
        meth = kwargs0["method"]
        logger.debug(f"Resample pressure using {meth}.")
        press_out = press.raster.reproject(
            dst_crs=self.model.crs, dst_res=dst_res, **kwargs
        ).fillna(fill_value)

        press_out = press_out.rename("press_2d")

        # rename dimensions to match SFINCS conventions (always x and y)
        y_dim, x_dim = press_out.raster.dims
        press_out = press_out.rename({y_dim: "y", x_dim: "x"})

        # add to forcing
        self.set(press_out, name="press_2d")
        # update the model config
        self.model.config.set("netampfile", "sfincs_netamp.nc")


# %% Wind
class SfincsWind(SfincsMeteo):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs_netamuv.nc"
        self._data: xr.DataArray = None
        super().__init__(
            model=model,
        )

    def read(self, filename: str | Path = None):
        super().read(
            variable="wind",
            filename=filename,
        )

    def write(self, filename: str | Path = None, fmt: str = "%7.2f"):
        super().write(
            variable="wind",
            filename=filename,
            fmt=fmt,
        )

    @hydromt_step
    def create(self, wind, buffer: float = 5e3, dst_res=None, **kwargs):
        """Setup wind forcing from a gridded spatially varying data source.

        Adds one model layer:

        * **netamuamv** forcing: distributed wind [m/s]

        Parameters
        ----------
        wind, str, Path, xr.Dataset
            Path to wind rasterdataset (including eastward and northward components) netcdf file or xarray dataset.

            * Required variables: ['wind10_u' (m/s), 'wind10_v' (m/s)]
            * Required coordinates: ['time', 'y', 'x']
        buffer: float, optional
            Buffer (m) around the model domain to read the wind data.
            This is useful to avoid edge effects when the wind data is clipped at the model domain.
        dst_res: float
            output resolution (m), by default None and computed from source data.
        """
        # get data for model domain and config time range
        wind = self.data_catalog.get_rasterdataset(
            wind,
            bbox=self.model.bbox,
            buffer=buffer,
            time_range=self.model.get_model_time(),
            variables=["wind10_u", "wind10_v"],
        )

        y_dim, x_dim = wind.raster.dims
        # check if x, y coordinates are 2D
        if wind.coords[x_dim].size < 2 or wind.coords[y_dim].size < 2:
            raise ValueError(
                "Wind does not have 2D coordinates after spatial clipping."
                "Check the input data and the model region or consider increasing the buffer."
            )
        # check time coordinates
        if wind.coords["time"].size < 2:
            raise ValueError(
                "Wind does not overlap with the model time range"
                "Check the input data and the model configuration."
            )

        # get the time interval of the input data in seconds
        time_interval = da_to_timedelta(wind).total_seconds()

        # check if time interval is set in the model config, else use default from SFINCS
        dtwnd = self.model.config.get("dtwnd", 1800)
        if dtwnd > time_interval:
            self.model.config.set("dtwnd", time_interval)
            logger.warning(
                f"dtwnd ({dtwnd}) was larger than the time interval of the wind data ({time_interval}) and therefore lowered."
            )

        # reproject to model utm crs
        # downscaling to model grid is not recommended
        kwargs0 = dict(align=dst_res is not None, method="nearest_index")
        kwargs0.update(kwargs)
        meth = kwargs0["method"]
        logger.debug(f"Resample wind using {meth}.")

        wind_out = wind.raster.reproject(
            dst_crs=self.model.crs, dst_res=dst_res, **kwargs
        ).fillna(0)

        # rename dimensions to match SFINCS conventions (always x and y)
        y_dim, x_dim = wind_out.raster.dims
        wind_out = wind_out.rename({y_dim: "y", x_dim: "x"})

        # add to forcing
        self.set(wind_out, name="wind_2d")
        # update the model config
        self.model.config.set("netamuamvfile", "sfincs_netamuv.nc")
        if self.model.config.get("wndfile", None) is not None:
            logger.warning(
                "wndfile was previously set, but is now replaced by netamuamvfile."
            )
            self.model.config.set("wndfile", None)
        # update the default filename
        self._filename = "sfincs_netamuv.nc"

    @hydromt_step
    def create_uniform(self, timeseries=None, magnitude=None, direction=None):
        """Setup spatially uniform wind forcing (wind).

        Adds model layers:

        * **windfile** forcing: uniform wind magnitude [m/s] and direction [deg]

        Parameters
        ----------
        timeseries, str, Path
            Path to tabulated timeseries csv file with time index in first column,
            magnitude in second column and direction in third column
            see :py:meth:`hydromt.open_timeseries_from_table`, for details.
            Note: tabulated timeseries files cannot yet be set through the data_catalog yml file.
        magnitude: float
            Magnitude of the wind [m/s]
        direction: float
            Direction where the wind is coming from [deg], e.g. 0 is north, 90 is east, etc.
        """
        tstart, tstop = self.model.get_model_time()
        if timeseries is not None:
            df_ts = self.data_catalog.get_dataframe(
                timeseries,
                time_range=(tstart, tstop),
                source_kwargs={
                    "driver": {
                        "name": "pandas",
                        "options": {"index_col": 0, "parse_dates": True},
                    }
                },
            )
        elif magnitude is not None and direction is not None:
            df_ts = pd.DataFrame(
                index=pd.date_range(*self.model.get_model_time(), periods=2),
                data=np.array([[magnitude, direction], [magnitude, direction]]),
                columns=["mag", "dir"],
            )
        else:
            raise ValueError(
                "Either timeseries or magnitude and direction must be provided"
            )

        df_ts.name = "wind"
        df_ts.index.name = "time"
        df_ts.columns.name = "index"
        da = xr.DataArray(
            df_ts.values,
            dims=("time", "index"),
            coords={"time": df_ts.index, "index": ["mag", "dir"]},
        )
        self.set(da, name="wind")

        # update the model config
        self.model.config.set("wndfile", "sfincs.wnd")
        if self.model.config.get("netamuamvfile", None) is not None:
            logger.warning(
                "netamuamvfile was previously set, but is now replaced by wndfile."
            )
            self.model.config.set("netamuamvfile", None)
        # update the default filename
        self._filename = "sfincs.wnd"


# %% DDB GUI focused additional functions:
# - yet unsupported in DDB-
