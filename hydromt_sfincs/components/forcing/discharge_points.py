import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from hydromt import hydromt_step
from hydromt.gis.vector import GeoDataArray, GeoDataset
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsDischargePoints(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        # self._filename: str = "sfincs.dis"  # FIXME - List(str = "sfincs.dis" and str = "sfincs.src" or str = "sfincs_netbndbzsbzi.nc")
        self._data = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> xr.DataArray:
        """Discharge boundary conditions point data.

        Return xr.DataArray
        """
        if self._data is None:
            self._initialize()

        assert self._data is not None
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._data is None:
            self._data = xr.DataArray()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @property
    def nr_points(self) -> int:
        """Number of discharge points."""
        if hasattr(self.data, "index"):
            return len(self.data.index)
        else:
            return 0

    def read(self, format: str = None):
        """Read SFINCS discharge points (*.dis, *.src files) or netcdf file.

        The format of the discharge conditions files can be specified,
        otherwise it is determined from the model configuration.

        Parameters
        ----------
        format : str, optional
            Format of the discharge files, "asc" or "netcdf".
        """

        if format is None:
            if self.model.config.get("netsrcdisfile"):
                format = "netcdf"
            else:
                format = "asc"

        if format == "asc":
            gdf = self.read_discharge_points()
            # Check if there are any points
            if not gdf.empty:
                df = self.read_discharge_timeseries()
            self.set(df=df, gdf=gdf, merge=False)
        elif format == "netcdf":
            # Read netcdf file
            da = self.read_discharge_conditions_netcdf()
            self.set(geodataset=da, merge=False)

    def read_discharge_points(self, filename: str | Path = None):
        """Read SFINCS discharge points (*.src) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "srcfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if src file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(f"Discharge points file not found: {abs_file_path}")

        # Read bnd file
        # TODO check if we want read_xyn? Before we used read_xy, so without name column
        gdf = utils.read_xyn(abs_file_path, crs=self.model.crs)
        return gdf

    def read_discharge_timeseries(self, filename: str | Path = None):
        """Read SFINCS discharge condition timeseries (*.bzs) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "disfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if dis file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Discharge timeseries file not found: {abs_file_path}"
            )

        # Read bzs file (this creates one DataFrame with all timeseries)
        df = utils.read_timeseries(abs_file_path, tref=self.model.config.get("tref"))
        df.index.name = "time"
        df.columns.name = "index"
        return df

    def read_discharge_conditions_netcdf(self, filename: str | Path = None):
        """Read SFINCS discharge conditions netcdf file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if netsrcdisfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "netsrcdisfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if netbndbzsbzifile exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"discharge condition netcdf file not found: {abs_file_path}"
            )

        # Read netcdf file
        ds = GeoDataArray.from_netcdf(abs_file_path, crs=self.model.crs, chunks="auto")
        return ds

    def write(self, format: str = None):
        """Write SFINCS discharges (*.src, *.dis files) or netcdf file.

        The format of the discharge files can be specified,
        otherwise it is determined from the model configuration.

        Parameters
        ----------
        format : str, optional
            Format of the discharge files, "asc" (default), or "netcdf".
        """

        if self.nr_points == 0:
            # There are no discharge points
            return

        if format is None:
            if self.model.config.get("netsrcdisfile"):
                format = "netcdf"
            else:
                format = "asc"

        if format == "asc":
            self.write_discharge_points()
            self.write_discharge_timeseries()
        else:
            self.write_discharge_conditions_netcdf()

    def write_discharge_points(self, filename: str | Path = None):
        """Write SFINCS discharge points (*.src) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bndfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "srcfile", value=filename, default="sfincs.src"
        )

        # Write src file
        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # parse data to geodataframe
        try:
            gdf = self.data.vector.to_gdf()
        except Exception:
            raise ValueError(f"Locations missing for discharge forcing")

        # TODO check whether write_xyn or write_xy
        utils.write_xyn(abs_file_path, gdf, fmt=fmt)

    def write_discharge_timeseries(self, filename: str | Path = None):
        """Write SFINCS discharge timeseries (*.dis) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bzsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "disfile", value=filename, default="sfincs.dis"
        )

        # parse data to dataframe
        da = self.data.transpose("time", ...)
        df = da.to_pandas()

        # Write to file
        utils.write_timeseries(abs_file_path, df, self.model.config.get("tref"))

    def write_discharge_conditions_netcdf(self, filename: str | Path = None):
        """Write SFINCS discharge conditions netcdf file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if netsrcdisfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "netsrcdisfile", value=filename, default="sfincs_netsrcdisfile.nc"
        )
        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        ds = self.data
        ds.vector.to_netcdf(abs_file_path)

    def add_point(
        self,
        x: float,
        y: float,
        name: str = None,
        discharge: float = 0.0,
    ):
        """Add a single point to the discharge data.

        Parameters
        ----------
        x : float
            x-coordinate of the point
        y : float
            y-coordinate of the point
        name : str, optional
            Name of the point.
        discharge : float, optional
            Discharge of the point. Defaults to 0.0.
        """

        # Check how many points are present
        new_index = self.nr_points + 1

        # Create a GeoDataFrame with a single point
        if name is None:
            name = f"point_{new_index}"

        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([x], [y]), crs=self.model.crs
        )
        gdf["name"] = name

        self.set_locations(gdf=gdf, discharge=discharge, merge=True)

    def delete(self, index: Union[int, List[int]]):
        """Delete a single point from the discharge data.

        Parameters
        ----------
        index : int or list of int
            Index or list of indices of points to be deleted.
        """

        if self.nr_points == 0:
            return

        if not isinstance(index, list):
            index = [index]

        # Check if indices are within range
        if any(x > (self.nr_points - 1) for x in index):
            raise ValueError("One of the indices exceeds length of index range!")
        self._data = self.data.drop_isel(index=index)

        if self.nr_points == 0:
            self.model.config.set("srcfile", None)
            self.model.config.set("disfile", None)
            self.model.config.set("netsrcdisfile", None)

    def clear(self):
        """Clean GeoDataFrame with discharge points."""
        self._data = xr.DataArray()

        self.model.config.set("srcfile", None)
        self.model.config.set("disfile", None)
        self.model.config.set("netsrcdisfile", None)

    @hydromt_step
    def create_timeseries(
        self,
        index: Union[int, List[int]] = None,
        shape: str = "constant",
        timestep: float = 600.0,
        offset: float = 0.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
        period: float = 43200.0,
        peak: float = 1.0,
        tpeak: float = 86400.0,
        duration: float = 43200.0,
    ):
        """Applies time series discharges for each point
        Create numpy datetime64 array for time series with python datetime.datetime objects

        Parameters
        ----------
        shape : str
            Shape of the time series. Options are "constant", "sine", or "gaussian".
        timestep : float
            Time step [s]
        offset : float
            Offset of the time series [m]
        amplitude : float
            Amplitude of the sine wave [m]
        phase : float
            Phase of the sine wave [degrees]
        period : float
            Period of the sine wave [s]
        peak : float
            Peak of the Gaussian wave [m]
        tpeak : float
            Time of the peak of the Gaussian wave [s] with respect to the model reference time
        duration : float
            Duration of the Gaussian wave [s]
        """

        if self.nr_points == 0:
            raise ValueError(
                "Cannot create timeseries without existing discharge points"
            )

        t0 = np.datetime64(self.model.config.get("tstart"))
        t1 = np.datetime64(self.model.config.get("tstop"))
        if shape == "constant":
            dt = np.timedelta64(int((t1 - t0).astype(float) / 1e6), "s")
        else:
            dt = np.timedelta64(int(timestep), "s")
        time = np.arange(t0, t1 + dt, dt)
        dtsec = dt.astype(float)
        # Convert time to seconds since tref
        tsec = (
            (time - np.datetime64(self.model.config.get("tref")))
            .astype("timedelta64[s]")
            .astype(float)
        )
        nt = len(tsec)
        if shape == "constant":
            q = [offset] * nt
        elif shape == "sine":
            q = offset + amplitude * np.sin(
                2 * np.pi * tsec / period + phase * np.pi / 180
            )
        elif shape == "gaussian":
            q = offset + peak * np.exp(-(((tsec - tpeak) / (0.25 * duration)) ** 2))
        else:
            raise NotImplementedError(
                f"Shape '{shape}' is not implemented. Use 'constant', 'sine', or 'gaussian'."
            )

        times = pd.date_range(
            start=t0, end=t1, freq=pd.tseries.offsets.DateOffset(seconds=dtsec)
        )

        if index is None:
            index = list(self.data.index.values)
        elif not isinstance(index, list):
            index = [index]

        # Create DataFrame: rows = time, columns = locations (index), values = q (same for all)
        df = pd.DataFrame(
            data=np.tile(q, (len(index), 1)).T, index=times, columns=index
        )

        # Call set_timeseries to update your object's data
        self.set_timeseries(df)

    @hydromt_step
    def create(
        self,
        geodataset=None,
        timeseries=None,
        locations=None,
        merge=True,
        buffer: float = None,
    ):
        """Setup discharge forcing.

        Discharge timeseries are read from a `geodataset` (geospatial point timeseries)
        or a tabular `timeseries` dataframe. At least one of these must be provided.

        The tabular timeseries data is combined with `locations` if provided,
        or with existing 'src' locations if previously set, e.g., with the
        `setup_river_inflow` method.

        Adds model layers:

        * **dis** forcing: discharge time series [m3/s]

        Parameters
        ----------
        geodataset: str, Path, xr.Dataset, optional
            Path, data source name, or xarray data object for geospatial point timeseries.
        timeseries: str, Path, pd.DataFrame, optional
            Path, data source name, or pandas data object for tabular timeseries.
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for bnd point locations.
            It should contain a 'index' column matching the column names in `timeseries`.
        merge : bool, optional
            If True, merge locations with existing forcing data, by default True.
        buffer: float, optional
            Buffer [m] around model boundary within the model region
            select discharge gauges, by default None.

        See Also
        --------
        setup_river_inflow
        """

        gdf_locs, df_ts = None, None
        tstart, tstop = self.model.get_model_time()  # model time
        # buffer
        region = self.model.region
        if buffer is not None:  # TODO this assumes the model crs is projected
            region = region.boundary.buffer(buffer).clip(self.model.region)
        # read discharge data from geodataset or geodataframe
        if geodataset is not None:
            # read and clip data in time & space
            da = self.data_catalog.get_geodataset(
                geodataset,
                geom=region,
                variables=["discharge"],
                time_range=(tstart, tstop),
                crs=self.model.crs,
            )
            df_ts = da.transpose(..., da.vector.index_dim).to_pandas()
            gdf_locs = da.vector.to_gdf()
        elif timeseries is not None:
            df_ts = self.data_catalog.get_dataframe(
                timeseries,
                time_range=(tstart, tstop),
                driver={
                    "name": "pandas",
                    "options": {"index_col": 0, "parse_dates": True},
                },
            )
            df_ts.columns = df_ts.columns.map(int)  # parse column names to integers

        # read location data (if not already read from geodataset)
        if gdf_locs is None and locations is not None:
            gdf_locs = self.data_catalog.get_geodataframe(
                locations,
                geom=region,
            ).to_crs(self.model.crs)
            if "index" in gdf_locs.columns:
                gdf_locs = gdf_locs.set_index("index")
            # filter df_ts timeseries based on gdf_locs index
            # this allows to use a subset of the locations in the timeseries
            if df_ts is not None and np.isin(gdf_locs.index, df_ts.columns).all():
                df_ts = df_ts.reindex(gdf_locs.index, axis=1, fill_value=0)
        elif gdf_locs is None and self.data is not None:
            logger.info(
                "No locations provided, using existing discharge points from data."
            )
            # gdf_locs = self.data.vector.to_gdf() #NOTE this is now done in set_timeseries ...
        elif gdf_locs is None:
            raise ValueError("No discharge boundary (src) points provided.")

        self.set(df=df_ts, gdf=gdf_locs, merge=merge)

    def set(
        self,
        df: pd.DataFrame = None,
        gdf: gpd.GeoDataFrame = None,
        geodataset: "GeoDataArray" = None,
        merge: bool = True,
    ):
        """Set discharge data using a GeoDataArray or df + gdf combo."""
        if geodataset is not None:
            if df is not None or gdf is not None:
                raise ValueError(
                    "Provide either 'geodataset' or ('df' and 'gdf'), not both."
                )
            if not hasattr(geodataset, "vector") or not hasattr(geodataset, "dims"):
                raise ValueError("Invalid GeoDataArray provided")
            if geodataset.vector.crs != self.model.crs:
                geodataset = geodataset.vector.to_crs(self.model.crs)
            self._data = geodataset.transpose("time", "index")
            return

        if df is None and gdf is None:
            raise ValueError("Must provide 'df' or 'gdf' (or a GeoDataArray)")

        # update locations and timeseries
        if gdf is not None:
            new_indices = self.set_locations(gdf, merge=merge)
            # merging might alter the indices, so we need to update df
            if df is not None:
                df.columns = new_indices
        if df is not None:
            self.set_timeseries(df)

    def set_locations(
        self, gdf: gpd.GeoDataFrame, discharge: float = 0.0, merge: bool = True
    ):
        """Add or update discharge locations. Create dummy timeseries if needed."""
        gdf = self._validate_and_prepare_gdf(gdf)

        if self.nr_points > 0 and merge:
            # parse data to dataframe
            df0 = self.data.transpose(..., self.data.vector.index_dim).to_pandas()
            gdf0 = self.data.vector.to_gdf()

            # TODO drop based on name instead of index?
            # if set(gdf0.index) != set(gdf.index):
            #     # merge locations; overwrite existing locations with the same index/name
            #     gdf0 = gdf0.drop(gdf.index, errors="ignore")
            #     df0 = df0.reindex(gdf0.index, axis=1, fill_value=0)

            if "name" in gdf0.columns and "name" in gdf.columns:
                if set(gdf0.name) != set(gdf.name):
                    # merge locations; overwrite existing locations with the same name
                    gdf0 = gdf0[
                        ~gdf0.name.isin(gdf.name)
                    ]  # drop rows with matching names
                    df0 = df0.reindex(gdf0.index, axis=1, fill_value=0)

            # create a similar df with the same index as the first point but with columns of gdf
            df_new = pd.DataFrame(index=df0.index, columns=gdf.index, data=discharge)

            gdf = self._align_gdf_and_df(gdf, df_new)

            # merge the new data with the existing data
            gdf_combined = pd.concat([gdf0, gdf], ignore_index=True)
            df_combined = pd.concat([df0, df_new], axis=1)
            df_combined.columns = gdf_combined.index

            # set the data and return new indices
            new_indices = gdf_combined.index.difference(range(len(gdf0)))
            self._finalize_set(df_combined, gdf_combined)
        else:
            # make sure indices start at 0 and are unique
            gdf = gdf.reset_index(drop=True)

            # Overwrite with dummy timeseries
            df_new = pd.DataFrame(
                index=pd.date_range(*self.model.get_model_time(), periods=2),
                columns=gdf.index,
                data=discharge,
            )
            gdf = self._align_gdf_and_df(gdf, df_new)

            # Set the data and return new indices
            new_indices = gdf.index
            self._finalize_set(df_new, gdf)

        return new_indices

    def set_timeseries(self, df: pd.DataFrame):
        """Add or update timeseries for existing locations. Only works if locations already exist."""
        df = self._validate_and_prepare_df(df)

        if self.nr_points == 0:
            raise ValueError("Cannot set timeseries without existing locations")

        gdf = self.data.vector.to_gdf()
        existing_df = self.data.transpose(..., "index").to_pandas()

        # Merge time series
        existing_df = existing_df.drop(columns=df.columns, errors="ignore")
        df = pd.concat([existing_df, df], axis=1).sort_index()
        df = df.interpolate().bfill().fillna(0)

        self._finalize_set(df, gdf)

    def _validate_and_prepare_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate and prepare GeoDataFrame for discharge points. If gdf is None, use existing data."""
        if gdf is None:
            if self.nr_points > 0:
                gdf = self.data.vector.to_gdf()
            else:
                raise ValueError("gdf must be provided if no data exists yet")

        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("gdf must be a GeoDataFrame")
        if not gdf.index.is_integer() and gdf.index.is_unique:
            raise ValueError("gdf index must be unique integers")
        if not gdf.geometry.type.isin(["Point"]).all():
            raise ValueError("gdf geometry must be Point")
        if gdf.crs != self.model.crs:
            gdf = gdf.to_crs(self.model.crs)

        # Make sure gdf is within model.region
        # FIXME : this will drop points and always needs the grid to be availabel ... therefore tests fail
        # if not gdf.geometry.within(self.model.region).all():
        #     logger.warning(
        #         "Some discharge points are outside the active model region. They will be ignored."
        #     )
        #     gdf = gdf[gdf.geometry.within(self.model.region)]

        return gdf

    def _validate_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare DataFrame for discharge timeseries."""
        if df is None:
            return
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a DataFrame")
        if not df.columns.is_integer() and df.columns.is_unique:
            raise ValueError("df column names must be unique integers")

        if df.index.inferred_type in ["integer", "floating"]:
            if self.model.config.get("tref") is None:
                raise ValueError(
                    "tref must be set in config to convert numeric index to datetime"
                )
            tref = self.model.config.get("tref")
            df.index = tref + pd.to_timedelta(df.index, unit="s")

        tstart, tstop = self.model.get_model_time()
        if df.index.min() > tstart or df.index.max() < tstop:
            logger.warning(
                "The provided timeseries does not cover the entire model time period."
            )
        if df.shape[0] < 2:
            raise ValueError(
                "The provided timeseries must have at least two data points."
            )

        return df

    def _align_gdf_and_df(
        self, gdf: gpd.GeoDataFrame, df: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        if gdf.index.size == df.columns.size and not set(gdf.index) == set(df.columns):
            for col in gdf.select_dtypes(include=np.integer).columns:
                if set(gdf[col]) == set(df.columns):
                    gdf = gdf.set_index(col)
                    logger.info(f"Setting gdf index to column '{col}'")
                    break
            else:
                gdf = gdf.set_index(df.columns)
                logger.info(
                    "No matching column found in gdf; assuming order is correct"
                )

        if not set(gdf.index) == set(df.columns):
            raise ValueError("gdf index and df columns must match")

        return gdf

    def _finalize_set(self, df: pd.DataFrame, gdf: gpd.GeoDataFrame):
        """Finalize internal state update."""
        gdf.index.name = "index"
        df.columns.name = "index"
        df.index.name = "time"

        da = GeoDataArray.from_gdf(gdf.to_crs(self.model.crs), data=df, name="dis")
        self._data = da.transpose("time", "index")
