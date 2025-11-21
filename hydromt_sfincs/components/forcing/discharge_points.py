import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from hydromt import hydromt_step
from hydromt.gis.vector import GeoDataset

from hydromt_sfincs import utils

from .boundary_conditions import SfincsBoundaryBase

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsDischargePoints(SfincsBoundaryBase):
    """Discharge point component for SFINCS models.

    This component handles reading and writing of discharge points and their
    associated time series data in SFINCS format, including both ASCII and netCDF files.
    """

    _default_varname = "dis"

    def __init__(self, model: "SfincsModel"):
        super().__init__(model)

    def read(self, format: str = None):
        """Read SFINCS discharge points (.dis, .src files) or netcdf file.

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
                self.set(df=df, gdf=gdf, merge=False, drop_duplicates=False)
        elif format == "netcdf":
            # Read netcdf file
            da = self.read_discharge_conditions_netcdf()
            self.set(geodataset=da, merge=False, drop_duplicates=False)

    def read_discharge_points(self, filename: str | Path = None):
        """Read SFINCS discharge points (.src) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "srcfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return gpd.GeoDataFrame()

        # Check if src file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(f"Discharge points file not found: {abs_file_path}")

        # Read bnd file
        # TODO check if we want read_xyn? Before we used read_xy, so without name column
        gdf = utils.read_xyn(abs_file_path, crs=self.model.crs)
        return gdf

    def read_discharge_timeseries(self, filename: str | Path = None):
        """Read SFINCS discharge condition timeseries (.dis) file"""

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
        ds = GeoDataset.from_netcdf(abs_file_path, crs=self.model.crs, chunks="auto")
        return ds

    def write(self, format: str = None):
        """Write SFINCS discharges (.src, .dis files) or netcdf file.

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

        if self.model.write_gis:
            utils.write_vector(
                self.gdf,
                name="dis",
                root=join(self.model.root.path, "gis"),
                logger=logger,
            )

    def write_discharge_points(self, filename: str | Path = None):
        """Write SFINCS discharge points (.src) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bndfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "srcfile", value=filename, default="sfincs.src"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write src file
        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # TODO check whether write_xyn or write_xy
        utils.write_xyn(abs_file_path, self.gdf, fmt=fmt)

    def write_discharge_timeseries(self, filename: str | Path = None):
        """Write SFINCS discharge timeseries (.dis) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bzsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "disfile", value=filename, default="sfincs.dis"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # parse data to dataframe
        da = self.data["dis"].transpose("time", ...)
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

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        ds = self.data.load()

        # Write netcdf file safely (might get locked, e..g in other notebooks)
        final_path = utils.write_netcdf_safely(ds, abs_file_path)
        if final_path != abs_file_path:
            self.model.config.set("netsrcdisfile", final_path.name)

    def delete(self, index: Union[int, List[int]]):
        "Delete boundary points and clear config when no points remain."
        super().delete(index)
        if self.nr_points == 0:
            self.model.config.set("srcfile", None)
            self.model.config.set("disfile", None)
            self.model.config.set("netsrcdisfile", None)

    def clear(self):
        "Clear boundary points and unset associated config keys."
        super().clear()
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
        drop_duplicates: bool = True,
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
        drop_duplicates : bool, optional
            If True, drop duplicate points in gdf based on 'name' column or geometry.

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
            )
            df_ts = da.transpose(..., da.vector.index_dim).to_pandas()
            gdf_locs = da.vector.to_gdf()
        elif timeseries is not None:
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

        self.set(df=df_ts, gdf=gdf_locs, merge=merge, drop_duplicates=drop_duplicates)
        # update config
        if geodataset is not None:
            # if a geodataset is used, keep the format to netcdf
            self.model.config.set("netsrcdisfile", "sfincs_netsrcdisfile.nc")
        else:
            self.model.config.set("srcfile", "sfincs.src")
            self.model.config.set("disfile", "sfincs.dis")
