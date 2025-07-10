from pathlib import Path
from typing import List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import Transformer

from hydromt.model import Model
from hydromt.model.components import ModelComponent
from hydromt_sfincs import utils


class SfincsDischargePoints(ModelComponent):
    def __init__(
        self,
        model: Model,
    ):
        # self._filename: str = "sfincs.dis"  # FIXME - List(str = "sfincs.dis" and str = "sfincs.src" or str = "sfincs_netbndbzsbzi.nc")
        self._data = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Observation point data.

        Return geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

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
            self.read_discharge_points()
            # Check if there are any points
            if not self.data.empty:
                self.read_discharge_timeseries()
        elif format == "netcdf":
            # Read netcdf file
            self.read_discharges_netcdf()

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

        # HydroMT does not have open_vector at the moment ...
        # Read bnd file
        # gdf = utils.read_xy(abs_file_path, crs=self.model.crs)
        # # Add columns for timeseries and astro and add empty DataFrames
        # gdf["timeseries"] = pd.DataFrame()
        # gdf["astro"] = pd.DataFrame()
        # # Add to self.data
        # self.data = gdf

        # Read the bnd file
        df = pd.read_csv(
            abs_file_path, index_col=False, header=None, names=["x", "y"], sep="\s+"
        )

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = str(ind + 1).zfill(4)
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {
                "name": name,
                "timeseries": pd.DataFrame(),
                "geometry": point,
            }
            gdf_list.append(d)
        self._data = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

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

        # Now we need to split the timeseries into the different points
        for idx, row in self.data.iterrows():
            # Get the timeseries for this point
            ts = pd.DataFrame(df.iloc[:, idx])
            # Set the column name to wl
            ts.columns = ["q"]
            # # Set the index to time
            # ts.index.name = "time"
            # Add to the point
            self._data.at[idx, "timeseries"] = ts

    def read_discharge_conditions_netcdf(self, filename: str | Path = None):
        """Read SFINCS discharge conditions netcdf file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if netbndbzsbzifile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "netbndbzsbzifile", value=filename
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
        ds = xr.open_dataset(abs_file_path)

        # Loop through discharge points
        # FIXME - we first need to get the points!
        for ip, point in self.data.iterrows():
            # Get the timeseries for this point
            ts = ds["timeseries"].sel(point=ip).to_dataframe()
            # Add to the point
            self.data.at[ip, "timeseries"] = ts

            # Get the astro for this point
            astro = ds["astro"].sel(point=ip).to_dataframe()
            # Add to the point
            self._data.at[ip, "astro"] = astro

    def write(self, format: str = None):
        """Write SFINCS discharges (*.src, *.dis files) or netcdf file.

        The format of the discharge files can be specified,
        otherwise it is determined from the model configuration.

        Parameters
        ----------
        format : str, optional
            Format of the discharge files, "asc" (default), or "netcdf".
        """

        if self.data.empty:
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
            self.write_discharges_netcdf()

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

        utils.write_xyn(abs_file_path, self.data, fmt=fmt)

    def write_discharge_timeseries(self, filename: str | Path = None):
        """Write SFINCS discharge timeseries (*.dis) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bzsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "disfile", value=filename, default="sfincs.dis"
        )

        # Get all timeseries and stick in one DataFrame
        df = pd.DataFrame()
        for ip, point in self.data.iterrows():
            df = pd.concat([df, point["timeseries"]["q"]], axis=1)

        # Write to file
        # This does NOT work at the moment!
        # utils.write_timeseries(abs_file_path, df, self.model.config.get("tref"))

        # For now use 'ugly' to_csv method without control of column width
        # Convert time index to datetime64
        time = pd.to_datetime(df.index)
        tref = self.model.config.get("tref")
        time = (time - tref).total_seconds()
        df.index = time
        df.to_csv(
            abs_file_path, index=True, sep=" ", header=False, float_format="%0.3f"
        )

        # to_fwf(df, abs_file_path)

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set discharge data.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with discharge points.
        merge : bool, optional
            Merge data with existing data, by default True.
        """

        # when merge = False, clear the data with
        if not merge:
            self.clear()

        # TODO can this be done more efficiently?
        for i, row in gdf.iterrows():
            single_gdf = gdf.loc[[i]]
            # set discharge to zero
            self.add_point(
                gdf=single_gdf,
            )

    def add_point(
        self,
        gdf: gpd.GeoDataFrame = None,
        name: str = None,
        x: float = None,
        y: float = None,
        q: float = 0.0,
    ):
        """Add a single point to the discharge data. Either gdf,
        or x, y must be provided.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with a single point
        name : str
            Name of the point
        x : float
            x-coordinate of the point
        y : float
            y-coordinate of the point
        wl : float
            Water level of the point
        """
        if gdf is not None:
            if len(gdf) != 1:
                raise ValueError(
                    "Only GeoDataFrame with a single point in a can be added."
                )
            gdf = gdf.to_crs(self.model.crs)
            if "timeseries" not in gdf:
                gdf["timeseries"] = [pd.DataFrame()] * len(gdf)
            # reset index
            gdf = gdf.reset_index(drop=True)
        else:
            # Create a GeoDataFrame with a single point
            if x is None or y is None or name is None:
                raise ValueError("Either gdf or x, y, and name must be provided.")
            point = shapely.geometry.Point(x, y)
            gdf = gpd.GeoDataFrame(
                [
                    {
                        "name": name,
                        "timeseries": pd.DataFrame(),
                        "geometry": point,
                    }
                ],
                crs=self.model.crs,
            )

        # Check if there is data in the timeseries
        if gdf["timeseries"][0].empty:
            # Now add the water level
            if not self.data.empty:
                # Set water level at same times as first existing point by copying
                gdf.at[0, "timeseries"] = self.data.iloc[0]["timeseries"].copy()
                gdf.at[0, "timeseries"]["q"] = q
            else:
                # First point, so need to generate df with constant water level
                time = [self.model.config.get("tstart"), self.model.config.get("tstop")]
                q = [q] * 2
                # Create DataFrame with columns time and wl
                df = pd.DataFrame()
                df["time"] = time
                df["q"] = q
                df = df.set_index("time")
                gdf.at[0, "timeseries"] = df
        else:
            # Check if the timeseries is the same length as the first point
            if len(gdf["timeseries"][0]) != len(self.data.iloc[0]["timeseries"]):
                raise ValueError(
                    "Timeseries in gdf must be the same length as the first point in the discharge conditions data."
                )

        # Add to self.data
        self._data = pd.concat([self.data, gdf], ignore_index=True)

    def delete(self, index: Union[int, List[int]]):
        """Delete a single point from the discharge data.

        Parameters
        ----------
        index : int or list of int
            Index or list of indices of points to be deleted.
        """

        if self.data.empty:
            return

        if not isinstance(index, list):
            index = [index]
        # Check if indices are within range
        if any(x > (len(self.data.index) - 1) for x in index):
            raise ValueError("One of the indices exceeds length of index range!")
        self._data = self.data.drop(index).reset_index(drop=True)

        if self.data.empty:
            self.model.config.set("srcfile", None)
            self.model.config.set("disfile", None)
            self.model.config.set("netsrcdisfile", None)

    def clear(self):
        """Clean GeoDataFrame with discharge points."""
        self._data = gpd.GeoDataFrame()

    def set_timeseries(
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
            Time of the peak of the Gaussian wave [s]
        duration : float
            Duration of the Gaussian wave [s]
        """

        if self.data.empty:
            return

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
            # Not implemented
            return

        times = pd.date_range(
            start=t0, end=t1, freq=pd.tseries.offsets.DateOffset(seconds=dtsec)
        )

        if index is None:
            index = list(self.data.index)
        elif not isinstance(index, list):
            index = [index]

        for i in index:
            df = pd.DataFrame()
            df["time"] = times
            df["q"] = q
            df = df.set_index("time")
            self._data.at[i, "timeseries"] = df


# def to_fwf(df, fname, floatfmt=".3f"):
#     indx = df.index.tolist()
#     vals = df.values.tolist()
#     for it, t in enumerate(vals):
#         t.insert(0, indx[it])
#     content = tabulate(vals, [], tablefmt="plain", floatfmt=floatfmt)
#     open(fname, "w").write(content)
