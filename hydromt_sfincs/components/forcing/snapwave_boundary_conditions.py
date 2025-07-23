from pathlib import Path
from typing import List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely
import xarray as xr

from hydromt.model import Model
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils


class SnapWaveBoundaryConditions(ModelComponent):
    def __init__(
        self,
        model: Model,
    ):
        self.data = gpd.GeoDataFrame()
        super().__init__(
            model=model,
        )

    # @property
    # def data(self) -> gpd.GeoDataFrame:
    #     """Water level boundary conditions data.

    #     Return pd.GeoDataFrame
    #     """
    #     if self._data is None:
    #         self._initialize()
    #     return self._data

    # def _initialize(self) -> None:
    #     """Initialize boundary conditions data."""
    #     if self._data is None:
    #         self._data = gpd.GeoDataFrame()

    def read(self, format: str = None):
        """Read SFINCS boundary conditions (*.bnd, *.bzs, *.bca files) or netcdf file.

        The format of the boundary conditions files can be specified,
        otherwise it is determined from the model configuration.

        Parameters
        ----------
        format : str, optional
            Format of the boundary conditions files, "asc" or "netcdf".
        """

        if format is None:
            if self.model.config.get("netbndbzsbzifile"):
                format = "netcdf"
            else:
                format = "asc"

        if format == "asc":
            self.read_boundary_points()
            # Check if there are any points
            if not self.data.empty:
                self.read_boundary_conditions_timeseries()
                # Read astro if bcafile is defined
                if self.model.config.get("bcafile"):
                    self.read_boundary_conditions_astro()
        elif format == "netcdf":
            # Read netcdf file
            self.read_boundary_conditions_netcdf()

    def read_boundary_points(self, filename: str | Path = None):
        """Read SnapWave boundary condition points (*.bnd) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bndfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if bnd file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Boundary condition points file not found: {abs_file_path}"
            )

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
        self.data = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def read_boundary_conditions_timeseries(self, filename: str | Path = None):
        """Read SFINCS boundary condition timeseries (*.bhs, *.btp, *.bwd, *.bds) files"""

        # Check that read mode is on
        self.root._assert_read_mode()

        ### BHS file

        # Get absolute file name and set it in config if snapwave_bhsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bhsfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if bzs file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Boundary condition timeseries file not found: {abs_file_path}"
            )

        # Read bhs file (this creates one DataFrame with all timeseries)
        dfhs = utils.read_timeseries(abs_file_path, tref=self.model.config.get("tref"))

        ### BTP file

        # Get absolute file name and set it in config if snapwave_bhsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_btpfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if bzs file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Boundary condition timeseries file not found: {abs_file_path}"
            )

        # Read btp file (this creates one DataFrame with all timeseries)
        dftp = utils.read_timeseries(abs_file_path, tref=self.model.config.get("tref"))

        ### BWD file

        # Get absolute file name and set it in config if snapwave_bwdfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bwdfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if snapwave_bwd file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Boundary condition timeseries file not found: {abs_file_path}"
            )

        # Read bwd file (this creates one DataFrame with all timeseries)
        dfwd = utils.read_timeseries(abs_file_path, tref=self.model.config.get("tref"))

        ### BDS file

        # Get absolute file name and set it in config if snapwave_bdsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bdsfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if snapwave_bds file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Boundary condition timeseries file not found: {abs_file_path}"
            )

        # Read bds file (this creates one DataFrame with all timeseries)
        dfds = utils.read_timeseries(abs_file_path, tref=self.model.config.get("tref"))

        # Now we need to split the timeseries into the different points
        for idx, row in self.data.iterrows():
            # Get the timeseries for this point
            tshs = pd.DataFrame(dfhs.iloc[:, idx])
            tstp = pd.DataFrame(dftp.iloc[:, idx])
            tswd = pd.DataFrame(dfwd.iloc[:, idx])
            tsds = pd.DataFrame(dfds.iloc[:, idx])
            # Set the column name to hs
            tshs.columns = ["hs"]
            tstp.columns = ["tp"]
            tswd.columns = ["wd"]
            tsds.columns = ["ds"]
            # Concatenate the DataFrames
            ts = pd.concat([tshs, tstp, tswd, tsds], axis=1)
            # Add to the point
            self.data.at[idx, "timeseries"] = ts

    def read_boundary_conditions_netcdf(self, filename: str | Path = None):
        """Read SFINCS boundary conditions netcdf file"""

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
                f"Boundary condition netcdf file not found: {abs_file_path}"
            )

        # Read netcdf file
        ds = xr.open_dataset(abs_file_path)

        # Loop through boundary points
        # FIXME - we first need to get the points!
        for ip, point in self.data.iterrows():
            # Get the timeseries for this point
            ts = ds["timeseries"].sel(point=ip).to_dataframe()
            # Add to the point
            self.data.at[ip, "timeseries"] = ts

        ds.close()

    def write(self, format: str = None):
        """Write SnapWave boundary conditions (*.bnd, *.bhs, *.btp, *.bwd, *.bds files) or netcdf file.

        The format of the boundary conditions files can be specified,
        otherwise it is determined from the model configuration.

        Parameters
        ----------
        format : str, optional
            Format of the boundary conditions files, "asc" (default), or "netcdf".
        """

        if self.data.empty:
            # There are no boundary points
            return

        if format is None:
            if self.model.config.get("netbndsnapwavefile"):
                format = "netcdf"
            else:
                format = "asc"

        if format == "asc":
            self.write_boundary_points()
            self.write_boundary_conditions_timeseries()
        else:
            self.write_boundary_conditions_netcdf()

    def write_boundary_points(self, filename: str | Path = None):
        """Write SnapWave boundary condition points (*.bnd) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bndfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bndfile", value=filename, default="snapwave.bnd"
        )

        # Write bnd file
        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"
        utils.write_xy(abs_file_path, self.data, fmt=fmt)

    def write_boundary_conditions_timeseries(self, filename: str | Path = None):
        """Write SnapWave boundary condition timeseries (*.bhs, *.btp, *.bwd, *.bds) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # BHS

        # Get absolute file name and set it in config if snapwave_bhsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bhsfile", value=filename, default="snapwave.bhs"
        )

        # Get all timeseries and stick in one DataFrame
        df = pd.DataFrame()
        for ip, point in self.data.iterrows():
            df = pd.concat([df, point["timeseries"]["hs"]], axis=1)

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

        # BTP

        # Get absolute file name and set it in config if snapwave_btpfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_btpfile", value=filename, default="snapwave.btp"
        )

        # Get all timeseries and stick in one DataFrame
        df = pd.DataFrame()
        for ip, point in self.data.iterrows():
            df = pd.concat([df, point["timeseries"]["tp"]], axis=1)

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

        # BWD

        # Get absolute file name and set it in config if snapwave_bwdfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bwdfile", value=filename, default="snapwave.bwd"
        )

        # Get all timeseries and stick in one DataFrame
        df = pd.DataFrame()
        for ip, point in self.data.iterrows():
            df = pd.concat([df, point["timeseries"]["wd"]], axis=1)

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

        # BDS

        # Get absolute file name and set it in config if snapwave_bdsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bdsfile", value=filename, default="snapwave.bds"
        )

        # Get all timeseries and stick in one DataFrame
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

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SnapWave boundary conditions data.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with boundary points.
        merge : bool, optional
            Merge data with existing data, by default True.
        """

        if merge:
            self.data = pd.concat([self.data, gdf], ignore_index=True)
        else:
            self.data = gdf

    def add_point(
        self,
        gdf: gpd.GeoDataFrame = None,
        x: float = None,
        y: float = None,
        hs: float = 1.0,
        tp: float = 10.0,
        wd: float = 270.0,
        ds: float = 20.0,
    ):
        """Add a single point to the boundary conditions data. Either gdf,
        or x, y must be provided.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with a single point
        x : float
            x-coordinate of the point
        y : float
            y-coordinate of the point
        hs : float
            Wave height of the point
        tp : float
            Peak period of the point
        wd : float
            Wave direction of the point
        ds : float
            Directional spread of the point
        """
        if gdf is not None:
            if len(gdf) != 1:
                raise ValueError(
                    "Only GeoDataFrame with a single point in a can be added."
                )
            gdf = gdf.to_crs(self.model.crs)
            if "timeseries" not in gdf:
                gdf["timeseries"] = pd.DataFrame()
        else:
            # Create a GeoDataFrame with a single point
            if x is None or y is None:
                raise ValueError("Either gdf or x, y, and name must be provided.")
            point = shapely.geometry.Point(x, y)
            gdf = gpd.GeoDataFrame(
                [
                    {
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
                gdf.at[0, "timeseries"]["hs"] = hs
                gdf.at[0, "timeseries"]["tp"] = tp
                gdf.at[0, "timeseries"]["wd"] = wd
                gdf.at[0, "timeseries"]["ds"] = ds
            else:
                # First point, so need to generate df with constant water level
                time = [self.model.config.get("tstart"), self.model.config.get("tstop")]
                hs = [hs] * 2
                tp = [tp] * 2
                wd = [wd] * 2
                ds = [ds] * 2
                # Create DataFrame with columns time and wl
                df = pd.DataFrame()
                df["time"] = time
                df["hs"] = hs
                df["tp"] = tp
                df["wd"] = wd
                df["ds"] = ds
                df = df.set_index("time")
                gdf.at[0, "timeseries"] = df
        else:
            # Check if the timeseries is the same length as the first point
            if len(gdf["timeseries"][0]) != len(self.data.iloc[0]["timeseries"]):
                raise ValueError(
                    "Timeseries in gdf must be the same length as the first point in the boundary conditions data."
                )

        # Add to self.data
        self.data = pd.concat([self.data, gdf], ignore_index=True)

    def delete(self, index: Union[int, List[int]]):
        """Delete a single point from the SnapWave boundary conditions data.

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
        self.data = self.data.drop(index).reset_index(drop=True)

        if self.data.empty:
            self.model.config.set("snapwave_bndfile", None)
            self.model.config.set("snapwave_btpfile", None)
            self.model.config.set("snapwave_bwdfile", None)
            self.model.config.set("snapwave_bdsfile", None)
            # self.model.config.set("netbndsnapwavefile", None)

    def clear(self):
        """Clean GeoDataFrame with boundary points."""
        self.data = gpd.GeoDataFrame()

    def set_timeseries(
        self,
        index: Union[int, List[int]] = None,
        shape: str = "constant",
        timestep: float = 600.0,
        hs: float = 1.0,
        tp: float = 10.0,
        wd: float = 270.0,
        ds: float = 20.0,
        tpeak: float = 86400.0,
        duration: float = 43200.0,
    ):
        """Applies time series boundary conditions for each point
        Create numpy datetime64 array for time series with python datetime.datetime objects

        Parameters
        ----------
        shape : str
            Shape of the time series. Options are "constant" or "gaussian".
        timestep : float
            Time step [s]
        hs : float
            Wave height [m]
        tp : float
            Peak period [s]
        wd : float
            Wave direction [degrees]
        ds : float
            Directional spread [degrees]
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
            hs = [hs] * nt
            tp = [tp] * nt
            wd = [wd] * nt
            ds = [ds] * nt
        elif shape == "gaussian":
            hs = hs * np.exp(-(((tsec - tpeak) / (0.25 * duration)) ** 2))
            tp = [tp] * nt
            wd = [wd] * nt
            ds = [ds] * nt
        else:
            # Not implemented
            raise ValueError(
                f"Shape {shape} not implemented for SnapWave boundary conditions!"
            )

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
            df["hs"] = hs
            df["tp"] = tp
            df["wd"] = wd
            df["ds"] = ds
            df = df.set_index("time")
            self.data.at[i, "timeseries"] = df

    def get_boundary_points_from_mask(self, min_dist=None, bnd_dist=5000.0):
        # Should move this to mask? Yes.
        if min_dist is None:
            # Set minimum distance between to grid boundary points on polyline to 2 * dx
            min_dist = self.model.quadtree_grid.data.attrs["dx"] * 2

        mask = self.model.quadtree_grid.data["snapwave_mask"]
        ibnd = np.where(mask == 2)
        xz, yz = self.model.quadtree_grid.face_coordinates()
        xp = xz[ibnd]
        yp = yz[ibnd]

        # Make boolean array for points that are include in a polyline
        used = np.full(xp.shape, False, dtype=bool)

        # Make list of polylines. Each polyline is a list of indices of boundary points.
        polylines = []

        while True:
            if np.all(used):
                # All boundary grid points have been used. We can stop now.
                break

            # Find first the unused points
            i1 = np.where(~used)[0][0]

            # Set this point to used
            used[i1] = True

            # Start new polyline with index i1
            polyline = [i1]

            while True:
                # Compute distances to all points that have not been used
                xpunused = xp[~used]
                ypunused = yp[~used]
                # Get all indices of unused points
                unused_indices = np.where(~used)[0]

                dst = np.sqrt((xpunused - xp[i1]) ** 2 + (ypunused - yp[i1]) ** 2)
                if np.all(np.isnan(dst)):
                    break
                inear = np.nanargmin(dst)
                inearall = unused_indices[inear]
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.append(inearall)
                    used[inearall] = True
                    i1 = inearall
                else:
                    # Last point found
                    break

            # Now work the other way
            # Start with first point of polyline
            i1 = polyline[0]
            while True:
                if np.all(used):
                    # All boundary grid points have been used. We can stop now.
                    break
                # Now we go in the other direction
                xpunused = xp[~used]
                ypunused = yp[~used]
                unused_indices = np.where(~used)[0]
                dst = np.sqrt((xpunused - xp[i1]) ** 2 + (ypunused - yp[i1]) ** 2)
                inear = np.nanargmin(dst)
                inearall = unused_indices[inear]
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.insert(0, inearall)
                    used[inearall] = True
                    # Set index of next point
                    i1 = inearall
                else:
                    # Last nearby point found
                    break

            if len(polyline) > 1:
                polylines.append(polyline)

        gdf_list = []
        ip = 0
        # Transform to web mercator to get distance in metres
        if self.model.crs.is_geographic:
            transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        # Loop through polylines
        for polyline in polylines:
            x = xp[polyline]
            y = yp[polyline]
            points = [(x, y) for x, y in zip(x.ravel(), y.ravel())]
            line = shapely.geometry.LineString(points)
            if self.model.crs.is_geographic:
                # Line in web mercator (to get length in metres)
                xm, ym = transformer.transform(x, y)
                pointsm = [(xm, ym) for xm, ym in zip(xm.ravel(), ym.ravel())]
                linem = shapely.geometry.LineString(pointsm)
                num_points = int(linem.length / bnd_dist) + 2
            else:
                num_points = int(line.length / bnd_dist) + 2
            # Interpolate to new points
            new_points = [
                line.interpolate(i / float(num_points - 1), normalized=True)
                for i in range(num_points)
            ]
            # Loop through points in polyline
            for point in new_points:
                name = str(ip + 1).zfill(4)
                d = {
                    "name": name,
                    "timeseries": pd.DataFrame(),
                    "geometry": point,
                }
                gdf_list.append(d)
                ip += 1

        self.data = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

        self.set_timeseries(
            shape="constant",
            timestep=600.0,
            hs=1.0,
            tp=10.0,
            wd=270.0,
            ds=20.0,
        )


# def to_fwf(df, fname, floatfmt=".3f"):
#     indx = df.index.tolist()
#     vals = df.values.tolist()
#     for it, t in enumerate(vals):
#         t.insert(0, indx[it])
#     content = tabulate(vals, [], tablefmt="plain", floatfmt=floatfmt)
#     open(fname, "w").write(content)
