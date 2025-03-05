import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, List
import shapely
from pyproj import Transformer

from hydromt.model.components import ModelComponent
from hydromt.model import Model
from hydromt_sfincs import utils

# Are we now importing from CHT packages ?!
# from cht_tide import predict


class SfincsBoundaryConditions(ModelComponent):
    def __init__(
        self,
        model: Model,
    ):
        self._data = gpd.GeoDataFrame()
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Water level boundary conditions data.

        Return pd.GeoDataFrame
        """
        if self._data.is_empty:
            # Does this not lead to an inifinite loop,
            # if self._data is empty and we use self.data?
            # At least it seems to, if we use self.data in the other methods
            # I still don't really understand why need this.
            self.read()
        return self._data

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
            if not self._data.empty:
                self.read_boundary_conditions_timeseries()
                # Read astro if bcafile is defined
                if self.model.config.get("bcafile"):
                    self.read_boundary_conditions_astro()
        elif format == "netcdf":
            # Read netcdf file
            self.read_boundary_conditions_netcdf()

    def write(self, format: str = None):
        """Write SFINCS boundary conditions (*.bnd, *.bzs, *.bca files) or netcdf file.

        The format of the boundary conditions files can be specified,
        otherwise it is determined from the model configuration.

        Parameters
        ----------
        format : str, optional
            Format of the boundary conditions files, "asc" (default), or "netcdf".
        """

        if self._data.empty:
            # There are no boundary points
            return

        if format is None:
            if self.model.config.get("netbndbzsbzifile"):
                format = "netcdf"
            else:
                format = "asc"

        if format == "asc":
            self.write_boundary_points()
            self.write_boundary_conditions_timeseries()
            if self.model.config.get("bcafile"):
                self.write_boundary_conditions_astro()
        else:
            self.write_boundary_conditions_netcdf()

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set boundary conditions data.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with boundary points.
        merge : bool, optional
            Merge data with existing data, by default True.
        """

        if merge:
            self._data = pd.concat([self._data, gdf], ignore_index=True)
        else:
            self._data = gdf

    def add(
        self,
        gdf: Union[gpd.GeoDataFrame, Path, str],
        merge: bool = True,
        wl: float = 0.0,
    ):
        """Add boundary conditions data.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame, str, or Path
            GeoDataFrame with boundary points, str or Path to geojson file.
        merge : bool, optional
            Merge data with existing data, by default True.
        wl : float, optional
            Water level of the point, by default 0.0.
        """
        # So basically the same as set?
        # Or should we first make sure that the CRS is the same?
        # Or do we do that in the set method?
        # Also, do we expect that gdf has "timeseries" and "astro" columns?
        # If not, we should add them here. Maybe let add_point
        # call the add method? Or the other way around?
        # After setting crs, we could loopthrough the rows and call add_point?
        # So many quandaries!
        # I suggest we get rid of add_point, and allow x, y, wl to be passed to add
        # But then gdf would be optional,
        # and we would need to check if x, y, wl are provided.
        # So maybe just do it like this ...

        # Check if gdf is a string or path
        if isinstance(gdf, (str, Path)):
            if isinstance(gdf, str):
                gdf = Path(gdf)
            if not gdf.exists():
                raise FileNotFoundError(f"File not found: {gdf}")
            gdf = gpd.read_file(gdf)
        gdf = gdf.to_crs(self.model.crs)

        # Now loop through points and add them
        for ip, point in gdf.iterrows():
            self.add_point(point.geometry.x, point.geometry.y, wl=wl, merge=merge)

    def delete(self, index: Union[int, List[int]]):
        """Delete a single point from the boundary conditions data.

        Parameters
        ----------
        index : int or list of int
            Index or list of indices of points to be deleted.
        """

        if self._data.empty:
            return

        if not isinstance(index, list):
            index = [index]
        # Check if indices are within range
        if any(x > (len(self._data.index) - 1) for x in index):
            raise ValueError("One of the indices exceeds length of index range!")
        self._data = self._data.drop(index).reset_index(drop=True)

        if self._data.empty:
            self.model.config.set("bndfile", None)
            self.model.config.set("bzsfile", None)
            self.model.config.set("bcafile", None)
            self.model.config.set("netbndbzsbzifile", None)

    def clear(self):
        """Clean GeoDataFrame with boundary points."""
        self._data = gpd.GeoDataFrame()

    def read_boundary_points(self, filename: str | Path = None):
        """Read SFINCS boundary condition points (*.bnd) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bndfile", value=filename
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
                "astro": pd.DataFrame(),
                "geometry": point,
            }
            gdf_list.append(d)

        gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

        self.set(gdf, merge=False)

    def read_boundary_conditions_timeseries(self, filename: str | Path = None):
        """Read SFINCS boundary condition timeseries (*.bzs) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bzsfile", value=filename
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

        # Read bzs file (this creates one DataFrame with all timeseries)
        df = utils.read_timeseries(abs_file_path, tref=self.model.config.get("tref"))

        # Now we need to split the timeseries into the different points
        for idx, row in self._data.iterrows():
            # Get the timeseries for this point
            ts = pd.DataFrame(df.iloc[:, idx])
            # Set the column name to wl
            ts.columns = ["wl"]
            # # Set the index to time
            # ts.index.name = "time"
            # Add to the point
            self._data.at[idx, "timeseries"] = ts

    def read_boundary_conditions_astro(self, filename: str | Path = None):
        """Read SFINCS boundary condition astro (*.bca) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if bcafile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bcafile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if bca file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Boundary condition astro file not found: {abs_file_path}"
            )

        # Read bca file, which is actually some sort of toml file
        d = IniStruct(filename=abs_file_path)
        # Loop through boundary points
        for ip, point in self._data.iterrows():
            # Set data in row of gdf
            self._data.at[ip, "astro"] = d.section[ip]._data

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
        for ip, point in self._data.iterrows():
            # Get the timeseries for this point
            ts = ds["timeseries"].sel(point=ip).to_dataframe()
            # Add to the point
            self._data.at[ip, "timeseries"] = ts

            # Get the astro for this point
            astro = ds["astro"].sel(point=ip).to_dataframe()
            # Add to the point
            self._data.at[ip, "astro"] = astro

    def write_boundary_points(self, filename: str | Path = None):
        """Write SFINCS boundary condition points (*.bnd) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bndfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bndfile", value=filename, default="sfincs.bnd"
        )

        # Write bnd file
        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"
        utils.write_xy(abs_file_path, self._data, fmt=fmt)

    def write_boundary_conditions_timeseries(self, filename: str | Path = None):
        """Write SFINCS boundary condition timeseries (*.bzs) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bzsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bzsfile", value=filename, default="sfincs.bzs"
        )

        # Get all timeseries and stick in one DataFrame
        df = pd.DataFrame()
        for ip, point in self._data.iterrows():
            df = pd.concat([df, point["timeseries"]["wl"]], axis=1)

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

    def write_boundary_conditions_astro(self, filename: str | Path = None):
        """Write SFINCS boundary condition astro (*.bca) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bcafile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bcafile", value=filename, default="sfincs.bca"
        )

        # Write bca file
        # Create IniStruct
        # Get rid of this IniStruct business !
        # Isn't bca a toml file? No, it is not. I just tested it.
        # There is probably something better in hydrolib-core
        d = IniStruct()
        # Loop through boundary points
        for ip, point in self._data.iterrows():
            # Add data to IniStruct
            d.section[ip]._data = point["astro"]
        # Write to file
        d.write(abs_file_path)

    def add_point(
        self,
        x: float,
        y: float,
        wl: float = 0.0,
        merge: bool = True,
    ):
        """Add a single point to the boundary conditions data.

        Parameters
        ----------
        x : float
            x-coordinate of the point
        y : float
            y-coordinate of the point
        wl : float, optional
            Water level of the point, by default 0.0
        """

        point = shapely.geometry.Point(x, y)
        gdf = gpd.GeoDataFrame(
            [
                {
                    "timeseries": pd.DataFrame(),
                    "astro": pd.DataFrame(),
                    "geometry": point,
                }
            ],
            crs=self.model.crs,
        )

        # Check if there is data in the timeseries
        if gdf["timeseries"][0].empty:
            # Now add the water level
            if not self._data.empty:
                # Set water level at same times as first existing point by copying
                gdf.at[0, "timeseries"] = self._data.iloc[0]["timeseries"].copy()
                gdf.at[0, "timeseries"]["wl"] = wl
            else:
                # First point, so need to generate df with constant water level
                time = [self.model.config.get("tstart"), self.model.config.get("tstop")]
                wl = [wl] * 2
                # Create DataFrame with columns time and wl
                df = pd.DataFrame()
                df["time"] = time
                df["wl"] = wl
                df = df.set_index("time")
                gdf.at[0, "timeseries"] = df
        else:
            # Check if the timeseries is the same length as the first point
            if len(gdf["timeseries"][0]) != len(self._data.iloc[0]["timeseries"]):
                raise ValueError(
                    "Timeseries in gdf must be the same length as the first point in the boundary conditions data."
                )

        # Add to self._data
        self.set(gdf, merge=merge)

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
        """Applies time series boundary conditions for each point
        Create numpy datetime64 array for time series with python datetime.datetime objects

        Parameters
        ----------
        shape : str
            Shape of the time series. Options are "constant", "sine", "gaussian", "astronomical".
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

        if self._data.empty:
            return

        if shape == "astronomical":
            # Use existing method
            self.generate_bzs_from_bca(dt=timestep, offset=offset, write=False)
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
            wl = [offset] * nt
        elif shape == "sine":
            wl = offset + amplitude * np.sin(
                2 * np.pi * tsec / period + phase * np.pi / 180
            )
        elif shape == "gaussian":
            wl = offset + peak * np.exp(-(((tsec - tpeak) / (0.25 * duration)) ** 2))
        elif shape == "astronomical":
            # Not implemented
            return

        times = pd.date_range(
            start=t0, end=t1, freq=pd.tseries.offsets.DateOffset(seconds=dtsec)
        )

        if index is None:
            index = list(self._data.index)
        elif not isinstance(index, list):
            index = [index]

        for i in index:
            df = pd.DataFrame()
            df["time"] = times
            df["wl"] = wl
            df = df.set_index("time")
            self._data.at[i, "timeseries"] = df

    def generate_bzs_from_bca(
        self, dt: float = 600.0, offset: float = 0.0, write_file: bool = True
    ):
        """Generate bzs file from bca file"""

        if self._data.empty:
            return

        if not self.model.input.variables.bzsfile:
            self.model.input.variables.bzsfile = "sfincs.bzs"

        times = pd.date_range(
            start=self.model.input.variables.tstart,
            end=self.model.input.variables.tstop,
            freq=pd.tseries.offsets.DateOffset(seconds=dt),
        )

        # Make boundary conditions based on bca file
        for icol, point in self.gdf.iterrows():
            v = predict(point.astro, times) + offset
            ts = pd.Series(v, index=times)
            # Convert this pandas series to a DataFrame
            df = pd.DataFrame()
            df["time"] = ts.index
            df["wl"] = ts.values
            df = df.set_index("time")
            self.gdf.at[icol, "timeseries"] = df

        if write_file:
            self.write_boundary_conditions_timeseries()

    def get_boundary_points_from_mask(self, min_dist=None, bnd_dist=5000.0):
        # Should move this to mask?

        if min_dist is None:
            # Set minimum distance between to grid boundary points on polyline to 2 * dx
            min_dist = self.model.quadtree_grid._data.attrs["dx"] * 2

        mask = self.model.quadtree_grid._data["mask"]
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
                    "astro": pd.DataFrame(),
                    "geometry": point,
                }
                gdf_list.append(d)
                ip += 1

        gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

        self.set(gdf, merge=False)

    def to_xarray(self):
        """Convert boundary conditions data to xarray dataset.

        Returns
        -------
        xarray.Dataset
            xarray dataset with boundary conditions data.
        """

        # This has not yet been tested !

        ds = xr.Dataset()

        if self._data.empty:
            return ds

        # Dimensions are time and point
        ds["time"] = self._data.iloc[0]["timeseries"].index
        ds["point"] = self._data.index

        # Create numpy arrays for boundary locations
        x = np.empty(len(self._data))
        y = np.empty(len(self._data))
        for ip, point in self._data.iterrows():
            x[ip] = point["geometry"].x
            y[ip] = point["geometry"].y

        # Create numpy array for water level
        wl = np.empty((len(ds["time"]), len(ds["point"])))
        for ip, point in self._data.iterrows():
            wl[:, ip] = point["timeseries"]["wl"].values

        # Add to dataset
        ds["x"] = x
        ds["y"] = y
        ds["wl"] = (("time", "point"), wl)

        return ds
