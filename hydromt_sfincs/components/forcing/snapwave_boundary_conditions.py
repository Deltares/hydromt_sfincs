import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from cht_tide import predict
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely
import xarray as xr

from hydromt import hydromt_step
from hydromt.gis.vector import GeoDataset
from hydromt_sfincs import utils
from .deltares_ini import IniStruct
from .boundary_conditions import SfincsBoundaryBase

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SnapWaveBoundaryConditions(SfincsBoundaryBase):
    """Water level boundary component for SFINCS models.

    This component handles reading and writing of wave boundary conditions
    in SFINCS-SnapWave format, including both ASCII and netCDF files.

    This component builds on the SfincsBoundaryBase class of boundary_conditions.py.
    """

    # %% core HydroMT-SFINCS functions:
    # __init__
    # read
    # read_boundary_points
    # read_boundary_conditions_timeseries_perfile
    # read_boundary_conditions_netcdf
    # write
    # write_boundary_points
    # write_boundary_conditions_timeseries
    # write_boundary_conditions_netcdf
    # set > inherited from SfincsBoundaryBase
    # delete
    # clear

    _default_varname = ["hs", "tp", "wd", "ds"]  # used in set_locations among others

    def __init__(self, model: "SfincsModel"):
        super().__init__(model)

    def read(self, format: str = None):
        """Read SFINCS-SnapWave wave boundary conditions (snapwave*.bnd, *.bhs/btp/bwd/bds, files) or netcdf file.

        The format of the boundary conditions files can be specified,
        otherwise it is determined from the model configuration.

        Parameters
        ----------
        format : str, optional
            Format of the boundary conditions files, "asc" or "netcdf".
        """

        if format is None:
            if self.model.config.get(
                "netsnapwavefile"
            ):  # FIXME - discuss whether to change name or not
                format = "netcdf"
            else:
                format = "asc"

        if format == "asc":
            gdf = self.read_boundary_points()

            # Check if there are any points
            if not gdf.empty:
                filenames = [
                    "snapwave_bhsfile",
                    "snapwave_btpfile",
                    "snapwave_bwdfile",
                    "snapwave_bdsfile",
                ]
                vars = ["hs", "tp", "wd", "ds"]
                da_lst = []
                for i, varname in enumerate(filenames):
                    df = self.read_boundary_conditions_timeseries_perfile(
                        var=vars[i], varname=varname
                    )

                    da = xr.DataArray(df, dims=("time", "index"), name=vars[i])
                    da_lst.append(da)

                # assumed is that all timestamps and number of points are the same in all files
                # (as they should be for SFINCS-SnapWave kernel)
                ds = xr.merge(da_lst[:])

                gds = GeoDataset.from_gdf(gdf=gdf, data_vars=ds)

                self.set(geodataset=gds, merge=False, drop_duplicates=False)

        elif format == "netcdf":
            # Read netcdf file
            ds = self.read_boundary_conditions_netcdf()
            self.set(geodataset=ds, merge=False, drop_duplicates=False)

    def read_boundary_points(self, filename: str | Path = None):
        """Read SnapWave boundary condition points snapwave_bndfile (*.bnd) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "snapwave_bndfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return gpd.GeoDataFrame()

        # Check if bnd file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"SnapWave boundary points file not found: {abs_file_path}"
            )

        # Read bnd file
        # TODO check if we want read_xyn? Before we used read_xy, so without name column
        gdf = utils.read_xyn(abs_file_path, crs=self.model.crs)
        gdf = gdf.set_crs(self.model.crs)

        return gdf

    def read_boundary_conditions_timeseries_perfile(
        self, var: str, varname: str, filename: str | Path = None
    ):
        """Read SFINCS boundary condition timeseries (*.bhs, *.btp, *.bwd or *.bds) files"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(varname, value=filename)

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if timeseries file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Boundary condition timeseries file not found: {abs_file_path}"
            )

        # Read timeseries file (this creates one DataFrame with all timeseries)
        df = utils.read_timeseries(abs_file_path, tref=self.model.config.get("tref"))
        df.index.name = "time"
        df.columns.name = "index"
        # FIXME - or df.columns.name = str(var) # hs, tp, wd or ds ?

        return df

    def read_boundary_conditions_netcdf(self, filename: str | Path = None):
        """Read SFINCS-SnapWave boundary conditions netcdf file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if netsnapwavefile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "netsnapwavefile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if netsnapwavefile exists
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Boundary condition netcdf file not found: {abs_file_path}"
            )

        # Read netcdf file
        ds = GeoDataset.from_netcdf(abs_file_path, crs=self.model.crs, chunks="auto")

        # FIXME - should we check if the dataset has the right variables?
        for var in ["hs", "tp", "wd", "ds"]:
            if var not in ds:
                raise ValueError(
                    f"Variable {var} not found in SnapWave boundary conditions netcdf file!"
                )

        # Rename variables to match internal naming
        ds = ds.rename({"stations": "index"}) if "stations" in ds.dims else ds
        return ds

        # # Loop through boundary points
        # # FIXME - we first need to get the points!
        # for ip, point in self.data.iterrows():
        #     # Get the timeseries for this point
        #     ts = ds["timeseries"].sel(point=ip).to_dataframe()
        #     # Add to the point
        #     self.data.at[ip, "timeseries"] = ts

        # ds.close()

    def write(self, format: str = None):
        """Write SnapWave boundary conditions (*.bnd, *.bhs, *.btp, *.bwd, *.bds files) or netcdf file.

        The format of the boundary conditions files can be specified,
        otherwise it is determined from the model configuration.

        Parameters
        ----------
        format : str, optional
            Format of the boundary conditions files, "asc" (default), or "netcdf".
        """
        if len(self.data.data_vars) == 0:
            # There are no boundary points in Dataset
            return

        if format is None:
            if self.model.config.get("netsnapwavefile"):
                format = "netcdf"
            else:
                format = "asc"

        if format == "asc":
            self.write_boundary_points()
            # Write timeseries per file
            filenames = [
                "snapwave_bhsfile",
                "snapwave_btpfile",
                "snapwave_bwdfile",
                "snapwave_bdsfile",
            ]
            vars = [
                "hs",
                "tp",
                "wd",
                "ds",
            ]  # FIXME - should be 'hs' or 'bhs'? in kernel is 'hs', but it's 'bhsfile'...
            for i, varname in enumerate(filenames):
                self.write_boundary_conditions_timeseries(var=vars[i], varname=varname)
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

        gdf = self.data.vector.to_gdf()

        utils.write_xy(abs_file_path, gdf, fmt=fmt)

    def write_boundary_conditions_timeseries(
        self, var: str, varname: str, filename: str | Path = None
    ):
        """Write SnapWave boundary condition timeseries (*.bhs, *.btp, *.bwd, *.bds) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if timeseries file is not None
        abs_file_path = self.model.config.get_set_file_variable(
            varname, value=filename, default="snapwave.b" + var
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # parse data to dataframe
        da = self.data[var].transpose("time", ...)
        df = da.to_pandas()

        # Write to file
        utils.write_timeseries(
            abs_file_path, df, self.model.config.get("tref"), fmt="%7.2f"
        )

    def write_boundary_conditions_netcdf(self, filename: str | Path = None):
        """Write SFINCS-SnapWave boundary condition netcdf (.nc) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if netsnapwavefile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "netsnapwavefile", value=filename, default="snapwave.nc"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        ds = self.data.load()

        # FIXME - check if right vars are present - usefull?
        for var in ["hs", "tp", "wd", "ds"]:
            if var not in ds:
                raise ValueError(f"Variable {var} not found in SnapWave self.data!")

        # set time encoding
        tref = self.model.config.get("tref")
        tref_str = tref.strftime("%Y-%m-%d %H:%M:%S")

        encoding = dict(time={"units": f"minutes since {tref_str}", "dtype": "float64"})

        # rename variables to match sfincs naming
        ds = ds.rename({"index": "stations"}) if "index" in ds.dims else ds

        # Write netcdf file safely (might get locked)
        final_path = utils.write_netcdf_safely(ds, abs_file_path, encoding=encoding)
        if final_path != abs_file_path:
            self.model.config.set("netsnapwavefile", final_path.name)

    def delete(self, index: Union[int, List[int]]):
        "Delete boundary points and clear config when no points remain."
        super().delete(index)
        if self.nr_points == 0:
            self.model.config.set("snapwave_bndfile", None)
            self.model.config.set("snapwave_bhsfile", None)
            self.model.config.set("snapwave_btpfile", None)
            self.model.config.set("snapwave_bwdfile", None)
            self.model.config.set("snapwave_bdsfile", None)
            self.model.config.set("netsnapwavefile", None)

    def clear(self):
        "Clear boundary points and unset associated config keys."
        super().clear()
        self.model.config.set("snapwave_bndfile", None)
        self.model.config.set("snapwave_bhsfile", None)
        self.model.config.set("snapwave_btpfile", None)
        self.model.config.set("snapwave_bwdfile", None)
        self.model.config.set("snapwave_bdsfile", None)
        self.model.config.set("netsnapwavefile", None)

    @hydromt_step
    def create(
        self,
        geodataset: Union[str, Path, xr.Dataset] = None,
        timeseries: List[Union[str, Path, pd.DataFrame]] = None,
        locations: Union[str, Path, gpd.GeoDataFrame] = None,
        buffer: float = 5e3,
        merge: bool = True,
        drop_duplicates: bool = True,
    ):
        """Create snapwave forcing.

        Snapwave boundary conditions are read from a `geodataset` (geospatial point timeseries)
        or a tabular `timeseries` dataframe. At least one of these must be provided.

        The tabular timeseries data is combined with `locations` if provided,
        or with existing 'snapwave_bnd' locations if previously set.

        Adds model forcing layers:

        * **hs** forcing: significant wave height time series [m]
        * **tp** forcing: peak wave period time series [s]
        * **wd** forcing: wave direction time series [° wrt North, in clockwise direction]
        * **ds** forcing: wave directional spreading time series [°]

        Parameters
        ----------
        geodataset: str, Path, xr.Dataset, optional
            Path, data source name, or xarray data object for geospatial point timeseries.
        timeseries: List of str, Path, pd.DataFrame, optional
            Path, data source name, or pandas data object for tabular timeseries for all 4 variables.
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for snapwave_bnd point locations.
            It should contain a 'index' column matching the column names in `timeseries`.
        buffer: float, optional
            Buffer [m] around model water level boundary cells to select wave data gauges,
            by default 5 km.
        merge : bool, optional
            If True, merge with existing forcing data, by default True.
        drop_duplicates : bool, optional
            If True, drop duplicate points in gdf based on 'name' column or geometry.

        See Also
        --------
        set
        """
        gdf_locs, df_ts = None, None
        tstart, tstop = self.model.get_model_time()  # model time
        vars = ["hs", "tp", "wd", "ds"]
        # buffer around msk==2 values
        if not self.model.grid_type == "quadtree":
            raise ValueError("SnapWave is not supported for regular grid models!")
        else:
            region = self.model.region

        # read wave data from geodataset or geodataframe
        if geodataset is not None:
            # read and clip data in time & space
            da = self.data_catalog.get_geodataset(
                geodataset,
                geom=region,
                buffer=buffer,
                variables=vars,
                time_range=(tstart, tstop),
            )
            self.set(geodataset=da, merge=False, drop_duplicates=False)
            self.model.config.set("netsnapwavefile", "snapwave.nc")

        # read wave data from separate timeseries and locations input
        else:
            # first, read locations data
            used_existing = False

            # read location data from locations input:
            if gdf_locs is None and locations is not None:
                gdf_locs = self.data_catalog.get_geodataframe(
                    locations,
                    geom=region,
                    buffer=buffer,
                ).to_crs(self.model.crs)
                if "index" in gdf_locs.columns:
                    gdf_locs = gdf_locs.set_index("index")
                # filter df_ts timeseries based on gdf_locs index
                # this allows to use a subset of the locations in the timeseries
                # FIXME - TL: do we still want this?
                if df_ts is not None and np.isin(gdf_locs.index, df_ts.columns).all():
                    df_ts = df_ts.reindex(gdf_locs.index, axis=1, fill_value=0)

            # no locations provided, using existing wave boundary points from data:
            # (e.g. added using get_boundary_points_from_mask)
            elif gdf_locs is None and "hs" in self.data:
                used_existing = True
                gdf_locs = self.data["hs"].vector.to_gdf()

            elif gdf_locs is None:
                raise ValueError("No wave boundary (bnd) points provided.")

            # It is still possible that all points are outside the region+buffer, this error should provide clear feedback
            if gdf_locs.is_empty.all():
                raise ValueError(
                    "All wave boundary points provided are outside the active model domain plus specified buffer. "
                    "Check the provided locations or increase the value of the buffer argument."
                )

            # set/ update forcing
            if used_existing is False:
                # set the new locations
                self.set_locations(
                    gdf=gdf_locs, merge=merge, drop_duplicates=drop_duplicates
                )
                # old: self.set(df=df_ts, gdf=gdf_locs, merge=merge, drop_duplicates=drop_duplicates)

            # secondly, read time-series data
            if timeseries is not None:
                # loop over list of timeseries inputs for each variable
                if len(timeseries) != 4:
                    raise ValueError(
                        "Timeseries input must be a list of 4 items, for hs, tp, wd, and ds."
                    )
                else:
                    for i, varname in enumerate(timeseries):
                        df_ts = self.data_catalog.get_dataframe(
                            timeseries[i],
                            time_range=(tstart, tstop),
                            source_kwargs={
                                "driver": {
                                    "name": "pandas",
                                    "options": {
                                        "index_col": 0,
                                        "parse_dates": True,
                                    },
                                }
                            },
                        )
                        # df_ts.columns.name = "index"
                        df_ts.columns = df_ts.columns.map(
                            int
                        )  # parse column names to integers

                        # set per variable
                        self.set_timeseries(
                            df=df_ts,
                            varname=vars[i],
                        )

            # update config
            self.model.config.set("snapwave_bndfile", "snapwave.bnd")
            self.model.config.set("snapwave_bhsfile", "snapwave.bhs")
            self.model.config.set("snapwave_btpfile", "snapwave.btp")
            self.model.config.set("snapwave_bdsfile", "snapwave.bds")
            self.model.config.set("snapwave_bwdfile", "snapwave.bwd")

    # @hydromt_step
    # def create_from_grid(
    # self,

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # def add_point_with_timeseries()
    #     """Add a single point with timeseries to the boundary conditions data.
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def add_point(  # FIXME - do we want to make a copy of add_point in boundary_conditions.py that support multiple vars?
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
        else:
            # Create a GeoDataFrame with a single point
            if x is None or y is None:
                raise ValueError("Either gdf or x, y, and name must be provided.")
            point = shapely.geometry.Point(x, y)
            gdf = gpd.GeoDataFrame(
                [
                    {
                        "geometry": point,
                    }
                ],
                crs=self.model.crs,
            )

        # make up a new df with timeseries data
        times = pd.date_range(*self.model.get_model_time(), periods=2)

        df_hs = np.array([hs, hs])
        df_hs = pd.DataFrame(data=df_hs, index=times, columns=[0])  # 2])
        df_hs.columns.name = "index"
        df_hs.index.name = "time"

        df_tp = np.array([tp, tp])
        df_tp = pd.DataFrame(data=df_tp, index=times, columns=[0])
        df_tp.columns.name = "index"
        df_tp.index.name = "time"

        df_wd = np.array([wd, wd])
        df_wd = pd.DataFrame(data=df_wd, index=times, columns=[0])
        df_wd.columns.name = "index"
        df_wd.index.name = "time"

        df_ds = np.array([ds, ds])
        df_ds = pd.DataFrame(data=df_ds, index=times, columns=[0])
        df_ds.columns.name = "index"
        df_ds.index.name = "time"

        # Add data
        self.create(
            locations=gdf,
            timeseries=[df_hs, df_tp, df_wd, df_ds],
            merge=False,  # FIXME - does not work currently
            drop_duplicates=False,  # FIXME - not sure it works
        )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # def set_timeseries(
    #     self,
    #     index: Union[int, List[int]] = None,
    #     shape: str = "constant",
    #     timestep: float = 600.0,
    #     hs: float = 1.0,
    #     tp: float = 10.0,
    #     wd: float = 270.0,
    #     ds: float = 20.0,
    #     tpeak: float = 86400.0,
    #     duration: float = 43200.0,
    # ):
    #     """Applies time series boundary conditions for each point
    #     Create numpy datetime64 array for time series with python datetime.datetime objects

    #     Parameters
    #     ----------
    #     shape : str
    #         Shape of the time series. Options are "constant" or "gaussian".
    #     timestep : float
    #         Time step [s]
    #     hs : float
    #         Wave height [m]
    #     tp : float
    #         Peak period [s]
    #     wd : float
    #         Wave direction [degrees]
    #     ds : float
    #         Directional spread [degrees]
    #     tpeak : float
    #         Time of the peak of the Gaussian wave [s]
    #     duration : float
    #         Duration of the Gaussian wave [s]
    #     """

    #     if self.data.empty:
    #         return

    #     t0 = np.datetime64(self.model.config.get("tstart"))
    #     t1 = np.datetime64(self.model.config.get("tstop"))
    #     if shape == "constant":
    #         dt = np.timedelta64(int((t1 - t0).astype(float) / 1e6), "s")
    #     else:
    #         dt = np.timedelta64(int(timestep), "s")
    #     time = np.arange(t0, t1 + dt, dt)
    #     dtsec = dt.astype(float)
    #     # Convert time to seconds since tref
    #     tsec = (
    #         (time - np.datetime64(self.model.config.get("tref")))
    #         .astype("timedelta64[s]")
    #         .astype(float)
    #     )
    #     nt = len(tsec)
    #     if shape == "constant":
    #         hs = [hs] * nt
    #         tp = [tp] * nt
    #         wd = [wd] * nt
    #         ds = [ds] * nt
    #     elif shape == "gaussian":
    #         hs = hs * np.exp(-(((tsec - tpeak) / (0.25 * duration)) ** 2))
    #         tp = [tp] * nt
    #         wd = [wd] * nt
    #         ds = [ds] * nt
    #     else:
    #         # Not implemented
    #         raise ValueError(
    #             f"Shape {shape} not implemented for SnapWave boundary conditions!"
    #         )

    #     times = pd.date_range(
    #         start=t0, end=t1, freq=pd.tseries.offsets.DateOffset(seconds=dtsec)
    #     )

    #     if index is None:
    #         index = list(self.data.index)
    #     elif not isinstance(index, list):
    #         index = [index]

    #     for i in index:
    #         df = pd.DataFrame()
    #         df["time"] = times
    #         df["hs"] = hs
    #         df["tp"] = tp
    #         df["wd"] = wd
    #         df["ds"] = ds
    #         df = df.set_index("time")
    #         self.data.at[i, "timeseries"] = df

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
