import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely
import xarray as xr

from hydromt import hydromt_step
from hydromt.gis.vector import GeoDataset
from hydromt_sfincs import utils
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

    _default_varname = ["hs", "tp", "wd", "ds"]  # used in set_locations among others
    _default_values = [1.0, 10.0, 270.0, 20.0]

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
                ds = self.read_boundary_conditions_timeseries()

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
        gdf = utils.read_xyn(abs_file_path, crs=self.model.crs)

        return gdf

    def read_boundary_conditions_timeseries(self):
        """Read SFINCS boundary condition timeseries (*.bhs, *.btp, *.bwd or *.bds) files.
        This function is used to read the timeseries files for each variable separately
        and return as combined geodataset with all variables."""

        filenames = [
            "snapwave_bhsfile",
            "snapwave_btpfile",
            "snapwave_bwdfile",
            "snapwave_bdsfile",
        ]
        vars = ["hs", "tp", "wd", "ds"]
        da_lst = []

        for i, varname in enumerate(filenames):
            df = self.read_boundary_conditions_timeseries_perfile(varname=varname)

            da = xr.DataArray(df, dims=("time", "index"), name=vars[i])
            da_lst.append(da)

        # assumed is that all timestamps and number of points are the same in all files
        # (as they should be for SFINCS-SnapWave kernel)
        ds = xr.merge(da_lst[:])

        return ds

    def read_boundary_conditions_timeseries_perfile(
        self, varname: str, filename: str | Path = None
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

        for var in ["hs", "tp", "wd", "ds"]:
            if var not in ds:
                raise ValueError(
                    f"Variable {var} not found in SnapWave boundary conditions netcdf file!"
                )

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
        if self.nr_points == 0:
            # There are no boundary points
            return

        if format is None:
            if self.model.config.get("netsnapwavefile"):
                format = "netcdf"
            else:
                format = "asc"

        if format == "asc":
            self.write_boundary_points()
            # Write timeseries per file
            for var in ["hs", "tp", "wd", "ds"]:
                varname = f"snapwave_b{var}file"
                self.write_boundary_conditions_timeseries(var=var, varname=varname)

            self.model.config.set("netsnapwavefile", None)
        else:
            self.write_boundary_conditions_netcdf()
            self.model.config.set("snapwave_bndfile", None)
            self.model.config.set("snapwave_bhsfile", None)
            self.model.config.set("snapwave_btpfile", None)
            self.model.config.set("snapwave_bwdfile", None)
            self.model.config.set("snapwave_bdsfile", None)

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

        utils.write_xyn(abs_file_path, gdf, fmt=fmt)

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
        buffer: float = 25e3,
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
            Buffer [m] around model wave boundary cells to select wave data gauges,
            by default 25 km.
        merge : bool, optional
            If True, merge with existing forcing data, by default True.
        drop_duplicates : bool, optional
            If True, drop duplicate points in gdf based on 'name' column or geometry.

        See Also
        --------
        set
        """

        # SnapWave is only supported for quadtree grid models
        if not self.model.grid_type == "quadtree":
            raise ValueError("SnapWave is not supported for regular grid models!")

        # Check if snapwave_mask is available in model grid, as this is needed to select points around snapwave boundary
        if "snapwave_mask" not in self.model.quadtree_grid.data:
            raise ValueError(
                "SnapWave mask not found in model grid! Make sure to create the snapwave_mask in the quadtree grid before running this step."
            )
        # if present, check whether is has values of 2, which indicate the snapwave boundary cells
        if not np.any(self.model.quadtree_grid.data["snapwave_mask"] == 2):
            raise ValueError(
                "No snapwave boundary cells found in snapwave_mask! Make sure to create the snapwave_mask in the quadtree grid before running this step."
            )

        gdf_locs, df_ts = None, None
        tstart, tstop = self.model.get_model_time()  # model time

        # Create a buffer around the snapwave boundary cells (msk==2)
        gdf_msk = utils.get_bounds_vector(
            da_msk=self.model.quadtree_grid.data["snapwave_mask"],
        )
        gdf_msk2 = gdf_msk[gdf_msk["value"] == 2]
        gdf_msk2["geometry"] = gdf_msk2.buffer(buffer)
        # gdf_msk2 is now used to clip geodataset to get wanted locations
        region = gdf_msk2

        # Read wave data from geodataset, timeseries and locations
        if geodataset is not None:
            # read and clip data in time & space
            ds = self.data_catalog.get_geodataset(
                geodataset,
                geom=region,
                buffer=buffer,
                variables=self._default_varname,
                time_range=(tstart, tstop),
            )
            self.set(geodataset=ds, merge=False, drop_duplicates=False)
            self.model.config.set("netsnapwavefile", "snapwave.nc")
            return

        # Read wave data from separate timeseries and locations input
        # First read locations
        if locations is not None:
            gdf_locs = self.data_catalog.get_geodataframe(
                locations, geom=region, buffer=buffer
            ).to_crs(self.model.crs)
            if "index" in gdf_locs.columns:
                gdf_locs = gdf_locs.set_index("index")
            used_existing = False
        # No locations provided, using existing wave boundary points from data:
        elif self.nr_points > 0:
            gdf_locs = self.gdf
            used_existing = True
        else:
            raise ValueError("No wave boundary (bnd) points provided.")

        # It is still possible that all points are outside the region+buffer, this error should provide clear feedback
        if gdf_locs.is_empty.all():
            raise ValueError(
                "All wave boundary points provided are outside the active model domain plus specified buffer. "
                "Check the provided locations or increase the value of the buffer argument."
            )

        # Set/ update forcing
        if not used_existing:
            self.set_locations(
                gdf=gdf_locs,
                value=self._default_values,
                merge=merge,
                drop_duplicates=drop_duplicates,
            )

        # Secondly, read time-series data
        if timeseries is not None:
            # loop over list of timeseries inputs for each variable
            if len(timeseries) != len(self._default_varname):
                raise ValueError(
                    f"Timeseries does not provide data for all variables {self._default_varname}!, "
                    f"make sure to provide a list of {len(self._default_varname)} timeseries inputs."
                )
            for ts, varname in zip(timeseries, self._default_varname):
                df_ts = self.data_catalog.get_dataframe(
                    ts,
                    time_range=(tstart, tstop),
                    source_kwargs={
                        "driver": {
                            "name": "pandas",
                            "options": {"index_col": 0, "parse_dates": True},
                        }
                    },
                )
                # ensure column labels match integer location indices
                df_ts.columns = df_ts.columns.map(int)
                # set timeseries to data per variable
                self.set_timeseries(df_ts, varname=varname)

        # Lastly, always update config
        self.model.config.set("snapwave_bndfile", "snapwave.bnd")
        for var in self._default_varname:
            self.model.config.set(f"snapwave_b{var}file", f"snapwave.b{var}")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    @hydromt_step
    def create_from_grid(
        self,
        data: Union[str, Path, xr.Dataset],
        buffer: float = 25e3,
    ):
        """Create snapwave forcing from 2D gridded dataset (e.g. ERA5 results).

        Snapwave boundary conditions are read from a `geodataset` (geospatial point timeseries).
        Data points are selected within a `buffer` around the model region.
        Expected coordinates in input geodataset are ('lon','lat')
        or ('longitude','latitude') or ('x','y')

        Adds model forcing layers:

        * **hs** forcing: significant wave height time series [m]
        * **tp** forcing: peak wave period time series [s]
        * **wd** forcing: wave direction time series [° wrt North, in clockwise direction]
        * **ds** forcing: wave directional spreading time series [°]

        Parameters
        ----------
        data: str, Path, xr.Dataset, optional
            Path, data source name, or xarray data object for raster dataset with variables
            hs, tp, wd and ds.
        buffer: float, optional
            Buffer [m] around model wave boundary cells to select wave data gauges,
            by default 25 km.

        See Also
        --------
        set
        """

        # SnapWave is only supported for quadtree grid models
        if not self.model.grid_type == "quadtree":
            raise ValueError("SnapWave is not supported for regular grid models!")

        # Check if snapwave_mask is available in model grid, as this is needed to select points around snapwave boundary
        if "snapwave_mask" not in self.model.quadtree_grid.data:
            raise ValueError(
                "SnapWave mask not found in model grid! Make sure to create the snapwave_mask in the quadtree grid before running this step."
            )
        # if present, check whether is has values of 2, which indicate the snapwave boundary cells
        if not np.any(self.model.quadtree_grid.data["snapwave_mask"] == 2):
            raise ValueError(
                "No snapwave boundary cells found in snapwave_mask! Make sure to create the snapwave_mask in the quadtree grid before running this step."
            )

        # get data for model domain and config time range
        ds = self.data_catalog.get_rasterdataset(
            data,
            bbox=self.model.bbox,
            buffer=5,
            time_range=self.model.get_model_time(),
            variables=self._default_varname,
        )

        # Create a buffer around the snapwave boundary cells (msk==2)
        gdf_msk = utils.get_bounds_vector(
            da_msk=self.model.quadtree_grid.data["snapwave_mask"],
        )
        gdf_msk2 = gdf_msk[gdf_msk["value"] == 2]
        gdf_msk2["geometry"] = gdf_msk2.buffer(buffer)

        # Select cells within buffer around snapwave boundary cells (msk==2)
        msk = ds.raster.geometry_mask(gdf_msk2, all_touched=True)

        # Check if any points are selected, otherwise raise error
        if msk.sum() == 0:
            raise ValueError(
                "No data points found within buffer around snapwave boundary cells! "
                "Try increasing the buffer distance or check the input data and model grid."
            )

        # Convert selected cells to point dataset
        spatial_dims = ds.raster.dims
        ds = ds.stack(index=spatial_dims)
        msk = msk.stack(index=spatial_dims)
        ds = ds.sel(index=msk)
        gds = GeoDataset.from_netcdf(ds)

        # Set data to model
        self.set(geodataset=gds, merge=False, drop_duplicates=False)
        self.model.config.set("netsnapwavefile", "snapwave.nc")

        # Log some information about the created snapwave boundary conditions
        logger.info(f"Added {ds.index.size} snapwave boundary points to the model.")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def add_point(
        self,
        x: float,
        y: float,
        name: str = None,
        hs: float = 1.0,
        tp: float = 10.0,
        wd: float = 270.0,
        ds: float = 20.0,
        drop_duplicates: bool = True,
    ):
        """Add a single point to the boundary conditions data.
        x, y must be provided.

        Parameters
        ----------

        x : float
            x-coordinate of the point
        y : float
            y-coordinate of the point
        hs : float
            Wave height in meters of the point
        tp : float
            Peak period in seconds of the point
        wd : float
            Wave direction in nautical degrees of the point
        ds : float
            Directional spread in degrees of the point
        """

        super().add_point(
            x=x,
            y=y,
            name=name,
            value=[hs, tp, wd, ds],
            drop_duplicates=drop_duplicates,
        )

    @hydromt_step
    def create_timeseries(
        self,
        index: Union[int, List[int]] = None,
        shape: str = "constant",
        timestep: float = 600.0,
        offset: float = 0.0,
        hs: float = 1.0,
        tp: float = 10.0,
        wd: float = 270.0,
        ds: float = 20.0,
        tpeak: float = 43200.0,
        duration: float = 86400.0,
    ):
        """Applies time series boundary conditions for each point
        Create numpy datetime64 array for time series with python datetime.datetime objects

        Parameters
        ----------
        shape : str
            Shape of the time series. Options are "constant" or "gaussian".
            Gaussian shape is only applied to hs time-series,
            with hs[0]=offset and hs[peak]=hs.
        timestep : float
            Time step [s]
        offset : float
            Vertical offset of the gaussian time series [m]
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

        if self.nr_points == 0:
            raise ValueError(
                "Cannot create timeseries without existing waterlevel boundary points"
            )

        t0 = np.datetime64(self.model.config.get("tstart"))
        t1 = np.datetime64(self.model.config.get("tstop"))
        if shape == "constant":
            dt = np.timedelta64(int((t1 - t0).astype(float) / 1e6), "s")
        else:
            dt = np.timedelta64(int(timestep), "s")
        time = np.arange(t0, t1 + dt, dt)
        dtsec = dt.astype(float)
        # Convert time to seconds since tstart
        tsec = (time - t0).astype("timedelta64[s]").astype(float)
        # # Convert time to seconds since tref
        # tsec = (
        #     (time - np.datetime64(self.model.config.get("tref")))
        #     .astype("timedelta64[s]")
        #     .astype(float)
        # )
        nt = len(tsec)
        if shape == "constant":
            hs = [hs] * nt
            tp = [tp] * nt
            wd = [wd] * nt
            ds = [ds] * nt
        elif shape == "gaussian":
            hs = offset + (hs - offset) * np.exp(
                -(((tsec - tpeak) / (0.25 * duration)) ** 2)
            )
            tp = [tp] * nt
            wd = [wd] * nt
            ds = [ds] * nt
        else:
            raise NotImplementedError(
                f"Shape '{shape}' is not implemented. Use 'constant' or 'gaussian'."
            )
        times = pd.date_range(
            start=t0, end=t1, freq=pd.tseries.offsets.DateOffset(seconds=dtsec)
        )

        if index is None:
            index = list(self.data.index.values)
        elif not isinstance(index, list):
            index = [index]

        # Create DataFrame: rows = time, columns = locations (index), values = hs (same for all)
        df_hs = pd.DataFrame(
            data=np.tile(hs, (len(index), 1)).T, index=times, columns=index
        )
        df_tp = pd.DataFrame(
            data=np.tile(tp, (len(index), 1)).T, index=times, columns=index
        )
        df_wd = pd.DataFrame(
            data=np.tile(wd, (len(index), 1)).T, index=times, columns=index
        )
        df_ds = pd.DataFrame(
            data=np.tile(ds, (len(index), 1)).T, index=times, columns=index
        )

        # Call set_timeseries to update your object's data
        self.set_timeseries(df_hs, varname="hs")
        self.set_timeseries(df_tp, varname="tp")
        self.set_timeseries(df_wd, varname="wd")
        self.set_timeseries(df_ds, varname="ds")

    def create_boundary_points_from_mask(self, bnd_dist=5000.0, method="absolute"):
        """Get boundary points from snapwave boundary cells of mask in quadtree grid.

        Parameters
        ----------
        bnd_dist : float, optional
            Distance [m] between boundary points, by default 5000.0.
        method : str, optional
            Method to use for creating boundary points. When "absolute", points are created at a fixed distance
            of bnd_dist along the boundary. When "relative", points are created such that there is always a point
            at the start and end of the boundary and the distance between points is approximately bnd_dist.
        """

        if not self.model.grid_type == "quadtree":
            raise ValueError("SnapWave is not supported for regular grid models!")

        # Get snapwave mask from model grid
        mask = self.model.quadtree_grid.data["snapwave_mask"]

        # get boundary vector based on mask
        gdf_msk = utils.get_bounds_vector(mask)
        gdf_msk2 = gdf_msk[gdf_msk["value"] == 2]

        # convert to meters if CRS is geographic
        if self.model.crs.is_geographic:
            bnd_dist = bnd_dist / 111111.0

        # create boundary points from mask boundary vector
        gdf_points = utils.create_boundary_points(
            gdf_msk2, bnd_dist=bnd_dist, method=method, crs=self.model.crs
        )

        # set locations
        self.set_locations(gdf_points, merge=False)
