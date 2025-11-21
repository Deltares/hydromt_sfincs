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


class SfincsWaterLevel(SfincsBoundaryBase):
    """Water level boundary component for SFINCS models.

    This component handles reading and writing of water level boundary conditions
    in SFINCS format, including both ASCII and netCDF files. Furthermore, it
    allows to generate time series from astronomical constituents (if available).
    """

    _default_varname = "bzs"

    def __init__(self, model: "SfincsModel"):
        super().__init__(model)

    def read(self, format: str = None):
        """Read SFINCS boundary conditions (.bnd, .bzs, .bca files) or netcdf file.

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
            gdf = self.read_boundary_points()
            # Check if there are any points
            if not gdf.empty:
                df = self.read_boundary_conditions_timeseries()
                self.set(df=df, gdf=gdf, merge=False, drop_duplicates=False)
                # Read astro if bcafile is defined
                if self.model.config.get("bcafile"):
                    self.read_boundary_conditions_astro()
        elif format == "netcdf":
            # Read netcdf file
            ds = self.read_boundary_conditions_netcdf()
            self.set(geodataset=ds, merge=False, drop_duplicates=False)

    def read_boundary_points(self, filename: str | Path = None):
        """Read SFINCS boundary condition points (.bnd) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bndfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return gpd.GeoDataFrame()

        # Check if bnd file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(f"Discharge points file not found: {abs_file_path}")

        # Read bnd file
        # TODO check if we want read_xyn? Before we used read_xy, so without name column
        gdf = utils.read_xyn(abs_file_path, crs=self.model.crs)
        return gdf

    def read_boundary_conditions_timeseries(self, filename: str | Path = None):
        """Read SFINCS boundary condition timeseries (.bzs) file"""

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
        df.index.name = "time"
        df.columns.name = "index"
        return df

    def read_boundary_conditions_astro(self, filename: str | Path = None):
        """Read SFINCS boundary condition astro (.bca) file"""

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

        if self.nr_points == 0:
            return

        # Read bca file, which is actually some sort of toml file
        d = IniStruct(filename=abs_file_path)

        # Store all constituents in a list
        section_data = [d.section[i].data for i in range(self.nr_points)]

        # Add constituents
        self._data = add_constituents(self.data, section_data)

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
        ds = GeoDataset.from_netcdf(abs_file_path, crs=self.model.crs, chunks="auto")
        return ds

    def write(self, format: str = None):
        """Write SFINCS boundary conditions (.bnd, .bzs, .bca files) or netcdf file.

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

        if self.model.write_gis:
            utils.write_vector(
                self.gdf,
                name="bnd",
                root=join(self.model.root.path, "gis"),
                logger=logger,
            )

    def write_boundary_points(self, filename: str | Path = None):
        """Write SFINCS boundary condition points (.bnd) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bndfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bndfile", value=filename, default="sfincs.bnd"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write bnd file
        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # TODO check whether write_xyn or write_xy
        utils.write_xyn(abs_file_path, self.gdf, fmt=fmt)

    def write_boundary_conditions_timeseries(self, filename: str | Path = None):
        """Write SFINCS boundary condition timeseries (.bzs) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bzsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bzsfile", value=filename, default="sfincs.bzs"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # parse data to dataframe
        da = self.data["bzs"].transpose("time", ...)
        df = da.to_pandas()

        # Write to file
        utils.write_timeseries(abs_file_path, df, self.model.config.get("tref"))

    def write_boundary_conditions_astro(self, filename: str | Path = None):
        """Write SFINCS boundary condition astro (.bca) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if bcafile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bcafile", value=filename, default="sfincs.bca"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        amp = self.data["amplitude"].to_pandas()
        pha = self.data["phase"].to_pandas()

        with open(abs_file_path, "w") as fid:
            for ip in self.data.index.values:
                # Optional: if you have names from your dataset
                if "name" in self.data.coords:
                    name = f"sfincs_{int(self.data.name.sel(index=ip).item()):04d}"
                else:
                    name = f"sfincs_{ip+1:04d}"

                fid.write(f"[forcing]\n")
                fid.write(f"Name                            = {name}\n")
                fid.write(f"Function                        = astronomic\n")
                fid.write(f"Quantity                        = astronomic component\n")
                fid.write(f"Unit                            = -\n")
                fid.write(
                    f"Quantity                        = waterlevelbnd amplitude\n"
                )
                fid.write(f"Unit                            = m\n")
                fid.write(f"Quantity                        = waterlevelbnd phase\n")
                fid.write(f"Unit                            = deg\n")

                # Write each constituent (skip NaN)
                for constituent in self.data.constituent.values:
                    a = amp.loc[ip, constituent]
                    p = pha.loc[ip, constituent]
                    if not (pd.isna(a) or pd.isna(p)):
                        fid.write(f"{constituent:6s}{a:10.5f}{p:10.2f}\n")

                fid.write("\n")

    def write_boundary_conditions_netcdf(self, filename: str | Path = None):
        """Write SFINCS boundary condition netcdf (.nc) file"""

        # Check that write mode is on
        self.root._assert_write_mode()

        # Get absolute file name and set it in config if netbndbzsbzifile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "netbndbzsbzifile", value=filename, default="sfincs_netbndbzsbzifile.nc"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        ds = self.data.load()

        # Write netcdf file safely (might get locked)
        final_path = utils.write_netcdf_safely(ds, abs_file_path)
        if final_path != abs_file_path:
            self.model.config.set("netbndbzsbzifile", final_path.name)

    def delete(self, index: Union[int, List[int]]):
        "Delete boundary points and clear config when no points remain."
        super().delete(index)
        if self.nr_points == 0:
            self.model.config.set("bndfile", None)
            self.model.config.set("bzsfile", None)
            self.model.config.set("bcafile", None)
            self.model.config.set("netbndbzsbzifile", None)

    def clear(self):
        "Clear boundary points and unset associated config keys."
        super().clear()
        self.model.config.set("bndfile", None)
        self.model.config.set("bzsfile", None)
        self.model.config.set("bcafile", None)
        self.model.config.set("netbndbzsbzifile", None)

    @hydromt_step
    def create(
        self,
        geodataset: Union[str, Path, xr.Dataset] = None,
        timeseries: Union[str, Path, pd.DataFrame] = None,
        locations: Union[str, Path, gpd.GeoDataFrame] = None,
        offset: Union[str, Path, xr.Dataset] = None,
        buffer: float = 5e3,
        merge: bool = True,
        drop_duplicates: bool = True,
    ):
        """Setup waterlevel forcing.

        Waterlevel boundary conditions are read from a `geodataset` (geospatial point timeseries)
        or a tabular `timeseries` dataframe. At least one of these must be provided.

        The tabular timeseries data is combined with `locations` if provided,
        or with existing 'bnd' locations if previously set.

        Adds model forcing layers:

        * **bzs** forcing: waterlevel time series [m+ref]

        Parameters
        ----------
        geodataset: str, Path, xr.Dataset, optional
            Path, data source name, or xarray data object for geospatial point timeseries.
        timeseries: str, Path, pd.DataFrame, optional
            Path, data source name, or pandas data object for tabular timeseries.
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for bnd point locations.
            It should contain a 'index' column matching the column names in `timeseries`.
        offset: str, Path, xr.Dataset, float, optional
            Path, data source name, constant value or xarray raster data for gridded offset
            between vertical reference of elevation and waterlevel data,
            The offset is added to the waterlevel data.
        buffer: float, optional
            Buffer [m] around model water level boundary cells to select waterlevel gauges,
            by default 5 km.
        merge : bool, optional
            If True, merge with existing forcing data, by default True.
        drop_duplicates : bool, optional
            If True, drop duplicate points in gdf based on 'name' column or geometry.

        See Also
        --------
        set_forcing_1d
        """
        gdf_locs, df_ts = None, None
        tstart, tstop = self.model.get_model_time()  # model time
        # buffer around msk==2 values
        if not self.model.grid_type == "quadtree":
            if np.any(self.model.grid.mask == 2):
                # get region around waterlevel boundary cells
                region = self.model.grid.mask.where(
                    self.model.grid.mask == 2, 0
                ).raster.vectorize()
            else:
                raise ValueError(
                    "No waterlevel boundary cells (mask==2) in model grid."
                )
        else:
            region = self.model.region
        # read waterlevel data from geodataset or geodataframe
        if geodataset is not None:
            # read and clip data in time & space
            da = self.data_catalog.get_geodataset(
                geodataset,
                geom=region,
                buffer=buffer,
                variables=["waterlevel"],
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

        used_existing = False
        # read location data (if not already read from geodataset)
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
            if df_ts is not None and np.isin(gdf_locs.index, df_ts.columns).all():
                df_ts = df_ts.reindex(gdf_locs.index, axis=1, fill_value=0)
        elif gdf_locs is None and "bzs" in self.data:
            # no locations provided, using existing waterlevel boundary points from data
            used_existing = True
            gdf_locs = self.data[
                "bzs"
            ].vector.to_gdf()  # NOTE this is now done in set_timeseries ...
        elif gdf_locs is None:
            raise ValueError("No waterlevel boundary (bnd) points provided.")
        # It is still possible that all points are outside the region+buffer, this error should provide clear feedback
        if gdf_locs.is_empty.all():
            raise ValueError(
                "All waterlevel boundary points provided are outside the active model domain plus specified buffer. "
                "Check the provided locations or increase the value of the buffer argument."
            )

        # optionally read offset data and correct df_ts
        if offset is not None and gdf_locs is not None:
            if isinstance(offset, (float, int)):
                df_ts += offset
            else:
                da_offset = self.data_catalog.get_rasterdataset(
                    offset,
                    bbox=self.model.bbox,
                    buffer=5,
                )
                offset_pnts = da_offset.raster.sample(gdf_locs)
                df_offset = offset_pnts.to_pandas().reindex(df_ts.columns).fillna(0)
                df_ts = df_ts + df_offset
                offset = offset_pnts.mean().values
            logger.debug(f"waterlevel forcing: applied offset (avg: {offset:+.2f})")

        # set/ update forcing
        if used_existing:
            gdf_locs = None  # only update timeseries for existing points
        self.set(df=df_ts, gdf=gdf_locs, merge=merge, drop_duplicates=drop_duplicates)
        # update config
        if geodataset is not None:
            # when reading from geodataset, keep the format to netcdf
            self.model.config.set("netbndbzsbzifile", "sfincs_netbndbzsbzifile.nc")
        else:
            self.model.config.set("bndfile", "sfincs.bnd")
            self.model.config.set("bzsfile", "sfincs.bzs")

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
            # Use existing method
            self.generate_bzs_from_bca(dt=timestep, offset=offset, write_file=False)
            return
        else:
            raise NotImplementedError(
                f"Shape '{shape}' is not implemented. Use 'constant', 'sine', 'gaussian' or 'astronomical'."
            )

        times = pd.date_range(
            start=t0, end=t1, freq=pd.tseries.offsets.DateOffset(seconds=dtsec)
        )

        if index is None:
            index = list(self.data.index.values)
        elif not isinstance(index, list):
            index = [index]

        # Create DataFrame: rows = time, columns = locations (index), values = wl (same for all)
        df = pd.DataFrame(
            data=np.tile(wl, (len(index), 1)).T, index=times, columns=index
        )

        # Call set_timeseries to update your object's data
        self.set_timeseries(df)

    @hydromt_step
    def create_timeseries_from_astro(
        self,
        dt: float = 600.0,
        offset: float = 0.0,
    ):
        """Generates boundary time series file from astronomical components

        Parameters
        ----------
        dt : float, optional, default 600.0
            Time step [s]
        offset : float, optional, default 0.0
            Offset of the time series [m]
        """

        if self.nr_points == 0:
            return

        times = pd.date_range(
            start=self.model.config.get("tstart"),
            end=self.model.config.get("tstop"),
            freq=pd.tseries.offsets.DateOffset(seconds=dt),
        )

        df_ts = pd.DataFrame(index=times)
        for ip in self.data.index.values:
            df = pd.DataFrame(
                {
                    "constituent": self.data.constituent.values,
                    "amplitude": self.data["amplitude"].loc[ip].values,
                    "phase": self.data["phase"].loc[ip].values,
                }
            )
            df = df.dropna(subset=["amplitude", "phase"])  # remove NaN paddings
            df = df.set_index("constituent")

            v = predict(df, times) + offset
            ts = pd.Series(v, index=times)
            df_ts[ip] = ts

        # Call set_timeseries to update your object's data
        self.set_timeseries(df_ts)

    def generate_bzs_from_bca(
        self, dt: float = 600.0, offset: float = 0.0, write_file: bool = True
    ):
        """Function called in CoSMoS to generate bzs file from bca file.
        Should probably update CoSMoS and only use generate_timeseries_from_astro"""

        self.create_timeseries_from_astro(dt=dt, offset=offset)

        if write_file:
            self.write_boundary_conditions_timeseries()

    def create_boundary_points_from_mask(self, min_dist=None, bnd_dist=5000.0):
        """Get boundary points from mask in quadtree grid.
        Should make utils function as sfincs_snapwave_boundary conditions uses nearly same code
        Also, regular grid has similar code. Maybe that is more efficient or better.
        """
        if self.model.grid_type == "regular":
            # get waterlevel boundary vector based on mask
            gdf_msk = utils.get_bounds_vector(self.model.grid.mask)
            gdf_msk2 = gdf_msk[gdf_msk["value"] == 2]

            # convert to meters if crs is geographic
            if self.model.crs.is_geographic:
                bnd_dist = bnd_dist / 111111.0

            # create points along boundary
            points = []
            for _, row in gdf_msk2.iterrows():
                distances = np.arange(0, row.geometry.length, bnd_dist)
                for d in distances:
                    point = row.geometry.interpolate(d)
                    points.append((point.x, point.y))

            # create geodataframe with points
            gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(*zip(*points)), crs=self.model.crs
            )
        elif self.model.grid_type == "quadtree":
            if min_dist is None:
                # Set minimum distance between to grid boundary points on polyline to 2 * dx
                min_dist = self.model.quadtree_grid.data.attrs["dx"] * 2

            mask = self.model.quadtree_grid.data["mask"]
            ibnd = np.where(mask == 2)
            xz, yz = self.model.quadtree_grid.face_coordinates
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

        self.set_locations(gdf, merge=False)


def add_constituents(ds, section_data):
    """
    Attach tidal constituent data to an existing Dataset that already has
    a 'bzs(time, index)' variable.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Dataset containing 'bzs(time, index)'.
    section_data : list of pandas.DataFrame
        One DataFrame per index point.
        Each DataFrame must have:
            - index: constituent names
            - first column: amplitude
            - second column: phase

    Returns
    -------
    xr.Dataset
        Same dataset, with two new variables:
        - amplitude(index, constituent)
        - phase(index, constituent)
    """

    # Ensure we have a Dataset
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name="bzs")

    # Collect all constituent names across all points
    all_constituents = sorted(set().union(*[df.index for df in section_data]))

    # Allocate arrays (index x constituent)
    n_points = len(section_data)
    n_const = len(all_constituents)
    amp = np.full((n_points, n_const), np.nan)
    pha = np.full((n_points, n_const), np.nan)

    # Fill arrays
    for i, df in enumerate(section_data):
        for cname in df.index:
            j = all_constituents.index(cname)
            amp[i, j] = df.iloc[df.index.get_loc(cname), 0]  # amplitude
            pha[i, j] = df.iloc[df.index.get_loc(cname), 1]  # phase

    # Attach to dataset
    ds["amplitude"] = (("index", "constituent"), amp)
    ds["phase"] = (("index", "constituent"), pha)
    ds = ds.assign_coords(constituent=all_constituents)

    return ds
