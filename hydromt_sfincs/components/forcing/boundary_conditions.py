import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydromt import hydromt_step
from hydromt.gis.vector import GeoDataArray, GeoDataset
from hydromt.model.components import ModelComponent

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsBoundaryBase(ModelComponent):
    """
    Base class containing common functionality for point-based SFINCS boundary
    components (e.g., water level boundaries and discharge points).

    Subclasses must set the class attribute `_default_varname` to the variable
    name used inside the dataset (for example "bzs" for water level, or "dis"
    for discharge).
    """

    _default_varname: Union[
        str, List[str]
    ] = None  # must be set in subclass ("dis" or "bzs" or ["hs","tp","wd","ds"])

    def __init__(self, model: "SfincsModel"):
        """
        Initialize the base component.

        Parameters
        ----------
        model : SfincsModel
            Reference to the parent model instance.
        """
        self._data = None
        super().__init__(model=model)

    @property
    def data(self):
        """
        Get the internal xarray dataset/dataarray containing point timeseries
        and geometry information. If not yet initialized, `_initialize()` is called.
        """
        if self._data is None:
            self._initialize()
        assert self._data is not None
        return self._data

    def _initialize(self, skip_read: bool = False) -> None:
        """
        Initialize the internal container. If the model root is in reading mode
        this method will attempt to read data from configured files unless
        `skip_read` is True.
        """
        if self._data is None:
            # default to an empty dataset
            self._data = xr.Dataset()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @property
    def nr_points(self) -> int:
        """
        Return the number of point locations currently stored.
        """
        if hasattr(self.data, "index"):
            return len(self.data.index)
        return 0

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """
        Return the point locations as a GeoDataFrame. If no points are present,
        an empty GeoDataFrame is returned.
        """
        if self.nr_points > 0:
            g = self.data.vector.to_gdf().reset_index(drop=True)
            # Ensure "name" column exists for display purposes (not saved to file, just for display in list)
            if "name" not in g.columns:
                g["name"] = [f"Point {i+1:03d}" for i in g.index]
            return g
        return gpd.GeoDataFrame()

    def set(
        self,
        df: pd.DataFrame = None,
        gdf: gpd.GeoDataFrame = None,
        geodataset: "GeoDataArray" = None,
        merge: bool = True,
        drop_duplicates: bool = True,
    ):
        """
        Set or update the internal data from either a GeoDataArray (geodataset)
        or the pair (df, gdf). If `geodataset` is provided, `df` and `gdf` must
        be None.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Timeseries table (index=time, columns=index identifiers).
        gdf : gpd.GeoDataFrame, optional
            Point locations corresponding to the dataframe columns.
        geodataset : GeoDataArray, optional
            Geospatial xarray object with vector attribute for geometry.
        merge : bool, optional
            If True, merge new locations with existing data instead of replacing.
        drop_duplicates : bool, optional
            If True, drop duplicate points in gdf based on 'name' column or geometry.
        """
        if geodataset is not None:
            if df is not None or gdf is not None:
                raise ValueError(
                    "Provide either 'geodataset' or ('df' and 'gdf'), not both."
                )
            if not hasattr(geodataset, "vector") or not hasattr(geodataset, "dims"):
                raise ValueError("Invalid GeoDataArray provided")
            if geodataset.vector.crs != self.model.crs:
                geodataset = geodataset.vector.to_crs(self.model.crs)
            # keep dataset dims ordering consistent
            index_dim = geodataset.vector.index_dim
            if index_dim != "index":
                # rename the index dimension to "index" if needed
                geodataset = geodataset.rename({index_dim: "index"})
            self._data = geodataset.transpose("time", "index", ...)
            return

        if df is None and gdf is None:
            raise ValueError("Must provide 'df' or 'gdf' (or a GeoDataArray)")

        # update locations and timeseries
        if gdf is not None:
            new_indices = self.set_locations(
                gdf, merge=merge, drop_duplicates=drop_duplicates
            )
            # merging might alter the indices, so update df columns if given
            if df is not None:
                df.columns = new_indices
        if df is not None:
            self.set_timeseries(df)

    def set_locations(
        self,
        gdf: gpd.GeoDataFrame,
        value: Union[float, list] = 0.0,
        merge: bool = True,
        drop_duplicates: bool = True,
    ):
        """
        Add or update point locations. When merging with existing data, the
        new locations are appended; duplicates are removed by name or geometry.
        A dummy timeseries is created for new points if necessary.
        """
        gdf = self._validate_and_prepare_gdf(gdf)

        # If merging with existing data, combine datasets
        if self.nr_points > 0 and merge:
            gdf0 = self.data.vector.to_gdf()
            ds0 = self.data.copy()

            # Drop duplicates from existing dataset based on new gdf if requested
            if drop_duplicates:
                gdf0_dedup = self._drop_duplicate_points(gdf0, gdf)

                ds0 = ds0.reindex(index=gdf0_dedup.index, fill_value=0)
                nr_points_removed = self.nr_points - len(gdf0_dedup)

                if nr_points_removed > 0:
                    logger.info(
                        "Removed {} duplicate points based on 'name' or geometry.".format(
                            nr_points_removed
                        )
                    )

                gdf0 = gdf0_dedup
            # Create dataset with dummy values for new points and combine with existing dataset
            ds_new = self._create_dummy_dataset(gdf, ds0.time, value)
            gds_new = GeoDataset.from_gdf(gdf, ds_new, keep_cols=False)

            # Ensure new dataset has same variable names and ordering as existing dataset
            varnames = (
                [self._default_varname]
                if isinstance(self._default_varname, str)
                else self._default_varname
            )
            gds_combined = xr.concat([ds0[varnames], gds_new], dim="index")
            gds_combined = gds_combined.assign_coords(
                index=np.arange(gds_combined.sizes["index"])
            )
            # Find where the added points ended up in the combined dataset and return those indices
            new_indices = range(len(gdf0), gds_combined.sizes["index"])
            self.set(geodataset=gds_combined)
        else:
            # No merging, overwrite with new dataset with simple dummy timeseries
            time = pd.date_range(*self.model.get_model_time(), periods=2)
            ds_new = self._create_dummy_dataset(gdf, time, value)
            # Combine geometry and dataset into a GeoDataset, assign new integer index and store
            # This is where the name column dissappears!!! Should keep_cols=True be used instead?
            # Or make sure the "name" column is added to the dataset as well?
            # gds_new = GeoDataset.from_gdf(gdf, ds_new, keep_cols=False)
            gds_new = GeoDataset.from_gdf(gdf, ds_new, keep_cols=True)
            gds_new = gds_new.assign_coords(index=np.arange(gds_new.sizes["index"]))
            # Return the new indices which will be 0..N-1 since we are replacing all data
            new_indices = gds_new.index
            self.set(geodataset=gds_new)

        return new_indices

    def set_timeseries(
        self, data: Union[pd.DataFrame, xr.Dataset], varname: str = None
    ):
        """
        Add or update timeseries for existing locations. If the dataframe
        contains all indices it will fully replace the variable, otherwise
        it will be merged, aligned in time and interpolated as required.

        Parameters
        ----------
        data : Union[pd.DataFrame, xr.Dataset]
            Timeseries dataset, with dimensions "time" and "index". Either pandas with single
            variable or xarray Dataset with multiple variables.
        varname : str, optional
            Variable name to write in the dataset (defaults to subclass _default_varname).
        """

        if self.nr_points == 0:
            raise ValueError("Cannot set timeseries without existing locations")

        if varname is None:
            varname = self._default_varname

        if isinstance(data, pd.DataFrame):
            da = self._validate_and_prepare_df(data)
            ds = da.to_dataset(name=varname)
        else:
            ds = data

        if len(ds.indexes["index"]) == self.nr_points:
            # full replacement
            combined = ds[varname]
        else:
            # partial update: align time dims and merge
            all_times = self.data[varname].indexes["time"].union(ds.indexes["time"])
            existing = self.data[varname].reindex(time=all_times)
            ds = ds.reindex(time=all_times)

            combined = existing.copy()
            combined.loc[dict(index=ds.indexes["index"])] = ds[varname]

            # Fill missing values along time dimension
            combined = combined.interpolate_na(dim="time").bfill("time").fillna(0)

        # Replace variable in dataset and ensure time coordinate ordering
        self._data = self.data.reindex(time=combined.time)
        self._data[varname] = combined

    @hydromt_step
    def add_point(
        self,
        x: float,
        y: float,
        name: str = None,
        value: Union[float, list] = 0.0,
        drop_duplicates: bool = True,
    ):
        """
        Convenience to add a single point with a default value for its timeseries.

        Parameters
        ----------
        x, y : float
            Coordinates of the point.
        name : str, optional
            Optional point name.
        value : float or list, optional
            Default timeseries value(s) assigned to the new point. Can be a single float or a list of floats for multiple variables.
        drop_duplicates : bool, optional
            If True, drop duplicate points in gdf based on 'name' column or geometry.
        """
        new_index = self.nr_points + 1
        if name is None:
            name = f"point_{new_index}"

        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([x], [y]), crs=self.model.crs
        )
        gdf["name"] = name
        self.set_locations(
            gdf=gdf, value=value, merge=True, drop_duplicates=drop_duplicates
        )

    def delete(self, index: Union[int, List[int]]):
        """
        Delete one or more point indices from the internal dataset.

        Parameters
        ----------
        index : int or list of int
            Index or list of indices to remove.
        """
        if self.nr_points == 0:
            return
        if not isinstance(index, list):
            index = [index]
        if any(x > (self.nr_points - 1) for x in index):
            raise ValueError("One of the indices exceeds length of index range!")
        # Drop the points from the dataset and reassign a new integer index
        self._data = self.data.drop_isel(index=index).assign_coords(index=np.arange(self.nr_points - len(index)))

    def clear(self):
        """
        Remove all stored points and reset internal dataset to empty.
        """
        self._data = xr.Dataset()

    def _validate_and_prepare_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate a GeoDataFrame that is to be used as point locations. Ensures
        correct dtype, integer unique index and CRS. If gdf is None and data
        exists, the existing geometry is returned.
        """
        if gdf is None:
            if self.nr_points > 0:
                gdf = self.data.vector.to_gdf()
            else:
                raise ValueError("gdf must be provided if no data exists yet")

        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("gdf must be a GeoDataFrame")
        if not pd.api.types.is_integer_dtype(gdf.index) and gdf.index.is_unique:
            raise ValueError("gdf index must be unique integers")
        if not gdf.geometry.type.isin(["Point"]).all():
            raise ValueError("gdf geometry must be Point")
        if gdf.crs != self.model.crs:
            gdf = gdf.to_crs(self.model.crs)

        return gdf

    def _validate_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a timeseries DataFrame. Convert numeric indexes to datetimes
        using tref if necessary and verify coverage against model time.
        """
        if df is None:
            return
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a DataFrame")
        if not pd.api.types.is_integer_dtype(df.columns) and df.columns.is_unique:
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

        da = xr.DataArray(
            df,
            dims=("time", "index"),
            coords={"time": df.index, "index": df.columns},
        )
        return da

    def _drop_duplicate_points(self, gdf_existing, gdf_new):
        """Drop points from gdf_existing that duplicate gdf_new by name or geometry."""
        if "name" in gdf_existing.columns and "name" in gdf_new.columns:
            return gdf_existing[~gdf_existing["name"].isin(gdf_new["name"])]

        precision = 6 if self.model.crs.is_geographic else 2

        def rounded_coords(gdf):
            return gdf.geometry.apply(
                lambda geom: (round(geom.x, precision), round(geom.y, precision))
            )

        coords_existing = rounded_coords(gdf_existing)
        coords_new = rounded_coords(gdf_new)

        return gdf_existing[~coords_existing.isin(coords_new)]

    def _create_dummy_dataset(self, gdf, time, value):
        """Create a dummy xarray.Dataset for given locations and time axis."""
        varnames = (
            [self._default_varname]
            if isinstance(self._default_varname, str)
            else self._default_varname
        )

        if isinstance(value, list):
            if len(value) != len(varnames):
                raise ValueError("Length of value list must match number of variables")

        da_list = []
        for i, var in enumerate(varnames):
            var_value = value[i] if isinstance(value, list) else value
            da_list.append(
                xr.DataArray(
                    np.full((len(time), len(gdf.index)), var_value),
                    dims=("time", "index"),
                    coords={
                        "time": time,
                        "index": gdf.index.values,
                    },
                    name=var,
                )
            )

        return xr.merge(da_list)
