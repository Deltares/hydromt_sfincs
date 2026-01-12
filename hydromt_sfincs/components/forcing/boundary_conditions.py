import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydromt.gis.vector import GeoDataArray
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

    _default_varname: str = None  # must be set in subclass ("dis" or "bzs")

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
            return self.data.vector.to_gdf()
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
        value: float = 0.0,
        merge: bool = True,
        drop_duplicates: bool = True,
    ):
        """
        Add or update point locations. When merging with existing data, the
        new locations are appended; duplicates are removed by name or geometry.
        A dummy timeseries is created for new points if necessary.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with Point geometries for new locations.
        value : float, optional
            Default value used for dummy timeseries.
        merge : bool, optional
            If True and existing points exist, merge new locations with existing.
        drop_duplicates : bool, optional
            If True, drop duplicate points in gdf based on 'name' column or geometry.
        """
        gdf = self._validate_and_prepare_gdf(gdf)

        if self.nr_points > 0 and merge:
            # parse existing to dataframe
            df0 = (
                self.data[self._default_varname]
                .transpose(..., self.data.vector.index_dim)
                .to_pandas()
            )
            gdf0 = self.data.vector.to_gdf()

            # TODO discuss whether we want to filter; or make it users responsibility
            if drop_duplicates:
                if "name" in gdf0.columns and "name" in gdf.columns:
                    # remove existing points with the same name
                    gdf0 = gdf0[~gdf0["name"].isin(gdf["name"])]
                else:
                    # fallback to geometry-based matching to avoid duplicates
                    if self.model.crs.is_geographic:
                        precision = 6
                    else:
                        precision = 2
                    gdf0["__coords__"] = gdf0.geometry.apply(
                        lambda geom: (
                            round(geom.x, precision),
                            round(geom.y, precision),
                        )
                    )
                    gdf["__coords__"] = gdf.geometry.apply(
                        lambda geom: (
                            round(geom.x, precision),
                            round(geom.y, precision),
                        )
                    )
                    gdf0 = gdf0[~gdf0["__coords__"].isin(gdf["__coords__"])]
                    gdf0 = gdf0.drop(columns="__coords__")
                    gdf = gdf.drop(columns="__coords__")

                df0 = df0.reindex(gdf0.index, axis=1, fill_value=0)
                nr_points_removed = self.nr_points - len(gdf0)
                if nr_points_removed > 0:
                    logger.info(
                        "Removed {} duplicate points based on 'name' or geometry.".format(
                            nr_points_removed
                        )
                    )

            # create matching dataframe for new points
            df_new = pd.DataFrame(index=df0.index, columns=gdf.index, data=value)
            gdf = self._align_gdf_and_df(gdf, df_new)

            gdf_combined = pd.concat([gdf0, gdf], ignore_index=True)
            df_combined = pd.concat([df0, df_new], axis=1)
            df_combined.columns = gdf_combined.index

            new_indices = gdf_combined.index.difference(range(len(gdf0)))
            self._finalize_set(df_combined, gdf_combined)
        else:
            # overwrite existing data with new minimal timeseries
            gdf = gdf.reset_index(drop=True)
            df_new = pd.DataFrame(
                index=pd.date_range(*self.model.get_model_time(), periods=2),
                columns=gdf.index,
                data=value,
            )
            gdf = self._align_gdf_and_df(gdf, df_new)
            new_indices = gdf.index
            self._finalize_set(df_new, gdf)

        return new_indices

    def set_timeseries(self, df: pd.DataFrame, varname: str = None):
        """
        Add or update timeseries for existing locations. If the dataframe
        contains all indices it will fully replace the variable, otherwise
        it will be merged, aligned in time and interpolated as required.

        Parameters
        ----------
        df : pd.DataFrame
            Timeseries table, indexed by time. Columns correspond to point indices.
        varname : str, optional
            Variable name to write in the dataset (defaults to subclass _default_varname).
        """
        df = self._validate_and_prepare_df(df)

        if self.nr_points == 0:
            raise ValueError("Cannot set timeseries without existing locations")

        if varname is None:
            varname = self._default_varname

        new_da = xr.DataArray(
            df,
            dims=("time", "index"),
            coords={"time": df.index, "index": df.columns},
            name=varname,
        )

        if len(new_da.indexes["index"]) == self.nr_points:
            # full replacement
            combined = new_da
        else:
            # partial update: align time dims and merge
            all_times = self.data[varname].indexes["time"].union(new_da.indexes["time"])
            existing = self.data[varname].reindex(time=all_times)
            new_da = new_da.reindex(time=all_times)

            combined = existing.copy()
            combined.loc[dict(index=new_da.indexes["index"])] = new_da

            # Fill missing values along time dimension
            combined = combined.interpolate_na(dim="time").bfill("time").fillna(0)

        # Replace variable in dataset and ensure time coordinate ordering
        self._data = self._data.reindex(time=combined.time)
        self._data[varname] = combined

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

        return df

    def _align_gdf_and_df(
        self, gdf: gpd.GeoDataFrame, df: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        """
        Align gdf index with dataframe columns. If sizes match but indices differ,
        try to infer an index column in gdf to set as the index. Otherwise assume
        order is correct.
        """
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

    def _finalize_set(
        self, df: pd.DataFrame, gdf: gpd.GeoDataFrame, varname: str = None
    ):
        """
        Finalize updating internal dataset from (df, gdf) by creating a GeoDataArray
        and storing a dataset transposed to ('time', 'index').
        """
        if varname is None:
            varname = self._default_varname

        gdf.index.name = "index"
        df.columns.name = "index"
        df.index.name = "time"

        da = GeoDataArray.from_gdf(gdf.to_crs(self.model.crs), data=df, name=varname)
        ds = da.to_dataset()
        self._data = ds.transpose("time", "index")

    def add_point(
        self,
        x: float,
        y: float,
        name: str = None,
        value: float = 0.0,
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
        value : float, optional
            Default timeseries value assigned to the new point.
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
        self._data = self.data.drop_isel(index=index)

    def clear(self):
        """
        Remove all stored points and reset internal dataset to empty.
        """
        self._data = xr.Dataset()
