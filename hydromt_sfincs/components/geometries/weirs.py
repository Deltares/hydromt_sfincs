import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(__name__)


class SfincsWeirs(ModelComponent):
    """SFINCS weir geometry component.

    This component handles reading, writing, and creating weir geometries for
    SFINCS models. Weirs can be used to represent flow control structures, dikes, and levees, and
    are represented as LineString geometries (with elevation) in a GeoDataFrame.
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.weir"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Weirs lines data, returned as a GeoDataFrame."""
        if self._data is None:
            self._initialize()
        return self._data

    # %% core HydroMT-SFINCS functions:
    # _initialize
    # read
    # write
    # set
    # create
    # delete
    # clear

    def _initialize(self, skip_read=False) -> None:
        """Initialize weir lines."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS weir (.weir) file. Filename is obtained from config if not provided."""

        # Check that read mode is on
        self.root._assert_read_mode()

        # get absolute file path and set it in config if weirfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "weirfile", value=filename
        )

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(f"Weir file not found: {abs_file_path}")

        # Read weir file:
        struct = utils.read_geoms(abs_file_path)  # =utils.py function
        gdf = utils.linestring2gdf(struct, crs=self.model.crs)  # =utils.py function

        # Add to self._data
        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS weir (.weir) file, and set weirfile in config (if it was not already set)."""

        # check that write mode is on
        self.root._assert_write_mode()

        # check if data present:
        if self.data.empty:
            logger.debug("No weir data available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            key="weirfile",
            value=filename,
            default="sfincs.weir",
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # Get linestring geometries from gdf
        struct = utils.gdf2linestring(self.data)

        # Write to weirfile
        utils.write_geoms(abs_file_path, struct, stype="weir", fmt=fmt)

        # write also as geojson:
        if self.model.write_gis:
            utils.write_vector(
                self.data,
                name="weir",
                root=join(self.root.path, "gis"),
                logger=logger,
            )

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SFINCS weir lines.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with weir lines to self.data
        merge: bool
            Merge with existing weir. If False, overwrite existing weirs.

        .. note::
            When directly using the set method, the GeoDataFrame needs to be in the same CRS as SFINCS model.
        """
        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Weirs must be of type LineString.")
        if not gdf.crs == self.model.crs:
            raise ValueError(
                f"Weirs CRS {gdf.crs} does not match model CRS {self.model.crs}."
            )

        # Check if any of the cross sections fall completely outside the model domain
        # If so, give a warning and remove these lines
        outside = gdf.disjoint(self.model.region.union_all())
        if outside.any():
            logger.warning(
                "Some weirs fall outside model domain. Removing these lines."
            )
            gdf = gdf[~outside]

        # Check if there are any cross sections left
        if gdf.empty:
            # logger.warning("All thin dams fall outside model domain!")
            # return
            raise ValueError("All weirs fall outside model domain!")

        if merge and self.data is not None:
            gdf0 = self.data
            # add the new data behind the original
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new weirs to existing ones.")

        self._data = gdf  # set gdf in self._data

    @hydromt_step
    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        dep: Union[str, Path, xr.DataArray] = None,
        buffer: float = None,
        dz: float = None,
        merge: bool = True,
        **kwargs,
    ):
        """Create model weir lines (old name: setup_structures).

        If elevation 'z' at weir locations is not provided, it can be calculated
        from the model elevation directly (dep supplied, but not dz),
        or from the model elevation plus an additional set elevation 'dz'
        (dep & dz supplied).

        Adds model layers:

        * **weir** geom: weir lines

        Parameters
        ----------
        locations: str, Path, gpd.GeoDataFrame
            Path, data source name, or geopandas object for weir lines.
        dep : str, Path, xr.DataArray, optional
            Data source name, Path, or xarray raster object ('elevation') describing the depth in an
            alternative resolution which is used for sampling the weir.

            .. note::
                Currently, you can only supply one datasource for dep,
                or use the -coarser- active dep data in self.grid.data if dep not provided,
                but not your whole elevation_list list!

            .. note::
                Tip: use fine resolution dep_subgrid.tif for merged high-res data
                in case of using multiple elevation datasets.
        buffer : float, optional
            If provided, describes the distance from the centerline to the foot of the structure.
            This distance is supplied to the raster.sample as the window (wdw).
        dz: float, optional
            If provided, for weir structures the z value is calculated from
            the model elevation (dep) plus dz.
        merge: bool, optional
            If True, merge the new weir lines with the existing ones. By default True.
        """
        gdf = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, **kwargs
        ).to_crs(self.model.crs)

        # make sure MultiLineString are converted to LineString
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Weirs must be of type LineString.")

        # expected columns in gdf
        cols = {
            "weir": ["name", "z", "par1", "geometry"],
        }

        # keep relevant columns
        gdf = gdf[[c for c in cols["weir"] if c in gdf.columns]]

        # check whether z values are part of the gdf, or need to be calculated
        gdf_has_z = (
            gdf.geometry.apply(lambda geom: geom.has_z).all() or "z" in gdf.columns
        )

        # check if z values are provided or can be calculated
        if not gdf_has_z and (dep is None and dz is None):
            # check if z values are part of the linestrings, so called linestringZ
            raise ValueError(
                "Weir structure requires z values, or 'dep' or 'dz' input to determine these on the fly."
            )
        elif dep is not None or dz is not None:
            # determine elevation from dep and dz, if data parsed
            gdf = self.determine_weir_elevation(gdf, dep, buffer, dz)
            # if dep is not provided, the active dep data in self.grid.data is loaded,
            # within function determine_weir_elevation
            logger.info("Determined elevations for weir based on elevation data.")

        # Set the weir data
        self.set(gdf, merge)
        # Set config
        self.model.config.set("weirfile", "sfincs.weir")

    def delete(
        self,
        index: Union[list, int],
    ):
        """Remove one or more weir, based on index.

        Parameters
        ----------
        index: list, int
            Specify weirs to be dropped from GeoDataFrame.
            If int, drop a single weir based on index.
            If list, drop multiple weir based on index.
        """
        # Turn int or str into list
        if isinstance(index, int):
            index = [index]

        # Check that any integer in list is not larger than the number of lines
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")

        # Drop lines from GeoDataFrame
        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from weirs")

        # Check if any cross sections are left
        if self.data.empty:
            logger.warning("All weirs have been removed!")
            # Set crsfile to None
            self.model.config.set("weirfile", None)

    def clear(self):
        """Clean GeoDataFrame with weirs."""
        self._data = gpd.GeoDataFrame()
        # Set weirfile to None in config
        self.model.config.set("weirfile", None)

    # %% HydroMT-SFINCS focused additional functions:
    # determine_weir_elevation

    def determine_weir_elevation(
        self,
        gdf: gpd.GeoDataFrame,
        dep: Union[str, Path, xr.DataArray] = None,
        buffer: float = None,
        dz: float = None,
    ):
        """Determine z values for weir structures.  Called by .create() function if dep (/and dz) are provided.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame with weir lines (without z values)
        dep : str, Path, xr.DataArray, optional
            Data source name, Path, or xarray raster object ('elevation') describing the depth in an
            alternative resolution which is used for sampling the weir.
            **NOTE** - currently, you can only supply one datasource for dep,
                or use the -coarser- active dep data in self.grid.data if dep not provided,
                but not your whole elevation_list list!
            **NOTE** Tip: use fine resolution dep_subgrid.tif for merged high-res data
                in case of using multiple elevation datasets.
        buffer : float, optional
            If provided, describes the distance from the centerline to the foot of the structure.
            This distance is supplied to the raster.sample as the window (wdw).
        dz: float, optional
            If provided, describes the vertical offset to be applied to the weir elevation.
        """
        # taken from old 'sfincs.py'>setup_structures function

        structs = utils.gdf2linestring(gdf)  # check if it parsed correct

        # get elevation data either from model itself, or separate input
        if dep is None or dep == "dep":
            assert "dep" in self.model.grid.data, "dep layer not found"
            elv = self.model.grid.data["dep"]
        else:
            elv = self.data_catalog.get_rasterdataset(
                dep, geom=self.model.region, buffer=5, variables=["elevation"]
            )
        # mask nodata values
        elv = elv.raster.mask_nodata()

        # calculate window size from buffer
        if buffer is not None:
            res = abs(elv.raster.res[0])
            if elv.raster.crs.is_geographic:
                res = res * 111111.0
            window_size = int(np.ceil(buffer / res))
        else:
            window_size = 0
        logger.debug(f"Sampling elevation with window size {window_size}")

        # interpolate dep data to points of weirs
        structs_out = []
        for s in structs:
            pnts = gpd.points_from_xy(x=s["x"], y=s["y"])
            zb = elv.raster.sample(
                gpd.GeoDataFrame(geometry=pnts, crs=self.model.crs), wdw=window_size
            )
            if zb.ndim > 1:
                zb = zb.max(axis=1)

            if zb.isnull().any():
                # get id and coordinates of nan point(s)
                nan_id = zb.isnull().idxmax().values

                xtmp = s["x"][nan_id]
                ytmp = s["y"][nan_id]

                logger.warning(
                    f"Weir point {xtmp} {ytmp} has no elevation data. Filled now with nearest non-NaN value. Please check your input!"
                )

                # Interpolate missing values
                zb = zb.interpolate_na(dim="index", method="nearest")

                # Forward fill to handle NaN at the ends
                zb = zb.ffill(dim="index")

                # Backward fill to handle NaN at the ends
                zb = zb.bfill(dim="index")

                if zb.isnull().any():
                    # if still didn't work, raise error
                    raise ValueError("Filling NaN values failed for weirs ")

            s["z"] = zb.values

            # in case of dz, add this to the elevation
            if dz is not None:
                s["z"] += float(dz)

            structs_out.append(s)

        gdf = utils.linestring2gdf(structs_out, crs=self.model.crs)

        return gdf

    # %% DDB GUI focused additional functions:
    # snap_to_grid
    # list_names

    def snap_to_grid(self):
        """Returns GeoDataFrame with weirs snapped to model grid."""
        # FIXME - this probably only works for quadtree grids for now
        if self.model.grid_type != "quadtree":
            raise NotImplementedError(
                "Snap to grid is only implemented for quadtree grids."
            )
        snap_gdf = self.model.grid.snap_to_grid(self.data)
        return snap_gdf

    def list_names(self):
        """Give list of names of cross sections."""
        if self.data.empty:
            return []
        names = list(self.data["name"])
        return names
