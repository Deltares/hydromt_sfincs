import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(__name__)


class SfincsThinDams(ModelComponent):
    """SFINCS thin dams geometry component.

    This component handles reading, writing, and creating thin dam geometries for
    SFINCS models. Thin dams are infinitely high walls represented as LineString geometries
    in a GeoDataFrame.
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.thd"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Thin dam data, returned as a GeoDataFrame."""
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
        """Initialize thin dams."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS thin dams (.thd) file. Filename is obtained from config if not provided."""

        # Check that read mode is on
        self.root._assert_read_mode()

        # get absolute file path and set it in config if thdfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "thdfile", value=filename
        )

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(f"Thin dams file not found: {abs_file_path}")

        # Read thd file
        struct = utils.read_geoms(abs_file_path)  # =utils.py function
        gdf = utils.linestring2gdf(struct, crs=self.model.crs)  # =utils.py function

        # Add to self._data
        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS thin dams (.thd) file, and set thdfile in config (if it was not already set)"""

        # check that write mode is on
        self.root._assert_write_mode()

        # check if data present:
        if self.data.empty:
            logger.debug("No thin dams data available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            key="thdfile",
            value=filename,
            default="sfincs.thd",
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # Get linestring geometries from gdf
        struct = utils.gdf2linestring(self.data)

        # Write to thd file
        utils.write_geoms(abs_file_path, struct, stype="thd", fmt=fmt)

        # write also as geojson:
        if self.model.write_gis:
            utils.write_vector(
                self.data,
                name="thd",
                root=join(self.root.path, "gis"),
                logger=logger,
            )

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SFINCS thin dams.

        Parameters
        ----------
        gpd.GeoDataFrame :
            Set geopandas object with LineString geometries.
        merge: bool
            Merge with existing thin dams. If False, overwrite existing thin dams.

        .. note::
            When directly using the set method, the GeoDataFrame needs to be in the same CRS as SFINCS model.
        """

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Thin dams must be of type LineString.")
        if not gdf.crs == self.model.crs:
            raise ValueError(
                f"Thin Dams CRS {gdf.crs} does not match model CRS {self.model.crs}."
            )

        # Check if any of the thin dams fall completely outside the model domain
        # If so, give a warning and remove these lines
        outside = gdf.disjoint(self.model.region.union_all())
        if outside.any():
            logger.warning(
                "Some thin dams fall outside model domain. Removing these lines."
            )
            gdf = gdf[~outside]

        # Check if there are any cross sections left
        if gdf.empty:
            # logger.warning("All thin dams fall outside model domain!")
            # return
            raise ValueError("All thin dams fall outside model domain!")

        if merge and self.data is not None:
            gdf0 = self.data
            # add the new data behind the original
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new thin dams to existing ones.")

        self._data = gdf  # set gdf in self._data

    @hydromt_step
    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Create model thin dams (old name: setup_structures).

        Adds model layers:

        * **thd** geom: thin dams

        Parameters
        ----------
        locations: str, Path, gpd.GeoDataFrame
            Path, data source name, or geopandas object for thin dam locations.
        merge: bool, optional
            If True, merge the new thin dams with the existing ones. By default True.
        """
        gdf = self.data_catalog.get_geodataframe(
            locations,
            geom=self.model.region,
            **kwargs,
        ).to_crs(self.model.crs)

        # make sure MultiLineString are converted to LineString
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Thin dams must be of type LineString.")

        # If Linestring z, e.g. when you put in a geojson with height from a weirfile
        # then get rid of the z component
        if gdf.has_z.any():
            gdf["geometry"] = gdf["geometry"].apply(
                lambda geom: LineString([(x, y) for x, y, z in geom.coords])
            )

        # Set the thin dams data
        self.set(gdf, merge)
        # Set config
        self.model.config.set("thdfile", "sfincs.thd")

    def delete(
        self,
        index: Union[list, int],
    ):
        """Remove one or more thin dams.

        Parameters
        ----------
        index: list, int
            Specify thin dams to be dropped from GeoDataFrame.
            If int, drop a single thin dam based on index.
            If list, drop multiple thin dams based on index.
        """
        # Turn int or str into list
        if isinstance(index, int):
            index = [index]

        # Check that any integer in list is not larger than the number of lines
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")

        # Drop lines from GeoDataFrame
        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from thin dams")

        # Check if any cross sections are left
        if self.data.empty:
            logger.warning("All thin dams have been removed!")
            # Set crsfile to None
            self.model.config.set("thdfile", None)

    def clear(self):
        """Clean GeoDataFrame with thin dams."""
        self._data = gpd.GeoDataFrame()
        # Set thdfile to None in config
        self.model.config.set("thdfile", None)

    # %% DDB GUI focused additional functions:
    # snap_to_grid
    # list_names

    def snap_to_grid(self):
        """Returns GeoDataFrame with thin dams snapped to model grid."""
        if self.model.grid_type != "quadtree":
            raise NotImplementedError(
                "Snap to grid is only implemented for quadtree grids."
            )
        snap_gdf = self.model.quadtree_grid.snap_to_grid(self.data)
        return snap_gdf

    def list_names(self):
        """Give list of names of thin dams."""
        if self.data.empty:
            return []
        # The thin dams do not really have names,
        # but we can use the index and turn into strings
        names = [str(i + 1) for i in self.data.index]
        return names
