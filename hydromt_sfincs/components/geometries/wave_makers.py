import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(__name__)


class SfincsWaveMakers(ModelComponent):
    """SFINCS wave makers geometry component.

    This component handles reading, writing, and creating wave maker geometries for
    SFINCS models. Wave makers are represented as LineString geometries in a GeoDataFrame.
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.wvm"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Wavemaker lines data, returned as a GeoDataFrame."""
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
        """Initialize wavemaker lines."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS wave makers (.wvm) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if wvmfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "wvmfile", value=filename
        )

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(f"Wave makers file not found: {abs_file_path}")

        # Read wvm file
        struct = utils.read_geoms(abs_file_path)
        gdf = utils.linestring2gdf(struct, crs=self.model.crs)

        # Add to self._data
        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS wave makers (.wvm) file, and set wvmfile in config (if it was not already set)."""

        # Check that data is not empty
        if self.data.empty:
            logger.info("No wave makers available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            "wvmfile",
            value=filename,
            default="sfincs.wvm",
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

        # Write to wvm file
        utils.write_geoms(abs_file_path, struct, stype="wvm", fmt=fmt)

        # write also as geojson:
        if self.model.write_gis:
            utils.write_vector(
                self.data,
                name="wvm",
                root=join(self.root.path, "gis"),
                logger=logger,
            )

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SFINCS wave makers.

        Parameters
        ----------
        gpd.GeoDataFrame :
            Set geopandas object with LineString geometries.
        merge: bool
            Merge with existing wave makers. If False, overwrite existing wave makers.

        .. note::
            When directly using the set method, the GeoDataFrame needs to be in the same CRS as SFINCS model.
        """

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Wave makers must be of type LineString.")
        if not gdf.crs == self.model.crs:
            raise ValueError(
                f"Wave makers CRS {gdf.crs} does not match model CRS {self.model.crs}."
            )

        # Check if any of the wave makers fall completely outside the model domain
        # If so, give a warning and remove these lines
        outside = gdf.disjoint(self.model.region.union_all())
        if outside.any():
            logger.warning(
                "Some wave makers fall outside model domain. Removing these lines."
            )
            gdf = gdf[~outside]

        # Check if there are any wave makers left
        if gdf.empty:
            # logger.warning("All wave makers fall outside model domain!")
            # return
            raise ValueError("All wave makers fall outside model domain!")

        if merge and self.data is not None:
            gdf0 = self.data
            # add the new data behind the original
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new wave makers to existing ones.")

        self._data = gdf  # set gdf in self._data

    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Create model wave makers (old name: non existent).

        Adds model layers:

        * **wvm** geom: wave makers

        Parameters
        ----------
        locations: str, Path, gpd.GeoDataFrame
            Path, data source name, or geopandas object for thin dam locations.
        merge: bool, optional
            If True, merge the new wave makers with the existing ones. By default True.

        .. note::
            The order in which you have specified the points matters for the direction
            in which waves will be generated.
        """
        gdf = self.data_catalog.get_geodataframe(
            locations,
            geom=self.model.region,
            **kwargs,
        ).to_crs(self.model.crs)

        # make sure MultiLineString are converted to LineString
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("wave makers must be of type LineString.")

        # If Linestring z, e.g. when you put in a geojson with height from a weirfile
        # then get rid of the z component
        if gdf.has_z.any():
            gdf["geometry"] = gdf["geometry"].apply(
                lambda geom: LineString([(x, y) for x, y, z in geom.coords])
            )
        # Set the wave makers data
        self.set(gdf, merge)
        # Set config
        self.model.config.set("wvmfile", "sfincs.wvm")

    def delete(
        self,
        index: Union[list, int],
    ):
        """Remove one or more wave makers.

        Parameters
        ----------
        index: list, int
            Specify wave makers to be dropped from GeoDataFrame.
            If int, drop a single wave maker based on index.
            If list, drop multiple wave makers based on index.
        """
        # Turn int or str into list
        if isinstance(index, int):
            index = [index]

        # Check that any integer in list is not larger than the number of lines
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")

        # Drop lines from GeoDataFrame
        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from wave makers")

        # Check if any wave makers are left
        if self.data.empty:
            logger.warning("All wave makers have been removed!")
            # Set crsfile to None
            self.model.config.set("wvmfile", None)

    def clear(self):
        """Clean GeoDataFrame with wave makers."""
        self._data = gpd.GeoDataFrame()
        # Set crsfile to None
        self.model.config.set("wvmfile", None)

    # %% DDB GUI focused additional functions:
    # snap_to_grid
    # list_names

    def snap_to_grid(self):
        """Returns GeoDataFrame with wave makers snapped to model grid."""
        # FIXME - this probably only works for quadtree grids for now
        if self.model.grid_type != "quadtree":
            raise NotImplementedError(
                "Snap to grid is only implemented for quadtree grids."
            )
        snap_gdf = self.model.quadtree_grid.snap_to_grid(self.data)
        return snap_gdf

    def list_names(self):
        """Give list of names of wave makers."""
        # The wave makers do not really have names,
        # but we can use the index and turn into strings
        names = [str(i + 1) for i in self.data.index]
        return names
