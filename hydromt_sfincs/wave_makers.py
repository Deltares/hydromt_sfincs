import logging
import geopandas as gpd

# import shapely
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING, Union, List

# import os
from os.path import abspath, join, exists

from hydromt.model.components import ModelComponent
from hydromt.model import Model
from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel
logger = logging.getLogger(__name__)


class SfincsWaveMakers(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self.data = gpd.GeoDataFrame()
        super().__init__(
            model=model,
        )

    # @property
    # def data(self) -> gpd.GeoDataFrame:
    #     """Cross-section lines data.

    #     Return geopandas.GeoDataFrame
    #     """
    #     if self._data is None:
    #         self._initialize()
    #     return self._data

    # %% core HydroMT-SFINCS functions:
    # _initialize
    # read
    # write
    # set
    # create
    # delete
    # clear

    # def _initialize(self, skip_read=False) -> None:
    #     """Initialize cross-section lines."""
    #     if self._data is None:
    #         # self._data = dict()
    #         self._data = gpd.GeoDataFrame()  # FIXME - right?
    #         if self.root.is_reading_mode() and not skip_read:
    #             self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS wave makers (*.wvm) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "wvmfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined, so no wave makers in this model
            return

        # Check if wvm file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(f"wave makers file not found: {abs_file_path}")

        # Read wvm file
        struct = utils.read_geoms(abs_file_path)
        gdf = utils.linestring2gdf(struct, crs=self.model.crs)

        # Add to self.data
        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS wave makers (*.wvm) file,
        and set wvmfile in config (if it was not already set)"""

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

        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # Get linestring geometries from gdf
        struct = utils.gdf2linestring(self.data)
        # Write to wvm file
        utils.write_geoms(abs_file_path, struct, stype="wvm", fmt=fmt)

        # TODO - write also as geojson - TL: at what level do we want to do that?
        # if self._write_gis:
        #     self.write_vector(variables=["geoms"])

    def set(self, gdf: Union[gpd.GeoDataFrame, str, Path], merge: bool = True):
        """Set SFINCS wave makers.

        Arguments
        ---------
        str, Path, gpd.GeoDataFrame :
            data source name, Path, or geopandas object with LineString geometries.
        merge: bool
            Merge with existing wave makers. If False, overwrite existing wave makers.
        """

        # Check if gdf is a string or Path. If so, read the file.
        if isinstance(gdf, (str, Path)):
            gdf = self.data_catalog.get_geodataframe(
                gdf, geom=self.model.region, assert_gtype="LineString"
            ).to_crs(self.model.crs)

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("wave makers must be of type LineString.")

        # Check if any of the cross sections fall completely outside the model domain
        # If so, give a warning and remove these lines
        outside = gdf.disjoint(self.model.region)
        if outside.any():
            logger.warning(
                "Some wave makers fall outside model domain. Removing these lines."
            )
            gdf = gdf[~outside]

        # Check if there are any cross sections left
        if gdf.empty:
            logger.warning("All wave makers fall outside model domain!")
            return

        if merge:
            self.data = pd.concat([self.gdf, gdf], ignore_index=True)
            logger.info("Adding new wave makers to existing ones")
        else:
            self.data = gdf
            logger.info("Setting new wave makers")

    def delete(
        self,
        index: Union[list, int],
    ):
        """Remove one or more wave makers.

        Arguments
        ---------
        index: list, int
            Specify wave makers to be dropped from GeoDataFrame.
            If int, drop a single wave maker based on index.
            If list, drop multiple wave makers based on index.
        """
        # Turn int or str into list
        if type(index) == int:
            index = [index]

        # Check that any integer in list is not larger than the number of lines
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")

        # Drop lines from GeoDataFrame
        self.data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from wave makers")

        # Check if any cross sections are left
        if self.data.empty:
            logger.warning("All wave makers have been removed!")
            # Set crsfile to None
            self.model.config.set("wvmfile", None)

    def clear(self):
        """Clean GeoDataFrame with wave makers."""
        self.data = gpd.GeoDataFrame()
        # Set crsfile to None
        self.model.config.set("wvmfile", None)

    def list_names(self):
        """Give list of names of wave makers."""
        # The wave makers do not really have names,
        # but we can use the index and turn into strings
        names = [str(i + 1) for i in self.data.index]
        return names

    def snap_to_grid(self):
        """Returns GeoDataFrame with wave makers snapped to model grid."""
        # TODO - this probably only works for quadtree grids for now
        snap_gdf = self.model.grid.snap_to_grid(self.data)
        return snap_gdf
