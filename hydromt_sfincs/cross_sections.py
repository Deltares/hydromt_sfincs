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


class SfincsCrossSections(ModelComponent):
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
        """Read SFINCS cross-sections (*.crs) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_file_path("crsfile", value=filename)

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if crs file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(f"Cross-sections file not found: {abs_file_path}")

        # Read crs file
        struct = utils.read_geoms(abs_file_path)
        gdf = utils.linestring2gdf(struct, crs=self.model.crs)

        # Add to self.data
        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS cross-sections (*.crs) file,
        and set crsfile in config (if it was not already set)"""

        # Check that data is not empty
        if self.data.empty:
            logger.info("No cross-sections available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.set_file_path(
            "crsfile", "sfincs.crs", value=filename
        )

        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # Get linestring geometries from gdf
        struct = utils.gdf2linestring(self.data)
        # Write to crs file
        utils.write_geoms(abs_file_path, struct, stype="crs", fmt=fmt)

        # TODO - write also as geojson - TL: at what level do we want to do that?
        # if self._write_gis:
        #     self.write_vector(variables=["geoms"])

    def set(self, gdf: Union[gpd.GeoDataFrame, str, Path], merge: bool = True):
        """Set SFINCS cross-sections.

        Arguments
        ---------
        str, Path, gpd.GeoDataFrame :
            data source name, Path, or geopandas object with LineString geometries and "name" column.
        merge: bool
            Merge with existing cross-sections. If False, overwrite existing cross-sections.
        """

        # Check if gdf is a string or Path. If so, read the file.
        if isinstance(gdf, (str, Path)):
            gdf = self.data_catalog.get_geodataframe(
                gdf, geom=self.model.region, assert_gtype="LineString"
            ).to_crs(self.model.crs)

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Cross-sections must be of type LineString.")

        # Check that gdf has a name column
        if "name" not in gdf.columns:
            raise ValueError("Cross-sections must have a 'name' column.")

        # Check that all rows have a unique name
        if not gdf["name"].is_unique:
            raise ValueError("Cross-section names must be unique.")

        # Check if any of the cross sections fall completely outside the model domain
        # If so, give a warning and remove these lines
        outside = gdf.disjoint(self.model.region)
        if outside.any():
            logger.warning(
                "Some cross-sections fall outside model domain. Removing these lines."
            )
            gdf = gdf[~outside]

        # Check if there are any cross sections left
        if gdf.empty:
            logger.warning("All cross-sections fall outside model domain!")
            return

        if merge:
            self.data = pd.concat([self.gdf, gdf], ignore_index=True)
            logger.info("Adding new cross-sections to existing ones")
        else:
            self.data = gdf
            logger.info("Setting new cross-sections")

    def delete(
        self,
        index: Union[list, int, str],
    ):
        """Remove one or more cross-sections.

        Arguments
        ---------
        index: list, int, str
            Specify cross-sections to be dropped from GeoDataFrame.
            If int or str, drop a single cross-section based on index or name.
            If list, drop multiple cross-sections based on index or name.
        """
        # Turn int or str into list
        if type(index) == int or type(index) == str:
            index = [index]

        # Replace names with indices
        for i, indx in enumerate(index):
            if type(indx) == str:
                # Find row index of name
                names = list(self.data.name)
                if indx not in names:
                    raise ValueError("Cross section " + indx + " not found!")
                else:
                    index[i] = names.index(indx)

        # Check that any integer in list is not larger than the number of lines
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")

        # Drop lines from GeoDataFrame
        self.data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from cross-sections")

        # Check if any cross sections are left
        if self.data.empty:
            logger.warning("All cross-sections have been removed!")
            # Set crsfile to None
            self.model.config.set("crsfile", None)

    def clear(self):
        """Clean GeoDataFrame with cross sections."""
        self.data = gpd.GeoDataFrame()
        # Set crsfile to None
        self.model.config.set("crsfile", None)

    def list_names(self):
        """Give list of names of cross sections."""
        names = list(self.data.name)
        return names

    def snap_to_grid(self):
        """Returns GeoDataFrame with cross-sections snapped to model grid."""
        # TODO - this probably only works for quadtree grids for now
        snap_gdf = self.model.grid.snap_to_grid(self.data)
        return snap_gdf
