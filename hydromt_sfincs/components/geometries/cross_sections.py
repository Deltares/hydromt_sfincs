import logging
import os
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


class SfincsCrossSections(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.crs"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Cross-section lines data.

        Return geopandas.GeoDataFrame
        """
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
        """Initialize cross-section lines."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS cross-sections (*.crs) file"""

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file path and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "crsfile", value=filename
        )

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(f"Cross-sections file not found: {abs_file_path}")

        # Read crs file
        struct = utils.read_geoms(abs_file_path)
        gdf = utils.linestring2gdf(struct, crs=self.model.crs)

        # Add to self._data
        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS cross-sections (*.crs) file,
        and set crsfile in config (if it was not already set)"""

        # Check that data is not empty
        if self.data.empty:
            logger.info("No cross-sections available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            "crsfile",
            value=filename,
            default="sfincs.crs",
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

        # write also as geojson:
        if self.model.write_gis:
            root = join(self.model.root.path, "gis")

            if not os.path.isdir(root):
                os.makedirs(root)

            self.data.to_file(join(root, "crs.geojson"), driver="GeoJSON")

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SFINCS cross-sections.

        Arguments
        ---------
        gpd.GeoDataFrame :
            Set geopandas object with LineString geometries.
        merge: bool
            Merge with existing thin dams. If False, overwrite existing thin dams.
        **NOTE** - coordinates of LineString geometries in GeoDataFrame need to be in the same CRS as SFINCS model.
        """

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Cross-sections must be of type LineString.")

        # Check that gdf has a name column
        # if "name" not in gdf.columns:
        #     raise ValueError("Cross-sections must have a 'name' column.")
        # FIXME - TL: should we check on this for cross-sections, observation points, weirs and thin dams (too)?

        # Check that all rows have a unique name
        # if not gdf["name"].is_unique:
        # raise ValueError("Cross-section names must be unique.")
        # FIXME - TL: should we check on this for cross-sections, observation points, weirs and thin dams (too)?

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
            # logger.warning("All cross-sections fall outside model domain!")
            # return
            raise ValueError("All cross-sections fall outside model domain!")

        if merge and self.data is not None:
            gdf0 = self.data
            # add the new data behind the original
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new cross-sections to existing ones.")

        self._data = gdf  # set gdf in self._data

    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Create model cross-sections.
        (old name: setup_observation_lines)

        Adds model layers:

        * **crs** geom: cross-section lines

        Arguments
        ---------
        locations: str, Path, gpd.GeoDataFrame
            Path, data source name, or geopandas object for thin cross-section locations.
        merge: bool, optional
            If True, merge the new cross-sections with the existing ones. By default True.
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

        self.set(gdf, merge)

    def delete(
        self,
        index: Union[list, int],
    ):
        """Remove one or more cross-sections.

        Arguments
        ---------
        index: list, int
            Specify cross-sections to be dropped from GeoDataFrame.
            If int, drop a single cross-section based on index.
            If list, drop multiple cross-sections based on index.
        """
        # Turn int or str into list
        if type(index) == int:
            index = [index]

        # Check that any integer in list is not larger than the number of lines
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")

        # Drop lines from GeoDataFrame
        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from cross-sections")

        # Check if any cross sections are left
        if self.data.empty:
            logger.warning("All cross-sections have been removed!")
            # Set crsfile to None
            self.model.config.set("crsfile", None)

    def clear(self):
        """Clean GeoDataFrame with cross sections."""
        self._data = gpd.GeoDataFrame()
        # Set crsfile to None in config
        self.model.config.set("crsfile", None)  # FIXME - TL: do we want that?

    # %% DDB GUI focused additional functions:
    # snap_to_grid
    # list_names
    # delete_line - FIXME - do we want to have this as option to remove a single on by name (string) for weir/observation point/wavemaker?

    def snap_to_grid(self):
        """Returns GeoDataFrame with cross-sections snapped to model grid."""
        # FIXME - this probably only works for quadtree grids for now
        snap_gdf = self.model.grid.snap_to_grid(self.data)
        return snap_gdf

    def list_names(self):
        """Give list of names of cross sections."""
        if self.data.empty:
            return []
        names = list(self.data["name"])
        return names

    def delete_line(self, index: Union[int, str]):
        """Remove one cross-section based on index or name.

        Arguments
        ---------
        index: int, str
            Specify cross-section to be dropped from GeoDataFrame.
            If int or str, drop a single cross-section based on index or name.
        """

        # Replace names with indices
        if type(index) == str:
            # Find row index of name
            names = list(self.data.name)

            if index not in names:
                raise ValueError("Cross section " + index + " not found!")

        self.delete(index)
        return
