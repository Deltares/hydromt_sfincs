import logging
import geopandas as gpd
import shapely
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING, Union, List
import os
from os.path import abspath, join, exists

from hydromt.model.components import ModelComponent
from hydromt.model import Model
from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(__name__)


class SfincsObservationPoints(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.obs"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Observation point data.

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
        """Initialize observation points."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS observation points (*.obs) file."""

        # check that read mode is on
        self.root._assert_read_mode()

        # get absolute file path and set it in config if obsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "obsfile", value=filename
        )

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(f"Observation points file not found: {abs_file_path}")

        # Read input file:
        gdf = utils.read_xyn(abs_file_path, crs=self.model.region.crs)  # =utils.py function

        # Add to self._data
        self.set(gdf, merge=False)

    def write(self, filename=None):
        """Write SFINCS observation points (*.obs) file,
        and set obsfile in config (if it was not already set)"""

        # check that write mode is on
        self.root._assert_write_mode()

        # check if data present:
        if self.data.empty:
            logger.debug("No observation points data available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            key="obsfile", value=filename, default="sfincs.obs"
        )

        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        utils.write_xyn(abs_file_path, self.data, fmt=fmt)  # =utils.py function

        # write also as geojson:
        if self.model._write_gis:
            root = join(self.model.root.path, "gis")

            if not os.path.isdir(root):
                os.makedirs(root)

            self.data.to_file(join(root, f"obs.geojson"), driver="GeoJSON")

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SFINCS observation points.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with observation points to self.data
        merge: bool
            Merge with existing observation points. If False, overwrite existing observation points.
        **NOTE** - coordinates of points in GeoDataFrame need to be in the same CRS as SFINCS model.
        """
        if not gdf.geometry.type.isin(["Point"]).all():
            raise ValueError("Observation points must be of type Point.")

        # Clip points outside of model region:
        within = gdf.within(self.model.region.unary_union)
        # within = gdf.within(self.model.region.union_all)
        # > FIXME - tried to overcome deprecation warning of unary_union, but suggested alternative does not work
        # NOTE - .within does same as 'inpolygon' function

        if within.any() == True:
            if within.all() == False:
                # keep points that fall within region
                gdf = gdf[within]

                # write away the names of points that are removed
                gdf_name = gdf.name[~within]
                logger.info(
                    "Some of the observation points fall out of model domain. Removing points: "
                    + str(gdf_name.values)
                )
        else:
            raise ValueError("None of observation points fall within model domain.")

        if merge and self.data is not None:
            gdf0 = self.data
            # add the new data behind the original
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new observation points to existing ones.")

        self._data = gdf  # set gdf in self.data

    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Create model observation point locations.
        (old name: setup_observation_points)

        Adds model layers:

        * **obs** geom: observation point locations

        Arguments
        ---------
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for observation point locations.
        merge: bool, optional
            If True, merge the new observation points with the existing ones. By default True.
        """
        gdf = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, **kwargs
        ).to_crs(self.model.crs)

        if not gdf.geometry.type.isin(["Point"]).all():
            raise ValueError("Observation points should be of type Point")

        self.set(gdf, merge)

    def delete(
        self,
        index: Union[list, int],
    ):
        """Remove one or more observation points.

        Arguments
        ---------
        index: list, int
            Specify observation points to be dropped from GeoDataFrame.
            If int, drop a single observation point based on index.
            If list, drop multiple observation points based on index.
        """
        # Turn int or str into list
        if type(index) == int:
            index = [index]

        # Check that any integer in list is not larger than the number of points
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")
        # FIXME len(self.data.index) or len(self.data.index)

        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping point(s) from observations")

    def clear(self):
        """Clean GeoDataFrame with observation points."""
        self._data = gpd.GeoDataFrame()
        # Set obsfile to None
        self.model.config.set("thdfile", None) #FIXME - TL: do we want that?

    # %% DDB GUI focused additional functions:
    # add_point
    # delete_point
    # list_names

    def add_point(
        self,
        x: float,
        y: float,
        name: str,
    ):
        """Add single point to observation points.

        Arguments
        ---------
        x: float
            x-coordinate for point to be added
        y: float
            y-coordinate for point to be added
        name: str
            Name for point to be added
        **NOTE** - x&y values need to be in the same CRS as SFINCS model.
        """
        point = shapely.geometry.Point(x, y)
        d = {"name": name, "long_name": None, "geometry": point}

        # Create a new GeoDataFrame for the Point
        gdf = gpd.GeoDataFrame([d], crs=self.model.region.crs)

        self.set(gdf, merge=True)

    def delete_point(
        self,
        name_or_index: Union[str, int],
    ):
        """Remove point from observation points.
        This function finds the wanted index, after which the generic delete function is called.

        Arguments
        ---------
        name_or_index: str, int
            Specify either name (str) or index (int) of point to be dropped from GeoDataFrame of observations.
        """
        if type(name_or_index) == str:
            index = None
            for id, row in self.data.iterrows():
                if row["name"] == name_or_index:
                    index = int(id)
            if index == None:
                raise ValueError("Point " + name_or_index + " not found!")
        elif type(name_or_index) == int:
            index = int(name_or_index)
        else:
            raise ValueError("Wrong input type given for function delete_point")

        self.delete(index)
        return

    def list_names(self):
        """Give list of names of observation points."""
        if self.data.empty:
            return []
        names = list(self.data["name"])
        return names
