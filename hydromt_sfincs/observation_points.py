import logging
import geopandas as gpd
import shapely
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING, Union, List
from os.path import join

from hydromt.model.components import ModelComponent
from hydromt.model import Model
from hydromt_sfincs import utils

# from hydromt.config import get_set_config_file_variable

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
        """Initialize geoms."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename=None):
        """Read in all observation points."""
        if filename is None:
            self._filename = self.model.config.get("obsfile")
            filename = join(self.model.root.path, self._filename)

        # FIXME check is file exist

        # Read input file:
        gdf = utils.read_xyn(filename, crs=self.model.region.crs)  # =utils.py function

        # Add to self._data
        self.set(gdf, merge=False)

    def write(self, filename=None):
        """Write obsfile."""

        # check if data present:
        if self.data.empty:
            return

        # call function to get back full filepath of config variable "obsfile"
        # function also updates the name in case a filename is provided to this function
        # and if not the case, and obsfile doesn't exist yet, it is initialised with the default of "sfincs.obs"

        file_path = self.model.config.get_set_config_file_variable(
            key="obsfile", value=filename, default_filename="sfincs.obs"
        )

        # change precision of coordinates according to crs
        # FIXME - incorporate in utils.xyn or not?
        if self.model.crs.is_geographic:
            fmt = "%.6f"
        else:
            fmt = "%.1f"

        utils.write_xyn(file_path, self.data, fmt=fmt)  # =utils.py function

        # TODO - write also as geojson - TL: at what level do we want to do that?
        # if self._write_gis:
        #     self.write_vector(variables=["geoms"])

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set observation points.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with observation points to self.data
        name: str
            Geometry name.
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
            locations, geom=self.model.region, assert_gtype="Point", **kwargs
        ).to_crs(self.model.crs)

        self.set(gdf, merge)

        # TODO - add to config: self.model.config
        # self.model.config(f"{name}file", f"sfincs.{name}")
        # self.set_config(f"{name}file", f"sfincs.{name}")

    def delete(
        self,
        index: int,  # FIXME - now we expect list [int] - does that make sense?/should we specify here?
    ):
        """Remove (multiple) point(s) from observation points.

        Arguments
        ---------
        index: int
            Specify indices (int) of point(s) to be dropped from GeoDataFrame of observations.
        """
        if any(x > (len(self.data.index) - 1) for x in index):
            raise ValueError("One of the indices exceeds length of index range!")

        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping point(s) from observations")

    def clear(self):
        """Clean GeoDataFrame with observation points."""
        self._data = gpd.GeoDataFrame()

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
        # self._data = self.data.append(d)  # add point directly to gdf > has been deprecated...

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

        self.delete(index=[index])  # calls the generic delete function as list
        return

    def list_names(self):
        """Give list of names of observation points."""
        names = list(self.data.name)
        return names
