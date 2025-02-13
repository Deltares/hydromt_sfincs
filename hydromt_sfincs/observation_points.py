import geopandas as gpd
import shapely
import pandas as pd
from pathlib import Path
from typing import Union

from hydromt.model.components import ModelComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class SfincsObservationPoints(ModelComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs.obs"
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> pd.GeoDataFrame:
        """Model geometries.

        Return dict of geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize geoms."""
        if self._data is None:
            self._data = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self):
        """Read in all observation points."""
        # Read input file:
        gdf = utils.read_xyn(self._filename, crs=self.model.crs) #=utils.py function

        # Add to self._data            
        self.set(gdf, merge=False)

    def write(self, filename=None): #TODO - TL: filename=None - still needed?
        """Write obsfile."""
        # change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%.6f"
        else:
            fmt = "%.1f"

        # #TODO add to config - 
        # # If filename is not None:
            # self.config.XXX
            # self._filename = XXX

        utils.write_xyn(self._filename, self.data, fmt=fmt) #=utils.py function

        # TODO - write also as geojson - TL: at what level do we want to do that?
        # if self._write_gis:
        #     self.write_vector(variables=["geoms"])

    def set(
            self,
            gdf: gpd.GeoDataFrame,
            merge: bool = True
    ):
        """Add data to the geom component.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with observation points to self.data
        name: str
            Geometry name.
        """        
        if not gdf.geometry.type.isin(["Point"]).all():
            raise ValueError("Observation points must be of type Point.")

        # Clip points outside of model region:
        within = gdf.within(self.model.region) # same as 'inpolygon' function
        if within.all() == False:
            raise ValueError("None of observation points fall within model domain.")
        elif within.any() == False:
            gdf = gdf[~within]
            self.logger.info("Some of observation points fall out of model domain. Removing points.")

        if merge and self.data is not None:
            gdf0 = self.data
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf0], ignore_index=True))
            self.logger.info("Adding new observation points to existing ones.")
        
        # set gdf in self.data    
        self._data = gdf

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

        Parameters
        ---------
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for observation point locations.
        merge: bool, optional
            If True, merge the new observation points with the existing ones. By default True.
        """
        # FIXME ensure the catalog is loaded before adding any new entries
        # self.data_catalog.sources TODO: check if still needed

        gdf = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, assert_gtype="Point", **kwargs
        ).to_crs(self.crs)

        self.set(gdf, merge)

        #TODO - add to config: self.model.config
        # self.model.config(f"{name}file", f"sfincs.{name}")
        # self.set_config(f"{name}file", f"sfincs.{name}")

    def add_point(self, 
                  x: float, 
                  y: float,
                  name: str,
                  ):
        """Add single point to observation points.
        
        Parameters
        ---------
        x: float
            x-coordinate for point to be added
        y: float
            y-coordinate for point to be added
        name: str        
            Name for point to be added
        NOTE - x&y values need to be in the same CRS as SFINCS model.
        """
        point = shapely.geometry.Point(x, y)
        d = {"name": name, "long_name": None, "geometry": point}

        self.data.append(d) #add point directly to gdf
        
    def add_points(self, 
                   gdf: gpd.GeoDataFrame,
                   ):
        """Add multiple points to observation points.
        
        Parameters
        ---------
        gdf: gpd.GeoDataFrame
            GeoDataFrame with locations and names of observations points to be added.
        NOTE - coordinates of points in GeoDataFrame need to be in the same CRS as SFINCS model.
        """        
        self.set(gdf, merge=True)

    def delete_point(self, 
                     name_or_index: Union[str, int],
                     ):
        """Remove (multiple) point(s) from observation points.
        
        Parameters
        ---------
        name_or_index: str, int
            Specify either name (str) or index (int) of points to be dropped from GeoDataFrame of observations.
        """                
        if type(name_or_index) == str:
            name = name_or_index
            for index, row in self.data.iterrows():
                if row["name"] == name_or_index:
                    self.data.drop(index).reset_index(drop=True)
                    self.logger.info('Dropping point from observations')
                    return
            raise ValueError("Point " + name + " not found!")    
        elif type(name_or_index) == int:
            index = name_or_index
            if len(self.gdf.index) < index + 1:
                raise ValueError("Index exceeds length!")    
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
            self.logger.info('Dropping point from observations')
            return
        else:
            raise ValueError('Wrong input type given for function delete_point')        
        
    def clear(self):
        """Clean GeoDataFrame with observation points."""
        self.data  = gpd.GeoDataFrame()

    def list_names(self):
        """Give list of names of observation points."""
        names = list(self.data.name)
        return names