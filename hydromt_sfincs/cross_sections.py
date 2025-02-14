import geopandas as gpd
import shapely
import pandas as pd
from pathlib import Path
from typing import Union

from hydromt.model.components import ModelComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class SfincsCrossSections(ModelComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs.crs"
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> pd.GeoDataFrame:
        """Cross-section lines data.

        Return geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize cross-section lines."""
        if self._data is None:
            # self._data = dict() 
            self._data = gpd.GeoDataFrame()#FIXME - right?
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self):
        """Read in all cross-section lines."""
        # Read input file:
        struct = utils.read_geoms(self._filename) #=utils.py function
        gdf = utils.linestring2gdf(struct, crs=self.model.crs) #=utils.py function
        # Add to self._data            
        self.set(gdf, merge=False)

    def write(self, filename=None): #TODO - TL: filename=None - still needed?
        """Write crsfile."""
        # change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%.6f"
        else:
            fmt = "%.1f"

        # #TODO add to config - 
        # # If filename is not None:
            # self.config.XXX
            # self._filename = XXX
        struct = utils.gdf2linestring(self.data)
        utils.write_geoms(self._filename, struct, stype="crs", fmt=fmt) #=utils.py function

        # TODO - write also as geojson - TL: at what level do we want to do that?
        # if self._write_gis:
        #     self.write_vector(variables=["geoms"])

    def set(
            self,
            gdf: gpd.GeoDataFrame,
            merge: bool = True
    ):
        """Set cross-section lines.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with Cross-section lines to self.data
        name: str
            Geometry name.
        """        
        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Cross-sections must be of type LineString.")

        # Clip points outside of model region:
        within = gdf.within(self.model.region) # same as 'inpolygon' function
        if within.all() == False:
            raise ValueError("None of cross-sections fall within model domain.")
        elif within.any() == False:
            gdf = gdf[~within]
            self.logger.info("Some of cross-sections fall out of model domain. Removing lines.")

        if merge and self.data is not None:
            gdf0 = self.data
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf0], ignore_index=True))
            self.logger.info("Adding new cross-sections to existing ones.")
        
        # set gdf in self.data    
        self._data = gdf

    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Create model cross-section lines.
        (old name: setup_observation_lines)

        Adds model layers:

        * **crs** geom: cross-section lines

        Arguments
        ---------
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for cross-section lines.
        merge: bool, optional
            If True, merge the new cross-section lines with the existing ones. By default True.
        """
        # FIXME ensure the catalog is loaded before adding any new entries
        # self.data_catalog.sources TODO: check if still needed

        gdf = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, assert_gtype=None, **kwargs
        ).to_crs(self.crs)

        # make sure MultiLineString are converted to LineString
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)        

        self.set(gdf, merge)

        #TODO - add to config: self.model.config
        # self.model.config(f"{name}file", f"sfincs.{name}")
        # self.set_config(f"{name}file", f"sfincs.{name}")

    def add(self,
                   gdf: gpd.GeoDataFrame,
                   ):
        """Add multiple lines to cross-sections.
        
        Arguments
        ---------
        gdf: gpd.GeoDataFrame
            GeoDataFrame with locations and names of cross-section lines to be added.
        **NOTE** - coordinates of cross-sections in GeoDataFrame need to be in the same CRS as SFINCS model.
        """        
        self.set(gdf, merge=True)

    def delete(self,
                   index: int, #FIXME - should this be List(int)?
                   ):
        """Remove (multiple) line(s) from cross-sections.
        
        Arguments
        ---------
        index: int
            Specify indices (int) of point(s) to be dropped from GeoDataFrame of cross-sections.
        """        
        if index.any() > (len(self.data.index)-1): #TODO - check if this is correct
            raise ValueError("One of the indices exceeds length of index range!")    
        
        self.data.drop(index).reset_index(drop=True)
        self.logger.info('Dropping line(s) from cross-sections')    

    def clear(self):
        """Clean GeoDataFrame with cross sections."""
        self.data  = gpd.GeoDataFrame()

    def add_line(self, 
                  x: float, 
                  y: float,
                  name: str,
                  ):
        """Add single line to cross-sections.
        
        Arguments
        ---------
        x: float
            multiple x-coordinates for line to be added, minimum of 2 data points per line
        y: float
            multiple y-coordinates for line to be added, minimum of 2 data points per line
        name: str        
            Name for line to be added
        **NOTE** - x&y coordinates need to be in the same CRS as SFINCS model.
        """
        line = shapely.geometry.LineString(x, y)
        d = {"name": name, "long_name": None, "geometry": line}

        self.data.append(d) #add line directly to gdf
    
    def delete_line(self, 
                     name_or_index: Union[str, int],
                     ):
        """Remove line from cross-sections.
        This function finds the wanted index, after which the generic delete function is called.
        
        Arguments
        ---------
        name_or_index: str, int
            Specify either name (str) or index (int) of cross-section to be dropped from GeoDataFrame.
        """                
        if type(name_or_index) == str:
            for id, row in self.data.iterrows():
                if row["name"] == name_or_index:
                    index = id
            raise ValueError("Cross section " + name_or_index + " not found!")    
        elif type(name_or_index) == int:
            index = name_or_index
        else:
            raise ValueError('Wrong input type given for function delete_line')        
        
        self.delete(index) #calls the generic delete function
        return
    
    def list_names(self):
        """Give list of names of cross sections."""
        names = list(self.data.name)
        return names