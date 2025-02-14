import geopandas as gpd
import shapely
import pandas as pd
from pathlib import Path
from typing import Union

from hydromt.model.components import ModelComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class SfincsThinDams(ModelComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs.thd"
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> pd.GeoDataFrame:
        """Thin dams lines data.

        Return geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize thin dam lines."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()#FIXME - right?
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self):
        """Read in all thin dam lines."""
        # Read input file:
        struct = utils.read_geoms(self._filename) #=utils.py function
        gdf = utils.linestring2gdf(struct, crs=self.model.crs) #=utils.py function
                  
        self.set(gdf, merge=False) # Add to self._data  

    def write(self, filename=None): #TODO - TL: filename=None - still needed?
        """Write thdfile."""
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
        utils.write_geoms(self._filename, struct, stype="thd", fmt=fmt) #=utils.py function

        # TODO - write also as geojson - TL: at what level do we want to do that?
        # if self._write_gis:
        #     self.write_vector(variables=["geoms"])

    def set(
            self,
            gdf: gpd.GeoDataFrame,
            merge: bool = True
    ):
        """Set thin dam lines.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with thin dam lines to self.data
        name: str
            Geometry name.
        """        
        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("thin dams must be of type LineString.")

        # Clip points outside of model region:
        within = gdf.within(self.model.region) # same as 'inpolygon' function
        if within.all() == False:
            raise ValueError("None of thin dams fall within model domain.")
        elif within.any() == False:
            gdf = gdf[~within]
            self.logger.info("Some of thin dams fall out of model domain. Removing lines.")

        if merge and self.data is not None:
            gdf0 = self.data
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf0], ignore_index=True))
            self.logger.info("Adding new thin dams to existing ones.")
        
        # set gdf in self.data    
        self._data = gdf

    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Create model thin dam lines.
        (old name: setup_structures)

        Adds model layers:

        * **thd** geom: thin dam lines

        Arguments
        ---------
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for thin dam lines.
        merge: bool, optional
            If True, merge the new thin dam lines with the existing ones. By default True.
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
        """Add multiple lines to thin dams.
        
        Arguments
        ---------
        gdf: gpd.GeoDataFrame
            GeoDataFrame with locations and names of thin dam lines to be added.
        **NOTE** - coordinates of thin dams in GeoDataFrame need to be in the same CRS as SFINCS model.
        """        
        self.set(gdf, merge=True)

    def delete(self,
                   index: int, #FIXME - should this be List(int)?
                   ):
        """Remove (multiple) line(s) from thin dams.
        
        Arguments
        ---------
        index: int
            Specify indices (int) of thin dam(s) to be dropped from GeoDataFrame of thin dams.
        """        
        if index.any() > (len(self.data.index)-1): #TODO - check if this is correct
            raise ValueError("One of the indices exceeds length of index range!")    
        
        self.data.drop(index).reset_index(drop=True)
        self.logger.info('Dropping line(s) from thin dams')    

    def clear(self):
        """Clean GeoDataFrame with thin dams."""
        self.data  = gpd.GeoDataFrame()

    def snap_to_grid(self):
        snap_gdf = self.model.grid.snap_to_grid(self.gdf) #FIXME - snap_to_grid should be function in grid.py!
        return snap_gdf
    
    def list_names(self):
        """Give list of names of cross sections."""
        names = list(self.data.name)
        return names