import geopandas as gpd
import shapely
import pandas as pd
from pathlib import Path
from typing import Union

from hydromt.model.components import ModelComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class Sfincswavemakerss(ModelComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs.wvm"
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> pd.GeoDataFrame:
        """wavemakers lines data.

        Return geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data

#%% core HydroMT-SFINCS functions:
    # _initialize
    # read
    # write
    # set
    # create
    # add
    # delete
    # clear
    
    def _initialize(self, skip_read=False) -> None:
        """Initialize wavemakers lines."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()#FIXME - right?
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self):
        """Read in all wavemakers lines."""
        # Read input file:
        struct = utils.read_geoms(self._filename) #=utils.py function
        gdf = utils.linestring2gdf(struct, crs=self.model.crs) #=utils.py function
                  
        self.set(gdf, merge=False) # Add to self._data  

    def write(self, filename=None): #TODO - TL: filename=None - still needed?
        """Write wvmfile."""
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
        utils.write_geoms(self._filename, struct, stype="wvm", fmt=fmt) #=utils.py function

        # TODO - write also as geojson - TL: at what level do we want to do that?
        # if self._write_gis:
        #     self.write_vector(variables=["geoms"])

    def set(
            self,
            gdf: gpd.GeoDataFrame,
            merge: bool = True
    ):
        """Set wavemakers lines.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with wavemakers lines to self.data
        name: str
            Geometry name.
        """        
        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("wavemakers must be of type LineString.")

        # Clip points outside of model region:
        within = gdf.within(self.model.region) # same as 'inpolygon' function
        if within.all() == False:
            raise ValueError("None of wavemakers fall within model domain.")
        elif within.any() == False:
            gdf = gdf[~within]
            self.logger.info("Some of wavemakers fall out of model domain. Removing lines.")

        if merge and self.data is not None:
            gdf0 = self.data
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf0], ignore_index=True))
            self.logger.info("Adding new wavemakers to existing ones.")
        
        # set gdf in self.data    
        self._data = gdf

    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Create model wavemakers lines.
        (old name: -)

        Adds model layers:

        * **wvm** geom: wavemakers lines

        Arguments
        ---------
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for wavemakers lines.
        merge: bool, optional
            If True, merge the new wavemakers lines with the existing ones. By default True.
        """
        # FIXME ensure the catalog is loaded before adding any new entries
        # self.data_catalog.sources TODO: check if still needed

        gdf = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, assert_gtype=None, **kwargs
        ).to_crs(self.model.crs)

        # make sure MultiLineString are converted to LineString
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)        

        self.set(gdf, merge)

        #TODO - add to config: self.model.config
        # self.model.config(f"{name}file", f"sfincs.{name}")
        # self.set_config(f"{name}file", f"sfincs.{name}")

    def add(self,
                   gdf: gpd.GeoDataFrame,
                   ):
        """Add multiple lines to wavemakers.
        
        Arguments
        ---------
        gdf: gpd.GeoDataFrame
            GeoDataFrame with locations and names of wavemakers lines to be added.
        **NOTE** - coordinates of wavemakers in GeoDataFrame need to be in the same CRS as SFINCS model.
        """        
        self.set(gdf, merge=True)

    def delete(self,
                   index: int, #FIXME - should this be List(int)?
                   ):
        """Remove (multiple) line(s) from wavemakers.
        
        Arguments
        ---------
        index: int
            Specify indices (int) of wavemakers(s) to be dropped from GeoDataFrame of wavemakers.
        """        
        if index.any() > (len(self.data.index)-1): #TODO - check if this is correct
            raise ValueError("One of the indices exceeds length of index range!")    
        
        self.data.drop(index).reset_index(drop=True)
        self.logger.info('Dropping line(s) from wavemakers')    

    def clear(self):
        """Clean GeoDataFrame with wavemakerss."""
        self.data  = gpd.GeoDataFrame()
    
#%% DDB GUI focused additional functions:
    # delete_polyline
    # list_names
    
    def delete_polyline(self, 
                    index: int,
                    ):
        """Remove single polyline from wavemakers.
        This function sets the wanted index, after which the generic delete function is called.
        
        Arguments
        ---------
        name_or_index: str, int
            Specify either name (str) or index (int) of cross-section to be dropped from GeoDataFrame.
        """                
        self.delete(index) #calls the generic delete function
        return
    
    def list_names(self):
        """Give list of names of cross sections."""
        names = list(self.data.name)
        return names