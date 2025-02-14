import geopandas as gpd
import shapely
import pandas as pd
from pathlib import Path
from typing import Union

from hydromt.model.components import ModelComponent
from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

class SfincsWeirs(ModelComponent):
    def __init__(
        self,        
        model: SfincsModel,
    ):
        self._filename: str = "sfincs.weir"
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model, 
        )    

    @property
    def data(self) -> pd.GeoDataFrame:
        """Weirs lines data.

        Return geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize weir lines."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()#FIXME - right?
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self):
        """Read in all weir lines."""
        # Read input file:
        struct = utils.read_geoms(self._filename) #=utils.py function
        gdf = utils.linestring2gdf(struct, crs=self.model.crs) #=utils.py function
                  
        self.set(gdf, merge=False) # Add to self._data  

    def write(self, filename=None): #TODO - TL: filename=None - still needed?
        """Write weirfile."""
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
        utils.write_geoms(self._filename, struct, stype="weir", fmt=fmt) #=utils.py function

        # TODO - write also as geojson - TL: at what level do we want to do that?
        # if self._write_gis:
        #     self.write_vector(variables=["geoms"])

    def set(
            self,
            gdf: gpd.GeoDataFrame,
            merge: bool = True
    ):
        """Set weir lines.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with weir lines to self.data
        name: str
            Geometry name.
        """        
        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Weirs must be of type LineString.")

        # Clip points outside of model region:
        within = gdf.within(self.model.region) # same as 'inpolygon' function
        # FYI - An object is said to be within `region` if at least one of its points is located
        # in the `interior` and no points are located in the `exterior` of the other.

        if within.all() == False:
            raise ValueError("None of weirs fall within model domain.")
        elif within.any() == False:
            gdf = gdf[~within]
            self.logger.info("Some of weirs fall out of model domain. Removing lines.")

        if merge and self.data is not None:
            gdf0 = self.data
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf0], ignore_index=True))
            self.logger.info("Adding new weirs to existing ones.")
        
        # set gdf in self.data    
        self._data = gdf

    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ):
        """Create model weir lines.
        (old name: setup_structures)

        Adds model layers:

        * **weir** geom: weir lines

        Arguments
        ---------
        locations: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for weir lines.
        merge: bool, optional
            If True, merge the new weir lines with the existing ones. By default True.
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
        """Add multiple lines to weirs.
        
        Arguments
        ---------
        gdf: gpd.GeoDataFrame
            GeoDataFrame with locations and names of weir lines to be added.
        **NOTE** - coordinates of weirs in GeoDataFrame need to be in the same CRS as SFINCS model.
        """        
        self.set(gdf, merge=True)

    def delete(self,
                   index: int, #FIXME - should this be List(int)?
                   ):
        """Remove (multiple) line(s) from weirs.
        
        Arguments
        ---------
        index: int
            Specify indices (int) of weir(s) to be dropped from GeoDataFrame of weirs.
        """        
        if index.any() > (len(self.data.index)-1): #TODO - check if this is correct
            raise ValueError("One of the indices exceeds length of index range!")    
        
        self.data.drop(index).reset_index(drop=True)
        self.logger.info('Dropping line(s) from weirs')    

    def clear(self):
        """Clean GeoDataFrame with weirs."""
        self.data  = gpd.GeoDataFrame()

    def snap_to_grid(self):
        snap_gdf = self.model.grid.snap_to_grid(self.gdf) #FIXME - snap_to_grid should be function in grid.py!
        return snap_gdf
    
    def list_names(self):
        """Give list of names of cross sections."""
        names = list(self.data.name)
        return names