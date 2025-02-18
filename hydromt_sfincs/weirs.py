import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
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
        structures: Union[str, Path, gpd.GeoDataFrame],
        dep: Union[str, Path, xr.DataArray] = None,
        buffer: float = None,
        dz: float = None,        
        merge: bool = True,
        **kwargs,
    ):
        """Create model weir lines.
        (old name: setup_structures)

        If elevation at weir locations is not provided, it can be calculated from the model elevation (dep) plus dz.

        Adds model layers:

        * **weir** geom: weir lines

        Arguments
        ---------
        structures: str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for weir lines.
        dep : str, Path, xr.DataArray, optional
            Path, data source name, or xarray raster object ('elevtn') describing the depth in an
            alternative resolution which is used for sampling the weir.
        buffer : float, optional
            If provided, describes the distance from the centerline to the foot of the structure.
            This distance is supplied to the raster.sample as the window (wdw).            
        dz: float, optional
            If provided, for weir structures the z value is calculated from
            the model elevation (dep) plus dz.            
        merge: bool, optional
            If True, merge the new weir lines with the existing ones. By default True.
        """
        # FIXME ensure the catalog is loaded before adding any new entries
        # self.data_catalog.sources TODO: check if still needed

        gdf = self.data_catalog.get_geodataframe(
            structures, geom=self.model.region, assert_gtype=None, **kwargs
        ).to_crs(self.model.crs)

        # expected columns in gdf
        cols = {
            "weir": ["name", "z", "par1", "geometry"],
        }

        # keep relevant columns
        gdf = gdf[
            [c for c in cols["weir"] if c in gdf.columns]
        ]  

        # check if z values are provided or can be calculated
        if not "z" in gdf.columns and (dep is None and dz is None):
            raise ValueError("Weir structure requires z values, or 'dep' or 'dz' input to determine these on the fly.")
        elif (dep is not None or dz is not None):
            # determine elevation from dep and dz, if data parsed
            gdf = self.determine_weir_elevation(gdf, dep, buffer, dz)

        # make sure MultiLineString are converted to LineString
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)        

        self.set(gdf, merge)

        #TODO - add to config: self.model.config
        # self.model.config(f"{name}file", f"sfincs.{name}")
        # self.set_config(f"{name}file", f"sfincs.{name}")

    def add(
        self,
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

    def delete(
        self,
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

#%% HydroMT-SFINCS focused additional functions:
    # determine_weir_elevation
    
    def determine_weir_elevation( #FIXME - should this be in utils.py or not?
        self, 
        gdf: gpd.GeoDataFrame,
        dep: Union[str, Path, xr.DataArray] = None,
        buffer: float = None,
        dz: float = None,            
        ):
        """Determine z values for weir structures."""
        # taken from old 'sfincs.py'>setup_structures function

        structs = utils.gdf2linestring(gdf)  # check if it parsed correct

        # get elevation data either from model itself, or separate input
        if dep is None or dep == "dep":
            assert "dep" in self.model.grid, "dep layer not found"
            elv = self.model.grid["dep"]
        else:
            elv = self.data_catalog.get_rasterdataset(
                dep, geom=self.region, buffer=5, variables=["elevtn"]
            )

        # calculate window size from buffer
        if buffer is not None:
            res = abs(elv.raster.res[0])
            if elv.raster.crs.is_geographic:
                res = res * 111111.0
            window_size = int(np.ceil(buffer / res))
        else:
            window_size = 0
        self.logger.debug(f"Sampling elevation with window size {window_size}")

        # interpolate dep data to points of weirs
        structs_out = []
        for s in structs:
            pnts = gpd.points_from_xy(x=s["x"], y=s["y"])
            zb = elv.raster.sample(
                gpd.GeoDataFrame(geometry=pnts, crs=self.crs), wdw=window_size
            )
            if zb.ndim > 1:
                zb = zb.max(axis=1)

            s["z"] = zb.values

            # in case of dz, add this to the elevation
            if dz is not None:
                s["z"] += float(dz)

            structs_out.append(s)

        gdf = utils.linestring2gdf(structs_out, crs=self.crs)
    
        return gdf

#%% DDB GUI focused additional functions:
    # snap_to_grid
    # list_names

    def snap_to_grid(self):
        snap_gdf = self.model.grid.snap_to_grid(self.gdf) #FIXME - snap_to_grid should be function in grid.py!
        return snap_gdf
    
    def list_names(self):
        """Give list of names of cross sections."""
        names = list(self.data.name)
        return names