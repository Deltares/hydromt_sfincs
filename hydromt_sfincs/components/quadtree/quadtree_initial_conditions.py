import logging
from typing import TYPE_CHECKING, List, Union
from pathlib import Path

import geopandas as gpd
import numpy as np
import xugrid as xu

from hydromt.model.components import ModelComponent

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {"initial_conditions": {"standard_name": "initial water level", "unit": "m+ref"}}


class SfincsQuadtreeInitialConditions(ModelComponent):

    """SFINCS initial conditions component for quadtree grids."""
    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.quadtree_grid.data["ini"]
        super().__init__(
            model=model,
        )    

    @property
    def data(self):
        """Get the data from the model grid."""
        return self.model.quadtree_grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.quadtree_grid.mask

    def read(self):
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The mask values are read when the quadtree grid is read
        pass

    def write(self):
        # The mask values are written when the quadtree grid is written
        pass

    # %% core HydroMT-SFINCS functions:
    # read
    # write
    # set > already in grid.set()
    # create

    # Initial water level
    def create(
        self,
        ini: Union[str, Path, gpd.GeoDataFrame] = None,
        ini_buffer: int = 0, #FIXME - meter or pixels?
        reproj_method="average",
        reset_ini: bool = True,
    ):
        """Setup spatially varying initial water level (inifile).

        Adds model layers to SfincsModel.grid.data:

        * **ini** map: initial water level [m+ref]

        Parameters
        ----------
        ini : str, Path, RasterDataset or GeoDataFrame with 'ini' column
            Spatially varying initial water level [m+ref]
        ini_buffer: float, optional
            If larger than zero, extend the `ini` gdf geometry with a buffer [m], 
            by default 0.            
        reproj_method : str, optional
            Resampling method for reprojecting the initial water level data to the model grid.
            By default 'average'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        reset_ini: bool, optional
            If True (default), reset existing ini layer. If False updating existing ini layer.

        """

        mname = "ini"

        # Add logger info
        logger.info("Creating spatially varying initial water level.")

        # get initial water level data        
        if isinstance(ini, gpd.GeoDataFrame):            
            # input is a geodataframe with a value 'ini' to rasterize
            gdf_ini = self.data_catalog.get_geodataframe(
                ini, 
                bbox=self.model.bbox,
            )

            if ini_buffer > 0:  # NOTE assumes model in projected CRS!
                gdf_ini["geometry"] = gdf_ini.to_crs(self.model.crs).buffer(
                    ini_buffer
                )

            # Parse wanted value within polygon:
            inival = float(gdf_ini["ini"].unique())

            # if reset_ini = True start empty, otherwise start with existing ini layer
            if reset_ini:
                # start with empty ini layer
                da_ini = xu.full_like(
                    self.mask,
                    fill_value=np.nan,
                    dtype="float32",
                )
            else:
                # start with existing ini layer
                da_ini = self.data[mname]         

            # Burn the polygon into the raster: pixels inside polygon get 1, others get 0
            burned = xu.burn_vector_geometry(
                gdf_ini["geometry"], self.data, fill=0, all_touched=True
            )

            da_ini = xu.where(burned > 0, inival, da_ini)
            
        else:  
            # input is a rasterdataset/file with ini values
            if ini is not None:
                da_ini = self.data_catalog.get_rasterdataset(
                    ini,
                    bbox=self.model.bbox,
                    buffer=10,
                )

            # reproject initial water level data to model grid
            da_ini = da_ini.raster.mask_nodata()  # set nodata to nan
            da_ini = da_ini.raster.reproject_like(self.mask, method=reproj_method)

        # check on nan values
        if np.logical_and(np.isnan(da_ini), self.mask >= 1).any():
            logger.warning("NaN values found in initial water level data; filled with 0")
            da_ini = da_ini.fillna(0)

        # set grid
        da_ini.attrs.update(**_ATTRS.get(mname, {}))
        self.model.quadtree_grid.set(da_ini, name=mname)

        # update config: remove default zsini and set inifile
        self.model.config.set(f"nc{mname}file", f"sfincs_ini.nc")
        # set spatially uniform zsini to None in config
        self.model.config.set("zsini", None)