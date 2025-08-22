import logging
from typing import TYPE_CHECKING, List

import numpy as np
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {"initial_conditions": {"standard_name": "initial water level", "unit": "m+ref"}}


class SfincsInitialConditions(ModelComponent):

    """SFINCS initial conditions component."""
    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.grid.data["ini"]
        super().__init__(
            model=model,
        )    

    @property
    def data(self):
        """Get the data from the model grid."""
        return self.model.grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.grid.mask

    def read(self):
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The ini file is read when all grid files are read in regulargrid.py
        pass

    def write(self):
        # The ini file is written when all grid files are written in regulargrid.py
        pass

    # Original HydroMT-SFINCS setup_ functions:
    # not yet implemented

    #FIXME, so only for inifile? No config manipulation needed for zsini and/or rstfile/dtrstout/rstout?

    # %% core HydroMT-SFINCS functions:
    # read
    # write
    # set > already in grid.set()
    # create
    # clear >TODO?

    # Initial water level
    @hydromt_step
    def create(
        self,
        ini= None,
        reproj_method="average",
    ):
        """Setup spatially varying initial water level (inifile).

        Adds model layers to SfincsModel.grid.data:

        * **ini** map: initial water level [m+ref]

        Parameters
        ----------
        ini : str, Path, or RasterDataset
            Spatially varying initial water level [m+ref]
        reproj_method : str, optional
            Resampling method for reprojecting the initial water level data to the model grid.
            By default 'average'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        """

        # Add logger info
        logger.info("Creating spatially varying initial water level.")

        # get initial water level data
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
        da_ini.raster.set_nodata(-9999.0)

        # set grid
        mname = "ini"
        da_ini.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_ini, name=mname)

        # update config: remove default inf and set qinf map
        self.model.config.set(f"{mname}file", f"sfincs.{mname}")
        # set spatially uniform zsini to None in config
        self.model.config.set("zsini", None)

# %% DDB GUI focused additional functions:
# interpolate >FIXME > not needed?