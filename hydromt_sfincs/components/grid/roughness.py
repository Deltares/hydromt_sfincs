import logging
from typing import TYPE_CHECKING, List

import numpy as np
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import workflows

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {"manning": {"standard_name": "manning roughness", "unit": "s.m-1/3"}}


class SfincsRoughness(ModelComponent):
    """SFINCS roughness component."""

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.grid.data["manning"]
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

    # %% core HydroMT-SFINCS functions:
    # read
    # write
    # set > already in grid.set()
    # create
    # clear >TODO ?

    def read(self):
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The manning file is read when all grid files are read in regulargrid.py
        pass

    def write(self):
        # The manning file is written when all grid files are written in regulargrid.py
        pass

    # Roughness
    @hydromt_step
    def create(
        self,
        datasets_rgh: List[dict] = [],
        manning_land=0.04,
        manning_sea=0.02,
        rgh_lev_land=0,
    ):
        """Setup model manning roughness map (manningfile) from gridded manning data or a combinataion of gridded
        land-use/land-cover map and manning roughness mapping table.

        Adds model layers:

        * **man** map: manning roughness coefficient [s.m-1/3]

        Parameters
        ---------
        datasets_rgh : List[dict], optional
            List of dictionaries with Manning's n datasets. Each dictionary should at least contain one of the following:
            * (1) manning: filename (or Path) of gridded data with manning values
            * (2) lulc (and reclass_table) :a combination of a filename of gridded landuse/landcover and a mapping table.
            In additon, optional merge arguments can be provided e.g.: merge_method, gdf_valid_fn
        manning_land, manning_sea : float, optional
            Constant manning roughness values for land and sea, by default 0.04 and 0.02 s.m-1/3
            Note that these values are only used when no Manning's n datasets are provided, or to fill the nodata values
        rgh_lev_land : float, optional
            Elevation level to distinguish land and sea roughness (when using manning_land and manning_sea), by default 0.0
        """

        if len(datasets_rgh) > 0:
            datasets_rgh = self.model._parse_datasets_rgh(datasets_rgh)
        else:
            datasets_rgh = []

        # fromdep keeps track of whether any manning values should be based on the depth or not
        fromdep = len(datasets_rgh) == 0
        if len(datasets_rgh) > 0:
            da_man = workflows.merge_multi_dataarrays(
                da_list=datasets_rgh,
                da_like=self.mask,
                interp_method="linear",
                logger=logger,
            )
            fromdep = np.isnan(da_man).where(self.mask > 0, False).any()
        if "dep" in self.data and fromdep:
            da_man0 = xr.where(
                self.data["dep"] >= rgh_lev_land, manning_land, manning_sea
            )
        elif fromdep:
            da_man0 = xr.full_like(self.mask, manning_land, dtype=np.float32)

        if len(datasets_rgh) > 0 and fromdep:
            logger.warning("nan values in manning roughness array")
            da_man = da_man.where(~np.isnan(da_man), da_man0)
        elif fromdep:
            da_man = da_man0
        da_man.raster.set_nodata(-9999.0)

        # set grid
        mname = "manning"
        da_man.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_man, name=mname)
        # set file name in config
        self.model.config.set(f"{mname}file", f"sfincs.{mname[:3]}")
