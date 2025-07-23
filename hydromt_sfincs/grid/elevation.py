import logging
from typing import TYPE_CHECKING, List

import numpy as np
import xarray as xr

from hydromt.model.components import ModelComponent
from hydromt_sfincs import workflows

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {"dep": {"standard_name": "elevation", "unit": "m+ref"}}


class SfincsElevation(ModelComponent):
    """SFINCS elevation component."""

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.grid.data["mask"]
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
        # The mask values are read when the quadtree grid is read
        pass

    def write(self):
        # The mask values are written when the quadtree grid is written
        pass

    def create(
        self,
        datasets_dep: List[dict],
        buffer_cells: int = 0,  # not in list
        interp_method: str = "linear",  # used for buffer cells only
    ):
        """Interpolate topobathy (dep) data to the model grid.

        Adds model grid layers:

        * **dep**: combined elevation/bathymetry [m+ref]

        Parameters
        ----------
        datasets_dep : List[dict]
            List of dictionaries with topobathy data, each containing a dataset name or Path (elevtn) and optional merge arguments e.g.:
            [{'elevtn': merit_hydro, 'zmin': 0.01}, {'elevtn': gebco, 'offset': 0, 'merge_method': 'first', 'reproj_method': 'bilinear'}]
            For a complete overview of all merge options, see :py:func:`hydromt.workflows.merge_multi_dataarrays`
        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels, by default 0
        interp_method : str, optional
            Interpolation method used to fill the buffer cells , by default "linear"
        """

        # retrieve model resolution to determine zoom level for xyz-datasets
        if not self.model.grid.crs.is_geographic:
            res = np.abs(self.mask.raster.res[0])
        else:
            res = np.abs(self.mask.raster.res[0]) * 111111.0

        datasets_dep = self.model._parse_datasets_dep(datasets_dep, res=res)

        da_dep = workflows.merge_multi_dataarrays(
            da_list=datasets_dep,
            da_like=self.mask,
            buffer_cells=buffer_cells,
            interp_method=interp_method,
            logger=logger,
        )

        # check if no nan data is present in the bed levels
        nmissing = int(np.sum(np.isnan(da_dep.values)))
        if nmissing > 0:
            logger.warning(f"Interpolate elevation at {nmissing} cells")
            da_dep = da_dep.raster.interpolate_na(method="rio_idw", extrapolate=True)

        # set the dep layer in the model data
        mname = "dep"
        da_dep.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_dep, name=mname)

        # TODO add to config, or is that only done when writing?
        self.model.config.set("depfile", "sfincs.dep")
