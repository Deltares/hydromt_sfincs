import logging
from typing import TYPE_CHECKING, List

import numpy as np
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs.workflows.merge import merge_multi_dataarrays
from hydromt_sfincs.components.quadtree import SfincsQuadtreeMixin

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsQuadtreeRoughness(SfincsQuadtreeMixin, ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the roughness is stored in the model.quadtree_grid.data["manning"]
        super().__init__(
            model=model,
        )

    @property
    def data(self):
        """Get the data from the quadtree grid."""
        return self.model.quadtree_grid.data

    @property
    def mask(self):
        """Get the mask from the quadtree grid."""
        return self.model.quadtree_mask.data["mask"]

    def read(self):
        # The mask elevation are read when the quadtree grid is read
        pass

    def write(self):
        # The mask elevation are written when the quadtree grid is written
        pass

    @hydromt_step
    def create(
        self,
        roughness_list: List[dict],
        manning_land: float = 0.05,
        manning_sea: float = 0.02,
        rgh_lev_land: int = 0,
        nrmax: int = 2000,
    ):
        """Interpolate roughness (Manning's n) data to the model grid.

        Adds model grid layers:

        * **manning**: combined roughness [s/m^(1/3)]

        Parameters
        ---------
        roughness_list : List[dict], optional
            List of dictionaries with Manning's n datasets. Each dictionary should at least contain one of the following:
            * (1) manning: filename (or Path) of gridded data with manning values
            * (2) lulc (and reclass_table) :a combination of a filename of gridded landuse/landcover and a mapping table.
            In additon, optional merge arguments can be provided e.g.: merge_method, gdf_valid_fn
        manning_land, manning_sea : float, optional
            Constant manning roughness values for land and sea, by default 0.04 and 0.02 s.m-1/3
            Note that these values are only used when no Manning's n datasets are provided, or to fill the nodata values
        rgh_lev_land : float, optional
            Elevation level to distinguish land and sea roughness (when using manning_land and manning_sea), by default 0.0
        nrmax : int, optional
            Maximum number of points to interpolate in a single chunk, by default 2000.
        """

        n_cells = self.data.grid.n_face
        manning = np.full(n_cells, np.nan)

        # Parse roughness list and prepare for interpolation
        roughness_list = self.model._parse_roughness_list(roughness_list)

        # Define the function to compute roughness for a chunk of the quadtree grid
        def compute_roughness(da_like, ilev=None):
            if len(roughness_list) > 0:
                da_man = merge_multi_dataarrays(
                    da_list=roughness_list,
                    da_like=da_like,
                    interp_method="linear",
                    logger=logger,
                )
            else:
                da_man = xr.full_like(da_like, np.nan, dtype=np.float32)
            return da_man

        # Apply the roughness computation in chunks over the quadtree grid
        self.compute_quadtree(compute_roughness, manning, nrmax=nrmax)

        # Now fill any remaining nans with depth-based manning
        nr_nan = np.isnan(manning).sum()
        if "z" in self.data:
            logger.info(
                f"Filled {nr_nan} remaining cells in manning with depth-based roughness."
            )
            manning0 = xr.where(
                self.data["z"] >= rgh_lev_land, manning_land, manning_sea
            )
            manning = np.where(np.isnan(manning), manning0, manning)
        else:
            logger.info(
                f"Filled {nr_nan} remaining cells in manning with manning sea roughness, "
                f"since no elevation information is available."
            )
            manning = np.where(np.isnan(manning), manning_sea, manning)

        # Convert manning to ugrid-dataarray and set in self.data
        da = xr.DataArray(manning, dims=[self.data.grid.face_dimension])
        uda = xu.UgridDataArray(da, self.data.grid)
        self.model.quadtree_grid.set(uda, name="manning")

        # Set netcdf manning to config
        self.model.config.set("manningfile", "roughness.nc")
