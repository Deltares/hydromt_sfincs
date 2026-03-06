import logging
from typing import TYPE_CHECKING, List

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import MeshComponent

from hydromt_sfincs.utils import make_regular_grid
from hydromt_sfincs.workflows.merge import (
    merge_multi_dataarrays,
)

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsQuadtreeRoughness(MeshComponent):
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

        nlev = self.data.attrs["nr_levels"]
        xy = self.data.grid.face_coordinates
        nr_cells = len(xy)
        manning = np.full(nr_cells, np.nan)
        dx = self.data.attrs["dx"]
        dy = self.data.attrs["dy"]
        res = min(dx, dy)

        if self.model.crs.is_geographic:
            res *= 111111.0  # convert to meters

        # 0-based level indices
        level = self.data["level"].values - 1

        # Precompute index slices per level
        level_indices = [np.where(level == ilev)[0] for ilev in range(nlev)]

        # Precompute roughness sets per level
        roughness_list = self.model._parse_roughness_list(roughness_list)

        # get m and n indices
        n = self.data["n"] - 1  # 0-based
        m = self.data["m"] - 1  # 0-based

        def process_level(ilev):
            idx = level_indices[ilev]
            xz, yz = xy[idx, 0], xy[idx, 1]
            n_level, m_level = n[idx], m[idx]
            z_level = self.data["z"][idx] if "z" in self.data else None
            mask_level = self.mask[idx]
            dxmin, dymin = dx / 2**ilev, dy / 2**ilev

            logger.info(f"Processing roughness level {ilev + 1} of {nlev} ...")

            # Determine chunking
            x_min, x_max = xz.min() - dxmin, xz.max() + dxmin
            y_min, y_max = yz.min() - dymin, yz.max() + dymin
            x_chunks = np.arange(x_min, x_max, nrmax * dxmin)
            y_chunks = np.arange(y_min, y_max, nrmax * dymin)

            mgl = np.full(len(idx), np.nan)

            def process_chunk(ix, iy):
                if ix < len(x_chunks) - 1:
                    x0, x1 = x_chunks[ix], x_chunks[ix + 1]
                else:
                    x0, x1 = x_chunks[ix], x_max

                if iy < len(y_chunks) - 1:
                    y0, y1 = y_chunks[iy], y_chunks[iy + 1]
                else:
                    y0, y1 = y_chunks[iy], y_max

                in_chunk = np.where((xz >= x0) & (xz < x1) & (yz >= y0) & (yz < y1))[0]
                if len(in_chunk) == 0:
                    return

                da_like = make_regular_grid(
                    x0=self.data.attrs["x0"],
                    y0=self.data.attrs["y0"],
                    dx=dxmin,
                    dy=dymin,
                    mmax=m_level[in_chunk].max().values + 1,
                    nmax=n_level[in_chunk].max().values + 1,
                    rotation=self.data.attrs["rotation"],
                    crs=self.model.crs,
                    mmin=m_level[in_chunk].min().values,
                    nmin=n_level[in_chunk].min().values,
                    make_ugrid=False,
                )

                if len(roughness_list) > 0:
                    da_man = merge_multi_dataarrays(
                        da_list=roughness_list,
                        da_like=da_like,
                        interp_method="linear",
                        logger=logger,
                    )
                else:
                    da_man = xr.full_like(da_like, np.nan, dtype=np.float32)

                # Get the indices of the "active" unstructured grid points in the regular grid
                idx_y = np.searchsorted(da_like.n.values, n_level[in_chunk].values)
                idx_x = np.searchsorted(da_like.m.values, m_level[in_chunk].values)
                mgl[in_chunk] = da_man.values[idx_y, idx_x]

            # Parallel or sequential chunk processing
            if len(x_chunks) > 1 or len(y_chunks) > 1:
                logger.info(
                    f"Processing in {len(x_chunks)} x {len(y_chunks)} chunks ..."
                )
                for ix in range(len(x_chunks)):
                    for iy in range(len(y_chunks)):
                        process_chunk(ix, iy)
            else:
                process_chunk(0, 0)

            # Clip values on zmin and zmax and return
            return idx, mgl

        # Loop over levels
        for ilev in range(nlev):
            idx, mgl_level = process_level(ilev)
            manning[idx] = mgl_level

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

        # Set manning values in self.data
        self.data["manning"] = xu.UgridDataArray(
            xr.DataArray(data=manning, dims=[self.data.grid.face_dimension]),
            self.data.grid,
        )

        # Set netcdf manning to config
        self.model.config.set("manningfile", "roughness.nc")
