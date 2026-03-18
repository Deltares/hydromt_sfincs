import logging
import gc
import time
from typing import TYPE_CHECKING, List

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs.workflows.merge import merge_multi_dataarrays
from hydromt_sfincs.components.quadtree import SfincsQuadtreeMixin

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsQuadtreeElevation(SfincsQuadtreeMixin, ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the elevation is stored in the model.quadtree_grid.data["z"]
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
        elevation_list: List[dict],
        nrmax: int = 2000,
        buffer_cells: int = 0,
        interp_method: str = "linear",
        zmin: float = -1.0e9,
        zmax: float = 1.0e9,
        bathymetry_database: object = None,
    ):
        """Interpolate topobathy (z) data to the model grid.

        Adds model grid layers:

        * **z**: combined elevation/bathymetry [m+ref]

        Parameters
        ----------
        elevation_list : List[dict]
            List of dictionaries with topobathy data, each containing a dataset name or Path (elevation) and optional merge arguments e.g.:
            [{'elevation': merit_hydro, 'zmin': 0.01}, {'elevation': gebco, 'offset': 0, 'merge_method': 'first', 'reproj_method': 'bilinear'}]
            For a complete overview of all merge options, see :py:func:`hydromt.workflows.merge_multi_dataarrays`
        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels, by default 0
        interp_method : str, optional
            Interpolation method used to fill the buffer cells, by default "linear"
        """

        nlev = self.data.attrs["nr_levels"]
        xy = self.data.grid.face_coordinates
        n_cells = self.data.grid.n_face
        zz = np.full(n_cells, np.nan)
        dx = self.data.attrs["dx"]
        dy = self.data.attrs["dy"]
        res = min(dx, dy)

        if self.model.crs.is_geographic:
            res *= 111111.0  # convert to meters

        # get 0-based level
        level = self.data["level"].values - 1

        # Precompute elevation sets per level
        # Add try statement here for compatibility with cht_bathymetry approach
        try:
            elevation_list_per_level = [
                self.model._parse_datasets_elevation(elevation_list, res=res / (2**ilev))
                for ilev in range(nlev)
            ]
        except Exception as e:
            print("Using bathymetry database for interpolation.")

        if bathymetry_database is None:
            # Generic workflow using compute_quadtree
            def compute_elevation(da_like, ilev=None):
                da_dep = merge_multi_dataarrays(
                    da_list=elevation_list_per_level[ilev],
                    da_like=da_like,
                    buffer_cells=buffer_cells,
                    interp_method=interp_method,
                    logger=logger,
                )

                # check if no nan data is present in the bed levels
                nmissing = int(np.sum(np.isnan(da_dep.values)))
                if nmissing > 0:
                    logger.warning(f"Interpolate elevation at {nmissing} cells")
                    da_dep = da_dep.raster.interpolate_na(
                        method="rio_idw", extrapolate=True
                    )
                return da_dep

            self.compute_quadtree(
                compute_elevation,
                zz,
                nrmax=nrmax,
                clip=(zmin, zmax),
            )
        else:
            # TODO remove code when ddb also works with data_catalog
            level_indices = [np.where(level == ilev)[0] for ilev in range(nlev)]
            for ilev in range(nlev):
                idx = level_indices[ilev]
                xz, yz = xy[idx, 0], xy[idx, 1]
                dxmin, dymin = dx / 2**ilev, dy / 2**ilev

                # Determine chunking
                x_min, x_max = xz.min() - dxmin, xz.max() + dxmin
                y_min, y_max = yz.min() - dymin, yz.max() + dymin
                x_chunks = np.arange(x_min, x_max, nrmax * dxmin)
                y_chunks = np.arange(y_min, y_max, nrmax * dymin)

                zgl = np.full(len(idx), np.nan)

                def process_chunk(ix, iy):
                    if ix < len(x_chunks) - 1:
                        x0, x1 = x_chunks[ix], x_chunks[ix + 1]
                    else:
                        x0, x1 = x_chunks[ix], x_max
                    if iy < len(y_chunks) - 1:
                        y0, y1 = y_chunks[iy], y_chunks[iy + 1]
                    else:
                        y0, y1 = y_chunks[iy], y_max

                    in_chunk = np.where(
                        (xz >= x0) & (xz < x1) & (yz >= y0) & (yz < y1)
                    )[0]
                    if len(in_chunk) == 0:
                        return

                    zgl[in_chunk] = bathymetry_database.get_bathymetry_on_points(
                        xz[in_chunk],
                        yz[in_chunk],
                        min(dxmin, dymin),
                        self.model.crs,
                        elevation_list,
                    )

                for ix in range(len(x_chunks)):
                    for iy in range(len(y_chunks)):
                        process_chunk(ix, iy)

                zz[idx] = zgl

        # Convert elevation to ugrid-dataarray and set in self.data
        da = xr.DataArray(zz, dims=[self.data.grid.face_dimension])
        uda = xu.UgridDataArray(da, self.data.grid)
        self.model.quadtree_grid.set(uda, name="z")

    @hydromt_step
    def create_uniform(self, zb):
        self.data["z"][:] = zb

    def interpolate_bathymetry(self, x, y, z, method="linear"):
        """x, y, and z are numpy arrays with coordinates and bathymetry values"""
        xy = self.data.grid.face_coordinates
        # zz = np.full(self.nr_cells, np.nan)
        xz = xy[:, 0]
        yz = xy[:, 1]
        zz = interp2(x, y, z, xz, yz, method=method)
        ugrid2d = self.data.grid
        self.data["z"] = xu.UgridDataArray(
            xr.DataArray(data=zz, dims=[ugrid2d.face_dimension]), ugrid2d
        )


def interp2(x0, y0, z0, x1, y1, method="linear"):
    f = RegularGridInterpolator(
        (y0, x0), z0, bounds_error=False, fill_value=np.nan, method=method
    )
    # reshape x1 and y1
    if x1.ndim > 1:
        sz = x1.shape
        x1 = x1.reshape(sz[0] * sz[1])
        y1 = y1.reshape(sz[0] * sz[1])
        # interpolate
        z1 = f((y1, x1)).reshape(sz)
    else:
        z1 = f((y1, x1))

    return z1
