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

from hydromt_sfincs.workflows.map_overlay import ElevationOverlay
from hydromt_sfincs.workflows.merge import (
    merge_multi_dataarrays,
)
from hydromt_sfincs.components.quadtree import SfincsQuadtreeMixin

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsQuadtreeElevation(SfincsQuadtreeMixin, ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The elevation data lives on model.quadtree_grid.data["z"];
        # the renderer holds the only local state.
        super().__init__(
            model=model,
        )
        self._overlay = ElevationOverlay()

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
        fill_missing: bool = True,
        zmin: float = -1.0e9,
        zmax: float = 1.0e9,
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
        n_cells = self.data.grid.n_face
        zz = np.full(n_cells, np.nan)
        dx = self.data.attrs["dx"]
        dy = self.data.attrs["dy"]
        res = min(dx, dy)

        if self.model.crs.is_geographic:
            res *= 111111.0  # convert to meters

        # Precompute elevation sets per level
        # Add try statement here for compatibility with cht_bathymetry approach
        elevation_list_per_level = [
            self.model._parse_datasets_elevation(elevation_list, res=res / (2**ilev))
            for ilev in range(nlev)
        ]

        # Generic workflow using compute_quadtree; fill_missing is taken
        # from the enclosing create() call
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
            if nmissing > 0 and fill_missing:
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

        # Convert elevation to ugrid-dataarray and set in self.data
        da = xr.DataArray(zz, dims=[self.data.grid.face_dimension])
        uda = xu.UgridDataArray(da, self.data.grid)
        self.model.quadtree_grid.set(uda, name="z", overwrite_grid=True)

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

    # ------------------------------------------------------------------
    # Map overlay (delegates to ElevationOverlay in workflows.map_overlay)
    # ------------------------------------------------------------------

    def clear_overlay(self) -> None:
        """Invalidate the cached elevation-overlay trimesh."""
        self._overlay.invalidate()

    def map_overlay(
        self,
        file_name,
        xlim=None,
        ylim=None,
        cmap="gist_earth",
        cmin=None,
        cmax=None,
        width: int = 800,
        **kwargs,
    ) -> bool:
        """Render a PNG elevation overlay.

        One-line wrapper around
        :py:class:`hydromt_sfincs.workflows.map_overlay.ElevationOverlay`.
        """
        if self.data is None or "z" not in self.data:
            return False
        return self._overlay.render(
            face_xy=self.data.grid.face_coordinates,
            z=self.data["z"].values[:],
            level=self.data["level"].values[:],
            dx0=self.data.attrs["dx"],
            dy0=self.data.attrs["dy"],
            rotation=self.data.attrs["rotation"],
            source_crs=self.model.crs,
            file_name=file_name,
            xlim=xlim,
            ylim=ylim,
            cmap=cmap,
            cmin=cmin,
            cmax=cmax,
            width=width,
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
