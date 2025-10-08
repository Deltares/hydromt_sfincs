import logging
import gc
import time
from typing import TYPE_CHECKING, List

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import MeshComponent

from hydromt_sfincs.utils import make_regular_grid, partition_quadtree
from hydromt_sfincs.workflows.merge import (
    merge_multi_dataarrays,
    merge_multi_dataarrays_on_mesh,
)

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsQuadtreeElevation(MeshComponent):
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
        elevation_sets: List[dict],
        partition_by_level: bool = True,
        partition_in_blocks: bool = False,
        nrmax: int = 2000,  # not in list
        buffer_cells: int = 0,  # not in list
        interp_method: str = "linear",  # used for buffer cells only
    ):
        """Interpolate topobathy (dep) data to the model grid.

        Adds model grid layers:

        * **dep**: combined elevation/bathymetry [m+ref]

        Parameters
        ----------
        elevation_sets : List[dict]
            List of dictionaries with topobathy data, each containing a dataset name or Path (elevation) and optional merge arguments e.g.:
            [{'elevation': merit_hydro, 'zmin': 0.01}, {'elevation': gebco, 'offset': 0, 'merge_method': 'first', 'reproj_method': 'bilinear'}]
            For a complete overview of all merge options, see :py:func:`hydromt.workflows.merge_multi_dataarrays`
        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels, by default 0
        interp_method : str, optional
            Interpolation method used to fill the buffer cells , by default "linear"
        """

        # get resolution and number of level
        res = self.data.attrs["dx"]
        nrlevels = self.data.attrs["nr_levels"]

        # convert to meters if geographic
        if self.model.crs.is_geographic:
            res = res * 111111.0
        # append parsed datasets per level
        elevation_sets_per_level = []
        for ilev in range(nrlevels):
            # compute resolution at level
            res_level = res / (2**ilev)
            elevation_sets_per_level.append(
                self.model._parse_datasets_elevation(elevation_sets, res=res_level)
            )

        # check if partitions are already defined
        if partition_by_level or partition_in_blocks:
            # create partitions if not already done
            if not hasattr(self, "partitions"):
                t0 = time.time()
                self.partitions = partition_quadtree(
                    quadtree=self.data,
                    partition_by_level=partition_by_level,
                    partition_in_blocks=partition_in_blocks,
                    nrmax=nrmax,
                    logger=logger,
                )
                t1 = time.time()
                logger.debug(f"Partitioning quadtree took {t1-t0:.2f} s.")
            count = 0
            for partition in self.partitions:
                # time each partition
                t0 = time.time()
                ilev = partition.level.max().values - 1
                # merge multiple datasets on mesh
                uda = merge_multi_dataarrays_on_mesh(
                    da_list=elevation_sets_per_level[ilev],
                    mesh2d=partition.grid,
                )
                partition["z"] = uda
                t1 = time.time()
                logger.debug(
                    f"Dep merged for partition {count} at level {ilev} in {t1-t0:.2f} s."
                )
                count += 1
                gc.collect()
            t0 = time.time()
            merged = xu.merge_partitions(self.partitions)
            t1 = time.time()
            logger.debug(f"Merging partitions took {t1-t0:.2f} s.")
            reordered = merged.ugrid.reindex_like(self.data.grid)
            # add data to grid
            # FIXME
            # self.model.quadtree_grid.set(reordered, name="z")
            self.data["z"] = reordered["z"]
        else:
            t0 = time.time()
            # when not partitioned, use the full grid with the highest resolution data
            uda = merge_multi_dataarrays_on_mesh(
                da_list=elevation_sets_per_level[-1], mesh2d=self.data.grid, logger=logger
            )
            t1 = time.time()
            logger.debug(f"Merging dep took {t1-t0:.2f} s.")
            # add data to grid
            # FIXME
            # self.model.quadtree_grid.set(uda, name="z")
            self.data["z"] = uda

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

    def set_uniform_bathymetry(self, zb):
        self.data["z"][:] = zb

    def set_bathymetry(self, elevation_sets, zmin=-1.0e9, zmax=1.0e9, quiet=True):
        # Number of refinement levels
        nlev = self.data.attrs["nr_levels"]
        # Cell centre coordinates
        xy = self.data.grid.face_coordinates
        # Get number of cells
        nr_cells = len(xy)
        # Initialize bathymetry array
        zz = np.full(nr_cells, np.nan)
        # cell size of coarsest level
        dx = self.data.attrs["dx"]

        # Determine first indices and number of cells per refinement level
        # This is also done when the grid is built, but that information is not stored
        ifirst = np.zeros(nlev, dtype=int)
        ilast = np.zeros(nlev, dtype=int)
        level = self.data["level"].values[:] - 1  # 0-based
        for ilev in range(0, nlev):
            # Find index of first cell with this level
            ifirst[ilev] = np.where(level == ilev)[0][0]
            # Find index of last cell with this level
            if ilev < nlev - 1:
                ilast[ilev] = np.where(level == ilev + 1)[0][0] - 1
            else:
                ilast[ilev] = nr_cells - 1

        # convert to meters if geographic
        if self.model.crs.is_geographic:
            dx = dx * 111111.0
        # append parsed datasets per level
        elevation_sets_per_level = []
        for ilev in range(nlev):
            # compute resolution at level
            res_level = dx / (2**ilev)
            elevation_sets_per_level.append(
                self.model._parse_datasets_elevation(elevation_sets, res=res_level)
            )

        # get m and n indices
        n = self.data["n"]
        m = self.data["m"]

        # Loop through all levels
        for ilev in range(nlev):
            if not quiet:
                print(
                    "Processing bathymetry level "
                    + str(ilev + 1)
                    + " of "
                    + str(nlev)
                    + " ..."
                )

            # First and last cell indices in this level
            i0 = ifirst[ilev]
            i1 = ilast[ilev]

            # Make blocks of cells in this level only
            cell_indices_in_level = np.arange(i0, i1 + 1, dtype=int)

            xz = xy[cell_indices_in_level, 0]
            yz = xy[cell_indices_in_level, 1]
            dxmin = dx / 2**ilev

            da_like = make_regular_grid(
                x0=self.data.attrs["x0"],
                y0=self.data.attrs["y0"],
                dx=dxmin,
                dy=dxmin,
                mmax=m[i0 : i1 + 1].max().values + 1,
                nmax=n[i0 : i1 + 1].max().values + 1,
                rotation=self.data.attrs["rotation"],
                crs=self.model.crs,
                mmin=m[i0 : i1 + 1].min().values,
                nmin=n[i0 : i1 + 1].min().values,
                make_ugrid=False,
            )

            da_dep = merge_multi_dataarrays(
                da_list=elevation_sets_per_level[ilev],
                da_like=da_like,
                # buffer_cells=buffer_cells,
                # interp_method=interp_method,
                logger=logger,
            )

            # Flatten n and m indices of cells in this level
            n_flat = n[cell_indices_in_level].values
            m_flat = m[cell_indices_in_level].values

            # Find integer indices along the coordinate arrays
            idx_y = np.searchsorted(da_dep.n.values, n_flat)
            idx_x = np.searchsorted(da_dep.m.values, m_flat)

            # Select the values
            zgl = da_dep.values[idx_y, idx_x]

            # zgl = bathymetry_database.get_bathymetry_on_points(xz,
            #                                                    yz,
            #                                                    dxmin,
            #                                                    self.model.crs,
            #                                                    bathymetry_sets)

            # Limit zgl to zmin and zmax
            zgl = np.maximum(zgl, zmin)
            zgl = np.minimum(zgl, zmax)

            zz[cell_indices_in_level] = zgl

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
