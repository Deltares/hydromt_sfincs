import logging
import gc
import os
import time
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import MeshComponent

from hydromt_sfincs.utils import make_regular_grid
from hydromt_sfincs.workflows.merge import (
    merge_multi_dataarrays,
)

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import export_image
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

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
            Interpolation method used to fill the buffer cells , by default "linear"
        """

        nlev = self.data.attrs["nr_levels"]
        xy = self.data.grid.face_coordinates
        nr_cells = len(xy)
        zz = np.full(nr_cells, np.nan)
        dx = self.data.attrs["dx"]
        dy = self.data.attrs["dy"]
        res = min(dx, dy)

        if self.model.crs.is_geographic:
            res *= 111111.0  # convert to meters

        # 0-based level indices
        level = self.data["level"].values - 1

        # Precompute index slices per level
        level_indices = [np.where(level == ilev)[0] for ilev in range(nlev)]

        # Precompute elevation sets per level
        # Add try statement here for compatibility with cht_bathymetry approach
        # try:
        elevation_list_per_level = [
            self.model._parse_datasets_elevation(elevation_list, res=res / (2**ilev))
            for ilev in range(nlev)
        ]
        # except Exception as e:
        #     print("Using bathymetry database for interpolation.")

        # get m and n indices
        n = self.data["n"] - 1  # 0-based
        m = self.data["m"] - 1  # 0-based

        def process_level(ilev):
            idx = level_indices[ilev]
            xz, yz = xy[idx, 0], xy[idx, 1]
            n_level, m_level = n[idx], m[idx]
            dxmin, dymin = dx / 2**ilev, dy / 2**ilev

            logger.info(f"Processing bathymetry level {ilev + 1} of {nlev} ...")

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

                in_chunk = np.where((xz >= x0) & (xz < x1) & (yz >= y0) & (yz < y1))[0]
                if len(in_chunk) == 0:
                    return

                # if bathymetry_database is not None:
                #     zgl[in_chunk] = bathymetry_database.get_bathymetry_on_points(
                #         xz[in_chunk],
                #         yz[in_chunk],
                #         min(dxmin, dymin),
                #         self.model.crs,
                #         elevation_list,
                #     )
                # else:
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
                da_dep = merge_multi_dataarrays(
                    da_list=elevation_list_per_level[ilev],
                    da_like=da_like,
                    buffer_cells=buffer_cells,
                    interp_method=interp_method,
                    logger=logger,
                )
                idx_y = np.searchsorted(da_dep.n.values, n_level[in_chunk].values)
                idx_x = np.searchsorted(da_dep.m.values, m_level[in_chunk].values)
                zgl[in_chunk] = da_dep.values[idx_y, idx_x]

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
            return idx, np.clip(zgl, zmin, zmax)

        # Loop over levels
        for ilev in range(nlev):
            idx, zgl_level = process_level(ilev)
            zz[idx] = zgl_level

        # Save result
        self.data["z"] = xu.UgridDataArray(
            xr.DataArray(data=zz, dims=[self.data.grid.face_dimension]), self.data.grid
        )

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
    # Datashader overlay
    # ------------------------------------------------------------------

    def get_datashader_dataframe(self):
        """Build a trimesh DataFrame for datashader rendering.

        Creates vertices (4 corners per cell) and simplices (2 triangles
        per cell) in EPSG:3857 for use with ``datashader.Canvas.trimesh()``.
        """
        if self.data is None or "z" not in self.data:
            self.datashader_vertices = pd.DataFrame()
            self.datashader_simplices = pd.DataFrame()
            return

        grid = self.data
        xy = grid.grid.face_coordinates
        z = grid["z"].values[:]
        level = grid["level"].values[:] - 1  # 0-based
        dx0 = grid.attrs["dx"]
        dy0 = grid.attrs["dy"]
        rotation = grid.attrs["rotation"]
        cosrot = np.cos(np.radians(rotation))
        sinrot = np.sin(np.radians(rotation))

        valid = np.isfinite(z)
        n_valid = np.sum(valid)
        if n_valid == 0:
            self.datashader_vertices = pd.DataFrame()
            self.datashader_simplices = pd.DataFrame()
            return

        cx = xy[valid, 0]
        cy = xy[valid, 1]
        cz = z[valid]
        clevel = level[valid]

        hdx = (dx0 / 2.0 ** clevel) / 2.0
        hdy = (dy0 / 2.0 ** clevel) / 2.0

        dx_offsets = np.array([-1, 1, 1, -1], dtype=np.float64)
        dy_offsets = np.array([-1, -1, 1, 1], dtype=np.float64)

        local_x = hdx[:, None] * dx_offsets[None, :]
        local_y = hdy[:, None] * dy_offsets[None, :]

        vx = cx[:, None] + cosrot * local_x - sinrot * local_y
        vy = cy[:, None] + sinrot * local_x + cosrot * local_y

        vx_flat = vx.ravel()
        vy_flat = vy.ravel()
        vz_flat = np.repeat(cz, 4)

        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        vx_3857, vy_3857 = transformer.transform(vx_flat, vy_flat)

        if self.model.crs.is_geographic and np.max(xy[:, 0]) > 180.0:
            vx_3857[vx_3857 < 0] += 40075016.68557849

        self.datashader_vertices = pd.DataFrame({
            "x": vx_3857, "y": vy_3857, "z": vz_flat,
        })

        cell_idx = np.arange(n_valid, dtype=np.int32)
        v_base = cell_idx * 4
        t0 = np.column_stack([v_base, v_base + 1, v_base + 2])
        t1 = np.column_stack([v_base, v_base + 2, v_base + 3])
        tris = np.vstack([t0, t1])

        self.datashader_simplices = pd.DataFrame({
            "v0": tris[:, 0], "v1": tris[:, 1], "v2": tris[:, 2],
        })

    def clear_datashader_dataframe(self):
        """Clear cached datashader data."""
        self.datashader_vertices = pd.DataFrame()
        self.datashader_simplices = pd.DataFrame()

    def map_overlay(
        self,
        file_name,
        xlim=None,
        ylim=None,
        cmap="gist_earth",
        cmin=None,
        cmax=None,
        width=800,
        **kwargs,
    ):
        """Create a map overlay image of the bathymetry using datashader.

        Parameters
        ----------
        file_name : str
            Output image file name (without extension — .png is appended).
        xlim : list
            [lon_min, lon_max] in WGS84 degrees.
        ylim : list
            [lat_min, lat_max] in WGS84 degrees.
        cmap : str or list
            Matplotlib colormap name or list of hex colors.
        cmin, cmax : float, optional
            Color scale range. If None, auto-scaled from data.
        width : int
            Output image width in pixels.

        Returns
        -------
        bool
            True if the image was created successfully.
        """
        if not HAS_DATASHADER:
            logger.warning("datashader is not installed.")
            return False

        if self.data is None or "z" not in self.data:
            return False

        try:
            if not hasattr(self, "datashader_vertices") or self.datashader_vertices.empty:
                self.get_datashader_dataframe()

            if self.datashader_vertices.empty:
                return False

            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            x_range = (xl0, xl1)
            y_range = (yl0, yl1)

            ratio = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
            height = int(width * ratio)

            cvs = ds.Canvas(
                x_range=x_range, y_range=y_range,
                plot_width=width, plot_height=height,
            )

            mesh = ds.utils.mesh(
                self.datashader_vertices,
                self.datashader_simplices,
            )

            agg = cvs.trimesh(
                self.datashader_vertices,
                self.datashader_simplices,
                mesh=mesh,
                agg=ds.mean("z"),
                interp=False,
            )

            if isinstance(cmap, str):
                from matplotlib import colormaps
                cmap = colormaps[cmap]

            span = None
            if cmin is not None and cmax is not None:
                span = (cmin, cmax)

            img = tf.shade(agg, cmap=cmap, span=span, how="linear")

            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.splitext(os.path.basename(file_name))[0]
            export_image(img, name, export_path=path)
            return True

        except Exception as e:
            logger.warning(f"map_overlay failed: {e}")
            return False


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
