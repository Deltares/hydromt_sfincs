import logging
import os
import warnings
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xugrid as xu
from matplotlib import path
from pyproj import Transformer

from hydromt.model.components import ModelComponent

np.warnings = warnings

# optional dependency
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import export_image

    HAS_DATASHADER = True

except ImportError:
    HAS_DATASHADER = False

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

# TODO actually use the logger instead of print statements
logger = logging.getLogger(__name__)


class SnapWaveQuadtreeMask(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.quadtree_grid.data["mask"] array
        self.data = None
        super().__init__(
            model=model,
        )
        # For plotting map overlay (This is the only data that is stored in the object! All other data is stored in the model.grid.data["mask"])
        self.datashader_dataframe = pd.DataFrame()

    def read(self):
        # The mask values are read when the quadtree grid is read
        pass

    def write(self):
        # The mask values are written when the quadtree grid is written
        pass

    def build(
        self,
        zmin=99999.0,
        zmax=-99999.0,
        include_polygon=None,
        exclude_polygon=None,
        open_boundary_polygon=None,
        neumann_boundary_polygon=None,
        include_zmin=-99999.0,
        include_zmax=99999.0,
        exclude_zmin=-99999.0,
        exclude_zmax=99999.0,
        open_boundary_zmin=-99999.0,
        open_boundary_zmax=99999.0,
        neumann_boundary_zmin=-99999.0,
        neumann_boundary_zmax=99999.0,
        update_datashader_dataframe=False,
        quiet=True,
    ):
        if not quiet:
            print("Building mask ...")

        nr_cells = self.model.quadtree_grid.data.sizes["mesh2d_nFaces"]

        mask = np.zeros(nr_cells, dtype=np.int8)
        x, y = self.model.quadtree_grid.face_coordinates
        z = self.model.quadtree_grid.data["z"].values[:]

        # Indices are 1-based in SFINCS so subtract 1 for python 0-based indexing
        mu = self.model.quadtree_grid.data["mu"].values[:]
        mu1 = self.model.quadtree_grid.data["mu1"].values[:] - 1
        mu2 = self.model.quadtree_grid.data["mu2"].values[:] - 1
        nu = self.model.quadtree_grid.data["nu"].values[:]
        nu1 = self.model.quadtree_grid.data["nu1"].values[:] - 1
        nu2 = self.model.quadtree_grid.data["nu2"].values[:] - 1
        md = self.model.quadtree_grid.data["md"].values[:]
        md1 = self.model.quadtree_grid.data["md1"].values[:] - 1
        md2 = self.model.quadtree_grid.data["md2"].values[:] - 1
        nd = self.model.quadtree_grid.data["nd"].values[:]
        nd1 = self.model.quadtree_grid.data["nd1"].values[:] - 1
        nd2 = self.model.quadtree_grid.data["nd2"].values[:] - 1

        if zmin >= zmax:
            # Do not include any points initially
            if include_polygon is None:
                print(
                    "WARNING: Entire mask set to zeros! Please ensure zmax is greater than zmin, or provide include polygon(s) !"
                )
                return
        else:
            if z is not None:
                # Set initial mask based on zmin and zmax
                iok = np.where((z >= zmin) & (z <= zmax))
                mask[iok] = 1
            else:
                print(
                    "WARNING: Entire mask set to zeros! No depth values found on grid."
                )

        # Include polygons
        if include_polygon is not None:
            for ip, polygon in include_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                iok = np.where((inpol) & (z >= include_zmin) & (z <= include_zmax))
                mask[iok] = 1

        # Exclude polygons
        if exclude_polygon is not None:
            for ip, polygon in exclude_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                iok = np.where((inpol) & (z >= exclude_zmin) & (z <= exclude_zmax))
                mask[iok] = 0

        # Open boundary polygons
        if open_boundary_polygon is not None:
            for ip, polygon in open_boundary_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok = np.where(
                    (inpol)
                    & (mask > 0)
                    & (z >= open_boundary_zmin)
                    & (z <= open_boundary_zmax)
                )
                for ic in iok[0]:
                    okay = False
                    # Check neighbors, cell must have at least one inactive neighbor
                    # Left
                    if md[ic] <= 0:
                        # Coarser or equal to the left
                        if md1[ic] >= 0:
                            # Cell has neighbor to the left
                            if mask[md1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer to the left
                        if md1[ic] >= 0:
                            # Cell has neighbor to the left
                            if mask[md1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if md2[ic] >= 0:
                            # Cell has neighbor to the left
                            if mask[md2[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True

                    # Below
                    if nd[ic] <= 0:
                        # Coarser or equal below
                        if nd1[ic] >= 0:
                            # Cell has neighbor below
                            if mask[nd1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer below
                        if nd1[ic] >= 0:
                            # Cell has neighbor below
                            if mask[nd1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if nd2[ic] >= 0:
                            # Cell has neighbor below
                            if mask[nd2[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True

                    # Right
                    if mu[ic] <= 0:
                        # Coarser or equal to the right
                        if mu1[ic] >= 0:
                            # Cell has neighbor to the right
                            if mask[mu1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer to the left
                        if mu1[ic] >= 0:
                            # Cell has neighbor to the right
                            if mask[mu1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if mu2[ic] >= 0:
                            # Cell has neighbor to the right
                            if mask[mu2[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True

                    # Above
                    if nu[ic] <= 0:
                        # Coarser or equal above
                        if nu1[ic] >= 0:
                            # Cell has neighbor above
                            if mask[nu1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                    else:
                        # Finer below
                        if nu1[ic] >= 0:
                            # Cell has neighbor above
                            if mask[nu1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True
                        if nu2[ic] >= 0:
                            # Cell has neighbor above
                            if mask[nu2[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 2
                            okay = True

                    if okay:
                        mask[ic] = 2

        # Outflow boundary polygons
        if neumann_boundary_polygon is not None:
            for ip, polygon in neumann_boundary_polygon.iterrows():
                inpol = inpolygon(x, y, polygon["geometry"])
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok = np.where(
                    (inpol)
                    & (mask > 0)
                    & (z >= neumann_boundary_zmin)
                    & (z <= neumann_boundary_zmax)
                )
                for ic in iok[0]:
                    okay = False
                    # Check neighbors, cell must have at least one inactive neighbor
                    # Left
                    if md[ic] <= 0:
                        # Coarser or equal to the left
                        if md1[ic] >= 0:
                            # Cell has neighbor to the left
                            if mask[md1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer to the left
                        if md1[ic] >= 0:
                            # Cell has neighbor to the left
                            if mask[md1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if md2[ic] >= 0:
                            # Cell has neighbor to the left
                            if mask[md2[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True

                    # Below
                    if nd[ic] <= 0:
                        # Coarser or equal below
                        if nd1[ic] >= 0:
                            # Cell has neighbor below
                            if mask[nd1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer below
                        if nd1[ic] >= 0:
                            # Cell has neighbor below
                            if mask[nd1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if nd2[ic] >= 0:
                            # Cell has neighbor below
                            if mask[nd2[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True

                    # Right
                    if mu[ic] <= 0:
                        # Coarser or equal to the right
                        if mu1[ic] >= 0:
                            # Cell has neighbor to the right
                            if mask[mu1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer to the left
                        if mu1[ic] >= 0:
                            # Cell has neighbor to the right
                            if mask[mu1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if mu2[ic] >= 0:
                            # Cell has neighbor to the right
                            if mask[mu2[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True

                    # Above
                    if nu[ic] <= 0:
                        # Coarser or equal above
                        if nu1[ic] >= 0:
                            # Cell has neighbor above
                            if mask[nu1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    else:
                        # Finer below
                        if nu1[ic] >= 0:
                            # Cell has neighbor above
                            if mask[nu1[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                        if nu2[ic] >= 0:
                            # Cell has neighbor above
                            if mask[nu2[ic]] == 0:
                                # And it's inactive
                                okay = True
                        else:
                            # No neighbor, so set mask = 3
                            okay = True
                    if okay:
                        mask[ic] = 3

        # Now add the data arrays
        ugrid2d = self.model.quadtree_grid.data.grid
        self.model.quadtree_grid.data["snapwave_mask"] = xu.UgridDataArray(
            xr.DataArray(data=mask, dims=[ugrid2d.face_dimension]), ugrid2d
        )

    def to_gdf(self, option="all"):
        """Returns a geodataframe with points for each cell in the mask"""

        nr_cells = self.model.quadtree_grid.data.sizes["mesh2d_nFaces"]

        if nr_cells == 0:
            # Return empty geodataframe
            return gpd.GeoDataFrame()
        xz, yz = self.model.quadtree_grid.face_coordinates()
        mask = self.model.quadtree_grid.data["snapwave_mask"]
        gdf_list = []
        okay = np.zeros(mask.shape, dtype=int)
        if option == "all":
            iok = np.where((mask > 0))
        elif option == "include":
            iok = np.where((mask == 1))
        elif option == "open":
            iok = np.where((mask == 2))
        elif option == "outflow":
            iok = np.where((mask == 3))
        else:
            iok = np.where((mask > -999))
        okay[iok] = 1
        for icel in range(nr_cells):
            if okay[icel] == 1:
                point = shapely.geometry.Point(xz[icel], yz[icel])
                d = {"geometry": point}
                gdf_list.append(d)

        if gdf_list:
            gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        else:
            # Cannot set crs of gdf with empty list
            gdf = gpd.GeoDataFrame(gdf_list)

        return gdf

    def has_open_boundaries(self):
        """Returns True if mask contains open boundaries (mask = 2)"""
        mask = self.model.quadtree_grid.data["snapwave_mask"]
        if mask is None:
            return False
        if np.any(mask == 2):
            return True
        else:
            return False

    def get_datashader_dataframe(self):
        """Sets the datashader dataframe for plotting"""
        # Create a dataframe with points elements
        # Coordinates of cell centers
        x = self.model.quadtree_grid.data.grid.face_coordinates[:, 0]
        y = self.model.quadtree_grid.data.grid.face_coordinates[:, 1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x) > 180.0:
                cross_dateline = True
        mask = self.model.quadtree_grid.data["snapwave_mask"].values[:]
        # Get rid of cells with mask = 0
        iok = np.where(mask > 0)
        x = x[iok]
        y = y[iok]
        mask = mask[iok]
        if np.size(x) == 0:
            # Return empty dataframe
            self.datashader_dataframe = pd.DataFrame()
            return
        # Transform all to 3857 (web mercator)
        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        x, y = transformer.transform(x, y)
        if cross_dateline:
            x[x < 0] += 40075016.68557849

        self.datashader_dataframe = pd.DataFrame(dict(x=x, y=y, mask=mask))

    def clear_datashader_dataframe(self):
        """Clears the datashader dataframe"""
        # Called in model.grid.build method
        self.datashader_dataframe = pd.DataFrame()

    def map_overlay(
        self,
        file_name,
        xlim=None,
        ylim=None,
        active_color="yellow",
        boundary_color="red",
        neumann_color="green",
        px=2,
        width=800,
    ):
        """Creates a map overlay image of the mask

        Parameters
        ----------
        file_name : str
            The file name of the image
        xlim : list, optional
            The x limits of the image
        ylim : list, optional
            The y limits of the image
        active_color : str, optional
            The color of the active cells
        boundary_color : str, optional
            The color of the boundary cells
        neumann_color : str, optional
            The color of the neumann cells
        px : int, optional
            The marker size in pixels
        width : int, optional
            The width of the image in pixels

        Returns
        -------
        bool
            True if the image was created successfully, False otherwise
        """

        # check if datashader is available
        if not HAS_DATASHADER:
            logger.warning("Datashader is not available. Please install datashader.")
            return False

        if self.model.quadtree_grid.data is None:
            # No grid or mask points
            return False

        try:
            # Check if datashader dataframe is empty (maybe it was not made yet, or it was cleared)
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

            # If it is still empty (because there are no active cells), return False
            if self.datashader_dataframe.empty:
                return False

            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)

            cvs = ds.Canvas(
                x_range=xlim, y_range=ylim, plot_height=height, plot_width=width
            )

            # Instead, we can create separate images for each mask and stack them
            dfact = self.datashader_dataframe[self.datashader_dataframe["mask"] == 1]
            dfbnd = self.datashader_dataframe[self.datashader_dataframe["mask"] == 2]
            dfout = self.datashader_dataframe[self.datashader_dataframe["mask"] == 3]
            img_a = tf.shade(
                tf.spread(cvs.points(dfact, "x", "y", ds.any()), px=px),
                cmap=active_color,
            )
            img_b = tf.shade(
                tf.spread(cvs.points(dfbnd, "x", "y", ds.any()), px=px),
                cmap=boundary_color,
            )
            img_o = tf.shade(
                tf.spread(cvs.points(dfout, "x", "y", ds.any()), px=px),
                cmap=neumann_color,
            )
            img = tf.stack(img_a, img_b, img_o)

            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True

        except Exception as e:
            print(e)
            return False


def get_neighbors_in_larger_cell(n, m):
    nnbr = [-1, -1, -1, -1]
    mnbr = [-1, -1, -1, -1]
    if not odd(n) and not odd(m):
        # lower left
        nnbr[0] = n + 1
        mnbr[0] = m
        nnbr[1] = n
        mnbr[1] = m + 1
        nnbr[2] = n + 1
        mnbr[2] = m + 1
    elif not odd(n) and odd(m):
        # lower right
        nnbr[1] = n
        mnbr[1] = m - 1
        nnbr[2] = n + 1
        mnbr[2] = m - 1
        nnbr[3] = n + 1
        mnbr[3] = m
    elif odd(n) and not odd(m):
        # upper left
        nnbr[1] = n - 1
        mnbr[1] = m
        nnbr[2] = n - 1
        mnbr[2] = m + 1
        nnbr[3] = n
        mnbr[3] = m + 1
    else:
        # upper right
        nnbr[1] = n - 1
        mnbr[1] = m - 1
        nnbr[2] = n - 1
        mnbr[2] = m
        nnbr[3] = n
        mnbr[3] = m - 1
    return nnbr, mnbr


def odd(num):
    if (num % 2) == 1:
        return True
    else:
        return False


def even(num):
    if (num % 2) == 0:
        return True
    else:
        return False


def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)


def binary_search(vals, val):
    indx = np.searchsorted(vals, val)
    if indx < np.size(vals):
        if vals[indx] == val:
            return indx
    return None
