import logging
import os
from pathlib import Path
import warnings
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xugrid as xu
from matplotlib import path
from pyproj import Transformer

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

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
logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsQuadtreeMask(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.quadtree_grid.data["mask"] array
        super().__init__(
            model=model,
        )
        # For plotting map overlay (This is the only data that is stored in the object! All other data is stored in the model.grid.data["mask"])
        self.datashader_dataframe = pd.DataFrame()

    @property
    def data(self):
        return self.model.quadtree_grid.data

    @property
    def empty_mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.quadtree_grid.empty_mask

    @property
    def face_coordinates(self):
        return self.model.quadtree_grid.face_coordinates

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
        outflow_boundary_polygon=None,
        neumann_boundary_polygon=None,
        downstream_boundary_polygon=None,
        include_zmin=-99999.0,
        include_zmax=99999.0,
        exclude_zmin=-99999.0,
        exclude_zmax=99999.0,
        open_boundary_zmin=-99999.0,
        open_boundary_zmax=99999.0,
        outflow_boundary_zmin=-99999.0,
        outflow_boundary_zmax=99999.0,
        neumann_boundary_zmin=-99999.0,
        neumann_boundary_zmax=99999.0,
        downstream_boundary_zmin=-99999.0,
        downstream_boundary_zmax=99999.0,
        update_datashader_dataframe=False,
        quiet=True,
    ):
        if not quiet:
            print("Building mask ...")

        nr_cells = self.model.quadtree_grid.data.sizes["mesh2d_nFaces"]

        mask = np.zeros(nr_cells, dtype=np.int8)
        x, y = self.face_coordinates
        z = self.data["z"].values[:]

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
            self.set_boundary_mask(
                mask, open_boundary_polygon, open_boundary_zmin, open_boundary_zmax, 2
            )

        # Outflow boundary polygons
        if outflow_boundary_polygon is not None:
            self.set_boundary_mask(
                mask,
                outflow_boundary_polygon,
                outflow_boundary_zmin,
                outflow_boundary_zmax,
                3,
            )

        # Downstream river boundary polygons
        if downstream_boundary_polygon is not None:
            self.set_boundary_mask(
                mask,
                downstream_boundary_polygon,
                downstream_boundary_zmin,
                downstream_boundary_zmax,
                5,
            )

        # Neumann boundary polygons
        if neumann_boundary_polygon is not None:
            self.set_boundary_mask(
                mask,
                neumann_boundary_polygon,
                neumann_boundary_zmin,
                neumann_boundary_zmax,
                6,
            )

        if update_datashader_dataframe:
            # For use in DelftDashboard
            self.get_datashader_dataframe()

        # Now add the data arrays
        ugrid2d = self.data.grid
        self.data["mask"] = xu.UgridDataArray(
            xr.DataArray(data=mask, dims=[ugrid2d.face_dimension]), ugrid2d
        )

    def set_boundary_mask(self, mask, boundary_polygon, zmin, zmax, mask_value):
        """Set the mask value for a given polygon"""
        x, y = self.face_coordinates
        z = self.data["z"].values[:]

        # Indices are 1-based in SFINCS so subtract 1 for python 0-based indexing
        mu = self.data["mu"].values[:]
        mu1 = self.data["mu1"].values[:] - 1
        mu2 = self.data["mu2"].values[:] - 1
        nu = self.data["nu"].values[:]
        nu1 = self.data["nu1"].values[:] - 1
        nu2 = self.data["nu2"].values[:] - 1
        md = self.data["md"].values[:]
        md1 = self.data["md1"].values[:] - 1
        md2 = self.data["md2"].values[:] - 1
        nd = self.data["nd"].values[:]
        nd1 = self.data["nd1"].values[:] - 1
        nd2 = self.data["nd2"].values[:] - 1

        for ip, polygon in boundary_polygon.iterrows():
            inpol = inpolygon(x, y, polygon["geometry"])
            # Only consider points that are:
            # 1) Inside the polygon
            # 2) Have a mask > 0
            # 3) z>=zmin
            # 4) z<=zmax
            iok = np.where((inpol) & (mask > 0) & (z >= zmin) & (z <= zmax))
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
                    mask[ic] = mask_value

    @hydromt_step
    def create(
        self,
        model: str = "sfincs",
        mask: Union[str, Path, gpd.GeoDataFrame] = None,
        include_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        mask_buffer: int = 0,
        zmin: float = None,
        zmax: float = None,
        all_touched: bool = False,
        reset_mask: bool = True,
        copy_sfincsmask: bool = False,
    ):
        logger.info("Building mask ...")

        assert model in [
            "sfincs",
            "snapwave",
        ], "Model must be either 'sfincs' or 'snapwave'!"

        if model == "sfincs":
            varname = "mak"
        elif model == "snapwave":
            varname = "snapwave_mask"

        if copy_sfincsmask and model == "snapwave":
            assert "mask" in self.data, "SFINCS mask not found!"
            logger.info("Using SFINCS mask for SnapWave mask ...")
            self.data[varname] = self.data["mask"]
            return

        logger.info("Build new mask for: " + model + " ...")

        # read geometries from file, data catalog or use provided geodataframe
        gdf_mask, gdf_include, gdf_exclude = None, None, None
        bbox = self.model.region.to_crs(4326).total_bounds
        if mask is not None:
            if not isinstance(mask, gpd.GeoDataFrame) and str(mask).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_mask = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=mask), crs=self.model.crs
                )
            else:
                gdf_mask = self.data_catalog.get_geodataframe(mask, bbox=bbox)
            if mask_buffer > 0:  # NOTE assumes model in projected CRS!
                gdf_mask["geometry"] = gdf_mask.to_crs(self.model.crs).buffer(
                    mask_buffer
                )
        if include_mask is not None:
            if not isinstance(include_mask, gpd.GeoDataFrame) and str(
                include_mask
            ).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_include = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=include_mask), crs=self.model.crs
                )
            else:
                gdf_include = self.data_catalog.get_geodataframe(
                    include_mask, bbox=bbox
                )
        if exclude_mask is not None:
            if not isinstance(exclude_mask, gpd.GeoDataFrame) and str(
                exclude_mask
            ).endswith(".pol"):
                gdf_exclude = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=exclude_mask), crs=self.model.crs
                )
            else:
                gdf_exclude = self.data_catalog.get_geodataframe(
                    exclude_mask, bbox=bbox
                )

        uda_mask0 = None
        if not reset_mask and varname in self.data:
            # use current active mask
            uda_mask0 = self.data[varname] > 0
        elif gdf_mask is not None:
            # initialize mask with given geodataframe
            uda_mask0 = (
                xu.burn_vector_geometry(
                    gdf_mask, self.data, fill=0, all_touched=all_touched
                )
                > 0
            )

        # always initialize an inactive mask
        uda_mask = self.empty_mask > 0

        if zmin is not None or zmax is not None:
            if "z" not in self.data:
                raise ValueError("z required in combination with zmin / zmax")
            uda_dep = self.data["z"]
            if zmin is not None or zmax is not None:
                _msk = uda_dep != np.nan
                if zmin is not None:
                    _msk = np.logical_and(_msk, uda_dep >= zmin)
                if zmax is not None:
                    _msk = np.logical_and(_msk, uda_dep <= zmax)
                if uda_mask0 is not None:
                    # if mask was provided; keep active mask only within valid elevations
                    uda_mask = np.logical_and(uda_mask0, _msk)
                else:
                    # no mask provided; set mask to valid elevations
                    uda_mask = _msk
            elif zmin is None and zmax is None and uda_mask0 is not None:
                # in case a mask/region was provided, but you didn't want to update the mask based on elevation
                # just continue with the provided mask
                uda_mask = uda_mask0

        # TODO add fill and drop area?

        if gdf_include is not None:
            try:
                _msk = (
                    xu.burn_vector_geometry(
                        gdf_include, self.data, fill=0, all_touched=all_touched
                    )
                    > 0
                )
                uda_mask = np.logical_or(uda_mask, _msk)  # NOTE logical OR statement
            except:
                logger.debug("No mask cells found within include polygon!")
        if gdf_exclude is not None:
            try:
                _msk = (
                    xu.burn_vector_geometry(
                        gdf_exclude, self.data, fill=0, all_touched=all_touched
                    )
                    > 0
                )
                uda_mask = np.logical_and(uda_mask, ~_msk)
            except:
                logger.debug("No mask cells found within exclude polygon!")

        # add mask to grid
        self.data[varname] = xu.UgridDataArray(
            xr.DataArray(data=uda_mask, dims=[self.data.grid.face_dimension]),
            self.data.grid,
        )

    @hydromt_step
    def set_bounds(
        self,
        model: str = "sfincs",
        btype: str = "waterlevel",
        include_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        include_mask_buffer: int = 0,
        zmin: float = None,
        zmax: float = None,
        # connectivity: int = 8,
        all_touched: bool = True,
        reset_bounds: bool = True,
        copy_sfincsmask: bool = False,
    ):
        assert model in [
            "sfincs",
            "snapwave",
        ], "Model must be either 'sfincs' or 'snapwave'!"

        if model == "sfincs":
            varname = "mask"
        elif model == "snapwave":
            varname = "snapwave_mask"

        if copy_sfincsmask and model == "snapwave":
            assert "msk" in self.data, "SFINCS mask not found!"
            logger.info("Using SFINCS mask for SnapWave mask ...")
            self.data[varname] = self.data["mask"]
            return

        if varname not in self.data:
            raise ValueError("First setup active mask for model: " + model)
        else:
            uda_mask = self.data[varname]

        if "z" not in self.data and (zmin is not None or zmax is not None):
            raise ValueError("z required in combination with zmin / zmax")
        else:
            uda_dep = self.data["z"]

        btype = btype.lower()
        if model == "sfincs":
            bvalues = {"waterlevel": 2, "outflow": 3, "downstream": 5, "neumann": 6}
            if btype not in bvalues:
                raise ValueError(
                    'btype must be one of "waterlevel", "outflow", "downstream", "neumann"'
                )
        elif model == "snapwave":
            bvalues = {"waves": 2, "neumann": 3}
            if btype not in bvalues:
                raise ValueError('btype must be one of "waves", "neumann"')

        # get include / exclude geometries
        gdf_include, gdf_exclude = None, None
        bbox = self.model.bbox
        if include_mask is not None:
            if not isinstance(include_mask, gpd.GeoDataFrame) and str(
                include_mask
            ).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_include = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=include_mask), crs=self.model.crs
                )
            else:
                gdf_include = self.data_catalog.get_geodataframe(
                    include_mask, bbox=bbox
                )
            if include_mask_buffer > 0:
                if self.model.crs.is_geographic:
                    include_mask_buffer = include_mask_buffer / 111111.0
                gdf_include["geometry"] = gdf_include.to_crs(self.model.crs).buffer(
                    include_mask_buffer
                )
        if exclude_mask is not None:
            if not isinstance(exclude_mask, gpd.GeoDataFrame) and str(
                exclude_mask
            ).endswith(".pol"):
                gdf_exclude = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=exclude_mask), crs=self.model.crs
                )
            else:
                gdf_exclude = self.data_catalog.get_geodataframe(
                    exclude_mask, bbox=bbox
                )

        bvalue = bvalues[btype]

        if reset_bounds:  # reset existing boundary cells
            logger.debug(f"{btype} (mask={bvalue:d}) boundary cells reset.")
            uda_mask = uda_mask.where(uda_mask != np.uint8(bvalue), np.uint8(1))
            if (
                zmin is None
                and zmax is None
                and gdf_include is None
                and gdf_exclude is None
            ):
                self.data[varname] = xu.UgridDataArray(
                    xr.DataArray(data=uda_mask, dims=[self.data.grid.face_dimension]),
                    self.data.grid,
                )
                return

        # find boundary cells of the active mask
        bounds_org = self._find_boundary_cells(varname)
        bounds = bounds_org.copy()

        if zmin is not None:
            bounds = np.logical_and(bounds, uda_dep >= zmin)
        if zmax is not None:
            bounds = np.logical_and(bounds, uda_dep <= zmax)
        if gdf_include is not None:
            uda_include = (
                xu.burn_vector_geometry(
                    gdf_include, self.data, fill=0, all_touched=all_touched
                )
                > 0
            )
            bounds = np.logical_and(bounds, uda_include)
        if gdf_exclude is not None:
            uda_exclude = (
                xu.burn_vector_geometry(
                    gdf_exclude, self.data, fill=0, all_touched=all_touched
                )
                > 0
            )
            bounds = np.logical_and(bounds, ~uda_exclude)

        # TODO avoid any msk3 cells neighboring msk2 cells
        ncells = np.count_nonzero(bounds.values)
        if ncells > 0:
            uda_mask = uda_mask.where(~bounds, np.uint8(bvalue))

        # # try to include 'diagonally connected msk=2 neighbouring cells'
        # if connectivity == 4:
        #     self.bounds_msk2 = uda_mask.copy()
        #     bounds_msk2 = self._find_boundary_cells_msk2()  # uda_mask)

        #     ncells = bounds_msk2.sum()  # np.count_nonzero(bounds_msk2.sum())
        #     if ncells > 0:
        #         uda_mask = uda_mask.where(~bounds_msk2, np.uint8(bvalue))

        # add mask to grid
        self.data[varname] = xu.UgridDataArray(
            xr.DataArray(data=uda_mask, dims=[self.data.grid.face_dimension]),
            self.data.grid,
        )

    def to_gdf(self, option="all"):
        """Returns a geodataframe with points for each cell in the mask"""

        nr_cells = self.model.quadtree_grid.data.sizes["mesh2d_nFaces"]

        if nr_cells == 0:
            # Return empty geodataframe
            return gpd.GeoDataFrame()
        xz, yz = self.face_coordinates
        mask = self.data["mask"]
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
        elif option == "downstream":
            iok = np.where((mask == 5))
        elif option == "neumann":
            iok = np.where((mask == 6))
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
        mask = self.model.quadtree_grid.data["mask"]
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
        x = self.face_coordinates[:, 0]
        y = self.face_coordinates[:, 1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x) > 180.0:
                cross_dateline = True
        mask = self.model.quadtree_grid.data["mask"].values[:]
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
        downstream_color="blue",
        neumann_color="purple",
        outflow_color="green",
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
        outflow_color : str, optional
            The color of the outflow cells
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
            dfneu = self.datashader_dataframe[self.datashader_dataframe["mask"] == 5]
            dfdwn = self.datashader_dataframe[self.datashader_dataframe["mask"] == 6]
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
                cmap=outflow_color,
            )
            img_n = tf.shade(
                tf.spread(cvs.points(dfneu, "x", "y", ds.any()), px=px),
                cmap=neumann_color,
            )
            img_d = tf.shade(
                tf.spread(cvs.points(dfdwn, "x", "y", ds.any()), px=px),
                cmap=downstream_color,
            )
            img = tf.stack(img_a, img_b, img_o, img_n, img_d)

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

    def _find_boundary_cells(self, varname):
        mu = self.data["mu"].values[:]
        mu1 = self.data["mu1"].values[:] - 1
        mu2 = self.data["mu2"].values[:] - 1
        nu = self.data["nu"].values[:]
        nu1 = self.data["nu1"].values[:] - 1
        nu2 = self.data["nu2"].values[:] - 1
        md = self.data["md"].values[:]
        md1 = self.data["md1"].values[:] - 1
        md2 = self.data["md2"].values[:] - 1
        nd = self.data["nd"].values[:]
        nd1 = self.data["nd1"].values[:] - 1
        nd2 = self.data["nd2"].values[:] - 1

        # mask = self.data["msk"].values[:]
        mask = self.data[varname].values[:]  # TL: can be both sfincs or snapwave msk

        nr_cells = self.data.sizes["mesh2d_nFaces"]

        bounds = np.zeros(nr_cells, dtype=bool)

        # Check left neighbors
        left_coarser = md <= 0  # Coarser or equal to the left
        left_finer1 = (md1 >= 0) & (mask[md1] == 0)  # Cell to the left and inactive
        left_finer2 = (md2 >= 0) & (
            mask[md2] == 0
        )  # (Finer) cell to the left and inactive
        bounds |= (left_coarser & (left_finer1)) | (  # cell to the left is inactive
            ~left_coarser & (left_finer1 | left_finer2)
        )  # one of the finer cells to the left is inactive

        # Check right neighbors
        right_coarser = mu <= 0
        right_finer1 = (mu1 >= 0) & (mask[mu1] == 0)
        right_finer2 = (mu2 >= 0) & (mask[mu2] == 0)
        bounds |= (right_coarser & (right_finer1 | right_finer2)) | (
            ~right_coarser & (right_finer1 | right_finer2)
        )

        # Check bottom neighbors
        below_coarser = nd <= 0
        below_finer1 = (nd1 >= 0) & (mask[nd1] == 0)
        below_finer2 = (nd2 >= 0) & (mask[nd2] == 0)
        bounds |= (below_coarser & (below_finer1 | below_finer2)) | (
            ~below_coarser & (below_finer1 | below_finer2)
        )

        # Check top neighbors
        above_coarser = nu <= 0
        above_finer1 = (nu1 >= 0) & (mask[nu1] == 0)
        above_finer2 = (nu2 >= 0) & (mask[nu2] == 0)
        bounds |= (above_coarser & (above_finer1 | above_finer2)) | (
            ~above_coarser & (above_finer1 | above_finer2)
        )

        # Handling boundary cells
        bounds[md1 == -1] = True  # Left boundary
        bounds[mu1 == -1] = True  # Right boundary
        bounds[nd1 == -1] = True  # Bottom boundary
        bounds[nu1 == -1] = True  # Top boundary

        # Get rid of the inactive boundary cells that were added
        # in the previous step
        bounds[mask == 0] = False

        return bounds


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
