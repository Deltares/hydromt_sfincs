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

    @hydromt_step
    def create(
        self,
        model: str = "sfincs",
        zmin: float = None,
        zmax: float = None,
        include_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        include_zmin: float = None,
        include_zmax: float = None,
        exclude_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_zmin: float = None,
        exclude_zmax: float = None,
        open_boundary_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        open_boundary_zmin: float = None,
        open_boundary_zmax: float = None,
        outflow_boundary_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        outflow_boundary_zmin: float = None,
        outflow_boundary_zmax: float = None,
        neumann_boundary_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        neumann_boundary_zmin: float = None,
        neumann_boundary_zmax: float = None,
        downstream_boundary_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        downstream_boundary_zmin: float = None,
        downstream_boundary_zmax: float = None,
        all_touched: bool = False,
        update_datashader_dataframe=False,
    ):
        """Setup active model mask and add boundaries. Note that boundary types can only be set when polygons are provided.

        Parameters
        ----------
        model : str, optional
            Model type, either 'sfincs' (default) or 'snapwave', for which the mask will be created.
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for active model cells.
        include_polygon, exclude_polygon: str, Path, gpd.GeoDataFrame, optional
            Path or data source name of polygons to include/exclude from the active model domain.
            Note that include (second last) and exclude (last) areas are processed after other critera,
            i.e. `zmin`, `zmax` and thus overrule these criteria for active model cells.
        include_zmin, include_zmax: float, optional
            Minimum and maximum elevation thresholds for included model cells.
        exclude_zmin, exclude_zmax: float, optional
            Minimum and maximum elevation thresholds for excluded model cells.
        open_boundary_polygon, outflow_boundary_polygon, neumann_boundary_polygon, downstream_boundary_polygon: str, Path, gpd.GeoDataFrame, optional
            Path or data source name for geometries with areas to include as open boundary, outflow boundary, neumann boundary, or downstream boundary.
            For each polygon, also the minimum and maximum elevation thresholds can be specified using the corresponding `*_zmin` and `*_zmax` arguments.

        See also:
        ---------
        * `create_active` method to setup active model cells
        * `create_boundary` method to setup boundary cells of a specific type

        """

        # Create active model cells
        self.create_active(
            model=model,
            zmin=zmin,
            zmax=zmax,
            include_polygon=include_polygon,
            include_zmin=include_zmin,
            include_zmax=include_zmax,
            exclude_polygon=exclude_polygon,
            exclude_zmin=exclude_zmin,
            exclude_zmax=exclude_zmax,
            all_touched=all_touched,
        )

        # Add boundary cels
        if open_boundary_polygon is not None:
            self.create_boundary(
                model=model,
                btype="waterlevel",
                include_polygon=open_boundary_polygon,
                include_zmin=open_boundary_zmin,
                include_zmax=open_boundary_zmax,
                all_touched=all_touched,
            )

        if outflow_boundary_polygon is not None:
            self.create_boundary(
                model=model,
                btype="outflow",
                include_polygon=outflow_boundary_polygon,
                include_zmin=outflow_boundary_zmin,
                include_zmax=outflow_boundary_zmax,
                all_touched=all_touched,
            )

        if downstream_boundary_polygon is not None:
            self.create_boundary(
                model=model,
                btype="downstream",
                include_polygon=downstream_boundary_polygon,
                include_zmin=downstream_boundary_zmin,
                include_zmax=downstream_boundary_zmax,
                all_touched=all_touched,
            )

        if neumann_boundary_polygon is not None:
            self.create_boundary(
                model=model,
                btype="neumann",
                include_polygon=neumann_boundary_polygon,
                include_zmin=neumann_boundary_zmin,
                include_zmax=neumann_boundary_zmax,
                all_touched=all_touched,
            )

        if update_datashader_dataframe:
            # For use in DelftDashboard
            self.get_datashader_dataframe()

    @hydromt_step
    def create_active(
        self,
        model: str = "sfincs",
        zmin: float = None,
        zmax: float = None,
        include_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        include_zmin: float = None,
        include_zmax: float = None,
        exclude_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_zmin: float = None,
        exclude_zmax: float = None,
        all_touched: bool = False,
        reset_mask: bool = True,
        copy_sfincsmask: bool = False,
    ):
        """Setup active model cells.

        The SFINCS model mask defines inactive (msk=0), active (msk=1), and waterlevel boundary (msk=2)
        and outflow boundary (msk=3) cells. This method sets the active and inactive cells.

        Active model cells are based on a region and cells with valid elevation (i.e. not nodata),
        optionally bounded by areas inside the include geomtries, outside the exclude geomtries,
        larger or equal than a minimum elevation threshhold and smaller or equal than a
        maximum elevation threshhold.
        All conditions are combined using a logical AND operation.

        Adds mask layer to quadtree grid:

        * **mask** map: model mask [-]

        Parameters
        ----------
        model : str, optional
            Model type, either 'sfincs' (default) or 'snapwave', for which the mask will be created.
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for active model cells.
        include_polygon, exclude_polygon: str, Path, gpd.GeoDataFrame, optional
            Path or data source name of polygons to include/exclude from the active model domain.
            Note that include (second last) and exclude (last) areas are processed after other critera,
            i.e. `zmin`, `zmax` and thus overrule these criteria for active model cells.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.
        reset_mask: bool, optional
            If True, reset existing mask before creating new active model cells.
        copy_sfincsmask: bool, optional
            If True and model is 'snapwave', copy the SFINCS mask to the SnapWave mask.
        """

        logger.info("Building mask ...")

        assert model in [
            "sfincs",
            "snapwave",
        ], "Model must be either 'sfincs' or 'snapwave'!"

        if model == "sfincs":
            varname = "mask"
        elif model == "snapwave":
            varname = "snapwave_mask"

        if copy_sfincsmask and model == "snapwave":
            assert "mask" in self.data, "SFINCS mask not found!"
            logger.info("Using SFINCS mask for SnapWave mask ...")
            self.data[varname] = self.data["mask"]
            return

        logger.info("Build new mask for: " + model + " ...")

        # read geometries from file, data catalog or use provided geodataframe
        gdf_include, gdf_exclude = None, None
        bbox = self.model.region.to_crs(4326).total_bounds

        # FIXME do we still want to support .pol files?
        if include_polygon is not None:
            if not isinstance(include_polygon, gpd.GeoDataFrame) and str(
                include_polygon
            ).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_include = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=include_polygon), crs=self.model.crs
                )
            else:
                gdf_include = self.data_catalog.get_geodataframe(
                    include_polygon, bbox=bbox
                )
        if exclude_polygon is not None:
            if not isinstance(exclude_polygon, gpd.GeoDataFrame) and str(
                exclude_polygon
            ).endswith(".pol"):
                gdf_exclude = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=exclude_polygon), crs=self.model.crs
                )
            else:
                gdf_exclude = self.data_catalog.get_geodataframe(
                    exclude_polygon, bbox=bbox
                )

        # get mask and dep data
        uda_mask = self.data["mask"] if "mask" in self.data else None

        uda_mask0 = None
        if not reset_mask and uda_mask is not None:
            # use current active mask
            uda_mask0 = uda_mask > 0

        # always initialize an inactive mask, note this resets any existing mask
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

        # apply include / exclude masks, first include (within zmin/zmax), then exclude (within zmin/zmax)
        if gdf_include is not None:
            try:
                _msk = (
                    xu.burn_vector_geometry(
                        gdf_include, self.data, fill=0, all_touched=all_touched
                    )
                    > 0
                )
                if include_zmin is not None or include_zmax is not None:
                    if "z" not in self.data:
                        raise ValueError(
                            "z required in combination with include_zmin / include_zmax"
                        )
                    uda_dep = self.data["z"]
                    if include_zmin is not None:
                        _msk = np.logical_and(_msk, uda_dep >= include_zmin)
                    if include_zmax is not None:
                        _msk = np.logical_and(_msk, uda_dep <= include_zmax)
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
                if exclude_zmin is not None or exclude_zmax is not None:
                    if "z" not in self.data:
                        raise ValueError(
                            "z required in combination with exclude_zmin / exclude_zmax"
                        )
                    uda_dep = self.data["z"]
                    if exclude_zmin is not None:
                        _msk = np.logical_and(_msk, uda_dep >= exclude_zmin)
                    if exclude_zmax is not None:
                        _msk = np.logical_and(_msk, uda_dep <= exclude_zmax)
                uda_mask = np.logical_and(uda_mask, ~_msk)
            except:
                logger.debug("No mask cells found within exclude polygon!")

        # add mask to grid
        self.data[varname] = xu.UgridDataArray(
            xr.DataArray(data=uda_mask, dims=[self.data.grid.face_dimension]),
            self.data.grid,
        )

    @hydromt_step
    def create_boundary(
        self,
        model: str = "sfincs",
        btype: str = "waterlevel",
        zmin: float = None,
        zmax: float = None,
        include_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        include_zmin: float = None,
        include_zmax: float = None,
        include_polygon_buffer: int = 0,
        exclude_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_zmin: float = None,
        exclude_zmax: float = None,
        all_touched: bool = True,
        reset_bounds: bool = True,
        copy_sfincsmask: bool = False,
    ):
        """Set boundary cells in the model mask.

        The SFINCS model mask defines inactive (mask=0), active (mask=1), and waterlevel boundary (mask=2)
        and outflow boundary (mask=3), downstream (mask=5) or neumann (mask=6) cells.
        Active cells set using the `create_active` method,
        while this method sets the different types of boundary cells, see `btype` argument.

        Boundary cells at the edge of the active model domain,
        optionally bounded by areas inside the include geomtries, outside the exclude geomtries,
        larger or equal than a minimum elevation threshhold and smaller or equal than a
        maximum elevation threshhold. All conditions are combined using a logical AND operation.

        Updates mask layer in quadtree grid:

        * **mask** map: model mask [-]

        Parameters
        ----------
        model : str, optional
            Model type, either 'sfincs' (default) or 'snapwave', for which the mask will be created.
        btype: str, optional
            Boundary type {'waterlevel', 'outflow', 'downstream', 'neumann'} for model='sfincs',
                {'waves', 'neumann'} for model='snapwave'
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for all boundary cells.
        include_polygon, exclude_polygon: str, Path, gpd.GeoDataFrame, optional
            Path or data source name for geometries with areas to include/exclude from
            the model boundary. These can be combined with `include_zmin` and `include_zmax` to
            further refine the selection of cells within the polygons.
        reset_bounds: bool, optional
            If True, reset existing boundary cells of the selected boundary
            type (`btype`) before setting new boundary cells, by default False.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.
        connectivity, {4, 8}:
            The connectivity used to detect the model edge, if 4 only horizontal and vertical
            connections are used, if 8 (default) also diagonal connections.
        """

        assert model in [
            "sfincs",
            "snapwave",
        ], "Model must be either 'sfincs' or 'snapwave'!"

        # specify mask variable name based on model
        if model == "sfincs":
            varname = "mask"
        elif model == "snapwave":
            varname = "snapwave_mask"

        # copy SFINCS mask to SnapWave mask when requested
        if copy_sfincsmask and model == "snapwave":
            assert "mask" in self.data, "SFINCS mask not found!"
            logger.info("Using SFINCS mask for SnapWave mask ...")
            self.data[varname] = self.data["mask"]
            return

        # check if mask already exists
        if varname not in self.data:
            raise ValueError("First setup active mask for model: " + model)
        else:
            uda_mask = self.data[varname]

        if "z" not in self.data and (zmin is not None or zmax is not None):
            raise ValueError("z required in combination with zmin / zmax")
        else:
            uda_dep = self.data["z"]

        # check boundary type
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
        if include_polygon is not None:
            if not isinstance(include_polygon, gpd.GeoDataFrame) and str(
                include_polygon
            ).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_include = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=include_polygon), crs=self.model.crs
                )
            else:
                gdf_include = self.data_catalog.get_geodataframe(
                    include_polygon, bbox=bbox
                )
            if include_polygon_buffer > 0:
                if self.model.crs.is_geographic:
                    include_polygon_buffer = include_polygon_buffer / 111111.0
                gdf_include["geometry"] = gdf_include.to_crs(self.model.crs).buffer(
                    include_polygon_buffer
                )
        if exclude_polygon is not None:
            if not isinstance(exclude_polygon, gpd.GeoDataFrame) and str(
                exclude_polygon
            ).endswith(".pol"):
                gdf_exclude = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=exclude_polygon), crs=self.model.crs
                )
            else:
                gdf_exclude = self.data_catalog.get_geodataframe(
                    exclude_polygon, bbox=bbox
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

        # find all boundary cells of the active mask
        bounds0 = self._find_boundary_cells(varname)
        bounds = bounds0.copy()

        # check general zmin and zmax
        if zmin is not None:
            bounds = np.logical_and(bounds, uda_dep >= zmin)
        if zmax is not None:
            bounds = np.logical_and(bounds, uda_dep <= zmax)

        # apply include / exclude masks, first include (within zmin/zmax), then exclude (within zmin/zmax)
        if gdf_include is not None:
            uda_include = (
                xu.burn_vector_geometry(
                    gdf_include, self.data, fill=0, all_touched=all_touched
                )
                > 0
            )
            if include_zmin is not None or include_zmax is not None:
                if "z" not in self.data:
                    raise ValueError(
                        "z required in combination with include_zmin / include_zmax"
                    )
                uda_dep = self.data["z"]
                if include_zmin is not None:
                    uda_include = np.logical_and(uda_include, uda_dep >= include_zmin)
                if include_zmax is not None:
                    uda_include = np.logical_and(uda_include, uda_dep <= include_zmax)
            bounds = np.logical_and(bounds, uda_include)
        if gdf_exclude is not None:
            uda_exclude = (
                xu.burn_vector_geometry(
                    gdf_exclude, self.data, fill=0, all_touched=all_touched
                )
                > 0
            )
            if exclude_zmin is not None or exclude_zmax is not None:
                if "z" not in self.data:
                    raise ValueError(
                        "z required in combination with exclude_zmin / exclude_zmax"
                    )
                uda_dep = self.data["z"]
                if exclude_zmin is not None:
                    uda_exclude = np.logical_and(uda_exclude, uda_dep >= exclude_zmin)
                if exclude_zmax is not None:
                    uda_exclude = np.logical_and(uda_exclude, uda_dep <= exclude_zmax)
            bounds = np.logical_and(bounds, ~uda_exclude)

        # set new boundary cells, keep same mask value for existing boundary cells
        ncells = np.count_nonzero(bounds.values)
        if ncells > 0:
            uda_mask = uda_mask.where(~bounds, np.uint8(bvalue))

        # TODO avoid any msk3 cells neighboring msk2 cells
        # TODO try to include 'diagonally connected msk=2 neighbouring cells'
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
