import logging
import os
from pathlib import Path
import warnings
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from hydromt import hydromt_step

from hydromt_sfincs.components.quadtree import SfincsQuadtreeMask

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


class SnapWaveQuadtreeMask(SfincsQuadtreeMask):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        super().__init__(
            model=model,
        )
        # For plotting map overlay (This is the only data that is stored in the object!
        # All other data is stored in the model.grid.data["mask"])
        self.datashader_dataframe = pd.DataFrame()

    @property
    def has_open_boundaries(self):
        """Returns True if mask contains open boundaries (mask = 2)"""
        if "snapwave_mask" not in self.data:
            return False

        mask = self.data["snapwave_mask"]
        return (mask == 2).any().item()

    @hydromt_step
    def create(
        self,
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
        neumann_boundary_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        neumann_boundary_zmin: float = None,
        neumann_boundary_zmax: float = None,
        all_touched: bool = False,
        update_datashader_dataframe=False,
    ):
        """Setup active model snapwave mask and add boundaries. Note that boundary types can only be set when polygons are provided.

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
        open_boundary_polygon, neumann_boundary_polygon: str, Path, gpd.GeoDataFrame, optional
            Path or data source name for geometries with areas to include as open boundary or neumann boundary.
            For each polygon, also the minimum and maximum elevation thresholds can be specified using the corresponding `*_zmin` and `*_zmax` arguments.

        See also:
        ---------
        * `create_active` method to setup active model cells
        * `create_boundary` method to setup boundary cells of a specific type

        """
        super().create(
            model="snapwave",
            zmin=zmin,
            zmax=zmax,
            include_polygon=include_polygon,
            include_zmin=include_zmin,
            include_zmax=include_zmax,
            exclude_polygon=exclude_polygon,
            exclude_zmin=exclude_zmin,
            exclude_zmax=exclude_zmax,
            open_boundary_polygon=open_boundary_polygon,
            open_boundary_zmin=open_boundary_zmin,
            open_boundary_zmax=open_boundary_zmax,
            neumann_boundary_polygon=neumann_boundary_polygon,
            neumann_boundary_zmin=neumann_boundary_zmin,
            neumann_boundary_zmax=neumann_boundary_zmax,
            all_touched=all_touched,
            update_datashader_dataframe=update_datashader_dataframe,
        )

    @hydromt_step
    def create_active(
        self,
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

        The Snapwave model mask defines inactive (msk=0), active (msk=1), and wave boundary (msk=2)
        cells. This method sets the active and inactive cells.

        Active model cells are based on a region and cells with valid elevation (i.e. not nodata),
        optionally bounded by areas inside the include geomtries, outside the exclude geomtries,
        larger or equal than a minimum elevation threshhold and smaller or equal than a
        maximum elevation threshhold.
        All conditions are combined using a logical AND operation.

        Adds layer to quadtree grid:

        * **snapwave_mask** map: model mask [-]

        Parameters
        ----------
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
            If True, ccopy the SFINCS mask to the SnapWave mask.
        """

        super().create_active(
            model="snapwave",
            zmin=zmin,
            zmax=zmax,
            include_polygon=include_polygon,
            include_zmin=include_zmin,
            include_zmax=include_zmax,
            exclude_polygon=exclude_polygon,
            exclude_zmin=exclude_zmin,
            exclude_zmax=exclude_zmax,
            all_touched=all_touched,
            reset_mask=reset_mask,
            copy_sfincsmask=copy_sfincsmask,
        )

    @hydromt_step
    def create_boundary(
        self,
        btype: str = "waves",
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
        connectivity: int = 8,
    ):
        """Set boundary cells in the model mask.

        The Snapwave model mask defines inactive (mask=0), active (mask=1), wave boundary (mask=2)
        and neumann boundary (mask=3) cells. Active cells set using the `create_active` method,
        while this method sets the different types of boundary cells, see `btype` argument.

        Boundary cells at the edge of the active model domain,
        optionally bounded by areas inside the include geomtries, outside the exclude geomtries,
        larger or equal than a minimum elevation threshhold and smaller or equal than a
        maximum elevation threshhold. All conditions are combined using a logical AND operation.

        Updates snapwave mask layer in quadtree grid:

        * **snapwave_mask** map: model mask [-]

        Parameters
        ----------
        btype: str, optional
            Boundary type {'waves', 'neumann'}, by default 'waves'.
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
        super().create_boundary(
            model="snapwave",
            btype=btype,
            zmin=zmin,
            zmax=zmax,
            include_polygon=include_polygon,
            include_zmin=include_zmin,
            include_zmax=include_zmax,
            include_polygon_buffer=include_polygon_buffer,
            exclude_polygon=exclude_polygon,
            exclude_zmin=exclude_zmin,
            exclude_zmax=exclude_zmax,
            all_touched=all_touched,
            reset_bounds=reset_bounds,
            copy_sfincsmask=copy_sfincsmask,
            connectivity=connectivity,
        )

    def get_datashader_dataframe(self, variable="snapwave_mask"):
        """Sets the datashader dataframe for plotting"""
        super().get_datashader_dataframe(variable=variable)

    def clear_datashader_dataframe(self):
        """Clears the datashader dataframe"""
        self.datashader_dataframe = pd.DataFrame()

    def map_overlay(
        self,
        file_name,
        xlim=None,
        ylim=None,
        colors=None,
        px=2,
        width=800,
        **kwargs,
    ):
        """Create a map overlay image of the SnapWave mask using datashader.

        Parameters
        ----------
        file_name : str
            Output image file name.
        xlim, ylim : list, optional
            Geographic (lon/lat) extent of the image.
        colors : dict, optional
            Mapping of integer mask values to colour strings, e.g.
            ``{1: "yellow", 2: "red", 3: "green"}``.  By default
            ``{1: "yellow", 2: "red", 3: "green"}``.
        px : int, optional
            Marker radius in pixels, by default 2.
        width : int, optional
            Output image width in pixels, by default 800.

        Returns
        -------
        bool
            True if the image was created successfully, False otherwise.
        """
        if colors is None:
            colors = {1: "yellow", 2: "red", 3: "green"}

        if not HAS_DATASHADER:
            logger.warning("Datashader is not available. Please install datashader.")
            return False

        if len(self.model.quadtree_grid.data.data_vars) == 0:
            return False

        try:
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

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

            images = []
            for mask_val, color in colors.items():
                df_sub = self.datashader_dataframe[
                    self.datashader_dataframe["snapwave_mask"] == mask_val
                ]
                if len(df_sub) > 0:
                    images.append(
                        tf.shade(
                            tf.spread(cvs.points(df_sub, "x", "y", ds.any()), px=px),
                            cmap=color,
                        )
                    )

            if not images:
                return False

            img = images[0]
            for im in images[1:]:
                img = tf.stack(img, im)

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
