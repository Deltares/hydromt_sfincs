import logging
import os
from os.path import abspath, basename, dirname, isabs, isfile, join
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xugrid as xu
from pyproj import CRS, Transformer

from hydromt.model.components import MeshComponent, ModelComponent
from hydromt_sfincs.utils import xu_open_dataset
from hydromt_sfincs.subgrid import SubgridTableQuadtree

from hydromt_sfincs.quadtree_builder import build_quadtree_xugrid, cut_inactive_cells

# optional dependency
try:
    import datashader.transfer_functions as tf
    from datashader import Canvas
    from datashader.utils import export_image

    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(__name__)


class QuadtreeGrid(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs_grid.nc"
        self.data: xu.UgridDataset = None
        self._data: xu.UgridDataset = None
        self.version = 0
        # Subgrid should be separate model component
        # self.subgrid = SubgridTableQuadtree()
        self.datashader_dataframe = pd.DataFrame()

        super().__init__(
            model=model,
        )

    @property
    def crs(self):
        if self.data is None:
            return None
        return self.data.grid.crs

    @property
    def face_coordinates(self):
        if self.data is None:
            return None
        xy = self.data.grid.face_coordinates
        return xy[:, 0], xy[:, 1]

    @property
    def exterior(self):
        if self.data is None:
            return gpd.GeoDataFrame()
        indx = self.data.grid.edge_node_connectivity[self.data.grid.exterior_edges, :]
        x = self.data.grid.node_x[indx]
        y = self.data.grid.node_y[indx]

        # Make linestrings from numpy arrays x and y
        linestrings = [
            shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))
        ]
        # Merge linestrings
        merged = shapely.ops.linemerge(linestrings)
        # Merge polygons
        polygons = shapely.ops.polygonize(merged)

        return gpd.GeoDataFrame(geometry=list(polygons), crs=self.crs)

    @property
    def empty_mask(self):
        if self.data is None:
            return None
        # create empty mask
        da0 = xr.DataArray(
            data=np.zeros(shape=len(self.data.grid.face_coordinates)),
            dims=self.data.grid.face_dimension,
        )
        return xu.UgridDataArray(da0, self.data.grid)

    @property
    def mask(self) -> xu.UgridDataArray:
        """Return the mask of the quadtree grid."""
        if "mask" in self.data:
            da_mask = self.data["mask"]
        else:
            da_mask = self.empty_mask
        return da_mask

    # %% core HydroMT-SFINCS functions:
    # _data (coming from MeshComponent)
    # _initialize (coming from MeshComponent)
    # read
    # write
    # set (coming from MeshComponent)
    # create

    def read(self, filename: str | Path = None):
        """Reads a quadtree netcdf file and stores it in the QuadtreeGrid object."""

        # check if in read mode and initialize grid
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if qtrfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "qtrfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if qtr file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(f"Quadtree grid file not found: {abs_file_path}")

        self.data = xu.load_dataset(abs_file_path)
        # set CRS (not sure if that should be stored in the netcdf in this way)
        self.data.grid.set_crs(CRS.from_wkt(self.data["crs"].crs_wkt))

    def write(self, filename: str | Path = None, version: int = 0):
        """Writes a QuadTree SFINCS netcdf file."""

        # Get absolute file name and set it in config if bndfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "qtrfile", value=filename, default="sfincs.qtr"
        )

        # And write the file
        ds = self.data.ugrid.to_dataset()
        ds.attrs = self.data.attrs
        ds.to_netcdf(abs_file_path)
        ds.close()

    def set(
        self,
        x0: float,
        y0: float,
        nmax: int,
        mmax: int,
        dx: float,
        dy: float,
        rotation: float,
        refinement_polygons: Optional[gpd.GeoDataFrame] = None,
        bathymetry_sets: Optional[List] = None,
        bathymetry_database: Optional = None,
    ):
        """Build the Quadtree grid.

        Parameters
        ----------
        x0 : float
            x-coordinate of the lower left corner of the grid.
        y0 : float
            y-coordinate of the lower left corner of the grid.
        nmax : int
            Maximum number of cells in x-direction.
        mmax : int
            Maximum number of cells in y-direction.
        dx : float
            Cell size in x-direction.
        dy : float
            Cell size in y-direction.
        rotation : float
            Rotation angle of the grid in degrees.
        refinement_polygons : gpd.GeoDataFrame, optional
            GeoDataFrame with polygons that define areas where the grid should be refined.
        bathymetry_sets : list, optional
            List of bathymetry sets.
        bathymetry_database : str, optional
            Path to the bathymetry database.
        """

        # Clear datashader dataframes
        self.clear_datashader_dataframe()
        self.model.quadtree_mask.clear_datashader_dataframe()

        # Get the CRS from the model config
        epsg = self.model.config.get("epsg", None)
        crs = CRS.from_epsg(epsg) if epsg is not None else CRS.from_epsg(4326)

        # Build the quadtree grid
        self.data = build_quadtree_xugrid(
            x0,
            y0,
            nmax,
            mmax,
            dx,
            dy,
            rotation,
            crs,
            refinement_polygons=refinement_polygons,
            bathymetry_sets=bathymetry_sets,
            bathymetry_database=bathymetry_database,
        )

    def cut_inactive_cells(self):
        # Clear datashader dataframes (new ones will be created when needed by map_overlay methods)
        self.clear_datashader_dataframe()
        self.model.quadtree_mask.clear_datashader_dataframe()
        # Cut inactive cells
        self.data = cut_inactive_cells(self.data)

    def snap_to_grid(self, polyline, max_snap_distance=1.0):
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        geom_list = []
        for _, line in polyline.iterrows():
            geom = line["geometry"]
            if geom.geom_type == "LineString":
                geom_list.append(geom)
        gdf = gpd.GeoDataFrame({"geometry": geom_list})
        _, snapped_gdf = xu.snap_to_grid(
            gdf, self.data.grid, max_snap_distance=max_snap_distance
        )
        snapped_gdf = snapped_gdf.set_crs(self.crs)
        return snapped_gdf

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):
        """Create a map overlay of the grid

        Parameters
        ----------
        file_name : str | Path
            File name of the map overlay
        xlim : list, optional
            x-axis limits (longitude)
        ylim : list, optional
            y-axis limits (latitude)
        color : str, optional
            Color of the grid lines
        width : int, optional
            Width of the map overlay in pixels

        Returns
        -------
        bool
            True if the map overlay was created successfully, False otherwise
        """
        # TODO: xlim and ylim should not be optional and be called lonlim and latlim or just give bbox

        # check if datashader is available
        if not HAS_DATASHADER:
            logger.warning("Datashader is not available. Please install datashader.")
            return False

        if self.data is None:
            # No grid (yet)
            return False

        try:
            # Check if datashader dataframe is empty (maybe it was not made yet, or it was cleared)
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)
            cvs = Canvas(
                x_range=xlim, y_range=ylim, plot_height=height, plot_width=width
            )
            agg = cvs.line(
                self.datashader_dataframe, x=["x1", "x2"], y=["y1", "y2"], axis=1
            )
            img = tf.shade(agg)
            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True
        except Exception as e:
            return False

    def get_datashader_dataframe(self):
        """Creates a dataframe with line elements for datashader"""
        # Create a dataframe with line elements
        x1 = self.data.grid.edge_node_coordinates[:, 0, 0]
        x2 = self.data.grid.edge_node_coordinates[:, 1, 0]
        y1 = self.data.grid.edge_node_coordinates[:, 0, 1]
        y2 = self.data.grid.edge_node_coordinates[:, 1, 1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x1) > 180.0 or np.max(x2) > 180.0:
                cross_dateline = True
        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        if cross_dateline:
            x1[x1 < 0] += 40075016.68557849
            x2[x2 < 0] += 40075016.68557849
        self.datashader_dataframe = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def clear_datashader_dataframe(self):
        """Clears the datashader dataframe"""
        self.datashader_dataframe = pd.DataFrame()
