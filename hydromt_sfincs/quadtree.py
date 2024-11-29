import logging
import os
import time
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
from matplotlib import path
from pyproj import CRS, Transformer

np.warnings = warnings

import geopandas as gpd
import pandas as pd
import shapely

try:
    import xugrid as xu
except ImportError:
    raise ImportError("xugrid is not installed. Please install it first.")
import xarray as xr

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import export_image
except ImportError:
    raise ImportError("datashader is not installed. Please install it first.")

from hydromt_sfincs.subgrid import SubgridTableQuadtree

logger = logging.getLogger(__name__)


class QuadtreeGrid:
    def __init__(self, logger=logger):
        self.nr_cells = 0
        self.nr_refinement_levels = 1
        self.version = 0

        self.data = None
        self.subgrid = SubgridTableQuadtree()
        self.df = None

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

    def read(self, file_name: Union[str, Path] = "sfincs.nc"):
        """Reads a quadtree netcdf file and stores it in the QuadtreeGrid object."""

        self.data = xu.open_dataset(file_name)
        self.data.close()  # TODO check if close works/is needed

        # TODO make similar to fortran conventions?
        # Rename to python conventions
        self.data = self.data.rename({"z": "dep"}) if "z" in self.data else self.data
        self.data = (
            self.data.rename({"mask": "msk"}) if "mask" in self.data else self.data
        )
        self.data = (
            self.data.rename({"snapwave_mask": "snapwave_msk"})
            if "snapwave_mask" in self.data
            else self.data
        )

        self.nr_cells = self.data.sizes["mesh2d_nFaces"]

        # set CRS (not sure if that should be stored in the netcdf in this way)
        # self.data.crs = CRS.from_wkt(self.data["crs"].crs_wkt)
        self.data.grid.set_crs(CRS.from_wkt(self.data["crs"].crs_wkt))

        for key, value in self.data.attrs.items():
            setattr(self, key, value)

    def write(self, file_name: Union[str, Path] = "sfincs.nc", version: int = 0):
        """Writes a quadtree SFINCS netcdf file."""

        # TODO do we want to cut inactive cells here? Or already when creating the mask?

        attrs = self.data.attrs
        ds = self.data.ugrid.to_dataset()

        # TODO make similar to fortran conventions
        # RENAME TO FORTRAN CONVENTION
        ds = ds.rename({"dep": "z"}) if "dep" in ds else ds
        ds = ds.rename({"msk": "mask"}) if "msk" in ds else ds
        ds = (
            ds.rename({"snapwave_msk": "snapwave_mask"}) if "snapwave_msk" in ds else ds
        )

        ds.attrs = attrs
        ds.to_netcdf(file_name)

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):
        if self.data is None:
            # No grid (yet)
            return False
        try:
            if not hasattr(self, "df"):
                self.df = None
            if self.df is None:
                self._get_datashader_dataframe()

            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)
            cvs = ds.Canvas(
                x_range=xlim, y_range=ylim, plot_height=height, plot_width=width
            )
            agg = cvs.line(self.df, x=["x1", "x2"], y=["y1", "y2"], axis=1)
            img = tf.shade(agg)
            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True
        except Exception:
            return False

    def snap_to_grid(self, polyline, max_snap_distance=1.0):
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        geom_list = []
        for iline, line in polyline.iterrows():
            geom = line["geometry"]
            if geom.geom_type == "LineString":
                geom_list.append(geom)
        gdf = gpd.GeoDataFrame({"geometry": geom_list})
        snapped_uds, snapped_gdf = xu.snap_to_grid(
            gdf, self.data.grid, max_snap_distance=max_snap_distance
        )
        snapped_gdf = snapped_gdf.set_crs(self.crs)
        return snapped_gdf

    # Internal functions
    def _get_datashader_dataframe(self):
        # Create a dataframe with line elements
        x1 = self.data.grid.edge_node_coordinates[:, 0, 0]
        x2 = self.data.grid.edge_node_coordinates[:, 1, 0]
        y1 = self.data.grid.edge_node_coordinates[:, 0, 1]
        y2 = self.data.grid.edge_node_coordinates[:, 1, 1]
        transformer = Transformer.from_crs(self.crs, 3857, always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        self.df = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))
