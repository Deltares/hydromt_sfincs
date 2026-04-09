import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union, List

import numpy as np

import geopandas as gpd
import pandas as pd
from shapely import Point, node
from shapely.geometry import LineString

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsRiverBoundaryPoints(ModelComponent):
    """SFINCS river boundary points component.

    This component handles reading, writing, and creating river boundary points
    in a SFINCS model.
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.bdr"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """River boundary points data, returns geopandas.GeoDataFrame"""
        if self._data is None:
            self._initialize()
        return self._data

    @property
    def nr_points(self) -> int:
        """
        Return the number of point locations currently stored.
        """
        if hasattr(self.data, "index"):
            return len(self.data.index)
        return 0

    def _initialize(self, skip_read: bool = False) -> None:
        """Initialize river boundary points."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS river boundary points (.bdr) file. Filename is obtained from config if not provided."""

        # check that read mode is on
        self.root._assert_read_mode()

        # get absolute file path and set it in config if bdrfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "bdrfile", value=filename
        )

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(
                f"River boundary points file not found: {abs_file_path}"
            )

        # Read input file:
        # TODO we can move the utils to here, since only used here?
        gdf = utils.read_bdr(abs_file_path, crs=self.model.crs)

        # Add to self._data
        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS river boundary (.bdr) file,
        and make sure bdrfile is in config (if it was not already set)."""

        # check that write mode is on
        self.root._assert_write_mode()

        # check if data present:
        if self.data.empty:
            logger.debug("No river boundary points data available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            key="bdrfile", value=filename, default="sfincs.bdr"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # TODO we can move the utils to here, since only used here?
        utils.write_bdr_points(abs_file_path, self.data, fmt=fmt)

        # write also as geojson:
        if self.model.write_gis:
            utils.write_vector(
                self.data,
                name="bdr",
                root=join(self.model.root.path, "gis"),
                logger=logger,
            )

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SFINCS river boundary points.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with river boundary points to self.data.
            Note that the gdf should have the same CRS as the model.
        merge: bool
            Merge with existing river boundary points. If False, overwrite existing river boundary points.

        .. note::
            When directly using the set method, the GeoDataFrame needs to be in the same CRS as SFINCS model.
        """

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("River boundary points must be of type LineString.")
        if not gdf.crs == self.model.crs:
            raise ValueError(
                f"River boundary points CRS {gdf.crs} does not match model CRS {self.model.crs}."
            )

        region = self.model.region.union_all()

        def endpoints_inside(ls):
            x0, y0, *_ = ls.coords[0]
            x1, y1, *_ = ls.coords[-1]
            return Point(x0, y0).covered_by(region) and Point(x1, y1).covered_by(region)

        within = gdf.geometry.apply(endpoints_inside)

        if within.any() == True:
            if within.all() == False:
                # keep geometries that fall within region
                gdf = gdf[within]

                # write away the names of geometries that are removed
                if "name" in gdf.columns:
                    gdf_name = gdf.loc[~within, "name"]
                    logger.info(
                        "Some of the river boundary points fall out of model domain. Removing points: "
                        + str(gdf_name.values)
                    )
        else:
            raise ValueError("None of river boundary points fall within model domain.")
        if merge and not self.data.empty:
            gdf0 = self.data
            # add the new data behind the original
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new river boundary points to existing ones.")

        self._data = gdf  # set gdf in self.data

    @hydromt_step
    def create(
        self,
        locations,
        internal_dist: float = 1000.0,
        slope: float = None,
        merge: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        gdf_out_pts = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, **kwargs
        ).to_crs(self.model.crs)

        if not (gdf_out_pts.geom_type == "LineString").all():
            raise ValueError("gdf_out_pts must contain LineString geometries.")

        rows = []
        for _, prow in gdf_out_pts.iterrows():
            line = prow.geometry

            # extract endpoints and force 2D
            coords = [(x, y) for x, y, *_ in line.coords]
            line = LineString(coords)

            p_on = Point(coords[0])
            p_in = Point(coords[-1])

            gdf_on = gpd.GeoDataFrame(geometry=[p_on], crs=gdf_out_pts.crs)
            gdf_in = gpd.GeoDataFrame(geometry=[p_in], crs=gdf_out_pts.crs)

            # slope
            if slope is None:
                if self.model.grid_type == "regular":
                    # --- regular grid ---
                    if (len(self.model.subgrid.data.data_vars) > 0):
                        # regular + subgrid
                        z = self.model.subgrid.data.z_zmin
                    else:
                        # regular only
                        z = self.model.grid.data.dep

                    z_in = z.raster.sample(gdf_in).item()
                    z_on = z.raster.sample(gdf_on).item()

                else:
                    # --- quadtree grid ---
                    if (len(self.model.quadtree_subgrid.data.data_vars) > 0):
                        # quadtree + subgrid
                        z = self.model.quadtree_subgrid.data.z_zmin.ugrid
                    else:
                        # quadtree only
                        z = self.model.quadtree_grid.data.z.ugrid
                    
                    z_in = z.sel_points(x=p_in.x, y=p_in.y).item()
                    z_on = z.sel_points(x=p_on.x, y=p_on.y).item()
                
                denom = internal_dist  # or line.length if preferred
                slope_i = 0.0 if denom == 0 else (z_in - z_on) / denom

            else:
                slope_i = float(slope)

            rows.append(
                {
                    "geometry": line,
                    "slope": float(prow.get("slope", slope_i)),
                    "distance": float(prow.get("distance", internal_dist)),
                }
            )

        gdf_boundary_lines = gpd.GeoDataFrame(rows, crs=gdf_out_pts.crs)
        self.set(gdf_boundary_lines, merge=merge)
        self.model.config.set("bdrfile", "sfincs.bdr")
        return gdf_boundary_lines

    def delete(self, index: Union[int, List[int]]):
        """
        Delete one or more point indices from the internal dataset.

        Parameters
        ----------
        index : int or list of int
            Index or list of indices to remove.
        """
        if self.nr_points == 0:
            return
        if not isinstance(index, list):
            index = [index]
        if any(x > (self.nr_points - 1) for x in index):
            raise ValueError("One of the indices exceeds length of index range!")
        self._data = self.data.drop(index=index)

    def clear(self):
        """
        Remove all stored points and reset internal dataset to empty.
        """
        self._data = gpd.GeoDataFrame()  # reset to empty GeoDataFrame
        self.model.config.set("bdrfile", None)
