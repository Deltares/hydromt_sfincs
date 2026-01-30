import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

import geopandas as gpd
import pandas as pd
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

    def write_bdr_points(
        self, fn: Union[str, Path], gdf_bdr: gpd.GeoDataFrame, fmt="%.1f"
    ) -> None:
        """Write SFINCS downstream river boundary points file (.bdr).

        Each row:
        xbdr ybdr xbdr_in ybdr_in slope distance

        NOTE: This version expects geometry to be LineString with 2 vertices:
        - first vertex: boundary point
        - second vertex: inland control point
        """
        gdf = gdf_bdr.copy()
        if gdf.empty:
            Path(fn).parent.mkdir(parents=True, exist_ok=True)
            Path(fn).write_text("")
            return

        if not all(gdf.geom_type == "LineString"):
            raise ValueError("gdf_bdr geometry must be LineString (boundary->inland).")

        # ensure each line has at least 2 coordinates (we'll use first & last)
        def _endpoints(line: LineString):
            coords = list(line.coords)
            if len(coords) < 2:
                raise ValueError("Each LineString must have at least 2 vertices.")
            (xbdr, ybdr) = coords[0]
            (x_in, y_in) = coords[-1]
            return xbdr, ybdr, x_in, y_in

        endpoints = gdf.geometry.apply(_endpoints)
        gdf["xbdr"] = endpoints.apply(lambda t: t[0])
        gdf["ybdr"] = endpoints.apply(lambda t: t[1])
        gdf["x_bdr_in"] = endpoints.apply(lambda t: t[2])
        gdf["y_bdr_in"] = endpoints.apply(lambda t: t[3])

        # required columns
        required = ["slope", "distance"]
        missing = [c for c in required if c not in gdf.columns]
        if missing:
            raise ValueError(f"Missing required columns in gdf_bdr: {missing}")

        # order columns as SFINCS expects
        gdf = gdf[["xbdr", "ybdr", "x_bdr_in", "y_bdr_in", "slope", "distance"]]

        # format coords
        for col in ["xbdr", "ybdr", "x_bdr_in", "y_bdr_in"]:
            gdf[col] = gdf[col].apply(lambda x: fmt % float(x))

        gdf["slope"] = gdf["slope"].apply(lambda x: f"{float(x):.6f}")
        gdf["distance"] = gdf["distance"].apply(lambda x: f"{float(x):.3f}")

        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        gdf.to_csv(fn, sep=" ", index=False, header=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS drainage structures (.drn) file,
        and make sure bdrfile is in config (if it was not already set)."""

        # check that write mode is on
        self.root._assert_write_mode()

        # check if data present:
        if self.data.empty:
            logger.debug("No drainage structures data available to write.")
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
        self.write_bdr_points(abs_file_path, self.data, fmt=fmt)

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

        # Clip geometries outside of model region:
        within = gdf.within(self.model.region.union_all())

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
        gdf_out_pts: gpd.GeoDataFrame,
        gdf_riv: gpd.GeoDataFrame,
        internal_dist: float = 1000.0,
        slope: float = None,
        reverse_river_geom: bool = False,
        merge: bool = False,
    ) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame for SFINCS .bdr output.

        Creates downstream boundary points with internal control points,
        slope, and distance.

        Parameters
        ----------
        gdf_out_pts : gpd.GeoDataFrame
            Outflow points (Point geometries).
        gdf_riv : gpd.GeoDataFrame
            River centerlines (LineString geometries).
        internal_dist : float, optional
            Distance [m] from boundary point to internal control point, by default 1000.0
        slope : float, optional
            Slope value to use for all outflow points. If None, slope is computed
            from the model elevation data, by default None.

        Output columns:
        geometry (LineString) = downstream boundary point (xbdr,ybdr) to internal control point (xbdr_in,ybdr_in)
        x_bdr_in, y_bdr_in = internal control point coords
        slope, distance

        """
        if gdf_out_pts.empty:
            return gdf_out_pts.copy()

        if not all(gdf_out_pts.geom_type == "Point"):
            raise ValueError(
                "gdf_out_pts must contain Point geometries (not polygons)."
            )

        gdf_lines = gdf_riv[["geometry"]].copy().reset_index(drop=True)

        rows = []
        for _, prow in gdf_out_pts.iterrows():
            p = prow.geometry

            # find nearest river line (simple; replace with sjoin_nearest for speed if desired)
            dmin = np.inf
            line_best = None
            for _, lrow in gdf_lines.iterrows():
                d = lrow.geometry.distance(p)
                if d < dmin:
                    dmin = d
                    line_best = lrow.geometry

            if line_best is None:
                continue

            # snap outflow point to river line (nice-to-have)
            s0 = line_best.project(p)
            p_on = line_best.interpolate(s0)

            # pick internal point upstream/downstream depending on line direction
            if reverse_river_geom:
                s_in = min(s0 + internal_dist, line_best.length)
            else:
                s_in = max(s0 - internal_dist, 0.0)
            p_in = line_best.interpolate(s_in)

            if slope is None:
                z_in = self.model.quadtree_grid.data.z.ugrid.sel_points(
                    x=p_in.x, y=p_in.y
                ).item()
                z_on = self.model.quadtree_grid.data.z.ugrid.sel_points(
                    x=p_on.x, y=p_on.y
                ).item()

                slope_i = (z_in - z_on) / internal_dist

                logger.info(
                    f"Computed slope={slope_i:.4f} for outflow point at {p_on.x:.1f}, {p_on.y:.1f}"
                )
            else:
                slope_i = float(slope)

            rows.append(
                {
                    "geometry": LineString([(p_on.x, p_on.y), (p_in.x, p_in.y)]),
                    "slope": (
                        float(prow.get("slope", slope_i))
                        if hasattr(prow, "get")
                        else float(slope_i)
                    ),
                    "distance": (
                        float(prow.get("distance", internal_dist))
                        if hasattr(prow, "get")
                        else float(internal_dist)
                    ),
                }
            )

        gdf_boundary_lines = gpd.GeoDataFrame(rows, crs=gdf_out_pts.crs)

        self.set(gdf_boundary_lines, merge=merge)

        # set config
        self.model.config.set("bdrfile", "sfincs.bdr")

        return gdf_boundary_lines
