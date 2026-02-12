import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union

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
            x0, y0 = ls.coords[0]
            x1, y1 = ls.coords[-1]
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

            # extract endpoints
            coords = list(line.coords)
            p_on = Point(coords[0])
            p_in = Point(coords[-1])

            # slope
            if slope is None:
                if self.model.grid_type == "regular":
                    # --- regular grid ---
                    if (
                        hasattr(self.model, "subgrid")
                        and self.model.subgrid is not None
                        and self.model.subgrid.data is not None
                        and len(self.model.subgrid.data.data_vars) > 0
                        and "z" in self.model.subgrid.data
                    ):
                        # regular + subgrid
                        z = self.model.subgrid.data.z
                        z_in = z.sel(x=p_in.x, y=p_in.y, method="nearest").item()
                        z_on = z.sel(x=p_on.x, y=p_on.y, method="nearest").item()
                    else:
                        # regular only
                        z = self.model.grid.data.z
                        z_in = z.sel(x=p_in.x, y=p_in.y, method="nearest").item()
                        z_on = z.sel(x=p_on.x, y=p_on.y, method="nearest").item()

                else:
                    # --- quadtree grid ---
                    if (
                        hasattr(self.model, "quadtree_subgrid")
                        and self.model.quadtree_subgrid is not None
                        and self.model.quadtree_subgrid.data is not None
                        and len(self.model.quadtree_subgrid.data.data_vars) > 0
                        and "z" in self.model.quadtree_subgrid.data
                    ):
                        # quadtree + subgrid
                        zsrc = self.model.quadtree_subgrid.data.z.ugrid
                        z_in = zsrc.sel_points(x=p_in.x, y=p_in.y).item()
                        z_on = zsrc.sel_points(x=p_on.x, y=p_on.y).item()
                    else:
                        # quadtree only
                        zsrc = self.model.quadtree_grid.data.z.ugrid
                        z_in = zsrc.sel_points(x=p_in.x, y=p_in.y).item()
                        z_on = zsrc.sel_points(x=p_on.x, y=p_on.y).item()

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

        # # ---- helpers (defined once) ----
        # tol = 5.0  # endpoint snap tolerance in CRS units (meters)

        # def kpt(pt: Point):
        #     return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

        # def endpoints(ls: LineString):
        #     c = list(ls.coords)
        #     return Point(c[0]), Point(c[-1])

        # def other_node(seg, node):
        #     _, k0, k1, _ = seg
        #     return k1 if node == k0 else k0

        # def longest_path_from(node, prev_seg, segs, node2segs, visited, cap_len):
        #     best_len = 0.0
        #     best_path = []
        #     if cap_len is not None and cap_len <= 0:
        #         return 0.0, []

        #     for si in node2segs.get(node, []):
        #         if si == prev_seg or si in visited:
        #             continue

        #         geom, k0, k1, L = segs[si]
        #         visited.add(si)

        #         nxt = other_node(segs[si], node)
        #         new_cap = None if cap_len is None else cap_len - L
        #         sub_len, sub_path = longest_path_from(
        #             nxt, si, segs, node2segs, visited, new_cap
        #         )

        #         tot = L + sub_len
        #         if tot > best_len:
        #             best_len = tot
        #             best_path = [si] + sub_path

        #         visited.remove(si)

        #         if cap_len is not None and best_len >= cap_len:
        #             break

        #     return best_len, best_path

        # def interpolate_along_path(start_node, segs, path, dist):
        #     # your original is fine
        #     remaining = dist
        #     node = start_node
        #     for si in path:
        #         geom, k0, k1, L = segs[si]
        #         if remaining <= L:
        #             return (
        #                 geom.interpolate(remaining)
        #                 if node == k0
        #                 else geom.interpolate(L - remaining)
        #             )
        #         remaining -= L
        #         node = k1 if node == k0 else k0
        #     return Point(node[0], node[1])

        # rows = []

        # for _, prow in gdf_out_pts.iterrows():
        #     line = prow.geometry
            
        #     p = prow.geometry

        #     # candidates near p using spatial index
        #     try:
        #         cand_idx = list(sidx.nearest(p.bounds, num_results=25))
        #         cand = gdf_lines.iloc[cand_idx]
        #     except Exception:
        #         cand = gdf_lines

        #     if cand.empty:
        #         continue

        #     dists = cand.distance(p)
        #     line_best = cand.loc[dists.idxmin(), "geometry"]  # FIXED

        #     # snap outlet to line
        #     s0 = line_best.project(p)
        #     p_on = line_best.interpolate(s0)

        #     # local subnetwork
        #     local_gdf = gdf_lines[
        #         gdf_lines.intersects(p_on.buffer(internal_dist * 2.0))
        #     ].copy()
        #     if local_gdf.empty:
        #         # fallback: stay on best line only
        #         s_in = max(s0 - internal_dist, 0.0)
        #         p_in = line_best.interpolate(s_in)
        #     else:
        #         local_gdf = local_gdf.reset_index(drop=True)

        #         # starting segment in local set
        #         seg0 = int(local_gdf.distance(p_on).idxmin())
        #         geom0 = local_gdf.loc[seg0, "geometry"]

        #         s0_local = geom0.project(p_on)
        #         p_on = geom0.interpolate(s0_local)

        #         # build adjacency
        #         segs = []
        #         node2segs = {}
        #         for i, geom in enumerate(local_gdf.geometry):
        #             p0, p1 = endpoints(geom)
        #             k0, k1 = kpt(p0), kpt(p1)
        #             segs.append((geom, k0, k1, geom.length))
        #             node2segs.setdefault(k0, []).append(i)
        #             node2segs.setdefault(k1, []).append(i)

        #         # choose which direction to go AWAY from the outflow point p
        #         p0, p1 = endpoints(geom0)
        #         k0, k1 = segs[seg0][1], segs[seg0][2]

        #         # decide which endpoint is "downstream-ish" by proximity to p (not p_on)
        #         if p.distance(p0) <= p.distance(p1):
        #             downstream_node = k0
        #             upstream_node = k1
        #             dist_to_up_end = geom0.length - s0_local
        #             step_sign = +1  # move toward end
        #         else:
        #             downstream_node = k1
        #             upstream_node = k0
        #             dist_to_up_end = s0_local
        #             step_sign = -1  # move toward start

        #         remaining = float(internal_dist)

        #         if remaining <= dist_to_up_end:
        #             p_in = geom0.interpolate(s0_local + step_sign * remaining)
        #         else:
        #             remaining -= dist_to_up_end
        #             visited = {seg0}
        #             cap = remaining + internal_dist * 0.25
        #             _, best_path = longest_path_from(
        #                 upstream_node,
        #                 prev_seg=seg0,
        #                 segs=segs,
        #                 node2segs=node2segs,
        #                 visited=visited,
        #                 cap_len=cap,
        #             )
        #             p_in = interpolate_along_path(
        #                 upstream_node, segs, best_path, remaining
        #             )

        #     # slope
        #     if slope is None:
        #         z_in = self.model.quadtree_grid.data.z.ugrid.sel_points(
        #             x=p_in.x, y=p_in.y
        #         ).item()
        #         z_on = self.model.quadtree_grid.data.z.ugrid.sel_points(
        #             x=p_on.x, y=p_on.y
        #         ).item()
        #         slope_i = (z_in - z_on) / internal_dist
        #     else:
        #         slope_i = float(slope)

        #     rows.append(
        #         {
        #             "geometry": LineString([(p_on.x, p_on.y), (p_in.x, p_in.y)]),
        #             "slope": (
        #                 float(prow.get("slope", slope_i))
        #                 if hasattr(prow, "get")
        #                 else float(slope_i)
        #             ),
        #             "distance": (
        #                 float(prow.get("distance", internal_dist))
        #                 if hasattr(prow, "get")
        #                 else float(internal_dist)
        #             ),
        #         }
        #     )

        # gdf_boundary_lines = gpd.GeoDataFrame(rows, crs=gdf_out_pts.crs)

        # if debug:
        #     import matplotlib.pyplot as plt

        #     fig, ax = plt.subplots(1, 1)
        #     self.model.region.plot(ax=ax, color="red", alpha=0.3)
        #     gdf_lines.plot(ax=ax, color="gray", linewidth=1)
        #     gdf_boundary_lines.plot(ax=ax, color="blue", linewidth=2)
        #     plt.show()


