import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import numpy as np
from shapely import Point
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils, workflows

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(__name__)


class SfincsRivers(ModelComponent):
    """This class contains functions to create and manage river inflow/outflow points in the SFINCS model.
    The methods in this class change the model mask and add discharge points where rivers enter the model domain.
    The rivers themselves are not used by the SFINCS model, but useful for visualization.
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """River geometry data.

        Return gpd.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self):
        """Initialize the SfincsRivers component."""
        # get the data from the model
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            # self._data["downstream_boundary_points"] = gpd.GeoDataFrame()

    # Original HydroMT-SFINCS setup_ functions:
    # setup_river_inflow
    # setup_river_outflow
    # FIXME - also functions like burn in river???
    # FIXME - also new functions to read/process/burn in river cross-section data???

    def read(self):
        pass

    def write(self):
        """Write the river inflow data to a gis-file. Note: this is not used by SFINCS, but useful for model visualization."""

        # bdrpts = self.data["downstream_boundary_points"]

        if self.data.empty:
            return

        # write also as geojson:
        if self.model.write_gis:
            utils.write_vector(
                self.data,
                name="river_centerlines",
                root=join(self.model.root.path, "gis"),
                logger=logger,
            )

    @hydromt_step
    def create_river_inflow(
        self,
        rivers: Union[str, Path, gpd.GeoDataFrame] = None,
        hydrography: Union[str, Path, xr.Dataset] = None,
        buffer: float = 200,
        river_upa: float = 10.0,
        river_len: float = 1e3,
        river_width: float = 500,
        merge: bool = False,
        first_index: int = 1,
        keep_rivers_geom: bool = False,
        reverse_river_geom: bool = False,
        src_type: str = "inflow",
    ):
        """Setup discharge (src) points where a river enters the model domain.

        If `rivers` is not provided, river centerlines are extracted from the
        `hydrography` dataset based on the `river_upa` threshold.

        Waterlevel or outflow boundary cells intersecting with the river
        are removed from the model mask.

        Discharge is set to zero at these points, but can be updated
        using the `setup_discharge_forcing` or `setup_discharge_forcing_from_grid` methods.

        Note: this method assumes the rivers are directed from up- to downstream. Use
        `reverse_river_geom=True` if the rivers are directed from downstream to upstream.

        Adds model layers:

        * **dis** forcing: discharge forcing
        * **mask** map: SFINCS mask layer (only if `river_width` > 0)
        * **rivers_inflow** geoms: river centerline (if `keep_rivers_geom`; not used by SFINCS)

        Parameters
        ----------
        rivers : str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for river centerline data.
            If present, the 'uparea' and 'rivlen' attributes are used.
        hydrography: str, Path, xr.Dataset optional
            Path, data source name, or a xarray raster object for hydrography data.

            * Required layers: ['uparea', 'flwdir'].
        buffer: float, optional
            Buffer around the model region boundary to define in/outflow points [m],
            by default 200 m. We suggest to use a buffer of at least twice the hydrography
            resolution. Inflow points are moved to a downstreawm confluence if within the buffer.
        river_upa : float, optional
            Minimum upstream area threshold for rivers [km2], by default 10.0
        river_len: float, optional
            Mimimum river length within the model domain threshhold [m], by default 1 km.
        river_width: float, optional
            Estimated constant width [m] of the inflowing river. Boundary cells within
            half the width are forced to be closed (mask = 1) to avoid instabilities with
            nearby open or waterlevel boundary cells, by default 500 m.
        merge: bool, optional
            If True, merge rivers source points with existing points, by default False.
        first_index: int, optional
            First index for the river source points, by default 1.
        keep_rivers_geom: bool, optional
            If True, keep a geometry of the rivers "rivers_inflow" in geoms. By default False.
        reverse_river_geom: bool, optional
            If True, assume that segments in 'rivers' are drawn from downstream to upstream.
            Only used if 'rivers' is not None, By default False
        src_type: {'inflow', 'headwater'}, optional
            Source type, by default 'inflow'
            If 'inflow', return points where the river flows into the model domain.
            If 'headwater', return all headwater (including inflow) points within the model domain.

        See Also
        --------
        setup_discharge_forcing
        setup_discharge_forcing_from_grid
        """

        # FIXME what to do with these variables
        all_touched = False

        # get hydrography data
        da_uparea = None
        if hydrography is not None:
            ds = self.data_catalog.get_rasterdataset(
                hydrography,
                bbox=self.model.bbox,
                variables=["uparea", "flwdir"],
                buffer=5,
            )
            da_uparea = ds["uparea"]  # reused in river_source_points

        # FIXME reuse from inflow/outflow and get from self.data
        # get river centerlines
        # if (
        #     isinstance(rivers, str)
        #     and rivers == "rivers_outflow"
        #     and rivers in self.geoms
        # ):
        #     # reuse rivers from setup_river_in/outflow
        #     gdf_riv = self.geoms[rivers]
        # el
        if rivers is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                rivers, geom=self.model.region
            ).to_crs(self.model.crs)
        elif hydrography is not None:
            gdf_riv = workflows.river_centerline_from_hydrography(
                da_flwdir=ds["flwdir"],
                da_uparea=da_uparea,
                river_upa=river_upa,
                river_len=river_len,
                gdf_mask=self.model.region,
            )
        elif hydrography is None:
            raise ValueError("Either hydrography or rivers must be provided.")

        # get river inflow / headwater source points
        gdf_src = workflows.river_source_points(
            gdf_riv=gdf_riv,
            gdf_mask=self.model.region,
            src_type=src_type,
            buffer=buffer,
            river_upa=river_upa,
            river_len=river_len,
            da_uparea=da_uparea,
            reverse_river_geom=reverse_river_geom,
            logger=logger,
        )
        if gdf_src.empty:
            logger.info("No river source points found.")
            return

        # set forcing src pnts
        # TODO is this first_index variable still needed here? This depends on how we filter index/name/location
        gdf_src.index = gdf_src.index + first_index
        self.model.discharge_points.set_locations(
            gdf=gdf_src.copy(deep=True), merge=merge
        )

        # set river
        if keep_rivers_geom:
            self._data = gdf_riv

        # update mask if river_width > 0
        if "rivwth" in gdf_src.columns:
            river_width = gdf_src["rivwth"].fillna(river_width)

        if self.model.grid_type == "quadtree":
            logger.warning(
                "For quadtree grids, the mask is not updated around river source points."
                "Please carefully check the model mask."
            )
            # TODO - add for quadtree grids
        else:
            if np.any(river_width > 0) and np.any(self.model.grid.mask > 1):
                # apply buffer
                gdf_src["geometry"] = gdf_src.buffer(river_width / 2)
                # find intersect of buffer and model grid
                da_mask = self.model.grid.mask
                da_include = da_mask.raster.geometry_mask(
                    gdf_src, all_touched=all_touched
                )
                reset_msk = np.logical_and(da_include, da_mask > 1)
                # update model mask
                n = int(reset_msk.sum().item())
                if n > 0:
                    da_mask = da_mask.where(~reset_msk, np.uint8(1))
                    self.model.grid.set(da_mask, "msk")
                    logger.info(f"Boundary cells (n={n}) updated around src points.")

    def create_river_outflow(
        self,
        rivers: Union[str, Path, gpd.GeoDataFrame] = None,
        hydrography: Union[str, Path, xr.Dataset] = None,
        river_upa: float = 10.0,
        river_len: float = 1e3,
        river_width: float = 500,
        gdf_out_include: gpd.GeoDataFrame = None,
        buffer: float = None,
        internal_dist: float = 1000.0,
        slope: float = None,
        keep_rivers_geom: bool = False,
        reset_bounds: bool = False,
        reverse_river_geom: bool = False,
    ):
        """Setup open boundary cells (mask=5) where a river flows
        out of the model domain.

        If `rivers` is not provided, river centerlines are extracted from the
        `hydrography` dataset based on the `river_upa` threshold.

        Downstream river mask that intersect with centerlines are set to
        open boundary cells (mask=5). Downstream boundary points are stored
        in the `downstream_boundary_points` data for SFINCS .bdr file output.

        Note: this method assumes the rivers are directed from up- to downstream.

        Adds / edits model layers:

        * **msk** map: edited by adding downstream points (msk=5)
        * **downstream_boundary_points** data: GeoDataFrame with downstream boundary points for SFINCS .bdr file
        * **rivers_outflow** geoms: river centerline (if `keep_rivers_geom`; not used by SFINCS)

        Parameters
        ----------
        rivers : str, Path, gpd.GeoDataFrame, optional
            Path, data source name, or geopandas object for river centerline data.
            If present, the 'uparea' and 'rivlen' attributes are used.
        hydrography: str, Path, xr.Dataset optional
            Path, data source name, or a xarray raster object for hydrography data.

            * Required layers: ['uparea', 'flwdir'].
        river_upa : float, optional
            Minimum upstream area threshold for rivers [km2], by default 10.0
        river_len: float, optional
            Mimimum river length within the model domain threshhold [m], by default 1000 m.
        river_width: int, optional
            The width [m] of the open boundary cells in the SFINCS msk file.
            By default 500m, i.e.: 250m to each side of the outflow location.
        gdf_out_include: gpd.GeoDataFrame, optional
            Outflow geometries to include as outflow boundary cells. This overrides the detected outflow points.
        buffer: float, optional
            Buffer around the model region boundary to define outflow points [m],
        internal_dist : float, optional
            Minimum internal distance [m] used when selecting outflow locations,
            by default 1000.0.
        slope : float, optional
            Minimum slope threshold used in outflow detection.
        append_bounds: bool, optional
            If True, write new outflow boundary cells on top of existing. If False (default),
            first reset existing outflow boundary cells to normal active cells.
        keep_rivers_geom: bool, optional
            If True, keep a geometry of the rivers "rivers_outflow" in geoms. By default False.
        reset_bounds: bool, optional
            If True, reset existing outlfow boundary cells before setting new boundary cells,
            by default False.
        btype: {'waterlevel', 'outflow', 'downstream'}, optional
            Boundary type
        reverse_river_geom: bool, optional
            If True, assume that segments in 'rivers' are drawn from downstream to upstream.
            Only used if rivers is not None, By default False

        See Also
        --------
        setup_mask_bounds
        """
        # get hydrography data
        da_uparea = None
        if hydrography is not None:
            ds = self.data_catalog.get_rasterdataset(
                hydrography,
                bbox=self.model.bbox,
                variables=["uparea", "flwdir"],
                buffer=10000,
            )
            da_uparea = ds["uparea"]  # reused in river_source_points
        # get river centerlines
        if not self.data.empty and rivers is None and hydrography is None:
            gdf_riv = self.data
            logger.info("Reusing existing river centerlines.")
        elif rivers is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                rivers, geom=self.model.region
            ).to_crs(self.model.crs)
        elif hydrography is not None:
            gdf_riv = workflows.river_centerline_from_hydrography(
                da_flwdir=ds["flwdir"],
                da_uparea=da_uparea,
                river_upa=river_upa,
                river_len=river_len,
                gdf_mask=self.model.region,
            )
        elif hydrography is None:
            raise ValueError(
                "No centerlines present in component.Either hydrography or rivers must be provided."
            )

        # get river inflow / headwater source points
        gdf_out = workflows.river_source_points(
            gdf_riv=gdf_riv,
            gdf_mask=self.model.region,
            src_type="outflow",
            buffer=buffer,
            river_upa=river_upa,
            river_len=river_len,
            da_uparea=da_uparea,
            reverse_river_geom=reverse_river_geom,
            internal_distance=internal_dist,
            logger=logger,
        )

        if gdf_out.empty:
            self.logger.info("No river outflow points found.")
            return

        gdf_out_lines = gdf_out.copy()

        gdf_out["geometry"] = gdf_out.geometry.apply(
            lambda geom: Point(geom.coords[0])
            if geom.geom_type == "LineString"
            else geom
        )

        if len(gdf_out) > 0:
            if "rivwth" in gdf_out.columns:
                river_width = gdf_out["rivwth"].fillna(river_width)
            gdf_out["geometry"] = gdf_out.buffer(river_width / 2)

        # update mask
        n = len(gdf_out.index)
        logger.info(f"Found {n} valid river outflow points.")
        if n > 0:
            if self.model.grid_type == "regular":
                self.model.mask.create_boundary(
                    btype="downstream",
                    include_polygon=gdf_out,
                    reset_bounds=reset_bounds,
                )
            elif self.model.grid_type == "quadtree":
                self.model.quadtree_mask.create_boundary(
                    btype="downstream",
                    include_polygon=gdf_out,
                    reset_bounds=reset_bounds,
                )
        # elif reset_bounds:
        #     self.mask.create_boundary(btype=btype, reset_bounds=reset_bounds)

        self.model.river_boundary_points.create(
            locations=gdf_out_lines,
            internal_dist=internal_dist,
            slope=slope,
            merge=False,
        )

        # keep river centerlines
        if keep_rivers_geom and len(gdf_riv) > 0:
            self._data = gdf_riv


# %% core HydroMT-SFINCS functions:
# _initialize
# create:
# create_inflow
# create_outflow
# clear

# %% DDB GUI focused additional functions:
# - yet unsupported in DDB-
