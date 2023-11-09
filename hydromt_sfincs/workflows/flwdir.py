"""Flow direction and river network workflows for SFINCS models."""
import logging
from typing import Tuple

import geopandas as gpd
import hydromt
import numpy as np
import pyflwdir
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "river_boundary_points",
    "river_centerline_from_hydrography",
]


def river_centerline_from_hydrography(
    da_flwdir: xr.DataArray,
    da_uparea: xr.DataArray,
    river_upa: float = 10,
    river_len: float = 1e3,
    mask: gpd.GeoDataFrame = None,
) -> gpd.GeoDataFrame:
    """Returns the centerline of rivers based on a flow direction
    raster data (`da_flwdir`).

    Parameters
    ----------
    da_flwdir: xarray.DataArray
        D8 flow direction raster data.
    da_uparea: xarray.DataArray, optional
        River mask raster data. Used to mask da_flwdir, by default None.
    river_upa : float, optional
        Minimum upstream area threshold for rivers [km2], by default 10.0
    river_len: float, optional
        Mimimum river length [m] within the model domain to define river cells,
        by default 1000 m.
    mask: geopandas.GeoDataFrame, optional
        Polygon to clip river center lines before calculating the river length,
        by default None.

    Returns
    -------
    gdf_riv: geopandas.GeoDataFrame
        River line vector data.
    """
    # get river network from hydrography based on upstream area mask
    flwdir = hydromt.flw.flwdir_from_da(da_flwdir, mask=da_uparea >= river_upa)
    gdf_riv = gpd.GeoDataFrame.from_features(flwdir.streams(), crs=da_flwdir.raster.crs)
    # clip to mask and remove empty geometries
    if mask is not None:
        gdf_riv = gdf_riv.to_crs(mask.crs).clip(mask.unary_union)
    gdf_riv = gdf_riv[~gdf_riv.is_empty]
    # create river network from gdf to get distance from outlet 'rivlen'
    gdf_riv["rivlen"] = gdf_riv.geometry.length
    flwdir = pyflwdir.from_dataframe(gdf_riv.set_index("idx"), ds_col="idx_ds")
    gdf_riv["rivlen"] = flwdir.accuflux(gdf_riv["rivlen"].values, direction="down")
    gdf_riv = gdf_riv[gdf_riv["rivlen"] >= river_len]
    return gdf_riv


def river_boundary_points(
    region: gpd.GeoDataFrame,
    res: float,
    gdf_riv: gpd.GeoDataFrame = None,
    da_flwdir: xr.DataArray = None,
    da_uparea: xr.DataArray = None,
    river_upa: float = 10,
    river_len: float = 1e3,
    inflow: bool = True,
    reverse_river_geom: bool = False,
    logger: logging.Logger = logger,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Returns the locations where a river flows in (`inflow=True`)
    or out (`inflow=False`) of the model region.

    Rivers are based on either a river network vector data (`gdf_riv`) or
    a flow direction raster data (`da_flwdir`).

    Parameters
    ----------
    region: geopandas.GeoDataFrame
        Polygon of model region of interest.
    res: float
        Model resolution [m].
    gdf_riv: geopandas.GeoDataFrame, optional
        River network vector data, by default None.
    da_flwdir: xarray.DataArray, optional
        D8 flow direction raster data, by default None.
    da_uparea: xarray.DataArray, optional
        River upstream area raster data, by default None.
    river_upa : float, optional
        Minimum upstream area threshold for rivers [km2], by default 10.0
    river_len: float, optional
        Mimimum river length [m] within the model domain to define river cells, by default 1000 m.
    inflow: bool, optional
        If True, return inflow otherwise outflow boundary points, by default True.
    reverse_river_geom: bool, optional
        If True, assume that segments in 'rivers' are drawn from downstream to upstream.
        Only used if 'rivers' is not None, By default False

    Returns
    -------
    gdf_src, gdf_riv: geopandas.GeoDataFrame
        In-/outflow points and river line vector data.
    """
    if not isinstance(region, (gpd.GeoDataFrame, gpd.GeoSeries)) and np.all(
        region.geometry.type == "Polygon"
    ):
        raise ValueError("Boundary must be a GeoDataFrame of LineStrings.")
    if res > 0.01 and region.crs.is_geographic:
        # provide warning
        logger.warning(
            "The region crs is geographic, while the resolution seems to be in meters."
        )

    if gdf_riv is None and (da_flwdir is None or da_uparea is None):
        raise ValueError("Either gdf_riv or da_flwdir and da_uparea must be provided.")
    elif gdf_riv is None:  # get river network from hydrography
        gdf_riv = river_centerline_from_hydrography(
            da_flwdir, da_uparea, river_upa, river_len, mask=region
        )
    else:
        # clip river to model region
        gdf_riv = gdf_riv.to_crs(region.crs).clip(region.unary_union)
        # filter river network based on uparea and length
        if "uparea" in gdf_riv.columns:
            gdf_riv = gdf_riv[gdf_riv["uparea"] >= river_upa]
        if "rivlen" in gdf_riv.columns:
            gdf_riv = gdf_riv[gdf_riv["rivlen"] > river_len]
    # a positive dx results in a point near the start of the line (inflow)
    # a negative dx results in a point near the end of the line (outflow)
    dx = res / 5 if inflow else -res / 5
    if reverse_river_geom:
        dx = -dx

    # move point a bit into the model domain
    gdf_pnt = gdf_riv.interpolate(dx).to_frame("geometry")
    # keep points on boundary cells
    bnd = region.boundary.buffer(res).unary_union  # NOTE should be single geom
    gdf_pnt = gdf_pnt[gdf_pnt.within(bnd)].reset_index(drop=True)

    # add uparea attribute if da_uparea is provided
    if da_uparea is not None:
        gdf_pnt["uparea"] = da_uparea.raster.sample(gdf_pnt).values
        gdf_pnt = gdf_pnt.sort_values("uparea", ascending=False).reset_index(drop=True)
    if "rivwth" in gdf_riv.columns:
        gdf_pnt = hydromt.gis_utils.nearest_merge(
            gdf_pnt, gdf_riv, columns=["rivwth"], max_dist=10
        )

    return gdf_pnt, gdf_riv
