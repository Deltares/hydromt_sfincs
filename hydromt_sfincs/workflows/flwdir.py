import numpy as np
import geopandas as gpd
from scipy import ndimage
import hydromt
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "river_inflow_points",
    "river_outflow_points",
]


def river_inflow_points(
    da_flwdir,
    da_uparea,
    region=None,
    river_upa=25,
    river_len=0,
    return_river=True,
    logger=logger,
):
    """
    Returns the locations where a river enters the region.
    Rivers are based on a minimum upstream area and minimum length within the model domain.

    Parameters
    ----------
    da_flwdir: xarray.DataArray
        D8 flow direction raster data.
    da_uparea: xarray.DataArray
        Upstream area raster data [km2].
    region: geopandas.GeoDataFrame, optional
        Polygon of region of interest. By default all valid cells in ds are used to
        determine the region of interest.
    river_upa: float, optional
        Mimimum upstream area threshold [km2] to define river cells, by default 25 km2.
    river_len: float, optional
        Mimimum river length [m] within the model domain to define river cells, by default 0 m.
    return_river: bool, optional
        If True, return a vectorized river GeoDataFrame

    Returns
    -------
    gdf_src, gdf_riv: geopandas.GeoDataFrame
        Inflow point and river line vector data.
    """
    src_crs = da_flwdir.raster.crs
    if region is not None:
        da_mask = da_flwdir.raster.geometry_mask(region)
    else:
        da_mask = da_flwdir != da_flwdir.raster.nodata
    da_mask_eroded = ndimage.binary_erosion(da_mask, structure=np.ones((3, 3)))
    da_mask_edge = np.logical_xor(da_mask_eroded, da_mask)

    # initialize flwdir with river cells inside region only
    rivmsk = np.logical_and(da_uparea >= river_upa, da_mask)
    flwdir = hydromt.flw.flwdir_from_da(da_flwdir, mask=rivmsk)

    # set source points at inflowing river edge cells with sufficient length in domain
    rivmsk_edge = np.logical_and(rivmsk, da_mask_edge).values
    checks = np.logical_and(flwdir.distnc > river_len, flwdir.n_upstream == 0)
    idxs_source = np.where(np.logical_and(checks, rivmsk_edge).ravel())[0]

    gdf_src = gpd.GeoDataFrame()
    if len(idxs_source) > 0:
        # get coordinates
        source_xy = flwdir.xy(idxs_source)
        gdf_src = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*source_xy), crs=src_crs)
        # add upstream area, important to match discharge model grid later
        gdf_src["uparea"] = da_uparea.values.flat[idxs_source]
    logger.debug(f"{len(idxs_source)} river inflow point locations found.")

    if return_river:
        logger.debug(f"Vectorize river.")
        feats = flwdir.streams()
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs)
        return gdf_src, gdf_riv
    else:
        return gdf_src, None


def river_outflow_points(
    da_flwdir,
    da_rivmsk,
    region=None,
    return_river=True,
    logger=logger,
):
    """
    Returns point locations where rivers leave the region.

    Parameters
    ----------
    da_flwdir: xarray.DataArray
        D8 flow direction raster data.
    da_rivmsk: xarray.DataArray
        Boolean river mask.
    region: geopandas.GeoDataFrame
        Polygon of region of interest.
    return_river: bool, optional
        If True, return a vectorized river GeoDataFrame

    Returns
    -------
    gdf_src, gdf_riv: geopandas.GeoDataFrame
        Inflow point and river line vector data.
    """
    src_crs = da_flwdir.raster.crs
    if region is not None:
        da_mask = da_flwdir.raster.geometry_mask(region)
    else:
        da_mask = da_flwdir != da_flwdir.raster.nodata
    da_mask_eroded = ndimage.binary_erosion(da_mask, structure=np.ones((3, 3)))
    da_mask_edge = np.logical_xor(da_mask_eroded, da_mask)

    # initialize flwdir with all cells in domain
    flwdir = hydromt.flw.flwdir_from_da(da_flwdir, mask=da_mask)
    idxs0 = flwdir.idxs_pit
    rivmsk = da_rivmsk.values

    # git pits at domain edge on flwdir grid with minimal uparea within model domain
    select = np.logical_and(da_mask_edge.values.flat[idxs0], rivmsk.flat[idxs0])
    idxs_out = idxs0[select]
    logger.debug(f"{len(idxs_out)} river outflow point locations found.")

    gdf_out = gpd.GeoDataFrame()
    if len(idxs_out) > 0:
        gdf_out = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*flwdir.xy(idxs_out)), crs=src_crs
        )

    if return_river and np.any(rivmsk):
        logger.debug(f"Vectorize river for outflow points.")
        feats = hydromt.flw.flwdir_from_da(da_flwdir, mask=rivmsk).streams()
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs)
        return gdf_out, gdf_riv
    else:
        return gdf_out, None
