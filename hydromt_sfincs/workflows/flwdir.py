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
    ds,
    region=None,
    river_upa=25,
    river_len=0,
    dst_crs=None,
    return_river=True,
    flwdir_name="flwdir",
    uparea_name="uparea",
    logger=logger,
):
    """
    Returns the most downstream point locations where a river enters the region.
    Rivers are based on the flow direction data in the dataset and a minumum upstream
    area threshold.

    Parameters
    ----------
    ds: xarray.Dataset
        Hydrography raster data, should contain flwdir_name, uparea_name variables.
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
    src_crs = ds.raster.crs
    if dst_crs is None:
        dst_crs = src_crs
    if region is not None:
        da_mask = ds.raster.geometry_mask(region)
    else:
        da_mask = ds[flwdir_name] != ds[flwdir_name].raster.nodata
    da_mask_eroded = ndimage.binary_erosion(da_mask, structure=np.ones((3, 3)))
    da_mask_edge = np.logical_xor(da_mask_eroded, da_mask)

    # initialize flwdir with river cells inside region only
    rivmsk = np.logical_and(ds[uparea_name] >= river_upa, da_mask)
    flwdir = hydromt.flw.flwdir_from_da(ds[flwdir_name], mask=rivmsk)

    # set source points at inflowing region edge cells
    rivmsk = np.logical_and(rivmsk, flwdir.upstream_sum(rivmsk.astype(np.int8)) == 0)
    idxs_source = np.where(np.logical_and(rivmsk, da_mask_edge).values.ravel())[0]

    gdf_src = gpd.GeoDataFrame()
    if len(idxs_source) > 0:
        # filter based on length until pit
        if river_len > 0:
            path_length = flwdir.path(idxs=idxs_source, unit="m")[1]
            idxs_source = idxs_source[path_length > river_len]
        # get coordinates
        source_xy = flwdir.xy(idxs_source)
        gdf_src = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*source_xy), crs=src_crs
        ).to_crs(dst_crs)
        gdf_src["uparea"] = ds[uparea_name].values.flat[idxs_source]
    logger.debug(f"{len(idxs_source)} river inflow point locations found.")

    if return_river:
        logger.debug(f"Vectorize river.")
        feats = flwdir.streams()
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs).to_crs(dst_crs)
        return gdf_src, gdf_riv
    else:
        return gdf_src, None


def river_outflow_points(
    da_flwdir,
    region=None,
    river_upa=5.0,
    return_river=True,
    dst_crs=None,
    logger=logger,
):
    """
    Returns point locations where rivers leave the region.
    Rivers are based on the flow direction data in the dataset and a minumum upstream
    area threshold based on cells within the model domain.

    Parameters
    ----------
    da_flwdir: xarray.DataArray
        Flow directoin raster data.
    region: geopandas.GeoDataFrame
        Polygon of region of interest.
    river_upa: float, optional
        Mimimum upstream area threshold [km2] to define river cells, by default 5 km2.
    return_river: bool, optional
        If True, return a vectorized river GeoDataFrame

    Returns
    -------
    gdf_src, gdf_riv: geopandas.GeoDataFrame
        Inflow point and river line vector data.
    """
    src_crs = da_flwdir.raster.crs
    if dst_crs is None:
        dst_crs = src_crs
    if region is not None:
        da_mask = da_flwdir.raster.geometry_mask(region)
    else:
        da_mask = da_flwdir != da_flwdir.raster.nodata
    da_mask_eroded = ndimage.binary_erosion(da_mask, structure=np.ones((3, 3)))
    da_mask_edge = np.logical_xor(da_mask_eroded, da_mask)

    # initialize flwdir with all cells in domain and get uparea
    flwdir = hydromt.flw.flwdir_from_da(da_flwdir, mask=da_mask)
    rivmsk = flwdir.upstream_area(unit="km2") > river_upa

    # git pits at domain edge on flwdir grid with minimal uparea within model domain
    idxs0 = flwdir.idxs_pit
    select = np.logical_and(da_mask_edge.values.flat[idxs0], rivmsk.flat[idxs0])
    idxs_out = idxs0[select]
    logger.debug(f"{len(idxs_out)} river outflow point locations found.")

    gdf_out = gpd.GeoDataFrame()
    if len(idxs_out) > 0:
        gdf_out = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*flwdir.xy(idxs_out)), crs=src_crs
        ).to_crs(dst_crs)

    if return_river and np.any(rivmsk):
        logger.debug(f"Vectorize river for outflow points.")
        feats = hydromt.flw.flwdir_from_da(da_flwdir, mask=rivmsk).streams()
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs).to_crs(dst_crs)
        return gdf_out, gdf_riv
    else:
        return gdf_out, None
