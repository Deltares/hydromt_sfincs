import numpy as np
import geopandas as gpd
from scipy import ndimage
import hydromt
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "river_boundary_points",
]


def river_boundary_points(
    da_flwdir,
    da_uparea,
    region=None,
    river_upa=25.0,
    river_len=1e3,
    btype="inflow",
    return_river=True,
    logger=logger,
):
    """
    Returns the locations where a river flows in (`btype='inflow'`) or out (`btype='outflow'`) of the region.
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
        Mimimum river length [m] within the model domain to define river cells, by default 1000 m.
    btype: {'inflow', 'outflow'}
        Return inflow  (default) or outflow boundary points.
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
    if river_len > 0:
        dx_headwater = np.where(flwdir.n_upstream == 0, flwdir.distnc, 0)
        rivlen = flwdir.fillnodata(dx_headwater, nodata=0, direction="down", how="max")
        rivmsk = np.logical_and(rivmsk, rivlen >= river_len)

    if btype == "inflow":
        bcells = np.logical_and.reduce((flwdir.n_upstream == 0, rivmsk, da_mask_edge))
    elif btype == "outflow":
        bcells = np.logical_and.reduce((flwdir.distnc == 0, rivmsk, da_mask_edge))

    # set source points at inflowing river edge cells with sufficient length in domain
    idxs = np.where(bcells.ravel())[0]
    gdf_src = gpd.GeoDataFrame()
    if len(idxs) > 0:
        # get coordinates
        source_xy = flwdir.xy(idxs)
        gdf_src = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*source_xy), crs=src_crs)
        # add upstream area, important to match discharge model grid later
        gdf_src["uparea"] = da_uparea.values.flat[idxs]
    logger.debug(f"{len(idxs)} river {btype} point locations found.")

    if return_river:
        logger.debug(f"Vectorize river.")
        feats = flwdir.streams(mask=rivmsk)
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs)
        return gdf_src, gdf_riv
    else:
        return gdf_src, None
