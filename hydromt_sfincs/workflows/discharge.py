import geopandas as gpd
import numpy as np
import xarray as xr
import logging
import hydromt
from scipy import ndimage

logger = logging.getLogger(__name__)

__all__ = ["snap_discharge", "river_inflow_points", "river_outflow_points"]


def snap_discharge(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    wdw: int = 1,
    max_error: float = 0.1,
    uparea_name: str = "uparea",
    discharge_name: str = "discharge",
    logger=logger,
) -> xr.DataArray:
    """
    Snaps point locations to grid cell with smallest absolute difference in upstream area
    within `wdw` around the original location. Both `ds` and `gdf` should have a upstream area
    variable called `uparea_name' with same unit.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset with discharge and optional uparea variable.
    gdf: geopandas.GeoDataFrame[Points]
        Dataframe with Point geometries of locations of interest.
    wdw: int, optional
        Window size in number of cells around discharge boundary locations
        to snap to, only used if ``uparea_fn`` is provided. By default 1.
    max_error: float, optional
        Maximum relative error between the discharge boundary location upstream area
        and the upstream area of the best fit grid cell, only used if "discharge"
        staticgeoms has a "uparea" column. By default 0.1.

    Returns
    -------
    da_q: xarray.DataArray
        DataArray with snapped discharge values per valid point location.
    """
    if uparea_name in ds and uparea_name in gdf.columns:
        ds_wdw = ds.raster.sample(gdf, wdw=wdw)
        logger.debug(
            f"Snapping {discharge_name} points to best matching uparea cell within wdw (size={wdw})."
        )
        upa0 = xr.DataArray(gdf[uparea_name], dims=("index"))
        upa_dff = np.abs(ds_wdw[uparea_name].where(ds_wdw["mask"]).load() - upa0) / upa0
        idx = upa_dff.argmin("wdw")
        valid = np.where(upa_dff.isel(wdw=idx) <= max_error)[0]
        if valid.size < gdf.index.size:
            logger.warning(
                f"{valid.size}/{gdf.index.size} {discharge_name} points with a "
                f"rel. upstream area error smaller or equal to {max_error:.2f}."
                " Removing boundary point(s) with larger error."
            )
        idx = idx.isel(index=valid)
        da_q = ds_wdw.isel(wdw=idx.load(), index=valid).reset_coords()[discharge_name]
    else:
        logger.debug(
            f"No {uparea_name} variable found in ds or gdf; "
            f"sampling {discharge_name} points from nearest grid cell."
        )
        da_q = ds.raster.sample(gdf).reset_coords()[discharge_name]

    return da_q


def river_inflow_points(
    ds,
    region,
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
    region: geopandas.GeoDataFrame
        Polygon of region of interest.
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
    if dst_crs is None:
        dst_crs = ds.raster.crs
    da_mask = ds.raster.geometry_mask(region)
    # initialize flwdir with river cells only (including outside basin)
    rivmsk = ds[uparea_name] >= river_upa
    flwdir = hydromt.flw.flwdir_from_da(ds[flwdir_name], mask=rivmsk)

    # set source points at headwater indices on river
    # find river cells outside model domain
    idxs0 = np.where(np.logical_and(rivmsk, ~da_mask).values.ravel())[0]
    # snap to first downsteam cell in model domain
    idxs_source = flwdir.snap(idxs=idxs0, mask=da_mask)[0]
    idxs_source = np.unique(idxs_source[da_mask.values.flat[idxs_source]])

    gdf_src = gpd.GeoDataFrame()
    if len(idxs_source) > 0:
        # keep only most downstream source point on each stream
        # check if downstream path from any source points intersects with another point
        source_mask = np.full(flwdir.size, False, bool)
        source_mask[idxs_source] = True
        paths = flwdir.path(
            idxs=idxs_source, unit="m", mask=~da_mask
        )  # returns paths and lengths
        valid = np.array(
            [l > river_len and ~np.any(source_mask[p[1:]]) for p, l in zip(*paths)]
        )
        idxs_source = idxs_source[valid]
        # get coordinates
        source_xy = flwdir.xy(idxs_source)
        gdf_src = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*source_xy), crs=ds.raster.crs
        ).to_crs(dst_crs)
        gdf_src["uparea"] = ds[uparea_name].values.flat[idxs_source]
    logger.debug(f"{len(idxs_source)} river inflow point locations found.")

    if return_river:
        logger.debug(f"Vectorize river.")
        feats = flwdir.vectorize(mask=np.logical_and(da_mask, rivmsk).values)
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=ds.raster.crs).to_crs(
            dst_crs
        )
        return gdf_src, gdf_riv
    else:
        return gdf_src


def river_outflow_points(
    ds,
    region,
    river_upa=25,
    return_river=True,
    dst_crs=None,
    flwdir_name="flwdir",
    uparea_name="uparea",
    logger=logger,
):
    """
    Returns point locations where rivers leave the region.
    Rivers are based on the flow direction data in the dataset and a minumum upstream
    area threshold.

    Parameters
    ----------
    ds: xarray.Dataset
        Hydrography raster data, should contain flwdir_name, uparea_name variables.
    region: geopandas.GeoDataFrame
        Polygon of region of interest.
    river_upa: float, optional
        Mimimum upstream area threshold [km2] to define river cells, by default 25 km2.
    return_river: bool, optional
        If True, return a vectorized river GeoDataFrame

    Returns
    -------
    gdf_src, gdf_riv: geopandas.GeoDataFrame
        Inflow point and river line vector data.
    """
    if dst_crs is None:
        dst_crs = ds.raster.crs
    da_mask = ds.raster.geometry_mask(region)
    # initialize flwdir with river cells only (including outside basin)
    rivmsk = np.logical_and(da_mask, ds[uparea_name] >= river_upa)
    flwdir = hydromt.flw.flwdir_from_da(ds[flwdir_name], mask=rivmsk)

    # git pits at domain edge on flwdir grid
    idxs0 = flwdir.idxs_pit
    da_mask_eroded = ndimage.binary_erosion(da_mask, structure=np.ones((3, 3)))
    da_mask_edge = np.logical_xor(da_mask_eroded, da_mask)
    idxs_outflw = idxs0[da_mask_edge.values.flat[idxs0]]
    logger.debug(f"{len(idxs_outflw)} river outflow point locations found.")

    gdf_out = gpd.GeoDataFrame()
    if len(idxs_outflw) > 0:
        gdf_out = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*flwdir.xy(idxs_outflw)), crs=ds.raster.crs
        ).to_crs(dst_crs)

    if return_river:
        logger.debug(f"Vectorize river for outflow points.")
        feats = flwdir.vectorize()
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=ds.raster.crs).to_crs(
            dst_crs
        )
        return gdf_out, gdf_riv
    else:
        return gdf_out
