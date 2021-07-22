import geopandas as gpd
import numpy as np
import xarray as xr
import logging

logger = logging.getLogger(__name__)


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
