"""Workflows for discharge boundary conditions."""
import logging

import geopandas as gpd
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "snap_discharge",
]


def snap_discharge(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    wdw: int = 1,
    rel_error: float = 0.05,
    abs_error: float = 50,
    uparea_name: str = "uparea",
    discharge_name: str = "discharge",
    logger=logger,
) -> xr.DataArray:
    """
    Snaps point locations to grid cell with smallest difference in upstream area
    within `wdw` around the original location if the local cell does not meet the
    error criteria. Both the upstream area variable named ``uparea_name`` in
    ``ds`` and ``gdf`` as well as ``abs_error`` should have the same unit (typically km2).

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset with discharge and optional uparea variable.
    gdf: geopandas.GeoDataFrame[Points]
        Dataframe with Point geometries of locations of interest.
    wdw: int, optional
        Window size in number of cells around discharge boundary locations
        to snap to, only used if ``uparea_fn`` is provided. By default 1.
    rel_error, abs_error: float, optional
        Maximum relative error (default 0.05) and absolute error (default 50 km2)
        between the discharge boundary location upstream area and the upstream area of
        the best fit grid cell, only used if "discharge" staticgeoms has a "uparea" column.

    Returns
    -------
    ds: xarray.Dataset
        snapped dataset
    """
    ds_wdw = ds.raster.sample(gdf, wdw=wdw)
    # check if valid discharge
    valid = ds_wdw[discharge_name].notnull().any("time")
    if uparea_name in ds and uparea_name in gdf.columns:
        logger.debug(
            f"Snapping {discharge_name} points to best matching uparea cell within wdw (size={wdw})."
        )
        # find cells in window with smallest difference in uparea
        upa0 = xr.DataArray(gdf[uparea_name], dims=("index"))
        upa_dff = np.abs(
            ds_wdw[uparea_name].where(ds_wdw[uparea_name] > 0).load() - upa0
        )
        i_wdw = upa_dff.fillna(np.inf).argmin("wdw")
        # find valid cells based on error criteria
        upa_check = np.logical_or((upa_dff / upa0) <= rel_error, upa_dff <= abs_error)
        valid = np.logical_and(valid, upa_check)
    else:
        logger.debug(
            f"No {uparea_name} variable found in ds or gdf; "
            f"sampling {discharge_name} points from nearest grid cell."
        )
        # calculate distance to center cell (measured in cells)
        ar_wdw = np.abs(np.arange(-wdw, wdw + 1))
        dist = np.hypot(*np.meshgrid(ar_wdw, ar_wdw)).ravel()
        ds_wdw["dist"] = xr.Variable(
            ("index", "wdw"), np.tile(dist, (ds_wdw["index"].size, 1))
        )
        # find nearest valid cell in window
        i_wdw = ds_wdw["dist"].where(valid, np.inf).argmin("wdw").load()
    # filter valid cells
    idx_valid = np.where(valid.isel(wdw=i_wdw).values)[0]
    if idx_valid.size < gdf.index.size:
        logger.warning(
            f"{idx_valid.size}/{gdf.index.size} {discharge_name} points successfully snapped."
        )
    i_wdw = i_wdw.isel(index=idx_valid)
    # return discharge at valid cells
    ds_out = ds_wdw.isel(wdw=i_wdw.load(), index=idx_valid)
    return ds_out
