"""Workflow to merge multiple datasets into a single dataset used for elevation and manning data."""
import logging
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from scipy import ndimage

from .bathymetry import burn_river_rect

logger = logging.getLogger(__name__)

__all__ = ["merge_multi_dataarrays", "merge_dataarrays"]


def merge_multi_dataarrays(
    da_list: List[dict],
    gdf_list: List[dict] = [],
    da_like: xr.DataArray = None,
    reproj_kwargs: Dict = {},
    buffer_cells: int = 0,  # not in list
    interp_method: str = "linear",  # not in list
    logger=logger,
) -> xr.DataArray:
    """Merge a list of data arrays by reprojecting these to a common destination grid
    and combine valid values.

    Parameters
    ----------
    da_list : List[dict]
        list of dicts with xr.DataArrays and optional merge arguments.
        Possible merge arguments are:

        * reproj_method: str, optional
            Reprojection method, if not provided, method is based on resolution (average when resolution of destination grid is coarser then data reosltuion, else bilinear).
        * offset: xr.DataArray, float, optional
            Dataset with spatially varying offset or float with uniform offset
        * zmin, zmax : float, optional
            Range of valid elevations for da2 -  only valid cells are not merged.
            Note: applied after offset!
        * gdf_valid: gpd.GeoDataFrame, optional
            Geometry of the valid region for da2
    da_like : xr.Dataarray, optional
        Destination grid, by default None.
        If provided the output data is projected to this grid, otherwise to the first input grid.
    reproj_kwargs: dict, optional
        Keyword arguments for reprojecting the data to the destination grid. Only used of no da_like is provided.
    buffer_cells : int, optional
        Number of cells between datasets to ensure smooth transition of bed levels, by default 0
    interp_method : str, optional
        Interpolation method used to fill the buffer cells , by default "linear"

    Returns
    -------
    xr.DataArray
        merged data array

    See Also:
    ---------
    :py:func:`~hydromt_sfincs.workflows.merge.merge_dataarrays`

    """

    # start with common grid
    method = da_list[0].get("reproj_method", None)
    da1 = da_list[0].get("da")

    # get resolution of da1 in meters
    dx_1 = (
        np.abs(da1.raster.res[0])
        if not da1.raster.crs.is_geographic
        else np.abs(da1.raster.res[0]) * 111111.0
    )

    # if no reprojection method is specified, base method on resolutions
    # if resolution dataset >= resolution destination grid: bilinear
    # if resolution dataset < resolution destination grid: average

    if method is None and da_like is not None:
        dx_like = (
            np.abs(da_like.raster.res[0])
            if not da_like.raster.crs.is_geographic
            else np.abs(da_like.raster.res[0]) * 111111.0
        )
        if dx_1 >= dx_like:
            method = "bilinear"
        else:
            method = "average"
    else:
        method = "bilinear"

    if da_like is not None:  # reproject first raster to destination grid
        # clip before reproject
        bbox = da_like.raster.transform_bounds(da1.raster.crs)
        da1 = da1.raster.clip_bbox(bbox, buffer=2)
        if np.any(np.array(da1.shape) <= 2):
            # no data in da1 so use an empty array like da_like
            logger.debug("No data da1, start with empty array")
            da1 = xr.full_like(da_like, np.nan)
        else:
            # TODO: this applies to the whole dataset, not only the clipped part
            da1 = da1.load().raster.reproject_like(da_like)
    elif reproj_kwargs:
        # TODO
        da1 = da1.raster.reproject(method=method, **reproj_kwargs).load()
    logger.debug(f"Reprojection method of first dataset is: {method}")

    # set nodata to np.nan, Note this might change the dtype to float
    da1 = da1.raster.mask_nodata()

    # get valid cells of first dataset
    da1 = _add_offset_mask_invalid(
        da1,
        offset=da_list[0].get("offset", None),
        min_valid=da_list[0].get("zmin", None),
        max_valid=da_list[0].get("zmax", None),
        gdf_valid=da_list[0].get("gdf_valid", None),
        reproj_method="bilinear",  # always bilinear!
    )

    # combine with next dataset
    for i in range(1, len(da_list)):
        merge_method = da_list[i].get("merge_method", "first")
        if merge_method == "first" and not np.any(np.isnan(da1.values)):
            continue

        # base reprojection method on resolution of datasets
        reproj_method = da_list[i].get("reproj_method", None)
        da2 = da_list[i].get("da")

        if reproj_method is None:
            dx_2 = (
                np.abs(da2.raster.res[0])
                if not da2.raster.crs.is_geographic
                else np.abs(da2.raster.res[0]) * 111111.0
            )
            if dx_2 >= dx_1:
                reproj_method = "bilinear"
            else:
                reproj_method = "average"
        else:
            reproj_method = "bilinear"
        logger.debug(f"Reprojection method of dataset {str(i)} is: {method}")

        da1 = merge_dataarrays(
            da1,
            da2=da2,
            offset=da_list[i].get("offset", None),
            min_valid=da_list[i].get("zmin", None),
            max_valid=da_list[i].get("zmax", None),
            gdf_valid=da_list[i].get("gdf_valid", None),
            reproj_method=reproj_method,
            merge_method=merge_method,
            buffer_cells=buffer_cells,
            interp_method=interp_method,
        )

    # burn in rivers
    for i in range(len(gdf_list)):
        cs_type = gdf_list[i].get("type", "rectangular")
        if cs_type == "rectangular":
            # width or gdf_riv_mask is required
            # zb is used if provided, otherwise depth is used
            da1 = burn_river_rect(
                da_elv=da1,
                gdf_riv=gdf_list[i].get("gdf"),
                gdf_riv_mask=gdf_list[i].get("gdf_mask", None),
                rivdph_name=gdf_list[i].get("depth", "rivdph"),
                rivwth_name=gdf_list[i].get("width", "rivwth"),
                rivbed_name=gdf_list[i].get("zb", "rivbed"),
            )
        else:
            raise NotImplementedError(f"Cross section type {cs_type} not implemented.")
    return da1


def merge_dataarrays(
    da1: xr.DataArray,
    da2: xr.DataArray,
    offset: Union[xr.DataArray, float] = None,
    min_valid: float = None,
    max_valid: float = None,
    gdf_valid: gpd.GeoDataFrame = None,
    buffer_cells: int = 0,
    merge_method: str = "first",
    reproj_method: str = "bilinear",
    interp_method: str = "linear",
) -> xr.DataArray:
    """Return merged data from two data arrays.

    Valid cells of da2 are merged with da1 according to merge_method.
    Valid cells are based on its nodata value; the min_valid-max_valid range; and the gd_valid region.

    If `buffer` > 0, values at the interface between both data arrays
    are interpolate to create a smooth surface.

    If `offset` is provided, a (spatially varying) offset is added to the
    second dataset to convert the vertical datum before merging.

    Parameters
    ----------
    da1, da2: xr.DataArray
        Data arrays to be merged.
    offset: xr.DataArray, float, optional
        Dataset with spatially varying offset or float with uniform offset
    min_valid, max_valid : float, optional
        Range of valid values for da2 -  only valid cells are not merged.
        Note: applied after offset!
    gdf_valid: gpd.GeoDataFrame, optional
        Geometry of the valid region for da2
    buffer_cells: int, optional
        Buffer (number of cells) around valid cells in da1 (if `merge_method='first'`)
        or da2 (if `merge_method='last'`) where values are interpolated
        to create a smooth surface between both datasets, by default 0.
    merge_method: {'first','last', 'mean', 'max', 'min'}, optional
        merge method, by default 'first':
        * first: use valid new where existing invalid
        * last: use valid new
        * mean: use mean of valid new and existing
        * max: use max of valid new and existing
        * min: use min of valid new and existing
    reproj_method: {'bilinear', 'cubic', 'nearest', 'average', 'max', 'min'}
        Method used to reproject the offset and second dataset to the grid of the
        first dataset, by default 'bilinear'.
        See :py:meth:`rasterio.warp.reproject` for more methods
    interp_method, {'linear', 'nearest', 'rio_idw'}
        Method used to interpolate the buffer cells, by default 'linear'.

    Returns
    -------
    da_out: xr.DataArray
        Merged dataarray
    """

    nodata = da1.raster.nodata
    dtype = da1.dtype
    if not np.isnan(nodata):
        da1 = da1.raster.mask_nodata()
    # clip before reproject
    bbox = da1.raster.transform_bounds(da2.raster.crs)
    da2 = da2.raster.clip_bbox(bbox, buffer=2)
    if np.any(np.array(da2.shape) <= 2):
        logger.debug(f"No data in dataset 2 within bbox [{bbox}], skip")
        return da1

    ## reproject da2 and reset nodata value to match da1 nodata
    da2 = da2.load().raster.reproject_like(da1, method=reproj_method)
    da2 = da2.raster.mask_nodata()

    da2 = _add_offset_mask_invalid(
        da=da2,
        offset=offset,
        min_valid=min_valid,
        max_valid=max_valid,
        gdf_valid=gdf_valid,
        reproj_method="bilinear",  # always bilinear!
    )
    # merge based merge_method
    if merge_method == "first":
        mask = ~np.isnan(da1)
    elif merge_method == "last":
        mask = np.isnan(da2)
    elif merge_method == "mean":
        mask = np.isnan(da1)
        da2 = (da1 + da2) / 2
    elif merge_method == "max":
        mask = da1 >= da2
    elif merge_method == "min":
        mask = da1 <= da2
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")
    da_out = da1.where(mask, da2)
    da_out.raster.set_nodata(np.nan)
    # identify buffer cells and interpolate data
    if buffer_cells > 0 and interp_method:
        mask_dilated = ndimage.binary_dilation(
            mask, structure=np.ones((3, 3)), iterations=buffer_cells
        )
        mask_buf = np.logical_xor(mask, mask_dilated)
        da_out = da_out.where(~mask_buf, np.nan)
        da_out_interp = da_out.raster.interpolate_na(method=interp_method)
        da_out = da_out.where(~mask_buf, da_out_interp)

    da_out = da_out.fillna(nodata).astype(dtype)
    da_out.raster.set_nodata(nodata)
    return da_out


## Helper functions
def _add_offset_mask_invalid(
    da,
    offset=None,
    min_valid=None,
    max_valid=None,
    gdf_valid=None,
    reproj_method: str = "bilinear",
):
    ## add offset
    if offset is not None:
        if isinstance(offset, xr.DataArray):
            offset = (
                offset.raster.reproject_like(da, method=reproj_method)
                .raster.mask_nodata()
                .fillna(0)
            )
        da = da.where(np.isnan(da), da + offset)
    # mask invalid values before merging
    if min_valid is not None:
        da = da.where(da >= min_valid, np.nan)
    if max_valid is not None:
        da = da.where(da <= max_valid, np.nan)
    if gdf_valid is not None:
        da = da.where(da.raster.geometry_mask(gdf_valid), np.nan)
    return da
