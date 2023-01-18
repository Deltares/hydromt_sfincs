import geopandas as gpd
from hydromt import raster
import logging
import numpy as np
from scipy import ndimage
from typing import Union, List, Dict, Any
import xarray as xr


logger = logging.getLogger(__name__)

__all__ = ["merge_multi_dataarrays", "merge_dataarrays"]


def merge_multi_dataarrays(
    da_list: List[xr.DataArray],
    da_like: xr.DataArray = None,
    offset: List[Union[xr.DataArray, float]] = None,
    min_valid: List[float] = None,
    max_valid: List[float] = None,
    gdf_valid: List[gpd.GeoDataFrame] = None,
    reproj_method: Union[List[str], str] = "bilinear",
    reproj_kwargs: Dict = {},
    buffer_cells: int = 0,  # not in list
    interp_method: str = "linear",  # not in list
    merge_method: str = "first",  # not in list
    logger=logger,
) -> xr.DataArray:
    """Merge a list of data arrays by reprojecting these to a common destination grid
    and combine valid values.

    see :py:func:`~hydromt_sfincs.workflows.merge.merge_dataarrays`

    Parameters
    ----------
    da_list : List[xr.DataArray]
        list of data arrays to merge
    da_like : xr.Dataarray, optional
        Destination grid, by default None.
        If provided the output data is projected to this grid, otherwise to the first input grid.

    Returns
    -------
    xr.DataArray
        merge data array
    """

    def list_get(lst: List, i: int, fallback: Any = None) -> Any:
        if not isinstance(lst, list):
            lst = [lst]
        return (lst + [fallback] * i)[i]

    # start with common grid
    method = list_get(reproj_method, 0, "bilinear")
    if da_like is not None:  # reproject first raster to destination grid
        da1 = da_list[0].raster.reproject_like(da_like, method=method)
    elif reproj_kwargs:
        da1 = da_list[0].raster.reproject(method=method, **reproj_kwargs)
    else:  # start with first raster as destination grid
        da1 = da_list[0]

    # set nodata to np.nan, Note this might change the dtype to float
    da1 = da1.raster.mask_nodata()

    # get valid cells of first dataset
    da1 = _add_offset_mask_invalid(
        da1,
        offset=list_get(offset, 0),
        min_valid=list_get(min_valid, 0),
        max_valid=list_get(max_valid, 0),
        gdf_valid=list_get(gdf_valid, 0),
        reproj_method="bilinear",  # always bilinear!
    )

    # combine with next dataset
    for i, da2 in enumerate(da_list[1:]):
        if merge_method == "first" and not np.any(np.isnan(da1.values)):
            break
        da1 = merge_dataarrays(
            da1,
            da2,
            offset=list_get(offset, i + 1),
            min_valid=list_get(min_valid, i + 1),
            max_valid=list_get(max_valid, i + 1),
            gdf_valid=list_get(gdf_valid, i + 1),
            reproj_method=list_get(reproj_method, i + 1, "bilinear"),
            buffer_cells=buffer_cells,
            merge_method=merge_method,
            interp_method=interp_method,
        )

    # TODO identify holes in merged elevation and interpolate?
    # na_holes = ndimage.binary_fill_holes(np.isnan(da1.values), structure=np.ones((3, 3)))
    # if np.any(na_holes) > 0 and interp_method:
    #     logger.debug(f"Interpolate data at {int(np.sum(na_holes))} cells")
    #     da1 = da1.raster.interpolate_na(method=interp_method).where(na_holes)
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
    merge_method: {'first','last'}, optional
        merge method, by default 'first':

        * first: use valid new where existing invalid
        * last: use valid new
    reproj_method: {'bilinear', 'cubic', 'nearest'}
        Method used to reproject the offset and second dataset to the grid of the
        first dataset, by default 'bilinear'
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
    ## reproject da2 and reset nodata value to match da1 nodata
    da2 = da2.raster.reproject_like(da1, method=reproj_method).raster.mask_nodata()
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
