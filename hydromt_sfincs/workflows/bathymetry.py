import geopandas as gpd
import numpy as np
import xarray as xr
from scipy import ndimage

# from pyflwdir.spread import spread2d
from typing import Union
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "merge_topobathy",
    "mask_topobathy",
    # "get_rivbank_dz",
    # "get_rivwth",
]


def mask_topobathy(
    da_elv: xr.DataArray, elv_min: float, elv_max: float = None
) -> xr.DataArray:
    """Return mask of valid elevation cells within [elv_min, elv_max] range.
    Note that local sinks (isolated regions with elv < elv_min) are kept.
    """
    dep_mask = da_elv != da_elv.raster.nodata
    if elv_min is not None:
        # active cells: contiguous area above depth threshold
        _msk = ndimage.binary_fill_holes(da_elv >= elv_min)
        dep_mask = dep_mask.where(_msk, False)

    if elv_max is not None:
        dep_mask = dep_mask.where(da_elv <= elv_max, False)

    return dep_mask


def merge_topobathy(
    da1: xr.DataArray,
    da2: xr.DataArray,
    da_offset: Union[xr.DataArray, float] = None,
    merge_buffer: int = 0,
    merge_method: str = "first",
    reproj_method="bilinear",
    logger=logger,
) -> xr.DataArray:
    """Return merged topobathy data from two datasets.

    Values from the second dataset are used where `da_mask` equals True or,
    if not provided, where the first dataset has missing values.

    If `merge_buffer` > 0, values of da2 are replaced with linearly
    interpolated values within the buffer.

    If `da_offset` is provided, a (spatially varying) offset is applied to the
    second dataset to convert the vertical datum before merging.

    Parameters
    ----------
    da1, da2: xr.DataArray
        Datasets with topobathy data to be merged.
    da_offset: xr.DataArray, float, optional
        Dataset with spatially varying offset or float with uniform offset
    merge_method: str {'first','last','min','max'}
        merge method, by default 'first':

        * first: use valid new where existing invalid
        * last: use valid new
        * min: pixel-wise min of existing and new
        * max: pixel-wise max of existing and new
    merge_buffer: int
        Buffer (number of cells) within the da_mask==True region where topobathy
        values are based on linear interpolation for a smooth transition, by default 0.
    reproj_method: str
        Method used to reproject the offset and second dataset to the grid of the
        first dataset, by default 'bilinear'


    Returns
    -------
    da_out: xr.DataArray
        Merged topobathy dataset
    """

    nodata = da1.raster.nodata
    if nodata is None or np.isnan(nodata):
        raise ValueError("da1 nodata value should be a finite value.")
    ## reproject da2 and reset nodata value to match da1 nodata
    da2 = (
        da2.raster.reproject_like(da1, method=reproj_method)
        .raster.mask_nodata()
        .fillna(nodata)
    )
    ## add offset
    if da_offset is not None:
        if isinstance(da_offset, xr.DataArray):
            da_offset = (
                da_offset.raster.reproject_like(da1, method=reproj_method)
                .raster.mask_nodata()
                .fillna(0)
            )
        da2 = da2.where(da2 == nodata, da2 + da_offset)
    # merge based merge_method
    if merge_method == "first":
        mask = da1 != nodata
    elif merge_method == "last":
        mask = da2 == nodata
    elif merge_method == "min":
        mask = da1 < da2
    elif merge_method == "max":
        mask = da1 > da2
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")
    da_out = da1.where(mask, da2)
    da_out.raster.set_nodata(nodata)
    # identify holes in merged elevation
    struct = np.ones((3, 3))
    na_mask = da_out != nodata
    if np.any(~na_mask.values):
        na_mask = ndimage.binary_fill_holes(na_mask, structure=struct)
    # identify buffer cells and set to nodata
    if merge_buffer > 0:
        mask_dilated = ndimage.binary_dilation(mask, struct, iterations=merge_buffer)
        mask_buf = np.logical_xor(mask, mask_dilated)
        da_out = da_out.where(~mask_buf, nodata)
    # interpolate buffer and holes ( nodata values )
    nempty = np.sum(da_out.values[na_mask] == nodata)
    if nempty > 0:
        logger.debug(f"Interpolate topobathy at {int(nempty)} cells")
        da_out = da_out.raster.interpolate_na(method="linear")
        da_out = da_out.where(na_mask, nodata)  # reset extrapolated area
    return da_out


# TODO publish new pyflwdir version
# def get_rivbank_dz(
#     gdf_stream: gpd.GeoDataFrame,
#     da_rivmask: xr.DataArray,
#     da_hand: xr.DataArray,
#     nmin=5,
# ) -> np.ndarray:
#     """Return median river bank height estimated from a river mask and height above
#     nearest drainage maps.

#     The river bank is defined based on edge pixels of da_rivmask. For each river bank
#     pixel the nearest stream is found. The river bank height is then calculated from
#     the median HAND value of all bank pixels with a HAND value larger than zero.
#     """
#     # rasterize streams
#     gdf_stream["segid"] = np.arange(1, gdf_stream.index.size + 1, dtype=np.int32)
#     segid = da_hand.raster.rasterize(gdf_stream, "segid").values.astype(np.int32)
#     # NOTE: the assumption is that da_rivmask is at higher resolution compared to da_hand!
#     da_rivmask_max = da_rivmask.raster.reproject_like(da_hand, method="max")
#     mask = ndimage.binary_fill_holes(da_rivmask_max)  # remove islands
#     # find nearest stream segment for all river bank cells
#     segid_spread = spread2d(segid, mask)[0]
#     # get edge of riv mask -> riv banks
#     mask_ = ndimage.binary_erosion(mask, np.ones((3, 3)))
#     bnk_mask = np.logical_and(da_hand > 0, np.logical_xor(mask, mask_))
#     riv_mask = np.logical_and(
#         np.logical_and(da_hand >= 0, da_rivmask_max), np.logical_xor(bnk_mask, mask)
#     )
#     # get median HAND for each stream -> riv bank dz
#     seg_dzbank = ndimage.labeled_comprehension(
#         da_hand.values,
#         labels=np.where(bnk_mask, segid_spread, np.int32(0)),
#         index=gdf_stream["segid"].values,
#         func=lambda x: 0 if x.size < nmin else np.median(x),
#         out_dtype=da_hand.dtype,
#         default=0,
#     )
#     # get rivmask
#     return seg_dzbank, riv_mask, bnk_mask


# def get_rivwth(
#     gdf_stream: gpd.GeoDataFrame,
#     da_rivmask: xr.DataArray,
#     rivlen_name=None,
#     nmin=5,
# ) -> np.ndarray:
#     """Return mean river width along segments in gdf_stream."""
#     assert da_rivmask.raster.crs.is_projected
#     # get/check river length
#     if rivlen_name is None:
#         rivlen_name = "rivlen"
#         gdf_stream[rivlen_name] = gdf_stream.to_crs(32736).length
#     elif rivlen_name not in gdf_stream.columns:
#         raise ValueError(f"{rivlen_name} column not found in gdf_stream")
#     # rasterize streams
#     gdf_stream["segid"] = np.arange(1, gdf_stream.index.size + 1, dtype=np.int32)
#     segid = da_rivmask.raster.rasterize(gdf_stream, "segid").values.astype(np.int32)
#     # remove islands
#     mask = ndimage.binary_fill_holes(da_rivmask)
#     # find nearest stream segment for all river cells
#     segid_spread = spread2d(segid, mask)[0]
#     # get average width
#     cellarea = abs(np.multiply(*da_rivmask.raster.res))
#     seg_count = ndimage.sum(da_rivmask, segid_spread, gdf_stream["segid"].values)
#     seg_width = np.where(
#         seg_count > nmin, seg_count * cellarea / gdf_stream[rivlen_name], -9999
#     )
#     return seg_width
