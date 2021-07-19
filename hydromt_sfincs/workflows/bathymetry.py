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
    # "get_rivbank_dz"
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
    da_mask: xr.DataArray = None,
    merge_buffer: int = 0,
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
    da_mask: xr.DataArray, optional
        Boolean mask of same shape as da1, with True values where to merge da2 values.
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
    if da_mask is None:
        da_mask = da1 != nodata
    elif not da1.raster.identical_grid(da_mask):
        raise ValueError("da_mask and da1 grids don't match")
    ## reproject da2
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
        da2 = xr.where(da2 != nodata, da2 + da_offset, nodata)
    # merge based on da_mask or if not provided da1 nodata values
    da_out = da1.where(~da_mask, da2)
    # identify holes in merged elevation
    struct = np.ones((3, 3))
    if np.any(da_out != nodata):
        mask_holes = np.logical_xor(
            da_out != nodata,
            ndimage.binary_fill_holes(da_out != nodata, structure=struct),
        )
        da_out = da_out.where(~mask_holes)
    # identify buffer cells
    if merge_buffer > 0:
        mask_buffer = np.logical_and(
            ndimage.binary_dilation(~da_mask, struct, iterations=merge_buffer),
            ndimage.binary_erosion(da_mask, struct, iterations=merge_buffer),
        )
        da_out = da_out.where(~mask_buffer)
    # interpolate buffer and holes
    nempty = np.sum(np.isnan(da_out.values))
    if nempty > 0:
        logger.debug(f"Interpolate topobathy at {int(nempty)} cells")
        da_out.raster.set_nodata(np.nan)
        da_out = da_out.raster.interpolate_na(method="linear")
    da_out.raster.set_nodata(nodata)
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
#     # get edge of riv mask -> riv banks
#     mask = ndimage.binary_fill_holes(da_rivmask.values)  # remove islands
#     mask_ = ndimage.binary_erosion(mask, np.ones((3, 3)))
#     banks = np.logical_and(da_hand.values > 0, np.logical_xor(mask, mask_))
#     # find nearest stream for each riv bank cell
#     bank_segid = np.where(banks, spread2d(segid, mask)[0], np.int32(0))
#     # get median HAND for each stream -> riv bank dz
#     fmed = lambda x: 0 if x.size < nmin else np.median(x)
#     dz_bank = ndimage.labeled_comprehension(
#         da_hand.values,
#         labels=bank_segid,
#         index=gdf_stream["segid"].values,
#         func=fmed,
#         out_dtype=da_hand.dtype,
#         default=0,
#     )
#     return np.maximum(0, dz_bank)
