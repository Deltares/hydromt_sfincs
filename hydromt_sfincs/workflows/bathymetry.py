import geopandas as gpd
import numpy as np
import xarray as xr
from scipy import ndimage
from pyflwdir.spread import spread2d
import logging

logger = logging.getLogger(__name__)

__all__ = ["get_rivbank_dz"]


def get_rivbank_dz(
    gdf_stream: gpd.GeoDataFrame,
    da_rivmask: xr.DataArray,
    da_hand: xr.DataArray,
    nmin=5,
) -> np.ndarray:
    """Return median river bank height estimated from a river mask and height above
    nearest drainage maps.

    The river bank is defined based on edge pixels of da_rivmask. For each river bank
    pixel the nearest stream is found. The river bank height is then calculated from
    the median HAND value of all bank pixels with a HAND value larger than zero.
    """
    # rasterize streams
    gdf_stream["segid"] = np.arange(1, gdf_stream.index.size + 1, dtype=np.int32)
    segid = da_hand.raster.rasterize(gdf_stream, "segid").values.astype(np.int32)
    # get edge of riv mask -> riv banks
    mask = ndimage.binary_fill_holes(da_rivmask.values)  # remove islands
    mask_ = ndimage.binary_erosion(mask, np.ones((3, 3)))
    banks = np.logical_and(da_hand.values > 0, np.logical_xor(mask, mask_))
    # find nearest stream for each riv bank cell
    bank_segid = np.where(banks, spread2d(segid, mask)[0], np.int32(0))
    # get median HAND for each stream -> riv bank dz
    fmed = lambda x: 0 if x.size < nmin else np.median(x)
    dz_bank = ndimage.labeled_comprehension(
        da_hand.values,
        labels=bank_segid,
        index=gdf_stream["segid"].values,
        func=fmed,
        out_dtype=da_hand.dtype,
        default=0,
    )
    return np.maximum(0, dz_bank)
