"""Landuse related workflows for SFINCS."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


__all__ = ["cn_to_s"]


def cn_to_s(da_cn, da_mask=None, nodata=-9999):
    """Convert Curve Numbers to potential maximum soil moisture retention S [inch]."""
    # set nodata values to CN 100 (zero infiltration)
    # avoid CN = 0 values; minumum expected value is ~30
    da_cn = np.maximum(1, da_cn.raster.mask_nodata().fillna(100))
    da_s = np.maximum(1000 / da_cn - 10, 0).round(3)
    if da_mask is not None:
        da_s = da_s.where(da_mask, nodata)
    da_s.raster.set_nodata(nodata)
    return da_s
