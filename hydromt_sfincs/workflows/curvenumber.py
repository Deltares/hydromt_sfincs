"""Workflow for curve number."""
import logging
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from scipy import ndimage

logger = logging.getLogger(__name__)


# Merge data for Curvenumber
def scs_recovery_determination(da_landuse, da_HSG, da_Ksat, df_map, da_mask_block):
    """Setup model the Soil Conservation Service (SCS) Curve Number (CN) files.
    More information see http://new.streamstech.com/wp-content/uploads/2018/07/SWMM-Reference-Manual-Part-I-Hydrology-1.pdf

    Parameters
    ----------
    dataset_landuse : filename (or Path) of gridded data with land use classes (e.g. NLCD)
    dataset_HSG     : filename (or Path) of gridded data with hydrologic soil group classes (HSG)
    dataset_Ksat    : filename (or Path) of gridded data with saturated hydraulic conductivity (Ksat)
    reclass_table   : mapping table that related landuse and HSG to each other (matrix; not list)
    """
    # Started
    da_smax = xr.full_like(da_mask_block, -9999, dtype=np.float32)
    da_kr = xr.full_like(da_mask_block, -9999, dtype=np.float32)

    # Interpolate soil type to landuse
    da_HSG_to_landuse = da_HSG.raster.reproject_like(
        da_landuse, method="nearest"
    ).load()

    # Curve numbers to grid: go over NLCD classes and HSG classes
    da_CN = xr.full_like(da_landuse, np.NaN, dtype=np.float32)
    for i in range(df_map.index.size):
        for j in range(df_map.columns.size):
            ind = (da_landuse == df_map.index[i]) & (
                da_HSG_to_landuse == int(df_map.columns[j])
            )
            da_CN = da_CN.where(~ind, df_map.values[i, j])

    # Convert CN to maximum soil retention (S) model grid and interpolate
    da_CN = np.maximum(da_CN, 0)  # always positive
    da_CN = np.minimum(da_CN, 100)  # not higher than 100
    da_s = np.maximum(1000 / da_CN - 10, 0)  # Equation 4.41
    da_s = da_s.fillna(0.0)  # NaN means no infiltration = 0
    ind = np.isfinite(da_s)  # inf values will be set to
    da_s = da_s.where(ind, 0.0)  # no infiltration
    da_s = da_s * 0.0254  # maximum value in meter (constant)

    # Interpolate Smax
    da_smax = da_s.raster.reproject_like(da_smax, method="average").load()

    # Interpolate Ksat to grid, define recovery as percentage
    # Reference information fom Table 4.7
    # Very low 	    0 - 0.01
    # Low 		    0.01 - 0.1      clay        0.07 µm/s   0.01 inch/hr    0.1%
    # Med-low 	    0.1 - 1         loam        0.9 µm/s    0.13 inch/hr    0.5%
    # med-high 	    1 - 10          Loamy sand  8.3 µm/s    1.18 inch/hr    1.4%
    # high 		    10 - 100        Sand        33 µm/s     4.74 inch/hr    2.9%
    # very high 	100 - Inf
    da_kr = da_Ksat.raster.reproject_like(da_kr, method="average").load()
    da_kr = np.minimum(da_kr, 100)  # not higher than 100
    da_kr = da_kr * 0.141732
    # from micrometers per second to inch/hr    (constant)
    da_kr = np.sqrt(da_kr) / 75
    # recovery in percentage of Smax per hour   (Eq. 4.36)

    # Ensure no NaNs
    da_smax = da_smax.fillna(0)
    da_kr = da_kr.fillna(0)

    # Done
    return da_smax, da_kr
