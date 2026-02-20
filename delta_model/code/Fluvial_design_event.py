# FLUVIAL DESIGN EVENT FROM THIS WEBSITE
# https://deltares-research.github.io/HydroFlows/_generated/hydroflows.methods.discharge.fluvial_design_events.html

import xarray as xr
import pandas as pd
from hydromt.stats import extreme_value_analysis

def get_fluvial_design_event(ds_discharge, return_period, dist='gev'):
    """
    Standalone version of FluvialDesignEvents logic.
    
    Parameters:
    - ds_discharge: xarray.Dataset containing discharge time series (dim: 'time')
    - return_period: int/float (e.g., 100 for a 1-in-100 year event)
    - dist: string, the distribution to fit (default 'gev')
    """
    
    # 1. Calculate Annual Maxima
    da_annual_max = ds_discharge.resample(time='1AS').max()
    
    # 2. Fit Extreme Value Distribution
    # This uses the underlying logic HydroFlows calls via HydroMT
    params = extreme_value_analysis.fit_extrema(
        da_annual_max, 
        distribution=dist
    )
    
    # 3. Calculate the Peak Discharge for the Return Period
    peak_discharge = extreme_value_analysis.get_return_value(
        params, 
        return_period=return_period
    )
    
    return peak_discharge

# Example usage:
# ds = xr.open_dataset("your_discharge_data.nc")
# peak_q = get_fluvial_design_event(ds, return_period=50)