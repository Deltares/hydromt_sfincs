# -*- coding: utf-8 -*-

from os.path import join
import numpy as np
import pandas as pd
import xarray as xr
import logging


logger = logging.getLogger(__name__)


__all__ = ["landuse", "cn_to_s"]


RESAMPLING = {"landuse": "nearest", "lai": "average"}
DTYPES = {"landuse": np.int16}


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


def landuse(da, ds_like, fn_map, logger=logger, params=None):
    """Returns landuse map and related parameter maps.
    The parameter maps are prepared based on landuse map and
    mapping table as provided in the generic data folder of hydromt.


    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing LULC classes.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    fn_map : str, Path
        path to mapping file

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded landuse based maps
    """
    # read csv with remapping values
    df = pd.read_csv(fn_map, index_col=0, sep=",|;", engine="python", dtype=DTYPES)
    # limit dtypes to avoid gdal errors downstream
    ddict = {"float64": np.float32, "int64": np.int32}
    dtypes = {c: ddict.get(str(df[c].dtype), df[c].dtype) for c in df.columns}
    df = pd.read_csv(fn_map, index_col=0, sep=",|;", engine="python", dtype=dtypes)
    keys = df.index.values
    if params is None:
        params = [p for p in df.columns if p != "description"]
    elif not np.all(np.isin(params, df.columns)):
        missing = [p for p in params if p not in df.columns]
        raise ValueError(f"Parameter(s) missing in mapping file: {missing}")
    # setup ds out
    ds_out = xr.Dataset(coords=ds_like.raster.coords)
    # setup reclass method
    def reclass(x):
        return np.vectorize(d.get)(x, nodata)

    # apply for each parameter
    for param in params:
        method = RESAMPLING.get(param, "average")
        values = df[param].values
        nodata = values[-1]  # NOTE values is set in last row
        d = dict(zip(keys, values))  # NOTE global param in reclass method
        da = da.raster.interpolate_na(method="nearest")
        logger.info(f"Deriving {param} using {method} resampling (nodata={nodata}).")
        da_param = xr.apply_ufunc(
            reclass, da, dask="parallelized", output_dtypes=[values.dtype]
        )
        da_param.attrs.update(_FillValue=nodata)  # first set new nodata values
        ds_out[param] = da_param.raster.reproject_like(
            ds_like, method=method
        )  # then resample

    return ds_out
