import logging

import numpy as np
import xarray as xr
from affine import Affine
from hydromt.gis import full_from_transform

from hydromt_sfincs.workflows.merge import merge_multi_dataarrays


def test_merge_multi_dataarrays_respects_reproj_method(caplog):
    caplog.set_level(logging.DEBUG, logger="hydromt_sfincs.workflows.merge")
    crs = "EPSG:32633"
    da_like = full_from_transform(
        Affine(10, 0, 0, 0, -10, 40), (4, 4), nodata=-9999.0, crs=crs
    )
    source = full_from_transform(
        Affine(5, 0, 0, 0, -5, 40), (8, 8), nodata=-9999.0, crs=crs
    )
    yy, xx = np.indices((8, 8))
    source.values[:] = np.where((yy + xx) % 2 == 0, 10.0, 20.0)
    empty = xr.full_like(da_like, np.nan)
    empty.raster.set_nodata(np.nan)

    expected = {
        method: source.raster.reproject_like(da_like, method=method)
        for method in ("nearest", "bilinear", "average")
    }

    for method in ("nearest", "average"):
        caplog.clear()
        first = merge_multi_dataarrays(
            [{"da": source, "reproj_method": method}], da_like=da_like
        )
        second = merge_multi_dataarrays(
            [
                {"da": empty},
                {"da": source, "reproj_method": method, "merge_method": "last"},
            ],
            da_like=da_like,
        )

        assert f"Reprojection method of first dataset is: {method}" in caplog.text
        assert f"Reprojection method of dataset 1 is: {method}" in caplog.text
        np.testing.assert_allclose(
            first.values, expected[method].values, equal_nan=True
        )
        np.testing.assert_allclose(
            second.values, expected[method].values, equal_nan=True
        )

    default = merge_multi_dataarrays([{"da": source}], da_like=da_like)
    np.testing.assert_allclose(
        default.values, expected["average"].values, equal_nan=True
    )
