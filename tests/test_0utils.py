"""Test sfincs utils"""

import pytest
import numpy as np
from shapely.geometry import MultiLineString, Point
import geopandas as gpd
import copy

from hydromt_sfincs import utils


def test_geoms(tmp_dir, weirs):
    gdf = utils.linestring2gdf(weirs)
    assert gdf.index.size == len(weirs)
    assert np.all(gdf.geometry.type == "LineString")
    weirs1 = utils.gdf2linestring(gdf)
    for i in range(len(weirs)):
        assert sorted(weirs1[i].items()) == sorted(weirs[i].items())
    # single item MulitLineString should also work (often result of gpd.read_file)
    geoms = [MultiLineString([gdf.geometry.values[0].coords[:]])]
    struct = utils.gdf2linestring(gpd.GeoDataFrame(geometry=geoms))
    assert struct[0]["x"] == weirs[0]["x"]
    # non LineString geomtry types raise a ValueError
    with pytest.raises(ValueError, match="Invalid geometry type"):
        utils.gdf2linestring(gpd.GeoDataFrame(geometry=[Point(0, 0)]))


@pytest.mark.parametrize(
    "rotation, uv_points",
    [
        (0.0, True),
        (0.0, False),
        (15.0, True),
        (15.0, False),
    ],
)
def test_make_regular_grid(rotation, uv_points):
    # grid parameters
    x0 = 316200
    y0 = 5051400.0
    dx = dy = 200
    mmin, nmin = 0, 0
    mmax, nmax = 660, 460
    refi = 10

    # make a regular grid
    da = utils.make_regular_grid(
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        mmin=mmin,
        nmin=nmin,
        mmax=mmax,
        nmax=nmax,
        refi=refi,
        uv_points=uv_points,
        rotation=rotation,
    )
    da_transform = da.raster.transform

    # compute expected transform
    transform, width, height = utils.make_regular_grid_transform(
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        mmin=mmin,
        nmin=nmin,
        mmax=mmax,
        nmax=nmax,
        refi=refi,
        uv_points=uv_points,
        rotation=rotation,
    )

    # assertions
    np.testing.assert_allclose(da_transform, transform, atol=1e-8)
    assert da.shape == (height, width)
