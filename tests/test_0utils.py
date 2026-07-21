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


def test_read_xy_index_is_zero_based(tmp_dir):
    """read_xy returns a canonical 0-based index (bnd/src points)."""
    fn = join(tmp_dir, "sfincs.bnd")
    with open(fn, "w") as f:
        f.write("0.0 0.0\n10.0 0.0\n10.0 10.0\n")
    gdf = utils.read_xy(fn, crs=32633)
    assert list(gdf.index) == [0, 1, 2]


def test_write_xyn_drops_z_of_3d_points(tmp_dir):
    """write_xyn must tolerate 3D (X, Y, Z) geometries from GIS by dropping Z."""
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[Point(1.0, 2.0, 99.0), Point(3.0, 4.0, 88.0)],
        crs=32633,
    )
    fn = join(tmp_dir, "sfincs.obs")
    utils.write_xyn(fn, gdf)  # must not raise on the Z coordinate
    gdf_rt = utils.read_xyn(fn)
    assert len(gdf_rt) == 2
    np.testing.assert_allclose(
        [gdf_rt.geometry.x.tolist(), gdf_rt.geometry.y.tolist()],
        [[1.0, 3.0], [2.0, 4.0]],
    )
