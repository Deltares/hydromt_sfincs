"""Test sfincs utils"""

from datetime import datetime
from pyproj.crs.crs import CRS
from affine import Affine
import pytest
from os.path import join, dirname, abspath, isfile
import numpy as np
import xarray as xr
from shapely.geometry import MultiLineString, Point
import geopandas as gpd
import copy

from hydromt_sfincs import utils

from .conftest import TESTMODELDIR


def test_bin_map(model_config, tmp_dir):
    # get shape from config
    nmax = model_config.config.get("nmax")
    mmax = model_config.config.get("mmax")
    shape = (nmax, mmax)

    # read binary maps
    ind = utils.read_binary_map_index(join(TESTMODELDIR, "sfincs.ind"))
    msk = utils.read_binary_map(
        join(TESTMODELDIR, "sfincs.msk"), ind, shape=shape, dtype="u1", mv=0
    )
    assert [v in [0, 1, 2, 3] for v in np.unique(msk)]
    assert ind.max() == ind[-1]

    # write binary maps
    fn_out = str(tmp_dir.joinpath("sfincs.ind"))
    utils.write_binary_map_index(fn_out, msk)
    ind1 = utils.read_binary_map_index(fn_out)
    assert np.all(ind == ind1)

    fn_out = str(tmp_dir.joinpath("sfincs.msk"))
    utils.write_binary_map(fn_out, msk, msk, dtype="u1")
    msk1 = utils.read_binary_map(fn_out, ind1, shape=shape, dtype="u1", mv=0)
    assert np.all(msk1 == msk1)


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
    # weir structure requires z data
    w = copy.deepcopy(weirs[0])
    w.pop("z")
    with pytest.raises(ValueError, match='"z" value missing'):
        utils.write_geoms("fail", [w], stype="weir")
    # test I/O
    fn_out = str(tmp_dir.joinpath("test.weir"))
    utils.write_geoms(fn_out, weirs, stype="WEIR")
    weirs2 = utils.read_geoms(fn_out)
    weirs[1]["name"] = "WEIR02"  # a name is added when writing the file
    for i in range(len(weirs)):
        assert sorted(weirs2[i].items()) == sorted(weirs[i].items())


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
