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

EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples", "sfincs_riverine")


def test_inp(tmpdir):
    conf = utils.read_inp(join(EXAMPLEDIR, "sfincs.inp"))
    assert isinstance(conf, dict)
    assert "mmax" in conf
    fn_out = str(tmpdir.join("sfincs.inp"))
    utils.write_inp(fn_out, conf)
    conf1 = utils.read_inp(fn_out)
    assert conf == conf1

    shape, transform, crs = utils.get_spatial_attrs(conf)
    assert isinstance(crs, CRS)
    assert isinstance(transform, Affine)
    assert len(shape) == 2
    crs = utils.get_spatial_attrs(conf, crs=4326)[-1]
    assert crs.to_epsg() == 4326
    conf.pop("epsg")
    crs = utils.get_spatial_attrs(conf)[-1]
    assert crs is None

    with pytest.raises(NotImplementedError, match="Rotated grids"):
        conf.update(rotation=1)
        utils.get_spatial_attrs(conf)
    with pytest.raises(ValueError, match='"mmax" or "nmax"'):
        conf.pop("mmax")
        utils.get_spatial_attrs(conf)

    dt = utils.parse_datetime(conf["tref"])
    assert isinstance(dt, datetime)
    with pytest.raises(ValueError, match="Unknown type for datetime"):
        utils.parse_datetime(22)


def test_bin_map(tmpdir):
    conf = utils.read_inp(join(EXAMPLEDIR, "sfincs.inp"))
    shape = utils.get_spatial_attrs(conf)[0]
    ind = utils.read_binary_map_index(join(EXAMPLEDIR, "sfincs.ind"))
    msk = utils.read_binary_map(
        join(EXAMPLEDIR, "sfincs.msk"), ind, shape=shape, dtype="u1", mv=0
    )
    assert [v in [0, 1, 2, 3] for v in np.unique(msk)]
    assert ind.max() == ind[-1]

    fn_out = str(tmpdir.join("sfincs.ind"))
    utils.write_binary_map_index(fn_out, msk)
    ind1 = utils.read_binary_map_index(fn_out)
    assert np.all(ind == ind1)

    fn_out = str(tmpdir.join("sfincs.msk"))
    utils.write_binary_map(fn_out, msk, msk, dtype="u1")
    msk1 = utils.read_binary_map(fn_out, ind1, shape=shape, dtype="u1", mv=0)
    assert np.all(msk1 == msk1)


def test_structures(tmpdir, weirs):
    gdf = utils.structures2gdf(weirs)
    assert gdf.index.size == len(weirs)
    assert np.all(gdf.geometry.type == "LineString")
    weirs1 = utils.gdf2structures(gdf)
    for i in range(len(weirs)):
        assert sorted(weirs1[i].items()) == sorted(weirs[i].items())
    # single item MulitLineString should also work (often result of gpd.read_file)
    geoms = [MultiLineString([gdf.geometry.values[0].coords[:]])]
    struct = utils.gdf2structures(gpd.GeoDataFrame(geometry=geoms))
    assert struct[0]["x"] == weirs[0]["x"]
    # non LineString geomtry types raise a ValueError
    with pytest.raises(ValueError, match="Invalid geometry type"):
        utils.gdf2structures(gpd.GeoDataFrame(geometry=[Point(0, 0)]))
    # weir structure requires z data
    w = copy.deepcopy(weirs[0])
    w.pop("z")
    with pytest.raises(ValueError, match='"z" value missing'):
        utils.write_structures("fail", [w], stype="weir")
    # test I/O
    fn_out = str(tmpdir.join("test.weir"))
    utils.write_structures(fn_out, weirs, stype="WEIR")
    weirs2 = utils.read_structures(fn_out)
    weirs[1]["name"] = "WEIR02"  # a name is added when writing the file
    for i in range(len(weirs)):
        assert sorted(weirs2[i].items()) == sorted(weirs[i].items())
