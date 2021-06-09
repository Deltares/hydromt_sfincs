"""Test sfincs utils"""

from _pytest.monkeypatch import V
import pytest
from os.path import join, dirname, abspath, isfile
import numpy as np
import xarray as xr
from shapely.geometry import MultiLineString, Point
import geopandas as gpd
import copy

import hydromt_sfincs.sfincs_utils as utils


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
