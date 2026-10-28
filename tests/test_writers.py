from os.path import join
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from hydromt_sfincs.readers import (
    read_binary_map,
    read_binary_map_index,
    read_geoms,
    read_xyn,
)
from hydromt_sfincs.writers import (
    write_binary_map,
    write_binary_map_index,
    write_geoms,
    write_xyn,
)

from .conftest import TESTMODELDIR


def test_write_binary_map(model_config, tmp_dir):
    # Read the binary maps
    shape = (model_config.config.get("nmax"), model_config.config.get("mmax"))
    ind = read_binary_map_index(Path(TESTMODELDIR, "sfincs.ind"))
    msk = read_binary_map(
        Path(TESTMODELDIR, "sfincs.msk"), ind, shape=shape, dtype="u1", mv=0
    )
    # write binary maps
    fn_out = str(tmp_dir.joinpath("sfincs.ind"))
    write_binary_map_index(fn_out, msk)
    ind1 = read_binary_map_index(fn_out)
    assert np.all(ind == ind1)

    fn_out = str(tmp_dir.joinpath("sfincs.msk"))
    write_binary_map(fn_out, msk, msk, dtype="u1")
    msk1 = read_binary_map(fn_out, ind1, shape=shape, dtype="u1", mv=0)
    assert np.all(msk1 == msk1)


def test_write_geoms(tmp_dir):
    # Read the weirs
    g = read_geoms(Path(TESTMODELDIR, "sfincs.weir"))

    # Write it
    fn_out = str(tmp_dir.joinpath("test.weir"))
    write_geoms(fn_out, g, stype="WEIR")

    # Test the output, roundtrip
    g2 = read_geoms(fn_out)
    for i in range(len(g)):
        assert sorted(g2[i].items()) == sorted(g[i].items())


def test_write_geoms_errors():
    # Read the weirs
    g = read_geoms(Path(TESTMODELDIR, "sfincs.weir"))[0]
    g.pop("elevation")

    # Assert the error while writing
    with pytest.raises(ValueError, match='"elevation" value missing'):
        write_geoms("fail", [g], stype="weir")


def test_write_xyn_drops_z_of_3d_points(tmp_dir):
    """write_xyn must tolerate 3D (X, Y, Z) geometries from GIS by dropping Z."""

    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[Point(1.0, 2.0, 99.0), Point(3.0, 4.0, 88.0)],
        crs=32633,
    )
    fn = join(tmp_dir, "sfincs.obs")
    write_xyn(fn, gdf)  # must not raise on the Z coordinate
    gdf_rt = read_xyn(fn)
    assert len(gdf_rt) == 2
    np.testing.assert_allclose(
        [gdf_rt.geometry.x.tolist(), gdf_rt.geometry.y.tolist()],
        [[1.0, 3.0], [2.0, 4.0]],
    )
