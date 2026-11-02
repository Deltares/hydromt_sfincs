from os.path import join
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import dask.array as da
import rasterio

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
    write_raster,
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


def test_write_raster_dataarray_without_mask_writes_full_data(tmp_path):
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    data = xr.DataArray(
        da.from_array(values, chunks=(2, 2)),
        dims=("y", "x"),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        name="hmax",
    )
    data.raster.set_crs(32633)

    write_raster(data, tmp_path)

    with rasterio.open(tmp_path / "hmax.tif") as written:
        np.testing.assert_array_equal(written.read(1), values)


def test_write_raster_dataarray_with_explicit_mask(tmp_path):
    coords = {"y": [1.0, 0.0], "x": [0.0, 1.0]}
    data = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        dims=("y", "x"),
        coords=coords,
        name="hmax",
    )
    mask = xr.DataArray(
        np.array([[1, 1], [1, 0]], dtype=np.uint8),
        dims=("y", "x"),
        coords=coords,
        name="mask",
    )
    data.raster.set_crs(32633)
    mask.raster.set_crs(32633)

    write_raster(data, tmp_path, mask=mask, compress="LZW")

    assert (tmp_path / "hmax.tif").is_file()


def test_write_raster_dataset_uses_mask_variable(tmp_path):
    coords = {"y": [1.0, 0.0], "x": [0.0, 1.0]}
    hmax = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        dims=("y", "x"),
        coords=coords,
        name="hmax",
    )
    mask = xr.DataArray(
        np.array([[1, 1], [1, 0]], dtype=np.uint8),
        dims=("y", "x"),
        coords=coords,
        name="mask",
    )
    hmax.raster.set_crs(32633)
    mask.raster.set_crs(32633)

    write_raster(xr.Dataset({"hmax": hmax, "mask": mask}), tmp_path)

    assert (tmp_path / "hmax.tif").is_file()
