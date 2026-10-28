import numpy as np
from pyproj import CRS
import pytest
from unittest.mock import MagicMock, patch
import os

from xugrid import UgridDataArray
from hydromt_sfincs.sfincs import SfincsModel

from .conftest import TESTDATADIR, TESTMODELDIR


def test_ind(reggrid, mask):
    ind = reggrid.ind(mask)
    assert ind[0] == 254
    assert ind[-1] == 2939
    assert ind.size == np.sum(mask > 0)


def test_grid_read_loads_index_once(model_config):
    grid = model_config.grid
    with patch("numpy.fromfile", wraps=np.fromfile) as read_file:
        grid.read()
        grid._read_layers(["dep"])
        grid._read_layers(["dep"])
    index_path = model_config.config.get("indexfile", abs_path=True)
    index_reads = [
        call for call in read_file.call_args_list if call.args[0] == index_path
    ]
    assert len(index_reads) == 1


def test_read_ind_reads_file(model_init, tmp_path):
    grid = model_init.grid
    index_path = tmp_path / "sfincs.ind"
    np.array([2, 1, 3], dtype="u4").tofile(index_path)
    with patch("numpy.fromfile", wraps=np.fromfile) as read_file:
        np.testing.assert_array_equal(grid.read_ind(index_path), [0, 2])
        np.testing.assert_array_equal(grid.read_ind(index_path), [0, 2])
        assert read_file.call_count == 2


def test_indices_use_file_then_refresh_from_mask(model_config, tmp_path):
    grid = model_config.grid
    np.testing.assert_array_equal(
        grid.indices,
        grid.read_ind(model_config.config.get("indexfile", abs_path=True)),
    )
    grid.read()
    original_indices = grid.indices.copy()
    grid.data["mask"].values[:] = 0
    grid.data["mask"].values[0, 0] = 1
    np.testing.assert_array_equal(grid.indices, original_indices)
    grid.data["dep"].values[:] = 0
    grid.root.set(tmp_path, mode="w+")
    grid.model.config.write()
    grid.write()
    np.testing.assert_array_equal(grid.indices, [0])


def test_grid_io(model_config, tmp_path):
    # create a model instance with configuration read
    model = model_config

    # check if grid-variables are set properly
    # this happens with update_grid_from_config()
    assert model.grid.mmax == 84
    assert model.grid.nmax == 36
    assert model.grid.dx == 150
    assert model.grid.dy == 150
    assert model.grid.x0 == 318650.0
    assert model.grid.y0 == 5040000.0
    assert model.grid.rotation == 27.0
    # model crs comes from sfincs.inp in this case
    assert model.crs == CRS.from_epsg(32633)

    # even though properties are set, the grid is not created yet
    assert model.grid._data is None

    # now read the grid, this reads the mask and dep
    model.grid.read()

    # check the shape model.grid.data
    assert model.grid.data.raster.shape == (36, 84)

    # check the variables in the grid
    assert "mask" in model.grid.data.variables
    assert len(model.grid.data.data_vars) == 2

    # now write the model grid
    model.root.set(tmp_path, mode="w+")
    model.config.write()
    model.grid.write()

    # and read it again
    model1 = SfincsModel(root=tmp_path, mode="r")
    model1.config.read()
    model1.grid.read()

    # assert the grid is the same
    assert model.grid.data.equals(model1.grid.data)

    # check that the quadtree grid was created
    assert isinstance(model.quadtree_grid.data, UgridDataArray)
    assert model.quadtree_grid.crs == model.grid.crs


def test_grid_component_data_contains_only_owned_layers(model_config):
    model = model_config
    grid = model.grid
    grid.read()
    components = {
        "elevation": ("dep",),
        "mask": ("mask",),
        "roughness": ("manning",),
        "storage_volume": ("vol",),
        "initial_conditions": ("zs",),
        "infiltration": model.infiltration.grid_variables,
    }
    for name in set().union(*map(set, components.values())):
        grid.set(grid.mask.astype(np.float32), name=name)

    for name, variables in components.items():
        assert set(model.components[name].data.data_vars) == set(variables)


def test_regular_infiltration_clear_preserves_grid_layers(model_config):
    model = model_config
    model.grid.read()
    mask = model.grid.data["mask"].copy(deep=True)
    dep = model.grid.data["dep"].copy(deep=True)
    model.grid.set(model.grid.mask.astype(np.float32), name="qinf")

    model.infiltration.clear()

    assert "qinf" not in model.grid.data
    np.testing.assert_array_equal(model.grid.data["mask"].values, mask.values)
    np.testing.assert_array_equal(model.grid.data["dep"].values, dep.values)


def test_grid_create(model_init):
    model = model_init
    # create a simple regular grid similar to sfincs_test
    grid_params = {
        "mmax": 84,
        "nmax": 36,
        "dx": 150,
        "dy": 150,
        "x0": 318650.0,
        "y0": 5040000.0,
        "rotation": 27.0,
        "epsg": 32633,
    }

    # create the grid (note this actually calls model.reggrid.create)
    model.grid.create(**grid_params)

    assert model.crs == CRS.from_epsg(32633)


def test_grid_create_from_region(model_init):
    model = model_init
    region = model.data_catalog.get_geodataframe(
        os.path.join(TESTDATADIR, "region.geojson"),
    )

    model.grid.create_from_region(
        region={"geom": region},
        res=150,
        crs="utm",
        rotated=False,
        align=True,
    )

    assert model.crs == CRS.from_epsg(32633)
    assert model.grid.mmax == 91
    assert model.grid.nmax == 70
    assert np.isclose(model.grid.dx, 150, atol=1e-3)
    assert np.isclose(model.grid.dy, 150, atol=1e-3)
    assert np.isclose(model.grid.x0, 316200.0, atol=1e-3)
    assert np.isclose(model.grid.y0, 5040000.0, atol=1e-3)
    assert np.isclose(model.grid.rotation, 0, atol=1e-3)


def test_grid_create_from_region_rotated(model_init):
    model = model_init
    region = model.data_catalog.get_geodataframe(
        os.path.join(TESTDATADIR, "region.geojson"),
    )

    model.grid.create_from_region(
        region={"geom": region}, res=150, crs="utm", rotated=True
    )

    assert model.crs == CRS.from_epsg(32633)
    assert model.grid.mmax == 84
    assert model.grid.nmax == 36
    assert np.isclose(model.grid.dx, 150, atol=1e-3)
    assert np.isclose(model.grid.dy, 150, atol=1e-3)
    assert np.isclose(model.grid.x0, 318650.0, atol=1e-3)
    assert np.isclose(model.grid.y0, 5040000.0, atol=1e-3)
    assert np.isclose(model.grid.rotation, 27.0, atol=1e-3)
