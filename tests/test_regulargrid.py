import numpy as np
from pyproj import CRS
import pytest
import os

from hydromt_sfincs.sfincs import SfincsModel

from .conftest import TESTDATADIR, TESTMODELDIR


def test_ind(reggrid, mask):
    ind = reggrid.ind(mask)
    assert ind[0] == 254
    assert ind[-1] == 2939
    assert ind.size == np.sum(mask > 0)


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


def test_get_indices_at_points_south_up_flatten(model_init):
    """Indices must point into the C-order flatten of the SOUTH-UP raster.

    Regression: the function used to return the SFINCS-internal Fortran
    order (``col * nmax + row``), silently scrambling every index-COG
    lookup for regular grids (downscale_floodmap / downscale_velocity).
    """
    model = model_init
    model.grid.create(
        mmax=10, nmax=6, dx=100, dy=100, x0=1000.0, y0=5000.0, rotation=0, epsg=32633
    )
    grid = model.grid

    # centre of column 3, southern row 2 -> row * mmax + col
    x = np.array([[1000.0 + 3.5 * 100]])
    y = np.array([[5000.0 + 2.5 * 100]])
    assert grid.get_indices_at_points(x, y)[0, 0] == 2 * 10 + 3

    # consistency with the raster flatten: pixel [r, c] of the model raster
    mask = grid.empty_mask
    assert mask.y.values[0] < mask.y.values[-1]  # south-up convention
    r, c = 4, 7
    xc = float(mask.x.values[c])
    yc = float(mask.y.values[r])
    ind = grid.get_indices_at_points(np.array([[xc]]), np.array([[yc]]))
    assert ind[0, 0] == r * grid.mmax + c


def test_get_indices_at_points_rotated_and_outside(model_init):
    model = model_init
    model.grid.create(
        mmax=8, nmax=5, dx=50, dy=50, x0=2000.0, y0=6000.0, rotation=27.0, epsg=32633
    )
    grid = model.grid

    # rotated cell centre maps back to row * mmax + col
    col, row = 6, 3
    xw, yw = grid.transform * (col + 0.5, row + 0.5)
    ind = grid.get_indices_at_points(np.array([[xw]]), np.array([[yw]]))
    assert ind[0, 0] == row * grid.mmax + col

    # points outside the grid return -999
    ind = grid.get_indices_at_points(np.array([[-1e6]]), np.array([[-1e6]]))
    assert ind[0, 0] == -999
