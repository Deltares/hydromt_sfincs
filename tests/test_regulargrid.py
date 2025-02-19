import numpy as np
from pyproj import CRS
import os

from .conftest import TESTDATADIR, TESTMODELDIR


def test_ind(reggrid, mask):
    ind = reggrid.ind(mask)
    assert ind[0] == 254
    assert ind[-1] == 2939
    assert ind.size == np.sum(mask > 0)


def test_grid_create(model):
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


def test_grid_create_from_region(model):
    region = model.data_catalog.get_geodataframe(
        os.path.join(TESTDATADIR, "region.geojson"),
    )

    # create the grid (note this actually calls model.reggrid.create)
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
