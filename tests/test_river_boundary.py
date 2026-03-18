from random import random, uniform
from unittest import result

import pytest
import numpy as np
from pathlib import Path
from os.path import isfile, join

from shapely import LineString
import geopandas as gpd

from hydromt_sfincs import SfincsModel

from .conftest import TESTMODELDIR_INLAND


def test_river_boundary_io(model_config_inland, tmp_dir):
    # original model
    model_config_inland.quadtree_grid.read()
    model_config_inland.river_boundary_points.read()
    assert model_config_inland.river_boundary_points.data is not None
    assert len(model_config_inland.river_boundary_points.data.index) == 1

    # write river boundary points to file
    model_config_inland.root.set(tmp_dir, mode="w+")
    model_config_inland.river_boundary_points.write()
    model_config_inland.quadtree_grid.write()  # Should not be needed?
    model_config_inland.config.write()
    assert isfile(tmp_dir / "sfincs.bdr")

    # read back-in to check if it remained the same
    mod = SfincsModel(root=model_config_inland.root.path, mode="r")
    mod.config.read()
    mod.quadtree_grid.read()  # Should not be needed?
    mod.river_boundary_points.read()
    assert len(mod.river_boundary_points.data.index) == 1
    assert mod.river_boundary_points.test_equal(
        model_config_inland.river_boundary_points
    )

    # now change the filename in the configuration
    mod.config.update(
        {
            "bdrfile": None,
        }
    )
    # delete the old files
    for f in ["sfincs.bdr"]:
        file_path = join(tmp_dir, f)
        if isfile(file_path):
            Path(file_path).unlink()
    # write to netcdf file
    mod.root.set(tmp_dir, mode="w+")
    mod.river_boundary_points.write()
    mod.quadtree_grid.write()  # Should not be needed?
    mod.config.write()
    assert isfile(tmp_dir / "sfincs.bdr")

    # read back-in to check if it remained the same
    mod2 = SfincsModel(root=mod.root.path, mode="r")
    mod2.config.read()
    mod2.quadtree_grid.read()  # Should not be needed?
    mod2.river_boundary_points.read()
    assert len(mod2.river_boundary_points.data.index) == 1
    assert mod2.river_boundary_points.test_equal(
        model_config_inland.river_boundary_points
    )


def test_add_point(model_config_inland):
    """Test adding a river boundary line to the model."""

    nr_points = model_config_inland.river_boundary_points.nr_points

    minx, miny, maxx, maxy = model_config_inland.region.total_bounds

    x = minx + 10000
    y = miny + 10000

    line = LineString([(x, y), (x + 1000, y)])  # minimal valid line

    gdf_random = gpd.GeoDataFrame(
        {
            "geometry": [line],
            "slope": [0.001],
            "distance": [2500],
        },
        crs=model_config_inland.region.crs,
    )

    model_config_inland.river_boundary_points.set(
        gdf=gdf_random,
        merge=True,
    )

    assert model_config_inland.river_boundary_points.nr_points == nr_points + 1


def test_create_from_hydrograph(model_config_inland):
    # Read the existing/reference boundary points
    model_config_inland.river_boundary_points.read()
    expected = model_config_inland.river_boundary_points.data.copy()

    # check that mas = 5 values are correctly assigned to data.mask
    mask = model_config_inland.quadtree_grid.data.mask.copy()
    nr_masked = np.sum(mask == 5)
    idx_masked = np.where(mask == 5)

    assert nr_masked > 0
    assert idx_masked[0][0] == 1216

    # Recreate them from hydrography
    model_config_inland.rivers.create_river_outflow(
        hydrography="merit_hydro",
        buffer=50,
        internal_dist=5000,
        reset_bounds=True,
        keep_rivers_geom=True,
    )

    result = model_config_inland.river_boundary_points.data

    # check number of features
    assert len(result) == len(expected)

    # check attributes
    assert np.isclose(result["slope"].values, expected["slope"].values, rtol=1.0e-3)
    assert np.isclose(
        result["distance"].values, expected["distance"].values, rtol=1.0e-3
    )

    # check geometry
    for g1, g2 in zip(result.geometry, expected.geometry):
        assert g1.equals_exact(g2, tolerance=1e-1)

    # check that mas = 5 values are correctly assigned to data.mask
    mask2 = model_config_inland.quadtree_grid.data.mask.copy()
    nr_masked2 = np.sum(mask2 == 5)
    idx_masked2 = np.where(mask2 == 5)

    assert nr_masked2 == nr_masked
    assert np.array_equal(idx_masked, idx_masked2)


def test_create_river_outflow_from_rivers(model_config_inland):
    model_config_inland.quadtree_grid.read()
    model_config_inland.river_boundary_points.read()

    expected = model_config_inland.river_boundary_points.data.copy()

    dir_riv = model_config_inland.root.path / "gis" / "river_centerlines.geojson"
    gdf_riv = model_config_inland.data_catalog.get_geodataframe(dir_riv)

    model_config_inland.rivers.create_river_outflow(
        rivers=gdf_riv,
        buffer=50,
        internal_dist=5000,
        reset_bounds=True,
        keep_rivers_geom=True,
    )

    result = model_config_inland.river_boundary_points.data

    # check number of features
    assert len(result) == len(expected)

    # check attributes
    assert np.isclose(result["slope"].values, expected["slope"].values, rtol=1.0e-3)
    assert np.isclose(
        result["distance"].values, expected["distance"].values, rtol=1.0e-3
    )

    # check geometry
    for g1, g2 in zip(result.geometry, expected.geometry):
        assert g1.equals_exact(g2, tolerance=1e-1)

    # remove uparea column
    gdf_riv = gdf_riv.drop(columns=["uparea"])

    with pytest.raises((ValueError), match="uparea"):
        model_config_inland.rivers.create_river_outflow(
            rivers=gdf_riv,
            buffer=50,
            internal_dist=5000,
            reset_bounds=True,
            keep_rivers_geom=True,
        )


def test_delete_clear(model_config_inland):
    """Test deleting a discharge point from the model."""
    model_config_inland.river_boundary_points.read()
    nr_points = model_config_inland.river_boundary_points.nr_points

    # Delete the point
    model_config_inland.river_boundary_points.delete(index=[0])

    # Check that the number of points has decreased
    assert model_config_inland.river_boundary_points.nr_points == nr_points - 1

    # Delete all points
    model_config_inland.river_boundary_points.clear()

    # Check that all points are deleted
    assert model_config_inland.river_boundary_points.nr_points == 0
    assert model_config_inland.config.get("bdrfile") is None
