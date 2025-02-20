import numpy as np
import geopandas as gpd
from pyproj import CRS
import os
from os.path import join

from .conftest import TESTDATADIR, TESTMODELDIR


def test_observation_points_io(model_config, tmp_path):
    # goal:
    # - test read existing sfincs.obs file
    # - test writing to new location
    # - read in again, and compare the 2

    # read existing sfincs.obs file
    model_config.observation_points.read()

    # get the data of read in file
    obs0 = model_config.observation_points.data

    # write to testfolder
    obsfile = join(tmp_path, "sfincs.obs")
    model_config.observation_points.write(filename=obsfile)

    # read in again
    model_config.observation_points.read(obsfile)
    obs1 = model_config.observation_points.data

    # compare whether the 2 gdf's are the same
    assert obs0.equals(obs1)


def test_observation_points_create(model_config):
    # goal: test if obsfile can be made from an existing geojson
    # goal: compare to similar values from existing ascii sfincs.obs file
    # goal: check behaviour merge = False and True

    # points from sfincs.obs file read in (because .data that initializes, and not ._data)
    obs0 = model_config.observation_points.data

    # read in related geojson
    gdf = model_config.data_catalog.get_geodataframe(
        join(TESTMODELDIR, "gis", "obs.geojson")
    )

    # call create
    model_config.observation_points.create(locations=gdf, merge=False)

    # check if sizes are the same
    obs1 = model_config.observation_points.data
    assert obs1.shape == obs0.shape  # (3,2) > both 3 points

    # check if coordinates are similar (do to rounding in ascii sfincs.obs not exactly the same)
    assert np.isclose(obs1.geometry.x.values, obs0.geometry.x.values, rtol=0.001).all()
    assert np.isclose(obs1.geometry.y.values, obs0.geometry.y.values, rtol=0.001).all()

    # add again with merge = True and should have 6 points now
    model_config.observation_points.create(locations=gdf, merge=True)
    obs2 = model_config.observation_points.data
    assert obs2.size == 12  # (6,2) > now 6 points


def test_observation_points_clear(model_config):
    # load including data
    obs0 = model_config.observation_points.data

    # call clear
    model_config.observation_points.clear()

    # check if actually cleared
    assert model_config.observation_points.data.empty


# def test_observation_points_add_delete(model):
# goal: check if point can be added (one in and one outside of region)
# goal: check if points outside of region are actually clipped

# def test_observation_points_add_delete_point(model):
# goal: check if single point added/deleted as GUI style works

# def test_observation_points_gis(model):
# goal: check writing of geojson
