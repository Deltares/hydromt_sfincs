import numpy as np
from pyproj import CRS
import os
from os.path import join

from .conftest import TESTDATADIR, TESTMODELDIR


def test_observation_points_io(model_config, tmp_path):
    # goal: test that reads the input from existing model

    # read existing sfincs.obs file
    obs0 = model_config.observation_points.read()

    # write to testfolder
    obsfile = join(tmp_path, "sfincs.obs")
    model_config.observation_points.write(filename=obsfile)

    # read in again
    obs1 = model_config.observation_points.read(obsfile)

    # compare whether they are the same
    assert obs0 == obs1


# def test_observation_points_create(model):
# goal: check if obsfile can be made from a geojson
# gdf = model.data_catalog.get_geodataframe(
#     os.path.join(TESTDATADIR, "obs.geojson"),
# )

# def test_observation_points_set(model):
# goal: check if points outside of region are actually clipped

# def test_observation_points_merge(model):
# goal: check behaviour merge = False and True

# def test_observation_points_add_delete_point(model):
# goal: check if single point added/deleted as GUI style works

# def test_observation_points_gis(model):
# goal: check writing of geojson
