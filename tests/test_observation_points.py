import pytest
import numpy as np
from pathlib import Path
from os.path import isfile, join

from .conftest import TESTDATADIR, TESTMODELDIR


def test_observation_points_io(model_config, tmp_path):
    # goal:
    # - test read existing sfincs.obs file
    # - test writing to new location
    # - read in again, and compare the 2
    # - clear obsfile in config
    # - write again and check whether is added to config
    # - also write without specifying name
    # - and with filename without path
    # - test reading random nonexisting file
    # - test writing without data should raise warning

    # read existing sfincs.obs file
    model_config.observation_points.read()

    # get the data of read in file
    obs0 = model_config.observation_points.data

    # change root to tmpfolder
    model_config.root.set(tmp_path, mode="r+")

    # write to testfolder
    obsfile = join(tmp_path, "sfincs.obs")
    model_config.observation_points.write(filename=obsfile)

    # check if file is made
    assert isfile(obsfile)

    # read in again
    model_config.observation_points.read(obsfile)
    obs1 = model_config.observation_points.data

    # compare whether the 2 gdf's are the same
    assert obs0.equals(obs1)

    # clear in config
    model_config.config.set("obsfile", None)

    # write again and check whether 'obsfile' is added to config
    model_config.observation_points.write(filename=obsfile)

    obs2 = model_config.config.get("obsfile")
    # assert obsfile == obs2
    assert "sfincs.obs" == obs2

    # clear in config
    model_config.config.set("obsfile", None)

    # write without filename specified
    model_config.observation_points.write()

    # check if added with default name
    obs3 = model_config.config.get("obsfile")
    assert "sfincs.obs" == obs3

    # clear in config
    model_config.config.set("obsfile", None)

    # write with filename not as path
    model_config.observation_points.write(filename="sfincs_test.obs")

    # check if added with default name
    obs4 = model_config.config.get("obsfile")
    assert "sfincs_test.obs" == obs4

    # reading random nonexisting file > should raise warning
    with pytest.raises(IOError):
        model_config.observation_points.read(
            filename="random/nonexistent/path/sfincs.obs"
        )

    # call clear
    model_config.observation_points.clear()

    # write as new name, result
    filename2 = "sfincs_test2.obs"
    model_config.observation_points.write(filename=filename2)

    # result should be that no file is created
    assert Path(join(tmp_path, filename2)).exists() == False


def test_observation_points_create(model_config):
    # goal: test if obsfile can be made from an existing geojson
    # goal: compare to similar values from existing ascii sfincs.obs file
    # goal: check behaviour merge = False and True

    # points from sfincs.obs file read in (because .data that initializes, and not ._data)
    obs0 = model_config.observation_points.data

    # read in related geojson
    gdf_fn = Path(TESTMODELDIR) / "gis" / "obs.geojson"

    # call create
    model_config.observation_points.create(locations=gdf_fn, merge=False)

    # check if sizes are the same
    obs1 = model_config.observation_points.data
    assert obs1.shape == obs0.shape  # (3,2) > both 3 points

    # check if coordinates are similar (due to rounding in ascii sfincs.obs not exactly the same)
    assert np.isclose(obs1.geometry.x.values, obs0.geometry.x.values, rtol=0.001).all()
    assert np.isclose(obs1.geometry.y.values, obs0.geometry.y.values, rtol=0.001).all()

    # add again with merge = True and should have 6 points now
    model_config.observation_points.create(locations=gdf_fn, merge=True)
    obs2 = model_config.observation_points.data
    assert obs2.size == 12  # (6,2) > now 6 points


def test_observation_points_add_delete(model_config):
    # goal: check if point can be added through .add()
    # goal: check if points can be deleted
    # goal: check if single point added/deleted as GUI style works
    # goal: check if points outside of region are actually clipped

    # start with existing points
    obs0 = model_config.observation_points.data

    # add again
    model_config.observation_points.set(gdf=obs0)

    # check if points are added
    obs1 = model_config.observation_points.data
    assert len(obs1) == 6  # (6,2) > now 6 points

    # delete indexes 1,2,4,5
    model_config.observation_points.delete(index=[1, 2, 4, 5])
    obs2 = model_config.observation_points.data

    # remaining should be twice the same point geometry
    assert obs2.geometry.iloc[0].equals(obs2.geometry.iloc[1])

    # remove a single point by index
    model_config.observation_points.delete_point(name_or_index=1)
    obs3 = model_config.observation_points.data
    assert len(obs3) == 1  # (1,2) > now 1 point

    # add a random point in grid
    model_config.observation_points.add_point(x=320000, y=5042890, name="test")
    obs4 = model_config.observation_points.data
    assert len(obs4) == 2  # (2,2) > now 2 points

    # remove a single point by name
    model_config.observation_points.delete_point(name_or_index="test")
    obs5 = model_config.observation_points.data
    assert len(obs5) == 1  # (1,2) > now 1 point

    # add a random point outside of region, and check if error is raised
    with pytest.raises(ValueError):
        model_config.observation_points.add_point(x=320000, y=5012898, name="test")

    # delete an index larger than amount of current points, and check if error is raised
    with pytest.raises(ValueError):
        model_config.observation_points.delete(index=[42])


def test_observation_points_clear(model_config):
    # load including data
    obs0 = model_config.observation_points.data

    # call clear
    model_config.observation_points.clear()

    # check if actually cleared
    assert model_config.observation_points.data.empty


# def test_observation_points_gis(model):
# goal: check writing of geojson
