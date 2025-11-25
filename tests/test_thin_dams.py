import pytest
import numpy as np
from pathlib import Path
from os.path import isfile, join
import geopandas as gpd
from shapely.geometry import LineString

from .conftest import TESTDATADIR, TESTMODELDIR


def test_thin_dams_io(model_config, tmp_path):
    # goal:
    # - test read existing sfincs.thd file
    # - test writing to new location
    # - read in again, and compare the 2
    # - clear thdfile in config
    # - write again and check whether is added to config
    # - also write without specifying name
    # - and with filename without path
    # - test reading random nonexisting file
    # - test writing without data should raise warning

    # read existing sfincs.thd file
    model_config.thin_dams.read()

    # get the data of read in file
    obs0 = model_config.thin_dams.data

    # change root to tmpfolder
    model_config.root.set(tmp_path, mode="r+")

    # write to testfolder
    thdfile = join(tmp_path, "sfincs.thd")
    model_config.thin_dams.write(filename=thdfile)

    # check if file is made
    assert isfile(thdfile)

    # read in again
    model_config.thin_dams.read(thdfile)
    obs1 = model_config.thin_dams.data

    # compare whether the 2 gdf's are the same
    assert obs0.equals(obs1)

    # clear in config
    model_config.config.set("thdfile", None)

    # write again and check whether 'thdfile' is added to config
    model_config.thin_dams.write(filename=thdfile)

    obs2 = model_config.config.get("thdfile")
    # assert thdfile == obs2
    assert "sfincs.thd" == obs2

    # clear in config
    model_config.config.set("thdfile", None)

    # write without filename specified
    model_config.thin_dams.write()

    # check if added with default name
    obs3 = model_config.config.get("thdfile")
    assert "sfincs.thd" == obs3

    # clear in config
    model_config.config.set("thdfile", None)

    # write with filename not as path
    model_config.thin_dams.write(filename="sfincs_test.thd")

    # check if added with default name
    obs4 = model_config.config.get("thdfile")
    assert "sfincs_test.thd" == obs4

    # reading random nonexisting file > should raise warning
    with pytest.raises(IOError):
        model_config.thin_dams.read(filename="random/nonexistent/path/sfincs.thd")

    # call clear
    model_config.thin_dams.clear()

    # write as new name, result
    filename2 = "sfincs_test2.thd"
    model_config.thin_dams.write(filename=filename2)

    # result should be that no file is created
    assert Path(join(tmp_path, filename2)).exists() == False


def test_thin_dams_create(model_config):
    # goal: test if thdfile can be made from an existing geojson
    # goal: compare to similar values from existing ascii sfincs.thd file
    # goal: check behaviour merge = False and True

    # points from sfincs.thd file read in (because .data that initializes, and not ._data)
    obs0 = model_config.thin_dams.data

    # read in related geojson
    gdf_fn = join(TESTMODELDIR, "gis", "thd_clean.geojson")

    # call create
    model_config.thin_dams.create(locations=gdf_fn, merge=False)

    # check if sizes are the same
    obs1 = model_config.thin_dams.data

    # after convert multilinestring of obs0 to linestring,
    # which is now done as part of thin_dams.create()
    obs0 = obs0.explode()
    # assert obs1.shape == obs0.shape  # FIXME-obs0 has "name" included, while obs1 does not. Problem ?

    # Directly count the number of coordinate pairs in each geometry and compare
    obs0_counts = obs0.geometry.map(lambda geom: len(geom.coords))
    obs1_counts = obs1.geometry.map(lambda geom: len(geom.coords))
    assert obs0_counts.max() == obs1_counts.max()

    # check if coordinates are similar (due to rounding in ascii sfincs.thd not exactly the same)
    for geom_a, geom_b in zip(obs0.geometry, obs1.geometry):
        obs0coords = list(geom_a.coords)
        obs1coords = list(geom_b.coords)
        # assert np.isclose(obs0coords.x.values, obs1coords.x.values, rtol=0.001).all()
        assert np.isclose(obs0coords, obs1coords, rtol=0.001).all()

    # add again with merge = True and should have 2 gdfs now
    model_config.thin_dams.create(locations=gdf_fn, merge=True)
    obs2 = model_config.thin_dams.data

    assert obs2.shape[0] == 2


def test_thin_dams_add_delete(model_config):
    # goal: check if thin dams can be deleted
    # goal: check if thin dams outside of region are actually clipped
    # goal: check if deleting not existing index raises error

    # start with existing points
    obs0 = model_config.thin_dams.data

    # add again
    model_config.thin_dams.set(gdf=obs0, merge=True)

    # check if points are added
    obs1 = model_config.thin_dams.data
    assert obs1.shape[0] == 2

    # delete indexes 1
    model_config.thin_dams.delete(index=[1])
    obs2 = model_config.thin_dams.data

    # remaining should be only the first thin dam
    assert obs2.shape[0] == 1

    # add a random point outside of region, and check if error is raised
    with pytest.raises(ValueError):
        coordinates = [
            [(320000, 5012898), (320002, 5012900)],  # Line 1
        ]

        # Create a list of LineStrings
        geometry = [LineString(coords) for coords in coordinates]
        gdf = gpd.GeoDataFrame(geometry=geometry)

        # model_config.thin_dams.create(locations=gdf, merge=False)
        model_config.thin_dams.set(gdf=gdf, merge=True)

    # delete an index larger than amount of current points, and check if error is raised
    with pytest.raises(ValueError):
        model_config.thin_dams.delete(index=[42])


def test_thin_dams_clear(model_config):
    # load including data
    obs0 = model_config.thin_dams.data

    # call clear
    model_config.thin_dams.clear()

    # check if actually cleared
    assert model_config.thin_dams.data.empty


# def test_thin_dams_gis(model):
# goal: check writing of geojson
