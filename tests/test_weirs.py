import pytest
import numpy as np
from pathlib import Path
from os.path import isfile, join
import geopandas as gpd
from shapely.geometry import LineString

from .conftest import TESTDATADIR, TESTMODELDIR


def test_weirs_io(model_config, tmp_path):
    # goal:
    # - test read existing sfincs.weir file
    # - test writing to new location
    # - read in again, and compare the 2
    # - clear weirfile in config
    # - write again and check whether is added to config
    # - also write without specifying name
    # - and with filename without path
    # - test reading random nonexisting file
    # - test writing without data should raise warning

    # read existing sfincs.weir file
    model_config.weirs.read()

    # get the data of read in file
    obs0 = model_config.weirs.data

    # change root to tmpfolder
    model_config.root.set(tmp_path, mode="r+")

    # write to testfolder
    weirfile = join(tmp_path, "sfincs.weir")
    model_config.weirs.write(filename=weirfile)

    # check if file is made
    assert isfile(weirfile)

    # read in again
    model_config.weirs.read(weirfile)
    obs1 = model_config.weirs.data

    # compare whether the 2 gdf's are the same
    assert obs0.equals(obs1)

    # clear in config
    model_config.config.set("weirfile", None)

    # write again and check whether 'weirfile' is added to config
    model_config.weirs.write(filename=weirfile)

    obs2 = model_config.config.get("weirfile")
    # assert weirfile == obs2
    assert "sfincs.weir" == obs2

    # clear in config
    model_config.config.set("weirfile", None)

    # write without filename specified
    model_config.weirs.write()

    # check if added with default name
    obs3 = model_config.config.get("weirfile")
    assert "sfincs.weir" == obs3

    # clear in config
    model_config.config.set("weirfile", None)

    # write with filename not as path
    model_config.weirs.write(filename="sfincs_test.weir")

    # check if added with default name
    obs4 = model_config.config.get("weirfile")
    assert "sfincs_test.weir" == obs4

    # reading random nonexisting file > should raise warning
    with pytest.raises(IOError):
        model_config.weirs.read(filename="random/nonexistent/path/sfincs.weir")

    # call clear
    model_config.weirs.clear()

    # write as new name, result
    filename2 = "sfincs_test2.weir"
    model_config.weirs.write(filename=filename2)

    # result should be that no file is created
    assert Path(join(tmp_path, filename2)).exists() == False


def test_weirs_create(model_config):
    # goal: test if weirfile can be made from an existing geojson
    # goal: compare to similar values from existing ascii sfincs.weir file
    # goal: check behaviour merge = False and True

    # points from sfincs.weir file read in (because .data that initializes, and not ._data)
    obs0 = model_config.weirs.data

    # read in related geojson
    gdf = model_config.data_catalog.get_geodataframe(
        join(TESTMODELDIR, "gis", "weir.geojson")
    )

    # call create
    model_config.weirs.create(locations=gdf, merge=False)

    # check if sizes are the same
    obs1 = model_config.weirs.data

    # after convert multilinestring of obs0 to linestring,
    # which is now done as part of weirs.create()
    obs0 = obs0.explode()
    # assert obs1.shape == obs0.shape  # FIXME-obs0 has "name" included, while obs1 does not. Problem ?

    # Directly count the number of coordinate pairs in each geometry and compare
    obs0_counts = obs0.geometry.map(lambda geom: len(geom.coords))
    obs1_counts = obs1.geometry.map(lambda geom: len(geom.coords))
    assert obs0_counts.max() == obs1_counts.max()

    # check if coordinates are similar (due to rounding in ascii sfincs.weir not exactly the same)
    for geom_a, geom_b in zip(obs0.geometry, obs1.geometry):
        obs0coords = list(geom_a.coords)
        obs1coords = list(geom_b.coords)
        # assert np.isclose(obs0coords.x.values, obs1coords.x.values, rtol=0.001).all()
        assert np.isclose(obs0coords, obs1coords, atol=0.001).all()

    # add again with merge = True and should have 2 gdfs now
    model_config.weirs.create(locations=gdf, merge=True)
    obs2 = model_config.weirs.data

    assert obs2.shape[0] == 2


def test_determine_weir_elevation(model_config):
    # goal: test if weirfile can be made from an existing geojson without 'z' values
    # goal: first based on only dep elevation that is actively provided
    # goal: then also based on dep and adding extra dz elevation
    # goal: then based on active dep (no dep provided)

    # points from sfincs.weir file read in (because .data that initializes, and not ._data)
    obs0 = model_config.weirs.data

    # read in related geojson
    gdf = model_config.data_catalog.get_geodataframe(
        join(TESTMODELDIR, "gis", "thd_clean.geojson")
    )

    # should give error if no dep nor dz provided
    with pytest.raises(ValueError):
        model_config.weirs.create(locations=gdf, merge=False, dep=None, dz=None)

    # read in related dep.tif
    # dep = model_config.data_catalog.get_rasterdataset(
    #     join(TESTMODELDIR, "gis", "dep.tif")
    # )
    dep_fn = join(TESTMODELDIR, "gis", "dep.tif")
    # Then call create with dep provided
    model_config.weirs.create(locations=gdf, merge=False, dep=dep_fn, dz=None)

    # check if data is added
    obs0 = model_config.weirs.data

    # Extract the z values
    z_values = obs0.geometry.apply(lambda point: point.coords[0][2])

    # check first elevation value
    assert np.isclose(z_values[0], 0.021235, atol=1e-05)

    # then make again using dz provided
    model_config.weirs.create(locations=gdf, merge=False, dep=dep_fn, dz=2.0)

    # check first elevation value
    obs1 = model_config.weirs.data
    z_values1 = obs1.geometry.apply(lambda point: point.coords[0][2])

    assert np.isclose(
        z_values1[0], 2.021235, atol=1e-05
    )  # should be 2.0 higher than before

    # then make again using dz provided, but no dep (should use active dep)
    # FIXME: add this option
    # model_config.weirs.create(locations=gdf, merge=False, dep=None, dz=2.0)

    # # check first elevation value
    # obs2 = model_config.weirs.data
    # z_values2 = obs2.geometry.apply(lambda point: point.coords[0][2])

    # assert z_values2[0] == 1.0 #should again be 1.0


def test_weirs_add_delete(model_config):
    # goal: check if weirs can be deleted
    # goal: check if weirs outside of region are actually clipped
    # goal: check if deleting not existing index raises error

    # start with existing points
    obs0 = model_config.weirs.data

    # add again
    model_config.weirs.set(gdf=obs0, merge=True)

    # check if points are added
    obs1 = model_config.weirs.data
    assert obs1.shape[0] == 2

    # delete indexes 1
    model_config.weirs.delete(index=[1])
    obs2 = model_config.weirs.data

    # remaining should be only the first weir
    assert obs2.shape[0] == 1

    # add a random point outside of region, and check if error is raised
    with pytest.raises(ValueError):
        coordinates = [
            [(320000, 5012898), (320002, 5012900)],  # Line 1
        ]

        # Create a list of LineStrings
        geometry = [LineString(coords) for coords in coordinates]
        gdf = gpd.GeoDataFrame(geometry=geometry)

        # model_config.weirs.create(locations=gdf, merge=False)
        model_config.weirs.set(gdf=gdf, merge=True)

    # delete an index larger than amount of current points, and check if error is raised
    with pytest.raises(ValueError):
        model_config.weirs.delete(index=[42])


def test_weirs_clear(model_config):
    # load including data
    obs0 = model_config.weirs.data

    # call clear
    model_config.weirs.clear()

    # check if actually cleared
    assert model_config.weirs.data.empty


# def test_weirs_gis(model):
# goal: check writing of geojson
