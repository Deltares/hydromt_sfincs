import pytest
import numpy as np
from pathlib import Path
from os.path import isfile, join
import geopandas as gpd
from shapely.geometry import LineString

from .conftest import TESTDATADIR, TESTMODELDIR


def test_cross_sections_io(model_config, tmp_path):
    # goal:
    # - test read existing sfincs.crs file
    # - test writing to new location
    # - read in again, and compare the 2
    # - clear crsfile in config
    # - write again and check whether is added to config
    # - also write without specifying name
    # - and with filename without path
    # - test reading random nonexisting file
    # - test writing without data should raise warning

    # read existing sfincs.crs file
    model_config.cross_sections.read()

    # get the data of read in file
    obs0 = model_config.cross_sections.data

    # change root to tmpfolder
    model_config.root.set(tmp_path, mode="r+")

    # write to testfolder
    crsfile = join(tmp_path, "sfincs.crs")
    model_config.cross_sections.write(filename=crsfile)

    # check if file is made
    assert isfile(crsfile)

    # read in again
    model_config.cross_sections.read(crsfile)
    obs1 = model_config.cross_sections.data

    # compare whether the 2 gdf's are the same
    assert obs0.equals(obs1)

    # clear in config
    model_config.config.set("crsfile", None)

    # write again and check whether 'crsfile' is added to config
    model_config.cross_sections.write(filename=crsfile)

    obs2 = model_config.config.get("crsfile")
    # assert crsfile == obs2
    assert "sfincs.crs" == obs2

    # clear in config
    model_config.config.set("crsfile", None)

    # write without filename specified
    model_config.cross_sections.write()

    # check if added with default name
    obs3 = model_config.config.get("crsfile")
    assert "sfincs.crs" == obs3

    # clear in config
    model_config.config.set("crsfile", None)

    # write with filename not as path
    model_config.cross_sections.write(filename="sfincs_test.crs")

    # check if added with default name
    obs4 = model_config.config.get("crsfile")
    assert "sfincs_test.crs" == obs4

    # reading random nonexisting file > should raise warning
    with pytest.raises(IOError):
        model_config.cross_sections.read(filename="random/nonexistent/path/sfincs.crs")

    # call clear
    model_config.cross_sections.clear()

    # write as new name, result
    filename2 = "sfincs_test2.crs"
    model_config.cross_sections.write(filename=filename2)

    # result should be that no file is created
    assert Path(join(tmp_path, filename2)).exists() == False


def test_cross_sections_create(model_config):
    # goal: test if crsfile can be made from an existing geojson
    # goal: compare to similar values from existing ascii sfincs.crs file
    # goal: check behaviour merge = False and True

    # points from sfincs.crs file read in (because .data that initializes, and not ._data)
    obs0 = model_config.cross_sections.data

    # read in related geojson
    gdf = model_config.data_catalog.get_geodataframe(
        join(TESTMODELDIR, "gis", "crs.geojson")
    )

    # call create
    model_config.cross_sections.create(locations=gdf, merge=False)

    # check if sizes are the same
    obs1 = model_config.cross_sections.data

    # after convert multilinestring of obs0 to linestring,
    # which is now done as part of cross_sections.create()
    obs0 = obs0.explode()
    # assert obs1.shape == obs0.shape  # FIXME-obs0 has "name" included, while obs1 does not. Problem ?

    # Directly count the number of coordinate pairs in each geometry and compare
    obs0_counts = obs0.geometry.map(lambda geom: len(geom.coords))
    obs1_counts = obs1.geometry.map(lambda geom: len(geom.coords))
    assert obs0_counts.max() == obs1_counts.max()

    # check if coordinates are similar (due to rounding in ascii sfincs.crs not exactly the same)
    for geom_a, geom_b in zip(obs0.geometry, obs1.geometry):
        obs0coords = list(geom_a.coords)
        obs1coords = list(geom_b.coords)
        # assert np.isclose(obs0coords.x.values, obs1coords.x.values, rtol=0.001).all()
        assert np.isclose(obs0coords, obs1coords, rtol=0.001).all()

    # add again with merge = True and should have 6 gdfs now
    model_config.cross_sections.create(locations=gdf, merge=True)
    obs2 = model_config.cross_sections.data

    assert obs2.shape[0] == 6


def test_cross_sections_add_delete(model_config):
    # goal: check if cross-sections can be deleted
    # goal: check if cross-sections outside of region are actually clipped
    # goal: check if deleting not existing index raises error

    # start with existing points
    obs0 = model_config.cross_sections.data

    # add again
    model_config.cross_sections.set(gdf=obs0, merge=True)

    # check if points are added
    obs1 = model_config.cross_sections.data
    assert obs1.shape[0] == 6  # =2 time 3 cross-sections

    # delete indexes 1
    model_config.cross_sections.delete(index=[1, 2, 3, 4, 5])
    obs2 = model_config.cross_sections.data

    # remaining should be only the first cross-section dam
    assert obs2.shape[0] == 1

    # add a random point outside of region, and check if error is raised
    with pytest.raises(ValueError):
        coordinates = [
            [(320000, 5012898), (320002, 5012900)],  # Line 1
        ]

        # Create a list of LineStrings
        geometry = [LineString(coords) for coords in coordinates]
        gdf = gpd.GeoDataFrame(geometry=geometry)

        # model_config.cross_sections.create(locations=gdf, merge=False)
        model_config.cross_sections.set(gdf=gdf, merge=True)

    # delete an index larger than amount of current points, and check if error is raised
    with pytest.raises(ValueError):
        model_config.cross_sections.delete(index=[42])


def test_cross_sections_clear(model_config):
    # load including data
    obs0 = model_config.cross_sections.data

    # call clear
    model_config.cross_sections.clear()

    # check if actually cleared
    assert model_config.cross_sections.data.empty


# def test_cross_sections_gis(model):
# goal: check writing of geojson
