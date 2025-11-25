import pytest
import numpy as np
from pathlib import Path
from os.path import isfile, join

from hydromt_sfincs import SfincsModel

from .conftest import TESTDATADIR, TESTMODELDIR


def test_water_level_boundary_io(model_config, tmp_dir):
    # read water level boundary from files
    model_config.grid.read()
    model_config.water_level.read()
    assert model_config.water_level.data is not None
    assert len(model_config.water_level.data.index) == 2

    # write water level to file
    model_config.root.set(tmp_dir, mode="w+")
    model_config.water_level.write()
    model_config.config.write()
    assert isfile(tmp_dir / "sfincs.bnd")
    assert isfile(tmp_dir / "sfincs.bzs")
    assert isfile(tmp_dir / "tide_FES2014.bca")
    # assert isfile(tmp_dir, "src.geojson")

    # read back-in to check if it remained the same
    mod = SfincsModel(root=model_config.root.path, mode="r")
    mod.config.read()
    mod.water_level.read()
    assert len(mod.water_level.data.index) == 2
    assert mod.water_level.test_equal(model_config.water_level)

    # now change the filename in the configuration
    mod.config.update(
        {
            "bndfile": None,
            "bzsfile": None,
            "netbndbzsbzifile": "sfincs_netbndbzsbzifile.nc",
        }
    )
    # delete the old files
    for f in ["sfincs.bnd", "sfincs.bzs"]:
        file_path = join(tmp_dir, f)
        if isfile(file_path):
            Path(file_path).unlink()
    # write to netcdf file
    mod.root.set(tmp_dir, mode="w+")
    mod.water_level.write()
    mod.config.write()
    assert isfile(tmp_dir / "sfincs_netbndbzsbzifile.nc")
    assert not isfile(tmp_dir / "sfincs.bnd")
    assert not isfile(tmp_dir / "sfincs.bzs")

    # read back-in to check if it remained the same
    mod2 = SfincsModel(root=mod.root.path, mode="r")
    mod2.config.read()
    mod2.water_level.read()
    assert len(mod2.water_level.data.index) == 2
    assert mod2.water_level.test_equal(model_config.water_level)


def test_add_point(model_config):
    """Test adding a discharge point to the model."""
    nr_points = model_config.water_level.nr_points

    # determine point in the middle of the grid
    gdf = model_config.region
    point = gdf.geometry.union_all().centroid

    model_config.water_level.add_point(
        x=point.x, y=point.y, value=-10.0, name="test_point"
    )

    # Check that the number of points has increased and value is set correctly
    assert model_config.water_level.nr_points == nr_points + 1
    assert np.mean(model_config.water_level.data["bzs"].isel(index=-1).values) == -10.0

    # assert the astronomic constituents are dropped after adding point
    assert "constituent" not in model_config.water_level.data.dims


def test_drop_duplicates(model_config, tmp_dir):
    """Test dropping duplicate points when writing file.
    Tested only for add_point method, but would be the same
    for create equivalents
    """
    nr_points = model_config.water_level.nr_points

    # determine point in the middle of the grid
    gdf = model_config.region
    point = gdf.geometry.union_all().centroid

    model_config.water_level.add_point(
        x=point.x, y=point.y, value=-10.0, name="test_point"
    )

    assert model_config.water_level.nr_points == nr_points + 1

    # and again
    model_config.water_level.add_point(
        x=point.x, y=point.y, value=-10.0, name="test_point"
    )

    # by default drop_duplicates=True, so no point should be added
    assert model_config.water_level.nr_points == nr_points + 1

    # and again
    model_config.water_level.add_point(
        x=point.x, y=point.y, value=-10.0, name="test_point2", drop_duplicates=False
    )

    # now point should be added
    assert model_config.water_level.nr_points == nr_points + 2

    # don't need tide here:
    model_config.config.set("bcafile", None)

    # write water level to file
    model_config.root.set(tmp_dir, mode="w+")

    model_config.water_level.write()
    model_config.config.write()

    # read back-in to check if it remained the same
    mod2 = SfincsModel(root=tmp_dir, mode="r")
    mod2.config.read()
    mod2.water_level.read()

    # now write - here duplicates are not dropped so read in files are not changed
    assert len(mod2.water_level.data.index) == nr_points + 2


def test_create_timeseries(model_config):
    model_config.water_level.read()
    assert model_config.water_level.nr_points > 0

    # now add constant timeseries for each point
    model_config.water_level.create_timeseries(
        shape="constant",
        offset=10,
    )

    # Check that the timeseries is created correctly
    for idx in range(model_config.water_level.nr_points):
        point_data = model_config.water_level.data["bzs"].isel(index=idx)
        assert point_data.values.min() == 10
        assert point_data.values.max() == 10
        assert len(point_data.time) == 2

    # now add a Gaussian timeseries for the first point
    model_config.water_level.create_timeseries(
        index=0,
        shape="gaussian",
        offset=0,
        peak=5,
        tpeak=5 * 86400,
        duration=2 * 86400,
        timestep=3600,
    )

    # Check that the timeseries is created correctly
    point_data = model_config.water_level.data["bzs"].isel(index=0)
    assert np.isclose(point_data.values.min(), 0.1, atol=1e-2)
    assert np.isclose(point_data.values.max(), 5, atol=1e-2)
    assert len(point_data.time) == 49  # 49 hours with 1 hour timestep

    # also check that the min, max of the other points are still the same
    for idx in range(1, model_config.water_level.nr_points):
        point_data = model_config.water_level.data["bzs"].isel(index=idx)
        assert point_data.values.min() == 10
        assert point_data.values.max() == 10
        # but length has changed accordingly
        assert len(point_data.time) == 49

    # lastly add a sine timeseries for the second and third point
    model_config.water_level.create_timeseries(
        index=[1, 2],
        shape="sine",
        offset=0,
        amplitude=1,
        period=86400,
        timestep=3600,
    )
    # Check that the timeseries is created correctly
    for idx in range(1, model_config.water_level.nr_points):
        point_data = model_config.water_level.data["bzs"].isel(index=idx)
        assert point_data.values.min() == -1
        assert point_data.values.max() == 1
        # but length has changed accordingly
        assert len(point_data.time) == 49


def test_create_timeseries_from_astro(model_config):
    model_config.water_level.read()
    assert model_config.water_level.nr_points > 0

    # assert astronomic constituents are in the data
    assert "constituent" in model_config.water_level.data.dims

    # make sure all values are set to 0
    model_config.water_level.data["bzs"].values[:] = 0

    for idx in range(model_config.water_level.nr_points):
        point_data = model_config.water_level.data["bzs"].isel(index=idx)
        assert point_data.values.min() == 0
        assert point_data.values.max() == 0
        # but length remained the same
        assert len(point_data.time) == 289

    # now create timeseries from astro, with different freq then already present
    model_config.water_level.create_timeseries_from_astro(
        dt=300,
        offset=0,
    )

    # check that values changed back from 0s to real tidal values
    for idx in range(model_config.water_level.nr_points):
        point_data = model_config.water_level.data["bzs"].isel(index=idx)
        assert point_data.values.min() < 0
        assert point_data.values.max() > 0
        # length has changed accordingly
        assert len(point_data.time) == 577


def test_create(model_config):
    """Test creating discharge points from a GeoDataFrame and csv file."""
    src_file = Path(TESTMODELDIR) / "gis" / "bnd.geojson"

    print(src_file.resolve())
    print(src_file.exists())

    # Create discharge points from GeoDataFrame
    model_config.water_level.create(locations=src_file, merge=False)

    # Check that the number of points is correct
    assert model_config.water_level.nr_points == 2
    # show that dummy data is set
    for idx in range(0, model_config.water_level.nr_points):
        point_data = model_config.water_level.data["bzs"].sel(index=idx)
        assert point_data.values.min() == 0.0
        assert point_data.values.max() == 0.0
        assert len(point_data.time) == 2

    # now add timeseries from csv file, index in csv says 1
    csv_file = Path(TESTDATADIR) / "local_data" / "discharge.csv"
    model_config.water_level.create(timeseries=csv_file)
    # show that index 1 is changed into timeseries
    point_data = model_config.water_level.data["bzs"].sel(index=1)
    assert point_data.values.min() == 2.0
    assert point_data.values.max() == 5.0
    assert len(point_data.time) == 3

    # now copy the geodataarray and clear the data
    da = model_config.water_level.data.copy()
    model_config.water_level.clear()
    assert model_config.water_level.nr_points == 0

    # create a new discharge points object with the same data and check
    model_config.water_level.create(geodataset=da, merge=False)
    assert model_config.water_level.nr_points == 2
    # show that dummy data is set for point 0, 2 and timeseries for point 1
    for idx in range(model_config.water_level.nr_points):
        point_data = model_config.water_level.data["bzs"].sel(index=idx)
        if idx == 1:
            assert point_data.values.min() == 2.0
            assert point_data.values.max() == 5.0
        else:
            assert point_data.values.min() == 0.0
            assert point_data.values.max() == 0.0
        assert len(point_data.time) == 3

    # finally add points based on gdf and df
    gdf = model_config.region
    points_gdf = gdf.set_geometry(gdf.geometry.centroid)
    df = model_config.data_catalog.get_dataframe(
        csv_file,
        source_kwargs={
            "driver": {
                "name": "pandas",
                "options": {"index_col": 0, "parse_dates": True},
            }
        },
    )
    # alter it a bit to have different values, first with existing index,
    df = df.mul(2)
    df.columns = [2]
    points_gdf.index = [2]
    model_config.water_level.create(locations=points_gdf, timeseries=df, merge=True)
    # Check that the number of points is correct and values are set in the last point
    assert model_config.water_level.nr_points == 3
    assert model_config.water_level.data["bzs"].isel(index=-1).values.max() == 10.0

    # now with indices that do not exist yet; should be reset to 0
    df = df.mul(0.3)
    df.columns = [7]
    points_gdf.index = [7]
    model_config.water_level.create(locations=points_gdf, timeseries=df, merge=False)

    assert model_config.water_level.nr_points == 1
    assert model_config.water_level.data["bzs"].index[-1] == 0


def test_delete_clear(model_config):
    """Test deleting a discharge point from the model."""
    nr_points = model_config.water_level.nr_points

    # Delete the 2nd point
    model_config.water_level.delete(index=[1])

    # Check that the number of points has decreased
    assert model_config.water_level.nr_points == nr_points - 1

    # Try again, but make sure an error is raised since the point does not exist
    with pytest.raises(ValueError):
        model_config.water_level.delete(index=[1])

    # Delete all points
    model_config.water_level.clear()

    # Check that all points are deleted
    assert model_config.water_level.nr_points == 0
    assert model_config.config.get("bndfile") is None
    assert model_config.config.get("bzsfile") is None
    assert model_config.config.get("bcafile") is None


def test_netcdf_io(model_config, tmp_dir):
    """Test reading and writing water level boundary to netcdf file."""
    model_config.water_level.read()
    assert model_config.water_level.nr_points > 0

    data = model_config.water_level.data.copy()

    # write to netcdf file
    model_config.root.set(tmp_dir, mode="r+")
    netcdf_file = join(tmp_dir, "water_level_boundary.nc")
    # change in config to netcdf file
    model_config.config.update(
        {
            "bndfile": None,
            "bzsfile": None,
            "netbndbzsbzifile": str(netcdf_file),
        }
    )
    model_config.water_level.write()
    assert isfile(netcdf_file)

    # read back-in to check if it remained the same
    model_config.water_level.clear()
    assert model_config.water_level.nr_points == 0

    # set config and read
    model_config.config.set("netbndbzsbzifile", str(netcdf_file))
    model_config.water_level.read()
    assert model_config.water_level.nr_points > 0

    # assert the data is the same as before
    assert model_config.water_level.data.equals(data)
