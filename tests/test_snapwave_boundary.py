import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import isfile, join
import xarray as xr

from hydromt_sfincs import SfincsModel

from .conftest import TESTDATADIR, TESTMODELDIR


def test_snapwave_boundary_io(quadtree_model_config, tmp_dir):
    # assign the model_config to a local variable for easier access
    mod = quadtree_model_config

    # read snapwave boundary from files
    mod.quadtree_grid.read()
    mod.snapwave_boundary_conditions.read()
    assert mod.snapwave_boundary_conditions.data is not None
    assert len(mod.snapwave_boundary_conditions.data.index) == 2

    # write snapwave to file
    mod.root.set(tmp_dir, mode="w+")
    mod.write()

    assert isfile(tmp_dir / "snapwave.bnd")
    assert isfile(tmp_dir / "snapwave.bhs")
    assert isfile(tmp_dir / "snapwave.btp")
    assert isfile(tmp_dir / "snapwave.bwd")
    assert isfile(tmp_dir / "snapwave.bds")
    # assert isfile(tmp_dir, "snapwave.geojson")

    # read back-in to check if it remained the same
    mod = SfincsModel(root=mod.root.path, mode="r")
    mod.config.read()
    mod.snapwave_boundary_conditions.read()
    assert len(mod.snapwave_boundary_conditions.data.index) == 2
    assert mod.snapwave_boundary_conditions.test_equal(mod.snapwave_boundary_conditions)

    # now change the filename in the configuration
    mod.config.update(
        {
            "snapwave_bndfile": None,
            "snapwave_bhsfile": None,
            "snapwave_btpfile": None,
            "snapwave_bwdfile": None,
            "snapwave_bdsfile": None,
            "netsnapwavefile": "snapwave.nc",
        }
    )
    # delete the old files
    for f in [
        "snapwave.bnd",
        "snapwave.bhs",
        "snapwave.btp",
        "snapwave.bwd",
        "snapwave.bds",
    ]:
        file_path = join(tmp_dir, f)
        if isfile(file_path):
            Path(file_path).unlink()

    # write to netcdf file
    mod.root.set(tmp_dir, mode="w+")
    mod.snapwave_boundary_conditions.write()
    mod.config.write()
    assert isfile(tmp_dir / "snapwave.nc")
    assert not isfile(tmp_dir / "snapwave.bnd")
    assert not isfile(tmp_dir / "snapwave.bhs")
    assert not isfile(tmp_dir / "snapwave.btp")
    assert not isfile(tmp_dir / "snapwave.bwd")
    assert not isfile(tmp_dir / "snapwave.bds")

    # read back-in to check if it remained the same
    mod2 = SfincsModel(root=mod.root.path, mode="r")
    mod2.config.read()
    mod2.snapwave_boundary_conditions.read()
    assert len(mod2.snapwave_boundary_conditions.data.index) == 2
    assert mod2.snapwave_boundary_conditions.test_equal(
        mod.snapwave_boundary_conditions
    )


def test_add_point(quadtree_model_config):
    """Test adding a wave point to the model."""
    # assign the quadtree_model_config to a local variable for easier access
    mod = quadtree_model_config

    # read snapwave boundary from files
    mod.quadtree_grid.read()

    mod.snapwave_boundary_conditions.clear()
    nr_points = mod.snapwave_boundary_conditions.nr_points

    # determine point in the middle of the grid
    gdf = mod.region
    point = gdf.geometry.union_all().centroid

    mod.snapwave_boundary_conditions.add_point(
        x=point.x, y=point.y, hs=5.0, tp=12.0, wd=180.0, ds=30.0  # , name="test_point"
    )

    # Check that the number of points has increased and value is set correctly
    assert mod.snapwave_boundary_conditions.nr_points == nr_points + 1
    assert (
        np.mean(mod.snapwave_boundary_conditions.data["hs"].isel(index=-1).values)
        == 5.0
    )
    assert (
        np.mean(mod.snapwave_boundary_conditions.data["tp"].isel(index=-1).values)
        == 12.0
    )
    assert (
        np.mean(mod.snapwave_boundary_conditions.data["wd"].isel(index=-1).values)
        == 180.0
    )
    assert (
        np.mean(mod.snapwave_boundary_conditions.data["ds"].isel(index=-1).values)
        == 30.0
    )


def test_create_timeseries(quadtree_model_config):
    # assign the model_config to a local variable for easier access
    mod = quadtree_model_config

    mod.snapwave_boundary_conditions.read()
    assert mod.snapwave_boundary_conditions.nr_points > 0

    # now add constant timeseries for each point
    mod.snapwave_boundary_conditions.create_timeseries(
        shape="constant",
        hs=3,
        tp=12,
        wd=180,
        ds=30,
    )

    # Check that the timeseries is created correctly
    for idx in range(mod.snapwave_boundary_conditions.nr_points):
        point_data = mod.snapwave_boundary_conditions.data["hs"].isel(index=idx)
        assert point_data.values.min() == 3
        assert point_data.values.max() == 3
        assert len(point_data.time) == 2
    for idx in range(mod.snapwave_boundary_conditions.nr_points):
        point_data = mod.snapwave_boundary_conditions.data["tp"].isel(index=idx)
        assert point_data.values.min() == 12
        assert point_data.values.max() == 12

    # now add a Gaussian timeseries for the second point
    mod.snapwave_boundary_conditions.create_timeseries(
        index=1,  # only for second point
        shape="gaussian",
        timestep=3600,
        offset=2,
        hs=5,
        tp=14,
        wd=245,
        ds=25,
        tpeak=1.0 * 86400,
        duration=2.0 * 86400,
    )

    # Check that the timeseries is created correctly
    point_data = mod.snapwave_boundary_conditions.data["hs"].isel(index=1)
    assert np.isclose(point_data.values.min(), 2.0, atol=0.06)
    assert point_data.values.max() == 5
    point_data = mod.snapwave_boundary_conditions.data["tp"].isel(index=1)
    assert point_data.values.min() == 14
    assert point_data.values.max() == 14
    point_data = mod.snapwave_boundary_conditions.data["wd"].isel(index=1)
    assert point_data.values.min() == 245
    assert point_data.values.max() == 245
    point_data = mod.snapwave_boundary_conditions.data["ds"].isel(index=1)
    assert point_data.values.min() == 25
    assert point_data.values.max() == 25

    assert len(point_data.time) == 49  # 49 hours with 1 hour timestep

    # also check that the min, max of the other points are still the same
    point_data = mod.snapwave_boundary_conditions.data["hs"].isel(index=0)
    assert point_data.values.min() == 3
    assert point_data.values.max() == 3
    # but length has changed accordingly
    assert len(point_data.time) == 49


def test_create(quadtree_model_config):
    """Test creating discharge points from a GeoDataFrame and csv file."""
    # assign the quadtree_model_config to a local variable for easier access
    mod = quadtree_model_config

    # Model has data already; copy it and clear
    da = mod.snapwave_boundary_conditions.data.copy()
    mod.snapwave_boundary_conditions.clear()
    assert mod.snapwave_boundary_conditions.nr_points == 0

    # create a new wave input using the geodataset with the same data and check
    mod.snapwave_boundary_conditions.create(geodataset=da, merge=False)
    assert mod.snapwave_boundary_conditions.nr_points == 2
    # compare da to mod.snapwave_boundary_conditions.data
    assert mod.snapwave_boundary_conditions.data.equals(da)

    # FIXME - should have a check that if not all variables are provided into the
    # geodataset or timeseries, an error is raised

    # Model has data already; copy it and clear
    mod.snapwave_boundary_conditions.clear()
    src_file = Path(TESTMODELDIR) / "gis" / "bnd.geojson"

    # Add wave points from geojson file
    mod.snapwave_boundary_conditions.create(locations=src_file, merge=False)

    # Check that the number of points is correct
    assert mod.snapwave_boundary_conditions.nr_points == 2

    # show that dummy data is set to 0
    default_values = [1.0, 10.0, 270.0, 20.0]
    for idx in range(0, mod.snapwave_boundary_conditions.nr_points):
        # also for other variables in a loop
        for i, var in enumerate(["hs", "tp", "wd", "ds"]):
            point_data = mod.snapwave_boundary_conditions.data[var].sel(index=idx)
            assert point_data.values.min() == default_values[i]
            assert point_data.values.max() == default_values[i]
            assert len(point_data.time) == 2

    # now add timeseries from csv file, index in csv says 1
    hs_file = Path(TESTDATADIR) / "local_data" / "hs_bc.csv"
    tp_file = Path(TESTDATADIR) / "local_data" / "tp_bc.csv"
    wd_file = Path(TESTDATADIR) / "local_data" / "wd_bc.csv"
    ds_file = Path(TESTDATADIR) / "local_data" / "ds_bc.csv"

    mod.snapwave_boundary_conditions.create(
        timeseries=[hs_file, tp_file, wd_file, ds_file]
    )
    # show that index 1 is changed into timeseries
    point_data = mod.snapwave_boundary_conditions.data["hs"].sel(index=1)
    assert point_data.values.min() == 0.43
    assert point_data.values.max() == 2.42
    assert len(point_data.time) == 3

    # finally add points based on gdf and df
    gdf = mod.region
    points_gdf = gdf.set_geometry(gdf.geometry.centroid)
    points_gdf.index = [0]
    # points_gdf.index = [2] # FIXME - does not work, because set_locations resets indices to 0 - problem or not?
    points_gdf.index.name = "index"

    # make up a new df with timeseries data
    times = np.arange(
        np.datetime64("2010-02-05T00:00"),
        np.datetime64("2010-02-07T01:00"),
        np.timedelta64(1, "D"),
    )

    df_hs = np.array([1.0, 5.0, 15.0])
    df_hs = pd.DataFrame(data=df_hs, index=times, columns=[0])  # 2])
    df_hs.columns.name = "index"
    df_hs.index.name = "time"

    df_tp = np.array([10.0, 15.0, 11.0])
    df_tp = pd.DataFrame(data=df_tp, index=times, columns=[0])
    df_tp.columns.name = "index"
    df_tp.index.name = "time"

    df_wd = np.array([180.0, 190.0, 200.0])
    df_wd = pd.DataFrame(data=df_wd, index=times, columns=[0])
    df_wd.columns.name = "index"
    df_wd.index.name = "time"

    df_ds = np.array([30.0, 35.0, 25.0])
    df_ds = pd.DataFrame(data=df_ds, index=times, columns=[0])
    df_ds.columns.name = "index"
    df_ds.index.name = "time"

    mod.snapwave_boundary_conditions.create(
        locations=points_gdf,
        timeseries=[df_hs, df_tp, df_wd, df_ds],
        # merge=True #FIXME - does not work currently
        merge=False,
    )
    # Check that the number of points is correct and values are set in the last point
    assert mod.snapwave_boundary_conditions.nr_points == 1
    # check geometry is same as gdf.geometry.centroid
    assert mod.snapwave_boundary_conditions.data.geometry == gdf.geometry.centroid
    # check a value
    assert (
        mod.snapwave_boundary_conditions.data["hs"].isel(index=-1).values.max() == 15.0
    )

    mod.snapwave_boundary_conditions.create(
        locations=points_gdf,
        timeseries=[df_hs, df_tp, df_wd, df_ds],
        merge=True,
        drop_duplicates=False,
    )

    assert mod.snapwave_boundary_conditions.nr_points == 2


def test_create_from_grid(quadtree_model_config):
    """Test creating wave input from 2D gridded dataset like ERA5"""
    # assign the quadtree_model_config to a local variable for easier access
    mod = quadtree_model_config

    # Model has data already; copy it
    da_org = mod.snapwave_boundary_conditions.data.copy()

    # Now we overwrite the timeseries with data from a 2D gridded dataset
    mod.snapwave_boundary_conditions.create_from_grid(
        data="era5_dummy",
    )

    # Make sure that data is updated
    assert not mod.snapwave_boundary_conditions.data.equals(da_org)

    # mod.snapwave_boundary_conditions.create(geodataset=data, merge=False)

    # create a new wave input using the geodataset - directly from file
    # mod.snapwave_boundary_conditions.create(geodataset=filename, merge=False)


def test_delete_clear(quadtree_model_config):
    """Test deleting a wave point from the model."""
    # assign the quadtree_model_config to a local variable for easier access
    mod = quadtree_model_config

    nr_points = mod.snapwave_boundary_conditions.nr_points

    # Delete the 2nd point
    mod.snapwave_boundary_conditions.delete(index=[1])

    # Check that the number of points has decreased
    assert mod.snapwave_boundary_conditions.nr_points == nr_points - 1

    # Try again, but make sure an error is raised since the point does not exist
    with pytest.raises(ValueError):
        mod.snapwave_boundary_conditions.delete(index=[1])

    # Delete all points
    mod.snapwave_boundary_conditions.clear()

    # Check that all points are deleted
    assert mod.snapwave_boundary_conditions.nr_points == 0
    assert mod.config.get("snapwave_bndfile") is None
    assert mod.config.get("snapwave_bhsfile") is None
    assert mod.config.get("snapwave_btpfile") is None
    assert mod.config.get("snapwave_bwdfile") is None
    assert mod.config.get("snapwave_bdsfile") is None
    assert mod.config.get("netsnapwavefile") is None
