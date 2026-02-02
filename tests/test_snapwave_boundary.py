import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from os.path import isfile, join
import xarray as xr

from hydromt_sfincs import SfincsModel

from .conftest import TESTDATADIR, TESTMODELDIR


def test_snapwave_boundary_io(model_config, tmp_dir):
    model_config = SfincsModel(root=join(TESTDATADIR, "sfincs_test_quadtree"), mode="r")

    # read snapwave boundary from files
    model_config.quadtree_grid.read()
    model_config.snapwave_boundary_conditions.read()
    assert model_config.snapwave_boundary_conditions.data is not None
    assert len(model_config.snapwave_boundary_conditions.data.index) == 2

    # write snapwave to file
    model_config.root.set(tmp_dir, mode="w+")
    model_config.write()

    assert isfile(tmp_dir / "snapwave.bnd")
    assert isfile(tmp_dir / "snapwave.bhs")
    assert isfile(tmp_dir / "snapwave.btp")
    assert isfile(tmp_dir / "snapwave.bwd")
    assert isfile(tmp_dir / "snapwave.bds")
    # assert isfile(tmp_dir, "snapwave.geojson")

    # read back-in to check if it remained the same
    mod = SfincsModel(root=model_config.root.path, mode="r")
    mod.config.read()
    mod.snapwave_boundary_conditions.read()
    assert len(mod.snapwave_boundary_conditions.data.index) == 2
    assert mod.snapwave_boundary_conditions.test_equal(
        model_config.snapwave_boundary_conditions
    )

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
        model_config.snapwave_boundary_conditions
    )


def test_add_point(model_config):
    """Test adding a wave point to the model."""
    model_config = SfincsModel(root=join(TESTDATADIR, "sfincs_test_quadtree"), mode="r")

    # read snapwave boundary from files
    model_config.quadtree_grid.read()

    model_config.snapwave_boundary_conditions.clear()
    nr_points = model_config.snapwave_boundary_conditions.nr_points

    # determine point in the middle of the grid
    gdf = model_config.region
    point = gdf.geometry.unary_union.centroid

    model_config.snapwave_boundary_conditions.add_point(
        x=point.x, y=point.y, hs=5.0, tp=12.0, wd=180.0, ds=30.0  # , name="test_point"
    )

    # Check that the number of points has increased and value is set correctly
    assert model_config.snapwave_boundary_conditions.nr_points == nr_points + 1
    assert (
        np.mean(
            model_config.snapwave_boundary_conditions.data["hs"].isel(index=-1).values
        )
        == 5.0
    )
    assert (
        np.mean(
            model_config.snapwave_boundary_conditions.data["tp"].isel(index=-1).values
        )
        == 12.0
    )
    assert (
        np.mean(
            model_config.snapwave_boundary_conditions.data["wd"].isel(index=-1).values
        )
        == 180.0
    )
    assert (
        np.mean(
            model_config.snapwave_boundary_conditions.data["ds"].isel(index=-1).values
        )
        == 30.0
    )


def test_create_timeseries(model_config):
    model_config = SfincsModel(root=join(TESTDATADIR, "sfincs_test_quadtree"), mode="r")

    model_config.snapwave_boundary_conditions.read()
    assert model_config.snapwave_boundary_conditions.nr_points > 0

    # now add constant timeseries for each point
    model_config.snapwave_boundary_conditions.create_timeseries(
        shape="constant",
        hs=3,
        tp=12,
        wd=180,
        ds=30,
    )

    # Check that the timeseries is created correctly
    for idx in range(model_config.snapwave_boundary_conditions.nr_points):
        point_data = model_config.snapwave_boundary_conditions.data["hs"].isel(
            index=idx
        )
        assert point_data.values.min() == 3
        assert point_data.values.max() == 3
        assert len(point_data.time) == 2
    for idx in range(model_config.snapwave_boundary_conditions.nr_points):
        point_data = model_config.snapwave_boundary_conditions.data["tp"].isel(
            index=idx
        )
        assert point_data.values.min() == 12
        assert point_data.values.max() == 12

    # now add a Gaussian timeseries for the second point
    model_config.snapwave_boundary_conditions.create_timeseries(
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
    point_data = model_config.snapwave_boundary_conditions.data["hs"].isel(index=1)
    assert np.isclose(point_data.values.min(), 2.0, atol=0.06)
    assert point_data.values.max() == 5
    point_data = model_config.snapwave_boundary_conditions.data["tp"].isel(index=1)
    assert point_data.values.min() == 14
    assert point_data.values.max() == 14
    point_data = model_config.snapwave_boundary_conditions.data["wd"].isel(index=1)
    assert point_data.values.min() == 245
    assert point_data.values.max() == 245
    point_data = model_config.snapwave_boundary_conditions.data["ds"].isel(index=1)
    assert point_data.values.min() == 25
    assert point_data.values.max() == 25

    assert len(point_data.time) == 49  # 49 hours with 1 hour timestep

    # also check that the min, max of the other points are still the same
    point_data = model_config.snapwave_boundary_conditions.data["hs"].isel(index=0)
    assert point_data.values.min() == 3
    assert point_data.values.max() == 3
    # but length has changed accordingly
    assert len(point_data.time) == 49


def test_create(model_config):
    """Test creating discharge points from a GeoDataFrame and csv file."""

    model_config = SfincsModel(root=join(TESTDATADIR, "sfincs_test_quadtree"), mode="r")
    model_config.read()
    # Create wave input from GeoDataSet

    # Model has data already; copy it and clear
    da = model_config.snapwave_boundary_conditions.data.copy()
    model_config.snapwave_boundary_conditions.clear()
    assert model_config.snapwave_boundary_conditions.nr_points == 0

    # create a new wave input using the geodataset with the same data and check
    model_config.snapwave_boundary_conditions.create(geodataset=da, merge=False)
    assert model_config.snapwave_boundary_conditions.nr_points == 2
    # compare da to model_config.snapwave_boundary_conditions.data
    assert model_config.snapwave_boundary_conditions.data.equals(da)

    # FIXME - should have a check that if not all variables are provided into the
    # geodataset or timeseries, an error is raised

    # Model has data already; copy it and clear
    model_config.snapwave_boundary_conditions.clear()
    src_file = Path(TESTMODELDIR) / "gis" / "bnd.geojson"

    # Add wave points from geojson file
    model_config.snapwave_boundary_conditions.create(locations=src_file, merge=False)

    # Check that the number of points is correct
    assert model_config.snapwave_boundary_conditions.nr_points == 2

    # show that dummy data is set to 0
    for idx in range(0, model_config.snapwave_boundary_conditions.nr_points):
        # also for other variables in a loop
        for var in ["hs", "tp", "wd", "ds"]:
            point_data = model_config.snapwave_boundary_conditions.data[var].sel(
                index=idx
            )
            assert point_data.values.min() == 0.0
            assert point_data.values.max() == 0.0
            assert len(point_data.time) == 2

    # now add timeseries from csv file, index in csv says 1
    hs_file = Path(TESTDATADIR) / "local_data" / "hs_bc.csv"
    tp_file = Path(TESTDATADIR) / "local_data" / "tp_bc.csv"
    wd_file = Path(TESTDATADIR) / "local_data" / "wd_bc.csv"
    ds_file = Path(TESTDATADIR) / "local_data" / "ds_bc.csv"

    model_config.snapwave_boundary_conditions.create(
        timeseries=[hs_file, tp_file, wd_file, ds_file]
    )
    # show that index 1 is changed into timeseries
    point_data = model_config.snapwave_boundary_conditions.data["hs"].sel(index=1)
    assert point_data.values.min() == 0.43
    assert point_data.values.max() == 2.42
    assert len(point_data.time) == 3

    # finally add points based on gdf and df
    gdf = model_config.region
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

    model_config.snapwave_boundary_conditions.create(
        locations=points_gdf,
        timeseries=[df_hs, df_tp, df_wd, df_ds],
        # merge=True #FIXME - does not work currently
        merge=False,
    )
    # Check that the number of points is correct and values are set in the last point
    assert model_config.snapwave_boundary_conditions.nr_points == 1
    # check geometry is same as gdf.geometry.centroid
    assert (
        model_config.snapwave_boundary_conditions.data.geometry == gdf.geometry.centroid
    )
    # check a value
    assert (
        model_config.snapwave_boundary_conditions.data["hs"].isel(index=-1).values.max()
        == 15.0
    )

    # FIXME now try merging again > does not work yet
    # model_config.snapwave_boundary_conditions.create(
    #     locations=points_gdf,
    #     timeseries=[df_hs, df_tp, df_wd, df_ds],
    #     merge=True, #FIXME - does not work currently
    #     drop_duplicates = False,
    #     # merge=False,
    # )
    # # # now with indices that do not exist yet; should be reset to 0
    # df = df.mul(0.3)
    # df.columns = [7]
    # points_gdf.index = [7]
    # model_config.snapwave_boundary_conditions.create(
    #     locations=points_gdf, timeseries=df, merge=False
    # )

    # assert model_config.snapwave_boundary_conditions.nr_points == 1
    # assert model_config.snapwave_boundary_conditions.data["hs"].index[-1] == 0


def test_create_from_grid(model_config):
    """Test creating wave input from 2D gridded dataset like ERA5"""

    model_config = SfincsModel(root=join(TESTDATADIR, "sfincs_test_quadtree"), mode="r")
    model_config.read()

    # Create wave input from GeoDataSet
    filename = join(model_config.root.path, "ERA5_dummy_input_withcrs.nc")

    # Model has data already; copy it and clear
    da_org = model_config.snapwave_boundary_conditions.data.copy()
    model_config.snapwave_boundary_conditions.clear()
    assert model_config.snapwave_boundary_conditions.nr_points == 0

    # create a new wave input using the geodataset - manually renaming

    da = xr.open_dataset(filename)

    # Rename variables to match snapwave names
    da = da.rename({"swh": "hs"})
    da = da.rename({"pp1d": "tp"})
    da = da.rename({"mwd": "wd"})
    da = da.rename({"wdw": "ds"})
    # da.raster.set_crs(4326)
    da.vector.set_crs(4326)

    model_config.snapwave_boundary_conditions.create(geodataset=da, merge=False)
    # model_config.snapwave_boundary_conditions.create(geodataset=data, merge=False)

    # create a new wave input using the geodataset - directly from file
    # model_config.snapwave_boundary_conditions.create(geodataset=filename, merge=False)


def test_delete_clear(model_config):
    """Test deleting a wave point from the model."""
    model_config = SfincsModel(root=join(TESTDATADIR, "sfincs_test_quadtree"), mode="r")
    model_config.read()

    nr_points = model_config.snapwave_boundary_conditions.nr_points

    # Delete the 2nd point
    model_config.snapwave_boundary_conditions.delete(index=[1])

    # Check that the number of points has decreased
    assert model_config.snapwave_boundary_conditions.nr_points == nr_points - 1

    # Try again, but make sure an error is raised since the point does not exist
    with pytest.raises(ValueError):
        model_config.snapwave_boundary_conditions.delete(index=[1])

    # Delete all points
    model_config.snapwave_boundary_conditions.clear()

    # Check that all points are deleted
    assert model_config.snapwave_boundary_conditions.nr_points == 0
    assert model_config.config.get("snapwave_bndfile") is None
    assert model_config.config.get("snapwave_bhsfile") is None
    assert model_config.config.get("snapwave_btpfile") is None
    assert model_config.config.get("snapwave_bwdfile") is None
    assert model_config.config.get("snapwave_bdsfile") is None
    assert model_config.config.get("netsnapwavefile") is None
