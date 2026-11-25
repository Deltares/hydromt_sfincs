import logging
from datetime import datetime
from os.path import abspath, dirname, isfile, join

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromt_sfincs.sfincs import SfincsModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_meteo_io(model_config, tmp_dir):
    # this model contains uniform wnd, so it should use read_uniform
    model_config.wind.read()
    assert "wind" in model_config.wind.data

    # this model contains gridded precipitation, so it should use read_gridded
    model_config.precipitation.read()
    assert "precip_2d" in model_config.precipitation.data

    # now change the root of the model to a temporary directory and write the data
    model_config.root.set(tmp_dir, mode="w")
    model_config.wind.write()
    fn = model_config.config.get("wndfile", abs_path=True)
    assert isfile(fn)
    model_config.precipitation.write()
    fn = model_config.config.get("netamprfile", abs_path=True)
    assert isfile(fn)
    model_config.config.write()

    # now read the data back in
    mod = SfincsModel(root=tmp_dir, mode="r")
    mod.config.read()
    mod.wind.read()
    assert mod.wind.test_equal(model_config.wind)
    mod.precipitation.read()
    assert mod.precipitation.test_equal(model_config.precipitation)


def test_create_uniform_precip(model_config):
    timeseries = join(TESTDATADIR, "local_data", "discharge.csv")

    model_config.precipitation.create_uniform(timeseries=timeseries)

    assert "precip" in model_config.precipitation.data


def test_set_precip_reindexes_existing_data_to_new_time_index(model_config):
    old_times = pd.date_range("2010-02-05", periods=2, freq="D")
    new_times = pd.date_range("2020-02-05", periods=2, freq="D")

    model_config.precipitation.set(
        xr.DataArray([1.0, 2.0], coords={"time": old_times}, dims="time"),
        name="precip",
    )
    model_config.precipitation.set(
        xr.DataArray([3.0, 4.0], coords={"time": new_times}, dims="time"),
        name="precip",
    )

    data = model_config.precipitation.data
    assert data.indexes["time"].equals(new_times)
    assert data["precip"].values.tolist() == [3.0, 4.0]


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("offset_minutes", [0, 30])
def test_create_precip_preserves_time_brackets(
    model_config, tmp_path, aggregate, offset_minutes
):
    model_config.grid.read()
    model_config.root.set(tmp_path, mode="w")
    tstart, tstop = model_config.get_model_time()
    offset = pd.Timedelta(minutes=offset_minutes)
    times = pd.date_range(
        tstart - pd.Timedelta(hours=2) + offset,
        tstop + pd.Timedelta(hours=2) + offset,
        freq="h",
    )
    expected_times = pd.date_range(tstart - offset, tstop + offset, freq="h")
    rates = pd.Series(np.arange(1, len(times) + 1, dtype=float), index=times)
    xmin, ymin, xmax, ymax = model_config.bbox
    rain = xr.DataArray(
        np.broadcast_to(rates.values[:, None, None], (len(times), 6, 6)).copy(),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": times,
            "latitude": np.linspace(ymax + 0.05, ymin - 0.05, 6),
            "longitude": np.linspace(xmin - 0.05, xmax + 0.05, 6),
        },
        name="precip",
        attrs={"units": "mm/hr"},
    )
    rain.raster.set_crs(4326)

    model_config.precipitation.create(
        precip=rain, cumulative_input=False, aggregate=aggregate
    )

    result = model_config.precipitation.data
    assert result.indexes["time"].equals(expected_times)
    if aggregate:
        actual_rates = result["precip"]
    else:
        assert result.raster.crs == model_config.crs
        assert result.raster.crs != rain.raster.crs
        assert result.raster.dims == ("y", "x")
        center = (
            model_config.region.to_crs(model_config.crs).geometry.union_all().centroid
        )
        actual_rates = result["precip_2d"].sel(x=center.x, y=center.y, method="nearest")
    np.testing.assert_allclose(actual_rates.values, rates.loc[expected_times].values)


def test_create_meteo_latlon(tmp_dir):
    region = join(TESTDATADIR, "region.geojson")

    # create a model instance with geographical coordinates
    mod = SfincsModel(root=tmp_dir, mode="w", data_libs=["artifact_data"])
    mod.grid.create_from_region(
        region={"geom": region},
        crs=4326,
        res=0.01,
        rotated=False,
        dec_origin=3,
    )
    mod.mask.create(include_polygon=region)

    # set the model time
    mod.config.update(
        {
            "tref": datetime(2010, 2, 1),
            "tstart": datetime(2010, 2, 5),
            "tstop": datetime(2010, 2, 7),
        }
    )

    # get the forcing data from the data catalog
    ds = mod.data_catalog.get_rasterdataset("era5_hourly_zarr")

    # ensure its coordinates are in lat/lon
    y_dim, x_dim = ds.raster.dims
    assert y_dim == "latitude"
    assert x_dim == "longitude"
    assert ds.raster.crs == "EPSG:4326"

    # now rename the wind variables to align with the hydromt-sfincs conventions
    ds = ds.rename({"u10": "wind10_u", "v10": "wind10_v"})

    # add precipitation, wind and presssure to the model
    mod.precipitation.create(precip=ds)
    assert "precip_2d" in mod.precipitation.data
    mod.wind.create(wind=ds)
    assert "wind10_u" in mod.wind.data
    assert "wind10_v" in mod.wind.data
    mod.pressure.create(press=ds)
    assert "press_2d" in mod.pressure.data

    # now check the dimensions of the forcing data of each model component
    # should allways be (time, y, x) following sfincs conventions
    components = [mod.precipitation, mod.wind, mod.pressure]
    for comp in components:
        assert comp.data is not None
        assert comp.data.raster.dims == ("y", "x")
        assert comp.data.raster.crs == "EPSG:4326"


def test_create_meteo_time_bounds(model_config, caplog):
    # get the forcing data from the data catalog
    ds = model_config.data_catalog.get_rasterdataset("era5_hourly_zarr")

    # forcing starts too late -> should warn
    ds_late = ds.sel(time=slice("2010-02-06", None))

    with caplog.at_level(logging.WARNING):
        model_config.precipitation.create(precip=ds_late)

    assert "forcing does not cover the full model period" in caplog.text

    # forcing covers tstart but ends before tstop -> should warn
    ds_short = ds.sel(time=slice("2010-02-01", "2010-02-06"))

    with caplog.at_level(logging.WARNING):
        model_config.precipitation.create(precip=ds_short)

    assert "forcing does not cover the full model period" in caplog.text
