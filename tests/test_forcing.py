from os.path import abspath, dirname, join, isfile
from datetime import datetime

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
    model_config.root.set(tmp_dir, mode="w+")
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


def test_create_meteo_latlon(tmp_dir):
    region = join(TESTDATADIR, "region.geojson")

    # create a model instance with geographical coordinates
    mod = SfincsModel(root=tmp_dir, mode="w+", data_libs=["artifact_data"])
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
    mod.precipitation.create(precip=ds, buffer=10)
    assert "precip_2d" in mod.precipitation.data
    mod.wind.create(wind=ds, buffer=10)
    assert "wind10_u" in mod.wind.data
    assert "wind10_v" in mod.wind.data
    mod.pressure.create(press=ds, buffer=10)
    assert "press_2d" in mod.pressure.data

    # now check the dimensions of the forcing data of each model component
    # should allways be (time, y, x) following sfincs conventions
    components = [mod.precipitation, mod.wind, mod.pressure]
    for comp in components:
        assert comp.data is not None
        assert comp.data.raster.dims == ("y", "x")
        assert comp.data.raster.crs == "EPSG:4326"
