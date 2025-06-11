from os.path import abspath, dirname, join

import numpy as np

from hydromt_sfincs.sfincs import SfincsModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_setup_meteo_latlon(tmp_dir):
    region = join(TESTDATADIR, "region.geojson")

    # create a model instance with geographical coordinates
    mod = SfincsModel(root=tmp_dir, mode="w+")
    mod.setup_grid_from_region(
        region={"geom": region}, crs=4326, res=0.01, rotated=False, dec_origin=3
    )
    mod.setup_mask_active(mask=region)

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
    mod.setup_precip_forcing_from_grid(precip=ds)
    mod.setup_wind_forcing_from_grid(wind=ds)
    mod.setup_pressure_forcing_from_grid(press=ds)

    # first make sure the forcing data is set correctly
    assert ["precip_2d", "wind10_u", "wind10_v", "press_2d"] == list(mod.forcing.keys())

    # now check the dimensions of the forcing data
    for key, da in mod.forcing.items():
        # spatial coordinates should always be (y, x)
        assert da.dims == ("time", "y", "x")
        assert da.raster.crs == "EPSG:4326"
