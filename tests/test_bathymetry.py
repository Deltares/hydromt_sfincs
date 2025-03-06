import hydromt
import numpy as np
import xarray as xr

from hydromt_sfincs.workflows import bathymetry


def test_bathymetry():
    # get some data
    bbox = 12.64, 45.50, 12.81, 45.59
    data_cat = hydromt.DataCatalog("artifact_data")
    da_elv0 = data_cat.get_rasterdataset("merit_hydro", variables=["elevtn"], bbox=bbox)
    da_man0 = xr.zeros_like(da_elv0)
    da_elv = da_elv0.raster.reproject(dst_crs="utm", dst_res=30).load()
    da_man = xr.zeros_like(da_elv)
    gdf_riv = data_cat.get_geodataframe("hydro_rivers_lin", bbox=bbox)
    da_mask = (data_cat.get_rasterdataset("grwl_mask", bbox=bbox) > 0).astype(np.uint8)
    da_mask.raster.set_nodata(0)
    gdf_mask = da_mask.raster.vectorize()
    gdf_riv["rivwth"] = 100
    gdf_riv["rivdph"] = 4
    gdf_riv["manning"] = 0.035
    gdf_riv["rivbed"] = -9
    # test with mask > rivwth not used
    da_elv1, da_man1 = bathymetry.burn_river_rect(
        da_elv=da_elv,
        da_man=da_man,
        gdf_riv=gdf_riv.drop(columns="rivbed"),
        gdf_riv_mask=gdf_mask,
    )
    diff = (da_elv - da_elv1).load()
    assert (diff >= 0).all()
    assert (diff > 0).sum() == 2124
    assert np.allclose(da_man1.values[diff.values > 0], 0.035)
    # test without mask
    da_elv1, _ = bathymetry.burn_river_rect(
        da_elv=da_elv0,
        da_man=None,
        gdf_riv=gdf_riv.drop(columns="rivbed"),
    )
    diff = (da_elv0 - da_elv1).load()
    assert (diff > 0).sum() == 292
    # test with rivbed
    da_elv1, _ = bathymetry.burn_river_rect(
        da_elv=da_elv0,
        da_man=None,
        gdf_riv=gdf_riv,
    )
    assert np.allclose(da_elv1.values[diff.values > 0], -9)
    # test with zb
    points = gdf_riv.geometry.interpolate(0.5, normalized=True)
    gdf_zb = gdf_riv.assign(geometry=points)
    da_elv1, da_man1 = bathymetry.burn_river_rect(
        da_elv=da_elv0,
        da_man=da_man0,
        gdf_riv=gdf_riv,
        # get manning from zb
        gdf_zb=gdf_zb.drop(columns="manning"),
    )
    diff = (da_elv0 - da_elv1).load()
    assert (diff > 0).sum() == 292
    assert np.allclose(da_man1.values[diff.values > 0], 0.035)
