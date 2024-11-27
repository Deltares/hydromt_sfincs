import pytest
from hydromt_sfincs.workflows.flwdir import (
    river_centerline_from_hydrography,
    river_source_points,
)
import geopandas as gpd
import numpy as np


def test_river_centerline_from_hydrography(hydrography):
    # get data
    da_flwdir, da_uparea, gdf_mask = hydrography
    # get river centerlines
    gdf_riv = river_centerline_from_hydrography(da_flwdir, da_uparea, gdf_mask=gdf_mask)
    # check
    assert isinstance(gdf_riv, gpd.GeoDataFrame)
    assert np.isin(["geometry", "rivlen", "uparea"], gdf_riv.columns).all()
    assert np.isclose(gdf_riv["rivlen"].max(), 19665.50)
    assert gdf_riv.index.size == 11
    # no rivers (uparea threshold too high)
    gdf_riv = river_centerline_from_hydrography(da_flwdir, da_uparea, river_upa=1e6)
    assert gdf_riv.empty
    # no rivers (river length threshold too high)
    gdf_riv = river_centerline_from_hydrography(da_flwdir, da_uparea, river_len=1e6)
    assert gdf_riv.empty


def test_river_source_points(hydrography, data_catalog):
    # get data
    da_flwdir, da_uparea, gdf_mask = hydrography
    gdf_mask = gdf_mask.to_crs("EPSG:32633")

    # test with derived centerline
    gdf_riv = river_centerline_from_hydrography(da_flwdir, da_uparea, gdf_mask=gdf_mask)
    kwargs = dict(gdf_riv=gdf_riv, gdf_mask=gdf_mask)
    gdf_src = river_source_points(src_type="inflow", **kwargs)
    assert gdf_src.index.size == 3
    gdf_src = river_source_points(src_type="headwater", **kwargs)
    assert gdf_src.index.size == 6
    gdf_src = river_source_points(src_type="outflow", da_uparea=da_uparea, **kwargs)
    assert gdf_src.index.size == 1
    assert np.isin(["geometry", "uparea"], gdf_src.columns).all()
    np.allclose(gdf_src.geometry[0].coords[:], [(322650.3, 5044385.7)])

    # test reverse oriented line
    gdf_riv = data_catalog.get_geodataframe("hydro_rivers_lin")
    kwargs = dict(gdf_riv=gdf_riv, gdf_mask=gdf_mask, reverse_river_geom=True)
    gdf_src = river_source_points(src_type="inflow", **kwargs)
    assert gdf_src.index.size == 1  # this data only one river
    np.allclose(gdf_src.geometry[0].coords[:], [(322650.3, 5044385.7)])
    gdf_src = river_source_points(src_type="outflow", **kwargs)
    np.allclose(gdf_src.geometry[0].coords[:], [(322554.0, 5044434.7)])
    assert gdf_src.index.size == 1  # this data only one river
    gdf_src = river_source_points(src_type="headwater", **kwargs)
    assert gdf_src.index.size == 2

    # test errors
    with pytest.raises(ValueError, match="src_type must be either"):
        gdf_src = river_source_points(
            gdf_riv=gdf_riv, gdf_mask=gdf_mask, src_type="wrong"
        )
    with pytest.raises(TypeError, match="gdf_mask must be"):
        gdf_src = river_source_points(gdf_riv=gdf_riv, gdf_mask=gdf_riv)
    with pytest.raises(TypeError, match="gdf_riv must be"):
        gdf_src = river_source_points(gdf_riv=gdf_mask, gdf_mask=gdf_mask)
