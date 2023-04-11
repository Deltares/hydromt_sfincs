
import pytest
from  hydromt_sfincs.workflows import bathymetry
from shapely.geometry import LineString, Point
from  affine import Affine
import hydromt
import geopandas as gpd
import numpy as np


def test_bathymetry():
    # get some data
    data_cat = hydromt.DataCatalog('artifact_data')
    da_elv = data_cat.get_rasterdataset('merit_hydro', variables=['elevtn'])
    gdf_riv = data_cat.get_geodataframe('rivers_lin2019_v1')
    gdf_riv['rivwth'] = 200
    gdf_riv['rivdph'] = 4
    # 
    da_elv1 = bathymetry.burn_river_rect(
        da_elv, gdf_riv,
    )
    assert ((da_elv - da_elv1) >= 0).all()