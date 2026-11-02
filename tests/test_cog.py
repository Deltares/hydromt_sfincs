"""Tests for the topobathy and index COG workflows."""

from os.path import join

import numpy as np
import pytest
import rasterio

from hydromt_sfincs.workflows.cog import create_index_cog, create_topobathy_cog
from hydromt_sfincs.workflows.downscaling import make_index_cog

RES = 200.0
ELEVATION_LIST = [{"elevation": "merit_hydro"}]


@pytest.mark.parametrize("model_fixture", ["model", "quadtree_model"])
def test_topobathy_and_index_cog(model_fixture, request, tmp_path):
    mod = request.getfixturevalue(model_fixture)
    fn_topo = join(tmp_path, "topobathy.tif")
    fn_index = join(tmp_path, "index.tif")

    create_topobathy_cog(mod, ELEVATION_LIST, fn_topo, res=RES)

    xmin, ymin, xmax, ymax = mod.bounds
    with rasterio.open(fn_topo) as src:
        # north-up raster covering the model bounds, rounded out to RES
        assert src.transform.e < 0
        assert src.bounds.left <= xmin < src.bounds.left + RES
        assert src.bounds.right - RES < xmax <= src.bounds.right
        assert src.bounds.bottom <= ymin < src.bounds.bottom + RES
        assert src.bounds.top - RES < ymax <= src.bounds.top
        assert src.width == round((src.bounds.right - src.bounds.left) / RES)
        assert src.height == round((src.bounds.top - src.bounds.bottom) / RES)
        assert np.isnan(src.nodata)
        assert np.all(np.isfinite(src.read(1)))
        shape, transform = src.shape, src.transform

    create_index_cog(mod, fn_index, fn_topo)

    with rasterio.open(fn_index) as src:
        assert src.shape == shape
        assert src.transform == transform
        assert src.dtypes[0] == "uint32"
        assert src.nodata == 2147483647
        index = src.read(1)
        # some pixels fall inside and some outside the active grid
        assert np.any(index != src.nodata)
        assert np.any(index == src.nodata)


def test_topobathy_cog_river_list_none(model, tmp_path):
    fn = join(tmp_path, "topobathy.tif")
    model.grid_component.create_topobathy_cog(
        fn, ELEVATION_LIST, resolution=RES, river_list=None
    )
    with rasterio.open(fn) as src:
        assert src.width > 0 and src.height > 0


def test_make_index_cog_deprecated_keywords(model, tmp_path):
    fn_topo = join(tmp_path, "topobathy.tif")
    fn_index = join(tmp_path, "index.tif")
    create_topobathy_cog(model, ELEVATION_LIST, fn_topo, res=RES)

    # the deprecated alias must keep accepting the old keyword names
    with pytest.warns(FutureWarning):
        make_index_cog(model=model, indices_fn=fn_index, topobathy_fn=fn_topo)

    with rasterio.open(fn_index) as src, rasterio.open(fn_topo) as ref:
        assert src.shape == ref.shape
