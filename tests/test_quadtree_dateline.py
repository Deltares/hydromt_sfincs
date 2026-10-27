"""Tests for quadtree grids across the dateline: refinement and map overlays."""

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import Polygon, box

from hydromt_sfincs import SfincsModel
from hydromt_sfincs.components.quadtree.quadtree_builder import shift_polygon_to_grid
from hydromt_sfincs.workflows import map_overlay as mo

# Geographic grid of 20 x 20 cells of 1 degree, from 170 to 190 degrees east
X0, Y0, NMAX, MMAX, DX = 170.0, -10.0, 20, 20, 1.0
NR_REFINED = 948  # refined cells for a 4 x 10 degree level-2 polygon inside the grid


def build(tmp_path, polygon, x0=X0, mmax=MMAX):
    model = SfincsModel(root=str(tmp_path), mode="w")
    gdf = None
    if polygon is not None:
        gdf = gpd.GeoDataFrame({"refinement_level": [2]}, geometry=[polygon], crs=4326)
    model.quadtree_grid.create(
        x0, Y0, NMAX, mmax, DX, DX, 0.0, epsg=4326, refinement_polygons=gdf
    )
    return model.quadtree_grid.data


@pytest.mark.parametrize(
    "polygon",
    [
        box(172, -5, 176, 5),  # west of the dateline
        box(182, -5, 186, 5),  # east, same numbering as the grid
        box(-178, -5, -174, 5),  # east, numbered -180..180
        box(178, -5, 182, 5),  # across the dateline, continuous
        Polygon([(178, -5), (-178, -5), (-178, 5), (178, 5)]),  # across, wrapped
    ],
    ids=["west", "east", "east-wrapped", "across", "across-wrapped"],
)
def test_refinement_any_longitude_convention(tmp_path, polygon):
    level = build(tmp_path, polygon)["level"].values
    assert int((level > 1).sum()) == NR_REFINED


def test_refinement_grid_in_negative_numbering(tmp_path):
    level = build(tmp_path, box(182, -5, 186, 5), x0=-190.0)["level"].values
    assert int((level > 1).sum()) == NR_REFINED


def test_polygon_outside_grid_does_not_crash(tmp_path):
    level = build(tmp_path, box(100, -5, 104, 5))["level"].values
    assert len(level) == NMAX * MMAX
    assert not np.any(level > 1)


def test_shift_polygon_to_grid():
    shifted = shift_polygon_to_grid(box(-178, -5, -174, 5), 170.0, 190.0)
    assert shifted.bounds == pytest.approx((182.0, -5.0, 186.0, 5.0))
    assert shift_polygon_to_grid(box(172, -5, 176, 5), 170.0, 190.0).equals(
        box(172, -5, 176, 5)
    )
    wrapped = Polygon([(178, -5), (-178, -5), (-178, 5), (178, 5)])
    assert shift_polygon_to_grid(wrapped, 170.0, 190.0).bounds == pytest.approx(
        (178.0, -5.0, 182.0, 5.0)
    )


def overlay_columns(png):
    """Return the first and last pixel column that contain something."""
    from PIL import Image

    alpha = np.array(Image.open(png))[..., 3]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    return int(cols.min()), int(cols.max())


@pytest.mark.parametrize(
    "x0, view, expected",
    [
        (-188.0, [-260.0, -100.0], (360, 500)),  # grid and view west of -180
        (170.0, [-260.0, -100.0], (350, 490)),  # grid 0-360, view west of -180
        (-188.0, [150.0, -150.0], (293, 667)),  # view given across the dateline
    ],
)
def test_grid_overlay_across_dateline(tmp_path, x0, view, expected):
    pytest.importorskip("datashader")
    data = build(tmp_path / "model", None, x0=x0, mmax=28)
    df = mo.make_edge_dataframe(data.grid, CRS(4326))
    png = str(tmp_path / "grid.png")
    assert mo.make_map_overlay(df, png, xlim=view, ylim=[-30.0, 30.0], width=800)
    first, last = overlay_columns(png)
    # Only the grid itself is drawn (no lines across the whole map)
    assert abs(first - expected[0]) <= 2 and abs(last - expected[1]) <= 2


def test_edges_projected_grid_across_dateline(tmp_path):
    # UTM zone 60S grid spanning about 178.9E to 180.8E
    model = SfincsModel(root=str(tmp_path), mode="w")
    model.quadtree_grid.create(
        700000.0, 7700000.0, 20, 20, 10000.0, 10000.0, 0.0, epsg=32760
    )
    df = mo.make_edge_dataframe(model.quadtree_grid.data.grid, CRS(32760))
    xmin = min(df.x1.min(), df.x2.min())
    xmax = max(df.x1.max(), df.x2.max())
    span_degrees = (xmax - xmin) / mo.WORLD_WIDTH * 360.0
    assert span_degrees == pytest.approx(1.94, abs=0.05)
