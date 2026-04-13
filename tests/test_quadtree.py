from datetime import datetime
import gc
from os.path import join, dirname, abspath
import numpy as np
import os
from pathlib import Path
from pyproj import CRS
import pytest
import shutil
import xarray as xr
import xugrid as xu

from hydromt_sfincs import SfincsModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_quadtree_io(tmp_dir):
    # Start with model to make sure the root is set
    mod0 = SfincsModel(root=join(TESTDATADIR, "sfincs_test_quadtree"), mode="r")

    # read the config
    mod0.config.read()
    # read the quadtree grid from the netcdf file
    mod0.quadtree_grid.read()
    # Check the face coordinates
    face_coordinates = mod0.quadtree_grid.face_coordinates
    assert len(face_coordinates[0] == 4452)
    # Check the mask variable
    msk = mod0.quadtree_grid.data["mask"]
    assert np.sum(msk.values) == 4298
    # Check the crs
    crs = mod0.quadtree_grid.crs
    assert crs == CRS.from_epsg(32633)

    # now write the quadtree to a new location
    mod0.root.set(tmp_dir, mode="w+")
    mod0.quadtree_grid.write()
    mod0.config.write()

    # now read the quadtree from the new location
    mod1 = SfincsModel(root=mod0.root.path, mode="r")
    # read the new file and check the msk variable
    mod1.config.read()
    mod1.quadtree_grid.read()
    # assert the crs is the same
    assert mod1.quadtree_grid.crs == mod0.quadtree_grid.crs
    # assert the msk variable is the same
    assert np.sum(mod1.quadtree_grid.data["mask"].values) == 4298
    # assert the dep variable is the same
    assert np.sum(mod0.quadtree_grid.data["z"].values) == np.sum(
        mod1.quadtree_grid.data["z"].values
    )

    # remove the files, they both get locked because of loading after closure?
    os.remove(mod1.root.path / "sfincs.nc")


def test_quadtree_create_index_tiles(quadtree_model, tmp_dir):
    # Zoom range has to be high enough that tile pixels resolve the
    # ~13 x 10 km test model; keep it tight so the test stays fast.
    zoom_range = [12, 13]

    # PNG (default format)
    root_png = tmp_dir / "tiles_png"
    quadtree_model.quadtree_grid.create_index_tiles(
        root=root_png, zoom_range=zoom_range
    )
    png_files = list((root_png / "indices").rglob("*.png"))
    assert len(png_files) > 0
    # Tiles are nested as <indices>/<zoom>/<x>/<y>.png
    zoom_levels = {int(p.parts[-3]) for p in png_files}
    assert zoom_levels.issubset(set(range(zoom_range[0], zoom_range[1] + 1)))
    assert max(zoom_levels) == zoom_range[1]
    # HTML viewer is written by default alongside PNG tiles
    html_file = root_png / "indices" / "index.html"
    assert html_file.is_file()
    assert "{z}/{x}/{y}.png" in html_file.read_text()

    # Binary format
    root_bin = tmp_dir / "tiles_bin"
    quadtree_model.quadtree_grid.create_index_tiles(
        root=root_bin, zoom_range=zoom_range, fmt="bin"
    )
    dat_files = list((root_bin / "indices").rglob("*.dat"))
    assert len(dat_files) > 0
    # Each .dat tile should be 256*256 int32 = 262144 bytes
    assert dat_files[0].stat().st_size == 256 * 256 * 4
    # Indices in the .dat tile must be valid cell indices (or -999 nodata)
    data = np.fromfile(dat_files[0], dtype=np.int32).reshape(256, 256)
    nr_cells = len(quadtree_model.quadtree_grid.data["level"])
    valid = data[data != -999]
    assert valid.size > 0
    assert valid.min() >= 0
    assert valid.max() < nr_cells


def test_xu_open_dataset_delete(tmp_dir):
    # copy the test data to the tmp_path
    fn = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    fn_copy = tmp_dir.joinpath("sfincs.nc")

    shutil.copy(fn, fn_copy)

    ds = xu.open_dataset(fn_copy)
    ds.close()
    os.remove(fn_copy)


def test_xu_open_dataset_overwrite(tmp_dir):
    # copy the test data to the tmp_path
    fn = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    fn_copy = tmp_dir.joinpath("sfincs.nc")

    shutil.copy(fn, fn_copy)

    # lazy load
    ds = xu.open_dataset(fn_copy)
    ds.close()

    # now perform a computation on the dataset
    ds = ds.ugrid.to_dataset()

    # NOTE this will raise a PermissionError because the file is lazily loaded
    with pytest.raises(PermissionError):
        ds.to_netcdf(fn_copy)
