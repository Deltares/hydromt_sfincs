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


def test_quadtree_write_excludes_component_layers(quadtree_model):
    # manning is owned by the roughness component, so it must not end up in sfincs.nc
    mask = quadtree_model.quadtree_grid.data["mask"]
    manning = mask.astype(np.float32)
    manning.values = np.full(mask.values.shape, 0.03, dtype=np.float32)
    quadtree_model.quadtree_grid.set(manning, name="manning")

    quadtree_model.quadtree_grid.write()
    quadtree_model.quadtree_roughness.write()
    quadtree_model.config.write()

    qtrfile = quadtree_model.root.path / "sfincs.nc"
    manningfile = quadtree_model.config.get("manningfile", abs_path=True)
    assert manningfile.is_file()

    with xr.open_dataset(qtrfile) as ds:
        assert "mask" in ds.data_vars
        assert "manning" not in ds.data_vars
    with xr.open_dataset(manningfile) as ds:
        assert "manning" in ds.data_vars

    # the component pulls the grid in itself, so no quadtree_grid.read() here
    mod1 = SfincsModel(root=quadtree_model.root.path, mode="r")
    mod1.config.read()
    mod1.quadtree_roughness.read()
    assert "manning" in mod1.quadtree_grid.data
    assert np.allclose(mod1.quadtree_grid.data["manning"].values, 0.03)


def test_quadtree_layer_file_is_ugrid(quadtree_model):
    # per-layer files must stay inspectable in QGIS, which needs the UGRID
    # topology and the CF metadata MDAL looks for
    mask = quadtree_model.quadtree_grid.data["mask"]
    vol = mask.astype(np.float32)
    vol.values = np.full(mask.values.shape, 1.5, dtype=np.float32)
    quadtree_model.quadtree_grid.set(vol, name="vol")

    quadtree_model.quadtree_storage_volume.write()
    volfile = quadtree_model.config.get("volfile", abs_path=True)

    with xr.open_dataset(volfile) as ds:
        assert "UGRID" in ds.attrs.get("Conventions", "")
        assert "mesh2d_node_x" in ds.variables
        assert "mesh2d_node_y" in ds.variables
        assert ds["mesh2d_crs"].attrs["epsg_code"].startswith("EPSG:")


def test_quadtree_initial_conditions_io(quadtree_model):
    # zs was historically written to `inifile` but read from `zsfile`; this pins
    # that both sides now use the same key
    mask = quadtree_model.quadtree_grid.data["mask"]
    zs = mask.astype(np.float32)
    zs.values = np.full(mask.values.shape, 0.75, dtype=np.float32)
    quadtree_model.quadtree_grid.set(zs, name="zs")

    quadtree_model.quadtree_grid.write()
    quadtree_model.quadtree_initial_conditions.write()
    quadtree_model.config.write()

    inifile = quadtree_model.config.get("inifile", abs_path=True)
    assert inifile.is_file()
    with xr.open_dataset(quadtree_model.root.path / "sfincs.nc") as ds:
        assert "zs" not in ds.data_vars

    mod1 = SfincsModel(root=quadtree_model.root.path, mode="r")
    mod1.config.read()
    mod1.quadtree_initial_conditions.read()
    assert "zs" in mod1.quadtree_grid.data
    assert np.allclose(mod1.quadtree_grid.data["zs"].values, 0.75)


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
