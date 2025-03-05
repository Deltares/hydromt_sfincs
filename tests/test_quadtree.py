from datetime import datetime
from os.path import join, dirname, abspath
import numpy as np
import os
from pathlib import Path
from pyproj import CRS
import pytest
import shutil
import xarray as xr
import xugrid as xu

from hydromt_sfincs import utils
from hydromt_sfincs.quadtree import QuadtreeGrid

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_quadtree_io(tmp_dir):
    # copy the test data to the tmp_path
    fn = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    fn_copy = tmp_dir.joinpath("sfincs.nc")

    # Create file + copy
    shutil.copy(fn, fn_copy)
    print(fn, fn_copy)

    # Initialize a QuadtreeGrid object
    qtr = QuadtreeGrid()
    # Read a quadtree netcdf file
    qtr.read(fn_copy)
    # Check the face coordinates
    face_coordinates = qtr.face_coordinates
    assert len(face_coordinates[0] == 4452)
    # Check the msk variable
    msk = qtr.data["msk"]
    assert np.sum(msk.values) == 4298
    # Check the crs
    crs = qtr.crs
    assert crs == CRS.from_epsg(32633)

    # now write the quadtree to a new file
    fn = join(tmp_dir, "sfincs_out.nc")
    qtr.write(fn)

    # here we can still remove the file (but we dont want it)
    # os.remove(fn)

    # read the new file and check the msk variable
    qtr2 = QuadtreeGrid()
    qtr2.read(fn)
    # assert the crs is the same
    assert qtr2.crs == qtr.crs
    # assert the msk variable is the same
    assert np.sum(qtr2.data["msk"].values) == 4298
    # assert the dep variable is the same
    assert np.sum(qtr.data["dep"].values) == np.sum(qtr2.data["dep"].values)

    # remove the files, why is this failing?
    os.remove(fn)
    os.remove(fn_copy)


def test_overwrite_quadtree_nc(tmp_dir):
    fn = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    fn_copy = tmp_dir.joinpath("sfincs.nc")

    # Create file + copy
    shutil.copy(fn, fn_copy)
    print(fn, fn_copy)

    # Open the copy with xu_open_dataset
    # This opens the file lazily
    ds = utils.xu_open_dataset(fn_copy)

    # Convert to dataset
    ds = ds.ugrid.to_dataset()

    # Try to write
    # NOTE this should fail because it still has lazy references to the file
    with pytest.raises(PermissionError):
        ds.to_netcdf(fn_copy)

    # Now perform the check and lazy loading check
    utils.check_exists_and_lazy(ds, fn_copy)

    # Try to overwrite the file
    ds.to_netcdf(fn_copy)

    # Remove the copied file
    os.remove(fn_copy)


def test_utils_open_dataset(tmp_dir):
    # copy the test data to the tmp_path
    fn = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    fn_copy = tmp_dir.joinpath("sfincs.nc")

    shutil.copy(fn, fn_copy)

    ds = utils.xu_open_dataset(fn_copy)
    os.remove(fn_copy)


def test_xu_open_dataset(tmp_dir):
    # copy the test data to the tmp_path
    fn = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    fn_copy = tmp_dir.joinpath("sfincs.nc")

    shutil.copy(fn, fn_copy)

    ds = xu.open_dataset(fn_copy)
    ds.close()
    os.remove(fn_copy)
