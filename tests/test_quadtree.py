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

from hydromt_sfincs import utils
from hydromt_sfincs.quadtree import QuadtreeGrid

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_quadtree_io(tmp_dir):
    # copy the test data to the tmp_path
    fn = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    fn_copy = tmp_dir.joinpath("sfincs.nc")

    # Create file + copy
    shutil.copy(fn, fn_copy)

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

    # read the new file and check the msk variable
    qtr2 = QuadtreeGrid()
    qtr2.read(fn)
    # assert the crs is the same
    assert qtr2.crs == qtr.crs
    # assert the msk variable is the same
    assert np.sum(qtr2.data["msk"].values) == 4298
    # assert the dep variable is the same
    assert np.sum(qtr.data["dep"].values) == np.sum(qtr2.data["dep"].values)

    # remove the files, they both get locked because of loading after closure?
    os.remove(fn)
    os.remove(fn_copy)


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

    # # Now perform the check and lazy loading check
    # utils.check_exists_and_lazy(ds, fn_copy)

    # # Try to overwrite the file
    # ds.to_netcdf(fn_copy)

    # # Remove the copied file
    # os.remove(fn_copy)


# def test_lazy_xu_open_dataset(tmp_dir):
#     # copy the test data to the tmp_path
#     fn = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
#     fn_copy = tmp_dir.joinpath("sfincs.nc")

#     shutil.copy(fn, fn_copy)

#     # lazy load
#     ds = xu.open_dataset(fn_copy)
#     ds.close()

#     # further down the code, you would like to obtain the data
#     ds.load().close()

#     # now it can't be removed
#     with pytest.raises(PermissionError):
#         os.remove(fn_copy)
