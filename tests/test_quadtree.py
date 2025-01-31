import os
import shutil
from os.path import abspath, dirname, join

import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pyproj import CRS

from hydromt_sfincs import utils
from hydromt_sfincs.quadtree import QuadtreeGrid

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_quadtree_io(tmpdir):
    # Initialize a QuadtreeGrid object
    qtr = QuadtreeGrid()
    # Read a quadtree netcdf file
    qtr.read(join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc"))
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
    fn = tmpdir.join("sfincs_out.nc")
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


def test_overwrite_quadtree_nc(tmpdir):
    ncfile = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    nc_copy = join(str(tmpdir), "sfincs.nc")

    # Create file + copy
    shutil.copy(ncfile, nc_copy)

    # Open the copy with xu_open_dataset, creating a lazy handler pointing to the original file and a UGridDataset
    uds = utils.xu_open_dataset(nc_copy)

    # Convert to dataset
    ds = uds.ugrid.to_dataset()
    # Note that `ds` here is a different ds that doesnt have a reference to the file, so closing it is not possible

    # # Try to write
    # NOTE this should fail because lazy_file still has lazy references to the file
    with pytest.raises(PermissionError):
        ds.to_netcdf(nc_copy)

    # Would like to close the file but cannot because this ds is not the same as the one opened with xu_open_dataset
    ds.close()

    # Try to overwrite the file
    # Note that this fails because not all data was read in!
    ds.to_netcdf(nc_copy)

    # Remove the copied file
    os.remove(nc_copy)


def test_overwrite_quadtree_nc_with_load_dataset(tmpdir):
    ncfile = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    nc_copy = join(str(tmpdir), "sfincs.nc")

    # Create file + copy
    shutil.copy(ncfile, nc_copy)

    # This opens, loads and closes the file
    full_ds = xr.load_dataset(nc_copy)
    uds = xu.UgridDataset(full_ds)

    # Convert to dataset
    ds = uds.ugrid.to_dataset()

    # Try to overwrite the file
    ds.to_netcdf(nc_copy)

    # Remove the copied file
    os.remove(nc_copy)


def test_overwrite_quadtree_nc_with_open_dataset_contextmanager(tmpdir):
    ncfile = join(TESTDATADIR, "sfincs_test_quadtree", "sfincs.nc")
    nc_copy = join(str(tmpdir), "sfincs.nc")

    # Create file + copy
    shutil.copy(ncfile, nc_copy)

    # This is essentially the same as xu_open_dataset, but with xr.open_dataset
    with xr.open_dataset(nc_copy) as lazy_ds:
        uds = xu.UgridDataset(lazy_ds)

        # Convert to dataset
        ds = uds.ugrid.to_dataset()

        with pytest.raises(PermissionError):
            # This fails because the file is open
            ds.to_netcdf(nc_copy)

    # File is now closed
    os.remove(nc_copy)

    with pytest.raises(KeyError):
        # This fails because not all data was read in!
        ds.to_netcdf(nc_copy)
