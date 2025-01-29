from os.path import join, dirname, abspath
import numpy as np
import os
from pyproj import CRS
import shutil

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

    # Open the copy with xu_open_dataset
    # This opens the file lazily
    ds = utils.xu_open_dataset(nc_copy)

    # Convert to dataset
    ds = ds.ugrid.to_dataset()

    # Try to write
    # NOTE this should fail because it still has lazy references to the file
    try:
        ds.to_netcdf(nc_copy)
    except PermissionError:
        pass

    # Now perform the check and lazy loading check
    utils.check_exists_and_lazy(ds, nc_copy)

    # Try to overwrite the file
    ds.to_netcdf(nc_copy)

    # Remove the copied file
    os.remove(nc_copy)
