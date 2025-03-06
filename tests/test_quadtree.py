from os.path import join, dirname, abspath
import numpy as np
from pyproj import CRS

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
