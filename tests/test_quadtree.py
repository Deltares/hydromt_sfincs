from os.path import join, dirname, abspath
import numpy as np
from pyproj import CRS

from hydromt_sfincs import SfincsModel
from hydromt import Model
from hydromt_sfincs.quadtree import QuadtreeGrid

TESTDATADIR = join(dirname(abspath(__file__)), "data")


def test_quadtree_io(tmp_path):
    # Start with model to make sure the root is set
    model0 = Model(root=join(TESTDATADIR, "sfincs_test_quadtree"), mode="r")

    # Initialize a QuadtreeGrid object
    qtr = QuadtreeGrid(model0)
    # Read a quadtree netcdf file
    qtr.read()
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
    fn = join(tmp_path, "sfincs.nc")
    qtr.write(fn)

    model1 = Model(root=tmp_path, mode="r")
    # read the new file and check the msk variable
    qtr2 = QuadtreeGrid(model1)
    qtr2.read(fn)
    # assert the crs is the same
    assert qtr2.crs == qtr.crs
    # assert the msk variable is the same
    assert np.sum(qtr2.data["msk"].values) == 4298
    # assert the dep variable is the same
    assert np.sum(qtr.data["dep"].values) == np.sum(qtr2.data["dep"].values)
