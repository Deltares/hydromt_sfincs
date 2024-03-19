import numpy as np


def test_ind(reggrid, mask):
    ind = reggrid.ind(mask)
    assert ind[0] == 254
    assert ind[-1] == 2939
    assert ind.size == np.sum(mask > 0)
