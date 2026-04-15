from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

from hydromt_sfincs.utils import build_overviews

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel


def make_index_cog(
    model: "SfincsModel",
    indices_fn: Union[str, Path],
    topobathy_fn: Union[str, Path],
    nrmax: int = 2000,
    nodata: int = 2147483647,
):
    """Make a Cloud Optimzied Geotiff (COG) file with the correspodning indices of the SFINCS
    grid cells to the high-resolution DEM COG.

    Parameters
    ----------
    model : SfincsModel
        The SfincsModel instance containing the grid information.
    indices_fn : Union[str, Path]
        The filename for the output COG file containing the indices. Note that this file only works
        for this SFINCS model and the topobathy file provided.
    topobathy_fn : Union[str, Path]
        The filename of the topobathy file from which to read the coordinates.
    nrmax : int, optional
        The maximum number of cells in a block, by default 2000.
    nodata : int, optional
        The nodata value to use in the output COG file, by default 2147483647
        (which is the maximum value for a 32-bit unsigned integer).

    See also:
    ----------
    hydromt_sfincs.utils.build_overviews : Function to build overviews for the COG file.
    hydromt_sfincs.utils.downscale_floodmap : Workflow to downscale flood maps
    """

    # Read coordinates from topobathy file
    with rasterio.open(topobathy_fn) as src:
        # Get the CRS of the grid
        dem_crs = src.crs
        # Get the transform of the grid
        dem_transform = src.transform
        # Get the width and height of the grid
        width = src.width
        height = src.height

        n1, m1 = src.shape
        nrcb = nrmax  # nr of cells in a block
        nrbn = int(np.ceil(n1 / nrcb))  # nr of blocks in n direction
        nrbm = int(np.ceil(m1 / nrcb))  # nr of blocks in m direction

        # avoid blocks with width or height of 1
        merge_last_col = False
        merge_last_row = False
        if m1 % nrcb == 1:
            nrbm -= 1
            merge_last_col = True
        if n1 % nrcb == 1:
            nrbn -= 1
            merge_last_row = True

        profile = dict(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.uint32,
            crs=dem_crs,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            transform=dem_transform,
            nodata=nodata,
            predictor=2,
            profile="COG",
            BIGTIFF="YES",  # Add the BIGTIFF option here
        )

    with rasterio.open(indices_fn, "w", **profile):
        pass

    # Get the computational grid component of the model
    grid_comp = model.quadtree_grid if model.grid_type == "quadtree" else model.grid

    ## Loop through blocks
    for ii in range(nrbm):
        bm0 = ii * nrcb  # Index of first m in block
        bm1 = min(bm0 + nrcb, m1)  # last m in block
        if merge_last_col and ii == (nrbm - 1):
            bm1 += 1

        for jj in range(nrbn):
            bn0 = jj * nrcb  # Index of first n in block
            bn1 = min(bn0 + nrcb, n1)  # last n in block

            if merge_last_row and jj == (nrbn - 1):
                bn1 += 1

            # Define a window to read a block of data
            window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

            # Calculate the coordinates of the center of each pixel in the block
            x_coords = dem_transform[2] + (np.arange(bm0, bm1) + 0.5) * dem_transform[0]
            y_coords = dem_transform[5] + (np.arange(bn0, bn1) + 0.5) * dem_transform[4]
            xx, yy = np.meshgrid(x_coords, y_coords)

            # Transform the coordinates to the model's CRS and get the corresponding indices
            proj = Transformer.from_crs(dem_crs, model.crs, always_xy=True)
            xx, yy = proj.transform(xx, yy)
            indices = grid_comp.get_indices_at_points(xx, yy)
            indices[np.where(indices == -999)] = nodata

            # Fill the array with indices
            ii = np.empty((bn1 - bn0, bm1 - bm0), dtype=np.uint32)
            ii[:, :] = indices

            with rasterio.open(indices_fn, "r+") as fm_tif:
                fm_tif.write(
                    ii,
                    window=window,
                    indexes=1,
                )
        # add overviews
        build_overviews(fn=indices_fn, resample_method="nearest")
