"""Cloud-Optimized GeoTIFF (COG) outputs for SFINCS quadtree grids."""

import logging
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.rio.overview import get_maximum_overview_level

from hydromt_sfincs import utils, workflows

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

__all__ = ["create_index_cog", "create_topobathy_cog", "build_overviews"]

logger = logging.getLogger(f"hydromt.{__name__}")


def create_topobathy_cog(
    model: "SfincsModel",
    elevation_list: List[dict],
    filename: Union[str, Path],
    res: float,
    river_list: List[dict] = [],
    bounds: Optional[List[float]] = None,
    buffer_cells: int = 0,
    nrmax: int = 2000,  # blocksize
    z_minimum: float = -99999.0,
):
    """Create method for subgrid tables based on a list of
    elevation and Manning's roughness datasets.

    These datasets are used to derive relations between the water level
    and the volume in a cell to do the continuity update,
    and a representative water depth used to calculate momentum fluxes.

    This allows that one can compute on a coarser computational grid,
    while still accounting for the local topography and roughness.

    Parameters
    ----------
    elevation_list : List[dict]
        List of dictionaries with topobathy data.
        Each should minimally contain a data catalog source name, data file path,
        or xarray raster object ('elevation').
        Optional merge arguments include: 'zmin', 'zmax', 'mask', 'offset', 'reproj_method',
        and 'merge_method', see example below. For a complete overview of all merge options,
        see :py:func:`hydromt.workflows.merge_multi_dataarrays`

        ::

            [
                {'elevation': 'merit_hydro', 'zmin': 0.01},
                {'elevation': 'gebco', 'offset': 0, 'merge_method': 'first', reproj_method: 'bilinear'}
            ]

    river_list : List[dict], optional
        List of dictionaries with river datasets. Each dictionary should at least
        contain a river centerline data and optionally a river mask:

        * centerlines: filename (or Path) of river centerline with attributes
            rivwth (river width [m]; required if not river mask provided),
            rivdph or rivbed (river depth [m]; river bedlevel [m+REF]),
            manning (Manning's n [s/m^(1/3)]; optional)
        * mask (optional): filename (or Path) of river mask
        * point_zb (optional): filename (or Path) of river points with bed (z) values
        * river attributes (optional): "rivdph", "rivbed", "rivwth", "manning"
            to fill missing values
        * arguments to the river burn method (optional):
            segment_length [m] (default 500m) and riv_bank_q [0-1] (default 0.5)
            which used to estimate the river bank height in case river depth is provided.

        For more info see :py:func:`hydromt.workflows.bathymetry.burn_river_rect`

        ::

            [{'centerlines': 'river_lines', 'mask': 'river_mask', 'manning': 0.035}]

    buffer_cells : int, optional
        Number of cells between datasets to ensure smooth transition of bed levels,
        by default 0
    nrmax : int, optional
        Maximum number of cells per subgrid-block, by default 2000
        These blocks are used to prevent memory issues while working with large datasets
    z_minimum : float, optional
        Minimum depth in the subgrid tables, by default -99999.0
    """

    elevation_list = model._parse_datasets_elevation(elevation_list, res=res)

    if len(river_list) > 0:
        river_list = model._parse_river_list(river_list)

    # create a grid that covers the desired region at the desired resolution
    if bounds is None:
        bounds = model.bounds
    x0, y0, x1, y1 = bounds[0], bounds[1], bounds[2], bounds[3]

    # Round out to nearest dx
    x0 = x0 - (x0 % res)
    x1 = x1 + (res - x1 % res)
    y0 = y0 - (y0 % res)
    y1 = y1 + (res - y1 % res)

    xx = np.arange(x0, x1, res) + 0.5 * res
    yy = np.arange(y1, y0, -res) - 0.5 * res
    xx, yy = np.meshgrid(xx, yy)

    da_grid = utils.make_regular_grid(
        x0=x0,
        y0=y0,
        dx=res,
        dy=res,
        mmax=len(xx),
        nmax=len(yy),
        rotation=0.0,
        crs=model.crs,
    )

    grid_dim = da_grid.raster.shape
    x_dim, y_dim = da_grid.raster.x_dim, da_grid.raster.y_dim

    # create COGs for topobathy/manning
    profile = dict(
        driver="GTiff",
        width=da_grid.sizes[x_dim],
        height=da_grid.sizes[y_dim],
        count=1,
        dtype=np.float32,
        crs=da_grid.raster.crs,
        transform=da_grid.raster.transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=2,
        profile="COG",
        nodata=np.nan,
        BIGTIFF="YES",  # Add the BIGTIFF option here
    )

    # Create the output directory if it doesn't exist and create an empty COG file
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(filename, "w", **profile):
        pass

    # Determine the number of blocks in each direction
    n1, m1 = grid_dim
    nrcb = int(np.floor(nrmax))  # nr of regular cells in a block
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

    logger.info("Number of regular cells in a block : " + str(nrcb))
    logger.info("Number of blocks in n direction    : " + str(nrbn))
    logger.info("Number of blocks in m direction    : " + str(nrbm))

    logger.info(f"Grid size            : dx={res}, dy={res}")

    ## Loop through blocks
    ib = -1
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

            # Count
            ib += 1
            logger.info(
                f"block {ib + 1}/{nrbn * nrbm} -- "
                f"col {bm0}:{bm1-1} | row {bn0}:{bn1-1}"
            )

            # calculate transform and shape of block at cell and subgrid level
            # copy da_mask block to avoid accidently changing da_mask
            slice_block = {x_dim: slice(bm0, bm1), y_dim: slice(bn0, bn1)}
            da_grid_block = da_grid.isel(slice_block).load()

            # get subgrid bathymetry tile
            da_dep = workflows.merge_multi_dataarrays(
                da_list=elevation_list,
                da_like=da_grid_block,
                interp_method="linear",
                buffer_cells=buffer_cells,
            )

            # set minimum depth
            da_dep = np.maximum(da_dep, z_minimum)

            # NOTE: this is still open for discussion, but for now we interpolate
            # raise warning if NaN values in active cells
            if np.any(np.isnan(da_dep.values)) > 0:
                npx = int(np.sum(np.isnan(da_dep.values)))
                logger.warning(f"Interpolate elevation data at {npx} subgrid pixels")
            # always interpolate/extrapolate to avoid NaN values
            da_dep = da_dep.raster.interpolate_na(method="rio_idw", extrapolate=True)

            # burn rivers in bathymetry and manning
            if len(river_list) > 0:
                logger.debug("Burn rivers in bathymetry and manning data")
                for riv_kwargs in river_list:
                    da_dep, _ = workflows.bathymetry.burn_river_rect(
                        da_elv=da_dep, logger=logger, **riv_kwargs
                    )

            x_dim_dep, y_dim_dep = da_dep.raster.x_dim, da_dep.raster.y_dim
            window = Window(
                bm0,
                bn0,
                da_dep.sizes[x_dim_dep],
                da_dep.sizes[y_dim_dep],
            )
            # write the block to the output COG
            with rasterio.open(filename, "r+") as dep_tif:
                dep_tif.write(
                    da_dep.values,
                    window=window,
                    indexes=1,
                )

    # Create COG overviews for faster visualization
    build_overviews(
        fn=filename,
        resample_method="average",
    )


def create_index_cog(
    model: "SfincsModel",
    filename: Union[str, Path],
    filename_topobathy: Union[str, Path],
    nrmax: int = 2000,
    nodata: int = 2147483647,
):
    """Make a Cloud Optimzied Geotiff (COG) file with the correspodning indices of the SFINCS
    grid cells to the high-resolution DEM COG.

    Parameters
    ----------
    model : SfincsModel
        The SfincsModel instance containing the grid information.
    filename : Union[str, Path]
        The filename for the output COG file containing the indices. Note that this file only works
        for this SFINCS model and the topobathy file provided.
    filename_topobathy : Union[str, Path],
        The filename of the topobathy file from which to read the coordinates.
    nrmax : int, optional
        The maximum number of cells in a block, by default 2000.
    nodata : int, optional
        The nodata value to use in the output COG file, by default 2147483647
        (which is the maximum value for a 32-bit unsigned integer).

    See also:
    ----------
    hydromt_sfincs.workflows.cog.build_overviews : Function to build overviews for the COG file.
    hydromt_sfincs.workflows.downscaling.downscale_floodmap : Workflow to downscale flood maps
    """

    grid = model.grid_component

    # Read coordinates from topobathy file
    with rasterio.open(filename_topobathy) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        width = src.width
        height = src.height

        n1, m1 = src.shape

        nrcb = nrmax
        nrbn = int(np.ceil(n1 / nrcb))
        nrbm = int(np.ceil(m1 / nrcb))

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
            BIGTIFF="YES",
        )

    with rasterio.open(filename, "w", **profile):
        pass

    # Create transformer once
    proj = Transformer.from_crs(
        dem_crs,
        grid.crs,
        always_xy=True,
    )

    # Loop through blocks
    for ibm in range(nrbm):
        bm0 = ibm * nrcb
        bm1 = min(bm0 + nrcb, m1)

        if merge_last_col and ibm == nrbm - 1:
            bm1 += 1

        for ibn in range(nrbn):
            bn0 = ibn * nrcb
            bn1 = min(bn0 + nrcb, n1)

            if merge_last_row and ibn == nrbn - 1:
                bn1 += 1

            # Define a window to read a block of data
            window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

            # Calculate the coordinates of the center of each pixel in the block
            x_coords = dem_transform[2] + (np.arange(bm0, bm1) + 0.5) * dem_transform[0]
            y_coords = dem_transform[5] + (np.arange(bn0, bn1) + 0.5) * dem_transform[4]
            xx, yy = np.meshgrid(x_coords, y_coords)

            # Convert DEM coordinates to SFINCS grid CRS
            xx, yy = proj.transform(xx, yy)

            # Get SFINCS indices
            indices = grid.get_indices_at_points(xx, yy)
            indices = np.asarray(indices)
            indices[indices == -999] = nodata

            out = np.empty(
                (bn1 - bn0, bm1 - bm0),
                dtype=np.uint32,
            )
            out[:, :] = indices

            with rasterio.open(filename, "r+") as fm_tif:
                fm_tif.write(
                    out,
                    window=window,
                    indexes=1,
                )

    build_overviews(
        fn=filename,
        resample_method="nearest",
    )


def build_overviews(
    fn: Union[str, Path],
    resample_method: str = "average",
    overviews: Union[list, str] = "auto",
    logger=logger,
):
    """Build overviews for GeoTIFF file.

    Overviews are reduced resolution versions of your dataset that can speed up
    rendering when you don’t need full resolution. By precomputing the upsampled
    pixels, rendering can be significantly faster when zoomed out.

    Parameters
    ----------
    fn : str, Path
        Path to GeoTIFF file.
    method: str
        Resampling method, by default "average". Other option is "nearest".
    overviews: list of int, optional
        List of overview levels, by default "auto". When set to "auto" the
        overview levels are determined based on the size of the dataset.
    """

    # Endswith is not a method of Path so convert to str
    if isinstance(fn, Path):
        fn = str(fn)

    # check if fn is a geotiff file
    extensions = [".tif", ".tiff"]
    assert any(
        fn.endswith(ext) for ext in extensions
    ), f"File {fn} is not a GeoTIFF file."

    # open rasterio dataset
    with rasterio.open(fn, "r+") as src:
        # determine overviews when not provided
        if overviews == "auto":
            bs = src.profile.get("blockxsize", 256)
            max_level = get_maximum_overview_level(src.width, src.height, bs)
            overviews = [2**j for j in range(1, max_level + 1)]
        if not isinstance(overviews, list):
            raise ValueError("overviews should be a list of integers or 'auto'.")

        resampling = getattr(Resampling, resample_method, None)
        if resampling is None:
            raise ValueError(f"Resampling method unknown: {resample_method}")

        no = len(overviews)
        logger.info(f"Building {no} overviews with {resample_method}")

        # create new overviews, resampling with average method
        src.build_overviews(overviews, resampling)

        # update dataset tags
        src.update_tags(ns="rio_overview", resampling=resample_method)
