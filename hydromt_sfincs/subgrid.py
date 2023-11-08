"""
SubgridTableRegular class to create, read and write sfincs subgrid (sbg) files.
"""
import gc
import logging
import os

import numpy as np
import rasterio
import xarray as xr
from numba import njit
from rasterio.windows import Window

from . import utils
from . import workflows

logger = logging.getLogger(__name__)


class SubgridTableRegular:
    def __init__(self, version=0):
        # A regular subgrid table contains only for cells with msk>0
        self.version = version

    def load(self, file_name, mask):
        """Load subgrid table from file for a regular grid with given mask."""

        if isinstance(mask, xr.DataArray):
            mask = mask.values

        iok = np.where(np.transpose(mask) > 0)
        iok = (iok[1], iok[0])

        nmax = np.shape(mask)[0]
        mmax = np.shape(mask)[1]

        grid_dim = (nmax, mmax)

        file = open(file_name, "rb")

        # File version
        # self.version = np.fromfile(file, dtype=np.int32, count=1)[0]
        self.nr_cells = np.fromfile(file, dtype=np.int32, count=1)[0]
        self.nr_uv_points = np.fromfile(file, dtype=np.int32, count=1)[0]
        self.nbins = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Z points
        self.z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        # self.z_zmean = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_depth = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_hrep = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_navg = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_hrep = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_navg = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        self.z_zmin[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        self.z_zmax[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        # self.z_zmean[iok[0], iok[1]] = np.fromfile(
        #     file, dtype=np.float32, count=self.nr_cells
        # )
        self.z_volmax[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        for ibin in range(self.nbins):
            self.z_depth[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )

        self.u_zmin[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        self.u_zmax[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        _ = np.fromfile(file, dtype=np.float32, count=self.nr_cells)  # not used
        for ibin in range(self.nbins):
            self.u_hrep[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )
        for ibin in range(self.nbins):
            self.u_navg[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )

        self.v_zmin[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        self.v_zmax[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        _ = np.fromfile(file, dtype=np.float32, count=self.nr_cells)  # not used
        for ibin in range(self.nbins):
            self.v_hrep[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )
        for ibin in range(self.nbins):
            self.v_navg[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )

        file.close()

    def save(self, file_name, mask):
        """Save the subgrid data to a binary file."""
        if isinstance(mask, xr.DataArray):
            mask = mask.values

        iok = np.where(np.transpose(mask) > 0)
        iok = (iok[1], iok[0])

        nmax = np.shape(self.z_zmin)[0]
        mmax = np.shape(self.z_zmin)[1]

        # Add 1 because indices in SFINCS start with 1, not 0
        ind = np.ravel_multi_index(iok, (nmax, mmax), order="F") + 1

        file = open(file_name, "wb")
        # file.write(np.int32(self.version))  # version
        file.write(np.int32(np.size(ind)))  # Nr of active points
        file.write(np.int32(1))  # min
        file.write(np.int32(self.nbins))

        # Z
        v = self.z_zmin[iok]
        file.write(np.float32(v))
        v = self.z_zmax[iok]
        file.write(np.float32(v))
        v = self.z_volmax[iok]
        file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.z_depth[ibin, :, :])[iok]
            file.write(np.float32(v))

        # U
        v = self.u_zmin[iok]
        file.write(np.float32(v))
        v = self.u_zmax[iok]
        file.write(np.float32(v))
        dhdz = np.full(np.shape(v), 1.0)
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ibin in range(self.nbins):
            v = np.squeeze(self.u_hrep[ibin, :, :])[iok]
            file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.u_navg[ibin, :, :])[iok]
            file.write(np.float32(v))

        # V
        v = self.v_zmin[iok]
        file.write(np.float32(v))
        v = self.v_zmax[iok]
        file.write(np.float32(v))
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ibin in range(self.nbins):
            v = np.squeeze(self.v_hrep[ibin, :, :])[iok]
            file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.v_navg[ibin, :, :])[iok]
            file.write(np.float32(v))

        file.close()

    def build(
        self,
        da_mask: xr.DataArray,
        datasets_dep: list[dict],
        datasets_rgh: list[dict] = [],
        datasets_riv: list[dict] = [],
        nbins=10,
        nr_subgrid_pixels=20,
        nrmax=2000,
        max_gradient=5.0,
        z_minimum=-99999.0,
        manning_land: float = 0.04,
        manning_sea: float = 0.02,
        rgh_lev_land: float = 0.0,
        buffer_cells: int = 0,
        write_dep_tif: bool = False,
        write_man_tif: bool = False,
        highres_dir: str = None,
        logger=logger,
    ):
        """Create subgrid tables for regular grid based on a list of depth,
        Manning's rougnhess and river datasets.

        Parameters
        ----------
        da_mask : xr.DataArray
            Mask of the SFINCS domain, with 1,2,3 for active (and boundary) cells
            and 0 for inactive cells.
        datasets_dep : List[dict]
            List of dictionaries with topobathy data, each containing an xarray.DataSet
            and optional merge arguments e.g.:
            [
                {'da': <xr.Dataset>, 'zmin': 0.01},
                {'da': <xr.Dataset>, 'merge_method': 'first', reproj_method: 'bilinear'}
            ]
            For a complete overview of all merge options,
            see :py:function:~hydromt.workflows.merge_multi_dataarrays
        datsets_rgh : List[dict], optional
            List of dictionaries with Manning's n data, each containing an
            xarray.DataSet with manning values and optional merge arguments
        datasets_riv : List[dict], optional
            List of dictionaries with river datasets. Each dictionary should at least
            contain the following:
            * gdf_riv: line vector of river centerline with
              river depth ("rivdph") [m] OR bed level ("rivbed") [m+REF],
              river width ("rivwth"), and
              river manning ("manning") attributes [m]
            * gdf_riv_mask (optional): polygon vector of river mask. If provided
              "rivwth" in river is not used and can be omitted.
            * arguments for :py:function:~hydromt.workflows.bathymetry.burn_river_rect
            e.g.: [{'gdf_riv': <gpd.GeoDataFrame>, 'gdf_riv_mask': <gpd.GeoDataFrame>}]
        nbins : int, optional
            Number of bins in which hypsometry is subdivided, by default 10
        nr_subgrid_pixels : int, optional
            Number of subgrid pixels per computational cell, by default 20
        nrmax : int, optional
            Maximum number of cells per subgrid-block, by default 2000
            These blocks are used to prevent memory issues
        max_gradient : float, optional
            If slope in hypsometry exceeds this value, then smoothing is applied, to
            prevent numerical stability problems, by default 5.0
        z_minimum : float, optional
            Minimum depth in the subgrid tables, by default -99999.0
        manning_land, manning_sea : float, optional
            Constant manning roughness values for land and sea,
            by default 0.04 and 0.02 s.m-1/3
            Note that these values are only used when no Manning's n datasets are
            provided, or to fill the nodata values
        rgh_lev_land : float, optional
            Elevation level to distinguish land and sea roughness (when using
            manning_land and manning_sea), by default 0.0
        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels,
            by default 0
        write_dep_tif : bool, optional
            Create geotiff of the merged topobathy on the subgrid resolution,
            by default False
        write_man_tif : bool, optional
            Create geotiff of the merged roughness on the subgrid resolution,
            by default False
        highres_dir : str, optional
            Directory where high-resolution geotiffs for topobathy and manning
            are stored, by default None
        """

        if write_dep_tif or write_man_tif:
            assert highres_dir is not None, "highres_dir must be specified"

        refi = nr_subgrid_pixels
        self.nbins = nbins
        grid_dim = da_mask.raster.shape
        x_dim, y_dim = da_mask.raster.x_dim, da_mask.raster.y_dim

        # determine the output dimensions and transform to match da_mask grid
        # NOTE: this is only usef for writing the cloud optimized geotiffs
        output_width = da_mask.sizes[x_dim] * nr_subgrid_pixels
        output_height = da_mask.sizes[y_dim] * nr_subgrid_pixels
        output_transform = da_mask.raster.transform * da_mask.raster.transform.scale(
            1 / nr_subgrid_pixels
        )

        # create COGs for topobathy/manning
        profile = dict(
            driver="GTiff",
            width=output_width,
            height=output_height,
            count=1,
            dtype=np.float32,
            crs=da_mask.raster.crs,
            transform=output_transform,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            predictor=2,
            profile="COG",
            nodata=np.nan,
            BIGTIFF="YES",  # Add the BIGTIFF option here
        )
        if write_dep_tif:
            # create the CloudOptimizedGeotiff containing the merged topobathy data
            fn_dep_tif = os.path.join(highres_dir, "dep_subgrid.tif")
            with rasterio.open(fn_dep_tif, "w", **profile):
                pass

        if write_man_tif:
            # create the CloudOptimizedGeotiff creating the merged manning roughness
            fn_man_tif = os.path.join(highres_dir, "manning_subgrid.tif")
            with rasterio.open(fn_man_tif, "w", **profile):
                pass

        # Z points
        self.z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        # self.z_zmean = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_depth = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_hrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_navg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_hrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.v_navg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)

        dx, dy = da_mask.raster.res
        dxp = dx / refi  # size of subgrid pixel
        dyp = dy / refi  # size of subgrid pixel

        n1, m1 = grid_dim
        nrcb = int(np.floor(nrmax / refi))  # nr of regular cells in a block
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

        logger.info(f"Grid size of flux grid            : dx={dx}, dy={dy}")
        logger.info(f"Grid size of subgrid pixels       : dx={dxp}, dy={dyp}")

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
                da_mask_block = da_mask.isel(slice_block).load()
                check_block = np.all([s > 1 for s in da_mask_block.shape])
                assert check_block, f"unexpected block shape {da_mask_block.shape}"
                nactive = int(np.sum(da_mask_block > 0))
                if nactive == 0:  # not active cells in block
                    logger.debug("Skip block - No active cells")
                    continue
                transform = da_mask_block.raster.transform
                # add refi cells overlap in both dimensions for u and v in last row/col
                reproj_kwargs = dict(
                    dst_crs=da_mask.raster.crs,
                    dst_transform=transform * transform.scale(1 / refi),
                    dst_width=(da_mask_block.raster.width + 1) * refi,
                    dst_height=(da_mask_block.raster.height + 1) * refi,
                )
                da_mask_sbg = da_mask_block.raster.reproject(
                    method="nearest", **reproj_kwargs
                ).load()

                # get subgrid bathymetry tile
                da_dep = workflows.merge_multi_dataarrays(
                    da_list=datasets_dep,
                    da_like=da_mask_sbg,
                    interp_method="linear",
                    buffer_cells=buffer_cells,
                )

                # set minimum depth
                da_dep = np.maximum(da_dep, z_minimum)
                # TODO what to do with remaining cell with nan values
                # NOTE: this is still open for discussion, but for now we interpolate
                # raise warning if NaN values in active cells
                if np.any(np.isnan(da_dep.values[da_mask_sbg > 0])) > 0:
                    npx = int(np.sum(np.isnan(da_dep.values[da_mask_sbg > 0])))
                    logger.warning(
                        f"Interpolate elevation data at {npx} subgrid pixels"
                    )
                # always interpolate/extrapolate to avoid NaN values
                da_dep = da_dep.raster.interpolate_na(
                    method="rio_idw", extrapolate=True
                )

                # get subgrid manning roughness tile
                if len(datasets_rgh) > 0:
                    da_man = workflows.merge_multi_dataarrays(
                        da_list=datasets_rgh,
                        da_like=da_mask_sbg,
                        interp_method="linear",
                        buffer_cells=buffer_cells,
                    )
                    # raise warning if NaN values in active cells
                    if np.isnan(da_man.values[da_mask_sbg > 0]).any():
                        npx = int(np.sum(np.isnan(da_man.values[da_mask_sbg > 0])))
                        logger.warning(
                            f"Fill manning roughness data at {npx} subgrid pixels with default values"
                        )
                    # always fill based on land/sea elevation to avoid NaN values
                    da_man0 = xr.where(
                        da_dep >= rgh_lev_land, manning_land, manning_sea
                    )
                    da_man = da_man.where(~np.isnan(da_man), da_man0)
                else:
                    da_man = xr.where(da_dep >= rgh_lev_land, manning_land, manning_sea)
                    da_man.raster.set_nodata(np.nan)

                # burn rivers in bathymetry and manning
                if len(datasets_riv) > 0:
                    logger.debug("Burn rivers in bathymetry and manning data")
                    for riv_kwargs in datasets_riv:
                        da_dep, da_man = workflows.bathymetry.burn_river_rect(
                            da_elv=da_dep, da_man=da_man, logger=logger, **riv_kwargs
                        )

                # optional write tile to file
                # NOTE tiles have overlap! da_dep[:-refi,:-refi]
                window = Window(
                    bm0 * nr_subgrid_pixels,
                    bn0 * nr_subgrid_pixels,
                    da_dep[:-refi, :-refi].sizes[x_dim],
                    da_dep[:-refi, :-refi].sizes[y_dim],
                )
                if write_dep_tif:
                    # write the block to the output COG
                    with rasterio.open(fn_dep_tif, "r+") as dep_tif:
                        dep_tif.write(
                            da_dep.where(da_mask_sbg > 0)[:-refi, :-refi].values,
                            window=window,
                            indexes=1,
                        )
                if write_man_tif:
                    with rasterio.open(fn_man_tif, "r+") as man_tif:
                        man_tif.write(
                            da_man.where(da_mask_sbg > 0)[:-refi, :-refi].values,
                            window=window,
                            indexes=1,
                        )

                # check for NaN values for entire tile
                check_nans = np.all(np.isfinite(da_dep))
                assert check_nans, "NaN values in depth array"
                check_nans = np.all(np.isfinite(da_man))
                assert check_nans, "NaN values in manning roughness array"

                yg = da_dep.raster.ycoords.values
                if yg.ndim == 1:
                    yg = np.repeat(np.atleast_2d(yg), da_dep.raster.shape[0], axis=0)

                # Now compute subgrid properties
                logger.debug(f"Processing subgrid tables for {nactive} active cells..")
                sn, sm = slice(bn0, bn1), slice(bm0, bm1)
                (
                    self.z_zmin[sn, sm],
                    self.z_zmax[sn, sm],
                    self.z_volmax[sn, sm],
                    self.z_depth[:, sn, sm],
                    self.u_zmin[sn, sm],
                    self.u_zmax[sn, sm],
                    self.u_hrep[:, sn, sm],
                    self.u_navg[:, sn, sm],
                    self.v_zmin[sn, sm],
                    self.v_zmax[sn, sm],
                    self.v_hrep[:, sn, sm],
                    self.v_navg[:, sn, sm],
                ) = process_tile(
                    da_mask_block.values,
                    da_dep.values,
                    da_man.values,
                    dxp,
                    dyp,
                    refi,
                    nbins,
                    yg,
                    max_gradient,
                    da_mask.raster.crs.is_geographic,
                )

                del da_mask_block, da_dep, da_man
                gc.collect()

        # Create COG overviews
        if write_dep_tif:
            utils.build_overviews(
                fn=fn_dep_tif,
                resample_method="average",
                overviews="auto",
                logger=logger,
            )
        if write_man_tif:
            utils.build_overviews(
                fn=fn_man_tif,
                resample_method="average",
                overviews="auto",
                logger=logger,
            )

    def to_xarray(self, dims, coords):
        """Convert subgrid class to xarray dataset."""
        ds_sbg = xr.Dataset(coords={"bins": np.arange(self.nbins), **coords})
        ds_sbg.attrs.update({"_FillValue": np.nan})

        zlst2 = ["z_zmin", "z_zmax", "z_zmin", "z_volmax"]  # "z_zmean",
        uvlst2 = ["u_zmin", "u_zmax", "v_zmin", "v_zmax"]
        lst3 = ["z_depth", "u_hrep", "u_navg", "v_hrep", "v_navg"]
        # 2D arrays
        for name in zlst2 + uvlst2:
            if hasattr(self, name):
                ds_sbg[name] = xr.Variable(dims, getattr(self, name))
        # 3D arrays
        for name in lst3:
            if hasattr(self, name):
                ds_sbg[name] = xr.Variable(("bins", *dims), getattr(self, name))
        return ds_sbg

    def from_xarray(self, ds_sbg):
        """Convert xarray dataset to subgrid class."""
        for name in ds_sbg.data_vars:
            setattr(self, name, ds_sbg[name].values)


@njit
def process_tile(
    mask, zg, manning_grid, dxp, dyp, refi, nbins, yg, max_gradient, is_geographic=False
):
    """calculate subgrid properties for a single tile"""
    # Z points
    grid_dim = mask.shape
    z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    # z_zmean = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_depth = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)

    # U points
    u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_hrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_navg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)

    # V points
    v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_hrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
    v_navg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)

    # Loop through all active cells in this block
    for n in range(mask.shape[0]):  # row
        for m in range(mask.shape[1]):  # col
            if mask[n, m] < 1:
                # Not an active point
                continue

            nn = int(n * refi)
            mm = int(m * refi)

            # # Compute pixel size in metres
            if is_geographic:
                mean_lat = float(np.abs(np.mean(yg[nn : nn + refi, mm : mm + refi])))
                dxpm = float(dxp * 111111.0 * np.cos(np.pi * mean_lat / 180.0))
                dypm = float(dyp * 111111.0)
            else:
                dxpm = float(dxp)
                dypm = float(dyp)

            # First the volumes in the cells
            zgc = zg[nn : nn + refi, mm : mm + refi]
            zvmin = -20.0
            z, v, zmin, zmax = subgrid_v_table(
                zgc.flatten(), dxpm, dypm, nbins, zvmin, max_gradient
            )
            z_zmin[n, m] = zmin
            z_zmax[n, m] = zmax
            z_volmax[n, m] = v[-1]
            z_depth[:, n, m] = z[1:]

            # Now the U/V points
            # U
            nn = n * refi
            mm = m * refi + int(0.5 * refi)
            zgu = zg[nn : nn + refi, mm : mm + refi]
            zgu = np.transpose(zgu)
            manning = manning_grid[nn : nn + refi, mm : mm + refi]
            manning = np.transpose(manning)
            zmin, zmax, hrep, navg, zz = subgrid_q_table(
                zgu.flatten(), manning.flatten(), nbins
            )
            u_zmin[n, m] = zmin
            u_zmax[n, m] = zmax
            u_hrep[:, n, m] = hrep
            u_navg[:, n, m] = navg

            # V
            nn = n * refi + int(0.5 * refi)
            mm = m * refi
            zgu = zg[nn : nn + refi, mm : mm + refi]
            manning = manning_grid[nn : nn + refi, mm : mm + refi]
            zmin, zmax, hrep, navg, zz = subgrid_q_table(
                zgu.flatten(), manning.flatten(), nbins
            )
            v_zmin[n, m] = zmin
            v_zmax[n, m] = zmax
            v_hrep[:, n, m] = hrep
            v_navg[:, n, m] = navg

    return (
        z_zmin,
        z_zmax,
        # z_zmean,
        z_volmax,
        z_depth,
        u_zmin,
        u_zmax,
        u_hrep,
        u_navg,
        v_zmin,
        v_zmax,
        v_hrep,
        v_navg,
    )


@njit
def get_dzdh(z, V, a):
    # change in level per unit of volume (m/m)
    dz = np.diff(z)
    # change in volume (normalized to meters)
    dh = np.maximum(np.diff(V) / a, 0.001)
    return dz / dh


@njit
def isclose(a, b, rtol=1e-05, atol=1e-08):
    return abs(a - b) <= (atol + rtol * abs(b))


@njit
def subgrid_v_table(
    elevation: np.ndarray,
    dx: float,
    dy: float,
    nbins: int,
    zvolmin: float,
    max_gradient: float,
):
    """
    map vector of elevation values into a hypsometric volume - depth relationship
    for one grid cell

    Parameters
    ----------
    elevation: np.ndarray
        subgrid elevation values for one grid cell [m]
    dx: float
        x-directional cell size (typically not known at this level) [m]
    dy: float
        y-directional cell size (typically not known at this level) [m]
    nbins: int
        number of bins to use for the hypsometric curve
    zvolmin: float
        minimum elevation value to use for volume calculation (typically -20 m)
    max_gradient: float
        maximum gradient to use for volume calculation (typically 0.1)

    Return
    ------
    z, V: np.ndarray
        sorted elevation values, volume per elevation value
    zmin, zmax: float
        minimum, and maximum elevation values
    """

    # Cell area
    a = float(elevation.size * dx * dy)

    # Set minimum elevation to -20 (needed with single precision), and sort
    ele_sort = np.sort(np.maximum(elevation, zvolmin).flatten())

    # Make sure each consecutive point is larger than previous
    for j in range(1, ele_sort.size):
        if ele_sort[j] <= ele_sort[j - 1]:
            ele_sort[j] += 1.0e-6

    depth = ele_sort - ele_sort.min()

    volume = np.zeros_like(depth)
    volume[1:] = np.cumsum((np.diff(depth) * dx * dy) * np.arange(1, depth.size))

    # Resample volumes to discrete bins
    steps = np.arange(nbins + 1) / nbins
    V = steps * volume.max()
    dvol = volume.max() / nbins
    # scipy not supported in numba jit
    # z = interpolate.interp1d(volume, ele_sort)(V)
    z = np.interp(V, volume, ele_sort)
    dzdh = get_dzdh(z, V, a)
    n = 0
    while (
        dzdh.max() > max_gradient and not (isclose(dzdh.max(), max_gradient))
    ) and n < nbins:
        # reshape until gradient is satisfactory
        idx = np.where(dzdh == dzdh.max())[0]
        z[idx + 1] = z[idx] + max_gradient * (dvol / a)
        dzdh = get_dzdh(z, V, a)
        n += 1
    return z, V, elevation.min(), z.max()


@njit
def subgrid_q_table(elevation: np.ndarray, manning: np.ndarray, nbins: int):
    """
    map elevation values into a hypsometric hydraulic radius - depth relationship

    Parameters
    ----------
    elevation: np.ndarray
        subgrid elevation values for one grid cell [m]
    manning: np.ndarray
        subgrid manning roughness values for one grid cell [s m^(-1/3)]
    nbins: int
        number of bins to use for the hypsometric curve

    Returns
    -------
    zmin, zmax: float
        minimum and maximum elevation values used for hypsometric curve
    hrep, navg, zz: np.ndarray
        conveyance depth, average manning roughness, and elevation values
        for each bin
    """
    hrep = np.zeros(nbins, dtype=np.float32)
    navg = np.zeros(nbins, dtype=np.float32)
    zz = np.zeros(nbins, dtype=np.float32)

    n = int(elevation.size)  # Nr of pixels in grid cell
    n05 = int(n / 2)

    zmin_a = np.min(elevation[0:n05])
    zmax_a = np.max(elevation[0:n05])

    zmin_b = np.min(elevation[n05:])
    zmax_b = np.max(elevation[n05:])

    zmin = max(zmin_a, zmin_b)
    zmax = max(zmax_a, zmax_b)

    # Make sure zmax is a bit higher than zmin
    if zmax < zmin + 0.01:
        zmax += 0.01

    # Determine bin size
    dbin = (zmax - zmin) / nbins

    # Loop through bins
    for ibin in range(nbins):
        # Top of bin
        zbin = zmin + (ibin + 1) * dbin
        zz[ibin] = zbin

        ibelow = np.where(elevation <= zbin)  # index of pixels below bin level
        # water depth in each pixel
        h = np.maximum(zbin - np.maximum(elevation, zmin), 0.0)
        qi = h ** (5.0 / 3.0) / manning  # unit discharge in each pixel
        q = np.sum(qi) / n  # combined unit discharge for cell

        navg[ibin] = manning[ibelow].mean()  # mean manning's n
        hrep[ibin] = (q * navg[ibin]) ** (3.0 / 5.0)  # conveyance depth

    return zmin, zmax, hrep, navg, zz
