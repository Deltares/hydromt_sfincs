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

    def read(self, file_name):
        # Read from netcdf file with xarray
        self.ds = xr.open_dataset(file_name)
        self.ds.close() # Should this be closed ?
        # Should we also make self.z_zmin, self.z_zmax, etc. ?

    def write(self, file_name):
        # Write XArray dataset to netcdf file

        if isinstance(mask, xr.DataArray):
            mask = mask.values

        # Array iok where mask > 0 
        iok = np.where(np.transpose(mask) > 0)
        iok = (iok[1], iok[0])

        # Make new xarray dataset
        ds = xr.Dataset()
        ds.attrs.update({"_FillValue": np.nan})
        ds["z_zmin"] = xr.DataArray(self.z_zmin[iok], dims=("cells"))
        ds["z_zmax"] = xr.DataArray(self.z_zmax[iok], dims=("cells"))
        ds["z_volmax"] = xr.DataArray(self.z_volmax[iok], dims=("cells"))
        ds["z_level"] = xr.DataArray(self.z_level[:, iok], dims=("bins", "cells"))
        ds["u_zmin"] = xr.DataArray(self.u_zmin[iok], dims=("cells"))
        ds["u_zmax"] = xr.DataArray(self.u_zmax[iok], dims=("cells"))
        ds["u_havg"] = xr.DataArray(self.u_havg[:, iok], dims=("bins", "cells"))
        ds["u_nrep"] = xr.DataArray(self.u_nrep[:, iok], dims=("bins", "cells"))
        ds["u_pwet"] = xr.DataArray(self.u_pwet[:, iok], dims=("bins", "cells"))
        ds["u_ffit"] = xr.DataArray(self.u_ffit[iok], dims=("cells"))
        ds["u_navg"] = xr.DataArray(self.u_navg[iok], dims=("cells"))
        ds["v_zmin"] = xr.DataArray(self.v_zmin[iok], dims=("cells"))
        ds["v_zmax"] = xr.DataArray(self.v_zmax[iok], dims=("cells"))
        ds["v_havg"] = xr.DataArray(self.v_havg[:, iok], dims=("bins", "cells"))
        ds["v_nrep"] = xr.DataArray(self.v_nrep[:, iok], dims=("bins", "cells"))
        ds["v_pwet"] = xr.DataArray(self.v_pwet[:, iok], dims=("bins", "cells"))
        ds["v_ffit"] = xr.DataArray(self.v_ffit[iok], dims=("cells"))
        ds["v_navg"] = xr.DataArray(self.v_navg[iok], dims=("cells"))

        # Need to swap the first and second dimensions to match the FORTRAN convention in SFINCS
        ds["z_level"] = ds["z_level"].swap_dims({"bins": "cells"})
        ds["u_havg"]  = ds["u_havg"].swap_dims({"bins": "cells"})
        ds["u_nrep"]  = ds["u_nrep"].swap_dims({"bins": "cells"})
        ds["u_pwet"]  = ds["u_pwet"].swap_dims({"bins": "cells"})
        ds["v_havg"]  = ds["v_havg"].swap_dims({"bins": "cells"})
        ds["v_nrep"]  = ds["v_nrep"].swap_dims({"bins": "cells"})
        ds["v_pwet"]  = ds["v_pwet"].swap_dims({"bins": "cells"})

        # Write to netcdf file
        self.ds.to_netcdf(file_name)

    # Following should not me use anymore !
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
        self.z_level = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_havg = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_nrep = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_pwet = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_nrep = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_ffit = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_havg = np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_nrep= np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_pwet= np.full(
            (self.nbins, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_nrep = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_ffit = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)

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
            self.z_level[ibin, iok[0], iok[1]] = np.fromfile(
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
            self.u_havg[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )
        for ibin in range(self.nbins):
            self.u_nrep[ibin, iok[0], iok[1]] = np.fromfile(
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
            self.v_havg[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )
        for ibin in range(self.nbins):
            self.v_nrep[ibin, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )

        file.close()

    # Following should not me use anymore !
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
            v = np.squeeze(self.z_level[ibin, :, :])[iok]
            file.write(np.float32(v))

        # U
        v = self.u_zmin[iok]
        file.write(np.float32(v))
        v = self.u_zmax[iok]
        file.write(np.float32(v))
        dhdz = np.full(np.shape(v), 1.0)
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ibin in range(self.nbins):
            v = np.squeeze(self.u_havg[ibin, :, :])[iok]
            file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.u_nrep[ibin, :, :])[iok]
            file.write(np.float32(v))

        # V
        v = self.v_zmin[iok]
        file.write(np.float32(v))
        v = self.v_zmax[iok]
        file.write(np.float32(v))
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ibin in range(self.nbins):
            v = np.squeeze(self.v_havg[ibin, :, :])[iok]
            file.write(np.float32(v))
        for ibin in range(self.nbins):
            v = np.squeeze(self.v_nrep[ibin, :, :])[iok]
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
        huthresh: float = 0.01,
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
        huthresh : float, optional
            Threshold depth in SFINCS model, by default 0.01 m
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
        # NOTE: this is only used for writing the cloud optimized geotiffs
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
        self.z_level = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_havg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_nrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_pwet = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_ffit = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_navg = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_havg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.v_nrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.v_pwet = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.v_ffit = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)
        self.v_navg = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)

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
                    self.z_level[:, sn, sm],
                    self.u_zmin[sn, sm],
                    self.u_zmax[sn, sm],
                    self.u_havg[:, sn, sm],
                    self.u_nrep[:, sn, sm],
                    self.u_pwet[:, sn, sm],
                    self.u_ffit[sn, sm],
                    self.u_navg[sn, sm],
                    self.u_zmax[sn, sm],
                    self.v_zmin[sn, sm],
                    self.v_zmax[sn, sm],
                    self.v_havg[:, sn, sm],
                    self.v_nrep[:, sn, sm],
                    self.v_pwet[:, sn, sm],
                    self.v_ffit[sn, sm],
                    self.v_navg[sn, sm],
                ) = process_tile_regular(
                    da_mask_block.values,
                    da_dep.values,
                    da_man.values,
                    dxp,
                    dyp,
                    refi,
                    nbins,
                    yg,
                    max_gradient,
                    huthresh,
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
        uvlst2 = ["u_zmin", "u_zmax", "u_ffit", "u_navg", "v_zmin", "v_zmax", "v_ffit", "v_navg"]
        lst3 = ["z_depth", "u_havg", "u_nrep", "u_pwet", "v_avg", "v_nrep", "v_pwet"]
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


class SubgridTableQuadtree:
    # This code is still slow as it does not use numba

    def __init__(self, version=0):
        # A quadtree subgrid table contains data for EACH cell, u and v point in the quadtree mesh,
        # regardless of the mask value!
        self.version = version

    def read(self, file_name):

        if not os.path.isfile(file_name):
            print("File " + file_name + " does not exist!")
            return

        # Read from netcdf file with xarray
        self.ds = xr.open_dataset(file_name)
        # Swap the first and second dimensions to convert from FORTRAN convention in SFINCS to Python
        self.ds["z_level"] = self.ds["z_level"].swap_dims({"bins": "cells"})
        self.ds["uv_havg"] = self.ds["uv_havg"].swap_dims({"bins": "uv_points"})
        self.ds["uv_nrep"] = self.ds["uv_nrep"].swap_dims({"bins": "uv_points"})
        self.ds["uv_pwet"] = self.ds["uv_pwet"].swap_dims({"bins": "uv_points"})
        self.ds.close() # Should this be closed ?

    def write(self, file_name):
        # Write XArray dataset to netcdf file
        # Need to switch the first and second dimensions to match the FORTRAN convention in SFINCS
        self.ds["z_level"] = self.ds["z_level"].swap_dims({"bins": "cells"})
        self.ds["uv_havg"] = self.ds["uv_havg"].swap_dims({"bins": "uv_points"})
        self.ds["uv_nrep"] = self.ds["uv_nrep"].swap_dims({"bins": "uv_points"})
        self.ds["uv_pwet"] = self.ds["uv_pwet"].swap_dims({"bins": "uv_points"})
        self.ds.to_netcdf(file_name)

    def build(
        self,
        ds_mesh: xr.Dataset,
        datasets_dep: list[dict],
        datasets_rgh: list[dict] = [],
        datasets_riv: list[dict] = [],
        nbins=10,
        nr_subgrid_pixels=20,
        nrmax=2000,
        max_gradient=5.0,
        z_minimum=-99999.0,
        z_multiply=1.0,
        manning_land: float = 0.04,
        manning_sea: float = 0.02,
        rgh_lev_land: float = 0.0,
        huthresh: float = 0.01,
        buffer_cells: int = 0,
        write_dep_tif: bool = False,
        write_man_tif: bool = False,
        highres_dir: str = None,
        logger=logger,
        progress_bar=None,
    ):
        """Create subgrid tables for regular grid based on a list of depth,
        Manning's rougnhess and river datasets.

        Parameters
        ----------
        ds_mesh : xr.Dataset
            Quadtree mesh of the SFINCS domain.
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
        huthresh : float, optional
            Threshold depth in SFINCS model, by default 0.01 m
        z_multiply : float, optional
            Multiplication factor for bed levels, by default 1.0
        manning_land, manning_sea : float, optional
            Constant manning roughness values for land and sea,
            by default 0.04 and 0.02 s.m-1/3
            Note that these values are only used when no Manning's n datasets are
            provided, or to fill the nodata values
        rgh_lev_land : float, optional
            Elevation level to distinguish land and sea roughness (when using
            manning_land and manning_sea), by default 0.0 m
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
        progress_bar : Object, optional
            Guitares progress bar object
        """

        # Dimensions etc
        refi   = nr_subgrid_pixels
        # nr_cells is length of dimension "cells" in ds_mesh. CHECK IF THIS IS CORRECT
        nr_cells = ds_mesh.dims["cells"]
        is_geographic = ds_mesh.attrs["is_geographic"] # TO DO crs
        nlevs  = ds_mesh.attrs["nr_levels"]
        cosrot = np.cos(ds_mesh.attrs["rotation"]*np.pi/180)
        sinrot = np.sin(ds_mesh.attrs["rotation"]*np.pi/180)
        nrmax  = 2000

        # Grid neighbors
        level = ds_mesh["level"].values[:]
        n   = ds_mesh["n"].values[:]
        m   = ds_mesh["m"].values[:]
        nu  = ds_mesh["nu"].values[:]
        nu1 = ds_mesh["nu1"].values[:]
        nu2 = ds_mesh["nu2"].values[:]
        mu  = ds_mesh["mu"].values[:]
        mu1 = ds_mesh["mu1"].values[:]
        mu2 = ds_mesh["mu2"].values[:]

        # U/V points 
        # Need to count the number of uv points in order allocate arrays (probably better to store this in the grid)
        # if self.model.grid_type == "quadtree":   
        # For quadtree grids, all points are stored
        index_nu1 = np.zeros(nr_cells, dtype=int)
        index_nu2 = np.zeros(nr_cells, dtype=int)
        index_mu1 = np.zeros(nr_cells, dtype=int)
        index_mu2 = np.zeros(nr_cells, dtype=int)        
        index_nm  = np.zeros(nr_cells, dtype=int)
        npuv = 0
        for ip in range(nr_cells):
            index_nm[ip] = ip
            if mu1[ip]>=0:
                index_mu1[ip] = npuv
                npuv += 1
            if mu2[ip]>=0:
                index_mu2[ip] = npuv
                npuv += 1
            if nu1[ip]>=0:
                index_nu1[ip] = npuv
                npuv += 1
            if nu2[ip]>=0:
                index_nu2[ip] = npuv
                npuv += 1

        # Create xarray dataset with empty arrays
        self.ds = xr.Dataset()
        self.ds.attrs["version"] = self.version
        self.ds["z_zmin"] = xr.DataArray(np.zeros(nr_cells), dims=["cells"])
        self.ds["z_zmax"] = xr.DataArray(np.zeros(nr_cells), dims=["cells"])
        self.ds["z_volmax"] = xr.DataArray(np.zeros(nr_cells), dims=["cells"])
        self.ds["z_level"] = xr.DataArray(np.zeros((nr_cells, nbins)), dims=["bins", "cells"])
        self.ds["uv_zmin"] = xr.DataArray(np.zeros(npuv), dims=["uv_points"])
        self.ds["uv_zmax"] = xr.DataArray(np.zeros(npuv), dims=["uv_points"])
        self.ds["uv_havg"] = xr.DataArray(np.zeros((nbins, npuv)), dims=["bins", "uv_points"])
        self.ds["uv_nrep"] = xr.DataArray(np.zeros((nbins, npuv)), dims=["bins", "uv_points"])
        self.ds["uv_pwet"] = xr.DataArray(np.zeros((nbins, npuv)), dims=["bins", "uv_points"])
        self.ds["uv_ffit"] = xr.DataArray(np.zeros(npuv), dims=["uv_points"])
        self.ds["uv_navg"] = xr.DataArray(np.zeros(npuv), dims=["uv_points"])
        
        # Determine first indices and number of cells per refinement level
        ifirst = np.zeros(nlevs, dtype=int)
        ilast  = np.zeros(nlevs, dtype=int)
        nr_cells_per_level = np.zeros(nlevs, dtype=int)
        ireflast = -1
        for ic in range(nr_cells):
            if level[ic]>ireflast:
                ifirst[level[ic]] = ic
                ireflast = level[ic]
        for ilev in range(nlevs - 1):
            ilast[ilev] = ifirst[ilev + 1] - 1
        ilast[nlevs - 1] = nr_cells - 1
        for ilev in range(nlevs):
            nr_cells_per_level[ilev] = ilast[ilev] - ifirst[ilev] + 1 

        # Loop through all levels
        for ilev in range(nlevs):

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Processing level " + str(ilev + 1) + " of " + str(nlevs) + " ...")
            
            # Make blocks off cells in this level only
            cell_indices_in_level = np.arange(ifirst[ilev], ilast[ilev] + 1, dtype=int)
            nr_cells_in_level = np.size(cell_indices_in_level)
            
            if nr_cells_in_level == 0:
                continue

            n0 = np.min(n[ifirst[ilev]:ilast[ilev] + 1])
            n1 = np.max(n[ifirst[ilev]:ilast[ilev] + 1]) # + 1 # add extra cell to compute u and v in the last row/column
            m0 = np.min(m[ifirst[ilev]:ilast[ilev] + 1])
            m1 = np.max(m[ifirst[ilev]:ilast[ilev] + 1]) # + 1 # add extra cell to compute u and v in the last row/column
            
            dx   = ds_mesh.attrs["dx"]/2**ilev      # cell size
            dy   = ds_mesh.attrs["dy"]/2**ilev      # cell size
            dxp  = dx/refi              # size of subgrid pixel
            dyp  = dy/refi              # size of subgrid pixel
            
            nrcb = int(np.floor(nrmax/refi))         # nr of regular cells in a block            
            nrbn = int(np.ceil((n1 - n0 + 1)/nrcb))  # nr of blocks in n direction
            nrbm = int(np.ceil((m1 - m0 + 1)/nrcb))  # nr of blocks in m direction

            print("Number of regular cells in a block : " + str(nrcb))
            print("Number of blocks in n direction    : " + str(nrbn))
            print("Number of blocks in m direction    : " + str(nrbm))
            
            print("Grid size of flux grid             : dx= " + str(dx) + ", dy= " + str(dy))
            print("Grid size of subgrid pixels        : dx= " + str(dxp) + ", dy= " + str(dyp))

            ib = -1
            ibt = 1

            if progress_bar:
                progress_bar.set_text("               Generating Sub-grid Tables (level " + str(ilev) + ") ...                ")
                progress_bar.set_maximum(nrbm * nrbn)

            ## Loop through blocks
            for ii in range(nrbm):
                for jj in range(nrbn):
                    
                    # Count
                    ib += 1
                    
                    bn0 = n0  + jj*nrcb               # Index of first n in block
                    bn1 = min(bn0 + nrcb - 1, n1) + 1 # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
                    bm0 = m0  + ii*nrcb               # Index of first m in block
                    bm1 = min(bm0 + nrcb - 1, m1) + 1 # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

                    print("--------------------------------------------------------------")
                    print("Processing block " + str(ib + 1) + " of " + str(nrbn*nrbm) + " ...")

                    # Now build the pixel matrix
                    x00 = 0.5*dxp + bm0*refi*dyp
                    x01 = x00 + (bm1 - bm0 + 1)*refi*dxp
                    y00 = 0.5*dyp + bn0*refi*dyp
                    y01 = y00 + (bn1 - bn0 + 1)*refi*dyp
                    
                    x0 = np.arange(x00, x01, dxp)
                    y0 = np.arange(y00, y01, dyp)
                    xg0, yg0 = np.meshgrid(x0, y0)
                    # Rotate and translate
                    xg = ds_mesh.attrs["x0"] + cosrot*xg0 - sinrot*yg0
                    yg = ds_mesh.attrs["y0"] + sinrot*xg0 + cosrot*yg0

                    # Clear variables
                    del x0, y0, xg0, yg0
                    
                    # # Create xarray data array with xg and yg
                    # da_xg = xr.DataArray(xg, dims=["y", "x"])
                    # da_yg = xr.DataArray(yg, dims=["y", "x"])
                    # Create xarray data array with depths and coordinate da_xg and da_yg
                    da_sbg = xr.DataArray(np.zeros((bn1 - bn0 + 1)*refi, (bm1 - bm0 + 1)*refi), dims=["n", "m"], coords={"x": xg, "y": yg})
                    # Make sure da_sbg has the correct CRS
                    da_sbg.raster.set_crs(ds_mesh.attrs["crs"])             

                    # get subgrid bathymetry tile
                    da_dep = workflows.merge_multi_dataarrays(
                        da_list=datasets_dep,
                        da_like=da_sbg,
                        interp_method="linear",
                        buffer_cells=buffer_cells,
                    )

                    # set minimum depth
                    da_dep = np.maximum(da_dep, z_minimum)
                    # TODO what to do with remaining cell with nan values
                    # NOTE: this is still open for discussion, but for now we interpolate
                    # raise warning if NaN values in active cells
                    npx = int(np.sum(np.isnan(da_dep.values[:])))
                    logger.warning(
                        f"Interpolate elevation data at {npx} subgrid pixels"
                    )
                    # always interpolate/extrapolate to avoid NaN values
                    da_dep = da_dep.raster.interpolate_na(
                        method="rio_idw", extrapolate=True
                    )

                    # Multiply bed level da_dep with factor z_multiply (this should not normally be done)
                    # da_dep.values[:] = da_dep.values[:]*z_multiply

                    # get subgrid manning roughness tile
                    if len(datasets_rgh) > 0:
                        da_man = workflows.merge_multi_dataarrays(
                            da_list=datasets_rgh,
                            da_like=da_sbg,
                            interp_method="linear",
                            buffer_cells=buffer_cells,
                        )
                        # raise warning if NaN values in active cells
                        npx = int(np.sum(np.isnan(da_man.values[:])))
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
                                        
                    # Now compute subgrid properties

                    # Make arrays with indices of cells (and uv points) in this block

                    # First we loop through all the possible cells in this block
                    index_cells_in_block = np.zeros(nrcb*nrcb, dtype=int)
                    # index_uv_points_in_block = np.zeros(4*nrcb*nrcb, dtype=int)
                    # Loop through all cells in this level
                    nr_cells_in_block = 0
                    # nr_uv_points_in_block = 0
                    # Check if cells fall within this block
                    for ic in range(nr_cells_in_level):
                        indx = cell_indices_in_level[ic] # index of the whole quadtree
                        if n[indx]>=bn0 and n[indx]<bn1 and m[indx]>=bm0 and m[indx]<bm1:
                            # Cell falls inside block
                            index_cells_in_block[nr_cells_in_block] = indx
                            nr_cells_in_block += 1
                            # TO DO indices for uv points
                            # and nn, mm, dir and type -1,0,1       

                    index_cells_in_block = index_cells_in_block[0:nr_cells_in_block]

                    print("Number of active cells in block    : " + str(nr_cells_in_block))

                    # Better to first loop through cells and then through uv points ?

                    # Loop through all active cells in this block
                    for ic in range(nr_cells_in_block):

                        # Process cell

                        # Index in full quadtree mesh
                        indx = index_cells_in_block[ic]

                        # Get indices of pixels in block
                        nn  = (n[indx] - bn0) * refi # First pixel n index in cell
                        mm  = (m[indx] - bm0) * refi # First pixel m index in cell

                        # Matrix with pixels in cell
                        zv = da_dep.values[nn : nn + refi, mm : mm + refi].flatten()

                        # Compute pixel size in metres
                        if is_geographic:
                            # Compute latitude of cell (Find a better and faster way to do this. Probably just use the cell center).
                            mean_lat  = ds_mesh.face_coordinates["y"][indx]    # TODO: Check if this is correct
                            # ygc = yg[nn : nn + refi, mm : mm + refi]
                            # mean_lat =np.abs(np.mean(ygc))
                            dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
                            dypm = dyp*111111.0
                        else:
                            dxpm = dxp
                            dypm = dyp
                        
                        zvmin = -20.0
                        z, v, zmin, zmax = subgrid_v_table(zv, dxpm, dypm, nbins, zvmin, max_gradient)

                        self.ds["z_zmin"].values[ic] = zmin
                        self.ds["z_zmax"].values[ic] = zmax
                        self.ds["z_volmax"].values[ic]  = v[-1]
                        self.ds["z_level"].values[:,ic] = z
                        
                        # Now the U/V points
                        # First right
                        if mu[indx] <= 0:
                            if mu1[indx] >= 0:
                                nn  = (n[indx] - bn0)*refi
                                mm  = (m[indx] - bm0)*refi + int(0.5*refi)
                                zv  = np.transpose(da_dep.values[nn : nn + refi, mm : mm + refi]).flatten()
                                mv  = np.transpose(da_man.values[nn : nn + refi, mm : mm + refi]).flatten()
                                iuv = index_mu1[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, mv, nbins, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][:, iuv] = havg
                                    self.ds["uv_nrep"][:, iuv] = nrep
                                    self.ds["uv_pwet"][:, iuv] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg
                        else:        
                            if mu1[indx] >= 0:
                                nn = (n[indx] - bn0)*refi
                                mm = (m[indx] - bm0)*refi + int(3*refi/4)
                                zgu = da_dep.values[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zgu = np.transpose(zgu)
                                zv  = zgu.flatten()
                                manning = da_man.values[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = np.transpose(manning)
                                manning = manning.flatten()
                                iuv = index_mu1[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][:, iuv] = havg
                                    self.ds["uv_nrep"][:, iuv] = nrep
                                    self.ds["uv_pwet"][:, iuv] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg
                            if mu2[indx] >= 0:
                                nn = (n[indx] - bn0)*refi + int(refi/2)
                                mm = (m[indx] - bm0)*refi + int(3*refi/4)
                                zgu = da_dep.values[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zgu = np.transpose(zgu)
                                zv  = zgu.flatten()
                                manning = da_man.values[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = np.transpose(manning)
                                manning = manning.flatten()
                                iuv = index_mu2[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][:, iuv] = havg
                                    self.ds["uv_nrep"][:, iuv] = nrep
                                    self.ds["uv_pwet"][:, iuv] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg

                        # Now above
                        if nu[indx] <= 0:
                            if nu1[indx] >= 0:
                                nn = (n[indx] - bn0)*refi + int(0.5*refi)
                                mm = (m[indx] - bm0)*refi
                                zgu = da_dep.values[nn : nn + refi, mm : mm + refi]
                                zv  = zgu.flatten()
                                manning = da_man.values[nn : nn + refi, mm : mm + refi]
                                manning = manning.flatten()
                                iuv = index_nu1[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][:, iuv] = havg
                                    self.ds["uv_nrep"][:, iuv] = nrep
                                    self.ds["uv_pwet"][:, iuv] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg
                        else:        
                            if nu1[indx] >= 0:
                                nn = (n[indx] - bn0)*refi + int(3*refi/4)
                                mm = (m[indx] - bm0)*refi
                                zgu = da_dep.values[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zv  = zgu.flatten()
                                manning = da_man.values[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = manning.flatten()
                                iuv = index_nu1[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][:, iuv] = havg
                                    self.ds["uv_nrep"][:, iuv] = nrep
                                    self.ds["uv_pwet"][:, iuv] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg
                            if nu2[indx] >= 0:
                                nn = (n[indx] - bn0)*refi + int(3*refi/4)
                                mm = (m[indx] - bm0)*refi + int(refi/2)
                                zgu = da_dep.values[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                zv  = zgu.flatten()
                                manning = da_man.values[nn : nn + int(refi/2), mm : mm + int(refi/2)]
                                manning = manning.flatten()
                                iuv = index_nu2[indx]
                                if iuv>=0:
                                    zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
                                    self.ds["uv_zmin"][iuv]   = zmin
                                    self.ds["uv_zmax"][iuv]   = zmax
                                    self.ds["uv_havg"][:, iuv] = havg
                                    self.ds["uv_nrep"][:, iuv] = nrep
                                    self.ds["uv_pwet"][:, iuv] = pwet
                                    self.ds["uv_ffit"][iuv]   = ffit
                                    self.ds["uv_navg"][iuv]   = navg

                    if progress_bar:
                        print(ibt)
                        progress_bar.set_value(ibt)
                        if progress_bar.was_canceled():
                            return
                        ibt += 1

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


@njit
def process_tile_regular(
    mask, zg, manning_grid, dxp, dyp, refi, nbins, yg, max_gradient, huthresh, is_geographic=False
):
    """calculate subgrid properties for a single tile"""
    # Z points
    grid_dim = mask.shape
    z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_level = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)

    # U points
    u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_havg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_nrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_pwet = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_ffit = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_navg = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)

    # V points
    v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_havg = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
    v_nrep = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
    v_pwet = np.full((nbins, *grid_dim), fill_value=np.nan, dtype=np.float32)
    v_ffit = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_navg = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)

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
            z_level[:, n, m] = z

            # Now the U/V points
            # U
            nn = n * refi
            mm = m * refi + int(0.5 * refi)
            zgu = zg[nn : nn + refi, mm : mm + refi]
            zgu = np.transpose(zgu)
            manning = manning_grid[nn : nn + refi, mm : mm + refi]
            manning = np.transpose(manning)
            zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(
                zgu.flatten(), manning.flatten(), nbins, huthresh
            )
            u_zmin[n, m] = zmin
            u_zmax[n, m] = zmax
            u_havg[:, n, m] = havg
            u_nrep[:, n, m] = nrep
            u_pwet[:, n, m] = pwet
            u_ffit[n, m] = ffit
            u_navg[n, m] = navg

            # V
            nn = n * refi + int(0.5 * refi)
            mm = m * refi
            zgu = zg[nn : nn + refi, mm : mm + refi]
            manning = manning_grid[nn : nn + refi, mm : mm + refi]
            zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(
                zgu.flatten(), manning.flatten(), nbins, huthresh
            )
            v_zmin[n, m] = zmin
            v_zmax[n, m] = zmax
            v_havg[:, n, m] = havg
            v_nrep[:, n, m] = nrep
            v_pwet[:, n, m] = pwet
            v_ffit[n, m] = ffit
            v_navg[n, m] = navg

    return (
        z_zmin,
        z_zmax,
        z_volmax,
        z_level,
        u_zmin,
        u_zmax,
        u_havg,
        u_nrep,
        u_pwet,
        u_ffit,
        u_navg,
        v_zmin,
        v_zmax,
        v_havg,
        v_nrep,
        v_pwet,
        v_ffit,
        v_navg,
    )


### Following code mode be updated and used at some point
# @njit
# def process_block_quadtree(ds_mesh,
#                            ds_sbg,
#                            da_dep,
#                            da_man,
#                            index_cells_in_block,
#                            bn0,
#                            bm0,
#                            refi,
#                            nbins,
#                            is_geographic,
#                            max_gradient
#                            ):

#     """calculate subgrid properties for a single tile"""
#     nr_cells_in_block = index_cells_in_block.shape[0]

#     # Loop through all active cells in this block
#     for ic in range(nr_cells_in_block):

#         # Process cell

#         # Index in full quadtree mesh
#         indx = index_cells_in_block[ic]

#         # Get indices of pixels in block
#         nn  = (n[indx] - bn0) * refi # First pixel n index in cell
#         mm  = (m[indx] - bm0) * refi # First pixel m index in cell

#         # Matrix with pixels in cell
#         zv = da_dep.values[nn : nn + refi, mm : mm + refi].flatten()

#         # Compute pixel size in metres
#         if is_geographic:
#             # Compute latitude of cell

#             ygc = yg[nn : nn + refi, mm : mm + refi]

#             mean_lat =np.abs(np.mean(ygc))
#             dxpm = dxp*111111.0*np.cos(np.pi*mean_lat/180.0)
#             dypm = dyp*111111.0
#         else:
#             dxpm = dxp
#             dypm = dyp
        
# #        zv  = zgc.flatten()   
#         zvmin = -20.0
#         z, v, zmin, zmax = subgrid_v_table(zv, dxpm, dypm, nbins, zvmin, max_gradient)

#         # Check if this is an active point 
#         ds_sbg["z_zmin"].values[ic] = zmin
#         z_zmax[ic]    = zmax
#         z_volmax[ic]  = v[-1]
#         z_level[:,ic] = z
        
#         # Now the U/V points
#         # First right
#         if mu[indx] <= 0:
#             if mu1[indx] >= 0:
#                 nn  = (n[indx] - bn0)*refi
#                 mm  = (m[indx] - bm0)*refi + int(0.5*refi)
#                 zgu = zg[nn : nn + refi, mm : mm + refi]
#                 zgu = np.transpose(zgu)
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + refi, mm : mm + refi]
#                 manning = np.transpose(manning)
#                 manning = manning.flatten()
#                 iuv = index_mu1[indx]
#                 if iuv>=0:
#                     zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
#                     self.ds["uv_zmin"][iuv]   = zmin
#                     self.ds["uv_zmax"][iuv]   = zmax
#                     self.ds["uv_havg"][iuv,:] = havg
#                     self.ds["uv_nrep"][iuv,:] = nrep
#                     self.ds["uv_pwet"][iuv,:] = pwet
#                     self.ds["uv_ffit"][iuv]   = ffit
#                     self.ds["uv_navg"][iuv]   = navg
#         else:        
#             if mu1[indx] >= 0:
#                 nn = (n[indx] - bn0)*refi
#                 mm = (m[indx] - bm0)*refi + int(3*refi/4)
#                 zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 zgu = np.transpose(zgu)
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 manning = np.transpose(manning)
#                 manning = manning.flatten()
#                 iuv = index_mu1[indx]
#                 if iuv>=0:
#                     zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
#                     self.ds["uv_zmin"][iuv]   = zmin
#                     self.ds["uv_zmax"][iuv]   = zmax
#                     self.ds["uv_havg"][iuv,:] = havg
#                     self.ds["uv_nrep"][iuv,:] = nrep
#                     self.ds["uv_pwet"][iuv,:] = pwet
#                     self.ds["uv_ffit"][iuv]   = ffit
#                     self.ds["uv_navg"][iuv]   = navg
#             if mu2[indx] >= 0:
#                 nn = (n[indx] - bn0)*refi + int(refi/2)
#                 mm = (m[indx] - bm0)*refi + int(3*refi/4)
#                 zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 zgu = np.transpose(zgu)
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 manning = np.transpose(manning)
#                 manning = manning.flatten()
#                 iuv = index_mu2[indx]
#                 if iuv>=0:
#                     zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
#                     self.ds["uv_zmin"][iuv]   = zmin
#                     self.ds["uv_zmax"][iuv]   = zmax
#                     self.ds["uv_havg"][iuv,:] = havg
#                     self.ds["uv_nrep"][iuv,:] = nrep
#                     self.ds["uv_pwet"][iuv,:] = pwet
#                     self.ds["uv_ffit"][iuv]   = ffit
#                     self.ds["uv_navg"][iuv]   = navg

#         # Now above
#         if nu[indx] <= 0:
#             if nu1[indx] >= 0:
#                 nn = (n[indx] - bn0)*refi + int(0.5*refi)
#                 mm = (m[indx] - bm0)*refi
#                 zgu = zg[nn : nn + refi, mm : mm + refi]
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + refi, mm : mm + refi]
#                 manning = manning.flatten()
#                 iuv = index_nu1[indx]
#                 if iuv>=0:
#                     zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
#                     self.ds["uv_zmin"][iuv]   = zmin
#                     self.ds["uv_zmax"][iuv]   = zmax
#                     self.ds["uv_havg"][iuv,:] = havg
#                     self.ds["uv_nrep"][iuv,:] = nrep
#                     self.ds["uv_pwet"][iuv,:] = pwet
#                     self.ds["uv_ffit"][iuv]   = ffit
#                     self.ds["uv_navg"][iuv]   = navg
#         else:        
#             if nu1[indx] >= 0:
#                 nn = (n[indx] - bn0)*refi + int(3*refi/4)
#                 mm = (m[indx] - bm0)*refi
#                 zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 manning = manning.flatten()
#                 iuv = index_nu1[indx]
#                 if iuv>=0:
#                     zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
#                     self.ds["uv_zmin"][iuv]   = zmin
#                     self.ds["uv_zmax"][iuv]   = zmax
#                     self.ds["uv_havg"][iuv,:] = havg
#                     self.ds["uv_nrep"][iuv,:] = nrep
#                     self.ds["uv_pwet"][iuv,:] = pwet
#                     self.ds["uv_ffit"][iuv]   = ffit
#                     self.ds["uv_navg"][iuv]   = navg
#             if nu2[indx] >= 0:
#                 nn = (n[indx] - bn0)*refi + int(3*refi/4)
#                 mm = (m[indx] - bm0)*refi + int(refi/2)
#                 zgu = zg[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 zv  = zgu.flatten()
#                 manning = manning_grid[nn : nn + int(refi/2), mm : mm + int(refi/2)]
#                 manning = manning.flatten()
#                 iuv = index_nu2[indx]
#                 if iuv>=0:
#                     zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(zv, manning, nbins, huthresh)
#                     self.ds["uv_zmin"][iuv]   = zmin
#                     self.ds["uv_zmax"][iuv]   = zmax
#                     self.ds["uv_havg"][iuv,:] = havg
#                     self.ds["uv_nrep"][iuv,:] = nrep
#                     self.ds["uv_pwet"][iuv,:] = pwet
#                     self.ds["uv_ffit"][iuv]   = ffit
#                     self.ds["uv_navg"][iuv]   = navg

#     return (
#         z_zmin,
#         z_zmax,
#         # z_zmean,
#         z_volmax,
#         z_depth,
#         u_zmin,
#         u_zmax,
#         u_hrep,
#         u_navg,
#         v_zmin,
#         v_zmax,
#         v_hrep,
#         v_navg,
#     )



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
def subgrid_q_table(elevation: np.ndarray, manning: np.ndarray, nbins: int, huthresh:float):
    """
    map vector of elevation values into a hypsometric hydraulic radius - depth relationship for one u/v point
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    manning : np.ndarray (nr of pixels in one cell) containing subgrid manning roughness values for one grid cell [s m^(-1/3)]
    nbins : int, number of vertical bins [-]
    huthresh : float, threshold depth [m]
    Returns
    -------
    zmin : float, minimum elevation [m]
    zmax : float, maximum elevation [m]
    havg : np.ndarray (nbins) grid-average depth for vertical levels [m]
    nrep : np.ndarray (nbins) representative roughness for vertical levels [m1/3/s] ?
    pwet : np.ndarray (nbins) wet fraction for vertical levels [-] ?
    navg : float, grid-average Manning's n [m 1/3 / s]
    ffit : float, fitting coefficient [-]
    zz   : np.ndarray (nbins) elevation of vertical levels [m]
    """
    # Initialize output arrays
    havg = np.zeros(nbins)
    nrep = np.zeros(nbins)
    pwet = np.zeros(nbins)
    zz   = np.zeros(nbins)

    
    n   = int(np.size(elevation)) # Nr of pixels in grid cell
    n05 = int(n/2) # Index of middle pixel
  
    dd_a      = elevation[0:n05] # Pixel elevations side A 
    dd_b      = elevation[n05:] # Pixel elevations side B
    manning_a = manning[0:n05] # Pixel manning side A
    manning_b = manning[n05:] # Pixel manning side B

    zmin_a      = np.min(dd_a) # Minimum elevation side A
    zmax_a      = np.max(dd_a) # Maximum elevation side A
    
    zmin_b      = np.min(dd_b) # Minimum elevation side B
    zmax_b      = np.max(dd_b) # Maximum elevation side B
    
    zmin = max(zmin_a, zmin_b) + huthresh # Minimum elevation of uv point
    zmax = max(zmax_a, zmax_b) # Maximum elevation of uv point
    
    # Make sure zmax is always a bit higher than zmin
    if zmax<zmin + 0.001:
       zmax = max(zmax, zmin + 0.001)

    # Determine bin size (metres)
    dbin = (zmax - zmin)/(nbins - 1)

    # Grid mean roughness
    navg = np.mean(manning)
     
    # Loop through bins
    for ibin in range(nbins):

        # Top of bin
        zbin = zmin + ibin * dbin
        zz[ibin] = zbin
        
        # ibelow = np.where(elevation<=zbin)                           # index of pixels below bin level
        h      = np.maximum(zbin - elevation, 0.0)    # water depth in each pixel
        iwet   = np.where(zbin - elevation>-1.0e-6)[0]                           # indices of wet pixels
        hmean  = np.mean(h)
        havg[ibin] = hmean                      # conveyance depth
        pwet[ibin] = len(iwet)/n                # wet fraction

        # Side A
        h_a    = np.maximum(zbin - dd_a, 0.0)       # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        q_a    = h_a**(5.0/3.0)/manning_a           # Determine 'flux' for each pixel
        q_a    = np.mean(q_a)                       # Wet-average flux through all the pixels
        
        # Side B
        h_b    = np.maximum(zbin - dd_b, 0.0)       # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        q_b    = h_b**(5.0/3.0)/manning_b           # Determine 'flux' for each pixel
        q_b    = np.mean(q_b)                       # Wet-average flux through all the pixels
        
        q_ab   = np.minimum(q_a, q_b)

        q_all = h**(5.0/3.0)/manning               # Determine 'flux' for each pixel
        q_all = np.mean(q_all)                    # Wet-average flux through all the pixels
        
        # Weighted average of q_ab and q_all
        w = (ibin) / (nbins - 1)
        q = (1.0 - w) * q_ab + w * q_all

        nrep[ibin] = hmean**(5.0/3.0) / q  # Representative n for qmean and hmean

    nrep_top = nrep[-1]    
    havg_top = havg[-1]

    ### Fitting for nrep above zmax

    # Determine nfit at zfit
    zfit  = zmax + zmax - zmin
    h     = np.maximum(zfit - elevation, 0.0)      # water depth in each pixel
    hfit  = havg_top + zmax - zmin                 # mean water depth in cell as computed in SFINCS (assuming linear relation between water level and water depth above zmax)
    q     = h**(5.0/3.0)/manning                   # unit discharge in each pixel
    qmean = np.mean(q)                             # combined unit discharge for cell

    nfit  = hfit**(5.0/3.0) / qmean
    
    # Actually apply fit on gn2 (this is what is used in sfincs)
    gnavg2 = 9.81 * navg**2
    gnavg_top2 = 9.81 * nrep_top**2

    if gnavg2/gnavg_top2 > 0.99 and gnavg2/gnavg_top2 < 1.01:
        # gnavg2 and gnavg_top2 are almost identical
        ffit = 0.0
    else:
        if navg > nrep_top:
            if nfit > navg:
                nfit = nrep_top + 0.9*(navg - nrep_top)
            if nfit < nrep_top:
                nfit = nrep_top + 0.1*(navg - nrep_top)
        else:
            if nfit < navg:
                nfit = nrep_top + 0.9*(navg - nrep_top)
            if nfit > nrep_top:
                nfit = nrep_top + 0.1*(navg - nrep_top)
        gnfit2 = 9.81 * nfit**2
        ffit = (((gnavg2 - gnavg_top2) / (gnavg2 - gnfit2)) - 1) / (zfit - zmax)
         
    return zmin, zmax, havg, nrep, pwet, ffit, navg, zz       

# @njit
# def subgrid_q_table_old(elevation: np.ndarray, manning: np.ndarray, nbins: int):
#     """
#     map elevation values into a hypsometric hydraulic radius - depth relationship

#     Parameters
#     ----------
#     elevation: np.ndarray
#         subgrid elevation values for one grid cell [m]
#     manning: np.ndarray
#         subgrid manning roughness values for one grid cell [s m^(-1/3)]
#     nbins: int
#         number of bins to use for the hypsometric curve

#     Returns
#     -------
#     zmin, zmax: float
#         minimum and maximum elevation values used for hypsometric curve
#     hrep, navg, zz: np.ndarray
#         conveyance depth, average manning roughness, and elevation values
#         for each bin
#     """
#     hrep = np.zeros(nbins, dtype=np.float32)
#     navg = np.zeros(nbins, dtype=np.float32)
#     zz = np.zeros(nbins, dtype=np.float32)

#     n = int(elevation.size)  # Nr of pixels in grid cell
#     n05 = int(n / 2)

#     zmin_a = np.min(elevation[0:n05])
#     zmax_a = np.max(elevation[0:n05])

#     zmin_b = np.min(elevation[n05:])
#     zmax_b = np.max(elevation[n05:])

#     zmin = max(zmin_a, zmin_b)
#     zmax = max(zmax_a, zmax_b)

#     # Make sure zmax is a bit higher than zmin
#     if zmax < zmin + 0.01:
#         zmax += 0.01

#     # Determine bin size
#     dbin = (zmax - zmin) / nbins

#     # Loop through bins
#     for ibin in range(nbins):
#         # Top of bin
#         zbin = zmin + (ibin + 1) * dbin
#         zz[ibin] = zbin

#         ibelow = np.where(elevation <= zbin)  # index of pixels below bin level
#         # water depth in each pixel
#         h = np.maximum(zbin - np.maximum(elevation, zmin), 0.0)
#         qi = h ** (5.0 / 3.0) / manning  # unit discharge in each pixel
#         q = np.sum(qi) / n  # combined unit discharge for cell

#         navg[ibin] = manning[ibelow].mean()  # mean manning's n
#         hrep[ibin] = (q * navg[ibin]) ** (3.0 / 5.0)  # conveyance depth

#     return zmin, zmax, hrep, navg, zz
