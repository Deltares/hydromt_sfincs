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
    def __init__(self, version=1):
        # A regular subgrid table contains only for cells with msk>0
        self.version = version

    # new way of reading netcdf subgrid tables
    def read(self, file_name, mask):
        """Load subgrid table from netcdf file."""

        self.version = 1

        # Read data from netcdf file with xarray
        ds = xr.open_dataset(file_name)

        # transpose to have level as first dimension
        ds = ds.transpose("levels", "npuv", "np")

        # grid dimensions
        grid_dim = mask.shape

        # get number of levels, point and uv points
        self.nlevels, self.nr_cells, self.nr_uv_points = (
            ds.sizes["levels"],
            ds.sizes["np"],
            ds.sizes["npuv"],
        )

        # find indices of active cells
        index_nm, index_mu1, index_nu1 = utils.find_uv_indices(mask)
        active_indices = np.where(index_nm > -1)[0]

        # convert 1D indices to 2D indices
        active_cells = np.unravel_index(active_indices, grid_dim, order="F")

        # Initialize the data-arrays
        # Z points
        self.z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_level = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_havg = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_nrep = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_pwet = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_ffit = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_navg = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_havg = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_nrep = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_pwet = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_ffit = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)
        self.v_navg = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)

        # Now read the data and add it to the data-arrays
        # use index_nm of the active cells in the new dataset
        self.z_zmin[active_cells] = ds["z_zmin"].values.flatten()
        self.z_zmax[active_cells] = ds["z_zmax"].values.flatten()
        self.z_volmax[active_cells] = ds["z_volmax"].values.flatten()
        for ilevel in range(self.nlevels):
            self.z_level[ilevel, active_cells[0], active_cells[1]] = ds["z_level"][
                ilevel
            ].values.flatten()

        # now use index_mu1 and index_nu1 to put the values of the active cells in the new dataset
        var_list = ["zmin", "zmax", "ffit", "navg"]
        for var in var_list:
            uv_var = ds["uv_" + var].values.flatten()

            # Dynamically set the attribute for self.u_var and self.v_var
            u_attr_name = f"u_{var}"
            v_attr_name = f"v_{var}"

            # Retrieve the current attribute values
            u_array = getattr(self, u_attr_name)
            v_array = getattr(self, v_attr_name)

            # Update only the active indices
            u_array[active_cells] = uv_var[index_mu1[active_indices]]
            v_array[active_cells] = uv_var[index_nu1[active_indices]]

            # Set the modified arrays back to the attributes
            setattr(self, u_attr_name, u_array)
            setattr(self, v_attr_name, v_array)

        var_list_levels = ["havg", "nrep", "pwet"]
        for var in var_list_levels:
            for ilevel in range(self.nlevels):
                uv_var = ds["uv_" + var][ilevel].values.flatten()

                # Dynamically set the attribute for self.u_var and self.v_var
                u_attr_name = f"u_{var}"
                v_attr_name = f"v_{var}"

                # Retrieve the current attribute values
                u_array = getattr(self, u_attr_name)
                v_array = getattr(self, v_attr_name)

                # Update only the active indices
                u_array[ilevel, active_cells[0], active_cells[1]] = uv_var[
                    index_mu1[active_indices]
                ]
                v_array[ilevel, active_cells[0], active_cells[1]] = uv_var[
                    index_nu1[active_indices]
                ]

                # Set the modified arrays back to the attributes
                setattr(self, u_attr_name, u_array)
                setattr(self, v_attr_name, v_array)

        # close the dataset
        ds.close()

    # new way of writing netcdf subgrid tables
    def write(self, file_name, mask):
        """Write subgrid table to netcdf file for a regular grid with given mask.
        Values are only written for active cells (mask > 0)."""

        ds = self.to_xarray(dims=mask.raster.dims, coords=mask.raster.coords)

        # Need to transpose to match the FORTRAN convention in SFINCS
        ds = ds.transpose("levels", "x", "y")

        # find indices of active cells
        index_nm, index_mu1, index_nu1 = utils.find_uv_indices(mask)

        # get number of levels
        nlevels = self.nlevels

        active_cells = index_nm > -1
        active_indices = np.where(active_cells)[0]

        # get nr of active points (where index_nm > -1)
        nr_z_points = index_nm.max() + 1
        nr_uv_points = max(index_mu1.max(), index_nu1.max()) + 1

        # Make a new xarray dataset where we only keep the values of the active cells (index_nm > -1)
        # use index_nm to put the values of the active cells in the new dataset
        ds_new = xr.Dataset(attrs={"_FillValue": np.nan})

        # Z points
        variables = ["z_zmin", "z_zmax", "z_volmax"]
        for var in variables:
            ds_new[var] = xr.DataArray(
                ds[var].values.flatten()[active_cells], dims=("np")
            )

        z_level = np.zeros((nlevels, nr_z_points))
        for ilevel in range(nlevels):
            z_level[ilevel] = ds["z_level"][ilevel].values.flatten()[active_cells]
        ds_new["z_level"] = xr.DataArray(z_level, dims=("levels", "np"))

        # u and v points
        var_list = ["zmin", "zmax", "ffit", "navg"]
        for var in var_list:
            uv_var = np.zeros(nr_uv_points)
            uv_var[index_mu1[active_indices]] = ds["u_" + var].values.flatten()[
                active_cells
            ]
            uv_var[index_nu1[active_indices]] = ds["v_" + var].values.flatten()[
                active_cells
            ]
            ds_new[f"uv_{var}"] = xr.DataArray(uv_var, dims=("npuv"))

        var_list_levels = ["havg", "nrep", "pwet"]
        for var in var_list_levels:
            uv_var = np.zeros((nlevels, nr_uv_points))
            for ilevel in range(nlevels):
                uv_var[ilevel, index_mu1[active_indices]] = ds["u_" + var][
                    ilevel
                ].values.flatten()[active_cells]
                uv_var[ilevel, index_nu1[active_indices]] = ds["v_" + var][
                    ilevel
                ].values.flatten()[active_cells]
            ds_new[f"uv_{var}"] = xr.DataArray(uv_var, dims=("levels", "npuv"))

        # ensure levels is last dimension
        ds_new = ds_new.transpose("npuv", "np", "levels")

        # Write to netcdf file
        ds_new.to_netcdf(file_name)

    # Following remains for backward compatibility, but should soon not be used anymore
    def read_binary(self, file_name, mask):
        """Load subgrid table from file for a regular grid with given mask."""

        self.version = 0

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

        # Initialize the data-arrays
        self.nr_cells = np.fromfile(file, dtype=np.int32, count=1)[0]
        self.nr_uv_points = np.fromfile(file, dtype=np.int32, count=1)[0]
        self.nlevels = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Z points
        self.z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        # self.z_zmean = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_depth = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_hrep = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_navg = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_hrep = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_navg = np.full(
            (self.nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # Now read the data
        self.z_zmin[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        self.z_zmax[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        self.z_volmax[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        for ilevel in range(self.nlevels):
            self.z_depth[ilevel, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )

        self.u_zmin[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        self.u_zmax[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        _ = np.fromfile(file, dtype=np.float32, count=self.nr_cells)  # not used
        for ilevel in range(self.nlevels):
            self.u_hrep[ilevel, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )
        for ilevel in range(self.nlevels):
            self.u_navg[ilevel, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )

        self.v_zmin[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        self.v_zmax[iok[0], iok[1]] = np.fromfile(
            file, dtype=np.float32, count=self.nr_cells
        )
        _ = np.fromfile(file, dtype=np.float32, count=self.nr_cells)  # not used
        for ilevel in range(self.nlevels):
            self.v_hrep[ilevel, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )
        for ilevel in range(self.nlevels):
            self.v_navg[ilevel, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )

        file.close()

    # Following remains for backward compatibility, but should soon not be used anymore
    def write_binary(self, file_name, mask):
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
        file.write(np.int32(self.nlevels))

        # Z
        v = self.z_zmin[iok]
        file.write(np.float32(v))
        v = self.z_zmax[iok]
        file.write(np.float32(v))
        v = self.z_volmax[iok]
        file.write(np.float32(v))
        for ilevel in range(self.nlevels):
            v = np.squeeze(self.z_depth[ilevel, :, :])[iok]
            file.write(np.float32(v))

        # U
        v = self.u_zmin[iok]
        file.write(np.float32(v))
        v = self.u_zmax[iok]
        file.write(np.float32(v))
        dhdz = np.full(np.shape(v), 1.0)
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ilevel in range(self.nlevels):
            v = np.squeeze(self.u_hrep[ilevel, :, :])[iok]
            file.write(np.float32(v))
        for ilevel in range(self.nlevels):
            v = np.squeeze(self.u_navg[ilevel, :, :])[iok]
            file.write(np.float32(v))

        # V
        v = self.v_zmin[iok]
        file.write(np.float32(v))
        v = self.v_zmax[iok]
        file.write(np.float32(v))
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ilevel in range(self.nlevels):
            v = np.squeeze(self.v_hrep[ilevel, :, :])[iok]
            file.write(np.float32(v))
        for ilevel in range(self.nlevels):
            v = np.squeeze(self.v_navg[ilevel, :, :])[iok]
            file.write(np.float32(v))

        file.close()

    # This is the new way of building subgrid tables, that will end up in netcdf files
    def build(
        self,
        da_mask: xr.DataArray,
        datasets_dep: list[dict],
        datasets_rgh: list[dict] = [],
        datasets_riv: list[dict] = [],
        nlevels: int = 10,
        nr_subgrid_pixels: int = 20,
        nrmax: int = 2000,
        max_gradient: float = 99999.0,
        z_minimum: float = -99999.0,
        huthresh: float = 0.01,
        q_table_option: int = 2,
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
        nlevels : int, optional
            Number of levels in which hypsometry is subdivided, by default 10
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
        q_table_option : int, optional
            Option for the computation of the representative roughness and conveyance depth at u/v points, by default 2.
            1: "old" weighting method, compliant with SFINCS < v2.1.1, taking the avarage of the adjecent cells
            2: "improved" weighting method, recommended for SFINCS >= v2.1.1, that takes into account the wet fractions of the adjacent cells
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

        self.version = 1

        if write_dep_tif or write_man_tif:
            assert highres_dir is not None, "highres_dir must be specified"

        refi = nr_subgrid_pixels
        self.nlevels = nlevels
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
        self.z_level = np.full(
            (nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_havg = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_nrep = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_pwet = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_ffit = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_navg = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_havg = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.v_nrep = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
        self.v_pwet = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
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
                x_dim_dep, y_dim_dep = da_dep.raster.x_dim, da_dep.raster.y_dim
                window = Window(
                    bm0 * nr_subgrid_pixels,
                    bn0 * nr_subgrid_pixels,
                    da_dep[:-refi, :-refi].sizes[x_dim_dep],
                    da_dep[:-refi, :-refi].sizes[y_dim_dep],
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
                    yg = np.repeat(np.atleast_2d(yg).T, da_dep.raster.shape[1], axis=1)

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
                    nlevels,
                    yg,
                    max_gradient,
                    huthresh,
                    q_table_option,
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
        """Convert old binary subgrid class to xarray dataset."""
        ds_sbg = xr.Dataset(coords={"levels": np.arange(self.nlevels), **coords})
        ds_sbg.attrs.update({"_FillValue": np.nan})

        zlst2 = ["z_zmin", "z_zmax", "z_volmax"]
        if self.version == 0:
            uvlst2 = ["u_zmin", "u_zmax", "v_zmin", "v_zmax"]
            lst3 = ["z_depth", "u_hrep", "u_navg", "v_hrep", "v_navg"]

        elif self.version == 1:
            uvlst2 = [
                "u_zmin",
                "u_zmax",
                "u_ffit",
                "u_navg",
                "v_zmin",
                "v_zmax",
                "v_ffit",
                "v_navg",
            ]
            lst3 = [
                "z_level",
                "u_havg",
                "u_nrep",
                "u_pwet",
                "v_havg",
                "v_nrep",
                "v_pwet",
            ]

        # 2D arrays
        for name in zlst2 + uvlst2:
            if hasattr(self, name):
                ds_sbg[name] = xr.Variable(dims, getattr(self, name))
        # 3D arrays
        for name in lst3:
            if hasattr(self, name):
                ds_sbg[name] = xr.Variable(("levels", *dims), getattr(self, name))
        return ds_sbg

    def from_xarray(self, ds_sbg):
        """Convert xarray dataset to subgrid class."""
        for name in ds_sbg.data_vars:
            setattr(self, name, ds_sbg[name].values)


@njit
def process_tile_regular(
    mask,
    zg,
    manning_grid,
    dxp,
    dyp,
    refi,
    nlevels,
    yg,
    max_gradient,
    huthresh,
    q_table_option,
    is_geographic=False,
):
    """calculate subgrid properties for a single tile"""
    # Z points
    grid_dim = mask.shape
    z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_level = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)

    # U points
    u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_havg = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_nrep = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_pwet = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_ffit = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_navg = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)

    # V points
    v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_havg = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    v_nrep = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    v_pwet = np.full((nlevels, *grid_dim), fill_value=np.nan, dtype=np.float32)
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
                zgc.flatten(), dxpm, dypm, nlevels, zvmin, max_gradient
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
                zgu.flatten(), manning.flatten(), nlevels, huthresh, q_table_option
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
                zgu.flatten(), manning.flatten(), nlevels, huthresh, q_table_option
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
    nlevels: int,
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
    nlevels: int
        number of levels to use for the hypsometric curve
    zvolmin: float
        minimum elevation value to use for volume calculation (typically -20 m)
    max_gradient: float
        maximum gradient to use for volume calculation

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

    # Resample volumes to discrete levels
    steps = np.arange(nlevels) / (nlevels - 1)
    V = steps * volume.max()
    dvol = volume.max() / (nlevels - 1)
    # scipy not supported in numba jit
    # z = interpolate.interp1d(volume, ele_sort)(V)
    z = np.interp(V, volume, ele_sort)
    dzdh = get_dzdh(z, V, a)
    n = 0
    while (
        dzdh.max() > max_gradient and not (isclose(dzdh.max(), max_gradient))
    ) and n < nlevels:
        # reshape until gradient is satisfactory
        idx = np.where(dzdh == dzdh.max())[0]
        z[idx + 1] = z[idx] + max_gradient * (dvol / a)
        dzdh = get_dzdh(z, V, a)
        n += 1
    return z, V, elevation.min(), z.max()


@njit
def subgrid_q_table(
    elevation: np.ndarray,
    manning: np.ndarray,
    nlevels: int,
    huthresh: float,
    option: int,
):
    """
    map vector of elevation values into a hypsometric hydraulic radius - depth relationship for one u/v point
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    manning : np.ndarray (nr of pixels in one cell) containing subgrid manning roughness values for one grid cell [s m^(-1/3)]
    nlevels : int, number of vertical levels [-]
    huthresh : float, threshold depth [m]
    option : int, option to use "old" or "new" method for computing conveyance depth at u/v points

    Returns
    -------
    zmin : float, minimum elevation [m]
    zmax : float, maximum elevation [m]
    havg : np.ndarray (nlevels) grid-average depth for vertical levels [m]
    nrep : np.ndarray (nlevels) representative roughness for vertical levels [m1/3/s] ?
    pwet : np.ndarray (nlevels) wet fraction for vertical levels [-] ?
    navg : float, grid-average Manning's n [m 1/3 / s]
    ffit : float, fitting coefficient [-]
    zz   : np.ndarray (nlevels) elevation of vertical levels [m]
    """
    # Initialize output arrays
    havg = np.zeros(nlevels)
    nrep = np.zeros(nlevels)
    pwet = np.zeros(nlevels)
    zz = np.zeros(nlevels)

    n = int(elevation.size)  # Nr of pixels in grid cell
    # n   = int(np.size(elevation)) # Nr of pixels in grid cell

    n05 = int(n / 2)  # Index of middle pixel

    dd_a = elevation[0:n05]  # Pixel elevations side A
    dd_b = elevation[n05:]  # Pixel elevations side B
    manning_a = manning[0:n05]  # Pixel manning side A
    manning_b = manning[n05:]  # Pixel manning side B

    zmin_a = np.min(dd_a)  # Minimum elevation side A
    zmax_a = np.max(dd_a)  # Maximum elevation side A

    zmin_b = np.min(dd_b)  # Minimum elevation side B
    zmax_b = np.max(dd_b)  # Maximum elevation side B

    zmin = max(zmin_a, zmin_b) + huthresh  # Minimum elevation of uv point
    zmax = max(zmax_a, zmax_b)  # Maximum elevation of uv point

    # Make sure zmax is always a bit higher than zmin
    if zmax < zmin + 0.001:
        zmax = max(zmax, zmin + 0.001)

    # Determine level size (metres)
    dlevel = (zmax - zmin) / (nlevels - 1)

    # Option can be either 1 ("old, compliant with SFINCS < v2.1.") or 2 ("new", recommended SFINCS >= v2.1.)
    option = option

    # Loop through levels
    for ibin in range(nlevels):
        # Top of bin
        zbin = zmin + ibin * dlevel
        zz[ibin] = zbin

        h = np.maximum(zbin - elevation, 0.0)  # water depth in each pixel

        pwet[ibin] = (zbin - elevation > -1.0e-6).sum() / n

        # Side A
        h_a = np.maximum(
            zbin - dd_a, 0.0
        )  # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        q_a = h_a ** (5.0 / 3.0) / manning_a  # Determine 'flux' for each pixel
        q_a = np.mean(q_a)  # Grid-average flux through all the pixels
        h_a = np.mean(h_a)  # Grid-average depth through all the pixels

        # Side B
        h_b = np.maximum(
            zbin - dd_b, 0.0
        )  # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        q_b = h_b ** (5.0 / 3.0) / manning_b  # Determine 'flux' for each pixel
        q_b = np.mean(q_b)  # Grid-average flux through all the pixels
        h_b = np.mean(h_b)  # Grid-average depth through all the pixels

        # Compute q and h
        q_all = np.mean(
            h ** (5.0 / 3.0) / manning
        )  # Determine grid average 'flux' for each pixel
        h_all = np.mean(h)  # grid averaged depth of A and B combined
        q_min = np.minimum(q_a, q_b)
        h_min = np.minimum(h_a, h_b)

        if option == 1:
            # Use old 1 option (weighted average of q_ab and q_all) option (min at bottom bin, mean at top bin)
            w = (ibin) / (
                nlevels - 1
            )  # Weight (increase from 0 to 1 from bottom to top bin)
            q = (1.0 - w) * q_min + w * q_all  # Weighted average of q_min and q_all
            hmean = h_all

        elif option == 2:
            # Use newer 2 option (minimum of q_a an q_b, minimum of h_a and h_b increasing to h_all, using pwet for weighting) option
            pwet_a = (zbin - dd_a > -1.0e-6).sum() / (n / 2)
            pwet_b = (zbin - dd_b > -1.0e-6).sum() / (n / 2)
            # Weight increases linearly from 0 to 1 from bottom to top bin use percentage wet in sides A and B
            w = 2 * np.minimum(pwet_a, pwet_b) / (pwet_a + pwet_b)
            q = (1.0 - w) * q_min + w * q_all  # Weighted average of q_min and q_all
            hmean = (1.0 - w) * h_min + w * h_all  # Weighted average of h_min and h_all

        havg[ibin] = hmean  # conveyance depth
        nrep[ibin] = hmean ** (5.0 / 3.0) / q  # Representative n for qmean and hmean

    nrep_top = nrep[-1]
    havg_top = havg[-1]

    ### Fitting for nrep above zmax

    # Determine nfit at zfit
    zfit = zmax + zmax - zmin
    hfit = (
        havg_top + zmax - zmin
    )  # mean water depth in cell as computed in SFINCS (assuming linear relation between water level and water depth above zmax)

    # Compute q and navg
    h = np.maximum(zfit - elevation, 0.0)  # water depth in each pixel
    q = np.mean(h ** (5.0 / 3.0) / manning)  # combined unit discharge for cell
    navg = np.mean(manning)

    nfit = hfit ** (5.0 / 3.0) / q

    # Actually apply fit on gn2 (this is what is used in sfincs)
    gnavg2 = 9.81 * navg**2
    gnavg_top2 = 9.81 * nrep_top**2

    if gnavg2 / gnavg_top2 > 0.99 and gnavg2 / gnavg_top2 < 1.01:
        # gnavg2 and gnavg_top2 are almost identical
        ffit = 0.0
    else:
        if navg > nrep_top:
            if nfit > navg:
                nfit = nrep_top + 0.9 * (navg - nrep_top)
            if nfit < nrep_top:
                nfit = nrep_top + 0.1 * (navg - nrep_top)
        else:
            if nfit < navg:
                nfit = nrep_top + 0.9 * (navg - nrep_top)
            if nfit > nrep_top:
                nfit = nrep_top + 0.1 * (navg - nrep_top)
        gnfit2 = 9.81 * nfit**2
        ffit = (((gnavg2 - gnavg_top2) / (gnavg2 - gnfit2)) - 1) / (zfit - zmax)

    return zmin, zmax, havg, nrep, pwet, ffit, navg, zz
