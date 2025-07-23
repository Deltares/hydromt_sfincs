# -*- coding: utf-8 -*-
"""
Subgrid table builder for SFINCS Quadtree model
Created on Mon Mar 03 2025

@author: ormondt
"""
import time

import numpy as np
import xarray as xr
from numba import njit
from pyproj import CRS

from hydromt_sfincs.workflows.subgrid import *


def build_subgrid_table_quadtree(
    grid,
    bathymetry_sets,
    roughness_sets,
    manning_land=0.06,
    manning_water=0.020,
    manning_level=1.0,
    nr_levels=10,
    nr_subgrid_pixels=20,
    max_gradient=999.0,
    depth_factor=1.0,
    huthresh=0.01,
    zmin=-999999.0,
    zmax=999999.0,
    weight_option="min",
    bathymetry_database=None,
    quiet=True,
    progress_bar=None,
    logger=None,
) -> xr.Dataset:
    subgrid_table = SubgridTableQuadtree()

    subgrid_table.build(
        grid,
        bathymetry_sets,
        roughness_sets,
        manning_land,
        manning_water,
        manning_level,
        nr_levels,
        nr_subgrid_pixels,
        max_gradient,
        depth_factor,
        huthresh,
        zmin,
        zmax,
        weight_option,
        bathymetry_database,
        quiet,
        progress_bar,
        logger,
    )

    return subgrid_table.ds


class SubgridTableQuadtree:
    def __init__(self, data=None):
        self.ds = xr.Dataset()

    def build(
        self,
        grid,  # xugrid dataset (i.e. model.grid)
        bathymetry_sets,  # list of bathymetry datasets
        roughness_sets,  # list of roughness datasets
        manning_land,
        manning_water,
        manning_level,
        nr_levels,
        nr_subgrid_pixels,
        max_gradient,
        depth_factor,
        huthresh,
        zmin,
        zmax,
        weight_option,
        bathymetry_database,
        quiet,
        progress_bar,
        logger,
    ):
        version = "1.0"

        time_start = time.time()

        crs = CRS(int(grid.crs.values))

        msg = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        log_info(msg, logger, quiet)

        msg = "Building subgrid tables for SFINCS Quadtree model ..."
        log_info(msg, logger, quiet)

        # Dimensions etc
        refi = nr_subgrid_pixels
        nr_cells = grid.sizes["mesh2d_nFaces"]
        nr_ref_levs = grid.attrs["nr_levels"]  # number of refinement levels
        cosrot = np.cos(grid.attrs["rotation"] * np.pi / 180)
        sinrot = np.sin(grid.attrs["rotation"] * np.pi / 180)
        nrmax = 2000
        zminimum = zmin
        zmaximum = zmax

        # Grid neighbors (subtract 1 from indices to get zero-based indices)
        level = grid["level"].values[:] - 1
        n = grid["n"].values[:] - 1
        m = grid["m"].values[:] - 1
        nu = grid["nu"].values[:]
        nu1 = grid["nu1"].values[:] - 1
        nu2 = grid["nu2"].values[:] - 1
        mu = grid["mu"].values[:]
        mu1 = grid["mu1"].values[:] - 1
        mu2 = grid["mu2"].values[:] - 1

        # U/V points
        # Need to count the number of uv points in order allocate arrays (probably better to store this in the grid)
        # Loop through cells to count number of velocity points
        npuv = 0
        for ip in range(nr_cells):
            if mu1[ip] >= 0:
                npuv += 1
            if mu2[ip] >= 0:
                npuv += 1
            if nu1[ip] >= 0:
                npuv += 1
            if nu2[ip] >= 0:
                npuv += 1

        # Allocate some arrays with info about the uv points
        uv_index_z_nm = np.zeros(npuv, dtype=int)
        uv_index_z_nmu = np.zeros(npuv, dtype=int)
        uv_flags_dir = np.zeros(npuv, dtype=int)
        uv_flags_level = np.zeros(npuv, dtype=int)
        uv_flags_type = np.zeros(npuv, dtype=int)

        # Determine what type of uv point it is
        ip = -1
        for ic in range(nr_cells):
            if mu[ic] <= 0:
                # Regular or coarser to the right
                if mu1[ic] >= 0:
                    ip += 1
                    uv_index_z_nm[ip] = ic
                    uv_index_z_nmu[ip] = mu1[ic]
                    uv_flags_dir[ip] = 0
                    uv_flags_level[ip] = level[ic]
                    uv_flags_type[ip] = mu[ic]
            else:
                if mu1[ic] >= 0:
                    ip += 1
                    uv_index_z_nm[ip] = ic
                    uv_index_z_nmu[ip] = mu1[ic]
                    uv_flags_dir[ip] = 0  # x
                    uv_flags_level[ip] = level[ic] + 1
                    uv_flags_type[ip] = mu[ic]
                if mu2[ic] >= 0:
                    ip += 1
                    uv_index_z_nm[ip] = ic
                    uv_index_z_nmu[ip] = mu2[ic]
                    uv_flags_dir[ip] = 0  # x
                    uv_flags_level[ip] = level[ic] + 1
                    uv_flags_type[ip] = mu[ic]
            if nu[ic] <= 0:
                # Regular or coarser above
                if nu1[ic] >= 0:
                    ip += 1
                    uv_index_z_nm[ip] = ic
                    uv_index_z_nmu[ip] = nu1[ic]
                    uv_flags_dir[ip] = 1
                    uv_flags_level[ip] = level[ic]
                    uv_flags_type[ip] = nu[ic]
            else:
                if nu1[ic] >= 0:
                    ip += 1
                    uv_index_z_nm[ip] = ic
                    uv_index_z_nmu[ip] = nu1[ic]
                    uv_flags_dir[ip] = 1  # y
                    uv_flags_level[ip] = level[ic] + 1
                    uv_flags_type[ip] = nu[ic]
                if nu2[ic] >= 0:
                    ip += 1
                    uv_index_z_nm[ip] = ic
                    uv_index_z_nmu[ip] = nu2[ic]
                    uv_flags_dir[ip] = 1
                    uv_flags_level[ip] = level[ic] + 1
                    uv_flags_type[ip] = nu[ic]

        npc = nr_cells

        # Create numpy arrays with empty arrays
        self.z_zmin = np.zeros(npc, dtype=np.float32)
        self.z_zmax = np.zeros(npc, dtype=np.float32)
        self.z_volmax = np.zeros(npc, dtype=np.float32)
        self.z_level = np.zeros((nr_levels, npc), dtype=np.float32)
        self.uv_zmin = np.zeros(npuv, dtype=np.float32)
        self.uv_zmax = np.zeros(npuv, dtype=np.float32)
        self.uv_havg = np.zeros((nr_levels, npuv), dtype=np.float32)
        self.uv_nrep = np.zeros((nr_levels, npuv), dtype=np.float32)
        self.uv_pwet = np.zeros((nr_levels, npuv), dtype=np.float32)
        self.uv_ffit = np.zeros(npuv, dtype=np.float32)
        self.uv_navg = np.zeros(npuv, dtype=np.float32)

        # Determine first indices and number of cells per refinement level
        ifirst = np.zeros(nr_ref_levs, dtype=int)
        ilast = np.zeros(nr_ref_levs, dtype=int)
        nr_cells_per_level = np.zeros(nr_ref_levs, dtype=int)
        ireflast = -1
        for ic in range(nr_cells):
            if level[ic] > ireflast:
                ifirst[level[ic]] = ic
                ireflast = level[ic]
        for ilev in range(nr_ref_levs - 1):
            ilast[ilev] = ifirst[ilev + 1] - 1
        ilast[nr_ref_levs - 1] = nr_cells - 1
        for ilev in range(nr_ref_levs):
            nr_cells_per_level[ilev] = ilast[ilev] - ifirst[ilev] + 1

        # Loop through all levels
        for ilev in range(nr_ref_levs):
            msg = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            log_info(msg, logger, quiet)
            msg = f"Processing level {ilev + 1} of {nr_ref_levs} ..."
            log_info(msg, logger, quiet)

            # Make blocks off cells in this level only
            cell_indices_in_level = np.arange(ifirst[ilev], ilast[ilev] + 1, dtype=int)
            nr_cells_in_level = np.size(cell_indices_in_level)

            if nr_cells_in_level == 0:
                continue

            n0 = np.min(n[ifirst[ilev] : ilast[ilev] + 1])
            n1 = np.max(
                n[ifirst[ilev] : ilast[ilev] + 1]
            )  # + 1 # add extra cell to compute u and v in the last row/column
            m0 = np.min(m[ifirst[ilev] : ilast[ilev] + 1])
            m1 = np.max(
                m[ifirst[ilev] : ilast[ilev] + 1]
            )  # + 1 # add extra cell to compute u and v in the last row/column

            dx = grid.attrs["dx"] / 2**ilev  # cell size
            dy = grid.attrs["dy"] / 2**ilev  # cell size
            dxp = dx / refi  # size of subgrid pixel
            dyp = dy / refi  # size of subgrid pixel

            nrcb = int(np.floor(nrmax / refi))  # nr of regular cells in a block
            nrbn = int(np.ceil((n1 - n0 + 1) / nrcb))  # nr of blocks in n direction
            nrbm = int(np.ceil((m1 - m0 + 1) / nrcb))  # nr of blocks in m direction

            msg = "Number of regular cells in a block : " + str(nrcb)
            log_info(msg, logger, quiet)
            msg = "Number of blocks in n direction    : " + str(nrbn)
            log_info(msg, logger, quiet)
            msg = "Number of blocks in m direction    : " + str(nrbm)
            log_info(msg, logger, quiet)

            msg = (
                "Grid size of flux grid             : dx= "
                + str(dx)
                + ", dy= "
                + str(dy)
            )
            log_info(msg, logger, quiet)
            msg = (
                "Grid size of subgrid pixels        : dx= "
                + str(dxp)
                + ", dy= "
                + str(dyp)
            )
            log_info(msg, logger, quiet)

            ibt = 1
            if progress_bar:
                progress_bar.set_text(
                    "               Generating Sub-grid Tables (level "
                    + str(ilev)
                    + ") ...                "
                )
                progress_bar.set_maximum(nrbm * nrbn)

            ### CELL CENTRES

            # Loop through blocks
            ib = -1
            for ii in range(nrbm):
                for jj in range(nrbn):
                    # Count
                    ib += 1

                    msg = (
                        "--------------------------------------------------------------"
                    )
                    log_info(msg, logger, quiet)
                    msg = (
                        "Processing block "
                        + str(ib + 1)
                        + " of "
                        + str(nrbn * nrbm)
                        + " ..."
                    )
                    log_info(msg, logger, quiet)

                    # Block n,m indices
                    bn0 = n0 + jj * nrcb  # Index of first n in block
                    bn1 = (
                        min(bn0 + nrcb - 1, n1) + 1
                    )  # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
                    bm0 = m0 + ii * nrcb  # Index of first m in block
                    bm1 = (
                        min(bm0 + nrcb - 1, m1) + 1
                    )  # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

                    ###########
                    # Indices #
                    ###########

                    # First we loop through all the possible cells in this block
                    index_cells_in_block = np.zeros(nrcb * nrcb, dtype=int)

                    # Loop through all cells in this level
                    nr_cells_in_block = 0
                    for ic in range(nr_cells_in_level):
                        indx = cell_indices_in_level[ic]  # index of the whole quadtree
                        if (
                            n[indx] >= bn0
                            and n[indx] < bn1
                            and m[indx] >= bm0
                            and m[indx] < bm1
                        ):
                            # Cell falls inside block
                            index_cells_in_block[nr_cells_in_block] = indx
                            nr_cells_in_block += 1
                    index_cells_in_block = index_cells_in_block[0:nr_cells_in_block]

                    msg = f"Number of cells in this block      : {nr_cells_in_block}"
                    log_info(msg, logger, quiet)

                    yc = grid.grid.face_coordinates[index_cells_in_block, 1]

                    ##############
                    # Bathy/topo #
                    ##############

                    # Get the numpy array zg with bathy/topo values for this block

                    msg = "Getting bathy/topo ..."
                    log_info(msg, logger, quiet)

                    if bathymetry_database:
                        # Delft Dashboard
                        # Get bathymetry on subgrid from bathymetry database

                        # Build the pixel matrix
                        x00 = 0.5 * dxp + bm0 * refi * dxp
                        x01 = x00 + (bm1 - bm0 + 1) * refi * dxp
                        y00 = 0.5 * dyp + bn0 * refi * dyp
                        y01 = y00 + (bn1 - bn0 + 1) * refi * dyp

                        x0 = np.arange(x00, x01, dxp)
                        y0 = np.arange(y00, y01, dyp)
                        xg0, yg0 = np.meshgrid(x0, y0)

                        # Rotate and translate
                        xg = grid.attrs["x0"] + cosrot * xg0 - sinrot * yg0
                        yg = grid.attrs["y0"] + sinrot * xg0 + cosrot * yg0

                        # Clear variables
                        del x0, y0, xg0, yg0

                        zg = bathymetry_database.get_bathymetry_on_grid(
                            xg, yg, crs, bathymetry_sets, method="linear"
                        )
                    else:
                        # HydroMT
                        # zg = da_dep.values
                        pass

                    # Multiply zg with depth factor (had to use 0.9746 to get arrival
                    # times right in the Pacific)
                    zg = zg * depth_factor

                    # Set minimum depth
                    zg = np.maximum(zg, zminimum)
                    zg = np.minimum(zg, zmaximum)

                    # replace NaNs with 0.0
                    zg[np.isnan(zg)] = 0.0

                    ##########################
                    # Process cells in block #
                    ##########################

                    msg = "Processing cells ..."
                    log_info(msg, logger, quiet)

                    (
                        self.z_zmin[index_cells_in_block],
                        self.z_zmax[index_cells_in_block],
                        self.z_volmax[index_cells_in_block],
                        self.z_level[:, index_cells_in_block],
                    ) = process_block_cells(
                        zg,  # depth array
                        nr_cells_in_block,  # number of cells in this block
                        n[index_cells_in_block],  # n index of cells in this block
                        m[index_cells_in_block],  # m index of cells in this block
                        yc,  # y coordinate of cells in this block
                        bn0,  # first n index in this block
                        bm0,  # first m index in this block
                        dxp,  # pixel size in x direction
                        dyp,  # pixel size in y direction
                        refi,  # refinement factor
                        nr_levels,  # number of levels
                        max_gradient,  # maximum gradient
                        crs.is_geographic,  # is geographic
                    )

                    if progress_bar:
                        progress_bar.set_value(ibt)
                        if progress_bar.was_canceled():
                            return
                        ibt += 1

            # UV Points

            # Loop through blocks
            ib = -1
            for ii in range(nrbm):
                for jj in range(nrbn):
                    # Count
                    ib += 1

                    msg = (
                        "--------------------------------------------------------------"
                    )
                    log_info(msg, logger, quiet)
                    msg = (
                        f"Processing U/V points in block {ib + 1} of {nrbn * nrbm} ..."
                    )
                    log_info(msg, logger, quiet)

                    # Block n,m indices
                    bn0 = n0 + jj * nrcb  # Index of first n in block
                    bn1 = (
                        min(bn0 + nrcb - 1, n1) + 1
                    )  # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
                    bm0 = m0 + ii * nrcb  # Index of first m in block
                    bm1 = (
                        min(bm0 + nrcb - 1, m1) + 1
                    )  # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

                    ###########
                    # Indices #
                    ###########

                    # First we loop through all the uv points to find the ones in this block
                    index_uv_points_in_block = np.zeros(npuv, dtype=int)
                    n_nm = np.zeros(npuv, dtype=int)
                    m_nm = np.zeros(npuv, dtype=int)
                    n_nmu = np.zeros(npuv, dtype=int)
                    m_nmu = np.zeros(npuv, dtype=int)
                    z_zmin_nm = np.zeros(npuv, dtype=float)
                    z_zmin_nmu = np.zeros(npuv, dtype=float)

                    iuv = 0

                    for ip in range(npuv):
                        # Check if this uv point is at the correct level
                        if uv_flags_level[ip] != ilev:
                            continue

                        # Check if this uv point is in this block
                        nm = uv_index_z_nm[ip]  # Index of left hand cell neighbor
                        nmu = uv_index_z_nmu[ip]  # Index of right hand cell neighbor

                        # Now build the pixel matrix
                        if uv_flags_type[ip] <= 0:
                            # Normal point or fine to coarse
                            if (
                                n[nm] < bn0
                                or n[nm] >= bn1
                                or m[nm] < bm0
                                or m[nm] >= bm1
                            ):
                                # Outside block
                                continue
                        else:
                            # Coarse to fine
                            if (
                                n[nmu] < bn0
                                or n[nmu] >= bn1
                                or m[nmu] < bm0
                                or m[nmu] >= bm1
                            ):
                                # Outside block
                                continue

                        # Found a uv point in this block
                        index_uv_points_in_block[iuv] = ip
                        n_nm[iuv] = n[nm]
                        m_nm[iuv] = m[nm]
                        n_nmu[iuv] = n[nmu]
                        m_nmu[iuv] = m[nmu]
                        z_zmin_nm[iuv] = self.z_zmin[nm]
                        z_zmin_nmu[iuv] = self.z_zmin[nmu]
                        iuv += 1

                    nr_uv_points_in_block = iuv

                    if iuv == 0:
                        # No uv points in this block
                        continue

                    # Found all the cells in this block
                    index_uv_points_in_block = index_uv_points_in_block[
                        0:nr_uv_points_in_block
                    ]
                    n_nm = n_nm[0:nr_uv_points_in_block]
                    m_nm = m_nm[0:nr_uv_points_in_block]
                    n_nmu = n_nmu[0:nr_uv_points_in_block]
                    m_nmu = m_nmu[0:nr_uv_points_in_block]
                    z_zmin_nm = z_zmin_nm[0:nr_uv_points_in_block]
                    z_zmin_nmu = z_zmin_nmu[0:nr_uv_points_in_block]

                    msg = f"Number of U/V points in this block: {nr_uv_points_in_block}"
                    log_info(msg, logger, quiet)

                    ###########################
                    # Bathy/topo and Mannning #
                    ###########################

                    msg = "Getting bathy/topo ..."
                    log_info(msg, logger, quiet)

                    # Get the numpy array zg with bathy/topo values for this block
                    if bathymetry_database:
                        # Delft Dashboard
                        # Get bathymetry on subgrid from bathymetry database

                        # Build the pixel matrix
                        x00 = (
                            0.5 * dxp + bm0 * refi * dxp - 0.5 * refi * dxp
                        )  # start half a cell to the left
                        x01 = x00 + (bm1 - bm0 + 1) * refi * dxp
                        y00 = (
                            0.5 * dyp + bn0 * refi * dyp - 0.5 * refi * dyp
                        )  # start half a cell below
                        y01 = y00 + (bn1 - bn0 + 1) * refi * dyp

                        x0 = np.arange(x00, x01, dxp)
                        y0 = np.arange(y00, y01, dyp)
                        xg0, yg0 = np.meshgrid(x0, y0)

                        # Rotate and translate
                        xg = grid.attrs["x0"] + cosrot * xg0 - sinrot * yg0
                        yg = grid.attrs["y0"] + sinrot * xg0 + cosrot * yg0

                        # Clear variables
                        del x0, y0, xg0, yg0

                        zg = bathymetry_database.get_bathymetry_on_grid(
                            xg, yg, crs, bathymetry_sets
                        )

                    else:
                        # HydroMT
                        # zg = da_dep.values
                        pass

                    # Multiply zg with depth factor (had to use 0.9746 to get arrival
                    # times right in the Pacific)
                    zg = zg * depth_factor

                    # Set minimum depth
                    zg = np.maximum(zg, zminimum)
                    zg = np.minimum(zg, zmaximum)

                    # replace NaNs with 0.0
                    zg[np.isnan(zg)] = 0.0

                    # Manning's n values

                    # TODO: Implement roughness sets for HydroMT

                    # Initialize roughness of subgrid at NaN
                    manning_grid = np.full(np.shape(xg), np.nan)

                    if roughness_sets:  # this still needs to be implemented
                        manning_grid = bathymetry_database.get_bathymetry_on_grid(
                            xg, yg, crs, roughness_sets
                        )

                    # Fill in remaining NaNs with default values
                    isn = np.where(np.isnan(manning_grid))
                    try:
                        manning_grid[
                            (isn and np.where(zg <= manning_level))
                        ] = manning_water
                    except:
                        pass
                    manning_grid[(isn and np.where(zg > manning_level))] = manning_land

                    ###############################
                    # Process U/V points in block #
                    ###############################

                    msg = "Processing U/V points ..."
                    log_info(msg, logger, quiet)

                    (
                        self.uv_zmin[index_uv_points_in_block],
                        self.uv_zmax[index_uv_points_in_block],
                        self.uv_havg[:, index_uv_points_in_block],
                        self.uv_nrep[:, index_uv_points_in_block],
                        self.uv_pwet[:, index_uv_points_in_block],
                        self.uv_ffit[index_uv_points_in_block],
                        self.uv_navg[index_uv_points_in_block],
                    ) = process_block_uv_points(
                        zg,  # depth array
                        manning_grid,  # manning array
                        nr_uv_points_in_block,  # number of cells in this block
                        n_nm,  # n index of cells in this block
                        m_nm,  # m index of cells in this block
                        n_nmu,  # n index of nmu neighbor in this block
                        m_nmu,  # m index of nmu neighbor in this block
                        z_zmin_nm,  # zmin of nm neighbor
                        z_zmin_nmu,  # zmin of nmu neighbor
                        uv_flags_type[index_uv_points_in_block],  # type of uv point
                        uv_flags_dir[index_uv_points_in_block],  # direction of uv point
                        bn0,  # first n index in this block
                        bm0,  # first m index in this block
                        refi,  # refinement factor
                        nr_levels,  # number of levels
                        huthresh,  # huthresh
                        weight_option,  # weight option ("min" or "mean")
                    )

                    if progress_bar:
                        progress_bar.set_value(ibt)
                        if progress_bar.was_canceled():
                            return
                        ibt += 1

        # Now create the xarray dataset (do we transpose here? is this necessary for fortan?)
        self.ds = xr.Dataset()
        self.ds.attrs["version"] = version
        self.ds["z_zmin"] = xr.DataArray(self.z_zmin, dims=["np"])
        self.ds["z_zmax"] = xr.DataArray(self.z_zmax, dims=["np"])
        self.ds["z_volmax"] = xr.DataArray(self.z_volmax, dims=["np"])
        self.ds["z_level"] = xr.DataArray(
            np.transpose(self.z_level), dims=["np", "levels"]
        )
        self.ds["uv_zmin"] = xr.DataArray(self.uv_zmin, dims=["npuv"])
        self.ds["uv_zmax"] = xr.DataArray(self.uv_zmax, dims=["npuv"])
        self.ds["uv_havg"] = xr.DataArray(
            np.transpose(self.uv_havg), dims=["npuv", "levels"]
        )
        self.ds["uv_nrep"] = xr.DataArray(
            np.transpose(self.uv_nrep), dims=["npuv", "levels"]
        )
        self.ds["uv_pwet"] = xr.DataArray(
            np.transpose(self.uv_pwet), dims=["npuv", "levels"]
        )
        self.ds["uv_ffit"] = xr.DataArray(self.uv_ffit, dims=["npuv"])
        self.ds["uv_navg"] = xr.DataArray(self.uv_navg, dims=["npuv"])

        msg = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        log_info(msg, logger, quiet)

        time_end = time.time()
        msg = f"Time elapsed: {(time_end - time_start):.1f} seconds"
        log_info(msg, logger, quiet)

        msg = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        log_info(msg, logger, quiet)

        # All done

        # Close progress bar
        if progress_bar:
            progress_bar.close()


@njit
def process_block_cells(
    zg,  # array with bathy/topo values for this block
    nr_cells_in_block,  # number of cells in this block
    n,  # n index of cells in this block
    m,  # m index of cells in this block
    yc,  # y coordinate of cells in this block
    bn0,  # first n index in this block
    bm0,  # first m index in this block
    dxp,  # pixel size in x direction
    dyp,  # pixel size in y direction
    refi,  # refinement factor
    nr_levels,  # number of levels
    max_gradient,  # maximum gradient
    is_geographic,  # is geographic
):
    """calculate subgrid properties for a single block of cells"""

    z_zmin = np.full((nr_cells_in_block), fill_value=np.nan, dtype=np.float32)
    z_zmax = np.full((nr_cells_in_block), fill_value=np.nan, dtype=np.float32)
    z_volmax = np.full((nr_cells_in_block), fill_value=np.nan, dtype=np.float32)
    z_level = np.full(
        (nr_levels, nr_cells_in_block), fill_value=np.nan, dtype=np.float32
    )

    for ic in range(nr_cells_in_block):
        # Get the index in the entire quadtree
        # indx = index_cells_in_block[ic]

        # Pixel indices for this cell
        nn = (n[ic] - bn0) * refi
        mm = (m[ic] - bm0) * refi

        # Bathy/topo data for this cell
        zgc = zg[nn : nn + refi, mm : mm + refi]

        # Compute pixel size in metres
        if is_geographic:
            mean_lat = float(np.abs(yc[ic]))
            dxpm = float(dxp * 111111.0 * np.cos(np.pi * mean_lat / 180.0))
            dypm = float(dyp * 111111.0)
        else:
            dxpm = float(dxp)
            dypm = float(dyp)

        # Bathy/topo data for this cell
        zgc = zg[nn : nn + refi, mm : mm + refi]

        zvmin = -20.0
        z, v, zmin, zmax = subgrid_v_table(
            zgc.flatten(), dxpm, dypm, nr_levels, zvmin, max_gradient
        )
        z_zmin[ic] = zmin
        z_zmax[ic] = zmax
        z_volmax[ic] = v[-1]
        z_level[:, ic] = z

    return (
        z_zmin,
        z_zmax,
        z_volmax,
        z_level,
    )


@njit
def process_block_uv_points(
    zg,  # array with bathy/topo values for this block
    manning,  # array with manning values for this block
    nr_uv_points_in_block,  # number of cells in this block
    n,  # n index of nm neighbor in this block
    m,  # m index of nm neighbor in this block
    n_nmu,  # n index of nmu neighbor in this block
    m_nmu,  # m index of nmu neighbor in this block
    z_zmin_nm,  # zmin of nm neighbor
    z_zmin_nmu,  # zmin of nmu neighbor
    uv_flags_type,  # type of uv point
    uv_flags_dir,  # direction of uv point
    bn0,  # first n index in this block
    bm0,  # first m index in this block
    refi,  # refinement factor
    nr_levels,  # number of levels
    huthresh,  # huthresh
    weight_option,  # weight option
):
    """calculate subgrid properties for a single block of uv points"""

    uv_zmin = np.full((nr_uv_points_in_block), fill_value=np.nan, dtype=np.float32)
    uv_zmax = np.full((nr_uv_points_in_block), fill_value=np.nan, dtype=np.float32)
    uv_havg = np.full(
        (nr_levels, nr_uv_points_in_block), fill_value=np.nan, dtype=np.float32
    )
    uv_nrep = np.full(
        (nr_levels, nr_uv_points_in_block), fill_value=np.nan, dtype=np.float32
    )
    uv_pwet = np.full(
        (nr_levels, nr_uv_points_in_block), fill_value=np.nan, dtype=np.float32
    )
    uv_ffit = np.full((nr_uv_points_in_block), fill_value=np.nan, dtype=np.float32)
    uv_navg = np.full((nr_uv_points_in_block), fill_value=np.nan, dtype=np.float32)

    for ip in range(nr_uv_points_in_block):
        # Get the pixel indices for this uv point

        # Now build the pixel matrix
        if uv_flags_type[ip] <= 0:
            # Normal point or fine to coarse

            if uv_flags_dir[ip] == 0:
                # x
                nn = (n[ip] - bn0) * refi
                mm = (m[ip] - bm0) * refi + int(0.5 * refi)
            else:
                # y
                nn = (n[ip] - bn0) * refi + int(0.5 * refi)
                mm = (m[ip] - bm0) * refi

        else:  # uv_flags_type[ip] == 1
            # Coarse to fine

            if uv_flags_dir[ip] == 0:
                # x
                nn = (n_nmu[ip] - bn0) * refi
                mm = (m_nmu[ip] - bm0) * refi - int(0.5 * refi)
            else:
                # y
                nn = (n_nmu[ip] - bn0) * refi - int(0.5 * refi)
                mm = (m_nmu[ip] - bm0) * refi

        # Pixel block actually starts half a (or one?) grid cell to the left and below,
        # so need to add 0.5*refi
        nn += int(0.5 * refi)
        mm += int(0.5 * refi)

        # Pixel indices for this cell
        zg_uv = zg[nn : nn + refi, mm : mm + refi]
        manning_uv = manning[nn : nn + refi, mm : mm + refi]

        if uv_flags_dir[ip] == 0:
            zg_uv = np.transpose(zg_uv)
            manning_uv = np.transpose(manning_uv)

        zg_uv = zg_uv.flatten()
        manning_uv = manning_uv.flatten()

        zmin, zmax, havg, nrep, pwet, ffit, navg, zz = subgrid_q_table(
            zg_uv,
            manning_uv,
            nr_levels,
            huthresh,
            2,
            z_zmin_nm[ip],
            z_zmin_nmu[ip],
            weight_option,
        )

        uv_zmin[ip] = zmin
        uv_zmax[ip] = zmax
        uv_havg[:, ip] = havg
        uv_nrep[:, ip] = nrep
        uv_pwet[:, ip] = pwet
        uv_ffit[ip] = ffit
        uv_navg[ip] = navg

    return (
        uv_zmin,
        uv_zmax,
        uv_havg,
        uv_nrep,
        uv_pwet,
        uv_ffit,
        uv_navg,
    )


def log_info(msg, logger, quiet):
    """Log info message to logger and print to console"""
    if logger:
        logger.info(msg)
    if not quiet:
        print(msg)
