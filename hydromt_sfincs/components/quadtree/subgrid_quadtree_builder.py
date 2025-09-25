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
import matplotlib.path as path

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
    roughness_type="manning",
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
        roughness_type,
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
        roughness_type,
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

        # Determine what type of uv point it is # TODO - TL: we didn't have this in _insiders?
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

            # TODO - TL: here missing is "# Check if active SFINCS cells exist in mask"

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

                    if nr_cells_in_block == 0:
                        # No cells in this block
                        continue

                    # TODO - TL: here missing is "# Check if active SFINCS cells exist in mask"

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
                        # manning_grid = bathymetry_database.get_bathymetry_on_grid(
                        #     xg, yg, crs, roughness_sets
                        # )
                        # Loop through roughness sets, check if one has polygon file
                        manning_grid = bathymetry_database.get_bathymetry_on_grid(
                            xg, yg, crs, roughness_sets
                        )

                        for roughness_set in roughness_sets:
                            if (
                                "polygon_file" in roughness_set
                                and "value" in roughness_set
                            ):
                                polygon_file = roughness_set["polygon_file"]
                                # Read the polygon file and get the values
                                gdf = gpd.read_file(polygon_file)
                                value = roughness_set["value"]

                                # Loop through polygons in gdf
                                inpols = np.full(xg.shape, False)
                                for ip, polygon in gdf.iterrows():
                                    inpol = inpolygon(xg, yg, polygon["geometry"])
                                    inpols = np.logical_or(inpols, inpol)

                                manning_grid[inpols] = value
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
                        roughness_type,
                    )

                    if progress_bar:
                        progress_bar.set_value(ibt)
                        if progress_bar.was_canceled():
                            return
                        ibt += 1

        # Now create the xarray dataset (FIXME do we transpose here? is this necessary for fortan?)
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
    roughness_type,  # roughness type (manning or chezy)
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
            roughness_type,
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
    rgh: np.ndarray,
    nr_levels: int,
    huthresh: float,
    option: int = 2,
    z_zmin_a: float = -99999.0,
    z_zmin_b: float = -99999.0,
    weight_option: str = "min",
    roughness_type: str = "manning",
):
    """
    map vector of elevation values into a hypsometric hydraulic radius - depth relationship for one u/v point
    Parameters
    ----------
    elevation : np.ndarray (nr of pixels in one cell) containing subgrid elevation values for one grid cell [m]
    rgh : np.ndarray (nr of pixels in one cell) containing subgrid roughness values for one grid cell [s m^(-1/3)]
    nr_levels : int, number of vertical levels [-]
    huthresh : float, threshold depth [m]
    option : int, option to use "old" or "new" method for computing conveyance depth at u/v points
    z_zmin_a : float, elevation of lowest pixel in neighboring cell A [m]
    z_zmin_b : float, elevation of lowest pixel in neighboring cell B [m]
    weight_option : str, weight of q between sides A and B ("min" or "mean")
    roughness_type : str, "manning" or "chezy"

    Returns
    -------
    zmin : float, minimum elevation [m]
    zmax : float, maximum elevation [m]
    havg : np.ndarray (nr_levels) grid-average depth for vertical levels [m]
    nrep : np.ndarray (nr_levels) representative roughness for vertical levels [m1/3/s] ?
    pwet : np.ndarray (nr_levels) wet fraction for vertical levels [-] ?
    navg : float, grid-average Manning's n [m 1/3 / s]
    ffit : float, fitting coefficient [-]
    zz   : np.ndarray (nr_levels) elevation of vertical levels [m]
    """
    # Initialize output arrays
    havg = np.zeros(nr_levels)
    nrep = np.zeros(nr_levels)
    pwet = np.zeros(nr_levels)
    zz = np.zeros(nr_levels)

    n = int(np.size(elevation))  # Nr of pixels in grid cell
    # print(f"n = {n}")
    # print(f"n05 = {int(n/2)}")
    n05 = int(n / 2)  # Nr of pixels in half grid cell
    # print(f"n05 = {n05}")

    # Sort elevation and manning values by side A and B
    dd_a = elevation[0:n05]
    dd_b = elevation[n05:]
    rgh_a = rgh[0:n05]
    rgh_b = rgh[n05:]

    # Ensure that pixels are at least as high as the minimum elevation in the neighbouring cells
    # This should always be the case, but there may be errors in the interpolation to the subgrid pixels
    dd_a = np.maximum(dd_a, z_zmin_a)
    dd_b = np.maximum(dd_b, z_zmin_b)

    # Determine min and max elevation
    zmin_a = np.min(dd_a)
    zmax_a = np.max(dd_a)
    zmin_b = np.min(dd_b)
    zmax_b = np.max(dd_b)

    # Add huthresh to zmin
    zmin = max(zmin_a, zmin_b) + huthresh
    zmax = float(max(zmax_a, zmax_b))

    # Make sure zmax is at least 0.01 m higher than zmin
    zmax = max(zmax, zmin + 0.01)

    # Determine bin size
    # print("nr_levels -1= ", nr_levels -1)
    dlevel = (zmax - zmin) / (nr_levels - 1)
    # print("dlevel = ", dlevel)

    # Option can be either 1 ("old") or 2 ("new")
    # Should never use option 1 !
    option = 2

    # Loop through levels
    for ibin in range(nr_levels):
        # Top of bin
        zbin = zmin + ibin * dlevel
        zz[ibin] = zbin

        h = np.maximum(zbin - elevation, 0.0)  # water depth in each pixel

        h_a = np.maximum(
            zbin - dd_a, 0.0
        )  # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).
        h_b = np.maximum(
            zbin - dd_b, 0.0
        )  # Depth of all pixels (but set min pixel height to zbot). Can be negative, but not zero (because zmin = zbot + huthresh, so there must be pixels below zb).

        # # print min of rgh_a, rgh_b, rgh
        # print("rgh_a = ", np.min(rgh_a))
        # print("rgh_b = ", np.min(rgh_b))
        # print("rgh = ", np.min(rgh))

        if roughness_type == "manning":
            manning_a = rgh_a
            manning_b = rgh_b
            manning = rgh
        elif roughness_type == "chezy":
            manning_a = (1.0 / rgh_a) * h_a ** (1.0 / 6.0)
            manning_b = (1.0 / rgh_b) * h_b ** (1.0 / 6.0)
            manning = (1.0 / rgh) * h ** (1.0 / 6.0)
            manning_a = np.maximum(
                manning_a, 0.001
            )  # Set minimum value to avoid division by zero
            manning_b = np.maximum(
                manning_b, 0.001
            )  # Set minimum value to avoid division by zero
            manning = np.maximum(
                manning, 0.001
            )  # Set minimum value to avoid division by zero

        # print("manning_a = ", np.min(manning_a))
        # print("manning_b = ", np.min(manning_b))
        # print("manning = ", np.min(manning))

        # Side A
        q_a = h_a ** (5.0 / 3.0) / manning_a  # Determine 'flux' for each pixel
        q_a = np.mean(q_a)  # Grid-average flux through all the pixels
        h_a = np.mean(h_a)  # Grid-average depth through all the pixels

        # Side B
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
                nr_levels - 1
            )  # Weight (increase from 0 to 1 from bottom to top bin)
            q = (1.0 - w) * q_min + w * q_all  # Weighted average of q_min and q_all
            hmean = h_all
            # Wet fraction
            pwet[ibin] = (zbin > elevation + huthresh).sum() / n

        elif option == 2:
            # Use newer 2 option (minimum of q_a an q_b, minimum of h_a and h_b increasing to h_all, using pwet for weighting) option
            # This is done by making sure that the wet fraction is 0.0 in the first level on the shallowest side (i.e. if ibin==0, pwet_a or pwet_b must be 0.0).
            # As a result, the weight w will be 0.0 in the first level on the shallowest side.

            pwet_a = (zbin > dd_a).sum() / int(n / 2)
            pwet_b = (zbin > dd_b).sum() / int(n / 2)

            if ibin == 0:
                # Ensure that at bottom level, either pwet_a or pwet_b is 0.0
                if pwet_a < pwet_b:
                    pwet_a = 0.0
                else:
                    pwet_b = 0.0
            elif ibin == nr_levels - 1:
                # Ensure that at top level, both pwet_a and pwet_b are 1.0
                pwet_a = 1.0
                pwet_b = 1.0

            if weight_option == "mean":
                # Weight increases linearly from 0 to 1 from bottom to top bin use percentage wet in sides A and B
                w = 2 * np.minimum(pwet_a, pwet_b) / max(pwet_a + pwet_b, 1.0e-9)
                q = (1.0 - w) * q_min + w * q_all  # Weighted average of q_min and q_all
                hmean = (
                    1.0 - w
                ) * h_min + w * h_all  # Weighted average of h_min and h_all

            else:
                # Take minimum of q_a and q_b
                if q_a < q_b:
                    q = q_a
                    hmean = h_a
                else:
                    q = q_b
                    hmean = h_b

            pwet[ibin] = 0.5 * (pwet_a + pwet_b)  # Combined pwet_a and pwet_b

        havg[ibin] = hmean  # conveyance depth
        # print(f"q = {q}")
        nrep[ibin] = hmean ** (5.0 / 3.0) / q  # Representative n for qmean and hmean

    nrep_top = nrep[-1]
    havg_top = havg[-1]

    ### Fitting for nrep above zmax

    # Determine nfit at zfit
    zfit = float(zmax + zmax - zmin)
    hfit = (
        havg_top + zmax - zmin
    )  # mean water depth in cell as computed in SFINCS (assuming linear relation between water level and water depth above zmax)

    # Compute q and navg
    if weight_option == "mean":
        # Use entire uv point
        h = np.maximum(zfit - elevation, 0.0)  # water depth in each pixel
        q = np.mean(h ** (5.0 / 3.0) / manning)  # combined unit discharge for cell
        navg = np.mean(manning)

    else:
        if roughness_type == "manning":
            manning_a = rgh_a
            manning_b = rgh_b
            manning = rgh
        elif roughness_type == "chezy":
            manning_a = (1.0 / rgh_a) * h_a ** (1.0 / 6.0)
            manning_b = (1.0 / rgh_b) * h_b ** (1.0 / 6.0)
            manning = (1.0 / rgh) * h ** (1.0 / 6.0)
            manning_a = np.maximum(
                manning_a, 0.001
            )  # Set minimum value to avoid division by zero
            manning_b = np.maximum(
                manning_b, 0.001
            )  # Set minimum value to avoid division by zero
            manning = np.maximum(
                manning, 0.001
            )  # Set minimum value to avoid division by zero

        # Use minimum of q_a and q_b
        if q_a < q_b:
            h = np.maximum(zfit - dd_a, 0.0)  # water depth in each pixel

            if roughness_type == "manning":
                manning_a = rgh_a
            elif roughness_type == "chezy":
                manning_a = (1.0 / rgh_a) * h ** (1.0 / 6.0)
                manning_a = np.maximum(
                    manning_a, 0.001
                )  # Set minimum value to avoid division by zero

            q = np.mean(
                h ** (5.0 / 3.0) / manning_a
            )  # combined unit discharge for cell
            navg = np.mean(manning_a)
        else:
            h = np.maximum(zfit - dd_b, 0.0)
            if roughness_type == "manning":
                manning_b = rgh_b
            elif roughness_type == "chezy":
                manning_b = (1.0 / rgh_b) * h ** (1.0 / 6.0)
                manning_b = np.maximum(
                    manning_b, 0.001
                )  # Set minimum value to avoid division by zero
            q = np.mean(h ** (5.0 / 3.0) / manning_b)
            navg = np.mean(manning_b)

    # print(f"qq = {q}")
    nfit = hfit ** (5.0 / 3.0) / q

    # Actually apply fit on gn2 (this is what is used in sfincs)
    gnavg2 = float(9.81 * navg**2)
    gnavg_top2 = float(9.81 * nrep_top**2)

    # print("almost done")
    # print(f"gnavg2 = {float(gnavg2)}")
    # print(f"gnavg_top2 = {float(gnavg_top2)}")
    # print(f"nrep_top = {float(nrep_top)}")
    # print(f"zfit - zmax = {float(zfit - zmax)}")

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
        gnfit2 = float(9.81 * nfit**2)
        zfit = max(zfit, zmax + 1.0e-6)
        gnavg2 = max(gnavg2, gnfit2 + 1.0e-8)
        ffit = (((gnavg2 - gnavg_top2) / (gnavg2 - gnfit2)) - 1) / (zfit - zmax)

    # print(f"gnavg2 - gnfit2 = {gnavg2 - gnfit2}")
    # print(f"zfit - zmax = {zfit - zmax}")
    # print(f"ffit = {ffit}")
    # print("done")

    return zmin, zmax, havg, nrep, pwet, ffit, navg, zz


def log_info(msg, logger, quiet):
    """Log info message to logger and print to console"""
    if logger:
        logger.info(msg)
    if not quiet:
        print(msg)


def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)
