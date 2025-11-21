import time
import warnings

import numpy as np
import xarray as xr
import xugrid as xu
from matplotlib import path
from pyproj import CRS
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon
from shapely.prepared import prep

from hydromt_sfincs import workflows

np.warnings = warnings


def build_quadtree_xugrid(
    x0,
    y0,
    nmax,
    mmax,
    dx,
    dy,
    rotation,
    crs,
    refinement_polygons=None,
    elevation_list=None,
    bathymetry_database=None,
):
    """
    Build a quadtree grid using the xugrid library.

    Parameters
    ----------
    x0, y0 : float
        Origin of the grid.
    nmax, mmax : int
        Number of cells in x and y direction.
    dx, dy : float
        Grid spacing in x and y direction.
    rotation : float
        Rotation of the grid in degrees.
    refinement_polygons : geopandas.GeoDataFrame
        GeoDataFrame with polygons that define the refinement levels.
    elevation_list : dict
        Dictionary with datasets dependencies.
    bathymetry_database : object
        Bathymetry database object.

    Returns
    -------
    xugrid.UgridDataArray
    """

    grid = QuadtreeGrid()

    grid.build(
        x0,
        y0,
        nmax,
        mmax,
        dx,
        dy,
        rotation,
        crs,
        refinement_polygons,
        elevation_list=elevation_list,
        bathymetry_database=bathymetry_database,
    )

    return grid.data


def cut_inactive_cells(data):
    """Cuts inactive cells (where mask == 0) from Quadtree xugrid dataset

    Parameters
    ----------
    data : xu.UgridDataset
        Quadtree xugrid dataset

    Returns
    -------
    xu.UgridDataset
    """
    grid = QuadtreeGrid(data=data)
    grid.cut_inactive_cells()
    return grid.data


class QuadtreeGrid:
    def __init__(self, data=None):
        self.data = data

    def build(
        self,
        x0,
        y0,
        nmax,
        mmax,
        dx,
        dy,
        rotation,
        crs,
        refinement_polygons,
        elevation_list,
        bathymetry_database,
    ):
        self.x0 = x0
        self.y0 = y0
        self.nmax = nmax
        self.mmax = mmax
        self.dx = dx
        self.dy = dy
        self.rotation = rotation
        self.cosrot = np.cos(rotation * np.pi / 180)
        self.sinrot = np.sin(rotation * np.pi / 180)
        self.crs = crs

        self.nr_cells = 0
        self.nr_refinement_levels = 1
        self.version = 0

        self.refinement_polygons = refinement_polygons
        self.elevation_list = elevation_list if elevation_list is not None else []
        self.bathymetry_database = bathymetry_database

        # Make regular grid
        self.get_regular_grid()

        # Initialize data arrays
        self.initialize_data_arrays()

        # Refine all cells
        if self.refinement_polygons is not None:
            self.refine_mesh()

        # Initialize data arrays
        self.initialize_data_arrays()

        # Get all neighbor arrays (mu, mu1, mu2, nu, nu1, nu2)
        self.get_neighbors()

        # Get uv points
        self.get_uv_points()

        # Create xugrid dataset
        self.to_xugrid()

    def get_regular_grid(self):
        # Build initial grid with one level
        ns = np.linspace(0, self.nmax - 1, self.nmax, dtype=int)
        ms = np.linspace(0, self.mmax - 1, self.mmax, dtype=int)
        self.m, self.n = np.meshgrid(ms, ns)
        self.n = np.transpose(self.n).flatten()
        self.m = np.transpose(self.m).flatten()
        self.nr_cells = self.nmax * self.mmax
        self.level = np.zeros(self.nr_cells, dtype=int)
        self.nr_refinement_levels = 1
        # Determine ifirst and ilast for each level
        self.find_first_cells_in_level()
        # Compute cell center coordinates self.x and self.y
        self.compute_cell_center_coordinates()

    def initialize_data_arrays(self):
        # Set indices of neighbors to -1
        self.mu = np.zeros(self.nr_cells, dtype=np.int8)
        self.mu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.mu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md = np.zeros(self.nr_cells, dtype=np.int8)
        self.md1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu = np.zeros(self.nr_cells, dtype=np.int8)
        self.nu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd = np.zeros(self.nr_cells, dtype=np.int8)
        self.nd1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd2 = np.zeros(self.nr_cells, dtype=int) - 1

        # Set initial depth
        self.z = np.zeros(self.nr_cells, dtype=float)
        # Set initial SFINCS mask to zeros
        self.mask = np.zeros(self.nr_cells, dtype=np.int8)
        # Set initial SnapWave mask to zeros
        self.snapwave_mask = np.zeros(self.nr_cells, dtype=np.int8)

    def refine_mesh(self):
        # Loop through rows in gdf and create list of polygons
        # Determine maximum refinement level

        start = time.time()
        print("Refining ...")

        self.ref_pols = []
        for irow, row in self.refinement_polygons.iterrows():
            iref = row["refinement_level"]
            polygon = {"geometry": row["geometry"], "refinement_level": iref}
            if "zmin" in row:
                polygon["zmin"] = row["zmin"]
            else:
                polygon["zmin"] = -np.inf
            if "zmax" in row:
                polygon["zmax"] = row["zmax"]
            else:
                polygon["zmax"] = np.inf
            self.ref_pols.append(polygon)

        # Loop through refinement polygons and start refining
        for polygon in self.ref_pols:
            # Refine, reorder, find first cells in level
            self.refine_in_polygon(polygon)

        print("Time elapsed : " + str(time.time() - start) + " s")

    def refine_in_polygon(self, polygon):
        # Finds cell to refine and calls refine_cells

        # Loop through refinement levels for this polygon
        for ilev in range(polygon["refinement_level"]):
            # Refine cells in refinement polygons
            # Compute grid spacing for this level
            dx = self.dx / 2**ilev
            dy = self.dy / 2**ilev
            nmax = self.nmax * 2**ilev
            mmax = self.mmax * 2**ilev
            # Add buffer of 0.5*dx around polygon
            polbuf = polygon["geometry"]
            # Rotate polbuf to grid (this is needed to find cells that could fall within polbuf)
            coords = polbuf.exterior.coords[:]
            npoints = len(coords)
            polx = np.zeros(npoints)
            poly = np.zeros(npoints)

            for ipoint, point in enumerate(polbuf.exterior.coords[:]):
                # Cell centres
                polx[ipoint] = self.cosrot * (point[0] - self.x0) + self.sinrot * (
                    point[1] - self.y0
                )
                poly[ipoint] = -self.sinrot * (point[0] - self.x0) + self.cosrot * (
                    point[1] - self.y0
                )

            # Find cells cells in grid that could fall within polbuf
            n0 = int(np.floor(np.min(poly) / dy)) - 1
            n1 = int(np.ceil(np.max(poly) / dy)) + 1
            m0 = int(np.floor(np.min(polx) / dx)) - 1
            m1 = int(np.ceil(np.max(polx) / dx)) + 1

            n0 = min(max(n0, 0), nmax - 1)
            n1 = min(max(n1, 0), nmax - 1)
            m0 = min(max(m0, 0), mmax - 1)
            m1 = min(max(m1, 0), mmax - 1)

            # Generate grid (corners)
            nn, mm = np.meshgrid(np.arange(n0, n1 + 2), np.arange(m0, m1 + 2))
            xx = self.x0 + self.cosrot * mm * dx - self.sinrot * nn * dy
            yy = self.y0 + self.sinrot * mm * dx + self.cosrot * nn * dy
            in_polygon = grid_in_polygon(xx, yy, polygon["geometry"])
            in_polygon = np.transpose(in_polygon).flatten()

            nn, mm = np.meshgrid(np.arange(n0, n1 + 1), np.arange(m0, m1 + 1))
            nn = np.transpose(nn).flatten()
            mm = np.transpose(mm).flatten()

            # Indices of cells in level within polbuf
            nn_in = nn[in_polygon]
            mm_in = mm[in_polygon]
            nm_in = nmax * mm_in + nn_in

            # Find existing cells of this level in nmi array
            n_level = self.n[self.ifirst[ilev] : self.ilast[ilev] + 1]
            m_level = self.m[self.ifirst[ilev] : self.ilast[ilev] + 1]
            nm_level = m_level * nmax + n_level

            # Find indices all cells to be refined
            ind_ref = binary_search(nm_level, nm_in)

            ind_ref = ind_ref[ind_ref >= 0]

            # ind_ref = ind_ref[ind_ref < np.size(nm_level)]
            if not np.any(ind_ref):
                continue
            # Index of cells to refine
            ind_ref += self.ifirst[ilev]

            # But only where elevation is between zmin and zmax
            if len(self.elevation_list) > 0 and (
                polygon["zmin"] > -20000.0 or polygon["zmax"] < 20000.0
            ):
                # self.to_xugrid()
                # self.compute_cell_center_coordinates()
                if self.bathymetry_database is not None:
                    zmin, zmax = self.get_bathymetry_min_max(
                        ind_ref=ind_ref,
                        ilev=ilev,
                        elevation_list=self.elevation_list,
                        bathymetry_database=self.bathymetry_database,
                    )
                else:
                    zmin, zmax = self.get_bathymetry_min_max(
                        ind_ref=ind_ref,
                        ilev=ilev,
                        elevation_list=self.elevation_list[ilev],
                    )
                # z = self.data["z"][ind_ref]
                ind_ref = ind_ref[
                    np.logical_and(zmax > polygon["zmin"], zmin < polygon["zmax"])
                ]

            if not np.any(ind_ref):
                continue

            self.refine_cells(ind_ref, ilev)

    def refine_cells(self, ind_ref, ilev):
        # Refine cells with index ind_ref

        # First find lower-level neighbors (these will be refined in the next iteration)
        if ilev > 0:
            ind_nbr = self.find_lower_level_neighbors(ind_ref, ilev)
        else:
            ind_nbr = np.empty(0, dtype=int)

        # n and m indices of cells to be refined
        n = self.n[ind_ref]
        m = self.m[ind_ref]

        # New cells
        nnew = np.zeros(4 * len(ind_ref), dtype=int)
        mnew = np.zeros(4 * len(ind_ref), dtype=int)
        lnew = np.zeros(4 * len(ind_ref), dtype=int) + ilev + 1
        nnew[0::4] = n * 2  # lower left
        nnew[1::4] = n * 2 + 1  # upper left
        nnew[2::4] = n * 2  # lower right
        nnew[3::4] = n * 2 + 1  # upper right
        mnew[0::4] = m * 2  # lower left
        mnew[1::4] = m * 2  # upper left
        mnew[2::4] = m * 2 + 1  # lower right
        mnew[3::4] = m * 2 + 1  # upper right
        # Add new cells to grid
        self.n = np.append(self.n, nnew)
        self.m = np.append(self.m, mnew)
        self.level = np.append(self.level, lnew)
        # Remove old cells from grid
        self.n = np.delete(self.n, ind_ref)
        self.m = np.delete(self.m, ind_ref)
        self.level = np.delete(self.level, ind_ref)

        self.nr_cells = len(self.n)
        self.initialize_data_arrays()

        # Update nr_refinement_levels at max of ilev + 2 and self.nr_refinement_levels
        self.nr_refinement_levels = np.maximum(self.nr_refinement_levels, ilev + 2)
        # Reorder cells
        self.reorder()
        # Update ifirst and ilast
        self.find_first_cells_in_level()
        # Compute cell center coordinates self.x and self.y
        self.compute_cell_center_coordinates()

        if np.any(ind_nbr):
            self.refine_cells(ind_nbr, ilev - 1)

    def get_neighbors(self):
        # Get mu, mu1, mu2, nu, nu1, nu2 for all cells

        start = time.time()

        print("Finding neighbors ...")

        # Get nm indices for all cells
        nm_all = np.zeros(self.nr_cells, dtype=int)
        for ilev in range(self.nr_refinement_levels):
            nmax = self.nmax * 2**ilev + 1
            i0 = self.ifirst[ilev]
            i1 = self.ilast[ilev] + 1
            n = self.n[i0:i1]
            m = self.m[i0:i1]
            nm_all[i0:i1] = m * nmax + n

        # Loop over levels
        for ilev in range(self.nr_refinement_levels):
            nmax = self.nmax * 2**ilev + 1

            # First and last cell in this level
            i0 = self.ifirst[ilev]
            i1 = self.ilast[ilev] + 1

            # Initialize arrays for this level
            mu = np.zeros(i1 - i0, dtype=int)
            mu1 = np.zeros(i1 - i0, dtype=int) - 1
            mu2 = np.zeros(i1 - i0, dtype=int) - 1
            nu = np.zeros(i1 - i0, dtype=int)
            nu1 = np.zeros(i1 - i0, dtype=int) - 1
            nu2 = np.zeros(i1 - i0, dtype=int) - 1

            # Get n and m indices for this level
            n = self.n[i0:i1]
            m = self.m[i0:i1]
            nm = nm_all[i0:i1]

            # Now look for neighbors

            # Same level

            # Right
            nm_to_find = nm + nmax
            inb = binary_search(nm, nm_to_find)
            mu1[inb >= 0] = inb[inb >= 0] + i0

            # Above
            nm_to_find = nm + 1
            inb = binary_search(nm, nm_to_find)
            nu1[inb >= 0] = inb[inb >= 0] + i0

            ## Coarser level neighbors
            if ilev > 0:
                nmaxc = (
                    self.nmax * 2 ** (ilev - 1) + 1
                )  # Number of cells in coarser level in n direction

                i0c = self.ifirst[ilev - 1]  # First cell in coarser level
                i1c = self.ilast[ilev - 1] + 1  # Last cell in coarser level

                nmc = nm_all[i0c:i1c]  # Coarser level nm indices
                nc = n // 2  # Coarser level n index of this cells in this level
                mc = m // 2  # Coarser level m index of this cells in this level

                # Right
                nmc_to_find = (mc + 1) * nmaxc + nc
                inb = binary_search(nmc, nmc_to_find)
                inb[np.where(even(m))[0]] = -1
                # Set mu and mu1 for inb>=0
                mu1[inb >= 0] = inb[inb >= 0] + i0c
                mu[inb >= 0] = -1

                # Above
                nmc_to_find = mc * nmaxc + nc + 1
                inb = binary_search(nmc, nmc_to_find)
                inb[np.where(even(n))[0]] = -1
                # Set nu and nu1 for inb>=0
                nu1[inb >= 0] = inb[inb >= 0] + i0c
                nu[inb >= 0] = -1

            # Finer level neighbors
            if ilev < self.nr_refinement_levels - 1:
                nmaxf = (
                    self.nmax * 2 ** (ilev + 1) + 1
                )  # Number of cells in finer level in n direction

                i0f = self.ifirst[ilev + 1]  # First cell in finer level
                i1f = self.ilast[ilev + 1] + 1  # Last cell in finer level
                nmf = nm_all[i0f:i1f]  # Finer level nm indices

                # Right

                # Lower row
                nf = n * 2  # Finer level n index of this cells in this level
                mf = m * 2 + 1  # Finer level m index of this cells in this level
                nmf_to_find = (mf + 1) * nmaxf + nf
                inb = binary_search(nmf, nmf_to_find)
                mu1[inb >= 0] = inb[inb >= 0] + i0f
                mu[inb >= 0] = 1

                # Upper row
                nf = n * 2 + 1  # Finer level n index of this cells in this level
                mf = m * 2 + 1  # Finer level m index of this cells in this level
                nmf_to_find = (mf + 1) * nmaxf + nf
                inb = binary_search(nmf, nmf_to_find)
                mu2[inb >= 0] = inb[inb >= 0] + i0f
                mu[inb >= 0] = 1

                # Above

                # Left column
                nf = n * 2 + 1  # Finer level n index of this cells in this level
                mf = m * 2  # Finer level m index of this cells in this level
                nmf_to_find = mf * nmaxf + nf + 1
                inb = binary_search(nmf, nmf_to_find)
                nu1[inb >= 0] = inb[inb >= 0] + i0f
                nu[inb >= 0] = 1

                # Right column
                nf = n * 2 + 1  # Finer level n index of this cells in this level
                mf = m * 2 + 1  # Finer level m index of this cells in this level
                nmf_to_find = mf * nmaxf + nf + 1
                inb = binary_search(nmf, nmf_to_find)
                nu2[inb >= 0] = inb[inb >= 0] + i0f
                nu[inb >= 0] = 1

            # Fill in mu, mu1, mu2, nu, nu1, nu2 for this level
            self.mu[i0:i1] = mu
            self.mu1[i0:i1] = mu1
            self.mu2[i0:i1] = mu2
            self.nu[i0:i1] = nu
            self.nu1[i0:i1] = nu1
            self.nu2[i0:i1] = nu2

        print("Time elapsed : " + str(time.time() - start) + " s")

        # Making global model
        # Check if CRS is geographic
        if self.crs.is_geographic:
            # Now check if mmax * dx is 360
            if self.mmax * self.dx > 359 and self.mmax * self.dx < 361:
                # We have a global model
                # Loop through all points
                for ilev in range(self.nr_refinement_levels):
                    i0 = self.ifirst[ilev]
                    i1 = self.ilast[ilev] + 1
                    nmaxf = self.nmax * 2**ilev + 1
                    mf = self.mmax * 2**ilev - 1
                    nmf = nm_all[i0:i1]
                    # Loop through all cells
                    for i in range(i0, i1):
                        if self.m[i] > 0:
                            # This cell is not on the left of the model
                            break
                        # Now find matching cell on the right
                        # nm index of cell on RHS of grid
                        nmf_to_find = mf * nmaxf + self.n[i]
                        iright = np.where(nmf == nmf_to_find)[0]
                        if iright.size > 0:
                            iright = iright + i0
                            self.mu[iright] = 0
                            self.mu1[iright] = i
                            self.mu2[iright] = -1

        print("Setting neighbors left and below ...")

        # Right

        iok1 = np.where(self.mu1 >= 0)[0]
        # Same level
        iok2 = np.where(self.mu == 0)[0]
        # Indices of cells that have a same level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Indices of neighbors
        imu = self.mu1[iok]
        self.md[imu] = 0
        self.md1[imu] = iok

        # Coarser
        iok2 = np.where(self.mu == -1)[0]
        # Indices of cells that have a coarse level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Odd
        iok_odd = iok[np.where(odd(self.n[iok]))]
        iok_even = iok[np.where(even(self.n[iok]))]
        imu = self.mu1[iok_odd]
        self.md[imu] = 1
        self.md1[imu] = iok_odd
        imu = self.mu1[iok_even]
        self.md[imu] = 1
        self.md2[imu] = iok_even

        # Finer
        # Lower
        iok1 = np.where(self.mu1 >= 0)[0]
        # Same level
        iok2 = np.where(self.mu == 1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        imu = self.mu1[iok]
        self.md[imu] = -1
        self.md1[imu] = iok
        # Upper
        iok1 = np.where(self.mu2 >= 0)[0]
        # Same level
        iok2 = np.where(self.mu == 1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        imu = self.mu2[iok]
        self.md[imu] = -1
        self.md1[imu] = iok

        # Above
        iok1 = np.where(self.nu1 >= 0)[0]
        # Same level
        iok2 = np.where(self.nu == 0)[0]
        # Indices of cells that have a same level neighbor above
        iok = np.intersect1d(iok1, iok2)
        # Indices of neighbors
        inu = self.nu1[iok]
        self.nd[inu] = 0
        self.nd1[inu] = iok

        # Coarser
        iok2 = np.where(self.nu == -1)[0]
        # Indices of cells that have a coarse level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Odd
        iok_odd = iok[np.where(odd(self.m[iok]))]
        iok_even = iok[np.where(even(self.m[iok]))]
        inu = self.nu1[iok_odd]
        self.nd[inu] = 1
        self.nd1[inu] = iok_odd
        inu = self.nu1[iok_even]
        self.nd[inu] = 1
        self.nd2[inu] = iok_even

        # Finer
        # Left
        iok1 = np.where(self.nu1 >= 0)[0]
        # Same level
        iok2 = np.where(self.nu == 1)[0]
        # Indices of cells that have finer level neighbor above
        iok = np.intersect1d(iok1, iok2)
        inu = self.nu1[iok]
        self.nd[inu] = -1
        self.nd1[inu] = iok
        # Upper
        iok1 = np.where(self.nu2 >= 0)[0]
        # Same level
        iok2 = np.where(self.nu == 1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        inu = self.nu2[iok]
        self.nd[inu] = -1
        self.nd1[inu] = iok

        print("Time elapsed : " + str(time.time() - start) + " s")

    def get_uv_points(self):
        start = time.time()
        print("Getting uv points ...")

        # Get uv points (do we actually need to do this?)
        self.uv_index_z_nm = np.zeros((self.nr_cells * 4), dtype=int)
        self.uv_index_z_nmu = np.zeros((self.nr_cells * 4), dtype=int)
        self.uv_dir = np.zeros((self.nr_cells * 4), dtype=int)
        # Loop through points (FIXME SHOULD TRY TO VECTORIZE THIS, but try to keep same order of uv points
        nuv = 0
        for ip in range(self.nr_cells):
            if self.mu1[ip] >= 0:
                self.uv_index_z_nm[nuv] = ip
                self.uv_index_z_nmu[nuv] = self.mu1[ip]
                self.uv_dir[nuv] = 0
                nuv += 1
            if self.mu2[ip] >= 0:
                self.uv_index_z_nm[nuv] = ip
                self.uv_index_z_nmu[nuv] = self.mu2[ip]
                self.uv_dir[nuv] = 0
                nuv += 1
            if self.nu1[ip] >= 0:
                self.uv_index_z_nm[nuv] = ip
                self.uv_index_z_nmu[nuv] = self.nu1[ip]
                self.uv_dir[nuv] = 1
                nuv += 1
            if self.nu2[ip] >= 0:
                self.uv_index_z_nm[nuv] = ip
                self.uv_index_z_nmu[nuv] = self.nu2[ip]
                self.uv_dir[nuv] = 1
                nuv += 1
        self.uv_index_z_nm = self.uv_index_z_nm[0:nuv]
        self.uv_index_z_nmu = self.uv_index_z_nmu[0:nuv]
        self.uv_dir = self.uv_dir[0:nuv]
        self.nr_uv_points = nuv

        print("Time elapsed : " + str(time.time() - start) + " s")

    def reorder(self):
        # Reorder cells
        # Sort cells by level, then m, then n
        i = np.lexsort((self.n, self.m, self.level))
        self.n = self.n[i]
        self.m = self.m[i]
        self.level = self.level[i]

    def find_first_cells_in_level(self):
        # Find first cell in each level
        self.ifirst = np.zeros(self.nr_refinement_levels, dtype=int)
        self.ilast = np.zeros(self.nr_refinement_levels, dtype=int)
        for ilev in range(0, self.nr_refinement_levels):
            # Find index of first cell with this level
            ifirst = np.where(self.level == ilev)[0]
            if ifirst.size == 0:
                self.ifirst[ilev] = -1
            else:
                self.ifirst[ilev] = ifirst[0]
            # self.ifirst[ilev] = np.where(self.level == ilev)[0][0]
            # Find index of last cell with this level
            if ilev < self.nr_refinement_levels - 1:
                # self.ilast[ilev] = np.where(self.level == ilev + 1)[0][0] - 1
                self.ilast[ilev] = np.where(self.level > ilev)[0][0] - 1
            else:
                self.ilast[ilev] = self.nr_cells - 1

    def find_lower_level_neighbors(self, ind_ref, ilev):
        # ind_ref are the indices of the cells that need to be refined

        n = self.n[ind_ref]
        m = self.m[ind_ref]

        n_odd = np.where(odd(n))
        m_odd = np.where(odd(m))
        n_even = np.where(even(n))
        m_even = np.where(even(m))

        ill = np.intersect1d(n_even, m_even)
        iul = np.intersect1d(n_odd, m_even)
        ilr = np.intersect1d(n_even, m_odd)
        iur = np.intersect1d(n_odd, m_odd)

        n_nbr = np.zeros((2, np.size(n)), dtype=int)
        m_nbr = np.zeros((2, np.size(n)), dtype=int)

        # LL
        n0 = np.int32(n[ill] / 2)
        m0 = np.int32(m[ill] / 2)
        n_nbr[0, ill] = n0 - 1
        m_nbr[0, ill] = m0
        n_nbr[1, ill] = n0
        m_nbr[1, ill] = m0 - 1
        # UL
        n0 = np.int32((n[iul] - 1) / 2)
        m0 = np.int32(m[iul] / 2)
        n_nbr[0, iul] = n0 + 1
        m_nbr[0, iul] = m0
        n_nbr[1, iul] = n0
        m_nbr[1, iul] = m0 - 1
        # LR
        n0 = np.int32(n[ilr] / 2)
        m0 = np.int32((m[ilr] - 1) / 2)
        n_nbr[0, ilr] = n0 - 1
        m_nbr[0, ilr] = m0
        n_nbr[1, ilr] = n0
        m_nbr[1, ilr] = m0 + 1
        # UR
        n0 = np.int32((n[iur] - 1) / 2)
        m0 = np.int32((m[iur] - 1) / 2)
        n_nbr[0, iur] = n0 + 1
        m_nbr[0, iur] = m0
        n_nbr[1, iur] = n0
        m_nbr[1, iur] = m0 + 1

        nmax = self.nmax * 2 ** (ilev - 1) + 1

        n_nbr = n_nbr.flatten()
        m_nbr = m_nbr.flatten()
        nm_nbr = m_nbr * nmax + n_nbr
        nm_nbr = np.sort(np.unique(nm_nbr, return_index=False))

        # Actual cells in the coarser level
        n_level = self.n[self.ifirst[ilev - 1] : self.ilast[ilev - 1] + 1]
        m_level = self.m[self.ifirst[ilev - 1] : self.ilast[ilev - 1] + 1]
        nm_level = m_level * nmax + n_level

        # Find
        ind_nbr = binary_search(nm_level, nm_nbr)
        ind_nbr = ind_nbr[ind_nbr >= 0]

        if np.any(ind_nbr):
            ind_nbr += self.ifirst[ilev - 1]

        return ind_nbr

    def compute_cell_center_coordinates(self):
        # Compute cell center coordinates
        # Loop through refinement levels
        dx = self.dx / 2**self.level
        dy = self.dy / 2**self.level
        self.x = (
            self.x0
            + self.cosrot * (self.m + 0.5) * dx
            - self.sinrot * (self.n + 0.5) * dy
        )
        self.y = (
            self.y0
            + self.sinrot * (self.m + 0.5) * dx
            + self.cosrot * (self.n + 0.5) * dy
        )

    def get_ugrid2d(self):
        tic = time.perf_counter()

        n = self.n
        m = self.m
        level = self.level

        nmax = self.nmax * 2 ** (self.nr_refinement_levels - 1) + 1

        face_nodes_n = np.full((8, self.nr_cells), -1, dtype=int)
        face_nodes_m = np.full((8, self.nr_cells), -1, dtype=int)
        face_nodes_nm = np.full((8, self.nr_cells), -1, dtype=int)

        # Highest refinement level
        ifac = 2 ** (self.nr_refinement_levels - level - 1)
        dxf = self.dx / 2 ** (self.nr_refinement_levels - 1)
        dyf = self.dy / 2 ** (self.nr_refinement_levels - 1)

        face_n = n * ifac
        face_m = m * ifac

        # First do the 4 corner points
        face_nodes_n[0, :] = face_n
        face_nodes_m[0, :] = face_m
        face_nodes_n[2, :] = face_n
        face_nodes_m[2, :] = face_m + ifac
        face_nodes_n[4, :] = face_n + ifac
        face_nodes_m[4, :] = face_m + ifac
        face_nodes_n[6, :] = face_n + ifac
        face_nodes_m[6, :] = face_m

        # Find cells with refinement below
        i = np.where(self.nd == 1)
        face_nodes_n[1, i] = face_n[i]
        face_nodes_m[1, i] = face_m[i] + ifac[i] / 2
        # Find cells with refinement to the right
        i = np.where(self.mu == 1)
        face_nodes_n[3, i] = face_n[i] + ifac[i] / 2
        face_nodes_m[3, i] = face_m[i] + ifac[i]
        # Find cells with refinement above
        i = np.where(self.nu == 1)
        face_nodes_n[5, i] = face_n[i] + ifac[i]
        face_nodes_m[5, i] = face_m[i] + ifac[i] / 2
        # Find cells with refinement to the left
        i = np.where(self.md == 1)
        face_nodes_n[7, i] = face_n[i] + ifac[i] / 2
        face_nodes_m[7, i] = face_m[i]

        # Flatten
        face_nodes_n = face_nodes_n.transpose().flatten()
        face_nodes_m = face_nodes_m.transpose().flatten()

        # Compute nm value of nodes
        face_nodes_nm = nmax * face_nodes_m + face_nodes_n
        nopoint = max(face_nodes_nm) + 1
        # Set missing points to very high number
        face_nodes_nm[np.where(face_nodes_n == -1)] = nopoint

        # Get the unique nm values
        xxx, index, irev = np.unique(
            face_nodes_nm, return_index=True, return_inverse=True
        )
        j = np.where(xxx == nopoint)[0][0]  # Index of very high number
        # irev2 = np.reshape(irev, (self.nr_cells, 8))
        # face_nodes_all = irev2.transpose()
        face_nodes_all = np.reshape(irev, (self.nr_cells, 8)).transpose()
        face_nodes_all[np.where(face_nodes_all == j)] = -1

        face_nodes = np.full(
            face_nodes_all.shape, -1
        )  # Create a new array filled with -1
        for i in range(face_nodes.shape[1]):
            idx = np.where(face_nodes_all[:, i] != -1)[0]
            face_nodes[: len(idx), i] = face_nodes_all[idx, i]

        # Now get rid of all the rows where all values are -1
        # Create a mask where each row is True if not all elements in the row are -1
        mask = (face_nodes != -1).any(axis=1)

        # Use this mask to index face_nodes
        face_nodes = face_nodes[mask]

        node_n = face_nodes_n[index[:j]]
        node_m = face_nodes_m[index[:j]]
        node_x = self.x0 + self.cosrot * (node_m * dxf) - self.sinrot * (node_n * dyf)
        node_y = self.y0 + self.sinrot * (node_m * dxf) + self.cosrot * (node_n * dyf)

        toc = time.perf_counter()

        print(f"Got rid of duplicates in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()

        nodes = np.transpose(np.vstack((node_x, node_y)))
        faces = np.transpose(face_nodes)
        fill_value = -1

        ugrid2d = xu.Ugrid2d(nodes[:, 0], nodes[:, 1], fill_value, faces)

        ugrid2d.set_crs(self.crs)

        # Set datashader df to None
        self.df = None

        toc = time.perf_counter()

        print(f"Made XUGrid in {toc - tic:0.4f} seconds")

        return ugrid2d

    def to_xugrid(self):
        print("Making XUGrid ...")

        # Create the grid
        ugrid2d = self.get_ugrid2d()

        # Create the dataset
        self.data = xu.UgridDataset(grids=ugrid2d)

        # Add attributes
        attrs = {
            "x0": self.x0,
            "y0": self.y0,
            "nmax": self.nmax,
            "mmax": self.mmax,
            "dx": self.dx,
            "dy": self.dy,
            "rotation": self.rotation,
            "nr_levels": self.nr_refinement_levels,
        }
        self.data.attrs = attrs

        # Now add the data arrays
        self.data["crs"] = self.crs.to_epsg()
        self.data["crs"].attrs = self.crs.to_cf()
        self.data["level"] = xu.UgridDataArray(
            xr.DataArray(data=self.level + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["z"] = xu.UgridDataArray(
            xr.DataArray(data=self.z, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["mask"] = xu.UgridDataArray(
            xr.DataArray(data=self.mask, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["snapwave_mask"] = xu.UgridDataArray(
            xr.DataArray(data=self.snapwave_mask, dims=[ugrid2d.face_dimension]),
            ugrid2d,
        )

        self.data["n"] = xu.UgridDataArray(
            xr.DataArray(data=self.n + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["m"] = xu.UgridDataArray(
            xr.DataArray(data=self.m + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )

        self.data["mu"] = xu.UgridDataArray(
            xr.DataArray(data=self.mu, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["mu1"] = xu.UgridDataArray(
            xr.DataArray(data=self.mu1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["mu2"] = xu.UgridDataArray(
            xr.DataArray(data=self.mu2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["md"] = xu.UgridDataArray(
            xr.DataArray(data=self.md, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["md1"] = xu.UgridDataArray(
            xr.DataArray(data=self.md1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["md2"] = xu.UgridDataArray(
            xr.DataArray(data=self.md2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )

        self.data["nu"] = xu.UgridDataArray(
            xr.DataArray(data=self.nu, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["nu1"] = xu.UgridDataArray(
            xr.DataArray(data=self.nu1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["nu2"] = xu.UgridDataArray(
            xr.DataArray(data=self.nu2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["nd"] = xu.UgridDataArray(
            xr.DataArray(data=self.nd, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["nd1"] = xu.UgridDataArray(
            xr.DataArray(data=self.nd1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )
        self.data["nd2"] = xu.UgridDataArray(
            xr.DataArray(data=self.nd2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d
        )

    def get_bathymetry_min_max(
        self, ind_ref, ilev, elevation_list, bathymetry_database=None, quiet=True
    ):
        """ "Used to determine min and max bathymetry of a cell (used for refinement)"""

        if not quiet:
            print("Getting bathymetry data ...")

        dx = self.dx / 2**ilev
        dy = self.dy / 2**ilev

        if bathymetry_database is not None:
            xz = (
                self.x0
                + self.cosrot * (self.m[ind_ref] + 0.5) * dx
                - self.sinrot * (self.n[ind_ref] + 0.5) * dy
            )
            yz = (
                self.y0
                + self.sinrot * (self.m[ind_ref] + 0.5) * dx
                + self.cosrot * (self.n[ind_ref] + 0.5) * dy
            )

            # Compute the four corner coordinates of the cell, given that the cosine of the rotation is cosrot and the sine is sinrot and the cell center is xz, yz
            xcor = np.zeros((4, np.size(xz)))
            ycor = np.zeros((4, np.size(xz)))
            xcor[0, :] = xz - 0.5 * self.cosrot * dx - 0.5 * self.sinrot * dy
            ycor[0, :] = yz - 0.5 * self.sinrot * dx + 0.5 * self.cosrot * dy
            xcor[1, :] = xz + 0.5 * self.cosrot * dx - 0.5 * self.sinrot * dy
            ycor[1, :] = yz + 0.5 * self.sinrot * dx + 0.5 * self.cosrot * dy
            xcor[2, :] = xz + 0.5 * self.cosrot * dx + 0.5 * self.sinrot * dy
            ycor[2, :] = yz + 0.5 * self.sinrot * dx - 0.5 * self.cosrot * dy
            xcor[3, :] = xz - 0.5 * self.cosrot * dx + 0.5 * self.sinrot * dy
            ycor[3, :] = yz - 0.5 * self.sinrot * dx - 0.5 * self.cosrot * dy

            if self.crs.is_geographic:
                dx = dx * 111000.0

            # Now loop through the 4 corners and get the minimum and maximum bathymetry
            for i in range(4):
                zgl = bathymetry_database.get_bathymetry_on_points(
                    xcor[i, :], ycor[i, :], dx, self.crs, elevation_list
                )
                if i == 0:
                    zmin = zgl
                    zmax = zgl
                else:
                    zmin = np.minimum(zmin, zgl)
                    zmax = np.maximum(zmax, zgl)
            return zmin, zmax
        else:
            # Build DataArray covering all corners
            m = self.m[ind_ref]
            n = self.n[ind_ref]
            corner_offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

            m_corners = (m[:, None] + corner_offsets[:, 0]).T
            n_corners = (n[:, None] + corner_offsets[:, 1]).T

            m_all = np.arange(m_corners.ravel().min(), m_corners.ravel().max() + 1)
            n_all = np.arange(n_corners.ravel().min(), n_corners.ravel().max() + 1)

            M_all, N_all = np.meshgrid(m_all, n_all)

            # Compute rotated coordinates for all corners in full grid
            x_all = self.x0 + self.cosrot * M_all * dx - self.sinrot * N_all * dy
            y_all = self.y0 + self.sinrot * M_all * dx + self.cosrot * N_all * dy

            # Create the DataArray
            da_corners = xr.DataArray(
                np.full(M_all.shape, np.nan),
                coords={
                    "n": ("y", n_all),
                    "m": ("x", m_all),
                    "xc": (("y", "x"), x_all),
                    "yc": (("y", "x"), y_all),
                },
                dims=("y", "x"),
            )
            da_corners.raster.set_crs(self.crs)

            # Fill da_corners with topobathy data
            da_dep = workflows.merge_multi_dataarrays(
                da_list=elevation_list,
                da_like=da_corners,
                interp_method="linear",
                buffer_cells=0,
            )

            # Flatten the corner indices (4 x ind_ref)
            n_flat = n_corners.flatten()
            m_flat = m_corners.flatten()

            # Find integer indices along the coordinate arrays
            idx_y = np.searchsorted(da_dep.n.values, n_flat)
            idx_x = np.searchsorted(da_dep.m.values, m_flat)

            # Select the values
            z_corners = da_dep.values[idx_y, idx_x]

            # Reshape back to (4, ind_ref)
            z_corners = z_corners.reshape(n_corners.shape)

            # Compute zmin and zmax per cell
            zmin = np.min(z_corners, axis=0)
            zmax = np.max(z_corners, axis=0)

            return zmin, zmax

    def cut_inactive_cells(self):
        print("Removing inactive cells ...")

        # Set crs
        self.crs = CRS.from_epsg(self.data["crs"].values)

        # In the xugrid data, the indices are 1-based, so we need to subtract 1
        n = self.data["n"].values[:] - 1
        m = self.data["m"].values[:] - 1
        level = self.data["level"].values[:] - 1
        z = self.data["z"].values[:]
        mask = self.data["mask"].values[:]
        swmask = self.data["snapwave_mask"].values[:]

        indx = np.where((mask + swmask) > 0)

        self.nr_refinement_levels = self.data.attrs["nr_levels"]
        self.nmax = self.data.attrs["nmax"]
        self.mmax = self.data.attrs["mmax"]
        self.x0 = self.data.attrs["x0"]
        self.y0 = self.data.attrs["y0"]
        self.rotation = self.data.attrs["rotation"]
        self.cosrot = np.cos(np.radians(self.rotation))
        self.sinrot = np.sin(np.radians(self.rotation))
        self.dx = self.data.attrs["dx"]
        self.dy = self.data.attrs["dy"]
        self.nr_cells = np.size(indx)
        self.n = n[indx]
        self.m = m[indx]
        self.level = level[indx]
        self.z = z[indx]
        self.mask = mask[indx]
        self.snapwave_mask = swmask[indx]

        # Set indices of neighbors to -1
        self.mu = np.zeros(self.nr_cells, dtype=np.int8)
        self.mu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.mu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md = np.zeros(self.nr_cells, dtype=np.int8)
        self.md1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu = np.zeros(self.nr_cells, dtype=np.int8)
        self.nu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd = np.zeros(self.nr_cells, dtype=np.int8)
        self.nd1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd2 = np.zeros(self.nr_cells, dtype=int) - 1

        self.find_first_cells_in_level()
        self.get_neighbors()
        self.get_uv_points()
        self.to_xugrid()


def odd(num):
    return np.mod(num, 2) == 1


def even(num):
    return np.mod(num, 2) == 0


def inpolygon(
    xq, yq, p
):  # p is a Polygon object, xq and yq are arrays of x and y coordinates
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    # Create list of points in tuples
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    # Create list with inout logicals (starting with False)
    inp = [False for i in range(xq.shape[0])]
    # Now start with exterior
    # Check if point is in exterior
    pth = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    # Check if point is in exterior
    inext = pth.contains_points(q).astype(bool)
    # Set inp to True where inext is True
    inp = np.logical_or(inp, inext)
    # Check if point is in interior
    for interior in p.interiors:
        pth = path.Path([(crds[0], crds[1]) for i, crds in enumerate(interior.coords)])
        inint = pth.contains_points(q).astype(bool)
        inp = np.logical_xor(inp, inint)
    # inp = inexterior(q, p, inp)
    return inp.reshape(shape)


def binary_search(val_array, vals):
    indx = np.searchsorted(val_array, vals)  # ind is size of vals
    not_ok = np.where(indx == len(val_array))[0]
    # size of vals, points that are out of bounds
    indx[
        np.where(indx == len(val_array))[0]
    ] = 0  # Set to zero to avoid out of bounds error
    is_ok = np.where(val_array[indx] == vals)[0]  # size of vals
    indices = np.zeros(len(vals), dtype=int) - 1
    indices[is_ok] = indx[is_ok]
    indices[not_ok] = -1
    return indices


def interp2(x0, y0, z0, x1, y1, method="linear"):
    f = RegularGridInterpolator(
        (y0, x0), z0, bounds_error=False, fill_value=np.nan, method=method
    )
    # reshape x1 and y1
    if x1.ndim > 1:
        sz = x1.shape
        x1 = x1.reshape(sz[0] * sz[1])
        y1 = y1.reshape(sz[0] * sz[1])
        # interpolate
        z1 = f((y1, x1)).reshape(sz)
    else:
        z1 = f((y1, x1))

    return z1


def grid_in_polygon(x, y, p):
    # Dimensions of the cells
    rows, cols = x.shape[0] - 1, x.shape[1] - 1

    # Create polygons for each cell
    x1 = x[:-1, :-1].flatten()
    y1 = y[:-1, :-1].flatten()
    x2 = x[1:, 1:].flatten()
    y2 = y[1:, 1:].flatten()
    x3 = x[:-1, 1:].flatten()
    y3 = y[:-1, 1:].flatten()
    x4 = x[1:, :-1].flatten()
    y4 = y[1:, :-1].flatten()

    # Prepare the list of cell polygons
    cell_polygons = np.array(
        [
            Polygon([(x1[i], y1[i]), (x3[i], y3[i]), (x2[i], y2[i]), (x4[i], y4[i])])
            for i in range(len(x1))
        ]
    )

    # Prepare the polygon for faster intersections
    prepared_p = prep(p)

    # Vectorized intersection checks
    inp = np.array([prepared_p.intersects(cell) for cell in cell_polygons])

    # Reshape the result back to the grid shape
    inp = inp.reshape(rows, cols)

    return inp
