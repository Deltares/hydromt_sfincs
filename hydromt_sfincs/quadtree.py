import time
import os
import logging
from matplotlib import path
import numpy as np
from pyproj import CRS, Transformer
from pathlib import Path
from typing import List, Optional, Union
import warnings
np.warnings = warnings

import geopandas as gpd
import pandas as pd
import shapely

try:
    import xugrid as xu
except ImportError:
    raise ImportError("xugrid is not installed. Please install it first.")
import xarray as xr

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import export_image
except ImportError:
    raise ImportError("datashader is not installed. Please install it first.")

from hydromt import workflows
from hydromt_sfincs.subgrid import SubgridTableQuadtree
from hydromt_sfincs.workflows.merge import merge_multi_dataarrays_on_mesh


logger = logging.getLogger(__name__)


class QuadtreeGrid:
    def __init__(self, logger=logger):
        
        self.nr_cells = 0
        self.nr_refinement_levels = 1
        self.version = 0
        
        self.data = None
        self.subgrid = SubgridTableQuadtree()
        self.df = None

    @property
    def crs(self):
        if self.data is None:
            return None
        return self.data.grid.crs

    @property
    def face_coordinates(self):
        if self.data is None:
            return None
        xy = self.data.grid.face_coordinates
        return xy[:, 0], xy[:,1]

    @property
    def exterior(self):
        if self.data is None:
            return gpd.GeoDataFrame()
        indx = self.data.grid.edge_node_connectivity[self.data.grid.exterior_edges, :]
        x = self.data.grid.node_x[indx]
        y = self.data.grid.node_y[indx]

        # Make linestrings from numpy arrays x and y
        linestrings = [shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))]
        # Merge linestrings
        merged = shapely.ops.linemerge(linestrings)
        # Merge polygons
        polygons = shapely.ops.polygonize(merged)
        
        return gpd.GeoDataFrame(geometry=list(polygons), crs=self.crs) 
    
    @property
    def empty_mask(self):
        if self.data is None:
            return None
        # create empty mask
        da0 = xr.DataArray(
            data=np.zeros(shape=len(self.data.grid.face_coordinates)),
            dims=self.data.grid.face_dimension,
        )
        return xu.UgridDataArray(da0, self.data.grid)

    def read(self, file_name:Union[str, Path] = "sfincs.nc"):
        """Reads a quadtree netcdf file and stores it in the QuadtreeGrid object."""

        self.data = xu.open_dataset(file_name)
        self.data.close()#TODO check if close works/is needed

        self.nr_cells = self.data.dims['mesh2d_nFaces']

        # set CRS (not sure if that should be stored in the netcdf in this way)
        # self.data.crs = CRS.from_wkt(self.data["crs"].crs_wkt)
        self.data.grid.set_crs(CRS.from_wkt(self.data["crs"].crs_wkt))     

        for key, value in self.data.attrs.items():
            setattr(self, key, value)

    def write(self, file_name: Union[str, Path] = "sfincs.nc", version:int=0):
        """Writes a quadtree SFINCS netcdf file."""
       
       # TODO do we want to cut inactive cells here? Or already when creating the mask?
        
        attrs = self.data.attrs
        ds = self.data.ugrid.to_dataset()

        # TODO make similar to fortran conventions
        # RENAME TO FORTRAN CONVENTION
        ds = ds.rename({"dep": "z"}) if "dep" in ds else ds
        ds = ds.rename({"msk": "mask"}) if "msk" in ds else ds
        ds = ds.rename({"snapwave_msk": "snapwave_mask"}) if "snapwave_msk" in ds else ds

        ds.attrs = attrs
        ds.to_netcdf(file_name)

    def build(
        self,
        x0: float, 
        y0: float, 
        dx: float, 
        dy: float, 
        nmax: int, 
        mmax: int, 
        epsg: int = None, 
        rotation: float = 0,
        gdf_refinement: gpd.GeoDataFrame = None
    ):
        """Builds a quadtree SFINCS grid."""

        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax
        self.mmax = mmax
        self.rotation = rotation
        self.gdf_refinement = gdf_refinement
        if epsg is not None:
            self.epsg = epsg

        # create cos and sin of rotation
        self.cosrot = np.cos(self.rotation*np.pi/180)
        self.sinrot = np.sin(self.rotation*np.pi/180)

        print("Building mesh ...")

        start = time.time()

        # Make regular grid
        self._get_regular_grid()
 
        # Initialize data arrays 
        self._initialize_data_arrays()

        # Refine all cells 
        if self.gdf_refinement is not None:
            self._refine_mesh()

        # Initialize data arrays
        self._initialize_data_arrays()

        # Get all neighbor arrays (mu, mu1, mu2, nu, nu1, nu2)
        self._get_neighbors()

        # Get uv points
        self._get_uv_points()

        # Create xugrid dataset 
        self._to_xugrid()
        
        self._clear_temporary_arrays()

        print("Time elapsed : " + str(time.time() - start) + " s")

    def setup_dep(
            self,
            datasets_dep: List[dict],
            # buffer_cells: int = 0,  # not in list
            # interp_method: str = "linear",  # used for buffer cells only):
            logger=logger,
    ):
        # TODO add buffer cells and interpolation, see merge function for more info
        
        # merge multiple datasets on mesh
        uda = merge_multi_dataarrays_on_mesh(da_list = datasets_dep, 
                                             mesh2d = self.data.grid,
                                             logger=logger)
        # add data to grid    
        self.data["dep"] = uda

    def setup_mask_active(
            self,
            model: str = "sfincs",
            gdf_mask: gpd.GeoDataFrame = None,
            gdf_include: gpd.GeoDataFrame = None,
            gdf_exclude: gpd.GeoDataFrame = None,
            zmin: float = None,
            zmax: float = None,
            all_touched: bool = False,
            reset_mask: bool = True,
            copy_sfincsmask: bool = False,
            logger=logger,
    ):
        logger.info("Building mask ...")

        assert model in ["sfincs", "snapwave"], "Model must be either 'sfincs' or 'snapwave'!"

        if model is "sfincs":
            varname = "msk"
        elif model is "snapwave":
            varname = "snapwave_msk"

        if copy_sfincsmask is True and model is "snapwave": 
            assert "msk" in self.data, "SFINCS mask not found!"
            logger.info("Using SFINCS mask for SnapWave mask ...")
            self.data[varname] = self.data["msk"]
            return
        
        logger.info("Build new mask for: " + model + " ...")

        uda_mask0 = None
        if not reset_mask and varname in self.data:
            # use current active mask
            uda_mask0 = self.data[varname] > 0
        elif gdf_mask is not None:
            # initialize mask with given geodataframe
            uda_mask0 = xu.burn_vector_geometry(gdf_mask, self.data, fill=0, all_touched=all_touched) > 0

        # always initialize an inactive mask
        uda_mask = self.empty_mask > 0

        if "dep" not in self.data and (zmin is not None or zmax is not None):
            raise ValueError("dep required in combination with zmin / zmax")

        uda_dep = self.data["dep"]
        if zmin is not None or zmax is not None:
            _msk = uda_dep != np.nan
            if zmin is not None:
                _msk = np.logical_and(_msk, uda_dep >= zmin)
            if zmax is not None:
                _msk = np.logical_and(_msk, uda_dep <= zmax)
            if uda_mask0 is not None:
                # if mask was provided; keep active mask only within valid elevations
                uda_mask = np.logical_and(uda_mask0, _msk)
            else:
                # no mask provided; set mask to valid elevations
                uda_mask = _msk
        elif zmin is None and zmax is None and uda_mask0 is not None:
            # in case a mask/region was provided, but you didn't want to update the mask based on elevation
            # just continue with the provided mask
            uda_mask = uda_mask0
        
        # TODO add fill and drop area?

        if gdf_include is not None:
            try:
                _msk = xu.burn_vector_geometry(gdf_include, self.data, fill=0, all_touched=all_touched) > 0
                uda_mask = np.logical_or(uda_mask, _msk)  # NOTE logical OR statement
            except:
                logger.debug(f"No mask cells found within include polygon!")
        if gdf_exclude is not None:
            try:
                _msk = xu.burn_vector_geometry(gdf_exclude, self.data, fill=0, all_touched=all_touched) > 0
                uda_mask = np.logical_and(uda_mask, ~_msk)
            except:
                logger.debug(f"No mask cells found within exclude polygon!")

        # add mask to grid
        self.data[varname] = xu.UgridDataArray(xr.DataArray(data=uda_mask, dims=[self.data.grid.face_dimension]), self.data.grid)    

    def setup_mask_bounds(
            self,
            model: str = "sfincs",
            btype: str = "waterlevel",
            gdf_include: gpd.GeoDataFrame = None,
            gdf_exclude: gpd.GeoDataFrame = None,
            zmin: float = None,
            zmax: float = None,
            connectivity: int = 8,
            all_touched: bool = True,
            reset_bounds: bool = True,
            copy_sfincsmask: bool = False,
            logger=logger,
    ):
        assert model in ["sfincs", "snapwave"], "Model must be either 'sfincs' or 'snapwave'!"

        if model is "sfincs":
            varname = "msk"
        elif model is "snapwave":
            varname = "snapwave_msk"

        if copy_sfincsmask is True and model is "snapwave": 
            assert "msk" in self.data, "SFINCS mask not found!"
            logger.info("Using SFINCS mask for SnapWave mask ...")
            self.data[varname] = self.data["msk"]
            return
        
        if varname not in self.data:
            raise ValueError("First setup active mask for model: " + model)
        else:
            uda_mask = self.data[varname]
        
        if "dep" not in self.data and (zmin is not None or zmax is not None):
            raise ValueError("dep required in combination with zmin / zmax")
        else:
            uda_dep = self.data["dep"]
        
        btype = btype.lower()
        bvalues = {"waterlevel": 2, "outflow": 3}
        if btype not in bvalues:
            raise ValueError('btype must be one of "waterlevel", "outflow"')
        bvalue = bvalues[btype]

        if reset_bounds:  # reset existing boundary cells
            logger.debug(f"{btype} (mask={bvalue:d}) boundary cells reset.")
            uda_mask = uda_mask.where(uda_mask != np.uint8(bvalue), np.uint8(1))
            if (
                zmin is None
                and zmax is None
                and gdf_include is None
                and gdf_exclude is None
            ):
                return uda_mask

        # find boundary cells of the active mask        
        bounds_org = self._find_boundary_cells(varname)

        #same with Xugrid functionality:
        # s = None if connectivity == 4 else np.ones((3, 3), int)
        # da_mask = self.data[varname] > 0
        # bounds0 = np.logical_xor(
        #     da_mask, da_mask.ugrid.binary_erosion(iterations=2)#border_value=True)#da_mask.values>0)#self.data[varname] > 0)#, structure=s)
        # )
        # bounds0 = np.logical_xor(
        #     self.data[varname] > 0, self.data[varname].ugrid.binary_erosion(self.data[varname] > 0)#, structure=s)
        # )        
        # bounds = bounds0.copy()
        bounds = bounds_org.copy()

        if zmin is not None:
            bounds = np.logical_and(bounds, uda_dep >= zmin)
        if zmax is not None:
            bounds = np.logical_and(bounds, uda_dep <= zmax)
        if gdf_include is not None:
            uda_include = xu.burn_vector_geometry(gdf_include, self.data, fill=0, all_touched=all_touched) > 0
            bounds = np.logical_and(bounds, uda_include)
        if gdf_exclude is not None:
            uda_exclude = xu.burn_vector_geometry(gdf_exclude, self.data, fill=0, all_touched=all_touched) > 0
            bounds = np.logical_and(bounds, ~uda_exclude)  

        # TODO avoid any msk3 cells neighboring msk2 cells
            
        ncells = np.count_nonzero(bounds.values)
        if ncells > 0:
            uda_mask = uda_mask.where(~bounds, np.uint8(bvalue))

        # try to include 'diagonally connected msk=2 neighbouring cells'
        if connectivity == 4:
            self.bounds_msk2 = uda_mask.copy()
            bounds_msk2 = self._find_boundary_cells_msk2()#uda_mask)

            ncells = bounds_msk2.sum()#np.count_nonzero(bounds_msk2.sum())
            if ncells > 0:
                uda_mask = uda_mask.where(~bounds_msk2, np.uint8(bvalue))
            
        # add mask to grid
        self.data[varname] = xu.UgridDataArray(xr.DataArray(data=uda_mask, dims=[self.data.grid.face_dimension]), self.data.grid)    

    # TODO, this method can be removed, as it is replaced by setup_mask_active and setup_mask_bounds
    def setup_mask(
            self,
            model="sfincs",  # "sfincs" for SFINCS: data["mask"], "snapwave" for SnapWave: data["snapwavemask"]
            copy_sfincs_mask2snapwave=False,
            zmin=99999.0,
            zmax=-99999.0,
            include_polygon=None,
            exclude_polygon=None,
            open_boundary_polygon=None,
            outflow_boundary_polygon=None,
            include_zmin=-99999.0,
            include_zmax= 99999.0,
            exclude_zmin=-99999.0,
            exclude_zmax= 99999.0,
            open_boundary_zmin=-99999.0,
            open_boundary_zmax= 99999.0,
            outflow_boundary_zmin=-99999.0,
            outflow_boundary_zmax= 99999.0,
            quiet=True,
    ):

        if not quiet:
            print("Building mask ...")
            
        if model is "sfincs":
            varname = "mask"
        elif model is "snapwave":
            varname = "snapwave_mask"
        else:
            print("Requested model to build mask for not recognized! Choose either 'sfincs' or 'snapwave' ...")
                       
        if copy_sfincs_mask2snapwave is True and model is "snapwave" and self.data["mask"] is not None: #TODO: check whether the 'self.data["mask"] is not None' is robust in code order
            print("Using SFINCS mask for Snapwave mask ...")
            self.data[varname] = self.data["mask"]
        else:
            print("Build new mask for: " + model + " ...")

            time_start = time.time()           
            mask = np.zeros(self.nr_cells, dtype=np.int8)
            x, y = self.face_coordinates
            if "dep" in self.data:
                z    = self.data["dep"].values[:]
            else:
                z    = None

            mu    = self.data["mu"].values[:]
            mu1   = self.data["mu1"].values[:] - 1
            mu2   = self.data["mu2"].values[:] - 1
            nu    = self.data["nu"].values[:]
            nu1   = self.data["nu1"].values[:] - 1 
            nu2   = self.data["nu2"].values[:] - 1
            md    = self.data["md"].values[:]
            md1   = self.data["md1"].values[:] - 1
            md2   = self.data["md2"].values[:] - 1
            nd    = self.data["nd"].values[:]
            nd1   = self.data["nd1"].values[:] - 1
            nd2   = self.data["nd2"].values[:] - 1 

            if zmin>=zmax:
                # Do not include any points initially
                if include_polygon is None:
                    print("WARNING: Entire mask set to zeros! Please ensure zmax is greater than zmin, or provide include polygon(s) !")
                    return
            else:
                if z is not None:                
                    # Set initial mask based on zmin and zmax
                    iok = np.where((z>=zmin) & (z<=zmax))
                    mask[iok] = 1
                else:
                    print("WARNING: Entire mask set to zeros! No depth values found on grid.") #DOTO - question - also/not for only include_polygon case?
                            
            # Include polygons
            if include_polygon is not None:
                for ip, polygon in include_polygon.iterrows():
                    inpol = inpolygon(x, y, polygon["geometry"])
                    # iok   = np.where((inpol) & (z>=include_zmin) & (z<=include_zmax))
                    iok   = np.where((inpol) ) # TODO: question - do we want include to depend on elevation, or 'overwrite' this?                
                    mask[iok] = 1

            # Exclude polygons
            if exclude_polygon is not None:
                for ip, polygon in exclude_polygon.iterrows():
                    inpol = inpolygon(x, y, polygon["geometry"])
                    iok   = np.where((inpol) & (z>=exclude_zmin) & (z<=exclude_zmax))
                    mask[iok] = 0

            print("Time elapsed for creating active mask: " + str(time.time() - time_start) + " s")

            time_start = time.time()
            # Open boundary polygons
            if open_boundary_polygon is not None:
                for ip, polygon in open_boundary_polygon.iterrows():
                    inpol = inpolygon(x, y, polygon["geometry"])
                    # Only consider points that are:
                    # 1) Inside the polygon
                    # 2) Have a mask > 0
                    # 3) z>=zmin
                    # 4) z<=zmax
                    iok   = np.where((inpol) & (mask>0) & (z>=open_boundary_zmin) & (z<=open_boundary_zmax))
                    for ic in iok[0]:
                        okay = False
                        # Check neighbors, cell must have at least one inactive neighbor
                        # Left
                        if md[ic]<=0:
                            # Coarser or equal to the left
                            if md1[ic]>=0:
                                # Cell has neighbor to the left
                                if mask[md1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                        else:
                            # Finer to the left
                            if md1[ic]>=0:
                                # Cell has neighbor to the left
                                if mask[md1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                            if md2[ic]>=0:
                                # Cell has neighbor to the left
                                if mask[md2[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                            
                        # Below
                        if nd[ic]<=0:
                            # Coarser or equal below
                            if nd1[ic]>=0:
                                # Cell has neighbor below
                                if mask[nd1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                        else:
                            # Finer below
                            if nd1[ic]>=0:
                                # Cell has neighbor below
                                if mask[nd1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                            if nd2[ic]>=0:
                                # Cell has neighbor below
                                if mask[nd2[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True

                        # Right
                        if mu[ic]<=0:
                            # Coarser or equal to the right
                            if mu1[ic]>=0:
                                # Cell has neighbor to the right
                                if mask[mu1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                        else:
                            # Finer to the left
                            if mu1[ic]>=0:
                                # Cell has neighbor to the right
                                if mask[mu1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                            if mu2[ic]>=0:
                                # Cell has neighbor to the right
                                if mask[mu2[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True

                        # Above
                        if nu[ic]<=0:
                            # Coarser or equal above
                            if nu1[ic]>=0:
                                # Cell has neighbor above
                                if mask[nu1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                        else:
                            # Finer below
                            if nu1[ic]>=0:
                                # Cell has neighbor above
                                if mask[nu1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                            if nu2[ic]>=0:
                                # Cell has neighbor above
                                if mask[nu2[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 2
                                okay = True
                            
                        if okay:
                            mask[ic] = 2

            # Outflow boundary polygons
            if outflow_boundary_polygon is not None:
                for ip, polygon in outflow_boundary_polygon.iterrows():
                    inpol = inpolygon(x, y, polygon["geometry"])
                    # Only consider points that are:
                    # 1) Inside the polygon
                    # 2) Have a mask > 0
                    # 3) z>=zmin
                    # 4) z<=zmax
                    iok   = np.where((inpol) & (mask>0) & (z>=outflow_boundary_zmin) & (z<=outflow_boundary_zmax))
                    for ic in iok[0]:
                        okay = False
                        # Check neighbors, cell must have at least one inactive neighbor
                        # Left
                        if md[ic]<=0:
                            # Coarser or equal to the left
                            if md1[ic]>=0:
                                # Cell has neighbor to the left
                                if mask[md1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                        else:
                            # Finer to the left
                            if md1[ic]>=0:
                                # Cell has neighbor to the left
                                if mask[md1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                            if md2[ic]>=0:
                                # Cell has neighbor to the left
                                if mask[md2[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                            
                        # Below
                        if nd[ic]<=0:
                            # Coarser or equal below
                            if nd1[ic]>=0:
                                # Cell has neighbor below
                                if mask[nd1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                        else:
                            # Finer below
                            if nd1[ic]>=0:
                                # Cell has neighbor below
                                if mask[nd1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                            if nd2[ic]>=0:
                                # Cell has neighbor below
                                if mask[nd2[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True

                        # Right
                        if mu[ic]<=0:
                            # Coarser or equal to the right
                            if mu1[ic]>=0:
                                # Cell has neighbor to the right
                                if mask[mu1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                        else:
                            # Finer to the left
                            if mu1[ic]>=0:
                                # Cell has neighbor to the right
                                if mask[mu1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                            if mu2[ic]>=0:
                                # Cell has neighbor to the right
                                if mask[mu2[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True

                        # Above
                        if nu[ic]<=0:
                            # Coarser or equal above
                            if nu1[ic]>=0:
                                # Cell has neighbor above
                                if mask[nu1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                        else:
                            # Finer below
                            if nu1[ic]>=0:
                                # Cell has neighbor above
                                if mask[nu1[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True
                            if nu2[ic]>=0:
                                # Cell has neighbor above
                                if mask[nu2[ic]]==0:
                                    # And it's inactive
                                    okay = True
                            else:
                                # No neighbor, so set mask = 3
                                okay = True                        
                        if okay:
                            mask[ic] = 3
            print("Time elapsed for creating mask boundaries : " + str(time.time() - time_start) + " s")

            # Now add the data arrays
            ugrid2d = self.data.grid
            self.data[varname] = xu.UgridDataArray(xr.DataArray(data=mask, dims=[ugrid2d.face_dimension]), ugrid2d)

    def setup_subgrid(
        self,
        datasets_dep: list[dict],
        datasets_rgh: list[dict] = [],
        datasets_riv: list[dict] = [],
        nlevels=10,
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
        parallel=False,        
    ):
        self.subgrid.build(
            ds_mesh = self.data,
            datasets_dep = datasets_dep,
            datasets_rgh = datasets_rgh,
            datasets_riv = datasets_riv,
            nlevels = nlevels,
            nr_subgrid_pixels = nr_subgrid_pixels,
            nrmax = nrmax,
            max_gradient = max_gradient,
            z_minimum = z_minimum,
            z_multiply = z_multiply,
            manning_land = manning_land,
            manning_sea = manning_sea,
            rgh_lev_land = rgh_lev_land,
            huthresh = huthresh,
            buffer_cells = buffer_cells,
            write_dep_tif = write_dep_tif,
            write_man_tif = write_man_tif,
            highres_dir = highres_dir,
            logger = logger,
            progress_bar = None,
            parallel = parallel,
        )


    def _get_datashader_dataframe(self):
        # Create a dataframe with line elements
        x1 = self.data.grid.edge_node_coordinates[:,0,0]
        x2 = self.data.grid.edge_node_coordinates[:,1,0]
        y1 = self.data.grid.edge_node_coordinates[:,0,1]
        y2 = self.data.grid.edge_node_coordinates[:,1,1]
        transformer = Transformer.from_crs(self.crs,
                                            3857,
                                            always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        self.df = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):
        if self.data is None:
            # No grid (yet)
            return False
        try:
            if not hasattr(self, "df"):
                self.df = None
            if self.df is None: 
                self._get_datashader_dataframe()

            transformer = Transformer.from_crs(4326,
                                        3857,
                                        always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)
            cvs = ds.Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)
            agg = cvs.line(self.df, x=['x1', 'x2'], y=['y1', 'y2'], axis=1)
            img = tf.shade(agg)
            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True
        except Exception as e:
            return False


    def snap_to_grid(self, polyline, max_snap_distance=1.0):
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        geom_list = []
        for iline, line in polyline.iterrows():
            geom = line["geometry"]
            if geom.geom_type == 'LineString':
                geom_list.append(geom)
        gdf = gpd.GeoDataFrame({'geometry': geom_list})    
        print("Snapping to grid ...")
        snapped_uds, snapped_gdf = xu.snap_to_grid(gdf, self.data.grid, max_snap_distance=max_snap_distance)
        print("Snapping to grid done.")
        snapped_gdf = snapped_gdf.set_crs(self.crs)
        return snapped_gdf

## Quadtree helper functions
    def _get_regular_grid(self):
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
        self._find_first_cells_in_level()
        # Compute cell center coordinates self.x and self.y
        self._compute_cell_center_coordinates()


    def _find_first_cells_in_level(self):
        # Find first cell in each level
        self.ifirst = np.zeros(self.nr_refinement_levels, dtype=int)
        self.ilast = np.zeros(self.nr_refinement_levels, dtype=int)
        for ilev in range(0, self.nr_refinement_levels):
            # Find index of first cell with this level
            self.ifirst[ilev] = np.where(self.level == ilev)[0][0]
            # Find index of last cell with this level
            if ilev<self.nr_refinement_levels - 1:
                self.ilast[ilev] = np.where(self.level == ilev + 1)[0][0] - 1
            else:
                self.ilast[ilev] = self.nr_cells - 1

    def _compute_cell_center_coordinates(self):
        # Compute cell center coordinates
        # Loop through refinement levels
        dx = self.dx/2**self.level
        dy = self.dy/2**self.level
        self.x = self.x0 + self.cosrot * (self.m + 0.5) * dx - self.sinrot * (self.n + 0.5) * dy
        self.y = self.y0 + self.sinrot * (self.m + 0.5) * dx + self.cosrot * (self.n + 0.5) * dy

    def _get_ugrid2d(self):

        tic = time.perf_counter()

        n = self.n
        m = self.m
        level = self.level

        nmax       = self.nmax * 2**(self.nr_refinement_levels - 1) + 1

        face_nodes_n = np.full((8,self.nr_cells), -1, dtype=int)
        face_nodes_m = np.full((8,self.nr_cells), -1, dtype=int)
        face_nodes_nm = np.full((8,self.nr_cells), -1, dtype=int)

        # HIghest refinement level 
        ifac = 2**(self.nr_refinement_levels - level - 1)
        dxf = self.dx / 2**(self.nr_refinement_levels - 1)
        dyf = self.dy / 2**(self.nr_refinement_levels - 1)

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
        i = np.where(self.nd==1)
        face_nodes_n[1, i] = face_n[i]
        face_nodes_m[1, i] = face_m[i] + ifac[i]/2
        # Find cells with refinement to the right
        i = np.where(self.mu==1)
        face_nodes_n[3, i] = face_n[i] + ifac[i]/2
        face_nodes_m[3, i] = face_m[i] + ifac[i]
        # Find cells with refinement above
        i = np.where(self.nu==1)
        face_nodes_n[5, i] = face_n[i] + ifac[i]
        face_nodes_m[5, i] = face_m[i] + ifac[i]/2
        # Find cells with refinement to the left
        i = np.where(self.md==1)
        face_nodes_n[7, i] = face_n[i] + ifac[i]/2
        face_nodes_m[7, i] = face_m[i]

        # Flatten
        face_nodes_n = face_nodes_n.transpose().flatten()
        face_nodes_m = face_nodes_m.transpose().flatten()

        # Compute nm value of nodes        
        face_nodes_nm = nmax * face_nodes_m + face_nodes_n
        nopoint = max(face_nodes_nm) + 1
        # Set missing points to very high number
        face_nodes_nm[np.where(face_nodes_n==-1)] = nopoint

        # Get the unique nm values
        xxx, index, irev = np.unique(face_nodes_nm, return_index=True, return_inverse=True)
        j = np.where(xxx==nopoint)[0][0] # Index of very high number
        # irev2 = np.reshape(irev, (self.nr_cells, 8))
        # face_nodes_all = irev2.transpose()
        face_nodes_all = np.reshape(irev, (self.nr_cells, 8)).transpose()
        face_nodes_all[np.where(face_nodes_all==j)] = -1

        face_nodes = np.full(face_nodes_all.shape, -1)  # Create a new array filled with -1
        for i in range(face_nodes.shape[1]):
            idx = np.where(face_nodes_all[:,i] != -1)[0]
            face_nodes[:len(idx), i] = face_nodes_all[idx, i]  

        # Now get rid of all the rows where all values are -1
        # Create a mask where each row is True if not all elements in the row are -1
        mask = (face_nodes != -1).any(axis=1)

        # Use this mask to index face_nodes
        face_nodes = face_nodes[mask]

        node_n = face_nodes_n[index[:j]]
        node_m = face_nodes_m[index[:j]]
        node_x = self.x0 + self.cosrot*(node_m*dxf) - self.sinrot*(node_n*dyf)
        node_y = self.y0 + self.sinrot*(node_m*dxf) + self.cosrot*(node_n*dyf)

        toc = time.perf_counter()

        print(f"Got rid of duplicates in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()

        nodes = np.transpose(np.vstack((node_x, node_y)))
        faces = np.transpose(face_nodes)
        fill_value = -1

        ugrid2d = xu.Ugrid2d(nodes[:, 0], nodes[:, 1], fill_value, faces)

        if self.epsg is not None:
            ugrid2d.set_crs(CRS.from_user_input(self.epsg))

        # Set datashader df to None
        self.df = None 

        toc = time.perf_counter()

        print(f"Made XUGrid in {toc - tic:0.4f} seconds")

        return ugrid2d

    def _cut_inactive_cells(self):

        print("Removing inactive cells ...")

        # In the xugrid data, the indices are 1-based, so we need to subtract 1 
        n = self.data["n"].values[:] - 1
        m = self.data["m"].values[:] - 1
        level = self.data["level"].values[:] - 1
        dep = self.data["dep"].values[:]
        mask = self.data["msk"].values[:]
        swmask = self.data["snapwave_msk"].values[:]

        indx = np.where((mask + swmask)>0)
            
        self.nr_cells = np.size(indx)
        self.n        = n[indx]
        self.m        = m[indx]
        self.level    = level[indx]
        self.dep        = dep[indx] 
        self.mask     = mask[indx]
        self.snapwave_mask = swmask[indx]

        # Set indices of neighbors to -1
        self.mu  = np.zeros(self.nr_cells, dtype=np.int8)
        self.mu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.mu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md  = np.zeros(self.nr_cells, dtype=np.int8)
        self.md1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu  = np.zeros(self.nr_cells, dtype=np.int8)
        self.nu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd  = np.zeros(self.nr_cells, dtype=np.int8)
        self.nd1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd2 = np.zeros(self.nr_cells, dtype=int) - 1

        self._find_first_cells_in_level()
        self._get_neighbors() 
        self._get_uv_points()
        self._to_xugrid()

    def _initialize_data_arrays(self):
        # Set indices of neighbors to -1
        self.mu  = np.zeros(self.nr_cells, dtype=np.int8)
        self.mu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.mu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md  = np.zeros(self.nr_cells, dtype=np.int8)
        self.md1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu  = np.zeros(self.nr_cells, dtype=np.int8)
        self.nu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd  = np.zeros(self.nr_cells, dtype=np.int8)
        self.nd1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd2 = np.zeros(self.nr_cells, dtype=int) - 1

    def _refine_mesh(self): 
        # Loop through rows in gdf and create list of polygons
        # Determine maximum refinement level

        start = time.time()
        print("Refining ...")

        self.ref_pols = []
        for irow, row in self.gdf_refinement.iterrows():
            iref = row["refinement_level"]
            polygon = {"geometry": row["geometry"], "refinement_level": iref}
            self.ref_pols.append(polygon)

        # Loop through refinement polygons and start refining
        for polygon in self.ref_pols:
            # Refine, reorder, find first cells in level
            self._refine_in_polygon(polygon)

        print("Time elapsed : " + str(time.time() - start) + " s")

    def _refine_in_polygon(self, polygon):
        # Finds cell to refine and calls refine_cells

        # Loop through refinement levels for this polygon
        for ilev in range(polygon["refinement_level"]):

            # Refine cells in refinement polygons
            # Compute grid spacing for this level
            dx = self.dx/2**ilev
            dy = self.dy/2**ilev
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
                polx[ipoint] =   self.cosrot*(point[0] - self.x0) + self.sinrot*(point[1] - self.y0)
                poly[ipoint] = - self.sinrot*(point[0] - self.x0) + self.cosrot*(point[1] - self.y0)

            # Find cells cells in grid that could fall within polbuf 
            n0 = int(np.floor(np.min(poly) / dy)) - 1
            n1 = int(np.ceil(np.max(poly) / dy)) + 1
            m0 = int(np.floor(np.min(polx) / dx)) - 1
            m1 = int(np.ceil(np.max(polx) / dx)) + 1

            n0 = min(max(n0, 0), nmax - 1)
            n1 = min(max(n1, 0), nmax - 1)
            m0 = min(max(m0, 0), mmax - 1)
            m1 = min(max(m1, 0), mmax - 1)

            # Compute cell centre coordinates of cells in this level in this block
            nn, mm = np.meshgrid(np.arange(n0, n1 + 1), np.arange(m0, m1 + 1))
            nn = np.transpose(nn).flatten()
            mm = np.transpose(mm).flatten()

            xcor = np.zeros((4, np.size(nn)))
            ycor = np.zeros((4, np.size(nn)))
            xcor[0,:] = self.x0 + self.cosrot * (mm + 0) * dx - self.sinrot * (nn + 0) * dy
            ycor[0,:] = self.y0 + self.sinrot * (mm + 0) * dx + self.cosrot * (nn + 0) * dy
            xcor[1,:] = self.x0 + self.cosrot * (mm + 1) * dx - self.sinrot * (nn + 0) * dy
            ycor[1,:] = self.y0 + self.sinrot * (mm + 1) * dx + self.cosrot * (nn + 0) * dy
            xcor[2,:] = self.x0 + self.cosrot * (mm + 1) * dx - self.sinrot * (nn + 1) * dy
            ycor[2,:] = self.y0 + self.sinrot * (mm + 1) * dx + self.cosrot * (nn + 1) * dy
            xcor[3,:] = self.x0 + self.cosrot * (mm + 0) * dx - self.sinrot * (nn + 1) * dy
            ycor[3,:] = self.y0 + self.sinrot * (mm + 0) * dx + self.cosrot * (nn + 1) * dy

            # Create np array with False for all cells
            inp = np.zeros(np.size(nn), dtype=bool)
            # Loop through 4 corner points
            # If any corner points falls within the polygon, inp is set to True
            for j in range(4):
                inp0 = inpolygon(np.squeeze(xcor[j,:]),
                                 np.squeeze(ycor[j,:]),
                                 polygon["geometry"])
                inp[np.where(inp0)] = True
            in_polygon = np.where(inp)[0]

            # Indices of cells in level within polbuf
            nn_in = nn[in_polygon]
            mm_in = mm[in_polygon]
            nm_in = nmax * mm_in + nn_in

            # Find existing cells of this level in nmi array
            n_level = self.n[self.ifirst[ilev]:self.ilast[ilev] + 1]
            m_level = self.m[self.ifirst[ilev]:self.ilast[ilev] + 1]
            nm_level = m_level * nmax + n_level

            # Find indices all cells to be refined
            ind_ref = binary_search(nm_level, nm_in)

            ind_ref=ind_ref[ind_ref>=0]

            # ind_ref = ind_ref[ind_ref < np.size(nm_level)]
            if not np.any(ind_ref):
                continue
            # Index of cells to refine
            ind_ref += self.ifirst[ilev]

            self._refine_cells(ind_ref, ilev)

    def _refine_cells(self, ind_ref, ilev):
        # Refine cells with index ind_ref

        # First find lower-level neighbors (these will be refined in the next iteration)
        if ilev>0:
            ind_nbr = self._find_lower_level_neighbors(ind_ref, ilev)
        else:
            ind_nbr = np.empty(0, dtype=int)    

        # n and m indices of cells to be refined
        n = self.n[ind_ref]
        m = self.m[ind_ref]

        # New cells
        nnew = np.zeros(4 * len(ind_ref), dtype=int)
        mnew = np.zeros(4 * len(ind_ref), dtype=int)
        lnew = np.zeros(4 * len(ind_ref), dtype=int) + ilev + 1
        nnew[0::4] = n*2      # lower left
        nnew[1::4] = n*2 + 1  # upper left
        nnew[2::4] = n*2      # lower right
        nnew[3::4] = n*2 + 1  # upper right
        mnew[0::4] = m*2      # lower left
        mnew[1::4] = m*2      # upper left
        mnew[2::4] = m*2 + 1  # lower right
        mnew[3::4] = m*2 + 1  # upper right
        # Add new cells to grid
        self.n = np.append(self.n, nnew)
        self.m = np.append(self.m, mnew)
        self.level = np.append(self.level, lnew)
        # Remove old cells from grid
        self.n = np.delete(self.n, ind_ref)
        self.m = np.delete(self.m, ind_ref)
        self.level = np.delete(self.level, ind_ref)        
        self.nr_cells = len(self.n)
        # Update nr_refinement_levels at max of ilev + 2 and self.nr_refinement_levels
        self.nr_refinement_levels = np.maximum(self.nr_refinement_levels, ilev + 2)
        # Reorder cells
        self._reorder()
        # Update ifirst and ilast
        self._find_first_cells_in_level()
        # Compute cell center coordinates self.x and self.y
        self._compute_cell_center_coordinates()

        if np.any(ind_nbr):
            self._refine_cells(ind_nbr, ilev - 1)

    def _reorder(self):
        # Reorder cells
        # Sort cells by level, then m, then n
        i = np.lexsort((self.n, self.m, self.level))
        self.n = self.n[i]
        self.m = self.m[i]
        self.level = self.level[i]

    def _get_uv_points(self):

        start = time.time()
        print("Getting uv points ...")

        # Get uv points (do we actually need to do this?)
        self.uv_index_z_nm  = np.zeros((self.nr_cells*4), dtype=int)
        self.uv_index_z_nmu = np.zeros((self.nr_cells*4), dtype=int)
        self.uv_dir         = np.zeros((self.nr_cells*4), dtype=int)
        # Loop through points (SHOULD TRY TO VECTORIZE THIS, but try to keep same order of uv points
        nuv = 0
        for ip in range(self.nr_cells):
            if self.mu1[ip]>=0:
                self.uv_index_z_nm[nuv] = ip        
                self.uv_index_z_nmu[nuv] = self.mu1[ip]     
                self.uv_dir[nuv] = 0
                nuv += 1
            if self.mu2[ip]>=0:
                self.uv_index_z_nm[nuv] = ip        
                self.uv_index_z_nmu[nuv] = self.mu2[ip]     
                self.uv_dir[nuv] = 0     
                nuv += 1
            if self.nu1[ip]>=0:
                self.uv_index_z_nm[nuv] = ip        
                self.uv_index_z_nmu[nuv] = self.nu1[ip]     
                self.uv_dir[nuv] = 1     
                nuv += 1
            if self.nu2[ip]>=0:
                self.uv_index_z_nm[nuv] = ip
                self.uv_index_z_nmu[nuv] = self.nu2[ip]
                self.uv_dir[nuv] = 1
                nuv += 1
        self.uv_index_z_nm  = self.uv_index_z_nm[0 : nuv]
        self.uv_index_z_nmu = self.uv_index_z_nmu[0 : nuv]
        self.uv_dir         = self.uv_dir[0 : nuv]
        self.nr_uv_points   = nuv

        print("Time elapsed : " + str(time.time() - start) + " s")

    def _get_neighbors(self):
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
            mu1[inb>=0] = inb[inb>=0] + i0

            # Above
            nm_to_find = nm + 1
            inb = binary_search(nm, nm_to_find)
            nu1[inb>=0] = inb[inb>=0] + i0

            ## Coarser level neighbors
            if ilev>0:

                nmaxc = self.nmax * 2**(ilev - 1) + 1   # Number of cells in coarser level in n direction 

                i0c = self.ifirst[ilev - 1]  # First cell in coarser level                
                i1c = self.ilast[ilev - 1] + 1 # Last cell in coarser level

                nmc = nm_all[i0c:i1c] # Coarser level nm indices
                nc = n // 2 # Coarser level n index of this cells in this level
                mc = m // 2 # Coarser level m index of this cells in this level 

                # Right
                nmc_to_find = (mc + 1) * nmaxc + nc
                inb = binary_search(nmc, nmc_to_find)
                inb[np.where(even(m))[0]] = -1
                # Set mu and mu1 for inb>=0
                mu1[inb>=0] = inb[inb>=0] + i0c
                mu[inb>=0] = -1

                # Above
                nmc_to_find = mc * nmaxc + nc + 1
                inb = binary_search(nmc, nmc_to_find)
                inb[np.where(even(n))[0]] = -1
                # Set nu and nu1 for inb>=0
                nu1[inb>=0] = inb[inb>=0] + i0c
                nu[inb>=0] = -1

            # Finer level neighbors
            if ilev<self.nr_refinement_levels - 1:

                nmaxf = self.nmax * 2**(ilev + 1) + 1 # Number of cells in finer level in n direction

                i0f = self.ifirst[ilev + 1]  # First cell in finer level
                i1f = self.ilast[ilev + 1] + 1 # Last cell in finer level
                nmf = nm_all[i0f:i1f] # Finer level nm indices

                # Right

                # Lower row
                nf = n * 2 # Finer level n index of this cells in this level
                mf = m * 2 + 1 # Finer level m index of this cells in this level
                nmf_to_find = (mf + 1) * nmaxf + nf
                inb = binary_search(nmf, nmf_to_find)
                mu1[inb>=0] = inb[inb>=0] + i0f
                mu[inb>=0] = 1

                # Upper row
                nf = n * 2 + 1# Finer level n index of this cells in this level
                mf = m * 2 + 1 # Finer level m index of this cells in this level
                nmf_to_find = (mf + 1) * nmaxf + nf
                inb = binary_search(nmf, nmf_to_find)
                mu2[inb>=0] = inb[inb>=0] + i0f
                mu[inb>=0] = 1

                # Above

                # Left column
                nf = n * 2 + 1 # Finer level n index of this cells in this level
                mf = m * 2 # Finer level m index of this cells in this level
                nmf_to_find = mf * nmaxf + nf + 1
                inb = binary_search(nmf, nmf_to_find)
                nu1[inb>=0] = inb[inb>=0] + i0f
                nu[inb>=0] = 1

                # Right column
                nf = n * 2 + 1 # Finer level n index of this cells in this level
                mf = m * 2 + 1 # Finer level m index of this cells in this level
                nmf_to_find = mf * nmaxf + nf + 1
                inb = binary_search(nmf, nmf_to_find)
                nu2[inb>=0] = inb[inb>=0] + i0f
                nu[inb>=0] = 1

            # Fill in mu, mu1, mu2, nu, nu1, nu2 for this level
            self.mu[i0:i1] = mu
            self.mu1[i0:i1] = mu1
            self.mu2[i0:i1] = mu2
            self.nu[i0:i1] = nu
            self.nu1[i0:i1] = nu1
            self.nu2[i0:i1] = nu2

        print("Time elapsed : " + str(time.time() - start) + " s")

        print("Setting neighbors left and below ...")

        # Right
       
        iok1 = np.where(self.mu1>=0)[0]
        # Same level
        iok2 = np.where(self.mu==0)[0]
        # Indices of cells that have a same level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Indices of neighbors
        imu = self.mu1[iok]
        self.md[imu] = 0
        self.md1[imu] = iok

        # Coarser
        iok2 = np.where(self.mu==-1)[0]
        # Indices of cells that have a coarse level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Odd
        iok_odd  = iok[np.where(odd(self.n[iok]))]
        iok_even = iok[np.where(even(self.n[iok]))]
        imu = self.mu1[iok_odd]
        self.md[imu] = 1
        self.md1[imu] = iok_odd
        imu = self.mu1[iok_even]
        self.md[imu] = 1
        self.md2[imu] = iok_even

        # Finer
        # Lower
        iok1 = np.where(self.mu1>=0)[0]
        # Same level
        iok2 = np.where(self.mu==1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        imu = self.mu1[iok]
        self.md[imu] = -1
        self.md1[imu] = iok
        # Upper
        iok1 = np.where(self.mu2>=0)[0]
        # Same level
        iok2 = np.where(self.mu==1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        imu = self.mu2[iok]
        self.md[imu] = -1
        self.md1[imu] = iok

        # Above
        iok1 = np.where(self.nu1>=0)[0]
        # Same level
        iok2 = np.where(self.nu==0)[0]
        # Indices of cells that have a same level neighbor above
        iok = np.intersect1d(iok1, iok2)
        # Indices of neighbors
        inu = self.nu1[iok]
        self.nd[inu] = 0
        self.nd1[inu] = iok

        # Coarser
        iok2 = np.where(self.nu==-1)[0]
        # Indices of cells that have a coarse level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Odd
        iok_odd  = iok[np.where(odd(self.m[iok]))]
        iok_even = iok[np.where(even(self.m[iok]))]
        inu = self.nu1[iok_odd]
        self.nd[inu] = 1
        self.nd1[inu] = iok_odd
        inu = self.nu1[iok_even]
        self.nd[inu] = 1
        self.nd2[inu] = iok_even

        # Finer
        # Left
        iok1 = np.where(self.nu1>=0)[0]
        # Same level
        iok2 = np.where(self.nu==1)[0]
        # Indices of cells that have finer level neighbor above
        iok = np.intersect1d(iok1, iok2)
        inu = self.nu1[iok]
        self.nd[inu] = -1
        self.nd1[inu] = iok
        # Upper
        iok1 = np.where(self.nu2>=0)[0]
        # Same level
        iok2 = np.where(self.nu==1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        inu = self.nu2[iok]
        self.nd[inu] = -1
        self.nd1[inu] = iok

        print("Time elapsed : " + str(time.time() - start) + " s")

    def _to_xugrid(self):    

        print("Making XUGrid ...")

        # Create the grid
        ugrid2d = self._get_ugrid2d()

        # Create the dataset
        self.data = xu.UgridDataset(grids=ugrid2d)

        # Add attributes
        attrs = {"x0": self.x0,
                 "y0": self.y0,
                 "nmax": self.nmax,
                 "mmax": self.mmax,
                 "dx": self.dx,
                 "dy": self.dy,
                 "rotation": self.rotation,
                 "nr_levels": self.nr_refinement_levels}
        self.data.attrs = attrs

        # Now add the data arrays
        self.data["crs"] = ugrid2d.crs.to_epsg()
        self.data["crs"].attrs = ugrid2d.crs.to_cf()
        self.data["level"] = xu.UgridDataArray(xr.DataArray(data=self.level + 1, dims=[ugrid2d.face_dimension]), ugrid2d)

        # add dep, msk and snapwave_msk if they exist in self
        if hasattr(self, "dep"):
            self.data["dep"] = xu.UgridDataArray(xr.DataArray(data=self.dep, dims=[ugrid2d.face_dimension]), ugrid2d)
        if hasattr(self, "mask"):
            self.data["msk"] = xu.UgridDataArray(xr.DataArray(data=self.mask, dims=[ugrid2d.face_dimension]), ugrid2d)
        if hasattr(self, "snapwave_mask"):
            self.data["snapwave_msk"] = xu.UgridDataArray(xr.DataArray(data=self.snapwave_mask, dims=[ugrid2d.face_dimension]), ugrid2d)

        self.data["n"] = xu.UgridDataArray(xr.DataArray(data=self.n + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["m"] = xu.UgridDataArray(xr.DataArray(data=self.m + 1, dims=[ugrid2d.face_dimension]), ugrid2d)

        self.data["mu"]  = xu.UgridDataArray(xr.DataArray(data=self.mu, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["mu1"] = xu.UgridDataArray(xr.DataArray(data=self.mu1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["mu2"] = xu.UgridDataArray(xr.DataArray(data=self.mu2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["md"]  = xu.UgridDataArray(xr.DataArray(data=self.md, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["md1"] = xu.UgridDataArray(xr.DataArray(data=self.md1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["md2"] = xu.UgridDataArray(xr.DataArray(data=self.md2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)

        self.data["nu"]  = xu.UgridDataArray(xr.DataArray(data=self.nu, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["nu1"] = xu.UgridDataArray(xr.DataArray(data=self.nu1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["nu2"] = xu.UgridDataArray(xr.DataArray(data=self.nu2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["nd"]  = xu.UgridDataArray(xr.DataArray(data=self.nd, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["nd1"] = xu.UgridDataArray(xr.DataArray(data=self.nd1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.data["nd2"] = xu.UgridDataArray(xr.DataArray(data=self.nd2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)

        # Get rid of temporary arrays
        self._clear_temporary_arrays()

    def _clear_temporary_arrays(self):
        pass

    def _find_boundary_cells(self, varname):

        mu = self.data["mu"].values[:]
        mu1 = self.data["mu1"].values[:] - 1
        mu2 = self.data["mu2"].values[:] - 1
        nu = self.data["nu"].values[:]
        nu1 = self.data["nu1"].values[:] - 1
        nu2 = self.data["nu2"].values[:] - 1
        md = self.data["md"].values[:]
        md1 = self.data["md1"].values[:] - 1
        md2 = self.data["md2"].values[:] - 1
        nd = self.data["nd"].values[:]
        nd1 = self.data["nd1"].values[:] - 1
        nd2 = self.data["nd2"].values[:] - 1

        # mask = self.data["msk"].values[:]
        mask = self.data[varname].values[:] # TL: can be both sfincs or snapwave msk        

        bounds = np.zeros(self.nr_cells, dtype=bool)

        # Check left neighbors
        left_coarser = md <= 0 # Coarser or equal to the left
        left_finer1 = (md1 >= 0) & (mask[md1] == 0) # Cell to the left and inactive
        left_finer2 = (md2 >= 0) & (mask[md2] == 0) # (Finer) cell to the left and inactive
        bounds |= (
            (left_coarser & (left_finer1)) | # cell to the left is inactive
            (~left_coarser & (left_finer1 | left_finer2)) # one of the finer cells to the left is inactive
        )

        # Check right neighbors
        right_coarser = mu <= 0
        right_finer1 = (mu1 >= 0) & (mask[mu1] == 0)
        right_finer2 = (mu2 >= 0) & (mask[mu2] == 0)
        bounds |= (
            (right_coarser & (right_finer1 | right_finer2)) |
            (~right_coarser & (right_finer1 | right_finer2))
        )

        # Check bottom neighbors
        below_coarser = nd <= 0
        below_finer1 = (nd1 >= 0) & (mask[nd1] == 0)
        below_finer2 = (nd2 >= 0) & (mask[nd2] == 0)
        bounds |= (
            (below_coarser & (below_finer1 | below_finer2)) |
            (~below_coarser & (below_finer1 | below_finer2))
        )

        # Check top neighbors
        above_coarser = nu <= 0
        above_finer1 = (nu1 >= 0) & (mask[nu1] == 0)
        above_finer2 = (nu2 >= 0) & (mask[nu2] == 0)
        bounds |= (
            (above_coarser & (above_finer1 | above_finer2)) |
            (~above_coarser & (above_finer1 | above_finer2))
        )

        # Handling boundary cells
        bounds[md1 == -1] = True  # Left boundary
        bounds[mu1 == -1] = True  # Right boundary
        bounds[nd1 == -1] = True  # Bottom boundary
        bounds[nu1 == -1] = True  # Top boundary

        # Get rid of the inactive boundary cells that were added
        # in the previous step
        bounds[mask == 0] = False

        return bounds

    def _find_boundary_cells_msk2(self):

        mu = self.data["mu"].values[:]
        mu1 = self.data["mu1"].values[:] - 1
        mu2 = self.data["mu2"].values[:] - 1
        nu = self.data["nu"].values[:]
        nu1 = self.data["nu1"].values[:] - 1
        nu2 = self.data["nu2"].values[:] - 1
        md = self.data["md"].values[:]
        md1 = self.data["md1"].values[:] - 1
        md2 = self.data["md2"].values[:] - 1
        nd = self.data["nd"].values[:]
        nd1 = self.data["nd1"].values[:] - 1
        nd2 = self.data["nd2"].values[:] - 1

        mask = self.bounds_msk2.values[:]
        
        bounds = np.zeros(self.nr_cells, dtype=bool)

        # When upper and right are msk=2
        above_coarser = nu <= 0
        right_coarser = mu <= 0        
        above_finer1 = (nu1 >= 0) & (mask[nu1] == 2)
        right_finer1 = (mu1 >= 0) & (mask[mu1] == 2)
        bounds |= (
            ((mask == 1) & (above_coarser & right_coarser) & (above_finer1 & right_finer1)) #|
            # (~below_coarser & (below_finer1 | below_finer2))
        )

        # When upper and left are msk=2
        above_coarser = nu <= 0
        left_coarser = md <= 0        
        above_finer1 = (nu1 >= 0) & (mask[nu1] == 2)
        left_finer1 = (md1 >= 0) & (mask[md1] == 2)
        bounds |= (
            ((mask == 1) & (above_coarser & left_coarser) & (above_finer1 & left_finer1)) #|
            # (~below_coarser & (below_finer1 | below_finer2))
        )

        # When lower and left are msk=2
        lower_coarser = nd <= 0
        left_coarser = md <= 0        
        below_finer1 = (nd1 >= 0) & (mask[nd1] == 2)
        left_finer1 = (md1 >= 0) & (mask[md1] == 2)
        bounds |= (
            ((mask == 1) & (lower_coarser & left_coarser) & (below_finer1 & left_finer1)) #|
            # (~below_coarser & (below_finer1 | below_finer2))
        )

        # When lower and right are msk=2
        lower_coarser = nd <= 0
        right_coarser = mu <= 0        
        below_finer1 = (nd1 >= 0) & (mask[nd1] == 2)
        right_finer1 = (mu1 >= 0) & (mask[mu1] == 2)
        bounds |= (
            ((mask == 1) & (lower_coarser & right_coarser) & (below_finer1 & right_finer1)) #|
            # (~below_coarser & (below_finer1 | below_finer2))
        )    

        # # Handling boundary cells
        # bounds[md1 == -1] = True  # Left boundary
        # bounds[mu1 == -1] = True  # Right boundary
        # bounds[nd1 == -1] = True  # Bottom boundary
        # bounds[nu1 == -1] = True  # Top boundary

        # # Get rid of the inactive boundary cells that were added
        # # in the previous step
        # bounds[mask == 0] = False

        return bounds
 
    def _find_lower_level_neighbors(self, ind_ref, ilev):
        # ind_ref are the indices of the cells that need to be refined

        n = self.n[ind_ref]
        m = self.m[ind_ref]

        n_odd = np.where(odd(n))
        m_odd = np.where(odd(m))
        n_even = np.where(even(n))
        m_even = np.where(even(m))
        
        ill   = np.intersect1d(n_even, m_even)
        iul   = np.intersect1d(n_odd, m_even)
        ilr   = np.intersect1d(n_even, m_odd)
        iur   = np.intersect1d(n_odd, m_odd)

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

        nmax = self.nmax * 2**(ilev - 1) + 1

        n_nbr = n_nbr.flatten()
        m_nbr = m_nbr.flatten()
        nm_nbr = m_nbr * nmax + n_nbr
        nm_nbr = np.sort(np.unique(nm_nbr, return_index=False))

        # Actual cells in the coarser level 
        n_level = self.n[self.ifirst[ilev - 1]:self.ilast[ilev - 1] + 1]
        m_level = self.m[self.ifirst[ilev - 1]:self.ilast[ilev - 1] + 1]
        nm_level = m_level * nmax + n_level

        # Find  
        ind_nbr = binary_search(nm_level, nm_nbr)
        ind_nbr = ind_nbr[ind_nbr>=0]

        if np.any(ind_nbr):
            ind_nbr += self.ifirst[ilev - 1]

        return ind_nbr

## Internal functions

def odd(num):
    return np.mod(num, 2) == 1

def even(num):
    return np.mod(num, 2) == 0

def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)

def binary_search(val_array, vals):    
    indx = np.searchsorted(val_array, vals) # ind is size of vals 
    not_ok = np.where(indx==len(val_array))[0] # size of vals, points that are out of bounds
    indx[np.where(indx==len(val_array))[0]] = 0 # Set to zero to avoid out of bounds error
    is_ok = np.where(val_array[indx] == vals)[0] # size of vals
    indices = np.zeros(len(vals), dtype=int) - 1
    indices[is_ok] = indx[is_ok]
    indices[not_ok] = -1
    return indices

def gdf2list(gdf_in):
   gdf_out = []
   for feature in gdf_in.iterfeatures():
      gdf_out.append(gpd.GeoDataFrame.from_features([feature]))
   return gdf_out    