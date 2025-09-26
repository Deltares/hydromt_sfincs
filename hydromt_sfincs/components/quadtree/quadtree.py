import logging
import os
import gc
from os.path import isfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import shapely
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import MeshComponent
from hydromt.model.processes.grid import create_grid_from_region

from hydromt_sfincs.utils import make_regular_grid, partition_quadtree
from hydromt_sfincs.workflows.merge import merge_multi_dataarrays, merge_multi_dataarrays_on_mesh
from .quadtree_builder import build_quadtree_xugrid, cut_inactive_cells

# optional dependency
try:
    import datashader.transfer_functions as tf
    from datashader import Canvas
    from datashader.utils import export_image

    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(__name__)

_QT_MAPS = ["vol"]


class SfincsQuadtreeGrid(MeshComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.nc"
        self._data: xu.UgridDataset = None
        self.version = 0
        self.datashader_dataframe = pd.DataFrame()

        super().__init__(
            model=model,
        )

    # NOTE @data and @initialize are inherited from the MeshComponent

    @property
    def crs(self) -> CRS:
        """Return the coordinate reference system of the regular grid."""
        if self.data.grid.crs is not None:
            return self.data.grid.crs
        else:
            raise ValueError("No CRS defined for the quadtree grid.")

    @property
    def face_coordinates(self):
        if self.data is None:
            return None
        xy = self.data.grid.face_coordinates
        return xy[:, 0], xy[:, 1]

    @property
    def exterior(self):
        if self.data is None:
            return gpd.GeoDataFrame()
        indx = self.data.grid.edge_node_connectivity[self.data.grid.exterior_edges, :]
        x = self.data.grid.node_x[indx]
        y = self.data.grid.node_y[indx]

        # Make linestrings from numpy arrays x and y
        linestrings = [
            shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))
        ]
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

    @property
    def mask(self) -> xu.UgridDataArray:
        """Return the mask of the quadtree grid."""
        if "mask" in self.data:
            da_mask = self.data["mask"]
        else:
            da_mask = self.empty_mask
        return da_mask

    def read(
        self, filename: Union[str, Path] = "sfincs.nc", data_vars: List[dict] = None
    ):
        """Reads a quadtree netcdf file and stores it in the QuadtreeGrid object.

        Parameters
        ----------
        file_name : str or Path, optional
            Path to the netcdf file to read, by default "sfincs.nc".
        data_vars : List[dict], optional
            List of dictionaries with variable names and file names to read additional variables,
            by default None. Each dictionary should have keys "variable" and "file_name", e.g.:
            data_vars = [{"variable":"vol", "file_name":"storage_volume.nc"}]
        """

        # check if in read mode and initialize grid
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if qtrfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "qtrfile", value=filename
        )
        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined
            return

        # Check if qtr file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(f"Quadtree grid file not found: {abs_file_path}")

        # load dataset and set CRS
        with xu.load_dataset(abs_file_path) as ds:
            ds.grid.set_crs(CRS.from_wkt(ds["crs"].crs_wkt))

            # rename variables to match Python conventions
            ds = ds.rename({"z": "dep"}) if "z" in ds else ds
            # and for backwards compatibility msk (old) -> mask (new)
            ds = ds.rename({"msk": "mask"}) if "msk" in ds else ds
            ds = (
                ds.rename({"snapwave_msk": "snapwave_mask"})
                if "snapwave_msk" in ds
                else ds
            )

            # store attributes
            self.nr_cells = ds.sizes["mesh2d_nFaces"]
            for key, value in ds.attrs.items():
                setattr(self, key, value)

            self._data = ds

        # self.get_exterior() # FIXME - TL: why is this needed in cht_sfincs? > also, is now a property

        # check which seperate data variables should be read
        if data_vars is None:
            data_vars = _QT_MAPS
        elif isinstance(data_vars, str):
            data_vars = list(data_vars)
        variables = []
        for var in data_vars:
            fn_var = self.model.config.get(
                f"{var}file", fallback=f"{var}.nc", abs_path=True
            )
            if isfile(fn_var):
                variables.append({"variable": var, "file_name": fn_var})

        if len(variables) > 0:
            for var in variables:
                try:
                    with xu.load_dataset(var["file_name"]) as ds:
                        self._data[var["variable"]] = ds[var["variable"]]
                except Exception as e:
                    logger.error(f"Error reading variable {var['variable']}: {e}")
                    continue

    def write(
        self, filename: Union[str, Path] = "sfincs.nc", data_vars: List[dict] = None
    ):
        """Writes a quadtree SFINCS netcdf file.

        Parameters
        ----------
        filename : str or Path, optional
            Path to the netcdf file to write, by default "sfincs.nc".
        data_vars : List[dict], optional
            List of dictionaries with variable names and file names to write additional variables,
            by default None. Each dictionary should have keys "variable" and "file_name", e.g.:
            data_vars = [{"variable":"vol", "file_name":"storage_volume.nc"}]
        """

        # TODO do we want to cut inactive cells here? Or already when creating the mask?

        attrs = self.data.attrs
        ds = self.data.ugrid.to_dataset()
        # FIXME set the CRS manually, since when is this needed?
        ds["crs"] = self.crs.to_epsg()
        ds["crs"].attrs = self.crs.to_cf()

        # certain variables are stored as individual netcdfs because they might change between scnearios;
        # in Python we keep everything in the same object so they are splitted here
        # check which data variables should be written separately
        if data_vars is None:
            data_vars = _QT_MAPS
        elif isinstance(data_vars, str):
            data_vars = list(data_vars)
        variables = []
        for var in data_vars:
            fn_var = self.model.config.get(f"{var}file", abs_path=True)
            if fn_var is not None:
                variables.append({"variable": var, "file_name": fn_var})

        if len(variables) > 0:
            for var in variables:
                try:
                    # get the single variable and convert to dataset
                    # NOTE this allows to read as a standalone file with spatial metadata
                    ds_var = self.data[
                        [var["variable"], "mesh2d_node_x", "mesh2d_node_y"]
                    ].ugrid.to_dataset()
                    ds_var.to_netcdf(var["file_name"])
                    # drop the variable from ds
                    ds = ds.drop_vars(var["variable"])
                except Exception as e:
                    logger.error(f"Error writing variable {var['variable']}: {e}")
                    continue

        # RENAME TO FORTRAN CONVENTION
        ds = ds.rename({"dep": "z"}) if "dep" in ds else ds

        # Get absolute file name and set it in config if bndfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "qtrfile", value=filename, default="sfincs.nc"
        )

        # And write the file
        ds.attrs = attrs
        ds.to_netcdf(abs_file_path)
        ds.close()

    def create(
        self,
        x0: float,
        y0: float,
        nmax: int,
        mmax: int,
        dx: float,
        dy: float,
        rotation: float,
        epsg: int,
        refinement_polygons: Optional[gpd.GeoDataFrame] = None,
        datasets_dep: List[List[dict]] = None,  
    ):
        """Build the Quadtree grid.

        Parameters
        ----------
        x0 : float
            x-coordinate of the lower left corner of the grid.
        y0 : float
            y-coordinate of the lower left corner of the grid.
        nmax : int
            Maximum number of cells in x-direction.
        mmax : int
            Maximum number of cells in y-direction.
        dx : float
            Cell size in x-direction.
        dy : float
            Cell size in y-direction.
        rotation : float
            Rotation angle of the grid in degrees.
        epsg : int
            EPSG code of the coordinate reference system.
        refinement_polygons : gpd.GeoDataFrame, optional
            GeoDataFrame with polygons that define areas where the grid should be refined.
        datasets_dep : List[List[dict]], optional
            List of lists of dictionaries with variable names and dataset names to use for depth
        """

        # Clear datashader dataframes
        self.clear_datashader_dataframe()
        self.model.quadtree_mask.clear_datashader_dataframe()

        # Set grid type and crs in model
        self.model.grid_type = "quadtree"
        crs = CRS.from_epsg(epsg) 

        datasets_dep_per_level = []
        if datasets_dep is not None:
            # Create grid without refinement first
            # NOTE this is used to determine model properties while parsing datasets_dep
            self._data = make_regular_grid(
                x0, y0,
                dx, dy,
                nmax, mmax,
                rotation,
                crs,
                make_ugrid=True
            )
            # Parse the datasets for all refinement levels
            res = dx  # coarsest level
            levels =  set(refinement_polygons["refinement_level"].unique())
            # convert to meters if geographic
            if crs.is_geographic:
                res = res * 111111.0
            # append parsed datasets per level
            for lev in range(max(levels)):
                # compute resolution at level
                res_level = res / (2**lev)
                datasets_dep_per_level.append(
                    self.model._parse_datasets_dep(datasets_dep, res=res_level)
                )

        # Build the quadtree grid
        self._data = build_quadtree_xugrid(
            x0,
            y0,
            nmax,
            mmax,
            dx,
            dy,
            rotation,
            crs,
            refinement_polygons=refinement_polygons,
            datasets_dep=datasets_dep_per_level,
        )

        # self.get_exterior() # FIXME - TL: why is this needed in cht_sfincs? > also, is now a property

    def create_from_region(
        self,
        region: dict,
        res: float = 100,
        crs: Union[str, int] = "utm",
        rotated: bool = False,
        hydrography_fn: str = None,
        basin_index_fn: str = None,
        align: bool = True,
        dec_origin: int = 0,
        dec_rotation: int = 3,
        refinement_polygons: Optional[gpd.GeoDataFrame] = None,
        datasets_dep: List[List[dict]] = None,
    ):
        """Setup a quadtree grid from a region.

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:

            * {'bbox': [xmin, ymin, xmax, ymax]}
            * {'geom': 'path/to/polygon_geometry'}

            Note: For the 'bbox' option the coordinates need to be provided in WG84/EPSG:4326.

            For a complete overview of all region options,
            see :py:func:`hydromt.workflows.basin_mask.parse_region`
        res : float, optional
            grid resolution, by default 100 m
        crs : Union[str, int], optional
            coordinate reference system of the grid
            if "utm" (default) the best UTM zone is selected
            else a pyproj crs string or epsg code (int) can be provided
        rotated : bool, optional
            if True, a minimum rotated rectangular grid is fitted around the region, by default False
        hydrography_fn : str
            Name of data source for hydrography data.
        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.
        align : bool, optional
            If True (default), align target transform to resolution.
            Note that this has only been implemented for non-rotated grids.
        dec_origin : int, optional
            number of decimals to round the origin coordinates, by default 0
        dec_rotation : int, optional
            number of decimals to round the rotation angle, by default 3
        refinement_polygons : gpd.GeoDataFrame, optional
            GeoDataFrame with polygons that define areas where the grid should be refined.
        datasets_dep : List[List[dict]], optional
            List of lists of dictionaries with variable names and dataset names to use for depth            

        See Also
        --------
        hydromt.workflows.basin_mask.parse_region
        """

        ds = create_grid_from_region(
            region=region,
            data_catalog=self.model.data_catalog,
            res=res,
            crs=crs,
            region_crs=4326,
            rotated=rotated,
            hydrography_path=hydrography_fn,
            basin_index_path=basin_index_fn,
            add_mask=False,
            align=align,
            dec_origin=dec_origin,
            dec_rotation=dec_rotation,
        )

        # check for y-resolution
        # TODO discuss with hydrom-core if this behavior is desired
        if ds.raster.res[1] < 0:
            ds = ds.raster.flipud()

        # derive grid properties from grid
        nmax, mmax = ds.raster.shape
        dx, dy = ds.raster.res
        x0, y0 = ds.raster.origin
        rotation = ds.raster.rotation
        epsg = ds.raster.crs.to_epsg()

        # now parse everything to the quadtree create method
        self.create(
            x0=x0,
            y0=y0,
            nmax=nmax,
            mmax=mmax,
            dx=dx,
            dy=dy,
            rotation=rotation,
            epsg=epsg,
            refinement_polygons=refinement_polygons,
            datasets_dep=datasets_dep,
        )            

    def interpolate_bathymetry(self, x, y, z, method="linear"):
        """x, y, and z are numpy arrays with coordinates and bathymetry values"""
        xy = self.data.grid.face_coordinates
        # zz = np.full(self.nr_cells, np.nan)
        xz = xy[:, 0]
        yz = xy[:, 1]
        zz = interp2(x, y, z, xz, yz, method=method)
        ugrid2d = self.data.grid
        self.data["z"] = xu.UgridDataArray(xr.DataArray(data=zz, dims=[ugrid2d.face_dimension]), ugrid2d)

    def set_uniform_bathymetry(self, zb):
        self.data["z"][:] = zb

    def set_bathymetry(self, datasets_dep, zmin=-1.0e9, zmax=1.0e9, quiet=True):
        
        # Number of refinement levels
        nlev = self.data.attrs["nr_levels"]
        # Cell centre coordinates
        xy = self.data.grid.face_coordinates
        # Get number of cells
        nr_cells = len(xy)
        # Initialize bathymetry array
        zz = np.full(nr_cells, np.nan)
        # cell size of coarsest level
        dx = self.data.attrs["dx"]

        # Determine first indices and number of cells per refinement level
        # This is also done when the grid is built, but that information is not stored
        ifirst = np.zeros(nlev, dtype=int)
        ilast = np.zeros(nlev, dtype=int)
        level = self.data["level"].values[:] - 1 # 0-based
        for ilev in range(0, nlev):
            # Find index of first cell with this level
            ifirst[ilev] = np.where(level == ilev)[0][0]
            # Find index of last cell with this level
            if ilev < nlev - 1:
                ilast[ilev] = np.where(level == ilev + 1)[0][0] - 1
            else:
                ilast[ilev] = nr_cells - 1

        # convert to meters if geographic
        if self.model.crs.is_geographic:
            dx = dx * 111111.0
        # append parsed datasets per level
        datasets_dep_per_level = []
        for ilev in range(nlev):
            # compute resolution at level
            res_level = dx / (2**ilev)
            datasets_dep_per_level.append(
                self.model._parse_datasets_dep(datasets_dep, res=res_level)
            )

        # get m and n indices
        n = self.data["n"]
        m = self.data["m"]

        # Loop through all levels
        for ilev in range(nlev):

            if not quiet:
                print("Processing bathymetry level " + str(ilev + 1) + " of " + str(nlev) + " ...")

            # First and last cell indices in this level            
            i0 = ifirst[ilev]
            i1 = ilast[ilev]
            
            # Make blocks of cells in this level only
            cell_indices_in_level = np.arange(i0, i1 + 1, dtype=int)
                  
            xz  = xy[cell_indices_in_level, 0]
            yz  = xy[cell_indices_in_level, 1]
            dxmin = dx / 2**ilev

            da_like = make_regular_grid(
                x0=self.data.attrs["x0"],
                y0=self.data.attrs["y0"],
                dx=dxmin,
                dy=dxmin,
                mmax=m[i0:i1+1].max().values + 1,
                nmax=n[i0:i1+1].max().values + 1,
                rotation=self.data.attrs["rotation"],
                crs=self.model.crs,
                mmin=m[i0:i1+1].min().values,
                nmin=n[i0:i1+1].min().values,
                make_ugrid=False,
            )

            da_dep = merge_multi_dataarrays(
                da_list=datasets_dep_per_level[ilev],
                da_like=da_like,
                # buffer_cells=buffer_cells,
                # interp_method=interp_method,
                logger=logger,
            )

            # Flatten n and m indices of cells in this level
            n_flat = n[cell_indices_in_level].values
            m_flat = m[cell_indices_in_level].values

            # Find integer indices along the coordinate arrays
            idx_y = np.searchsorted(da_dep.n.values, n_flat)
            idx_x = np.searchsorted(da_dep.m.values, m_flat)

            # Select the values
            zgl = da_dep.values[idx_y, idx_x]

            # zgl = bathymetry_database.get_bathymetry_on_points(xz,
            #                                                    yz,
            #                                                    dxmin,
            #                                                    self.model.crs,
            #                                                    bathymetry_sets)
            
            # Limit zgl to zmin and zmax
            zgl = np.maximum(zgl, zmin)
            zgl = np.minimum(zgl, zmax)

            zz[cell_indices_in_level] = zgl

        ugrid2d = self.data.grid
        self.data["z"] = xu.UgridDataArray(xr.DataArray(data=zz, dims=[ugrid2d.face_dimension]), ugrid2d)

    @hydromt_step
    def create_elevation(
        self,
        datasets_dep: List[dict],
        partition_by_level: bool = True,
        partition_in_blocks: bool = False,
        nrmax: int = 2000,  # not in list
        buffer_cells: int = 0,  # not in list
        interp_method: str = "linear",  # used for buffer cells only
    ):
        """Interpolate topobathy (dep) data to the model grid.

        Adds model grid layers:

        * **dep**: combined elevation/bathymetry [m+ref]

        Parameters
        ----------
        datasets_dep : List[dict]
            List of dictionaries with topobathy data, each containing a dataset name or Path (elevtn) and optional merge arguments e.g.:
            [{'elevtn': merit_hydro, 'zmin': 0.01}, {'elevtn': gebco, 'offset': 0, 'merge_method': 'first', 'reproj_method': 'bilinear'}]
            For a complete overview of all merge options, see :py:func:`hydromt.workflows.merge_multi_dataarrays`
        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels, by default 0
        interp_method : str, optional
            Interpolation method used to fill the buffer cells , by default "linear"
        """

        # get resolution and number of level
        res = self.data.attrs["dx"]
        nrlevels = self.data.attrs["nr_levels"]

        # convert to meters if geographic
        if self.model.crs.is_geographic:
            res = res * 111111.0
        # append parsed datasets per level
        datasets_dep_per_level = []
        for ilev in range(nrlevels):
            # compute resolution at level
            res_level = res / (2**ilev)
            datasets_dep_per_level.append(
                self.model._parse_datasets_dep(datasets_dep, res=res_level)
            )

        # check if partitions are already defined
        if partition_by_level or partition_in_blocks:
            # create partitions if not already done
            if not hasattr(self, "partitions"):
                t0 = time.time()
                self.partitions = partition_quadtree(
                    quadtree=self.data, 
                    partition_by_level=partition_by_level, 
                    partition_in_blocks=partition_in_blocks, 
                    nrmax=nrmax, 
                    logger=logger
                )
                t1 = time.time()
                logger.debug(f"Partitioning quadtree took {t1-t0:.2f} s.")
            count = 0
            for partition in self.partitions:
                # time each partition
                t0 = time.time()
                ilev = partition.level.max().values - 1
                # merge multiple datasets on mesh
                uda = merge_multi_dataarrays_on_mesh(
                    da_list=datasets_dep_per_level[ilev],
                    mesh2d=partition.grid,
                )
                partition["z"] = uda
                t1 = time.time()
                logger.debug(
                    f"Dep merged for partition {count} at level {ilev} in {t1-t0:.2f} s."
                )
                count += 1
                gc.collect()
            t0 = time.time()
            merged = xu.merge_partitions(self.partitions)
            t1 = time.time()
            logger.debug(f"Merging partitions took {t1-t0:.2f} s.")
            reordered = merged.ugrid.reindex_like(self.data.grid)
            # add data to grid
            self.data["z"] = reordered["z"]
        else:
            t0 = time.time()
            # when not partitioned, use the full grid with the highest resolution data
            uda = merge_multi_dataarrays_on_mesh(
                da_list=datasets_dep_per_level[-1], mesh2d=self.data.grid, logger=logger
            )
            t1 = time.time()
            logger.debug(f"Merging dep took {t1-t0:.2f} s.")
            # add data to grid
            self.data["z"] = uda        


    def cut_inactive_cells(self):
        # Clear datashader dataframes (new ones will be created when needed by map_overlay methods)
        self.clear_datashader_dataframe()
        self.model.quadtree_mask.clear_datashader_dataframe()
        # Cut inactive cells
        self._data = cut_inactive_cells(self.data)
        # self.get_exterior() # FIXME - TL: why is this needed in cht_sfincs? > also, is now a property

    def snap_to_grid(self, polyline):
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        # If geographic coordinates, set max_snap_distance to 0.1 degrees
        if self.model.crs.is_geographic:
            max_snap_distance = 1.0e-6
        else:
            max_snap_distance = 0.1

        geom_list = []
        for _, line in polyline.iterrows():
            geom = line["geometry"]
            if geom.geom_type == "LineString":
                geom_list.append(geom)
        gdf = gpd.GeoDataFrame({"geometry": geom_list})
        _, snapped_gdf = xu.snap_to_grid(
            gdf, self.data.grid, max_snap_distance=max_snap_distance
        )
        snapped_gdf = snapped_gdf.set_crs(self.crs)
        return snapped_gdf

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):
        """Create a map overlay of the grid

        Parameters
        ----------
        file_name : str | Path
            File name of the map overlay
        xlim : list, optional
            x-axis limits (longitude)
        ylim : list, optional
            y-axis limits (latitude)
        color : str, optional
            Color of the grid lines
        width : int, optional
            Width of the map overlay in pixels

        Returns
        -------
        bool
            True if the map overlay was created successfully, False otherwise
        """
        # TODO: xlim and ylim should not be optional and be called lonlim and latlim or just give bbox

        # check if datashader is available
        if not HAS_DATASHADER:
            logger.warning("Datashader is not available. Please install datashader.")
            return False

        if self.data is None:
            # No grid (yet)
            return False

        try:
            # Check if datashader dataframe is empty (maybe it was not made yet, or it was cleared)
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)
            cvs = Canvas(
                x_range=xlim, y_range=ylim, plot_height=height, plot_width=width
            )
            agg = cvs.line(
                self.datashader_dataframe, x=["x1", "x2"], y=["y1", "y2"], axis=1
            )
            img = tf.shade(agg)
            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True
        except Exception:
            return False

    def get_indices_at_points(self, x, y):
        # x and y are 2D arrays of coordinates (x, y) in the same projection as the model
        # if x is a float, convert to 2D array
        if np.ndim(x) == 0:
            x = np.array([[x]])
        if np.ndim(y) == 0:
            y = np.array([[y]])

        x0 = self.data.attrs["x0"]
        y0 = self.data.attrs["y0"]
        dx = self.data.attrs["dx"]
        dy = self.data.attrs["dy"]
        nmax = self.data.attrs["nmax"]
        mmax = self.data.attrs["mmax"]
        rotation = self.data.attrs["rotation"]
        nr_refinement_levels = self.data.attrs["nr_levels"]

        nr_cells = len(self.data["level"])

        cosrot = np.cos(-rotation * np.pi / 180)
        sinrot = np.sin(-rotation * np.pi / 180)

        # Now rotate around origin of SFINCS model
        x00 = x - x0
        y00 = y - y0
        xg = x00 * cosrot - y00 * sinrot
        yg = x00 * sinrot + y00 * cosrot

        # Find index of first cell in each level
        if not hasattr(self.data, "ifirst"):
            ifirst = np.zeros(nr_refinement_levels, dtype=int)
            for ilev in range(0, nr_refinement_levels):
                # Find index of first cell with this level
                ifirst[ilev] = np.where(self.data["level"].to_numpy()[:] == ilev + 1)
                [0][0]
            self.ifirst = ifirst

        ifirst = self.ifirst

        i0_lev = []
        i1_lev = []
        nmax_lev = []
        mmax_lev = []
        nm_lev = []

        for level in range(nr_refinement_levels):
            i0 = ifirst[level]
            if level < nr_refinement_levels - 1:
                i1 = ifirst[level + 1]
            else:
                i1 = nr_cells
            i0_lev.append(i0)
            i1_lev.append(i1)
            nmax_lev.append(np.amax(self.data["n"].to_numpy()[i0:i1]) + 1)
            mmax_lev.append(np.amax(self.data["m"].to_numpy()[i0:i1]) + 1)
            nn = self.data["n"].to_numpy()[i0:i1] - 1
            mm = self.data["m"].to_numpy()[i0:i1] - 1
            nm_lev.append(mm * nmax_lev[level] + nn)

        # Initialize index array
        indx = np.full(np.shape(x), -999, dtype=np.int32)

        for ilev in range(nr_refinement_levels):
            nmax = nmax_lev[ilev]
            mmax = mmax_lev[ilev]
            i0 = i0_lev[ilev]
            i1 = i1_lev[ilev]
            dxr = dx / 2**ilev
            dyr = dy / 2**ilev
            iind = np.floor(xg / dxr).astype(int)
            jind = np.floor(yg / dyr).astype(int)
            # Now check whether this cell exists on this level
            ind = iind * nmax + jind
            ind[iind < 0] = -999
            ind[jind < 0] = -999
            ind[iind >= mmax] = -999
            ind[jind >= nmax] = -999

            ingrid = np.isin(
                ind, nm_lev[ilev], assume_unique=False
            )  # return boolean for each pixel that falls inside a grid cell
            incell = np.where(
                ingrid
            )  # tuple of arrays of pixel indices that fall in a cell

            if incell[0].size > 0:
                # Now find the cell indices
                try:
                    cell_indices = (
                        binary_search(nm_lev[ilev], ind[incell[0], incell[1]])
                        + i0_lev[ilev]
                    )
                    indx[incell[0], incell[1]] = cell_indices
                except Exception as e:
                    print("Error in binary search: ", str(e))
                    pass

        return indx

    # Internal functions
    def get_datashader_dataframe(self):
        """Creates a dataframe with line elements for datashader"""
        x1 = self.data.grid.edge_node_coordinates[:, 0, 0]
        x2 = self.data.grid.edge_node_coordinates[:, 1, 0]
        y1 = self.data.grid.edge_node_coordinates[:, 0, 1]
        y2 = self.data.grid.edge_node_coordinates[:, 1, 1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x1) > 180.0 or np.max(x2) > 180.0:
                cross_dateline = True
        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        if cross_dateline:
            x1[x1 < 0] += 40075016.68557849
            x2[x2 < 0] += 40075016.68557849
        self.datashader_dataframe = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def clear_datashader_dataframe(self):
        """Clears the datashader dataframe"""
        self.datashader_dataframe = pd.DataFrame()

    def make_topobathy_cog(
        self, filename, bathymetry_sets, bathymetry_database=None, dx=10.0
    ):
        """Make a COG file with topobathy. Now only works for projected coordinates. This always make the topobathy COG in the same projection as the model."""

        # Get the bounds of the grid
        bounds = self.bounds()

        x0 = bounds[0]
        y0 = bounds[1]
        x1 = bounds[2]
        y1 = bounds[3]

        # Round up and down to nearest dx
        x0 = x0 - (x0 % dx)
        x1 = x1 + (dx - x1 % dx)
        y0 = y0 - (y0 % dx)
        y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx
        zz = np.empty(
            (
                len(yy),
                len(xx),
            ),
            dtype=np.float32,
        )

        xx, yy = np.meshgrid(xx, yy)
        zz = bathymetry_database.get_bathymetry_on_points(
            xx, yy, dx, self.model.crs, bathymetry_sets
        )

        # And now to cog (use -999 as the nodata value)
        with rasterio.open(
            filename,
            "w",
            driver="COG",
            height=zz.shape[0],
            width=zz.shape[1],
            count=1,
            dtype=zz.dtype,
            crs=self.model.crs,
            transform=from_origin(x0, y1, dx, dx),
            nodata=-999.0,
        ) as dst:
            dst.write(zz, 1)

    def make_index_cog(self, filename, filename_topobathy):
        # def make_index_cog(self, filename, dx=10.0):
        """Make a COG file with indices of the quadtree grid cells."""

        # Read coordinates from topobathy file
        with rasterio.open(filename_topobathy) as src:
            # Get the bounds of the grid
            bounds = src.bounds
            dx = src.res[0]
            # Get the CRS of the grid
            self.model.crs = src.crs
            # Get the nodata value
            nodata = src.nodata
            # Get the transform of the grid
            transform = src.transform
            # Get the width and height of the grid
            width = src.width
            height = src.height

        # Now create numpy arrays with the coordinates of geotiff
        # Get the coordinates of the grid
        x0 = bounds.left
        y0 = bounds.bottom
        x1 = bounds.right
        y1 = bounds.top

        # # Round up and down to nearest dx
        # x0 = x0 - (x0 % dx)
        # x1 = x1 + (dx - x1 % dx)
        # y0 = y0 - (y0 % dx)
        # y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx

        nodata = 2147483647

        # # # Get the bounds of the grid
        # # bounds = self.bounds()

        # x0 = bounds[0]
        # y0 = bounds[1]
        # x1 = bounds[2]
        # y1 = bounds[3]

        # # Round up and down to nearest dx
        # x0 = x0 - (x0 % dx)
        # x1 = x1 + (dx - x1 % dx)
        # y0 = y0 - (y0 % dx)
        # y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx
        ii = np.empty(
            (
                len(yy),
                len(xx),
            ),
            dtype=np.uint32,
        )

        # # Create empty ds
        # ds = xr.Dataset(
        #     {
        #         "index": (["y", "x"], ii),
        #     },
        #     coords={
        #         "x": xx,
        #         "y": yy,
        #     },
        # )
        # # Set no data value in ds
        # ds["index"].attrs["_FillValue"] = nodata

        # Go through refinement levels in grid
        xx, yy = np.meshgrid(xx, yy)
        indices = self.get_indices_at_points(xx, yy)
        indices[np.where(indices == -999)] = nodata

        # Fill the array with indices
        ii[:, :] = indices

        # # Write first to netcdf
        # ds.to_netcdf("index.nc")

        # And now to cog (use -999 as the nodata value)
        with rasterio.open(
            filename,
            "w",
            driver="COG",
            height=height,
            width=width,
            count=1,
            dtype=ii.dtype,
            crs=self.model.crs,
            transform=transform,
            nodata=nodata,
            overview_resampling=Resampling.nearest,
        ) as dst:
            dst.write(ii, 1)


def binary_search(val_array, vals):
    indx = np.searchsorted(val_array, vals)  # ind is size of vals
    not_ok = np.where(indx == len(val_array))[
        0
    ]  # size of vals, points that are out of bounds
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