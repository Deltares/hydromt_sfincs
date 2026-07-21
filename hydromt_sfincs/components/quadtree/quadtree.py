"""Quadtree grid component for the SFINCS model.

Defines :class:`SfincsQuadtreeGrid`, a mesh-backed grid that supports
multi-level refinement. Provides I/O, grid construction, mask / bathymetry
handling, point-to-cell lookup, tiled index output for web viewers, and
utilities used by DelftDashboard such as map overlays and grid snapping.
"""

import logging
import os
from os.path import isfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
import shapely

import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import MeshComponent
from hydromt.model.processes.grid import create_grid_from_region

from hydromt_sfincs.utils import make_regular_grid
from hydromt_sfincs.workflows.cog import make_quadtree_index_cog, make_topobathy_cog
from hydromt_sfincs.workflows.map_overlay import MeshOverlay
from hydromt_sfincs.workflows.tiling import (
    create_topobathy_tiles,
    int2png,
    make_index_tiles,
    tile_window,
    write_html,
)
from .quadtree_builder import build_quadtree_xugrid, cut_inactive_cells

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_QT_MAPS = ["manning", "vol", "ini", "infiltration"]


class SfincsQuadtreeGrid(MeshComponent):
    """Quadtree grid component attached to an :class:`SfincsModel`."""

    def __init__(
        self,
        model: "SfincsModel",
    ) -> None:
        """Initialise the component with a back-reference to its parent model."""
        self._filename: str = "sfincs.nc"
        self._data: xu.UgridDataset = None
        self.version = 0
        self._overlay = MeshOverlay()

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
    def face_coordinates(self) -> Optional[tuple]:
        """Return the (x, y) coordinates of the cell face centres.

        Returns
        -------
        tuple of np.ndarray, or None
            A pair ``(x, y)`` of 1-D arrays with cell centre coordinates in
            the model CRS, or ``None`` if no grid has been loaded.
        """
        if self.data is None:
            return None
        xy = self.data.grid.face_coordinates
        return xy[:, 0], xy[:, 1]

    @property
    def exterior(self) -> gpd.GeoDataFrame:
        """Return the outer boundary of the active grid as polygons.

        Returns
        -------
        geopandas.GeoDataFrame
            Polygon geometries in the model CRS, or an empty GeoDataFrame
            if no grid has been loaded.
        """
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
    def empty_mask(self) -> Optional[xu.UgridDataArray]:
        """Return an all-zero mask with the shape of the current grid."""
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

    def read(self, filename: Union[str, Path] = None, data_vars: List[dict] = None):
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
        # xugrid reads mesh2d_crs automatically and sets grid.crs from crs_wkt.
        # For older files, fall back progressively.
        ds = xu.load_dataset(abs_file_path)
        ds.close()
        if ds.grid.crs is None:
            with xr.open_dataset(abs_file_path) as raw:
                ds.grid.set_crs(CRS.from_wkt(raw["crs"].attrs["crs_wkt"]))

        # rename variables to match Python conventions
        # ds = ds.rename({"z": "dep"}) if "z" in ds else ds
        # and for backwards compatibility msk (old) -> mask (new)
        ds = ds.rename({"msk": "mask"}) if "msk" in ds else ds
        ds = (
            ds.rename({"snapwave_msk": "snapwave_mask"}) if "snapwave_msk" in ds else ds
        )

        # store attributes
        self.nr_cells = ds.sizes["mesh2d_nFaces"]
        for key, value in ds.attrs.items():
            setattr(self, key, value)

        self._data = ds

        # Make sure epsg is stored in the config as well
        self.model.config.set("epsg", self.model.crs.to_epsg())

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
                    ds = xu.load_dataset(var["file_name"])
                    ds.close()
                    ds.grid.set_crs(self.model.crs)
                    self.set(ds)
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
        # xugrid writes a 'mesh2d_crs' variable with full CF metadata from
        # pyproj. Add epsg/epsg_code so MDAL can auto-detect the CRS in QGIS.
        epsg = self.crs.to_epsg()
        if "mesh2d_crs" in ds:
            ds["mesh2d_crs"].attrs["epsg"] = epsg
            ds["mesh2d_crs"].attrs["epsg_code"] = f"EPSG:{epsg}"

        # certain variables are stored as individual netcdfs because they might change between scnearios;
        # in Python we keep everything in the same object so they are splitted here
        # check which data variables should be written separately
        if data_vars is None:
            data_vars = _QT_MAPS
        elif isinstance(data_vars, str):
            data_vars = list(data_vars)
        variables = []
        for var in data_vars:
            fn_var = self.model.config.get(f"{var}file", abs_path=True)  # TO DO

            if var == "infiltration":
                fn_var = self.model.config.get(f"{var}_file", abs_path=True)

            if fn_var is not None:
                fn_var.parent.mkdir(parents=True, exist_ok=True)
                variables.append({"variable": var, "file_name": fn_var})

        if len(variables) > 0:
            for var in variables:
                if var["variable"] == "infiltration":
                    # determine which infiltration variables to write based on the infiltration type
                    inftype = self.model.config.get("infiltration_type")
                    (
                        write_vars,
                        remove_vars,
                    ) = self.model.quadtree_infiltration.get_vars_by_infiltration_type(
                        inftype
                    )

                    # Log what is being removed (only if anything to remove)
                    if remove_vars:
                        logger.info(
                            f"Removing unused infiltration variables not matching type '{inftype}': {remove_vars}"
                        )

                    # Drop unwanted variables from dataset BEFORE writing
                    ds = ds.drop_vars(remove_vars, errors="ignore")
                else:
                    write_vars = [var["variable"]]
                try:
                    # get the single variable and convert to dataset
                    # NOTE this allows to read as a standalone file with spatial metadata
                    ds_var = self.data[
                        write_vars + ["mesh2d_node_x", "mesh2d_node_y"]
                    ].ugrid.to_dataset()
                    ds_var.to_netcdf(var["file_name"])
                    # drop the variable from ds
                    ds = ds.drop_vars(write_vars)
                except Exception as e:
                    logger.error(f"Error writing variables {write_vars}: {e}")
                    continue

        # RENAME TO FORTRAN CONVENTION
        ds = ds.rename({"dep": "z"}) if "dep" in ds else ds

        # Get absolute file name and set it in config if bndfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "qtrfile", value=filename, default="sfincs.nc"
        )
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Make sure epsg is stored in the config as well
        self.model.config.set("epsg", self.model.crs.to_epsg())

        # And write the file
        attrs["Conventions"] = "CF-1.8 UGRID-1.0 Deltares-0.10"
        ds.attrs = attrs

        # Cast all int8/uint8 variables to int32 — MDAL rejects the entire mesh
        # when it encounters these small integer types on the face dimension.
        # The SFINCS Fortran kernel reads them into integer*1 arrays via
        # nf90_get_var; NetCDF auto-conversion handles int32→int8 transparently.
        _small_int = (np.int8, np.uint8)
        for var in list(ds.data_vars):
            if ds[var].dtype in _small_int:
                ds[var] = ds[var].astype(np.int32)

        # xugrid's to_dataset() omits units on node coordinates; add them so
        # MDAL can interpret the coordinate system correctly in QGIS.
        geo = self.model.crs.is_geographic
        coord_units = {
            "mesh2d_node_x": "degrees_east" if geo else "m",
            "mesh2d_node_y": "degrees_north" if geo else "m",
        }
        crs_var_name = "mesh2d_crs" if "mesh2d_crs" in ds else "crs"
        for coord, units in coord_units.items():
            if coord in ds:
                if "units" not in ds[coord].attrs:
                    ds[coord].attrs["units"] = units
                ds[coord].attrs["grid_mapping"] = crs_var_name

        ds.to_netcdf(abs_file_path)
        ds.close()

    @hydromt_step
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
        elevation_list: List[List[dict]] = None,
        bathymetry_database: Optional[object] = None,
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
            Cell size in x-direction, needs to be positive.
        dy : float
            Cell size in y-direction, needs to be positive.
        rotation : float
            Rotation angle of the grid in degrees.
        epsg : int
            EPSG code of the coordinate reference system.
        refinement_polygons : gpd.GeoDataFrame, optional
            GeoDataFrame with polygons that define areas where the grid should be refined.
        elevation_list : List[List[dict]], optional
            List of lists of dictionaries with variable names and dataset names to use for depth
        bathymetry_database : object, optional
            Bathymetry database object.
        """

        # Invalidate cached overlays (grid + mask)
        self._overlay.invalidate()
        self.model.quadtree_mask.clear_overlay()

        # Set grid type and crs in model
        self.model._grid_type = "quadtree"
        crs = CRS.from_epsg(epsg)

        elevation_list_per_level = []
        if elevation_list is not None and bathymetry_database is None:
            # Create grid without refinement first
            # NOTE this is used to determine model properties while parsing elevation_list
            self._data = make_regular_grid(
                x0, y0, dx, dy, mmax, nmax, rotation=rotation, crs=crs, make_ugrid=True
            )
            # Parse the datasets for all refinement levels
            res = dx  # coarsest level
            levels = set(refinement_polygons["refinement_level"].unique())
            # convert to meters if geographic
            if crs.is_geographic:
                res = res * 111111.0
            # append parsed datasets per level
            for lev in range(max(levels)):
                # compute resolution at level
                res_level = res / (2**lev)
                elevation_list_per_level.append(
                    self.model._parse_datasets_elevation(elevation_list, res=res_level)
                )
            elevation_list = elevation_list_per_level

        # Build the quadtree grid
        ds = build_quadtree_xugrid(
            x0,
            y0,
            nmax,
            mmax,
            dx,
            dy,
            rotation,
            crs,
            refinement_polygons=refinement_polygons,
            elevation_list=elevation_list,
            bathymetry_database=bathymetry_database,
        )
        # add nFaces coordinates to grid
        ds = xu.UgridDataset(ds.ugrid.to_dataset())
        crs_var = ds["mesh2d_crs"] if "mesh2d_crs" in ds else ds["crs"]
        ds.grid.set_crs(CRS.from_wkt(crs_var.crs_wkt))
        self._data = ds

        # Make sure epsg is stored in the config as well
        self.model.config.set("epsg", self.model.crs.to_epsg())
        # Set 'crsgeo' flag in the config based on whether the CRS is geographic
        self.model.config.set("crsgeo", int(self.model.crs.is_geographic))

    @hydromt_step
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
        elevation_list: List[List[dict]] = None,
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
        elevation_list : List[List[dict]], optional
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
            elevation_list=elevation_list,
        )

    @hydromt_step
    def cut_inactive_cells(self) -> None:
        """Remove cells that are outside the active mask from the grid.

        Also invalidates any cached overlays so rendering picks up the
        new geometry on the next call.
        """
        self._overlay.invalidate()
        self.model.quadtree_mask.clear_overlay()
        self._data = cut_inactive_cells(self.data)
        # self.get_exterior() # FIXME - TL: why is this needed in cht_sfincs? > also, is now a property

    def snap_to_grid(self, polyline: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Snap a set of lines to the edges of the quadtree grid.

        Parameters
        ----------
        polyline : geopandas.GeoDataFrame
            Input lines to snap. Only ``LineString`` geometries are used;
            other geometry types are ignored.

        Returns
        -------
        geopandas.GeoDataFrame
            Snapped lines in the model CRS, or an empty GeoDataFrame if the
            input is empty.
        """
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

    def map_overlay(
        self,
        file_name: Union[str, Path],
        xlim: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        color: str = "black",
        width: int = 800,
    ) -> bool:
        """Render a PNG map overlay of the grid edges.

        One-line wrapper around
        :py:class:`hydromt_sfincs.workflows.map_overlay.MeshOverlay`.
        """
        if self.data is None:
            return False
        return self._overlay.render(
            ugrid=self.data.grid,
            source_crs=self.model.crs,
            file_name=file_name,
            xlim=xlim,
            ylim=ylim,
            color=color,
            width=width,
        )

    def get_indices_at_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the cell index at each (x, y) sample.

        Resolves the quadtree by traversing refinement levels from coarse
        to fine; points that do not fall inside any active cell at any
        level are returned as ``-999``.

        Parameters
        ----------
        x : np.ndarray
            2-D array of x-coordinates in the model CRS (scalars are
            promoted to shape ``(1, 1)``).
        y : np.ndarray
            2-D array of y-coordinates in the model CRS, same shape as
            ``x``.

        Returns
        -------
        np.ndarray
            2-D ``int32`` array with the cell index for each sample, or
            ``-999`` for samples outside the active grid.
        """
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
                levels = self.data["level"].to_numpy()[:]
                indices = np.where(levels == ilev + 1)[0]
                ifirst[ilev] = indices[0]
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
            # return boolean for each pixel that falls inside a grid cell
            ingrid = np.isin(ind, nm_lev[ilev], assume_unique=False)
            # tuple of arrays of pixel indices that fall in a cell
            incell = np.where(ingrid)

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

    def clear_overlay(self) -> None:
        """Invalidate the cached edge-overlay dataframe."""
        self._overlay.invalidate()

    def create_topobathy_cog(
        self,
        filename: Union[str, Path],
        bathymetry_sets: List[dict],
        bathymetry_database: Optional[object] = None,
        dx: float = 10.0,
    ) -> None:
        """Write a COG raster sampling the model topobathy.

        Thin wrapper around
        :py:func:`hydromt_sfincs.workflows.cog.make_topobathy_cog`.

        Parameters
        ----------
        filename : str or Path
            Output COG file path.
        bathymetry_sets : list of dict
            Dataset list passed through to
            ``bathymetry_database.get_bathymetry_on_points``.
        bathymetry_database : object, optional
            Backing bathymetry database providing
            ``get_bathymetry_on_points``. Required for this method to
            produce data.
        dx : float, optional
            Raster resolution in model CRS units, by default ``10.0``.
        """
        make_topobathy_cog(
            quadtree_grid=self,
            filename=filename,
            bathymetry_sets=bathymetry_sets,
            bathymetry_database=bathymetry_database,
            dx=dx,
        )

    def create_index_tiles(
        self,
        root: Union[str, Path],
        region: Optional[gpd.GeoDataFrame] = None,
        zoom_range: Union[int, List[int]] = [0, 13],
        fmt: str = "png",
        write_html_viewer: bool = True,
        max_workers: Optional[int] = None,
        logger: logging.Logger = logger,
    ) -> None:
        """Create webmercator index tiles for this quadtree grid.

        Thin wrapper around
        :py:func:`hydromt_sfincs.workflows.tiling.make_index_tiles`.

        Parameters
        ----------
        root : Union[str, Path]
            Parent directory; tiles land in ``<root>/indices``.
        region : gpd.GeoDataFrame, optional
            Area for which tiles are generated. Defaults to the grid
            exterior.
        zoom_range : Union[int, List[int]], optional
            Range of zoom levels, by default ``[0, 13]``.
        fmt : str, optional
            ``"png"`` (default) or ``"bin"`` (raw int32).
        write_html_viewer : bool, optional
            If True (default) and ``fmt == "png"``, also write an
            ``index.html`` Leaflet viewer alongside the tiles.
        max_workers : int, optional
            Number of worker threads used to render tiles concurrently.
            Defaults to ``os.cpu_count()``. Pass ``1`` to disable
            parallelism.
        """
        make_index_tiles(
            quadtree_grid=self,
            root=root,
            region=region,
            zoom_range=zoom_range,
            fmt=fmt,
            write_html_viewer=write_html_viewer,
            max_workers=max_workers,
            logger=logger,
        )

    def create_topobathy_tiles(
        self,
        root: Union[str, Path],
        elevation_list: Optional[List[dict]] = None,
        region: Optional[gpd.GeoDataFrame] = None,
        index_path: Optional[Union[str, Path]] = None,
        zoom_range: Union[int, List[int]] = [0, 13],
        z_range: List[float] = [-20000.0, 20000.0],
        fmt: str = "bin",
        write_html_viewer: bool = True,
        max_workers: Optional[int] = None,
        logger: logging.Logger = logger,
    ) -> None:
        """Create webmercator topobathy tiles for this quadtree grid.

        Thin wrapper around
        :py:func:`hydromt_sfincs.workflows.tiling.create_topobathy_tiles`.

        Parameters
        ----------
        root : Union[str, Path]
            Parent directory; tiles land in ``<root>/topobathy``.
        elevation_list : List[dict], optional
            Topobathy datasets. Entries may be in DDB name-only format
            (``{"name": ..., "zmin": ..., "zmax": ...}``) or hydromt format
            (with a ``"da"`` DataArray). DDB entries are auto-resolved via
            the model's data catalog. If ``None``, all sources in the
            model's data catalog are used instead.
        region : gpd.GeoDataFrame, optional
            Area for which tiles are generated. Defaults to the grid
            exterior.
        index_path : Union[str, Path], optional
            Directory containing index tiles; if given, topobathy tiles
            are only written where index tiles exist.
        zoom_range : Union[int, List[int]], optional
            Range of zoom levels, by default ``[0, 13]``.
        z_range : List[float], optional
            Valid elevation range; tiles entirely outside are skipped.
        fmt : str, optional
            ``"bin"`` (default), ``"png"``, or ``"tif"``.
        write_html_viewer : bool, optional
            If True (default) and ``fmt == "png"``, also write an
            ``index.html`` Leaflet viewer alongside the tiles.
        max_workers : int, optional
            Number of worker threads used to render tiles concurrently.
            Defaults to ``os.cpu_count()``. Pass ``1`` to disable
            parallelism.
        """
        if region is None:
            region = self.exterior

        if isinstance(zoom_range, int):
            zr = [0, zoom_range]
        else:
            zr = zoom_range

        # Auto-convert DDB-format elevation_list (name-only) to hydromt format
        if elevation_list and "da" not in elevation_list[0]:
            res = 40075016.686 / 256 / 2 ** zr[1]
            elevation_list = self.model._parse_datasets_elevation(
                elevation_list, res=res
            )

        # When no elevation_list is given, the workflow falls back to the
        # model's data catalog (if populated).
        data_catalog = self.model.data_catalog if elevation_list is None else None

        create_topobathy_tiles(
            root=root,
            region=region,
            elevation_list=elevation_list,
            data_catalog=data_catalog,
            index_path=index_path,
            zoom_range=zr,
            z_range=z_range,
            fmt=fmt,
            write_html_viewer=write_html_viewer,
            max_workers=max_workers,
            logger=logger,
        )

    def create_index_cog(
        self,
        filename: Union[str, Path],
        filename_topobathy: Union[str, Path],
    ) -> None:
        """Write a COG raster mapping each pixel to a quadtree cell index.

        Thin wrapper around
        :py:func:`hydromt_sfincs.workflows.cog.make_quadtree_index_cog`.

        Parameters
        ----------
        filename : str or Path
            Output COG file path.
        filename_topobathy : str or Path
            Reference topobathy COG whose grid / CRS define the output.
        """
        make_quadtree_index_cog(
            quadtree_grid=self,
            filename=filename,
            filename_topobathy=filename_topobathy,
        )


def binary_search(val_array: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Return the position of each ``vals`` entry within ``val_array``.

    Entries that are not present in ``val_array`` are returned as ``-1``.
    ``val_array`` must be sorted.

    Parameters
    ----------
    val_array : np.ndarray
        Sorted 1-D array to search within.
    vals : np.ndarray
        1-D array of values to locate.

    Returns
    -------
    np.ndarray
        1-D ``int`` array of the same length as ``vals`` holding the match
        index into ``val_array``, or ``-1`` where no match exists.
    """
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
