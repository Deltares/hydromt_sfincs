"""RegularGrid class for SFINCS."""

import logging
import math
import os
from os.path import isfile, join
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer
import pandas as pd
from shapely.geometry import LineString

from hydromt import hydromt_step
from hydromt.model.components import GridComponent
from hydromt.model.processes.grid import create_grid_from_region

from hydromt_sfincs import utils
from hydromt_sfincs.workflows.tiling import int2png, tile_window

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_MAPS = ["mask", "dep", "scs", "manning", "qinf", "smax", "seff", "ks", "vol", "ini"]


class SfincsGrid(GridComponent):
    """Regular grid component of the SfincsModel class.

    This class implements reading and writing of SFINCS binary grid files,
    as well as methods to create a regular grid for a region of interest.

    The data for all gridded variables is stored in the `data` attribute as an
    xarray Dataset. However, the creation of new data layers,, such as elevation or roughness,
    is done in separate model component classes, such as `SfincsElevation` or `SfincsRoughness`.

    See Also
    --------
    :py:class:`~hydromt_sfincs.components.grid.elevation.SfincsElevation`
    :py:class:`~hydromt_sfincs.components.grid.mask.SfincsMask`
    :py:class:`~hydromt_sfincs.components.grid.roughness.SfincsRoughness`
    :py:class:`~hydromt_sfincs.components.grid.infiltration.SfincsInfiltration`
    :py:class:`~hydromt_sfincs.components.grid.storage_volume.SfincsStorageVolume`
    :py:class:`~hydromt_sfincs.components.grid.initial_conditions.SfincsInitialConditions`
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        super().__init__(
            model=model,
            filename="sfincs.nc",
            region_filename="region.geojson",
        )
        # initialize data attribute
        self._data = None
        self.datashader_dataframe = pd.DataFrame()

    @property
    def transform(self):
        """Return the affine transform of the regular grid."""
        transform = (
            Affine.translation(self.x0, self.y0)
            * Affine.rotation(self.rotation)
            * Affine.scale(self.dx, self.dy)
        )
        return transform

    @property
    def coordinates(self, x_dim="x", y_dim="y"):
        """Return the coordinates of the cell-centers the regular grid."""
        if self.transform.b == 0:
            x_coords, _ = self.transform * (
                np.arange(self.mmax) + 0.5,
                np.zeros(self.mmax) + 0.5,
            )
            _, y_coords = self.transform * (
                np.zeros(self.nmax) + 0.5,
                np.arange(self.nmax) + 0.5,
            )
            coords = {
                y_dim: (y_dim, y_coords),
                x_dim: (x_dim, x_coords),
            }
        else:
            x_coords, y_coords = (
                self.transform
                * self.transform.translation(0.5, 0.5)
                * np.meshgrid(np.arange(self.mmax), np.arange(self.nmax))
            )
            coords = {
                "yc": ((y_dim, x_dim), y_coords),
                "xc": ((y_dim, x_dim), x_coords),
            }
        return coords

    @property
    def edges(self):
        """Return the coordinates of the cell-edges the regular grid."""
        x_edges, y_edges = (
            self.transform
            * self.transform.translation(0, 0)
            * np.meshgrid(np.arange(self.mmax + 1), np.arange(self.nmax + 1))
        )
        return x_edges, y_edges

    @property
    def empty_mask(self) -> xr.DataArray:
        """Return mask with only inactive cells"""
        da_mask = xr.DataArray(
            name="mask",
            data=np.zeros((self.nmax, self.mmax), dtype=np.uint8),
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": 0},
        )
        da_mask.raster.set_crs(self.crs)
        return da_mask

    @property
    def crs(self) -> CRS:
        """Return the coordinate reference system of the regular grid."""
        if self.epsg is not None:
            return CRS.from_epsg(self.epsg)
        elif self.data.raster.crs is not None:
            return self.data.raster.crs
        else:
            raise ValueError("No CRS defined for the regular grid.")

    @property
    def mask(self) -> xr.DataArray:
        """Return the mask of the regular grid."""
        if "mask" in self.data:
            da_mask = self.data["mask"]
        else:
            da_mask = self.empty_mask
        return da_mask

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Return the active region of the regular grid."""
        if "mask" in self.data and np.any(self.data["mask"] > 0):
            da = xr.where(self.data["mask"] > 0, 1, 0).astype(np.int16)
            da.raster.set_nodata(0)
            gdf = da.raster.vectorize().dissolve()
        elif self.data is not None:
            gdf = self.empty_mask.raster.box
        else:
            raise ValueError("No grid data available to derive region.")
        if not gdf.crs:
            gdf.set_crs(self.model.crs, inplace=True)
        return gdf

    def read(self, data_vars: Union[List, str] = None) -> None:
        """Read SFINCS binary grid files and save to `data` attribute.
        Filenames are taken from the `model.config` attribute (i.e. input file).

        Parameters
        ----------
        data_vars : Union[List, str], optional
            List of data variables to read, by default None (all).
        """
        # check if in read mode and initialize grid
        self.root._assert_read_mode()
        # self._initialize_grid(skip_read=True)

        # first update grid from config
        self.update_grid_from_config()

        # now read in the actual files
        da_lst = []
        if data_vars is None:
            data_vars = _MAPS
            provide_warnings = False  # all variables are asked, so no warnings
        elif isinstance(data_vars, list):
            provide_warnings = True  # specific variables are asked, so provide warnings
        elif isinstance(data_vars, str):
            data_vars = list(data_vars)
            provide_warnings = True  # specific variables are asked, so provide warnings

        # read index file
        ind_fn = self.model.config.get(
            "indexfile", fallback="sfincs.ind", abs_path=True
        )
        if not isfile(ind_fn):
            raise IOError(f".ind path {ind_fn} does not exist")

        dtypes = {"mask": "u1"}
        mvs = {"mask": 0}
        ind = self.read_ind(ind_fn=ind_fn)

        for name in data_vars:
            if name == "mask":
                # mask is special, it is always read
                fn = self.model.config.get_set_file_variable("mskfile", "sfincs.msk")
            else:
                fn = self.model.config.get(
                    f"{name}file", fallback=f"sfincs.{name}", abs_path=True
                )
            if not isfile(fn):
                if provide_warnings:
                    logger.warning(f"{name}file not found at {fn}")
                continue
            dtype = dtypes.get(name, "f4")
            mv = mvs.get(name, -9999.0)
            da = self.read_map(fn, ind, dtype, mv, name=name)
            da_lst.append(da)
        ds = xr.merge(da_lst)
        epsg = self.model.config.get("epsg", None)
        if epsg is not None:
            ds.raster.set_crs(epsg)
        self.set(ds)

        # # TODO - fix this properly; but to create overlays in GUIs,
        # # we always convert regular grids to a UgridDataArray
        # self.quadtree = QuadtreeGrid(logger=logger)
        # if self.config.get("rotation", 0) != 0:  # This is a rotated regular grid
        #     self.quadtree.data = UgridDataArray.from_structured(
        #         self.mask, "xc", "yc"
        #     )
        # else:
        #     self.quadtree.data = UgridDataArray.from_structured(self.mask)
        # self.quadtree.data.grid.set_crs(self.crs)

        # keep some metadata maps from gis directory

        # fns = glob.glob(join(self.root, "gis", "*.tif"))
        # fns = [
        #     fn
        #     for fn in fns
        #     if basename(fn).split(".")[0] not in self.grid.data_vars
        # ]
        # if fns:
        #     ds = hydromt.open_mfraster(fns).load()
        #     self.set_grid(ds)
        #     ds.close()

    def write(
        self,
        data_vars: Union[List, str] = None,
    ) -> None:
        """Write SFINCS grid to binary files including map index file.
        Filenames are taken from the `config` attribute (i.e. input file).

        If `write_gis` property is True, all grid variables are written to geotiff
        files in a "gis" subfolder.

        Parameters
        ----------
        data_vars : Union[List, str], optional
            List of data variables to write, by default None (all)
        """
        self.root._assert_write_mode

        dtypes = {"mask": "u1"}  # default to f4
        if len(self.data.data_vars) > 0 and "mask" in self.data:
            # make sure orientation is S->N
            ds_out = self.data
            if ds_out.raster.res[1] < 0:
                ds_out = ds_out.raster.flipud()
            mask = ds_out["mask"].values

            logger.debug("Write binary map indices based on mask.")
            if self.model.config.get("indexfile") is None:
                self.model.config.set("indexfile", "sfincs.ind")
            abs_file_path = self.model.config.get("indexfile", abs_path=True)
            # Create parent directories if they do not exist
            abs_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Write index file
            self.write_ind(ind_fn=abs_file_path, mask=mask)

            if data_vars is None:  # write all maps
                data_vars = [v for v in _MAPS if v in ds_out]
            elif isinstance(data_vars, str):
                data_vars = list(data_vars)
                # always rewrite the mask
                data_vars.append("mask") if "mask" not in data_vars else data_vars

            logger.debug(f"Write binary map files: {data_vars}.")
            for name in data_vars:
                # Set file name and get absolute path
                if name == "mask":
                    abs_file_path = self.model.config.get_set_file_variable(
                        "mskfile", "sfincs.msk"
                    )
                else:
                    abs_file_path = self.model.config.get_set_file_variable(
                        f"{name}file",
                        f"sfincs.{name}",
                    )

                # write to binary model files
                self.write_map(
                    map_fn=abs_file_path,
                    data=ds_out[name].values,
                    mask=mask,
                    dtype=dtypes.get(name, "f4"),
                )

                # write to gis-files for visualization
                if self.model.write_gis:
                    utils.write_raster(
                        ds_out[name],
                        root=join(self.model.root.path, "gis"),
                        mask=mask,
                        logger=logger,
                    )

            # write the model region to a geojson file for visualization
            if self.model.write_gis:
                utils.write_vector(
                    self.region,
                    name="region",
                    root=join(self.model.root.path, "gis"),
                    logger=logger,
                )

    @hydromt_step
    def create(
        self,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nmax: int,
        mmax: int,
        rotation: float,
        epsg: int,
    ):
        """Create a regular grid for the SfincsModel.

        Parameters
        ----------
        x0, y0 : float
            x,y coordinates of the origin of the grid
        dx, dy : float
            grid cell size in x and y direction
        mmax, nmax : int
            number of grid cells in x and y direction
        rotation : float, optional
            rotation of grid [degree angle], by default None
        epsg : int, optional
            epsg-code of the coordinate reference system
        """

        # update the grid attributes in the model config
        self.model.config.update(
            {
                "x0": x0,
                "y0": y0,
                "dx": dx,
                "dy": dy,
                "nmax": nmax,
                "mmax": mmax,
                "rotation": rotation,
                "epsg": epsg,
            }
        )
        # create grid based on config
        self.update_grid_from_config()

        # initialize a grid without variables
        ds = xr.Dataset(
            coords=self.coordinates,
        )
        ds.raster.set_crs(self.model.crs)

        # set the grid in the model data
        self.set(ds)

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
    ):
        """Create a regular grid for the SfincsModel based on a region.

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
        grid_type : str, optional
            grid type, "regular" (default) or "quadtree"
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

        See Also
        --------
        :py:func:`~hydromt.model.processes.create_grid_from_region`
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

        # add the grid to the model
        self.set(ds)

        # update the grid attributes in the model config
        self.update_config_from_grid()

    # %% supporting HydroMT-SFINCS functions:
    # other:
    # - ind
    # - read_ind
    # - read_map
    # - write_ind
    # - write_map
    # - to_gdf

    def ind(self, mask: np.ndarray) -> np.ndarray:
        """Return indices of active cells in mask."""
        assert mask.shape == (self.nmax, self.mmax)
        ind = np.where(mask.ravel(order="F"))[0]
        return ind

    def read_ind(
        self,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> np.ndarray:
        """Read indices of active cells in mask from binary file."""
        _ind = np.fromfile(ind_fn, dtype="u4")
        ind = _ind[1:] - 1  # convert to zero based index
        assert _ind[0] == ind.size

        return ind

    def read_map(
        self,
        map_fn: Union[str, Path],
        ind: np.ndarray,
        dtype: Union[str, np.dtype] = "f4",
        mv: float = -9999.0,
        name: str = None,
    ) -> xr.DataArray:
        """Read one of the grid variables of the SFINCS model map from a binary file."""

        data = np.full((self.mmax, self.nmax), mv, dtype=dtype)
        data.flat[ind] = np.fromfile(map_fn, dtype=dtype)
        data = data.transpose()

        da = xr.DataArray(
            name=map_fn.split(".")[-1] if name is None else name,
            data=data,
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": mv},
        )
        return da

    def write_ind(
        self,
        mask: np.ndarray,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> None:
        """Write indices of active cells in mask to binary file."""
        assert mask.shape == (self.nmax, self.mmax)
        # Add 1 because indices in SFINCS start with 1, not 0
        ind = self.ind(mask)
        indices_ = np.array(np.hstack([np.array(len(ind)), ind + 1]), dtype="u4")
        indices_.tofile(ind_fn)

    def write_map(
        self,
        map_fn: Union[str, Path],
        data: np.ndarray,
        mask: np.ndarray,
        dtype: Union[str, np.dtype] = "f4",
    ) -> None:
        """Write one of the grid variables of the SFINCS model map to a binary file."""

        data_out = np.asarray(data.transpose()[mask.transpose() > 0], dtype=dtype)
        data_out.tofile(map_fn)

    def update_grid_from_config(self):
        """Update grid properties based on `config` (sfincs.inp) attributes"""

        # assert model.config exists
        if not hasattr(self.model, "config"):
            raise AttributeError("Model has no config attribute")

        self.x0 = self.model.config.get("x0")
        self.y0 = self.model.config.get("y0")
        self.dx = self.model.config.get("dx")
        self.dy = self.model.config.get("dy")
        self.nmax = self.model.config.get("nmax")
        self.mmax = self.model.config.get("mmax")
        self.rotation = self.model.config.get("rotation", 0)
        self.epsg = self.model.config.get("epsg", None)

        # Set 'crsgeo' flag in the config based on whether the CRS is geographic
        if self.epsg is not None:
            crs = CRS.from_epsg(self.epsg)
            self.model.config.set("crsgeo", int(crs.is_geographic))

    def update_config_from_grid(self):
        """Update `config` (sfincs.inp) attributes based on grid properties"""

        # derive grid properties from grid
        self.nmax, self.mmax = self.data.raster.shape
        self.dx, self.dy = self.data.raster.res
        self.x0, self.y0 = self.data.raster.origin
        self.rotation = self.data.raster.rotation
        self.epsg = self.data.raster.crs.to_epsg()

        # round raster resolution and rotation based on CRS type
        if self.data.raster.crs.is_geographic:
            self.dx = round(self.dx, 6)
            self.dy = round(self.dy, 6)
            self.rotation = round(self.rotation, 6)
            crsgeo = 1
        else:
            self.dx = round(self.dx, 3)
            self.dy = round(self.dy, 3)
            self.rotation = round(self.rotation, 3)
            crsgeo = 0

        # update the grid properties in the config
        self.model.config.update(
            {
                "x0": self.x0,
                "y0": self.y0,
                "dx": self.dx,
                "dy": self.dy,
                "nmax": self.nmax,
                "mmax": self.mmax,
                "rotation": self.rotation,
                "epsg": self.epsg,
                "crsgeo": crsgeo,
            }
        )

    def _get_grid_lines(self):
        """Return lists of start and end coordinates for all grid lines."""
        x, y = self.edges

        x1_list, y1_list, x2_list, y2_list = [], [], [], []

        # Vertical lines
        for i in range(self.nmax + 1):
            x1_list.append(x[i, 0])
            y1_list.append(y[i, 0])
            x2_list.append(x[i, -1])
            y2_list.append(y[i, -1])

        # Horizontal lines
        for j in range(self.mmax + 1):
            x1_list.append(x[0, j])
            y1_list.append(y[0, j])
            x2_list.append(x[-1, j])
            y2_list.append(y[-1, j])

        return (
            np.array(x1_list),
            np.array(y1_list),
            np.array(x2_list),
            np.array(y2_list),
        )

    def to_gdf(self):
        """Return a GeoDataFrame with a geometry for each grid line."""
        x1, y1, x2, y2 = self._get_grid_lines()
        lines = [LineString([(x1[i], y1[i]), (x2[i], y2[i])]) for i in range(len(x1))]
        return gpd.GeoDataFrame(geometry=lines, crs=self.model.crs)

    def get_datashader_dataframe(self):
        """Create a datashader-friendly DataFrame for the regular grid."""
        x1, y1, x2, y2 = self._get_grid_lines()

        # Check if the grid crosses the dateline
        cross_dateline = False
        if self.model.crs.is_geographic:
            if np.max(x1) > 180.0 or np.max(x2) > 180.0:
                cross_dateline = True

        # Transform to Web Mercator for Datashader
        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)

        # Handle dateline wrapping
        if cross_dateline:
            x1[x1 < 0] += 40075016.68557849
            x2[x2 < 0] += 40075016.68557849

        self.datashader_dataframe = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def clear_datashader_dataframe(self):
        """Clears the datashader dataframe"""
        self.datashader_dataframe = pd.DataFrame()

    # %% DDB GUI focused additional functions:
    # create_index_tiles > FIXME - TL: still needed?
    # map_overlay
    # snap_to_grid
    # _get_datashader_dataframe

    # TODO - missing as in cht_sfincs:
    # Many...

    def create_index_tiles(
        self,
        root: Union[str, Path],
        region: gpd.GeoDataFrame,
        zoom_range: Union[int, List[int]] = [0, 13],
        fmt: str = "bin",
        logger: logging.Logger = logger,
    ):
        """Create index tiles for a region. Index tiles are used to quickly map webmercator tiles to the corresponding SFINCS cell.

        Parameters
        ----------
        region : gpd.GeoDataFrame
            GeoDataFrame containing the region of interest
        root : Union[str, Path]
            Directory where index tiles are stored
        zoom_range : Union[int, List[int]], optional
            Range of zoom levels for which tiles are created, by default [0,13]
        fmt : str, optional
            Format of index tiles, either "bin" (binary, default) or "png"
        """

        index_path = os.path.join(root, "indices")
        npix = 256

        # for binary format, use .dat extension
        if fmt == "bin":
            extension = "dat"
        # for net, tif and png extension and format are the same
        else:
            extension = fmt

        # if only one zoom level is specified, create tiles up to that zoom level (inclusive)
        if isinstance(zoom_range, int):
            zoom_range = [0, zoom_range]

        # get bounding box of sfincs model
        minx, miny, maxx, maxy = region.total_bounds
        transformer = Transformer.from_crs(region.crs.to_epsg(), 3857)

        # rotation of the model
        cosrot = math.cos(-self.rotation * math.pi / 180)
        sinrot = math.sin(-self.rotation * math.pi / 180)

        # axis order is different for geographic and projected CRS
        if region.crs.is_geographic:
            minx, miny = map(
                max, zip(transformer.transform(miny, minx), [-20037508.34] * 2)
            )
            maxx, maxy = map(
                min, zip(transformer.transform(maxy, maxx), [20037508.34] * 2)
            )
        else:
            minx, miny = map(
                max, zip(transformer.transform(minx, miny), [-20037508.34] * 2)
            )
            maxx, maxy = map(
                min, zip(transformer.transform(maxx, maxy), [20037508.34] * 2)
            )

        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            logger.debug("Processing zoom level " + str(izoom))

            zoom_path = os.path.join(index_path, str(izoom))

            for transform, col, row in tile_window(izoom, minx, miny, maxx, maxy):
                # transform is a rasterio Affine object
                # col, row are the tile indices
                file_name = os.path.join(
                    zoom_path, str(col), str(row) + "." + extension
                )

                # get the coordinates of the tile in webmercator projection
                x = np.arange(0, npix) + 0.5
                y = np.arange(0, npix) + 0.5
                x3857, y3857 = transform * (x, y)
                x3857, y3857 = np.meshgrid(x3857, y3857)

                # convert to SFINCS coordinates
                x, y = transformer.transform(x3857, y3857, direction="INVERSE")

                # Now rotate around origin of SFINCS model
                x00 = x - self.x0
                y00 = y - self.y0
                xg = x00 * cosrot - y00 * sinrot
                yg = x00 * sinrot + y00 * cosrot

                # determine the SFINCS cell indices
                iind = np.floor(xg / self.dx).astype(int)
                jind = np.floor(yg / self.dy).astype(int)
                ind = iind * self.nmax + jind
                ind[iind < 0] = -999
                ind[jind < 0] = -999
                ind[iind >= self.mmax] = -999
                ind[jind >= self.nmax] = -999

                # only write tiles that link to at least one SFINCS cell
                if np.any(ind >= 0):
                    if not os.path.exists(os.path.join(zoom_path, str(col))):
                        os.makedirs(os.path.join(zoom_path, str(col)))
                    # And write indices to file
                    if fmt == "bin":
                        fid = open(file_name, "wb")
                        fid.write(ind)
                        fid.close()
                    elif fmt == "png":
                        # for png, change nodata -999 nodata into 0
                        ind[ind == -999] = 0
                        int2png(ind, file_name)

    def get_indices_at_points(self, x, y):
        # x and y are 2D arrays of coordinates (x, y) in the same projection as the model
        # if x is a float, convert to 2D array
        if np.ndim(x) == 0:
            x = np.array([[x]])
        if np.ndim(y) == 0:
            y = np.array([[y]])

        # get the coordinates of the SFINCS model
        x0 = self.x0
        y0 = self.y0
        dx = self.dx
        dy = self.dy
        nmax = self.nmax
        mmax = self.mmax
        rotation = self.rotation

        cosrot = np.cos(-rotation * np.pi / 180)
        sinrot = np.sin(-rotation * np.pi / 180)

        # Now rotate around origin of SFINCS model
        x00 = x - x0
        y00 = y - y0
        xg = x00 * cosrot - y00 * sinrot
        yg = x00 * sinrot + y00 * cosrot

        # determine the SFINCS cell indices
        iind = np.floor(xg / dx).astype(int)
        jind = np.floor(yg / dy).astype(int)
        ind = iind * nmax + jind
        ind[iind < 0] = -999
        ind[jind < 0] = -999
        ind[iind >= mmax] = -999
        ind[jind >= nmax] = -999

        return ind
