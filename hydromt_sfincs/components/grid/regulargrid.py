"""RegularGrid class for SFINCS."""

import logging
from os.path import isfile, join
from pathlib import Path
from typing import TYPE_CHECKING, List, Union, Optional

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from pyproj import CRS
from shapely.geometry import LineString

from hydromt import hydromt_step
from hydromt.model.components import GridComponent
from hydromt.model.processes.grid import create_grid_from_region

from hydromt_sfincs import writers
from hydromt_sfincs.readers import read_binary_map
from hydromt_sfincs.workflows.cog import create_index_cog, create_topobathy_cog

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_MAPS = [
    "mask",
    "dep",
    "scs",
    "manning",
    "qinf",
    "smax",
    "seff",
    "ks",
    "psi",
    "sigma",
    "f0",
    "fc",
    "kd",
    "vol",
    "zs",
]
_MAP_EXCEPTIONS = {"zs": ("inifile", "sfincs.ini")}


class SfincsGrid(GridComponent):
    """Regular grid component of the SfincsModel class.

    This class implements reading and writing of SFINCS binary grid files,
    as well as methods to create a regular grid for a region of interest.

    The data for all gridded variables is stored in the `data` attribute as an
    xarray Dataset. However, the creation of new data layers,, such as elevation or roughness,
    is done in separate model component classes, such as `SfincsElevation` or `SfincsRoughness`.

    Layer ownership
    ---------------
    A component may claim the layers it is responsible for through a
    ``grid_variables`` attribute. ``write()`` skips those layers unless called
    with ``write_components=True``; the component writes them itself through
    :py:meth:`_write_layers`. ``read()`` loads the mask and elevation;
    component-owned layers are read by their respective components.

    This grid always owns the cell index file (``sfincs.ind``), the mask and the
    elevation, since every other binary map is indexed against them.
    Mask and elevation component ``read()`` methods ensure this grid is loaded.
    Their ``write()`` methods are no-ops; persistence is handled here.
    Component ``data`` properties expose only their relevant layers; this
    grid remains the full dataset and the destination for ``set()`` calls.

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
        self._indices = None

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
    def indices(self) -> np.ndarray:
        """Return active-cell indices in SFINCS (south-to-north) order.

        Return the index loaded for existing maps, or derive it from the current
        mask when the grid is newly created.
        """
        if self._indices is not None:
            return self._indices
        if self._data is not None:
            mask = self.mask
            if mask.raster.res[1] < 0:
                mask = mask.raster.flipud()
            return self.ind(mask.values)
        self._indices = self.read_ind(
            self.model.config.get("indexfile", fallback="sfincs.ind", abs_path=True)
        )
        return self._indices

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

    def read(
        self,
        read_components: bool = False,
    ) -> None:
        """Read SFINCS binary grid files and save to `data` attribute.
        Filenames are taken from the `model.config` attribute (i.e. input file).

        Reads the mask and elevation. Component-owned layers are loaded only
        when ``read_components`` is True, or through component ``read()`` methods.

        Parameters
        ----------
        read_components : bool, optional
            Also call ``read()`` on the components owning layers here, by
            default False. Those read their own files, so this matters for
            layers that are not plain binary maps, such as the bucket netcdf.
        """
        # check if in read mode and initialize grid
        self.root._assert_read_mode()
        self._initialize_grid(skip_read=True)

        # first update grid from config
        self.update_grid_from_config()

        ind_fn = self.model.config.get(
            "indexfile", fallback="sfincs.ind", abs_path=True
        )
        if not isfile(ind_fn):
            raise IOError(f".ind path {ind_fn} does not exist")
        self._indices = None
        self._indices = self.read_ind(ind_fn=ind_fn)
        self._read_layers(["mask", "dep"])

        if read_components:
            # after the grid itself, so the components find the mask and index
            for name in self.model._REGULAR_GRID_NAMES:
                component = self.model.components.get(name)
                if component is not None and getattr(component, "grid_variables", None):
                    component.read()

    def write(
        self,
        write_components: bool = False,
    ) -> None:
        """Write SFINCS grid to binary files including map index file.
        Filenames are taken from the `config` attribute (i.e. input file).

        .. note::
            Layers claimed by a component through its ``grid_variables`` attribute
            are only written when ``write_components`` is True, or by calling that
            component's ``write()`` directly. This covers ``manning``, ``vol``,
            ``zs`` and all infiltration layers.

        Parameters
        ----------
        write_components : bool, optional
            Also write the layers owned by other components, by calling their
            ``write()`` after the index file exists, by default False.
        """
        self.root._assert_write_mode()

        if len(self.data.data_vars) > 0 and "mask" in self.data:
            # make sure orientation is S->N
            ds_out = self.data
            if ds_out.raster.res[1] < 0:
                ds_out = ds_out.raster.flipud()
            mask = ds_out["mask"].values
            self._indices = self.ind(mask)

            logger.debug("Write binary map indices based on mask.")
            if self.model.config.get("indexfile") is None:
                self.model.config.set("indexfile", "sfincs.ind")
            abs_file_path = self.model.config.get("indexfile", abs_path=True)
            # Create parent directories if they do not exist
            abs_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Write index file
            self.write_ind(ind_fn=abs_file_path, mask=mask)

            # component-owned layers are written by the component itself
            owned = set()
            for component in self.model.components.values():
                owned.update(getattr(component, "grid_variables", ()) or ())
            data_vars = [v for v in _MAPS if v in ds_out and v not in owned]
            # always rewrite the mask
            for name in ("mask", "dep"):
                if name in ds_out and name not in data_vars:
                    data_vars.append(name)

            logger.debug(f"Write binary map files: {data_vars}.")
            self._write_layers(data_vars)

            # write the model region to a geojson file for visualization
            if self.model.write_gis:
                writers.write_vector(
                    self.region,
                    name="region",
                    root=join(self.model.root.path, "gis"),
                )

        if write_components:
            # after the index file, so the components can write against it
            for name in self.model._REGULAR_GRID_NAMES:
                component = self.model.components.get(name)
                if component is not None and getattr(component, "grid_variables", None):
                    component.write()

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
        self._indices = None

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
        self._indices = None

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

    def _write_layers(self, variables: List[str]) -> None:
        """Write variables from `data` as SFINCS binary maps, one file per variable.

        The index file is written first if it does not exist yet, since every
        binary map is indexed against it.
        """
        self.root._assert_write_mode()
        for name in variables:
            if name in self.data:
                continue
            config_key, _ = _MAP_EXCEPTIONS.get(name, (f"{name}file", None))
            fn = self.model.config.get(config_key, abs_path=True)
            if fn is None:
                continue  # never configured, so nothing can go missing
            if isfile(fn):
                # already on disk at the write root, so there is nothing to do
                logger.debug(f"'{name}' not loaded; leaving {Path(fn).name} as is.")
            else:
                logger.warning(
                    f"'{config_key}' is set to {Path(fn).name} but '{name}' is not "
                    f"loaded and the file is not present here, so it will be missing "
                    f"from this model. Read the component first, or use model.read()."
                )
        variables = [v for v in variables if v in self.data]
        if not variables:
            return
        if self._indices is None:
            raise RuntimeError("Call grid.write() before writing grid layers.")

        # binary maps are stored S->N, flip if needed
        ds_out = self.data
        if ds_out.raster.res[1] < 0:
            ds_out = ds_out.raster.flipud()

        mask = ds_out["mask"].values
        for name in variables:
            config_key, default = _MAP_EXCEPTIONS.get(
                name, (f"{name}file", f"sfincs.{name}")
            )
            if name == "mask":
                config_key, default = "mskfile", "sfincs.msk"
            abs_file_path = self.model.config.get_set_file_variable(
                config_key, default=default
            )
            abs_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.write_map(
                abs_file_path,
                data=ds_out[name].values,
                mask=mask,
                dtype="u1" if name == "mask" else "f4",
            )

            if self.model.write_gis:
                writers.write_raster(
                    ds_out[name],
                    root=join(self.model.root.path, "gis"),
                    mask=mask,
                )

    def _read_layers(self, variables: List[str]) -> None:
        """Read SFINCS binary maps into `data`, one file per variable."""
        self.root._assert_read_mode()
        if self._indices is None:
            raise RuntimeError("Call grid.read() before reading grid layers.")

        arrays = []
        for name in variables:
            config_key, fallback = _MAP_EXCEPTIONS.get(
                name, (f"{name}file", f"sfincs.{name}")
            )
            if name == "mask":
                config_key = "mskfile"
                fn = self.model.config.get_set_file_variable(config_key, "sfincs.msk")
            elif name == "dep":
                fn = self.model.config.get(config_key, fallback=fallback, abs_path=True)
            else:
                # Component-owned maps are optional unless configured explicitly.
                fn = self.model.config.get(config_key, abs_path=True)
                if fn is None:
                    continue
            if not isfile(fn):
                logger.warning(f"{config_key} not found at {fn}")
                continue
            dtype, nodata = ("u1", 0) if name == "mask" else ("f4", np.nan)
            arrays.append(self.read_map(fn, self._indices, dtype, nodata, name=name))
        if arrays:
            ds = xr.merge(arrays, compat="no_conflicts")
            epsg = self.model.config.get("epsg", None)
            if epsg is not None:
                ds.raster.set_crs(epsg)
            self.set(ds)

    def read_map(
        self,
        map_fn: Union[str, Path],
        ind: np.ndarray,
        dtype: Union[str, np.dtype] = "f4",
        mv: float = np.nan,
        name: str = None,
    ) -> xr.DataArray:
        """Read one of the grid variables of the SFINCS model map from a binary file."""

        data = read_binary_map(map_fn, ind, (self.nmax, self.mmax), mv, dtype)

        da = xr.DataArray(
            name=map_fn.split(".")[-1] if name is None else name,
            data=data,
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": mv},
        )

        if name != "mask":
            da = da.raster.mask_nodata()
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

        # make sure there is no NaN in the data to be written, otherwise SFINCS will crash
        if np.any(np.isnan(data[mask > 0])):
            raise ValueError(
                f"Data to be written to {map_fn} contains NaN values. "
                "Please replace NaN values with a valid value before writing."
            )
        writers.write_binary_map(map_fn, data, mask, dtype=dtype)

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

    # %% DDB GUI focused additional functions:
    # map_overlay

    def create_topobathy_cog(
        self,
        filename: Union[str, Path],
        elevation_list: List[dict],
        resolution: float = 10.0,
        river_list: Optional[List[dict]] = None,
        buffer_cells: int = 0,
        z_minimum: float = -20000.0,
        nrmax: int = 2000,
    ) -> None:
        """Write a COG raster sampling the model topobathy.

        Parameters
        ----------
        filename : str or Path
            Output COG file path (north-up raster in the model CRS, nodata NaN)
            with the merged elevation datasets.
        elevation_list : List[dict]
            List of dictionaries with datasets and merge arguments to use for depth.
        resolution : float, optional
            Desired resolution in model CRS units, by default 10.0.
        river_list : List[dict], optional
            List of dictionaries with variable names and dataset names to use for river depth.
        buffer_cells : int, optional
            Number of buffer cells for interpolation between datasets, by default 0.
        z_minimum : float, optional
            Minimum elevation value to use for the output raster, by default -20000.0.
        nrmax : int, optional
            Maximum number of cells per block, by default 2000
            These blocks are used to prevent memory issues while working with large datasets
        """

        create_topobathy_cog(
            model=self.model,
            elevation_list=elevation_list,
            river_list=river_list,
            filename=filename,
            res=resolution,
            buffer_cells=buffer_cells,
            z_minimum=z_minimum,
            nrmax=nrmax,
        )

    def create_index_cog(
        self,
        filename: Union[str, Path],
        filename_topobathy: Union[str, Path],
        nrmax: int = 2000,
    ) -> None:
        """Write a COG raster mapping each pixel to a grid cell index.

        Thin wrapper around
        :py:func:`hydromt_sfincs.workflows.cog.create_index_cog`.

        Parameters
        ----------
        filename : str or Path
            Output COG file path. This uint32 raster contains the grid cell index for
            each pixel, or the nodata value 2147483647 for pixels outside the active grid.
            Note that this index file only works in combination with the topobathy COG
            used for creation.
        filename_topobathy : str or Path
            Reference topobathy COG whose grid / CRS define the output.
        nrmax : int, optional
            Maximum number of cells per block, by default 2000
            These blocks are used to prevent memory issues while working with large datasets
        """
        create_index_cog(
            model=self.model,
            filename=filename,
            filename_topobathy=filename_topobathy,
            nrmax=nrmax,
        )

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
