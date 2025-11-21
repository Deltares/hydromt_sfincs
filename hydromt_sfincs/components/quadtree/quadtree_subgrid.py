import logging
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from hydromt.model.components import ModelComponent

from .subgrid_quadtree_builder import build_subgrid_table_quadtree

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

# TODO actually use the logger instead of print statements
logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsQuadtreeSubgridTable(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._data: xr.Dataset = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> xr.Dataset:
        """Model static gridded data as xarray.Dataset."""
        if self._data is None:
            self._initialize_grid()
        assert self._data is not None
        return self._data

    def _initialize_grid(self, skip_read: bool = False) -> None:
        """Initialize grid object."""
        if self._data is None:
            self._data = xr.Dataset()
            if self.root.is_reading_mode() and not skip_read:
                abs_file_path = self.model.config.get_set_file_variable(
                    "sbgfile",
                )
                if abs_file_path is None:
                    # File name not defined, so no subgrid in this model
                    return
                # if netcdf, read it with xarray
                self.read(filename=abs_file_path)

    def read(self, filename: str | Path = None):
        """Read SFINCS subgrid table (*.nc) file for Quadree grid

        Args:
            filename (str | Path, optional): File name to read. Defaults to None.
        """

        # First check whether this model uses a quadtree grid
        if not self.model.grid_type == "quadtree":
            logger.warning(
                "Quadtree subgrid table can only be used with quadtree grid. No subgrid table read."
            )
            return

        # Check that read mode is on
        self.root._assert_read_mode()

        # Get absolute file name and set it in config if crsfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "sbgfile", value=filename
        )

        # Check if abs_file_path is None
        if abs_file_path is None:
            # File name not defined, so no subgrid in this model
            return

        # Check if thd file exists
        if not abs_file_path.exists():
            raise FileNotFoundError(f"Subgrid file not found: {abs_file_path}")

        # Read from netcdf file with xarray
        self.data = xr.load_dataset(filename)

    def write(self, filename: str | Path = None):
        """Write SFINCS subgrid table (*.sbg) file for Quadree grid

        Args:
            filename (str | Path, optional): File name to write. Defaults to None.
        """

        # Check that data is not empty
        if len(self.data.data_vars) == 0:
            logger.info("No subgrid table available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            "sbgfile",
            value=filename,
            default="sfincs_subgrid.nc",
        )

        # Write XArray dataset to netcdf file
        self.data.to_netcdf(abs_file_path)

    def create(
        self,
        bathymetry_sets,
        roughness_list: list = [],
        manning_land=0.04,
        manning_water=0.020,
        manning_level=1.0,
        nr_levels=10,
        nr_subgrid_pixels=20,
        nrmax=2000,
        max_gradient=999.0,
        depth_factor=1.0,
        huthresh=0.01,
        zmin=-999999.0,
        zmax=999999.0,
        write_dep_tif=True,
        write_man_tif=False,
        weight_option="min",
        bathymetry_database=None,
        quiet=False,
        progress_bar=None,
    ):
        """Build SFINCS subgrid table for quadtree grid

        FIXME WARNING: this only works when called from Delft Dashboard
        The hydromt_sfincs.subgrid_quadtree_builder needs to be updated
        to work with data catalogs

        Args:
            bathymetry_sets (list): List of bathymetry data sets
            roughness_list (list): List of roughness data sets
            manning_land (float, optional): Manning's n value for land. Defaults to 0.04.
            manning_water (float, optional): Manning's n value for water. Defaults to 0.020.
            manning_level (float, optional): Manning's n value for level. Defaults to 1.0.
            nr_levels (int, optional): Number of levels in the quadtree. Defaults to 10.
            nr_subgrid_pixels (int, optional): Number of pixels in the subgrid. Defaults to 20.
            nrmax (int, optional): Maximum number of points per cell. Defaults to 2000.
            max_gradient (float, optional): Maximum gradient. Defaults to 999.0.
            depth_factor (float, optional): Depth factor. Defaults to 1.0.
            huthresh (float, optional): Huthresh. Defaults to 0.01.
            zmin (float, optional): Minimum elevation. Defaults to -999999.0.
            zmax (float, optional): Maximum elevation. Defaults to 999999.0.
            weight_option (str, optional): Weight option. Defaults to "min".
            bathymetry_database (str, optional): Bathymetry database. Defaults to None.
            quiet (bool, optional): Quiet mode. Defaults to False.
            progress_bar (tqdm, optional): Progress bar. Defaults to None.
        """

        if bathymetry_database is None:
            # get resolution and number of level
            res = self.model.quadtree_grid.data.attrs["dx"]
            nrlevels = self.model.quadtree_grid.data.attrs["nr_levels"]

            # convert to meters if geographic
            if self.model.crs.is_geographic:
                res = res * 111111.0
            # append parsed datasets per level
            elevation_list_per_level = []
            for ilev in range(nrlevels):
                # compute resolution per level
                res_level = res / (2**ilev)
                # convert to subgrid resolution for this level
                res_subgrid = res_level / nr_subgrid_pixels
                # parse datasets closest to subgrid resolution
                elevation_list_per_level.append(
                    self.model._parse_datasets_elevation(
                        bathymetry_sets, res=res_subgrid
                    )
                )
            bathymetry_sets = elevation_list_per_level

            if len(roughness_list) > 0:
                # NOTE conversion from landuse/landcover to manning happens here
                roughness_list = self.model._parse_roughness_list(roughness_list)

            # if len(river+sets) > 0:
            #     rivers_sets = self.model._parse_river_list(river_list)
            # folder where high-resolution topobathy and manning geotiffs are stored

            if write_dep_tif or write_man_tif:
                highres_dir = self.model.root.path / "subgrid"
                # check if directory exists using pathlib, otherwise create it
                if not highres_dir.exists():
                    highres_dir.mkdir(parents=True, exist_ok=True)
            else:
                highres_dir = None

        self._data = build_subgrid_table_quadtree(
            grid=self.model.quadtree_grid.data,
            bathymetry_sets=bathymetry_sets,
            roughness_list=roughness_list,
            manning_land=manning_land,
            manning_water=manning_water,
            manning_level=manning_level,
            nr_levels=nr_levels,
            nrmax=nrmax,
            nr_subgrid_pixels=nr_subgrid_pixels,
            max_gradient=max_gradient,
            depth_factor=depth_factor,
            huthresh=huthresh,
            zmin=zmin,
            zmax=zmax,
            highres_dir=highres_dir,
            write_dep_tif=write_dep_tif,
            write_man_tif=write_man_tif,
            weight_option=weight_option,
            bathymetry_database=bathymetry_database,
            quiet=quiet,
            progress_bar=progress_bar,
            logger=logger,
        )
