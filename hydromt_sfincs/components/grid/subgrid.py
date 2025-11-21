"""
SubgridTableRegular class to create, read and write sfincs subgrid (sbg) files.
"""

import gc
import logging
import os
from typing import TYPE_CHECKING, List

import numpy as np
import rasterio
import xarray as xr
from numba import njit
from rasterio.windows import Window

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils, workflows
from hydromt_sfincs.workflows.subgrid import subgrid_v_table, subgrid_q_table

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsSubgridTable(ModelComponent):
    """SFINCS Subgrid Table Component.

    This component contains methods to create, read and write subgrid tables for the SFINCS model
    on regular grids. Subgrid tables are used to represent subgrid-scale variations in bed level
    and roughness within each grid cell, allowing for more accurate simulations of flow dynamics.

    .. note::
        The subgrid table data is stored in the component's data attribute as an xarray.Dataset.
    """

    def __init__(
        self,
        model: "SfincsModel",
        version: int = 1,
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
                if abs_file_path.suffix != ".nc":
                    # if not netcdf, assume it is a binary file
                    self.read_binary(filename=abs_file_path)
                else:
                    # if netcdf, read it with xarray
                    self.read(filename=abs_file_path)

    # new way of reading netcdf subgrid tables
    def read(self, filename: str = None):
        """Load subgrid table from file for a regular grid with given mask.
        If filename is not specified, sthe filename is taken from the model configuration.
        """

        # Check that read mode is on
        self.root._assert_read_mode()

        # get absolute file path
        abs_file_path = self.model.config.get_set_file_variable(
            "sbgfile", value=filename
        )

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(f"Subgrid file not found: {abs_file_path}")

        # check if the file is a netcdf file
        if abs_file_path.suffix == ".nc":
            # read netcdf file
            self.read_netcdf(filename=abs_file_path)
        else:
            # read binary file
            self.read_binary(filename=abs_file_path)

    def read_netcdf(self, filename: str = None):
        """Load subgrid table from netcdf file for a regular grid with given mask."""
        # netcdf, so set version to 1
        self.version = 1

        # get the mask from the model
        mask = self.model.grid.mask

        # Read data from netcdf file with xarray
        with xr.open_dataset(filename) as ds:
            # transpose to have level as first dimension
            ds = ds.transpose("levels", "npuv", "np")

            # grid dimensions
            grid_dim = mask.shape

            # get number of levels, point and uv points
            self.nr_levels, self.nr_cells, self.nr_uv_points = (
                ds.sizes["levels"],
                ds.sizes["np"],
                ds.sizes["npuv"],
            )

            # find indices of active cells
            index_nm, index_mu1, index_nu1 = utils.find_uv_indices(mask)
            active_indices = np.where(index_nm > -1)[0]
            active_u_indices = np.where(index_mu1 > -1)[0]
            active_v_indices = np.where(index_nu1 > -1)[0]

            # convert 1D indices to 2D indices
            active_cells = np.unravel_index(active_indices, grid_dim, order="F")
            active_u_cells = np.unravel_index(active_u_indices, grid_dim, order="F")
            active_v_cells = np.unravel_index(active_v_indices, grid_dim, order="F")

            # Initialize the data-arrays
            # Z points
            self.z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
            self.z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
            self.z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
            self.z_level = np.full(
                (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
            )

            # U points
            self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
            self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
            self.u_havg = np.full(
                (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
            )
            self.u_nrep = np.full(
                (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
            )
            self.u_pwet = np.full(
                (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
            )
            self.u_ffit = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)
            self.u_navg = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)

            # V points
            self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
            self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
            self.v_havg = np.full(
                (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
            )
            self.v_nrep = np.full(
                (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
            )
            self.v_pwet = np.full(
                (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
            )
            self.v_ffit = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)
            self.v_navg = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)

            # Now read the data and add it to the data-arrays
            # use index_nm of the active cells in the new dataset
            self.z_zmin[active_cells] = ds["z_zmin"].values.flatten()
            self.z_zmax[active_cells] = ds["z_zmax"].values.flatten()
            self.z_volmax[active_cells] = ds["z_volmax"].values.flatten()
            for ilevel in range(self.nr_levels):
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
                u_array[active_u_cells] = uv_var[index_mu1[index_mu1 > -1]]
                v_array[active_v_cells] = uv_var[index_nu1[index_nu1 > -1]]

                # Set the modified arrays back to the attributes
                setattr(self, u_attr_name, u_array)
                setattr(self, v_attr_name, v_array)

            var_list_levels = ["havg", "nrep", "pwet"]
            for var in var_list_levels:
                for ilevel in range(self.nr_levels):
                    uv_var = ds["uv_" + var][ilevel].values.flatten()

                    # Dynamically set the attribute for self.u_var and self.v_var
                    u_attr_name = f"u_{var}"
                    v_attr_name = f"v_{var}"

                    # Retrieve the current attribute values
                    u_array = getattr(self, u_attr_name)
                    v_array = getattr(self, v_attr_name)

                    # Update only the active indices
                    u_array[ilevel, active_u_cells[0], active_u_cells[1]] = uv_var[
                        index_mu1[index_mu1 > -1]
                    ]
                    v_array[ilevel, active_v_cells[0], active_v_cells[1]] = uv_var[
                        index_nu1[index_nu1 > -1]
                    ]

                    # Set the modified arrays back to the attributes
                    setattr(self, u_attr_name, u_array)
                    setattr(self, v_attr_name, v_array)

        # store the data in the _data attribute
        self._data = self.to_xarray(dims=mask.raster.dims, coords=mask.raster.coords)

    # Following remains for backward compatibility, but should soon not be used anymore
    def read_binary(self, filename: str = None):
        """Load subgrid table from binary file for a regular grid with given mask."""

        # set version to old binary format
        self.version = 0

        # get the mask from the model
        mask = self.model.grid.mask

        if isinstance(mask, xr.DataArray):
            mask = mask.values

        iok = np.where(np.transpose(mask) > 0)
        iok = (iok[1], iok[0])

        nmax = np.shape(mask)[0]
        mmax = np.shape(mask)[1]

        grid_dim = (nmax, mmax)

        file = open(filename, "rb")

        # File version
        # self.version = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Initialize the data-arrays
        self.nr_cells = np.fromfile(file, dtype=np.int32, count=1)[0]
        self.nr_uv_points = np.fromfile(file, dtype=np.int32, count=1)[0]
        self.nr_levels = np.fromfile(file, dtype=np.int32, count=1)[0]

        # Z points
        self.z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        # self.z_zmean = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.z_depth = np.full(
            (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_hrep = np.full(
            (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_navg = np.full(
            (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_hrep = np.full(
            (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_navg = np.full(
            (self.nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
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
        for ilevel in range(self.nr_levels):
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
        for ilevel in range(self.nr_levels):
            self.u_hrep[ilevel, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )
        for ilevel in range(self.nr_levels):
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
        for ilevel in range(self.nr_levels):
            self.v_hrep[ilevel, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )
        for ilevel in range(self.nr_levels):
            self.v_navg[ilevel, iok[0], iok[1]] = np.fromfile(
                file, dtype=np.float32, count=self.nr_cells
            )

        file.close()

        # store the data in the _data attribute
        self._data = self.to_xarray(
            dims=self.model.grid.mask.raster.dims,
            coords=self.model.grid.mask.raster.coords,
        )

    # new way of writing netcdf subgrid tables
    def write(self, filename: str = None):
        """Write subgrid table to file for a regular grid with given mask. Values are only written
        for active cells (mask > 0). If filename is not specified, the filename is taken from the model
        configuration."""

        # Check that write mode is on
        self.root._assert_write_mode()

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

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # check if the file is a netcdf file
        if abs_file_path.suffix == ".nc":
            # read netcdf file
            self.write_netcdf(filename=abs_file_path)
        else:
            # read binary file
            self.write_binary(filename=abs_file_path)

    def write_netcdf(self, filename: str = None):
        """Save the subgrid data to a netcdf file for a regular grid with given mask. Values are only written
        for active cells (mask > 0)."""
        # get the mask from the model and convert to xarray
        mask = self.model.grid.mask
        ds = self.to_xarray(dims=mask.raster.dims, coords=mask.raster.coords)

        # Need to transpose to match the FORTRAN convention in SFINCS
        ds = ds.transpose("levels", "x", "y")

        # find indices of active cells
        index_nm, index_mu1, index_nu1 = utils.find_uv_indices(mask)

        # get number of levels
        nr_levels = self.nr_levels

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

        z_level = np.zeros((nr_levels, nr_z_points))
        for ilevel in range(nr_levels):
            z_level[ilevel] = ds["z_level"][ilevel].values.flatten()[active_cells]
        ds_new["z_level"] = xr.DataArray(z_level, dims=("levels", "np"))

        # u and v points
        var_list = ["zmin", "zmax", "ffit", "navg"]
        for var in var_list:
            uv_var = np.zeros(nr_uv_points)
            uv_var[index_mu1[index_mu1 > -1]] = ds["u_" + var].values.flatten()[
                index_mu1 > -1
            ]
            uv_var[index_nu1[index_nu1 > -1]] = ds["v_" + var].values.flatten()[
                index_nu1 > -1
            ]
            ds_new[f"uv_{var}"] = xr.DataArray(uv_var, dims=("npuv"))

        var_list_levels = ["havg", "nrep", "pwet"]
        for var in var_list_levels:
            uv_var = np.zeros((nr_levels, nr_uv_points))
            for ilevel in range(nr_levels):
                uv_var[ilevel, index_mu1[index_mu1 > -1]] = ds["u_" + var][
                    ilevel
                ].values.flatten()[index_mu1 > -1]
                uv_var[ilevel, index_nu1[index_nu1 > -1]] = ds["v_" + var][
                    ilevel
                ].values.flatten()[index_nu1 > -1]
            ds_new[f"uv_{var}"] = xr.DataArray(uv_var, dims=("levels", "npuv"))

        # ensure levels is last dimension
        ds_new = ds_new.transpose("npuv", "np", "levels")

        # Write to netcdf file
        ds_new.to_netcdf(filename)

    # Following remains for backward compatibility, but should soon not be used anymore
    def write_binary(self, filename: str = None):
        """Save the subgrid data to a binary file. Values are only written for active cells (mask > 0)."""

        # get the mask from the model
        mask = self.model.grid.mask

        if isinstance(mask, xr.DataArray):
            mask = mask.values

        iok = np.where(np.transpose(mask) > 0)
        iok = (iok[1], iok[0])

        nmax = np.shape(self.z_zmin)[0]
        mmax = np.shape(self.z_zmin)[1]

        # Add 1 because indices in SFINCS start with 1, not 0
        ind = np.ravel_multi_index(iok, (nmax, mmax), order="F") + 1

        file = open(filename, "wb")
        # file.write(np.int32(self.version))  # version
        file.write(np.int32(np.size(ind)))  # Nr of active points
        file.write(np.int32(1))  # min
        file.write(np.int32(self.nr_levels))

        # Z
        v = self.z_zmin[iok]
        file.write(np.float32(v))
        v = self.z_zmax[iok]
        file.write(np.float32(v))
        v = self.z_volmax[iok]
        file.write(np.float32(v))
        for ilevel in range(self.nr_levels):
            v = np.squeeze(self.z_depth[ilevel, :, :])[iok]
            file.write(np.float32(v))

        # U
        v = self.u_zmin[iok]
        file.write(np.float32(v))
        v = self.u_zmax[iok]
        file.write(np.float32(v))
        dhdz = np.full(np.shape(v), 1.0)
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ilevel in range(self.nr_levels):
            v = np.squeeze(self.u_hrep[ilevel, :, :])[iok]
            file.write(np.float32(v))
        for ilevel in range(self.nr_levels):
            v = np.squeeze(self.u_navg[ilevel, :, :])[iok]
            file.write(np.float32(v))

        # V
        v = self.v_zmin[iok]
        file.write(np.float32(v))
        v = self.v_zmax[iok]
        file.write(np.float32(v))
        file.write(np.float32(dhdz))  # Not used in SFINCS anymore
        for ilevel in range(self.nr_levels):
            v = np.squeeze(self.v_hrep[ilevel, :, :])[iok]
            file.write(np.float32(v))
        for ilevel in range(self.nr_levels):
            v = np.squeeze(self.v_navg[ilevel, :, :])[iok]
            file.write(np.float32(v))

        file.close()

    # This is the new way of building subgrid tables, that will end up in netcdf files
    @hydromt_step
    def create(
        self,
        elevation_list: List[dict],
        roughness_list: List[dict] = [],
        river_list: List[dict] = [],
        buffer_cells: int = 0,
        nr_levels: int = 10,
        nbins: int = None,
        nr_subgrid_pixels: int = 20,
        nrmax: int = 2000,  # blocksize
        max_gradient: float = 99999.0,
        z_minimum: float = -99999.0,
        huthresh: float = 0.01,
        q_table_option: int = 2,
        weight_option: str = "min",
        manning_land: float = 0.04,
        manning_sea: float = 0.02,
        rgh_lev_land: float = 0.0,
        write_dep_tif: bool = False,
        write_man_tif: bool = False,
    ):
        """Create method for subgrid tables based on a list of
        elevation and Manning's roughness datasets.

        These datasets are used to derive relations between the water level
        and the volume in a cell to do the continuity update,
        and a representative water depth used to calculate momentum fluxes.

        This allows that one can compute on a coarser computational grid,
        while still accounting for the local topography and roughness.

        Parameters
        ----------
        elevation_list : List[dict]
            List of dictionaries with topobathy data.
            Each should minimally contain a data catalog source name, data file path,
            or xarray raster object ('elevation').
            Optional merge arguments include: 'zmin', 'zmax', 'mask', 'offset', 'reproj_method',
            and 'merge_method', see example below. For a complete overview of all merge options,
            see :py:func:`hydromt.workflows.merge_multi_dataarrays`

            ::

                [
                    {'elevation': 'merit_hydro', 'zmin': 0.01},
                    {'elevation': 'gebco', 'offset': 0, 'merge_method': 'first', reproj_method: 'bilinear'}
                ]

        roughness_list : List[dict], optional
            List of dictionaries with Manning's n datasets. Each dictionary should at
            least contain one of the following:

            * manning: filename (or Path) of gridded data with manning values
            * lulc (and reclass_table): a combination of a filename of gridded
              landuse/landcover and a mapping table.

            In additon, optional merge arguments can be provided, e.g.:

            ::

                [
                    {'manning': 'manning_data'},
                    {'lulc': 'esa_worlcover', 'reclass_table': 'esa_worlcover_mapping'}
                ]

        river_list : List[dict], optional
            List of dictionaries with river datasets. Each dictionary should at least
            contain a river centerline data and optionally a river mask:

            * centerlines: filename (or Path) of river centerline with attributes
              rivwth (river width [m]; required if not river mask provided),
              rivdph or rivbed (river depth [m]; river bedlevel [m+REF]),
              manning (Manning's n [s/m^(1/3)]; optional)
            * mask (optional): filename (or Path) of river mask
            * point_zb (optional): filename (or Path) of river points with bed (z) values
            * river attributes (optional): "rivdph", "rivbed", "rivwth", "manning"
              to fill missing values
            * arguments to the river burn method (optional):
              segment_length [m] (default 500m) and riv_bank_q [0-1] (default 0.5)
              which used to estimate the river bank height in case river depth is provided.

            For more info see :py:func:`hydromt.workflows.bathymetry.burn_river_rect`

           ::

                [{'centerlines': 'river_lines', 'mask': 'river_mask', 'manning': 0.035}]

        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels,
            by default 0
        nbins : int, optional
            Number of bins in which hypsometry is subdivided, by default 10
            Note that this keyword is deprecated and will be removed in future versions.
        nr_levels: int, optional
            Number of levels to describe hypsometry, by default 10
        nr_subgrid_pixels : int, optional
            Number of subgrid pixels per computational cell, by default 20.
            Note that this value must be a multiple of 2.
        nrmax : int, optional
            Maximum number of cells per subgrid-block, by default 2000
            These blocks are used to prevent memory issues while working with large datasets
        max_gradient : float, optional
            If slope in hypsometry exceeds this value, then smoothing is applied,
            to prevent numerical stability problems, by default 5.0
        z_minimum : float, optional
            Minimum depth in the subgrid tables, by default -99999.0
        huthresh : float, optional
            Threshold depth in SFINCS model, by default 0.01 m
        q_table_option : int, optional
            Option for the computation of the representative roughness and conveyance depth at u/v points, by default 2.
            1: "old" weighting method, compliant with SFINCS < v2.1.1, taking the avarage of the adjacent cells
            2: "improved" weighting method, recommended for SFINCS >= v2.1.1, that takes into account the wet fractions of the adjacent cells
        weight_option : str, optional
            Weighting factor of the adjacent cells for the flux q at u/v points, by default "min"
        manning_land, manning_sea : float, optional
            Constant manning roughness values for land and sea, by default 0.04 and 0.02 s.m-1/3
            Note that these values are only used when no Manning's n datasets are provided,
            or to fill the nodata values
        rgh_lev_land : float, optional
            Elevation level to distinguish land and sea roughness
            (when using manning_land and manning_sea), by default 0.0
        write_dep_tif, write_man_tif : bool, optional
            Write geotiff of the merged topobathy / roughness on the subgrid resolution.
            These files are not used by SFINCS, but can be used for visualisation and
            downscaling of the floodmaps. Unlinke the SFINCS files it is written
            to disk at execution of this method. By default False
        """

        if not self.model.grid.mask.raster.crs.is_geographic:
            res = np.abs(self.model.grid.mask.raster.res[0]) / nr_subgrid_pixels
        else:
            res = (
                np.abs(self.model.grid.mask.raster.res[0])
                * 111111.0
                / nr_subgrid_pixels
            )

        elevation_list = self.model._parse_datasets_elevation(elevation_list, res=res)

        if len(roughness_list) > 0:
            # NOTE conversion from landuse/landcover to manning happens here
            roughness_list = self.model._parse_roughness_list(roughness_list)

        if len(river_list) > 0:
            river_list = self.model._parse_river_list(river_list)

        # folder where high-resolution topobathy and manning geotiffs are stored
        if write_dep_tif or write_man_tif:
            highres_dir = os.path.join(self.model.root.path, "subgrid")
            if not os.path.isdir(highres_dir):
                os.makedirs(highres_dir)
        else:
            highres_dir = None

        if nbins is not None:
            logger.warning(
                "Keyword nbins is deprecated and will be removed in future versions. Please use nr_levels instead."
            )
            nr_levels = nbins

        if q_table_option == 1 and max_gradient > 20.0:
            raise ValueError(
                "For the old q_table_option, a max_gradient of 5.0 is recommended to improve numerical stability"
            )

        # get the mask from the model
        da_mask = self.model.grid.mask

        self.version = 1

        # check if nr_subgrid_pixels is a multiple of 2
        # this is needed for symmetry around the uv points
        if nr_subgrid_pixels % 2 != 0:
            raise ValueError(
                "nr_subgrid_pixels must be a multiple of 2 for subgrid table"
            )

        refi = nr_subgrid_pixels
        self.nr_levels = nr_levels
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
            (nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )

        # U points
        self.u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.u_havg = np.full(
            (nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_nrep = np.full(
            (nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_pwet = np.full(
            (nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.u_ffit = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)
        self.u_navg = np.full((grid_dim), fill_value=np.nan, dtype=np.float32)

        # V points
        self.v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
        self.v_havg = np.full(
            (nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_nrep = np.full(
            (nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
        self.v_pwet = np.full(
            (nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32
        )
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
                    da_list=elevation_list,
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
                if len(roughness_list) > 0:
                    da_man = workflows.merge_multi_dataarrays(
                        da_list=roughness_list,
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
                if len(river_list) > 0:
                    logger.debug("Burn rivers in bathymetry and manning data")
                    for riv_kwargs in river_list:
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
                    nr_levels,
                    yg,
                    max_gradient,
                    huthresh,
                    q_table_option,
                    weight_option,
                    da_mask.raster.crs.is_geographic,
                )

                del da_mask_block, da_dep, da_man
                gc.collect()

        # convert to xarray dataset and set to data
        self._data = self.to_xarray(
            dims=self.model.grid.mask.raster.dims,
            coords=self.model.grid.mask.raster.coords,
        )

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

        # set other manning options to None in config
        self.model.config.set("manning", None)
        self.model.config.set("manning_land", None)
        self.model.config.set("manning_sea", None)
        self.model.config.set("rgh_lev_land", None)
        self.model.config.set("manningfile", None)
        logger.info(
            "Set other manning options to None in config that are unused  in SFINCS in case"
            " of subgrid (manning, manning_land, manning_sea, rgh_lev_land, manningfile)."
        )

    def to_xarray(self, dims, coords):
        """Convert old binary subgrid class to xarray dataset."""
        ds_sbg = xr.Dataset(coords={"levels": np.arange(self.nr_levels), **coords})
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
    nr_levels,
    yg,
    max_gradient,
    huthresh,
    q_table_option,
    weight_option,
    is_geographic=False,
):
    """calculate subgrid properties for a single tile"""
    # Z points
    grid_dim = mask.shape
    z_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_volmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    z_level = np.full((nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32)

    # U points
    u_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_havg = np.full((nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_nrep = np.full((nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_pwet = np.full((nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    u_ffit = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    u_navg = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)

    # V points
    v_zmin = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_zmax = np.full(grid_dim, fill_value=np.nan, dtype=np.float32)
    v_havg = np.full((nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    v_nrep = np.full((nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32)
    v_pwet = np.full((nr_levels, *grid_dim), fill_value=np.nan, dtype=np.float32)
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
                zgc.flatten(), dxpm, dypm, nr_levels, zvmin, max_gradient
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
                zgu.flatten(),
                manning.flatten(),
                nr_levels,
                huthresh,
                q_table_option,
                weight_option=weight_option,
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
                zgu.flatten(),
                manning.flatten(),
                nr_levels,
                huthresh,
                q_table_option,
                weight_option=weight_option,
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
