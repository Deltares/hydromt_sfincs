import logging
from typing import TYPE_CHECKING, Dict, Union, Optional
from os.path import isabs, isfile

from pyproj import CRS
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsOutput(ModelComponent):
    """SFINCS model output component.

    This component handles reading and storing model results from SFINCS. The results
    are stored in a dictionary and can be accessed via the `data` property.
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._data: dict = None
        super().__init__(
            model=model,
        )

    @property
    def data(
        self,
    ) -> Dict[str, Union[xr.Dataset, xr.DataArray, xu.UgridDataArray, xu.UgridDataset]]:
        """Model results. Returns dict of xarray.DataArray or xarray.Dataset."""
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize results."""
        if self._data is None:
            self._data = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(
        self,
        chunksize=100,
        drop=["crs", "sfincsgrid"],
        fn_map="sfincs_map.nc",
        fn_his="sfincs_his.nc",
        **kwargs,
    ):
        """Read results from sfincs_map.nc and sfincs_his.nc and save to the `results` attribute.
        The staggered nc file format is translated into hydromt.RasterDataArray formats.
        Additionally, hmax is computed from zsmax and zb if present.

        Parameters
        ----------
        chunksize: int, optional
            chunk size along time dimension, by default 100
        drop: list, optional
            list of variables to drop, by default ["crs", "sfincsgrid"]
        fn_map: str, optional
            filename of sfincs_map.nc, by default "sfincs_map.nc"
        fn_his: str, optional
            filename of sfincs_his.nc, by default "sfincs_his.nc"
        """

        # Check that read mode is on
        self.root._assert_read_mode()

        # Read the config file to determine the grid type
        self.model.config.read()

        if not isabs(fn_map):
            fn_map = self.model.root.path / fn_map
        if isfile(fn_map):
            self.read_map_file(
                fn_map=fn_map,
                drop=drop,
                chunksize=chunksize,
                **kwargs,
            )
        else:
            logger.warning(f"File {fn_map} not found.")

        if not isabs(fn_his):
            fn_his = self.model.root.path / fn_his
        if isfile(fn_his):
            ds_his = self.read_his_file(
                fn_his=fn_his,
                drop=drop,
                chunksize=chunksize,
            )
            self.set(ds_his, split_dataset=True)
        else:
            logger.warning(f"File {fn_his} not found.")

    def write(self):
        """Writing results to sfincs_map.nc and sfincs_his.nc files is not part of
        the hydromt.sfincs package. This is done by the SFINCS kernel itself."""
        pass

    def read_his_file(
        self,
        fn_his: str = "sfincs_his.nc",
        drop: list = ["crs", "sfincsgrid"],
        chunksize: int = 100,
    ) -> xr.Dataset:
        """Read the sfincs_his.nc file and return it as an xarray Dataset."""
        ds_his = utils.read_sfincs_his_results(
            fn_his, crs=self.model.crs, chunksize=chunksize
        )
        # drop double vars (map files has priority)
        drop_vars = [v for v in ds_his.data_vars if v in self.data or v in drop]
        ds_his = ds_his.drop_vars(drop_vars)
        return ds_his

    def read_map_file(
        self,
        fn_map: str = "sfincs_map.nc",
        drop: list = ["crs", "sfincsgrid"],
        **kwargs,
    ) -> xr.Dataset:
        """Read the sfincs_map.nc file and return it as an xarray Dataset."""

        if self.model.grid_type is None:
            logger.warning("Grid type not known, trying to read from config file. ")
            self.model.config.read()
        if self.model.grid_type == "regular":
            ds_face, ds_edge = utils.read_sfincs_map_results(
                fn_map,
                ds_like=self.model.grid.mask,
                drop=drop,
                logger=logger,
                **kwargs,
            )
            # save as dict of DataArray
            self.set(ds_face, split_dataset=True)
            self.set(ds_edge, split_dataset=True)
        elif self.model.grid_type == "quadtree":
            with xu.load_dataset(fn_map) as ds:
                # set coords
                ds = ds.set_coords(["mesh2d_node_x", "mesh2d_node_y"])
                # get crs variable, drop it and set it correctly
                crs = ds["crs"].values
                ds.drop_vars("crs")
                ds.grid.set_crs(CRS.from_user_input(crs))
                self.set(ds, split_dataset=True)

    def set(
        self,
        data: Union[xr.DataArray, xr.Dataset],
        name: Optional[str] = None,
        split_dataset: Optional[bool] = False,
    ):
        """Add data to results attribute.

        Dataset can either be added as is (default) or split into several
        DataArrays using the split_dataset argument.

        Arguments
        ---------
        data: xarray.Dataset or xarray.DataArray
            New forcing data to add
        name: str, optional
            Results name, required if data is xarray.Dataset and split_dataset=False.
        split_dataset: bool, optional
            If True (False by default), split a Dataset to store each variable
            as a DataArray.
        """
        # Initialize results if not done yet
        self._initialize()

        data_dict = _check_data(data, name, split_dataset)
        for name in data_dict:
            if name in self._data:
                logger.warning(f"Replacing result: {name}")
            self._data[name] = data_dict[name]


def _check_data(
    data: Union[xr.DataArray, xr.Dataset, xu.UgridDataArray, xu.UgridDataset],
    name: Optional[str] = None,
    split_dataset=True,
) -> Dict:
    if isinstance(data, xr.DataArray) or isinstance(data, xu.UgridDataArray):
        # NOTE name can be different from data.name !
        if data.name is None and name is not None:
            data.name = name
        elif name is None and data.name is not None:
            name = data.name
        elif data.name is None and name is None:
            raise ValueError("Name required for DataArray.")
        data = {name: data}
    elif isinstance(data, xr.Dataset) or isinstance(
        data, xu.UgridDataset
    ):  # return dict for consistency
        if split_dataset:
            data = {name: data[name] for name in data.data_vars}
        elif name is None:
            raise ValueError("Name required for Dataset.")
        else:
            data = {name: data}
    else:
        raise ValueError(f'Data type "{type(data).__name__}" not recognized')
    return data


# %% DDB GUI focused additional functions:
# read_his_file
# read_zsmax
# read_cumulative_precipitation
