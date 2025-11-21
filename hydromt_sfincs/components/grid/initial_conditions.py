import logging
from typing import TYPE_CHECKING, List, Union
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {
    "initial_conditions": {"standard_name": "initial water level", "unit": "m+ref"}
}


class SfincsInitialConditions(ModelComponent):
    """SFINCS Initial Conditions Component.

    This component contains methods to add initial water level data to the SFINCS model
    on regular grids.

    .. note::
        The initial water level data is stored in the model grid's data dataset under the key "ini".

    See Also
    --------
    :py:class:`~hydromt_sfincs.components.grid.regulargrid.SfincsGrid`

    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.grid.data["ini"]
        super().__init__(
            model=model,
        )

    @property
    def data(self):
        """Get the data from the model grid."""
        return self.model.grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.grid.mask

    def read(self):
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The ini file is read when all grid files are read in regulargrid.py
        pass

    def write(self):
        # The ini file is written when all grid files are written in regulargrid.py
        pass

    # Original HydroMT-SFINCS setup_ functions:
    # was not yet implemented

    # %% core HydroMT-SFINCS functions:
    # read
    # write
    # set > already in grid.set()
    # create
    # create_from_polygon

    # Initial water level
    @hydromt_step
    def create(
        self,
        ini: Union[str, Path, xr.DataArray],
        fill_value: float = -9999.0,
        reproj_method="average",
    ):
        """Setup spatially varying initial water level (inifile).

        Adds and overwrites model layers to SfincsModel.grid.data:

        * **ini** map: initial water level [m+ref]

        Parameters
        ----------
        ini : str, Path, RasterDataset
            Spatially varying initial water level [m+ref]
        reproj_method : str, optional
            Resampling method for reprojecting the initial water level data to the model grid.
            By default 'average'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        fill_value: float, optional
            Fill value for areas without data, by default -9999.0. For cells with initial water levels of -9999.0,
            the SFINCS kernel will set the initial water level to the bed level.
        """

        mname = "ini"

        # Add logger info
        logger.info("Creating spatially varying initial water level.")

        # get initial water level data
        da_ini = self.data_catalog.get_rasterdataset(
            ini,
            bbox=self.model.bbox,
            buffer=10,
        )

        # reproject initial water level data to model grid
        da_ini = da_ini.raster.mask_nodata()  # set nodata to nan
        da_ini = da_ini.raster.reproject_like(self.mask, method=reproj_method)

        # check on nan values
        if np.logical_and(np.isnan(da_ini), self.mask >= 1).any():
            logger.warning(
                "NaN values found in initial water level data; filled with fill_value {}".format(
                    fill_value
                )
            )
            da_ini = da_ini.fillna(fill_value)
        da_ini.raster.set_nodata(np.nan)

        # set grid
        da_ini.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_ini, name=mname)

        # update config: remove default zsini and set inifile
        self.model.config.set(f"{mname}file", f"sfincs.{mname}")
        # set spatially uniform zsini to None in config
        self.model.config.set("zsini", None)

    # Initial water level from polygon
    @hydromt_step
    def create_from_polygon(
        self,
        ini: Union[str, Path, gpd.GeoDataFrame],
        ini_value: Union[float, List[float]] = None,
        ini_buffer: int = 0,
        fill_value: float = -9999.0,
        reset_ini: bool = True,
    ):
        """Setup spatially varying initial water level (inifile).

        Adds model layers to SfincsModel.grid.data:

        * **ini** map: initial water level [m+ref]

        Parameters
        ----------
        ini : str, Path, GeoDataFrame with optional 'ini' column
            Spatially varying initial water level [m+ref]
        ini_value: float or List[float], optional
            If provided, use this value (or list of values) for the initial water level inside the polygon(s).
        ini_buffer: float, optional
            If larger than zero, extend the `ini` gdf geometry with a buffer [m],
            by default 0.
        fill_value: float, optional
            Fill value for areas outside the polygon, by default -9999.0. For cells with initial water levels of -9999.0,
            the SFINCS kernel will set the initial water level to the bed level.
        reset_ini: bool, optional
            If True (default), reset existing ini layer. If False updating existing ini layer.

        """

        mname = "ini"

        # Add logger info
        logger.info("Creating spatially varying initial water level.")

        # get initial water level data geodataframe,
        # with a value 'ini' to rasterize
        gdf_ini = self.data_catalog.get_geodataframe(
            ini,
            bbox=self.model.bbox,
        )

        if ini_buffer > 0:  # NOTE assumes model in projected CRS!
            gdf_ini["geometry"] = gdf_ini.to_crs(self.model.crs).buffer(ini_buffer)

        # check if input is polygon or multipolygon
        if not gdf_ini.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError(
                "Input geodataframe 'ini' should contain only Polygon or MultiPolygon geometries."
            )

        # if ini_value is provided, use this value (or list of values) for the initial water level inside the polygon(s).
        if ini_value is not None:
            if isinstance(ini_value, list):
                if len(ini_value) != len(gdf_ini):
                    raise ValueError(
                        "If ini_value is a list, its length should match the number of polygons in 'ini'."
                    )
                gdf_ini["ini"] = ini_value
            else:
                gdf_ini["ini"] = float(ini_value)

        # check if 'ini' column is present
        if "ini" not in gdf_ini.columns:
            raise ValueError(
                "Input geodataframe 'ini' should contain a column 'ini' with initial water level values per polygon."
            )

        # if reset_ini = True start empty, otherwise start with existing ini layer
        if reset_ini:
            # start with empty ini layer
            da_ini = xr.full_like(
                self.mask,
                fill_value=np.nan,
                dtype="float32",
            )
        else:
            # start with existing ini layer
            da_ini = self.data[mname]

        # loop over all polygons and rasterize
        for _, row in gdf_ini.iterrows():
            ini_single = row["ini"]
            gdf_ini_single = gpd.GeoDataFrame(
                [row], columns=gdf_ini.columns, crs=gdf_ini.crs
            )
            da_ini0 = self.mask.raster.geometry_mask(gdf_ini_single)
            # where da_ini0 is True, set values of da_ini to ini_single:
            da_ini = xr.where(da_ini0, ini_single, da_ini)

        # check on nan values
        if np.logical_and(np.isnan(da_ini), self.mask >= 1).any():
            logger.warning(
                "NaN values found in initial water level data; filled with fill_value {}".format(
                    fill_value
                )
            )
            da_ini = da_ini.fillna(fill_value)
        da_ini.raster.set_nodata(np.nan)

        # set grid
        da_ini.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_ini, name=mname)

        # update config: remove default zsini and set inifile
        self.model.config.set(f"{mname}file", f"sfincs.{mname}")
        # set spatially uniform zsini to None in config
        self.model.config.set("zsini", None)
