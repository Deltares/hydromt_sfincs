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

_ATTRS = {"zs": {"standard_name": "initial water level", "unit": "m+ref"}}


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
        zsini: Union[str, Path, xr.DataArray],
        fill_value: float = -9999.0,
        reproj_method="average",
    ):
        """Setup spatially varying initial water level (inifile).

        Adds and overwrites model layers to SfincsModel.grid.data:

        * **zs** map: initial water level [m+ref]

        Parameters
        ----------
        zsini : str, Path, xr.DataArray
            Spatially varying initial water level [m+ref]
        reproj_method : str, optional
            Resampling method for reprojecting the initial water level data to the model grid.
            By default 'average'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        fill_value: float, optional
            Fill value for areas without data, by default -9999.0. For cells with initial water levels of -9999.0,
            the SFINCS kernel will set the initial water level to the bed level.
        """

        mname = "zs"

        # Add logger info
        logger.info("Creating spatially varying initial water level.")

        # get initial water level data
        da_zsini = self.data_catalog.get_rasterdataset(
            zsini,
            bbox=self.model.bbox,
            buffer=10,
        )

        # reproject initial water level data to model grid
        da_zsini = da_zsini.raster.mask_nodata()  # set nodata to nan
        da_zsini = da_zsini.raster.reproject_like(self.mask, method=reproj_method)

        # check on nan values
        if np.logical_and(np.isnan(da_zsini), self.mask >= 1).any():
            logger.warning(
                "NaN values found in initial water level data; filled with fill_value {}".format(
                    fill_value
                )
            )
            da_zsini = da_zsini.fillna(fill_value)
        da_zsini.raster.set_nodata(np.nan)

        # set grid
        da_zsini.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_zsini, name=mname)

        # update config: remove default zsini and set inifile
        self.model.config.set(f"inifile", f"sfincs.ini")
        # set spatially uniform zsini to 0.0 in config
        self.model.config.set("zsini", 0.0)

    # Initial water level from polygon
    @hydromt_step
    def create_from_polygon(
        self,
        zsini: Union[str, Path, gpd.GeoDataFrame],
        zsini_value: Union[float, List[float]] = None,
        zsini_buffer: int = 0,
        fill_value: float = -9999.0,
        reset_zsini: bool = True,
    ):
        """Setup spatially varying initial water level (inifile).

        Adds model layers to SfincsModel.grid.data:

        * **zs** map: initial water level [m+ref]

        Parameters
        ----------
        zsini : str, Path, GeoDataFrame with optional 'zs' column
            Spatially varying initial water level [m+ref]
        zsini_value: float or List[float], optional
            If provided, use this value (or list of values) for the initial water level inside the polygon(s).
        zsini_buffer: float, optional
            If larger than zero, extend the `zsini` gdf geometry with a buffer [m],
            by default 0.
        fill_value: float, optional
            Fill value for areas outside the polygon, by default -9999.0. For cells with initial water levels of -9999.0,
            the SFINCS kernel will set the initial water level to the bed level.
        reset_zsini: bool, optional
            If True (default), reset existing zs layer. If False updating existing zs layer.

        """

        mname = "zs"

        # Add logger info
        logger.info("Creating spatially varying initial water level.")

        # get initial water level data geodataframe,
        # with a value 'zsini' to rasterize
        gdf_zsini = self.data_catalog.get_geodataframe(
            zsini,
            bbox=self.model.bbox,
        )

        if zsini_buffer > 0:  # NOTE assumes model in projected CRS!
            gdf_zsini["geometry"] = gdf_zsini.to_crs(self.model.crs).buffer(
                zsini_buffer
            )

        # check if input is polygon or multipolygon
        if not gdf_zsini.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError(
                "Input geodataframe 'zsini' should contain only Polygon or MultiPolygon geometries."
            )

        # if ini_value is provided, use this value (or list of values) for the initial water level inside the polygon(s).
        if zsini_value is not None:
            if isinstance(zsini_value, list):
                if len(zsini_value) != len(gdf_zsini):
                    raise ValueError(
                        "If zsini_value is a list, its length should match the number of polygons in 'zsini'."
                    )
                gdf_zsini["zsini"] = zsini_value
            else:
                gdf_zsini["zsini"] = float(zsini_value)

        # check if 'ini' column is present
        if "zsini" not in gdf_zsini.columns:
            raise ValueError(
                "Input geodataframe 'zsini' should contain a column 'zsini' with initial water level values per polygon."
            )

        # if reset_ini = True start empty, otherwise start with existing ini layer
        if reset_zsini:
            # start with empty ini layer
            da_zsini = xr.full_like(
                self.mask,
                fill_value=np.nan,
                dtype="float32",
            )
        else:
            # start with existing ini layer
            da_zsini = self.data[mname]

        # loop over all polygons and rasterize
        for _, row in gdf_zsini.iterrows():
            zsini_single = row["zsini"]
            gdf_zsini_single = gpd.GeoDataFrame(
                [row], columns=gdf_zsini.columns, crs=gdf_zsini.crs
            )
            da_zsini0 = self.mask.raster.geometry_mask(gdf_zsini_single)
            # where da_zsini0 is True, set values of da_zsini to zsini_single:
            da_zsini = xr.where(da_zsini0, zsini_single, da_zsini)

        # check on nan values
        if np.logical_and(np.isnan(da_zsini), self.mask >= 1).any():
            logger.warning(
                "NaN values found in initial water level data; filled with fill_value {}".format(
                    fill_value
                )
            )
            da_zsini = da_zsini.fillna(fill_value)
        da_zsini.raster.set_nodata(np.nan)

        # set grid
        da_zsini.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_zsini, name=mname)

        # update config: remove default zsini and set inifile
        self.model.config.set(f"inifile", f"sfincs.ini")
        # set spatially uniform zsini to 0.0 in config
        self.model.config.set("zsini", 0.0)
