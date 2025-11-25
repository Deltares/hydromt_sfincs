import logging
from typing import TYPE_CHECKING, List, Union
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent
from hydromt.model.processes.mesh import mesh2d_from_rasterdataset

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {
    "initial_conditions": {"standard_name": "initial water level", "unit": "m+ref"}
}


class SfincsQuadtreeInitialConditions(ModelComponent):
    """SFINCS initial conditions component for quadtree grids."""

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.quadtree_grid.data["ini"]
        super().__init__(
            model=model,
        )

    @property
    def data(self):
        """Get the data from the model grid."""
        return self.model.quadtree_grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.quadtree_grid.mask

    def read(self):
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The mask values are read when the quadtree grid is read
        pass

    def write(self):
        # The mask values are written when the quadtree grid is written
        pass

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
        ini: Union[str, Path] = None,
        fill_value: float = -9999.0,
        reproj_method="average",
    ):
        """Setup spatially varying initial water level (ncinifile).

        Adds and overwrites model layers to SfincsModel.quadtree_grid.data:

        * **ini** map: initial water level [m+ref]

        Parameters
        ----------
        ini : str, Path, RasterDataset
            Spatially varying initial water level [m+ref]
        fill_value: float, optional
            Fill value for areas outside the polygon, by default -9999.0.
        reproj_method : str, optional
            Resampling method for reprojecting the initial water level data to the model grid.
            By default 'average'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`

        """

        mname = "ini"

        # Add logger info
        logger.info("Creating spatially varying initial water level.")

        # get rasterdataset with initial water level data
        da_ini = self.data_catalog.get_rasterdataset(
            ini,
            bbox=self.model.bbox,
            buffer=10,
            variables=[mname],
        )

        # reproject initial water level data to model grid
        # da_ini = da_ini.raster.mask_nodata()  # set nodata to nan
        # reproject single-dataset to mesh
        mesh2d = self.mask.grid
        uda_ini = mesh2d_from_rasterdataset(
            ds=da_ini,
            mesh2d=mesh2d,
            resampling_method=reproj_method,
        )

        # check on nan values
        if np.logical_and(np.isnan(uda_ini), self.mask >= 1).any():
            logger.warning(
                "NaN values found in initial water level data; filled with fill_value {}".format(
                    fill_value
                )
            )
            uda_ini = uda_ini.fillna(fill_value)
        # FIXME add nodata
        # uda_ini..set_nodata(np.nan)

        # set grid
        da_ini.attrs.update(**_ATTRS.get(mname, {}))
        self.model.quadtree_grid.data[mname] = uda_ini[mname]
        # FIXME: ideally we would use the set method, but that's not working here properly
        # self.model.quadtree_grid.set(da_ini, name=mname)

        # update config: remove default zsini and set inifile
        self.model.config.set(f"nc{mname}file", f"sfincs_ini.nc")
        # set spatially uniform zsini to None in config
        self.model.config.set("zsini", None)

    # Initial water level
    @hydromt_step
    def create_from_polygon(
        self,
        ini: Union[str, Path, gpd.GeoDataFrame] = None,
        ini_buffer: float = 0.0,
        ini_value: Union[float, List[float]] = None,
        fill_value: float = -9999.0,
        reset_ini: bool = True,
    ):
        """Setup spatially varying initial water level (ncinifile).

        Adds model layers to SfincsModel.quadtree_grid.data:

        * **ini** map: initial water level [m+ref]

        Parameters
        ----------
        ini : str, Path, GeoDataFrame with 'ini' column
            Spatially varying initial water level [m+ref]
        ini_buffer: float, optional
            If larger than zero, extend the `ini` gdf geometry with a buffer [m],
            by default 0.
        ini_value: float or list of float, optional
            If provided, use this value(s) for the initial water level within the polygon(s),
        fill_value: float, optional
            Fill value for areas outside the polygon, by default -9999.0.
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

        # check if input is polygon or multipolygon
        if not gdf_ini.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError(
                "Input geodataframe 'ini' should contain only Polygon or MultiPolygon geometries."
            )

        # apply buffer if requested
        if ini_buffer > 0:  # NOTE assumes model in projected CRS!
            gdf_ini["geometry"] = gdf_ini.to_crs(self.model.crs).buffer(ini_buffer)

        # check if 'ini' values are provided or ini column is present
        if ini_value is not None:
            # use provided value(s)
            if isinstance(ini_value, list):
                if len(ini_value) != len(gdf_ini):
                    raise ValueError(
                        "Length of 'ini_value' list should match number of polygons in input geodataframe 'ini'."
                    )
                gdf_ini["ini"] = ini_value
            else:
                gdf_ini["ini"] = ini_value
        elif "ini" not in gdf_ini.columns:
            raise ValueError(
                "Input geodataframe 'ini' should contain a column 'ini' with initial water level values per polygon."
            )

        # if reset_ini = True start empty, otherwise start with existing ini layer
        if reset_ini:
            # start with empty ini layer
            da_ini = xu.full_like(
                self.mask,
                fill_value=np.nan,
                dtype="float32",
            )
        else:
            # start with existing ini layer
            da_ini = self.data[mname]

        # loop over polygons and rasterize initial water level values
        for gdf in gdf_ini.iterrows():
            # Parse wanted value within polygon:
            inival = float(gdf["ini"])
            # Burn the polygon into the raster: pixels inside polygon get 1, others get 0
            burned = xu.burn_vector_geometry(
                gdf["geometry"], self.data, fill=0, all_touched=True
            )
            da_ini = xr.where(burned > 0, inival, da_ini)

        # check on nan values
        if np.logical_and(np.isnan(da_ini), self.mask >= 1).any():
            logger.warning(
                "NaN values found in initial water level data; filled with fill_value (default -9999.0)"
            )
            da_ini = da_ini.fillna(fill_value)
            # FIXME - should we still set the nodata value? if so to what?

        # set grid
        da_ini.attrs.update(**_ATTRS.get(mname, {}))
        self.model.quadtree_grid.data[mname] = da_ini
        # FIXME: ideally we would use the set method, but that's not working here properly
        # self.model.quadtree_grid.set(da_ini, name=mname)

        # update config: remove default zsini and set inifile
        self.model.config.set(f"nc{mname}file", f"sfincs_ini.nc")
        # set spatially uniform zsini to None in config
        self.model.config.set("zsini", None)
