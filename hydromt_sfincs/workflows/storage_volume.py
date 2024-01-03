import logging
from typing import Union, List

import geopandas as gpd
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def add_storage_volume(
    da_vol: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    volume: Union[float, List[float]] = None,
    height: Union[float, List[float]] = None,
    logger=logger,
) -> xr.DataArray:
    """Add storage volume to a grid based on a GeoDataFrame with storage locations.

    Parameters
    ----------
    da_vol : xr.DataArray
        DataArray with the grid to which the storage volume should be added.
    gdf : gpd.GeoDataFrame
        GeoDataFrame with the storage locations (polygon or point geometry file).
        Optional "volume" or "height" attributes can be provided to set the storage volume.
    volume : Union[float, List[float]], optional
        Volume of the storage locations [m3], by default None.
    height : Union[float, List[float]], optional
        Height of the storage locations [m], by default None.

    Returns
    -------
    xr.DataArray
        DataArray with the grid including the storage volume.

    """

    # loop over the gdf rows and rasterize each geometry
    for i, _ in gdf.iterrows():
        # create a gdf with only the current row
        single_gdf = gpd.GeoDataFrame(gdf.loc[[i]]).reset_index(drop=True)

        single_vol = single_gdf.get("volume", np.nan)
        single_height = single_gdf.get("height", np.nan)

        # check if volume or height is provided in the gdf or as input
        if np.isnan(single_vol).all():  # volume not provided or nan
            if np.isnan(single_height).all():  # height not provided or nan
                if volume is not None:  # volume provided as input (list)
                    single_vol = volume if not isinstance(volume, list) else volume[i]
                elif height is not None:  # height provided as input (list)
                    single_height = (
                        height if not isinstance(height, list) else height[i]
                    )
                else:  # no volume or height provided
                    logger.warning(
                        f"No volume or height provided for storage location {i}"
                    )
                    continue
            else:  # height provided in gdf
                single_height = single_height[0]
        else:  # volume provided in gdf
            single_vol = single_vol[0]

        # check if gdf has point or polyhon geometry
        if single_gdf.geometry.type[0] == "Point":
            # get x and y coordinate of the point in crs of the grid
            x, y = single_gdf.geometry.iloc[0].coords[0]

            # check if the grid is rotated
            if da_vol.raster.rotation != 0:
                # rotate the point
                x, y = ~da_vol.raster.transform * (x, y)
                # select the grid cell nearest to the point
                closest_point = da_vol.reindex(x=da_vol.x, y=da_vol.y).sel(
                    x=x, y=y, method="nearest"
                )
            else:
                # select the grid cell nearest to the point
                closest_point = da_vol.sel(x=x, y=y, method="nearest")

            # add the volume to the grid cell
            if not np.isnan(single_vol).all():
                da_vol.loc[
                    dict(x=closest_point.x.item(), y=closest_point.y.item())
                ] += single_vol
            else:
                logger.warning("No volume provided for storage location of type Point")

        elif single_gdf.geometry.type[0] == "Polygon":
            # rasterize the geometry
            area = da_vol.raster.rasterize_geometry(single_gdf, method="area")
            total_area = area.sum().values

            # calculate the volume per cell and add it to the grid
            if not np.isnan(single_vol).all():
                da_vol += area / total_area * single_vol
            elif not np.isnan(single_height).all():
                da_vol += area * single_height
            else:
                logger.warning(
                    "No volume or height provided for storage location of type Polygon"
                )

    return da_vol
