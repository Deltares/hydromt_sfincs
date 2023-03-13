
import os
import numpy as np
from pathlib import Path
from pyproj import CRS, Transformer
from typing import Union, Optional, List, Dict, Tuple
import xarray as xr
import geopandas as gpd

from .merge import merge_multi_dataarrays
from hydromt_sfincs.utils import deg2num,num2deg,elevation2png,tile_window

def create_topobathy_tiles(
        root:Union[str, Path],
        region:gpd.GeoDataFrame,
        da_dep_lst:List[dict],
        index_path: Union[str, Path] = None,
        zoom_range: Union[int, List[int]] = [0, 13],
        z_range: List[int] = [-20000.0, 20000.0],
        fmt = "bin",
    ):
    """_summary_

    Parameters
    ----------
    root : Union[str, Path]
        _description_
    region : gpd.GeoDataFrame
        _description_
    da_dep_lst : List[dict]
        _description_
    index_path : Union[str, Path], optional
        _description_, by default None
    zoom_range : Union[int, List[int]], optional
        _description_, by default [0, 13]
    z_range : List[int], optional
        _description_, by default [-20000.0, 20000.0]
    format : str, optional
        _description_, by default "bin"
    """    

    assert len(da_dep_lst) > 0, "No DEMs provided"

    topobathy_path = os.path.join(root, "topobathy")
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
    
    # get bounding box of region
    minx, miny, maxx, maxy = region.total_bounds
    transformer = Transformer.from_crs(region.crs.to_epsg(), 3857)

    # axis order is different for geographic and projected CRS
    if region.crs.is_geographic:
        minx, miny = map(
            max, zip(transformer.transform(miny, minx), [-20037508.34] * 2)
        )
        maxx, maxy = map(min, zip(transformer.transform(maxy, maxx), [20037508.34] * 2))
    else:
        minx, miny = map(
            max, zip(transformer.transform(minx, miny), [-20037508.34] * 2)
        )
        maxx, maxy = map(min, zip(transformer.transform(maxx, maxy), [20037508.34] * 2))

    for izoom in range(zoom_range[0], zoom_range[1] + 1):
        
        print("Processing zoom level " + str(izoom))
    
        zoom_path = os.path.join(topobathy_path, str(izoom))

        for transform, col, row in tile_window(izoom, minx, miny, maxx, maxy):
            # transform is a rasterio Affine object
            # col, row are the tile indices
            file_name = os.path.join(zoom_path, str(col), str(row) + "." + extension)

            if index_path:
                # Only make tiles for which there is an index file (can be .dat or .png)
                index_file_name_dat = os.path.join(index_path, str(izoom), str(col), str(row) + ".dat")
                index_file_name_png = os.path.join(index_path, str(izoom), str(col), str(row) + ".png")
                if not os.path.exists(index_file_name_dat) and not os.path.exists(index_file_name_png):
                    continue

            x = np.arange(0, npix) + 0.5
            y = np.arange(0, npix) + 0.5
            x3857, y3857 = transform * (x, y)
            zg    = np.float32(np.full([npix, npix], np.nan))

            da_dep = xr.DataArray(
                    zg,
                    coords={'y': y3857, 'x': x3857},
                    dims=["y", "x"],
                    )
            da_dep.raster.set_crs(3857)

            # get subgrid bathymetry tile
            da_dep = merge_multi_dataarrays(
                da_list=da_dep_lst,
                da_like=da_dep,
            )

            if np.isnan(da_dep.values).all():
                # only nans in this tile
                continue

            if np.nanmax(da_dep.values) < z_range[0] or np.nanmin(da_dep.values) > z_range[1]:
                # all values in tile outside z_range
                continue

            if not os.path.exists(os.path.join(zoom_path, str(col))):
                os.makedirs(os.path.join(zoom_path, str(col)))

            if fmt == "bin":
                # And write indices to file
                fid = open(file_name, "wb")
                fid.write(da_dep.values)
                fid.close()
            elif fmt == "png":
                elevation2png(da_dep, file_name)
            elif fmt == "tif":
                da_dep.raster.to_raster(file_name)

# def make_topobathy_tiles_old(
#         root:Union[str, Path],
#         region:gpd.GeoDataFrame,
#         da_dep_lst:List[dict],
#         index_path: Union[str, Path] = None,
#         zoom_range: Union[int, List[int]] = [0, 13],
#         z_range: List[int] = [-20000.0, 20000.0],
#         fmt = "bin",
#     ):
#     """_summary_

#     Parameters
#     ----------
#     root : Union[str, Path]
#         _description_
#     region : gpd.GeoDataFrame
#         _description_
#     da_dep_lst : List[dict]
#         _description_
#     index_path : Union[str, Path], optional
#         _description_, by default None
#     zoom_range : Union[int, List[int]], optional
#         _description_, by default [0, 13]
#     z_range : List[int], optional
#         _description_, by default [-20000.0, 20000.0]
#     format : str, optional
#         _description_, by default "bin"
#     """    

#     assert len(da_dep_lst) > 0, "No DEMs provided"

#     # for binary format, use .dat extension
#     if fmt == "bin":
#         extension = "dat"
#     # for net, tif and png extension and format are the same    
#     else:
#         extension = fmt

#     topobathy_path = os.path.join(root, "topobathy")
#     npix = 256

#     # if only one zoom level is specified, create tiles up to that zoom level (inclusive)
#     if isinstance(zoom_range, int):
#         zoom_range = [0, zoom_range]
    
#     # get bounding box of region in EPSG:4326
#     minx, miny, maxx, maxy = region.total_bounds
#     transformer = Transformer.from_crs(region.crs.to_epsg(), 4326)
#     lat_range, lon_range = transformer.transform([minx, maxx], [miny, maxy])

#     for izoom in range(zoom_range[0], zoom_range[1] + 1):
        
#         print("Processing zoom level " + str(izoom))
    
#         zoom_path = os.path.join(topobathy_path, str(izoom))
    
#         dxy = (40075016.686/npix) / 2 ** izoom
#         xx = np.linspace(0.0, (npix - 1)*dxy, num=npix)
#         yy = xx[:]
#         xv, yv = np.meshgrid(xx, yy)
    
#         ix0, iy0 = deg2num(lat_range[0], lon_range[0], izoom)
#         ix1, iy1 = deg2num(lat_range[1], lon_range[1], izoom)

#         transformer_4326_to_3857 = Transformer.from_crs(
#             CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
#         )         
    
#         for i in range(ix0, ix1 + 1):
        
#             path_okay   = False
#             zoom_path_i = os.path.join(zoom_path, str(i))
        
#             for j in range(iy0, iy1 + 1):
                        
#                 file_name = os.path.join(zoom_path_i, str(j) + "." + extension)
                
#                 if index_path:
#                     # Only make tiles for which there is an index file (can be .dat or .png)
#                     index_file_name_dat = os.path.join(index_path, str(izoom), str(i), str(j) + ".dat")
#                     index_file_name_png = os.path.join(index_path, str(izoom), str(i), str(j) + ".png")
#                     if not os.path.exists(index_file_name_dat) and not os.path.exists(index_file_name_png):
#                         continue
        
#                 # Compute lat/lon at ll corner of tile
#                 lat, lon = num2deg(i, j, izoom)                
        
#                 # Convert origin to Global Mercator
#                 xo, yo = transformer_4326_to_3857.transform(lon,lat)
        
#                 # Tile grid on Global mercator
#                 x3857 = xv[:] + xo + 0.5*dxy
#                 y3857 = yv[:] + yo + 0.5*dxy
#                 zg    = np.float32(np.full([npix, npix], np.nan))

#                 # convert into xarray
#                 da_dep = xr.DataArray(
#                     zg,
#                     coords = {
#                         "yc": (("y","x"), y3857),
#                         "xc": (("y","x"), x3857),
#                     },
#                     dims=["y", "x"],
#                 )
#                 da_dep.raster.set_crs(3857)

#                 # get subgrid bathymetry tile
#                 da_dep = merge_multi_dataarrays(
#                     da_list=da_dep_lst,
#                     da_like=da_dep,
#                 )

#                 if np.isnan(da_dep.values).all():
#                     # only nans in this tile
#                     continue
                    
#                 if np.nanmax(da_dep.values)<z_range[0] or np.nanmin(da_dep.values)>z_range[1]:
#                     # all values in tile outside z_range
#                     continue
                                    
#                 if not path_okay:
#                     if not os.path.exists(zoom_path_i):
#                         os.makedirs(zoom_path_i)
#                         path_okay = True

#                 if fmt == "bin":     
#                     # And write indices to file
#                     fid = open(file_name, "wb")
#                     fid.write(da_dep.values)
#                     fid.close()
#                 elif fmt == "png":
#                     elevation2png(da_dep, file_name)
#                 elif fmt == "tif":
#                     da_dep.raster.to_raster(file_name)