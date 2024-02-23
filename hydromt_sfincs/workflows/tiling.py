"""Tiling functions for fast visualization of the SFINCS model in- and output data."""
import logging
import math
import os
from itertools import product
from pathlib import Path
from typing import List, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from PIL import Image
from pyproj import Transformer

from .merge import merge_multi_dataarrays

__all__ = ["create_topobathy_tiles", "downscale_floodmap_webmercator"]

logger = logging.getLogger(__name__)


def downscale_floodmap_webmercator(
    zsmax: Union[np.array, xr.DataArray],
    index_path: str,
    topobathy_path: str,
    floodmap_path: str,
    hmin: float = 0.05,
    zoom_range: Union[int, List[int]] = [0, 13],
    fmt_in: str = "bin",
    fmt_out: str = "png",
    merge: bool = True,  # FIXME: this is not implemented yet
):
    """Create a downscaled floodmap for (model) region in webmercator tile format

    Parameters
    ----------
    zsmax : Union[np.array, xr.DataArray]
        DataArray with maximum water level (m) for each cell
    index_path : str
        Directory with index files
    topobathy_path : str
        Directory with topobathy files
    floodmap_path : str
        Directory where floodmap files will be stored
    hmin : float, optional
        Minimum water depth considered as "flooded", by default 0.05 m
    zoom_range : Union[int, List[int]], optional
        Range of zoom levels, by default [0, 13]
    fmt_in : str, optional
        Format of the index and topobathy tiles, by default "bin"
    fmt_out : str, optional
        Format of the floodmap tiles to be created, by default "png"
    merge : bool, optional
        Merge floodmap tiles with existing floodmap tiles
        (this could for example happen when there is overlap between models),
        by default True
    """

    # if zsmax is an xarray, convert to numpy array
    if isinstance(zsmax, xr.DataArray):
        zsmax = zsmax.values
    zsmax = zsmax.flatten()

    # if only one zoom level is specified, create tiles up to that zoom level (inclusive)
    if isinstance(zoom_range, int):
        zoom_range = [0, zoom_range]

    for izoom in range(zoom_range[0], zoom_range[1] + 1):
        index_zoom_path = os.path.join(index_path, str(izoom))

        if not os.path.exists(index_zoom_path):
            continue

        # list the available x-folders
        x_folders = [f.path for f in os.scandir(index_zoom_path) if f.is_dir()]

        # loop over x-folders
        for x_folder in x_folders:
            x = os.path.basename(x_folder)
            # list the available y-files with fmt_in extension
            y_files = []
            # Iterate directory
            for file in os.listdir(x_folder):
                # check only text files
                if file.endswith(fmt_in):
                    y_files.append(file)

            # loop over y-files
            for y_file in y_files:
                # read the index file
                index_fn = os.path.join(x_folder, y_file)
                if fmt_in == "bin":
                    ind = np.fromfile(index_fn, dtype="i4")
                elif fmt_in == "png":
                    ind = png2int(index_fn)

                # read the topobathy file
                dep_fn = os.path.join(topobathy_path, str(izoom), x, y_file)
                if fmt_in == "bin":
                    dep = np.fromfile(dep_fn, dtype="f4")
                elif fmt_in == "png":
                    dep = png2elevation(dep_fn)

                # create the floodmap
                hmax = zsmax[ind]
                hmax = hmax - dep
                hmax[hmax < hmin] = np.nan
                hmax = hmax.reshape(256, 256)

                # save the floodmap
                if np.isnan(hmax).all():
                    # only nans in this tile
                    continue

                if not os.path.exists(os.path.join(floodmap_path, str(izoom), x)):
                    os.makedirs(os.path.join(floodmap_path, str(izoom), x))

                floodmap_fn = os.path.join(
                    floodmap_path, str(izoom), x, y_file.replace(fmt_in, fmt_out)
                )
                if fmt_out == "bin":
                    # And write indices to file
                    fid = open(floodmap_fn, "wb")
                    fid.write(hmax)
                    fid.close()
                elif fmt_out == "png":
                    elevation2png(hmax, floodmap_fn)


def create_topobathy_tiles(
    root: Union[str, Path],
    region: gpd.GeoDataFrame,
    datasets_dep: List[dict],
    index_path: Union[str, Path] = None,
    zoom_range: Union[int, List[int]] = [0, 13],
    z_range: List[int] = [-20000.0, 20000.0],
    fmt="bin",
    logger=logger,
):
    """Create webmercator topobathy tiles for a given region.

    Parameters
    ----------
    root : Union[str, Path]
        Directory where the topobathy tiles will be stored.
    region : gpd.GeoDataFrame
        GeoDataFrame defining the region for which the tiles will be created.
    datasets_dep : List[dict]
        List of dictionaries containing the bathymetry dataarrays.
    index_path : Union[str, Path], optional
        Directory where index tiles are stored, by default None
    zoom_range : Union[int, List[int]], optional
        Range of zoom levels for which tiles are created, by default [0, 13]
    z_range : List[int], optional
        Range of valid elevations, by default [-20000.0, 20000.0]
    format : str, optional
        The desired output format of the topobathy tiles, by default "bin". Also "png" and "tif" are supported.
    """
    # TODO change the order of the zoom_levels
    # basing large scale zoom levels on the high-resolution ones prevents memory errors

    assert len(datasets_dep) > 0, "No DEMs provided"

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
        logger.debug("Processing zoom level " + str(izoom))

        zoom_path = os.path.join(topobathy_path, str(izoom))

        for transform, col, row in tile_window(izoom, minx, miny, maxx, maxy):
            # transform is a rasterio Affine object
            # col, row are the tile indices
            file_name = os.path.join(zoom_path, str(col), str(row) + "." + extension)

            if index_path:
                # Only make tiles for which there is an index file (can be .dat or .png)
                index_file_name_dat = os.path.join(
                    index_path, str(izoom), str(col), str(row) + ".dat"
                )
                index_file_name_png = os.path.join(
                    index_path, str(izoom), str(col), str(row) + ".png"
                )
                if not os.path.exists(index_file_name_dat) and not os.path.exists(
                    index_file_name_png
                ):
                    continue

            x = np.arange(0, npix) + 0.5
            y = np.arange(0, npix) + 0.5
            x3857, y3857 = transform * (x, y)
            zg = np.float32(np.full([npix, npix], np.nan))

            da_dep = xr.DataArray(
                zg,
                coords={"y": y3857, "x": x3857},
                dims=["y", "x"],
            )
            da_dep.raster.set_crs(3857)

            # get subgrid bathymetry tile
            da_dep = merge_multi_dataarrays(
                da_list=datasets_dep,
                da_like=da_dep,
            )

            if np.isnan(da_dep.values).all():
                # only nans in this tile
                continue

            if (
                np.nanmax(da_dep.values) < z_range[0]
                or np.nanmin(da_dep.values) > z_range[1]
            ):
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


def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to webmercator tile number"""
    lat_rad = math.radians(lat_deg)
    n = 2**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(-lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """Convert webmercator tile number to lat/lon"""
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(-lat_rad)
    return (lat_deg, lon_deg)


def rgba2int(rgba):
    """Convert rgba tuple to int"""
    r, g, b, a = rgba
    return (r * 256**3) + (g * 256**2) + (b * 256) + a


def int2rgba(int_val):
    """Convert int to rgba tuple"""
    r = (int_val // 256**3) % 256
    g = (int_val // 256**2) % 256
    b = (int_val // 256) % 256
    a = int_val % 256
    return (r, g, b, a)


def elevation2rgb(val):
    """Convert elevation to rgb tuple"""
    val += 32768
    r = np.floor(val / 256)
    g = np.floor(val % 256)
    b = np.floor((val - np.floor(val)) * 256)

    return (r, g, b)


def rgb2elevation(r, g, b):
    """Convert rgb tuple to elevation"""
    val = (r * 256 + g + b / 256) - 32768
    return val


def png2int(png_file):
    """Convert png to int array"""
    # Open the PNG image
    image = Image.open(png_file)

    # Convert the image to RGBA mode if it's not already in RGBN mode
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Get the pixel data from the image
    pixel_data = list(image.getdata())

    # Convert RGBA values to unique integers
    val = []
    for rgba in pixel_data:
        val.append(rgba2int(rgba))

    return val


def int2png(val, png_file):
    """Convert int array to png"""
    # Convert index integers to RGBA values
    rgba = np.zeros((256 * 256, 4), "uint8")
    r, g, b, a = int2rgba(val)

    rgba[:, 0] = r.flatten()
    rgba[:, 1] = g.flatten()
    rgba[:, 2] = b.flatten()
    rgba[:, 3] = a.flatten()

    rgba = rgba.reshape([256, 256, 4])

    # Create PIL Image from RGB values and save as PNG
    img = Image.fromarray(rgba)
    img.save(png_file)


def png2elevation(png_file):
    """Convert png to elevation array based on terrarium interpretation"""
    img = Image.open(png_file)
    arr = np.array(img.convert("RGB"))
    # Convert RGB values to elevation values
    elevations = np.apply_along_axis(rgb2elevation, 2, arr)
    return elevations


def elevation2png(val, png_file):
    """Convert elevation array to png using terrarium interpretation"""

    rgb = np.zeros((256 * 256, 3), "uint8")
    r, g, b = elevation2rgb(val)

    rgb[:, 0] = r.values.flatten()
    rgb[:, 1] = g.values.flatten()
    rgb[:, 2] = b.values.flatten()

    rgb = rgb.reshape([256, 256, 3])

    # Create PIL Image from RGB values and save as PNG
    img = Image.fromarray(rgb)
    img.save(png_file)


def tile_window(zl, minx, miny, maxx, maxy):
    """Window generator for a given zoom level and bounding box"""
    dxy = (20037508.34 * 2) / (2**zl)
    # Origin displacement
    odx = np.floor(abs(-20037508.34 - minx) / dxy)
    ody = np.floor(abs(20037508.34 - maxy) / dxy)

    # Set the new origin
    minx = -20037508.34 + odx * dxy
    maxy = 20037508.34 - ody * dxy

    # Create window generator
    lu = product(np.arange(minx, maxx, dxy), np.arange(maxy, miny, -dxy))
    for l, u in lu:
        col = round(odx + (l - minx) / dxy)
        row = round(ody + (maxy - u) / dxy)
        yield Affine(dxy / 256, 0, l, 0, -dxy / 256, u), col, row
