"""Tiling functions for fast visualization of the SFINCS model in- and output data."""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from PIL import Image
from pyproj import Transformer

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

from .merge import merge_multi_dataarrays

__all__ = [
    "write_html",
    "downscale_floodmap_webmercator",
    "create_index_tiles",
    "create_topobathy_tiles",
]

logger = logging.getLogger(__name__)


def write_html(
    file_name: Union[str, Path],
    title: str = "hydromt-sfincs tiles",
    legend_title: str = "Legend",
    max_native_zoom: int = 19,
) -> None:
    """Write a standalone Leaflet HTML viewer for a tiled map layer.

    The generated page loads OpenStreetMap and Esri World Imagery base
    layers and overlays tiles from the URL pattern ``{z}/{x}/{y}.png``
    relative to the HTML file. Drop it alongside a ``{z}/{x}/{y}.png``
    tile tree and open it in a browser to preview the tiles.

    Parameters
    ----------
    file_name : str or Path
        Output HTML file path.
    title : str, optional
        Page heading shown above the map, by default
        ``"hydromt-sfincs tiles"``.
    legend_title : str, optional
        Text displayed inside the legend box, by default ``"Legend"``.
    max_native_zoom : int, optional
        Maximum native zoom level of the tile layer, by default ``19``.
    """
    with open(file_name, "w") as f:
        f.write("<!DOCTYPE html>\r\n")
        f.write("<head>\r\n")
        f.write(
            "  <meta http-equiv='content-type' content='text/html; charset=UTF-8' />\r\n"
        )
        f.write("  <script>\r\n")
        f.write("    L_NO_TOUCH = false;\r\n")
        f.write("    L_DISABLE_3D = false;\r\n")
        f.write("  </script>\r\n")
        f.write(
            "  <style>html, body {width: 100%;height: 100%;margin: 0;padding: 0;}</style>\r\n"
        )
        f.write(
            "  <script src='https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.js'></script>\r\n"
        )
        f.write(
            "  <script src='https://code.jquery.com/jquery-1.12.4.min.js'></script>\r\n"
        )
        f.write(
            "  <script src='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js'></script>\r\n"
        )
        f.write(
            "  <script src='https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js'></script>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css'/>\r\n"
        )
        f.write(
            "  <link rel='stylesheet' href='https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css'/>\r\n"
        )
        f.write(
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no' />\r\n"
        )
        f.write("\r\n")
        f.write("  <style>\r\n")
        f.write("      #map { width: 800px; height: 500px; }\r\n")
        f.write(
            "      .info { padding: 6px 8px; font: 17px/19px Arial, Helvetica, sans-serif; background: white; background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; } .info h4 { margin: 0 0 5px; color: #777; }\r\n"
        )
        f.write(
            "      .legend     { text-align: center; line-height: 18px; color: #555; } .legend i     { width: 20px; height: 15px; float: left; margin-right: 8px; opacity: 0.7; border-style: solid; border-width: 1px;}\r\n"
        )
        f.write("  </style>\r\n")
        f.write("\r\n")
        f.write("</head>\r\n")
        f.write("<body>\r\n")
        f.write(f"  <h3> {title}</h3>\r\n")
        f.write("  <div id='map' style='width: 100%; height: 90%;'></div>\r\n")
        f.write("</body>\r\n")
        f.write("<script>\r\n")
        f.write("\r\n")
        f.write("// Base layers\r\n")
        f.write(
            "var tile_layer_osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',\r\n"
        )
        f.write(
            "    {'attribution': 'Data by http://openstreetmap.org href=http://www.openstreetmap.org',\r\n"
        )
        f.write("     'detectRetina': false,\r\n")
        f.write("     'maxZoom': 19,\r\n")
        f.write("     'minZoom': 0,\r\n")
        f.write("     'noWrap': false,\r\n")
        f.write("     'opacity': 1,\r\n")
        f.write("     'maxNativeZoom': 13,\r\n")
        f.write("     'subdomains': 'abc',\r\n")
        f.write("     'tms': false});\r\n")
        f.write(
            "var Esri_WorldImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {\r\n"
        )
        f.write(
            "	attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'\r\n"
        )
        f.write("});\r\n")
        f.write("\r\n")
        f.write("// Data layer\r\n")
        f.write("var tile_layer = L.tileLayer(\r\n")
        f.write("    '{z}/{x}/{y}.png',\r\n")
        f.write("    {'attribution': 'hydromt-sfincs',\r\n")
        f.write("     'detectRetina': false,\r\n")
        f.write("     'opacity': 0.7,\r\n")
        f.write(f"     'maxNativeZoom': {max_native_zoom},\r\n")
        f.write("     'maxZoom': 19,\r\n")
        f.write("     'minZoom': 0,\r\n")
        f.write("     'noWrap': false,\r\n")
        f.write("     'subdomains': 'abc',\r\n")
        f.write("     'zIndex':10,\r\n")
        f.write("     'tms': false}\r\n")
        f.write(");\r\n")
        f.write("\r\n")
        f.write("var legend = L.control({position: 'bottomright'});\r\n")
        f.write("legend.onAdd = function (map) {\r\n")
        f.write("        var div = L.DomUtil.create('div', 'info legend')\r\n")
        f.write(f"        div.innerHTML += '{legend_title}<br>'\r\n")
        f.write("        return div;\r\n")
        f.write("};\r\n")
        f.write("\r\n")
        f.write("// Map\r\n")
        f.write("var map = L.map('map',{\r\n")
        f.write("    center: [0, 0],\r\n")
        f.write("    crs: L.CRS.EPSG3857,\r\n")
        f.write("    zoom: 2,\r\n")
        f.write("    zoomControl: true,\r\n")
        f.write("    preferCanvas: false,\r\n")
        f.write("    layers: [tile_layer_osm, tile_layer]\r\n")
        f.write("    }\r\n")
        f.write(");\r\n")
        f.write("\r\n")
        f.write("legend.addTo(map);\r\n")
        f.write("\r\n")
        f.write("// Layer control\r\n")
        f.write("var baseMaps = {\r\n")
        f.write("    'Open Street Map': tile_layer_osm,\r\n")
        f.write("    'Satellite': Esri_WorldImagery\r\n")
        f.write("};\r\n")
        f.write("\r\n")
        f.write("var overlayMaps = {};\r\n")
        f.write("\r\n")
        f.write("L.control.layers(baseMaps, overlayMaps).addTo(map);\r\n")
        f.write("\r\n")
        f.write("</script>\r\n")


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


def create_index_tiles(
    model: "SfincsModel",
    root: Union[str, Path] = None,
    region: Optional[gpd.GeoDataFrame] = None,
    zoom_range: Union[int, List[int]] = [0, 13],
    fmt: str = "png",
    write_html_viewer: bool = True,
    max_workers: Optional[int] = None,
    logger: logging.Logger = logger,
) -> None:
    """Create webmercator index tiles for an SFINCS grid.

    Index tiles map each pixel of a webmercator XYZ tile to the index of
    the corresponding SFINCS cell (``-999`` where there is no match).

    Tiles are written to:
    ``<root>/indices/<zoom>/<x>/<y>.<ext>``

    Parameters
    ----------
    model : SfincsModel
        SFINCS model containing the grid component.
    root : Union[str, Path]
        Parent directory where index tiles are stored. Defaults to the model root path.
    region : gpd.GeoDataFrame, optional
        Region for which tiles are generated. Defaults to the grid region.
    zoom_range : Union[int, List[int]], optional
        Range of zoom levels. A single integer is interpreted as
        ``[0, zoom_range]``.
    fmt : str, optional
        Output format: ``"png"`` or ``"bin"``.
    write_html_viewer : bool, optional
        Write a Leaflet viewer when using PNG output.
    max_workers : int, optional
        Number of worker threads.
    """

    grid = model.grid_component

    if region is None:
        region = grid.region

    if root is None:
        root = os.path.join(model.root.path, "tiling")

    index_path = os.path.join(root, "indices")
    npix = 256

    extension = "dat" if fmt == "bin" else fmt

    if isinstance(zoom_range, int):
        zoom_range = [0, zoom_range]

    # Region bounds in WebMercator
    minx, miny, maxx, maxy = region.total_bounds

    transformer = Transformer.from_crs(
        region.crs,
        3857,
        always_xy=True,
    )

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

    transformer_inv = Transformer.from_crs(
        3857,
        grid.crs,
        always_xy=True,
    )

    # Eagerly touch the lazy ifirst cache to avoid a race in worker threads.
    _ = grid.get_indices_at_points(np.array([[0.0]]), np.array([[0.0]]))

    def _process_tile(args):
        izoom, transform, col, row = args

        zoom_path = os.path.join(index_path, str(izoom))

        file_name = os.path.join(
            zoom_path,
            str(col),
            f"{row}.{extension}",
        )

        # Pixel centers
        x = np.arange(npix) + 0.5
        y = np.arange(npix) + 0.5

        x3857, y3857 = transform * (x, y)
        x3857, y3857 = np.meshgrid(x3857, y3857)

        # Convert to model CRS
        xx, yy = transformer_inv.transform(
            x3857,
            y3857,
        )

        # Lookup SFINCS cell indices
        indices = grid.get_indices_at_points(xx, yy)

        indices = np.asarray(indices)
        indices[indices < 0] = -999

        # Skip empty tiles
        if np.any(indices >= 0):
            os.makedirs(
                os.path.join(zoom_path, str(col)),
                exist_ok=True,
            )

            if fmt == "bin":
                with open(file_name, "wb") as fid:
                    fid.write(indices.astype(np.int32))

            elif fmt == "png":
                indices_png = indices.copy()
                indices_png[indices_png == -999] = 0
                int2png(indices_png, file_name)

    max_workers = max_workers or os.cpu_count() or 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            logger.debug(f"Processing zoom level {izoom}")
            for transform, col, row in tile_window(izoom, minx, miny, maxx, maxy):
                futures.append(
                    executor.submit(_process_tile, izoom, transform, col, row)
                )

        for future in futures:
            future.result()

    if write_html_viewer and fmt == "png":
        os.makedirs(index_path, exist_ok=True)

        write_html(
            os.path.join(index_path, "index.html"),
            title="Index tiles",
            legend_title="Cell indices",
            max_native_zoom=zoom_range[1],
        )


def create_topobathy_tiles(
    root: Union[str, Path],
    region: gpd.GeoDataFrame,
    elevation_list: Optional[List[dict]] = None,
    data_catalog=None,
    index_path: Optional[Union[str, Path]] = None,
    zoom_range: Union[int, List[int]] = [0, 13],
    z_range: List[int] = [-20000.0, 20000.0],
    fmt: str = "png",
    write_html_viewer: bool = True,
    max_workers: Optional[int] = None,
    logger: logging.Logger = logger,
) -> None:
    """Create webmercator topobathy tiles for a given region.

    Parameters
    ----------
    root : Union[str, Path]
        Directory where the topobathy tiles will be stored.
    region : gpd.GeoDataFrame
        GeoDataFrame defining the region for which the tiles will be created.
    elevation_list : List[dict], optional
        List of dictionaries containing the bathymetry dataarrays. If
        ``None``, falls back to ``data_catalog`` (all sources in the
        catalog are treated as elevation datasets).
    data_catalog : hydromt.DataCatalog, optional
        Fallback data catalog used to build an elevation_list when
        ``elevation_list`` is ``None``. Explicit ``elevation_list``
        entries always win.
    index_path : Union[str, Path], optional
        Directory where index tiles are stored, by default None
    zoom_range : Union[int, List[int]], optional
        Range of zoom levels for which tiles are created, by default [0, 13]
    z_range : List[int], optional
        Range of valid elevations, by default [-20000.0, 20000.0]
    fmt : str, optional
        The desired output format of the topobathy tiles, by default "png". Also "bin" and "tif" are supported.
    write_html_viewer : bool, optional
        If True (default), also write an ``index.html`` Leaflet viewer
        alongside the tiles so they can be previewed in a browser.
    """
    # TODO change the order of the zoom_levels
    # basing large scale zoom levels on the high-resolution ones prevents memory errors

    topobathy_path = os.path.join(root, "topobathy")
    npix = 256

    # if only one zoom level is specified, create tiles up to that zoom level
    if isinstance(zoom_range, int):
        zoom_range = [0, zoom_range]

    # Fall back to the data catalog when no explicit elevation_list was
    # supplied. An explicit elevation_list always wins over the catalog.
    if (
        elevation_list is None or len(elevation_list) == 0
    ) and data_catalog is not None:
        res = 40075016.686 / 256 / 2 ** zoom_range[1]
        bbox = tuple(region.to_crs(4326).total_bounds)
        elevation_list = _resolve_elevation_list_from_catalog(
            data_catalog, res=res, bbox=bbox
        )

    assert elevation_list is not None and len(elevation_list) > 0, "No DEMs provided"

    # for binary format, use .dat extension
    if fmt == "bin":
        extension = "dat"
    # for net, tif and png extension and format are the same
    else:
        extension = fmt

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

    def _process_tile(izoom: int, transform: Affine, col: int, row: int) -> None:
        zoom_path = os.path.join(topobathy_path, str(izoom))
        file_name = os.path.join(zoom_path, str(col), f"{row}.{extension}")

        if index_path:
            # Only make tiles for which there is an index file (.dat or .png)
            idx_dat = os.path.join(index_path, str(izoom), str(col), f"{row}.dat")
            idx_png = os.path.join(index_path, str(izoom), str(col), f"{row}.png")
            if not os.path.exists(idx_dat) and not os.path.exists(idx_png):
                return

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
        # Tell rioxarray which dims are spatial so it can derive the
        # transform from the coordinate arrays (avoids
        # NotGeoreferencedWarning during reprojection).
        da_dep.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

        da_dep = merge_multi_dataarrays(
            da_list=elevation_list,
            da_like=da_dep,
        )

        if np.isnan(da_dep.values).all():
            return

        if (
            np.nanmax(da_dep.values) < z_range[0]
            or np.nanmin(da_dep.values) > z_range[1]
        ):
            return

        os.makedirs(os.path.join(zoom_path, str(col)), exist_ok=True)

        if fmt == "bin":
            with open(file_name, "wb") as fid:
                fid.write(da_dep.values)
        elif fmt == "png":
            elevation2png(da_dep, file_name)
        elif fmt == "tif":
            da_dep.raster.to_raster(file_name)

    max_workers = max_workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            logger.debug(f"Processing zoom level {izoom}")
            for transform, col, row in tile_window(izoom, minx, miny, maxx, maxy):
                futures.append(
                    executor.submit(_process_tile, izoom, transform, col, row)
                )
        for future in futures:
            future.result()

    if write_html_viewer and fmt == "png":
        os.makedirs(topobathy_path, exist_ok=True)
        write_html(
            os.path.join(topobathy_path, "index.html"),
            title="Topobathy tiles",
            legend_title="Topobathy",
            max_native_zoom=zoom_range[1],
        )


## Internal helper functions
def _resolve_elevation_list_from_catalog(
    data_catalog, res: Optional[float] = None, bbox=None
) -> List[dict]:
    """Build an elevation_list by resolving every source in a data catalog.

    Used as a fallback when no explicit ``elevation_list`` is supplied to
    the tiling workflows. Assumes all sources in ``data_catalog`` are
    raster elevation datasets (true in DelftDashboard, where the model's
    catalog is populated only with topobathy sources).
    """
    resolved: List[dict] = []
    for name in list(data_catalog.sources):
        try:
            zoom = (res, "meter") if res is not None else None
            da = data_catalog.get_rasterdataset(name, bbox=bbox, buffer=10, zoom=zoom)
            da.name = "elevation"
        except Exception:
            logger.warning(f"No data in domain for {name}, skipped.")
            continue
        resolved.append({"da": da})
    return resolved


## Helper functions for tile math


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
