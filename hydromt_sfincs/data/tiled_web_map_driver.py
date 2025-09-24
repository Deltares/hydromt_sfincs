"""Driver using rasterio for RasterDataset."""

import math
from multiprocessing.pool import ThreadPool
from logging import Logger, getLogger
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import dask.array as da
from dask import delayed
import geopandas as gpd
import numpy as np
from PIL import Image
import toml
import xarray as xr
from pyproj import CRS

from hydromt._typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
    Zoom,
)
from hydromt._typing.error import NoDataStrategy, exec_nodata_strat
from hydromt._utils.temp_env import temp_env
from hydromt._utils.uris import _strip_scheme
from hydromt.config import SETTINGS
from hydromt.data_catalog.drivers import RasterDatasetDriver
from hydromt.gis._raster_utils import _cellres

logger = getLogger(f"hydromt.{__name__}")


class ZoomLevel:
    def __init__(self):
        self.ntiles = 0
        self.ij_available = None

class TiledWebMapDriver(RasterDatasetDriver):
    """Driver using rasterio for RasterDataset."""

    name = "tiled_web_map"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # defaults
        defaults = dict(
            uri=None,
            parameter="elevation",
            encoder="terrarium",
            encoder_vmin=None,
            encoder_vmax=None,
            npix=256,
            initialized=False,
            download=True,
            availability_exists=False,
            availability_loaded=False,
            max_zoom=0,
            s3_client=None,
            s3_bucket=None,
            s3_key=None,
            s3_region=None,
            source="unknown",
            vertical_reference_level="MSL",
            vertical_units="m",
            difference_with_msl=0.0,
        )

        # merge defaults with user-provided kwargs
        self.options.update({**defaults, **kwargs})

    @property
    def uri(self) -> str | None:
        return self.options.get("uri")

    @uri.setter
    def uri(self, value: str | None):
        self.options["uri"] = value

    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        zoom: Optional[Zoom] = None,
        chunks: Optional[dict] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> xr.Dataset:
        """Read Tiled Web Map data to an xarray DataSet."""

        # if variables is None:
        #     variables = "elevation"
        # if variables not in ["elevation", "floodmap", "index", "data", "rgb"]:
        #     raise ValueError(
        #         "Parameter must be one of the following: elevation, floodmap, index, data, rgb"
        #     )

        # initialize the dataset options if not yet done
        if not self.options.get("initialized"):
            self.uri = uris[0]
            self._read_metadata()
            # Check if available_tiles.nc exists. If not, just read the folders to get the zoom range.
            nc_file = os.path.join(self.uri, "available_tiles.nc")
            self.options["availability_loaded"] = False
            if os.path.exists(nc_file):
                self.options["availability_exists"] = True
            else:
                self.options["availability_exists"] = False
            # If availability info exists but is not yet loaded, load it now
            if self.options["availability_exists"] and not self.options["availability_loaded"]:
                self._read_availability()
                self.options["availability_loaded"] = True
            # Check if s3 parameters are provided, if so, set download to True
            if (
                self.options.get("s3_bucket") is not None
                and self.options.get("s3_key") is not None
                and self.options.get("s3_region") is not None
            ):
                self.options["download"] = True
            else:
                self.options["download"] = False
            # now set the initialized flag to True
            self.options["initialized"] = True

        # Determine zoom level, highest resolution is used if not specified
        izoom = get_zoom_level_for_resolution(zoom=zoom,mask=mask,max_zoom=self.options.get("max_zoom"))

        # Determine tile indices to read, if mask is provided, use that to limit the area
        if mask is not None:
            # Get bounds from mask
            bbox = mask.to_crs("EPSG:3857").total_bounds  # minx, miny, maxx, maxy
            # Determine the indices of required tiles
            ix0, iy0 = xy2num(bbox[0], bbox[3], izoom)
            ix1, iy1 = xy2num(bbox[2], bbox[1], izoom)
            # Make sure indices are within bounds
            ix0, iy0 = max(0, ix0), max(0, iy0)
            iy1 = min(2**izoom - 1, iy1)
        elif self.options.get("availability_loaded", False):
            # Use all available tiles at this zoom level
            zoom_level = self.options["zoom_levels"][izoom]
            ix0 = min(ij // zoom_level.ntiles for ij in zoom_level.ij_available)
            ix1 = max(ij // zoom_level.ntiles for ij in zoom_level.ij_available)
            iy0 = min(ij % zoom_level.ntiles for ij in zoom_level.ij_available)
            iy1 = max(ij % zoom_level.ntiles for ij in zoom_level.ij_available)
        else:
            raise ValueError("Either bbox/geom or availability information must be provided to read Tiled Web Map data.")
        
        # Download missing tiles if required
        if self.options["download"]:
            self._download_missing_tiles(ix0, ix1, iy0, iy1, izoom)#, waitbox=waitbox)

        # Get dict of available tiles
        tile_dict = self._get_tile_paths(ix0, ix1, iy0, iy1, izoom)
        xs = sorted(set(i for i, _ in tile_dict.keys()))
        ys = sorted(set(j for _, j in tile_dict.keys()))

        # Create dask array from tiles, without loading all tiles into memory
        delayed_tiles = [
            [delayed(png2elevation)(tile_dict[(x, y)]) for x in xs if (x, y) in tile_dict]
            for y in ys
        ]
        sample_tile = np.array(png2elevation(next(iter(tile_dict.values()))))
        dask_tiles = da.block([
            [da.from_delayed(t, shape=sample_tile.shape, dtype=sample_tile.dtype) for t in row]
            for row in delayed_tiles
        ])

        # Compute x and y coordinates
        x0, y0 = num2xy(ix0, iy1 + 1, izoom)
        x1, y1 = num2xy(ix1 + 1, iy0, izoom)
        nx, ny = dask_tiles.shape[1], dask_tiles.shape[0]
        # Data is stored in centres of pixels so we need to shift the coordinates
        dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
        x = np.linspace(x0 + 0.5*dx, x1 - 0.5*dx, nx)
        y = np.linspace(y0 + 0.5*dy, y1 - 0.5*dy, ny)

        data = np.flipud(dask_tiles)

        # Create xarray Dataset
        # TODO make name of variable configurable?
        elevation = xr.Dataset(
            {
                "elevtn": (("y", "x"), data)  # variable name + dims + data
            },
            coords={"x": x, "y": y},
            attrs={"crs": "EPSG:3857", "z_level": izoom},
        )

        # Optionally rechunk the data for efficient processing later on
        if chunks is not None:
            elevation = elevation.chunk(chunks=chunks)

        return elevation


    def write(self, path: StrPath, ds: xr.Dataset, **kwargs) -> str:
        """Write out a RasterDataset using rasterio."""
        raise NotImplementedError()

    def _read_metadata(self):
        # Read metadata file
        tml_file = os.path.join(self.uri, "metadata.tml")
        if os.path.exists(tml_file):
            tml = toml.load(tml_file)
            # update self.options with tml values
            self.options.update(tml)
            # for key in tml:
            #     setattr(self, key, tml[key])

    def _read_availability(self):
        # Read netcdf file with dimensions
        nc_file = os.path.join(self.uri, "available_tiles.nc")
        nc_file = os.path.normpath(nc_file)

        with xr.open_dataset(nc_file) as ds:
            self.options["zoom_levels"] = []
            # Loop through zoom levels
            for izoom in range(self.options.get("max_zoom", 0) + 1):
                n = 2**izoom
                iname = f"i_available_{izoom}"
                jname = f"j_available_{izoom}"
                iav = ds[iname].to_numpy()[:]
                jav = ds[jname].to_numpy()[:]
                zoom_level = ZoomLevel()
                zoom_level.ntiles = n
                zoom_level.ij_available = iav * n + jav
                self.options["zoom_levels"].append(zoom_level)

    def _download_missing_tiles(self, ix0, ix1, iy0, iy1, izoom, waitbox=None):
        """Ensure all required tiles for given range exist locally."""
        download_file_list = []
        download_key_list = []

        for i in range(ix0, ix1 + 1):
            itile = np.mod(i, 2**izoom)  # wrap around
            ifolder = str(itile)
            for j in range(iy0, iy1 + 1):
                png_file = os.path.join(self.uri, str(izoom), ifolder, str(j) + ".png")
                if not os.path.exists(png_file):
                    # Check availability matrix if present
                    if self.options["availability_exists"] and not self._check_availability(i, j, izoom):
                        continue
                    download_file_list.append(png_file)
                    download_key_list.append(f"{self.options.get('s3_key')}/{izoom}/{ifolder}/{j}.png")
                    Path(png_file).parent.mkdir(parents=True, exist_ok=True)

        if len(download_file_list) > 0:
            if waitbox is not None:
                wb = waitbox("Downloading topography tiles ...")
            if self.options["s3_client"] is None:
                self.options["s3_client"] = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            with ThreadPool() as pool:
                pool.starmap(
                    self._download_tile_parallel,
                    [(self.options.get('s3_bucket'), key, file) for key, file in zip(download_key_list, download_file_list)],
                )
            if waitbox is not None:
                wb.close()

    def _get_tile_paths(self, ix0, ix1, iy0, iy1, izoom):
        """Return dict of {(ix, iy): local_path} for available tiles."""
        tile_dict = {}
        for i in range(ix0, ix1 + 1):
            itile = np.mod(i, 2**izoom)
            ifolder = str(itile)
            for j in range(iy0, iy1 + 1):
                png_file = os.path.join(self.uri, str(izoom), ifolder, f"{j}.png")
                if os.path.exists(png_file):
                    tile_dict[(i, j)] = png_file
        return tile_dict

    def _check_availability(self, i, j, izoom):
        # Check if tile exists at all
        zoom_level = self.options["zoom_levels"][izoom]
        ij = i * zoom_level.ntiles + j
        # Use numpy array for fast search
        available = np.isin(ij, zoom_level.ij_available)
        return available

    def _download_tile(self, i, j, izoom):
        key = f"{self.options.get('s3_key')}/{izoom}/{i}/{j}.png"
        filename = os.path.join(self.uri, str(izoom), str(i), str(j) + ".png")
        try:
            self.options["s3_client"].download_file(
                Bucket=self.options.get('s3_bucket'),  # assign bucket name
                Key=key,  # key is the file name
                Filename=filename,
            )  # storage file path
            logger.info(f"Downloaded {key}")
            okay = True
        except Exception:
            # Download failed
            logger.error(f"Failed to download {key}")
            okay = False
        return okay

    def _download_tile_parallel(self, bucket, key, file):
        try:
            # Make sure the folder exists
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

            self.options["s3_client"].download_file(
                Bucket=bucket,  # assign bucket name
                Key=key,  # key is the file name
                Filename=file,
            )  # storage file path
            logger.info(f"Downloaded {key}")
            okay = True

        except Exception as e:
            # Download failed
            logger.error(f"Failed to download {key}: {e}")
            okay = False

        return okay

def get_zoom_level_for_resolution(zoom, mask:gpd.GeoDataFrame=None, max_zoom:int=23):
    # Get required zoom level
    # Make a dict of zoom levels and resolutions
    zls_dict = {i: 156543.03 / 2**i for i in range(max_zoom + 1)}

    if zoom is None:
        overview_level = max_zoom
        logger.debug(f"No zoom level specified. Using highest resolution zoom level {overview_level}")
        return overview_level
    if isinstance(zoom, int):
        overview_level = zoom
        if overview_level not in zls_dict:
            raise ValueError(
                f"Overview level {overview_level} not defined.Select from {zls_dict}."
            )
        dst_res = zls_dict[overview_level]
    elif (
        isinstance(zoom, tuple)
        and isinstance(zoom[0], (int, float))
        and isinstance(zoom[1], str)
        and len(zoom) == 2
    ):
        src_res, src_res_unit = zoom
        # convert res if different unit than crs
        source_crs = CRS.from_user_input(3857)  # WebMercator
        dst_crs_unit = source_crs.axis_info[0].unit_name
        dst_res = src_res
        if dst_crs_unit != src_res_unit:
            known_units = ["degree", "metre", "US survey foot", "meter", "foot"]
            if src_res_unit not in known_units:
                raise TypeError(
                    f"zoom_level unit {src_res_unit} not understood;"
                    f" should be one of {known_units}"
                )
            if dst_crs_unit not in known_units:
                raise NotImplementedError(
                    f"no conversion available for {src_res_unit} to {dst_crs_unit}"
                )
            conversions = {
                "foot": 0.3048,
                "metre": 1,  # official pyproj units
                "US survey foot": 0.3048,  # official pyproj units
            }  # to meter
            if src_res_unit == "degree" or dst_crs_unit == "degree":
                lat = 0
                if mask is not None:
                    lat = mask.to_crs(4326).centroid.y.item()
                conversions["degree"] = _cellres(lat=lat)[1]
            fsrc = conversions.get(src_res_unit, 1)
            fdst = conversions.get(dst_crs_unit, 1)
            dst_res = src_res * fsrc / fdst
        # find nearest zoom level
        res = list(zls_dict.values())
        zls = list(zls_dict.keys())
        overview_level = np.where(res < dst_res)[0]
        if len(overview_level) == 0:
            overview_level = zls[-1]
        else:
            overview_level = int(overview_level[0])
    else:
        raise TypeError(f"zoom_level not understood: {zoom}")
    logger.debug(f"Using overview level {overview_level} ({dst_res:.2f})")
    return overview_level

def xy2num(easting, northing, zoom):
    lat, lon = webmercator_to_lat_lon(easting, northing)
    ix, it = lat_lon_to_tile_indices(lat, lon, zoom)
    return ix, it

def num2xy(xtile, ytile, zoom):
    """Returns upper left x and y of slippy tile"""
    # Return upper left corner of tile
    n = 2**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    x, y = lat_lon_to_webmercator(lat_deg, lon_deg)
    return x, y


def webmercator_to_lat_lon(easting, northing):
    lon = (easting / 20037508.34) * 180
    lat = (180 / math.pi) * (
        2 * math.atan(math.exp(northing / 20037508.34 * math.pi)) - (math.pi / 2)
    )
    return lat, lon

def lat_lon_to_webmercator(lat, lon):
    # Convert latitude and longitude to Web Mercator coordinates
    x = lon * 20037508.34 / 180
    y = (math.log(math.tan((90 + lat) * math.pi / 360)) / math.pi) * 20037508.34
    return x, y

def lat_lon_to_tile_indices(lat, lon, zoom):
    tile_x = int((lon + 180) / 360 * (2**zoom))
    tile_y = int(
        (
            1
            - (
                math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
                / math.pi
            )
        )
        / 2
        * (2**zoom)
    )
    return tile_x, tile_y


def png2elevation(png_file, encoder="terrarium", encoder_vmin=0.0, encoder_vmax=1.0):
    """Convert png to elevation array based on terrarium interpretation"""
    img = Image.open(png_file)
    # Convert RGB values to elevation values
    if encoder == "terrarium":
        rgb = np.array(img.convert("RGB")).astype(float)
        elevation = (rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256) - 32768.0
        # where val is less than -32767, set to NaN
        elevation[np.where(elevation < -32767.0)] = np.nan
    elif encoder == "terrarium16":
        rgb = np.array(img.convert("RGB")).astype(float)
        elevation = (rgb[:, :, 0] * 256 + rgb[:, :, 1]) - 32768.0
        # where val is less than -32767, set to NaN
        elevation[np.where(elevation < -32767.0)] = np.nan
    elif encoder == "uint8":
        rgb = np.array(img.convert("RGB")).astype(int)
        elevation = rgb[:, :, 0]
        elevation[np.where(elevation == 255)] = -1
    elif encoder == "uint16":
        rgb = np.array(img.convert("RGB")).astype(int)
        elevation = rgb[:, :, 0] * 256 + rgb[:, :, 1]
        elevation[np.where(elevation == 65535)] = -1
    elif encoder == "uint24":
        rgb = np.array(img.convert("RGB")).astype(int)
        elevation = rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]
        elevation[np.where(elevation == 16777215)] = -1
    elif encoder == "uint32":
        rgb = np.array(img.convert("RGBA")).astype(int)
        elevation = (
            rgb[:, :, 0] * 16777216
            + rgb[:, :, 1] * 65536
            + rgb[:, :, 2] * 256
            + rgb[:, :, 3]
        )
        elevation[np.where(elevation == 4294967295)] = -1
    elif encoder == "float8":
        rgb = np.array(img.convert("RGB")).astype(float)
        i = rgb[:, :, 0]
        elevation = encoder_vmin + (encoder_vmax - encoder_vmin) * i / 254
        elevation[np.where(i == 0)] = np.nan
    elif encoder == "float16":
        rgb = np.array(img.convert("RGB")).astype(float)
        i = rgb[:, :, 0] * 256 + rgb[:, :, 1]
        elevation = encoder_vmin + (encoder_vmax - encoder_vmin) * i / 65534
        elevation[np.where(i == 0)] = np.nan
    elif encoder == "float24":
        rgb = np.array(img.convert("RGB")).astype(float)
        i = rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]
        elevation = encoder_vmin + (encoder_vmax - encoder_vmin) * i / 16777214
        elevation[np.where(i == 0)] = np.nan
    elif encoder == "float32":
        rgb = np.array(img.convert("RGBA")).astype(float)
        i = (
            rgb[:, :, 0] * 16777216
            + rgb[:, :, 1] * 65536
            + rgb[:, :, 2] * 256
            + rgb[:, :, 3]
        )
        elevation = encoder_vmin + (encoder_vmax - encoder_vmin) * i / 4294967294
        elevation[np.where(i == 0)] = np.nan
    return elevation
