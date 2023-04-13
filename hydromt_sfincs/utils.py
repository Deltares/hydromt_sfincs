"""
"""

import os
import numpy as np
from affine import Affine
import pyproj
from pyproj.crs.crs import CRS
import rasterio
import xarray as xr
import pandas as pd
import geopandas as gpd
from datetime import datetime
from configparser import ConfigParser
from shapely.geometry import LineString, Polygon
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
import io
import copy
import logging
import hydromt
from hydromt.io import write_xy
from scipy import ndimage
from pyflwdir.regions import region_area
from hydromt_sfincs.workflows import tiling

__all__ = [
    "read_inp",
    "write_inp",
    "rotated_grid",
    "read_binary_map",
    "write_binary_map",
    "read_binary_map_index",
    "write_binary_map_index",
    "read_ascii_map",
    "write_ascii_map",
    "read_timeseries",
    "write_timeseries",
    "get_bounds_vector",
    "mask2gdf",
    "read_xy",
    "write_xy",
    "read_xyn",
    "write_xyn",
    "read_geoms",
    "write_geoms",
    "gdf2linestring",
    "gdf2polygon",
    "linestring2gdf",
    "polygon2gdf",
    "read_sfincs_map_results",
    "read_sfincs_his_results",
    "downscale_floodmap",
    "downscale_floodmap_webmercator",
]

logger = logging.getLogger(__name__)


## CONFIG: sfincs.inp ##


class ConfigParserSfincs(ConfigParser):
    def __init__(self, **kwargs):
        defaults = dict(
            comment_prefixes=("!", "/", "#"),
            inline_comment_prefixes=("!"),
            allow_no_value=True,
            delimiters=("="),
        )
        defaults.update(**kwargs)
        super(ConfigParserSfincs, self).__init__(**defaults)

    def read_file(self, f, **kwargs):
        def add_header(f, header_name="dummy"):
            """add header"""
            yield "[{}]\n".format(header_name)
            for line in f:
                yield line

        super(ConfigParserSfincs, self).read_file(add_header(f), **kwargs)

    def _write_section(self, fp, section_name, section_items, delimiter):
        """Write a single section to the specified `fp'."""
        for key, value in section_items:
            value = self._interpolation.before_write(self, section_name, key, value)
            fp.write("{:<15} {:<1} {:<}\n".format(key, self._delimiters[0], value))
        fp.write("\n")


def read_inp(fn: Union[str, Path]) -> Dict:
    """Read sfincs.inp file and parse to dictionary."""
    return hydromt.config.configread(
        fn, abs_path=False, cf=ConfigParserSfincs, noheader=True
    )


def write_inp(fn: Union[str, Path], conf: Dict) -> None:
    """Write sfincs.inp file from dictionary."""
    return hydromt.config.configwrite(fn, conf, cf=ConfigParserSfincs, noheader=True)


def get_spatial_attrs(config: Dict, crs: Union[int, CRS] = None, logger=logger):
    """Returns geospatial attributes shape, crs and transform from config dict.

    The config dict should contain the following keys:

    * for shape: mmax, nmax
    * for crs: epsg
    * for transform: dx, dy, x0, y0, rotation

    Parameters
    ----------
    config: Dict
        sfincs.inp configuration
    crs: int, CRS
        Coordinate reference system

    Returns
    -------
    shape: tuple of int
        width, height
    transform: Affine.transform
        Geospatial transform
    crs: pyproj.CRS, None
        Coordinate reference system
    """
    # retrieve rows and cols
    cols = config.get("mmax")
    rows = config.get("nmax")
    if cols is None or rows is None:
        raise ValueError('"mmax" or "nmax" not defined in sfincs.inp')

    # retrieve CRS
    if crs is None and "epsg" in config:
        crs = pyproj.CRS.from_epsg(int(config.get("epsg")))
    elif crs is not None:
        crs = pyproj.CRS.from_user_input(crs)
    else:
        logger.warning('"epsg" code not defined in sfincs.inp, unknown CRS.')

    # retrieve spatial transform
    dx = config.get("dx")
    dy = config.get("dy")
    west = config.get("x0")
    south = config.get("y0")
    rotation = config.get("rotation", 0)  # clockwise rotation [degrees]
    if west is None or south is None:
        logger.warning(
            'Either one of "x0" or "y0" not defined in sfincs.inp, '
            "falling back to origin at (0, 0)."
        )
        west, south = 0, 0
    if dx is None or dy is None:
        logger.warning(
            'Either one of "dx" or "dy" not defined in sfincs.inp, '
            "falling back unity resolution (1, 1)."
        )
        dx, dy = 1, 1

    transform = (
        Affine.translation(west, south)
        * Affine.rotation(rotation)
        * Affine.scale(dx, dy)
    )

    return (rows, cols), transform, crs

def rotated_grid(pol: Polygon, res: float) -> Tuple[float, float, int, int, float]:
    """Returns the origin (x0, y0), shape (mmax, nmax) and rotation
    of the rotated grid fitted to the minimum rotated rectangle around the
    area of interest (pol). The grid shape is defined by the resolution (res).

    Parameters
    ----------
    pol : Polygon
        Polygon of the area of interest
    res : float
        Resolution of the grid
    """

    def _azimuth(point1, point2):
        """azimuth between 2 points (interval 0 - 180)"""
        angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        return np.degrees(angle)

    def _dist(a, b):
        """distance between points"""
        return np.hypot(b[0] - a[0], b[1] - a[1])

    mrr = pol.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)[:-1, :]  # get coordinates of all corners
    # get origin based on the corner with the smallest distance to origin
    # after translation to account for possible negative coordinates
    ib = np.argmin(
        np.hypot(coords[:, 0] - coords[:, 0].min(), coords[:, 1] - coords[:, 1].min())
    )
    ir = (ib + 1) % 4
    il = (ib + 3) % 4
    x0, y0 = coords[ib, :]
    az1 = _azimuth((x0, y0), coords[ir, :])
    az2 = _azimuth((x0, y0), coords[il, :])
    axis1 = _dist((x0, y0), coords[ir, :])
    axis2 = _dist((x0, y0), coords[il, :])
    if az2 < az1:
        rot = az2
        mmax = int(np.ceil(axis2 / res))
        nmax = int(np.ceil(axis1 / res))
    else:
        rot = az1
        mmax = int(np.ceil(axis1 / res))
        nmax = int(np.ceil(axis2 / res))

    return x0, y0, mmax, nmax, rot


## BINARY MAPS: sfincs.ind, sfincs.msk, sfincs.dep etc. ##


def write_binary_map_index(fn_ind: Union[str, Path], msk: np.ndarray) -> None:
    """Write flat index of binary map file.

    NOTE: The array should be in S->N and W->E orientation, with origin in the SW corner.

    Parameters
    ----------
    fn_ind: str, Path
        Path to output map index file.
    msk: np.ndarray
        2D array of sfincs mask map, where invalid cells have value 0.
    """
    # the index number file of sfincs starts with the length of the index numbers
    indices = np.where(msk.transpose().flatten() > 0)[0] + 1  # convert to 1-based index
    indices_ = np.array(np.hstack([np.array(len(indices)), indices]), dtype="u4")
    indices_.tofile(fn_ind)


def read_binary_map_index(fn_ind: Union[str, Path]) -> np.ndarray:
    """Read binary map index file.

    Parameters
    ----------
    fn_ind: str, Path
        Path to map index file.

    Returns
    -------
    ind: np.ndarray
        1D array of flat index of binary maps.
    """
    _ind = np.fromfile(fn_ind, dtype="u4")
    ind = _ind[1:] - 1  # convert to zero based index
    assert _ind[0] == ind.size
    return ind


def write_binary_map(
    fn: Union[str, Path],
    data: np.ndarray,
    msk: np.ndarray,
    dtype: Union[str, np.dtype] = "f4",
) -> None:
    """Write binary map file.

    NOTE: The array should be in S->N and W->E orientation, with origin in the SW corner.

    Parameters
    ----------
    fn str, Path
        Path to output map index file.
    data: np.ndarray
        2D array of sfincs map.
    msk: np.ndarray
        2D array of sfincs mask map, where invalid cells have value 0.
    dtype: str, np.dtype, optional
        Data type, by default "f4". For sfincs.msk file use dtype="u1".
    """
    data_out = np.asarray(data.transpose()[msk.transpose() > 0], dtype=dtype)
    data_out.tofile(fn)


def read_binary_map(
    fn: Union[str, Path],
    ind: np.ndarray,
    shape: Tuple[int],
    mv: float = -9999.0,
    dtype: str = "f4",
) -> np.ndarray:
    """Read binary map.

    Parameters
    ----------
    fn: str, Path
        Path to map file.
    ind: np.ndarray
        1D array of flat index of binary maps.
    shape: tuple of int
        (nrow, ncol) shape of output map.
    mv: int or float
        missing value, by default -9999.0.
    dtype: str, np.dtype, optional
        Data type, by default "f4". For sfincs.msk file use dtype="u1".

    Returns
    -------
    ind: np.ndarray
        1D array of flat index of binary maps.
    """
    assert ind.max() <= np.multiply(*shape)
    nrow, ncol = shape
    data = np.full((ncol, nrow), mv, dtype=dtype)
    data.flat[ind] = np.fromfile(fn, dtype=dtype)
    data = data.transpose()
    return data


## ASCII maps: sfincs.restart ##


def read_ascii_map(fn: Union[str, Path]) -> np.ndarray:
    """Read ascii map

    Parameters
    ----------
    fn : str, Path
        Path to ascii map file.

    Returns
    -------
    data : np.ndarray
        2D array of sfincs map.
    """
    data = np.loadtxt(fn).astype(np.float32)
    return data


def write_ascii_map(fn: Union[str, Path], data: np.ndarray, fmt: str = "%8.3f") -> None:
    """Write ascii map

    NOTE: The array should be in S->N and W->E orientation, with origin in the SW corner.

    Parameters
    ----------
    fn : str, Path
        Path to ascii map file.
    data : np.ndarray
        2D array of sfincs map.
    fmt : str, optional
        Value format, by default "%8.3f". See numpy.savetxt for more options.
    """
    with open(fn, "w") as f:
        np.savetxt(f, data, fmt=fmt)


## XY files: bnd / src ##
# write_xy defined in hydromt.io


def read_xy(fn: Union[str, Path], crs: Union[int, CRS] = None) -> gpd.GeoDataFrame:
    """Read sfincs xy files and parse to GeoDataFrame.

    Parameters
    ----------
    fn : str, Path
        Path to ascii xy file.
    crs: int, CRS
        Coordinate reference system

    Returns
    -------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with point geomtries
    """
    gdf = hydromt.open_vector(fn, crs=crs, driver="xy")
    gdf.index = np.arange(1, gdf.index.size + 1, dtype=int)  # index starts at 1
    return gdf


def read_xyn(fn: str, crs: int = None):
    df = pd.read_csv(fn, index_col=False, header=None, delim_whitespace=True).rename(
        columns={0: "x", 1: "y"}
    )
    if len(df.columns) > 2:
        df = df.rename(columns={2: "name"})
    else:
        df["name"] = df.index

    points = gpd.points_from_xy(df["x"], df["y"])
    gdf = gpd.GeoDataFrame(df.drop(columns=["x", "y"]), geometry=points, crs=crs)

    return gdf


def write_xyn(fn: str = "sfincs.obs", gdf: gpd.GeoDataFrame = None, crs: int = None):
    with open(fn, "w") as fid:
        for point in gdf.iterfeatures():
            x, y = point["geometry"]["coordinates"]
            try:
                name = point["properties"]["name"]
            except:
                name = "obs" + str(point["id"])
            if crs.is_geographic:
                string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
            else:
                string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
            fid.write(string)


## ASCII TIMESERIES: bzs / dis / precip ##


def parse_datetime(dt: Union[str, datetime], format="%Y%m%d %H%M%S") -> datetime:
    """Checks and/or parses datetime from a string, default sfincs datetime string format"""
    if isinstance(dt, str):
        dt = datetime.strptime(dt, format)
    elif not isinstance(dt, datetime):
        raise ValueError(f"Unknown type for datetime: {type(dt)})")
    return dt


def read_timeseries(fn: Union[str, Path], tref: Union[str, datetime]) -> pd.DataFrame:
    """Read ascii timeseries files such as sfincs.bzs, sfincs.dis and sfincs.precip.
    The first column (time index) is parsed to datetime format assumming it represents
    seconds from `tref`.

    Parameters
    ----------
    fn: str, Path
        Path to output timeseries file.
    tref: datetime.datetime, str
        Datetime of tref, string in "%Y%m%d %H%M%S" format.

    Returns
    -------
    df: pd.DataFrame
        Dataframe of timeseries with parsed time index.
    """
    tref = parse_datetime(tref)
    df = pd.read_csv(fn, delim_whitespace=True, index_col=0, header=None)
    df.index = pd.to_datetime(df.index.values, unit="s", origin=tref)
    df.columns = df.columns.values.astype(int)
    df.index.name = "time"
    df.columns.name = "index"
    return df


def write_timeseries(
    fn: Union[str, Path],
    df: Union[pd.DataFrame, pd.Series],
    tref: Union[str, datetime],
    fmt: str = "%7.2f",
) -> None:
    """Write pandas.DataFrame to fixed width ascii timeseries files
    such as sfincs.bzs, sfincs.dis and sfincs.precip. The output time index is given in
    seconds from tref.

    Parameters
    ----------
    fn: str, Path
        Path to output timeseries file.
    df: pd.DataFrame
        Dataframe of timeseries.
    tref: datetime.datetime, str
        Datetime of tref, string in "%Y%m%d %H%M%S" format.
    fmt: str, optional
        Output value format, by default "%7.2f".
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    elif not isinstance(df, pd.DataFrame):
        raise ValueError(f"Unknown type for df: {type(df)})")
    tref = parse_datetime(tref)
    if df.index.size == 0:
        raise ValueError("df does not contain data.")
    data = df.reset_index().values
    data[:, 0] = (df.index - tref).total_seconds()
    # calculate required width for time column; hard coded single decimal precision
    # format for other columns is based on fmt`argument
    w = int(np.floor(np.log10(data[-1, 0]))) + 3
    fmt_lst = [f"%{w}.1f"] + [fmt for _ in range(df.columns.size)]
    fmt_out = " ".join(fmt_lst)
    with open(fn, "w") as f:
        np.savetxt(f, data, fmt=fmt_out)


## MASK


def get_bounds_vector(da_msk: xr.DataArray) -> gpd.GeoDataFrame:
    """Get bounds of vectorized mask as GeoDataFrame.

    Parameters
    ----------
    da_msk: xr.DataArray
        Mask as DataArray with values 0 (inactive), 1 (active),
        and boundary cells 2 (waterlevels) and 3 (outflow).

    Returns
    -------
    gdf_msk: gpd.GeoDataFrame
        GeoDataFrame with line geometries of mask boundaries.
    """
    gdf_msk = da_msk.raster.vectorize()
    gdf_msk["geometry"] = gdf_msk.buffer(1)  # small buffer for rounding errors
    region = (da_msk >= 1).astype("int16").raster.vectorize()
    region = region[region["value"] == 1].drop(columns="value")
    region["geometry"] = region.boundary
    gdf_msk = gdf_msk[gdf_msk["value"] != 1]
    gdf_msk = gpd.overlay(
        region, gdf_msk, "intersection", keep_geom_type=False
    ).explode()
    gdf_msk = gdf_msk[gdf_msk.length > 0]
    return gdf_msk


def mask2gdf(
    da_mask: xr.DataArray,
    option: str = "all",
) -> gpd.GeoDataFrame:
    """Convert a boolean mask to a GeoDataFrame of polygons.

    Parameters
    ----------
    da_mask: xr.DataArray
        Mask with integer values.
    option: {"all", "active", "wlev", "outflow"}

    Returns
    -------
    gdf: geopandas.GeoDataFrame
        GeoDataFrame of Points.
    """
    if option == "all":
        da_mask = da_mask != da_mask.raster.nodata
    elif option == "active":
        da_mask = da_mask == 1
    elif option == "wlev":
        da_mask = da_mask == 2
    elif option == "outflow":
        da_mask = da_mask == 3

    indices = np.stack(np.where(da_mask), axis=-1)

    if "x" in da_mask.coords:
        x = da_mask.coords["x"].values[indices[:, 1]]
        y = da_mask.coords["y"].values[indices[:, 0]]
    else:
        x = da_mask.coords["xc"].values[indices[:, 0], indices[:, 1]]
        y = da_mask.coords["yc"].values[indices[:, 0], indices[:, 1]]

    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=da_mask.raster.crs)

    if len(points) > 0:
        return gpd.GeoDataFrame(points, crs=da_mask.raster.crs)
    else:
        return None


## STRUCTURES: thd / weir ##


def gdf2linestring(gdf: gpd.GeoDataFrame) -> List[Dict]:
    """Convert GeoDataFrame[LineString] to list of structure dictionaries

    The x,y are taken from the geometry.
    For weir structures to additional paramters are required, a "z" (elevation) and
    "par1" (Cd coefficient in weir formula) are required which should be supplied
    as columns (or z-coordinate) of the GeoDataFrame. These columns should either
    contain a float or 1D-array of floats with same length as the LineString.

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame with LineStrings geometries
        GeoDataFrame structures.

    Returns
    -------
    feats: list of dict
        List of dictionaries describing structures.
    """
    feats = []
    for _, item in gdf.iterrows():
        feat = item.drop("geometry").dropna().to_dict()
        # check geom
        line = item.geometry
        if line.type == "MultiLineString" and len(line.geoms) == 1:
            line = line.geoms[0]
        if line.type != "LineString":
            raise ValueError("Invalid geometry type, only LineString is accepted.")
        xyz = tuple(zip(*line.coords[:]))
        feat["x"], feat["y"] = list(xyz[0]), list(xyz[1])
        if len(xyz) == 3:
            feat["z"] = list(xyz[2])
        feats.append(feat)
    return feats


def gdf2polygon(gdf: gpd.GeoDataFrame) -> List[Dict]:
    """Convert GeoDataFrame[Polygon] to list of structure dictionaries

    The x,y are taken from the geometry.

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame with LineStrings geometries
        GeoDataFrame structures.

    Returns
    -------
    feats: list of dict
        List of dictionaries describing structures.
    """
    feats = []
    for _, item in gdf.iterrows():
        feat = item.drop("geometry").dropna().to_dict()
        # check geom
        poly = item.geometry
        if poly.type == "MultiPolygon" and len(poly.geoms) == 1:
            poly = poly.geoms[0]
        if poly.type != "Polygon":
            raise ValueError("Invalid geometry type, only Polygon is accepted.")
        x, y = poly.exterior.coords.xy
        feat["x"], feat["y"] = list(x), list(y)
        feats.append(feat)
    return feats


def linestring2gdf(feats: List[Dict], crs: Union[int, CRS] = None) -> gpd.GeoDataFrame:
    """Convert list of structure dictionaries to GeoDataFrame[LineString]

    Parameters
    ----------
    feats: list of dict
        List of dictionaries describing structures.
    crs: int, CRS
        Coordinate reference system

    Returns
    -------
    gdf: geopandas.GeoDataFrame
        GeoDataFrame structures
    """
    records = []
    for f in feats:
        feat = copy.deepcopy(f)
        xyz = [feat.pop("x"), feat.pop("y")]
        if "z" in feat and np.atleast_1d(feat["z"]).size == len(xyz[0]):
            xyz.append(feat.pop("z"))
        feat.update({"geometry": LineString(list(zip(*xyz)))})
        records.append(feat)
    gdf = gpd.GeoDataFrame.from_records(records)
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
    return gdf


def polygon2gdf(
    feats: List[Dict],
    crs: Union[int, CRS] = None,
    zmin: float = None,
    zmax: float = None,
) -> gpd.GeoDataFrame:
    """Convert list of structure dictionaries to GeoDataFrame[Polygon]

    Parameters
    ----------
    feats: list of dict
        List of dictionaries describing polygons.
    crs: int, CRS
        Coordinate reference system

    Returns
    -------
    gdf: geopandas.GeoDataFrame
        GeoDataFrame structures
    """
    records = []
    for f in feats:
        feat = copy.deepcopy(f)
        xy = [feat.pop("x"), feat.pop("y")]
        feat.update({"geometry": Polygon(list(zip(*xy)))})
        records.append(feat)
    gdf = gpd.GeoDataFrame.from_records(records)
    gdf["zmin"] = zmin
    gdf["zmax"] = zmax
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
    return gdf


def write_geoms(
    fn: Union[str, Path], feats: List[Dict], stype: str = "thd", fmt="%.1f"
) -> None:
    """Write list of structure dictionaries to file

    Parameters
    ----------
    fn: str, Path
        Path to output structure file.
    feats: list of dict
        List of dictionaries describing structures.
        For pli, pol and thd files "x" and "y" are required, "name" is optional.
        For weir files "x", "y" and "z" are required, "name" and "par1" are optional.
    stype: {'pli', 'pol', 'thd', 'weir'}
        Geom type polylines (pli), polygons (pol) thin dams (thd) or weirs (weir).
    fmt: str
        format for "z" and "par1" fields.

    Examples
    --------
    >>> feats = [
            {
                "name": 'WEIR01',
                "x": [0, 10, 20],
                "y": [100, 100, 100],
                "z": 5.0,
                "par1": 0.6,
            },
            {
                "name": 'WEIR02',
                "x": [100, 110, 120],
                "y": [100, 100, 100],
                "z": [5.0, 5.1, 5.0],
                "par1": 0.6,
            },
        ]
    >>> write_structures('sfincs.weir', feats, stype='weir')
    """
    cols = {"pli": 2, "pol": 2, "thd": 2, "weir": 4}[stype.lower()]
    fmt = ["%.0f", "%.0f"] + [fmt for _ in range(cols - 2)]
    if stype.lower() == "weir" and np.any(["z" not in f for f in feats]):
        raise ValueError('"z" value missing for weir files.')
    with open(fn, "w") as f:
        for i, feat in enumerate(feats):
            name = feat.get("name", i + 1)
            if isinstance(name, int):
                name = f"{stype:s}{name:02d}"
            rows = len(feat["x"])
            a = np.zeros((rows, cols), dtype=np.float32)
            a[:, 0] = np.asarray(feat["x"]).round(0)
            a[:, 1] = np.asarray(feat["y"]).round(0)
            if stype.lower() == "weir":
                a[:, 2] = feat["z"]
                a[:, 3] = feat.get("par1", 0.6)
            s = io.BytesIO()
            np.savetxt(s, a, fmt=fmt)
            f.write(f"{name}\n")
            f.write(f"{rows:d} {cols:d}\n")
            f.write(s.getvalue().decode())


def read_geoms(fn: Union[str, Path]) -> List[Dict]:
    """Read structure files to list of dictionaries.

    Parameters
    ----------
    fn : str, Path
        Path to structure file.

    Returns
    -------
    feats: list of dict
        List of dictionaries describing structures.
    """
    feats = []
    col_names = ["x", "y", "z", "par1"]
    with open(fn, "r") as f:
        while True:
            name = f.readline().strip()
            if not name:  # EOF
                break
            feat = {"name": name}
            rows, cols = [int(v) for v in f.readline().strip().split(maxsplit=2)]
            for c in range(cols):
                feat[col_names[c]] = [0.0 for _ in range(rows)]
            for r in range(rows):
                for c, v in enumerate(f.readline().strip().split(maxsplit=cols)):
                    feat[col_names[c]][r] = float(v)
            if cols > 2:
                for c in col_names[2:]:
                    if np.unique(feat[c]).size == 1:
                        feat[c] = feat[c][0]
            feats.append(feat)
    return feats


## OUTPUT: sfincs_map.nc, sfincs_his.nc ##


def read_sfincs_map_results(
    fn_map: Union[str, Path],
    hmin: float = 0.0,
    crs: Union[int, CRS] = None,
    chunksize: int = 100,
    drop: List[str] = ["crs", "sfincsgrid"],
    logger=logger,
    **kwargs,
) -> Tuple[xr.Dataset]:
    """Read sfincs_map.nc staggered grid netcdf files and parse to two
    hydromt.RasterDataset objects: one with face and one with edge variables.

    Additionally, hmax is computed from zsmax and zb if present.

    Parameters
    ----------
    fn_map : str, Path
        Path to sfincs_map.nc file
    hmin: float, optional
        Minimum water depth to consider in hmax map, i.e. cells with lower depth
        get a nodata values assigned. By deafult 0.0
    crs: int, CRS
        Coordinate reference system
    chunksize: int, optional
        chunk size along time dimension, by default 100
    drop : List[str], optional
        Variables to drop from reading, by default ["crs", "sfincsgrid"]

    Returns
    -------
    ds_face, ds_edge: hydromt.RasterDataset
        Parsed SFINCS output map file
    """

    ds_map = xr.open_dataset(fn_map, chunks={"time": chunksize}, **kwargs)
    dvars = list(ds_map.data_vars.keys())
    edge_dims = [var for var in dvars if (var.endswith("_x") or var.endswith("_y"))]
    ds_map = ds_map.set_coords(["x", "y"] + edge_dims)
    crs = ds_map["crs"].item() if ds_map["crs"].item() > 0 else crs

    if ds_map["inp"].attrs.get("rotation") != 0:
        logger.warning("Cannot parse rotated maps. Skip reading sfincs.map.nc")
        return xr.Dataset(), xr.Dataset()

    # split general+face and edge vars
    face_vars = list(ds_map.data_vars.keys())
    edge_vars = []
    if len(edge_dims) == 2:
        edge = edge_dims[0][:-2]
        face_vars = [
            v for v in face_vars if f"{edge}_m" not in ds_map[v].dims and v not in drop
        ]
        edge_vars = [
            v for v in face_vars if f"{edge}_m" in ds_map[v].dims and v not in drop
        ]

    # read face vars
    face_coords = {
        "x": xr.IndexVariable("x", ds_map["x"].isel(n=0).values),
        "y": xr.IndexVariable("y", ds_map["y"].isel(m=0).values),
    }
    ds_face = (
        ds_map[face_vars]
        .drop_vars(["x", "y"])
        .rename({"n": "y", "m": "x"})
        .assign_coords(face_coords)
        .transpose(..., "y", "x")
    )
    # compute hmax
    if "zsmax" in ds_face:
        logger.info('Computing "hmax = max(zsmax) - zb"')
        hmax = ds_face["zsmax"].max("timemax") - ds_face["zb"]
        hmax = hmax.where(hmax > hmin, -9999)
        hmax.raster.set_nodata(-9999)
        ds_face["hmax"] = hmax
    # set spatial attrs
    ds_face.raster.set_spatial_dims(x_dim="x", y_dim="y")
    ds_face.raster.set_crs(crs)

    # get edge vars
    ds_edge = xr.Dataset()
    if len(edge_vars) > 0:
        edge_coords = {
            f"{edge}_x": xr.IndexVariable(
                f"{edge}_x", ds_map[f"{edge}_x"].isel(edge_n=0).values
            ),
            f"{edge}_y": xr.IndexVariable(
                f"{edge}_y", ds_map[f"{edge}_y"].isel(edge_m=0).values
            ),
        }
        ds_edge = (
            ds_map[edge_vars]
            .drop_vars([f"{edge}_x", f"{edge}_y"])
            .rename({f"{edge}_n": f"{edge}_y", f"{edge}_m": f"{edge}_x"})
            .assign_coords(edge_coords)
            .transpose(..., f"{edge}_y", f"{edge}_x")
        ).rename({f"{edge}_x": "x", f"{edge}_y": "y"})
        ds_edge.raster.set_crs(crs)

    return ds_face, ds_edge


def read_sfincs_his_results(
    fn_his: Union[str, Path],
    crs: Union[int, CRS] = None,
    chunksize: int = 100,
    **kwargs,
) -> xr.Dataset:
    """Read sfincs_his.nc point timeseries netcdf file and parse to hydromt.GeoDataset object.

    Parameters
    ----------
    fn_his : str, Path
        Path to sfincs_his.nc file
    crs: int, CRS
        Coordinate reference system
    chunksize: int, optional
        chunk size along time dimension, by default 100

    Returns
    -------
    ds_his: xr.Dataset
        Parsed SFINCS output his file.
    """

    ds_his = xr.open_dataset(fn_his, chunks={"time": chunksize}, **kwargs)
    crs = ds_his["crs"].item() if ds_his["crs"].item() != 0 else crs
    dvars = list(ds_his.data_vars.keys())
    # set coordinates & spatial dims
    cvars = ["id", "name", "x", "y"]
    ds_his = ds_his.set_coords([v for v in dvars if v.split("_")[-1] in cvars])
    ds_his.vector.set_spatial_dims(
        x_name="station_x", y_name="station_y", index_dim="stations"
    )
    # set crs
    ds_his.vector.set_crs(crs)

    return ds_his


def downscale_floodmap(
    zsmax: xr.DataArray,
    dep: xr.DataArray,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    floodmap_fn: Union[Path, str] = "floodmap.tif",
    reproj_method: str = "nearest",
):
    """Create a downscaled floodmap for (model) region.

    Parameters
    ----------
    zsmax : xr.DataArray
        Maximum water level (m). When multiple timesteps provided, maximum over all timesteps is used.
    dep : xr.DataArray
        High-resolution DEM (m) of model region.
    hmin : float, optional
        Minimum water depth (m) to be considered as "flooded", by default 0.05
    gdf_mask : gpd.GeoDataFrame, optional
        Geodataframe with polygons to mask floodmap, example containing the landarea, by default None
        Note that the area outside the polygons is set to nodata.
    floodmap_fn : Union[Path, str], optional
        Name (path) of output floodmap, by default "floodmap.tif"
    reproj_method : str, optional
        Reprojection method for downscaling the water levels, by default "nearest". 
        Other option is "bilinear".
    """

    # interpolate zsmax to dep grid
    zsmax = zsmax.raster.reproject_like(dep, method=reproj_method)
    zsmax.raster.set_nodata(np.nan)

    # compute hmax
    if "timemax" in zsmax.dims:
        hmax = zsmax.max("timemax") - dep
    elif "time" in zsmax.dims:
        hmax = zsmax.max("time") - dep
    else:
        hmax = zsmax - dep

    # remove flood-depths below threshold
    hmax = hmax.where(hmax > hmin, np.nan)

    if gdf_mask is not None:
        mask = hmax.raster.geometry_mask(gdf_mask, all_touched=True)
        hmax = hmax.where(mask)
        floodmap_fn = floodmap_fn.replace(".tif", "_mask.tif")

    if floodmap_fn is None:
        return hmax

    # write floodmap
    hmax.raster.to_raster(
        floodmap_fn,
        driver="GTiff",
        dtype=np.float32,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=2,
        profile="COG",
        nodata=np.nan,
    )


def downscale_floodmap_webmercator(
    zsmax: Union[np.array, xr.DataArray],
    index_path: str,
    topobathy_path: str,
    floodmap_path: str,
    hmin: float = 0.05,
    zoom_range: Union[int, List[int]] = [0, 13],
    fmt_in: str = "bin",
    fmt_out: str = "png",
    merge: bool = True,
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
        Merge floodmap tiles with existing floodmap tiles (this could for example happen when there is overlap between models), by default True
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
                    ind = tiling.png2int(index_fn)

                # read the topobathy file
                dep_fn = os.path.join(topobathy_path, str(izoom), x, y_file)
                if fmt_in == "bin":
                    dep = np.fromfile(dep_fn, dtype="f4")
                elif fmt_in == "png":
                    dep = tiling.png2elevation(dep_fn)

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
                    tiling.elevation2png(hmax, floodmap_fn)
