"""
HydroMT-SFINCS utilities functions for reading and writing SFINCS specific input and output files,
as well as some common data conversions.
"""

import copy
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import hydromt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.rio.overview import get_maximum_overview_level
from rasterio.windows import Window
import xarray as xr
from hydromt.io import write_xy
from pyproj.crs.crs import CRS
from shapely.geometry import LineString, Polygon


__all__ = [
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
    "write_xy",  # defined in hydromt.io
    "read_xyn",
    "write_xyn",
    "read_geoms",
    "write_geoms",
    "read_drn",
    "write_drn",
    "gdf2linestring",
    "gdf2polygon",
    "linestring2gdf",
    "polygon2gdf",
    "read_sfincs_map_results",
    "read_sfincs_his_results",
    "downscale_floodmap",
    "rotated_grid",
    "build_overviews",
    "find_uv_indices",
]

logger = logging.getLogger(__name__)


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
    df = pd.read_csv(fn, index_col=False, header=None, sep="\s+").rename(
        columns={0: "x", 1: "y"}
    )
    if len(df.columns) > 2:
        df = df.rename(columns={2: "name"})
    else:
        df["name"] = df.index

    points = gpd.points_from_xy(df["x"], df["y"])
    gdf = gpd.GeoDataFrame(df.drop(columns=["x", "y"]), geometry=points, crs=crs)

    return gdf


def write_xyn(fn: str = "sfincs.obs", gdf: gpd.GeoDataFrame = None, fmt: str = "%.1f"):
    # strip %-sign of fmt if present
    fmt = fmt.replace("%", "")

    with open(fn, "w") as fid:
        for point in gdf.iterfeatures():
            x, y = point["geometry"]["coordinates"]
            try:
                name = point["properties"]["name"]
            except:
                name = "obs" + str(point["id"])
            string = f'{x:{fmt}} {y:{fmt}} "{name}"\n'
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
    df = pd.read_csv(fn, index_col=0, header=None, sep="\s+")
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
    w = int(np.floor(np.log10(abs(data[-1, 0])))) + 3
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
    # small buffer for rounding errors
    if da_msk.raster.crs.is_geographic:
        gdf_msk["geometry"] = gdf_msk.buffer(1e-6)
    else:
        gdf_msk["geometry"] = gdf_msk.buffer(1)
    region = (da_msk >= 1).astype("int16").raster.vectorize()
    region = region[region["value"] == 1].drop(columns="value")
    region["geometry"] = region.boundary
    gdf_msk = gdf_msk[gdf_msk["value"] != 1]
    gdf_msk = gpd.overlay(
        region, gdf_msk, "intersection", keep_geom_type=False
    ).explode(index_parts=True)
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
        if line.geom_type == "MultiLineString" and len(line.geoms) == 1:
            line = line.geoms[0]
        if line.geom_type != "LineString":
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
    gdf.set_geometry("geometry", inplace=True)
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
    gdf.set_geometry("geometry", inplace=True)
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
    return gdf


def write_geoms(
    fn: Union[str, Path],
    feats: List[Dict],
    stype: str = "thd",
    fmt: str = "%.1f",
    fmt_z: str = "%.1f",
) -> None:
    """Write list of structure dictionaries to file

    Parameters
    ----------
    fn: str, Path
        Path to output structure file.
    feats: list of dict
        List of dictionaries describing structures.
        For pli, pol, thd anc crs files "x" and "y" are required, "name" is optional.
        For weir files "x", "y" and "z" are required, "name" and "par1" are optional.
    stype: {'pli', 'pol', 'thd', 'weir', 'crs'}
        Geom type polylines (pli), polygons (pol) thin dams (thd), weirs (weir)
        or cross-sections (crs).
    fmt: str
        format for "x" and "y" fields.
    fmt_z: str
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
    cols = {"pli": 2, "pol": 2, "thd": 2, "weir": 4, "crs": 2}[stype.lower()]

    fmt = [fmt, fmt] + [fmt_z for _ in range(cols - 2)]
    if stype.lower() == "weir" and np.any(["z" not in f for f in feats]):
        raise ValueError('"z" value missing for weir files.')
    with open(fn, "w") as f:
        for i, feat in enumerate(feats):
            name = feat.get("name", i + 1)
            if isinstance(name, int):
                name = f"{stype:s}{name:02d}"
            rows = len(feat["x"])
            a = np.zeros((rows, cols), dtype=np.float32)
            a[:, 0] = np.asarray(feat["x"])
            a[:, 1] = np.asarray(feat["y"])
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


def write_drn(fn: Union[str, Path], gdf_drainage: gpd.GeoDataFrame, fmt="%.1f") -> None:
    """Write structure files from list of dictionaries.

    Parameters
    ----------
    fn : str, Path
        Path to structure file.
    drainage : gpd.GeoDataFrame
        Dataframe with drainage structure parameters and geometry.
    fmt : str
        Format for coordinate values.
    """

    # expected columns for drainage structures
    col_names = [
        "xsnk",
        "ysnk",
        "xsrc",
        "ysrc",
        "type",
        "par1",
        "par2",
        "par3",
        "par4",
        "par5",
    ]

    gdf = copy.deepcopy(gdf_drainage)
    # get geometry linestring and convert to xsnk, ysnk, xsrc, ysrc
    endpoints = gdf.boundary.explode(index_parts=True).unstack()
    gdf["xsnk"] = endpoints[0].x
    gdf["ysnk"] = endpoints[0].y
    gdf["xsrc"] = endpoints[1].x
    gdf["ysrc"] = endpoints[1].y
    gdf.drop(["geometry"], axis=1, inplace=True)

    # reorder columns based on col_names
    gdf = gdf[col_names]

    # change the format of the coordinates according to fmt
    for col in ["xsnk", "ysnk", "xsrc", "ysrc"]:
        gdf[col] = gdf[col].apply(lambda x: fmt % x)

    # write to file
    gdf.to_csv(fn, sep=" ", index=False, header=False)


def read_drn(fn: Union[str, Path], crs: int = None) -> gpd.GeoDataFrame:
    """Read drainage structure files to geodataframe.

    Parameters
    ----------
    fn : str, Path
        Path to drainge structure file.
    crs : int
        EPSG code for coordinate reference system.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe with drainage structure parameters and geometry.
    """

    # expected columns for drainage structures
    col_names = [
        "xsnk",
        "ysnk",
        "xsrc",
        "ysrc",
        "type",
        "par1",
        "par2",
        "par3",
        "par4",
        "par5",
    ]

    # read structure file
    df = pd.read_csv(fn, sep="\\s+", names=col_names)

    # get geometry linestring
    geom = [
        LineString([(xsnk, ysnk), (xsrc, ysrc)])
        for xsnk, ysnk, xsrc, ysrc in zip(
            df["xsnk"], df["ysnk"], df["xsrc"], df["ysrc"]
        )
    ]
    df.drop(["xsnk", "ysnk", "xsrc", "ysrc"], axis=1, inplace=True)

    # convert to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
    return gdf


## OUTPUT: sfincs_map.nc, sfincs_his.nc ##


def read_sfincs_map_results(
    fn_map: Union[str, Path],
    ds_like: xr.Dataset,
    chunksize: int = 100,
    drop: List[str] = ["crs", "sfincsgrid"],
    logger=logger,
    **kwargs,
) -> Tuple[xr.Dataset]:
    """Read sfincs_map.nc staggered grid netcdf files and parse to two
    hydromt.RasterDataset objects: one with face and one with edge variables.

    Parameters
    ----------
    fn_map : str, Path
        Path to sfincs_map.nc file
    ds_like: xr.Dataset
        Dataset with grid information to use for parsing.
    chunksize: int, optional
        chunk size along time dimension, by default 100
    drop : List[str], optional
        Variables to drop from reading, by default ["crs", "sfincsgrid"]

    Returns
    -------
    ds_face, ds_edge: hydromt.RasterDataset
        Parsed SFINCS output map file
    """
    rm = {
        "x": "xc",
        "y": "yc",
        "corner_x": "corner_xc",
        "corner_y": "corner_yc",
        "n": "y",
        "m": "x",
        "corner_n": "corner_y",
        "corner_m": "corner_x",
    }
    ds_map = xr.open_dataset(fn_map, chunks={"time": chunksize}, **kwargs)
    ds_map = ds_map.rename(
        {k: v for k, v in rm.items() if (k in ds_map or k in ds_map.dims)}
    )
    ds_map = ds_map.set_coords(
        [var for var in ds_map.data_vars.keys() if (var in rm.values())]
    )

    # support for older sfincs_map.nc files
    # check if x,y dimensions are in the order y,x
    ds_map = ds_map.transpose(..., "y", "x", "corner_y", "corner_x")

    # split face and edge variables
    scoords = ds_like.raster.coords
    tcoords = {tdim: ds_map[tdim] for tdim in ds_map.dims if tdim.startswith("time")}
    ds_face = xr.Dataset(coords={**scoords, **tcoords})
    ds_edge = xr.Dataset()
    for var in ds_map.data_vars:
        if var in drop:
            continue
        if "x" in ds_map[var].dims and "y" in ds_map[var].dims:
            # drop to overwrite with ds_like.raster.coords
            ds_face[var] = ds_map[var].drop_vars(["xc", "yc"])
        elif ds_map[var].ndim == 0:
            ds_face[var] = ds_map[var]
        else:
            ds_edge[var] = ds_map[var]

    # add crs
    if ds_like.raster.crs is not None:
        ds_face.raster.set_crs(ds_like.raster.crs)
        ds_edge.raster.set_crs(ds_like.raster.crs)

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
    crs = ds_his["crs"].item() if ds_his["crs"].item() > 0 else crs
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
    dep: Union[Path, str, xr.DataArray],
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    floodmap_fn: Union[Path, str] = None,
    reproj_method: str = "nearest",
    nrmax: int = 2000,
    logger=logger,
    **kwargs,
):
    """Create a downscaled floodmap for (model) region.

    Parameters
    ----------
    zsmax : xr.DataArray
        Maximum water level (m). When multiple timesteps provided, maximum over all timesteps is used.
    dep : Path, str, xr.DataArray
        High-resolution DEM (m) of model region:
        * If a Path or str is provided, the DEM is read from disk and the floodmap
        is written to disk (recommened for datasets that do not fit in memory.)
        * If a xr.DataArray is provided, the floodmap is returned as xr.DataArray
        and only written to disk when floodmap_fn is provided.
    hmin : float, optional
        Minimum water depth (m) to be considered as "flooded", by default 0.05
    gdf_mask : gpd.GeoDataFrame, optional
        Geodataframe with polygons to mask floodmap, example containing the landarea, by default None
        Note that the area outside the polygons is set to nodata.
    floodmap_fn : Union[Path, str], optional
        Name (path) of output floodmap, by default None. If provided, the floodmap is written to disk.
    reproj_method : str, optional
        Reprojection method for downscaling the water levels, by default "nearest".
        Other option is "bilinear".
    nrmax : int, optional
        Maximum number of cells per block, by default 2000. These blocks are used to prevent memory issues.
    kwargs : dict, optional
        Additional keyword arguments passed to `RasterDataArray.to_raster`.

    Returns
    -------
    hmax: xr.Dataset
        Downscaled and masked floodmap.

    See Also
    --------
    hydromt.raster.RasterDataArray.to_raster
    """
    # get maximum water level
    timedim = set(zsmax.dims) - set(zsmax.raster.dims)
    if timedim:
        zsmax = zsmax.max(timedim)

    # Hydromt expects a string so if a Path is provided, convert to str
    if isinstance(floodmap_fn, Path):
        floodmap_fn = str(floodmap_fn)

    if isinstance(dep, xr.DataArray):
        hmax = _downscale_floodmap_da(
            zsmax=zsmax,
            dep=dep,
            hmin=hmin,
            gdf_mask=gdf_mask,
            reproj_method=reproj_method,
        )

        # write floodmap
        if floodmap_fn is not None:
            if not kwargs:  # write COG by default
                kwargs = dict(
                    driver="GTiff",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    compress="deflate",
                    predictor=2,
                    profile="COG",
                )
            hmax.raster.to_raster(floodmap_fn, **kwargs)

            # add overviews
            build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)

        hmax.name = "hmax"
        hmax.attrs.update({"long_name": "Maximum flood depth", "units": "m"})
        return hmax

    elif isinstance(dep, (str, Path)):
        if floodmap_fn is not None:
            raise ValueError(
                "floodmap_fn should be provided when dep is a Path or str."
            )

        with rasterio.open(dep) as src:
            # Define block size
            n1, m1 = src.shape
            nrcb = nrmax  # nr of cells in a block
            nrbn = int(np.ceil(n1 / nrcb))  # nr of blocks in n direction
            nrbm = int(np.ceil(m1 / nrcb))  # nr of blocks in m direction

            # avoid blocks with width or height of 1
            merge_last_col = False
            merge_last_row = False
            if m1 % nrcb == 1:
                nrbm -= 1
                merge_last_col = True
            if n1 % nrcb == 1:
                nrbn -= 1
                merge_last_row = True

            profile = dict(
                driver="GTiff",
                width=src.width,
                height=src.height,
                count=1,
                dtype=np.float32,
                crs=src.crs,
                transform=src.transform,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
                predictor=2,
                profile="COG",
                nodata=np.nan,
                BIGTIFF="YES",  # Add the BIGTIFF option here
            )

            with rasterio.open(floodmap_fn, "w", **profile):
                pass

            ## Loop through blocks
            for ii in range(nrbm):
                bm0 = ii * nrcb  # Index of first m in block
                bm1 = min(bm0 + nrcb, m1)  # last m in block
                if merge_last_col and ii == (nrbm - 1):
                    bm1 += 1

                for jj in range(nrbn):
                    bn0 = jj * nrcb  # Index of first n in block
                    bn1 = min(bn0 + nrcb, n1)  # last n in block
                    if merge_last_row and jj == (nrbn - 1):
                        bn1 += 1

                    # Define a window to read a block of data
                    window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

                    # Read the block of data
                    block_data = src.read(window=window)

                    # check for nan-data
                    if np.all(np.isnan(block_data)):
                        continue

                    # TODO directly use the rasterio warp method rather than the raster.reproject see PR #145
                    # Convert row and column indices to pixel coordinates
                    cols, rows = np.indices((bm1 - bm0, bn1 - bn0))
                    x_coords, y_coords = src.transform * (cols + bm0, rows + bn0)

                    # Create xarray DataArray with coordinates
                    block_dep = xr.DataArray(
                        block_data.squeeze().transpose(),
                        dims=("y", "x"),
                        coords={
                            "yc": (("y", "x"), y_coords),
                            "xc": (("y", "x"), x_coords),
                        },
                    )
                    block_dep.raster.set_crs(src.crs)

                    block_hmax = _downscale_floodmap_da(
                        zsmax=zsmax,
                        dep=block_dep,
                        hmin=hmin,
                        gdf_mask=gdf_mask,
                        reproj_method=reproj_method,
                    )

                    with rasterio.open(floodmap_fn, "r+") as fm_tif:
                        fm_tif.write(
                            np.transpose(block_hmax.values),
                            window=window,
                            indexes=1,
                        )

        # add overviews
        build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)


def rotated_grid(
    pol: Polygon, res: float, dec_origin=0, dec_rotation=3
) -> Tuple[float, float, int, int, float]:
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
        return round(np.degrees(angle), dec_rotation)

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
    x0, y0 = round(x0, dec_origin), round(y0, dec_origin)
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


def build_overviews(
    fn: Union[str, Path],
    resample_method: str = "average",
    overviews: Union[list, str] = "auto",
    logger=logger,
):
    """Build overviews for GeoTIFF file.

    Overviews are reduced resolution versions of your dataset that can speed up
    rendering when you don’t need full resolution. By precomputing the upsampled
    pixels, rendering can be significantly faster when zoomed out.

    Parameters
    ----------
    fn : str, Path
        Path to GeoTIFF file.
    method: str
        Resampling method, by default "average". Other option is "nearest".
    overviews: list of int, optional
        List of overview levels, by default "auto". When set to "auto" the
        overview levels are determined based on the size of the dataset.
    """

    # Endswith is not a method of Path so convert to str
    if isinstance(fn, Path):
        fn = str(fn)

    # check if fn is a geotiff file
    extensions = [".tif", ".tiff"]
    assert any(
        fn.endswith(ext) for ext in extensions
    ), f"File {fn} is not a GeoTIFF file."

    # open rasterio dataset
    with rasterio.open(fn, "r+") as src:
        # determine overviews when not provided
        if overviews == "auto":
            bs = src.profile.get("blockxsize", 256)
            max_level = get_maximum_overview_level(src.width, src.height, bs)
            overviews = [2**j for j in range(1, max_level + 1)]
        if not isinstance(overviews, list):
            raise ValueError("overviews should be a list of integers or 'auto'.")

        resampling = getattr(Resampling, resample_method, None)
        if resampling is None:
            raise ValueError(f"Resampling method unknown: {resample_method}")

        no = len(overviews)
        logger.info(f"Building {no} overviews with {resample_method}")

        # create new overviews, resampling with average method
        src.build_overviews(overviews, resampling)

        # update dataset tags
        src.update_tags(ns="rio_overview", resampling=resample_method)


def _downscale_floodmap_da(
    zsmax: xr.DataArray,
    dep: xr.DataArray,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    reproj_method: str = "nearest",
) -> xr.DataArray:
    """Create a downscaled floodmap for (model) region.

    Parameters
    ----------
    zsmax : xr.DataArray
        Maximum water level (m). When multiple timesteps provided, maximum over all timesteps is used.
    dep : Path, str, xr.DataArray
        High-resolution DEM (m) of model region:
    hmin : float, optional
        Minimum water depth (m) to be considered as "flooded", by default 0.05
    gdf_mask : gpd.GeoDataFrame, optional
        Geodataframe with polygons to mask floodmap, example containing the landarea, by default None
        Note that the area outside the polygons is set to nodata.
    """

    # interpolate zsmax to dep grid
    zsmax = zsmax.raster.reproject_like(dep, method=reproj_method)
    zsmax = zsmax.raster.mask_nodata()  # make sure nodata is nan

    # get flood depth
    hmax = (zsmax - dep).astype("float32")
    hmax.raster.set_nodata(np.nan)

    # mask floodmap
    hmax = hmax.where(hmax > hmin)
    if gdf_mask is not None:
        mask = hmax.raster.geometry_mask(gdf_mask, all_touched=True)
        hmax = hmax.where(mask)

    return hmax


def find_uv_indices(mask: xr.DataArray):
    """The subgrid tables for a regular SFINCS grid are organized as flattened arrays, meaning
    2D arrays (y,x) are transformed into 1D arrays, only containing values for active cells.

    For the cell centers, this is straightforward, we just find the indices of the active cells.
    However, the u and v points are saved in combined arrays. Since u and v points are absent
    at the boundaries of the domain, the index arrays are used to determine the location of the
    u and v points in the combined flattened arrays.



    Parameters
    ----------
    mask: xr.DataArray
        Mask with integer values specifying the active cells of the SFINCS domain.

    Returns
    -------
    index_nm: np.ndarray
        Index array for the active cell centers.
    index_mu1: np.ndarray
        Index of upstream u-point in combined uv-array.
    index_nu1: np.ndarray
        Index of upstream v-point in combined uv-array.

    """

    mask = mask.values

    # nr of cells
    nr_cells = mask.shape[0] * mask.shape[1]

    # get the index of the u and v points in a combined array
    mu1 = np.zeros(nr_cells, dtype=int) - 1
    nu1 = np.zeros(nr_cells, dtype=int) - 1

    ms = np.linspace(0, mask.shape[1] - 1, mask.shape[1], dtype=int)
    ns = np.linspace(0, mask.shape[0] - 1, mask.shape[0], dtype=int)

    m, n = np.meshgrid(ms, ns)

    m = np.transpose(m).flatten()
    n = np.transpose(n).flatten()

    mask = mask.transpose().flatten()

    nmax = n.max() + 1
    nms = m * nmax + n

    for ic in range(nr_cells):
        # nu1
        nn = n[ic] + 1
        if nn < nmax:
            mm = m[ic]
            nm = mm * nmax + nn
            j = binary_search(nms, nm)
            if j is not None:
                nu1[ic] = j
        # mu1
        nn = n[ic]
        mm = m[ic] + 1
        nm = mm * nmax + nn
        j = binary_search(nms, nm)
        if j is not None:
            mu1[ic] = j

    # For regular grids, only the points with mask > 0 are stored
    # The index arrays determine the location in the flattened arrays (with values for all active points)
    # Initialize index arrays with -1, inactive cells will remain -1
    index_nm = np.zeros(nr_cells, dtype=int) - 1
    index_mu1 = np.zeros(nr_cells, dtype=int) - 1
    index_nu1 = np.zeros(nr_cells, dtype=int) - 1
    npuv = 0
    npc = 0
    # Loop through all cells
    for ip in range(nr_cells):
        # Check if this cell is active
        if mask[ip] > 0:
            index_nm[ip] = npc
            npc += 1
            if mu1[ip] >= 0:
                if mask[mu1[ip]] > 0:
                    index_mu1[ip] = npuv
                    npuv += 1
            if nu1[ip] >= 0:
                if mask[nu1[ip]] > 0:
                    index_nu1[ip] = npuv
                    npuv += 1

    return index_nm, index_mu1, index_nu1


def binary_search(vals, val):
    indx = np.searchsorted(vals, val)
    if indx < np.size(vals):
        if vals[indx] == val:
            return indx
    return None
