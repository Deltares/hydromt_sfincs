"""Standalone reader functions for HydroMT-SFINCS."""

import logging
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg
import xarray as xr
from hydromt.readers import open_vector
from pyproj.crs import CRS

from hydromt_sfincs.utils import parse_datetime

__all__ = [
    "read_ascii_map",
    "read_bdr",
    "read_binary_map",
    "read_binary_map_index",
    "read_config",
    "read_drn",
    "read_geoms",
    "read_sfincs_his_results",
    "read_sfincs_map_results",
    "read_timeseries",
    "read_xy",
    "read_xyn",
]

logger = logging.getLogger(f"hydromt.{__name__}")


## Configuration: sfincs.inp ##
def read_config(
    filename: Path | str,
) -> dict[str, Any]:
    """Read the SFINCS input file (sfincs.inp).

    Parameters
    ----------
    filename : Path | str
        The path to the input file containing the SFINCS model settings.

    Returns
    -------
    dict[str, Any]
        The model settings.
    """
    # Ensure typing
    filename = Path(filename)
    # Check if exists
    if not filename.exists():
        raise FileNotFoundError(
            f"SFINCS input file '{filename.as_posix()}' does not exist."
        )

    # Read the file line by line
    with open(filename, "r") as fid:
        lines = fid.readlines()

    inp_dict = {}
    for line in lines:
        # Check if first character is #
        if line.strip().startswith("#"):
            # Full line comment
            continue
        # Find last character before #
        comment_idx = line.find("#")
        if comment_idx >= 0:
            line = line[:comment_idx]
        line = [x.strip() for x in line.split("=")]
        if len(line) != 2:
            continue
        name, val = line
        if name in ["tref", "tstart", "tstop"]:
            try:
                val = datetime.strptime(val, "%Y%m%d %H%M%S")
            except ValueError:
                raise ValueError(f'"{name} = {val}" not understood.')
        elif name in ["cdwnd", "cdval"]:
            val = [float(x) for x in val.split()]
        elif name == "utmzone":
            val = str(val)
        else:
            try:
                val = literal_eval(val)
            except Exception:
                pass

        if name == "crs":
            name = "epsg"
        elif name == "dtout":
            name = "dtmapout"

        inp_dict[name] = val

    # Return the config values
    return inp_dict


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


## ASCII TIMESERIES: bzs / dis / precip ##
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
    df = pd.read_csv(fn, index_col=0, header=None, sep=r"\s+")
    df.index = pd.to_datetime(df.index.values, unit="s", origin=tref)
    df.columns = df.columns.values.astype(int) - 1  # convert to zero-based index
    df.index.name = "time"
    df.columns.name = "index"
    return df


## BINARY MAPS: sfincs.ind, sfincs.msk, sfincs.dep etc. ##
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
    with xr.open_dataset(fn_his, chunks={"time": chunksize}, **kwargs) as ds_his:
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
    with xr.open_dataset(fn_map, chunks={"time": chunksize}, **kwargs) as ds_map:
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
        tcoords = {
            tdim: ds_map[tdim] for tdim in ds_map.dims if tdim.startswith("time")
        }
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


## STRUCTURES: thd / weir ##
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
    col_names = ["x", "y", "elevation", "par1"]
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
            # Always create a list
            # if cols > 2:
            #     for c in col_names[2:]:
            #         if np.unique(feat[c]).size == 1:
            #             feat[c] = feat[c][0]
            feats.append(feat)
    return feats


def read_bdr(fn: Union[str, Path], crs: int = None) -> gpd.GeoDataFrame:
    """Read river boundary file to geodataframe.

    Parameters
    ----------
    fn : str, Path
        Path to river boundary file.
    crs : int
        EPSG code for coordinate reference system.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe with river boundary parameters and geometry.
    """

    # expected columns for river boundary structures
    col_names = ["xbdr", "ybdr", "x_bdr_in", "y_bdr_in", "slope", "distance"]

    # read structure file
    df = pd.read_csv(fn, sep="\\s+", names=col_names)

    # get geometry linestring
    geom = [
        sg.LineString([(xbdr, ybdr), (x_bdr_in, y_bdr_in)])
        for xbdr, ybdr, x_bdr_in, y_bdr_in in zip(
            df["xbdr"], df["ybdr"], df["x_bdr_in"], df["y_bdr_in"]
        )
    ]
    df.drop(["xbdr", "ybdr", "x_bdr_in", "y_bdr_in"], axis=1, inplace=True)

    # convert to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
    return gdf


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
        "par6",
    ]

    # read structure file
    df = pd.read_csv(fn, sep="\\s+", header=None, dtype=float)

    # trim or expand columns to expected size
    n_expected = len(col_names)

    if df.shape[1] < n_expected:
        # add missing columns
        for i in range(df.shape[1], n_expected):
            df[i] = 0.0
    elif df.shape[1] > n_expected:
        # drop extra columns
        df = df.iloc[:, :n_expected]

    # assign names
    df.columns = col_names

    # get geometry linestring
    geom = [
        sg.LineString([(xsnk, ysnk), (xsrc, ysrc)])
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


## XY files: bnd / src ##
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
        GeoDataFrame with point geometries
    """
    gdf = open_vector(fn, crs=crs, driver="xy")
    gdf.index = np.arange(0, gdf.index.size, dtype=int)  # index starts at 0
    return gdf


def read_xyn(fn: Union[str, Path], crs: Union[int, CRS] = None) -> gpd.GeoDataFrame:
    """Read xyn files.

    For example observation points with names. When name column is not present,
    it will be generated as "point001", "point002", etc.

    Parameters
    ----------
    fn : str, Path
        Path to xyn file.
    crs: int, CRS
        Coordinate reference system.

    Returns
    -------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with point geometries.
    """
    df = pd.read_csv(fn, index_col=False, header=None, sep=r"\s+").rename(
        columns={0: "x", 1: "y"}
    )
    if len(df.columns) > 2:
        df = df.rename(columns={2: "name"})
    else:
        df["name"] = [f"point{i:03d}" for i in range(1, len(df) + 1)]

    points = gpd.points_from_xy(df["x"], df["y"])
    gdf = gpd.GeoDataFrame(df.drop(columns=["x", "y"]), geometry=points)
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
    return gdf
