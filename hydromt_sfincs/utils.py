"""
HydroMT-SFINCS utilities functions for reading and writing SFINCS specific input and output files,
as well as some common data conversions.
"""

import copy
from datetime import datetime
import io
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple, Union

from affine import Affine
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import xugrid as xu
from xugrid.core.wrap import UgridDataArray
from pyproj.crs.crs import CRS
from rasterio.enums import Resampling
from rasterio.rio.overview import get_maximum_overview_level
from rasterio.windows import Window
from shapely.geometry import LineString, Polygon

import hydromt
from hydromt.writers import write_xy
from hydromt.readers import open_vector
from hydromt.data_catalog.drivers import RasterioDriver
from hydromt.gis.gis_utils import zoom_to_overview_level
from hydromt.gis.vector import GeoDataset


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
    "write_vector",
    "write_raster",
    "gdf2linestring",
    "gdf2polygon",
    "linestring2gdf",
    "polygon2gdf",
    "read_sfincs_map_results",
    "read_sfincs_his_results",
    "downscale_floodmap",
    "dilate_zsmax",
    "apply_energy_head",
    "compute_flow_connected_mask",
    "remove_disconnected_flooding",
    "rotated_grid",
    "build_overviews",
    "find_uv_indices",
    "make_regular_grid",
    "make_regular_grid_transform",
    "partition_quadtree",
    "write_netcdf_safely",
]

logger = logging.getLogger(f"hydromt.{__name__}")


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
    df = pd.read_csv(fn, index_col=False, header=None, sep=r"\s+").rename(
        columns={0: "x", 1: "y"}
    )
    points = gpd.points_from_xy(df["x"], df["y"])
    gdf = gpd.GeoDataFrame(geometry=points)
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
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
    gdf = gpd.GeoDataFrame(df.drop(columns=["x", "y"]), geometry=points)
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
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
                name = "point" + str(point["id"])
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
    df.columns = df.columns.values.astype(int) - 1  # convert to zero-based index
    df.index.name = "time"
    df.columns.name = "index"
    return df


def write_timeseries(
    fn: Union[str, Path],
    df: Union[pd.DataFrame, pd.Series],
    tref: Union[str, datetime],
    fmt: str = "%7.3f",
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
def get_bounds_vector(
    da_msk: Union[xr.DataArray, xu.UgridDataArray],
) -> gpd.GeoDataFrame:
    """Get bounds of vectorized mask as GeoDataFrame.

    Parameters
    ----------
    da_msk: Union[xr.DataArray, xu.UgridDataArray]
        Mask as DataArray with values 0 (inactive), 1 (active),
        and boundary cells 2 (waterlevels) and 3 (outflow).

    Returns
    -------
    gdf_msk: gpd.GeoDataFrame
        GeoDataFrame with line geometries of mask boundaries.
    """
    if isinstance(da_msk, xr.DataArray):
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
    elif isinstance(da_msk, xu.UgridDataArray):
        lines = []

        xz = da_msk.grid.face_coordinates[:, 0]
        yz = da_msk.grid.face_coordinates[:, 1]
        min_dist = da_msk.grid.edge_length.max() * 2

        mask_vals = np.unique(da_msk.values)
        mask_vals = mask_vals[mask_vals > 1]

        for mval in mask_vals:
            # Indices for this mask value
            ibnd = np.where(da_msk.values == mval)

            xp = xz[ibnd]
            yp = yz[ibnd]

            if xp.size == 0:
                continue

            used = np.full(xp.shape, False, dtype=bool)
            polylines = []

            while True:
                if np.all(used):
                    break

                i1 = np.where(~used)[0][0]
                used[i1] = True
                polyline = [i1]

                # Forward direction
                while True:
                    xpunused = xp[~used]
                    ypunused = yp[~used]
                    unused_indices = np.where(~used)[0]

                    if unused_indices.size == 0:
                        break

                    dst = np.sqrt((xpunused - xp[i1]) ** 2 + (ypunused - yp[i1]) ** 2)
                    inear = np.nanargmin(dst)
                    inearall = unused_indices[inear]

                    if dst[inear] < min_dist:
                        polyline.append(inearall)
                        used[inearall] = True
                        i1 = inearall
                    else:
                        break

                # Backward direction
                i1 = polyline[0]
                while True:
                    xpunused = xp[~used]
                    ypunused = yp[~used]
                    unused_indices = np.where(~used)[0]

                    if unused_indices.size == 0:
                        break

                    dst = np.sqrt((xpunused - xp[i1]) ** 2 + (ypunused - yp[i1]) ** 2)
                    inear = np.nanargmin(dst)
                    inearall = unused_indices[inear]

                    if dst[inear] < min_dist:
                        polyline.insert(0, inearall)
                        used[inearall] = True
                        i1 = inearall
                    else:
                        break

                if len(polyline) > 1:
                    polylines.append(polyline)

            # Convert polylines to LineStrings
            for polyline in polylines:
                x = xp[polyline]
                y = yp[polyline]
                coords = list(zip(x.ravel(), y.ravel()))

                line = LineString(coords)

                if line.length == 0:
                    continue

                lines.append(
                    {
                        "value": int(mval),
                        "geometry": line,
                    }
                )

        gdf_msk = gpd.GeoDataFrame(lines, crs=da_msk.grid.crs)
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
    stype: {'pli', 'pol', 'thd', 'weir', 'crs', 'wvm'}
        Geom type polylines (pli), polygons (pol) thin dams (thd), weirs (weir),
        cross-sections (crs) or wavemaker (wvm).
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
    cols = {"pli": 2, "pol": 2, "thd": 2, "weir": 4, "crs": 2, "wvm": 2}[stype.lower()]

    fmt = [fmt, fmt] + [fmt_z for _ in range(cols - 2)]
    if stype.lower() == "weir" and np.any(["z" not in f for f in feats]):
        raise ValueError('"z" value missing for weir files.')
    with open(fn, "w") as f:
        for i, feat in enumerate(feats):
            name = feat.get("name", i + 1)
            if isinstance(name, int):
                name = f"{stype:s}{name:02d}"
            rows = len(feat["x"])
            a = np.zeros((rows, cols), dtype=np.float64)
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

    # change the format/precision of the coordinates according to fmt
    for col in ["xsnk", "ysnk", "xsrc", "ysrc"]:
        precision = fmt.split(".")[-1].replace("%", "").replace("f", "")
        gdf[col] = gdf[col].round(int(precision))

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


def write_vector(
    data: Union[xr.Dataset, gpd.GeoDataFrame],
    name: str,
    root: Union[str, Path],
    logger=logger,
    **kwargs,
):
    """Write model vector (geoms) variables to geojson files.

    NOTE: these files are not used by the model by just saved for visualization/
    analysis purposes.

    Parameters
    ----------
    data: geopandas.GeoDataFrame, xr.Dataset
        The data to write to file. If an xr.Dataset is provided, it should contain geometry variables
        that can be converted to a geopandas.GeoDataFrame.
    name: str
        The name of the variable to write to file. This will be used as the filename.
    root: Path, str, optional
        The output folder path.
    kwargs:
        Key-word arguments passed to geopandas.GeoDataFrame.to_file(driver='GeoJSON').
    """
    kwargs.update(driver="GeoJSON")  # fixed

    # check root
    if not os.path.isdir(root):
        os.makedirs(root)

    if isinstance(data, gpd.GeoDataFrame):
        gdf = data
    else:
        try:
            gdf = data.vector.to_gdf()
        except:
            logger.debug(f"Variable {name} could not be written to vector file.")
            pass

    gdf.to_file(os.path.join(root, f"{name}.geojson"), **kwargs)


def write_raster(
    data: Union[xr.Dataset, xr.DataArray],
    root: Union[str, Path],
    mask: xr.DataArray = None,
    driver="GTiff",
    compress="deflate",
    logger=logger,
    **kwargs,
):
    """Write model 2D raster variables to geotiff files.

    NOTE: these files are not used by the model by just saved for visualization/
    analysis purposes.

    Parameters
    ----------
    variables: str, list, optional
        Model variables are a combination of attribute and layer (optional) using <attribute>.<layer> syntax.
        Known ratster attributes are ["grid", "states", "results"].
        Different variables can be combined in a list.
        By default, variables is ["grid", "states", "results.hmax"]
    root: Path, str, optional
        The output folder path. If None it defaults to the <model_root>/gis folder (Default)
    kwargs:
        Key-word arguments passed to hydromt.RasterDataset.to_raster(driver='GTiff', compress='lzw').
    """

    # check variables
    if isinstance(data, xr.Dataset):
        variables = list(data.data_vars.keys())
        variables = [variables]
    elif isinstance(data, xr.DataArray):
        variables = [data.name]
    else:
        raise ValueError(
            f"Unsupported data type for writing raster: {type(data)}. "
            "Expected xr.Dataset or xr.DataArray."
        )

    # check mask
    if mask is None:
        if "mask" in data:
            mask = data["mask"]
        else:
            raise ValueError("No mask provided and no 'mask' variable found in data.")

    # check root
    if not os.path.isdir(root):
        os.makedirs(root)

    # save to file
    for var in variables:
        da = data[var] if isinstance(data, xr.Dataset) else data
        name = da.name
        if len(da.dims) != 2:
            # try to reduce to 2D by taking maximum over time dimension
            if "time" in da.dims:
                da = da.max("time")
            elif "timemax" in da.dims:
                da = da.max("timemax")
            # if still not 2D, skip
            if len(da.dims) != 2:
                logger.warning(f"Variable {name} has more than 2 dimensions: skipping.")
                continue
        # If the raster type is float, set nodata to np.nan
        if da.dtype == "float32" or da.dtype == "float64":
            da.raster.set_nodata(np.nan)
        # only write active cells to gis files
        da = da.where(mask > 0, da.raster.nodata).raster.mask_nodata()
        if da.raster.res[1] > 0:  # make sure orientation is N->S
            da = da.raster.flipud()
        da.raster.to_raster(
            os.path.join(root, f"{name}.tif"),
            driver=driver,
            compress=compress,
            **kwargs,
        )


def dilate_zsmax(
    zsmax: Union[xu.UgridDataArray, xr.DataArray],
    factor: float,
) -> Union[xu.UgridDataArray, xr.DataArray]:
    """Cell-space WSE dilation — works on both quadtree and regular grids.

    For each *already-wet* cell/pixel, raise its zsmax to the max of (own,
    wet-neighbour values) within a radius of ``(0.5 + factor)`` cell widths.
    Dry cells stay dry — the wet-cell set is preserved (key safeguard:
    no new cells are flooded).

    Typical use is to close 1 m-DEM connectivity gaps behind coarse-cell
    levees, where the parent cell's single WSE sits below the levee crest
    on the fine DEM.  Expanding each cell's WSE plateau by a modest
    fraction of its own size lets the flood cross the crest continuously
    without introducing new wet cells elsewhere.

    Parameters
    ----------
    zsmax : xu.UgridDataArray or xr.DataArray
        Maximum water level (m).  NaN where dry.  Quadtree grids are
        dispatched to a cKDTree-based implementation on the cell centres;
        regular grids to a ``scipy.ndimage.maximum_filter`` with a disk
        footprint in pixel units.
    factor : float
        Fraction of cell size.  ``factor=0`` picks up only the cell itself
        on a uniform grid; ``factor=0.5`` reaches the 4 edge-neighbours;
        ``factor=1.0`` reaches ~1.5 cell widths (full 3×3 stencil on a
        uniform grid).  Must be ``>= 0``.  Returned unchanged when
        ``factor <= 0``.

    Returns
    -------
    xu.UgridDataArray or xr.DataArray
        Dilated zsmax on the same grid and with the same wet-cell set as
        the input.  ``dilated >= zsmax`` on every wet cell.
    """
    if isinstance(zsmax, xu.UgridDataArray):
        return _dilate_zsmax_quadtree(zsmax, factor)
    if isinstance(zsmax, xr.DataArray):
        return _dilate_zsmax_regular(zsmax, factor)
    raise TypeError(
        "zsmax must be xu.UgridDataArray or xr.DataArray; "
        f"got {type(zsmax).__name__}."
    )


def _dilate_zsmax_quadtree(
    zsmax: xu.UgridDataArray,
    factor: float,
) -> xu.UgridDataArray:
    """Quadtree-grid dilation via cKDTree (see :func:`dilate_zsmax`)."""
    from scipy.spatial import cKDTree

    grid = zsmax.ugrid.grid
    face_x, face_y = grid.face_coordinates.T
    fb = grid.face_bounds                     # (n, 4): xmin, ymin, xmax, ymax
    dx_cell = fb[:, 2] - fb[:, 0]
    dy_cell = fb[:, 3] - fb[:, 1]
    dcell = np.maximum(dx_cell, dy_cell)

    vals = zsmax.values.astype(np.float64, copy=True)
    wet_before = ~np.isnan(vals)

    if factor <= 0.0:
        out = zsmax.copy()
        out.values = vals.astype(zsmax.dtype)
        return out

    tree = cKDTree(np.column_stack([face_x, face_y]))
    radii = dcell * (0.5 + factor)

    nbr_lists = tree.query_ball_point(
        np.column_stack([face_x, face_y]), r=radii,
    )

    dilated = vals.copy()
    for i, nbrs in enumerate(nbr_lists):
        if not nbrs:
            continue
        nbr_vals = vals[nbrs]
        wet_nbrs = nbr_vals[~np.isnan(nbr_vals)]
        if wet_nbrs.size == 0:
            continue
        nbr_max = float(wet_nbrs.max())
        if np.isnan(dilated[i]):
            dilated[i] = nbr_max
        else:
            dilated[i] = max(dilated[i], nbr_max)

    # Enforce the no-new-wet-cells constraint
    dilated[~wet_before] = np.nan

    _check_dilation_invariants(vals, dilated, wet_before)

    out = zsmax.copy()
    out.values = dilated.astype(zsmax.dtype)
    return out


def _dilate_zsmax_regular(
    zsmax: xr.DataArray,
    factor: float,
) -> xr.DataArray:
    """Regular-grid dilation via a disk footprint max-filter.

    Footprint radius is ``(0.5 + factor)`` pixels (Euclidean), matching
    the quadtree convention: ``factor=0.5`` reaches edge neighbours,
    ``factor=1.0`` reaches diagonal (full 3×3 stencil).
    """
    from scipy.ndimage import maximum_filter

    vals = zsmax.values.astype(np.float64, copy=True)
    wet_before = ~np.isnan(vals)

    if factor <= 0.0:
        out = zsmax.copy()
        out.values = vals.astype(zsmax.dtype)
        return out

    # Disk footprint in pixel units (Euclidean radius 0.5 + factor)
    R = 0.5 + float(factor)
    r = int(np.ceil(R))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    footprint = (xx * xx + yy * yy) <= (R * R)

    # Fill NaN with -inf so the max ignores dry cells
    filled = np.where(wet_before, vals, -np.inf)
    dilated = maximum_filter(filled, footprint=footprint, mode="constant", cval=-np.inf)

    # Enforce no-new-wet-cells: dry cells stay dry
    dilated = np.where(wet_before, dilated, np.nan)

    # A wet cell with no wet neighbour in-range still has its own value
    # in the footprint centre, so dilated >= vals on every wet cell.
    _check_dilation_invariants(vals, dilated, wet_before)

    out = zsmax.copy()
    out.values = dilated.astype(zsmax.dtype)
    return out


def _check_dilation_invariants(vals, dilated, wet_before):
    """Assert wet-set preservation + monotonic lift for dilation helpers."""
    wet_after = ~np.isnan(dilated)
    if not np.array_equal(wet_before, wet_after):
        raise RuntimeError("dilation changed the wet-cell set")
    raised = np.where(
        wet_before,
        dilated - np.where(wet_before, vals, 0.0),
        0.0,
    )
    if not np.all(raised >= -1e-9):
        raise RuntimeError("dilation lowered zsmax on some cell")


def apply_energy_head(
    zsmax: xu.UgridDataArray,
    qmax: xu.UgridDataArray,
    zb: Optional[xu.UgridDataArray] = None,
    hmin: float = 0.05,
    q_threshold: float = 0.01,
) -> xu.UgridDataArray:
    """Add the velocity head v²/(2g) to zsmax (Bernoulli correction).

    Lifts the water level on wet cells where the unit discharge exceeds
    ``q_threshold``, converting zsmax to the total-energy head
    ``H = zsmax + v² / (2g)``.  The wet-cell set is preserved: NaN cells
    stay NaN.

    This is a **method-agnostic pre-step** — the returned UgridDataArray
    can be consumed by any downscaling method (constant, bilinear, raw,
    volume-family, etc.).

    Parameters
    ----------
    zsmax : xu.UgridDataArray
        Maximum water level (m) on a SFINCS quadtree grid.  NaN where dry.
    qmax : xu.UgridDataArray
        Maximum unit discharge magnitude (m²/s), **cell-centred** — one
        value per cell, with the same shape and grid as ``zsmax``.  This is
        the convention SFINCS writes to ``sfincs_map.nc`` (variable
        ``qmax``) when ``storefluxmax=1``; no face-to-centre reduction is
        needed.  The sign is ignored (``|qmax|`` is used internally).
        The formula matches the legacy in-bilinear branch at
        ``utils._downscale_bilinear``: ``vel_head = q² / (h² * 2g)`` with
        ``h = max(zsmax - zb, hmin)``.
    zb : xu.UgridDataArray, optional
        Bed elevation (m) at cell centres, used to estimate depth.  If
        omitted, a constant depth of ``hmin`` is assumed (conservative —
        overestimates velocity and therefore the head correction).
    hmin : float, optional
        Minimum depth (m) for velocity estimation, by default 0.05.
    q_threshold : float, optional
        Minimum unit discharge magnitude (m²/s) to apply the correction,
        by default 0.01.  Cells below this threshold keep their original
        zsmax.

    Returns
    -------
    xu.UgridDataArray
        zsmax with the velocity head added on qualifying cells.  Same grid
        and same wet-cell set as the input.  ``result >= zsmax`` on every
        wet cell (velocity head is always non-negative).
    """
    GRAVITY = 9.81

    zs_vals = zsmax.values.astype(np.float64, copy=True)
    q_vals = np.abs(qmax.values.astype(np.float64, copy=False))
    wet_before = ~np.isnan(zs_vals)

    if zb is not None:
        zb_vals = zb.values.astype(np.float64, copy=False)
        h = np.where(wet_before, np.maximum(zs_vals - zb_vals, hmin), hmin)
    else:
        h = np.full_like(zs_vals, hmin)

    v = np.where(h > 0, q_vals / h, 0.0)
    dH = np.where(
        np.isfinite(q_vals) & (q_vals > q_threshold),
        0.5 * v * v / GRAVITY,
        0.0,
    )

    zs_new = zs_vals + np.where(wet_before, dH, 0.0)
    zs_new[~wet_before] = np.nan

    # Invariants
    wet_after = ~np.isnan(zs_new)
    if not np.array_equal(wet_before, wet_after):
        raise RuntimeError("energy-head correction changed the wet-cell set")
    raised = np.where(
        wet_before,
        zs_new - np.where(wet_before, zs_vals, 0.0),
        0.0,
    )
    if not np.all(raised >= -1e-9):
        raise RuntimeError("energy-head correction lowered zsmax on some cell")

    out = zsmax.copy()
    out.values = zs_new.astype(zsmax.dtype)
    return out


def downscale_floodmap(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: Union[Path, str, xr.DataArray],
    method: str = "constant",
    indices: Union[Path, str, xr.DataArray] = None,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    floodmap_fn: Union[Path, str] = None,
    zsmap_fn: Union[Path, str] = None,
    dilation: Optional[float] = None,
    energy_flux: Optional[bool] = None,
    qmax: xu.UgridDataArray = None,
    zb: xu.UgridDataArray = None,
    q_threshold: float = 0.01,
    q_scale: float = 0.5,
    reproj_method: str = "nearest",
    zoom_level: Optional[Union[int, tuple]] = None,
    nrmax: int = 2000,
    logger=logger,
    **kwargs,
):
    """Create a downscaled floodmap for (model) region.

    Supports multiple downscaling methods via the *method* parameter:

    * ``"raw"`` -- Paint each DEM pixel with its parent cell's WSE
      (nearest-neighbor).  No DEM subtraction, no wet/dry masking.
      Only produces a water-level raster (*zsmap_fn*).
    * ``"constant"`` -- Assign each DEM pixel the WSE of its parent cell
      (optionally via a pre-computed index COG), then subtract the DEM.
      This is the classic "bathtub" approach.
    * ``"bilinear"`` -- Bilinearly interpolate WSE from surrounding cell
      centers (Sanders & Schubert 2019), then subtract the DEM.

    Parameters
    ----------
    zsmax : xr.DataArray or xu.UgridDataArray
        Maximum water level (m).  When multiple timesteps are present the
        maximum over all timesteps is used.
    dep : Path, str, or xr.DataArray
        High-resolution DEM (m) of the model region.
    method : str, optional
        Downscaling method, by default ``"constant"``.
    indices : Path, str, or xr.DataArray, optional
        Pre-computed cell-index raster (only used by ``"constant"``).
    hmin : float, optional
        Minimum water depth (m) to be considered flooded, by default 0.05.
        Ignored by ``"raw"``.
    gdf_mask : gpd.GeoDataFrame, optional
        Polygons to mask the output (area outside is set to NaN).
    floodmap_fn : Path or str, optional
        Output flood-depth GeoTIFF.  Required for all methods except ``"raw"``.
    zsmap_fn : Path or str, optional
        Output water-level GeoTIFF.
    dilation : float, optional
        Cell-space WSE dilation factor.  When ``> 0``, each wet cell's
        zsmax is raised to the maximum of its wet neighbours within a
        radius of ``(0.5 + dilation)`` cell widths.  Dry cells stay dry
        (the wet-cell set is preserved).  Works on both quadtree
        (``xu.UgridDataArray``) and regular (``xr.DataArray``) ``zsmax``
        via :func:`dilate_zsmax`.  Default ``None`` (no dilation).
    energy_flux : bool, optional
        Method-agnostic Bernoulli / velocity-head correction switch.  When
        ``True``, ``zsmax`` is pre-modified via
        :func:`apply_energy_head` — ``H = zsmax + v²/(2g)`` on cells with
        ``|qmax| > q_threshold`` — before dispatch, so every downscaling
        method consumes the energy-adjusted water level.  Requires ``qmax``.
        When ``False``, the legacy in-bilinear Bernoulli blend (if any) is
        disabled by setting ``qmax`` to ``None`` internally.  Default
        ``None`` → *legacy auto*: if ``qmax`` is provided with
        ``method="bilinear"``, the in-bilinear blend runs; otherwise no
        correction is applied.
    qmax : xu.UgridDataArray, optional
        Maximum unit discharge (m²/s).  With ``energy_flux=True`` (any
        method), it feeds the pre-step velocity head; with
        ``energy_flux=None`` and ``method="bilinear"``, it feeds the legacy
        in-bilinear blend (face-based ``qmax`` with upstream energy
        propagation — see ``q_scale``).  Requires ``storefluxmax=1`` in the
        SFINCS configuration.
    zb : xu.UgridDataArray, optional
        Bed elevation at cell centres (m).  Used with *qmax* to compute
        water depth for velocity estimation.  If omitted, *hmin* is used as
        the minimum depth (conservative: overestimates velocity).
    q_threshold : float, optional
        Minimum unit discharge (m²/s) to activate the energy-head or
        upstream-energy propagation, by default 0.01.
    q_scale : float, optional
        Unit discharge (m²/s) at which the legacy in-bilinear upstream blend
        factor reaches 1.0, by default 0.5.  Ignored by the
        ``energy_flux=True`` pre-step path.
    reproj_method : str, optional
        Reprojection method for ``"constant"`` downscaling, by default
        ``"nearest"``.
    zoom_level : int or tuple, optional
        Overview level of the raster dataset (only for ``"constant"``).
    nrmax : int, optional
        Block size in pixels, by default 2000.
    logger : logging.Logger, optional
        Logger instance.
    kwargs : dict, optional
        Extra keyword arguments forwarded to ``RasterDataArray.to_raster``
        (only for the in-memory ``"constant"`` path).

    Returns
    -------
    hmax : xr.DataArray or None
        Returned only when *dep* is an ``xr.DataArray`` and
        ``method="constant"``.  Otherwise results are written to disk.
    """
    _VALID_METHODS = {"raw", "constant", "bilinear"}
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}.  Choose from {sorted(_VALID_METHODS)}."
        )

    # --- Reduce time dimension -----------------------------------------------
    if isinstance(zsmax, xu.UgridDataArray):
        timedim = set(zsmax.dims) - set(zsmax.ugrid.grid.dims)
    else:
        timedim = set(zsmax.dims) - set(zsmax.raster.dims)
    if timedim:
        logger.info(f"Taking maximum water level over {timedim} dimension(s).")
        zsmax = zsmax.max(timedim)

    if qmax is not None and isinstance(qmax, xu.UgridDataArray):
        q_timedim = set(qmax.dims) - set(qmax.ugrid.grid.dims)
        if q_timedim:
            qmax = qmax.max(q_timedim)

    # --- Pre-step 1: cell-space WSE dilation (quadtree or regular grid) ------
    if dilation is not None and dilation > 0.0:
        logger.info(f"Applying WSE dilation with factor={dilation:g}.")
        zsmax = dilate_zsmax(zsmax, factor=float(dilation))

    # --- Pre-step 2: energy-flux (Bernoulli velocity head) -------------------
    # Method-agnostic: runs before dispatch, so every method consumes the
    # energy-adjusted zsmax.  Takes precedence over the legacy in-bilinear
    # blend (which is disabled by setting qmax=None after the pre-step).
    if energy_flux is True:
        if not isinstance(zsmax, xu.UgridDataArray):
            raise ValueError(
                "energy_flux=True requires zsmax on a SFINCS quadtree "
                "(xu.UgridDataArray); got xr.DataArray."
            )
        if qmax is None:
            raise ValueError("energy_flux=True requires qmax.")
        logger.info("Applying velocity-head correction (energy_flux=True).")
        zsmax = apply_energy_head(
            zsmax, qmax=qmax, zb=zb, hmin=hmin, q_threshold=q_threshold,
        )
        qmax = None  # prevent the legacy in-bilinear branch from re-applying
    elif energy_flux is False:
        qmax = None  # force-disable the legacy in-bilinear branch too
    # energy_flux is None → legacy auto-behaviour: qmax passes through to
    #   _downscale_bilinear and drives its in-function Bernoulli blend.

    # --- In-memory path (xr.DataArray dep) -- only for "constant" ------------
    if isinstance(dep, xr.DataArray):
        if method != "constant":
            raise ValueError(
                "Only method='constant' supports xr.DataArray dep.  "
                "Use a file path for other methods."
            )
        if isinstance(floodmap_fn, Path):
            floodmap_fn = str(floodmap_fn)
        if indices is not None:
            if isinstance(indices, (str, Path)) and not isinstance(dep, (str, Path)):
                raise ValueError("index should be xr.DataArray when dep is xr.DataArray.")
            elif isinstance(indices, xr.DataArray) and not isinstance(dep, xr.DataArray):
                raise ValueError("index should be str/Path when dep is str/Path.")
        hmax = _downscale_floodmap_da(
            zsmax=zsmax, dep=dep, indices=indices, hmin=hmin,
            gdf_mask=gdf_mask, reproj_method=reproj_method,
        )
        if floodmap_fn is not None:
            if not kwargs:
                kwargs = dict(
                    driver="GTiff", tiled=True, blockxsize=256, blockysize=256,
                    compress="deflate", predictor=2, profile="COG",
                )
            hmax.raster.to_raster(floodmap_fn, **kwargs)
            build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)
        hmax.name = "hmax"
        hmax.attrs.update({"long_name": "Maximum flood depth", "units": "m"})
        return hmax

    # --- File-based path (dep is str/Path) -----------------------------------
    if method == "raw":
        if zsmap_fn is None:
            raise ValueError("zsmap_fn is required for method='raw'.")
    else:
        if floodmap_fn is None:
            raise ValueError("floodmap_fn is required when dep is a file path.")

    # Dispatch to the appropriate file-based implementation
    if method == "constant":
        _downscale_constant(
            zsmax=zsmax, dep=dep, indices=indices, hmin=hmin, gdf_mask=gdf_mask,
            floodmap_fn=floodmap_fn, zsmap_fn=zsmap_fn, reproj_method=reproj_method,
            zoom_level=zoom_level, nrmax=nrmax, logger=logger,
        )
    elif method == "raw":
        _downscale_raw(
            zsmax=zsmax, dep=dep, zsmap_fn=zsmap_fn, gdf_mask=gdf_mask,
            nrmax=nrmax, logger=logger, indices=indices,
        )
    elif method == "bilinear":
        _downscale_bilinear(
            zsmax=zsmax, dep=dep, hmin=hmin, gdf_mask=gdf_mask,
            floodmap_fn=floodmap_fn, zsmap_fn=zsmap_fn, nrmax=nrmax, logger=logger,
            indices=indices, qmax=qmax, zb=zb, q_threshold=q_threshold, q_scale=q_scale,
        )


# =============================================================================
#  Shared helpers
# =============================================================================

def _open_dem_geometry(dep):
    """Read only grid geometry (no elevation values) from a DEM GeoTIFF."""
    with rasterio.open(str(dep)) as src:
        return dict(
            transform=src.transform,
            width=src.width,
            height=src.height,
            crs=src.crs,
            dx=src.transform[0],
            dy=src.transform[4],
        )


def _make_output_profile(geo):
    """Standard COG profile for float32 output rasters."""
    return dict(
        driver="GTiff", width=geo["width"], height=geo["height"],
        count=1, dtype=np.float32, crs=geo["crs"], transform=geo["transform"],
        tiled=True, blockxsize=256, blockysize=256,
        compress="deflate", predictor=2, profile="COG",
        nodata=np.nan, BIGTIFF="YES",
    )


def _create_output_rasters(profile, floodmap_fn=None, zsmap_fn=None):
    """Create empty output GeoTIFF(s)."""
    if floodmap_fn is not None:
        with rasterio.open(str(floodmap_fn), "w", **profile):
            pass
    if zsmap_fn is not None:
        with rasterio.open(str(zsmap_fn), "w", **profile):
            pass


def _apply_mask_and_overviews(
    floodmap_fn, zsmap_fn, gdf_mask, geo, logger,
):
    """Apply polygon mask and build overviews on output raster(s)."""
    if gdf_mask is not None:
        logger.info("Applying polygon mask...")
        from rasterio.features import geometry_mask
        mask = geometry_mask(
            gdf_mask.geometry,
            out_shape=(geo["height"], geo["width"]),
            transform=geo["transform"],
            invert=True, all_touched=True,
        )
        for fn in [floodmap_fn, zsmap_fn]:
            if fn is None:
                continue
            with rasterio.open(str(fn), "r+") as dst:
                data = dst.read(1)
                data[~mask] = np.nan
                dst.write(data, indexes=1)

    for fn in [floodmap_fn, zsmap_fn]:
        if fn is not None:
            build_overviews(fn=str(fn), resample_method="nearest", logger=logger)


# =============================================================================
#  Method: raw  (nearest-neighbor WSE, no DEM subtraction)
# =============================================================================

def _downscale_raw(zsmax, dep, zsmap_fn, gdf_mask, nrmax, logger, indices=None):
    vals = zsmax.values
    wet = ~np.isnan(vals)
    if np.sum(wet) < 1:
        logger.warning("No wet cells found."); return

    geo = _open_dem_geometry(dep)
    profile = _make_output_profile(geo)
    _create_output_rasters(profile, zsmap_fn=zsmap_fn)

    nrcb = nrmax
    nrbn = int(np.ceil(geo["height"] / nrcb))
    nrbm = int(np.ceil(geo["width"] / nrcb))
    total = nrbn * nrbm
    done = 0

    if indices is not None:
        # ----- Index-COG path: exact cell containment, no interpolation -----
        logger.info(f"Raw quadtree (index-COG): {np.sum(wet)} wet cells")
        nodata_idx = 2147483647
        indices_src = rasterio.open(str(indices))

        for ii in range(nrbm):
            bm0 = ii * nrcb; bm1 = min(bm0 + nrcb, geo["width"])
            for jj in range(nrbn):
                bn0 = jj * nrcb; bn1 = min(bn0 + nrcb, geo["height"])
                window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

                idx_block = indices_src.read(1, window=window)
                zs_block = np.full(idx_block.shape, np.nan, dtype=np.float32)
                valid = idx_block != nodata_idx
                zs_block[valid] = vals[idx_block[valid]]

                with rasterio.open(str(zsmap_fn), "r+") as dst:
                    dst.write(zs_block, window=window, indexes=1)

                done += 1
                if done % 25 == 0 or done == total:
                    logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

        indices_src.close()
    else:
        # ----- Fallback: NearestNDInterpolator (legacy behaviour) -----------
        from scipy.interpolate import NearestNDInterpolator

        grid = zsmax.ugrid.grid
        face_x, face_y = grid.face_coordinates.T
        interpolator = NearestNDInterpolator(
            np.column_stack([face_x[wet], face_y[wet]]), vals[wet],
        )
        logger.info(f"Raw quadtree (nearest-interp fallback): {np.sum(wet)} wet cells")

        for ii in range(nrbm):
            bm0 = ii * nrcb; bm1 = min(bm0 + nrcb, geo["width"])
            for jj in range(nrbn):
                bn0 = jj * nrcb; bn1 = min(bn0 + nrcb, geo["height"])
                window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

                xx, yy = np.meshgrid(
                    geo["transform"][2] + (np.arange(bm0, bm1) + 0.5) * geo["dx"],
                    geo["transform"][5] + (np.arange(bn0, bn1) + 0.5) * geo["dy"],
                )
                zs_block = interpolator(
                    np.column_stack([xx.ravel(), yy.ravel()])
                ).reshape(xx.shape).astype(np.float32)

                with rasterio.open(str(zsmap_fn), "r+") as dst:
                    dst.write(zs_block, window=window, indexes=1)

                done += 1
                if done % 25 == 0 or done == total:
                    logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

    _apply_mask_and_overviews(None, zsmap_fn, gdf_mask, geo, logger)
    logger.info(f"Raw quadtree water level map saved to: {zsmap_fn}")


# =============================================================================
#  Method: constant  (index-COG based, file path)
# =============================================================================

def _downscale_constant(
    zsmax, dep, indices, hmin, gdf_mask,
    floodmap_fn, zsmap_fn, reproj_method, zoom_level, nrmax, logger,
):
    """File-based constant (bathtub) downscaling via _downscale_floodmap_da."""
    if isinstance(floodmap_fn, Path):
        floodmap_fn = str(floodmap_fn)

    # indices validation
    if indices is not None:
        if not isinstance(indices, (str, Path)):
            raise ValueError("indices should be str/Path when dep is str/Path.")

    if zoom_level is not None:
        zls_dict, crs = RasterioDriver._get_zoom_levels_and_crs(dep)
        overview_level = zoom_to_overview_level(
            zoom=zoom_level, zls_dict=zls_dict, source_crs=crs,
        )
        if overview_level:
            overview_level -= 1
        else:
            overview_level = None
    else:
        overview_level = None

    _open_kwargs = {"overview_level": overview_level} if overview_level is not None else {}
    with rasterio.open(dep, **_open_kwargs) as src:
        if indices is not None:
            indices_src = rasterio.open(indices, **_open_kwargs)

        n1, m1 = src.shape
        nrcb = nrmax
        nrbn = int(np.ceil(n1 / nrcb))
        nrbm = int(np.ceil(m1 / nrcb))

        merge_last_col = m1 % nrcb == 1
        merge_last_row = n1 % nrcb == 1
        if merge_last_col:
            nrbm -= 1
        if merge_last_row:
            nrbn -= 1

        profile = dict(
            driver="GTiff", width=src.width, height=src.height,
            count=1, dtype=np.float32, crs=src.crs, transform=src.transform,
            tiled=True, blockxsize=256, blockysize=256,
            compress="deflate", predictor=2, profile="COG",
            nodata=np.nan, BIGTIFF="YES",
        )
        with rasterio.open(floodmap_fn, "w", **profile):
            pass
        if zsmap_fn is not None:
            with rasterio.open(zsmap_fn, "w", **profile):
                pass

        total = nrbm * nrbn
        done = 0
        skipped = 0
        logger.info(
            f"Constant WSE: {total} blocks to process "
            f"({m1}x{n1} pixels, block size {nrcb})"
        )

        for ii in range(nrbm):
            bm0 = ii * nrcb
            bm1 = min(bm0 + nrcb, m1)
            if merge_last_col and ii == (nrbm - 1):
                bm1 += 1
            for jj in range(nrbn):
                bn0 = jj * nrcb
                bn1 = min(bn0 + nrcb, n1)
                if merge_last_row and jj == (nrbn - 1):
                    bn1 += 1

                window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

                # Read indices first — skip block early if no SFINCS cells
                if indices is not None:
                    block_idx = indices_src.read(window=window)
                    if np.all(block_idx == indices_src.nodata):
                        done += 1
                        skipped += 1
                        continue

                block_data = src.read(window=window)
                if np.all(np.isnan(block_data)):
                    done += 1
                    skipped += 1
                    continue

                if src.transform[1] == 0 and src.transform[3] == 0:
                    x_coords = src.transform[2] + (np.arange(bm0, bm1) + 0.5) * src.transform[0]
                    y_coords = src.transform[5] + (np.arange(bn0, bn1) + 0.5) * src.transform[4]
                    block_dep = xr.DataArray(
                        block_data.squeeze(), dims=("y", "x"),
                        coords={"y": ("y", y_coords), "x": ("x", x_coords)},
                    )
                    if indices is not None:
                        block_idx = xr.DataArray(
                            block_idx.squeeze(), dims=("y", "x"),
                            coords={"y": ("y", y_coords), "x": ("x", x_coords)},
                        )
                else:
                    cols, rows = np.meshgrid(np.arange(bm0, bm1), np.arange(bn0, bn1))
                    xc, yc = src.transform * (cols + 0.5, rows + 0.5)
                    block_dep = xr.DataArray(
                        block_data.squeeze(), dims=("y", "x"),
                        coords={"yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
                    )
                    if indices is not None:
                        block_idx = xr.DataArray(
                            block_idx.squeeze(), dims=("y", "x"),
                            coords={"yc": (("y", "x"), yc), "xc": (("y", "x"), xc)},
                        )

                block_dep.raster.set_crs(src.crs.to_epsg())
                if indices is not None:
                    block_idx.raster.set_nodata(int(indices_src.nodata))
                    block_idx.raster.set_crs(indices_src.crs.to_epsg())

                block_hmax = _downscale_floodmap_da(
                    zsmax=zsmax, dep=block_dep,
                    indices=block_idx if indices is not None else None,
                    hmin=hmin, gdf_mask=gdf_mask, reproj_method=reproj_method,
                )

                with rasterio.open(floodmap_fn, "r+") as fm:
                    fm.write(block_hmax.values, window=window, indexes=1)
                if zsmap_fn is not None:
                    block_zs = (block_hmax + block_dep).astype(np.float32)
                    block_zs = block_zs.where(~np.isnan(block_hmax))
                    with rasterio.open(zsmap_fn, "r+") as zs:
                        zs.write(block_zs.values, window=window, indexes=1)

                done += 1
                if done % 25 == 0 or done == total:
                    logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

        if skipped:
            logger.info(f"  Skipped {skipped}/{total} empty blocks")

    build_overviews(fn=floodmap_fn, resample_method="nearest", logger=logger)
    if zsmap_fn is not None:
        build_overviews(fn=zsmap_fn, resample_method="nearest", logger=logger)


# =============================================================================
#  Method: bilinear  (LinearNDInterpolator, block-based)
# =============================================================================

def _downscale_bilinear(zsmax, dep, hmin, gdf_mask, floodmap_fn, zsmap_fn, nrmax, logger,
                         indices=None, qmax=None, zb=None, q_threshold=0.01, q_scale=0.5):
    from scipy.interpolate import LinearNDInterpolator

    grid = zsmax.ugrid.grid
    face_x, face_y = grid.face_coordinates.T
    vals = zsmax.values
    if np.sum(~np.isnan(vals)) < 3:
        logger.warning("Fewer than 3 wet cells; cannot interpolate."); return

    if qmax is not None:
        # qmax is face-based (one value per cell, same shape as zsmax)
        q_cell_max = np.abs(qmax.values).astype(np.float64)  # (n_faces,)

        # Step 2 — local Bernoulli: H_local = zsmax + (q/h)²/2g
        h_cell = np.maximum(vals - zb.values, hmin) if zb is not None else np.full(len(vals), hmin)
        vel_head = q_cell_max**2 / (h_cell**2 * 2.0 * 9.81)
        H_local = np.where(~np.isnan(vals), vals + vel_head, np.nan)

        # Step 3 — upstream energy propagation via face-face adjacency
        # edge_face_connectivity may raise IndexError on malformed quadtree grids
        # where the internal invert_dense call fails; fall back to KD-tree in that case.
        _ef_ok = False
        try:
            ef = grid.edge_face_connectivity
            _ef_ok = ef.ndim == 2 and ef.shape[1] >= 2
        except (IndexError, ValueError):
            pass
        if _ef_ok:
            both_valid = (ef[:, 0] >= 0) & (ef[:, 1] >= 0)
            f0_safe = np.where(ef[:, 0] >= 0, ef[:, 0], 0)
            f1_safe = np.where(ef[:, 1] >= 0, ef[:, 1], 0)
            edge_q = np.where(both_valid, np.maximum(q_cell_max[f0_safe], q_cell_max[f1_safe]), 0.0)
            active = (edge_q > q_threshold) & both_valid
            ef_f0 = ef[active, 0]
            ef_f1 = ef[active, 1]
        else:
            # Malformed connectivity — rebuild face pairs spatially via KD-tree
            from scipy.spatial import cKDTree as _cKDTree
            fb = grid.face_bounds          # (n_face, 4): xmin, ymin, xmax, ymax
            _dx = fb[:, 2] - fb[:, 0]
            _dy = fb[:, 3] - fb[:, 1]
            _n_face = len(face_x)
            _max_cell = max(_dx.max(), _dy.max())
            _tree = _cKDTree(np.column_stack([face_x, face_y]))
            _offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
            _probes = np.vstack([
                np.column_stack([face_x + _dx * _offsets[d, 0],
                                 face_y + _dy * _offsets[d, 1]])
                for d in range(4)
            ])
            _dists, _idxs = _tree.query(_probes, k=1, distance_upper_bound=_max_cell + 1.0)
            _src = np.tile(np.arange(_n_face), 4)
            _valid = (_dists < _max_cell + 0.5) & (_src != _idxs)
            _src, _idxs = _src[_valid], _idxs[_valid]
            _keep = _src < _idxs
            _pairs = np.unique(np.column_stack([_src[_keep], _idxs[_keep]]), axis=0)
            _f0_all, _f1_all = _pairs[:, 0], _pairs[:, 1]
            _edge_q_all = np.maximum(q_cell_max[_f0_all], q_cell_max[_f1_all])
            _active = _edge_q_all > q_threshold
            ef_f0 = _f0_all[_active]
            ef_f1 = _f1_all[_active]
            active = _active  # for .any() check below

        upstream_H = np.full(len(vals), np.nan)
        if active.any():
            f0, f1 = ef_f0, ef_f1
            H0, H1 = H_local[f0], H_local[f1]
            fwd = ~np.isnan(H0) & (np.isnan(H1) | (H0 > H1))
            rev = ~np.isnan(H1) & (np.isnan(H0) | (H1 > H0))
            np.fmax.at(upstream_H, f1[fwd], H0[fwd])
            np.fmax.at(upstream_H, f0[rev], H1[rev])

        # Step 4 — blend: H_eff = H_local + blend × max(0, upstream_H − H_local)
        blend = np.minimum(1.0, q_cell_max / q_scale)
        H_eff = H_local.copy()
        boosted = ~np.isnan(upstream_H) & ~np.isnan(H_local)
        H_eff[boosted] += blend[boosted] * np.maximum(0.0, upstream_H[boosted] - H_local[boosted])

        # Also extend to truly dry cells that receive upstream energy (transitional cells)
        dry_with_upstream = np.isnan(vals) & ~np.isnan(upstream_H)
        H_eff[dry_with_upstream] = upstream_H[dry_with_upstream] * blend[dry_with_upstream]

        n_boost = int(np.sum(boosted & (upstream_H > H_local)))
        n_trans = int(np.sum(dry_with_upstream))
        logger.info(f"Bilinear WSE: Bernoulli + upstream energy applied ({n_boost} boosted, {n_trans} transitional cells)")
    else:
        H_eff = vals.copy()

    wet_ext = ~np.isnan(H_eff)
    interpolator = LinearNDInterpolator(
        np.column_stack([face_x[wet_ext], face_y[wet_ext]]), H_eff[wet_ext],
    )
    logger.info(f"Bilinear WSE: interpolant from {np.sum(wet_ext)} cells")

    # Open index COG to mask pixels not belonging to a wet SFINCS cell.
    # Without this, LinearNDInterpolator fills across dry-cell gaps inside
    # the convex hull of wet cell centres.
    indices_src = None
    idx_nodata = None
    if indices is not None:
        indices_src = rasterio.open(str(indices))
        idx_nodata = indices_src.nodata
        logger.info("  Using index COG to mask dry-cell gaps")

    geo = _open_dem_geometry(dep)
    profile = _make_output_profile(geo)
    _create_output_rasters(profile, floodmap_fn, zsmap_fn)

    nrcb = nrmax
    nrbn = int(np.ceil(geo["height"] / nrcb))
    nrbm = int(np.ceil(geo["width"] / nrcb))
    total = nrbn * nrbm
    done = 0

    for ii in range(nrbm):
        bm0 = ii * nrcb; bm1 = min(bm0 + nrcb, geo["width"])
        for jj in range(nrbn):
            bn0 = jj * nrcb; bn1 = min(bn0 + nrcb, geo["height"])
            window = Window(bm0, bn0, bm1 - bm0, bn1 - bn0)

            with rasterio.open(str(dep)) as src:
                dem_block = src.read(1, window=window).astype(np.float64)

            xx, yy = np.meshgrid(
                geo["transform"][2] + (np.arange(bm0, bm1) + 0.5) * geo["dx"],
                geo["transform"][5] + (np.arange(bn0, bn1) + 0.5) * geo["dy"],
            )
            zs_interp = interpolator(
                np.column_stack([xx.ravel(), yy.ravel()])
            ).reshape(dem_block.shape).astype(np.float32)

            # Mask pixels outside any wet SFINCS cell
            if indices_src is not None:
                idx_block = indices_src.read(1, window=window)
                outside = (idx_block == idx_nodata)
                # Also mask pixels whose parent cell is dry (NaN zsmax)
                inside = ~outside
                if inside.any():
                    pidx = idx_block[inside].astype(int)
                    parent_zs = vals[pidx]
                    parent_H  = H_eff[pidx]
                    dry_parent = np.isnan(parent_zs) & np.isnan(parent_H)
                    mask_arr = np.zeros_like(outside)
                    mask_arr[inside] = dry_parent
                    outside |= mask_arr
                zs_interp[outside] = np.nan

            hmax_block = (zs_interp - dem_block).astype(np.float32)
            hmax_block[np.isnan(hmax_block)] = np.nan
            hmax_block[hmax_block <= hmin] = np.nan
            hmax_block[np.isnan(dem_block)] = np.nan
            zs_block = zs_interp.copy()
            zs_block[np.isnan(hmax_block)] = np.nan

            if np.any(~np.isnan(hmax_block)):
                with rasterio.open(str(floodmap_fn), "r+") as dst:
                    dst.write(hmax_block, window=window, indexes=1)
                if zsmap_fn is not None:
                    with rasterio.open(str(zsmap_fn), "r+") as dst:
                        dst.write(zs_block, window=window, indexes=1)

            done += 1
            if done % 25 == 0 or done == total:
                logger.info(f"  Block {done}/{total} ({100*done/total:.0f}%)")

    if indices_src is not None:
        indices_src.close()

    _apply_mask_and_overviews(floodmap_fn, zsmap_fn, gdf_mask, geo, logger)
    logger.info(f"Bilinear WSE floodmap saved to: {floodmap_fn}")


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


def _cell_to_pixel_window(cxmin, cxmax, cymin, cymax, transform, width, height):
    """Convert cell bounding box to DEM pixel window.

    Returns pixel indices such that only pixels whose **center** falls within
    the cell are included.  This avoids counting an extra ring of pixels along
    cell edges (which would inflate volumes by ~20% for typical quadtree cells).

    Parameters
    ----------
    cxmin, cxmax, cymin, cymax : float
        Cell bounding box in map coordinates.
    transform : affine.Affine
        DEM affine transform.
    width, height : int
        DEM dimensions in pixels.

    Returns
    -------
    col0, col1, row0, row1 : int
        Pixel index window [col0:col1, row0:row1).  Returns None when the
        window is empty.
    """
    dx = transform[0]   # positive
    dy = transform[4]   # negative (north-up)

    # Pixel whose center is at: x = origin_x + (col + 0.5) * dx
    # Include pixel if center_x >= cxmin  =>  col >= (cxmin - origin_x) / dx - 0.5
    # Exclude pixel if center_x >= cxmax  =>  col >= (cxmax - origin_x) / dx - 0.5
    col0 = int(np.ceil((cxmin - transform[2]) / dx - 0.5))
    col1 = int(np.ceil((cxmax - transform[2]) / dx - 0.5))

    # y-axis: dy is negative, so row increases as y decreases
    # Pixel center_y = origin_y + (row + 0.5) * dy
    # Include pixel if center_y <= cymax  =>  row >= (cymax - origin_y) / dy - 0.5
    # Exclude pixel if center_y <= cymin  =>  row >= (cymin - origin_y) / dy - 0.5
    row0 = int(np.ceil((cymax - transform[5]) / dy - 0.5))
    row1 = int(np.ceil((cymin - transform[5]) / dy - 0.5))

    # Clip to DEM extent
    col0 = max(0, col0)
    col1 = min(width, col1)
    row0 = max(0, row0)
    row1 = min(height, row1)

    if col1 <= col0 or row1 <= row0:
        return None
    return col0, col1, row0, row1


def _downscale_floodmap_da(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: xr.DataArray,
    indices: xr.DataArray = None,
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

    if indices is None:
        # interpolate zsmax to dep grid
        if isinstance(zsmax, xr.DataArray):
            zsmax = zsmax.raster.reproject_like(dep, method=reproj_method)
        elif isinstance(zsmax, xu.UgridDataArray):
            # if non-rotated grid, use xugrid rasterize_like
            if dep.raster.transform[1] == 0 and dep.raster.transform[3] == 0:
                zsmax = zsmax.ugrid.rasterize_like(dep)
            # if rotated grid, use xugrid regridder
            else:
                # need to convert dep to unstructured to enable xugrid regridder
                uda_dep = xu.UgridDataArray.from_structured(dep, "xc", "yc")
                regridder = xu.CentroidLocatorRegridder(source=zsmax, target=uda_dep)
                result = regridder.regrid(zsmax)
                # map back to structured
                zsmax = dep.copy(data=result.values.reshape(dep.shape))

        zsmax = zsmax.raster.mask_nodata()  # make sure nodata is nan

        # get flood depth
        hmax = (zsmax - dep).astype("float32")
        hmax.raster.set_nodata(np.nan)
    else:
        # make sure index is same shape as dep
        if indices.shape != dep.shape:
            raise ValueError(
                "Indices shape {} does not match dep shape {}.".format(
                    indices.shape, dep.shape
                )
            )

        # Get the no_data value from the indices array
        nan_val_indices = indices.raster.nodata  # indices.attrs["_FillValue"]
        # Set the no_data mask
        no_data_mask = indices == nan_val_indices

        # Turn indices into numpy array and set no_data values to 0
        indices = np.squeeze(indices.values[:])
        indices[np.where(indices == nan_val_indices)] = 0

        zsmax = zsmax.raster.mask_nodata()  # make sure nodata is nan

        # Compute water depth
        zs_numpy = zsmax.values[:].flatten()
        h = zs_numpy[indices] - dep.values[:]

        # Set water depth to NaN where indices are no data
        h[no_data_mask] = np.nan

        # Turn h into a DataArray with the same dimensions as zb
        # ds = xr.Dataset()
        hmax = xr.DataArray(h, dims=["y", "x"], coords={"y": dep.y, "x": dep.x})
        hmax.raster.set_nodata(np.nan)
        hmax.raster.set_crs(dep.raster.crs)

    # mask floodmap
    hmax = hmax.where(hmax > hmin)

    if gdf_mask is not None:
        mask = hmax.raster.geometry_mask(gdf_mask, all_touched=True)
        hmax = hmax.where(mask)

    return hmax


def compute_flow_connected_mask(
    zsmax: xu.UgridDataArray,
    sfincs_nc: Union[Path, str],
    hmin: float = 0.05,
    zs_tol: float = 0.01,
    strict_downhill: bool = False,
    logger=logger,
):
    """Compute a boolean mask of cells reachable from boundary/source cells
    via the quadtree neighbor connectivity.

    Uses the neighbor connectivity arrays (mu1/md1/nu1/nd1) from sfincs.nc.
    Starting from boundary cells (mask == 2 or 3), a BFS flood fill propagates
    to neighboring wet cells.

    Two modes are supported:

    * **strict_downhill=False** (default): any wet neighbor is considered
      reachable.  This removes truly isolated wet cells that have no
      wet-neighbor path to the boundary — the most common artifact from
      bilinear interpolation.

    * **strict_downhill=True**: only propagate to neighbors whose WSE is at
      most ``zs_tol`` higher than the current cell.  This enforces a
      physical flow-direction constraint but may be too strict when using
      ``zsmax`` (maximum over time), because different cells may have
      peaked at different times.

    Parameters
    ----------
    zsmax : xu.UgridDataArray
        Maximum water level (m) on the quadtree mesh.
    sfincs_nc : Path or str
        Path to the SFINCS model definition file (sfincs.nc) containing
        neighbor connectivity and mask arrays.
    hmin : float, optional
        Minimum water depth threshold, by default 0.05.
    zs_tol : float, optional
        Tolerance (m) for the downhill criterion.  A neighbor with
        ``zs[nb] <= zs[ci] + zs_tol`` is considered downstream.
        Only used when *strict_downhill* is True.  Default 0.01 m.
    strict_downhill : bool, optional
        If True, only propagate to neighbors with lower (or near-equal) WSE.
        If False, propagate to any wet neighbor.  Default False.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    reachable : np.ndarray of bool, shape (n_faces,)
        True for cells that are connected to boundary sources.
    """
    from collections import deque

    # --- Reduce time dimension ---
    timedim = set(zsmax.dims) - set(zsmax.ugrid.grid.dims)
    if timedim:
        zsmax = zsmax.max(timedim)
    zs = zsmax.values
    n_faces = len(zs)

    # --- Read neighbor connectivity from sfincs.nc ---
    uds = xu.open_dataset(str(sfincs_nc))
    mask_arr = uds["mask"].values if "mask" in uds else np.ones(n_faces, dtype=int)

    # Neighbor arrays: 1-based indices, 0 = no neighbor
    nb_keys = []
    neighbors = {}
    for name in ["mu1", "md1", "nu1", "nd1", "mu2", "md2", "nu2", "nd2"]:
        if name in uds:
            arr = uds[name].values.astype(np.int64) - 1  # 0-based; original 0 -> -1
            neighbors[name] = arr
            nb_keys.append(name)

    # --- Identify source cells ---
    wet = ~np.isnan(zs) & (zs > -999)
    is_boundary = (mask_arr == 2) | (mask_arr == 3)
    sources = np.where(wet & is_boundary)[0]

    if len(sources) == 0:
        # Fallback: cells with no upstream wet neighbor
        logger.info("No boundary cells found; using local-maximum WSE cells.")
        is_source = np.zeros(n_faces, dtype=bool)
        for i in range(n_faces):
            if not wet[i]:
                continue
            has_upstream = False
            for key in nb_keys:
                nb = neighbors[key][i]
                if 0 <= nb < n_faces and wet[nb] and zs[nb] > zs[i] + zs_tol:
                    has_upstream = True
                    break
            if not has_upstream:
                is_source[i] = True
        sources = np.where(is_source)[0]

    logger.info(
        f"Flow connectivity: {len(sources)} source cells, "
        f"{np.sum(wet)} wet cells, strict_downhill={strict_downhill}"
    )

    # --- BFS from sources ---
    reachable = np.zeros(n_faces, dtype=bool)
    queue = deque()
    for s in sources:
        reachable[s] = True
        queue.append(s)

    while queue:
        ci = queue.popleft()
        for key in nb_keys:
            nb = neighbors[key][ci]
            if nb < 0 or nb >= n_faces:
                continue
            if reachable[nb] or not wet[nb]:
                continue
            if strict_downhill and zs[nb] > zs[ci] + zs_tol:
                continue  # skip uphill neighbors beyond tolerance
            reachable[nb] = True
            queue.append(nb)

    n_reachable = np.sum(reachable)
    n_disconnected = np.sum(wet) - n_reachable
    logger.info(
        f"  {n_reachable} cells reachable, "
        f"{n_disconnected} wet cells disconnected "
        f"({100*n_disconnected/max(1, np.sum(wet)):.1f}%)"
    )

    return reachable


def remove_disconnected_flooding(
    depth_fn: Union[Path, str],
    bnd_fn: Union[Path, str],
    hmin: float = 0.02,
    connection_fn: Union[Path, str] = None,
    output_fns: dict = None,
    logger=logger,
):
    """Remove disconnected flooding from a downscaled depth raster.

    Identifies wet pixels reachable from SFINCS boundary points via BFS
    flood-fill (8-connectivity), then masks pixels that are wet but
    unreachable.  This replaces the need for a manually drawn source
    polygon (as in the legacy Part-3 workflow).

    Uses a vectorised level-set BFS instead of ``scipy.ndimage.label``
    to avoid allocating a full int64 label array, which can exceed
    available memory for large high-resolution domains.

    Parameters
    ----------
    depth_fn : Path or str
        Path to the downscaled flood-depth GeoTIFF.
    bnd_fn : Path or str
        Path to the SFINCS boundary point file (``sfincs.bnd``).
        Each point marks a location where water enters the domain.
    hmin : float, optional
        Minimum water depth (m) to be considered wet, by default 0.02.
    connection_fn : Path or str, optional
        If provided, a connection-mask GeoTIFF is written here with values:
        0 = dry, 1 = connected to boundary, 2 = disconnected (wet but
        unreachable from any boundary point).
    output_fns : dict, optional
        Dictionary of ``{input_fn: output_fn}`` pairs.  For each pair the
        input raster is read, pixels where ``connection != 1`` are set to
        NaN, and the result is written to *output_fn*.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    None
    """
    # --- 1. Get raster metadata without loading the full array ---------------
    with rasterio.open(str(depth_fn)) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    # --- 2. Build wet mask tile-by-tile (avoids holding full depth) ----------
    wet = np.empty((height, width), dtype=np.bool_)
    with rasterio.open(str(depth_fn)) as src:
        for _, window in src.block_windows(1):
            r0, c0 = window.row_off, window.col_off
            block = src.read(1, window=window)
            wet[r0 : r0 + window.height, c0 : c0 + window.width] = block > hmin

    n_wet = int(np.sum(wet))
    logger.info(
        f"Disconnected-flooding removal: {n_wet} wet pixels "
        f"(hmin={hmin} m)"
    )

    # --- 3. Read boundary points and map to pixel coordinates ----------------
    bnd_gdf = read_xy(str(bnd_fn), crs=crs)
    bnd_x = np.array([p.x for p in bnd_gdf.geometry])
    bnd_y = np.array([p.y for p in bnd_gdf.geometry])

    inv_transform = ~transform
    bnd_col, bnd_row = inv_transform * (bnd_x, bnd_y)
    bnd_row = np.round(bnd_row).astype(int)
    bnd_col = np.round(bnd_col).astype(int)

    # --- 4. BFS flood-fill from boundary points (8-connectivity) -------------
    connected = np.zeros((height, width), dtype=np.bool_)
    seed_indices = []
    n_bnd_wet = 0

    for r, c in zip(bnd_row, bnd_col):
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    if wet[rr, cc] and not connected[rr, cc]:
                        connected[rr, cc] = True
                        seed_indices.append(rr * width + cc)
        if 0 <= r < height and 0 <= c < width and wet[r, c]:
            n_bnd_wet += 1

    logger.info(
        f"  {len(bnd_gdf)} boundary points, {n_bnd_wet} on wet pixels, "
        f"{len(seed_indices)} seed pixels for BFS"
    )

    # Vectorised level-set BFS: expand frontier one ring at a time using numpy
    _neighbors = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    wet_flat = wet.ravel()
    conn_flat = connected.ravel()
    frontier = np.array(seed_indices, dtype=np.intp)

    iteration = 0
    total_processed = len(frontier)
    while len(frontier) > 0:
        iteration += 1

        fr = frontier // width
        fc = frontier % width

        new_seeds = []
        for dr, dc in _neighbors:
            nr = fr + dr
            nc = fc + dc
            valid = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
            flat_idx = nr[valid] * width + nc[valid]
            mask = wet_flat[flat_idx] & ~conn_flat[flat_idx]
            new_pixels = flat_idx[mask]
            if len(new_pixels) > 0:
                conn_flat[new_pixels] = True
                new_seeds.append(new_pixels)

        if new_seeds:
            frontier = np.unique(np.concatenate(new_seeds))
            total_processed += len(frontier)
        else:
            frontier = np.array([], dtype=np.intp)

        if iteration % 500 == 0:
            logger.info(
                f"  BFS iteration {iteration}: {total_processed:,} pixels "
                f"reached, frontier {len(frontier):,}"
            )

    n_connected = int(np.sum(connected))
    n_disconnected = n_wet - n_connected
    logger.info(
        f"  {n_connected} connected pixels, "
        f"{n_disconnected} disconnected pixels removed "
        f"({100 * n_disconnected / max(1, n_wet):.1f}%)"
    )

    # Free wet mask — connected mask is all we need from here
    del wet, wet_flat

    # --- 5. Write connection mask raster (optional) --------------------------
    if connection_fn is not None:
        conn_profile = dict(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="int32",
            crs=crs,
            transform=transform,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            nodata=0,
        )
        with rasterio.open(str(depth_fn)) as src_dep:
            with rasterio.open(str(connection_fn), "w", **conn_profile) as dst:
                for _, window in dst.block_windows(1):
                    r0 = window.row_off
                    c0 = window.col_off
                    h_blk = window.height
                    w_blk = window.width
                    dep_blk = src_dep.read(1, window=window)
                    wet_blk = dep_blk > hmin
                    conn_blk = connected[r0 : r0 + h_blk, c0 : c0 + w_blk]
                    blk = np.zeros((h_blk, w_blk), dtype=np.int32)
                    blk[wet_blk] = 2
                    blk[conn_blk & wet_blk] = 1
                    dst.write(blk, 1, window=window)
        logger.info(f"  Connection mask written: {connection_fn}")

    # --- 6. Mask additional rasters (optional) -------------------------------
    if output_fns:
        for input_fn, output_fn in output_fns.items():
            with rasterio.open(str(input_fn)) as src_var:
                out_meta = src_var.meta.copy()
                out_meta.update(
                    dtype="float32",
                    nodata=np.nan,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    compress="deflate",
                    predictor=2,
                )
                with rasterio.open(str(output_fn), "w", **out_meta) as dst:
                    for _, window in dst.block_windows(1):
                        r0 = window.row_off
                        c0 = window.col_off
                        var_block = src_var.read(1, window=window)
                        conn_blk = connected[
                            r0 : r0 + window.height, c0 : c0 + window.width
                        ]
                        masked_block = np.where(
                            conn_blk, var_block, np.nan
                        ).astype(np.float32)
                        dst.write(masked_block, 1, window=window)
            logger.info(f"  Masked raster written: {output_fn}")

    return None


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


def make_regular_grid(
    x0,
    y0,
    dx,
    dy,
    mmax,
    nmax,
    rotation=0.0,
    crs=None,
    mmin=0,
    nmin=0,
    refi=1,
    name="var",
    dtype=float,
    fill_value=np.nan,
    uv_points=False,
    make_ugrid=False,
):
    """
    Create an xarray.DataArray with spatial coordinates based on grid definition.

    Parameters
    ----------
    x0, y0 : float
        Origin (lower-left corner) in physical coordinates.
    dx, dy : float
        Grid spacing in x and y directions (coarse resolution).
    mmin, mmax, nmin, nmax : int
        Grid index bounds.
    refi : int
        Refinement factor (number of subcells per coarse cell).
    rotation : float
        Rotation angle in degrees.
    uv_points : bool, optional
        If True, place points at cell corners (UV points);
        if False, place points at cell centers (default).
    """

    # Refined spacing
    dxp, dyp = dx / refi, dy / refi

    # Number of points; 1 coarse cell extra for uv_points
    nx = (mmax - mmin + int(uv_points)) * refi
    ny = (nmax - nmin + int(uv_points)) * refi

    # Index ranges
    m_range = np.arange(nx)
    n_range = np.arange(ny)

    # Offset in grid units
    offset_x = 0.5 * dxp + mmin * dx - (0.5 * dx if uv_points else 0)
    offset_y = 0.5 * dyp + nmin * dy - (0.5 * dy if uv_points else 0)

    # Affine transform
    transform = (
        Affine.translation(x0, y0) * Affine.rotation(rotation) * Affine.scale(dxp, dyp)
    )

    # Generate coordinates
    if transform.b == 0.0:  # No rotation → rectilinear
        x_coords, _ = transform * (
            m_range + offset_x / dxp,
            np.zeros(nx) + offset_y / dyp,
        )
        _, y_coords = transform * (
            np.zeros(ny) + offset_x / dxp,
            n_range + offset_y / dyp,
        )
        coords = {
            "m": ("x", m_range + mmin * refi),
            "n": ("y", n_range + nmin * refi),
            "x": x_coords,
            "y": y_coords,
        }
        dims = ("y", "x")
    else:  # With rotation → 2D coordinate arrays
        m_mesh, n_mesh = np.meshgrid(m_range, n_range)
        x_coords, y_coords = transform * (
            m_mesh + offset_x / dxp,
            n_mesh + offset_y / dyp,
        )
        coords = {
            "m": ("x", m_range + mmin * refi),
            "n": ("y", n_range + nmin * refi),
            "xc": (("y", "x"), x_coords),
            "yc": (("y", "x"), y_coords),
        }
        dims = ("y", "x")

    # DataArray with fill value
    data = np.full((ny, nx), fill_value, dtype=dtype)
    da = xr.DataArray(data, dims=dims, coords=coords, name=name)

    # CRS/Ugrid handling
    if make_ugrid:
        if rotation != 0.0:
            da = UgridDataArray.from_structured(da, "xc", "yc")
        else:
            da = UgridDataArray.from_structured(da)
        if crs is not None:
            da.grid.set_crs(crs)
    else:
        if crs is not None:
            da.raster.set_crs(crs)

    return da


def make_regular_grid_transform(
    x0, y0, dx, dy, mmax, nmax, mmin=0, nmin=0, rotation=0.0, refi=1, uv_points=False
):
    """
    Compute affine transform for a regular grid
    (possibly rotated) without allocating arrays. The affine corresponds
    to the **bottom-left corner** of the first pixel (raster convention),
    while preserving original UV offset logic.
    """
    dxp = dx / refi
    dyp = dy / refi

    if uv_points:
        width = (mmax - mmin + 1) * refi
        height = (nmax - nmin + 1) * refi
    else:
        width = (mmax - mmin) * refi
        height = (nmax - nmin) * refi

    # offset in pixel units like make_regular_grid
    if uv_points:
        offset_x = 0.5 * dxp + mmin * dx - 0.5 * dx
        offset_y = 0.5 * dyp + nmin * dy - 0.5 * dy
    else:
        offset_x = 0.5 * dxp + mmin * dx
        offset_y = 0.5 * dyp + nmin * dy

    if rotation == 0.0:
        # non-rotated: simple translation
        tx = x0 + offset_x - 0.5 * dxp
        ty = y0 + offset_y - 0.5 * dyp
        transform = Affine.translation(tx, ty) * Affine.scale(dxp, dyp)
    else:
        # rotated: compute cos/sin once
        theta = np.deg2rad(rotation)
        cosrot = np.cos(theta)
        sinrot = np.sin(theta)

        # apply the half-cell shift in rotated coordinates
        x0_shifted = (
            x0 + (offset_x - 0.5 * dxp) * cosrot - (offset_y - 0.5 * dyp) * sinrot
        )
        y0_shifted = (
            y0 + (offset_x - 0.5 * dxp) * sinrot + (offset_y - 0.5 * dyp) * cosrot
        )

        # base affine for rotation and scaling
        transform = (
            Affine.translation(x0_shifted, y0_shifted)
            * Affine.rotation(rotation)
            * Affine.scale(dxp, dyp)
        )

    return transform, width, height


def partition_quadtree(
    quadtree: xu.UgridDataset,
    partition_by_level: bool = True,
    partition_in_blocks: bool = True,
    nrmax: int = 2000,
    logger=logger,
):
    """Partition a 2D unstructured grid into blocks.

    Parameters
    ----------
    quadtree : xu.UgridDataset
        Unstructured 2D grid.
    partition_by_level : bool, optional
        Partition by level, by default True
    partition_in_blocks : bool, optional
        Partition in blocks, by default False
    nrmax : int, optional
        Maximum number of cells per block, by default 2000

    Returns
    -------
    Partitions : List[xu.UgridDataset]
        List of partitiones, by levels, in spatial blocks, or both.
    """

    if partition_by_level:
        if "level" not in quadtree:
            raise ValueError("No 'level' attribute found in quadtree.")
        partitions = quadtree.ugrid.partition_by_label(quadtree["level"] - 1)
    else:
        partitions = [quadtree]

    partitions_new = []
    if partition_in_blocks:
        for level, partition in enumerate(partitions):
            if len(partition.coords["mesh2d_nFaces"]) > 0:
                logger.debug(
                    f"Partition level {level} has {len(partition.coords['mesh2d_nFaces'])} faces: "
                )

                # approximate nr of cells in x and y direction based on resolution (to prevent too large datasets loaded in memory)
                dx = partition.dx / (2**level)
                dy = partition.dy / (2**level)
                logger.debug(f"dx, dy:  {dx}, {dy}")
                xmin, ymin, xmax, ymax = partition.ugrid.grid.bounds
                nmax = int(np.ceil((ymax - ymin) / dy))
                mmax = int(np.ceil((xmax - xmin) / dx))
                logger.debug(f"mmax: {mmax}, nmax: {nmax}")

                # check if partition is too large and split in smaller blocks
                nrbn = int(np.ceil(nmax / nrmax))  # nr of blocks in n direction
                nrbm = int(np.ceil(mmax / nrmax))  # nr of blocks in m direction

                # if too large, split on spatial extent (so not the traditional partitions ...)
                if nrbn > 1 or nrbm > 1:
                    logger.debug(
                        f"Partition level {level} is too large, splitting in {nrbn} x {nrbm} blocks"
                    )
                    # Create coordinate ranges for slicing
                    x_edges = np.linspace(xmin, xmax, nrbm + 1)
                    y_edges = np.linspace(ymin, ymax, nrbn + 1)
                    # Generate all index pairs in a vectorized manner
                    index_pairs = np.array(
                        np.meshgrid(np.arange(nrbm), np.arange(nrbn))
                    ).T.reshape(-1, 2)
                    # Use a list comprehension with vectorized index pairs
                    subsets = [
                        partition.ugrid.sel(
                            x=slice(x_edges[ii], x_edges[ii + 1]),
                            y=slice(y_edges[jj], y_edges[jj + 1]),
                        )
                        for ii, jj in index_pairs
                    ]
                    for subset in subsets:
                        if len(subset.coords["mesh2d_nFaces"]) > 0:
                            # subset.level = level
                            partitions_new.append(subset)
                else:
                    # partition.level = level
                    partitions_new.append(partition)

    if len(partitions_new) > 0:
        return partitions_new
    else:
        return partitions


def write_netcdf_safely(ds, abs_file_path: Path, encoding=None) -> Path:
    """
    NetCDF files have the tendency to get locked by other processes (e.g. other notebooks), or because they were
    opened in a lazy manner. This function attempts to write the dataset to the specified path,
    only when it actually changed, and if the file is locked, it will create a versioned file instead.

    Parameters
    ----------
    ds : xarray.Dataset or GeoDataset
        Dataset to write (should already have CRS if needed).
    abs_file_path : Path
        Absolute target path for the NetCDF file.
    encoding: dict, optional
        Encoding dictionary passed to xarray.to_netcdf, here for instance used for time variable;
        e.g. encoding = dict(time={"units": f"minutes since {tref_str}", "dtype": "float64"}))

    Returns
    -------
    Path
        The path the dataset was actually written to (may be versioned if original locked).
    """
    ds = ds.load()  # ensure fully in memory

    # Step 1: skip if file exists and dataset is unchanged
    if abs_file_path.exists():
        try:
            existing_ds = GeoDataset.from_netcdf(
                abs_file_path, crs=ds.raster.crs, chunks="auto"
            )
            changed = not ds.equals(existing_ds)
            existing_ds.close()
        except Exception:
            changed = True  # fail-safe

        if not changed:
            logger.info(f"No changes detected; skipping write to {abs_file_path}")
            return abs_file_path

    # Step 2: remove cryptic encoding per variable
    for var in ds.data_vars:
        ds[var].encoding.clear()  # remove all encoding hints

    # Step 3: write to temporary file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".nc", dir=abs_file_path.parent)
    os.close(tmp_fd)
    ds.vector.to_xy().to_netcdf(tmp_path, encoding=encoding)

    # Step 4: move temp file to target, or create versioned file if locked
    try:
        shutil.move(tmp_path, abs_file_path)
        final_path = abs_file_path
    except PermissionError:
        # File is locked — create versioned file
        base, ext, parent = (
            abs_file_path.stem,
            abs_file_path.suffix,
            abs_file_path.parent,
        )
        i = 1
        while True:
            versioned_path = parent / f"{base}_v{i}{ext}"
            if not versioned_path.exists():
                break
            i += 1
        shutil.move(tmp_path, versioned_path)
        logger.warning(f"Original file locked. Saved new version as {versioned_path}")
        final_path = versioned_path

    return final_path
