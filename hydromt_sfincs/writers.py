"""Standalone writer functions for HydroMT-SFINCS."""

import copy
import io
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt.gis.vector import GeoDataset
from hydromt.writers import write_xy

from hydromt_sfincs.utils import parse_datetime

__all__ = [
    "write_ascii_map",
    "write_bdr",
    "write_binary_map",
    "write_binary_map_index",
    "write_drn",
    "write_geoms",
    "write_netcdf_safely",
    "write_raster",
    "write_timeseries",
    "write_vector",
    "write_xy",
    "write_xyn",
]

logger = logging.getLogger(f"hydromt.{__name__}")


## ASCII maps: sfincs.restart ##
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


## ASCII TIMESERIES: bzs / dis / precip ##
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


## BINARY MAPS: sfincs.ind, sfincs.msk, sfincs.dep etc. ##
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


## STRUCTURES: thd / weir ##
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
                "elevation": 5.0,
                "par1": 0.6,
            },
            {
                "name": 'WEIR02',
                "x": [100, 110, 120],
                "y": [100, 100, 100],
                "elevation": [5.0, 5.1, 5.0],
                "par1": 0.6,
            },
        ]
    >>> write_structures('sfincs.weir', feats, stype='weir')
    """
    cols = {"pli": 2, "pol": 2, "thd": 2, "weir": 4, "crs": 2, "wvm": 2}[stype.lower()]

    fmt = [fmt, fmt] + [fmt_z for _ in range(cols - 2)]
    if stype.lower() == "weir" and np.any(["elevation" not in f for f in feats]):
        raise ValueError('"elevation" value missing for weir files.')
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
                a[:, 2] = feat["elevation"]
                a[:, 3] = feat.get("par1", 0.6)
            s = io.BytesIO()
            np.savetxt(s, a, fmt=fmt)
            f.write(f"{name}\n")
            f.write(f"{rows:d} {cols:d}\n")
            f.write(s.getvalue().decode())


def write_bdr(fn: Union[str, Path], gdf_bdr: gpd.GeoDataFrame, fmt="%.1f") -> None:
    """Write SFINCS downstream river boundary points file (.bdr).

    Each row:
    xbdr ybdr xbdr_in ybdr_in slope distance

    NOTE: This version expects geometry to be LineString with 2 vertices:
    - first vertex: boundary point
    - second vertex: inland control point
    """
    # expected columns for river boundary structures
    gdf = copy.deepcopy(gdf_bdr)
    # get geometry linestring and convert to xsnk, ysnk, xsrc, ysrc
    endpoints = gdf.boundary.explode(index_parts=True).unstack()
    gdf["xbdr"] = endpoints[0].x
    gdf["ybdr"] = endpoints[0].y
    gdf["x_bdr_in"] = endpoints[1].x
    gdf["y_bdr_in"] = endpoints[1].y
    gdf.drop(["geometry"], axis=1, inplace=True)

    # required columns
    required = ["slope", "distance"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns in gdf_bdr: {missing}")

    # order columns as SFINCS expects
    gdf = gdf[["xbdr", "ybdr", "x_bdr_in", "y_bdr_in", "slope", "distance"]]

    # format coords
    for col in ["xbdr", "ybdr", "x_bdr_in", "y_bdr_in"]:
        gdf[col] = gdf[col].apply(lambda x: fmt % float(x))

    gdf["slope"] = gdf["slope"].apply(lambda x: f"{float(x):.6f}")
    gdf["distance"] = gdf["distance"].apply(lambda x: f"{float(x):.3f}")

    Path(fn).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_csv(fn, sep=" ", index=False, header=False)


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
        "par6",
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
    if fmt[0] == "%":
        fmt = fmt[1:]
    with open(fn, "w") as f:
        for _, row in gdf.iterrows():
            f.write(
                f"{row.xsnk:{fmt}} {row.ysnk:{fmt}} {row.xsrc:{fmt}} {row.ysrc:{fmt}} "
                f"{row.type:2.0f} {row.par1:10.3f} {row.par2:10.3f} "
                f"{row.par3:10.3f} {row.par4:10.3f} {row.par5:10.3f} {row.par6:10.3f}\n"
            )
    # gdf.to_csv(fn, sep=" ", index=False, header=False)


## XY files: bnd / src ##
def write_xyn(fn: str = "sfincs.obs", gdf: gpd.GeoDataFrame = None, fmt: str = "%.1f"):
    """Write xyn files, for example observation points with names. When name column is not present, it will be generated as "point001", "point002", etc."""
    # strip %-sign of fmt if present
    fmt = fmt.replace("%", "")

    with open(fn, "w") as fid:
        for point in gdf.iterfeatures():
            # only take first two coordinates if geometry is 3D
            x, y = point["geometry"]["coordinates"][:2]
            if "properties" in point and "name" in point["properties"]:
                name = point["properties"]["name"]
            else:
                name = None
                # name = "point" + str(point["id"])
            if name is not None:
                string = f'{x:{fmt}} {y:{fmt}} "{name}"\n'
            else:
                string = f"{x:{fmt}} {y:{fmt}}\n"
            fid.write(string)


## Generic writers ##
def write_vector(
    data: Union[xr.Dataset, gpd.GeoDataFrame],
    name: str,
    root: Union[str, Path],
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
