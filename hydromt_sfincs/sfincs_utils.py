import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from typing import List, Dict, Union
from pathlib import Path
import io
import copy

__all__ = ["gdf2structures", "structures2gdf", "write_structures", "read_structures"]


def gdf2structures(gdf: gpd.GeoDataFrame) -> List[Dict]:
    """Convert GeoDataFrame[LineString] to list of structure dictionaries

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        GeoDataFrame structures

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
        if line.type == "MultiLineString" and len(line) == 1:
            line = line[0]
        if line.type != "LineString":
            raise ValueError("Invalid geometry type, only LineString is accepted.")
        xyz = tuple(zip(*line.coords[:]))
        feat["x"], feat["y"] = list(xyz[0]), list(xyz[1])
        if len(xyz) == 3:
            feat["z"] = list(xyz[2])
        feats.append(feat)
    return feats


def structures2gdf(feats: List[Dict], crs=None) -> gpd.GeoDataFrame:
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


def write_structures(
    fn: Union[str, Path], feats: List[Dict], stype: str = "thd", fmt="%.1f"
) -> None:
    """Write list of structure dictionaries to file

    Parameters
    ----------
    fn: str, Path
        Path to output structure file.
    feats: list of dict
        List of dictionaries describing structures.
        For thd files "x" and "y" are required, "name" is optional.
        For weir files "x", "y" and "z" are required, "name" and "par1" are optional.
    stype: {'thd', 'weir'}
        Structure type thin dams (thd) or weirs (weir).
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
    >>> write_structures(feats, stype='weir')
    """
    cols = {"thd": 2, "weir": 4}[stype.lower()]
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


def read_structures(fn: Union[str, Path]) -> List[Dict]:
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
