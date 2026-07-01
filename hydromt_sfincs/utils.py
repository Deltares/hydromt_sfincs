"""
HydroMT-SFINCS utilities functions for reading and writing SFINCS specific input and output files,
as well as some common data conversions.
"""

import copy
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from affine import Affine
import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
import xugrid as xu
from xugrid.core.wrap import UgridDataArray
from pyproj.crs.crs import CRS
from rasterio.enums import Resampling
from rasterio.rio.overview import get_maximum_overview_level
from rasterio.windows import Window
from shapely.geometry import LineString, Polygon

from hydromt.data_catalog.drivers import RasterioDriver
from hydromt.gis.gis_utils import zoom_to_overview_level


__all__ = [
    "get_bounds_vector",
    "create_boundary_points",
    "mask2gdf",
    "gdf2linestring",
    "gdf2polygon",
    "linestring2gdf",
    "polygon2gdf",
    "downscale_floodmap",
    "rotated_grid",
    "build_overviews",
    "find_uv_indices",
    "make_regular_grid",
    "make_regular_grid_transform",
    "partition_quadtree",
]

logger = logging.getLogger(f"hydromt.{__name__}")


def parse_datetime(dt: Union[str, datetime], format="%Y%m%d %H%M%S") -> datetime:
    """Checks and/or parses datetime from a string, default sfincs datetime string format"""
    if isinstance(dt, str):
        dt = datetime.strptime(dt, format)
    elif not isinstance(dt, datetime):
        raise ValueError(f"Unknown type for datetime: {type(dt)})")
    return dt


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
    # check if da_msk has values greater than 1, if not raise error
    if da_msk.max() <= 1:
        raise ValueError(
            "The mask should have values greater than 1 to determine boundary cells."
        )

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


def create_boundary_points(gdf_lines, bnd_dist, method="normalized", crs=None):
    """
    Generate points along line geometries in a GeoDataFrame.

    Parameters
    ----------
    gdf_lines : GeoDataFrame
        GeoDataFrame containing line geometries.
    bnd_dist : float
        Distance between points (for 'absolute') or approximate segment length (for 'normalized').
    method : str, optional
        'absolute' for fixed-distance spacing,
        'normalized' for equal fraction spacing along each line.
    crs : CRS, optional
        Coordinate reference system for the output GeoDataFrame.

    Returns
    -------
    GeoDataFrame
        Points along the input line geometries.
    """
    points = []

    for _, row in gdf_lines.iterrows():
        line = row.geometry

        if method == "absolute":
            distances = np.arange(0, line.length + bnd_dist, bnd_dist)
            for d in distances:
                d = min(d, line.length)
                pt = line.interpolate(d)
                points.append((pt.x, pt.y))
        elif method == "normalized":
            num_points = int(line.length / bnd_dist) + 2
            for i in range(num_points):
                t = i / float(num_points - 1)
                pt = line.interpolate(t, normalized=True)
                points.append((pt.x, pt.y))
        else:
            raise ValueError(f"Unknown method: {method}")

    gdf_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*points)), crs=crs)
    return gdf_points


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


def gdf2linestring(gdf: gpd.GeoDataFrame) -> List[Dict]:
    """Convert GeoDataFrame[LineString] to list of structure dictionaries

    The x,y are taken from the geometry.
    For weir structures to additional paramters are required, a "elevation" (elevation) and
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
            feat["elevation"] = list(xyz[2])
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
        xy = [feat.pop("x"), feat.pop("y")]
        feat.update({"geometry": LineString(list(zip(*xy)))})
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


def downscale_floodmap(
    zsmax: Union[xr.DataArray, xu.UgridDataArray],
    dep: Union[Path, str, xr.DataArray],
    indices: Union[Path, str, xr.DataArray] = None,
    hmin: float = 0.05,
    gdf_mask: gpd.GeoDataFrame = None,
    floodmap_fn: Union[Path, str] = None,
    reproj_method: str = "nearest",
    zoom_level: Optional[Union[int, tuple]] = None,
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
    indices: Path, str, xr.DataArray, optional
        Indices of the corresponding SFINCS cells to the DEM cells.
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
    zoom_level : int, tuple, optional
        Overview level of the raster dataset (0 is highest resolution), if present.
        Using a tuple the zoom level can be specified as (<zoom_resolution>, <unit>), e.g., (1000, 'meter')
        Note, this only works when dep is a Path or str.
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
    if isinstance(zsmax, xu.UgridDataArray):
        timedim = set(zsmax.dims) - set(zsmax.ugrid.grid.dims)
    else:
        timedim = set(zsmax.dims) - set(zsmax.raster.dims)
    if timedim:
        logger.info(f"Multiple values present in {timedim} dimension.")
        logger.info(f"Downscaling floodmap for maximum water level over {timedim}.")
        zsmax = zsmax.max(timedim)

    # Hydromt expects a string so if a Path is provided, convert to str
    if isinstance(floodmap_fn, Path):
        floodmap_fn = str(floodmap_fn)

    # indices (if provided) should be of the same type as dep
    if indices is not None:
        if isinstance(indices, (str, Path)):
            if not isinstance(dep, (str, Path)):
                raise ValueError(
                    "index should be a xr.DataArray when dep is a xr.DataArray."
                )
        elif isinstance(indices, xr.DataArray):
            if not isinstance(dep, xr.DataArray):
                raise ValueError(
                    "index should be a str or Path when dep is a str or Path."
                )
        else:
            raise ValueError("index should be a str, Path or xr.DataArray.")

    if isinstance(dep, xr.DataArray):
        hmax = _downscale_floodmap_da(
            zsmax=zsmax,
            dep=dep,
            indices=indices,
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
        if floodmap_fn is None:
            raise ValueError(
                "floodmap_fn should be provided when dep is a Path or str."
            )

        if zoom_level is not None:
            zls_dict, crs = RasterioDriver._get_zoom_levels_and_crs(dep)
            overview_level = zoom_to_overview_level(
                zoom=zoom_level, zls_dict=zls_dict, source_crs=crs
            )
            if overview_level:
                # NOTE: overview levels start at zoom_level 1, see _get_zoom_levels_and_crs
                overview_level -= 1
        else:
            # use highest resolution by default
            overview_level = 0

        with rasterio.open(dep, overview_level=overview_level) as src:
            # check if index is provided and open it if it is
            if indices is not None:
                indices_src = rasterio.open(indices, overview_level=overview_level)

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

                    if indices is not None:
                        # Read the corresponding index block
                        block_indices = indices_src.read(window=window)

                    # Determine if rotation is zero
                    if src.transform[1] == 0 and src.transform[3] == 0:  # No rotation
                        # Compute the 1D coordinates for x and y using the affine transformation
                        x_coords = (
                            src.transform[2]
                            + (np.arange(bm0, bm1) + 0.5) * src.transform[0]
                        )
                        y_coords = (
                            src.transform[5]
                            + (np.arange(bn0, bn1) + 0.5) * src.transform[4]
                        )

                        # Create xarray DataArray with coordinates
                        block_dep = xr.DataArray(
                            block_data.squeeze(),
                            dims=("y", "x"),
                            coords={
                                "y": ("y", y_coords),
                                "x": ("x", x_coords),
                            },
                        )
                        # create xarray DataArray with coordinates for index
                        if indices is not None:
                            block_indices = xr.DataArray(
                                block_indices.squeeze(),
                                dims=("y", "x"),
                                coords={
                                    "y": ("y", y_coords),
                                    "x": ("x", x_coords),
                                },
                            )
                    else:
                        # Convert row and column indices to pixel coordinates
                        cols, rows = np.meshgrid(
                            np.arange(bm0, bm1), np.arange(bn0, bn1)
                        )
                        x_coords, y_coords = src.transform * (cols + 0.5, rows + 0.5)

                        # Create xarray DataArray with coordinates
                        block_dep = xr.DataArray(
                            block_data.squeeze(),
                            dims=("y", "x"),
                            coords={
                                "yc": (("y", "x"), y_coords),
                                "xc": (("y", "x"), x_coords),
                            },
                        )
                        # create xarray DataArray with coordinates for index
                        if indices is not None:
                            block_indices = xr.DataArray(
                                block_indices.squeeze(),
                                dims=("y", "x"),
                                coords={
                                    "yc": (("y", "x"), y_coords),
                                    "xc": (("y", "x"), x_coords),
                                },
                            )

                    # make sure the nodata value and crs are set
                    block_dep.raster.set_crs(src.crs.to_epsg())
                    if indices is not None:
                        block_indices.raster.set_nodata(int(indices_src.nodata))
                        block_indices.raster.set_crs(indices_src.crs.to_epsg())

                    block_hmax = _downscale_floodmap_da(
                        zsmax=zsmax,
                        dep=block_dep,
                        indices=block_indices if indices is not None else None,
                        hmin=hmin,
                        gdf_mask=gdf_mask,
                        reproj_method=reproj_method,
                    )

                    with rasterio.open(floodmap_fn, "r+") as fm_tif:
                        fm_tif.write(
                            block_hmax.values,
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
                uda_dep = xu.UgridDataArray.from_structured2d(dep, "xc", "yc")
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
            da = UgridDataArray.from_structured2d(da, "xc", "yc")
        else:
            da = UgridDataArray.from_structured2d(da)
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
