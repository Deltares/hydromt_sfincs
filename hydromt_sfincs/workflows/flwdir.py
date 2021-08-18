from re import X
import numpy as np
import xarray as xr
import geopandas as gpd
from scipy import ndimage
import hydromt
import pyflwdir
import logging
from rasterio.transform import from_origin

from .. import gis_utils

logger = logging.getLogger(__name__)

__all__ = [
    "reproject_hydrography",
    # "lddcreate",
    "d8_from_dem",
    "river_inflow_points",
    "river_outflow_points",
]

# NOTE: yields different results at each call with same data !
# def lddcreate(
#     da_elv,
#     outflowdepth=1e99,
#     corevolume=1e99,
#     corearea=1e99,
#     catchmentprecip=1e99,
# ):
#     """Wrapper around pcraster.lddcreate to derive flow direcition from an elevation grid."""
#     import pcraster as pcr

#     # set pcr clone
#     assert da_elv.raster.res[1] < 0
#     nrow, ncol = da_elv.raster.shape
#     res, west, north = np.asarray(da_elv.raster.transform)[[0, 2, 5]]
#     pcr.setclone(nrow, ncol, res, west, north)
#     # derive new flow directions
#     elv_pcr = pcr.numpy2pcr(pcr.Scalar, da_elv.values, da_elv.raster.nodata)
#     ldd_pcr = pcr.lddcreate(
#         elv_pcr, outflowdepth, corevolume, corearea, catchmentprecip
#     )
#     ldd_np = pcr.pcr2numpy(ldd_pcr, 255)
#     # return xarray data array
#     da_out = xr.DataArray(
#         dims=da_elv.raster.dims,
#         coords=da_elv.raster.coords,
#         data=ldd_np,
#         name="flwdir",
#     )
#     da_out.raster.set_nodata(255)
#     da_out.raster.set_crs(da_elv.raster.crs)
#     return da_out


def d8_from_dem(
    da_elv,
    gdf_stream=None,
    max_depth=-1.0,
):
    """Derive D8 flow directions from an elevation grid.

    Outlets occur at the edge of the data or at the interface with nodata values.
    A local depressions is filled based on its lowest pour point level if the pour point
    depth is smaller than the maximum pour point depth `max_depth`, otherwise the lowest
    elevation in the depression becomes a pit.

    Based on: Wang, L., & Liu, H. (2006). https://doi.org/10.1080/13658810500433453

    Parameters
    ----------
    da_elv: 2D xarray.DataArray
        elevation raster
    gdf_stream: geopandas.GeoDataArray, optional
        stream vector layer with 'uparea' [km2] column which is used to burn
        the river in the elevation data.
    max_depth: float, optional
        Maximum pour point depth. Depressions with a larger pour point
        depth are set as pit. A negative value (default) equals an infitely
        large pour point depth causing all depressions to be filled.

    Returns
    -------
    da_flw: 2D xarray.DataArray
        D8 flow direction data
    """
    nodata = da_elv.raster.nodata
    crs = da_elv.raster.crs
    assert da_elv.raster.res[1] < 0
    assert nodata is not None and ~np.isnan(nodata)
    # burn in river if
    if gdf_stream is not None and "uparea" in gdf_stream.columns:
        gdf_stream = gdf_stream.sort_values(by="uparea")
        dst_rivupa = da_elv.raster.rasterize(gdf_stream, col_name="uparea", nodata=0)
        # make sure the rivers have a slope and are below all other elevation cells.
        # river elevation = min(elv) - log10(uparea[m2]) from rasterized river uparea.
        elvmin = da_elv.where(da_elv != nodata).min()
        elvriv = elvmin - np.log10(np.maximum(1.0, dst_rivupa * 1e3))
        # synthetic elevation with river burned in
        da_elv = elvriv.where(np.logical_and(da_elv != nodata, dst_rivupa > 0), da_elv)
        da_elv.raster.set_nodata(nodata)
        da_elv.raster.set_crs(crs)
    # derive new flow directions from (synthetic) elevation
    d8 = pyflwdir.dem.fill_depressions(
        da_elv.values.astype(np.float32),
        max_depth=max_depth,
        nodata=da_elv.raster.nodata,
    )[1]
    # return xarray data array
    da_flw = xr.DataArray(
        dims=da_elv.raster.dims,
        coords=da_elv.raster.coords,
        data=d8,
        name="flwdir",
    )
    da_flw.raster.set_nodata(247)
    da_flw.raster.set_crs(crs)
    return da_flw


def reproject_hydrography(
    ds_hydro,
    da_elv,
    river_upa=5,
    method="bilinear",
    uparea_name="uparea",
    flwdir_name="flwdir",
    logger=logger,
):
    """Reproject flow direction and upstream area data to da_elv crs and resolution.
    Note that the resolution of da_elv should be similar or smaller for good results.

    The reprojection is done by creating a synthetic elevation grid based on
    reprojected elevation and upstream count grids. Rivers are first vectorized
    and then reprojected and rasterized for better preservation."""
    # TODO fix for case without ds_hydro, but with gdf_stream
    # check N->S orientation
    assert da_elv.raster.res[1] < 0
    assert ds_hydro.raster.res[1] < 0
    crs = da_elv.raster.crs
    bbox = np.asarray(da_elv.raster.bounds)
    # pad da_elv to avoid boundary problems
    buf0 = 2
    nrow, ncol = da_elv.raster.shape
    t = da_elv.raster.transform
    dst_transform = from_origin(
        t[2] - buf0 * t[0], t[5] + buf0 * abs(t[4]), t[0], abs(t[4])
    )
    dst_shape = nrow + buf0 * 2, ncol + buf0 * 2
    xcoords, ycoords = hydromt.gis_utils.affine_to_coords(dst_transform, dst_shape)
    da_elv = xr.DataArray(
        dims=da_elv.raster.dims,
        coords={da_elv.raster.x_dim: xcoords, da_elv.raster.y_dim: ycoords},
        data=np.pad(da_elv.values, buf0, "edge"),
        attrs=da_elv.attrs,
    )
    da_elv.raster.set_crs(crs)
    # reproject uparea & elevation with buffer
    da_upa = ds_hydro[uparea_name].raster.reproject_like(da_elv, method=method)
    max_upa = da_upa.where(da_upa != da_upa.raster.nodata).max().values
    nodata = da_elv.raster.nodata
    # vectorize and reproject river uparea
    mask = ds_hydro[uparea_name] > river_upa
    flwdir_src = hydromt.flw.flwdir_from_da(ds_hydro[flwdir_name], mask=mask)
    feats = flwdir_src.vectorize(uparea=ds_hydro[uparea_name].values)
    gdf_stream = gpd.GeoDataFrame.from_features(feats, crs=ds_hydro.raster.crs)
    gdf_stream = gdf_stream.sort_values(by="uparea")
    # only area with upa otherwise the outflows are not resolved!
    # synthetic elevation -> reprojected elevation - log10(reprojected uparea[m2])
    elvsyn = xr.where(
        np.logical_and(da_elv != nodata, da_upa != da_upa.raster.nodata),
        da_elv - np.log10(np.maximum(1.0, da_upa * 1e3)),
        nodata,
    )
    elvsyn.raster.set_nodata(nodata)
    elvsyn.raster.set_crs(crs)
    # get flow directions
    da_flw = d8_from_dem(elvsyn, gdf_stream).raster.clip_bbox(bbox)
    # calculate upstream area with uparea from rivers at edge
    flwdir = hydromt.flw.flwdir_from_da(da_flw, ftype="d8")
    da_flw.data = flwdir.to_array()  # to set outflow pits after clip
    area = flwdir.area / 1e6
    # get inflow cells: headwater river cells at edge
    rivupa = da_flw.raster.rasterize(gdf_stream, col_name="uparea", nodata=0)
    _edge = pyflwdir.gis_utils.get_edge(da_flw.values == 247)
    headwater = np.logical_and(
        rivupa.values > 0, flwdir.upstream_sum(rivupa.values > 0) == 0
    )
    inflow_idxs = np.where(np.logical_and(headwater, _edge).ravel())[0]
    if inflow_idxs.size > 0:
        # use nearest mapping to avoid duplicating uparea when reprojecting to higher res.
        gdf0 = gpd.GeoDataFrame(
            index=inflow_idxs,
            geometry=gpd.points_from_xy(*flwdir.xy(inflow_idxs)),
            crs=crs,
        )
        gdf0["idx2"], gdf0["dst2"] = gis_utils.nearest(gdf0, gdf_stream)
        gdf0 = gdf0.sort_values(by="dst2").drop_duplicates("idx2")
        gdf0["uparea"] = gdf_stream.loc[gdf0["idx2"].values, "uparea"].values
        # set stream uparea to selected inflow cells and calculate total uparea
        area.flat[gdf0.index.values] = gdf0["uparea"].values
        logger.info(
            f"Calculating upstream area with {gdf0.index.size} input cell at the domain edge."
        )
    da_upa = xr.DataArray(
        dims=da_flw.raster.dims,
        coords=da_flw.raster.coords,
        data=flwdir.accuflux(area).astype(np.float32),
        name="uparea",
    )
    da_upa.raster.set_nodata(-9999)
    da_upa.raster.set_crs(crs)
    max_upa1 = da_upa.max().values
    logger.info(
        f"Reprojected maximum upstream area: {max_upa1:.2f} km2 ({max_upa:.2f} km2)"
    )
    return xr.merge([da_flw, da_upa])


def river_inflow_points(
    ds,
    region=None,
    river_upa=25,
    river_len=0,
    dst_crs=None,
    return_river=True,
    flwdir_name="flwdir",
    uparea_name="uparea",
    logger=logger,
):
    """
    Returns the most downstream point locations where a river enters the region.
    Rivers are based on the flow direction data in the dataset and a minumum upstream
    area threshold.

    Parameters
    ----------
    ds: xarray.Dataset
        Hydrography raster data, should contain flwdir_name, uparea_name variables.
    region: geopandas.GeoDataFrame, optional
        Polygon of region of interest. By default all valid cells in ds are used to
        determine the region of interest.
    river_upa: float, optional
        Mimimum upstream area threshold [km2] to define river cells, by default 25 km2.
    river_len: float, optional
        Mimimum river length [m] within the model domain to define river cells, by default 0 m.
    return_river: bool, optional
        If True, return a vectorized river GeoDataFrame

    Returns
    -------
    gdf_src, gdf_riv: geopandas.GeoDataFrame
        Inflow point and river line vector data.
    """
    src_crs = ds.raster.crs
    if dst_crs is None:
        dst_crs = src_crs
    if region is not None:
        da_mask = ds.raster.geometry_mask(region)
    else:
        da_mask = ds[flwdir_name] != ds[flwdir_name].raster.nodata
    da_mask_eroded = ndimage.binary_erosion(da_mask, structure=np.ones((3, 3)))
    da_mask_edge = np.logical_xor(da_mask_eroded, da_mask)

    # initialize flwdir with river cells inside region only
    rivmsk = np.logical_and(ds[uparea_name] >= river_upa, da_mask)
    flwdir = hydromt.flw.flwdir_from_da(ds[flwdir_name], mask=rivmsk)

    # set source points at inflowing region edge cells
    rivmsk = np.logical_and(rivmsk, flwdir.upstream_sum(rivmsk.astype(np.int8)) == 0)
    idxs_source = np.where(np.logical_and(rivmsk, da_mask_edge).values.ravel())[0]

    gdf_src = gpd.GeoDataFrame()
    if len(idxs_source) > 0:
        # filter based on length until pit
        if river_len > 0:
            path_length = flwdir.path(idxs=idxs_source, unit="m")[1]
            idxs_source = idxs_source[path_length > river_len]
        # get coordinates
        source_xy = flwdir.xy(idxs_source)
        gdf_src = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*source_xy), crs=src_crs
        ).to_crs(dst_crs)
        gdf_src["uparea"] = ds[uparea_name].values.flat[idxs_source]
    logger.debug(f"{len(idxs_source)} river inflow point locations found.")

    if return_river:
        logger.debug(f"Vectorize river.")
        feats = flwdir.streams()
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs).to_crs(dst_crs)
        return gdf_src, gdf_riv
    else:
        return gdf_src, None


def river_outflow_points(
    da_flwdir,
    region=None,
    river_upa=5.0,
    return_river=True,
    dst_crs=None,
    logger=logger,
):
    """
    Returns point locations where rivers leave the region.
    Rivers are based on the flow direction data in the dataset and a minumum upstream
    area threshold based on cells within the model domain.

    Parameters
    ----------
    da_flwdir: xarray.DataArray
        Flow directoin raster data.
    region: geopandas.GeoDataFrame
        Polygon of region of interest.
    river_upa: float, optional
        Mimimum upstream area threshold [km2] to define river cells, by default 5 km2.
    return_river: bool, optional
        If True, return a vectorized river GeoDataFrame

    Returns
    -------
    gdf_src, gdf_riv: geopandas.GeoDataFrame
        Inflow point and river line vector data.
    """
    src_crs = da_flwdir.raster.crs
    if dst_crs is None:
        dst_crs = src_crs
    if region is not None:
        da_mask = da_flwdir.raster.geometry_mask(region)
    else:
        da_mask = da_flwdir != da_flwdir.raster.nodata
    da_mask_eroded = ndimage.binary_erosion(da_mask, structure=np.ones((3, 3)))
    da_mask_edge = np.logical_xor(da_mask_eroded, da_mask)

    # initialize flwdir with all cells in domain and get uparea
    flwdir = hydromt.flw.flwdir_from_da(da_flwdir, mask=da_mask)
    rivmsk = flwdir.upstream_area(unit="km2") > river_upa

    # git pits at domain edge on flwdir grid with minimal uparea within model domain
    idxs0 = flwdir.idxs_pit
    select = np.logical_and(da_mask_edge.values.flat[idxs0], rivmsk.flat[idxs0])
    idxs_out = idxs0[select]
    logger.debug(f"{len(idxs_out)} river outflow point locations found.")

    gdf_out = gpd.GeoDataFrame()
    if len(idxs_out) > 0:
        gdf_out = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*flwdir.xy(idxs_out)), crs=src_crs
        ).to_crs(dst_crs)

    if return_river and np.any(rivmsk):
        logger.debug(f"Vectorize river for outflow points.")
        feats = hydromt.flw.flwdir_from_da(da_flwdir, mask=rivmsk).streams()
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs).to_crs(dst_crs)
        return gdf_out, gdf_riv
    else:
        return gdf_out, None
