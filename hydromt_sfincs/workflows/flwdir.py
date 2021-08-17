import numpy as np
import xarray as xr
import geopandas as gpd
from scipy import ndimage
import hydromt
import pyflwdir
import logging
from rasterio.transform import from_origin

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
    max_depth=-1.0,
):
    """Wrapper around pyflwdir.from_dem to derive flow direcition from an elevation grid."""
    assert da_elv.raster.res[1] < 0
    # return xarray data array
    da_out = xr.DataArray(
        dims=da_elv.raster.dims,
        coords=da_elv.raster.coords,
        data=pyflwdir.dem.fill_depressions(
            da_elv.values, nodata=da_elv.raster.nodata, max_depth=max_depth
        )[1],
        name="flwdir",
    )
    da_out.raster.set_nodata(247)
    da_out.raster.set_crs(da_elv.raster.crs)
    return da_out


def reproject_hydrography(
    ds,
    ds_like,
    river_upa=5,
    method="bilinear",
    uparea_name="uparea",
    elevtn_name="elevtn",
    flwdir_name="flwdir",
    buffer=5,
):
    """Reproject flow direction and upstream area data to ds_like crs and resolution.
    Note that the resolution of ds_like should be similar or smaller for good results.

    The reprojection is done by creating a synthetic elevation grid based on
    reprojected elevation and upstream count grids. Rivers are first vectorized
    and then reprojected and rasterized for better preservation."""
    # check N->S orientation
    assert ds_like.raster.res[1] < 0
    assert ds.raster.res[1] < 0
    # reproject uparea & elevation with buffer
    ds = ds.raster.clip_geom(ds_like.raster.box, buffer=buffer)
    nrow, ncol = ds_like.raster.shape
    t = ds_like.raster.transform
    dst_transform = from_origin(
        t[2] - buffer * t[0], t[5] + buffer * abs(t[4]), t[0], abs(t[4])
    )
    kwargs = dict(
        dst_crs=ds_like.raster.crs,
        dst_transform=dst_transform,
        dst_width=ncol + buffer * 2,
        dst_height=nrow + buffer * 2,
    )
    dst_upa = ds[uparea_name].raster.reproject(**kwargs, method=method)
    dst_elv = ds[elevtn_name].raster.reproject(**kwargs, method=method)
    # vectorize and reproject river uparea
    nodata = dst_elv.raster.nodata
    mask = ds[uparea_name] > river_upa
    flwdir_src = hydromt.flw.flwdir_from_da(ds[flwdir_name], mask=mask)
    feats = flwdir_src.vectorize(uparea=ds[uparea_name].values)
    gdf_stream = gpd.GeoDataFrame.from_features(feats, crs=ds.raster.crs)
    gdf_stream = gdf_stream.sort_values(by="uparea")
    dst_rivupa = dst_elv.raster.rasterize(gdf_stream, col_name="uparea", nodata=0)
    dst_rivupa.name = uparea_name
    # synthetic elevation with
    # river -> min(elv) - log10(uparea[m2]) from rasterized river uparea.
    # other -> reprojected elevation - log10(uparea[m2]) from reprojected uparea
    elvmin = ds[elevtn_name].where(ds[elevtn_name] != nodata).min()
    elvsyn = xr.where(
        dst_rivupa > 0,
        elvmin - np.log10(np.maximum(1.1, dst_rivupa * 1e3)),
        dst_elv - np.log10(np.maximum(1.1, dst_upa * 1e3)),
    )
    elvsyn = elvsyn.where(dst_elv != nodata, nodata).astype(np.float32)
    elvsyn.raster.set_nodata(nodata)
    elvsyn.raster.set_crs(dst_elv.raster.crs)
    # derive new flow directions from synthetic elevation
    da_flw = d8_from_dem(elvsyn)
    flwdir = hydromt.flw.flwdir_from_da(da_flw)
    # get upstream area
    area = flwdir.area / 1e6  # cellarea [km2]
    if river_upa > 0:
        ds_temp = xr.merge([da_flw, dst_rivupa])
        ds_temp.raster.set_crs(ds_like.raster.crs)
        gdf0 = river_inflow_points(
            ds_temp,
            region=ds_like.raster.box,
            river_upa=river_upa,
            river_len=0,
            dst_crs=ds_like.raster.crs,
            return_river=False,
            flwdir_name=flwdir_name,
            uparea_name=uparea_name,
        )[0]
        if gdf0.index.size > 0:  # set incoming river uparea at boundary cells
            upa0 = ds[uparea_name].raster.sample(gdf0)
            # make sure incomming upstream area is not duplicated
            idx = upa0.to_dataframe().drop_duplicates(["x", "y"]).index.values
            idxs0 = flwdir.index(gdf0.loc[idx].geometry.x, gdf0.loc[idx].geometry.y)
            area.flat[idxs0] = upa0.sel(index=idx).values
    uparea = flwdir.accuflux(area, nodata=nodata)
    uparea = np.where(dst_elv.values != nodata, uparea, nodata)
    # return dataset with flwdir and uparea
    ds_out = xr.Dataset(coords=dst_elv.raster.coords)
    ds_out.raster.set_crs(dst_elv.raster.crs)
    dims = dst_elv.raster.dims
    flwdir_data = flwdir.to_array(ftype=flwdir_src.ftype)
    ds_out[flwdir_name] = xr.Variable(dims=dims, data=flwdir_data)
    ds_out[flwdir_name].raster.set_nodata(flwdir_src._core._mv)
    ds_out[uparea_name] = xr.Variable(dims=dims, data=uparea)
    ds_out[uparea_name].raster.set_nodata(nodata)
    ds_out["elvsyn"] = elvsyn
    # return ds_like extent
    ds_out = ds_out.raster.clip_bbox(ds_like.raster.bounds)
    return ds_out


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
    rivmsk = rivmsk.where(flwdir.n_upstream == 0, False)  # limit to headwater cells
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
