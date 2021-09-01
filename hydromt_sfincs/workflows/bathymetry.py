import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pyflwdir
import xarray as xr
from scipy import ndimage
from typing import Union
import logging

from ..gis_utils import nearest, spread2d

logger = logging.getLogger(__name__)

__all__ = [
    "merge_topobathy",
    "mask_topobathy",
    "get_rivbank_dz",
    "get_rivwth",
    "get_river_bathymetry",
    "get_estuary_bathymetry",
    "burn_bathymetry",
]


def mask_topobathy(
    da_elv: xr.DataArray, elv_min: float, elv_max: float = None
) -> xr.DataArray:
    """Return mask of valid elevation cells within [elv_min, elv_max] range.
    Note that local sinks (isolated regions with elv < elv_min) are kept.
    """
    dep_mask = da_elv != da_elv.raster.nodata
    if elv_min is not None:
        # active cells: contiguous area above depth threshold
        _msk = ndimage.binary_fill_holes(da_elv >= elv_min)
        dep_mask = dep_mask.where(_msk, False)

    if elv_max is not None:
        dep_mask = dep_mask.where(da_elv <= elv_max, False)

    return dep_mask


def merge_topobathy(
    da1: xr.DataArray,
    da2: xr.DataArray,
    da_offset: Union[xr.DataArray, float] = None,
    merge_buffer: int = 0,
    merge_method: str = "first",
    elv_min: float = None,
    elv_max: float = None,
    reproj_method="bilinear",
    logger=logger,
) -> xr.DataArray:
    """Return merged topobathy data from two datasets.

    Values from the second dataset are used where `da_mask` equals True or,
    if not provided, where the first dataset has missing values.

    If `merge_buffer` > 0, values of da2 are replaced with linearly
    interpolated values within the buffer.

    If `da_offset` is provided, a (spatially varying) offset is applied to the
    second dataset to convert the vertical datum before merging.

    Parameters
    ----------
    da1, da2: xr.DataArray
        Datasets with topobathy data to be merged.
    da_offset: xr.DataArray, float, optional
        Dataset with spatially varying offset or float with uniform offset
    merge_method: str {'first','last','min','max'}
        merge method, by default 'first':

        * first: use valid new where existing invalid
        * last: use valid new
        * min: pixel-wise min of existing and new
        * max: pixel-wise max of existing and new
    elv_min, elv_max : float, optional
        Minimum and maximum elevation caps for new topobathy cells, cells outside
        this range are linearly interpolated. Note: applied after offset!
    merge_buffer: int
        Buffer (number of cells) within the da_mask==True region where topobathy
        values are based on linear interpolation for a smooth transition, by default 0.
    reproj_method: str
        Method used to reproject the offset and second dataset to the grid of the
        first dataset, by default 'bilinear'


    Returns
    -------
    da_out: xr.DataArray
        Merged topobathy dataset
    """

    nodata = da1.raster.nodata
    if nodata is None or np.isnan(nodata):
        raise ValueError("da1 nodata value should be a finite value.")
    ## reproject da2 and reset nodata value to match da1 nodata
    da2 = (
        da2.raster.reproject_like(da1, method=reproj_method)
        .raster.mask_nodata()
        .fillna(nodata)
    )
    ## add offset
    if da_offset is not None:
        if isinstance(da_offset, xr.DataArray):
            da_offset = (
                da_offset.raster.reproject_like(da1, method=reproj_method)
                .raster.mask_nodata()
                .fillna(0)
            )
        da2 = da2.where(da2 == nodata, da2 + da_offset)
    # merge based merge_method
    if merge_method == "first":
        mask = da1 != nodata
    elif merge_method == "last":
        mask = da2 == nodata
    elif merge_method == "min":
        mask = da1 < da2
    elif merge_method == "max":
        mask = da1 > da2
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}")
    da_out = da1.where(mask, da2)
    da_out.raster.set_nodata(nodata)
    # identify holes in merged elevation
    struct = np.ones((3, 3))
    na_mask = da_out != nodata
    if np.any(~na_mask.values):
        na_mask = ndimage.binary_fill_holes(na_mask, structure=struct)
    # mask invalid elevation values
    if elv_min is not None:
        da_out = da_out.where(~np.logical_and(~mask, da2 < elv_min), nodata)
    if elv_max is not None:
        da_out = da_out.where(~np.logical_and(~mask, da2 > elv_max), nodata)
    # identify buffer cells and set to nodata
    if merge_buffer > 0:
        mask_dilated = ndimage.binary_dilation(mask, struct, iterations=merge_buffer)
        mask_buf = np.logical_xor(mask, mask_dilated)
        da_out = da_out.where(~mask_buf, nodata)
    # interpolate invalid elevtn, buffer and holes ( nodata values )
    nempty = np.sum(da_out.values[na_mask] == nodata)
    if nempty > 0:
        logger.debug(f"Interpolate topobathy at {int(nempty)} cells")
        da_out = da_out.raster.interpolate_na(method="linear")
        da_out = da_out.where(na_mask, nodata)  # reset extrapolated area
    return da_out


def get_rivbank_dz(
    gdf_stream: gpd.GeoDataFrame,
    da_rivmask: xr.DataArray,
    da_hand: xr.DataArray,
    nmin=5,
) -> np.ndarray:
    """Return median river bank height estimated from a river mask and height above
    nearest drainage maps.

    The river bank is defined based on pixels adjacent of da_rivmask. For each river bank
    pixel the nearest stream is found. The river bank height is then calculated from
    the median HAND value of all bank pixels with a HAND value larger than zero.
    """
    # rasterize streams
    gdf_stream["segid"] = np.arange(1, gdf_stream.index.size + 1, dtype=np.int32)
    segid = da_hand.raster.rasterize(gdf_stream, "segid").astype(np.int32)
    segid.raster.set_nodata(0)
    segid.name = "segid"
    # NOTE: the assumption is that banks are found in cells adjacent to any da_rivmask cell
    da_rivmask = da_rivmask.raster.reproject_like(da_hand)
    _mask = ndimage.binary_fill_holes(da_rivmask)  # remove islands
    mask = ndimage.binary_dilation(_mask, np.ones((3, 3)))
    da_mask = xr.DataArray(
        coords=da_hand.raster.coords, dims=da_hand.raster.dims, data=mask
    )
    da_mask.raster.set_crs(da_hand.raster.crs)
    # find nearest stream segment for all river bank cells
    segid_spread = spread2d(da_obs=segid, da_mask=da_mask)
    # get edge of riv mask -> riv banks
    bnk_mask = np.logical_and(da_hand > 0, np.logical_xor(da_mask, _mask))
    riv_mask = np.logical_and(
        np.logical_and(da_hand >= 0, da_rivmask), np.logical_xor(bnk_mask, da_mask)
    )
    # get median HAND for each stream -> riv bank dz
    seg_dzbank = ndimage.labeled_comprehension(
        da_hand.values,
        labels=np.where(bnk_mask, segid_spread["segid"].values, np.int32(0)),
        index=gdf_stream["segid"].values,
        func=lambda x: 0 if x.size < nmin else np.median(x),
        out_dtype=da_hand.dtype,
        default=0,
    )
    # get da_rivmask
    return seg_dzbank, riv_mask, bnk_mask


def get_rivwth(
    gdf_stream: gpd.GeoDataFrame,
    da_rivmask: xr.DataArray,
    nmin=5,
) -> np.ndarray:
    """Return mean river width along segments in gdf_stream derived by
    the stream area of each segment divided by the segment length."""
    assert da_rivmask.raster.crs.is_projected
    gdf_stream = gdf_stream.copy()
    # get/check river length
    if "rivlen" not in gdf_stream.columns:
        gdf_stream["rivlen"] = gdf_stream.to_crs(da_rivmask.raster.crs).length
    # rasterize streams
    gdf_stream["segid"] = np.arange(1, gdf_stream.index.size + 1, dtype=np.int32)
    segid = da_rivmask.raster.rasterize(gdf_stream, "segid").astype(np.int32)
    segid.raster.set_nodata(0)
    segid.name = "segid"
    # remove islands to get total width of braided rivers
    da_mask = da_rivmask.copy()
    da_mask.data = ndimage.binary_fill_holes(da_mask.values)
    # find nearest stream segment for all river cells
    segid_spread = spread2d(segid, da_mask)
    # get average width based on da_rivmask area and segment length
    cellarea = abs(np.multiply(*da_rivmask.raster.res))
    seg_count = ndimage.sum(
        da_rivmask, segid_spread["segid"].values, gdf_stream["segid"].values
    )
    rivwth = seg_count * cellarea / gdf_stream["rivlen"]
    rivwth = np.where(seg_count > nmin, rivwth, -9999)
    return rivwth, segid_spread["segid"].where(da_rivmask, 0)


def get_estuary_bathymetry(
    gdf_stream, da_rivmask, min_convergence=1e-2, smooth_n=1, max_elv=2
):
    # NOTE: works best with 2-5km river segments.
    assert da_rivmask.raster.crs.is_projected
    assert np.all(np.isin(["idx", "idx_ds", "uparea", "elevtn"], gdf_stream.columns))
    # set flwdir
    gdf_est = gdf_stream.copy()
    # get/check river length
    if "rivlen" not in gdf_stream.columns:
        gdf_est["rivlen"] = gdf_stream.to_crs(da_rivmask.raster.crs).length
    flw = pyflwdir.from_dataframe(gdf_est.set_index("idx"))
    flw.main_upstream(gdf_est["uparea"].values)
    # get rivwth from mask based on main stream only and smooth
    main_stem = np.hstack([flw.idxs_us_main[flw.idxs_us_main >= 0], flw.idxs_pit])
    rivwth = np.full(gdf_est.index.size, -9999.0, np.float32)
    rivwth[main_stem], segids = get_rivwth(gdf_est.iloc[main_stem], da_rivmask)
    rivwth = flw.moving_average(rivwth, smooth_n, restrict_strord=True, nodata=-9999.0)
    # estuaries defined as area from pit moving upstream until point where
    # width convergence falls below "min_convergence" threshold
    rivlen = gdf_est["rivlen"].values
    estuary = np.full(rivwth.size, False, bool)
    elv_check = gdf_est["elevtn"].values[flw.idxs_pit] < max_elv
    wth_check = rivwth[flw.idxs_pit] != -9999.0
    estuary[flw.idxs_pit[np.logical_and(elv_check, wth_check)]] = True
    for idx in flw.idxs_seq:  # down- to upstream
        if not estuary[idx]:
            continue
        idx1 = flw.idxs_us_main[idx]
        if (
            rivlen[idx] > 0
            and ((rivwth[idx] - rivwth[idx1]) / rivlen[idx]) > min_convergence
        ):
            estuary[idx1] = True
    gdf_est["estuary"] = estuary
    # estuary mask
    da_estmask = segids.isin(np.where(estuary[main_stem])[0] + 1).astype(np.int8)
    da_estmask.raster.set_nodata(0)
    da_estmask.raster.set_crs(da_rivmask.raster.crs)
    # set riverwidth from mask
    if "rivwth" not in gdf_est.columns:
        gdf_est["rivwth"] = np.nan
    gdf_est.loc[estuary, "rivwth"] = rivwth[estuary]
    #  set constant depth within estuary based on most upstream estimate
    if "rivdph" in gdf_est.columns:
        gdf_est0 = gdf_est[gdf_est["estuary"]]
        flw0 = pyflwdir.from_dataframe(gdf_est0.set_index("idx"))
        rivdph = gdf_est0["rivdph"]
        rivdph[flw0.n_upstream > 0] = -1
        rivdph = flw0.fillnodata(rivdph, -1, "down")
        gdf_est.loc[gdf_est0.index, "rivdph"] = rivdph
    return gdf_est, da_estmask


def get_river_bathymetry(
    gdf_stream,
    gdf_data=None,
    max_dist=None,
    method="powlaw",
    hc=0.27,
    hp=0.30,
    hmin=1.0,
    wmin=30.0,
    n=0.03,
    smin=1e-5,
):
    """Create complete `rivwth` and `rivdph` columns in the `gdf_stream` GeoDataFrame.

    Missing `rivwth`, `qbankfull` and `rivslp` values are filled with data from nearest segment in gdf_data
    with a `max_dist` distance and remaining gaps are filled by propagating valid values downstream (except for rivslp).

    Missing `rivdph` values are based on a 1) power-law relationship or 2) mannings equation, both based
    on bankfull discharge.

    """
    cols = ["qbankfull", "rivwth"]
    if method == "manning":
        cols = cols + ["rivslp"]

    # merge missing data in gdf_stream with data from gdf_data
    if gdf_data is not None:
        assert np.any(np.isin(cols, gdf_data.columns))
        idx_nn, dst = nearest(gdf_stream, gdf_data)
        gdf1 = gdf_data.loc[idx_nn]
        max_dist = gdf1["rivwth"].values / 2.0 if max_dist is None else max_dist
        valid = dst < max_dist
        gdf_stream.loc[valid, "index_right"] = idx_nn[valid]
        for col in cols:
            if col not in gdf_data:
                continue
            new_vals = gdf1.loc[valid, col].values
            if col in gdf_stream:
                old_vals = gdf_stream.loc[valid, col].values
                new_vals = np.where(np.isnan(old_vals), new_vals, old_vals)
            gdf_stream.loc[valid, col] = new_vals
    assert np.all(np.isin(cols, gdf_stream.columns))

    # fill nodata in rivwth and qbankfull by simply propagating values downstream
    if np.any(np.isnan(gdf_stream[cols])) and np.all(
        np.isin(["idx", "idx_ds"], gdf_stream.columns)
    ):
        flw = pyflwdir.from_dataframe(gdf_stream.set_index("idx"))
        rivwth = gdf_stream["rivwth"].fillna(-9999)
        rivqbf = gdf_stream["qbankfull"].fillna(-9999)
        gdf_stream["rivwth"] = flw.fillnodata(rivwth, -9999)
        gdf_stream["qbankfull"] = flw.fillnodata(rivqbf, -9999)

    # river depth based on power-law relation.
    # default values form Moody & Troutman 2002: hc=0.27 & hp=0.30
    rivqbf = np.maximum(0, gdf_stream["qbankfull"].fillna(0))
    gdf_stream["rivwth"] = np.maximum(wmin, gdf_stream["rivwth"].fillna(wmin))
    if method == "powlaw":
        rivdph = hc * rivqbf ** hp
    # river depth based on manning equation
    elif method == "manning":
        rivslp = np.maximum(smin, gdf_stream["rivslp"].fillna(smin))
        fmanning = lambda n, q, s, w: ((n * q) / (np.sqrt(s) * w)) ** (3 / 5)
        rivdph = fmanning(n, rivqbf, rivslp, gdf_stream["rivwth"])
    else:
        raise ValueError(f'Method {method} unknown: use one of "powlaw", "manning".')
        # TODO: add gradually varying flow method
    if "rivdph" in gdf_stream:
        old_vals = gdf_stream["rivdph"]
        rivdph = np.where(np.isnan(old_vals), rivdph, old_vals)
    gdf_stream["rivdph"] = np.maximum(rivdph, hmin)
    return gdf_stream


def burn_bathymetry(gdf_stream, da_elevtn, da_rivmask=None, add_rivbnk_dz=False):
    cols = ["rivdph", "uparea"]
    if da_rivmask is None:
        cols += ["rivwth"]
    if add_rivbnk_dz:
        cols += ["rivbnk_dz"]
    check_cols = np.isin(cols, gdf_stream.columns)
    if not np.all(check_cols):
        miss_cols = '", "'.join(np.array(cols)[~check_cols])
        raise ValueError(f'Missing gdf columns: "{miss_cols}"')
    gdf_stream_buf = gdf_stream.copy()
    gdf_stream_buf["geometry"] = gdf_stream_buf.buffer(gdf_stream_buf["rivwth"] / 2.0)
    if da_rivmask is not None:
        # mask model cells which are completely inside rivmask using 'min' interpolation
        da_rivmask = da_rivmask.raster.reproject_like(da_elevtn, "min") != 0
        # remove segments which are in mask from buffered stream lines
        index, crs = gdf_stream.index, gdf_stream.crs
        p0, p1 = zip(
            *[(Point(g.coords[0]), Point(g.coords[-1])) for g in gdf_stream.geometry]
        )
        in_mask = np.logical_and(
            da_rivmask.raster.sample(gpd.GeoSeries(p0, index=index, crs=crs)),
            da_rivmask.raster.sample(gpd.GeoSeries(p1, index=index, crs=crs)),
        )
        gdf_stream_buf.loc[in_mask.values, "rivwth"] = 1
    else:
        da_rivmask = xr.full_like(da_elevtn, False, bool)
    # buffer and rasterize streams
    da_rivbuffer = da_rivmask.raster.geometry_mask(gdf_stream_buf)
    # combine stream buffer
    da_rivmask = np.logical_or(da_rivmask, da_rivbuffer)

    # get river centerline z plus optional riverbank dz
    nodata = da_elevtn.raster.nodata
    mask = da_elevtn != nodata
    riv_center_mask = da_elevtn.raster.geometry_mask(gdf_stream)
    riv_center_z0 = da_elevtn.where(riv_center_mask, nodata)
    if add_rivbnk_dz:
        riv_center_z0 = riv_center_z0 + da_elevtn.raster.rasterize(
            gdf_stream.sort_values(by="uparea"),
            col_name="rivbnk_dz",
            nodata=0,
        )
    # river bedlevel based on z0 - rivdph
    riv_center_zb = riv_center_z0 - da_elevtn.raster.rasterize(
        gdf_stream.sort_values(by="uparea"),
        col_name="rivdph",
        nodata=0,
    )
    # spread in river mask
    riv_center_zb.raster.set_nodata(nodata)
    riv_center_zb.name = "elevtn"
    zb_riv = spread2d(riv_center_zb, da_rivmask)["elevtn"]
    # combine bathymetry with elevtn
    riv_mask = np.logical_or(da_rivmask, riv_center_mask)
    zb = zb_riv.where(riv_mask, da_elevtn).where(mask, nodata)
    zb.raster.set_nodata(-9999)
    return zb, riv_mask
