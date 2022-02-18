import geopandas as gpd
import numpy as np
import pyflwdir
import xarray as xr
from scipy import ndimage
from typing import Union, Tuple, Optional
import logging

from hydromt.gis_utils import nearest_merge, nearest, spread2d
from hydromt.workflows import rivers

logger = logging.getLogger(__name__)

__all__ = [
    "merge_topobathy",
    "get_rivbank_dz",
    "get_river_bathymetry",
    "burn_river_zb",
]


def merge_topobathy(
    da1: xr.DataArray,
    da2: xr.DataArray,
    da_offset: Union[xr.DataArray, float] = None,
    merge_buffer: int = 0,
    merge_method: str = "first",
    elv_min: float = None,
    elv_max: float = None,
    reproj_method: str = "bilinear",
    interp_method: str = "linear",
    max_width: int = 0,
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
    merge_method: {'first','last','min','max'}, optional
        merge method, by default 'first':

        * first: use valid new where existing invalid
        * last: use valid new
        * min: pixel-wise min of existing and new
        * max: pixel-wise max of existing and new
    elv_min, elv_max : float, optional
        Minimum and maximum elevation caps for new topobathy cells, cells outside
        this range are linearly interpolated. Note: applied after offset!
    merge_buffer: int, optional
        Buffer (number of cells) within the da_mask==True region where topobathy
        values are based on linear interpolation for a smooth transition, by default 0.
    max_width: int, optional
        Maximum width (number of cells) to append to valid da1 cells. By default 0.
    reproj_method: {'bilinear', 'cubic', 'nearest'}
        Method used to reproject the offset and second dataset to the grid of the
        first dataset, by default 'bilinear'
    interp_method, {'linear', 'nearest', 'rio_idw'}
        Method used to interpolate holes of nodata in the merged dataset, by default 'linear'.

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
    # mask invalid elevation before merging
    if elv_min is not None:
        da2 = da2.where(da2 >= elv_min, nodata)
    if elv_max is not None:
        da2 = da2.where(da2 <= elv_max, nodata)
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
    # identify buffer cells and set to nodata
    if merge_buffer > 0:
        mask_dilated = ndimage.binary_dilation(mask, struct, iterations=merge_buffer)
        mask_buf = np.logical_xor(mask, mask_dilated)
        da_out = da_out.where(~mask_buf, nodata)
    # interpolate invalid elevtn, buffer and holes ( nodata values )
    nempty = np.sum(da_out.values[na_mask] == nodata)
    if nempty > 0 and interp_method:
        logger.debug(f"Interpolate topobathy at {int(nempty)} cells")
        da_out = da_out.raster.interpolate_na(method=interp_method)
        da_out = da_out.where(na_mask, nodata)  # reset extrapolated area
    if max_width > 0:
        mask_dilated = ndimage.binary_dilation(mask, struct, iterations=max_width)
        da_out = da_out.where(mask_dilated, nodata)
    return da_out


def get_rivbank_dz(
    gdf_riv: gpd.GeoDataFrame,
    da_msk: xr.DataArray,
    da_hnd: xr.DataArray,
    nmin: int = 20,
    q: float = 25.0,
) -> np.ndarray:
    """Return river bank height estimated as from height above nearest drainage
    (HAND) values adjecent to river cells. For each feature in `gdf_riv` the nearest
    river bank cells are identified and the bank heigth is estimated based on a quantile
    value `q`.

    Parameters
    ----------
    gdf_riv : gpd.GeoDataFrame
        River segments
    da_msk : xr.DataArray of bool
        River mask
    da_hnd : xr.DataArray of float
        Height above nearest drain (HAND) map
    nmin : int, optional
        Minimum threshold for valid river bank cells, by default 20
    q : float, optional
        quantile [0-100] for river bank estimate, by default 25.0

    Returns
    -------
    rivbank_dz: np.ndarray
        riverbank elevations for each segment in `gdf_riv`
    da_riv_mask, da_bnk_mask: xr.DataArray:
        River and river-bank masks
    """
    # rasterize streams
    gdf_riv = gdf_riv.copy()
    gdf_riv["segid"] = np.arange(1, gdf_riv.index.size + 1, dtype=np.int32)
    valid = gdf_riv.length > 0  # drop pits or any zero length segments
    segid = da_hnd.raster.rasterize(gdf_riv[valid], "segid").astype(np.int32)
    segid.raster.set_nodata(0)
    segid.name = "segid"
    # NOTE: the assumption is that banks are found in cells adjacent to any da_msk cell
    da_msk = da_msk.raster.reproject_like(da_hnd, method="nearest")
    _mask = ndimage.binary_fill_holes(da_msk)  # remove islands
    mask = ndimage.binary_dilation(_mask, np.ones((3, 3)))
    da_mask = xr.DataArray(
        coords=da_hnd.raster.coords, dims=da_hnd.raster.dims, data=mask
    )
    da_mask.raster.set_crs(da_hnd.raster.crs)
    # find nearest stream segment for all river bank cells
    segid_spread = spread2d(da_obs=segid, da_mask=da_mask)
    # get edge of riv mask -> riv banks
    da_bnk_mask = np.logical_and(da_hnd > 0, np.logical_xor(da_mask, _mask))
    da_riv_mask = np.logical_and(
        np.logical_and(da_hnd >= 0, da_msk), np.logical_xor(da_bnk_mask, da_mask)
    )
    # get median HAND for each stream -> riv bank dz
    rivbank_dz = ndimage.labeled_comprehension(
        da_hnd.values,
        labels=np.where(da_bnk_mask, segid_spread["segid"].values, np.int32(0)),
        index=gdf_riv["segid"].values,
        func=lambda x: 0 if x.size < nmin else np.percentile(x, q),
        out_dtype=da_hnd.dtype,
        default=-9999,
    )
    return rivbank_dz, da_riv_mask, da_bnk_mask


def get_river_bathymetry(
    ds: xr.Dataset,
    flwdir: pyflwdir.FlwdirRaster,
    gdf_riv: gpd.GeoDataFrame = None,
    gdf_qbf: gpd.GeoDataFrame = None,
    rivdph_method: str = "gvf",
    rivwth_method: str = "mask",
    river_upa: float = 100.0,
    river_len: float = 1e3,
    min_rivdph: float = 1.0,
    min_rivwth: float = 50.0,
    segment_length: float = 5e3,
    smooth_length: float = 10e3,
    min_convergence: float = 0.01,
    max_dist: float = 100.0,
    rivbank: bool = True,
    rivbankq: float = 25,
    constrain_estuary: bool = True,
    constrain_rivbed: bool = True,
    elevtn_name: str = "elevtn",
    uparea_name: str = "uparea",
    rivmsk_name: str = "rivmsk",
    logger=logger,
    **kwargs,
) -> Tuple[gpd.GeoDataFrame, xr.DataArray]:
    """Estimate river bedlevel zb using gradually varying flow (gvf), manning's equation
    (manning) or a power-law relation (powlaw) rivdph_method. The river is based on flow
    directions with and minimum upstream area threshold.

    Parameters
    ----------
    ds : xr.Dataset
        Model map layers containing `elevnt_name`, `uparea_name` and `rivmsk_name` (optional)
        variables.
    flwdir : pyflwdir.FlwdirRaster
        Flow direction object
    gdf_riv : gpd.GeoDataFrame, optional
        River attribute data with "qbankfull" and "rivwth" data, by default None
    gdf_qbf : gpd.GeoDataFrame, optional
        Bankfull river discharge data with "qbankfull" column, by default None
    rivdph_method : {'gvf', 'manning', 'powlaw', 'geom'}
        River depth estimate method, by default 'gvf'
    rivwth_method : {'geom', 'mask'}
        River width estimate method, by default 'mask'
    river_upa : float, optional
        Minimum upstream area threshold for rivers [km2], by default 100.0
    river_len: float, optional
        Mimimum river length [m] within the model domain to define river cells, by default 1000.
    min_rivwth, min_rivdph: float, optional
        Minimum river width [m] (by default 50.0 m) and depth [m] (by default 1.0 m)
    segment_length : float, optional
        Approximate river segment length [m], by default 5e3
    smooth_length : float, optional
        Approximate smoothing length [m], by default 10e3
    min_convergence : float, optional
        Minimum width convergence threshold to define estuaries [m/m], by default 0.01
    max_dist : float, optional
        Maximum distance threshold to spatially merge `gdf_riv` and `gdf_qbf`, by default 100.0
    rivbank: bool, optional
        If True (default), approximate the reference elevation for the river depth based
        on the river bankfull elevation at cells neighboring river cells. Otherwise
        use the elevation of the local river cell as reference level.
    rivbankq : float, optional
        quantile [1-100] for river bank estimation, by default 25
    constrain_estuary : bool, optional
        If True (default) fix the river depth in estuaries based on the upstream river depth.
    constrain_rivbed : bool, optional
        If True (default) correct the river bed level to be hydrologically correct

    Returns
    -------
    gdf_riv: gpd.GeoDataFrame
        River segments with bed level (zb) estimates
    da_msk: xr.DataArray:
        River mask
    """
    raster_kwargs = dict(coords=ds.raster.coords, dims=ds.raster.dims)
    da_elv = ds[elevtn_name]

    # get vector of stream segments
    da_upa = ds[uparea_name]
    rivd8 = da_upa > river_upa
    if river_len > 0:
        dx_headwater = np.where(flwdir.upstream_sum(rivd8) == 0, flwdir.distnc, 0)
        rivlen = flwdir.fillnodata(dx_headwater, nodata=0, direction="down", how="max")
        rivd8 = np.logical_and(rivd8, rivlen >= river_len)
    feats = flwdir.streams(
        max_len=int(round(segment_length / ds.raster.res[0])),
        uparea=da_upa.values,
        elevtn=da_elv.values,
        rivdst=flwdir.distnc,
        strord=flwdir.stream_order(mask=rivd8),
        mask=rivd8,
    )
    gdf_stream = gpd.GeoDataFrame.from_features(feats, crs=ds.raster.crs)
    # gdf_stream = gdf_stream[gdf_stream['rivdst']>0]   # remove pits
    flw = pyflwdir.from_dataframe(gdf_stream.set_index("idx"))
    _ = flw.main_upstream(uparea=gdf_stream["uparea"].values)

    # merge gdf_riv with gdf_stream
    if gdf_riv is not None:
        cols = [c for c in ["rivwth", "qbankfull"] if c in gdf_riv]
        gdf_riv = nearest_merge(gdf_stream, gdf_riv, columns=cols, max_dist=max_dist)
        gdf_riv["rivlen"] = gdf_riv["rivdst"] - flw.downstream(gdf_riv["rivdst"])
    else:
        gdf_riv = gdf_stream
    # merge gdf_qbf (qbankfull) with gdf_riv
    if gdf_qbf is not None and "qbankfull" in gdf_qbf.columns:
        if "qbankfull" in gdf_riv:
            gdf_riv = gdf_riv.drop(colums="qbankfull")
        idx_nn, dists = nearest(gdf_qbf, gdf_riv)
        valid = dists < max_dist
        gdf_riv.loc[idx_nn[valid], "qbankfull"] = gdf_qbf["qbankfull"].values[valid]
        logger.info(f"{sum(valid)}/{len(idx_nn)} qbankfull boundary points set.")
    check_q = rivdph_method == "geom"
    assert check_q or "qbankfull" in gdf_riv.columns, 'gdf_riv has no "qbankfull" data'
    # propagate qbankfull and rivwth values
    if rivwth_method == "geom" and "rivwth" not in gdf_riv.columns:
        logger.info(f'Setting missing "rivwth" to min_rivwth {min_rivwth:.2f}m')
        gdf_riv["rivwth"] = min_rivwth
    if rivdph_method == "geom" and "rivdph" not in gdf_riv.columns:
        logger.info(f'Setting missing "rivdph" to min_rivdph {min_rivdph:.2f}m')
        gdf_riv["rivdph"] = min_rivdph
    for col in ["qbankfull", "rivwth", "rivdph"]:
        if col not in gdf_riv.columns:
            continue
        data = gdf_riv[col].fillna(-9999)
        data = flw.fillnodata(data, -9999, direction="down", how="max")
        gdf_riv[col] = np.maximum(0, data)

    # create river mask with river polygon
    if rivmsk_name not in ds and "rivwth":
        assert gdf_riv.crs.is_projected
        gdf_riv_buf = gdf_riv.copy()
        buf = np.maximum(gdf_riv_buf["rivwth"] / 2, 1)
        gdf_riv_buf["geometry"] = gdf_riv_buf.buffer(buf)
        da_msk = np.logical_and(
            ds.raster.geometry_mask(gdf_riv_buf), da_elv != da_elv.raster.nodata
        )
    elif rivmsk_name in ds:
        #  merge river mask with river line
        da_msk = ds.raster.geometry_mask(gdf_riv, all_touched=True)
        da_msk = np.logical_or(da_msk, ds[rivmsk_name] > 0)
    else:
        raise ValueError("No river width or river mask provided.")

    ## get zs
    da_hnd = xr.DataArray(flwdir.hand(rivd8.values, da_elv.values), **raster_kwargs)
    da_hnd = da_hnd.where(da_elv >= 0, 0)
    da_hnd.raster.set_crs(ds.raster.crs)
    if rivbank:
        logger.info("Deriving bankfull river surface elevation from its banks.")
        dz = get_rivbank_dz(gdf_riv, da_msk=da_msk, da_hnd=da_hnd, q=rivbankq)[0]
        dz = flw.fillnodata(dz, -9999, direction="down", how="max")
    else:
        logger.info("Deriving bankfull river surface elevation at its center line.")
        dz = 0
    gdf_riv["zs"] = np.maximum(
        gdf_riv["elevtn"], flw.dem_adjust(gdf_riv["elevtn"] + dz)
    )
    gdf_riv["rivbank_dz"] = gdf_riv["zs"] - gdf_riv["elevtn"]

    # estimate stream segment average width from river mask
    if rivwth_method == "mask":
        logger.info("Deriving river segment average width from permanent water mask.")
        rivwth = rivers.river_width(gdf_riv, da_rivmask=da_msk)
        gdf_riv["rivwth"] = flw.fillnodata(rivwth, -9999, direction="down", how="max")
    smooth_n = int(np.round(smooth_length / segment_length / 2))
    if smooth_n > 0:
        logger.info(f"Smoothing river width (n={smooth_n}).")
        gdf_riv["rivwth"] = flw.moving_average(
            gdf_riv["rivwth"], n=smooth_n, restrict_strord=True
        )
    gdf_riv["rivwth"] = np.maximum(gdf_riv["rivwth"], min_rivwth)

    # estimate river depth, smooth and correct
    if "rivdph" not in gdf_riv.columns:
        gdf_riv["rivdph"] = rivers.river_depth(
            data=gdf_riv,
            flwdir=flw,
            method=rivdph_method,
            rivzs_name="zs",
            min_rivdph=min_rivdph,
            **kwargs,
        )
    if constrain_estuary:
        # set width from mask and depth constant in estuaries
        # estuaries based on convergence of width from river mask
        gdf_riv["estuary"] = flw.classify_estuaries(
            elevtn=gdf_riv["elevtn"],
            rivwth=gdf_riv["rivwth"],
            rivdst=gdf_riv["rivdst"],
            min_convergence=min_convergence,
        )
        gdf_riv["rivdph"] = np.where(
            gdf_riv["estuary"] == 1, -9999, gdf_riv["rivdph"].values
        )
    gdf_riv["rivdph"] = flw.fillnodata(gdf_riv["rivdph"], -9999, "down")
    gdf_riv["rivdph"] = np.maximum(gdf_riv["rivdph"], min_rivdph)

    # calculate bed level from river depth
    gdf_riv["zb"] = gdf_riv["zs"] - gdf_riv["rivdph"]
    if constrain_rivbed:
        gdf_riv["zb"] = flw.dem_adjust(gdf_riv["zb"])
    gdf_riv["zb"] = np.minimum(gdf_riv["zb"], gdf_riv["elevtn"])
    gdf_riv["rivdph"] = gdf_riv["zs"] - gdf_riv["zb"]

    # calculate rivslp
    if "rivslp" not in gdf_riv:
        dz = gdf_riv["zb"] - flw.downstream(gdf_riv["zb"])
        dx = gdf_riv["rivdst"] - flw.downstream(gdf_riv["rivdst"])
        # fill nodata with upstream neighbors and set lower bound of zero
        gdf_riv["rivslp"] = np.maximum(
            0, flw.fillnodata(np.where(dx > 0, dz / dx, -1), -1)
        )

    return gdf_riv, da_msk


def burn_river_zb(
    gdf_riv: gpd.GeoDataFrame,
    da_elv: xr.DataArray,
    da_msk: xr.DataArray,
    flwdir: pyflwdir.FlwdirRaster = None,
    river_d4: bool = True,
    logger=logger,
):
    """Burn bedlevels from `gdf_riv` (column zb) into the DEM `da_elv` at river cells
    indicated in `da_msk`.

    Parameters
    ----------
    gdf_riv: gpd.GeoDataFrame
        River segments with bed level (zb) estimates
    da_elv : xr.DataArray of float
        Elevation raster
    da_msk: xr.DataArray of bool:
        River mask
    flwdir : pyflwdir.FlwdirRaster, optional
        Flow direction object
    river_d4 : bool, optional
        If True (default) ensure river cells have D4 connectivity

    Returns
    -------
    da_elv1: xr.DataArray
        DEM with bedlevels burned in.
    """
    assert da_elv.raster.identical_grid(da_msk)
    logger.debug("Burn bedlevel values into DEM.")
    nodata = da_elv.raster.nodata
    # drop invalid segments, e.g. at pits
    gdf_riv = gdf_riv[np.logical_and(gdf_riv.length > 0, np.isfinite(gdf_riv["zb"]))]
    zb = da_elv.raster.rasterize(gdf_riv, col_name="zb", nodata=-9999)
    # interpolate values if rivslp and rivdst is given
    if np.all(np.isin(["rivslp", "rivdst"], gdf_riv.columns)) and flwdir is not None:
        logger.debug("Interpolate bedlevel values")
        gdf_riv1 = gdf_riv[gdf_riv["rivdst"] > 0]
        slp = da_elv.raster.rasterize(gdf_riv1, col_name="rivslp", nodata=0)
        dst0 = da_elv.raster.rasterize(gdf_riv1, col_name="rivdst", nodata=0)
        dst0 = np.where(dst0 > 0, flwdir.distnc - dst0, 0)
        zb = (zb + dst0 * slp).where(zb != -9999, -9999)
    zb.raster.set_nodata(-9999)
    zb.name = "elevtn"
    # spread values inside river mask and replace da_elv values
    da_elv1 = spread2d(zb, da_msk)["elevtn"]
    da_msk = np.logical_and(da_msk, da_elv1 != -9999)
    da_elv1 = np.minimum(da_elv, da_elv1.where(da_msk, da_elv))
    da_elv1 = da_elv1.where(da_elv != nodata, nodata)

    if river_d4 and flwdir is not None:
        logger.debug("Correct for D4 connectivity bed level")
        elevtn = flwdir.dem_dig_d4(da_elv1.values, rivmsk=da_msk.values, nodata=nodata)
        da_elv1 = xr.DataArray(
            data=elevtn,
            coords=da_elv.raster.coords,
            dims=da_elv.raster.dims,
        )
        da_msk = np.logical_or(da_msk, da_elv1 < da_elv)

    # set attrs and return
    da_elv1.raster.set_nodata(nodata)
    da_elv1.raster.set_crs(da_elv.raster.crs)
    return da_elv1, da_msk
