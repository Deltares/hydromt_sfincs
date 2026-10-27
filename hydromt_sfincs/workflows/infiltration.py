"""Infiltration estimation workflows.

Metadata, configuration bookkeeping, and grid I/O helpers live in
:py:mod:`hydromt_sfincs.components.infiltration_common`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "INCH_TO_METER",
    "adjust_curve_number",
    "bucket_from_hsg",
    "bucket_from_hsg_landuse",
    "cn_to_s",
    "constant_infiltration_from_ksat_lulc",
    "curve_number_from_landuse_hsg",
    "curve_number_with_recovery",
    "green_ampt_from_hsg",
    "green_ampt_from_hsg_landuse",
    "horton_from_hsg",
    "horton_from_hsg_landuse",
    "mask_invalid_ksat",
    "normalize_hsg_codes",
]

INCH_TO_METER = 0.0254


MODIFIER_COLUMNS = ("surface_factor", "storage_factor", "drainage_factor")

DUAL_HSG_DRAINED_MAPPING = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
}


def _ensure_modifier_dataframe(df_modifiers: pd.DataFrame) -> pd.DataFrame:
    """Validate and select the required land-use modifier columns.

    The land-use class may be supplied as the index or in a ``lulc`` column.

    Parameters
    ----------
    df_modifiers : pd.DataFrame
        Modifier table containing ``surface_factor``, ``storage_factor``, and
        ``drainage_factor`` columns.

    Returns
    -------
    pd.DataFrame
        Modifier table with only the required columns and land-use classes as
        its index.
    """
    df = df_modifiers
    if "lulc" in df.columns:
        df = df.set_index("lulc")
    missing = set(MODIFIER_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing land-use modifier columns: {sorted(missing)}")
    return df[list(MODIFIER_COLUMNS)]


def _modifier_layers(
    da_lulc: xr.DataArray,
    df_modifiers: pd.DataFrame,
    *,
    default: float = 1.0,
) -> xr.Dataset:
    """Reclassify land-use classes into infiltration modifier factors.

    Unmatched finite classes receive ``default`` for each modifier, while
    raster nodata remains nodata.

    Parameters
    ----------
    da_lulc : xr.DataArray
        Land-use or land-cover class raster.
    df_modifiers : pd.DataFrame
        Modifier table indexed by land-use class.
    default : float, optional
        Neutral value used for unmatched classes, by default 1.0.

    Returns
    -------
    xr.Dataset
        Raster layers for the surface, storage, and drainage factors.
    """
    df = _ensure_modifier_dataframe(df_modifiers)
    nodata = da_lulc.raster.nodata
    if nodata is not None and np.isfinite(nodata):
        df = df.copy()
        df.loc[nodata] = np.nan
    ds = da_lulc.raster.reclassify(df).astype(np.float32)

    finite = np.isfinite(da_lulc)
    valid = finite
    if nodata is not None and np.isfinite(nodata):
        valid = valid & (da_lulc != nodata)
    n_unmatched = int((valid & ~da_lulc.isin(df.index.values)).sum())
    if n_unmatched:
        logger.warning(
            f"{n_unmatched} cells hold land-use classes that are absent from the "
            f"modifier table; a neutral factor of {default} is applied there. "
            f"Check that the modifier table matches the land-use dataset."
        )

    for column in MODIFIER_COLUMNS:
        ds[column] = ds[column].fillna(default).where(finite).astype(np.float32)
    return ds


def _nan_nodata(ds: xr.Dataset) -> xr.Dataset:
    """Declare NaN as nodata on every variable.

    Resampling only skips NaN when it is the declared nodata; arithmetic can drop
    the attribute (older xarray drops attrs by default), so set it on the output.
    """
    for name in ds.data_vars:
        da = ds[name]
        da.raster.set_nodata(np.nan)
        ds[name] = da
    return ds


def mask_invalid_ksat(da_ksat: xr.DataArray) -> xr.DataArray:
    """Return Ksat as float32 with nodata and negative values set to NaN.

    Ksat is expected in mm/hr, the unit SFINCS reads; convert other units in
    the data catalog (e.g. ``unit_mult: 3.6`` for um/s). Missing values are
    kept as NaN nodata so resampling skips them instead of pulling cell
    averages down; active cells left without data are filled (with a warning)
    when the layers are set on the model grid.

    Parameters
    ----------
    da_ksat : xr.DataArray
        Saturated hydraulic conductivity raster [mm/hr].

    Returns
    -------
    xr.DataArray
        Ksat with NaN as the declared nodata, so resampling skips those pixels.
    """
    if da_ksat.raster.nodata is not None:
        da_ksat = da_ksat.raster.mask_nodata()
    # negative Ksat is physically impossible, so it marks undeclared nodata such
    # as a raw -9999 that was never converted to NaN
    da_ksat = da_ksat.astype(np.float32)
    da_ksat = da_ksat.where(da_ksat >= 0.0)
    da_ksat.raster.set_nodata(np.nan)
    return da_ksat


def normalize_hsg_codes(
    da_hsg: xr.DataArray,
    mode: str = "drained",
) -> xr.DataArray:
    """Normalize single- and dual-HSG classes to the requested convention.

    In ``drained`` mode, dual classes 5-8 map to drained classes 1-4. Other
    finite values not represented by the mapping become NaN. ``"native"``
    leaves class values unchanged.

    Parameters
    ----------
    da_hsg : xr.DataArray
        Raster of hydrologic soil group classes.
    mode : {"drained", "native"}, optional
        Normalization convention, by default ``"drained"``.

    Returns
    -------
    xr.DataArray
        HSG raster normalized to the requested convention.
    """
    if mode == "native":
        return da_hsg.astype(np.float32)
    if mode != "drained":
        raise ValueError("dual_hsg must be either 'native' or 'drained'.")

    nodata = da_hsg.raster.nodata
    present_values = [value.item() for value in np.unique(da_hsg.values)]
    unmapped_values = [
        value
        for value in present_values
        if np.isfinite(value)
        and value not in DUAL_HSG_DRAINED_MAPPING
        and (nodata is None or value != nodata)
    ]
    if unmapped_values:
        logger.warning(
            "Unsupported HSG codes [1-8] not recognized by the HSG mapping "
            f"{unmapped_values}. These values will be treated as missing."
        )

    da_norm = xr.full_like(da_hsg, np.nan, dtype=np.float32)
    for source_value, target_value in DUAL_HSG_DRAINED_MAPPING.items():
        if source_value in present_values:
            logger.info(
                f"Mapping HSG value {source_value} to drained value {target_value}"
            )
        da_norm = da_norm.where(da_hsg != source_value, np.float32(target_value))
    return da_norm.where(np.isfinite(da_hsg))


def _reclassify_hsg(
    da_hsg: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    require_dual_hsg: bool = False,
) -> xr.Dataset:
    """Map HSG classes to parameter layers and report unmapped finite values.

    The raster's declared nodata value is preserved during reclassification.

    Parameters
    ----------
    da_hsg : xr.DataArray
        Raster of hydrologic soil group classes.
    df_map : pd.DataFrame
        Parameter table indexed by HSG class.
    require_dual_hsg : bool, optional
        Raise an error when native dual-HSG values 5-8 are present in the
        raster but absent from the parameter table.

    Returns
    -------
    xr.Dataset
        Reclassified parameter rasters, one variable per table column.
    """
    nodata = da_hsg.raster.nodata
    if nodata is not None and np.isfinite(nodata):
        df_map = df_map.copy()
        if nodata not in df_map.index:
            df_map.loc[nodata] = np.nan
        da_hsg = da_hsg.fillna(nodata)
    # add logger warning for unmapped HSG values
    unique_values = np.unique(da_hsg.values).tolist()
    unmapped_values = [
        v for v in unique_values if v not in df_map.index and np.isfinite(v)
    ]
    unmapped_dual_hsg = sorted(set(unmapped_values) & set(DUAL_HSG_DRAINED_MAPPING))
    if require_dual_hsg and unmapped_dual_hsg:
        raise ValueError(
            "Native dual-HSG values are present in the raster but missing from "
            f"the reclassification table: {unmapped_dual_hsg}."
        )
    if unmapped_values:
        logger.warning(
            "Recognized HSG codes missing from the reclassification table: "
            f"{unmapped_values}."
        )
    return da_hsg.raster.reclassify(df_map).astype(np.float32)


def cn_to_s(da_cn, da_mask=None, nodata=-9999, output_unit="inch", fill_cn=100.0):
    """Convert curve numbers to potential maximum soil moisture retention S.

    Missing curve numbers are treated as ``fill_cn`` (by default 100, so S = 0),
    and curve numbers are floored at 1. Retention is rounded to 0.001 inches
    before optional conversion to metres.

    Parameters
    ----------
    da_cn : xr.DataArray
        Curve-number raster.
    da_mask : xr.DataArray, optional
        Boolean or binary mask; cells outside the mask are set to ``nodata``.
    nodata : float, optional
        Output nodata value, by default -9999.
    output_unit : {"inch", "m"}, optional
        Output unit for S, by default ``"inch"``.
    fill_cn : float or None, optional
        Curve number used for missing input, by default 100 (S = 0). None keeps
        missing input missing, e.g. to skip it in a later area average.

    Returns
    -------
    xr.DataArray
        Potential maximum soil moisture retention S in the requested unit.
    """
    # missing CN is mapped to fill_cn (default 100, zero infiltration); CN >= 1
    da_cn = da_cn.raster.mask_nodata()
    if fill_cn is not None:
        da_cn = da_cn.fillna(fill_cn)
    da_cn = np.maximum(1, da_cn)
    da_s = np.maximum(1000 / da_cn - 10, 0).round(3)
    if output_unit == "m":
        da_s = da_s * INCH_TO_METER
    elif output_unit != "inch":
        raise ValueError("output_unit must be either 'inch' or 'm'.")
    if da_mask is not None:
        da_s = da_s.where(da_mask, nodata)
    try:
        da_s.raster.set_nodata(nodata)
    except (AttributeError, ValueError):
        logger.debug("Could not set nodata on curve-number retention", exc_info=True)
    return da_s


def curve_number_from_landuse_hsg(
    da_lulc: xr.DataArray,
    da_hsg: xr.DataArray,
    df_map: pd.DataFrame,
) -> xr.DataArray:
    """Map land-use and HSG classes to curve numbers.

    HSG values are nearest-neighbour reprojected onto the land-use grid before
    the two classes are looked up in ``df_map``.

    Parameters
    ----------
    da_lulc : xr.DataArray
        Land-use or land-cover class raster; defines the output grid.
    da_hsg : xr.DataArray
        Hydrologic soil group class raster.
    df_map : pd.DataFrame
        Curve-number table indexed by land-use class, with HSG classes as
        columns.

    Returns
    -------
    xr.DataArray
        Curve numbers on the land-use grid. A table value of 0 marks a class
        without infiltration (open water) and is returned as CN 100 (S = 0);
        unmapped combinations and negative table values (no value) are NaN,
        the declared nodata.
    """
    da_cn = xr.full_like(da_lulc, np.nan, dtype=np.float32).rename("cn")

    # Interpolate soil type to landuse
    da_hsg_to_lulc = da_hsg.raster.reproject_like(da_lulc, method="nearest")

    for landuse_value, row in df_map.iterrows():
        for hsg_value in df_map.columns:
            mask = (da_lulc == landuse_value) & (da_hsg_to_lulc == int(hsg_value))
            da_cn = da_cn.where(~mask, np.float32(row[hsg_value]))
    # negative table values (e.g. -9999) mean "no value"
    da_cn = da_cn.where(da_cn >= 0.0)
    # CN 0 marks open water: no infiltration, so it counts as impervious data
    # (S = 0) in any later aggregation rather than being skipped as missing
    da_cn = da_cn.where(da_cn != 0.0, np.float32(100.0))
    # full_like copies the land-use nodata (e.g. 255), but missing pixels are
    # NaN; declare NaN as nodata so resampling skips them instead of returning
    # NaN for a cell depending on where the NaN pixels sit
    da_cn.raster.set_nodata(np.nan)
    return da_cn


def adjust_curve_number(
    da_cn: xr.DataArray,
    antecedent_moisture: str | None = "avg",
) -> xr.DataArray:
    """Adjust CN-II values to a requested antecedent moisture condition.

    Average-condition values are returned unchanged. Dry and wet values are
    converted using the standard empirical CN relationships and clipped to
    [0, 100].

    Parameters
    ----------
    da_cn : xr.DataArray
        Curve-number raster for average antecedent moisture.
    antecedent_moisture : {"avg", "dry", "wet"} or None, optional
        Target condition, by default ``"avg"``.

    Returns
    -------
    xr.DataArray
        Curve numbers adjusted to the requested condition.
    """
    if antecedent_moisture in (None, "avg"):
        return da_cn.astype(np.float32)
    if antecedent_moisture == "dry":
        da_adj = da_cn / (2.281 - 0.01281 * da_cn)
    elif antecedent_moisture == "wet":
        da_adj = da_cn / (0.427 + 0.00573 * da_cn)
    else:
        raise ValueError(
            "antecedent_moisture must be one of None, 'avg', 'dry', or 'wet'."
        )
    da_adj = (
        da_adj.clip(min=0.0, max=100.0).where(np.isfinite(da_cn)).astype(np.float32)
    )
    # arithmetic can drop the nodata attribute; restore it so resampling skips NaN
    da_adj.raster.set_nodata(da_cn.raster.nodata)
    return da_adj


def curve_number_with_recovery(
    da_lulc: xr.DataArray,
    da_hsg: xr.DataArray,
    da_ksat: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    effective: float,
    ksat_max: float | None = 360.0,
    da_mask: xr.DataArray | None = None,
) -> xr.Dataset:
    """Estimate SCS curve-number-with-recovery parameters.

    Curve numbers are derived from land use and HSG, converted to maximum soil
    retention, then combined with Ksat to calculate effective retention and
    recovery rate. Inputs must already be-read xarray objects and a DataFrame.

    Parameters
    ----------
    da_lulc : xr.DataArray
        Land-use or land-cover class raster.
    da_hsg : xr.DataArray
        Hydrologic soil group class raster.
    da_ksat : xr.DataArray
        Saturated hydraulic conductivity raster [mm/hr].
    df_map : pd.DataFrame
        Curve-number table indexed by land-use class, with HSG classes as
        columns.
    effective : float
        Fraction of ``smax`` used for effective retention, e.g. 0.5 for 50%.
    ksat_max : float or None, optional
        Upper cap for Ksat in mm/hr; None
        disables the cap. By default 360.0 (100 um/s), high enough to only
        remove outliers.

    Returns
    -------
    xr.Dataset
        ``smax`` and ``seff`` in metres and ``ks`` in mm/hr. Cells without
        valid Ksat stay NaN in ``ks``.
    """
    # Derive curve number from land use and HSG
    da_cn = curve_number_from_landuse_hsg(da_lulc, da_hsg, df_map)
    # Convert CN to maximum soil retention (S) model grid and interpolate
    # open water arrives as CN 100 (S = 0), a real value that the area average
    # must include; unmapped pixels stay NaN nodata and are skipped
    da_smax = cn_to_s(da_cn, output_unit="m", nodata=np.nan, fill_cn=None)
    da_smax = da_smax.astype(np.float32)
    da_smax.raster.set_nodata(np.nan)
    da_smax.name = "smax"
    # invalid Ksat pixels become NaN nodata on the source grid
    # so the average below skips them instead of counting them as zero
    da_ks = mask_invalid_ksat(da_ksat)
    # Reproject to mask if provided
    if da_mask is not None:
        da_smax = da_smax.raster.reproject_like(da_mask, method="average").fillna(0.0)
        da_ks = da_ks.raster.reproject_like(da_mask, method="average")
    da_ks = da_ks.astype(np.float32)
    da_ks.name = "ks"
    if ksat_max is not None:  # cap in mm/hr
        da_ks = np.minimum(da_ks, ksat_max)
    # Compute effective soil retention based on the maximum soil retention and the effective fraction
    da_seff = (da_smax * effective).astype(np.float32)
    da_seff.name = "seff"
    return _nan_nodata(xr.Dataset({"smax": da_smax, "seff": da_seff, "ks": da_ks}))


def constant_infiltration_from_ksat_lulc(
    da_ksat: xr.DataArray,
    da_lulc: xr.DataArray,
    df_modifiers: pd.DataFrame,
    *,
    da_mask: xr.DataArray | None = None,
    qinf_min: float = 0.01,
    qinf_max: float = 19.9,
    base_min: float = 0.1,
    base_max: float = 15.0,
) -> xr.DataArray:
    """Estimate spatially varying constant infiltration from Ksat and land use.

    Ksat [mm/hr] is normalized between its 5th and 95th
    percentiles, taken over ``da_mask`` cells when given and otherwise over all
    valid pixels. The normalized value is scaled by the land-use surface
    modifier. ``da_ksat``, ``da_lulc`` and ``da_mask`` must share one grid.
    Inputs must already be-read xarray objects and a DataFrame.

    Parameters
    ----------
    da_ksat : xr.DataArray
        Saturated hydraulic conductivity raster [mm/hr].
    da_lulc : xr.DataArray
        Land-use or land-cover class raster.
    df_modifiers : pd.DataFrame
        Modifier table indexed by land-use class.
    da_mask : xr.DataArray, optional
        Active-cell mask used to determine Ksat percentiles.
    qinf_min, qinf_max : float, optional
        Minimum and maximum output infiltration rates, by default 0.01 and
        19.9.
    base_min, base_max : float, optional
        Infiltration range before applying the land-use modifier, by default
        0.1 and 15.0.

    Returns
    -------
    xr.DataArray
        Spatially varying constant infiltration rate.
    """
    df_modifiers = _ensure_modifier_dataframe(df_modifiers)
    surface_factor = _modifier_layers(da_lulc, df_modifiers)["surface_factor"]

    # nodata and negative Ksat stay NaN; only valid values are floored for the log
    ks_values = mask_invalid_ksat(da_ksat).values
    ks_values = np.where(np.isfinite(ks_values), np.maximum(ks_values, 0.01), np.nan)
    log_values = np.where(np.isfinite(ks_values), np.log10(ks_values), np.nan).astype(
        np.float32
    )

    valid = np.isfinite(log_values)
    if da_mask is not None:
        valid &= np.asarray(da_mask.values) > 0
    valid_values = log_values.astype(float)[valid]
    if valid_values.size == 0:
        raise ValueError("No finite active Ksat values available to estimate qinf.")
    p5 = np.nanpercentile(valid_values, 5.0)
    p95 = np.nanpercentile(valid_values, 95.0)
    if not np.isfinite(p5) or not np.isfinite(p95):
        raise ValueError("Could not derive finite active-domain Ksat percentiles.")
    if p95 <= p5:
        norm_values = np.where(np.isfinite(log_values), 0.5, np.nan).astype(np.float32)
    else:
        norm_values = np.clip((log_values - p5) / (p95 - p5), 0.0, 1.0).astype(
            np.float32
        )

    surface_values = np.asarray(surface_factor.values).astype(np.float32)
    qinf_base = (base_min + norm_values * (base_max - base_min)).astype(np.float32)
    qinf_values = qinf_base * surface_values
    qinf_values = np.where(
        np.isfinite(surface_values) & np.isfinite(log_values),
        np.clip(qinf_values, qinf_min, qinf_max),
        np.nan,
    ).astype(np.float32)
    da_qinf = surface_factor.copy(data=qinf_values).rename("qinf")
    da_qinf.raster.set_nodata(np.nan)
    return da_qinf


def green_ampt_from_hsg(
    da_hsg: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    dual_hsg: str = "drained",
) -> xr.Dataset:
    """Estimate Green-Ampt parameters from HSG classes and optional Ksat.

    ``psi`` and ``sigma`` are reclassified from ``df_map``. ``ks`` is derived
    from ``da_ksat`` when supplied; otherwise the table must include ``ks``.
    Inputs must already be-read xarray objects and a DataFrame.

    Parameters
    ----------
    da_hsg : xr.DataArray
        Soil hydrologic group classification.
    df_map : pd.DataFrame
        Mapping from HSG classes to Green-Ampt parameters psi, sigma (and optionally ks).
    da_ksat : xr.DataArray, optional
        Saturated hydraulic conductivity; when supplied it overrides table
        ``ks`` [mm/hr], by default None.
    dual_hsg : {"drained", "native"}, optional
        HSG normalization mode, by default ``"drained"``. Native dual-HSG
        classes require corresponding rows in ``df_map``.

    Returns
    -------
    xr.Dataset
        Green-Ampt parameters (psi, sigma, ks) for each grid cell.
    """
    da_hsg = normalize_hsg_codes(da_hsg, mode=dual_hsg)
    ds = _reclassify_hsg(
        da_hsg,
        df_map,
        require_dual_hsg=dual_hsg == "native",
    )
    if da_ksat is not None:
        ds["ks"] = mask_invalid_ksat(da_ksat)
    if "ks" not in ds:
        raise ValueError(
            "Green-Ampt estimation requires either hydraulic conductivity (ks)."
        )
    required = {"psi", "sigma", "ks"}
    missing = required.difference(ds.data_vars)
    if missing:
        raise ValueError(f"Missing Green-Ampt parameter columns: {sorted(missing)}")
    return _nan_nodata(ds[list(sorted(required, key=("psi", "sigma", "ks").index))])


def green_ampt_from_hsg_landuse(
    da_hsg: xr.DataArray,
    da_lulc: xr.DataArray,
    df_map: pd.DataFrame,
    df_modifiers: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    dual_hsg: str = "drained",
) -> xr.Dataset:
    """Estimate Green-Ampt parameters from soil classes, Ksat, and land use.

    Starts with :func:`green_ampt_from_hsg`, then scales ``sigma`` by the
    storage modifier and ``ks`` by the surface modifier. Inputs must already
    be-read xarray objects and DataFrames.

    Parameters
    ----------
    da_hsg : xr.DataArray
        Hydrologic soil group class raster.
    da_lulc : xr.DataArray
        Land-use or land-cover class raster.
    df_map : pd.DataFrame
        Green-Ampt parameter table indexed by HSG class.
    df_modifiers : pd.DataFrame
        Land-use modifier table indexed by class.
    da_ksat : xr.DataArray, optional
        Saturated hydraulic conductivity raster [mm/hr]; overrides table ``ks``.
    dual_hsg : {"drained", "native"}, optional
        HSG normalization mode, by default ``"drained"``.

    Returns
    -------
    xr.Dataset
        Green-Ampt ``psi``, ``sigma``, and ``ks`` rasters.
    """
    ds = green_ampt_from_hsg(
        da_hsg,
        df_map,
        da_ksat=da_ksat,
        dual_hsg=dual_hsg,
    )
    factors = _modifier_layers(da_lulc, df_modifiers)
    storage_factor = factors["storage_factor"]
    surface_factor = factors["surface_factor"]
    ds["sigma"] = (ds["sigma"] * storage_factor).astype(np.float32)
    ds["ks"] = (ds["ks"] * surface_factor).astype(np.float32)
    return _nan_nodata(ds[["psi", "sigma", "ks"]])


def horton_from_hsg(
    da_hsg: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    dual_hsg: str = "drained",
) -> xr.Dataset:
    """Estimate Horton parameters from HSG classes and optional Ksat.

    Reclassifies ``kd`` and either ``fc``/``f0`` or ``fc_scale``/``f0_scale``.
    If Ksat is supplied, it determines ``fc`` and the optional ``fc_scale`` is
    applied. Inputs must already be-read xarray objects and a DataFrame.

    Parameters
    ----------
    da_hsg : xr.DataArray
        Hydrologic soil group class raster.
    df_map : pd.DataFrame
        Horton parameter table indexed by HSG class.
    da_ksat : xr.DataArray, optional
        Saturated hydraulic conductivity raster [mm/hr] used to derive ``fc``.
    dual_hsg : {"drained", "native"}, optional
        HSG normalization mode, by default ``"drained"``. Native dual-HSG
        classes require corresponding rows in ``df_map``.

    Returns
    -------
    xr.Dataset
        Horton ``f0``, ``fc``, and ``kd`` rasters.
    """
    da_hsg = normalize_hsg_codes(da_hsg, mode=dual_hsg)
    ds = _reclassify_hsg(
        da_hsg,
        df_map,
        require_dual_hsg=dual_hsg == "native",
    )
    if da_ksat is not None:
        da_fc = mask_invalid_ksat(da_ksat)
        if "fc_scale" in ds:
            da_fc = da_fc * ds["fc_scale"]
        ds["fc"] = da_fc.astype(np.float32)
    if "fc" not in ds:
        raise ValueError("Horton estimation requires fc or ksat input.")
    if "f0" not in ds:
        if "f0_scale" not in ds:
            raise ValueError("Horton estimation requires f0 or f0_scale input.")
        ds["f0"] = (ds["fc"] * ds["f0_scale"]).astype(np.float32)
    if "kd" not in ds:
        raise ValueError("Horton estimation requires kd values.")
    return _nan_nodata(ds[["f0", "fc", "kd"]])


def horton_from_hsg_landuse(
    da_hsg: xr.DataArray,
    da_lulc: xr.DataArray,
    df_map: pd.DataFrame,
    df_modifiers: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    dual_hsg: str = "drained",
) -> xr.Dataset:
    """Estimate Horton parameters from soil classes, Ksat, and land use.

    Starts with :func:`horton_from_hsg`, then applies land-use surface,
    storage, and drainage modifiers. Inputs must already be-read xarray
    objects and DataFrames.

    Parameters
    ----------
    da_hsg : xr.DataArray
        Hydrologic soil group class raster.
    da_lulc : xr.DataArray
        Land-use or land-cover class raster.
    df_map : pd.DataFrame
        Horton parameter table indexed by HSG class.
    df_modifiers : pd.DataFrame
        Land-use modifier table indexed by class.
    da_ksat : xr.DataArray, optional
        Saturated hydraulic conductivity raster [mm/hr] used to derive ``fc``.
    dual_hsg : {"drained", "native"}, optional
        HSG normalization mode, by default ``"drained"``.

    Returns
    -------
    xr.Dataset
        Horton ``f0``, ``fc``, and ``kd`` rasters.
    """
    ds = horton_from_hsg(
        da_hsg,
        df_map,
        da_ksat=da_ksat,
        dual_hsg=dual_hsg,
    )
    factors = _modifier_layers(da_lulc, df_modifiers)
    surface_factor = factors["surface_factor"]
    storage_factor = factors["storage_factor"]
    drainage_factor = factors["drainage_factor"]
    f0_factor = xr.where(
        surface_factor >= storage_factor, surface_factor, storage_factor
    )
    ds["fc"] = (ds["fc"] * surface_factor).astype(np.float32)
    ds["f0"] = (ds["f0"] * f0_factor).astype(np.float32)
    ds["kd"] = (ds["kd"] * drainage_factor).astype(np.float32)
    return _nan_nodata(ds[["f0", "fc", "kd"]])


def bucket_from_hsg(
    da_hsg: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    bucket_loss: float | None = None,
    dual_hsg: str = "drained",
) -> xr.Dataset:
    """Estimate bucket parameters from HSG classes and optional Ksat.

    Storage is derived from ``storage_depth_mm * effective_fraction`` when
    needed. Drainage is derived from Ksat or ``residence_time_hr`` if the
    table does not provide ``bucket_k``. Inputs must already be-read xarray
    objects and a DataFrame.

    Parameters
    ----------
    da_hsg : xr.DataArray
        Hydrologic soil group class raster.
    df_map : pd.DataFrame
        Bucket parameter table indexed by HSG class.
    da_ksat : xr.DataArray, optional
        Saturated hydraulic conductivity raster [mm/hr] used to derive ``bucket_k``.
    bucket_loss : float, optional
        Loss fraction used when the table has no ``bucket_loss`` column.
    dual_hsg : {"drained", "native"}, optional
        HSG normalization mode, by default ``"drained"``. Native dual-HSG
        classes require corresponding rows in ``df_map``.

    Returns
    -------
    xr.Dataset
        Bucket ``bucket_smax``, ``bucket_k``, and ``bucket_loss`` rasters.
    """
    da_hsg = normalize_hsg_codes(da_hsg, mode=dual_hsg)
    ds = _reclassify_hsg(
        da_hsg,
        df_map,
        require_dual_hsg=dual_hsg == "native",
    )
    if "bucket_smax" not in ds:
        if not {"storage_depth_mm", "effective_fraction"}.issubset(ds.data_vars):
            raise ValueError(
                "Bucket estimation requires bucket_smax or storage_depth_mm and effective_fraction."
            )
        ds["bucket_smax"] = (ds["storage_depth_mm"] * ds["effective_fraction"]).astype(
            np.float32
        )
    if "bucket_k" not in ds:
        if da_ksat is None:
            if "residence_time_hr" not in ds:
                raise ValueError(
                    "Bucket estimation requires bucket_k, residence_time_hr, or ksat."
                )
            ds["bucket_k"] = xr.where(
                ds["residence_time_hr"] > 0.0,
                1.0 / ds["residence_time_hr"],
                0.0,
            ).astype(np.float32)
        else:
            ks_mmhr = mask_invalid_ksat(da_ksat)
            drain_factor = ds["drain_factor"] if "drain_factor" in ds else 1.0
            residence_hours = xr.where(
                ks_mmhr > 0.0,
                np.maximum(ds["bucket_smax"] / ks_mmhr * drain_factor, 1.0),
                np.nan,
            )
            # zero Ksat means no drainage (k = 0); missing Ksat stays missing
            ds["bucket_k"] = xr.where(
                np.isfinite(residence_hours),
                1.0 / residence_hours,
                xr.where(ks_mmhr.isnull(), np.nan, 0.0),
            ).astype(np.float32)
    if "bucket_loss" not in ds:
        loss = 0.0 if bucket_loss is None else bucket_loss
        ds["bucket_loss"] = xr.full_like(da_hsg, np.float32(loss), dtype=np.float32)
    return _nan_nodata(ds[["bucket_smax", "bucket_k", "bucket_loss"]])


def bucket_from_hsg_landuse(
    da_hsg: xr.DataArray,
    da_lulc: xr.DataArray,
    df_map: pd.DataFrame,
    df_modifiers: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    dual_hsg: str = "drained",
    bucket_loss: float | None = 0.10,
) -> xr.Dataset:
    """Estimate bucket parameters from soil classes, Ksat, and land use.

    Starts with :func:`bucket_from_hsg`, then scales storage and drainage by
    land-use modifiers. Inputs must already be-read xarray objects and
    DataFrames.

    Parameters
    ----------
    da_hsg : xr.DataArray
        Hydrologic soil group class raster.
    da_lulc : xr.DataArray
        Land-use or land-cover class raster.
    df_map : pd.DataFrame
        Bucket parameter table indexed by HSG class.
    df_modifiers : pd.DataFrame
        Land-use modifier table indexed by class.
    da_ksat : xr.DataArray, optional
        Saturated hydraulic conductivity raster [mm/hr] used to derive ``bucket_k``.
    dual_hsg : {"drained", "native"}, optional
        HSG normalization mode, by default ``"drained"``.
    bucket_loss : float or None, optional
        Loss fraction, by default 0.10. ``None`` also selects 0.10.

    Returns
    -------
    xr.Dataset
        Bucket ``bucket_smax``, ``bucket_k``, and ``bucket_loss`` rasters.
    """
    ds = bucket_from_hsg(
        da_hsg,
        df_map,
        da_ksat=da_ksat,
        bucket_loss=bucket_loss,
        dual_hsg=dual_hsg,
    )
    factors = _modifier_layers(da_lulc, df_modifiers)
    storage_factor = factors["storage_factor"]
    drainage_factor = factors["drainage_factor"]
    ds["bucket_smax"] = (ds["bucket_smax"] * storage_factor).astype(np.float32)
    ds["bucket_k"] = (ds["bucket_k"] * drainage_factor).astype(np.float32)
    loss_value = 0.10 if bucket_loss is None else bucket_loss
    ds["bucket_loss"] = xr.full_like(
        ds["bucket_smax"], np.float32(loss_value), dtype=np.float32
    )
    return _nan_nodata(ds[["bucket_smax", "bucket_k", "bucket_loss"]])
