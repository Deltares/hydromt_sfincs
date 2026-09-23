"""Shared infiltration metadata, configuration helpers, and estimation workflows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "ALL_VARS",
    "BUCKET_VARS",
    "DEFAULT_BUCKETFILE",
    "DEFAULT_INFILTRATIONFILE",
    "FLAVORS",
    "INCH_TO_METER",
    "MICROMETER_PER_SECOND_TO_MM_PER_HOUR",
    "NON_BUCKET_VARS",
    "REGULAR_GRID_VARS",
    "VARIABLES",
    "adjust_curve_number",
    "binary_vars",
    "bucket_from_soil_landuse",
    "bucket_from_soil",
    "classify_nlcd_groups",
    "clear_data",
    "cn_to_s",
    "configure",
    "configured_flavor",
    "constant_infiltration_from_ksat_lulc",
    "curve_number_from_landuse_hsg",
    "curve_number_with_recovery",
    "flavor_variables",
    "get_attrs",
    "green_ampt_from_soil_landuse",
    "green_ampt_from_soil",
    "horton_from_soil_landuse",
    "horton_from_soil",
    "InfiltrationFlavor",
    "InfiltrationVariable",
    "ksat_to_mmhr",
    "nlcd_modifier_layer",
    "normalize_hsg_codes",
    "regular_active_vector",
    "regular_vector_to_da",
    "reset_config",
    "sidecar_dataset",
]

INCH_TO_METER = 0.0254
MICROMETER_PER_SECOND_TO_MM_PER_HOUR = 3.6
DEFAULT_INFILTRATIONFILE = "sfincs.infiltration.nc"
DEFAULT_BUCKETFILE = "sfincs.bucket.nc"


@dataclass(frozen=True)
class InfiltrationVariable:
    """Metadata for a supported infiltration variable."""

    name: str
    config_key: str | None
    default_filename: str | None
    standard_name: str
    unit: str
    fill_value: float = -9999.0


@dataclass(frozen=True)
class InfiltrationFlavor:
    """Configuration for an infiltration flavor."""

    code: str
    variables: tuple[str, ...]


VARIABLES: "OrderedDict[str, InfiltrationVariable]" = OrderedDict(
    [
        (
            "qinf",
            InfiltrationVariable(
                name="qinf",
                config_key="qinffile",
                default_filename="sfincs.qinf",
                standard_name="infiltration rate",
                unit="mm.hr-1",
            ),
        ),
        (
            "scs",
            InfiltrationVariable(
                name="scs",
                config_key="scsfile",
                default_filename="sfincs.scs",
                standard_name="potential soil moisture retention",
                unit="inch",
            ),
        ),
        (
            "smax",
            InfiltrationVariable(
                name="smax",
                config_key="smaxfile",
                default_filename="sfincs.smax",
                standard_name="potential maximum soil moisture retention",
                unit="m",
            ),
        ),
        (
            "seff",
            InfiltrationVariable(
                name="seff",
                config_key="sefffile",
                default_filename="sfincs.seff",
                standard_name="effective potential maximum soil moisture retention",
                unit="m",
            ),
        ),
        (
            "ks",
            InfiltrationVariable(
                name="ks",
                config_key="ksfile",
                default_filename="sfincs.ks",
                standard_name="saturated hydraulic conductivity",
                unit="mm.hr-1",
            ),
        ),
        (
            "psi",
            InfiltrationVariable(
                name="psi",
                config_key="psifile",
                default_filename="sfincs.psi",
                standard_name="wetting front suction head",
                unit="mm",
            ),
        ),
        (
            "sigma",
            InfiltrationVariable(
                name="sigma",
                config_key="sigmafile",
                default_filename="sfincs.sigma",
                standard_name="soil moisture deficit",
                unit="-",
            ),
        ),
        (
            "f0",
            InfiltrationVariable(
                name="f0",
                config_key="f0file",
                default_filename="sfincs.f0",
                standard_name="initial infiltration capacity",
                unit="mm.hr-1",
            ),
        ),
        (
            "fc",
            InfiltrationVariable(
                name="fc",
                config_key="fcfile",
                default_filename="sfincs.fc",
                standard_name="asymptotic infiltration capacity",
                unit="mm.hr-1",
            ),
        ),
        (
            "kd",
            InfiltrationVariable(
                name="kd",
                config_key="kdfile",
                default_filename="sfincs.kd",
                standard_name="horton decay coefficient",
                unit="hr-1",
            ),
        ),
        (
            "bucket_smax",
            InfiltrationVariable(
                name="bucket_smax",
                config_key=None,
                default_filename=None,
                standard_name="bucket maximum storage",
                unit="mm",
            ),
        ),
        (
            "bucket_k",
            InfiltrationVariable(
                name="bucket_k",
                config_key=None,
                default_filename=None,
                standard_name="bucket drainage coefficient",
                unit="hr-1",
            ),
        ),
        (
            "bucket_loss",
            InfiltrationVariable(
                name="bucket_loss",
                config_key=None,
                default_filename=None,
                standard_name="bucket loss fraction",
                unit="-",
            ),
        ),
    ]
)

FLAVORS: dict[str, InfiltrationFlavor] = {
    "con": InfiltrationFlavor(code="con", variables=tuple()),
    "c2d": InfiltrationFlavor(code="c2d", variables=("qinf",)),
    "cna": InfiltrationFlavor(code="cna", variables=("scs",)),
    "cnb": InfiltrationFlavor(code="cnb", variables=("smax", "seff", "ks")),
    "gai": InfiltrationFlavor(code="gai", variables=("psi", "sigma", "ks")),
    "hor": InfiltrationFlavor(code="hor", variables=("f0", "fc", "kd")),
    "bkt": InfiltrationFlavor(
        code="bkt", variables=("bucket_smax", "bucket_k", "bucket_loss")
    ),
}

REGULAR_GRID_VARS = tuple(name for name in VARIABLES if not name.startswith("bucket_"))
NON_BUCKET_VARS = tuple(name for name in REGULAR_GRID_VARS)
BUCKET_VARS = FLAVORS["bkt"].variables
ALL_VARS = tuple(VARIABLES)


def get_attrs(name: str) -> dict[str, str]:
    """Return metadata attrs for an infiltration variable."""
    meta = VARIABLES[name]
    return {"standard_name": meta.standard_name, "unit": meta.unit}


def flavor_variables(flavor: str) -> tuple[str, ...]:
    """Return required variable names for a flavor."""
    return FLAVORS[flavor].variables


def binary_vars() -> tuple[str, ...]:
    """Return infiltration variables stored as regular-grid binary maps."""
    return REGULAR_GRID_VARS


def configured_flavor(config) -> str | None:
    """Infer the configured infiltration flavor from model config."""
    if config.get("bucketfile") not in (None, "none"):
        return "bkt"
    inffile = config.get("infiltrationfile")
    if inffile not in (None, "none"):
        return config.get("infiltrationtype")
    if config.get("qinf") not in (None, 0.0):
        return "con"
    if config.get("qinffile") not in (None, "none"):
        return "c2d"
    if config.get("scsfile") not in (None, "none"):
        return "cna"
    if all(
        config.get(key) not in (None, "none")
        for key in ("smaxfile", "sefffile", "ksfile")
    ):
        return "cnb"
    if all(
        config.get(key) not in (None, "none")
        for key in ("psifile", "sigmafile", "ksfile")
    ):
        return "gai"
    if all(
        config.get(key) not in (None, "none") for key in ("f0file", "fcfile", "kdfile")
    ):
        return "hor"
    return None


def clear_data(ds: xr.Dataset, keep: Iterable[str] = ()) -> xr.Dataset:
    """Drop infiltration variables except those in ``keep``."""
    keep = set(keep)
    drop = [name for name in ALL_VARS if name in ds and name not in keep]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def reset_config(config) -> None:
    """Remove all infiltration-related configuration except defaults."""
    config.set("qinf", None)
    config.set("infiltrationfile", None)
    config.set("infiltrationtype", None)
    config.set("bucketfile", None)
    config.set("bucket_loss_frac", None)
    for meta in VARIABLES.values():
        if meta.config_key is not None:
            config.set(meta.config_key, None)


def configure(config, flavor: str, grid_type: str) -> None:
    """Update model config for one infiltration flavor."""
    reset_config(config)
    if flavor == "con":
        return
    if flavor == "bkt":
        config.set("bucketfile", DEFAULT_BUCKETFILE)
        return
    if grid_type == "regular":
        for name in flavor_variables(flavor):
            config.set(VARIABLES[name].config_key, VARIABLES[name].default_filename)
    elif grid_type == "quadtree":
        config.set("infiltrationfile", DEFAULT_INFILTRATIONFILE)
        config.set("infiltrationtype", flavor)
    else:
        raise ValueError(f"Unsupported grid_type: {grid_type}")


def regular_active_vector(data: xr.DataArray, mask: xr.DataArray) -> np.ndarray:
    """Flatten active regular-grid cells in SFINCS order."""
    values = np.asarray(data.values, dtype=np.float32)
    mask_values = np.asarray(mask.values)
    return values.transpose()[mask_values.transpose() > 0]


def regular_vector_to_da(
    values: np.ndarray,
    mask: xr.DataArray,
    like: xr.DataArray,
    *,
    fill_value: float = -9999.0,
) -> xr.DataArray:
    """Map active-cell vectors to a full regular-grid data array."""
    data = np.full(mask.shape[::-1], fill_value, dtype=np.float32)
    data.flat[np.where(mask.values.ravel(order="F"))[0]] = np.asarray(
        values, dtype=np.float32
    )
    data = data.transpose()
    da = xr.DataArray(
        data=data,
        coords=like.coords,
        dims=like.dims,
        name=like.name,
        attrs={"_FillValue": fill_value},
    )
    try:
        da.raster.set_crs(mask.raster.crs)
        da.raster.set_nodata(fill_value)
    except Exception:
        pass
    return da


def sidecar_dataset(
    data: Mapping[str, np.ndarray | xr.DataArray],
    dim_size: int,
) -> xr.Dataset:
    """Create a minimal SFINCS netCDF sidecar dataset."""
    coords = {"mesh2d_nFaces": np.arange(dim_size, dtype=np.int32)}
    ds = xr.Dataset(coords=coords)
    for name, values in data.items():
        if hasattr(values, "values"):
            values = values.values
        ds[name] = xr.DataArray(
            np.asarray(values, dtype=np.float32),
            dims=("mesh2d_nFaces",),
            attrs=get_attrs(name),
        )
    return ds


NLCD_GROUPS = {
    "water": (11,),
    "urban_low": (21,),
    "urban_med": (22, 23),
    "urban_high": (24,),
    "barren": (31,),
    "forest": (41, 42, 43),
    "shrub_grass": (52, 71),
    "crops": (81, 82),
    "wetlands": (90, 95),
}

NLCD_GROUP_CODES = {
    group_name: index for index, group_name in enumerate(NLCD_GROUPS.keys(), start=1)
}

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


def _ensure_dataframe(df_map: pd.DataFrame) -> pd.DataFrame:
    return df_map.copy()


def _ensure_modifier_dataframe(df_map: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dataframe(df_map)
    if "nlcd_group" in df.columns:
        df = df.set_index("nlcd_group")
    required = {"surface_factor", "storage_factor", "drainage_factor"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing NLCD modifier columns: {sorted(missing)}")
    return df


def _as_values(data) -> np.ndarray:
    return np.asarray(getattr(data, "values", data))


def _like_with_values(
    like,
    values,
    *,
    dtype=np.float32,
    name: str | None = None,
):
    da = like.copy(deep=True)
    da.values = np.asarray(values, dtype=dtype)
    if name is not None:
        da.name = name
    return da


def _full_like(
    like: xr.DataArray,
    fill_value: float = np.nan,
    *,
    dtype=np.float32,
    name: str | None = None,
) -> xr.DataArray:
    try:
        da = xr.full_like(like, fill_value, dtype=dtype)
    except TypeError:
        da = _like_with_values(
            like,
            np.full_like(_as_values(like), fill_value, dtype=dtype),
            dtype=dtype,
        )
    if name is not None:
        da.name = name
    return da


def _modifier_layer(
    da_lulc: xr.DataArray,
    df_modifiers: pd.DataFrame,
    column: str,
    *,
    default: float = 1.0,
) -> xr.DataArray:
    da_groups = classify_nlcd_groups(da_lulc)
    da_factor = _full_like(
        da_lulc,
        np.float32(default),
        dtype=np.float32,
        name=column,
    )
    for group_name, group_code in NLCD_GROUP_CODES.items():
        if group_name not in df_modifiers.index:
            continue
        da_factor = da_factor.where(
            da_groups != group_code,
            np.float32(df_modifiers.loc[group_name, column]),
        )
    return da_factor.where(np.isfinite(da_lulc))


def nlcd_modifier_layer(
    da_lulc: xr.DataArray,
    df_modifiers: pd.DataFrame,
    column: str,
    *,
    default: float = 1.0,
) -> xr.DataArray:
    """Create an NLCD-based modifier layer from already-read inputs."""
    return _modifier_layer(
        da_lulc,
        _ensure_modifier_dataframe(df_modifiers),
        column,
        default=default,
    )


def _valid_values(
    da: xr.DataArray,
    da_mask: xr.DataArray | None = None,
) -> np.ndarray:
    values = _as_values(da).astype(float)
    valid = np.isfinite(values)
    if da_mask is not None:
        valid &= _as_values(da_mask) > 0
    return values[valid]


def ksat_to_mmhr(
    da_ksat: xr.DataArray,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
) -> xr.DataArray:
    """Convert saturated hydraulic conductivity to mm/hr."""
    da_ks = da_ksat.astype(np.float32) * factor_ksat
    return da_ks.fillna(0.0)


def normalize_hsg_codes(
    da_hsg: xr.DataArray,
    mode: str | None = "drained",
) -> xr.DataArray:
    """Normalize single- and dual-HSG classes to the requested convention."""
    if mode in (None, "none", "native"):
        return da_hsg.astype(np.float32)
    if mode != "drained":
        raise ValueError("dual_hsg must be one of None, 'native', or 'drained'.")

    da_norm = _full_like(da_hsg, np.nan, dtype=np.float32, name=da_hsg.name)
    for source_value, target_value in DUAL_HSG_DRAINED_MAPPING.items():
        da_norm = da_norm.where(da_hsg != source_value, np.float32(target_value))
    return da_norm.where(np.isfinite(da_hsg))


def classify_nlcd_groups(da_lulc: xr.DataArray) -> xr.DataArray:
    """Classify NLCD land-use codes into infiltration modifier groups."""
    da_groups = _full_like(da_lulc, 0, dtype=np.int16, name="nlcd_group")
    for group_name, nlcd_codes in NLCD_GROUPS.items():
        group_code = NLCD_GROUP_CODES[group_name]
        mask = da_lulc == nlcd_codes[0]
        for nlcd_code in nlcd_codes[1:]:
            mask = mask | (da_lulc == nlcd_code)
        da_groups = da_groups.where(~mask, np.int16(group_code))
    return da_groups.where(np.isfinite(da_lulc), 0)


def cn_to_s(da_cn, da_mask=None, nodata=-9999, output_unit="inch"):
    """Convert Curve Numbers to potential maximum soil moisture retention S."""
    # nodata is mapped to CN 100 (zero infiltration); CN is floored at 1
    da_cn = np.maximum(1, da_cn.raster.mask_nodata().fillna(100))
    da_s = np.maximum(1000 / da_cn - 10, 0).round(3)
    if output_unit == "m":
        da_s = da_s * INCH_TO_METER
    elif output_unit != "inch":
        raise ValueError("output_unit must be either 'inch' or 'm'.")
    if da_mask is not None:
        da_s = da_s.where(da_mask, nodata)
    try:
        da_s.raster.set_nodata(nodata)
    except Exception:
        pass
    return da_s


def curve_number_from_landuse_hsg(
    da_landuse: xr.DataArray,
    da_hsg: xr.DataArray,
    df_map: pd.DataFrame,
) -> xr.DataArray:
    """Map already-read land use and HSG rasters to curve numbers."""
    df_map = _ensure_dataframe(df_map)
    da_cn = _full_like(da_landuse, np.nan, name="cn")
    for landuse_value, row in df_map.iterrows():
        for hsg_value in df_map.columns:
            mask = (da_landuse == landuse_value) & (da_hsg == int(hsg_value))
            da_cn = da_cn.where(~mask, np.float32(row[hsg_value]))
    return da_cn.where(da_cn > 0.0)


def adjust_curve_number(
    da_cn: xr.DataArray,
    antecedent_moisture: str | None = "avg",
) -> xr.DataArray:
    """Adjust CN-II values to the requested antecedent moisture condition."""
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
    return da_adj.clip(min=0.0, max=100.0).where(np.isfinite(da_cn)).astype(np.float32)


def curve_number_with_recovery(
    da_landuse: xr.DataArray,
    da_hsg: xr.DataArray,
    da_ksat: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    effective: float,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
    ksat_max: float | None = 100.0,
    da_mask: xr.DataArray | None = None,
) -> xr.Dataset:
    """Estimate SCS curve-number-with-recovery parameters.

    All inputs must be already-read xarray objects and a pandas DataFrame.

    Parameters
    ----------
    effective : float
        Fraction of ``smax`` that is effective soil retention, e.g. 0.5 for 50%.
    factor_ksat : float, optional
        Factor converting Ksat to mm/hr, by default micrometer per second to mm/hr.
    ksat_max : float, optional
        Upper cap applied to raw Ksat before unit conversion, following the
        recovery-rate ranges of SWMM Table 4.7. Set to None to disable.
    da_mask : xr.DataArray, optional
        If given, outputs are reprojected onto this grid using 'average'.
    """
    da_cn = curve_number_from_landuse_hsg(da_landuse, da_hsg, df_map)
    da_cn = da_cn.where(da_cn > 0.0)
    da_smax = cn_to_s(da_cn, output_unit="m", nodata=0.0).astype(np.float32)
    da_smax.name = "smax"
    if ksat_max is not None:
        da_ksat = np.minimum(da_ksat, ksat_max)
    da_ks = ksat_to_mmhr(da_ksat, factor_ksat=factor_ksat).astype(np.float32)
    da_ks.name = "ks"
    if da_mask is not None:
        da_smax = da_smax.raster.reproject_like(da_mask, method="average").fillna(0.0)
        da_ks = da_ks.raster.reproject_like(da_mask, method="average").fillna(0.0)
    da_seff = (da_smax * effective).astype(np.float32)
    da_seff.name = "seff"
    return xr.Dataset({"smax": da_smax, "seff": da_seff, "ks": da_ks})


def constant_infiltration_from_ksat_lulc(
    da_ksat: xr.DataArray,
    da_lulc: xr.DataArray,
    df_modifiers: pd.DataFrame,
    *,
    da_mask: xr.DataArray | None = None,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
    qinf_min: float = 0.01,
    qinf_max: float = 19.9,
    base_min: float = 0.1,
    base_max: float = 15.0,
) -> xr.DataArray:
    """Estimate spatially varying constant infiltration from Ksat and land use.

    All inputs must be already-read xarray objects and a pandas DataFrame.
    """
    df_modifiers = _ensure_modifier_dataframe(df_modifiers)
    surface_factor = _modifier_layer(da_lulc, df_modifiers, "surface_factor")

    ks_values = _as_values(da_ksat).astype(np.float32) * factor_ksat
    ks_values = np.where(np.isfinite(ks_values), np.maximum(ks_values, 0.01), np.nan)
    log_values = np.where(np.isfinite(ks_values), np.log10(ks_values), np.nan).astype(
        np.float32
    )
    log_ks = _like_with_values(da_ksat, log_values, name="log_ks")

    valid_values = _valid_values(log_ks, da_mask=da_mask)
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

    surface_values = _as_values(surface_factor).astype(np.float32)
    qinf_base = (base_min + norm_values * (base_max - base_min)).astype(np.float32)
    qinf_values = qinf_base * surface_values
    qinf_values = np.where(
        np.isfinite(surface_values) & np.isfinite(log_values),
        np.clip(qinf_values, qinf_min, qinf_max),
        np.nan,
    ).astype(np.float32)
    da_qinf = _like_with_values(surface_factor, qinf_values, name="qinf")
    return da_qinf


def _reclassify(da_soil: xr.DataArray, df_map: pd.DataFrame) -> xr.Dataset:
    df_map = _ensure_dataframe(df_map)
    ds = da_soil.raster.reclassify(df_map).astype(np.float32)
    return ds


def green_ampt_from_soil(
    da_soil: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
) -> xr.Dataset:
    """Estimate Green-Ampt parameters from soil classes and optional Ksat.

    All inputs must be already-read xarray objects and a pandas DataFrame.
    """
    ds = _reclassify(da_soil, _ensure_dataframe(df_map))
    if da_ksat is not None:
        ds["ks"] = ksat_to_mmhr(da_ksat, factor_ksat=factor_ksat)
    if "ks" not in ds:
        raise ValueError("Green-Ampt estimation requires either ks or ksat input.")
    required = {"psi", "sigma", "ks"}
    missing = required.difference(ds.data_vars)
    if missing:
        raise ValueError(f"Missing Green-Ampt parameter columns: {sorted(missing)}")
    return ds[list(sorted(required, key=("psi", "sigma", "ks").index))]


def green_ampt_from_soil_landuse(
    da_soil: xr.DataArray,
    da_lulc: xr.DataArray,
    df_map: pd.DataFrame,
    df_modifiers: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
    dual_hsg: str | None = "drained",
) -> xr.Dataset:
    """Estimate Green-Ampt parameters from soil classes, Ksat, and land use.

    All inputs must be already-read xarray objects and pandas DataFrames.
    """
    da_soil = normalize_hsg_codes(da_soil, mode=dual_hsg)
    df_modifiers = _ensure_modifier_dataframe(df_modifiers)
    ds = green_ampt_from_soil(
        da_soil,
        df_map,
        da_ksat=da_ksat,
        factor_ksat=factor_ksat,
    )
    storage_factor = _modifier_layer(da_lulc, df_modifiers, "storage_factor")
    surface_factor = _modifier_layer(da_lulc, df_modifiers, "surface_factor")
    ds["sigma"] = (ds["sigma"] * storage_factor).astype(np.float32)
    ds["ks"] = (ds["ks"] * surface_factor).astype(np.float32)
    return ds[["psi", "sigma", "ks"]]


def horton_from_soil(
    da_soil: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
) -> xr.Dataset:
    """Estimate Horton parameters from soil classes and optional Ksat.

    All inputs must be already-read xarray objects and a pandas DataFrame.
    """
    ds = _reclassify(da_soil, _ensure_dataframe(df_map))
    if da_ksat is not None:
        da_fc = ksat_to_mmhr(da_ksat, factor_ksat=factor_ksat)
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
    return ds[["f0", "fc", "kd"]]


def horton_from_soil_landuse(
    da_soil: xr.DataArray,
    da_lulc: xr.DataArray,
    df_map: pd.DataFrame,
    df_modifiers: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
    dual_hsg: str | None = "drained",
) -> xr.Dataset:
    """Estimate Horton parameters from soil classes, Ksat, and land use.

    All inputs must be already-read xarray objects and pandas DataFrames.
    """
    da_soil = normalize_hsg_codes(da_soil, mode=dual_hsg)
    df_modifiers = _ensure_modifier_dataframe(df_modifiers)
    ds = horton_from_soil(
        da_soil,
        df_map,
        da_ksat=da_ksat,
        factor_ksat=factor_ksat,
    )
    surface_factor = _modifier_layer(da_lulc, df_modifiers, "surface_factor")
    storage_factor = _modifier_layer(da_lulc, df_modifiers, "storage_factor")
    drainage_factor = _modifier_layer(da_lulc, df_modifiers, "drainage_factor")
    f0_factor = xr.where(
        surface_factor >= storage_factor, surface_factor, storage_factor
    )
    ds["fc"] = (ds["fc"] * surface_factor).astype(np.float32)
    ds["f0"] = (ds["f0"] * f0_factor).astype(np.float32)
    ds["kd"] = (ds["kd"] * drainage_factor).astype(np.float32)
    return ds[["f0", "fc", "kd"]]


def bucket_from_soil(
    da_soil: xr.DataArray,
    df_map: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
    bucket_loss: float | None = None,
) -> xr.Dataset:
    """Estimate bucket parameters from soil classes and optional Ksat.

    All inputs must be already-read xarray objects and a pandas DataFrame.
    """
    ds = _reclassify(da_soil, _ensure_dataframe(df_map))
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
            ks_mmhr = ksat_to_mmhr(da_ksat, factor_ksat=factor_ksat)
            drain_factor = ds["drain_factor"] if "drain_factor" in ds else 1.0
            residence_hours = xr.where(
                ks_mmhr > 0.0,
                np.maximum(ds["bucket_smax"] / ks_mmhr * drain_factor, 1.0),
                np.nan,
            )
            ds["bucket_k"] = xr.where(
                np.isfinite(residence_hours), 1.0 / residence_hours, 0.0
            ).astype(np.float32)
    if "bucket_loss" not in ds:
        loss = 0.0 if bucket_loss is None else bucket_loss
        ds["bucket_loss"] = _full_like(
            da_soil, np.float32(loss), dtype=np.float32, name="bucket_loss"
        )
    return ds[["bucket_smax", "bucket_k", "bucket_loss"]]


def bucket_from_soil_landuse(
    da_soil: xr.DataArray,
    da_lulc: xr.DataArray,
    df_map: pd.DataFrame,
    df_modifiers: pd.DataFrame,
    *,
    da_ksat: xr.DataArray | None = None,
    factor_ksat: float = MICROMETER_PER_SECOND_TO_MM_PER_HOUR,
    dual_hsg: str | None = "drained",
    bucket_loss: float | None = 0.10,
) -> xr.Dataset:
    """Estimate bucket parameters from soil classes, Ksat, and land use.

    All inputs must be already-read xarray objects and pandas DataFrames.
    """
    da_soil = normalize_hsg_codes(da_soil, mode=dual_hsg)
    df_modifiers = _ensure_modifier_dataframe(df_modifiers)
    ds = bucket_from_soil(
        da_soil,
        df_map,
        da_ksat=da_ksat,
        factor_ksat=factor_ksat,
        bucket_loss=bucket_loss,
    )
    storage_factor = _modifier_layer(da_lulc, df_modifiers, "storage_factor")
    drainage_factor = _modifier_layer(da_lulc, df_modifiers, "drainage_factor")
    ds["bucket_smax"] = (ds["bucket_smax"] * storage_factor).astype(np.float32)
    ds["bucket_k"] = (ds["bucket_k"] * drainage_factor).astype(np.float32)
    loss_value = 0.10 if bucket_loss is None else bucket_loss
    ds["bucket_loss"] = _full_like(
        ds["bucket_smax"],
        np.float32(loss_value),
        dtype=np.float32,
        name="bucket_loss",
    )
    return ds[["bucket_smax", "bucket_k", "bucket_loss"]]
