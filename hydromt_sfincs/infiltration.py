"""Shared infiltration metadata and helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import xarray as xr

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

REGULAR_GRID_VARS = tuple(
    name for name in VARIABLES if not name.startswith("bucket_")
)
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
    if all(config.get(key) not in (None, "none") for key in ("smaxfile", "sefffile", "ksfile")):
        return "cnb"
    if all(config.get(key) not in (None, "none") for key in ("psifile", "sigmafile", "ksfile")):
        return "gai"
    if all(config.get(key) not in (None, "none") for key in ("f0file", "fcfile", "kdfile")):
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
