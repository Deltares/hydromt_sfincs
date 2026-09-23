"""Infiltration metadata, configuration bookkeeping, and grid I/O helpers.

Shared by the regular-grid and quadtree infiltration components. The estimation
maths lives in :py:mod:`hydromt_sfincs.workflows.infiltration`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from hydromt_sfincs.components.config import SfincsConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ALL_VARS",
    "BUCKET_VARS",
    "DEFAULT_BUCKETFILE",
    "DEFAULT_INFILTRATIONFILE",
    "FLAVORS",
    "InfiltrationVariable",
    "VARIABLES",
    "clear_data",
    "configure",
    "configured_flavor",
    "flavor_variables",
    "get_attrs",
    "regular_active_vector",
    "regular_vector_to_da",
    "reset_config",
    "sidecar_dataset",
]

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


VARIABLES: dict[str, InfiltrationVariable] = {
    name: InfiltrationVariable(name, config_key, filename, standard_name, unit)
    for name, config_key, filename, standard_name, unit in [
        ("qinf", "qinffile", "sfincs.qinf", "infiltration rate", "mm.hr-1"),
        ("scs", "scsfile", "sfincs.scs", "potential soil moisture retention", "inch"),
        (
            "smax",
            "smaxfile",
            "sfincs.smax",
            "potential maximum soil moisture retention",
            "m",
        ),
        (
            "seff",
            "sefffile",
            "sfincs.seff",
            "effective potential maximum soil moisture retention",
            "m",
        ),
        ("ks", "ksfile", "sfincs.ks", "saturated hydraulic conductivity", "mm.hr-1"),
        ("psi", "psifile", "sfincs.psi", "wetting front suction head", "mm"),
        ("sigma", "sigmafile", "sfincs.sigma", "soil moisture deficit", "-"),
        ("f0", "f0file", "sfincs.f0", "initial infiltration capacity", "mm.hr-1"),
        ("fc", "fcfile", "sfincs.fc", "asymptotic infiltration capacity", "mm.hr-1"),
        ("kd", "kdfile", "sfincs.kd", "horton decay coefficient", "hr-1"),
        ("bucket_smax", None, None, "bucket maximum storage", "mm"),
        ("bucket_k", None, None, "bucket drainage coefficient", "hr-1"),
        ("bucket_loss", None, None, "bucket loss fraction", "-"),
    ]
}

FLAVORS: dict[str, tuple[str, ...]] = {
    "con": (),
    "c2d": ("qinf",),
    "cna": ("scs",),
    "cnb": ("smax", "seff", "ks"),
    "gai": ("psi", "sigma", "ks"),
    "hor": ("f0", "fc", "kd"),
    "bkt": ("bucket_smax", "bucket_k", "bucket_loss"),
}

BUCKET_VARS = FLAVORS["bkt"]
ALL_VARS = tuple(VARIABLES)


def get_attrs(name: str) -> dict[str, str]:
    """Return metadata attrs for an infiltration variable."""
    meta = VARIABLES[name]
    return {"standard_name": meta.standard_name, "unit": meta.unit}


def flavor_variables(flavor: str) -> tuple[str, ...]:
    """Return required variable names for a flavor."""
    return FLAVORS[flavor]


def configured_flavor(config: "SfincsConfig") -> str | None:
    """Infer the configured infiltration flavor from model config."""
    if config.get("bucketfile") not in (None, "none"):
        return "bkt"
    if config.get("infiltrationfile") not in (None, "none"):
        return config.get("infiltrationtype")

    file_flavors = {
        "c2d": ("qinffile",),
        "cna": ("scsfile",),
        "cnb": ("smaxfile", "sefffile", "ksfile"),
        "gai": ("psifile", "sigmafile", "ksfile"),
        "hor": ("f0file", "fcfile", "kdfile"),
    }
    matches = [
        flavor
        for flavor, keys in file_flavors.items()
        if all(config.get(key) not in (None, "none") for key in keys)
    ]
    if len(matches) > 1:
        logger.warning(
            f"Config matches multiple infiltration flavors {matches}; using "
            f"'{matches[0]}'. Check for leftover infiltration file entries."
        )
    if matches:
        return matches[0]
    # a uniform rate is only decisive when no parameter files are configured
    if config.get("qinf") not in (None, 0.0):
        return "con"
    return None


def clear_data(ds: xr.Dataset, keep: Iterable[str] = ()) -> xr.Dataset:
    """Drop infiltration variables except those in ``keep``."""
    keep = set(keep)
    drop = [name for name in ALL_VARS if name in ds and name not in keep]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def reset_config(config: "SfincsConfig") -> None:
    """Remove all infiltration-related configuration except defaults."""
    config.set("qinf", None)
    config.set("infiltrationfile", None)
    config.set("infiltrationtype", None)
    config.set("bucketfile", None)
    config.set("bucket_loss_frac", None)
    for meta in VARIABLES.values():
        if meta.config_key is not None:
            config.set(meta.config_key, None)


def configure(config: "SfincsConfig", flavor: str, grid_type: str) -> None:
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
    except (AttributeError, ValueError):
        logger.debug(f"Could not set crs/nodata on {like.name}", exc_info=True)
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
