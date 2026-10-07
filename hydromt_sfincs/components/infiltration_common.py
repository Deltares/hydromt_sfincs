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
    "DEFAULT_INFILTRATIONFILE",
    "FLAVORS",
    "InfiltrationVariable",
    "VARIABLES",
    "clear_data",
    "configure",
    "configured_flavor",
    "flavor_variables",
    "get_attrs",
    "reset_config",
]

DEFAULT_INFILTRATIONFILE = "sfincs.infiltration.nc"


@dataclass(frozen=True)
class InfiltrationVariable:
    """Metadata for a supported infiltration variable."""

    name: str
    config_key: str | None
    default_filename: str | None
    standard_name: str
    unit: str
    fill_value: float = 0.0
    """Real value written where an active cell has no input data."""


VARIABLES: dict[str, InfiltrationVariable] = {
    name: InfiltrationVariable(
        name, config_key, filename, standard_name, unit, fill_value=fill_value
    )
    for name, config_key, filename, standard_name, unit, fill_value in [
        ("qinf", "qinffile", "sfincs.qinf", "infiltration rate", "mm.hr-1", 0.0),
        (
            "scs",
            "scsfile",
            "sfincs.scs",
            "potential soil moisture retention",
            "inch",
            0.0,  # S = 0 means all rainfall runs off
        ),
        (
            "smax",
            "smaxfile",
            "sfincs.smax",
            "potential maximum soil moisture retention",
            "m",
            0.0,
        ),
        (
            "seff",
            "sefffile",
            "sfincs.seff",
            "effective potential maximum soil moisture retention",
            "m",
            0.0,
        ),
        (
            "ks",
            "ksfile",
            "sfincs.ks",
            "saturated hydraulic conductivity",
            "mm.hr-1",
            0.0,  # governs Green-Ampt and CN-recovery; 0 is impermeable
        ),
        ("psi", "psifile", "sfincs.psi", "wetting front suction head", "mm", 0.0),
        ("sigma", "sigmafile", "sfincs.sigma", "soil moisture deficit", "-", 0.0),
        (
            "f0",
            "f0file",
            "sfincs.f0",
            "initial infiltration capacity",
            "mm.hr-1",
            0.0,
        ),
        (
            "fc",
            "fcfile",
            "sfincs.fc",
            "asymptotic infiltration capacity",
            "mm.hr-1",
            0.0,
        ),
        # TODO kd=0 means no decay, so f stays at f0;
        ("kd", "kdfile", "sfincs.kd", "horton decay coefficient", "hr-1", 0.0),
        ("bucket_smax", None, None, "bucket maximum storage", "mm", 0.0),
        ("bucket_k", None, None, "bucket drainage coefficient", "hr-1", 0.0),
        ("bucket_loss", None, None, "bucket loss fraction", "-", 0.0),
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


def _require_lulc_modifiers(lulc, lulc_modifier_table) -> None:
    if lulc is not None and lulc_modifier_table is None:
        raise ValueError(
            "Provide lulc_modifier_table when lulc is used; the table must match "
            "the land-cover dataset's class codes."
        )


def configured_flavor(config: "SfincsConfig", grid_type: str) -> str | None:
    """Infer the configured infiltration flavor from model config."""
    if grid_type not in ("regular", "quadtree"):
        raise ValueError(f"Unsupported grid_type: {grid_type}")
    if grid_type == "quadtree" and config.get("inffile") not in (None, "none"):
        return config.get("inftype")
    if grid_type == "quadtree":
        return "con" if config.get("qinf") not in (None, 0.0) else None

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
        if flavor != "bkt"
        and all(config.get(key) not in (None, "none") for key in keys)
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
    config.set("inffile", None)
    config.set("inftype", None)
    for meta in VARIABLES.values():
        if meta.config_key is not None:
            config.set(meta.config_key, None)


def configure(config: "SfincsConfig", flavor: str, grid_type: str) -> None:
    """Update model config for one infiltration flavor."""
    if grid_type not in ("regular", "quadtree"):
        raise ValueError(f"Unsupported grid_type: {grid_type}")
    if grid_type == "regular" and flavor == "bkt":
        raise ValueError("Bucket infiltration is only supported on quadtree grids")
    reset_config(config)
    if flavor == "con":
        return
    if grid_type == "regular":
        for name in flavor_variables(flavor):
            meta = VARIABLES[name]
            if meta.config_key is None or meta.default_filename is None:
                raise ValueError(f"No regular-grid binary file is defined for {name}")
            config.set(meta.config_key, meta.default_filename)
    elif grid_type == "quadtree":
        config.set("inffile", DEFAULT_INFILTRATIONFILE)
        config.set("inftype", flavor)
