"""Pydantic model for SFINCS configuration variables (sfincs.inp).

Fields follow the order and defaults of the SFINCS kernel's
``sfincs_input.f90``.

Write policy (controls which fields appear in sfincs.inp):

* ``json_schema_extra={"always": True}`` — always written, even when the
  value equals the field default.  Use for fields that must always be
  present (time control, CRS, grid).
* No ``json_schema_extra`` (default) — written only when the value differs
  from the field default.
* ``json_schema_extra={"condition": "<expr>"}`` — written only when the
  Python expression *<expr>* evaluates to ``True`` against the current
  model-field values **and** the value differs from the field default.
* ``json_schema_extra={"min_version": "<X.Y.Z>"}`` — the keyword was introduced in
  this SFINCS kernel version; not written when ``sfincs_version`` is older.
* ``json_schema_extra={"legacy_alias": [("<name>", "<X.Y.Z>"), ...]}`` —
  earlier names of the keyword, with the last SFINCS version in which ``<name>``
  was the kernel keyword. They are read into this field. The list follows the kernel's
  lookup order: when several aliases are in a file, the first one listed is used.
* ``json_schema_extra={"max_version": "<X.Y.Z>"}`` — field is deprecated as of
  this kernel version; when ``sfincs_version`` is newer it is only kept if
  explicitly present in the source file (never force-written via ``always``).

Keywords are written as they were read or set. They are only changed (with a
warning) when they are not valid for the ``sfincs_version`` of the file, or, for a
file without ``sfincs_version``, when they predate
``MIN_SUPPORTED_SFINCS_VERSION``. ``write(update_to_latest=True)`` writes all
keywords for ``LATEST_SFINCS_VERSION``.
"""

import functools
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_serializer
from pydantic_settings import BaseSettings

from hydromt_sfincs import LATEST_SFINCS_VERSION, MIN_SUPPORTED_SFINCS_VERSION

logger = logging.getLogger(f"hydromt.{__name__}")
NOW = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


def _read_config(filename: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read the raw key-value pairs from a SFINCS input file (sfincs.inp).

    Only splits lines into raw string key-value pairs (stripping comments); all
    type coercion and validation is done by :py:meth:`SfincsConfigVariables.read`
    via pydantic.

    Parameters
    ----------
    filename : str | Path
        The path to the input file containing the SFINCS model settings.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        The raw model settings (as strings), and any inline trailing comment
        (after ``#``) found per key.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(
            f"SFINCS input file '{filename.as_posix()}' does not exist."
        )

    with open(filename, "r") as fid:
        lines = fid.readlines()

    inp_dict = {}
    comments = {}
    for line in lines:
        # Check if first character is #
        if line.strip().startswith("#"):
            # Full line comment
            continue
        # Find last character before #
        comment_idx = line.find("#")
        comment = ""
        if comment_idx >= 0:
            comment = line[comment_idx + 1 :].strip()
            line = line[:comment_idx]
        line = [x.strip() for x in line.split("=")]
        if len(line) != 2:
            continue
        name, val = line
        inp_dict[name] = val
        if comment:
            comments[name] = comment

    return inp_dict, comments


def _legacy_aliases(extra: dict) -> list[tuple[str, str | None]]:
    """Return the legacy keywords of a field as ``(name, max_version)`` pairs."""
    return [
        (alias, None) if isinstance(alias, str) else tuple(alias)
        for alias in extra.get("legacy_alias", [])
    ]


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a simple 'X.Y.Z'-style version string into a comparable tuple."""
    return tuple(int(part) for part in version.strip().split("."))


def _try_parse_version(version: str | None) -> tuple[int, ...] | None:
    """Parse a version string, returning None when it is missing or not 'X.Y.Z'."""
    if version is None:
        return None
    try:
        return _parse_version(str(version))
    except ValueError:
        return None


def _keyword_valid(keyword: str, key: str, extra: dict, version: tuple) -> bool:
    """Return whether SFINCS ``version`` reads field ``key`` under ``keyword``."""
    if keyword == key:
        min_version = extra.get("min_version")
        return min_version is None or version >= _parse_version(min_version)
    for alias, max_version in _legacy_aliases(extra):
        if alias == keyword:
            return max_version is not None and version <= _parse_version(max_version)
    return False


def _keyword_for_version(key: str, extra: dict, version: tuple) -> str | None:
    """Return the keyword under which SFINCS ``version`` reads field ``key``.

    The current name is preferred; otherwise the legacy alias that was the kernel
    keyword at ``version``. Returns None when ``version`` does not know the field.
    """
    if _keyword_valid(key, key, extra, version):
        return key
    candidates = sorted(
        (_parse_version(max_version), alias)
        for alias, max_version in _legacy_aliases(extra)
        if max_version is not None and version <= _parse_version(max_version)
    )
    return candidates[0][1] if candidates else None


def _is_obsolete_alias(keyword: str, extra: dict) -> bool:
    """Return whether a legacy keyword predates the minimum supported SFINCS version."""
    max_version = dict(_legacy_aliases(extra)).get(keyword)
    return max_version is not None and _parse_version(max_version) < _parse_version(
        MIN_SUPPORTED_SFINCS_VERSION
    )


def _version_status(
    extra: dict,
    user_sfincs_version: str | None,
    min_supported_version: str = MIN_SUPPORTED_SFINCS_VERSION,
) -> str:
    """Classify a field as "ok", "too_old", "deprecated" or "unknown".

    Without a ``sfincs_version`` the targeted SFINCS version is not known, so a
    field with a ``min_version`` or ``max_version`` is "unknown".
    """
    min_version = extra.get("min_version")
    max_version = extra.get("max_version")
    if not min_version and not max_version:
        return "ok"
    if user_sfincs_version is None:
        return "unknown"
    try:
        current = _parse_version(str(user_sfincs_version))
        if min_version is not None and current < _parse_version(min_version):
            return "too_old"
        if max_version is not None and current > _parse_version(max_version):
            return "deprecated"
    except ValueError:
        return "unknown"
    return "ok"


def _is_below_min_supported(
    user_sfincs_version: str | None,
    min_supported_version: str = MIN_SUPPORTED_SFINCS_VERSION,
) -> bool:
    """Return whether a configured SFINCS version is below the package minimum."""
    if user_sfincs_version is None:
        return False
    try:
        return _parse_version(str(user_sfincs_version)) < _parse_version(
            min_supported_version
        )
    except ValueError:
        return False


class SfincsConfigVariables(BaseSettings):
    """SFINCS configuration variables with defaults matching sfincs_input.f90."""

    model_config = ConfigDict(extra="allow")

    # ================================================================
    # Grid
    # ================================================================
    mmax: int | None = Field(
        None,
        ge=1,
        description="Number of grid cells in x-direction",
        json_schema_extra={"condition": "qtrfile is None"},
    )
    nmax: int | None = Field(
        None,
        ge=1,
        description="Number of grid cells in y-direction",
        json_schema_extra={"condition": "qtrfile is None"},
    )
    dx: float | None = Field(
        None,
        gt=0,
        description="Grid size in x-direction",
        json_schema_extra={"condition": "qtrfile is None"},
    )
    dy: float | None = Field(
        None,
        gt=0,
        description="Grid size in y-direction",
        json_schema_extra={"condition": "qtrfile is None"},
    )
    x0: float | None = Field(
        None,
        description="Origin of the grid in the x-direction",
        json_schema_extra={"condition": "qtrfile is None"},
    )
    y0: float | None = Field(
        None,
        description="Origin of the grid in the y-direction",
        json_schema_extra={"condition": "qtrfile is None"},
    )
    rotation: float = Field(
        0.0,
        gt=-360,
        lt=360,
        description="Rotation of the grid in degrees from the x-axis (east) in anti-clockwise direction",
        json_schema_extra={"condition": "qtrfile is None"},
    )

    # ================================================================
    # Time (always written)
    # ================================================================
    tref: datetime = Field(
        NOW,
        description="Reference time for simulation",
        json_schema_extra={"always": True},
    )
    tstart: datetime = Field(
        NOW,
        description="Start time for the simulation",
        json_schema_extra={"always": True},
    )
    tstop: datetime = Field(
        NOW + timedelta(days=1),
        description="Stop time for the simulation",
        json_schema_extra={"always": True},
    )
    tspinup: float = Field(
        0.0,
        ge=0.0,
        description="Duration of spinup period (seconds)",
        json_schema_extra={"always": True},
    )
    t0out: float = Field(-999.0, description="Output start time (seconds)")
    t1out: float = Field(-999.0, description="Output stop time (seconds)")

    # ================================================================
    # Output intervals
    # ================================================================
    dthisout: float = Field(
        600.0,
        description="Timeseries output interval (seconds)",
        json_schema_extra={"always": True},
    )
    dtmapout: float = Field(
        3600.0,
        ge=0.0,
        description="Spatial map output interval (seconds)",
        json_schema_extra={"always": True, "legacy_alias": [("dtout", "2.4.1")]},
    )
    dtmaxout: float = Field(
        86400.0,
        ge=0.0,
        description="Maximum map output interval (seconds)",
        json_schema_extra={"always": True},
    )
    dtrstout: float = Field(
        0.0,
        ge=0.0,
        description="Restart file output interval (seconds)",
    )
    trstout: float = Field(
        -999.0,
        description="Restart file output after specific time (seconds)",
        json_schema_extra={"always": True},
    )
    dtwave: float = Field(
        3600.0,
        description="Interval of running SnapWave (seconds)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    dtwnd: float = Field(
        1800.0,
        description="Interval of updating wind forcing (seconds)",
        json_schema_extra={"always": False},
    )

    # ================================================================
    # Numerics
    # ================================================================
    alpha: float = Field(
        0.5,
        ge=0.001,
        le=1.0,
        description="Numerical time step reduction for CFL-condition (-)",
        json_schema_extra={"always": True},
    )
    theta: float = Field(
        1.0,
        ge=0.8,
        le=1.0,
        description="Numerical smoothing factor in momentum equation (-)",
    )
    hmin_cfl: float = Field(
        0.1,
        gt=0.0,
        description="Minimum water depth for CFL condition (meters)",
    )
    dtmax: float = Field(
        60.0,
        gt=0.0,
        description="Maximum allowed internal timestep (seconds)",
    )
    huthresh: float = Field(
        0.01,
        gt=0.0,
        lt=1.0,
        description="Threshold water depth (meters)",
        json_schema_extra={"always": True},
    )
    huvmin: float = Field(
        0.0,
        description="Minimum water depth for velocity computation (meters)",
        json_schema_extra={"min_version": "2.3.2"},
    )
    advection: int = Field(
        1,
        ge=0,
        le=1,
        description="Enable advection (1: yes, 0: no)",
        json_schema_extra={"always": True},
    )
    advlim: float = Field(
        1.0,
        ge=0.0,
        description="Maximum value of the advection term in the momentum equation (-)",
    )
    advection_scheme: str = Field(
        "upw1",
        description="Advection scheme ('upw1' or 'original')",
    )
    advection_mask: int = Field(
        1,
        ge=0,
        le=1,
        description="Option to turn on the advection mask (1: on, 0: off)",
    )
    coriolis: int = Field(
        1,
        ge=0,
        le=1,
        description="Enable Coriolis term (1: on, 0: off)",
        json_schema_extra={"always": True},
    )
    viscosity: int = Field(
        1,
        ge=0,
        le=1,
        description="Enable viscosity term (1: yes, 0: no)",
        json_schema_extra={"always": True},
    )
    nuvisc: float = Field(
        0.01,
        ge=0.0,
        description="Viscosity coefficient per metre of grid cell length (-)",
        json_schema_extra={"always": False},
    )
    nuviscfac: float = Field(
        100.0,
        ge=0.0,
        description="Viscosity factor (-)",
        json_schema_extra={"min_version": "2.1.3"},
    )
    friction2d: int = Field(
        1,
        ge=0,
        le=1,
        description="2D friction component in the momentum equation (1: on, 0: off)",
    )
    uvlim: float = Field(
        10.0,
        description="Velocity limiter (m/s)",
        json_schema_extra={"min_version": "2.1.3"},
    )
    uvmax: float = Field(
        1000.0,
        description="Maximum velocity (m/s)",
        json_schema_extra={"min_version": "2.1.3"},
    )
    wiggle_suppression: int = Field(
        1,
        ge=0,
        le=1,
        description="Wiggle suppression (1: on, 0: off)",
    )
    wiggle_factor: float = Field(
        0.1,
        ge=0.0,
        description="Wiggle suppression factor (-)",
    )
    wiggle_threshold: float = Field(
        0.1,
        ge=0.0,
        description="Wiggle suppression minimum depth threshold (m)",
    )
    slopelim: float = Field(
        9999.9,
        ge=0.0,
        description="Slope limiter (-)",
    )
    btrelax: float = Field(
        3600.0,
        ge=0.0,
        description="Relaxation in uvmean (seconds)",
    )
    stopdepth: float | None = Field(
        None,
        ge=0.0,
        description="[DEPRECATED] Minimum water depth for stopping the simulation (m)",
        json_schema_extra={"max_version": "2.1.2"},
    )

    # ================================================================
    # Roughness
    # ================================================================
    manning: float = Field(
        0.04,
        gt=0.0,
        lt=0.5,
        description="Manning's n coefficient for spatially uniform roughness (s/m^(1/3))",
    )
    manning_land: float | None = Field(
        None,
        gt=0.0,
        lt=0.5,
        description="Manning's n for land areas; 0.04 recommended, -999 = not set (s/m^(1/3))",
    )
    manning_sea: float | None = Field(
        None,
        gt=0.0,
        lt=0.5,
        description="Manning's n for sea areas; 0.02 recommended, -999 = not set (s/m^(1/3))",
    )
    rgh_lev_land: float | None = Field(
        None,
        description="Elevation level to distinguish land and sea roughness (m)",
    )

    # ================================================================
    # Physics
    # ================================================================
    rhoa: float = Field(
        1.25,
        gt=0.0,
        description="Air density (kg/m^3)",
        json_schema_extra={"always": False},
    )
    rhow: float = Field(
        1024.0,
        gt=0.0,
        description="Water density (kg/m^3)",
        json_schema_extra={"always": False},
    )
    latitude: float = Field(
        0.0,
        description="Latitude of the grid center (degrees)",
    )
    baro: int = Field(
        1,
        ge=0,
        le=1,
        description="Enable atmospheric pressure term (1: yes, 0: no)",
        json_schema_extra={"always": False},
    )
    pavbnd: float = Field(
        0.0,
        description="Atmospheric pressure at boundary (Pa)",
        json_schema_extra={"always": False},
    )
    gapres: float = Field(
        101200.0,
        description="Background atmospheric pressure for spiderweb (Pa)",
    )

    # ================================================================
    # Initial and boundary conditions
    # ================================================================
    zsini: float | None = Field(
        0.0,
        description="Initial water level in entire domain (meters)",
        json_schema_extra={"always": True},
    )
    bndtype: int = Field(
        1,
        ge=1,
        description="Boundary type (-)",
    )
    btfilter: float = Field(
        60.0,
        ge=0.0,
        description="Water level boundary timeseries filtering period (seconds)",
        json_schema_extra={"always": False},
    )
    use_bcafile: int = Field(
        1,
        ge=0,
        le=1,
        description="Use tidal boundary condition file (1: on, 0: off)",
        json_schema_extra={"min_version": "2.3.0"},
    )

    # ================================================================
    # Infiltration
    # ================================================================
    qinf: float | None = Field(
        0.0,
        ge=0.0,
        description="Infiltration rate, spatially uniform (mm/hr)",
        json_schema_extra={"always": True},
    )
    qinf_zmin: float = Field(
        0.0,
        description="Minimum elevation for spatially uniform infiltration (m)",
    )
    sfacinf: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Curve Number initial abstraction factor (-)",
    )
    horton_kr_kd: float = Field(
        10.0,
        description="Horton infiltration recovery vs decay ratio (-)",
    )

    # ================================================================
    # Wavemaker
    # ================================================================
    wavemaker_nfreqs_ig: int = Field(
        100,
        ge=1,
        le=500,
        description="Wave maker number of frequency bins IG spectrum (-)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("nfreqsig", "2.4.1")],
        },
    )
    wavemaker_freqmin_ig: float = Field(
        0.0,
        ge=0.0,
        description="Minimum frequency wave maker IG spectrum (Hz)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("freqminig", "2.4.1")],
        },
    )
    wavemaker_freqmax_ig: float = Field(
        0.1,
        ge=0.0,
        description="Maximum frequency wave maker IG spectrum (Hz)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("freqmaxig", "2.4.1")],
        },
    )
    wavemaker_filter_time: float = Field(
        600.0,
        ge=0.0,
        description="Filtering duration for wave maker mean water level (s)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("wmtfilter", "2.4.1")],
        },
    )
    wavemaker_filter_fred: float = Field(
        0.99,
        description="Filtering variable in wave maker (-)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("wmfred", "2.4.1")],
        },
    )
    wavemaker_signal: str = Field(
        "spectrum",
        description="Wavemaker signal type ('spectrum' or 'mon')",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("wmsignal", "2.4.1")],
        },
    )
    wavemaker_hmin: float = Field(
        0.1,
        description="Wavemaker minimum water depth (m)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("wmhmin", "2.4.1")],
        },
    )
    wavemaker_nfreqs_inc: int = Field(
        100,
        description="Wave maker number of frequency bins incident wave spectrum (-)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("nfreqsinc", "2.4.1")],
        },
    )
    wavemaker_freqmin_inc: float = Field(
        0.04,
        description="Minimum frequency wave maker incident wave spectrum (Hz)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("freqmininc", "2.4.1")],
        },
    )
    wavemaker_freqmax_inc: float = Field(
        1.0,
        description="Maximum frequency wave maker incident wave spectrum (Hz)",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("freqmaxinc", "2.4.1")],
        },
    )
    wavemaker_tinc2ig: float = Field(
        -1.0,
        description="Wave maker ratio of IG to incident wave period; <= 0 uses Herbers (-)",
        json_schema_extra={"min_version": "2.3.3"},
    )
    wavemaker_hm0_ig_factor: float = Field(
        1.0,
        description="Wave maker IG wave height scale factor (-)",
        json_schema_extra={"min_version": "2.3.3"},
    )
    wavemaker_hm0_inc_factor: float = Field(
        1.0,
        description="Wave maker incident wave height scale factor (-)",
        json_schema_extra={"min_version": "2.3.3"},
    )
    wavemaker_gammax: float = Field(
        1.0,
        description="Wave maker maximum ratio of wave height over water depth (-)",
        json_schema_extra={"min_version": "2.3.3"},
    )
    wavemaker_tpmin: float = Field(
        1.0,
        description="Wave maker minimum peak wave period (s)",
        json_schema_extra={"min_version": "2.3.3"},
    )
    wavemaker_hig: int = Field(
        1,
        ge=0,
        le=1,
        description="Wave maker include IG waves when forced by SnapWave (1: on, 0: off)",
        json_schema_extra={"min_version": "2.3.3"},
    )
    wavemaker_hinc: int = Field(
        0,
        ge=0,
        le=1,
        description="Wave maker include incident waves when forced by SnapWave (1: on, 0: off)",
        json_schema_extra={"min_version": "2.3.3"},
    )

    # ================================================================
    # Meteo
    # ================================================================
    amprblock: int = Field(
        1,
        ge=0,
        le=1,
        description="Use ampr data as block interpolation (1: yes, 0: no)",
    )
    spwmergefrac: float = Field(
        0.5,
        gt=0.0,
        lt=1.0,
        description="Spiderweb merge factor with background wind and pressure (-)",
    )
    usespwprecip: int = Field(
        1,
        ge=0,
        le=1,
        description="Use rainfall from spiderweb (1: on, 0: off)",
    )
    spinup_meteo: int = Field(
        0,
        ge=0,
        le=1,
        description="Apply spinup to meteo forcing (1: on, 0: off)",
    )
    waveage: float = Field(
        -999.0,
        description="Determine Cd with wave age (-)",
    )
    factor_wind: float = Field(
        1.0,
        description="Wind forcing scale factor (-)",
        json_schema_extra={"min_version": "2.3.0"},
    )
    factor_pres: float = Field(
        1.0,
        description="Pressure forcing scale factor (-)",
        json_schema_extra={"min_version": "2.3.0"},
    )
    factor_prcp: float = Field(
        1.0,
        description="Precipitation forcing scale factor (-)",
        json_schema_extra={"min_version": "2.3.0"},
    )
    factor_spw_size: float = Field(
        1.0,
        description="Spiderweb size scale factor (-)",
        json_schema_extra={"min_version": "2.3.0"},
    )

    # ================================================================
    # SnapWave coupling
    # ================================================================
    snapwave: int = Field(
        0,
        ge=0,
        le=1,
        description="Enable coupled SnapWave solver (1: on, 0: off)",
    )
    snapwave_wind: int = Field(
        0,
        ge=0,
        le=1,
        description="SnapWave wind growth process (1: on, 0: off)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_use_nearest: int = Field(
        1,
        ge=0,
        le=1,
        description="SnapWave use nearest interpolation (1: on, 0: off)",
        json_schema_extra={"min_version": "2.3.0", "condition": "snapwave == 1"},
    )
    snapwave_igwaves: int = Field(
        1,
        ge=0,
        le=1,
        description="SnapWave IG wave computation (1: on, 0: off)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_dtheta: float = Field(
        10.0,
        gt=0,
        description="SnapWave directional resolution (degrees)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_crit: float = Field(
        0.001,
        gt=0,
        description="SnapWave convergence criterion (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_hmin: float = Field(
        0.1,
        gt=0,
        description="SnapWave minimum water depth (m)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_vegetation: int = Field(
        0,
        ge=0,
        le=1,
        description="SnapWave vegetation dissipation (1: on, 0: off)",
        json_schema_extra={"min_version": "2.4.0", "condition": "snapwave == 1"},
    )
    snapwave_waveforces_ratio: float = Field(
        1.0,
        description="Multiplication factor on SnapWave wave forces (-)",
        json_schema_extra={
            "min_version": "2.4.0",
            "condition": "snapwave == 1",
            "legacy_alias": [
                ("snapwave_waveforces_factor", "2.3.3"),
                ("waveforces_ratio", None),
            ],
        },
    )
    radstr: int = Field(
        0,
        ge=0,
        le=1,
        description="Radiation stress forcing from SnapWave (1: on, 0: off)",
    )
    snapwave_gamma: float = Field(
        0.7,
        description="SnapWave wave breaking parameter (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_gammax: float = Field(
        999.0,
        description="SnapWave maximum ratio of wave height over water depth (-)",
        json_schema_extra={"min_version": "2.2.0", "condition": "snapwave == 1"},
    )
    snapwave_alpha: float = Field(
        1.0,
        description="SnapWave wave dissipation coefficient (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_fw: float = Field(
        0.01,
        description="SnapWave bottom friction factor incident waves (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_fwig: float = Field(
        0.015,
        description="SnapWave bottom friction factor IG waves (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_fw_ratio: float = Field(
        1.0,
        description="SnapWave ratio of land to sea friction factor incident waves (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_fwig_ratio: float = Field(
        1.0,
        description="SnapWave ratio of land to sea friction factor IG waves (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_dt: float = Field(
        36000.0,
        description="SnapWave internal time step, large value means stationary (seconds)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_tol: float = Field(
        1000.0,
        description="SnapWave tolerance (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_niter: int = Field(
        10,
        description="SnapWave maximum number of iterations (-)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_baldock_ratio: float = Field(
        0.2,
        description="SnapWave Baldock ratio incident waves (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_baldock_exponent: int = Field(
        2,
        description="SnapWave exponent of Baldock dissipation enhancement factor; 0 is unused (-)",
        json_schema_extra={"min_version": "2.4.0", "condition": "snapwave == 1"},
    )
    snapwave_Tpini: float = Field(
        1.0,
        description="SnapWave initial peak wave period (s)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_mwind: int = Field(
        2,
        description="SnapWave wind growth option (-)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_sigmin: float = Field(
        2 * math.pi / 25.0,
        description="SnapWave minimum radial frequency (rad/s)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_sigmax: float = Field(
        2 * math.pi,
        description="SnapWave maximum radial frequency (rad/s)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_jadcgdx: int = Field(
        1,
        ge=0,
        le=1,
        description="SnapWave include dcg/dx term (1: on, 0: off)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_c_dispT: float = Field(
        1.0,
        description="SnapWave wave period dispersion coefficient (-)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_sector: float = Field(
        180.0,
        description="SnapWave directional sector (degrees)",
        json_schema_extra={"min_version": "2.1.2", "condition": "snapwave == 1"},
    )
    snapwave_relax_factor_DoverA: float = Field(
        0.25,
        description="SnapWave underrelaxation factor for DoverA; 1.0 disables (-)",
        json_schema_extra={"min_version": "2.4.0", "condition": "snapwave == 1"},
    )
    snapwave_relax_factor_DoverE: float = Field(
        0.25,
        description="SnapWave underrelaxation factor for DoverE; 1.0 disables (-)",
        json_schema_extra={"min_version": "2.4.0", "condition": "snapwave == 1"},
    )
    snapwave_alpha_ig: float = Field(
        1.0,
        description="SnapWave wave dissipation coefficient IG waves (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_gammaig: float = Field(
        0.7,
        description="SnapWave wave breaking parameter IG waves (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_gamma_fac_br: float = Field(
        0.45,
        description="SnapWave factor on gamma to determine the incident wave breaking point (-)",
        json_schema_extra={"min_version": "2.4.0", "condition": "snapwave == 1"},
    )
    snapwave_shinc2ig: float = Field(
        1.0,
        description="SnapWave ratio of IG source term subtracted from incident wave energy (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_alphaigfac: float = Field(
        1.0,
        description="SnapWave multiplication factor for IG shoaling source term (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_baldock_ratio_ig: float = Field(
        0.2,
        description="SnapWave Baldock ratio IG waves (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_ig_opt: int = Field(
        1,
        description="SnapWave IG wave option (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_iterative_srcig: int = Field(
        0,
        ge=0,
        le=1,
        description="SnapWave compute IG source term in iterative loop (1: on, 0: off)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_use_herbers: int = Field(
        1,
        ge=0,
        le=1,
        description="SnapWave compute IG boundary conditions with Herbers (1: on, 0: off)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_tpig_opt: int = Field(
        1,
        description="SnapWave IG wave period option for Herbers (1: Tm01, 2: Tpsmooth, 3: Tp, 4: Tm-1,0)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_jonswapgamma: float = Field(
        3.3,
        description="SnapWave JONSWAP gamma of offshore spectrum for Herbers (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_eeinc2ig: float = Field(
        0.01,
        description="SnapWave ratio of IG to incident wave energy, used if snapwave_use_herbers = 0 (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_Tinc2ig: float = Field(
        7.0,
        description="SnapWave ratio of IG to incident wave period, used if snapwave_use_herbers = 0 (-)",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    wave_enhanced_roughness: int = Field(
        0,
        ge=0,
        le=1,
        description="Wave enhanced roughness (1: on, 0: off)",
        json_schema_extra={"min_version": "2.3.0"},
    )

    # ================================================================
    # Non-hydrostatic
    # ================================================================
    nonh: int = Field(
        0,
        ge=0,
        le=1,
        description="Enable non-hydrostatic mode (1: on, 0: off)",
        json_schema_extra={"min_version": "2.2.0"},
    )
    nh_fnudge: float = Field(
        0.9,
        description="Non-hydrostatic nudge factor (-)",
        json_schema_extra={"min_version": "2.2.0", "condition": "nonh == 1"},
    )
    nh_tstop: float = Field(
        -999.0,
        description="Non-hydrostatic stop time (seconds)",
        json_schema_extra={"min_version": "2.2.0", "condition": "nonh == 1"},
    )
    nh_tol: float = Field(
        0.001,
        description="Non-hydrostatic tolerance (-)",
        json_schema_extra={"min_version": "2.2.0", "condition": "nonh == 1"},
    )
    nh_itermax: int = Field(
        100,
        description="Non-hydrostatic maximum iterations (-)",
        json_schema_extra={"min_version": "2.2.0", "condition": "nonh == 1"},
    )

    # ================================================================
    # Structures
    # ================================================================
    structure_relax: float = Field(
        10.0,
        description="Structure relaxation factor (-)",
        json_schema_extra={"min_version": "2.2.0"},
    )

    # ================================================================
    # Bathtub
    # ================================================================
    bathtub: int = Field(
        0,
        ge=0,
        le=1,
        description="Enable bathtub mode (1: on, 0: off)",
        json_schema_extra={"min_version": "2.3.2"},
    )
    bathtub_fachs: float = Field(
        0.2,
        description="Bathtub wave height factor (-)",
        json_schema_extra={"min_version": "2.3.2", "condition": "bathtub == 1"},
    )
    bathtub_dt: float = Field(
        -999.0,
        description="Bathtub time step (seconds)",
        json_schema_extra={"min_version": "2.3.2", "condition": "bathtub == 1"},
    )

    # ================================================================
    # Other settings
    # ================================================================
    sfincs_version: str | None = Field(
        None,
        description="Target SFINCS kernel version; controls min_version/max_version field handling",
    )
    global_model: int = Field(
        0,
        ge=0,
        le=1,
        description="Global spherical model that wraps over the edge (1: on, 0: off)",
        json_schema_extra={
            "min_version": "2.5.0",
            "legacy_alias": [("global", "2.4.1")],
        },
    )
    crsgeo: int = Field(
        0,
        ge=0,
        le=1,
        description="Geographical coordinate system flag (1: yes, 0: no)",
    )
    epsg: int | None = Field(
        None,
        description="EPSG code for spatial reference system",
        json_schema_extra={"legacy_alias": [("crs", "2.0.0")]},
    )
    utmzone: str | None = Field(
        None,
        description="UTM zone for spatial reference (-)",
    )
    inputformat: str = Field(
        "bin",
        description="Input file format (bin or asc)",
    )
    outputformat: str = Field(
        "net",
        description="Output file format (net or asc or bin)",
    )
    outputtype_map: str | None = Field(
        None,
        description="Output format for map file (net or asc or bin)",
    )
    outputtype_his: str | None = Field(
        None,
        description="Output format for his file (net or asc or bin)",
    )
    nc_deflate_level: int = Field(
        2,
        description="Netcdf deflate level (-)",
    )
    rugdepth: float = Field(
        0.05,
        description="Runup gauge depth threshold (m)",
        json_schema_extra={"min_version": "2.3.0"},
    )
    h73table: int = Field(
        0,
        ge=0,
        le=1,
        description="Use h73 table (1: on, 0: off)",
        json_schema_extra={"min_version": "2.2.0"},
    )

    # ================================================================
    # Wind drag
    # ================================================================
    cdnrb: int = Field(
        3,
        description="Number of wind speed ranges for drag coefficient",
        json_schema_extra={"always": False},
    )
    cdwnd: list[float] | None = Field(
        [0.0, 28.0, 50.0],
        description="Wind speed ranges for drag coefficient (m/s)",
        json_schema_extra={"always": False},
    )
    cdval: list[float] | None = Field(
        [0.001, 0.0025, 0.0015],
        description="Drag coefficient values corresponding to cdwnd",
        json_schema_extra={"always": False},
    )

    # ================================================================
    # Domain files (example of Optional → | None)
    # ================================================================
    qtrfile: str | None = Field(None, description="Quadtree file")
    depfile: str | None = Field(None, description="Depth file")
    inifile: str | None = Field(None, description="Initial water level file")
    rstfile: str | None = Field(None, description="Restart file")
    mskfile: str | None = Field(None, description="Mask file")
    indexfile: str | None = Field(None, description="Index file")
    sbgfile: str | None = Field(None, description="Subgrid file")
    thdfile: str | None = Field(None, description="Thin dam structure file")
    weirfile: str | None = Field(None, description="Weir structure file")
    manningfile: str | None = Field(None, description="Manning's n file")
    drnfile: str | None = Field(None, description="Drainage structure file")
    volfile: str | None = Field(None, description="Storage volume file")
    dkbfile: str | None = Field(
        None,
        description="Dike breaches file",
        json_schema_extra={"min_version": "2.5.0"},
    )
    urbfile: str | None = Field(
        None,
        description="Urban drainage zones file",
        json_schema_extra={"min_version": "2.5.0"},
    )
    vegfile: str | None = Field(
        None,
        description="Vegetation characteristics file",
        json_schema_extra={
            "min_version": "2.5.0",
            "legacy_alias": [("vegetationfile", "2.4.1")],
        },
    )
    vegtype: str | None = Field(
        None,
        description="Vegetation type lookup table file (toml)",
        json_schema_extra={
            "min_version": "2.5.0",
            "legacy_alias": [("vegetationtype_toml", "2.4.1")],
        },
    )

    # ================================================================
    # Forcing files
    # ================================================================
    bndfile: str | None = Field(None, description="Water level boundary points file")
    bzsfile: str | None = Field(None, description="Water level time-series file")
    bcafile: str | None = Field(
        None,
        description="Tidal boundary component file",
        json_schema_extra={"min_version": "2.3.0"},
    )
    bzifile: str | None = Field(None, description="Individual wave water level file")
    bdrfile: str | None = Field(
        None,
        description="Downstream river boundary file",
        json_schema_extra={"min_version": "2.3.0"},
    )
    wavemaker_wfpfile: str | None = Field(
        None,
        description="Wavemaker location points file",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("wfpfile", "2.4.1")],
        },
    )
    wavemaker_whifile: str | None = Field(
        None,
        description="Wavemaker IG wave height file",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("whifile", "2.4.1")],
        },
    )
    wavemaker_wtifile: str | None = Field(
        None,
        description="Wavemaker IG wave period file",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("wtifile", "2.4.1")],
        },
    )
    wavemaker_wstfile: str | None = Field(
        None,
        description="Wavemaker setup file",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("wstfile", "2.4.1")],
        },
    )
    srcfile: str | None = Field(None, description="Discharge input points file")
    disfile: str | None = Field(None, description="Discharge input time-series file")
    spwfile: str | None = Field(None, description="Spiderweb tropical cyclone file")
    wndfile: str | None = Field(None, description="Spatially uniform wind file")
    prcfile: str | None = Field(
        None,
        description="Spatially uniform precipitation file",
        json_schema_extra={"legacy_alias": [("precipfile", "2.4.1")]},
    )
    amufile: str | None = Field(None, description="Wind u-component file")
    amvfile: str | None = Field(None, description="Wind v-component file")
    ampfile: str | None = Field(None, description="Atmospheric pressure file")
    amprfile: str | None = Field(
        None,
        description="Precipitation file",
        json_schema_extra={"legacy_alias": [("amrfile", None)]},
    )
    z0lfile: str | None = Field(None, description="Wind reduction over land file")
    wavemaker_wvmfile: str | None = Field(
        None,
        description="Wave maker input points file",
        json_schema_extra={
            "min_version": "2.3.3",
            "legacy_alias": [("wvmfile", "2.4.1")],
        },
    )
    wavemaker_timeseries_wvmfile: str | None = Field(
        None,
        description="Wave maker input points file, forced by IG time series",
        json_schema_extra={"min_version": "2.4.1"},
    )
    qinffile: str | None = Field(None, description="Infiltration file")
    inffile: str | None = Field(
        None,
        description="Infiltration parameters NetCDF file",
        json_schema_extra={
            "min_version": "2.5.0",
            "legacy_alias": [
                ("infiltrationfile", None),
                ("infiltration_file", "2.4.1"),
            ],
        },
    )
    inftype: str | None = Field(
        None,
        description="Infiltration type (con, c2d, cna, cnb, gai, hor, bkt, r2d)",
        json_schema_extra={
            "min_version": "2.5.0",
            "legacy_alias": [
                ("infiltrationtype", None),
                ("infiltration_type", "2.4.1"),
            ],
        },
    )

    # ================================================================
    # Curve Number / Green-Ampt / Horton files
    # ================================================================
    scsfile: str | None = Field(None, description="Curve Number max soil moisture file")
    smaxfile: str | None = Field(None, description="Curve Number max storage file")
    sefffile: str | None = Field(None, description="Curve Number initial storage file")
    psifile: str | None = Field(None, description="Green-Ampt suction head file")
    sigmafile: str | None = Field(
        None, description="Green-Ampt max moisture deficit file"
    )
    ksfile: str | None = Field(
        None, description="Green-Ampt hydraulic conductivity file"
    )
    f0file: str | None = Field(
        None, description="Horton max infiltration capacity file"
    )
    fcfile: str | None = Field(None, description="Horton min infiltration rate file")
    kdfile: str | None = Field(None, description="Horton decay constant file")

    # ================================================================
    # Netcdf input files
    # ================================================================
    netbndbzsbzifile: str | None = Field(
        None, description="Netcdf water level input file"
    )
    netsrcdisfile: str | None = Field(None, description="Netcdf discharge input file")
    netamuamvfile: str | None = Field(None, description="Netcdf wind input file")
    netamprfile: str | None = Field(None, description="Netcdf precipitation input file")
    netampfile: str | None = Field(
        None, description="Netcdf atmospheric pressure input file"
    )
    netspwfile: str | None = Field(None, description="Netcdf spiderweb input file")

    # ================================================================
    # SnapWave forcing files
    # ================================================================
    snapwave_bndfile: str | None = Field(
        None,
        description="SnapWave boundary points file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_bhsfile: str | None = Field(
        None,
        description="SnapWave wave height file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_btpfile: str | None = Field(
        None,
        description="SnapWave wave period file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_bwdfile: str | None = Field(
        None,
        description="SnapWave wave direction file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_bdsfile: str | None = Field(
        None,
        description="SnapWave wave spreading file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    netsnapwavefile: str | None = Field(
        None, description="Netcdf SnapWave boundary file"
    )
    snapwave_jonswapfile: str | None = Field(
        None,
        description="SnapWave JONSWAP boundary conditions file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_encfile: str | None = Field(
        None,
        description="SnapWave enclosure file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_upwfile: str | None = Field(
        None,
        description="SnapWave upwind neighbours file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_mskfile: str | None = Field(
        None,
        description="SnapWave mask file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_depfile: str | None = Field(
        None,
        description="SnapWave depth file",
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_ncfile: str | None = Field(
        None,
        description="SnapWave netcdf grid file",
        json_schema_extra={"condition": "snapwave == 1"},
    )

    # ================================================================
    # Output observation files
    # ================================================================
    obsfile: str | None = Field(None, description="Observation points file")
    crsfile: str | None = Field(None, description="Cross-section lines file")
    rugfile: str | None = Field(
        None,
        description="Runup gauges file",
        json_schema_extra={"min_version": "2.3.0"},
    )

    # ================================================================
    # Output storage options
    # ================================================================
    storevelmax: int = Field(
        0, ge=0, le=1, description="Write max velocity output (1: yes, 0: no)"
    )
    storefluxmax: int = Field(
        0, ge=0, le=1, description="Write max flux output (1: yes, 0: no)"
    )
    storevel: int = Field(
        0,
        ge=0,
        le=1,
        description="Write instantaneous velocity output (1: yes, 0: no)",
        json_schema_extra={"always": True},
    )
    storecumprcp: int = Field(
        0,
        ge=0,
        le=1,
        description="Write cumulative precipitation output (1: yes, 0: no)",
        json_schema_extra={"always": True},
    )
    storetwet: int = Field(
        0, ge=0, le=1, description="Write time-wet output (1: yes, 0: no)"
    )
    storehsubgrid: int = Field(
        0, ge=0, le=1, description="Write subgrid depth output (1: yes, 0: no)"
    )
    storehmean: int = Field(
        0,
        ge=0,
        le=1,
        description="Write mean water depth output (1: yes, 0: no)",
        json_schema_extra={"min_version": "2.2.0"},
    )
    storemeteo: int = Field(
        0,
        ge=0,
        le=1,
        description="Write meteo output (1: yes, 0: no)",
        json_schema_extra={"always": True},
    )
    storemaxwind: int = Field(
        0, ge=0, le=1, description="Write max wind speed output (1: yes, 0: no)"
    )
    storefw: int = Field(
        0, ge=0, le=1, description="Write wave forces output (1: yes, 0: no)"
    )
    storewavdir: int = Field(
        0, ge=0, le=1, description="Write wave direction output (1: yes, 0: no)"
    )
    storeqdrain: int = Field(
        1,
        ge=0,
        le=1,
        description="[DEPRECATED] Write drainage discharge output (1: yes, 0: no)",
        json_schema_extra={"max_version": "2.4.1"},
    )
    storezvolume: int = Field(
        0, ge=0, le=1, description="Write storage volume output (1: yes, 0: no)"
    )
    storestoragevolume: int = Field(
        0,
        ge=0,
        le=1,
        description="Write total storage volume output (1: yes, 0: no)",
        json_schema_extra={"min_version": "2.3.0"},
    )
    store_tsunami_arrival_time: int = Field(
        0, ge=0, le=1, description="Write tsunami arrival time output (1: yes, 0: no)"
    )
    store_dynamic_bed_level: int = Field(
        0,
        ge=0,
        le=1,
        description="Write dynamic bed level output (1: yes, 0: no)",
        json_schema_extra={"min_version": "2.1.3"},
    )
    storetzsmax: int = Field(
        0,
        ge=0,
        le=1,
        description="Write time of maximum water level output (1: yes, 0: no)",
        json_schema_extra={"always": True, "min_version": "2.3.0"},
    )
    storezvolmax: int = Field(
        1,
        ge=0,
        le=1,
        description="Write maximum subgrid cell volume output, subgrid only (1: yes, 0: no)",
        json_schema_extra={"min_version": "2.5.0"},
    )
    store_river_discharge: int = Field(
        0,
        ge=0,
        le=1,
        description="Write river point discharge output (1: yes, 0: no)",
        json_schema_extra={"min_version": "2.5.0"},
    )
    store_urban_drainage_discharge: int = Field(
        0,
        ge=0,
        le=1,
        description="Write urban drainage discharge output (1: yes, 0: no)",
        json_schema_extra={"min_version": "2.5.0"},
    )
    store_cumulative_urban_drainage: int = Field(
        0,
        ge=0,
        le=1,
        description="Write cumulative urban drainage output (1: yes, 0: no)",
        json_schema_extra={"min_version": "2.5.0"},
    )
    storesnapwavegrid: int = Field(
        0,
        ge=0,
        le=1,
        description="Write SnapWave grid output (1: yes, 0: no)",
        json_schema_extra={"min_version": "2.4.0", "condition": "snapwave == 1"},
    )
    writeruntime: int = Field(
        0,
        ge=0,
        le=1,
        description="Write runtimes.txt at the end of the simulation (1: yes, 0: no)",
    )
    output_on_quadtree_mesh: int = Field(
        0,
        ge=0,
        le=1,
        description="Write quadtree without refinement on quadtree mesh (1: yes) or regular grid (0: no)",
        json_schema_extra={
            "min_version": "2.5.0",
            "legacy_alias": [("regular_output_on_mesh", "2.4.1")],
        },
    )
    twet_threshold: float = Field(
        0.01, ge=0.0, description="Time-wet minimum depth threshold (m)"
    )
    timestep_analysis: int = Field(
        0,
        ge=0,
        le=1,
        description="Enable timestep analysis output (1: yes, 0: no)",
        json_schema_extra={"always": True, "min_version": "2.3.2"},
    )
    debug: int = Field(
        0, ge=0, le=1, description="Debug mode — write every timestep (1: yes, 0: no)"
    )
    percentage_done: int = Field(
        5,
        ge=1,
        description="Percentage done output interval (%)",
        json_schema_extra={"min_version": "2.1.3"},
    )

    # ================================================================
    # Read / write
    # ================================================================
    # Keyword under which each field was explicitly read from the source file (or
    # set afterwards), e.g. {"wavemaker_wvmfile": "wvmfile"}; these fields always
    # round-trip on write() under that keyword, even if they equal the default.
    _explicit_keys: dict[str, str] = PrivateAttr(default_factory=dict)
    # Inline trailing comment (after '#') read per key from the source file, used
    # by write(write_comments=True) in preference over the schema description.
    _comments: dict[str, str] = PrivateAttr(default_factory=dict)

    @classmethod
    @functools.cache
    def _legacy_alias_map(cls) -> dict[str, str]:
        """Map each legacy keyword to the name of its current field."""
        return {
            legacy: name
            for name, field in cls.model_fields.items()
            for legacy, _ in _legacy_aliases(field.json_schema_extra or {})
        }

    @classmethod
    def resolve_key(cls, key: str) -> str:
        """Return the current field name for a (legacy) keyword, e.g. 'wvmfile'."""
        return cls._legacy_alias_map().get(key, key)

    @classmethod
    def _check_keyword(
        cls, key: str, keyword: str, sfincs_version: str | None
    ) -> str | None:
        """Return the keyword to use for field ``key`` that was given as ``keyword``.

        A keyword that the targeted ``sfincs_version`` does not read is replaced
        (with a warning) by the one it does; None when that version does not know
        the field. Without a (valid) ``sfincs_version`` the keyword is kept, unless
        it predates ``MIN_SUPPORTED_SFINCS_VERSION``.
        """
        extra = cls.model_fields[key].json_schema_extra or {}
        version = _try_parse_version(sfincs_version)
        if version is not None:
            if _keyword_valid(keyword, key, extra, version):
                return keyword
            new_keyword = _keyword_for_version(key, extra, version)
            reason = f"is not valid for sfincs_version {sfincs_version}"
        elif _is_obsolete_alias(keyword, extra):
            min_version = _parse_version(MIN_SUPPORTED_SFINCS_VERSION)
            new_keyword = _keyword_for_version(key, extra, min_version)
            reason = (
                "predates the minimum supported SFINCS version "
                f"{MIN_SUPPORTED_SFINCS_VERSION}"
            )
        else:
            if keyword != key:
                logger.info(f"Legacy keyword '{keyword}' of '{key}' is kept.")
            return keyword
        if new_keyword is not None:
            logger.warning(f"'{keyword}' {reason}; replaced by '{new_keyword}'.")
        return new_keyword

    @classmethod
    def read(
        cls,
        filename: str | Path,
    ) -> "SfincsConfigVariables":
        """Read a SFINCS input file (sfincs.inp) into a validated instance.

        Keywords are kept as read (and written back as such), except those that
        are not valid for the ``sfincs_version`` in the file or, without
        ``sfincs_version``, that predate ``MIN_SUPPORTED_SFINCS_VERSION``; these
        are replaced or removed with a warning.
        """
        filename = Path(filename)
        inp_dict, comments = _read_config(filename=filename)

        model_fields = cls.model_fields
        sfincs_version = inp_dict.get("sfincs_version")
        if sfincs_version is not None and _try_parse_version(sfincs_version) is None:
            logger.warning(
                f"sfincs_version {sfincs_version} is not an 'X.Y.Z' version; "
                "keywords are kept as read."
            )
        if sfincs_version is None:
            logger.info(
                f"No sfincs_version in {filename}; keywords are kept as read and "
                "written as such. Use config.write(update_to_latest=True) to write "
                f"them for SFINCS version {LATEST_SFINCS_VERSION}."
            )
        if _is_below_min_supported(sfincs_version):
            logger.warning(
                f"sfincs_version {sfincs_version} is below the minimum supported "
                f"SFINCS version {MIN_SUPPORTED_SFINCS_VERSION}."
            )

        # Read legacy keywords into their current field, remembering the keyword
        # they were read with so that write() can write them back as such.
        keywords = {}
        for legacy, key in cls._legacy_alias_map().items():
            if legacy not in inp_dict:
                continue
            value = inp_dict.pop(legacy)
            comment = comments.pop(legacy, None)
            if key in inp_dict:
                logger.warning(
                    f"'{legacy}' is ignored, since '{keywords.get(key, key)}' is also set."
                )
                continue
            inp_dict[key] = value
            keywords[key] = legacy
            if comment:
                comments[key] = comment
            logger.debug(f"Legacy keyword '{legacy}' is read as '{key}'.")

        # Check the keywords against the targeted sfincs_version: deprecated fields
        # and fields the version does not know are dropped, and keywords the version
        # does not read are replaced. An unversioned file keeps all fields.
        for key in list(inp_dict):
            field_info = model_fields.get(key)
            if field_info is None:
                continue
            extra = field_info.json_schema_extra or {}
            if _version_status(extra, sfincs_version) == "deprecated":
                inp_dict.pop(key)
                comments.pop(key, None)
                keywords.pop(key, None)
                logger.warning(
                    f"'{key}' is not supported for sfincs_version {sfincs_version}; removed."
                )
                continue

            keyword = cls._check_keyword(key, keywords.get(key, key), sfincs_version)
            if keyword is None:
                inp_dict.pop(key)
                comments.pop(key, None)
                keywords.pop(key, None)
                logger.warning(
                    f"'{key}' is not supported for sfincs_version {sfincs_version}; removed."
                )
            else:
                keywords[key] = keyword

        # Warn about keys not recognized by the schema; they are kept as
        # pass-through attributes (Config.extra = "allow") and always rewritten.
        unknown_keys = sorted(set(inp_dict) - set(model_fields))
        if unknown_keys:
            logger.warning(
                f"Unrecognized key(s) in {filename}: {unknown_keys}. "
                "They will be kept and rewritten as-is."
            )

        # Full pydantic validation (type coercion, field constraints, validators)
        instance = cls.model_validate(inp_dict)
        instance._explicit_keys = {key: keywords.get(key, key) for key in inp_dict}
        instance._comments = comments
        return instance

    def write(
        self,
        filename: str | Path,
        write_description: bool = False,
        write_comments: bool = False,
        explicit_only: bool = False,
        update_to_latest: bool = False,
    ) -> None:
        """Write the configuration variables to a SFINCS input file (sfincs.inp).

        Parameters:
        -----------
        filename (str | Path):
            The file to write the configuration to.
        write_description (bool):
            If True, append the schema field description as a trailing comment.
            Default is False.
        write_comments (bool):
            If True, append the original inline comment read from the source
            file (if any) as a trailing comment, taking priority over
            write_description for keys that have one. Default is False.
        explicit_only (bool):
            If True, write only fields explicitly read or set. Default is False.
        update_to_latest (bool):
            If True, write the configuration for ``LATEST_SFINCS_VERSION``:
            current keywords instead of legacy ones, no deprecated fields and
            ``sfincs_version`` set to that version. The configuration itself is
            not changed. Default is False, which writes keywords as read or set.
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        model_fields = type(self).model_fields
        # model_dump() applies the write-only filtering/formatting via the
        # _serialize_for_write model_serializer below.
        data_dict = self.model_dump(
            context={
                "explicit_only": explicit_only,
                "update_to_latest": update_to_latest,
            }
        )

        with open(filename, "w") as fid:
            for key, value in data_dict.items():
                string = f"{key.ljust(20)} = {value}"

                # comments and descriptions are stored under the current field name
                field_key = self.resolve_key(key)
                comment = self._comments.get(field_key) if write_comments else None
                if not comment and write_description and field_key in model_fields:
                    comment = model_fields[field_key].description
                if comment:
                    string = string.ljust(50) + f" # {comment}"

                fid.write(string + "\n")

    @model_serializer(mode="wrap")
    def _serialize_for_write(self, handler, info) -> dict:
        """Filter and format field values for writing to sfincs.inp.

        Drops fields whose ``condition``/``min_version``/``max_version`` policy
        excludes them, or that equal their default and were never explicitly
        read/set; writes explicit fields under the keyword they were read/set with
        (unless the targeted ``sfincs_version`` does not read it); formats the
        remaining values (numbers, joined lists, dates) as plain strings/numbers
        ready to write. Used exclusively by :py:meth:`write`; use
        :py:meth:`to_dict` for the unfiltered, native-typed configuration.
        """
        data = handler(self)
        model_fields = type(self).model_fields
        context = info.context or {}
        update_to_latest = context.get("update_to_latest", False)
        if update_to_latest:
            data["sfincs_version"] = LATEST_SFINCS_VERSION
        sfincs_version = data.get("sfincs_version")
        version = _try_parse_version(sfincs_version)

        result = {}
        for key, value in data.items():
            field_info = model_fields.get(key)
            extra = (field_info.json_schema_extra or {}) if field_info else {}
            explicitly_set = key in self._explicit_keys or (
                update_to_latest and key == "sfincs_version"
            )

            if context.get("explicit_only") and not explicitly_set:
                continue

            # Never write None
            if value is None:
                continue

            # Evaluate condition when present
            condition = extra.get("condition")
            if condition is not None:
                try:
                    matches = eval(condition, {}, data)
                except Exception as e:  # noqa
                    raise ValueError(f"Condition eval failed for key '{key}': {e}")
                if not matches:
                    continue

            version_status = _version_status(extra, sfincs_version)
            if update_to_latest and version_status == "deprecated":
                if explicitly_set:
                    logger.warning(
                        f"'{key}' is deprecated for sfincs_version {sfincs_version}; "
                        "not written."
                    )
                continue

            # Write under the keyword the field was read/set with, unless the
            # targeted sfincs_version does not read it; then use the keyword it
            # does read, or skip the field when that version does not know it.
            file_key = self._explicit_keys.get(key, key)
            if (
                field_info is not None
                and version is not None
                and not _keyword_valid(file_key, key, extra, version)
            ):
                new_key = _keyword_for_version(key, extra, version)
                if explicitly_set and new_key != file_key:
                    logger.info(
                        f"'{file_key}' is written as '{new_key}' for "
                        f"sfincs_version {sfincs_version}."
                    )
                if new_key is None:
                    continue
                file_key = new_key

            # Decide always vs skip-if-default
            # A deprecated field is never force-written via "always"; it is only
            # kept when it was explicitly read/set (round-trip fidelity). A field
            # with an unconfirmed min_version (missing or unparsable sfincs_version) is
            # treated the same way, since we can't confirm the kernel supports it.
            suppress_always = version_status == "deprecated" or (
                version_status == "unknown" and "min_version" in extra
            )
            always = extra.get("always", False) and not suppress_always
            if (
                not always
                and not explicitly_set
                and field_info is not None
                and value == field_info.default
            ):
                continue

            # Format for the sfincs.inp text format; other values (numbers, plain
            # strings) are already fine as-is and get stringified on write.
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            elif hasattr(value, "strftime"):
                value = value.strftime("%Y%m%d %H%M%S")

            result[file_key] = value

        return result

    def to_dict(self, explicit_only: bool = False) -> dict:
        """Return the configuration as a plain dict of native Python values.

        Unlike :py:meth:`model_dump`, this is not filtered or formatted for the
        sfincs.inp file: every set field (including conditionally-irrelevant or
        default-valued ones) is included with its native Python type (e.g.
        ``datetime``, ``list[float]``), making it suitable for introspection.

        Parameters
        ----------
        explicit_only (bool):
            If True, return only fields explicitly read or set. Default is False.
        """
        data = {name: getattr(self, name) for name in type(self).model_fields}
        data.update(self.model_extra or {})
        if explicit_only:
            data = {
                key: value for key, value in data.items() if key in self._explicit_keys
            }
        return data

    def set_value(
        self, key: str, value, skip_validation: bool = False
    ) -> "SfincsConfigVariables":
        """Return a new, validated instance with a single attribute updated.

        Parameters:
        -----------
        key (str):
            The key to set the value for.
        value (Any):
            The value to set.
        skip_validation (bool):
            If True, skips pydantic validation of the new value.
        """
        return self.set_values({key: value}, skip_validation=skip_validation)

    def set_values(
        self, updates: dict, skip_validation: bool = False
    ) -> "SfincsConfigVariables":
        """Return a new, validated instance with multiple attributes updated.

        Parameters:
        -----------
        updates (dict):
            Mapping of key-value pairs to update. Legacy keywords (e.g. 'wvmfile')
            are set on their current field and written as such, unless the
            targeted ``sfincs_version`` does not read them.
        skip_validation (bool):
            If True, skips pydantic validation of the new values.
        """
        keywords = {self.resolve_key(key): key for key in updates}
        updates = {self.resolve_key(key): value for key, value in updates.items()}
        model_fields = type(self).model_fields
        unknown_keys = [key for key in updates if key not in model_fields]
        for key in unknown_keys:
            logger.warning(
                f"'{key}' is not a valid attribute of SfincsConfig. Adding it as a custom attribute."
            )

        if skip_validation:
            instance = self.model_copy()
            for key, value in updates.items():
                setattr(instance, key, value)
        else:
            new_data = self.to_dict()
            new_data.update(updates)
            instance = type(self).model_validate(new_data)

        # Keywords the targeted sfincs_version does not read are set under the
        # keyword it does
        sfincs_version = instance.sfincs_version
        explicit_keys = dict(self._explicit_keys)
        for key, keyword in keywords.items():
            if key in model_fields:
                checked = type(self)._check_keyword(key, keyword, sfincs_version)
                if checked is None:
                    logger.warning(
                        f"'{key}' is not supported for sfincs_version "
                        f"{sfincs_version}; it is not written."
                    )
                keyword = checked or key
            explicit_keys[key] = keyword

        instance._explicit_keys = explicit_keys
        instance._comments = dict(self._comments)
        return instance

    # ================================================================
    # Validators
    # ================================================================
    @field_validator("tref", "tstart", "tstop", mode="before")
    @classmethod
    def parse_custom_datetime(cls, v):
        if isinstance(v, str):
            try:
                return datetime.strptime(v.strip(), "%Y%m%d %H%M%S")
            except ValueError:
                raise ValueError(
                    f"Invalid datetime format: {v}. Expected format: YYYYMMDD HHMMSS"
                )
        return v

    @field_validator("cdwnd", "cdval", mode="before")
    @classmethod
    def parse_space_separated_floats(cls, v):
        if isinstance(v, str):
            return [float(x) for x in v.split()]
        return v


sfincs_config_variables = SfincsConfigVariables()
