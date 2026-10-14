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
* ``json_schema_extra={"min_version": "<X.Y.Z>"}`` — field was introduced in
  this SFINCS kernel version; not written when ``sfincs_version`` is older.
* ``json_schema_extra={"max_version": "<X.Y.Z>"}`` — field is deprecated as of
  this kernel version; when ``sfincs_version`` is newer it is only kept if
  explicitly present in the source file (never force-written via ``always``).
"""

import logging
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_serializer
from pydantic_settings import BaseSettings

from hydromt_sfincs import MIN_SUPPORTED_SFINCS_VERSION

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


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a simple 'X.Y.Z'-style version string into a comparable tuple."""
    return tuple(int(part) for part in version.strip().split("."))


def _version_status(
    extra: dict,
    user_sfincs_version: str | None,
    min_supported_version: str = MIN_SUPPORTED_SFINCS_VERSION,
) -> str:
    """Classify a field as compatible, unsupported, deprecated, or unknown."""
    min_version = extra.get("min_version")
    max_version = extra.get("max_version")
    if not min_version and not max_version:
        return "ok"
    if user_sfincs_version is None:
        # Unversioned/legacy file: a field with a max_version is inherently a
        # legacy/deprecated key regardless of version; a min_version-only field
        # is of unknown validity since we can't confirm the target kernel supports it.
        return "deprecated" if max_version is not None else "unknown"
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
        json_schema_extra={"always": True, "min_version": "2.3.0"},
    )
    dtout: float | None = Field(
        None,
        description="[DEPRECATED] legacy alias of dtmapout",
        json_schema_extra={"max_version": "2.3.0", "new_name": "dtmapout"},
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
        json_schema_extra={"always": True},
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
        json_schema_extra={"always": True},
    )
    nuviscfac: float = Field(
        100.0,
        ge=0.0,
        description="Viscosity factor (-)",
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
    )
    uvmax: float = Field(
        1000.0,
        description="Maximum velocity (m/s)",
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
        json_schema_extra={"always": True},
    )
    rhow: float = Field(
        1024.0,
        gt=0.0,
        description="Water density (kg/m^3)",
        json_schema_extra={"always": True},
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
        json_schema_extra={"always": True},
    )
    pavbnd: float = Field(
        0.0,
        description="Atmospheric pressure at boundary (Pa)",
        json_schema_extra={"always": True},
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
        json_schema_extra={"always": True},
    )
    use_bcafile: int = Field(
        1,
        ge=0,
        le=1,
        description="Use tidal boundary condition file (1: on, 0: off)",
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
    nfreqsig: int = Field(
        100,
        ge=1,
        le=500,
        description="Wave maker number of frequency bins IG spectrum (-)",
    )
    freqminig: float = Field(
        0.0,
        ge=0.0,
        description="Minimum frequency wave maker IG spectrum (Hz)",
    )
    freqmaxig: float = Field(
        0.1,
        ge=0.0,
        description="Maximum frequency wave maker IG spectrum (Hz)",
    )
    wmtfilter: float = Field(
        600.0,
        ge=0.0,
        description="Filtering duration for wave maker mean water level (s)",
    )
    wmfred: float = Field(
        0.99,
        description="Filtering variable in wave maker (-)",
    )
    wmsignal: str = Field(
        "spectrum",
        description="Wavemaker signal type ('spectrum' or 'mon')",
    )
    wmhmin: float = Field(
        0.1,
        description="Wavemaker minimum water depth (m)",
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
    )
    factor_pres: float = Field(
        1.0,
        description="Pressure forcing scale factor (-)",
    )
    factor_prcp: float = Field(
        1.0,
        description="Precipitation forcing scale factor (-)",
    )
    factor_spw_size: float = Field(
        1.0,
        description="Spiderweb size scale factor (-)",
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
        json_schema_extra={"condition": "snapwave == 1"},
    )
    snapwave_use_nearest: int = Field(
        1,
        ge=0,
        le=1,
        description="SnapWave use nearest interpolation (1: on, 0: off)",
        json_schema_extra={"condition": "snapwave == 1"},
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
    snapwave_nrsweeps: int = Field(
        4,
        ge=1,
        description="SnapWave maximum number of sweeps (-)",
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
    wave_enhanced_roughness: int = Field(
        0,
        ge=0,
        le=1,
        description="Wave enhanced roughness (1: on, 0: off)",
    )

    # ================================================================
    # Non-hydrostatic
    # ================================================================
    nonh: int = Field(
        0,
        ge=0,
        le=1,
        description="Enable non-hydrostatic mode (1: on, 0: off)",
    )
    nh_fnudge: float = Field(
        0.9,
        description="Non-hydrostatic nudge factor (-)",
        json_schema_extra={"condition": "nonh == 1"},
    )
    nh_tstop: float = Field(
        -999.0,
        description="Non-hydrostatic stop time (seconds)",
        json_schema_extra={"condition": "nonh == 1"},
    )
    nh_tol: float = Field(
        0.001,
        description="Non-hydrostatic tolerance (-)",
        json_schema_extra={"condition": "nonh == 1"},
    )
    nh_itermax: int = Field(
        100,
        description="Non-hydrostatic maximum iterations (-)",
        json_schema_extra={"condition": "nonh == 1"},
    )

    # ================================================================
    # Structures
    # ================================================================
    structure_relax: float = Field(
        10.0,
        description="Structure relaxation factor (-)",
    )

    # ================================================================
    # Bathtub
    # ================================================================
    bathtub: int = Field(
        0,
        ge=0,
        le=1,
        description="Enable bathtub mode (1: on, 0: off)",
    )
    bathtub_fachs: float = Field(
        0.2,
        description="Bathtub wave height factor (-)",
        json_schema_extra={"condition": "bathtub == 1"},
    )
    bathtub_dt: float = Field(
        -999.0,
        description="Bathtub time step (seconds)",
        json_schema_extra={"condition": "bathtub == 1"},
    )

    # ================================================================
    # Other settings
    # ================================================================
    sfincs_version: str | None = Field(
        None,
        description="Target SFINCS kernel version; controls min_version/max_version field handling",
    )
    global_: int = Field(
        0,
        ge=0,
        le=1,
        alias="global",
        description="Global spherical model that wraps over the edge (1: on, 0: off)",
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
        json_schema_extra={"min_version": "2.3.0"},
    )
    crs: int | None = Field(
        None,
        description="[DEPRECATED] legacy alias of epsg",
        json_schema_extra={"max_version": "2.0.0", "new_name": "epsg"},
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
    )
    h73table: int = Field(
        0,
        ge=0,
        le=1,
        description="Use h73 table (1: on, 0: off)",
    )

    # ================================================================
    # Wind drag
    # ================================================================
    cdnrb: int = Field(
        3,
        description="Number of wind speed ranges for drag coefficient",
        json_schema_extra={"always": True},
    )
    cdwnd: list[float] | None = Field(
        [0.0, 28.0, 50.0],
        description="Wind speed ranges for drag coefficient (m/s)",
        json_schema_extra={"always": True},
    )
    cdval: list[float] | None = Field(
        [0.001, 0.0025, 0.0025],
        description="Drag coefficient values corresponding to cdwnd",
        json_schema_extra={"always": True},
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

    # ================================================================
    # Forcing files
    # ================================================================
    bndfile: str | None = Field(None, description="Water level boundary points file")
    bzsfile: str | None = Field(None, description="Water level time-series file")
    bcafile: str | None = Field(None, description="Tidal boundary component file")
    bzifile: str | None = Field(None, description="Individual wave water level file")
    bdrfile: str | None = Field(None, description="Downstream river boundary file")
    wfpfile: str | None = Field(None, description="Wavemaker location points file")
    whifile: str | None = Field(None, description="Wavemaker IG wave height file")
    wtifile: str | None = Field(None, description="Wavemaker IG wave period file")
    wstfile: str | None = Field(None, description="Wavemaker setup file")
    srcfile: str | None = Field(None, description="Discharge input points file")
    disfile: str | None = Field(None, description="Discharge input time-series file")
    spwfile: str | None = Field(None, description="Spiderweb tropical cyclone file")
    wndfile: str | None = Field(None, description="Spatially uniform wind file")
    prcfile: str | None = Field(
        None, description="Spatially uniform precipitation file"
    )
    precipfile: str | None = Field(
        None, description="LEGACY precipitation file (use prcfile)"
    )
    amufile: str | None = Field(None, description="Wind u-component file")
    amvfile: str | None = Field(None, description="Wind v-component file")
    ampfile: str | None = Field(None, description="Atmospheric pressure file")
    amprfile: str | None = Field(None, description="Precipitation file")
    z0lfile: str | None = Field(None, description="Wind reduction over land file")
    wvmfile: str | None = Field(None, description="Wave maker input points file")
    qinffile: str | None = Field(None, description="Infiltration file")
    infiltration_file: str | None = Field(
        None, description="Infiltration file (alternative)"
    )
    infiltration_type: str | None = Field(None, description="Infiltration type")

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

    # ================================================================
    # Output observation files
    # ================================================================
    obsfile: str | None = Field(None, description="Observation points file")
    crsfile: str | None = Field(None, description="Cross-section lines file")
    rugfile: str | None = Field(None, description="Runup gauges file")

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
        0, ge=0, le=1, description="Write mean water depth output (1: yes, 0: no)"
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
        1, ge=0, le=1, description="Write drainage discharge output (1: yes, 0: no)"
    )
    storezvolume: int = Field(
        0, ge=0, le=1, description="Write storage volume output (1: yes, 0: no)"
    )
    storestoragevolume: int = Field(
        0, ge=0, le=1, description="Write total storage volume output (1: yes, 0: no)"
    )
    store_tsunami_arrival_time: int = Field(
        0, ge=0, le=1, description="Write tsunami arrival time output (1: yes, 0: no)"
    )
    store_dynamic_bed_level: int = Field(
        0, ge=0, le=1, description="Write dynamic bed level output (1: yes, 0: no)"
    )
    regular_output_on_mesh: int = Field(
        1,
        ge=0,
        le=1,
        description="Write quadtree without refinement on quadtree mesh (1: yes) or regular grid (0: no)",
    )
    twet_threshold: float = Field(
        0.01, ge=0.0, description="Time-wet minimum depth threshold (m)"
    )
    tsunami_arrival_threshold: float = Field(
        0.01, ge=0.0, description="Tsunami arrival minimum depth threshold (m)"
    )
    timestep_analysis: int = Field(
        0, ge=0, le=1, description="Enable timestep analysis output (1: yes, 0: no)"
    )
    debug: int = Field(
        0, ge=0, le=1, description="Debug mode — write every timestep (1: yes, 0: no)"
    )
    percentage_done: int = Field(
        5, ge=1, description="Percentage done output interval (%)"
    )

    # ================================================================
    # Read / write
    # ================================================================
    # Keys that were explicitly present in the source file (or set afterwards);
    # these always round-trip on write(), even if they equal the field default.
    _explicit_keys: set[str] = PrivateAttr(default_factory=set)
    # Inline trailing comment (after '#') read per key from the source file, used
    # by write(write_comments=True) in preference over the schema description.
    _comments: dict[str, str] = PrivateAttr(default_factory=dict)

    @classmethod
    def read(
        cls,
        filename: str | Path,
    ) -> "SfincsConfigVariables":
        """Read a SFINCS input file (sfincs.inp) into a validated instance."""
        filename = Path(filename)
        inp_dict, comments = _read_config(filename=filename)

        # Warn about keys not recognized by the schema; they are kept as
        # pass-through attributes (Config.extra = "allow") and always rewritten.
        model_fields = cls.model_fields
        sfincs_version = inp_dict.get("sfincs_version")
        if _is_below_min_supported(sfincs_version):
            message = (
                f"sfincs_version {sfincs_version} is below the minimum supported "
                f"SFINCS version {MIN_SUPPORTED_SFINCS_VERSION}."
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            logger.warning(message)

        for key, value in list(inp_dict.items()):
            field_info = model_fields.get(key)
            if field_info is None:
                continue
            extra = field_info.json_schema_extra or {}
            status = _version_status(extra, sfincs_version)
            if sfincs_version is None or status not in {"too_old", "deprecated"}:
                continue

            new_name = extra.get("new_name")
            new_field = model_fields.get(new_name) if new_name is not None else None
            new_extra = new_field.json_schema_extra or {} if new_field else {}
            replacement_supported = (
                new_field is not None
                and _version_status(new_extra, sfincs_version) == "ok"
            )
            if replacement_supported and new_name not in inp_dict:
                inp_dict[new_name] = value
                if key in comments:
                    comments[new_name] = comments[key]
            inp_dict.pop(key)
            comments.pop(key, None)
            logger.warning(
                f"'{key}' is not supported for sfincs_version {sfincs_version}; "
                f"removed{f' or migrated to {new_name!r}' if new_name else ''}."
            )

        unknown_keys = sorted(set(inp_dict) - set(model_fields))
        if unknown_keys:
            logger.warning(
                f"Unrecognized key(s) in {filename}: {unknown_keys}. "
                "They will be kept and rewritten as-is."
            )

        # Warn about keys that are outside the range of the targeted sfincs_version
        for key in inp_dict:
            field_info = model_fields.get(key)
            if field_info is None:
                continue
            extra = field_info.json_schema_extra or {}
            status = _version_status(extra, sfincs_version)
            if status == "deprecated":
                message = f"'{key}' is deprecated for sfincs_version {sfincs_version}."
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                logger.warning(message)
            elif status == "too_old":
                logger.warning(
                    f"'{key}' is not yet supported for sfincs_version {sfincs_version}."
                )

        # Full pydantic validation (type coercion, field constraints, validators)
        instance = cls.model_validate(inp_dict)
        instance._explicit_keys = set(inp_dict.keys())
        instance._comments = comments
        return instance

    def write(
        self,
        filename: str | Path,
        write_description: bool = False,
        write_comments: bool = False,
        explicit_only: bool = False,
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
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        model_fields = type(self).model_fields
        # model_dump() applies the write-only filtering/formatting via the
        # _serialize_for_write model_serializer below.
        data_dict = self.model_dump(context={"explicit_only": explicit_only})

        with open(filename, "w") as fid:
            for key, value in data_dict.items():
                string = f"{key.ljust(20)} = {value}"

                comment = self._comments.get(key) if write_comments else None
                if not comment and write_description and key in model_fields:
                    comment = model_fields[key].description
                if comment:
                    string = string.ljust(50) + f" # {comment}"

                fid.write(string + "\n")

    @model_serializer(mode="wrap")
    def _serialize_for_write(self, handler, info) -> dict:
        """Filter and format field values for writing to sfincs.inp.

        Drops fields whose ``condition``/``min_version``/``max_version`` policy
        excludes them, or that equal their default and were never explicitly
        read/set; formats the remaining values (numbers, joined lists, dates) as
        plain strings/numbers ready to write. Used exclusively by :py:meth:`write`;
        use :py:meth:`to_dict` for the unfiltered, native-typed configuration.
        """
        data = handler(self)
        model_fields = type(self).model_fields
        sfincs_version = data.get("sfincs_version")

        result = {}
        for key, value in data.items():
            if (
                info.context
                and info.context.get("explicit_only")
                and key not in self._explicit_keys
            ):
                continue

            # Never write None
            if value is None:
                continue

            field_info = model_fields.get(key)
            extra = (field_info.json_schema_extra or {}) if field_info else {}

            # Evaluate condition when present
            condition = extra.get("condition")
            if condition is not None:
                try:
                    matches = eval(condition, {}, data)
                except Exception as e:  # noqa
                    raise ValueError(f"Condition eval failed for key '{key}': {e}")
                if not matches:
                    continue

            # Skip fields not (yet) valid for the targeted sfincs_version
            version_status = _version_status(extra, sfincs_version)
            if version_status == "too_old":
                continue

            # Decide always vs skip-if-default
            # A deprecated field is never force-written via "always"; it is only
            # kept when it was explicitly read/set (round-trip fidelity). A field
            # with an unconfirmed min_version (unversioned file) is treated the
            # same way, since we can't confirm the target kernel supports it.
            suppress_always = version_status == "deprecated" or (
                version_status == "unknown" and "min_version" in extra
            )
            always = extra.get("always", False) and not suppress_always
            explicitly_set = key in self._explicit_keys
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

            result[key] = value

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
            Mapping of key-value pairs to update.
        skip_validation (bool):
            If True, skips pydantic validation of the new values.
        """
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

        instance._explicit_keys = self._explicit_keys | set(updates.keys())
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
