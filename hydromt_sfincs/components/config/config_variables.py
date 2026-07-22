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
"""

from datetime import datetime, timedelta

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class SfincsConfigVariables(BaseSettings):
    """SFINCS configuration variables with defaults matching sfincs_input.f90."""

    class Config:
        extra = "allow"

    # ================================================================
    # Grid
    # ================================================================
    mmax: int | None = Field(
        None,
        ge=1,
        description="Number of grid cells in x-direction",
        json_schema_extra={"always": True},
    )
    nmax: int | None = Field(
        None,
        ge=1,
        description="Number of grid cells in y-direction",
        json_schema_extra={"always": True},
    )
    dx: float | None = Field(
        None,
        gt=0,
        description="Grid size in x-direction",
        json_schema_extra={"always": True},
    )
    dy: float | None = Field(
        None,
        gt=0,
        description="Grid size in y-direction",
        json_schema_extra={"always": True},
    )
    x0: float | None = Field(
        None,
        description="Origin of the grid in the x-direction",
        json_schema_extra={"always": True},
    )
    y0: float | None = Field(
        None,
        description="Origin of the grid in the y-direction",
        json_schema_extra={"always": True},
    )
    rotation: float = Field(
        0.0,
        gt=-360,
        lt=360,
        description="Rotation of the grid in degrees from the x-axis (east) in anti-clockwise direction",
    )

    # ================================================================
    # Time (always written)
    # ================================================================
    tref: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="Reference time for simulation",
        json_schema_extra={"always": True},
    )
    tstart: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="Start time for the simulation",
        json_schema_extra={"always": True},
    )
    tstop: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        + timedelta(days=1),
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
        json_schema_extra={"always": True},
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
    zsini: float = Field(
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
    qinf: float = Field(
        0.0,
        ge=0.0,
        description="Infiltration rate, spatially uniform (mm/hr)",
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
    ncinifile: str | None = Field(None, description="Netcdf initial water level file")
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


sfincs_config_variables = SfincsConfigVariables()
