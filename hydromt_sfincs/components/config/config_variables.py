from datetime import datetime, timedelta
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class SfincsConfigVariables(BaseSettings):
    # Attributes with descriptions and defaults
    #
    # - follows order of SFINCS kernel's sfincs_input.f90
    # (https://github.com/Deltares/SFINCS/blob/main/source/src/sfincs_input.f90)
    # - and description and value range of https://sfincs.readthedocs.io/en/latest/parameters.html
    #
    # Settings

    # Make sure that extra variables are allowed in the model
    class Config:
        extra = "allow"  # Allow extra fields

    mmax: int | None = Field(
        None, examples=10, ge=1, description="Number of grid cells in x-direction"
    )
    nmax: int | None = Field(
        None, examples=10, ge=1, description="Number of grid cells in y-direction"
    )
    dx: float | None = Field(
        None, examples=10.0, gt=0, description="Grid size in x-direction"
    )
    dy: float | None = Field(
        None, examples=10.0, gt=0, description="Grid size in y-direction"
    )
    x0: float | None = Field(
        None, examples=0.0, description="Origin of the grid in the x-direction"
    )
    y0: float | None = Field(
        None, examples=0.0, description="Origin of the grid in the y-direction"
    )
    rotation: float | None = Field(
        None,
        examples=0.0,
        gt=-360,
        lt=360,
        description="Rotation of the grid in degrees from the x-axis (east) in anti-clockwise direction",
    )

    # Get today's date, set hours to 0
    tref: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="Reference time for simulation (datetime)",
    )
    tstart: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        description="Start time for the simulation (datetime)",
    )
    tstop: datetime = Field(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        + timedelta(days=1),
        description="Stop time for the simulation (datetime)",
    )
    tspinup: float | None = Field(
        None,
        examples=0.0,
        ge=0.0,
        description="Duration of spinup period for boundary conditions after tstart (seconds)",
    )
    t0out: float | None = Field(
        None, examples=-999.0, description="Output start time (seconds)"
    )
    t1out: float | None = Field(
        None, examples=-999.0, description="Output stop time (seconds)"
    )

    dtmapout: float = Field(
        3600.0, ge=0.0, description="Spatial map output interval (seconds)"
    )

    dtmaxout: float = Field(
        86400.0, ge=0.0, description="Maximum map output interval (seconds)"
    )

    dtrstout: float | None = Field(
        None, examples=0.0, ge=0.0, description="Restart file output interval (seconds)"
    )

    trstout: float | None = Field(
        None,
        examples=-999.0,
        description="Restart file output after specific time (seconds)",
    )
    dthisout: float = Field(600.0, description="Timeseries output interval (seconds)")
    dtwave: float | None = Field(
        None, examples=1800.0, description="Interval of running SnapWave (seconds)"
    )
    dtwnd: float | None = Field(
        None, examples=1800.0, description="Interval of updating wind forcing (seconds)"
    )
    alpha: float = Field(
        0.5,
        ge=0.001,
        le=1.0,
        description="Numerical time step reduction for CFL-condition (-)",
    )
    theta: float = Field(
        1.0,
        ge=0.8,
        le=1.0,
        description="Numerical smoothing factor in momentum equation (-)",
    )
    hmin_cfl: float | None = Field(
        None,
        examples=0.1,
        gt=0.0,
        description="Minimum water depth for cfl condition in max timestep determination (meters)",
    )
    manning: float | None = Field(
        None,
        examples=0.04,
        gt=0.0,
        lt=0.5,
        description="Manning's n coefficient for spatially uniform roughness, if no other manning options specified (s/m^(1/3))",
    )
    manning_land: float | None = Field(
        0.04,
        gt=0.0,
        lt=0.5,
        description="Manning's n coefficient for land areas, if no other manning options specified (s/m^(1/3))",
    )
    manning_sea: float | None = Field(
        0.02,
        gt=0.0,
        lt=0.5,
        description="Manning's n coefficient for sea areas, if no other manning options specified (s/m^(1/3))",
    )
    rgh_lev_land: float | None = Field(
        0.0,
        gt=-9999,
        lt=9999,
        description="Elevation level to distinguish land and sea roughness (meters above reference level)",
    )
    zsini: float | None = Field(
        0.0,
        description="Initial water level in entire domain - where above bed level (meters)",
    )
    qinf: float | None = Field(
        0.0,
        ge=0.0,
        lt=20.0,
        description="Infiltration rate, spatially uniform and constant in time (mm/hr)",
    )
    dtmax: float | None = Field(
        None,
        examples=1000.0,
        gt=0.0,
        description="Maximum allowed internal timestep (seconds)",
    )
    huthresh: float = Field(
        0.01, gt=0.0, lt=1.0, description="Threshold water depth (meters)"
    )
    rhoa: float | None = Field(
        None, examples=1.15, gt=1.0, lt=1.5, description="Air density (kg/m^3)"
    )
    rhow: float | None = Field(
        None,
        examples=1024.0,
        gt=1000.0,
        lt=1100.0,
        description="Water density (kg/m^3)",
    )
    inputformat: str = Field("bin", description="Input file format (bin or asc)")
    outputformat: str = Field(
        "net", description="Output file format (net or asc or bin)"
    )
    outputtype_map: str | None = Field(
        None,
        examples="net",
        description="Output file format for spatial map file (net or asc or bin)",
    )
    outputtype_his: str | None = Field(
        None,
        examples="net",
        description="Output file format for observation his file (net or asc or bin)",
    )
    nc_deflate_level: int | None = Field(
        None, examples=2, description="Netcdf deflate level (-))"
    )
    bndtype: int | None = Field(
        None,
        ge=1,
        ls=1,
        description="Boundary type, only bndtype=1 is supported currently (-)",
    )
    advection: int = Field(
        1, ge=0, le=1, description="Enable advection (1: yes, 0: no)"
    )
    nfreqsig: int | None = Field(
        None,
        examples=100,
        ge=1,
        le=500,
        description="Wave maker number of frequency bins IG spectrum (-)",
    )
    freqminig: float | None = Field(
        None,
        examples=0.001,
        ge=0.0,
        le=1.0,
        description="Minimum frequency wave maker IG spectrum (Hz)",
    )
    freqmaxig: float | None = Field(
        None,
        examples=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum frequency wave maker IG spectrum (Hz)",
    )
    latitude: float | None = Field(
        0.0,
        description="Latitude of the grid center (degrees), needed for Coriolis term for projected coordinate systems",
    )
    pavbnd: float | None = Field(
        None, examples=101300.0, description="Atmospheric pressure at boundary (Pa)"
    )
    gapres: float | None = Field(
        None,
        examples=101300.0,
        description="Background atmospheric pressure used by spiderweb pressure conversion (Pa)",
    )
    baro: int | None = Field(
        1, description="Enable atmospheric pressure term (1: yes, 0: no)"
    )
    utmzone: str | None = Field(None, description="UTM zone for spatial reference (-)")
    epsg: int | None = Field(None, description="EPSG code for spatial reference system")
    advlim: float | None = Field(
        None,
        examples=1.0,
        ge=0.0,
        le=9999.9,
        description="Maximum value of the advection term in the momentum equation (-)",
    )
    slopelim: float | None = Field(
        None, examples=9999.9, ge=0.0, le=9999.9, description=">currently not used< (-)"
    )
    qinf_zmin: float | None = Field(
        None,
        examples=1.0,
        ge=-100,
        le=100,
        description="Minimum elevation level above for what cells the spatially uniform, constant in time infiltration rate 'qinf' is added (meters above reference)",
    )
    btfilter: float | None = Field(
        None,
        examples=60.0,
        ge=0.0,
        le=3600.0,
        description="Water level boundary timeseries filtering period (seconds)",
    )
    sfacinf: float | None = Field(
        None,
        examples=0.2,
        ge=0.0,
        le=1.0,
        description="Curve Number infiltration initial abstraction or the amount of water before runoff, such as infiltration, or rainfall interception by vegetation.",
    )
    # radstr > unclear if used
    crsgeo: int | None = Field(
        None,
        examples=1,
        description="Geographical coordinate system flag (1: yes, 0: no)",
    )
    coriolis: int | None = Field(
        1,
        description="Ability to turn off Coriolis term, only if crsgeo = True and latitude specified for projected CRS (1: on, 0: off)",
    )
    amprblock: int | None = Field(
        None,
        examples=1,
        ge=0,
        le=1,
        description="Use data in ampr file as block rather than linear interpolation (1: yes, 0: no)",
    )
    spwmergefrac: float | None = Field(
        None,
        examples=0.5,
        gt=0.0,
        lt=1.0,
        description="Spiderweb merge factor with background wind and pressure (-)",
    )
    usespwprecip: int | None = Field(
        None,
        examples=1,
        ge=0,
        le=1,
        description="Ability to use rainfall from spiderweb  (1: on, 0: off)",
    )
    global_: int | None = Field(
        None,
        ge=0,
        le=1,
        alias="global",
        description="Ability to make a global spherical SFINCS model that wraps 'over the edge' (1: on, 0: off)",
    )
    nuvisc: float | None = Field(
        None,
        examples=0.01,
        ge=0.0,
        description="Viscosity coefficient 'per meter of grid cell length', used if 'viscosity=1' and multiplied internally with the grid cell size (per quadtree level in quadtree mesh mode) (-)",
    )
    viscosity: int = Field(
        1, ge=0, le=1, description="Enable viscosity term (1: yes, 0: no)"
    )
    spinup_meteo: int | None = Field(
        None,
        examples=1,
        ge=0,
        le=1,
        description="Option to also apply spinup to the meteo forcing (1: on, 0: off)",
    )
    waveage: float | None = Field(
        None,
        examples=-999.0,
        description="Determine Cd with wave age based on LGX method (-)",
    )
    snapwave: int | None = Field(
        None,
        examples=0,
        ge=0,
        le=1,
        description="Option to turn on the determination of spectral wave conditions and include incident wave-induced setup through the coupled internal SnapWave solver (1: on, 0: off)",
    )
    # dtoutfixed > currently not used
    wmtfilter: float | None = Field(
        None,
        examples=600.0,
        ge=0.0,
        le=3600.0,
        description="Filtering duration for wave maker to determine mean water level component (-)",
    )
    wmfred: float | None = Field(
        None,
        examples=0.99,
        description="Filtering variable in wave maker to determine mean water level component (-)",
    )
    wmsignal: str | None = Field(
        None,
        examples="spectrum",
        description="Wavemaker options ('spectrum' or 'mon(ochromatic)')",
    )
    advection_scheme: str | None = Field(
        None, examples="upw1", description="Wavemaker options ('upw1' or 'original')"
    )
    btrelax: float | None = Field(
        None,
        examples=3600.0,
        ge=0.0,
        le=10800.0,
        description="Internal parameter of SFINCS for relaxation in uvmean (s)",
    )
    wiggle_suppression: int | None = Field(
        None,
        examples=1,
        ge=0,
        le=1,
        description="Option to turn on the wiggle surpression (1: on, 0: off)",
    )
    wiggle_factor: float | None = Field(
        None,
        examples=0.1,
        ge=0.0,
        description="Wiggle suppresion factor (-)",
    )
    wiggle_threshold: float | None = Field(
        None,
        examples=0.1,
        ge=0.0,
        description="Wiggle suppresion minimum depth threshold (-)",
    )
    friction2d: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Option to turn on the 2D friction component in the momentum equation (1: on, 0: off)",
    )
    advection_mask: int | None = Field(
        None,
        examples=1,
        ge=0,
        le=1,
        description="Option to turn on the advection_mask(1: on, 0: off)",
    )
    use_bcafile: int | None = Field(
        None,
        examples=1,
        ge=0,
        le=1,
        description="Option to turn on the use of the tidal boundary condition file (1: on, 0: off)",
    )
    #
    # Domain
    #
    qtrfile: str | None = Field(
        None, examples="sfincs.nc", description="Name of the quadtree file"
    )
    depfile: str | None = Field(None, description="Name of the depth file")
    inifile: str | None = Field(
        None, description="Name of the initial water level condition file"
    )
    rstfile: str | None = Field(None, description="Name of the restart file")
    ncinifile: str | None = Field(
        None, description="Name of the Netcdf initial water level condition file"
    )
    mskfile: str | None = Field(None, description="Name of the mask file")
    indexfile: str | None = Field(None, description="Name of the index file")
    # cstfile: > not used
    sbgfile: str | None = Field(None, description="Name of the subgrid file")
    thdfile: str | None = Field(None, description="Name of the thin dam structure file")
    weirfile: str | None = Field(None, description="Name of the weir structure file")
    manningfile: str | None = Field(None, description="Name of the Manning's n file")
    drnfile: str | None = Field(None, description="Name of the drainage structure file")
    volfile: str | None = Field(None, description="Name of the storage volume file")
    #
    # Forcing
    #
    bndfile: str | None = Field(
        None, description="Name of the water level boundary points file"
    )
    bzsfile: str | None = Field(
        None, description="Name of the water level time-series file"
    )
    bcafile: str | None = Field(
        None, description="Name of the tidal boundary component file"
    )
    bzifile: str | None = Field(
        None, description="Name of the individual wave water level time-series file"
    )
    bdrfile: str | None = Field(
        None, description="Name of the downstream river boundary input file"
    )
    wfpfile: str | None = Field(
        None, description="Name of the wavemaker location input points file"
    )
    whifile: str | None = Field(
        None, description="Name of the wavemaker IG wave height input file"
    )
    wtifile: str | None = Field(
        None, description="Name of the wavemaker IG wave period input file"
    )
    wstfile: str | None = Field(
        None, description="Name of the wavemaker setup input file"
    )
    srcfile: str | None = Field(
        None, description="Name of the discharge input points file"
    )
    disfile: str | None = Field(
        None, description="Name of the discharge input time-series file"
    )
    spwfile: str | None = Field(
        None, description="Name of the spiderweb tropical cyclone file"
    )
    wndfile: str | None = Field(
        None, description="Name of the spatially uniform wind file"
    )
    prcfile: str | None = Field(
        None, description="Name of the spatially uniform precipitation file"
    )
    precipfile: str | None = Field(
        None,
        description="LEGACY OPTION - Name of the spatially uniform precipitation file > now use: prcfile",
    )
    amufile: str | None = Field(
        None, description="Name of the u-component of the wind file"
    )
    amvfile: str | None = Field(
        None, description="Name of the v-component of the wind file"
    )
    ampfile: str | None = Field(
        None, description="Name of the atmospheric pressure file"
    )
    amprfile: str | None = Field(None, description="Name of the precipitation file")
    z0lfile: str | None = Field(
        None, description="Name of the wind reduction over land input file"
    )
    wvmfile: str | None = Field(
        None, description="Name of the wave maker input points file"
    )
    qinffile: str | None = Field(
        None,
        description="Name of the spatially-uniform, constant in time infiltration file",
    )
    #
    # Curve Number files
    #
    scsfile: str | None = Field(
        None,
        description="Name of the Curve Number infiltration method A - maximum soil moisture storage capacity file",
    )
    smaxfile: str | None = Field(
        None,
        description="Name of the Curve Number infiltration method B - maximum soil moisture storage capacity file",
    )
    sefffile: str | None = Field(
        None,
        description="Name of the Curve Number infiltration method B - initial soil moisture storage capacity file",
    )
    #
    # Green and Ampt files
    #
    psifile: str | None = Field(
        None,
        description="Name of the Green and Ampt infiltration  method - suction head file",
    )
    sigmafile: str | None = Field(
        None,
        description="Name of the Green and Ampt infiltration method - maximum moisture deficit file",
    )
    ksfile: str | None = Field(
        None,
        description="Name of the Green and Ampt infiltration method - saturated hydraulic conductivity file",
    )
    #
    # Horton file
    #
    f0file: str | None = Field(
        None,
        description="Name of the Horton infiltration method - Maximum (Initial) Infiltration Capacity file",
    )
    fcfile: str | None = Field(
        None,
        description="Name of the Horton infiltration method - Minimum (Asymptotic) Infiltration Rate file",
    )
    kdfile: str | None = Field(
        None,
        description="Name of the Horton infiltration method - empirical constant (hr-1) of decay file",
    )
    horton_kr_kd: float | None = Field(
        None, description="Horton infiltration recovery vs decay ration (-)"
    )
    #
    # Netcdf input
    #
    netbndbzsbzifile: str | None = Field(
        None, description="Name of the Netcdf type water level input file"
    )
    netsrcdisfile: str | None = Field(
        None, description="Name of the Netcdf type discharge input file"
    )
    netamuamvfile: str | None = Field(
        None, description="Name of the Netcdf type wind amu-amv input file"
    )
    netamprfile: str | None = Field(
        None, description="Name of the Netcdf type precipitation input file"
    )
    netampfile: str | None = Field(
        None, description="Name of the Netcdf type atmoshperic pressure input file"
    )
    netspwfile: str | None = Field(
        None,
        description="Name of the Netcdf type spiderweb tropical cyclone input file",
    )
    #
    # Output
    #
    obsfile: str | None = Field(
        None, description="Name of the observation points input file"
    )
    crsfile: str | None = Field(
        None, description="Name of the cross-section lines input file"
    )
    storevelmax: int | None = Field(
        0,
        ge=0,
        le=1,
        description="Option to write maximum velocity output to netcdf map output (1: yes, 0: no)",
    )
    storefluxmax: int | None = Field(
        0,
        ge=0,
        le=1,
        description="Option to write maximum flux output to netcdf map output (1: yes, 0: no)",
    )
    storevel: int | None = Field(
        0,
        ge=0,
        le=1,
        description="Option to write instantaneous velocity output to netcdf map output (1: yes, 0: no)",
    )
    storecumprcp: int | None = Field(
        0,
        ge=0,
        le=1,
        description="Option to write cumulative precipitation output to netcdf map output (1: yes, 0: no)",
    )
    storetwet: int | None = Field(
        None,
        examples=0,
        ge=0,
        le=1,
        description="Option to write 'twet' time wet output to netcdf map output (1: yes, 0: no)",
    )
    storehsubgrid: int | None = Field(
        None,
        examples=0,
        ge=0,
        le=1,
        description="Option to write -approximated- depth output for Subgrid models to netcdf map output (1: yes, 0: no)",
    )
    twet_threshold: float | None = Field(
        None,
        examples=0.01,
        ge=0.0,
        description="Time wet 'twet' minimum depth threshold (m)",
    )
    store_tsunami_arrival_time: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Option to write tsunami arrival time output to netcdf map output (1: yes, 0: no)",
    )
    tsunami_arrival_threshold: float | None = Field(
        None,
        ge=0.0,
        description="Tsunami arrival time minimum depth threshold (m)",
    )
    storeqdrain: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Option to write discharge through drainage structure output to netcdf map output (1: yes, 0: no)",
    )
    storezvolume: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Option to write storage volume output to netcdf map output (1: yes, 0: no)",
    )
    # writeruntime > not used currently
    debug: int | None = Field(
        None,
        examples=0,
        ge=0,
        le=1,
        description="Option to turn on debug mode and write every timestep to netcdf map output (1: yes, 0: no)",
    )
    storemeteo: int | None = Field(
        0,
        ge=0,
        le=1,
        description="Option to write meteo output to netcdf map output (1: yes, 0: no)",
    )
    storemaxwind: int | None = Field(
        0,
        ge=0,
        le=1,
        description="Option to write maximum wind speed output to netcdf map output (1: yes, 0: no)",
    )
    storefw: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Option to write wave forces to netcdf map output (1: yes, 0: no)",
    )
    storewavdir: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Option to write wave direction to netcdf map output (1: yes, 0: no)",
    )
    regular_output_on_mesh: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Option to write regular grid model on quadtree mesh type output in netcdf map output (1: yes, 0: no)",
    )
    #
    # Coupled SnapWave solver related
    #
    snapwave_wind: int | None = Field(
        None,
        ge=0,
        le=1,
        description="Option in integrated SnapWave solver to turn on wind growth process (1: yes, 0: no)",
    )
    snapwave_bndfile: str | None = Field(
        None,
        description="Name of the SnapWave boundary points file",
    )
    snapwave_bhsfile: str | None = Field(
        None,
        description="Name of the SnapWave wave height time-series file",
    )
    snapwave_btpfile: str | None = Field(
        None,
        description="Name of the SnapWave wave period time-series file",
    )
    snapwave_bwdfile: str | None = Field(
        None,
        description="Name of the SnapWave wave direction time-series file",
    )
    snapwave_bdsfile: str | None = Field(
        None,
        description="Name of the SnapWave wave spreading time-series file",
    )
    #
    # Wind drag
    #
    cdnrb: int | None = Field(
        None, examples=3, description="Number of wind speed ranges for drag coefficient"
    )
    cdwnd: List[float] | None = Field(
        None,
        examples=[0.0, 28.0, 50.0],
        description="Wind speed ranges for drag coefficient (m/s)",
    )
    cdval: List[float] | None = Field(
        None,
        examples=[0.001, 0.0025, 0.0015],
        description="Drag coefficient values corresponding to cdwnd",
    )
    #
    # Other used files - NOTE: not part of nor recognized by SFINCS kernel itself!!!
    # corfile: str | None = Field(None, description="Name of the correction file")

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
