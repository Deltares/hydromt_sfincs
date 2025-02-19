from datetime import datetime
from typing import List, Optional, Dict, Any
from ast import literal_eval
from pydantic import Field
from pydantic_settings import BaseSettings

from hydromt.model import Model
from hydromt.model.components import ModelComponent


class SfincsInputVariables(BaseSettings):
    # Attributes with descriptions and defaults
    # - follow order of SFINCS kernel's sfincs_input.f90
    # - and description of https://sfincs.readthedocs.io/en/latest/parameters.html
    #
    # Settings:
    mmax: int = Field(10, ge=1, description="Number of grid cells in x-direction")
    nmax: int = Field(10, ge=1, description="Number of grid cells in y-direction")
    dx: float = Field(10.0, gt=0, description="Grid size in x-direction")
    dy: float = Field(10.0, gt=0, description="Grid size in y-direction")
    x0: float = Field(0.0, description="Origin of the grid in the x-direction")
    y0: float = Field(0.0, description="Origin of the grid in the y-direction")
    rotation: float = Field(
        0.0,
        gt=-360,
        lt=360,
        description="Rotation of the grid in degrees from the x-axis (east) in anti-clockwise direction",
    )
    tref: datetime = Field(
        datetime(2010, 2, 1, 0, 0, 0),
        description="Reference time for simulation (datetime)",
    )
    tstart: datetime = Field(
        datetime(2010, 2, 1, 0, 0, 0),
        description="Start time for the simulation (datetime)",
    )
    tstop: datetime = Field(
        datetime(2010, 2, 2, 0, 0, 0),
        description="Stop time for the simulation (datetime)",
    )
    tspinup: float = Field(
        0.0,
        ge=0,
        description="Duration of spinup period for boundary conditions after tstart (seconds)",
    )
    t0out: Optional[datetime] = Field(None, description="Output start time (datetime)")
    t1out: Optional[datetime] = Field(None, description="Output stop time (datetime)")
    dtout: float = Field(
        3600.0, ge=0, description="Spatial map output interval (seconds)"
    )
    dtmaxout: float = Field(
        86400, ge=0, description="Maximum map output interval (seconds)"
    )  # FIXME - TL: why was 'int'?
    dtrstout: float = Field(
        0.0, ge=0, description="Restart file output interval (seconds)"
    )
    trstout: float = Field(
        -999.0, description="Restart file output after specific time (seconds)"
    )
    dthisout: float = Field(600.0, description="Timeseries output interval (seconds)")
    dtwave: float = Field(None, description="Interval of running SnapWave (seconds)")
    dtwnd: float = Field(
        None, description="Interval of updating wind forcing (seconds)"
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
    hmin_cfl: float = Field(
        0.1,
        gt=0.0,
        description="Minimum water depth for cfl condition in max timestep determination (meters)",
    )
    hmin_uv: float = Field(
        0.1,
        gt=0.0,
        description="Minimum water depth for uv velocity determination in momentum equation (meters)",
    )
    manning: float = Field(
        None,
        gt=0.0,
        lt=0.5,
        description="Manning's n coefficient for spatially uniform roughness, if no other manning options specified (s/m^(1/3))",
    )
    manning_land: float = Field(
        None,
        gt=0.0,
        lt=0.5,
        description="Manning's n coefficient for land areas, if no other manning options specified (s/m^(1/3))",
    )
    manning_sea: float = Field(
        None,
        gt=0.0,
        lt=0.5,
        description="Manning's n coefficient for sea areas, if no other manning options specified (s/m^(1/3))",
    )
    rgh_lev_land: float = Field(
        None,
        gt=-9999,
        lt=9999,
        description="Elevation level to distinguish land and sea roughness (meters above reference level)",
    )
    zsini: float = Field(
        0.0,
        description="Initial water level in entire domain - where above bed level (meters)",
    )
    qinf: float = Field(
        None,
        gt=0.0,
        lt=20.0,
        description="Infiltration rate, spatially uniform and constant in time (mm/hr)",
    )
    dtmax: float = Field(
        60.0, gt=0.0, description="Maximum allowed internal timestep (seconds)"
    )
    huthresh: float = Field(
        0.01, gt=0.0, lt=1.0, description="Threshold water depth (meters)"
    )
    rhoa: float = Field(None, gt=1.0, lt=1.5, description="Air density (kg/m^3)")
    rhow: float = Field(
        None, gt=1000.0, lt=1100.0, description="Water density (kg/m^3)"
    )
    inputformat: str = Field("bin", description="Input file format (bin or asc)")
    outputformat: str = Field(
        "net", description="Output file format (net or asc or bin)"
    )
    outputtype_map: str = Field(
        None, description="Output file format for spatial map file (net or asc or bin)"
    )
    outputtype_his: str = Field(
        None,
        description="Output file format for observation his file (net or asc or bin)",
    )
    nc_deflate_level: int = Field(None, description="Netcdf deflate level (-))")
    bndtype: int = Field(
        None, ge=1, ls=1, description="Boundary type, only bndtype=1 is supported (-)"
    )
    advection: int = Field(
        1, ge=0, le=1, description="Enable advection (1: yes, 0: no)"
    )
    nfreqsig: int = Field(
        None,
        ge=0,
        le=500,
        description="Wave maker number of frequency bins IG spectrum (-)",
    )
    freqminig: float = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum frequency wave maker IG spectrum (Hz)",
    )
    freqmaxig: float = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum frequency wave maker IG spectrum (Hz)",
    )
    latitude: float = Field(None, description="Latitude of the grid center (degrees)")
    pavbnd: float = Field(None, description="Atmospheric pressure at boundary (Pa)")
    gapres: float = Field(
        None,
        description="Background atmospheric pressure used by spiderweb pressure conversion (Pa)",
    )
    baro: int = Field(
        None, description="Enable atmospheric pressure term (1: yes, 0: no)"
    )
    utmzone: Optional[str] = Field(
        None, description="UTM zone for spatial reference (-)"
    )
    epsg: Optional[int] = Field(
        None, description="EPSG code for spatial reference system"
    )
    stopdepth: float = Field(
        100.0,
        gt=0.0,
        lt=15000,
        description="Water depth based on which the minimal time step is determined below which the simulation is classified as unstable and stopped (meters)",
    )
    advlim: float = Field(
        None,
        ge=0.0,
        le=9999.9,
        description="Maximum value of the advection term in the momentum equation (-)",
    )
    slopelim: float = Field(
        None, ge=0.0, le=9999.9, description=">currently not used< (-)"
    )
    qinf_zmin: float = Field(
        None,
        ge=-100,
        le=100,
        description="Minimum elevation level above for what cells the spatially uniform, constant in time infiltration rate 'qinf' is added (meters above reference)",
    )
    btfilter: float = Field(
        None,
        ge=0.0,
        le=3600.0,
        description="Water level boundary timeseries filtering period (seconds)",
    )
    sfacinf: float = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Curve Number infiltration initial abstraction or the amount of water before runoff, such as infiltration, or rainfall interception by vegetation.",
    )
    # radstr > unclear if used
    crsgeo: int = Field(
        None, description="Geographical coordinate system flag (1: yes, 0: no)"
    )
    coriolis: int = Field(
        None,
        description="Ability to turn off Coriolis term, only if crsgeo = True (1: on, 0: off)",
    )
    amprblock: int = Field(
        1,
        description="Use data in ampr file as block rather than linear interpolation (1: yes, 0: no)",
    )

    spwmergefrac: float = Field(
        None,
        gt=0.0,
        lt=1.0,
        description="Spiderweb merge factor with background wind and pressure (-)",
    )
    usespwprecip: int = Field(
        None,
        description="Ability to use rainfall from spiderweb  (1: on, 0: off)",
    )
    # global: int = Field(
    #     None,
    #     description="Ability to make a global spherical SFINCS model that wraps 'over the edge' (1: on, 0: off)",
    # ) #FIXME > clash with 'global' keyword, leave out for now
    nuvisc: float = Field(
        None,
        ge=0.0,
        description="Viscosity coefficient 'per meter of grid cell length', used if 'viscosity=1' and multiplied internally with the grid cell size (per quadtree level in quadtree mesh mode) (-)",
    )
    viscosity: int = Field(1, description="Enable viscosity term (1: yes, 0: no)")

    # TODO:
    # spinup_meteo
    # waveage
    # snapwave
    # dtoutfixed
    # wmtfilter
    # wmfred
    # wmsignal
    # wmrandom
    # advection_scheme
    # btrelax
    # wiggle_suppression > used?
    # wiggle_factor
    # wiggle_threshold

    depfile: Optional[str] = Field(None, description="Path to the depth file")
    mskfile: Optional[str] = Field(None, description="Path to the mask file")
    indexfile: Optional[str] = Field(None, description="Path to the index file")
    cstfile: Optional[str] = Field(None, description="Path to the coastline file")
    bndfile: Optional[str] = Field(None, description="Path to the boundary file")
    bzsfile: Optional[str] = Field(None, description="Path to the bathymetry file")
    bzifile: Optional[str] = Field(
        None, description="Path to the initial bathymetry file"
    )
    bwvfile: Optional[str] = Field(None, description="Path to the wave file")
    bhsfile: Optional[str] = Field(
        None, description="Path to the significant wave height file"
    )
    btpfile: Optional[str] = Field(
        None, description="Path to the bottom topography file"
    )
    bwdfile: Optional[str] = Field(None, description="Path to the wind file")
    bdsfile: Optional[str] = Field(None, description="Path to the discharge file")
    bcafile: Optional[str] = Field(None, description="Path to the calibration file")
    corfile: Optional[str] = Field(None, description="Path to the correction file")
    srcfile: Optional[str] = Field(None, description="Path to the source file")
    disfile: Optional[str] = Field(None, description="Path to the distribution file")
    inifile: Optional[str] = Field(
        None, description="Path to the initial conditions file"
    )
    sbgfile: Optional[str] = Field(None, description="Path to the subgrid file")
    qtrfile: Optional[str] = Field(None, description="Path to the quarter file")
    spwfile: Optional[str] = Field(None, description="Path to the spectral wave file")
    amufile: Optional[str] = Field(
        None, description="Path to the u-component of the wind file"
    )
    amvfile: Optional[str] = Field(
        None, description="Path to the v-component of the wind file"
    )
    ampfile: Optional[str] = Field(None, description="Path to the pressure file")
    amprfile: Optional[str] = Field(None, description="Path to the precipitation file")
    wndfile: Optional[str] = Field(None, description="Path to the wind file")
    precipfile: Optional[str] = Field(
        None, description="Path to the precipitation file"
    )
    obsfile: Optional[str] = Field(None, description="Path to the observation file")
    crsfile: Optional[str] = Field(
        None, description="Path to the coordinate reference system file"
    )
    thdfile: Optional[str] = Field(None, description="Path to the threshold file")
    manningfile: Optional[str] = Field(None, description="Path to the Manning's n file")
    scsfile: Optional[str] = Field(
        None, description="Path to the soil conservation service file"
    )
    rstfile: Optional[str] = Field(None, description="Path to the restart file")
    wfpfile: Optional[str] = Field(
        None, description="Path to the wind forcing parameter file"
    )
    whifile: Optional[str] = Field(
        None, description="Path to the wave height input file"
    )
    wtifile: Optional[str] = Field(
        None, description="Path to the wave period input file"
    )
    wstfile: Optional[str] = Field(None, description="Path to the wave spectrum file")

    cdnrb: int = Field(
        None, description="Number of wind speed ranges for drag coefficient"
    )
    cdwnd: List[float] = Field(
        None,
        description="Wind speed ranges for drag coefficient (m/s)",
    )
    cdval: List[float] = Field(
        None,
        description="Drag coefficient values corresponding to cdwnd",
    )


class SfincsInput(ModelComponent):
    """Class to read and write SFINCS input files."""

    def __init__(self, model: Model):
        self._filename = "sfincs.inp"
        self._data: SfincsInputVariables = None
        super().__init__(model=model)

    @property
    def data(self) -> SfincsInputVariables:
        """Return the SfincsInput object."""
        if self._data is None:
            self._data = SfincsInputVariables()
        return self._data

    def read(self, filename: str) -> None:
        """Read a text file and populate SfincsInput."""
        with open(filename, "r") as fid:
            lines = fid.readlines()

        inp_dict = {}
        for line in lines:
            line = [x.strip() for x in line.split("=")]
            if len(line) != 2:
                continue
            name, val = line
            if name in ["tref", "tstart", "tstop"]:
                try:
                    val = datetime.strptime(val, "%Y%m%d %H%M%S")
                except ValueError:
                    ValueError(f'"{name} = {val}" not understood.')
            elif name in ["cdwnd", "cdval"]:
                val = [float(x) for x in val.split()]
            elif name == "utmzone":
                val = str(val)
            else:
                try:
                    val = literal_eval(val)
                except Exception:
                    pass

            if name == "crs":
                name = "epsg"

            inp_dict[name] = val

        # Convert dictionary to SfincsInput instance
        self._data = SfincsInputVariables(**inp_dict)

    def write(self, filename: str) -> None:
        """Write the instance's attributes to a file."""
        with open(filename, "w") as fid:
            for key, value in self.data.dict(exclude_unset=True).items():
                if value is None:
                    continue
                if isinstance(value, float):  # remove insignificant traling zeros
                    string = f"{key.ljust(20)} = {value}\n"
                elif isinstance(value, int):
                    string = f"{key.ljust(20)} = {value}\n"
                elif isinstance(value, list):
                    valstr = " ".join([str(v) for v in value])
                    string = f"{key.ljust(20)} = {valstr}\n"
                elif hasattr(value, "strftime"):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f"{key.ljust(20)} = {dstr}\n"
                else:
                    string = f"{key.ljust(20)} = {value}\n"
                fid.write(string)

    def get(self, key: str, fallback: Any = None) -> Any:
        """Get a value with validation check."""
        return self.data.model_dump().get(key, fallback)

    def set(self, key: str, value: Any) -> None:
        """Set a value with validation using Pydantic's model_copy."""
        if not hasattr(self.data, key):
            raise KeyError(f"'{key}' is not a valid attribute of SfincsInput.")

        # Validate the new data
        # FIXME implement this in a better way
        try:
            value = SfincsInputVariables(**{key: value}).__dict__[key]
        except Exception as e:
            raise TypeError(f"Invalid input type for '{key}'")

        self._data = self._data.model_copy(update={key: value})

    def update(self, dict: Dict[str, Any]) -> None:
        """
        Update multiple attributes with validation from a dictionary with key-value pairs.

        Parameters:
        -----------
        dict (Dict[str, Any]):
            A dictionary containing key-value pairs to update the attributes.
            For example, dict = {'mmax': 100, 'nmax': 50}.
        """
        # set each key-value pair in the dictionary
        for key, value in dict.items():
            self.set(key, value)
