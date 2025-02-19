from datetime import datetime
from typing import List, Optional, Dict, Any
from ast import literal_eval
from pydantic import Field
from pydantic_settings import BaseSettings

from hydromt.model import Model
from hydromt.model.components import ModelComponent


class SfincsInputVariables(BaseSettings):
    # Attributes with descriptions and defaults
    mmax: int = Field(
        10, description="Maximum number of grid points in the x-direction"
    )
    nmax: int = Field(
        10, description="Maximum number of grid points in the y-direction"
    )
    dx: float = Field(10.0, gt=0, description="Grid spacing in the x-direction")
    dy: float = Field(10.0, gt=0, description="Grid spacing in the y-direction")
    x0: float = Field(0.0, description="Origin of the grid in the x-direction")
    y0: float = Field(0.0, description="Origin of the grid in the y-direction")
    rotation: float = Field(0.0, description="Rotation angle of the grid (degrees)")
    epsg: Optional[int] = Field(
        None, description="EPSG code for spatial reference system"
    )
    latitude: float = Field(0.0, description="Latitude of the grid center")
    utmzone: Optional[str] = Field(None, description="UTM zone for spatial reference")
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
    tspinup: float = Field(0.0, ge=0, description="Spin-up time (seconds)")
    t0out: Optional[datetime] = Field(None, description="Output start time (datetime)")
    dtout: float = Field(3600.0, description="Output interval (seconds)")
    dtmapout: Optional[float] = Field(None, description="Map output interval (seconds)")
    dthisout: float = Field(600.0, description="Timeseries output interval (seconds)")
    dtrstout: float = Field(0.0, description="Restart file output interval (seconds)")
    dtmaxout: int = Field(86400, description="Maximum map output interval (seconds)")
    trstout: float = Field(-999.0, description="Restart file time (seconds)")
    dtwnd: float = Field(1800.0, description="Wind forcing interval (seconds)")
    alpha: float = Field(0.5, description="Numerical diffusion parameter")
    theta: float = Field(1.0, description="Numerical weighting parameter")
    huthresh: float = Field(0.01, description="Threshold water depth (meters)")
    manning: float = Field(
        0.04, description="Manning's n coefficient for overall roughness"
    )
    manning_land: float = Field(0.04, description="Manning's n for land areas")
    manning_sea: float = Field(0.02, description="Manning's n for sea areas")
    rgh_lev_land: float = Field(0.0, description="Roughness level for land")
    zsini: float = Field(0.0, description="Initial surface elevation (meters)")
    qinf: float = Field(0.0, description="Inflow discharge (m^3/s)")
    rhoa: float = Field(1.25, description="Air density (kg/m^3)")
    rhow: float = Field(1024.0, description="Water density (kg/m^3)")
    dtmax: float = Field(60.0, description="Maximum allowed timestep (seconds)")
    advection: int = Field(1, description="Enable advection (1: yes, 0: no)")
    baro: int = Field(1, description="Enable baroclinicity (1: yes, 0: no)")
    pavbnd: int = Field(
        0, description="Atmospheric pressure boundary condition (1: yes, 0: no)"
    )
    gapres: float = Field(101200.0, description="Atmospheric pressure (Pa)")
    stopdepth: float = Field(100.0, description="Stopping water depth (meters)")
    crsgeo: int = Field(
        0, description="Geographical coordinate system flag (1: yes, 0: no)"
    )
    btfilter: float = Field(60.0, description="Bottom filter interval (seconds)")
    viscosity: int = Field(1, description="Numerical viscosity parameter")

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

    inputformat: str = Field("bin", description="Input file format (bin or text)")
    outputformat: str = Field("net", description="Output file format (net or text)")

    cdnrb: int = Field(
        3, description="Number of wind speed ranges for drag coefficient"
    )
    cdwnd: List[float] = Field(
        [0.0, 28.0, 50.0],
        description="Wind speed ranges for drag coefficient (m/s)",
    )
    cdval: List[float] = Field(
        [0.001, 0.0025, 0.0015],
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
