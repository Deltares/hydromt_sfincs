"""
SfincsInput class to read and write sfincs input (inp) files.
"""

from ast import literal_eval
from datetime import datetime
from typing import Dict, Any


class SfincsInput:
    def __init__(self):
        """Initialize SfincsInput class with default values"""
        self.mmax = 10
        self.nmax = 10
        self.dx = 10.0
        self.dy = 10.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.rotation = 0.0
        self.epsg = None
        self.latitude = 0.0
        self.utmzone = None
        self.tref = datetime(2010, 2, 1, 0, 0, 0)
        self.tstart = datetime(2010, 2, 1, 0, 0, 0)
        self.tstop = datetime(2010, 2, 2, 0, 0, 0)
        self.tspinup = 60.0
        self.t0out = None
        self.dtout = 3600.0
        self.dtmapout = None
        self.dthisout = 600.0
        self.dtrstout = 0.0
        self.dtmaxout = 86400
        self.trstout = -999.0
        self.dtwnd = 1800.0
        self.alpha = 0.5
        self.theta = 1.0
        self.huthresh = 0.01
        self.manning = 0.04
        self.manning_land = 0.04
        self.manning_sea = 0.02
        self.rgh_lev_land = 0.0
        self.zsini = 0.0
        self.qinf = 0.0
        self.rhoa = 1.25
        self.rhow = 1024.0
        self.dtmax = 60.0
        self.advection = 1
        self.baro = 1
        self.pavbnd = 0
        self.gapres = 101200.0
        self.stopdepth = 100.0
        self.crsgeo = 0
        self.btfilter = 60.0
        self.viscosity = 1

        self.depfile = None
        self.mskfile = None
        self.indexfile = None
        self.cstfile = None
        self.bndfile = None
        self.bzsfile = None
        self.bzifile = None
        self.bwvfile = None
        self.bhsfile = None
        # self.bhifile=None
        # self.bstfile=None
        self.btpfile = None
        self.bwdfile = None
        self.bdsfile = None
        self.bcafile = None
        self.corfile = None
        self.srcfile = None
        self.disfile = None
        self.inifile = None
        self.sbgfile = None
        self.qtrfile = None
        self.spwfile = None
        self.amufile = None
        self.amvfile = None
        self.ampfile = None
        self.amprfile = None
        self.wndfile = None
        self.precipfile = None
        self.obsfile = None
        self.crsfile = None
        self.thdfile = None
        self.manningfile = None
        self.scsfile = None
        self.rstfile = None
        self.wfpfile = None
        self.whifile = None
        self.wtifile = None
        self.wstfile = None

        self.inputformat = "bin"
        self.outputformat = "net"

        self.cdnrb = 3
        self.cdwnd = [0.0, 28.0, 50.0]
        self.cdval = [0.001, 0.0025, 0.0015]

    def read(self, inp_fn: str) -> None:
        """Read sfincs input file and set attributes to values in file."""
        with open(inp_fn, "r") as fid:
            lines = fid.readlines()

        inp_dict = dict()
        for line in lines:
            line = [x.strip() for x in line.split("=")]
            if len(line) != 2:  #  Empty or unrecognized line
                continue
            name, val = line
            if name in ["tref", "tstart", "tstop"]:
                try:
                    val = datetime.strptime(val, "%Y%m%d %H%M%S")
                except ValueError:
                    ValueError(f'"{name} = {val}" not understood.')
            elif name in ["cdwnd", "cdval"]:
                vals = []
                [vals.append(float(val)) for val in val.split()]
                val = vals
            elif name == "utmzone":
                val = str(val)
            else:
                try:
                    val = literal_eval(val)
                except Exception:  # normal string
                    pass
            if name == "crs":
                name = "epsg"
            inp_dict[name] = val
            setattr(self, name, val)

        # set default values to None if not found in inp_dict to avoid writing these later
        for name in self.__dict__:
            if name not in inp_dict:
                setattr(self, name, None)

    def write(self, inp_fn: str) -> None:
        """Write sfincs input file from attributes."""
        fid = open(inp_fn, "w")
        for key, value in self.__dict__.items():
            if not value is None:
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
        fid.close()

    @staticmethod
    def from_dict(inp_dict: Dict) -> None:
        """Create SfincsInput object from dictionary."""
        inp = SfincsInput()
        for name, val in inp_dict.items():
            setattr(inp, name, val)
        # set default values to None if not found in inp_dict
        for name in inp.__dict__:
            if name not in inp_dict:
                setattr(inp, name, None)
        return inp

    @staticmethod
    def from_file(inp_fn: str) -> None:
        """Create SfincsInput object from input file."""
        inp = SfincsInput()
        inp.read(inp_fn)
        return inp

    def to_dict(self) -> Dict:
        """Return dictionary of attributes."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __getitem__(self, name: str) -> Any:
        """Return attribute value."""
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        """Set attribute value."""
        setattr(self, name, value)

    def __repr__(self) -> str:
        """Return string representation of object."""
        return f"{self.__class__.__name__}({self.to_dict()})"

    def __eq__(self, __value: object) -> bool:
        """Return True if objects are equal."""
        if isinstance(__value, self.__class__):
            return self.__dict__ == __value.__dict__
        return False
