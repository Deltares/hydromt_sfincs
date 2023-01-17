from datetime import datetime
from ast import literal_eval
from typing import Dict


class SfincsInput:
    def __init__(self):
        self.mmax = 10
        self.nmax = 10
        self.dx = 10.0
        self.dy = 10.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.rotation = 0.0
        self.crs = None
        self.latitude = 0.0
        self.tref = None
        self.tstart = None
        self.tstop = None
        self.tspinup = 60.0
        self.t0out = None
        self.dtout = None
        self.dtmapout = 600.0
        self.dthisout = 600.0
        self.dtrstout = 0.0
        self.dtmaxout = 0.0
        self.trstout = -999.0
        self.dtwnd = 1800.0
        self.alpha = 0.75
        self.theta = 0.90
        self.huthresh = 0.01
        self.manning = 0.04
        self.manning_land = 0.04
        self.manning_sea = 0.02
        self.rgh_lev_land = 0.0
        self.zsini = 0.0
        self.qinf = 0.0
        self.igperiod = 120.0
        self.rhoa = 1.25
        self.rhow = 1024.0
        self.dtmax = 999.0
        self.maxlev = 999.0
        self.bndtype = 1
        self.advection = 0
        self.baro = 0
        self.pavbnd = 0
        self.gapres = 101200.0
        self.advlim = 9999.9
        self.stopdepth = 1000.0
        self.crsgeo = 0

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
            else:
                try:
                    val = literal_eval(val)
                except ValueError:  # normal string
                    pass
            if name == "epsg":
                name = "crs"
            inp_dict[name] = val
            setattr(self, name, val)

        # set default values to None if not found in inp_dict to avoid writing these later
        for name in self.__dict__:
            if name not in inp_dict:
                setattr(self, name, None)

    def write(self, inp_fn: str) -> None:
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
        inp = SfincsInput()
        for name, val in inp_dict.items():
            setattr(inp, name, val)
        # set default values to None if not found in inp_dict
        for name in inp.__dict__:
            if name not in inp_dict:
                setattr(inp, name, None)
        return inp

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}
