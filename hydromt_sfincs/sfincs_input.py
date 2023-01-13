import datetime


class SfincsInput:
    def __init__(self):
        self.mmax = 0
        self.nmax = 0
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
        #        self.bhifile=None
        #        self.bstfile=None
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

    def read_input_file(self, fn_inp) -> None:

        with open(fn_inp, "r") as fid:
            lines = fid.readlines()
        inp = dict()

        for line in lines:
            str = line.split("=")
            if len(str) == 1:
                # Empty line
                continue
            name = str[0].strip()
            val = str[1].strip()
            try:
                # First try to convert to int
                val = int(val)
            except ValueError:
                try:
                    # Now try to convert to float
                    val = float(val)
                except:
                    pass
            if name == "tref":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None
            if name == "tstart":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None
            if name == "tstop":
                try:
                    val = datetime.datetime.strptime(val.rstrip(), "%Y%m%d %H%M%S")
                except:
                    val = None
            if name == "epsg":
                name = "crs"
            inp[name] = val

        # set default values to None if not found in sfincs.inp
        for name, val in self.__dict__.items():
            setattr(self, name, inp.get(name, None))

    def write_input_file(self, fn_inp) -> None:
        fid = open(fn_inp, "w")
        for key, value in self.__dict__.items():
            if not value is None:
                if type(value) == "float":
                    string = f"{key.ljust(20)} = {float(value)}\n"
                elif type(value) == "int":
                    string = f"{key.ljust(20)} = {int(value)}\n"
                elif type(value) == list:
                    valstr = ""
                    for v in value:
                        valstr += str(v) + " "
                    string = f"{key.ljust(20)} = {valstr}\n"
                elif isinstance(value, datetime.date):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f"{key.ljust(20)} = {dstr}\n"
                else:
                    string = f"{key.ljust(20)} = {value}\n"
                fid.write(string)
        fid.close()

    @staticmethod
    def from_dict(inp_dict) -> None:
        # (over)write sfincs.inp values based on values from inp_dict
        inp = SfincsInput()
        for name, val in inp_dict.items():
            setattr(inp, name, val)
        return inp

    def to_dict(self):
        return self.__dict__
