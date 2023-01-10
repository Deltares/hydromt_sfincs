import datetime
import os
import numpy as np
import xarray as xr
from typing import Union
from pathlib import Path

from .regulargrid import RegularGrid
from .sfincs_input import SfincsInput
from . import utils


class Sfincs:
    _GEOMS = {
        "weirs": "weir",
        "thin_dams": "thd",
    }  # parsed to dict of geopandas.GeoDataFrame
    _FORCING_1D = {
        "waterlevel": ("bzs", "bnd"),  #  timeseries, locations tuple
        "discharge": ("dis", "src"),
        "precip": ("precip", None),
        "waves": ("bzi", "bwv"),  # TODO bhs, btp, bwd, bds (timeseries) bwv (loc)
        "wavemaker": (None, None),  # TODO whi, wti, wst (timeseries) and wvp (loc)
    }
    _FORCING_2D = {
        "precip": "netampr",
    }
    _FORCING_SPW = {"spiderweb": "spw"}
    _MAPS = ["dep", "scs", "manning", "qinf"]

    def __init__(self, root: Union[str, Path] = "") -> None:

        self.root = root
        self.inp = SfincsInput()
        self.grid_type = None
        self.grid = None

        #
        self.forcing = {}
        self.geoms = {}

    def read_input_file(self, fn_inp: Union[str, Path] = "sfincs.inp"):
        # Reads sfincs.inp
        # check if not none and not absolute
        if not os.path.isabs(fn_inp) and self.root:
            fn_inp = os.path.join(self.root, fn_inp)
        else:
            self.root = os.path.dirname(fn_inp)

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
        for name, val in self.inp.__dict__.items():
            setattr(self.inp, name, inp.get(name, None))

        # update grid properties based on sfincs.inp
        grid_type = "quadtree" if self.inp.qtrfile is not None else "regular"
        self.create_grid(grid_type=grid_type)

    def write_input_file(self, fn_inp: Union[str, Path] = "sfincs.inp"):
        fid = open(os.path.join(self.root, fn_inp), "w")
        for key, value in self.inp.__dict__.items():
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

    def create_grid(self, grid_type, **kwargs):
        # initialize grid based kwargs and defaults from inp
        if grid_type == "regular":
            for key in ["x0", "y0", "dx", "dy", "nmax", "mmax", "rotation", "crs"]:
                if key not in kwargs:
                    kwargs.update({key: getattr(self.inp, key)})
            self.grid = RegularGrid(**kwargs)
        elif grid_type == "quadtree":
            pass

    def read_mask(self):
        assert self.grid is not None, "do create_grid() first"
        self.grid.read_mask(
            msk_fn=os.path.join(self.root, self.inp.mskfile),
            ind_fn=os.path.join(self.root, self.inp.indexfile),
        )

    def write_mask(
        self, msk_fn: Union[str, Path] = None, ind_fn: Union[str, Path] = None
    ):
        assert self.grid is not None, "do create_grid() first"

        if msk_fn is None:
            msk_fn = self.inp.mskfile
        setattr(self.inp, f"mskfile", msk_fn)  # update inp

        if ind_fn is None:
            ind_fn = self.inp.indexfile
        setattr(self.inp, f"indexfile", ind_fn)  # update inp

        self.grid.write_mask(
            msk_fn=os.path.join(self.root, msk_fn),
            ind_fn=os.path.join(self.root, ind_fn),
        )

    def read_map(self, name):
        assert self.grid is not None, "do create_grid() first"
        map_fn = getattr(self.inp, f"{name}file")
        if map_fn is None:
            raise ValueError(f"{name}file not defined in sfincs inp file")
        self.grid.read_map(map_fn=os.path.join(self.root, map_fn), name=name)

    def write_map(self, name, map_fn: Union[str, Path] = None):
        assert self.grid is not None, "do create_grid() first"
        if not name in self.grid.data:
            raise ValueError(f"{name}")
        if map_fn is None:
            # if not provided read from inp or fallback to sfincs.<name>
            map_fn = getattr(self.inp, f"{name}file", f"sfincs.{name}")
        setattr(self.inp, f"{name}file", map_fn)  # update inp
        self.grid.write_map(
            map_fn=os.path.join(self.root, map_fn),
            data=self.grid.data[name],
        )

    def read_forcing_1d(self, name):
        ts_name, xy_name = self._FORCING_1D.get(name, (None, None))
        if ts_name:
            ts_fn = getattr(self.inp, f"{ts_name}file")
            if ts_fn is None:
                raise ValueError(f"{ts_name}file not defined in sfincs inp file")
            df = utils.read_timeseries(os.path.join(self.root, ts_fn), self.inp.tref)
            self.forcing.update({ts_name: df})
        if xy_name:
            xy_fn = getattr(self.inp, f"{xy_name}file")
            if xy_fn is None:
                raise ValueError(f"{xy_name}file not defined in sfincs inp file")
            xy = utils.read_xy(os.path.join(self.root, xy_fn), self.grid.crs)
            self.forcing.update({xy_name: xy})

    def write_forcing_1d(
        self, name, ts_fn: Union[str, Path] = None, xy_fn: Union[str, Path] = None
    ):
        ts_name, xy_name = self._FORCING_1D.get(name, (None, None))

        if ts_name:
            if not ts_name in self.forcing:
                raise ValueError(f"{ts_name}")

            if ts_fn is None:
                # if not provided read from inp or fallback to sfincs.<name>
                ts_fn = getattr(self.inp, f"{ts_name}file", f"sfincs.{ts_name}")
            setattr(self.inp, f"{ts_name}file", ts_fn)  # update inp
            utils.write_timeseries(
                fn=os.path.join(self.root, ts_fn),
                df=self.forcing[ts_name],
                tref=self.inp.tref,
            )

        if xy_name:
            if not xy_name in self.forcing:
                raise ValueError(f"{xy_name}")
            if xy_fn is None:
                # if not provided read from inp or fallback to sfincs.<name>
                xy_fn = getattr(self.inp, f"{xy_name}file", f"sfincs.{xy_name}")
            setattr(self.inp, f"{xy_name}file", xy_fn)  # update inp
            utils.write_xy(fn=os.path.join(self.root, xy_fn), gdf=self.forcing[xy_name])

    # TODO: test read and write functions for 2D forcing
    def read_forcing_2d(self, name):
        varname = self._FORCING_2D.get(name, None)

        if varname:
            fn = getattr(self.inp, f"{varname}file")
            if fn is None:
                raise ValueError(f"{fn}file not defined in sfincs inp file")
            da = xr.open_dataarray(fn, chunks={"time": 24})  # lazy
            self.forcing.update({varname: da})

    def write_forcing_2d(self, name: str, fn: Union[str, Path] = None):
        varname = self._FORCING_2D.get(name, None)

        if varname:
            if not varname in self.forcing:
                raise ValueError(f"{varname}")

            if fn is None:
                # if not provided read from inp or fallback to sfincs.<name>
                fn = getattr(self.inp, f"{varname}file", f"sfincs.{varname}")
            setattr(self.inp, f"{varname}file", fn)  # update inp
            self.forcing[varname].to_netcdf(
                os.path.join(self.root, fn), encoding=encoding
            )

    def read_obs(self):
        assert self.inp.obsfile is not None, "obsfile not defined in sfincs inp file"

        fn = self.inp.obsfile
        gdf = utils.read_xy(os.path.join(self.root, fn), crs=self.grid.crs)

        self.geoms.update({"obs": gdf})

    def write_obs(self, fn: Union[str, Path] = None):
        assert self.geoms["obs"] is not None, "create observation points first"

        if fn is None:
            # if not provided read from inp or fallback to sfincs.obs
            fn = getattr(self.inp, f"obsfile", f"sfincs.obs")
        setattr(self.inp, f"{varname}file", fn)  # update inp

        utils.write_xy(fn=os.path.join(self.root, fn), gdf=self.geoms["obs"])

    def read_structure(self, name):
        struct_name = self._GEOMS.get(name, None)

        if struct_name:
            struct_fn = getattr(self.inp, f"{struct_name}file")
            if struct_fn is None:
                raise ValueError(f"{ts_name}file not defined in sfincs inp file")
            struct = utils.read_structures(ps.path.join(self.root, struct_fn))
            gdf = utils.structures2gdf(struct, crs=self.grid.crs)
            self.geoms.update({struct_name: gdf})

    def write_structure(self, name, struct_fn: Union[str, Path] = None):
        struct_name = self._GEOMS.get(name, None)

        if struct_name:
            if not struct_name in self.geoms:
                raise ValueError(f"{struct_name}")

            if struct_fn is None:
                struct_fn = getattr(
                    self.inp, f"{struct_name}file", f"sfincs.{struct_name}"
                )
            setattr(self.inp, f"{struct_name}file", struct_name)  # update inp
            struct = utils.gdf2structures(self.geoms[struct_name])
            utils.write_structures(
                fn=os.path.join(self.root, struct_fn), feats=struct, stype=struct_name
            )

    def read_his_results():
        pass

    def read_map_results():
        pass
