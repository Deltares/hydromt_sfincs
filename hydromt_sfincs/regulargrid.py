import numpy as np
import math
import xarray as xr
from affine import Affine
from typing import Union
from pathlib import Path
from pyproj import CRS
from .sfincs_input import SfincsInput


class RegularGrid:
    def __init__(self, x0, y0, dx, dy, nmax, mmax, rotation, crs=None):

        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax  # height
        self.mmax = mmax  # width
        self.rotation = rotation
        self.shape = (nmax, mmax)
        self.crs = None
        if crs is not None:
            self.crs = CRS.from_user_input(crs)
        self.data = xr.Dataset()

        # cosrot = math.cos(rotation * math.pi / 180)
        # sinrot = math.sin(rotation * math.pi / 180)

        # xx = np.linspace(
        #     0.5 * self.dx, self.mmax * self.dx - 0.5 * self.dx, num=self.mmax
        # )
        # yy = np.linspace(
        #     0.5 * self.dy, self.nmax * self.dy - 0.5 * self.dy, num=self.nmax
        # )

        # xg0, yg0 = np.meshgrid(xx, yy)
        # xg = self.x0 + xg0 * cosrot - yg0 * sinrot
        # yg = self.y0 + xg0 * sinrot + yg0 * cosrot
        # self.xz = xg
        # self.yz = yg

    @property
    def mask(self):
        if "mask" in self.data:
            return self.data["mask"]

    @staticmethod
    def from_inp(inp: SfincsInput) -> None:
        return RegularGrid(
            inp.x0, inp.y0, inp.dx, inp.dy, inp.nmax, inp.mmax, inp.rotation
        )

    @property
    def affine(self):
        return Affine(self.dx, 0, self.x0, 0, self.dy, self.y0) * Affine.rotation(
            self.rotation
        )

    @property
    def coordinates(self):
        # TODO fix for ratated grids
        transform = self.affine * Affine.translation(0.5, 0.5)
        if self.affine.is_rectilinear:
            x_coords, _ = transform * (np.arange(self.mmax), np.zeros(self.mmax))
            _, y_coords = transform * (np.zeros(self.nmax), np.arange(self.nmax))
        else:
            x_coords, y_coords = transform * np.meshgrid(
                np.arange(self.mmax),
                np.arange(self.nmax),
            )
        return {"y": y_coords, "x": x_coords}

    @property
    def ind(self) -> np.ndarray:
        # assert ind.max() <= np.multiply(*shape)
        iok = np.where(np.transpose(self.mask.values) > 0)
        iok = (iok[1], iok[0])
        ind = np.ravel_multi_index(iok, (self.nmax, self.mmax), order="F")
        return ind

    def write_mask(
        self,
        msk_fn: Union[str, Path] = "sfincs.msk",
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> None:

        # Add 1 because indices in SFINCS start with 1, not 0
        indices_ = np.array(
            np.hstack([np.array(len(self.ind)), self.ind + 1]), dtype="u4"
        )
        indices_.tofile(ind_fn)

        self.write_map(map_fn=msk_fn, data=self.mask.values, dtype="u1")

    def read_mask(
        self,
        msk_fn: Union[str, Path] = "sfincs.msk",
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> xr.DataArray:
        _ind = np.fromfile(ind_fn, dtype="u4")
        ind = _ind[1:] - 1  # convert to zero based index
        assert _ind[0] == ind.size

        # mask = utils.read_binary_map()
        nrow, ncol = self.shape
        mask = np.full((ncol, nrow), 0, dtype="u1")
        mask.flat[ind] = np.fromfile(msk_fn, dtype=mask.dtype)
        mask = mask.transpose()

        da_mask = xr.DataArray(
            name="mask",
            data=mask,
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": 0},
        )
        if len(self.data.data_vars) == 0:
            # overwrite data property if empty
            self.data = da_mask.to_dataset()
        else:
            self.data.update(da_mask.to_dataset())

        return da_mask

    def write_map(
        self,
        map_fn: Union[str, Path],
        data: Union[xr.DataArray, np.ndarray],
        dtype: Union[str, np.dtype] = "f4",
    ) -> None:

        if isinstance(data, xr.DataArray):
            data = data.values

        data_out = np.asarray(
            data.transpose()[self.mask.values.transpose() > 0], dtype=dtype
        )
        data_out.tofile(map_fn)

    def read_map(
        self,
        map_fn: Union[str, Path],
        dtype: Union[str, np.dtype] = "f4",
        mv: float = -9999.0,
        name: str = None,
    ) -> xr.DataArray:
        nrow, ncol = self.shape
        data = np.full((ncol, nrow), mv, dtype=dtype)
        data.flat[self.ind] = np.fromfile(map_fn, dtype=dtype)
        data = data.transpose()

        da = xr.DataArray(
            name=map_fn.split(".")[-1] if name is None else name,
            data=data,
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": mv},
        )
        self.data.update(da.to_dataset())
        # da.raster.write_crs(self.crs)
        # TODO da = ...
        return da
