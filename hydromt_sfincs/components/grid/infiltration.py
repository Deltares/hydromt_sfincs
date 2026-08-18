import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import DATADIR, workflows
from hydromt_sfincs.infiltration import (
    BUCKET_VARS,
    DEFAULT_BUCKETFILE,
    VARIABLES,
    clear_data,
    configure,
    configured_flavor,
    flavor_variables,
    get_attrs,
    regular_active_vector,
    regular_vector_to_da,
    reset_config,
    sidecar_dataset,
)

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsInfiltration(ModelComponent):
    """SFINCS infiltration component for regular grids."""

    def __init__(self, model: "SfincsModel"):
        super().__init__(model=model)

    @property
    def data(self):
        return self.model.grid.data

    @property
    def mask(self):
        return self.model.grid.mask

    def _read_rasterdataset(self, source):
        if isinstance(source, (xr.DataArray, xr.Dataset)):
            return source
        return self.data_catalog.get_rasterdataset(
            source,
            bbox=self.model.bbox,
            buffer=10,
        )

    def _read_dataframe(self, source):
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return self.data_catalog.get_dataframe(
            source,
            source_kwargs={
                "driver": {"name": "pandas", "options": {"index_col": 0}}
            },
        )

    @staticmethod
    def _as_dataarray(data, variable=None):
        if isinstance(data, xr.DataArray):
            return data
        if variable and variable in data.data_vars:
            return data[variable]
        if len(data.data_vars) == 1:
            return next(iter(data.data_vars.values()))
        raise ValueError(f"Could not determine raster variable '{variable}'.")

    def _sample(self, source, *, variable=None, method="average"):
        da = self._as_dataarray(self._read_rasterdataset(source), variable=variable)
        try:
            da = da.raster.mask_nodata()
        except Exception:
            pass
        return da.raster.reproject_like(self.mask, method=method)

    def _sample_from_dataset(self, ds: xr.Dataset, name: str, method="average"):
        da = ds[name]
        try:
            da = da.raster.mask_nodata()
        except Exception:
            pass
        return da.raster.reproject_like(self.mask, method=method)

    def _prepare(self, da: xr.DataArray, name: str):
        da = da.astype(np.float32)
        if np.logical_and(np.isnan(da), self.mask > 0).any():
            logger.warning("NaN values found in %s data; filled with 0", name)
            da = da.fillna(0.0)
        fill_value = VARIABLES[name].fill_value
        da = da.where(self.mask > 0, fill_value)
        da.name = name
        da.attrs.update(get_attrs(name))
        try:
            da.raster.set_crs(self.model.crs)
            da.raster.set_nodata(fill_value)
        except Exception:
            pass
        return da

    def _drop_existing_layers(self):
        self.model.grid._data = clear_data(self.data, keep=())

    def _set_layers(self, layers: dict[str, xr.DataArray], flavor: str):
        self._drop_existing_layers()
        for name, da in layers.items():
            self.model.grid.set(self._prepare(da, name))
        configure(self.model.config, flavor=flavor, grid_type="regular")

    def _table_path(self, name: str) -> Path:
        return Path(DATADIR) / "infiltration" / name

    def _default_lookup(self, kind: str) -> Path:
        mapping = {
            "green_ampt": "hsg_green_ampt.csv",
            "horton": "hsg_horton.csv",
            "bucket": "hsg_bucket.csv",
            "lulc_modifiers": "nlcd_infiltration_modifiers.csv",
        }
        return self._table_path(mapping[kind])

    def _constant_layer(self, value: float, name: str):
        da = xr.full_like(self.mask, np.float32(value), dtype=np.float32)
        da.name = name
        return da

    def _curve_number_from_landuse_hsg(self, lulc, hsg, reclass_table):
        da_landuse = self._as_dataarray(self._read_rasterdataset(lulc))
        da_hsg = self._as_dataarray(self._read_rasterdataset(hsg))
        da_hsg = da_hsg.raster.reproject_like(da_landuse, method="nearest")
        df_map = self._read_dataframe(reclass_table)
        return workflows.curve_number_from_landuse_hsg(da_landuse, da_hsg, df_map)

    def _read_sidecar(self, filename: Path, variables):
        if not filename.exists():
            raise FileNotFoundError(filename)
        with xr.open_dataset(filename) as ds:
            layers = {}
            for name in variables:
                if name not in ds:
                    raise ValueError(f"Missing variable '{name}' in {filename}.")
                layers[name] = regular_vector_to_da(
                    ds[name].values,
                    self.mask,
                    self.mask.rename(name),
                    fill_value=VARIABLES[name].fill_value,
                )
        return layers

    def read(self):
        """Read infiltration data not handled by the grid component."""
        flavor = configured_flavor(self.model.config)
        if flavor is None or flavor == "con":
            return
        if flavor == "bkt":
            bucketfile = self.model.config.get("bucketfile", abs_path=True)
            if bucketfile is None:
                bucketfile = self.model.config.get("infiltration_file", abs_path=True)
            if bucketfile is None:
                return
            layers = self._read_sidecar(bucketfile, BUCKET_VARS)
            self._set_layers(layers, flavor="bkt")
            if self.model.config.get("bucketfile") is not None:
                self.model.config.set("bucketfile", Path(bucketfile).name)
            elif self.model.config.get("infiltration_file") is not None:
                self.model.config.set("infiltration_file", Path(bucketfile).name)
            return
        inffile = self.model.config.get("infiltration_file", abs_path=True)
        if inffile is None:
            return
        layers = self._read_sidecar(inffile, flavor_variables(flavor))
        self._set_layers(layers, flavor=flavor)
        self.model.config.set("infiltration_file", Path(inffile).name)

    def write(self):
        """Write regular-grid infiltration sidecars not handled by the grid."""
        if not all(name in self.data for name in BUCKET_VARS):
            return
        bucketfile = self.model.config.get_set_file_variable(
            "bucketfile", default=DEFAULT_BUCKETFILE
        )
        bucketfile.parent.mkdir(parents=True, exist_ok=True)
        ds = sidecar_dataset(
            {
                name: regular_active_vector(self.data[name], self.mask)
                for name in BUCKET_VARS
            },
            int((self.mask > 0).sum()),
        )
        ds.to_netcdf(bucketfile)

    @hydromt_step
    def create_uniform_constant(self, qinf: float):
        """Create a uniform constant infiltration rate in model config."""
        self._drop_existing_layers()
        reset_config(self.model.config)
        self.model.config.set("qinf", float(qinf))

    @hydromt_step
    def create_cn_from_landuse_hsg(
        self,
        lulc,
        hsg,
        reclass_table,
        antecedent_moisture="avg",
        reproj_method="med",
    ):
        """Create Curve Number infiltration from land use and hydrologic soil groups."""
        da_cn = self._curve_number_from_landuse_hsg(lulc, hsg, reclass_table)
        da_cn = workflows.adjust_curve_number(
            da_cn,
            antecedent_moisture=antecedent_moisture,
        )
        self.create_cn(
            da_cn,
            antecedent_moisture=None,
            reproj_method=reproj_method,
        )

    @hydromt_step
    def create_constant(
        self,
        qinf=None,
        ksat=None,
        lulc=None,
        reclass_table=None,
        lulc_modifiers=None,
        factor_ksat=3.6,
        reproj_method="average",
    ):
        """Create a spatially varying constant infiltration map."""
        if qinf is not None:
            da_qinf = self._sample(qinf, method=reproj_method)
        elif ksat is not None and lulc is not None:
            da_ksat = self._sample(ksat, method="average")
            da_lulc = self._sample(lulc, method="nearest")
            if lulc_modifiers is None:
                lulc_modifiers = self._default_lookup("lulc_modifiers")
            df_modifiers = self._read_dataframe(lulc_modifiers)
            da_qinf = workflows.constant_infiltration_from_ksat_lulc(
                da_ksat,
                da_lulc,
                df_modifiers,
                da_mask=self.mask,
                factor_ksat=factor_ksat,
            )
        elif lulc is not None:
            if reclass_table is None:
                raise IOError(f"Infiltration mapping file should be provided for {lulc}")
            da_lulc = self._as_dataarray(self._read_rasterdataset(lulc))
            df_map = self._read_dataframe(reclass_table)
            da_qinf = da_lulc.raster.reclassify(df_map[["qinf"]])["qinf"]
            da_qinf = da_qinf.raster.reproject_like(self.mask, method=reproj_method)
        else:
            raise ValueError(
                "Provide qinf, ksat+lulc, or lulc+reclass_table when setting up constant infiltration."
            )

        self._set_layers({"qinf": da_qinf}, flavor="c2d")

    @hydromt_step
    def create_cn(self, cn, antecedent_moisture="avg", reproj_method="med"):
        """Create Curve Number infiltration without recovery."""
        da_org = self._read_rasterdataset(cn)
        var_name = "cn" if antecedent_moisture is None else f"cn_{antecedent_moisture}"
        da_cn = self._sample(da_org, variable=var_name, method=reproj_method)
        da_scs = workflows.cn_to_s(da_cn, self.mask > 0).round(3)

        self._set_layers({"scs": da_scs}, flavor="cna")

    @hydromt_step
    def create_cn_with_recovery(
        self,
        lulc,
        hsg,
        ksat,
        reclass_table,
        effective,
        factor_ksat=3.6,
        block_size=2000,
    ):
        """Create model the Soil Conservation Service (SCS) Curve Number (CN) files for SFINCS
        including recovery term based on the soil saturation

        Parameters
        ---------
        lulc : str, Path, or RasterDataset
            Landuse/landcover data set
        hsg : str, Path, or RasterDataset
            HSG (Hydrological Similarity Group) in integers
        ksat : str, Path, or RasterDataset
            Ksat (saturated hydraulic conductivity) [mm/hr]
        reclass_table : str, Path, or RasterDataset
            reclass table to relate landcover with soiltype
        effective : float
            estimate of percentage effective soil, e.g. 0.50 for 50%
        factor_ksat : float
            factor to convert units of Ksat, e.g. from micrometer per second to mm/hr
        block_size : float
            maximum block size - use larger values will get more data in memory but can be faster, default=2000
        """

        del block_size  # kept for backwards compatibility
        da_landuse = self._as_dataarray(self._read_rasterdataset(lulc))
        da_hsg = self._as_dataarray(self._read_rasterdataset(hsg))
        da_hsg = da_hsg.raster.reproject_like(da_landuse, method="nearest")
        da_ksat = self._as_dataarray(self._read_rasterdataset(ksat))
        da_ksat = da_ksat.raster.reproject_like(da_landuse, method="average")
        df_map = self._read_dataframe(reclass_table)
        ds = workflows.curve_number_with_recovery(
            da_landuse,
            da_hsg,
            da_ksat,
            df_map,
            effective=effective,
            factor_ksat=factor_ksat,
        )
        layers = {
            name: self._sample_from_dataset(ds, name, method="average")
            for name in ("smax", "seff", "ks")
        }
        self._set_layers(layers, flavor="cnb")

    @hydromt_step
    def create_green_ampt(
        self,
        psi=None,
        sigma=None,
        ks=None,
        *,
        soil=None,
        hsg=None,
        ksat=None,
        lulc=None,
        reclass_table=None,
        lulc_modifiers=None,
        dual_hsg="drained",
        factor_ksat=3.6,
        reproj_method="average",
    ):
        """Create Green-Ampt infiltration parameters."""
        if all(value is not None for value in (psi, sigma, ks)):
            layers = {
                "psi": self._sample(psi, method=reproj_method),
                "sigma": self._sample(sigma, method=reproj_method),
                "ks": self._sample(ks, method=reproj_method),
            }
        else:
            soil_source = hsg if hsg is not None else soil
            if soil_source is None:
                raise ValueError(
                    "Provide psi, sigma and ks rasters, or provide soil/hsg data for estimation."
                )
            if reclass_table is None and hsg is not None:
                reclass_table = self._default_lookup("green_ampt")
            if reclass_table is None:
                raise ValueError("A reclass_table is required for Green-Ampt estimation.")
            da_soil = self._as_dataarray(self._read_rasterdataset(soil_source))
            df_map = self._read_dataframe(reclass_table)
            da_ksat = None
            if ksat is not None:
                da_ksat = self._as_dataarray(self._read_rasterdataset(ksat))
                da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")
            if lulc is not None:
                if lulc_modifiers is None:
                    lulc_modifiers = self._default_lookup("lulc_modifiers")
                da_lulc = self._as_dataarray(self._read_rasterdataset(lulc))
                da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
                df_modifiers = self._read_dataframe(lulc_modifiers)
                ds = workflows.green_ampt_from_soil_landuse(
                    da_soil,
                    da_lulc,
                    df_map,
                    df_modifiers,
                    da_ksat=da_ksat,
                    factor_ksat=factor_ksat,
                    dual_hsg=dual_hsg,
                )
            else:
                ds = workflows.green_ampt_from_soil(
                    da_soil,
                    df_map,
                    da_ksat=da_ksat,
                    factor_ksat=factor_ksat,
                )
            layers = {
                name: self._sample_from_dataset(ds, name, method=reproj_method)
                for name in ("psi", "sigma", "ks")
            }
        self._set_layers(layers, flavor="gai")

    @hydromt_step
    def create_horton(
        self,
        f0=None,
        fc=None,
        kd=None,
        *,
        soil=None,
        hsg=None,
        ksat=None,
        lulc=None,
        reclass_table=None,
        lulc_modifiers=None,
        dual_hsg="drained",
        factor_ksat=3.6,
        reproj_method="average",
    ):
        """Create modified Horton infiltration parameters."""
        if all(value is not None for value in (f0, fc, kd)):
            layers = {
                "f0": self._sample(f0, method=reproj_method),
                "fc": self._sample(fc, method=reproj_method),
                "kd": self._sample(kd, method=reproj_method),
            }
        else:
            soil_source = hsg if hsg is not None else soil
            if soil_source is None:
                raise ValueError(
                    "Provide f0, fc and kd rasters, or provide soil/hsg data for estimation."
                )
            if reclass_table is None and hsg is not None:
                reclass_table = self._default_lookup("horton")
            if reclass_table is None:
                raise ValueError("A reclass_table is required for Horton estimation.")
            da_soil = self._as_dataarray(self._read_rasterdataset(soil_source))
            df_map = self._read_dataframe(reclass_table)
            da_ksat = None
            if ksat is not None:
                da_ksat = self._as_dataarray(self._read_rasterdataset(ksat))
                da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")
            if lulc is not None:
                if lulc_modifiers is None:
                    lulc_modifiers = self._default_lookup("lulc_modifiers")
                da_lulc = self._as_dataarray(self._read_rasterdataset(lulc))
                da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
                df_modifiers = self._read_dataframe(lulc_modifiers)
                ds = workflows.horton_from_soil_landuse(
                    da_soil,
                    da_lulc,
                    df_map,
                    df_modifiers,
                    da_ksat=da_ksat,
                    factor_ksat=factor_ksat,
                    dual_hsg=dual_hsg,
                )
            else:
                ds = workflows.horton_from_soil(
                    da_soil,
                    df_map,
                    da_ksat=da_ksat,
                    factor_ksat=factor_ksat,
                )
            layers = {
                name: self._sample_from_dataset(ds, name, method=reproj_method)
                for name in ("f0", "fc", "kd")
            }
        self._set_layers(layers, flavor="hor")

    @hydromt_step
    def create_bucket(
        self,
        bucket_smax=None,
        bucket_k=None,
        bucket_loss=None,
        *,
        soil=None,
        hsg=None,
        ksat=None,
        lulc=None,
        reclass_table=None,
        lulc_modifiers=None,
        dual_hsg="drained",
        factor_ksat=3.6,
        reproj_method="average",
    ):
        """Create bucket-model infiltration parameters."""
        if bucket_smax is not None and bucket_k is not None:
            loss_value = 0.0 if bucket_loss is None else bucket_loss
            if bucket_loss is None or np.isscalar(bucket_loss):
                da_loss = self._constant_layer(loss_value, "bucket_loss")
            else:
                da_loss = self._sample(bucket_loss, method=reproj_method)
            layers = {
                "bucket_smax": self._sample(bucket_smax, method=reproj_method),
                "bucket_k": self._sample(bucket_k, method=reproj_method),
                "bucket_loss": da_loss,
            }
        else:
            soil_source = hsg if hsg is not None else soil
            if soil_source is None:
                raise ValueError(
                    "Provide bucket_smax and bucket_k rasters, or provide soil/hsg data for estimation."
                )
            if reclass_table is None and hsg is not None:
                reclass_table = self._default_lookup("bucket")
            if reclass_table is None:
                raise ValueError("A reclass_table is required for bucket estimation.")
            da_soil = self._as_dataarray(self._read_rasterdataset(soil_source))
            df_map = self._read_dataframe(reclass_table)
            da_ksat = None
            if ksat is not None:
                da_ksat = self._as_dataarray(self._read_rasterdataset(ksat))
                da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")
            if lulc is not None:
                if lulc_modifiers is None:
                    lulc_modifiers = self._default_lookup("lulc_modifiers")
                da_lulc = self._as_dataarray(self._read_rasterdataset(lulc))
                da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
                df_modifiers = self._read_dataframe(lulc_modifiers)
                ds = workflows.bucket_from_soil_landuse(
                    da_soil,
                    da_lulc,
                    df_map,
                    df_modifiers,
                    da_ksat=da_ksat,
                    factor_ksat=factor_ksat,
                    dual_hsg=dual_hsg,
                    bucket_loss=0.10 if bucket_loss is None else bucket_loss,
                )
            else:
                ds = workflows.bucket_from_soil(
                    da_soil,
                    df_map,
                    da_ksat=da_ksat,
                    factor_ksat=factor_ksat,
                    bucket_loss=bucket_loss if np.isscalar(bucket_loss) else None,
                )
            layers = {
                name: self._sample_from_dataset(ds, name, method=reproj_method)
                for name in BUCKET_VARS
            }
        self._set_layers(layers, flavor="bkt")
