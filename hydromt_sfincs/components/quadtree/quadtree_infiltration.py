import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import DATADIR, workflows
from hydromt_sfincs.components.quadtree import SfincsQuadtreeMixin
from hydromt_sfincs.workflows.infiltration import (
    BUCKET_VARS,
    DEFAULT_BUCKETFILE,
    DEFAULT_INFILTRATIONFILE,
    VARIABLES,
    clear_data,
    configure,
    configured_flavor,
    flavor_variables,
    get_attrs,
    reset_config,
    sidecar_dataset,
)

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsQuadtreeInfiltration(SfincsQuadtreeMixin, ModelComponent):
    """SFINCS infiltration component for quadtree grids.

    Unsuffixed ``create_*`` methods use final SFINCS parameter maps directly.
    Methods with a ``_from_soil`` suffix estimate those parameters from HSG,
    optional Ksat, and optional land-use modifiers.
    """

    def __init__(self, model: "SfincsModel"):
        super().__init__(model=model)

    @property
    def data(self):
        return self.model.quadtree_grid.data

    @property
    def mask(self):
        return self.model.quadtree_grid.mask

    def _set_layers(self, layers, flavor: str):
        self.clear()
        for name, layer in layers.items():
            values = (
                layer.values
                if isinstance(layer, (xr.DataArray, xu.UgridDataArray))
                else layer
            )
            da = xr.DataArray(values, dims=[self.data.grid.face_dimension])
            uda = xu.UgridDataArray(da, self.data.grid)
            uda = uda.astype(np.float32)
            uda.name = name
            uda.attrs.update(get_attrs(name))
            self.model.quadtree_grid.set(uda, name=name)

        configure(self.model.config, flavor=flavor, grid_type="quadtree")

    def get_vars_by_infiltration_type(self, infiltration_type: str):
        """Return infiltration variables to write and stale variables to remove."""
        write_vars = list(flavor_variables(infiltration_type))
        data_vars = self.data if isinstance(self.data, dict) else self.data.data_vars
        remove_vars = [
            name for name in VARIABLES if name not in write_vars and name in data_vars
        ]
        return write_vars, remove_vars

    def _read_sidecar(self, filename: Path, variables):
        if not filename.exists():
            raise FileNotFoundError(filename)
        with xr.open_dataset(filename) as ds:
            layers = {}
            for name in variables:
                if name not in ds:
                    raise ValueError(f"Missing variable '{name}' in {filename}.")
                da = self.mask.copy(deep=True)
                da.values = ds[name].values.astype(np.float32)
                da.name = name
                layers[name] = da
        return layers

    def read(self):
        """Read quadtree infiltration sidecars."""
        flavor = configured_flavor(self.model.config)
        if flavor is None or flavor == "con":
            return
        if flavor == "bkt":
            bucketfile = self.model.config.get("bucketfile", abs_path=True)
            if bucketfile is None:
                bucketfile = self.model.config.get("infiltrationfile", abs_path=True)
            if bucketfile is None:
                return
            layers = self._read_sidecar(bucketfile, BUCKET_VARS)
            self._set_layers(layers, flavor="bkt")
            if self.model.config.get("bucketfile") is not None:
                self.model.config.set("bucketfile", Path(bucketfile).name)
            elif self.model.config.get("infiltrationfile") is not None:
                self.model.config.set("infiltrationfile", Path(bucketfile).name)
            return
        inffile = self.model.config.get("infiltrationfile", abs_path=True)
        if inffile is None:
            return
        layers = self._read_sidecar(inffile, flavor_variables(flavor))
        self._set_layers(layers, flavor=flavor)
        self.model.config.set("infiltrationfile", Path(inffile).name)

    def write(self):
        """Write quadtree infiltration sidecars."""
        flavor = configured_flavor(self.model.config)
        if flavor is None or flavor == "con":
            return
        if flavor == "bkt":
            filename = self.model.config.get_set_file_variable(
                "bucketfile", default=DEFAULT_BUCKETFILE
            )
            variables = BUCKET_VARS
        else:
            filename = self.model.config.get_set_file_variable(
                "infiltrationfile", default=DEFAULT_INFILTRATIONFILE
            )
            variables = flavor_variables(flavor)
        filename.parent.mkdir(parents=True, exist_ok=True)
        ds = sidecar_dataset(
            {
                name: np.asarray(self.data[name].values, dtype=np.float32)
                for name in variables
            },
            len(self.mask.values),
        )
        ds.to_netcdf(filename)

    @hydromt_step
    def create_uniform_constant(self, qinf: float):
        """Create a uniform constant infiltration rate in model config."""
        self.clear()
        self.model.config.set("qinf", float(qinf))

    @hydromt_step
    def create_constant(
        self,
        qinf=None,
        lulc=None,
        reclass_table=None,
        reproj_method="average",
        nrmax=2000,
    ):
        """Create spatially varying constant infiltration rate.

        Adds model layers to SfincsModel.quadtree_grid.data:

        * **qinf** map: constant infiltration rate [mm/hr]

        Parameters
        ----------
        qinf : str, Path, or RasterDataset
            Spatially varying infiltration rates [mm/hr]. If this is an xr.Dataset,
            it must contain a ``qinf`` variable.
        lulc : str, Path, or RasterDataset
            Land-use dataset. Must be combined with ``reclass_table``.
        reclass_table : str, Path, or pd.DataFrame
            Reclassification table with a ``qinf`` column to convert land-use classes
            to infiltration rates [mm/hr].
        reproj_method : str, optional
            Resampling method for reprojecting source data to quadtree blocks.
        nrmax : int, optional
            Maximum number of cells per quadtree block.
        """
        # Add logger info
        logger.info("Creating constant spatially varying infiltration rate.")

        # get infiltration data
        if qinf is not None:
            da_qinf = self.data_catalog.get_rasterdataset(
                qinf,
                bbox=self.model.bbox,
                buffer=10,
            )
            if isinstance(da_qinf, xr.Dataset):
                if "qinf" not in da_qinf.data_vars:
                    raise ValueError(f"Could not find variable qinf in {qinf}")
                da_qinf = da_qinf["qinf"]
        # TODO check if this one is really necessary?
        # elif lulc is not None and ksat is not None:
        #     da_ksat = self.data_catalog.get_rasterdataset(
        #         ksat, bbox=self.model.bbox, buffer=10
        #     )
        #     da_lulc = self.data_catalog.get_rasterdataset(
        #         lulc, bbox=self.model.bbox, buffer=10
        #     )
        #     if lulc_modifiers is None:
        #         lulc_modifiers = (
        #             Path(DATADIR)
        #             / "infiltration"
        #             / "nlcd_infiltration_modifiers.csv"
        #         )
        #     if isinstance(lulc_modifiers, pd.DataFrame):
        #         df_modifiers = lulc_modifiers.copy()
        #     else:
        #         df_modifiers = self.data_catalog.get_dataframe(
        #             lulc_modifiers,
        #             source_kwargs={
        #                 "driver": {"name": "pandas", "options": {"index_col": 0}}
        #             },
        #         )
        #     da_qinf = workflows.constant_infiltration_from_ksat_lulc(
        #         da_ksat,
        #         da_lulc,
        #         df_modifiers,
        #         da_mask=self.mask,
        #         factor_ksat=factor_ksat,
        #     )
        elif lulc is not None:
            # landuse/landcover should always be combined with mapping
            if reclass_table is None:
                raise IOError(
                    f"Infiltration mapping file should be provided for {lulc}"
                )
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc,
                bbox=self.model.bbox,
                buffer=10,
                variables=["lulc"],
            )
            df_map = self.data_catalog.get_dataframe(
                reclass_table,
                variables=["qinf"],
                source_kwargs={
                    "driver": {"name": "pandas", "options": {"index_col": 0}}
                },
            )
            # reclassify
            da_qinf = da_lulc.raster.reclassify(df_map)["qinf"]
        else:
            raise ValueError(
                "Either qinf or lulc must be provided when setting up constant infiltration."
            )

        # set nodata to nan before reprojecting/interpolating
        da_qinf = da_qinf.raster.mask_nodata()

        n_cells = self.data.grid.n_face
        qinf = np.full(n_cells, np.nan)

        # Function to compute infiltration values for a chunk of the quadtree grid
        def compute_constant_infiltration(da_like, ilev=None):
            # reproject infiltration data to model grid
            da_out = da_qinf.raster.reproject_like(da_like, method=reproj_method)
            return da_out

        # Compute constant infiltration in chunks over the quadtree grid
        self.compute_quadtree(
            compute_constant_infiltration,
            qinf,
            nrmax=nrmax,
        )

        # check on nan values
        if np.logical_and(np.isnan(qinf), self.mask >= 1).any():
            logger.warning("NaN values found in infiltration data; filled with 0")
            qinf = np.where(np.isnan(qinf), 0, qinf)

        # set grid
        self._set_layers({"qinf": qinf}, flavor="c2d")

    @hydromt_step
    def create_cn(self, cn, antecedent_moisture="avg", reproj_method="med", nrmax=2000):
        """Create Curve Number infiltration without recovery for quadtree grids.

        Adds model layers:

        * **scs** map: potential maximum soil moisture retention [inch]

        Parameters
        ---------
        cn : str, Path, or RasterDataset
            Name of gridded curve number map.
        antecedent_moisture : {'dry', 'avg', 'wet'}, optional
            Antecedent runoff conditions.
            By default `avg`
        reproj_method : str, optional
            Resampling method for reprojecting curve number data to quadtree blocks.
        nrmax : int, optional
            Maximum number of cells per quadtree block.

        """
        # Add logger info
        logger.info(
            f"Creating curve number values for SFINCS quadtree grid with antecedent moisture condition: {antecedent_moisture}."
        )

        # get data
        da_org = self.model.data_catalog.get_rasterdataset(
            cn, bbox=self.model.bbox, buffer=10
        )
        v = "cn"
        if antecedent_moisture:
            v = f"cn_{antecedent_moisture}"
        if isinstance(da_org, xr.Dataset) and v in da_org.data_vars:
            da_org = da_org[v]
        elif isinstance(da_org, xr.Dataset):
            raise ValueError(f"Could not find variable {v} in {cn}")

        n_cells = self.data.grid.n_face
        scs = np.full(n_cells, np.nan)

        def compute_cn_infiltration(da_like, ilev=None):
            da_cn = da_org.raster.reproject_like(da_like, method=reproj_method)
            da_scs = workflows.cn_to_s(da_cn).round(3)
            return da_scs

        self.compute_quadtree(
            compute_cn_infiltration,
            scs,
            nrmax=nrmax,
        )

        # check on nan values
        if np.logical_and(np.isnan(scs), self.mask >= 1).any():
            logger.warning(
                "NaN values found in curve-number data; filled with 100 (impermeable)"
            )
            scs = np.where(np.isnan(scs), 100, scs)

        self._set_layers({"scs": scs}, flavor="cna")

    @hydromt_step
    def create_cn_from_landuse_hsg(
        self,
        lulc,
        hsg,
        reclass_table,
        antecedent_moisture="avg",
        reproj_method="median",
    ):
        """Create Curve Number infiltration from land use and HSG.

        Adds model layers:

        * **scs** map: potential maximum soil moisture retention [inch]

        Parameters
        ----------
        lulc : str, Path, or RasterDataset
            Name of gridded land use map.
        hsg : str, Path, or RasterDataset
            Name of gridded hydrologic soil group map.
        reclass_table : str, Path, or DataFrame
            Reclassification table mapping land use and hydrologic soil groups to curve numbers.
        antecedent_moisture : {'dry', 'avg', 'wet'}, optional
            Antecedent runoff conditions.
            By default `avg`
        reproj_method : str, optional
            Resampling method for reprojecting curve number data to quadtree blocks.
        """

        da_lulc = self.data_catalog.get_rasterdataset(
            lulc, bbox=self.model.bbox, buffer=10, variables=["lulc"]
        )
        if isinstance(da_lulc, xr.Dataset):
            da_lulc = next(iter(da_lulc.data_vars.values()))
        da_hsg = self.data_catalog.get_rasterdataset(
            hsg, bbox=self.model.bbox, buffer=10, variables=["hsg"]
        )
        if isinstance(da_hsg, xr.Dataset):
            da_hsg = next(iter(da_hsg.data_vars.values()))
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )
        da_hsg = da_hsg.raster.reproject_like(da_lulc, method="nearest")
        da_cn = workflows.curve_number_from_landuse_hsg(da_lulc, da_hsg, df_map)
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
        """Create Curve Number infiltration with recovery for quadtree grids.

        Adds **smax**, **seff**, and **ks** maps on quadtree faces. Input data are
        read by the component; the computation receives xarray objects and a
        pandas DataFrame.
        """
        del block_size  # kept for backwards compatibility
        da_landuse = self.data_catalog.get_rasterdataset(
            lulc, bbox=self.model.bbox, buffer=10
        )
        if isinstance(da_landuse, xr.Dataset):
            da_landuse = next(iter(da_landuse.data_vars.values()))
        da_hsg = self.data_catalog.get_rasterdataset(
            hsg, bbox=self.model.bbox, buffer=10
        )
        if isinstance(da_hsg, xr.Dataset):
            da_hsg = next(iter(da_hsg.data_vars.values()))
        da_hsg = da_hsg.raster.reproject_like(da_landuse, method="nearest")
        da_ksat = self.data_catalog.get_rasterdataset(
            ksat, bbox=self.model.bbox, buffer=10
        )
        if isinstance(da_ksat, xr.Dataset):
            da_ksat = next(iter(da_ksat.data_vars.values()))
        da_ksat = da_ksat.raster.reproject_like(da_landuse, method="average")
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )
        ds = workflows.curve_number_with_recovery(
            da_landuse,
            da_hsg,
            da_ksat,
            df_map,
            effective=effective,
            factor_ksat=factor_ksat,
        )
        outputs = {
            name: np.full(self.data.grid.n_face, np.nan)
            for name in ("smax", "seff", "ks")
        }

        def compute_block(da_like, ilev=None):
            return tuple(
                ds[name].raster.reproject_like(da_like, method="average")
                for name in outputs
            )

        self.compute_quadtree(compute_block, outputs)
        self._set_layers(outputs, flavor="cnb")

    @hydromt_step
    def create_green_ampt(
        self,
        psi,
        sigma,
        ks,
        reproj_method="average",
    ):
        """Create Green-Ampt infiltration from final parameter maps.

        Adds model layers:

        * **psi** map: wetting front suction head [mm]
        * **sigma** map: soil moisture deficit [-]
        * **ks** map: saturated hydraulic conductivity [mm/hr]

        Parameters
        ----------
        psi, sigma, ks : str, Path, RasterDataset, or UgridDataArray
            Data with final Green-Ampt parameters. Dataset inputs must contain
            variables named ``psi``, ``sigma``, and ``ks`` respectively.
        reproj_method : str, optional
            Resampling method for reprojecting raster inputs to quadtree blocks.
        """
        names = ("psi", "sigma", "ks")
        layers = {}
        raster_sources = {}
        for name, source in zip(names, (psi, sigma, ks)):
            if isinstance(source, xu.UgridDataArray):
                layers[name] = source
                continue
            if isinstance(source, xu.UgridDataset):
                layers[name] = source[name]
                continue
            da = self.data_catalog.get_rasterdataset(
                source, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da, xr.Dataset):
                if name not in da.data_vars:
                    raise ValueError(f"Could not find variable {name} in {source}")
                da = da[name]
            raster_sources[name] = da.raster.mask_nodata()

        if raster_sources:
            outputs = {
                name: np.full(self.data.grid.n_face, np.nan) for name in raster_sources
            }

            def compute_block(da_like, ilev=None):
                return tuple(
                    raster_sources[name].raster.reproject_like(
                        da_like, method=reproj_method
                    )
                    for name in outputs
                )

            self.compute_quadtree(compute_block, outputs)
            layers.update(outputs)
        self._set_layers(layers, flavor="gai")

    @hydromt_step
    def create_green_ampt_from_soil(
        self,
        hsg,
        ksat=None,
        lulc=None,
        reclass_table=None,
        lulc_modifiers=None,
        dual_hsg="drained",
        factor_ksat=3.6,
        reproj_method="average",
    ):
        """Estimate Green-Ampt infiltration from HSG and optional landuse.

        Parameters
        ----------
        hsg : str, Path, or RasterDataset
            Hydrologic soil group map. By default, values are reclassified with
            the bundled ``hsg_green_ampt.csv`` table.
        ksat : str, Path, or RasterDataset, optional
            Saturated hydraulic conductivity map. If provided, it overrides or
            derives ``ks`` values from the reclassification table.
        lulc : str, Path, or RasterDataset, optional
            Land-use map used to apply NLCD infiltration modifiers.
        reclass_table : str, Path, or DataFrame, optional
            Table mapping HSG classes to Green-Ampt parameters.
        lulc_modifiers : str, Path, or DataFrame, optional
            Table with land-use modifier factors.
        dual_hsg : {None, 'native', 'drained'}, optional
            How to handle dual HSG classes.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr.
        reproj_method : str, optional
            Resampling method for reprojecting final parameter maps to quadtree blocks.
        """
        if reclass_table is None:
            reclass_table = Path(DATADIR) / "infiltration" / "hsg_green_ampt.csv"

        da_soil = self.data_catalog.get_rasterdataset(
            hsg, bbox=self.model.bbox, buffer=10, variables=["hsg"]
        )
        if isinstance(da_soil, xr.Dataset):
            da_soil = next(iter(da_soil.data_vars.values()))
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )
        da_ksat = None
        if ksat is not None:
            da_ksat = self.data_catalog.get_rasterdataset(
                ksat, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da_ksat, xr.Dataset):
                da_ksat = next(iter(da_ksat.data_vars.values()))
            da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")

        if lulc is not None:
            if lulc_modifiers is None:
                lulc_modifiers = (
                    Path(DATADIR) / "infiltration" / "nlcd_infiltration_modifiers.csv"
                )
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc, bbox=self.model.bbox, buffer=10, variables=["lulc"]
            )
            if isinstance(da_lulc, xr.Dataset):
                da_lulc = da_lulc["lulc"]
            da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
            df_modifiers = self.data_catalog.get_dataframe(
                lulc_modifiers,
                source_kwargs={
                    "driver": {"name": "pandas", "options": {"index_col": 0}}
                },
            )
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

        outputs = {
            name: np.full(self.data.grid.n_face, np.nan)
            for name in ("psi", "sigma", "ks")
        }

        def compute_block(da_like, ilev=None):
            return tuple(
                ds[name].raster.reproject_like(da_like, method=reproj_method)
                for name in outputs
            )

        self.compute_quadtree(compute_block, outputs)
        self._set_layers(outputs, flavor="gai")

    @hydromt_step
    def create_horton(
        self,
        f0,
        fc,
        kd,
        reproj_method="average",
    ):
        """Create Horton infiltration from final parameter maps.

        Adds model layers:

        * **f0** map: initial infiltration capacity [mm/hr]
        * **fc** map: asymptotic infiltration capacity [mm/hr]
        * **kd** map: Horton decay coefficient [hr-1]

        Parameters
        ----------
        f0, fc, kd : str, Path, RasterDataset, or UgridDataArray
            Data with final Horton parameters. Dataset inputs must contain
            variables named ``f0``, ``fc``, and ``kd`` respectively.
        reproj_method : str, optional
            Resampling method for reprojecting raster inputs to quadtree blocks.
        """
        names = ("f0", "fc", "kd")
        layers = {}
        raster_sources = {}
        for name, source in zip(names, (f0, fc, kd)):
            if isinstance(source, xu.UgridDataArray):
                layers[name] = source
                continue
            if isinstance(source, xu.UgridDataset):
                layers[name] = source[name]
                continue
            da = self.data_catalog.get_rasterdataset(
                source, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da, xr.Dataset):
                if name not in da.data_vars:
                    raise ValueError(f"Could not find variable {name} in {source}")
                da = da[name]
            raster_sources[name] = da.raster.mask_nodata()

        if raster_sources:
            outputs = {
                name: np.full(self.data.grid.n_face, np.nan) for name in raster_sources
            }

            def compute_block(da_like, ilev=None):
                return tuple(
                    raster_sources[name].raster.reproject_like(
                        da_like, method=reproj_method
                    )
                    for name in outputs
                )

            self.compute_quadtree(compute_block, outputs)
            layers.update(outputs)
        self._set_layers(layers, flavor="hor")

    @hydromt_step
    def create_horton_from_soil(
        self,
        hsg,
        ksat=None,
        lulc=None,
        reclass_table=None,
        lulc_modifiers=None,
        dual_hsg="drained",
        factor_ksat=3.6,
        reproj_method="average",
    ):
        """Estimate Horton infiltration from HSG and optional landuse.

        Parameters
        ----------
        hsg : str, Path, or RasterDataset
            Hydrologic soil group map. By default, values are reclassified with
            the bundled ``hsg_horton.csv`` table.
        ksat : str, Path, or RasterDataset, optional
            Saturated hydraulic conductivity map. If provided, it overrides or
            derives ``fc`` values from the reclassification table.
        lulc : str, Path, or RasterDataset, optional
            Land-use map used to apply NLCD infiltration modifiers.
        reclass_table : str, Path, or DataFrame, optional
            Table mapping HSG classes to Horton parameters.
        lulc_modifiers : str, Path, or DataFrame, optional
            Table with land-use modifier factors.
        dual_hsg : {None, 'native', 'drained'}, optional
            How to handle dual HSG classes.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr.
        reproj_method : str, optional
            Resampling method for reprojecting final parameter maps to quadtree blocks.
        """
        if reclass_table is None:
            reclass_table = Path(DATADIR) / "infiltration" / "hsg_horton.csv"

        da_soil = self.data_catalog.get_rasterdataset(
            hsg, bbox=self.model.bbox, buffer=10, variables=["hsg"]
        )
        if isinstance(da_soil, xr.Dataset):
            da_soil = next(iter(da_soil.data_vars.values()))
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )
        da_ksat = None
        if ksat is not None:
            da_ksat = self.data_catalog.get_rasterdataset(
                ksat, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da_ksat, xr.Dataset):
                da_ksat = next(iter(da_ksat.data_vars.values()))
            da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")

        if lulc is not None:
            if lulc_modifiers is None:
                lulc_modifiers = (
                    Path(DATADIR) / "infiltration" / "nlcd_infiltration_modifiers.csv"
                )
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc, bbox=self.model.bbox, buffer=10, variables=["lulc"]
            )
            if isinstance(da_lulc, xr.Dataset):
                da_lulc = da_lulc["lulc"]
            da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
            df_modifiers = self.data_catalog.get_dataframe(
                lulc_modifiers,
                source_kwargs={
                    "driver": {"name": "pandas", "options": {"index_col": 0}}
                },
            )
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

        outputs = {
            name: np.full(self.data.grid.n_face, np.nan) for name in ("f0", "fc", "kd")
        }

        def compute_block(da_like, ilev=None):
            return tuple(
                ds[name].raster.reproject_like(da_like, method=reproj_method)
                for name in outputs
            )

        self.compute_quadtree(compute_block, outputs)
        self._set_layers(outputs, flavor="hor")

    @hydromt_step
    def create_bucket(
        self,
        bucket_smax,
        bucket_k,
        bucket_loss=None,
        reproj_method="average",
    ):
        """Create bucket infiltration from final parameter maps.

        Adds model layers:

        * **bucket_smax** map: bucket maximum storage [mm]
        * **bucket_k** map: bucket drainage coefficient [hr-1]
        * **bucket_loss** map: bucket loss fraction [-]

        Parameters
        ----------
        bucket_smax, bucket_k : str, Path, RasterDataset, or UgridDataArray
            Data with final bucket parameters. Dataset inputs must contain
            variables named ``bucket_smax`` and ``bucket_k`` respectively.
        bucket_loss : float, str, Path, RasterDataset, or UgridDataArray, optional
            Uniform loss fraction or map with final bucket loss fractions.
            Defaults to 0.0.
        reproj_method : str, optional
            Resampling method for reprojecting raster inputs to quadtree blocks.
        """
        names = ("bucket_smax", "bucket_k")
        layers = {}
        raster_sources = {}
        for name, source in zip(names, (bucket_smax, bucket_k)):
            if isinstance(source, xu.UgridDataArray):
                layers[name] = source
                continue
            if isinstance(source, xu.UgridDataset):
                layers[name] = source[name]
                continue
            da = self.data_catalog.get_rasterdataset(
                source, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da, xr.Dataset):
                if name not in da.data_vars:
                    raise ValueError(f"Could not find variable {name} in {source}")
                da = da[name]
            raster_sources[name] = da.raster.mask_nodata()

        if bucket_loss is None or np.isscalar(bucket_loss):
            layers["bucket_loss"] = np.full(
                self.data.grid.n_face,
                np.float32(0.0 if bucket_loss is None else bucket_loss),
            )
        elif isinstance(bucket_loss, xu.UgridDataArray):
            layers["bucket_loss"] = bucket_loss
        else:
            da_loss = self.data_catalog.get_rasterdataset(
                bucket_loss, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da_loss, xr.Dataset):
                da_loss = da_loss["bucket_loss"]
            raster_sources["bucket_loss"] = da_loss.raster.mask_nodata()

        if raster_sources:
            outputs = {
                name: np.full(self.data.grid.n_face, np.nan) for name in raster_sources
            }

            def compute_block(da_like, ilev=None):
                return tuple(
                    raster_sources[name].raster.reproject_like(
                        da_like, method=reproj_method
                    )
                    for name in outputs
                )

            self.compute_quadtree(compute_block, outputs)
            layers.update(outputs)
        self._set_layers(layers, flavor="bkt")

    @hydromt_step
    def create_bucket_from_soil(
        self,
        hsg,
        ksat=None,
        lulc=None,
        reclass_table=None,
        lulc_modifiers=None,
        dual_hsg="drained",
        factor_ksat=3.6,
        bucket_loss=None,
        reproj_method="average",
    ):
        """Estimate bucket infiltration from HSG and optional landuse.

        Parameters
        ----------
        hsg : str, Path, or RasterDataset
            Hydrologic soil group map. By default, values are reclassified with
            the bundled ``hsg_bucket.csv`` table.
        ksat : str, Path, or RasterDataset, optional
            Saturated hydraulic conductivity map. If provided, it helps derive
            ``bucket_k`` values.
        lulc : str, Path, or RasterDataset, optional
            Land-use map used to apply NLCD infiltration modifiers.
        reclass_table : str, Path, or DataFrame, optional
            Table mapping HSG classes to bucket parameters.
        lulc_modifiers : str, Path, or DataFrame, optional
            Table with land-use modifier factors.
        dual_hsg : {None, 'native', 'drained'}, optional
            How to handle dual HSG classes.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr.
        bucket_loss : float, optional
            Uniform bucket loss fraction. Defaults to 0.0 without land use and
            0.10 with land-use modifiers.
        reproj_method : str, optional
            Resampling method for reprojecting final parameter maps to quadtree blocks.
        """
        if reclass_table is None:
            reclass_table = Path(DATADIR) / "infiltration" / "hsg_bucket.csv"

        da_soil = self.data_catalog.get_rasterdataset(
            hsg, bbox=self.model.bbox, buffer=10, variables=["hsg"]
        )
        if isinstance(da_soil, xr.Dataset):
            da_soil = next(iter(da_soil.data_vars.values()))
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )
        da_ksat = None
        if ksat is not None:
            da_ksat = self.data_catalog.get_rasterdataset(
                ksat, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da_ksat, xr.Dataset):
                da_ksat = next(iter(da_ksat.data_vars.values()))
            da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")

        if lulc is not None:
            if lulc_modifiers is None:
                lulc_modifiers = (
                    Path(DATADIR) / "infiltration" / "nlcd_infiltration_modifiers.csv"
                )
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc, bbox=self.model.bbox, buffer=10, variables=["lulc"]
            )
            if isinstance(da_lulc, xr.Dataset):
                da_lulc = da_lulc["lulc"]
            da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
            df_modifiers = self.data_catalog.get_dataframe(
                lulc_modifiers,
                source_kwargs={
                    "driver": {"name": "pandas", "options": {"index_col": 0}}
                },
            )
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

        outputs = {name: np.full(self.data.grid.n_face, np.nan) for name in BUCKET_VARS}

        def compute_block(da_like, ilev=None):
            return tuple(
                ds[name].raster.reproject_like(da_like, method=reproj_method)
                for name in outputs
            )

        self.compute_quadtree(compute_block, outputs)
        self._set_layers(outputs, flavor="bkt")

    def clear(self):
        """Clear all infiltration layers from the model."""
        self.model.quadtree_grid._data = clear_data(self.data, keep=())
        reset_config(self.model.config)
