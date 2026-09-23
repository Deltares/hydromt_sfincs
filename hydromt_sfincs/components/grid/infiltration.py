import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Union

import numpy as np
import pandas as pd
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import DATADIR, workflows
from hydromt_sfincs.components.grid.regulargrid_mixin import SfincsRegularGridMixin
from hydromt_sfincs.components.infiltration_common import (
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


class SfincsInfiltration(SfincsRegularGridMixin, ModelComponent):
    """SFINCS infiltration component for regular grids.

    Unsuffixed ``create_*`` methods use final SFINCS parameter maps directly.
    Methods with a ``_from_soil`` suffix estimate those parameters from HSG,
    optional Ksat, and optional land-use modifiers.
    """

    def __init__(self, model: "SfincsModel"):
        super().__init__(model=model)

    @property
    def data(self) -> xr.Dataset:
        return self.model.grid.data

    @property
    def mask(self) -> xr.DataArray:
        return self.model.grid.mask

    def _set_layers(self, layers: dict[str, xr.DataArray], flavor: str) -> None:
        self.clear()
        for name, da in layers.items():
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
            self.model.grid.set(da)
        configure(self.model.config, flavor=flavor, grid_type="regular")

    def _read_sidecar(
        self, filename: Path, variables: Iterable[str]
    ) -> dict[str, xr.DataArray]:
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

    def read(self) -> None:
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

    def write(self) -> None:
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
    def create_uniform_constant(self, qinf: float) -> None:
        """Create a uniform constant infiltration rate in model config.

        Sets the ``qinf`` config entry and clears any existing infiltration
        layers; no grid layers are added.

        Parameters
        ----------
        qinf : float
            Uniform infiltration rate [mm/hr].
        """
        self.clear()
        self.model.config.set("qinf", float(qinf))

    @hydromt_step
    def create_constant(
        self,
        qinf: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        lulc: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        reclass_table: Union[str, Path, pd.DataFrame, None] = None,
        reproj_method: str = "average",
    ) -> None:
        """Create spatially varying constant infiltration rate.

        Adds model layers to SfincsModel.grid.data:

        * **qinf** map: constant infiltration rate [mm/hr]

        Parameters
        ----------
        qinf : str, Path, or RasterDataset
            Spatially varying infiltration rates [mm/hr]. If this is an xr.Dataset,
            it must contain a ``qinf`` variable.
        lulc: str, Path, or RasterDataset
            Landuse/landcover dataset. Must be combined with ``reclass_table``.
        reclass_table: str, Path, or pd.DataFrame
            Reclassification table with a ``qinf`` column to convert land-use classes
            to infiltration rates [mm/hr].
        reproj_method : str, optional
            Resampling method for reprojecting the infiltration data to the model grid.
            By default 'average'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
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

        # reproject infiltration data to model grid
        da_qinf = da_qinf.raster.mask_nodata()  # set nodata to nan
        da_qinf = da_qinf.raster.reproject_like(self.mask, method=reproj_method)

        # check on nan values
        if np.logical_and(np.isnan(da_qinf), self.mask >= 1).any():
            logger.warning("NaN values found in infiltration data; filled with 0")
            da_qinf = da_qinf.fillna(0)
        da_qinf.raster.set_nodata(-9999.0)

        # set grid
        self._set_layers({"qinf": da_qinf}, flavor="c2d")

    @hydromt_step
    def create_cn(
        self,
        cn: Union[str, Path, xr.DataArray, xr.Dataset],
        antecedent_moisture: str = "avg",
        reproj_method: str = "med",
    ) -> None:
        """Create Curve Number infiltration without recovery.

        Adds model layers:

        * **scs** map: potential maximum soil moisture retention [inch]

        Parameters
        ----------
        cn : str, Path, or RasterDataset
            Curve number data. Dataset inputs must contain a ``cn`` variable, or
            ``cn_<antecedent_moisture>`` when ``antecedent_moisture`` is set.
        antecedent_moisture : {'dry', 'avg', 'wet'}, optional
            Antecedent runoff condition used to select the source variable. Set to
            None when the input already holds adjusted curve numbers.
            By default 'avg'.
        reproj_method : str, optional
            Resampling method for reprojecting the curve number data to the model grid.
            By default 'med'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        """
        # Add logger info
        logger.info(
            f"Creating curve number values for SFINCS with antecedent moisture condition: {antecedent_moisture}."
        )

        # get data
        da_org = self.data_catalog.get_rasterdataset(
            cn, bbox=self.model.bbox, buffer=10
        )
        # read variable
        v = "cn"
        if antecedent_moisture:
            v = f"cn_{antecedent_moisture}"
        if isinstance(da_org, xr.Dataset) and v in da_org.data_vars:
            da_org = da_org[v]
        elif not isinstance(da_org, xr.DataArray):
            raise ValueError(f"Could not find variable {v} in {cn}")

        # reproject using median
        da_cn = da_org.raster.reproject_like(self.mask, method=reproj_method)

        # convert to potential maximum soil moisture retention S (1000/CN - 10) [inch]
        da_scs = workflows.cn_to_s(da_cn, self.mask > 0).round(3)
        self._set_layers({"scs": da_scs}, flavor="cna")

    @hydromt_step
    def create_cn_from_landuse_hsg(
        self,
        lulc: Union[str, Path, xr.DataArray, xr.Dataset],
        hsg: Union[str, Path, xr.DataArray, xr.Dataset],
        reclass_table: Union[str, Path, pd.DataFrame],
        antecedent_moisture: str = "avg",
        reproj_method: str = "med",
    ) -> None:
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
            Resampling method for reprojecting the curve number data to the model grid.
            By default 'med'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        """

        da_lulc = self.data_catalog.get_rasterdataset(
            lulc, bbox=self.model.bbox, buffer=10, variables=["lulc"]
        )
        da_hsg = self.data_catalog.get_rasterdataset(
            hsg, bbox=self.model.bbox, buffer=10, variables=["hsg"]
        )
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )
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
        lulc: Union[str, Path, xr.DataArray, xr.Dataset],
        hsg: Union[str, Path, xr.DataArray, xr.Dataset],
        ksat: Union[str, Path, xr.DataArray, xr.Dataset],
        reclass_table: Union[str, Path, pd.DataFrame],
        effective: float,
        factor_ksat: float = 3.6,
        block_size: int = 2000,
    ) -> None:
        """Create Curve Number infiltration with recovery.

        Adds model layers:

        * **smax** map: maximum soil moisture retention [m]
        * **seff** map: effective soil moisture retention [m]
        * **ks** map: saturated hydraulic conductivity [mm/hr]

        The block traversal is handled by
        :py:meth:`SfincsRegularGridMixin.compute_regular_grid`; each block uses
        :py:func:`hydromt_sfincs.workflows.curve_number_with_recovery`.

        Parameters
        ----------
        lulc : str, Path, or RasterDataset
            Landuse/landcover data set.
        hsg : str, Path, or RasterDataset
            Hydrologic soil group map in integers.
        ksat : str, Path, or RasterDataset
            Saturated hydraulic conductivity, in the units implied by ``factor_ksat``.
        reclass_table : str, Path, or DataFrame
            Reclassification table relating land cover and soil type to curve numbers.
        effective : float
            Fraction of ``smax`` that is effective soil retention, e.g. 0.50 for 50%.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr, by default 3.6
            (micrometer per second to mm/hr).
        block_size : int, optional
            Maximum block size in model cells. Larger values hold more data in
            memory but can be faster, by default 2000.
        """

        # Add logger info
        logger.info("Creating curve number values for SFINCS including recovery term.")

        # Read the datafiles
        da_landuse = self.data_catalog.get_rasterdataset(
            lulc, bbox=self.model.bbox, buffer=10
        )
        da_HSG = self.data_catalog.get_rasterdataset(
            hsg, bbox=self.model.bbox, buffer=10
        )
        da_Ksat = self.data_catalog.get_rasterdataset(
            ksat, bbox=self.model.bbox, buffer=10
        )
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )

        # Define outputs
        layers = {
            "smax": xr.full_like(self.mask, -9999.0, dtype=np.float32),
            "seff": xr.full_like(self.mask, -9999.0, dtype=np.float32),
            "ks": xr.full_like(self.mask, -9999.0, dtype=np.float32),
        }

        # Compute resolution land use (we are assuming that is the finest)
        resolution_landuse = np.mean(
            [abs(da_landuse.raster.res[0]), abs(da_landuse.raster.res[1])]
        )
        if da_landuse.raster.crs.is_geographic:
            resolution_landuse = (
                resolution_landuse * 111111.0
            )  # assume 1 degree is 111km

        def compute_cn_recovery_block(da_like):
            ds = workflows.curve_number_with_recovery(
                da_landuse,
                da_HSG,
                da_Ksat,
                df_map,
                effective=effective,
                factor_ksat=factor_ksat,
                da_mask=da_like,
            )
            return {name: ds[name] for name in ("smax", "seff", "ks")}

        layers = self.compute_regular_grid(
            compute_cn_recovery_block,
            layers,
            block_size=block_size,
            source_resolution=resolution_landuse,
        )
        self._set_layers(layers, flavor="cnb")

    @hydromt_step
    def create_green_ampt(
        self,
        psi: Union[str, Path, xr.DataArray, xr.Dataset],
        sigma: Union[str, Path, xr.DataArray, xr.Dataset],
        ks: Union[str, Path, xr.DataArray, xr.Dataset],
        reproj_method: str = "average",
    ) -> None:
        """Create Green-Ampt infiltration from final parameter maps.

        Adds model layers:

        * **psi** map: wetting front suction head [mm]
        * **sigma** map: soil moisture deficit [-]
        * **ks** map: saturated hydraulic conductivity [mm/hr]

        Parameters
        ----------
        psi, sigma, ks : str, Path, or RasterDataset
            Raster data with final Green-Ampt parameters. Dataset inputs must
            contain variables named ``psi``, ``sigma``, and ``ks`` respectively.
        reproj_method : str, optional
            Resampling method for reprojecting the parameter maps to the model grid.
        """
        layers = {}
        for name, source in {"psi": psi, "sigma": sigma, "ks": ks}.items():
            da = self.data_catalog.get_rasterdataset(
                source, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da, xr.Dataset):
                if name not in da.data_vars:
                    raise ValueError(f"Could not find variable {name} in {source}")
                da = da[name]
            da = da.raster.mask_nodata()
            layers[name] = da.raster.reproject_like(self.mask, method=reproj_method)
        self._set_layers(layers, flavor="gai")

    @hydromt_step
    def create_green_ampt_from_soil(
        self,
        hsg: Union[str, Path, xr.DataArray, xr.Dataset],
        ksat: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        lulc: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        reclass_table: Union[str, Path, pd.DataFrame, None] = None,
        lulc_modifiers: Union[str, Path, pd.DataFrame, None] = None,
        dual_hsg: str = "drained",
        factor_ksat: float = 3.6,
        reproj_method: str = "average",
    ) -> None:
        """Estimate Green-Ampt infiltration from HSG and optional landuse.

        Adds model layers:

        * **psi** map: wetting front suction head [mm]
        * **sigma** map: soil moisture deficit [-]
        * **ks** map: saturated hydraulic conductivity [mm/hr]

        Parameters
        ----------
        hsg : str, Path, or RasterDataset
            Hydrologic soil group map. By default, values are reclassified with
            the bundled ``hsg_green_ampt.csv`` table.
        ksat : str, Path, or RasterDataset, optional
            Saturated hydraulic conductivity map. If provided, it overrides or
            derives ``ks`` values from the reclassification table.
        lulc : str, Path, or RasterDataset, optional
            Land-use map used to apply infiltration modifiers. Its classes must
            match the index of ``lulc_modifiers``.
        reclass_table : str, Path, or DataFrame, optional
            Table mapping HSG classes to Green-Ampt parameters.
        lulc_modifiers : str, Path, or DataFrame, optional
            Table with land-use modifier factors, indexed by land-use class. By
            default the bundled NLCD table is used.
        dual_hsg : {None, 'native', 'drained'}, optional
            How to handle dual HSG classes, by default 'drained'.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr, by default 3.6.
        reproj_method : str, optional
            Resampling method for reprojecting final parameter maps to the model grid.
            By default 'average'.
        """
        if reclass_table is None:
            reclass_table = Path(DATADIR) / "infiltration" / "hsg_green_ampt.csv"

        da_soil = self.data_catalog.get_rasterdataset(
            hsg,
            bbox=self.model.bbox,
            buffer=10,
        )
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )

        da_ksat = None
        if ksat is not None:
            da_ksat = self.data_catalog.get_rasterdataset(
                ksat, bbox=self.model.bbox, buffer=10
            )
            da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")

        if lulc is not None:
            if lulc_modifiers is None:
                lulc_modifiers = (
                    Path(DATADIR) / "infiltration" / "nlcd_infiltration_modifiers.csv"
                )
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc,
                bbox=self.model.bbox,
                buffer=10,
            )
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

        layers = {}
        for name in ("psi", "sigma", "ks"):
            da = ds[name]
            try:
                da = da.raster.mask_nodata()
            except Exception:
                pass
            layers[name] = da.raster.reproject_like(self.mask, method=reproj_method)
        self._set_layers(layers, flavor="gai")

    @hydromt_step
    def create_horton(
        self,
        f0: Union[str, Path, xr.DataArray, xr.Dataset],
        fc: Union[str, Path, xr.DataArray, xr.Dataset],
        kd: Union[str, Path, xr.DataArray, xr.Dataset],
        reproj_method: str = "average",
    ) -> None:
        """Create Horton infiltration from final parameter maps.

        Adds model layers:

        * **f0** map: initial infiltration capacity [mm/hr]
        * **fc** map: asymptotic infiltration capacity [mm/hr]
        * **kd** map: Horton decay coefficient [hr-1]

        Parameters
        ----------
        f0, fc, kd : str, Path, or RasterDataset
            Raster data with final Horton parameters. Dataset inputs must contain
            variables named ``f0``, ``fc``, and ``kd`` respectively.
        reproj_method : str, optional
            Resampling method for reprojecting the parameter maps to the model grid.
        """
        layers = {}
        for name, source in {"f0": f0, "fc": fc, "kd": kd}.items():
            da = self.data_catalog.get_rasterdataset(
                source, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da, xr.Dataset):
                if name not in da.data_vars:
                    raise ValueError(f"Could not find variable {name} in {source}")
                da = da[name]
            da = da.raster.mask_nodata()
            layers[name] = da.raster.reproject_like(self.mask, method=reproj_method)
        self._set_layers(layers, flavor="hor")

    @hydromt_step
    def create_horton_from_soil(
        self,
        hsg: Union[str, Path, xr.DataArray, xr.Dataset],
        ksat: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        lulc: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        reclass_table: Union[str, Path, pd.DataFrame, None] = None,
        lulc_modifiers: Union[str, Path, pd.DataFrame, None] = None,
        dual_hsg: str = "drained",
        factor_ksat: float = 3.6,
        reproj_method: str = "average",
    ) -> None:
        """Estimate Horton infiltration from HSG and optional landuse.

        Adds model layers:

        * **f0** map: initial infiltration capacity [mm/hr]
        * **fc** map: asymptotic infiltration capacity [mm/hr]
        * **kd** map: Horton decay coefficient [hr-1]

        Parameters
        ----------
        hsg : str, Path, or RasterDataset
            Hydrologic soil group map. By default, values are reclassified with
            the bundled ``hsg_horton.csv`` table.
        ksat : str, Path, or RasterDataset, optional
            Saturated hydraulic conductivity map. If provided, it overrides or
            derives ``fc`` values from the reclassification table.
        lulc : str, Path, or RasterDataset, optional
            Land-use map used to apply infiltration modifiers. Its classes must
            match the index of ``lulc_modifiers``.
        reclass_table : str, Path, or DataFrame, optional
            Table mapping HSG classes to Horton parameters.
        lulc_modifiers : str, Path, or DataFrame, optional
            Table with land-use modifier factors, indexed by land-use class. By
            default the bundled NLCD table is used.
        dual_hsg : {None, 'native', 'drained'}, optional
            How to handle dual HSG classes, by default 'drained'.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr, by default 3.6.
        reproj_method : str, optional
            Resampling method for reprojecting final parameter maps to the model grid.
            By default 'average'.
        """
        if reclass_table is None:
            reclass_table = Path(DATADIR) / "infiltration" / "hsg_horton.csv"

        da_soil = self.data_catalog.get_rasterdataset(
            hsg,
            bbox=self.model.bbox,
            buffer=10,
        )
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )

        da_ksat = None
        if ksat is not None:
            da_ksat = self.data_catalog.get_rasterdataset(
                ksat, bbox=self.model.bbox, buffer=10
            )
            da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")

        if lulc is not None:
            if lulc_modifiers is None:
                lulc_modifiers = (
                    Path(DATADIR) / "infiltration" / "nlcd_infiltration_modifiers.csv"
                )
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc,
                bbox=self.model.bbox,
                buffer=10,
            )
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

        layers = {}
        for name in ("f0", "fc", "kd"):
            da = ds[name]
            try:
                da = da.raster.mask_nodata()
            except Exception:
                pass
            layers[name] = da.raster.reproject_like(self.mask, method=reproj_method)
        self._set_layers(layers, flavor="hor")

    @hydromt_step
    def create_bucket(
        self,
        bucket_smax: Union[str, Path, xr.DataArray, xr.Dataset],
        bucket_k: Union[str, Path, xr.DataArray, xr.Dataset],
        bucket_loss: Union[float, str, Path, xr.DataArray, xr.Dataset, None] = None,
        reproj_method: str = "average",
    ) -> None:
        """Create bucket infiltration from final parameter maps.

        Adds model layers:

        * **bucket_smax** map: bucket maximum storage [mm]
        * **bucket_k** map: bucket drainage coefficient [hr-1]
        * **bucket_loss** map: bucket loss fraction [-]

        Parameters
        ----------
        bucket_smax, bucket_k : str, Path, or RasterDataset
            Raster data with final bucket parameters. Dataset inputs must contain
            variables named ``bucket_smax`` and ``bucket_k`` respectively.
        bucket_loss : float, str, Path, or RasterDataset, optional
            Uniform loss fraction or raster data with a ``bucket_loss`` variable.
            Defaults to 0.0.
        reproj_method : str, optional
            Resampling method for reprojecting the parameter maps to the model grid.
        """
        layers = {}
        for name, source in {
            "bucket_smax": bucket_smax,
            "bucket_k": bucket_k,
        }.items():
            da = self.data_catalog.get_rasterdataset(
                source, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da, xr.Dataset):
                if name not in da.data_vars:
                    raise ValueError(f"Could not find variable {name} in {source}")
                da = da[name]
            da = da.raster.mask_nodata()
            layers[name] = da.raster.reproject_like(self.mask, method=reproj_method)

        loss_value = 0.0 if bucket_loss is None else bucket_loss
        if bucket_loss is None or np.isscalar(bucket_loss):
            da_loss = xr.full_like(self.mask, np.float32(loss_value), dtype=np.float32)
            da_loss.name = "bucket_loss"
        else:
            da_loss = self.data_catalog.get_rasterdataset(
                bucket_loss, bbox=self.model.bbox, buffer=10
            )
            if isinstance(da_loss, xr.Dataset):
                if "bucket_loss" not in da_loss.data_vars:
                    raise ValueError(
                        f"Could not find variable bucket_loss in {bucket_loss}"
                    )
                da_loss = da_loss["bucket_loss"]
            da_loss = da_loss.raster.mask_nodata()
            da_loss = da_loss.raster.reproject_like(self.mask, method=reproj_method)
        layers["bucket_loss"] = da_loss
        self._set_layers(layers, flavor="bkt")

    @hydromt_step
    def create_bucket_from_soil(
        self,
        hsg: Union[str, Path, xr.DataArray, xr.Dataset],
        ksat: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        lulc: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        reclass_table: Union[str, Path, pd.DataFrame, None] = None,
        lulc_modifiers: Union[str, Path, pd.DataFrame, None] = None,
        dual_hsg: str = "drained",
        factor_ksat: float = 3.6,
        bucket_loss: Union[float, None] = None,
        reproj_method: str = "average",
    ) -> None:
        """Estimate bucket infiltration from HSG and optional landuse.

        Adds model layers:

        * **bucket_smax** map: bucket maximum storage [mm]
        * **bucket_k** map: bucket drainage coefficient [hr-1]
        * **bucket_loss** map: bucket loss fraction [-]

        Parameters
        ----------
        hsg : str, Path, or RasterDataset
            Hydrologic soil group map. By default, values are reclassified with
            the bundled ``hsg_bucket.csv`` table.
        ksat : str, Path, or RasterDataset, optional
            Saturated hydraulic conductivity map. If provided, it helps derive
            ``bucket_k`` values.
        lulc : str, Path, or RasterDataset, optional
            Land-use map used to apply infiltration modifiers. Its classes must
            match the index of ``lulc_modifiers``.
        reclass_table : str, Path, or DataFrame, optional
            Table mapping HSG classes to bucket parameters.
        lulc_modifiers : str, Path, or DataFrame, optional
            Table with land-use modifier factors, indexed by land-use class. By
            default the bundled NLCD table is used.
        dual_hsg : {None, 'native', 'drained'}, optional
            How to handle dual HSG classes, by default 'drained'.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr, by default 3.6.
        bucket_loss : float, optional
            Uniform bucket loss fraction. Defaults to 0.0 without land use and
            0.10 with land-use modifiers.
        reproj_method : str, optional
            Resampling method for reprojecting final parameter maps to the model grid.
            By default 'average'.
        """
        if reclass_table is None:
            reclass_table = Path(DATADIR) / "infiltration" / "hsg_bucket.csv"

        da_soil = self.data_catalog.get_rasterdataset(
            hsg,
            bbox=self.model.bbox,
            buffer=10,
        )
        df_map = self.data_catalog.get_dataframe(
            reclass_table,
            source_kwargs={"driver": {"name": "pandas", "options": {"index_col": 0}}},
        )

        da_ksat = None
        if ksat is not None:
            da_ksat = self.data_catalog.get_rasterdataset(
                ksat, bbox=self.model.bbox, buffer=10
            )
            da_ksat = da_ksat.raster.reproject_like(da_soil, method="average")

        if lulc is not None:
            if lulc_modifiers is None:
                lulc_modifiers = (
                    Path(DATADIR) / "infiltration" / "nlcd_infiltration_modifiers.csv"
                )
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc, bbox=self.model.bbox, buffer=10, variables=["lulc"]
            )
            da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
            df_modifiers = self.data_catalog.get_dataframe(
                lulc_modifiers,
                source_kwargs={
                    "driver": {"name": "pandas", "options": {"index_col": 0}}
                },
            )
            loss_value = 0.10 if bucket_loss is None else bucket_loss
            ds = workflows.bucket_from_soil_landuse(
                da_soil,
                da_lulc,
                df_map,
                df_modifiers,
                da_ksat=da_ksat,
                factor_ksat=factor_ksat,
                dual_hsg=dual_hsg,
                bucket_loss=loss_value,
            )
        else:
            ds = workflows.bucket_from_soil(
                da_soil,
                df_map,
                da_ksat=da_ksat,
                factor_ksat=factor_ksat,
                bucket_loss=bucket_loss if np.isscalar(bucket_loss) else None,
            )

        layers = {}
        for name in BUCKET_VARS:
            da = ds[name]
            try:
                da = da.raster.mask_nodata()
            except Exception:
                pass
            layers[name] = da.raster.reproject_like(self.mask, method=reproj_method)
        self._set_layers(layers, flavor="bkt")

    def clear(self) -> None:
        """Clear all infiltration layers from the model."""
        self.model.grid._data = clear_data(self.data, keep=())
        reset_config(self.model.config)
