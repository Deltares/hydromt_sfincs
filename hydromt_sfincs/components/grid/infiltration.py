import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import DATADIR, workflows
from hydromt_sfincs.utils import fill_nan_in_mask
from hydromt_sfincs.components.grid.regulargrid_mixin import SfincsRegularGridMixin
from hydromt_sfincs.components.infiltration_common import (
    ALL_VARS,
    BUCKET_VARS,
    VARIABLES,
    clear_data,
    configure,
    configured_flavor,
    flavor_variables,
    get_attrs,
    reset_config,
    _require_lulc_modifiers,
)

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsInfiltration(SfincsRegularGridMixin, ModelComponent):
    """SFINCS infiltration component for regular grids.

    Unsuffixed ``create_*`` methods estimate parameters from HSG, optional
    Ksat, and optional land-use modifiers. Methods with a ``_from_maps`` suffix
    use final SFINCS parameter maps directly.
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
        if flavor == "bkt":
            raise ValueError("Bucket infiltration is only supported on quadtree grids")
        self.clear()
        for name, da in layers.items():
            da = da.astype(np.float32)
            da = da.copy(
                data=fill_nan_in_mask(
                    da.values, self.mask.values, name, VARIABLES[name].fill_value
                )
            )
            da = da.where(self.mask > 0)
            da.name = name
            da.attrs.update(get_attrs(name))
            try:
                da.raster.set_crs(self.model.crs)
                da.raster.set_nodata(np.nan)
            except Exception:
                pass
            self.model.grid.set(da)
        configure(self.model.config, flavor=flavor, grid_type="regular")

    grid_variables = tuple(name for name in ALL_VARS if name not in BUCKET_VARS)

    def read(self) -> None:
        """Read the infiltration layers from their own files."""
        # the grid holds the mask and cell index these layers are written against;
        # check _data directly, since the data property already triggers a read
        if self.model.grid._data is None:
            self.model.grid.read(read_components=False)
        flavor = configured_flavor(self.model.config, grid_type="regular")
        if flavor is None or flavor == "con":
            return
        self.model.grid.read_layers(list(flavor_variables(flavor)))

    def write(self) -> None:
        """Write each regular-grid infiltration layer to its own binary file."""
        flavor = configured_flavor(self.model.config, grid_type="regular")
        if flavor is None or flavor == "con":
            return
        self.model.grid.write_layers(list(flavor_variables(flavor)))

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
        #     if lulc_modifier_table is None:
        #         lulc_modifier_table = (
        #             Path(DATADIR)
        #             / "infiltration"
        #             / "nlcd_infiltration_modifiers.csv"
        #         )
        #     if isinstance(lulc_modifier_table, pd.DataFrame):
        #         df_modifiers = lulc_modifier_table.copy()
        #     else:
        #         df_modifiers = self.data_catalog.get_dataframe(
        #             lulc_modifier_table,
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
            "smax": xr.full_like(self.mask, np.nan, dtype=np.float32),
            "seff": xr.full_like(self.mask, np.nan, dtype=np.float32),
            "ks": xr.full_like(self.mask, np.nan, dtype=np.float32),
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
        hsg: Union[str, Path, xr.DataArray, xr.Dataset],
        ksat: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        lulc: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        reclass_table: Union[str, Path, pd.DataFrame, None] = None,
        lulc_modifier_table: Union[str, Path, pd.DataFrame, None] = None,
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
            match the index of ``lulc_modifier_table``.
        reclass_table : str, Path, or DataFrame, optional
            Table mapping HSG classes to Green-Ampt parameters.
        lulc_modifier_table : str, Path, or DataFrame, optional
            Required with ``lulc``. Table of modifier factors keyed by the
            land-cover class codes in the supplied dataset.
        dual_hsg : {None, 'native', 'drained'}, optional
            How to handle dual HSG classes, by default 'drained'.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr, by default 3.6.
        reproj_method : str, optional
            Resampling method for reprojecting final parameter maps to the model grid.
            By default 'average'.
        """
        _require_lulc_modifiers(lulc, lulc_modifier_table)
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
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc,
                bbox=self.model.bbox,
                buffer=10,
            )
            da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
            df_modifiers = self.data_catalog.get_dataframe(
                lulc_modifier_table,
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
    def create_green_ampt_from_maps(
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
        psi, sigma, ks : str, Path, xr.DataArray, or xr.Dataset
            Raster data with final Green-Ampt parameters. Dataset inputs must
            contain variables named ``psi``, ``sigma``, and ``ks`` respectively.
        reproj_method : str, optional
            Resampling method for reprojecting the parameter maps to the model grid.
        """
        layers = {}
        for name, source in {"psi": psi, "sigma": sigma, "ks": ks}.items():
            da = self.data_catalog.get_rasterdataset(
                source,
                bbox=self.model.bbox,
                buffer=10,
                variables=[name],
            )
            da = da.raster.mask_nodata()
            layers[name] = da.raster.reproject_like(self.mask, method=reproj_method)
        self._set_layers(layers, flavor="gai")

    @hydromt_step
    def create_horton(
        self,
        hsg: Union[str, Path, xr.DataArray, xr.Dataset],
        ksat: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        lulc: Union[str, Path, xr.DataArray, xr.Dataset, None] = None,
        reclass_table: Union[str, Path, pd.DataFrame, None] = None,
        lulc_modifier_table: Union[str, Path, pd.DataFrame, None] = None,
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
            match the index of ``lulc_modifier_table``.
        reclass_table : str, Path, or DataFrame, optional
            Table mapping HSG classes to Horton parameters.
        lulc_modifier_table : str, Path, or DataFrame, optional
            Required with ``lulc``. Table of modifier factors keyed by the
            land-cover class codes in the supplied dataset.
        dual_hsg : {None, 'native', 'drained'}, optional
            How to handle dual HSG classes, by default 'drained'.
        factor_ksat : float, optional
            Factor used to convert Ksat units to mm/hr, by default 3.6.
        reproj_method : str, optional
            Resampling method for reprojecting final parameter maps to the model grid.
            By default 'average'.
        """
        _require_lulc_modifiers(lulc, lulc_modifier_table)
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
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc,
                bbox=self.model.bbox,
                buffer=10,
            )
            da_lulc = da_lulc.raster.reproject_like(da_soil, method="nearest")
            df_modifiers = self.data_catalog.get_dataframe(
                lulc_modifier_table,
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
    def create_horton_from_maps(
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
        f0, fc, kd : str, Path, xr.DataArray, or xr.Dataset
            Raster data with final Horton parameters. Dataset inputs must contain
            variables named ``f0``, ``fc``, and ``kd`` respectively.
        reproj_method : str, optional
            Resampling method for reprojecting the parameter maps to the model grid.
        """
        layers = {}
        for name, source in {"f0": f0, "fc": fc, "kd": kd}.items():
            da = self.data_catalog.get_rasterdataset(
                source,
                bbox=self.model.bbox,
                buffer=10,
                variables=[name],
            )
            da = da.raster.mask_nodata()
            layers[name] = da.raster.reproject_like(self.mask, method=reproj_method)
        self._set_layers(layers, flavor="hor")

    def clear(self) -> None:
        """Clear all infiltration layers from the model."""
        self.model.grid._data = clear_data(self.data, keep=())
        reset_config(self.model.config)
