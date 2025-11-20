import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import workflows

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {
    "qinf": {
        "standard_name": "infiltration rate",
        "unit": "mm.hr-1",
    },
    "scs": {
        "standard_name": "potential soil moisture retention",
        "unit": "in",
    },
    "smax": {
        "standard_name": "potential maximum soil moisture retention",
        "unit": "in",
    },
    "seff": {
        "standard_name": "effective potential maximum soil moisture retention",
        "unit": "in",
    },
    "ks": {
        "standard_name": "saturated hydraulic conductivity",
        "unit": "mm.hr-1",
    },
}


class SfincsInfiltration(ModelComponent):
    """SFINCS Infiltration Component.

    This component contains methods to add infiltration data to the SFINCS model
    on regular grids. Various infiltration parameterizations can be used, including
    spatially varying constant infiltration rates and curve number based methods.

    .. note::
        The infiltration data is stored in the model grid's data dataset under the keys "qinf", "scs", "smax", "seff" and "ks".

    See Also
    --------
    :py:class:`~hydromt_sfincs.components.grid.regulargrid.SfincsGrid`

    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.grid.data["qinf"]
        # (or "scs" or "smax", "seff" & "ks")
        super().__init__(
            model=model,
        )

    @property
    def data(self):
        """Get the data from the model grid."""
        return self.model.grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.grid.mask

    # %% core HydroMT-SFINCS functions:
    # read
    # write
    # set > already in grid.set()
    # create_constant
    # create_cn
    # create_cn_with_recovery
    # clear >TODO?

    def read(self):
        """Not implemented, infiltration data is read when the grid is read."""
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The infiltration file(s) are read when all grid files are read in regulargrid.py
        pass

    def write(self):
        """Not implemented, infiltration data is written when the grid is written."""
        # The infiltration file(s) are written when all grid files are written in regulargrid.py
        pass

    # Function to create constant spatially varying infiltration
    @hydromt_step
    def create_constant(
        self,
        qinf=None,
        lulc=None,
        reclass_table=None,
        reproj_method="average",
    ):
        """Create spatially varying constant infiltration rate (qinffile).

        Adds model layers to SfincsModel.grid.data:

        * **qinf** map: constant infiltration rate [mm/hr]

        Parameters
        ----------
        qinf : str, Path, or RasterDataset
            Spatially varying infiltration rates [mm/hr]
        lulc: str, Path, or RasterDataset
            Landuse/landcover data set
        reclass_table: str, Path, or pd.DataFrame
            Reclassification table to convert landuse/landcover to infiltration rates [mm/hr]
        reproj_method : str, optional
            Resampling method for reprojecting the infiltration data to the model grid.
            By default 'average'. For more information see, :py:meth:`hydromt.raster.RasterDataArray.reproject_like`
        """

        # Add logger info
        logger.info("Creating constant spatially varying infiltration rate.")

        # get infiltration data
        if qinf is not None:
            da_inf = self.data_catalog.get_rasterdataset(
                qinf,
                bbox=self.model.bbox,
                buffer=10,
            )
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
            da_inf = da_lulc.raster.reclassify(df_map)["qinf"]
        else:
            raise ValueError(
                "Either qinf or lulc must be provided when setting up constant infiltration."
            )

        # reproject infiltration data to model grid
        da_inf = da_inf.raster.mask_nodata()  # set nodata to nan
        da_inf = da_inf.raster.reproject_like(self.mask, method=reproj_method)

        # check on nan values
        if np.logical_and(np.isnan(da_inf), self.mask >= 1).any():
            logger.warning("NaN values found in infiltration data; filled with 0")
            da_inf = da_inf.fillna(0)
        da_inf.raster.set_nodata(-9999.0)

        # set grid
        mname = "qinf"
        da_inf.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_inf, name=mname)

        # update config: remove default inf and set qinf map
        self.model.config.set(f"{mname}file", f"sfincs.{mname}")
        # set spatially uniform qinf to None in config
        self.model.config.set("qinf", None)

        # loop over other infiltration methods ATTRS and remove them from config when present
        for name in _ATTRS.keys():
            if name != mname:
                # get from config
                if self.model.config.get(f"{name}file", None) is not None:
                    logger.info(f"Removing {name}file from model config.")
                    self.model.config.set(f"{name}file", None)

    # Function to create curve number for SFINCS
    @hydromt_step
    def create_cn(self, cn, antecedent_moisture="avg", reproj_method="med"):
        """Create model potential maximum soil moisture retention map (scsfile)
        from gridded curve number map.

        Adds model layers:

        * **scs** map: potential maximum soil moisture retention [inch]

        Parameters
        ---------
        cn: str, Path, or RasterDataset
            Name of gridded curve number map.

            * Required layers without antecedent runoff conditions: ['cn']
            * Required layers with antecedent runoff conditions: ['cn_dry', 'cn_avg', 'cn_wet']
        antecedent_moisture: {'dry', 'avg', 'wet'}, optional
            Antecedent runoff conditions.
            None if data has no antecedent runoff conditions.
            By default `avg`
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

        # set grid
        mname = "scs"
        da_scs.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_scs, name=mname)
        # update config:
        self.model.config.set(f"{mname}file", f"sfincs.{mname}")
        # set spatially unfiform qinf to None in config
        self.model.config.set("qinf", None)

        # loop over other infiltration methods ATTRS and remove them from config when present
        for name in _ATTRS.keys():
            if name != mname:
                # get from config
                if self.model.config.get(f"{name}file", None) is not None:
                    logger.info(f"Removing {name}file from model config.")
                    self.model.config.set(f"{name}file", None)

    # Function to create curve number for SFINCS including recovery via saturated hydraulic conductivity [mm/hr]
    @hydromt_step
    def create_cn_with_recovery(
        self, lulc, hsg, ksat, reclass_table, effective, factor_ksat=1, block_size=2000
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
        df_map = self.data_catalog.get_dataframe(reclass_table)

        # Define outputs
        da_smax = xr.full_like(self.mask, -9999, dtype=np.float32)
        da_ks = xr.full_like(self.mask, -9999, dtype=np.float32)

        # Compute resolution land use (we are assuming that is the finest)
        resolution_landuse = np.mean(
            [abs(da_landuse.raster.res[0]), abs(da_landuse.raster.res[1])]
        )
        if da_landuse.raster.crs.is_geographic:
            resolution_landuse = (
                resolution_landuse * 111111.0
            )  # assume 1 degree is 111km

        # Define the blocks
        nrmax = block_size
        nmax = np.shape(self.mask)[0]
        mmax = np.shape(self.mask)[1]
        refi = (
            self.model.config.get("dx") / resolution_landuse
        )  # finest resolution of landuse
        nrcb = int(np.floor(nrmax / refi))  # nr of regular cells in a block
        nrbn = int(np.ceil(nmax / nrcb))  # nr of blocks in n direction
        nrbm = int(np.ceil(mmax / nrcb))  # nr of blocks in m direction
        x_dim, y_dim = self.mask.raster.x_dim, self.mask.raster.y_dim

        # avoid blocks with width or height of 1
        merge_last_col = False
        merge_last_row = False
        if mmax % nrcb == 1:
            nrbm -= 1
            merge_last_col = True
        if nmax % nrcb == 1:
            nrbn -= 1
            merge_last_row = True

        ## Loop through blocks
        ib = -1
        for ii in range(nrbm):
            bm0 = ii * nrcb  # Index of first m in block
            bm1 = min(bm0 + nrcb, mmax)  # last m in block
            if merge_last_col and ii == (nrbm - 1):
                bm1 += 1

            for jj in range(nrbn):
                bn0 = jj * nrcb  # Index of first n in block
                bn1 = min(bn0 + nrcb, nmax)  # last n in block
                if merge_last_row and jj == (nrbn - 1):
                    bn1 += 1

                # Count
                ib += 1
                logger.debug(
                    f"\nblock {ib + 1}/{nrbn * nrbm} -- "
                    f"col {bm0}:{bm1 - 1} | row {bn0}:{bn1 - 1}"
                )

                # calculate transform and shape of block at cell and subgrid level
                da_mask_block = self.mask.isel(
                    {x_dim: slice(bm0, bm1), y_dim: slice(bn0, bn1)}
                ).load()

                # Call workflow
                (
                    da_smax_block,
                    da_ks_block,
                ) = workflows.curvenumber.scs_recovery_determination(
                    da_landuse, da_HSG, da_Ksat, df_map, da_mask_block
                )

                # New place in the overall matrix
                sn, sm = slice(bn0, bn1), slice(bm0, bm1)
                da_smax[sn, sm] = da_smax_block
                da_ks[sn, sm] = da_ks_block

        # Done
        logger.info("Done with determination of curve number values (in blocks).")

        # Convert ks - (e.g. from micrometer per second to mm/hr which is required in SFINCS)
        da_ks = da_ks * factor_ksat

        # Specify the effective soil retention (seff)
        da_seff = da_smax
        da_seff = da_seff * effective
        da_seff.raster.set_nodata(da_smax.raster.nodata)

        # set grids for seff, smax and ks (saturated hydraulic conductivity)
        names = ["smax", "seff", "ks"]
        data = [da_smax, da_seff, da_ks]
        for name, da in zip(names, data):
            # Give metadata to the layer and set grid
            da.attrs.update(**_ATTRS.get(name, {}))
            self.model.grid.set(da, name=name)

            # update config: set maps
            self.model.config.set(f"{name}file", f"sfincs.{name}")  # give it to SFINCS

        # set spatially unfiform qinf to None in config
        self.model.config.set("qinf", None)

        # loop over other infiltration methods ATTRS and remove them from config when present
        for name in _ATTRS.keys():
            if name not in names:
                # get from config
                if self.model.config.get(f"{name}file", None) is not None:
                    logger.info(f"Removing {name}file from model config.")
                    self.model.config.set(f"{name}file", None)
