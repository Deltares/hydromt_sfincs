import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs.components.quadtree.quadtree_mixin import SfincsQuadtreeMixin
from hydromt_sfincs import workflows

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_ATTRS = {
    "qinf": {
        "standard_name": "infiltration rate",
        "unit": "mm.hr-1",
        "infiltration_type": "c2d",
    },
    "scs": {
        "standard_name": "potential soil moisture retention",
        "unit": "in",
        "infiltration_type": "cna",
    },
    "smax": {
        "standard_name": "potential maximum soil moisture retention",
        "unit": "m",
        "infiltration_type": "cnb",
    },
    "seff": {
        "standard_name": "effective potential maximum soil moisture retention",
        "unit": "m",
        "infiltration_type": "cnb",
    },
    "ks": {
        "standard_name": "saturated hydraulic conductivity",
        "unit": "mm.hr-1",
        "infiltration_type": "cnb",
    },
}


class SfincsQuadtreeInfiltration(SfincsQuadtreeMixin, ModelComponent):
    """SFINCS infiltration component for quadtree grids. Note that the infiltration component
    can contain multiple variables, depending on the infiltration type. The variables are stored
    in the model.quadtree_grid.data with names specified in _ATTRS."""

    def __init__(
        self,
        model: "SfincsModel",
    ):
        super().__init__(
            model=model,
        )
        self.attrs = _ATTRS

    @property
    def data(self):
        """Get the data from the model quadtree grid."""
        return self.model.quadtree_grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model quadtree grid."""
        return self.model.quadtree_grid.data["mask"]

    def read(self):
        """Read infiltration data for quadtree grid."""
        pass

    def write(self):
        """Write infiltration files for quadtree grid."""
        pass

    def get_vars_by_infiltration_type(self, infiltration_type):
        # Variables that belong to the requested type
        vars_to_write = {
            var
            for var, attrs in self.attrs.items()
            if attrs.get("infiltration_type") == infiltration_type
        }

        # Variables that belong to other types
        vars_other_types = {
            var
            for var, attrs in self.attrs.items()
            if attrs.get("infiltration_type") != infiltration_type
        }

        # Only remove variables that actually exist in self.data
        vars_to_remove = vars_other_types.intersection(self.data.keys())

        return list(vars_to_write), list(vars_to_remove)

    # Function to create constant spatially varying infiltration
    @hydromt_step
    def create_constant(
        self,
        qinf=None,
        lulc=None,
        reclass_table=None,
        reproj_method="mean",
        nrmax=2000,
    ):
        """Setup spatially varying constant infiltration rate (qinffile) for quadtree grid.

        Adds model layers to SfincsModel.quadtree_grid.data:

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
            Method to sample from raster data to mesh. By default mean. Options include
            {"centroid", "barycentric", "mean", "harmonic_mean", "geometric_mean", "sum",
            "minimum", "maximum", "mode", "median", "max_overlap"}.

        See Also:
        ---------
        :py:meth:`~hydromt.model.processes.mesh.mesh2d_from_rasterdataset`
        """

        # Add logger info
        logger.info(
            "Creating constant spatially varying infiltration rate for quadtree grid."
        )

        # Get infiltration data
        if qinf is not None:
            da_inf = self.model.data_catalog.get_rasterdataset(
                qinf,
                bbox=self.model.bbox,
                buffer=10,
                variables=["qinf"],
            )
        elif lulc is not None:
            # landuse/landcover should always be combined with mapping
            if reclass_table is None:
                raise IOError(
                    f"Infiltration mapping file should be provided for {lulc}"
                )
            da_lulc = self.model.data_catalog.get_rasterdataset(
                lulc,
                bbox=self.model.bbox,
                buffer=10,
                variables=["lulc"],
            )
            df_map = self.model.data_catalog.get_dataframe(
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

        # set nodata to nan before reprojecting/interpolating
        da_inf = da_inf.raster.mask_nodata()

        n_cells = self.data.grid.n_face
        qinf = np.full(n_cells, np.nan)

        # Function to compute infiltration values for a chunk of the quadtree grid
        def compute_constant_infiltration(da_like, ilev=None):
            # reproject infiltration data to model grid
            da_out = da_inf.raster.reproject_like(da_like, method=reproj_method)
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

        # Convert constant qinf to ugrid-dataarray and set in self.data
        da = xr.DataArray(qinf, dims=[self.data.grid.face_dimension])
        uda = xu.UgridDataArray(da, self.data.grid)
        self.model.quadtree_grid.set(uda, name="qinf")

        # Update config: remove default inf and set qinf map
        self.model.config.update(
            {
                "infiltration_file": "infiltration.nc",
                "infiltration_type": "c2d",
                "qinf": None,
            }
        )

    # Function to create curve number for SFINCS quadtree
    @hydromt_step
    def create_cn(self, cn, antecedent_moisture="avg", reproj_method="med", nrmax=2000):
        """Setup model potential maximum soil moisture retention map (scsfile)
        from gridded curve number map for quadtree grid.

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
            f"Creating curve number values for SFINCS quadtree grid with antecedent moisture condition: {antecedent_moisture}."
        )

        # get data
        da_org = self.model.data_catalog.get_rasterdataset(
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

        n_cells = self.data.grid.n_face
        scs = np.full(n_cells, np.nan)

        def compute_cn_infiltration(da_like, ilev=None):
            # reproject using reproj method
            da_cn = da_org.raster.reproject_like(da_like, method=reproj_method)
            # convert to potential maximum soil moisture retention S (1000/CN - 10) [inch]
            da_scs = workflows.cn_to_s(da_cn).round(3)
            return da_scs

        # Compute constant infiltration in chunks over the quadtree grid
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

        # Convert curve number to ugrid-dataarray and set in self.data
        da = xr.DataArray(scs, dims=[self.data.grid.face_dimension])
        uda = xu.UgridDataArray(da, self.data.grid)
        self.model.quadtree_grid.set(uda, name="scs")

        # Update config: remove default inf and set scs map
        self.model.config.update(
            {
                "infiltration_file": "infiltration.nc",
                "infiltration_type": "cna",
                "qinf": None,
            }
        )

    # Function to create curve number for SFINCS including recovery via saturated hydraulic conductivity [mm/hr]
    @hydromt_step
    def create_cn_with_recovery(
        self, lulc, hsg, ksat, reclass_table, effective, nrmax=2000
    ):
        """Create curve number for SFINCS quadtree grid including recovery via saturated hydraulic conductivity
        including recovery term based on the soil saturation.

        Adds model layers:

        * **smax** map: maximum soil moisture storage capacity (potential storage capacity) in meters
        * **seff** map: soil moisture storage capacity at the start of the simulation (effective storage capacity) in meters
        * **ks** map: saturated hydraulic conductivity [mm/hr]

        Parameters
        ---------
        lulc : str, Path, or RasterDataset
            Landuse/landcover data set
        hsg : str, Path, or RasterDataset
            HSG (Hydrological Similarity Group) in integers
        ksat : str, Path, or RasterDataset
            Ksat (saturated hydraulic conductivity) [micrometer per second]
        reclass_table : str, Path, or RasterDataset
            reclass table to relate landcover with soiltype
        effective : float
            estimate of percentage effective soil, e.g. 0.50 for 50%
        nrmax : int
            maximum block size - use larger values will get more data in memory but can be faster, default=2000
        """

        # Add logger info
        logger.info(
            "Creating curve number values for SFINCS quadtree grid including recovery term."
        )

        # Read the datafiles
        da_landuse = self.model.data_catalog.get_rasterdataset(
            lulc, bbox=self.model.bbox, buffer=10
        )
        da_HSG = self.model.data_catalog.get_rasterdataset(
            hsg, bbox=self.model.bbox, buffer=10
        )
        da_Ksat = self.model.data_catalog.get_rasterdataset(
            ksat, bbox=self.model.bbox, buffer=10
        )
        df_map = self.model.data_catalog.get_dataframe(reclass_table)

        # Compute resolution land use (we are assuming that is the finest)
        resolution_landuse = np.mean(
            [abs(da_landuse.raster.res[0]), abs(da_landuse.raster.res[1])]
        )
        if da_landuse.raster.crs.is_geographic:
            resolution_landuse = (
                resolution_landuse * 111111.0
            )  # assume 1 degree is 111km

        logger.info("Processing curve number determination for quadtree grid")

        smax = np.full(self.data.grid.n_face, np.nan)
        ks = np.full(self.data.grid.n_face, np.nan)

        def compute_cn_recovery(da_like, ilev=None):
            da_smax, da_ks = workflows.curvenumber.scs_recovery_determination(
                da_landuse, da_HSG, da_Ksat, df_map, da_like
            )
            return da_smax, da_ks

        # Compute constant infiltration in chunks over the quadtree grid
        self.compute_quadtree(compute_cn_recovery, output=(smax, ks), nrmax=nrmax)

        # Done
        logger.info("Done with determination of curve number with recovery.")

        # check on nan values
        if np.logical_and(np.isnan(smax), self.mask >= 1).any():
            logger.warning(
                "NaN values found in storage capacity data; filled with 0 (impermeable)"
            )
            smax = np.where(np.isnan(smax), 0, smax)
        if np.logical_and(np.isnan(ks), self.mask >= 1).any():
            logger.warning(
                "NaN values found in hydraulic conductivity data; filled with 0 (no recovery)"
            )
            ks = np.where(np.isnan(ks), 0, ks)

        # Specify the effective soil retention (seff)
        seff = smax
        seff = seff * effective

        # Convert ks, smax, seff to ugrid-dataset and set in self.data
        da_smax = xr.DataArray(smax, dims=[self.data.grid.face_dimension])
        da_ks = xr.DataArray(ks, dims=[self.data.grid.face_dimension])
        da_seff = xr.DataArray(seff, dims=[self.data.grid.face_dimension])
        ds = xr.Dataset({"smax": da_smax, "ks": da_ks, "seff": da_seff})
        uds = xu.UgridDataset(ds, self.data.grid)
        self.model.quadtree_grid.set(uds)

        # Update config: remove default inf and set scs map
        self.model.config.update(
            {
                "infiltration_file": "infiltration.nc",
                "infiltration_type": "cnb",
                "qinf": None,
            }
        )
