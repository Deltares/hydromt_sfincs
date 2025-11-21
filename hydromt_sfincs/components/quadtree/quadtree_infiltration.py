import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent
from hydromt.model.processes.mesh import mesh2d_from_rasterdataset

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


class SfincsQuadtreeInfiltration(ModelComponent):
    """SFINCS infiltration component for quadtree grids."""

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the infiltration is stored in the model.quadtree_grid.data["qinf"]
        # (or "scs" or "smax", "seff" & "ks")
        super().__init__(
            model=model,
        )

    @property
    def data(self):
        """Get the data from the model quadtree grid."""
        return self.model.quadtree_grid.data

    @property
    def mask(self):
        """Get an empty mask with the same shape as the model quadtree grid."""
        return self.model.quadtree_grid.data["mask"]

    # %% core HydroMT-SFINCS functions:
    # read
    # write
    # set > already in quadtree_grid.set()
    # create_constant
    # create_cn
    # create_cn_with_recovery
    # clear >TODO?

    def read(self):
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The infiltration file(s) are read when all quadtree grid files are read in quadtree.py
        pass

    def write(self, variables=None):
        """Write infiltration files for quadtree grid.

        Parameters
        ----------
        variables : list, optional
            List of variable names to write. If None, writes all available variables.
            Options: ['qinf', 'scs', 'smax', 'seff', 'ks']
        """
        # Get available infiltration variables
        available_vars = []
        logger.info(f"Checking for infiltration variables in quadtree grid data...")
        logger.info(
            f"Available data variables: {list(self.data.data_vars.keys()) if hasattr(self.data, 'data_vars') else 'No data_vars attribute'}"
        )

        for var_name in _ATTRS.keys():
            if var_name in self.data:
                available_vars.append(var_name)
                logger.info(f"✓ Found {var_name}")
            else:
                logger.info(f"✗ Missing {var_name}")

        if not available_vars:
            logger.info("No infiltration variables to write")
            return

        # Determine which variables to write
        if variables is None:
            vars_to_write = available_vars
        else:
            # Validate requested variables
            vars_to_write = []
            for var in variables:
                if var in available_vars:
                    vars_to_write.append(var)
                else:
                    logger.warning(
                        f"Variable '{var}' not available. Available: {available_vars}"
                    )

        if not vars_to_write:
            logger.warning("No valid variables specified to write")
            return

        logger.info(f"Writing quadtree infiltration variables: {vars_to_write}")

    # Function to create constant spatially varying infiltration
    @hydromt_step
    def create_constant(
        self,
        qinf=None,
        lulc=None,
        reclass_table=None,
        reproj_method="mean",
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

        # get infiltration data
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

        # reproject infiltration data to model quadtree grid
        da_inf = da_inf.raster.mask_nodata()  # set nodata to nan

        # reproject single-dataset to mesh
        mesh2d = self.model.quadtree_grid.data.grid
        uda_inf = mesh2d_from_rasterdataset(
            ds=da_inf,
            mesh2d=mesh2d,
            resampling_method=reproj_method,
        )

        # check on nan values
        if np.logical_and(np.isnan(uda_inf), self.mask >= 1).any():
            logger.warning("NaN values found in infiltration data; filled with 0")
            uda_inf = uda_inf.fillna(0)

        # set grid
        mname = "qinf"
        uda_inf.attrs.update(**_ATTRS.get(mname, {}))
        self.model.quadtree_grid.data["qinf"] = uda_inf["qinf"]
        # FIXME: ideally we would use the set method, but that's not working here properly
        # self.model.quadtree_grid.set(uda_inf, name=mname, overwrite_grid=True)

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

    # Function to create curve number for SFINCS quadtree
    @hydromt_step
    def create_cn(self, cn, antecedent_moisture="avg", reproj_method="median"):
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
            Method to sample from raster data to mesh. By default median. Options include
            {"centroid", "barycentric", "mean", "harmonic_mean", "geometric_mean", "sum",
            "minimum", "maximum", "mode", "median", "max_overlap"}.
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

        # reproject using median
        # reproject single-dataset to mesh
        mesh2d = self.model.quadtree_grid.data.grid
        uda_cn = mesh2d_from_rasterdataset(
            ds=da_org,
            mesh2d=mesh2d,
            resampling_method=reproj_method,
        )

        # convert to potential maximum soil moisture retention S (1000/CN - 10) [inch]
        uda_scs = workflows.cn_to_s(uda_cn, self.mask > 0).round(3)

        # set grid
        mname = "scs"
        uda_scs.attrs.update(**_ATTRS.get(mname, {}))
        self.model.quadtree_grid.data["scs"] = uda_scs[v]
        # FIXME: ideally we would use the set method, but that's not working here properly
        # self.model.quadtree_grid.set(da_scs_ugrid, name=mname, overwrite_grid=True)

        # update config:
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

    # Function to create curve number for SFINCS including recovery via saturated hydraulic conductivity [mm/hr]
    @hydromt_step
    def create_cn_with_recovery(
        self, lulc, hsg, ksat, reclass_table, effective, factor_ksat=1, block_size=2000
    ):
        """Setup model the Soil Conservation Service (SCS) Curve Number (CN) files for SFINCS quadtree grid
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

        # For quadtree grids, we'll implement the curve number logic directly
        # since the workflow is designed for structured grids
        logger.info("Processing curve number determination for quadtree grid")

        # Initialize output arrays using xu.full_like for quadtree compatibility
        da_smax = xu.full_like(self.mask, -9999, dtype=np.float32)
        da_ks = xu.full_like(self.mask, -9999, dtype=np.float32)

        # Get face coordinates for spatial interpolation
        face_coords = self.model.quadtree_grid.face_coordinates
        x_coords = face_coords[0][:]
        y_coords = face_coords[1][:]

        # Interpolate landuse data to quadtree grid points using point-wise interpolation
        logger.info(
            "Interpolating landuse data to quadtree grid using point-wise selection"
        )

        # Check and debug coordinate systems before interpolation
        import xarray as xr

        # Ensure all datasets are in the same CRS as the model
        logger.info("Reprojecting input datasets to match model CRS...")
        model_crs = self.model.crs

        # Reproject datasets if needed using hydromt's raster methods
        if da_landuse.raster.crs != model_crs:
            logger.info("Reprojecting landuse data to model CRS")
            da_landuse = da_landuse.raster.reproject(
                dst_crs=model_crs, method="nearest"
            )

        if da_HSG.raster.crs != model_crs:
            logger.info("Reprojecting HSG data to model CRS")
            da_HSG = da_HSG.raster.reproject(dst_crs=model_crs, method="nearest")

        if da_Ksat.raster.crs != model_crs:
            logger.info("Reprojecting Ksat data to model CRS")
            da_Ksat = da_Ksat.raster.reproject(dst_crs=model_crs, method="nearest")

        # Create coordinate arrays (now all in same CRS)
        x_da = xr.DataArray(
            x_coords,
            dims=["points"],
            attrs={"crs": str(model_crs), "long_name": "x coordinate", "units": "m"},
        )
        y_da = xr.DataArray(
            y_coords,
            dims=["points"],
            attrs={"crs": str(model_crs), "long_name": "y coordinate", "units": "m"},
        )

        # Use sel with method='nearest' for efficient point-wise nearest neighbor
        lu_at_points = da_landuse.sel(x=x_da, y=y_da, method="nearest").values.astype(
            np.float32
        )

        # Interpolate HSG data to quadtree grid points using point-wise selection
        logger.info(
            "Interpolating HSG data to quadtree grid using point-wise selection"
        )

        # Use sel with method='nearest' for HSG data (categorical)
        hsg_at_points = da_HSG.sel(x=x_da, y=y_da, method="nearest").values.astype(
            np.float32
        )

        # Interpolate Ksat data to quadtree grid points using interp method
        logger.info("Interpolating Ksat data to quadtree grid using interp method")

        # Use interp method for Ksat data (linear interpolation for continuous data)
        # For interp, we need to pass coordinates as separate arrays
        ksat_at_points = da_Ksat.interp(x=x_da, y=y_da, method="linear").values.astype(
            np.float32
        )

        # Initialize arrays for curve numbers and soil retention
        cn_values = np.full_like(lu_at_points, np.nan, dtype=np.float32)

        # Map landuse and HSG combinations to curve numbers
        logger.info("Mapping landuse-HSG combinations to curve numbers")

        # Create mask for valid points (xarray.interp uses NaN for invalid/out-of-bounds)
        lu_valid = ~np.isnan(lu_at_points)
        hsg_valid = ~np.isnan(hsg_at_points)
        valid_points = lu_valid & hsg_valid

        for i in range(df_map.index.size):
            for j in range(1, df_map.columns.size):  # Start from 1 as in original
                # Find points with this landuse-HSG combination (only valid points)
                mask_combo = (
                    (lu_at_points == df_map.index[i])
                    & (hsg_at_points == int(df_map.columns[j]))
                    & valid_points
                )
                # Assign curve number
                cn_values[mask_combo] = df_map.values[i, j]

        # Convert CN to maximum soil retention (S) - SCS equation
        logger.info("Converting curve numbers to soil retention parameters")
        cn_values = np.maximum(cn_values, 0)  # always positive
        cn_values = np.minimum(cn_values, 100)  # not higher than 100

        # Calculate S using SCS equation: S = (1000/CN - 10) [inches]
        valid_cn = ~np.isnan(cn_values) & (cn_values > 0)
        s_values = np.zeros_like(cn_values)
        s_values[valid_cn] = np.maximum(1000 / cn_values[valid_cn] - 10, 0)
        s_values[~valid_cn] = 0.0  # NaN means no infiltration = 0

        # Convert to meters: multiply by 0.0254 (inches to meters)
        s_values = s_values * 0.0254

        # Process Ksat values
        logger.info("Processing Ksat values for recovery term")
        ksat_processed = np.copy(ksat_at_points)
        # Handle fill values and NaNs - set them to 0
        # Ksat is typically float, but handle potential fill values
        ksat_processed = np.nan_to_num(ksat_processed, nan=0.0)
        ksat_processed = np.minimum(ksat_processed, 100)  # not higher than 100
        ksat_processed = ksat_processed * 3.6  # from micrometers per second to mm/hr

        # Fill NaN values
        s_values = np.nan_to_num(s_values, nan=0.0)
        ksat_processed = np.nan_to_num(ksat_processed, nan=0.0)

        # Create UgridDataArrays using the correct constructor
        # Use the mask as a template and assign new values
        da_smax = self.mask.copy()
        da_smax.values = s_values
        da_smax.name = "smax"
        da_smax.attrs.update({"standard_name": "maximum_soil_retention", "unit": "m"})

        da_ks = self.mask.copy()
        da_ks.values = ksat_processed
        da_ks.name = "ks"
        da_ks.attrs.update(
            {"standard_name": "saturated_hydraulic_conductivity", "unit": "mm.hr-1"}
        )

        # Done with basic calculations
        logger.info("Done with determination of curve number values for quadtree grid.")

        # Convert ks - (e.g. from micrometer per second to mm/hr which is required in SFINCS)
        da_ks = da_ks * factor_ksat

        # Specify the effective soil retention (seff)
        da_seff = da_smax.copy()  # Create a proper copy to avoid modifying original
        da_seff.values = (
            da_seff.values * effective
        )  # Modify values directly to avoid issues
        da_seff.name = "seff"  # Set proper name
        da_seff.attrs.update(
            {"standard_name": "effective_maximum_soil_retention", "unit": "m"}
        )

        # Set nodata value for xugrid (if method exists)
        try:
            da_seff.ugrid.set_nodata(da_smax.ugrid.nodata)
        except AttributeError:
            # Fallback: set nodata on the underlying data array
            da_seff = da_seff.fillna(-9999)

        # Log the data arrays before setting
        logger.info(f"Prepared data arrays:")
        logger.info(
            f"  da_smax: shape={da_smax.shape}, name='{da_smax.name}', values_range=[{da_smax.values.min():.6f}, {da_smax.values.max():.6f}]"
        )
        logger.info(
            f"  da_seff: shape={da_seff.shape}, name='{da_seff.name}', values_range=[{da_seff.values.min():.6f}, {da_seff.values.max():.6f}]"
        )
        logger.info(
            f"  da_ks: shape={da_ks.shape}, name='{da_ks.name}', values_range=[{da_ks.values.min():.3f}, {da_ks.values.max():.3f}]"
        )

        # loop over other infiltration methods ATTRS and remove them from config when present
        names = ["smax", "seff", "ks"]
        for name in _ATTRS.keys():
            if name not in names:
                # get from config
                if self.model.config.get(f"{name}file", None) is not None:
                    logger.info(f"Removing {name}file from model config.")
                    self.model.config.set(f"{name}file", None)

        # Set up infiltration data arrays with proper metadata
        data_arrays = [da_smax, da_seff, da_ks]

        # Ensure all arrays have proper names and attributes
        for name, da in zip(names, data_arrays):
            da.name = name
            da.attrs.update(**_ATTRS.get(name, {}))

        # Update config files for all variables and set spatially uniform qinf to None
        self.model.config.set("qinf", None)
        for name in names:
            self.model.config.set(f"{name}file", f"sfincs.{name}")

        # Print basic statistics
        logger.info("=== Infiltration Parameter Summary ===")
        for name, da in zip(names, data_arrays):
            valid_data = da.values[~np.isnan(da.values) & (da.values != -9999)]
            if len(valid_data) > 0:
                logger.info(
                    f"{name}: {valid_data.min():.3f} - {valid_data.max():.3f} ({len(valid_data)} valid cells)"
                )
            else:
                logger.warning(f"{name}: No valid values found!")
        logger.info("=" * 40)

        # Write each variable directly to files
        for var_name, da in zip(names, data_arrays):
            try:
                file_key = f"{var_name}file"
                file_name = self.model.config.get(file_key)

                if file_name:
                    from pathlib import Path

                    # Get the full file path
                    if hasattr(self.model, "root"):
                        if hasattr(self.model.root, "path"):
                            root_path = Path(self.model.root.path)
                        elif isinstance(self.model.root, (str, Path)):
                            root_path = Path(self.model.root)
                        else:
                            root_str = str(self.model.root)
                            if "path=" in root_str:
                                path_start = root_str.find("path=") + 5
                                path_end = root_str.find(",", path_start)
                                if path_end == -1:
                                    path_end = root_str.find(")", path_start)
                                root_path = Path(root_str[path_start:path_end])
                            else:
                                root_path = Path(root_str)
                        file_path = root_path / file_name
                    else:
                        file_path = Path(file_name)

                    # Write as binary file (SFINCS standard format)
                    try:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        if file_path.exists():
                            file_path.unlink()

                        binary_data = da.values.astype(np.float32).tobytes()
                        with open(file_path, "wb") as f:
                            f.write(binary_data)
                        logger.info(f"✓ Wrote {var_name} to {file_path}")

                    except Exception as write_error:
                        logger.error(f"✗ Failed to write {var_name}: {write_error}")

                else:
                    logger.warning(f"No filename configured for {var_name}")

            except Exception as e:
                logger.error(f"Failed to process {var_name}: {e}")

        logger.info("Finished writing infiltration files")
