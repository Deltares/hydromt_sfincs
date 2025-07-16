"""RegularGrid class for SFINCS."""

import logging
import math
import os
import glob
from os.path import abspath, basename, dirname, isabs, isfile, join
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import xarray as xr
import geopandas as gpd
from affine import Affine
from pyproj import CRS, Transformer
from scipy import ndimage
from shapely.geometry import LineString

from pyflwdir.regions import region_area

from hydromt.model.components import GridComponent
from hydromt.model.processes.grid import create_grid_from_region

from hydromt_sfincs import workflows, utils
from hydromt_sfincs.subgrid import SubgridTableRegular
from hydromt_sfincs.workflows.tiling import int2png, tile_window

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")

_MAPS = ["mask", "dep", "scs", "manning", "qinf", "smax", "seff", "ks", "vol"]
_ATTRS = {
    "dep": {"standard_name": "elevation", "unit": "m+ref"},
    "mask": {"standard_name": "mask", "unit": "-"},
    "scs": {
        "standard_name": "potential maximum soil moisture retention",
        "unit": "in",
    },
    "qinf": {"standard_name": "infiltration rate", "unit": "mm.hr-1"},
    "manning": {"standard_name": "manning roughness", "unit": "s.m-1/3"},
    "vol": {"standard_name": "storage volume", "unit": "m3"},
}


class SfincsGrid(GridComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        super().__init__(
            model=model,
            filename="sfincs.nc",
            region_filename="region.geojson",
        )

        # # set spatial attributes
        # self.update_grid_from_config()

    @property
    def transform(self):
        """Return the affine transform of the regular grid."""
        transform = (
            Affine.translation(self.x0, self.y0)
            * Affine.rotation(self.rotation)
            * Affine.scale(self.dx, self.dy)
        )
        return transform

    @property
    def coordinates(self, x_dim="x", y_dim="y"):
        """Return the coordinates of the cell-centers the regular grid."""
        if self.transform.b == 0:
            x_coords, _ = self.transform * (
                np.arange(self.mmax) + 0.5,
                np.zeros(self.mmax) + 0.5,
            )
            _, y_coords = self.transform * (
                np.zeros(self.nmax) + 0.5,
                np.arange(self.nmax) + 0.5,
            )
            coords = {
                y_dim: (y_dim, y_coords),
                x_dim: (x_dim, x_coords),
            }
        else:
            x_coords, y_coords = (
                self.transform
                * self.transform.translation(0.5, 0.5)
                * np.meshgrid(np.arange(self.mmax), np.arange(self.nmax))
            )
            coords = {
                "yc": ((y_dim, x_dim), y_coords),
                "xc": ((y_dim, x_dim), x_coords),
            }
        return coords

    @property
    def edges(self):
        """Return the coordinates of the cell-edges the regular grid."""
        x_edges, y_edges = (
            self.transform
            * self.transform.translation(0, 0)
            * np.meshgrid(np.arange(self.mmax + 1), np.arange(self.nmax + 1))
        )
        return x_edges, y_edges

    @property
    def empty_mask(self) -> xr.DataArray:
        """Return mask with only inactive cells"""
        da_mask = xr.DataArray(
            name="msk",
            data=np.zeros((self.nmax, self.mmax), dtype=np.uint8),
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": 0},
        )
        da_mask.raster.set_crs(self.model.crs)
        return da_mask

    @property
    def crs(self) -> CRS:
        """Return the coordinate reference system of the regular grid."""
        if self.epsg is not None:
            return CRS.from_epsg(self.epsg)
        elif self.data.raster.crs is not None:
            return self.data.raster.crs
        else:
            raise ValueError("No CRS defined for the regular grid.")

    @property
    def mask(self) -> xr.DataArray:
        """Return the mask of the regular grid."""
        if "mask" in self.data:
            da_mask = self.data["mask"]
        else:
            da_mask = self.empty_mask
        return da_mask

    @property
    def region(self) -> gpd.GeoDataFrame:
        """Return the active region of the regular grid."""
        if "mask" in self.data and np.any(self.data["mask"] > 0):
            da = xr.where(self.data["mask"] > 0, 1, 0).astype(np.int16)
            da.raster.set_nodata(0)
            return da.raster.vectorize().dissolve()
        elif self.data is not None:
            return self.empty_mask.raster.box

    def read(self, data_vars: Union[List, str] = None) -> None:
        """Read SFINCS binary grid files and save to `data` attribute.
        Filenames are taken from the `model.config` attribute (i.e. input file).

        Parameters
        ----------
        data_vars : Union[List, str], optional
            List of data variables to read, by default None (all)
        """
        # check if in read mode and initialize grid
        self.root._assert_read_mode()
        self._initialize_grid(skip_read=True)

        # first update grid from config
        self.update_grid_from_config()

        # now read in the actual files
        da_lst = []
        if data_vars is None:
            data_vars = _MAPS
            provide_warnings = (
                False  # all possible variables are read, no warnings needed
            )
        elif isinstance(data_vars, str):
            data_vars = list(data_vars)
            provide_warnings = True  # specific variables are asked, so provide warnings

        # read index file
        ind_fn = self.model.config.get(
            "indexfile", fallback="sfincs.ind", abs_path=True
        )
        if not isfile(ind_fn):
            raise IOError(f".ind path {ind_fn} does not exist")

        dtypes = {"msk": "u1"}
        mvs = {"msk": 0}
        ind = self.read_ind(ind_fn=ind_fn)

        for name in data_vars:
            fn = self.model.config.get(
                f"{name}file", fallback=f"sfincs.{name}", abs_path=True
            )
            if not isfile(fn):
                if provide_warnings:
                    logger.warning(f"{name}file not found at {fn}")
                continue
            dtype = dtypes.get(name, "f4")
            mv = mvs.get(name, -9999.0)
            da = self.read_map(fn, ind, dtype, mv, name=name)
            da_lst.append(da)
        ds = xr.merge(da_lst)
        epsg = self.model.config.get("epsg", None)
        if epsg is not None:
            ds.raster.set_crs(epsg)
        self.set(ds)

        # # TODO - fix this properly; but to create overlays in GUIs,
        # # we always convert regular grids to a UgridDataArray
        # self.quadtree = QuadtreeGrid(logger=logger)
        # if self.config.get("rotation", 0) != 0:  # This is a rotated regular grid
        #     self.quadtree.data = UgridDataArray.from_structured(
        #         self.mask, "xc", "yc"
        #     )
        # else:
        #     self.quadtree.data = UgridDataArray.from_structured(self.mask)
        # self.quadtree.data.grid.set_crs(self.crs)

        # keep some metadata maps from gis directory

        # fns = glob.glob(join(self.root, "gis", "*.tif"))
        # fns = [
        #     fn
        #     for fn in fns
        #     if basename(fn).split(".")[0] not in self.grid.data_vars
        # ]
        # if fns:
        #     ds = hydromt.open_mfraster(fns).load()
        #     self.set_grid(ds)
        #     ds.close()

    def write(
        self,
        data_vars: Union[List, str] = None,
    ) -> None:
        """Write SFINCS grid to binary files including map index file.
        Filenames are taken from the `config` attribute (i.e. input file).

        If `write_gis` property is True, all grid variables are written to geotiff
        files in a "gis" subfolder.

        Parameters
        ----------
        data_vars : Union[List, str], optional
            List of data variables to write, by default None (all)
        """
        self.root._assert_write_mode

        dtypes = {"mask": "u1"}  # default to f4
        if len(self.data.data_vars) > 0 and "mask" in self.data:
            # make sure orientation is S->N
            ds_out = self.data
            if ds_out.raster.res[1] < 0:
                ds_out = ds_out.raster.flipud()
            mask = ds_out["mask"].values

            logger.debug("Write binary map indices based on mask.")
            if self.model.config.get("indexfile") is None:
                self.model.config.set("indexfile", "sfincs.ind")
            self.write_ind(
                ind_fn=self.model.config.get("indexfile", abs_path=True), mask=mask
            )

            if data_vars is None:  # write all maps
                data_vars = [v for v in _MAPS if v in ds_out]
            elif isinstance(data_vars, str):
                data_vars = list(data_vars)
                # always rewrite the mask
                data_vars.append("mask") if "mask" not in data_vars else data_vars

            logger.debug(f"Write binary map files: {data_vars}.")
            for name in data_vars:
                # Set file name and get absolute path
                if name == "mask":
                    abs_file_path = self.model.config.get_set_file_variable(
                        "mskfile", "sfincs.msk"
                    )
                else:
                    abs_file_path = self.model.config.get_set_file_variable(
                        f"{name}file",
                        f"sfincs.{name}",
                    )

                # write to binary model files
                self.write_map(
                    map_fn=abs_file_path,
                    data=ds_out[name].values,
                    mask=mask,
                    dtype=dtypes.get(name, "f4"),
                )

                # write to gis-files for visualization
                if self.model.write_gis:
                    utils.write_raster(
                        ds_out[name],
                        root=join(self.model.root.path, "gis"),
                        mask=mask,
                        logger=logger,
                    )

    def create(
        self,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nmax: int,
        mmax: int,
        rotation: float,
        epsg: int,
    ):
        """Setup a regular or quadtree grid.

        Parameters
        ----------
        x0, y0 : float
            x,y coordinates of the origin of the grid
        dx, dy : float
            grid cell size in x and y direction
        mmax, nmax : int
            number of grid cells in x and y direction
        rotation : float, optional
            rotation of grid [degree angle], by default None
        epsg : int, optional
            epsg-code of the coordinate reference system
        """

        # update the grid attributes in the model config
        self.model.config.update(
            {
                "x0": x0,
                "y0": y0,
                "dx": dx,
                "dy": dy,
                "nmax": nmax,
                "mmax": mmax,
                "rotation": rotation,
                "epsg": epsg,
            }
        )
        self.update_grid_from_config()

        # initialize a grid without variables
        ds = xr.Dataset(
            coords=self.coordinates,
        )
        ds.raster.set_crs(self.model.crs)

        # set the grid in the model data
        self.set(ds)

    def create_from_region(
        self,
        region: dict,
        res: float = 100,
        crs: Union[str, int] = "utm",
        rotated: bool = False,
        hydrography_fn: str = None,
        basin_index_fn: str = None,
        align: bool = False,
        dec_origin: int = 0,
        dec_rotation: int = 3,
    ):
        """Setup a regular or quadtree grid from a region.

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g.:

            * {'bbox': [xmin, ymin, xmax, ymax]}
            * {'geom': 'path/to/polygon_geometry'}

            Note: For the 'bbox' option the coordinates need to be provided in WG84/EPSG:4326.

            For a complete overview of all region options,
            see :py:func:`hydromt.workflows.basin_mask.parse_region`
        res : float, optional
            grid resolution, by default 100 m
        crs : Union[str, int], optional
            coordinate reference system of the grid
            if "utm" (default) the best UTM zone is selected
            else a pyproj crs string or epsg code (int) can be provided
        grid_type : str, optional
            grid type, "regular" (default) or "quadtree"
        rotated : bool, optional
            if True, a minimum rotated rectangular grid is fitted around the region, by default False
        hydrography_fn : str
            Name of data source for hydrography data.
        basin_index_fn : str
            Name of data source with basin (bounding box) geometries associated with
            the 'basins' layer of `hydrography_fn`. Only required if the `region` is
            based on a (sub)(inter)basins without a 'bounds' argument.
        align : bool, optional
            If True (default), align target transform to resolution.
            Note that this has only been implemented for non-rotated grids.
        dec_origin : int, optional
            number of decimals to round the origin coordinates, by default 0
        dec_rotation : int, optional
            number of decimals to round the rotation angle, by default 3

        See Also
        --------
        hydromt.model.processes.create_grid_from_region
        """

        ds = create_grid_from_region(
            region=region,
            data_catalog=self.model.data_catalog,
            res=res,
            crs=crs,
            region_crs=4326,
            rotated=rotated,
            hydrography_path=hydrography_fn,
            basin_index_path=basin_index_fn,
            add_mask=False,
            align=align,
            dec_origin=dec_origin,
            dec_rotation=dec_rotation,
        )

        # add the grid to the model
        self.set(ds)

        # update the grid attributes in the model config
        self.update_config_from_grid()

    def create_dep(
        self,
        datasets_dep: List[dict],
        buffer_cells: int = 0,  # not in list
        interp_method: str = "linear",  # used for buffer cells only
    ):
        """Interpolate topobathy (dep) data to the model grid.

        Adds model grid layers:

        * **dep**: combined elevation/bathymetry [m+ref]

        Parameters
        ----------
        datasets_dep : List[dict]
            List of dictionaries with topobathy data, each containing a dataset name or Path (elevtn) and optional merge arguments e.g.:
            [{'elevtn': merit_hydro, 'zmin': 0.01}, {'elevtn': gebco, 'offset': 0, 'merge_method': 'first', 'reproj_method': 'bilinear'}]
            For a complete overview of all merge options, see :py:func:`hydromt.workflows.merge_multi_dataarrays`
        buffer_cells : int, optional
            Number of cells between datasets to ensure smooth transition of bed levels, by default 0
        interp_method : str, optional
            Interpolation method used to fill the buffer cells , by default "linear"
        """

        # retrieve model resolution to determine zoom level for xyz-datasets
        if not self.model.grid.crs.is_geographic:
            res = np.abs(self.mask.raster.res[0])
        else:
            res = np.abs(self.mask.raster.res[0]) * 111111.0

        datasets_dep = self.model._parse_datasets_dep(datasets_dep, res=res)

        da_dep = workflows.merge_multi_dataarrays(
            da_list=datasets_dep,
            da_like=self.mask,
            buffer_cells=buffer_cells,
            interp_method=interp_method,
            logger=logger,
        )

        # check if no nan data is present in the bed levels
        nmissing = int(np.sum(np.isnan(da_dep.values)))
        if nmissing > 0:
            logger.warning(f"Interpolate elevation at {nmissing} cells")
            da_dep = da_dep.raster.interpolate_na(method="rio_idw", extrapolate=True)

        # set the dep layer in the model data
        mname = "dep"
        da_dep.attrs.update(**_ATTRS.get(mname, {}))
        self.set(da_dep, name=mname)

        # TODO add to config, or is that only done when writing?
        self.model.config.set("depfile", "sfincs.dep")

    # Roughness
    def create_manning_roughness(
        self,
        datasets_rgh: List[dict] = [],
        manning_land=0.04,
        manning_sea=0.02,
        rgh_lev_land=0,
    ):
        """Setup model manning roughness map (manningfile) from gridded manning data or a combinataion of gridded
        land-use/land-cover map and manning roughness mapping table.

        Adds model layers:

        * **man** map: manning roughness coefficient [s.m-1/3]

        Parameters
        ---------
        datasets_rgh : List[dict], optional
            List of dictionaries with Manning's n datasets. Each dictionary should at least contain one of the following:
            * (1) manning: filename (or Path) of gridded data with manning values
            * (2) lulc (and reclass_table) :a combination of a filename of gridded landuse/landcover and a mapping table.
            In additon, optional merge arguments can be provided e.g.: merge_method, gdf_valid_fn
        manning_land, manning_sea : float, optional
            Constant manning roughness values for land and sea, by default 0.04 and 0.02 s.m-1/3
            Note that these values are only used when no Manning's n datasets are provided, or to fill the nodata values
        rgh_lev_land : float, optional
            Elevation level to distinguish land and sea roughness (when using manning_land and manning_sea), by default 0.0
        """

        if len(datasets_rgh) > 0:
            datasets_rgh = self.model._parse_datasets_rgh(datasets_rgh)
        else:
            datasets_rgh = []

        # fromdep keeps track of whether any manning values should be based on the depth or not
        fromdep = len(datasets_rgh) == 0
        if len(datasets_rgh) > 0:
            da_man = workflows.merge_multi_dataarrays(
                da_list=datasets_rgh,
                da_like=self.mask,
                interp_method="linear",
                logger=logger,
            )
            fromdep = np.isnan(da_man).where(self.mask > 0, False).any()
        if "dep" in self.data and fromdep:
            da_man0 = xr.where(
                self.data["dep"] >= rgh_lev_land, manning_land, manning_sea
            )
        elif fromdep:
            da_man0 = xr.full_like(self.mask, manning_land, dtype=np.float32)

        if len(datasets_rgh) > 0 and fromdep:
            logger.warning("nan values in manning roughness array")
            da_man = da_man.where(~np.isnan(da_man), da_man0)
        elif fromdep:
            da_man = da_man0
        da_man.raster.set_nodata(-9999.0)

        # set grid
        mname = "manning"
        da_man.attrs.update(**_ATTRS.get(mname, {}))
        self.set(da_man, name=mname)
        # set file name in config
        self.model.config.set(f"{mname}file", f"sfincs.{mname[:3]}")

    # Function to create constant spatially varying infiltration
    def create_constant_infiltration(
        self,
        qinf=None,
        lulc=None,
        reclass_table=None,
        reproj_method="average",
    ):
        """Setup spatially varying constant infiltration rate (qinffile).

        Adds model layers:

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
            )
            # TODO set index col to 0
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
            self.logger.warning("NaN values found in infiltration data; filled with 0")
            da_inf = da_inf.fillna(0)
        da_inf.raster.set_nodata(-9999.0)

        # set grid
        mname = "qinf"
        da_inf.attrs.update(**_ATTRS.get(mname, {}))
        self.set(da_inf, name=mname)

        # update config: remove default inf and set qinf map
        self.model.config.set(f"{mname}file", f"sfincs.{mname}")
        # FIXME remove default or other infiltration methods?

    # Function to create curve number for SFINCS
    def create_cn_infiltration(
        self, cn, antecedent_moisture="avg", reproj_method="med"
    ):
        """Setup model potential maximum soil moisture retention map (scsfile)
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
        self.set(da_scs, name=mname)
        # update config:
        # FIXME remove default infiltration values and set scs map??
        self.model.config.set(f"{mname}file", f"sfincs.{mname}")

    # Function to create curve number for SFINCS including recovery via saturated hydraulic conductivity [mm/hr]
    def create_cn_infiltration_with_ks(
        self, lulc, hsg, ksat, reclass_table, effective, factor_ksat=1, block_size=2000
    ):
        """Setup model the Soil Conservation Service (SCS) Curve Number (CN) files for SFINCS
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
        df_map = self.data_catalog.get_dataframe(reclass_table, index_col=0)

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
        logger.info("Done with determination of values (in blocks).")

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
            self.set_grid(da, name=name)

            # update config: set maps
            self.model.config.set(f"{name}file", f"sfincs.{name}")  # give it to SFINCS

        # Remove qinf variable in sfincs
        # FIXME should we remove other infiltration variables?

    # %% supporting HydroMT-SFINCS functions:
    # other:
    # - ind
    # - read_ind
    # - read_map
    # - write_ind
    # - write_map
    # - to_vector_lines

    def ind(self, mask: np.ndarray) -> np.ndarray:
        """Return indices of active cells in mask."""
        assert mask.shape == (self.nmax, self.mmax)
        ind = np.where(mask.ravel(order="F"))[0]
        return ind

    def read_ind(
        self,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> np.ndarray:
        """Read indices of active cells in mask from binary file."""
        _ind = np.fromfile(ind_fn, dtype="u4")
        ind = _ind[1:] - 1  # convert to zero based index
        assert _ind[0] == ind.size

        return ind

    def read_map(
        self,
        map_fn: Union[str, Path],
        ind: np.ndarray,
        dtype: Union[str, np.dtype] = "f4",
        mv: float = -9999.0,
        name: str = None,
    ) -> xr.DataArray:
        """Read one of the grid variables of the SFINCS model map from a binary file."""

        data = np.full((self.mmax, self.nmax), mv, dtype=dtype)
        data.flat[ind] = np.fromfile(map_fn, dtype=dtype)
        data = data.transpose()

        da = xr.DataArray(
            name=map_fn.split(".")[-1] if name is None else name,
            data=data,
            coords=self.coordinates,
            dims=("y", "x"),
            attrs={"_FillValue": mv},
        )
        return da

    def write_ind(
        self,
        mask: np.ndarray,
        ind_fn: Union[str, Path] = "sfincs.ind",
    ) -> None:
        """Write indices of active cells in mask to binary file."""
        assert mask.shape == (self.nmax, self.mmax)
        # Add 1 because indices in SFINCS start with 1, not 0
        ind = self.ind(mask)
        indices_ = np.array(np.hstack([np.array(len(ind)), ind + 1]), dtype="u4")
        indices_.tofile(ind_fn)

    def write_map(
        self,
        map_fn: Union[str, Path],
        data: np.ndarray,
        mask: np.ndarray,
        dtype: Union[str, np.dtype] = "f4",
    ) -> None:
        """Write one of the grid variables of the SFINCS model map to a binary file."""

        data_out = np.asarray(data.transpose()[mask.transpose() > 0], dtype=dtype)
        data_out.tofile(map_fn)

    def update_grid_from_config(self):
        """Update grid properties based on `config` (sfincs.inp) attributes"""

        # assert model.config exists
        if not hasattr(self.model, "config"):
            raise AttributeError("Model has no config attribute")

        self.x0 = self.model.config.get("x0")
        self.y0 = self.model.config.get("y0")
        self.dx = self.model.config.get("dx")
        self.dy = self.model.config.get("dy")
        self.nmax = self.model.config.get("nmax")
        self.mmax = self.model.config.get("mmax")
        self.rotation = self.model.config.get("rotation", 0)
        self.epsg = self.model.config.get("epsg", None)

    def update_config_from_grid(self):
        """Update `config` (sfincs.inp) attributes based on grid properties"""

        # derive grid properties from grid
        self.nmax, self.mmax = self.data.raster.shape
        self.dx, self.dy = self.data.raster.res
        self.x0, self.y0 = self.data.raster.origin
        self.rotation = self.data.raster.rotation
        self.epsg = self.data.raster.crs.to_epsg()

        # update the grid properties in the config
        self.model.config.update(
            {
                "x0": self.x0,
                "y0": self.y0,
                "dx": self.dx,
                "dy": self.dy,
                "nmax": self.nmax,
                "mmax": self.mmax,
                "rotation": self.rotation,
                "epsg": self.epsg,
            }
        )

    def to_vector_lines(self):
        """Return a geopandas GeoDataFrame with a geometry for each grid line."""

        x, y = self.edges

        # create vertical lines
        vertical_lines = []
        for i in range(self.nmax + 1):
            line = LineString([(x[i, 0], y[i, 0]), (x[i, -1], y[i, -1])])
            vertical_lines.append(line)

        # create horizontal lines
        horizontal_lines = []
        for j in range(self.mmax + 1):
            line = LineString([(x[0, j], y[0, j]), (x[-1, j], y[-1, j])])
            horizontal_lines.append(line)

        # combine lines into a single list
        grid_lines = vertical_lines + horizontal_lines

        return gpd.GeoDataFrame(geometry=grid_lines, crs=self.model.crs)

    # %% DDB GUI focused additional functions:
    # create_index_tiles > FIXME - TL: still needed?
    # map_overlay
    # snap_to_grid
    # _get_datashader_dataframe

    # TODO - missing as in cht_sfincs:
    # Many...

    def create_index_tiles(
        self,
        root: Union[str, Path],
        region: gpd.GeoDataFrame,
        zoom_range: Union[int, List[int]] = [0, 13],
        fmt: str = "bin",
        logger: logging.Logger = logger,
    ):
        """Create index tiles for a region. Index tiles are used to quickly map webmercator tiles to the corresponding SFINCS cell.

        Parameters
        ----------
        region : gpd.GeoDataFrame
            GeoDataFrame containing the region of interest
        root : Union[str, Path]
            Directory where index tiles are stored
        zoom_range : Union[int, List[int]], optional
            Range of zoom levels for which tiles are created, by default [0,13]
        fmt : str, optional
            Format of index tiles, either "bin" (binary, default) or "png"
        """

        index_path = os.path.join(root, "indices")
        npix = 256

        # for binary format, use .dat extension
        if fmt == "bin":
            extension = "dat"
        # for net, tif and png extension and format are the same
        else:
            extension = fmt

        # if only one zoom level is specified, create tiles up to that zoom level (inclusive)
        if isinstance(zoom_range, int):
            zoom_range = [0, zoom_range]

        # get bounding box of sfincs model
        minx, miny, maxx, maxy = region.total_bounds
        transformer = Transformer.from_crs(region.crs.to_epsg(), 3857)

        # rotation of the model
        cosrot = math.cos(-self.rotation * math.pi / 180)
        sinrot = math.sin(-self.rotation * math.pi / 180)

        # axis order is different for geographic and projected CRS
        if region.crs.is_geographic:
            minx, miny = map(
                max, zip(transformer.transform(miny, minx), [-20037508.34] * 2)
            )
            maxx, maxy = map(
                min, zip(transformer.transform(maxy, maxx), [20037508.34] * 2)
            )
        else:
            minx, miny = map(
                max, zip(transformer.transform(minx, miny), [-20037508.34] * 2)
            )
            maxx, maxy = map(
                min, zip(transformer.transform(maxx, maxy), [20037508.34] * 2)
            )

        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            logger.debug("Processing zoom level " + str(izoom))

            zoom_path = os.path.join(index_path, str(izoom))

            for transform, col, row in tile_window(izoom, minx, miny, maxx, maxy):
                # transform is a rasterio Affine object
                # col, row are the tile indices
                file_name = os.path.join(
                    zoom_path, str(col), str(row) + "." + extension
                )

                # get the coordinates of the tile in webmercator projection
                x = np.arange(0, npix) + 0.5
                y = np.arange(0, npix) + 0.5
                x3857, y3857 = transform * (x, y)
                x3857, y3857 = np.meshgrid(x3857, y3857)

                # convert to SFINCS coordinates
                x, y = transformer.transform(x3857, y3857, direction="INVERSE")

                # Now rotate around origin of SFINCS model
                x00 = x - self.x0
                y00 = y - self.y0
                xg = x00 * cosrot - y00 * sinrot
                yg = x00 * sinrot + y00 * cosrot

                # determine the SFINCS cell indices
                iind = np.floor(xg / self.dx).astype(int)
                jind = np.floor(yg / self.dy).astype(int)
                ind = iind * self.nmax + jind
                ind[iind < 0] = -999
                ind[jind < 0] = -999
                ind[iind >= self.mmax] = -999
                ind[jind >= self.nmax] = -999

                # only write tiles that link to at least one SFINCS cell
                if np.any(ind >= 0):
                    if not os.path.exists(os.path.join(zoom_path, str(col))):
                        os.makedirs(os.path.join(zoom_path, str(col)))
                    # And write indices to file
                    if fmt == "bin":
                        fid = open(file_name, "wb")
                        fid.write(ind)
                        fid.close()
                    elif fmt == "png":
                        # for png, change nodata -999 nodata into 0
                        ind[ind == -999] = 0
                        int2png(ind, file_name)
