# -*- coding: utf-8 -*-
import os
from os.path import join, isfile, abspath, dirname, basename
import glob
import numpy as np
import logging
from numpy.core.fromnumeric import var
import rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import xarray as xr
import pyproj
from typing import Dict, Tuple, List

import hydromt
from hydromt.models.model_api import Model
from hydromt.vector import GeoDataArray
from hydromt.raster import RasterDataset, RasterDataArray
from xarray.core import variable

from . import workflows, utils, plots, DATADIR


__all__ = ["SfincsModel"]

logger = logging.getLogger(__name__)


class SfincsModel(Model):
    _NAME = "sfincs"
    _GEOMS = {
        "waterlevel": "bnd",
        "discharge": "src",
        "gauges": "obs",
        "weirs": "weir",
        "thin_dams": "thd",
    }
    _1DFORCING = {
        "waterlevel": "bzs",
        "discharge": "dis",
        "precip": "precip",
    }
    _2DFORCING = {
        "precip": "netampr",
    }
    _MAPS = {
        "elevtn": "dep",
        "mask": "msk",
        "curve_number": "scs",
        "manning": "manning",
        "infiltration": "qinf",
    }
    _FOLDERS = [""]  # to create root folder without subfolders
    _CONF = "sfincs.inp"
    _DATADIR = DATADIR
    _ATTRS = {
        "dep": {"standard_name": "elevation", "unit": "m"},
        "msk": {"standard_name": "mask", "unit": "-"},
        "scs": {"standard_name": "curve number", "unit": "-"},
        "qinf": {"standard_name": "infiltration rate", "unit": "mm.hr-1"},
        "manning": {"standard_name": "manning roughness", "unit": "s.m-1/3"},
        "bzs": {"standard_name": "waterlevel", "unit": "m+EGM96"},
        "dis": {"standard_name": "discharge", "unit": "m3.s-1"},
        "netampr": {"standard_name": "precipitation", "unit": "mm.hr-1"},
        "precip": {"standard_name": "precipitation", "unit": "mm.hr-1"},
    }

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn="sfincs.inp",
        write_gis=True,
        data_libs=None,
        deltares_data=None,
        artifact_data=None,
        logger=logger,
    ):
        """
        The SFINCS model class (SfincsModel) contains methods to read, write, setup and edit
        `SFINCS <https://sfincs.readthedocs.io/en/latest/>`_ models.

        Parameters
        ----------
        root: str, Path, optional
            Path to model folder
        mode: {'w', 'r+', 'r'}
            Open model in write, append or reading mode, by default 'w'
        config_fn: str, Path, optional
            Filename of model config file, by default "sfincs.inp"
        write_gis: bool
            Write model files additionally to geotiff and geojson, by default True
        opt: dict
            Values for model setup options, used when called from CLI
        sources: dict
            Library with references to data sources
        """
        # model paths
        self._write_gis = write_gis
        if write_gis:
            self._FOLDERS.append("gis")

        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            deltares_data=deltares_data,
            artifact_data=artifact_data,
            logger=logger,
        )

    def setup_basemaps(
        self,
        region,
        res=100,
        crs="utm",
        basemaps_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
        landmask_fn="osm_landareas",
        bathymetry_fn="gebco",
        mdt_fn="dtu10mdt_egm96",
        merge_buffer=0,
        mask_fn=None,
        dep_contour=-1,
        method="bilinear",
    ):
        """Define model region and setup combined bathymetry/elevation and mask static model layers.

        To merge elevation and bathymetry data, elevation data is taken for land and
        bathymetry for sea cells, see ``landmask_fn``. An optional ``merge_buffer``
        can be set to define a buffer aourd the land cells where the bathymetry is
        estimated from a linear interpolation of elevation and bathymetry data.

        The model mask defines 0) Inactive, 1) active, and 2) waterlevel boundary cells.
        Active cells set based on valid bathymetry/elevation values: a) within ``mask_fn``,
        b) within ``region`` for land and above the ``dep_contour`` for sea cells, or
        c) within ``region``. Water level boundary cells defined by sea cells adjecent
        to active cells.

        Adds model layers:

        * **dep** map: combined elevation/bathymetry [m+ref]
        * **mask** map: mask [-]
        * **region** geom: polygon of active model region
        * **bbox** geom: polygon bounding box of active model region


        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}
            See :py:meth:`~hydromt.cli.parse_region()` for all options
        res : float
            Model resolution [m], by default 100 m.
        crs : str, int
            Model Coordinate Reference System as epsg code, by default 'utm' in which
            case the region centroid UTM zone is used.
        basemaps_fn : str
            Path or data source name for hydrography raster data, by default 'merit_hydro'.

            * Required variables: ['elevtn'].
            * Required variables to delineate a (sub)basin: ['flwdir', 'uparea', 'basins']
        landmask_fn : str, optional
            Path or data source name for land mask polygon, by default 'osm_landareas'.
            Valid elevation cells within this geometry define land cells, all other cells
            are sea cells.
        bathymetry_fn : str, optional
            Path or data source name for bathymetry raster data, by default 'gebco'.
            Bathymetry data with values smaller than zero are used to fill the area
            outside the land mask.

            * Required variables: ['elevtn']
        mdt_fn : str, optional
            Path or data source name for MDT data, by default 'dtu10mdt_egm96'.
            Difference between vertical reference of elevation and bathymetry data.
            MDT values are Adds to the bathymetry data before merging.

            * Required variables: ['mdt']
        merge_buffer : int, optional
            Buffer (number of cells) around land cells where the bathymetry is estimated from a linear
            interpolation between the elevation and bathymetry data sources, by default 0.
        mask_fn : str, optional
            Path or data source name for polygon of active cells, by default None.
            Region of valid cells in the
        dep_contour: float, optional,
            Minimum depth threshold for active model cells. Must be negative, by default -1.
        method: str, optional
            Method used to reproject elevation and bathymetry data, by default 'bilinear'


        """
        # read data (lazy!) and return dataset
        ds_org = self.data_catalog.get_rasterdataset(
            basemaps_fn, single_var_as_array=False
        )
        # get basin geometry and clip data
        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        basin_geom = None
        geom = region.get("geom", None)
        bbox = region.get("bbox", None)
        if kind == "interbasin":
            bas_index = self.data_catalog[basin_index_fn]
            basin_geom = hydromt.workflows.get_basin_geometry(
                ds=ds_org,
                kind=kind,
                basin_index=bas_index,
                logger=self.logger,
                **region,
            )[0]
            if bbox is None and geom is None:
                geom = basin_geom
        elif kind not in ["bbox", "geom"]:
            raise ValueError(
                "Unknown region kind for SFINCS model: {kind}, "
                "select from ['bbox', 'geom', 'interbasin']"
            )
        if geom is not None and geom.crs is None:
            raise ValueError('SFINCS region "geom" has no CRS')
        # parse dst_crs. if 'utm' the best utm zone is calculated based on bbox_epsg4326
        bbox_epsg4326 = bbox if bbox is not None else geom.to_crs(4326).total_bounds
        dst_crs = hydromt.gis_utils.parse_crs(crs, bbox_epsg4326)
        # transfrom bbox/geom to geom with destination CRS to deal with nonlinear
        # transformations along domain edges when clipping
        if geom is not None:
            # to epsg required to be understood when writing GEOJSON
            dst_geom = geom.to_crs(dst_crs.to_epsg())
        else:
            dst_bbox = transform_bounds(pyproj.CRS.from_epsg(4326), dst_crs, *bbox)
            dst_geom = gpd.GeoDataFrame(geometry=[box(*dst_bbox)], crs=dst_crs)
        # reproject to destination CRS and clip to actual extent
        da_elv = ds_org["elevtn"]
        if da_elv.raster.crs != dst_crs:
            da_elv = (
                da_elv.raster.clip_geom(geom=dst_geom, buffer=5)
                .load()
                .raster.reproject(
                    dst_res=res, dst_crs=dst_crs, align=True, method=method
                )
                .raster.clip_geom(dst_geom, align=res)
                .raster.mask_nodata()  # force nodata value to be np.nan
                .round(3)  # mm precision
            )
        # make sure orientation is S -> N
        if da_elv.raster.res[1] < 0:
            da_elv = da_elv.reindex(
                {da_elv.raster.y_dim: list(reversed(da_elv.raster.ycoords))}
            )

        # read and rasterize land geometry
        # land mask contains all valid dep values inside land geometry
        if landmask_fn is not None:
            gdf_mask = self.data_catalog.get_geodataframe(
                landmask_fn, geom=dst_geom
            ).to_crs(da_elv.raster.crs)
            lnd_msk = da_elv.raster.geometry_mask(gdf_mask, all_touched=True)
            da_elv = da_elv.where(lnd_msk)
        lnd_msk = np.isfinite(da_elv)
        lnd_msk.raster.set_nodata(0)

        # TODO generalize and move merge to workflows
        # merge bathymetry if bathymetry_fn where lnd_msk is True
        if bathymetry_fn is not None and np.any(~lnd_msk):
            da_bat = (
                self.data_catalog.get_rasterdataset(
                    bathymetry_fn, geom=dst_geom, buffer=2, variables=["elevtn"]
                )
                .load()
                .raster.reproject_like(da_elv, method=method)
                .raster.mask_nodata()
            )
            da_bat = da_bat.where(da_bat < 0)  # use values below MSL only
            # correct vertical datum with mean dynamic topography data
            if mdt_fn is not None:
                da_mdt = (
                    self.data_catalog.get_rasterdataset(
                        mdt_fn, geom=dst_geom, buffer=2, variables=["mdt"]
                    )
                    .raster.reproject_like(da_elv, method=method)
                    .raster.mask_nodata()
                    .fillna(0)
                )
                mmdt = np.mean(da_mdt.values[~lnd_msk.values])
                self.logger.debug(f"Mean vertical corection from MDT: {mmdt:.3f}")
                da_bat = da_bat + da_mdt
            # merge based on lnd_msk
            da_elv = da_elv.where(lnd_msk, da_bat)
            # create buffer between data sources and interpolate for smooth boundary
            if merge_buffer > 0:
                lnd_msk_np = lnd_msk.values
                buf_dilate = ndimage.binary_dilation(
                    lnd_msk_np, structure=np.ones((3, 3)), iterations=merge_buffer
                )
                buf = xr.Variable(da_elv.dims, np.logical_xor(buf_dilate, lnd_msk_np))
                da_elv = da_elv.where(buf != 1, da_elv.raster.nodata)
            # interpolate missing in data
            nempty = np.sum(np.isnan(da_elv))
            if nempty > 0:
                self.logger.debug(f"Interpolate bathymetry at {int(nempty)} cells")
                da_elv = da_elv.raster.interpolate_na(method="linear")
            self.logger.info(f"Bathymetry data merged; buffer {merge_buffer} cells")
        da_elv = da_elv.fillna(-9999)
        da_elv.raster.set_nodata(-9999)
        self.logger.info(f"Bathymetry generated")

        # generate mask; valid dep cells and basin_geom should minimally be included
        if basin_geom is not None:
            bas_msk = da_elv.raster.geometry_mask(basin_geom, all_touched=True)
            lnd_msk = np.logical_or(lnd_msk, bas_msk)
        else:
            bas_msk = xr.full_like(da_elv, True, bool)
        msk = lnd_msk.values
        if mask_fn:
            # active cells: valid dep values inside polygon AND basin
            gdf_mask = self.data_catalog.get_geodataframe(mask_fn, geom=dst_geom)
            valid_msk = da_elv.raster.geometry_mask(gdf_mask, all_touched=True)
            msk = np.logical_and(valid_msk, da_elv != da_elv.raster.nodata).values
        elif dep_contour is not None:
            # active cells: contiguous area above depth threshold offshore OR in basin on land
            _msk = np.where(lnd_msk, bas_msk, da_elv >= dep_contour)
            # keep contiguous area with lnd_msk cells
            lbls, nlbls = ndimage.label(_msk, np.ones((3, 3)))
            labs = np.arange(1, nlbls + 1)
            lbl_valid = labs[ndimage.measurements.sum(lnd_msk, lbls, labs) > 0]
            # fill holes
            msk = ndimage.binary_fill_holes(np.isin(lbls, lbl_valid))

        # create mask: 0) inactive; 1) active 2) h boundary cells
        mask = msk.astype(np.int8)
        # apply dilation to mask to get boundary around mask
        if np.any(np.logical_and(~msk, ~lnd_msk)):
            _msk = ndimage.binary_dilation(msk, structure=np.ones((3, 3)))
        else:
            _msk = ndimage.binary_erosion(msk, structure=np.ones((3, 3)))
        # TODO: bnd@sea based on lnd_msk, include other options for non-coastal models
        bnd = np.logical_and(np.logical_xor(_msk, msk), ~lnd_msk)
        mask[bnd] = 2
        da_mask = xr.DataArray(
            dims=da_elv.raster.dims, coords=da_elv.raster.coords, data=mask
        )
        da_mask.raster.set_nodata(0)

        # set staticmap
        da_elv = da_elv.where(da_mask > 0, da_elv.raster.nodata)
        da_elv.attrs.update(**self._ATTRS.get(self._MAPS["elevtn"], {}))
        self.set_staticmaps(data=da_elv, name=self._MAPS["elevtn"])
        da_mask.attrs.update(**self._ATTRS.get(self._MAPS["mask"], {}))
        self.set_staticmaps(data=da_mask, name=self._MAPS["mask"])
        self.update_spatial_attrs()  # update geospatial header in config

        # update mask staticgeom
        da_mask = (da_mask != 0).astype(np.int16)
        da_mask.raster.set_crs(dst_crs)
        da_mask.raster.set_nodata(0)
        self.set_staticgeoms(da_mask.raster.vectorize(), "region")
        self.set_staticgeoms(da_mask.raster.box, "bbox")

        # log n active cells
        ncells = np.sum(msk == 1)
        self.logger.info(f"Mask generated; {ncells:d} active cells.")

    def setup_river_inflow(self, river_upa=25.0, basemaps_fn="merit_hydro"):
        """Setup river inflow (source) points where a river enters the model domain.

        NOTE: to ensure a river only enters the model domain once, use the 'basin',
        subbasin, or outlet region options.

        Adds model layers:

        * **src** geoms: discharge boundary point locations
        * **dis** forcing: dummy discharge timeseries
        * **river** geoms: river centerline (not used by SFINCS; for plotting only)

        Parameters
        ----------
        basemaps_f: str, Path
            Path or data source name for hydrography raster data, by default 'merit_hydro'.

            * Required layers: ['uparea', 'flwdir'].
        river_upa: float, optional
            Mimimum upstream area threshold [km2] to define river cells, by default 25 km2.
        """
        # TODO reproject flwdir, then burn rivers in D4
        # read data and rasterize basin mask > used to initialize flwdir in river workflow
        ds = self.data_catalog.get_rasterdataset(
            basemaps_fn, geom=self.region, buffer=2
        )
        src_crs = ds.raster.crs.to_epsg()
        dst_crs = self.crs.to_epsg()
        basmsk = ds.raster.geometry_mask(self.region)

        self.logger.debug(
            f"Get river cells for inflow source points; upstream area threshold: {river_upa} km2."
        )
        # initialize flwdir with river cells only (including outside basin)
        rivmsk = ds["uparea"] >= river_upa
        ds["mask"] = rivmsk
        flwdir = hydromt.flw.flwdir_from_da(ds["flwdir"], mask=True)

        self.logger.debug(f"Set river source points (.src).")
        # set source points at headwater indices on river
        # find river cells outside model domain
        idxs0 = np.where(np.logical_and(rivmsk, ~basmsk).values.ravel())[0]
        # snap to first downsteam cell in model domain
        idxs_source = flwdir.snap(idxs=idxs0, mask=basmsk)[0]
        idxs_source = np.unique(idxs_source[basmsk.values.flat[idxs_source]])
        if len(idxs_source) > 0:
            source_xy = flwdir.xy(idxs_source)
            gdf_src = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(*source_xy), crs=src_crs
            ).to_crs(dst_crs)
            gdf_src["uparea"] = ds["uparea"].values.flat[idxs_source]
            self.logger.debug(f"{len(idxs_source)} river inflow point locations set.")
            self.set_staticgeoms(gdf_src, name=self._GEOMS["discharge"])
            # set dummy timeseries to keep valid sfincs model
            da = self._dummy_ts(gdf_src, name="discharge", fill_value=0)
            self._update_forcing_1d(da, name="discharge")

        # vectorize river
        self.logger.debug(f"Vectorize river.")
        feats = flwdir.vectorize(mask=np.logical_and(basmsk, rivmsk).values)
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs)
        gdf_riv = gdf_riv.to_crs(dst_crs)
        gdf_riv.index = gdf_riv.index.values + 1  # one based index
        self.set_staticgeoms(gdf_riv, name="rivers")

    def setup_river_outflow(
        self, river_upa=25.0, outflow_width=2000, basemaps_fn="merit_hydro"
    ):
        """Setup river outflow boundary (msk=3) where a river flows out of the model domain.

        Adds / edits model layers:

        * **msk** map: edited by adding outflow points (msk=3)
        * **river_out** geoms: river centerline (not used by SFINCS; for plotting only)

        Parameters
        ----------
        basemaps_f: str, Path
            Path or data source name for hydrography raster data, by default 'merit_hydro'.

            * Required layers: ['uparea', 'flwdir'].
        outflow_width: int, optional
            The width [m] of the outflow boundary in the SFINCS msk file.
            By default 2km, i.e.: 1km to each side of the outflow point.
        river_upa: float, optional
            Mimimum upstream area threshold [km2] to define river cells, by default 25 km2.
        """
        # read data and rasterize basin mask > used to initialize flwdir in river workflow
        ds = self.data_catalog.get_rasterdataset(
            basemaps_fn, geom=self.region, buffer=2
        )
        src_crs = ds.raster.crs.to_epsg()
        dst_crs = self.crs.to_epsg()
        basmsk = ds.raster.geometry_mask(self.region)

        self.logger.debug(
            f"Get river cells for outflow points; upstream area threshold: {river_upa} km2."
        )
        # initialize flwdir with river cells only (including outside basin)
        rivmsk = ds["uparea"] >= river_upa
        ds["mask"] = np.logical_and(ds["uparea"] >= river_upa, basmsk)
        flwdir = hydromt.flw.flwdir_from_da(ds["flwdir"], mask=True)

        # git pits at domain edge on flwdir grid
        idxs0 = flwdir.idxs_pit
        basmsk_eroded = ndimage.binary_erosion(basmsk, structure=np.ones((3, 3)))
        edge_basmask = np.logical_xor(basmsk_eroded, basmsk)
        idxs_outflw = np.unique(idxs0[edge_basmask.values.flat[idxs0]])

        if len(idxs_outflw) > 0:
            self.logger.debug(f"Set msk=3 outflow points.")
            da_mask = self.staticmaps[self._MAPS["mask"]]
            gdf_outflw = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(*flwdir.xy(idxs_outflw)), crs=src_crs
            ).to_crs(dst_crs)
            # apply buffer
            gdf_outflw_buf = gpd.GeoDataFrame(
                geometry=gdf_outflw.buffer(outflow_width / 2.0), crs=gdf_outflw.crs
            )
            # find intersect of buffer and model grid
            msk = da_mask.values
            msk_eroded = ndimage.binary_erosion(msk, structure=np.ones((3, 3)))
            edge_model = np.logical_xor(msk_eroded, msk)
            outflw_buf = da_mask.raster.geometry_mask(gdf_outflw_buf).values
            outflw_msk = np.logical_and(outflw_buf, edge_model)
            # assign mask == 3 boundary and update staticmaps
            msk[outflw_msk] = 3
            da_mask.data = msk
            self.set_staticmaps(da_mask)
            self.logger.debug(f"{len(idxs_outflw)} river outflow point locations set.")

        # vectorize river
        self.logger.debug(f"Vectorize rivers used for setting outflow points.")
        feats = flwdir.vectorize(mask=np.logical_and(basmsk, rivmsk).values)
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs)
        gdf_riv.index = gdf_riv.index.values + 1  # one based index
        gdf_riv = gdf_riv.to_crs(dst_crs)
        self.set_staticgeoms(gdf_riv, name="rivers_out")

    def setup_cn_infiltration(self, cn_fn="gcn250", antecedent_runoff_conditions="avg"):
        """Setup model curve number map from gridded curve number map.

        Adds model layers:

        * **scs** map: curve number for infiltration

        Parameters
        ---------
        cn_fn: str, optional
            Name of gridded curve number map.

            * Required layers without antecedent runoff conditions: ['cn']
            * Required layers with antecedent runoff conditions: ['cn_dry', 'cn_avg', 'cn_wet']
        antecedent_runoff_conditions: {'dry', 'avg', 'wet'}, optional
            Antecedent runoff conditions.
            None if data has no atecedent runoff conditions.
            By default `avg`
        """
        # get data
        v = "cn"
        if antecedent_runoff_conditions:
            v = f"cn_{antecedent_runoff_conditions}"
        da_org = self.data_catalog.get_rasterdataset(
            cn_fn, geom=self.region, buffer=10, variables=[v]
        )
        da_msk = self.staticmaps[self._MAPS["mask"]] > 0
        # reproject using median
        # force nodata value to be 100 (zero infiltration)
        da_scs = da_org.raster.reproject_like(da_msk, method="med")
        da_scs = da_scs.raster.mask_nodata().fillna(100)
        # mask and set nodata, precision
        da_scs = da_scs.where(da_msk, -9999).round(3)
        da_scs.raster.set_nodata(-9999)
        # set staticmaps
        sfincs_name = self._MAPS["curve_number"]
        da_scs.attrs.update(**self._ATTRS.get(sfincs_name, {}))
        self.set_staticmaps(da_scs, name=sfincs_name)
        # update config: remove default infiltration values and set cn map
        self.config.pop("qinf", None)
        self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name}")

    def setup_manning_roughness(self, lulc_fn="vito", map_fn=None):
        """Setup model curve number map from gridded curve number map.

        Adds model layers:

        * **man** map: manning roughness coefficient

        Parameters
        ---------
        lulc_fn: str, optional
            Name of landuse-landcover map.

            * Required layers: ['lulc']
        map_fn: path-like, optional
            CSV mapping file with lulc classes in the index column and manning values
            in another column with 'N' as header.
        """
        if map_fn is None:
            map_fn = join(DATADIR, "lulc", f"{lulc_fn}_mapping.csv")
        if not os.path.isfile(map_fn):
            raise IOError(f"Mannng roughness mapping file not found: {map_fn}")
        da_org = self.data_catalog.get_rasterdataset(
            lulc_fn, geom=self.region, buffer=10, variables=["lulc"]
        )
        da_msk = self.staticmaps[self._MAPS["mask"]] > 0
        # reproject and reclassify
        # TODO use generic names for parameters
        # FIXME use hydromt general version!!
        da_man = workflows.landuse(
            da_org, da_msk, map_fn, logger=self.logger, params=["N"]
        )["N"]
        # mask and set precision
        da_man = da_man.where(da_msk, da_man.raster.nodata).round(3)
        # set staticmaps
        sfincs_name = self._MAPS["manning"]
        da_man.attrs.update(**self._ATTRS.get(sfincs_name, {}))
        self.set_staticmaps(da_man, name=sfincs_name)
        # update config: remove default manning values and set maning map
        for v in ["manning_land", "manning_sea", "rgh_lev_land"]:
            self.config.pop(v, None)
        self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name[:3]}")

    def setup_gauges(self, gauges_fn, overwrite=False, **kwargs):
        """Setup model observation point locations.

        Adds model layers:

        * **obs** geom: observation point locations

        Parameters
        ---------
        gauges_fn: str
            Path to observation points geometry file.
            See :py:meth:`~hydromt.open_vector`, for accepted files.
        overwrite: bool, optional
            If True, overwrite existing gauges instead of appending the new gauges.
        """
        name = self._GEOMS["gauges"]
        # ensure the catalog is loaded before adding any new entries
        self.data_catalog.sources
        gdf = self.data_catalog.get_geodataframe(
            gauges_fn, geom=self.region, assert_gtype="Point", **kwargs
        ).to_crs(self.crs)
        if not overwrite and name in self.staticgeoms:
            gdf0 = self._staticgeoms.pop(name)
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf0], ignore_index=True))
            self.logger.info(f"Adding new gauges to existing gauges.")
        self.set_staticgeoms(gdf, name)
        self.set_config(f"{name}file", f"sfincs.{name}")
        self.logger.info(f"{name} set based on {gauges_fn}")

    def setup_structures(
        self, structures_fn, stype, dz=None, overwrite=False, **kwargs
    ):
        """Setup thin dam or weir structures.

        Adds model layer (depending on `stype`):

        * **thd** geom: thin dam
        * **weir** geom: weir / levee

        Parameters
        ----------
        structures_fn : str, Path
            Path to structure line geometry file.
            The "name" (for thd and weir), "z" and "par1" (for weir only) are optional.
            For weirs: `dz` must be provided if gdf has no "z" column or Z LineString;
            "par1" defaults to 0.6 if gdf has no "par1" column.
        stype : {'thd', 'weir'}
            Structure type.
        overwrite: bool, optional
            If True, overwrite existing 'stype' structures instead of appending the
            new structures.
        dz: float, optional
            If provided, for weir structures the z value is calculated from
            the model elevation (dep) plus dz.
        """
        cols = {
            "thd": ["name", "geometry"],
            "weir": ["name", "z", "par1", "geometry"],
        }
        assert stype in cols
        # read, clip and reproject
        gdf = self.data_catalog.get_geodataframe(
            structures_fn, geom=self.region, **kwargs
        ).to_crs(self.crs)
        gdf = gdf[[c for c in cols[stype] if c in gdf.columns]]  # keep relevant cols
        structs = utils.gdf2structures(gdf)  # check if it parsed correct
        # sample zb values from dep file and set z = zb + dz
        if stype == "weir" and dz is not None:
            elv = self.staticmaps[self._MAPS["elevtn"]]
            structs_out = []
            for s in structs:
                pnts = gpd.points_from_xy(x=s["x"], y=s["y"])
                zb = elv.raster.sample(gpd.GeoDataFrame(geometry=pnts, crs=self.crs))
                s["z"] = zb.values + float(dz)
                structs_out.append(s)
            gdf = utils.structures2gdf(structs_out, crs=self.crs)
        elif stype == "weir" and np.any(["z" not in s for s in structs]):
            raise ValueError("Weir structure requires z values.")
        # combine with exisiting structures if present
        if not overwrite and stype in self.staticgeoms:
            gdf0 = self._staticgeoms.pop(stype)
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf0], ignore_index=True))
            self.logger.info(f"Adding {stype} structures to existing structures.")
        # set structures
        self.set_staticgeoms(gdf, stype)
        self.set_config(f"{stype}file", f"sfincs.{stype}")
        self.logger.info(f"{stype} structure set based on {structures_fn}")

    ### FORCING
    def setup_h_forcing(
        self,
        gauges_fn=None,
        timeseries_fn=None,
        mdt_fn=None,
        buffer=0,
        **kwargs,
    ):
        """Setup waterlevel boundary point locations and time series.

        Waterlevel gauge locations are selected from gauges_fn based on location and time.
        The selection on location is based on an area `buffer` m around the model region.
        The selection of time is based on the model config time settings.
        The vertical reference level of the waterlevel data can be corrected to match
        the dep vertical reference level with based on the `mdt_fn` mean dynamical topography.

        If no timeseries files is provided or the model time period is ouside the data range,
        dummy timeseries with zero values are set. If only a timeseries file is provided
        it is used to update the boundary condition at waterlevel point locations in staticgeoms
        with matching IDs.


        Adds model layers:

        * **bnd** geom: waterlevel gauge point locations
        * **bzs** forcing: waterlevel time series [m+ref]

        Parameters
        ----------
        gauges_fn: str, Path
            Path to points geometry or geodataset netcdf file.
            See :py:meth:`~hydromt.open_vector`, for accepted point geometry files.
            See :py:meth:`~hydromt.open_geodataset`, for accepted geodataset netcdf files.

            * Required variables if netcdf: ['waterlevel']
        timeseries_fn: str, Path
            Path to timeseries file associated with gauges_fn. Set to None if gauges_fn
            is a geodataset netcdf file including timeseries data.
            See :py:meth:`~hydromt.open_geodataset`, for accepted files.
        mdt_fn: str, optional
            Path or data source name for mean dynamic topography data, by default 'dtu10mdt_egm96'.
            Difference between vertical reference of elevation and waterlevel data,
            Adds to the waterlevel data before merging.

            * Required variables: ['mdt']
        buffer: float
            Buffer around model region from which to select waterlevel gauges

        """
        name = "waterlevel"
        bnd = self.staticmaps[self._MAPS["mask"]] == 2
        if not np.any(bnd).item():
            # No waterlevel boundary remove bnd/bzs from sfincs.inp
            self.logger.info(f"{name} forcing: no waterlevel boundary cells in model.")
            self._update_forcing_1d(None, name)
            return

        # slice time
        tstart = utils.parse_datetime(self.config["tstart"])
        tstop = utils.parse_datetime(self.config["tstop"])
        if gauges_fn is not None:
            # read amd clip data
            fext = str(gauges_fn).split(".")[-1].lower()
            if fext in self._GEOMS and fext not in kwargs:
                kwargs.update(driver="xy")
            da = self.data_catalog.get_geodataset(
                gauges_fn,
                geom=self.region,
                buffer=buffer,
                fn_ts=timeseries_fn,
                variables=[name],
                time_tuple=(tstart, tstop),
                **kwargs,
            )
        else:
            if self._GEOMS[name] in self.staticgeoms:
                # add timeseries data to existing gdf
                gdf = self.staticgeoms[self._GEOMS[name]]
            else:
                # create bnd point on single waterlevel boundary cell
                x, y = self.staticmaps.raster.xy(*np.where(bnd))
                gdf = gpd.GeoDataFrame(
                    index=[1], geometry=gpd.points_from_xy(x[[0]], y[[0]]), crs=self.crs
                )
            if timeseries_fn is not None:
                da_ts = hydromt.open_timeseries(timeseries_fn, name=name).sel(
                    time=slice(tstart, tstop)
                )
                da = GeoDataArray.from_gdf(gdf, da_ts, index_dim="index")
                self.logger.debug(f"{name} forcing: add time series to gauges.")
            else:
                da = self._dummy_ts(gdf, name, fill_value=0)  # dummy
                mdt_fn = None  # do not correct dummy values
                self.logger.debug(f"{name} forcing: add dummy data to gauges.")
        # correct for MDT
        if mdt_fn is not None and isfile(mdt_fn):
            da_mdt = self.data_catalog.get_rasterdataset(
                mdt_fn, geom=self.region, buffer=buffer, variables=["mdt"]
            )
            mdt_pnts = da_mdt.raster.sample(da.vector.to_gdf()).fillna(0)
            da = da + mdt_pnts
            mdt_avg = mdt_pnts.mean().values
            self.logger.debug(f"{name} forcing: applied MDT (avg: {mdt_avg:+.2f})")
        self._update_forcing_1d(da, name)

    def setup_q_forcing_from_grid(
        self, discharge_fn=None, uparea_fn=None, wdw=1, max_error=0.1
    ):
        """Setup discharge boundary based on gridded discharge data and pre-set
        river inflow locations, for instance by using the ``set_rivers()`` method.

        If an upstream area grid is provided the discharge boundary condition is
        snapped to the best fitting grid cell within a ``wdw`` neighboring cells.
        The best fit is dermined based on the minimal relative upstream area error if
        an upstream area value is available for the discharge boundary locations;
        otherwise it is based on maximum upstream area.

        Adds model layers:

        * **src** geom: discharge gauge point locations
        * **dis** forcing: discharge time series [m3/s]

        Parameters
        ----------
        discharge_fn: str, Path, optional
            Path to gridded discharge netcdf file.

            * Required variables: ['discharge' (m3/s)]
        uparea_fn: str, Path, optional
            Path to upstream area grid in gdal (e.g. geotiff) or netcdf format.

            * Required variables: ['uparea' (km2)]
        wdw: int, optional
            Window size in number of cells around discharge boundary locations
            to snap to, only used if ``uparea_fn`` is provided. By default 1.
        max_error: float, optional
            Maximum relative error between the discharge boundary location upstream area
            and the upstream area of the best fit grid cell, only used if "discharge"
            staticgeoms has a "uparea" column. By default 0.1.
        """
        name = "discharge"
        # parse time slice time
        if discharge_fn is None:
            return
        elif self._GEOMS[name] not in self.staticgeoms:
            self.logger.warning(
                "No discharge inflow points in staticgeoms. "
                "Run ``setup_river_inflow()`` method first to determine inflow locations."
            )
            return
        gdf = self.staticgeoms[self._GEOMS[name]]
        tstart = utils.parse_datetime(self.config["tstart"])
        tstop = utils.parse_datetime(self.config["tstop"])
        da_q = self.data_catalog.get_rasterdataset(
            discharge_fn,
            geom=self.region,
            buffer=1,
            time_tuple=(tstart, tstop),
            variables=[name],
        )
        if uparea_fn is not None:
            da_upa = self.data_catalog.get_rasterdataset(
                uparea_fn, geom=self.region, buffer=1, variables=["uparea"]
            )
            da_upa = da_upa.rename(
                {
                    da_upa.raster.x_dim: da_q.raster.x_dim,
                    da_upa.raster.y_dim: da_q.raster.y_dim,
                }
            )
            ds_wdw = xr.merge([da_q, da_upa]).raster.sample(gdf, wdw=wdw)
            if "uparea" in gdf.columns:
                self.logger.info(f"{name} forcing: snap boundary to best uparea cell.")
                upa0 = xr.DataArray(gdf["uparea"], dims=("index"))
                upa_dff = np.abs(ds_wdw["uparea"].load() - upa0) / upa0
                idx = upa_dff.argmin("wdw")
                valid = np.where(upa_dff.isel(wdw=idx) <= max_error)[0]
                if valid.size < gdf.index.size:
                    self.logger.warning(
                        f"{valid.size}/{gdf.index.size} inflow boundary points with a "
                        f"rel. upstream area error smaller or equal to {max_error:.2f}."
                        " Removing boundary point(s) with larger error."
                    )
                idx = idx.isel(index=valid)
            else:
                self.logger.info(f"{name} forcing: snap boundary to max uparea cell.")
                idx = ds_wdw["uparea"].argmax("wdw")
                valid = np.arange(idx.index.size, dtype=np.int)
            da_pnt = ds_wdw.isel(wdw=idx.load(), index=valid).reset_coords()[name]
        else:
            da_pnt = da_q.raster.sample(gdf).reset_coords()[name]
        # set original locations; parse and update forcing
        da_out = GeoDataArray.from_gdf(gdf.iloc[valid, :], da_pnt, index_dim="index")
        self._update_forcing_1d(da_out, name)

    def setup_q_forcing(self, gauges_fn=None, timeseries_fn=None, **kwargs):
        """Setup discharge boundary based on point location time series.

        If no timeseries file is provided or dummy timeseries with zero values are set.

        If the timeseries file is provided, but no gauges file, the timeseries are
        matched with existing discharge lacations in staticgeoms, if available.
        These can be set for river inflow locations using the ``set_rivers()`` method.

        Adds model layers:

        * **src** geom: discharge gauge point locations
        * **dis** forcing: discharge time series [m3/s]

        Parameters
        ----------
        gauges_fn: str, Path, optional
            Path to points geometry or geodataset netcdf file.
            See :py:meth:`~hydromt.open_vector`, for accecpted point geometry files.
            See :py:meth:`~hydromt.open_geodataset`, for accecpted geodataset netcdf files.

            * Required variables if netcdf: ['discharge']
        timeseries_fn: str, Path, optional
            Path to timeseries file associated with gauges_fn. Set to None if gauges_fn
            is a geodataset netcdf file including timeseries data.
            See :py:meth:`~hydromt.open_geodataset`, for accecpted files.

        """
        name = "discharge"
        has_src = self._GEOMS[name] in self.staticgeoms
        # time slice
        tstart = utils.parse_datetime(self.config["tstart"])
        tstop = utils.parse_datetime(self.config["tstop"])
        if gauges_fn is None and not has_src:
            if timeseries_fn is not None:
                self.logger.warning(
                    "No discharge inflow points in staticgeoms. "
                    "Run ``setup_river_inflow()`` method first to determine inflow locations."
                )
            self._update_forcing_1d(None, name)  # remove dis/src from config
            return
        elif gauges_fn is not None:
            # read amd clip data
            fext = str(gauges_fn).split(".")[-1].lower()
            if fext in self._GEOMS and fext not in kwargs:
                kwargs.update(driver="xy")
            da = self.data_catalog.get_geodataset(
                gauges_fn,
                geom=self.region,
                fn_ts=timeseries_fn,
                variables=[name],
                time_tuple=(tstart, tstop),
                **kwargs,
            )
        else:
            # read timeseries data and match with existing gdf
            gdf = self.staticgeoms[self._GEOMS[name]]
            if timeseries_fn is not None:
                da_ts = hydromt.open_timeseries_from_table(
                    timeseries_fn, name=name
                ).sel(time=slice(tstart, tstop))
                da = GeoDataArray.from_gdf(gdf, da_ts, index_dim="index")
                self.logger.debug(f"{name} forcing: add time series to preset gauges.")
            else:
                da = self._dummy_ts(gdf, name, fill_value=0)  # dummy timeseries
                self.logger.debug(f"{name} forcing: add dummy data to preset gauges.")
        self._update_forcing_1d(da.fillna(0.0), name)

    def setup_p_forcing_from_grid(
        self, precip_fn=None, dst_res=None, aggregate=False, **kwargs
    ):
        """Setup precipitation forcing from a gridded spatially varying data source.

        If aggregate is True, the distributed  precipitation is aggregated for the model
        domain and a spatially uniform ascii `precipfile` is added to the model.
        If aggregate is False, the distributed precipitation are reprojected to the model
        CRS and written to the netcdf `netamprfile`.

        Adds one of these model layer:

        * **netamprfile**: distributed precipitation [mm/hr]
        * **precipfile**: uniform precipitation [mm/hr]

        Parameters
        ----------
        precip_fn, str, Path
            Path to precipitation rasterdataset netcdf file.

            * Required variables: ['precip' (mm)]
        dst_res: float
            output resolution (m), by default None and computed from source data.
            Only used in combination with aggregate=False
        aggregate: bool, {'mean', 'median'}, optional
            Method to aggregate distributed input precipitation data. If True, mean
            aggregation is used, if False (default) the data is not aggregated and
            spatially distributed precipitation is returned.
        """
        variable = "precip"
        # get data for model domain and config time range
        tstart = utils.parse_datetime(self.config["tstart"])
        tstop = utils.parse_datetime(self.config["tstop"])
        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=self.region,
            buffer=2,
            time_tuple=(tstart, tstop),
            variables=[variable],
        )

        # aggregate or reproject in space
        if aggregate:
            stat = aggregate if isinstance(aggregate, str) else "mean"
            self.logger.debug(f"Aggregate {variable} using {stat}.")
            precip_out = precip.raster.zonal_stats(self.region, stats=stat)[
                f"precip_{stat}"
            ]
            precip_out = precip_out.where(precip_out >= 0, 0).fillna(0)
        else:
            # reproject to model utm crs
            # NOTE: currently SFINCS errors (stack overflow) on large files,
            # downscaling to model grid is not recommended
            kwargs0 = dict(align=dst_res is not None, method="nearest_index")
            kwargs0.update(kwargs)
            meth = kwargs0["method"]
            self.logger.debug(f"Resample {variable} using {meth}.")
            precip_out = precip.raster.reproject(
                dst_crs=self.crs, dst_res=dst_res, **kwargs
            ).fillna(0)

        # resample in time
        precip_out = hydromt.workflows.resample_time(
            precip_out,
            freq=pd.to_timedelta("1H"),
            conserve_mass=True,
            upsampling="bfill",
            downsampling="sum",
            logger=self.logger,
        )
        precip_out.name = "Precipitation"  # capital is important for netamprfile

        # set correct names and attrs and add forcing
        if aggregate:
            # precipfile = sfincs.precip
            sfincs_name = self._1DFORCING[variable]
            fn_out = f"sfincs.{sfincs_name}"
        else:
            # netamprfile = precip.nc
            sfincs_name = self._2DFORCING[variable]
            fn_out = f"{variable}.nc"
        precip_out.attrs.update(**self._ATTRS.get(sfincs_name, {}))
        self.set_config(f"{sfincs_name}file", fn_out)
        self.set_forcing(precip_out, name=sfincs_name)

    def setup_p_forcing(self, precip_fn=None, **kwargs):
        """Setup spatially uniform precipitation forcing.

        Adds model layers:

        * **precipfile**: uniform precipitation [mm/hr]

        Parameters
        ----------
        precip_fn, str, Path
            Path to precipitation csv or sfincs.prcp file
        """
        variable = "precip"
        precip_out = hydromt.open_timeseries_from_table(precip_fn, **kwargs)
        if precip_out[precip_out.vector.index_dim].size > 1:
            raise ValueError(f"Uniform {variable} should have index size of one.")
        # precipfile = sfincs.precip
        sfincs_name = self._1DFORCING[variable]
        self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name}")
        self.set_forcing(precip_out, name=sfincs_name)

    def plot_forcing(self, fn_out="forcing.png", **kwargs):
        """Plot model timeseries forcing.

        For distributed forcing a spatial avarage is plotted.

        Parameters
        ----------
        fn_out: str
            Path to output figure file.
            If a basename is given it is saved to <model_root>/figs/<fn_out>
            If None, no file is saved.
        forcing : Dict of xr.DataArray
            Model forcing

        Returns
        -------
        fig, axes
            Model fig and ax objects
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if self.forcing:
            # update missingn attributes for plot labels
            for name in self.forcing:
                attrs = self._ATTRS.get(name, {})
                attrs.update(**self.forcing[name].attrs)
                self.forcing[name].attrs.update(**attrs)
            fig, axes = plots.plot_forcing(self.forcing, **kwargs)

            # set xlim to model tstart - tend
            tstart = utils.parse_datetime(self.config["tstart"])
            tstop = utils.parse_datetime(self.config["tstop"])
            axes[-1].set_xlim(mdates.date2num([tstart, tstop]))

            # save figure
            if fn_out is not None:
                if not os.path.isabs(fn_out):
                    fn_out = join(self.root, "figs", fn_out)
                if not os.path.isdir(dirname(fn_out)):
                    os.makedirs(dirname(fn_out))
                plt.savefig(fn_out, dpi=225, bbox_inches="tight")
            return fig, axes

    def plot_basemap(
        self,
        fn_out: str = "basemap.png",
        variable: str = "dep",
        shaded: bool = True,
        bmap: str = "sat",
        zoomlevel: int = 11,
        figsize: Tuple[int] = None,
        geoms: List[str] = ["rivers", "src", "bnd", "obs"],
        geom_kwargs: Dict = {},
        legend_kwargs: Dict = {},
        **kwargs,
    ):
        """Create basemap plot.

        Parameters
        ----------
        fn_out: str
            Path to output figure file.
            If a basename is given it is saved to <model_root>/figs/<fn_out>
            If None, no file is saved.
        staticmaps : xr.Dataset
            Dataset with model maps
        staticgeoms : Dict of geopandas.GeoDataFrame
            Model geometries
        variable : str, optional
            Map name to plot, by default 'dep'
        shaded : bool, optional
            Add shaded (only if variable is True), by default True
        bmap : {'sat', ''}
            background map, by default "sat"
        zoomlevel : int, optional
            zoomlevel, by default 11
        figsize : Tuple[int], optional
            figure size, by default None
        geoms : List[str], optional
            list of model geometries to plot, by default ["rivers", "src", "bnd", "obs"]
        geom_kwargs : Dict, optional
            Model geometry styling, passed to geopands.GeoDataFrame.plot method
        legend_kwargs : Dict, optional
            Legend kwargs, passed to ax.legend method.

        Returns
        -------
        fig, axes
            Model fig and ax objects
        """
        import matplotlib.pyplot as plt

        fig, ax = plots.plot_basemap(
            self.staticmaps,
            self.staticgeoms,
            variable=variable,
            shaded=shaded,
            bmap=bmap,
            zoomlevel=zoomlevel,
            figsize=figsize,
            geoms=geoms,
            geom_kwargs=geom_kwargs,
            legend_kwargs=legend_kwargs,
            **kwargs,
        )

        if fn_out is not None:
            if not os.path.isabs(fn_out):
                fn_out = join(self.root, "figs", fn_out)
            if not os.path.isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))
            plt.savefig(fn_out, dpi=225, bbox_inches="tight")

        return fig, ax

    # I/O
    def read(self):
        """Read the complete model schematization and configuration from file."""
        self.read_config()
        self.read_staticmaps()
        self.read_staticgeoms()
        self.read_forcing()
        self.read_states()
        self.read_results()
        self.logger.info("Model read")

    def write(self):
        """Write the complete model schematization and configuration to file."""
        self.logger.info(f"Write model data to {self.root}")
        self.write_config()
        self.write_staticmaps()
        self.write_staticgeoms()
        self.write_forcing()
        self.write_states()

    def read_staticmaps(self, crs=None):
        """Read SFNCS binary staticmaps and save to `staticmaps` attribute.
        Known binary files mentioned in the sfincs.inp configuration file are read,
        including: msk/dep/scs/manning/qinf.

        Parameters
        ----------
        crs: int, CRS
            Coordinate reference system, if provided use instead of epsg code from sfincs.inp
        """
        # read geospatial attributes from sfincs.inp, save with with S->N orientation
        shape, transform, crs = self.get_spatial_attrs(crs=crs)

        # read raw numbers and reshape to 2D arrays
        fn_ind = abspath(join(self._root, self.config.get("indexfile")))
        if not isfile(fn_ind):
            raise IOError(f".ind path {fn_ind} does not exist")
        ind = utils.read_binary_map_index(fn_ind)

        dtypes = {"msk": "u1"}
        mvs = {"msk": 0, "scs": 0}
        data_vars = {}
        for name, sfincs_name in self._MAPS.items():
            if f"{sfincs_name}file" in self.config:
                fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                if not isfile(fn):
                    self.logger.warning(f"{sfincs_name}file not found at {fn}")
                    continue
                dtype = dtypes.get(sfincs_name, "f4")
                mv = mvs.get(sfincs_name, -9999.0)
                data = utils.read_binary_map(fn, ind, shape, mv, dtype)
                data_vars.update({sfincs_name: (data, mv)})

        # create dataset and set as staticmaps
        ds = RasterDataset.from_numpy(
            data_vars=data_vars,
            transform=transform,
            crs=crs,
        )
        for name in ds.data_vars:
            ds[name].attrs.update(self._ATTRS.get(name, {}))
        self.set_staticmaps(ds)

    def write_staticmaps(self):
        """Write SFINCS staticmaps to binary files including map index file.
        Filenames are taken from the `config` attribute.

        If `write_gis` property is True, all staticmaps are written to geotiff
        files in a "gis" subfolder.
        """
        if not self._write:
            raise IOError("Model opened in read-only")
        elif not self._staticmaps:
            return

        ds_out = self.staticmaps
        if ds_out.raster.res[1] < 0:  # make sure orientation is S -> N
            yrev = list(reversed(ds_out.raster.ycoords))
            ds_out = ds_out.reindex({ds_out.raster.y_dim: yrev})

        self.logger.debug("Write binary map indices based on mask.")
        msk = ds_out[self._MAPS["mask"]].values
        fn_ind = self.get_config("indexfile", abs_path=True)
        utils.write_binary_map_index(fn_ind, msk=msk)

        dvars = self.staticmaps.raster.vars
        self.logger.debug(f"Write binary map files: {dvars}.")
        dtypes = {"msk": "u1"}  # default to f4
        for sfincs_name in dvars:
            if f"{sfincs_name}file" not in self.config:
                self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name}")
            fn_out = self.get_config(f"{sfincs_name}file", abs_path=True)
            utils.write_binary_map(
                fn_out,
                ds_out[sfincs_name].values,
                msk=msk,
                dtype=dtypes.get(sfincs_name, "f4"),
            )

        if self._write_gis:
            self.write_gis("staticmaps")

    def write_gis(self, variables="all"):
        """Write variables to GIS files in <root>/<gis>/.
        Raster data is written to geotiff and vector data to geojson files.
        NOTE: these files are not used by the model.
        """
        _all = ["staticmaps", "staticgeoms", "states", "results.hmax"]
        if variables == "all":
            variables = _all
        elif isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list):
            raise ValueError(f'"variables" should be a list, not {type(list)}.')
        for var in variables:
            vsplit = var.split(".")
            attr = vsplit[0]
            if not (attr in _all or var in _all):
                self.logger.info(f"Unknown variable {var}: select one of {_all}")
                continue
            obj = getattr(self, f"_{attr}")
            if len(obj) == 0:
                self.logger.info(f"Variable {var} empty, skip writing file.")
                continue  # empty
            self.logger.info(f"Write GIS files for {var} to 'gis' subfolder")
            vars = vsplit[1:] if len(vsplit) >= 2 else list(obj.keys())
            for name in vars:
                if name not in obj:
                    continue
                if attr == "staticgeoms":
                    fn_gis = join(self.root, "gis", f"{name}.geojson")
                    obj[name].to_file(fn_gis, driver="GeoJSON")
                else:
                    da = obj[name]
                    if da.raster.res[1] > 0:  # make sure orientation is N->S
                        yrev = list(reversed(da.raster.ycoords))
                        da = da.reindex({da.raster.y_dim: yrev})
                    da.raster.to_raster(join(self.root, "gis", f"{name}.tif"))

    def read_staticgeoms(self):
        """Read geometry files if and save to `staticgeoms` attribute.
        Known geometry files mentioned in the sfincs.inp configuration file are read,
        including: bnd/src/obs xy files and thd/weir structure files.

        If other geojson files are present in a "gis" subfolder folder, those are read as well.
        """
        if not self._write:
            self._staticgeoms = {}  # fresh start in read-only mode
        # read _GEOMS model files
        for sfincs_name in self._GEOMS.values():
            if f"{sfincs_name}file" in self.config:
                fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                if fn is None:
                    continue
                elif not isfile(fn):
                    self.logger.warning(f"{sfincs_name}file not found at {fn}")
                    continue
                if sfincs_name in ["thd", "weir"]:
                    struct = utils.read_structures(fn)
                    gdf = utils.structures2gdf(struct, crs=self.crs)
                else:
                    gdf = utils.read_xy(fn, crs=self.crs)
                gdf.index = np.arange(1, gdf.index.size + 1, dtype=int)  # start at 1
                self.set_staticgeoms(gdf, name=sfincs_name)
        # read additional geojson files from gis directory
        for fn in glob.glob(join(self.root, "gis", "*.geojson")):
            name = basename(fn).replace(".geojson", "")
            if name in self._GEOMS.values():
                continue
            gdf = hydromt.open_vector(fn, crs=self.crs)
            self.set_staticgeoms(gdf, name=name)

    def write_staticgeoms(self):
        """Write staticgeoms to bnd/src/obs xy files and thd/weir structure files.
        Filenames are based on the `config` attribute.

        If `write_gis` property is True, all staticgeoms are written to geojson
        files in a "gis" subfolder.
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self._staticgeoms:
            self.logger.info("Write staticgeom files")
            for sfincs_name, gdf in self.staticgeoms.items():
                if sfincs_name in self._GEOMS.values():
                    if f"{sfincs_name}file" not in self.config:
                        self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name}")
                    fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                    if sfincs_name in ["thd", "weir"]:
                        struct = utils.gdf2structures(gdf)
                        utils.write_structures(fn, struct, stype=sfincs_name)
                    else:
                        utils.write_xy(fn, gdf, fmt="%8.2f")
            if self._write_gis:
                self.write_gis("staticgeoms")

    def read_forcing(self):
        """Read forcing files and save to `forcing` attribute.
        Known forcing files mentioned in the sfincs.inp configuration file are read,
        including: bzd/dis/precip ascii files and the netampr netcdf file.
        """
        if not self._write:
            # start fresh in read-only mode
            self._forcing = {}
        forcing_dict = {**self._1DFORCING, **self._2DFORCING}  # merge dicts
        tref = utils.parse_datetime(self.config["tref"])
        for name, sfincs_name in forcing_dict.items():
            fn = self.get_config(f"{sfincs_name}file", abs_path=True)
            if fn is None:
                continue
            elif not isfile(fn):
                self.logger.warning(f"{sfincs_name}file not found at {fn}")
                continue
            # read timeseries
            if sfincs_name in self._1DFORCING.values():
                df = utils.read_timeseries(fn, tref)
                # add locations
                gdf = self.staticgeoms.get(self._GEOMS[name], None)
                if gdf is not None:
                    dims = ("time", "index")
                    da = GeoDataArray.from_gdf(gdf, df, dims=dims, name=name)
                else:
                    da = xr.DataArray(df)  # TODO: TEST
            else:
                da = xr.open_dataarray(fn, chunks={"time": 24})  # lazy
            self.set_forcing(da, name=sfincs_name)

    def write_forcing(self):
        """Write forcing to ascii (bzd/dis/precip) and netcdf (netampr) files.
        Filenames are based on the `config` attribute.
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self._forcing:
            self.logger.info("Write forcing files")
            tref = utils.parse_datetime(self.config["tref"])
            # for nc files -> time in minutes since tref
            tref_str = tref.strftime("%Y-%m-%d %H:%M:%S")
            encoding = dict(
                time={"units": f"minutes since {tref_str}", "dtype": "float64"}
            )
            fnames = list(self._1DFORCING.values()) + list(self._2DFORCING.values())
            for sfincs_name in self._forcing:
                if sfincs_name not in fnames:
                    logger.warning(f"{sfincs_name} forcing unknown and skipped.")
                    continue
                if f"{sfincs_name}file" not in self.config:
                    self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name}")
                fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                if sfincs_name in self._1DFORCING.values():
                    # spatially uniform forcing
                    df = self._forcing[sfincs_name].to_series().unstack(0)
                    utils.write_timeseries(fn, df, tref)
                else:
                    # spatially distributed forcing
                    self._forcing[sfincs_name].to_netcdf(fn, encoding=encoding)

    def read_states(self, crs=None):
        """Read waterlevel state (zsini) from ascii file and save to `states` attribute.
        The inifile if mentioned in the sfincs.inp configuration file is read.

        Parameters
        ----------
        crs: int, CRS
            Coordinate reference system, if provided use instead of epsg code from sfincs.inp
        """
        if not self._write:
            # start fresh in read-only mode
            self._states = {}
        if "inifile" in self.config:
            fn = self.get_config("inifile", abs_path=True)
            if not isfile(fn):
                self.logger.warning("inifile not found at {fn}")
                return
            shape, transform, crs = self.get_spatial_attrs(crs=crs)
            zsini = RasterDataArray.from_numpy(
                data=utils.read_ascii_map(fn),  # orientation S-N
                transform=transform,
                crs=crs,
                nodata=-9999,  # TODO: check what a good nodatavalue is
            )
            if zsini.shape != shape:
                raise ValueError('The shape of "inifile" and maps does not match.')
            if self._MAPS["mask"] in self._staticmaps:
                zsini = zsini.where(self.staticmaps[self._MAPS["mask"]] != 0)
            self.set_states(zsini, "zsini")

    def write_states(self, fmt="%8.3f"):
        """Write waterlevel state (zsini)  to ascii map file.
        The filenames is based on the `config` attribute.
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        assert len(self._states) <= 1
        for name in self._states:
            if f"inifile" not in self.config:
                self.set_config(f"inifile", f"sfincs.{name}")
            fn = self.get_config("inifile", abs_path=True)
            da = self._states[name].fillna(0)  # TODO check proper nodata value
            if da.raster.res[1] < 0:  # orientation is S->N
                da = da.reindex({da.raster.y_dim: list(reversed(da.raster.ycoords))})
            utils.write_ascii_map(fn, da.values, fmt=fmt)
        if self._write_gis:
            self.write_gis("states")

    def read_results(self, chunksize=100, drop=["crs", "sfincsgrid"]):
        """Read results from sfincs_map.nc and sfincs_his.nc and save to the `results` attribute.
        The staggered nc file format is translated into hydromt.RasterDataArray formats.
        Additionally, hmax is computed from zsmax and zb if present.

        Parameters
        ----------
        chunksize: int, optional
            chunk size along time dimension, by default 100
        """

        fn_map = join(self.root, "sfincs_map.nc")
        if isfile(fn_map):
            ds_face, ds_edge = utils.read_sfincs_map_results(
                fn_map, crs=self.crs, chunksize=chunksize, drop=drop, logger=self.logger
            )
            # save as dict of DataArray
            self.set_results(ds_face)
            self.set_results(ds_edge)

        fn_his = join(self.root, "sfincs_his.nc")
        if isfile(fn_his):
            ds_his = utils.read_sfincs_his_results(
                fn_his, crs=self.crs, chunksize=chunksize
            )
            # drop double vars (map files has priority)
            drop_vars = [v for v in ds_his.data_vars if v in self._results or v in drop]
            ds_his = ds_his.drop_vars(drop_vars)
            self.set_results(ds_his)

    def write_results(self):
        pass  # TODO remove from model API

    def _update_forcing_1d(self, da, name):
        """Set 1D forcing (and remove if da is None) and update config accordingly."""
        fname = self._1DFORCING.get(name, None)
        gname = self._GEOMS.get(name, name)
        if fname is None:
            raise ValueError(f"unknown 1D forcing name {name}")
        n = da.vector.index.size if da is not None else 0
        self.logger.debug(f"{name} forcing: setting data at {n} points.")
        if n > 0:  # save to forcing dict and set in config
            if not isinstance(da, xr.DataArray):
                raise ValueError(f"{name} forcing: object should be xarray.DataArray.")
            # make sure time is on last dim
            da = da.transpose(da.vector.index_dim, da.vector.time_dim)
            if da.time.size == 0:  # set dummy values if time dim has zero
                self.logger.warning(
                    f"{name} forcing: no timeseries data in object, setting dummy values."
                )
                da = self._dummy_ts(da.vector.to_gdf(), name, fill_value=0)
            # and crs is set (should always be model crs at this stage)
            if da.vector.crs is None:
                da.vector.set_crs(self.crs.to_epsg())
            elif da.vector.crs != self.crs:
                da = da.vector.to_crs(self.crs.to_epsg())
            # fix order based on x_dim (for comparibility between OS)
            da = da.sortby(da.vector.x_dim, ascending=True)
            dim = da.vector.index_dim
            da[dim] = xr.IndexVariable(dim, np.arange(1, da[dim].size + 1, dtype=int))
            self.set_forcing(da, fname)
            self.set_staticgeoms(da.vector.to_gdf(), gname)
            # edit inp file
            self.logger.debug(f"{name} forcing: updating sfincs.inp.")
            for sname in [fname, gname]:
                self.set_config(f"{sname}file", f"sfincs.{sname}")
        else:  # remove forcing data and sfincs.inp entries
            if fname in self._forcing:
                self.logger.debug(f"{name} forcing: removing data.")
                self._forcing.pop(name)
            if f"{fname}file" in self.config:
                self.logger.debug(
                    f"{name} forcing: removing {fname}file from sfincs.inp."
                )
                self._config.pop(f"{fname}file")

    ## model configuration

    def set_crs(self, crs):
        super(SfincsModel, self).set_crs(crs)
        self.update_spatial_attrs()

    def _configread(self, fn):
        return utils.read_inp(fn)

    def _configwrite(self, fn):
        return utils.write_inp(fn, self.config)

    def update_spatial_attrs(self):
        """Update geospatial `config` (sfincs.inp) attributes based on staticmaps"""
        dx, dy = self.res
        west, south, _, _ = self.bounds
        if self.crs is not None:
            self.set_config("epsg", self.crs.to_epsg())
        self.set_config("mmax", self.width)
        self.set_config("nmax", self.height)
        self.set_config("dx", dx)
        self.set_config("dy", abs(dy))  # dy is always positive (orientation is S -> N)
        self.set_config("x0", west)
        self.set_config("y0", south)

    def get_spatial_attrs(self, crs=None):
        """Get geospatial `config` (sfincs.inp) attributes.

        Parameters
        ----------
        crs: int, CRS
            Coordinate reference system

        Returns
        -------
        shape: tuple of int
            width, height
        transform: Affine.transform
            Geospatial transform
        crs: pyproj.CRS
            Coordinate reference system
        """
        return utils.get_spatial_attrs(self.config, crs=crs, logger=self.logger)

    def _dummy_ts(self, gdf, name, fill_value=0):
        tstart = utils.parse_datetime(self.config["tstart"])
        tstop = utils.parse_datetime(self.config["tstop"])
        df = pd.DataFrame(
            index=pd.DatetimeIndex([tstart, tstop]),
            columns=gdf.index.values,
            data=np.full((2, gdf.index.size), fill_value, dtype=np.float32),
        )
        da = GeoDataArray.from_gdf(gdf, df, dims=("time", "index"), name=name)
        return da
