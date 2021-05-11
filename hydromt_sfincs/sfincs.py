# -*- coding: utf-8 -*-
import os
from os.path import join, isfile, abspath, dirname, basename
import glob
import numpy as np
import logging
from configparser import ConfigParser
import rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import xarray as xr
import pyproj
from datetime import datetime
import pyflwdir

import hydromt
from hydromt.models.model_api import Model
from hydromt.vector import GeoDataArray
from hydromt.raster import RasterDataset

from . import workflows, DATADIR

__all__ = ["SfincsModel"]

logger = logging.getLogger(__name__)


class ConfigParserSfincs(ConfigParser):
    def __init__(self, **kwargs):
        defaults = dict(
            comment_prefixes=("!", "/", "#"),
            inline_comment_prefixes=("!"),
            allow_no_value=True,
            delimiters=("="),
        )
        defaults.update(**kwargs)
        super(ConfigParserSfincs, self).__init__(**defaults)

    def read_file(self, f, **kwargs):
        def add_header(f, header_name="dummy"):
            """add header"""
            yield "[{}]\n".format(header_name)
            for line in f:
                yield line

        super(ConfigParserSfincs, self).read_file(add_header(f), **kwargs)

    def _write_section(self, fp, section_name, section_items, delimiter):
        """Write a single section to the specified `fp'."""
        for key, value in section_items:
            value = self._interpolation.before_write(self, section_name, key, value)
            fp.write("{:<15} {:<1} {:<}\n".format(key, self._delimiters[0], value))
        fp.write("\n")


class SfincsModel(Model):
    _NAME = "sfincs"
    _GEOMS = {
        "waterlevel": "bnd",
        "discharge": "src",
        "gauges": "obs",
    }
    _1DFORCING = {
        "waterlevel": "bzs",
        "discharge": "dis",
    }
    _2DFORCING = {
        "precip": "netampr",
    }
    _MAPS = {
        "elevtn": "dep",
        "mask": "msk",
        "curve_number": "scs",
        "manning": "manning",
    }
    _FOLDERS = [""]  # to create root folder without subfolders
    _dtfmt = "%Y%m%d %H%M%S"
    _CONF = "sfincs.inp"
    _DATADIR = DATADIR
    _ATTRS = {
        "dep": {"standard_name": "elevation", "unit": "m"},
        "msk": {"standard_name": "mask", "unit": "-"},
        "scs": {"standard_name": "curve number", "unit": "-"},
        "man": {"standard_name": "manning roughness", "unit": "-"},
        "bzs": {"standard_name": "waterlevel", "unit": "m+EGM96"},
        "dis": {"standard_name": "discharge", "unit": "m3.s-1"},
        "netampr": {"standard_name": "precipitation", "unit": "mm.hr-1"},
    }

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn="sfincs.inp",
        write_gis=True,
        opt={},
        data_libs=None,
        deltares_data=None,
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
            da_elv = da_elv.raster.clip_geom(geom=dst_geom, buffer=5).raster.reproject(
                dst_res=res, dst_crs=dst_crs, align=True, method=method
            )
        da_elv = da_elv.raster.clip_geom(dst_geom, align=res)
        # force nodata value to be np.nan
        da_elv = da_elv.raster.mask_nodata()

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
            bas_msk = xr.full_like(da_elv, True, np.bool)
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
        self._config_update()  # update geospatial header in config

        # update mask staticgeom
        da_mask = (da_mask != 0).astype(np.int16)
        da_mask.raster.set_crs(dst_crs)
        da_mask.raster.set_nodata(0)
        self.set_staticgeoms(da_mask.raster.vectorize(), "region")
        self.set_staticgeoms(da_mask.raster.box, "bbox")

        # log n active cells
        ncells = np.sum(msk == 1)
        self.logger.info(f"Mask generated; {ncells:d} active cells.")

    def setup_rivers(self, river_upa=25.0, basemaps_fn="merit_hydro"):
        """Setup river source points where a river enters the model domain.

        NOTE: to ensure a river only enters the model domain once, use the 'basin',
        subbasin, or outlet region options.

        Adds model layers:

        * **src** geoms: discharge boundary point locations
        * **river** geoms: river centerline

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

        self.logger.debug(f"Get river cells; upstream area threshold: {river_upa} km2.")
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

        # vectorize river
        self.logger.debug(f"Vectorize river.")
        feats = flwdir.vectorize(mask=np.logical_and(basmsk, rivmsk).values)
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs)
        gdf_riv.index = gdf_riv.index.values + 1  # one based index
        gdf_riv = gdf_riv.to_crs(dst_crs)
        self.set_staticgeoms(gdf_riv, name="rivers")

    def setup_rivers_downstream(self, river_upa=25.0, outflw_width=1000, basemaps_fn="merit_hydro"):
        """Setup river source points where a river flows out of the model domain downstream.

        NOTE: to ensure a river only leaves the model domain once, use the 'basin',
        subbasin, or outlet region options.

        Adds / edits model layers:

        * **msk** map: edited by adding outflow points (msk=3)
        * **src** geoms: discharge boundary point locations
        * **river** geoms: river centerline
        * **dis** forcing: dummy discharge timeseries

        Parameters
        ----------
        basemaps_f: str, Path
            Path or data source name for hydrography raster data, by default 'merit_hydro'.

            * Required layers: ['uparea', 'flwdir'].
        river_upa: float, optional
            Mimimum upstream area threshold [km2] to define river cells, by default 25 km2.
        """
        # read data and rasterize basin mask > used to initialize flwdir in river workflow
        ds = self.data_catalog.get_rasterdataset(basemaps_fn, geom=self.region, buffer=2)
        src_crs = ds.raster.crs.to_epsg()
        dst_crs = self.crs.to_epsg()
        basmsk = ds.raster.geometry_mask(self.region)  # Q: okay that this is ds.raster instead of .rio?

        self.logger.debug(f"Get river cells; upstream area threshold: {river_upa} km2.")
        # initialize flwdir with river cells only (including outside basin)
        rivmsk = ds["uparea"] >= river_upa
        ds["mask"] = np.logical_and(ds["uparea"] >= river_upa, basmsk)
        flwdir = hydromt.flw.flwdir_from_da(ds["flwdir"], mask=True)

        # git pits at domain edge
        idxs0 = flwdir.idxs_pit
        _msk = ndimage.binary_erosion(basmsk, structure=np.ones((3, 3)))
        edge = np.logical_xor(_msk, basmsk)
        tmp = edge.values.flat[idxs0]
        idxs_outflw = np.unique(idxs0[edge.values.flat[idxs0]])
        edge.astype('int').raster.to_raster(r'edge.tif')
        rivmsk.astype('int').raster.to_raster(r'rivmsk.tif')

        if len(idxs_outflw) > 0:
            self.logger.debug(f"Set msk=3 outflow points.")
            da_mask = self.staticmaps[self._MAPS["mask"]]
            gdf_outflw = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(*flwdir.xy(idxs_outflw)), crs=src_crs
            ).to_crs(dst_crs)
            # apply buffer
            gdf_outflw_buf = gpd.GeoDataFrame(
                geometry=gdf_outflw.buffer(outflw_width / 2.0), crs=gdf_outflw.crs
            )
            msk = da_mask.values
            _msk = ndimage.binary_erosion(msk, structure=np.ones((3, 3)))  # Q: double line?
            outflw_buf = da_mask.raster.geometry_mask(gdf_outflw_buf).values
            outflw_msk = np.logical_and(outflw_buf, np.logical_xor(_msk, msk))

            msk[outflw_msk] = 3
            da_mask.data = msk
            self.set_staticmaps(da_mask)
            self.logger.debug(f"{len(idxs_outflw)} river outflow point locations set.")

        # vectorize river
        self.logger.debug(f"Vectorize river downstream.")
        feats = flwdir.vectorize(mask=np.logical_and(basmsk, rivmsk).values)
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs)
        gdf_riv.index = gdf_riv.index.values + 1  # one based index
        gdf_riv = gdf_riv.to_crs(dst_crs)
        self.set_staticgeoms(gdf_riv, name="rivers_downstream")

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
        # FIXME remove with new ini files
        if cn_fn is None:
            return
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
        # mask and set
        da_scs = da_scs.where(da_msk, -9999)
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
        # FIXME remove with new ini files
        if lulc_fn is None:
            return
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
        # mask
        da_man = da_man.where(da_msk, da_man.raster.nodata)
        # set staticmaps
        sfincs_name = self._MAPS["manning"]
        da_man.attrs.update(**self._ATTRS.get(sfincs_name, {}))
        self.set_staticmaps(da_man, name=sfincs_name)
        # update config: remove default manning values and set maning map
        for v in ["manning_land", "manning_sea", "rgh_lev_land"]:
            self.config.pop(v, None)
        self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name[:3]}")

    def setup_gauges(self, gauges_fn=None, **kwargs):
        """Setup model observation point locations.

        Adds model layers:

        * **obs** geom: observation point locations

        Parameters
        ---------
        gauges_fn: str, optional
            Path to observation points geometry file.
            See :py:meth:`~hydromt.open_vector`, for accepted files.
        """
        if gauges_fn is not None:
            name = self._GEOMS["gauges"]
            # ensure the catalog is loaded before adding any new entries
            self.data_catalog.sources
            gdf = self.data_catalog.get_geodataframe(
                str(gauges_fn), geom=self.region, assert_gtype="Point", **kwargs
            ).to_crs(self.crs)
            self.set_staticgeoms(gdf, name)
            self.set_config(f"{name}file", f"sfincs.{name}")
            self.logger.info(f"{name} set based on {gauges_fn}")

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
        tstart = datetime.strptime(self.config["tstart"], self._dtfmt)
        tstop = datetime.strptime(self.config["tstop"], self._dtfmt)
        if gauges_fn is not None:
            # read amd clip data
            fext = str(gauges_fn).split(".")[-1].lower()
            if fext in self._GEOMS and driver not in kwargs:
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
                "Run ``setup_rivers()`` method first to determine inflow locations."
            )
            return
        gdf = self.staticgeoms[self._GEOMS[name]]
        tstart = datetime.strptime(self.config["tstart"], self._dtfmt)
        tstop = datetime.strptime(self.config["tstop"], self._dtfmt)
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
        gauges_fn, str, Path, optional
            Path to points geometry or geodataset netcdf file.
            See :py:meth:`~hydromt.open_vector`, for accecpted point geometry files.
            See :py:meth:`~hydromt.open_geodataset`, for accecpted geodataset netcdf files.

            * Required variables if netcdf: ['discharge']
        timeseries_fn, str, Path, optional
            Path to timeseries file associated with gauges_fn. Set to None if gauges_fn
            is a geodataset netcdf file including timeseries data.
            See :py:meth:`~hydromt.open_geodataset`, for accecpted files.

        """
        name = "discharge"
        has_src = self._GEOMS[name] in self.staticgeoms
        # time slice
        tstart = datetime.strptime(self.config["tstart"], self._dtfmt)
        tstop = datetime.strptime(self.config["tstop"], self._dtfmt)
        if gauges_fn is None and not has_src:
            if timeseries_fn is not None:
                self.logger.warning(
                    "No discharge inflow points in staticgeoms. "
                    "Run ``setup_rivers()`` method first to determine inflow locations."
                )
            self._update_forcing_1d(None, name)  # remove dis/src from config
            return
        elif gauges_fn is not None:
            # read amd clip data
            fext = str(gauges_fn).split(".")[-1].lower()
            if fext in self._GEOMS and driver not in kwargs:
                kwargs.update(driver="xy")
            da = self.data_catalog.get_geodataset(
                gauges_fn,
                geom=self.region,
                fn_ts=timeseries_fn,
                variables=[name],
                time_tuple=slice(tstart, tstop),
                **kwargs,
            )
        else:
            # read timeseries data and match with existing gdf
            gdf = self.staticgeoms[self._GEOMS[name]]
            if timeseries_fn is not None:
                da_ts = hydromt.open_timeseries_from_table(timeseries_fn, name=name).sel(
                    time=slice(tstart, tstop)
                )
                da = GeoDataArray.from_gdf(gdf, da_ts, index_dim="index")
                self.logger.debug(f"{name} forcing: add time series to preset gauges.")
            else:
                da = self._dummy_ts(gdf, name, fill_value=0)  # dummy timeseries
                self.logger.debug(f"{name} forcing: add dummy data to preset gauges.")
        self._update_forcing_1d(da.fillna(0.0), name)

    def setup_p_forcing_gridded(self, precip_fn=None, dst_res=None, **kwargs):
        """Setup gridded precipitation forcing.

        The input precipitation grids are reprojected to the model CRS.

        Adds model layers:

        * **netamprfile**: gridded precipitation [mm/hr]

        Parameters
        ----------
        gauges_fn, str, Path
            Path to precipitation rasterdataset netcdf file.

            * Required variables: ['precip' (mm)]
        dst_res: float
            output resolution (m), by default None and computed from source data.

        """
        variable = "precip"
        if precip_fn is None:
            return
        # get data for model domain and config time range
        tstart = datetime.strptime(self.config["tstart"], self._dtfmt)
        tstop = datetime.strptime(self.config["tstop"], self._dtfmt)
        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=self.region,
            buffer=2,
            time_tuple=(tstart, tstop),
            variables=[variable],
        )

        # reproject to utm zone:
        # NOTE: currently SFINCS errors (stack overflow) on large files,
        # downscaling to model grid is not recommended
        kwargs0 = dict(
            align=dst_res is not None,
            method="nearest_index",
            freq=pd.to_timedelta("1H"),
        )
        kwargs0.update(kwargs)
        precip_out = precip.raster.reproject(
            dst_crs=self.crs, dst_res=dst_res, **kwargs
        ).fillna(0)

        # set correct names and attrs and add forcing
        sfincs_name = self._2DFORCING[variable]
        precip_out.attrs.update(**self._ATTRS.get(sfincs_name, {}))
        self.set_config(f"{sfincs_name}file", f"{variable}.nc")
        precip_out.name = "Precipitation"  # capital is important
        self.set_forcing(precip_out, name=sfincs_name)

    def plot_forcing(self, fn_out="forcing.png", **kwargs):
        import matplotlib.pyplot as plt

        if self.forcing:
            n = len(self.forcing.keys())
            kwargs0 = dict(sharex=True, figsize=(6, n * 3))
            kwargs0.update(**kwargs)
            fig, axes = plt.subplots(n, 1, **kwargs0)
            axes = [axes] if n == 1 else axes
            for i, name in enumerate(self.forcing):
                da = self.forcing[name].squeeze()
                attrs = self._ATTRS.get(name, {})
                longname = attrs.get("standard_name", "")
                if da.ndim == 3:
                    da = da.mean(dim=[da.raster.x_dim, da.raster.y_dim])
                da.attrs.update(attrs)
                da.plot.line(ax=axes[i], x="time")
                axes[i].set_title(f"SFINCS {longname} boundaries ({name})")
            if fn_out is not None:
                if not os.path.isabs(fn_out):
                    fn_out = join(self.root, "figs", fn_out)
                if not os.path.isdir(dirname(fn_out)):
                    os.makedirs(dirname(fn_out))
                plt.savefig(fn_out, dpi=225, bbox_inches="tight")
            return fig, axes

    def plot_basemap(
        self,
        fn_out="basemap.png",
        variable="dep",
        shaded=True,
        bmap="sat",
        zoomlevel=11,
        figsize=[6.4 * 1.2, 4.8 * 1.2],
        geoms=["rivers_downstream", "rivers","src"],
        geom_kwargs={},
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import colors, patheffects
        import cartopy.io.img_tiles as cimgt
        import cartopy.crs as ccrs

        # read crs and utm zone > convert to cartopy
        ds = self.staticmaps
        wkt = ds.raster.crs.to_wkt()
        if "UTM zone " not in wkt:
            raise ValueError("Model CRS UTM zone not found.")
        utm_zone = ds.raster.crs.to_wkt().split("UTM zone ")[1][:3]
        utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
        extent = np.array(ds.raster.box.buffer(2e3).total_bounds)[[0, 2, 1, 3]]

        # create fig with geo-axis and set background
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(projection=utm)
        ax.set_extent(extent, crs=utm)
        if bmap == "sat":
            ax.add_image(cimgt.QuadtreeTiles(), zoomlevel)
        elif bmap == "osm":
            ax.add_image(cimgt.OSM(), zoomlevel)

        # make nice cmap
        if "cmap" not in kwargs or "norm" not in kwargs:
            if variable == self._MAPS["elevtn"]:
                vmin, vmax = da = (
                    ds[variable].raster.mask_nodata().quantile([0.0, 0.98])
                )
                vmin, vmax = kwargs.pop("vmin", vmin), kwargs.pop("vmax", vmax)
                c_bat = plt.cm.terrain(np.linspace(0, 0.17, 256))
                c_dem = plt.cm.terrain(np.linspace(0.25, 1, 256))
                if vmin < 0:
                    c_all = np.vstack((c_bat, c_dem))
                    cmap = colors.LinearSegmentedColormap.from_list("bat_dem", c_all)
                    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                else:
                    cmap = colors.LinearSegmentedColormap.from_list("dem", c_dem)
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                cmap, norm = kwargs.pop("cmap", cmap), kwargs.pop("norm", norm)
                kwargs.update(norm=norm, cmap=cmap)
        if variable in ds:
            da = ds[variable].raster.mask_nodata()
            da.plot(transform=utm, ax=ax, zorder=1, **kwargs)
            if shaded and variable == self._MAPS["elevtn"]:
                ls = colors.LightSource(azdeg=315, altdeg=45)
                dx, dy = da.raster.res
                _rgb = ls.shade(
                    da.fillna(0).values,
                    norm=kwargs["norm"],
                    cmap=kwargs["cmap"],
                    blend_mode="soft",
                    dx=dx,
                    dy=dy,
                    vert_exag=50,
                )
                rgb = xr.DataArray(
                    dims=("y", "x", "rgb"), data=_rgb, coords=da.raster.coords
                )
                rgb = xr.where(np.isnan(da), np.nan, rgb)
                rgb.plot.imshow(transform=utm, ax=ax, zorder=1)

        # add geoms
        geom_kwargs0 = {
            "rivers_downstream": dict(linestyle="--", linewidth=1.0, color="r"),
            "rivers": dict(linestyle="--", linewidth=1.0, color="b"),
            "src": dict(marker="^", markersize=75, c="w", edgecolor="k", annotate=True),
        }
        geom_kwargs0.update(geom_kwargs)
        ann_kwargs = dict(
            xytext=(3, 3),
            textcoords="offset points",
            zorder=4,
            path_effects=[
                patheffects.Stroke(linewidth=3, foreground="w"),
                patheffects.Normal(),
            ],
        )
        if self.staticgeoms:
            for name in geoms:
                gdf = self.staticgeoms.get(name, None)
                if gdf is None:
                    continue
                annotate = geom_kwargs0[name].pop("annotate", False)
                gdf.plot(ax=ax, zorder=3, **geom_kwargs0[name])
                if annotate:
                    for label, row in gdf.iterrows():
                        x, y = row.geometry.x, row.geometry.y
                        ax.annotate(label, xy=(x, y), **ann_kwargs)
            self.region.boundary.plot(ax=ax, ls="-", lw=0.5, color="k", zorder=2)

        # title and labels
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.set_ylabel(f"y coordinate UTM zone {utm_zone} [m]")
        ax.set_xlabel(f"x coordinate UTM zone {utm_zone} [m]")
        variable = "base" if variable is None else variable
        ax.set_title(f"SFINCS {variable} map")
        if fn_out is not None:
            if not os.path.isabs(fn_out):
                fn_out = join(self.root, "figs", fn_out)
            if not os.path.isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))
            plt.savefig(fn_out, dpi=225, bbox_inches="tight")

        return fig, ax

    # I/O
    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.read_config()
        self.read_staticmaps()
        self.read_staticgeoms()
        self.read_forcing()
        self.logger.info("Model read")

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Write model data to {self.root}")
        self.write_config()
        self.write_staticmaps()
        self.write_staticgeoms()
        self.write_forcing()

    def read_staticmaps(self, crs=None, nodata=-9999.0):
        """Read and SFNCS binary staticmaps

        Parameters
        ---------
        crs : coordinate ref. system
            this crs is assigned to the staticmaps, by default read from config file.
        nodata : float, optional
            assigned to missing values (mask = 0)
        """
        # retrieve rows and cols
        if crs is None and self.config.get("epsg", None) is not None:
            crs = pyproj.CRS.from_epsg(int(self.config.get("epsg")))
        cols = self.config["mmax"]
        rows = self.config["nmax"]
        dx = self.config["dx"]
        dy = self.config["dy"]
        west = self.config["x0"]
        south = self.config["y0"]
        rotdeg = self.config["rotation"]  # clockwise rotation in degrees
        if rotdeg != 0:
            raise NotImplementedError("Rotated grids cannot be parsed yet.")
        # TODO: extend to use rotated grids with rotated affine
        #     # code below generates a 2D coordinate grids.
        #     xx = np.linspace(0, dx * (cols - 1), cols)
        #     yy = np.linspace(0, dy * (rows - 1), rows)
        #     xi, yi = np.meshgrid(xx, yy)
        #     rot = rotdeg * np.pi / 180
        #     # xgrid and ygrid not used for now
        #     xgrid = x0 + np.cos(rot) * xi - np.sin(rot) * yi
        #     ygrid = y0 + np.sin(rot) * xi + np.cos(rot) * yi
        # ...
        else:
            north = south + dy * rows
            transform = rasterio.transform.from_origin(west, north, dx, dy)

        # read raw numbers and reshape to 2D arrays
        fn_ind = abspath(join(self._root, self.config.get("indexfile")))
        if not isfile(fn_ind):
            raise IOError(f".ind path {fn_ind} does not exist")
        ind = np.fromfile(fn_ind, dtype="u4")[1:] - 1  # convert to zero based index

        dtypes = {
            "msk": "u1",
        }
        mvs = {
            "msk": 0,
            "scs": 0,
        }
        data_vars = {}
        for name, sfincs_name in self._MAPS.items():
            fn = self.get_config(f"{sfincs_name}file", abs_path=True)
            if fn is None:
                continue
            elif not isfile(fn):
                raise IOError(f"{sfincs_name} path {fn} does not exist")
            dtype = dtypes.get(sfincs_name, "f4")
            mv = mvs.get(sfincs_name, nodata)
            data = np.full((cols, rows), mv, dtype=dtype)
            data.flat[ind] = np.fromfile(fn, dtype=dtype)
            data = np.flipud(data.transpose())
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

    def write_staticmaps(self, write_figs=False):
        """Write binary SFINCS staticmaps.
        NOTE: This will only write the bathymetry, mask and indices data,
        not the geographic information
        If write_gis, write staticmaps to gis geotif as well.
        """
        if not self._write:
            raise IOError("Model opened in read-only")

        self.logger.debug("Derive indices from mask.")
        # bin files index of sfincs is transposed compared to numpy 2D arrays
        msk = np.flipud(self.staticmaps[self._MAPS["mask"]].values).transpose()
        msk_ = np.array(msk[msk > 0], dtype="u1")
        # the index number file of sfincs starts with the length of the index numbers,
        # add that below
        indices = np.where(msk.flatten() > 0)[0] + 1  # one-based index
        indices_ = np.array(np.hstack([np.array(len(indices)), indices]), dtype="u4")
        # write to file
        fn_msk = self.get_config("mskfile", abs_path=True)
        fn_ind = self.get_config("indexfile", abs_path=True)
        msk_.tofile(fn_msk)
        indices_.tofile(fn_ind)

        dtypes = {"dep": "f4"}
        for sfincs_name in self.staticmaps:
            if sfincs_name == self._MAPS["mask"]:
                continue
            fn_out = self.get_config(f"{sfincs_name}file", abs_path=True)
            if fn_out is None:
                fn_out = join(self.root, f"sfincs.{sfincs_name[:3]}")
                self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name}")
            data = np.flipud(self.staticmaps[sfincs_name].values).transpose()
            data_ = np.array(data[msk > 0], dtype=dtypes.get(sfincs_name, "f4"))
            data_.tofile(fn_out)

        if self._write_gis:
            self.logger.info("Write GIS raster files to 'gis' subfolder")
            ds_out = self._staticmaps
            ds_out.raster.to_mapstack(join(self.root, "gis"))

    @property
    def staticgeoms(self):
        """geopandas.GeoDataFrame representation of all model geometries"""
        if self._staticgeoms:
            if self._read:
                self.read_staticgeoms()
        return self._staticgeoms

    def read_staticgeoms(self):
        """Read bnd/src/obs SFINCS xy files
        If other geojson files are present gis folder, read those as well.
        """
        if not self._write:
            self._staticgeoms = {}  # fresh start in read-only mode
        # read _GEOMS model files
        for name, sfincs_name in self._GEOMS.items():
            if f"{sfincs_name}file" in self.config:
                fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                if not isfile(fn):
                    continue
                gdf = hydromt.open_vector(fn, crs=self.crs, driver="xy")
                gdf.index = gdf.index.values + 1  # start index at one
                self.set_staticgeoms(gdf, name=sfincs_name)
        # read additional geojson files from gis directory
        for fn in glob.glob(join(self.root, "gis", "*.geojson")):
            sfincs_name = basename(fn).replace(".geojson", "")
            if sfincs_name in self._GEOMS.values():
                continue
            gdf = hydromt.open_vector(fn, crs=self.crs)
            self.set_staticgeoms(gdf, name=sfincs_name)

    def write_staticgeoms(self):
        """Write bnd/src/obs to SFINCS xy files
        If write_gis, write all staticgeoms to geojson files in gis subfolder
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self._staticgeoms:
            self.logger.info("Write staticgeom files")
            # NOTE: this only works for point vector file
            for name, gdf in self.staticgeoms.items():
                sfincs_name = self._GEOMS.get(name, name)
                if f"{sfincs_name}file" in self.config:
                    fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                    hydromt.write_xy(fn, gdf, fmt="%8.2f")
                if self._write_gis:
                    fn_gis = join(self.root, "gis", f"{sfincs_name}.geojson")
                    gdf.to_file(fn_gis, driver="GeoJSON")

    def read_forcing(self):
        """Read bzs/dis forcing timeseries and netampr gridded forcing"""
        if not self._write:
            # start fresh in read-only mode
            self._forcing = {}
        if self.staticgeoms:
            for name, sfincs_name in self._1DFORCING.items():
                gdf = self.staticgeoms.get(self._GEOMS[name], None)
                fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                if gdf is None or fn is None:
                    continue
                elif not isfile(fn):
                    self.logger.warning(f"{sfincs_name} not found at {fn}")
                    continue
                # read timeseries
                tref = datetime.strptime(self.config["tref"], self._dtfmt)
                df = _read_timeseries(fn, tref)
                dims = ("time", "index")
                da = GeoDataArray.from_gdf(gdf, df, dims=dims, name=name)
                self.set_forcing(da, name=sfincs_name)
        for name, sfincs_name in self._2DFORCING.items():
            fn = self.get_config(f"{sfincs_name}file", abs_path=True)
            if fn is None:
                continue
            elif not isfile(fn):
                self.logger.warning(f"{sfincs_name} not found at {fn}")
                continue
            # NOTE: assume single variable per nc file
            self.set_forcing(xr.open_dataarray(fn), name=sfincs_name)

    def write_forcing(self):
        """write bzs/dis 1D forcing and netampr 2D forcing files"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.forcing:
            self.logger.info("Write forcing files")
            tref = datetime.strptime(self.config["tref"], self._dtfmt)
            for name, sfincs_name in self._1DFORCING.items():
                if sfincs_name not in self._forcing:
                    continue
                if self.get_config(f"{sfincs_name}file") is None:
                    self.set_config(f"{sfincs_name}file", f"sfincs.{sfincs_name}")
                fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                df = self._forcing[sfincs_name].to_series().unstack(0)
                _write_timeseries(fn, df, tref)
            for name, sfincs_name in self._2DFORCING.items():
                if sfincs_name not in self._forcing:
                    continue
                if self.get_config(f"{sfincs_name}file") is None:
                    self.set_config(f"{sfincs_name}file", f"{sfincs_name}.nc")
                fn = self.get_config(f"{sfincs_name}file", abs_path=True)
                # time in minutes
                tref_str = tref.strftime("%Y-%m-%d %H:%M:%S")
                encoding = dict(
                    time={"units": f"minutes since {tref_str}", "dtype": "float64"}
                )
                # write to file and set config key
                self._forcing[sfincs_name].to_netcdf(fn, encoding=encoding)
            # TODO remove here with new ini files
            # self.plot_forcing()

    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray"""
        return self._states
        # raise NotImplementedError()

    def write_states(self):
        """write states at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray"""
        return self._results
        # raise NotImplementedError()

    def write_results(self):
        """write results at <root/?/> in model ready format"""
        pass
        # raise NotImplementedError()

    def _update_forcing_1d(self, da, name):
        """ "Set 1D forcing (and remove if da is None) and update config accordingly"""
        sfincs_name = self._1DFORCING.get(name, None)
        if sfincs_name is None:
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
            # reset index dim as these get lost later anyway
            index = xr.IndexVariable(
                da.vector.index_dim, np.arange(da.vector.index.size)
            )
            da[da.vector.index_dim] = index
            self.set_forcing(da, sfincs_name)
            self.set_staticgeoms(da.vector.to_gdf(), name)
            # edit inp file
            self.logger.debug(f"{name} forcing: updating sfincs.inp.")
            for sname in [self._GEOMS[name], self._1DFORCING[name]]:
                self.set_config(f"{sname}file", f"sfincs.{sname}")
        else:  # remove forcing data and sfincs.inp entries
            if name in self._forcing:
                self.logger.debug(f"{name} forcing: removing data.")
                self._forcing.pop(name)
            for sname in [self._GEOMS[name], self._1DFORCING[name]]:
                if f"{sname}file" in self.config:
                    self.logger.debug(
                        f"{name} forcing: removing {sname}file from sfincs.inp."
                    )
                    self._config.pop(f"{sname}file")

    ## model configuration

    def set_crs(self, crs):
        super(SfincsModel, self).set_crs(crs)
        self._config_update()

    def _configread(self, fn):
        return hydromt.config.configread(
            fn, abs_path=False, cf=ConfigParserSfincs, noheader=True
        )

    def _configwrite(self, fn):
        return hydromt.config.configwrite(
            fn, self.config, cf=ConfigParserSfincs, noheader=True
        )

    def _config_update(self):
        """Update geospatial head based on staticmaps"""
        dx, dy = self.res
        west, south, _, _ = self.bounds
        if self.crs is not None:
            self.set_config("epsg", self.crs.to_epsg())
        self.set_config("mmax", self.width)
        self.set_config("nmax", self.height)
        self.set_config("dx", dx)
        self.set_config("dy", -dy)  # dy is always positive
        self.set_config("x0", west)
        self.set_config("y0", south)

    def _dummy_ts(self, gdf, name, fill_value=0):
        tstart = datetime.strptime(self.config["tstart"], self._dtfmt)
        tstop = datetime.strptime(self.config["tstop"], self._dtfmt)
        df = pd.DataFrame(
            index=pd.DatetimeIndex([tstart, tstop]),
            columns=gdf.index.values,
            data=np.full((2, gdf.index.size), fill_value, dtype=np.float32),
        )
        da = GeoDataArray.from_gdf(gdf, df, dims=("time", "index"), name=name)
        return da


def _read_timeseries(fn, tref):
    """Read asc timeseries files such as sfincs.bzs and sfincs.dis
    index is parsed to datetime format assumming seconds from tref.
    """
    if not isinstance(tref, datetime):
        raise ValueError("tref should be datetime.datetime.")
    # header columns start at 1
    df = pd.read_csv(fn, delim_whitespace=True, index_col=0, header=None)
    df.index = pd.to_datetime(df.index.values, unit="s", origin=tref)
    df.columns = df.columns.values
    return df


def _write_timeseries(fn, df, tref, fmt="%7.2f"):
    """Write pandas DataFrame to fixed width timeseries files
    such as sfincs.bzs and sfincs.dis
    index is in seconds from tref
    """
    if not isinstance(tref, datetime):
        raise ValueError("tref should be datetime.datetime.")
    if df.index.size == 0:
        raise ValueError("df does not contain data.")
    data = df.reset_index().values
    data[:, 0] = (df.index - tref).total_seconds()
    w = int(np.floor(np.log10(data[-1, 0]))) + 3
    fmt_lst = [f"%{w}.1f"] + [fmt for _ in range(df.columns.size)]
    fmt_out = " ".join(fmt_lst)
    with open(fn, "w") as f:
        np.savetxt(f, data, fmt=fmt_out)
