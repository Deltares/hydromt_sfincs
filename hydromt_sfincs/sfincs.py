# -*- coding: utf-8 -*-
import os
from os.path import join, isfile, abspath, dirname, basename
import glob
import numpy as np
import logging
from numpy.core.fromnumeric import var
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
        "gauges": "obs",
        "weirs": "weir",
        "thin_dams": "thd",
    }  # parsed to dict of geopandas.GeoDataFrame
    _FORCING = {
        "waterlevel": ("bzs", "bnd"),  #  timeseries, locations tuple
        "discharge": ("dis", "src"),
        "precip": ("precip", None),
        "precip2D": ("netampr", None),
    }  # parsed to dict of hydromt.GeoDataArray if locations else xarray.DataArray
    _MAPS = {
        "elevtn": "dep",
        "mask": "msk",
        "curve_number": "scs",
        "manning": "manning",
        "infiltration": "qinf",
    }  # parsed to hydromt.RasterDataset
    _FOLDERS = []
    _CONF = "sfincs.inp"
    _DATADIR = DATADIR
    _ATTRS = {
        "dep": {"standard_name": "elevation", "unit": "m+ref"},
        "msk": {"standard_name": "mask", "unit": "-"},
        "scs": {
            "standard_name": "potential maximum soil moisture retention",
            "unit": "in",
        },
        "qinf": {"standard_name": "infiltration rate", "unit": "mm.hr-1"},
        "manning": {"standard_name": "manning roughness", "unit": "s.m-1/3"},
        "bzs": {"standard_name": "waterlevel", "unit": "m+ref"},
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
        if write_gis and "gis" not in self._FOLDERS:
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
        reproj_method="bilinear",
    ):
        """Define model region and setup topobathy (depfile) model layer.

        To merge elevation and bathymetry data, elevation data is taken for land and
        bathymetry for sea cells, see ``landmask_fn``. An optional ``merge_buffer``
        can be set to define a buffer aourd the land cells where the bathymetry is
        estimated from a linear interpolation of elevation and bathymetry data.



        Adds model layers:

        * **dep** map: combined elevation/bathymetry [m+ref]

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest, e.g. {'bbox': [xmin, ymin, xmax, ymax]}
            See :py:meth:`~hydromt.cli.parse_region()` for all options
        res : float
            Model resolution [m], by default 100 m.
        crs : str, int, optional
            Model Coordinate Reference System as epsg code, by default 'utm' in which
            case the region centroid UTM zone is used. If None, the basemaps crs is used
            and must be a projected CRS.
        basemaps_fn : str
            Path or data source name for hydrography raster data, by default 'merit_hydro'.

            * Required variables: ['elevtn'].
            * Required variables to delineate a (sub)basin: ['flwdir', 'uparea', 'basins']
        reproj_method: str, optional
            Method used to reproject topobathy data, by default 'bilinear'


        """
        name = self._MAPS["elevtn"]
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
        if crs is None:
            dst_crs = ds_org.raster.crs
        else:
            dst_crs = hydromt.gis_utils.parse_crs(crs, bbox_epsg4326)
        if dst_crs.is_geographic:
            raise ValueError("A projected CRS is required.")
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
        reproj_kwargs = dict(
            dst_res=res, dst_crs=dst_crs, align=True, method=reproj_method
        )
        if da_elv.raster.crs != dst_crs:
            da_elv = (
                da_elv.raster.clip_geom(geom=dst_geom, buffer=20)
                .raster.reproject(**reproj_kwargs)
                .raster.clip_geom(dst_geom, mask=True)
                .raster.mask_nodata()
                .fillna(-9999)  # force nodata value to be -9999
                .round(3)  # mm precision
            )
            da_elv.raster.set_nodata(-9999)
        # make sure orientation is S -> N
        if da_elv.raster.res[1] < 0:
            ycoords_rev = {da_elv.raster.y_dim: list(reversed(da_elv.raster.ycoords))}
            da_elv = da_elv.reindex(ycoords_rev)

        # set staticmaps
        da_elv.attrs.update(**self._ATTRS.get(name, {}))
        self.set_staticmaps(data=da_elv, name=name)
        # update config
        self.update_spatial_attrs()

    def setup_merge_topobathy(
        self,
        topobathy_fn,
        offset_fn=None,
        mask_fn=None,
        offset_constant=0,
        merge_buffer=0,
        reproj_method="bilinear",
    ):
        """Updates the model topobathy data (depfile) with a new topobathy source.

        The current toptobathy data is overwritten with values from the new dataset
        witin the `mask_fn` polygon, or, if not provided, where the current topobathy
        has missing values.

        If `merge_buffer` > 0, values of the new dataset are replaced with
        linearly interpolated values between both sources within the buffer.

        If `offset_fn` is provided, a (spatially varying) offset is applied to the
        new dataset to convert the vertical datum before merging.

        Updates model layers:

        * **dep** map: combined elevation/bathymetry [m+ref]

        Parameters
        ----------
        topobathy_fn : str, optional
            Path or data source name for topobathy raster data.

            * Required variables: ['elevtn']
        mask_fn : str, optional
            Path or data source name for mask polygon. Values from topobathy_fn within this
            geometry are merged with the model topobathy file.
        offset_fn : str, optional
            Difference between vertical reference of the current model topobathy and the new
            datasourcs. The offset is added to the new source before merging.
        merge_buffer : int, optional
            Buffer (number of cells) within the mask region where topobathy
            values are based on linear interpolation for a smooth transition, by default 0.
        reproj_method: str
            Method used to reproject the new source and offset data to the model grid,
            by default 'bilinear'
        """
        name = self._MAPS["elevtn"]
        assert name in self.staticmaps
        da_elv2 = self.data_catalog.get_rasterdataset(
            topobathy_fn, geom=self.region, buffer=10, variables=["elevtn"]
        )
        da_elv = self.staticmaps[name]
        kwargs = dict(reproj_method=reproj_method, merge_buffer=merge_buffer)
        # mask
        if mask_fn is not None:
            gdf_mask = self.data_catalog.get_geodataframe(mask_fn, geom=self.region)
            da_mask = da_elv.raster.geometry_mask(gdf_mask)
            kwargs.update(da_mask=da_mask)
        # offset
        if offset_fn is not None:
            da_offset = self.data_catalog.get_rasterdataset(
                offset_fn, geom=self.region, buffer=10
            )
            # variable name not important, but must be single variable
            assert isinstance(da_offset, xr.DataArray)
            kwargs.update(da_offset=da_offset)
        elif offset_constant > 0:
            kwargs.update(da_offset=offset_constant)
        # merge
        da_dep_merged = workflows.merge_topobathy(da_elv, da_elv2, **kwargs)
        self.set_staticmaps(data=da_dep_merged, name=name)
        return

    @property
    def mask(self):
        """Returns model mask map."""
        name = self._MAPS["mask"]
        da_mask = None
        if name not in self._staticmaps and self._MAPS["elevtn"] in self._staticmaps:
            da_elv = self.staticmaps[self._MAPS["elevtn"]]
            da_mask = (da_elv != da_elv.raster.nodata).astype(np.uint8)
            da_mask.raster.set_nodata(0)
        elif name in self._staticmaps:
            da_mask = self.staticmaps[name]
        return da_mask

    def setup_mask(self, active_mask_fn=None, elv_min=-1, elv_max=None):
        """Creates mask of active model cells.

        Active model cells are based on with valid elevation; within the
        active mask polygon and between minimum and maximum elevation contour lines.
        Note that local sinks (isolated regions with elv < elv_min) are kept as active cells.

        The model mask defines 0) Inactive, 1) active, and 2) waterlevel boundary cells
        and 3) outflow boundary cells. Note that this method does not set the boundary
        cells, use `setup_bounds` istead.

        Sets model layers:

        * **msk** map: model mask [-]

        Parameters
        ----------
        active_mask_fn : str, optional
            Path or data source name for mask polygon. Models cells outside the polygon are set as
            inactive (mask == 0) and within the polygon as active (mask == 1) cells.
        elv_min, elv_max : float, optional
            Minimum and maximum elevation thresholds for active model cells.
        """
        da_mask = self.mask
        if active_mask_fn is not None:
            gdf_active_mask = self.data_catalog.get_geodataframe(
                active_mask_fn, geom=self.region
            )
            da_valid = da_mask.raster.geometry_mask(gdf_active_mask)
            da_mask = da_mask.where(da_valid, np.uint8(0))

        if elv_min is not None or elv_max is not None:
            da_elv = self.staticmaps[self._MAPS["elevtn"]]
            dep_mask = workflows.mask_topobathy(da_elv, elv_min, elv_max)
            da_mask = da_mask.where(dep_mask, np.uint8(0))

        self.set_staticmaps(da_mask, self._MAPS["mask"])
        ncells = np.count_nonzero(da_mask.values)
        self.logger.debug(f"Mask with {ncells:d} active cells set.")

    def setup_bounds(
        self, btype="waterlevel", include_mask_fn=None, exclude_mask_fn=None
    ):
        """Set boundary cells in the model mask.

        Boundary cells are cells at the edge of the active model domain, optionally bounded
        by areas to include or exclude. If no geometries to in- or exclude are provided,
        boundary cells are limited to cells with elevation smaller or equal to zero.
        Currently dynamic water level and outflow boundaries are supported, see `btype` argument.

        The model mask defines 0) Inactive, 1) active, and 2) waterlevel boundary cells
        and 3) outflow boundary cells. Active cells are inferred from valid elevation cells
        if not previously set using the `set_mask` method.

        Updates model layers:

        * **msk** map: model mask [-]

        Parameters
        ----------
        btype: {'waterlevel', 'outflow'}
            Boundary type
        include_mask_fn, exclude_mask_fn: str, optional
            Path or data source name for mask polygon with areas to include/exclude
            from the model boundary.
        """
        btype = btype.lower()
        bvalues = {"waterlevel": 2, "outflow": 3}
        if btype not in bvalues:
            raise ValueError('btype must be one of "waterlevel", "outflow"')
        bvalue = bvalues[btype]
        # read geometries
        gdf_include, gdf_exclude = None, None
        if include_mask_fn:
            gdf_include = self.data_catalog.get_geodataframe(
                include_mask_fn, geom=self.region
            )
        if exclude_mask_fn:
            gdf_exclude = self.data_catalog.get_geodataframe(
                include_mask_fn, geom=self.region
            )
        # mask values
        da_mask = self.mask
        bounds = utils.mask_bounds(da_mask, gdf_include, gdf_exclude)
        # limit to cells with elevation <= 0
        if gdf_include is None and gdf_exclude is None:
            da_elv = self.staticmaps[self._MAPS["elevtn"]]
            bounds = bounds.where(da_elv <= 0, False)
        # update model mask
        ncells = np.count_nonzero(bounds.values)
        if ncells > 0:
            da_mask = da_mask.where(~bounds, np.uint8(bvalue))
            self.set_staticmaps(da_mask, self._MAPS["mask"])
            self.logger.debug(
                f"{ncells:d} {btype} (mask={bvalue:d}) boundary cells set."
            )

    def setup_river_inflow(self, basemaps_fn="merit_hydro", river_upa=25.0):
        """Setup river inflow (source) points where a river enters the model domain.

        NOTE: to ensure a river only enters the model domain once, use the 'interbasin',
        region option in `setup_basemaps`.

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
            # set forcing with dummy timeseries to keep valid sfincs model
            self.set_forcing_1d(xy=gdf_src, name="discharge")

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
            da_mask = self.mask
            gdf_outflw = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(*flwdir.xy(idxs_outflw)), crs=src_crs
            ).to_crs(dst_crs)
            # apply buffer
            gdf_outflw_buf = gpd.GeoDataFrame(
                geometry=gdf_outflw.buffer(outflow_width / 2.0), crs=gdf_outflw.crs
            )
            self.logger.debug(f"{len(idxs_outflw)} river outflow points found.")
            # find intersect of buffer and model grid
            bounds = utils.mask_bounds(da_mask, gdf_include=gdf_outflw_buf)
            # update model mask
            ncells = np.count_nonzero(bounds.values)
            if ncells > 0:
                da_mask = da_mask.where(~bounds, np.uint8(3))
                self.set_staticmaps(da_mask, self._MAPS["mask"])
                self.logger.debug(f"{ncells:d} outflow (mask=3) boundary cells set.")

        # vectorize river
        self.logger.debug(f"Vectorize rivers used for setting outflow points.")
        feats = flwdir.vectorize(mask=np.logical_and(basmsk, rivmsk).values)
        gdf_riv = gpd.GeoDataFrame.from_features(feats, crs=src_crs)
        gdf_riv.index = gdf_riv.index.values + 1  # one based index
        gdf_riv = gdf_riv.to_crs(dst_crs)
        self.set_staticgeoms(gdf_riv, name="rivers_out")

    def setup_cn_infiltration(self, cn_fn="gcn250", antecedent_runoff_conditions="avg"):
        """Setup model potential maximum soil moisture retention map from gridded curve number map.

        Adds model layers:

        * **scs** map: potential maximum soil moisture retention [inch]

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
        # reproject using median
        da_cn = da_org.raster.reproject_like(self.staticmaps, method="med")
        # TODO set CN=100 based on water mask
        # convert to potential maximum soil moisture retention S (1000/CN - 10) [inch]
        da_scs = workflows.cn_to_s(da_cn, self.mask > 0)
        # set staticmaps
        mname = self._MAPS["curve_number"]
        da_scs.attrs.update(**self._ATTRS.get(mname, {}))
        self.set_staticmaps(da_scs, name=mname)
        # update config: remove default infiltration values and set scs map
        self.config.pop("qinf", None)
        self.set_config(f"{mname}file", f"sfincs.{mname}")

    def setup_manning_roughness(self, lulc_fn="vito", map_fn=None):
        """Setup model manning rouchness map from gridded land-use/land-cover map
        and mapping table.

        Adds model layers:

        * **man** map: manning roughness coefficient [s.m-1/3]

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
        # reproject and reclassify
        # TODO use generic names for parameters
        # FIXME use hydromt general version!!
        da_msk = self.mask > 0
        da_man = workflows.landuse(
            da_org, da_msk, map_fn, logger=self.logger, params=["N"]
        )["N"]
        # mask and set precision
        da_man = da_man.where(da_msk, da_man.raster.nodata).round(3)
        # set staticmaps
        mname = self._MAPS["manning"]
        da_man.attrs.update(**self._ATTRS.get(mname, {}))
        self.set_staticmaps(da_man, name=mname)
        # update config: remove default manning values and set maning map
        for v in ["manning_land", "manning_sea", "rgh_lev_land"]:
            self.config.pop(v, None)
        self.set_config(f"{mname}file", f"sfincs.{mname[:3]}")

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
        dataset_fn=None,
        timeseries_fn=None,
        mdt_fn=None,
        buffer=0,
        **kwargs,
    ):
        """Setup waterlevel boundary point locations (bnd) and time series (bzs).

        Use dataset_fn to set the waterlevel boundary from a dataset of point location
        timeseries. The dataset is clipped to the model region plus `buffer` [m], and
        model time based on the model config tstart and tstop entries.

        Use timeseries_fn to set a spatially uniform waterlevel boundary representative
        for all waterlevel boundary cells (msk==2), The timeseries is clipped based on
        the model config tstart and tstop entries. The boundary point location is infered
        from the model mask.

        If timeseries_fn and dataset_fn are both not provided a dummy (h=0) waterlevel
        boundary is set.

        The vertical reference level of the waterlevel data can be corrected to match
        the vertical reference level of the model elevation (dep) layer by adding the
        local `mdt_fn` mean dynamical topography value to the waterlevels.

        Adds model layers:

        * **bnd** geom: waterlevel gauge point locations
        * **bzs** forcing: waterlevel time series [m+ref]

        Parameters
        ----------
        dataset_fn: str, Path
            Path or data source name for geospatial point timeseries file.
            This can either be a netcdf file with geospatial coordinates
            or a combined point location file with a timeseries data csv file
            which can be setup through the data_catalog.yml file.
            See :py:meth:`hydromt.open_geodataset`, for accepted files.

            * Required variables if netcdf: ['waterlevel']
            * Required coordinates if netcdf: ['time', 'index', 'y', 'x']
        timeseries_fn: str, Path
            Path to tabulated timeseries csv file with time index in first column
            and location IDs in the first row,
            see :py:meth:`hydromt.open_timeseries_from_table`, for details.
            Note: tabulated timeseries files cannot yet be set through the data_catalog yml file.
        mdt_fn: str, optional
            Path or data source name for mean dynamic topography gridded data.
            Difference between vertical reference of elevation and waterlevel data,
            Adds to the waterlevel data before merging.

            * Required variables: ['mdt']
        buffer: float
            Buffer around model region from which to select waterlevel gauges

        """
        name = "waterlevel"
        msk2 = self.mask == 2
        if not np.any(msk2).item():
            # No waterlevel boundary remove bnd/bzs from sfincs.inp
            self.logger.warning(
                "No waterlevel boundary cells (msk==2) in model mask. "
                "Update the mask layer first before setting waterlevel timeseries."
            )
            return

        tstart, tstop = self.get_model_time()  # model time
        if dataset_fn is not None:
            # read and clip data in time & space
            da = self.data_catalog.get_geodataset(
                dataset_fn,
                geom=self.region,
                buffer=buffer,
                variables=[name],
                time_tuple=(tstart, tstop),
                **kwargs,
            )
        else:
            # create bnd point on single waterlevel boundary cell
            x, y = self.staticmaps.raster.xy(*np.where(msk2))
            gdf = gpd.GeoDataFrame(
                index=[1], geometry=gpd.points_from_xy(x[[0]], y[[0]]), crs=self.crs
            )
            if timeseries_fn is not None:
                da_ts = hydromt.open_timeseries(timeseries_fn, name=name).sel(
                    time=slice(tstart, tstop)
                )
                assert (
                    da_ts["index"].size == 1
                ), "Uniform waterlevel should contain single time series."
                da = GeoDataArray.from_gdf(gdf, da_ts, index_dim="index")
            else:
                self.set_forcing_1d(xy=gdf, name=name)  # dummy timeseries
                return
        # correct for MDT
        if mdt_fn is not None and isfile(mdt_fn):
            da_mdt = self.data_catalog.get_rasterdataset(
                mdt_fn, geom=self.region, buffer=buffer, variables=["mdt"]
            )
            mdt_pnts = da_mdt.raster.sample(da.vector.to_gdf()).fillna(0)
            da = da + mdt_pnts
            mdt_avg = mdt_pnts.mean().values
            self.logger.debug(f"{name} forcing: applied MDT (avg: {mdt_avg:+.2f})")
        self.set_forcing_1d(ts=da, name=name)

    def setup_q_forcing(self, dataset_fn=None, timeseries_fn=None, **kwargs):
        """Setup discharge boundary point locations (src) and time series (dis).

        Use dataset_fn to set the discharge boundary from a dataset of point location
        timeseries. Only locations within the model domain are selected.

        Use timeseries_fn to set discharge boundary conditions to pre-set (src) locations,
        e.g. after the `setup_river_inflow` method.

        The dataset/timeseries are clipped to the model time based on the model config
        tstart and tstop entries.

        Adds model layers:

        * **src** geom: discharge gauge point locations
        * **dis** forcing: discharge time series [m3/s]

        Parameters
        ----------
        dataset_fn: str, Path
            Path or data source name for geospatial point timeseries file.
            This can either be a netcdf file with geospatial coordinates
            or a combined point location file with a timeseries data csv file
            which can be setup through the data_catalog yml file.
            See :py:meth:`~hydromt.open_geodataset`, for accepted files.

            * Required variables if netcdf: ['discharge']
            * Required coordinates if netcdf: ['time', 'index', 'y', 'x']
        timeseries_fn: str, Path
            Path to tabulated timeseries csv file with time index in first column
            and location IDs in the first row,
            see :py:meth:`~hydromt.open_timeseries_from_table`, for details.
            NOTE: tabulated timeseries files cannot yet be set through the data_catalog yml file.

        """
        name = "discharge"
        fname = self._FORCING[name][0]
        tstart, tstop = self.get_model_time()  # time slice
        if dataset_fn is None and fname not in self.forcing:
            self.logger.warning(
                "No discharge inflow (src) points set: Run ``setup_river_inflow()`` method first or provide locations."
            )
            return
        elif dataset_fn is not None:
            # read and clip data
            da = (
                self.data_catalog.get_geodataset(
                    dataset_fn,
                    geom=self.region,
                    variables=[name],
                    time_tuple=(tstart, tstop),
                    **kwargs,
                )
                .fillna(0.0)
                .rename(fname)
            )
            self.set_forcing_1d(ts=da, name=name)
        elif timeseries_fn is not None:
            # read timeseries data and match with existing gdf
            gdf = self.forcing[fname].vector.to_gdf()
            da_ts = hydromt.open_timeseries_from_table(timeseries_fn, name=name)
            da_ts = da_ts.sel(time=slice(tstart, tstop)).fillna(0.0)
            self.set_forcing_1d(ts=da_ts, xy=gdf, name=name)
        else:
            raise ValueError('Either "dataset_fn" or "timeseries_fn" must be provided.')

    def setup_q_forcing_from_grid(
        self, discharge_fn, locs_fn=None, uparea_fn=None, wdw=1, max_error=0.1
    ):
        """Setup discharge boundary location (src) and timeseries (dis) based on a
        gridded discharge dataset.

        If `locs_fn` is not provided, the discharge source locations are expected to be
        pre-set, e.g. using the `setup_river_inflow` method.

        If an upstream area grid is provided the discharge boundary condition is
        snapped to the best fitting grid cell within a `wdw` neighboring cells.
        The best fit is dermined based on the minimal relative upstream area error if
        an upstream area value is available for the discharge boundary locations;
        otherwise it is based on maximum upstream area.

        Adds model layers:

        * **src** geom: discharge gauge point locations
        * **dis** forcing: discharge time series [m3/s]

        Parameters
        ----------
        discharge_fn: str, Path, optional
            Path or data source name for gridded discharge timeseries dataset.

            * Required variables: ['discharge' (m3/s)]
            * Required coordinates: ['time', 'y', 'x']
        locs_fn: str, Path, optional
            Path or data source name for point location dataset.
            See :py:meth:`~hydromt.open_vector`, for accepted files.

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
        if locs_fn is not None:
            gdf = self.data_catalog.get_geodataframe(
                locs_fn, geom=self.region, assert_gtype="Point"
            ).to_crs(self.crs)
        elif self._GEOMS[name] not in self.staticgeoms:
            gdf = self.staticgeoms[self._GEOMS[name]]
        else:
            self.logger.warning(
                'No discharge inflow points in staticgeoms. Provide locations using "locs_fn" or '
                'run "setup_river_inflow()" method first to determine inflow locations.'
            )
            return
        # read data
        ds = self.data_catalog.get_rasterdataset(
            discharge_fn,
            geom=self.region,
            buffer=1,
            time_tuple=self.get_model_time(),  # model time
            variables=[name],
            single_var_as_array=False,
        )
        if uparea_fn is not None:
            da_upa = self.data_catalog.get_rasterdataset(
                uparea_fn, geom=self.region, buffer=1, variables=["uparea"]
            )
            rm = {
                da_upa.raster.x_dim: ds.raster.x_dim,
                da_upa.raster.y_dim: ds.raster.y_dim,
            }
            ds = xr.merge([ds, da_upa.rename(rm)])

        da_q = workflows.snap_discharge(
            ds=ds,
            gdf=gdf,
            wdw=wdw,
            max_error=max_error,
            uparea_name="uparea",
            discharge_name=name,
            logger=self.logger,
        )

        # merge location data data
        da_q = GeoDataArray.from_gdf(gdf.loc[da_q.index, :], da_q, index_dim="index")
        # update forcing
        self.set_forcing_1d(name=name, da=da_q)

    def setup_p_forcing_from_grid(
        self, precip_fn=None, dst_res=None, aggregate=False, **kwargs
    ):
        """Setup precipitation forcing from a gridded spatially varying data source.

        If aggregate is True, spatially uniform precipitation forcing is added to
        the model based on the mean precipitation over the model domain.
        If aggregate is False, distributed precipitation is added to the model as netcdf file.
        The data is reprojected to the model CRS (and destination resolution `dst_res` if provided).

        Adds one of these model layer:

        * **netamprfile** forcing: distributed precipitation [mm/hr]
        * **precipfile** forcing: uniform precipitation [mm/hr]

        Parameters
        ----------
        precip_fn, str, Path
            Path to precipitation rasterdataset netcdf file.

            * Required variables: ['precip' (mm)]
            * Required coordinates: ['time', 'y', 'x']

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
        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=self.region,
            buffer=2,
            time_tuple=self.get_model_time(),
            variables=[variable],
        )

        # aggregate or reproject in space
        if aggregate:
            stat = aggregate if isinstance(aggregate, str) else "mean"
            self.logger.debug(f"Aggregate {variable} using {stat}.")
            precip_out = precip.raster.zonal_stats(self.region, stats=stat)[
                f"precip_{stat}"
            ]
            precip_out = precip_out.where(precip_out >= 0, 0).fillna(0).squeeze()
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
        fname = self._FORCING[variable][0]
        fname2 = self._FORCING[f"{variable}2D"][0]
        if aggregate:
            # remove netamprfile
            self._forcing.pop(fname2, None)
            self._config.pop(f"{fname2}file", None)
            # precipfile = sfincs.precip
            fn_out = f"sfincs.{fname}"
        else:
            # remove precipfile
            self._forcing.pop(fname, None)
            self._config.pop(f"{fname}file", None)
            # netamprfile = precip.nc
            fn_out = f"{variable}.nc"
            fname = fname2
        precip_out.attrs.update(**self._ATTRS.get(fname, {}))
        self.set_config(f"{fname}file", fn_out)
        self.set_forcing(precip_out, name=fname)

    def setup_p_forcing(self, precip_fn=None, **kwargs):
        """Setup spatially uniform precipitation forcing (precip).

        Adds model layers:

        * **precipfile** forcing: uniform precipitation [mm/hr]

        Parameters
        ----------
        precip_fn, str, Path
            Path to tabulated timeseries csv file with time index in first column
            and location IDs in the first row,
            see :py:meth:`~hydromt.open_timeseries_from_table`, for details.
            Note: tabulated timeseries files cannot yet be set through the data_catalog yml file.
        """
        ts = hydromt.open_timeseries_from_table(precip_fn, **kwargs)
        self.set_forcing_1d(name="precip", ts=ts)
        # remove netamprfile
        fname2 = self._FORCING["precip2D"][0]
        self._forcing.pop(fname2, None)
        self._config.pop(f"{fname2}file", None)

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
            # update missing attributes for plot labels
            for name in self.forcing:
                attrs = self._ATTRS.get(name, {})
                self.forcing[name].attrs.update(**attrs)
            fig, axes = plots.plot_forcing(self.forcing, **kwargs)

            # set xlim to model tstart - tend
            tstart, tstop = self.get_model_time()
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
        plot_bounds: bool = True,
        bmap: str = "sat",
        zoomlevel: int = 11,
        figsize: Tuple[int] = None,
        geoms: List[str] = None,
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
            Add shade to variable (only for variable = 'dep'), by default True
        plot_bounds : bool, optional
            Add waterlevel (msk=2) and open (msk=3) boundary conditions to plot.
        bmap : {'sat', ''}
            background map, by default "sat"
        zoomlevel : int, optional
            zoomlevel, by default 11
        figsize : Tuple[int], optional
            figure size, by default None
        geoms : List[str], optional
            list of model geometries to plot, by default all model geometries.
        geom_kwargs : Dict of Dict, optional
            Model geometry styling per geometry, passed to geopands.GeoDataFrame.plot method.
            For instance: {'src': {'markersize': 30}}.
        legend_kwargs : Dict, optional
            Legend kwargs, passed to ax.legend method.

        Returns
        -------
        fig, axes
            Model fig and ax objects
        """
        import matplotlib.pyplot as plt

        # combine staticgeoms and forcing locations
        sg = self.staticgeoms.copy()
        for fname, gname in self._FORCING.values():
            if fname in self._forcing and gname is not None:
                sg.update({gname: self._forcing[fname].vector.to_gdf()})

        # make sure staticmaps are set
        if self._MAPS["mask"] not in self.staticmaps:
            self.set_staticmaps(self.mask, self._MAPS["mask"])

        fig, ax = plots.plot_basemap(
            self.staticmaps,
            staticgeoms=sg,
            variable=variable,
            shaded=shaded,
            plot_bounds=plot_bounds,
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
        mvs = {"msk": 0}
        data_vars = {}
        for name, mname in self._MAPS.items():
            if f"{mname}file" in self.config:
                fn = self.get_config(f"{mname}file", abs_path=True)
                if not isfile(fn):
                    self.logger.warning(f"{mname}file not found at {fn}")
                    continue
                dtype = dtypes.get(mname, "f4")
                mv = mvs.get(mname, -9999.0)
                data = utils.read_binary_map(fn, ind, shape, mv, dtype)
                data_vars.update({mname: (data, mv)})

        # create dataset and set as staticmaps
        ds = RasterDataset.from_numpy(
            data_vars=data_vars,
            transform=transform,
            crs=crs,
        )
        for name in ds.data_vars:
            ds[name].attrs.update(**self._ATTRS.get(name, {}))
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

        # make sure a mask if present
        da_mask = self.mask
        if self._MAPS["mask"] not in self.staticmaps:
            self.set_staticmaps(da_mask, self._MAPS["mask"])
        self.logger.debug("Write binary map indices based on mask.")
        msk = da_mask.values
        fn_ind = self.get_config("indexfile", abs_path=True)
        utils.write_binary_map_index(fn_ind, msk=msk)

        dvars = self.staticmaps.raster.vars
        self.logger.debug(f"Write binary map files: {dvars}.")
        dtypes = {"msk": "u1"}  # default to f4
        for mname in dvars:
            if f"{mname}file" not in self.config:
                self.set_config(f"{mname}file", f"sfincs.{mname}")
            fn_out = self.get_config(f"{mname}file", abs_path=True)
            utils.write_binary_map(
                fn_out,
                ds_out[mname].values,
                msk=msk,
                dtype=dtypes.get(mname, "f4"),
            )

        if self._write_gis:
            self.write_raster("staticmaps")

    def read_staticgeoms(self):
        """Read geometry files if and save to `staticgeoms` attribute.
        Known geometry files mentioned in the sfincs.inp configuration file are read,
        including: bnd/src/obs xy files and thd/weir structure files.

        If other geojson files are present in a "gis" subfolder folder, those are read as well.
        """
        if not self._write:
            self._staticgeoms = {}  # fresh start in read-only mode
        # read _GEOMS model files
        for gname in self._GEOMS.values():
            if f"{gname}file" in self.config:
                fn = self.get_config(f"{gname}file", abs_path=True)
                if fn is None:
                    continue
                elif not isfile(fn):
                    self.logger.warning(f"{gname}file not found at {fn}")
                    continue
                if gname in ["thd", "weir"]:
                    struct = utils.read_structures(fn)
                    gdf = utils.structures2gdf(struct, crs=self.crs)
                else:
                    gdf = utils.read_xy(fn, crs=self.crs)
                self.set_staticgeoms(gdf, name=gname)
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
            for gname, gdf in self.staticgeoms.items():
                if gname in self._GEOMS.values():
                    if f"{gname}file" not in self.config:
                        self.set_config(f"{gname}file", f"sfincs.{gname}")
                    fn = self.get_config(f"{gname}file", abs_path=True)
                    if gname in ["thd", "weir"]:
                        struct = utils.gdf2structures(gdf)
                        utils.write_structures(fn, struct, stype=gname)
                    else:
                        utils.write_xy(fn, gdf, fmt="%8.2f")
            if self._write_gis:
                self.write_vector()

    def read_forcing(self):
        """Read forcing files and save to `forcing` attribute.
        Known forcing files mentioned in the sfincs.inp configuration file are read,
        including: bzd/dis/precip ascii files and the netampr netcdf file.
        """
        if not self._write:
            # start fresh in read-only mode
            self._forcing = {}
        tref = utils.parse_datetime(self.config["tref"])
        for name, (fname, gname) in self._FORCING.items():
            fn = self.get_config(f"{fname}file", abs_path=True)
            if fn is None:
                continue
            elif not isfile(fn):
                self.logger.warning(f"{fname}file not found at {fn}")
                continue
            # read forcing
            if "net" in fname:
                da = xr.open_dataarray(fn, chunks={"time": 24})  # lazy
                self.set_forcing(da, name=fname)

            else:
                df = utils.read_timeseries(fn, tref)
                if gname is not None:  # read bzd/src locations
                    fn_geom = self.get_config(f"{gname}file", abs_path=True)
                    if not isfile(fn):
                        self.logger.warning(f"{gname}file not found at {fn_geom}")
                        continue
                    gdf = utils.read_xy(fn_geom, crs=self.crs)
                else:
                    df = df[df.columns[0]]  # to series for spatially uniform forcing
                    gdf = None
                self.set_forcing_1d(ts=df, xy=gdf, name=name)

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
            names = {f[0]: f[1] for f in self._FORCING.values()}
            for fname in self._forcing:
                if fname not in names:
                    logger.warning(f"{fname} forcing unknown and skipped.")
                    continue
                if f"{fname}file" not in self.config:
                    self.set_config(f"{fname}file", f"sfincs.{fname}")
                fn = self.get_config(f"{fname}file", abs_path=True)
                da = self._forcing[fname]
                if "net" in fname:  # spatially distributed forcing
                    da.to_netcdf(fn, encoding=encoding)
                else:
                    if len(da.dims) == 2:  # forcing at point locations
                        df = da.to_series().unstack(0)
                        gname = names[fname]
                        if gname is None:
                            raise ValueError(f"Locations missing for {fname}")
                        gdf = self._forcing[fname].vector.to_gdf()
                        if f"{gname}file" not in self.config:
                            self.set_config(f"{gname}file", f"sfincs.{gname}")
                        fn_xy = self.get_config(f"{gname}file", abs_path=True)
                        utils.write_xy(fn_xy, gdf, fmt="%8.2f")
                    else:  # spatially uniform forcing
                        df = da.to_series().to_frame()
                    utils.write_timeseries(fn, df, tref)

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
                zsini = zsini.where(self.mask != 0, -9999)
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
            self.write_raster("states")

    def read_results(self, chunksize=100, drop=["crs", "sfincsgrid"], **kwargs):
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
                fn_map,
                crs=self.crs,
                chunksize=chunksize,
                drop=drop,
                logger=self.logger,
                **kwargs,
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

    def write_raster(
        self, variables=["staticmaps", "states", "results.hmax"], root=None, **kwargs
    ):
        """Write model 2D raster variables to geotiff files.

        NOTE: these files are not used by the model by just saved for visualization/
        analysis purposes.

        Parameters
        ----------
        variables: str, list, optional
            Model variables are a combination of attribute and layer (optional) using <attribute>.<layer> syntax.
            Known ratster attributes are ["staticmaps", "states", "results"].
            Different variables can be combined in a list.
            By default, variables is ["staticmaps", "states", "results.hmax"]
        root: Path, str, optional
            The output folder path. If None it defautls to the <model_root>/gis folder (Default)
        kwargs:
            Key-word arguments passed to hydromt.RasterDataset.to_raster(driver='GTiff').
        """
        kwargs.update(driver="GTiff")
        # check variables
        if isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list):
            raise ValueError(f'"variables" should be a list, not {type(list)}.')
        # check root
        if root is None:
            root = join(self.root, "gis")
        if not os.path.isdir(root):
            os.makedirs(root)
        # save to file
        for var in variables:
            vsplit = var.split(".")
            attr = vsplit[0]
            obj = getattr(self, f"_{attr}")
            if obj is None or len(obj) == 0:
                continue  # empty
            self.logger.debug(f"Write GIS files for {var} to 'gis' subfolder")
            layers = vsplit[1:] if len(vsplit) >= 2 else list(obj.keys())
            for layer in layers:
                if layer not in obj:
                    self.logger.warning(f"Variable {attr}.{layer} not found: skipping.")
                    continue
                da = obj[layer]
                if len(da.dims) != 2 or "time" in da.dims:
                    continue
                if da.raster.res[1] > 0:  # make sure orientation is N->S
                    yrev = list(reversed(da.raster.ycoords))
                    da = da.reindex({da.raster.y_dim: yrev})
                da.raster.to_raster(join(root, f"{layer}.tif"), **kwargs)

    def write_vector(self, variables=None, root=None, **kwargs):
        """Write model vector (staticgeoms) variables to geojson files.

        NOTE: these files are not used by the model by just saved for visualization/
        analysis purposes.

        Parameters
        ----------
        variables: str, list, optional
            Staticgeoms variables. By bedault all staticgeoms are saved.
        root: Path, str, optional
            The output folder path. If None it defautls to the <model_root>/gis folder (Default)
        kwargs:
            Key-word arguments passed to geopandas.GeoDataFrame.to_file(driver='GeoJSON').
        """
        kwargs.update(driver="GeoJSON")  # fixed
        # check variables
        if variables is None or variables == "staticgeoms":
            variables = list(self._staticgeoms.keys())
        elif isinstance(variables, str):
            variables = [variables]
        if not isinstance(variables, list):
            raise ValueError(f'"variables" should be a list, not {type(list)}.')
        # check root
        if root is None:
            root = join(self.root, "gis")
        if not os.path.isdir(root):
            os.makedirs(root)
        # save to file
        for var in variables:
            if var not in self._staticgeoms:
                self.logger.warning(f'Var "{var}" not found in staticgeoms: skipping.')
                continue
            self._staticgeoms[var].to_file(join(root, f"{var}.geojson"), **kwargs)

    def set_staticmaps(self, data, name=None):
        """Add data to staticmaps.

        All layers of staticmaps must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to staticmaps
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            or to select a variable from a Dataset.
        """
        # make sure data is in S -> N orientation
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            if data.raster.res[1] < 0:
                data = data.reindex(
                    {data.raster.y_dim: list(reversed(data.raster.ycoords))}
                )
        super().set_staticmaps(data, name)

    def set_forcing_1d(self, name, ts=None, xy=None):
        """Set 1D forcing and update staticgoms and config accordingly.

        For waterlevel and discharge forcing point locations are required to set the
        combined src/dis and bnd/bzs files. If only point locations (and no timeseries)
        are given a dummy timeseries with zero values is set.

        If ts and xy are both None, the

        Parameters
        ----------
        name: {'waterlevel', 'discharge', 'precip'}
            Name of forcing type.
        ts: pandas.DataFrame, xarray.DataArray
            Timeseries data. If DataArray it should contain time and index dims; if
            DataFrame the index should be a datetime index and the columns the location
            index.
        xy: geopandas.GeoDataFrame
            Forcing point locations
        """
        fname, gname = self._FORCING.get(name, (None, None))
        if fname is None:
            names = [f[0] for f in self._FORCING.values() if "net" not in f[0]]
            raise ValueError(f'Unknown forcing "{name}", select from {names}')
        # sort out ts and xy types
        if isinstance(ts, (pd.DataFrame, pd.Series)):
            assert np.dtype(ts.index).type == np.datetime64
            ts.index.name = "time"
            if isinstance(ts, pd.DataFrame):
                ts.columns.name = "index"
                ts = xr.DataArray(ts, dims=("time", "index"), name=fname)
            else:  # spatially uniform forcing
                ts = xr.DataArray(ts, dims=("time"), name=fname)
        if isinstance(xy, gpd.GeoDataFrame):
            if ts is not None:
                ts = GeoDataArray.from_gdf(xy, ts, index_dim="index")
            else:
                ts = self._dummy_ts(xy, name, fill_value=0)  # dummy timeseries
        if not isinstance(ts, xr.DataArray):
            raise ValueError(
                f"{name} forcing: Unknown type for ts {type(ts)} should be xarray.DataArray."
            )
        # check if locations (bzs / dis)
        if gname is not None:
            assert len(ts.dims) == 2
            # make sure time is on last dim
            ts = ts.transpose(ts.vector.index_dim, ts.vector.time_dim)
            # set crs
            if ts.vector.crs is None:
                ts.vector.set_crs(self.crs.to_epsg())
            elif ts.vector.crs != self.crs:
                ts = ts.vector.to_crs(self.crs.to_epsg())
            # fix order based on x_dim after setting crs (for comparibility between OS)
            ts = ts.sortby(ts.vector.x_dim, ascending=True)
            # reset index
            dim = ts.vector.index_dim
            ts[dim] = xr.IndexVariable(dim, np.arange(1, ts[dim].size + 1, dtype=int))
            n = ts.vector.index.size
            self.logger.debug(f"{name} forcing: setting {gname} data for {n} points.")
        else:
            if not (len(ts.dims) == 1 and "time" in ts.dims):
                raise ValueError(
                    f"{name} forcing: uniform forcing should have single 'time' dimension."
                )

        # set forcing
        self.logger.debug(f"{name} forcing: setting {fname} data.")
        ts.attrs.update(**self._ATTRS.get(fname, {}))
        self.set_forcing(ts, fname)
        # edit inp file
        if self._write:
            self.logger.debug(f"{name} forcing: updating sfincs.inp.")
            self.set_config(f"{fname}file", f"sfincs.{fname}")
            if gname is not None:
                self.set_config(f"{gname}file", f"sfincs.{gname}")

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

    def get_model_time(self):
        """Return (tstart, tstop) tuple with parsed model start and end time"""
        tstart = utils.parse_datetime(self.config["tstart"])
        tstop = utils.parse_datetime(self.config["tstop"])
        return tstart, tstop

    ## helper method

    def _dummy_ts(self, gdf, name, fill_value=0):
        df = pd.DataFrame(
            index=pd.DatetimeIndex(list(self.get_model_time())),
            columns=gdf.index.values,
            data=np.full((2, gdf.index.size), fill_value, dtype=np.float32),
        )
        ts = GeoDataArray.from_gdf(gdf, df, dims=("time", "index"), name=name)
        return ts
