import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union


import numpy as np
import geopandas as gpd
import pandas as pd
from scipy import ndimage

from pyflwdir.regions import region_area

from hydromt.model.components import ModelComponent
from hydromt_sfincs import utils

# optional dependency
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.utils import export_image

    HAS_DATASHADER = True

except ImportError:
    HAS_DATASHADER = False

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


_ATTRS = {"mask": {"standard_name": "mask", "unit": "-"}}


class SfincsMask(ModelComponent):
    """SFINCS mask component.

    This component handles the model mask, which defines active and inactive cells in the model grid.
    The mask is stored in the model grid data under the key "mask".
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        # The data for the mask is stored in the model.grid.data["mask"]
        super().__init__(
            model=model,
        )
        # For plotting map overlay (This is the only data that is stored in the object! All other data is stored in the model.grid.data["mask"])
        # self.datashader_dataframe = pd.DataFrame()

    @property
    def data(self):
        """Get the data from the model grid."""
        return self.model.grid.data

    @property
    def empty_mask(self):
        """Get an empty mask with the same shape as the model grid."""
        return self.model.grid.empty_mask

    @property
    def transform(self):
        """Get the affine transform of the model grid."""
        return self.model.grid.transform

    def read(self):
        # TODO discuss what we want to return/read here, pass is not so informative ..
        # The mask values are read when the quadtree grid is read
        pass

    def write(self):
        # The mask values are written when the quadtree grid is written
        pass

    def create(
        self,
        mask: Union[str, Path, gpd.GeoDataFrame] = None,
        include_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        mask_buffer: int = 0,
        zmin: float = None,
        zmax: float = None,
        fill_area: float = 10.0,
        drop_area: float = 0.0,
        connectivity: int = 8,
        all_touched: bool = True,
        reset_mask: bool = True,
    ):
        """Create an integer mask with inactive (mask=0) and active (mask=1) cells, optionally bounded
        by several criteria.

        Parameters
        ----------
        mask: str, Path, gpd.GeoDataFrame, optional
            Path or data source name of polygons to initiliaze active mask with; proceding arguments can be used to include/exclude cells
            If not given, existing mask (if present) used, else mask is initialized empty.
        include_mask, exclude_mask: str, Path, gpd.GeoDataFrame, optional
            Path or data source name of polygons to include/exclude from the active model domain.
            Note that include (second last) and exclude (last) areas are processed after other critera,
            i.e. `zmin`, `zmax` and `drop_area`, and thus overrule these criteria for active model cells.
        mask_buffer: float, optional
            If larger than zero, extend the `include_mask` geometry with a buffer [m],
            by default 0.
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for active model cells.
        fill_area : float, optional
            Maximum area [km2] of contiguous cells below `zmin` or above `zmax` but surrounded
            by cells within the valid elevation range to be kept as active cells, by default 10 km2.
        drop_area : float, optional
            Maximum area [km2] of contiguous cells to be set as inactive cells, by default 0 km2.
        connectivity: {4, 8}
            The connectivity used to define contiguous cells, if 4 only horizontal and vertical
            connections are used, if 8 (default) also diagonal connections.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.
        reset_mask: bool, optional
            If True (default), reset existing mask layer. If False updating existing mask.

        """

        # read geometries
        gdf_mask, gdf_include, gdf_exclude = None, None, None
        bbox = self.model.region.to_crs(4326).total_bounds
        if mask is not None:
            if not isinstance(mask, gpd.GeoDataFrame) and str(mask).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_mask = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=mask), crs=self.model.crs
                )
            else:
                gdf_mask = self.data_catalog.get_geodataframe(mask, bbox=bbox)
            if mask_buffer > 0:  # NOTE assumes model in projected CRS!
                gdf_mask["geometry"] = gdf_mask.to_crs(self.model.crs).buffer(
                    mask_buffer
                )
        if include_mask is not None:
            if not isinstance(include_mask, gpd.GeoDataFrame) and str(
                include_mask
            ).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_include = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=include_mask), crs=self.model.crs
                )
            else:
                gdf_include = self.data_catalog.get_geodataframe(
                    include_mask, bbox=bbox
                )
        if exclude_mask is not None:
            if not isinstance(exclude_mask, gpd.GeoDataFrame) and str(
                exclude_mask
            ).endswith(".pol"):
                gdf_exclude = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=exclude_mask), crs=self.model.crs
                )
            else:
                gdf_exclude = self.data_catalog.get_geodataframe(
                    exclude_mask, bbox=bbox
                )

        # get mask and dep data
        da_mask = self.data["mask"] if "mask" in self.data else None
        da_dep = self.data["dep"] if "dep" in self.data else None

        # if mask already defined and reset_mask is False,
        # use the current active mask as starting point
        da_mask0 = None
        if not reset_mask and da_mask is not None:
            # use current active mask
            da_mask0 = da_mask > 0
        elif gdf_mask is not None:
            # start with active mask within provided region
            da_mask0 = (
                self.empty_mask.raster.geometry_mask(gdf_mask, all_touched=all_touched)
                > 0
            )
        # always intiliaze an inactive mask
        da_mask = self.empty_mask > 0

        latlon = self.model.crs.is_geographic

        if da_dep is None and (zmin is not None or zmax is not None):
            raise ValueError("dep required in combination with zmin / zmax")
        elif da_dep is not None and not da_dep.raster.identical_grid(da_mask):
            raise ValueError("dep does not match regular grid")

        s = None if connectivity == 4 else np.ones((3, 3), int)
        if zmin is not None or zmax is not None:
            _msk = da_dep != da_dep.raster.nodata
            if zmin is not None:
                _msk = np.logical_and(_msk, da_dep >= zmin)
            if zmax is not None:
                _msk = np.logical_and(_msk, da_dep <= zmax)
            if da_mask0 is not None:
                # if mask was provided; keep active mask only within valid elevations
                da_mask = np.logical_and(da_mask0, _msk)
            else:
                # no mask provided; set mask to valid elevations
                da_mask = _msk
        elif zmin is None and zmax is None and da_mask0 is not None:
            # in case a mask/region was provided, but you didn't want to update the mask based on elevation
            # just continue with the provided mask
            da_mask = da_mask0

        if fill_area > 0:
            _msk1 = np.logical_xor(
                da_mask, ndimage.binary_fill_holes(da_mask, structure=s)
            )
            regions = ndimage.measurements.label(_msk1, structure=s)[0]
            # TODO check if region_area works for rotated grids!
            lbls, areas = region_area(regions, self.transform, latlon)
            n = int(sum(areas / 1e6 < fill_area))
            logger.info(f"{n} gaps outside valid elevation range < {fill_area} km2.")
            da_mask = np.logical_or(
                da_mask, np.isin(regions, lbls[areas / 1e6 < fill_area])
            )
        if drop_area > 0:
            regions = ndimage.measurements.label(da_mask.values, structure=s)[0]
            lbls, areas = region_area(regions, self.transform, latlon)
            _msk = np.isin(regions, lbls[areas / 1e6 >= drop_area])
            n = int(sum(areas / 1e6 < drop_area))
            logger.info(f"{n} regions < {drop_area} km2 dropped.")
            da_mask = np.logical_and(da_mask, _msk)

        if gdf_include is not None:
            try:
                _msk = da_mask.raster.geometry_mask(
                    gdf_include, all_touched=all_touched
                )
                da_mask = np.logical_or(da_mask, _msk)  # NOTE logical OR statement
            except:
                logger.debug("No mask cells found within include polygon!")
        if gdf_exclude is not None:
            try:
                _msk = da_mask.raster.geometry_mask(
                    gdf_exclude, all_touched=all_touched
                )
                da_mask = np.logical_and(da_mask, ~_msk)
            except:
                logger.debug("No mask cells found within exclude polygon!")

        # update sfincs mask name, nodata value and crs
        da_mask = da_mask.where(da_mask, 0).astype(np.uint8).rename("mask")
        da_mask.raster.set_nodata(0)
        da_mask.raster.set_crs(self.model.crs)

        # set the mask in the model data
        mname = "mask"
        da_mask.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_mask, name=mname)

        # add msk and ind to config
        self.model.config.update({"indexfile": "sfincs.ind", "mskfile": "sfincs.msk"})

    def set_bounds(
        self,
        btype: str = "waterlevel",
        include_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_mask: Union[str, Path, gpd.GeoDataFrame] = None,
        include_mask_buffer: int = 0,
        zmin: float = None,
        zmax: float = None,
        connectivity: int = 8,
        all_touched: bool = False,
        reset_bounds: bool = False,
    ):
        """Set boundary cells in the model mask.

        The SFINCS model mask defines inactive (mask=0), active (mask=1), and waterlevel boundary (mask=2)
        and outflow boundary (mask=3) cells. Active cells set using the `setup_mask` method,
        while this method sets both types of boundary cells, see `btype` argument.

        Boundary cells at the edge of the active model domain,
        optionally bounded by areas inside the include geomtries, outside the exclude geomtries,
        larger or equal than a minimum elevation threshhold and smaller or equal than a
        maximum elevation threshhold.
        All conditions are combined using a logical AND operation.

        Updates model layers:

        * **mask** map: model mask [-]

        Parameters
        ----------
        btype: {'waterlevel', 'outflow'}
            Boundary type
        include_mask, exclude_mask: str, Path, gpd.GeoDataFrame, optional
            Path or data source name for geometries with areas to include/exclude from
            the model boundary.
        zmin, zmax : float, optional
            Minimum and maximum elevation thresholds for boundary cells.
            Note that when include and exclude areas are used, the elevation range is
            only applied on cells within the include area and outside the exclude area.
        reset_bounds: bool, optional
            If True, reset existing boundary cells of the selected boundary
            type (`btype`) before setting new boundary cells, by default False.
        all_touched: bool, optional
            if True (default) include (or exclude) a cell in the mask if it touches any of the
            include (or exclude) geometries. If False, include a cell only if its center is
            within one of the shapes, or if it is selected by Bresenham's line algorithm.
        connectivity, {4, 8}:
            The connectivity used to detect the model edge, if 4 only horizontal and vertical
            connections are used, if 8 (default) also diagonal connections.
        """

        # get include / exclude geometries
        gdf_include, gdf_exclude = None, None
        bbox = self.model.bbox
        if include_mask is not None:
            if not isinstance(include_mask, gpd.GeoDataFrame) and str(
                include_mask
            ).endswith(".pol"):
                # NOTE polygons should be in same CRS as model
                gdf_include = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=include_mask), crs=self.model.crs
                )
            else:
                gdf_include = self.data_catalog.get_geodataframe(
                    include_mask, bbox=bbox
                )
            if include_mask_buffer > 0:
                if self.model.crs.is_geographic:
                    include_mask_buffer = include_mask_buffer / 111111.0
                gdf_include["geometry"] = gdf_include.to_crs(self.model.crs).buffer(
                    include_mask_buffer
                )
        if exclude_mask is not None:
            if not isinstance(exclude_mask, gpd.GeoDataFrame) and str(
                exclude_mask
            ).endswith(".pol"):
                gdf_exclude = utils.polygon2gdf(
                    feats=utils.read_geoms(fn=exclude_mask), crs=self.model.crs
                )
            else:
                gdf_exclude = self.data_catalog.get_geodataframe(
                    exclude_mask, bbox=bbox
                )

        # get mask and dep data
        if "mask" in self.data:
            da_mask = self.data["mask"]
        else:
            raise ValueError(
                "No mask data found in model, please create a mask first using `SfincsModel.mask.create`"
            )
        da_dep = self.data["dep"] if "dep" in self.data else None

        # check if mask and dep data are compatible
        if da_dep is None and (zmin is not None or zmax is not None):
            raise ValueError("da_dep required in combination with zmin / zmax")
        elif da_dep is not None and not da_dep.raster.identical_grid(da_mask):
            raise ValueError("da_dep does not match regular grid")

        btype = btype.lower()
        bvalues = {"waterlevel": 2, "outflow": 3}
        if btype not in bvalues:
            raise ValueError('btype must be one of "waterlevel", "outflow"')
        bvalue = bvalues[btype]

        if reset_bounds:  # reset existing boundary cells
            logger.debug(f"{btype} (mask={bvalue:d}) boundary cells reset.")
            da_mask = da_mask.where(da_mask != np.uint8(bvalue), np.uint8(1))
            if (
                zmin is None
                and zmax is None
                and gdf_include is None
                and gdf_exclude is None
            ):
                return da_mask

        s = None if connectivity == 4 else np.ones((3, 3), int)
        bounds0 = np.logical_xor(
            da_mask > 0, ndimage.binary_erosion(da_mask > 0, structure=s)
        )
        bounds = bounds0.copy()

        if zmin is not None:
            bounds = np.logical_and(bounds, da_dep >= zmin)
        if zmax is not None:
            bounds = np.logical_and(bounds, da_dep <= zmax)
        if gdf_include is not None:
            da_include = da_mask.raster.geometry_mask(
                gdf_include, all_touched=all_touched
            )
            # bounds = np.logical_or(bounds, np.logical_and(bounds0, da_include))
            bounds = np.logical_and(bounds, da_include)
        if gdf_exclude is not None:
            da_exclude = da_mask.raster.geometry_mask(
                gdf_exclude, all_touched=all_touched
            )
            bounds = np.logical_and(bounds, ~da_exclude)

        # avoid any msk3 cells neighboring msk2 cells
        if bvalue == 3 and np.any(da_mask == 2):
            msk2_dilated = ndimage.binary_dilation(
                (da_mask == 2).values,
                structure=np.ones((3, 3)),
                iterations=1,  # minimal one cell distance between msk2 and msk3 cells
            )
            bounds = bounds.where(~msk2_dilated, False)

        ncells = np.count_nonzero(bounds.values)
        if ncells > 0:
            da_mask = da_mask.where(~bounds, np.uint8(bvalue))

        # update the mask in the model data
        mname = "mask"
        da_mask.attrs.update(**_ATTRS.get(mname, {}))
        self.model.grid.set(da_mask, name=mname)

    def to_gdf(self, option: str = "all") -> gpd.GeoDataFrame:
        """Convert a boolean mask to a GeoDataFrame of polygons.

        Parameters
        ----------
        option: {"all", "active", "wlev", "outflow"}

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame of Points.
        """
        if option == "all":
            da_mask = da_mask != da_mask.raster.nodata
        elif option == "active":
            da_mask = da_mask == 1
        elif option == "wlev":
            da_mask = da_mask == 2
        elif option == "outflow":
            da_mask = da_mask == 3

        indices = np.stack(np.where(da_mask), axis=-1)

        if "x" in da_mask.coords:
            x = da_mask.coords["x"].values[indices[:, 1]]
            y = da_mask.coords["y"].values[indices[:, 0]]
        else:
            x = da_mask.coords["xc"].values[indices[:, 0], indices[:, 1]]
            y = da_mask.coords["yc"].values[indices[:, 0], indices[:, 1]]

        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x, y), crs=da_mask.raster.crs
        )

        if len(points) > 0:
            return gpd.GeoDataFrame(points, crs=da_mask.raster.crs)
        else:
            return None

    def has_open_boundaries(self):
        """Returns True if mask contains open boundaries (mask = 2)"""
        mask = self.data["mask"]
        if mask is None:
            return False
        if np.any(mask == 2):
            return True
        else:
            return False

    def get_datashader_dataframe(self):
        raise NotImplementedError(
            "Datashader dataframe not yet implemented for regular models"
        )

    def clear_datashader_dataframe(self):
        """Clear the datashader dataframe."""
        if hasattr(self, "datashader_dataframe"):
            self.datashader_dataframe = pd.DataFrame()

    def map_overlay(self):
        raise NotImplementedError("Map overlay not yet implemented for regular models")
