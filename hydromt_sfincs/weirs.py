import logging
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import TYPE_CHECKING, Union, List
import os
from os.path import abspath, join, exists

from hydromt.model.components import ModelComponent
from hydromt.model import Model
from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(__name__)


class SfincsWeirs(ModelComponent):
    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.weir"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Weirs lines data.

        Return geopandas.GeoDataFrame
        """
        if self._data is None:
            self._initialize()
        return self._data

    # %% core HydroMT-SFINCS functions:
    # _initialize
    # read
    # write
    # set
    # create
    # delete
    # clear

    def _initialize(self, skip_read=False) -> None:
        """Initialize weir lines."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS weir (*.weir) file."""

        # Check that read mode is on
        self.root._assert_read_mode()

        # get absolute file path and set it in config if weirfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "weirfile", value=filename)

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(f"Weir file not found: {abs_file_path}")
        
        # Read weir file:
        struct = utils.read_geoms(abs_file_path)  # =utils.py function
        gdf = utils.linestring2gdf(struct, crs=self.model.crs)  # =utils.py function

        # Add to self._data
        self.set(gdf, merge=False)  

    def write(self, filename: str | Path = None):
        """Write SFINCS weir (*.weir) file,
        and set weirfile in config (if it was not already set)"""
        
        # check that write mode is on
        self.root._assert_write_mode()

        # check if data present:
        if self.data.empty:
            logger.debug("No weir data available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            key="weirfile",
            value=filename,
            default="sfincs.weir",
        )

        # change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # Get linestring geometries from gdf
        struct = utils.gdf2linestring(self.data)

        # Write to weirfile
        utils.write_geoms(abs_file_path, struct, stype="weir", fmt=fmt)

        # write also as geojson:
        if self.model.write_gis:
            root = join(self.model.root.path, "gis")

            if not os.path.isdir(root):
                os.makedirs(root)

            self.data.to_file(join(root, f"weir.geojson"), driver="GeoJSON")

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SFINCS weir lines.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with weir lines to self.data
        merge: bool
            Merge with existing weir. If False, overwrite existing weirs.
        **NOTE** - coordinates of LineString geometries in GeoDataFrame need to be in the same CRS as SFINCS model.
        """
        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Weirs must be of type LineString.")

        # Check if any of the cross sections fall completely outside the model domain
        # If so, give a warning and remove these lines
        outside = gdf.disjoint(self.model.region)
        if outside.any():
            logger.warning(
                "Some weirs fall outside model domain. Removing these lines."
            )
            gdf = gdf[~outside]

        # Check if there are any cross sections left
        if gdf.empty:
            # logger.warning("All thin dams fall outside model domain!")
            # return
            raise ValueError("All weirs fall outside model domain!")

        if merge and self.data is not None:
            gdf0 = self.data
            # add the new data behind the original
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new weirs to existing ones.")

        self._data = gdf  # set gdf in self._data

    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        dep: Union[str, Path, xr.DataArray] = None,
        buffer: float = None,
        dz: float = None,
        merge: bool = True,
        **kwargs,
    ):
        """Create model weir lines.
        (old name: setup_structures)

        If elevation 'z' at weir locations is not provided, it can be calculated 
        from the model elevation directly (dep supplied, but not dz),
        or from the model elevation plus an additional set elevation 'dz' 
        (dep & dz supplied).

        Adds model layers:

        * **weir** geom: weir lines

        Arguments
        ---------
        locations: str, Path, gpd.GeoDataFrame
            Path, data source name, or geopandas object for weir lines.
        dep : str, Path, xr.DataArray, optional
            Data source name, Path, or xarray raster object ('elevtn') describing the depth in an
            alternative resolution which is used for sampling the weir.
            **NOTE** - currently, you can only supply one datasource for dep, 
                or use the -courser- active dep data in self.grid.data if dep not provided,
                but not your whole datasets_dep list!
            **NOTE** Tip: use fine resolution dep_subgrid.tif for merged high-res data 
                in case of using multiple elevation datasets.
        buffer : float, optional
            If provided, describes the distance from the centerline to the foot of the structure.
            This distance is supplied to the raster.sample as the window (wdw).
        dz: float, optional
            If provided, for weir structures the z value is calculated from
            the model elevation (dep) plus dz.
        merge: bool, optional
            If True, merge the new weir lines with the existing ones. By default True.
        """
        gdf = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, **kwargs
        ).to_crs(self.model.crs)

        # make sure MultiLineString are converted to LineString
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Weirs must be of type LineString.")
        
        # expected columns in gdf
        cols = {
            "weir": ["name", "z", "par1", "geometry"],
        }

        # keep relevant columns
        gdf = gdf[[c for c in cols["weir"] if c in gdf.columns]]

        # check if z values are provided or can be calculated
        if not "z" in gdf.columns and (dep is None and dz is None):
            raise ValueError(
                "Weir structure requires z values, or 'dep' or 'dz' input to determine these on the fly."
            )
        elif dep is not None or dz is not None:
        
            # determine elevation from dep and dz, if data parsed
            gdf = self.determine_weir_elevation(gdf, dep, buffer, dz)
            # if dep is not provided, the active dep data in self.grid.data is loaded,
            # within function determine_weir_elevation            
            logger.info("Determined elevations for weir based on elevation data.")

        self.set(gdf, merge)

    def delete(
        self,
        index: Union[list, int],
    ):
        """Remove one or more weir.

        Arguments
        ---------
        index: list, int
            Specify thin dams to be dropped from GeoDataFrame.
            If int, drop a single weir based on index.
            If list, drop multiple weir based on index.
        """
        # Turn int or str into list
        if type(index) == int:
            index = [index]

        # Check that any integer in list is not larger than the number of lines
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")

        # Drop lines from GeoDataFrame
        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from weirs")

        # Check if any cross sections are left
        if self.data.empty:
            logger.warning("All weirs have been removed!")
            # Set crsfile to None
            self.model.config.set("weirfile", None)

    def clear(self):
        """Clean GeoDataFrame with weirs."""
        self._data = gpd.GeoDataFrame()
        # Set weirfile to None in config
        self.model.config.set("weirfile", None) #FIXME - TL: do we want that?        

    # %% HydroMT-SFINCS focused additional functions:
    # determine_weir_elevation

    def determine_weir_elevation(  # FIXME - TL: should this be in utils.py or not?
        self,
        gdf: gpd.GeoDataFrame,
        dep: Union[str, Path, xr.DataArray] = None,
        buffer: float = None,
        dz: float = None,
    ):
        """Determine z values for weir structures.
        Called by .create() function if dep (/and dz) are provided.        
        """
        # taken from old 'sfincs.py'>setup_structures function

        structs = utils.gdf2linestring(gdf)  # check if it parsed correct

        # get elevation data either from model itself, or separate input
        if dep is None or dep == "dep":
            assert "dep" in self.model.grid.data, "dep layer not found"
            elv = self.model.grid.data["dep"]
        else:
            elv = self.data_catalog.get_rasterdataset(
                dep, geom=self.model.region, buffer=5, variables=["elevtn"]
            )

        # calculate window size from buffer
        if buffer is not None:
            res = abs(elv.raster.res[0])
            if elv.raster.crs.is_geographic:
                res = res * 111111.0
            window_size = int(np.ceil(buffer / res))
        else:
            window_size = 0
        logger.debug(f"Sampling elevation with window size {window_size}")

        # interpolate dep data to points of weirs
        structs_out = []
        for s in structs:
            pnts = gpd.points_from_xy(x=s["x"], y=s["y"])
            zb = elv.raster.sample(
                gpd.GeoDataFrame(geometry=pnts, crs=self.model.crs), wdw=window_size
            )
            if zb.ndim > 1:
                zb = zb.max(axis=1)

            if zb.isnull().any():

                # get id of nan point
                nan_id = zb.isnull().idxmax().values

                # Interpolate missing values
                zb = zb.interpolate_na(dim="index", method='nearest')

                xtmp = s["x"][nan_id]
                ytmp = s["y"][nan_id]

                logger.warning(f"Weir point {xtmp} {ytmp} has no elevation data. Filled now with nearest non-NaN value. Please check your input!")

                if zb.isnull().any():
                    # might still fail if first or last point is NaN, because then we need to extrapolate

                    # Forward fill to handle NaN at the ends
                    zb = zb.ffill(dim="index")
                    
                    # Backward fill to handle NaN at the ends
                    zb = zb.bfill(dim="index")

                    if zb.isnull().any():
                        # if still didn't work, raise error        
                        raise ValueError(
                            "Filling NaN values failed for weirs "
                        )

            s["z"] = zb.values

            # in case of dz, add this to the elevation
            if dz is not None:
                s["z"] += float(dz)

            structs_out.append(s)

        gdf = utils.linestring2gdf(structs_out, crs=self.model.crs)

        return gdf

    # %% DDB GUI focused additional functions:
    # snap_to_grid
    # list_names

    def snap_to_grid(self):
        snap_gdf = self.model.grid.snap_to_grid(
            self.gdf
        )  # FIXME - snap_to_grid should be function in grid.py!
        return snap_gdf

    def list_names(self):
        """Give list of names of cross sections."""
        if self.data.empty:
            return []
        names = list(self.data["name"])
        return names