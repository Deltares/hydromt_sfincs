import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsDrainageStructures(ModelComponent):
    """SFINCS drainage structures component.

    This component handles reading, writing, and creating drainage structures
    such as pumps, culverts, and valves in a SFINCS model.
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.drn"
        self._data: gpd.GeoDataFrame = None
        super().__init__(
            model=model,
        )

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Drainage structures data, returns geopandas.GeoDataFrame"""
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read: bool = False) -> None:
        """Initialize drainage structures."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: str | Path = None):
        """Read SFINCS drainage structures (.drn) file. Filename is obtained from config if not provided."""

        # check that read mode is on
        self.root._assert_read_mode()

        # get absolute file path and set it in config if drnfile is not None
        abs_file_path = self.model.config.get_set_file_variable(
            "drnfile", value=filename
        )

        # check if abs_file_path is None or does not exist
        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(
                f"Drainage structures file not found: {abs_file_path}"
            )

        # Read input file:
        # TODO we can move the utils to here, since only used here?
        gdf = utils.read_drn(abs_file_path, crs=self.model.crs)

        # Add to self._data
        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS drainage structures (.drn) file,
        and make sure drnfile is in config (if it was not already set)."""

        # check that write mode is on
        self.root._assert_write_mode()

        # check if data present:
        if self.data.empty:
            logger.debug("No drainage structures data available to write.")
            return

        # Set file name and get absolute path
        abs_file_path = self.model.config.get_set_file_variable(
            key="drnfile", value=filename, default="sfincs.drn"
        )

        # Create parent directories if they do not exist
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Change precision of coordinates according to crs
        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # TODO we can move the utils to here, since only used here?
        utils.write_drn(abs_file_path, self.data, fmt=fmt)

        # write also as geojson:
        if self.model.write_gis:
            utils.write_vector(
                self.data,
                name="drn",
                root=join(self.model.root.path, "gis"),
                logger=logger,
            )

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True):
        """Set SFINCS drainage structures.

        Parameters
        ----------
        gdf: geopandas.GeoDataFrame
            Set GeoDataFrame with drainage structures to self.data.
            Note that the gdf should have the same CRS as the model.
        merge: bool
            Merge with existing drainage structures. If False, overwrite existing drainage structures.

        .. note::
            When directly using the set method, the GeoDataFrame needs to be in the same CRS as SFINCS model.
        """

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Drainage structures must be of type LineString.")
        if not gdf.crs == self.model.crs:
            raise ValueError(
                f"Drainage structures CRS {gdf.crs} does not match model CRS {self.model.crs}."
            )

        # Clip geometries outside of model region:
        within = gdf.within(self.model.region.union_all())

        if within.any() == True:
            if within.all() == False:
                # keep geometries that fall within region
                gdf = gdf[within]

                # write away the names of geometries that are removed
                gdf_name = gdf.name[~within]
                logger.info(
                    "Some of the drainage structures fall out of model domain. Removing structures: "
                    + str(gdf_name.values)
                )
        else:
            raise ValueError("None of drainage structures fall within model domain.")

        if merge and not self.data.empty:
            gdf0 = self.data
            # add the new data behind the original
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new drainage structures to existing ones.")

        self._data = gdf  # set gdf in self.data

    @hydromt_step
    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        stype: str = "pump",
        discharge: float = 0.0,
        merge: bool = True,
        **kwargs,
    ):
        """Create drainage structures such as pumps, culverts, or valves (old name: setup_drainage_structures).

        Adds model layer:
        * **drn** geom: drainage pump or culvert

        Parameters
        ----------
        locations : str, Path
            Path, data source name, or geopandas object to structure line geometry file.
            The line should consist of only 2 points (else first and last points are used), ordered from up to downstream.
            The "type" (1 for pump, 2 for culvert and 3 for valve), "par1" ("discharge" also accepted) variables are optional.
            If "type" or "par1" are not provided, they are based on stype or discharge Parameters.
        stype : {'pump', 'culvert', 'valve'}, optional
            Structure type, by default "pump". stype is converted to integer "type" to match with SFINCS expectations.
        discharge : float, optional
            Discharge of the structure, by default 0.0. For culverts and one-way-valves, this is the maximum discharge,
            since actual discharge depends on waterlevel gradient
        merge : bool, optional
            If True, merge with existing drainage locations, by default True.
        """

        stype = stype.lower()
        svalues = {"pump": 1, "culvert": 2, "valve": 3}
        if stype not in svalues:
            raise ValueError('stype must be one of "pump", "culvert", "valve"')
        svalue = svalues[stype]

        # read, clip and reproject
        gdf_structures = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, **kwargs
        ).to_crs(self.model.crs)

        # check if type (int) is present in gdf, else overwrite from args
        # TODO also add check if type is interger?
        if "type" not in gdf_structures:
            gdf_structures["type"] = svalue
        # if discharge is provided, rename to par1
        if "discharge" in gdf_structures:
            gdf_structures = gdf_structures.rename(columns={"discharge": "par1"})

        # add par1, par2, par3, par4, par5 if not present
        # NOTE only par1 is used in the model
        if "par1" not in gdf_structures:
            gdf_structures["par1"] = discharge
        if "par2" not in gdf_structures:
            gdf_structures["par2"] = 0
        if "par3" not in gdf_structures:
            gdf_structures["par3"] = 0
        if "par4" not in gdf_structures:
            gdf_structures["par4"] = 0
        if "par5" not in gdf_structures:
            gdf_structures["par5"] = 0

        # multi to single lines
        lines = gdf_structures.explode(column="geometry").reset_index(drop=True)
        # get start [0] and end [1] points
        endpoints = lines.boundary.explode(index_parts=True).unstack()
        # merge start and end points into a single linestring
        gdf_structures["geometry"] = endpoints.apply(
            lambda x: LineString(x.values.tolist()), axis=1
        )

        # set structures
        self.set(gdf_structures, merge=merge)
        # set config
        self.model.config.set("drnfile", "sfincs.drn")
