"""Runup gauge component for SFINCS models.

Manages wave run-up measurement lines stored as LineString geometries in a
GeoDataFrame, read from and written to a SFINCS ``.rug`` file.
"""

import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(__name__)


class SfincsRunupGauges(ModelComponent):
    """SFINCS Runup Gauges Component.

    Handles reading, writing, and creation of runup gauge lines, which are
    used to measure wave run-up along cross-shore transects in the SFINCS model.

    The data is stored as a GeoDataFrame containing LineString geometries, and
    written to an ascii SFINCS runup gauges (``.rug``) file.
    """

    def __init__(self, model: "SfincsModel") -> None:
        """Initialize the runup gauges component.

        Parameters
        ----------
        model : SfincsModel
            Parent model instance.
        """
        self._filename: str = "sfincs.rug"
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model)

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Runup gauge lines as a GeoDataFrame."""
        if self._data is None:
            self._initialize()
        return self._data

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Runup gauge lines as a GeoDataFrame (alias for ``data``)."""
        return self.data

    @property
    def nr_lines(self) -> int:
        """Return the number of runup gauge lines."""
        if hasattr(self.data, "index"):
            return len(self.data.index)
        return 0

    @property
    def list_names(self) -> List[str]:
        """Return list of runup gauge names."""
        if self.nr_lines == 0:
            return []
        return list(self.data["name"])

    def _initialize(self, skip_read: bool = False) -> None:
        """Initialize runup gauge data."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def read(self, filename: Union[str, Path] = None) -> None:
        """Read SFINCS runup gauge (``.rug``) file.

        Parameters
        ----------
        filename : str or Path, optional
            File path. Obtained from config if not provided.
        """
        self.root._assert_read_mode()

        abs_file_path = self.model.config.get_set_file_variable(
            "rugfile", value=filename
        )

        if abs_file_path is None:
            return
        if not abs_file_path.exists():
            raise FileNotFoundError(f"Runup gauges file not found: {abs_file_path}")

        struct = utils.read_geoms(abs_file_path)
        gdf = utils.linestring2gdf(struct, crs=self.model.crs)

        self.set(gdf, merge=False)

    def write(self, filename: Union[str, Path] = None) -> None:
        """Write SFINCS runup gauge (``.rug``) file.

        Parameters
        ----------
        filename : str or Path, optional
            Output file path. Defaults to ``sfincs.rug``.
        """
        if self.data.empty:
            logger.debug("No runup gauges available to write.")
            return

        abs_file_path = self.model.config.get_set_file_variable(
            "rugfile",
            value=filename,
            default="sfincs.rug",
        )

        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        struct = utils.gdf2linestring(self.data)
        utils.write_geoms(abs_file_path, struct, stype="rug", fmt=fmt)

        if self.model.write_gis:
            utils.write_vector(
                self.data,
                name="rug",
                root=join(self.root.path, "gis"),
                logger=logger,
            )

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True) -> None:
        """Set runup gauge lines.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with LineString geometries in the model CRS.
        merge : bool, optional
            Merge with existing runup gauges. If ``False``, overwrite.
        """
        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Runup gauges must be of type LineString.")
        if not gdf.crs == self.model.crs:
            raise ValueError(
                f"Runup gauge CRS {gdf.crs} does not match model CRS {self.model.crs}."
            )

        outside = gdf.disjoint(self.model.region.union_all())
        if outside.any():
            logger.warning(
                "Some runup gauges fall outside model domain. Removing these lines."
            )
            gdf = gdf[~outside]

        if gdf.empty:
            raise ValueError("All runup gauges fall outside model domain!")

        if merge and self.data is not None and not self.data.empty:
            gdf = gpd.GeoDataFrame(pd.concat([self.data, gdf], ignore_index=True))
            logger.info("Adding new runup gauges to existing ones.")

        self._data = gdf

    @hydromt_step
    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        merge: bool = True,
        **kwargs,
    ) -> None:
        """Create runup gauges from a data source.

        Parameters
        ----------
        locations : str, Path, or gpd.GeoDataFrame
            Path, data source name, or GeoDataFrame with LineString geometries.
        merge : bool, optional
            If ``True``, merge with existing runup gauges. By default ``True``.
        """
        gdf = self.data_catalog.get_geodataframe(
            locations,
            geom=self.model.region,
            **kwargs,
        ).to_crs(self.model.crs)

        gdf = gdf.explode(index_parts=True).reset_index(drop=True)

        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Runup gauges must be of type LineString.")

        if gdf.has_z.any():
            gdf["geometry"] = gdf["geometry"].apply(
                lambda geom: LineString([(x, y) for x, y, z in geom.coords])
            )

        self.set(gdf, merge)
        self.model.config.set("rugfile", "sfincs.rug")

    @hydromt_step
    def add_xy(
        self,
        name: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Add a single runup gauge defined by start and end coordinates.

        Parameters
        ----------
        name : str
            Name of the runup gauge.
        x0, y0 : float
            Start point coordinates.
        x1, y1 : float
            End point coordinates.
        """
        line = LineString([(x0, y0), (x1, y1)])
        gdf = gpd.GeoDataFrame({"name": [name], "geometry": [line]}, crs=self.model.crs)
        self.set(gdf, merge=True)

    def delete(self, index: Union[List[int], int]) -> None:
        """Remove one or more runup gauges by index.

        Parameters
        ----------
        index : int or list of int
            Indices to remove.
        """
        if isinstance(index, int):
            index = [index]
        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")
        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from runup gauges")
        if self.data.empty:
            logger.warning("All runup gauges have been removed!")
            self.model.config.set("rugfile", None)

    def clear(self) -> None:
        """Remove all runup gauge data."""
        self._data = gpd.GeoDataFrame()
        self.model.config.set("rugfile", None)
