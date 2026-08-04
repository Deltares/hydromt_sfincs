"""Urban-drainage-area component for SFINCS.

Wraps the ``urbfile`` input described in
``SFINCS/docs/input_urban_drainage.rst``. Each row in the component's
GeoDataFrame is one drainage zone; the zone geometry is a Polygon, and
the scalar properties (name, type, outfall, design_precip, ...) are
stored as regular columns. On disk the zones are serialised to a TOML
``sfincs.urb`` file plus one or more Delft3D-style ``.pol`` polygon
files (zones that share a ``polygon_file`` are written together in the
same polygon file).

Two zone ``type``s are currently supported:

* ``piped_drainage``
* ``injection_well``

Both use the same set of columns, but only the keys relevant to each
zone's type are serialised to the TOML file.
"""

from __future__ import annotations

import logging
import re
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import tomli_w

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import utils, readers, writers

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel


logger = logging.getLogger(f"hydromt.{__name__}")


_VALID_TYPES = ("piped_drainage", "injection_well")


# Default value for every non-core column. :meth:`set` applies these to
# every row so that the GeoDataFrame always carries the full schema
# regardless of which zone types are present. Columns whose default is
# ``np.nan`` are the XOR / per-type fields (``design_precip`` vs
# ``max_outfall_rate`` for piped drainage, ``injection_rate`` /
# ``maximum_capacity`` for injection wells); the relevant one is
# populated by the user and the irrelevant one stays NaN.
_DEFAULTS: dict = {
    "h_threshold": 0.02,
    "outfall_x": 0.0,
    "outfall_y": 0.0,
    "design_precip": np.nan,
    "max_outfall_rate": np.nan,
    "dh_design_min": 0.1,
    "include_outfall": True,
    "check_valve": False,
    "injection_rate": 0.2,
    "maximum_capacity": 1000.0,
}


class SfincsUrbanDrainageAreas(ModelComponent):
    """SFINCS urban-drainage-area component.

    Stores one row per drainage zone as a polygon with scalar attribute
    columns. Reads and writes the ``sfincs.urb`` TOML file and its
    associated ``.pol`` polygon files.
    """

    def __init__(
        self,
        model: "SfincsModel",
    ):
        self._filename: str = "sfincs.urb"
        self._data: Optional[gpd.GeoDataFrame] = None
        super().__init__(
            model=model,
        )

    # ----- data access -------------------------------------------------

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Urban drainage areas as a GeoDataFrame (one polygon per row)."""
        if self._data is None:
            self._initialize()
        return self._data

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Alias of :attr:`data`."""
        return self.data

    @property
    def nr_areas(self) -> int:
        # Go through self.data (not self._data) so a saved urb file is
        # lazily read on first access in reading mode.
        if self.data is None or self.data.empty:
            return 0
        return len(self.data.index)

    @property
    def list_names(self) -> List[str]:
        if self.data is None or self.data.empty:
            return []
        return list(self.data["name"].astype(str).values)

    def _initialize(self, skip_read: bool = False) -> None:
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if (
                self.root.is_reading_mode()
                and not skip_read
                and self.model.config.get("urbfile") is not None
            ):
                self.read()

    # ----- read / write ------------------------------------------------

    def read(self, filename: Union[str, Path] = None) -> None:
        """Read a SFINCS ``*.urb`` TOML file and its referenced ``.pol`` polygons.

        The filename is taken from ``urbfile`` in the model config if not
        explicitly given.
        """
        self.root._assert_read_mode()

        abs_file_path = self.model.config.get_set_file_variable(
            "urbfile", value=filename
        )
        if abs_file_path is None:
            return
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Urban drainage areas file not found: {abs_file_path}"
            )

        with open(abs_file_path, "rb") as f:
            doc = tomllib.load(f)

        zones = doc.get("urban_drainage_zone", [])
        if not zones:
            self._data = gpd.GeoDataFrame()
            return

        from shapely.geometry import Polygon

        urb_root = abs_file_path.parent
        # Cache polygons per polygon_file so multiple zones sharing a file
        # only trigger one read.
        poly_cache: Dict[Path, Dict[str, np.ndarray]] = {}

        rows: List[dict] = []
        geoms: List = []
        for i, z in enumerate(zones):
            name = z.get("name")
            ztype = z.get("type")
            polygon_file = z.get("polygon_file")
            if not name or not ztype or not polygon_file:
                raise ValueError(
                    f"Zone {i} in {abs_file_path} is missing a required key "
                    f"('name', 'type', 'polygon_file')"
                )
            if ztype not in _VALID_TYPES:
                raise ValueError(
                    f"Zone '{name}' has unsupported type {ztype!r}; "
                    f"expected one of {_VALID_TYPES}"
                )

            poly_path = (urb_root / polygon_file).resolve()
            if poly_path not in poly_cache:
                feats = readers.read_geoms(poly_path)
                poly_cache[poly_path] = {
                    str(f["name"]): np.column_stack([f["x"], f["y"]])
                    for f in feats
                }
            polys = poly_cache[poly_path]
            if name not in polys:
                raise KeyError(
                    f"Zone '{name}' not found in polygon file {polygon_file}"
                )

            geoms.append(Polygon(polys[name]))
            row = dict(z)
            # TOML stores piped outfall as a 2-element ``outfall = [x, y]``
            # array; the gdf keeps the split ``outfall_x`` / ``outfall_y``
            # columns so the rest of the code can index into them directly.
            outfall = row.pop("outfall", None)
            if outfall is not None:
                row["outfall_x"] = float(outfall[0])
                row["outfall_y"] = float(outfall[1])
            rows.append(row)

        gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=self.model.crs)
        self.set(gdf, merge=False)

    def write(self, filename: Union[str, Path] = None) -> None:
        """Write the ``sfincs.urb`` TOML file and its ``.pol`` polygon files."""
        self.root._assert_write_mode()

        if self._data is None or self._data.empty:
            logger.debug("No urban drainage areas available to write.")
            return

        abs_file_path = self.model.config.get_set_file_variable(
            key="urbfile", value=filename, default=self._filename
        )
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Group polygons by their target .pol file so a single file holds
        # every zone that shares the same ``polygon_file`` value.
        poly_groups: Dict[Path, List[dict]] = {}
        zone_tables: List[dict] = []
        for _, row in self._data.iterrows():
            polygon_file = row["polygon_file"]
            poly_path = (abs_file_path.parent / polygon_file).resolve()
            coords = np.asarray(row.geometry.exterior.coords, dtype=float)
            feat = {
                "name": str(row["name"]),
                "x": list(coords[:, 0]),
                "y": list(coords[:, 1]),
            }
            poly_groups.setdefault(poly_path, []).append(feat)
            zone_tables.append(_row_to_toml_table(row))

        fmt = "%.6f" if self.model.crs is not None and self.model.crs.is_geographic else "%.1f"
        for poly_path, feats in poly_groups.items():
            poly_path.parent.mkdir(parents=True, exist_ok=True)
            writers.write_geoms(poly_path, feats, stype="pol", fmt=fmt)

        # tomli_w always wraps arrays across multiple lines. Collapse
        # the two-element ``outfall`` array back onto one line for
        # readability — done as a post-process text substitution since
        # the library has no inline-array option.
        doc = tomli_w.dumps({"urban_drainage_zone": zone_tables})
        doc = re.sub(
            r"outfall = \[\s*([^,\]]+?)\s*,\s*([^,\]]+?)\s*,?\s*\]",
            r"outfall = [\1, \2]",
            doc,
        )
        with open(abs_file_path, "w") as f:
            f.write(doc)

        if self.model.write_gis:
            writers.write_vector(
                self._data,
                name="urb",
                root=join(self.model.root.path, "gis"),
                logger=logger,
            )

    # ----- set / create / delete / clear -------------------------------

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True) -> None:
        """Set the component's GeoDataFrame, optionally merging with existing rows.

        Parameters
        ----------
        gdf
            GeoDataFrame of Polygon geometries in the model CRS, with at
            minimum the columns ``name``, ``type``, ``polygon_file``, plus
            the fields relevant to the zone's type.
        merge
            When ``True`` (default), append to existing rows; otherwise
            replace.
        """
        if not gdf.geometry.type.isin(["Polygon"]).all():
            raise ValueError("Urban drainage areas must be Polygon geometries.")
        if gdf.crs is None:
            raise ValueError("Input GeoDataFrame must have a CRS set.")
        if self.model.crs is not None and gdf.crs != self.model.crs:
            gdf = gdf.to_crs(self.model.crs)

        missing = [c for c in ("name", "type", "polygon_file") if c not in gdf.columns]
        if missing:
            raise ValueError(f"Required column(s) missing from gdf: {missing}")

        if gdf["name"].duplicated().any():
            dups = gdf.loc[gdf["name"].duplicated(), "name"].tolist()
            raise ValueError(f"Duplicate zone name(s): {dups}")

        bad_types = ~gdf["type"].isin(_VALID_TYPES)
        if bad_types.any():
            raise ValueError(
                f"Invalid type(s): {gdf.loc[bad_types, 'type'].unique().tolist()}; "
                f"expected one of {_VALID_TYPES}"
            )

        for _, row in gdf.iterrows():
            if row["type"] == "piped_drainage":
                has_precip = pd.notna(row.get("design_precip"))
                has_rate = pd.notna(row.get("max_outfall_rate"))
                if has_precip and has_rate:
                    raise ValueError(
                        f"Zone {row['name']!r}: specify only one of "
                        "'design_precip' or 'max_outfall_rate'"
                    )
                if not (has_precip or has_rate):
                    raise ValueError(
                        f"Zone {row['name']!r}: piped_drainage needs "
                        "'design_precip' or 'max_outfall_rate'"
                    )
            elif row["type"] == "injection_well":
                if not pd.notna(row.get("injection_rate")):
                    raise ValueError(
                        f"Zone {row['name']!r}: injection_well needs 'injection_rate'"
                    )
                if not pd.notna(row.get("maximum_capacity")):
                    raise ValueError(
                        f"Zone {row['name']!r}: injection_well needs 'maximum_capacity'"
                    )

        # Clip to model region.
        region = self.model.region
        if region is not None and not region.empty:
            within = gdf.within(region.union_all())
            if within.any() == False:  # noqa: E712
                raise ValueError(
                    "None of the urban drainage areas fall within the model domain."
                )
            if not within.all():
                logger.info(
                    "Dropping %d urban drainage area(s) that fall outside the model domain: %s",
                    int((~within).sum()),
                    list(gdf.loc[~within, "name"].astype(str).values),
                )
                gdf = gdf.loc[within].reset_index(drop=True)

        if merge and self._data is not None and not self._data.empty:
            gdf = gpd.GeoDataFrame(
                pd.concat([self._data, gdf], ignore_index=True),
                crs=self.model.crs,
            )
            logger.info("Adding new urban drainage areas to existing ones.")

        gdf = gdf.reset_index(drop=True)

        # Ensure every non-core column exists with its default and fill
        # any missing values. After this, every row carries the full
        # column set regardless of its zone type — columns that are
        # irrelevant for a given row still hold their default. Boolean
        # columns go through .map so they land as bool dtype (pandas'
        # fillna on object-dtype columns is noisy and deprecated).
        for col, default in _DEFAULTS.items():
            if col not in gdf.columns:
                gdf[col] = default
            elif isinstance(default, bool):
                gdf[col] = gdf[col].map(
                    lambda v, d=default: d if pd.isna(v) else bool(v)
                ).astype(bool)
            else:
                gdf[col] = gdf[col].fillna(default)

        self._data = gdf

    @hydromt_step
    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        polygon_file: str = "sfincs.pol",
        merge: bool = True,
        **kwargs,
    ) -> None:
        """Create urban drainage areas from a polygon source.

        The input must carry a ``type`` column for every row (one of
        ``"piped_drainage"`` or ``"injection_well"``) plus any per-type
        fields the zone needs (e.g. ``design_precip`` + ``outfall_x/y``
        for piped zones, ``injection_rate`` + ``maximum_capacity`` for
        injection wells). ``name`` is auto-generated when absent;
        ``polygon_file`` is filled with ``polygon_file`` so zones sharing
        a value land in the same ``.pol`` file.

        Parameters
        ----------
        locations
            Path, data-catalog source name, or GeoDataFrame of polygons.
        polygon_file
            Default ``polygon_file`` value for rows that don't already
            have one.
        merge
            If ``True`` (default), append to existing rows.
        """
        gdf = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, **kwargs
        ).to_crs(self.model.crs)

        if "type" not in gdf.columns:
            raise ValueError(
                "Input GeoDataFrame must have a 'type' column with values "
                f"from {_VALID_TYPES}."
            )
        if "name" not in gdf.columns:
            gdf = gdf.assign(
                name=[f"urb_{i + 1:04d}" for i in range(len(gdf))]
            )
        if "polygon_file" not in gdf.columns:
            gdf = gdf.assign(polygon_file=polygon_file)

        self.set(gdf, merge=merge)
        self.model.config.set("urbfile", self._filename)

    def delete(self, index: Union[int, List[int]]) -> None:
        """Drop one or more urban drainage areas by row index."""
        if isinstance(index, int):
            index = [index]
        n = len(self.data)
        if not index or max(index) >= n or min(index) < 0:
            raise ValueError("One of the indices exceeds the index range.")
        self._data = self.data.drop(index).reset_index(drop=True)
        if self._data.empty:
            logger.warning("All urban drainage areas have been removed.")
            self.model.config.set("urbfile", None)

    def clear(self) -> None:
        """Empty the component and clear ``urbfile`` from the config."""
        self._data = gpd.GeoDataFrame()
        self.model.config.set("urbfile", None)


def _row_to_toml_table(row: pd.Series) -> dict:
    """Convert one GeoDataFrame row into its TOML-serialisable dict.

    ``set()`` guarantees every relevant column is populated, so only
    the per-type emission logic is applied here. ``design_precip`` and
    ``max_outfall_rate`` are XOR — whichever the user populated is
    emitted and the other is skipped.
    """
    out: dict = {
        "name": str(row["name"]),
        "type": str(row["type"]),
        "polygon_file": str(row["polygon_file"]),
        "h_threshold": float(row["h_threshold"]),
    }

    if row["type"] == "piped_drainage":
        out["outfall"] = [float(row["outfall_x"]), float(row["outfall_y"])]
        if pd.notna(row["design_precip"]):
            out["design_precip"] = float(row["design_precip"])
        else:
            out["max_outfall_rate"] = float(row["max_outfall_rate"])
        out["dh_design_min"] = float(row["dh_design_min"])
        out["include_outfall"] = bool(row["include_outfall"])
        out["check_valve"] = bool(row["check_valve"])
    elif row["type"] == "injection_well":
        out["injection_rate"] = float(row["injection_rate"])
        out["maximum_capacity"] = float(row["maximum_capacity"])

    return out
