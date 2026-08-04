import json
import logging
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import pandas as pd
import tomli_w
from shapely.geometry import LineString

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs import readers, writers

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


# Default values for the non-core (type-specific) columns. These are
# stamped on every row via :meth:`_set_defaults` before file data is
# applied on top, so every row always has a value for every column
# regardless of structure type.
_DEFAULTS: dict = {
    "q": 0.0,
    "flow_coef": 0.6,
    "direction": "both",
    "width": 10.0,
    "height": 2.0,
    "sill_elevation": 0.0,
    "mannings_n": 0.024,
    "invert_1": 0.0,
    "invert_2": 0.0,
    "submergence_ratio": 0.67,
    "opening_duration": 0.0,
    "closing_duration": 0.0,
}

# Valid gate-rule operations (see sfincs_src_structures.f90): each rule is
# an {"operation": ..., "when": ...} dict; rules are evaluated in order and
# the first whose "when" expression fires sets the gate target
# (open -> 1.0, close -> 0.0, hold -> freeze). If none fires the gate holds.
_RULE_OPERATIONS = ("open", "close", "hold")


class SfincsDrainageStructures(ModelComponent):
    """SFINCS drainage structures component.

    This component handles reading, writing, and creating drainage structures
    such as pumps, culverts, and gates in a SFINCS model.
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

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Drainage structures data, returns geopandas.GeoDataFrame"""
        return self.data

    @property
    def nr_lines(self) -> int:
        """
        Return the number of line locations currently stored.
        """
        if hasattr(self.data, "index"):
            return len(self.data.index)
        return 0

    @property
    def list_names(self):
        """Give list of names of drainage structures."""
        if self.data.empty:
            return []
        ipump = 0
        iculvert_simple = 0
        iculvert = 0
        igate = 0
        names = []
        for index, row in self.data.iterrows():
            t = int(row["type"])
            if t == 1:
                ipump += 1
                names.append(f"Pump {ipump}")
            elif t == 2:
                iculvert_simple += 1
                names.append(f"Culvert {iculvert_simple}")
            elif t == 4:
                igate += 1
                names.append(f"Gate {igate}")
            elif t == 5:
                iculvert += 1
                names.append(f"Culvert {iculvert}")
            else:
                names.append(f"Drainage {index + 1}")
        return names

    def _initialize(self, skip_read: bool = False) -> None:
        """Initialize drainage structures."""
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def _set_defaults(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Stamp every type-specific column on ``gdf`` with its default.

        Called before per-type values are copied in from a file, so every
        row ends up with a value for every column regardless of which
        structure type it is.
        """
        for col, default in _DEFAULTS.items():
            gdf[col] = default
        # Ordered list of gate control rules per row; each entry is an
        # {"operation": "open"/"close"/"hold", "when": "<expr>"} dict.
        gdf["rules"] = [[] for _ in range(len(gdf))]
        return gdf

    def read(self, filename: str | Path = None):
        """Read SFINCS drainage structures file.

        Auto-detects the new TOML format (``[[src_structure]]`` array)
        and falls back to the legacy fixed-column ``.drn`` reader.
        """

        self.root._assert_read_mode()

        abs_file_path = self.model.config.get_set_file_variable(
            "drnfile", value=filename
        )

        if abs_file_path is None:
            return
        elif not abs_file_path.exists():
            raise FileNotFoundError(
                f"Drainage structures file not found: {abs_file_path}"
            )

        # Detect format: parseable as TOML with a ``src_structure`` key → new
        # format; anything else → legacy fixed-column reader.
        is_toml = False
        try:
            with open(abs_file_path, "rb") as f:
                doc = tomllib.load(f)
            is_toml = isinstance(doc, dict) and "src_structure" in doc
        except tomllib.TOMLDecodeError:
            is_toml = False

        if is_toml:
            self.read_toml(abs_file_path)
            return

        # Legacy fixed-column reader → translate par1..par6 into the
        # named-column schema. Defaults are stamped first, then the
        # per-type values from the file overwrite them.
        legacy = readers.read_drn(abs_file_path, crs=self.model.crs)
        if legacy.empty:
            self.set(legacy, merge=False)
            return

        gdf = gpd.GeoDataFrame(
            {
                "name": ["" for _ in range(len(legacy))],
                "type": legacy["type"].astype(int).values,
            },
            geometry=legacy.geometry.values,
            crs=self.model.crs,
        )
        self._set_defaults(gdf)

        for idx, row in legacy.iterrows():
            t = int(row["type"])
            if t == 1:
                # pump
                gdf.at[idx, "q"] = float(row["par1"])
            elif t == 2:
                # culvert_simple (two-way)
                gdf.at[idx, "flow_coef"] = float(row["par1"])
                gdf.at[idx, "direction"] = "both"
            elif t == 3:
                # legacy check_valve → culvert_simple with direction=positive
                gdf.at[idx, "type"] = 2
                gdf.at[idx, "flow_coef"] = float(row["par1"])
                gdf.at[idx, "direction"] = "positive"
            elif t == 4:
                # gate
                gdf.at[idx, "width"] = float(row["par1"])
                gdf.at[idx, "sill_elevation"] = float(row["par2"])
                gdf.at[idx, "mannings_n"] = float(row["par3"])
                zmin = float(row["par4"])
                zmax = float(row["par5"])
                gdf.at[idx, "rules"] = [
                    {"operation": "open", "when": f"z1>{zmin} & z1<{zmax}"}
                ]
                closing = float(row["par6"])
                gdf.at[idx, "closing_duration"] = closing
                gdf.at[idx, "opening_duration"] = closing
            elif t == 5:
                # culvert (detailed)
                gdf.at[idx, "flow_coef"] = float(row["par1"])
                gdf.at[idx, "width"] = float(row["par2"])
                gdf.at[idx, "height"] = float(row["par3"])
                gdf.at[idx, "invert_1"] = float(row["par4"])
                gdf.at[idx, "invert_2"] = float(row["par5"])
                gdf.at[idx, "submergence_ratio"] = float(row["par6"])
                gdf.at[idx, "direction"] = "both"
            else:
                logger.warning(
                    "Skipping drainage structure at row %d with unknown type %d",
                    idx,
                    t,
                )

        self.set(gdf, merge=False)

    def read_toml(self, filename: str | Path) -> None:
        """Read a drainage-structures file in the new TOML format.

        Parses ``[[src_structure]]`` entries as described in
        ``sfincs_src_structures.f90`` and stores them in the GeoDataFrame
        using integer ``type`` codes (1=pump, 2=culvert_simple, 4=gate,
        5=culvert). The legacy ``check_valve`` type is resolved on read
        into ``culvert_simple`` + ``direction="positive"``.

        Gate control rules are read from the ordered ``[[src_structure.rule]]``
        tables (``operation`` = ``"open"``/``"close"``/``"hold"`` + ``when``)
        into the ``rules`` column: an ordered list of
        ``{"operation": ..., "when": ...}`` dicts per structure. Legacy files
        using scalar ``rules_open`` / ``rules_close`` keys are still accepted
        as a fallback and converted into that list.
        """
        with open(filename, "rb") as f:
            doc = tomllib.load(f)
        structs = doc.get("src_structure", []) or []

        names: list[str] = []
        types: list[int] = []
        geoms: list = []
        entries: list[dict] = []
        for entry in structs:
            ztype = str(entry.get("type", "")).lower()
            if ztype == "pump":
                t = 1
            elif ztype in ("culvert_simple", "check_valve"):
                t = 2
            elif ztype == "gate":
                t = 4
            elif ztype == "culvert":
                t = 5
            else:
                logger.warning(
                    "Skipping src_structure %r: unsupported type %r",
                    entry.get("name"),
                    ztype,
                )
                continue

            src_1 = entry.get("src_1", [0.0, 0.0])
            src_2 = entry.get("src_2", [0.0, 0.0])
            x1, y1 = float(src_1[0]), float(src_1[1])
            x2, y2 = float(src_2[0]), float(src_2[1])

            names.append(str(entry.get("name", "") or ""))
            types.append(t)
            geoms.append(LineString([(x1, y1), (x2, y2)]))
            entries.append({"_toml_type": ztype, **entry})

        gdf = gpd.GeoDataFrame(
            {"name": names, "type": types},
            geometry=geoms,
            crs=self.model.crs,
        )
        self._set_defaults(gdf)

        for idx, entry in enumerate(entries):
            ztype = entry["_toml_type"]
            direction = str(entry.get("direction", "") or "").lower()
            t = int(gdf.at[idx, "type"])
            if t == 1:
                gdf.at[idx, "q"] = float(entry.get("q", _DEFAULTS["q"]))
            elif t == 2:
                # culvert_simple (or legacy check_valve alias)
                if ztype == "check_valve":
                    gdf.at[idx, "direction"] = "positive"
                elif direction in ("both", "positive", "negative"):
                    gdf.at[idx, "direction"] = direction
                else:
                    gdf.at[idx, "direction"] = "both"
                gdf.at[idx, "flow_coef"] = float(
                    entry.get("flow_coef", _DEFAULTS["flow_coef"])
                )
            elif t == 4:
                gdf.at[idx, "width"] = float(
                    entry.get("width", _DEFAULTS["width"])
                )
                gdf.at[idx, "sill_elevation"] = float(
                    entry.get("sill_elevation", _DEFAULTS["sill_elevation"])
                )
                gdf.at[idx, "mannings_n"] = float(
                    entry.get("mannings_n", _DEFAULTS["mannings_n"])
                )
                closing = float(
                    entry.get("closing_duration", _DEFAULTS["closing_duration"])
                )
                gdf.at[idx, "closing_duration"] = closing
                gdf.at[idx, "opening_duration"] = float(
                    entry.get("opening_duration", closing)
                )
            elif t == 5:
                if direction in ("both", "positive", "negative"):
                    gdf.at[idx, "direction"] = direction
                else:
                    gdf.at[idx, "direction"] = "both"
                gdf.at[idx, "flow_coef"] = float(
                    entry.get("flow_coef", _DEFAULTS["flow_coef"])
                )
                gdf.at[idx, "width"] = float(
                    entry.get("width", _DEFAULTS["width"])
                )
                gdf.at[idx, "height"] = float(
                    entry.get("height", _DEFAULTS["height"])
                )
                gdf.at[idx, "invert_1"] = float(
                    entry.get("invert_1", _DEFAULTS["invert_1"])
                )
                gdf.at[idx, "invert_2"] = float(
                    entry.get("invert_2", _DEFAULTS["invert_2"])
                )
                gdf.at[idx, "submergence_ratio"] = float(
                    entry.get("submergence_ratio", _DEFAULTS["submergence_ratio"])
                )

            # Rules apply to all non-pump types. The SFINCS TOML format
            # carries them as an ordered array of tables under "rule"
            # (each {operation = "open"/"close"/"hold", when = "<expr>"}).
            # Order matters: SFINCS evaluates the rules in order and the
            # first whose "when" fires wins. Older files used scalar
            # rules_open / rules_close keys; accept those as a fallback.
            if t != 1:
                rules: list[dict] = []
                for j, r in enumerate(entry.get("rule", []) or []):
                    op = str(r.get("operation", "")).lower().strip()
                    when = str(r.get("when", "") or "").strip()
                    if op not in _RULE_OPERATIONS or not when:
                        logger.warning(
                            "Skipping invalid rule %d of src_structure %r "
                            "(operation=%r, when=%r)",
                            j + 1,
                            entry.get("name", idx),
                            op,
                            when,
                        )
                        continue
                    rules.append({"operation": op, "when": when})
                # Fall back to legacy scalar keys when no rule tables present.
                if not rules:
                    rule_open = str(entry.get("rules_open", "") or "").strip()
                    rule_close = str(entry.get("rules_close", "") or "").strip()
                    if rule_open:
                        rules.append({"operation": "open", "when": rule_open})
                    if rule_close:
                        rules.append({"operation": "close", "when": rule_close})
                gdf.at[idx, "rules"] = rules

        self.set(gdf, merge=False)

    def write(self, filename: str | Path = None):
        """Write SFINCS drainage structures.

        Always writes two files side by side:

        * The legacy fixed-column file (``sfincs.drn`` by default, or
          whatever ``filename`` / ``drnfile`` resolves to), kept as a
          human-readable artifact. It is lossy: it cannot carry rules,
          direction, or detailed-culvert geometry.
        * A companion TOML file named ``sfincs.toml.drn`` next to the
          legacy file, carrying the full new schema (direction, rules,
          detailed-culvert geometry, etc.).

        On a successful TOML write, ``drnfile`` is pointed at
        ``sfincs.toml.drn`` so SFINCS reads the full-fidelity file
        directly instead of re-transcribing the lossy legacy ``.drn``
        over the top of it. If the TOML write fails, ``drnfile`` is left
        on the legacy file as a fallback.
        """

        self.root._assert_write_mode()

        if self.data.empty:
            logger.debug("No drainage structures data available to write.")
            return

        abs_file_path = self.model.config.get_set_file_variable(
            key="drnfile", value=filename, default="sfincs.drn"
        )

        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model.crs.is_geographic:
            fmt = "%11.6f"
        else:
            fmt = "%11.1f"

        # Legacy .drn write — build a par-backed view on the fly so the
        # stored gdf stays in the named-column schema.
        legacy = self._to_legacy_gdf()
        writers.write_drn(abs_file_path, legacy, fmt=fmt)

        # Companion TOML file — always ``sfincs.toml.drn`` next to the
        # legacy file. Preserve the ``drnfile`` config entry so it
        # keeps pointing at the legacy file.
        drnfile_backup = self.model.config.get("drnfile")
        try:
            self.write_toml(filename=abs_file_path.parent / "sfincs.toml.drn")
            self.model.config.set("drnfile", "sfincs.toml.drn")

        except Exception as e:
            logger.warning(
                "Failed to write TOML drainage structures file: %s", e, "continuing with legacy .drn file only."
            )
            self.model.config.set("drnfile", drnfile_backup)

        if self.model.write_gis:
            # GeoJSON fields cannot hold lists — serialize the rules
            # column to a JSON string for the gis sidecar only.
            gis_gdf = self.data.copy()
            if "rules" in gis_gdf.columns:
                gis_gdf["rules"] = gis_gdf["rules"].apply(
                    lambda r: json.dumps(r) if isinstance(r, list) else ""
                )
            writers.write_vector(
                gis_gdf,
                name="drn",
                root=join(self.model.root.path, "gis"),
            )

    def _to_legacy_gdf(self) -> gpd.GeoDataFrame:
        """Build a legacy par1..par6 GeoDataFrame from the named-column data.

        Used only for writing the fixed-column ``.drn`` file via
        :func:`writers.write_drn`. The stored gdf is not modified.
        """
        n = len(self.data)
        legacy = gpd.GeoDataFrame(
            {
                "type": [0] * n,
                "par1": [0.0] * n,
                "par2": [0.0] * n,
                "par3": [0.0] * n,
                "par4": [0.0] * n,
                "par5": [0.0] * n,
                "par6": [0.0] * n,
            },
            geometry=self.data.geometry.values,
            crs=self.model.crs,
        )
        for idx, row in self.data.iterrows():
            t = int(row["type"])
            if t == 1:
                legacy.at[idx, "type"] = 1
                legacy.at[idx, "par1"] = float(row["q"])
            elif t == 2:
                direction = str(row.get("direction", "") or "both").lower()
                if direction == "positive":
                    # Round-trip legacy check_valve so SFINCS (which has
                    # no direction flag in the fixed-column format) still
                    # gets the one-way semantics.
                    legacy.at[idx, "type"] = 3
                else:
                    legacy.at[idx, "type"] = 2
                legacy.at[idx, "par1"] = float(row["flow_coef"])
            elif t == 4:
                legacy.at[idx, "type"] = 4
                legacy.at[idx, "par1"] = float(row["width"])
                legacy.at[idx, "par2"] = float(row["sill_elevation"])
                legacy.at[idx, "par3"] = float(row["mannings_n"])
                legacy.at[idx, "par6"] = float(row["closing_duration"])
            elif t == 5:
                legacy.at[idx, "type"] = 5
                legacy.at[idx, "par1"] = float(row["flow_coef"])
                legacy.at[idx, "par2"] = float(row["width"])
                legacy.at[idx, "par3"] = float(row["height"])
                legacy.at[idx, "par4"] = float(row["invert_1"])
                legacy.at[idx, "par5"] = float(row["invert_2"])
                legacy.at[idx, "par6"] = float(row["submergence_ratio"])
            else:
                legacy.at[idx, "type"] = t
        return legacy

    def write_toml(self, filename: str | Path = None) -> None:
        """Write drainage structures in the new TOML format.

        Default filename is ``sfincs.toml.drn``. Emits all per-type
        parameters for each ``[[src_structure]]`` entry. For non-pump
        structures, the ``rules`` list is written in order as
        ``[[src_structure.rule]]`` tables (``operation`` =
        ``"open"``/``"close"``/``"hold"``, ``when`` = expression), which is
        the format the SFINCS reader expects. Structures with no rules get
        no ``rule`` tables (SFINCS treats them as always open).
        """
        self.root._assert_write_mode()

        if self.data.empty:
            logger.debug("No drainage structures available to write.")
            return

        abs_file_path = self.model.config.get_set_file_variable(
            key="drnfile", value=filename, default="sfincs.toml.drn"
        )
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        tables: list[dict] = []
        for idx, row in self.data.iterrows():
            coords = list(row.geometry.coords)
            x1, y1 = float(coords[0][0]), float(coords[0][1])
            x2, y2 = float(coords[-1][0]), float(coords[-1][1])

            t = int(row["type"])
            name = str(row.get("name", "") or f"drn_{idx + 1:03d}")

            if t == 1:
                type_str = "pump"
            elif t == 2:
                type_str = "culvert_simple"
            elif t == 4:
                type_str = "gate"
            elif t == 5:
                type_str = "culvert"
            else:
                logger.warning(
                    "Skipping drainage structure %r with unknown type %s",
                    name,
                    t,
                )
                continue

            entry: dict = {
                "type": type_str,
                "name": name,
                "src_1": [x1, y1],
                "src_2": [x2, y2],
            }
            if t == 1:
                entry["q"] = float(row["q"])
            elif t == 2:
                direction = str(row.get("direction", "") or "both").lower()
                if direction not in ("both", "positive", "negative"):
                    direction = "both"
                entry["direction"] = direction
                entry["flow_coef"] = float(row["flow_coef"])
            elif t == 4:
                entry["width"] = float(row["width"])
                entry["sill_elevation"] = float(row["sill_elevation"])
                entry["mannings_n"] = float(row["mannings_n"])
                entry["opening_duration"] = float(row["opening_duration"])
                entry["closing_duration"] = float(row["closing_duration"])
            elif t == 5:
                direction = str(row.get("direction", "") or "both").lower()
                if direction not in ("both", "positive", "negative"):
                    direction = "both"
                entry["direction"] = direction
                entry["flow_coef"] = float(row["flow_coef"])
                entry["width"] = float(row["width"])
                entry["height"] = float(row["height"])
                entry["invert_1"] = float(row["invert_1"])
                entry["invert_2"] = float(row["invert_2"])
                entry["submergence_ratio"] = float(row["submergence_ratio"])

            # Rules apply to every non-pump type. SFINCS reads them as an
            # ordered array of tables under the "rule" sub-key
            # (each -> [[src_structure.rule]] with "operation" + "when").
            # Emit the stored rules list in order — order matters, the
            # first rule whose "when" fires wins.
            if t != 1:
                stored = row.get("rules")
                rules: list[dict] = []
                if isinstance(stored, list):
                    for r in stored:
                        op = str(r.get("operation", "")).lower().strip()
                        when = str(r.get("when", "") or "").strip()
                        if op in _RULE_OPERATIONS and when:
                            rules.append({"operation": op, "when": when})
                if rules:
                    entry["rule"] = rules

            tables.append(entry)

        with open(abs_file_path, "wb") as f:
            tomli_w.dump({"src_structure": tables}, f)

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
                gdf = gdf[within]
                gdf_name = gdf.name[~within]
                logger.info(
                    "Some of the drainage structures fall out of model domain. Removing structures: "
                    + str(gdf_name.values)
                )
        else:
            raise ValueError("None of drainage structures fall within model domain.")

        if merge and not self.data.empty:
            gdf0 = self.data
            gdf = gpd.GeoDataFrame(pd.concat([gdf0, gdf], ignore_index=True))
            logger.info("Adding new drainage structures to existing ones.")

        self._data = gdf

    @hydromt_step
    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        stype: str = "pump",
        discharge: float = 0.0,
        flow_coef: float = 0.6,
        width: float = 10.0,
        height: float = 2.0,
        sill_elevation: float = 0.0,
        mannings_n: float = 0.024,
        invert_1: float = 0.0,
        invert_2: float = 0.0,
        submergence_ratio: float = 0.67,
        opening_duration: float = 0.0,
        closing_duration: float = 0.0,
        rules: list = None,
        merge: bool = True,
        **kwargs,
    ):
        """Create drainage structures such as pumps, culverts, or gates.

        Adds model layer:

        * **drn** geom: drainage pump, culvert, or gate

        Parameters
        ----------
        locations : str, Path
            Path, data source name, or geopandas object to structure line geometry file.
            The line should consist of only 2 points (else first and last points are used),
            ordered from up to downstream. The "type" column is optional.
        stype : {'pump', 'culvert_simple', 'culvert', 'gate', 'valve', 'check_valve'}, optional
            Structure type, by default "pump". Legacy aliases ``valve`` and ``check_valve``
            resolve to ``culvert_simple`` with ``direction="positive"``.
        discharge : float, optional
            Pump discharge (m3/s), by default 0.0.
        flow_coef : float, optional
            Flow coefficient for culverts, by default 0.6.
        width : float, optional
            Gate width or culvert width (m), by default 10.0.
        height : float, optional
            Detailed culvert height (m), by default 2.0.
        sill_elevation : float, optional
            Gate sill elevation (m), by default 0.0.
        mannings_n : float, optional
            Gate Manning's n, by default 0.024.
        invert_1, invert_2 : float, optional
            Detailed culvert invert elevations at the two endpoints (m), by default 0.0.
        submergence_ratio : float, optional
            Detailed culvert submergence ratio, by default 0.67.
        opening_duration, closing_duration : float, optional
            Gate opening/closing ramp durations (s), by default 0.0.
        rules : list of dict, optional
            Ordered gate control rules, each an
            ``{"operation": "open"/"close"/"hold", "when": "<expr>"}`` dict.
            Rules are evaluated in order; the first whose ``when`` expression
            fires wins. By default no rules (structure stays fully open).
        merge : bool, optional
            If True, merge with existing drainage locations, by default True.
        """

        stype = stype.lower()
        svalues = {
            "pump": (1, ""),
            "culvert_simple": (2, "both"),
            "culvert": (5, "both"),
            "valve": (2, "positive"),
            "check_valve": (2, "positive"),
            "gate": (4, ""),
        }
        if stype not in svalues:
            raise ValueError(
                "stype must be one of " + ", ".join(sorted(svalues)) + "."
            )
        t, default_direction = svalues[stype]

        # read, clip and reproject
        gdf_in = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, **kwargs
        ).to_crs(self.model.crs)

        # multi to single lines
        lines = gdf_in.explode(column="geometry").reset_index(drop=True)
        # start [0] and end [1] points → simple two-point LineString
        endpoints = lines.boundary.explode(index_parts=True).unstack()
        geom = endpoints.apply(lambda x: LineString(x.values.tolist()), axis=1)
        lines = lines.reset_index(drop=True).set_geometry(geom.reset_index(drop=True))

        # Build the new-schema gdf with defaults, then overwrite per row.
        n = len(lines)
        if "type" in lines.columns:
            types = lines["type"].astype(int).values
        else:
            types = [t] * n
        if "name" in lines.columns:
            names = [str(v) for v in lines["name"].values]
        else:
            names = ["" for _ in range(n)]

        gdf = gpd.GeoDataFrame(
            {"name": names, "type": list(types)},
            geometry=lines.geometry.values,
            crs=self.model.crs,
        )
        self._set_defaults(gdf)

        # Overwrite defaults with scalar kwargs per-type (only values that
        # apply to that type are written).
        for idx in range(n):
            row_t = int(gdf.at[idx, "type"])
            if row_t == 1:
                gdf.at[idx, "q"] = float(discharge)
            elif row_t == 2:
                gdf.at[idx, "direction"] = default_direction or "both"
                gdf.at[idx, "flow_coef"] = float(flow_coef)
            elif row_t == 4:
                gdf.at[idx, "width"] = float(width)
                gdf.at[idx, "sill_elevation"] = float(sill_elevation)
                gdf.at[idx, "mannings_n"] = float(mannings_n)
                gdf.at[idx, "opening_duration"] = float(opening_duration)
                gdf.at[idx, "closing_duration"] = float(closing_duration)
            elif row_t == 5:
                gdf.at[idx, "direction"] = default_direction or "both"
                gdf.at[idx, "flow_coef"] = float(flow_coef)
                gdf.at[idx, "width"] = float(width)
                gdf.at[idx, "height"] = float(height)
                gdf.at[idx, "invert_1"] = float(invert_1)
                gdf.at[idx, "invert_2"] = float(invert_2)
                gdf.at[idx, "submergence_ratio"] = float(submergence_ratio)

            if row_t != 1:
                gdf.at[idx, "rules"] = list(rules) if rules else []

        # Let any explicit columns in the input gdf override the scalars.
        for col in _DEFAULTS:
            if col in gdf_in.columns:
                gdf[col] = gdf_in[col].values

        self.set(gdf, merge=merge)
        self.model.config.set("drnfile", "sfincs.drn")

    def delete(
        self,
        index: Union[list, int],
    ):
        """Remove one or more drainage structures.

        Parameters
        ----------
        index: list, int
            Specify drainage structures to be dropped from GeoDataFrame.
            If int, drop a single drainage structure based on index.
            If list, drop multiple drainage structures based on index.
        """
        if isinstance(index, int):
            index = [index]

        if max(index) > (len(self.data) - 1) or min(index) < 0:
            raise ValueError("One of the indices exceeds length of index range!")

        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Dropping line(s) from drainage structures")

        if self.data.empty:
            logger.warning("All drainage structures have been removed!")
            self.model.config.set("drnfile", None)

    def clear(self):
        """Clean GeoDataFrame with drainage structures."""
        self._data = gpd.GeoDataFrame()
        self.model.config.set("drnfile", None)
