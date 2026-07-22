import logging
import math
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

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

from hydromt_sfincs import utils

if TYPE_CHECKING:
    from hydromt_sfincs.sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


# Default column values for dike breach structures.
_DEFAULTS: dict = {
    "z_crest": 0.0,
    "z_min": 0.0,
    "B0": 0.0,
    "t_breach": 0.0,
    "t0": 0.0,
    "dike_core": 1,
    "obs_1_x": float("nan"),
    "obs_1_y": float("nan"),
    "obs_2_x": float("nan"),
    "obs_2_y": float("nan"),
    "rules_open": "",
    "rules_close": "",
}

# Integer type code used in the shared drn schema.
_TYPE_CODE = 6


class SfincsDikeBreaches(ModelComponent):
    """SFINCS dike breach component.

    Manages two-phase dike breach structures (Verheij-Knaap, 2003).
    Structures are stored as LineStrings (src_1 → src_2) and written to
    the TOML drainage-structures file (``dkbfile``) as
    ``[[src_structure]]`` entries with ``type = "dike_breach"``.

    Phase 1: crest lowers linearly from ``z_crest`` to ``z_min`` over
    ``t0`` seconds at fixed width ``B0``.

    Phase 2: crest fixed at ``z_min``, breach widens according to the
    Verheij formula; widening only occurs while the inside water level
    (obs_1) exceeds the outside (obs_2).
    """

    def __init__(self, model: "SfincsModel"):
        self._filename: str = "sfincs.dkb"
        self._data: gpd.GeoDataFrame = None
        super().__init__(model=model)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> gpd.GeoDataFrame:
        """Dike breach structures as a GeoDataFrame."""
        if self._data is None:
            self._initialize()
        return self._data

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Alias for :attr:`data`."""
        return self.data

    @property
    def nr_lines(self) -> int:
        """Number of dike breach structures currently stored."""
        if hasattr(self.data, "index"):
            return len(self.data.index)
        return 0

    @property
    def list_names(self) -> list:
        """Return a list of names for all dike breach structures."""
        if self.data.empty:
            return []
        names = []
        for i, (_, row) in enumerate(self.data.iterrows()):
            name = str(row.get("name", "") or "")
            names.append(name if name else f"Dike breach {i + 1}")
        return names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(self, skip_read: bool = False) -> None:
        if self._data is None:
            self._data = gpd.GeoDataFrame()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def _set_defaults(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Stamp every dike-breach column on *gdf* with its default value."""
        for col, default in _DEFAULTS.items():
            gdf[col] = default
        return gdf

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, filename: str | Path = None) -> None:
        """Read dike breach structures from the drainage-structures file.

        Auto-detects TOML format (``[[src_structure]]`` with
        ``type = "dike_breach"``); falls back gracefully when the file
        contains no dike breach entries.
        """
        self.root._assert_read_mode()

        abs_file_path = self.model.config.get_set_file_variable(
            "dkbfile", value=filename
        )
        if abs_file_path is None:
            return
        if not abs_file_path.exists():
            raise FileNotFoundError(
                f"Dike breach file not found: {abs_file_path}"
            )

        try:
            with open(abs_file_path, "rb") as f:
                doc = tomllib.load(f)
        except tomllib.TOMLDecodeError:
            raise ValueError(
                f"Dike breach file {abs_file_path} is not valid TOML."
            )

        if not isinstance(doc, dict) or "src_structure" not in doc:
            return

        structs = doc.get("src_structure", []) or []
        breach_entries = [
            e for e in structs
            if str(e.get("type", "")).lower() == "dike_breach"
        ]

        if not breach_entries:
            return

        names, geoms, entries = [], [], []
        for entry in breach_entries:
            src_1 = entry.get("src_1", [0.0, 0.0])
            src_2 = entry.get("src_2", [0.0, 0.0])
            x1, y1 = float(src_1[0]), float(src_1[1])
            x2, y2 = float(src_2[0]), float(src_2[1])
            names.append(str(entry.get("name", "") or ""))
            geoms.append(LineString([(x1, y1), (x2, y2)]))
            entries.append(entry)

        gdf = gpd.GeoDataFrame(
            {"name": names, "type": [_TYPE_CODE] * len(names)},
            geometry=geoms,
            crs=self.model.crs,
        )
        self._set_defaults(gdf)

        for idx, entry in enumerate(entries):
            gdf.at[idx, "z_crest"]   = float(entry.get("z_crest",  _DEFAULTS["z_crest"]))
            gdf.at[idx, "z_min"]     = float(entry.get("z_min",    _DEFAULTS["z_min"]))
            gdf.at[idx, "B0"]        = float(entry.get("B0",       _DEFAULTS["B0"]))
            gdf.at[idx, "t_breach"]  = float(entry.get("t_breach", _DEFAULTS["t_breach"]))
            gdf.at[idx, "t0"]        = float(entry.get("t0",       _DEFAULTS["t0"]))
            gdf.at[idx, "dike_core"] = int(entry.get("dike_core", _DEFAULTS["dike_core"]))
            if "obs_1" in entry:
                gdf.at[idx, "obs_1_x"] = float(entry["obs_1"][0])
                gdf.at[idx, "obs_1_y"] = float(entry["obs_1"][1])
            if "obs_2" in entry:
                gdf.at[idx, "obs_2_x"] = float(entry["obs_2"][0])
                gdf.at[idx, "obs_2_y"] = float(entry["obs_2"][1])
            # Trigger rules: read the ordered "rule" array of tables
            # ({operation, when}); fall back to legacy scalar keys.
            rule_open, rule_close = "", ""
            for r in entry.get("rule", []) or []:
                op = str(r.get("operation", "")).lower()
                when = str(r.get("when", "") or "")
                if op == "open" and not rule_open:
                    rule_open = when
                elif op == "close" and not rule_close:
                    rule_close = when
            if not rule_open:
                rule_open = str(entry.get("rules_open", "") or "")
            if not rule_close:
                rule_close = str(entry.get("rules_close", "") or "")
            gdf.at[idx, "rules_open"]  = rule_open
            gdf.at[idx, "rules_close"] = rule_close

        self.set(gdf, merge=False)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, filename: str | Path = None) -> None:
        """Write dike breach structures to ``sfincs.dkb``.

        Writes all dike breach structures as ``[[src_structure]]`` entries
        with ``type = "dike_breach"`` to the file referenced by the
        ``dkbfile`` config key.  SFINCS reads this file independently of
        ``drnfile`` and appends the entries to the same structure pool.
        """
        self.root._assert_write_mode()

        if self.data.empty:
            logger.debug("No dike breach structures to write.")
            return

        abs_file_path = self.model.config.get_set_file_variable(
            key="dkbfile", value=filename, default="sfincs.dkb"
        )
        abs_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(abs_file_path, "wb") as f:
            tomli_w.dump({"src_structure": self._to_toml_tables()}, f)

        if self.model.write_gis:
            utils.write_vector(
                self.data,
                name="dike_breach",
                root=join(self.model.root.path, "gis"),
                logger=logger,
            )

    def _to_toml_tables(self) -> list[dict]:
        """Build the list of TOML dicts for all dike breach structures."""
        tables: list[dict] = []
        for idx, row in self.data.iterrows():
            coords = list(row.geometry.coords)
            x1, y1 = float(coords[0][0]), float(coords[0][1])
            x2, y2 = float(coords[-1][0]), float(coords[-1][1])
            name = str(row.get("name", "") or f"breach_{idx + 1:03d}")

            entry: dict = {
                "type":      "dike_breach",
                "name":      name,
                "src_1":     [x1, y1],
                "src_2":     [x2, y2],
                "z_crest":   float(row["z_crest"]),
                "z_min":     float(row["z_min"]),
                "B0":        float(row["B0"]),
                "t_breach":  float(row["t_breach"]),
                "t0":        float(row["t0"]),
                "dike_core": int(row["dike_core"]),
            }
            # obs_1 / obs_2 are optional; only written when explicitly set.
            obs_1_x = float(row.get("obs_1_x", float("nan")))
            obs_1_y = float(row.get("obs_1_y", float("nan")))
            obs_2_x = float(row.get("obs_2_x", float("nan")))
            obs_2_y = float(row.get("obs_2_y", float("nan")))
            if not (math.isnan(obs_1_x) or math.isnan(obs_1_y)):
                entry["obs_1"] = [obs_1_x, obs_1_y]
            if not (math.isnan(obs_2_x) or math.isnan(obs_2_y)):
                entry["obs_2"] = [obs_2_x, obs_2_y]
            # Trigger rules go in the ordered "rule" array of tables
            # ({operation, when}); SFINCS ignores scalar rules_open/close.
            rules: list[dict] = []
            rule_open = str(row.get("rules_open", "") or "").strip()
            rule_close = str(row.get("rules_close", "") or "").strip()
            if rule_open:
                rules.append({"operation": "open", "when": rule_open})
            if rule_close:
                rules.append({"operation": "close", "when": rule_close})
            if rules:
                entry["rule"] = rules
            tables.append(entry)
        return tables

    # ------------------------------------------------------------------
    # Set / delete / clear
    # ------------------------------------------------------------------

    def set(self, gdf: gpd.GeoDataFrame, merge: bool = True) -> None:
        """Store dike breach structures.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            LineString GeoDataFrame in the model CRS.
        merge : bool
            If True, append to existing structures; if False, replace.
        """
        if not gdf.geometry.type.isin(["LineString"]).all():
            raise ValueError("Dike breach structures must be of type LineString.")
        if not gdf.crs == self.model.crs:
            raise ValueError(
                f"Dike breach structures CRS {gdf.crs} does not match model CRS "
                f"{self.model.crs}."
            )

        within = gdf.within(self.model.region.union_all())
        if within.any():
            if not within.all():
                removed = gdf.loc[~within, "name"] if "name" in gdf.columns else gdf[~within].index
                logger.info(
                    "Some dike breach structures fall outside model domain and will be removed: %s",
                    list(removed),
                )
                gdf = gdf[within]
        else:
            raise ValueError("None of the dike breach structures fall within the model domain.")

        if merge and not self.data.empty:
            gdf = gpd.GeoDataFrame(
                pd.concat([self.data, gdf], ignore_index=True),
                crs=self.model.crs,
            )
            logger.info("Adding new dike breach structures to existing ones.")

        self._data = gdf

    def delete(self, index: Union[list, int]) -> None:
        """Remove one or more dike breach structures by index.

        Parameters
        ----------
        index : int or list of int
            Row index/indices to remove.
        """
        if isinstance(index, int):
            index = [index]
        if max(index) > len(self.data) - 1 or min(index) < 0:
            raise ValueError("One of the indices exceeds the valid index range.")
        self._data = self.data.drop(index).reset_index(drop=True)
        logger.info("Removed %d dike breach structure(s).", len(index))
        if self.data.empty:
            logger.warning("All dike breach structures have been removed.")
            self.model.config.set("dkbfile", None)

    def clear(self) -> None:
        """Remove all dike breach structures."""
        self._data = gpd.GeoDataFrame()
        self.model.config.set("dkbfile", None)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def _resolve_obs_points(
        self,
        obs: Optional[Union[list, "gpd.GeoDataFrame", str, Path]],
        n: int,
    ) -> Optional[list]:
        """Resolve *obs* into a list of ``n`` ``(x, y)`` tuples.

        Parameters
        ----------
        obs:
            Accepted inputs:

            * ``None`` — no observation point; returns ``None``.
            * ``[x, y]`` — single coordinate pair; replicated for all *n*
              structures.
            * :class:`~geopandas.GeoDataFrame` with Point geometries — one
              row per structure (length *n*) **or** one row applied to all.
            * ``str`` or :class:`~pathlib.Path` — data-catalogue name or file
              path; loaded as a GeoDataFrame and treated as above.

        n:
            Number of structures (expected output length).

        Returns
        -------
        list of (x, y) tuples, length *n*, or ``None``.
        """
        if obs is None:
            return None

        # ------------------------------------------------------------------
        # 1. Plain [x, y] list
        # ------------------------------------------------------------------
        if isinstance(obs, (list, tuple)) and len(obs) == 2:
            first, second = obs[0], obs[1]
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                # Single coordinate pair — broadcast to all n structures.
                return [(float(first), float(second))] * n

        # ------------------------------------------------------------------
        # 2. str / Path — load from data catalogue or file
        # ------------------------------------------------------------------
        if isinstance(obs, (str, Path)):
            obs = self.data_catalog.get_geodataframe(
                obs, geom=self.model.region
            ).to_crs(self.model.crs)

        # ------------------------------------------------------------------
        # 3. GeoDataFrame with Point geometries
        # ------------------------------------------------------------------
        if isinstance(obs, gpd.GeoDataFrame):
            if not obs.geometry.geom_type.isin(["Point"]).all():
                raise ValueError(
                    "obs_1/obs_2 GeoDataFrame must contain only Point geometries."
                )
            obs = obs.to_crs(self.model.crs)
            coords = [(float(geom.x), float(geom.y)) for geom in obs.geometry]
            if len(coords) == 1:
                return coords * n  # broadcast single point
            if len(coords) == n:
                return coords
            raise ValueError(
                f"obs GeoDataFrame has {len(coords)} rows but {n} structure(s) were "
                "supplied.  Provide either 1 row (broadcast) or one row per structure."
            )

        raise TypeError(
            f"obs_1/obs_2 must be None, [x, y], a GeoDataFrame, or a str/Path; "
            f"got {type(obs)}."
        )

    @staticmethod
    def _t_breach_to_seconds(
        t_breach: Union[float, int, str, datetime],
        tref: datetime,
    ) -> float:
        """Convert *t_breach* to seconds since *tref*.

        Accepts:
        * ``float`` / ``int`` — already in seconds, returned as-is.
        * ``str`` — parsed as ``"YYYY-MM-DD HH:MM:SS"`` (or any format
          understood by :func:`datetime.fromisoformat`), then differenced
          against *tref*.
        * :class:`datetime` — differenced directly against *tref*.
        """
        if isinstance(t_breach, (int, float)):
            return float(t_breach)
        if isinstance(t_breach, str):
            t_breach = datetime.fromisoformat(t_breach)
        if isinstance(t_breach, datetime):
            delta = t_breach - tref
            return delta.total_seconds()
        raise TypeError(
            f"t_breach must be a float (seconds) or a datetime/ISO string, got {type(t_breach)}"
        )

    @hydromt_step
    def create(
        self,
        locations: Union[str, Path, gpd.GeoDataFrame],
        z_crest: float = 0.0,
        z_min: float = 0.0,
        B0: float = 0.0,
        t_breach: Union[float, int, str, datetime] = 0.0,
        t0: float = 0.0,
        dike_core: int = 1,
        obs_1: Optional[Union[list, gpd.GeoDataFrame, str, Path]] = None,
        obs_2: Optional[Union[list, gpd.GeoDataFrame, str, Path]] = None,
        rules_open: str = "",
        rules_close: str = "",
        merge: bool = True,
        **kwargs,
    ) -> None:
        """Create dike breach structures.

        Adds model layer:

        * **dike_breach** geom: dike breach line (src_1 = inside, src_2 = outside)

        Parameters
        ----------
        locations : str, Path, or gpd.GeoDataFrame
            Path, data catalogue name, or GeoDataFrame with LineString
            geometries. Lines must have exactly 2 points (or first/last
            are used), ordered inside → outside.
        z_crest : float, optional
            Initial dike crest elevation (m+datum), by default 0.0.
        z_min : float, optional
            Minimum crest elevation at end of Phase 1 (m+datum), by default 0.0.
        B0 : float, optional
            Initial breach width at start of Phase 2 (m), by default 0.0.
        t_breach : float, int, str, or datetime, optional
            Breach trigger time. Accepts seconds since ``tref`` as a
            ``float``/``int``, or a datetime as a :class:`datetime` object
            or ISO 8601 string (``"YYYY-MM-DD HH:MM:SS"``). When a datetime
            is supplied, ``tref`` must be set in the model config.
            By default 0.0.  
        t0 : float, optional
            Phase 1 duration — time for crest to lower from ``z_crest``
            to ``z_min`` (s), by default 0.0.
        dike_core : int, optional
            Core material: 1 = sand (uc = 0.2 m/s), 2 = clay (uc = 0.5 m/s),
            by default 1.
        obs_1 : [x, y], GeoDataFrame, str, or Path, optional
            Observation point(s) on the inside (high-head side) for water
            level sensing and breach widening control. Accepts:

            * ``[x, y]`` — single coordinate pair applied to all structures.
            * :class:`~geopandas.GeoDataFrame` (or path / data-catalogue name)
              with Point geometries — one row per structure, or one row
              applied to all structures.

            Defaults to ``src_1`` when not specified.
        obs_2 : [x, y], GeoDataFrame, str, or Path, optional
            Observation point(s) on the outside (low-head side). Same
            accepted formats as ``obs_1``. Defaults to ``src_2`` when not
            specified.
        rules_open, rules_close : str, optional
            Logical trigger expressions (e.g. ``"z1 > 3.0"``), by default "".
        merge : bool, optional
            If True, append to existing structures, by default True.
        """
        # Resolve t_breach to seconds since tref, supporting datetime input.
        if not isinstance(t_breach, (int, float)):
            tref = self.model.config.get("tref")
            if tref is None:
                raise ValueError(
                    "t_breach was given as a datetime but 'tref' is not set in the "
                    "model config. Set tref before calling create(), or supply "
                    "t_breach as seconds (float)."
                )
        t_breach_s = self._t_breach_to_seconds(t_breach, self.model.config.get("tref"))
        # Read and reproject input locations.
        gdf_in = self.data_catalog.get_geodataframe(
            locations, geom=self.model.region, **kwargs
        ).to_crs(self.model.crs)

        # Explode multi-lines; keep only two-point (first/last) LineStrings.
        lines = gdf_in.explode(column="geometry").reset_index(drop=True)
        endpoints = lines.boundary.explode(index_parts=True).unstack()
        geom = endpoints.apply(lambda x: LineString(x.values.tolist()), axis=1)
        lines = lines.reset_index(drop=True).set_geometry(geom.reset_index(drop=True))

        n = len(lines)
        names = (
            [str(v) for v in lines["name"].values]
            if "name" in lines.columns
            else [""] * n
        )

        gdf = gpd.GeoDataFrame(
            {"name": names, "type": [_TYPE_CODE] * n},
            geometry=lines.geometry.values,
            crs=self.model.crs,
        )
        self._set_defaults(gdf)

        # Resolve obs_1 / obs_2 into per-structure coordinate lists.
        obs_1_coords = self._resolve_obs_points(obs_1, n)
        obs_2_coords = self._resolve_obs_points(obs_2, n)

        for idx in range(n):
            gdf.at[idx, "z_crest"]     = float(z_crest)
            gdf.at[idx, "z_min"]       = float(z_min)
            gdf.at[idx, "B0"]          = float(B0)
            gdf.at[idx, "t_breach"]    = float(t_breach_s)
            gdf.at[idx, "t0"]          = float(t0)
            gdf.at[idx, "dike_core"]   = int(dike_core)
            if obs_1_coords is not None:
                gdf.at[idx, "obs_1_x"] = obs_1_coords[idx][0]
                gdf.at[idx, "obs_1_y"] = obs_1_coords[idx][1]
            if obs_2_coords is not None:
                gdf.at[idx, "obs_2_x"] = obs_2_coords[idx][0]
                gdf.at[idx, "obs_2_y"] = obs_2_coords[idx][1]
            gdf.at[idx, "rules_open"]  = rules_open
            gdf.at[idx, "rules_close"] = rules_close

        # Columns in the input GeoDataFrame override scalar kwargs.
        for col in _DEFAULTS:
            if col in gdf_in.columns:
                gdf[col] = gdf_in[col].values

        self.set(gdf, merge=merge)
        self.model.config.set("dkbfile", "sfincs.dkb")
