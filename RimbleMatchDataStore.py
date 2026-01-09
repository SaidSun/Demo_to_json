# RimbleMatchDataStore.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import os
import json
import uuid
import math
from datetime import datetime, timezone

import pandas as pd


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class DataPaths:
    # parquet
    ticks: str
    grenades: str

    # csv
    rounds: str
    kills: str
    damages: str
    shots: str
    bomb: str
    smokes: str
    infernos: str
    footsteps: str

    # json
    header_json: str  # *_header_all.json


@dataclass(frozen=True)
class ClockConfig:
    freeze_time_seconds: int = 15
    round_time_seconds: int = 115
    bomb_time_seconds: int = 40
    tickrate_demoparser2: Optional[float] = None
    tickrate_rounds_median: Optional[float] = None


# -----------------------------
# Utilities
# -----------------------------
def _safe_read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        print(f"[WARN] cannot read json {path}: {e}")
        return {}


def _must_exist(path: str, label: str) -> str:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


# -----------------------------
# Main store
# -----------------------------
class RimbleMatchDataStore:
    """
    Eager store:
      - читает ВСЕ таблицы в __init__ и держит их в атрибутах
      - читает header_all.json и meta.json и готовит карты/clock_cfg
      - умеет:
          * last tick per map
          * base fields
          * metadata cumulative (до текущей карты включительно)
          * clock.currentSeconds по rounds(start/freeze_end) + bomb events
          * map bounds из ticks X/Y
          * teams (shallow) без results/players
    """

    _UUID_NS_TEAMS = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    def __init__(
        self,
        paths: DataPaths,
        match_meta_path: Optional[str],
        external_meta: Optional[Dict[str, Any]] = None,
        *,
        csv_kwargs: Optional[Dict[str, Any]] = None,
        parquet_kwargs: Optional[Dict[str, Any]] = None,
        validate_paths: bool = True,
    ) -> None:
        self.paths = paths
        self.match_meta_path = match_meta_path
        self.external_meta = external_meta or {}

        self.csv_kwargs = csv_kwargs or {"encoding": "utf-8"}
        self.parquet_kwargs = parquet_kwargs or {}

        if validate_paths:
            _must_exist(self.paths.header_json, "header_json")

            _must_exist(self.paths.ticks, "ticks parquet")
            _must_exist(self.paths.grenades, "grenades parquet")

            _must_exist(self.paths.rounds, "rounds csv")
            _must_exist(self.paths.kills, "kills csv")
            _must_exist(self.paths.damages, "damages csv")
            _must_exist(self.paths.shots, "shots csv")
            _must_exist(self.paths.bomb, "bomb csv")
            _must_exist(self.paths.smokes, "smokes csv")
            _must_exist(self.paths.infernos, "infernos csv")
            _must_exist(self.paths.footsteps, "footsteps csv")

            if match_meta_path:
                _must_exist(match_meta_path, "match_meta_path")

        # --- load header/meta ---
        self.header_all: Dict[str, Any] = _safe_read_json(self.paths.header_json)
        self.meta_obj: Dict[str, Any] = _safe_read_json(match_meta_path) if match_meta_path else {}

        self.header_by_game_num: Dict[int, Dict[str, Any]] = {}
        self.clock_cfg_by_game_num: Dict[int, ClockConfig] = {}
        self._init_by_game_num()

        # --- EAGER: load ALL frames ---
        self.df_ticks: pd.DataFrame = self._read_parquet(self.paths.ticks)
        self.df_grenades: pd.DataFrame = self._read_parquet(self.paths.grenades)

        self.df_rounds: pd.DataFrame = self._read_csv(self.paths.rounds)
        self.df_kills: pd.DataFrame = self._read_csv(self.paths.kills)
        self.df_damages: pd.DataFrame = self._read_csv(self.paths.damages)
        self.df_shots: pd.DataFrame = self._read_csv(self.paths.shots)
        self.df_bomb: pd.DataFrame = self._read_csv(self.paths.bomb)
        self.df_smokes: pd.DataFrame = self._read_csv(self.paths.smokes)
        self.df_infernos: pd.DataFrame = self._read_csv(self.paths.infernos)
        self.df_footsteps: pd.DataFrame = self._read_csv(self.paths.footsteps)

        # caches
        self._round_tick_cache: Optional[Dict[Tuple[int, int], Dict[str, Optional[int]]]] = None
        self._last_tick_by_game_cache: Optional[Dict[int, int]] = None

    # --------------------
    # init from header/meta
    # --------------------
    def _init_by_game_num(self) -> None:
        by_game_num = (self.header_all.get("by_game_num") or {})
        if isinstance(by_game_num, dict):
            for g_str, g_obj in by_game_num.items():
                try:
                    g = int(g_str)
                except Exception:
                    continue
                self.header_by_game_num[g] = g_obj or {}

        meta_by_game_num = (self.meta_obj.get("meta_by_game_num") or {})
        if isinstance(meta_by_game_num, dict):
            for g_str, m in meta_by_game_num.items():
                try:
                    g = int(g_str)
                except Exception:
                    continue
                m = m or {}
                self.clock_cfg_by_game_num[g] = ClockConfig(
                    freeze_time_seconds=int(m.get("freeze_time_seconds", 15)),
                    round_time_seconds=int(m.get("round_time_seconds", 115)),
                    bomb_time_seconds=int(m.get("bomb_time_seconds", 40)),
                    tickrate_demoparser2=float(m["tickrate_demoparser2"]) if m.get("tickrate_demoparser2") is not None else None,
                    tickrate_rounds_median=float(m["tickrate_rounds_median"]) if m.get("tickrate_rounds_median") is not None else None,
                )

        if not self.clock_cfg_by_game_num:
            for g in self.header_by_game_num.keys():
                self.clock_cfg_by_game_num[g] = ClockConfig()

    @property
    def game_nums(self) -> List[int]:
        return sorted(self.header_by_game_num.keys())

    # --------------------
    # IO
    # --------------------
    def _read_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, **self.csv_kwargs)

    def _read_parquet(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(path, **self.parquet_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read parquet: {path}. "
                f"Install parquet engine (e.g. `pip install pyarrow`) or pass working engine params. "
                f"Original error: {e}"
            ) from e

    # --------------------
    # Helpers
    # --------------------
    @staticmethod
    def filter_by_game_num(df: pd.DataFrame, game_num: int) -> pd.DataFrame:
        if "game_num" not in df.columns:
            raise KeyError("DataFrame has no `game_num` column, cannot split by map.")
        return df[df["game_num"] == int(game_num)].copy()

    # ----------------------------
    # updated_at parsing/formatting
    # ----------------------------
    @staticmethod
    def _parse_updated_at(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        s = str(value).strip()

        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(s2)
            except Exception:
                return None

        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    @staticmethod
    def _format_date_time(dt: datetime) -> Tuple[str, str]:
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")

    # ----------------------------
    # tickrate helpers
    # ----------------------------
    def _get_tickrate_for_game(self, game_num: int) -> Optional[float]:
        cfg = self.clock_cfg_by_game_num.get(int(game_num))
        if not cfg:
            return None
        return cfg.tickrate_demoparser2 or cfg.tickrate_rounds_median

    # ----------------------------
    # last tick getters per map
    # ----------------------------
    def _build_last_tick_by_game_cache(self) -> None:
        if self._last_tick_by_game_cache is not None:
            return
        cache: Dict[int, int] = {}
        for g in self.game_nums:
            df = self.filter_by_game_num(self.df_ticks, g)
            if df.empty:
                continue
            cache[g] = int(df["tick"].max())
        self._last_tick_by_game_cache = cache

    def get_last_tick_value(self, game_num: int) -> int:
        self._build_last_tick_by_game_cache()
        game_num = int(game_num)
        if game_num not in self._last_tick_by_game_cache:
            raise ValueError(f"No ticks for game_num={game_num}")
        return int(self._last_tick_by_game_cache[game_num])

    def get_last_round_num(self, game_num: int) -> int:
        game_num = int(game_num)
        df = self.filter_by_game_num(self.df_ticks, game_num)
        if df.empty:
            raise ValueError(f"No ticks for game_num={game_num}")
        last_tick = int(df["tick"].max())
        df_last = df[df["tick"] == last_tick]
        return int(df_last["round_num"].max())

    def get_map_name_for_game(self, game_num: int) -> Optional[str]:
        game_num = int(game_num)
        df = self.filter_by_game_num(self.df_ticks, game_num)
        if df.empty or "map" not in df.columns:
            return None
        vals = df["map"].dropna().unique().tolist()
        return vals[0] if vals else None

    # ----------------------------
    # base fields for Rimble frame
    # ----------------------------
    def build_base_fields_for_map_last_tick(self, game_num: int) -> Dict[str, Any]:
        game_num = int(game_num)
        last_tick = self.get_last_tick_value(game_num)
        round_num = self.get_last_round_num(game_num)
        current_map = self.get_map_name_for_game(game_num)

        updated_at_raw = self.external_meta.get("updated_at")
        dt = self._parse_updated_at(updated_at_raw)

        if dt is None:
            dt = datetime.now(timezone.utc)
            date_str, time_str = self._format_date_time(dt)
            updated_at_out = dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_str, time_str = self._format_date_time(dt)
            updated_at_out = updated_at_raw

        started_default = True
        finished_default = any(g > game_num for g in self.game_nums)
        forfeited = bool(self.external_meta.get("forfeited", False))

        if "matchid" in self.external_meta:
            matchid_out = str(self.external_meta["matchid"])
        else:
            matchid_out = str(self.df_ticks["match_id"].iloc[0]) if (not self.df_ticks.empty and "match_id" in self.df_ticks.columns) else ""

        return {
            "timeframe": int(last_tick),  # пока приравниваем к tick
            "date": date_str,
            "time": time_str,

            "matchid": matchid_out,
            "match_status": self.external_meta.get("match_status", "UNKNOWN"),
            "league": self.external_meta.get("league"),
            "match_format": self.external_meta.get("match_format"),

            "current_map": current_map,
            "game_num": int(game_num),
            "round_num": int(round_num),

            "started": bool(self.external_meta.get("started", started_default)),
            "started_at": self.external_meta.get("started_at"),

            "finished": bool(self.external_meta.get("finished", finished_default)),
            "match_duration_seconds": self.external_meta.get("match_duration_seconds"),

            "forfeited": forfeited,
            "updated_at": updated_at_out,
        }

    # ----------------------------
    # map bounds + map info
    # ----------------------------
    @staticmethod
    def _norm_map_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        n = str(name).strip().lower()
        if n.startswith("de_"):
            n = n[3:]
        return n

    @staticmethod
    def _uuid5(namespace: uuid.UUID, value: str) -> str:
        return str(uuid.uuid5(namespace, value))

    def _compute_bounds_from_ticks(
        self,
        game_num: int,
        *,
        padding: float = 200.0,
        drop_zeros: bool = True,
        use_quantiles: bool = True,
        q_low: float = 0.001,
        q_high: float = 0.999,
    ) -> Optional[Dict[str, Dict[str, float]]]:
        game_num = int(game_num)
        df = self.filter_by_game_num(self.df_ticks, game_num)
        if df.empty or "X" not in df.columns or "Y" not in df.columns:
            return None

        xy = df[["X", "Y"]].copy().dropna()
        if xy.empty:
            return None

        if drop_zeros:
            xy = xy[(xy["X"] != 0) | (xy["Y"] != 0)]
            if xy.empty:
                return None

        if use_quantiles:
            try:
                x_min = float(xy["X"].quantile(q_low))
                x_max = float(xy["X"].quantile(q_high))
                y_min = float(xy["Y"].quantile(q_low))
                y_max = float(xy["Y"].quantile(q_high))
            except Exception:
                x_min = float(xy["X"].min())
                x_max = float(xy["X"].max())
                y_min = float(xy["Y"].min())
                y_max = float(xy["Y"].max())
        else:
            x_min = float(xy["X"].min())
            x_max = float(xy["X"].max())
            y_min = float(xy["Y"].min())
            y_max = float(xy["Y"].max())

        if not all(math.isfinite(v) for v in [x_min, x_max, y_min, y_max]):
            return None

        return {
            "min": {"x": x_min - padding, "y": y_min - padding},
            "max": {"x": x_max + padding, "y": y_max + padding},
        }

    def _map_info(self, game_num: int) -> Dict[str, Any]:
        map_name_raw = self.get_map_name_for_game(game_num)
        name_norm = self._norm_map_name(map_name_raw) or ""
        bounds = self._compute_bounds_from_ticks(game_num)

        return {
            "id": self._uuid5(uuid.NAMESPACE_URL, f"rimble-map:{name_norm}"),
            "name": name_norm if name_norm else map_name_raw,
            "bounds": bounds,
        }

    # ----------------------------
    # Round tick cache from df_rounds
    # ----------------------------
    def _build_round_tick_cache(self) -> None:
        if self._round_tick_cache is not None:
            return

        cache: Dict[Tuple[int, int], Dict[str, Optional[int]]] = {}

        if self.df_rounds is None or self.df_rounds.empty:
            self._round_tick_cache = cache
            return

        for _, row in self.df_rounds.iterrows():
            try:
                g = int(row["game_num"])
                r = int(row["round_num"])
            except Exception:
                continue

            def _to_int_or_none(v: Any) -> Optional[int]:
                try:
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        return None
                    return int(v)
                except Exception:
                    return None

            cache[(g, r)] = {
                "start": _to_int_or_none(row.get("start")),
                "freeze_end": _to_int_or_none(row.get("freeze_end")),
                "end": _to_int_or_none(row.get("end")),
                "official_end": _to_int_or_none(row.get("official_end")),
            }

        self._round_tick_cache = cache

    def get_round_ticks(self, game_num: int, round_num: int) -> Dict[str, Optional[int]]:
        self._build_round_tick_cache()
        return (self._round_tick_cache.get((int(game_num), int(round_num))) or {}).copy()

    def get_round_num_at_tick(self, game_num: int, tick: int) -> int:
        df = self.filter_by_game_num(self.df_ticks, int(game_num))
        df_t = df[df["tick"] == int(tick)]
        if df_t.empty:
            df2 = df[df["tick"] <= int(tick)]
            if df2.empty:
                raise ValueError(f"No ticks for game_num={game_num} up to tick={tick}")
            return int(df2["round_num"].max())
        return int(df_t["round_num"].max())

    # ----------------------------
    # Bomb state helpers (from df_bomb)
    # ----------------------------
    def _get_active_bomb_remaining_seconds(
        self,
        game_num: int,
        round_num: int,
        tick: int,
        tickrate: float,
        bomb_time_seconds: float,
    ) -> Optional[float]:
        if self.df_bomb is None or self.df_bomb.empty:
            return None

        df = self.df_bomb
        df = df[
            (df["game_num"] == int(game_num))
            & (df["round_num"] == int(round_num))
            & (df["tick"] <= int(tick))
        ]
        if df.empty:
            return None

        plants = df[df["event"].astype(str).str.lower().str.contains("plant", na=False)]
        if plants.empty:
            return None

        plant_tick = int(plants["tick"].max())
        after_plant = df[df["tick"] >= plant_tick]
        if after_plant.empty:
            return None

        ended = after_plant[
            after_plant["event"].astype(str).str.lower().str.contains("defus|explod|detonat", na=False)
        ]
        if not ended.empty:
            return None

        elapsed = (int(tick) - int(plant_tick)) / float(tickrate)
        remaining = max(0.0, float(bomb_time_seconds) - float(elapsed))
        return remaining

    # ----------------------------
    # Clock computation
    # ----------------------------
    def compute_round_clock_at_tick(self, game_num: int, tick: int) -> Dict[str, Any]:
        game_num = int(game_num)
        tick = int(tick)

        tickrate = self._get_tickrate_for_game(game_num)
        cfg = self.clock_cfg_by_game_num.get(game_num) or ClockConfig()

        if not tickrate or tickrate <= 0:
            return {"currentSeconds": 0, "ticking": False, "ticksBackwards": True, "type": "gameClock"}

        round_num = self.get_round_num_at_tick(game_num, tick)
        rticks = self.get_round_ticks(game_num, round_num)

        freeze_end_tick = rticks.get("freeze_end")
        start_tick = rticks.get("start")

        # fallback
        if freeze_end_tick is None:
            if start_tick is not None:
                freeze_end_tick = int(start_tick + int(cfg.freeze_time_seconds * tickrate))
            else:
                freeze_end_tick = tick

        round_time = float(cfg.round_time_seconds)

        if tick < int(freeze_end_tick):
            ticking = False
            current_seconds = round_time
        else:
            ticking = True
            active_elapsed = (tick - int(freeze_end_tick)) / float(tickrate)
            current_seconds = max(0.0, round_time - active_elapsed)

        # bomb override if active
        bomb_remaining = self._get_active_bomb_remaining_seconds(
            game_num=game_num,
            round_num=round_num,
            tick=tick,
            tickrate=float(tickrate),
            bomb_time_seconds=float(cfg.bomb_time_seconds),
        )
        if bomb_remaining is not None:
            ticking = True
            current_seconds = bomb_remaining

        return {"currentSeconds": current_seconds, "ticking": ticking, "ticksBackwards": True, "type": "gameClock"}

    def _clock_info_at_tick(self, game_num: int, tick: int, *, finished: bool) -> Dict[str, Any]:
        """
        ВАЖНО: даже для finished карты currentSeconds оставляем как на её last_tick.
        Но ticking делаем False.
        """
        matchid = str(self.external_meta.get("matchid", "unknown-match"))
        c = self.compute_round_clock_at_tick(int(game_num), int(tick))
        cur = c["currentSeconds"]
        ticking = False if finished else c["ticking"]

        return {
            "currentSeconds": cur,
            "id": self._uuid5(uuid.NAMESPACE_URL, f"rimble-clock:{matchid}:game:{int(game_num)}"),
            "ticking": ticking,
            "ticksBackwards": True,
            "type": "gameClock",
        }

    # ----------------------------
    # Game duration (optional)
    # ----------------------------
    def _estimate_game_duration_seconds(self, game_num: int) -> Optional[float]:
        if self.df_rounds is None or self.df_rounds.empty:
            return None

        df_r = self.filter_by_game_num(self.df_rounds, int(game_num))
        if df_r.empty:
            return None

        tickrate = self._get_tickrate_for_game(int(game_num))
        if not tickrate or tickrate <= 0:
            return None

        try:
            start_min = int(pd.to_numeric(df_r["start"], errors="coerce").dropna().min())
        except Exception:
            return None

        end_max = None
        for col in ("official_end", "end"):
            if col in df_r.columns:
                try:
                    v = pd.to_numeric(df_r[col], errors="coerce").dropna()
                    if not v.empty:
                        end_max = int(v.max())
                        break
                except Exception:
                    pass

        if end_max is None or end_max < start_min:
            return None

        return (end_max - start_min) / float(tickrate)

    # ----------------------------
    # METADATA builders (cumulative)
    # ----------------------------
    def build_metadata_item_for_game_at_tick(
        self,
        game_num: int,
        tick: int,
        *,
        is_current_game: bool,
        finished_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        game_num = int(game_num)
        tick = int(tick)

        map_info = self._map_info(game_num)

        finished_default = (not is_current_game)
        finished = bool(finished_override) if finished_override is not None else finished_default

        started_default = True
        forfeited = bool(self.external_meta.get("forfeited", False))
        paused = bool(self.external_meta.get("paused", False))
        started_at = self.external_meta.get("started_at")

        per_game_dur = self.external_meta.get("game_duration_seconds_by_game", {})
        game_duration_seconds = per_game_dur.get(game_num) if isinstance(per_game_dur, dict) else None
        if game_duration_seconds is None:
            game_duration_seconds = self._estimate_game_duration_seconds(game_num)

        return {
            "game_number": int(game_num),
            "map": map_info,
            "started": bool(self.external_meta.get("started", started_default)),
            "started_at": started_at,
            "finished": finished,
            "forfeited": forfeited,
            "paused": paused,
            "clock": self._clock_info_at_tick(game_num, tick, finished=finished),
            "structures": [],
            "non_player_characters": [],
            "game_duration_seconds": game_duration_seconds,
        }

    def build_metadata_for_frame_last_tick(self, current_game_num: int) -> List[Dict[str, Any]]:
        current_game_num = int(current_game_num)
        self._build_last_tick_by_game_cache()

        md: List[Dict[str, Any]] = []
        for g in self.game_nums:
            if g > current_game_num:
                break
            is_current = (g == current_game_num)
            tick = int(self._last_tick_by_game_cache[g])
            md.append(self.build_metadata_item_for_game_at_tick(g, tick, is_current_game=is_current))
        return md

    # ----------------------------
    # TEAMS (shallow)
    # ----------------------------
    def _team_side_to_rimble(self, side: Optional[str]) -> Optional[str]:
        if not side:
            return None
        s = str(side).strip().upper()
        if s == "T":
            return "terrorists"
        if s == "CT":
            return "counter-terrorists"
        if s.lower() in ("terrorists", "counter-terrorists"):
            return s.lower()
        return None

    def _make_team_id(self, designation: int, team_name: str) -> str:
        matchid = str(self.external_meta.get("matchid", "unknown-match"))
        base = f"{matchid}|team|{designation}|{team_name}"
        return str(uuid.uuid5(self._UUID_NS_TEAMS, base))

    def get_teams_from_header(self, game_num: int) -> Dict[str, Dict[str, Any]]:
        bg = (self.header_all or {}).get("by_game_num") or {}
        node = bg.get(str(game_num)) or {}
        meta = node.get("meta") or {}
        t1 = meta.get("team1") or {}
        t2 = meta.get("team2") or {}
        return {
            "team1": {"name": t1.get("name"), "side": t1.get("side")},
            "team2": {"name": t2.get("name"), "side": t2.get("side")},
        }

    def _calc_current_score_for_map(self, game_num: int, round_num: int) -> Tuple[int, int]:
        if self.df_rounds is None or self.df_rounds.empty:
            return (0, 0)
    
        df = self.filter_by_game_num(self.df_rounds, int(game_num))
        if df.empty:
            return (0, 0)
    
        df = df[df["round_num"] <= int(round_num)]
    
        col = "team_winner" if "team_winner" in df.columns else ("winner" if "winner" in df.columns else None)
        if not col:
            return (0, 0)
    
        h = self.get_teams_from_header(int(game_num))
        team1_name = (h["team1"].get("name") or "").strip().lower()
        team2_name = (h["team2"].get("name") or "").strip().lower()
        t1_side = self._team_side_to_rimble(h["team1"].get("side"))
    
        s1 = 0
        s2 = 0
    
        for v in df[col].tolist():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
    
            # numeric 1/2
            if isinstance(v, (int, float)):
                iv = int(v)
                if iv == 1:
                    s1 += 1
                    continue
                if iv == 2:
                    s2 += 1
                    continue
    
            vv = str(v).strip().lower()
            if not vv:
                continue
    
            # direct tokens
            if "team1" in vv or vv in ("1", "t1"):
                s1 += 1
                continue
            if "team2" in vv or vv in ("2", "t2"):
                s2 += 1
                continue
    
            # by team names (если в winner пишут название команды)
            if team1_name and team1_name in vv:
                s1 += 1
                continue
            if team2_name and team2_name in vv:
                s2 += 1
                continue
    
            # by side tokens
            # осторожно: "ct" может встречаться как часть слова, но обычно ок
            if vv in ("t", "terrorists") or "terrorist" in vv or vv.endswith("_t"):
                if t1_side == "terrorists":
                    s1 += 1
                else:
                    s2 += 1
                continue
    
            if vv in ("ct", "counter-terrorists", "counter_terrorists") or "counter" in vv or "ct_" in vv or vv.endswith("_ct"):
                if t1_side == "counter-terrorists":
                    s1 += 1
                else:
                    s2 += 1
                continue
    
            # если совсем не распознали — просто пропускаем
    
        return (s1, s2)


    def build_teams_shallow_for_map(self, game_num: int, round_num: int) -> List[Dict[str, Any]]:
        h = self.get_teams_from_header(int(game_num))
        team1_name = h["team1"].get("name") or "Team 1"
        team2_name = h["team2"].get("name") or "Team 2"
    
        team1_side = self._team_side_to_rimble(h["team1"].get("side"))
        team2_side = self._team_side_to_rimble(h["team2"].get("side"))
    
        # текущий счёт именно на текущей карте
        score1, score2 = self._calc_current_score_for_map(int(game_num), int(round_num))
    
        # cumulative results (карты 1..game_num)
        res1, res2, gw1, gw2 = self.build_results_cumulative_for_current_game(
            current_game_num=int(game_num),
            current_round_num=int(round_num),
        )
    
        # loss_streak пока оставляем заглушкой = 0 (позже сделаем)
        return [
            {
                "designation": 1,
                "name": team1_name,
                "id": self._make_team_id(1, team1_name),
                "side": team1_side,
    
                "loss_streak": 0,
    
                "picked_maps": [],
                "banned_maps": [],
                "remaining_available_maps": [],
                "map_side_picked": {},
    
                "current_score": int(score1),
                "games_won": int(gw1),
    
                "results": res1,
                "players": [],
            },
            {
                "designation": 2,
                "name": team2_name,
                "id": self._make_team_id(2, team2_name),
                "side": team2_side,
    
                "loss_streak": 0,
    
                "picked_maps": [],
                "banned_maps": [],
                "remaining_available_maps": [],
                "map_side_picked": {},
    
                "current_score": int(score2),
                "games_won": int(gw2),
    
                "results": res2,
                "players": [],
            },
        ]


    # ----------------------------
    # Final builder: base + metadata + teams
    # ----------------------------
    def build_rimble_json_for_map_last_tick(self, game_num: int) -> Dict[str, Any]:
        """
        Пока собираем:
          - base fields
          - metadata (cumulative)
          - teams (shallow)
        """
        base = self.build_base_fields_for_map_last_tick(int(game_num))
        base["metadata"] = self.build_metadata_for_frame_last_tick(int(game_num))
        base["teams"] = self.build_teams_shallow_for_map(
            game_num=int(game_num),
            round_num=int(base.get("round_num") or 0),
        )
        return base

    def _get_max_round_num_for_game(self, game_num: int) -> int:
        """
        Максимальный round_num, который есть в df_rounds для данной карты.
        """
        df = self.filter_by_game_num(self.df_rounds, int(game_num))
        if df.empty or "round_num" not in df.columns:
            return 0
        return int(df["round_num"].max())
    
    
    def _get_map_name_norm_for_game(self, game_num: int) -> Optional[str]:
        return self._norm_map_name(self.get_map_name_for_game(int(game_num)))
    
    
    def _build_single_game_result(
        self,
        game_num: int,
        *,
        up_to_round_num: int,
        is_final: bool,
    ) -> Dict[str, Any]:
        """
        Один элемент results для одной карты (game_num).
        up_to_round_num:
          - для прошлых карт: max round_num этой карты
          - для текущей карты: текущий round_num (на последнем тике)
        """
        s1, s2 = self._calc_current_score_for_map(int(game_num), int(up_to_round_num))
    
        winner_designation: Optional[int] = None
        if is_final:
            if s1 > s2:
                winner_designation = 1
            elif s2 > s1:
                winner_designation = 2
    
        return {
            "type": "gameResult",          # просто маркер типа (пока)
            "game_number": int(game_num),
            "map": self._get_map_name_norm_for_game(int(game_num)),
            "score": {"team1": int(s1), "team2": int(s2)},
            "is_final": bool(is_final),
            "winner_designation": winner_designation,
        }
    
    
    def build_results_cumulative_for_current_game(
        self,
        current_game_num: int,
        current_round_num: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
        """
        Возвращает:
          (results_for_team1, results_for_team2, games_won_team1, games_won_team2)
    
        results одинаковые по содержанию для обеих команд (внутри есть общий счёт),
        но это ок для "поверхностного уровня". Позже можем сделать team-centric fields.
        """
        current_game_num = int(current_game_num)
        current_round_num = int(current_round_num)
    
        results: List[Dict[str, Any]] = []
        games_won_1 = 0
        games_won_2 = 0
    
        for g in self.game_nums:
            if g > current_game_num:
                break
    
            if g < current_game_num:
                # прошлая карта → финальный результат
                max_r = self._get_max_round_num_for_game(g)
                r = max_r
                is_final = True
            else:
                # текущая карта → текущий результат на last-tick
                r = current_round_num
                is_final = False
    
            gr = self._build_single_game_result(g, up_to_round_num=r, is_final=is_final)
            results.append(gr)
    
            if is_final:
                wd = gr.get("winner_designation")
                if wd == 1:
                    games_won_1 += 1
                elif wd == 2:
                    games_won_2 += 1
    
        # одинаковый список results кладём в обе команды (пока)
        return results, results, games_won_1, games_won_2

