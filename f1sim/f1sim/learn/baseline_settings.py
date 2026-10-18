"""The per-map tuned settings of the baselines, for the console (2026-10-19).

The fair comparison tunes every baseline per map (`pipeline/tools/baselines/tune*.py`), and the user
judges methods in the console -- where, until this, a baseline drove on its published defaults while
the table reported its tuned self. This file is the bridge: `SETTINGS_FILE` holds, per map (the
track's name) and per method, the setting the report used, one for racing (`race`) and one for the
time trial (`solo`); the console's "내 차" uses it when "맵별 튜닝 설정" is on.

Format (written by `pipeline/tools/baselines/export_console_settings.py`)::

    {"version": 1,
     "maps": {"korea_2025_iccas": {"forzaeth": {"race": {"stack_params": {...}, "limits": [8, 7, 5],
                                                         "source": "tune2 race #11"},
                                                "solo": {...}},
                                   "ftg": {"race": {"options": {...}, "source": "..."}, ...},
                                   "tinylidarnet": {"race": {"options": {...}, "weights": "...pt"}}}}}
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .common import RUNS_DIR

SETTINGS_FILE = os.path.join(RUNS_DIR, "baseline_settings.json")


def load_all(path: Optional[str] = None) -> dict:
    p = path or SETTINGS_FILE
    if not os.path.isfile(p):
        return {}
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return d.get("maps", {}) if isinstance(d, dict) else {}


def lookup(track_name: str, kind: str, solo: bool, path: Optional[str] = None) -> Optional[dict]:
    """The tuned setting of `kind` on `track_name` for a session with (`solo=False`) or without
    opponents, or None when there is none. A race setting stands in for a missing solo one and the
    other way round: the method's own tuned numbers beat its published defaults either way."""
    per = (load_all(path).get(track_name) or {}).get(kind) or {}
    first, second = ("solo", "race") if solo else ("race", "solo")
    s = per.get(first) or per.get(second)
    return dict(s) if isinstance(s, dict) else None
