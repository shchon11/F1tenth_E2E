"""Physics profiles: per-car-setup YAML files with value, error, source and identification status for every entry,
layered on top of the simulator's defaults WITHOUT changing them.

    from f1sim.profiles.loader import load_profile, config_from_profile, make_simulator
    prof = load_profile("competition_2026_08")           # a name in this folder, or a path
    cfg = config_from_profile(prof)                      # params.Config with the profile's values applied
    sim = make_simulator(prof, track, num_envs=16)       # sim.Simulator on that Config

A profile entry looks like

    vehicle:
      B_f: {value: 0.97, err: 0.10, unit: "1/rad", source: estimated, status: measured, note: "..."}

`source`  measured (read straight from a bag) | estimated (fitted from bags) | spec (data sheet / ini) | assumed
`status`  measured | lower_bound (only a bound is in the data) | not_identified (the data cannot tell; value kept
          from the default or a spec, its sensitivity stated) | not_applied (recorded, but the simulator has no
          parameter for it -- e.g. the asymmetric servo window)

Only entries whose group/field exist in `params.Config` and whose `apply` is not false are written into the Config;
the rest stay in the profile as facts (`unapplied()` lists them). `randomization` entries widen or narrow the
domain-randomisation ranges in `Config.rand.ranges`, so an unidentified value can be trained over its plausible span.
`params.Config.from_dict` does the writing -- the existing, so far uncalled-by-training, override path.
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = ("vehicle", "actuator", "lidar", "odom", "imu", "sim")
SOURCES = ("measured", "estimated", "spec", "assumed")
STATUSES = ("measured", "lower_bound", "not_identified", "not_applied")


def available() -> List[str]:
    return sorted(f[:-5] for f in os.listdir(HERE) if f.endswith(".yaml"))


def load_profile(name_or_path: str) -> dict:
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(HERE, name_or_path if name_or_path.endswith(".yaml") else name_or_path + ".yaml")
    with open(path) as f:
        prof = yaml.safe_load(f) or {}
    prof.setdefault("_path", os.path.abspath(path))
    validate(prof)
    return prof


def _entries(prof: dict):
    for grp in GROUPS:
        for k, e in (prof.get(grp) or {}).items():
            yield grp, k, e


def validate(prof: dict) -> None:
    """Every entry has a value, a known source and a known status. Raises ValueError listing what is wrong."""
    bad = []
    for grp, k, e in _entries(prof):
        if not isinstance(e, dict) or "value" not in e:
            bad.append(f"{grp}.{k}: entry must be a mapping with 'value'")
            continue
        if e.get("source") not in SOURCES:
            bad.append(f"{grp}.{k}: source {e.get('source')!r} not in {SOURCES}")
        if e.get("status") not in STATUSES:
            bad.append(f"{grp}.{k}: status {e.get('status')!r} not in {STATUSES}")
    if bad:
        raise ValueError("profile %s:\n  " % prof.get("_path", "?") + "\n  ".join(bad))


def overrides(prof: dict) -> Tuple[dict, List[str]]:
    """({group: {field: value}} for Config.from_dict, [entries that are not applied and why])."""
    from ..params import Config
    base = Config()
    d: Dict[str, dict] = {}
    skipped = []
    for grp, k, e in _entries(prof):
        if e.get("apply") is False or e.get("status") == "not_applied":
            skipped.append(f"{grp}.{k}: {e.get('note', 'not applied')}")
            continue
        if not hasattr(getattr(base, grp), k):
            skipped.append(f"{grp}.{k}: no such field in params.Config (kept as a fact only)")
            continue
        cur = getattr(getattr(base, grp), k)
        v = e["value"]
        if isinstance(cur, bool):
            v = bool(v)
        elif isinstance(cur, int) and not isinstance(cur, bool):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        d.setdefault(grp, {})[k] = v
    rand = prof.get("randomization") or {}
    if rand:
        d["rand"] = {"ranges": {k: tuple(v["range"] if isinstance(v, dict) else v) for k, v in rand.items()}}
    return d, skipped


def config_from_profile(prof, base_dict: Optional[dict] = None):
    """params.Config with the profile applied over the defaults (and over `base_dict`, if given, first)."""
    from ..params import Config
    if isinstance(prof, str):
        prof = load_profile(prof)
    d, _ = overrides(prof)
    if base_dict:
        merged = copy.deepcopy(base_dict)
        for g, kv in d.items():
            if g == "rand":
                merged.setdefault("rand", {}).setdefault("ranges", {}).update(kv["ranges"])
            else:
                merged.setdefault(g, {}).update(kv)
        d = merged
    return Config.from_dict(d)


def unapplied(prof) -> List[str]:
    if isinstance(prof, str):
        prof = load_profile(prof)
    return overrides(prof)[1]


def make_simulator(prof, track, num_envs: int = 1, device: Optional[str] = None, base_dict: Optional[dict] = None,
                   **kw):
    """sim.Simulator on the profile's Config."""
    from ..sim import Simulator
    cfg = config_from_profile(prof, base_dict)
    return Simulator(track, cfg=cfg, num_envs=num_envs, device=device, **kw)


def table(prof) -> str:
    """one line per entry: group.field value +- err unit [source/status] -- for logs and the console."""
    if isinstance(prof, str):
        prof = load_profile(prof)
    rows = []
    for grp, k, e in _entries(prof):
        err = e.get("err")
        es = "" if err is None else (f" +- {err}" if not isinstance(err, (list, tuple)) else f" [{err[0]}, {err[1]}]")
        rows.append(f"{grp}.{k} = {e['value']}{es} {e.get('unit', '')} [{e['source']}/{e['status']}]")
    return "\n".join(rows)
