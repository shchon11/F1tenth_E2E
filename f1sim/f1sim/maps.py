"""Map catalog: procedural styles, f1tenth_gym SLAM/drawn maps, f1tenth_racetracks (1:10 real circuits).

Names:  gen:competition:7   gen:hallway:2   gen:circuit:0
        rt:Spielberg        (any directory of f1tenth_racetracks)
        gym:levine          (levine, berlin, skirk, vegas, stata_basement)
        /abs/path/map.yaml  (any ROS map; centerline csv next to it is picked up)
"""
from __future__ import annotations

import glob
import re
import os
import numpy as np
from typing import List, Optional

from .track import Track

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # src/f1sim
EXTERNAL = os.path.normpath(os.path.join(_HERE, "..", "external"))
RACETRACKS = os.path.join(EXTERNAL, "f1tenth_racetracks")
GYM_MAPS = os.path.join(EXTERNAL, "f1tenth_gym", "gym", "f110_gym", "envs", "maps")
GYM_BOUNDARY = {"levine": "wall", "stata_basement": "wall", "berlin": "duct", "skirk": "duct", "vegas": "duct"}

# SLAM maps of real competition venues. Values: (yaml, boundary, min clearance for the automatic
# centerline, note[, seed_xy of the lane, despeckle area m^2, extra from_ros_map kwargs]). Practice tracks (plechaty, korea_kim) were
# dropped from the catalog: not competition layouts.
TUW = os.path.join(EXTERNAL, "f1tenth_maps", "maps")
KOR = os.path.join(EXTERNAL, "korea_teams")
REAL = {
    "icra2022": (os.path.join(EXTERNAL, "f1tenth-racing-stack-ICRA22", "maps", "icra_2.yaml"), "duct", 0.45, "ICRA 2022 F1TENTH Grand Prix track (Philadelphia), team SLAM map"),
    "blackbox2021_1": (os.path.join(TUW, "blackbox2021_1.yaml"), "duct", 0.35, "TU Wien BlackBox race 2021"),
    "blackbox2021_2": (os.path.join(TUW, "blackbox2021_2.yaml"), "duct", 0.30, "TU Wien BlackBox race 2021", None, 0.15),
    "blackbox2021_3": (os.path.join(TUW, "blackbox2021_3.yaml"), "duct", 0.30, "TU Wien BlackBox race 2021"),
    "blackbox2022_1": (os.path.join(TUW, "blackbox2022_1.yaml"), "duct", 0.40, "TU Wien BlackBox race 2022"),
    "blackbox2022_2": (os.path.join(TUW, "blackbox2022_2.yaml"), "duct", 0.35, "TU Wien BlackBox race 2022", None, 0.15),
    "blackbox2022_3": (os.path.join(TUW, "blackbox2022_3.yaml"), "duct", 0.30, "TU Wien BlackBox race 2022"),
    # community SLAM maps added 2026-09-08 for layout diversity (ppo_v9 memorized its 8 layouts)
    "berlin": (os.path.join(GYM_MAPS, "berlin.yaml"), "duct", 0.40, "F1TENTH Berlin race map (f1tenth_gym)"),
    "skirk": (os.path.join(GYM_MAPS, "skirk.yaml"), "duct", 0.40, "Skirkanich hall race (f1tenth_gym)"),
    "columbia_small": (os.path.join(TUW, "columbia_small.yaml"), "duct", 0.40, "Columbia race map (f1tenth_maps)"),
    "torino_small": (os.path.join(TUW, "torino_redraw_small.yaml"), "duct", 0.35, "Torino race map, redrawn (f1tenth_maps)"),
    "mtl": (os.path.join(TUW, "mtl.yaml"), "duct", 0.40, "Montreal race map (f1tenth_maps)", None, 0.15),
    "porto": (os.path.join(TUW, "porto.yaml"), "duct", 0.30, "Porto race map, narrow (f1tenth_maps)"),
    "korea_2025_iccas": (os.path.join(KOR, "KORA_K3", "src", "kora_k3", "maps", "real.yaml"), "duct", 0.35, "4th F1TENTH Korea Championship 2025 (ICCAS) race track, team SLAM map", (3.0, 0.0), 0.05,
                         dict(keep_region=True, unknown_floor_depth=5.0)),   # raw SLAM: spray, specks. The outline is a hose too
                                                                              # (track built from ducts in a bigger hall): floor beyond
}
CENTERLINE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "f1sim", "centerlines")


def racetrack_names() -> List[str]:
    return sorted(os.path.basename(os.path.dirname(y)) for y in glob.glob(os.path.join(RACETRACKS, "*", "*_map.yaml")))


def gym_map_names() -> List[str]:
    return sorted(os.path.splitext(os.path.basename(y))[0] for y in glob.glob(os.path.join(GYM_MAPS, "*.yaml")))


def catalog() -> List[str]:
    names = [f"gen:{s}:<seed>" for s in ("competition", "hallway", "circuit")]
    names += [f"rt:{n}" for n in racetrack_names()] + [f"gym:{n}" for n in gym_map_names()] + [f"real:{n}" for n in REAL]
    return names


def _with_auto_centerline(track: Track, min_clearance: float, seed_xy=None) -> Track:
    """Extract (and cache) the main loop of a map that has no centerline."""
    import hashlib
    import numpy as np
    if track.centerline is not None:
        return track
    os.makedirs(CENTERLINE_CACHE, exist_ok=True)
    key = hashlib.md5(np.packbits(track.occupancy).tobytes() + f"{min_clearance}{seed_xy}v8".encode()).hexdigest()[:12]
    path = os.path.join(CENTERLINE_CACHE, f"{track.name}_{key}.csv")
    if os.path.exists(path):
        track.centerline = np.loadtxt(path, delimiter=",")
    else:
        track.centerline_from_free_space(min_clearance=min_clearance, seed_xy=seed_xy)
        np.savetxt(path, track.centerline, delimiter=",")
    return track


MODIFIERS = ("~rev", "~mir")
_BASE_CACHE = {}


def load(name: str, **kw) -> Track:
    """Catalog loader. Trailing modifiers: `~rev` = same map driven the other way round,
    `~mir` = mirrored map (e.g. real:icra2022~rev, rt:Monza~mir~rev)."""
    mods = []
    while name.endswith(MODIFIERS):
        for m in MODIFIERS:
            if name.endswith(m):
                mods.append(m); name = name[:-len(m)]
    ck = (name, repr(sorted(kw.items())))
    if ck not in _BASE_CACHE:                              # one copy per process: both directions share grids
        _BASE_CACHE[ck] = _load_base(name, **kw)
    t = _BASE_CACHE[ck]
    for m in reversed(mods):
        t = t.reversed() if m == "~rev" else t.mirrored()
    return t


def _split_suffix(name: str, key: str):
    """'real:x+obs1+pk3' , '+pk' -> ('real:x+obs1', '3'); no suffix -> (name, None)."""
    m = re.search(re.escape(key) + r"(\d+)", name)
    return (name[:m.start()] + name[m.end():], m.group(1)) if m else (name, None)


def _pockets(t: Track, pk):
    """x+pk<seed>: dead-end side pockets, one per ~25 m of lane (3-8)."""
    if pk is None:
        return t
    L = float(np.linalg.norm(np.roll(t.centerline, -1, 0) - t.centerline, axis=1).sum())
    return t.with_pockets(seed=int(pk), n=int(np.clip(round(L / 25.0), 3, 8)))


def _load_base(name: str, **kw) -> Track:
    if name.startswith("gen:"):
        _, style, seed = (name.split(":") + ["0"])[:3]
        seed, obs = (seed.split("+obs") + [None])[:2]            # gen:competition:3+obs -> random lane obstacles
        return Track.generate_random(int(seed), style=style, lane_obstacles=(obs is not None), **kw)
    name, pk = _split_suffix(name, "+pk")                    # x+pk<seed>: dead-end side pockets (after +obs)
    if name.startswith("rt:"):
        n = name[3:]
        d = os.path.join(RACETRACKS, n)
        ys = glob.glob(os.path.join(d, "*_map.yaml"))
        if not ys:
            raise FileNotFoundError(f"racetrack {n} not found under {RACETRACKS}")
        cl = glob.glob(os.path.join(d, "*_centerline.csv"))
        return _pockets(Track.from_ros_map(ys[0], centerline_csv=cl[0] if cl else None, name=n, **kw), pk)
    if name.startswith("gym:"):
        n = name[4:]
        kw.setdefault("boundary", GYM_BOUNDARY.get(n, "duct"))
        clearance = kw.pop("min_clearance", 0.45)
        return _with_auto_centerline(Track.from_ros_map(os.path.join(GYM_MAPS, n + ".yaml"), name=n, **kw), clearance)
    if name.startswith("real:"):
        n, obs = (name[5:].split("+obs") + [None])[:2]          # real:icra2022+obs3 -> 3 lane obstacles
        entry = REAL[n]; yaml_path, boundary, clearance = entry[:3]; seed_xy = entry[4] if len(entry) > 4 else None
        kw.setdefault("boundary", boundary)
        if len(entry) > 5 and entry[5] is not None:
            kw.setdefault("despeckle_m2", entry[5])
        if len(entry) > 6:
            for k_, v_ in entry[6].items():
                kw.setdefault(k_, v_)
            if kw.get("keep_region"):
                kw.setdefault("seed_xy", seed_xy)
        clearance = kw.pop("min_clearance", clearance)
        t = _with_auto_centerline(Track.from_ros_map(yaml_path, name=n, **kw), clearance, seed_xy)
        if obs is not None:                                     # real:x+obs<seed>: boxes every ~35 m of lane
            L = float(np.linalg.norm(np.roll(t.centerline, -1, 0) - t.centerline, axis=1).sum())
            t = t.with_lane_obstacles(seed=int(obs), n=int(np.clip(round(L / 35.0), 3, 8)))
        return _pockets(t, pk)
    clearance = kw.pop("min_clearance", None)
    t = Track.from_ros_map(name, **kw)
    return _with_auto_centerline(t, clearance) if clearance else t


def load_many(names, **kw) -> List[Track]:
    return [load(n, **kw) for n in names]


def random_set(n: int, seed: int = 0, styles=("competition", "competition", "hallway", "circuit")) -> List[Track]:
    """n procedural tracks, cycling through styles, seeds derived from `seed`."""
    return [Track.generate_random(seed * 1000 + i, style=styles[i % len(styles)]) for i in range(n)]
