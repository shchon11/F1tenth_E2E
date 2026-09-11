"""Map catalog: procedural styles, f1tenth_gym SLAM/drawn maps, f1tenth_racetracks (1:10 real circuits).

Names:  gen:competition:7   gen:hallway:2   gen:circuit:0
        rt:Spielberg        (any directory of f1tenth_racetracks)
        gym:levine          (levine, berlin, skirk, vegas, stata_basement)
        /abs/path/map.yaml  (any ROS map; centerline csv next to it is picked up)
    Obstacle suffixes: `+obs<seed>` = boxes hugging the lane edge (the racing line stays clear),
                       `+rlobs<seed>` = boxes standing on the racing line (the car must plan around)
"""
from __future__ import annotations

import glob
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
ASSET_MAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "maps")
KOR = os.path.join(EXTERNAL, "korea_teams")
REAL = {
    "icra2022": (os.path.join(EXTERNAL, "f1tenth-racing-stack-ICRA22", "maps", "icra_2.yaml"), "duct", 0.45, "ICRA 2022 F1TENTH Grand Prix track (Philadelphia), team SLAM map"),
    "blackbox2021_1": (os.path.join(TUW, "blackbox2021_1.yaml"), "duct", 0.35, "TU Wien BlackBox race 2021"),
    "blackbox2021_2": (os.path.join(TUW, "blackbox2021_2.yaml"), "duct", 0.30, "TU Wien BlackBox race 2021", None, 0.15),
    "blackbox2021_3": (os.path.join(TUW, "blackbox2021_3.yaml"), "duct", 0.30, "TU Wien BlackBox race 2021"),
    "blackbox2022_1": (os.path.join(TUW, "blackbox2022_1.yaml"), "duct", 0.40, "TU Wien BlackBox race 2022"),
    "blackbox2022_2": (os.path.join(TUW, "blackbox2022_2.yaml"), "duct", 0.35, "TU Wien BlackBox race 2022", None, 0.15),
    "blackbox2022_3": (os.path.join(TUW, "blackbox2022_3.yaml"), "duct", 0.30, "TU Wien BlackBox race 2022"),
    # The venue this car actually raced on, taken from the /map the stack was localising against in
    # real_data/01_competition (scripts/extract_bag_map.py). 8.4 x 22.6 m, duct hose boundary.
    "korea_2026_competition": (os.path.join(ASSET_MAPS, "korea_2026_competition.yaml"), "duct", 0.35,
                               "2026 competition venue, from the team's own recordings"),
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


def _raceline_obstacles(track: Track, seed: int, n: Optional[int] = None, **kw) -> Track:
    """Boxes standing on the *raceline*, so the fast line is blocked and the car has to plan around
    them. `+obs` hugs the wall and leaves the line clear; `+rlobs` does not."""
    from .raceline import Raceline
    rl = Raceline.build_cached(track)
    line = rl.xy
    kw.setdefault("speeds", rl.v)                    # the profile decides the reaction time a box creates
    if n is None:
        length = float(np.linalg.norm(np.roll(line, -1, 0) - line, axis=1).sum())
        n = int(np.clip(round(length / 25.0), 2, 10))          # roughly one box every 25 m of line
    return track.with_line_obstacles(line, seed=seed, n=n, **kw)


def _static_props(track: Track, seed: int, n: Optional[int] = None) -> Track:
    """`+props<seed>`: modelled obstacles -- boxes, crates, a drum -- placed the way `+obs` places its
    rectangles, but held as finite convex sections instead of stamped into the grid.

    A separate suffix rather than a replacement for `+obs`. That one rasterises rotated rectangles,
    and every checkpoint in the catalogue was trained against exactly those; redefining the name
    would move the ground under those runs without saying so. A caller picks which it wants."""
    if n is None:
        L = float(np.linalg.norm(np.roll(track.centerline, -1, 0) - track.centerline, axis=1).sum())
        n = int(np.clip(round(L / 35.0), 3, 8))
    return track.with_static_props(seed=seed, n=n)


def _split_obstacle_suffix(name: str):
    """`x+rlobs7` -> (x, 'rlobs', 7); `x+obs7` -> (x, 'obs', 7); `x+pinch7` -> (x, 'pinch', 7);
    `x+props7` -> (x, 'props', 7); otherwise (name, None, None).
    `+rlobs` must be tested first: splitting on '+obs' would cut 'x+rlobs7' into 'x+rl' and '7'."""
    for tag in ("+rlobs", "+obs", "+pinch", "+props"):
        if tag in name:
            base, _, spec = name.partition(tag)
            return base, tag[1:], spec
    return name, None, None


def _load_base(name: str, **kw) -> Track:
    if name.startswith("gen:"):
        _, style, seed = (name.split(":") + ["0"])[:3]
        seed, kind, spec = _split_obstacle_suffix(seed)          # gen:competition:3+obs7 / +rlobs7
        if kind == "rlobs":
            return _raceline_obstacles(Track.generate_random(int(seed), style=style, **kw), int(spec))
        if kind == "obs":
            obstacle_seed = int(spec)
            t = Track.generate_random(int(seed), style=style, **kw)
            n_obs = int(np.random.default_rng(obstacle_seed + 7).integers(1, 5))
            return t.with_lane_obstacles(seed=obstacle_seed, n=n_obs)
        if kind == "pinch":                                      # gen:x:3+pinch7: the lane closes down
            t = Track.generate_random(int(seed), style=style, **kw)
            return t.with_pinches(seed=int(spec), n=int(np.random.default_rng(int(spec) + 5).integers(2, 5)))
        if kind == "props":                                      # gen:x:3+props7: modelled props
            return _static_props(Track.generate_random(int(seed), style=style, **kw), int(spec))
        return Track.generate_random(int(seed), style=style, **kw)
    if name.startswith("rt:"):
        n, kind, spec = _split_obstacle_suffix(name[3:])        # rt:Monza+props3
        d = os.path.join(RACETRACKS, n)
        ys = glob.glob(os.path.join(d, "*_map.yaml"))
        if not ys:
            raise FileNotFoundError(f"racetrack {n} not found under {RACETRACKS}")
        cl = glob.glob(os.path.join(d, "*_centerline.csv"))
        t = Track.from_ros_map(ys[0], centerline_csv=cl[0] if cl else None, name=n, **kw)
        if kind == "props":                                     # rt:x+props<seed>: modelled props
            return _static_props(t, int(spec))
        if kind is not None:
            raise ValueError(f"racetrack names support '+props<seed>', not '+{kind}': {name!r}")
        return t
    if name.startswith("gym:"):
        n = name[4:]
        kw.setdefault("boundary", GYM_BOUNDARY.get(n, "duct"))
        clearance = kw.pop("min_clearance", 0.45)
        return _with_auto_centerline(Track.from_ros_map(os.path.join(GYM_MAPS, n + ".yaml"), name=n, **kw), clearance)
    if name.startswith("real:"):
        n, kind, spec = _split_obstacle_suffix(name[5:])        # real:icra2022+obs3 / +rlobs3
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
        if kind == "pinch":                                     # real:x+pinch<seed>: the lane closes down
            return t.with_pinches(seed=int(spec), n=int(np.random.default_rng(int(spec) + 5).integers(2, 5)))
        if kind == "rlobs":                                     # real:x+rlobs<seed>: boxes ON the raceline
            return _raceline_obstacles(t, int(spec))
        if kind == "obs":                                       # real:x+obs<seed>: boxes every ~35 m of lane
            L = float(np.linalg.norm(np.roll(t.centerline, -1, 0) - t.centerline, axis=1).sum())
            t = t.with_lane_obstacles(seed=int(spec), n=int(np.clip(round(L / 35.0), 3, 8)))
        if kind == "props":                                     # real:x+props<seed>: modelled props
            t = _static_props(t, int(spec))
        return t
    clearance = kw.pop("min_clearance", None)
    t = Track.from_ros_map(name, **kw)
    return _with_auto_centerline(t, clearance) if clearance else t


def load_many(names, **kw) -> List[Track]:
    return [load(n, **kw) for n in names]


def random_set(n: int, seed: int = 0, styles=("competition", "competition", "hallway", "circuit")) -> List[Track]:
    """n procedural tracks, cycling through styles, seeds derived from `seed`."""
    return [Track.generate_random(seed * 1000 + i, style=styles[i % len(styles)]) for i in range(n)]
