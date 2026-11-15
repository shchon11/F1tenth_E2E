"""Maps added from recordings without editing code: one folder per map, discovered at import.

`maps.REAL` and `tracks._REAL` are the hand-written catalogue of the venues this project started
with. A map made later from a bag (`bag_map`, then `slam_map --install`) lands here instead, as
data, and is a catalogue entry like any other: `real:<id>` for the loader (`user:<id>` is the
same map), `real/<slug>` in the short grammar and the console's list, every obstacle family and
direction, `+warp<k>` variants, raceline caches keyed by its name.

    <maps dir>/<id>/<id>.yaml + image   the map as built (map_server pair, unknown cells kept)
    <maps dir>/<id>/map.json            how to load it (below)

`map.json` holds what a `maps.REAL` tuple holds, by name:

    slug           short id, lowercase, <= 12 characters (real/<slug>)
    display        name in lists
    note           one line, where it came from
    boundary       "duct" | "wall"
    min_clearance  [m] the traced centre line stays this far off anything solid
    seed_xy        [m] a point on the lane (keeps the right free region, picks the loop)
    despeckle_m2   [m^2] occupied specks smaller than this are scan noise
    direction      "ccw" | "cw": the way the car drove it in the recording
    from_ros_map   further keywords of `Track.from_ros_map`
    source         provenance (bag, time window, map source, scale check) -- not read by the loader

The maps dir is `$F1SIM_MAPS`, else `~/.f1sim/maps`: outside the repository, like the raceline and
bank caches the map's training needs anyway.

Torch-free and import-cheap on purpose: `tracks.py` (the console's catalogue) reads it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Dict, List, Optional

META = "map.json"


def maps_dir() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("F1SIM_MAPS") or os.path.join("~", ".f1sim", "maps")))


def _image_of(yaml_path: str) -> Optional[str]:
    """The image a map yaml names, found next to it (map_saver writes .pgm when the yaml says .png)."""
    img = ""
    try:
        with open(yaml_path) as f:
            for line in f:
                if line.strip().startswith("image:"):
                    img = line.split(":", 1)[1].strip().strip("'\"")
                    break
    except OSError:
        return None
    if not img:
        return None
    if not os.path.isabs(img):
        img = os.path.join(os.path.dirname(yaml_path), img)
    if os.path.exists(img):
        return img
    base = os.path.splitext(img)[0]
    for ext in (".pgm", ".png", ".jpg", ".jpeg"):
        if os.path.exists(base + ext):
            return base + ext
    return None


def slug_for(map_id: str, taken=()) -> str:
    """A short id for `real/<slug>`: lowercase letters, digits and '-', at most 12 characters."""
    base = re.sub(r"[^a-z0-9-]+", "", map_id.lower().replace("_", "-")).strip("-")[:12] or "map"
    slug, n = base, 2
    while slug in set(taken):
        tail = str(n)
        slug, n = base[:12 - len(tail)] + tail, n + 1
    return slug


def _entry(d: str, map_id: str) -> Optional[dict]:
    yaml_path = os.path.join(d, map_id + ".yaml")
    meta_path = os.path.join(d, META)
    if not os.path.isfile(yaml_path) or _image_of(yaml_path) is None:
        return None
    meta = {}
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path))
        except (OSError, ValueError):
            meta = {}
    e = dict(id=map_id, yaml=yaml_path, slug=str(meta.get("slug") or slug_for(map_id)),
             display=str(meta.get("display") or map_id), note=str(meta.get("note") or f"{map_id}, from a recording"),
             boundary=str(meta.get("boundary") or "duct"), min_clearance=float(meta.get("min_clearance", 0.35)),
             seed_xy=tuple(meta["seed_xy"]) if meta.get("seed_xy") else None,
             despeckle_m2=meta.get("despeckle_m2"), direction=str(meta.get("direction") or "ccw"),
             from_ros_map=dict(meta.get("from_ros_map") or {}), source=meta.get("source") or {})
    return e


def entries(root: Optional[str] = None) -> List[dict]:
    """Every map in the maps dir, sorted by id. A folder without a readable `<id>.yaml` + image is skipped."""
    root = root or maps_dir()
    if not os.path.isdir(root):
        return []
    out = []
    for map_id in sorted(os.listdir(root)):
        d = os.path.join(root, map_id)
        if os.path.isdir(d):
            e = _entry(d, map_id)
            if e is not None:
                out.append(e)
    return out


def get(map_id: str, root: Optional[str] = None) -> Optional[dict]:
    root = root or maps_dir()
    d = os.path.join(root, map_id)
    return _entry(d, map_id) if os.path.isdir(d) else None


def install(yaml_src: str, map_id: str, meta: Dict, *, root: Optional[str] = None, overwrite: bool = False) -> str:
    """Copy a map_server pair into `<maps dir>/<map_id>/` as `<map_id>.yaml` + image, with `meta` as
    map.json. The map is copied as built -- unknown cells and all -- and cleaned by the loader with the
    meta's keywords every time, exactly like a `maps.REAL` entry, so `real:<map_id>` loads the same
    track whichever way it is reached."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", map_id):
        raise ValueError(f"지도 이름은 영문·숫자·_·- 만 됩니다: {map_id!r}")
    root = root or maps_dir()
    d = os.path.join(root, map_id)
    if os.path.exists(os.path.join(d, map_id + ".yaml")) and not overwrite:
        raise FileExistsError(f"{d} 에 이미 있습니다 (덮어쓰려면 overwrite)")
    img = _image_of(yaml_src)
    if img is None:
        raise FileNotFoundError(f"{yaml_src}: 그림 파일을 찾지 못했습니다")
    os.makedirs(d, exist_ok=True)
    ext = os.path.splitext(img)[1]
    shutil.copyfile(img, os.path.join(d, map_id + ext))
    lines = []
    with open(yaml_src) as f:
        for line in f:
            lines.append(f"image: {map_id}{ext}\n" if line.strip().startswith("image:") else line)
    with open(os.path.join(d, map_id + ".yaml"), "w") as f:
        f.writelines(lines)
    taken = [e["slug"] for e in entries(root) if e["id"] != map_id]
    meta = dict(meta)
    meta.setdefault("slug", slug_for(map_id, taken))
    json.dump(meta, open(os.path.join(d, META), "w"), indent=1, ensure_ascii=False)
    return os.path.join(d, map_id + ".yaml")
