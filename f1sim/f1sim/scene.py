"""Scene documents for the environment editor (환경 page): the on-disk format, a torch-free
editing API, and the `python -m f1sim.scene` CLI that does the parts needing `Track`.

A scene is what the user builds interactively: two boolean layers on one grid (`duct` hoses the
LiDAR can look over, `tall` walls it cannot), placed props -- the six procedural styles plus
imported meshes -- and, once validated, a centerline. It becomes a `Track` through `to_track()`,
which is the only place the simulator's representation enters; everything else here is numpy.

On disk (root `~/f1sim_scenes`, or `$F1SIM_SCENES`)::

    <root>/<name>/scene.json      schema 1 -- everything but the grids
    <root>/<name>/layers.npz      duct, tall: bool (H, W), np.savez_compressed
    <root>/<name>/assets/<file>   imported meshes, copied in verbatim

Row index increases with +y, as in `Track`; `origin` is the lower-left corner of cell (0, 0);
the centre of cell (row, col) is `origin + (col + 0.5, row + 0.5) * resolution`.

This module never imports torch. `f1sim.track` is imported inside the few methods that need it
(`to_track`, `PropPlacement.to_static_prop`) and by the CLI, so the console process can import
this module at startup and hand the heavy work to `python -m f1sim.scene ...` as a subprocess.
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import props as _props

SCENE_SCHEMA_VERSION = 1
#: Default root; `$F1SIM_SCENES` overrides it. Read again on every use so a test (or a user) can
#: point it elsewhere after import.
SCENES_ROOT = os.environ.get("F1SIM_SCENES", "~/f1sim_scenes")
#: The narrowest lane `validate` accepts, metres between the walls on either side of the centerline.
MIN_CORRIDOR_M = 0.6
#: Clearance a spawn needs: the car's half diagonal (0.58 x 0.31 m body) plus a little.
SPAWN_CLEARANCE_M = 0.35
MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".stl", ".ply")
LAYERS = ("duct", "tall", "free")


def scenes_root() -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get("F1SIM_SCENES", SCENES_ROOT)))


def _is_path(s: str) -> bool:
    return os.sep in s or (os.altsep is not None and os.altsep in s) or s.startswith(("~", "."))


def _check_name(name: str) -> str:
    name = str(name).strip()
    if not name or _is_path(name) or name in (".", ".."):
        raise ValueError(f"scene name must be a plain directory name, got {name!r}")
    return name


def scene_dir(name_or_path: str) -> str:
    """`"name"` -> `<root>/name`; `"/abs/dir"` or `"/abs/dir/scene.json"` -> `/abs/dir`."""
    s = str(name_or_path).strip()
    if os.path.basename(s) == "scene.json":
        s = os.path.dirname(s)
    if _is_path(s):
        return os.path.abspath(os.path.expanduser(s))
    return os.path.join(scenes_root(), _check_name(s))


def list_scenes() -> List[dict]:
    """Saved scenes under the root, newest first. Pure filesystem: safe on the GUI thread."""
    out = []
    try:
        entries = list(os.scandir(scenes_root()))
    except OSError:
        return out
    for e in entries:
        if not e.is_dir():
            continue
        p = os.path.join(e.path, "scene.json")
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        out.append({
            "name": e.name, "dir": e.path,
            "modified": str(meta.get("modified") or ""),
            "props": len(meta.get("props") or []),
            "shape": [int(v) for v in (meta.get("shape") or [0, 0])],
            "source": dict(meta.get("source") or {}),
            "notes": str(meta.get("notes") or ""),
        })
    out.sort(key=lambda s: s["modified"], reverse=True)
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ==================================================================== props and assets
@dataclass
class AssetInfo:
    """One imported mesh file, copied under `<dir>/assets/`."""
    id: str
    file: str                       # relative to the scene dir, e.g. "assets/tire_stack.glb"
    name: str = ""
    up: str = "z"                   # "y" | "z": which file axis points up
    scale: float = 1.0              # import-time scale (target_height bakes into this)
    tris: int = 0
    size: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])   # (w, d, h) at `scale`
    collision: str = "bands"        # "bands" (per-band convex hulls) | "hull" (one prism)

    def to_json(self) -> dict:
        return {"id": self.id, "file": self.file, "name": self.name, "up": self.up,
                "scale": float(self.scale), "tris": int(self.tris),
                "size": [float(v) for v in self.size], "collision": self.collision}

    @staticmethod
    def from_json(d: dict) -> "AssetInfo":
        return AssetInfo(id=str(d["id"]), file=str(d["file"]), name=str(d.get("name") or ""),
                         up=str(d.get("up") or "z"), scale=float(d.get("scale", 1.0)),
                         tris=int(d.get("tris", 0)), size=[float(v) for v in (d.get("size") or [0, 0, 0])],
                         collision=str(d.get("collision") or "bands"))


#: Built props by resolved spec, shared by every placement: `footprint_world` is called on every
#: hover / drag, and a build is milliseconds for a crate and a file load for a mesh.
_PROP_CACHE: Dict[tuple, "_props.Prop"] = {}


@dataclass
class PropPlacement:
    """One placed prop: `track.StaticProp` plus an id and, for meshes, the asset it refers to.

    `dims` is a dict (JSON-friendly); for style `mesh` it holds an optional per-placement `scale`
    multiplier, and the file path comes from the asset. `doc` is a back-reference the owning
    `SceneDoc` sets so `build()` needs no argument -- that is what lets
    `viewer.geometry.build_track_geometry(doc)` treat placements exactly like `StaticProp`s.
    """
    id: str
    style: str
    x: float
    y: float
    yaw: float = 0.0
    seed: int = 0
    dims: dict = field(default_factory=dict)
    asset: Optional[str] = None
    doc: Optional["SceneDoc"] = field(default=None, repr=False, compare=False)

    # ---- resolution
    def _doc(self, doc) -> Optional["SceneDoc"]:
        return doc if doc is not None else self.doc

    def resolved_dims(self, doc=None) -> dict:
        """The keyword dims `props.build` gets: the asset's path / up folded in for meshes."""
        dims = {k: v for k, v in (self.dims or {}).items()}
        if self.style != "mesh":
            return dims
        d = self._doc(doc)
        if self.asset is not None:
            if d is None:
                raise ValueError(f"prop {self.id}: a mesh placement needs its scene to resolve asset {self.asset!r}")
            info = d.get_asset(self.asset)
            if info is None:
                raise ValueError(f"prop {self.id}: unknown asset {self.asset!r}")
            dims["path"] = d.asset_path(self.asset)
            dims["scale"] = float(info.scale) * float(dims.get("scale", 1.0))
            dims["up"] = info.up
        else:                                    # a bare mesh (from a track's StaticProp)
            dims.setdefault("scale", 1.0)
            dims.setdefault("up", "z")
        return dims

    def _key(self, doc=None) -> tuple:
        dims = self.resolved_dims(doc)
        stamp = None
        if self.style == "mesh":
            try:
                stamp = os.path.getmtime(dims["path"])
            except (OSError, KeyError):
                stamp = None
        return (self.style, int(self.seed), tuple(sorted(dims.items())), stamp)

    def build(self, doc=None) -> "_props.Prop":
        """The prop this placement stands for (cached by style / seed / dims / asset file)."""
        key = self._key(doc)
        prop = _PROP_CACHE.get(key)
        if prop is None:
            prop = _props.build(self.style, seed=int(self.seed), **self.resolved_dims(doc))
            _PROP_CACHE[key] = prop
        return prop

    def footprint_world(self, doc=None) -> np.ndarray:
        """(K,2) envelope footprint rotated by `yaw` and translated to (x, y)."""
        return _to_world(self.build(doc).envelope.footprint, self.x, self.y, self.yaw)

    def to_static_prop(self, doc=None):
        from .track import StaticProp
        dims = self.resolved_dims(doc)
        return StaticProp(self.style, float(self.x), float(self.y), float(self.yaw), int(self.seed),
                          tuple(sorted(dims.items())))

    def to_json(self) -> dict:
        d = {"id": self.id, "style": self.style, "x": float(self.x), "y": float(self.y),
             "yaw": float(self.yaw), "seed": int(self.seed),
             "dims": {k: (float(v) if not isinstance(v, str) else v) for k, v in (self.dims or {}).items()}}
        if self.asset is not None:
            d["asset"] = self.asset
        return d

    @staticmethod
    def from_json(d: dict) -> "PropPlacement":
        return PropPlacement(id=str(d["id"]), style=str(d["style"]), x=float(d["x"]), y=float(d["y"]),
                             yaw=float(d.get("yaw", 0.0)), seed=int(d.get("seed", 0)),
                             dims=dict(d.get("dims") or {}), asset=d.get("asset"))


def _to_world(poly: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    p = np.asarray(poly, np.float64)
    return np.stack([c * p[:, 0] - s * p[:, 1] + x, s * p[:, 0] + c * p[:, 1] + y], 1)


def _convex_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons (either winding)."""
    for poly in (a, b):
        e = np.roll(poly, -1, 0) - poly
        n = np.stack([e[:, 1], -e[:, 0]], 1)
        pa, pb = a @ n.T, b @ n.T
        if np.any((pa.max(0) < pb.min(0) - 1e-9) | (pb.max(0) < pa.min(0) - 1e-9)):
            return False
    return True


# ==================================================================== the document
@dataclass(eq=False)
class SceneDoc:
    name: str
    resolution: float
    origin: Tuple[float, float]
    duct: np.ndarray
    tall: np.ndarray
    duct_height: float = 0.33
    wall_height: float = 1.0
    centerline: Optional[np.ndarray] = None
    props: List[PropPlacement] = field(default_factory=list)
    assets: List[AssetInfo] = field(default_factory=list)
    notes: str = ""
    source: dict = field(default_factory=dict)
    dir: Optional[str] = None
    modified: str = ""

    def __post_init__(self):
        self.duct = np.ascontiguousarray(np.asarray(self.duct, bool))
        self.tall = np.ascontiguousarray(np.asarray(self.tall, bool))
        if self.duct.shape != self.tall.shape or self.duct.ndim != 2:
            raise ValueError(f"duct {self.duct.shape} and tall {self.tall.shape} must be the same (H, W)")
        self.origin = (float(self.origin[0]), float(self.origin[1]))
        self.resolution = float(self.resolution)
        if self.centerline is not None:
            self.centerline = np.asarray(self.centerline, np.float64).reshape(-1, 2)
            if len(self.centerline) < 3:
                self.centerline = None
        if not self.source:
            self.source = {"map": None, "created": _now()}
        for p in self.props:
            p.doc = self

    # ---------------------------------------------------------------- construction / persistence
    @staticmethod
    def new_blank(name: str, width_m: float, height_m: float, resolution: float = 0.05,
                  duct_height: float = 0.33) -> "SceneDoc":
        """Empty free space `width_m x height_m`, origin (0, 0)."""
        W = max(1, int(math.ceil(float(width_m) / float(resolution) - 1e-9)))
        H = max(1, int(math.ceil(float(height_m) / float(resolution) - 1e-9)))
        z = np.zeros((H, W), bool)
        return SceneDoc(_check_name(name), float(resolution), (0.0, 0.0), z, z.copy(),
                        duct_height=float(duct_height))

    @staticmethod
    def load(name_or_path: str) -> "SceneDoc":
        d = scene_dir(name_or_path)
        jp = os.path.join(d, "scene.json")
        if not os.path.isfile(jp):
            raise FileNotFoundError(f"scene not found: {jp}")
        with open(jp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        schema = int(meta.get("schema", 1))
        if schema > SCENE_SCHEMA_VERSION:
            raise ValueError(f"{jp}: schema {schema} is newer than this code ({SCENE_SCHEMA_VERSION})")
        with np.load(os.path.join(d, "layers.npz")) as z:
            duct, tall = z["duct"].astype(bool), z["tall"].astype(bool)
        shape = tuple(int(v) for v in (meta.get("shape") or duct.shape))
        if duct.shape != shape:
            raise ValueError(f"{d}: layers.npz is {duct.shape}, scene.json says {shape}")
        cl = meta.get("centerline")
        doc = SceneDoc(
            name=str(meta.get("name") or os.path.basename(d)),
            resolution=float(meta["resolution"]), origin=tuple(meta["origin"]),
            duct=duct, tall=tall,
            duct_height=float(meta.get("duct_height", 0.33)), wall_height=float(meta.get("wall_height", 1.0)),
            centerline=np.asarray(cl, np.float64) if cl else None,
            props=[PropPlacement.from_json(p) for p in (meta.get("props") or [])],
            assets=[AssetInfo.from_json(a) for a in (meta.get("assets") or [])],
            notes=str(meta.get("notes") or ""), source=dict(meta.get("source") or {}),
            dir=d, modified=str(meta.get("modified") or ""))
        return doc

    def to_json(self) -> dict:
        H, W = self.shape
        return {
            "schema": SCENE_SCHEMA_VERSION, "name": self.name, "notes": self.notes,
            "resolution": float(self.resolution), "origin": [float(self.origin[0]), float(self.origin[1])],
            "shape": [int(H), int(W)],
            "duct_height": float(self.duct_height), "wall_height": float(self.wall_height),
            "centerline": (None if self.centerline is None
                           else [[float(x), float(y)] for x, y in np.asarray(self.centerline, np.float64)]),
            "props": [p.to_json() for p in self.props],
            "assets": [a.to_json() for a in self.assets],
            "source": dict(self.source), "modified": self.modified,
        }

    def save(self, name_or_path: Optional[str] = None) -> str:
        """Write `scene.json` + `layers.npz` (and carry the assets along on a save-as). Returns the dir."""
        if name_or_path is not None:
            d = scene_dir(name_or_path)
            if not _is_path(str(name_or_path)):
                self.name = _check_name(name_or_path)
        else:
            d = self.dir or scene_dir(self.name)
        os.makedirs(os.path.join(d, "assets"), exist_ok=True)
        if self.dir and os.path.abspath(self.dir) != os.path.abspath(d):
            for a in self.assets:
                src = os.path.join(self.dir, a.file)
                dst = os.path.join(d, a.file)
                if os.path.isfile(src) and not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
        self.modified = _now()
        np.savez_compressed(os.path.join(d, "layers.npz"), duct=self.duct, tall=self.tall)
        tmp = os.path.join(d, "scene.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(d, "scene.json"))
        self.dir = d
        return d

    def copy(self) -> "SceneDoc":
        """Deep copy (undo snapshots). Placements are re-bound to the copy."""
        doc = SceneDoc(
            name=self.name, resolution=self.resolution, origin=self.origin,
            duct=self.duct.copy(), tall=self.tall.copy(),
            duct_height=self.duct_height, wall_height=self.wall_height,
            centerline=None if self.centerline is None else self.centerline.copy(),
            props=[PropPlacement(p.id, p.style, p.x, p.y, p.yaw, p.seed, dict(p.dims), p.asset) for p in self.props],
            assets=[copy.deepcopy(a) for a in self.assets],
            notes=self.notes, source=copy.deepcopy(self.source), dir=self.dir, modified=self.modified)
        return doc

    @staticmethod
    def from_track(track, name: str, source_map: Optional[str] = None) -> "SceneDoc":
        """A scene holding a track's duct / tall layers, duct height, centerline and props."""
        props = []
        for i, sp in enumerate(tuple(getattr(track, "props", ()) or ())):
            props.append(PropPlacement(f"p{i + 1}", str(sp.style), float(sp.x), float(sp.y), float(sp.yaw),
                                       int(sp.seed), dict(sp.dims), None))
        return SceneDoc(
            name=_check_name(name), resolution=float(track.resolution),
            origin=(float(track.origin[0]), float(track.origin[1])),
            duct=np.asarray(track.duct, bool).copy(), tall=np.asarray(track.tall, bool).copy(),
            duct_height=float(track.duct_height),
            centerline=None if track.centerline is None else np.asarray(track.centerline, np.float64).copy(),
            props=props, source={"map": source_map, "created": _now()})

    def to_track(self):
        """`f1sim.track.Track` of this scene (imports torch through `track.py`)."""
        from .track import Track
        t = Track.from_occupancy(self.occupancy, self.resolution, self.origin,
                                 None if self.centerline is None else self.centerline.copy(),
                                 f"scene:{self.name}", duct=self.duct.copy(), tall=self.tall.copy(),
                                 duct_height=float(self.duct_height))
        t.props = tuple(p.to_static_prop(self) for p in self.props)
        return t

    # ---------------------------------------------------------------- derived
    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(int(v) for v in self.duct.shape)

    @property
    def occupancy(self) -> np.ndarray:
        return self.duct | self.tall

    def bounds(self) -> Tuple[float, float, float, float]:
        H, W = self.shape
        return (self.origin[0], self.origin[1],
                self.origin[0] + W * self.resolution, self.origin[1] + H * self.resolution)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor((float(y) - self.origin[1]) / self.resolution)),
                int(math.floor((float(x) - self.origin[0]) / self.resolution)))

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        return (self.origin[0] + (int(col) + 0.5) * self.resolution,
                self.origin[1] + (int(row) + 0.5) * self.resolution)

    def _cell_box(self, x0: float, y0: float, x1: float, y1: float):
        """Clipped (r0, r1, c0, c1) of the cells whose centres can lie in the world box."""
        H, W = self.shape
        res = self.resolution
        c0 = max(0, int(math.floor((x0 - self.origin[0]) / res)))
        c1 = min(W, int(math.ceil((x1 - self.origin[0]) / res)) + 1)
        r0 = max(0, int(math.floor((y0 - self.origin[1]) / res)))
        r1 = min(H, int(math.ceil((y1 - self.origin[1]) / res)) + 1)
        return r0, r1, c0, c1

    def _centres(self, r0, r1, c0, c1):
        xs = self.origin[0] + (np.arange(c0, c1) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(r0, r1) + 0.5) * self.resolution
        return xs[None, :], ys[:, None]

    # ---------------------------------------------------------------- grid editing
    def _apply(self, layer: str, sl, m: np.ndarray, value: bool) -> bool:
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
        if not m.any():
            return False
        if layer == "free":
            d, t = self.duct[sl], self.tall[sl]
            changed = bool((d & m).any() or (t & m).any())
            d[m] = False
            t[m] = False
            return changed
        arr = self.duct if layer == "duct" else self.tall
        sub = arr[sl]
        changed = bool((sub[m] != bool(value)).any())
        sub[m] = bool(value)
        return changed

    def paint_disc(self, layer: str, x: float, y: float, radius: float, value: bool = True) -> bool:
        radius = max(float(radius), 0.0)
        r0, r1, c0, c1 = self._cell_box(x - radius, y - radius, x + radius, y + radius)
        if r1 <= r0 or c1 <= c0:
            return False
        xs, ys = self._centres(r0, r1, c0, c1)
        m = (xs - x) ** 2 + (ys - y) ** 2 <= radius ** 2 + 1e-12
        if not m.any():                      # thinner than a cell: still mark the cell under the point
            rr, cc = self.world_to_cell(x, y)
            if 0 <= rr < self.shape[0] and 0 <= cc < self.shape[1]:
                m[rr - r0, cc - c0] = True
        return self._apply(layer, (slice(r0, r1), slice(c0, c1)), m, value)

    def paint_polyline(self, layer: str, pts: Sequence, width: float, value: bool = True,
                       closed: bool = False) -> bool:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return False
        hw = max(float(width), 0.0) / 2.0
        if len(pts) == 1:
            return self.paint_disc(layer, pts[0, 0], pts[0, 1], hw, value)
        segs = list(zip(pts[:-1], pts[1:]))
        if closed and len(pts) > 2:
            segs.append((pts[-1], pts[0]))
        changed = False
        for a, b in segs:
            lo, hi = np.minimum(a, b) - hw, np.maximum(a, b) + hw
            r0, r1, c0, c1 = self._cell_box(lo[0], lo[1], hi[0], hi[1])
            if r1 <= r0 or c1 <= c0:
                continue
            xs, ys = self._centres(r0, r1, c0, c1)
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-18:
                t = np.zeros(np.broadcast(xs, ys).shape)
            else:
                t = np.clip(((xs - a[0]) * ab[0] + (ys - a[1]) * ab[1]) / L2, 0.0, 1.0)
            dx = xs - (a[0] + t * ab[0])
            dy = ys - (a[1] + t * ab[1])
            m = dx * dx + dy * dy <= hw * hw + 1e-12
            if not m.any():                  # a stroke thinner than a cell still leaves its trace
                n = max(2, 2 * int(math.ceil(math.sqrt(L2) / self.resolution)) + 1)   # 2 per cell
                for s in np.linspace(0.0, 1.0, n):
                    rr, cc = self.world_to_cell(a[0] + s * ab[0], a[1] + s * ab[1])
                    if r0 <= rr < r1 and c0 <= cc < c1:
                        m[rr - r0, cc - c0] = True
            changed |= self._apply(layer, (slice(r0, r1), slice(c0, c1)), m, value)
        return changed

    def paint_rect(self, layer: str, x0: float, y0: float, x1: float, y1: float, value: bool = True) -> bool:
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        r0, r1, c0, c1 = self._cell_box(x0, y0, x1, y1)
        if r1 <= r0 or c1 <= c0:
            return False
        xs, ys = self._centres(r0, r1, c0, c1)
        m = (xs >= x0 - 1e-9) & (xs <= x1 + 1e-9) & (ys >= y0 - 1e-9) & (ys <= y1 + 1e-9)
        return self._apply(layer, (slice(r0, r1), slice(c0, c1)), m, value)

    def _polygon_mask(self, pts: np.ndarray, perimeter: bool = False) -> np.ndarray:
        """Cells whose centre lies inside the world polygon (plus the cells its edges cross when
        `perimeter`, so a shape thinner than a cell still leaves its trace)."""
        from skimage.draw import polygon, polygon_perimeter
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        H, W = self.shape
        m = np.zeros((H, W), bool)
        if len(pts) < 3:
            return m
        r = (pts[:, 1] - self.origin[1]) / self.resolution - 0.5
        c = (pts[:, 0] - self.origin[0]) / self.resolution - 0.5
        rr, cc = polygon(r, c, shape=(H, W))
        m[rr, cc] = True
        if perimeter:
            rr, cc = polygon_perimeter(r, c, shape=(H, W), clip=True)
            m[rr, cc] = True
        return m

    def fill_polygon(self, layer: str, pts: Sequence, value: bool = True) -> bool:
        m = self._polygon_mask(np.asarray(pts, np.float64).reshape(-1, 2))
        return self._apply(layer, (slice(None), slice(None)), m, value)

    def paint_border(self, layer: str, thickness: float) -> bool:
        H, W = self.shape
        n = max(1, int(math.ceil(float(thickness) / self.resolution - 1e-9)))
        m = np.ones((H, W), bool)
        if H > 2 * n and W > 2 * n:
            m[n:H - n, n:W - n] = False
        return self._apply(layer, (slice(None), slice(None)), m, True)

    def resize_canvas(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """New canvas covering the world box, snapped outward to the existing grid; content keeps
        its world position, new cells are free. Props and the centerline are untouched."""
        res = self.resolution
        ox, oy = self.origin
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        c_lo = int(math.floor((x0 - ox) / res + 1e-9))
        r_lo = int(math.floor((y0 - oy) / res + 1e-9))
        c_hi = int(math.ceil((x1 - ox) / res - 1e-9))
        r_hi = int(math.ceil((y1 - oy) / res - 1e-9))
        W, H = c_hi - c_lo, r_hi - r_lo
        if W < 1 or H < 1:
            raise ValueError("canvas must be at least one cell in each direction")
        H0, W0 = self.shape
        duct = np.zeros((H, W), bool)
        tall = np.zeros((H, W), bool)
        sr0, sr1 = max(0, r_lo), min(H0, r_hi)
        sc0, sc1 = max(0, c_lo), min(W0, c_hi)
        if sr1 > sr0 and sc1 > sc0:
            duct[sr0 - r_lo:sr1 - r_lo, sc0 - c_lo:sc1 - c_lo] = self.duct[sr0:sr1, sc0:sc1]
            tall[sr0 - r_lo:sr1 - r_lo, sc0 - c_lo:sc1 - c_lo] = self.tall[sr0:sr1, sc0:sc1]
        self.duct, self.tall = duct, tall
        self.origin = (ox + c_lo * res, oy + r_lo * res)

    def content_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """World bbox of everything: occupied cells, prop footprints, centerline. None when empty."""
        boxes = []
        occ = self.occupancy
        if occ.any():
            rows = np.flatnonzero(occ.any(1))
            cols = np.flatnonzero(occ.any(0))
            boxes.append((self.origin[0] + cols[0] * self.resolution, self.origin[1] + rows[0] * self.resolution,
                          self.origin[0] + (cols[-1] + 1) * self.resolution,
                          self.origin[1] + (rows[-1] + 1) * self.resolution))
        for p in self.props:
            try:
                fp = p.footprint_world(self)
            except Exception:
                fp = np.array([[p.x, p.y]])
            boxes.append((fp[:, 0].min(), fp[:, 1].min(), fp[:, 0].max(), fp[:, 1].max()))
        if self.centerline is not None:
            cl = self.centerline
            boxes.append((cl[:, 0].min(), cl[:, 1].min(), cl[:, 0].max(), cl[:, 1].max()))
        if not boxes:
            return None
        b = np.asarray(boxes, np.float64)
        return (float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max()))

    def fit_canvas(self, margin: float = 2.0) -> None:
        b = self.content_bounds()
        if b is None:
            return
        m = float(margin)
        self.resize_canvas(b[0] - m, b[1] - m, b[2] + m, b[3] + m)

    # ---------------------------------------------------------------- props
    def _fresh_id(self, prefix: str, taken) -> str:
        n = 1
        for t in taken:
            if t.startswith(prefix) and t[len(prefix):].isdigit():
                n = max(n, int(t[len(prefix):]) + 1)
        return f"{prefix}{n}"

    def add_prop(self, style: str, x: float, y: float, yaw: float = 0.0, seed: int = 0, dims=None,
                 asset: Optional[str] = None) -> PropPlacement:
        if style != "mesh" and style not in _props.STYLES:
            raise ValueError(f"unknown prop style {style!r}; known: {', '.join(_props.STYLES)}, mesh")
        if style == "mesh" and asset is not None and self.get_asset(asset) is None:
            raise ValueError(f"unknown asset {asset!r}")
        p = PropPlacement(self._fresh_id("p", [q.id for q in self.props]), style, float(x), float(y),
                          float(yaw), int(seed), dict(dims or {}), asset, doc=self)
        self.props.append(p)
        return p

    def remove_prop(self, pid: str) -> None:
        self.props = [p for p in self.props if p.id != pid]

    def get_prop(self, pid: str) -> Optional[PropPlacement]:
        for p in self.props:
            if p.id == pid:
                return p
        return None

    def prop_footprints(self) -> List[Tuple[str, np.ndarray]]:
        """[(id, (K,2) world polygon)] for picking. A prop that cannot be built contributes a
        small square at its position, so it can still be selected and deleted."""
        out = []
        for p in self.props:
            try:
                out.append((p.id, p.footprint_world(self)))
            except Exception:
                r = 0.15
                out.append((p.id, np.array([[p.x - r, p.y - r], [p.x + r, p.y - r],
                                            [p.x + r, p.y + r], [p.x - r, p.y + r]])))
        return out

    def bake_prop(self, pid: str, layer: str = "tall") -> bool:
        """Rasterise the prop's true footprint into `layer` and remove the prop.

        Built-ins: the union of their section band polygons (what the LiDAR / contact code
        actually bounds them with). Meshes: every triangle projected onto the ground. Cells whose
        centre is covered are set, and so are the cells each outline crosses, so a post thinner
        than a cell does not vanish in the bake."""
        p = self.get_prop(pid)
        if p is None:
            return False
        if layer not in ("duct", "tall"):
            raise ValueError("bake into 'duct' or 'tall'")
        prop = p.build(self)
        H, W = self.shape
        m = np.zeros((H, W), bool)
        if p.style == "mesh":
            from skimage.draw import polygon
            c, s = math.cos(p.yaw), math.sin(p.yaw)
            for part in prop.parts:
                tri = part.pos.astype(np.float64)[part.idx.reshape(-1, 3)][:, :, :2]      # (M,3,2)
                wx = c * tri[:, :, 0] - s * tri[:, :, 1] + p.x
                wy = s * tri[:, :, 0] + c * tri[:, :, 1] + p.y
                r = (wy - self.origin[1]) / self.resolution - 0.5
                cc = (wx - self.origin[0]) / self.resolution - 0.5
                for i in range(len(tri)):
                    rr, ci = polygon(r[i], cc[i], shape=(H, W))
                    m[rr, ci] = True
                    # the three corner cells, so a sliver triangle still marks something
                    rf = np.clip(np.rint(r[i]).astype(int), 0, H - 1)
                    cf = np.clip(np.rint(cc[i]).astype(int), 0, W - 1)
                    inside = (r[i] >= -0.5) & (r[i] <= H - 0.5) & (cc[i] >= -0.5) & (cc[i] <= W - 0.5)
                    m[rf[inside], cf[inside]] = True
        else:
            polys = [b["polygon"] for b in _props.sections(prop)] or [prop.envelope.footprint]
            for poly in polys:
                m |= self._polygon_mask(_to_world(poly, p.x, p.y, p.yaw), perimeter=True)
        changed = self._apply(layer, (slice(None), slice(None)), m, True)
        self.remove_prop(pid)
        return changed or True

    # ---------------------------------------------------------------- assets
    def get_asset(self, aid: str) -> Optional[AssetInfo]:
        for a in self.assets:
            if a.id == aid:
                return a
        return None

    def asset_path(self, aid: str) -> str:
        a = self.get_asset(aid)
        if a is None:
            raise ValueError(f"unknown asset {aid!r}")
        if not self.dir:
            raise ValueError("the scene has no directory yet: save it first")
        return os.path.abspath(os.path.join(self.dir, a.file))

    def import_asset(self, path: str, name: Optional[str] = None, up: str = "auto", scale: float = 1.0,
                     target_height: Optional[float] = None) -> AssetInfo:
        """Copy a mesh file into `<dir>/assets/` and register it.

        `up="auto"` is y-up for .glb/.gltf (the glTF convention) and z-up otherwise;
        `target_height` scales the asset so it stands that many metres tall. Loads the mesh once
        (through `props.mesh_asset`, so a broken file fails here rather than at the first
        placement). Needs `self.dir`: save first.
        """
        if not self.dir:
            raise ValueError("save the scene before importing assets (it needs a directory)")
        path = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"mesh file not found: {path}")
        stem, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lower()
        if ext not in MESH_EXTENSIONS:
            raise ValueError(f"unsupported mesh format {ext!r}; use one of {', '.join(MESH_EXTENSIONS)}")
        if up == "auto":
            up = "y" if ext in (".glb", ".gltf") else "z"
        if up not in ("y", "z"):
            raise ValueError(f"up must be 'y', 'z' or 'auto', got {up!r}")
        adir = os.path.join(self.dir, "assets")
        os.makedirs(adir, exist_ok=True)
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in stem) or "asset"
        dst_name, k = f"{safe}{ext}", 2
        while os.path.exists(os.path.join(adir, dst_name)):
            dst_name = f"{safe}_{k}{ext}"
            k += 1
        dst = os.path.join(adir, dst_name)
        shutil.copy2(path, dst)
        try:
            prop = _props.mesh_asset(dst, scale=1.0, up=up)
        except Exception:
            try:
                os.remove(dst)
            except OSError:
                pass
            raise
        x0, y0, _, x1, y1, h = prop.envelope.bounds()
        s = float(scale)
        if target_height is not None:
            s = float(target_height) / h
        info = AssetInfo(id=self._fresh_id("a", [a.id for a in self.assets]), file=f"assets/{dst_name}",
                         name=str(name or stem), up=up, scale=s, tris=int(prop.n_tris),
                         size=[float((x1 - x0) * s), float((y1 - y0) * s), float(h * s)], collision="bands")
        self.assets.append(info)
        return info

    def remove_asset(self, aid: str) -> None:
        a = self.get_asset(aid)
        if a is None:
            return
        users = [p.id for p in self.props if p.asset == aid]
        if users:
            raise ValueError(f"asset {aid!r} is still placed by {', '.join(users)}")
        if self.dir:
            try:
                os.remove(os.path.join(self.dir, a.file))
            except OSError:
                pass
        self.assets = [b for b in self.assets if b.id != aid]

    # ---------------------------------------------------------------- validation (torch-free)
    def quick_issues(self) -> List[dict]:
        """Cheap checks the page runs after every edit: `[{"level","msg","x","y","prop"}]`."""
        issues: List[dict] = []

        def add(level, msg, x=None, y=None, prop=None):
            issues.append({"level": level, "msg": msg,
                           "x": None if x is None else float(x), "y": None if y is None else float(y),
                           "prop": prop})

        occ = self.occupancy
        if occ.all():
            add("error", "빈 공간이 없습니다. 차가 달릴 바닥을 남겨 두세요.")
        if self.centerline is None:
            add("warn", "중심선이 없습니다. 검증하면 빈 공간에서 자동으로 뽑습니다.")
        for a in self.assets:
            p = os.path.join(self.dir, a.file) if self.dir else a.file
            if not os.path.isfile(p):
                add("error", f"메시 파일이 없습니다: {a.file} ({a.name})")
        x0, y0, x1, y1 = self.bounds()
        foots: List[Tuple[str, np.ndarray]] = []
        for p in self.props:
            if p.style == "mesh" and p.asset is not None and self.get_asset(p.asset) is None:
                add("error", f"장애물 {p.id}: 없는 메시({p.asset})를 가리킵니다.", p.x, p.y, p.id)
                continue
            try:
                fp = p.footprint_world(self)
            except FileNotFoundError as exc:
                add("error", f"장애물 {p.id}: 메시 파일이 없습니다 ({exc}).", p.x, p.y, p.id)
                continue
            except Exception as exc:
                add("error", f"장애물 {p.id} ({p.style}) 을 만들지 못했습니다: {exc}", p.x, p.y, p.id)
                continue
            if (fp[:, 0] < x0).any() or (fp[:, 0] > x1).any() or (fp[:, 1] < y0).any() or (fp[:, 1] > y1).any():
                add("error", f"장애물 {p.id} ({p.style}) 이 캔버스 밖으로 나갑니다.", p.x, p.y, p.id)
            m = self._polygon_mask(fp)
            if (m & occ).any():
                add("error", f"장애물 {p.id} ({p.style}) 이 덕트나 벽과 겹칩니다.", p.x, p.y, p.id)
            foots.append((p.id, fp))
        for i in range(len(foots)):
            for j in range(i + 1, len(foots)):
                if _convex_overlap(foots[i][1], foots[j][1]):
                    q = self.get_prop(foots[i][0])
                    add("warn", f"장애물 {foots[i][0]} 과 {foots[j][0]} 이 겹칩니다.", q.x, q.y, foots[i][0])
        return issues


# ==================================================================== lane detection
def lane_seed(occupancy: np.ndarray, resolution: float, origin, min_clearance: float = MIN_CORRIDOR_M / 2,
              min_hole_m2: float = 2.0) -> Optional[Tuple[float, float]]:
    """A world point inside the lane of a loop track, or None when there is no loop.

    `Track.centerline_from_free_space` extracts the equidistant curve of *one* free region and,
    without a seed, takes the largest. On a duct-hose ring drawn inside a hall that is the floor
    outside the ring, not the lane -- and the editor's scenes are exactly that. The lane is the
    innermost annulus: a free region (at `min_clearance` from anything solid) that has an infield
    hole and whose hole contains no other region with a hole. The outside floor fails the second
    part (its hole holds the lane), the infield fails the first (no hole). Ties go to the larger
    region. Holes under `min_hole_m2` are pillars and are filled first, as the extractor does.
    Torch-free (scipy / skimage), so the page can call it too.
    """
    from scipy import ndimage
    from skimage.morphology import remove_small_holes
    occ = np.asarray(occupancy, bool)
    edt = ndimage.distance_transform_edt(~occ) * float(resolution)
    free = edt >= float(min_clearance)
    lab, n = ndimage.label(free)
    if n == 0:
        return None
    area = int(min_hole_m2 / float(resolution) ** 2)
    cands = []
    for k in range(1, n + 1):
        comp = lab == k
        comp_f = remove_small_holes(comp, area_threshold=area)
        filled = ndimage.binary_fill_holes(comp_f)
        if (filled & ~comp_f).any():
            cands.append((k, int(comp.sum()), filled))
    if not cands:
        return None
    best = None
    for k, size, filled in cands:
        nested = any(j != k and bool(filled[lab == j].any()) for j, _, _ in cands)
        if nested:
            continue
        if best is None or size > best[1]:
            best = (k, size)
    if best is None:                                   # every annulus holds another: take the smallest
        best = min(cands, key=lambda c: c[1])[:2]
    comp = lab == best[0]
    d = np.where(comp, edt, 0.0)
    r, c = np.unravel_index(int(np.argmax(d)), d.shape)
    return (float(float(origin[0]) + (int(c) + 0.5) * float(resolution)),
            float(float(origin[1]) + (int(r) + 0.5) * float(resolution)))


# ==================================================================== CLI
def _emit(obj: dict, code: int = 0) -> int:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return code


def _log(msg: str) -> None:
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _sample_edt(track, xy: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    rc = np.stack([(xy[:, 1] - track.origin[1]) / track.resolution - 0.5,      # cell centres
                   (xy[:, 0] - track.origin[0]) / track.resolution - 0.5])
    return ndimage.map_coordinates(track.edt, rc, order=1, mode="nearest")


def cli_import(map_name: str, target: str) -> int:
    from . import maps
    _log(f"loading {map_name} ...")
    track = maps.load(map_name)
    name = os.path.basename(scene_dir(target).rstrip(os.sep))
    doc = SceneDoc.from_track(track, name, source_map=map_name)
    d = doc.save(target)
    return _emit({"dir": d, "props": len(doc.props), "shape": list(doc.shape)})


def validate(target: str, centerline: str = "keep", raceline: bool = False, seed_xy=None) -> dict:
    """The full check (needs torch through `Track`). Writes an auto-extracted centerline back.

    `seed_xy` picks the free region the centerline is extracted from; without it the previous
    centerline's first point is used, and failing that `lane_seed`."""
    doc = SceneDoc.load(target)
    issues = list(doc.quick_issues())
    stats: Dict[str, object] = {"props": len(doc.props)}
    track = doc.to_track()

    if doc.centerline is None or centerline == "auto":
        _log("extracting the centerline from free space ...")
        try:
            clearance = max(0.3, MIN_CORRIDOR_M / 2)
            seed = tuple(seed_xy) if seed_xy is not None else None
            if seed is None and doc.centerline is not None:
                seed = (float(doc.centerline[0, 0]), float(doc.centerline[0, 1]))
            if seed is None:
                seed = lane_seed(doc.occupancy, doc.resolution, doc.origin, min_clearance=clearance)
            if seed is not None:
                stats["seed_xy"] = [float(seed[0]), float(seed[1])]
            cl = track.centerline_from_free_space(min_clearance=clearance, seed_xy=seed)
            doc.centerline = np.asarray(cl, np.float64)
            doc.save()
            issues = [i for i in issues if "중심선" not in i["msg"]]
        except Exception as exc:
            issues.append({"level": "error", "msg": f"중심선을 찾지 못했습니다: {exc}", "x": None, "y": None, "prop": None})

    cl = track.centerline
    if cl is not None and len(cl) >= 3:
        seg = np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1)
        stats["length_m"] = float(seg.sum())
        if seg[-1] > 4.0 * max(float(seg[:-1].mean()), 1e-6):
            issues.append({"level": "error", "msg": "중심선이 닫혀 있지 않습니다.", "x": float(cl[-1, 0]),
                           "y": float(cl[-1, 1]), "prop": None})
        d = _sample_edt(track, cl)
        i = int(np.argmin(d))
        stats["min_width_m"] = float(2.0 * d[i])
        if 2.0 * d[i] < MIN_CORRIDOR_M:
            issues.append({"level": "error",
                           "msg": f"중심선 위 가장 좁은 곳이 {2.0 * d[i]:.2f} m 입니다 (최소 {MIN_CORRIDOR_M:.1f} m).",
                           "x": float(cl[i, 0]), "y": float(cl[i, 1]), "prop": None})
        # a spawn: some centerline point clear of walls and of every prop
        clear = d >= SPAWN_CLEARANCE_M
        if clear.any() and doc.props:
            foots = [fp for _, fp in doc.prop_footprints()]
            for k in np.flatnonzero(clear):
                pt = cl[k]
                for fp in foots:
                    if (np.linalg.norm(fp - pt, axis=1) < SPAWN_CLEARANCE_M).any() or _point_in_poly(pt, fp):
                        clear[k] = False
                        break
        if not clear.any():
            issues.append({"level": "error", "msg": "출발 위치가 없습니다: 중심선 어디에도 차가 설 자리가 없습니다.",
                           "x": None, "y": None, "prop": None})
    for p in doc.props:
        try:
            p.build(doc)
        except Exception as exc:
            if not any(i.get("prop") == p.id and i["level"] == "error" for i in issues):
                issues.append({"level": "error", "msg": f"장애물 {p.id} ({p.style}) 을 만들지 못했습니다: {exc}",
                               "x": p.x, "y": p.y, "prop": p.id})
    if raceline and track.centerline is not None and not any(i["level"] == "error" for i in issues):
        from .raceline import Raceline
        _log("building the raceline ...")
        t0 = time.perf_counter()
        rl = Raceline.build_cached(track)
        stats["raceline_s"] = float(time.perf_counter() - t0)
        xy = np.asarray(rl.xy, np.float64)
        stats["raceline_length_m"] = float(np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1).sum())
        stats["raceline_min_clearance_m"] = float(_sample_edt(track, xy).min())
    ok = not any(i["level"] == "error" for i in issues)
    return {"ok": ok, "issues": issues, "stats": stats}


def _point_in_poly(pt, poly) -> bool:
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def cli_export_ros(target: str, out_dir: str) -> int:
    doc = SceneDoc.load(target)
    track = doc.to_track()
    yaml_path = track.save_ros_map(out_dir, name=doc.name)
    return _emit({"yaml": yaml_path, "dir": os.path.abspath(out_dir)})


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m f1sim.scene",
                                 description="Environment editor scenes: import a catalogue map, validate, export.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("import", help="catalogue map -> scene")
    a.add_argument("map_name"); a.add_argument("scene")
    v = sub.add_parser("validate", help="load, extract the centerline if needed, check")
    v.add_argument("scene"); v.add_argument("--centerline", choices=("auto", "keep"), default="keep")
    v.add_argument("--raceline", action="store_true")
    v.add_argument("--seed", nargs=2, type=float, metavar=("X", "Y"), default=None,
                   help="a point inside the lane, when the scene has several free regions")
    e = sub.add_parser("export-ros", help="write yaml + pgm (+ centerline csv)")
    e.add_argument("scene"); e.add_argument("out_dir")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "import":
            return cli_import(args.map_name, args.scene)
        if args.cmd == "validate":
            rep = validate(args.scene, centerline=args.centerline, raceline=args.raceline, seed_xy=args.seed)
            return _emit(rep, 0 if rep["ok"] else 1)
        if args.cmd == "export-ros":
            return cli_export_ros(args.scene, args.out_dir)
    except Exception as exc:                      # stdout stays one JSON object, whatever happened
        import traceback
        _log(traceback.format_exc())
        return _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "issues": [{"level": "error", "msg": str(exc), "x": None, "y": None, "prop": None}]}, 1)
    return 2


if __name__ == "__main__":                        # `python -m f1sim.scene`: delegate to the real module
    from f1sim.scene import main as _main
    sys.exit(_main())
