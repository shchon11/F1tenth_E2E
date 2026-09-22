"""Static map geometry as plain arrays, torch-free.

`build_track_geometry` is the body of `SimWorker.build_geometry`, moved here verbatim so the
environment editor -- which runs in the console process, where torch is never imported -- can turn
a `SceneDoc` into the same meshes the worker sends for a `Track`. Same keys, same
`GEOMETRY_VERSION`, same smoothing; `sim_worker` re-exports everything it used to define.

`track_like` needs `shape, resolution, origin, duct, tall, duct_height, centerline, name, props`
with each prop offering `.build()` (and `.x .y .yaw .style .seed .dims`): `Track` (props are
`StaticProp`) and `SceneDoc` (props are `PropPlacement`, bound to their document) both do.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np

#: Bumped when `build_track_geometry` changes what it emits, so a console can branch on what it
#: actually received rather than on which build it thinks it is talking to. 1 = canvas-shaped floor
#: quad and the padded-border contours; 2 = content-derived framing, apron + backdrop, smoothed
#: contours. A payload without the key is version 1 by definition, which keeps an older worker
#: readable.
GEOMETRY_VERSION = 2

#: Gaussian blur applied to the occupancy mask before marching squares, in grid cells. Marching
#: squares on a binary grid can only put a vertex at a cell midpoint, so a diagonal wall comes out as
#: treads and risers; blurring lets the 0.5 crossing fall between them. Chosen at 0.5 because that is
#: where the axis-aligned segment share stops improving much (Korea duct 13 % -> 4.5 %) while the
#: measured deviation from the unblurred contour is still a fraction of a cell -- the tuning table is
#: in work/claude-map-redesign/evidence/contour_tuning.json.
SMOOTH_SIGMA = 0.5

#: Wall extrusion height when the track-like object does not carry one (`Track` never does).
DEFAULT_WALL_HEIGHT = 1.0


class PropBuildError(ValueError):
    """A placed prop that cannot be built. `sim_worker` turns this into its `StartConfigError`."""


def _bbox(point_sets):
    """Axis-aligned bounds of the drawn content, or None when nothing was drawn."""
    pts = [np.asarray(p, np.float64).reshape(-1, 2) for p in point_sets if p is not None and len(p)]
    if not pts:
        return None
    a = np.vstack(pts)
    return (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max()))


def placement_key(sp) -> tuple:
    """What makes two placements share one build: style, seed, dims and (for meshes) the asset.

    `StaticProp.dims` is a tuple of pairs and `PropPlacement.dims` a dict; `tuple(dict)` would be
    the keys alone, so two placements of the same style with different sizes would share a mesh."""
    dims = getattr(sp, "dims", ()) or ()
    if isinstance(dims, dict):
        dims = tuple(sorted((str(k), v) for k, v in dims.items()))
    else:
        dims = tuple(dims)
    return (str(sp.style), int(sp.seed), dims, getattr(sp, "asset", None))


def prop_batches(track) -> list:
    """Every placed prop on `track`, transformed into world space and merged by material.

    Merged because the alternative is one draw call per box: a map can place a few dozen, and the
    whole catalogue uses four materials, so this is four uploads however many props there are.
    Built here rather than in the console's render thread because it is numpy work, and the thread
    that would otherwise do it is the one holding the GL context.

    `build()` is deterministic in `(style, seed, dims)`, so identical placements share one build --
    a row of the same crate costs one mesh and a per-instance transform.

    **A prop that will not build raises.** It must not be skipped: the placement stays in
    `track.props`, so the simulator still collides with it and the LiDAR still returns it, and
    dropping only its mesh produces an obstacle that is there in every way except on screen. That
    invisible collider is the exact failure `StaticProp` exists to prevent -- see its docstring.
    Refusing to prepare the session is the only honest outcome. A track that places nothing is a
    different thing entirely and returns an empty list.
    """
    placed = tuple(getattr(track, "props", ()) or ())
    if not placed:
        return []
    parts: "OrderedDict[str, list]" = OrderedDict()
    counts: Dict[str, int] = {}
    cache: Dict[tuple, Any] = {}
    for sp in placed:
        key = placement_key(sp)
        prop = cache.get(key)
        if prop is None:
            try:
                prop = sp.build()
            except Exception as exc:
                raise PropBuildError(
                    f"맵의 정적 장애물 '{getattr(sp, 'style', '?')}' 을 만들지 못했습니다: {exc}\n"
                    f"이 장애물은 시뮬레이터와 LiDAR 에는 그대로 존재하므로, 그리지 못한 채로 "
                    f"주행하면 화면에 없는 충돌체가 됩니다. 세션을 시작하지 않습니다."
                ) from exc
            cache[key] = prop
        c, s_ = math.cos(float(sp.yaw)), math.sin(float(sp.yaw))
        rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]], np.float32)
        off = np.array([float(sp.x), float(sp.y), 0.0], np.float32)
        for part in prop.parts:
            bucket = parts.setdefault(part.material, [])
            bucket.append((part.pos @ rot.T + off, part.nrm @ rot.T, part.col, part.idx))
            counts[part.material] = counts.get(part.material, 0) + 1
    out = []
    for material, chunks in parts.items():
        pos_l, nrm_l, col_l, idx_l, base = [], [], [], [], 0
        for pos, nrm, col, idx in chunks:
            pos_l.append(pos)
            nrm_l.append(nrm)
            col_l.append(col)
            idx_l.append(idx.astype(np.int32) + base)
            base += len(pos)
        idx = np.concatenate(idx_l)
        out.append({
            "material": material,
            "pos": np.concatenate(pos_l).astype(np.float32),
            "nrm": np.concatenate(nrm_l).astype(np.float32),
            "col": np.concatenate(col_l).astype(np.float32),
            "idx": idx,
            "n_props": counts.get(material, 0),
            "n_tris": int(len(idx) // 3),
        })
    return out


def shape_meshes(shapes) -> list:
    """One mesh per procedural catalogue shape (`procedural_obstacles.Shape`), in the shape's own
    frame: a list, per shape, of its parts as {"material", "pos", "nrm", "col", "idx"}.

    The procedural layout is not placed geometry -- it is redrawn at every reset, and a shoved piece
    moves -- so unlike `prop_batches` nothing here is transformed or merged. The viewer draws each
    shape instanced, at the poses every frame carries. Built from the same `props.build` call the
    catalogue took its envelope from, so what is drawn is what the LiDAR and the contact test see.

    A shape that will not build raises, for `prop_batches`' reason: it would be an obstacle that is
    there in every way except on screen.
    """
    from .. import props as _props
    out = []
    for sh in shapes:
        try:
            prop = _props.build(sh.style, seed=0, **dict(sh.dims))
        except Exception as exc:
            raise PropBuildError(
                f"학습 장애물 모양 '{sh.style}' 을 만들지 못했습니다: {exc}\n"
                f"그리지 못한 채로 주행하면 화면에 없는 충돌체가 됩니다. 세션을 시작하지 않습니다.") from exc
        out.append([{"material": part.material,
                     "pos": np.asarray(part.pos, np.float32), "nrm": np.asarray(part.nrm, np.float32),
                     "col": np.asarray(part.col, np.float32), "idx": np.asarray(part.idx, np.int32),
                     "n_props": 1, "n_tris": int(len(part.idx) // 3)} for part in prop.parts])
    return out


def build_track_geometry(track, raceline=None, only_props: bool = False):
    """Static map meshes as plain arrays, ready for the console to upload.

    Contours, smoothing, the duct offset and the tube/wall vertex generation all happen here.
    The render thread's share is a buffer write. `only_props=True` returns just the prop batches
    (the list `prop_batches` gives), which is the fast path while a prop is being dragged.

    `raceline` is anything with `.xy` and `.v` (a `Raceline`), or None.
    """
    if only_props:
        return prop_batches(track)
    from .gl_scene import (apron_mesh_arrays, backdrop_mesh_arrays, content_frame,
                           duct_mesh_arrays, wall_mesh_arrays)
    from .contours import track_contours_mask
    t0 = time.perf_counter()
    H, W = track.shape
    res = float(track.resolution)
    bounds = (float(track.origin[0]), float(track.origin[1]),
              float(track.origin[0] + W * res), float(track.origin[1] + H * res))
    wall_height = float(getattr(track, "wall_height", None) or DEFAULT_WALL_HEIGHT)

    def occupied(xy, mask=track.duct, tr=track):
        # floor, not truncation: `astype(int)` rounds toward zero, so a probe just outside the
        # map on the low side lands on cell 0 and reads whatever is there. Out of bounds is
        # unknown, not solid, and reporting it as solid biases the duct's solid-side vote.
        j = np.floor((xy[:, 0] - tr.origin[0]) / tr.resolution).astype(np.int64)
        i = np.floor((xy[:, 1] - tr.origin[1]) / tr.resolution).astype(np.int64)
        ok = (i >= 0) & (i < mask.shape[0]) & (j >= 0) & (j < mask.shape[1])
        out = np.zeros(len(xy), bool)
        out[ok] = mask[i[ok], j[ok]]
        return out

    def is_solid(xy, tr=track):
        col = int(round((xy[0] - tr.origin[0]) / tr.resolution))
        row = int(round((xy[1] - tr.origin[1]) / tr.resolution))
        return 0 <= row < H and 0 <= col < W and bool(tr.tall[row, col])

    # `outside_occupied=False`: the map file's canvas perimeter is not a wall. See track_contours.
    # `clip_canvas_edge` on the tall layer only: that is where a cropped SLAM surround lives, and
    # its outward silhouette is the crude rectangle around the track. Ducts are the track's own
    # walls -- clipping those would delete real boundary on a map that runs to its edge.
    duct_c = track_contours_mask(track, track.duct, outside_occupied=False, sigma_cells=SMOOTH_SIGMA)
    tall_c = track_contours_mask(track, track.tall, outside_occupied=False, sigma_cells=SMOOTH_SIGMA,
                                 clip_canvas_edge=True)
    ducts = duct_mesh_arrays(duct_c, track.duct_height, occupied=occupied, resolution=res)
    walls = wall_mesh_arrays(tall_c, height=wall_height, is_solid=is_solid, resolution=res)

    centerline = np.asarray(track.centerline, np.float32) if track.centerline is not None else None
    raceline_xy = np.asarray(raceline.xy, np.float32) if raceline is not None else None
    # Frame on what is actually drawn, never on the canvas or on the occupancy grid: with
    # `unknown_is_obstacle` the occupied cells can span the whole file.
    # heading from the track itself, size from everything drawn -- see content_frame
    orient = [centerline] if centerline is not None else duct_c
    angle, centre, extent = content_frame(orient, duct_c + tall_c + [centerline, raceline_xy])
    content_bounds = _bbox(duct_c + tall_c + [centerline, raceline_xy])
    return {
        "name": track.name, "bounds": bounds, "duct_height": float(track.duct_height),
        "floor": apron_mesh_arrays(angle, centre, extent), "ducts": ducts, "walls": walls,
        "backdrop": backdrop_mesh_arrays(centre, extent),
        "content_bounds": content_bounds,
        "presentation": {"up_angle_deg": float(math.degrees(angle)),
                         "center": (float(centre[0]), float(centre[1])),
                         "extent": (float(extent[0]), float(extent[1]))},
        "geometry_version": GEOMETRY_VERSION,
        # Optional: absent, or an empty list, on every track that places nothing.
        "props": prop_batches(track),
        "centerline": centerline,
        "raceline_xy": raceline_xy,
        "raceline_v": (np.asarray(raceline.v, np.float32) if raceline is not None else None),
        "build_ms": (time.perf_counter() - t0) * 1e3,
    }
