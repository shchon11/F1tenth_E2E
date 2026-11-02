"""One obstacle layout: what the console's 장애물 편집 edits, saves and drives on (2026-11-03).

The user, 11-03: "콘솔에서 왜 장애물 시드 못바꾸게해? 그리고 시드만 바꾸는게 아니라 내가 직접 장애물을
추가/제거/위치/변경해서 바로 돌리게 할 수도 있었으면 좋겠어."

A layout is a list of pieces, each one shape of the training generator's catalogue
(`procedural_obstacles.ROW_SHAPES` / `SMALL_SHAPES`, the crates a policy trained among) standing at a pose.
It is written in two forms:

* **in a session** -- ``{"sid", "x", "y", "yaw"}``: `sid` indexes the session's catalogue, which the worker
  sends with the session facts (``prop_catalogue``, built by `viewer.prop_layout.catalogue`);
* **portable** -- ``{"style", "dims", "x", "y", "yaw"}``: what a file and `SessionConfig.prop_layout` hold. The
  catalogue index is an accident of build order; the style and its dimensions are what the piece *is*, so a
  saved layout still loads after the catalogue grows.

Neither torch nor Qt: the window and the worker both import this.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

#: Where 저장 / 불러오기 open by default. User data, next to `~/f1sim_scenes`.
LAYOUT_DIR = os.path.join(os.path.expanduser("~"), "f1sim_layouts")
FORMAT = "f1sim-obstacle-layout"
VERSION = 1

#: What each catalogue style is called on screen (`procedural_obstacles.STYLE_MASS` lists the same six).
STYLE_LABEL = {
    "cardboard_box": "골판지 상자",
    "wooden_crate": "나무 상자",
    "steel_drum": "철제 드럼",
    "barrier_block": "방호 블록",
    "crate_stack_low": "낮은 상자 더미",
    "marker_post": "표지 기둥",
}

#: Two dims match when they differ by less than this [m] -- a file written by hand with 0.3 for 0.30.
DIM_TOL = 1e-4


def shape_label(style: str, dims: Dict[str, float]) -> str:
    """'나무 상자 30×34 cm', '철제 드럼 ⌀29 cm': the name and the footprint a person would measure."""
    name = STYLE_LABEL.get(style, style)
    d = {k: float(v) for k, v in dict(dims or {}).items()}
    if "radius" in d:
        return f"{name} ⌀{2.0 * d['radius'] * 100:.0f} cm"
    if "width" in d and "depth" in d:
        return f"{name} {d['width'] * 100:.0f}×{d['depth'] * 100:.0f} cm"
    return name


def _dims_equal(a: Dict[str, float], b: Dict[str, float]) -> bool:
    a, b = dict(a or {}), dict(b or {})
    if set(a) != set(b):
        return False
    return all(abs(float(a[k]) - float(b[k])) < DIM_TOL for k in a)


def match_shape(style: str, dims: Dict[str, float], catalogue: Sequence[Dict[str, Any]]) -> int:
    """The catalogue index of (style, dims). ValueError, in Korean, when the catalogue has no such piece."""
    for e in catalogue:
        if e.get("style") == style and _dims_equal(e.get("dims") or {}, dims):
            return int(e["id"])
    raise ValueError(f"이 시뮬레이터의 장애물 목록에 없는 모양입니다: {shape_label(style, dims)} ({style} {dict(dims or {})})")


def _number(v, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"장애물 {what} 값이 숫자가 아닙니다: {v!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"장애물 {what} 값이 유한하지 않습니다: {v!r}")
    return x


def clean_portable(pieces) -> List[Dict[str, Any]]:
    """Portable pieces checked and normalised (floats, yaw wrapped to (-pi, pi]). ValueError on a bad one."""
    if not isinstance(pieces, (list, tuple)):
        raise ValueError(f"장애물 목록이 리스트가 아닙니다: {type(pieces).__name__}")
    out = []
    for i, p in enumerate(pieces):
        if not isinstance(p, dict):
            raise ValueError(f"장애물 {i + 1}: 객체가 아닙니다 ({type(p).__name__})")
        style = str(p.get("style") or "")
        if not style:
            raise ValueError(f"장애물 {i + 1}: 모양(style)이 없습니다")
        dims = p.get("dims") or {}
        if not isinstance(dims, dict):
            raise ValueError(f"장애물 {i + 1}: 치수(dims)가 객체가 아닙니다")
        yaw = _number(p.get("yaw", 0.0), "각도")
        out.append({"style": style, "dims": {str(k): _number(v, "치수") for k, v in dims.items()},
                    "x": _number(p.get("x"), "x"), "y": _number(p.get("y"), "y"),
                    "yaw": math.atan2(math.sin(yaw), math.cos(yaw))})
    return out


def to_portable(pieces: Sequence[Dict[str, Any]], catalogue: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Session pieces (`sid`) -> portable ones (`style`, `dims`)."""
    by_id = {int(e["id"]): e for e in catalogue}
    out = []
    for p in pieces:
        e = by_id.get(int(p["sid"]))
        if e is None:
            raise ValueError(f"장애물 모양 번호 {p['sid']} 가 이 세션의 목록에 없습니다")
        out.append({"style": e["style"], "dims": dict(e["dims"]),
                    "x": float(p["x"]), "y": float(p["y"]), "yaw": float(p["yaw"])})
    return out


def from_portable(pieces, catalogue: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Portable pieces -> session ones (`sid` in this catalogue). ValueError names the first piece it cannot place."""
    out = []
    for i, p in enumerate(clean_portable(pieces)):
        try:
            sid = match_shape(p["style"], p["dims"], catalogue)
        except ValueError as exc:
            raise ValueError(f"장애물 {i + 1}: {exc}") from None
        out.append({"sid": sid, "x": p["x"], "y": p["y"], "yaw": p["yaw"]})
    return out


def save(path: str, pieces: Sequence[Dict[str, Any]], *, map_id: str = "", seed: Optional[int] = None,
         note: str = "") -> str:
    """Write a portable layout as JSON (atomically). Returns the path written."""
    data = {"format": FORMAT, "version": VERSION, "map": str(map_id or ""),
            "seed_origin": (None if seed is None else int(seed)), "note": str(note or ""),
            "pieces": clean_portable(pieces)}
    path = os.path.expanduser(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def load(path: str) -> Dict[str, Any]:
    """{"pieces": portable, "map": str, "seed_origin": int|None, "note": str}. ValueError, in Korean, otherwise."""
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(f"배치 파일을 열 수 없습니다: {exc}") from None
    except ValueError as exc:
        raise ValueError(f"배치 파일이 JSON 이 아닙니다: {exc}") from None
    if isinstance(data, list):                      # a bare list of pieces is accepted too
        data = {"pieces": data}
    if not isinstance(data, dict) or "pieces" not in data:
        raise ValueError("배치 파일에 장애물 목록(pieces)이 없습니다")
    if data.get("format", FORMAT) != FORMAT:
        raise ValueError(f"장애물 배치 파일이 아닙니다 (format {data.get('format')!r})")
    if int(data.get("version", VERSION)) > VERSION:
        raise ValueError(f"더 새로운 콘솔이 쓴 배치 파일입니다 (version {data.get('version')})")
    seed = data.get("seed_origin")
    return {"pieces": clean_portable(data["pieces"]), "map": str(data.get("map") or ""),
            "seed_origin": (None if seed is None else int(seed)), "note": str(data.get("note") or "")}


def default_name(map_id: str, seed: Optional[int] = None) -> str:
    """`korea26_s4417.json` / `korea26_편집.json` -- the map, and where the layout came from."""
    base = (map_id or "layout").replace("real/", "").replace("/", "_").replace(":", "_") or "layout"
    return f"{base}_s{int(seed)}.json" if seed is not None else f"{base}_편집.json"
