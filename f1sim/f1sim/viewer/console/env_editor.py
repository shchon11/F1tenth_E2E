"""The environment editor page of the console: build and edit driving environments in 3D.

What this page is for. Every map the simulator can drive on used to come from a catalogue entry,
a generator seed, or a session with someone editing PGM files by hand. This page lets the user
make one directly: paint duct hoses and tall walls, draw wall polylines, place the built-in props,
import mesh assets (GLB / glTF / OBJ / STL) and put them on the track, then save the scene and
drive on it from the 주행 page. Nothing here imports torch -- the document (`f1sim.scene`), the
geometry build (`f1sim.viewer.geometry`) and the props (`f1sim.props`) are numpy; what needs a
`Track` (centerline extraction, validation, catalogue import) runs as a subprocess of
`python -m f1sim.scene`.

Three parts:

* `EditorState` -- the document plus its undo stack. Every edit goes through `commit()`, which
  snapshots the document before the change, so undo is a copy-back rather than an inverse
  operation per tool. Brush strokes are one commit per stroke, not per mouse move.
* `Tool` subclasses -- what a mouse press / move / release means for the active tool. They only
  mutate the document through the state and only draw through the viewport's overlay setters.
* `EnvEditorPage` -- the widget: scene list and file actions on the left, the viewport in the
  middle with a tool bar over it, the inspector (selected prop, layers, scene settings, asset
  library, validation) on the right.

The document is rebuilt into geometry after edits on a worker thread (`_GeometryBuilder`): grid
edits trigger a full rebuild (contours + tubes + walls, debounced), prop edits only rebuild the
prop batches, which is what keeps dragging a crate interactive on a large map.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from . import theme
from .theme import C, SP
from .widgets import Card, Collapsible, FieldRow, FilterList, KeyValueList, SegmentedButtons, hline, label

MAX_UNDO = 60
#: Debounce for full geometry rebuilds after grid edits (ms). A brush stroke fires one rebuild at
#: its end; the debounce merges several quick strokes.
REBUILD_DEBOUNCE_MS = 120

#: Tool keys, their labels, tooltip and shortcut. Order is the toolbar order.
TOOLS = [
    ("select", "선택·이동", "클릭으로 선택, 드래그로 이동. 경로는 꼭짓점을 잡아 고칩니다  (V)", "V"),
    ("brush_duct", "덕트 브러시", "덕트 호스 칠하기. LiDAR 빔이 위로 넘어가는 낮은 벽  (B)", "B"),
    ("brush_tall", "벽 브러시", "높은 벽 칠하기. 빔을 항상 막는 벽  (W)", "W"),
    ("erase", "지우개", "칠한 덕트와 벽을 지우기 (경로로 만든 것은 선택해서 삭제)  (E)", "E"),
    ("path_track", "트랙 경로", "차선 중심을 그리면 양쪽에 덕트 호스가 서고 센터라인이 됩니다. 직선·곡선 혼용  (T)", "T"),
    ("path_duct", "덕트 경로", "덕트 호스 한 줄을 경로로. 클릭 직선, Ctrl+클릭 곡선  (L)", "L"),
    ("path_tall", "벽 경로", "높은 벽 한 줄을 경로로. 클릭 직선, Ctrl+클릭 곡선  (K)", "K"),
    ("rect", "사각형", "드래그로 사각 벽 채우기. Shift: 덕트  (M)", "M"),
    ("polygon", "다각형 채우기", "클릭으로 꼭짓점, 더블클릭·Enter 로 채우기  (P)", "P"),
    ("place", "배치", "라이브러리에서 고른 에셋을 클릭한 자리에 놓기. Ctrl+휠: 방향  (A)", "A"),
]
TOOL_KEYS = [t[0] for t in TOOLS]

BUILTIN_PROPS = [
    ("cardboard_box", "종이 상자", "0.36 × 0.30 × 0.30 m"),
    ("wooden_crate", "나무 궤짝", "0.42 × 0.34 × 0.36 m"),
    ("steel_drum", "드럼통", "r 0.145 × 0.44 m"),
    ("crate_stack_low", "낮은 궤짝 더미", "0.62 × 0.44 × 0.26 m"),
    ("barrier_block", "방호 블록", "0.54 × 0.26 × 0.22 m"),
    ("marker_post", "표지 기둥", "r 0.055 × 0.62 m"),
]

MESH_FILTER = "메쉬 (*.glb *.gltf *.obj *.stl *.ply *.off);;모든 파일 (*)"


def _rgba(hex_colour: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    h = hex_colour.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, float(alpha))


def _scene_module():
    from f1sim import scene as S
    return S


# ================================================================ state + undo
class EditorState(QtCore.QObject):
    """The document, its undo stack, and the selection. Emits when any of them changes."""
    changed = QtCore.pyqtSignal(str)            # "grid" | "props" | "doc" | "selection"
    dirty_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc = None                          # f1sim.scene.SceneDoc | None
        self._undo: List[tuple] = []             # (label, snapshot)
        self._redo: List[tuple] = []
        self.selection: List[str] = []
        #: (path id, vertex index or None) -- exclusive with the prop selection
        self.path_sel: Tuple[Optional[str], Optional[int]] = (None, None)
        self._dirty = False

    # -- document
    def set_doc(self, doc, dirty: bool = False):
        self.doc = doc
        self._undo.clear()
        self._redo.clear()
        self.selection = []
        self.path_sel = (None, None)
        self._set_dirty(dirty)
        self.changed.emit("doc")

    @property
    def dirty(self) -> bool:
        return self._dirty

    def _set_dirty(self, on: bool):
        if on != self._dirty:
            self._dirty = on
            self.dirty_changed.emit(on)

    def mark_saved(self):
        self._set_dirty(False)

    # -- edits
    def commit(self, label_text: str, fn: Callable[[], object], kind: str = "grid"):
        """Snapshot, apply `fn`, and record. Returns fn's result. A `fn` that returns False
        (nothing changed) is not recorded, so a brush stroke over empty space is not an undo step."""
        if self.doc is None:
            return None
        snap = self.doc.copy()
        result = fn()
        if result is False:
            return result
        self._undo.append((label_text, snap))
        if len(self._undo) > MAX_UNDO:
            del self._undo[0]
        self._redo.clear()
        self._set_dirty(True)
        self.changed.emit(kind)
        return result

    def begin_stroke(self, label_text: str):
        """For tools that mutate over many mouse moves: one snapshot at the start, one undo step."""
        if self.doc is None:
            return
        self._stroke = (label_text, self.doc.copy())

    def end_stroke(self, changed: bool, kind: str = "grid"):
        st = getattr(self, "_stroke", None)
        self._stroke = None
        if st is None or not changed:
            return
        self._undo.append(st)
        if len(self._undo) > MAX_UNDO:
            del self._undo[0]
        self._redo.clear()
        self._set_dirty(True)
        self.changed.emit(kind)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_label(self) -> str:
        return self._undo[-1][0] if self._undo else ""

    def undo(self):
        if not self._undo or self.doc is None:
            return
        label_text, snap = self._undo.pop()
        self._redo.append((label_text, self.doc.copy()))
        self.doc = snap
        self._prune_selection()
        self._set_dirty(True)
        self.changed.emit("doc")

    def redo(self):
        if not self._redo or self.doc is None:
            return
        label_text, snap = self._redo.pop()
        self._undo.append((label_text, self.doc.copy()))
        self.doc = snap
        self._prune_selection()
        self._set_dirty(True)
        self.changed.emit("doc")

    # -- selection
    def _prune_selection(self):
        ids = {p.id for p in self.doc.props} if self.doc is not None else set()
        self.selection = [s for s in self.selection if s in ids]
        pid, vi = self.path_sel
        q = self.doc.get_path(pid) if (self.doc is not None and pid is not None) else None
        if q is None:
            self.path_sel = (None, None)
        elif vi is not None and vi >= len(q.points):
            self.path_sel = (pid, None)

    def select(self, ids: Sequence[str]):
        ids = list(ids)
        if ids != self.selection or (ids and self.path_sel[0] is not None):
            self.selection = ids
            if ids:
                self.path_sel = (None, None)
            self.changed.emit("selection")

    def select_path(self, pid: Optional[str], vi: Optional[int]):
        new = (pid, vi)
        if new != self.path_sel or (pid is not None and self.selection):
            self.path_sel = new
            if pid is not None:
                self.selection = []
            self.changed.emit("selection")

    def selected_path(self):
        pid, _vi = self.path_sel
        return self.doc.get_path(pid) if (self.doc is not None and pid is not None) else None

    def selected_props(self) -> list:
        if self.doc is None:
            return []
        by_id = {p.id: p for p in self.doc.props}
        return [by_id[i] for i in self.selection if i in by_id]


# ================================================================ tools
#: Ground-decal colours per layer: what the brush is about to leave, what a path band will be.
LAYER_COLOUR = {"duct": C["warn"], "tall": "#d9d9d9", "free": C["danger"], "track": C["ego"]}
#: Contextual key hints per tool, shown under the toolbar (the Blender status-bar convention: the
#: keys that matter *now*, not a manual).
TOOL_HINTS = {
    "select": "클릭 선택 · 드래그 이동 · Shift+클릭 추가 · 빈 곳 드래그 범위 선택 · 경로 꼭짓점 드래그 · "
              "Alt+클릭 꼭짓점 추가 · Tab/더블클릭 직선↔곡선 · Delete 삭제 · 방향키 미세 이동 · R 회전",
    "brush": "드래그로 칠하기 (칠하는 자리가 색으로 미리 보이고 벽은 따라서 자랍니다) · Ctrl+휠 굵기 (벽·지우개; "
             "덕트는 항상 호스 지름 굵기) · 경로로 만든 벽은 지우개가 아니라 선택해서 고치거나 지웁니다",
    "path": "클릭 직선 꼭짓점 · Ctrl+클릭 곡선 꼭짓점 · Tab 마지막 꼭짓점 직선↔곡선 · Shift 45° 맞춤 · "
            "Backspace 한 점 취소 · C 닫기/열기 · Enter·더블클릭·우클릭 완료 · Esc 취소",
    "rect": "드래그로 사각 벽 · Shift 덕트 · Esc 취소",
    "polygon": "클릭 꼭짓점 · Enter·더블클릭 채우기 · Backspace 한 점 취소 · Esc 취소",
    "place": "클릭 배치 · Ctrl+휠 / R 방향 · 라이브러리에서 에셋 선택",
}


def _snap45(x0, y0, x, y):
    """(x, y) moved onto the nearest 45° ray from (x0, y0), keeping its distance."""
    dx, dy = x - x0, y - y0
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return x, y
    a = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
    return x0 + d * math.cos(a), y0 + d * math.sin(a)


class Tool:
    """Base: a tool reads the page for its parameters and mutates only via `page.state`."""
    name = ""                      # the TOOLS key; not `key`, which is the key-event handler below
    hint = ""

    def __init__(self, page: "EnvEditorPage"):
        self.page = page

    @property
    def state(self) -> EditorState:
        return self.page.state

    @property
    def vp(self):
        return self.page.viewport

    def activate(self):
        pass

    def deactivate(self):
        self.vp.set_preview(None, 0.0)
        self.vp.set_fill("tool", [], LAYER_COLOUR["duct"])
        self.vp.set_marquee(None)

    def press(self, x, y, button, mods, hit):
        pass

    def move(self, x, y, buttons, mods):
        pass

    def release(self, x, y, button, mods):
        pass

    def hover(self, x, y, hit):
        self.vp.set_cursor("cross", x, y)

    def double_click(self, x, y):
        pass

    def key(self, key, mods) -> bool:
        return False

    def wheel(self, delta, mods) -> bool:
        return False


class SelectTool(Tool):
    """Props and paths: pick, drag, marquee, vertex handles."""
    name = "select"
    hint = TOOL_HINTS["select"]

    def __init__(self, page):
        super().__init__(page)
        self._mode = None                      # "props" | "vertex" | "path" | "marquee" | None
        self._start = None
        self._orig = None
        self._moved = False
        self._hover_vertex = None              # (pid, idx)

    def _tol(self):
        return self.vp.handle_radius()

    def hover(self, x, y, hit):
        self.vp.set_cursor("none", x, y)
        doc = self.state.doc
        if hit:
            self.vp.set_hover(self.page.footprint_of(hit))
            self._hover_vertex = None
            self.page.set_hover_vertex(None)
            return
        pid, vi, _si = doc.path_hit(x, y, self._tol(), self._tol() * 1.6)
        if pid is not None:
            q = doc.get_path(pid)
            self._hover_vertex = (pid, vi) if vi is not None else None
            self.page.set_hover_vertex(self._hover_vertex)
            if vi is None:
                self.vp.set_hover(q.polyline() if not q.closed else np.vstack([q.polyline(), q.polyline()[:1]]))
            else:
                self.vp.set_hover(None)
        else:
            self._hover_vertex = None
            self.page.set_hover_vertex(None)
            self.vp.set_hover(None)

    def press(self, x, y, button, mods, hit):
        if button != QtCore.Qt.LeftButton:
            return
        doc = self.state.doc
        self._moved = False
        self._start = (x, y)
        if hit:
            if mods & QtCore.Qt.ShiftModifier:
                sel = list(self.state.selection)
                (sel.remove if hit in sel else sel.append)(hit)
                self.state.select(sel)
            elif hit not in self.state.selection:
                self.state.select([hit])
            self._mode = "props"
            self._orig = {p.id: (p.x, p.y) for p in self.state.selected_props()}
            self.state.begin_stroke("이동")
            return
        pid, vi, si = doc.path_hit(x, y, self._tol(), self._tol() * 1.6)
        if pid is not None:
            q = doc.get_path(pid)
            if vi is None and mods & QtCore.Qt.AltModifier:
                # insert a vertex on this segment and pick it up straight away
                self.state.begin_stroke("꼭짓점 추가")
                vi = doc.insert_path_vertex(pid, si, x, y, smooth=q.points[si][2] or q.points[(si + 1) % len(q.points)][2])
                self._moved = True
                self.state.select_path(pid, vi)
                self._mode = "vertex"
                self._orig = (x, y)
                self.page.begin_live_grid()
                return
            self.state.select_path(pid, vi)
            if vi is not None:
                self._mode = "vertex"
                self._orig = tuple(q.points[vi][:2])
                self.state.begin_stroke("꼭짓점 이동")
            else:
                self._mode = "path"
                self._orig = [tuple(p) for p in q.points]
                self.state.begin_stroke("경로 이동")
            self.page.begin_live_grid()
            return
        self._mode = "marquee"
        if not (mods & QtCore.Qt.ShiftModifier):
            self.state.select([])

    def move(self, x, y, buttons, mods):
        if self._mode is None or self.state.doc is None or self._start is None:
            return
        dx, dy = x - self._start[0], y - self._start[1]
        if not self._moved and math.hypot(dx, dy) < 0.01:
            return
        self._moved = True
        doc = self.state.doc
        snap = self.page.snap_step() if not (mods & QtCore.Qt.AltModifier) else 0.0
        if self._mode == "props":
            for p in doc.props:
                if p.id in self._orig:
                    ox, oy = self._orig[p.id]
                    nx, ny = ox + dx, oy + dy
                    if snap > 0:
                        nx, ny = round(nx / snap) * snap, round(ny / snap) * snap
                    p.x, p.y = float(nx), float(ny)
            self.page.props_moved_live()
        elif self._mode == "vertex":
            pid, vi = self.state.path_sel
            q = doc.get_path(pid)
            if q is None or vi is None:
                return
            nx, ny = x, y
            if mods & QtCore.Qt.ShiftModifier and len(q.points) > 1:
                ref = q.points[vi - 1] if vi > 0 else q.points[(vi + 1) % len(q.points)]
                nx, ny = _snap45(ref[0], ref[1], nx, ny)
            nx, ny = self.page.snap_point(nx, ny) if snap > 0 else (nx, ny)
            q.points[vi] = (float(nx), float(ny), q.points[vi][2])
            self.page.path_edited_live(q)
        elif self._mode == "path":
            pid, _vi = self.state.path_sel
            q = doc.get_path(pid)
            if q is None:
                return
            sx, sy = (round(dx / snap) * snap, round(dy / snap) * snap) if snap > 0 else (dx, dy)
            q.points = [(ox + sx, oy + sy, sm) for ox, oy, sm in self._orig]
            self.page.path_edited_live(q)
        elif self._mode == "marquee":
            self.vp.set_marquee((self._start[0], self._start[1], x, y))

    def release(self, x, y, button, mods):
        if button != QtCore.Qt.LeftButton or self._mode is None:
            return
        mode, self._mode = self._mode, None
        doc = self.state.doc
        if mode == "props":
            self.state.end_stroke(self._moved, kind="props")
            if not self._moved:
                self.state._stroke = None
                self.page.refresh_overlays()
        elif mode in ("vertex", "path"):
            self.page.end_live_grid()
            if self._moved:
                doc.path_changed()
            self.state.end_stroke(self._moved, kind="grid")
            if not self._moved:
                self.state._stroke = None
                self.page.refresh_overlays()
        elif mode == "marquee":
            self.vp.set_marquee(None)
            if self._moved and self._start is not None:
                x0, x1 = sorted((self._start[0], x))
                y0, y1 = sorted((self._start[1], y))
                inside = [p.id for p in doc.props if x0 <= p.x <= x1 and y0 <= p.y <= y1]
                sel = list(self.state.selection) if mods & QtCore.Qt.ShiftModifier else []
                self.state.select(sel + [i for i in inside if i not in sel])
            elif not self._moved:
                self.state.select([])
                self.state.select_path(None, None)
        self._start = None

    def double_click(self, x, y):
        doc = self.state.doc
        if doc is None:
            return
        pid, vi, _si = doc.path_hit(x, y, self._tol(), self._tol() * 1.6)
        if pid is not None and vi is not None:
            self.state.select_path(pid, vi)
            self.page.toggle_vertex_smooth()

    def key(self, key, mods) -> bool:
        if key in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.page.delete_selection()
            return True
        if key == QtCore.Qt.Key_Tab:
            self.page.toggle_vertex_smooth()
            return True
        if key == QtCore.Qt.Key_C and self.state.path_sel[0] is not None:
            self.page.toggle_path_closed()
            return True
        if key == QtCore.Qt.Key_R and self.state.selection:
            self.page.rotate_selection(math.radians(-15 if mods & QtCore.Qt.ShiftModifier else 15))
            return True
        if key == QtCore.Qt.Key_D and mods & QtCore.Qt.ControlModifier:
            self.page.duplicate_selection()
            return True
        arrows = {QtCore.Qt.Key_Left: (-1, 0), QtCore.Qt.Key_Right: (1, 0),
                  QtCore.Qt.Key_Up: (0, 1), QtCore.Qt.Key_Down: (0, -1)}
        if key in arrows:
            step = 0.01 if mods & QtCore.Qt.ShiftModifier else max(0.01, self.page.snap_step() or 0.05)
            self.page.nudge_selection(arrows[key][0] * step, arrows[key][1] * step)
            return True
        return False

    def wheel(self, delta, mods) -> bool:
        if mods & QtCore.Qt.ControlModifier and self.state.selection:
            self.page.rotate_selection(math.radians(5.0 if delta > 0 else -5.0))
            return True
        return False


class BrushTool(Tool):
    """Paint discs into one layer along the mouse path; one undo step per stroke.

    Feedback while the button is down is immediate and two-fold: the stroke so far is drawn as a
    decal in the layer's colour under the cursor, and the real geometry (hose tubes, walls) is
    rebuilt on a timer so it grows along behind the brush."""
    hint = TOOL_HINTS["brush"]

    def __init__(self, page, key, layer, value):
        super().__init__(page)
        self.name, self.layer, self.value = key, layer, value
        self._down = False
        self._changed = False
        self._last = None
        self._trail: List[Tuple[float, float]] = []

    @property
    def colour(self) -> str:
        return LAYER_COLOUR[self.layer]

    def _dab(self, x, y):
        doc = self.state.doc
        r = self.page.brush_radius(self.layer)
        if self._last is not None:
            lx, ly = self._last
            d = math.hypot(x - lx, y - ly)
            n = int(d / max(r * 0.5, doc.resolution)) + 1
            for i in range(1, n + 1):
                t = i / n
                self._changed |= bool(doc.paint_disc(self.layer, lx + (x - lx) * t, ly + (y - ly) * t, r, self.value))
        else:
            self._changed |= bool(doc.paint_disc(self.layer, x, y, r, self.value))
        self._last = (x, y)
        self._trail.append((x, y))
        self.vp.set_fill("tool", [("band", np.asarray(self._trail), 2 * r, False)], self.colour)

    def hover(self, x, y, hit):
        self.vp.set_cursor("brush", x, y, self.page.brush_radius(self.layer), colour=self.colour)

    def press(self, x, y, button, mods, hit):
        if button != QtCore.Qt.LeftButton or self.state.doc is None:
            return
        self._down, self._changed, self._last, self._trail = True, False, None, []
        self.state.begin_stroke({"duct": "덕트 칠하기", "tall": "벽 칠하기", "free": "지우기"}[self.layer])
        self._dab(x, y)
        self.page.begin_live_grid()

    def move(self, x, y, buttons, mods):
        self.vp.set_cursor("brush", x, y, self.page.brush_radius(self.layer), colour=self.colour)
        if self._down:
            self._dab(x, y)

    def release(self, x, y, button, mods):
        if button != QtCore.Qt.LeftButton or not self._down:
            return
        self._down = False
        self.vp.set_fill("tool", [], self.colour)
        self.page.end_live_grid()
        self.state.end_stroke(self._changed, kind="grid")

    def wheel(self, delta, mods) -> bool:
        if mods & QtCore.Qt.ControlModifier and self.layer != "duct":
            self.page.step_brush(1 if delta > 0 else -1)
            return True
        return False


class PathTool(Tool):
    """Draw a vector path: straight segments between corner vertices, curves through smooth ones.

    `kind` is `duct` / `tall` (a band of the brush width) or `track` (a lane: the centreline you
    draw, hoses along both edges, and the scene's centerline set from it)."""
    hint = TOOL_HINTS["path"]

    def __init__(self, page, key, kind):
        super().__init__(page)
        self.name, self.kind = key, kind
        self.pts: List[Tuple[float, float, bool]] = []
        self.closed = kind == "track"
        self._cursor: Optional[Tuple[float, float, bool]] = None

    def deactivate(self):
        self.pts = []
        self.page.set_path_handles([])
        super().deactivate()

    @property
    def colour(self) -> str:
        return LAYER_COLOUR[self.kind]

    def _width(self) -> float:
        if self.kind == "track":
            return self.page.lane_width()
        if self.kind == "duct":
            return float(self.state.doc.duct_height)
        return self.page.line_width()

    def _band(self) -> float:
        return float(self.state.doc.duct_height)

    def _shapes(self, pts, closed):
        from f1sim.scene import offset_polyline, sample_path
        pl = sample_path(pts, closed)
        if len(pl) == 0:
            return [], pl
        if self.kind == "track":
            half = 0.5 * self._width()
            return ([("band", offset_polyline(pl, +half, closed), self._band(), closed),
                     ("band", offset_polyline(pl, -half, closed), self._band(), closed)], pl)
        return [("band", pl, self._width(), closed)], pl

    def _preview(self):
        doc = self.state.doc
        if doc is None:
            return
        pts = list(self.pts)
        if self._cursor is not None and pts:
            pts.append(self._cursor)
        closed = self.closed and len(pts) >= 3
        shapes, pl = self._shapes(pts, closed) if pts else ([], np.zeros((0, 2)))
        self.vp.set_fill("tool", shapes, self.colour)
        if len(pl) >= 2:
            self.vp.set_preview(pl, 0.0, closed=closed, colour=self.colour)
        else:
            self.vp.set_preview(None, 0.0)
        handles = [(x, y, "smooth" if sm else "corner", "normal") for x, y, sm in self.pts]
        if self._cursor is not None:
            handles.append((self._cursor[0], self._cursor[1], "smooth" if self._cursor[2] else "corner", "hover"))
        self.page.set_path_handles(handles)

    def _next_point(self, x, y, mods) -> Tuple[float, float, bool]:
        if mods & QtCore.Qt.ShiftModifier and self.pts:
            x, y = _snap45(self.pts[-1][0], self.pts[-1][1], x, y)
        x, y = self.page.snap_point(x, y)
        return (float(x), float(y), bool(mods & QtCore.Qt.ControlModifier))

    def hover(self, x, y, hit):
        mods = QtWidgets.QApplication.keyboardModifiers()
        self._cursor = self._next_point(x, y, mods)
        self.vp.set_cursor("cross", self._cursor[0], self._cursor[1])
        self._preview()

    def press(self, x, y, button, mods, hit):
        if button == QtCore.Qt.RightButton:
            self.commit()
            return
        if button != QtCore.Qt.LeftButton:
            return
        p = self._next_point(x, y, mods)
        if self.pts and math.hypot(p[0] - self.pts[-1][0], p[1] - self.pts[-1][1]) < 1e-6:
            return
        self.pts.append(p)
        self._preview()

    def move(self, x, y, buttons, mods):
        self.hover(x, y, "")

    def double_click(self, x, y):
        self.commit()

    def commit(self):
        pts = list(self.pts)
        self.pts = []
        self._cursor = None
        self.vp.set_preview(None, 0.0)
        self.vp.set_fill("tool", [], self.colour)
        self.page.set_path_handles([])
        doc = self.state.doc
        if len(pts) < 2 or doc is None:
            return
        closed = self.closed and len(pts) >= 3
        kind, width = self.kind, self._width()
        band = self._band() if kind == "track" else None
        q = self.state.commit("경로" if kind != "track" else "트랙 경로",
                              lambda: doc.add_path(kind, pts, width, closed=closed, band=band), "grid")
        if q is not None:
            self.state.select_path(q.id, None)

    def key(self, key, mods) -> bool:
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.commit()
            return True
        if key == QtCore.Qt.Key_Escape:
            self.pts = []
            self._preview()
            return True
        if key == QtCore.Qt.Key_Backspace and self.pts:
            self.pts.pop()
            self._preview()
            return True
        if key == QtCore.Qt.Key_Tab and self.pts:
            x, y, sm = self.pts[-1]
            self.pts[-1] = (x, y, not sm)
            self._preview()
            return True
        if key == QtCore.Qt.Key_C:
            self.closed = not self.closed
            self.page.chk_close.setChecked(self.closed)
            self._preview()
            return True
        return False


class PolygonTool(Tool):
    """Click vertices; commit fills the polygon with tall wall (Shift on commit: duct)."""
    name = "polygon"
    hint = TOOL_HINTS["polygon"]

    def __init__(self, page):
        super().__init__(page)
        self.pts: List[Tuple[float, float]] = []
        self._cursor = None

    def deactivate(self):
        self.pts = []
        super().deactivate()

    def _preview(self):
        pts = self.pts + ([self._cursor] if self._cursor and self.pts else [])
        if len(pts) >= 3:
            self.vp.set_fill("tool", [("poly", np.asarray(pts))], LAYER_COLOUR["tall"])
            self.vp.set_preview(np.asarray(pts, np.float64), 0.0, closed=True, colour=LAYER_COLOUR["tall"])
        elif len(pts) >= 1:
            self.vp.set_fill("tool", [], LAYER_COLOUR["tall"])
            self.vp.set_preview(np.asarray(pts, np.float64), 0.0, colour=LAYER_COLOUR["tall"])
        else:
            self.vp.set_fill("tool", [], LAYER_COLOUR["tall"])
            self.vp.set_preview(None, 0.0)

    def hover(self, x, y, hit):
        self._cursor = self.page.snap_point(x, y)
        self.vp.set_cursor("cross", *self._cursor)
        self._preview()

    def press(self, x, y, button, mods, hit):
        if button == QtCore.Qt.RightButton:
            self.commit()
            return
        if button != QtCore.Qt.LeftButton:
            return
        p = self.page.snap_point(x, y)
        if self.pts and math.hypot(p[0] - self.pts[-1][0], p[1] - self.pts[-1][1]) < 1e-6:
            return
        self.pts.append(p)
        self._preview()

    def double_click(self, x, y):
        self.commit()

    def commit(self, layer: str = "tall"):
        pts = list(self.pts)
        self.pts = []
        self.vp.set_preview(None, 0.0)
        self.vp.set_fill("tool", [], LAYER_COLOUR["tall"])
        if len(pts) < 3 or self.state.doc is None:
            return
        arr = np.asarray(pts, np.float64)
        self.state.commit("다각형 채우기", lambda: self.state.doc.fill_polygon(layer, arr, True), "grid")

    def key(self, key, mods) -> bool:
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.commit("duct" if mods & QtCore.Qt.ShiftModifier else "tall")
            return True
        if key == QtCore.Qt.Key_Escape:
            self.pts = []
            self._preview()
            return True
        if key == QtCore.Qt.Key_Backspace and self.pts:
            self.pts.pop()
            self._preview()
            return True
        return False


class RectTool(Tool):
    name = "rect"
    hint = TOOL_HINTS["rect"]

    def __init__(self, page):
        super().__init__(page)
        self._start = None
        self._layer = "tall"

    def press(self, x, y, button, mods, hit):
        if button != QtCore.Qt.LeftButton:
            return
        self._start = self.page.snap_point(x, y)
        self._layer = "duct" if mods & QtCore.Qt.ShiftModifier else "tall"

    def move(self, x, y, buttons, mods):
        self.vp.set_cursor("cross", x, y)
        if self._start is None:
            return
        self._layer = "duct" if mods & QtCore.Qt.ShiftModifier else "tall"
        x1, y1 = self.page.snap_point(x, y)
        x0, y0 = self._start
        rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
        self.vp.set_fill("tool", [("poly", rect)], LAYER_COLOUR[self._layer])
        self.vp.set_preview(rect, 0.0, closed=True, colour=LAYER_COLOUR[self._layer])

    def release(self, x, y, button, mods):
        if button != QtCore.Qt.LeftButton or self._start is None:
            return
        x0, y0 = self._start
        x1, y1 = self.page.snap_point(x, y)
        self._start = None
        self.vp.set_preview(None, 0.0)
        self.vp.set_fill("tool", [], LAYER_COLOUR["tall"])
        if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6 or self.state.doc is None:
            return
        layer = self._layer
        self.state.commit("사각형", lambda: self.state.doc.paint_rect(layer, x0, y0, x1, y1, True), "grid")

    def key(self, key, mods) -> bool:
        if key == QtCore.Qt.Key_Escape and self._start is not None:
            self._start = None
            self.vp.set_preview(None, 0.0)
            self.vp.set_fill("tool", [], LAYER_COLOUR["tall"])
            return True
        return False


class PlaceTool(Tool):
    name = "place"
    hint = TOOL_HINTS["place"]

    def __init__(self, page):
        super().__init__(page)
        self.yaw = 0.0

    def hover(self, x, y, hit):
        px, py = self.page.snap_point(x, y)
        poly = self.page.library_footprint(px, py, self.yaw)
        self.vp.set_cursor("place", px, py, 0.25, self.yaw)
        self.vp.set_hover(poly)
        self.vp.set_fill("tool", [("poly", poly)] if poly is not None and len(poly) >= 3 else [], C["accent"])

    def press(self, x, y, button, mods, hit):
        if button != QtCore.Qt.LeftButton:
            return
        px, py = self.page.snap_point(x, y)
        pid = self.page.place_library_item(px, py, self.yaw)
        if pid:
            self.state.select([pid])

    def wheel(self, delta, mods) -> bool:
        if mods & QtCore.Qt.ControlModifier:
            self.yaw = (self.yaw + math.radians(15.0 if delta > 0 else -15.0)) % (2 * math.pi)
            return True
        return False

    def key(self, key, mods) -> bool:
        if key == QtCore.Qt.Key_R:
            self.yaw = (self.yaw + math.radians(-15 if mods & QtCore.Qt.ShiftModifier else 15)) % (2 * math.pi)
            return True
        return False


# ================================================================ background workers
class _GeometryBuilder(QtCore.QThread):
    """Builds `TrackGeometry` arrays off the GUI thread. One build at a time; a request that
    arrives while one is running is coalesced into the next."""
    built = QtCore.pyqtSignal(object, bool, float)      # geometry dict, only_props, ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = QtCore.QMutex()
        self._pending = None                              # (doc copy, only_props)
        self._wake = QtCore.QWaitCondition()
        self._quit = False

    def request(self, doc, only_props: bool):
        self._lock.lock()
        try:
            if self._pending is not None and not self._pending[1]:
                only_props = False                        # a full build already queued wins
            self._pending = (doc.copy(), only_props)
            self._wake.wakeAll()
        finally:
            self._lock.unlock()
        if not self.isRunning():
            self.start()

    def stop(self):
        self._lock.lock()
        self._quit = True
        self._wake.wakeAll()
        self._lock.unlock()
        self.wait(2000)

    def run(self):
        from f1sim.viewer.geometry import build_track_geometry
        while True:
            self._lock.lock()
            while self._pending is None and not self._quit:
                self._wake.wait(self._lock)
            if self._quit:
                self._lock.unlock()
                return
            doc, only_props = self._pending
            self._pending = None
            self._lock.unlock()
            t0 = time.perf_counter()
            try:
                geom = build_track_geometry(doc, only_props=only_props)
                if only_props:                     # the fast path returns the batches alone
                    geom = {"props": tuple(geom or ())}
            except Exception as exc:               # the page reports; the thread must live on
                geom = {"error": f"{type(exc).__name__}: {exc}"}
            self.built.emit(geom, only_props, (time.perf_counter() - t0) * 1e3)


class _SubprocessJob(QtCore.QObject):
    """`python -m f1sim.scene <args>` detached from the GUI thread; stdout JSON on finish."""
    finished = QtCore.pyqtSignal(int, object, str)     # returncode, parsed json or None, stderr tail

    def __init__(self, args: List[str], parent=None):
        super().__init__(parent)
        self.args = args
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.SeparateChannels)
        self.proc.finished.connect(self._done)
        self.proc.errorOccurred.connect(self._error)
        self._stderr = ""

    def start(self):
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(_repo_f1sim_dir())
        self.proc.start(sys.executable, ["-m", "f1sim.scene"] + list(self.args))

    def _error(self, _err):
        self.finished.emit(-1, None, f"프로세스를 시작하지 못했습니다: {self.proc.errorString()}")

    def _done(self, code, _status):
        out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        err = bytes(self.proc.readAllStandardError()).decode("utf-8", "replace")
        parsed = None
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    break
                except ValueError:
                    continue
        self.finished.emit(int(code), parsed, err[-2000:])


def _repo_f1sim_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ================================================================ the page
class EnvEditorPage(QtWidgets.QWidget):
    drive_requested = QtCore.pyqtSignal(str)        # track id, e.g. "scene/my_hall"
    scenes_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None, viewport_factory=None):
        super().__init__(parent)
        self.state = EditorState(self)
        self.state.changed.connect(self._on_state_changed)
        self.state.dirty_changed.connect(lambda _: self._update_title())
        self._tool: Optional[Tool] = None
        self._tools: Dict[str, Tool] = {}
        self._geom_pending_full = False
        self._last_geometry = None               # TrackGeometry of the last full build
        self._validation: Optional[dict] = None
        self._job: Optional[_SubprocessJob] = None
        self._active = False
        self._footprints: Dict[str, np.ndarray] = {}

        self._builder = _GeometryBuilder(self)
        self._builder.built.connect(self._on_built)
        self._rebuild_timer = QtCore.QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(REBUILD_DEBOUNCE_MS)
        self._rebuild_timer.timeout.connect(self._rebuild_full)

        self._build_ui(viewport_factory)
        self._make_tools()
        self.set_tool("select")
        self.refresh_scene_list()
        self._update_title()
        self._update_buttons()

    # ---------------------------------------------------------------- UI
    def _build_ui(self, viewport_factory):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(SP[1], SP[1], SP[1], SP[1])
        root.setSpacing(SP[1])

        # -- left: scenes + file actions
        left_area = QtWidgets.QScrollArea()
        left_area.setObjectName("PanelScroll")
        left_area.setWidgetResizable(True)
        left_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_area.setMinimumWidth(320)
        left_area.setMaximumWidth(400)
        left_inner = QtWidgets.QWidget()
        left_inner.setObjectName("Panel")
        lv = QtWidgets.QVBoxLayout(left_inner)
        lv.setContentsMargins(SP[1], SP[1], SP[1], SP[1])
        lv.setSpacing(SP[2])

        cur = Card("현재 환경")
        self.title_label = label("—", "body")
        self.title_label.setObjectName("Mono")
        self.title_label.setWordWrap(True)
        cur.add(self.title_label)
        self.summary_label = label("", "hint")
        cur.add(self.summary_label)
        row = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("저장 (Ctrl+S)")
        self.btn_save.clicked.connect(self.save)
        row.addWidget(self.btn_save)
        self.btn_save_as = QtWidgets.QPushButton("다른 이름으로…")
        self.btn_save_as.setObjectName("GhostButton")
        self.btn_save_as.clicked.connect(self.save_as)
        row.addWidget(self.btn_save_as)
        cur.add(row)
        row2 = QtWidgets.QHBoxLayout()
        self.btn_validate = QtWidgets.QPushButton("검증 (센터라인·주행 가능)")
        self.btn_validate.clicked.connect(self.validate)
        row2.addWidget(self.btn_validate)
        cur.add(row2)
        self.btn_drive = QtWidgets.QPushButton("이 환경으로 주행 →")
        self.btn_drive.setObjectName("PrimaryButton")
        self.btn_drive.clicked.connect(self._drive)
        cur.add(self.btn_drive)
        cur.add(label("저장한 뒤 주행 페이지의 맵 목록 '내 환경 (에디터)' 에 같은 이름으로 나타납니다. "
                      "센터라인이 없으면 검증이 자동으로 뽑아 넣습니다.", "hint"))
        lv.addWidget(cur)

        new = Card("새로 만들기")
        self.spin_w = QtWidgets.QDoubleSpinBox()
        self.spin_w.setRange(4.0, 200.0)
        self.spin_w.setValue(20.0)
        self.spin_w.setSuffix(" m")
        self.spin_h = QtWidgets.QDoubleSpinBox()
        self.spin_h.setRange(4.0, 200.0)
        self.spin_h.setValue(14.0)
        self.spin_h.setSuffix(" m")
        self.spin_res = QtWidgets.QDoubleSpinBox()
        self.spin_res.setRange(0.02, 0.2)
        self.spin_res.setSingleStep(0.01)
        self.spin_res.setDecimals(3)
        self.spin_res.setValue(0.05)
        self.spin_res.setSuffix(" m/셀")
        nr = QtWidgets.QHBoxLayout()
        nr.addWidget(FieldRow("가로", self.spin_w))
        nr.addWidget(FieldRow("세로", self.spin_h))
        nr.addWidget(FieldRow("해상도", self.spin_res))
        new.add(nr)
        self.chk_border = QtWidgets.QCheckBox("가장자리에 벽 두르기")
        self.chk_border.setChecked(True)
        new.add(self.chk_border)
        self.btn_new = QtWidgets.QPushButton("빈 환경 만들기")
        self.btn_new.clicked.connect(self.new_blank)
        new.add(self.btn_new)
        lv.addWidget(new)

        # -- random track from a recipe of features (f1sim.trackgen)
        lv.addWidget(self._build_generator_card())

        # -- start from a catalogue map: the same grouped picker the driving page uses. The list
        # arrives from the worker (map names need torch); until then the search box still accepts
        # a typed name such as `real:korea_2026_competition`.
        imp_card = Card("기존 맵에서 시작")
        self.map_picker = FilterList("맵 이름 검색 · 직접 입력도 됨", rows=6)
        self.map_picker.set_status("맵 목록을 읽는 중입니다… (worker 시작 후 채워집니다)", "hint")
        self.map_picker.committed.connect(lambda _k: self.import_from_catalog())
        imp_card.add(self.map_picker)
        self.btn_import = QtWidgets.QPushButton("이 맵 복제해서 편집")
        self.btn_import.clicked.connect(self.import_from_catalog)
        imp_card.add(self.btn_import)
        imp_card.add(label("고른 맵(gen:/real:/rt:/gym:)을 통째로 복제해 새 환경으로 엽니다. 벽·덕트·장애물이 그대로 옵니다. "
                           "목록에 없는 이름은 검색칸에 직접 쳐도 됩니다.", "hint"))
        lv.addWidget(imp_card)

        # -- start from a SLAM map: the pair `slam_toolbox` / `map_saver` writes. The heavy half
        # (grid clean-up, lane tracing, the raceline check) is `f1sim.slam_map`, run as a job like
        # every other Track operation on this page; what lands here is an ordinary scene, so the
        # brushes, the paths and 칠한 덕트 -> 경로 all work on it.
        slam_card = Card("SLAM 맵 불러오기")
        row = QtWidgets.QHBoxLayout()
        self.slam_path = QtWidgets.QLineEdit()
        self.slam_path.setPlaceholderText("map.yaml (또는 map.pgm / map.png, 폴더도 됩니다)")
        row.addWidget(self.slam_path, 1)
        self.btn_slam_browse = QtWidgets.QPushButton("찾기…")
        self.btn_slam_browse.clicked.connect(self._browse_slam)
        row.addWidget(self.btn_slam_browse)
        slam_card.add(row)
        opt = QtWidgets.QHBoxLayout()
        opt.setSpacing(SP[0])
        opt.addWidget(label("경계", "hint"))
        self.slam_boundary = QtWidgets.QComboBox()
        self.slam_boundary.addItem("덕트 호스 (F1TENTH 트랙)", "duct")
        self.slam_boundary.addItem("벽 (건물·집기)", "wall")
        opt.addWidget(self.slam_boundary, 1)
        slam_card.add(opt)
        opt2 = QtWidgets.QHBoxLayout()
        opt2.setSpacing(SP[0])
        opt2.addWidget(label("최소 폭", "hint"))
        self.slam_clearance = QtWidgets.QDoubleSpinBox()
        self.slam_clearance.setRange(0.15, 2.0); self.slam_clearance.setSingleStep(0.05)
        self.slam_clearance.setValue(0.35); self.slam_clearance.setSuffix(" m")
        opt2.addWidget(self.slam_clearance)
        self.slam_keep = QtWidgets.QCheckBox("바깥 잡음 잘라내기")
        self.slam_keep.setChecked(True)
        opt2.addWidget(self.slam_keep)
        opt2.addStretch(1)
        slam_card.add(opt2)
        # The one escape hatch the tracing has: which free region is the lane. Off by default (the
        # largest is taken, which is usually right); on, it is a point the user knows is on the lane.
        seed = QtWidgets.QHBoxLayout()
        seed.setSpacing(SP[0])
        self.slam_use_seed = QtWidgets.QCheckBox("주행선 위의 한 점")
        seed.addWidget(self.slam_use_seed)
        self.slam_seed_x = QtWidgets.QDoubleSpinBox(); self.slam_seed_y = QtWidgets.QDoubleSpinBox()
        for sb, lab_ in ((self.slam_seed_x, "x"), (self.slam_seed_y, "y")):
            sb.setRange(-1000.0, 1000.0); sb.setDecimals(2); sb.setSingleStep(0.5)
            sb.setPrefix(f"{lab_} "); sb.setSuffix(" m"); sb.setEnabled(False)
            seed.addWidget(sb)
        self.slam_use_seed.toggled.connect(self.slam_seed_x.setEnabled)
        self.slam_use_seed.toggled.connect(self.slam_seed_y.setEnabled)
        seed.addStretch(1)
        slam_card.add(seed)
        self.btn_slam = QtWidgets.QPushButton("이 맵 불러와서 편집")
        self.btn_slam.clicked.connect(self.import_from_slam)
        slam_card.add(self.btn_slam)
        slam_card.add(label("slam_toolbox / map_saver 가 만든 yaml + pgm(png) 을 그대로 넣으면 주행선(센터라인)까지 "
                            "자동으로 뽑아 새 환경으로 엽니다. 스캔이 새어나간 곳이나 끊긴 벽은 불러온 뒤 붓으로 고치면 "
                            "됩니다. 트랙이 여러 갈래로 잡히면 주행선 위의 한 점을 골라 다시 불러오세요.", "hint"))
        lv.addWidget(slam_card)

        lst = Card("내 환경")
        self.scene_list = QtWidgets.QListWidget()
        self.scene_list.setMinimumHeight(140)
        self.scene_list.itemDoubleClicked.connect(lambda _it: self.open_selected())
        lst.add(self.scene_list)
        lr = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("열기")
        self.btn_open.clicked.connect(self.open_selected)
        lr.addWidget(self.btn_open)
        self.btn_dup = QtWidgets.QPushButton("복제")
        self.btn_dup.setObjectName("GhostButton")
        self.btn_dup.clicked.connect(self.duplicate_selected)
        lr.addWidget(self.btn_dup)
        self.btn_delete_scene = QtWidgets.QPushButton("삭제")
        self.btn_delete_scene.setObjectName("DangerButton")
        self.btn_delete_scene.clicked.connect(self.delete_selected)
        lr.addWidget(self.btn_delete_scene)
        lst.add(lr)
        self.scene_root_label = label("", "hint")
        lst.add(self.scene_root_label)
        lv.addWidget(lst, 1)
        left_area.setWidget(left_inner)
        root.addWidget(left_area, 0)

        # -- centre: toolbar + viewport + status
        centre = QtWidgets.QVBoxLayout()
        centre.setSpacing(SP[0])
        tools_row = QtWidgets.QHBoxLayout()
        tools_row.setSpacing(SP[1])
        self.tool_buttons = SegmentedButtons([(k, t, tip) for k, t, tip, _sc in TOOLS])
        self.tool_buttons.selected.connect(self.set_tool)
        tools_row.addWidget(self.tool_buttons)
        tools_row.addStretch(1)
        centre.addLayout(tools_row)
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(SP[1])
        self.spin_brush = QtWidgets.QDoubleSpinBox()
        self.spin_brush.setRange(0.05, 3.0)
        self.spin_brush.setSingleStep(0.05)
        self.spin_brush.setValue(0.25)
        self.spin_brush.setSuffix(" m")
        self.spin_brush.setToolTip("벽 브러시·지우개 반지름, 벽 경로 두께의 절반 (Ctrl+휠). 덕트는 항상 호스 지름(환경 설정의 덕트 높이) 굵기입니다.")
        bar.addWidget(label("벽 브러시", "field"))
        bar.addWidget(self.spin_brush)
        self.chk_snap = QtWidgets.QCheckBox("격자 스냅")
        self.chk_snap.setChecked(True)
        self.chk_snap.setToolTip("점을 0.1 m 격자에 맞춥니다. Alt 를 누르고 드래그하면 무시")
        bar.addWidget(self.chk_snap)
        self.spin_lane = QtWidgets.QDoubleSpinBox()
        self.spin_lane.setRange(0.6, 6.0)
        self.spin_lane.setSingleStep(0.1)
        self.spin_lane.setValue(1.6)
        self.spin_lane.setSuffix(" m")
        self.spin_lane.setToolTip("트랙 경로의 차선 폭 (호스 중심 사이 거리)")
        bar.addWidget(label("차선 폭", "field"))
        bar.addWidget(self.spin_lane)
        self.chk_close = QtWidgets.QCheckBox("경로 닫기")
        self.chk_close.setToolTip("경로 도구가 마지막 점을 첫 점과 잇습니다 (C)")
        self.chk_close.toggled.connect(self._on_close_toggled)
        bar.addWidget(self.chk_close)
        bar.addStretch(1)
        self.btn_undo = QtWidgets.QPushButton("되돌리기")
        self.btn_undo.setObjectName("GhostButton")
        self.btn_undo.clicked.connect(self.undo)
        bar.addWidget(self.btn_undo)
        self.btn_redo = QtWidgets.QPushButton("다시 실행")
        self.btn_redo.setObjectName("GhostButton")
        self.btn_redo.clicked.connect(self.redo)
        bar.addWidget(self.btn_redo)
        self.btn_frame = QtWidgets.QPushButton("전체 보기 (F)")
        self.btn_frame.setObjectName("GhostButton")
        self.btn_frame.clicked.connect(lambda: self.viewport.frame_all())
        bar.addWidget(self.btn_frame)
        self.btn_top = QtWidgets.QPushButton("위에서 (5)")
        self.btn_top.setObjectName("GhostButton")
        self.btn_top.setCheckable(True)
        self.btn_top.toggled.connect(lambda on: self.viewport.set_top_view(on))
        bar.addWidget(self.btn_top)
        centre.addLayout(bar)
        self.tool_hint = label("", "hint")
        self.tool_hint.setWordWrap(True)
        centre.addWidget(self.tool_hint)

        if viewport_factory is None:
            from .editor_viewport import EditorViewport
            viewport_factory = EditorViewport
        self.viewport = viewport_factory()
        self.viewport.setObjectName("Viewport")
        self.viewport.setMinimumSize(320, 240)
        self.viewport.ground_pressed.connect(self._on_press)
        self.viewport.ground_moved.connect(self._on_move)
        self.viewport.ground_released.connect(self._on_release)
        self.viewport.hovered.connect(self._on_hover)
        self.viewport.left_widget.connect(lambda: self.viewport.set_cursor("none", 0, 0))
        self.viewport.wheel_edit.connect(self._on_wheel)
        self.viewport.key_pressed.connect(self._on_key)
        self.viewport.camera_changed.connect(self._update_corner)
        self.viewport.upload_timed.connect(lambda _n, ms: self._set_status(f"업로드 {ms:.0f} ms", transient=True))
        if hasattr(self.viewport, "double_clicked"):
            self.viewport.double_clicked.connect(self._on_double_click)
        centre.addWidget(self.viewport, 1)

        self.status_line = label("", "hint")
        self.status_line.setObjectName("Mono")
        centre.addWidget(self.status_line)
        root.addLayout(centre, 1)

        # -- right: inspector
        right_area = QtWidgets.QScrollArea()
        right_area.setObjectName("PanelScroll")
        right_area.setWidgetResizable(True)
        right_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        right_area.setMinimumWidth(300)
        right_area.setMaximumWidth(380)
        right_inner = QtWidgets.QWidget()
        right_inner.setObjectName("Panel")
        rv = QtWidgets.QVBoxLayout(right_inner)
        rv.setContentsMargins(SP[1], SP[1], SP[1], SP[1])
        rv.setSpacing(SP[2])

        lib = Card("에셋 라이브러리")
        self.lib_list = QtWidgets.QListWidget()
        self.lib_list.setMinimumHeight(150)
        self.lib_list.currentRowChanged.connect(lambda _r: self._on_library_pick())
        lib.add(self.lib_list)
        lb = QtWidgets.QHBoxLayout()
        self.btn_import_asset = QtWidgets.QPushButton("메쉬 가져오기…")
        self.btn_import_asset.clicked.connect(self.import_asset_dialog)
        lb.addWidget(self.btn_import_asset)
        self.btn_remove_asset = QtWidgets.QPushButton("에셋 제거")
        self.btn_remove_asset.setObjectName("GhostButton")
        self.btn_remove_asset.clicked.connect(self.remove_selected_asset)
        lb.addWidget(self.btn_remove_asset)
        lib.add(lb)
        self.lib_note = label("GLB · glTF · OBJ · STL · PLY. 가져온 메쉬는 씬 폴더에 복사되고, 충돌체는 높이 밴드별 "
                              "볼록 단면으로 자동 생성됩니다 (LiDAR 와 접촉이 같은 형상을 봅니다).", "hint")
        lib.add(self.lib_note)
        rv.addWidget(lib)

        insp = Card("선택")
        self.insp_none = label("장애물이나 경로를 클릭하면 여기서 수치로 조정합니다.", "hint")
        insp.add(self.insp_none)
        # -- a path: width, hose band (track), closed, all-smooth / all-corner, direction, delete
        self.path_form = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(self.path_form)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(SP[1])
        self.path_name = label("", "body")
        self.path_name.setObjectName("Mono")
        pv.addWidget(self.path_name)
        pg = QtWidgets.QGridLayout()
        pg.setHorizontalSpacing(SP[1])
        self.spin_path_width = QtWidgets.QDoubleSpinBox()
        self.spin_path_width.setRange(0.05, 8.0)
        self.spin_path_width.setSingleStep(0.05)
        self.spin_path_width.setDecimals(2)
        self.spin_path_width.setSuffix(" m")
        self.row_path_width = FieldRow("폭", self.spin_path_width)
        pg.addWidget(self.row_path_width, 0, 0)
        self.spin_path_band = QtWidgets.QDoubleSpinBox()
        self.spin_path_band.setRange(0.05, 1.5)
        self.spin_path_band.setSingleStep(0.01)
        self.spin_path_band.setDecimals(2)
        self.spin_path_band.setSuffix(" m")
        self.row_path_band = FieldRow("호스 굵기", self.spin_path_band)
        pg.addWidget(self.row_path_band, 0, 1)
        pv.addLayout(pg)
        self.chk_path_closed = QtWidgets.QCheckBox("닫힌 경로 (C)")
        pv.addWidget(self.chk_path_closed)
        self.path_stats = label("", "hint")
        pv.addWidget(self.path_stats)
        pr = QtWidgets.QHBoxLayout()
        self.btn_all_smooth = QtWidgets.QPushButton("모두 곡선")
        self.btn_all_smooth.setObjectName("GhostButton")
        self.btn_all_smooth.clicked.connect(lambda: self.set_path_all_smooth(True))
        pr.addWidget(self.btn_all_smooth)
        self.btn_all_corner = QtWidgets.QPushButton("모두 직선")
        self.btn_all_corner.setObjectName("GhostButton")
        self.btn_all_corner.clicked.connect(lambda: self.set_path_all_smooth(False))
        pr.addWidget(self.btn_all_corner)
        pv.addLayout(pr)
        pr = QtWidgets.QHBoxLayout()
        self.btn_reverse = QtWidgets.QPushButton("방향 뒤집기")
        self.btn_reverse.setObjectName("GhostButton")
        self.btn_reverse.setToolTip("트랙 경로의 주행 방향 (센터라인은 항상 반시계 방향으로 저장됩니다)")
        self.btn_reverse.clicked.connect(self.reverse_path)
        pr.addWidget(self.btn_reverse)
        self.btn_del_path = QtWidgets.QPushButton("삭제")
        self.btn_del_path.setObjectName("DangerButton")
        self.btn_del_path.clicked.connect(self.delete_selection)
        pr.addWidget(self.btn_del_path)
        pv.addLayout(pr)
        pv.addWidget(label("꼭짓점: 드래그 이동 · 더블클릭/Tab 직선↔곡선 · Alt+클릭 선 위에 추가 · Delete 삭제", "hint"))
        self.path_form.setVisible(False)
        insp.add(self.path_form)
        self.spin_path_width.valueChanged.connect(self._on_path_edit)
        self.spin_path_band.valueChanged.connect(self._on_path_edit)
        self.chk_path_closed.toggled.connect(self._on_path_edit)
        self.insp_form = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(self.insp_form)
        fv.setContentsMargins(0, 0, 0, 0)
        fv.setSpacing(SP[1])
        self.insp_name = label("", "body")
        self.insp_name.setObjectName("Mono")
        fv.addWidget(self.insp_name)
        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(SP[1])
        self.spin_x = QtWidgets.QDoubleSpinBox()
        self.spin_y = QtWidgets.QDoubleSpinBox()
        for s in (self.spin_x, self.spin_y):
            s.setRange(-1000.0, 1000.0)
            s.setDecimals(3)
            s.setSingleStep(0.05)
            s.setSuffix(" m")
        self.spin_yaw = QtWidgets.QDoubleSpinBox()
        self.spin_yaw.setRange(-360.0, 360.0)
        self.spin_yaw.setDecimals(1)
        self.spin_yaw.setSingleStep(5.0)
        self.spin_yaw.setSuffix(" °")
        self.spin_yaw.setWrapping(True)
        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(0, 9999)
        g.addWidget(FieldRow("x", self.spin_x), 0, 0)
        g.addWidget(FieldRow("y", self.spin_y), 0, 1)
        g.addWidget(FieldRow("방향", self.spin_yaw), 1, 0)
        g.addWidget(FieldRow("seed (외형)", self.spin_seed), 1, 1)
        fv.addLayout(g)
        self.dims_box = QtWidgets.QWidget()
        self.dims_layout = QtWidgets.QGridLayout(self.dims_box)
        self.dims_layout.setContentsMargins(0, 0, 0, 0)
        self.dims_layout.setHorizontalSpacing(SP[1])
        fv.addWidget(self.dims_box)
        self._dim_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        ir = QtWidgets.QHBoxLayout()
        self.btn_dup_prop = QtWidgets.QPushButton("복제 (Ctrl+D)")
        self.btn_dup_prop.setObjectName("GhostButton")
        self.btn_dup_prop.clicked.connect(self.duplicate_selection)
        ir.addWidget(self.btn_dup_prop)
        self.btn_bake = QtWidgets.QPushButton("벽으로 굽기")
        self.btn_bake.setObjectName("GhostButton")
        self.btn_bake.setToolTip("장애물의 바닥 자국을 격자에 찍고 장애물은 지웁니다. 볼록하지 않은 메쉬를 정확한 "
                                 "높은 벽으로 만들 때 씁니다 (빔이 위로 넘지 못하게 됩니다).")
        self.btn_bake.clicked.connect(self.bake_selection)
        ir.addWidget(self.btn_bake)
        self.btn_del_prop = QtWidgets.QPushButton("삭제")
        self.btn_del_prop.setObjectName("DangerButton")
        self.btn_del_prop.clicked.connect(self.delete_selection)
        ir.addWidget(self.btn_del_prop)
        fv.addLayout(ir)
        self.insp_form.setVisible(False)
        insp.add(self.insp_form)
        rv.addWidget(insp)
        for s in (self.spin_x, self.spin_y, self.spin_yaw):
            s.valueChanged.connect(self._on_inspector_edit)
        self.spin_seed.valueChanged.connect(self._on_inspector_edit)
        self._insp_updating = False

        layers = Card("레이어")
        self.chk_show_ducts = QtWidgets.QCheckBox("덕트 호스")
        self.chk_show_walls = QtWidgets.QCheckBox("높은 벽")
        self.chk_show_props = QtWidgets.QCheckBox("장애물")
        self.chk_show_floor = QtWidgets.QCheckBox("바닥")
        for c in (self.chk_show_ducts, self.chk_show_walls, self.chk_show_props, self.chk_show_floor):
            c.setChecked(True)
            c.toggled.connect(self._apply_layer_visibility)
            layers.add(c)
        rv.addWidget(layers)

        settings = Collapsible("환경 설정", expanded=False)
        self.spin_duct_h = QtWidgets.QDoubleSpinBox()
        self.spin_duct_h.setRange(0.05, 1.5)
        self.spin_duct_h.setSingleStep(0.01)
        self.spin_duct_h.setDecimals(2)
        self.spin_duct_h.setSuffix(" m")
        self.spin_duct_h.valueChanged.connect(self._on_duct_height)
        settings.add(FieldRow("덕트 높이", self.spin_duct_h, "호스 지름. 대회 덕트는 0.33 m 안팎"))
        self.spin_wall_h = QtWidgets.QDoubleSpinBox()
        self.spin_wall_h.setRange(0.2, 5.0)
        self.spin_wall_h.setSingleStep(0.1)
        self.spin_wall_h.setDecimals(1)
        self.spin_wall_h.setSuffix(" m")
        self.spin_wall_h.valueChanged.connect(self._on_wall_height)
        settings.add(FieldRow("벽 표시 높이", self.spin_wall_h, "그림용. 시뮬레이터는 높은 벽을 무한히 높게 봅니다"))
        sr = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton("캔버스 맞추기")
        self.btn_fit.setObjectName("GhostButton")
        self.btn_fit.setToolTip("내용 주변 2 m 여백으로 캔버스를 줄이거나 늘립니다")
        self.btn_fit.clicked.connect(self.fit_canvas)
        sr.addWidget(self.btn_fit)
        self.btn_border = QtWidgets.QPushButton("가장자리 벽")
        self.btn_border.setObjectName("GhostButton")
        self.btn_border.clicked.connect(self.add_border)
        sr.addWidget(self.btn_border)
        vr = QtWidgets.QHBoxLayout()
        self.btn_vec_duct = QtWidgets.QPushButton("칠한 덕트 → 경로")
        self.btn_vec_duct.setToolTip("브러시로 칠했거나 가져온 맵에서 온 덕트 셀을 꼭짓점이 있는 경로로 바꿉니다. "
                                     "그 뒤로는 선택 도구로 잡아 고칠 수 있습니다.")
        self.btn_vec_duct.clicked.connect(lambda: self.vectorize("duct"))
        vr.addWidget(self.btn_vec_duct)
        self.btn_vec_tall = QtWidgets.QPushButton("칠한 벽 → 경로")
        self.btn_vec_tall.setToolTip("벽 셀을 중심선 경로로 바꿉니다 (두께는 원래 벽의 중간값). 넓은 방 윤곽 같은 큰 면은 "
                                     "경로보다 칠한 채로 두는 편이 낫습니다.")
        self.btn_vec_tall.clicked.connect(lambda: self.vectorize("tall"))
        vr.addWidget(self.btn_vec_tall)
        settings.add(vr)
        self.btn_clear_cl = QtWidgets.QPushButton("센터라인 지우기")
        self.btn_clear_cl.setObjectName("GhostButton")
        self.btn_clear_cl.setToolTip("트랙 모양을 크게 바꿨으면 지우고 검증으로 다시 뽑으세요")
        self.btn_clear_cl.clicked.connect(self.clear_centerline)
        sr.addWidget(self.btn_clear_cl)
        settings.add(sr)
        self.edit_notes = QtWidgets.QPlainTextEdit()
        self.edit_notes.setPlaceholderText("메모")
        self.edit_notes.setMaximumHeight(60)
        self.edit_notes.textChanged.connect(self._on_notes)
        settings.add(self.edit_notes)
        rv.addWidget(settings)

        val = Card("검증")
        self.val_list = QtWidgets.QListWidget()
        self.val_list.setMinimumHeight(110)
        self.val_list.setWordWrap(True)
        self.val_list.itemClicked.connect(self._on_issue_clicked)
        val.add(self.val_list)
        self.val_stats = KeyValueList()
        for k in ("센터라인", "길이", "최소 폭", "장애물"):
            self.val_stats.set(k, "—")
        val.add(self.val_stats)
        rv.addWidget(val, 1)
        right_area.setWidget(right_inner)
        root.addWidget(right_area, 0)

        self._fill_library()
        self._install_shortcuts()

    def _build_generator_card(self) -> QtWidgets.QWidget:
        from f1sim import trackgen as TG
        card = Card("랜덤 트랙 생성")
        row = QtWidgets.QHBoxLayout()
        self.combo_gen_size = QtWidgets.QComboBox()
        for k, v in TG.SIZES.items():
            self.combo_gen_size.addItem(k, v)
        self.combo_gen_size.setCurrentIndex(1)
        row.addWidget(FieldRow("규모", self.combo_gen_size), 1)
        self.spin_gen_lane = QtWidgets.QDoubleSpinBox()
        self.spin_gen_lane.setRange(0.8, 4.0)
        self.spin_gen_lane.setSingleStep(0.1)
        self.spin_gen_lane.setValue(1.6)
        self.spin_gen_lane.setSuffix(" m")
        row.addWidget(FieldRow("차선 폭", self.spin_gen_lane))
        self.spin_gen_seed = QtWidgets.QSpinBox()
        self.spin_gen_seed.setRange(0, 999999)
        self.spin_gen_seed.setValue(int(time.time()) % 1000)
        row.addWidget(FieldRow("seed", self.spin_gen_seed))
        card.add(row)
        # features: a checkbox (may appear) and a count (-1 = free, shown as "자동")
        self._gen_feature_widgets: Dict[str, Tuple[QtWidgets.QCheckBox, QtWidgets.QSpinBox]] = {}
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(SP[1])
        grid.setVerticalSpacing(2)
        items = [("run", k, v) for k, v in TG.RUN_FEATURES.items()] + [("turn", k, v) for k, v in TG.TURN_FEATURES.items()]
        default_on = {"straight", "chicane", "slalom_fast", "corner", "sweeper", "hairpin"}
        for i, (_grp, key, spec) in enumerate(items):
            chk = QtWidgets.QCheckBox(spec["label"])
            chk.setChecked(key in default_on)
            chk.setToolTip(spec["hint"])
            cnt = QtWidgets.QSpinBox()
            cnt.setRange(-1, 8)
            cnt.setValue(-1)
            cnt.setSpecialValueText("자동")
            cnt.setToolTip("개수. '자동'이면 무작위로 몇 개든")
            cnt.setFixedWidth(64)
            chk.toggled.connect(cnt.setEnabled)
            cnt.setEnabled(chk.isChecked())
            grid.addWidget(chk, i // 2, (i % 2) * 2)
            grid.addWidget(cnt, i // 2, (i % 2) * 2 + 1)
            self._gen_feature_widgets[key] = (chk, cnt)
        card.add(grid)
        gr = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("생성")
        self.btn_generate.setObjectName("PrimaryButton")
        self.btn_generate.clicked.connect(self.generate_track)
        gr.addWidget(self.btn_generate)
        self.btn_regenerate = QtWidgets.QPushButton("다른 seed 로 다시")
        self.btn_regenerate.setObjectName("GhostButton")
        self.btn_regenerate.clicked.connect(self._regenerate)
        gr.addWidget(self.btn_regenerate)
        card.add(gr)
        self.gen_note = label("고른 항목들로 닫힌 트랙을 무작위로 만듭니다. 결과는 트랙 경로라서 바로 꼭짓점을 잡아 고칠 수 있고, "
                              "같은 항목·seed 면 같은 트랙이 나옵니다. 배치 생성은 `python -m f1sim.trackgen`.", "hint")
        card.add(self.gen_note)
        return card

    def generator_recipe(self):
        from f1sim import trackgen as TG
        runs = {k: (cnt.value() if chk.isChecked() else 0) for k, (chk, cnt) in self._gen_feature_widgets.items()
                if k in TG.RUN_FEATURES}
        turns = {k: (cnt.value() if chk.isChecked() else 0) for k, (chk, cnt) in self._gen_feature_widgets.items()
                 if k in TG.TURN_FEATURES}
        return TG.TrackRecipe(runs=runs, turns=turns, size_m=float(self.combo_gen_size.currentData()),
                              lane_width=float(self.spin_gen_lane.value()))

    def generate_track(self, seed: Optional[int] = None) -> bool:
        from f1sim import trackgen as TG
        if not self._confirm_discard():
            return False
        seed = int(self.spin_gen_seed.value()) if seed is None else int(seed)
        recipe = self.generator_recipe()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            g = TG.generate(recipe, seed)
        except RuntimeError as exc:
            self._set_status(str(exc), danger=True)
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        name = self._unique_name(f"rand_{seed}")
        doc = g.to_scene(name)
        self.load_doc(doc, dirty=True)
        feats = " · ".join(dict.fromkeys(TG.RUN_FEATURES.get(f, TG.TURN_FEATURES.get(f, {})).get("label", f)
                                         for f in g.features))
        self._set_status(f"'{name}' 생성 · 길이 {g.length_m:.1f} m · {g.attempts}번째 시도 · {feats}. "
                         f"손질한 뒤 저장하세요.")
        return True

    def _regenerate(self):
        self.spin_gen_seed.setValue(self.spin_gen_seed.value() + 1)
        self.generate_track()

    def _install_shortcuts(self):
        def sc(seq, fn):
            s = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            s.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)
            return s
        sc("Ctrl+S", self.save)
        sc("Ctrl+Z", self.undo)
        sc("Ctrl+Y", self.redo)
        sc("Ctrl+Shift+Z", self.redo)
        for key, _t, _tip, short in TOOLS:
            sc(short, lambda k=key: self.set_tool(k))

    def _fill_library(self):
        self.lib_list.clear()
        for style, name, size in BUILTIN_PROPS:
            it = QtWidgets.QListWidgetItem(f"{name}    {size}")
            it.setData(QtCore.Qt.UserRole, ("builtin", style))
            it.setToolTip(style)
            self.lib_list.addItem(it)
        doc = self.state.doc
        if doc is not None:
            for a in doc.assets:
                sz = " × ".join(f"{v:.2f}" for v in a.size) + " m" if a.size else ""
                it = QtWidgets.QListWidgetItem(f"▣ {a.name}    {sz}  · {a.tris} tri")
                it.setData(QtCore.Qt.UserRole, ("asset", a.id))
                it.setToolTip(a.file)
                self.lib_list.addItem(it)
        if self.lib_list.count() and self.lib_list.currentRow() < 0:
            self.lib_list.setCurrentRow(0)

    # ---------------------------------------------------------------- tools
    def _make_tools(self):
        self._tools = {
            "select": SelectTool(self),
            "brush_duct": BrushTool(self, "brush_duct", "duct", True),
            "brush_tall": BrushTool(self, "brush_tall", "tall", True),
            "erase": BrushTool(self, "erase", "free", False),
            "path_track": PathTool(self, "path_track", "track"),
            "path_duct": PathTool(self, "path_duct", "duct"),
            "path_tall": PathTool(self, "path_tall", "tall"),
            "rect": RectTool(self),
            "polygon": PolygonTool(self),
            "place": PlaceTool(self),
        }

    def set_tool(self, key: str):
        if key not in self._tools:
            return
        if self._tool is not None:
            self._tool.deactivate()
        self._tool = self._tools[key]
        self._tool.activate()
        self.tool_buttons.set_current(key)
        self.viewport.set_hover(None)
        self.viewport.set_cursor("none", 0, 0)
        self.tool_hint.setText(self._tool.hint)
        if isinstance(self._tool, PathTool):
            self.chk_close.blockSignals(True)
            self.chk_close.setChecked(self._tool.closed)
            self.chk_close.blockSignals(False)
        self.viewport.update()
        self._update_corner()

    def _on_close_toggled(self, on: bool):
        if isinstance(self._tool, PathTool):
            self._tool.closed = bool(on)
            self._tool._preview()

    def lane_width(self) -> float:
        return float(self.spin_lane.value())

    # -- live feedback while a stroke or a vertex drag is in progress
    LIVE_MS = 180

    def begin_live_grid(self):
        """While the button is down: rebuild the real geometry on a fixed cadence so the hoses and
        walls grow behind the brush, in addition to the instant decal the tool draws."""
        if not hasattr(self, "_live_timer"):
            self._live_timer = QtCore.QTimer(self)
            self._live_timer.setInterval(self.LIVE_MS)
            self._live_timer.timeout.connect(self._live_tick)
        self._live_dirty = False
        self._live_timer.start()

    def _live_tick(self):
        if getattr(self, "_live_path", None) is not None and self.state.doc is not None:
            self.state.doc.path_changed()
            self._live_path = None
        self._rebuild_full()

    def end_live_grid(self):
        t = getattr(self, "_live_timer", None)
        if t is not None:
            t.stop()
        self._live_path = None
        self._rebuild_full()

    def path_edited_live(self, q):
        """A path's points changed under the mouse: instant decal now, raster on the next tick."""
        self._live_path = q
        self.refresh_overlays(live_path=q)

    def set_hover_vertex(self, hv):
        if getattr(self, "_hover_vertex", None) != hv:
            self._hover_vertex = hv
            self.refresh_overlays()

    def set_path_handles(self, items):
        """Handles a path tool shows while drawing (the selection's handles are drawn by
        `refresh_overlays`; the two never show at once because drawing clears the selection)."""
        self.viewport.set_handles(items)

    def current_tool(self) -> str:
        return self._tool.name if self._tool else ""

    def brush_radius(self, layer: Optional[str] = None) -> float:
        """The wall / eraser brush is the spin box; the duct brush is always half the hose
        diameter -- a duct is a hose, and a wider band would draw (and scan) as two of them."""
        if layer == "duct" and self.state.doc is not None:
            return 0.5 * float(self.state.doc.duct_height)
        return float(self.spin_brush.value())

    def line_width(self) -> float:
        return 2.0 * float(self.spin_brush.value())

    def step_brush(self, direction: int):
        v = self.spin_brush.value() * (1.15 if direction > 0 else 1 / 1.15)
        self.spin_brush.setValue(max(self.spin_brush.minimum(), min(self.spin_brush.maximum(), v)))
        self.viewport.update()

    def snap_step(self) -> float:
        return 0.1 if self.chk_snap.isChecked() else 0.0

    def snap_point(self, x: float, y: float) -> Tuple[float, float]:
        s = self.snap_step()
        if s <= 0:
            return float(x), float(y)
        return round(x / s) * s, round(y / s) * s

    def close_lines(self) -> bool:
        return self.chk_close.isChecked()

    # -- viewport events → tool
    def _on_press(self, x, y, button, mods, hit):
        if self._tool is None or self.state.doc is None:
            return
        self._tool.press(x, y, int(button), QtCore.Qt.KeyboardModifiers(mods), hit)
        self.viewport.update()

    def _on_move(self, x, y, buttons, mods):
        if self._tool is None or self.state.doc is None:
            return
        self._tool.move(x, y, int(buttons), QtCore.Qt.KeyboardModifiers(mods))
        self.viewport.update()

    def _on_release(self, x, y, button, mods):
        if self._tool is None or self.state.doc is None:
            return
        self._tool.release(x, y, int(button), QtCore.Qt.KeyboardModifiers(mods))
        self.viewport.update()

    def _on_hover(self, x, y, hit):
        if self._tool is None or self.state.doc is None:
            return
        self._tool.hover(x, y, hit)
        self._set_coords(x, y)
        self.viewport.update()

    def _on_double_click(self, x, y):
        if self._tool is not None:
            self._tool.double_click(x, y)
            self.viewport.update()

    def _on_wheel(self, delta, mods):
        if self._tool is not None and self._tool.wheel(int(delta), QtCore.Qt.KeyboardModifiers(mods)):
            self.viewport.update()

    def _on_key(self, key, mods):
        m = QtCore.Qt.KeyboardModifiers(mods)
        if self._tool is not None and self._tool.key(int(key), m):
            self.viewport.update()
            return
        if key == QtCore.Qt.Key_Escape:
            self.state.select([])
            self.set_tool("select")

    # ---------------------------------------------------------------- document lifecycle
    def set_active(self, on: bool):
        self._active = on
        if on:
            self.refresh_scene_list()
            self.viewport.update()

    def refresh_scene_list(self):
        S = _scene_module()
        cur = self.scene_list.currentItem().data(QtCore.Qt.UserRole) if self.scene_list.currentItem() else None
        self.scene_list.blockSignals(True)
        self.scene_list.clear()
        try:
            scenes = S.list_scenes()
        except Exception as exc:
            scenes = []
            self._set_status(f"환경 목록을 읽지 못했습니다: {exc}")
        for s in scenes:
            shape = s.get("shape") or (0, 0)
            it = QtWidgets.QListWidgetItem(f"{s['name']}    {s.get('props', 0)}개 장애물 · {shape[1]}×{shape[0]}")
            it.setData(QtCore.Qt.UserRole, s["name"])
            it.setToolTip(s.get("dir", ""))
            self.scene_list.addItem(it)
            if s["name"] == cur:
                self.scene_list.setCurrentItem(it)
        self.scene_list.blockSignals(False)
        self.scene_root_label.setText(f"{S.scenes_root()}  ·  {len(scenes)}개")

    def _confirm_discard(self) -> bool:
        if not self.state.dirty or self.state.doc is None:
            return True
        r = QtWidgets.QMessageBox.question(
            self, "저장하지 않은 변경", f"'{self.state.doc.name}' 에 저장하지 않은 변경이 있습니다. 저장할까요?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save)
        if r == QtWidgets.QMessageBox.Cancel:
            return False
        if r == QtWidgets.QMessageBox.Save:
            return self.save()
        return True

    def _unique_name(self, base: str) -> str:
        S = _scene_module()
        names = {s["name"] for s in S.list_scenes()}
        if base not in names:
            return base
        i = 2
        while f"{base}_{i}" in names:
            i += 1
        return f"{base}_{i}"

    def new_blank(self):
        if not self._confirm_discard():
            return
        S = _scene_module()
        name = self._unique_name(time.strftime("scene_%m%d_%H%M"))
        doc = S.SceneDoc.new_blank(name, float(self.spin_w.value()), float(self.spin_h.value()),
                                   resolution=float(self.spin_res.value()))
        if self.chk_border.isChecked():
            doc.paint_border("tall", 0.15)
        self.load_doc(doc, dirty=True)
        self._set_status(f"'{name}' 을 만들었습니다. 저장하면 {S.scenes_root()} 아래에 폴더가 생깁니다.")

    def load_doc(self, doc, dirty: bool = False):
        self.state.set_doc(doc, dirty=dirty)
        self._validation = None
        self.val_list.clear()
        self._fill_library()
        self._sync_settings()
        self._rebuild_full(frame=True)

    def open_scene(self, name: str) -> bool:
        if not self._confirm_discard():
            return False
        S = _scene_module()
        try:
            doc = S.SceneDoc.load(name)
        except Exception as exc:
            self._set_status(f"'{name}' 을 열지 못했습니다: {exc}", danger=True)
            return False
        self.load_doc(doc)
        self._set_status(f"'{name}' 을 열었습니다.")
        return True

    def open_selected(self):
        it = self.scene_list.currentItem()
        if it is not None:
            self.open_scene(it.data(QtCore.Qt.UserRole))

    def duplicate_selected(self):
        it = self.scene_list.currentItem()
        if it is None:
            return
        S = _scene_module()
        src = it.data(QtCore.Qt.UserRole)
        try:
            doc = S.SceneDoc.load(src)
            doc.name = self._unique_name(f"{src}_copy")
            doc.dir = None
            new_dir = doc.save(doc.name)
            # assets are referenced relative to the scene dir: copy them across
            src_assets = os.path.join(S.scene_dir(src), "assets")
            if os.path.isdir(src_assets):
                import shutil
                shutil.copytree(src_assets, os.path.join(new_dir, "assets"), dirs_exist_ok=True)
        except Exception as exc:
            self._set_status(f"복제 실패: {exc}", danger=True)
            return
        self.refresh_scene_list()
        self.scenes_changed.emit()
        self._set_status(f"'{src}' 을 '{doc.name}' 으로 복제했습니다.")

    def delete_selected(self):
        it = self.scene_list.currentItem()
        if it is None:
            return
        name = it.data(QtCore.Qt.UserRole)
        S = _scene_module()
        d = S.scene_dir(name)
        r = QtWidgets.QMessageBox.warning(self, "환경 삭제", f"'{name}' 폴더를 지웁니다:\n{d}\n되돌릴 수 없습니다.",
                                          QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                                          QtWidgets.QMessageBox.Cancel)
        if r != QtWidgets.QMessageBox.Yes:
            return
        import shutil
        try:
            shutil.rmtree(d)
        except OSError as exc:
            self._set_status(f"삭제 실패: {exc}", danger=True)
            return
        if self.state.doc is not None and self.state.doc.name == name:
            self.state.doc.dir = None
        self.refresh_scene_list()
        self.scenes_changed.emit()

    def save(self) -> bool:
        doc = self.state.doc
        if doc is None:
            return False
        try:
            if doc.dir is None:
                doc.name = self._unique_name(doc.name)
            doc.notes = self.edit_notes.toPlainText()
            doc.save()
        except Exception as exc:
            self._set_status(f"저장 실패: {exc}", danger=True)
            return False
        self.state.mark_saved()
        self.refresh_scene_list()
        self.scenes_changed.emit()
        self._update_title()
        self._set_status(f"저장했습니다: {doc.dir}")
        return True

    def save_as(self):
        doc = self.state.doc
        if doc is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "다른 이름으로 저장", "환경 이름", text=doc.name)
        if not ok or not name.strip():
            return
        name = name.strip().replace("/", "_")
        old_dir = doc.dir
        doc.name, doc.dir = name, None
        try:
            new_dir = doc.save(name)
            if old_dir and os.path.isdir(os.path.join(old_dir, "assets")):
                import shutil
                shutil.copytree(os.path.join(old_dir, "assets"), os.path.join(new_dir, "assets"), dirs_exist_ok=True)
        except Exception as exc:
            self._set_status(f"저장 실패: {exc}", danger=True)
            doc.dir = old_dir
            return
        self.state.mark_saved()
        self.refresh_scene_list()
        self.scenes_changed.emit()
        self._update_title()
        self._set_status(f"저장했습니다: {new_dir}")

    def set_maps(self, cat):
        """The worker's map catalogue, forwarded by the window; scenes are excluded (복제 covers them).

        Base tracks, listed by display name with the id beside them -- the same rows the driving
        page's map card shows. 복제 hands whichever is selected to `maps.load`, which takes either
        grammar, so an id works here exactly as a loader name did.
        """
        from .catalog import SCENES_GROUP
        if not getattr(cat, "ready", False):
            return
        items = []
        for tid in cat.ids():
            g = cat.group_of(tid) or ""
            if g == SCENES_GROUP:
                continue
            e = cat.entry(tid)
            items.append((g, tid, e.get("display") or tid,
                          f"{tid} · {e.get('family_label') or e.get('family', '')}"))
        self.map_picker.set_items(items)
        self.map_picker.set_status(f"{len({k for _g, k, _t, _s in items})} 개", "hint")

    def catalog_choice(self) -> str:
        """The picked map, else whatever was typed in the search box."""
        return (self.map_picker.selected() or self.map_picker.search.text()).strip()

    def import_from_catalog(self):
        name = self.catalog_choice()
        if not name:
            self._set_status("복제할 맵을 목록에서 고르거나 이름을 입력하세요 (예: gen/comp-0).", danger=True)
            return
        if not self._confirm_discard():
            return
        target = self._unique_name(name.replace(":", "_").replace("/", "_").replace("+", "_")
                                       .replace("~", "_").replace("@", "_").replace("#", "_"))
        self._run_job(["import", name, target], f"'{name}' 을 가져오는 중… (Track 로드, 몇 초)",
                      lambda code, out, err: self._imported(code, out, err, target))

    # ---------------------------------------------------------------- SLAM map import
    def _browse_slam(self):
        start = os.path.dirname(self.slam_path.text().strip()) or os.path.expanduser("~")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "SLAM 맵 고르기", start, "맵 (*.yaml *.yml *.pgm *.png);; 모든 파일 (*)")
        if path:
            self.slam_path.setText(path)

    def import_from_slam(self):
        path = self.slam_path.text().strip()
        if not path:
            self._set_status("불러올 SLAM 맵(yaml 또는 이미지)을 고르세요.", danger=True)
            return
        if not os.path.exists(os.path.expanduser(path)):
            self._set_status(f"파일이 없습니다: {path}", danger=True)
            return
        if not self._confirm_discard():
            return
        stem = os.path.splitext(os.path.basename(os.path.expanduser(path).rstrip(os.sep)))[0] or "slam_map"
        stem = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in stem)
        target = self._unique_name(stem or "slam_map")
        args = ["import-slam", os.path.expanduser(path), target,
                "--boundary", self.slam_boundary.currentData(),
                "--min-clearance", f"{self.slam_clearance.value():.2f}"]
        if self.slam_use_seed.isChecked():
            args += ["--seed-xy", f"{self.slam_seed_x.value():.3f}", f"{self.slam_seed_y.value():.3f}"]
        if not self.slam_keep.isChecked():
            args += ["--no-keep-region"]
        self._run_job(args, f"SLAM 맵을 읽는 중… (격자 정리·주행선 추출, 십여 초)",
                      lambda code, out, err: self._imported_slam(code, out, err, target))

    def _imported_slam(self, code, out, err, target):
        if code != 0 or not out:
            msg = (err.strip().splitlines() or [str(code)])[-1]
            self._set_status(f"SLAM 맵 불러오기 실패: {msg}", danger=True)
            return
        self.refresh_scene_list()
        self.scenes_changed.emit()
        if not self.open_scene(target):
            return
        lane = out.get("lane_length_m") or 0.0
        half = out.get("half_width_min_m") or 0.0
        lap = out.get("raceline_lap_s")
        msg = (f"'{target}' 으로 불러왔습니다 · 주행선 {lane:.1f} m · 최소 반폭 {half:.2f} m"
               + (f" · 레이스라인 {lap:.2f} s" if lap else ""))
        problems = out.get("problems") or []
        if problems:
            # Imported anyway: a map with a pinch point or a failed raceline is exactly the map
            # someone opens the editor to fix, and refusing it would leave them nothing to fix.
            self._set_status(msg + " · 확인 필요: " + problems[0], danger=True)
        else:
            self._set_status(msg)

    def _imported(self, code, out, err, target):
        if code != 0 or not out:
            self._set_status(f"가져오기 실패: {err.strip().splitlines()[-1] if err.strip() else code}", danger=True)
            return
        self.refresh_scene_list()
        self.scenes_changed.emit()
        if self.open_scene(target):
            self._set_status(f"'{target}' 으로 가져왔습니다 · 장애물 {out.get('props', 0)}개 · {out.get('shape')}")

    # ---------------------------------------------------------------- validation + drive
    def validate(self, then: Optional[Callable[[bool], None]] = None):
        if self.state.doc is None:
            return
        if self.state.dirty or self.state.doc.dir is None:
            if not self.save():
                return
        name = self.state.doc.dir
        # A track path *is* the centerline: keep it. Otherwise extract one from the free space.
        mode = "keep" if self.state.doc.track_path is not None else "auto"
        self._run_job(["validate", name, "--centerline", mode], "검증 중… (센터라인 확인, 몇 초)",
                      lambda code, out, err: self._validated(code, out, err, then))

    def _validated(self, code, out, err, then):
        if out is None:
            self._set_status(f"검증 실패: {err.strip().splitlines()[-1] if err.strip() else code}", danger=True)
            if then:
                then(False)
            return
        self._validation = out
        self.val_list.clear()
        for iss in out.get("issues", []):
            it = QtWidgets.QListWidgetItem(("⚠ " if iss.get("level") == "warn" else "✖ ") + iss.get("msg", ""))
            it.setData(QtCore.Qt.UserRole, iss)
            it.setForeground(QtGui.QColor(C["warn"] if iss.get("level") == "warn" else C["danger"]))
            self.val_list.addItem(it)
        if not out.get("issues"):
            it = QtWidgets.QListWidgetItem("✔ 문제 없음")
            it.setForeground(QtGui.QColor(C["ok"]))
            self.val_list.addItem(it)
        st = out.get("stats", {})
        self.val_stats.set("센터라인", "있음" if st.get("centerline") else "없음")
        self.val_stats.set("길이", f"{st['length_m']:.1f} m" if st.get("length_m") else "—")
        self.val_stats.set("최소 폭", f"{st['min_width_m']:.2f} m" if st.get("min_width_m") else "—")
        self.val_stats.set("장애물", str(st.get("props", "—")))
        # the validator may have written a centerline back: reload it without losing undo
        try:
            S = _scene_module()
            fresh = S.SceneDoc.load(self.state.doc.dir)
            if fresh.centerline is not None:
                self.state.doc.centerline = fresh.centerline
                self._rebuild_full()
        except Exception:
            pass
        ok = bool(out.get("ok"))
        self._set_status("검증 통과 · 주행 가능" if ok else "검증에서 문제가 나왔습니다. 목록을 보세요.", danger=not ok)
        if then:
            then(ok)

    def _drive(self):
        doc = self.state.doc
        if doc is None:
            return

        def go(ok):
            if ok or QtWidgets.QMessageBox.question(
                    self, "검증 문제", "검증에서 문제가 나왔습니다. 그래도 주행 페이지로 넘길까요?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes:
                self.drive_requested.emit(f"scene/{doc.name}")
        self.validate(then=go)

    def _run_job(self, args, status, cb):
        if self._job is not None:
            self._set_status("이전 작업이 아직 진행 중입니다.", danger=True)
            return
        self._set_status(status)
        self._set_busy(True)
        job = _SubprocessJob(args, self)

        def done(code, out, err):
            self._job = None
            self._set_busy(False)
            cb(code, out, err)
        job.finished.connect(done)
        self._job = job
        job.start()

    def _set_busy(self, on: bool):
        for b in (self.btn_validate, self.btn_drive, self.btn_import, self.btn_new, self.btn_open):
            b.setEnabled(not on)

    # ---------------------------------------------------------------- geometry
    def grid_edited_live(self):
        """Called during a brush stroke: schedule a debounced full rebuild."""
        self._rebuild_timer.start()

    def props_moved_live(self):
        """Called during a prop drag: rebuild just the prop batches, immediately."""
        if self.state.doc is not None:
            self._builder.request(self.state.doc, only_props=True)
        self.refresh_overlays()

    def _rebuild_full(self, frame: bool = False):
        self._rebuild_timer.stop()
        if self.state.doc is None:
            self.viewport.set_geometry(None)
            return
        self._frame_after_build = frame or getattr(self, "_frame_after_build", False)
        self._builder.request(self.state.doc, only_props=False)

    def _on_built(self, geom, only_props, ms):
        if not isinstance(geom, dict):
            return
        if "error" in geom:
            self._set_status(f"지오메트리 빌드 실패: {geom['error']}", danger=True)
            return
        from .viewport import TrackGeometry
        if only_props:
            if self._last_geometry is None:
                return
            base = self._last_geometry
            merged = TrackGeometry(**{**_geom_kwargs(base), "props": geom.get("props")})
        else:
            merged = TrackGeometry(**_geom_kwargs(geom))
        self._last_geometry = merged
        self.viewport.set_geometry(merged)
        if getattr(self, "_frame_after_build", False) and not only_props:
            self._frame_after_build = False
            QtCore.QTimer.singleShot(0, self.viewport.frame_all)
        self.refresh_overlays()
        self._set_status(f"{'장애물' if only_props else '지오메트리'} 빌드 {ms:.0f} ms", transient=True)
        self._update_corner()

    def _apply_layer_visibility(self):
        self.viewport.set_layer_visibility(ducts=self.chk_show_ducts.isChecked(),
                                           walls=self.chk_show_walls.isChecked(),
                                           props=self.chk_show_props.isChecked(),
                                           floor=self.chk_show_floor.isChecked())
        self.viewport.update()

    # ---------------------------------------------------------------- overlays + picking
    def footprint_of(self, pid: str) -> Optional[np.ndarray]:
        return self._footprints.get(pid)

    def refresh_overlays(self, live_path=None):
        doc = self.state.doc
        self._footprints = {}
        if doc is None:
            self.viewport.set_pickables([])
            self.viewport.set_selection([])
            self.viewport.set_gizmo(0, 0, 0, on=False)
            self.viewport.set_handles([])
            self.viewport.set_lines("path", [])
            return
        try:
            items = doc.prop_footprints()
        except Exception as exc:
            self._set_status(f"장애물 형상 오류: {exc}", danger=True)
            items = []
        self._footprints = {pid: poly for pid, poly in items}
        self.viewport.set_pickables(items)
        sel = [self._footprints[i] for i in self.state.selection if i in self._footprints]
        self.viewport.set_selection(sel)
        props = self.state.selected_props()
        if len(props) == 1:
            p = props[0]
            r = 0.45
            fp = self._footprints.get(p.id)
            if fp is not None and len(fp):
                r = max(0.3, float(np.max(np.hypot(fp[:, 0] - p.x, fp[:, 1] - p.y))) + 0.12)
            self.viewport.set_gizmo(p.x, p.y, p.yaw, radius=r, on=True)
        else:
            self.viewport.set_gizmo(0, 0, 0, on=False)
        self._draw_path_selection(live_path)
        self.viewport.update()

    def _draw_path_selection(self, live_path=None):
        """Handles and the centre line of the selected path; a live decal while it is dragged."""
        from f1sim.scene import offset_polyline
        doc = self.state.doc
        pid, vi = self.state.path_sel
        q = doc.get_path(pid) if pid is not None else None
        hv = getattr(self, "_hover_vertex", None)
        if q is None:
            if not isinstance(self._tool, PathTool):
                self.viewport.set_handles([])
            self.viewport.set_lines("path", [])
            if live_path is None:
                self.viewport.set_fill("pathsel", [], C["accent"])
            return
        handles = []
        for i, (x, y, sm) in enumerate(q.points):
            state = "selected" if i == vi else ("hover" if hv == (pid, i) else "normal")
            handles.append((x, y, "smooth" if sm else "corner", state))
        self.viewport.set_handles(handles)
        pl = q.polyline()
        colour = _rgba(C["accent"], 0.9)
        lines = [(pl, colour, 1.5, q.closed)]
        if q.kind == "track":
            half = 0.5 * q.width
            faint = _rgba(C["ego"], 0.5)
            lines.append((offset_polyline(pl, +half, q.closed), faint, 1.0, q.closed))
            lines.append((offset_polyline(pl, -half, q.closed), faint, 1.0, q.closed))
        # the control polygon, so a curve's vertices read as its handles
        ctrl = np.asarray([(x, y) for x, y, _ in q.points], np.float64)
        lines.append((ctrl, _rgba(C["text.2"], 0.6), 1.0, q.closed))
        self.viewport.set_lines("path", lines)
        if live_path is not None:
            band = q.band if q.band is not None else float(doc.duct_height)
            if q.kind == "track":
                half = 0.5 * q.width
                shapes = [("band", offset_polyline(pl, +half, q.closed), band, q.closed),
                          ("band", offset_polyline(pl, -half, q.closed), band, q.closed)]
            else:
                shapes = [("band", pl, q.width, q.closed)]
            self.viewport.set_fill("pathsel", shapes, LAYER_COLOUR[q.kind])
        else:
            self.viewport.set_fill("pathsel", [], C["accent"])

    def library_item(self) -> Optional[tuple]:
        it = self.lib_list.currentItem()
        return it.data(QtCore.Qt.UserRole) if it is not None else None

    def library_footprint(self, x, y, yaw) -> Optional[np.ndarray]:
        """Footprint the place tool shows before the click."""
        item = self.library_item()
        doc = self.state.doc
        if item is None or doc is None:
            return None
        S = _scene_module()
        try:
            if item[0] == "builtin":
                p = S.PropPlacement(id="_preview", style=item[1], x=x, y=y, yaw=yaw, seed=0, dims={})
            else:
                p = S.PropPlacement(id="_preview", style="mesh", x=x, y=y, yaw=yaw, seed=0,
                                    dims={}, asset=item[1])
            return p.footprint_world(doc)
        except Exception:
            return None

    def place_library_item(self, x, y, yaw) -> Optional[str]:
        item = self.library_item()
        doc = self.state.doc
        if item is None or doc is None:
            return None
        seed = len(doc.props)

        def do():
            if item[0] == "builtin":
                p = doc.add_prop(item[1], x, y, yaw, seed=seed)
            else:
                p = doc.add_prop("mesh", x, y, yaw, seed=0, asset=item[1])
            return p.id
        try:
            return self.state.commit("배치", do, "props")
        except Exception as exc:
            self._set_status(f"배치 실패: {exc}", danger=True)
            return None

    def _on_library_pick(self):
        if self.current_tool() not in ("place",) and self.library_item() is not None and self._active:
            pass                                   # picking in the library does not change the tool

    # ---------------------------------------------------------------- selection edits
    def toggle_vertex_smooth(self):
        doc = self.state.doc
        pid, vi = self.state.path_sel
        q = doc.get_path(pid) if (doc is not None and pid is not None) else None
        if q is None or vi is None:
            return

        def do():
            x, y, sm = q.points[vi]
            q.points[vi] = (x, y, not sm)
            doc.path_changed()
            return True
        self.state.commit("직선↔곡선", do, "grid")

    def toggle_path_closed(self):
        doc = self.state.doc
        q = self.state.selected_path()
        if q is None:
            return

        def do():
            q.closed = not q.closed
            doc.path_changed()
            return True
        self.state.commit("경로 닫기/열기", do, "grid")

    def set_path_all_smooth(self, smooth: bool):
        doc = self.state.doc
        q = self.state.selected_path()
        if q is None:
            return

        def do():
            q.points = [(x, y, bool(smooth)) for x, y, _ in q.points]
            doc.path_changed()
            return True
        self.state.commit("모두 곡선" if smooth else "모두 직선", do, "grid")

    def reverse_path(self):
        doc = self.state.doc
        q = self.state.selected_path()
        if q is None:
            return

        def do():
            q.points = list(reversed(q.points))
            doc.path_changed()
            return True
        self.state.commit("경로 방향 뒤집기", do, "grid")

    def nudge_selection(self, dx: float, dy: float):
        doc = self.state.doc
        if doc is None:
            return
        pid, vi = self.state.path_sel
        props = self.state.selected_props()
        if props:
            def do():
                for p in props:
                    p.x, p.y = float(p.x + dx), float(p.y + dy)
                return True
            self.state.commit("미세 이동", do, "props")
        elif pid is not None:
            q = doc.get_path(pid)

            def do():
                if vi is not None:
                    x, y, sm = q.points[vi]
                    q.points[vi] = (x + dx, y + dy, sm)
                else:
                    q.points = [(x + dx, y + dy, sm) for x, y, sm in q.points]
                doc.path_changed()
                return True
            self.state.commit("미세 이동", do, "grid")

    def delete_selection(self):
        doc = self.state.doc
        pid, vi = self.state.path_sel
        if pid is not None and doc is not None:
            if vi is not None:
                self.state.commit("꼭짓점 삭제", lambda: doc.remove_path_vertex(pid, vi), "grid")
                q = doc.get_path(pid)
                self.state.select_path(pid if q is not None else None, None)
            else:
                self.state.commit("경로 삭제", lambda: (doc.remove_path(pid), True)[1], "grid")
                self.state.select_path(None, None)
            return
        ids = list(self.state.selection)
        if not ids or self.state.doc is None:
            return

        def do():
            for i in ids:
                doc.remove_prop(i)
            return True
        self.state.commit("삭제", do, "props")
        self.state.select([])

    def duplicate_selection(self):
        props = self.state.selected_props()
        if not props:
            return
        doc = self.state.doc
        new_ids: List[str] = []

        def do():
            for p in props:
                q = doc.add_prop(p.style, p.x + 0.4, p.y + 0.4, p.yaw, seed=p.seed + 1,
                                 dims=dict(p.dims), asset=p.asset)
                new_ids.append(q.id)
            return True
        self.state.commit("복제", do, "props")
        self.state.select(new_ids)

    def rotate_selection(self, d_yaw: float):
        props = self.state.selected_props()
        if not props:
            return

        def do():
            for p in props:
                p.yaw = float((p.yaw + d_yaw) % (2 * math.pi))
            return True
        self.state.commit("회전", do, "props")

    def bake_selection(self):
        ids = list(self.state.selection)
        doc = self.state.doc
        if not ids or doc is None:
            return

        def do():
            changed = False
            for i in ids:
                changed |= bool(doc.bake_prop(i, "tall"))
            return changed
        self.state.commit("벽으로 굽기", do, "grid")
        self.state.select([])

    # ---------------------------------------------------------------- inspector
    def _sync_inspector(self):
        props = self.state.selected_props()
        q = self.state.selected_path()
        one = len(props) == 1
        self.insp_form.setVisible(bool(props))
        self.path_form.setVisible(q is not None)
        self.insp_none.setVisible(not props and q is None)
        if q is not None:
            self._sync_path_form(q)
        if not props:
            return
        p = props[0]
        self._insp_updating = True
        try:
            if one:
                asset = p.asset or ""
                self.insp_name.setText(f"{p.id} · {p.style}" + (f" · {asset}" if asset else ""))
            else:
                self.insp_name.setText(f"{len(props)}개 선택")
            self.spin_x.setValue(p.x)
            self.spin_y.setValue(p.y)
            self.spin_yaw.setValue(math.degrees(p.yaw))
            self.spin_seed.setValue(int(p.seed))
            for w in (self.spin_x, self.spin_y, self.spin_seed):
                w.setEnabled(one)
            self._rebuild_dim_spins(p if one else None)
        finally:
            self._insp_updating = False

    def _sync_path_form(self, q):
        self._insp_updating = True
        try:
            kind = {"duct": "덕트 경로", "tall": "벽 경로", "track": "트랙 경로"}[q.kind]
            pid, vi = self.state.path_sel
            self.path_name.setText(f"{q.id} · {kind}" + (f" · 꼭짓점 {vi + 1}/{len(q.points)}" if vi is not None else ""))
            self.row_path_width.label.setText("차선 폭" if q.kind == "track" else "폭")
            self.spin_path_width.setValue(float(q.width) if q.kind != "duct" else float(self.state.doc.duct_height))
            self.spin_path_width.setEnabled(q.kind != "duct")
            self.row_path_width.set_hint("덕트 굵기 = 호스 지름 (환경 설정의 덕트 높이)" if q.kind == "duct" else "")
            self.row_path_band.setVisible(q.kind == "track")
            self.spin_path_band.setValue(float(q.band if q.band is not None else self.state.doc.duct_height))
            self.chk_path_closed.setChecked(bool(q.closed))
            n_sm = sum(1 for p in q.points if p[2])
            self.path_stats.setText(f"꼭짓점 {len(q.points)}개 (곡선 {n_sm}) · 길이 {q.length():.1f} m")
            self.btn_reverse.setVisible(q.kind == "track")
        finally:
            self._insp_updating = False

    def _on_path_edit(self, _v=None):
        if self._insp_updating:
            return
        doc = self.state.doc
        q = self.state.selected_path()
        if q is None or doc is None:
            return
        width = float(self.spin_path_width.value())
        band = float(self.spin_path_band.value())
        closed = bool(self.chk_path_closed.isChecked())

        def do():
            q.width = width
            if q.kind == "track":
                q.band = band
            q.closed = closed
            doc.path_changed()
            return True
        self.state.commit("경로 설정", do, "grid")

    def _rebuild_dim_spins(self, p):
        while self.dims_layout.count():
            item = self.dims_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._dim_spins = {}
        if p is None:
            return
        from f1sim import props as PR
        keys = [k for k in PR.DIMS.get(p.style, ()) if k not in ("path", "up")]
        if not keys:
            return
        defaults = _default_dims(p.style)
        for i, k in enumerate(keys):
            s = QtWidgets.QDoubleSpinBox()
            if k == "facets":
                s.setRange(6, 64)
                s.setDecimals(0)
                s.setSingleStep(1)
            elif k == "scale":
                s.setRange(0.01, 50.0)
                s.setDecimals(3)
                s.setSingleStep(0.05)
            elif k == "taper":
                s.setRange(0.1, 1.0)
                s.setDecimals(2)
                s.setSingleStep(0.05)
            else:
                s.setRange(0.02, 5.0)
                s.setDecimals(3)
                s.setSingleStep(0.02)
                s.setSuffix(" m")
            s.setValue(float(p.dims.get(k, defaults.get(k, s.value()))))
            s.valueChanged.connect(lambda _v, key=k: self._on_dim_edit(key))
            self._dim_spins[k] = s
            self.dims_layout.addWidget(FieldRow({"width": "가로", "depth": "세로", "height": "높이", "radius": "반지름",
                                                 "facets": "면 수", "taper": "경사", "scale": "배율"}.get(k, k), s),
                                       i // 2, i % 2)

    def _on_inspector_edit(self, _v=None):
        if self._insp_updating:
            return
        props = self.state.selected_props()
        if not props:
            return
        x, y = float(self.spin_x.value()), float(self.spin_y.value())
        yaw = math.radians(float(self.spin_yaw.value()))
        seed = int(self.spin_seed.value())
        one = len(props) == 1

        def do():
            for p in props:
                if one:
                    p.x, p.y, p.seed = x, y, seed
                p.yaw = yaw
            return True
        self.state.commit("수치 조정", do, "props")

    def _on_dim_edit(self, key: str):
        if self._insp_updating:
            return
        props = self.state.selected_props()
        if len(props) != 1:
            return
        p = props[0]
        v = float(self._dim_spins[key].value())
        if key == "facets":
            v = int(v)

        def do():
            p.dims[key] = v
            return True
        try:
            self.state.commit("크기 조정", do, "props")
        except Exception as exc:
            self._set_status(f"크기 조정 실패: {exc}", danger=True)

    # ---------------------------------------------------------------- settings
    def _sync_settings(self):
        doc = self.state.doc
        self._insp_updating = True
        try:
            if doc is not None:
                self.spin_duct_h.setValue(float(doc.duct_height))
                self.spin_wall_h.setValue(float(doc.wall_height))
                self.edit_notes.setPlainText(doc.notes or "")
        finally:
            self._insp_updating = False

    def _on_duct_height(self, v):
        if self._insp_updating or self.state.doc is None:
            return
        doc = self.state.doc

        def do():
            doc.duct_height = float(v)
            return True
        self.state.commit("덕트 높이", do, "grid")

    def _on_wall_height(self, v):
        if self._insp_updating or self.state.doc is None:
            return
        doc = self.state.doc

        def do():
            doc.wall_height = float(v)
            return True
        self.state.commit("벽 높이", do, "grid")

    def _on_notes(self):
        if self._insp_updating or self.state.doc is None:
            return
        self.state.doc.notes = self.edit_notes.toPlainText()
        self.state._set_dirty(True)

    def fit_canvas(self):
        doc = self.state.doc
        if doc is None:
            return
        self.state.commit("캔버스 맞추기", lambda: (doc.fit_canvas(2.0), True)[1], "grid")
        self._frame_after_build = True

    def add_border(self):
        doc = self.state.doc
        if doc is None:
            return
        self.state.commit("가장자리 벽", lambda: doc.paint_border("tall", 0.15), "grid")

    def vectorize(self, layer: str):
        """Painted cells of a layer become editable paths (see `SceneDoc.vectorize_layer`)."""
        doc = self.state.doc
        if doc is None:
            return
        made: List[str] = []

        def do():
            paths = doc.vectorize_layer(layer)
            made.extend(q.id for q in paths)
            return bool(paths)
        try:
            self.state.commit("덕트 → 경로" if layer == "duct" else "벽 → 경로", do, "grid")
        except Exception as exc:
            self._set_status(f"경로 변환 실패: {exc}", danger=True)
            return
        if made:
            self.state.select_path(made[0], None)
            self.set_tool("select")
            self._set_status(f"{'덕트' if layer == 'duct' else '벽'} 경로 {len(made)}개를 만들었습니다. 선택 도구로 꼭짓점을 잡아 고치세요.")
        else:
            self._set_status("변환할 칠한 셀이 없습니다 (경로로 만든 것은 이미 경로입니다).")

    def clear_centerline(self):
        doc = self.state.doc
        if doc is None or doc.centerline is None:
            return

        def do():
            doc.centerline = None
            return True
        self.state.commit("센터라인 지우기", do, "grid")

    # ---------------------------------------------------------------- assets
    def import_asset_dialog(self):
        doc = self.state.doc
        if doc is None:
            self._set_status("먼저 환경을 만들거나 여세요.", danger=True)
            return
        if doc.dir is None and not self.save():
            return
        path, _f = QtWidgets.QFileDialog.getOpenFileName(self, "메쉬 가져오기", os.path.expanduser("~"), MESH_FILTER)
        if not path:
            return
        self.import_asset(path)

    def import_asset(self, path: str, target_height: Optional[float] = None) -> Optional[str]:
        doc = self.state.doc
        if doc is None:
            return None
        if doc.dir is None and not self.save():
            return None
        self._set_status(f"{os.path.basename(path)} 읽는 중…")
        QtWidgets.QApplication.processEvents()
        try:
            info = doc.import_asset(path, target_height=target_height)
        except Exception as exc:
            self._set_status(f"메쉬 가져오기 실패: {exc}", danger=True)
            return None
        self.state._set_dirty(True)
        self._fill_library()
        for i in range(self.lib_list.count()):
            if self.lib_list.item(i).data(QtCore.Qt.UserRole) == ("asset", info.id):
                self.lib_list.setCurrentRow(i)
        sz = " × ".join(f"{v:.2f}" for v in info.size) if info.size else "?"
        self._set_status(f"'{info.name}' 가져옴 · {info.tris} 삼각형 · {sz} m. 배치 도구(A)로 놓으세요.")
        self.set_tool("place")
        return info.id

    def remove_selected_asset(self):
        item = self.library_item()
        doc = self.state.doc
        if item is None or item[0] != "asset" or doc is None:
            return
        try:
            self.state.commit("에셋 제거", lambda: (doc.remove_asset(item[1]), True)[1], "props")
        except Exception as exc:
            self._set_status(f"에셋 제거 실패: {exc}", danger=True)
            return
        self._fill_library()

    # ---------------------------------------------------------------- state → UI
    def _on_state_changed(self, kind: str):
        if kind == "doc":
            self._rebuild_full()
            self._fill_library()
            self._sync_settings()
        elif kind == "grid":
            self._rebuild_timer.start()
        elif kind == "props":
            if self.state.doc is not None:
                self._builder.request(self.state.doc, only_props=True)
        self.refresh_overlays()
        self._sync_inspector()
        self._update_buttons()
        self._update_title()
        self._run_quick_issues()

    def _run_quick_issues(self):
        doc = self.state.doc
        if doc is None or self._validation is not None:
            return
        try:
            issues = doc.quick_issues()
        except Exception:
            return
        self.val_list.clear()
        for iss in issues:
            it = QtWidgets.QListWidgetItem(("⚠ " if iss.get("level") == "warn" else "✖ ") + iss.get("msg", ""))
            it.setData(QtCore.Qt.UserRole, iss)
            it.setForeground(QtGui.QColor(C["warn"] if iss.get("level") == "warn" else C["danger"]))
            self.val_list.addItem(it)

    def _on_issue_clicked(self, item):
        iss = item.data(QtCore.Qt.UserRole) or {}
        if iss.get("x") is not None and iss.get("y") is not None:
            self.viewport.look_at_point(float(iss["x"]), float(iss["y"]))
        if iss.get("prop"):
            self.state.select([iss["prop"]])

    def _update_buttons(self):
        has = self.state.doc is not None
        for b in (self.btn_save, self.btn_save_as, self.btn_validate, self.btn_drive, self.btn_import_asset,
                  self.btn_fit, self.btn_border, self.btn_clear_cl, self.btn_vec_duct, self.btn_vec_tall):
            b.setEnabled(has)
        self.btn_undo.setEnabled(self.state.can_undo())
        self.btn_undo.setToolTip(f"되돌리기: {self.state.undo_label()}  (Ctrl+Z)" if self.state.can_undo() else "되돌릴 것이 없습니다")
        self.btn_redo.setEnabled(self.state.can_redo())
        sel = bool(self.state.selection)
        for b in (self.btn_dup_prop, self.btn_bake, self.btn_del_prop):
            b.setEnabled(sel)
        item = self.library_item()
        self.btn_remove_asset.setEnabled(bool(item and item[0] == "asset"))

    def _update_title(self):
        doc = self.state.doc
        if doc is None:
            self.title_label.setText("—")
            self.summary_label.setText("환경을 새로 만들거나 목록에서 여세요.")
            return
        star = " *" if self.state.dirty else ""
        self.title_label.setText(f"{doc.name}{star}")
        H, W = doc.shape
        res = doc.resolution
        n_assets = len(doc.assets)
        cl = "센터라인 있음" if doc.centerline is not None else "센터라인 없음 (검증이 뽑음)"
        self.summary_label.setText(f"{W * res:.1f} × {H * res:.1f} m · {res:.3f} m/셀 · 장애물 {len(doc.props)}개 · "
                                   f"에셋 {n_assets}개 · {cl}" + ("" if doc.dir else " · 아직 저장 안 됨"))

    def _update_corner(self):
        doc = self.state.doc
        if doc is None:
            self.viewport.set_corner_text("")
            return
        tool = next((t for t in TOOLS if t[0] == self.current_tool()), None)
        self.viewport.set_corner_text(f"{doc.name} · {tool[1] if tool else ''}")

    def _set_coords(self, x, y):
        doc = self.state.doc
        if doc is None:
            return
        r, c = doc.world_to_cell(x, y)
        inside = 0 <= r < doc.shape[0] and 0 <= c < doc.shape[1]
        what = ""
        if inside:
            what = "덕트" if doc.duct[r, c] else ("벽" if doc.tall[r, c] else "바닥")
        self._coord_text = f"x {x:7.2f}  y {y:7.2f}   {what}"
        self._refresh_status()

    def _set_status(self, text: str, danger: bool = False, transient: bool = False):
        if transient:
            self._transient = text
        else:
            self._status = text
            self._status_danger = danger
        self._refresh_status()

    def _refresh_status(self):
        parts = [getattr(self, "_coord_text", ""), getattr(self, "_status", ""), getattr(self, "_transient", "")]
        self.status_line.setText("   ·   ".join(p for p in parts if p))
        danger = getattr(self, "_status_danger", False)
        self.status_line.setObjectName("HintDanger" if danger else "Mono")
        self.status_line.style().unpolish(self.status_line)
        self.status_line.style().polish(self.status_line)

    def undo(self):
        self.state.undo()

    def redo(self):
        self.state.redo()

    def shutdown(self):
        self._builder.stop()

    def closeEvent(self, ev):
        self.shutdown()
        super().closeEvent(ev)


def _geom_kwargs(g) -> dict:
    """Keys of a worker geometry dict / TrackGeometry that `TrackGeometry(**kw)` accepts."""
    from .viewport import TrackGeometry
    fields = set(TrackGeometry.__dataclass_fields__)
    if isinstance(g, dict):
        return {k: v for k, v in g.items() if k in fields}
    return {k: getattr(g, k) for k in fields}


def _default_dims(style: str) -> dict:
    """Default dimensions of a built-in style, read from its builder's signature."""
    import inspect
    from f1sim import props as PR
    if style == "mesh":
        return {"scale": 1.0}
    fn = PR.REGISTRY.get(style)
    if fn is None:
        return {}
    out = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name != "seed" and param.default is not inspect.Parameter.empty and isinstance(param.default, (int, float)):
            out[name] = float(param.default)
    return out
