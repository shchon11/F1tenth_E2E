"""장애물 편집: the obstacle layout of a running session, edited by hand on the map (2026-11-03).

The user, 11-03: "시드만 바꾸는게 아니라 내가 직접 장애물을 추가/제거/위치/변경해서 바로 돌리게 할 수도 있었으면
좋겠어." The session is paused, the 3D view looks straight down at the map (`ViewportWidget.begin_layout_edit`), and
the pieces are moved with the mouse:

* click a crate to pick it, drag to move it; with "＋ 놓기" on, a click on the floor puts one there;
* wheel zooms at the cursor, a right (or middle) drag -- or a left drag on the empty floor -- moves the view;
* Shift+wheel, Q / E or the ↺ ↻ buttons turn the picked one; the shape list changes what it is; Delete removes it;
  Ctrl+Z undoes.

Nothing reaches the simulator until "▶ 이 배치로 달리기": the edit is local, so a half-made layout never meets a car.
The pieces are the training generator's own shapes (`console.layout_file`), so what is placed is what the session
can collide with, see with the LiDAR and push.

Two classes: `LayoutEditor` is the layout and the operations on it (no widgets, so a test drives it directly), and
`LayoutEditPanel` is the HUD card that edits it.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from . import layout_file as LF
from .theme import C

#: [m] a click this close outside a crate's outline still picks it: a 10 cm post is 4 px from a high camera.
PICK_MARGIN = 0.06
#: Undo steps kept.
UNDO_DEPTH = 200
#: Most pieces one layout may hold: what the 3D view can draw of one shape (`ViewportWidget.DYN_PROP_INSTANCES`).
MAX_PIECES = 96
#: What a new piece is when nothing has been chosen: the generator's middle wooden crate.
DEFAULT_SHAPE = ("wooden_crate", {"width": 0.30, "depth": 0.34, "height": 0.36})


def _inside(poly: np.ndarray, pt: np.ndarray, margin: float = 0.0) -> bool:
    """Is `pt` inside the convex polygon `poly` (k, 2), grown by `margin`? Either winding."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    e = b - a
    area = float(np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]))
    out = np.stack([e[:, 1], -e[:, 0]], 1) * (1.0 if area > 0 else -1.0)      # outward normals
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    dist = np.sum((pt[None, :] - a) * out, axis=1)
    return bool(np.all(dist <= margin))


class LayoutEditor(QtCore.QObject):
    """The layout being edited: pieces `{"sid", "x", "y", "yaw"}` in the session's catalogue, a selection, a mode."""

    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.catalogue: List[Dict] = []
        self._by_id: Dict[int, Dict] = {}
        self._fp: Dict[int, np.ndarray] = {}
        self.pieces: List[Dict] = []
        self.selected: Optional[int] = None
        #: "select" (pick and drag) or "add" (a click on the floor puts a piece there)
        self.mode = "select"
        self.add_sid: Optional[int] = None
        self.slots = 0
        self.seed_origin: Optional[int] = None
        self.centerline: Optional[np.ndarray] = None
        self._undo: List[Tuple[List[Dict], Optional[int]]] = []
        self.dirty = False

    # ------------------------------------------------------------------ set up
    def begin(self, pieces, catalogue, slots: int, centerline=None, seed: Optional[int] = None):
        self.catalogue = list(catalogue or [])
        self._by_id = {int(e["id"]): e for e in self.catalogue}
        self._fp = {int(e["id"]): np.asarray(e["footprint"], np.float64) for e in self.catalogue}
        self.pieces = [{"sid": int(p["sid"]), "x": float(p["x"]), "y": float(p["y"]), "yaw": float(p["yaw"])}
                       for p in (pieces or []) if int(p["sid"]) in self._by_id]
        self.slots = int(slots)
        self.seed_origin = seed
        self.centerline = None if centerline is None else np.asarray(centerline, np.float64).reshape(-1, 2)
        self.selected = None
        self.mode = "select"
        self._undo = []
        self.dirty = False
        if self.add_sid not in self._by_id:
            self.add_sid = self._default_sid()
        self.changed.emit()

    def _default_sid(self) -> Optional[int]:
        style, dims = DEFAULT_SHAPE
        try:
            return LF.match_shape(style, dims, self.catalogue)
        except ValueError:
            return int(self.catalogue[0]["id"]) if self.catalogue else None

    # ------------------------------------------------------------------ reading
    def shape(self, sid: int) -> Dict:
        return self._by_id[int(sid)]

    def label(self, sid: Optional[int]) -> str:
        return self._by_id[int(sid)]["label"] if sid is not None and int(sid) in self._by_id else "—"

    def footprint_world(self, i: int) -> np.ndarray:
        p = self.pieces[i]
        fp = self._fp[int(p["sid"])]
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        return np.stack([p["x"] + fp[:, 0] * c - fp[:, 1] * s, p["y"] + fp[:, 0] * s + fp[:, 1] * c], 1)

    def rows(self) -> np.ndarray:
        """(k, 4) [shape id, x, y, yaw] -- the layout as the 3D scene draws a frame's crates."""
        if not self.pieces:
            return np.zeros((0, 4), np.float32)
        return np.asarray([[p["sid"], p["x"], p["y"], p["yaw"]] for p in self.pieces], np.float32)

    def hit(self, ground_at: Callable[[float], Optional[Tuple[float, float]]]) -> Optional[int]:
        """The piece under the cursor. `ground_at(z)` is where the cursor's ray crosses height z: a crate is tested
        at its own upper half, which is what a perspective camera shows of it. The nearest centre wins a tie."""
        best, best_d = None, float("inf")
        for i, p in enumerate(self.pieces):
            h = float(self._by_id[int(p["sid"])].get("height", 0.3))
            g = ground_at(0.6 * h)
            if g is None:
                continue
            pt = np.asarray(g, np.float64)
            if _inside(self.footprint_world(i), pt, PICK_MARGIN):
                d = float(np.hypot(pt[0] - p["x"], pt[1] - p["y"]))
                if d < best_d:
                    best, best_d = i, d
        return best

    def default_yaw(self, x: float, y: float) -> float:
        """Square to the track where it is put, the way the generator sets its pieces down (width across the lane)."""
        cl = self.centerline
        if cl is None or len(cl) < 3:
            return 0.0
        j = int(np.argmin(np.sum((cl - np.array([x, y])) ** 2, axis=1)))
        t = cl[(j + 1) % len(cl)] - cl[j - 1]
        return math.atan2(t[1], t[0]) + 0.5 * math.pi

    @property
    def full(self) -> bool:
        return self.slots > 0 and len(self.pieces) >= self.slots

    # ------------------------------------------------------------------ editing
    def checkpoint(self):
        """Remember the layout as it is, for 되돌리기. Called before every change (a drag: once, at its start)."""
        self._undo.append(([dict(p) for p in self.pieces], self.selected))
        if len(self._undo) > UNDO_DEPTH:
            del self._undo[0]

    def _touched(self):
        self.dirty = True
        self.changed.emit()

    def select(self, i: Optional[int]):
        self.selected = i if (i is not None and 0 <= i < len(self.pieces)) else None
        self.changed.emit()

    def set_mode(self, mode: str):
        self.mode = "add" if mode == "add" else "select"
        self.changed.emit()

    def add(self, x: float, y: float, sid: Optional[int] = None) -> Optional[int]:
        sid = self.add_sid if sid is None else sid
        if sid is None or int(sid) not in self._by_id or len(self.pieces) >= MAX_PIECES:
            return None
        self.checkpoint()
        self.pieces.append({"sid": int(sid), "x": float(x), "y": float(y), "yaw": self.default_yaw(x, y)})
        self.selected = len(self.pieces) - 1
        self._touched()
        return self.selected

    def move(self, i: int, x: float, y: float):
        if 0 <= i < len(self.pieces):
            self.pieces[i]["x"], self.pieces[i]["y"] = float(x), float(y)
            self._touched()

    def remove(self, i: Optional[int] = None):
        i = self.selected if i is None else i
        if i is None or not 0 <= i < len(self.pieces):
            return
        self.checkpoint()
        del self.pieces[i]
        self.selected = None
        self._touched()

    def rotate(self, degrees: float, i: Optional[int] = None):
        i = self.selected if i is None else i
        if i is None or not 0 <= i < len(self.pieces):
            return
        self.checkpoint()
        yaw = self.pieces[i]["yaw"] + math.radians(degrees)
        self.pieces[i]["yaw"] = math.atan2(math.sin(yaw), math.cos(yaw))
        self._touched()

    def set_yaw(self, radians: float, i: Optional[int] = None):
        i = self.selected if i is None else i
        if i is None or not 0 <= i < len(self.pieces):
            return
        yaw = math.atan2(math.sin(radians), math.cos(radians))
        if abs(yaw - self.pieces[i]["yaw"]) < 1e-9:
            return
        self.checkpoint()
        self.pieces[i]["yaw"] = yaw
        self._touched()

    def set_shape(self, sid: int, i: Optional[int] = None):
        """The picked piece becomes `sid`; with nothing picked, `sid` is what "＋ 놓기" puts down next."""
        if int(sid) not in self._by_id:
            return
        i = self.selected if i is None else i
        if i is None or not 0 <= i < len(self.pieces):
            self.add_sid = int(sid)
            self.changed.emit()
            return
        if int(self.pieces[i]["sid"]) == int(sid):
            return
        self.checkpoint()
        self.pieces[i]["sid"] = int(sid)
        self.add_sid = int(sid)
        self._touched()

    def clear(self):
        if not self.pieces:
            return
        self.checkpoint()
        self.pieces = []
        self.selected = None
        self._touched()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.pieces, self.selected = self._undo.pop()
        self._touched()
        return True

    def replace(self, pieces: List[Dict]):
        """A loaded file: the whole layout at once, undoable."""
        self.checkpoint()
        self.pieces = [{"sid": int(p["sid"]), "x": float(p["x"]), "y": float(p["y"]), "yaw": float(p["yaw"])}
                       for p in pieces]
        self.selected = None
        self._touched()

    def portable(self) -> List[Dict]:
        return LF.to_portable(self.pieces, self.catalogue)


class LayoutEditPanel(QtWidgets.QFrame):
    """The HUD card of 장애물 편집. Emits what the window has to do; edits the `LayoutEditor` itself."""

    run_requested = QtCore.pyqtSignal()
    cancel_requested = QtCore.pyqtSignal()
    save_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    fit_requested = QtCore.pyqtSignal()

    HELP = ("• 장애물 클릭: 고르기 · 끌기: 옮기기\n"
            "• '＋ 놓기'를 켜고 바닥 클릭: 그 자리에 놓기\n"
            "• 휠: 확대·축소 · 오른쪽 끌기(빈 바닥 끌기): 화면 이동\n"
            "• Shift+휠 또는 Q / E: 돌리기 · Delete: 지우기 · Ctrl+Z: 되돌리기 · F: 전체 보기\n"
            "• 편집하는 동안 시뮬레이션은 멈춰 있습니다.")

    def __init__(self, editor: LayoutEditor, parent=None):
        super().__init__(parent)
        self.ed = editor
        self.setObjectName("Tile")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 14)
        v.setSpacing(8)
        cap = QtWidgets.QLabel("장애물 편집")
        cap.setObjectName("TileCaption")
        v.addWidget(cap)
        self.count = QtWidgets.QLabel("")
        self.count.setStyleSheet(f"color: {C['text.0']}; font-size: 17px; font-weight: 700;")
        v.addWidget(self.count)
        self.origin = QtWidgets.QLabel("")
        self.origin.setStyleSheet(f"color: {C['text.1']}; font-size: 12px;")
        self.origin.setWordWrap(True)
        v.addWidget(self.origin)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.btn_select = self._btn("선택·이동", "장애물을 클릭해 고르고 끌어서 옮깁니다.", checkable=True)
        self.btn_add = self._btn("＋ 놓기", "켜고 바닥을 클릭하면 아래 모양을 그 자리에 놓습니다.  (A)", checkable=True)
        self.btn_select.clicked.connect(lambda: self.ed.set_mode("select"))
        self.btn_add.clicked.connect(lambda: self.ed.set_mode("add" if self.ed.mode != "add" else "select"))
        row.addWidget(self.btn_select, 1)
        row.addWidget(self.btn_add, 1)
        v.addLayout(row)

        self.shape_caption = QtWidgets.QLabel("놓을 모양")
        self.shape_caption.setObjectName("TileCaption")
        v.addWidget(self.shape_caption)
        self.combo_shape = QtWidgets.QComboBox()
        self.combo_shape.setToolTip("고른 장애물이 있으면 그 장애물의 모양을 바꾸고, 없으면 '＋ 놓기'로 놓을 모양을 정합니다.\n"
                                    "모두 학습 때 생성기가 쓰는 모양(크기·무게 그대로)입니다.")
        self.combo_shape.activated.connect(self._on_shape)
        v.addWidget(self.combo_shape)

        yrow = QtWidgets.QHBoxLayout()
        yrow.setSpacing(6)
        ylab = QtWidgets.QLabel("각도")
        ylab.setObjectName("TileCaption")
        yrow.addWidget(ylab)
        self.spin_yaw = QtWidgets.QSpinBox()
        self.spin_yaw.setRange(-180, 180)
        self.spin_yaw.setWrapping(True)
        self.spin_yaw.setSuffix(" °")
        self.spin_yaw.setSingleStep(5)
        self.spin_yaw.setToolTip("고른 장애물이 놓인 방향 (지도 기준, 반시계 +).")
        self.spin_yaw.valueChanged.connect(self._on_yaw)
        yrow.addWidget(self.spin_yaw, 1)
        self.btn_ccw = self._btn("↺ 15°", "반시계로 15° 돌립니다.  (Q)")
        self.btn_cw = self._btn("↻ 15°", "시계로 15° 돌립니다.  (E)")
        self.btn_ccw.clicked.connect(lambda: self.ed.rotate(15.0))
        self.btn_cw.clicked.connect(lambda: self.ed.rotate(-15.0))
        yrow.addWidget(self.btn_ccw)
        yrow.addWidget(self.btn_cw)
        v.addLayout(yrow)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(6)
        self.btn_delete = self._btn("삭제", "고른 장애물을 지웁니다.  (Delete)")
        self.btn_undo = self._btn("되돌리기", "마지막 편집을 되돌립니다.  (Ctrl+Z)")
        self.btn_clear = self._btn("모두 지우기", "장애물을 모두 치웁니다 (되돌릴 수 있습니다).")
        self.btn_delete.clicked.connect(lambda: self.ed.remove())
        self.btn_undo.clicked.connect(self.ed.undo)
        self.btn_clear.clicked.connect(self.ed.clear)
        for b in (self.btn_delete, self.btn_undo, self.btn_clear):
            row2.addWidget(b, 1)
        v.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        row3.setSpacing(6)
        self.btn_save = self._btn("저장…", "이 배치를 JSON 파일로 저장합니다 (~/f1sim_layouts).")
        self.btn_load = self._btn("불러오기…", "저장해 둔 배치 파일을 불러옵니다.")
        self.btn_fit = self._btn("전체 보기", "확대·이동한 화면을 맵 전체로 되돌립니다.  (F)")
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.btn_load.clicked.connect(self.load_requested.emit)
        self.btn_fit.clicked.connect(self.fit_requested.emit)
        row3.addWidget(self.btn_save, 1)
        row3.addWidget(self.btn_load, 1)
        row3.addWidget(self.btn_fit, 1)
        v.addLayout(row3)

        self.note = QtWidgets.QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color: {C['warn']}; font-size: 12px;")
        self.note.setVisible(False)
        v.addWidget(self.note)
        help_ = QtWidgets.QLabel(self.HELP)
        help_.setWordWrap(True)
        help_.setStyleSheet(f"color: {C['text.2']}; font-size: 12px;")
        v.addWidget(help_)

        self.btn_run = QtWidgets.QPushButton("▶  이 배치로 달리기")
        self.btn_run.setObjectName("StartButton")
        self.btn_run.setFixedHeight(42)
        self.btn_run.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_run.setToolTip("편집한 배치를 시뮬레이터에 넣고 처음부터 다시 달립니다. 리셋해도 이 배치 그대로입니다.")
        self.btn_run.clicked.connect(self.run_requested.emit)
        v.addWidget(self.btn_run)
        self.btn_cancel = self._btn("편집 취소", "바꾼 것을 버리고 하던 주행으로 돌아갑니다.")
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        v.addWidget(self.btn_cancel)

        self.ed.changed.connect(self.refresh)
        self.refresh()

    @staticmethod
    def _btn(text: str, tip: str, checkable: bool = False) -> QtWidgets.QPushButton:
        b = QtWidgets.QPushButton(text)
        b.setObjectName("IconButton")
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setFixedHeight(34)
        b.setCursor(QtCore.Qt.PointingHandCursor)
        return b

    def set_catalogue(self, catalogue: List[Dict]):
        """The shape list, by style then size, the generator's own names (`layout_file.shape_label`)."""
        self.combo_shape.blockSignals(True)
        self.combo_shape.clear()
        order = list(LF.STYLE_LABEL)
        for e in sorted(catalogue, key=lambda e: (order.index(e["style"]) if e["style"] in order else 99,
                                                  e.get("label", ""))):
            mass = float(e.get("mass") or 0.0)
            self.combo_shape.addItem(f"{e['label']} · {mass:g} kg" if mass else e["label"], int(e["id"]))
        self.combo_shape.blockSignals(False)
        self.refresh()

    def set_note(self, text: str):
        """A line from the window (a file that did not load, the worker's refusal); cleared by passing ""."""
        self._ext_note = text or ""
        self.refresh()

    def _on_shape(self, idx: int):
        sid = self.combo_shape.itemData(idx)
        if sid is not None:
            self.ed.set_shape(int(sid))

    def _on_yaw(self, deg: int):
        if self.ed.selected is not None:
            self.ed.set_yaw(math.radians(float(deg)))

    def refresh(self):
        ed = self.ed
        n = len(ed.pieces)
        self.count.setText(f"장애물 {n}개" + (f"  ·  더 놓을 수 있는 칸 {max(0, ed.slots - n)}" if ed.slots else ""))
        self.origin.setText((f"시드 {ed.seed_origin} 에서 시작" if ed.seed_origin is not None else "직접 만든 배치")
                            + ("  ·  바뀜" if ed.dirty else ""))
        self.btn_select.setChecked(ed.mode == "select")
        self.btn_add.setChecked(ed.mode == "add")
        sel = ed.selected
        has = sel is not None
        self.shape_caption.setText("고른 장애물의 모양" if has else "놓을 모양")
        sid = ed.pieces[sel]["sid"] if has else ed.add_sid
        i = self.combo_shape.findData(int(sid)) if sid is not None else -1
        if i >= 0 and i != self.combo_shape.currentIndex():
            self.combo_shape.blockSignals(True)
            self.combo_shape.setCurrentIndex(i)
            self.combo_shape.blockSignals(False)
        self.spin_yaw.blockSignals(True)
        self.spin_yaw.setValue(int(round(math.degrees(ed.pieces[sel]["yaw"]))) if has else 0)
        self.spin_yaw.blockSignals(False)
        for w in (self.spin_yaw, self.btn_ccw, self.btn_cw, self.btn_delete):
            w.setEnabled(has)
        self.btn_undo.setEnabled(bool(ed._undo))
        self.btn_clear.setEnabled(n > 0)
        note = getattr(self, "_ext_note", "")
        if not note and ed.slots and n > ed.slots:
            note = (f"장애물이 이 세션의 칸({ed.slots})보다 많습니다 — '이 배치로 달리기'를 누르면 칸을 늘려 "
                    f"세션을 다시 시작합니다 (10~20초).")
        self.note.setText(note)
        self.note.setVisible(bool(note))
