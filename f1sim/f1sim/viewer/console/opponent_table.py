"""The 상대차 table: one row per opponent car, shared by the 주행 page and the 학습 page.

The console had one combo, `상대차 주행 방식`, with two entries. That is the right control for a
two-car race and the wrong one for any other: it says the same thing about every other car at once,
and a race of three is interesting precisely when the two other cars are not the same car.

So this widget is a *table*: `race_size - 1` rows, one per grid slot, each editing an
`f1sim.opponent_slots.OpponentSlot`. It is one widget and not two because the driving page and the
training page must not be able to disagree about what a slot is -- the 주행 page builds a session out
of it and the 학습 page emits `--opp-slots` out of it, and a table that produced something the
trainer could not parse would be a second configuration language with no way to diff it against the
first. `slots()` and `set_slots()` are the whole interface, and what travels between them is the
spec, not widget state.

Three details worth stating, because each was a choice:

* **Columns, not a form per car.** A table is the thing that lets someone see at a glance that row 2
  is the only one with events; a stack of identical forms hides exactly that. The table scrolls
  horizontally in the narrow driving panel, which is the price.
* **The teacher is a selection, and a kind the tree does not have is listed and disabled.** The
  combo offers `raceline` (the teacher this viewer has always run) and `interactive`
  (`f1sim.interactive_teacher`, which scores its plans against where the other cars are predicted to
  be) side by side, with 기본 and 업그레이드 티처 as the two presets, so choosing the upgraded
  teacher is one click and choosing it is what it takes. A kind whose module is missing is still in
  the combo, greyed, with the reason in its tooltip: leaving it out would make "there is no such
  thing" and "it has not merged yet" look the same.
* **The checkpoint cell says what the file is.** Its controller arm and whether it carries memory
  decide whether it can drive at all, and the loader's refusal is shown here rather than at start.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from ... import opponent_events as opp_ev
from ... import opponent_slots as osl
from .theme import C, SP
from .widgets import label

#: Columns, in order: (key, header, width). The widths are what a 420 px panel can show two of at
#: once; the table scrolls for the rest rather than shrinking anything into illegibility.
COLUMNS = (
    ("car", "차량", 40),
    ("kind", "종류", 118),
    ("checkpoint", "체크포인트", 150),
    ("speed", "속도 배율", 128),
    ("grip", "그립 라벨", 104),
    ("cap", "속도 cap", 84),
    ("events", "이벤트", 168),
    ("rate", "빈도/10s", 74),
    ("reactive", "반응형 확률", 212),
    ("spawn", "스폰", 86),
)
COLUMN_KEYS = tuple(k for k, _, _ in COLUMNS)

#: One character per timed event, because four two-character labels do not fit a cell narrow
#: enough to keep the table readable in the 420 px driving panel. The full name is the tooltip.
EVENT_SHORT = {"brake": "제", "stop": "정", "shift": "차", "weave": "지"}

#: What each column means, on the header's tooltip. A header narrow enough to fit ten columns in a
#: 420 px panel cannot also be the explanation.
COLUMN_HELP = {
    "car": "그리드 슬롯 번호. 학습 차는 0번이고 표에 없습니다.",
    "kind": "이 차를 누가 모는지. 병합 전인 종류는 회색으로 남고 이유가 툴팁에 있습니다.",
    "checkpoint": "정책 체크포인트 경로. 툴팁에 제어기 arm 과 메모리 종류가 나옵니다.",
    "speed": "속도 배율 하한–상한. 같으면 고정, 다르면 리셋마다 그 범위에서 뽑습니다.",
    "grip": "티처의 속도 프로파일이 가정하는 마찰. 티처 종류에만 적용됩니다.",
    "cap": "이 차만의 속도 상한 [m/s]. 0 이면 세션 상한을 그대로 씁니다.",
    "events": "제동 / 정지 / 차선 변경 / 지그재그 — 이 차가 받을 수 있는 타이머 이벤트.",
    "rate": "이 차가 10초당 받는 타이머 이벤트 수.",
    "reactive": "레이스마다 이 차에 주어질 확률: defend / yield / line / oblivious 순서.",
    "spawn": "학습 차를 기준으로 이 차가 어디서 출발하는지 (앞 / 뒤 / 나란히 / 무작위).",
}
EVENT_LONG = {"brake": "제동 (brake)", "stop": "정지 (stop)",
              "shift": "차선 변경 (shift)", "weave": "지그재그 (weave)"}


#: `(path, mtime, size) -> (note, arm)`. The table asks about a checkpoint on every keystroke -- the
#: page rebuilds its preview from `slots()` -- and reading a 400 MB file's header each time is a GUI
#: thread doing file I/O while someone types. Keyed on the stat, so a file replaced under the picker
#: is read again, the same rule `training.checkpoint_sha` follows.
_CK_CACHE: dict = {}


def _checkpoint_facts(path: str):
    """`(note, arm)` for a checkpoint file. Cached; never instantiates the model.

    Read with `weights_only=True` and never instantiated: this runs on the GUI thread while someone
    is still typing, and building an actor to find out its name would freeze the window.
    """
    if not path:
        return "", "legacy"
    try:
        st = os.stat(path)
    except OSError:
        return f"파일이 없습니다: {path}", "legacy"
    key = (path, st.st_mtime, st.st_size)
    if key not in _CK_CACHE:
        if len(_CK_CACHE) > 64:
            _CK_CACHE.clear()
        _CK_CACHE[key] = _read_checkpoint_facts(path)
    return _CK_CACHE[key]


def describe_checkpoint(path: str) -> str:
    """One line about a checkpoint file, or the loader's own reason it cannot drive a car here."""
    return _checkpoint_facts(path)[0]


def checkpoint_arm(path: str) -> str:
    """The controller arm a checkpoint records, `legacy` when it records none or cannot be read."""
    return _checkpoint_facts(path)[1]


def _read_checkpoint_facts(path: str):
    if not os.path.isfile(path):
        return f"파일이 없습니다: {path}", "legacy"
    try:
        import torch

        from ...learn.model import controller_arm_of
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:                                  # unreadable, or torch missing
        return f"읽을 수 없습니다: {exc}", "legacy"
    meta = dict(ck.get("meta") or {})
    arm = controller_arm_of(ck)
    if "opp_token" in meta:
        # The oracle-planner arm: its actor takes the simulator's true opponent state as an input,
        # which the console cannot produce. `learn.model.load_checkpoint` says exactly this at start;
        # saying it here means it is said before the session is built rather than after.
        return ("특권 상대차(oracle) 체크포인트입니다 — 시뮬의 참 상대차 상태를 입력으로 받으므로 "
                "콘솔·내보내기·ROS 노드에서 주행할 수 없습니다. A0 대조군(opp_token 없음)을 쓰세요.", arm)
    if int(meta.get("cond_dim", 0)):
        return ("조건부(conditional) 체크포인트입니다 — 조건 입력 없이는 주행할 수 없습니다.", arm)
    mem = "메모리" if meta.get("memory") else (
        "스캔 채널" if (meta.get("scan_channels") or {}).get("channels") else "피드포워드")
    return f"제어기 {arm} · {mem} · act_dim {int(meta.get('act_dim', 2))}", arm


def _spin(lo: float, hi: float, value: float, step: float, decimals: int, width: int,
          suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    sp = QtWidgets.QDoubleSpinBox()
    sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(decimals)
    sp.setValue(value); sp.setFixedWidth(width)
    if suffix:
        sp.setSuffix(suffix)
    sp.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
    return sp


def _cell(*widgets, spacing: int = 2) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(2, 0, 2, 0)
    h.setSpacing(spacing)
    for x in widgets:
        h.addWidget(x)
    return w


class SlotRow(QtCore.QObject):
    """The widgets of one car, and the conversion both ways between them and an `OpponentSlot`."""
    changed = QtCore.pyqtSignal()

    def __init__(self, index: int, parent: "OpponentSlotTable"):
        super().__init__(parent)
        self.index = index
        self.table = parent
        self.car = label(str(index), "body")
        self.car.setAlignment(QtCore.Qt.AlignCenter)

        self.kind = QtWidgets.QComboBox()
        for k in osl.KINDS:
            # An unavailable kind says so in the row itself, not only in a tooltip: greyed text
            # alone reads as "not applicable here", which is a different thing from "not merged
            # yet". Dropping it entirely would make "no such kind" and "not here yet" identical.
            self.kind.addItem(k.label if k.available else f"{k.label} [병합 후 활성]", k.name)
            i = self.kind.count() - 1
            if k.available:
                self.kind.setItemData(i, k.note, QtCore.Qt.ToolTipRole)
            else:
                self.kind.setItemData(i, 0, QtCore.Qt.UserRole - 1)      # not selectable
                self.kind.setItemData(i, k.unavailable_message(), QtCore.Qt.ToolTipRole)
        self.kind.currentIndexChanged.connect(self._on_kind)

        self.ckpt = QtWidgets.QPushButton("—")
        self.ckpt.setObjectName("GhostButton")
        self.ckpt.setStyleSheet("text-align: left;")
        self.ckpt.clicked.connect(self._pick_checkpoint)
        self._ckpt_path = ""
        self._ckpt_arm = "legacy"

        self.lo = _spin(0.05, 3.0, 1.0, 0.05, 2, 58)
        self.hi = _spin(0.05, 3.0, 1.0, 0.05, 2, 58)
        self.lo.setToolTip("속도 배율 하한. 상한과 같으면 고정값입니다.")
        self.hi.setToolTip("속도 배율 상한. 리셋마다 [하한, 상한]에서 하나 뽑습니다.")
        # An inverted range is not a state the table can be left in: the spec refuses lo > hi, and a
        # cell that can hold a value `slots()` then raises on would make every reader of this widget
        # handle an exception for a typing mistake. Raising the upper spin's floor makes it
        # unreachable instead.
        self.lo.valueChanged.connect(self.hi.setMinimum)
        self.hi.setMinimum(self.lo.value())

        self.grip = QtWidgets.QComboBox()
        for g in osl.GRIP_LABELS:
            self.grip.addItem(osl.GRIP_LABEL_SHORT[g], g)
            self.grip.setItemData(self.grip.count() - 1, osl.GRIP_LABEL_TEXT[g],
                                  QtCore.Qt.ToolTipRole)
        self.grip.setToolTip("티처의 속도 프로파일이 어떤 마찰을 가정하는지. 참값은 특권 정보(privileged)라 "
                             "학생이 볼 수 없는 값에 라벨이 의존합니다.")

        self.cap = _spin(0.0, 12.0, 0.0, 0.5, 1, 74, " m/s")
        self.cap.setSpecialValueText("없음")
        self.cap.setToolTip("이 차만의 속도 상한. 0 = 세션 상한을 그대로 씁니다.")

        self.events = {}
        boxes = []
        for name in opp_ev.EVENT_NAMES:
            cb = QtWidgets.QCheckBox(EVENT_SHORT[name])
            cb.setToolTip(EVENT_LONG[name])
            cb.toggled.connect(self._emit)
            self.events[name] = cb
            boxes.append(cb)
        self.events_cell = _cell(*boxes)

        self.rate = _spin(0.0, 20.0, 0.0, 0.5, 1, 64)
        self.rate.setToolTip("이 차가 10초당 받는 타이머 이벤트 수. 이벤트를 골랐으면 0보다 커야 합니다.")

        self.reactive = {}
        rboxes = []
        for name in opp_ev.REACTIVE_NAMES:
            sp = _spin(0.0, 1.0, 0.0, 0.05, 2, 48)
            sp.setToolTip(f"{name}: 레이스마다 이 확률로 이 차에 주어집니다.")
            sp.valueChanged.connect(self._emit)
            self.reactive[name] = sp
            rboxes.append(sp)
        self.reactive_cell = _cell(*rboxes)

        self.spawn = QtWidgets.QComboBox()
        for sp_ in osl.SLOT_SPAWNS:
            self.spawn.addItem(osl.SPAWN_TEXT[sp_], sp_)
        self.spawn.setToolTip("학습 차를 기준으로 이 차가 어디서 출발하는지.")

        for w in (self.lo, self.hi, self.cap, self.rate):
            w.valueChanged.connect(self._emit)
        for w in (self.grip, self.spawn):
            w.currentIndexChanged.connect(self._emit)
        self._on_kind()

    # ---------------------------------------------------------------- cells
    def widgets(self) -> dict:
        return {"car": self.car, "kind": self.kind, "checkpoint": self.ckpt,
                "speed": _cell(self.lo, label("–", "hint"), self.hi),
                "grip": self.grip, "cap": self.cap, "events": self.events_cell,
                "rate": self.rate, "reactive": self.reactive_cell, "spawn": self.spawn}

    # ---------------------------------------------------------------- state
    def _emit(self, *_a):
        if not self.table._loading:
            self.changed.emit()

    def _on_kind(self, *_a):
        """Grey out what this kind cannot carry, rather than accepting it and ignoring it."""
        kind = osl.kind_of(self.kind.currentData() or "raceline")
        self.ckpt.setEnabled(kind.checkpoint)
        self.grip.setEnabled(kind.teacher)
        for cb in self.events.values():
            cb.setEnabled(kind.teacher)
        self.rate.setEnabled(kind.teacher)
        for sp in self.reactive.values():
            sp.setEnabled(kind.teacher)
        if not kind.checkpoint:
            self._ckpt_path, self._ckpt_arm = "", "legacy"
            self.ckpt.setText("—"); self.ckpt.setToolTip("")
        self._emit()

    def _pick_checkpoint(self):
        start = os.path.dirname(self._ckpt_path) or os.path.expanduser("~/f1sim_runs")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.table, "상대차 체크포인트", start, "체크포인트 (*.pt);;모든 파일 (*)")
        if path:
            self.set_checkpoint(path)
            self._emit()

    def set_checkpoint(self, path: str):
        self._ckpt_path = str(path or "")
        note = describe_checkpoint(self._ckpt_path)
        self._ckpt_arm = checkpoint_arm(self._ckpt_path)
        self.ckpt.setText(os.path.basename(self._ckpt_path) or "—")
        self.ckpt.setToolTip(f"{self._ckpt_path}\n{note}" if self._ckpt_path else "")
        bad = note.startswith(("특권", "조건부", "읽을 수 없", "파일이 없"))
        self.ckpt.setStyleSheet("text-align: left;" + (f" color: {C['danger']};" if bad else ""))

    def checkpoint_problem(self) -> str:
        """The loader's reason this file cannot drive a car here, or "" when it can."""
        kind = osl.kind_of(self.kind.currentData() or "raceline")
        if not kind.checkpoint:
            return ""
        if not self._ckpt_path:
            return f"차량 {self.index}: 정책 체크포인트를 골라야 합니다."
        note = describe_checkpoint(self._ckpt_path)
        if note.startswith(("특권", "조건부", "읽을 수 없", "파일이 없")):
            return f"차량 {self.index}: {note}"
        return ""

    def slot(self) -> osl.OpponentSlot:
        kind = osl.kind_of(self.kind.currentData() or "raceline")
        lo, hi = float(self.lo.value()), float(self.hi.value())
        d = {"kind": kind.name,
             "speed_scale": (lo if abs(hi - lo) < 1e-9 else [lo, min(hi, max(lo, hi))]),
             "spawn": str(self.spawn.currentData() or "ahead")}
        if kind.checkpoint and self._ckpt_path:
            d["checkpoint"] = self._ckpt_path
            d["controller"] = self._ckpt_arm
        if kind.teacher:
            d["label_grip"] = str(self.grip.currentData() or "true")
            ev = [n for n in opp_ev.EVENT_NAMES if self.events[n].isChecked()]
            if ev:
                d["events"] = ev
                d["event_rate"] = float(self.rate.value())
            react = {n: float(self.reactive[n].value()) for n in opp_ev.REACTIVE_NAMES
                     if self.reactive[n].value() > 0}
            if react:
                d["reactive"] = react
        if self.cap.value() > 0:
            d["speed_cap"] = float(self.cap.value())
        return osl.OpponentSlot.from_dict(d)

    def set_slot(self, s: osl.OpponentSlot):
        i = self.kind.findData(s.kind)
        if i >= 0:
            self.kind.setCurrentIndex(i)
        self.set_checkpoint(s.checkpoint or "")
        lo, hi = s.speed_scale
        self.lo.setValue(lo); self.hi.setValue(hi)
        g = self.grip.findData(s.label_grip)
        self.grip.setCurrentIndex(max(0, g))
        self.cap.setValue(0.0 if s.speed_cap is None else float(s.speed_cap))
        for n, cb in self.events.items():
            cb.setChecked(n in s.events)
        self.rate.setValue(float(s.event_rate))
        for n, sp in self.reactive.items():
            sp.setValue(float((s.reactive or {}).get(n, 0.0)))
        j = self.spawn.findData(s.spawn)
        self.spawn.setCurrentIndex(max(0, j))
        self._on_kind()


class OpponentSlotTable(QtWidgets.QWidget):
    """`race_size - 1` rows of `OpponentSlot`, with presets and a "copy row 1" button.

    `changed` fires whenever anything a slot carries moves, so a page can refresh its own preview
    without knowing which control it was.
    """
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._rows: List[SlotRow] = []
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(SP[0])

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(SP[0])
        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.addItem("프리셋 …", "")
        for key, title, _fn in osl.PRESETS:
            self.combo_preset.addItem(title, key)
        self.combo_preset.currentIndexChanged.connect(self._apply_preset)
        bar.addWidget(self.combo_preset, 1)
        self.btn_same = QtWidgets.QPushButton("동일하게")
        self.btn_same.setObjectName("GhostButton")
        self.btn_same.setToolTip("첫 줄의 설정을 모든 줄에 복사합니다.")
        self.btn_same.clicked.connect(self.copy_first_row)
        bar.addWidget(self.btn_same)
        v.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels([h for _, h, _ in COLUMNS])
        for i, key in enumerate(COLUMN_KEYS):
            item = self.table.horizontalHeaderItem(i)
            if item is not None and COLUMN_HELP.get(key):
                item.setToolTip(COLUMN_HELP[key])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(QtCore.Qt.NoFocus)
        self.table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        hh = self.table.horizontalHeader()
        for i, (_k, _h, w) in enumerate(COLUMNS):
            self.table.setColumnWidth(i, w)
            hh.setSectionResizeMode(i, QtWidgets.QHeaderView.Fixed)
        self.table.setMinimumHeight(96)
        v.addWidget(self.table)

        self.note = label("", "hint")
        v.addWidget(self.note)
        self.set_count(1)

    # ---------------------------------------------------------------- shape
    def count(self) -> int:
        return len(self._rows)

    def set_count(self, n: int):
        """Grow or shrink to `race_size - 1` rows, keeping what the surviving rows already say."""
        n = max(0, int(n))
        while len(self._rows) > n:
            self._rows.pop()
            self.table.removeRow(self.table.rowCount() - 1)
        while len(self._rows) < n:
            row = SlotRow(len(self._rows) + 1, self)
            row.changed.connect(self._on_changed)
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setRowHeight(r, 30)
            cells = row.widgets()
            for i, key in enumerate(COLUMN_KEYS):
                self.table.setCellWidget(r, i, cells[key])
            self._rows.append(row)
        # Header + rows + the frame, with room for a horizontal scrollbar: the columns are fixed
        # width and the driving panel is 420 px, so one is always there.
        chrome = self.table.horizontalHeader().height() + self.table.horizontalScrollBar().height() + 6
        self.table.setMaximumHeight(max(96, 30 * max(1, n) + chrome))
        self.table.setMinimumHeight(min(self.table.maximumHeight(), 96))
        self._on_changed()

    # ---------------------------------------------------------------- state
    def slots(self) -> List[osl.OpponentSlot]:
        return [r.slot() for r in self._rows]

    def slot_dicts(self) -> List[dict]:
        return [s.to_dict() for s in self.slots()]

    def json(self) -> str:
        return osl.slots_json(self.slots())

    def set_slots(self, slots: Optional[Sequence]):
        """Load a table. Accepts `OpponentSlot`s, plain dicts, or a JSON string; None clears it."""
        parsed = osl.parse_slots(slots) if slots else ()
        self._loading = True
        try:
            self.set_count(len(parsed))
            for row, s in zip(self._rows, parsed):
                row.set_slot(s)
        finally:
            self._loading = False
        self._on_changed()

    def copy_first_row(self):
        if not self._rows:
            return
        first = self._rows[0].slot()
        self._loading = True
        try:
            for row in self._rows[1:]:
                row.set_slot(first)
        finally:
            self._loading = False
        self._on_changed()

    def _apply_preset(self, _idx):
        key = self.combo_preset.currentData()
        if not key:
            return
        cks = [r._ckpt_path for r in self._rows if r._ckpt_path]
        try:
            slots = osl.preset(str(key), len(self._rows), cks)
        except ValueError:
            return
        if any(s.checkpoint is None and osl.kind_of(s.kind).checkpoint for s in slots):
            # The recipe preset wants a checkpoint and none has been picked: fall back to the
            # entries it can build rather than writing a row that cannot start.
            slots = tuple(s if not osl.kind_of(s.kind).checkpoint
                          else osl.OpponentSlot.from_dict({"kind": "self",
                                                           "speed_scale": list(s.speed_scale)})
                          for s in slots)
        self.set_slots(slots)
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentIndex(0)
        self.combo_preset.blockSignals(False)

    # ---------------------------------------------------------------- validity
    def problem(self, race_size: Optional[int] = None) -> str:
        """The first reason this table cannot start a session, or "" when it can."""
        for row in self._rows:
            p = row.checkpoint_problem()
            if p:
                return p
        try:
            slots = self.slots()
        except ValueError as exc:
            return str(exc)
        try:
            osl.validate_slots(slots, (race_size if race_size is not None else len(slots) + 1),
                               require_files=True)
        except ValueError as exc:
            return str(exc)
        return ""

    def summary(self) -> str:
        try:
            return osl.mix_summary(self.slots())
        except ValueError:
            return ""

    def _on_changed(self, *_a):
        if self._loading:
            return
        problem = self.problem()
        self.note.setObjectName("HintWarn" if problem else "Hint")
        self.note.setText(problem or (osl.describe(self.slots()) if self._rows
                                      else "레이스당 차량 수가 1이라 상대차가 없습니다."))
        self.note.style().unpolish(self.note)
        self.note.style().polish(self.note)
        self.changed.emit()
