"""The console's training page (11-24, the user: "콘솔에서도 내가 학습할 수 있게 따로 뭐 ≡ 버튼같은거 눌러서 메뉴 펼쳐서
선택하면 학습 패널로 퀄리나 지금 학습세팅들 좀 만질 수 있게").

Training runs on the two-GPU PC whichever PC the console is opened on: this page only writes request files
(`learn.train_requests`) and reads the status the training PC's queue writes back (tools/train_queue.py) -- and each
reported run's progress.jsonl, which the queue sends along. No torch, no subprocess here.

Three columns: the settings form (a preset, the name, the map, where it starts, the settings people change, the GPUs,
the exam, the other flags), the selected run (its curves and its exam rows per checkpoint), and the runs with the
requests' outcomes."""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from ...learn import train_requests as TR
from . import theme
from .theme import C, SP
from .training import LineChart
from .widgets import Card, label

GPU_ITEMS = (("both", "둘 다", "4070과 5060이 함께 한 policy를 학습"), ("0", "4070", "4070 혼자"),
             ("1", "5060", "5060 혼자 (화면 출력도 맡은 GPU)"))
EXAM_ITEMS = (("lab3", "연구실 PC에서 25 update마다"), ("none", "없음"))
STALE_S = 180.0


def world_label(fname: str) -> str:
    """plan_world_korea26scan_qual.json -> korea26scan_qual"""
    s = fname[:-5] if fname.endswith(".json") else fname
    return s[len("plan_world_"):] if s.startswith("plan_world_") else s


def read_progress(path: str, max_bytes: int = 8_000_000) -> List[dict]:
    rows: List[dict] = []
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - max_bytes))
            for line in f.read().decode(errors="replace").splitlines():
                if line.startswith("{"):
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rows


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _ago(t: float) -> str:
    s = max(0.0, time.time() - t)
    return f"{s:.0f}초 전" if s < 90 else f"{s / 60:.0f}분 전" if s < 5400 else f"{s / 3600:.1f}시간 전"


class GroupHeader(QtWidgets.QWidget):
    """names over spans of a table's columns, following the columns as they resize"""

    def __init__(self, table: QtWidgets.QTableWidget, groups, parent=None):
        super().__init__(parent)
        self.table, self.groups = table, groups
        self.setFixedHeight(18)
        hv = table.horizontalHeader()
        hv.sectionResized.connect(lambda *_: self.update())
        hv.geometriesChanged.connect(self.update)

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        hv = self.table.horizontalHeader()
        top = self.window()
        dx = self.table.viewport().mapTo(top, QtCore.QPoint(0, 0)).x() - self.mapTo(top, QtCore.QPoint(0, 0)).x()
        f = p.font()
        f.setPointSizeF(8.0)
        p.setFont(f)
        for text, a, b in self.groups:
            x0 = dx + hv.sectionViewportPosition(a)
            x1 = dx + hv.sectionViewportPosition(b) + hv.sectionSize(b)
            p.setPen(QtGui.QColor(C["line.strong"]))
            p.drawLine(x0 + 4, self.height() - 2, x1 - 4, self.height() - 2)
            p.setPen(QtGui.QColor(C["text.2"]))
            text = p.fontMetrics().elidedText(text, QtCore.Qt.ElideRight, max(0, x1 - x0 - 8))
            p.drawText(QtCore.QRectF(x0, 0, x1 - x0, self.height() - 3), QtCore.Qt.AlignCenter, text)


class PlanTrainPage(QtWidgets.QWidget):
    """학습 (the ≡ menu). `watch_requested(run)`: show that run on the driving page."""
    watch_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, runs_dir: Optional[str] = None):
        super().__init__(parent)
        self.setObjectName("Centre")
        self.runs_dir = runs_dir or TR.runs_dir()
        self.status: Dict = {}
        self._status_mtime = 0.0
        self._sel: str = ""
        self._prog_key = None
        self._name_auto = True
        self._filled = False
        self._sent: List[dict] = []           # requests written here, until the status answers them
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(5000)
        self._timer.timeout.connect(self.refresh)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(SP[2], SP[2], SP[2], SP[2])
        h.setSpacing(SP[2])
        h.addWidget(self._build_form())
        h.addWidget(self._build_centre(), 1)
        h.addWidget(self._build_side())

    # ================================================================ construction
    def _build_form(self) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget()
        col.setFixedWidth(340)
        cv = QtWidgets.QVBoxLayout(col)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(SP[1])
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("PanelScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)
        v.setContentsMargins(0, 0, SP[1], 0)
        v.setSpacing(SP[2])

        fc = Card("새 학습")
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(SP[2])
        form.setVerticalSpacing(6)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        def row(text: str, w: QtWidgets.QWidget, tip: str = ""):
            lb = label(text, "field")
            if tip:
                lb.setToolTip(tip)
                w.setToolTip(tip)
            w.setMinimumWidth(60)
            if isinstance(w, QtWidgets.QComboBox):        # the longest item must not set the column's width
                w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
                w.setMinimumContentsLength(6)
            form.addRow(lb, w)
            return w

        self.cb_preset = row("불러오기", QtWidgets.QComboBox(), "고르면 아래 칸이 그 설정으로 채워짐")
        self.cb_preset.activated.connect(lambda i: self._apply_preset(self.cb_preset.itemData(i)))
        self.ed_name = row("이름", QtWidgets.QLineEdit(), "소문자로 시작, 소문자·숫자·_ (학습 PC에 없는 이름)")
        self.ed_name.textEdited.connect(lambda _t: setattr(self, "_name_auto", False))
        self.cb_world = row("맵", QtWidgets.QComboBox(), "학습 세계 설정 (트랙·상자 배치·접촉 무게)")
        self.cb_init = row("출발", QtWidgets.QComboBox(), "처음부터, 또는 이 checkpoint에서 이어서")
        self.cb_init.setMaxVisibleItems(18)
        self.sp_end = row("끝 update", QtWidgets.QSpinBox())
        self.sp_end.setRange(1, 100000)
        self.sp_end.setSingleStep(25)
        self.sp_end.setValue(1500)
        self.knob_ed: Dict[str, QtWidgets.QLineEdit] = {}
        for flag, text, _kind, hint in TR.KNOBS:
            ed = QtWidgets.QLineEdit()
            ed.setPlaceholderText("기본")
            self.knob_ed[flag] = row(text, ed, f"{hint}\n{flag} · 비우면 기본")
        self.cb_gpu = row("GPU", QtWidgets.QComboBox(), "\n".join(f"{t}: {tip}" for _k, t, tip in GPU_ITEMS))
        for key, text, _tip in GPU_ITEMS:
            self.cb_gpu.addItem(text, key)
        self.cb_exam = row("시험", QtWidgets.QComboBox(), "checkpoint마다 Korea 2026 (bag 스캔) 시험 (exam3)")
        for key, text in EXAM_ITEMS:
            self.cb_exam.addItem(text, key)
        fc.add(form)
        lb_more = label("그 밖의 인자", "field")
        fc.add(lb_more)
        self.ed_more = QtWidgets.QPlainTextEdit()
        self.ed_more.setObjectName("FlagsText")
        self.ed_more.setStyleSheet(f"font-family: \"{theme.MONO_FONT}\"; font-size: 11px;")
        self.ed_more.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.ed_more.setFixedHeight(100)
        self.ed_more.setToolTip("plan_rl 인자 그대로 (--이름 값). 위 칸에 있는 인자는 위 칸이 이김")
        fc.add(self.ed_more)
        v.addWidget(fc)
        v.addStretch(1)
        scroll.setWidget(inner)
        cv.addWidget(scroll, 1)
        self.form_msg = label("", "hint.danger")
        self.form_msg.setVisible(False)
        cv.addWidget(self.form_msg)
        self.btn_start = QtWidgets.QPushButton("학습 시작")
        self.btn_start.setObjectName("PrimaryButton")
        self.btn_start.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start)
        cv.addWidget(self.btn_start)
        return col

    def _build_centre(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(SP[2])
        head = Card()
        hh = QtWidgets.QHBoxLayout()
        hh.setSpacing(SP[2])
        self.h_name = QtWidgets.QLabel("—")
        self.h_name.setObjectName("HeaderTitle")
        self.h_state = QtWidgets.QLabel("")
        self.h_nums = QtWidgets.QLabel("")
        self.h_nums.setObjectName("Mono")
        hh.addWidget(self.h_name)
        hh.addWidget(self.h_state)
        hh.addStretch(1)
        hh.addWidget(self.h_nums)
        head.add(hh)
        self.h_sub = label("", "hint")
        head.add(self.h_sub)
        v.addWidget(head)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(SP[1])
        self.ch_laps = LineChart("바퀴/분", "")
        self.ch_speed = LineChart("속도", "m/s")
        self.ch_contact = LineChart("접촉 /분", "", lower_is_better=True,
                                    colours=(C["contact.wall"], C["contact.prop"]), labels=("벽", "상자"))
        self.ch_std = LineChart("조향 탐색 크기", "", colours=(C["accent"], C["text.1"]), labels=("실제", "망"))
        self.ch_std.setToolTip("첫 걸음 조향을 흔드는 폭. 실제 = max(망이 내는 값, 조향 탐색 바닥)")
        for i, ch in enumerate((self.ch_laps, self.ch_speed, self.ch_contact, self.ch_std)):
            ch.setMinimumHeight(150)
            grid.addWidget(ch, i // 2, i % 2)
        v.addLayout(grid, 5)

        ec = Card("시험")
        unit = label("칸: 세게 닿은 % (통과 수)", "hint")
        unit.setToolTip("길 위 상자를 지날 때 세게 닿은 비율 [%], 괄호는 그런 통과의 수 (칸이 좁으면 칸 위에 마우스를)")
        ec.head_layout.addWidget(unit)
        self.exam_tbl = QtWidgets.QTableWidget(0, 11)
        heads = [("u", "checkpoint update"),
                 ("한 바퀴", "빈 트랙 세션의 한 바퀴 [s] (세션 중앙값)"),
                 ("벽", "상자 세션에서 세션당 벽에 세게 닿은 횟수"),
                 ("상자", "상자 세션에서 세션당 상자에 세게 닿은 횟수"),
                 ("0.5~0.9", "길 위 상자가 지나가기 0.5~0.9초 전에야 보인 경우: 세게 닿은 % (통과 수)"),
                 ("0.9~1.5", "길 위 상자가 지나가기 0.9~1.5초 전에 보인 경우: 세게 닿은 % (통과 수)"),
                 ("1.5~3", "길 위 상자가 지나가기 1.5~3초 전에 보인 경우: 세게 닿은 % (통과 수)"),
                 ("1번째", "같은 상자를 그 세션에서 처음 지날 때: 세게 닿은 % (통과 수)"),
                 ("2번째", "같은 상자 두 번째 통과: 세게 닿은 % (통과 수)"),
                 ("3번째", "같은 상자 세 번째 통과: 세게 닿은 % (통과 수)"),
                 ("4번째~", "같은 상자 네 번째 통과부터: 세게 닿은 % (통과 수)")]
        for i, (t, tip) in enumerate(heads):
            it = QtWidgets.QTableWidgetItem(t)
            it.setToolTip(tip)
            self.exam_tbl.setHorizontalHeaderItem(i, it)
        self.exam_tbl.verticalHeader().setVisible(False)
        self.exam_tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.exam_tbl.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.exam_tbl.setFocusPolicy(QtCore.Qt.NoFocus)
        self.exam_tbl.setShowGrid(False)
        self.exam_tbl.setAlternatingRowColors(False)
        # the font in the sheet: the application sheet's QWidget font wins over an item's own
        self.exam_tbl.setStyleSheet(f"QTableWidget {{ background: {C['bg.card']}; border: none;"
                                    f" font-family: \"{theme.MONO_FONT}\"; font-size: 11px; }}"
                                    f"QTableWidget::item {{ padding: 1px 1px; }}"
                                    f"QHeaderView::section {{ font-family: \"{theme.UI_FONT}\"; font-size: 12px; }}")
        hv = self.exam_tbl.horizontalHeader()
        hv.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for j, wd in enumerate((38, 50, 42, 42)):
            hv.setSectionResizeMode(j, QtWidgets.QHeaderView.Fixed)
            hv.resizeSection(j, wd)
        hv.setDefaultAlignment(QtCore.Qt.AlignCenter)
        self.exam_tbl.verticalHeader().setDefaultSectionSize(24)
        # the groups of columns, named once over their spans rather than in every header
        ec.add(GroupHeader(self.exam_tbl, (("상자 세션", 2, 3), ("보인 때 (지나가기 몇 초 전)", 4, 6), ("같은 상자 몇 번째 통과", 7, 10))))
        self._exam_rows: List[dict] = []
        hv.sectionResized.connect(lambda i, _o, _n: i == 4 and self._fill_exams(self._exam_rows))
        ec.add(self.exam_tbl)
        self.exam_empty = label("", "hint")
        ec.add(self.exam_empty)
        v.addWidget(ec, 4)
        return w

    def _build_side(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setFixedWidth(290)
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(SP[2])
        rc = Card("학습 목록")
        self.run_list = QtWidgets.QListWidget()
        self.run_list.setMinimumHeight(180)
        self.run_list.setStyleSheet(f"font-family: \"{theme.MONO_FONT}\"; font-size: 12px;")
        self.run_list.currentItemChanged.connect(self._on_run_row)
        rc.add(self.run_list)
        bh = QtWidgets.QHBoxLayout()
        bh.setSpacing(SP[1])
        self.btn_stop = QtWidgets.QPushButton("멈춤")
        self.btn_stop.setObjectName("DangerButton")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_watch = QtWidgets.QPushButton("주행 화면에서 보기")
        self.btn_watch.setObjectName("GhostButton")
        self.btn_watch.setToolTip("이 학습의 최신 checkpoint를 주행 화면에서 고름")
        self.btn_watch.clicked.connect(lambda: self._sel and self.watch_requested.emit(self._sel))
        bh.addWidget(self.btn_stop)
        bh.addWidget(self.btn_watch, 1)
        rc.add(bh)
        v.addWidget(rc, 3)
        qc = Card("요청")
        self.req_list = QtWidgets.QListWidget()
        self.req_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.req_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.req_list.setWordWrap(True)
        qc.add(self.req_list)
        v.addWidget(qc, 2)
        self.st_line = label("", "hint")
        v.addWidget(self.st_line)
        return w

    # ================================================================ activity
    def set_active(self, on: bool):
        if on:
            self.refresh(force=True)
            self._timer.start()
        else:
            self._timer.stop()

    def refresh(self, force: bool = False):
        p = TR.status_path(self.runs_dir)
        try:
            mt = os.path.getmtime(p)
        except OSError:
            mt = 0.0
        if force or mt != self._status_mtime:
            self._status_mtime = mt
            self.status = TR.read_status(self.runs_dir)
            self._apply_status()
        self._show_run(self._sel)
        self._show_requests()

    def _apply_status(self):
        st = self.status
        self._fill_presets(st.get("presets") or [])
        # the map and start lists, keeping what is chosen
        w0 = self.cb_world.currentData()
        self.cb_world.blockSignals(True)
        self.cb_world.clear()
        for f in st.get("worlds") or []:
            self.cb_world.addItem(world_label(f), f)
        self.cb_world.blockSignals(False)
        if w0:
            self._set_combo(self.cb_world, w0)
        i0 = self.cb_init.currentData()
        self.cb_init.clear()
        self.cb_init.addItem("처음부터", "")
        runs = st.get("runs") or {}
        for n in st.get("order") or sorted(runs):
            for u in reversed((runs.get(n) or {}).get("checkpoints") or []):
                self.cb_init.addItem(f"{n} u{u}", f"{n}/plan_u{u}.pt")
        if i0 is not None:
            self._set_combo(self.cb_init, i0)
        for flag, ed in self.knob_ed.items():
            d = (st.get("knob_defaults") or {}).get(flag)
            ed.setPlaceholderText(f"기본 {d}" if d not in (None, "") else "기본 (없음)")
        if self._name_auto and st.get("next"):
            self.ed_name.setText(st["next"])
        if not self._filled and st.get("presets"):
            self._apply_preset(st["presets"][0])
            self._filled = True
        # the runs list
        cur = self._sel
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for n in st.get("order") or sorted(runs):
            r = runs.get(n) or {}
            pg = r.get("progress") or {}
            u = pg.get("update")
            lpm = pg.get("laps_per_min")
            txt = f"{'●' if r.get('alive') else '○'} {n:<9} u{u if u is not None else '—':<5} " \
                  f"{(f'{lpm:.2f}' if isinstance(lpm, (int, float)) else '—'):>5} 바퀴/분"
            it = QtWidgets.QListWidgetItem(txt)
            it.setData(QtCore.Qt.UserRole, n)
            it.setForeground(QtGui.QColor(C["primary"] if r.get("alive") else C["text.1"]))
            it.setToolTip(f"{n} · {'학습 중' if r.get('alive') else '멈춤'} · {world_label(r.get('world') or '')}")
            self.run_list.addItem(it)
        self.run_list.blockSignals(False)
        names = [self.run_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.run_list.count())]
        if cur not in names:
            cur = names[0] if names else ""
        if cur:
            self.run_list.setCurrentRow(names.index(cur))
        self._sel = cur
        self._prog_key = None

    @staticmethod
    def _set_combo(cb: QtWidgets.QComboBox, data) -> bool:
        i = cb.findData(data)
        if i >= 0:
            cb.setCurrentIndex(i)
        return i >= 0

    # ================================================================ presets and the form
    def _fill_presets(self, presets: List[dict]):
        cur = self.cb_preset.currentText()
        self.cb_preset.clear()
        for p in presets:
            self.cb_preset.addItem(p.get("label", "?"), p)
            frm = p.get("from", "")
            self.cb_preset.setItemData(self.cb_preset.count() - 1, f"{frm}의 설정" + (
                f", {p['init'].split('/')[0]} u{TR.init_update(p['init'])}에서 이어서" if p.get("init") else ", 처음부터"),
                QtCore.Qt.ToolTipRole)
        if not presets:
            self.cb_preset.addItem("학습 PC 상태를 기다리는 중", None)
        i = self.cb_preset.findText(cur)
        self.cb_preset.setCurrentIndex(max(i, 0))

    def _apply_preset(self, p: dict):
        if not p:
            return
        self._set_combo(self.cb_world, p.get("world", ""))
        if not self._set_combo(self.cb_init, p.get("init") or ""):
            self.cb_init.setCurrentIndex(0)
        self.sp_end.setValue(int(p.get("updates") or 1500))
        flags = [(str(f), str(v)) for f, v in p.get("flags") or []]
        knobs = {f for f, *_ in TR.KNOBS}
        for f, ed in self.knob_ed.items():
            ed.setText(next((v for g, v in flags if g == f), ""))
        self.ed_more.setPlainText(TR.flags_text([(f, v) for f, v in flags if f not in knobs], "\n"))
        self._set_msg("")

    def _set_msg(self, text: str, kind: str = "danger"):
        self.form_msg.setObjectName({"danger": "HintDanger", "ok": "Hint"}[kind])
        self.form_msg.setText(text)
        self.form_msg.setVisible(bool(text))
        self.form_msg.style().unpolish(self.form_msg)
        self.form_msg.style().polish(self.form_msg)

    def build_request(self) -> dict:
        """the start request the form describes (ValueError with the reason when it is not one)"""
        more = TR.parse_flags_text(self.ed_more.toPlainText())
        knobs = []
        for f, ed in self.knob_ed.items():
            t = ed.text().strip()
            if t:
                knobs.append([f, t])
        req = {"action": "start", "name": self.ed_name.text().strip(), "world": self.cb_world.currentData() or "",
               "init": self.cb_init.currentData() or "", "updates": int(self.sp_end.value()),
               "gpu": self.cb_gpu.currentData() or "both", "exam": self.cb_exam.currentData() or "lab3",
               "flags": TR.merge_flags(more, knobs)}
        TR.check_request(req)
        if req["name"] in set(self.status.get("taken") or []):
            raise ValueError(f"이름 {req['name']}은 학습 PC에 이미 있음")
        return req

    def _on_start(self):
        try:
            req = self.build_request()
        except (ValueError, TypeError) as e:
            self._set_msg(str(e))
            return
        busy = [n for n in self.status.get("busy") or []]
        if busy:
            ans = QtWidgets.QMessageBox.question(self, "학습 시작", f"지금 도는 {', '.join(busy)}를 멈추고 시작할까요?",
                                                 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                 QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return
            req["replace"] = busy
        self._send(req)
        self._name_auto = True

    def _on_stop(self):
        n = self._sel
        if not n:
            return
        ans = QtWidgets.QMessageBox.question(self, "학습 멈춤", f"{n} 학습을 멈출까요?",
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                             QtWidgets.QMessageBox.No)
        if ans == QtWidgets.QMessageBox.Yes:
            self._send({"action": "stop", "name": n})

    def _send(self, req: dict):
        req = dict(req, at=time.time())
        try:
            TR.write_request(req, self.runs_dir)
        except (OSError, ValueError) as e:
            self._set_msg(f"요청을 쓰지 못함: {e}")
            return
        self._sent.append({"at": req["at"], "action": req["action"], "name": req["name"]})
        self._set_msg("")
        self._show_requests()

    # ================================================================ the selected run
    def _on_run_row(self, cur, _prev=None):
        if cur is None:
            return
        self._sel = cur.data(QtCore.Qt.UserRole) or ""
        self._prog_key = None
        self._show_run(self._sel)

    def _show_run(self, name: str):
        r = (self.status.get("runs") or {}).get(name) or {}
        alive = bool(r.get("alive"))
        self.btn_stop.setEnabled(alive)
        self.btn_watch.setEnabled(bool(name))
        if not name:
            self.h_name.setText("—")
            self.h_state.setText("")
            self.h_nums.setText("")
            self.h_sub.setText("")
            for ch in (self.ch_laps, self.ch_speed, self.ch_contact, self.ch_std):
                ch.set_series()
            self._fill_exams([])
            return
        self.h_name.setText(name)
        self.h_state.setText("학습 중" if alive else "멈춤")
        self.h_state.setStyleSheet(f"color: {C['primary'] if alive else C['text.2']}; font-weight: 600;")
        pg = r.get("progress") or {}
        u, lpm = pg.get("update"), pg.get("laps_per_min")
        end = r.get("updates")
        self.h_nums.setText(f"u{u if u is not None else '—'}{f' / {end}' if end else ''}   "
                            f"{(f'{lpm:.2f}' if isinstance(lpm, (int, float)) else '—')} 바퀴/분")
        init = r.get("init") or ""
        src = f"{init.split('/')[-2]} u{TR.init_update(init)}에서" if init.count("/") >= 1 else "처음부터"
        self.h_sub.setText(f"{world_label(r.get('world') or '')} · {src}")
        path = os.path.join(self.runs_dir, name, "progress.jsonl")
        try:
            key = (name, os.path.getmtime(path), r.get("steer_floor"))
        except OSError:
            key = (name, None, None)
        if key != self._prog_key:
            self._prog_key = key
            self._fill_charts(read_progress(path), r.get("steer_floor") or 0.0)
        self._fill_exams(r.get("exams") or [])

    def _fill_charts(self, rows: List[dict], floor: float):
        if not rows:
            for ch in (self.ch_laps, self.ch_speed, self.ch_contact, self.ch_std):
                ch.set_placeholder("기록 없음")
                ch.set_series()
            return
        first = rows[0].get("update")
        for ch in (self.ch_laps, self.ch_speed, self.ch_contact, self.ch_std):
            ch.x_first = int(first) if isinstance(first, int) else 1
        col = lambda k: [_num(r.get(k)) for r in rows]
        self.ch_laps.set_series(col("laps_per_min"))
        self.ch_speed.set_series(col("speed"))
        self.ch_contact.set_series(col("contacts_wall_per_min"), col("contacts_prop_per_min"))
        raw = [_num((r.get("std") or [None])[0]) for r in rows]
        if floor and floor > 0:
            self.ch_std.set_series([max(x, floor) if x == x else x for x in raw], raw)
        else:
            self.ch_std.set_series(raw)

    def _fill_exams(self, rows: List[dict]):
        self._exam_rows = rows
        t = self.exam_tbl
        t.setRowCount(0)
        self.exam_empty.setText("" if rows else "시험 결과 없음")
        self.exam_empty.setVisible(not rows)
        # the count only where the column has room for it (the tooltip has it either way)
        room = t.horizontalHeader().sectionSize(4) >= QtGui.QFontMetrics(t.font()).horizontalAdvance("8.9 (828)") + 10

        def pct(v):
            if not (isinstance(v, list) and len(v) == 2):
                return "—"
            return f"{v[1]:.1f} ({v[0]})" if room else f"{v[1]:.1f}"

        def tip(v):
            return f"통과 {v[0]}번 · 세게 닿음 {v[1]:.1f}%" if isinstance(v, list) and len(v) == 2 else ""

        def f2(v, nd=2):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"
        for row in sorted(rows, key=lambda r: -int(r.get("u", 0))):
            cells = [str(row.get("u", "")), f2(row.get("lap")), f2(row.get("box_walls")), f2(row.get("box_boxes")),
                     pct(row.get("seen_05")), pct(row.get("seen_09")), pct(row.get("seen_15")),
                     pct(row.get("pass1")), pct(row.get("pass2")), pct(row.get("pass3")), pct(row.get("pass4"))]
            tips = [""] * 4 + [tip(row.get(k)) for k in ("seen_05", "seen_09", "seen_15", "pass1", "pass2", "pass3", "pass4")]
            i = t.rowCount()
            t.insertRow(i)
            for j, c in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(c)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                if tips[j]:
                    it.setToolTip(tips[j])
                if j == 0:
                    it.setForeground(QtGui.QColor(C["text.1"]))
                t.setItem(i, j, it)

    # ================================================================ requests
    def _show_requests(self):
        done = list(self.status.get("done") or [])
        answered = [float(d.get("req_at") or 0) for d in done]
        # matched by the request's own stamp (the console's clock; the queue's runs on the training PC)
        self._sent = [s for s in self._sent if not any(abs(a - s["at"]) < 1e-3 for a in answered)
                      and time.time() - s["at"] < 1800]
        word = {"start": "시작", "stop": "멈춤"}
        self.req_list.clear()
        for s in reversed(self._sent):
            it = QtWidgets.QListWidgetItem(f"{time.strftime('%H:%M', time.localtime(s['at']))} {word.get(s['action'], '')} "
                                           f"{s['name']} · 보냄, 학습 PC 답 기다리는 중")
            it.setForeground(QtGui.QColor(C["warn"]))
            self.req_list.addItem(it)
        for d in reversed(done[-12:]):
            res = str(d.get("result", ""))
            good = res.startswith(("시작함", "멈춤"))
            it = QtWidgets.QListWidgetItem(f"{time.strftime('%H:%M', time.localtime(d.get('at', 0)))} "
                                           f"{word.get(d.get('action'), '')} {d.get('name', '')} · {res}")
            it.setForeground(QtGui.QColor(C["text.0"] if good else C["text.1"]))
            it.setToolTip(f"{d.get('from', '')}에서 · {res}")
            self.req_list.addItem(it)
        at = self.status.get("at")
        if not at:
            self.st_line.setText("학습 PC 상태 없음")
            self.st_line.setStyleSheet(f"color: {C['warn']};")
        else:
            stale = time.time() - float(at) > STALE_S
            self.st_line.setText(f"학습 PC({self.status.get('host', '')}) {'응답 없음 · ' if stale else ''}{_ago(float(at))}")
            self.st_line.setStyleSheet(f"color: {C['warn'] if stale else C['text.2']};")
