"""Who drives the ego car (the console's "내 차", 2026-10-17).

The user: "f1sim console 에서 다 해볼 수 있는 거지? 직접 눈으로 보면서 ... 상대방으로도 싹 다 선택 가능하게".
The policy run picked above still decides the session's sensors and observation contract, and this
panel can hand the ego car to a published planner (on its own path follower) or to a sensor-only
baseline (TinyLidarNet, End2Race, Follow-the-Gap, the Disparity Extender). The same drivers are
opponent kinds in the table under 고급 설정. With another driver in the seat our policy is not run
(`SimWorker._step_once`), and the view draws that driver's own picture (`viewer.driver_viz`).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

from ... import opponent_slots as osl
from .opponent_table import BASELINES_DIR, PACE_KNOB
from .widgets import label

#: Mirrors `learn.common.EGO_PLANNER_KINDS` / `EGO_EXTERNAL_KINDS` (the GUI process does not import
#: `learn`; a test keeps the two equal).
PLANNERS = ("raceline", "forzaeth", "forzaeth_pred", "lane_switch")
EXTERNALS = ("tinylidarnet", "end2race", "ftg", "disparity")
#: A planner's line: the session's training line, or a faster one (its global planner's ax/ay --
#: the pace knob the evaluation tuned; cached for ICCAS, Korea 2026 and ICRA 2022).
LINES = (("학습 라인 7/6.5/4", None), ("빠른 라인 8/7/5", [8.0, 7.0, 5.0]),
         ("빠른 라인 9/7.5/6", [9.0, 7.5, 6.0]), ("빠른 라인 10/8/7", [10.0, 8.0, 7.0]))


class EgoDriverPanel(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 0)
        v.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label("내 차", "body"))
        self.kind = QtWidgets.QComboBox()
        self.kind.addItem("정책 런 (위에서 고른 체크포인트)", "policy")
        for name in PLANNERS:
            k = osl.kind_of(name)
            self.kind.addItem(f"{k.label} · 위치 참값", name)
            self.kind.setItemData(self.kind.count() - 1, k.note + "\n자기 경로 추종기로 직접 조향·속도를 냅니다. "
                                  "위치와 상대 차 상태를 참값으로 받습니다 (모듈형 스택의 특권 조건).",
                                  QtCore.Qt.ToolTipRole)
        for name in EXTERNALS:
            k = osl.kind_of(name)
            self.kind.addItem(k.label, name)
            self.kind.setItemData(self.kind.count() - 1, k.note, QtCore.Qt.ToolTipRole)
        self.kind.setToolTip("내 차를 누가 모는지. 정책 런 말고 다른 것을 고르면 우리 정책은 돌리지 않고, "
                             "조향·속도는 여기서 고른 주행기가 냅니다. 화면에는 그 방법이 계산한 것"
                             "(경로·웨이포인트·목표점, 빈 틈, 출력 궤적)을 그립니다.")
        row.addWidget(self.kind, 1)
        v.addLayout(row)

        opt = QtWidgets.QHBoxLayout()
        self.weights = QtWidgets.QPushButton("가중치 파일…")
        self.weights.setObjectName("GhostButton")
        self.weights.setStyleSheet("text-align: left;")
        self.weights.clicked.connect(self._pick_weights)
        self._weights_path = ""
        self.pace = QtWidgets.QDoubleSpinBox()
        self.pace.setRange(0.0, 15.0); self.pace.setSingleStep(0.5); self.pace.setDecimals(1)
        self.pace.setSuffix(" m/s"); self.pace.setSpecialValueText("기본 속도")
        self.pace.setToolTip("Follow-the-Gap max_speed / Disparity Extender top_speed. 기본 = 대회 설정 6.0.")
        self.line = QtWidgets.QComboBox()
        for text, lim in LINES:
            self.line.addItem(text, lim)
        self.line.setToolTip("플래너가 달리는 최소시간 라인의 가속도 한계 (a_lat/a_acc/a_brake). 상대 차는 "
                             "학습 라인 그대로입니다. 빠른 라인은 캐시에 있는 맵에서만 바로 열립니다.")
        opt.addWidget(self.weights, 1)
        opt.addWidget(self.pace)
        opt.addWidget(self.line)
        v.addLayout(opt)
        self.note = label("", "hint")
        self.note.setWordWrap(True)
        v.addWidget(self.note)

        self.kind.currentIndexChanged.connect(self._on_kind)
        self.pace.valueChanged.connect(lambda _v: self.changed.emit())
        self.line.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self._on_kind()

    # ---------------------------------------------------------------- state
    def kind_name(self) -> str:
        return str(self.kind.currentData() or "policy")

    def _on_kind(self, *_a):
        k = self.kind_name()
        net = k in ("tinylidarnet", "end2race")
        self.weights.setVisible(net)
        self.pace.setVisible(k in PACE_KNOB)
        self.line.setVisible(k in PLANNERS)
        if not net:
            self._weights_path = ""
        self.weights.setText(os.path.basename(self._weights_path) or "가중치 파일…")
        self.weights.setToolTip(self._weights_path or f"공개 가중치 또는 이 환경에서 학습한 가중치 ({BASELINES_DIR})")
        self.note.setText("" if k == "policy" else
                          "우리 정책은 돌리지 않습니다 (위의 런은 센서·관측 구성에만 씁니다). 화면에는 이 "
                          "주행기가 계산한 것을 그립니다.")
        self.changed.emit()

    def _pick_weights(self):
        start = os.path.dirname(self._weights_path) or BASELINES_DIR
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "내 차 가중치", start,
                                                        "가중치 (*.pt *.pth);;모든 파일 (*)")
        if path:
            self._weights_path = str(path)
            self._on_kind()

    def problem(self) -> str:
        k = self.kind_name()
        if k in ("tinylidarnet", "end2race"):
            if not self._weights_path:
                return "내 차 주행기의 가중치 파일을 고르세요."
            if not os.path.isfile(self._weights_path):
                return f"내 차 가중치 파일이 없습니다: {self._weights_path}"
        return ""

    def config(self) -> Dict[str, Any]:
        """The SessionConfig fields (`ego_kind`, `ego_weights`, `ego_options`, `ego_limits`)."""
        k = self.kind_name()
        opts = {PACE_KNOB[k]: float(self.pace.value())} if k in PACE_KNOB and self.pace.value() > 0 else {}
        lim: Optional[List[float]] = self.line.currentData() if k in PLANNERS else None
        return {"ego_kind": k, "ego_weights": self._weights_path if k in ("tinylidarnet", "end2race") else "",
                "ego_options": opts, "ego_limits": (list(lim) if lim else None)}

    def set_config(self, d: Dict[str, Any]) -> None:
        k = str(d.get("ego_kind") or "policy")
        i = self.kind.findData(k)
        self._weights_path = str(d.get("ego_weights") or "")
        self.kind.setCurrentIndex(max(0, i))
        knob = PACE_KNOB.get(k)
        self.pace.setValue(float((d.get("ego_options") or {}).get(knob, 0.0)) if knob else 0.0)
        lim = d.get("ego_limits")
        j = next((n for n, (_t, l) in enumerate(LINES) if (l or None) == (list(lim) if lim else None)), 0)
        self.line.setCurrentIndex(j)
        self._on_kind()
