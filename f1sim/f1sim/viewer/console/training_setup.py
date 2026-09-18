"""Torch-free, schema-backed training configuration and stage-aware resume UI."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets, sip

from . import catalog, theme
from .opponent_table import OpponentSlotTable
from .widgets import Card, Collapsible, label


MODES = (
    ("ppo", "PPO · 정책 강화학습", "ppo"),
    ("speed", "자동 제어 · 속도 헤드 적응", "ppo"),
    ("full", "자동 제어 · 전체 정책 적응", "ppo"),
    ("dagger", "DAgger · Teacher 증류", "dagger"),
    ("grip_collect", "마찰 추정 · 주행 데이터 수집", "grip_collect"),
    ("grip_fit", "마찰 추정 · 데이터로 학습", "grip_fit"),
    ("grip_final", "마찰 추정 · 최종 평가", "grip_final"),
    ("grip_pilot", "마찰 추정 · 제한된 파일럿", "grip_pilot"),
)

MODE_NOTES = {
    "ppo": "보상, 환경, 모델과 학습 조건을 직접 설정합니다. 기본 레시피는 현재 에셋 장애물을 사용하는 시작점입니다.",
    "speed": "조향 경로를 고정하고 속도 출력만 학습합니다. 자동 제어기와 원본 기준 정책을 함께 지정하세요.",
    "full": "전체 정책을 학습하며 원본 정책의 순환 상태로 계산한 곡률 KL을 사용합니다.",
    "dagger": "Teacher 모드, 혼합 비율, 수집·증류·평가 일정을 한곳에서 설정합니다.",
    "grip_collect": "학습 맵을 네 분할로 나눠 주행 데이터를 수집합니다. 같은 설정·출력 폴더로 중단 지점부터 재개합니다.",
    "grip_fit": "고정된 epoch 수로 추정기를 학습합니다. 기존 추정기에서 시작하고 TRAIN 재생 데이터를 섞을 수 있습니다. 새 후보마다 비어 있는 출력 폴더를 사용합니다.",
    "grip_final": "후보의 최종 분할을 한 번 평가합니다. 평가 기록은 다시 덮어쓰지 않습니다.",
    "grip_pilot": "제한된 합성 수집·학습·평가 실험입니다. 결과는 자동으로 주행 모델에 승격되지 않습니다.",
}


class IntegerEditor(QtWidgets.QLineEdit):
    """An integer control without QSpinBox's silent signed-32-bit clamp."""
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"[+-]?\d+"), self))
        self.textChanged.connect(lambda _text: self.valueChanged.emit(self.value()))

    def setValue(self, value):
        self.setText(str(int(value)))

    def value(self):
        try:
            return int(self.text())
        except ValueError:
            return self.text()

    def stepBy(self, steps):
        value = self.value()
        self.setValue((value if isinstance(value, int) else 0) + steps)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            self.stepBy(1 if event.key() == QtCore.Qt.Key_Up else -1)
        else:
            super().keyPressEvent(event)


class CheckpointInspector(QtCore.QObject):
    """Read weights-only CPU metadata without blocking or importing Torch in Qt."""
    ready = QtCore.pyqtSignal(str, dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self._closing = False
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        if parent is not None:
            parent.destroyed.connect(self.shutdown)
        application = QtCore.QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.shutdown)

    @property
    def busy(self):
        return self.process is not None

    def inspect(self, path):
        self._closing = False
        if self.process:
            self.process.kill()
        process = QtCore.QProcess(self)
        self.process = process
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        for key, value in {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}.items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        from .training import REPO_F1SIM
        process.setWorkingDirectory(REPO_F1SIM)
        process.finished.connect(lambda code, _status: self._finished(process, str(path), code))
        process.errorOccurred.connect(lambda error: self._start_failed(process, error))
        self.timer.start(30_000)
        process.start(sys.executable, ["-m", "f1sim.viewer.console.checkpoint_metadata", str(path)])

    def _start_failed(self, process, error):
        if self._closing or sip.isdeleted(self.timer):
            return
        if process is self.process and error == QtCore.QProcess.FailedToStart:
            self.timer.stop()
            self.process = None
            self.failed.emit(process.errorString())
            process.deleteLater()

    def _timeout(self):
        if self._closing:
            return
        if self.process:
            process, self.process = self.process, None
            process.kill()
            self.failed.emit("체크포인트 정보 확인 시간이 초과되었습니다.")

    def _finished(self, process, path, code):
        if self._closing or sip.isdeleted(self.timer) or sip.isdeleted(process):
            return
        if process is not self.process:
            process.deleteLater()
            return
        self.timer.stop()
        self.process = None
        try:
            if code:
                raise ValueError(bytes(process.readAllStandardError()).decode("utf-8", "replace")[-2000:])
            raw = bytes(process.readAllStandardOutput())
            if len(raw) > 131072:
                raise ValueError("체크포인트 정보가 너무 큽니다.")
            result = json.loads(raw)
            if not isinstance(result, dict) or result.get("format") != "f1sim-checkpoint-metadata-v1":
                raise ValueError("체크포인트 정보 형식이 올바르지 않습니다.")
            self.ready.emit(path, result)
        except (ValueError, TypeError) as exc:
            self.failed.emit(str(exc))
        finally:
            if not sip.isdeleted(process):
                process.deleteLater()

    def shutdown(self, *_args):
        """Stop only this inspector's child before Qt deletes its signal receivers."""
        self._closing = True
        if not sip.isdeleted(self.timer):
            self.timer.stop()
        process, self.process = self.process, None
        if process is None or sip.isdeleted(process):
            return
        # Disconnect our lambdas before kill can deliver finished/error during parent teardown.
        for signal in (process.finished, process.errorOccurred):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass
        if process.state() != QtCore.QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1000)
        process.deleteLater()

    def event(self, event):
        if event.type() == QtCore.QEvent.DeferredDelete:
            self.shutdown()
        return super().event(event)


class ValueEditor(QtWidgets.QWidget):
    """One real typed control per parser field; no free-form command line escape hatch."""
    changed = QtCore.pyqtSignal()

    def __init__(self, field, parent=None):
        super().__init__(parent)
        self.field = field
        self.kind = field["kind"]
        self.control = None
        self.enabled_box = None
        self.items = []
        vector_length = field.get("ui_length") or (field.get("nargs") if isinstance(field.get("nargs"), int) else 0)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        optional_number = ((field.get("default") is None and self.kind in ("int", "float", "list"))
                           or (field.get("ui_length") and field.get("default") == "")) and not field.get("required")
        if optional_number:
            self.enabled_box = QtWidgets.QCheckBox("직접 지정" if field.get("ui_length") else "지정")
            self.enabled_box.setToolTip("끄면 학습기의 기본값을 사용합니다.")
            row.addWidget(self.enabled_box)
            self.enabled_box.toggled.connect(self._enabled)
        if self.kind == "bool":
            self.control = QtWidgets.QCheckBox("사용")
            self.control.toggled.connect(self.changed)
        elif field.get("choices"):
            self.control = QtWidgets.QComboBox()
            if field.get("default") is None and not field.get("required"):
                self.control.addItem("자동 / 지정 안 함", None)
            for choice in field["choices"]:
                self.control.addItem(str(choice), choice)
            self.control.currentIndexChanged.connect(self.changed)
        elif field["dest"] == "device":
            self.control = QtWidgets.QComboBox()
            self.control.setEditable(True)
            self.control.addItem("cuda", "cuda")
            self.control.addItem("cpu", "cpu")
            self.control.currentTextChanged.connect(self.changed)
        elif self.kind == "int":
            self.control = IntegerEditor()
            self.control.valueChanged.connect(self.changed)
        elif self.kind == "list" and vector_length:
            self.control = QtWidgets.QWidget()
            cells = QtWidgets.QHBoxLayout(self.control)
            cells.setContentsMargins(0, 0, 0, 0)
            labels = field.get("component_labels") or []
            for i in range(vector_length):
                sub = dict(field, kind=field.get("item_kind", "str"), nargs=None, ui_length=None, default="", required=True)
                item = ValueEditor(sub)
                item.changed.connect(self.changed)
                if i < len(labels):
                    column = QtWidgets.QVBoxLayout()
                    column.addWidget(label(labels[i], "field"))
                    column.addWidget(item)
                    cells.addLayout(column, 1)
                else:
                    cells.addWidget(item)
                self.items.append(item)
        else:
            self.control = QtWidgets.QLineEdit()
            self.control.setObjectName("SearchBox")
            if self.kind == "float":
                validator = QtGui.QDoubleValidator(self.control)
                validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
                validator.setLocale(QtCore.QLocale.c())
                self.control.setValidator(validator)
                self.control.setPlaceholderText("예: 0.00005 또는 5e-5")
            elif self.kind == "list":
                self.control.setPlaceholderText("값을 쉼표로 구분")
            self.control.textChanged.connect(self.changed)
        self.control.setMinimumWidth(0)
        self.control.setAccessibleName(field.get("label", field["dest"]))
        self.control.setToolTip(field.get("help", ""))
        row.addWidget(self.control, 1)
        if self.kind == "path" or field.get("path_mode"):
            self.browse = QtWidgets.QPushButton("찾기…")
            self.browse.clicked.connect(self._browse)
            row.addWidget(self.browse)
        elif field["dest"] == "opp_pool":
            self.browse = QtWidgets.QPushButton("정책 파일 추가…")
            self.browse.clicked.connect(self._browse_pool)
            row.addWidget(self.browse)
        self.set_value(field.get("default"))

    def _enabled(self, on):
        self.control.setEnabled(on)
        self.changed.emit()

    def _browse(self):
        mode = self.field.get("path_mode", "input_file")
        current = self.control.text()
        if mode.endswith("dir"):
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "폴더 선택", current)
        elif mode == "output_file":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "저장 위치 선택", current)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "파일 선택", current)
        if path:
            self.control.setText(path)

    def _browse_pool(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "상대 정책 풀에 추가", catalog.RUNS_DIR, "정책 (*.pt)")
        if files:
            self.set_value(list(self.value() or []) + files)

    def value(self):
        if self.enabled_box and not self.enabled_box.isChecked():
            return None
        if self.items:
            return [item.value() for item in self.items]
        if isinstance(self.control, QtWidgets.QCheckBox):
            return self.control.isChecked()
        if isinstance(self.control, QtWidgets.QComboBox):
            if self.control.isEditable():
                return self.control.currentText().strip()
            return self.control.currentData()
        if isinstance(self.control, IntegerEditor):
            return self.control.value()
        text = self.control.text().strip()
        if not text:
            return None if self.field.get("default") is None else ""
        if self.kind == "float":
            try:
                return float(text)
            except ValueError:
                return text             # validation gives the field name, rather than raising in a signal
        if self.kind == "list":
            return [v.strip() for v in text.split(",") if v.strip()]
        return text

    def set_value(self, value):
        if self.enabled_box:
            enabled = value is not None and value != "" and value != []
            self.enabled_box.setChecked(enabled)
            self.control.setEnabled(enabled)
        if self.items:
            values = value if isinstance(value, (list, tuple)) else str(value or "").replace(",", " ").split()
            if not values:
                values = self.field.get("component_defaults") or []
            for i, item in enumerate(self.items):
                item.set_value(values[i] if i < len(values) else "")
        elif isinstance(self.control, QtWidgets.QCheckBox):
            self.control.setChecked(bool(value))
        elif isinstance(self.control, QtWidgets.QComboBox):
            index = self.control.findData(value)
            if index >= 0:
                self.control.setCurrentIndex(index)
            elif self.control.isEditable():
                self.control.setCurrentText(str(value or ""))
            else:
                raise ValueError(f"{self.field['label']}: 지원하지 않는 선택값 {value!r}")
        elif isinstance(self.control, IntegerEditor):
            self.control.setValue(int(value or 0))
        else:
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(x) for x in value)
            self.control.setText("" if value is None else str(value))


class TrainingSetupForm(QtWidgets.QWidget):
    launch_requested = QtCore.pyqtSignal(str, list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        from . import training_schema as schema
        self.schema_api = schema
        self.mode = ""
        self._cache = {}
        self._resume_modes = set()
        self._metadata_cache = {}
        self._inspection_mode = None
        self._pending_launch = False
        self.inspector = CheckpointInspector(self)
        self.inspector.ready.connect(self._checkpoint_ready)
        self.inspector.failed.connect(self._checkpoint_failed)
        self._quiet = True
        self._runs = []
        self._resuming = False
        self.fields = {}
        self.rows = {}
        self.groups = {}
        self.editors = {}
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        top = Card("학습 설정")
        bar = QtWidgets.QHBoxLayout()
        self.combo_mode = QtWidgets.QComboBox()
        for mode, title, _kind in MODES:
            self.combo_mode.addItem(title, mode)
        bar.addWidget(self.combo_mode, 2)
        self.combo_recipe = QtWidgets.QComboBox()
        self.combo_recipe.setMinimumWidth(0)
        bar.addWidget(self.combo_recipe, 2)
        for title, handler in (("설정 저장", self._save_dialog), ("설정 열기", self._load_dialog)):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(handler)
            bar.addWidget(button)
        top.add(bar)
        self.recipe_note = label("", "hint")
        self.recipe_note.setWordWrap(True)
        top.add(self.recipe_note)
        root.addWidget(top)

        body = QtWidgets.QHBoxLayout()
        nav = QtWidgets.QVBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("설정 검색 · 보상, teacher…")
        self.search.setClearButtonEnabled(True)
        self.search.setObjectName("SearchBox")
        nav.addWidget(self.search)
        self.sections = QtWidgets.QListWidget()
        self.sections.setMinimumWidth(160)
        self.sections.setMaximumWidth(220)
        nav.addWidget(self.sections, 1)
        self.coverage_note = label("", "hint")
        self.coverage_note.setWordWrap(True)
        nav.addWidget(self.coverage_note)
        body.addLayout(nav)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.setMinimumWidth(0)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)

        preview_fold = Collapsible("실행 내용 확인", expanded=False)
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(95)
        self.preview.setStyleSheet(f"font-family: '{theme.MONO_FONT}'; font-size: 10px;")
        preview_fold.add(self.preview)
        root.addWidget(preview_fold)
        foot = QtWidgets.QHBoxLayout()
        self.launch_note = label("", "hint")
        self.launch_note.setWordWrap(True)
        foot.addWidget(self.launch_note, 1)
        self.btn_resume_file = QtWidgets.QPushButton("체크포인트에서 이어서…")
        self.btn_resume_file.clicked.connect(self._resume_dialog)
        foot.addWidget(self.btn_resume_file)
        self.btn_launch = QtWidgets.QPushButton("학습 시작")
        self.btn_launch.setObjectName("PrimaryButton")
        self.btn_launch.clicked.connect(self._launch)
        foot.addWidget(self.btn_launch)
        root.addLayout(foot)
        self.sections.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.search.textChanged.connect(self._filter)
        self.combo_mode.currentIndexChanged.connect(self._mode_changed)
        self.combo_recipe.currentIndexChanged.connect(self._apply_recipe)
        self._quiet = False
        self._mode_changed()

    @property
    def kind(self):
        return next(kind for mode, _title, kind in MODES if mode == self.mode)

    def closeEvent(self, event):
        self.inspector.shutdown()
        super().closeEvent(event)

    def set_mode(self, mode):
        index = self.combo_mode.findData(mode)
        if index < 0:
            raise ValueError(f"Unknown training mode: {mode}")
        self.combo_mode.setCurrentIndex(index)

    def _mode_changed(self, *_args):
        if self._quiet:
            return
        if self.mode:
            self._cache[self.mode] = self.values()
            if self._resuming:
                self._resume_modes.add(self.mode)
            else:
                self._resume_modes.discard(self.mode)
        self.mode = self.combo_mode.currentData()
        self._quiet = True
        self._resuming = self.mode in self._resume_modes
        self.schema = self.schema_api.get_schema(self.kind)
        self.fields = {f["dest"]: f for f in self.schema["fields"]}
        self.editors, self.rows, self.groups = {}, {}, {}
        self.sections.clear()
        while self.stack.count():
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()
        group_order = list(dict.fromkeys(f.get("group", "기타") for f in self.schema["fields"]))
        for group in group_order:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(12, 4, 12, 12)
            layout.setSpacing(10)
            title = label(group, "section")
            layout.addWidget(title)
            layout.addStretch(1)
            scroll = QtWidgets.QScrollArea()
            scroll.setObjectName("PanelScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
            self.sections.addItem(group)
            self.groups[group] = (page, layout, [])
        self.tracks = None
        self.opp_table = None
        for field in self.schema["fields"]:
            dest = field["dest"]
            if dest == "tracks":
                from .training import TrackPicker
                self.tracks = TrackPicker()
                self.tracks.changed.connect(self._refresh_preview)
                self._add_row(field, self.tracks)
                continue
            if dest == "obstacle_draws" and "tracks" in self.fields:
                continue
            if dest == "opp_slots":
                box = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(box)
                layout.setContentsMargins(0, 0, 0, 0)
                self.chk_slots = QtWidgets.QCheckBox("상대 차량마다 종류·속도·스폰·이벤트 설정")
                self.opp_table = OpponentSlotTable()
                self.opp_table.changed.connect(self._refresh_preview)
                layout.addWidget(self.chk_slots)
                layout.addWidget(self.opp_table)
                self.chk_slots.toggled.connect(self._slots_toggled)
                self._add_row(field, box)
                continue
            editor = ValueEditor(field)
            editor.changed.connect(self._refresh_preview)
            self.editors[dest] = editor
            self._add_row(field, editor)
        self.combo_recipe.blockSignals(True)
        self.combo_recipe.clear()
        self.combo_recipe.addItem("기본 설정 / 사용자 정의", "custom")
        if self.mode == "ppo":
            from .training import RECIPES
            for recipe in RECIPES:
                if recipe.key != "custom":
                    self.combo_recipe.addItem(recipe.title, recipe.key)
        self.combo_recipe.blockSignals(False)
        self.combo_recipe.setVisible(self.mode == "ppo")
        values = self._cache.get(self.mode)
        if values is None:
            values = {f["dest"]: copy.deepcopy(f.get("default")) for f in self.schema["fields"]}
            if self.mode in ("speed", "full"):
                values.update(controller="auto", adaptation=self.mode, kl_scope="curvature", memory="gru",
                              action_mode="plan", cond="none", opp_token="off",
                              fresh_opt=True, scan_stack=6, scan_stride=1, hist_len=20,
                              envs=48, race_size=3, horizon=32,
                              total=(256 if self.mode == "speed" else 512) * 32 * 16)
            if self.mode.startswith("grip"):
                values["out"] = os.path.join(catalog.RUNS_DIR, f"{self.mode}_{time.strftime('%m%d_%H%M%S')}"
                                             + (".json" if self.mode == "grip_final" else ""))
            if self.mode == "grip_fit":
                values.update(seed=701, epochs=10)
        self.set_values(values)
        if self.tracks and self.mode not in self._cache and values.get("tracks") in ("train", "heldout", "eval"):
            self.tracks.select_split("train" if values["tracks"] == "train" else "heldout")
        self._compat_aliases()
        if self.mode == "ppo" and self.mode not in self._cache:
            self.combo_recipe.setCurrentIndex(self.combo_recipe.findData("origrecipe"))
        self.recipe_note.setText(MODE_NOTES[self.mode])
        self.btn_resume_file.setEnabled(self.kind in ("ppo", "dagger"))
        self.btn_launch.setText("최종 평가 실행" if self.mode == "grip_final" else
                                "데이터 수집 시작 / 재개" if self.mode == "grip_collect" else
                                "학습 이어서 시작" if self._resuming else "학습 시작")
        self._quiet = False
        if self.mode == "ppo" and self.mode not in self._cache:
            self._apply_recipe()
        self.sections.setCurrentRow(0)
        self._filter(self.search.text())
        self._refresh_preview()

    def _add_row(self, field, editor):
        dest = field["dest"]
        row = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        title = field.get("label", dest)
        heading = label(title + (" *" if field.get("required") else ""), "field")
        heading.setToolTip(field.get("flag", "") + "\n" + field.get("help", ""))
        layout.addWidget(heading)
        layout.addWidget(editor)
        help_text = field.get("help", "")
        if help_text:
            hint = label(help_text, "hint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        group = field.get("group", "기타")
        self.groups[group][1].insertWidget(self.groups[group][1].count() - 1, row)
        self.groups[group][2].append(dest)
        self.rows[dest] = row

    def _compat_aliases(self):
        # Small public handles used by console integration and the shared opponent table tests.
        for alias, dest in (("spin_race", "race_size"), ("spin_seed", "seed"), ("spin_envs", "envs"),
                            ("combo_opp", "opponent"), ("combo_device", "device"), ("edit_name", "name")):
            if dest in self.editors:
                setattr(self, alias, self.editors[dest].control)
        if "race_size" in self.editors and self.opp_table:
            self.editors["race_size"].changed.connect(self._race_changed)
            self._race_changed()
        if "adaptation" in self.editors:
            self.editors["adaptation"].changed.connect(self._adaptation_changed)
            self._adaptation_changed()

    def _adaptation_changed(self):
        adaptive = self.editors["adaptation"].value() != "off"
        for dest in ("reference", "research_estimator", "kl_scope"):
            if dest in self.editors:
                self.editors[dest].setEnabled(adaptive)
                self.editors[dest].setToolTip("자동 제어의 속도 / 전체 정책 적응 모드에서 사용합니다." if not adaptive else self.fields[dest].get("help", ""))

    def _race_changed(self):
        if self.opp_table:
            try:
                self.opp_table.set_count(max(0, int(self.editors["race_size"].value() or 1) - 1))
            except ValueError:
                pass                       # an intermediate '-' is legal while editing

    def _slots_toggled(self, enabled):
        if self.opp_table:
            self.opp_table.setVisible(enabled)
        from .training import SLOT_SUPERSEDED_FLAGS
        for flag in (*SLOT_SUPERSEDED_FLAGS, "--opponent"):
            dest = flag[2:].replace("-", "_")
            if dest in self.editors:
                self.editors[dest].setEnabled(not enabled)
        self._refresh_preview()

    def values(self):
        values = {dest: editor.value() for dest, editor in self.editors.items()}
        if self.tracks:
            values["tracks"] = self.tracks.spec()
            if "obstacle_draws" in self.fields:
                values["obstacle_draws"] = self.tracks.draws()
        if self.opp_table:
            on = self.chk_slots.isChecked()
            values["opp_slots"] = self.opp_table.slot_dicts() if on else None
            if on:
                from .training import SLOT_SUPERSEDED_FLAGS
                for flag in (*SLOT_SUPERSEDED_FLAGS, "--opponent"):
                    dest = flag[2:].replace("-", "_")
                    if dest in self.fields:
                        values[dest] = copy.deepcopy(self.fields[dest].get("default"))
        return values

    def set_values(self, values):
        unknown = set(values) - set(self.fields)
        if unknown:
            raise ValueError(f"지원하지 않는 설정: {', '.join(sorted(unknown))}")
        quiet, self._quiet = self._quiet, True
        for dest, editor in self.editors.items():
            if dest in values:
                editor.set_value(values[dest])
        if self.tracks and "tracks" in values:
            self.tracks.set_spec(str(values["tracks"] or ""))
            if values.get("obstacle_draws") is not None:
                self.tracks.spin_draws.setValue(int(values["obstacle_draws"]))
        if self.opp_table:
            self._race_changed()
            slots = values.get("opp_slots")
            if slots:
                from ...opponent_slots import parse_slots
                self.opp_table.set_slots(parse_slots(slots) if isinstance(slots, str) else slots)
            self.chk_slots.setChecked(bool(slots))
            self._slots_toggled(bool(slots))
        self._quiet = quiet
        self._refresh_preview()

    def _apply_recipe(self, *_args):
        if self._quiet or self.mode != "ppo":
            return
        from .training import COMMON_FLAGS, FROZEN_ORIGINAL, RECIPES
        key = self.combo_recipe.currentData()
        recipe = next((r for r in RECIPES if r.key == key), None)
        if recipe is None or key == "custom":
            self.recipe_note.setText(MODE_NOTES[self.mode])
            return
        self._resuming = False
        self.btn_launch.setText("학습 시작")
        args = shlex.split(COMMON_FLAGS + " " + recipe.race_flags)
        values = self.schema_api.parse_argv("ppo", args)
        values.update(tracks=recipe.tracks, race_size=recipe.race_size, opponent=recipe.opponent,
                      aux_grip=recipe.aux_grip, aux_opp=recipe.aux_opp, lr=recipe.lr,
                      lr_end=recipe.lr_end, kl_coef=recipe.kl, envs=recipe.envs, total=recipe.total,
                      controller="legacy", adaptation="off", init=FROZEN_ORIGINAL, obstacle_draws=8,
                      name=f"cl_{key}_s701_{time.strftime('%m%d_%H%M%S')}", seed=701)
        self.set_values(values)
        if recipe.tracks in ("train", "heldout", "eval"):
            self.tracks.select_split("train" if recipe.tracks == "train" else "heldout")
        self.recipe_note.setText(recipe.note)

    def _filter(self, text):
        query = text.strip().casefold()
        first = -1
        shown = 0
        for i, (group, (_page, _layout, destinations)) in enumerate(self.groups.items()):
            count = 0
            for dest in destinations:
                field = self.fields[dest]
                haystack = " ".join(str(field.get(k, "")) for k in ("dest", "flag", "label", "help", "group")).casefold()
                if dest == "tracks" and "obstacle_draws" in self.fields:
                    haystack += " " + str(self.fields["obstacle_draws"]).casefold()
                visible = not query or query in haystack
                self.rows[dest].setVisible(visible)
                count += int(visible) * (2 if dest == "tracks" and "obstacle_draws" in self.fields else 1)
            self.sections.item(i).setHidden(count == 0)
            self.sections.item(i).setText(f"{group}  ·  {count}")
            shown += count
            if count and first < 0:
                first = i
        if first >= 0 and (query or self.sections.currentRow() < 0 or self.sections.currentItem().isHidden()):
            self.sections.setCurrentRow(first)
        self.coverage_note.setText(f"{len(self.fields)}개 지원 설정\n{shown}개 표시" + (" · 검색 결과 없음" if shown == 0 else ""))

    def argv(self):
        values = self.values()
        metadata = self._metadata_for(values.get("init")) if self.kind in ("ppo", "dagger") else None
        if self._resuming and metadata and metadata.get("model_contract"):
            expected = self._checkpoint_contract(metadata)
            actual_values = self.schema_api.coerce_values(self.kind, values)
            expected_values = self.schema_api.coerce_values(self.kind, expected)
            changed = [key for key in expected if actual_values[key] != expected_values[key]]
            if changed:
                raise ValueError("재개 중에는 저장된 관측·모델 구조를 바꿀 수 없습니다: " + ", ".join(changed))
        critic_error = self._resume_critic_error(values, metadata)
        if critic_error:
            raise ValueError(critic_error)
        if self.kind == "ppo" and values.get("adaptation") != "off" and metadata:
            if not values.get("reference") and not metadata.get("original_reference", {}).get("path"):
                raise ValueError("첫 적응 학습에는 원본 D3 기준 정책을 지정해야 합니다.")
            same_stage = metadata.get("adaptation") == values["adaptation"]
            if bool(values.get("fresh_opt")) == same_stage:
                raise ValueError("같은 단계 재개는 Adam 복원, 새 단계 진입은 새 옵티마이저를 사용해야 합니다.")
        argv = self.schema_api.build_argv(self.kind, values, python=sys.executable)
        name = str(values.get("name") or os.path.abspath(os.path.expanduser(str(values.get("out") or self.mode))))
        return name, argv, str(values.get("device") or "cpu")

    def _refresh_preview(self, *_args):
        if self._quiet:
            return
        try:
            _name, argv, _device = self.argv()
            self.preview.setPlainText(shlex.join(argv))
            values = self.values()
            needs_contract = self.kind in ("ppo", "dagger") and (self._resuming or
                             (self.kind == "ppo" and values.get("adaptation") != "off"))
            if needs_contract and not self._metadata_for(values.get("init")):
                self.launch_note.setText("실행 전에 체크포인트의 관측·모델 설정과 학습 단계를 확인합니다.")
            else:
                if self._resuming and self.kind == "ppo" and values.get("adaptation") != "off":
                    note = "설정 준비됨 · 기존 단계의 전체 예산과 기준 정책을 유지합니다."
                elif self._resuming:
                    note = "저장된 관측·모델·행동 설정을 복원했습니다. 나머지 학습 파라미터는 현재 설정을 유지합니다."
                else:
                    note = "설정 준비됨"
                self.launch_note.setText(note)
        except (ValueError, TypeError) as exc:
            self.preview.setPlainText(str(exc))
            self.launch_note.setText(str(exc).splitlines()[0][:250])

    def _launch(self):
        try:
            name, argv, device = self.argv()
            values = self.values()
            needs_contract = self.kind in ("ppo", "dagger") and (self._resuming or
                             (self.kind == "ppo" and values.get("adaptation") != "off"))
            if needs_contract and not self._metadata_for(values.get("init")):
                self._begin_inspection(values["init"], launch=True)
                return
            if self.opp_table and self.chk_slots.isChecked():
                problem = self.opp_table.problem(int(self.values().get("race_size", 1)))
                if problem:
                    raise ValueError(problem)
            if self.kind in ("ppo", "dagger") and not self._resuming:
                run = Path(catalog.RUNS_DIR) / name
                if run.is_dir() and any(run.glob("*.pt")):
                    raise ValueError("체크포인트가 있는 런입니다. '이어서'를 사용하거나 새 이름을 지정하세요.")
        except (ValueError, TypeError) as exc:
            self.launch_note.setText(str(exc))
            return
        self.launch_requested.emit(name, argv, device)

    def set_runs(self, runs):
        self._runs = list(runs)

    def save_config(self, path):
        modes = dict(self._cache)
        modes[self.mode] = self.values()
        resume_modes = set(self._resume_modes)
        if self._resuming:
            resume_modes.add(self.mode)
        else:
            resume_modes.discard(self.mode)
        payload = {"format": "f1sim-training-panel-v1", "mode": self.mode, "modes": modes,
                   "resume_modes": sorted(resume_modes)}
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
        temporary.replace(path)

    def load_config(self, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("format") != "f1sim-training-panel-v1":
            raise ValueError("지원하지 않는 학습 설정 파일입니다.")
        mode = payload.get("mode")
        valid_modes = {key for key, _, _ in MODES}
        modes = payload.get("modes")
        if mode not in valid_modes or not isinstance(modes, dict) or mode not in modes or any(k not in valid_modes or not isinstance(v, dict) for k, v in modes.items()):
            raise ValueError("잘못된 학습 모드 또는 설정 구조입니다.")
        restored_modes = {}
        for key, values in modes.items():
            kind = next(kind for name, _label, kind in MODES if name == key)
            restored_modes[key] = self.schema_api.coerce_values(kind, values)
        resume_modes = payload.get("resume_modes", [])
        if not isinstance(resume_modes, list) or any(key not in valid_modes for key in resume_modes):
            raise ValueError("잘못된 재개 모드 목록입니다.")
        self._cache = restored_modes
        self._resume_modes = set(resume_modes)
        self.mode = ""                 # loading must not overwrite the selected saved mode with stale edits
        self.combo_mode.blockSignals(True)
        self.combo_mode.setCurrentIndex(self.combo_mode.findData(mode))
        self.combo_mode.blockSignals(False)
        self._mode_changed()

    def _save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "학습 설정 저장", "training.json", "JSON (*.json)")
        if path:
            try:
                self.save_config(path)
                self.launch_note.setText(f"설정 저장: {path}")
            except (OSError, ValueError) as exc:
                self.launch_note.setText(str(exc))

    def _load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "학습 설정 열기", "", "JSON (*.json)")
        if path:
            try:
                self.load_config(path)
            except (OSError, ValueError) as exc:
                self.launch_note.setText(str(exc))

    def load_job(self, argv, checkpoint="", checkpoint_metadata=None):
        module = argv[argv.index("-m") + 1] if "-m" in argv else ""
        start = argv.index(module) + 1 if module else 0
        if module == "f1sim.learn.ppo":
            kind = "ppo"
        elif module == "f1sim.learn.dagger":
            kind = "dagger"
        elif module == "f1sim.learn.policy_grip_data" and len(argv) > start:
            kind = "grip_" + argv[start]
            start += 1
        elif module == "f1sim.learn.adaptive_grip_train":
            kind = "grip_pilot"
        else:
            raise ValueError("이 작업의 학습 모듈을 확인할 수 없습니다.")
        values = self.schema_api.parse_argv(kind, list(argv[start:]))
        mode = values.get("adaptation", "off") if kind == "ppo" else kind
        if mode == "off":
            mode = "ppo"
        self.set_mode(mode)
        self._resuming = False
        self.set_values(values)
        if checkpoint:
            self.resume_checkpoint(checkpoint, metadata=checkpoint_metadata)
        elif kind == "grip_collect":
            self._resuming = True
            self.launch_note.setText("수집 설정과 출력 폴더를 복원했습니다. 이미 완료한 데이터 조각은 건너뜁니다.")
        else:
            self.launch_note.setText("작업 설정을 복원했습니다. 새 실행에는 새 출력 경로를 지정하세요.")

    def _metadata_for(self, path):
        try:
            path = Path(path).expanduser().resolve(strict=True)
            stat = path.stat()
            return self._metadata_cache.get((str(path), stat.st_size, stat.st_mtime_ns))
        except (OSError, TypeError):
            return None

    def _remember_metadata(self, path, metadata):
        path = Path(path).expanduser().resolve(strict=True)
        stat = path.stat()
        if ("file_size" in metadata and metadata["file_size"] != stat.st_size) or (
                "mtime_ns" in metadata and metadata["mtime_ns"] != stat.st_mtime_ns):
            raise ValueError("정보를 확인하는 동안 체크포인트 파일이 변경되었습니다. 다시 선택하세요.")
        self._metadata_cache[(str(path), stat.st_size, stat.st_mtime_ns)] = metadata

    def _begin_inspection(self, path, *, launch=False):
        self._inspection_mode = self.mode
        self._inspection_init = self.values().get("init")
        self._pending_launch = launch
        self.btn_launch.setEnabled(False)
        self.btn_resume_file.setEnabled(False)
        self.launch_note.setText("체크포인트 단계 확인 중… CPU 별도 프로세스에서 읽고 있습니다.")
        self.inspector.inspect(path)

    def _checkpoint_failed(self, message):
        self._pending_launch = False
        self.btn_launch.setEnabled(True)
        self.btn_resume_file.setEnabled(self.kind in ("ppo", "dagger"))
        self.launch_note.setText(f"체크포인트 확인 실패: {message}")

    def _checkpoint_ready(self, path, metadata):
        launch, mode = self._pending_launch, self._inspection_mode
        self._pending_launch = False
        self.btn_launch.setEnabled(True)
        self.btn_resume_file.setEnabled(self.kind in ("ppo", "dagger"))
        if self.mode != mode or self.values().get("init") != self._inspection_init:
            self.launch_note.setText("학습 모드가 변경되어 이전 체크포인트 요청을 적용하지 않았습니다.")
            return
        try:
            self._remember_metadata(path, metadata)
            self._apply_checkpoint_metadata(path, metadata)
            if launch:
                self._launch()
        except (ValueError, TypeError, OSError) as exc:
            self.launch_note.setText(str(exc))

    def resume_checkpoint(self, path, *, metadata=None):
        if self.kind not in ("ppo", "dagger"):
            raise ValueError("이 작업은 정책 체크포인트 재개를 지원하지 않습니다.")
        metadata = metadata or self._metadata_for(path)
        if metadata is None:
            self._begin_inspection(path)
            return
        self._remember_metadata(path, metadata)
        self._apply_checkpoint_metadata(path, metadata)

    def _apply_checkpoint_metadata(self, path, metadata):
        values = self.values()
        values["init"] = str(path)
        contract_updates = self._checkpoint_contract(metadata)
        values.update(contract_updates)
        if self.kind == "ppo":
            target_stage = values.get("adaptation", "off")
            source_stage = metadata.get("adaptation", "off")
            if target_stage == "full" and source_stage not in ("speed", "full"):
                raise ValueError("전체 정책 적응은 속도 단계 체크포인트에서 시작해야 합니다.")
            if target_stage == "speed" and source_stage == "full":
                raise ValueError("전체 정책 단계에서 속도 전용 단계로 되돌아갈 수 없습니다.")
            same_stage = target_stage != "off" and source_stage == target_stage
            if target_stage != "off":
                values["fresh_opt"] = not same_stage
                reference = metadata.get("original_reference") or {}
                if reference.get("path"):
                    values["reference"] = reference["path"]
                if same_stage:
                    schedule = metadata.get("stage_schedule")
                    if not isinstance(schedule, dict) or schedule.get("stage") != target_stage:
                        raise ValueError("같은 단계 재개에 필요한 학습 일정이 체크포인트에 없습니다.")
                    parameters = schedule.get("hyperparameters") or {}
                    values.update({key: value for key, value in parameters.items()
                                   if key in self.fields and key not in ("tracks", "opponent")})
                    if parameters.get("tracks"):
                        values["tracks"] = ",".join(parameters["tracks"])
                    environment = parameters.get("environment") or {}
                    aliases = {"reward_progress": "progress_reward", "reward_alive": "alive_reward",
                               "reward_collision": "collision_penalty", "reward_steer_rate": "steer_penalty",
                               "reward_proximity": "proximity_penalty", "reward_wrong_way": "wrong_way_penalty",
                               "reward_collision_speed": "collision_speed_penalty", "reward_plan_clearance": "plan_clearance_penalty",
                               "reward_lap": "lap_bonus", "reward_lap_time": "lap_time_bonus", "reward_overtake": "overtake_bonus",
                               "reward_car_contact": "car_contact_penalty", "reward_car_proximity": "car_proximity_penalty",
                               "reward_sideslip": "sideslip_penalty", "reward_ttc": "ttc_penalty", "opponent_slots": "opp_slots"}
                    for key, value in environment.items():
                        dest = aliases.get(key, key[:-6] if key.endswith("_range") else key)
                        if dest in self.fields and dest != "opponent":
                            values[dest] = abs(value) if key == "reward_collision" else value
                    if environment.get("max_steps"):
                        values["episode_s"] = environment["max_steps"] / 40.
                    if "reward_overtake_hold" in environment:
                        values["overtake_sustained"] = [environment["overtake_hold_dist"], environment["overtake_hold_time"], environment["reward_overtake_hold"]]
                    values["opp_slots"] = parameters.get("opponent_slots") or None
                    if not values["opp_slots"] and parameters.get("opponent"):
                        values["opponent"] = parameters["opponent"]
                    values["total"] = schedule["total_steps"]
                    if metadata.get("run"):
                        values["name"] = metadata["run"]
                if metadata.get("embedded_estimator"):
                    values["estimator"] = None
                self._resuming = same_stage
            else:
                if source_stage != "off":
                    raise ValueError("적응 학습 체크포인트입니다. 속도 / 전체 정책 적응 작업을 먼저 선택하세요.")
                values["fresh_opt"] = False
                self._resuming = True
        else:
            if metadata.get("phase") != "dagger" or not isinstance(metadata.get("iter"), int):
                raise ValueError("DAgger 반복 번호가 체크포인트에 없습니다.")
            values["start_iter"] = metadata["iter"] + 1
            mix = metadata.get("collection_mix")
            if isinstance(mix, dict) and "requested_solo_fraction" in mix:
                values["solo_fraction"] = mix["requested_solo_fraction"]
                values["solo_tracks"] = ",".join(mix.get("solo_tracks") or [])
                if mix.get("traffic_teacher"):
                    values["teacher_kind"] = mix["traffic_teacher"]
                if "solo_steps" in mix and "traffic_steps" in mix:
                    values["steps"] = mix["solo_steps"] + mix["traffic_steps"]
            self._resuming = True
        # Saved observations/model inputs are authoritative even when a caller has a different
        # form preset. A resume must never become the learner's tolerant architecture migration.
        values.update(contract_updates)
        self.set_values(values)
        self.btn_launch.setText("같은 단계 이어서 시작" if self._resuming else "새 단계 학습 시작")
        self._refresh_preview()
        critic_error = self._resume_critic_error(values, metadata)
        if critic_error:
            self.launch_note.setText("관측·모델 설정을 복원했습니다. " + critic_error)
        elif self.kind == "ppo" and values.get("adaptation") != "off" and not values.get("reference"):
            self.launch_note.setText("첫 적응 학습에는 원본 D3 기준 정책을 지정해야 합니다.")
        elif not self._resuming:
            self.launch_note.setText("새 단계 진입: 현재 전체 예산을 유지하고 새 옵티마이저를 사용합니다.")
        elif contract_updates and values.get("adaptation", "off") == "off":
            self.launch_note.setText("저장된 관측·모델·행동 설정을 복원했습니다. 나머지 학습 파라미터는 현재 설정을 유지합니다.")

    def _resume_critic_error(self, values, metadata):
        if not self._resuming or not metadata or not metadata.get("model_contract"):
            return ""
        contract = self.schema_api.get_observation_contract(self.kind)
        model = metadata["model_contract"]
        race = values.get("race_size")
        if not isinstance(race, int) or race < 1:
            return "재개할 레이스의 차량 수를 지정하세요."
        if not contract.get("privileged_dims") or "priv_adapters" not in contract:
            return "현재 설정 스키마에 critic 입력 계약이 없어 재개 호환성을 확인할 수 없습니다."
        raw_dim = contract["privileged_dims"]["solo" if race == 1 else "race"]
        adapter = values.get("critic_priv_adapter") or model.get("priv_adapter") or ""
        expected = raw_dim
        if adapter:
            rule = contract["priv_adapters"].get(adapter)
            if rule is None or self.kind != "ppo":
                return "저장된 critic 입력 어댑터는 이 학습 모드에서 재현할 수 없습니다."
            if raw_dim != rule["input"]:
                return f"critic 입력 어댑터는 {rule['input']}차원 관측이 필요합니다. 호환되는 차량 수 또는 어댑터를 선택하세요."
            expected = rule["output"]
        if model.get("priv_dim") != expected:
            return (f"재개할 critic 입력은 {model.get('priv_dim')}차원, 현재 차량 설정은 {expected}차원입니다. "
                    "호환되는 차량 수 또는 어댑터를 선택하거나 새 학습으로 시작하세요.")
        return ""

    def _checkpoint_contract(self, metadata):
        """Restore recorded inputs; refuse fixed runtime contracts the CLI cannot reproduce."""
        if "observation_spec" not in metadata and "model_contract" not in metadata:
            return {}                     # stage-only metadata used by isolated controller tests
        spec, model = metadata.get("observation_spec"), metadata.get("model_contract")
        if not isinstance(spec, dict) or not spec or not isinstance(model, dict) or not model:
            raise ValueError("체크포인트에 관측·모델 설정이 없어 정확하게 재개할 수 없습니다.")
        contract = self.schema_api.get_observation_contract(self.kind)
        for key, expected in contract["fixed"].items():
            actual = spec.get(key)
            if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isclose(
                    float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"저장된 {key}={actual!r} 설정을 현재 학습기가 재현할 수 없습니다 (지원값 {expected}).")
        for saved, observation in (("n_stack", "scan_stack"), ("n_beams", "n_beams"), ("act_dim", "act_dim")):
            if saved not in model or observation not in spec or model[saved] != spec[observation]:
                raise ValueError(f"체크포인트 모델과 관측의 {observation} 정보가 일치하지 않습니다.")
        if metadata.get("observation_proprio_dim") is not None and model.get("proprio_dim") != metadata["observation_proprio_dim"]:
            raise ValueError("체크포인트의 관측 차원과 모델 입력 차원이 일치하지 않습니다.")
        action_mode = metadata.get("action_mode")
        if action_mode is None:
            action_mode = next((mode for mode, dim in contract["action_dims"].items() if dim == model["act_dim"]), None)
        if action_mode not in contract["action_dims"] or contract["action_dims"][action_mode] != model["act_dim"]:
            raise ValueError("저장된 행동 차원을 현재 direct / plan 행동으로 재현할 수 없습니다.")
        updates = {"action_mode": action_mode}
        for source, dest in contract["restorable"].items():
            if source in spec and dest in self.fields:
                updates[dest] = spec[source]
        for key in ("scan_deltas", "scan_stem", "temporal_encoder"):
            if key in model and key in self.fields:
                updates[key] = model[key]
        token = model.get("opp_token", "off") or "off"
        if token != (spec.get("opp_token", "off") or "off"):
            raise ValueError("체크포인트의 상대차 관측 설정이 모델과 일치하지 않습니다.")
        updates["opp_token"] = token
        memory = model.get("memory")
        if memory and (not isinstance(memory, dict) or not {"kind", "hidden_size", "layers", "critic"} <= set(memory)):
            raise ValueError("체크포인트의 GRU 구조 기록이 불완전합니다.")
        updates["memory"] = memory["kind"] if memory else "off"
        if memory:
            if memory.get("layers") != contract["memory"]["layers"]:
                raise ValueError("저장된 GRU 계층 수는 현재 학습 패널에서 재현할 수 없습니다.")
            updates["memory_hidden"] = memory["hidden_size"]
            if "memory_critic" in self.fields:
                updates["memory_critic"] = memory["critic"]
            elif memory.get("critic") != contract["memory"]["critic"]:
                raise ValueError("저장된 critic 메모리 설정은 이 학습 모드에서 재현할 수 없습니다.")
        channels = model.get("scan_channels") or {}
        if not isinstance(channels, dict):
            raise ValueError("체크포인트의 LiDAR 추가 채널 기록을 해석할 수 없습니다.")
        updates["scan_channels"] = list(channels.get("channels") or [])
        if "memory_tau_s" in channels:
            updates["scan_memory_tau"] = channels["memory_tau_s"]
        if "cond" in self.fields:
            updates["cond"] = (model.get("cond") or {}).get("source") if model.get("cond_dim") else "none"
        elif model.get("cond_dim"):
            raise ValueError("조건부 관측 모델은 이 학습 모드에서 정확하게 재개할 수 없습니다.")
        if "critic_priv_adapter" in self.fields and model.get("priv_adapter"):
            updates["critic_priv_adapter"] = model["priv_adapter"]
        elif "critic_priv_adapter" not in self.fields and model.get("priv_adapter"):
            raise ValueError("저장된 critic 입력 어댑터는 이 학습 모드에서 지원되지 않습니다.")
        motion = model.get("motion")
        if "motion_memory" in self.fields:
            updates["motion_memory"] = bool(motion)
            if motion:
                for source, dest in (("hidden_size", "motion_hidden"), ("channels", "motion_channels"), ("critic", "motion_critic")):
                    if source in motion:
                        updates[dest] = motion[source]
            else:
                updates.update(aux_motion=0., aux_opp_mask=0.)
        for block, coefficient, parameters in (
                ("future_head", "aux_future", {"k": "aux_future_k", "width": "aux_future_width"}),
                ("floor_head", "aux_floor", {"width": "aux_floor_width"})):
            if coefficient in self.fields:
                if model.get(block):
                    updates.update({dest: model[block][source] for source, dest in parameters.items() if source in model[block]})
                else:
                    updates[coefficient] = 0.
        # Check restored enum/type contracts without checking unrelated paths or changing widgets.
        self.schema_api.coerce_values(self.kind, updates)
        return updates

    def _resume_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "이어서 학습할 정책 체크포인트", catalog.RUNS_DIR, "정책 (*.pt)")
        if not path:
            return
        try:
            # A recorded launch restores the full observation/stage contract before resuming.
            from .training import JobManager
            job = next((j for j in JobManager().list_jobs() if Path(j.run_dir) == Path(path).parent), None)
            if job and self.mode not in ("speed", "full"):
                self.load_job(job.argv, path)
            else:
                self.resume_checkpoint(path)
        except (ValueError, OSError) as exc:
            self.launch_note.setText(str(exc))
