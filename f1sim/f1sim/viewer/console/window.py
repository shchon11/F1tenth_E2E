"""The console window: one place to pick what to watch, watch it, and see whether to believe it.

Layout follows the questions in the order they get asked. Left is *what to run* and is only
touched between sessions; centre is the picture and the controls that act on it; right is *is this
real* -- the live numbers, kept apart from the picture so a healthy frame rate can never be
mistaken for a healthy simulation.

The window never talks to torch, a GPU, or a file that takes time to read. It emits requests and
renders what it is told. Everything slow lives in the worker process, which is what keeps the
event loop answering while a checkpoint loads or CUDA graphs are captured.
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from . import app as _app, catalog, theme
from ... import tracks
from .catalog import GROUP_CAVEAT, GROUP_HINT, GROUP_ORDER, MapCatalog, RunInfo, format_age
from .frames import Freshness, INTERP_LAG_FRAMES
from .opponent_table import OpponentSlotTable
from .overlays import ActivationPanel, DashPanel, PolicyInputPanel
from .protocol import (SessionConfig, STAGE_TEXT, STATE_FAILED, STATE_IDLE, STATE_PAUSED,
                       STATE_PREPARING, STATE_RUNNING, STATE_STOPPING)
from .. import recorder as REC
from .theme import C, SP
from .viewport import CAMERA_KEYS, CAMERA_MODES, MAX_RENDER_CARS, ViewportWidget
from .widgets import (Card, Collapsible, FieldRow, FilterList, FlowLayout, KeyValueList,
                      MetricTile, PendingButton, PendingToggle, SegmentedButtons, Sparkbar,
                      StateBadge, hline, label)

#: How long a command may go unacknowledged before the status bar says so. The UI stays responsive
#: either way; this only decides when to admit that the worker is slow.
ACK_WARN_S = 0.7


def _scroll_panel(width: int) -> Tuple[QtWidgets.QScrollArea, QtWidgets.QVBoxLayout]:
    area = QtWidgets.QScrollArea()
    area.setObjectName("PanelScroll")
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    inner = QtWidgets.QWidget()
    inner.setObjectName("Panel")
    v = QtWidgets.QVBoxLayout(inner)
    v.setContentsMargins(SP[2], SP[2], SP[2], SP[2])
    v.setSpacing(SP[2])
    area.setWidget(inner)
    area.setMinimumWidth(260)
    area.setMaximumWidth(max(width, 420))
    return area, v


def _cuda_device_names() -> List[str]:
    """`cuda:i` for every visible GPU. Names come from `nvidia-smi` rather than torch: this is the
    GUI process, and the whole point of the console's split is that it never imports torch."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=4).stdout.strip()
    except Exception:
        return ["cuda"]
    names = []
    for line in out.splitlines():
        idx, _, name = line.partition(",")
        if idx.strip().isdigit():
            names.append(f"cuda:{idx.strip()}")
    return names or ["cuda"]


class ConsoleWindow(QtWidgets.QMainWindow):
    """The whole UI. A controller connects to the `*_requested` signals and drives `apply_*`."""

    # what the user asked for; a controller turns these into worker commands
    start_requested = QtCore.pyqtSignal(object)        # SessionConfig
    cancel_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    pause_requested = QtCore.pyqtSignal(bool)
    reset_requested = QtCore.pyqtSignal()
    focus_requested = QtCore.pyqtSignal(int)
    overlay_requested = QtCore.pyqtSignal(dict)
    describe_requested = QtCore.pyqtSignal(str)
    mu_requested = QtCore.pyqtSignal(str, float)          # (mode, mu) live friction control
    dial_requested = QtCore.pyqtSignal(float)             # grip dial for a conditional ("dial") policy
    retry_requested = QtCore.pyqtSignal()
    close_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # ASCII: some taskbars read X11's legacy Latin-1 `WM_NAME` and mangle a Korean title.
        # `app.APP_NAME` is the one place that decides it. The Korean name is on the header below.
        self.setWindowTitle(_app.APP_NAME)
        self.resize(1600, 950)
        self.setMinimumSize(1120, 700)

        self.state = STATE_IDLE
        self.generation = 0
        self.runs: List[RunInfo] = []
        self.maps = MapCatalog()
        self._selected_run: Optional[str] = None
        self._selected_map: Optional[str] = None
        #: A remembered map id, waiting for the worker's catalogue. The list is built in the worker
        #: (map names need torch), so at construction time there is nothing to select in yet.
        self._pref_map: Optional[str] = None
        #: Seeds the worker's draw when the scenario leaves the obstacle seed open. Re-rolled by
        #: 다시 뽑기; part of the config, so a re-roll is a new generation and the facts strip can
        #: print the number that was actually used.
        self._session_seed: int = __import__("random").randrange(1, 10 ** 9)
        self._session_cars = 0
        self._session_ids: List[int] = []
        self._running_facts: Dict[str, object] = {}
        self._last_error: Optional[dict] = None
        self._pending: Dict[str, float] = {}
        #: The stage the worker last reported, when it started, and its note. Preparation is
        #: reported from these rather than from the ack ledger -- `start` is answered by `ready`,
        #: not by an ack, so the ack timer says nothing useful about it.
        self._stage: Optional[str] = None
        self._stage_since: Optional[float] = None
        self._stage_note: str = ""
        self._policy_autofold_done = False
        self._sidebar_autofold_done = False
        self._closing_allowed = False

        self._build_ui()
        self._install_shortcuts()
        # Last, and after every widget exists: the controls a person set on the previous launch.
        # A remembered map waits for the worker's catalogue (`set_maps`); everything else applies now.
        self.restore_prefs()
        self.apply_state(STATE_IDLE)

    # ================================================================ construction
    def _build_ui(self):
        root = QtWidgets.QWidget()
        root.setObjectName("Panel")
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.left_panel = self._build_left()
        self.centre = self._build_centre()
        self.right_panel = self._build_right()
        # Two pages in the middle: driving (the viewport) and training. Training is the console's
        # second job, not a dialog: it launches and watches PPO runs and hands their checkpoints
        # back to the driving page.
        from .training import TrainingPage
        self.training = TrainingPage()
        self.training.view_checkpoint_requested.connect(self._view_checkpoint_from_training)
        self.training.runs_changed.connect(self._runs_changed_from_training)
        # The stack must not let the training page's minimum sizes govern the driving page: a
        # QStackedWidget reports the maximum of its pages, which would squeeze the viewport at
        # small window sizes even while the training page is hidden.
        self.training.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        # The third page: the environment editor. Builds and edits scenes (`f1sim.scene`) with
        # its own GL viewport, and hands a saved scene to the driving page as `scene:<name>`.
        from .env_editor import EnvEditorPage
        self.editor = EnvEditorPage()
        self.editor.drive_requested.connect(self._drive_from_editor)
        self.editor.scenes_changed.connect(self._scenes_changed_from_editor)
        self.editor.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.centre_stack = QtWidgets.QStackedWidget()
        self.centre_stack.setObjectName("Centre")
        self.centre_stack.addWidget(self.centre)
        self.centre_stack.addWidget(self.training)
        self.centre_stack.addWidget(self.editor)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.centre_stack)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([330, 900, 350])
        outer.addWidget(self.splitter, 1)
        outer.addWidget(self._build_status())
        self.setCentralWidget(root)

    # ---------------------------------------------------------------- header
    def _build_header(self) -> QtWidgets.QWidget:
        head = QtWidgets.QFrame()
        head.setObjectName("Header")
        head.setFixedHeight(44)
        h = QtWidgets.QHBoxLayout(head)
        h.setContentsMargins(SP[2], SP[1], SP[2], SP[1])
        h.setSpacing(SP[2])

        title = QtWidgets.QLabel("f1sim 주행 콘솔")
        title.setObjectName("HeaderTitle")
        h.addWidget(title)

        self.badge = StateBadge()
        h.addWidget(self.badge)
        self.mode_buttons = SegmentedButtons([("drive", "주행", "정책을 골라 달리는 화면"),
                                              ("train", "학습", "PPO 학습을 설정·실행하고 현황을 보는 화면"),
                                              ("edit", "환경", "트랙·벽·장애물을 3D 로 직접 만들고 고치는 화면")])
        self.mode_buttons.set_current("drive")
        self.mode_buttons.selected.connect(self.set_mode)
        h.addWidget(self.mode_buttons)

        self.header_summary = QtWidgets.QLabel("—")
        self.header_summary.setObjectName("HeaderSub")
        self.header_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        h.addWidget(self.header_summary, 1)
        self.header_rec = QtWidgets.QLabel("● REC")
        self.header_rec.setStyleSheet(f"color: {C['danger']}; font-weight: 600;")
        self.header_rec.setToolTip("3D 화면을 mp4 로 쓰는 중입니다.")
        self.header_rec.setVisible(False)
        h.addWidget(self.header_rec)

        self.btn_left_panel = QtWidgets.QPushButton("설정 패널")
        self.btn_left_panel.setObjectName("GhostButton")
        self.btn_left_panel.setCheckable(True)
        self.btn_left_panel.setChecked(True)
        self.btn_left_panel.setToolTip("왼쪽 설정 패널 접기/펴기")
        self.btn_left_panel.toggled.connect(lambda on: self.left_panel.setVisible(on))
        h.addWidget(self.btn_left_panel)

        self.btn_right_panel = QtWidgets.QPushButton("계기 패널")
        self.btn_right_panel.setObjectName("GhostButton")
        self.btn_right_panel.setCheckable(True)
        self.btn_right_panel.setChecked(True)
        self.btn_right_panel.setToolTip("오른쪽 계기 패널 접기/펴기")
        self.btn_right_panel.toggled.connect(lambda on: self.right_panel.setVisible(on))
        h.addWidget(self.btn_right_panel)

        self.btn_help = QtWidgets.QPushButton("도움말 (F1)")
        self.btn_help.setObjectName("GhostButton")
        self.btn_help.clicked.connect(self.show_help)
        h.addWidget(self.btn_help)
        return head

    # ---------------------------------------------------------------- left: what to run
    def _build_left(self) -> QtWidgets.QWidget:
        area, v = _scroll_panel(340)
        self._left_scroll = area
        self._panel_nudged_for = 0
        #: The scrolling part of the left panel, kept as an attribute so a caller (a screenshot
        #: script, a test) can put a card on screen without reaching through the widget tree.
        self.left_scroll = area

        # -- what is selected, always visible at the top. The lists below can be scrolled away;
        # what you are about to start cannot be. Full names, never elided: a run called
        # `ppo_race_0910` and one called `ppo_race_0910b` must not read the same.
        sel_card = Card("다음 시작 설정")
        self.sel_run = label("런: —", "body")
        self.sel_run.setObjectName("Mono")
        self.sel_run.setWordWrap(True)
        self.sel_run.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        sel_card.add(self.sel_run)
        self.sel_map = label("맵: —", "body")
        self.sel_map.setObjectName("Mono")
        self.sel_map.setWordWrap(True)
        self.sel_map.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        sel_card.add(self.sel_map)
        # A short status, not a paragraph. The full caveat lives on the tooltip: the panel's job
        # is to say what is selected, not to teach the split convention every time you look at it.
        self.sel_note = label("", "hint")
        self.sel_note.setToolTip(GROUP_CAVEAT)
        sel_card.add(self.sel_note)
        # The picker shows what Start *would* use. While a session is running that can differ from
        # what is actually on screen -- a list refresh moves the selection, or someone starts
        # choosing the next thing to look at. Two run names in one window with nothing saying which
        # is which is how a screenshot ends up labelled with the wrong policy, so when they differ
        # this says so out loud and names the one that is running.
        self.sel_running = label("", "hint.warn")
        self.sel_running.setWordWrap(True)
        sel_card.add(self.sel_running)
        v.addWidget(sel_card)

        # -- policy run
        run_card = Card("① 정책 런")
        self.run_list = FilterList("런 이름 검색", rows=5)
        self.run_list.activated.connect(self._on_run_selected)
        run_card.add(self.run_list)
        self.ckpt_fold = Collapsible("체크포인트 상세", expanded=False)
        self.ckpt_info = KeyValueList()
        self.ckpt_info.set("체크포인트", "—")
        self.ckpt_info.set("저장", "—")
        self.ckpt_info.set("진행", "—")
        self.ckpt_info.set("저장 시점 지표", "—",
                           tooltip="체크포인트에 기록된 값입니다. 지금 화면의 주행 성능이 아닙니다.")
        self.ckpt_fold.add(self.ckpt_info)
        self.run_note = label("체크포인트가 갱신되면 자동으로 다시 읽습니다.", "hint")
        self.ckpt_fold.add(self.run_note)
        run_card.add(self.ckpt_fold)
        v.addWidget(run_card)

        # -- map: the base track, then the scenario built on it
        #
        # The list holds *maps*, one row each. It used to hold every direction, obstacle family and
        # placement seed as its own row -- two hundred rows, forty-character names, the same map
        # twenty times -- which is not a list of maps and cannot be scanned. Direction, obstacle
        # and seed are three controls under it, defaulting to 정방향 / 없음 / 무작위, so picking a
        # map is enough to drive it.
        map_card = Card("② 맵")
        self.map_group = QtWidgets.QComboBox()
        self.map_group.addItem("목록 읽는 중…")
        self.map_group.setEnabled(False)
        self.map_group.setToolTip(GROUP_CAVEAT)
        self.map_group.currentIndexChanged.connect(lambda _: self._refresh_map_list())
        map_card.add(self.map_group)
        self.map_list = FilterList("맵 검색 (Ctrl+F) — 세 그룹 밖의 맵도 전부", rows=6)
        self.map_list.activated.connect(self._on_map_selected)
        self.map_list.search.textChanged.connect(lambda _t: self._refresh_map_list())
        map_card.add(self.map_list)
        self.map_note = label("맵을 고르면 바로 달릴 수 있습니다. 아래에서 방향·장애물만 바꾸세요.", "hint")
        self.map_note.setToolTip(GROUP_CAVEAT)
        map_card.add(self.map_note)
        self.map_list.tree.setToolTip(GROUP_CAVEAT)
        map_card.add(hline())

        scen = label("시나리오", "field")
        map_card.add(scen)
        self.seg_direction = SegmentedButtons(
            [(d, tracks.DIRECTION_LABEL[d],
              {"": "기록된 그대로 달립니다.", "rev": "같은 맵을 반대 방향으로 돕니다.",
               "mir": "좌우를 뒤집은 맵입니다 (코너가 반대로).",
               "mir+rev": "좌우를 뒤집고 반대 방향으로 돕니다."}[d])
             for d in tracks.DIRECTIONS])
        self.seg_direction.set_current("")
        self.seg_direction.selected.connect(lambda _k: self._on_scenario_changed())
        map_card.add(FieldRow("방향", self.seg_direction, ""))

        # 기본 / 없음 / the families. `기본` is the map as authored -- an editor scene keeps the
        # obstacles its author placed -- and `없음` removes them. Those used to be one entry
        # labelled 없음, which is what the user caught: a custom scene picked with "없음" came up
        # full of boxes.
        obs_row = QtWidgets.QHBoxLayout()
        obs_row.setSpacing(SP[0])
        self.combo_obstacle = QtWidgets.QComboBox()
        for o in tracks.ASSET_OBSTACLES:
            self.combo_obstacle.addItem(tracks.ASSET_PLACEMENT_LABEL[o], o)
        self.combo_obstacle.setCurrentIndex(self.combo_obstacle.findData(""))
        self.combo_obstacle.currentIndexChanged.connect(lambda _: self._on_scenario_changed())
        obs_row.addWidget(self.combo_obstacle, 1)
        # A family ADDS to what the map has. This is how a custom scene is used as a bare track for
        # one: `scene:x+bare+hard3`. Enabled only where it means something -- a map with placed
        # obstacles and a family selected.
        self.chk_bare_first = QtWidgets.QCheckBox("배치 장애물 먼저 제거")
        self.chk_bare_first.setToolTip(
            "장애물 종류는 맵이 이미 가진 것 위에 더합니다.\n"
            "켜면 작성자가 배치한 장애물을 먼저 걷어내고 그 위에 올립니다 (id 로는 +bare).")
        self.chk_bare_first.toggled.connect(lambda _: self._on_scenario_changed())
        self.chk_bare_first.hide()
        obs_box = QtWidgets.QWidget()
        obs_box.setLayout(obs_row)
        self.row_obstacle = FieldRow("장애물", obs_box, tracks.OBSTACLE_HINT[""])
        map_card.add(self.row_obstacle)

        # Seed. "무작위" is the default because it is what the user asked for: pick a map, get
        # boxes somewhere, drive. The number is drawn in the worker from the session seed and shown
        # in the facts strip, so a placement worth keeping can be typed back in as 고정.
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.setSpacing(SP[0])
        self.combo_seed = QtWidgets.QComboBox()
        self.combo_seed.addItem("무작위", "random")
        self.combo_seed.addItem("고정", "fixed")
        self.combo_seed.currentIndexChanged.connect(self._on_seed_mode)
        seed_row.addWidget(self.combo_seed, 1)
        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(44)
        self.spin_seed.setEnabled(False)
        self.spin_seed.valueChanged.connect(lambda _: self._on_scenario_changed())
        seed_row.addWidget(self.spin_seed, 1)
        self.btn_reroll = QtWidgets.QPushButton("다시 뽑기")
        self.btn_reroll.setObjectName("GhostButton")
        self.btn_reroll.setToolTip("장애물 배치를 새로 뽑습니다. 다른 맵이 되므로 세션을 다시 시작합니다.")
        self.btn_reroll.clicked.connect(self._on_reroll)
        seed_row.addWidget(self.btn_reroll)
        seed_box = QtWidgets.QWidget()
        seed_box.setLayout(seed_row)
        self.row_seed = FieldRow("시드", seed_box, "무작위: 시작할 때 하나 뽑고 화면 상단에 그 번호를 보여줍니다.")
        map_card.add(self.row_seed)

        self.scenario_line = label("", "hint")
        self.scenario_line.setObjectName("Mono")
        self.scenario_line.setWordWrap(True)
        self.scenario_line.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.scenario_line.setToolTip("시나리오 이름입니다. --tracks 에 그대로 넣을 수 있습니다.")
        map_card.add(self.scenario_line)
        v.addWidget(map_card)

        # -- shape of the session
        cfg_card = Card("구성  ·  선택 사항")
        self.spin_races = QtWidgets.QSpinBox()
        self.spin_races.setRange(1, 256)
        self.spin_races.setValue(1)
        self.spin_races.valueChanged.connect(self._update_car_math)
        self.row_races = FieldRow("레이스 수", self.spin_races,
                                  "서로 독립된 병렬 주행. 같은 레이스가 아니면 부딪히지 않습니다.")
        cfg_card.add(self.row_races)

        self.spin_grid = QtWidgets.QSpinBox()
        self.spin_grid.setRange(1, 8)
        self.spin_grid.setValue(1)
        self.spin_grid.valueChanged.connect(self._update_car_math)
        self.row_grid = FieldRow("레이스당 차량 수", self.spin_grid,
                                 "1 = 단독주행. 2 이상이면 같은 트랙에서 상대차와 겨룹니다.")
        cfg_card.add(self.row_grid)

        self.car_math = label("", "hint")
        self.car_math.setObjectName("Mono")
        cfg_card.add(self.car_math)

        self.spin_cap = QtWidgets.QDoubleSpinBox()
        self.spin_cap.setRange(0.5, 12.0)
        self.spin_cap.setSingleStep(0.5)
        self.spin_cap.setDecimals(1)
        self.spin_cap.setSuffix(" m/s")
        self.spin_cap.setValue(6.0)
        self.row_cap = FieldRow("속도 상한", self.spin_cap, "체크포인트가 학습된 상한을 따릅니다.")
        cfg_card.add(self.row_cap)

        # -- surface friction: random per car (what training saw) or one pinned value, changeable
        # while the session runs. The panel's "시뮬 참값 μ" shows what the car actually has.
        mu_row = QtWidgets.QHBoxLayout()
        mu_row.setSpacing(SP[0])
        self.combo_mu = QtWidgets.QComboBox()
        self.combo_mu.addItem("랜덤 (학습과 동일)", "random")
        self.combo_mu.addItem("고정", "fixed")
        self.combo_mu.currentIndexChanged.connect(self._on_mu_mode)
        mu_row.addWidget(self.combo_mu, 1)
        self.spin_mu = QtWidgets.QDoubleSpinBox()
        self.spin_mu.setRange(0.40, 1.40)
        self.spin_mu.setDecimals(3)
        self.spin_mu.setSingleStep(0.01)
        self.spin_mu.setValue(1.049)
        self.spin_mu.setEnabled(False)
        mu_row.addWidget(self.spin_mu, 1)
        self.btn_mu_apply = PendingButton("적용")
        self.btn_mu_apply.setToolTip("주행 중인 세션의 모든 차량에 지금 적용합니다.")
        self.btn_mu_apply.setEnabled(False)
        self.btn_mu_apply.clicked.connect(self._on_mu_apply)
        mu_row.addWidget(self.btn_mu_apply)
        mu_box = QtWidgets.QWidget()
        mu_box.setLayout(mu_row)
        self.row_mu = FieldRow("노면 마찰 μ", mu_box,
                               "학습 범위 0.734~1.154, 공칭 1.049. 고정값은 리셋 뒤에도 유지되고 주행 중 바꿀 수 있습니다.")
        cfg_card.add(self.row_mu)

        # -- grip dial: for a conditional ("dial") checkpoint, the friction the POLICY IS TOLD to
        # use, which is not the friction the floor has. Friction cannot be read from the car's
        # sensors at a pace it survives, so a dial policy is given the number instead; a dial under
        # the real friction is slow and safe, and over it is neither. Hidden for a checkpoint that
        # has no such input -- a control that does nothing is worse than no control.
        dial_row = QtWidgets.QHBoxLayout()
        dial_row.setSpacing(SP[0])
        self.spin_dial = QtWidgets.QDoubleSpinBox()
        self.spin_dial.setRange(0.40, 1.40)
        self.spin_dial.setDecimals(3)
        self.spin_dial.setSingleStep(0.01)
        self.spin_dial.setValue(0.734)
        dial_row.addWidget(self.spin_dial, 1)
        self.btn_dial_apply = PendingButton("적용")
        self.btn_dial_apply.setToolTip("주행 중인 정책이 쓰는 그립 값을 지금 바꿉니다.")
        self.btn_dial_apply.setEnabled(False)
        self.btn_dial_apply.clicked.connect(self._on_dial_apply)
        dial_row.addWidget(self.btn_dial_apply)
        dial_box = QtWidgets.QWidget()
        dial_box.setLayout(dial_row)
        self.row_dial = FieldRow("그립 다이얼", dial_box,
                                 "정책에게 '이만큼의 그립을 써라' 라고 알려 주는 값입니다. 실제 노면보다 낮게 잡으면 "
                                 "느리지만 안전하고, 높게 잡으면 빠르지도 안전하지도 않습니다.")
        self.row_dial.setVisible(False)             # shown when a dial checkpoint is running
        cfg_card.add(self.row_dial)
        v.addWidget(cfg_card)

        # -- recording. Folded, with its button on the fold's own header row: pressing 녹화 stays
        # one click, and the seven settings behind it stop standing between 구성 and 고급 설정 on
        # the panel a first-time user scrolls. (Isaac Sim's property panel does the same thing with
        # collapsable frames -- everything is there, almost nothing is open.)
        v.addWidget(self._build_record_card())

        # -- advanced
        adv = Collapsible("고급 설정", expanded=False)
        self.adv_fold = adv
        adv.toggle.toggled.connect(lambda _on: self._queue_panel_cap())
        self.chk_dr = QtWidgets.QCheckBox("차량마다 마찰/지연 무작위화 (학습과 동일)")
        self.chk_dr.setChecked(True)
        self.chk_dr.setToolTip("끄면 모든 차가 공칭 파라미터로 달립니다. 미끄러짐이 정책 탓인지 "
                               "낮은 마찰 뽑기 탓인지 구분할 때 끄세요.")
        adv.add(self.chk_dr)
        self.chk_soft = QtWidgets.QCheckBox("충돌해도 계속 주행 (soft contact)")
        self.chk_soft.setChecked(True)
        self.chk_soft.setToolTip(
            "켜면 벽·덕트·장애물에 닿아도 리셋하지 않고 접촉을 해석합니다: 덕트는 늘어나며 차를 "
            "천천히 세우고(실측 56개 충돌로 보정), 장애물은 밀리고, IMU·서스펜션이 충격을 느끼며, "
            "박힌 차는 정지 명령으로 후진해 빠져나옵니다.\n"
            "끄면 학습·평가 기본값(terminate)처럼 닿는 순간 그 차를 리셋합니다.")
        adv.add(self.chk_soft)
        self.chk_stoch = QtWidgets.QCheckBox("학습처럼 행동을 샘플링")
        adv.add(self.chk_stoch)
        # The per-car table, in place of the one combo that used to say the same thing about every
        # other car at once. Same widget as the training page's, so a session and a training command
        # cannot describe an opponent differently.
        self.opp_table = OpponentSlotTable()
        self.opp_table.set_count(max(0, self.spin_grid.value() - 1))
        self.opp_table.changed.connect(self._on_slots_changed)
        self.row_opp = FieldRow(
            "상대차 (차량별 설정)", self.opp_table,
            "레이스당 차량 수 - 1 줄. 줄마다 종류·체크포인트·속도·이벤트·스폰을 따로 정합니다. "
            "teacher 는 선택입니다: 'raceline teacher' 는 이 뷰어가 늘 쓰던 것, "
            "'interactive teacher' 는 상대 차의 예측된 미래까지 보고 계획을 고르는 업그레이드된 쪽입니다 "
            "(프리셋 '업그레이드 teacher').")
        adv.add(self.row_opp)
        self.combo_device = QtWidgets.QComboBox()
        # Every card by name, not just "cuda": `torch.device("cuda")` is device 0, and on a machine
        # where device 0 is training that is exactly the one the viewer must not take. "auto" picks
        # whichever has the most memory free, which is the same answer without having to know.
        self.combo_device.addItems(["auto"] + _cuda_device_names() + ["cpu"])
        adv.add(FieldRow("연산 장치", self.combo_device,
                         "auto: 메모리가 가장 많이 남은 GPU 를 고릅니다 (학습 중인 카드를 피합니다)."))
        self.grip_note = label("노면 한계 자동 추정", "hint")
        self.grip_note.setWordWrap(True)
        self.grip_note.setToolTip("일반 주행은 센서 기반 자동 런타임을 사용합니다. 실험용 제어기와 추정기 경로는 "
                                  "명시적인 벤치마크·디버그 호출에서만 지정합니다.")
        adv.add(self.grip_note)
        self.combo_ros = QtWidgets.QComboBox()
        self.combo_ros.addItem("끄기", "off")
        self.combo_ros.addItem("센서 토픽 발행 (정책이 주행)", "publish")
        self.combo_ros.addItem("센서 발행 + /drive 로 외부 제어", "drive")
        adv.add(FieldRow("ROS2 연동", self.combo_ros,
                         "차량 0 의 /scan /odom /sensors/imu 와 장면 전체(/f1sim/viz/*)를 ROS2 토픽으로 냅니다. "
                         "'외부 제어'는 차량 0 을 /drive 구독으로 움직입니다 (예: ros2 launch f1sim_ros "
                         "pure_pursuit.launch.py). ROS 워크스페이스를 소싱한 셸에서 콘솔을 열어야 합니다."))
        v.addWidget(adv)

        v.addStretch(1)

        # -- primary actions, pinned under the scroll area so they are always reachable
        wrap = QtWidgets.QWidget()
        wrap.setObjectName("Panel")
        wv = QtWidgets.QVBoxLayout(wrap)
        wv.setContentsMargins(0, 0, 0, 0)
        wv.setSpacing(0)
        wv.addWidget(area, 1)
        bar = QtWidgets.QFrame()
        bar.setObjectName("Panel")
        bar_v = QtWidgets.QVBoxLayout(bar)
        bar_v.setContentsMargins(SP[2], SP[1], SP[2], SP[2])
        bar_v.setSpacing(SP[0])
        self.start_hint = label("", "hint")
        self.start_hint.setWordWrap(True)
        bar_v.addWidget(self.start_hint)
        bh = QtWidgets.QHBoxLayout()
        bh.setContentsMargins(0, 0, 0, 0)
        bh.setSpacing(SP[1])
        bar_v.addLayout(bh)
        self.btn_start = PendingButton("시작", role="PrimaryButton")
        self.btn_start.setToolTip("선택한 런과 맵으로 세션을 시작합니다.")
        self.btn_start.clicked.connect(self._on_start)
        bh.addWidget(self.btn_start, 2)
        self.btn_cancel = PendingButton("취소")
        self.btn_cancel.setToolTip("준비 중인 세션을 중단합니다. 준비 중에도 계속 누를 수 있습니다.")
        self.btn_cancel.clicked.connect(self._on_cancel)
        bh.addWidget(self.btn_cancel, 1)
        wv.addWidget(bar)
        wrap.setMaximumWidth(self.PANEL_W)
        self._left_wrap = wrap
        return wrap

    def _build_record_card(self) -> QtWidgets.QWidget:
        """The 녹화 form: where the clip goes, how big, how fast, and what is in it.

        Recording renders the same frames the session is showing into an offscreen buffer at the
        size chosen here, so the window can be any size and the clip is still 1080p. Encoding runs
        on its own thread behind a bounded queue: a frame the encoder cannot take is dropped and
        counted rather than made to wait, because the thread being protected is the one painting the
        window.
        """
        card = Collapsible("녹화", expanded=False)
        self.record_fold = card
        self.edit_record = QtWidgets.QLineEdit()
        self.edit_record.setPlaceholderText(REC.VIDEO_DIR)
        self.edit_record.setToolTip("저장 폴더. 파일 이름은 맵과 시각으로 자동으로 붙습니다.")
        browse = QtWidgets.QPushButton("…")
        browse.setObjectName("GhostButton")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._on_record_dir)
        dir_row = QtWidgets.QHBoxLayout(); dir_row.setSpacing(SP[0])
        dir_row.addWidget(self.edit_record, 1); dir_row.addWidget(browse)
        dir_box = QtWidgets.QWidget(); dir_box.setLayout(dir_row)
        card.add(FieldRow("저장 폴더", dir_box, f"비우면 {REC.VIDEO_DIR}"))

        self.combo_res = QtWidgets.QComboBox()
        for label_, w_, h_ in REC.RESOLUTIONS:
            self.combo_res.addItem(label_, (w_, h_))
        self.combo_res.setToolTip("녹화 해상도. 창 크기와 무관하게 이 크기로 따로 렌더링합니다.")
        self.combo_fps = QtWidgets.QComboBox()
        for f_ in REC.FPS_CHOICES:
            self.combo_fps.addItem(f"{f_} fps", f_)
        self.combo_fps.setCurrentIndex(list(REC.FPS_CHOICES).index(30))
        g = QtWidgets.QGridLayout(); g.setHorizontalSpacing(SP[1]); g.setVerticalSpacing(SP[0])
        g.addWidget(FieldRow("해상도", self.combo_res, ""), 0, 0)
        g.addWidget(FieldRow("프레임", self.combo_fps, ""), 0, 1)
        card.add(g)

        self.combo_rec_cam = QtWidgets.QComboBox()
        self.combo_rec_cam.addItem("현재 카메라", "")
        for key, text, tip in CAMERA_MODES:
            self.combo_rec_cam.addItem(text, key)
            self.combo_rec_cam.setItemData(self.combo_rec_cam.count() - 1, tip, QtCore.Qt.ToolTipRole)
        self.combo_rec_cam.setToolTip("녹화만 이 카메라로 찍습니다. 화면은 그대로 둡니다.")
        self.spin_rec_secs = QtWidgets.QDoubleSpinBox()
        self.spin_rec_secs.setRange(0.0, 3600.0); self.spin_rec_secs.setDecimals(0)
        self.spin_rec_secs.setSingleStep(5.0); self.spin_rec_secs.setSuffix(" 초")
        self.spin_rec_secs.setSpecialValueText("수동 정지")
        self.spin_rec_secs.setToolTip("0 이면 다시 누를 때까지 계속 녹화합니다.")
        g2 = QtWidgets.QGridLayout(); g2.setHorizontalSpacing(SP[1])
        g2.addWidget(FieldRow("카메라", self.combo_rec_cam, ""), 0, 0)
        g2.addWidget(FieldRow("길이", self.spin_rec_secs, ""), 0, 1)
        card.add(g2)

        self.chk_rec_overlay = QtWidgets.QCheckBox("오버레이 포함 (LiDAR 점·레이싱 라인·차량 라벨)")
        self.chk_rec_overlay.setChecked(True)
        self.chk_rec_overlay.setToolTip("끄면 주행만 담긴 깨끗한 영상이 됩니다. 화면 표시는 그대로입니다.")
        card.add(self.chk_rec_overlay)

        self.combo_rec_enc = QtWidgets.QComboBox()
        for key in REC.ENCODERS:
            self.combo_rec_enc.addItem(REC.ENCODER_LABEL[key], key)
        self.combo_rec_enc.setToolTip(
            "자동: GPU 인코더(h264_nvenc)가 실제로 되는 기계면 그걸 쓰고, 안 되면 libx264 로 돌아갑니다.\n"
            "어느 쪽이 쓰였는지는 끝난 파일 줄에 적힙니다. 렌더링은 세션이 쓰는 GL 그대로입니다.")
        card.add(FieldRow("인코더", self.combo_rec_enc, ""))

        self.btn_record = PendingToggle("● 녹화 시작")
        self.btn_record.setToolTip("3D 화면을 mp4 로 저장합니다.  (R)")
        self.btn_record.requested.connect(lambda _want: self._on_record_toggle())
        card.add_header(self.btn_record)
        self.record_note = label("", "hint")
        self.record_note.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.record_note.setOpenExternalLinks(False)
        self.record_note.linkActivated.connect(self._on_open_video)
        card.add(self.record_note)
        if not REC.ffmpeg_available():
            self.btn_record.setEnabled(False)
            self.record_note.setText("ffmpeg 이 없어 녹화할 수 없습니다 (apt install ffmpeg).")
        #: The recording in progress, and a 4 Hz timer that shows how it is going and stops it when
        #: the requested length is reached. The recorder itself is driven by the viewport's paint.
        self._recorder = None
        self._rec_timer = QtCore.QTimer(self)
        self._rec_timer.setInterval(250)
        self._rec_timer.timeout.connect(self._record_tick)
        self._last_video = ""
        return card

    # ---------------------------------------------------------------- recording
    def record_settings(self) -> "REC.RecordSpec":
        """The form as a spec, with the path filled in from the map and the clock."""
        w, h = self.combo_res.currentData() or (1280, 720)
        spec = REC.RecordSpec(path="", width=int(w), height=int(h),
                              fps=int(self.combo_fps.currentData() or 30),
                              camera=str(self.combo_rec_cam.currentData() or ""),
                              overlays=self.chk_rec_overlay.isChecked(),
                              seconds=float(self.spin_rec_secs.value()),
                              encoder=str(self.combo_rec_enc.currentData() or "auto"))
        spec = spec.resolved(self.viewport.fb_size())
        import dataclasses
        import os
        name = os.path.basename(REC.default_video_path(self._running_scenario()))
        folder = self.edit_record.text().strip() or REC.VIDEO_DIR
        return dataclasses.replace(spec, path=os.path.join(os.path.expanduser(folder), name))

    def _running_scenario(self) -> str:
        f = self._running_facts or {}
        return str(f.get("scenario") or f.get("map") or self._scenario() or "f1sim")

    @property
    def recording(self) -> bool:
        return self._recorder is not None

    def _on_record_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "녹화 저장 폴더", self.edit_record.text().strip() or REC.VIDEO_DIR)
        if d:
            self.edit_record.setText(d)

    def _on_record_toggle(self):
        if self._recorder is not None:
            self._stop_recording("수동 정지")
            return
        if self.state not in (STATE_RUNNING, STATE_PAUSED):
            self.record_note.setText("세션이 돌고 있을 때만 녹화할 수 있습니다.")
            self.btn_record.settle(False)
            return
        spec = self.record_settings()
        rec = REC.VideoRecorder(spec)
        if not rec.ok:
            self.record_note.setObjectName("HintWarn")
            self.record_note.setText(rec.error or "녹화를 시작할 수 없습니다.")
            self.btn_record.settle(False)
            return
        self._recorder = rec
        self.viewport.start_recording(rec)
        self.viewport.notify(f"● 녹화 시작 — {os.path.basename(spec.path)}", "danger")
        # The progress line lives inside the fold; open it so a recording in progress is visible.
        if hasattr(self, "record_fold"):
            self.record_fold.toggle.setChecked(True)
        self._rec_timer.start()
        self.btn_record.setText("■ 녹화 중지")
        self.btn_record.settle(True)
        self.record_note.setObjectName("Hint")
        self.record_note.setText(f"녹화 중 — {os.path.basename(spec.path)}")
        self._update_rec_badge()

    def _stop_recording(self, why: str = ""):
        rec, self._recorder = self._recorder, None
        self.viewport.stop_recording()
        self._rec_timer.stop()
        self.btn_record.setText("● 녹화 시작")
        self.btn_record.settle(False)
        self._update_rec_badge()
        if rec is None:
            return
        rec.stop()
        self._last_video = rec.spec.path
        kind = "HintWarn" if rec.error else "Hint"
        self.record_note.setObjectName(kind)
        if rec.error:
            self.record_note.setText(rec.summary())
        else:
            self.record_note.setText(f"{rec.summary()}   <a href='#open'>열기</a>")
        self.record_note.style().unpolish(self.record_note)
        self.record_note.style().polish(self.record_note)
        self.status_text.setText(f"녹화 저장: {rec.spec.path}" + (f" ({why})" if why else ""))
        self.viewport.notify(("녹화 오류 — " + (rec.error or "")) if rec.error
                             else f"■ 녹화 저장 — {os.path.basename(rec.spec.path)}",
                             "danger" if rec.error else "good")

    def _record_tick(self):
        rec = self._recorder
        if rec is None:
            return
        if rec.error:
            self._stop_recording("오류")
            return
        drops = f" · 버린 프레임 {rec.dropped}" if rec.dropped else ""
        self.record_note.setText(f"녹화 중 {rec.duration:.1f}초 · {rec.written}프레임{drops}")
        if rec.spec.seconds and rec.duration >= rec.spec.seconds:
            self._stop_recording(f"{rec.spec.seconds:.0f}초 도달")

    def _on_open_video(self, _href: str = ""):
        if not self._last_video or not os.path.isfile(self._last_video):
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._last_video))

    def _update_rec_badge(self):
        """The header says so while a clip is being written. It is a file being created on someone's
        disk; a UI that does that silently is a UI that fills a disk silently."""
        if hasattr(self, "header_rec"):
            self.header_rec.setVisible(self.recording)

    # ---------------------------------------------------------------- centre: the picture
    def _build_centre(self) -> QtWidgets.QWidget:
        centre = QtWidgets.QWidget()
        centre.setObjectName("Centre")
        v = QtWidgets.QVBoxLayout(centre)
        v.setContentsMargins(SP[1], SP[1], SP[1], SP[1])
        v.setSpacing(SP[1])

        self.viewport = ViewportWidget()
        v.addWidget(self.viewport, 1)

        # -- control bar. Groups of related controls, laid out with a wrapping flow so a narrow
        # window moves a group to the next line instead of shaving the text off its buttons.
        bar = QtWidgets.QFrame()
        bar.setObjectName("Card")
        flow = FlowLayout(bar, margin=SP[1], h_spacing=SP[2], v_spacing=SP[1])

        def group() -> Tuple[QtWidgets.QWidget, QtWidgets.QHBoxLayout]:
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(SP[0])
            flow.addWidget(w)
            return w, lay

        _, h = group()

        def sep():
            line = QtWidgets.QFrame()
            line.setObjectName("BarSep")
            line.setFrameShape(QtWidgets.QFrame.VLine)
            line.setFixedWidth(1)
            return line

        self.btn_pause = PendingToggle("일시정지")
        self.btn_pause.setToolTip("시뮬레이션을 멈춥니다. 화면과 조작은 계속 살아 있습니다.  (Space)")
        self.btn_pause.requested.connect(self._on_pause_requested)
        h.addWidget(self.btn_pause)

        self.btn_reset = PendingButton("리셋")
        self.btn_reset.setToolTip("같은 맵에서 에피소드를 처음부터 다시 시작합니다.  (Ctrl+R)")
        self.btn_reset.clicked.connect(self._on_reset)
        h.addWidget(self.btn_reset)

        self.btn_stop = PendingButton("정지", role="DangerButton")
        self.btn_stop.setToolTip("세션을 끝내고 worker를 정리합니다.  (Ctrl+.)")
        self.btn_stop.clicked.connect(self._on_stop)
        h.addWidget(self.btn_stop)

        _, h = group()
        h.addWidget(sep())
        h.addWidget(label("카메라", "field"))
        self.camera_buttons = SegmentedButtons(CAMERA_MODES)
        self.camera_buttons.set_current("overview")
        self.camera_buttons.selected.connect(self._on_camera)
        h.addWidget(self.camera_buttons)

        _, h = group()
        h.addWidget(sep())
        h.addWidget(label("주시 차량", "field"))
        self.btn_focus_prev = QtWidgets.QPushButton("◀")
        self.btn_focus_prev.setFixedWidth(32)
        self.btn_focus_prev.setToolTip("이전 차량  ( [ )")
        self.btn_focus_prev.clicked.connect(lambda: self._step_focus(-1))
        h.addWidget(self.btn_focus_prev)
        self.combo_focus = QtWidgets.QComboBox()
        self.combo_focus.setMinimumWidth(96)
        self.combo_focus.setToolTip("계기와 정책 패널이 설명하는 차량. 3D에서 하늘색으로 표시됩니다.")
        self.combo_focus.activated.connect(self._on_focus_combo)
        h.addWidget(self.combo_focus)
        self.btn_focus_next = QtWidgets.QPushButton("▶")
        self.btn_focus_next.setFixedWidth(32)
        self.btn_focus_next.setToolTip("다음 차량  ( ] )")
        self.btn_focus_next.clicked.connect(lambda: self._step_focus(1))
        h.addWidget(self.btn_focus_next)

        _, h = group()
        self.btn_shot = QtWidgets.QPushButton("스크린샷")
        self.btn_shot.setObjectName("GhostButton")
        h.addWidget(sep())
        self.btn_shot.setToolTip("3D 화면을 PNG로 저장합니다. 왼쪽 '녹화' 카드의 해상도로 찍습니다.  (S)")
        self.btn_shot.clicked.connect(self._on_screenshot)
        h.addWidget(self.btn_shot)
        # The same action as the card's button, within reach of the picture it films.
        self.btn_record_bar = QtWidgets.QPushButton("● 녹화")
        self.btn_record_bar.setObjectName("GhostButton")
        self.btn_record_bar.setToolTip("녹화 시작 / 중지. 설정은 왼쪽 '녹화' 카드에 있습니다.  (R)")
        self.btn_record_bar.clicked.connect(self._on_record_toggle)
        h.addWidget(self.btn_record_bar)
        v.addWidget(bar)

        # -- policy panels, foldable so they never crowd the picture
        # The row is genuinely mixed and the title has to say so: the left panel is the actor's
        # own observation and its action, but the g-g trace beside it is the simulator's measured
        # acceleration, which the actor never sees. Calling the whole row "정책이 보는 것"
        # invited reading the friction circle as something the policy estimates.
        self.policy_fold = Collapsible("정책 입·출력과 시뮬 참값", expanded=True)
        self.policy_fold.setToolTip(
            "왼쪽 BEV는 정책이 실제로 받는 관측(스캔)과 정책이 내는 행동입니다.\n"
            "오른쪽 계기 중 g-g는 시뮬레이터가 측정한 가속도(시뮬 참값)이며, 정책의 관측이\n"
            "아닙니다. 정책이 접지력을 추정한 결과로 읽지 마세요.")
        self.policy_fold.add_header(label("스캔 반경", "field"))
        self.combo_span = QtWidgets.QComboBox()
        # A 1 m-wide corridor inside an 8 m frame is a few pixels of signal in a lot of empty
        # circle. The scale stays a fixed, labelled number rather than auto-fitting, because the
        # whole point of the panel is that a distance read off it is the distance.
        for m in (2, 3, 4, 6, 8, 10):
            self.combo_span.addItem(f"{m} m", float(m))
        self.combo_span.setCurrentIndex(2)
        self.combo_span.setMaximumWidth(90)
        self.combo_span.setToolTip("패널의 반경(실제 거리). 좁은 트랙에서는 3–4 m가 읽기 좋습니다.")
        self.combo_span.currentIndexChanged.connect(
            lambda _: (setattr(self.policy_panel, "span", float(self.combo_span.currentData())),
                       self.policy_panel.update()))
        self.policy_fold.add_header(self.combo_span)
        # apply it once now: a control that reads "4 m" while the panel is drawing 8 m is a lie,
        # and setCurrentIndex before connect() fires nothing
        self.policy_panel_span_sync = lambda: (
            setattr(self.policy_panel, "span", float(self.combo_span.currentData())),
            self.policy_panel.update())
        # The two panels also wrap: side by side when there is room, stacked when there is not.
        # Squeezing three gauges into 100 px each elides every label, which is a panel that costs
        # space and answers nothing.
        row_host = QtWidgets.QWidget()
        self._policy_flow = FlowLayout(row_host, margin=0, h_spacing=SP[1], v_spacing=SP[1])
        self.policy_panel = PolicyInputPanel()
        # sized so that, together with three gauges at their readable minimum, the pair still
        # fits on one row inside the narrowest supported window rather than needing a scroll
        self.policy_panel.setFixedSize(216, 182)
        self._policy_flow.addWidget(self.policy_panel)
        self.dash_panel = DashPanel()
        self.dash_panel.setProperty("flowExpand", True)     # take the row's spare width
        self._policy_flow.addWidget(self.dash_panel)
        self.policy_panel_span_sync()
        self.policy_fold.add(row_host)
        self.policy_fold.toggle.toggled.connect(
            lambda _: (self._apply_responsive(), self._queue_panel_cap()))
        self.activation_panel = ActivationPanel()
        self.activation_panel.setVisible(False)
        self.policy_fold.add(self.activation_panel)

        # The analysis panels scroll rather than push the picture off the screen. On a small window
        # the wrapped dash lands on a second row, and without this that row simply was not there --
        # the panel was open, and half of it was outside the widget. Scrolling is the honest
        # alternative to shrinking gauges until their labels elide.
        self.policy_scroll = QtWidgets.QScrollArea()
        self.policy_scroll.setObjectName("PanelScroll")
        self.policy_scroll.setWidgetResizable(True)
        self.policy_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.policy_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.policy_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        self.policy_scroll.setWidget(self.policy_fold)
        v.addWidget(self.policy_scroll)
        self.viewport.setMinimumHeight(240)
        return centre

    #: The setup sidebar's width. `PANEL_W_SLOTS` is what it grows to while the 상대차 table has
    #: rows: ten fixed-width columns do not fit 420 px, and a table one column wide is a table
    #: nobody can read. It only grows when there is a table, and only while the window has the room
    #: (`_panel_width`), so a narrow screen keeps the picture.
    PANEL_W = 420
    PANEL_W_SLOTS = 640
    #: What the 3D view keeps when the sidebar grows for the table.
    CENTRE_MIN_W = 600

    #: Below this width the two sidebars leave the 3D view too small to drive by, so the setup
    #: sidebar -- which is only needed between sessions -- folds away once a session is running.
    NARROW_W = 1400

    #: Below this height the policy panels would take a third of the picture they annotate.
    SHORT_H = 820

    def closeEvent(self, ev):
        """Closing asks the controller to wind the worker down; it does not wait for it here.

        Blocking a close handler is how a window ends up as a grey rectangle the desktop offers to
        force-quit. Instead the close is refused once, the controller runs its staged shutdown on a
        timer with the status bar narrating it, and `allow_close()` closes for real when the worker
        is actually gone.
        """
        self.save_prefs()
        if self._closing_allowed:
            ev.accept()
            return
        ev.ignore()
        self._closing_allowed = False
        self.close_requested.emit()

    def allow_close(self):
        self._closing_allowed = True
        self.viewport.teardown()      # while the context still exists
        try:
            self.editor.shutdown()
            self.editor.viewport.teardown()
        except Exception:
            pass
        self.close()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._apply_responsive()
        self._queue_panel_cap()

    def _cap_policy_panel(self):
        """Hold the diagnostic panel to the height its content actually needs at this width.

        The 3D view takes the slack, which is the point: `sizeHint()` is not width-aware -- the two
        panels sit in a flow layout whose hint is the stacked, one-column height no matter how wide
        the row really is. At 1600x950 that asked for 414 px to show 228 px of content, so 186 px
        of empty space sat under the gauges and the picture was handed 358 px instead of 544.

        `heightForWidth` asks the flow what it will do at a given width, and returns -1 when the
        fold is collapsed and its content hidden -- hence the fallback. It also needs the width to
        be settled, which inside `resizeEvent` it is not, so this is additionally re-run once the
        layout pass is over (see `_queue_panel_cap`).
        """
        if not hasattr(self, "policy_scroll"):
            return
        if not self.policy_fold.toggle.isChecked():
            self.policy_scroll.setMaximumHeight(40)
            return
        # Never more than 45% of the window and never so tall the viewport is squeezed out. When
        # the content is taller than that -- the narrow window where the panels stack -- the cap
        # wins and the area scrolls, rather than clipping the second row away.
        cap = max(120, int(min(self.height() * 0.45, self.height() - 400)))
        w = max(1, self.policy_scroll.viewport().width())
        h4w = self.policy_fold.heightForWidth(w)
        want = (h4w if h4w > 0 else self.policy_fold.sizeHint().height()) + 6
        self.policy_scroll.setMaximumHeight(min(cap, want))

    def _queue_panel_cap(self):
        """Re-cap once this layout pass has finished and the widths are real. Coalesced."""
        if getattr(self, "_cap_queued", False):
            return
        self._cap_queued = True

        def run():
            self._cap_queued = False
            self._cap_policy_panel()
            # Widths are only real after the layout pass, and the sidebar's width depends on the
            # splitter's: asking for it inside `resizeEvent` reads zeros on the first show.
            self._apply_panel_width()

        QtCore.QTimer.singleShot(0, run)

    def _apply_responsive(self):
        """Give the 3D view the room, without ever hiding a control for good.

        Each rule fires once and sets a flag, so a fold the user undoes by hand stays undone. The
        header buttons bring either sidebar back at any size, and nothing here removes a control
        from the tab order permanently -- it is layout, not capability.
        """
        if not hasattr(self, "policy_fold"):
            return
        if (not self._policy_autofold_done and self.height() < self.SHORT_H
                and self.policy_fold.toggle.isChecked()):
            self.policy_fold.toggle.setChecked(False)
            self._policy_autofold_done = True
        self._cap_policy_panel()
        self._apply_panel_width()
        driving = self.state in (STATE_RUNNING, STATE_PAUSED)
        if (not self._sidebar_autofold_done and driving and self.width() < self.NARROW_W
                and self.btn_left_panel.isChecked()):
            self.btn_left_panel.setChecked(False)
            self._sidebar_autofold_done = True
            self.status_text.setText(
                "창이 좁아 설정 패널을 접었습니다. 위의 '설정 패널' 버튼으로 다시 펼 수 있습니다.")

    def _apply_panel_width(self):
        """Widen the setup sidebar while the 상대차 table has rows, if the window can spare it.

        The table is ten fixed-width columns; at the sidebar's usual 330 px three of them are
        visible and the rest is a horizontal scrollbar, which is not a table anyone can read. So the
        sidebar grows when there is a table to show and shrinks back when there is not -- and only
        while the window is wide enough that the 3D view keeps 900 px, because the picture is what
        the page is for. A splitter the user has dragged wider than this is left alone.
        """
        if not hasattr(self, "_left_wrap") or not hasattr(self, "opp_table"):
            return
        want = self.PANEL_W
        showing_slots = self.row_opp.isVisible() and self.opp_table.count()
        if showing_slots and self.width() >= self.NARROW_W:
            want = min(self.PANEL_W_SLOTS, max(self.PANEL_W, self.width() - 900))
        self._left_wrap.setMaximumWidth(want)
        self._left_scroll.setMaximumWidth(want)
        if self._panel_nudged_for == want or not self.isVisible():
            # Already offered this width once (a later drag is the user's), or the window has not
            # been laid out yet -- a splitter re-distributes on first show, so a nudge before that
            # is simply discarded.
            return
        sizes = self.splitter.sizes()
        if len(sizes) != 3 or not sizes[0] or sizes[1] <= 0:
            return
        self._panel_nudged_for = want
        delta = min(want - sizes[0], max(0, sizes[1] - self.CENTRE_MIN_W))
        if delta > 0:
            self.splitter.setSizes([sizes[0] + delta, sizes[1] - delta, sizes[2]])

    # ---------------------------------------------------------------- right: is this real
    def _build_right(self) -> QtWidgets.QWidget:
        area, v = _scroll_panel(360)

        drive = Card("주행")
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(SP[2])
        grid.setVerticalSpacing(SP[1])
        self.m_speed = MetricTile("속도", "m/s")
        self.m_lap = MetricTile("랩", "", small=True)
        self.m_coll = MetricTile("충돌", "대", small=True)
        self.m_wall = MetricTile("가장 가까운 벽", "m", small=True)
        grid.addWidget(self.m_speed, 0, 0)
        grid.addWidget(self.m_lap, 0, 1)
        grid.addWidget(self.m_coll, 1, 0)
        grid.addWidget(self.m_wall, 1, 1)
        drive.add(grid)
        drive.add(hline())
        self.drive_info = KeyValueList()
        self.drive_info.set("조향", "—")
        self.drive_info.set("자세 roll/pitch", "—")
        self.drive_info.set("랩 진행", "—")
        self.drive_info.set("시뮬 참값 μ", "—",
                            tooltip="시뮬레이터가 이 차에 실제로 적용한 마찰계수입니다. 정책이 "
                                    "관측할 수 없는 값이며, 추정치가 아니라 참값입니다.")
        self.drive_info.set("추정 μ", "추정 대기",
                            tooltip="센서 기반 추정의 q50(중앙값)과 컨트롤러에 적용한 보수적 상한 "
                                    "used_mu입니다. 시뮬 참값 μ와 다를 수 있습니다.")
        drive.add(self.drive_info)
        v.addWidget(drive)

        # Two numbers by default, because two numbers are what someone actually needs to know
        # whether the picture can be trusted: how fast the simulation is really going, and how fast
        # this window is really drawing. The rest is real and kept, but folded away -- a panel of
        # a dozen counters reads as an engineering log, and then nobody reads any of it.
        run_card = Card("실행 상태")
        tg = QtWidgets.QGridLayout()
        tg.setHorizontalSpacing(SP[2])
        tg.setVerticalSpacing(SP[1])
        self.m_simrate = MetricTile("sim 배속", "× 실시간", small=True,
                                    tooltip="worker가 잰 값: 실제 1초 동안 시뮬레이션이 나아간 초. "
                                            "1.00이면 실시간입니다.")
        self.m_fps = MetricTile("렌더 fps", "fps", small=True,
                                tooltip="실제 처리량입니다: 최근 구간에서 완료한 프레임 수 ÷ 그 시간.\n"
                                        "간격의 중앙값을 역수 취한 값이 아닙니다 — 1,1,98 ms 세 프레임은 "
                                        "30 fps이지 1000 fps가 아닙니다.\n"
                                        "분포는 아래 '프레임 간격' p50/p95/최대를 함께 보세요.")
        tg.addWidget(self.m_simrate, 0, 0)
        tg.addWidget(self.m_fps, 0, 1)
        run_card.add(tg)
        # The distinction between these two numbers matters, but it is a tooltip, not a paragraph
        # that costs three lines of a small window on every frame.
        for tile in (self.m_simrate, self.m_fps):
            tile.setToolTip(tile.toolTip() + "\n\nsim 배속과 렌더 fps는 서로 다른 것을 잽니다. "
                                             "렌더가 빨라도 물리가 빨라진 것이 아닙니다.")
        self.health_line = label("—", "hint")
        run_card.add(self.health_line)

        diag = Collapsible("상세 진단", expanded=False)
        self.spark = Sparkbar()
        diag.add(self.spark)
        self.frame_stats = label("프레임 간격 —", "hint")
        self.frame_stats.setObjectName("Mono")
        self.frame_stats.setToolTip("최근 유지 중인 간격들의 분포 (최대 120개). 처리량이 아니라 분포이며, "
                                    "위의 렌더 fps와 함께 읽어야 버벅임이 보입니다.")
        diag.add(self.frame_stats)
        self.draw_stats = label("그리기 시간 —", "hint")
        self.draw_stats.setObjectName("Mono")
        self.draw_stats.setToolTip("한 프레임을 실제로 그리는 데 걸린 시간(paintGL 본문).\n"
                                   "간격은 큰데 이 값이 작으면 그리기가 느린 것이 아니라 "
                                   "다시 그리라는 요청이 드물게 온 것입니다.")
        diag.add(self.draw_stats)
        self.trust_info = KeyValueList()
        self.trust_info.set("데이터 도착", "—", tooltip="마지막 프레임을 받은 뒤 지난 시간")
        self.trust_info.set("데이터 생성", "—",
                            tooltip="worker가 그 프레임을 만든 뒤 지난 시간. 도착은 빠른데 이 값이 "
                                    "크면 밀린 프레임이 쌓이고 있다는 뜻입니다.")
        self.trust_info.set("seq 지연", "—", tooltip="worker의 최신 프레임 번호와 지금 그리는 번호의 차")
        self.trust_info.set("프레임 드롭", "—", tooltip="worker에서 버린 수 / GUI에서 버린 수")
        self.trust_info.set("렌더러", "—")
        diag.add(self.trust_info)
        run_card.add(diag)
        v.addWidget(run_card)

        # The console had nowhere to *keep* what the worker said: `MSG_LOG` overwrote one status
        # line and the stage name replaced itself, so a prepare that takes forty minutes -- a cold
        # raceline cache does -- looked identical to a hang, and a message that scrolled past was
        # gone. This is the history, folded away because on a normal session there is nothing in it
        # worth the space.
        logs = Collapsible("진행 로그", expanded=False)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)       # a ring buffer: old lines fall off the top
        self.log_view.setMinimumHeight(120)
        self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        # Same look as the training page's own log box, which this is the drive page's version of.
        self.log_view.setStyleSheet(f"font-family: '{theme.MONO_FONT}', monospace; "
                                    f"font-size: 10px; background: {C['bg.window']};")
        logs.add(self.log_view)
        row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("복사")
        btn_copy.setObjectName("GhostButton")
        btn_copy.setToolTip("로그 전체를 클립보드로. 오류를 그대로 붙여넣을 때 쓰세요.")
        btn_copy.clicked.connect(lambda: QtGui.QGuiApplication.clipboard().setText(
            self.log_view.toPlainText()))
        btn_clear = QtWidgets.QPushButton("지우기")
        btn_clear.setObjectName("GhostButton")
        btn_clear.clicked.connect(self.log_view.clear)
        row.addWidget(btn_copy); row.addWidget(btn_clear); row.addStretch(1)
        logs.add(row)
        self.log_panel = logs
        v.addWidget(logs)

        show = Card("표시")
        self.chk_lidar = QtWidgets.QCheckBox("LiDAR 점")
        self.chk_lidar.setChecked(True)
        self.chk_plan = QtWidgets.QCheckBox("plan / 예측 궤적")
        self.chk_plan.setChecked(True)
        self.chk_trails = QtWidgets.QCheckBox("주행 궤적")
        self.chk_trails.setChecked(True)
        self.chk_raceline = QtWidgets.QCheckBox("레이싱 라인 / 중심선")
        self.chk_raceline.setChecked(True)
        self.chk_labels = QtWidgets.QCheckBox("차량 번호")
        self.chk_labels.setChecked(True)
        for w in (self.chk_lidar, self.chk_plan, self.chk_trails, self.chk_raceline, self.chk_labels):
            show.add(w)
            w.toggled.connect(self._apply_local_overlays)
        v.addWidget(show)

        cost = Collapsible("비용이 큰 표시", expanded=False)
        cost.add(label("아래 항목은 worker에서 추가 GPU 작업을 합니다. 필요할 때만 켜세요.", "hint"))
        self.chk_saliency = QtWidgets.QCheckBox("빔 saliency (정책이 어디를 보는지)")
        self.chk_saliency.setToolTip("행동에 대한 스캔의 기울기. autograd backward가 필요합니다.")
        self.chk_saliency.toggled.connect(self._emit_overlay_request)
        cost.add(self.chk_saliency)
        self.chk_internals = QtWidgets.QCheckBox("신경망 내부 활성화")
        self.chk_internals.toggled.connect(self._on_internals)
        cost.add(self.chk_internals)
        self.chk_interp = QtWidgets.QCheckBox("프레임 보간 (부드럽게)")
        self.chk_interp.setChecked(True)
        self.chk_interp.setToolTip("시뮬 프레임 사이를 시뮬의 실측 속도로 보간합니다. "
                                   "물리를 빠르게 만들지 않으며, 끄면 받은 프레임만 그대로 그립니다.")
        self.chk_interp.toggled.connect(lambda on: setattr(self.viewport, "interpolate", on))
        cost.add(self.chk_interp)
        v.addWidget(cost)

        v.addStretch(1)
        return area

    # ---------------------------------------------------------------- status bar
    def _build_status(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setObjectName("StatusBar")
        bar.setMinimumHeight(38)
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(SP[2], SP[0], SP[2], SP[0])
        h.setSpacing(SP[2])
        self.status_text = QtWidgets.QLabel("대기")
        self.status_text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        h.addWidget(self.status_text, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        h.addWidget(self.progress)
        self.btn_retry = QtWidgets.QPushButton("다시 시도")
        self.btn_retry.setVisible(False)
        self.btn_retry.clicked.connect(self.retry_requested.emit)
        h.addWidget(self.btn_retry)
        self.btn_detail = QtWidgets.QPushButton("오류 자세히")
        self.btn_detail.setObjectName("GhostButton")
        self.btn_detail.setVisible(False)
        self.btn_detail.clicked.connect(self._show_error_detail)
        h.addWidget(self.btn_detail)
        return bar

    # ---------------------------------------------------------------- shortcuts
    def _install_shortcuts(self):
        def sc(seq: str, fn: Callable):
            s = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
            s.setContext(QtCore.Qt.ApplicationShortcut)
            s.activated.connect(fn)
            return s

        # The single-key shortcuts belong to the driving page. On the editor page those keys are
        # tool keys (S would take a screenshot while the user meant nothing of the sort), so each
        # one checks the page before acting; application context is kept so they work with focus
        # anywhere in the window.
        def drive_only(fn: Callable):
            return lambda: fn() if self.current_mode() == "drive" else None

        sc("Space", drive_only(lambda: self.btn_pause.click() if self.btn_pause.isEnabled() else None))
        for i, key in enumerate(CAMERA_KEYS):
            sc(str(i + 1), drive_only(lambda k=key: self._on_camera(k, from_shortcut=True)))
        sc("[", drive_only(lambda: self._step_focus(-1)))
        sc("]", drive_only(lambda: self._step_focus(1)))
        sc("Ctrl+R", drive_only(lambda: self.btn_reset.click() if self.btn_reset.isEnabled() else None))
        sc("Ctrl+.", drive_only(lambda: self.btn_stop.click() if self.btn_stop.isEnabled() else None))
        sc("Ctrl+F", drive_only(lambda: self.map_list.search.setFocus(QtCore.Qt.ShortcutFocusReason)))
        sc("S", drive_only(lambda: self._on_screenshot()))
        sc("R", drive_only(lambda: self._on_record_toggle()))
        sc("F1", self.show_help)
        sc("H", drive_only(self.show_help))

    # ================================================================ data in
    def set_runs(self, runs: List[RunInfo]):
        self.runs = runs
        items = [("", r.name, r.name, r.subtitle) for r in runs]
        self.run_list.set_items(items)
        if not runs:
            self.run_list.set_status(f"{catalog.RUNS_DIR} 에 체크포인트가 있는 런이 없습니다.", "warn")
        self._update_start_enabled()

    def set_maps(self, cat: MapCatalog):
        self.maps = self._with_local_scenes(cat)
        cat = self.maps
        self.editor.set_maps(cat)
        self.map_group.blockSignals(True)
        self.map_group.clear()
        if cat.error:
            self.map_group.addItem("맵 목록 실패")
            self.map_group.setEnabled(False)
            self.map_list.set_status(cat.error, "danger")
        elif not cat.ready:
            self.map_group.addItem("목록 읽는 중…")
            self.map_group.setEnabled(False)
        else:
            names = [g for g in GROUP_ORDER if g in cat.groups] + \
                    [g for g in cat.groups if g not in GROUP_ORDER]
            for g in names:
                self.map_group.addItem(f"{g}  ({len(cat.groups[g])})", g)
            self.map_group.setEnabled(True)
            # Land on the biggest group rather than whatever sorted first: an empty list is the
            # console's least useful first impression, and 내 환경 is empty until someone draws one.
            best = max(range(self.map_group.count()),
                       key=lambda i: len(cat.groups.get(self.map_group.itemData(i) or "", ())),
                       default=-1)
            if best >= 0:
                self.map_group.setCurrentIndex(best)
        self.map_group.blockSignals(False)
        self._refresh_map_list()
        if self._pref_map and cat.ready and self._pref_map in set(cat.ids()):
            self._on_map_selected(self._pref_map)      # the map this window was last used with
            self.map_list.select(self._pref_map)
        self._pref_map = None
        self._update_start_enabled()

    def _map_rows(self, ids: List[str], with_group: bool):
        """(group, id, display, "id · family") rows. The display name leads because that is what a
        person is looking for; the id stays visible because it is what goes in `--tracks`."""
        rows = []
        for tid in ids:
            e = self.maps.entry(tid)
            group = (self.maps.group_of(tid) or "") if with_group else ""
            rows.append((group, tid, e.get("display") or tid,
                         f"{tid} · {e.get('family_label') or e.get('family', '')}"))
        return rows

    def _refresh_map_list(self):
        """The selected group, or -- the moment anything is typed -- the whole catalogue.

        Three groups is the right number of groups and the wrong number of *places to look*: the
        maps that are in neither split (the other twenty racetracks, the gym maps, any generator
        seed) still have to be reachable. They are reachable by typing, which is how anyone with a
        name in mind was going to find them anyway.
        """
        g = self.map_group.currentData()
        if not self.maps.ready or g is None:
            self.map_list.set_items([])
            self.map_list.set_status("맵 목록을 읽는 중입니다…" if not self.maps.error else "", "hint")
            return
        searching = bool(self.map_list.search.text().strip())
        if searching:
            self.map_list.set_items(self._map_rows(self.maps.ids(), with_group=True))
            return
        ids = self.maps.groups.get(g, [])
        self.map_list.set_items(self._map_rows(ids, with_group=False))
        hint = GROUP_HINT.get(g, "")
        self.map_list.set_status(f"{len(ids)} 개 · {hint}" if hint else f"{len(ids)} 개")

    def set_checkpoint_info(self, info: Optional[dict], error: Optional[str] = None):
        if error:
            for k in ("체크포인트", "저장", "진행", "저장 시점 지표"):
                self.ckpt_info.set(k, "—")
            self.run_note.setObjectName("HintDanger")
            self.run_note.setText(error)
            self.run_note.style().unpolish(self.run_note)
            self.run_note.style().polish(self.run_note)
            return
        self.run_note.setObjectName("Hint")
        self.run_note.setText("체크포인트가 갱신되면 자동으로 다시 읽습니다.")
        self.run_note.style().unpolish(self.run_note)
        self.run_note.style().polish(self.run_note)
        if not info:
            return
        self.ckpt_info.set("체크포인트", info.get("file", "—"))
        age = info.get("age_s")
        self.ckpt_info.set("저장", f"{format_age(age)} 전" if age is not None else "—")
        self.ckpt_info.set("진행", info.get("progress", "—"))
        self.ckpt_info.set("저장 시점 지표", info.get("metrics", "—"))
        source = str(info.get("cond_source") or "")
        if source == "dial":
            self.ckpt_info.set("그립 다이얼", "받습니다 — 구성 > 그립 다이얼에서 조절")
            self.run_note.setText("그립 다이얼 정책입니다. 구성 > 그립 다이얼에서 정책이 쓸 그립을 지정하세요.")
        elif source:
            self.ckpt_info.set("그립 다이얼", f"조건 입력 '{source}' (다이얼 아님)")
        cap = info.get("speed_cap")
        if cap:
            self.spin_cap.setValue(float(cap))
            self.row_cap.set_hint(f"체크포인트가 학습된 상한 {float(cap):.1f} m/s 를 따릅니다.", "hint")
        else:
            self.row_cap.set_hint("체크포인트에 학습 상한 기록이 없어 기본값을 씁니다.", "warn")

    # ---------------------------------------------------------------- state machine
    def apply_state(self, state: str, detail: str = ""):
        """One function decides everything the state controls, so nothing can disagree with it."""
        if state != STATE_PREPARING:
            self._stage = self._stage_since = None
            self._stage_note = ""
        elif self._stage_since is None:
            self._stage_since = time.monotonic()
        self.state = state
        self.badge.set_state(state, detail)
        running = state in (STATE_RUNNING, STATE_PAUSED)
        preparing = state == STATE_PREPARING

        self.btn_start.setEnabled(state in (STATE_IDLE, STATE_FAILED) and self._can_start())
        self._update_start_hint()
        self.btn_cancel.setEnabled(preparing)
        self.btn_pause.setEnabled(running)
        self.btn_reset.setEnabled(running)
        self.btn_stop.setEnabled(running or preparing)
        self.combo_focus.setEnabled(running)
        self.btn_focus_prev.setEnabled(running)
        self.btn_focus_next.setEnabled(running)
        self.btn_shot.setEnabled(running or preparing)
        if hasattr(self, "btn_record"):
            can_record = running and REC.ffmpeg_available()
            self.btn_record.setEnabled(can_record or self.recording)
            self.btn_record_bar.setEnabled(can_record or self.recording)
            if self.recording and not running:
                # The session is gone; nothing will paint another frame into the clip. Close it
                # rather than leave a file that silently stopped growing.
                self._stop_recording("세션 종료")
        self.btn_mu_apply.setEnabled(running)

        # settings stay editable during PREPARING on purpose: waiting is exactly when someone
        # realises they picked the wrong map
        for w in (self.run_list, self.map_list, self.map_group, self.spin_races, self.spin_grid,
                  self.spin_cap, self.chk_dr, self.chk_soft, self.chk_stoch, self.opp_table, self.combo_device,
                  self.combo_ros,
                  self.seg_direction, self.combo_obstacle, self.chk_bare_first):
            w.setEnabled(state in (STATE_IDLE, STATE_FAILED, STATE_PREPARING))
        # The seed row follows the obstacle choice, not the session state: 다시 뽑기 is exactly the
        # control you want while something is running, and it restarts the session itself.
        self._on_scenario_changed()

        self.progress.setVisible(preparing or state == STATE_STOPPING)
        was_running = self.viewport._running
        self.viewport.set_running(state in (STATE_RUNNING, STATE_PAUSED, STATE_PREPARING))
        self.viewport.paused = state == STATE_PAUSED
        if was_running and not self.viewport._running:
            # stop reporting a frame rate for a viewport that is no longer drawing: the last
            # session's number would sit there looking like a live measurement
            self.viewport.reset_timing()
            self.spark.clear()

        if state == STATE_PAUSED:
            self.viewport.set_badge("일시정지", C["accent"])
        elif state == STATE_PREPARING:
            self.viewport.set_badge(STAGE_TEXT.get(detail, detail) or "준비 중", C["warn"])
        elif state == STATE_FAILED:
            self.viewport.set_badge("오류", C["danger"])
        elif state == STATE_STOPPING:
            self.viewport.set_badge("정지 중", C["text.2"])
        else:
            self.viewport.set_badge("")

        if state == STATE_IDLE:
            self._update_start_hint()
            self.btn_pause.settle(False)
            self.viewport.set_empty_text("맵과 정책 런을 고르고 <b>시작</b>을 누르세요.")
        elif preparing:
            self.status_text.setText(f"준비 중 — {STAGE_TEXT.get(detail, detail or '')}")
            self.viewport.set_empty_text(
                f"<b>세션을 준비하는 중입니다.</b><br><br>"
                f"<span style='color:{C['text.2']}'>준비가 끝나기 전에도 설정을 바꾸거나 "
                f"취소할 수 있습니다.</span>")
        elif state == STATE_RUNNING:
            self.status_text.setText("주행 중")
        elif state == STATE_PAUSED:
            self.status_text.setText("일시정지 — 시뮬레이션이 멈춰 있습니다.")
        elif state == STATE_STOPPING:
            self.status_text.setText("정지 중 — worker를 정리하는 중입니다.")
        if state != STATE_FAILED:
            self.btn_retry.setVisible(False)
            self.btn_detail.setVisible(False)
        self._update_start_enabled()
        self._apply_responsive()
        if hasattr(self, "sel_running"):
            self._update_running_note()

    def set_stage(self, stage: str, note: str = ""):
        if self.state == STATE_PREPARING:
            if stage != self._stage:
                self._stage_since = time.monotonic()
            self._stage, self._stage_note = stage, note
            self.viewport.set_badge(STAGE_TEXT.get(stage, stage), C["warn"])
            self._show_preparing_progress()

    def log(self, text: str, kind: str = "info") -> None:
        """One timestamped line into the 진행 로그, and open the panel when it is bad news.

        Deliberately not the status line: that shows one thing at a time and is overwritten by the
        next. What was missing was a record -- what stage ran, how long it took, what the worker
        said on the way -- so that a slow start can be read afterwards instead of guessed at.
        """
        if not text:
            return
        view = getattr(self, "log_view", None)
        if view is None:
            return
        mark = {"error": "✖", "warn": "!", "done": "✔"}.get(kind, "·")
        view.appendPlainText(f"{time.strftime('%H:%M:%S')}  {mark} {text}")
        bar = view.verticalScrollBar()
        bar.setValue(bar.maximum())
        if kind == "error" and hasattr(self, "log_panel") and not self.log_panel.toggle.isChecked():
            self.log_panel.toggle.setChecked(True)

    def set_error(self, where: str, message: str, detail: str = "", retryable: bool = True):
        self._last_error = {"where": where, "message": message, "detail": detail}
        self.apply_state(STATE_FAILED)
        self.status_text.setText(f"오류 ({where}) — {message}")
        self.status_text.setStyleSheet(f"color: {C['danger']};")
        self.btn_retry.setVisible(retryable)
        self.btn_detail.setVisible(bool(detail))
        self.viewport.set_empty_text(
            f"<span style='color:{C['danger']}'><b>세션을 시작하지 못했습니다</b></span><br><br>"
            f"{QtGui.QGuiApplication.translate('', message)}<br><br>"
            f"<span style='color:{C['text.2']}'>설정을 고치고 다시 시작하거나, 아래 '다시 시도'를 "
            f"누르세요.</span>")

    def clear_error(self):
        self._last_error = None
        self.status_text.setStyleSheet("")
        self.btn_retry.setVisible(False)
        self.btn_detail.setVisible(False)

    def set_session_facts(self, facts: dict):
        """What the worker actually built -- never what the UI asked for."""
        self._running_facts = dict(facts)
        self.generation = int(facts.get("gen", self.generation))
        self._session_cars = int(facts.get("total_cars", 0))
        self._session_ids = list(facts.get("car_ids", []))
        run = facts.get("run", "—")
        # The scenario as built, with the drawn seed in it. This is the whole point of the strip:
        # "무작위" has to end in a number the user can see and type back in.
        mp = facts.get("scenario") or facts.get("map", "—")
        cars = facts.get("total_cars", 0)
        races = facts.get("races", 1)
        grid = facts.get("cars_per_race", 1)
        shown = min(cars, facts.get("max_render_cars", MAX_RENDER_CARS))
        device = facts.get("device", "?")
        summary = f"{run}  ·  {mp}  ·  {races}레이스 × {grid}대 = {cars}대"
        if shown < cars:
            summary += f" (화면 {shown}대)"
        obs = facts.get("obstacle_text") or ""
        if obs:
            n_props = int(facts.get("authored_props") or 0)
            summary += f"  ·  장애물 {obs}" + (f" ({n_props}개)" if n_props else "")
        mix = facts.get("opponent_mix") or ""
        if grid > 1 and mix:
            summary += f"  ·  상대차 {mix}"
        summary += f"  ·  {device}  ·  세션 #{self.generation}"
        ros = facts.get("ros2")
        if ros:
            summary += "  ·  ROS2 " + ("/drive 제어" if ros.get("mode") == "drive" else "발행")
        self.header_summary.setText(summary)
        self.header_summary.setToolTip(
            "\n".join([f"{facts.get('scenario_display') or ''}",
                        f"로더 이름: {facts.get('map_legacy') or mp}"]
                       + list(facts.get("opponent_slot_lines") or [])).strip())
        self.combo_focus.blockSignals(True)
        self.combo_focus.clear()
        for cid in self._session_ids[:shown]:
            self.combo_focus.addItem(f"차량 {cid}", cid)
        self.combo_focus.blockSignals(False)
        self.viewport.color_v_max = float(facts.get("speed_cap", 6.0) or 6.0)
        self.viewport.lidar_cfg = facts.get("lidar", {}) or {}
        self.viewport.vehicle_cfg = facts.get("vehicle", {}) or self.viewport.vehicle_cfg
        self.policy_panel.fov = float((facts.get("lidar") or {}).get("fov", 4.71238898))
        self.policy_panel.range_max = float((facts.get("lidar") or {}).get("range_max", 10.0))
        self.dash_panel.v_max = float(facts.get("v_max_policy", 8.0) or 8.0)
        self.dash_panel.clear_grip_estimate()
        self.drive_info.set("추정 μ", "추정 대기")
        self.viewport.set_corner_text(f"{mp}\n{run}")
        self._update_running_note()

    def set_gl_info(self, text: str):
        self.trust_info.set("렌더러", text)

    # ---------------------------------------------------------------- live numbers
    def update_telemetry(self, frame: Optional[dict], fresh: Freshness, stats: dict,
                         sim_rate: Optional[float], fps: Optional[float],
                         frame_ms: Optional[Tuple[float, float, float]]):
        if frame is None:
            for tile in (self.m_speed, self.m_lap, self.m_coll, self.m_wall):
                tile.set_unknown()
        else:
            f = int(frame.get("focus", 0))
            self.dash_panel.set_focus(f)
            self.dash_panel.set_grip_estimate(frame.get("grip_estimate"))
            n = int(frame.get("n", 0))
            if 0 <= f < n:
                self.m_speed.set_value(f"{float(frame['vx'][f]):5.2f}")
                self.m_lap.set_value(f"{int(frame['lap'][f])}")
                self.m_wall.set_value(f"{float(frame['wall'][f]):4.2f}",
                                      C["danger"] if float(frame["wall"][f]) < 0.15 else None)
                self.drive_info.set("조향", f"{math.degrees(float(frame['steer'][f])):+6.1f}°")
                self.drive_info.set("자세 roll/pitch",
                                    f"{math.degrees(float(frame['roll'][f])):+5.1f}° / "
                                    f"{math.degrees(float(frame['pitch'][f])):+5.1f}°")
                self.drive_info.set("랩 진행", f"{float(frame['s'][f]):.1f} m")
                mu = frame.get("mu")
                self.drive_info.set("시뮬 참값 μ", f"{float(mu):.3f}" if mu is not None else "—")
                self.drive_info.set("추정 μ", self.dash_panel.grip_status)
            crashed = int((frame["coll"] > 0.5).sum()) if frame.get("coll") is not None else 0
            self.m_coll.set_value(f"{crashed} / {n}", C["danger"] if crashed else None)

        self.m_simrate.set_value(f"{sim_rate:5.2f}" if sim_rate is not None else "—",
                                 None if sim_rate is None or sim_rate > 0.9 else C["warn"])
        self.m_fps.set_value(f"{fps:5.1f}" if fps is not None else "—")
        if frame_ms is not None:
            p50, p95, mx = frame_ms
            n = self.viewport.rate.count
            self.frame_stats.setText(
                f"프레임 간격(최근 {n})  p50 {p50:5.1f} ms   p95 {p95:5.1f} ms   최대 {mx:5.1f} ms")
        else:
            self.frame_stats.setText("프레임 간격 —")
        d = self.viewport.draw_percentiles()
        if d is not None:
            self.draw_stats.setText(f"그리기 시간  p50 {d[0]:5.1f} ms   p95 {d[1]:5.1f} ms   최대 {d[2]:5.1f} ms")
        else:
            self.draw_stats.setText("그리기 시간 —")

        if not fresh.have_data:
            self.trust_info.set("데이터 도착", "—")
            self.trust_info.set("데이터 생성", "—")
            self.trust_info.set("seq 지연", "—")
        else:
            self.trust_info.set("데이터 도착", f"{fresh.arrival_age * 1e3:6.0f} ms 전")
            self.trust_info.set("데이터 생성", f"{fresh.creation_age * 1e3:6.0f} ms 전")
            backlog = max(0, fresh.seq_gap - INTERP_LAG_FRAMES)
            self.trust_info.set("seq 지연",
                                f"{fresh.seq_gap} 프레임 (보간 {INTERP_LAG_FRAMES} 포함, 초과 {backlog})")
        # Four separate things. Retiring a frame after drawing it is how a small history works and
        # is not a loss; the others are.
        self.trust_info.set("프레임 회전", f"{stats.get('retired_after_draw', 0)} (그린 뒤 폐기 — 정상)",
                            tooltip="버퍼는 보간용으로 몇 장만 들고 있습니다. 이미 그린 프레임이 밀려나는 것은 "
                                    "정상 동작이며 손실이 아닙니다.")
        self.trust_info.set("프레임 손실",
                            f"GUI 미표시 {stats.get('dropped_undrawn_gui', 0)} · "
                            f"수신큐 {stats.get('dropped_receiver_queue', 0)} · "
                            f"worker {stats.get('dropped_worker', 0)}"
                            + (f" · 형식오류 {stats['malformed']}" if stats.get("malformed") else ""),
                            tooltip="GUI 미표시: 한 번도 그리지 못하고 버려진 프레임 (렌더가 못 따라감).\n"
                                    "수신큐: GUI의 인바운드 슬롯이 버린 것.\n"
                                    "worker: worker가 자기 송신 슬롯에서 버린 것.\n"
                                    "서로 다른 원인이라 합치지 않습니다.")
        rtt = stats.get("worker_rtt_ms")
        self.trust_info.set("worker 응답시간", f"{rtt:.0f} ms" if rtt is not None else "—",
                            tooltip="worker가 요청에 답하는 데 걸린 최소 시간. 같은 기기의 프로세스이므로 "
                                    "시계 차이가 아니라 응답 속도입니다 (데이터 나이 보정에 쓰지 않습니다).")

        # The stale badge belongs to RUNNING only: a paused session has old data by definition and
        # calling that an error would train people to ignore the badge.
        if self.state == STATE_RUNNING and fresh.is_stale():
            worst = max(fresh.arrival_age, fresh.creation_age)
            self.viewport.set_badge(f"데이터 정지 · {worst:.1f}초", C["danger"])
        elif self.state == STATE_RUNNING:
            self.viewport.set_badge("")
        self._update_health(fresh, stats)

    def _update_health(self, fresh: Freshness, stats: dict):
        """One line that says whether the folded diagnostics are worth opening."""
        kind, text = "hint", "—"
        if self.state == STATE_PAUSED:
            kind, text = "hint", "일시정지 중 — 데이터가 멈춘 것이 정상입니다."
        elif self.state == STATE_PREPARING:
            kind, text = "hint", "준비 중 — 아직 프레임이 오지 않습니다."
        elif not fresh.have_data:
            kind, text = "hint", "아직 받은 프레임이 없습니다."
        elif fresh.is_stale():
            kind = "warn"
            text = f"데이터가 {max(fresh.arrival_age, fresh.creation_age):.1f}초째 갱신되지 않았습니다."
        elif fresh.seq_gap > INTERP_LAG_FRAMES + 2:
            kind = "warn"
            text = (f"프레임이 {fresh.seq_gap - INTERP_LAG_FRAMES}개 밀려 있습니다 "
                    f"(보간 지연 {INTERP_LAG_FRAMES} 제외) — 상세 진단을 보세요.")
        else:
            lost = (stats.get("dropped_undrawn_gui", 0) + stats.get("dropped_receiver_queue", 0)
                    + stats.get("dropped_worker", 0))
            text = f"정상 — 지연 {fresh.creation_age * 1e3:.0f} ms, 손실 {lost}"
        self.health_line.setObjectName({"hint": "Hint", "warn": "HintWarn", "danger": "HintDanger"}[kind])
        self.health_line.setText(text)
        self.health_line.style().unpolish(self.health_line)
        self.health_line.style().polish(self.health_line)

    def push_frame_time(self, ms: float):
        self.spark.push(ms)

    def set_policy_overlay(self, scan, sal, plan_ref, v_max, sal_age=None):
        self.policy_panel.set_data(scan, sal, plan_ref, v_max, sal_age)

    def set_dash(self, *a):
        self.dash_panel.set_data(*a)

    def set_activations(self, hidden, stem):
        self.activation_panel.set_data(hidden, stem)

    def note_ack(self, command: str):
        """A command the controller has accepted: drop it from the pending ledger and settle its
        button. The ledger is what `tick_pending` reads, so anything that resolves a command has to
        come through here rather than acking the button on its own."""
        self._pending.pop(command, None)
        if command == "reset":
            self.btn_reset.ack()
        elif command == "stop":
            self.btn_stop.ack()
        elif command == "start":
            self.btn_start.ack()

    def settle_mu(self, mode: str, mu: float):
        """The worker confirmed a friction change: settle the button and say what is in force."""
        self.btn_mu_apply.ack()
        self._pending.pop("set_mu", None)
        self.status_text.setText(f"노면 마찰: {'고정 μ=' + format(mu, '.3f') if mode == 'fixed' else '랜덤 (리셋마다 다시 뽑음)'} — 모든 차량에 적용됨")
        self.viewport.notify(f"노면 마찰 μ = {mu:.3f}" if mode == "fixed" else "노면 마찰: 랜덤", "good")

    def show_dial(self, mu: Optional[float]):
        """The running checkpoint's dial, or None when it has no conditioning input."""
        self.row_dial.setVisible(mu is not None)
        self.btn_dial_apply.setEnabled(mu is not None)
        if mu is not None:
            self.spin_dial.blockSignals(True)
            self.spin_dial.setValue(float(mu))
            self.spin_dial.blockSignals(False)

    def settle_dial(self, mu: Optional[float]):
        """The worker confirmed a dial change."""
        self.btn_dial_apply.ack()
        self._pending.pop("set_dial", None)
        if mu is not None:
            self.status_text.setText(f"그립 다이얼: 정책이 μ={mu:.3f} 만큼의 그립을 쓰도록 지시받았습니다")
            self.viewport.notify(f"그립 다이얼 = {mu:.3f}", "good")

    def _on_dial_apply(self):
        self.btn_dial_apply.mark_pending()
        self._pending["set_dial"] = time.monotonic()
        self.dial_requested.emit(float(self.spin_dial.value()))

    def _on_mu_mode(self, _idx):
        fixed = self.combo_mu.currentData() == "fixed"
        self.spin_mu.setEnabled(fixed)

    def _on_mu_apply(self):
        self.btn_mu_apply.mark_pending()
        self._pending["set_mu"] = time.monotonic()
        self.mu_requested.emit(str(self.combo_mu.currentData() or "random"), float(self.spin_mu.value()))

    MODE_INDEX = {"drive": 0, "train": 1, "edit": 2}

    def set_mode(self, key: str):
        """Driving, training or the environment editor. The full-width pages hide the driving
        panels; they come back with the driving page."""
        key = key if key in self.MODE_INDEX else "drive"
        drive = key == "drive"
        self.mode_buttons.set_current(key)
        self.centre_stack.setCurrentIndex(self.MODE_INDEX[key])
        self.left_panel.setVisible(drive and self.btn_left_panel.isChecked())
        self.right_panel.setVisible(drive and self.btn_right_panel.isChecked())
        self.training.set_active(key == "train")
        self.editor.set_active(key == "edit")
        if self.state == STATE_IDLE and hasattr(self, "status_text"):
            if drive:
                self._update_start_hint()
            elif key == "train":
                self.status_text.setText("학습 — 설정을 고르고 '학습 시작'을 누르세요.")
            else:
                self.status_text.setText("환경 — 새로 만들거나 목록에서 고르세요.")

    def current_mode(self) -> str:
        for k, i in self.MODE_INDEX.items():
            if i == self.centre_stack.currentIndex():
                return k
        return "drive"

    SCENE_GROUP = catalog.SCENES_GROUP

    def _with_local_scenes(self, cat: MapCatalog) -> MapCatalog:
        """The scene group, read from the scenes folder here rather than taken from the worker.

        The worker lists scenes when it starts; the editor creates them while the worker is
        already up. The folder is the truth for both, and reading it is a directory listing."""
        if not cat.ready:
            return cat
        try:
            ids = catalog.scene_ids()
        except Exception:
            return cat
        groups = {k: v for k, v in cat.groups.items() if k != self.SCENE_GROUP}
        entries = dict(cat.entries)
        if ids:
            groups[self.SCENE_GROUP] = ids
            for tid in ids:
                entries.setdefault(tid, {"id": tid, "family": "scene",
                                         "family_label": tracks.FAMILY_LABEL["scene"],
                                         "display": f"{tid.split('/', 1)[1]} (에디터)",
                                         "legacy": f"scene:{tid.split('/', 1)[1]}", "note": "",
                                         "obstacles": list(tracks.OBSTACLES_BY_FAMILY["scene"]),
                                         "props": tracks.scene_props(tid) or 0})
        return MapCatalog(groups=groups, entries=entries, ready=cat.ready, error=cat.error)

    def _scenes_changed_from_editor(self):
        """The editor saved, duplicated or deleted a scene: refresh the map picker's scene group
        without a round trip to the worker."""
        if not self.maps.ready:
            return
        current_group = self.map_group.currentData()
        selected = self._selected_map
        self.set_maps(self.maps)
        if current_group is not None:
            i = self.map_group.findData(current_group)
            if i >= 0:
                self.map_group.setCurrentIndex(i)
        if selected:
            self.map_list.select(selected)

    def _drive_from_editor(self, track_id: str):
        """A scene saved on the editor page becomes the driving page's next map."""
        self._scenes_changed_from_editor()
        self.set_mode("drive")
        i = self.map_group.findData(self.SCENE_GROUP)
        if i >= 0:
            self.map_group.setCurrentIndex(i)
        if not self.map_list.select(track_id):
            # the worker's catalogue is not in yet: keep the name as the selection anyway
            pass
        self._on_map_selected(track_id)
        self.status_text.setText(f"{self.maps.display(track_id)} 을(를) 다음 시작에 사용합니다. "
                                 f"런을 고르고 '시작'을 누르세요.")

    def _view_checkpoint_from_training(self, run_name: str, ckpt_path: str):
        """A checkpoint picked on the training page becomes the driving page's next start."""
        self.set_mode("drive")
        if not self.run_list.select(run_name):
            pass
        self._selected_run = ckpt_path
        self.sel_run.setText(f"런: {self._run_label(ckpt_path)}")
        self.sel_run.setToolTip(ckpt_path)
        self._update_selection_note()
        self._update_start_enabled()
        self.describe_requested.emit(ckpt_path)
        self.status_text.setText(f"{os.path.basename(ckpt_path)} 을(를) 다음 시작에 사용합니다. '시작'을 누르세요.")

    def _runs_changed_from_training(self):
        from .catalog import list_runs
        self.set_runs(list_runs())

    def settle_pause(self, paused: bool):
        self.btn_pause.settle(paused)
        self.btn_pause.setText("재개" if paused else "일시정지")
        self._pending.pop("pause", None)

    def tick_pending(self):
        """Report a slow *runtime control*, and report preparation as preparation.

        `start` is not answered by an ack -- it is answered by `ready`, minutes later on a cold
        compile -- so measuring it against the 0.7 s ack threshold declared the worker unresponsive
        while it was in fact working, and overwrote the stage line that said so on every tick. What
        the user saw for ten minutes was "worker 응답 지연", not "맵 로드 중".

        Preparation now reports the stage the worker last reached and how long it has been going.
        The ack warning stays exactly as it was for the controls that really are acked -- pause,
        reset, stop, focus -- because a genuinely unresponsive worker still has to be visible.
        """
        if self.state == STATE_PREPARING:
            self._show_preparing_progress()
            return
        if not self._pending:
            return
        now = time.monotonic()
        acked = {k: v for k, v in self._pending.items() if k != "start"}
        if not acked:
            return
        worst_cmd, worst_dt = max(acked.items(), key=lambda kv: now - kv[1])
        dt = now - worst_dt
        if dt > ACK_WARN_S and self.state != STATE_FAILED:
            names = {"pause": "일시정지/재개", "reset": "리셋", "stop": "정지", "focus": "주시 차량"}
            self.status_text.setText(
                f"worker 응답 지연 {dt:.1f}초 — {names.get(worst_cmd, worst_cmd)} 명령을 아직 "
                f"확인하지 못했습니다. 화면 조작은 계속 가능합니다.")

    def _show_preparing_progress(self):
        """What the worker is doing, and for how long. Never a claim that it has finished."""
        started = self._pending.get("start") or self._stage_since
        if started is None:
            return
        total = time.monotonic() - started
        text = STAGE_TEXT.get(self._stage, self._stage) if self._stage else "시작 준비 중"
        here = f" · 이 단계 {time.monotonic() - self._stage_since:.0f}초" if self._stage_since else ""
        note = f" · {self._stage_note}" if self._stage_note else ""
        self.status_text.setText(f"준비 중 {total:.0f}초 — {text}{note}{here}.  취소할 수 있습니다.")

    # ================================================================ user actions
    def _start_blocker(self) -> str:
        """Why 시작 is not available, in one short phrase, or "" when it is."""
        if not self._selected_run and not self._selected_map:
            return "① 정책 런과 ② 맵을 고르세요."
        if not self._selected_run:
            return "① 정책 런을 고르세요."
        if not self._selected_map:
            return "② 맵을 고르세요."
        if hasattr(self, "opp_table") and self.spin_grid.value() > 1 \
                and self.opp_table.problem(self.spin_grid.value()):
            return "고급 설정의 상대차 표를 먼저 고치세요."
        return ""

    def _can_start(self) -> bool:
        if not (self._selected_run and self._selected_map):
            return False
        # A slot table that names a checkpoint the loader will refuse is a start that fails after a
        # minute of loading. The table already knows; the button asks it.
        if hasattr(self, "opp_table") and self.spin_grid.value() > 1:
            return not self.opp_table.problem(self.spin_grid.value())
        return True

    def _update_start_hint(self):
        """The line above 시작, and the idle status bar, which must never say different things.

        The status bar is left alone unless the console is idle: while a session is preparing,
        running or failed it is carrying that, and a message about what to pick next would be
        talking over it.
        """
        if not hasattr(self, "start_hint"):
            return
        if self.state in (STATE_RUNNING, STATE_PAUSED):
            self.start_hint.setText("시작을 다시 누르면 지금 고른 설정으로 새로 시작합니다.")
            return
        blocker = self._start_blocker()
        self.start_hint.setText(blocker or "준비됐습니다. 시작을 누르세요.")
        if (self.state == STATE_IDLE and hasattr(self, "status_text")
                and self.current_mode() == "drive"):
            self.status_text.setText(f"대기 — {blocker}" if blocker else "대기 — 준비됐습니다.")

    def _update_start_enabled(self):
        if self.state in (STATE_IDLE, STATE_FAILED):
            self.btn_start.setEnabled(self._can_start())
        self._update_start_hint()

    @staticmethod
    def _run_label(ckpt_path: str) -> str:
        """`spec_korea_s904 / ppo_latest.pt` from the checkpoint's absolute path.

        The run directory and the file are the two things that identify a policy; the rest of the
        path is the same for every row in the list, and printing it wraps the card onto two lines
        and hides the name at the end of the first.
        """
        d, f = os.path.split(ckpt_path)
        run = os.path.basename(d) or ckpt_path
        return f"{run} / {f}" if f else run

    def _on_run_selected(self, name: str):
        self._selected_run = name
        self.sel_run.setText(f"런: {self._run_label(name)}")
        self.sel_run.setToolTip(name)
        self._update_selection_note()
        self._update_start_enabled()
        self.describe_requested.emit(name)

    def _on_map_selected(self, name: str):
        self._selected_map = name
        self._sync_obstacle_options()
        self._on_scenario_changed()
        self._update_start_enabled()

    # ---------------------------------------------------------------- scenario controls
    def _sync_obstacle_options(self):
        """Offer only the obstacle families this track can actually carry.

        `maps._load_base` raises for `rt:Monza+obs3` -- a racetrack and an editor scene understand
        modelled props and nothing else. Offering the choice and then failing the start is the
        version of this that wastes a checkpoint load.
        """
        tid = self._selected_map or ""
        allowed = tracks.asset_obstacle_options(tid) if tid and tracks.is_spec(tid) else list(tracks.ASSET_OBSTACLES)
        n_props = self.maps.authored_props(tid) if tid else 0
        want = str(self.combo_obstacle.currentData() or "")
        self.combo_obstacle.blockSignals(True)
        self.combo_obstacle.clear()
        for o in tracks.ASSET_OBSTACLES:
            if o not in allowed:
                continue
            self.combo_obstacle.addItem(self._obstacle_label(o, n_props), o)
            i = self.combo_obstacle.count() - 1
            self.combo_obstacle.setItemData(i, tracks.ASSET_PLACEMENT_HINT[o], QtCore.Qt.ToolTipRole)
            if o == tracks.BARE and not n_props:
                # Listed and greyed rather than hidden: "this map has nothing placed on it" is an
                # answer, and hiding the entry would make 기본 look like the only thing there is.
                self.combo_obstacle.setItemData(i, 0, QtCore.Qt.UserRole - 1)
                self.combo_obstacle.setItemData(i, "이 맵은 배치 장애물이 없음 — '기본'과 같습니다.",
                                                QtCore.Qt.ToolTipRole)
        i = self.combo_obstacle.findData(want)
        if i >= 0 and want == tracks.BARE and not n_props:
            i = self.combo_obstacle.findData("")        # the map changed under a now-meaningless 없음
        self.combo_obstacle.setCurrentIndex(i if i >= 0 else 0)
        self.combo_obstacle.blockSignals(False)
        self._n_props = int(n_props)
        if len(allowed) < len(tracks.OBSTACLES):
            self.row_obstacle.set_hint(
                f"이 계열은 '{'/'.join(tracks.OBSTACLE_LABEL[o] for o in allowed if o)}' 만 지원합니다.",
                "hint")

    @staticmethod
    def _obstacle_label(kind: str, n_props: int) -> str:
        """The combo's text. The counts are the whole point: 기본 on a scene with three boxes has to
        say so, or it is the same silent claim 없음 used to make."""
        if not n_props:
            return tracks.ASSET_PLACEMENT_LABEL[kind]
        if kind == "":
            return f"{tracks.OBSTACLE_LABEL['']} (배치된 장애물 {n_props}개)"
        if kind == tracks.BARE:
            return f"{tracks.OBSTACLE_LABEL[tracks.BARE]} (배치 장애물 제거)"
        return tracks.ASSET_PLACEMENT_LABEL[kind]

    def _scenario(self) -> str:
        """The selection and the three controls as one spec string. `#<kind>:*` when the seed is
        random -- the worker draws it, not the GUI thread."""
        tid = self._selected_map or ""
        if not tid or not tracks.is_spec(tid):
            return tid                       # a name from outside the registry: pass it through
        spec = tid
        d = self.seg_direction.current() or ""
        if d:
            spec += f"@{d}"
        o = str(self.combo_obstacle.currentData() or "")
        bare = o == tracks.BARE or (o and self.chk_bare_first.isChecked())
        family = "" if o == tracks.BARE else o
        choice = tracks.obstacle_choice(bool(bare), family)
        if choice:
            spec += f"#{choice}"
            if family:                        # `bare` alone places nothing, so it takes no seed
                seed = "*" if str(self.combo_seed.currentData()) == "random" else str(self.spin_seed.value())
                spec += f":{seed}"
        return tracks.asset_scenario(spec)

    def _on_scenario_changed(self):
        o = str(self.combo_obstacle.currentData() or "")
        family = o and o != tracks.BARE
        n_props = int(getattr(self, "_n_props", 0))
        hint = tracks.ASSET_PLACEMENT_HINT.get(o, "")
        if family:
            hint = f"{hint} {tracks.OBSTACLE_ADDS_HINT}"
        self.row_obstacle.set_hint(hint, "hint")
        # The composition control only means something where there is something to remove and
        # something being added on top of it.
        self.chk_bare_first.setEnabled(bool(family) and n_props > 0)
        if not family:
            self.chk_bare_first.blockSignals(True)
            self.chk_bare_first.setChecked(o == tracks.BARE)
            self.chk_bare_first.blockSignals(False)
        random_seed = str(self.combo_seed.currentData()) == "random"
        for w in (self.combo_seed, self.btn_reroll):
            w.setEnabled(bool(family))
        self.spin_seed.setEnabled(bool(family) and not random_seed)
        self.btn_reroll.setEnabled(bool(family) and random_seed)
        spec = self._scenario()
        self.scenario_line.setText(spec)
        try:
            self.sel_map.setText(f"맵: {tracks.display(spec)}" if spec else "맵: —")
        except Exception:
            self.sel_map.setText(f"맵: {spec}" if spec else "맵: —")
        self.sel_map.setToolTip(spec)
        self._update_selection_note()

    def _on_seed_mode(self, _idx=0):
        self._on_scenario_changed()

    def _on_reroll(self):
        """A new placement is a new map, so this is a new session seed and a restart -- not a live
        command. Pressing it while nothing is running just changes what the next 시작 will build."""
        import random as _random
        self._session_seed = _random.randrange(1, 10 ** 9)
        self._on_scenario_changed()
        if self.state in (STATE_RUNNING, STATE_PAUSED):
            self._on_start()

    def _update_selection_note(self):
        tid = self._selected_map
        g = self.maps.group_of(tid) if tid else None
        self.sel_note.setText(f"{g} · 학습 이력 미확인" if g else "")
        self._update_running_note()

    def _update_running_note(self):
        """Say so when the picker no longer describes what is on screen."""
        running = self._running_facts
        if not running or self.state not in (STATE_RUNNING, STATE_PAUSED):
            self.sel_running.setText("")
            return
        differs = (self._selected_run != running.get("run")
                   or self._scenario() != running.get("map"))
        if differs:
            self.sel_running.setText(
                f"화면에서 도는 것은 {running.get('run', '?')} / "
                f"{running.get('scenario') or running.get('map', '?')} 입니다. "
                f"위 설정은 다음 '시작'에 쓰입니다.")
        else:
            self.sel_running.setText("")

    def _update_car_math(self):
        races, grid = self.spin_races.value(), self.spin_grid.value()
        total = races * grid
        text = f"총 {total}대 = {races}레이스 × {grid}대"
        if total > MAX_RENDER_CARS:
            text += f"   (화면에는 {MAX_RENDER_CARS}대까지 그립니다)"
        self.car_math.setText(text)
        if grid > 1:
            self.row_grid.set_hint(f"같은 트랙에서 {grid}대가 겨룹니다. 상대차는 스캔에도 잡힙니다.", "hint")
        else:
            self.row_grid.reset_hint()
        # The table is the other cars, so its height is the grid minus the learner. Rows already
        # filled in survive a change: raising the count from 2 to 3 adds a row, it does not reset
        # the two that were configured.
        if hasattr(self, "opp_table"):
            self.opp_table.set_count(max(0, grid - 1))
            self.row_opp.setVisible(grid > 1)
            self._queue_panel_cap()
            self._on_slots_changed()

    def _on_slots_changed(self):
        """Show the table's own objection on the field row, and keep 시작 honest about it."""
        if not hasattr(self, "opp_table"):
            return
        problem = self.opp_table.problem(self.spin_grid.value()) if self.spin_grid.value() > 1 else ""
        if problem:
            self.row_opp.set_hint(problem, "warn")
        else:
            self.row_opp.reset_hint()
        self._update_start_enabled()

    # ---------------------------------------------------------------- remembered settings
    def collect_prefs(self) -> dict:
        """The controls a person sets, as plain values. Read by `prefs.save`."""
        return {
            "run": self._selected_run or "",
            "map": self._selected_map or "",
            "direction": self.seg_direction.current() or "",
            "obstacle": str(self.combo_obstacle.currentData() or ""),
            "obstacle_seed_mode": str(self.combo_seed.currentData() or ""),
            "obstacle_seed": int(self.spin_seed.value()),
            "races": int(self.spin_races.value()),
            "cars_per_race": int(self.spin_grid.value()),
            "speed_cap": float(self.spin_cap.value()),
            "device": self.combo_device.currentText(),
            "randomize": bool(self.chk_dr.isChecked()),
            "collision_soft": bool(self.chk_soft.isChecked()),
            "stochastic": bool(self.chk_stoch.isChecked()),
            "bare_first": bool(self.chk_bare_first.isChecked()),
            "mu_mode": str(self.combo_mu.currentData() or "random"),
            "mu": float(self.spin_mu.value()),
            "dial": float(self.spin_dial.value()),
            "ros2": str(self.combo_ros.currentData() or "off"),
            "saliency": bool(self.chk_saliency.isChecked()),
            "internals": bool(self.chk_internals.isChecked()),
            # The recording form, and which folds were left open. Both are things a person sets
            # once and expects to find the way they left them -- which was the complaint that got
            # any of this remembered in the first place.
            "record_dir": self.edit_record.text().strip(),
            "record_res": int(self.combo_res.currentIndex()),
            "record_fps": int(self.combo_fps.currentData() or 30),
            "record_camera": str(self.combo_rec_cam.currentData() or ""),
            "record_seconds": float(self.spin_rec_secs.value()),
            "record_overlay": bool(self.chk_rec_overlay.isChecked()),
            "record_encoder": str(self.combo_rec_enc.currentData() or ""),
            "fold_record": bool(self.record_fold.toggle.isChecked()),
            "fold_advanced": bool(self.adv_fold.toggle.isChecked()),
            "fold_checkpoint": bool(self.ckpt_fold.toggle.isChecked()),
        }

    def apply_prefs(self, p: dict) -> None:
        """Put the remembered values back.

        Field by field, each guarded on its own: a value that no longer parses, a combo entry that
        has gone away, or a control that has been renamed costs *that* setting and nothing else. A
        single try block around the lot meant one bad field silently discarded every field after it,
        which is indistinguishable from not having saved at all.
        """
        if not p:
            return

        def attempt(fn, key):
            if key in p:
                try:
                    fn(p[key])
                except Exception:
                    pass

        def combo_text(combo):
            def run(value):
                i = combo.findText(str(value))
                if i >= 0:
                    combo.setCurrentIndex(i)
            return run

        def combo_data(combo):
            def run(value):
                i = combo.findData(str(value))
                if i >= 0:
                    combo.setCurrentIndex(i)
            return run

        def restore_run(value):
            # A checkpoint that has been deleted or renamed would leave the window claiming a
            # selection whose Start fails, so it is dropped and the field reads as a first launch.
            run = str(value or "")
            if run and os.path.exists(run):
                self.run_list.select(run)
                self._on_run_selected(run)

        # The map list is built in the worker (map names need torch), so at construction time there
        # is nothing to select in; `set_maps` applies this when the catalogue arrives.
        attempt(restore_run, "run")
        attempt(lambda v: setattr(self, "_pref_map", str(v) or None), "map")
        attempt(lambda v: self.seg_direction.set_current(str(v or "")), "direction")
        attempt(combo_data(self.combo_obstacle), "obstacle")
        attempt(combo_data(self.combo_seed), "obstacle_seed_mode")
        attempt(lambda v: self.spin_seed.setValue(int(v)), "obstacle_seed")
        attempt(lambda v: self.spin_races.setValue(int(v)), "races")
        attempt(lambda v: self.spin_grid.setValue(int(v)), "cars_per_race")
        attempt(lambda v: self.spin_cap.setValue(float(v)), "speed_cap")
        attempt(combo_text(self.combo_device), "device")
        attempt(lambda v: self.chk_dr.setChecked(bool(v)), "randomize")
        attempt(lambda v: self.chk_soft.setChecked(bool(v)), "collision_soft")
        attempt(lambda v: self.chk_stoch.setChecked(bool(v)), "stochastic")
        attempt(lambda v: self.chk_bare_first.setChecked(bool(v)), "bare_first")
        attempt(combo_data(self.combo_mu), "mu_mode")
        attempt(lambda v: self.spin_mu.setValue(float(v)), "mu")
        attempt(lambda v: self.spin_dial.setValue(float(v)), "dial")
        attempt(combo_data(self.combo_ros), "ros2")
        attempt(lambda v: self.chk_saliency.setChecked(bool(v)), "saliency")
        attempt(lambda v: self.chk_internals.setChecked(bool(v)), "internals")
        attempt(lambda v: self.edit_record.setText(str(v or "")), "record_dir")
        attempt(lambda v: self.combo_res.setCurrentIndex(int(v))
                if 0 <= int(v) < self.combo_res.count() else None, "record_res")
        def restore_fps(value):
            i = self.combo_fps.findData(int(value))
            if i >= 0:
                self.combo_fps.setCurrentIndex(i)

        attempt(restore_fps, "record_fps")
        attempt(combo_data(self.combo_rec_cam), "record_camera")
        attempt(lambda v: self.spin_rec_secs.setValue(float(v)), "record_seconds")
        attempt(lambda v: self.chk_rec_overlay.setChecked(bool(v)), "record_overlay")
        attempt(combo_data(self.combo_rec_enc), "record_encoder")
        attempt(lambda v: self.record_fold.toggle.setChecked(bool(v)), "fold_record")
        attempt(lambda v: self.adv_fold.toggle.setChecked(bool(v)), "fold_advanced")
        attempt(lambda v: self.ckpt_fold.toggle.setChecked(bool(v)), "fold_checkpoint")

    def save_prefs(self) -> None:
        from . import prefs
        prefs.save(self.collect_prefs())

    def restore_prefs(self) -> None:
        from . import prefs
        self.apply_prefs(prefs.load())

    def current_config(self) -> SessionConfig:
        return SessionConfig(
            run=self._selected_run or "latest",
            map_name=self._scenario(),
            seed=int(self._session_seed),
            races=self.spin_races.value(),
            cars_per_race=self.spin_grid.value(),
            speed_cap=float(self.spin_cap.value()),
            device=self.combo_device.currentText(),
            compile=False,
            randomize=self.chk_dr.isChecked(),
            collision_mode=("soft" if self.chk_soft.isChecked() else "terminate"),
            stochastic=self.chk_stoch.isChecked(),
            opponent=("slots" if self.spin_grid.value() > 1 else "teacher"),
            opponent_slots=(self.opp_table.slot_dicts() if self.spin_grid.value() > 1 else None),
            controller=SessionConfig.controller,
            estimator=SessionConfig.estimator,
            mu_mode=str(self.combo_mu.currentData() or "random"),
            mu=float(self.spin_mu.value()),
            ros2=str(self.combo_ros.currentData() or "off"),
            saliency=self.chk_saliency.isChecked(),
            internals=self.chk_internals.isChecked(),
            max_render_cars=MAX_RENDER_CARS,
            # The recording settings travel with the session so a saved config reproduces the clip.
            # They are excluded from `affects_simulation`, so changing one never restarts a run.
            record_dir=self.edit_record.text().strip(),
            record_width=int((self.combo_res.currentData() or (1280, 720))[0]),
            record_height=int((self.combo_res.currentData() or (1280, 720))[1]),
            record_fps=int(self.combo_fps.currentData() or 30),
            record_camera=str(self.combo_rec_cam.currentData() or ""),
            record_overlays=self.chk_rec_overlay.isChecked(),
            record_seconds=float(self.spin_rec_secs.value()),
            record_encoder=str(self.combo_rec_enc.currentData() or "auto"),
        )

    def _on_start(self):
        self.clear_error()
        self.save_prefs()              # the settings that were actually driven with
        self.btn_start.mark_pending()
        self._pending["start"] = time.monotonic()
        self.start_requested.emit(self.current_config())

    def _on_cancel(self):
        self.cancel_requested.emit()

    def _on_stop(self):
        self.btn_stop.mark_pending()
        self._pending["stop"] = time.monotonic()
        self.stop_requested.emit()

    def _on_reset(self):
        self.btn_reset.mark_pending()
        self._pending["reset"] = time.monotonic()
        self.viewport.notify("리셋", "info", 1.4)
        self.reset_requested.emit()

    def _on_pause_requested(self, want_paused: bool):
        self._pending["pause"] = time.monotonic()
        self.pause_requested.emit(want_paused)

    def _on_camera(self, key: str, from_shortcut: bool = False):
        # A camera angle is true the moment it is chosen: nothing in the worker decides it, so this
        # settles immediately rather than waiting for an ack it would never get.
        self.viewport.set_camera(key)
        if from_shortcut:
            self.camera_buttons.set_current(key)

    def sync_camera_buttons(self, key: str):
        self.camera_buttons.set_current(key)

    def _step_focus(self, delta: int):
        n = self.combo_focus.count()
        if n == 0:
            return
        i = (self.combo_focus.currentIndex() + delta) % n
        self.combo_focus.setCurrentIndex(i)
        self._on_focus_combo(i)

    def _on_focus_combo(self, index: int):
        cid = self.combo_focus.itemData(index)
        if cid is None:
            return
        self._pending["focus"] = time.monotonic()
        self.focus_requested.emit(int(cid))

    def _apply_local_overlays(self):
        v = self.viewport
        v.show_lidar = self.chk_lidar.isChecked()
        v.show_trails = self.chk_trails.isChecked()
        v.show_raceline = self.chk_raceline.isChecked()
        v.show_labels = self.chk_labels.isChecked()
        if not self.chk_plan.isChecked():
            v.plan = v.plan_pred = None
        v.update()
        self._emit_overlay_request()

    def _on_internals(self, on: bool):
        self.activation_panel.setVisible(on)
        if not on:
            self.activation_panel.clear()
        self._emit_overlay_request()

    def _emit_overlay_request(self):
        self.overlay_requested.emit({"saliency": self.chk_saliency.isChecked(),
                                     "internals": self.chk_internals.isChecked(),
                                     "plan": self.chk_plan.isChecked()})

    def _on_screenshot(self):
        """A PNG of the 3D view, at the resolution the 녹화 card is set to.

        The same offscreen render a recording uses, so a still and a frame of the clip are the same
        picture -- and so a 1080p screenshot does not depend on how big the window happens to be.
        """
        spec = self.record_settings()
        path = os.path.splitext(spec.path)[0] + ".png"
        ok = self.viewport.grab_png(path, spec.width, spec.height)
        self.status_text.setText(
            f"스크린샷 저장: {path} ({spec.width}×{spec.height})" if ok else "스크린샷 실패 (GL 화면 없음)")
        if ok:
            self._last_video = path

    def _show_error_detail(self):
        if not self._last_error:
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("오류 자세히")
        box.setText(self._last_error["message"])
        box.setDetailedText(self._last_error.get("detail", ""))
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.exec_()

    def show_help(self):
        rows = "".join(
            f"<tr><td style='padding:3px 14px 3px 0'><b>{k}</b></td><td style='padding:3px 0'>{v}</td></tr>"
            for k, v in [
                ("Space", "일시정지 / 재개"),
                ("H / F1", "이 도움말"),
                ("1 – 5", "카메라 (전체보기 · 추격 · 위에서 · 궤도 · 근접)"),
                ("[ , ]", "주시 차량 바꾸기"),
                ("Ctrl+R", "리셋"),
                ("Ctrl+.", "정지"),
                ("Ctrl+F", "맵 검색으로 이동"),
                ("S", "3D 화면 스크린샷"),
                ("R", "녹화 시작 / 중지"),
                ("드래그 / 휠", "궤도 카메라 회전 / 확대"),
                ("환경 페이지", "V 선택 · B 덕트 · W 벽 · E 지우개 · L/K 선 · M 사각형 · P 다각형 · A 배치 · "
                           "Ctrl+Z/Y 되돌리기 · F 전체 보기 · 5 위에서 · 가운데/Alt+드래그 회전 · "
                           "오른쪽/Shift+드래그 이동 · 휠 확대"),
            ])
        QtWidgets.QMessageBox.information(
            self, "도움말",
            "<p>모든 기능은 화면의 버튼으로 할 수 있습니다. 아래 단축키는 편의를 위한 것입니다.</p>"
            f"<table>{rows}</table>"
            "<p style='margin-top:12px'><b>sim 배속</b>은 worker가 잰 시뮬레이션 진행 속도이고, "
            "<b>렌더 fps</b>는 이 창이 그림을 바꾼 속도입니다. 서로 다른 값이며, 렌더가 빨라도 "
            "물리가 빨라진 것이 아닙니다.</p>")
