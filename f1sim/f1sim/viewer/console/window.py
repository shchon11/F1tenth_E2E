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
import time
from typing import Callable, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from . import catalog, theme
from .catalog import GROUP_CAVEAT, GROUP_HINT, GROUP_ORDER, MapCatalog, RunInfo, format_age
from .frames import Freshness, INTERP_LAG_FRAMES
from .overlays import ActivationPanel, DashPanel, PolicyInputPanel
from .protocol import (SessionConfig, STAGE_TEXT, STATE_FAILED, STATE_IDLE, STATE_PAUSED,
                       STATE_PREPARING, STATE_RUNNING, STATE_STOPPING)
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
    retry_requested = QtCore.pyqtSignal()
    close_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("f1sim 주행 콘솔")
        self.resize(1600, 950)
        self.setMinimumSize(1120, 700)

        self.state = STATE_IDLE
        self.generation = 0
        self.runs: List[RunInfo] = []
        self.maps = MapCatalog()
        self._selected_run: Optional[str] = None
        self._selected_map: Optional[str] = None
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
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.centre)
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
        head.setFixedHeight(56)
        h = QtWidgets.QHBoxLayout(head)
        h.setContentsMargins(SP[2], SP[1], SP[2], SP[1])
        h.setSpacing(SP[2])

        title = QtWidgets.QLabel("f1sim 주행 콘솔")
        title.setObjectName("HeaderTitle")
        h.addWidget(title)

        self.badge = StateBadge()
        h.addWidget(self.badge)

        self.header_summary = QtWidgets.QLabel("—")
        self.header_summary.setObjectName("HeaderSub")
        self.header_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        h.addWidget(self.header_summary, 1)

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
        run_card = Card("정책 런")
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

        # -- map
        map_card = Card("맵")
        self.map_group = QtWidgets.QComboBox()
        self.map_group.addItem("목록 읽는 중…")
        self.map_group.setEnabled(False)
        self.map_group.setToolTip(GROUP_CAVEAT)
        self.map_group.currentIndexChanged.connect(lambda _: self._refresh_map_list())
        map_card.add(self.map_group)
        self.map_list = FilterList("맵 이름 검색 (Ctrl+F)", rows=6)
        self.map_list.activated.connect(self._on_map_selected)
        map_card.add(self.map_list)
        self.map_note = label("선택한 맵 하나만 로드합니다.", "hint")
        self.map_note.setToolTip(GROUP_CAVEAT)
        map_card.add(self.map_note)
        self.map_list.tree.setToolTip(GROUP_CAVEAT)
        v.addWidget(map_card)

        # -- shape of the session
        cfg_card = Card("구성")
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
        v.addWidget(cfg_card)

        # -- advanced
        adv = Collapsible("고급 설정", expanded=False)
        # Not "CUDA 가속": the label said that and the box was ticked, so opening the viewer looked
        # like a choice about whether to use the GPU. It is not -- 연산 장치 decides that, and the
        # session runs on CUDA with this off. What this buys is `torch.compile`, which is a startup
        # cost paid before anything appears.
        self.chk_compile = QtWidgets.QCheckBox("torch.compile 사전 컴파일 (시작이 몇 분 느려짐)")
        self.chk_compile.setChecked(False)
        self.chk_compile.setToolTip(
            "GPU 사용 여부와는 무관합니다 — 그건 아래 '연산 장치' 가 정하고, 이 항목을 꺼도\n"
            "CUDA 로 돌아갑니다.\n\n"
            "켜면 torch 가 정책과 물리를 컴파일해 step 하나가 빨라집니다. 대신 세션을 시작할 때\n"
            "그 컴파일을 먼저 끝내야 하므로 화면이 나오기까지 몇 분이 걸릴 수 있습니다.\n"
            "차 한 대를 보는 용도라면 켜지 않는 편이 빠릅니다. 차가 많거나 오래 돌릴 때 이득입니다.\n\n"
            "준비 시간은 고정값이 아니라 torch 컴파일 캐시 상태에 좌우됩니다.\n"
            "준비 중에도 창은 계속 반응하며 [취소] 를 누를 수 있습니다.")
        adv.add(self.chk_compile)
        self.chk_dr = QtWidgets.QCheckBox("차량마다 마찰/지연 무작위화 (학습과 동일)")
        self.chk_dr.setChecked(True)
        self.chk_dr.setToolTip("끄면 모든 차가 공칭 파라미터로 달립니다. 미끄러짐이 정책 탓인지 "
                               "낮은 마찰 뽑기 탓인지 구분할 때 끄세요.")
        adv.add(self.chk_dr)
        self.chk_stoch = QtWidgets.QCheckBox("학습처럼 행동을 샘플링")
        adv.add(self.chk_stoch)
        self.combo_opponent = QtWidgets.QComboBox()
        self.combo_opponent.addItems(["teacher", "policy"])
        adv.add(FieldRow("상대차 주행 방식", self.combo_opponent,
                         "레이스당 차량 수가 2 이상일 때만 의미가 있습니다."))
        self.combo_device = QtWidgets.QComboBox()
        self.combo_device.addItems(["auto", "cuda", "cpu"])
        adv.add(FieldRow("연산 장치", self.combo_device, ""))
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
        bh = QtWidgets.QHBoxLayout(bar)
        bh.setContentsMargins(SP[2], SP[1], SP[2], SP[2])
        bh.setSpacing(SP[1])
        self.btn_start = PendingButton("시작", role="PrimaryButton")
        self.btn_start.setToolTip("선택한 런과 맵으로 세션을 시작합니다.")
        self.btn_start.clicked.connect(self._on_start)
        bh.addWidget(self.btn_start, 2)
        self.btn_cancel = PendingButton("취소")
        self.btn_cancel.setToolTip("준비 중인 세션을 중단합니다. 준비 중에도 계속 누를 수 있습니다.")
        self.btn_cancel.clicked.connect(self._on_cancel)
        bh.addWidget(self.btn_cancel, 1)
        wv.addWidget(bar)
        wrap.setMaximumWidth(420)
        return wrap

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
        h.addWidget(label("카메라", "field"))
        self.camera_buttons = SegmentedButtons(CAMERA_MODES)
        self.camera_buttons.set_current("overview")
        self.camera_buttons.selected.connect(self._on_camera)
        h.addWidget(self.camera_buttons)

        _, h = group()
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
        self.btn_shot.setToolTip("현재 3D 화면을 PNG로 저장합니다.  (S)")
        self.btn_shot.clicked.connect(self._on_screenshot)
        h.addWidget(self.btn_shot)
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
        if self._closing_allowed:
            ev.accept()
            return
        ev.ignore()
        self._closing_allowed = False
        self.close_requested.emit()

    def allow_close(self):
        self._closing_allowed = True
        self.viewport.teardown()      # while the context still exists
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
        driving = self.state in (STATE_RUNNING, STATE_PAUSED)
        if (not self._sidebar_autofold_done and driving and self.width() < self.NARROW_W
                and self.btn_left_panel.isChecked()):
            self.btn_left_panel.setChecked(False)
            self._sidebar_autofold_done = True
            self.status_text.setText(
                "창이 좁아 설정 패널을 접었습니다. 위의 '설정 패널' 버튼으로 다시 펼 수 있습니다.")

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

        show = Card("표시")
        self.chk_lidar = QtWidgets.QCheckBox("LiDAR 점")
        self.chk_lidar.setChecked(True)
        self.chk_plan = QtWidgets.QCheckBox("플랜 / 예측 궤적")
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

        sc("Space", lambda: self.btn_pause.click() if self.btn_pause.isEnabled() else None)
        for i, key in enumerate(CAMERA_KEYS):
            sc(str(i + 1), lambda k=key: self._on_camera(k, from_shortcut=True))
        sc("[", lambda: self._step_focus(-1))
        sc("]", lambda: self._step_focus(1))
        sc("Ctrl+R", lambda: self.btn_reset.click() if self.btn_reset.isEnabled() else None)
        sc("Ctrl+.", lambda: self.btn_stop.click() if self.btn_stop.isEnabled() else None)
        sc("Ctrl+F", lambda: self.map_list.search.setFocus(QtCore.Qt.ShortcutFocusReason))
        sc("S", self._on_screenshot)
        sc("F1", self.show_help)

    # ================================================================ data in
    def set_runs(self, runs: List[RunInfo]):
        self.runs = runs
        items = [("", r.name, r.name, r.subtitle) for r in runs]
        self.run_list.set_items(items)
        if not runs:
            self.run_list.set_status(f"{catalog.RUNS_DIR} 에 체크포인트가 있는 런이 없습니다.", "warn")
        self._update_start_enabled()

    def set_maps(self, cat: MapCatalog):
        self.maps = cat
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
        self.map_group.blockSignals(False)
        self._refresh_map_list()
        self._update_start_enabled()

    def _refresh_map_list(self):
        g = self.map_group.currentData()
        if not self.maps.ready or g is None:
            self.map_list.set_items([])
            self.map_list.set_status("맵 목록을 읽는 중입니다…" if not self.maps.error else "", "hint")
            return
        names = self.maps.groups.get(g, [])
        self.map_list.set_items([("", n, n, "") for n in names])
        hint = GROUP_HINT.get(g, "")
        self.map_list.set_status(f"{len(names)} 개 · {hint}" if hint else f"{len(names)} 개")

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
        self.btn_cancel.setEnabled(preparing)
        self.btn_pause.setEnabled(running)
        self.btn_reset.setEnabled(running)
        self.btn_stop.setEnabled(running or preparing)
        self.combo_focus.setEnabled(running)
        self.btn_focus_prev.setEnabled(running)
        self.btn_focus_next.setEnabled(running)
        self.btn_shot.setEnabled(running or preparing)

        # settings stay editable during PREPARING on purpose: waiting is exactly when someone
        # realises they picked the wrong map
        for w in (self.run_list, self.map_list, self.map_group, self.spin_races, self.spin_grid,
                  self.spin_cap, self.chk_compile, self.chk_dr, self.chk_stoch, self.combo_opponent,
                  self.combo_device):
            w.setEnabled(state in (STATE_IDLE, STATE_FAILED, STATE_PREPARING))

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
            self.status_text.setText("대기 — 런과 맵을 고르고 시작을 누르세요.")
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
        mp = facts.get("map", "—")
        cars = facts.get("total_cars", 0)
        races = facts.get("races", 1)
        grid = facts.get("cars_per_race", 1)
        shown = min(cars, facts.get("max_render_cars", MAX_RENDER_CARS))
        device = facts.get("device", "?")
        summary = f"{run}  ·  {mp}  ·  {races}레이스 × {grid}대 = {cars}대"
        if shown < cars:
            summary += f" (화면 {shown}대)"
        summary += f"  ·  {device}  ·  세션 #{self.generation}"
        self.header_summary.setText(summary)
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
        hint = ""
        if self._stage == "compile" and total > 20:
            # the one stage that legitimately takes minutes, and the only one worth explaining
            hint = ("  torch.compile 사전 컴파일은 몇 분이 걸릴 수 있습니다. "
                    "'고급 설정 > torch.compile 사전 컴파일' 을 끄면 즉시 시작합니다 — "
                    "끈 상태에서도 CUDA 그래프는 1초 남짓에 캡처되어 그대로 쓰입니다.")
        self.status_text.setText(f"준비 중 {total:.0f}초 — {text}{note}{here}.  취소할 수 있습니다.{hint}")

    # ================================================================ user actions
    def _can_start(self) -> bool:
        return bool(self._selected_run and self._selected_map)

    def _update_start_enabled(self):
        if self.state in (STATE_IDLE, STATE_FAILED):
            self.btn_start.setEnabled(self._can_start())

    def _on_run_selected(self, name: str):
        self._selected_run = name
        self.sel_run.setText(f"런: {name}")
        self.sel_run.setToolTip(name)
        self._update_selection_note()
        self._update_start_enabled()
        self.describe_requested.emit(name)

    def _on_map_selected(self, name: str):
        self._selected_map = name
        self.sel_map.setText(f"맵: {name}")
        self.sel_map.setToolTip(name)
        self._update_selection_note()
        self._update_start_enabled()

    def _update_selection_note(self):
        g = self.maps.group_of(self._selected_map) if self._selected_map else None
        self.sel_note.setText(f"{g} · 학습 이력 미확인" if g else "")
        self._update_running_note()

    def _update_running_note(self):
        """Say so when the picker no longer describes what is on screen."""
        running = self._running_facts
        if not running or self.state not in (STATE_RUNNING, STATE_PAUSED):
            self.sel_running.setText("")
            return
        differs = (self._selected_run != running.get("run")
                   or self._selected_map != running.get("map"))
        if differs:
            self.sel_running.setText(
                f"화면에서 도는 것은 {running.get('run', '?')} / {running.get('map', '?')} 입니다. "
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

    def current_config(self) -> SessionConfig:
        return SessionConfig(
            run=self._selected_run or "latest",
            map_name=self._selected_map or "",
            races=self.spin_races.value(),
            cars_per_race=self.spin_grid.value(),
            speed_cap=float(self.spin_cap.value()),
            device=self.combo_device.currentText(),
            compile=self.chk_compile.isChecked(),
            randomize=self.chk_dr.isChecked(),
            stochastic=self.chk_stoch.isChecked(),
            opponent=self.combo_opponent.currentText(),
            saliency=self.chk_saliency.isChecked(),
            internals=self.chk_internals.isChecked(),
            max_render_cars=MAX_RENDER_CARS,
        )

    def _on_start(self):
        self.clear_error()
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
        import os
        path = os.path.join(os.path.expanduser("~"), f"f1sim_{int(time.time())}.png")
        ok = self.viewport.grab_png(path)
        self.status_text.setText(f"스크린샷 저장: {path}" if ok else "스크린샷 실패 (GL 화면 없음)")

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
                ("1 – 5", "카메라 (전체보기 · 추격 · 위에서 · 궤도 · 근접)"),
                ("[ , ]", "주시 차량 바꾸기"),
                ("Ctrl+R", "리셋"),
                ("Ctrl+.", "정지"),
                ("Ctrl+F", "맵 검색으로 이동"),
                ("S", "3D 화면 스크린샷"),
                ("드래그 / 휠", "궤도 카메라 회전 / 확대"),
            ])
        QtWidgets.QMessageBox.information(
            self, "도움말",
            "<p>모든 기능은 화면의 버튼으로 할 수 있습니다. 아래 단축키는 편의를 위한 것입니다.</p>"
            f"<table>{rows}</table>"
            "<p style='margin-top:12px'><b>sim 배속</b>은 worker가 잰 시뮬레이션 진행 속도이고, "
            "<b>렌더 fps</b>는 이 창이 그림을 바꾼 속도입니다. 서로 다른 값이며, 렌더가 빨라도 "
            "물리가 빨라진 것이 아닙니다.</p>")
