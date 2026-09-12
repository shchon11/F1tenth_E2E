"""The training page of the console: set up a PPO run, launch it, watch it, open its checkpoints.

Three parts, none of which imports torch -- training runs in its own process and the page only
reads files:

* `JobManager` starts `python -m f1sim.learn.ppo` detached (its own session, so closing the console
  does not kill a run), keeps one JSON record per job under `<runs>/_console_jobs/`, and stops a job
  with SIGINT first and SIGTERM after a grace period. The trainer does not save on interrupt: what
  survives is the last periodic checkpoint (`--save-every`).
* `parse_progress` reads the trainer's own progress lines (`upd k/N steps … | rew/step … coll …/km
  prog … m lap … s | … kl_ref … | … steps/s`), from the job log or from the run's W&B
  `output.log`, so runs started outside the console are just as watchable.
* `TrainingPage` is the widget: a recipe form with presets on the left, live charts and the
  checkpoint list in the middle, jobs on the right. "주행 화면에서 보기" hands a checkpoint to the
  driving page.

The presets encode what the 2026-09-12 experiments established: finetune from the frozen original,
restore Adam, keep the learning rate low, and **train under the legacy controller** -- a policy
trained with the grip clamp in the loop learned to lean on it and lost avoidance and overtaking.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from . import catalog, theme
from .theme import C, SP
from .widgets import Card, Collapsible, FieldRow, KeyValueList, MetricTile, hline, label

JOBS_DIRNAME = "_console_jobs"
REPO_F1SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FROZEN_ORIGINAL = os.path.join(catalog.RUNS_DIR, "_baselines", "frozen_original_48cc698f.pt")
DEFAULT_ESTIMATOR = os.path.join(catalog.RUNS_DIR, "_estimators", "estimator_seed401.pt")

_UPD = re.compile(
    r"upd (?P<k>\d+)/(?P<n>\d+) steps (?P<steps>[\d.]+)M cap (?P<cap>[\d.]+) \| rew/step (?P<rew>[-\d.]+|nan) "
    r"coll (?P<coll>[-\d.]+|nan)/km prog (?P<prog>[-\d.]+|nan) m lap (?P<lap>[-\d.]+|nan) s \| gate (?P<gate>[-\d.]+|nan) "
    r"\((?P<tk>[-\d.]+) tk\) \| kl_ref (?P<kl>[-\d.]+|nan) \| (?P<sps>\d+) steps/s")


# ================================================================ recipes
COMMON_FLAGS = (
    "--action-mode plan --horizon 32 --minibatch 1024 --epochs 3 --cap0 9.0 --cap1 9.0 --cap-steps 5000000.0 "
    "--cap-gate 4.0 --cap-gate-quantile 0.9 --cap-gate-min-km 1.0 --kl-decay 40000000.0 --collision-penalty 10.0 "
    "--steer-penalty 0.05 --proximity-penalty 0.5 --safe-dist 0.3 --sideslip-penalty 0.5 --lap-bonus 5.0 "
    "--lap-time-bonus 2.0 --collision-speed-penalty 0.5 --critic-warmup 10 --episode-s 40.0 --scan-stack 6 "
    "--scan-stride 1 --hist-len 20 --gamma 0.99 --lam 0.95 --clip 0.2 --ent 0.0 --vf 0.5 --max-grad 0.5 "
    "--log-every 1 --amp --wandb-new")

SGR_TRACKS = "real:icra2022,real:blackbox2021_2,real:blackbox2022_1,rt:Spielberg,gen:control:1400"
R10_TRACKS = ("real:icra2022,real:icra2022~rev,real:blackbox2021_2,real:blackbox2021_2~rev,real:blackbox2022_1,"
              "real:blackbox2022_1~rev,rt:Spielberg,rt:Spielberg~rev,gen:control:1400,gen:control:1400~rev")


@dataclass
class Recipe:
    key: str
    title: str
    note: str
    tracks: str
    race_size: int = 1
    opponent: str = "teacher"
    aux_grip: float = 0.0
    aux_opp: float = 0.0
    race_flags: str = ""            # opponent rewards etc., only meaningful with race_size > 1
    lr: float = 5e-5
    lr_end: float = 2e-5
    kl: float = 0.05
    envs: int = 256
    total: int = 1_048_576


RECIPES: List[Recipe] = [
    Recipe("origrecipe", "원본 레이스 레시피 (권장)",
           "원본 정책이 학습된 조건 그대로: 149개 맵 변형(역방향·거울·장애물 포함), 2대 레이스, 혼합 상대차, "
           "보조 head. legacy 제어기로 학습해 배포 때만 클램프를 붙입니다. 2026-09-12 벤치마크에서 "
           "원본+추정 제어기와 동급 이상이었던 유일한 재학습 레시피입니다.",
           tracks="train", race_size=2, opponent="mixed", aux_grip=1.0, aux_opp=1.0,
           race_flags="--mixed-teacher-frac 0.5 --opp-speed 0.5 1.0 --overtake-bonus 1.0 "
                      "--car-proximity-penalty 0.8 --car-safe-gap 0.9 --car-contact-penalty 5.0"),
    Recipe("sgr", "SGR base (5개 맵 단독)",
           "노면 대응 스크린에 쓴 좁은 레시피. 학습 맵에서는 좋아지지만 진단 맵의 회피·추월은 잃었습니다. "
           "빠른 비교용.", tracks=SGR_TRACKS),
    Recipe("r10", "R10 (5개 맵 + 역방향)",
           "SGR 맵에 역방향을 더한 10개 경로. 결과는 SGR과 같은 방향이었습니다.", tracks=R10_TRACKS),
    Recipe("custom", "사용자 정의", "아래 값을 직접 정합니다.", tracks="train"),
]


# ================================================================ jobs
@dataclass
class Job:
    name: str
    pid: int
    argv: List[str]
    log: str
    started: float
    run_dir: str
    stop_requested: float = 0.0
    external: bool = False          # found in the process table, not started by this console

    @property
    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except OSError:
            return False
        # a zombie of ours answers kill(0) until reaped; the launcher reaps below
        return True


class JobManager:
    """Start, list and stop detached training processes. Pure files and signals."""

    def __init__(self, runs_dir: str = catalog.RUNS_DIR):
        self.runs_dir = runs_dir
        self.jobs_dir = os.path.join(runs_dir, JOBS_DIRNAME)
        self._procs: Dict[str, subprocess.Popen] = {}

    def _record(self, name: str) -> str:
        return os.path.join(self.jobs_dir, f"{name}.json")

    def launch(self, name: str, argv: List[str], device: str = "cuda") -> Job:
        os.makedirs(self.jobs_dir, exist_ok=True)
        run_dir = os.path.join(self.runs_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        log = os.path.join(run_dir, "console-train.log")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   PYTHONUNBUFFERED="1")
        env.pop("CUDA_VISIBLE_DEVICES", None) if device == "cuda" else env.update(CUDA_VISIBLE_DEVICES="")
        fh = open(log, "ab")
        fh.write(("$ " + " ".join(shlex.quote(a) for a in argv) + "\n").encode())
        proc = subprocess.Popen(argv, cwd=REPO_F1SIM, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        fh.close()
        job = Job(name=name, pid=proc.pid, argv=list(argv), log=log, started=time.time(), run_dir=run_dir)
        self._procs[name] = proc
        with open(self._record(name), "w") as f:
            json.dump({"name": name, "pid": job.pid, "argv": job.argv, "log": log, "started": job.started,
                       "run_dir": run_dir}, f, indent=1)
        return job

    def list_jobs(self) -> List[Job]:
        out: List[Job] = []
        try:
            names = sorted(os.listdir(self.jobs_dir))
        except OSError:
            return out
        for fn in names:
            if not fn.endswith(".json"):
                continue
            try:
                d = json.load(open(os.path.join(self.jobs_dir, fn)))
                out.append(Job(name=d["name"], pid=int(d["pid"]), argv=list(d.get("argv") or []),
                               log=d.get("log", ""), started=float(d.get("started", 0.0)),
                               run_dir=d.get("run_dir", ""), stop_requested=float(d.get("stop_requested", 0.0))))
            except Exception:
                continue
        # reap children we started so `alive` reflects the truth for them
        for name, p in list(self._procs.items()):
            if p.poll() is not None:
                self._procs.pop(name, None)
        # Trainers started from a shell or another tool: visible in the process table by their
        # module name, named by their own --name. Shown as external; stopped by pid, never by
        # process group, because their group is whatever shell started them.
        known = {(j.name, j.pid) for j in out}
        for j in discover_external_jobs(self.runs_dir):
            if (j.name, j.pid) not in known and not any(k.name == j.name and k.alive for k in out):
                out.append(j)
        out.sort(key=lambda j: j.started, reverse=True)
        return out

    def stop(self, job: Job, grace_s: float = 12.0) -> str:
        """SIGINT the process group; SIGTERM once the grace period has passed on a repeat call.
        Returns what was sent. The last periodic checkpoint is what remains of the run."""
        if not job.alive:
            return "이미 종료됨"
        now = time.time()
        sent = "SIGINT"
        kill = (lambda sig: os.kill(job.pid, sig)) if job.external else (lambda sig: os.killpg(job.pid, sig))
        try:
            if job.stop_requested and now - job.stop_requested > grace_s:
                kill(signal.SIGTERM)
                sent = "SIGTERM"
            else:
                kill(signal.SIGINT)
        except ProcessLookupError:
            return "이미 종료됨"
        if job.external:
            job.stop_requested = job.stop_requested or now
            return sent
        try:
            path = self._record(job.name)
            d = json.load(open(path))
            d["stop_requested"] = job.stop_requested or now
            json.dump(d, open(path, "w"), indent=1)
        except Exception:
            pass
        return sent

    def forget(self, job: Job) -> None:
        """Drop the record of a finished job (the run directory is untouched)."""
        if job.alive or job.external:
            return
        try:
            os.remove(self._record(job.name))
        except OSError:
            pass


def discover_external_jobs(runs_dir: str = catalog.RUNS_DIR) -> List[Job]:
    """PPO trainers in the process table, whoever started them. Linux /proc only; empty elsewhere."""
    out: List[Job] = []
    try:
        pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return out
    me = os.getpid()
    for pid in pids:
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if not argv or "f1sim.learn.ppo" not in argv or not os.path.basename(argv[0]).startswith("python"):
            continue
        name = argv[argv.index("--name") + 1] if "--name" in argv and argv.index("--name") + 1 < len(argv) else f"pid{pid}"
        run_dir = os.path.join(runs_dir, name)
        try:
            started = os.stat(f"/proc/{pid}").st_ctime
        except OSError:
            started = time.time()
        out.append(Job(name=name, pid=pid, argv=argv, log=run_log_path(run_dir) or "", started=started,
                       run_dir=run_dir, external=True))
    return out


def list_run_dirs(runs_dir: str = catalog.RUNS_DIR) -> List[Tuple[str, str, str, float]]:
    """(name, path, subtitle, mtime) for every run worth watching: those with a checkpoint (the
    picker's runs) plus those with only a progress log so far -- a run in its first minutes."""
    seen = {}
    for r in catalog.list_runs(runs_dir):
        seen[r.name] = (r.name, r.path, r.subtitle, r.mtime)
    try:
        entries = list(os.scandir(runs_dir))
    except OSError:
        entries = []
    for e in entries:
        if not e.is_dir() or e.name in seen or e.name.startswith("_"):
            continue
        log = run_log_path(e.path)
        if log:
            mt = os.path.getmtime(log)
            seen[e.name] = (e.name, e.path, f"기록만 · {catalog.format_age(time.time() - mt)} 전 (체크포인트 아직 없음)", mt)
    return sorted(seen.values(), key=lambda t: t[3], reverse=True)


# ================================================================ progress parsing
@dataclass
class Progress:
    update: int = 0
    n_updates: int = 0
    steps_m: float = 0.0
    cap: float = 0.0
    rew: List[float] = field(default_factory=list)
    coll: List[float] = field(default_factory=list)
    prog: List[float] = field(default_factory=list)
    lap: List[float] = field(default_factory=list)
    kl: List[float] = field(default_factory=list)
    sps: List[float] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)      # the tail, for the log box
    finished: bool = False
    error: Optional[str] = None

    @property
    def fraction(self) -> float:
        return (self.update / self.n_updates) if self.n_updates else 0.0


def _f(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_progress(text: str, tail: int = 60) -> Progress:
    p = Progress()
    lines = text.splitlines()
    for ln in lines:
        m = _UPD.search(ln)
        if m:
            p.update, p.n_updates = int(m["k"]), int(m["n"])
            p.steps_m, p.cap = _f(m["steps"]), _f(m["cap"])
            p.rew.append(_f(m["rew"])); p.coll.append(_f(m["coll"])); p.prog.append(_f(m["prog"]))
            p.lap.append(_f(m["lap"])); p.kl.append(_f(m["kl"])); p.sps.append(_f(m["sps"]))
        elif "Traceback" in ln or ln.startswith("SystemExit") or "Error:" in ln and "wandb" not in ln:
            p.error = ln.strip()[:200]
    if p.n_updates and p.update >= p.n_updates:
        p.finished = True
    p.lines = [ln for ln in lines if not ln.startswith("wandb:")][-tail:]
    return p


def run_log_path(run_dir: str) -> Optional[str]:
    """The newest progress log for a run: the console's own if the console launched it, else the
    W&B `output.log` the trainer writes for every run."""
    cands = []
    own = os.path.join(run_dir, "console-train.log")
    if os.path.isfile(own):
        cands.append(own)
    wb = os.path.join(run_dir, "wandb")
    try:
        for d in os.listdir(wb):
            p = os.path.join(wb, d, "files", "output.log")
            if d.startswith("run-") and os.path.isfile(p):
                cands.append(p)
    except OSError:
        pass
    if not cands:
        return None
    return max(cands, key=lambda p: os.path.getmtime(p))


def list_checkpoints(run_dir: str) -> List[Tuple[str, str, float]]:
    """(label, path, mtime) for every checkpoint in a run, newest last, update-sorted."""
    out = []
    try:
        for fn in os.listdir(run_dir):
            if fn.startswith("ppo_") and fn.endswith(".pt"):
                p = os.path.join(run_dir, fn)
                out.append((fn, p, os.path.getmtime(p)))
    except OSError:
        return out

    def key(t):
        m = re.match(r"ppo_u(\d+)\.pt", t[0])
        return (0, int(m.group(1))) if m else (1, {"ppo_latest.pt": 0, "ppo_final.pt": 1}.get(t[0], 2))
    return sorted(out, key=key)


# ================================================================ chart widget
class LineChart(QtWidgets.QWidget):
    """One series against update index. Draws the range and the last value; nothing is smoothed."""

    def __init__(self, title: str, unit: str = "", colour: str = "", lower_is_better: bool = False, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.title, self.unit = title, unit
        self.colour = colour or C["accent"]
        self.lower_is_better = lower_is_better
        self._y: List[float] = []

    def set_series(self, y: Sequence[float]):
        self._y = [float(v) for v in y]
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QtGui.QColor(C["bg.window"]))
        f = p.font(); f.setPointSizeF(8.0); p.setFont(f)
        p.setPen(QtGui.QColor(C["text.1"]))
        p.drawText(QtCore.QRectF(8, 4, w - 16, 16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.title)
        ys = [v for v in self._y if v == v]                     # drop nan
        if len(ys) < 2:
            p.setPen(QtGui.QColor(C["text.2"]))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "데이터 없음")
            return
        lo, hi = min(ys), max(ys)
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        x0, x1, y0, y1 = 48, w - 10, 30, h - 18
        p.setPen(QtGui.QPen(QtGui.QColor(C["line"]), 1))
        for k in range(4):
            yy = y0 + (y1 - y0) * k / 3
            p.drawLine(QtCore.QPointF(x0, yy), QtCore.QPointF(x1, yy))
        mono = QtGui.QFont(theme.MONO_FONT); mono.setPointSizeF(7.0); p.setFont(mono)
        p.setPen(QtGui.QColor(C["text.2"]))
        p.drawText(QtCore.QRectF(0, y0 - 7, x0 - 4, 14), QtCore.Qt.AlignRight, f"{hi:.3g}")
        p.drawText(QtCore.QRectF(0, y1 - 7, x0 - 4, 14), QtCore.Qt.AlignRight, f"{lo:.3g}")
        n = len(self._y)
        pts = []
        for i, v in enumerate(self._y):
            if v != v:
                continue
            x = x0 + (x1 - x0) * (i / max(1, n - 1))
            y = y1 - (y1 - y0) * ((v - lo) / (hi - lo))
            pts.append(QtCore.QPointF(x, y))
        p.setPen(QtGui.QPen(QtGui.QColor(self.colour), 1.6))
        p.drawPolyline(QtGui.QPolygonF(pts))
        last = pts[-1]
        p.setBrush(QtGui.QColor(self.colour)); p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(last, 3, 3)
        p.setPen(QtGui.QColor(C["text.0"]))
        mono.setPointSizeF(9.0); mono.setBold(True); p.setFont(mono)
        p.drawText(QtCore.QRectF(x0, 4, x1 - x0, 16), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                   f"{ys[-1]:.3g} {self.unit}".strip())
        p.setPen(QtGui.QColor(C["text.2"]))
        f.setPointSizeF(7.0); p.setFont(f)
        p.drawText(QtCore.QRectF(x0, y1 + 2, x1 - x0, 14), QtCore.Qt.AlignLeft, "update →")
        if self.lower_is_better:
            p.drawText(QtCore.QRectF(x0, y1 + 2, x1 - x0, 14), QtCore.Qt.AlignRight, "낮을수록 좋음")


# ================================================================ recipe form
class RecipeForm(QtWidgets.QWidget):
    launch_requested = QtCore.pyqtSignal(str, list, str)     # name, argv, device

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(SP[1])
        card = Card("학습 설정")
        self.combo_recipe = QtWidgets.QComboBox()
        for r in RECIPES:
            self.combo_recipe.addItem(r.title, r.key)
        self.combo_recipe.currentIndexChanged.connect(self._apply_recipe)
        card.add(FieldRow("레시피", self.combo_recipe, ""))
        self.recipe_note = label("", "hint")
        card.add(self.recipe_note)

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.setObjectName("SearchBox")
        card.add(FieldRow("런 이름", self.edit_name, "~/f1sim_runs/<이름>/ 에 저장됩니다."))
        self.combo_init = QtWidgets.QComboBox()
        self.combo_init.setEditable(True)
        card.add(FieldRow("시작 체크포인트", self.combo_init,
                          "기본은 동결된 원본 정책입니다. 다른 런의 ppo_final.pt 를 고르거나 경로를 직접 적을 수 있습니다."))
        self.spin_seed = QtWidgets.QSpinBox(); self.spin_seed.setRange(0, 99999); self.spin_seed.setValue(701)
        self.spin_total = QtWidgets.QSpinBox(); self.spin_total.setRange(8192, 200_000_000); self.spin_total.setSingleStep(65536)
        self.spin_total.setValue(1_048_576); self.spin_total.setGroupSeparatorShown(True)
        self.spin_envs = QtWidgets.QSpinBox(); self.spin_envs.setRange(8, 4096); self.spin_envs.setValue(256)
        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(SP[1]); grid.setVerticalSpacing(SP[0])
        grid.addWidget(FieldRow("시드", self.spin_seed, ""), 0, 0)
        grid.addWidget(FieldRow("총 스텝", self.spin_total, ""), 0, 1)
        grid.addWidget(FieldRow("병렬 환경 수", self.spin_envs, ""), 0, 2)
        card.add(grid)

        self.edit_lr = QtWidgets.QLineEdit("5e-5"); self.edit_lr_end = QtWidgets.QLineEdit("2e-5")
        self.edit_kl = QtWidgets.QLineEdit("0.05")
        g2 = QtWidgets.QGridLayout(); g2.setHorizontalSpacing(SP[1]); g2.setVerticalSpacing(SP[0])
        g2.addWidget(FieldRow("학습률", self.edit_lr, ""), 0, 0)
        g2.addWidget(FieldRow("학습률 (끝)", self.edit_lr_end, ""), 0, 1)
        g2.addWidget(FieldRow("KL 계수", self.edit_kl, "원본 정책에서 벗어나는 것을 억제합니다."), 0, 2)
        card.add(g2)

        self.chk_restore_opt = QtWidgets.QCheckBox("Adam 상태 복원 (권장)")
        self.chk_restore_opt.setChecked(True)
        self.chk_restore_opt.setToolTip("끄면 --fresh-opt: 새 옵티마이저로 시작합니다. 이전 실험들은 전부 이 상태로 돌았습니다.")
        card.add(self.chk_restore_opt)
        v.addWidget(card)

        adv = Collapsible("맵·레이스·제어기", expanded=False)
        self.edit_tracks = QtWidgets.QLineEdit("train")
        self.edit_tracks.setObjectName("SearchBox")
        adv.add(FieldRow("트랙", self.edit_tracks, "'train' = 전체 학습 카탈로그(149), 또는 쉼표로 나열"))
        self.spin_race = QtWidgets.QSpinBox(); self.spin_race.setRange(1, 4); self.spin_race.setValue(2)
        self.combo_opp = QtWidgets.QComboBox(); self.combo_opp.addItems(["mixed", "teacher", "policy"])
        g3 = QtWidgets.QGridLayout(); g3.setHorizontalSpacing(SP[1])
        g3.addWidget(FieldRow("레이스당 차량", self.spin_race, ""), 0, 0)
        g3.addWidget(FieldRow("상대차", self.combo_opp, ""), 0, 1)
        adv.add(g3)
        self.edit_aux_grip = QtWidgets.QLineEdit("1.0"); self.edit_aux_opp = QtWidgets.QLineEdit("1.0")
        g4 = QtWidgets.QGridLayout(); g4.setHorizontalSpacing(SP[1])
        g4.addWidget(FieldRow("aux grip", self.edit_aux_grip, "마찰 보조 head 가중치"), 0, 0)
        g4.addWidget(FieldRow("aux opp", self.edit_aux_opp, "상대차 보조 head 가중치"), 0, 1)
        adv.add(g4)
        self.combo_controller = QtWidgets.QComboBox(); self.combo_controller.addItems(["legacy", "fixed_low", "estimated"])
        self.combo_controller.currentTextChanged.connect(self._controller_hint)
        adv.add(FieldRow("학습 중 제어기", self.combo_controller, ""))
        self.ctrl_hint = label("", "hint.warn")
        adv.add(self.ctrl_hint)
        self.edit_estimator = QtWidgets.QLineEdit(DEFAULT_ESTIMATOR if os.path.isfile(DEFAULT_ESTIMATOR) else "")
        adv.add(FieldRow("노면 추정기", self.edit_estimator, "estimated 전용"))
        self.combo_wandb = QtWidgets.QComboBox(); self.combo_wandb.addItems(["online", "offline", "disabled"])
        self.combo_device = QtWidgets.QComboBox(); self.combo_device.addItems(["cuda", "cpu"])
        self.spin_save = QtWidgets.QSpinBox(); self.spin_save.setRange(1, 200); self.spin_save.setValue(10)
        g5 = QtWidgets.QGridLayout(); g5.setHorizontalSpacing(SP[1])
        g5.addWidget(FieldRow("W&B", self.combo_wandb, ""), 0, 0)
        g5.addWidget(FieldRow("장치", self.combo_device, ""), 0, 1)
        g5.addWidget(FieldRow("저장 주기 (업데이트)", self.spin_save, ""), 0, 2)
        adv.add(g5)
        self.edit_extra = QtWidgets.QLineEdit()
        self.edit_extra.setObjectName("SearchBox")
        adv.add(FieldRow("추가 인자", self.edit_extra, "그대로 명령 끝에 붙습니다."))
        v.addWidget(adv)

        prev = Collapsible("명령 미리보기", expanded=False)
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(120)
        self.preview.setStyleSheet(f"font-family: '{theme.MONO_FONT}', monospace; font-size: 10px; background: {C['bg.window']};")
        prev.add(self.preview)
        v.addWidget(prev)

        self.btn_launch = QtWidgets.QPushButton("학습 시작")
        self.btn_launch.setObjectName("PrimaryButton")
        self.btn_launch.clicked.connect(self._launch)
        v.addWidget(self.btn_launch)
        self.launch_note = label("", "hint")
        v.addWidget(self.launch_note)
        v.addStretch(1)

        for w in (self.edit_name, self.edit_lr, self.edit_lr_end, self.edit_kl, self.edit_tracks, self.edit_aux_grip,
                  self.edit_aux_opp, self.edit_estimator, self.edit_extra):
            w.textChanged.connect(self._refresh_preview)
        for w in (self.spin_seed, self.spin_total, self.spin_envs, self.spin_race, self.spin_save):
            w.valueChanged.connect(self._refresh_preview)
        for w in (self.combo_opp, self.combo_controller, self.combo_wandb, self.combo_device, self.combo_init):
            w.currentTextChanged.connect(self._refresh_preview)
        self.chk_restore_opt.toggled.connect(self._refresh_preview)
        self.set_runs([])
        self._apply_recipe(0)

    # -- state
    def set_runs(self, runs: List[catalog.RunInfo]):
        cur = self.combo_init.currentText()
        self.combo_init.blockSignals(True)
        self.combo_init.clear()
        self.combo_init.addItem("동결 원본 (frozen_original_48cc698f.pt)", FROZEN_ORIGINAL)
        for r in runs:
            fin = os.path.join(r.path, "ppo_final.pt")
            if os.path.isfile(fin):
                self.combo_init.addItem(f"{r.name}/ppo_final.pt", fin)
        self.combo_init.blockSignals(False)
        if cur:
            i = self.combo_init.findText(cur)
            self.combo_init.setCurrentIndex(max(0, i))
        self._refresh_preview()

    def _init_path(self) -> str:
        i = self.combo_init.currentIndex()
        data = self.combo_init.itemData(i) if i >= 0 and self.combo_init.itemText(i) == self.combo_init.currentText() else None
        return str(data) if data else self.combo_init.currentText().strip()

    def _apply_recipe(self, _idx):
        r = next(x for x in RECIPES if x.key == self.combo_recipe.currentData())
        self.recipe_note.setText(r.note)
        self.edit_tracks.setText(r.tracks)
        self.spin_race.setValue(r.race_size)
        self.combo_opp.setCurrentText(r.opponent)
        self.edit_aux_grip.setText(f"{r.aux_grip:g}"); self.edit_aux_opp.setText(f"{r.aux_opp:g}")
        self.edit_lr.setText(f"{r.lr:g}"); self.edit_lr_end.setText(f"{r.lr_end:g}"); self.edit_kl.setText(f"{r.kl:g}")
        self.spin_envs.setValue(r.envs); self.spin_total.setValue(r.total)
        self.combo_controller.setCurrentText("legacy")
        self.edit_name.setText(f"cl_{r.key}_legacy_s{self.spin_seed.value()}_{time.strftime('%m%d%H%M')}")
        self._refresh_preview()

    def _controller_hint(self, arm: str):
        if arm == "legacy":
            self.ctrl_hint.setText("")
        else:
            self.ctrl_hint.setText("주의: 클램프를 켠 채 학습한 정책은 클램프에 기대는 계획을 배워 저마찰·회피·추월이 "
                                   "나빠졌습니다 (2026-09-12, 4개 실행 모두). legacy로 학습하고 배포 때만 켜세요. "
                                   "또한 estimated/fixed_low 는 레이스당 차량 1에서만 학습됩니다.")
        self._refresh_preview()

    def argv(self) -> Tuple[str, List[str], str]:
        r = next(x for x in RECIPES if x.key == self.combo_recipe.currentData())
        name = self.edit_name.text().strip() or f"cl_run_{time.strftime('%m%d%H%M')}"
        race = self.spin_race.value()
        arm = self.combo_controller.currentText()
        parts = [sys.executable, "-m", "f1sim.learn.ppo"] + shlex.split(COMMON_FLAGS)
        parts += ["--tracks", self.edit_tracks.text().strip() or "train",
                  "--envs", str(self.spin_envs.value()), "--total", str(float(self.spin_total.value())),
                  "--lr", self.edit_lr.text().strip() or "5e-5", "--lr-end", self.edit_lr_end.text().strip() or "2e-5",
                  "--kl-coef", self.edit_kl.text().strip() or "0.05",
                  "--init", self._init_path(), "--seed", str(self.spin_seed.value()),
                  "--race-size", str(race), "--opponent", self.combo_opp.currentText(),
                  "--aux-grip", self.edit_aux_grip.text().strip() or "0", "--aux-opp", self.edit_aux_opp.text().strip() or "0",
                  "--device", self.combo_device.currentText(), "--wandb", self.combo_wandb.currentText(),
                  "--wandb-group", f"console-{r.key}", "--save-every", str(self.spin_save.value()),
                  "--name", name, "--controller", arm]
        if race > 1 and r.race_flags:
            parts += shlex.split(r.race_flags)
        if race == 1:
            parts += ["--cond", "none", "--critic-priv-adapter", "absent_opponent_17_to_21"]
        if arm == "estimated":
            parts += ["--estimator", self.edit_estimator.text().strip()]
        if not self.chk_restore_opt.isChecked():
            parts.append("--fresh-opt")
        extra = self.edit_extra.text().strip()
        if extra:
            parts += shlex.split(extra)
        return name, parts, self.combo_device.currentText()

    def _refresh_preview(self, *_a):
        try:
            _, parts, _ = self.argv()
            self.preview.setPlainText(" ".join(shlex.quote(a) for a in parts))
        except Exception as exc:
            self.preview.setPlainText(f"(인자 구성 실패: {exc})")

    def _launch(self):
        name, parts, device = self.argv()
        run_dir = os.path.join(catalog.RUNS_DIR, name)
        if os.path.isdir(run_dir) and any(f.endswith(".pt") for f in os.listdir(run_dir)):
            self.launch_note.setText(f"'{name}' 은 이미 체크포인트가 있는 런입니다. 이름을 바꾸세요.")
            return
        init = self._init_path()
        if not os.path.isfile(init):
            self.launch_note.setText(f"시작 체크포인트가 없습니다: {init}")
            return
        arm = self.combo_controller.currentText()
        if arm != "legacy" and self.spin_race.value() > 1:
            self.launch_note.setText(f"'{arm}' 제어기는 레이스당 차량 1에서만 학습됩니다.")
            return
        self.launch_requested.emit(name, parts, device)


# ================================================================ the page
class TrainingPage(QtWidgets.QWidget):
    view_checkpoint_requested = QtCore.pyqtSignal(str, str)     # run name, checkpoint path
    runs_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs = JobManager()
        self._runs: List[catalog.RunInfo] = []
        self._run_names: List[str] = []
        self._current_run: Optional[str] = None
        self._log_mtime: Tuple[Optional[str], float] = (None, 0.0)

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(SP[1], SP[1], SP[1], SP[1])
        root.setSpacing(SP[1])

        # -- left: the form
        left = QtWidgets.QScrollArea(); left.setObjectName("PanelScroll"); left.setWidgetResizable(True)
        left.setFrameShape(QtWidgets.QFrame.NoFrame); left.setMinimumWidth(360); left.setMaximumWidth(460)
        self.form = RecipeForm()
        self.form.launch_requested.connect(self._launch)
        left.setWidget(self.form)
        root.addWidget(left, 0)

        # -- centre: monitor
        centre = QtWidgets.QVBoxLayout(); centre.setSpacing(SP[1])
        head = Card("학습 현황")
        row = QtWidgets.QHBoxLayout(); row.setSpacing(SP[1])
        self.combo_run = QtWidgets.QComboBox(); self.combo_run.setMinimumWidth(260)
        self.combo_run.currentIndexChanged.connect(self._select_run)
        row.addWidget(label("런", "field")); row.addWidget(self.combo_run, 1)
        self.btn_refresh = QtWidgets.QPushButton("새로고침"); self.btn_refresh.setObjectName("GhostButton")
        self.btn_refresh.clicked.connect(self.refresh_all)
        row.addWidget(self.btn_refresh)
        head.add(row)
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 1000); self.progress.setValue(0)
        head.add(self.progress)
        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(SP[2]); grid.setVerticalSpacing(SP[0])
        self.m_upd = MetricTile("업데이트", "", small=True)
        self.m_steps = MetricTile("누적 스텝", "M", small=True)
        self.m_sps = MetricTile("처리량", "steps/s", small=True)
        self.m_eta = MetricTile("남은 시간", "", small=True)
        self.m_state = MetricTile("상태", "", small=True)
        self.m_gpu = MetricTile("GPU 메모리", "MiB", small=True, tooltip="nvidia-smi 기준, 이 PC 전체 사용량")
        for i, w in enumerate((self.m_upd, self.m_steps, self.m_sps, self.m_eta, self.m_state, self.m_gpu)):
            grid.addWidget(w, 0, i)
        head.add(grid)
        centre.addWidget(head)

        charts = QtWidgets.QGridLayout(); charts.setHorizontalSpacing(SP[1]); charts.setVerticalSpacing(SP[1])
        self.ch_rew = LineChart("보상 / 스텝", "", C["accent"])
        self.ch_coll = LineChart("충돌 / km", "/km", C["danger"], lower_is_better=True)
        self.ch_prog = LineChart("에피소드 진행", "m", C["ok"])
        self.ch_lap = LineChart("랩 타임", "s", C["warn"], lower_is_better=True)
        self.ch_kl = LineChart("KL (원본 대비)", "", C["text.1"])
        self.ch_sps = LineChart("처리량", "steps/s", C["text.1"])
        for i, ch in enumerate((self.ch_rew, self.ch_coll, self.ch_prog, self.ch_lap, self.ch_kl, self.ch_sps)):
            charts.addWidget(ch, i // 3, i % 3)
        centre.addLayout(charts, 1)
        self.log_box = QtWidgets.QPlainTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(150)
        self.log_box.setStyleSheet(f"font-family: '{theme.MONO_FONT}', monospace; font-size: 10px; background: {C['bg.window']};")
        fold = Collapsible("학습 로그 (끝부분)", expanded=True); fold.add(self.log_box)
        centre.addWidget(fold)
        root.addLayout(centre, 1)

        # -- right: jobs + checkpoints
        right = QtWidgets.QVBoxLayout(); right.setSpacing(SP[1])
        jobs = Card("실행 중인 학습")
        self.job_list = QtWidgets.QListWidget(); self.job_list.setMinimumWidth(300); self.job_list.setMaximumHeight(180)
        self.job_list.itemSelectionChanged.connect(self._job_selected)
        jobs.add(self.job_list)
        jr = QtWidgets.QHBoxLayout()
        self.btn_stop = QtWidgets.QPushButton("중지 (Ctrl+C 전송)"); self.btn_stop.setObjectName("DangerButton")
        self.btn_stop.clicked.connect(self._stop_job); jr.addWidget(self.btn_stop)
        self.btn_forget = QtWidgets.QPushButton("목록에서 제거"); self.btn_forget.setObjectName("GhostButton")
        self.btn_forget.clicked.connect(self._forget_job); jr.addWidget(self.btn_forget)
        jobs.add(jr)
        self.job_note = label("콘솔에서 시작한 학습은 콘솔을 닫아도 계속 돕니다. 셸 등에서 띄운 f1sim.learn.ppo 도 "
                              "'외부 실행'으로 잡혀 같이 보이고 중지할 수 있습니다.", "hint")
        jobs.add(self.job_note)
        right.addWidget(jobs)
        ck = Card("체크포인트")
        self.ckpt_list = QtWidgets.QListWidget(); self.ckpt_list.setMinimumWidth(300)
        ck.add(self.ckpt_list)
        self.btn_view = QtWidgets.QPushButton("주행 화면에서 보기")
        self.btn_view.setObjectName("PrimaryButton")
        self.btn_view.clicked.connect(self._view_checkpoint)
        ck.add(self.btn_view)
        ck.add(label("선택한 체크포인트로 주행 화면의 런을 바꿉니다. 시작을 누르면 그 가중치로 세션이 뜹니다.", "hint"))
        right.addWidget(ck, 1)
        root.addLayout(right, 0)

        self._timer = QtCore.QTimer(self); self._timer.setInterval(2000); self._timer.timeout.connect(self._tick)
        self._gpu_timer = QtCore.QTimer(self); self._gpu_timer.setInterval(5000); self._gpu_timer.timeout.connect(self._poll_gpu)
        self.refresh_all()

    # -- lifecycle
    def set_active(self, on: bool):
        if on:
            self.refresh_all(); self._timer.start(); self._gpu_timer.start(); self._poll_gpu()
        else:
            self._timer.stop(); self._gpu_timer.stop()

    def refresh_all(self):
        runs = catalog.list_runs()
        dirs = list_run_dirs()
        names = [d[0] for d in dirs]
        if names != self._run_names:
            self._runs, self._run_names = runs, names
            self.form.set_runs(runs)
            cur = self.combo_run.currentData()
            self.combo_run.blockSignals(True); self.combo_run.clear()
            for name, path, subtitle, _mt in dirs:
                self.combo_run.addItem(f"{name}  ·  {subtitle}", path)
            self.combo_run.blockSignals(False)
            i = self.combo_run.findData(cur) if cur else -1
            self.combo_run.setCurrentIndex(i if i >= 0 else 0)
            self.runs_changed.emit()
        self._refresh_jobs()
        self._select_run(self.combo_run.currentIndex())

    def _refresh_jobs(self):
        sel = self.job_list.currentItem().data(QtCore.Qt.UserRole) if self.job_list.currentItem() else None
        self.job_list.blockSignals(True); self.job_list.clear()
        for j in self.jobs.list_jobs():
            alive = j.alive
            age = catalog.format_age(time.time() - j.started)
            state = "실행 중" if alive else "종료됨"
            if alive and j.stop_requested:
                state = "중지 요청됨"
            if j.external:
                state += " · 외부 실행"
            it = QtWidgets.QListWidgetItem(f"{'●' if alive else '○'} {j.name}   {state} · {age} 전 · pid {j.pid}")
            it.setData(QtCore.Qt.UserRole, j.name)
            it.setForeground(QtGui.QColor(C["ok"] if alive else C["text.2"]))
            self.job_list.addItem(it)
            if j.name == sel:
                self.job_list.setCurrentItem(it)
        self.job_list.blockSignals(False)

    def _launch(self, name: str, argv: List[str], device: str):
        try:
            job = self.jobs.launch(name, argv, device)
        except Exception as exc:
            self.form.launch_note.setText(f"시작 실패: {exc}")
            return
        self.form.launch_note.setText(f"'{name}' 시작 (pid {job.pid}). 로그: {job.log}")
        self.refresh_all()
        i = self.combo_run.findData(job.run_dir)
        if i < 0:
            self.combo_run.addItem(f"{name}  ·  방금 시작", job.run_dir); i = self.combo_run.count() - 1
        self.combo_run.setCurrentIndex(i)

    def _job_selected(self):
        it = self.job_list.currentItem()
        if it is None:
            return
        name = it.data(QtCore.Qt.UserRole)
        i = self.combo_run.findData(os.path.join(catalog.RUNS_DIR, name))
        if i >= 0:
            self.combo_run.setCurrentIndex(i)

    def _stop_job(self):
        it = self.job_list.currentItem()
        if it is None:
            return
        name = it.data(QtCore.Qt.UserRole)
        job = next((j for j in self.jobs.list_jobs() if j.name == name), None)
        if job is None:
            return
        sent = self.jobs.stop(job)
        self.job_note.setText(f"{name}: {sent} 보냄. 마지막 저장 주기의 체크포인트까지 남습니다; 12초 뒤 다시 누르면 강제 종료.")
        self._refresh_jobs()

    def _forget_job(self):
        it = self.job_list.currentItem()
        if it is None:
            return
        job = next((j for j in self.jobs.list_jobs() if j.name == it.data(QtCore.Qt.UserRole)), None)
        if job is not None:
            self.jobs.forget(job)
        self._refresh_jobs()

    # -- monitor
    def _select_run(self, _idx):
        run_dir = self.combo_run.currentData()
        if not run_dir:
            self._current_run = None; return
        if run_dir != self._current_run:
            self._current_run = run_dir
            self._log_mtime = (None, 0.0)
        self._tick(force=True)

    def _tick(self, force: bool = False):
        run_dir = self._current_run
        if not run_dir:
            return
        log = run_log_path(run_dir)
        mtime = os.path.getmtime(log) if log else 0.0
        if not force and (log, mtime) == self._log_mtime:
            self._refresh_jobs(); return
        self._log_mtime = (log, mtime)
        text = ""
        if log:
            try:
                with open(log, "rb") as f:
                    f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4_000_000))
                    text = f.read().decode("utf-8", "replace")
            except OSError:
                text = ""
        p = parse_progress(text)
        self.progress.setValue(int(1000 * p.fraction))
        self.m_upd.set_value(f"{p.update}/{p.n_updates}" if p.n_updates else "—")
        self.m_steps.set_value(f"{p.steps_m:.2f}" if p.n_updates else "—")
        self.m_sps.set_value(f"{p.sps[-1]:.0f}" if p.sps else "—")
        job = next((j for j in self.jobs.list_jobs() if j.run_dir == run_dir), None)
        alive = bool(job and job.alive)
        if p.error and not p.finished:
            self.m_state.set_value("오류", C["danger"])
        elif p.finished:
            self.m_state.set_value("완료", C["ok"])
        elif alive and not p.n_updates:
            self.m_state.set_value("준비 중 (컴파일)", C["warn"])
        elif alive:
            self.m_state.set_value("실행 중", C["ok"])
        elif log and time.time() - mtime < 120:
            self.m_state.set_value("실행 중?", C["warn"])
        else:
            self.m_state.set_value("중단됨" if p.n_updates else "기록 없음", C["text.2"])
        if p.n_updates and p.update < p.n_updates and (alive or time.time() - mtime < 120) and p.sps:
            per_update = (p.steps_m * 1e6 / max(1, p.update)) if p.update else 0
            rate = sum(p.sps[-5:]) / len(p.sps[-5:])
            eta = (p.n_updates - p.update) * per_update / max(1.0, rate)
            self.m_eta.set_value(f"{eta / 60:.0f}분" if eta < 5400 else f"{eta / 3600:.1f}시간")
        else:
            self.m_eta.set_value("—")
        self.ch_rew.set_series(p.rew); self.ch_coll.set_series(p.coll); self.ch_prog.set_series(p.prog)
        self.ch_lap.set_series(p.lap); self.ch_kl.set_series(p.kl); self.ch_sps.set_series(p.sps)
        self.log_box.setPlainText("\n".join(p.lines))
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        self._refresh_ckpts(run_dir)
        self._refresh_jobs()

    def _refresh_ckpts(self, run_dir: str):
        sel = self.ckpt_list.currentItem().data(QtCore.Qt.UserRole) if self.ckpt_list.currentItem() else None
        self.ckpt_list.clear()
        for fn, path, mt in list_checkpoints(run_dir):
            it = QtWidgets.QListWidgetItem(f"{fn:16s}  {catalog.format_age(time.time() - mt)} 전")
            it.setData(QtCore.Qt.UserRole, path)
            it.setFont(QtGui.QFont(theme.MONO_FONT))
            self.ckpt_list.addItem(it)
            if path == sel:
                self.ckpt_list.setCurrentItem(it)
        if self.ckpt_list.currentItem() is None and self.ckpt_list.count():
            self.ckpt_list.setCurrentRow(self.ckpt_list.count() - 1)

    def _view_checkpoint(self):
        it = self.ckpt_list.currentItem()
        if it is None or not self._current_run:
            return
        self.view_checkpoint_requested.emit(os.path.basename(self._current_run), it.data(QtCore.Qt.UserRole))

    def _poll_gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
            used, total, util = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
            self.m_gpu.set_value(f"{used}/{total}", C["warn"] if int(used) > 0.85 * int(total) else None)
            self.m_gpu.setToolTip(f"GPU 사용률 {util}% (nvidia-smi)")
        except Exception:
            self.m_gpu.set_unknown()
