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
from ... import tracks
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

#: The two narrow recipes, written in the short grammar. Same five maps as before -- `track_names`
#: resolves either spelling, and `match_recipe` compares the resolved names, so a job started from
#: the old strings still matches its recipe in the jobs card.
SGR_TRACKS = "real/icra22,real/bb21-2,real/bb22-1,rt/spielberg,gen/control-1400"
R10_TRACKS = ("real/icra22,real/icra22@rev,real/bb21-2,real/bb21-2@rev,real/bb22-1,"
              "real/bb22-1@rev,rt/spielberg,rt/spielberg@rev,gen/control-1400,gen/control-1400@rev")


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


# ================================================================ what is running
#: `wandb.ai/<entity>/<project>/runs/<id>` as the trainer prints it. Taken from the log rather than
#: from a new trainer output format: every run already writes this line, including the ones started
#: from a shell months ago.
_WANDB = re.compile(r"https?://(?:\w+\.)?wandb\.ai/[\w.\-]+/[\w.\-]+/runs/[\w\-]+")
_SHA_IN_NAME = re.compile(r"([0-9a-f]{8,40})")
_CKPT_SHA_CACHE: Dict[Tuple[str, float, int], str] = {}


def argv_flags(argv: Sequence[str]) -> Dict[str, str]:
    """`--flag value` pairs, plus `--flag` -> "" for switches. Repeats keep the last, which is what
    the trainer's own argparse does."""
    out: Dict[str, str] = {}
    i = 0
    argv = list(argv)
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[a[2:]] = argv[i + 1]
                i += 2
                continue
            out[a[2:]] = ""
        i += 1
    return out


def checkpoint_sha(path: str, limit_mb: int = 400) -> str:
    """Short content sha of a checkpoint. Cached on (path, mtime, size), so the jobs card can ask
    for it every two seconds and read the file once.

    Content, not the name: two runs started from `ppo_final.pt` of two different runs have the same
    basename, and "which weights did this start from" is the question the card exists to answer.
    A sha already spelled into the file name (`frozen_original_48cc698f.pt`) is believed as is --
    that is where it came from.
    """
    base = os.path.basename(path)
    m = _SHA_IN_NAME.search(os.path.splitext(base)[0])
    if m:
        return m.group(1)[:8]
    try:
        st = os.stat(path)
    except OSError:
        return ""
    key = (path, st.st_mtime, st.st_size)
    if key in _CKPT_SHA_CACHE:
        return _CKPT_SHA_CACHE[key]
    if st.st_size > limit_mb * 1_000_000:
        return ""
    import hashlib
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    _CKPT_SHA_CACHE[key] = h.hexdigest()[:8]
    return _CKPT_SHA_CACHE[key]


def _norm_tracks(spec: str):
    """A `--tracks` value as a comparable thing, independent of which grammar it was written in.

    `real:blackbox2022_1~rev` and `real/bb22-1@rev` are the same track; a recipe rewritten in the
    new grammar must still recognise the jobs that were launched from its old spelling.
    """
    spec = (spec or "").strip()
    if not spec or "," not in spec and "/" not in spec and ":" not in spec:
        return spec                      # a split name ("train"), or nothing
    out = []
    for n in spec.split(","):
        n = n.strip()
        if not n:
            continue
        try:
            out.append(tracks.resolve(n))
        except Exception:
            out.append(n)
    return tuple(out)


def match_recipe(flags: Dict[str, str]) -> Optional[Recipe]:
    """Which preset an argv came from, by the fields the preset actually decides.

    Matched on the run, not recorded at launch: a job found in the process table has no record, and
    a recorded one may have been edited in the form before it was started. The fields compared are
    the ones a recipe sets and a user would have to change deliberately.
    """
    for r in RECIPES:
        if r.key == "custom":
            continue
        if _norm_tracks(flags.get("tracks", "")) != _norm_tracks(r.tracks):
            continue
        if int(flags.get("race-size", 1) or 1) != r.race_size:
            continue
        if flags.get("opponent", "teacher") != r.opponent:
            continue
        return r
    return None


def _fmt_steps(total: str) -> str:
    try:
        v = float(total)
    except (TypeError, ValueError):
        return total or "—"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M".replace(".00M", "M")
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def describe_tracks(spec: str, draws: str = "") -> str:
    """`--tracks` as a person reads it: the split by name, or how many maps and which ones.

    This is the line the user's third complaint is about -- "I cannot tell what is training" -- so
    it names maps rather than counting strings: `학습 분할 (149개 변형, 40개 맵)`, or
    `3개 맵 · Blackbox 2022 #1, Spielberg, 생성 control 1400`.
    """
    spec = (spec or "train").strip()
    if spec in ("train", "eval", "heldout", "eval_obstacles", "heldout_obstacles", "eval_all",
                "heldout_all"):
        try:
            s = tracks.split_summary(spec)
        except Exception:
            return spec
        group = s.get("group") or spec
        return f"{group} 분할 ({s['n_variants']}개 변형, {s['n_tracks']}개 맵)"
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        return "—"
    # Distinct *maps*, and the first three of those. A set written as 149 scenarios over 53 maps is
    # the same set said two ways, and naming "ICRA 2022, ICRA 2022 · 역방향, ICRA 2022 · 거울" as its
    # first three says nothing at all about what is being trained on.
    bases, shown, dirs, obs = [], [], [], []
    for n in names:
        try:
            sc = tracks.parse(n)
            key = sc.track or sc.raw or n
            display = tracks.get(sc.track).display if sc.track else (sc.raw or n)
            if sc.direction not in dirs:
                dirs.append(sc.direction)
            if sc.obstacle not in obs:
                obs.append(sc.obstacle)
        except Exception:
            key, display = n, n
        if key not in bases:
            bases.append(key)
            if len(shown) < 3:
                shown.append(display)
    more = f" 외 {len(bases) - 3}개" if len(bases) > 3 else ""
    head = (f"{len(names)}개 시나리오 · {len(bases)}개 맵 · " if len(names) != len(bases)
            else f"{len(bases)}개 맵 · ")
    # The maps are one half of what is being trained on; what is done to them is the other, and it
    # is the same few words however long the list is.
    policy = ""
    if [d for d in dirs if d]:
        policy += " · 방향 " + "/".join(tracks.DIRECTION_LABEL[d] for d in tracks.DIRECTIONS if d in dirs)
    if [o for o in obs if o]:
        policy += " · 장애물 " + "/".join(tracks.OBSTACLE_LABEL[o] for o in tracks.OBSTACLES if o in obs)
    # An open seed is not one track: it is `--obstacle-draws` rasterised placements of it, and the
    # difference is the difference between eight maps and one.
    n_draws = 0
    try:
        n_draws = int(draws)
    except (TypeError, ValueError):
        n_draws = 0
    drawn = sum(1 for n in names if "#" in n and n.endswith(":*"))
    tail = f" · 배치 {n_draws}개씩 ({len(names) - drawn + drawn * n_draws}개 로드)" \
        if drawn and n_draws > 1 else ""
    return head + ", ".join(shown) + more + policy + tail


@dataclass
class JobSummary:
    """Everything the jobs card shows about one training process.

    Built from the recorded argv (console jobs) or `/proc/<pid>/cmdline` (external ones) plus the
    log. No new trainer output format: the point is that a run started from a shell three weeks ago
    is just as readable as one started here five minutes ago.
    """
    name: str
    state: str
    alive: bool
    external: bool
    started: float
    elapsed_s: float
    eta_s: Optional[float] = None
    progress: str = ""
    recipe: str = ""
    init: str = ""
    tracks_text: str = ""
    race_text: str = ""
    events_text: str = ""
    controller: str = ""
    lr_text: str = ""
    total_text: str = ""
    wandb_url: str = ""
    log: str = ""
    pid: int = 0

    def lines(self) -> List[Tuple[str, str]]:
        """(label, value) rows, in the order they answer "what is this?"."""
        rows = [("레시피", self.recipe), ("트랙", self.tracks_text), ("레이스", self.race_text)]
        if self.events_text:
            rows.append(("이벤트", self.events_text))
        rows += [("제어기", self.controller), ("시작 체크포인트", self.init),
                 ("학습률", self.lr_text), ("총 스텝", self.total_text)]
        if self.wandb_url:
            rows.append(("W&B", self.wandb_url))
        rows.append(("로그", self.log))
        return [(k, v) for k, v in rows if v]


def summarize_job(job: "Job", log_text: str = "", progress: Optional[Progress] = None,
                  now: Optional[float] = None) -> JobSummary:
    """One training process, described from what it was started with."""
    now = time.time() if now is None else now
    f = argv_flags(job.argv)
    alive = job.alive
    p = progress
    state = "실행 중" if alive else "종료됨"
    if p is not None and p.finished:
        state = "완료"
    elif p is not None and p.error and not p.finished:
        state = "오류"
    elif alive and job.stop_requested:
        state = "중지 요청됨"
    elif alive and p is not None and not p.n_updates:
        state = "준비 중 (컴파일)"
    eta = None
    if p is not None and p.n_updates and p.update < p.n_updates and p.sps and alive:
        # Steps per update from the argv when it is there (`--envs` x `--horizon` is exactly what
        # the trainer collects), because the progress line prints cumulative steps to one decimal
        # in millions -- at 0.0M that division is zero and the ETA reads "0분" forever.
        try:
            per_update = float(f["envs"]) * float(f["horizon"])
        except (KeyError, TypeError, ValueError):
            per_update = (p.steps_m * 1e6 / max(1, p.update)) if p.update else 0.0
        rate = sum(p.sps[-5:]) / len(p.sps[-5:])
        eta = (p.n_updates - p.update) * per_update / max(1.0, rate)
    recipe = match_recipe(f)
    init = f.get("init", "")
    sha = checkpoint_sha(init) if init else ""
    race = int(f.get("race-size", 1) or 1)
    race_text = f"{race}대" + (f" · 상대차 {f.get('opponent', 'teacher')}" if race > 1 else " (단독)")
    events = f.get("opp-events", "")
    events_text = f"{events} (10초당 {f.get('opp-event-rate', '?')}회)" if events else ""
    wb = _WANDB.search(log_text or "")
    lr, lr_end = f.get("lr", ""), f.get("lr-end", "")
    return JobSummary(
        name=job.name, state=state, alive=alive, external=job.external, started=job.started,
        elapsed_s=max(0.0, now - job.started), eta_s=eta,
        progress=(f"{p.update}/{p.n_updates} 업데이트 · {p.steps_m:.2f}M 스텝"
                  if p is not None and p.n_updates else ""),
        recipe=(recipe.title if recipe else ("외부 실행" if job.external else "사용자 정의")),
        init=(f"{os.path.basename(init)}" + (f" · {sha}" if sha else "")) if init else "",
        tracks_text=describe_tracks(f.get("tracks", "train"), f.get("obstacle-draws", "")),
        race_text=race_text, events_text=events_text,
        controller=f.get("controller", "legacy"),
        lr_text=(f"{lr} → {lr_end}" if lr and lr_end else lr or ""),
        total_text=_fmt_steps(f.get("total", "")),
        wandb_url=wb.group(0) if wb else "",
        log=job.log, pid=job.pid)


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


# ================================================================ the track picker
class TrackPicker(QtWidgets.QWidget):
    """Which maps a run trains on, chosen as maps rather than as strings.

    The field this replaces was a `QLineEdit` holding `train`, and the hint under it said
    "'train' = 전체 학습 카탈로그(149)". To train on anything else you had to know the loader's
    grammar and type a hundred and forty-nine names, so in practice nobody chose anything.

    Here the three groups are tabs, each map is a checkbox, and the direction / obstacle policy is
    applied to whatever is ticked. "학습 셋 전체" is one button, and while that is what is selected
    the argv says `--tracks train` -- the curated list itself, not a reconstruction of it, so a run
    launched from the default is byte for byte the run the recipes were measured with.
    """
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(SP[0])

        presets = QtWidgets.QHBoxLayout()
        presets.setSpacing(SP[0])
        self.btn_all_train = QtWidgets.QPushButton("학습 셋 전체")
        self.btn_all_train.setObjectName("GhostButton")
        self.btn_all_train.setToolTip("프로젝트의 학습 분할 그대로 (--tracks train).")
        self.btn_all_train.clicked.connect(lambda: self.select_split("train"))
        presets.addWidget(self.btn_all_train)
        self.btn_all_heldout = QtWidgets.QPushButton("검증 셋 전체")
        self.btn_all_heldout.setObjectName("GhostButton")
        self.btn_all_heldout.setToolTip("검증 분할 그대로. 학습에 쓰면 그 뒤의 일반화 점수는 의미가 없습니다.")
        self.btn_all_heldout.clicked.connect(lambda: self.select_split("heldout"))
        presets.addWidget(self.btn_all_heldout)
        self.btn_none = QtWidgets.QPushButton("모두 해제")
        self.btn_none.setObjectName("GhostButton")
        self.btn_none.clicked.connect(self.clear_selection)
        presets.addWidget(self.btn_none)
        presets.addStretch(1)
        v.addLayout(presets)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self._lists: Dict[str, QtWidgets.QListWidget] = {}
        for group in tracks.GROUP_ORDER:
            lw = QtWidgets.QListWidget()
            lw.setMinimumHeight(150)
            lw.setToolTip(tracks.GROUP_HINT[group] + "\n\n" + tracks.GROUP_CAVEAT)
            lw.itemChanged.connect(self._on_item_changed)
            self._lists[group] = lw
            self.tabs.addTab(lw, group)
        v.addWidget(self.tabs)
        self.group_hint = label(tracks.GROUP_HINT[tracks.GROUP_ORDER[0]], "hint")
        self.tabs.currentChanged.connect(self._on_tab)
        v.addWidget(self.group_hint)

        # -- the policy applied to whatever is ticked
        self.dir_boxes: Dict[str, QtWidgets.QCheckBox] = {}
        drow = FlowRow()
        for d in tracks.DIRECTIONS:
            cb = QtWidgets.QCheckBox(tracks.DIRECTION_LABEL[d])
            cb.setChecked(True)
            cb.toggled.connect(self._on_policy_changed)
            self.dir_boxes[d] = cb
            drow.add(cb)
        v.addWidget(FieldRow("방향", drow, "선택한 맵마다 이 방향들을 모두 학습에 넣습니다."))

        self.obs_boxes: Dict[str, QtWidgets.QCheckBox] = {}
        orow = FlowRow()
        for o in tracks.OBSTACLES:
            cb = QtWidgets.QCheckBox(tracks.OBSTACLE_LABEL[o])
            cb.setChecked(o == "")
            cb.setToolTip(tracks.OBSTACLE_HINT[o])
            cb.toggled.connect(self._on_policy_changed)
            self.obs_boxes[o] = cb
            orow.add(cb)
        v.addWidget(FieldRow("장애물", orow, "여러 개를 고르면 맵마다 각각 만들어 넣습니다."))

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.setSpacing(SP[0])
        self.combo_seed = QtWidgets.QComboBox()
        self.combo_seed.addItem("무작위", "random")
        self.combo_seed.addItem("고정", "fixed")
        self.combo_seed.currentIndexChanged.connect(self._on_policy_changed)
        seed_row.addWidget(self.combo_seed, 1)
        self.spin_draws = QtWidgets.QSpinBox()
        self.spin_draws.setRange(1, 64)
        self.spin_draws.setValue(8)
        self.spin_draws.setPrefix("배치 ")
        self.spin_draws.setSuffix("개")
        self.spin_draws.valueChanged.connect(self._on_policy_changed)
        seed_row.addWidget(self.spin_draws, 1)
        self.spin_fixed = QtWidgets.QSpinBox()
        self.spin_fixed.setRange(0, 999999)
        self.spin_fixed.setValue(44)
        self.spin_fixed.setEnabled(False)
        self.spin_fixed.valueChanged.connect(self._on_policy_changed)
        seed_row.addWidget(self.spin_fixed, 1)
        box = QtWidgets.QWidget()
        box.setLayout(seed_row)
        self.row_seed = FieldRow("장애물 시드", box,
                                 "무작위: 맵마다 배치를 그만큼 뽑습니다 (--obstacle-draws). "
                                 "같은 --seed 면 같은 배치가 나옵니다.")
        v.addWidget(self.row_seed)

        self.summary = label("", "hint")
        self.summary.setObjectName("Mono")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)

        self._preset: Optional[str] = None
        #: A `--tracks` value the three controls cannot express, kept verbatim rather than
        #: approximated. Cleared by any edit in the picker.
        self._raw: str = ""
        self._quiet = False
        self.set_catalog()
        self.select_split("train")

    # -- data
    def set_catalog(self, scene_ids: Optional[Sequence[str]] = None):
        """Fill the three tabs. Torch-free: `f1sim.tracks` is the registry itself, so the picker
        is populated on the first paint rather than waiting for a worker."""
        if scene_ids is None:
            scene_ids = [f"scene/{s['name']}" for s in catalog.list_scenes()]
        groups = tracks.groups(scene_ids=list(scene_ids))
        self._quiet = True
        for group, lw in self._lists.items():
            checked = {lw.item(i).data(QtCore.Qt.UserRole) for i in range(lw.count())
                       if lw.item(i).checkState() == QtCore.Qt.Checked}
            lw.clear()
            for tid in groups.get(group, []):
                e = tracks.get(tid)
                it = QtWidgets.QListWidgetItem(f"{e.display}    {tid}")
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Checked if tid in checked else QtCore.Qt.Unchecked)
                it.setData(QtCore.Qt.UserRole, tid)
                it.setToolTip(e.note or e.display)
                lw.addItem(it)
            self.tabs.setTabText(list(self._lists).index(group),
                                 f"{group} ({len(groups.get(group, []))})")
        self._quiet = False
        self._refresh_summary()

    def selected_tracks(self) -> List[str]:
        out = []
        for lw in self._lists.values():
            for i in range(lw.count()):
                it = lw.item(i)
                if it.checkState() == QtCore.Qt.Checked:
                    out.append(it.data(QtCore.Qt.UserRole))
        return out

    def directions(self) -> List[str]:
        return [d for d in tracks.DIRECTIONS if self.dir_boxes[d].isChecked()] or [""]

    def obstacles(self) -> List[str]:
        return [o for o in tracks.OBSTACLES if self.obs_boxes[o].isChecked()] or [""]

    def draws(self) -> int:
        return int(self.spin_draws.value())

    # -- actions
    def select_split(self, split: str):
        want = set(tracks.split_tracks(split))
        self._raw = ""
        self._quiet = True
        for lw in self._lists.values():
            for i in range(lw.count()):
                it = lw.item(i)
                it.setCheckState(QtCore.Qt.Checked
                                 if it.data(QtCore.Qt.UserRole) in want else QtCore.Qt.Unchecked)
        for d, cb in self.dir_boxes.items():
            cb.setChecked(True)
        for o, cb in self.obs_boxes.items():
            cb.setChecked(o == "")
        self.combo_seed.setCurrentIndex(0)
        self._quiet = False
        self._preset = split
        self._refresh_summary()
        self.changed.emit()

    def set_spec(self, spec: str):
        """Show an existing `--tracks` value: a split name, or a comma list to reverse-engineer.

        A list is reduced back to (maps, directions, obstacles) and the controls are set to it. When
        that reduction does not reproduce the list -- a set with per-map obstacle seeds, which is
        exactly what the curated splits are -- the value is kept verbatim and the picker says so
        rather than quietly training on a different set.
        """
        spec = (spec or "").strip()
        self._raw = ""
        if spec in ("train", "eval", "heldout"):
            self.select_split("train" if spec == "train" else "heldout")
            return
        names = [n.strip() for n in spec.split(",") if n.strip()]
        scenarios = []
        for n in names:
            try:
                sc = tracks.parse(n)
            except Exception:
                sc = None
            if sc is None or sc.raw:
                self._raw = spec
                self._preset = None
                self._refresh_summary()
                self.changed.emit()
                return
            scenarios.append(sc)
        want_tracks = list(dict.fromkeys(sc.track for sc in scenarios))
        want_dirs = {sc.direction for sc in scenarios}
        want_obs = {sc.obstacle for sc in scenarios}
        self._quiet = True
        for lw in self._lists.values():
            for i in range(lw.count()):
                it = lw.item(i)
                it.setCheckState(QtCore.Qt.Checked
                                 if it.data(QtCore.Qt.UserRole) in want_tracks else QtCore.Qt.Unchecked)
        for d, cb in self.dir_boxes.items():
            cb.setChecked(d in want_dirs)
        for o, cb in self.obs_boxes.items():
            cb.setChecked(o in want_obs)
        seeds = {sc.seed for sc in scenarios if sc.obstacle}
        fixed = seeds and None not in seeds
        self.combo_seed.setCurrentIndex(1 if fixed else 0)
        if fixed and len(seeds) == 1:
            self.spin_fixed.setValue(int(next(iter(seeds))))
        self._quiet = False
        self._preset = None
        # Tracks the picker cannot show (a scene that has been deleted, a generator seed outside
        # the starter set) would silently vanish from the selection; keep the text instead.
        missing = [t for t in want_tracks if t not in set(self.selected_tracks())]
        # Compared in the short grammar, not the loader's: an entry whose seed is still `*` has no
        # loader name yet, and asking for one is how this check used to raise.
        rebuilt = [x for x in self.spec().split(",") if x] if self.spec() else []
        if missing or sorted(rebuilt) != sorted(sc.short() for sc in scenarios):
            self._raw = spec
        self._refresh_summary()
        self.changed.emit()

    def clear_selection(self):
        self._quiet = True
        for lw in self._lists.values():
            for i in range(lw.count()):
                lw.item(i).setCheckState(QtCore.Qt.Unchecked)
        self._quiet = False
        self._preset = None
        self._raw = ""
        self._refresh_summary()
        self.changed.emit()

    def _on_tab(self, i: int):
        group = tracks.GROUP_ORDER[i] if 0 <= i < len(tracks.GROUP_ORDER) else ""
        self.group_hint.setText(tracks.GROUP_HINT.get(group, ""))

    def _on_item_changed(self, _item):
        if self._quiet:
            return
        self._preset = None
        self._raw = ""
        self._refresh_summary()
        self.changed.emit()

    def _on_policy_changed(self, *_a):
        self.spin_draws.setEnabled(str(self.combo_seed.currentData()) == "random")
        self.spin_fixed.setEnabled(str(self.combo_seed.currentData()) == "fixed")
        if self._quiet:
            return
        self._preset = None
        self._raw = ""
        self._refresh_summary()
        self.changed.emit()

    # -- output
    def spec(self) -> str:
        """What goes after `--tracks`.

        `train` when the split itself is selected: the curated list is not reproducible from a
        per-track policy (its obstacle seeds differ per map and per family, by design), and a run
        launched from the default has to be the run the recipes were measured with.
        """
        if self._preset == "train":
            return "train"
        if self._preset == "heldout":
            return "heldout"
        if self._raw:
            return self._raw
        ids = self.selected_tracks()
        if not ids:
            return ""
        fixed = str(self.combo_seed.currentData()) == "fixed"
        out = []
        for tid in ids:
            allowed = tracks.get(tid).obstacle_options()
            for o in self.obstacles():
                if o not in allowed:
                    continue
                for d in self.directions():
                    s = tid + (f"@{d}" if d else "")
                    if o:
                        s += f"#{o}:{self.spin_fixed.value() if fixed else '*'}"
                    out.append(s)
        return ",".join(dict.fromkeys(out))

    def count(self) -> int:
        """How many tracks the trainer will actually load, random draws included."""
        spec = self.spec()
        if not spec:
            return 0
        if spec in ("train", "heldout"):
            return len(tracks.split_names(spec))
        return len(tracks.expand_all(spec.split(","), self.draws(), 0))

    def _refresh_summary(self):
        spec = self.spec()
        if not spec:
            self.summary.setText("맵을 하나 이상 고르세요.")
            return
        if spec in ("train", "heldout"):
            s = tracks.split_summary(spec)
            self.summary.setText(f"--tracks {spec}   →   {s['n_variants']}개 변형 / "
                                 f"{s['n_tracks']}개 맵 ({s['group']} 분할 그대로)")
            return
        if self._raw:
            self.summary.setText(f"직접 지정한 목록 그대로 씁니다 ({len(spec.split(','))}개). "
                                 f"아래 체크박스를 건드리면 이 값을 버립니다.")
            return
        n_specs = len(spec.split(","))
        self.summary.setText(f"{len(self.selected_tracks())}개 맵 → {n_specs}개 시나리오"
                             f" → 로드 {self.count()}개")


class FlowRow(QtWidgets.QWidget):
    """A row of small controls that wraps. `FlowLayout` without importing the viewport's copy."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._h = QtWidgets.QHBoxLayout(self)
        self._h.setContentsMargins(0, 0, 0, 0)
        self._h.setSpacing(SP[1])

    def add(self, w):
        self._h.addWidget(w)
        return w


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

        # The maps get their own card, above the fold. They are the first thing a run is about and
        # they were a one-line text box saying `train`.
        track_card = Card("학습할 맵")
        self.tracks = TrackPicker()
        self.tracks.changed.connect(self._refresh_preview)
        track_card.add(self.tracks)
        v.addWidget(track_card)

        adv = Collapsible("레이스·제어기", expanded=False)
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

        for w in (self.edit_name, self.edit_lr, self.edit_lr_end, self.edit_kl, self.edit_aux_grip,
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
        self.tracks.set_spec(r.tracks)
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
        parts += ["--tracks", self.tracks.spec() or "train",
                  "--obstacle-draws", str(self.tracks.draws()),
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


class JobCard(QtWidgets.QFrame):
    """One running (or finished) training process, described.

    The user's complaint was "지금 학습 돌리고있는것도 뭐 돌리고있는지도 모르겠고" -- the jobs list
    was `● name  실행 중 · 12분 전 · pid 4131` and nothing else, so the only way to find out what a
    run was doing was to remember what you typed, or to read `ps`. Everything here comes from the
    argv the process was started with (recorded for console jobs, `/proc/<pid>/cmdline` for the
    rest) and from the log it already writes; the trainer prints nothing new.
    """
    stop_requested = QtCore.pyqtSignal(str)
    forget_requested = QtCore.pyqtSignal(str)
    select_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.name = ""
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(SP[2], SP[1], SP[2], SP[1])
        v.setSpacing(SP[0])
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(SP[1])
        self.dot = label("●", "body")
        head.addWidget(self.dot)
        self.title = label("", "section")
        self.title.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        head.addWidget(self.title)
        self.state = label("", "hint")
        head.addWidget(self.state)
        head.addStretch(1)
        self.btn_open = QtWidgets.QPushButton("이 런 보기")
        self.btn_open.setObjectName("GhostButton")
        self.btn_open.clicked.connect(lambda: self.select_requested.emit(self.name))
        head.addWidget(self.btn_open)
        self.btn_stop = QtWidgets.QPushButton("중지")
        self.btn_stop.setObjectName("DangerButton")
        self.btn_stop.setToolTip("SIGINT (Ctrl+C) 를 보냅니다. 마지막 저장 주기의 체크포인트까지 남습니다. "
                                 "12초 뒤 다시 누르면 강제 종료합니다.")
        self.btn_stop.clicked.connect(lambda: self.stop_requested.emit(self.name))
        head.addWidget(self.btn_stop)
        self.btn_forget = QtWidgets.QPushButton("목록에서 제거")
        self.btn_forget.setObjectName("GhostButton")
        self.btn_forget.clicked.connect(lambda: self.forget_requested.emit(self.name))
        head.addWidget(self.btn_forget)
        v.addLayout(head)
        self.clock = label("", "hint")
        v.addWidget(self.clock)
        self.facts = KeyValueList()
        v.addWidget(self.facts)

    def set_summary(self, s: JobSummary):
        self.name = s.name
        colour = C["ok"] if s.alive else C["text.2"]
        if s.state == "오류":
            colour = C["danger"]
        elif s.state in ("중지 요청됨", "준비 중 (컴파일)"):
            colour = C["warn"]
        self.dot.setText("●" if s.alive else "○")
        self.dot.setStyleSheet(f"color: {colour};")
        self.title.setText(s.name)
        self.state.setText(s.state + (" · 외부 실행" if s.external else "") + f" · pid {s.pid}")
        self.state.setStyleSheet(f"color: {colour};")
        clock = (f"{time.strftime('%m-%d %H:%M', time.localtime(s.started))} 시작 · "
                 f"{catalog.format_age(s.elapsed_s)} 경과")
        if s.eta_s is not None:
            clock += (f" · 남은 시간 {s.eta_s / 60:.0f}분" if s.eta_s < 5400
                      else f" · 남은 시간 {s.eta_s / 3600:.1f}시간")
        if s.progress:
            clock += f" · {s.progress}"
        self.clock.setText(clock)
        for k, val in s.lines():
            self.facts.set(k, val)
        self.btn_stop.setEnabled(s.alive)
        self.btn_forget.setEnabled(not s.alive and not s.external)


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
        self._job_log_cache: Dict[str, Tuple[Tuple[str, float], str]] = {}

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

        # -- centre: what is running, then the monitor for whichever run is selected
        centre = QtWidgets.QVBoxLayout(); centre.setSpacing(SP[1])
        jobs = Card("지금 돌고 있는 학습")
        self.jobs_box = QtWidgets.QVBoxLayout()
        self.jobs_box.setSpacing(SP[1])
        self.jobs_box.setContentsMargins(0, 0, 0, 0)
        jobs.add(self.jobs_box)
        self.job_note = label("콘솔에서 시작한 학습은 콘솔을 닫아도 계속 돕니다. 셸 등에서 띄운 "
                              "f1sim.learn.ppo 도 '외부 실행'으로 잡혀 같이 보이고 중지할 수 있습니다.", "hint")
        jobs.add(self.job_note)
        centre.addWidget(jobs)
        self._job_cards: Dict[str, JobCard] = {}

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

        # -- right: checkpoints
        right = QtWidgets.QVBoxLayout(); right.setSpacing(SP[1])
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
        """One summary card per training process, newest first.

        The card is rebuilt in place rather than recreated: a card recreated every two seconds
        cannot be clicked, and its W&B link cannot be selected.
        """
        jobs = self.jobs.list_jobs()
        seen = []
        for i, j in enumerate(jobs):
            card = self._job_cards.get(j.name)
            if card is None:
                card = JobCard()
                card.stop_requested.connect(self._stop_job)
                card.forget_requested.connect(self._forget_job)
                card.select_requested.connect(self._open_job)
                self._job_cards[j.name] = card
            self.jobs_box.insertWidget(i, card)
            card.setVisible(True)
            card.set_summary(self.job_summary(j))
            seen.append(j.name)
        for name, card in list(self._job_cards.items()):
            if name not in seen:
                self.jobs_box.removeWidget(card)
                card.setParent(None)
                card.deleteLater()
                self._job_cards.pop(name, None)
        self.job_note.setVisible(True)
        if not jobs:
            self.job_note.setText("지금 돌고 있는 학습이 없습니다. 왼쪽에서 레시피를 고르고 '학습 시작'을 누르세요.")
        else:
            self.job_note.setText("콘솔에서 시작한 학습은 콘솔을 닫아도 계속 돕니다. 셸 등에서 띄운 "
                                  "f1sim.learn.ppo 도 '외부 실행'으로 잡혀 같이 보이고 중지할 수 있습니다.")

    def job_summary(self, job: Job) -> JobSummary:
        """The card's content. Reads the tail of the run's log for the progress line and the W&B
        url; cached by (path, mtime) so a two-second refresh is not a two-second file read."""
        log = job.log or run_log_path(job.run_dir) or ""
        text = ""
        if log:
            try:
                mt = os.path.getmtime(log)
            except OSError:
                mt = 0.0
            cached = self._job_log_cache.get(job.name)
            if cached and cached[0] == (log, mt):
                text = cached[1]
            else:
                try:
                    with open(log, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 400_000))
                        text = f.read().decode("utf-8", "replace")
                except OSError:
                    text = ""
                self._job_log_cache[job.name] = ((log, mt), text)
        return summarize_job(job, log_text=text, progress=parse_progress(text) if text else None)

    def _open_job(self, name: str):
        i = self.combo_run.findData(os.path.join(catalog.RUNS_DIR, name))
        if i >= 0:
            self.combo_run.setCurrentIndex(i)

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

    def _stop_job(self, name: str):
        job = next((j for j in self.jobs.list_jobs() if j.name == name), None)
        if job is None:
            return
        sent = self.jobs.stop(job)
        self.job_note.setText(f"{name}: {sent} 보냄. 마지막 저장 주기의 체크포인트까지 남습니다; "
                              f"12초 뒤 다시 누르면 강제 종료.")
        self._refresh_jobs()

    def _forget_job(self, name: str):
        job = next((j for j in self.jobs.list_jobs() if j.name == name), None)
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
