"""The simulation worker: one process, all of the torch, none of the UI.

Why a separate process at all
-----------------------------
The old interactive path ran the simulation and the renderer in one process and, on its default
settings (`compile=True` -> `viewer_threaded()` False), in one *loop*: `NativeViewer.run` stepped
the simulator and then drew, so a slow step was a late frame by construction. Around that, loading
a checkpoint, capturing CUDA graphs (15-20 s) and every `.item()` device sync all happened where
the window lived.

Splitting on a process boundary is not a claim that a thread could never work. It is a claim that
this way "the UI keeps answering" is structural rather than something to be re-measured after every
change: nothing here can hold the Qt event loop, and if this process dies or wedges, the console
survives to say so and offer a retry.

Threads inside this process
---------------------------
Three, and which one owns what is the whole design:

* **control** (the thread that runs `serve`) reads the control pipe. It answers the cheap questions
  itself and builds sessions. It never touches a session that a simulation thread is stepping.
* **sim** owns the live session exclusively. Commands that touch it -- pause, reset, focus, overlay
  -- are queued by the control thread and applied *here*, between steps, and the ack is sent from
  here once the change has actually happened. That is what makes the ack true: a `pause` ack means
  the simulation has stopped, not that the message arrived. Applying `env.reset()` from the control
  thread while this one is inside `env.step()` is a data race against CUDA graph state, and it was
  in the first draft of this file.
* **sender** drains a bounded drop-oldest slot onto the frame pipe. If the console stops reading,
  the pipe fills and *this* thread blocks while the simulation carries on and the slot discards the
  frames nobody was going to see.

Everything simulation-facing is reused as-is from `f1sim.learn.watch` -- checkpoint loading, the
speed-cap rule, the env construction, the compiled actor. Physics, policy, reward and env code are
not touched.

Honesty rules this file is responsible for
------------------------------------------
* Frame payloads are complete snapshots: nothing in one aliases a live tensor. On a CPU device
  `t.float().cpu().numpy()` returns a *view of the tensor's own storage*, so the next step rewrites
  a frame that has already been queued. Every array that leaves here is copied.
* `sim_rate`, `t`, `seq`, the checkpoint identity and its saved metrics are measured or read, never
  estimated. There is no fps number in here at all: the worker cannot see the display.
* A checkpoint whose observation layout does not match the environment this worker built is
  refused, not driven. `load_checkpoint` leaves shape-mismatched tensors at their fresh random
  initialisation and only records them in `extra["skipped"]`, so "it ran" is not evidence that the
  policy on screen is the policy in the file.
"""
from __future__ import annotations

import faulthandler
import math
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .console import protocol as P
from .console.protocol import LatestSlot
from .console.catalog import SCENES_GROUP

#: Frames the worker will let pile up before dropping the oldest. Two is enough to cover a hiccup
#: in the console without storing latency: what the viewer wants is the newest state, not a queue.
OUTBOUND_CAPACITY = 2

#: Cap on frames per second sent to the console. The simulation runs at 40 Hz control rate; there
#: is nothing to gain from sending faster than the display can show, and the pipe is shared with
#: control messages. Checked *before* the snapshot is built -- a snapshot costs a device->host
#: transfer, and paying for one to throw it away is the expensive kind of throttling.
MAX_FRAME_HZ = 60.0

#: Wall-clock window the reported `sim_rate` averages over. Long enough that a single slow step
#: does not dominate it, short enough to follow a session that changes pace.
RATE_WINDOW_S = 0.5

#: Track objects are expensive to build (~0.5 s each) and cheap to keep. Switching back to a map
#: you were just on should not go to disk again.
TRACK_CACHE_SIZE = 8

#: How long a swap waits for the outgoing simulation thread. It only has to finish the step it is
#: in; anything longer means torch is wedged, and then the right answer is to leak the session
#: rather than free an environment a live thread is still stepping.
SIM_JOIN_TIMEOUT = 10.0
# How long a build waits for the outgoing session to actually stop stepping before it gives up on
# capturing CUDA graphs. Generous on purpose: an eager step with four cars is ~90 ms, and the answer
# only has to be trustworthy, not fast.
HOLD_PARK_TIMEOUT_S = 5.0

#: Vehicle constants the console needs for car placement. They live in the renderer in the old
#: viewer (`NativeViewer`) and are not simulator parameters; passed through unchanged.
COG_Z, WHEEL_R = 0.06, 0.056

#: `gym_env._obs` divides the VESC roll/pitch estimate by this literal (`gym_env.py:335`). Unlike
#: the other observation normalizers it is not a config field, so a checkpoint trained with a
#: different one cannot be reproduced here -- only refused.
ATT_SCALE_FIXED = 0.35

#: Saved `extra["spec"]` scalar -> where the same number lives when building an environment.
#: These do not change any tensor's shape, which is exactly why they need their own check: a
#: checkpoint trained with `v_max` 8 driven in a `v_max` 10 environment sees every speed and every
#: action scaled differently while every dimension still matches.
NORMALIZER_FIELDS = ("v_max", "range_max", "gyro_scale", "accel_scale", "att_scale")

# `GEOMETRY_VERSION`, `SMOOTH_SIGMA`, `_bbox`, `prop_batches` and the body of `build_geometry` live
# in `viewer/geometry.py` now (torch-free, shared with the environment editor); re-exported here so
# every existing import keeps working.
from .geometry import GEOMETRY_VERSION, SMOOTH_SIGMA, PropBuildError, _bbox, build_track_geometry   # noqa: E402,F401
from .geometry import prop_batches as _prop_batches_impl                                            # noqa: E402


def _now() -> float:
    return time.monotonic()


def _bbox(point_sets):
    """Axis-aligned bounds of the drawn content, or None when nothing was drawn."""
    pts = [np.asarray(p, np.float64).reshape(-1, 2) for p in point_sets if p is not None and len(p)]
    if not pts:
        return None
    a = np.vstack(pts)
    return (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max()))


class WorkerExit(Exception):
    """Raised to unwind out of the serve loop on `shutdown`."""


class Cancelled(Exception):
    """A start was abandoned (the user changed their mind while it was preparing)."""


class MuPin:
    """Hold every car's true friction at one value across resets.

    `sim.P["mu"]` is a view into the parameter bank, and `ParamSet.resample` rewrites it on every
    reset, so a one-off write would last exactly one episode. This wraps `env._reset_envs` the way
    the grip runtime's command spy does and re-applies the value to the ids that were just reset.
    Only mu is pinned; the other randomised parameters keep doing what the session asked.
    """

    def __init__(self, env, mu: float):
        self.env, self.mu, self._orig = env, float(mu), None

    def install(self) -> "MuPin":
        if self._orig is not None:
            raise RuntimeError("already installed")
        self._orig = self.env._reset_envs

        def wrapped(ids):
            out = self._orig(ids)
            if ids.numel():
                self.env.sim.P["mu"][ids] = self.mu
            return out

        self.env._reset_envs = wrapped
        return self

    def set(self, mu: float) -> None:
        """Change the pinned value and apply it to every car now, mid-episode."""
        self.mu = float(mu)
        self.env.sim.P["mu"].fill_(self.mu)

    def release(self) -> None:
        if self._orig is not None:
            self.env._reset_envs = self._orig
            self._orig = None


class StartConfigError(ValueError):
    """A start request that cannot be satisfied, with a message meant for a person."""


# ==================================================================== pure helpers
# Module level and torch-free on purpose: the console-side tests and the worker lifecycle tests can
# exercise them without importing torch or starting a process.

def validate_start_config(cfg: P.SessionConfig) -> None:
    """Reject a start we cannot honour, before anything expensive happens.

    `races` x `cars_per_race` is a product, so unlike the old `--cars` / `--race-size` pair there
    is no combination to reconcile -- only ranges to check.
    """
    if int(cfg.races) < 1:
        raise StartConfigError(f"레이스 수는 1 이상이어야 합니다 (받은 값 {cfg.races}).")
    if int(cfg.cars_per_race) < 1:
        raise StartConfigError(f"레이스당 차량 수는 1 이상이어야 합니다 (받은 값 {cfg.cars_per_race}).")
    if not str(cfg.map_name).strip():
        raise StartConfigError("맵을 선택해 주세요.")
    if cfg.speed_cap is not None and not (0.0 < float(cfg.speed_cap) <= 30.0):
        raise StartConfigError(f"속도 상한 {cfg.speed_cap} m/s 는 범위를 벗어났습니다 (0 초과 30 이하).")
    if int(cfg.max_render_cars) < 1:
        raise StartConfigError(f"화면 표시 차량 수는 1 이상이어야 합니다 (받은 값 {cfg.max_render_cars}).")
    if str(cfg.device) not in ("auto", "cpu", "cuda") and not str(cfg.device).startswith("cuda:"):
        raise StartConfigError(f"장치 이름을 알 수 없습니다: {cfg.device!r} (auto / cpu / cuda).")
    ros2 = str(getattr(cfg, "ros2", "off") or "off")
    if ros2 not in ("off", "publish", "drive"):
        raise StartConfigError(f"ROS2 연동 모드를 알 수 없습니다: {ros2!r} (off / publish / drive).")
    if ros2 != "off":
        from .ros_link import ros2_available
        why = ros2_available()
        if why is not None:
            raise StartConfigError(
                "ROS2 연동에는 rclpy 와 메시지 패키지가 필요합니다. ROS 워크스페이스를 소싱한 셸"
                "(예: source activate.sh)에서 콘솔을 열어 주세요. 가져오기 실패: " + why)


def resolve_checkpoint(run: str, runs_dir: str, latest: str = "") -> str:
    """`run` (a run name, an absolute path, a .pt file, or "latest") -> a checkpoint path.

    Every failure here is something the user can act on, so each one says which path was tried.
    """
    run = (run or "latest").strip()
    if run == "latest":
        if not latest:
            raise FileNotFoundError(f"{runs_dir} 아래에 체크포인트가 있는 런이 없습니다.")
        path = latest
    elif os.path.isabs(run):
        path = run
    else:
        path = os.path.join(runs_dir, run)
    if path.endswith(".pt"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"체크포인트 파일이 없습니다: {path}")
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"런 디렉터리가 없습니다: {path}")
    for name in ("ppo_latest.pt", "student_latest.pt"):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{path} 에 ppo_latest.pt / student_latest.pt 가 없습니다.")


def obs_spec_problems(meta: Dict[str, Any], spec, skipped: List[str], env_act_dim: int) -> List[str]:
    """Everything about this checkpoint that does not match the environment we just built.

    Why this exists: `learn.model.load_checkpoint` builds the network from the checkpoint's own
    `meta`, then copies only the tensors whose shapes still match and records the rest in
    `extra["skipped"]`. A checkpoint that disagrees with its environment therefore does not fail --
    it drives, with part of the actor at its random initialisation. The picture looks like a policy
    and is not one, which is the worst possible failure for a tool whose job is to be believed.

    Returns a list of sentences; empty means the model and the environment agree.
    """
    out: List[str] = []
    pairs = (
        ("n_beams", "n_beams", int(getattr(spec, "n_beams", -1)), "LiDAR 빔 수"),
        ("n_stack", "scan_stack", int(getattr(spec, "scan_stack", -1)), "스캔 스택"),
        ("proprio_dim", None, int(getattr(spec, "proprio_dim", -1)), "proprio 차원"),
    )
    for meta_key, _spec_key, env_value, label in pairs:
        want = meta.get(meta_key)
        if want is None:
            continue
        if int(want) != env_value:
            out.append(f"{label}: 체크포인트 {int(want)} vs 이 세션 {env_value}")
    want_act = meta.get("act_dim")
    if want_act is not None and int(want_act) != int(env_act_dim):
        out.append(f"행동 차원: 체크포인트 {int(want_act)} vs 이 세션 {int(env_act_dim)}")
    actor_skipped = sorted(k for k in (skipped or []) if k.startswith("actor."))
    if actor_skipped:
        shown = ", ".join(actor_skipped[:4]) + (" 외" if len(actor_skipped) > 4 else "")
        out.append(f"actor 가중치 {len(actor_skipped)}개가 파일에서 읽히지 않고 초기값으로 남았습니다 ({shown})")
    return out


def normalizer_problems(saved_spec: Dict[str, Any], env_spec, tol: float = 1e-6) -> List[str]:
    """Scalar observation contract: what the checkpoint was normalised with vs what this env uses.

    Separate from `obs_spec_problems` because these mismatches are invisible to a shape check. Every
    one of them leaves `proprio_dim`, `n_beams` and `act_dim` identical while changing what the
    numbers *mean*: `speed / v_max`, `range / range_max`, `gyro / gyro_scale`. The policy would run
    and the picture would look plausible, while the car drove on observations it never saw in
    training.

    The worker sets the ones an `EnvConfig` / `Config` can carry, so anything still different here
    is something that could not be applied -- and is a refusal, not a warning.
    """
    out: List[str] = []
    labels = {"v_max": "속도 정규화 v_max", "range_max": "LiDAR range_max",
              "gyro_scale": "자이로 정규화", "accel_scale": "가속도 정규화",
              "att_scale": "roll/pitch 정규화"}
    for key in NORMALIZER_FIELDS:
        want = saved_spec.get(key)
        if want is None:
            continue
        have = getattr(env_spec, key, None)
        if have is None:
            continue
        if abs(float(want) - float(have)) > tol:
            out.append(f"{labels[key]}: 체크포인트 {float(want):g} vs 이 세션 {float(have):g}")
    return out


def where_of(exc: BaseException) -> str:
    """Which stage of the console's mental model an exception belongs to."""
    name = type(exc).__name__
    if isinstance(exc, StartConfigError):
        return "설정"
    if isinstance(exc, FileNotFoundError):
        return "체크포인트"
    if "CUDA" in str(exc) or "cuda" in name.lower() or "OutOfMemory" in name:
        return "GPU"
    if isinstance(exc, ValueError):
        return "설정"
    return "세션 준비"


def _copy(a) -> np.ndarray:
    """A numpy array that owns its memory.

    `tensor.float().cpu().numpy()` is a *view* of the tensor's storage whenever `float()` and
    `cpu()` are both no-ops, which they are for a float32 tensor already on the CPU. Sending that
    across the pipe is fine only because pickling copies; keeping it in a cache, or handing it to
    the bounded slot, is not -- the next step rewrites the frame that is already queued.
    """
    return np.array(a, copy=True)




def prop_batches(track) -> list:
    """`geometry.prop_batches`, with its failure reported as this module's `StartConfigError`.

    The obstacle stays in the simulation whether or not it can be drawn, so a prop that will not
    build refuses the session rather than vanishing from the screen -- see `geometry.prop_batches`.
    """
    try:
        return _prop_batches_impl(track)
    except PropBuildError as exc:
        raise StartConfigError(str(exc)) from exc


def _physics_fell_back(session) -> Optional[bool]:
    """Has `Simulator._guarded` swapped either compiled physics callable for its eager original?

    None when there is nothing to fall back from -- CPU, or compiling switched off -- because
    "False" there would read as "the compiled path is holding", which is a different claim.
    """
    try:
        sim = session["env"].sim
        if sim.device.type != "cuda" or not sim.cfg.sim.compile:
            return None
        # Only the callables that were actually wrapped can have fallen back. `Simulator.__init__`
        # sets `_roll = _roll_physics; _post = _post_roll` first and then replaces them, but it
        # replaces `_post` only in "reduce-overhead" mode -- so in the default mode `_post` is the
        # bare method by construction, and reading that as a fallback reports one that never
        # happened. (It did: the first run of this check said the physics had fallen back on both
        # maps while the RTF sat at 1.00, which is what gave it away.)
        wrapped = ["_roll"] + (["_post"] if sim.cfg.sim.compile_mode == "reduce-overhead" else [])
        eager_names = {"_roll": "_roll_physics", "_post": "_post_roll"}
        return any(getattr(getattr(sim, attr, None), "__qualname__", "").endswith(eager_names[attr])
                   for attr in wrapped)
    except Exception:
        return None


class SimWorker:
    """Owns the torch side. One instance per console process."""

    def __init__(self, control, frames):
        self.control = control            # duplex Connection to the console
        self.frames = frames              # one-way Connection, worker -> console
        self.gen = -1                     # generation currently being simulated
        self.prepare_gen: Optional[int] = None   # generation being built right now
        self.cancel_gen: Optional[int] = None
        self.seq = 0                      # frames produced in this generation
        self.running = False
        self.paused = False
        self.alive = True
        self.cfg: Optional[P.SessionConfig] = None
        self.session: Optional[Dict[str, Any]] = None
        self._track_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._slot = LatestSlot(OUTBOUND_CAPACITY)
        self._sender = threading.Thread(target=self._send_loop, name="frame-sender", daemon=True)
        self._sender.start()
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_rate = 0.0
        self._last_send = 0.0
        self._min_send_dt = 1.0 / MAX_FRAME_HZ
        self._overlay = {"saliency": False, "internals": False, "plan": True}
        self._sal_cache: Optional[np.ndarray] = None
        self._sal_time = 0.0
        self._focus = 0
        self._say_lock = threading.Lock()
        self._pending: "deque[dict]" = deque()      # commands for the sim thread
        self._pending_lock = threading.Lock()
        self._queued_start: Optional[dict] = None
        self._shutdown_requested = False
        self._heartbeat_raised = False
        #: Set while a new session is being built. The outgoing session stays *alive* -- so a
        #: cancel is free and its map stays on screen -- but stops stepping. See `_build_hold`.
        self._hold = False
        self._hold_parked = threading.Event()
        self._hold_parked_ok = False      # did the outgoing session confirm it stopped stepping?
        #: sessions whose simulation thread would not stop. Held so their tensors stay alive under
        #: the thread that is still using them; never freed, never stepped again.
        self._stuck: List[Dict[str, Any]] = []
        #: CUDA memory sampled at each session boundary, so repeated map switching can be shown to
        #: settle rather than assumed to. Bounded: only the last few boundaries are kept.
        self._mem_trace: "deque[dict]" = deque(maxlen=64)

    # ================================================================ plumbing
    def cuda_mem(self) -> Optional[Dict[str, Any]]:
        """Live vs cached vs driver-visible CUDA bytes, in MB.

        `nvidia-smi` alone cannot answer "did this leak": the caching allocator never returns freed
        blocks to the driver, so a flat process footprint and a growing one look the same from
        outside. `allocated` is live tensors, `reserved` is what the allocator holds either way, and
        `driver` is what the process is actually charged -- which also covers what the allocator
        does not account for at all (CUDA graph pools, cuBLAS/cuDNN workspaces, the context).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            MB = 1024.0 * 1024.0
            free_b, total_b = torch.cuda.mem_get_info()
            return {
                "allocated_mb": round(torch.cuda.memory_allocated() / MB, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / MB, 1),
                "max_reserved_mb": round(torch.cuda.max_memory_reserved() / MB, 1),
                # device-wide, so it moves with other processes too; kept for continuity with the
                # nvidia-smi series and labelled as such
                "device_used_mb": round((total_b - free_b) / MB, 1),
            }
        except Exception:
            return None

    def _mark_mem(self, where: str, **extra) -> None:
        m = self.cuda_mem()
        if m is not None:
            self._mem_trace.append(dict(where=where, gen=self.gen, **extra, **m))

    def say(self, kind: str, **fields) -> None:
        """Status back to the console, on the control channel so it stays ordered with acks.

        Locked because the simulation thread sends its own acks: `Connection.send` is not
        documented to be thread safe, and two interleaved pickles on one pipe is a corrupt stream.
        """
        with self._say_lock:
            try:
                self.control.send(P.message(kind, **fields))
            except (BrokenPipeError, OSError, EOFError):
                self.alive = False          # the console is gone; stop talking, keep tidying up

    def _send_loop(self) -> None:
        """Drains the bounded slot onto the pipe.

        Its own thread on purpose: if the console stops reading, the pipe fills and *this* thread
        blocks, while the simulation carries on and the slot quietly drops the frames nobody is
        going to see. The alternative -- sending from the simulation thread -- makes the physics
        run at the speed of the slowest consumer.
        """
        while True:
            item = self._slot.get(timeout=0.25)
            if item is None:
                if self._slot.closed:
                    return
                continue
            try:
                self.frames.send(item)
            except (BrokenPipeError, OSError, EOFError, ValueError):
                return                      # renderer gone: stop sending, never stop simulating

    def _drain_slot(self) -> None:
        """Throw away frames still waiting to be sent. Used at a generation swap so the console is
        not handed the previous map's cars after the new map's geometry."""
        while len(self._slot):
            if self._slot.get(timeout=0.0) is None:
                break

    def emit_frame(self, payload: dict, gen: int) -> None:
        if gen != self.gen:
            return
        payload["created_monotonic"] = _now()
        payload["worker_dropped"] = self._slot.dropped
        self._slot.put(payload)

    def stage(self, gen: int, name: str, note: str = "") -> None:
        self.say(P.MSG_STAGE, gen=gen, stage=name, note=note)
        self._check_cancel(gen)

    def _check_cancel(self, gen: int) -> None:
        """Drain pending control messages during a long build so cancel actually cancels.

        Preparing a session is the one place where the worker is busy for many seconds without
        looping. Polling here is what makes 취소 responsive rather than decorative.
        """
        while self.alive and self.control.poll():
            try:
                msg = self.control.recv()
            except (EOFError, OSError):
                self.alive = False
                break
            self.handle(msg, preparing=gen)
        if not self.alive or self._shutdown_requested:
            raise Cancelled()
        if self.cancel_gen is not None and self.cancel_gen == gen:
            raise Cancelled()

    # ================================================================ catalogue
    def map_catalog(self) -> Dict[str, List[str]]:
        """Named groups over the map catalog.

        The names say "기본 평가셋 / 기본 학습셋" rather than "미학습": these are the project's own
        split constants, and a run may have been resumed with a different split, so the honest claim
        is "this is the project's default split", not "this policy has never seen this map".
        Mirrors `console.catalog.GROUP_ORDER`.
        """
        from ..learn import common
        from .. import maps

        def uniq(seq):
            out, seen = [], set()
            for n in seq:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
            return out

        catalog = ([f"real:{k}" for k in maps.REAL]
                   + [f"rt:{k}" for k in maps.racetrack_names()]
                   + [f"gym:{k}" for k in maps.gym_map_names()]
                   + [f"gen:{s}:{i}" for s in ("competition", "control", "serpentine", "circuit", "hallway")
                      for i in range(4)])
        # `+props<seed>` asks the loader for the same map with modelled obstacles standing on it
        # (`Track.with_static_props`) -- boxes, crates, drums, held as finite convex sections. The
        # simulator collides with them and the LiDAR returns them.
        #
        # Offered over the eval maps rather than a hand-picked three, so the obstacle group is the
        # obstacle group rather than a demo. The suffix order matters and is the loader's, not a
        # choice: `<base>+props<seed>~rev`, the same order `+obs101~rev` uses. Bases that already
        # carry an obstacle suffix are skipped -- stacking two is not a thing the loader parses.
        #
        # None of this touches `common.EVAL_OBSTACLE_TRACKS`, and no existing `+obs` / `+rlobs`
        # name is reinterpreted: those keep meaning exactly what they meant.
        prop_seeds = (3, 7, 11)
        prop_bases, seen_base = [], set()
        for n in uniq(common.EVAL_TRACKS):
            base, _, direction = n.partition("~")
            if "+" in base:                      # already an obstacle variant; do not stack
                continue
            if not base.startswith(("real:", "rt:", "gen:")):
                continue
            key = (base, direction)
            if key in seen_base:
                continue
            seen_base.add(key)
            prop_bases.append((base, direction))
        props_variants = [f"{base}+props{seed}" + (f"~{d}" if d else "")
                          for base, d in prop_bases for seed in prop_seeds]
        groups: Dict[str, List[str]] = {}
        # The user's own environments (환경 page) come first when there are any: someone who just
        # pressed 주행 in the editor is looking for the scene they built, not the eval split.
        # Omitted entirely when empty, so the picker never shows an empty group.
        scenes = [f"scene:{n}" for n in maps.scene_names()]
        if scenes:
            groups[SCENES_GROUP] = scenes
        groups.update({
            "기본 평가셋": uniq(common.EVAL_TRACKS),
            # The obstacles a viewer session should normally be looking at.
            "장애물 (상자·궤짝·드럼)": uniq(props_variants),
            "기본 학습셋": uniq(common.TRAIN_TRACKS),
            # Kept, and labelled for what it is: the obstacle sets earlier experiments trained and
            # evaluated against. They are stamped into the occupancy grid, so they are rotated
            # rectangles of one height with no top -- not the modelled props above. Reproducing an
            # earlier result needs these; looking at obstacles does not.
            "이전 실험 재현 (격자 장애물)": uniq(common.EVAL_OBSTACLE_TRACKS),
            "전체 카탈로그": uniq(catalog + scenes),
        })
        return groups

    def describe_checkpoint(self, run: str) -> dict:
        """Read a checkpoint's metadata without building anything."""
        import torch
        from ..learn import common
        from ..learn.watch import checkpoint_speed_cap, describe_checkpoint_line, latest_run
        ckpt = resolve_checkpoint(run, common.RUNS_DIR, latest_run())
        extra = (torch.load(ckpt, map_location="cpu", weights_only=False) or {}).get("extra", {})
        metrics = extra.get("metrics") or {}
        bits = []
        if "collision_rate" in metrics:
            bits.append(f"충돌률 {metrics['collision_rate']:.2f}")
        if "progress_rate_mps" in metrics:
            bits.append(f"{metrics['progress_rate_mps']:.1f} m/s")
        lap = metrics.get("lap_time_s")
        if lap and lap == lap:
            bits.append(f"랩 {lap:.1f} s")
        return {
            "run": run,
            "path": ckpt,
            "file": os.path.basename(ckpt),
            "age_s": max(0.0, time.time() - os.path.getmtime(ckpt)),
            "progress": describe_checkpoint_line(extra, ckpt),
            "metrics": ("저장 시점: " + " · ".join(bits)) if bits else "체크포인트에 지표 기록 없음",
            "speed_cap": checkpoint_speed_cap(extra, ckpt, fallback=None),
        }

    # ================================================================ session build
    def load_track(self, name: str):
        from ..learn import common
        if name.startswith("scene:"):
            # An editor scene is edited and driven again inside one worker lifetime; a cached
            # track would be the scene as it was the first time. `maps.load` keys its own cache
            # on the files' mtimes, so a reload here is cheap when nothing changed.
            self._track_cache.pop(name, None)
        if name in self._track_cache:
            self._track_cache.move_to_end(name)
            return self._track_cache[name]
        try:
            tracks, _ = common.load_tracks([name], racelines=False, drop_infeasible=False)
        except Exception as exc:
            raise StartConfigError(f"맵 '{name}' 을 읽지 못했습니다: {exc}") from exc
        if not tracks:
            raise StartConfigError(f"맵 '{name}' 을 찾지 못했습니다. 목록에서 다시 골라 주세요.")
        track = tracks[0]
        self._track_cache[name] = track
        while len(self._track_cache) > TRACK_CACHE_SIZE:
            self._track_cache.popitem(last=False)
        return track

    def build_session(self, cfg: P.SessionConfig, gen: int) -> dict:
        """Everything from a config to a stepping simulator. Raises Cancelled if abandoned.

        Runs on the control thread, and deliberately: `torch.compile`'s CUDA graph trees keep
        per-thread state, so a graph captured on the simulation thread would not be reused and the
        session would silently fall back to eager. The warmup steps below are the capture.
        """
        import torch
        from ..gym_env import EnvConfig
        from ..learn import common
        from ..learn.model import load_checkpoint
        from ..learn.watch import (Introspector, actor_runner, checkpoint_speed_cap,
                                   describe_checkpoint_line, latest_compatible_run, latest_run,
                                   viewer_config)

        self.stage(gen, "checkpoint")
        if (cfg.run or "latest").strip() == "latest":
            # For an unattended "latest", resolve to the newest run this worker can actually open:
            # it loads with no opt-in flag, so a non-legacy controller arm or a conditional
            # checkpoint is refused by `load_checkpoint` and the session would fail on the run the
            # user is least surprised by. `latest_run()` keeps meaning "newest" for `describe` and
            # for the training side. Every skipped run is reported, so this is a visible choice
            # rather than a silent substitution.
            chosen, skipped = latest_compatible_run()
            for name, why in skipped:
                self.say(P.MSG_LOG, gen=gen,
                         text=f"'{name}' 건너뜀: {why} — 이 뷰어는 기본 컨트롤러로 실행합니다")
            if not chosen and skipped:
                raise StartConfigError(
                    "기본 컨트롤러로 열 수 있는 런이 없습니다: "
                    + ", ".join(f"{n}({w})" for n, w in skipped)
                    + ". 런을 직접 고르거나, 해당 런타임을 설치하는 호출자로 실행해 주세요.")
            ckpt_path = resolve_checkpoint(chosen or "latest", common.RUNS_DIR, latest_run())
            self.say(P.MSG_LOG, gen=gen,
                     text=f"체크포인트: {os.path.basename(os.path.dirname(ckpt_path))}"
                          f"/{os.path.basename(ckpt_path)}")
            # The MSG_LOG above is transient -- the console overwrites it on READY. The facts are
            # what the header keeps, so the substitution stays visible after the session starts.
            autoselect = {"requested": "latest", "skipped": [list(t) for t in skipped]}
        else:
            autoselect = None
            ckpt_path = resolve_checkpoint(cfg.run, common.RUNS_DIR, latest_run())

        if cfg.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise StartConfigError("CUDA 를 쓸 수 없습니다 (torch.cuda.is_available() = False). 장치를 cpu 로 바꿔 주세요.")
        # Nothing to compile on the CPU: inductor leaves this graph of hundreds of tiny ops alone.
        compile_enabled = bool(cfg.compile) and device.type == "cuda"

        # The plan-controller arm this session installs. A checkpoint trained under a non-legacy
        # arm may only run under that same arm; legacy-trained weights may run under any arm (the
        # benchmark's declared cross-runtime case, and the configuration that scored best).
        arm = str(getattr(cfg, "controller", "legacy") or "legacy")
        if arm not in ("legacy", "estimated", "fixed_low", "oracle"):
            raise StartConfigError(f"플랜 제어기 '{arm}' 은 이 뷰어가 지원하지 않습니다 (legacy / estimated / fixed_low / oracle).")
        if arm != "legacy" and int(cfg.cars_per_race) > 1:
            raise StartConfigError(f"플랜 제어기 '{arm}' 은 레이스당 차량 수 1에서만 지원합니다 "
                                   f"(학습·벤치마크와 같은 조건). 레이스당 차량 수를 1로 두거나 legacy 를 고르세요.")
        estimator_path = ""
        if arm == "estimated":
            from .console.protocol import SessionConfig as _SC
            estimator_path = (getattr(cfg, "estimator", "") or "").strip() or _SC.default_estimator()
            if not estimator_path or not os.path.isfile(estimator_path):
                raise StartConfigError("estimated 제어기는 노면 추정기 .pt 가 필요합니다. 고급 설정의 '노면 추정기' 에 "
                                       "경로를 넣거나 ~/f1sim_runs/_estimators/estimator_seed401.pt 를 두세요.")
        # Peek the arm the checkpoint was trained under before loading, so a mismatch is a start
        # message that names the setting to change rather than the loader's refusal.
        from ..learn.model import controller_arm_of
        try:
            recorded = controller_arm_of(torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True))
        except Exception:
            recorded = "legacy"                  # the loader below will say what is wrong with it
        if recorded != "legacy" and recorded != arm:
            raise StartConfigError(f"{os.path.basename(ckpt_path)}: '{recorded}' 제어기로 학습된 체크포인트입니다. "
                                   f"고급 설정의 플랜 제어기를 '{recorded}' 로 맞추세요 (현재 '{arm}').")
        model, extra = load_checkpoint(ckpt_path, device, allow_controller=(arm != "legacy"))
        model.eval()
        intro = Introspector(model)
        act_dim = model.meta.get("act_dim", 2)
        from ..mpc import ACT_DIM as PLAN_DIM
        if act_dim not in (2, PLAN_DIM):
            raise StartConfigError(
                f"{os.path.basename(ckpt_path)}: 행동 차원 {act_dim} 은 예전 플랜 표현입니다 "
                f"(현재는 {PLAN_DIM}). 더 최신 런을 고르세요.")
        mode = "plan" if act_dim == PLAN_DIM else "direct"
        if mode == "direct" and arm != "legacy":
            # The grip arms bound the *plan tracker*; a direct-action checkpoint has none. Refusing
            # the session for a setting that cannot apply (the deployment default is fixed_low)
            # made every direct checkpoint unopenable; say what happened and run it as it is.
            self.say(P.MSG_LOG, gen=gen,
                     text=f"플랜 제어기 '{arm}' 은 직접 행동(direct) 체크포인트에 적용되지 않아 legacy 로 실행합니다")
            arm = "legacy"
            estimator_path = ""
        speed_cap = float(cfg.speed_cap or checkpoint_speed_cap(extra, ckpt_path))
        self._check_cancel(gen)

        self.stage(gen, "map", cfg.map_name)
        track = self.load_track(cfg.map_name)
        self._check_cancel(gen)

        races = max(1, int(cfg.races))
        grid = max(1, int(cfg.cars_per_race))
        n_cars = races * grid                      # a product: nothing to truncate
        need_rl = grid > 1 and cfg.opponent == "teacher"

        self.stage(gen, "raceline")
        rls = None
        try:
            from ..raceline import Raceline
            rls = [Raceline.build_cached(track)]
        except Exception as exc:
            # The line is only drawn unless teacher opponents actually drive it.
            if need_rl:
                raise StartConfigError(
                    f"이 맵의 레이싱 라인을 만들지 못해 teacher 상대차를 넣을 수 없습니다: {exc}") from exc
            self.say(P.MSG_LOG, gen=gen, text=f"레이싱 라인 생략(그리기용): {exc}")
        self._check_cancel(gen)

        self.stage(gen, "env")
        spec = extra.get("spec") or {}
        env_cfg = EnvConfig(
            speed_cap=speed_cap, action_mode=mode, race_size=grid, opponent=cfg.opponent,
            max_steps=int(cfg.episode_s * 40),
            # opponents at the policy's own cap and the teacher at full raceline pace
            selfplay_front_cap=False, opp_speed_range=(1.0, 1.0),
            scan_stack=spec.get("scan_stack", 3), scan_stride=spec.get("scan_stride", 1),
            hist_len=spec.get("hist_len", 0), hist_stride=spec.get("hist_stride", 2),
            compile_tracker=compile_enabled,
            # a switch moves every car itself; re-drawing tracks on reset would scatter them
            resample_track_on_reset=False)
        if "action_history" in spec:
            # `watch.main` does not pass this through, so a checkpoint trained with a different
            # action history fails there with a shape error deep in the actor. Taking it from the
            # checkpoint's own spec is what makes the compatibility check below meaningful.
            env_cfg.action_history = int(spec["action_history"])
        # The scalar normalizers, which change no shape and so are invisible to a dimension check.
        # Reproduce the ones an EnvConfig / Config can carry; anything left over is caught below.
        if "v_max" in spec:
            env_cfg.v_max_policy = float(spec["v_max"])
        if "gyro_scale" in spec:
            env_cfg.imu_gyro_scale = float(spec["gyro_scale"])
        if "accel_scale" in spec:
            env_cfg.imu_accel_scale = float(spec["accel_scale"])
        if float(spec.get("att_scale", ATT_SCALE_FIXED)) != ATT_SCALE_FIXED:
            raise StartConfigError(
                f"{os.path.basename(ckpt_path)}: roll/pitch 정규화가 {float(spec['att_scale']):g} 로 "
                f"학습됐는데 시뮬레이터는 {ATT_SCALE_FIXED:g} 로 고정돼 있습니다 "
                f"(gym_env._obs). 이 체크포인트는 이 뷰어로 관측을 재현할 수 없습니다.")
        sim_cfg = viewer_config(compile_enabled, randomize=cfg.randomize)
        if "range_max" in spec:
            sim_cfg.lidar.range_max = float(spec["range_max"])
        env = common.make_env([track], n_cars, device, env_cfg, cfg=sim_cfg,
                              rls=rls if need_rl else None)
        env.sim.tid.fill_(0)                       # every car on the map that was asked for
        mu_pin = None
        if str(getattr(cfg, "mu_mode", "random")) == "fixed":
            mu_pin = MuPin(env, float(getattr(cfg, "mu", 1.0489))).install()   # before the first reset

        env_spec = common.obs_spec(env)
        problems = (obs_spec_problems(model.meta, env_spec, extra.get("skipped") or [], env.act_dim)
                    + normalizer_problems(spec, env_spec))
        if problems:
            raise StartConfigError(
                f"{os.path.basename(ckpt_path)} 의 관측 구성이 이 세션과 다릅니다 — "
                + "; ".join(problems)
                + ". 이대로 돌리면 파일에 없는 가중치가 초기값으로 남은 채 주행하게 되므로 "
                  "화면의 주행을 그 체크포인트의 정책으로 볼 수 없습니다. 다른 런을 고르세요.")

        obs, _info = env.reset()
        self._check_cancel(gen)

        self.stage(gen, "warmup")
        env.sim.warmup()
        self._check_cancel(gen)

        session = {
            "mu_pin": mu_pin,
            "env": env, "model": model, "extra": extra, "intro": intro, "device": device,
            "mode": mode, "ckpt_path": ckpt_path, "mtime": os.path.getmtime(ckpt_path),
            "autoselect": autoselect,
            "obs": obs, "speed_cap": speed_cap, "compile": compile_enabled,
            "act_fn": actor_runner(model, device, compile_enabled),
            "info_line": describe_checkpoint_line(extra, ckpt_path),
            "track": track, "cfg": cfg, "focus": 0, "k": 0,
            "gg": deque(maxlen=90),
            "last_reload": time.time(),
            "raceline": rls[0] if rls else None,
        }
        if compile_enabled:
            for i in range(8):
                self.stage(gen, "compile", f"{i + 1}/8")
                self._step_once(session)
        self._check_cancel(gen)

        self.stage(gen, "geometry", cfg.map_name)
        session["geometry"] = self.build_geometry(track, session["raceline"])
        # Last, so that nothing which can raise runs between installing the fast path's hooks and
        # handing the session over. A session abandoned after `install()` would keep the env alive
        # through the `sim -> _roll -> fastpath -> sim` cycle.
        if not compile_enabled:
            self._prepare_fastpath(session, gen)
        if arm != "legacy":
            # After the graphs: `install` hooks `tracker._solver`, and the fast path's captured MPC
            # arguments let the grip solver capture its own graph instead of running eager.
            from ..learn.grip_runtime import ControllerRuntime
            self.say(P.MSG_LOG, gen=gen, text=f"플랜 제어기 설치: {arm}"
                     + (f" | 추정기 {os.path.basename(estimator_path)}" if estimator_path else ""))
            rt = ControllerRuntime(env, arm, estimator_path or None, device=device)
            try:
                rt.install(graph_rt=session.get("fastpath"), adopt=False)   # adopted on the sim thread
                rt.begin(session["obs"])
            except BaseException:
                rt.release()
                raise
            session["controller"] = rt
            session["controller_arm"] = arm
            session["estimator_path"] = estimator_path
        ros2 = str(getattr(cfg, "ros2", "off") or "off")
        if ros2 != "off":
            # After everything that can refuse the session: a node that came up for a session
            # which then failed would leave topics on the graph with nobody behind them.
            self.stage(gen, "ros2", ros2)
            from .ros_link import RosLink
            link = RosLink(env.sim, track, session["raceline"], mode=ros2, car=0)
            session["ros"] = link
            self.say(P.MSG_LOG, gen=gen,
                     text=f"ROS2 연동 ({ros2}): 노드 {link.node.get_name()}, 차량 0 의 센서를 발행"
                          + (f", {link.drive_topic} 로 제어" if ros2 == "drive" else ""))
        return session

    # ------------------------------------------------------------- graph capture
    def _prepare_fastpath(self, session: dict, gen: int) -> None:
        """Capture this session's CUDA graphs: one per IMU phase, plus the plan solver.

        Viewer only, and only with `torch.compile` off -- the two are alternatives, not layers.
        Eligibility is decided first, so an ineligible session says why and runs eager; a capture
        that is attempted and *fails* is fatal, because it leaves the CUDA generator in a state
        where `torch.randn(device="cuda")` raises while ordinary arithmetic still works.
        """
        from .graph_fastpath import NotCapturable, SimGraphFastPath, roll_eligible
        env = session["env"]
        sim = env.sim
        ok, why = roll_eligible(sim)
        if not ok:
            self.say(P.MSG_LOG, gen=gen, text=f"CUDA 그래프 없이 실행합니다: {why}")
            return
        if not self._hold_parked_ok:
            # `_build_hold` could not confirm the outgoing session had stopped stepping. Capture
            # while another thread draws from the default CUDA generator is the failure this whole
            # module has to avoid, and it is not worth risking for a speed-up.
            self.say(P.MSG_LOG, gen=gen,
                     text="이전 세션이 멈춘 것을 확인하지 못해 CUDA 그래프 캡처를 건너뜁니다.")
            return

        # The exact arguments the real step passes, recorded from real steps rather than
        # reconstructed -- `delay_s` is a tensor or a float depending on the jitter setting, and
        # guessing that wrong would capture against the wrong signature. These steps double as the
        # warm-up the compiled path does eight of.
        self.stage(gen, "graph", "인자 기록")
        tracker = getattr(env, "tracker", None)
        lidar = getattr(sim, "lidar", None)
        rec: dict = {}
        eager_roll = sim._roll
        prev_solver = getattr(tracker, "_solver", None) if tracker is not None else None

        def roll_spy(*a):
            rec["roll"] = a
            return eager_roll(*a)

        sim._roll = roll_spy
        # The two prop leaves, recorded from the same steps. They are plain methods on their class,
        # so the spy is an instance attribute that has to be *deleted* afterwards rather than
        # assigned back -- putting the bound method in the instance dict would leave a permanent
        # reference to the object it is bound to.
        eager_contact = sim._prop_contact
        eager_merge = getattr(lidar, "_merge_props", None)

        def contact_spy(*a):
            rec["contact"] = a
            return eager_contact(*a)

        def merge_spy(*a):
            rec["merge"] = a
            return eager_merge(*a)

        sim._prop_contact = contact_spy
        if eager_merge is not None:
            lidar._merge_props = merge_spy
        if tracker is not None:
            from .. import mpc as _mpc
            base = prev_solver or (_mpc.solve_fast if tracker.compile_solver else _mpc.solve)

            def solve_spy(*a):
                rec["mpc"] = a
                return base(*a)

            tracker._solver = solve_spy
        try:
            for _ in range(2):
                self._step_once(session)
        finally:
            sim._roll = eager_roll
            sim.__dict__.pop("_prop_contact", None)
            if eager_merge is not None:
                lidar.__dict__.pop("_merge_props", None)
            if tracker is not None:
                tracker._solver = prev_solver
        if "roll" not in rec:
            self.say(P.MSG_LOG, gen=gen,
                     text="물리 호출을 관측하지 못해 CUDA 그래프 캡처를 건너뜁니다.")
            return

        t0 = time.perf_counter()
        fp = SimGraphFastPath(sim, log=lambda t: self.stage(gen, "graph", t))
        try:
            self.stage(gen, "graph", "물리")
            fp.capture_roll(rec["roll"])
            if tracker is not None and "mpc" in rec:
                self.stage(gen, "graph", "플랜 솔버")
                try:
                    fp.capture_mpc(tracker, rec["mpc"])
                except NotCapturable as exc:
                    # The physics graphs are the larger win; the solver staying eager is a partial
                    # result, not a failure.
                    self.say(P.MSG_LOG, gen=gen, text=f"플랜 솔버는 eager 로 둡니다: {exc}")
            # The prop leaves. A map without props skips them by eligibility, not by omission, and
            # either one staying eager is a smaller win rather than a failed session.
            if "contact" in rec:
                self.stage(gen, "graph", "프롭 접촉")
                try:
                    fp.capture_prop_contact(rec["contact"])
                except NotCapturable as exc:
                    self.say(P.MSG_LOG, gen=gen, text=f"프롭 접촉은 eager 로 둡니다: {exc}")
            if lidar is not None and "merge" in rec:
                self.stage(gen, "graph", "프롭 LiDAR 병합")
                try:
                    fp.capture_merge_props(lidar, rec["merge"])
                except NotCapturable as exc:
                    self.say(P.MSG_LOG, gen=gen, text=f"프롭 LiDAR 병합은 eager 로 둡니다: {exc}")
            fp.install(tracker)
        except NotCapturable as exc:
            fp.release()
            self.say(P.MSG_LOG, gen=gen, text=f"CUDA 그래프 없이 실행합니다: {exc}")
            return
        except BaseException:
            fp.release()
            raise
        session["fastpath"] = fp
        session["fastpath_ms"] = round((time.perf_counter() - t0) * 1e3, 1)

    # ---------------------------------------------------------------- geometry
    def build_geometry(self, track, raceline=None) -> dict:
        """Static map meshes as plain arrays, ready for the console to upload.

        The body is `geometry.build_track_geometry` (torch-free, shared with the environment
        editor); this keeps the worker's error type for a prop that cannot be built.
        """
        try:
            return build_track_geometry(track, raceline)
        except PropBuildError as exc:
            raise StartConfigError(str(exc)) from exc

    def session_facts(self, session: dict, gen: int) -> dict:
        """What was actually built. The console shows this, never what the user asked for."""
        env = session["env"]
        cfg: P.SessionConfig = session["cfg"]
        total = int(env.sim.B)
        shown = min(total, int(cfg.max_render_cars))
        return {
            "gen": gen,
            "run": os.path.basename(os.path.dirname(session["ckpt_path"])) or session["ckpt_path"],
            "checkpoint": os.path.basename(session["ckpt_path"]),
            # present only when "latest" resolved to something other than the newest run, so the
            # header can say which run was opened and what it passed over
            "checkpoint_autoselect": session.get("autoselect"),
            # The catalogue key, because that is what the console's map list is keyed by and what
            # the user picked. `Track.name` is the loader's own name for the same thing
            # ("gen:competition:2" -> "competition_2"), so showing it would leave the header naming
            # a map that is not selected in the list.
            "map": str(cfg.map_name),
            "map_track_name": str(session["track"].name),
            "races": int(cfg.races), "cars_per_race": int(cfg.cars_per_race),
            "total_cars": total, "max_render_cars": shown,
            "car_ids": list(range(shown)),
            "device": str(session["device"]),
            "compile": bool(session["compile"]),
            # What the actor callable actually is, not what was asked for: `actor_runner` returns
            # the eager forward unchanged when compiling is off or the device is not CUDA. This is
            # the *actor* only -- the physics is compiled separately and reported below, and one
            # must not be read as evidence for the other.
            "actor_compiled": bool(getattr(session.get("act_fn"), "compiled", False)),
            "controller": str(session.get("controller_arm") or "legacy"),
            "estimator": os.path.basename(session.get("estimator_path") or ""),
            "physics_compile_requested": bool(env.sim.cfg.sim.compile),
            "physics_compile_mode": str(env.sim.cfg.sim.compile_mode),
            # The explicit CUDA-graph fast path, reported separately from `compile` because they are
            # alternatives: this is what runs when `compile` is off. `graph_phases` is how many IMU
            # phases were captured -- fewer than `imu_period` would be a bug, not a partial win --
            # and `graph_mpc` says whether the plan solver got one too.
            "graph_fastpath": session.get("fastpath") is not None,
            "graph_phases": len(getattr(session.get("fastpath"), "_by_phase", ()) or ()),
            "graph_mpc": getattr(session.get("fastpath"), "mpc", None) is not None,
            "graph_prop_contact": getattr(session.get("fastpath"), "_leaf_contact", None) is not None,
            "graph_merge_props": getattr(session.get("fastpath"), "_leaf_merge", None) is not None,
            "graph_capture_ms": session.get("fastpath_ms"),
            # Sampling, from the simulator's own configuration. Not inferred from the frame: roll
            # and pitch are present in a frame whether or not the IMU is sampling.
            "imu_enabled": bool(env.sim.cfg.imu.enabled),
            "imu_rate_hz": float(env.sim.cfg.imu.imu_rate),
            "speed_cap": float(session["speed_cap"]),
            "v_max_policy": float(env.ecfg.v_max_policy),
            "action_mode": session["mode"],
            "info_line": session["info_line"],
            "ros2": session["ros"].facts() if session.get("ros") is not None else None,
            "randomize": bool(cfg.randomize),
            "mu_mode": str(getattr(cfg, "mu_mode", "random")),
            "mu": float(getattr(cfg, "mu", 1.0489)),
            "lidar": {"fov": float(env.cfg.lidar.fov), "range_max": float(env.range_max),
                      "n_beams": int(env.n_beams)},
            "vehicle": {"lr": float(env.cfg.vehicle.lr), "cog_z": COG_Z, "wheel_r": WHEEL_R},
            # CUDA bytes as this session came up, plus the boundaries crossed to get here. Repeated
            # map switching is the case that has to be shown to settle, and this is the series that
            # shows it.
            "cuda_mem": self.cuda_mem(),
            "cuda_mem_trace": list(self._mem_trace),
        }

    # ================================================================ stepping
    def _step_once(self, session) -> None:
        """One control step. No `.item()`, no host sync: everything the console needs is gathered
        once per *sent* frame in `_snapshot`."""
        import torch
        from ..learn.obs import flatten_obs
        env = session["env"]
        model = session["model"]
        rt = session.get("controller")
        link = session.get("ros")
        if link is not None and link.take_reset():
            # `/f1sim/reset` from the ROS side: the same reset as the console's button, on the
            # thread that owns the session. No ack: nothing on the console asked for it.
            session["obs"], _ = env.reset()
            if rt is not None:
                rt.begin(session["obs"])
            session["gg"].clear()
            self._sal_cache = None
            self.say(P.MSG_LOG, gen=self.gen, text="ROS2 /f1sim/reset: 전 차량을 리셋했습니다")
        if rt is not None:
            rt.pre_action(session["obs"])          # history, friction estimate, MPC limits: before the action
        scan, pro = flatten_obs(session["obs"])
        with torch.no_grad():
            mu = session["act_fn"](scan, pro).clone()
            if session["cfg"].stochastic:
                act = (mu + model.actor.log_std.exp() * torch.randn_like(mu)).clamp(-1, 1)
            else:
                act = mu
        if link is not None and link.mode == "drive":
            # The policy still produces an action (its observation history has to stay real),
            # but car 0 drives what `/drive` said; silence past the timeout is speed 0.
            steer, speed, _fresh = link.command()
            env.set_external_command(link.car, steer, speed)
        session["obs"], _rew, _term, _trunc, _info = env.step(act)
        if rt is not None:
            rt.post_step(_term, _trunc)             # issued command + episode boundaries
        session["k"] += 1
        if link is not None:
            plan_ref = None
            if session["mode"] == "plan" and env.tracker is not None and env.tracker.last_ref is not None \
                    and link.mode != "drive":
                plan_ref = env.tracker.last_ref[link.car]
            try:
                link.publish(env.last_result, env, plan_ref)
            except Exception as exc:
                # A ROS-side failure must not stop the simulation on screen; say it once.
                if not session.get("ros_failed"):
                    session["ros_failed"] = True
                    self.say(P.MSG_LOG, gen=self.gen, text=f"ROS2 발행 실패(이후 반복 생략): {exc}")

    def _reload_if_changed(self, session) -> None:
        """Pick up a newer checkpoint of the same run without restarting the session.

        A newer file can have a different observation layout, so the same compatibility check the
        session was built with runs again; failing it keeps the model that is already loaded.
        """
        if time.time() - session["last_reload"] < 5.0:
            return
        session["last_reload"] = time.time()
        try:
            mt = os.path.getmtime(session["ckpt_path"])
        except OSError:
            return
        if mt == session["mtime"]:
            return
        try:
            from ..learn import common
            from ..learn.model import load_checkpoint
            from ..learn.watch import Introspector, actor_runner, describe_checkpoint_line
            model, extra = load_checkpoint(session["ckpt_path"], session["device"],
                                           allow_controller=session.get("controller") is not None)
            model.eval()
            env_spec = common.obs_spec(session["env"])
            # The environment is already built, so a newer checkpoint with different normalizers
            # cannot be accommodated without a rebuild -- and a rebuild is the console's decision,
            # not a thing to do underneath a running session.
            problems = (obs_spec_problems(model.meta, env_spec, extra.get("skipped") or [],
                                          session["env"].act_dim)
                        + normalizer_problems(extra.get("spec") or {}, env_spec))
            if problems:
                self.say(P.MSG_LOG, gen=self.gen,
                         text=("새 체크포인트의 관측 구성이 이 세션과 달라 반영하지 않았습니다 ("
                               + "; ".join(problems) + "). 이전 체크포인트로 계속합니다."))
                session["mtime"] = mt          # do not re-read the same incompatible file
                return
            session.update(model=model, extra=extra, mtime=mt,
                           intro=Introspector(model),
                           info_line=describe_checkpoint_line(extra, session["ckpt_path"]),
                           act_fn=actor_runner(model, session["device"], session["compile"]))
            self._sal_cache = None
            self.say(P.MSG_LOG, gen=self.gen, text=f"체크포인트 갱신을 반영했습니다: {session['info_line']}")
        except Exception as exc:
            self.say(P.MSG_LOG, gen=self.gen, text=f"체크포인트 재로드 실패(이전 것을 계속 씁니다): {exc}")

    def _snapshot(self, session) -> dict:
        """One device->host gather for everything the console needs to draw a frame.

        Deliberately one transfer. Reading scalars one at a time with `.item()` drains the whole
        device queue each time, which next to a training job is the difference between 20 ms and
        125 ms a step -- and that cost is a wait, not work.

        Everything returned owns its memory (`_copy`), so a frame sitting in the outbound slot is a
        snapshot of the moment it was taken and not a window onto the simulator's live tensors.
        """
        import torch
        env = session["env"]
        sim = env.sim
        cfg: P.SessionConfig = session["cfg"]
        r = env.last_result
        n = int(min(sim.B, int(cfg.max_render_cars)))
        sel = torch.arange(n, device=sim.device)
        focus = int(session["focus"])
        focus = focus if 0 <= focus < n else 0        # the focus car is always one that is drawn

        Pk = ("mount_x", "mount_y", "mount_z", "mount_yaw", "mount_roll", "mount_pitch")
        pack = torch.cat([r.state[sel], r.attitude[sel], r.lap[sel, None].float(),
                          r.collision[sel, None].float(), r.s[sel, None], r.wall_dist[sel, None],
                          sim.car_rear[sel], sim.car_dims[sel, 0:1]], 1)
        nb = int(r.scan.shape[1])
        dash = torch.cat([sim.state[focus, [3, 6]], env.last_cmd[focus],
                          env.speed_cap[focus:focus + 1], sim.ay[focus:focus + 1],
                          sim.ax[focus:focus + 1], sim.P["mu"][focus:focus + 1] * 9.81])
        pieces = [pack.reshape(-1), r.scan[focus], r.scan_type[focus].float(),
                  torch.stack([sim.P[k][focus] for k in Pk]), dash]

        # The IMU produces a whole number of samples per control step only by coincidence: the
        # sampler is driven by imu_rate against the substep clock, so K varies step to step and is
        # 0 on a step that straddles no sample instant. Read the width now, average the samples
        # that exist, and send nothing at all when there are none -- never index a fixed K.
        k_imu = int(r.imu.shape[1]) if getattr(r, "imu", None) is not None else 0
        if k_imu > 0:
            pieces.append(r.imu[focus].mean(0))

        plan = pose = plan_pred = None
        if session["mode"] == "plan" and self._overlay.get("plan", True) and env.tracker is not None \
                and env.tracker.last_ref is not None:
            plan = env.tracker.last_ref[focus]
            pose = sim.state[focus, :3]
            pieces += [plan.reshape(-1), pose]
            if env.tracker.last_pred is not None:
                plan_pred = env.tracker.last_pred[focus]
                pieces.append(plan_pred.reshape(-1))
        flat = torch.cat(pieces).cpu().numpy()

        i = 0
        take = int(np.prod(pack.shape))
        p = flat[i:i + take].reshape(pack.shape); i += take
        scan = flat[i:i + nb]; i += nb
        styp = flat[i:i + nb]; i += nb
        pv = flat[i:i + len(Pk)]; i += len(Pk)
        dash_v = flat[i:i + 8]; i += 8
        imu_mean = None
        if k_imu > 0:
            imu_mean = _copy(flat[i:i + 6]); i += 6
        plan_w = plan_pred_w = None
        if plan is not None:
            k_, c_ = plan.shape
            ref = flat[i:i + k_ * c_].reshape(k_, c_); i += k_ * c_
            ps = flat[i:i + 3]; i += 3
            cs, sn = np.cos(ps[2]), np.sin(ps[2])
            plan_w = np.stack([ps[0] + ref[:, 0] * cs - ref[:, 1] * sn,
                               ps[1] + ref[:, 0] * sn + ref[:, 1] * cs, ref[:, 3]], 1).astype(np.float32)
            if plan_pred is not None:
                k2, c2 = plan_pred.shape
                rp = flat[i:i + k2 * c2].reshape(k2, c2); i += k2 * c2
                plan_pred_w = np.stack([ps[0] + rp[:, 0] * cs - rp[:, 1] * sn,
                                        ps[1] + rp[:, 0] * sn + rp[:, 1] * cs, rp[:, 3]], 1).astype(np.float32)

        session["gg"].append((float(dash_v[5]), float(dash_v[6])))
        ids = np.arange(n, dtype=np.int32)
        fr = {
            "gen": self.gen, "seq": self.seq, "t": float(r.t), "n": n,
            "x": _copy(p[:, 0]), "y": _copy(p[:, 1]), "yaw": _copy(p[:, 2]), "vx": _copy(p[:, 3]),
            "steer": _copy(p[:, 6]), "roll": _copy(p[:, 7]), "pitch": _copy(p[:, 8]),
            "lap": _copy(p[:, 9]), "coll": _copy(p[:, 10]), "s": _copy(p[:, 11]),
            "wall": _copy(p[:, 12]), "rear": _copy(p[:, 13:17]), "len": _copy(p[:, 17]),
            "scan": _copy(scan).astype(np.float32), "scan_type": _copy(styp).astype(np.int32),
            "focus": focus, "focus_env": focus, "ids": ids,
            "P": {k: float(pv[j]) for j, k in enumerate(Pk)},
            "control_dt": float(sim.control_dt),
            "sim_rate": float(self._sim_rate),
            # `actor_runner` swaps to eager on a RuntimeError and carries on. Carrying on is right;
            # letting the console keep calling it a compiled run is not.
            "actor_fell_back": bool(getattr(session.get("act_fn"), "fell_back_to_eager", False)),
            # The physics has its own guard (`Simulator._guarded`), which replaces the instance
            # attribute with the eager function for good. That substitution is what is checked
            # here -- the compiled path is a closure, the fallback is the bound method itself.
            "physics_fell_back": _physics_fell_back(session),
            "mu": float(dash_v[7]) / 9.81,
            # [v, v_cmd, v_cap, steer, steer_cmd, v_max, mu_g] -- the console unpacks this order and
            # takes `gg` from its own key (DashPanel's argument list interleaves them).
            "dash": [float(dash_v[0]), float(dash_v[3]), float(dash_v[4]), float(dash_v[1]),
                     float(dash_v[2]), float(env.ecfg.v_max_policy), float(dash_v[7])],
            "gg": list(session["gg"]),
            # The g-g trace is the simulator's body-frame acceleration, not the modelled
            # accelerometer: `sim.ax/ay` exist every step, where the IMU yields a varying number of
            # samples and sometimes none. `imu` below is the sensor, when it fired.
            "gg_source": "sim",
            "plan": plan_w, "plan_pred": plan_pred_w,
            "info_line": session["info_line"],
            "ckpt_age_s": max(0.0, time.time() - session["mtime"]),
        }
        if imu_mean is not None:
            fr["imu"] = imu_mean.astype(np.float32)     # (6,) gyro xyz, accel xyz, mean of k_imu
            fr["imu_samples"] = k_imu
        if sim.other_idx is not None:
            rivals = set(sim.other_idx[focus].tolist())
            fr["opponent"] = np.array([int(e) in rivals for e in ids], bool)
        if self._overlay.get("saliency"):
            sal, age = self._saliency(session)
            if sal is not None:
                fr["saliency"], fr["saliency_age"] = sal, age
        if self._overlay.get("internals"):
            hidden, stem = self._internals(session)
            if hidden is not None:
                fr["hidden"], fr["stem"] = hidden, stem
        self.seq += 1
        return fr

    def _saliency(self, session) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """|d action / d beam| for the focus car, throttled.

        An autograd backward pass per frame is not free, so it runs at about 2 Hz and the frames in
        between reuse the last map -- and the console is told how old it is rather than being left
        to assume it is current.
        """
        from ..learn.obs import flatten_obs
        now = _now()
        if self._sal_cache is not None and now - self._sal_time < 0.5:
            return self._sal_cache, now - self._sal_time
        scan, pro = flatten_obs(session["obs"])
        f = int(session["focus"])
        try:
            sal, _mu = session["intro"].saliency(scan, pro, f if f < scan.shape[0] else 0)
        except Exception:
            if self._sal_cache is None:
                return None, None
            return self._sal_cache, now - self._sal_time
        self._sal_cache = _copy(sal).astype(np.float32)
        self._sal_time = now
        return self._sal_cache, 0.0

    def _internals(self, session) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        h = session["intro"].h
        if "hidden" not in h or "stem" not in h:
            return None, None
        return (_copy(h["hidden"][0].detach().float().cpu().numpy()),
                _copy(h["stem"][0].detach().float().cpu().numpy()))

    # ---------------------------------------------------------------- sim-thread commands
    def _enqueue(self, msg: dict) -> None:
        with self._pending_lock:
            self._pending.append(msg)

    def _take_pending(self, only_pause: bool = False) -> List[dict]:
        with self._pending_lock:
            if only_pause:
                out = [m for m in self._pending if m.get("kind") == P.CMD_PAUSE]
                if out:
                    kept = [m for m in self._pending if m.get("kind") != P.CMD_PAUSE]
                    self._pending.clear()
                    self._pending.extend(kept)
                return out
            out = list(self._pending)
            self._pending.clear()
        return out

    def _apply_on_sim_thread(self, session, msg: dict) -> bool:
        """Apply one queued command to the live session and ack it. Returns True if the pacing
        anchor should be reset (the simulation's relationship to wall time changed)."""
        kind = msg.get("kind")
        seq = msg.get("seq")
        reanchor = False
        if kind == P.CMD_PAUSE:
            self.paused = bool(msg.get("paused"))
            reanchor = True
            self.say(P.MSG_ACK, seq=seq, command="pause", gen=self.gen,
                     state={"paused": self.paused, "last_seq": self.seq - 1,
                            "t": float(session["env"].sim.t)})
        elif kind == P.CMD_SET_MU:
            mode = str(msg.get("mode", "fixed"))
            mu = float(msg.get("mu", 1.0489))
            pin = session.get("mu_pin")
            if mode == "fixed":
                if pin is None:
                    pin = MuPin(session["env"], mu).install()
                    session["mu_pin"] = pin
                pin.set(mu)                       # every car, now; and every reset from here on
            elif pin is not None:
                pin.release()                     # cars keep their value until their next reset
                session["mu_pin"] = None
            self.say(P.MSG_ACK, seq=seq, command="set_mu", gen=self.gen,
                     state={"mu_mode": mode, "mu": mu, "last_seq": self.seq - 1,
                            "t": float(session["env"].sim.t)})
        elif kind == P.CMD_RESET:
            session["obs"], _ = session["env"].reset()
            if session.get("controller") is not None:
                session["controller"].begin(session["obs"])
            session["gg"].clear()
            self._sal_cache = None
            reanchor = True
            self.say(P.MSG_ACK, seq=seq, command="reset", gen=self.gen,
                     state={"last_seq": self.seq - 1, "t": float(session["env"].sim.t)})
        elif kind == P.CMD_FOCUS:
            shown = min(int(session["env"].sim.B), int(session["cfg"].max_render_cars))
            session["focus"] = max(0, min(shown - 1, int(msg.get("env", 0))))
            session["gg"].clear()
            self._sal_cache = None
            self.say(P.MSG_ACK, seq=seq, command="focus", gen=self.gen,
                     state={"env": session["focus"]})
        elif kind == P.CMD_OVERLAY:
            self._overlay.update({k: bool(v) for k, v in (msg.get("overlay") or {}).items()})
            if not self._overlay.get("saliency"):
                self._sal_cache = None
            self.say(P.MSG_ACK, seq=seq, command="overlay", gen=self.gen, state=dict(self._overlay))
        return reanchor

    # ---------------------------------------------------------------- training courtesy flag
    def _heartbeat(self) -> None:
        """"A viewer is on screen, leave me a slice of the GPU" -- the existing contract training
        jobs already honour (`common.viewer_active`)."""
        from ..learn import common
        common.viewer_heartbeat()
        self._heartbeat_raised = True

    def _heartbeat_clear(self) -> None:
        """Lower the flag, but only if this worker is the one that raised it.

        The flag is a single global file, so a worker that clears it unconditionally can hand a
        training job back the GPU while a *different* viewer is still watching.
        """
        if not self._heartbeat_raised:
            return
        self._heartbeat_raised = False
        try:
            from ..learn import common
            common.viewer_gone()
        except Exception:
            pass

    # ---------------------------------------------------------------- the loop
    def _sim_loop(self, session, gen: int) -> None:
        """Real-time paced stepping. The only thread that touches this session."""
        t0 = _now()
        sim_t0 = session["env"].sim.t
        rate_t, rate_s = t0, sim_t0
        n = 0
        hb = 0.0
        try:
            fastpath = session.get("fastpath")
            if fastpath is not None:
                # The graphs were captured on the control thread inside `build_session`; this is the
                # thread that will replay them, and the only one. Transfer before the first step,
                # with the device synchronise that makes the handover real.
                fastpath.adopt()
            ctrl = session.get("controller")
            if ctrl is not None:
                ctrl.adopt()                       # the grip solver graph, same rule as above
            while self.alive and self.gen == gen and self.running:
                # While a new session is being built, only `pause` is applied here: it is the one
                # command that touches no tensor. Everything else waits, because the control
                # thread is inside torch (see `_build_hold`).
                for msg in self._take_pending(only_pause=self._hold):
                    if self._apply_on_sim_thread(session, msg):
                        t0, sim_t0 = _now(), session["env"].sim.t
                if not (self.alive and self.gen == gen and self.running):
                    break
                if self.paused or self._hold:
                    # Nothing steps, so sim time cannot move -- which is what the pause ack claimed.
                    if self._hold:
                        self._hold_parked.set()
                    time.sleep(0.01)
                    t0, sim_t0 = _now(), session["env"].sim.t
                    continue
                self._hold_parked.clear()
                self._step_once(session)
                self._reload_if_changed(session)
                n += 1
                now = _now()
                # A fixed *wall-time* window, not a fixed step count. Averaging over N steps gives
                # every window the same weight regardless of how long it took, so a slow window
                # counts for as little as a fast one and the reported rate sits above the rate the
                # session is actually achieving. Measured on an RTX 4060 Ti with four cars: 0.34
                # reported against 0.26 observed over the same span. Time-weighting removes it.
                if now - rate_t >= RATE_WINDOW_S:
                    self._sim_rate = (session["env"].sim.t - rate_s) / (now - rate_t)
                    rate_t, rate_s = now, session["env"].sim.t
                if now - self._last_send >= self._min_send_dt:
                    self._last_send = now
                    self.emit_frame(self._snapshot(session), gen)
                if now - hb > 1.0:              # tell a training job a viewer is on screen
                    hb = now
                    self._heartbeat()
                lead = (session["env"].sim.t - sim_t0) - (_now() - t0)
                if lead > 0:
                    time.sleep(min(lead, 0.05))
                elif lead < -0.5:
                    t0, sim_t0 = _now(), session["env"].sim.t     # cannot keep up: re-anchor
        except Exception as exc:
            if self.gen != gen:
                return                       # this generation was already being replaced
            self.running = False
            if self.prepare_gen is not None:
                # A generation the console has already moved on from. Reporting it as a failure
                # would put an error card on screen for a session that is being replaced anyway.
                self.say(P.MSG_LOG, gen=gen,
                         text=f"교체 중이던 세션 #{gen} 이 중단됐습니다(무시): {exc}")
                return
            self.say(P.MSG_ERROR, gen=gen, where="시뮬레이션",
                     message=f"시뮬레이션이 중단됐습니다: {exc}",
                     detail=traceback.format_exc(), retryable=True)
            self.say(P.MSG_STATE, gen=gen, state=P.STATE_FAILED)

    # ================================================================ commands
    def handle(self, msg: dict, preparing: Optional[int] = None) -> None:
        """Dispatch one control message.

        `preparing` is the generation currently being built; in that mode the only thing that may
        start a new build is the queue, and a `start` implicitly cancels the build in flight
        because the console has already moved on.
        """
        kind = msg.get("kind")
        seq = msg.get("seq")

        if kind == P.CMD_HELLO:
            import torch
            from ..learn import common
            self.say(P.MSG_HELLO, seq=seq, pid=os.getpid(), monotonic=_now(),
                     torch=torch.__version__, cuda=bool(torch.cuda.is_available()),
                     device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                     runs_dir=common.RUNS_DIR, protocol=P.PROTOCOL_VERSION)

        elif kind == P.CMD_LIST_MAPS:
            try:
                self.say(P.MSG_MAPS, seq=seq, groups=self.map_catalog())
            except Exception as exc:
                self.say(P.MSG_MAPS, seq=seq, groups={},
                         error=f"맵 목록을 읽지 못했습니다: {type(exc).__name__}: {exc}")

        elif kind == P.CMD_DESCRIBE:
            run = msg.get("run", "latest")
            try:
                self.say(P.MSG_DESCRIBED, seq=seq, ok=True, run=run,
                         info=self.describe_checkpoint(run))
            except Exception as exc:
                self.say(P.MSG_DESCRIBED, seq=seq, ok=False, run=run, error=str(exc) or type(exc).__name__)

        elif kind == P.CMD_CANCEL:
            gen = int(msg.get("gen", -1))
            self.cancel_gen = gen
            self.say(P.MSG_ACK, seq=seq, command="cancel", gen=gen,
                     state={"preparing": self.prepare_gen})

        elif kind in (P.CMD_PAUSE, P.CMD_RESET, P.CMD_FOCUS, P.CMD_OVERLAY, P.CMD_SET_MU):
            # These act on a live session. If one is running, its own thread applies them and acks
            # from there, so the ack means "done" rather than "heard".
            if self._sim_thread is not None and self._sim_thread.is_alive():
                self._enqueue(msg)
            else:
                self._ack_without_session(msg)

        elif kind == P.CMD_STOP:
            if preparing is not None:
                self.cancel_gen = preparing
            gen = self.gen
            self.teardown()
            self.say(P.MSG_ACK, seq=seq, command="stop", gen=gen, state={"state": P.STATE_IDLE})
            self.say(P.MSG_STOPPED, gen=gen)
            self.gen = -1

        elif kind == P.CMD_SHUTDOWN:
            self._shutdown_requested = True
            if preparing is not None:
                self.cancel_gen = preparing
                return                       # unwound by _check_cancel, finished by serve()
            raise WorkerExit()

        elif kind == P.CMD_START:
            if preparing is not None:
                self._queued_start = msg     # the outer loop runs it
                self.cancel_gen = preparing  # the console has moved on; stop building the old one
                return
            self.start(msg)

        else:
            self.say(P.MSG_LOG, gen=self.gen, text=f"알 수 없는 명령을 무시했습니다: {kind!r}")

    def _ack_without_session(self, msg: dict) -> None:
        """Ack a live command that arrived with nothing running.

        The state reported is what is actually true now -- not what the command asked for. A focus
        or overlay choice is remembered so the next session starts with it.
        """
        kind, seq = msg.get("kind"), msg.get("seq")
        if kind == P.CMD_PAUSE:
            self.paused = bool(msg.get("paused"))
            state = {"paused": self.paused, "last_seq": self.seq - 1, "session": False}
        elif kind == P.CMD_FOCUS:
            self._focus = max(0, int(msg.get("env", 0)))
            state = {"env": self._focus, "session": False}
        elif kind == P.CMD_OVERLAY:
            self._overlay.update({k: bool(v) for k, v in (msg.get("overlay") or {}).items()})
            state = dict(self._overlay, session=False)
        else:                                     # reset
            state = {"session": False}
        self.say(P.MSG_ACK, seq=seq, command=str(kind), gen=self.gen, state=state)

    # ---------------------------------------------------------------- start / swap
    def start(self, msg: dict) -> None:
        """Build a new generation, then swap it in.

        Build-then-swap, not stop-then-build: the session already on screen keeps running while the
        next one is prepared, so cancelling costs nothing and a failed start leaves the user
        looking at the map they had rather than at an empty viewport.
        """
        try:
            gen = int(msg["gen"])
            cfg = P.SessionConfig.from_dict(msg.get("config") or {})
            validate_start_config(cfg)
        except Exception as exc:
            gen = int(msg.get("gen", -1))
            self.say(P.MSG_ERROR, gen=gen, where="설정", message=str(exc) or type(exc).__name__,
                     detail=traceback.format_exc(), retryable=True)
            return
        if gen <= self.gen:
            self.say(P.MSG_ERROR, gen=gen, where="설정",
                     message=f"세션 번호가 되돌아갔습니다 (받은 {gen}, 현재 {self.gen}). "
                             f"세션 번호는 콘솔이 발급하며 재사용되지 않아야 합니다.",
                     detail="", retryable=False)
            return

        # Imported here, not at module scope: the console process imports this module to find the
        # worker entry point, and `graph_fastpath` imports torch. By the time a build runs, torch is
        # loaded in this process anyway. It has to be in scope for the `except` clause below --
        # naming it there without importing it turned a capture failure into a bare NameError that
        # hid the real one.
        from .graph_fastpath import CaptureFailed
        self.prepare_gen = gen
        self.cancel_gen = None
        session = None
        self._build_hold(True)
        try:
            session = self._build_with_oom_retry(cfg, gen)
        except Cancelled:
            self.prepare_gen = None
            self._abandon_build()
            self.say(P.MSG_STOPPED, gen=gen, cancelled=True)
            return
        except CaptureFailed as exc:
            # Fatal to the process, not just to this build. A failed CUDA-graph capture leaves the
            # context in a state where `torch.randn(device="cuda")` raises while ordinary arithmetic
            # still works, so *any* later session in this process would run its physics and throw on
            # its sensor noise. Reporting an error and staying alive would hand the next retry that
            # context; the console spawns a fresh worker when it finds this one gone, so stopping
            # here is what makes the retry clean.
            self.prepare_gen = None
            # Order matters: stop first, tidy second. `_abandon_build` lifts the build hold, and a
            # hold that lifts while the loop is still alive lets the outgoing session take one more
            # step -- a step that draws sensor noise from the generator this capture just poisoned.
            self.alive = False          # `serve` breaks, tears down and says goodbye
            self.running = False
            try:
                self._abandon_build()
            except Exception:
                pass                    # the context is already suspect; exiting still has to work
            self.say(P.MSG_ERROR, gen=gen, where="CUDA 그래프",
                     message=f"CUDA 그래프 캡처에 실패해 이 worker 를 종료합니다. 다시 시도하면 "
                             f"새 worker 로 시작합니다 — {exc}",
                     detail=traceback.format_exc(), retryable=True)
            return
        except Exception as exc:
            self.prepare_gen = None
            self._abandon_build()
            self.say(P.MSG_ERROR, gen=gen, where=where_of(exc),
                     message=str(exc) or type(exc).__name__,
                     detail=traceback.format_exc(), retryable=True)
            return
        self.prepare_gen = None
        try:
            self._swap_in(session, cfg, gen)
        except Exception:
            # `_swap_in` failed, so this session was never handed over and nothing else will free
            # it. Its fast path still holds `sim._roll`, which is the reference cycle that would
            # keep the whole env resident.
            ctrl = session.pop("controller", None)
            if ctrl is not None:
                try:
                    ctrl.release()
                except Exception:
                    pass
            fp = session.pop("fastpath", None)
            if fp is not None:
                try:
                    fp.release()
                except Exception:
                    pass
            raise
        finally:
            self._build_hold(False)

    def _abandon_build(self) -> None:
        """A replacement that was cancelled or failed. Stop the session it was replacing.

        The held session is still loaded and would resume the moment the hold lifts, but the
        console has already left it: `start_session` adopts the new generation and clears the old
        geometry, and `stopped` / `error` put the window in IDLE or FAILED. Resuming it would leave
        a simulation stepping on the GPU that nothing on screen represents and nothing can stop --
        the console would have to be told to stop a session it believes is not running.

        Rolling the console back to the old view would be the other coherent answer, but it is not
        what this version does, so the honest thing is to make IDLE / FAILED true.
        """
        self._build_hold(False)
        if self.session is not None:
            self.say(P.MSG_LOG, gen=self.gen,
                     text="교체가 취소·실패해 이전 세션도 함께 정리합니다 "
                          "(화면이 이미 그 세션을 떠났기 때문입니다).")
        self.teardown()
        self.gen = -1

    def _build_hold(self, on: bool) -> bool:
        """Stop / restart the outgoing session's stepping for the duration of a build.

        Not an optimisation -- a correctness requirement. `torch.compile(mode="reduce-overhead")`
        captures CUDA graphs on this thread, and capture puts the default CUDA generator into a
        mode where any *other* thread drawing from it raises
        `Offset increment outside graph capture encountered unexpectedly`. That is exactly what a
        still-running simulation does on every step: `lidar.scan` calls `torch.randn_like` for the
        range noise. Observed on an RTX 4060 Ti, torch 2.10, on the second map switch of a session.

        The old session is only *held*, never torn down, so the map stays on screen, cancelling
        costs nothing, and resuming it is a flag flip. Held both on CUDA and CPU: two environments
        stepping in one process while a third thing builds is not a state worth having two variants
        of.

        Returns whether the outgoing session is known to have stopped. Capture is only safe on a
        `True`; a timeout is not "probably fine", it is "another thread may still be inside a step",
        which is the one condition this hold exists to rule out.
        """
        if not on:
            self._hold = False
            self._hold_parked_ok = False
            return False
        self._hold_parked.clear()
        self._hold = True
        thread = self._sim_thread
        if thread is None or not thread.is_alive():
            self._hold_parked_ok = True          # nothing is stepping: parked by construction
            return True
        # Wait for it to actually park, so the graph capture that follows does not overlap the step
        # it is already inside. A step is ~25 ms compiled and ~90 ms eager; the timeout is a
        # backstop, and reaching it means the answer is "no", not "carry on".
        self._hold_parked_ok = bool(self._hold_parked.wait(timeout=HOLD_PARK_TIMEOUT_S))
        return self._hold_parked_ok

    def _build_with_oom_retry(self, cfg: P.SessionConfig, gen: int) -> dict:
        """Build, and if the device runs out of memory because the previous session is still
        resident, free that one and try once more. Keeping the old session alive is worth a retry,
        not worth failing the start the user asked for."""
        from .graph_fastpath import CaptureFailed
        try:
            return self.build_session(cfg, gen)
        except CaptureFailed:
            # Never retried, even when the message says "out of memory". A capture that ran out of
            # memory still left the context poisoned: `torch.randn(device="cuda")` raises from it
            # while ordinary arithmetic keeps working. Building a second session on that context
            # would produce a simulator whose physics runs and whose sensor noise throws, which is
            # worse than failing. The worker restarts instead.
            raise
        except Exception as exc:
            if self.session is None or "out of memory" not in str(exc).lower():
                raise
            self.say(P.MSG_LOG, gen=gen,
                     text="메모리가 부족해 이전 세션을 먼저 정리하고 다시 시도합니다.")
            self.teardown()
            return self.build_session(cfg, gen)

    def _swap_in(self, session: dict, cfg: P.SessionConfig, gen: int) -> None:
        old = self.session
        self._stop_sim_thread()
        # The outgoing thread is gone and the build is finished, so nothing is left to protect.
        self._build_hold(False)
        self._drain_slot()          # nothing from the previous generation after the new geometry
        self.session = session
        self._release_session(old)
        self.gen = gen
        self.seq = 0                # the console resets its own accounting per generation
        self.paused = False
        self._sim_rate = 0.0
        self._sal_cache = None
        self._last_send = 0.0
        self.cfg = cfg
        self._overlay.update({"saliency": bool(cfg.saliency), "internals": bool(cfg.internals)})
        shown = min(int(session["env"].sim.B), int(cfg.max_render_cars))
        session["focus"] = max(0, min(shown - 1, int(self._focus)))
        # Commands aimed at the session that just went away. They still need an ack, or the console
        # leaves that button drawn as "적용 중" for the rest of the session.
        for msg in self._take_pending():
            self.say(P.MSG_ACK, seq=msg.get("seq"), command=str(msg.get("kind")), gen=gen,
                     state={"superseded": True, "paused": False, "env": session["focus"]})
        self.say(P.MSG_GEOMETRY, gen=gen, geometry=session["geometry"])
        self._mark_mem("ready")
        self.say(P.MSG_READY, gen=gen, facts=self.session_facts(session, gen))
        self.running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, args=(session, gen),
                                            name=f"sim-gen{gen}", daemon=True)
        self._sim_thread.start()

    # ---------------------------------------------------------------- teardown
    def _stop_sim_thread(self) -> bool:
        """Ask the simulation thread to finish and wait for it. False means it did not."""
        self.running = False
        thread, self._sim_thread = self._sim_thread, None
        if thread is None or not thread.is_alive():
            return True
        thread.join(SIM_JOIN_TIMEOUT)
        if thread.is_alive():
            self.say(P.MSG_LOG, gen=self.gen,
                     text=f"시뮬레이션 스레드가 {SIM_JOIN_TIMEOUT:.0f}초 안에 멈추지 않아 "
                          f"그 세션의 자원은 회수하지 않고 남겨 둡니다 (안전을 위해).")
            return False
        return True

    def _release_session(self, session: Optional[dict], joined: bool = True) -> None:
        """Drop a session's tensors. Only ever called for a session whose thread has stopped --
        freeing an environment a live thread is stepping is how a worker turns into a segfault
        instead of an error message."""
        if session is None:
            return
        if not joined:
            self._stuck.append(session)      # keep it alive under the thread still using it
            return
        try:
            device = session.get("device")
            cuda = device is not None and getattr(device, "type", "") == "cuda"
            if cuda:
                # `self.gen` is still the outgoing generation here: `_swap_in` releases the old
                # session before advancing it, and `teardown` releases the current one.
                self._mark_mem("before_release")
            # First, because it is what puts `sim._roll` and `tracker._solver` back. Leaving the
            # dispatcher installed would fail one step *after* the teardown, reading as an
            # unrelated crash, and its `sim -> _roll -> fastpath -> sim` cycle is what would keep
            # this env alive past the `gc.collect()` below.
            # The controller first: it wraps the solver the fast path installed, and restoring
            # them in the wrong order would leave the grip solver hooked to a released graph.
            link = session.pop("ros", None)
            if link is not None:
                try:
                    link.close()               # stops publishing before the tensors it reads go away
                except Exception:
                    pass
            ctrl = session.pop("controller", None)
            if ctrl is not None:
                try:
                    ctrl.release()
                except Exception:
                    pass
                del ctrl
            pin = session.pop("mu_pin", None)
            if pin is not None:
                try:
                    pin.release()
                except Exception:
                    pass
            fastpath = session.pop("fastpath", None)
            if fastpath is not None:
                fastpath.release()
                del fastpath           # the local is a reference too, and gc.collect() is below
            for key in ("env", "model", "act_fn", "intro", "obs"):
                session.pop(key, None)
            session.clear()
            if cuda:
                import gc
                import torch
                t0 = time.perf_counter()
                torch.cuda.synchronize()
                # The dropped env is reachable from reference cycles (module <-> tensors), so the
                # refcount alone does not free it and `empty_cache` then has nothing to hand back.
                gc.collect()
                torch.cuda.empty_cache()
                # This runs on the control thread between two sessions -- the old one has stopped
                # and the new one has not been swapped in -- so it delays nothing that is running.
                # It is timed anyway: a collect over a large heap is not free, and a cost nobody
                # measures is a cost nobody notices growing.
                self._mark_mem("after_release", release_ms=round((time.perf_counter() - t0) * 1e3, 1))
        except Exception:
            pass

    def teardown(self) -> None:
        """Stop the session, keep the process. Touches nothing outside this process."""
        joined = self._stop_sim_thread()
        session, self.session = self.session, None
        self._release_session(session, joined=joined)
        self._drain_slot()
        with self._pending_lock:
            self._pending.clear()
        self.paused = False
        self._sim_rate = 0.0
        self._sal_cache = None
        self._heartbeat_clear()

    # ================================================================ main loop
    def serve(self) -> None:
        while self.alive:
            if self._shutdown_requested:
                break                       # asked to go while a build was in flight
            if self._queued_start is not None:
                msg, self._queued_start = self._queued_start, None
                self.start(msg)
                continue
            try:
                if not self.control.poll(0.1):
                    continue
                msg = self.control.recv()
            except (EOFError, OSError):
                break                       # console gone: tear down and exit, quietly
            try:
                self.handle(msg)
            except WorkerExit:
                break
            except Cancelled:
                continue
            except Exception as exc:
                self.say(P.MSG_ERROR, gen=self.gen, where="worker",
                         message=f"{type(exc).__name__}: {exc}",
                         detail=traceback.format_exc(), retryable=True)
        self.teardown()
        self._slot.close()
        self.say(P.MSG_BYE)
        self._sender.join(timeout=1.0)


def main(control, frames, log_path: Optional[str] = None) -> None:
    """Process entry point. Never returns except by shutting down.

    The endpoint check is not defensive noise: `Pipe(duplex=False)` returns `(readable, writable)`,
    so handing the worker the first of the pair is an easy mistake that would otherwise show up as
    a broken frame channel long after the session looked healthy.
    """
    if log_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            sys.stderr = open(log_path, "a", buffering=1)
            faulthandler.enable(sys.stderr)
        except OSError:
            pass
    if not getattr(frames, "writable", True):
        try:
            control.send(P.message(
                P.MSG_ERROR, gen=-1, where="worker", retryable=False,
                message="프레임 파이프의 쓰기 쪽이 아니라 읽기 쪽이 worker 에 전달됐습니다. "
                        "Pipe(duplex=False) 는 (읽기, 쓰기) 순서이므로 worker 는 두 번째를 받아야 합니다.",
                detail="frames.writable is False"))
            control.send(P.message(P.MSG_BYE))
        except Exception:
            pass
        return
    worker = None
    try:
        worker = SimWorker(control, frames)
        worker.serve()
    except KeyboardInterrupt:
        pass
    except Exception:
        try:
            control.send(P.message(P.MSG_ERROR, gen=-1, where="worker",
                                   message="worker 프로세스가 예외로 종료됐습니다.",
                                   detail=traceback.format_exc(), retryable=True))
        except Exception:
            pass
    finally:
        if worker is not None:
            try:
                worker._slot.close()
            except Exception:
                pass
        for conn in (control, frames):
            try:
                conn.close()
            except Exception:
                pass
