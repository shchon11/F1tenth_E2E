"""Explicit CUDA-graph fast path for the viewer's physics and plan solver.

The viewer's default is now *not* to `torch.compile`, because compiling costs minutes before
anything appears. Eager then plays at about 0.27x real time on one car with 1081 beams, which is
not a viewer. This captures two boundaries as CUDA graphs instead -- about a second, once -- and
replays them.

Scope, deliberately narrow:

* `Simulator._roll_physics`, one graph per IMU phase.
* `mpc.solve`, through `PlanTracker._solver`, this session's tracker only.
* `Simulator._prop_contact` and `Lidar._merge_props`, the two pure prop leaves, on maps that have
  props. Both loop over the placed props in `prop_math`, so their launch cost grows with the prop
  count: with the first two boundaries graphed, Monza's eight props held the viewer at 0.76x while
  Korea's three reached 0.98x.
* `Actor.step` -- one control step of the policy, hidden state included -- through the standalone
  `graph_actor_step`. Opt-in and built by the caller during a session build (`learn.graph_runtime.
  prepare_actor_graph`), never lazily mid-step: a failed capture is fatal to the process, and the
  only place the worker is prepared for that is the build. A recurrent actor's hidden state is an
  ordinary tensor argument here, so it lands in a static buffer that each replay is copied into --
  the caller keeps one tensor and the graph updates it in place.

Nothing else. `Simulator.step` is never captured: it advances `self.t` and `_imu_phase`, rolls
`cmd_hist`, ORs `collided`, zeroes `prop_touched` and draws randomness -- Python effects that would
run once at capture and never again.

What makes this safe rather than fast-and-wrong:

* **Eligibility is decided before capture.** Soft walls, CPU, a non-float32 session, a batch the
  schedules do not cover -- all select eager up front and say so, rather than failing during capture.
* **A failed capture poisons the process.** After one, `torch.randn(..., device="cuda")` raises
  `Offset increment outside graph capture encountered unexpectedly` while ordinary arithmetic still
  works, so an eager fallback would give a simulator whose physics runs and whose sensor noise
  raises. `CaptureFailed` is therefore fatal to the session; the worker restarts. Nothing is
  restored on that path either -- see the RNG handling in `GraphedCallable`.
* **Ownership is transferred once, explicitly.** Capture happens on the control thread during the
  build; the simulation thread takes ownership before its first step, with a device synchronise and
  a check that it is the stream the graphs were captured on. The guard is never disabled.
* **Guards re-read live attributes** every call, so a rebound tensor is caught rather than replayed
  against a stale address, and every value the captured body reads at Python level -- the IMU phase
  schedule, `hist_len`, the hard-wall flag -- is part of the guard set.
* **The hooks it installs, it removes.** `release()` restores `sim._roll` and `tracker._solver`, so
  a teardown cannot leave a dispatcher pointing at graphs that no longer exist.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch


class NotCapturable(RuntimeError):
    """This configuration is not eligible. Decided before capture; the caller runs eager."""


class CaptureFailed(RuntimeError):
    """Capture was attempted and failed. The CUDA context is now suspect: do not continue."""


class GuardViolation(RuntimeError):
    """Something the graph captured by address has changed. Replaying would be wrong."""


_ATOM = (int, float, bool, str, bytes, type(None))


def _sig(t: torch.Tensor) -> tuple:
    return (int(t.data_ptr()), tuple(t.shape), t.dtype, t.device.type, t.device.index)


def _freeze(v) -> Any:
    """A snapshot of a value that a later mutation cannot follow.

    Keeping the caller's object and comparing it against itself always passes: a `PlanSpec` whose
    `N` was changed is still `==` to the reference we stored, because it *is* that object. So
    everything is reduced to immutable scalars here, recursively.

    Anything this does not understand raises rather than being silently accepted -- an unrecognised
    value would be baked into the graph by value and never checked again.
    """
    if torch.is_tensor(v):
        return _sig(v)
    if isinstance(v, _ATOM):
        return v
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return (type(v).__name__,) + tuple((f.name, _freeze(getattr(v, f.name)))
                                           for f in dataclasses.fields(v))
    if isinstance(v, torch.device):
        return (v.type, v.index)
    if isinstance(v, torch.dtype):
        return str(v)
    raise NotCapturable(f"cannot snapshot a {type(v).__name__} value; it would be baked into the "
                        f"graph without any guard on it")


def _dict_sig(d: Dict[str, Any]) -> tuple:
    """Keys *and* per-key value snapshots, as an ordered tuple.

    Snapshotted independently of the live dict. Holding a reference to the caller's dict and
    comparing it against itself passes unconditionally: rebinding a key or adding one mutates the
    very object the check reads. Non-tensor entries are frozen too rather than skipped -- `P` is all
    tensors today, and a scalar added to it later must not slip past the guard.
    """
    return tuple(sorted((k, _freeze(v)) for k, v in d.items()))


class GraphedCallable:
    """One captured graph with static input buffers, replayed under guards."""

    def __init__(self, fn: Callable, args: Sequence[Any], *, n_warmup: int = 3,
                 guards: Optional[Dict[str, Callable[[], Any]]] = None, name: str = ""):
        self.name = name or getattr(fn, "__name__", "fn")
        self._fn = fn
        self._guards = dict(guards or {})
        self._lock = threading.Lock()
        self._owner = threading.get_ident()
        self._stream = None
        self._adopted = False
        self._busy = False

        self._tensor_at = [i for i, a in enumerate(args) if torch.is_tensor(a)]
        self._dict_at = [i for i, a in enumerate(args) if isinstance(a, dict)]
        self._other = {i: _freeze(a) for i, a in enumerate(args)
                       if i not in self._tensor_at and i not in self._dict_at}
        self._static = [a.detach().clone() if torch.is_tensor(a) else a for a in args]
        self._dict_sigs = {i: _dict_sig(args[i]) for i in self._dict_at}
        self._guard_sigs = {k: self._read_guard(g) for k, g in self._guards.items()}

        rng = torch.cuda.get_rng_state_all()
        cpu_rng = torch.get_rng_state()
        # Pools are deliberately not shared between graphs. The phase graphs and the solver are not
        # replayed in capture order -- the solver runs first, resets reorder things -- and sharing a
        # pool is only sound when the replay order matches the capture order.
        self._capture_stream = torch.cuda.current_stream()
        try:
            side = torch.cuda.Stream()
            side.wait_stream(self._capture_stream)
            with torch.cuda.stream(side):
                for _ in range(n_warmup):
                    fn(*self._static)
            self._capture_stream.wait_stream(side)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._out = fn(*self._static)
        except Exception as exc:                       # noqa: BLE001 -- re-raised as fatal
            # No RNG restore here, deliberately. After a failed capture the CUDA generator is in the
            # poisoned state documented at the top of this module, and `set_rng_state_all` can raise
            # from it -- which would replace this exception and lose the real reason. The process is
            # being torn down anyway.
            raise CaptureFailed(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        # Success only. `Sim.warmup` rewinds the global RNG after its throw-away steps for the same
        # reason: warm-up must not consume the stream the session is about to draw from.
        torch.cuda.set_rng_state_all(rng)
        torch.set_rng_state(cpu_rng)

    # -- ownership ---------------------------------------------------------------
    def adopt(self) -> None:
        """Hand this graph to the calling thread. Once, synchronised, on the capture stream.

        The viewer captures on the control thread inside `build_session` and steps on a separate
        simulation thread created afterwards, so a plain owner check would reject every real step.
        Transferring is the honest fix; disabling the check is not -- the static buffers really are
        shared mutable state. The `synchronize()` is what makes the transfer safe: it is the only
        thing guaranteeing the capturing thread has no work left against these buffers.
        """
        with self._lock:
            if self._adopted:
                raise GuardViolation(f"{self.name}: ownership transfers exactly once, and this "
                                     f"graph has already been adopted")
            if self._busy:
                raise GuardViolation(f"{self.name}: cannot transfer while a replay is in flight")
            cur = torch.cuda.current_stream()
            if cur != self._capture_stream:
                raise GuardViolation(f"{self.name}: adopting thread is on {cur}, but the graph was "
                                     f"captured on {self._capture_stream}; the pool is ordered "
                                     f"against the capture stream")
            torch.cuda.synchronize()
            self._owner = threading.get_ident()
            self._stream = cur
            self._adopted = True

    def _read_guard(self, get) -> Any:
        return _freeze(get())

    def check_guards(self) -> None:
        for k, get in self._guards.items():
            now = self._read_guard(get)          # live read: never a cached reference
            if now != self._guard_sigs[k]:
                raise GuardViolation(f"{self.name}: {k} changed since capture "
                                     f"({self._guard_sigs[k]} -> {now})")

    def __call__(self, *args):
        if threading.get_ident() != self._owner:
            raise GuardViolation(
                f"{self.name}: replayed from a thread that does not own it. Static buffers are "
                f"shared mutable state; call adopt() from the stepping thread first.")
        if self._stream is not None and torch.cuda.current_stream() != self._stream:
            raise GuardViolation(f"{self.name}: current CUDA stream differs from the one this "
                                 f"graph was adopted on")
        if len(args) != len(self._static):
            raise GuardViolation(f"{self.name}: arity changed {len(self._static)} -> {len(args)}")
        try:
            self.check_guards()
            for i in self._dict_at:
                if _dict_sig(args[i]) != self._dict_sigs[i]:
                    raise GuardViolation(f"{self.name}: dict argument {i} was rebuilt, rekeyed or "
                                         f"revalued; the graph reads its captured addresses")
            for i, v in self._other.items():
                now = _freeze(args[i])
                if now != v:
                    raise GuardViolation(
                        f"{self.name}: non-tensor argument {i} changed {v} -> {now}")
        except NotCapturable as exc:     # an unfreezable value appeared after capture
            raise GuardViolation(f"{self.name}: {exc}") from exc
        with self._lock:
            self._busy = True
            try:
                for i in self._tensor_at:
                    src, dst = args[i], self._static[i]
                    if src.shape != dst.shape or src.dtype != dst.dtype or src.device != dst.device:
                        raise GuardViolation(
                            f"{self.name}: argument {i} is {tuple(src.shape)}/{src.dtype}/"
                            f"{src.device}, captured {tuple(dst.shape)}/{dst.dtype}/{dst.device}")
                    dst.copy_(src)
                self.graph.replay()
            finally:
                self._busy = False
        return _clone(self._out)


def _clone(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, tuple):
        return tuple(_clone(v) for v in x)
    if isinstance(x, list):
        return [_clone(v) for v in x]
    return x


# ---------------------------------------------------------------- eligibility
def _cuda_eligible(sim) -> Tuple[bool, str]:
    if sim.device.type != "cuda":
        return False, "CPU 세션"
    if getattr(sim.cfg.sim, "compile", False):
        return False, "torch.compile 이 켜져 있음"
    return True, ""


def roll_eligible(sim) -> Tuple[bool, str]:
    """Can this simulator's substep loop be captured? Decided before anything is attempted."""
    ok, why = _cuda_eligible(sim)
    if not ok:
        return False, why
    # Soft walls used to be refused here: `_resolve_wall_contact` rebound `self.prop_touched` to a
    # new tensor inside the loop, and a graph replays device work only, so the substep prop contacts
    # were silently dropped. It is an in-place `logical_or_` now, which a graph records like any
    # other write to a buffer it owns, and the early-out that skips the resolution when nothing is
    # touching is a host sync that `_capturing()` keeps out of the capture.
    n = int(getattr(sim, "imu_period", 0) or 0)
    if n <= 0 or len(getattr(sim, "imu_schedule", ()) or ()) != n:
        return False, f"IMU 스케줄이 예상과 다름 (period {n})"
    if sim.state.dtype != torch.float32:
        return False, f"float32 세션이 아님 ({sim.state.dtype})"
    return True, ""


def mpc_eligible(sim, tracker) -> Tuple[bool, str]:
    """The solver's own gate. Without it a CPU session would reach `CaptureFailed` -- fatal -- where
    `NotCapturable` -- run eager -- is the correct outcome."""
    ok, why = _cuda_eligible(sim)
    if not ok:
        return False, why
    if getattr(tracker, "compile_solver", False):
        return False, "tracker 가 이미 컴파일 솔버를 씀"
    # `build_ilqr_consts` matches the inline branch only at torch's default dtype: the inline code
    # builds `torch.zeros(6, 6, device=dev)` with no dtype at all. Anything else and the prebuilt
    # constants would carry a different precision through Vzz/Quu.
    if tracker.u_prev.dtype != torch.float32:
        return False, f"float32 tracker 가 아님 ({tracker.u_prev.dtype})"
    return True, ""


# ---------------------------------------------------------------- the actor's own step
def actor_eligible(actor, example_args: Sequence[Any]) -> Tuple[bool, str]:
    """Can this actor's control step be captured? Decided before anything is attempted.

    Separate from the simulator's gate because an actor has none of a simulator's Python-level
    effects: `Actor.step` is a pure function of (scan, proprio, hidden) plus its parameters, and a
    recurrent one returns its next hidden state rather than mutating anything. What it does need is
    a CUDA float32 session with training mode off and no torch.compile in the way.
    """
    if not example_args or not torch.is_tensor(example_args[0]):
        return False, "예시 입력이 텐서가 아님"
    dev = example_args[0].device
    if dev.type != "cuda":
        return False, "CPU 세션"
    if getattr(actor, "training", False):
        return False, "actor 가 train 모드임 (eval 로 두세요)"
    for i, t in enumerate(example_args):
        if t is None:
            continue
        if not torch.is_tensor(t):
            return False, f"인자 {i} 가 텐서가 아님"
        if t.dtype != torch.float32:
            return False, f"인자 {i} 가 float32 가 아님 ({t.dtype})"
        if t.device != dev:
            return False, f"인자 {i} 의 장치가 다름 ({t.device} vs {dev})"
    return True, ""


def graph_actor_step(actor, example_args: Sequence[Any], name: str = "actor.step") -> GraphedCallable:
    """Capture one control step of the actor, hidden state included.

    The hidden state is an ordinary tensor argument, which is exactly what makes this work: every
    argument becomes a STATIC buffer that each replay is copied into (`GraphedCallable.__call__`),
    so the caller keeps one hidden-state tensor for the life of the session and the graph updates
    it in place. Allocating a fresh hidden state per step -- or holding on to the tensor a replay
    produced -- is the way to get a graph that silently replays yesterday's state.

    Guarded on the actor's whole parameter list by address and shape, so a checkpoint reloaded into
    the same module object is caught rather than replayed against freed weights.

    Raises `NotCapturable` when the configuration is not eligible (the caller runs eager) and
    `CaptureFailed` if a capture is attempted and fails -- which, as everywhere in this module, is
    fatal to the process rather than something to retry.
    """
    ok, why = actor_eligible(actor, example_args)
    if not ok:
        raise NotCapturable(why)
    # Read LIVE, like every other guard in this module: closing over a captured `list(parameters())`
    # would compare that list against itself, and a checkpoint reloaded into the same module object
    # -- a rebound `Parameter`, the same names, new storage -- would replay against freed weights.
    guards = {
        "actor.parameters": lambda: tuple(_sig(t) for t in actor.parameters()),
        "actor.buffers": lambda: tuple(_sig(t) for t in actor.buffers()),
        "actor.training": lambda: bool(actor.training),
    }

    def step(scan, proprio, hidden):
        return actor.step(scan, proprio, None, hidden)

    return GraphedCallable(step, tuple(example_args), guards=guards, name=name)


class SimGraphFastPath:
    """Per-session holder: one graph per IMU phase, plus the plan solver's.

    Per instance and per session on purpose. A process-wide cache would keep graph memory pools
    alive across map changes and batch sizes for the lifetime of the worker; these are released
    with the session that made them.
    """

    def __init__(self, sim, log=None):
        self.sim = sim
        self._log = log or (lambda _t: None)
        self._by_phase: Dict[int, GraphedCallable] = {}
        self.mpc: Optional[GraphedCallable] = None
        self._consts = None
        self._tracker = None
        self._prev_roll = None
        self._prev_solver = None
        # The two pure prop leaves. Not part of `_roll_physics`: `_prop_contact` runs once per
        # control step outside the substep loop, and `_merge_props` runs inside the LiDAR trace.
        self._leaf_contact: Optional[GraphedCallable] = None
        self._leaf_merge: Optional[GraphedCallable] = None
        self._lidar = None
        self._shadows: list = []
        self._installed = False
        self._adopted = False

    # -- capture -----------------------------------------------------------------
    def _guards_for(self, sim) -> Dict[str, Callable[[], Any]]:
        """Everything `_roll_physics` reads at Python level, plus every address it bakes in.

        The Python-level reads matter as much as the tensors: `imu_on`, `soft_wall`, `hist_len`,
        `substeps` and `imu_ts` are read once at capture and become part of the graph's structure,
        so a later change to any of them would be replayed away silently.
        """
        g: Dict[str, Callable[[], Any]] = {
            "sim.tid": lambda: sim.tid,
            "sim.B": lambda: int(sim.B),
            "sim.car_dims": lambda: getattr(sim, "car_dims", None),
            "sim.corners": lambda: getattr(sim, "corners", None),
            "sim.imu_period": lambda: int(sim.imu_period),
            "sim.imu_schedule": lambda: tuple(tuple(s) for s in sim.imu_schedule),
            "sim.imu_ts": lambda: float(sim.imu_ts),
            "cfg.imu.enabled": lambda: bool(sim.cfg.imu.enabled),
            # `wheel_model` chooses which longitudinal model, which VESC feedback speed and whether
            # the IMU shock term runs -- all read at Python level inside `_roll_physics`, so they
            # are baked into the graph at capture and a later change would be replayed away.
            "sim.wheel_model": lambda: bool(getattr(sim, "wheel_model", False)),
            "cfg.sim.terminate_on_collision": lambda: bool(sim.cfg.sim.terminate_on_collision),
            "sim.hist_len": lambda: int(sim.hist_len),
            "sim.substeps": lambda: int(sim.substeps),
            "sim.control_dt": lambda: float(sim.control_dt),
            "sim.dt": lambda: float(sim.dt),
            # Read live: closing over the track object captured here would compare it against
            # itself, so a TrackTensors swap would never be seen.
            "sim.track": lambda: id(sim.track),
        }
        for attr in dir(sim.track):
            if attr.startswith("p_") and torch.is_tensor(getattr(sim.track, attr, None)):
                g[f"track.{attr}"] = (lambda a=attr: getattr(sim.track, a, None))
        return {k: v for k, v in g.items() if v() is not None}

    def capture_roll(self, example_args: Sequence[Any]) -> None:
        """Capture one graph per IMU phase, driving the phase explicitly through a full cycle."""
        sim = self.sim
        ok, why = roll_eligible(sim)
        if not ok:
            raise NotCapturable(why)
        guards = self._guards_for(sim)
        eager = sim._roll_physics
        period = int(sim.imu_period)
        saved_phase = int(sim._imu_phase)
        try:
            for phase in range(period):
                # `imu_idx` is a property returning `imu_schedule[_imu_phase]`, and the loop body
                # tests `k in self.imu_idx`. Without setting the phase here every graph would record
                # the phase that happened to be current, so three of the four would emit their IMU
                # sample at the wrong substep -- and phase 3, which emits two, would emit one.
                sim._imu_phase = phase
                self._by_phase[phase] = GraphedCallable(
                    eager, example_args, guards=guards, name=f"_roll_physics[phase {phase}]")
        finally:
            sim._imu_phase = saved_phase
        if len(self._by_phase) != period:
            raise CaptureFailed(f"IMU 위상 {period}개 중 {len(self._by_phase)}개만 캡처됨")
        self._log(f"물리 그래프 {len(self._by_phase)}개 캡처")

    def capture_mpc(self, tracker, example_args: Sequence[Any]) -> None:
        """Capture `mpc.solve` with this session's own constant tensors bound in.

        The constants are built here and held by this object, so they live and die with the
        session. `mpc.solve` keeps its default `consts=None` for every other caller.
        """
        from .. import mpc as _mpc
        ok, why = mpc_eligible(self.sim, tracker)
        if not ok:
            raise NotCapturable(why)
        z_like = example_args[0]
        self._consts = _mpc.build_ilqr_consts(tracker.spec, tracker.s_max, z_like.device)
        consts = self._consts

        def solve_bound(*args):
            return _mpc.solve(*args, consts=consts)

        self.mpc = GraphedCallable(solve_bound, example_args, name="mpc.solve")
        self._log("plan 솔버 그래프 캡처")

    # -- prop leaves --------------------------------------------------------------
    def _leaf_guards(self, owner) -> Dict[str, Callable[[], Any]]:
        """Guards for the two prop leaves.

        Both fetch the props they work on *inside* the body -- `track.props_for(tid)` -- so those
        tensors are baked into the graph at capture. Nothing in the argument list mentions them, and
        without these guards a rebound prop set would be replayed against the old addresses.

        `tid` is guarded by identity and shape, not by value: within one session it selects which
        track each env is on and never changes, and reading its values every step would mean a host
        synchronise on the hot path -- the cost this whole module exists to avoid. A different track
        means a different session, which builds and captures again.
        """
        sim = self.sim
        g: Dict[str, Callable[[], Any]] = {
            "sim.tid": lambda: sim.tid,
            "sim.car_dims": lambda: getattr(sim, "car_dims", None),
            "sim.corners": lambda: getattr(sim, "corners", None),
            "owner.track": lambda: id(getattr(owner, "track", None)),
        }
        tr = getattr(owner, "track", None)
        for attr in dir(tr):
            if attr.startswith("p_") and torch.is_tensor(getattr(tr, attr, None)):
                g[f"track.{attr}"] = (lambda a=attr: getattr(getattr(owner, "track", None), a, None))
        return {k: v for k, v in g.items() if v() is not None}

    @staticmethod
    def has_props(track) -> bool:
        """Whether this track carries any modelled props at all. A map without them has nothing for
        these two leaves to do, and capturing an empty one buys nothing."""
        p = getattr(track, "p_poses", None)
        return torch.is_tensor(p) and p.ndim >= 2 and int(p.shape[1]) > 0

    def capture_prop_contact(self, example_args: Sequence[Any]) -> None:
        ok, why = _cuda_eligible(self.sim)
        if not ok:
            raise NotCapturable(why)
        if not self.has_props(self.sim.track):
            raise NotCapturable("이 맵에는 프롭이 없습니다")
        self._leaf_contact = GraphedCallable(
            self.sim._prop_contact, example_args, guards=self._leaf_guards(self.sim),
            name="_prop_contact")
        self._log("프롭 접촉 그래프 캡처")

    def capture_merge_props(self, lidar, example_args: Sequence[Any]) -> None:
        ok, why = _cuda_eligible(self.sim)
        if not ok:
            raise NotCapturable(why)
        if not self.has_props(getattr(lidar, "track", None)):
            raise NotCapturable("이 맵에는 프롭이 없습니다")
        self._lidar = lidar
        self._leaf_merge = GraphedCallable(
            lidar._merge_props, example_args, guards=self._leaf_guards(lidar),
            name="lidar._merge_props")
        self._log("프롭 LiDAR 병합 그래프 캡처")

    @staticmethod
    def _shadow(obj, attr, fn) -> tuple:
        """Shadow a bound method with an instance attribute, remembering how to undo it.

        `sim._roll` is already an instance attribute, so restoring it is an assignment. These two
        are plain methods on the class, so the hook is a *new* entry in the instance dict and
        undoing it means deleting that entry -- assigning the bound method back would leave a
        permanent instance attribute holding a reference to the object it is bound to.
        """
        had_own = attr in obj.__dict__
        prev = obj.__dict__.get(attr)
        setattr(obj, attr, fn)
        return (obj, attr, had_own, prev)

    @staticmethod
    def _unshadow(entry) -> None:
        obj, attr, had_own, prev = entry
        if had_own:
            setattr(obj, attr, prev)
        else:
            obj.__dict__.pop(attr, None)

    # -- hooks -------------------------------------------------------------------
    def install(self, tracker=None) -> None:
        """Point the simulator and tracker at the graphs. Undone by `release()`."""
        if self._installed:
            raise GuardViolation("fast path is already installed")
        self._prev_roll = self.sim._roll
        self.sim._roll = self.roll
        if tracker is not None and self.mpc is not None:
            self._tracker = tracker
            self._prev_solver = getattr(tracker, "_solver", None)
            tracker._solver = self.mpc
        if self._leaf_contact is not None:
            self._shadows.append(self._shadow(self.sim, "_prop_contact", self._leaf_contact))
        if self._leaf_merge is not None and self._lidar is not None:
            self._shadows.append(self._shadow(self._lidar, "_merge_props", self._leaf_merge))
        self._installed = True

    # -- ownership ---------------------------------------------------------------
    def adopt(self) -> None:
        """Called once from the stepping thread, before its first step."""
        for gc in self._by_phase.values():
            gc.adopt()
        for gc in (self.mpc, self._leaf_contact, self._leaf_merge):
            if gc is not None:
                gc.adopt()
        self._adopted = True

    # -- dispatch ----------------------------------------------------------------
    def roll(self, *args):
        """Replay the graph for the phase the simulator is *about* to run.

        Dispatch is on `sim._imu_phase`, not on the sample count: phases 0-2 each emit one sample
        but at different substeps, so keying on the count would alias three different bodies. The
        lookup is a direct index -- a missing phase must raise, not wrap around onto a graph whose
        IMU sample lands somewhere else.
        """
        return self._by_phase[int(self.sim._imu_phase)](*args)

    def release(self) -> None:
        """Restore the hooks, then drop everything this session captured.

        Order matters: the dispatcher has to stop pointing at the graphs before the graphs go, or a
        step taken after teardown raises `KeyError` one step later and reads as an unrelated crash.
        """
        if self._installed and self.sim is not None:
            self.sim._roll = self._prev_roll
            if self._tracker is not None:
                self._tracker._solver = self._prev_solver
            for entry in reversed(self._shadows):
                self._unshadow(entry)
            self._installed = False
        self._shadows = []
        self._by_phase.clear()
        self.mpc = None
        self._leaf_contact = None
        self._leaf_merge = None
        self._lidar = None
        self._consts = None
        self._tracker = None
        self._prev_roll = None
        self._prev_solver = None
        # Dropped last, after the hooks are back. A caller holding this object -- `_release_session`
        # does, in a local -- would otherwise still reach the env through `fastpath.sim` while it
        # runs `gc.collect()`, and the env would survive that pass. Nothing leaks for the life of
        # the worker either way; it is a question of freeing on time.
        self.sim = None


# ---------------------------------------------------------------- the opponent teacher
def _view_key(t: torch.Tensor) -> tuple:
    """Which tensor, as a view: `sim.P["m"]` and `sim.params.buf` share one storage and are not
    one input."""
    return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.stride(), t.dtype)


class _InputRecorder:
    """Every CUDA tensor a call reads that it did not create, and whether it wrote to any of them.

    A `TorchFunctionMode`, so it sees each torch function and tensor method with its arguments.
    Inputs are keyed by view (`_view_key`); "created" is tracked by storage: an output whose storage
    is not an input's is the call's own. A view of an input shares the input's storage, so it stays
    an input -- which is what makes an in-place write through a view count as a write to the input.
    """

    def __init__(self, own: Sequence[torch.Tensor] = ()):
        from torch.overrides import TorchFunctionMode

        rec = self
        self.inputs: Dict[tuple, torch.Tensor] = {}
        self.in_storage: set = set()
        self.made: set = set()
        #: views the call itself made of an input (`P["lf"][:, None, None]`): they share the input's
        #: storage but are not held anywhere, and their source is already recorded
        self.derived: set = set()
        self.own = {t.untyped_storage().data_ptr() for t in own}
        self.written: list = []

        class _Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return rec._on(func, args, kwargs or {})

        self._mode = _Mode()

    def _on(self, func, args, kwargs):
        from torch.utils._pytree import tree_flatten
        flat = tree_flatten((args, kwargs))[0]
        for t in flat:
            if torch.is_tensor(t) and t.device.type == "cuda":
                p = t.untyped_storage().data_ptr()
                if p not in self.made and p not in self.own:
                    k = _view_key(t)
                    if k not in self.derived:
                        self.inputs.setdefault(k, t)
                        self.in_storage.add(p)
        name = getattr(func, "__name__", "")
        target = args[0] if args else None
        inplace = (name.endswith("_") and not name.endswith("__")) or name == "__setitem__" \
            or "out" in kwargs
        if inplace and torch.is_tensor(target) and target.untyped_storage().data_ptr() in self.in_storage:
            self.written.append(name)
        out = func(*args, **kwargs)
        for t in tree_flatten(out)[0]:
            if torch.is_tensor(t):
                p = t.untyped_storage().data_ptr()
                if p not in self.in_storage:
                    self.made.add(p)
                elif _view_key(t) not in self.inputs:
                    self.derived.add(_view_key(t))
        return out

    def __enter__(self):
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self._mode.__exit__(*exc)


def _tensor_paths(root, max_depth: int = 6) -> Dict[tuple, list]:
    """`_view_key` -> every attribute / key path from `root` that reaches that exact CUDA view.

    Walks f1sim objects, dicts, lists, tuples and modules only; anything else is opaque. A path is
    a tuple of ("a", attr) / ("k", key) steps, resolved by `_resolve`.
    """
    found: Dict[int, list] = {}
    seen: set = set()

    def rec(obj, path, depth):
        if torch.is_tensor(obj):
            if obj.device.type == "cuda":
                found.setdefault(_view_key(obj), []).append(path)
            return
        if depth > max_depth or isinstance(obj, _ATOM) or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                rec(v, path + (("k", k),), depth + 1)
        elif isinstance(obj, (list, tuple)) and len(obj) <= 64:
            for i, v in enumerate(obj):
                rec(v, path + (("k", i),), depth + 1)
        elif isinstance(obj, torch.nn.Module):
            for n, t in list(obj.named_parameters()) + list(obj.named_buffers()):
                rec(t, path + tuple(("a", part) for part in n.split(".")), depth + 1)
        elif type(obj).__module__.startswith("f1sim") and hasattr(obj, "__dict__"):
            for k, v in list(vars(obj).items()):
                rec(v, path + (("a", k),), depth + 1)

    rec(root, (), 0)
    return found


def _resolve(root, path):
    obj = root
    for kind, key in path:
        obj = getattr(obj, key) if kind == "a" else obj[key]
    return obj


def _assign(root, path, value) -> None:
    parent = _resolve(root, path[:-1])
    kind, key = path[-1]
    if kind == "a":
        setattr(parent, key, value)
    else:
        parent[key] = value


def teacher_eligible(env) -> Tuple[bool, str]:
    """Can the opponent teacher's call be captured? Decided before anything is attempted."""
    sim = getattr(env, "sim", None)
    if sim is None:
        return False, "시뮬레이터 없음"
    ok, why = _cuda_eligible(sim)
    if not ok:
        return False, why
    if int(getattr(env, "M", 1)) <= 1 or getattr(env, "teacher", None) is None:
        return False, "teacher 상대차 없음"
    if not bool(getattr(env, "teacher_any", False)):
        return False, "teacher 가 모는 차가 없음"
    tracker = getattr(env, "tracker", None)
    solver = getattr(tracker, "_solver", None)
    if isinstance(solver, GraphedCallable) and all(
            getattr(tracker, h, None) is None for h in ("_input_hook", "_plan_hook", "_command_hook")):
        # The interactive teacher previews the installed MPC in this configuration, and that call
        # would replay the solver's graph inside this capture -- which CUDA refuses, fatally.
        return False, "teacher 의 MPC 미리보기가 솔버 그래프를 재생함 (legacy 제어기)"
    return True, ""


def _setting(v):
    """A teacher setting as a guard value. A tensor-valued one (a per-car speed scale in a slot
    table, redrawn at every race reset) is an input the capture already follows by path, so only
    its layout is guarded; a Python value is baked into the graph and is guarded by value."""
    if torch.is_tensor(v):
        return ("tensor", tuple(v.shape), v.dtype, v.device.type)
    return v


class _TeacherCapture:
    """One teacher object's call, captured against private clones of everything it reads."""

    def __init__(self, env, eager, teacher, example_args: Sequence[Any]):
        self.teacher = teacher
        guards = {
            "teacher.speed_scale": lambda: _setting(getattr(teacher, "speed_scale", None)),
            "teacher.label_grip": lambda: _setting(getattr(teacher, "label_grip", None)),
            "teacher.offset_limit": lambda: _setting(getattr(teacher, "offset_limit", None)),
            "tracker": lambda: (id(env.tracker), type(env.tracker).__name__),
            "tracker.spec": lambda: env.tracker.spec,
            "tracker.hooks": lambda: tuple(id(getattr(env.tracker, h, None))
                                           for h in ("_input_hook", "_plan_hook", "_command_hook")),
            "tracker._solver": lambda: id(getattr(env.tracker, "_solver", None)),
            "env.B": lambda: int(env.B),
            "env.v_max_policy": lambda: float(env.ecfg.v_max_policy),
            "env.events": lambda: id(env.events),
        }

        # 1. Which tensors does the call read, and where do they live?
        rec = _InputRecorder([a for a in example_args if torch.is_tensor(a)])
        with torch.no_grad(), rec:
            eager(teacher, *example_args)
        if rec.written:
            raise NotCapturable(f"teacher 가 외부 텐서에 씁니다 ({', '.join(sorted(set(rec.written)))})")
        paths = _tensor_paths(env)
        refs = []
        for key, t in rec.inputs.items():
            ps = paths.get(key)
            if not ps:
                raise NotCapturable(f"출처를 찾을 수 없는 teacher 입력 {tuple(t.shape)}/{t.dtype}")
            refs.append((ps, [_resolve(env, p) for p in ps]))

        # 2. Point every path at a private clone, capture, and put the environment back.
        self.inputs = [(ps, live[0].detach().clone()) for ps, live in refs]
        try:
            for (ps, clone) in self.inputs:
                for p in ps:
                    _assign(env, p, clone)

            def body(*args):
                with torch.no_grad():
                    return eager(teacher, *args)

            self.gc = GraphedCallable(body, example_args, guards=guards,
                                      name=f"opponent teacher ({type(teacher).__name__})")
        finally:
            for (ps, _clone), (_, live) in zip(self.inputs, refs):
                for p, orig in zip(ps, live):
                    _assign(env, p, orig)


class TeacherGraph:
    """`env._teacher_normalized` for the race's teacher-driven cars, one CUDA graph per teacher.

    Beside a single viewed car, the teacher is most of a step: a candidate family, a Gauss-Newton
    fit per candidate, a speed certification and a collision preview, each a few hundred tiny
    kernels. At the viewer's batch sizes launching them costs far more than running them -- 37 ms of
    a 58 ms step with three cars, against ~20 ms for the whole step without opponents. A slot table
    naming another teacher kind (the console's `interactive` preset) calls it once per kind, every
    step, so each teacher object the env calls gets its own graph.

    The call is a pure function of tensors the environment holds, but several of those are rebound
    every step (`sim.state`, `last_result.odom`, `tracker.last_pred`, `_plan_pose`), so a graph that
    read them at their captured addresses would replay the capture step for ever. So:

    * **Every tensor the call reads is found, not listed.** Capture runs under `_InputRecorder`, and
      each input is traced back to its attribute paths from the environment. An input with no path,
      or a write into one, makes the configuration `NotCapturable` and the teacher stays eager.
    * **The graph reads private copies.** Each input is cloned and the environment's paths point at
      the clone for the duration of the capture only, then go back. Before every replay each live
      input is copied into its clone. Nothing the environment owns is ever written by the graph --
      so an old tensor someone kept (a previous pose, a view of the last state) is never touched.
    * **Anything it cannot follow sends that teacher back to eager, for good.** A live input whose
      shape or dtype changed, a guard -- the teacher's speed scale, grip mode, lane clamp, the
      tracker's spec and hooks, the call's non-tensor arguments -- that no longer matches. Paths
      that disagree about which tensor they hold send one step to eager.
    """

    def __init__(self, env, calls: Dict[Any, Sequence[Any]], log=None):
        ok, why = teacher_eligible(env)
        if not ok:
            raise NotCapturable(why)
        if not calls:
            raise NotCapturable("teacher 호출을 관측하지 못함")
        self.env = env
        self._log = log or (lambda _t: None)
        self._eager = env._teacher_normalized         # the bound method, not a previous shadow
        self._captures: Dict[int, _TeacherCapture] = {}
        self.fell_back: Dict[str, str] = {}           # teacher class -> why it went eager
        self.replays = 0
        self.eager_steps = 0
        self._installed = False
        for teacher, args in calls.items():
            self._captures[id(teacher)] = _TeacherCapture(env, self._eager, teacher, args)
        n = sum(len(c.inputs) for c in self._captures.values())
        self._log(f"상대차 teacher 그래프 {len(self._captures)}개 캡처 (입력 {n}개)")

    # -- ownership / hooks -----------------------------------------------------------
    def adopt(self) -> None:
        for c in self._captures.values():
            c.gc.adopt()

    def install(self) -> None:
        if self._installed:
            raise GuardViolation("teacher graph is already installed")
        self.env._teacher_normalized = self
        self._installed = True

    def release(self) -> None:
        if self._installed and self.env is not None:
            self.env.__dict__.pop("_teacher_normalized", None)
            self._installed = False
        self._captures = {}
        self.env = None

    # -- dispatch ---------------------------------------------------------------------
    def _give_up(self, cap: _TeacherCapture, why: str) -> None:
        self._captures.pop(id(cap.teacher), None)
        self.fell_back[type(cap.teacher).__name__] = why
        self._log(f"상대차 teacher ({type(cap.teacher).__name__}) 는 이제 eager 로 돕니다: {why}")

    def __call__(self, teacher, *args):
        cap = self._captures.get(id(teacher))
        if cap is None or cap.teacher is not teacher:
            return self._eager(teacher, *args)
        env = self.env
        live_of = []
        for ps, clone in cap.inputs:
            live = _resolve(env, ps[0])
            for p in ps[1:]:
                other = _resolve(env, p)
                if other is not live and _view_key(other) != _view_key(live):
                    # Two names for what was one tensor now hold different ones: which of them the
                    # call reads is not something this can know. Eager for this step only.
                    self.eager_steps += 1
                    return self._eager(teacher, *args)
            if live.shape != clone.shape or live.dtype != clone.dtype:
                self._give_up(cap, f"입력 {tuple(clone.shape)} 이 {tuple(live.shape)} 로 바뀜")
                return self._eager(teacher, *args)
            live_of.append(live)
        try:
            cap.gc.check_guards()
        except GuardViolation as exc:
            self._give_up(cap, str(exc))
            return self._eager(teacher, *args)
        for (_ps, clone), live in zip(cap.inputs, live_of):
            if live is not clone:
                clone.copy_(live)
        try:
            out = cap.gc(*args)
        except GuardViolation as exc:
            self._give_up(cap, str(exc))
            return self._eager(teacher, *args)
        self.replays += 1
        return out
