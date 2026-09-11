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
    # `_resolve_wall_contact` mutates `self.prop_touched` inside the loop; a graph replays device
    # work only and would drop that, silently losing substep prop contacts.
    if not bool(getattr(sim.cfg.sim, "terminate_on_collision", True)):
        return False, "soft wall 모드 (충돌 시 종료가 꺼져 있음)"
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
        self._log("플랜 솔버 그래프 캡처")

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
