"""Opt-in CUDA-graph runtime for a training env, reusing the viewer's `SimGraphFastPath`.

`torch.compile` on the physics substep loop reaches a CUDA graph, but only after an inductor compile
that costs minutes. Capturing the graph directly reaches the same replay in about a second. That
matters for a pilot of short runs, where compile time is a real share of the wall clock.

This module is a thin adapter, not a second implementation: the capture, the per-IMU-phase
dispatch, the guards, the ownership transfer and the release all live in
`f1sim.viewer.graph_fastpath`. Nothing about the physics or the solver is duplicated here, and no
viewer source is touched.

    rt = prepare_graph_runtime(env)      # None when the env is not eligible; the caller runs eager
    ...
    rt.release()                         # restores env.sim._roll and the tracker's solver

Eligibility, ownership and the failure rule are the fast path's own
(`graph_fastpath.roll_eligible`, `adopt`, `CaptureFailed`); this only decides *when* to capture and
holds the result alive for the session. A capture failure is fatal to the process by design -- after
one, `torch.randn(device="cuda")` raises while ordinary arithmetic still works, so falling back to
eager would leave a run whose physics is fine and whose sensor noise throws.
"""
from __future__ import annotations

import time
from typing import Optional

import torch


def prepare_graph_runtime(env, warmup_steps: int = 2, log=print):
    """Capture and install CUDA graphs for `env`'s physics and plan solver, on the calling thread.

    Returns the holder, or `None` when the env is not eligible -- in which case the caller simply
    runs as it did before. Raises `graph_fastpath.CaptureFailed` if a capture is attempted and
    fails, which must not be caught and retried in the same process.

    `warmup_steps` real steps run first, for two reasons: they are what the arguments are *recorded*
    from -- `delay_s` is a tensor or a float depending on the jitter setting, so guessing the
    signature is how a graph gets captured against the wrong thing -- and they get the allocator's
    first-touch and any lazy handle creation out of the way before the capture stream is used.
    Those steps advance the env, exactly as the viewer's do.
    """
    sim = env.sim
    if sim.device.type != "cuda":
        log("graph runtime: CPU env, running eager")
        return None
    from ..viewer.graph_fastpath import NotCapturable, SimGraphFastPath, roll_eligible
    ok, why = roll_eligible(sim)
    if not ok:
        log(f"graph runtime: not eligible ({why}); running eager")
        return None

    tracker = getattr(env, "tracker", None)
    rec: dict = {}
    eager_roll = sim._roll
    prev_solver = getattr(tracker, "_solver", None) if tracker is not None else None

    def roll_spy(*args):
        rec["roll"] = args
        return eager_roll(*args)

    sim._roll = roll_spy
    # (11-14) the two prop leaves, recorded from the same steps as the console does (`sim_worker`): the contact
    # test `sim.step` runs after every roll and the LiDAR's prop merge. Training never captured them, and the
    # contact test alone was ~21 % of a training env step at 640 envs, eager. Spies are instance attributes,
    # deleted afterwards (not assigned back: a bound method in the instance dict would pin the object).
    lidar = getattr(sim, "lidar", None)
    eager_contact = sim._prop_contact
    eager_merge = getattr(lidar, "_merge_props", None)

    def contact_spy(*a, **kw):
        if not kw:                        # only the plain `sim.step` call is what the leaf is captured for
            rec["contact"] = a
        return eager_contact(*a, **kw)

    def merge_spy(*a):
        rec["merge"] = a
        return eager_merge(*a)

    sim._prop_contact = contact_spy
    if eager_merge is not None:
        lidar._merge_props = merge_spy
    if tracker is not None:
        from .. import mpc as _mpc
        base = prev_solver or (_mpc.solve_fast if tracker.compile_solver else _mpc.solve)

        def solve_spy(*args):
            rec["mpc"] = args
            return base(*args)

        tracker._solver = solve_spy
    try:
        act = torch.zeros(env.B, env.act_dim, device=sim.device)
        for _ in range(max(1, warmup_steps)):
            env.step(act)
    finally:
        sim._roll = eager_roll
        sim.__dict__.pop("_prop_contact", None)
        if eager_merge is not None:
            lidar.__dict__.pop("_merge_props", None)
        if tracker is not None:
            tracker._solver = prev_solver
    if "roll" not in rec:
        log("graph runtime: no physics call observed; running eager")
        return None

    t0 = time.perf_counter()
    fp = SimGraphFastPath(sim, log=lambda t: log(f"graph runtime: {t}"))
    try:
        fp.capture_roll(rec["roll"])
        if tracker is not None and "mpc" in rec:
            try:
                fp.capture_mpc(tracker, rec["mpc"])
            except NotCapturable as exc:
                log(f"graph runtime: plan solver stays eager ({exc})")
        if "contact" in rec:
            try:
                fp.capture_prop_contact(rec["contact"])
                fp._leaf_contact = _LeafOrEager(fp._leaf_contact, eager_contact, log)
            except NotCapturable as exc:
                log(f"graph runtime: prop contact stays eager ({exc})")
        if lidar is not None and "merge" in rec:
            try:
                fp.capture_merge_props(lidar, rec["merge"])
                fp._leaf_merge = _LeafOrEager(fp._leaf_merge, eager_merge, log)
            except NotCapturable as exc:
                log(f"graph runtime: prop LiDAR merge stays eager ({exc})")
        fp.install(tracker)
        fp.adopt()                       # captured and replayed on this thread; adopt before stepping
    except NotCapturable as exc:
        fp.release()
        log(f"graph runtime: not eligible ({exc}); running eager")
        return None
    except BaseException:
        fp.release()
        raise
    # (11-14) with `opp_every` > 1 a step may hold the planners' last command and call none of them: the capture
    # records from one real step, so it runs with every step planning
    k_ = getattr(env, "opp_every", 1)
    env.opp_every = 1
    try:
        _capture_opponents(env, fp, act, log)
    finally:
        env.opp_every = k_
        env._opp_held = None
    log(f"graph runtime: captured in {time.perf_counter() - t0:.2f} s "
        f"({len(fp._by_phase)} physics graph(s), solver {'graphed' if fp.mpc else 'eager'}, "
        f"opponent planners {'graphed' if fp.teacher_graph is not None else 'eager'})")
    return fp


class _LeafOrEager:
    """A captured prop leaf that steps aside instead of stopping the run: a call with keywords (another call
    site than the one captured) or after its guards changed (a prop set rebound under it) goes to the eager
    method -- for good, after a guard change, said once. Everything else (adopt, release) is the graph's."""

    def __init__(self, graphed, eager, log):
        self._g, self._eager, self._log, self.dead = graphed, eager, log, False

    def __call__(self, *a, **kw):
        if kw or self.dead:
            return self._eager(*a, **kw)
        from ..viewer.graph_fastpath import GuardViolation
        try:
            return self._g(*a)
        except GuardViolation as exc:
            self.dead = True
            self._log(f"graph runtime: {getattr(self._g, 'name', 'leaf')} goes back to eager ({exc})")
            return self._eager(*a)

    def __getattr__(self, name):
        return getattr(self._g, name)


def _capture_opponents(env, fp, act, log) -> None:
    """The teacher-driven opponents' planners, one CUDA graph per planner (`TeacherGraph`).

    They are most of a training step once there are several kinds: in s911's recipe (a race of two,
    the opponent drawn per race from four planners) `_opponent_actions` was 61 % of an env step at
    256 envs, eager, each planner a few hundred small kernels asked for the whole batch. The viewer
    has captured them since `7f7e831`; training never did. Their calls are recorded from one real
    step, exactly as the viewer records them, and a planner that cannot be captured, or that a
    later change (a controller hook, a new shape) makes the graph unable to follow, stays or goes
    back to eager and says why -- the graph guards decide that, not this function.
    """
    from ..viewer.graph_fastpath import NotCapturable, TeacherGraph, teacher_eligible
    ok, why = teacher_eligible(env)
    if not ok:
        log(f"graph runtime: opponent planners stay eager ({why})")
        return
    rec: dict = {}
    eager = env._teacher_normalized

    def spy(teacher, *a):
        rec[teacher] = a
        return eager(teacher, *a)

    env._teacher_normalized = spy
    try:
        env.step(act)
    finally:
        env.__dict__.pop("_teacher_normalized", None)
    if not rec:
        log("graph runtime: no opponent planner call observed; they stay eager")
        return
    try:
        tg = TeacherGraph(env, rec, log=lambda text: log(f"graph runtime: {text}"))
    except NotCapturable as exc:
        log(f"graph runtime: opponent planners stay eager ({exc})")
        return
    tg.install()
    tg.adopt()                           # captured here, replayed here
    fp.teacher_graph = tg


def prepare_actor_graph(actor, scan, proprio, hidden=None, log=print):
    """Capture the actor's control step as a CUDA graph, or return None and say why.

    `scan`, `proprio` and `hidden` are one real example of each argument -- real, because the graph
    binds their addresses, dtypes and shapes and a guessed signature is how a graph gets captured
    against the wrong thing. `hidden` is None for a feedforward actor.

    The returned callable is `(scan, proprio, hidden) -> (mu, next hidden)` and holds the hidden
    state in a static buffer that every replay copies into, which is the only shape of this that is
    safe: a tensor a replay produced is overwritten by the next one.

    Call it where a `CaptureFailed` can be handled -- during a session build, not inside the step
    loop. `NotCapturable` is not an error: the caller runs eager, and this says so.
    """
    from ..viewer.graph_fastpath import NotCapturable, graph_actor_step
    args = (scan, proprio) + ((hidden,) if hidden is not None else ())
    if hidden is None:
        log("actor graph: feedforward actor, nothing to carry; running eager")
        return None
    try:
        g = graph_actor_step(actor, args)
    except NotCapturable as exc:
        log(f"actor graph: not eligible ({exc}); running eager")
        return None
    log("actor graph: captured (hidden state in a static buffer, updated in place)")
    return g


def release_graph_runtime(rt) -> None:
    """Put `sim._roll` and the tracker's solver back. Safe to call with `None`.

    Leaving the dispatcher installed after the graphs are gone fails one step later, which reads as
    an unrelated crash, and the `sim -> _roll -> holder -> sim` cycle keeps the env alive.
    """
    if rt is not None:
        rt.release()
