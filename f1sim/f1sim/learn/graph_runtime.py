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
        fp.install(tracker)
        fp.adopt()                       # captured and replayed on this thread; adopt before stepping
    except NotCapturable as exc:
        fp.release()
        log(f"graph runtime: not eligible ({exc}); running eager")
        return None
    except BaseException:
        fp.release()
        raise
    log(f"graph runtime: captured in {time.perf_counter() - t0:.2f} s "
        f"({len(fp._by_phase)} physics graph(s), solver {'graphed' if fp.mpc else 'eager'})")
    return fp


def release_graph_runtime(rt) -> None:
    """Put `sim._roll` and the tracker's solver back. Safe to call with `None`.

    Leaving the dispatcher installed after the graphs are gone fails one step later, which reads as
    an unrelated crash, and the `sim -> _roll -> holder -> sim` cycle keeps the env alive.
    """
    if rt is not None:
        rt.release()
