"""Guards and failure semantics of the viewer's CUDA-graph fast path.

CPU only. Nothing here captures a graph -- capture needs CUDA, and a *failed* capture poisons the
process, so the fatal-path tests inject `CaptureFailed` rather than provoking one. That is the
point of the exercise anyway: what the worker does when capture fails, not that it can fail.

The equivalence of a replayed graph against eager is measured separately on hardware; these are the
checks that must keep passing without a GPU.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from f1sim.mpc import PlanSpec, PlanTracker, solve
from f1sim.viewer import sim_worker
from f1sim.viewer.graph_fastpath import (CaptureFailed, GuardViolation, NotCapturable,
                                         SimGraphFastPath, _dict_sig, _freeze, mpc_eligible,
                                         roll_eligible)


# --------------------------------------------------------------------- snapshots
def test_freeze_sees_a_mutated_planspec():
    """The reason `_other` stores snapshots and not the object.

    Keeping the caller's `PlanSpec` and comparing it against itself is unconditionally equal: a
    changed `N` mutates the very object the check reads.
    """
    spec = PlanSpec()
    before = _freeze(spec)
    same_object = spec
    spec.N = spec.N + 1
    assert same_object == spec, "the reference compare that a snapshot has to replace"
    assert _freeze(spec) != before


def test_freeze_sees_a_changed_tuple_field():
    spec = PlanSpec()
    before = _freeze(spec)
    spec.q = (2.0, 6.0, 1.0, 0.4)
    assert _freeze(spec) != before


def test_freeze_rejects_a_value_it_cannot_snapshot():
    with pytest.raises(NotCapturable):
        _freeze(object())


def test_dict_sig_sees_a_rebound_key():
    P = {"a": torch.zeros(3)}
    before = _dict_sig(P)
    P["a"] = torch.zeros(3)                 # same values, new storage: the graph reads the address
    assert _dict_sig(P) != before


def test_dict_sig_sees_an_added_scalar():
    """A non-tensor entry is frozen too, not skipped: it would be baked in by value."""
    P = {"a": torch.zeros(3)}
    before = _dict_sig(P)
    P["susp_wn"] = 12.0
    assert _dict_sig(P) != before


def test_dict_sig_sees_a_changed_scalar():
    P = {"a": torch.zeros(3), "susp_wn": 12.0}
    before = _dict_sig(P)
    P["susp_wn"] = 13.0
    assert _dict_sig(P) != before


def test_dict_sig_rejects_an_unknown_value():
    with pytest.raises(NotCapturable):
        _dict_sig({"a": torch.zeros(3), "weird": object()})


# ------------------------------------------------------------------ eligibility
class _FakeSim:
    def __init__(self, **kw):
        self.device = torch.device("cpu")
        self.cfg = types.SimpleNamespace(
            sim=types.SimpleNamespace(compile=False, terminate_on_collision=True),
            imu=types.SimpleNamespace(enabled=True, imu_rate=50.0))
        self.imu_period = 4
        self.imu_schedule = [[19], [14], [9], [4, 24]]
        self.state = torch.zeros(1, 8)
        self._imu_phase = 0
        self._roll = self._roll_physics
        self.__dict__.update(kw)

    def _roll_physics(self, *a):
        return a


def test_cpu_session_is_not_eligible_and_says_so():
    ok, why = roll_eligible(_FakeSim())
    assert ok is False and why


def test_soft_wall_is_not_eligible():
    """`_resolve_wall_contact` mutates `prop_touched` inside the substep loop; a graph replays
    device work only and would silently drop those contacts."""
    sim = _FakeSim()
    sim.device = torch.device("cuda")        # not used: the wall check must come first in effect
    sim.cfg.sim.terminate_on_collision = False
    ok, why = roll_eligible(sim)
    assert ok is False
    assert "soft wall" in why


def test_schedule_disagreeing_with_period_is_not_eligible():
    sim = _FakeSim()
    sim.imu_period = 4
    sim.imu_schedule = [[19], [14]]
    ok, why = roll_eligible(sim)
    assert ok is False


def test_mpc_eligibility_refuses_a_compiled_tracker():
    sim = _FakeSim()
    trk = PlanTracker(1, "cpu", 0.32, 0.4, 9.0, compile_solver=True)
    ok, why = mpc_eligible(sim, trk)
    assert ok is False and why


def test_mpc_eligibility_refuses_a_non_float32_tracker():
    """`build_ilqr_consts` matches the inline branch only at torch's default dtype."""
    sim = _FakeSim()
    sim.device = torch.device("cuda")
    sim.cfg.sim.compile = False
    trk = PlanTracker(1, "cpu", 0.32, 0.4, 9.0, compile_solver=False)
    trk.u_prev = trk.u_prev.double()
    ok, why = mpc_eligible(sim, trk)
    assert ok is False
    assert "float32" in why


# ------------------------------------------------------------------- dispatch
def test_roll_refuses_an_uncaptured_phase_instead_of_wrapping():
    """A modulo over "how many graphs exist" would map phase 3 onto graph 1, whose IMU sample lands
    at a different substep. Wrong data, no error. A direct index raises instead."""
    sim = _FakeSim()
    fp = SimGraphFastPath(sim)
    fp._by_phase = {0: lambda *a: "phase0", 1: lambda *a: "phase1"}
    sim._imu_phase = 0
    assert fp.roll() == "phase0"
    sim._imu_phase = 3
    with pytest.raises(KeyError):
        fp.roll()


# --------------------------------------------------------------------- hooks
class _FakeTracker:
    def __init__(self):
        self._solver = None


def test_install_then_release_restores_both_hooks():
    sim = _FakeSim()
    original_roll = sim._roll
    trk = _FakeTracker()
    fp = SimGraphFastPath(sim)
    fp.mpc = lambda *a: None                 # stands in for a captured solver
    fp.install(trk)
    assert sim._roll == fp.roll
    assert trk._solver is fp.mpc
    fp.release()
    assert sim._roll is original_roll
    assert trk._solver is None


def test_release_drops_the_simulator_reference():
    """`_release_session` holds this object in a local while it runs `gc.collect()`; keeping `sim`
    would keep the whole env reachable through it for that pass."""
    sim = _FakeSim()
    fp = SimGraphFastPath(sim)
    fp.install(None)
    fp.release()
    assert fp.sim is None


def test_install_twice_is_refused():
    fp = SimGraphFastPath(_FakeSim())
    fp.install(None)
    with pytest.raises(GuardViolation):
        fp.install(None)


def test_release_is_idempotent():
    fp = SimGraphFastPath(_FakeSim())
    fp.install(None)
    fp.release()
    fp.release()                             # must not raise on the second teardown path


# ------------------------------------------------------- fatal capture failure
def _bare_worker():
    """A `SimWorker` without its constructor: these tests exercise two methods, and the real
    constructor opens pipes and starts a sender thread."""
    w = sim_worker.SimWorker.__new__(sim_worker.SimWorker)
    w.session = None
    w.gen = 1
    w.prepare_gen = None
    w.cancel_gen = None
    w.alive = True
    w.running = True
    w.said = []
    w.say = lambda kind, **kw: w.said.append((kind, kw))
    w.teardown = lambda: None
    return w


def test_capture_failure_is_not_retried_even_when_it_says_out_of_memory():
    """The OOM retry exists for a build that lost a race with the previous session's memory. A
    capture that ran out of memory still poisoned the context, so retrying on it is worse than
    failing -- and the message contains the words the retry looks for."""
    w = _bare_worker()
    w.session = {"env": object()}            # a resident previous session: the retry's precondition
    calls = []

    def build(cfg, gen):
        calls.append(gen)
        raise CaptureFailed("mpc.solve: OutOfMemoryError: CUDA out of memory")

    w.build_session = build
    with pytest.raises(CaptureFailed):
        w._build_with_oom_retry(object(), 2)
    assert calls == [2], "capture failure must not be retried on the same context"


def test_a_plain_oom_is_still_retried_once():
    """The pre-existing behaviour the test above must not have broken."""
    w = _bare_worker()
    w.session = {"env": object()}
    calls = []

    def build(cfg, gen):
        calls.append(gen)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return {"ok": True}

    w.build_session = build
    assert w._build_with_oom_retry(object(), 2) == {"ok": True}
    assert calls == [2, 2]


def test_a_capture_failure_stops_the_worker_before_it_lifts_the_build_hold():
    """Reporting the error and staying alive would hand the next retry a poisoned context, and the
    console retries on whatever worker is up. Order matters too: the hold must not lift while the
    loop is still alive, or the outgoing session takes one more step and draws sensor noise from
    the generator the capture just poisoned.
    """
    w = _bare_worker()
    seen = {}

    def abandon():
        seen["alive_when_abandoned"] = w.alive
        seen["running_when_abandoned"] = w.running

    w._abandon_build = abandon
    w._build_hold = lambda on: True
    w._swap_in = lambda *a: None
    w._build_with_oom_retry = lambda cfg, gen: (_ for _ in ()).throw(
        CaptureFailed("_roll_physics[phase 0]: RuntimeError: CUDA out of memory"))
    # A config that passes `validate_start_config`, so the build is actually reached: an empty one
    # fails validation and returns from the "설정" branch without ever getting near a capture.
    w.start({"gen": 2, "config": {"map_name": "real:korea_2026_competition"}})

    assert w.alive is False, "the process must stop; the console spawns a fresh worker"
    assert seen["alive_when_abandoned"] is False
    assert seen["running_when_abandoned"] is False
    errors = [kw for kind, kw in w.said if kind.endswith("error")]
    assert errors, "the console still has to be told why"
    assert "CUDA 그래프" in errors[-1]["where"]
    assert errors[-1]["retryable"] is True


def test_plan_tracker_solver_hook_defaults_to_the_module_level_choice():
    """Nothing global is replaced: a tracker with no hook picks exactly what it picked before."""
    trk = PlanTracker(1, "cpu", 0.32, 0.4, 9.0, compile_solver=False)
    assert trk._solver is None
    seen = []
    trk._solver = lambda *a: seen.append(a) or solve(*a)
    assert trk._solver is not None


def test_mpc_solve_still_defaults_to_building_its_constants_inline():
    """Training and every existing caller must be untouched by the opt-in `consts` argument.

    Checked by NAME rather than against the whole `__defaults__` tuple. The tuple version asserted
    `== (None,)` and so failed the moment `v_limit`, `bounds` and `a_walk` were added -- all three
    of which default to `None` and change nothing for an existing caller. It was reporting the
    arrival of new optional arguments, not the property the docstring describes.

    The property is: every optional argument of `solve` defaults to `None`, so a caller passing only
    the required ones gets the original behaviour -- constants built inline, no precomputed table,
    no extra limits. A non-`None` default is what would silently change training, and that is what
    this still catches.
    """
    import inspect

    from f1sim import mpc
    optional = {n: p.default for n, p in inspect.signature(mpc.solve).parameters.items()
                if p.default is not inspect.Parameter.empty}
    assert "consts" in optional, "the opt-in `consts` argument is gone; this test is out of date"
    assert all(v is None for v in optional.values()), optional
    assert mpc.ilqr.__defaults__[-1] is None
