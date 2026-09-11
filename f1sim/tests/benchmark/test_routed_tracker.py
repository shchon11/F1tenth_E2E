"""RoutedTracker: delegation, masking and lifecycle, on CPU with stub trackers.

These use stubs rather than a real env because the property under test is routing, and a stub can
assert things a live PlanTracker cannot: exactly which rows each side saw, that both were called on
the full batch, and that a candidate-side change leaves the opponent's output bit-identical.

`test_independence_is_bitwise` is the blocking fixture from the design: if it fails, the benchmark
is not measuring overtaking against a fixed opponent.
"""
from __future__ import annotations
import importlib
import pytest

torch = pytest.importorskip("torch")

B = 4


class StubTracker:
    """Records what it was called with; output depends on a knob so divergence is detectable."""

    def __init__(self, knob: float, spec: str):
        self.knob, self.spec = knob, spec
        self.wb, self.s_max, self.v_max = 0.33, 0.4, 8.0
        self.calls, self.resets = [], []
        self._solver = None
        self.u_prev = torch.zeros(B, 2)
        self.compile_solver = False
        self.last_ref = torch.full((B, 3, 4), knob)
        self.last_pred = torch.full((B, 3, 4), knob * 2)

    def __call__(self, action, v_meas, speed_cap, yaw_rate, delay=None):
        self.calls.append(action.clone())
        return action[:, :2] * self.knob

    def reset(self, ids):
        self.resets.append(ids.clone())


class StubEnv:
    def __init__(self, on_policy):
        self.on_policy = on_policy
        self.tracker = None


@pytest.fixture
def rt(bench):
    return importlib.import_module("f1sim.learn.benchmark.routed_tracker")


@pytest.fixture
def wired(rt):
    def _wire(knob_c=1.0, knob_r=2.0, mask=(True, False, True, False)):
        env = StubEnv(torch.tensor(mask))
        cand, ref = StubTracker(knob_c, "cand_spec"), StubTracker(knob_r, "ref_spec")
        router = rt.RoutedTracker(candidate=cand, reference=ref, env=env)
        env.tracker = router
        return env, cand, ref, router
    return _wire


def act():
    return torch.arange(B * 8, dtype=torch.float32).reshape(B, 8)


def test_both_trackers_see_the_full_batch(wired):
    """Row-subsetting is a shape error against GripMPC's env.B buffers, so both get all B rows."""
    _, cand, ref, router = wired()
    router(act(), torch.zeros(B), torch.ones(B), None)
    assert cand.calls[0].shape == (B, 8)
    assert ref.calls[0].shape == (B, 8)


def test_output_is_masked_per_row(wired):
    _, _, _, router = wired(knob_c=1.0, knob_r=2.0, mask=(True, False, True, False))
    out = router(act(), torch.zeros(B), torch.ones(B), None)
    a = act()[:, :2]
    assert torch.equal(out[0], a[0] * 1.0)       # candidate row
    assert torch.equal(out[1], a[1] * 2.0)       # opponent row
    assert torch.equal(out[2], a[2] * 1.0)
    assert torch.equal(out[3], a[3] * 2.0)


def test_independence_is_bitwise(wired):
    """BLOCKING. Swap the candidate's behaviour; opponent rows must not move one bit.

    This is the 2x2 of {actor} x {arm} from the design, reduced to what actually varies at the
    tracker boundary: the candidate's output. If opponent rows move, the opponent is being driven
    by the candidate's controller and no overtaking number from this suite means anything.
    """
    mask = (True, False, True, False)
    outs = []
    for knob_c in (1.0, 3.0, 7.0, -5.0):        # four different candidate behaviours
        _, _, _, router = wired(knob_c=knob_c, knob_r=2.0, mask=mask)
        outs.append(router(act(), torch.zeros(B), torch.ones(B), None))
    opp = ~torch.tensor(mask)
    for o in outs[1:]:
        assert torch.equal(o[opp], outs[0][opp]), "opponent rows changed with the candidate"
    # and the candidate rows really did differ, so the test is not vacuously passing
    cand_rows = torch.tensor(mask)
    assert not torch.equal(outs[1][cand_rows], outs[0][cand_rows])


def test_spec_is_the_reference_spec(wired):
    """Both teacher paths build plans from tracker.spec (gym_env 483 and 772)."""
    _, _, ref, router = wired()
    assert router.spec == ref.spec == "ref_spec"


def test_reset_reaches_both(wired):
    _, cand, ref, router = wired()
    ids = torch.tensor([0, 2])
    router.reset(ids)
    assert len(cand.resets) == 1 and len(ref.resets) == 1
    assert torch.equal(cand.resets[0], ids) and torch.equal(ref.resets[0], ids)


def test_plans_are_masked(wired):
    _, _, _, router = wired(knob_c=1.0, knob_r=2.0, mask=(True, False, True, False))
    assert router.last_ref[0].eq(1.0).all() and router.last_ref[1].eq(2.0).all()
    assert router.last_pred[0].eq(2.0).all() and router.last_pred[1].eq(4.0).all()


def test_plans_tolerate_one_side_being_none(wired):
    _, cand, _, router = wired()
    cand.last_ref = None
    assert router.last_ref is not None           # falls back rather than crashing a viewer


def test_solver_proxies_the_candidate_only(wired):
    """A later assignment must not silently hand the opponent the candidate's arm."""
    _, cand, ref, router = wired()
    sentinel = object()
    router._solver = sentinel
    assert cand._solver is sentinel
    assert ref._solver is None


def test_u_prev_and_compile_solver_delegate_to_candidate(wired):
    _, cand, ref, router = wired()
    cand.compile_solver = True
    ref.compile_solver = False
    assert router.compile_solver is True
    assert router.u_prev is cand.u_prev


def test_uninstall_restores_by_identity(wired):
    env, cand, _, router = wired()
    assert env.tracker is router
    router.uninstall()
    assert env.tracker is cand                   # identity, not equality


def test_mask_is_read_live(wired):
    """on_policy is rewritten on reset in mixed mode; a snapshot would route stale rows."""
    env, _, _, router = wired(mask=(True, True, False, False))
    first = router(act(), torch.zeros(B), torch.ones(B), None)
    env.on_policy = torch.tensor([False, False, True, True])
    second = router(act(), torch.zeros(B), torch.ones(B), None)
    assert not torch.equal(first, second)


def test_install_order_is_documented_and_ordered(rt):
    order = rt.install_order()
    assert order.index("prepare_graph_runtime(env)") < \
        order.index("ControllerRuntime(arm).install(graph_rt=holder)")
    assert order.index("ControllerRuntime(arm).install(graph_rt=holder)") < \
        next(i for i, s in enumerate(order) if s.startswith("env.tracker = RoutedTracker"))
