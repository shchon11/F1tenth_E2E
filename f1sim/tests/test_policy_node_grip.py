"""The ROS policy node's plan-controller arm: `fixed_low` must bind the tracker's speed reference on
the same `PlanTracker` object the car drives through, and `release()` must restore it exactly."""
import pytest

pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")

import f1sim_ros.policy_node as pn                       # noqa: E402
from f1sim.mpc import PlanTracker                        # noqa: E402


def _tight_fast_plan():
    # six curvature knots at a hard turn, two speed targets at the top of the scale
    return torch.tensor([[0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 1.0, 1.0]])


def _run(tracker):
    tracker.reset(torch.zeros(1, dtype=torch.long))
    tracker(_tight_fast_plan(), torch.tensor([3.0]), torch.tensor([9.0]), None, delay=torch.tensor([0.035]))
    return tracker.last_ref.clone(), tracker.u_prev.clone()


def test_fixed_low_binds_the_speed_reference_and_release_restores_it():
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
    ref_legacy, u_legacy = _run(tr)
    grip = pn.install_grip_arm(tr, "fixed_low", "cpu")
    assert grip is not None and float(grip.mu[0]) == pytest.approx(0.73423)
    ref_fixed, _ = _run(tr)
    # the reference speed profile (last column of the reference) must come down somewhere
    v_legacy, v_fixed = ref_legacy[..., -1], ref_fixed[..., -1]
    assert float(v_fixed.max()) < float(v_legacy.max()), (float(v_legacy.max()), float(v_fixed.max()))
    grip.release()
    ref_back, u_back = _run(tr)
    assert torch.allclose(ref_back, ref_legacy) and torch.allclose(u_back, u_legacy)


def test_legacy_installs_nothing_and_other_arms_are_refused():
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
    assert pn.install_grip_arm(tr, "legacy", "cpu") is None
    assert pn.install_grip_arm(None, "fixed_low", "cpu") is None
    with pytest.raises(ValueError):
        pn.install_grip_arm(tr, "estimated", "cpu")


def test_grip_mu_override_is_used():
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
    grip = pn.install_grip_arm(tr, "fixed_low", "cpu", mu=0.9)
    assert float(grip.mu[0]) == pytest.approx(0.9)
    grip.release()
