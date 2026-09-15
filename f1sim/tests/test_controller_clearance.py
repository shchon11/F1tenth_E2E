"""The controller node's `clearance` layer: it must bind the same `PlanTracker` the car drives
through, build its grid from `/scan` and nothing else, compose with `fixed_low` in either order, and
put the tracker back exactly as it was on `release()`.

Sibling of `test_controller_arms.py`, and deliberately the same shape: the controller is where a
wrong answer reaches a real car. The installers live in `f1sim_ros/deploy.py` since the split.
"""
import math

import pytest

pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")

import f1sim_ros.deploy as pn                            # noqa: E402
from f1sim.learn import clearance as cl                  # noqa: E402
from f1sim.learn.obs import ObsSpec                      # noqa: E402
from f1sim.mpc import PlanTracker                        # noqa: E402

SPEC = ObsSpec(n_beams=1081, act_dim=8)


def _fast_straight_plan():
    """Six zero-curvature knots and both speed targets at the top of the scale: straight, flat out."""
    return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])


def _blocked_scan(x=2.0, y=0.0, r=0.30, half_width=0.9):
    """Normalized ranges for a corridor with a post in it, cast from where the scanner really is."""
    ang = cl.beam_angles(SPEC.n_beams, pn.LIDAR_FOV)
    sa, ca = torch.sin(ang), torch.cos(ang)
    rng = torch.full((SPEC.n_beams,), float(SPEC.range_max))
    for sgn in (1.0, -1.0):
        t = torch.where(sa * sgn > 1e-6, half_width / (sa * sgn).clamp_min(1e-6),
                        torch.full_like(sa, 1e9))
        rng = torch.minimum(rng, t)
    cx, cy = x - pn.LIDAR_MOUNT_X, y
    b = ca * (-cx) + sa * (-cy)
    disc = b * b - (cx * cx + cy * cy - r * r)
    t = torch.where(disc > 0, -b - torch.sqrt(disc.clamp_min(0.0)), torch.full_like(b, 1e9))
    rng = torch.minimum(rng, torch.where(t > 0, t, torch.full_like(t, 1e9)))
    return (rng / SPEC.range_max).clamp(0.0, 1.0)[None]


def _run(tracker):
    tracker.reset(torch.zeros(1, dtype=torch.long))
    tracker(_fast_straight_plan(), torch.tensor([3.0]), torch.tensor([4.5]), None,
            delay=torch.tensor([0.035]))
    return tracker.last_ref.clone(), tracker.u_prev.clone()


def test_the_clearance_layer_bends_and_slows_the_plan_the_car_will_follow():
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
    ref_legacy, u_legacy = _run(tr)
    arm = pn.install_clearance_arm(tr, "clearance", "cpu", SPEC)
    assert arm is not None and arm.cspec.margin == pytest.approx(cl.MARGIN)
    arm.update_scan(_blocked_scan())
    ref_arm, _ = _run(tr)
    # the reference is what the iLQR follows and what the attribution script measures: it has to
    # have moved, and the speed along it has to have come down
    assert not torch.equal(ref_arm, ref_legacy)
    assert float(ref_arm[..., -1].max()) < float(ref_legacy[..., -1].max())
    arm.release()
    ref_back, u_back = _run(tr)
    assert torch.allclose(ref_back, ref_legacy) and torch.allclose(u_back, u_legacy)


def test_an_empty_scan_is_a_no_op_so_the_arm_does_nothing_before_it_has_been_fed():
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
    ref_legacy, _ = _run(tr)
    arm = pn.install_clearance_arm(tr, "clearance", "cpu", SPEC)
    ref_empty, _ = _run(tr)                  # never fed: the buffer is all "no return"
    assert torch.equal(ref_empty, ref_legacy)
    arm.release()


def test_the_arm_names_this_node_will_install_and_the_ones_it_refuses():
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
    assert pn.install_clearance_arm(tr, "legacy", "cpu", SPEC) is None
    assert pn.install_clearance_arm(tr, "fixed_low", "cpu", SPEC) is None
    assert pn.install_clearance_arm(None, "clearance", "cpu", SPEC) is None
    for bad in ("estimated", "oracle", "fixed_low+tcs", "clearance+fixed_low", "CLEARANCE"):
        with pytest.raises(ValueError):
            pn.install_clearance_arm(tr, bad, "cpu", SPEC)
        with pytest.raises(ValueError):
            pn.install_grip_arm(tr, bad, "cpu")
    assert pn.split_deployable("fixed_low+clearance") == ("fixed_low", True)
    assert pn.split_deployable("clearance") == ("legacy", True)


def test_fixed_low_and_clearance_compose_on_the_node_in_either_order():
    """`install_grip_arm` binds `_solver`, `install_clearance_arm` binds `_plan_hook`. On the car
    they are set up one after the other in `PolicyNode.__init__`, and which line comes first must
    not be able to change what is published."""
    scan = _blocked_scan(x=2.0, y=0.25, r=0.25, half_width=1.0)
    cmds = []
    for order in (("grip", "clear"), ("clear", "grip")):
        tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
        arms = []
        for which in order:
            if which == "grip":
                arms.append(pn.install_grip_arm(tr, "fixed_low+clearance", "cpu"))
            else:
                a = pn.install_clearance_arm(tr, "fixed_low+clearance", "cpu", SPEC)
                a.update_scan(scan)
                arms.append(a)
        tr.reset(torch.zeros(1, dtype=torch.long))
        cmds.append(tr(_fast_straight_plan(), torch.tensor([3.0]), torch.tensor([4.5]), None,
                       delay=torch.tensor([0.035])).clone())
        for a in arms:
            a.release()
        assert tr._solver is None and tr._plan_hook is None
    assert torch.equal(cmds[0], cmds[1]), cmds


def test_the_margin_parameter_reaches_the_spec():
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
    arm = pn.install_clearance_arm(tr, "clearance", "cpu", SPEC, margin=0.35)
    assert arm.cspec.margin == pytest.approx(0.35)
    arm.release()


def test_a_scanner_with_a_different_window_gets_the_bearings_it_actually_has():
    """`set_angles` is what stops a 240 deg driver being read as a 270 deg one, which would place
    every return at a bearing it does not have and leave nothing downstream looking wrong."""
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, 10.0)
    arm = pn.install_clearance_arm(tr, "clearance", "cpu", SPEC)
    assert float(arm.angles[0]) == pytest.approx(-pn.LIDAR_FOV / 2)
    narrow = torch.linspace(-math.pi * 2 / 3, math.pi * 2 / 3, SPEC.n_beams)
    arm.set_angles(narrow)
    assert float(arm.angles[-1]) == pytest.approx(math.pi * 2 / 3)
    assert arm.scan.shape == (1, SPEC.n_beams)
    arm.release()
