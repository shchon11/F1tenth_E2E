"""The controller node's plan-controller arm: `fixed_low` must bind the tracker's speed reference on
the same `PlanTracker` object the car drives through, and `release()` must restore it exactly.

The installers moved from `policy_node` to `f1sim_ros/deploy.py` with the split (the policy has no
tracker to install anything on any more), so this file follows them. The last test is the one the
move needs: on a fixed plan, the arms the shared module installs produce a command **bit-identical**
to the one the old monolithic node's own installers produced.
"""
import importlib.util
import os

import pytest

pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")

import f1sim_ros.deploy as pn                            # noqa: E402
from f1sim.learn.obs import ObsSpec                      # noqa: E402
from f1sim.mpc import PlanTracker                        # noqa: E402


def monolith():
    """The node as it shipped at `0b78111`, imported for its installers alone. `test_graph_parity`
    pins the file's hash; this uses it as the reference for one fixed plan."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference",
                        "monolithic_policy_node.py")
    spec = importlib.util.spec_from_file_location("monolithic_policy_node", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


# ==================================================================== against the old node
ARM_SPEC = ObsSpec(n_beams=1081, act_dim=8)


def _plan():
    """A plan that makes every layer do something: a hard turn, both speeds at the top."""
    return torch.tensor([[0.85, 0.60, 0.30, -0.20, -0.55, -0.80, 1.0, 0.8]])


def _blocked_scan():
    """A wall 1.5 m ahead across the whole window, so the clearance layer has to act."""
    r = torch.full((1, ARM_SPEC.n_beams), 0.15)      # 1.5 m / range_max
    return r


def _command(install, arm, feed_scan):
    tr = PlanTracker(1, "cpu", 0.3302, 0.4189, ARM_SPEC.v_max)
    grip = install["grip"](tr, arm, "cpu")
    clear = install["clearance"](tr, arm, "cpu", ARM_SPEC)
    if clear is not None and feed_scan:
        clear.update_scan(_blocked_scan())
    tr.reset(torch.zeros(1, dtype=torch.long))
    out = []
    for v in (1.0, 3.0, 5.0):
        out.append(tr(_plan(), torch.tensor([v]), torch.tensor([6.0]), torch.tensor([0.3]),
                      delay=torch.tensor([0.035])).clone())
    for a in (grip, clear):
        if a is not None:
            a.release()
    return torch.cat(out)


@pytest.mark.parametrize("arm", ["legacy", "fixed_low", "clearance", "fixed_low+clearance"])
@pytest.mark.parametrize("feed_scan", [True, False])
def test_the_shared_installers_are_the_old_nodes_installers(arm, feed_scan):
    """Bit-exact, on a fixed plan, for every arm the graph can run.

    `f1sim_ros/deploy.py` is a move, not a rewrite -- but "I only moved it" is a claim, and a
    different `GripSpec` or a different beam geometry would show up as a slightly different command
    and nowhere else.
    """
    mod = monolith()
    ref = _command({"grip": mod.install_grip_arm, "clearance": mod.install_clearance_arm},
                   arm, feed_scan)
    got = _command({"grip": pn.install_grip_arm, "clearance": pn.install_clearance_arm},
                   arm, feed_scan)
    assert torch.equal(got, ref), (arm, feed_scan, (got - ref).abs().max().item())


def test_the_geometry_constants_are_the_old_nodes_constants():
    """The clearance grid is built from these two numbers. A silent change to either places every
    return at a bearing it does not have, or the sensor where it is not."""
    mod = monolith()
    assert pn.LIDAR_FOV == mod.LIDAR_FOV
    assert pn.LIDAR_MOUNT_X == mod.LIDAR_MOUNT_X
    assert pn.DEPLOYABLE_ARMS == mod.DEPLOYABLE_ARMS
