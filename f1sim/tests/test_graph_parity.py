"""The load-bearing test of the split: the graph publishes the `/drive` the old node published.

`policy_node` used to be one node -- observation, actor, iLQR tracker, grip and clearance arms,
traction guard, `/drive`. It is two now, with an eight-float plan on a topic between them. That is
only a refactor if the car cannot tell, so:

* `f1sim/tests/reference/monolithic_policy_node.py` is the old node, byte for byte as it was at
  `0b78111`, kept for no other purpose. `test_the_reference_is_the_node_that_shipped` pins it.
* every arm the graph can run is replayed through both, over a **simulator bag** and a **real car
  bag**, and every command has to match to 1e-5 (they match exactly).
* the same plans are replayed through a bare `PlanTracker` with the arm installed by the
  TRAINING-side installer (`learn/grip_runtime`, `learn/clearance`), which is the path the
  benchmark scores. The controller has to add nothing to it.

Sequences: `scripts/make_graph_fixture.py`. Harness: `graph_replay.py`.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "f1sim_ros"))

pytest.importorskip("rclpy")
pytest.importorskip(
    "f1sim_interfaces.msg",
    reason="f1sim_interfaces is not built. It is a rosidl package, so it needs colcon:\n"
           "  mkdir -p ws/src && ln -s <repo>/f1sim_interfaces ws/src/ && \\\n"
           "  (cd ws && colcon build --packages-select f1sim_interfaces) && \\\n"
           "  source ws/install/setup.bash")
torch = pytest.importorskip("torch")

import graph_replay as gr                                   # noqa: E402
from f1sim.learn.model import ActorCritic, save_checkpoint  # noqa: E402
from f1sim.learn.obs import ObsSpec                         # noqa: E402

#: The frozen reference, as it was at the base commit of this branch. If this changes, the parity
#: claim is against a different node and has to be re-derived rather than re-blessed.
MONOLITH_SHA256 = "4acf202c2240d784f2464c1d7ddd82b5ca09b7c272cf2484416a1376b7d73d3d"

#: 1081 beams because both fixtures were recorded at 1081, and `act_dim=8` because a plan
#: checkpoint is the only kind that has a controller at all. Everything else is as small as the
#: model allows, so the file is a few MB and every test in this module can share it.
SPEC = ObsSpec(n_beams=1081, scan_stack=2, scan_stride=1, action_history=2, act_dim=8,
               hist_len=0, range_max=10.0, v_max=10.0)

ARMS = ("legacy", "fixed_low", "clearance", "fixed_low+clearance")
SEQUENCES = {"sim": gr.SIM_SEQ, "real": gr.REAL_SEQ}
FRAMES = 60
TOL = 1e-5                              # the contract's bound; the measured worst case is 0.0


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A deterministic plan checkpoint. Seeded inside a forked RNG, so the weights are the same on
    every run and the suite's global generator is left exactly as it was found."""
    meta = dict(n_stack=SPEC.scan_stack, n_beams=SPEC.n_beams, proprio_dim=SPEC.proprio_dim,
                priv_dim=9, act_dim=SPEC.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(20260915)
        model = ActorCritic(**meta)
    path = str(tmp_path_factory.mktemp("graph") / "plan.pt")
    save_checkpoint(path, model, {"spec": SPEC.__dict__.copy(), "phase": "ppo"})
    return path


def _plan_frames(seq, run):
    """A mask over the replayed frames: True where that frame produced a plan.

    There is exactly one `/drive` per frame -- a tracked command, or the zero the controller
    publishes while an input is stale -- so the mask is over commands. The graph only inhibits at
    the start of these sequences (they are continuous recordings, and the simulator bag's first
    IMU sample is one the bridge marked "orientation not measured here"), which makes the tracked
    commands a contiguous tail. That is asserted rather than assumed: a gap appearing mid-sequence
    would otherwise shift the comparison by one and still pass.
    """
    n = len(run.plans)
    assert n == len(run.tracked), (n, len(run.tracked))
    first = len(run.commands) - n
    assert run.tracked == list(range(first, len(run.commands))), (
        "the tracked commands are not a contiguous tail: the graph inhibited mid-sequence, and "
        "this alignment no longer holds")
    return [i >= first for i in range(len(run.commands))]


def params(checkpoint, arm="fixed_low", traction="off", **kw):
    p = {"checkpoint": checkpoint, "device": "cpu", "speed_cap": 4.0, "sensor_timeout": 0.25,
         "plan_timeout": 0.25, "controller": arm, "traction": traction, "wheelbase": 0.3302,
         "cmd_delay": 0.035, "steer_max": 0.4189, "enabled": True}
    p.update(kw)
    return p


@pytest.fixture(scope="module")
def sequences():
    out = {}
    for name, path in SEQUENCES.items():
        if not os.path.exists(path):
            pytest.skip(f"{path} is missing; rebuild it with scripts/make_graph_fixture.py")
        out[name] = gr.Sequence(path)
    return out


def test_the_reference_is_the_node_that_shipped():
    """The parity claim is against a specific file. Pinning its hash is what stops it drifting into
    "the split matches a copy of the split"."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference",
                        "monolithic_policy_node.py")
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert got == MONOLITH_SHA256, (
        f"{path} is no longer `git show 0b78111:f1sim_ros/f1sim_ros/policy_node.py`. It is the "
        f"reference the graph is measured against and must not be edited; if the reference really "
        f"has to move, re-derive the parity numbers and update MONOLITH_SHA256 deliberately.")


@pytest.mark.parametrize("seq_name", sorted(SEQUENCES))
@pytest.mark.parametrize("arm", ARMS)
def test_the_split_graph_publishes_the_monolithic_nodes_drive(arm, seq_name, sequences,
                                                              checkpoint, tmp_path):
    seq = sequences[seq_name]
    p = params(checkpoint, arm=arm)
    ref = gr.run_monolith(seq, FRAMES, p, tmp_path)
    run = gr.run_graph(seq, FRAMES, p, tmp_path)
    assert len(ref) == FRAMES, f"the reference published {len(ref)} commands for {FRAMES} scans"
    assert len(run.commands) == len(ref), (len(run.commands), len(ref))
    d = np.abs(np.asarray(run.commands) - np.asarray(ref))
    assert d.max() <= TOL, (f"{arm} on {seq_name}: worst steer {d[:, 0].max():.3e} rad, "
                            f"worst speed {d[:, 1].max():.3e} m/s")


@pytest.mark.parametrize("seq_name", sorted(SEQUENCES))
def test_the_traction_guard_is_the_same_guard_on_the_same_stream(seq_name, sequences, checkpoint,
                                                                 tmp_path):
    """The one layer that moved between nodes AND reads `/odom` at its own rate.

    The monolith fed it from `on_odom` and shaped in `on_scan`; the controller does both, off its
    own subscription. A guard fed a different sample stream would shape differently under a lock,
    and the whole point of it is what it does under one.
    """
    seq = sequences[seq_name]
    p = params(checkpoint, arm="fixed_low", traction="on")
    ref = gr.run_monolith(seq, FRAMES, p, tmp_path)
    run = gr.run_graph(seq, FRAMES, p, tmp_path)
    d = np.abs(np.asarray(run.commands) - np.asarray(ref))
    assert d.max() <= TOL, (f"traction on, {seq_name}: worst steer {d[:, 0].max():.3e}, "
                            f"worst speed {d[:, 1].max():.3e}")


@pytest.mark.parametrize("arm", ARMS)
def test_the_controller_is_the_in_process_tracker_the_benchmark_scores(arm, sequences, checkpoint,
                                                                       tmp_path):
    """The other half of the claim: the controller adds nothing to `mpc.PlanTracker`.

    The arms here are installed by the TRAINING-side code -- `learn.grip_runtime` /
    `learn.grip_control.GripMPC` and `learn.clearance.ClearanceArm` -- which is what
    `benchmark/model_adapter.prepare_cell` installs on `env.tracker`. The measured speed and yaw
    rate are recomputed from the fixture rather than read out of the node, so a bug in the node's
    IMU intake shows up here as a mismatch instead of being cancelled out.
    """
    from f1sim.learn import clearance as cl
    from f1sim.learn import grip_control as gc
    from f1sim.mpc import PlanTracker
    from f1sim_ros.deploy import LIDAR_FOV, LIDAR_MOUNT_X
    from f1sim.learn.obs import norm_scan, resample_ranges

    seq = sequences["sim"]
    p = params(checkpoint, arm=arm)
    run = gr.run_graph(seq, FRAMES, p, tmp_path)
    plans = run.plans
    # The frames a plan was published for: the graph inhibits until the first usable attitude, and
    # the sim bag's first IMU sample is one the bridge marked "orientation not measured here".
    frames_with_plans = [f for f, keep in zip(seq.frames(FRAMES), _plan_frames(seq, run)) if keep]

    tracker = PlanTracker(1, "cpu", p["wheelbase"], p["steer_max"], SPEC.v_max)
    base, clear = ("legacy", False) if arm == "legacy" else (
        ("legacy", True) if arm == "clearance" else
        ("fixed_low", arm.endswith("clearance")))
    grip = clearance = None
    if base == "fixed_low":
        grip = gc.GripMPC(tracker, gc.GripSpec(mode="fixed", mu_fixed=gc.MU_FIXED_LOW).validate(),
                          1, torch.device("cpu"), tracker.wb, tracker.s_max,
                          tracker.v_max).install(graph=False)
    if clear:
        clearance = cl.ClearanceArm(tracker, cl.ClearanceSpec().validate(), 1, torch.device("cpu"),
                                    SPEC.v_max, cl.beam_angles(SPEC.n_beams, LIDAR_FOV),
                                    SPEC.range_max, mount_x=LIDAR_MOUNT_X).install()
    delay = torch.tensor([p["cmd_delay"]])
    ref = []
    for f, plan in zip(frames_with_plans, plans):
        v = float(f.odom_v[-1])                     # the /odom sample this scan's snapshot took
        yaw = float(np.asarray(f.imu)[:, 2].mean())  # the mean gyro-z of this scan's IMU samples
        if clearance is not None:
            r = resample_ranges(f.ranges, f.range_max, SPEC.n_beams)
            clearance.update_scan(norm_scan(torch.as_tensor(r, dtype=torch.float32)[None],
                                            SPEC.range_max))
        cmd = tracker(torch.tensor([plan]), torch.tensor([v]), torch.tensor([p["speed_cap"]]),
                      torch.tensor([yaw]), delay=delay)[0]
        ref.append((max(-p["steer_max"], min(p["steer_max"], float(cmd[0]))), float(cmd[1])))
    for a in (grip, clearance):
        if a is not None:
            a.release()
    assert len(plans) >= FRAMES - 2, f"only {len(plans)} plans over {FRAMES} frames"
    d = np.abs(np.asarray(run.tracked_commands()) - np.asarray(ref))
    assert d.max() <= TOL, (f"{arm}: worst steer {d[:, 0].max():.3e} rad, "
                            f"worst speed {d[:, 1].max():.3e} m/s")
