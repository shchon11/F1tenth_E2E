"""A v2 checkpoint on the car's path: `/drive` from the real ROS graph, bit for bit the simulator's.

The redesign's actor (`learn.model_v2`) reads observation contract v2 and plans in WP4. On the car
that means: `policy_node` builds `ObsBuilderV2` from `/scan`, `/odom` and the IMU (the plans it
issued itself in the history ring), feeds the grip dial as an input column and carries the GRU state
from one scan callback to the next; `controller_node` reads the WP4 contract off the same checkpoint
and tracks the plan with the WP4 tracker. This file holds that path to the simulator's own:

* **in process** -- `F1VecEnv` on contract v2, one car, the checkpoint's `act` with its hidden state
  carried, the env's own tracker: the plan and the (steer, speed) command of every step, and the
  messages the simulator's bridges would have published for it (`F1VecEnv.message_inputs`);
* **through the graph** -- those messages as real `LaserScan` / `Imu` / `Odometry` through the real
  `PolicyNode` -> `Plan` -> `ControllerNode` (`graph_replay.run_graph`: real constructors, a real
  `rclpy` context and parameter file, only the transport replaced);

and every plan and every `/drive` must be the same bits, for a few hundred steps of closed-loop
driving. Two things the simulator knows and the car cannot are held out, and said so: the env tells
its tracker when the car touches something (`PlanTracker.contact`, the back-out gate -- the car has
no contact sensor), so it is zeroed before every in-process step; and the env's tracker latency is
its per-episode calibration draw, which the controller is given as its `cmd_delay` (on the car it is
the measured latency). The run is one episode (soft contacts, no time limit), so neither the env's
reset nor the node's `/f1sim/reset` is inside the window; the reset rule is held separately below.

The checkpoint is generated (deterministic, any machine); `F1SIM_V2_PARITY_CKPT=<path.pt>` runs the
same comparison on a trained one (UNIT 5a ran it on UNIT 4's smoke checkpoints). The generated one is
the contract WITH the issued actions (`ObsSpecV2.ego_actions` True, every v2 checkpoint before
2026-10-15), named explicitly: it is the one whose ego reads the node's own plans back from the ring.
The contract without them (the copycat study's arm A, the default since 2026-10-15) runs the same
comparison on a generated checkpoint of its own.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("rclpy")
pytest.importorskip("f1sim_interfaces.msg", reason="f1sim_interfaces is not built (colcon)")
torch = pytest.importorskip("torch")

import graph_replay as gr  # noqa: E402
from v2_fixtures import CAP, generated_v2_checkpoint  # noqa: E402

TRACK = "gen:competition:0"
STEPS = int(os.environ.get("F1SIM_V2_PARITY_STEPS", "300"))
DIAL = 0.9                       # inside the trained range; the node refuses a dial checkpoint without one


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    given = os.environ.get("F1SIM_V2_PARITY_CKPT")
    if given:
        return given
    # the issued plans in the ego (see the module doc); `checkpoint_no_actions` is the default contract
    return generated_v2_checkpoint(str(tmp_path_factory.mktemp("v2parity") / "student_latest.pt"),
                                   ego_actions=True)


@pytest.fixture(scope="module")
def checkpoint_no_actions(tmp_path_factory):
    return generated_v2_checkpoint(str(tmp_path_factory.mktemp("v2parity_a") / "student_latest.pt"),
                                   ego_actions=False)


def in_process(ckpt: str, steps: int, seed: int = 3):
    """Drive the simulator with the checkpoint, the way every in-process path does (`model.act`
    with the hidden state carried, the dial's `c`, the env's own WP4 tracker): per step, the
    messages of the observation the plan was made from, the plan, and the tracker's command."""
    from f1sim import Config
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    from f1sim.learn import conditioning as C
    from f1sim.learn.memory import Hidden
    from f1sim.learn.model import load_checkpoint
    model, extra = load_checkpoint(ckpt, "cpu")
    model.eval()
    spec = model.obs_spec
    tracks, _ = common.load_tracks([TRACK], racelines=False, drop_infeasible=False)
    cfg = Config(); cfg.rand.enabled = False; cfg.sim.compile = False
    ec = EnvConfig(action_mode="plan", race_size=1, speed_cap=CAP, obs_version=2, scan_stack=spec.n_frames,
                   scan_stride=spec.frame_stride, hist_len=spec.hist_len, hist_stride=spec.hist_stride,
                   v_max_policy=spec.v_max, imu_gyro_scale=spec.gyro_scale, ego_actions=spec.ego_actions,
                   compile_tracker=False,
                   collision_mode="soft", max_steps=10 ** 6, laps=10 ** 6, **common.plan_output_of(extra))
    env = common.make_env(tracks, 1, "cpu", ec, cfg=cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    c = C.mu_to_c(torch.tensor([DIAL]), C.CondSpec.from_meta(model.meta["cond"]))
    h = Hidden()
    rows = []
    for _t in range(steps):
        mi = env.message_inputs(0)
        with torch.no_grad():
            a, _lp, h = model.act(obs["scan"], obs["ego"], deterministic=True, c=c, h=h)
        # the car has no contact sensor: the simulator-only back-out gate is held out (module doc)
        env.tracker.contact = torch.zeros_like(env.tracker.contact)
        obs, _r, term, trunc, _info = env.step(a)
        rows.append({"mi": mi, "plan": a[0].numpy().copy(), "cmd": env.last_cmd_raw[0].numpy().copy(),
                     "contact": bool(env.last_result.collision[0])})
        if bool(term[0] | trunc[0]):
            break
    return {"rows": rows, "env": env, "delay": float(env.tracker_delay[0]),
            "wheelbase": float(env.cfg.vehicle.lf + env.cfg.vehicle.lr), "s_max": float(env.cfg.vehicle.s_max),
            "fov": float(env.cfg.lidar.fov), "range_max": float(env.range_max), "model": model}


def _quat(roll: float, pitch: float):
    """(w, x, y, z) of a roll / pitch attitude with zero yaw -- how the bridges publish `imu_att`."""
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return (cr * cp, sr * cp, cr * sp, -sr * sp)


def write_sequence(ref: dict, path: str, cov0: float = 0.0) -> "gr.Sequence":
    """The in-process run's messages as a `graph_replay` sequence: one scan per control step, its
    IMU samples and one odometry sample before it, 25 ms apart. `cov0` -1 marks every IMU sample
    "no orientation estimate" (`deploy.attitude_from_orientation` refuses it)."""
    rows = ref["rows"]
    imu = [r["mi"].imu for r in rows]
    ptr = np.concatenate([[0], np.cumsum([len(x) for x in imu])]).astype(np.int64)
    quat = np.concatenate([np.tile(_quat(*r["mi"].att), (len(r["mi"].imu), 1)) for r in rows])
    np.savez(path, ranges=np.stack([r["mi"].ranges for r in rows]).astype(np.float32),
             range_max=np.float64(ref["range_max"]), angle_min=np.float64(-ref["fov"] / 2),
             angle_max=np.float64(ref["fov"] / 2), imu=np.concatenate(imu).astype(np.float64),
             imu_quat=quat.astype(np.float64), imu_cov0=np.full(int(ptr[-1]), float(cov0)), imu_ptr=ptr,
             odom_v=np.array([r["mi"].speed for r in rows], np.float64),
             odom_ptr=np.arange(len(rows) + 1, dtype=np.int64),
             # from 1 s: a zero header stamp is "not a time" to the controller (`stamp_key`), and the
             # first plan would be paired by recency instead of by its scan's stamp
             t_scan=1.0 + np.arange(len(rows), dtype=np.float64) * 0.025, source=np.array("v2_parity"))
    return gr.Sequence(path)


def graph_params(ckpt: str, ref: dict) -> dict:
    """Both nodes' parameters: the checkpoint and the dial, and the controller configured as the env's
    tracker is -- this car's wheelbase and steering limit, its calibrated latency, no residual servo
    calibration (the env's command is compared before its own simulated calibration)."""
    return {"checkpoint": ckpt, "device": "cpu", "speed_cap": CAP, "sensor_timeout": 0.25, "enabled": True,
            "dial": DIAL, "steer_max": ref["s_max"], "controller": "legacy", "plan_timeout": 0.25,
            "traction": "off", "wheelbase": ref["wheelbase"], "cmd_delay": ref["delay"], "steer_bias": 0.0,
            "steer_gain": 1.0, "speed_gain": 1.0, "viz": False}


@pytest.mark.slow
def test_a_v2_checkpoint_drives_the_graph_bit_identically_to_the_simulator(checkpoint, tmp_path):
    _graph_parity(checkpoint, tmp_path)


@pytest.mark.slow
def test_a_v2_checkpoint_without_issued_actions_drives_the_graph_bit_identically(checkpoint_no_actions, tmp_path):
    """The node builds the 42-wide ego of the copycat study's arm A from the same three topics."""
    ref = _graph_parity(checkpoint_no_actions, tmp_path)
    assert ref["model"].obs_spec.ego_dim == 42 and ref["env"].ecfg.ego_actions is False


def _graph_parity(checkpoint, tmp_path) -> dict:
    ref = in_process(checkpoint, STEPS)
    rows = ref["rows"]
    assert len(rows) == STEPS, "the episode ended inside the window (see the module doc)"
    seq = write_sequence(ref, str(tmp_path / "v2_seq.npz"))
    keep = {}
    run = gr.run_graph(seq, len(rows), graph_params(checkpoint, ref), tmp_path, keep=keep)
    policy, controller = keep["policy"], keep["controller"]
    try:
        # the nodes read the checkpoint as what it is
        assert policy.v2 and type(policy.obs).__name__ == "ObsBuilderV2"
        assert policy.policy_state.hidden is not None and policy.policy_state.hidden.actor is not None
        assert controller.contract["speed_mode"] == "wp4" and not controller.attitude_required
        assert controller.tracker.spec == ref["env"].tracker.spec, "the controller built another tracker"
        # every frame planned (no inhibit: v2 has no attitude input), every plan was tracked, and
        # against its own scan's measurement (paired by the scan stamp, never by recency)
        assert len(run.plans) == len(rows) and len(run.tracked) == len(rows) == len(run.commands)
        assert controller._plan_unmatched == 0
        plans = np.array(run.plans, dtype=np.float32)
        want = np.stack([r["plan"] for r in rows])
        bad = np.nonzero(~(plans == want).all(1))[0]
        assert bad.size == 0, (f"plan differs first at step {bad[0]} of {len(rows)}: "
                               f"max |diff| {np.abs(plans - want).max():.3e}")
        got = np.array(run.tracked_commands(), dtype=np.float32)
        s_max = np.float32(ref["s_max"])
        want_cmd = np.stack([[np.clip(r["cmd"][0], -s_max, s_max), r["cmd"][1]] for r in rows]).astype(np.float32)
        bad = np.nonzero(~(got == want_cmd).all(1))[0]
        assert bad.size == 0, (f"/drive differs first at step {bad[0]} of {len(rows)}: "
                               f"max |diff| {np.abs(got - want_cmd).max():.3e}")
        # and the comparison covered real driving: the car moved, steered and the plans varied
        speeds = np.array([r["mi"].speed for r in rows])
        assert speeds.max() > 1.0 and np.abs(got[:, 0]).max() > 0.01 and plans.std(0).max() > 1e-3
        print(f"v2 graph parity: {len(rows)} steps bit-identical (plans and /drive); speed up to "
              f"{speeds.max():.2f} m/s, |steer| up to {np.abs(got[:, 0]).max():.3f} rad, "
              f"{sum(r['contact'] for r in rows)} contact steps")
        assert policy.obs.spec == ref["model"].obs_spec, "the node built another observation contract"
    finally:
        policy.destroy_node(); controller.destroy_node()
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    return ref


@pytest.mark.slow
def test_the_v2_node_clears_its_memory_on_reset_and_on_a_sensor_gap(checkpoint, tmp_path):
    """The GRU state and the history ring are the episode's: carried between callbacks (the plans of
    a second pass over the same frames differ while it is carried), and after `/f1sim/reset` the node
    plans exactly what a fresh node plans -- the same bits. `on_reset` and `_resume` share
    `_clear_memory`, so the sensor-gap path is the same code."""
    from std_msgs.msg import Empty as EmptyMsg
    ref = in_process(checkpoint, 40)
    seq = write_sequence(ref, str(tmp_path / "v2_short.npz"))
    pf = gr.write_params(str(tmp_path / "policy.yaml"),
                         {"f1sim_policy": {k: v for k, v in graph_params(checkpoint, ref).items()
                                           if k in gr.POLICY_PARAMS}})
    from f1sim_ros.policy_node import PolicyNode
    with gr.Context(pf):
        node = PolicyNode()
        clock = gr.Clock()
        gr._drive_clock(node, clock)
        plans = gr.Collector()
        node.pub_plan.publish = plans
        node.pub_state.publish = gr.Collector()
        node.pub_perception.publish = gr.Collector()
        gr._feed(seq, 40, clock, [node], scan_order=[node])
        first = [list(m.plan) for m in plans.msgs]
        assert node.policy_state.hidden.actor is not None and float(node.policy_state.hidden.actor.abs().sum()) > 0
        # carried: the same frames again, without a reset, are planned differently
        gr._feed(seq, 40, clock, [node], scan_order=[node])
        carried = [list(m.plan) for m in plans.msgs[40:]]
        assert carried != first
        node.on_reset(EmptyMsg())
        assert node.policy_state.hidden.actor is None and node.memory_cleared_reason == "reset"
        gr._feed(seq, 40, clock, [node], scan_order=[node])
        after = [list(m.plan) for m in plans.msgs[80:]]
        assert after == first, "after a reset the node must plan exactly what a fresh node plans"
        node.destroy_node()


@pytest.mark.slow
def test_a_v2_graph_drives_without_an_orientation_and_a_v1_rule_would_not(checkpoint, tmp_path):
    """Roll and pitch are not a v2 input (the quaternion correlated 0.14 with the truth, and five of
    thirteen car recordings swung it past 40 deg), so neither node waits for one: with every IMU
    sample marked "no orientation" the graph plans and drives every frame, and the plans and the
    commands are the very bits of the run that had one. The v1 rule, on the same intake, would have
    stopped on every frame."""
    from f1sim_ros.deploy import SensorIntake
    ref = in_process(checkpoint, 40)
    params = graph_params(checkpoint, ref)
    with_att = gr.run_graph(write_sequence(ref, str(tmp_path / "att.npz")), 40, params, tmp_path)
    without = gr.run_graph(write_sequence(ref, str(tmp_path / "noatt.npz"), cov0=-1.0), 40, params, tmp_path)
    assert len(without.plans) == 40 and len(without.tracked) == 40
    assert without.plans == with_att.plans and without.commands == with_att.commands
    intake = SensorIntake(lambda: 1.0, None, 0.25)
    intake.t_imu_mean = intake.t_odom = 1.0                    # fresh IMU and odometry, no orientation ever
    assert intake.stale(1.0) == ["attitude"] and intake.stale(1.0, attitude=False) == []
