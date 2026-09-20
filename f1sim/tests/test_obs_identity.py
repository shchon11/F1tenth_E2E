"""One observation builder: the batched env and the ROS node must produce the same tensors.

The observation is the contract between training and the car. Until now it was shared by
convention -- `learn/obs.py` held `ObsBuilder` for the node, and `gym_env` wrote the same
arithmetic out again for the batch -- and nothing could tell you if the two drifted. A drift there
does not crash: the policy simply drives on an input it was never trained on.

So the arithmetic is now one implementation (`obs.norm_scan` / `norm_imu` / `norm_att` /
`norm_speed` / `resample_ranges`, called by both), and this is the test that the composition agrees
too. `F1VecEnv.message_inputs` states a simulator step as the fields the graph's topics carry --
`LaserScan.ranges`, the `Imu` samples, `Odometry`'s wheel speed -- and `ObsBuilder.build_message`
turns those into an observation exactly as `policy_node` does. Nothing privileged crosses: if the
env had to invent a field for this, that field would be one the car cannot have.

Three checkpoint kinds, because the paths differ after the builder: a legacy feedforward one, a
recurrent one, and one with the extra scan channels. The last two run through
`learn.memory.PolicyRuntime`, which is shared, and this checks that feeding it the node's scan and
the env's scan lands in the same place -- including the decayed occupancy channel, which is state.

**Why the comparison starts a few steps in.** `env.reset()` fills the scan stack with the spawn
scan and then rolls one stepped scan into it, while a node starting mid-stream fills its stack with
the first scan it sees. Those are different histories of the same run, by construction, and neither
is wrong. After `(scan_stack - 1) * scan_stride` steps every frame in both stacks is a stepped
scan, and from there identity is exact -- which is the claim that matters, because it is the state
a node reaches a second after it starts.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from f1sim import Config                                      # noqa: E402
from f1sim.gym_env import EnvConfig                           # noqa: E402
from f1sim.learn import common                                # noqa: E402
from f1sim.learn.memory import memory_spec, runtime_for       # noqa: E402
from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint  # noqa: E402
from f1sim.learn.obs import ObsBuilder, flatten_obs           # noqa: E402

STEPS = 14
SPEED_CAP = 4.0


def _env(hist_len=4, hist_stride=2, scan_stack=3):
    tracks, rls = common.load_tracks(["gen:competition:0"], racelines=True)
    cfg = Config(); cfg.rand.enabled = False; cfg.sim.compile = False
    ec = EnvConfig(action_mode="plan", speed_cap=SPEED_CAP, hist_len=hist_len,
                   hist_stride=hist_stride, scan_stack=scan_stack, compile_tracker=False)
    env = common.make_env(tracks, 2, "cpu", ec, cfg=cfg, seed=1)
    return env, common.make_teacher(rls, env)


def _model(spec, kind, tmp_path):
    """A checkpoint of each kind, deterministic: the actions have to be comparable run to run."""
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=9, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(1515)
        ff = ActorCritic(**meta)
        if kind == "legacy":
            return ff.eval()
        base = str(tmp_path / f"{kind}.pt")
        save_checkpoint(base, ff, {"spec": {}})
        m, _e, _f = load_for_memory(
            base, "cpu",
            memory_spec(hidden_size=16) if kind == "memory" else None,
            scan_channels={"channels": ["memory", "edges"]} if kind == "channels" else None)
    return m.eval()


@pytest.mark.parametrize("kind", ["legacy", "memory", "channels"])
def test_the_env_and_the_node_build_the_same_observation(kind, tmp_path):
    env, teacher = _env()
    spec = common.obs_spec(env)
    model = _model(spec, kind, tmp_path)

    builder = ObsBuilder(spec, "cpu")
    env_rt = runtime_for(model, batch=1, device="cpu")
    node_rt = runtime_for(model, batch=1, device="cpu")

    obs, _ = env.reset(seed=1)
    # Seed the node's action history from the env's: the node has been driving, and its two
    # previous actions are the env's two previous actions. Nothing else is copied across.
    builder.push_action(env.act_hist[0, 1].cpu()); builder.push_action(env.act_hist[0, 0].cpu())
    builder._pending = None                       # the seeding pushes are not history rows
    builder.build_message(env.message_inputs(0))

    warm = (spec.scan_stack - 1) * spec.scan_stride
    worst = {"scan": 0.0, "proprio": 0.0, "action": 0.0}
    compared = 0
    for t in range(STEPS):
        a = env.teacher_label(teacher)
        builder.push_action(a[0].cpu())
        obs, _r, _te, _tr, _i = env.step(a)

        # The node's side: nothing but the fields the topics carry.
        scan_node, pro_node = builder.build_message(env.message_inputs(0))
        # The env's side: its own observation, row 0.
        scan_env, pro_env = flatten_obs({k: v[0:1] for k, v in obs.items()})

        if t < warm:
            continue
        compared += 1
        worst["scan"] = max(worst["scan"], (scan_node - scan_env).abs().max().item())
        worst["proprio"] = max(worst["proprio"], (pro_node - pro_env).abs().max().item())
        with torch.no_grad():
            a_env, _, env_rt.hidden = model.act(env_rt.observe(scan_env), pro_env,
                                                deterministic=True, h=env_rt.hidden)
            a_node, _, node_rt.hidden = model.act(node_rt.observe(scan_node), pro_node,
                                                  deterministic=True, h=node_rt.hidden)
        worst["action"] = max(worst["action"], (a_env - a_node).abs().max().item())

    assert compared >= STEPS - warm - 1, compared
    assert worst["scan"] == 0.0, f"{kind}: scan differs by {worst['scan']:.3e}"
    assert worst["proprio"] == 0.0, f"{kind}: proprio differs by {worst['proprio']:.3e}"
    assert worst["action"] == 0.0, f"{kind}: the actions differ by {worst['action']:.3e}"
    if node_rt.scan is not None:
        # The decayed occupancy channel is episode STATE, so agreeing on one step is not enough:
        # a divergence in it accumulates and this is where it would show.
        assert torch.equal(node_rt.scan.mem, env_rt.scan.mem)
    if node_rt.hidden is not None and node_rt.hidden.actor is not None:
        assert torch.equal(node_rt.hidden.actor, env_rt.hidden.actor)


def test_message_inputs_carries_nothing_the_car_does_not_publish():
    """The type is the point: if the env had to state a field the node cannot get, the observation
    contract would be a sim-to-real gap with a dataclass around it."""
    from f1sim.learn.obs import MessageInputs
    env, _ = _env()
    env.reset(seed=1)
    mi = env.message_inputs(0)
    assert set(MessageInputs.__dataclass_fields__) == {
        "ranges", "range_max", "speed", "imu", "att", "speed_cap"}
    assert mi.ranges.shape == (env.sim.cfg.lidar.n_beams,)
    assert mi.imu.ndim == 2 and mi.imu.shape[1] == 6
    assert len(mi.att) == 2
    # `Odometry.twist.twist.linear.x` -- the VESC wheel speed, which is what the bridge publishes.
    assert mi.speed == pytest.approx(float(env.last_result.odom[0, 3]))
    # ... and NOT the ground-truth body speed, which only exists in simulation.
    assert "state" not in MessageInputs.__dataclass_fields__


def test_the_nodes_extra_saturation_is_the_only_difference_and_it_is_for_the_real_driver():
    """`resample_ranges` saturates a non-positive range and the 65.533 m miss sentinel; the env's
    `_norm_scan` does neither, because the simulator emits neither. Stated as a test so the
    asymmetry is a decision rather than a discrepancy someone finds later."""
    from f1sim.learn.obs import norm_scan, resample_ranges
    raw = np.array([1.0, 0.0, -1.0, 65.533, np.inf, np.nan, 9.0], dtype=np.float32)
    node = norm_scan(torch.as_tensor(resample_ranges(raw, 10.0, len(raw))), 10.0)
    env_side = norm_scan(torch.as_tensor(raw), 10.0)
    assert node.tolist() == pytest.approx([0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9], abs=1e-6)
    assert env_side.tolist() == pytest.approx([0.1, 0.0, 0.0, 1.0, 1.0, 1.0, 0.9], abs=1e-6)
    clean = np.array([1.0, 9.0, 4.5], dtype=np.float32)          # anything a simulator produces
    assert torch.equal(norm_scan(torch.as_tensor(resample_ranges(clean, 10.0, 3)), 10.0),
                       norm_scan(torch.as_tensor(clean), 10.0))
