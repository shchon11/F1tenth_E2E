"""The `tcs` controller arm: the car's guard running inside the simulator loop.

What is under test is the wiring, not the detector -- `test_traction_guard.py` owns the detector,
and the point of this arm is that it runs that same class unmodified. So: that it is fed the
simulated sensors and only those, that it shapes the command the simulator actually receives, that
it composes with the tracker arms, that it forgets an episode when the episode ends, and that it
refuses to run where it could not possibly fire.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from f1sim import Config, Track
from f1sim.gym_env import EnvConfig, F1VecEnv
from f1sim.learn import grip_runtime as gr
from f1sim.learn.traction_arm import TractionArm
from f1sim_ros.traction import LOCK, TractionParams

B = 4


def make_env(wheel=True, mu=0.73, n=B, **over):
    tr = Track.generate_random(0, style="competition")
    cfg = Config()
    cfg.rand.enabled = False
    cfg.vehicle.wheel_model = wheel
    cfg.vehicle.mu = mu
    for k, v in over.items():
        grp, name = k.split(".")
        setattr(getattr(cfg, grp), name, v)
    return F1VecEnv(tr, cfg, EnvConfig(scan_stack=2, action_history=1, speed_cap=8.0),
                    num_envs=n, device="cpu")


# ------------------------------------------------------------------ arm selection

def test_the_composed_arm_names_are_the_ones_the_roster_and_the_flag_accept():
    assert gr.split_arm("tcs") == ("legacy", True)
    assert gr.split_arm("fixed_low+tcs") == ("fixed_low", True)
    assert gr.split_arm("fixed_low") == ("fixed_low", False)
    for bad in ("legacy+tcs", "tcs+fixed_low", "TCS", "tcs+tcs", "+tcs"):
        with pytest.raises(ValueError):
            gr.split_arm(bad)
    from f1sim.learn.benchmark.roster import ARMS as ROSTER_ARMS
    assert "fixed_low+tcs" in ROSTER_ARMS, "the deployment default must be pinnable in a roster"
    assert not any(a.startswith("oracle") for a in ROSTER_ARMS), "oracle is privileged, not a result"


def test_the_arm_refuses_where_the_residual_would_be_identically_zero():
    """With the wheel model off the wheel speed IS the body speed. An arm that can never fire is a
    legacy run wearing another arm's name, so it must not construct."""
    env = make_env(wheel=False)
    with pytest.raises(RuntimeError, match="wheel_model"):
        TractionArm(env)
    rt = gr.ControllerRuntime(env, "tcs", device="cpu")
    with pytest.raises(RuntimeError, match="wheel_model"):
        rt.install()


def test_install_puts_exactly_one_shaper_on_the_command_path_and_release_takes_it_off():
    env = make_env()
    assert env.cmd_shaper is None
    arm = TractionArm(env).install()
    assert getattr(env.cmd_shaper, "__self__", None) is arm    # bound methods are not identical objects
    with pytest.raises(RuntimeError, match="already"):
        TractionArm(env).install()
    arm.release()
    assert env.cmd_shaper is None


def test_the_legacy_base_installs_the_shaper_and_nothing_else():
    """`tcs` alone needs no plan tracker: it shapes a speed command, not a plan."""
    env = make_env()
    rt = gr.ControllerRuntime(env, "tcs", device="cpu").install()
    assert rt.traction is not None and env.cmd_shaper is not None
    assert rt.grip is None and rt.spy is None            # the tracker is untouched
    rt.release()
    assert env.cmd_shaper is None


# ------------------------------------------------------------------ what it is fed, and what it does

def _hard_stop(env, arm=None, steps=90, launch=18):
    """Launch, then command a full stop. Returns the commands the simulator received."""
    env.reset(seed=3)
    got = []
    for i in range(steps):
        a = torch.tensor([[0.25, 1.0]] * env.B) if i < launch else torch.tensor([[0.25, -1.0]] * env.B)
        env.step(a)
        got.append(env.last_cmd[:, 1].clone())
    return torch.stack(got)


def test_the_guard_fires_on_simulated_slip_and_raises_the_command_it_receives():
    """End to end: a hard stop out of a corner on mu 0.73 locks the wheel, the guard detects it from
    the simulated ERPM odometry and IMU alone, and the speed the simulator receives is raised."""
    env = make_env(mu=0.73)
    arm = TractionArm(env).install()
    with_arm = _hard_stop(env, arm)
    locks = arm.locks
    m = arm.metrics()
    arm.release()

    env2 = make_env(mu=0.73)
    without = _hard_stop(env2)

    assert locks > 0, "the guard never fired on a simulated lock"
    assert m["controller/tcs_release_max"] > 0.05, m
    assert m["controller/tcs_active_frac"] > 0.0
    # the shaper only ever raises a locked command, never lowers it below what was asked
    d = (with_arm - without)
    assert float(d.max()) > 0.05, float(d.max())
    assert float(d.max()) <= TractionParams().release_max + 1e-6


def test_it_is_fed_the_simulated_sensors_and_nothing_else(monkeypatch):
    """The three inputs must be the odometry, the IMU and the emulated motor current of the
    PREVIOUS step -- the most recent sample a command being issued now could have used."""
    env = make_env()
    arm = TractionArm(env).install()
    seen = []
    for g in arm.guards:
        orig = g.update
        g.update = lambda t, v, a, c, _o=orig: (seen.append((t, v, a, c)), _o(t, v, a, c))[1]
    env.reset(seed=1)
    prev = None
    for _ in range(4):
        prev = env.last_result
        before = len(seen)
        env.step(torch.tensor([[0.0, 0.5]] * env.B))
        rows = seen[before:before + env.B]
        assert len(rows) == env.B
        for b, (t, v, a, c) in enumerate(rows):
            assert t == pytest.approx(float(prev.odom_t[b]))
            assert v == pytest.approx(float(prev.odom[b, 3]))
            assert a == pytest.approx(float(prev.imu[b, -1, 3]))
            assert c == pytest.approx(float(prev.motor_current[b]))
    arm.release()


def test_an_external_command_bypasses_the_shaper():
    """A teleoperated or ROS-driven car is the mux output, which the node does not shape either."""
    env = make_env()
    TractionArm(env).install()
    env.reset(seed=2)
    env.set_external_command(0, 0.0, 3.0)
    for _ in range(30):
        env.step(torch.tensor([[0.0, -1.0]] * env.B))
        assert float(env.last_cmd[0, 1]) == pytest.approx(3.0)


def test_a_finished_episode_is_forgotten():
    """Carrying a latched release into the next episode would release the brake of a car that no
    longer exists."""
    env = make_env()
    rt = gr.ControllerRuntime(env, "tcs", device="cpu").install()
    obs, _ = env.reset(seed=5)
    rt.begin(obs)
    for _ in range(40):
        rt.pre_action(obs)
        obs, _, term, trunc, _ = env.step(torch.tensor([[0.3, 1.0]] * env.B))
        rt.post_step(term, trunc)
    g = rt.traction.guards[0]
    g._state = LOCK                                    # pretend it is latched
    rt._reset_mask = torch.tensor([True] + [False] * (env.B - 1))
    rt.pre_action(obs)
    assert rt.traction.guards[0].state.state == "ok"
    assert rt.traction.guards[0]._t is None, "reset must leave no history at all"
    rt.release()


def test_it_composes_with_the_control_arm():
    """`fixed_low+tcs` is the deployment default: the tracker's friction arm and the guard, both."""
    tr = Track.generate_random(0, style="competition")
    cfg = Config(); cfg.rand.enabled = False; cfg.vehicle.wheel_model = True
    env = F1VecEnv(tr, cfg, EnvConfig(scan_stack=2, action_history=1, speed_cap=8.0,
                                      action_mode="plan"), num_envs=B, device="cpu")
    rt = gr.ControllerRuntime(env, "fixed_low+tcs", device="cpu")
    assert (rt.base, rt.tcs) == ("fixed_low", True)
    assert rt.gspec.mode == "fixed"
    rt.install()
    assert rt.grip is not None and rt.traction is not None
    meta = rt.checkpoint_meta()
    assert meta["arm"] == "fixed_low+tcs"
    # every threshold the guard ran under travels with the checkpoint
    assert meta["traction"]["params"]["lock_rate"] == TractionParams().lock_rate
    assert meta["grip_spec"]["mode"] == "fixed"
    rt.release()
    assert env.cmd_shaper is None


def test_traction_parameters_are_refused_on_an_arm_that_has_no_guard():
    env = make_env()
    with pytest.raises(ValueError, match="only meaningful"):
        gr.ControllerRuntime(env, "fixed_low", device="cpu", traction_params=TractionParams())


def test_the_arm_costs_what_it_is_documented_to_cost():
    """Host Python over scalars plus two transfers a step. The bar is deliberately loose -- this is
    a regression guard against someone adding a per-env synchronise, not a benchmark."""
    import time
    env = make_env(n=32)
    arm = TractionArm(env).install()
    env.reset(seed=1)
    cmd = torch.tensor([[0.0, 3.0]] * 32)
    for _ in range(5):
        env.step(cmd)
    t0 = time.perf_counter()
    for _ in range(20):
        arm.shape(cmd)
    per_call = (time.perf_counter() - t0) / 20
    arm.release()
    assert per_call < 0.02, f"{1000 * per_call:.2f} ms for 32 cars"
