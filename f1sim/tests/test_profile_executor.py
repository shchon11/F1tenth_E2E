"""The profile executor: a plan's own speed profile, sent to the VESC directly, and the contract that says so."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from f1sim import mpc
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.params import Config
from f1sim.track import Track


def _open_floor():
    res, n = 0.1, 400
    occ = np.zeros((n, n), bool); occ[:2] = occ[-2:] = True; occ[:, :2] = occ[:, -2:] = True
    edt = ndimage.distance_transform_edt(~occ).astype(np.float32) * res
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    return Track(occupancy=occ, resolution=res, origin=(0.0, 0.0), edt=edt,
                 centerline=np.stack([20 + 15 * np.cos(th), 20 + 15 * np.sin(th)], 1), name="open",
                 duct=occ.copy(), tall=np.zeros_like(occ))


def _launch(speed_mode, speed_command, steps=20):
    """A car at 2 m/s asked for everything on a straight: mean acceleration over `steps` control steps."""
    cfg = Config(); cfg.sim.compile = False; cfg.rand.enabled = False
    env = common.make_env([_open_floor()], 2, "cpu", EnvConfig(action_mode="plan", race_size=1, collision_mode="soft",
                                                               speed_mode=speed_mode, speed_command=speed_command,
                                                               compile_tracker=False, speed_cap=9.0),
                          cfg=cfg, seed=1)
    env.reset(seed=1)
    env.sim.reset(torch.arange(2), torch.tensor([[5.0, 20.0, 0.0]] * 2), torch.full((2,), 2.0))
    a = torch.zeros(2, env.act_dim)
    a[:, -2:] = 1.0                                          # linear: v0 = v1 = max; envelope: a_hat max, v_end max
    v0 = float(env.sim.state[0, 3])
    for _ in range(steps):
        env.step(a)
    return (float(env.sim.state[0, 3]) - v0) / (steps * env.sim.control_dt)


def test_the_profile_command_drives_the_car_near_its_limit_where_the_tracker_does_not():
    through_tracker = _launch("envelope", "tracker")
    direct = _launch("envelope", "profile")
    assert direct > through_tracker + 0.5, (direct, through_tracker)
    assert direct > 3.5, direct                                   # a_max 5.6 less the power limit above 6 m/s


def test_the_contract_defaults_to_what_every_old_checkpoint_was_and_reads_either_place():
    assert common.plan_output_of({}) == {"speed_mode": "linear", "speed_command": "tracker"}
    assert common.plan_output_of({"speed_mode": "envelope", "speed_command": "profile"}) == \
        {"speed_mode": "envelope", "speed_command": "profile"}
    assert common.plan_output_of({"experiment": {"speed_mode": "knots", "speed_command": "profile"}}) == \
        {"speed_mode": "knots", "speed_command": "profile"}


def test_the_forward_pass_respects_the_cars_power_limit():
    spec = mpc.PlanSpec(speed_mode="envelope")
    a = torch.zeros(1, mpc.act_dim("envelope")); a[:, -2:] = 1.0      # straight, all the grip, top end speed
    _, Lp, prof = mpc.decode_profile(a, torch.tensor([6.0]), 10.0, torch.tensor([10.0]), spec)
    ds = float(Lp[0]) / (prof.shape[1] - 1)
    v = prof[0].numpy()
    gain = (v[1:] ** 2 - v[:-1] ** 2) / (2 * ds)                      # the acceleration each step implies
    assert gain.max() <= spec.a_drive_profile * spec.v_switch_profile / 6.0 + 1e-3


def test_budget_ceiling_bounds_the_whole_profile_and_the_grip_budget_does_not_move():
    spec = mpc.PlanSpec(speed_mode="budget")
    k = torch.zeros(2, mpc.N_KNOTS)
    a = mpc.encode_budget(k, torch.tensor([8.0, 8.0]), torch.tensor([9.0, 9.0]), torch.tensor([9.0, 3.0]), 10.0, spec)
    _, _, prof = mpc.decode_profile(a, torch.tensor([3.0, 3.0]), 10.0, torch.tensor([10.0, 10.0]), spec)
    assert float(prof[1].max()) <= 3.0 + 1e-4                          # the ceiling holds everywhere
    assert float(prof[0].max()) > 3.5                                  # and without it the car may go
    assert mpc.act_dim("budget") == mpc.N_KNOTS + 3


def test_the_budget_teacher_puts_slowing_down_in_the_ceiling_not_in_the_grip():
    from f1sim.raceline import Raceline
    from f1sim import maps
    tr = maps.load("gen:competition:0"); rl = Raceline.build_cached(tr)
    cfg = Config(); cfg.sim.compile = False; cfg.rand.enabled = False
    env = common.make_env([tr], 2, "cpu", EnvConfig(action_mode="plan", race_size=1, speed_mode="budget",
                                                     speed_command="profile", compile_tracker=False), cfg=cfg, seed=1, rls=[rl])
    teacher = common.make_teacher([rl], env)
    env.reset(seed=1)
    # on the teacher's own line (not the centreline), one car on it and one 0.4 m to its side
    p0 = torch.tensor(rl.xy[40], dtype=torch.float32); t = torch.tensor(rl.xy[41] - rl.xy[39], dtype=torch.float32)
    yaw0 = torch.atan2(t[1], t[0])
    xy = p0[None].expand(2, 2).clone(); yaw = yaw0.expand(2).clone()
    nrm = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
    env.sim.reset(torch.arange(2), torch.cat([xy + nrm * torch.tensor([0.0, 0.4])[:, None], yaw[:, None]], 1), torch.full((2,), 3.0))
    lab = env.teacher_label(teacher)
    a_hat = (lab[:, mpc.N_KNOTS] + 1) / 2 * env.tracker.spec.a_hat_max
    v_cap = (lab[:, mpc.N_KNOTS + 2] + 1) / 2 * env.ecfg.v_max_policy
    assert abs(float(a_hat[0] - a_hat[1])) < 1e-4                      # 0.4 m off the line: same grip
    assert float(v_cap[1]) < float(v_cap[0])                           # ... a lower ceiling instead


def test_an_opponent_under_another_contract_is_refused_even_when_its_width_fits():
    from types import SimpleNamespace
    import pytest
    from f1sim.learn.opponent_pool import _check_contract
    env = SimpleNamespace(ecfg=EnvConfig(action_mode="plan"))                  # linear / tracker
    _check_contract("old.pt", {}, env)                                          # every pre-contract checkpoint
    assert mpc.act_dim("envelope") == mpc.act_dim("linear")                     # why the shape check misses it
    with pytest.raises(ValueError, match="contract"):
        _check_contract("env.pt", {"speed_mode": "envelope", "speed_command": "profile"}, env)
    coupled = SimpleNamespace(ecfg=EnvConfig(action_mode="plan", speed_mode="budget", tracker_speed_weight=2.0,
                                             tracker_a_max=7.0))
    _check_contract("budT.pt", {"experiment": {"speed_mode": "budget", "tracker_speed_weight": 2.0,
                                               "tracker_a_max": 7.0}}, coupled)
    with pytest.raises(ValueError, match="contract"):
        _check_contract("bud.pt", {"speed_mode": "budget"}, coupled)             # same outputs, other tracker


def _launch_budget(steps=20, **kw):
    cfg = Config(); cfg.sim.compile = False; cfg.rand.enabled = False
    env = common.make_env([_open_floor()], 2, "cpu", EnvConfig(action_mode="plan", race_size=1, collision_mode="soft",
                                                               speed_mode="budget", compile_tracker=False, speed_cap=9.0,
                                                               tracker_speed_weight=2.0, tracker_a_max=7.0, **kw),
                          cfg=cfg, seed=1)
    env.reset(seed=1)
    env.sim.reset(torch.arange(2), torch.tensor([[5.0, 20.0, 0.0]] * 2), torch.full((2,), 2.0))
    a = torch.zeros(2, env.act_dim)
    a[:, -3:] = 1.0                                          # all the grip, top end speed, no ceiling
    v0 = float(env.sim.state[0, 3])
    for _ in range(steps):
        env.step(a)
    return (float(env.sim.state[0, 3]) - v0) / (steps * env.sim.control_dt)


def test_a_coupled_budget_plan_asked_for_everything_gets_near_the_cars_drive_limit():
    # 2026-09-30: the profile ramped from the measured speed while the iLQR started one latency
    # ahead of it, and the command trailed the motor lag: ~2.6 m/s^2 of the car's 5.6.
    old = _launch_budget()
    new = _launch_budget(tracker_profile_from_prediction=True, tracker_speed_ff=True)
    assert old < 3.0, old
    assert new > 3.8 and new > old + 1.0, (new, old)


def test_the_executor_flags_travel_with_the_contract():
    ex = {"experiment": {"speed_mode": "budget", "tracker_speed_weight": 2.0, "tracker_a_max": 7.0,
                         "tracker_profile_from_prediction": True, "tracker_speed_ff": True}}
    c = mpc.contract_of(ex)
    assert c["tracker_profile_from_prediction"] is True and c["tracker_speed_ff"] is True
    assert mpc.contract_spec(c).profile_from_prediction is True
    assert "tracker_speed_ff" not in mpc.contract_of({"tracker_speed_ff": False})   # old contracts compare equal
    from types import SimpleNamespace
    import pytest
    from f1sim.learn.opponent_pool import _check_contract
    env = SimpleNamespace(ecfg=EnvConfig(action_mode="plan", speed_mode="budget", tracker_speed_weight=2.0,
                                         tracker_a_max=7.0, tracker_profile_from_prediction=True, tracker_speed_ff=True))
    _check_contract("new.pt", ex, env)
    with pytest.raises(ValueError, match="contract"):
        _check_contract("budT.pt", {"experiment": {"speed_mode": "budget", "tracker_speed_weight": 2.0,
                                                   "tracker_a_max": 7.0}}, env)
