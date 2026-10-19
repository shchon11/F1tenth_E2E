"""Two things of 2026-10-19.

* Every driver drives a calibrated car. The tracker our policy drives through has always been handed
  the car's measured servo offset / gain and speed gain; external drivers (the baselines) reached the
  actuators raw, on a servo trimmed up to +-0.03 rad off. `F1VecEnv._external_commands` now puts
  their commands through the same correction, so the servo is where they asked to within the
  measurement's residual.
* "우리도 그냥 map 쓰자": `EnvConfig.plan_follower = "map"` follows the policy's plans with ForzaETH's
  MAP controller (`map_controller.MAPPlanTracker`) -- for the policy's cars only; the traffic keeps the
  iLQR it was built and raced with.
"""
import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common


class _Steady:
    """An external planner commanding the same (steer, speed) every step."""
    needs_speed = False

    def __init__(self, steer, speed):
        self.steer, self.speed = steer, speed
        self.rows = None

    def command_env(self, env, rows):
        return torch.tensor([[self.steer, self.speed]], device=env.device).expand(rows.numel(), 2).clone()

    def reset(self, mask):
        pass


@pytest.mark.parametrize("mode", ["direct", "plan"])
def test_an_external_driver_steers_a_calibrated_car(mode):
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"
    ecfg = EnvConfig(max_steps=1000, hist_len=0, action_mode=mode)
    env = common.make_env([maps.load("real:korea_2025_iccas")], 4, "cpu", ecfg, cfg=cfg, seed=5)
    env.reset()
    P = env.sim.P
    P["steer_bias"][:] = torch.tensor([0.03, -0.03, 0.02, 0.0])
    P["steer_gain"][:] = torch.tensor([1.08, 0.92, 1.0, 1.05])
    ids = torch.arange(env.B)
    (env._calibrate_tracker if mode == "plan" else env._calibrate_external)(ids)
    env.set_ego_driver(_Steady(0.10, 2.0))
    env.step(torch.zeros(env.B, env.act_dim))
    servo = env.last_cmd[:, 0] * P["steer_gain"] + P["steer_bias"]        # what the servo is asked for
    # within the measurement's residual: offset +-0.01 rad, gain +-4 % of 0.1 rad
    assert float((servo - 0.10).abs().max()) < 0.015, servo.tolist()


def test_map_follows_the_policy_rows_and_leaves_the_traffic_alone():
    """On the same inputs the wrapped tracker answers the traffic rows exactly as the bare iLQR tracker,
    and MAP answers the policy's rows (a race couples the cars, so whole trajectories would not show it)."""
    from f1sim import mpc
    from f1sim.map_controller import GRIP_LEVELS, MAPParams, MAPPlanTracker, SteerLUT
    steers = torch.linspace(0, 0.4, 41)
    vels = torch.linspace(0.5, 10.0, 20)
    table = (vels[None, :] ** 2 * torch.tan(steers)[:, None] / 0.33)[None].expand(len(GRIP_LEVELS), -1, -1).contiguous()
    lut = SteerLUT(steers, vels, table)
    bare = mpc.PlanTracker(4, "cpu", 0.33, 0.4189, 10.0, compile_solver=False)
    wrapped = MAPPlanTracker(mpc.PlanTracker(4, "cpu", 0.33, 0.4189, 10.0, compile_solver=False), lut, MAPParams(),
                             {"mu": 1.0, "mu_f_scale": 1.0})
    wrapped.map_rows = torch.tensor([True, False, True, False])
    wrapped.P = {"mu": torch.ones(4), "mu_f_scale": torch.ones(4)}
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        a = torch.rand(4, bare.spec and 8, generator=g) * 1.6 - 0.8
        v = torch.rand(4, generator=g) * 6 + 1
        cap = torch.full((4,), 9.0)
        r = (torch.rand(4, generator=g) - 0.5)
        want, got = bare(a, v, cap, r, 0.05), wrapped(a, v, cap, r, 0.05)
        assert torch.equal(want[~wrapped.map_rows], got[~wrapped.map_rows])
        assert not torch.allclose(want[wrapped.map_rows], got[wrapped.map_rows])


def test_a_plan_policy_env_runs_on_map():
    from f1sim.raceline import Raceline
    lim = common.teacher_limits(7, 6.5, 4)
    tr = maps.load("real:korea_2025_iccas")
    rl = Raceline.build_cached(tr, objective="min_time", **lim)
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"
    ecfg = EnvConfig(max_steps=1000, hist_len=0, action_mode="plan", plan_follower="map", race_size=2,
                     opponent="teacher", compile_tracker=False)
    env = common.make_env([tr], 4, "cpu", ecfg, cfg=cfg, seed=3, rls=[rl], teacher_limits=lim)
    env.reset()
    t = common.make_teacher([rl], env, grip="true", **lim)
    for _ in range(80):
        env.step(env.teacher_label(t))
    assert type(env.tracker).__name__ == "MAPPlanTracker" and env.tracker.last_map
    assert bool((env.sim.state[env.learner, 3] > 2.0).all())             # the policy's cars follow their plans


def test_the_plan_waypoints_are_the_plans_path_and_profile():
    from f1sim import mpc
    from f1sim.map_controller import MAPParams, MAPPlanTracker, PLAN_WP_SPACING, SteerLUT
    steers = torch.linspace(0, 0.4, 41)
    vels = torch.linspace(0.5, 10.0, 20)
    lut = SteerLUT(steers, vels, (vels[None, :] ** 2 * torch.tan(steers)[:, None] / 0.33)[None])
    inner = mpc.PlanTracker(2, "cpu", 0.33, 0.4189, 10.0, compile_solver=False)
    tr = MAPPlanTracker(inner, lut, MAPParams(), {"mu": 1.0, "mu_f_scale": 1.0})
    v_meas = torch.tensor([4.0, 6.0])
    k = torch.zeros(2, mpc.N_KNOTS)                                        # straight plans
    a = mpc.encode(k, torch.tensor([4.0, 6.0]), torch.tensor([6.0, 3.0]), 10.0, inner.spec, v_meas=v_meas)
    path, pv, pk = tr.waypoints(a, v_meas, torch.full((2,), 9.0))
    d = (path[:, 1:] - path[:, :-1]).norm(dim=-1)
    assert torch.allclose(d, torch.full_like(d, PLAN_WP_SPACING), atol=1e-4)
    assert torch.allclose(path[:, 0], torch.zeros(2, 2)) and float(path[..., 1].abs().max()) < 1e-6
    Lp = mpc.plan_length(v_meas, inner.spec)
    i_end = (Lp / PLAN_WP_SPACING).round().long()
    assert abs(float(pv[0, 0]) - 4.0) < 1e-3 and abs(float(pv[0, i_end[0]]) - 6.0) < 0.05   # v0 -> v1 along it
    assert abs(float(pv[1, i_end[1]]) - 3.0) < 0.05 and float(pv[1, -1]) == pytest.approx(3.0, abs=1e-4)
