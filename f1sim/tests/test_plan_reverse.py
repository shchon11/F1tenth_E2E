"""The plan's own reverse (2026-11-05, `EnvConfig.plan_reverse_ext`, the user: "정책 자체에서 찾아내가야지 이걸 규칙화하면 좀
이상할 것 같은데"): below -1 on the first plan speed the policy's car backs along the plan's curve, in [-1, 1] the plan
drives exactly as before, and MAP's own back-out rules are off for the policy's cars."""
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common, state_rl


def _env(ext):
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"
    ecfg = EnvConfig(max_steps=1000, hist_len=0, action_mode="plan", speed_mode="wp4", plan_follower="map",
                     compile_tracker=False, plan_reverse_ext=ext)
    env = common.make_env([maps.load("real:korea_2025_iccas")], 2, "cpu", ecfg, cfg=cfg, seed=1)
    env.reset(seed=1)
    return env


def test_below_minus_one_the_car_backs_along_the_plan():
    env = _env(0.5)
    a = torch.zeros(env.B, env.act_dim)
    a[:, state_rl.PLAN_KAPPA_DIMS:] = -1.0
    a[:, state_rl.PLAN_KAPPA_DIMS] = -1.5                                 # the whole extension: -reverse_speed
    for _ in range(40):
        env.step(a)
    assert env.tracker.auto_backout is False
    assert bool((env.sim.state[env.learner, 3] < -0.5).all())


def test_inside_the_usual_range_the_plan_drives_as_before():
    a = torch.zeros(2, 8)
    a[:, state_rl.PLAN_KAPPA_DIMS:] = 0.0                                 # 5 m/s
    a[:, 0] = 0.2
    st = []
    for ext in (0.0, 0.5):
        env = _env(ext)
        for _ in range(40):
            env.step(a)
        st.append(env.sim.state.clone())
    assert float(st[0][:, 3].min()) > 1.0
    assert torch.allclose(st[0], st[1])


def test_the_policys_range_reaches_the_extension_on_the_first_speed_only():
    ar = {"action": "plan", "plan_reverse_ext": 0.5}
    a = torch.full((1, 8), -3.0)
    out = state_rl.clamp_any(a, ar)
    assert float(out[0, state_rl.PLAN_KAPPA_DIMS]) == -1.5
    assert bool((out[0, [0, 1, 2, 3, 5, 6, 7]] == -1.0).all())
    lo, hi, _ = state_rl.mean_bounds(ar, 8)
    assert lo[state_rl.PLAN_KAPPA_DIMS] == -1.5 and min(lo[:4] + lo[5:]) == -1.0 and hi == [1.0] * 8
    assert state_rl.plan_env_fields(ar)["plan_reverse_ext"] == 0.5
    assert float(state_rl.clamp_any(a, {"action": "plan"}).min()) == -1.0  # older checkpoints: as before
