"""The teachers speak "wp4" (output-format design 5, migration step 3).

`RacelineTeacher.plan_action` encodes a wp4 plan itself -- the path by `wp4_path`'s recipe, the
speeds read off its grip profile at the waypoints -- and everything built on it (the env's guard,
the interactive teacher, the opponent planners, the rollout teacher, the layout-line teacher) drives
a wp4 race. The guard's candidates are previewed on the wp4 tracker (`mpc.solve`'s wp4 branch, or
`wp4_reference` on the reference fallback), its "stop" is a zero speed target in both formats (the
a_brake curve design 5 drew executed softer than the guard's own x0.3 candidate: RESULTS 4 and 7 of
docs/research/wp4-label-gates-2026-10-14/), and a linear label does not move by a bit.

The label-quality gates of design section 7 are closed-loop measurements, not unit tests:
`f1sim/scripts/wp4_label_replay.py` and `f1sim/scripts/wp4_label_gates.py`
(docs/research/wp4-label-gates-2026-10-14/).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import torch

from f1sim import Config, maps, mpc
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.raceline import Raceline
from f1sim.teacher import RacelineTeacher

WP4 = mpc.PlanSpec(speed_mode="wp4")
LIN = mpc.PlanSpec()
V_MAX = 10.0
TRACK = "gen:competition:0"


def _circle_teacher(radius: float = 4.0, n: int = 800) -> RacelineTeacher:
    """A teacher on a circular line: the one path whose every window the format holds exactly."""
    th = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    xy = np.stack([radius * np.cos(th), radius * np.sin(th)], 1)        # counter-clockwise: kappa = +1/R
    cfg = Config()
    return RacelineTeacher([Raceline.from_xy(xy, np.full(n, 4.0))], wheelbase=cfg.vehicle.lf + cfg.vehicle.lr,
                           vehicle=cfg.vehicle, mu_nominal=cfg.vehicle.mu, mu_f_scale_nominal=cfg.vehicle.mu_f_scale)


def _on_circle(t: RacelineTeacher, speeds, inward: float = 0.0) -> torch.Tensor:
    """Cars at evenly spread points of the circle, along it, `inward` metres towards its centre."""
    v = torch.as_tensor(speeds, dtype=torch.float32)
    idx = torch.arange(v.numel()) * (t.N // v.numel())
    p, tg = t.xy[0, idx], t.tan[0, idx]
    st = torch.zeros(v.numel(), 8)
    st[:, :2] = p * (1.0 - inward / p.norm(dim=1, keepdim=True))
    st[:, 2] = torch.atan2(tg[:, 1], tg[:, 0])
    st[:, 3] = v
    return st


def _plan_world(a: torch.Tensor, st: torch.Tensor, spec=WP4):
    """The decoded wp4 path in the world, over its own window (B, 49, 2), and the waypoint speeds."""
    k, L, v = mpc.decode_wp4(a, st[:, 3], V_MAX, torch.full_like(st[:, 3], V_MAX), spec)
    x, y, _, _, _ = mpc.wp4_path(k, L, spec, extend=1.0)
    c, s = torch.cos(st[:, 2]), torch.sin(st[:, 2])
    return torch.stack([st[:, :1] + x * c[:, None] - y * s[:, None], st[:, 1:2] + x * s[:, None] + y * c[:, None]], -1), v


@pytest.mark.parametrize("recipe", ["direct", "hybrid"])
def test_a_car_on_the_line_is_handed_the_line_and_its_profile(recipe):
    """On a circle the direct fit (and the hybrid, which is the direct fit with no offset) gives the
    circle back: the curvatures at 1/R, the window on the line to a few mm, and the speeds the
    profile's at each waypoint."""
    t = _circle_teacher(4.0)
    t.wp4_path = recipe
    st = _on_circle(t, [1.0, 2.5, 4.0, 5.5])
    a = t.plan_action(st, None, None, V_MAX, WP4)
    assert a.shape == (4, 8) and a.abs().max() <= 1.0
    k, _, v = mpc.decode_wp4(a, st[:, 3], V_MAX, torch.full((4,), V_MAX), WP4)
    assert (k - 0.25).abs().max() < 5e-3, k                     # the line points are 3 cm apart
    pts, _ = _plan_world(a, st)
    assert (pts.norm(dim=-1) - 4.0).abs().max() < 5e-3
    want = t.speed_at(torch.zeros(4, 4, dtype=torch.long), torch.zeros(4, 4, dtype=torch.long),
                      torch.full((4, 4), t.nominal_grip_index))                           # the circle's profile is flat
    assert (v - want).abs().max() < 1e-4


def test_the_refit_is_the_linear_plans_first_window():
    """`refit` hands back the linear 1.5 s plan's window, imperfections and all: on the same circle
    at 5.5 m/s that plan (six knots fitted from 40 % of an 8 m arc on) bends 0.29 / 0.21 / 0.25 /
    0.20 1/m where the line is 0.25 -- which is what the direct fit is for."""
    t = _circle_teacher(4.0)
    t.wp4_path = "refit"
    st = _on_circle(t, [1.0, 2.5, 4.0, 5.5])
    a = t.plan_action(st, None, None, V_MAX, WP4)
    t._defer_profile_projection = True
    lin = t.plan_action(st, None, None, V_MAX, LIN)
    k6 = lin[:, mpc.plan_layout(LIN).path] * LIN.kappa_max
    k_lim = 0.85 * min(LIN.kappa_max, math.tan(min(t.steer_max, t.vehicle.s_max)) / t.L)
    want = mpc.wp4_refit(k6, mpc.plan_length(st[:, 3], LIN), st[:, 3], WP4, iters=4, k_lim=torch.full((4,), k_lim))
    k, _, _ = mpc.decode_wp4(a, st[:, 3], V_MAX, torch.full((4,), V_MAX), WP4)
    assert (k - want).abs().max() < 1e-5
    assert (k[-1] - 0.25).abs().max() > 0.03                       # the 1.5 s plan's own error, kept


@pytest.mark.parametrize("recipe", RacelineTeacher.WP4_PATHS)
@pytest.mark.parametrize("inward", [0.3, -0.3])
def test_a_car_off_the_line_is_brought_back_inside_the_window(recipe, inward):
    """0.3 m inside or outside the circle, at 2-6 m/s: the window ends less than half that from the
    line, and crosses it by less than half that. Neither recipe is clean here, which is the point of
    measuring both -- the direct fit, pinned to the line from 1/8 of the window on, overshoots by up
    to 9 cm from inside; the 1.5 s plan's window by up to 14 cm from outside at 6 m/s."""
    t = _circle_teacher(4.0)
    t.wp4_path = recipe
    st = _on_circle(t, [2.0, 4.0, 5.0, 6.0], inward=inward)
    pts, _ = _plan_world(t.plan_action(st, None, None, V_MAX, WP4), st)
    off = (4.0 - pts.norm(dim=-1)) * math.copysign(1.0, inward)     # towards where the car started, +
    assert (off[:, 0] - 0.3).abs().max() < 1e-3
    assert (off[:, -1].abs() < 0.15).all(), off[:, -1]
    assert (off > -0.15).all(), off.min()


def test_hybrid_is_the_direct_fit_on_the_line_and_the_refit_on_an_offset():
    """The design's evidence (section 5): fitted straight to an offset line the 0.6 s plan snapped
    to it; the hybrid refits the teacher's own 1.5 s plan there, and follows the line itself with the
    direct fit. A ramping offset blends over its first 10 cm instead of jumping recipes."""
    tr = maps.load(TRACK)
    t = RacelineTeacher([Raceline.build_cached(tr)], wheelbase=0.33)
    idx = torch.arange(0, t.N, t.N // 6)[:6]
    st = torch.zeros(6, 8)
    st[:, :2] = t.xy[0, idx]; st[:, 2] = torch.atan2(t.tan[0, idx, 1], t.tan[0, idx, 0]); st[:, 3] = 4.0
    label = {}
    for recipe in ("direct", "refit", "hybrid"):
        t.wp4_path = recipe
        for name, off in (("none", None), ("zero", torch.zeros(6)), ("wide", torch.full((6,), 0.3)),
                          ("ramp", torch.full((6,), 0.05))):
            label[recipe, name] = t.plan_action(st, None, None, V_MAX, WP4, offset=off)
    path = mpc.plan_layout(WP4).path
    assert torch.equal(label["hybrid", "none"], label["direct", "none"])
    assert torch.equal(label["hybrid", "zero"], label["direct", "zero"])
    assert torch.equal(label["hybrid", "wide"], label["refit", "wide"])
    half = 0.5 * (label["direct", "ramp"][:, path] + label["refit", "ramp"][:, path])
    assert (label["hybrid", "ramp"][:, path] - half).abs().max() < 1e-5
    assert not torch.allclose(label["direct", "wide"][:, path], label["refit", "wide"][:, path], atol=1e-3)
    # the recipe is the path's alone: the speeds are the profile's whichever it is
    sp = mpc.plan_layout(WP4).speed
    assert torch.equal(label["direct", "wide"][:, sp], label["refit", "wide"][:, sp])


def test_a_linear_label_does_not_read_the_wp4_recipe():
    tr = maps.load(TRACK)
    t = RacelineTeacher([Raceline.build_cached(tr)], wheelbase=0.33)
    st = torch.zeros(5, 8)
    idx = torch.arange(0, t.N, t.N // 5)[:5]
    st[:, :2] = t.xy[0, idx] + 0.1; st[:, 2] = torch.atan2(t.tan[0, idx, 1], t.tan[0, idx, 0]); st[:, 3] = 5.0
    out = []
    for recipe in RacelineTeacher.WP4_PATHS:
        t.wp4_path = recipe
        out.append(t.plan_action(st, None, None, V_MAX, LIN, offset=torch.full((5,), 0.2)))
    assert all(torch.equal(out[0], o) for o in out[1:])


def test_the_linear_speed_options_are_refused_for_a_wp4_label():
    t = _circle_teacher()
    st = _on_circle(t, [3.0])
    for name, value in (("speed_horizon_s", 0.6), ("speed_lead_s", 0.5), ("speed_error_gain", 1.0)):
        t2 = _circle_teacher()
        setattr(t2, name, value)
        with pytest.raises(ValueError, match="linear plan's two speeds"):
            t2.plan_action(st, None, None, V_MAX, WP4)
        t2.plan_action(st, None, None, V_MAX, LIN)                    # still the linear teacher's to use
    t.wp4_path = "straight"
    with pytest.raises(ValueError, match="wp4_path"):
        t.plan_action(st, None, None, V_MAX, WP4)


def test_the_ridge_pulls_the_fit_towards_its_guess_and_zero_is_the_plain_fit():
    g = torch.Generator().manual_seed(3)
    k_true = (torch.rand(16, 4, generator=g) - 0.5) * 1.2
    L = torch.rand(16, generator=g) * 3 + 1.0
    x, y, _, _, _ = mpc.wp4_path(k_true, L, WP4, extend=1.0)
    si = torch.arange(1, mpc.WP_FIT_POINTS + 1) * (WP4.wp_grid // mpc.WP_FIT_POINTS)
    guess = k_true + 0.3
    plain = mpc.fit_wp4(guess, x[:, si], y[:, si], L, WP4)
    assert torch.equal(plain, mpc.fit_wp4(guess, x[:, si], y[:, si], L, WP4, ridge=0.0))
    assert (plain - k_true).abs().max() < 1e-3
    held = mpc.fit_wp4(guess, x[:, si], y[:, si], L, WP4, ridge=1e6)
    assert (held - guess).abs().max() < 1e-3


# ------------------------------------------------------------------------------- the env's teachers
@functools.lru_cache(maxsize=1)
def _track_and_raceline():
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(speed_mode="wp4", envs=4, seed=5, **kw):
    tr, rl = _track_and_raceline()
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(**{"action_mode": "plan", "speed_mode": speed_mode, "collision_mode": "soft", "max_steps": 4000,
                        "hist_len": 0, **kw})
    env = common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])
    return env, common.make_teacher([rl], env)


def _drive(env, teacher, steps):
    env.reset(seed=5)
    for _ in range(steps):
        env.step(env.teacher_label(teacher))


@pytest.mark.parametrize("evasion", ["stop", "curve"])
def test_stop_is_a_zero_speed_target_in_both_formats(evasion, monkeypatch):
    """The on-line brake is -1 on every speed column in both formats; the two braking evasions are
    too, unless `wp4_evasion_brake` is "curve" -- then a wp4 plan's follow the tracker's a_brake curve
    from the plan speed (`mpc.wp4_braking`). Linear never moves."""
    from f1sim.interactive_teacher import InteractiveTeacher
    monkeypatch.setattr(InteractiveTeacher, "wp4_evasion_brake", evasion)
    for mode in ("wp4", "linear"):
        env, t = _env(mode)
        _drive(env, t, 30)
        it = InteractiveTeacher(t, env)
        v = env._tracker_plan_speed()
        idx = t.project(env.sim.state[:, :2], env.sim.tid)[0]
        cand = it._candidates(env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy, env.tracker.spec, None, idx,
                              plan_speed=v)
        n_reg = it.offsets.numel() * it.speeds.numel()
        sp = mpc.plan_layout(env.tracker.spec).speed
        assert (cand[n_reg][..., sp] == -1).all(), mode                   # the brake
        brakes = cand[[n_reg + 2, n_reg + 4]][..., sp]                     # both braking evasions
        coasts = cand[[n_reg + 1, n_reg + 3]][..., sp]
        assert ((coasts + 1) / 2 * env.ecfg.v_max_policy - v.abs()[None, :, None]).abs().max() < 1e-4
        if mode == "linear" or evasion == "stop":
            assert (brakes == -1).all(), (mode, evasion)
            continue
        L = mpc.plan_length(v, env.tracker.spec)
        s = L[:, None] * torch.arange(1, 5) / 4
        want = (v.abs()[:, None] ** 2 - 2 * env.tracker.spec.a_brake * s).clamp_min(0).sqrt()
        got = (brakes + 1) / 2 * env.ecfg.v_max_policy
        assert (got - want[None]).abs().max() < 1e-4
        assert (v.abs() > 1.0).all() and (got[0, :, 0] > 0).all()   # not "stand still 0.15 s from now"


def test_the_guard_previews_wp4_plans_on_the_wp4_tracker_and_its_fallback():
    """The guard's preview is the installed tracker's own solve; on the reference fallback it is the
    wp4 reference walked over the scoring horizon -- for a constant-curvature plan, a circle."""
    from f1sim.interactive_teacher import InteractiveTeacher
    env, t = _env("wp4")
    _drive(env, t, 20)
    assert env._raceline_teacher_guards[t].last_preview_model == "legacy_mpc"
    it = InteractiveTeacher(t, env)
    v = env._tracker_plan_speed()
    k = 0.4
    a = mpc.encode_wp4(torch.full((env.B, 4), k), v.abs()[:, None].expand(-1, 4), env.ecfg.v_max_policy, WP4)
    world, psi, speed = it.rollout(a[None], env.sim.state, env.ecfg.v_max_policy, env.tracker.spec, plan_speed=v)
    st = env.sim.state
    centre = st[:, :2] + torch.stack([-torch.sin(st[:, 2]), torch.cos(st[:, 2])], 1) / k
    r = (world[0] - centre[:, None]).norm(dim=-1)
    assert (r - 1 / k).abs().max() < 0.02
    assert torch.isfinite(speed).all() and speed.shape == (1, env.B, it.horizon_times(env.tracker.spec).numel())


def test_the_guard_follows_a_recipe_set_after_it_was_built():
    env, t = _env("wp4")
    env.reset(seed=5)
    for _ in range(25):
        env.step(env.teacher_label(t))
    t.wp4_path = "direct"
    a_direct = env.teacher_label(t)
    t.wp4_path = "refit"
    a_refit = env.teacher_label(t)
    path = mpc.plan_layout(WP4).path
    assert not torch.allclose(a_direct[:, path], a_refit[:, path], atol=1e-4)
    raw = t.plan_action(env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy, env.tracker.spec,
                        offset=torch.zeros(env.B), plan_speed=env._tracker_plan_speed())
    chosen_line = env._raceline_teacher_guards[t].last_choice < 4      # one of the line's own speed candidates
    assert torch.allclose(a_refit[chosen_line][:, path], raw[chosen_line][:, path], atol=1e-6)


@pytest.mark.parametrize("kind", ["raceline", "interactive", "lane_switch", "forzaeth_pred", "forzaeth"])
def test_every_teacher_kind_drives_a_wp4_race(kind):
    """Teacher-driven opponents of every kind plan wp4 in a wp4 env (they raised before the teacher
    spoke it) and get as far as in a linear one: 1.5 s from the same spawn, 0.9 x their profile."""
    moved = {}
    for mode in ("linear", "wp4"):
        env, t = _env(mode, envs=4, race_size=2, opponent="slots", opponent_slots=({"kind": kind, "speed_scale": 0.9},))
        env.reset(seed=5)
        s0 = env.sim.s.clone()
        for _ in range(60):
            env.step(env.teacher_label(t))
        opp = ~env.learner
        a = env.prev_action[opp]
        assert torch.isfinite(a).all() and a.abs().max() <= 1.0
        assert torch.isfinite(env.sim.state).all()
        L = env.sim.track.length[env.sim.tid]
        moved[mode] = (torch.remainder(env.sim.s - s0 + L / 2, L) - L / 2)[opp]
    assert (moved["wp4"] > 0.8 * moved["linear"]).all(), moved


def test_the_rollout_teacher_decides_in_a_wp4_race():
    from f1sim.rollout_teacher import RolloutTeacher
    env, t = _env("wp4", envs=2, race_size=2, opponent="teacher")
    env.reset(seed=5)
    rt = RolloutTeacher(t, env, offsets=(-0.3, 0.0, 0.3), speeds=(1.0, 0.5), horizon_s=0.25, every=4, lidar_beams=None)
    assert rt.shadow.S.tracker.spec.speed_mode == "wp4"
    for _ in range(6):
        a = rt.plan_action(env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy, env.tracker.spec,
                           plan_speed=env._tracker_plan_speed())
        assert a.shape == (env.B, 8) and torch.isfinite(a).all()
        env.step(a)
    plans = rt.candidate_plans()
    assert plans.shape == (env.B, rt.K, 8) and torch.isfinite(plans).all()
    # a speed scale of 0.5 halves the waypoint speeds, through the layout's speed columns
    sp = mpc.plan_layout(WP4).speed
    full, half = plans[:, 2], plans[:, 3]                       # candidates (offset 0, x1.0) and (offset 0, x0.5)
    assert torch.allclose(((half[:, sp] + 1) / 2), ((full[:, sp] + 1) / 2) * 0.5, atol=1e-6)
