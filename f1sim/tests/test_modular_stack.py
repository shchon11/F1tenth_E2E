"""The modular baselines' own localisation and perception (2026-10-18).

The user: "모듈형은 왜 상대 위치랑 자기 위치를 gt로 써? ... Localization Perception prediction은
개나줬냐?" These check the three claims the stack rests on: the particle filter localises from odometry
and scan alone (and beats the odometry it starts from), the tracker finds the car ahead and its speed
from the scan, and a planner with an observer reads ONLY the observer -- no ground truth leaks in.
"""
import math

import numpy as np
import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.localization import ParticleFilter, PFParams


def _env(race_size=1, envs=1, seed=7):
    from f1sim.raceline import Raceline
    track = maps.load("real:korea_2025_iccas")
    lim = common.teacher_limits(7.0, 6.5, 4.0)
    rl = Raceline.build_cached(track, objective="min_time", **lim)
    cfg = Config(); cfg.sim.compile_mode = "none"
    ecfg = EnvConfig(max_steps=100000, hist_len=0, race_size=race_size, opponent="teacher", action_mode="direct",
                     opp_speed_range=(0.6, 0.6))
    env = common.make_env([track], envs * race_size, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl], teacher_limits=lim)
    env.reset()
    return env, rl, lim


def _drive(env, t):
    cmd = t(env.sim.state, env.sim.P, env.sim.tid)
    v_max = env.ecfg.v_max_policy
    a = torch.stack([cmd[:, 0] / env.s_max, cmd[:, 1] / v_max * 2 - 1], 1).clamp(-1, 1)
    env.step(a)


def test_the_particle_filter_localises_from_odometry_and_scan():
    env, rl, lim = _env()
    t = common.make_teacher([rl], env, grip="true", **lim)
    pf = ParticleFilter(env.sim, torch.arange(env.B), PFParams(n_particles=300))
    lr = env.last_result
    pf.reset(torch.ones(env.B, dtype=torch.bool), env.sim.state[:, :3].clone(), lr.odom[:, :3].clone())
    errs = []
    for k in range(160):
        _drive(env, t)
        lr = env.last_result
        est = pf.step(lr.odom[:, :3], lr.scan, env.sim.tid)
        if k >= 20:
            errs.append(float((est[0, :2] - env.sim.state[0, :2]).norm()))
    odo = float((lr.odom[0, :2] - env.sim.state[0, :2]).norm())
    assert np.median(errs) < 0.15 and max(errs) < 0.6, (np.median(errs), max(errs))
    assert np.median(errs[-20:]) < odo or odo < 0.1          # it is doing more than dead reckoning


def test_the_tracker_finds_the_car_ahead_and_its_speed():
    env, rl, lim = _env(race_size=2, envs=1, seed=3)
    drv = common.build_ego_driver(env, "forzaeth", rls=[rl], limits=lim, stack_params={"pf": {"n_particles": 200}})
    env.set_ego_driver(drv)
    env.reset()
    st, ego = drv.stack, int(drv.stack.rows[0])
    opp = int(env.sim.other_idx[ego, 0])
    found = []
    for k in range(120):
        env.step(torch.zeros(env.B, env.act_dim))
        S = env.sim.state
        rel = S[opp, :2] - S[ego, :2]
        ahead = float(rel[0] * math.cos(S[ego, 2]) + rel[1] * math.sin(S[ego, 2]))
        if 0.5 < ahead < 5.0 and k > 30:
            ok = st.obs_ok[ego]
            if bool(ok.any()):
                d = (st.obs_xy[ego][ok] - S[opp, :2]).norm(dim=1)
                j = int(d.argmin())
                found.append((float(d[j]), float(st.obs_vs[ego][ok][j] - S[opp, 3])))
            else:
                found.append((float("inf"), float("inf")))
    assert found, "the opponent was never 0.5-5 m ahead of the ego"
    pos = np.array([f[0] for f in found]); vel = np.array([f[1] for f in found])
    assert np.mean(pos < 0.6) > 0.8, pos
    assert np.median(np.abs(vel[np.isfinite(vel)])) < 1.0, vel


def test_a_planner_with_an_observer_reads_no_ground_truth():
    """Empty tracks, a real car ahead: the planner must see nothing (no simulator lookup behind it)."""
    env, rl, lim = _env(race_size=2, envs=1, seed=3)
    drv = common.build_ego_driver(env, "forzaeth", rls=[rl], limits=lim, stack_params={"pf": {"n_particles": 50}})
    sp, st = drv.teacher, drv.stack
    st.obs_ok.zero_()
    g, d, idx, tid, ego_d = sp._opponents(env.sim.state, env.sim.tid, torch.full((env.B,), -4.0))
    assert bool(torch.isinf(g).all())
    b_gap, _ = sp._blockage_ahead(env.sim.state, env.sim.tid, torch.full((env.B,), 4.0), 6.0)
    assert bool(torch.isinf(b_gap).all())
    # and a track it does have is what it plans against
    ego = int(st.rows[0])
    st.obs_ok[ego, 0] = True
    s_e = float(sp.base.project(env.sim.state[ego:ego + 1, :2], env.sim.tid[ego:ego + 1])[0][0]) * float(sp.base.ds[0])
    st.obs_s[ego, 0] = (s_e + 2.0) % float(sp.base.length[0])
    st.obs_d[ego, 0] = 0.3
    g, d, *_ = sp._opponents(env.sim.state, env.sim.tid, torch.full((env.B,), -4.0))
    assert abs(float(g[ego]) - 2.0) < 0.15 and abs(float(d[ego]) - 0.3) < 1e-6


def test_the_tuning_knobs_reach_the_planner_and_the_networks():
    """What tune.py turns must be what drives: the planner's own kwargs, the follower's, a speed scale."""
    import math as _m
    from f1sim.learn import baselines
    env, rl, lim = _env(race_size=2, envs=1, seed=3)
    drv = common.build_ego_driver(env, "forzaeth", rls=[rl], limits=lim,
                                  stack_params={"pf": {"n_particles": 50}, "teacher": {"lookahead_gain": 0.5},
                                                "planner": {"evasion_dist": 0.8}, "det": {"min_points": 4}})
    assert drv.teacher.evasion_dist == 0.8 and drv.teacher.base.k_ld == 0.5
    assert drv.stack.per.dp.min_points == 4 and drv.stack.pf.N == 50
    a = baselines.load("ftg", "none")
    b = baselines.load("ftg", "none", speed_scale=1.5)
    for d in (a, b):
        d.bind_scanner(n_beams=1081, fov=1.5 * _m.pi, range_max=10.0)
    scan = np.full((1, 1081), 3.0, np.float32)
    ca, cb = a.command(scan, np.array([1.0], np.float32)), b.command(scan, np.array([1.0], np.float32))
    assert cb[0, 0] == pytest.approx(ca[0, 0]) and cb[0, 1] == pytest.approx(1.5 * ca[0, 1])
