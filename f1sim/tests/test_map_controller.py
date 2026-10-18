"""The MAP controller (`f1sim.map_controller`, 2026-10-19) against ForzaETH's own code.

The planner baselines followed their path by pure pursuit until the user asked for ForzaETH's MAP
controller, warning that its published code and its paper disagree. So the port is checked against
the CODE itself: race_stack's `controller/controller/map.py` and `steering_lookup/lookup_steer_angle.py`,
loaded from the read-only clone in /data/ext_src (skipped where that clone is absent), are run on the
same scans of a path, the same poses, speeds, accelerations and opponents, and with our steering table
in their CSV format -- and the batched port must return their speed and steering, step after step,
their memory (the steering change limit, the trailing integral, the curvature they read one step late)
included. The table itself is checked against the physics it claims: at low speed and small steering
the car turns as the kinematic bicycle says.
"""
import importlib.util
import math
import os
import sys
import types

import numpy as np
import pytest
import torch

from f1sim import map_controller as M

RACE_STACK = "/data/ext_src/race_stack"
upstream_present = pytest.mark.skipif(not os.path.isdir(RACE_STACK), reason="ForzaETH race_stack clone not in /data/ext_src")


def _nominal_row():
    from f1sim import Config, maps
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    env = common.make_env([maps.load("real:korea_2025_iccas")], 1, "cpu",
                          EnvConfig(max_steps=100, hist_len=0, action_mode="direct"), cfg=Config(), seed=0)
    return {k: torch.tensor([float(v)]) for k, v in env.sim.params.nominal.items()}, env.sim


@pytest.fixture(scope="module")
def small_lut():
    """One grip level, ETH-PBL's steering grid, every fourth speed column (CPU-sized)."""
    P, sim = _nominal_row()
    steers, vels = M.lut_grids(float(P["s_max"][0]))
    vels = vels[::4].contiguous()
    table = M.build_lut(P, [float(P["mu"][0])], wheel=bool(sim.wheel_model), dt=float(sim.dt),
                        steers=steers, vels=vels, settle_s=2.0)
    return M.SteerLUT(steers, vels, table), P


def _load_upstream(lut: M.SteerLUT, tmp_path):
    """race_stack's map.py and lookup_steer_angle.py, verbatim, reading `lut` as their CSV."""
    share = tmp_path / "share"
    (share / "cfg").mkdir(parents=True)
    t = lut.table[0].numpy().astype(np.float64)
    csv = np.full((t.shape[0] + 1, t.shape[1] + 1), 0.0)
    csv[0, 1:] = lut.vels.numpy()
    csv[1:, 0] = lut.steers.numpy()
    csv[1:, 1:] = t
    np.savetxt(share / "cfg" / "TEST_lookup_table.csv", csv, delimiter=",")
    ament = types.ModuleType("ament_index_python")
    pk = types.ModuleType("ament_index_python.packages")
    pk.get_package_share_directory = lambda name: str(share)
    ament.packages = pk
    sys.modules["ament_index_python"] = ament
    sys.modules["ament_index_python.packages"] = pk
    pkg = types.ModuleType("steering_lookup")
    pkg.__path__ = []
    sys.modules["steering_lookup"] = pkg
    for name, path in (("steering_lookup.lookup_steer_angle",
                        f"{RACE_STACK}/system_identification/steering_lookup/steering_lookup/lookup_steer_angle.py"),
                       ("upstream_map", f"{RACE_STACK}/controller/controller/map.py")):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["upstream_map"], sys.modules["steering_lookup.lookup_steer_angle"]


def _path(seed: int, n: int = 500):
    """Waypoints 0.1 m apart (the spacing race_stack assumes) on a road of varying curvature, with a
    speed profile; race_stack's columns [x, y, v, d_ratio, s, kappa, psi, ax]."""
    rng = np.random.default_rng(seed)
    s = np.arange(n) * 0.1
    kappa = 0.6 * np.sin(s / (2.0 + rng.random())) + 0.3 * np.sin(s / 0.9 + rng.random() * 6)
    psi = np.cumsum(kappa) * 0.1
    x = np.cumsum(np.cos(psi)) * 0.1
    y = np.cumsum(np.sin(psi)) * 0.1
    v = 4.0 + 3.0 * np.cos(s / 3.0 + rng.random() * 6) - 1.5 * np.abs(kappa)
    arr = np.stack([x, y, v, np.full(n, 0.5), s, kappa, psi, np.zeros(n)], 1)
    return arr, psi


@upstream_present
def test_the_lookup_is_upstreams_lookup(small_lut, tmp_path):
    lut, _P = small_lut
    _m, lu = _load_upstream(lut, tmp_path)
    theirs = lu.LookupSteerAngle("TEST", print)
    rng = np.random.default_rng(1)
    acc = rng.uniform(-10.0, 10.0, 400)
    vel = rng.uniform(0.0, 11.0, 400)
    want = np.array([theirs.lookup_steer_angle(a, v) for a, v in zip(acc, vel)])
    got = lut.lookup(torch.tensor(acc, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32),
                     torch.zeros(400, dtype=torch.long), "code").numpy()
    assert np.allclose(got, want, atol=2e-5), np.abs(got - want).max()


@upstream_present
@pytest.mark.parametrize("params", [{}, M.SIM_PARAMS], ids=["NUC2", "SIM"])
def test_the_controller_is_upstreams_controller_step_for_step(small_lut, tmp_path, params):
    lut, P = small_lut
    up, _lu = _load_upstream(lut, tmp_path)
    p = M.MAPParams.from_dict(params)
    s_max = float(P["s_max"][0])
    for seed in range(3):
        wp, psi = _path(seed)
        theirs = up.MAP_Controller(p.t_clip_min, p.t_clip_max, p.m_l1, p.q_l1, p.speed_lookahead, p.lat_err_coeff,
                                   p.acc_scaler_for_steer, p.dec_scaler_for_steer, p.start_scale_speed, p.end_scale_speed,
                                   p.downscale_factor, p.speed_lookahead_for_steer, p.prioritize_dyn, p.trailing_gap,
                                   p.trailing_p_gain, p.trailing_i_gain, p.trailing_d_gain, p.blind_trailing_speed,
                                   p.loop_rate, "TEST", logger_info=lambda *a: None, logger_warn=lambda *a: None)
        ours = M.MAPController(lut, p, 1, "cpu", s_max)
        rng = np.random.default_rng(100 + seed)
        s_car, d, dpsi, v = 1.0, 0.1, 0.0, 3.0
        acc_buf = np.zeros(10)
        L = float(wp[-1, 4])
        for k in range(60):
            v = float(np.clip(v + rng.normal(0, 0.6), 0.3, 9.5))
            d = float(np.clip(d + rng.normal(0, 0.08), -0.7, 0.7))
            dpsi = float(np.clip(dpsi + rng.normal(0, 0.05), -0.35, 0.35))
            s_car += v * 0.025 * 0.5
            i = int(s_car / 0.1)
            nx, ny = -math.sin(psi[i]), math.cos(psi[i])
            x, y = wp[i, 0] + d * nx, wp[i, 1] + d * ny
            yaw = float(psi[i] + dpsi)
            new_acc = float(rng.uniform(-3, 3))
            acc_buf[1:] = acc_buf[:-1]
            acc_buf[0] = new_acc
            trailing = k >= 30
            opp = None
            if trailing:
                gap = float(rng.uniform(0.4, 6.0))
                opp = [s_car + gap, float(rng.uniform(-0.5, 0.5)), float(rng.uniform(0.0, 6.0)), False, bool(rng.random() < 0.7)]
            vs = v * math.cos(dpsi)
            frenet = np.array([s_car, d, vs, 0.0])
            sp_t, _a, _j, st_t, _pt, _l1, _idx = theirs.main_loop("TRAILING" if trailing else "GB_TRACK",
                                                                 np.array([[x, y, yaw]]), wp, v, opp, frenet,
                                                                 acc_buf.copy(), L)
            ours.push_imu(torch.tensor([[new_acc]]))
            f = lambda a: torch.tensor([a], dtype=torch.float32)
            kw = {}
            if trailing:
                kw = dict(trailing=torch.tensor([True]), opp_gap=f((opp[0] - s_car) % L), opp_vs=f(opp[2]),
                          opp_visible=torch.tensor([opp[4]]), ego_vs=f(vs))
            st_o, sp_o = ours(f(x).new_tensor([[x, y]]), f(yaw), f(v), torch.tensor(wp[None, :, :2], dtype=torch.float32),
                              torch.tensor(wp[None, :, 2], dtype=torch.float32), torch.tensor(wp[None, :, 5], dtype=torch.float32),
                              f(0.1), f(d), torch.zeros(1, dtype=torch.long), **kw)
            assert abs(float(sp_o) - float(sp_t)) < 2e-3, (seed, k, float(sp_o), float(sp_t))
            assert abs(float(st_o) - float(np.clip(st_t, -s_max, s_max))) < 2e-4, (seed, k, float(st_o), float(st_t))


def test_the_table_turns_like_the_bicycle_at_low_speed(small_lut):
    """From 1 to 1.5 m/s and under 0.03 rad the tyres barely slip: a_lat = v^2 tan(delta) / L within 6 %.
    (Below 1 m/s the speed loop holds the car a few per cent under its command against rolling
    resistance, and a_lat is taken at the speed it holds.)"""
    lut, P = small_lut
    L = float(P["lf"][0] + P["lr"][0])
    t = lut.table[0]
    checked = 0
    for vi, v in enumerate(lut.vels.tolist()):
        if v > 1.5:
            break
        if v < 1.0:
            continue
        for si, d in enumerate(lut.steers.tolist()):
            if 0.005 < d < 0.03:
                want = v * v * math.tan(d) / L
                assert abs(float(t[si, vi]) - want) / want < 0.06, (v, d, float(t[si, vi]), want)
                checked += 1
    assert checked >= 5


def test_the_switches_move_what_they_name():
    """`l1_speed`, `l1_point`, `steer_speed`: each changes the command only through what it names."""
    steers = torch.linspace(0, 0.4, 41)
    vels = torch.linspace(0.5, 10.0, 20)
    table = (vels[None, :] ** 2 * torch.tan(steers)[:, None] / 0.33)[None]          # a kinematic table
    lut = M.SteerLUT(steers, vels, table)
    s = torch.arange(200) * 0.1
    path = torch.stack([s, torch.zeros_like(s)], 1)[None]                            # a straight line, y = 0
    v_path = torch.full((1, 200), 6.0)
    kap = torch.zeros(1, 200)
    base = dict(pos=torch.tensor([[1.0, -0.4]]), yaw=torch.tensor([0.0]), v_now=torch.tensor([3.0]),
                path_xy=path, path_v=v_path, path_kappa=kap, spacing=torch.tensor([0.1]), d_global=torch.tensor([-0.4]),
                grip=torch.zeros(1, dtype=torch.long))

    def run(**kw):
        c = M.MAPController(lut, M.MAPParams(steer_rate_limit=10.0, **kw), 1, "cpu", 0.4189)
        c(**base)
        return c.last, c

    a, _ = run()
    b, _ = run(l1_speed="commanded")
    assert float(b["l1"]) > float(a["l1"])                    # planned 6 m/s against measured 3 m/s
    e, _ = run(l1_point="euclid")
    assert abs(float((e["l1_point"] - base["pos"]).norm()) - float(e["l1"])) < 1e-3   # at distance L1
    assert abs(float((a["l1_point"][0, 0] - 1.0)) - float(a["l1"])) < 0.06            # L1 of arc ahead
    m, _ = run(steer_speed="measured")
    assert float(m["lat_acc"]) < float(a["lat_acc"])          # v_t = 3 m/s against the planned 6
    dly, _ = run(delay_comp=0.1)
    assert int(dly["idx_near"]) == int(a["idx_near"]) + 3      # carried 3 m/s x 0.1 s = 0.3 m forward
