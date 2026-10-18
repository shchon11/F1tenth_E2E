"""What the console draws for a non-policy driver is what that driver computed (2026-10-18).

The user: "각 방법에 맞는 시각화 방식을 적용해놔" -- and a drawing is only worth having if it is the
method's own decision rather than a look-alike. So each explanation is checked against the command
the driver actually issues: the pure-pursuit target must reproduce the teacher's steering, FTG's
and the Disparity Extender's `explain` must carry the command `command()` returns, and the spline
drawn must pass through the offset the planner commands where the car is.
"""
import math

import numpy as np
import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import baselines, common
from f1sim.viewer import driver_viz as DV


@pytest.fixture(scope="module")
def race_env():
    from f1sim.raceline import Raceline
    track = maps.load("real:korea_2025_iccas")
    lim = common.teacher_limits(7.0, 6.5, 4.0)
    rl = Raceline.build_cached(track, objective="min_time", **lim)
    cfg = Config(); cfg.sim.compile_mode = "none"
    ecfg = EnvConfig(max_steps=4000, hist_len=0, race_size=2, opponent="teacher", action_mode="direct")
    env = common.make_env([track], 8, "cpu", ecfg, cfg=cfg, seed=5, rls=[rl], teacher_limits=lim)
    env.reset()
    return env, rl, lim


def _pp_steer(t, state, P, target, origin):
    """`RacelineTeacher.__call__`'s pure-pursuit steering from a given target, calibration included."""
    yaw = state[:, 2]
    if P is not None:
        yaw = yaw + state[:, 5] * (P["cmd_delay"] + 0.5 * P["servo_tau"])
    dx, dy = target[:, 0] - origin[:, 0], target[:, 1] - origin[:, 1]
    alpha = torch.remainder(torch.atan2(dy, dx) - yaw + math.pi, 2 * math.pi) - math.pi
    ld = torch.sqrt(dx ** 2 + dy ** 2).clamp_min(1e-3)
    steer = torch.atan(2 * (t.L + t.k_us * state[:, 3] ** 2) * torch.sin(alpha) / ld).clamp(-t.steer_max, t.steer_max)
    if P is not None:
        steer = ((steer - P["steer_bias"]) / P["steer_gain"]).clamp(-t.steer_max, t.steer_max)
    return steer


@pytest.mark.parametrize("with_offset", [False, True])
def test_the_drawn_target_is_the_one_pure_pursuit_steers_at(race_env, with_offset):
    env, rl, lim = race_env
    t = common.make_teacher([rl], env, grip="true", **lim)
    for _ in range(30):
        env.step(torch.zeros(env.B, env.act_dim))
    state, P, tid = env.sim.state, env.sim.P, env.sim.tid
    offset = (torch.linspace(-0.3, 0.3, env.B) if with_offset else None)
    want = t(state, P, tid, offset=offset)[:, 0]
    ex = t.explain_rows(state, P, tid, offset=offset)
    got = _pp_steer(t, state, P, ex["target"], ex["origin"])
    assert torch.allclose(got, want, atol=1e-5), (got, want)
    # and the path starts where the car is on the line, running forward at the profile's speeds
    assert ex["line"].shape == (env.B, 49, 2) and bool((ex["v"] > 0).all())


def test_the_spline_drawn_passes_through_the_commanded_offset(race_env):
    env, rl, lim = race_env
    from f1sim.spliner_teacher import OVERTAKING, SplinerTeacher
    base = common.make_teacher([rl], env, grip="true", **lim)
    sp = SplinerTeacher(base, env=env)
    for _ in range(40):
        env.step(torch.zeros(env.B, env.act_dim))
        sp(env.sim.state, env.sim.P, env.sim.tid)
    rows = torch.arange(env.B)
    s = torch.linspace(0.0, 12.0, 49)
    d = sp.path_offsets(rows, s, env.sim.state[:, 3])
    assert torch.allclose(d[:, 0], sp.last_offset.float(), atol=1e-5)
    # Force a pass and check the same thing where the spline is not flat.
    sp.last_state = torch.full_like(sp.last_state, OVERTAKING)
    sp.last_spline = (torch.full((env.B,), 0.65), torch.full((env.B,), 3.0), torch.ones(env.B))
    d = sp.path_offsets(rows, s, env.sim.state[:, 3])
    at_car = 0.65 * sp._shape_at(torch.full((env.B,), -3.0))
    assert torch.allclose(d[:, 0], at_car, atol=1e-6) and float(d.abs().max()) > 0.6


def _scans(n=12, seed=0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(-0.75 * math.pi, 0.75 * math.pi, 1081)
    out = []
    for _ in range(n):
        wl, wr = rng.uniform(0.8, 1.6, 2)
        with np.errstate(divide="ignore"):
            left = np.where(np.sin(ang) > 1e-3, wl / np.sin(ang), np.inf)
            right = np.where(np.sin(ang) < -1e-3, -wr / np.sin(ang), np.inf)
        r = np.minimum(np.minimum(left, right), 10.0)
        c = rng.integers(300, 780)
        r[c:c + 40] = np.minimum(r[c:c + 40], rng.uniform(0.6, 4.0))
        out.append(r.astype(np.float32))
    return np.stack(out)


@pytest.mark.parametrize("kind", ["ftg", "disparity"])
def test_a_reactive_explanation_carries_the_command_it_explains(kind):
    drv = baselines.load(kind, "none")
    drv.bind_scanner(n_beams=1081, fov=1.5 * math.pi, range_max=10.0)
    scans = _scans()
    speeds = np.linspace(0.5, 5.0, len(scans)).astype(np.float32)
    cmd = drv.command(scans, speeds)
    for i in range(len(scans)):
        ex = drv.explain(scans[i], float(speeds[i]))
        assert ex["steer"] == pytest.approx(float(cmd[i, 0]), abs=1e-6)
        assert ex["speed"] == pytest.approx(float(cmd[i, 1]), abs=1e-5)
        # the aim point is ahead-ish and in the direction the car steers
        tx, ty = ex["target"]
        assert tx > -1e-9
        if abs(ex["steer"]) > 0.05 and not ex.get("side_block"):
            assert np.sign(ty) == np.sign(ex["steer"])


def test_the_command_arc_turns_the_way_the_command_steers():
    left = DV.command_arc((0.0, 0.0, 0.0), 0.3, 4.0, 0.33)
    right = DV.command_arc((0.0, 0.0, 0.0), -0.3, 4.0, 0.33)
    assert left[-1, 1] > 0.5 and right[-1, 1] < -0.5
    assert np.allclose(left[:, 2], 4.0)
    # one second of travel at the commanded speed along the arc
    seg = np.linalg.norm(np.diff(left[:, :2], axis=0), axis=1).sum()
    assert seg == pytest.approx(4.0, rel=0.02)


def test_the_panel_draws_the_driver_overlay_in_the_car_frame():
    """The 2D panel: a world point straight ahead of the car lands straight up from it, in the
    driver's colour, and the legend is the driver's."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from f1sim.viewer.console.overlays import PolicyInputPanel
    from f1sim.viewer.console.session import _drv_in_body_frame
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yaw = 0.7
    car = np.array([3.0, -1.0])
    ahead = car + 2.0 * np.array([math.cos(yaw), math.sin(yaw)])
    fr = {"focus": 0, "n": 1, "x": np.array([car[0]]), "y": np.array([car[1]]), "yaw": np.array([yaw]),
          "drv": {"title": "Follow-the-Gap", "legend": "범례",
                  "lines": [{"xy": np.stack([car, ahead]), "rgba": DV.TARGET_LINE, "w": 2.0}],
                  "points": [{"xy": ahead[None], "rgba": DV.TARGET, "px": 13.0}]}}
    body = _drv_in_body_frame(fr)
    assert np.allclose(body["points"][0]["xy"], [[2.0, 0.0]], atol=1e-5)
    panel = PolicyInputPanel()
    panel.resize(230, 230)
    panel.span = 4.0
    panel.set_data(np.full(1081, 3.0, np.float32), None, None, 6.0, drv=body)
    img = panel.grab().toImage()
    assert panel.toolTip() == "범례"
    # the aim point: 2 m straight up from the car marker, in TARGET's cyan
    w, h = panel.width(), panel.height()
    plot_h = max(40, h - 18)
    ppm = (min(w, plot_h) * 0.44) / 4.0
    cx, cy = w * 0.5, plot_h * 0.55
    c = img.pixelColor(int(round(cx)), int(round(cy - 2.0 * ppm)))
    assert c.blue() > 200 and c.green() > 200 and c.red() < 140, (c.red(), c.green(), c.blue())
