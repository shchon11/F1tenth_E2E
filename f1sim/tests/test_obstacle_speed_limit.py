"""The speed limit for an obstacle on the plan's own path (`f1sim/obstacle_speed_limit.py`, 2026-10-26): which scan returns
count as an obstacle (small isolated clusters, no map), the limit along the plan's path, and the env holding the learner's
speed command to it."""
from __future__ import annotations

import math

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.obstacle_speed_limit import DEFAULT, cluster_keep, speed_limit

N = 1081
ANG = torch.linspace(-2.356, 2.356, N)


def corridor(half_width=1.0, far=6.0):
    """ranges of a straight corridor: walls `half_width` to each side, a far wall `far` ahead"""
    return torch.minimum(half_width / ANG.sin().abs().clamp_min(1e-3), far / ANG.cos().clamp_min(1e-3))


def with_box(r, x, y, half=0.15):
    """an axis-aligned square of side 2 * half centred (x, y) metres from the LiDAR, ray-cast into `r`"""
    c, s = ANG.cos(), ANG.sin()
    with torch.no_grad():
        tx = torch.stack([(x - half) / c, (x + half) / c]); ty = torch.stack([(y - half) / s, (y + half) / s])
        t0 = torch.maximum(tx.min(0).values, ty.min(0).values); t1 = torch.minimum(tx.max(0).values, ty.max(0).values)
    hit = (t1 >= t0) & (t0 > 0.05) & (t0 < r)
    return torch.where(hit, t0, r), hit


def points(r):
    return 0.297 + r * ANG.cos(), r * ANG.sin()


def test_a_box_is_an_obstacle_and_walls_are_not():
    r, box = with_box(corridor(), 2.0, 0.0)
    # a 1.2 m piece of wall 1.5 m ahead on the left: long, so not an obstacle
    rw = 1.5 / ANG.cos().clamp_min(1e-3); yw = rw * ANG.sin()
    piece = (yw > 0.2) & (yw < 1.4) & (ANG.cos() > 0) & (rw < r)
    r = torch.where(piece, rw, r)
    ok = torch.isfinite(r)[None]
    px, py = points(r)
    keep = cluster_keep(r[None], ok, px[None], py[None])[0]
    assert int(box.sum()) >= 3 and bool(keep[box].all())
    assert not bool(keep[piece].any())
    assert not bool(keep[~box & ~piece].any())                 # the corridor walls


def straight_plan(length=4.0, b=1):
    return torch.zeros(b, 4), torch.full((b,), length)


def test_a_box_on_the_path_sets_the_speed_it_can_still_stop_from():
    r, _ = with_box(corridor(), 2.0, 0.0)
    k, L = straight_plan()
    lim = speed_limit(r[None], ANG, k, L, torch.tensor([3.0]))
    # box face 2.0 - 0.15 m from the LiDAR, i.e. 0.297 + 1.85 = 2.147 m from the reference point; the first path point
    # within 0.19 m of it is at s = 2.0 (0.1 m steps)
    s_hit = 2.0
    want = math.sqrt(2 * DEFAULT.brake * (s_hit - DEFAULT.front - DEFAULT.gap_min - 3.0 * DEFAULT.lag))
    assert float(lim) == pytest.approx(want, rel=1e-4)


@pytest.mark.parametrize("why, x, y, length", [
    ("beside the path", 2.0, 0.6, 4.0),
    ("beyond the plan's own length", 2.0, 0.0, 1.5),
    ("behind the front bumper", -0.5, 0.0, 4.0),
])
def test_no_limit(why, x, y, length):
    r, _ = with_box(corridor(), x, y)
    k, L = straight_plan(length)
    assert torch.isinf(speed_limit(r[None], ANG, k, L, torch.tensor([3.0]))).all(), why


def test_a_box_where_the_nearest_car_output_puts_a_car_is_left_to_the_car_limit():
    r, _ = with_box(corridor(), 2.0, 0.0)
    k, L = straight_plan()
    reading = torch.tensor([[4.0, 2.05 / 5.0, 0.0, 1.0 / 5.0, 0.0]])      # a car seen 2.05 m ahead of the LiDAR
    assert torch.isinf(speed_limit(r[None], ANG, k, L, torch.tensor([3.0]), car_reading=reading)).all()
    unseen = reading.clone(); unseen[0, 0] = -4.0
    assert torch.isfinite(speed_limit(r[None], ANG, k, L, torch.tensor([3.0]), car_reading=unseen)).all()


def test_the_env_holds_the_learners_speed_command_for_a_box_on_its_path():
    from f1sim import mpc
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"
    env = common.make_env([maps.load("real:korea_2025_iccas")], 2, "cpu",
                          EnvConfig(max_steps=100, hist_len=0, action_mode="plan", speed_mode="wp4",
                                    compile_tracker=False, obstacle_speed_limit=True), cfg=cfg, seed=1)
    env.reset(seed=1)
    a = torch.zeros(2, env.act_dim)
    a[:, 4:] = 0.9                                             # a straight plan, every waypoint speed near the top
    ang = env.sim.lidar.angles
    c, s_ = ang.cos(), ang.sin()

    def box_scan(face):
        """a 0.3 m wide face `face` m straight ahead of the LiDAR, a wall 8 m away elsewhere"""
        on = (c > 0) & ((face / c.clamp_min(1e-3)) * s_).abs().le(0.15)
        return torch.where(on, face / c.clamp_min(1e-3), torch.full_like(c, 8.0))

    for _ in range(3):
        env.step(a)
    env.last_result.scan = torch.full((2, ang.shape[0]), float("inf"))
    env.step(a)
    assert not bool(env.obstacle_limited.any())                # nothing in sight
    # a box 0.9 m ahead on row 0's scan (inside the plan's length at this speed): the command is held to the limit the module
    # computes from the same inputs
    env.last_result.scan = torch.stack([box_scan(0.9), torch.full_like(c, float("inf"))])
    v_meas = env._tracker_plan_speed()
    k, L = mpc.decode_wp4(a.float(), v_meas.float(), float(env.tracker.v_max), env.speed_cap.float(), env.tracker.spec)[:2]
    lim = speed_limit(env.last_result.scan, ang, k, L, env.sim.state[:, 3], float(cfg.lidar.mount_x), float(cfg.lidar.mount_y),
                      range_max=float(cfg.lidar.range_max))
    assert math.isfinite(float(lim[0])) and math.isinf(float(lim[1]))
    env.step(a)
    assert float(env.last_cmd_raw[0, 1]) <= float(lim[0]) + 1e-5
    assert not bool(env.obstacle_limited[1])
    # a box right at the bumper: nothing left to stop in, so the command is held to 0
    env.last_result.scan = torch.stack([box_scan(0.35), torch.full_like(c, float("inf"))])
    env.step(a)
    assert bool(env.obstacle_limited[0]) and float(env.last_cmd_raw[0, 1]) <= 1e-5
