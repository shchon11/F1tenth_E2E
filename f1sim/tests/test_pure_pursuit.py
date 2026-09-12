"""The pure-pursuit geometry in `f1sim_ros.pure_pursuit_node`, without ROS."""
import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "f1sim_ros"))
pp = pytest.importorskip("f1sim_ros.pure_pursuit_node")


def _circle(r=5.0, n=200):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def test_on_a_circle_the_steer_matches_the_kinematic_bicycle():
    r, wb = 5.0, 0.3302
    path = _circle(r)
    # on the circle at angle 0, heading tangentially (+y), CCW
    steer, it, (tx, ty) = pp.pure_pursuit_step(r, 0.0, math.pi / 2, path, lookahead=1.0, wheelbase=wb)
    assert steer > 0                                            # left turn for a CCW circle
    assert steer == pytest.approx(math.atan(wb / r), rel=0.05)  # 1/r curvature, up to chord error
    assert math.hypot(tx - r, ty) >= 1.0 - 1e-9                 # target is at least the lookahead away


def test_target_is_ahead_not_behind():
    path = _circle(5.0)
    _steer, it, (tx, ty) = pp.pure_pursuit_step(5.0, 0.0, math.pi / 2, path, 1.5, 0.33)
    assert ty > 0                                               # forward along the CCW direction


def test_straight_line_offset_steers_back_toward_the_line():
    path = np.stack([np.linspace(0, 20, 400), np.zeros(400)], 1)
    steer_left, _, _ = pp.pure_pursuit_step(5.0, -0.5, 0.0, path, 1.5, 0.33)     # below the line
    steer_right, _, _ = pp.pure_pursuit_step(5.0, 0.5, 0.0, path, 1.5, 0.33)     # above the line
    assert steer_left > 0 > steer_right
    on_line, _, _ = pp.pure_pursuit_step(5.0, 0.0, 0.0, path, 1.5, 0.33)
    assert on_line == pytest.approx(0.0, abs=1e-9)


def test_steer_is_clamped_and_lookahead_follows_speed():
    path = _circle(0.6)
    steer, _, _ = pp.pure_pursuit_step(0.6, 0.0, math.pi / 2, path, 0.5, 0.33, max_steer=0.2)
    assert abs(steer) <= 0.2
    assert pp.lookahead_for_speed(0.0, 0.35, 0.8, 2.5) == 0.8
    assert pp.lookahead_for_speed(4.0, 0.35, 0.8, 2.5) == pytest.approx(1.4)
    assert pp.lookahead_for_speed(40.0, 0.35, 0.8, 2.5) == 2.5
