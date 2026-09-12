"""The ROS link's message builders, without rclpy: numpy in, Python floats and plain structures out.

rclpy's generated messages assert that every field is a Python float, so a numpy scalar slipping
through is a crash on the first tick (it was, in `bridge_node.publish_odom`, 2026-09-12). These
tests pin the types as much as the values.
"""
import math

import numpy as np
import pytest

from f1sim.viewer import ros_link as rl


def _all_py_floats(seq):
    return all(type(v) is float for v in seq)


def test_map_to_odom_closes_the_chain_and_returns_python_floats():
    st = np.array([3.0, -1.0, 0.7, 2.0, 0.1, 0.3, 0.0], dtype=np.float32)   # ground truth
    od = np.array([0.5, 0.25, 0.1, 2.0, 0.3], dtype=np.float32)              # drifted
    mx, my, dyaw = rl.map_to_odom(st, od)
    assert _all_py_floats((mx, my, dyaw))
    # map->odom * odom->base == ground truth
    c, s = math.cos(dyaw), math.sin(dyaw)
    x = mx + c * float(od[0]) - s * float(od[1])
    y = my + s * float(od[0]) + c * float(od[1])
    assert x == pytest.approx(float(st[0]), abs=1e-6) and y == pytest.approx(float(st[1]), abs=1e-6)
    assert (dyaw + float(od[2])) == pytest.approx(float(st[2]), abs=1e-6)


def test_car_boxes_centre_on_the_silhouette_and_colour_by_role():
    pose = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, math.pi / 2], [9.0, 9.0, 0.0]], dtype=np.float32)
    dims = np.array([[0.5, 0.3, 0.2]] * 3, dtype=np.float32)
    rear = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    boxes = rl.car_boxes(pose, dims, rear, ros_car=0, rivals=[1])
    assert [b["id"] for b in boxes] == [0, 1, 2]
    # silhouette spans [-0.1, 0.4] along the body: centre 0.15 ahead of the CoG
    assert boxes[0]["x"] == pytest.approx(0.15) and boxes[0]["y"] == pytest.approx(0.0)
    assert boxes[1]["x"] == pytest.approx(5.0) and boxes[1]["y"] == pytest.approx(5.15)
    assert boxes[0]["color"] == rl.COLOR_ROS_CAR
    assert boxes[1]["color"] == rl.COLOR_RIVAL
    assert boxes[2]["color"] == rl.COLOR_OTHER
    for b in boxes:
        assert _all_py_floats((b["x"], b["y"], b["z"], b["yaw"], b["sx"], b["sy"], b["sz"]))


def test_prop_wireframe_is_a_line_list_in_the_map_frame():
    fp = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])     # unit square
    pts = rl.prop_wireframe(fp, height=0.4, x=10.0, y=2.0, yaw=math.pi / 2)
    assert len(pts) == 4 * 6 and len(pts) % 2 == 0                          # 3 segments per vertex
    assert all(_all_py_floats(p) for p in pts)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    assert min(xs) == pytest.approx(9.5) and max(xs) == pytest.approx(10.5)
    assert min(ys) == pytest.approx(1.5) and max(ys) == pytest.approx(2.5)
    assert set(round(z, 6) for z in zs) == {0.0, 0.4}


def test_speed_colors_run_blue_to_red_and_closed_path_repeats_the_first_point():
    cols = rl.speed_colors(np.array([1.0, 3.0, 5.0]))
    assert cols[0][2] > cols[0][0] and cols[-1][0] > cols[-1][2]            # slow blue, fast red
    assert len(rl.speed_colors(np.array([2.0, 2.0]))) == 2                  # flat profile: no divide by zero
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    closed = rl.closed_path(xy)
    assert closed.shape == (4, 2) and np.allclose(closed[-1], xy[0])
    assert rl.closed_path(closed).shape == (4, 2)                           # already closed: unchanged


def test_plan_world_rotates_the_body_frame_reference():
    ref = np.array([[1.0, 0.0, 0, 0], [0.0, 1.0, 0, 0]])
    w = rl.plan_world(ref, [2.0, 3.0, math.pi / 2])
    assert np.allclose(w, [[2.0, 4.0], [1.0, 3.0]], atol=1e-9)


def test_quaternions():
    assert rl.yaw_quat(0.0) == (0.0, 0.0, 0.0, 1.0)
    x, y, z, w = rl.rpy_quat(0.0, 0.0, math.pi / 2)
    assert (x, y) == (0.0, 0.0) and z == pytest.approx(math.sqrt(0.5)) and w == pytest.approx(math.sqrt(0.5))
