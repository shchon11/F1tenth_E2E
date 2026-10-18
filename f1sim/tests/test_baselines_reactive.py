"""The reactive baselines (`f1sim.learn.baselines.reactive`, 2026-10-17).

Follow-the-Gap is a port of ForzaETH's `controller/ftg/ftg.py`; the test EXECUTES the vendored file
(rospy and the marker messages stubbed, the stack's competition parameters fed through the stubbed
`rospy.get_param`) and requires the same command on the same scans, so a wrong line in the port is
caught rather than restated. The Disparity Extender has no upstream code; its tests check the
behaviour its authors describe.
"""
import math
import os
import sys
import types

import numpy as np
import pytest

from f1sim.learn import baselines
from f1sim.learn.baselines import reactive

VENDOR = os.path.join(os.path.dirname(__file__), "..", "..", "external", "baselines", "forzaeth_ftg", "ftg.py")
PARAMS = {"/state_machine/debug": False, "/state_machine/safety_radius": 40, "/state_machine/max_lidar_dist": 9.0,
          "/state_machine/max_speed": 6.0, "/state_machine/range_offset": 180, "/state_machine/track_width": 2.6}


def _upstream_ftg():
    rospy = types.ModuleType("rospy")
    rospy.get_param = lambda name, default=None: PARAMS.get(name, default)
    rospy.Publisher = lambda *a, **k: types.SimpleNamespace(publish=lambda *a, **k: None)
    rospy.Time = types.SimpleNamespace(now=lambda: 0)
    vm = types.ModuleType("visualization_msgs"); vmm = types.ModuleType("visualization_msgs.msg")
    vmm.Marker = vmm.MarkerArray = object
    saved = {k: sys.modules.get(k) for k in ("rospy", "visualization_msgs", "visualization_msgs.msg")}
    sys.modules.update({"rospy": rospy, "visualization_msgs": vm, "visualization_msgs.msg": vmm})
    try:
        ns = {"__name__": "forzaeth_ftg"}
        exec(compile(open(VENDOR).read(), VENDOR, "exec"), ns)
        return ns["FTG"]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _scans(n=40, seed=0):
    """Corridors with blocks in them: a 1081-beam 270 deg scan of walls 0.8-1.6 m to each side and
    boxes ahead, plus pure noise, the shapes the gap logic branches on."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(-0.75 * math.pi, 0.75 * math.pi, 1081)
    out = []
    for k in range(n):
        wl, wr = rng.uniform(0.8, 1.6, 2)
        with np.errstate(divide="ignore"):
            left = np.where(np.sin(ang) > 1e-3, wl / np.sin(ang), np.inf)
            right = np.where(np.sin(ang) < -1e-3, -wr / np.sin(ang), np.inf)
        r = np.minimum(np.minimum(left, right), 10.0)
        for _ in range(rng.integers(0, 3)):
            c = rng.integers(300, 780); w = rng.integers(10, 60)
            r[c:c + w] = np.minimum(r[c:c + w], rng.uniform(0.6, 4.0))
        if k % 7 == 6:
            r = rng.uniform(0.3, 10.0, 1081)
        out.append(r.astype(np.float32))
    return np.stack(out)


@pytest.mark.skipif(not os.path.exists(VENDOR), reason="vendored ForzaETH ftg.py not checked out")
def test_ftg_port_commands_exactly_what_forzaeth_ftg_commands():
    FTG = _upstream_ftg()
    theirs = FTG(mapping=False)
    ours = baselines.load("ftg", "none")
    scans = _scans()
    speeds = np.linspace(0.0, 5.0, scans.shape[0]).astype(np.float32)
    ours.bind_scanner(n_beams=1081, fov=1.5 * math.pi, range_max=10.0)
    cmd = ours.command(scans, speeds)
    for i in range(scans.shape[0]):
        theirs.set_vel(float(speeds[i]))
        speed, steer = theirs.process_lidar(list(scans[i].astype(np.float64)))
        assert abs(cmd[i, 0] - steer) < 1e-6, (i, cmd[i, 0], steer)
        assert abs(cmd[i, 1] - speed) < 1e-5, (i, cmd[i, 1], speed)


def test_ftg_max_speed_is_the_only_pace_knob():
    lo, hi = reactive.FollowTheGap(max_speed=6.0), reactive.FollowTheGap(max_speed=9.0)
    straight = _scans(1, seed=3)[0]
    a = lo._row(straight.astype(np.float64), 1.0)
    b = hi._row(straight.astype(np.float64), 1.0 * 9.0 / 6.0)   # same radius: v / max_speed held
    assert abs(a[0] - b[0]) < 1e-9 and abs(b[1] / a[1] - 1.5) < 1e-9


def _corridor(left=1.0, right=1.0):
    ang = np.linspace(-0.75 * math.pi, 0.75 * math.pi, 1081)
    with np.errstate(divide="ignore"):
        l = np.where(np.sin(ang) > 1e-3, left / np.sin(ang), np.inf)
        r = np.where(np.sin(ang) < -1e-3, -right / np.sin(ang), np.inf)
    return np.minimum(np.minimum(l, r), 10.0), ang


def test_disparity_extender_drives_straight_down_a_straight_corridor():
    d = baselines.load("disparity", "none", top_speed=6.0)
    r, _ = _corridor()
    steer, v = d._row(r)
    assert abs(steer) < 0.02 and v == pytest.approx(6.0)


def test_disparity_extender_steers_away_from_a_box_on_the_left_of_the_line():
    d = baselines.load("disparity", "none", top_speed=6.0)
    r, ang = _corridor(1.5, 1.5)
    box = (ang > math.radians(-2)) & (ang < math.radians(12))     # a box ahead, left of the centreline
    r[box] = np.minimum(r[box], 2.0)
    steer, v = d._row(r)
    assert steer < -0.02, steer                                     # goes round it on the right
    assert v < 6.0                                                  # and slows for it


def _left_turn_scene(side_lateral: float):
    """Open road ahead-left (8 m), a wall 3 m off on the right, and on the left side/rear (beyond
    +90 deg) a wall `side_lateral` metres from the car's centreline."""
    ang = np.linspace(-0.75 * math.pi, 0.75 * math.pi, 1081)
    r = np.full(1081, 3.0)
    r[(ang > 0) & (ang <= math.pi / 2)] = 8.0
    side = ang > math.pi / 2
    r[side] = np.minimum(side_lateral / np.sin(ang[side]), 10.0)
    return r


def test_disparity_extender_does_not_turn_into_a_wall_beside_it():
    d = baselines.load("disparity", "none", top_speed=6.0)
    free, _ = d._row(_left_turn_scene(1.0))
    assert free > 0.0                                               # nothing beside it: it turns left
    blocked, _ = d._row(_left_turn_scene(0.18))                     # 0.18 m < half width 0.155 + 0.1
    assert blocked == 0.0                                           # a wall right beside it: straight on


def _proc(ftg, scan):
    return ftg._safety_border(ftg._preprocess_lidar(np.asarray(scan, np.float64)))


def test_ftg_adaptive_thresholds_answer_their_own_question():
    """`radius_mode` (2026-10-19; the user: "distance threshold를 고정은 너무 좀 무식하고"): each mode's
    free-space threshold moves with what it reads; "upstream" is the port the test above pins."""
    f = reactive.FollowTheGap
    corridor, _ = _corridor(1.0, 1.0)
    # stopping: the stopping distance, r_min + v t_react + v^2 / 2 a_brake, clipped
    st = f(radius_mode="stopping", r_min=1.0, r_max=6.0, t_react=0.3, a_brake=4.0)
    p = _proc(st, corridor)
    assert abs(st._get_radius(1.0, p) - (1.0 + 0.3 + 1.0 / 8.0)) < 1e-9
    assert abs(st._get_radius(5.0, p) - min(6.0, 1.0 + 1.5 + 25.0 / 8.0)) < 1e-9
    # scan: an open scene looks further than a narrow corridor
    sc = f(radius_mode="scan", scan_gain=0.6, scan_quantile=0.5, r_min=0.5, r_max=8.0)
    open_ = np.full(1081, 9.0)
    narrow, _ = _corridor(0.6, 0.6)
    assert sc._get_radius(3.0, _proc(sc, open_)) > sc._get_radius(3.0, _proc(sc, narrow)) + 1.0
    # passable: the threshold it settles on leaves a gap the car fits through at that distance,
    # and an open scene does not lower it at all
    ps = f(radius_mode="passable", r_min=0.5, r_max=8.0, car_width=0.31, gap_margin=0.2)
    for s in _scans(20, seed=5):
        p = _proc(ps, s)
        r = ps._get_radius(3.0, p)
        lo, hi = ps._find_largest_gap(p, r)
        assert r == 0.5 or (hi - lo) * ps.radians_per_elem * r >= 0.51 - 1e-9, (r, lo, hi)
    assert ps._get_radius(3.0, _proc(ps, open_)) == 8.0
    # stopping_passable never looks further than the stopping distance
    sp = f(radius_mode="stopping_passable", r_min=0.5, r_max=8.0, t_react=0.3, a_brake=4.0)
    assert sp._get_radius(2.0, _proc(sp, open_)) <= 0.5 + 0.6 + 0.5 + 1e-9


def test_ftg_deepest_target_aims_at_the_farthest_beam_of_the_gap():
    ftg = reactive.FollowTheGap(target="deepest", radius_mode="passable", r_min=0.5, r_max=4.0)
    scan, ang = _corridor(1.2, 1.2)
    scan = np.asarray(scan, np.float64).copy()
    scan[np.abs(ang) < 0.3] = np.minimum(scan[np.abs(ang) < 0.3], 3.0)   # a wall 3 m ahead ...
    scan[(ang > 0.35) & (ang < 0.45)] = 9.0                   # ... and a deep opening 20-25 deg to the left
    d = ftg._decide(scan, 2.0)
    p = d["proc"]
    lo, hi = ftg._find_largest_gap(p, d["radius"])
    k = lo + int(np.argmax(p[lo:hi])) + ftg.range_offset - 180
    want = math.atan2(math.cos(k * ftg.radians_per_elem), math.sin(k * ftg.radians_per_elem))
    assert abs(d["steer"] - max(-0.4, min(0.4, want))) < 1e-9
    assert d["steer"] > 0.2                                    # it turns toward the opening
