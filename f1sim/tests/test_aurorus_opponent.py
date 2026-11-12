"""`aurorus_opponent` (11-12): the batched port of AuRoRus's avoidance against the original.

Three kinds of check:
  * the original's own tests (`shared/frenet.rs`, tests module) with its numbers, on the port and on the scalar reference below;
  * numbers that follow from its config and our scanner (which beams, how many candidates, the switch's cycle counts);
  * the port against a scalar reference, one car at a time in plain Python floats, written from the Rust text function by
    function (`r_*` below keep the Rust names): single plans on random scenes, then whole control cycles over time.
The reference shares no code with the port. Where the original's test scene is mirror-symmetric the two may pick mirrored
paths (the sideways offsets -1.0 + k 0.1 are not exactly symmetric in floating point), so there only the original's own
assertions are made, on each."""
import bisect
import dataclasses
import math
import types

import numpy as np
import pytest
import torch

from f1sim.learn import aurorus_opponent as AO

PI = math.pi
F64 = torch.float64

# config/autonomous_control/frenet_overtaking.toml, as read 11-12
CFG = dict(kk_s=0.8, kd_s=0.0, look_ahead_gain_s=0.2, min_look_ahead_m=0.45, max_cross_track_m=1.5, lidar_downsample=5,
           desired_fov_deg=200.0, wall_clearance_m=0.2, switch_on_s=0.07, hysteresis_s=2.07, max_road_width_m=1.0,
           delta_road_width_m=0.1, path_point_distance_m=0.2, min_path_length_m=1.25, max_path_length_m=4.35,
           delta_path_length_m=1.5, path_fov_deg=30.0, robot_radius_m=0.35, k_jerk=0.01, k_length=0.1, k_distance=0.01,
           speed_decay_factor=0.999, min_speed_reduction_gain=0.7, decay_last_d_factor=0.99, weight_last_d=0.4,
           speed_curvature_exponent=0.3, braking_margin_m=0.3)


# ---------------------------------------------------------------- the scalar reference (from the Rust text)
def r_wrap(a):
    """`wrap_to_pi`"""
    w = (a + PI) % (2 * PI) - PI                                       # Python's % with a positive modulus = rem_euclid
    return w + 2 * PI if w <= -PI else w


class RLine:
    """`geometry/line.rs`: a closed polyline of (x, y, speed)"""

    def __init__(self, pts):
        self.p = [(float(x), float(y), float(v)) for x, y, v in pts]
        n = self.n = len(self.p)
        self.cum = [0.0]
        for i in range(n):
            a, b = self.p[i], self.p[(i + 1) % n]
            self.cum.append(self.cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.lap = self.cum[n]

    def seg_len(self, i):
        return self.cum[i + 1] - self.cum[i]

    def nearest(self, x, y, hint, window):
        n = self.n
        if hint is None:
            segs = list(range(n))
        else:
            first = (hint % n + n - 2) % n
            segs, covered = [], 0.0
            for k in range(n):                                         # take_while: the line before a segment <= window
                seg = (first + k) % n
                inside = covered <= window
                covered += self.seg_len(seg)
                if not inside:
                    break
                segs.append(seg)
        best = None
        for seg in segs:
            a, b = self.p[seg], self.p[(seg + 1) % n]
            dx, dy = b[0] - a[0], b[1] - a[1]
            len2 = dx * dx + dy * dy
            t = min(max(((x - a[0]) * dx + (y - a[1]) * dy) / len2, 0.0), 1.0) if len2 > 0 else 0.0
            px, py = a[0] + t * dx, a[1] + t * dy
            dist = math.hypot(x - px, y - py)
            if best is None or dist < best["distance"]:                # min_by: the first of equal minima
                best = dict(segment=seg, s=self.cum[seg] + t * self.seg_len(seg), x=px, y=py, distance=dist)
        return best

    def segment_at(self, s):
        s = s % self.lap
        return min(bisect.bisect_right(self.cum, s) - 1, self.n - 1)   # partition_point(c <= s) - 1

    def _seg_t(self, s):
        s = s % self.lap
        i = self.segment_at(s)
        ln = self.seg_len(i)
        return i, ((s - self.cum[i]) / ln if ln > 0 else 0.0)

    def at(self, s):
        i, t = self._seg_t(s)
        a, b = self.p[i], self.p[(i + 1) % self.n]
        return tuple(a[k] + t * (b[k] - a[k]) for k in range(3))

    def seg_heading(self, i):
        a, b = self.p[i % self.n], self.p[(i + 1) % self.n]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def heading_at(self, s):
        n = self.n
        i, t = self._seg_t(s)

        def vertex(j):
            before = self.seg_heading((j + n - 1) % n)
            return before + r_wrap(self.seg_heading(j % n) - before) / 2.0

        start, end = vertex(i), vertex(i + 1)
        return r_wrap(start + t * r_wrap(end - start))

    def lateral(self, x, y, heading, near):
        h = self.heading_at(near["s"])
        d = -math.sin(h) * (x - near["x"]) + math.cos(h) * (y - near["y"])
        return d, min(max(math.tan(r_wrap(heading - h)), -1.0), 1.0)

    def curvature_at(self, s):
        n = self.n
        i = self.segment_at(s)
        a, b, c = self.p[(i + n - 1) % n], self.p[i], self.p[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        sides = math.hypot(b[0] - a[0], b[1] - a[1]) * math.hypot(c[0] - b[0], c[1] - b[1]) * math.hypot(c[0] - a[0], c[1] - a[1])
        return 2.0 * cross / sides if sides > 0 else 0.0


def r_cubic(d0, slope0, d1, l):
    """`LateralCubic::new`"""
    rise = d1 - d0 - slope0 * l
    return [d0, slope0, slope0 / l + 3.0 * rise / (l * l), -(2.0 * rise + slope0 * l) / (l * l * l)]


def r_cubic_at(a, u):
    return a[0] + u * (a[1] + u * (a[2] + u * a[3]))


def r_cubic_slope(a, u):
    return a[1] + u * (2.0 * a[2] + 3.0 * u * a[3])


def r_samples(start, end, step):
    count = math.floor((end - start) / step + 1e-6) if step > 0 and end > start else 0
    return [start + k * step for k in range(int(count) + 1)]


def r_candidates(P, d_weight, s0, d0, slope0, within_fov):
    """`Planner::candidates`: end offsets outside, lengths inside"""
    out = []
    width = max(P["max_road_width_m"], 0.0)
    max_len = max(P["max_path_length_m"], P["min_path_length_m"])
    fov = math.radians(P["path_fov_deg"])
    for d1 in r_samples(-width, width, P["delta_road_width_m"]):
        for length in r_samples(P["min_path_length_m"], max_len, P["delta_path_length_m"]):
            if within_fov and abs(math.atan2(d1 - d0, length)) > fov:
                continue
            a = r_cubic(d0, slope0, d1, length)
            us = r_samples(0.0, length, min(P["path_point_distance_m"], length / 2.0))
            d = [r_cubic_at(a, u) for u in us]
            third = 6.0 * a[3]
            off = d[-1] - d_weight
            cost = P["k_jerk"] * (third * third * len(d)) + P["k_length"] / length + P["k_distance"] * off * off
            out.append(dict(s=[s0 + u for u in us], d=d, cost=cost, d1=d1, L=length))
    return out


def r_to_world(path, line, memo):
    """`to_world`: the path's points, step lengths and curvature in the world"""
    x, y = [], []
    for s, d in zip(path["s"], path["d"]):
        if s not in memo:
            memo[s] = (line.at(s), line.heading_at(s))
        (bx, by, _), h = memo[s]
        x.append(bx - d * math.sin(h)); y.append(by + d * math.cos(h))
    n = len(x)
    yaw = [math.atan2(y[i + 1] - y[i], x[i + 1] - x[i]) for i in range(n - 1)]
    ds = [math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(n - 1)]
    ds.append(ds[n - 2])
    curv = [0.0] * n
    for i in range(1, n - 1):
        mean = (ds[i - 1] + ds[i]) / 2.0
        curv[i] = r_wrap(yaw[i] - yaw[i - 1]) / mean if mean > 0 else 0.0
    curv[0] = curv[1]; curv[n - 1] = curv[n - 2]
    path.update(x=x, y=y, ds=ds, curv=curv)


def r_first_collision(path, obstacles, radius):
    along = 0.0
    for i in range(1, len(path["x"])):
        along += path["ds"][i - 1]
        for ox, oy in obstacles:
            dx, dy = path["x"][i] - ox, path["y"][i] - oy
            if dx * dx + dy * dy <= radius * radius:
                return along
    return None


def r_reach(cands, coll):
    """`reach`: None when one of the longest paths is clear, else the furthest any path gets"""
    longest = max(c["s"][-1] - c["s"][0] for c in cands)
    reach = 0.0
    for c, hit in zip(cands, coll):
        if hit is not None:
            reach = max(reach, hit)
        elif c["s"][-1] - c["s"][0] >= longest - 1e-9:
            return None
        else:
            reach = max(reach, sum(c["ds"][:-1]))
    return reach


def r_state():
    """`State::default()` + the caller's nearest-point hint"""
    return dict(hint=None, avoiding=False, since=None, d_weight=0.0, blocked_gain=1.0, previous_error=None)


def r_plan(P, st, line, free, obstacles, s0, d0, slope0, lookahead):
    """`Planner::plan`. free(x, y) -> bool. Updates st's d_weight / blocked_gain."""
    cands = r_candidates(P, st["d_weight"], s0, d0, slope0, True)
    if not cands:
        cands = r_candidates(P, st["d_weight"], s0, d0, slope0, False)
    memo = {}
    for c in cands:
        r_to_world(c, line, memo)
    radius = P["robot_radius_m"]
    coll = [r_first_collision(c, obstacles, radius) for c in cands]
    on = [all(free(c["x"][i], c["y"][i]) for i in range(1, len(c["x"]))) for c in cands]

    def cheapest(ok):
        best = None
        for i, c in enumerate(cands):
            if ok(i) and (best is None or c["cost"] < cands[best]["cost"]):
                best = i
        return best

    clear = cheapest(lambda i: coll[i] is None and on[i])
    if clear is not None:
        st["blocked_gain"] = 1.0
        best = clear
    else:
        st["blocked_gain"] *= P["speed_decay_factor"]
        best = cheapest(lambda i: coll[i] is None)
        if best is None:                                               # max_by(reached, then lower cost): the last of equals
            key = lambda i: (coll[i], -cands[i]["cost"])
            best = 0
            for i in range(1, len(cands)):
                if key(i) >= key(best):
                    best = i
    path = cands[best]
    reach = r_reach(cands, coll)
    st["d_weight"] = P["decay_last_d_factor"] * ((1.0 - P["weight_last_d"]) * st["d_weight"] + P["weight_last_d"] * path["d"][-1])
    n = len(path["x"])
    along, target = 0.0, n - 1
    for i in range(1, n):
        along += path["ds"][i - 1]
        if along >= lookahead:
            target = i
            break
    eps = float(np.finfo(np.float32).eps)
    pc = max(abs(path["curv"][target]), eps)
    lc = max(abs(line.curvature_at(path["s"][target])), eps)
    factor = min((lc / pc) ** P["speed_curvature_exponent"], 1.0)
    gain = min(max(st["blocked_gain"] * factor, min(P["min_speed_reduction_gain"], 1.0)), 1.0)
    return dict(free=clear is not None, d=path["d"], s=path["s"], xy=list(zip(path["x"], path["y"])),
                target=(path["x"][target], path["y"][target]), gain=gain, reach=reach, d_weight=st["d_weight"],
                blocked=st["blocked_gain"], d1=path["d1"], L=path["L"], collides=coll[best] is not None)


def r_stopping_speed(distance, margin, max_decel):
    return math.sqrt(2.0 * max(max_decel, 0.0) * max(distance - margin, 0.0))


# ---------------------------------------------------------------- the port behind the same result
def port_plans(core, tid, pose, s0, npt, look, obstacles, d_weight, blocked):
    """`AurorusCore._plan` for rows given as lists; obstacles: one list of (x, y) per row -> one dict per row, as `r_plan`"""
    R = len(tid)
    M = max(1, max(len(o) for o in obstacles))
    obs = torch.zeros(R, M, 2, dtype=F64); ok = torch.zeros(R, M, dtype=torch.bool)
    for r, o in enumerate(obstacles):
        if o:
            obs[r, :len(o)] = torch.tensor(o, dtype=F64); ok[r, :len(o)] = True
    t = lambda v: torch.tensor(v, dtype=F64)
    s0 = t(s0)
    target, gain, reach, dw, bl = core._plan(torch.tensor(tid), t(pose), s0, t(npt), t(look), obs, ok, t(d_weight), t(blocked))
    lp = core.last_plan
    out = []
    for r in range(R):
        sel = int(lp["chosen"][r]); n = int(core.cn[sel])
        out.append(dict(free=bool(lp["free"][r]), d=lp["d"][r, sel, :n].tolist(), s=(s0[r] + core.u[:n]).tolist(),
                        xy=[tuple(q) for q in lp["P"][r, sel, :n].tolist()], target=tuple(target[r].tolist()), gain=float(gain[r]),
                        reach=None if math.isinf(float(reach[r])) else float(reach[r]), d_weight=float(dw[r]),
                        blocked=float(bl[r]), d1=float(core.cd1[sel]), L=float(core.cL[sel]), collides=bool(lp["hit"][r, sel])))
    return out


def make_core(pts_per_track, free, n=1, **kw):
    """the port on lines given as lists of (x, y, v), all of one length"""
    A = torch.tensor(pts_per_track, dtype=F64)
    kw = dict(dict(n_beams=1081, fov=4.71238898, wheelbase=0.32, max_steer=0.4), **kw)
    return AO.AurorusCore(AO.Line(A[..., :2], A[..., 2]), free, n, **kw)


# ---------------------------------------------------------------- the original's test scenes (frenet.rs, tests)
def loop_line():
    """out along y = 0 and back along y = 6, points 0.2 m apart, 3 m/s"""
    return [(i * 0.2, 0.0, 3.0) for i in range(201)] + [(i * 0.2, 6.0, 3.0) for i in range(200, -1, -1)]


def corridor_free(x, y, band=None):
    """`corridor_grid()`: 800 x 60 cells of 0.05 m from (0, -1.5), all free (or only rows whose centre is within `band` of y = 0)"""
    col, row = math.floor((x - 0.0) / 0.05), math.floor((y + 1.5) / 0.05)
    ok = 0 <= col < 800 and 0 <= row < 60
    return ok and (band is None or abs(-1.5 + (row + 0.5) * 0.05) < band)


def corridor_free_t(band=None):
    def free(xy, tid):
        col, row = torch.floor(xy[..., 0] / 0.05), torch.floor((xy[..., 1] + 1.5) / 0.05)
        ok = (col >= 0) & (col < 800) & (row >= 0) & (row < 60)
        return ok if band is None else ok & ((-1.5 + (row + 0.5) * 0.05).abs() < band)
    return free


def both_plans(obstacles, d0=0.0, slope0=0.0, band=None, s0=5.0, lookahead=1.0):
    """the original's `planner.plan(&params(), &line, &grid, &obstacles, 5.0, d0, slope0, 1.0)` on the reference and the port"""
    pts = loop_line()
    ref = r_plan(CFG, r_state(), RLine(pts), lambda x, y: corridor_free(x, y, band), obstacles, s0, d0, slope0, lookahead)
    core = make_core([pts], corridor_free_t(band))
    port = port_plans(core, [0], [[s0, d0, math.atan(slope0)]], [s0], [[s0, 0.0]], [lookahead], [obstacles], [0.0], [1.0])[0]
    return [("reference", ref), ("port", port)]


def test_config_is_the_originals():
    mine = dataclasses.asdict(AO.AurorusConfig())
    assert mine.pop("max_speed") >= 1e3                                # the original's 4 m/s clamp is its simulator's
    assert mine == CFG


def test_the_cubic_meets_its_end_conditions():
    a = r_cubic(0.3, 0.2, -0.5, 2.5)
    assert abs(r_cubic_at(a, 0.0) - 0.3) < 1e-12 and abs(r_cubic_slope(a, 0.0) - 0.2) < 1e-12
    assert abs(r_cubic_at(a, 2.5) + 0.5) < 1e-12 and abs(r_cubic_slope(a, 2.5)) < 1e-12
    # the port's cubic: a car 0.3 off the line heading atan(0.2) across it; the candidate ending at -0.5 after 2.75 m
    pts = loop_line()
    core = make_core([pts], corridor_free_t())
    port_plans(core, [0], [[5.0, 0.3, math.atan(0.2)]], [5.0], [[5.0, 0.0]], [1.0], [[]], [0.0], [1.0])
    c = int(torch.nonzero((core.cd1 - (-0.5)).abs().lt(1e-9) & (core.cL - 2.75).abs().lt(1e-9)).flatten()[0])
    n = int(core.cn[c])
    d = core.last_plan["d"][0, c, :n].tolist()
    a = r_cubic(0.3, 0.2, core.cd1[c].item(), 2.75)
    assert max(abs(d[k] - r_cubic_at(a, 0.2 * k)) for k in range(n)) < 1e-12
    assert abs(d[0] - 0.3) < 1e-12 and abs((d[1] - d[0]) / 0.2 - 0.2) < 0.1


def test_the_path_field_of_view_is_symmetric():
    cands = r_candidates(CFG, 0.0, 0.0, 0.0, 0.0, True)
    ends = [c["d"][-1] for c in cands]
    assert abs(min(ends) + max(ends)) < 1e-9
    lim = 1.25 * math.tan(math.radians(30.0)) + 1e-9
    assert all(c["s"][-1] - c["s"][0] > 2.0 or abs(c["d"][-1]) <= lim for c in cands)
    pts = loop_line()
    core = make_core([pts], corridor_free_t())
    port_plans(core, [0], [[5.0, 0.0, 0.0]], [5.0], [[5.0, 0.0]], [1.0], [[]], [0.0], [1.0])
    keep = core.last_plan["keep"][0]
    end = core.last_plan["d"][0].gather(1, (core.cn - 1)[:, None]).squeeze(1)
    assert abs(float(end[keep].min() + end[keep].max())) < 1e-9
    assert bool(((core.u[core.cn - 1] > 2.0) | (end.abs() <= lim))[keep].all())
    # the same candidates, in the same order
    assert [(c["d1"], c["L"]) for c in cands] == [(float(a), float(b)) for a, b in zip(core.cd1[keep], core.cL[keep])]
    assert int(keep.sum()) == len(cands) == 57                         # 15 + 21 + 21 of the 63


def test_with_nothing_in_the_way_it_stays_on_the_line():
    for name, p in both_plans([]):
        assert p["free"], name
        assert all(abs(d) < 1e-9 for d in p["d"]), name
        assert abs(p["target"][0] - 6.0) < 0.21 and abs(p["target"][1]) < 1e-9, name
        assert p["gain"] == 1.0, name


def test_it_swerves_around_an_obstacle_on_the_line():
    obstacles = [(8.5, 0.0), (8.5, 0.1), (8.5, -0.1)]
    for name, p in both_plans(obstacles):
        assert p["free"] and p["reach"] is None, name
        for x, y in p["xy"]:
            assert corridor_free(x, y), name
            assert all(math.hypot(x - ox, y - oy) > CFG["robot_radius_m"] for ox, oy in obstacles), name
        assert abs(p["d"][-1]) > 0.3, name
        assert CFG["min_speed_reduction_gain"] <= p["gain"] < 1.0, name


def test_a_wall_of_obstacles_limits_the_reach():
    obstacles = [(8.0, i * 0.05) for i in range(-30, 31)]
    for name, p in both_plans(obstacles):
        assert p["reach"] is not None and 2.0 < p["reach"] < 3.5, name
        assert r_stopping_speed(p["reach"], 0.5, 4.0) < r_stopping_speed(p["reach"] + 1.0, 0.5, 4.0), name
    assert r_stopping_speed(0.3, 0.5, 4.0) == 0.0
    t = lambda v: torch.tensor(v, dtype=F64)
    assert float(AO.stopping_speed(t(0.3), 0.5, t(4.0))) == 0.0
    got = AO.stopping_speed(t([0.3, 0.4, 0.6, 1.2]), 0.3, t([8.0] * 4)).tolist()
    assert got == pytest.approx([r_stopping_speed(d, 0.3, 8.0) for d in (0.3, 0.4, 0.6, 1.2)], abs=1e-12)
    assert got == pytest.approx([0.0, 1.2649, 2.1909, 3.7947], abs=1e-4)


def test_without_a_free_path_it_still_keeps_clear_of_the_obstacles():
    obstacles = [(6.0, 0.0), (6.0, 0.1), (6.0, -0.1)]
    for name, p in both_plans(obstacles, band=0.3):
        assert not p["free"], name
        assert not p["collides"], name
        assert all(math.hypot(x - ox, y - oy) > CFG["robot_radius_m"] for x, y in p["xy"][1:] for ox, oy in obstacles), name


def test_a_path_starts_along_the_vehicle_slope():
    for name, p in both_plans([], d0=0.2, slope0=0.3):
        initial = (p["d"][1] - p["d"][0]) / (p["s"][1] - p["s"][0])
        assert abs(initial - 0.3) < 0.1, name


# ---------------------------------------------------------------- numbers that follow from the config and our scanner
def r_ray_angle(fov, n, i):
    """`LidarScan::ray_angle_rad`, in f32"""
    f = np.float32
    return f(0.0) if n <= 1 else -f(fov) / f(2.0) + f(i) * f(fov) / f(n - 1)


def r_fov_window(n, fov, want):
    """`reactive.rs::fov_window`, in f32"""
    f = np.float32
    fov, want = f(fov), f(want)
    if n <= 1 or want >= fov:
        return 0, n
    step = fov / f(n - 1)
    outside = min(int(np.ceil((fov - max(want, f(0.0))) / f(2.0) / step)), n // 2)
    return outside, n - outside


FOV = 4.71238898                                                       # `LidarParams.fov`, 1081 beams


def test_which_beams_it_reads():
    assert AO.fov_window(1081, FOV, math.radians(200.0)) == (140, 941)
    assert AO.fov_window(1, FOV, 1.0) == (0, 1) and AO.fov_window(1081, FOV, 5.0) == (0, 1081)
    assert AO.fov_window(1081, FOV, 0.0) == (540, 541)
    for n, fov, want in [(1081, FOV, math.radians(200.0)), (1081, FOV, 0.0), (1081, FOV, -1.0), (1081, FOV, FOV), (1081, FOV, 4.7),
                         (720, 6.2831855, 3.0), (271, FOV, 2.0), (2, 1.0, 0.5), (1080, 4.7, 1.0)]:
        assert AO.fov_window(n, fov, want) == r_fov_window(n, fov, want), (n, fov, want)
    core = make_core([loop_line()], corridor_free_t())
    assert core.beams.tolist() == list(range(140, 941, 5)) and len(core.beams) == 161
    assert core.beam_angle.tolist() == [float(r_ray_angle(FOV, 1081, i)) for i in range(140, 941, 5)]
    assert abs(core.beam_angle[0].item() + math.radians(100.0)) < 1e-5 and abs(core.beam_angle[80].item()) < 1e-6


def test_candidates_and_switch_counts():
    core = make_core([loop_line()], corridor_free_t())
    assert core.cd1.numel() == 63 and core.K == 22 and core.step == 0.2
    assert sorted(set(core.cn.tolist())) == [7, 14, 22] and sorted(set(core.cL.tolist())) == [1.25, 2.75, 4.25]
    assert core.cd1.tolist() == [d for d in r_samples(-1.0, 1.0, 0.1) for _ in range(3)]
    assert core.cL.tolist() == [1.25, 2.75, 4.25] * 21
    assert (core.n_on, core.n_off) == (3, 83)                          # on at the 4th cycle running, off at the 84th
    # the reference's switch at 25 ms cycles says the same
    st = r_state()
    seen = [1] * 3 + [0] + [1] * 3 + [0] + [1] * 10 + [0] * 83 + [1] + [0] * 90
    out = []
    for k, o in enumerate(seen, 1):
        r_switch(st, bool(o), k * 0.025, CFG["switch_on_s"], CFG["hysteresis_s"])
        out.append(st["avoiding"])
    on_at = out.index(True)
    assert on_at == 8 + 3                                              # the first run of 4: cycles 9..12 (index 11)
    assert all(out[on_at:on_at + 7 + 83 + 1 + 83])                     # 83 empty cycles, one point, 83 more: still avoiding
    assert out[on_at + 7 + 83 + 1 + 83] is False                       # the 84th empty cycle running


def r_switch(st, obstacles, now, on_s, off_s):
    """`Switch::update`: True when it has just turned to avoiding"""
    if obstacles == st["avoiding"]:
        st["since"] = None
        return False
    if st["since"] is None:
        st["since"] = now
    if now - st["since"] < (off_s if st["avoiding"] else on_s):
        return False
    st["avoiding"] = not st["avoiding"]
    st["since"] = None
    return st["avoiding"]


# ---------------------------------------------------------------- random closed lines with a band of free space about them
def wavy_radius(q, th):
    return q["R0"] * (1.0 + q["a2"] * math.cos(2.0 * th + q["p2"]) + q["a3"] * math.cos(3.0 * th + q["p3"]))


def wavy_tracks(seed=0, T=2, N=300):
    rng = np.random.default_rng(seed)
    par, pts = [], []
    for _ in range(T):
        q = dict(R0=rng.uniform(6.0, 8.0), a2=rng.uniform(0.05, 0.12), p2=rng.uniform(0, 2 * PI), a3=rng.uniform(0.03, 0.08),
                 p3=rng.uniform(0, 2 * PI), pv=rng.uniform(0, 2 * PI), cx=rng.uniform(-3.0, 3.0), cy=rng.uniform(-3.0, 3.0))
        row = []
        for i in range(N):
            th = 2 * PI * i / N
            r = wavy_radius(q, th)
            row.append((q["cx"] + r * math.cos(th), q["cy"] + r * math.sin(th), 4.0 + 2.0 * math.sin(th + q["pv"])))
        par.append(q); pts.append(row)
    return par, pts


def wavy_free(par, width):
    """free space = within width[t] of track t's line, radially: the same formula for one point and for tensors"""
    def scalar(t):
        q = par[t]
        return lambda x, y: abs(math.hypot(x - q["cx"], y - q["cy"]) - wavy_radius(q, math.atan2(y - q["cy"], x - q["cx"]))) < width[t]

    A = {k: torch.tensor([q[k] for q in par], dtype=F64) for k in par[0]}
    W = torch.tensor(width, dtype=F64)

    def batched(xy, tid):
        g = lambda v: v[tid].view(-1, *([1] * (xy.dim() - 2)))
        dx, dy = xy[..., 0] - g(A["cx"]), xy[..., 1] - g(A["cy"])
        th = torch.atan2(dy, dx)
        r = g(A["R0"]) * (1.0 + g(A["a2"]) * torch.cos(2.0 * th + g(A["p2"])) + g(A["a3"]) * torch.cos(3.0 * th + g(A["p3"])))
        return (torch.hypot(dx, dy) - r).abs() < g(W)
    return scalar, batched


def off_line(rl, s, d, dh=0.0):
    """the pose d to the left of the line at s, turned dh from its heading"""
    bx, by, _ = rl.at(s)
    h = rl.heading_at(s)
    return bx - d * math.sin(h), by + d * math.cos(h), h + dh


def ang_diff(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def test_line_matches_the_reference():
    par, pts = wavy_tracks(seed=3)
    L = AO.Line(torch.tensor(pts, dtype=F64)[..., :2], torch.tensor(pts, dtype=F64)[..., 2])
    RL = [RLine(p) for p in pts]
    rng = np.random.default_rng(5)
    tid = rng.integers(0, 2, 300)
    s = np.array([rng.uniform(-1.0, 2.0) * RL[t].lap for t in tid])    # before the start and past a lap too
    T, S = torch.tensor(tid), torch.tensor(s, dtype=F64)[:, None]
    xy, v = L.at(T, S)
    hd, cv = L.heading_at(T, S), L.curvature_at(T, S)
    for k in range(300):
        rl = RL[tid[k]]
        x, y, vv = rl.at(s[k])
        assert math.hypot(xy[k, 0, 0].item() - x, xy[k, 0, 1].item() - y) < 1e-10 and abs(v[k, 0].item() - vv) < 1e-10
        assert ang_diff(hd[k, 0].item(), rl.heading_at(s[k])) < 1e-10
        assert abs(cv[k, 0].item() - rl.curvature_at(s[k])) < 1e-9
    assert abs(float(L.lap[0]) - RL[0].lap) < 1e-10
    # the nearest point: without a hint, and searched forward from hints at, before and far from the true segment
    P, H, Wn, exp = [], [], [], []
    for k in range(300):
        rl = RL[tid[k]]
        x, y, _ = off_line(rl, s[k], rng.uniform(-1.2, 1.2))
        true = rl.nearest(x, y, None, 0.0)["segment"]
        hint = [None, true, (true - int(rng.integers(0, 12))) % rl.n, (true + int(rng.integers(1, 4))) % rl.n,
                int(rng.integers(0, rl.n))][k % 5]
        w = float(rng.uniform(0.5, 3.0))
        P.append((x, y)); H.append(-1 if hint is None else hint); Wn.append(w); exp.append(rl.nearest(x, y, hint, w))
    seg, ss, pt, dist = L.nearest(T, torch.tensor(P, dtype=F64), torch.tensor(H), torch.tensor(Wn, dtype=F64))
    for k in range(300):
        e = exp[k]
        assert int(seg[k]) == e["segment"], k
        assert abs(ss[k].item() - e["s"]) < 1e-10 and abs(dist[k].item() - e["distance"]) < 1e-10
        assert math.hypot(pt[k, 0].item() - e["x"], pt[k, 1].item() - e["y"]) < 1e-10
    assert len({e["segment"] for k, e in enumerate(exp) if k % 5 == 4} ^ {e["segment"] for k, e in enumerate(exp) if k % 5 == 0})


def test_a_line_that_comes_back_close_does_not_take_the_car_over():
    """out along y = 0, back along y = 0.4: the hint keeps the car on the leg it is on; of equal distances the first wins"""
    pts = [(i * 0.2, 0.0, 3.0) for i in range(51)] + [(i * 0.2, 0.4, 3.0) for i in range(50, -1, -1)]
    L = AO.Line(torch.tensor([pts], dtype=F64)[..., :2], torch.tensor([pts], dtype=F64)[..., 2])
    rl = RLine(pts)
    t = lambda v: torch.tensor(v, dtype=F64)
    z = torch.zeros(1, dtype=torch.long)
    for (x, y), hint, want_y, want_d in [((5.0, 0.3), None, 0.4, 0.1), ((5.0, 0.3), 25, 0.0, 0.3), ((5.0, 0.2), None, 0.0, 0.2),
                                         ((5.1, 0.1), 77, 0.4, 0.3)]:
        e = rl.nearest(x, y, hint, 2.0)
        seg, s, pt, dist = L.nearest(z, t([[x, y]]), torch.tensor([-1 if hint is None else hint]), t([2.0]))
        assert abs(e["y"] - want_y) < 1e-12 and abs(e["distance"] - want_d) < 1e-12
        assert int(seg[0]) == e["segment"] and abs(pt[0, 1].item() - want_y) < 1e-12 and abs(dist[0].item() - want_d) < 1e-12


# ---------------------------------------------------------------- single plans on random scenes: port == reference
def test_plans_match_the_reference_on_random_scenes():
    par, pts = wavy_tracks(seed=11)
    width = [0.8, 0.28]                                                # track 1: free space narrower than the car's radius
    f_scalar, f_batched = wavy_free(par, width)
    RL = [RLine(p) for p in pts]
    core = make_core(pts, f_batched, n=1)
    rng = np.random.default_rng(17)
    scenes = []
    for k in range(48):
        kind = k % 6
        tid = 1 if kind == 5 else int(rng.integers(0, 2)) if kind == 0 else 0
        rl = RL[tid]
        s = float(rng.uniform(0.0, rl.lap))
        lat = float(rng.uniform(-0.2, 0.2) if tid == 1 else rng.uniform(-0.6, 0.6))
        x, y, h = off_line(rl, s, lat, float(rng.uniform(-1.0, 1.0) if kind == 1 else rng.uniform(-0.3, 0.3)))
        mark = lambda ahead, d: off_line(rl, s + ahead, d)[:2]
        obstacles = []
        if kind in (2, 5):                                             # a few clusters of points ahead
            for _ in range(int(rng.integers(1, 4)) if kind == 2 else 1):
                a0 = float(rng.uniform(1.0, 4.0) if kind == 2 else rng.uniform(0.8, 1.5))   # kind 5: within every path's length
                d0 = float(rng.uniform(-0.8, 0.8) if kind == 2 else rng.uniform(-0.05, 0.05))
                obstacles += [mark(a0 + float(rng.normal(0, 0.05)), d0 + float(rng.normal(0, 0.05))) for _ in range(int(rng.integers(3, 7)))]
        if kind in (3, 4):                                             # points right across: beyond the short paths / before every end
            a0 = float(rng.uniform(2.2, 2.6) if kind == 3 else rng.uniform(0.7, 1.0))
            obstacles = [mark(a0, -1.8 + 0.06 * j + float(rng.normal(0, 0.005))) for j in range(61)]
        scenes.append(dict(tid=tid, pose=(x, y, h), look=float(rng.uniform(0.45, 1.9)), obstacles=obstacles,
                           d_weight=float(rng.uniform(-0.5, 0.5)), blocked=float(rng.uniform(0.7, 1.0))))
    ref, s0s, npts = [], [], []
    for sc in scenes:
        rl = RL[sc["tid"]]
        x, y, h = sc["pose"]
        near = rl.nearest(x, y, None, 0.0)
        d0, slope0 = rl.lateral(x, y, h, near)
        st = dict(r_state(), d_weight=sc["d_weight"], blocked_gain=sc["blocked"])
        ref.append(r_plan(CFG, st, rl, f_scalar(sc["tid"]), sc["obstacles"], near["s"], d0, slope0, sc["look"]))
        s0s.append(near["s"]); npts.append((near["x"], near["y"]))
    # the port: all scenes in one call (mixed tracks, obstacle lists of different lengths), its own nearest point
    T = torch.tensor([sc["tid"] for sc in scenes])
    _, s0p, nptp, _ = core.line.nearest(T, torch.tensor([sc["pose"][:2] for sc in scenes], dtype=F64),
                                        torch.full((len(scenes),), -1), torch.zeros(len(scenes), dtype=F64))
    assert (s0p - torch.tensor(s0s, dtype=F64)).abs().max().item() < 1e-10
    port = port_plans(core, T.tolist(), [sc["pose"] for sc in scenes], s0p.tolist(), nptp.tolist(), [sc["look"] for sc in scenes],
                      [sc["obstacles"] for sc in scenes], [sc["d_weight"] for sc in scenes], [sc["blocked"] for sc in scenes])
    for k, (a, b) in enumerate(zip(ref, port)):
        assert (a["free"], a["collides"], a["reach"] is None) == (b["free"], b["collides"], b["reach"] is None), k
        assert abs(a["d1"] - b["d1"]) < 1e-9 and a["L"] == b["L"], k
        assert max(abs(u - v) for u, v in zip(a["d"], b["d"])) < 1e-9 and len(a["d"]) == len(b["d"]), k
        assert max(math.hypot(u[0] - v[0], u[1] - v[1]) for u, v in zip(a["xy"], b["xy"])) < 1e-9, k
        assert math.hypot(a["target"][0] - b["target"][0], a["target"][1] - b["target"][1]) < 1e-9, k
        assert a["reach"] is None or abs(a["reach"] - b["reach"]) < 1e-9, k
        assert abs(a["d_weight"] - b["d_weight"]) < 1e-12 and abs(a["blocked"] - b["blocked"]) < 1e-12, k
        assert abs(a["gain"] - b["gain"]) < 1e-8, k
    # every branch of the choice was met
    assert sum(a["free"] for a in ref) >= 10
    assert sum(not a["free"] and not a["collides"] for a in ref) >= 3  # no free path, the cheapest that touches no point
    assert sum(a["collides"] for a in ref) >= 3                        # every path touches: the one that gets furthest
    assert sum(a["reach"] is not None for a in ref) >= 6
    assert sum(0.7 < a["gain"] < 1.0 for a in ref) >= 5 and sum(a["gain"] == 1.0 for a in ref) >= 1
    assert sum(abs(a["d1"]) > 0.05 for a in ref) >= 10


# ---------------------------------------------------------------- whole control cycles over time: port == reference
def r_find_obstacles(P, scan, pose, mount, free, n_beams=1081, fov=FOV, range_min=0.02, range_max=10.0):
    """`find_obstacles`. The scanner's place is passed as plain floats (the original holds it in f32)."""
    x, y, h = pose
    c, s = math.cos(h), math.sin(h)
    ox, oy = x + mount[0] * c - mount[1] * s, y + mount[0] * s + mount[1] * c
    lo, hi = r_fov_window(n_beams, fov, math.radians(P["desired_fov_deg"]))
    out = []
    for i in range(lo, hi, max(P["lidar_downsample"], 1)):
        rng = float(scan[i])
        if rng <= float(np.float32(range_min)) or rng >= float(np.float32(range_max)) or rng >= P["max_path_length_m"]:
            continue
        ang = h + float(r_ray_angle(fov, n_beams, i))
        pt = (ox + rng * math.cos(ang), oy + rng * math.sin(ang))
        if free(*pt):
            out.append(pt)
    return out


def r_pd(P, error, st, now):
    """`steering.rs::pd`"""
    derivative = 0.0
    if st["previous_error"] is not None:
        prev, at = st["previous_error"]
        if now - at > 0:
            derivative = min(max(P["kd_s"] * (error - prev) / (now - at), -0.2), 0.2)
    st["previous_error"] = (error, now)
    return P["kk_s"] * error + derivative


def r_control(P, st, line, free, obstacles, pose, speed, now, scale, wheelbase, max_steer, max_decel, max_speed=1e3):
    """`control` and what its caller does with the result (the hint; off the line: stop and forget) -> steering, speed, plan"""
    x, y, h = pose
    lookahead = P["look_ahead_gain_s"] * max(speed, 0.0) + P["min_look_ahead_m"]
    near = line.nearest(x, y, st["hint"], 1.5 * lookahead + 1.0)
    if near["distance"] > P["max_cross_track_m"]:
        st.update(r_state())
        return 0.0, 0.0, None
    if r_switch(st, len(obstacles) > 0, now, P["switch_on_s"], P["hysteresis_s"]):
        st["d_weight"], st["blocked_gain"] = 0.0, 1.0                  # `Planner::default()`
    reference = scale * line.at(near["s"])[2]
    plan = None
    if st["avoiding"]:
        d0, slope0 = line.lateral(x, y, h, near)
        plan = r_plan(P, st, line, free, obstacles, near["s"], d0, slope0, lookahead)
        target, v = plan["target"], reference * plan["gain"]
        if plan["reach"] is not None:
            v = min(v, r_stopping_speed(plan["reach"], P["braking_margin_m"], max_decel))
    else:
        tx, ty, _ = line.at(near["s"] + lookahead)
        target, v = (tx, ty), reference
    bx, by = x - wheelbase * math.cos(h), y - wheelbase * math.sin(h)
    error = r_wrap(math.atan2(target[1] - by, target[0] - bx) - h)
    steering = r_pd(P, error, st, now)
    st["hint"] = near["segment"]
    return min(max(steering, -max_steer), max_steer), min(max(v, 0.0), max_speed), plan


def disc_scan(origin, heading, discs, n=1081, fov=FOV):
    """ranges of n beams from `origin` to the nearest of the discs (x, y, radius); inf where a beam meets none"""
    ang = heading + (-fov / 2 + np.arange(n) * fov / (n - 1))
    ux, uy = np.cos(ang), np.sin(ang)
    best = np.full(n, np.inf)
    for cx, cy, r in discs:
        dx, dy = cx - origin[0], cy - origin[1]
        b = dx * ux + dy * uy
        disc = r * r - (dx * dx + dy * dy) + b * b
        t = b - np.sqrt(np.maximum(disc, 0.0))
        best = np.where((disc >= 0) & (t > 0), np.minimum(best, t), best)
    return best.astype(np.float32)


WB, STEER_MAX, DT = 0.3302, 0.4189, 0.025


def run_cycles(n_cycles, kd_s=0.0):
    """four cars, each cycle the same pose / speed / scan to the port and to the reference; the cars move on the PORT's
    commands. car 0: nothing in the way; car 1: a disc on its line; car 2: discs right across the track; car 3: a disc off the
    free space beside it, then put 2 m off its line for five cycles."""
    par, pts = wavy_tracks(seed=23)
    f_scalar, f_batched = wavy_free(par, [0.8, 0.8])
    RL = [RLine(p) for p in pts]
    tid = [0, 1, 0, 1]
    scale, a_brake = [0.6, 0.8, 1.0, 0.7], [5.0, 5.0, 8.0, 5.0]
    mount = [(0.0, 0.0), (0.12555, 0.0), (0.297, 0.01), (0.12555, -0.02)]
    start = [0.0, 0.0, RL[0].lap / 2, RL[1].lap / 2]
    discs = [[off_line(RL[0], start[2] + 5.0, -1.2 + 0.2 * j)[:2] + (0.12,) for j in range(13)],
             [off_line(RL[1], 4.0, 0.1)[:2] + (0.15,), off_line(RL[1], start[3] + 3.0, 1.6)[:2] + (0.2,)]]
    P = dict(CFG, kd_s=kd_s)
    core = make_core(pts, f_batched, n=4, wheelbase=WB, max_steer=STEER_MAX, dt=DT, cfg=AO.AurorusConfig(kd_s=kd_s),
                     mount=(torch.tensor([m[0] for m in mount], dtype=F64), torch.tensor([m[1] for m in mount], dtype=F64)))
    S = [r_state() for _ in range(4)]
    car = [list(off_line(RL[tid[c]], start[c], 0.0)) + [3.0] for c in range(4)]       # x, y, heading, speed
    log = dict(avoid=[[] for _ in range(4)], lost=[[] for _ in range(4)], reach=[[] for _ in range(4)], free=[[] for _ in range(4)],
               speed=[[] for _ in range(4)], gap=[[] for _ in range(4)], raw=[[] for _ in range(4)], blocked=[[] for _ in range(4)])
    for k in range(1, n_cycles + 1):
        poses, scans = [], []
        for c in range(4):
            x, y, h, _ = car[c]
            if c == 3 and 50 <= k < 55:                                # lifted 2 m to the left of where it is
                x, y = x - 2.0 * math.sin(h), y + 2.0 * math.cos(h)
            ox = x + mount[c][0] * math.cos(h) - mount[c][1] * math.sin(h)
            oy = y + mount[c][0] * math.sin(h) + mount[c][1] * math.cos(h)
            poses.append((x, y, h)); scans.append(disc_scan((ox, oy), h, discs[tid[c]]))
        steer, v = core.cycle(torch.tensor(poses, dtype=F64), torch.tensor([q[3] for q in car], dtype=F64),
                              torch.tensor(np.stack(scans)), torch.tensor(tid), torch.tensor(a_brake, dtype=F64),
                              torch.tensor(scale, dtype=F64))
        for c in range(4):
            st = S[c]
            obstacles = r_find_obstacles(P, scans[c], poses[c], mount[c], f_scalar(tid[c]))
            rs, rv, plan = r_control(P, st, RL[tid[c]], f_scalar(tid[c]), obstacles, poses[c], car[c][3], k * DT, scale[c], WB,
                                     STEER_MAX, a_brake[c])
            where = (k, c)
            assert abs(steer[c].item() - rs) < 1e-8 and abs(v[c].item() - rv) < 1e-8, where
            assert int(core.hint[c]) == (-1 if st["hint"] is None else st["hint"]), where
            assert bool(core.avoiding[c]) == st["avoiding"] and (int(core.since[c]) >= 0) == (st["since"] is not None), where
            assert abs(core.d_weight[c].item() - st["d_weight"]) < 1e-9, where
            assert abs(core.blocked_gain[c].item() - st["blocked_gain"]) < 1e-12, where
            pe = core.prev_err[c].item()
            assert math.isnan(pe) == (st["previous_error"] is None), where
            assert st["previous_error"] is None or abs(pe - st["previous_error"][0]) < 1e-9, where
            assert int(core.last["n_obst"][c]) == len(obstacles), where
            log["avoid"][c].append(st["avoiding"]); log["lost"][c].append(st["hint"] is None)
            log["reach"][c].append(plan is not None and plan["reach"] is not None)
            log["free"][c].append(plan is None or plan["free"]); log["blocked"][c].append(st["blocked_gain"])
            log["raw"][c].append(int((scans[c] < CFG["max_path_length_m"]).sum()))
            # the car moves: speed toward the command (up at 5 m/s^2, down at the braking the controller was told), kinematic bicycle
            x, y, h, sp = car[c]
            sp += min(max(v[c].item() - sp, -a_brake[c] * DT), 5.0 * DT)
            car[c] = [x + sp * math.cos(h) * DT, y + sp * math.sin(h) * DT, h + sp / WB * math.tan(steer[c].item()) * DT, sp]
            log["speed"][c].append(sp)
            log["gap"][c].append(min(math.hypot(car[c][0] - dx, car[c][1] - dy) - r for dx, dy, r in discs[tid[c]]))
    return log


def test_control_cycles_match_the_reference_over_time():
    log = run_cycles(230)
    a = np.array
    assert not any(log["avoid"][0]) and not any(log["lost"][0])        # nothing in the way: on its line throughout
    # car 1: sees the disc, goes around it without touching, and is back on the line 84 cycles after the last point
    av = a(log["avoid"][1])
    on, off = int(np.argmax(av)), len(av) - int(np.argmax(av[::-1]))
    assert on == 3 and av[on:off].all() and 100 < off < 230 and not av[off:].any()
    assert min(log["gap"][1]) > 0.2 and a(log["free"][1]).all()        # centre to the disc's edge; half the car's width is 0.155
    # car 2: discs right across: no path gets through, it brakes to a stop before them and stays (its centre 0.31 m from the
    # discs: the original's 0.35 m radius about the centre of gravity and 0.3 m braking margin leave about the nose's length)
    assert a(log["reach"][2]).sum() > 200 and (~a(log["free"][2])).sum() > 150 and a(log["avoid"][2])[4:].all()
    assert max(log["speed"][2][60:]) == 0.0 and 0.3 < min(log["gap"][2]) < 0.4
    assert log["blocked"][2][-1] < 0.999 ** 150                        # the speed gain kept decaying while nothing was free
    # car 3: the disc beside it is off the free space (beams hit it, no obstacle points); off its line -> 0, 0 and no hint
    assert max(log["raw"][3][:49]) > 0 and not any(log["avoid"][3][:49])
    assert log["lost"][3][49:54] == [True] * 5 and sum(log["lost"][3]) == 5


def test_control_cycles_match_with_the_derivative_term():
    log = run_cycles(60, kd_s=0.3)
    assert any(log["avoid"][1]) and sum(log["lost"][3]) == 5


# ---------------------------------------------------------------- the opponents' driver on a stand-in environment
def stand_in_env(n_races=6, mount_in_P=False):
    """rows 2k = the learner's car, 2k + 1 = the other car of race k; only what `AurorusOpponents` reads and writes"""
    par, pts = wavy_tracks(seed=31)
    f_scalar, f_batched = wavy_free(par, [0.8, 0.8])
    A = torch.tensor(pts, dtype=torch.float32)                         # the environment's lines are float32
    RL = [RLine(A[t].double().tolist()) for t in range(2)]
    B = 2 * n_races
    tid = (torch.arange(B) // 2) % 2
    state = torch.zeros(B, 8)
    for b in range(B):
        x, y, h = off_line(RL[int(tid[b])], 2.0 + 0.37 * b, 0.05 * (b % 3 - 1))
        state[b, AO.dyn.IX], state[b, AO.dyn.IY], state[b, AO.dyn.IYAW], state[b, AO.dyn.IVX] = x, y, h, 2.0 + 0.01 * b
    P = dict(lf=torch.full((B,), 0.15875), lr=torch.full((B,), 0.17145), s_max=torch.full((B,), 0.4189), a_brake=torch.full((B,), 5.0))
    if mount_in_P:                                                     # the scanner's place randomized per car
        P["mount_x"], P["mount_y"] = 0.1 + 0.01 * torch.arange(B), 0.002 * torch.arange(B) - 0.01
    sim = types.SimpleNamespace(
        state=state, tid=tid, P=P, control_dt=0.025, absent=torch.zeros(B, dtype=torch.bool), absent_any=False,
        ideal_vcap=torch.full((B,), 5.0),
        track=types.SimpleNamespace(sample_edt=lambda xy, t: f_batched(xy.double(), t).float()))      # 1 m from a wall where free
    env = types.SimpleNamespace(
        B=B, device=torch.device("cpu"), learner=torch.arange(B) % 2 == 0, sim=sim,
        cfg=types.SimpleNamespace(lidar=types.SimpleNamespace(fov=FOV, range_min=0.02, range_max=10.0, mount_x=0.297, mount_y=0.0)),
        teacher=types.SimpleNamespace(xy=A[..., :2].clone(), v=A[..., 2].clone()),
        last_result=types.SimpleNamespace(scan=torch.full((B, 1081), float("inf"))),
        ideal_cmd=torch.zeros(B, 2), ideal_extra=None, level_seed=torch.full((B,), -1, dtype=torch.long))
    return env, RL, f_scalar


@pytest.mark.parametrize("mount_in_P", [False, True])
def test_the_opponents_cars_get_the_controllers_commands(mount_in_P):
    env, RL, f_scalar = stand_in_env(6, mount_in_P)
    B, ix, iy, ih, iv = env.B, AO.dyn.IX, AO.dyn.IY, AO.dyn.IYAW, AO.dyn.IVX
    opp = AO.AurorusOpponents(env, share=1.0, scale=(0.6, 1.0), seed=3)
    rows = opp.rows.tolist()
    assert rows == [1, 3, 5, 7, 9, 11] and not bool(env.ideal_extra.any())
    P = env.sim.P
    mount = {b: (float(P["mount_x"][b]), float(P["mount_y"][b])) if mount_in_P else (float(np.float32(0.297)), 0.0) for b in rows}
    pose = {b: (float(env.sim.state[b, ix]), float(env.sim.state[b, iy]), float(env.sim.state[b, ih])) for b in rows}
    # a disc on the line 2.5 m ahead of the car of row 3, as its scanner sees it
    b = 3
    t3 = int(env.sim.tid[b])
    disc = off_line(RL[t3], RL[t3].nearest(pose[b][0], pose[b][1], None, 0.0)["s"] + 2.5, 0.0)[:2] + (0.15,)
    x, y, h = pose[b]
    origin = (x + mount[b][0] * math.cos(h) - mount[b][1] * math.sin(h), y + mount[b][0] * math.sin(h) + mount[b][1] * math.cos(h))
    env.last_result.scan[b] = torch.tensor(disc_scan(origin, h, [disc]))
    wb, smax = float(P["lf"][1] + P["lr"][1]), float(P["s_max"][1])
    S = {b: r_state() for b in rows}
    k = 0
    for _ in range(2):                                                 # two decisions of four control steps
        opp.plan()
        assert env.ideal_extra.tolist() == [i % 2 == 1 for i in range(B)]
        assert env.sim.ideal_vcap.tolist() == [1e3 if i % 2 else 5.0 for i in range(B)]
        for j in range(4):
            opp.command(j)
            k += 1
            for r, b in enumerate(rows):
                t = int(env.sim.tid[b])
                obstacles = r_find_obstacles(CFG, env.last_result.scan[b].numpy(), pose[b], mount[b], f_scalar(t))
                rs, rv, _ = r_control(CFG, S[b], RL[t], f_scalar(t), obstacles, pose[b], float(env.sim.state[b, iv]), k * 0.025,
                                      float(opp.scale[r]), wb, smax, 5.0)
                assert abs(env.ideal_cmd[b, 0].item() - rs) < 1e-5 and abs(env.ideal_cmd[b, 1].item() - rv) < 1e-5, (k, b)
                assert int(opp.core.last["n_obst"][r]) == len(obstacles)
            opp.stepped(torch.zeros(B, dtype=torch.bool))
        opp.after()
    assert opp.core.avoiding.tolist() == [False, True, False, False, False, False] and len(obstacles) == 0
    assert not bool(env.ideal_cmd[env.learner].any()) and bool((env.ideal_cmd[opp.rows, 1] > 1.0).all())


def test_which_races_it_drives_and_what_a_restart_does():
    env, _, _ = stand_in_env(200)
    opp = AO.AurorusOpponents(env, share=0.5, scale=(0.6, 1.0), seed=1)
    rows, on, sc = opp.rows, opp.on.clone(), opp.scale.clone()
    assert 70 <= int(on.sum()) <= 130 and 0.6 <= float(sc.min()) < 0.64 and 0.96 < float(sc.max()) <= 1.0
    assert abs(opp.share_on() - on.float().mean().item()) < 1e-6
    # a race without the other car: not driven, not counted
    env.sim.absent[rows[:50]] = True
    env.sim.absent_any = True
    opp.plan()
    live = on & ~env.sim.absent[rows]
    assert env.ideal_extra[rows].tolist() == live.tolist() and not bool(env.ideal_extra[env.learner].any())
    assert abs(opp.share_on() - on[50:].float().mean().item()) < 1e-6
    opp.command(0)
    assert bool((env.ideal_cmd[rows[live], 1] > 1.0).all()) and not bool(env.ideal_cmd[rows[~live]].any())
    # cars whose race ended during the decision: the controller's state of the old race is dropped, no command for the rest
    # of the decision, and after it the new race draws who drives and how fast
    opp.core.hint[60:70], opp.core.avoiding[60:70], opp.core.d_weight[60:70] = 5, True, 0.3
    done = torch.zeros(env.B, dtype=torch.bool)
    done[rows[60:70]] = True
    opp.stepped(done)
    assert bool((opp.core.hint[60:70] == -1).all()) and not bool(opp.core.avoiding[60:70].any()) and not bool(opp.core.d_weight[60:70].any())
    assert opp.restarted.nonzero().flatten().tolist() == list(range(60, 70))
    opp.command(1)
    assert not bool(env.ideal_cmd[rows[60:70]].any()) and bool((env.ideal_cmd[rows[live][-5:], 1] > 1.0).all())
    opp.after()
    other = torch.ones(200, dtype=torch.bool)
    other[60:70] = False
    assert torch.equal(opp.on[other], on[other]) and torch.equal(opp.scale[other], sc[other])
    assert bool((opp.scale[60:70] != sc[60:70]).all())
    opp.plan()
    assert not bool(opp.restarted.any())
    # a replayed race (`env.level_seed`) draws the same every time; the others draw anew
    env.level_seed[rows[100:110]] = 500 + torch.arange(10)
    env.level_seed[rows[110]] = 500
    idx = torch.arange(100, 130)
    opp.draw(idx)
    on1, sc1 = opp.on.clone(), opp.scale.clone()
    opp.draw(idx)
    assert torch.equal(opp.on[100:111], on1[100:111]) and torch.equal(opp.scale[100:111], sc1[100:111])
    assert opp.scale[110] == opp.scale[100] and len(set(opp.scale[100:110].tolist())) == 10
    assert bool((opp.scale[111:130] != sc1[111:130]).all())
    # a car put somewhere else between decisions: its place on the line is searched anew
    opp.core.hint[5:7] = 7
    env.sim.state[rows[5], :2] += 3.0
    env.sim.state[rows[6], :2] += 0.5
    opp.plan()
    assert opp.core.hint[5:7].tolist() == [-1, 7]


# ---------------------------------------------------------------- beams of one sweep taken while the car moves
def test_beams_taken_during_the_sweep_are_placed_from_where_the_car_was():
    """This simulator's scanner turns while the car moves: beam i was taken `sweep[i]` of a cycle before the scan's time,
    from the pose the car had then (`Lidar.time_frac`). A wall at x = 3 seen at 7 m/s while turning at 3 rad/s: placed
    from the cycle's pose alone its points come out more than 0.1 m off it; placed with `sweep` they are on it."""
    n = 1081
    sweep = (FOV / (2 * math.pi)) * (1.0 - torch.arange(n, dtype=F64) / (n - 1))
    free = lambda xy, tid: torch.ones(xy.shape[:-1], dtype=torch.bool)
    core = make_core([loop_line()], free, n=1, sweep=sweep, mount=(0.126, 0.0))
    plain = make_core([loop_line()], free, n=1, mount=(0.126, 0.0))
    prev = np.array([0.0, 0.0, 0.0]); pose = np.array([7.0 * DT, 0.01, 3.0 * DT])
    f = sweep.numpy()
    yaw = pose[2] - f * (pose[2] - prev[2])
    x = pose[0] - f * (pose[0] - prev[0]) + 0.126 * np.cos(yaw)
    ang = yaw + (-FOV / 2 + np.arange(n) * FOV / (n - 1))
    r = (3.0 - x) / np.maximum(np.cos(ang), 1e-9)
    scan = torch.tensor(np.where(r < 4.3, r, np.inf), dtype=F64)[None]
    tid = torch.zeros(1, dtype=torch.long)
    mx, my = core._mount(torch.arange(1))
    now, before = torch.tensor(pose)[None], torch.tensor(prev)[None]
    P, ok = core.obstacle_points(now, scan, tid, mx, my, before)
    assert int(ok.sum()) > 20
    assert float((P[ok][:, 0] - 3.0).abs().max()) < 1e-5             # the port's beam angles are f32, as the original's
    P0, ok0 = plain.obstacle_points(now, scan, tid, mx, my, before)
    assert torch.equal(ok0, ok) and float((P0[ok0][:, 0] - 3.0).abs().max()) > 0.1
    # no earlier pose (the first cycle after a restart): every beam from the cycle's pose, as the original
    P1, _ = core.obstacle_points(now, scan, tid, mx, my, now)
    assert torch.allclose(P1[ok], P0[ok], atol=1e-12)
    core.has_prev[:] = True
    core.reset(torch.tensor([True]))
    assert not bool(core.has_prev.any())


# ---------------------------------------------------------------- returns from the floor
def r_floor_range(a, roll, pitch, mount):
    """a beam at angle a of a scan plane on a body rolled (+ = right side down) then pitched (+ = nose down), the scanner
    at `mount` (ahead, left, up) in the body: the range at which it meets the floor (inf if it does not point down)"""
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    beam = Ry @ Rx @ np.array([math.cos(a), math.sin(a), 0.0])
    height = (Ry @ Rx @ np.array(mount))[2]
    return height / -beam[2] if beam[2] < 0 else math.inf


def test_where_a_tilted_scan_plane_meets_the_floor():
    """This simulator's body rolls and pitches and its scanner sees the floor; the original's does neither. A body rolled
    0.03 rad right side down, the scanner 0.11 m up: the beam straight to the right meets the floor 3.67 m away, the one
    to the left never."""
    free = lambda xy, tid: torch.ones(xy.shape[:-1], dtype=torch.bool)
    mount = (0.126, 0.01)
    tilts = [(0.03, 0.0), (0.0, 0.02), (-0.025, -0.01), (0.0, 0.0)]
    core = make_core([loop_line()], free, n=4, mount=mount, floor=AO.FloorGate(height=0.11, tol=0.0))
    idx = torch.arange(4)
    mx, my = core._mount(idx)
    tilt = torch.tensor(tilts, dtype=F64)
    fr = core.floor_range(tilt, idx, mx, my)
    for c, (roll, pitch) in enumerate(tilts):
        for m in range(0, int(core.beams.numel()), 7):
            want = r_floor_range(float(core.beam_angle[m]), roll, pitch, mount + (0.11,))
            got = float(fr[c, m])
            assert (math.isinf(want) and math.isinf(got)) or abs(got - want) < 1e-9 * max(1.0, want), (c, m, got, want)
    right = int((core.beam_angle + PI / 2).abs().argmin()); left = int((core.beam_angle - PI / 2).abs().argmin())
    assert abs(float(fr[0, right]) - 3.67) < 0.02 and math.isinf(float(fr[0, left])) and bool(torch.isinf(fr[3]).all())
    # a return 4 m away on every beam: obstacle points, except where the floor is that near. With no tilt handed over, or
    # no gate, all of them are points (the original's behaviour)
    pose = torch.tensor([[10.0, 0.0, 0.0]], dtype=F64).expand(4, 3)
    scan = torch.full((4, 1081), 4.0, dtype=F64)
    tid = torch.zeros(4, dtype=torch.long)
    _, ok = core.obstacle_points(pose, scan, tid, mx, my, None, tilt, idx)
    assert torch.equal(ok, fr > 4.0) and core.n_floor.tolist() == (fr <= 4.0).sum(1).tolist()
    assert 0 < int(ok[0].sum()) < ok.shape[1] and bool(ok[3].all()) and not bool(ok[0, right]) and bool(ok[0, left])
    _, ok_none = core.obstacle_points(pose, scan, tid, mx, my, None)
    assert bool(ok_none.all()) and core.n_floor is None
    plain = make_core([loop_line()], free, n=4, mount=mount)
    _, ok_plain = plain.obstacle_points(pose, scan, tid, mx, my, None, tilt, idx)
    assert bool(ok_plain.all())
    # `tol`: the tilt taken that much worse -- the level car's beams then end at 0.11 / 0.03 = 3.67 m, the rolled car's
    # right-hand beam at 0.11 / 0.06 = 1.83 m
    wide = make_core([loop_line()], free, n=4, mount=(0.0, 0.0), floor=AO.FloorGate(height=0.11, tol=0.03))
    z = torch.zeros(4, dtype=F64)
    fw = wide.floor_range(tilt, idx, z, z)
    assert float((fw[3] - 0.11 / 0.03).abs().max()) < 1e-9 and abs(float(fw[0, right]) - 0.11 * math.cos(0.03) / (math.sin(0.03) + 0.03)) < 1e-9


def test_the_tilt_estimate_follows_the_gyro_and_settles_on_the_cars_gains():
    est = AO.TiltEstimate(3, 0.03, 0.03, 0.008, tau=0.3)
    v = torch.tensor([5.0, 5.0, 0.0], dtype=F64); r = torch.tensor([1.8, -1.8, 0.0], dtype=F64)
    for _ in range(200):                                               # 5 s of a steady corner, no gyro samples
        t = est.update(v, r, None, 0.02, 0.025)
    roll = 0.03 * 5.0 * 1.8 / 9.81                                     # a left turn: right side down
    assert torch.allclose(t, torch.tensor([[roll, 0.0], [-roll, 0.0], [0.0, 0.0]], dtype=F64), atol=1e-6)
    # the gyro's rates are integrated at once (two samples 0.02 s apart), then pulled back with tau
    rates = torch.zeros(3, 2, 2, dtype=F64)
    rates[2, :, 0], rates[2, :, 1] = 0.5, -0.25
    t = est.update(v, r, rates, 0.02, 0.025)
    keep = 1.0 - 0.025 / 0.3
    assert torch.allclose(t[2], torch.tensor([0.02 * keep, -0.01 * keep], dtype=F64), atol=1e-12)
    assert torch.allclose(t[0], torch.tensor([roll, 0.0], dtype=F64), atol=1e-6)
    for _ in range(100):
        t = est.update(v, r, rates[:, :0], 0.02, 0.025)                # a cycle in which no sample came
    assert float(t[2].abs().max()) < 1e-4
    # braking at 5 m/s^2 dives by the small gain, the same throttle squats by the large one
    est = AO.TiltEstimate(2, 0.03, 0.03, 0.008, tau=0.3)
    v = torch.tensor([6.0, 2.0], dtype=F64); z = torch.zeros(2, dtype=F64)
    est.update(v, z, None, 0.02, 0.025)
    for _ in range(40):
        v = v + torch.tensor([-0.125, 0.125], dtype=F64)
        t = est.update(v, z, None, 0.02, 0.025)
    w = 1.0 - (1.0 - 0.025 / 0.3) ** 40
    assert torch.allclose(t[:, 1], torch.tensor([0.008 * 5 / 9.81 * w, -0.03 * 5 / 9.81 * w], dtype=F64), atol=1e-9)
    est.reset(torch.tensor([True, False]))
    assert not bool(est.tilt[0].any()) and bool(est.tilt[1].any()) and math.isnan(float(est.v_prev[0]))


def physics_env(n_races=6):
    env, RL, f_scalar = stand_in_env(n_races)
    B = env.B
    env.ext_cmd, env.ext_mask, env._ext_ids = torch.zeros(B, 2), torch.zeros(B, dtype=torch.bool), set()
    env.last_result.scan_type = torch.zeros(B, 1081, dtype=torch.int32)
    env.last_result.imu = torch.zeros(B, 1, 6)
    env.cfg.vehicle = types.SimpleNamespace(roll_per_g=0.03, pitch_per_g=0.03, dive_per_g=0.008)
    env.cfg.imu = types.SimpleNamespace(imu_rate=50.0)
    env.cfg.lidar.mount_z = 0.11
    return env


def test_commands_through_the_simulators_car():
    """`drive="physics"`: the same steering and speed, as the command of an externally driven row"""
    env, ref = physics_env(), stand_in_env()[0]
    B = env.B
    env.ideal_cmd = None                                               # nothing of the exact execution is touched
    opp = AO.AurorusOpponents(env, share=1.0, seed=3, drive="physics")
    twin = AO.AurorusOpponents(ref, share=1.0, seed=3)
    rows = opp.rows
    assert env.ideal_extra is None and not env._ext_ids
    env.sim.absent[rows[0]] = True                                     # a race without the other car
    env.sim.absent_any = True
    env._ext_ids.add(0)                                                # a row someone else drives stays in the env's book
    opp.plan(); twin.plan()
    assert env.ext_mask.tolist() == [i % 2 == 1 and i != 1 for i in range(B)] and env._ext_ids == {0, 3, 5, 7, 9, 11}
    opp.command(0); twin.command(0)
    assert torch.equal(env.ext_cmd[rows[1:]], ref.ideal_cmd[rows[1:]]) and bool((env.ext_cmd[rows[1:], 1] > 1.0).all())
    assert not bool(env.ext_cmd[env.learner].any()) and not bool(env.ext_cmd[rows[0]].any())
    env.sim.absent[rows[0]] = False
    opp.plan()
    assert bool(env.ext_mask[rows].all()) and env._ext_ids == {0, 1, 3, 5, 7, 9, 11}


def test_floor_returns_dropped_by_the_simulators_label_and_by_the_cars_own_estimate():
    scan = torch.full((12, 1081), 1.5)                                 # something 1.5 m away on every beam
    everywhere = lambda xy, t: torch.ones(xy.shape[:-1])               # all of it on the map's free space
    n_obst = {}
    for floor in ("off", "oracle", "gate"):
        env = physics_env()
        env.sim.track.sample_edt = everywhere
        env.last_result.scan = scan.clone()
        env.last_result.scan_type[:, :540] = AO.HIT_GROUND             # the right-hand half: the simulator says floor
        env.sim.state[:, AO.dyn.IVX], env.sim.state[:, AO.dyn.IR] = 5.0, 3.0   # a left turn at 1.5 g: 0.046 rad right side down
        env.sim.state[3, AO.dyn.IR] = 0.0                              # the car of row 3 drives straight
        opp = AO.AurorusOpponents(env, share=1.0, seed=3, floor=floor, floor_tol=0.03)
        opp.plan()
        for j in range(80):                                            # 2 s: the estimate has settled
            opp.command(j)
        n_obst[floor] = opp.core.last["n_obst"].clone()
        if floor == "gate":
            n_floor = opp.core.last["n_floor"]
            roll = 0.03 * 5.0 * 3.0 / 9.81
            assert torch.allclose(opp.tilt.tilt[0], torch.tensor([roll, 0.0], dtype=F64), atol=1e-4)
            assert not bool(opp.tilt.tilt[1].any())
            # 0.11 / (0.046 + 0.03) = 1.45 m: the beams to the right are dropped; the straight car's floor is 3.67 m away
            assert int(n_floor[1]) == 0 and bool((n_floor[[0, 2, 3, 4, 5]] > 0).all())
            assert torch.equal(n_obst["gate"] + n_floor, n_obst["off"])
            # a restart forgets the estimate
            done = torch.zeros(12, dtype=torch.bool)
            done[opp.rows[0]] = True
            opp.stepped(done)
            assert not bool(opp.tilt.tilt[0].any()) and bool(opp.tilt.tilt[2].any())
        else:
            assert opp.core.last["n_floor"] is None and opp.tilt is None
    # the label drops exactly the read beams of the right-hand half
    env = physics_env()
    env.sim.track.sample_edt = everywhere
    env.last_result.scan = scan.clone()
    env.last_result.scan[:, :540] = float("inf")
    env.sim.state[:, AO.dyn.IVX] = 5.0
    opp = AO.AurorusOpponents(env, share=1.0, seed=3)
    opp.plan(); opp.command(0)
    assert torch.equal(opp.core.last["n_obst"], n_obst["oracle"]) and bool((n_obst["oracle"] < n_obst["off"]).all())


# ---------------------------------------------------------------- the driver of an env's external rows (the console)
def real_env(slot=None, envs=3):
    """the simulator on the Korea 2025 map: `slot` as the other car of each race, or (None) cars alone"""
    from f1sim import Config, maps
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    from f1sim.raceline import Raceline
    track = maps.load("real:korea_2025_iccas")
    rl = Raceline.build_cached(track)
    cfg = Config(); cfg.sim.compile_mode = "none"
    if slot is None:
        ecfg = EnvConfig(max_steps=4000, hist_len=0, race_size=1, action_mode="direct")
        return common.make_env([track], envs, "cpu", ecfg, cfg=cfg, seed=3), rl
    ecfg = EnvConfig(max_steps=4000, hist_len=0, race_size=2, opponent="slots", opponent_slots=[slot], action_mode="direct")
    return common.make_env([track], envs * 2, "cpu", ecfg, cfg=cfg, seed=3, rls=[rl]), rl


def test_an_aurorus_slot_executes_what_its_controller_says_on_its_own_scan():
    """as `test_external_opponents` asks of every external driver: the car's command is the driver's, read from that car's
    own row and sent through its calibration -- here against a second controller fed by hand"""
    env, rl = real_env({"kind": "aurorus", "speed_scale": [0.7, 1.0], "options": {"floor": "off"}})
    env.reset()
    rows = env.external_slot_rows[1]
    drv = env.external_drivers[1]
    assert isinstance(drv, AO.AurorusDriver) and drv.scale == AO.SOLO_SCALE and rows.numel() == 3
    assert not bool(env.learner[rows].any())
    lo, hi = env.slot_scale[rows].min().item(), env.slot_scale[rows].max().item()
    assert 0.7 <= lo < hi <= 1.0                                       # the slot's draw for each race
    ref, _, _ = AO.build_core(env, rows, AO.driver_line(env, [rl]), floor="off")
    act = torch.zeros(env.B, env.act_dim)
    for _ in range(6):
        st = env.sim.state[rows]
        pose = torch.stack([st[:, AO.dyn.IX], st[:, AO.dyn.IY], st[:, AO.dyn.IYAW]], -1)
        steer, v = ref.cycle(pose, st[:, AO.dyn.IVX], env.last_result.scan[rows], env.sim.tid[rows], env.sim.P["a_brake"][rows],
                             AO.SOLO_SCALE * env.slot_scale[rows])
        env.step(act)
        got = env.last_cmd[rows]
        cal = (env.tracker_cal if getattr(env, "tracker_cal", None) is not None else env._ext_cal)[rows]
        want_s = ((steer.float() - cal[:, 0]) / cal[:, 1]).clamp(-float(env.s_max), float(env.s_max))
        want_v = torch.minimum(v.float() / cal[:, 2], env.speed_cap[rows])
        assert torch.allclose(got[:, 0], want_s, atol=1e-6) and torch.allclose(got[:, 1], want_v, atol=1e-5)
        assert bool((v > 0.5).all())                                   # it does ask the cars to go


def test_the_driver_forgets_a_car_that_starts_again():
    env, rl = real_env({"kind": "aurorus", "options": {"floor": "gate"}})
    env.reset()
    rows = env.external_slot_rows[1]
    drv = env.external_drivers[1]
    drv.reset(np.array([True, False, True]))                           # before its first command: nothing to forget
    act = torch.zeros(env.B, env.act_dim)
    for _ in range(4):
        env.step(act)
    core = drv.core
    assert bool(core.has_prev.all()) and drv.tilt is not None
    drv.tilt.tilt[:] = 0.01
    drv.reset(np.array([True, False, False]))                          # the env's mask over this slot's rows
    assert core.has_prev.tolist() == [False, True, True] and int(core.hint[0]) == -1
    assert not bool(drv.tilt.tilt[0].any()) and bool(drv.tilt.tilt[1].any())
    # a car put somewhere else without a reset (the console's drag): its place on the line is looked up afresh
    env.step(act)
    core.avoiding[1] = True
    env.sim.state[rows[1], :2] = env.sim.state[rows[2], :2] + 3.0
    drv.command_env(env, rows)
    assert not bool(core.avoiding[1])
    drv.reset()
    assert not bool(core.has_prev.any())


def test_the_ego_driver_and_its_picture():
    """`common.build_ego_driver(env, "aurorus")` on the console's 내 차, and what `viewer.driver_viz` draws for it"""
    from f1sim.learn import common
    from f1sim.viewer import driver_viz as DV
    env, rl = real_env(None, envs=2)
    env.reset()
    with pytest.raises(ValueError):
        common.build_ego_driver(env, "aurorus")                        # no line to follow
    drv = common.build_ego_driver(env, "aurorus", options={"floor": "off", "scale": 0.5}, rls=[rl])
    env.set_ego_driver(drv)
    assert drv.rows.tolist() == [0, 1] and DV.kind_title(drv)[0] == "aurorus"
    viz = DV.DriverViz(env)
    assert viz.driver(0) is drv

    def picture(row):
        flat = torch.cat([t.flatten().float() for t in viz.tensors(row)]).numpy()
        return viz.decode(row, flat, pose=env.sim.state[row, :3].tolist(), mount=(0.0, 0.0, 0.0), scan_m=None, cmd=(0.0, 0.0))

    path, fr = picture(0)                                              # before its first command: a picture all the same
    assert path.shape[1] == 3 and "oracle" in fr["legend"] and not fr["points"]
    act = torch.zeros(env.B, env.act_dim)
    for _ in range(40):
        env.step(act)
    assert bool((env.sim.state[:, AO.dyn.IVX] > 0.5).all())            # the cars drive
    path, fr = picture(1)
    core = drv.core
    s0 = core.last["s0"][1]
    pts, v = core.line.at(env.sim.tid[1].view(1), (s0 + torch.linspace(0.0, AO.VIZ_AHEAD_M, AO.VIZ_AHEAD_POINTS, dtype=core.dtype))[None])
    assert path.shape == (AO.VIZ_AHEAD_POINTS, 3)                      # its own line ahead, at the speed it asks for there
    assert np.allclose(path[:, :2], pts[0].numpy(), atol=1e-4) and np.allclose(path[:, 2], 0.5 * v[0].numpy(), atol=1e-4)
    assert "0.50배" in fr["legend"] and "장애물로 본 점 0개" in fr["legend"]
    assert len(fr["points"]) == 1 and np.allclose(fr["points"][0]["xy"][0], core.last["target"][1].numpy(), atol=1e-4)
    # something on its line 2.5 m ahead: after the switch-on time it avoids, and the picture is the chosen path
    mid, _ = core.line.at(env.sim.tid[1].view(1), (s0 + 2.5).view(1, 1))
    cx, cy = mid[0, 0].tolist()
    lid = env.cfg.lidar
    st = env.sim.state[1]
    x, y, h = float(st[AO.dyn.IX]), float(st[AO.dyn.IY]), float(st[AO.dyn.IYAW])
    mx = float(core.mount[0][1]) if hasattr(core, "mount") else float(lid.mount_x)
    scan = disc_scan((x + mx * math.cos(h), y + mx * math.sin(h)), h, [(cx, cy, 0.15)], n=int(env.last_result.scan.shape[1]),
                     fov=float(lid.fov))
    env.last_result.scan[1] = torch.tensor(scan)
    for _ in range(core.n_on + 1):
        drv.command_env(env, drv.rows)
    assert bool(core.avoiding[1]) and not bool(core.avoiding[0])
    path, fr = picture(1)
    d = drv.viz_decode(torch.cat([t.flatten().float() for t in viz.tensors(1)]).numpy())
    assert d["avoiding"] and len(d["path"]) >= 2 and len(d["obstacles"]) > 0
    assert np.allclose(path[:, :2], d["path"], atol=1e-5) and np.allclose(path[:, 2], d["speed"])
    assert float(np.linalg.norm(d["obstacles"] - np.array([cx, cy]), axis=1).max()) < 0.2     # the points it saw are the disc's
    assert float(np.linalg.norm(d["path"] - np.array([cx, cy]), axis=1).min()) > 0.15         # and the path goes round it
    assert "비켜 가려고 고른 경로" in fr["legend"] and f"{len(d['obstacles'])}개" in fr["legend"]
    assert len(fr["lines"]) == 2 and len(fr["points"]) == 2
    path0, fr0 = picture(0)                                            # the other car is still on its line
    assert path0.shape == (AO.VIZ_AHEAD_POINTS, 3) and "자기 선" in fr0["legend"]
