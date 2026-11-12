"""AuRoRus's obstacle avoidance as an opponent car, batched over rows (11-12).

Ported from the code, not from a description of it: github.com/Noce99/AuRoRus as cloned 11-12 --
`src/autonomous_control/frenet_overtaking.rs` (obstacle points, the on/off switch, one control cycle),
`src/autonomous_control/shared/frenet.rs` (the sideways candidate paths and how one is chosen),
`shared/steering.rs`, `shared/reactive.rs` (`fov_window`), `src/geometry/line.rs` and
`config/autonomous_control/frenet_overtaking.toml` (every default in `AurorusConfig`). The four opponent kinds the
training world had were one line follower with no perception (`plan_rl.SelfOpponents`' notes); this one drives a line
and, when its own LiDAR shows something on the track that the map does not have, goes around it.

One control cycle (40 Hz), per car:
  1. nearest point of its line (searched forward from the last one); further than 1.5 m from the line -> it stops
     and forgets its state.
  2. obstacle points = every 5th beam of the front 200 deg, shorter than 4.35 m, whose end lies on the map's free
     space (more than 0.2 m from a mapped wall). Walls are not obstacles; a box and another car are the same thing.
  3. a switch: points in 4 cycles running -> avoid; no points for 84 cycles running (2.07 s) -> back to the line.
  4. on the line: aim at the line 0.2 s x speed + 0.45 m ahead, at the line's speed there.
     avoiding: 63 paths (21 sideways offsets -1.0..+1.0 m at the end x 3 lengths 1.25 / 2.75 / 4.25 m along the line,
     cubic in the offset, starting from the car's own offset and heading); the cheapest one that touches no obstacle
     point (0.35 m) and stays on free space; failing that the cheapest that touches no point; failing that the one that
     gets furthest. Speed is cut where the path bends more than the line (to 0.7 at most) and, when every longest path
     is blocked, to what stops 0.3 m short of the furthest reach.
  5. steering = 0.8 x the bearing of the aim point seen from one wheelbase behind the centre of gravity.
It never reverses, knows nothing of a car beside or behind it (200 deg), and treats a moving car as a still one.

What is not the original's, plainly:
  * the line and its speeds are this simulator's minimum-time lines (`RacelineTeacher`, 7 / 6.5 / 4), each car's
    speed a per-race multiple of that profile in place of the original's fixed 0.6 x its authors' profile;
  * the car's limits are this simulator's (steering +-0.419 rad at 3.2 rad/s, 5 m/s^2 both ways, no 4 m/s top speed;
    the stopping-distance cap uses the car's own braking), wheelbase 0.3302 m against 0.32;
  * the scanner sits ahead of the centre of gravity (`mount_x`) where the original's template car has it on it, and
    the scan is this simulator's (noise, beams of one sweep taken while moving). The beam angles used are the
    nominal ones: a randomized scanner yaw (`mount_yaw`, +-0.015 rad) is not known to the car;
  * free space = `TrackTensors.sample_edt` > 0.2 m (nearest-cell look-up of the distance to the nearest mapped wall);
  * its own pose is the simulator's (the original's `pose_source = 1`, its simulation-only ground-truth mode; on the
    car it localises). Other cars and boxes come from its LiDAR only;
  * the unused second controller (`controller = 1`) is not ported.
The geometry is float64 as the original's: in float32 the curvature of a candidate path (points 0.2 m apart at
coordinates of tens of metres) is noise of 1e-5..1e-4 1/m, and the curvature ratio of step 4 turns that into a speed
cut on a straight line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from .. import dynamics as dyn


@dataclass
class AurorusConfig:
    """`config/autonomous_control/frenet_overtaking.toml` as read 11-12 (controller 0)"""
    kk_s: float = 0.8                    # steering per rad of bearing error
    kd_s: float = 0.0                    # its derivative term (clamped to +-0.2)
    look_ahead_gain_s: float = 0.2
    min_look_ahead_m: float = 0.45
    max_cross_track_m: float = 1.5
    lidar_downsample: int = 5
    desired_fov_deg: float = 200.0
    wall_clearance_m: float = 0.2
    switch_on_s: float = 0.07
    hysteresis_s: float = 2.07
    max_road_width_m: float = 1.0
    delta_road_width_m: float = 0.1
    path_point_distance_m: float = 0.2
    min_path_length_m: float = 1.25
    max_path_length_m: float = 4.35
    delta_path_length_m: float = 1.5
    path_fov_deg: float = 30.0
    robot_radius_m: float = 0.35
    k_jerk: float = 0.01
    k_length: float = 0.1
    k_distance: float = 0.01
    speed_decay_factor: float = 0.999
    min_speed_reduction_gain: float = 0.7
    decay_last_d_factor: float = 0.99
    weight_last_d: float = 0.4
    speed_curvature_exponent: float = 0.3
    braking_margin_m: float = 0.3
    max_speed: float = 1e3               # the original clamps to its simulator's 4 m/s; none here


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def samples(start: float, end: float, step: float) -> list:
    """`frenet.rs::samples`: start, start + step, ... not past end"""
    count = int(math.floor((end - start) / step + 1e-6)) if step > 0 and end > start else 0
    return [start + k * step for k in range(count + 1)]


def fov_window(n: int, fov: float, want: float) -> Tuple[int, int]:
    """`reactive.rs::fov_window`: the beams inside the wanted field of view, in the scan's own float32 arithmetic"""
    f = np.float32
    if n <= 1 or f(want) >= f(fov):
        return 0, n
    step = f(fov) / f(n - 1)
    outside = int(np.ceil((f(fov) - max(f(want), f(0.0))) / f(2.0) / step))
    outside = min(outside, n // 2)
    return outside, n - outside


def stopping_speed(distance: torch.Tensor, margin: float, max_decel: torch.Tensor) -> torch.Tensor:
    """the speed from which the car stops `margin` short of `distance`"""
    return torch.sqrt(2.0 * max_decel.clamp_min(0.0) * (distance - margin).clamp_min(0.0))


class Line:
    """`geometry/line.rs` for every track at once: closed polylines with a speed at each vertex.
    P (T, N, 2), V (T, N). Segment i runs P[i] -> P[i+1 mod N]; arc length s starts at P[0]."""

    def __init__(self, P: torch.Tensor, V: torch.Tensor, dtype=torch.float64):
        dev = P.device
        self.dtype = dtype
        P64 = P.detach().double().cpu()
        self.N = int(P.shape[1])
        nxt = torch.roll(P64, -1, 1)
        seg = nxt - P64
        ln = seg.norm(dim=-1)
        cum = torch.cat([torch.zeros_like(ln[:, :1]), ln.cumsum(1)], 1)
        head = torch.atan2(seg[..., 1], seg[..., 0])
        before = torch.roll(head, 1, 1)
        vert = before + wrap(head - before) / 2                        # heading at vertex i: half-way between its two segments
        a = torch.roll(P64, 1, 1)
        ab, bc, ac = P64 - a, nxt - P64, nxt - a
        cross = ab[..., 0] * bc[..., 1] - ab[..., 1] * bc[..., 0]
        sides = ab.norm(dim=-1) * bc.norm(dim=-1) * ac.norm(dim=-1)
        curv = torch.where(sides > 0, 2.0 * cross / sides.clamp_min(1e-300), torch.zeros_like(sides))
        f = lambda A: A.to(dtype).to(dev)
        self.P, self.V = f(P64), V.detach().to(dtype).to(dev)
        self.seg, self.len, self.cum, self.lap = f(seg), f(ln), f(cum), f(cum[:, -1])
        self.seg_head, self.vert_head, self.curv = f(head), f(vert), f(curv)   # curv[i]: circle through P[i-1], P[i], P[i+1]

    def _segment(self, tid: torch.Tensor, s: torch.Tensor):
        """`segment_at`, for s (R, K) already inside the lap -> the segment and how far along it (0..1)"""
        cum = self.cum[tid]
        i = (torch.searchsorted(cum, s.contiguous(), right=True) - 1).clamp(0, self.N - 1)
        ln = self.len[tid].gather(1, i)
        t = torch.where(ln > 0, (s - cum.gather(1, i)) / ln.clamp_min(1e-12), torch.zeros_like(s))
        return i, t

    def at(self, tid: torch.Tensor, s: torch.Tensor):
        """position and speed at arc length s (R, K) -> (R, K, 2), (R, K)"""
        s = torch.remainder(s, self.lap[tid][:, None])
        i, t = self._segment(tid, s)
        j = (i + 1) % self.N
        P, V = self.P[tid], self.V[tid]
        a = P.gather(1, i[..., None].expand(-1, -1, 2)); b = P.gather(1, j[..., None].expand(-1, -1, 2))
        va, vb = V.gather(1, i), V.gather(1, j)
        return a + t[..., None] * (b - a), va + t * (vb - va)

    def heading_at(self, tid: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """heading at s (R, K): turned evenly along a segment from one vertex heading to the next"""
        s = torch.remainder(s, self.lap[tid][:, None])
        i, t = self._segment(tid, s)
        vh = self.vert_head[tid]
        a, b = vh.gather(1, i), vh.gather(1, (i + 1) % self.N)
        return wrap(a + t * wrap(b - a))

    def curvature_at(self, tid: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s = torch.remainder(s, self.lap[tid][:, None])
        i, _ = self._segment(tid, s)
        return self.curv[tid].gather(1, i)

    def nearest(self, tid: torch.Tensor, xy: torch.Tensor, hint: torch.Tensor, window: torch.Tensor):
        """closest point of the line to xy (R, 2). hint (R,): the segment found last time, -1 = none. With a hint only
        the segments from two before it on, for `window` (R,) metres of line, are looked at (a line that comes back
        close to itself does not take the car over); of equal distances the first in that order wins.
        -> segment (R,), s (R,), point (R, 2), distance (R,)"""
        N, R = self.N, xy.shape[0]
        a, d = self.P[tid], self.seg[tid]                              # (R, N, 2)
        len2 = (d * d).sum(-1)
        t = torch.where(len2 > 0, ((xy[:, None, :] - a) * d).sum(-1) / len2.clamp_min(1e-12), torch.zeros_like(len2))
        t = t.clamp(0.0, 1.0)
        proj = a + t[..., None] * d
        dist = (xy[:, None, :] - proj).norm(dim=-1)
        has = hint >= 0
        first = torch.where(has, (hint % N + N - 2) % N, torch.zeros_like(hint))
        order = (first[:, None] + torch.arange(N, device=xy.device)[None]) % N
        cum = self.cum[tid]
        c0 = cum.gather(1, first[:, None])
        cj = cum.gather(1, order)
        covered = torch.where(order >= first[:, None], cj - c0, self.lap[tid][:, None] - c0 + cj)   # line before each segment
        ok = ~has[:, None] | (covered <= window[:, None])
        k = torch.where(ok, dist.gather(1, order), torch.full_like(dist, float("inf"))).argmin(1)
        seg = order.gather(1, k[:, None])
        s = cum.gather(1, seg) + t.gather(1, seg) * self.len[tid].gather(1, seg)
        seg = seg.squeeze(1)
        rows = torch.arange(R, device=xy.device)
        return seg, s.squeeze(1), proj[rows, seg], dist[rows, seg]


class AurorusCore:
    """The controller's state and one control cycle for `n` cars.

    line: `Line`; free(xy (..., 2), tid (R,)) -> bool (...): the map's free space (see the module notes);
    n_beams / fov / range_min / range_max: the scanner, beam i at -fov/2 + i fov/(n_beams-1);
    mount: the scanner's place ahead / left of the centre of gravity [m] (floats or (n,) tensors);
    wheelbase, max_steer: the car's; dt: the cycle [s] (the switch counts cycles)."""

    def __init__(self, line: Line, free: Callable, n: int, n_beams: int, fov: float, wheelbase: float, max_steer: float,
                 mount=(0.0, 0.0), dt: float = 0.025, range_min: float = 0.02, range_max: float = 10.0,
                 cfg: Optional[AurorusConfig] = None, device=None, chunk: int = 128):
        self.cfg = cfg = cfg or AurorusConfig()
        self.line, self.free, self.n = line, free, int(n)
        self.dev = dev = torch.device(device) if device is not None else line.P.device
        self.dtype = ft = line.dtype
        self.wheelbase, self.max_steer, self.dt = float(wheelbase), float(max_steer), float(dt)
        self.range_min, self.range_max = float(range_min), float(range_max)
        self.mount = mount
        self.chunk = int(chunk)
        lo, hi = fov_window(n_beams, fov, math.radians(cfg.desired_fov_deg))
        self.beams = torch.arange(lo, hi, cfg.lidar_downsample, device=dev)
        f = np.float32                                                 # `LidarScan::ray_angle_rad` works in f32
        ang = -f(fov) / f(2.0) + self.beams.cpu().numpy().astype(f) * f(fov) / f(n_beams - 1)
        self.beam_angle = torch.as_tensor(ang.astype(np.float64), dtype=ft, device=dev)
        # elapsed cycles after which `now - since >= wait` first holds
        self.n_on = int(math.ceil(cfg.switch_on_s / dt - 1e-9))
        self.n_off = int(math.ceil(cfg.hysteresis_s / dt - 1e-9))
        # the candidates, in the original's order: end offsets outside, lengths inside
        d1 = samples(-cfg.max_road_width_m, cfg.max_road_width_m, cfg.delta_road_width_m)
        Ls = samples(cfg.min_path_length_m, cfg.max_path_length_m, cfg.delta_path_length_m)
        self.cd1 = torch.tensor([d for d in d1 for _ in Ls], dtype=ft, device=dev)                     # (C,)
        self.cL = torch.tensor([L for _ in d1 for L in Ls], dtype=ft, device=dev)
        steps = {min(cfg.path_point_distance_m, L / 2) for L in Ls}
        assert len(steps) == 1, "paths shorter than two point distances are sampled at their own step; not ported"
        self.step = steps.pop()
        nv = [len(samples(0.0, L, self.step)) for L in Ls]
        self.cn = torch.tensor([n_ for _ in d1 for n_ in nv], dtype=torch.long, device=dev)            # points per path
        self.K = max(nv)
        self.u = self.step * torch.arange(self.K, dtype=ft, device=dev)                                # (K,) along the line
        kk = torch.arange(self.K, device=dev)
        self.valid = kk[None, :] < self.cn[:, None]                                                    # (C, K)
        self.check = self.valid & (kk[None, :] >= 1)                   # a path's first point (the car's place) is not tested
        self.k = 0                                                     # cycles so far
        self.hint = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.avoiding = torch.zeros(n, dtype=torch.bool, device=dev)
        self.since = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.d_weight = torch.zeros(n, dtype=ft, device=dev)
        self.blocked_gain = torch.ones(n, dtype=ft, device=dev)
        self.prev_err = torch.full((n,), float("nan"), dtype=ft, device=dev)
        self.lost = torch.zeros(n, dtype=torch.bool, device=dev)       # last cycle: further than max_cross_track_m from the line
        self.last = {}                                                 # what the last cycle saw and chose, for drawing and tests
        self.last_plan = {}

    def reset(self, mask: torch.Tensor) -> None:
        """cars (bool (n,) or indices) that start anew: no nearest-point hint, on the line, the planner's memory blank"""
        self.hint[mask] = -1
        self.avoiding[mask] = False
        self.since[mask] = -1
        self.d_weight[mask] = 0.0
        self.blocked_gain[mask] = 1.0
        self.prev_err[mask] = float("nan")
        self.lost[mask] = False

    def _mount(self, idx: torch.Tensor):
        mx, my = self.mount
        mx = mx[idx].to(self.dtype) if torch.is_tensor(mx) else torch.full((idx.numel(),), float(mx), dtype=self.dtype, device=self.dev)
        my = my[idx].to(self.dtype) if torch.is_tensor(my) else torch.full((idx.numel(),), float(my), dtype=self.dtype, device=self.dev)
        return mx, my

    def obstacle_points(self, pose: torch.Tensor, scan: torch.Tensor, tid: torch.Tensor, mx: torch.Tensor, my: torch.Tensor):
        """`find_obstacles`: -> points (R, M, 2) in the world and which of them count (R, M)"""
        cfg = self.cfg
        r = scan[:, self.beams]
        ok = torch.isfinite(r) & (r > self.range_min) & (r < self.range_max) & (r < cfg.max_path_length_m)
        r = torch.where(ok, r, torch.zeros_like(r))
        c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        ox = pose[:, 0] + mx * c - my * s
        oy = pose[:, 1] + mx * s + my * c
        ang = pose[:, 2:3] + self.beam_angle[None]
        pts = torch.stack([ox[:, None] + r * torch.cos(ang), oy[:, None] + r * torch.sin(ang)], -1)
        ok = ok & self.free(pts, tid)
        return pts, ok

    @torch.no_grad()
    def cycle(self, pose: torch.Tensor, speed: torch.Tensor, scan: torch.Tensor, tid: torch.Tensor, a_brake: torch.Tensor,
              scale: torch.Tensor, live: Optional[torch.Tensor] = None):
        """One control cycle. pose (n, 3) = x, y, heading of the centre of gravity; speed (n,) forward [m/s];
        scan (n, n_beams) ranges [m] (inf = no return); tid (n,); a_brake (n,) [m/s^2]; scale (n,): each car's multiple
        of the line's speeds; live (n,) bool: cars this drives now (the others get 0, 0 and keep their state).
        -> steering [rad] (n,), speed [m/s] (n,)"""
        cfg, line, dev, ft = self.cfg, self.line, self.dev, self.dtype
        self.k += 1
        steer_out = torch.zeros(self.n, dtype=ft, device=dev)
        v_out = torch.zeros(self.n, dtype=ft, device=dev)
        idx = torch.arange(self.n, device=dev) if live is None else torch.nonzero(live).flatten()
        if idx.numel() == 0:
            return steer_out, v_out
        p, sp, t = pose[idx].to(ft), speed[idx].to(ft), tid[idx]
        look = cfg.look_ahead_gain_s * sp.clamp_min(0.0) + cfg.min_look_ahead_m
        seg, s0, npt, dist = line.nearest(t, p[:, :2], self.hint[idx], 1.5 * look + 1.0)
        lost = dist > cfg.max_cross_track_m
        mx, my = self._mount(idx)
        pts, ok = self.obstacle_points(p, scan[idx].to(ft), t, mx, my)
        obst = ok.any(1)
        # the switch (`ObstacleSwitch::update`): a change of state has to hold for a time, counted here in cycles
        av, since = self.avoiding[idx], self.since[idx]
        same = obst == av
        since = torch.where(same, torch.full_like(since, -1), torch.where(since < 0, torch.full_like(since, self.k), since))
        wait = torch.where(av, torch.full_like(since, self.n_off), torch.full_like(since, self.n_on))
        flip = ~same & (self.k - since >= wait) & ~lost
        av = av ^ flip
        since = torch.where(flip, torch.full_like(since, -1), since)
        d_w, blocked = self.d_weight[idx], self.blocked_gain[idx]
        fresh = flip & av                                              # just turned on: a new planner
        d_w = torch.where(fresh, torch.zeros_like(d_w), d_w)
        blocked = torch.where(fresh, torch.ones_like(blocked), blocked)

        _, v_line = line.at(t, s0[:, None])
        ref = scale[idx].to(ft) * v_line.squeeze(1)
        target, _ = line.at(t, (s0 + look)[:, None])
        target = target.squeeze(1)
        v_cmd = ref.clone()
        plan_rows = torch.nonzero(av & ~lost).flatten()
        reach_all = torch.full_like(ref, float("inf"))
        gain_all = torch.ones_like(ref)
        for a in range(0, int(plan_rows.numel()), self.chunk):
            j = plan_rows[a:a + self.chunk]
            tg, gain, reach, d_w[j], blocked[j] = self._plan(t[j], p[j], s0[j], npt[j], look[j], pts[j], ok[j], d_w[j], blocked[j])
            target[j] = tg
            v = ref[j] * gain
            v_cmd[j] = torch.minimum(v, stopping_speed(reach, cfg.braking_margin_m, a_brake[idx][j].to(ft)))
            reach_all[j], gain_all[j] = reach, gain
        # steering (`steering.rs`): the aim point's bearing from one wheelbase behind the centre of gravity
        bx = p[:, 0] - self.wheelbase * torch.cos(p[:, 2])
        by = p[:, 1] - self.wheelbase * torch.sin(p[:, 2])
        err = wrap(torch.atan2(target[:, 1] - by, target[:, 0] - bx) - p[:, 2])
        prev = self.prev_err[idx]
        deriv = torch.where(torch.isnan(prev), torch.zeros_like(err), (err - prev) / self.dt)
        steer = (cfg.kk_s * err + (cfg.kd_s * deriv).clamp(-0.2, 0.2)).clamp(-self.max_steer, self.max_steer)
        v_cmd = v_cmd.clamp(0.0, cfg.max_speed)

        zero = torch.zeros_like(steer)
        steer_out[idx] = torch.where(lost, zero, steer)
        v_out[idx] = torch.where(lost, zero, v_cmd)
        self.hint[idx] = torch.where(lost, torch.full_like(seg, -1), seg)
        self.avoiding[idx] = av & ~lost
        self.since[idx] = torch.where(lost, torch.full_like(since, -1), since)
        self.d_weight[idx] = torch.where(lost, torch.zeros_like(d_w), d_w)
        self.blocked_gain[idx] = torch.where(lost, torch.ones_like(blocked), blocked)
        self.prev_err[idx] = torch.where(lost, torch.full_like(err, float("nan")), err)
        self.lost[idx] = lost
        self.last = dict(idx=idx, target=target, s0=s0, dist=dist, obst=obst, n_obst=ok.sum(1), reach=reach_all, gain=gain_all,
                         look=look, ref=ref)
        return steer_out, v_out

    def _plan(self, tid, pose, s0, npt, look, obs, obs_ok, d_weight, blocked):
        """`Planner::plan` for rows that avoid. -> aim point (R, 2), speed gain (R,), reach (R,; inf = no stop needed),
        the planner's two remembered values after it"""
        cfg, line = self.cfg, self.line
        R, C, K = int(tid.numel()), int(self.cd1.numel()), self.K
        dev = self.dev
        rows = torch.arange(R, device=dev)
        # the car's offset from the line and its heading relative to it (`Line::lateral`)
        h0 = line.heading_at(tid, s0[:, None]).squeeze(1)
        d0 = -torch.sin(h0) * (pose[:, 0] - npt[:, 0]) + torch.cos(h0) * (pose[:, 1] - npt[:, 1])
        slope0 = torch.tan(wrap(pose[:, 2] - h0)).clamp(-1.0, 1.0)
        d1, L = self.cd1[None], self.cL[None]                          # (1, C)
        keep = torch.atan2(d1 - d0[:, None], L).abs() <= math.radians(cfg.path_fov_deg)
        keep = keep | ~keep.any(1, keepdim=True)                       # none inside the cone: all of them
        # `LateralCubic`: offset d(u), d(0) = d0, d'(0) = slope0, d(L) = d1, d'(L) = 0
        rise = d1 - d0[:, None] - slope0[:, None] * L
        a2 = slope0[:, None] / L + 3.0 * rise / L ** 2
        a3 = -(2.0 * rise + slope0[:, None] * L) / L ** 3
        u = self.u[None, None, :]
        d = d0[:, None, None] + u * (slope0[:, None, None] + u * (a2[..., None] + u * a3[..., None]))   # (R, C, K)
        last = (self.cn - 1)[None, :, None].expand(R, C, 1)
        d_end = d.gather(2, last).squeeze(2)
        jerk = (6.0 * a3) ** 2 * self.cn[None].to(d.dtype)
        cost = cfg.k_jerk * jerk + cfg.k_length / L + cfg.k_distance * (d_end - d_weight[:, None]) ** 2
        # to the world: every path of a car shares its points along the line
        s = s0[:, None] + self.u[None]
        base, _ = line.at(tid, s)
        hd = line.heading_at(tid, s)
        P = torch.stack([base[:, None, :, 0] - d * torch.sin(hd)[:, None, :], base[:, None, :, 1] + d * torch.cos(hd)[:, None, :]], -1)
        dP = P[:, :, 1:] - P[:, :, :-1]
        ds = dP.norm(dim=-1)                                           # (R, C, K-1)
        along = torch.cat([torch.zeros_like(ds[..., :1]), ds.cumsum(-1)], -1)
        # obstacle points within the car's radius of a path point; where the first one is
        far = torch.full_like(obs, 1e6)
        ob = torch.where(obs_ok[..., None], obs, far)
        c0 = pose[:, None, :2]                                         # distances about the car: small numbers
        dmin = torch.cdist((P.reshape(R, C * K, 2) - c0), ob - c0, compute_mode="donot_use_mm_for_euclid_dist").amin(-1)
        hit = (dmin.view(R, C, K) <= cfg.robot_radius_m) & self.check[None]
        coll = hit.any(-1)
        k_hit = hit.float().argmax(-1)
        inf = torch.full_like(cost, float("inf"))
        reached = torch.where(coll, along.gather(2, k_hit[..., None]).squeeze(2), inf)
        on_map = (self.free(P, tid) | ~self.check[None]).all(-1)
        length_m = along.gather(2, last).squeeze(2)
        # the choice
        m1 = keep & ~coll & on_map
        m2 = keep & ~coll
        has1, has2 = m1.any(1), m2.any(1)
        sel1 = torch.where(m1, cost, inf).argmin(1)
        sel2 = torch.where(m2, cost, inf).argmin(1)
        r_max = torch.where(keep, reached, -inf).amax(1, keepdim=True)
        m3 = keep & (reached == r_max)
        m3 = m3 & (cost == torch.where(m3, cost, inf).amin(1, keepdim=True))
        sel3 = (C - 1) - m3.flip(1).float().argmax(1)                  # furthest, then cheapest, then the last of those
        sel = torch.where(has1, sel1, torch.where(has2, sel2, sel3))
        blocked = torch.where(has1, torch.ones_like(blocked), blocked * cfg.speed_decay_factor)
        # `reach`: None (inf) when one of the longest paths is clear, else the furthest any path gets
        u_end = self.u[self.cn - 1][None]
        longest = torch.where(keep, u_end.expand(R, C), -inf).amax(1, keepdim=True)
        clear = (keep & ~coll & (u_end >= longest - 1e-9)).any(1)
        reach = torch.where(keep, torch.where(coll, reached, length_m), torch.zeros_like(cost)).amax(1)
        reach = torch.where(clear, torch.full_like(reach, float("inf")), reach)
        d_weight = cfg.decay_last_d_factor * ((1.0 - cfg.weight_last_d) * d_weight + cfg.weight_last_d * d_end[rows, sel])
        # the aim point: the chosen path's first point at least `look` along it, else its end
        Ps, al, dss, dPs, n = P[rows, sel], along[rows, sel], ds[rows, sel], dP[rows, sel], self.cn[sel]
        kk = torch.arange(K, device=dev)[None]
        cand = (al >= look[:, None]) & (kk >= 1) & (kk < n[:, None])
        tgt = torch.where(cand.any(1), cand.float().argmax(1), n - 1)
        target = Ps[rows, tgt]
        # the path's curvature there against the line's (`to_world`: heading change over the mean of the two steps)
        yaw = torch.atan2(dPs[..., 1], dPs[..., 0])                    # (R, K-1)
        ti = torch.minimum(tgt, n - 2).clamp_min(1)
        g = lambda A, i: A.gather(1, i[:, None]).squeeze(1)
        mean = (g(dss, ti - 1) + g(dss, ti)) / 2
        pc = torch.where(mean > 0, wrap(g(yaw, ti) - g(yaw, ti - 1)) / mean.clamp_min(1e-12), torch.zeros_like(mean))
        eps = float(np.finfo(np.float32).eps)
        lc = line.curvature_at(tid, (s0 + self.u[tgt])[:, None]).squeeze(1)
        factor = (lc.abs().clamp_min(eps) / pc.abs().clamp_min(eps)).pow(cfg.speed_curvature_exponent).clamp_max(1.0)
        gain = (blocked * factor).clamp(min(cfg.min_speed_reduction_gain, 1.0), 1.0)
        self.last_plan = dict(P=P, d=d, keep=keep, cost=cost, hit=coll, reached=reached, on_map=on_map, chosen=sel, free=has1,
                              aim=tgt, points=self.cn)
        return target, gain, reach, d_weight, blocked


class AurorusOpponents:
    """The other cars of a race driven by `AurorusCore`, behind the interface `plan_rl.Driver` calls around a decision's
    control steps (`plan` / `command` / `stepped` / `after`, as `plan_rl.SelfOpponents`).

    Each race, with probability `share`, the other car is this driver's (else the scripted one's), at a speed multiple
    drawn from `scale` (lo, hi). Its steering and speed are driven exactly (`env.ideal_cmd`, `env.ideal_extra`: the
    kinematic execution the learner's plans get, with the car's steering-rate and acceleration limits and no tyre
    limit). Inputs: its own row's scan (`env.last_result.scan`) and the simulator's pose and speed of that row."""

    def __init__(self, env, share: float = 1.0, scale=(0.6, 1.0), seed: int = 0, cfg: Optional[AurorusConfig] = None,
                 rows: Optional[torch.Tensor] = None):
        self.env, self.share = env, float(share)
        self.scale_range = (float(scale[0]), float(scale[1]))
        self.rows = torch.nonzero(~env.learner).flatten() if rows is None else rows
        dev = env.device
        Bo = int(self.rows.numel())
        P = env.sim.P
        r0 = int(self.rows[0]) if Bo else 0
        track = env.sim.track
        clear = (cfg or AurorusConfig()).wall_clearance_m
        # the scanner's place as `gym_env._lidar_pose` has it: per car when randomized, else the configured one
        cl = env.cfg.lidar
        mx = P["mount_x"] if "mount_x" in P else torch.full((env.B,), float(cl.mount_x), device=dev)
        my = P["mount_y"] if "mount_y" in P else torch.full((env.B,), float(cl.mount_y), device=dev)
        self.core = AurorusCore(
            Line(env.teacher.xy, env.teacher.v), lambda xy, tid: track.sample_edt(xy, tid.view(-1, *([1] * (xy.dim() - 2)))) > clear,
            Bo, int(env.last_result.scan.shape[1]), float(env.cfg.lidar.fov), float(P["lf"][r0] + P["lr"][r0]), float(P["s_max"][r0]),
            mount=(mx[self.rows].float(), my[self.rows].float()), dt=float(env.sim.control_dt),
            range_min=float(env.cfg.lidar.range_min), range_max=float(env.cfg.lidar.range_max), cfg=cfg, device=dev)
        self.on = torch.zeros(Bo, dtype=torch.bool, device=dev)       # driven by this driver this race
        self.scale = torch.ones(Bo, device=dev)
        self.gen = torch.Generator(device="cpu"); self.gen.manual_seed(int(seed) + 9173)
        self.restarted = torch.zeros(Bo, dtype=torch.bool, device=dev)
        self.live = self.on.clone()
        self.xy = env.sim.state[self.rows, :2].clone()
        if getattr(env, "ideal_extra", None) is None:
            env.ideal_extra = torch.zeros(env.B, dtype=torch.bool, device=dev)
        self.draw(torch.arange(Bo, device=dev))

    def draw(self, rows: torch.Tensor) -> None:
        """rows (into the opponent rows) whose race just started: this driver or the scripted one, and how fast"""
        n = int(rows.numel())
        if n == 0:
            return
        u = torch.rand(n, 2, generator=self.gen)
        # a replayed race (`env.level_seed`) draws the same every time, as `SelfOpponents.draw`
        seed = getattr(self.env, "level_seed", None)
        if seed is not None:
            for i, sd in enumerate(seed[self.rows[rows]].tolist()):
                if sd >= 0:
                    g = torch.Generator(device="cpu"); g.manual_seed(int(sd) + 9173)
                    u[i] = torch.rand(2, generator=g)
        u = u.to(self.on.device)
        lo, hi = self.scale_range
        self.on[rows] = u[:, 0] < self.share
        self.scale[rows] = lo + (hi - lo) * u[:, 1]
        self.core.reset(rows)

    def _moved(self) -> None:
        """a car put somewhere else between decisions (`state_rl.OppRestart`): its nearest-point hint and switch are of
        the old place"""
        xy = self.env.sim.state[self.rows, :2]
        moved = (xy - self.xy).norm(dim=1) > 2.0
        if bool(moved.any()):
            self.core.reset(moved)
        self.xy = xy.clone()

    def plan(self) -> None:
        env = self.env
        self._moved()
        self.live = self.on & ~env.sim.absent[self.rows] if getattr(env.sim, "absent_any", False) else self.on.clone()
        self.restarted.zero_()
        env.sim.ideal_vcap[self.rows[self.live]] = 1e3
        env.ideal_extra[self.rows] = self.live

    def command(self, j: int) -> None:
        env, rows = self.env, self.rows
        st = env.sim.state[rows]
        pose = torch.stack([st[:, dyn.IX], st[:, dyn.IY], st[:, dyn.IYAW]], -1)
        steer, v = self.core.cycle(pose, st[:, dyn.IVX], env.last_result.scan[rows], env.sim.tid[rows],
                                   env.sim.P["a_brake"][rows], self.scale, self.live & ~self.restarted)
        env.ideal_cmd[rows, 0] = steer.to(env.ideal_cmd.dtype)
        env.ideal_cmd[rows, 1] = v.to(env.ideal_cmd.dtype)

    def stepped(self, done: torch.Tensor) -> None:
        d = done[self.rows]
        self.restarted |= d
        if bool(d.any()):
            self.core.reset(d)

    def after(self) -> None:
        self._moved()
        if bool(self.restarted.any()):
            self.draw(torch.nonzero(self.restarted).flatten())

    def share_on(self) -> float:
        pres = ~self.env.sim.absent[self.rows] if getattr(self.env.sim, "absent_any", False) else torch.ones_like(self.on)
        return float((self.on & pres).float().sum() / pres.float().sum().clamp_min(1.0))
