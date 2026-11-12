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
  * each beam is placed from where the car was when that beam was taken (`sweep`: the car's pose a cycle ago and
    the scanner's timing, `Lidar.time_frac`), where the original places every beam from the cycle's one pose (its
    own simulator casts them all from one pose too). Measured 11-12: this is NOT what made the port see obstacles
    on an empty track -- placed as the original does, placed like this, or with the simulator's sweep switched off,
    it was avoiding 36 / 30 / 35 % of the time;
  * returns from the floor can be dropped (`FloorGate` with `TiltEstimate`; `AurorusOpponents(floor="gate")`). The
    original takes every return on free space for an obstacle and its simulator's scanner is flat. Here the body
    rolls 1.7 deg/g with the scan plane 11 cm above the floor, so in a corner the beams of the low side meet the
    floor inside 4.35 m: on the real map without one box all 5566 obstacle points of a run were floor returns, and
    every wall contact of the solo record came while it was avoiding them. The gate uses what the car has: its gyro
    and speed for the body's tilt, and how its scanner is mounted;
  * `AurorusOpponents(drive="physics")` sends the steering and speed through the simulator's car (tyres, command
    delay, servo, motor), as the original's go to a car, and through that car's measured steering trim, steering gain
    and speed gain as every other driver's commands go (`gym_env._external_commands`; until 11-12 evening they went
    raw, to a servo trimmed up to +-0.03 rad off with gains up to 8 % off). "exact" is the kinematic execution the
    learner's plans get, which has no tyre limit: at the line's full speed the port cornered at up to 14 m/s^2 there;
  * free space = `TrackTensors.sample_edt` > 0.2 m (nearest-cell look-up of the distance to the nearest mapped wall);
  * its own pose is the simulator's (the original's `pose_source = 1`, its simulation-only ground-truth mode; on the
    car it localises). Other cars and boxes come from its LiDAR only;
  * the unused second controller (`controller = 1`) is not ported;
  * `AurorusConfig(follower="map")` hands the path to the MAP controller (`map_controller.MAPController`, the
    race_stack's, its steering table of this simulator's car) in place of the aim point of step 4 and the steering of
    step 5, which were made for the original's 4 m/s simulator. The path is waypoints 0.1 m apart along the line from
    1 m behind the car to 10 m ahead: the line itself, or while it avoids the chosen path -- its cubic offset to its
    end, then alongside the line at the end's offset -- with the line's speeds (x the car's multiple x step 4's cut for
    a bending path) at each. MAP's speed is still capped by step 4's stopping speed. What it sees and which path it
    picks are untouched. MAP's parameters are the caller's (`map_params`), tuned on the distance from the path it was
    handed, never on lap time or contacts; unnamed, the ones on file for the env's map (`map_params_for`).
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
from ..lidar import HIT_GROUND


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
    #: not the original's (None = it has one value, the one above, for both): the distance from the mapped walls a
    #: candidate path's centre must keep, apart from the one that makes a scan point an obstacle. 11-12, 2 boxes on
    #: Korea 2025: at a wall contact while avoiding the car was on its path (0.01-0.06 m) and the path's centre
    #: 0.29-0.32 m from the wall -- a 0.31 x 0.58 m body does not fit there at an angle. Raising the one value
    #: instead would also stop it seeing boxes that close to a wall.
    path_wall_clearance_m: Optional[float] = None
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
    # not the original's (module notes): what turns its path into steering and speed. "original" = the lines above;
    # "map" = the MAP controller on waypoints laid along the line / the chosen path
    follower: str = "original"
    map_spacing_m: float = 0.1           # the waypoints handed to MAP: this far apart along the line,
    map_behind_m: float = 1.0            # from this far behind the car's place on the line
    map_ahead_m: float = 10.0            # to this far ahead of it


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


@dataclass
class FloorGate:
    """Which returns are the floor (not the original's; see the module notes). A beam of a scan plane that is not level
    meets the floor at (the scanner's height) / (how steeply the beam points down); a return at or beyond that range,
    with the tilt taken `tol` [rad] worse than estimated, is not an obstacle.
    height: the scanner above the floor [m]; roll / pitch: how it sits on the car [rad], + = right side down / nose
    down as `lidar.Lidar.rays` has them. Floats or (n,) tensors."""
    height: object
    roll: object = 0.0
    pitch: object = 0.0
    tol: float = 0.01


class TiltEstimate:
    """The body's roll and pitch [rad] from the car's own gyro and speed, for `FloorGate`: the gyro's roll and pitch
    rates integrated, pulled with time constant `tau` toward where the suspension settles under the car's own
    acceleration (sideways = speed x yaw rate, along = the change of speed) by the car's gains [rad per g]:
    `roll_per_g`, `pitch_per_g` under throttle, `dive_per_g` under braking (floats or (n,) tensors).
    The VESC's own attitude estimate is no use for this: it leans on the accelerometer, which reads a corner as tilt."""

    G = 9.81

    def __init__(self, n: int, roll_per_g, pitch_per_g, dive_per_g, tau: float = 0.3, device=None, dtype=torch.float64):
        f = lambda v: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) else torch.full((n,), float(v), dtype=dtype, device=device))
        self.c_roll, self.c_pitch, self.c_dive, self.tau = f(roll_per_g), f(pitch_per_g), f(dive_per_g), float(tau)
        self.tilt = torch.zeros(n, 2, dtype=dtype, device=device)
        self.v_prev = torch.full((n,), float("nan"), dtype=dtype, device=device)

    def reset(self, mask: torch.Tensor) -> None:
        self.tilt[mask] = 0.0
        self.v_prev[mask] = float("nan")

    def update(self, speed: torch.Tensor, yaw_rate: torch.Tensor, rates: Optional[torch.Tensor], ts: float, dt: float) -> torch.Tensor:
        """One cycle of every car. speed (n,) [m/s]; yaw_rate (n,) [rad/s]; rates (n, K, 2): the gyro's roll and pitch
        rates of this cycle's K samples, `ts` apart (None or K = 0: none came); dt: the cycle. -> roll, pitch (n, 2)"""
        ft = self.tilt.dtype
        v, r = speed.to(ft), yaw_rate.to(ft)
        ax = torch.where(torch.isnan(self.v_prev), torch.zeros_like(v), (v - self.v_prev) / dt)
        self.v_prev = v.clone()
        settle = torch.stack([self.c_roll * v * r / self.G, -torch.where(ax > 0, self.c_pitch, self.c_dive) * ax / self.G], 1)
        t = self.tilt
        if rates is not None and rates.shape[1] > 0:
            t = t + rates.to(ft).sum(1) * ts
        self.tilt = t + min(dt / self.tau, 1.0) * (settle - t)
        return self.tilt


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
    free_path: the same for the candidate paths' points when they are to keep further from the walls (None = `free`);
    n_beams / fov / range_min / range_max: the scanner, beam i at -fov/2 + i fov/(n_beams-1);
    mount: the scanner's place ahead / left of the centre of gravity [m] (floats or (n,) tensors);
    wheelbase, max_steer: the car's; dt: the cycle [s] (the switch counts cycles);
    sweep (n_beams,): how long before the scan's time each beam was taken, in cycles -- each beam's end is then placed
    from where the car was at that moment (between the last cycle's pose and this one's). None = every beam from this
    cycle's pose, as the original does;
    floor: `FloorGate` -- with the body's tilt handed to `cycle`, returns from the floor are not obstacle points. None
    = every return on free space is one, as the original has it;
    map_ctrl: a `map_controller.MAPController` of n cars, for `cfg.follower == "map"` (`build_map`); map_grip: which
    of its steering tables the cars read (the nominal grip's: a car does not know its own)."""

    def __init__(self, line: Line, free: Callable, n: int, n_beams: int, fov: float, wheelbase: float, max_steer: float,
                 mount=(0.0, 0.0), dt: float = 0.025, range_min: float = 0.02, range_max: float = 10.0,
                 cfg: Optional[AurorusConfig] = None, device=None, chunk: int = 128, sweep: Optional[torch.Tensor] = None,
                 floor: Optional[FloorGate] = None, map_ctrl=None, map_grip: Optional[int] = None,
                 free_path: Optional[Callable] = None):
        self.cfg = cfg = cfg or AurorusConfig()
        assert cfg.follower in ("original", "map"), cfg.follower
        assert cfg.follower != "map" or map_ctrl is not None, 'follower "map" needs its controller (`build_map`)'
        self.line, self.free, self.n = line, free, int(n)
        self.free_path = free_path or free                             # `AurorusConfig.path_wall_clearance_m`
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
        self.sweep = None if sweep is None else sweep.to(dev)[self.beams].to(ft)
        self.pose_prev = torch.zeros(n, 3, dtype=ft, device=dev)       # last cycle's pose, where `has_prev`
        self.floor = floor
        if floor is not None:
            g = lambda v: (v.to(device=dev, dtype=ft) if torch.is_tensor(v) else torch.full((n,), float(v), dtype=ft, device=dev))
            self.fl_h, self.fl_roll, self.fl_pitch = g(floor.height), g(floor.roll), g(floor.pitch)
        self.has_prev = torch.zeros(n, dtype=torch.bool, device=dev)
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
        self.map = map_ctrl if cfg.follower == "map" else None
        if self.map is not None:
            from ..map_controller import GRIP_LEVELS
            g = GRIP_LEVELS.index(1.0) if map_grip is None else int(map_grip)
            self.map_grip = torch.full((n,), g, dtype=torch.long, device=dev)
        # the waypoints' places along the line, from the car's own
        self.wu = torch.arange(-cfg.map_behind_m, cfg.map_ahead_m + 0.5 * cfg.map_spacing_m, cfg.map_spacing_m, dtype=ft, device=dev)
        self.keep_way = False                                          # True: `last["way"]` under the original follower too (to measure tracking)

    def reset(self, mask: torch.Tensor) -> None:
        """cars (bool (n,) or indices) that start anew: no nearest-point hint, on the line, the planner's memory blank"""
        if self.map is not None:
            m = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
            m[mask] = True
            self.map.reset(m)
        self.hint[mask] = -1
        self.avoiding[mask] = False
        self.since[mask] = -1
        self.d_weight[mask] = 0.0
        self.blocked_gain[mask] = 1.0
        self.prev_err[mask] = float("nan")
        self.lost[mask] = False
        self.has_prev[mask] = False

    def _mount(self, idx: torch.Tensor):
        mx, my = self.mount
        mx = mx[idx].to(self.dtype) if torch.is_tensor(mx) else torch.full((idx.numel(),), float(mx), dtype=self.dtype, device=self.dev)
        my = my[idx].to(self.dtype) if torch.is_tensor(my) else torch.full((idx.numel(),), float(my), dtype=self.dtype, device=self.dev)
        return mx, my

    def floor_range(self, tilt: torch.Tensor, idx: torch.Tensor, mx: torch.Tensor, my: torch.Tensor) -> torch.Tensor:
        """the range (R, M) from which a return of each read beam is taken for the floor: the body of cars `idx` rolled
        and pitched by tilt (R, 2), the beam pointing down `tol` more than that says (`lidar.Lidar.rays`' geometry).
        inf where the beam does not point down"""
        phi = (tilt[:, 0] + self.fl_roll[idx])[:, None]
        th = (tilt[:, 1] + self.fl_pitch[idx])[:, None]
        ca, sa = torch.cos(self.beam_angle)[None], torch.sin(self.beam_angle)[None]
        down = ca * torch.sin(th) - sa * torch.sin(phi) * torch.cos(th) + self.floor.tol
        oz = -mx[:, None] * torch.sin(th) + (my[:, None] * torch.sin(phi) + self.fl_h[idx][:, None] * torch.cos(phi)) * torch.cos(th)
        return torch.where(down > 0, oz / down.clamp_min(1e-9), torch.full_like(down, float("inf")))

    def obstacle_points(self, pose: torch.Tensor, scan: torch.Tensor, tid: torch.Tensor, mx: torch.Tensor, my: torch.Tensor,
                        prev: Optional[torch.Tensor] = None, tilt: Optional[torch.Tensor] = None,
                        idx: Optional[torch.Tensor] = None):
        """`find_obstacles`: -> points (R, M, 2) in the world and which of them count (R, M).
        prev (R, 3): the pose a cycle ago, used with `sweep` to place each beam from where the car was when it was taken;
        tilt (R, 2) with the cars' numbers idx (R,): the body's roll and pitch, used with `floor` to drop floor returns"""
        cfg = self.cfg
        r = scan[:, self.beams]
        ok = torch.isfinite(r) & (r > self.range_min) & (r < self.range_max) & (r < cfg.max_path_length_m)
        self.n_floor = None
        if self.floor is not None and tilt is not None:
            fl = ok & (r >= self.floor_range(tilt, idx, mx, my))
            self.n_floor = fl.sum(1)
            ok = ok & ~fl
        r = torch.where(ok, r, torch.zeros_like(r))
        if self.sweep is None or prev is None:
            c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
            ox = pose[:, 0] + mx * c - my * s
            oy = pose[:, 1] + mx * s + my * c
            ang = pose[:, 2:3] + self.beam_angle[None]
            pts = torch.stack([ox[:, None] + r * torch.cos(ang), oy[:, None] + r * torch.sin(ang)], -1)
        else:
            f = self.sweep[None]                                       # (1, M)
            yaw = pose[:, 2:3] - f * wrap(pose[:, 2] - prev[:, 2])[:, None]
            x = pose[:, 0:1] - f * (pose[:, 0:1] - prev[:, 0:1])
            y = pose[:, 1:2] - f * (pose[:, 1:2] - prev[:, 1:2])
            c, s = torch.cos(yaw), torch.sin(yaw)
            ang = yaw + self.beam_angle[None]
            pts = torch.stack([x + mx[:, None] * c - my[:, None] * s + r * torch.cos(ang),
                               y + mx[:, None] * s + my[:, None] * c + r * torch.sin(ang)], -1)
        ok = ok & self.free(pts, tid)
        return pts, ok

    @torch.no_grad()
    def cycle(self, pose: torch.Tensor, speed: torch.Tensor, scan: torch.Tensor, tid: torch.Tensor, a_brake: torch.Tensor,
              scale: torch.Tensor, live: Optional[torch.Tensor] = None, tilt: Optional[torch.Tensor] = None,
              yaw_rate: Optional[torch.Tensor] = None, accel: Optional[torch.Tensor] = None):
        """One control cycle. pose (n, 3) = x, y, heading of the centre of gravity; speed (n,) forward [m/s];
        scan (n, n_beams) ranges [m] (inf = no return); tid (n,); a_brake (n,) [m/s^2]; scale (n,): each car's multiple
        of the line's speeds; live (n,) bool: cars this drives now (the others get 0, 0 and keep their state);
        tilt (n, 2): the body's roll and pitch [rad] as the car estimates them (`TiltEstimate`), for `floor`;
        yaw_rate (n,) [rad/s] and accel (n, k) -- the IMU's forward accelerations since the last cycle [m/s^2], oldest
        first -- are read by the "map" follower only (MAP's delay compensation and its steering scaling).
        -> steering [rad] (n,), speed [m/s] (n,)"""
        cfg, line, dev, ft = self.cfg, self.line, self.dev, self.dtype
        self.k += 1
        use_map = self.map is not None
        if use_map and accel is not None and accel.shape[1] > 0:
            self.map.push_imu(accel.to(self.map.acc_now.dtype))
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
        prev = torch.where(self.has_prev[idx, None], self.pose_prev[idx], p)
        pts, ok = self.obstacle_points(p, scan[idx].to(ft), t, mx, my, prev, None if tilt is None else tilt[idx].to(ft), idx)
        self.has_prev.zero_()                                          # a car not driven this cycle has no last pose next time
        self.pose_prev[idx] = p; self.has_prev[idx] = True
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
        stop_all = torch.full_like(ref, float("inf"))
        cubic = torch.zeros(int(idx.numel()), 6, dtype=ft, device=dev)
        for a in range(0, int(plan_rows.numel()), self.chunk):
            j = plan_rows[a:a + self.chunk]
            tg, gain, reach, d_w[j], blocked[j], cubic[j] = self._plan(t[j], p[j], s0[j], npt[j], look[j], pts[j], ok[j],
                                                                       d_w[j], blocked[j])
            target[j] = tg
            v = ref[j] * gain
            stop_all[j] = stopping_speed(reach, cfg.braking_margin_m, a_brake[idx][j].to(ft))
            v_cmd[j] = torch.minimum(v, stop_all[j])
            reach_all[j], gain_all[j] = reach, gain
        # steering (`steering.rs`): the aim point's bearing from one wheelbase behind the centre of gravity
        bx = p[:, 0] - self.wheelbase * torch.cos(p[:, 2])
        by = p[:, 1] - self.wheelbase * torch.sin(p[:, 2])
        err = wrap(torch.atan2(target[:, 1] - by, target[:, 0] - bx) - p[:, 2])
        prev = self.prev_err[idx]
        deriv = torch.where(torch.isnan(prev), torch.zeros_like(err), (err - prev) / self.dt)
        steer = (cfg.kk_s * err + (cfg.kd_s * deriv).clamp(-0.2, 0.2)).clamp(-self.max_steer, self.max_steer)
        way = way_v = way_k = None
        if use_map or self.keep_way:
            planned = torch.zeros_like(lost); planned[plan_rows] = True
            way, way_v, way_k = self._waypoints(t, s0, scale[idx].to(ft), planned, cubic, gain_all)
        if use_map:
            # MAP on the path as waypoints, in place of the aim point and the steering above; step 4's stopping speed stays
            h0 = line.heading_at(t, s0[:, None]).squeeze(1)
            off = -torch.sin(h0) * (p[:, 0] - npt[:, 0]) + torch.cos(h0) * (p[:, 1] - npt[:, 1])
            steer, v_map, target = self._map_call(idx, p, sp, way, way_v, way_k, off, None if yaw_rate is None else yaw_rate[idx])
            v_cmd = torch.minimum(v_map, stop_all)
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
        # `pts` / `ok`: the scan points and which of them it took for obstacles; `plan_rows`: the rows (into `idx`) that
        # planned a path -- `last_plan` holds the last chunk of them; `pose` / `npt`: the car and its nearest line point;
        # `way` (R, n, 2), `way_v`, `way_k` (R, n): the waypoints MAP was handed with their speeds and curvature ("map"
        # follower; `target` is then MAP's aim point), else None; `stop` (R,): step 4's stopping speed (inf = none)
        self.last = dict(idx=idx, target=target, s0=s0, dist=dist, obst=obst, n_obst=ok.sum(1), reach=reach_all, gain=gain_all,
                         look=look, ref=ref, pts=pts, ok=ok, plan_rows=plan_rows, pose=p, npt=npt, n_floor=self.n_floor,
                         way=way, way_v=way_v, way_k=way_k, stop=stop_all)
        return steer_out, v_out

    def _waypoints(self, tid, s0, scale, planned, cubic, gain):
        """The path as MAP's waypoints, for the "map" follower: xy (R, n, 2), speed (R, n), curvature (R, n), one every
        `map_spacing_m` of the PATH's own length (MAP counts waypoints for its look-ahead distance) from `map_behind_m`
        behind the car's place on the line. Rows not `planned`: the line. Rows `planned`: the chosen path -- `cubic`
        (R, 6) = d0, slope0, a2, a3, length, end offset of its `LateralCubic` -- then alongside the line at the end's
        offset; behind the car, straight back from where it is. Speeds: the line's x `scale` (x `gain` where planned)."""
        cfg, line = self.cfg, self.line
        R, n = int(tid.numel()), int(self.wu.numel())
        u = self.wu[None]
        s = s0[:, None] + u
        base, v = line.at(tid, s)
        hd = line.heading_at(tid, s)
        d0, sl, a2, a3, L, d1 = (c[:, None] for c in cubic.unbind(1))
        d = torch.where(u < 0, d0 + sl * u, torch.where(u <= L, d0 + u * (sl + u * (a2 + u * a3)), d1.expand(R, n)))
        d = torch.where(planned[:, None], d, torch.zeros_like(d))
        P = torch.stack([base[..., 0] - d * torch.sin(hd), base[..., 1] + d * torch.cos(hd)], -1)
        v = scale[:, None] * v * torch.where(planned, gain, torch.ones_like(gain))[:, None]
        # evenly along the path itself: an offset path's points are closer together inside a corner, further apart outside
        al = torch.cat([torch.zeros_like(d[:, :1]), (P[:, 1:] - P[:, :-1]).norm(dim=-1).cumsum(1)], 1)
        q = torch.minimum((cfg.map_spacing_m * torch.arange(n, dtype=d.dtype, device=d.device))[None].expand(R, n), al[:, -1:])
        i = (torch.searchsorted(al, q.contiguous(), right=True) - 1).clamp(0, n - 2)
        a, b = al.gather(1, i), al.gather(1, i + 1)
        w = ((q - a) / (b - a).clamp_min(1e-9)).clamp(0.0, 1.0)
        g2 = lambda A, k: A.gather(1, k[..., None].expand(-1, -1, 2))
        P = g2(P, i) + w[..., None] * (g2(P, i + 1) - g2(P, i))
        v = v.gather(1, i) + w * (v.gather(1, i + 1) - v.gather(1, i))
        # curvature as `_plan` takes a path's: the heading's change over the mean of the two steps
        dP = P[:, 1:] - P[:, :-1]
        st = dP.norm(dim=-1)
        yaw = torch.atan2(dP[..., 1], dP[..., 0])
        both = (st[:, 1:] > 1e-6) & (st[:, :-1] > 1e-6)                # the array's end repeats where the path ran out
        k = torch.where(both, wrap(yaw[:, 1:] - yaw[:, :-1]) / ((st[:, 1:] + st[:, :-1]) / 2).clamp_min(1e-6), torch.zeros_like(st[:, 1:]))
        return P, v, torch.cat([k[:, :1], k, k[:, -1:]], 1)

    def _map_call(self, idx, pose, speed, way, way_v, way_k, off, yaw_rate):
        """MAP for the cars `idx` (its state holds all n). off (R,): the car's offset from the line, MAP's lateral error
        to the global line. -> steering (R,), speed (R,), MAP's aim point (R, 2)"""
        m, f = self.map, self.map.curr_steer.dtype
        names = ("curr_steer", "i_gap", "acc_now", "curv_wp")
        full = [getattr(m, k) for k in names]
        for k, x in zip(names, full):
            setattr(m, k, x[idx])
        try:
            R = int(idx.numel())
            steer, v = m(pose[:, :2].to(f), pose[:, 2].to(f), speed.to(f), way.to(f), way_v.to(f), way_k.to(f),
                         torch.full((R,), self.cfg.map_spacing_m, dtype=f, device=self.dev), off.to(f), self.map_grip[idx],
                         yaw_rate=None if yaw_rate is None else yaw_rate.to(f))
            for k, x in zip(names, full):
                x[idx] = getattr(m, k)
        finally:
            for k, x in zip(names, full):
                setattr(m, k, x)
        return steer.to(self.dtype), v.to(self.dtype), m.last["l1_point"].to(self.dtype)

    def _plan(self, tid, pose, s0, npt, look, obs, obs_ok, d_weight, blocked):
        """`Planner::plan` for rows that avoid. -> aim point (R, 2), speed gain (R,), reach (R,; inf = no stop needed),
        the planner's two remembered values after it, the chosen path's `LateralCubic` (R, 6) = d0, slope0, a2, a3, its
        length along the line, its end offset (for the "map" follower's waypoints)"""
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
        on_map = (self.free_path(P, tid) | ~self.check[None]).all(-1)
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
        cubic = torch.stack([d0, slope0, a2[rows, sel], a3[rows, sel], self.cL[sel], self.cd1[sel]], 1)
        return target, gain, reach, d_weight, blocked, cubic


def map_params_for(env) -> Optional[dict]:
    """MAP's parameters on file for the env's map (`baseline_settings`): the port's own set when there is one (kind
    "aurorus", `options.map_params`: the steering-side values searched by the lateral RMSE to its own line, 11-12),
    else the set tuned per map for a car that follows a line alone (`raceline` / solo). A warped or scaled copy of a
    map reads its map's. None: nothing on file."""
    from . import baseline_settings
    maps = baseline_settings.load_all()
    for tr in env.sim.tracks:
        for name in sorted((m for m in maps if tr.name.startswith(m)), key=len, reverse=True):
            own = baseline_settings.lookup(name, "aurorus", solo=True)
            mp = ((own or {}).get("options") or {}).get("map_params")
            if mp:
                return dict(mp)
            st = baseline_settings.lookup(name, "raceline", solo=True)
            mp = ((st or {}).get("stack_params") or {}).get("map")
            if mp:
                return dict(mp)
    return None


def tuned_options(env) -> dict:
    """`AurorusDriver`'s options on file for the env's map (`baseline_settings`, kind "aurorus"): which follower drives
    its path and, when named, its scale. The MAP values themselves are read by `map_params_for`. {}: nothing on file."""
    from . import baseline_settings
    maps = baseline_settings.load_all()
    for tr in env.sim.tracks:
        for name in sorted((m for m in maps if tr.name.startswith(m)), key=len, reverse=True):
            own = baseline_settings.lookup(name, "aurorus", solo=False)
            if own:
                return {k: v for k, v in (own.get("options") or {}).items() if k != "map_params"}
    return {}


def build_map(env, n: int, params: Optional[dict] = None):
    """The MAP controller of `n` cars of `env` for `AurorusConfig(follower="map")`: the steering table of the
    simulator's nominal car (cached, the one the planner baselines' MAP reads) and `params` (`MAPParams` fields);
    None = the ones on file for the env's map (`map_params_for`), else `MAPParams`' own (the race_stack's car's)."""
    from .. import map_controller as M
    params = map_params_for(env) if params is None else params
    return M.MAPController(M.nominal_lut(env.sim), M.MAPParams.from_dict(params), int(n), env.device,
                           float(env.sim.params.nominal["s_max"]))


def car_calibration(env) -> torch.Tensor:
    """(B, 3) each car's measured steering trim [rad], steering gain and speed gain, the correction every driver's
    command gets on its way to the actuators (`gym_env._external_commands`)"""
    cal = getattr(env, "tracker_cal", None)
    cal = cal if cal is not None else getattr(env, "_ext_cal", None)
    if cal is None:
        env._calibrate_external(torch.arange(env.B, device=env.device))
        cal = env._ext_cal
    return cal


class AurorusOpponents:
    """The other cars of a race driven by `AurorusCore`, behind the interface `plan_rl.Driver` calls around a decision's
    control steps (`plan` / `command` / `stepped` / `after`, as `plan_rl.SelfOpponents`).

    Each race, with probability `share`, the other car is this driver's (else the scripted one's), at a speed multiple
    drawn from `scale` (lo, hi). Inputs: its own row's scan (`env.last_result.scan`) and the simulator's pose and speed
    of that row.
    drive: "exact" -- its steering and speed are driven exactly (`env.ideal_cmd`, `env.ideal_extra`: the kinematic
      execution the learner's plans get, with the car's steering-rate and acceleration limits and no tyre limit);
      "physics" -- they are the command to the simulator's car (`env.ext_cmd`, `env.ext_mask`: tyres, command delay,
      servo, motor), corrected by that car's measured steering trim and gains (`car_calibration`) as every driver's
      is. For rows of the learner this needs the env's exact execution off: `env.ideal_cmd = None` and
      `env.sim.ideal_rows.zero_()` IN PLACE (a GPU env's captured physics graph keeps reading that tensor; setting the
      attribute to None leaves those rows exact).
    map_params: `MAPParams` fields for `cfg.follower == "map"` (`build_map`; None = the ones on file for the map).
    floor: "off" -- every return on free space is an obstacle point, as the original has it;
      "gate" -- returns at the range where the car's own tilt estimate (gyro + speed, `TiltEstimate` with the nominal
      car's gains) puts the floor are dropped (`FloorGate`: the car's scanner height, a mount assumed level, `floor_tol`
      [rad] for what it does not know: the mount's tilt is randomized +-0.02 rad, the car's roll gain x0.6-1.6);
      "oracle" -- returns the SIMULATOR knows are the floor are dropped (`scan_type`): no car has that, a reference
      for the gate only."""

    def __init__(self, env, share: float = 1.0, scale=(0.6, 1.0), seed: int = 0, cfg: Optional[AurorusConfig] = None,
                 rows: Optional[torch.Tensor] = None, deskew: bool = True, drive: str = "exact", floor: str = "off",
                 floor_tol: float = 0.03, map_params: Optional[dict] = None):
        assert drive in ("exact", "physics") and floor in ("off", "gate", "oracle"), (drive, floor)
        self.env, self.share = env, float(share)
        self.drive, self.floor = drive, floor
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
        gate, self.tilt = None, None
        if floor == "gate":
            mz = P["mount_z"] if "mount_z" in P else torch.full((env.B,), float(cl.mount_z), device=dev)
            gate = FloorGate(height=mz[self.rows], tol=float(floor_tol))
            cv = env.cfg.vehicle
            self.tilt = TiltEstimate(Bo, cv.roll_per_g, cv.pitch_per_g, cv.dive_per_g, device=dev)
            self.imu_ts = 1.0 / float(env.cfg.imu.imu_rate)
        clear_p = (cfg or AurorusConfig()).path_wall_clearance_m
        self.core = AurorusCore(
            Line(env.teacher.xy, env.teacher.v), lambda xy, tid: track.sample_edt(xy, tid.view(-1, *([1] * (xy.dim() - 2)))) > clear,
            Bo, int(env.last_result.scan.shape[1]), float(env.cfg.lidar.fov), float(P["lf"][r0] + P["lr"][r0]), float(P["s_max"][r0]),
            mount=(mx[self.rows].float(), my[self.rows].float()), dt=float(env.sim.control_dt),
            range_min=float(env.cfg.lidar.range_min), range_max=float(env.cfg.lidar.range_max), cfg=cfg, device=dev,
            sweep=env.sim.lidar.time_frac if deskew and getattr(env.cfg.lidar, "motion_distortion", False) else None,
            floor=gate, map_ctrl=build_map(env, Bo, map_params) if (cfg or AurorusConfig()).follower == "map" else None,
            free_path=None if clear_p is None else
            (lambda xy, tid: track.sample_edt(xy, tid.view(-1, *([1] * (xy.dim() - 2)))) > clear_p))
        self._row_ids = self.rows.tolist()
        self.on = torch.zeros(Bo, dtype=torch.bool, device=dev)       # driven by this driver this race
        self.scale = torch.ones(Bo, device=dev)
        self.gen = torch.Generator(device="cpu"); self.gen.manual_seed(int(seed) + 9173)
        self.restarted = torch.zeros(Bo, dtype=torch.bool, device=dev)
        self.live = self.on.clone()
        self.xy = env.sim.state[self.rows, :2].clone()
        if drive == "exact" and getattr(env, "ideal_extra", None) is None:
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
        self._reset(rows)

    def _reset(self, rows: torch.Tensor) -> None:
        """rows (indices or a mask over the opponent rows) start anew"""
        self.core.reset(rows)
        if self.tilt is not None:
            self.tilt.reset(rows)

    def _moved(self) -> None:
        """a car put somewhere else between decisions (`state_rl.OppRestart`): its nearest-point hint and switch are of
        the old place"""
        xy = self.env.sim.state[self.rows, :2]
        moved = (xy - self.xy).norm(dim=1) > 2.0
        if bool(moved.any()):
            self._reset(moved)
        self.xy = xy.clone()

    def plan(self) -> None:
        env = self.env
        self._moved()
        self.live = self.on & ~env.sim.absent[self.rows] if getattr(env.sim, "absent_any", False) else self.on.clone()
        self.restarted.zero_()
        if self.drive == "physics":
            # the env's own book of externally commanded rows (`set_external_command`), kept for many rows at once
            env.ext_mask[self.rows] = self.live
            env._ext_ids.difference_update(self._row_ids)
            env._ext_ids.update(self.rows[self.live].tolist())
            if getattr(env, "ideal_extra", None) is not None:
                env.ideal_extra[self.rows] = False
        else:
            env.sim.ideal_vcap[self.rows[self.live]] = 1e3
            env.ideal_extra[self.rows] = self.live

    def command(self, j: int) -> None:
        env, rows = self.env, self.rows
        res = env.last_result
        st = env.sim.state[rows]
        pose = torch.stack([st[:, dyn.IX], st[:, dyn.IY], st[:, dyn.IYAW]], -1)
        scan, tilt = res.scan[rows], None
        imu = getattr(res, "imu", None)
        if self.floor == "oracle":
            scan = torch.where(res.scan_type[rows] == HIT_GROUND, torch.full_like(scan, float("inf")), scan)
        elif self.floor == "gate":
            tilt = self.tilt.update(st[:, dyn.IVX], st[:, dyn.IR], None if imu is None else imu[rows][:, :, :2], self.imu_ts,
                                    self.core.dt)
        steer, v = self.core.cycle(pose, st[:, dyn.IVX], scan, env.sim.tid[rows], env.sim.P["a_brake"][rows], self.scale,
                                   self.live & ~self.restarted, tilt, st[:, dyn.IR], None if imu is None else imu[rows][:, :, 3])
        if self.drive == "physics":
            cal = car_calibration(env)[rows].to(steer.dtype)
            steer = ((steer - cal[:, 0]) / cal[:, 1]).clamp(-self.core.max_steer, self.core.max_steer)
            v = v / cal[:, 2]
        out = env.ext_cmd if self.drive == "physics" else env.ideal_cmd
        out[rows, 0] = steer.to(out.dtype)
        out[rows, 1] = v.to(out.dtype)

    def stepped(self, done: torch.Tensor) -> None:
        d = done[self.rows]
        self.restarted |= d
        if bool(d.any()):
            self._reset(d)

    def after(self) -> None:
        self._moved()
        if bool(self.restarted.any()):
            self.draw(torch.nonzero(self.restarted).flatten())

    def share_on(self) -> float:
        pres = ~self.env.sim.absent[self.rows] if getattr(self.env.sim, "absent_any", False) else torch.ones_like(self.on)
        return float((self.on & pres).float().sum() / pres.float().sum().clamp_min(1.0))


# ----------------------------------------------------------------- one driver of an env's external rows (the console)
#: the speed multiple of a lone AuRoRus car when none is named (the console's 내 차): the fastest of 0.6 / 0.8 / 1.0 that
#: stayed off the walls on the tyres (Korea 2025, 2026-11-12 solo record: at 1.0 the car slides wide on corner exits)
SOLO_SCALE = 0.8
#: [m] of its own line drawn ahead of the car, and the points along it
VIZ_AHEAD_M = 8.0
VIZ_AHEAD_POINTS = 33


def build_core(env, rows: torch.Tensor, line: Line, cfg: Optional[AurorusConfig] = None, deskew: bool = True,
               floor: str = "off", floor_tol: float = 0.03, map_params: Optional[dict] = None):
    """`AurorusCore` for the cars `rows` of `env` on `line`, and what `floor="gate"` needs besides.
    map_params: `MAPParams` fields for `cfg.follower == "map"` (`build_map`).
    -> core, the tilt estimate (None unless "gate"), the IMU's sample time [s] (None unless "gate")"""
    dev = env.device
    Bo = int(rows.numel())
    P = env.sim.P
    r0 = int(rows[0]) if Bo else 0
    track = env.sim.track
    clear = (cfg or AurorusConfig()).wall_clearance_m
    # the scanner's place as `gym_env._lidar_pose` has it: per car when randomized, else the configured one
    cl = env.cfg.lidar
    mx = P["mount_x"] if "mount_x" in P else torch.full((env.B,), float(cl.mount_x), device=dev)
    my = P["mount_y"] if "mount_y" in P else torch.full((env.B,), float(cl.mount_y), device=dev)
    gate, tilt, imu_ts = None, None, None
    if floor == "gate":
        mz = P["mount_z"] if "mount_z" in P else torch.full((env.B,), float(cl.mount_z), device=dev)
        gate = FloorGate(height=mz[rows], tol=float(floor_tol))
        cv = env.cfg.vehicle
        tilt = TiltEstimate(Bo, cv.roll_per_g, cv.pitch_per_g, cv.dive_per_g, device=dev)
        imu_ts = 1.0 / float(env.cfg.imu.imu_rate)
    clear_p = (cfg or AurorusConfig()).path_wall_clearance_m
    core = AurorusCore(
        line, lambda xy, tid: track.sample_edt(xy, tid.view(-1, *([1] * (xy.dim() - 2)))) > clear,
        Bo, int(env.last_result.scan.shape[1]), float(env.cfg.lidar.fov), float(P["lf"][r0] + P["lr"][r0]), float(P["s_max"][r0]),
        mount=(mx[rows].float(), my[rows].float()), dt=float(env.sim.control_dt),
        range_min=float(env.cfg.lidar.range_min), range_max=float(env.cfg.lidar.range_max), cfg=cfg, device=dev,
        sweep=env.sim.lidar.time_frac if deskew and getattr(env.cfg.lidar, "motion_distortion", False) else None,
        floor=gate, map_ctrl=build_map(env, Bo, map_params) if (cfg or AurorusConfig()).follower == "map" else None,
        free_path=None if clear_p is None else
        (lambda xy, tid: track.sample_edt(xy, tid.view(-1, *([1] * (xy.dim() - 2)))) > clear_p))
    return core, tilt, imu_ts


def driver_line(env, rls=None, limits: Optional[dict] = None) -> Line:
    """The line an `AurorusDriver` follows in `env`: from the race lines `rls` (one per track) with the speed profile's
    limits `limits` (`common.teacher_limits`); without `rls`, the line the env's own scripted cars drive."""
    if rls:
        from .common import make_teacher
        t = make_teacher(rls, env, **(limits or {}))
    else:
        t = getattr(env, "teacher", None)
        if t is None or not hasattr(t, "xy"):
            raise ValueError("an AuRoRus car follows a race line: none was given (rls) and this env has none of its own")
    return Line(t.xy, t.v)


class AurorusDriver:
    """`AurorusCore` as an external driver of an env (`gym_env.set_ego_driver` / `set_external_drivers`: the console's
    내 차 or an opponent slot). One command per control cycle, sent through the car's calibration and normal physics by
    the env, as every external driver's is.

    What it reads: its own row's scan and gyro, and the SIMULATOR's pose and speed of its car -- the original's
    `pose_source = 1` (its simulation mode); on the real car the original localizes itself, which is not ported. It is
    told nothing about boxes or other cars.
    scale: the multiple of the line's speeds it asks for (`SOLO_SCALE` when none is named); an opponent slot's own
      draw for the race (`OpponentSlot.speed_scale`, `env.slot_scale`) multiplies it, so a slot's 1.0 is this driver's
      own full pace as it is a scripted car's.
    floor: as `AurorusOpponents` ("gate" here: the simulator's LiDAR sees the floor when the body tilts, and the
      original, written for a flat 2D scanner, takes those returns for obstacles).
    follower: "map" | "original" (`AurorusConfig.follower`; a `cfg` handed in keeps its own). None = "map": a car that
      races follows its path with the MAP controller, the original's aimed-point steering is for the comparison with
      the Rust code. map_params: `MAPParams` fields for "map" (`build_map`; None = the ones on file for the map).
    path_wall_clearance_m: `AurorusConfig.path_wall_clearance_m` (the value on file arrives here through `tuned_options`;
      a `cfg` handed in keeps its own)."""

    KIND = kind = "aurorus"
    needs_speed = False

    def __init__(self, line: Line, scale: Optional[float] = None, cfg: Optional[AurorusConfig] = None, deskew: bool = True,
                 floor: str = "gate", floor_tol: float = 0.03, follower: Optional[str] = None, map_params: Optional[dict] = None,
                 path_wall_clearance_m: Optional[float] = None):
        assert floor in ("off", "gate", "oracle"), floor
        if cfg is None:
            cfg = AurorusConfig(follower="map" if follower is None else follower, path_wall_clearance_m=path_wall_clearance_m)
        self.line, self.scale, self.cfg = line, float(SOLO_SCALE if scale is None else scale), cfg
        self.map_params = map_params
        self.deskew, self.floor, self.floor_tol = bool(deskew), floor, float(floor_tol)
        self.rows = None                                              # `set_ego_driver` fills it; else the first command's
        self.core = None
        self.tilt, self.imu_ts = None, None
        self.xy = None
        self.last_cmd = None
        self._at = {}

    def bind_scanner(self, **kw) -> None:
        """the scanner is read from the env when the first command is asked (`build_core`)"""

    def _build(self, env, rows: torch.Tensor) -> None:
        self.rows = rows
        self.core, self.tilt, self.imu_ts = build_core(env, rows, self.line, self.cfg, self.deskew, self.floor, self.floor_tol,
                                                       self.map_params)
        self.xy = env.sim.state[rows, :2].clone()
        self._at = {int(r): j for j, r in enumerate(rows.tolist())}

    def _reset(self, rows) -> None:
        self.core.reset(rows)
        if self.tilt is not None:
            self.tilt.reset(rows)

    def reset(self, mask=None) -> None:
        """mask: which of its rows start a new race (the env's bool array over them); None = all"""
        if self.core is None:
            return
        n = self.core.n
        m = torch.ones(n, dtype=torch.bool) if mask is None else torch.as_tensor(np.asarray(mask, bool))
        self._reset(m.to(self.core.dev))

    def command_env(self, env, rows: torch.Tensor) -> torch.Tensor:
        """(R, 2) steering [rad], speed [m/s] for its env rows `rows` (the same rows every call)"""
        if self.core is None:
            self._build(env, rows)
        res = env.last_result
        st = env.sim.state[rows]
        pose = torch.stack([st[:, dyn.IX], st[:, dyn.IY], st[:, dyn.IYAW]], -1)
        moved = (pose[:, :2] - self.xy).norm(dim=1) > 2.0             # put somewhere else: its line hint is of the old place
        if bool(moved.any()):
            self._reset(moved)
        self.xy = pose[:, :2].clone()
        scan, tilt = res.scan[rows], None
        imu = getattr(res, "imu", None)
        if self.floor == "oracle":
            scan = torch.where(res.scan_type[rows] == HIT_GROUND, torch.full_like(scan, float("inf")), scan)
        elif self.floor == "gate":
            tilt = self.tilt.update(st[:, dyn.IVX], st[:, dyn.IR], None if imu is None else imu[rows][:, :, :2], self.imu_ts,
                                    self.core.dt)
        scale = self.scale * env.slot_scale[rows]
        live = ~env.sim.absent[rows] if getattr(env.sim, "absent_any", False) else None
        steer, v = self.core.cycle(pose, st[:, dyn.IVX], scan, env.sim.tid[rows], env.sim.P["a_brake"][rows], scale, live, tilt,
                                   st[:, dyn.IR], None if imu is None else imu[rows][:, :, 3])
        self.last_cmd = torch.stack([steer, v], -1)
        return self.last_cmd

    # ---- its picture (`viewer.driver_viz`): what the last cycle saw and chose for one car
    def viz_tensors(self, env, row: int) -> list:
        """Flat tensors of env row `row` for the host: `viz_decode` reads them back (all blank before its first command)"""
        core, j = self.core, self._at.get(int(row))
        L = None if core is None else core.last
        nan = float("nan")
        if j is None or not L:
            self._viz_shape = (0, 0, 0)
            return [torch.full((10,), nan, device=env.device)]
        dev, K, NB = core.dev, core.K, int(core.beams.numel())
        head = torch.full((10,), nan, device=dev)
        path = torch.full((K, 2), nan, device=dev)
        obs = torch.full((NB, 2), nan, device=dev)
        ahead = torch.full((VIZ_AHEAD_POINTS, 3), nan, device=dev)
        hit = torch.nonzero(L["idx"] == j).flatten()
        if hit.numel():
            a = int(hit[0])
            sc = float(self.scale * env.slot_scale[int(row)])
            pj = L["plan_rows"]
            pos = torch.nonzero(pj == a).flatten()
            n_path = 0
            LP = core.last_plan if pos.numel() else None
            if LP is not None:
                k = int(pos[0]) - (int(pj.numel()) - int(LP["P"].shape[0]))   # `last_plan`: the last chunk of planning rows
                if k >= 0:
                    c = int(LP["chosen"][k])
                    n_path = int(LP["points"][c])
                    path[:n_path] = LP["P"][k, c, :n_path].float()
                    head[9] = float(LP["free"][k])
            tid = env.sim.tid[int(row)].view(1)
            s = L["s0"][a] + torch.linspace(0.0, VIZ_AHEAD_M, VIZ_AHEAD_POINTS, dtype=core.dtype, device=dev)
            pa, va = core.line.at(tid, s[None])
            ahead[:, :2] = pa[0].float(); ahead[:, 2] = sc * va[0].float()
            obs = torch.where(L["ok"][a][:, None], L["pts"][a].float(), obs)
            cmd_v = float(self.last_cmd[j, 1]) if self.last_cmd is not None else nan
            head[:9] = torch.tensor([float(bool(core.avoiding[j])), float(n_path), float(L["target"][a, 0]), float(L["target"][a, 1]),
                                     cmd_v, float(L["gain"][a]), float(L["reach"][a]), float(bool(core.lost[j])), sc], device=dev)
        self._viz_shape = (K, NB, VIZ_AHEAD_POINTS)
        return [head, path.flatten(), obs.flatten(), ahead.flatten()]

    def viz_decode(self, flat: np.ndarray) -> dict:
        """`viz_tensors` on the host -> avoiding, lost (bool), path (n, 2) its chosen path (empty unless avoiding),
        free (a path clear of every obstacle point existed), target (2,), speed (commanded) [m/s], gain, reach [m]
        (inf = no stop needed), scale, obstacles (m, 2) the scan points it took for obstacles, ahead (k, 3) its line
        ahead with the speed it asks for there"""
        K, NB, A = self._viz_shape
        f = np.asarray(flat, np.float64)
        h = f[:10]; i = 10
        path = f[i:i + 2 * K].reshape(K, 2); i += 2 * K
        obs = f[i:i + 2 * NB].reshape(NB, 2); i += 2 * NB
        ahead = f[i:i + 3 * A].reshape(A, 3)
        seen = np.isfinite(h[0])
        return dict(seen=bool(seen), avoiding=bool(seen and h[0] > 0.5), lost=bool(seen and h[7] > 0.5),
                    path=path[:int(h[1])] if seen else path[:0], free=bool(seen and h[9] > 0.5), target=h[2:4],
                    speed=float(h[4]), gain=float(h[5]), reach=float(h[6]), scale=float(h[8]),
                    obstacles=obs[np.isfinite(obs[:, 0])], ahead=ahead[np.isfinite(ahead[:, 0])])
