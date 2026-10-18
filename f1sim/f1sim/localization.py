"""Monte Carlo localisation for the modular baselines (2026-10-18).

The user: "모듈형은 왜 상대 위치랑 자기 위치를 gt로 써? 제정신이여? Localization Perception prediction은
개나줬냐?" A modular stack does not know where it is; it estimates it, and every error of that
estimate is part of the stack. This is the estimator the RoboRacer (F1TENTH) course teaches and the
published stacks descend from: the MIT racecar particle filter (Walsh & Karaman, ICRA 2018;
`github.com/mit-racecar/particle_filter`, `launch/localize.launch`) -- particles moved by the wheel
odometry, weighted by a beam model of a downsampled scan against the known map, resampled at every
scan. ForzaETH's SynPF is the same filter with a tuned motion model.

What it reads, and nothing else: the car's own VESC odometry (`/odom`: the dead-reckoned pose its
deltas are taken from), its LiDAR scan, the static map, and the LiDAR's NOMINAL mounting (the URDF's,
not this episode's misalignment). The static map is what a SLAM run before the race gives: walls and
hoses, not the crates placed afterwards, not other cars. The pose it starts from is the operator's
initial estimate -- the spawn pose blurred by `init_spread`, as the RViz "2D Pose Estimate" click.

Expected ranges come from the simulator's own tracer over the static grid only, at the nominal scan
height: the map the filter matches against is a perfect SLAM map. What stands between it and the
truth is everything a real filter faces -- range noise, motion distortion, body roll and pitch,
beams over the hoses, crates and cars that are not in the map, wheel slip in the odometry.

The sensor model is the MIT filter's table, built here in metres on a 5 cm lattice (their map's
pixel): a Gaussian hit (sigma_hit 8 px = 0.40 m), a linear short-reading term, a max-range spike and
a uniform floor, each column normalised; a scan's weight is the product over its beams raised to
1/squash. Their defaults are the defaults here; they were tuned for their car and map, and are
re-tuned on ours like every other baseline knob (`[[baseline-fair-tuning]]`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .dynamics import wrap_angle


@dataclass
class PFParams:
    """The MIT filter's `localize.launch` values, in metres and radians -- except two, tuned on this
    simulator the way a team tunes its filter on a bag (2026-10-18, `tools/baselines/pf_grid.sh`: 4 cars
    driving the raceline, pose error against the truth): disp_theta 0.25 -> 0.05 and sigma_hit 0.40 ->
    0.10. On ICRA 2022 that took the p95 error from 20.4 to 8.0 cm, the worst from 90 to 22 cm and the
    time lost beyond 0.5 m from 1.1 % to 0; ICCAS p95 10.5 cm, Korea 2026 9.6 cm, nothing lost."""
    n_particles: int = 1000            # theirs: max_particles 4000 (on a Jetson with RMGPU)
    angle_step: int = 18               # beams used: every 18th of 1081 -> 61
    max_range: float = 10.0
    z_hit: float = 0.75
    z_short: float = 0.01
    z_max: float = 0.07
    z_rand: float = 0.12
    sigma_hit: float = 0.10            # theirs 8 px at 0.05 m/px = 0.40; tuned here (below)
    squash: float = 2.2
    table_res: float = 0.05            # [m] the sensor model's lattice (their map pixel)
    disp_x: float = 0.05               # motion dispersion added every update, world x [m]
    disp_y: float = 0.025              # world y [m] (as theirs: added in the map frame)
    disp_theta: float = 0.05           # [rad]; theirs 0.25, tuned here (below)
    init_spread: tuple = (0.10, 0.10, 0.05)   # the operator's initial estimate: x, y [m], yaw [rad]


class ParticleFilter:
    """Batched MCL: one filter per tracked env row, `n_particles` each, all advanced in one call.

    `rows` are the env rows it localises (e.g. the ego car of every race). Everything is a tensor on
    the simulator's device; no host round trip per step.
    """

    def __init__(self, sim, rows: torch.Tensor, params: Optional[PFParams] = None,
                 mount=(0.297, 0.0, 0.110)):
        self.sim = sim
        self.p = params or PFParams()
        self.device = sim.state.device
        self.rows = rows.to(self.device).long()
        R, N = self.rows.numel(), int(self.p.n_particles)
        self.R, self.N = R, N
        lid = sim.cfg.lidar
        n_scan = int(lid.n_beams)
        self.beams = torch.arange(0, n_scan, int(self.p.angle_step), device=self.device)
        self.angles = torch.linspace(-0.5 * float(lid.fov), 0.5 * float(lid.fov), n_scan,
                                     device=self.device)[self.beams]
        self.K = self.beams.numel()
        self.mount = tuple(float(m) for m in mount)
        self.max_range = float(min(self.p.max_range, float(lid.range_max)))
        self.particles = torch.zeros(R, N, 3, device=self.device)
        self.weights = torch.full((R, N), 1.0 / N, device=self.device)
        self.estimate = torch.zeros(R, 3, device=self.device)
        self._odom_prev = torch.zeros(R, 3, device=self.device)
        self._ready = torch.zeros(R, dtype=torch.bool, device=self.device)
        self._table = self._sensor_table()
        # built once: a tensor made from Python numbers is a host copy a CUDA graph cannot hold
        self._spread = torch.tensor(list(self.p.init_spread), device=self.device)

    # ------------------------------------------------------------------ the MIT sensor model
    def _sensor_table(self) -> torch.Tensor:
        """(n, n) P(observed r | expected d) on the `table_res` lattice, columns normalised -- the
        construction of `precompute_sensor_model` in the MIT filter, in metres."""
        p = self.p
        n = int(round(self.max_range / p.table_res)) + 1
        r = torch.arange(n, dtype=torch.float64)[:, None]          # observed, in cells
        d = torch.arange(n, dtype=torch.float64)[None, :]          # expected
        sig = p.sigma_hit / p.table_res
        hit = torch.exp(-((r - d) ** 2) / (2.0 * sig * sig)) / (sig * math.sqrt(2.0 * math.pi))
        hit = hit / hit.sum(0, keepdim=True)                       # theirs normalise the hit part first
        t = p.z_hit * hit
        short = torch.where((r < d) & (d > 0), 2.0 * (d - r) / d.clamp_min(1.0), torch.zeros_like(hit))
        t = t + p.z_short * short
        t = t + p.z_max * (r == n - 1).to(t.dtype)
        t = t + p.z_rand * (r < n - 1).to(t.dtype) / float(n - 1)
        t = t / t.sum(0, keepdim=True)
        return t.to(torch.float32).to(self.device)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, which: torch.Tensor, pose: torch.Tensor, odom: torch.Tensor) -> None:
        """Start the filters `which` (bool (R,)) from the operator's estimate around `pose` (R, 3),
        with the odometry pose (R, 3) the next deltas are taken from."""
        # No early exit on an all-false mask: that test was a host synchronise every control step
        # (the stack asks every step), and the masked writes below leave the other rows as they were.
        noise = torch.randn(self.R, self.N, 3, device=self.device) * self._spread
        init = pose[:, None, :] + noise
        init[..., 2] = wrap_angle(init[..., 2])
        m = which[:, None, None]
        self.particles = torch.where(m, init, self.particles)
        self.weights = torch.where(which[:, None], torch.full_like(self.weights, 1.0 / self.N), self.weights)
        self.estimate = torch.where(which[:, None], pose, self.estimate)
        self._odom_prev = torch.where(which[:, None], odom, self._odom_prev)
        self._ready = self._ready | which

    # ------------------------------------------------------------------ one scan
    @torch.no_grad()
    def step(self, odom: torch.Tensor, scan: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """Motion update from the odometry pose (R, 3) since the last call, sensor update from the
        scan (R, n_beams) metres, resample; returns the expected pose (R, 3)."""
        self._motion(odom)
        self._sensor(scan, tid)
        self._resample()
        return self.estimate

    def _motion(self, odom: torch.Tensor) -> None:
        prev = self._odom_prev
        c, s = torch.cos(prev[:, 2]), torch.sin(prev[:, 2])
        dxw, dyw = odom[:, 0] - prev[:, 0], odom[:, 1] - prev[:, 1]
        dx = c * dxw + s * dyw                                        # in the previous odom frame
        dy = -s * dxw + c * dyw
        dth = wrap_angle(odom[:, 2] - prev[:, 2])
        self._odom_prev = odom.clone()
        th = self.particles[..., 2]
        ct, st = torch.cos(th), torch.sin(th)
        x = self.particles[..., 0] + ct * dx[:, None] - st * dy[:, None]
        y = self.particles[..., 1] + st * dx[:, None] + ct * dy[:, None]
        th = th + dth[:, None]
        p = self.p
        # The MIT filter's dispersion, added in the world frame every update.
        x = x + torch.randn_like(x) * p.disp_x
        y = y + torch.randn_like(y) * p.disp_y
        th = wrap_angle(th + torch.randn_like(th) * p.disp_theta)
        self.particles = torch.stack([x, y, th], -1)

    def expected_ranges(self, poses: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """(R, N, K) ranges the static map gives at `poses` (R, N, 3), on the nominal scan plane."""
        R, N = poses.shape[:2]
        mx, my, mz = self.mount
        th = poses[..., 2]
        ct, st = torch.cos(th), torch.sin(th)
        ox = poses[..., 0] + mx * ct - my * st
        oy = poses[..., 1] + mx * st + my * ct
        a = th[..., None] + self.angles                                # (R, N, K)
        dh = torch.stack([torch.cos(a), torch.sin(a)], -1).reshape(R * N, self.K, 2)
        origin = torch.stack([ox, oy, torch.full_like(ox, mz)], -1)[:, :, None, :].expand(R, N, self.K, 3)
        origin = origin.reshape(R * N, self.K, 3).contiguous()
        k = torch.zeros(R * N, self.K, device=self.device)
        rmax = torch.full((R * N,), self.max_range, device=self.device)
        tid_n = tid[:, None].expand(R, N).reshape(-1)
        lidar = self.sim.lidar
        from .lidar_triton import HAVE_TRITON, trace3d_triton
        if HAVE_TRITON and origin.is_cuda:
            r, _ = trace3d_triton(origin, dh.contiguous(), k, rmax, lidar.track, tid_n, lidar.max_iters, None)
        else:
            r, _ = lidar._trace_torch(origin, dh, k, rmax, tid_n)
        return r.reshape(R, N, self.K)

    def _sensor(self, scan: torch.Tensor, tid: torch.Tensor) -> None:
        z = torch.nan_to_num(scan[:, self.beams].float(), nan=self.max_range, posinf=self.max_range,
                             neginf=0.0).clamp(0.0, self.max_range)                       # (R, K)
        d = self.expected_ranges(self.particles, tid).clamp(0.0, self.max_range)         # (R, N, K)
        n = self._table.shape[0]
        zi = (z / self.p.table_res).round().long().clamp(0, n - 1)[:, None, :].expand_as(d)
        di = (d / self.p.table_res).round().long().clamp(0, n - 1)
        like = self._table[zi, di]                                                       # (R, N, K)
        logw = torch.log(like.clamp_min(1e-30)).sum(-1) / self.p.squash
        logw = logw - logw.amax(1, keepdim=True)
        w = torch.exp(logw)
        self.weights = w / w.sum(1, keepdim=True)
        P = self.particles
        mx = (self.weights * P[..., 0]).sum(1)
        my = (self.weights * P[..., 1]).sum(1)
        mt = torch.atan2((self.weights * torch.sin(P[..., 2])).sum(1), (self.weights * torch.cos(P[..., 2])).sum(1))
        self.estimate = torch.stack([mx, my, mt], 1)

    def _resample(self) -> None:
        idx = torch.multinomial(self.weights, self.N, replacement=True)                # (R, N)
        self.particles = torch.gather(self.particles, 1, idx[..., None].expand(-1, -1, 3))
        self.weights = torch.full_like(self.weights, 1.0 / self.N)
