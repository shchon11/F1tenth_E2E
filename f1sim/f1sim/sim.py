"""Vectorized F1TENTH simulator core (ROS-independent).

One `step()` = one control period (default 25 ms = LiDAR period @ 40 Hz), internally
substepped at `physics_dt` (default 1 ms). All tensors live on `device`.

Action per env: (steer_angle [rad], speed [m/s]) -- the same command that goes to
/drive (AckermannDriveStamped) on the real car.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from . import dynamics as dyn
from .actuators import servo_target, vesc_accel
from . import imu as imu_model
from .lidar import Lidar
from .odom import VescOdom
from .params import Config
from .randomization import ParamSet
from .track import Track, TrackTensors


@dataclass
class StepResult:
    scan: torch.Tensor          # (B, N) noisy ranges (inf for no return)
    scan_true: torch.Tensor     # (B, N) noise-free ranges (privileged)
    scan_type: torch.Tensor     # (B, N) int32 what each beam hit: 0 none, 1 duct, 2 tall object, 3 floor
    attitude: torch.Tensor      # (B, 2) body roll, pitch [rad] (sprung mass)
    odom: torch.Tensor          # (B, 5) VESC odom: x, y, yaw, v, yaw_rate (drifting)
    state: torch.Tensor         # (B, 7) ground truth: x, y, yaw, vx, vy, yaw_rate, steer
    imu: Optional[torch.Tensor] # (B, K, 6) IMU samples this step: gyro xyz [rad/s], accel xyz [m/s^2] (None if disabled)
    imu_att: Optional[torch.Tensor]  # (B, 3) VESC attitude estimate roll, pitch, yaw [rad] (drifting yaw)
    collision: torch.Tensor     # (B,) bool
    progress: torch.Tensor      # (B,) meters advanced along centerline this step (signed)
    s: torch.Tensor             # (B,) arclength position on centerline
    lateral: torch.Tensor       # (B,) lateral offset from centerline (left +)
    lap: torch.Tensor           # (B,) completed laps since reset
    wall_dist: torch.Tensor     # (B,) min distance from footprint corners to wall
    t: float                    # sim time [s]
    car_collision: Optional[torch.Tensor] = None   # (B,) bool: contact with another car this step (races)


class Simulator:
    def __init__(self, track, cfg: Optional[Config] = None, num_envs: int = 1,
                 device: Optional[str] = None, track_ids=None, race_size: int = 1):
        """track: a Track or a list of Tracks. Each env drives on tracks[track_ids[env]]
        (default: round-robin). Track ids can be changed per env at reset().
        race_size M > 1: consecutive envs form races of M cars that share a track, see each other
        in their LiDAR (oriented boxes with randomized size and porosity) and collide with each other."""
        self.cfg = cfg or Config()
        self.device = torch.device(device or self.cfg.sim.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.B = num_envs
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.cfg.sim.seed)
        torch.manual_seed(self.cfg.sim.seed)

        self.track = TrackTensors(track, self.device)
        self.tracks = self.track.tracks
        if track_ids is None:
            track_ids = torch.arange(num_envs) % self.track.T
        self.tid = torch.as_tensor(track_ids, dtype=torch.long).to(self.device)
        self.params = ParamSet(self.cfg, num_envs, self.device, self.gen)
        self.P = self.params.P
        self.lidar = Lidar(self.track, self.cfg.lidar.n_beams, self.cfg.lidar.fov, self.device, compile=self.cfg.sim.compile)

        self.dt = self.cfg.sim.physics_dt
        self.control_dt = 1.0 / self.cfg.sim.control_rate
        self.substeps = max(1, int(round(self.control_dt / self.dt)))
        max_delay = 0.0
        for k, (lo, hi) in self.cfg.rand.ranges.items():
            if k == "actuator.cmd_delay":
                max_delay = max(max_delay, hi)
        max_delay = max(max_delay, self.cfg.actuator.cmd_delay + self.cfg.actuator.cmd_delay_jitter)
        # command history: index 0 = current control step, 1 = previous, ... (commands are
        # piecewise constant per control step, so latency is a lookup into this history)
        self.hist_len = int(math.ceil(max_delay / self.control_dt)) + 2
        self.cmd_hist = torch.zeros(num_envs, self.hist_len, 2, device=self.device)
        self._roll = self._roll_physics
        if self.device.type == "cuda" and self.cfg.sim.compile:
            try:
                self._roll = torch.compile(self._roll_physics, dynamic=False)
            except Exception:
                self._roll = self._roll_physics

        self.state = torch.zeros(num_envs, dyn.STATE_DIM, device=self.device)
        self.ax = torch.zeros(num_envs, device=self.device)
        self.ay = torch.zeros(num_envs, device=self.device)
        self.pose_prev = torch.zeros(num_envs, 3, device=self.device)
        self.att = torch.zeros(num_envs, 4, device=self.device)          # roll, roll_rate, pitch, pitch_rate
        self.att_prev = torch.zeros(num_envs, 2, device=self.device)
        self.imu_state = imu_model.imu_state_init(num_envs, self.device)
        self.imu_idx = imu_model.sample_indices(self.cfg.imu.imu_rate, self.control_dt, self.dt, self.substeps)
        self.imu_ts = 1.0 / self.cfg.imu.imu_rate
        self.cmd = torch.zeros(num_envs, 2, device=self.device)        # last commanded (steer, speed)
        self.odom = VescOdom(num_envs, self.device)
        self.s = torch.zeros(num_envs, device=self.device)
        self.lap = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.collided = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.t = 0.0
        # footprint corner offsets in body frame relative to CoG
        vp = self.cfg.vehicle
        fx = vp.length / 2 + (vp.lf - vp.lr) / 2 * 0     # keep symmetric about CoG for simplicity
        self.corners = torch.tensor([[fx, vp.width / 2], [fx, -vp.width / 2],
                                     [-fx, vp.width / 2], [-fx, -vp.width / 2]], device=self.device)
        # races: env i is car (i % M) of race (i // M); other_idx lists the M-1 cars it shares the track with
        self.M = int(race_size)
        assert num_envs % self.M == 0, "num_envs must be a multiple of race_size"
        ar = torch.arange(num_envs, device=self.device); g, j = ar // self.M, ar % self.M
        self.other_idx = torch.stack([g * self.M + (j + 1 + c) % self.M for c in range(self.M - 1)], 1) if self.M > 1 else None
        self.car_dims = torch.zeros(num_envs, 3, device=self.device)      # LiDAR silhouette: length, width, height
        self.car_porosity = torch.zeros(num_envs, device=self.device)     # fraction of beams a car body swallows
        # competition rule: a small detection box on the rear bumper so the LiDAR of the car behind gets
        # a solid return: depth, width, bottom, top above the floor (+ its own small porosity)
        self.car_rear = torch.zeros(num_envs, 4, device=self.device)
        self.car_rear_porosity = torch.zeros(num_envs, device=self.device)
        self.car_collision = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._resample_car_dims(ar)
        self.reset()

    def _resample_car_dims(self, ids: torch.Tensor):
        n = ids.numel()
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=self.device, generator=self.gen)
        self.car_dims[ids] = torch.stack([u(0.45, 0.58), u(0.26, 0.34), u(0.15, 0.27)], 1)
        self.car_porosity[ids] = u(0.05, 0.30)
        # a small cardboard box strapped to the rear bumper at LiDAR height: depth, width, z bottom, z top
        self.car_rear[ids] = torch.stack([u(0.06, 0.11), u(0.13, 0.22), u(0.08, 0.12), u(0.22, 0.30)], 1)
        self.car_rear_porosity[ids] = u(0.0, 0.05)

    # what a LiDAR at ~15 cm actually sees of an F1TENTH car: (x, y offsets from the CoG, length,
    # width, z bottom, z top, porosity). The chassis and wheels sit below a level scan plane and only
    # show up when the plane tilts; the electronics deck and the rule-mandated rear box are what the
    # car behind normally gets returns from.
    CAR_PARTS = (
        (0.00, 0.000, 0.36, 0.20, 0.03, 0.12, 0.15),     # chassis plate, battery, cabling (open frame)
        (0.10, 0.000, 0.18, 0.14, 0.12, 0.24, 0.10),     # electronics deck + LiDAR tower
        (0.165, 0.120, 0.11, 0.045, 0.00, 0.11, 0.35),   # wheels (rubber: weak grazing returns)
        (0.165, -0.120, 0.11, 0.045, 0.00, 0.11, 0.35),
        (-0.165, 0.120, 0.11, 0.045, 0.00, 0.11, 0.35),
        (-0.165, -0.120, 0.11, 0.045, 0.00, 0.11, 0.35),
    )

    def _car_boxes(self, state: torch.Tensor):
        """Box sets the LiDAR of each env sees: the other cars' parts and their rear detection boxes."""
        o = self.other_idx
        st = state[o]                                                     # (B,C,7)
        yaw = st[:, :, 2]; c, s_ = torch.cos(yaw), torch.sin(yaw)
        scale = self.car_dims[o][..., 0] / 0.50                           # bigger / smaller builds
        sets = []
        for (ox, oy, lx, ly, zlo, zhi, poro) in self.CAR_PARTS:
            ox_, oy_ = ox * scale, oy * scale
            pos = torch.stack([st[:, :, 0] + ox_ * c - oy_ * s_, st[:, :, 1] + ox_ * s_ + oy_ * c, yaw], -1)
            dims = torch.stack([lx * scale, ly * scale, torch.full_like(scale, zhi)], -1)
            zr = torch.stack([torch.full_like(scale, zlo), torch.full_like(scale, zhi)], -1)
            sets.append((pos, dims, zr, self.car_porosity * (poro / 0.175)))
        rear = self.car_rear[o]                                           # (B,C,4) depth, width, z_lo, z_hi
        back = 0.5 * self.car_dims[o][..., 0] + 0.5 * rear[..., 0]        # box centre behind the body
        pos = torch.stack([st[:, :, 0] - back * c, st[:, :, 1] - back * s_, yaw], -1)
        sets.append((pos, torch.stack([rear[..., 0], rear[..., 1], rear[..., 3]], -1), rear[..., 2:4], self.car_rear_porosity))
        return sets

    def _car_contacts(self, state: torch.Tensor) -> torch.Tensor:
        """Separating-axis test between each car's footprint and the other cars of its race -> (B,) bool."""
        pts = self._footprint_corners(state)                              # (B,4,2)
        yaw = state[:, dyn.IYAW]
        ext = torch.stack([-self.car_rear[:, 0] * torch.cos(yaw), -self.car_rear[:, 0] * torch.sin(yaw)], 1)
        pts = torch.cat([pts, pts[:, 2:] + ext[:, None, :]], 1)         # + rear box corners (B,6,2)
        ax = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1); ay = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
        hit = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
        for c in range(self.M - 1):
            o = self.other_idx[:, c]
            po = pts[o]
            sep = torch.zeros_like(hit)
            for axis in (ax, ay, ax[o], ay[o]):
                pa = (pts * axis[:, None, :]).sum(-1); pb = (po * axis[:, None, :]).sum(-1)
                sep = sep | (pa.max(1).values < pb.min(1).values) | (pb.max(1).values < pa.min(1).values)
            hit = hit | ~sep
        return hit

    # ------------------------------------------------------------------ reset
    def sample_spawn(self, n: int, lateral_std: float = 0.3, yaw_std: float = 0.2,
                     s: Optional[torch.Tensor] = None, tid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Random poses along the centerline of each env's track (tid (n,), default track 0). Returns (n, 3)."""
        tid = torch.zeros(n, dtype=torch.long, device=self.device) if tid is None else tid.to(self.device)
        no_cl = ~self.track.cl_ok[tid] if self.track.cl is not None else torch.ones(n, dtype=torch.bool, device=self.device)
        if bool(no_cl.any()):
            # no centerline on that track: spawn at its most open free cell, random heading
            poses = torch.zeros(n, 3, device=self.device)
            for i, t in enumerate(tid.tolist()):
                if not no_cl[i]:
                    continue
                tr = self.tracks[t]
                idx = int(tr.edt.argmax()); row, col = idx // tr.occupancy.shape[1], idx % tr.occupancy.shape[1]
                poses[i, 0] = tr.origin[0] + col * tr.resolution; poses[i, 1] = tr.origin[1] + row * tr.resolution
            poses[:, 2] = torch.rand(n, device=self.device, generator=self.gen) * 2 * math.pi * (yaw_std > 0)
            if bool(no_cl.all()):
                return poses
            rest = self.sample_spawn(int((~no_cl).sum()), lateral_std, yaw_std, None if s is None else s[~no_cl], tid[~no_cl])
            poses[~no_cl] = rest
            return poses
        if s is None:
            s = torch.rand(n, device=self.device, generator=self.gen) * self.track.length[tid]
        s = s.to(self.device)
        xy, yaw = self.track.pose_at_s(s, tid)
        lat = torch.randn(n, device=self.device, generator=self.gen) * lateral_std
        nrm = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
        xy = xy + nrm * lat[:, None]
        yaw = yaw + torch.randn(n, device=self.device, generator=self.gen) * yaw_std
        pose = torch.cat([xy, yaw[:, None]], 1)
        # reject poses too close to walls by pulling them back to the centerline
        bad = self.track.sample_edt(xy, tid) < self.cfg.vehicle.width
        if bad.any():
            xy0, yaw0 = self.track.pose_at_s(s[bad], tid[bad])
            pose[bad] = torch.cat([xy0, yaw0[:, None]], 1)
        return pose

    def reset(self, env_ids: Optional[torch.Tensor] = None, poses: Optional[torch.Tensor] = None,
              speed: Optional[torch.Tensor] = None, track_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.B, device=self.device)
        n = env_ids.numel()
        if n == 0:
            return
        if track_ids is not None:
            self.tid[env_ids] = torch.as_tensor(track_ids, dtype=torch.long).to(self.device)
        if poses is None:
            poses = self.sample_spawn(n, tid=self.tid[env_ids])
        poses = poses.to(self.device)
        self.params.resample(env_ids)
        st = torch.zeros(n, dyn.STATE_DIM, device=self.device)
        st[:, :3] = poses
        if speed is not None:
            st[:, dyn.IVX] = speed
        self.state[env_ids] = st
        self.ax[env_ids] = 0.0
        self.ay[env_ids] = 0.0
        self.pose_prev[env_ids] = poses
        self.att[env_ids] = 0.0
        self.att_prev[env_ids] = 0.0
        self.imu_state[env_ids] = 0.0
        self.cmd[env_ids] = 0.0
        if speed is not None:
            self.cmd[env_ids, 1] = speed
        self.cmd_hist[env_ids] = self.cmd[env_ids][:, None, :]
        self.odom.reset(env_ids, poses)
        if self.track.cl is not None:
            s, _, _ = self.track.project(poses[:, :2], self.tid[env_ids])
            self.s[env_ids] = s
        self.lap[env_ids] = 0
        self.collided[env_ids] = False
        self.car_collision[env_ids] = False
        self.steps[env_ids] = 0
        self._resample_car_dims(env_ids)

    # ------------------------------------------------------------------ warm-up
    def warmup(self):
        """Run one throw-away step to trigger torch.compile / Triton JIT (10-20 s on first use),
        then restore the state. Call this before opening a window or starting a real-time loop."""
        keep = {k: getattr(self, k).clone() for k in ("state", "ax", "ay", "att", "att_prev", "pose_prev", "cmd",
                                                     "cmd_hist", "s", "lap", "collided", "steps", "imu_state")}
        odom, t = self.odom.state.clone(), self.t
        self.step(torch.zeros(self.B, 2, device=self.device))
        for k, v in keep.items():
            getattr(self, k).copy_(v)
        self.odom.state.copy_(odom); self.t = t
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor) -> StepResult:
        """action (B, 2): steer angle [rad], speed [m/s]. Envs flagged collided (when
        terminate_on_collision) are frozen until reset."""
        P = self.P
        action = action.to(self.device)
        self.cmd = torch.stack([action[:, 0].clamp(-self.cfg.vehicle.s_max, self.cfg.vehicle.s_max),
                                action[:, 1]], 1)
        delay_s = P["cmd_delay"]
        if self.cfg.actuator.cmd_delay_jitter > 0:
            delay_s = delay_s + torch.rand(self.B, device=self.device, generator=self.gen) * self.cfg.actuator.cmd_delay_jitter
        frozen = self.collided & self.cfg.sim.terminate_on_collision
        self.cmd_hist = torch.cat([self.cmd[:, None, :], self.cmd_hist[:, :-1]], 1)

        self.pose_prev = self.state[:, :3].clone()
        self.att_prev = self.att[:, [0, 2]].clone()
        state, ax, ay, att, imu_state, imu_samples = self._roll(self.state, self.ax, self.att, self.imu_state,
                                                                self.cmd_hist, delay_s, P)
        state = torch.where(frozen[:, None], self.state, state)
        att = torch.where(frozen[:, None], self.att, att)
        imu_state = torch.where(frozen[:, None], self.imu_state, imu_state)
        self.state, self.ax, self.ay, self.att, self.imu_state = state, ax, ay, att, imu_state
        self.t += self.control_dt
        self.steps += 1

        # --- collision check on footprint ---
        wall_dist = self._footprint_clearance(state)
        hit = wall_dist <= 0.0
        cars = None
        if self.M > 1:
            self.car_collision = self._car_contacts(state)
            hit = hit | self.car_collision
            cars = self._car_boxes(state)
        self.collided = self.collided | hit

        # --- sensors ---
        pose = state[:, :3]
        scan, scan_true, scan_type = self.lidar.scan(pose, self.pose_prev, P, self.cfg.lidar.motion_distortion,
                                                     att=att[:, [0, 2]], att_prev=self.att_prev, tid=self.tid, cars=cars)
        odom = self.odom.update(state[:, dyn.IVX], self.cmd[:, 0], P, self.control_dt)
        imu = imu_samples if self.cfg.imu.enabled else None
        imu_att = imu_state[:, 18:21].clone() if self.cfg.imu.enabled else None

        # --- progress ---
        if self.track.cl is not None:
            s, lateral, _ = self.track.project(pose[:, :2], self.tid)
            ds = s - self.s
            L = self.track.length[self.tid]
            ds = torch.where(ds > L / 2, ds - L, torch.where(ds < -L / 2, ds + L, ds))
            new_s = self.s + ds
            self.lap += ((new_s >= L) & (ds > 0)).long() - ((new_s < 0) & (ds < 0)).long()
            self.s = s
        else:
            s = torch.zeros(self.B, device=self.device); ds = s; lateral = s
        return StepResult(scan, scan_true, scan_type, att[:, [0, 2]].clone(), odom, state, imu, imu_att,
                          self.collided.clone(), ds, s, lateral, self.lap.clone(), wall_dist, self.t,
                          self.car_collision.clone() if self.M > 1 else None)

    def _roll_physics(self, state, ax, att, imu_state, cmd_hist, delay_s, P):
        """All substeps of one control period (compiled on CUDA)."""
        ay = torch.zeros_like(ax)
        roll, roll_rate, pitch, pitch_rate = att[:, 0], att[:, 1], att[:, 2], att[:, 3]
        wn, zeta = P["susp_wn"], P["susp_zeta"]
        imu_on = self.cfg.imu.enabled
        r_vec = torch.stack([P["imu_x"] - P["lr"], P["imu_y"], P["imu_z"] - P["h"]], 1)
        samples = []
        soft_wall = not self.cfg.sim.terminate_on_collision
        ar = torch.arange(self.B, device=state.device)
        for k in range(self.substeps):
            age = delay_s - (k + 1) * self.dt
            idx = torch.ceil(age / self.control_dt).long().clamp(0, self.hist_len - 1)
            eff = cmd_hist[ar, idx]                                   # (B,2) command in effect
            steer_tgt = servo_target(eff[:, 0], P)
            a_cmd = vesc_accel(eff[:, 1], state[:, dyn.IVX], P)
            r_old = state[:, dyn.IR]
            state, ax, ay = dyn.step_dynamics(state, steer_tgt, a_cmd, ax, P, P["servo_tau"], self.dt)
            # sprung mass: roll to the outside of the corner, dive under braking, squat under throttle
            roll_ss = P["roll_per_g"] * ay / dyn.G
            pitch_ss = -P["pitch_per_g"] * ax / dyn.G
            roll_acc = wn * wn * (roll_ss - roll) - 2 * zeta * wn * roll_rate
            pitch_acc = wn * wn * (pitch_ss - pitch) - 2 * zeta * wn * pitch_rate
            roll_rate = roll_rate + roll_acc * self.dt
            pitch_rate = pitch_rate + pitch_acc * self.dt
            roll = (roll + roll_rate * self.dt).clamp(-0.35, 0.35)
            pitch = (pitch + pitch_rate * self.dt).clamp(-0.35, 0.35)
            if imu_on:
                yaw_acc = (state[:, dyn.IR] - r_old) / self.dt
                imu_state = imu_model.substep(imu_state, ax, ay, state[:, dyn.IVX], roll, pitch, roll_rate, pitch_rate,
                                              state[:, dyn.IR], roll_acc, pitch_acc, yaw_acc, r_vec, P, self.dt)
                if k in self.imu_idx:
                    smp, imu_state = imu_model.sample(imu_state, P, self.imu_ts)
                    samples.append(smp)
            if soft_wall:
                state = self._resolve_wall_contact(state)
        imu_samples = torch.stack(samples, 1) if samples else torch.zeros(state.shape[0], 0, 6, device=state.device)
        return state, ax, ay, torch.stack([roll, roll_rate, pitch, pitch_rate], 1), imu_state, imu_samples

    # ------------------------------------------------------------------ helpers
    def _footprint_corners(self, state: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(state[:, dyn.IYAW]), torch.sin(state[:, dyn.IYAW])
        R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)   # (B,2,2)
        pts = torch.einsum("bij,kj->bki", R, self.corners) + state[:, None, :2]  # (B,4,2)
        return pts

    def _footprint_clearance(self, state: torch.Tensor) -> torch.Tensor:
        pts = self._footprint_corners(state)
        d = self.track.sample_edt(pts, self.tid[:, None])              # (B,4)
        # sample_edt is distance to wall cell center; subtract half a cell for the surface
        return d.min(1).values - 0.5 * self.track.t_res[self.tid]

    def _resolve_wall_contact(self, state: torch.Tensor) -> torch.Tensor:
        """Soft wall: push CoG out along EDT gradient and kill the into-wall velocity component."""
        clr = self._footprint_clearance(state)
        pen = (-clr).clamp_min(0.0)
        touching = pen > 0
        n = self.track.edt_gradient(state[:, :2], self.tid)            # away from wall (world)
        yaw = state[:, dyn.IYAW]
        c, s = torch.cos(yaw), torch.sin(yaw)
        vwx = state[:, dyn.IVX] * c - state[:, dyn.IVY] * s
        vwy = state[:, dyn.IVX] * s + state[:, dyn.IVY] * c
        vn = vwx * n[:, 0] + vwy * n[:, 1]
        into = (vn < 0) & touching
        k = self.cfg.sim.collision_restitution
        f = self.cfg.sim.wall_friction
        vwx2 = vwx - (1 + k) * vn * n[:, 0]
        vwy2 = vwy - (1 + k) * vn * n[:, 1]
        vwx2, vwy2 = vwx2 * (1 - f * 0.05), vwy2 * (1 - f * 0.05)
        vwx = torch.where(into, vwx2, vwx); vwy = torch.where(into, vwy2, vwy)
        st = state.clone()
        st[:, dyn.IVX] = vwx * c + vwy * s
        st[:, dyn.IVY] = -vwx * s + vwy * c
        st[:, :2] = st[:, :2] + n * (pen * touching.float())[:, None]
        st[:, dyn.IR] = torch.where(touching, st[:, dyn.IR] * 0.9, st[:, dyn.IR])
        return st

    # ------------------------------------------------------------------ introspection
    def scan_meta(self) -> Dict[str, float]:
        lp = self.cfg.lidar
        return dict(angle_min=-lp.fov / 2, angle_max=lp.fov / 2, angle_increment=self.lidar.angle_increment,
                    range_min=lp.range_min, range_max=lp.range_max, scan_time=1.0 / lp.rate,
                    time_increment=(lp.fov / (2 * math.pi)) / lp.rate / (lp.n_beams - 1))
