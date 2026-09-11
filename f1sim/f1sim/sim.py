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
                                # K varies between steps: the sensor runs on its own clock, so a
                                # 50 Hz IMU on a 40 Hz loop delivers 1, 1, 1, 2, ... per step
    imu_att: Optional[torch.Tensor]  # (B, 3) VESC attitude estimate roll, pitch, yaw [rad] (drifting yaw)
    collision: torch.Tensor     # (B,) bool
    progress: torch.Tensor      # (B,) meters advanced along centerline this step (signed)
    s: torch.Tensor             # (B,) arclength position on centerline
    lateral: torch.Tensor       # (B,) lateral offset from centerline (left +)
    lap: torch.Tensor           # (B,) completed laps since reset
    wall_dist: torch.Tensor     # (B,) min distance from footprint corners to wall
    t: float                    # sim time [s]
    car_collision: Optional[torch.Tensor] = None   # (B,) bool: contact with another car this step (races)
    imu_offsets: Optional[torch.Tensor] = None     # (K,) seconds before the END of this control step
                                                   # that each IMU sample was taken. Timestamp a
                                                   # sample as `step_end - imu_offsets[k]`; the last
                                                   # one is NOT generally at the step boundary.


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
        self.lidar = Lidar(self.track, self.cfg.lidar.n_beams, self.cfg.lidar.fov, self.device, compile=self.cfg.sim.compile,
                           post_mode=self.cfg.sim.compile_mode)

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
        self._roll = self._roll_physics; self._post = self._post_roll
        if self.device.type == "cuda" and self.cfg.sim.compile:
            self._roll = self._guarded(torch.compile(self._roll_physics, dynamic=False, mode=self.cfg.sim.compile_mode), "_roll", self._roll_physics)
            if self.cfg.sim.compile_mode == "reduce-overhead":
                self._post = self._guarded(torch.compile(self._post_roll, dynamic=False, mode="reduce-overhead"), "_post", self._post_roll)

        self.state = torch.zeros(num_envs, dyn.STATE_DIM, device=self.device)
        self.ax = torch.zeros(num_envs, device=self.device)
        self.ay = torch.zeros(num_envs, device=self.device)
        self.pose_prev = torch.zeros(num_envs, 3, device=self.device)
        self.att = torch.zeros(num_envs, 4, device=self.device)          # roll, roll_rate, pitch, pitch_rate
        self.att_prev = torch.zeros(num_envs, 2, device=self.device)
        self.imu_state = imu_model.imu_state_init(num_envs, self.device)
        # The IMU runs on its own clock. 50 Hz against a 40 Hz control loop does not divide, so the
        # number of samples in a control step cycles 1, 1, 1, 2 rather than being constant; the
        # schedule below is that cycle, and `imu_offsets_sched` says how long before each step's end
        # every sample falls so consumers can timestamp them. `imu_ts` is the sensor's own period,
        # which is now what the samples are actually spaced by -- `sample()` uses it for the gyro
        # bias random walk (sqrt(ts)), for integrating the gyro into the attitude estimate and for
        # the complementary-filter gain (ts / tau).
        self.imu_schedule, self.imu_offsets_sched, self.imu_period = imu_model.sample_schedule(
            self.cfg.imu.imu_rate, self.control_dt, self.dt, self.substeps)
        self._imu_phase = 0
        self.imu_ts = 1.0 / self.cfg.imu.imu_rate
        self._imu_offsets = [torch.tensor(o, device=self.device) for o in self.imu_offsets_sched]
        self.cmd = torch.zeros(num_envs, 2, device=self.device)        # last commanded (steer, speed)
        self.odom = VescOdom(num_envs, self.device)
        self.s = torch.zeros(num_envs, device=self.device)
        self.cl_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)   # last centerline index:
                                                                                    # anchors the windowed
                                                                                    # projection (see
                                                                                    # Track.project)
        self.lap = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.collided = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        #: Prop contact seen inside a substep, latched until `step` reads it. A box can be entered
        #: and pushed back out between two control steps, and a collision that the physics resolved
        #: but nothing reported is a collision the caller never learns about.
        self.prop_touched = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
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

    @property
    def imu_idx(self):
        """Substep indices this control step emits at (the current phase of the sample schedule)."""
        return self.imu_schedule[self._imu_phase]

    @property
    def imu_rate_delivered(self) -> float:
        """[Hz] the rate IMU samples come out at, averaged over one schedule cycle.

        Equal to `cfg.imu.imu_rate` by construction now that the schedule is phase-exact; kept as
        the thing consumers should read, so a future rate that cannot be scheduled exactly is
        visible rather than assumed.
        """
        return sum(len(idx) for idx in self.imu_schedule) / (self.imu_period * self.control_dt)

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
        (0.00, 0.000, 0.36, 0.20, 0.03, 0.05, 0.15),     # chassis plate, battery, cabling (open frame)  [EXPERIMENT z_hi 0.05]
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
        if getattr(self.cfg.lidar, "car_model", "mesh") == "mesh":
            sets.append(("mesh", st[:, :, :3].contiguous(), scale, self.car_porosity))
        for (ox, oy, lx, ly, zlo, zhi, poro) in ([] if sets else self.CAR_PARTS):
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
                     s: Optional[torch.Tensor] = None, tid: Optional[torch.Tensor] = None, min_clearance: Optional[float] = None) -> torch.Tensor:
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
        bad = self.track.sample_edt(xy, tid) < (self.cfg.vehicle.width if min_clearance is None else min_clearance)
        if getattr(self.track, "has_props", False):
            # Props are not in the EDT, so the wall test above cannot see them and a car would spawn
            # standing inside a crate. Reject on the same geometry the contact test uses; if the
            # centerline fallback is itself blocked, nudge along the lane until it is not.
            bad = bad | self._spawn_in_prop(pose, tid)
            if bad.any():
                xy0, yaw0 = self.track.pose_at_s(s[bad], tid[bad])
                pose[bad] = torch.cat([xy0, yaw0[:, None]], 1)
                still = bad & self._spawn_in_prop(pose, tid)   # the centerline point may be blocked too
                # Walk a full lap in even steps rather than a few short nudges. Stopping early and
                # returning the pose anyway would spawn the car inside a crate and call it a spawn;
                # if a whole lap has nowhere to stand, that is a broken map and it says so.
                L = self.track.length[tid].clamp_min(1e-6)
                for j in range(1, 33):
                    idx = torch.nonzero(still, as_tuple=True)[0]
                    if not idx.numel():
                        break
                    s_alt = (s[idx] + L[idx] * (j / 33.0)) % L[idx]
                    xy_a, yaw_a = self.track.pose_at_s(s_alt, tid[idx])
                    p_alt = torch.cat([xy_a, yaw_a[:, None]], 1)
                    ok = ~self._spawn_in_prop(p_alt, tid[idx])
                    pose[idx[ok]] = p_alt[ok]
                    still[idx[ok]] = False
                if bool(still.any()):
                    raise RuntimeError(
                        f"{int(still.sum())} env(s) have no spawn pose clear of the placed props "
                        f"anywhere on their lap; the track's props block the centerline")
            return pose
        if bad.any():
            xy0, yaw0 = self.track.pose_at_s(s[bad], tid[bad])
            pose[bad] = torch.cat([xy0, yaw0[:, None]], 1)
        return pose

    def _spawn_in_prop(self, pose: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """Whether each candidate spawn pose has its footprint overlapping a prop."""
        from .prop_math import prism_contacts
        c, s_ = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        R = torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2)
        pts = torch.einsum("bij,kj->bki", R, self.corners) + pose[:, None, :2]
        poses, pn, pd, z_lo, z_hi = self.track.props_for(tid)
        h = self.car_dims[:pose.shape[0], 2] if self.car_dims.shape[0] >= pose.shape[0] else 0.25
        depth, _, _ = prism_contacts(pts[:, [0, 1, 3, 2]], poses, pn, pd, z_lo, z_hi, h)
        return depth > 0

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
            s, _, i0 = self.track.project(poses[:, :2], self.tid[env_ids])   # global: no history yet
            self.s[env_ids] = s; self.cl_idx[env_ids] = i0
        self.lap[env_ids] = 0
        self.collided[env_ids] = False
        self.car_collision[env_ids] = False
        self.prop_touched[env_ids] = False
        self.steps[env_ids] = 0
        self._resample_car_dims(env_ids)

    # ------------------------------------------------------------------ warm-up
    def warmup(self):
        """Run throw-away steps to trigger torch.compile / Triton JIT (10-20 s on first use), then
        restore the state. Call this before opening a window or starting a real-time loop.

        One step per IMU schedule phase, because each phase emits a different number of samples at
        different substeps and so compiles to its own graph -- warming only the first would leave
        the rest to compile inside the real-time loop.

        Everything the steps touch has to be put back, including the sensor phase: leaving it
        advanced would slide the IMU's sample grid against `t` for the rest of the run.
        """
        keep = {k: getattr(self, k).clone() for k in ("state", "ax", "ay", "att", "att_prev", "pose_prev", "cmd",
                                                     "cmd_hist", "s", "lap", "collided", "steps", "imu_state",
                                                     "cl_idx", "car_collision", "prop_touched")}
        odom, t, phase = self.odom.state.clone(), self.t, self._imu_phase
        # The throw-away steps consume randomness from two places, not one: `self.gen` (spawn draws,
        # command-delay jitter) and the GLOBAL torch generator, which is what imu.sample's
        # torch.randn and the lidar/odom noise's *_like calls use. Both are rewound, so a seeded run
        # does not depend on whether warmup was called.
        #
        # Verified on CPU by tests/test_sim_audit.py. The CUDA global stream is restored the same
        # way but is NOT verified here -- no seeded-equivalence claim is made for CUDA until the
        # cached-CUDA run checks it.
        rng_local = self.gen.get_state()
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
        for _ in range(self.imu_period):
            self.step(torch.zeros(self.B, 2, device=self.device))
        for k, v in keep.items():
            getattr(self, k).copy_(v)
        self.odom.state.copy_(odom); self.t = t; self._imu_phase = phase
        self.gen.set_state(rng_local)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
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
        # the sample layout belongs to this step's phase; capture it before the phase advances
        imu_offsets = self._imu_offsets[self._imu_phase]
        state, ax, ay, att, imu_state, imu_samples = self._roll(self.state, self.ax, self.att, self.imu_state,
                                                                self.cmd_hist, delay_s, P)
        self._imu_phase = (self._imu_phase + 1) % self.imu_period
        if self.cfg.sim.compile_mode == "reduce-overhead":      # CUDA graphs reuse their output buffers: keep copies
            state, ax, ay, att, imu_state, imu_samples = (t.clone() for t in (state, ax, ay, att, imu_state, imu_samples))
        state = torch.where(frozen[:, None], self.state, state)
        att = torch.where(frozen[:, None], self.att, att)
        imu_state = torch.where(frozen[:, None], self.imu_state, imu_state)
        self.state, self.ax, self.ay, self.att, self.imu_state = state, ax, ay, att, imu_state
        self.t += self.control_dt
        self.steps += 1

        # --- collision check on footprint ---
        cars = None
        if self.M > 1:
            self.car_collision = self._car_contacts(state)
            cars = self._car_boxes(state)
        wall_dist, hit, odom_state, s, ds, lateral, lap, cl_idx = (
            t.clone() for t in self._post(state, self.odom.state, self.cmd, self.s, self.lap, self.tid, P, self.cl_idx))
        self.cl_idx = cl_idx
        if self.M > 1:
            hit = hit | self.car_collision
        if getattr(self.track, "has_props", False):
            # `_post` reads the occupancy EDT and props are not in it, so without this a car drives
            # through a box: `terminate_on_collision` never fires, and with soft walls the contact is
            # resolved by `_resolve_wall_contact` yet never reported. `prop_touched` also carries the
            # substeps, because a box can be entered and pushed back out inside one control step.
            hit = hit | self.prop_touched | (self._prop_contact(state)[0] > 0)
        self.prop_touched.zero_()          # in place: `warmup` restores by copy_ into this tensor
        self.collided = self.collided | hit
        self.odom.state = odom_state; self.s = s; self.lap = lap
        odom = odom_state

        # --- sensors ---
        pose = state[:, :3]
        scan, scan_true, scan_type = self.lidar.scan(pose, self.pose_prev, P, self.cfg.lidar.motion_distortion,
                                                     att=att[:, [0, 2]], att_prev=self.att_prev, tid=self.tid, cars=cars)
        imu = imu_samples if self.cfg.imu.enabled else None
        imu_att = imu_state[:, 18:21].clone() if self.cfg.imu.enabled else None
        return StepResult(scan, scan_true, scan_type, att[:, [0, 2]].clone(), odom, state, imu, imu_att,
                          self.collided.clone(), ds, s, lateral, self.lap.clone(), wall_dist, self.t,
                          self.car_collision.clone() if self.M > 1 else None,
                          imu_offsets if self.cfg.imu.enabled else None)

    def _guarded(self, compiled, attr, eager):
        """Compiled callable that falls back to eager for good if inductor/Triton fails at first use."""
        def f(*args, **kw):
            try:
                return compiled(*args, **kw)
            except Exception:
                setattr(self, attr, eager)
                return eager(*args, **kw)
        return f

    def _post_roll(self, state, odom_state, cmd, s_prev, lap_prev, tid, P, cl_idx):
        """Footprint clearance, VESC odometry and lane progress for one control step (compiled)."""
        wall_dist = self._footprint_clearance(state)
        hit = wall_dist <= 0.0
        odom_state = self.odom.update_pure(odom_state, state[:, dyn.IVX], cmd[:, 0], P, self.control_dt)
        if self.track.cl is not None:
            s, lateral, cl_idx = self.track.project(state[:, :2], tid, prev_idx=cl_idx)
            ds = s - s_prev
            L = self.track.length[tid]
            ds = torch.where(ds > L / 2, ds - L, torch.where(ds < -L / 2, ds + L, ds))
            new_s = s_prev + ds
            lap = lap_prev + ((new_s >= L) & (ds > 0)).long() - ((new_s < 0) & (ds < 0)).long()
        else:
            s = torch.zeros_like(s_prev); ds = s; lateral = s; lap = lap_prev
        return wall_dist, hit, odom_state, s, ds, lateral, lap, cl_idx

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

    def _prop_contact(self, state: torch.Tensor):
        """Penetration depth and outward normal against the placed props: (pen (B,), n (B,2)).

        Deliberately not routed through `_footprint_clearance`. That test samples the EDT at the
        car's four corners, which cannot see an obstacle that fits between them: a 5 cm marker post
        standing under the middle of the car leaves all four corners clear, so the car drives through
        it. SAT over both polygons' edge normals finds containment as readily as edge crossing.

        The normal has to come from the contacted prop as well. `edt_gradient` reads the occupancy
        grid, and props are not in the grid -- it would return the direction away from the nearest
        *wall*, which is unrelated to the box the car is actually touching."""
        from .prop_math import prism_contacts
        tr = self.track
        poses, pn, pd, z_lo, z_hi = tr.props_for(self.tid)
        # `self.corners` is [front-left, front-right, rear-left, rear-right] -- a bow tie, not a
        # perimeter. SAT takes edge normals from consecutive pairs, so feeding it that order hands it
        # the rectangle's two diagonals as separating axes and misses the axes that matter. [0,1,3,2]
        # walks the perimeter. The stored order is left as it is because `_footprint_clearance` and
        # `_car_boxes` index it, and reordering it there would be a change to code that is not mine.
        # `[:, [0, 1, 3, 2]]` would say the same thing, but advanced indexing with a Python list
        # builds the index tensor from host memory, and that host-to-device copy invalidates a CUDA
        # graph capture (`cudaErrorStreamCaptureInvalidated`). Same four corners, same perimeter
        # order, no host data.
        fl, fr, rl, rr = self._footprint_corners(state).unbind(1)
        corners = torch.stack((fl, fr, rr, rl), 1)
        # Body height, not CoG height: `vehicle.h` is where the mass sits (0.074 m) and doubling it
        # is not a silhouette. `car_dims[:, 2]` is the height the LiDAR already uses for this car.
        depth, normal, _ = prism_contacts(corners, poses, pn, pd, z_lo, z_hi, self.car_dims[:, 2])
        return depth.clamp_min(0.0), normal

    def _resolve_wall_contact(self, state: torch.Tensor) -> torch.Tensor:
        """Soft wall: push CoG out along EDT gradient and kill the into-wall velocity component."""
        clr = self._footprint_clearance(state)
        pen = (-clr).clamp_min(0.0)
        touching = pen > 0
        n = self.track.edt_gradient(state[:, :2], self.tid)            # away from wall (world)
        if getattr(self.track, "has_props", False):
            # Whichever of wall or prop is deeper owns this step's normal. Blending two normals
            # would push the car somewhere neither contact asks for.
            pen_p, n_p = self._prop_contact(state)
            self.prop_touched = self.prop_touched | (pen_p > 0)   # substeps, read once per step
            prop_deeper = pen_p > pen
            pen = torch.where(prop_deeper, pen_p, pen)
            n = torch.where(prop_deeper[:, None], n_p.to(n.dtype), n)
            touching = pen > 0
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
        """Header fields for the scan this simulator actually produces.

        The timing is derived from `control_dt`, not from `cfg.lidar.rate`, because that is what the
        simulator does: `step` emits exactly one scan per control step, and the motion-distortion
        interpolation in `Lidar.scan` spreads the beams over `pose - pose_prev`, which is one control
        step. `cfg.lidar.rate` never reaches either. At the default 40 Hz LiDAR on a 40 Hz control
        loop the two are identical (sweep 18.75 ms either way); they only diverge if someone
        configures a LiDAR rate different from the control rate, and then the old expression
        published a scan period and sweep the simulator was not running at.

        The absolute phase of the sweep against a real driver's header convention is NOT validated
        here -- only that the published duration matches the poses the beams were traced from.
        """
        lp = self.cfg.lidar
        sweep = (lp.fov / (2 * math.pi)) * self.control_dt          # what time_frac actually spans
        return dict(angle_min=-lp.fov / 2, angle_max=lp.fov / 2, angle_increment=self.lidar.angle_increment,
                    range_min=lp.range_min, range_max=lp.range_max, scan_time=self.control_dt,
                    time_increment=sweep / (lp.n_beams - 1))
