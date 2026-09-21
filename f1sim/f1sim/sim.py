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
from .prop_math import prism_contacts as prop_contacts
from .odom import VescOdom
from .params import Config
from .randomization import ParamSet
from .track import Track, TrackTensors


@dataclass
class StepResult:
    scan: torch.Tensor          # (B, N) noisy ranges (inf for no return)
    scan_true: torch.Tensor     # (B, N) noise-free ranges (privileged)
    scan_type: torch.Tensor     # (B, N) int32 what each beam hit, `lidar.HIT_*`: 0 none,
                                # 1 duct, 2 tall object (props included), 3 floor, 4 another car
    attitude: torch.Tensor      # (B, 2) body roll, pitch [rad] (sprung mass)
    odom: torch.Tensor          # (B, 5) VESC odom: x, y, yaw, v, yaw_rate (drifting)
    state: torch.Tensor         # (B, 8) ground truth: x, y, yaw, vx, vy, yaw_rate, steer, omega_r
                                # omega_r is rear-axle angular speed [rad/s]; it was appended, so
                                # the first seven columns are untouched
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
    odom_t: Optional[torch.Tensor] = None          # (B,) the stamp this step's /odom sample carries
                                                   # [s]. NOT `t`: the publish offset jitters by the
                                                   # measured distribution (odom.VescOdom.stamp), and
                                                   # a wheel-slip detector's whole input distribution
                                                   # is made of that. None with vehicle.wheel_model
                                                   # off, where the stamp is exactly `t`.
    motor_current: Optional[torch.Tensor] = None   # (B,) emulated VESC `current_motor` [A], the
                                                   # third input the traction guard takes. None with
                                                   # vehicle.wheel_model off.


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
        #: Read once, here: `vehicle.wheel_model` selects which longitudinal model `dynamics.py`
        #: runs, which speed the VESC loop closes on, whether `odom.py` quantises and jitters, and
        #: whether the IMU gets its impact term. It is structural -- baked into the compiled graph
        #: -- so `viewer/graph_fastpath.py` guards it and changing it needs a new Simulator.
        self.wheel_model = bool(self.cfg.vehicle.wheel_model)
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
        # The same reason `lidar.py` compiles `ray_prisms_hits`: `prism_contacts` walks its slots in
        # a Python loop, and one procedural layout is fifty of them. The *eager* function stays the
        # one `_resolve_wall_contact` uses, because that runs inside the already-compiled
        # `_roll_physics`, and the one `_spawn_in_prop` uses, because a reset passes a partial batch
        # whose size changes and `dynamic=False` would recompile for each.
        self._prism = prop_contacts
        if self.device.type == "cuda" and self.cfg.sim.compile:
            self._prism = self._guarded(torch.compile(prop_contacts, dynamic=False), "_prism", prop_contacts)
            self._roll = self._guarded(torch.compile(self._roll_physics, dynamic=False, mode=self.cfg.sim.compile_mode), "_roll", self._roll_physics)
            if self.cfg.sim.compile_mode == "reduce-overhead":
                self._post = self._guarded(torch.compile(self._post_roll, dynamic=False, mode="reduce-overhead"), "_post", self._post_roll)

        #: [m] of clear road a spawning car is guaranteed ahead of it; 0 is off and is every run
        #: before `_spawn_runway_blocked` existed. `F1VecEnv` sets it from `EnvConfig.spawn_runway`.
        self.spawn_runway = 0.0
        #: Whether a struck prop is shoved or is a wall with a crate's shape. `F1VecEnv` sets it.
        self.movable_obstacles = False
        self.state = torch.zeros(num_envs, dyn.STATE_DIM, device=self.device)
        self.ax = torch.zeros(num_envs, device=self.device)
        self.ay = torch.zeros(num_envs, device=self.device)
        self.pose_prev = torch.zeros(num_envs, 3, device=self.device)
        self.att = torch.zeros(num_envs, 6, device=self.device)          # roll, roll_rate, pitch, pitch_rate, road roll, road pitch (OU)
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
        #: The stamp the previous /odom sample carried, per env. `VescOdom.stamp` needs it for the
        #: catch-up branch, and it lives here rather than in the odometry state because the draw
        #: happens in `step`, outside the compiled region.
        self._odom_t_prev = torch.zeros(num_envs, device=self.device)
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
        #: Env rows in their own order, for `TrackTensors.props_for` when a per-env prop layout is
        #: attached. Held rather than built per call: `arange` inside a compiled region is host data
        #: and a host-to-device copy invalidates a CUDA graph capture.
        self.eid = torch.arange(num_envs, device=self.device)
        #: [m] how far from the car centre a prop centre can be and still touch the footprint: the
        #: car's own circumradius plus the widest prop's. Only the per-env layouts use it, to cull
        #: the contact SAT to the slots that can matter; 0 keeps every slot.
        self.prop_reach = 0.0
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
    def sample_spawn(self, n: int, lateral_std: float = 0.3, yaw_std=0.2,
                     s: Optional[torch.Tensor] = None, tid: Optional[torch.Tensor] = None,
                     min_clearance=None, lat: Optional[torch.Tensor] = None,
                     eid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Random poses along the centerline of each env's track (tid (n,), default track 0). Returns (n, 3).

        min_clearance: scalar, or an (n,) tensor when the rows want different bounds -- an alongside
        grid (`gym_env` `spawn_order`) puts cars deliberately off the centerline, and the default
        0.5 m pull-back would drag the whole grid back onto one line.
        yaw_std: scalar, or an (n,) tensor. Two cars spawned side by side need the *rotated*
        footprint to fit between the walls, and at the usual 0.2 rad jitter a 0.58 m car sweeps more
        sideways than a 1.4 m lane has to spare; cars lined up on a grid are aligned with the track
        anyway, so that caller lowers the jitter for those rows rather than giving up the grid.
        lat: (n,) explicit lateral offsets instead of a `lateral_std` draw. Given one, this function
        draws nothing for the lateral, which is what lets a caller place a grid abreast.
        eid (n,): which env row each pose belongs to. Only per-env prop layouts need it -- the
        rejection below has to test the pose against *that env's* crates, and a track id does not
        say which env is being reset.

        One interaction between the last two worth stating, because it is the reason
        `gym_env._reset_envs` tests an abreast grid against the props itself: the prop rejection
        below replaces a blocked pose with a *centerline* pose, which throws away whatever `lat`
        asked for. For a grid laid abreast that would put both cars on one line, i.e. in contact. So
        an abreast row is only ever handed to this function once its caller has established that the
        pose is clear of both the walls and the props, and the rejection never fires on it.
        """
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
            any_yaw = float(yaw_std.max()) if torch.is_tensor(yaw_std) else float(yaw_std)
            poses[:, 2] = torch.rand(n, device=self.device, generator=self.gen) * 2 * math.pi * (any_yaw > 0)
            if bool(no_cl.all()):
                return poses
            sel = lambda t: t if not torch.is_tensor(t) else t[~no_cl]
            rest = self.sample_spawn(int((~no_cl).sum()), lateral_std, sel(yaw_std),
                                     None if s is None else s[~no_cl], tid[~no_cl],
                                     min_clearance=sel(min_clearance),
                                     lat=None if lat is None else lat[~no_cl],
                                     eid=None if eid is None else eid[~no_cl])
            poses[~no_cl] = rest
            return poses
        if s is None:
            s = torch.rand(n, device=self.device, generator=self.gen) * self.track.length[tid]
        s = s.to(self.device)
        xy, yaw = self.track.pose_at_s(s, tid)
        lat = (torch.randn(n, device=self.device, generator=self.gen) * lateral_std if lat is None
               else lat.to(self.device))
        nrm = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
        xy = xy + nrm * lat[:, None]
        yaw = yaw + torch.randn(n, device=self.device, generator=self.gen) * (
            yaw_std.to(self.device) if torch.is_tensor(yaw_std) else yaw_std)
        pose = torch.cat([xy, yaw[:, None]], 1)
        # reject poses too close to walls by pulling them back to the centerline
        bad = self.track.sample_edt(xy, tid) < (self.cfg.vehicle.width if min_clearance is None else min_clearance)
        bad = bad | self._spawn_runway_blocked(s, tid, eid, lat)
        if getattr(self.track, "has_props", False) or self.spawn_runway > 0.0:
            # Props are not in the EDT, so the wall test above cannot see them and a car would spawn
            # standing inside a crate. Reject on the same geometry the contact test uses; if the
            # centerline fallback is itself blocked, nudge along the lane until it is not.
            bad = bad | self._spawn_in_prop(pose, tid, eid)
            if bad.any():
                xy0, yaw0 = self.track.pose_at_s(s[bad], tid[bad])
                pose[bad] = torch.cat([xy0, yaw0[:, None]], 1)
                # the centerline point may be blocked too, and so may the road ahead of it
                still = bad & (self._spawn_in_prop(pose, tid, eid)
                               | self._spawn_runway_blocked(s, tid, eid, None))
                # Walk a full lap in even stations rather than a few short nudges. Stopping early
                # and returning the pose anyway would spawn the car inside a crate and call it a
                # spawn; if a whole lap has nowhere to stand, that is a broken map and it says so.
                # All 32 stations are tested in one call rather than one call each. Same answer --
                # the pose taken is still the first station in order that is clear -- but with a
                # per-env obstacle layout the loop was thirty-two Python-looped SAT passes over the
                # slot dimension per reset, measured at 46 ms of a 116 ms step at `--envs 256`.
                L = self.track.length[tid].clamp_min(1e-6)
                idx = torch.nonzero(still, as_tuple=True)[0]
                if idx.numel():
                    J = 32
                    ar_j = torch.arange(1, J + 1, device=self.device, dtype=L.dtype)
                    s_alt = (s[idx][:, None] + L[idx][:, None] * (ar_j[None] / 33.0)) % L[idx][:, None]
                    tid_j = tid[idx].repeat_interleave(J)
                    xy_a, yaw_a = self.track.pose_at_s(s_alt.reshape(-1), tid_j)
                    p_alt = torch.cat([xy_a, yaw_a[:, None]], 1)
                    eid_j = None if eid is None else eid[idx].repeat_interleave(J)
                    ok = (~(self._spawn_in_prop(p_alt, tid_j, eid_j)
                            | self._spawn_runway_blocked(s_alt.reshape(-1), tid_j, eid_j, None))
                          ).reshape(-1, J)
                    first = ok.to(torch.uint8).argmax(1)
                    any_ok = ok.any(1)
                    chosen = p_alt.reshape(-1, J, 3)[torch.arange(idx.numel(), device=self.device), first]
                    pose[idx[any_ok]] = chosen[any_ok]
                    still[idx[any_ok]] = False
                if bool(still.any()):
                    raise RuntimeError(
                        f"{int(still.sum())} env(s) have no spawn pose clear of the placed props "
                        f"anywhere on their lap; the track's props block the centerline")
            return pose
        if bad.any():
            xy0, yaw0 = self.track.pose_at_s(s[bad], tid[bad])
            pose[bad] = torch.cat([xy0, yaw0[:, None]], 1)
        return pose

    def _spawn_runway_blocked(self, s: torch.Tensor, tid: torch.Tensor, eid, lat) -> torch.Tensor:
        """(n,) whether a car spawning at arc `s` has something in front it could not avoid.

        A car is placed at up to `spawn_speed_max` and its first control step is one step away. A
        crate two metres ahead of that is not an obstacle the policy failed to avoid, it is one it
        was never shown in time -- and it lands in the collision rate all the same, which makes the
        metric read worse the more interesting the layout is. The user's words (2026-09-21):
        *"지표가 과장되지 않도록 스폰 직후 피할 수 없는 지점에 장애물이나 벽이 나타나지 않도록"*.

        So the road ahead is sampled along the lane -- the props the distance field cannot see, and
        the walls it can -- and a blocked runway is rejected exactly like a blocked pose, which puts
        it through the same walk that already finds somewhere clear to stand.

        `spawn_runway` is the length in metres and 0 is off, which is every run before this existed.
        It is what the car would need to stop from `spawn_speed_max` plus its own length; the env
        sets it, and refuses to put obstacles on the racing line without one.
        """
        n = s.shape[0]
        if self.spawn_runway <= 0.0:
            return torch.zeros(n, dtype=torch.bool, device=self.device)
        # Sample no more than half a body width apart. Coarser and the check steps over a crate:
        # measured at 0.5 m spacing it let 1 car in 640 through, which is exactly the kind of
        # residual that reads later as "the policy occasionally fails at the start".
        K = max(6, int(math.ceil(self.spawn_runway / (0.5 * self.cfg.vehicle.width))))
        step = self.spawn_runway / K
        ds = torch.arange(1, K + 1, device=self.device, dtype=s.dtype) * step
        L = self.track.length[tid].clamp_min(1e-6)
        s_ahead = (s[:, None] + ds[None]) % L[:, None]                       # (n, K)
        tid_k = tid[:, None].expand(n, K).reshape(-1)
        xy, yaw = self.track.pose_at_s(s_ahead.reshape(-1), tid_k)
        if lat is not None:
            # follow the line the car is actually on, not the centerline it is offset from
            nrm = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
            xy = xy + nrm * lat[:, None].expand(n, K).reshape(-1, 1)
        need = 0.5 * self.cfg.vehicle.width
        blocked = self.track.sample_edt(xy, tid_k) < need
        props = getattr(self.track, "env_props", None)
        if props is not None:
            eid_k = (torch.arange(n, device=self.device) if eid is None else eid)
            eid_k = eid_k[:, None].expand(n, K).reshape(-1)
            blocked = blocked | (props.clearance(xy, eid_k) < need)
        return blocked.view(n, K).any(1)

    def _spawn_in_prop(self, pose: torch.Tensor, tid: torch.Tensor,
                       eid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Whether each candidate spawn pose has its footprint overlapping a prop.

        The eager `prism_contacts`, deliberately: this runs at reset on whatever partial batch is
        resetting, so a `dynamic=False` compile would recompile for every distinct size."""
        from .prop_math import prism_contacts
        c, s_ = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        R = torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2)
        pts = torch.einsum("bij,kj->bki", R, self.corners) + pose[:, None, :2]
        poses, pn, pd, z_lo, z_hi = self.track.props_near(tid, pose[:, :2], eid, self.prop_reach)
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
        # The rear wheel starts *rolling* at the spawn speed, not stopped: a car spawned at 4 m/s
        # with omega_r = 0 is a car spawned mid-lock, which would fire the guard on every reset and
        # put a 4 m/s slip transient at the head of every episode.
        st[:, dyn.IOMEGA] = st[:, dyn.IVX] / self.P["r_w"][env_ids]
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
        self._odom_t_prev[env_ids] = self.t
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
                                                     "cl_idx", "car_collision", "prop_touched",
                                                     "_odom_t_prev")}
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
        # Roll/pitch are alternate columns. A strided view avoids constructing a
        # CUDA index tensor from a Python list (and synchronizing) every step.
        self.att_prev = self.att[:, :3:2].clone()
        # the sample layout belongs to this step's phase; capture it before the phase advances
        imu_offsets = self._imu_offsets[self._imu_phase]
        state, ax, ay, att, imu_state, imu_samples, i_motor = self._roll(
            self.state, self.ax, self.att, self.imu_state, self.cmd_hist, delay_s, P)
        self._imu_phase = (self._imu_phase + 1) % self.imu_period
        if self.cfg.sim.compile_mode == "reduce-overhead":      # CUDA graphs reuse their output buffers: keep copies
            state, ax, ay, att, imu_state, imu_samples, i_motor = (
                t.clone() for t in (state, ax, ay, att, imu_state, imu_samples, i_motor))
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
        # Latched while a collision ends the episode -- the car is frozen and the flag has to
        # survive to the reset that reads it. Under soft contact it must NOT latch: the episode
        # carries on, and a flag that stays true is a car charged the collision penalty on every
        # remaining step. Measured before this line was conditional: reward per step ran
        # -0.95, -3.0, -8.1, -12.6 over four updates and kept going.
        self.collided = (self.collided | hit) if self.cfg.sim.terminate_on_collision else hit
        self.odom.state = odom_state; self.s = s; self.lap = lap
        odom = odom_state

        # --- sensors ---
        pose = state[:, :3]
        scan, scan_true, scan_type = self.lidar.scan(pose, self.pose_prev, P, self.cfg.lidar.motion_distortion,
                                                     att=att[:, :3:2], att_prev=self.att_prev, tid=self.tid, cars=cars,
                                                     eid=self.eid)
        imu = imu_samples if self.cfg.imu.enabled else None
        imu_att = imu_state[:, 18:21].clone() if self.cfg.imu.enabled else None
        # The /odom stamp is drawn here, outside `_post`: it is one cheap draw per step and keeping
        # it out of the compiled region means the sample layout of the graph does not depend on the
        # switch. With the wheel model off there is no jitter to model and the stamp is exactly `t`.
        if self.wheel_model:
            odom_t = self.odom.stamp(self.t, self._odom_t_prev, P, self.control_dt)
            self._odom_t_prev = odom_t
        else:
            odom_t = None
        return StepResult(scan, scan_true, scan_type, att[:, 0:3:2].clone(), odom, state, imu, imu_att,
                          self.collided.clone(), ds, s, lateral, self.lap.clone(), wall_dist, self.t,
                          self.car_collision.clone() if self.M > 1 else None,
                          imu_offsets if self.cfg.imu.enabled else None,
                          odom_t, i_motor if self.wheel_model else None)

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
        # vesc_to_odom publishes ERPM, i.e. the WHEEL speed. With the wheel model on those differ
        # from the body speed exactly when the axle slips, which is what makes the odometry a slip
        # sensor; with it off `omega_r * r_w` is identically `vx` and this is the old call.
        v_odom = state[:, dyn.IOMEGA] * P["r_w"] if self.wheel_model else state[:, dyn.IVX]
        odom_state = self.odom.update_pure(odom_state, v_odom, cmd[:, 0], P, self.control_dt,
                                           quantise=self.wheel_model)
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
        roll, roll_rate, pitch, pitch_rate, w_roll, w_pitch = (att[:, i] for i in range(6))
        wn, zeta = P["susp_wn"], P["susp_zeta"]
        # floor / tyre tilt: a stationary Ornstein-Uhlenbeck target (rms road_tilt, correlation
        # road_tau) added to the suspension's set point. Exact discretisation, so the rms is
        # road_tilt whatever the substep.
        ou_a = (self.dt / P["road_tau"]).clamp(0.0, 1.0)
        ou_k = 1.0 - ou_a
        ou_s = P["road_tilt"] * torch.sqrt(1.0 - ou_k * ou_k)
        imu_on = self.cfg.imu.enabled
        wheel_on = self.wheel_model
        r_vec = torch.stack([P["imu_x"] - P["lr"], P["imu_y"], P["imu_z"] - P["h"]], 1)
        samples = []
        a_cmd_last = torch.zeros_like(ax)
        soft_wall = not self.cfg.sim.terminate_on_collision
        ar = torch.arange(self.B, device=state.device)
        for k in range(self.substeps):
            age = delay_s - (k + 1) * self.dt
            idx = torch.ceil(age / self.control_dt).long().clamp(0, self.hist_len - 1)
            eff = cmd_hist[ar, idx]                                   # (B,2) command in effect
            steer_tgt = servo_target(eff[:, 0], P)
            # The VESC loop closes on ERPM -- the wheel -- not on the body. See actuators.vesc_accel.
            v_fb = state[:, dyn.IOMEGA] * P["r_w"] if wheel_on else state[:, dyn.IVX]
            a_cmd = vesc_accel(eff[:, 1], v_fb, P)
            a_cmd_last = dyn.motor_accel(a_cmd, v_fb, P)     # after the limits: what is applied
            r_old = state[:, dyn.IR]
            state, ax, ay = dyn.step_dynamics(state, steer_tgt, a_cmd, ax, P, P["servo_tau"], self.dt,
                                              wheel=wheel_on)
            # Before the sprung mass, so the springs feel the hit: the body does not deform but it
            # is on suspension, and a car that runs into something dives on its nose. The contact
            # changes the velocity outside the dynamics, so `ax`/`ay` alone never carried it --
            # the same reason the accelerometer was deaf to an impact.
            a_contact = None
            if soft_wall:
                state, a_contact = self._resolve_wall_contact(state, dt=self.dt)
            a_x = ax if a_contact is None else ax + a_contact[:, 0]
            a_y = ay if a_contact is None else ay + a_contact[:, 1]
            # sprung mass: roll to the outside of the corner, dive under braking, squat under throttle
            w_roll = w_roll * ou_k + torch.randn_like(w_roll) * ou_s
            w_pitch = w_pitch * ou_k + torch.randn_like(w_pitch) * ou_s
            roll_ss = P["roll_per_g"] * a_y / dyn.G + w_roll
            # asymmetric: the car squats under throttle far more than it dives under (regen-limited) braking
            pitch_ss = -torch.where(a_x > 0, P["pitch_per_g"], P["dive_per_g"]) * a_x / dyn.G + w_pitch
            roll_acc = wn * wn * (roll_ss - roll) - 2 * zeta * wn * roll_rate
            pitch_acc = wn * wn * (pitch_ss - pitch) - 2 * zeta * wn * pitch_rate
            roll_rate = roll_rate + roll_acc * self.dt
            pitch_rate = pitch_rate + pitch_acc * self.dt
            roll = (roll + roll_rate * self.dt).clamp(-0.35, 0.35)
            pitch = (pitch + pitch_rate * self.dt).clamp(-0.35, 0.35)
            if imu_on:
                # `yaw_acc` after the contact's damping, so the gyro sees the jolt too.
                yaw_acc = (state[:, dyn.IR] - r_old) / self.dt
                imu_state = imu_model.substep(imu_state, a_x, a_y, state[:, dyn.IVX], roll, pitch, roll_rate, pitch_rate,
                                              state[:, dyn.IR], roll_acc, pitch_acc, yaw_acc, r_vec, P, self.dt,
                                              shock=wheel_on)
                if k in self.imu_idx:
                    smp, imu_state = imu_model.sample(imu_state, P, self.imu_ts)
                    samples.append(smp)
        imu_samples = torch.stack(samples, 1) if samples else torch.zeros(state.shape[0], 0, 6, device=state.device)
        # `/sensors/core` `current_motor`, the guard's third (optional) input. The VESC commands a
        # current, and the torque it produces is that current times a constant: this inverts the
        # last substep's motor torque through `amp_per_nm`, which is calibrated in ActuatorParams
        # against the one place the recordings pin it -- the regen limit.
        i_motor = a_cmd_last * P["m"] * P["r_w"] * P["amp_per_nm"]
        return (state, ax, ay, torch.stack([roll, roll_rate, pitch, pitch_rate, w_roll, w_pitch], 1),
                imu_state, imu_samples, i_motor)

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

    def _prop_contact(self, state: torch.Tensor, fn=None, want_slot: bool = False):
        """Penetration depth and outward normal against the placed props: (pen (B,), n (B,2)).

        Deliberately not routed through `_footprint_clearance`. That test samples the EDT at the
        car's four corners, which cannot see an obstacle that fits between them: a 5 cm marker post
        standing under the middle of the car leaves all four corners clear, so the car drives through
        it. SAT over both polygons' edge normals finds containment as readily as edge crossing.

        The normal has to come from the contacted prop as well. `edt_gradient` reads the occupancy
        grid, and props are not in the grid -- it would return the direction away from the nearest
        *wall*, which is unrelated to the box the car is actually touching."""
        tr = self.track
        poses, pn, pd, z_lo, z_hi = tr.props_near(self.tid, state[:, :2], self.eid, self.prop_reach)
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
        if want_slot:
            *cols, slot_of = tr.props_near(self.tid, state[:, :2], self.eid, self.prop_reach,
                                           return_slot=True)
            poses, pn, pd, z_lo, z_hi = cols
            depth, normal, col = (fn or self._prism)(corners, poses, pn, pd, z_lo, z_hi,
                                                     self.car_dims[:, 2])
            rows = torch.arange(col.shape[0], device=col.device)
            slot = torch.where(col >= 0, slot_of[rows, col.clamp_min(0)],
                               torch.full_like(col, -1))
            return depth.clamp_min(0.0), normal, slot
        # Body height, not CoG height: `vehicle.h` is where the mass sits (0.074 m) and doubling it
        # is not a silhouette. `car_dims[:, 2]` is the height the LiDAR already uses for this car.
        depth, normal, _ = (fn or self._prism)(corners, poses, pn, pd, z_lo, z_hi, self.car_dims[:, 2])
        return depth.clamp_min(0.0), normal

    def _resolve_wall_contact(self, state: torch.Tensor, dt: Optional[float] = None):
        """Soft wall: push CoG out along EDT gradient and kill the into-wall velocity component.

        With `dt` it also returns the body-frame acceleration the contact imposed, (B, 2), so the
        IMU can be told about it. It has to be told: the contact changes the velocity *outside* the
        dynamics, and `ax`/`ay` come from the tyre and actuator forces, so an impact that takes the
        car from 8 m/s to a standstill used to be reported by the accelerometer as **nothing at
        all**. The 22 recordings say otherwise -- 62 events above 2.5 g across them, peaking at
        12.6 g on a hit that went 8.14 m/s to 0.00 in 20 ms -- and on a real car the impact is the
        largest thing the accelerometer ever sees. A policy that is meant to learn what to do after
        a touch has to be able to tell that one happened.
        """
        clr = self._footprint_clearance(state)
        pen = (-clr).clamp_min(0.0)
        touching = pen > 0
        n = self.track.edt_gradient(state[:, :2], self.tid)            # away from wall (world)
        prop_slot = None                   # set below when the props can be pushed
        if getattr(self.track, "has_props", False):
            # Whichever of wall or prop is deeper owns this step's normal. Blending two normals
            # would push the car somewhere neither contact asks for.
            movable = self.movable_obstacles and getattr(self.track, "env_props", None) is not None
            got = self._prop_contact(state, fn=prop_contacts, want_slot=movable)
            pen_p, n_p = got[0], got[1]
            self.prop_touched = self.prop_touched | (pen_p > 0)   # substeps, read once per step
            prop_deeper = pen_p > pen
            pen = torch.where(prop_deeper, pen_p, pen)
            n = torch.where(prop_deeper[:, None], n_p.to(n.dtype), n)
            touching = pen > 0
            prop_slot = got[2] if movable else None
        yaw = state[:, dyn.IYAW]
        c, s = torch.cos(yaw), torch.sin(yaw)
        vwx = state[:, dyn.IVX] * c - state[:, dyn.IVY] * s
        vwy = state[:, dyn.IVX] * s + state[:, dyn.IVY] * c
        vn = vwx * n[:, 0] + vwy * n[:, 1]
        into = (vn < 0) & touching
        # Which material this contact is against. The track carries a distance field for the duct
        # shell and one for the tall wall, so "is the thing I am touching a hose" is a lookup, not
        # a guess -- and the two behave differently enough that using one number for both was the
        # reason a duct impact rebounded like concrete.
        sp = self.cfg.sim
        is_duct = torch.zeros_like(pen, dtype=torch.bool)
        if getattr(self.track, "edt_duct", None) is not None:
            d_duct = self.track.sample_edt(state[:, :2], self.tid, field=self.track.edt_duct)
            d_tall = self.track.sample_edt(state[:, :2], self.tid, field=self.track.edt_tall)
            is_duct = d_duct <= d_tall
        if prop_slot is not None:
            # A crate is not a hose. Whatever the boundary is made of, a prop contact is the prop's.
            is_duct = is_duct & ~(prop_deeper & touching)
        k = torch.where(is_duct, torch.full_like(pen, sp.collision_restitution_duct),
                        torch.full_like(pen, sp.collision_restitution))
        tau = torch.where(is_duct, torch.full_like(pen, sp.contact_tau_duct),
                          torch.full_like(pen, sp.contact_tau_wall))
        # How much of the normal velocity this substep takes. A real contact lasts about 40 ms on
        # a duct hose -- sixteen substeps -- and taking all of it in one was what made an impact
        # invisible to everything that integrates, the suspension included.
        bleed = (self.dt / tau.clamp_min(1e-6)).clamp(max=1.0)
        f = sp.wall_friction
        # How much of the impulse the car keeps. Against a wall, all of it: the old expression is
        # the m_prop -> infinity limit of this one. Against a 10 kg crate with a 3.74 kg car, the
        # car keeps m_p/(m_c+m_p) = 0.73 of the bounce and the crate takes the rest and slides.
        share = torch.ones_like(vn)
        if prop_slot is not None:
            m_c = float(self.cfg.vehicle.m)
            rows = torch.arange(state.shape[0], device=state.device)
            m_p = torch.where(prop_slot >= 0,
                              self.track.env_props.p_mass[self.eid, prop_slot.clamp_min(0)],
                              torch.zeros_like(vn))
            movable_now = into & prop_deeper & (m_p > 0)
            share = torch.where(movable_now, m_p / (m_c + m_p), share)
            # Newton's third law, as an impulse on the prop: J = -(1+k) * mu_red * vn along n, and
            # the prop takes -J. `vn` is negative while driving in, so the crate goes away from us.
            mu_red = (m_c * m_p) / (m_c + m_p).clamp_min(1e-6)
            J = (1 + k) * bleed * (-vn) * mu_red                       # >= 0 while driving in
            imp = torch.where(movable_now[:, None], -J[:, None] * n, torch.zeros_like(n))
            self.track.env_props.shove(self.eid, prop_slot.clamp_min(0), imp)
        vwx2 = vwx - (1 + k) * share * bleed * vn * n[:, 0]
        vwy2 = vwy - (1 + k) * share * bleed * vn * n[:, 1]
        vwx2, vwy2 = vwx2 * (1 - f * 0.05), vwy2 * (1 - f * 0.05)
        vwx = torch.where(into, vwx2, vwx); vwy = torch.where(into, vwy2, vwy)
        st = state.clone()
        st[:, dyn.IVX] = vwx * c + vwy * s
        st[:, dyn.IVY] = -vwx * s + vwy * c
        # The overlap is shared the same way: a crate gets out of the way as much as the car does.
        # The hose gives: the car is returned over the contact rather than teleported clear of it,
        # which is what makes the penetration -- and so the acceleration the IMU and the suspension
        # see -- last the measured 40 ms instead of one substep.
        st[:, :2] = st[:, :2] + n * (pen * touching.float() * share * bleed)[:, None]
        st[:, dyn.IR] = torch.where(touching, st[:, dyn.IR] * (1.0 - 0.1 * bleed), st[:, dyn.IR])
        if dt is None:
            return st
        a_c = torch.stack([(st[:, dyn.IVX] - state[:, dyn.IVX]) / dt,
                           (st[:, dyn.IVY] - state[:, dyn.IVY]) / dt], 1)
        # Saturate like the part does. A 3 m/s change inside one 2.5 ms substep is 122 g, and the
        # MPU-class accelerometers these cars carry are configured to +-16 g -- the recordings peak
        # at 12.6, comfortably inside that. Without the clamp the policy would be handed a number
        # no real sensor can produce and would be right to have learned nothing from it.
        lim = self.cfg.imu.accel_range
        mag = a_c.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return st, a_c * (mag.clamp(max=lim) / mag)

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
