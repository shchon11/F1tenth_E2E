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
    wall_collision: Optional[torch.Tensor] = None  # (B,) bool: walls and props only -- `collision` also ORs in
                                                   # `car_collision`, so a car contact read through it is a wall too.
                                                   # Soft contact with `SimParams.wall_latch`: every wall contact
                                                   # the roll resolved this step, held until the footprint is a
                                                   # grid cell clear (before it: the step-end corner test only)
    # How hard a contact is, [m/s]: the speed of the two surfaces into each other along the contact
    # normal, at the contact point. A graze along a wall and a head-on hit are both one contact onset;
    # these tell them apart. 0 where nothing touched.
    wall_impact: Optional[torch.Tensor] = None     # (B,) walls and props. Soft contact: the largest speed into
                                                   # the surface that `_resolve_wall_contact` resolved this step
                                                   # (the `-vpn` its impulse is computed from), over all the
                                                   # step's resolutions -- including a touch resolved and left
                                                   # inside the step, which `wall_collision` does not report.
                                                   # Terminate: the crash state's, with `measure_impacts` only
                                                   # (None without it).
    car_impact: Optional[torch.Tensor] = None      # (B,) car contacts (`car_collision`): the closing speed of the
                                                   # two bodies at the end of the step (`Simulator._car_contacts`).
                                                   # None unless `measure_impacts`, and in a solo run.
    # Contact onsets per kind (soft contact with `SimParams.wall_latch`, None otherwise): columns wall, prop,
    # car. A contact of one kind starting while one of another lasts is its own onset -- a crate hit while
    # the car grinds a hose was no rising edge of the merged flag, so it was never counted or charged.
    contact_onsets: Optional[torch.Tensor] = None   # (B, 3) bool: the kind's contact began this step
    contact_impacts: Optional[torch.Tensor] = None  # (B, 3) [m/s] this step's impact per kind: walls and props as
                                                    # `wall_impact` (split by which owned the contact), cars the
                                                    # closing speed (`car_impact`, 0 where not measured)


#: Columns of `Simulator.contact_frame`, one row per car: this step's hardest contact (kind 0 none, 1 wall,
#: 2 prop, 3 car; where it touched, world [m]; how hard [m/s]), the contact held for the console's eye
#: (seconds left, and the hardest contact of the hold: kind, point, severity), and the episode's onsets per
#: kind, all of them and the hard ones (>= `contact_severity.HARD_MPS`).
CONTACT_FRAME_COLUMNS = ("kind", "x", "y", "severity", "held_s", "held_kind", "held_x", "held_y", "held_severity",
                         "onsets_wall", "onsets_prop", "onsets_car", "hard_wall", "hard_prop", "hard_car")
CONTACT_KINDS = ("none", "wall", "prop", "car")
#: [s] how long a contact stays on the console's contact flag after the car stops touching
CONTACT_HOLD_S = 0.3


def _capturing() -> bool:
    """Whether this call is being recorded into a CUDA graph or traced by `torch.compile` -- the
    two places a data-dependent Python branch is not allowed."""
    try:
        if torch.compiler.is_compiling():
            return True
    except AttributeError:
        pass
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


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
        motor_delay = float(self.cfg.actuator.motor_delay)
        for k, (lo, hi) in self.cfg.rand.ranges.items():
            if k == "actuator.motor_delay":
                motor_delay = max(motor_delay, hi)
        max_delay += motor_delay          # the speed channel looks this much further back
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
        #: Resolve soft contact every this many substeps (see `_roll_physics`). 5 is 5 ms at the
        #: 1 ms substep: a quarter of the duct's 20 ms contact time constant, so the response is
        #: still resolved finely, and a car at 9 m/s travels 4.5 cm into a crate before the next
        #: check -- inside the half-width margin every contact already tolerates.
        self.contact_every = max(1, int(getattr(self.cfg.sim, "contact_every", 5)))
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
        self.absent = torch.zeros(num_envs, dtype=torch.bool, device=self.device)   # `park`
        self.spawn_stats = {"rejected": 0, "group_walks": 0, "fallback_rows": 0, "crowded": 0}
        #: Prop contact seen inside a substep, latched until `step` reads it. A box can be entered
        #: and pushed back out between two control steps, and a collision that the physics resolved
        #: but nothing reported is a collision the caller never learns about.
        self.prop_touched = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        #: The same for the walls (`SimParams.wall_latch`): a wall contact the soft resolution handled
        #: inside the roll, latched until `step` reads it. The step-end corner test alone missed most
        #: of them -- the resolution pushes the car back out every few milliseconds. Written in place,
        #: like `prop_touched`, so a captured graph keeps writing into it.
        self.wall_touched = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        #: Whether the last step ended in wall contact (`wall_latch`'s release hysteresis, see `step`).
        self.wall_engaged = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        #: [m/s^2] body-frame acceleration the last contact resolution imposed, held until the next one
        #: (`SimParams.contact_accel_hold`): it covers the window that resolution resolved, which runs on
        #: into the next control step. Written in place at the end of the roll.
        self.contact_acc = torch.zeros(num_envs, 2, device=self.device)
        #: Car-car contact seen inside the roll (`SimParams.car_response`), latched like `wall_touched`: the
        #: response pushes two cars apart within the step, so the step-end test alone would miss it.
        self.car_touched = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        #: [m/s] the largest closing speed along the contact normal at the contact point that a car-car
        #: resolution saw this step -- the speed the impulse took (`StepResult.car_impact`). Zeroed in place
        #: before each roll.
        self.car_impact_roll = torch.zeros(num_envs, device=self.device)
        #: [m/s] `wall_impact` split by kind, (B, 2) walls, props: the step's largest speed into whichever
        #: owned each resolution's contact. In place inside the roll, zeroed before it.
        self.impact_kind = torch.zeros(num_envs, 2, device=self.device)
        #: Per kind (wall, prop, car), whether the last step ended in contact: the onsets' memory.
        self.kind_prev = torch.zeros(num_envs, 3, dtype=torch.bool, device=self.device)
        #: Where this step's hardest contact touched, world [m], and its speed [m/s] (the roll keeps the
        #: hardest resolution's point, in place; zeroed before each roll).
        self.contact_pt = torch.zeros(num_envs, 2, device=self.device)
        self.contact_pt_sev = torch.zeros(num_envs, device=self.device)
        #: What the console draws (`contact_frame`), per car, updated in place every step: this step's
        #: hardest contact, the one held for `CONTACT_HOLD_S`, and the episode's onsets per kind, all / hard.
        self.contact_now = torch.zeros(num_envs, 4, device=self.device)        # kind, x, y, severity
        self.contact_held = torch.zeros(num_envs, 5, device=self.device)       # seconds left, kind, x, y, severity
        self.contact_counts = torch.zeros(num_envs, 6, dtype=torch.long, device=self.device)
        #: [m/s] This step's wall / prop impact severity: the largest speed into the surface, at the
        #: contact point, of any contact `_resolve_wall_contact` resolved (`StepResult.wall_impact`).
        #: Written in place inside the roll -- a CUDA graph replays writes to this storage, a rebound
        #: attribute would be dropped (the `prop_touched` lesson) -- and zeroed in place before each roll.
        self.wall_impact = torch.zeros(num_envs, device=self.device)
        #: Whether `step` also measures the severities the roll does not record: a car contact's
        #: closing speed (races) and, under `terminate_on_collision`, a crash's speed into the wall.
        #: Off by default, so nothing pays for them unread; eager work after the roll, so it can be
        #: switched at any time (the evaluation's severity meter does, and an env charging by severity).
        self.measure_impacts = False
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
        # Fraction of the beams hitting another car that return nothing. Measured on 19 299 car-sized
        # objects in the competition recordings: 0.165 % of their beams (p50 0 %, p99 7 % per
        # object; docs/research/dynamics-collision-audit-2026-09-21.md section 1.2). It was 5-30 %
        # (mean 17.5 %): the other car flickered out of the scan a hundred times more often than on
        # the real track, and a policy trained on that cannot tell a real disappearance (occlusion)
        # from the usual flicker.
        self.car_porosity[ids] = u(0.0, 0.004)
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

    #: (B,) a car that is not in this race (an empty opponent seat). Set by `park`, cleared by `unpark`.
    absent_any: bool = False

    def park(self, ids: torch.Tensor) -> None:
        """Take these cars out of the race: parked 1 km off the map (each at its own spot), at rest, frozen,
        their wall / car contacts masked. Far away is what keeps them out of every other car's LiDAR,
        contact test and nearest-opponent column without a mask in each of those places -- for the
        terms that read *positions*. Not for the arc: `s` comes from a lane projection that searches
        only a few metres around the last index, so a parked car keeps a place on the lap, and
        `F1VecEnv.signed_gaps` has to mask it (it reads `absent`)."""
        if not hasattr(self, "absent") or self.absent.shape[0] != self.B:
            self.absent = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        if ids.numel() == 0:
            return
        self.absent[ids] = True
        self.absent_any = True
        st = self.state
        st[ids] = 0.0
        st[ids, 0] = -1000.0 - 5.0 * ids.to(st.dtype)
        st[ids, 1] = -1000.0
        if self.wheel_model:
            st[ids, dyn.IOMEGA] = 0.0
        self.collided[ids] = False

    def unpark(self, ids: torch.Tensor) -> None:
        """These cars are in the race again (the next `reset` places them)."""
        if not hasattr(self, "absent") or ids.numel() == 0:
            return
        self.absent[ids] = False
        self.absent_any = bool(self.absent.any())

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

    def _car_contacts(self, state: torch.Tensor, impact: bool = False):
        """Separating-axis test between each car's footprint and the other cars of its race -> (B,) bool.

        Each car collides with the body the other cars' LiDAR draws for it: the footprint is scaled
        by the same `car_dims` length / 0.50 the mesh is (`_car_boxes`). A fixed 0.58 m box on every
        car, whatever size it was drawn, put up to 11 cm of body where the scan showed nothing
        (docs/research/dynamics-collision-audit-2026-09-21.md section 2.4). Walls and props still use
        the car's own measured footprint (`_footprint_corners`).

        impact: also return (B,) the contact's closing speed [m/s] (`StepResult.car_impact`), the
        largest over the cars this one touches, 0 without contact. Soft mode has no car-car response,
        so the velocities at the end of the step are still the approach. The normal is the axis the
        bodies entered on: of the four SAT axes, the one whose overlap is the youngest at the current
        closing rate (penetration / closing speed). The axis of least overlap is the usual choice and
        is wrong exactly where it matters -- a fast corner-to-corner rear-end overlaps 7 cm along the
        cars and 2 cm across them, and least overlap would score the 3 m/s hit as the 0 m/s sideways
        graze. The velocities are taken at the contact point (the vertices of each body inside the
        other; the midpoint of the two centres when none is), so a spinning car's swinging tail counts
        and two cars cornering together are not read as closing on each other."""
        c_, s_ = torch.cos(state[:, dyn.IYAW]), torch.sin(state[:, dyn.IYAW])
        R = torch.stack([torch.stack([c_, -s_], -1), torch.stack([s_, c_], -1)], -2)          # (B,2,2)
        local = self.corners[None] * (self.car_dims[:, 0] / 0.50)[:, None, None]              # (B,4,2)
        pts = torch.einsum("bij,bkj->bki", R, local) + state[:, None, :2]                     # (B,4,2)
        yaw = state[:, dyn.IYAW]
        ext = torch.stack([-self.car_rear[:, 0] * torch.cos(yaw), -self.car_rear[:, 0] * torch.sin(yaw)], 1)
        pts = torch.cat([pts, pts[:, 2:] + ext[:, None, :]], 1)         # + rear box corners (B,6,2)
        ax = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1); ay = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
        hit = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
        if impact:
            sev = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
            xy, w = state[:, :2], state[:, dyn.IR]
            v = torch.stack([state[:, dyn.IVX] * c_ - state[:, dyn.IVY] * s_,
                             state[:, dyn.IVX] * s_ + state[:, dyn.IVY] * c_], 1)       # (B,2) world velocity
        for c in range(self.M - 1):
            o = self.other_idx[:, c]
            po = pts[o]
            sep = torch.zeros_like(hit)
            spans = []
            for axis in (ax, ay, ax[o], ay[o]):
                pa = (pts * axis[:, None, :]).sum(-1); pb = (po * axis[:, None, :]).sum(-1)
                sep = sep | (pa.max(1).values < pb.min(1).values) | (pb.max(1).values < pa.min(1).values)
                if impact:
                    spans.append((axis, pa, pb))
            hit = hit | ~sep
            if impact:
                sev = torch.maximum(sev, torch.where(~sep, self._car_closing_speed(spans, pts, po, xy, xy[o], v, v[o],
                                                                                   w, w[o]), torch.zeros_like(sev)))
        if impact:
            return hit, sev
        return hit

    @staticmethod
    def _car_closing_speed(spans, pts, po, x_a, x_b, v_a, v_b, w_a, w_b) -> torch.Tensor:
        """(B,) closing speed of body a (vertices `pts`) on body b (`po`) along their entry axis, at the
        contact point; `spans` holds (axis, a's projections, b's projections) for the four SAT axes, a's
        two first. See `_car_contacts`."""
        lo_a = [pa.min(1).values for _, pa, _ in spans]; hi_a = [pa.max(1).values for _, pa, _ in spans]
        lo_b = [pb.min(1).values for _, _, pb in spans]; hi_b = [pb.max(1).values for _, _, pb in spans]
        # the contact point: a's vertices inside b (tested on b's own axes, the last two) and b's inside a
        a_in_b = torch.ones_like(pts[..., 0], dtype=torch.bool)
        b_in_a = torch.ones_like(a_in_b)
        for k, (_, pa, pb) in enumerate(spans):
            if k < 2:
                b_in_a = b_in_a & (pb >= lo_a[k][:, None]) & (pb <= hi_a[k][:, None])
            else:
                a_in_b = a_in_b & (pa >= lo_b[k][:, None]) & (pa <= hi_b[k][:, None])
        wa, wb = a_in_b.to(pts.dtype), b_in_a.to(pts.dtype)
        cnt = wa.sum(1) + wb.sum(1)
        mid = 0.5 * (x_a + x_b)
        p = torch.where((cnt > 0)[:, None],
                        ((wa[..., None] * pts).sum(1) + (wb[..., None] * po).sum(1)) / cnt.clamp_min(1.0)[:, None], mid)
        ra, rb = p - x_a, p - x_b
        va = v_a + w_a[:, None] * torch.stack([-ra[:, 1], ra[:, 0]], 1)
        vb = v_b + w_b[:, None] * torch.stack([-rb[:, 1], rb[:, 0]], 1)
        v_rel = va - vb                                                    # a's velocity relative to b
        best_t = torch.full_like(mid[:, 0], float("inf"))
        best_s = torch.zeros_like(best_t)
        for k, (axis, _, _) in enumerate(spans):
            above = (lo_a[k] + hi_a[k]) > (lo_b[k] + hi_b[k])              # a on the +axis side of b
            depth = torch.where(above, hi_b[k] - lo_a[k], hi_a[k] - lo_b[k])
            vu = (v_rel * axis).sum(1)
            closing = torch.where(above, -vu, vu)                          # > 0: moving into b along this axis
            t = torch.where(closing > 1e-3, depth.clamp_min(0.0) / closing.clamp_min(1e-3),
                            torch.full_like(best_t, float("inf")))
            better = t < best_t
            best_t = torch.where(better, t, best_t)
            best_s = torch.where(better, closing, best_s)
        return best_s

    # ------------------------------------------------------------------ reset
    def sample_spawn(self, n: int, lateral_std: float = 0.3, yaw_std=0.2,
                     s: Optional[torch.Tensor] = None, tid: Optional[torch.Tensor] = None,
                     min_clearance=None, lat: Optional[torch.Tensor] = None,
                     eid: Optional[torch.Tensor] = None, group: Optional[torch.Tensor] = None,
                     active: Optional[torch.Tensor] = None, lat_fixed: Optional[torch.Tensor] = None,
                     avoid_s: Optional[torch.Tensor] = None, clear_gap: float = 0.0) -> torch.Tensor:
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

        The rest describe a *race*, and exist because every fallback below used to treat each car
        alone (2026-10-14, the G4 boot-camp audit: 122 of 379 charged car contacts began in the
        first 2 s of an episode, 59 of them on step 0, every one charged to the learner):

        group (n,): rows with the same id are one race laid out together. When a car of the group
        is rejected (a wall, its runway, a crate on its heading, a crate under it), the *whole*
        group walks the lap by one common arc offset, so every car keeps its lateral offset and its
        arc gap to the others; the station taken is the first one where every car of the group
        passes. Placement probe, boot-camp config, 1,600 races: 23 % of abreast and 26 % of grid
        cars were rejected, the per-car centerline pull and lap walk put 153 learner pairs in
        overlap at placement (both abreast cars walked to the same station; one abreast car pulled
        onto the centerline beside its partner; a walked car landing on another) and 116 grid
        pairs within 2.4 m of arc. None given: every row is its own group, the rule this function
        always had.
        active (n,): rows that take part. An inactive row (an empty seat, parked right after this)
        is neither tested nor allowed to hold the race up; it rides along with its group.
        lat_fixed (n,): rows whose `lat` is part of a formation (an abreast row). Such a row is never
        pulled to the centerline -- pulled onto the line its partner is beside, it is in contact.
        A free row (a staggered grid car, whose lateral is a draw) still may be, at its own arc.
        avoid_s (n, C), clear_gap: arc positions this row must stay `clear_gap` metres of arc away
        from (NaN: none) -- a car rejoining a race already under way, whose mates are where they
        are. Also what the rare fallback below uses when a group has no common station: the cars
        that failed walk on their own, each kept `clear_gap` from every other car of its group.
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
                                     eid=None if eid is None else eid[~no_cl],
                                     group=sel(group), active=sel(active), lat_fixed=sel(lat_fixed),
                                     avoid_s=sel(avoid_s), clear_gap=clear_gap)
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
        dyaw = torch.randn(n, device=self.device, generator=self.gen) * (
            yaw_std.to(self.device) if torch.is_tensor(yaw_std) else yaw_std)
        yaw = yaw + dyaw
        pose = torch.cat([xy, yaw[:, None]], 1)
        min_cl = self.cfg.vehicle.width if min_clearance is None else min_clearance
        # reject poses too close to walls by pulling them back to the centerline
        wall = self.track.sample_edt(xy, tid) < min_cl
        bad = wall | self._spawn_runway_blocked(s, tid, eid, lat) | self._spawn_heading_blocked(pose, tid, eid)
        heavy = getattr(self.track, "has_props", False) or self.spawn_runway > 0.0
        # No props, no runway: only the wall test can reject, and only a free row takes this short
        # path. An abreast row rejected by the walls (its caller's room check is meant to make that
        # impossible) still walks with its race rather than being pulled onto its partner's line.
        formation_bad = lat_fixed is not None and bool((bad & lat_fixed.to(self.device).bool()).any())
        if not heavy and avoid_s is None and not formation_bad:
            if bad.any():
                xy0, yaw0 = self.track.pose_at_s(s[bad], tid[bad])
                pose[bad] = torch.cat([xy0, yaw0[:, None]], 1)
            return pose
        # Props are not in the EDT, so the wall test above cannot see them and a car would spawn
        # standing inside a crate. Reject on the same geometry the contact test uses; if the
        # centerline fallback is itself blocked, walk the lap until a station is clear.
        z = torch.zeros(n, dtype=torch.bool, device=self.device)
        fixed = z if lat_fixed is None else lat_fixed.to(self.device).bool()
        act = ~z if active is None else active.to(self.device).bool()
        L = self.track.length[tid].clamp_min(1e-6)
        if heavy:
            bad = bad | self._spawn_in_prop(pose, tid, eid)
        bad = bad | self._spawn_too_close(s, L, avoid_s, clear_gap)
        own_ok = ~bad
        # A free row may still fall back to the centerline at its own arc: the arc gap to its race
        # mates is unchanged by that, and it is the fallback every run before this one had. A row in
        # a formation may not (see `lat_fixed`).
        free_bad = bad & ~fixed
        if bool(free_bad.any()):
            xy0, yaw0 = self.track.pose_at_s(s[free_bad], tid[free_bad])
            pose[free_bad] = torch.cat([xy0, yaw0[:, None]], 1)
        cen_bad = self._spawn_too_close(s, L, avoid_s, clear_gap)
        if heavy:
            cen_bad = cen_bad | (self._spawn_in_prop(pose, tid, eid) | self._spawn_runway_blocked(s, tid, eid, None)
                                 | self._spawn_heading_blocked(pose, tid, eid))
        still = (fixed & bad) | (free_bad & cen_bad)
        need = still & act
        self._spawn_stat("rejected", int((bad & act).sum()))
        if not bool(need.any()):
            return pose
        grp = torch.arange(n, device=self.device) if group is None else group.to(self.device).long()
        # Every row of a group that has a car to move -- the whole race moves, not the one car.
        moving = torch.isin(grp, grp[need])
        rows = torch.nonzero(moving, as_tuple=True)[0]
        gw, gidx = torch.unique(grp[rows], return_inverse=True)
        try_own = (fixed | own_ok)[rows]
        try_cen = ~fixed[rows]
        J = 32
        st_ok, st_l, own_l, own_s, s_alt, p_own, p_cen = self._spawn_stations(
            rows, s, L, tid, eid, lat, dyaw, min_cl, try_own, try_cen, act[rows], avoid_s, clear_gap, J)
        # Per group: a station is clear when every active car of the group is. Clear with the
        # straight line ahead of each car as well, where some station has it: on a bend the lane
        # runway follows the lane, while a car placed at speed goes straight until its steering
        # comes round -- a car set on a tight left-hander at 2.4 m/s with crates past the outside of
        # it had no path round them at full lock. Only a preference: a lap with no such station
        # keeps the lane rule rather than refusing to spawn.
        G = gw.numel()
        fail_s = torch.zeros(G, J, device=self.device).index_add_(0, gidx, (~st_ok).float())
        fail_l = torch.zeros(G, J, device=self.device).index_add_(0, gidx, (~st_l).float())
        strict, loose = fail_s == 0, fail_l == 0
        use_strict = strict.any(1)
        ok = torch.where(use_strict[:, None], strict, loose)
        first = ok.to(torch.uint8).argmax(1)
        found = ok.any(1)
        k = torch.arange(rows.numel(), device=self.device)
        j = first[gidx]
        own_here = try_own & torch.where(use_strict[gidx], own_s[k, j], own_l[k, j])
        chosen = torch.where(own_here[:, None], p_own[k, j], p_cen[k, j])
        placed = found[gidx]
        pose[rows[placed]] = chosen[placed]
        self._spawn_stat("group_walks", int(found.sum()))
        if bool((~found).any()):
            # No common station anywhere on the lap. The formation is given up rather than the
            # contact: the cars that were clear stay where they are, and each of the others walks
            # on its own, refusing every station within `clear_gap` of a car of its group already
            # placed. Never seen in the boot-camp probe; counted, so a map where it is not rare says so.
            s_out = s.clone()
            s_out[rows[placed]] = s_alt[k, j][placed]
            for g in torch.nonzero(~found, as_tuple=True)[0].tolist():
                mem = rows[gidx == g]
                for r in mem[need[mem]].tolist():
                    mates = mem[(mem != r) & act[mem] & ~need[mem]]
                    av = s_out[mates][None, :]
                    if avoid_s is not None:
                        av = torch.cat([avoid_s[r:r + 1].to(av.dtype), av], 1)
                    rr = torch.tensor([r], device=self.device)
                    p1, s1 = self._spawn_walk_one(rr, s, L, tid, eid, lat, dyaw, min_cl, fixed, av, clear_gap, J)
                    if p1 is None:
                        # crowded: no station clear of the props *and* of the other cars. Clear of
                        # the props is the old guarantee and is kept; the car gap is what gives.
                        self._spawn_stat("crowded", 1)
                        p1, s1 = self._spawn_walk_one(rr, s, L, tid, eid, lat, dyaw, min_cl, fixed, None, 0.0, J)
                    if p1 is None:
                        # Every one of the J stations (L/33 apart: 1.9 m on ICRA 2022) is blocked. A
                        # race that has run a while has shoved its movable crates around, and a car
                        # rejoining it keeps that layout; the clear stretches can lie between the
                        # stations. Look between them, everything else kept (2026-10-17: map3R2_s1321
                        # died here at update 413 of an 8-hour run).
                        self._spawn_stat("dense", 1)
                        p1, s1 = self._spawn_walk_one(rr, s, L, tid, eid, lat, dyaw, min_cl, fixed, None, 0.0,
                                                      4 * J + 3, den=4 * 33.0)
                    if p1 is None:
                        # Still none: no clear 3 m runway anywhere. Standing clear of the props and
                        # the walls is kept; the runway and the heading test give, counted.
                        self._spawn_stat("runway_waived", 1)
                        p1, s1 = self._spawn_walk_one(rr, s, L, tid, eid, lat, dyaw, min_cl, fixed, None, 0.0,
                                                      4 * J + 3, den=4 * 33.0, runway=False)
                    if p1 is None:
                        raise RuntimeError(
                            "1 env has no spawn pose clear of the placed props anywhere on its lap; "
                            "the track's props block the centerline")
                    pose[r] = p1; s_out[r] = s1
                    need[r] = False
                    self._spawn_stat("fallback_rows", 1)
        return pose

    #: Running counts from `sample_spawn` (host ints): cars rejected at their asked pose, races walked
    #: as one, cars placed by the per-car fallback, and those of them no station kept clear of their
    #: race mates. A census, so a map or a config where the fallbacks are not rare is visible.
    spawn_stats: Dict[str, int]

    def _spawn_stat(self, key: str, k: int) -> None:
        """Add to `spawn_stats`, creating it on a simulator put together without `__init__` (a test
        shell): a census must not be what a spawn dies on."""
        st = self.__dict__.get("spawn_stats")
        if st is None:
            st = self.spawn_stats = {"rejected": 0, "group_walks": 0, "fallback_rows": 0, "crowded": 0}
        st[key] = st.get(key, 0) + k

    def _spawn_too_close(self, s: torch.Tensor, L: torch.Tensor, avoid_s, gap: float) -> torch.Tensor:
        """(n,) whether arc `s` is within `gap` of any of the row's `avoid_s` (NaN entries ignored)."""
        if avoid_s is None or gap <= 0.0:
            return torch.zeros(s.shape[0], dtype=torch.bool, device=self.device)
        a = avoid_s.to(self.device, s.dtype)
        d = (a - s[:, None] + L[:, None] / 2) % L[:, None] - L[:, None] / 2
        return (d.abs() < gap).any(1)

    def _spawn_stations(self, rows, s, L, tid, eid, lat, dyaw, min_cl, try_own, try_cen, act,
                        avoid_s, clear_gap: float, J: int, den: float = 33.0, runway: bool = True):
        """Every row of `rows` tested at J stations L/den apart ahead of its asked arc (L/33 unless
        the fallback looks between them). `runway=False` drops the runway and heading tests and keeps
        walls and props: the last resort of a lap with no clear runway anywhere.

        Two candidates per station: the row's own pose moved along the lane (its lateral offset and
        heading jitter kept, tested against everything the asked pose was: walls at `min_cl`, the
        runway at that offset, props, a prop on its heading) and the centerline pose there (props,
        the centerline runway, the lane heading). Returns, per (row, station): clear with the
        heading test (strict) and without it (lane), for whichever candidates the row may take; the
        own candidate's two verdicts; the station arc; both candidate poses. Inactive rows pass.

        All J stations in one call per test rather than one call each: same answer, but with a
        per-env obstacle layout the loop was thirty-two Python-looped SAT passes over the slot
        dimension per reset, measured at 46 ms of a 116 ms step at `--envs 256`.
        """
        R = rows.numel()
        heavy = getattr(self.track, "has_props", False) or self.spawn_runway > 0.0
        ar_j = torch.arange(1, J + 1, device=self.device, dtype=L.dtype)
        s_alt = (s[rows][:, None] + L[rows][:, None] * (ar_j[None] / den)) % L[rows][:, None]   # (R, J)
        sf = s_alt.reshape(-1)
        rep = lambda t_: t_[rows].repeat_interleave(J)
        tid_j = rep(tid)
        eid_j = None if eid is None else rep(eid)
        lat_j = rep(lat); dyaw_j = rep(dyaw)
        mc = rep(min_cl) if torch.is_tensor(min_cl) else min_cl
        xy_c, yaw_c = self.track.pose_at_s(sf, tid_j)
        nrm = torch.stack([-torch.sin(yaw_c), torch.cos(yaw_c)], 1)
        p_own = torch.cat([xy_c + nrm * lat_j[:, None], (yaw_c + dyaw_j)[:, None]], 1)
        p_cen = torch.cat([xy_c, yaw_c[:, None]], 1)
        L_j = rep(L)
        av_j = None if avoid_s is None else avoid_s[rows].repeat_interleave(J, 0)
        close = self._spawn_too_close(sf, L_j, av_j, clear_gap)
        own_l = ~(close | (self.track.sample_edt(p_own[:, :2], tid_j) < mc))
        cen_l = ~close
        own_hd = cen_hd = torch.zeros_like(close)
        if heavy and runway:
            own_l = own_l & ~(self._spawn_runway_blocked(sf, tid_j, eid_j, lat_j) | self._spawn_in_prop(p_own, tid_j, eid_j))
            cen_l = cen_l & ~(self._spawn_runway_blocked(sf, tid_j, eid_j, None) | self._spawn_in_prop(p_cen, tid_j, eid_j))
            own_hd = self._spawn_heading_blocked(p_own, tid_j, eid_j)
            cen_hd = self._spawn_heading_blocked(p_cen, tid_j, eid_j)
        elif heavy:
            own_l = own_l & ~self._spawn_in_prop(p_own, tid_j, eid_j)
            cen_l = cen_l & ~self._spawn_in_prop(p_cen, tid_j, eid_j)
        v = lambda t_: t_.view(R, J)
        own_l, cen_l, own_hd, cen_hd = v(own_l), v(cen_l), v(own_hd), v(cen_hd)
        own_s, cen_s = own_l & ~own_hd, cen_l & ~cen_hd
        to, tc, ia = try_own[:, None], try_cen[:, None], ~act[:, None]
        st_ok = ia | (to & own_s) | (tc & cen_s)
        st_l = ia | (to & own_l) | (tc & cen_l)
        return st_ok, st_l, own_l, own_s, s_alt, p_own.view(R, J, 3), p_cen.view(R, J, 3)

    def _spawn_walk_one(self, rr, s, L, tid, eid, lat, dyaw, min_cl, fixed, avoid, gap: float, J: int,
                        den: float = 33.0, runway: bool = True):
        """One car's own lap walk (the fallback): first clear station's (pose, arc), or (None, None).
        `den` and `runway` as in `_spawn_stations`."""
        avoid_full = None
        if avoid is not None:
            avoid_full = torch.full((s.shape[0], avoid.shape[1]), float("nan"), device=self.device, dtype=s.dtype)
            avoid_full[rr] = avoid.to(s.dtype)
        f = fixed[rr]
        st_ok, st_l, own_l, own_s, s_alt, p_own, p_cen = self._spawn_stations(
            rr, s, L, tid, eid, lat, dyaw, min_cl, torch.ones_like(f), ~f,
            torch.ones_like(f), avoid_full, gap, J, den=den, runway=runway)
        ok = st_ok if bool(st_ok.any()) else st_l
        if not bool(ok.any()):
            return None, None
        j = int(ok[0].to(torch.uint8).argmax())
        strict = ok is st_ok
        use_own = bool((own_s if strict else own_l)[0, j])
        return (p_own if use_own else p_cen)[0, j], s_alt[0, j]

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
        # Half a body width from the line, measured *between* the samples too: an object whose
        # nearest point falls half a step from both neighbours is `hypot(need, step/2)` from the
        # nearer one. Checking `need` alone let a centre piece 0.144 m off the line through (need
        # 0.155 m) because its closest approach sat between two samples.
        need = math.hypot(0.5 * self.cfg.vehicle.width, 0.5 * step)
        blocked = self.track.sample_edt(xy, tid_k) < need
        props = getattr(self.track, "env_props", None)
        if props is not None:
            eid_k = (torch.arange(n, device=self.device) if eid is None else eid)
            eid_k = eid_k[:, None].expand(n, K).reshape(-1)
            # props keep `SPAWN_PROP_GAP` from the body's swept width as well (see
            # `_spawn_heading_blocked`): a car set on the centreline at 2.9 m/s with a crate 0.25 m
            # off its front corner's path, on a bending lane, touched it five steps in
            need_p = math.hypot(0.5 * self.cfg.vehicle.width + self.SPAWN_PROP_GAP, 0.5 * step)
            blocked = blocked | (props.clearance(xy, eid_k) < need_p)
        return blocked.view(n, K).any(1)

    #: [m] how close to a prop a car may be placed, when a spawn runway is required (see `_spawn_in_prop`).
    SPAWN_PROP_GAP = 0.25

    def _spawn_heading_blocked(self, pose: torch.Tensor, tid: torch.Tensor, eid) -> torch.Tensor:
        """(n,) whether a prop stands on the straight line the car is *pointing* along.

        `_spawn_runway_blocked` walks the lane at the car's lateral offset, but the pose it is
        checking carries `spawn_yaw_std` of heading noise on top of the lane's direction -- and a
        lane that bends away from the raceline adds more. Measured on 2026-09-24: a car placed
        0.43 m off the raceline at 2.9 m/s, about 40 degrees off it, with a crate 0.34 m from its
        nose on the line it was pointing along and a clear lane beside it. Nothing could have
        avoided that crate; the lane check passed it.

        Props only, over the same `spawn_runway`: a wall along the heading is a long surface the car
        can steer off, the lane check already keeps the lane clear of it, and a rejected pose
        falls back to the lane's own heading, where the two checks coincide.
        """
        n = pose.shape[0]
        props = getattr(self.track, "env_props", None)
        if self.spawn_runway <= 0.0 or props is None:
            return torch.zeros(n, dtype=torch.bool, device=self.device)
        K = max(6, int(math.ceil(self.spawn_runway / (0.5 * self.cfg.vehicle.width))))
        step = self.spawn_runway / K
        ds = torch.arange(1, K + 1, device=self.device, dtype=pose.dtype) * step
        hd = torch.stack([torch.cos(pose[:, 2]), torch.sin(pose[:, 2])], 1)
        xy = (pose[:, None, :2] + ds[None, :, None] * hd[:, None]).reshape(-1, 2)
        # The body's swept width plus the same sideways gap `_spawn_in_prop` keeps: a crate just
        # ahead of a front corner, 0.25 m clear of the straight path at 2.9 m/s, was hit five steps
        # later by a car turning towards the line.
        need = math.hypot(0.5 * self.cfg.vehicle.width + self.SPAWN_PROP_GAP, 0.5 * step)
        eid_k = (torch.arange(n, device=self.device) if eid is None else eid)
        eid_k = eid_k[:, None].expand(n, K).reshape(-1)
        return (props.clearance(xy, eid_k) < need).view(n, K).any(1)

    def _spawn_in_prop(self, pose: torch.Tensor, tid: torch.Tensor,
                       eid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Whether each candidate spawn pose has its footprint overlapping a prop -- or, when a
        runway is required, standing within `SPAWN_PROP_GAP` of one.

        The gap is the runway's rule applied sideways. Measured 2026-09-24: a car placed 0.23 m
        beside a crate, clear of it and of its runway, touched it within five steps at walking
        pace -- turning towards the line swings the tail outwards, into the crate. Nothing could
        have done otherwise from that pose, and it counted as the driver's contact.

        The eager `prism_contacts`, deliberately: this runs at reset on whatever partial batch is
        resetting, so a `dynamic=False` compile would recompile for every distinct size."""
        from .prop_math import prism_contacts
        c, s_ = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        R = torch.stack([torch.stack([c, -s_], -1), torch.stack([s_, c], -1)], -2)
        pts = torch.einsum("bij,kj->bki", R, self.corners) + pose[:, None, :2]
        poses, pn, pd, z_lo, z_hi = self.track.props_near(tid, pose[:, :2], eid, self.prop_reach)
        h = self.car_dims[:pose.shape[0], 2] if self.car_dims.shape[0] >= pose.shape[0] else 0.25
        depth, _, _ = prism_contacts(pts[:, [0, 1, 3, 2]], poses, pn, pd, z_lo, z_hi, h)
        inside = depth > 0
        props = getattr(self.track, "env_props", None)
        if self.spawn_runway <= 0.0 or props is None:
            return inside
        # the outline: corners and edge midpoints, and the prop distance there
        mid = 0.5 * (pts + pts[:, [1, 3, 0, 2]])
        outline = torch.cat([pts, mid], 1)                                   # (n, 8, 2)
        n = pose.shape[0]
        e = (torch.arange(n, device=self.device) if eid is None else eid)[:, None].expand(n, 8).reshape(-1)
        near = props.clearance(outline.reshape(-1, 2), e).view(n, 8).amin(1) < self.SPAWN_PROP_GAP
        return inside | near

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
        self.wall_touched[env_ids] = False
        self.wall_engaged[env_ids] = False
        self.impact_kind[env_ids] = 0.0
        self.kind_prev[env_ids] = False
        self.contact_pt[env_ids] = 0.0
        self.contact_pt_sev[env_ids] = 0.0
        self.contact_now[env_ids] = 0.0
        self.contact_held[env_ids] = 0.0
        self.contact_counts[env_ids] = 0
        self.contact_acc[env_ids] = 0.0
        self.car_touched[env_ids] = False
        self.car_impact_roll[env_ids] = 0.0
        self.wall_impact[env_ids] = 0.0
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
                                                     "wall_touched", "wall_engaged", "wall_impact",
                                                     "impact_kind", "kind_prev", "contact_acc",
                                                     "contact_pt", "contact_pt_sev", "contact_now",
                                                     "contact_held", "contact_counts",
                                                     "car_touched", "car_impact_roll", "_odom_t_prev")}
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
        if self.absent_any:
            # a parked car (an empty seat, `park`): no command, no motion
            self.cmd = torch.where(self.absent[:, None], torch.zeros_like(self.cmd), self.cmd)
            frozen = frozen | self.absent
        self.cmd_hist = torch.cat([self.cmd[:, None, :], self.cmd_hist[:, :-1]], 1)

        self.pose_prev = self.state[:, :3].clone()
        # Roll/pitch are alternate columns. A strided view avoids constructing a
        # CUDA index tensor from a Python list (and synchronizing) every step.
        self.att_prev = self.att[:, :3:2].clone()
        # the sample layout belongs to this step's phase; capture it before the phase advances
        imu_offsets = self._imu_offsets[self._imu_phase]
        # In place, before the roll records into it: this step's impacts and nothing older -- not a
        # warm-up's, not a graph capture's throw-away runs.
        self.wall_impact.zero_()
        self.impact_kind.zero_()
        self.contact_pt_sev.zero_()
        if self._car_response_on():
            self.car_impact_roll.zero_()
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
        car_impact = None
        if self.M > 1:
            if self.measure_impacts:
                self.car_collision, car_impact = self._car_contacts(state, impact=True)
            else:
                self.car_collision = self._car_contacts(state)
            if self._car_response_on():
                # Every car contact the roll resolved, not only an overlap left at the step's end (the
                # response pushes the cars apart within milliseconds), and its severity is the closing
                # speed the impulses took. Not the step-end reading: after an impulse the bodies separate,
                # and a corner hit leaves them spinning -- a 3 m/s corner-to-corner rear-end read 5.5 m/s
                # there, the spin of the corners, not the hit.
                self.car_collision = self.car_collision | self.car_touched
                self.car_touched.zero_()
                if car_impact is not None:
                    car_impact = self.car_impact_roll.clone()
            cars = self._car_boxes(state)
        wall_dist, hit, odom_state, s, ds, lateral, lap, cl_idx = (
            t.clone() for t in self._post(state, self.odom.state, self.cmd, self.s, self.lap, self.tid, P, self.cl_idx))
        self.cl_idx = cl_idx
        if self._wall_latch_on():
            # Every wall contact the roll resolved (`SimParams.wall_latch`), not only the ones still in
            # the wall cell at the end of the step. In place, for the same reason as `prop_touched`.
            hit = hit | self.wall_touched
            self.wall_touched.zero_()
            # And a contact ends when the footprint is a grid cell clear, not at the first resolution
            # that finds it outside the wall cell. The test reads a nearest-cell distance field, so the
            # penetration it sees is 0 or half a cell: a car resting on the hose is pushed 6 mm out,
            # creeps back across the cell edge and was a fresh contact every 50 ms (a 1.5 m/s hit and
            # stop: four onsets, at 1.49, 0.11, 0.05 and 0.01 m/s). Gaps under a cell the grid cannot
            # tell from contact; a car that backs a cell away and comes back has two contacts.
            hit = hit | (self.wall_engaged & (wall_dist < self.track.t_res[self.tid]))
            self.wall_engaged.copy_(hit & ~self.absent if self.absent_any else hit)
        wall_only = hit
        prop_now = None
        if getattr(self.track, "has_props", False):
            # `_post` reads the occupancy EDT and props are not in it, so without this a car drives
            # through a box: `terminate_on_collision` never fires, and with soft walls the contact is
            # resolved by `_resolve_wall_contact` yet never reported. `prop_touched` also carries the
            # substeps, because a box can be entered and pushed back out inside one control step.
            prop_now = self.prop_touched | (self._prop_contact(state)[0] > 0)
            hit = hit | prop_now
        self.prop_touched.zero_()          # in place: `warmup` restores by copy_ into this tensor
        if self.absent_any:
            hit = hit & ~self.absent
            if self.M > 1:
                self.car_collision = self.car_collision & ~self.absent
        # Walls and props alone, before the cars are ORed in: the wall constraint counts these, and read
        # through `collision` a car contact was charged at the wall's price as well as the car's.
        wall_hit = hit
        # Impact severities (`StepResult.wall_impact`, `car_impact`). Soft contact recorded its own in
        # the roll; a terminate crash is not resolved, so its state at the end of the step is the
        # impact. Read-only either way: nothing here feeds back into the physics.
        if not self.cfg.sim.terminate_on_collision:
            wall_impact = self.wall_impact.clone()
        elif self.measure_impacts:
            wall_impact = torch.where(wall_hit, self._wall_contact_speed(state), torch.zeros_like(self.wall_impact))
        else:
            wall_impact = None
        if self.absent_any:
            if wall_impact is not None:
                wall_impact = torch.where(self.absent, torch.zeros_like(wall_impact), wall_impact)
            if car_impact is not None:
                car_impact = torch.where(self.car_collision, car_impact, torch.zeros_like(car_impact))
        contact_onsets = contact_impacts = None
        # Onsets per kind. With the wall contact held until the car is a cell clear, a crate touched while a
        # hose is being scraped is no rising edge of the merged flag; counted per kind it is its own contact,
        # charged and filed with its own impact. Kept in every mode for the console (`contact_frame`); handed
        # to the env and the meters only with `wall_latch`, so a legacy run charges what it always did.
        z = torch.zeros_like(wall_only)
        kinds = torch.stack([wall_only, z if prop_now is None else prop_now,
                             self.car_collision if self.M > 1 else z], 1)
        if self.absent_any:
            kinds = kinds & ~self.absent[:, None]
        onsets_k = kinds & ~self.kind_prev
        self.kind_prev.copy_(kinds)
        if car_impact is not None:
            car_sev = car_impact
        elif self.M > 1 and self._car_response_on():
            car_sev = self.car_impact_roll
        else:
            car_sev = torch.zeros_like(self.wall_impact)
        impacts_k = torch.cat([self.impact_kind, car_sev[:, None].to(self.impact_kind.dtype)], 1)
        if self.absent_any:
            impacts_k = torch.where(self.absent[:, None], torch.zeros_like(impacts_k), impacts_k)
        self._update_contact_frame(kinds, onsets_k, impacts_k)
        if self._wall_latch_on():
            contact_onsets, contact_impacts = onsets_k, impacts_k
        if self.M > 1:
            hit = hit | self.car_collision
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
                          odom_t, i_motor if self.wheel_model else None, wall_hit.clone(),
                          wall_impact, car_impact, contact_onsets, contact_impacts)

    def _record_contact_point(self, pt: torch.Tensor, sev: torch.Tensor, touching: torch.Tensor) -> None:
        """Keep `pt` as this step's contact point where a contact touches at least as hard as the step's hardest
        so far (`contact_pt`, `contact_pt_sev`, in place -- the roll is captured with it)."""
        better = touching & (sev.to(self.contact_pt_sev.dtype) >= self.contact_pt_sev)
        self.contact_pt.copy_(torch.where(better[:, None], pt.to(self.contact_pt.dtype), self.contact_pt))
        self.contact_pt_sev.copy_(torch.where(better, sev.to(self.contact_pt_sev.dtype), self.contact_pt_sev))

    def contact_frame(self, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(n, len(CONTACT_FRAME_COLUMNS)) float: what the console needs to draw each car's contacts (see the
        columns). One tensor so the console's single device-to-host copy can carry it."""
        out = torch.cat([self.contact_now, self.contact_held, self.contact_counts.to(self.contact_now.dtype)], 1)
        return out if rows is None else out[rows]

    def _update_contact_frame(self, kinds: torch.Tensor, onsets: torch.Tensor, impacts: torch.Tensor) -> None:
        """The console's per-car contact record, in place (`contact_frame`): kinds/onsets (B, 3) bool and
        impacts (B, 3) [m/s] per kind (wall, prop, car), this step's."""
        from .contact_severity import HARD_MPS
        sev_k = torch.where(kinds, impacts, torch.full_like(impacts, -1.0))
        # the hardest kind touching; ties go to the later column (car over prop over wall)
        best = (sev_k + torch.arange(3, device=sev_k.device, dtype=sev_k.dtype) * 1e-6).argmax(1)
        any_ = kinds.any(1)
        kind = torch.where(any_, best + 1, torch.zeros_like(best)).to(self.contact_now.dtype)
        sev = torch.where(any_, sev_k.max(1).values.clamp_min(0.0), torch.zeros_like(kind))
        pt = torch.where(any_[:, None], self.contact_pt, torch.zeros_like(self.contact_pt))
        self.contact_now.copy_(torch.cat([kind[:, None], pt, sev[:, None]], 1))
        held = self.contact_held
        left = (held[:, 0] - self.control_dt).clamp_min(0.0)
        # a contact this step (re)starts the hold; it takes the held record when it is the hold's hardest
        take = any_ & ((held[:, 0] <= 0.0) | (sev >= held[:, 4]))
        rec = torch.cat([torch.full_like(sev, CONTACT_HOLD_S)[:, None], kind[:, None], pt, sev[:, None]], 1)
        new = torch.where(take[:, None], rec, held)
        new[:, 0] = torch.where(any_, torch.full_like(left, CONTACT_HOLD_S), left)
        self.contact_held.copy_(new)
        hard = onsets & (impacts >= HARD_MPS)
        self.contact_counts.add_(torch.cat([onsets, hard], 1).to(self.contact_counts.dtype))

    def _car_response_on(self) -> bool:
        """`SimParams.car_response` in a race under soft contact. Python-level, so `graph_fastpath` guards it."""
        return (self.M > 1 and bool(getattr(self.cfg.sim, "car_response", False))
                and not self.cfg.sim.terminate_on_collision and getattr(self, "car_touched", None) is not None)

    #: [m] the most a car-car resolution moves one car to separate two bodies. A contact found at its first
    #: resolution is at most (closing speed x 5 ms) deep -- 4.5 cm at 9 m/s; a deeper overlap (a placement
    #: that left two cars inside each other) is undone over a few resolutions rather than in one jump.
    CAR_PUSH_MAX = 0.05

    def _resolve_car_contact(self, state: torch.Tensor) -> torch.Tensor:
        """Soft car-car contact (`SimParams.car_response`): one resolution, every car against the other
        cars of its race. Returns the new state; records the contact (`car_touched`) and the closing speed
        the impulse took (`car_impact_roll`) in place.

        Each car is the body the other cars' LiDAR draws (`_car_contacts`: the footprint scaled by its
        `car_dims`, the rear detection box on the back). For each pair, by SAT on the four box axes:

        * the normal is the axis the bodies entered on -- the youngest overlap at the current closing rate
          (penetration / closing speed), as `_car_closing_speed` measures it -- or, with nothing closing,
          the axis of least overlap; the contact point is the mean of the vertices of each box inside the
          other; the relative velocity is taken at that point, yaw rate included;
        * the impulse is the rigid two-body one, J = -(1+e) v_n / (1/m_a + 1/m_b + (r_a x n)^2/I_a +
          (r_b x n)^2/I_b), each car's own mass and yaw inertia from its parameters, plus a tangential
          impulse up to `body_friction` x J; e is `car_restitution`;
        * each car is moved half the penetration out along the normal (at most `CAR_PUSH_MAX`).

        Every row computes its pair from the pair's state alone, in a fixed order (the lower env row is
        body a), so the two rows of a pair compute the same impulse bit for bit and each applies its own
        half: momentum is conserved exactly, nothing is scattered, the partner rows (`other_idx`) are
        static, nothing is read back to the host -- the resolution runs inside the compiled roll and is
        captured with it."""
        sp = self.cfg.sim
        e = float(getattr(sp, "car_restitution", 0.2))
        mu = float(getattr(sp, "body_friction", 0.3))
        yaw = state[:, dyn.IYAW]
        c_, s_ = torch.cos(yaw), torch.sin(yaw)
        R = torch.stack([torch.stack([c_, -s_], -1), torch.stack([s_, c_], -1)], -2)
        local = self.corners[None] * (self.car_dims[:, 0] / 0.50)[:, None, None]
        pts = torch.einsum("bij,bkj->bki", R, local) + state[:, None, :2]
        ext = torch.stack([-self.car_rear[:, 0] * c_, -self.car_rear[:, 0] * s_], 1)
        pts = torch.cat([pts, pts[:, 2:] + ext[:, None, :]], 1)                     # (B,6,2), as `_car_contacts`
        ex = torch.stack([c_, s_], 1)
        ey = torch.stack([-s_, c_], 1)
        xy = state[:, :2]
        w = state[:, dyn.IR]
        v = torch.stack([state[:, dyn.IVX] * c_ - state[:, dyn.IVY] * s_,
                         state[:, dyn.IVX] * s_ + state[:, dyn.IVY] * c_], 1)       # world velocity
        m, Iz = self.P["m"], self.P["Iz"]
        perp = lambda r: torch.stack([-r[:, 1], r[:, 0]], 1)
        cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        me = self.eid
        dv = torch.zeros_like(v)
        dw = torch.zeros_like(w)
        dx = torch.zeros_like(xy)
        hit = torch.zeros_like(w, dtype=torch.bool)
        sev = torch.zeros_like(w)
        pt = torch.zeros_like(xy)
        for c in range(self.M - 1):
            o = self.other_idx[:, c]
            lo, hi = torch.minimum(me, o), torch.maximum(me, o)
            a_side = me < o                                           # this row is body a of its pair
            PA, PB = pts[lo], pts[hi]
            axes = (ex[lo], ey[lo], ex[hi], ey[hi])
            pa = [(PA * ax_[:, None, :]).sum(-1) for ax_ in axes]
            pb = [(PB * ax_[:, None, :]).sum(-1) for ax_ in axes]
            loa, hia = [x.min(1).values for x in pa], [x.max(1).values for x in pa]
            lob, hib = [x.min(1).values for x in pb], [x.max(1).values for x in pb]
            sep = torch.zeros_like(hit)
            for k in range(4):
                sep = sep | (hia[k] < lob[k]) | (hib[k] < loa[k])
            over = ~sep
            # the contact point: a's vertices inside b (b's own axes) and b's inside a
            a_in_b = (pa[2] >= lob[2][:, None]) & (pa[2] <= hib[2][:, None]) \
                & (pa[3] >= lob[3][:, None]) & (pa[3] <= hib[3][:, None])
            b_in_a = (pb[0] >= loa[0][:, None]) & (pb[0] <= hia[0][:, None]) \
                & (pb[1] >= loa[1][:, None]) & (pb[1] <= hia[1][:, None])
            wa, wb = a_in_b.to(xy.dtype), b_in_a.to(xy.dtype)
            cnt = wa.sum(1) + wb.sum(1)
            p = torch.where((cnt > 0)[:, None],
                            ((wa[..., None] * PA).sum(1) + (wb[..., None] * PB).sum(1)) / cnt.clamp_min(1.0)[:, None],
                            0.5 * (xy[lo] + xy[hi]))
            ra, rb = p - xy[lo], p - xy[hi]
            v_rel = (v[hi] + w[hi][:, None] * perp(rb)) - (v[lo] + w[lo][:, None] * perp(ra))   # b relative to a
            # the normal, pointing from a to b: the entry axis, or least overlap when nothing closes
            best_t = torch.full_like(w, float("inf"))
            n = torch.zeros_like(xy); pen = torch.zeros_like(w)
            least = torch.full_like(w, float("inf"))
            n_l = torch.zeros_like(xy)
            for k in range(4):
                b_up = (lob[k] + hib[k]) > (loa[k] + hia[k])          # b on the +axis side of a
                depth = torch.where(b_up, hia[k] - lob[k], hib[k] - loa[k]).clamp_min(0.0)
                n_k = axes[k] * torch.where(b_up, 1.0, -1.0)[:, None]
                closing = -(v_rel * n_k).sum(1)
                t = torch.where(closing > 1e-3, depth / closing.clamp_min(1e-3), torch.full_like(w, float("inf")))
                better = t < best_t
                best_t = torch.where(better, t, best_t)
                n = torch.where(better[:, None], n_k, n)
                pen = torch.where(better, depth, pen)
                fewer = depth < least
                least = torch.where(fewer, depth, least)
                n_l = torch.where(fewer[:, None], n_k, n_l)
            entered = torch.isfinite(best_t)
            n = torch.where(entered[:, None], n, n_l)
            pen = torch.where(entered, pen, least)
            vn = (v_rel * n).sum(1)                                   # < 0: closing
            ran, rbn = cross(ra, n), cross(rb, n)
            ml, mh, Il, Ih = m[lo], m[hi], Iz[lo], Iz[hi]
            Jn = torch.where(over, (1 + e) * (-vn).clamp_min(0.0)
                             / (1.0 / ml + 1.0 / mh + ran * ran / Il + rbn * rbn / Ih), torch.zeros_like(w))
            t_ = perp(n)
            rat, rbt = cross(ra, t_), cross(rb, t_)
            Jt = -(v_rel * t_).sum(1) / (1.0 / ml + 1.0 / mh + rat * rat / Il + rbt * rbt / Ih)
            Jt = torch.maximum(torch.minimum(Jt, mu * Jn), -mu * Jn)
            imp = Jn[:, None] * n + Jt[:, None] * t_                  # on b; a takes -imp
            sgn = torch.where(a_side, -1.0, 1.0)
            my_imp = sgn[:, None] * imp
            dv = dv + my_imp / m[:, None]
            dw = dw + cross(torch.where(a_side[:, None], ra, rb), my_imp) / Iz
            push = torch.where(over, (0.5 * pen).clamp(max=self.CAR_PUSH_MAX), torch.zeros_like(pen))
            dx = dx + sgn[:, None] * n * push[:, None]
            hit = hit | over
            sev_c = torch.where(over, (-vn).clamp_min(0.0), torch.zeros_like(vn))
            pt = torch.where((over & (sev_c >= sev))[:, None], p, pt)
            sev = torch.maximum(sev, sev_c)
        self.car_touched.logical_or_(hit)
        self.car_impact_roll.copy_(torch.maximum(self.car_impact_roll, sev.to(self.car_impact_roll.dtype)))
        self._record_contact_point(pt, sev, hit)
        v2 = v + dv
        st = state.clone()
        st[:, :2] = xy + dx
        st[:, dyn.IVX] = v2[:, 0] * c_ + v2[:, 1] * s_
        st[:, dyn.IVY] = -v2[:, 0] * s_ + v2[:, 1] * c_
        st[:, dyn.IR] = w + dw
        return st

    def _props_move_in_roll(self) -> bool:
        """`SimParams.prop_relative` with movable props under soft contact: a struck prop is a second body,
        moved in place inside the roll (`ProceduralObstacles.drift` / `push`). Python-level, guarded."""
        return (bool(getattr(self.cfg.sim, "prop_relative", False)) and bool(getattr(self, "movable_obstacles", False))
                and not self.cfg.sim.terminate_on_collision
                and getattr(self.track, "env_props", None) is not None)

    def _wall_latch_on(self) -> bool:
        """`SimParams.wall_latch` under soft contact. Read at Python level inside the roll, so it is
        part of a captured graph's structure (`graph_fastpath` guards it)."""
        return (bool(getattr(self.cfg.sim, "wall_latch", False)) and not self.cfg.sim.terminate_on_collision
                and getattr(self, "wall_touched", None) is not None)

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
                                           quantise=self.wheel_model, yaw_rate_true=state[:, dyn.IR])
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
        # road_tilt whatever the substep. OFF by default -- it is noise in time, not dynamics, and
        # it was what rocked the car and its scan plane at a constant speed; see
        # `VehicleParams.road_tilt`. At 0 it is not run at all rather than run times zero.
        tilt_on = float(self.cfg.vehicle.road_tilt) > 0.0
        ou_a = (self.dt / P["road_tau"]).clamp(0.0, 1.0)
        ou_k = 1.0 - ou_a
        ou_s = P["road_tilt"] * torch.sqrt(1.0 - ou_k * ou_k)
        imu_on = self.cfg.imu.enabled
        wheel_on = self.wheel_model
        r_vec = torch.stack([P["imu_x"] - P["lr"], P["imu_y"], P["imu_z"] - P["h"]], 1)
        samples = []
        a_cmd_last = torch.zeros_like(ax)
        soft_wall = not self.cfg.sim.terminate_on_collision
        props_move = soft_wall and self._props_move_in_roll()
        cars = soft_wall and self._car_response_on()
        # `SimParams.contact_moments`: a contact turns the body about its CoG at the height it pushes. Per env:
        # [rad/s^2] of roll / pitch per [m/s^2] of contact acceleration across / along the car.
        moments = soft_wall and bool(getattr(self.cfg.sim, "contact_moments", False))
        if moments:
            k_roll = (P["h"] - P["contact_z_side"]) * P["m"] / P["Ixx"]
            k_pitch = (P["h"] - P["contact_z_end"]) * P["m"] / P["Iyy"]
        # `SimParams.contact_accel_hold`: the contact's acceleration over its whole window, carried across
        # control steps in `contact_acc` (the last window of a step reaches into the next one).
        hold = soft_wall and bool(getattr(self.cfg.sim, "contact_accel_hold", False)) \
            and getattr(self, "contact_acc", None) is not None
        a_hold = self.contact_acc if hold else None
        ar = torch.arange(self.B, device=state.device)
        for k in range(self.substeps):
            age = delay_s - (k + 1) * self.dt
            idx = torch.ceil(age / self.control_dt).long().clamp(0, self.hist_len - 1)
            eff = cmd_hist[ar, idx]                                   # (B,2) command in effect
            # the speed channel's own extra delay (`motor_delay`, 0 = the same command)
            idx_m = torch.ceil((age + P["motor_delay"]) / self.control_dt).long().clamp(0, self.hist_len - 1)
            eff = torch.stack([eff[:, 0], cmd_hist[ar, idx_m, 1]], 1)
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
            # Every `contact_every` substeps, over the time since the last one -- so the velocity
            # bled off per second, and with it the calibrated 40 ms duct contact, is unchanged; only
            # how finely it is integrated is. It has to be coarser than every substep because inside
            # a CUDA graph the resolution cannot early-out on "nothing touching", so its prop SAT ran
            # 25 times a control step whether anything was near or not: in the console, on the
            # user's own scenario (`real/iccas25#line:*!assets=mixed:1`), that was 41.6 ms a step
            # against terminate's 23.7 -- 0.60x realtime against 1.05x. `k` is a Python int, so the
            # schedule is fixed per substep at capture and costs a graph nothing.
            ce = self.contact_every
            if soft_wall and (k % ce == ce - 1 or k == self.substeps - 1):
                span = (k % ce) + 1
                if props_move:
                    # the props slide over the same window, so the contact sees them where they are, and a
                    # prop shoved into a wall stops at it
                    self.track.env_props.drift(self.dt * span, self.tid)
                v_pre = state[:, dyn.IVX:dyn.IVY + 1]
                state, a_contact = self._resolve_wall_contact(state, dt=self.dt * span)
                if cars:
                    v_wall = state[:, dyn.IVX:dyn.IVY + 1]
                    state = self._resolve_car_contact(state)
                    if not hold:
                        # the per-substep form (`contact_accel_hold` off), clipped the way it clips the walls
                        a_car = (state[:, dyn.IVX:dyn.IVY + 1] - v_wall) / (self.dt * span)
                        mag = a_car.norm(dim=1, keepdim=True).clamp_min(1e-9)
                        a_contact = a_contact + a_car * (mag.clamp(max=self.cfg.imu.accel_range) / mag)
                if hold:
                    # Everything this window's contacts did to the velocity, spread over the window and
                    # applied for its length -- so the springs and the accelerometer integrate the whole
                    # change. Not clipped here: the body takes whatever the contact gives it, and the
                    # accelerometer clips at its range when it is sampled (`imu.sample`).
                    a_hold = (state[:, dyn.IVX:dyn.IVY + 1] - v_pre) / (self.dt * span)
            if hold:
                a_contact = a_hold
            a_x = ax if a_contact is None else ax + a_contact[:, 0]
            a_y = ay if a_contact is None else ay + a_contact[:, 1]
            # sprung mass: roll to the outside of the corner, dive under braking, squat under throttle
            # The floor only shakes a car that is moving over it: `road_tilt` was measured driving,
            # and at a standstill the excitation is gone. The OU state is not reset -- it decays
            # through `ou_k` with the road's own time constant, which is the suspension settling
            # rather than the body snapping level. See `VehicleParams.road_tilt_v`.
            if tilt_on:
                excite = (state[:, dyn.IVX].abs() / P["road_tilt_v"].clamp_min(1e-6)).clamp(0.0, 1.0)
                w_roll = w_roll * ou_k + torch.randn_like(w_roll) * ou_s * excite
                w_pitch = w_pitch * ou_k + torch.randn_like(w_pitch) * ou_s * excite
            # The per-g gains are the tyres' (a force at the ground, through the suspension); with
            # `contact_moments` a contact's acceleration turns the body at its own height instead (below).
            s_x, s_y = (ax, ay) if moments else (a_x, a_y)
            roll_ss = P["roll_per_g"] * s_y / dyn.G + w_roll
            # asymmetric: the car squats under throttle far more than it dives under (regen-limited) braking
            pitch_ss = -torch.where(s_x > 0, P["pitch_per_g"], P["dive_per_g"]) * s_x / dyn.G + w_pitch
            roll_acc = wn * wn * (roll_ss - roll) - 2 * zeta * wn * roll_rate
            pitch_acc = wn * wn * (pitch_ss - pitch) - 2 * zeta * wn * pitch_rate
            if moments and a_contact is not None:
                # (h - z) F / I about the CoG, F = m a: a push across the car at the flank height rolls it, one
                # along it at the bumper height pitches it -- nose down when it is stopped from the front.
                roll_acc = roll_acc + k_roll * a_contact[:, 1]
                pitch_acc = pitch_acc - k_pitch * a_contact[:, 0]
            roll_rate = roll_rate + roll_acc * self.dt
            pitch_rate = pitch_rate + pitch_acc * self.dt
            if moments:
                # The suspension's travel ends at +-0.35 rad: there the body stops turning further, so the
                # rate into the stop is taken out (the gyro read a rate the clamped angle never followed).
                # Hard contacts reach it now; the tyres alone never did.
                roll_rate = torch.where((roll + roll_rate * self.dt).abs() >= 0.35,
                                        torch.where(roll_rate * roll > 0, torch.zeros_like(roll_rate), roll_rate), roll_rate)
                pitch_rate = torch.where((pitch + pitch_rate * self.dt).abs() >= 0.35,
                                         torch.where(pitch_rate * pitch > 0, torch.zeros_like(pitch_rate), pitch_rate), pitch_rate)
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
        if hold:
            self.contact_acc.copy_(a_hold)                 # in place: a captured graph keeps writing it
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
        if self._wall_latch_on():
            # the walls alone, before a prop can take over `touching` (props latch `prop_touched`)
            self.wall_touched.logical_or_(touching)
        prop_slot = None                   # set below when the props can be pushed
        prop_deeper = n_p = None
        if getattr(self.track, "has_props", False):
            # Whichever of wall or prop is deeper owns this step's normal. Blending two normals
            # would push the car somewhere neither contact asks for.
            movable = self.movable_obstacles and getattr(self.track, "env_props", None) is not None
            # `Simulator._prop_contact`, not `self._prop_contact`: the instance attribute is what
            # the console swaps for a spy while it records arguments and then for a captured leaf
            # graph -- both built for the `sim.step` call site, positional-only, one fixed arity.
            # This call is a different site inside the roll that needs the real computation and
            # passes keywords, and going through the attribute is what crashed session start with
            # "contact_spy() got an unexpected keyword argument 'fn'" the day soft walls became
            # capturable.
            got = Simulator._prop_contact(self, state, fn=prop_contacts, want_slot=movable)
            pen_p, n_p = got[0], got[1]
            # In place, and that is the whole reason soft walls can be captured: a CUDA graph replays
            # device work only, so rebinding the attribute to a new tensor happened once, at capture,
            # and every replay after it dropped the substep prop contacts. `graph_fastpath` refused
            # soft walls for exactly that, and the refusal silently cost both the viewer and training
            # their graphs the day soft became a default -- 0.1x realtime in the viewer, 185 steps/s
            # against 320 in `spec_korea_contact_s910`.
            self.prop_touched.logical_or_(pen_p > 0)              # substeps, read once per step
            prop_deeper = pen_p > pen
            pen = torch.where(prop_deeper, pen_p, pen)
            touching = pen > 0
            prop_slot = got[2] if movable else None
        # Nobody touching anything: nothing to resolve, so do none of it. This runs every substep --
        # 25 per control step -- and costs 2.0 ms of kernel launches a call in eager mode, which is
        # 50 ms of every 25 ms step spent resolving contacts that are not there. Measured: soft
        # contact took a viewer step from 70 to 121 ms, 0.36x realtime to 0.21x, and the user saw
        # 0.1x. The normal field (0.82 ms) and the duct/tall lookup (0.36 ms) only matter where the
        # car is actually touching, which on an open track is almost never.
        #
        # One host sync to decide, which is cheap next to the launches it saves -- but a sync cannot
        # be recorded into a CUDA graph or traced by `torch.compile`, so under either the full
        # branch-free path runs, which is what a graph wants anyway: its launches are free.
        if not _capturing() and not bool(touching.any()):
            return (state, torch.zeros(state.shape[0], 2, device=state.device, dtype=state.dtype)) \
                if dt is not None else state
        n = self.track.edt_gradient(state[:, :2], self.tid)            # away from wall (world)
        if prop_deeper is not None:
            n = torch.where(prop_deeper[:, None], n_p.to(n.dtype), n)
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
        # Over the time since the last resolution, which is `dt` when the caller resolves every few
        # substeps -- using `self.dt` there would bleed a fifth of what the calibration asks for.
        span_dt = self.dt if dt is None else dt
        bleed = (span_dt / tau.clamp_min(1e-6)).clamp(max=1.0)
        f = sp.wall_friction
        # How much of the impulse the car keeps. Against a wall, all of it: the old expression is
        # the m_prop -> infinity limit of this one. Against a 10 kg crate with a 3.74 kg car, the
        # car keeps m_p/(m_c+m_p) = 0.73 of the bounce and the crate takes the rest and slides.
        share = torch.ones_like(vn)
        # `SimParams.prop_relative`: a movable prop is a second body (below, after the contact point).
        rel = prop_slot is not None and self._props_move_in_roll()
        mov = v_p = None
        if rel:
            ep = self.track.env_props
            sl = prop_slot.clamp_min(0)
            m_p = torch.where(prop_slot >= 0, ep.p_mass[self.eid, sl], torch.zeros_like(vn))
            mov = prop_deeper & touching & (m_p > 0)                   # a movable prop owns this contact
            m_c = self.P["m"]
            share = torch.where(mov, m_p / (m_c + m_p).clamp_min(1e-6), share)
            v_p = torch.where(mov[:, None], ep.p_vel[self.eid, sl].to(n.dtype), torch.zeros_like(n))
        if prop_slot is not None and not rel:
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
        # The impulse acts where the car touches, not at its centre (2026-09-28). The contact point
        # is the footprint's extreme point against the normal -- one corner for a glancing hit, the
        # two front corners' midpoint for a square one (a soft arg-min, so it stays branch-free for
        # the graphs). A rigid body struck at an offset p takes J = -(1+k) v_pn / (1/m + (p x n)^2/Iz)
        # and turns by J (p x n) / Iz. The old form pushed the centre only, so a contact could damp
        # the yaw rate but never create it; in the recordings half the impacts (48 %) raise |yaw
        # rate| and half (50 %) flip its sign, against 29 % / 12 % simulated
        # (docs/research/dynamics-collision-audit-2026-09-21.md section 2.2). This car's Iz (0.047)
        # is small next to m p^2 (0.34 at a front corner), so a corner hit mostly spins it. A square
        # hit has p x n = 0 and is exactly the old centre impulse, so the calibrated decelerations
        # of head-on impacts are unchanged.
        r_now = state[:, dyn.IR]
        cw = torch.einsum("bij,kj->bki", torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2),
                          self.corners)                                            # (B,4,2) world offsets
        proj = (cw * n[:, None, :]).sum(-1)                                           # along the normal
        w = torch.softmax(-(proj - proj.min(1, keepdim=True).values) / 0.01, 1)       # the deepest corner(s)
        p_c = (w[..., None] * cw).sum(1)                                              # (B,2) contact offset
        pxn = (p_c[:, 0] * n[:, 1] - p_c[:, 1] * n[:, 0]) * getattr(sp, "contact_lever", 1.0)
        vpn = (vwx - r_now * p_c[:, 1]) * n[:, 0] + (vwy + r_now * p_c[:, 0]) * n[:, 1]
        if not getattr(sp, "contact_at_point", True):
            p_c = torch.zeros_like(p_c); pxn = torch.zeros_like(pxn); vpn = vn
        if v_p is not None:
            vpn = vpn - (v_p * n).sum(1)                                 # a moving prop: relative speed
        into = (vpn < 0) & touching
        # How hard this is (`wall_impact`): the speed into the surface at the contact point -- the
        # `-vpn` the impulse below takes -- kept as the step's maximum. A contact is first seen with
        # its whole approach speed and every resolution after bleeds it, so for an impact the maximum
        # is its first resolution. Only read here; in place, so a captured graph keeps recording it.
        self.wall_impact.copy_(torch.maximum(
            self.wall_impact, torch.where(into, -vpn, torch.zeros_like(vpn)).to(self.wall_impact.dtype)))
        # ... and split by which owned the contact, walls or a prop (`StepResult.contact_impacts`)
        sev_now = torch.where(into, -vpn, torch.zeros_like(vpn)).to(self.impact_kind.dtype)
        is_prop = prop_deeper if prop_deeper is not None else torch.zeros_like(into)
        self.impact_kind.copy_(torch.maximum(self.impact_kind, torch.stack(
            [torch.where(is_prop, torch.zeros_like(sev_now), sev_now), torch.where(is_prop, sev_now, torch.zeros_like(sev_now))], 1)))
        # ... and where it touched, for the console: the deepest corner(s) of this resolution if it is the
        # step's hardest so far (a resting contact, 0 m/s, is kept while nothing harder came)
        self._record_contact_point(state[:, :2] + p_c, sev_now, touching)
        m_car, Iz = self.P["m"], self.P["Iz"]
        J = (1 + k) * share * bleed * (-vpn) / (1.0 / m_car + pxn * pxn / Iz)       # >= 0 while driving in
        vwx2 = vwx + J / m_car * n[:, 0]
        vwy2 = vwy + J / m_car * n[:, 1]
        vwx2, vwy2 = vwx2 * (1 - f * 0.05), vwy2 * (1 - f * 0.05)
        r2 = r_now + J * pxn / Iz
        if mov is not None:
            # A movable prop is a second body (`SimParams.prop_relative`). The impulse is the two-body one
            # on the RELATIVE velocity at the contact point, J = -(1+k) u_n / (1/m_c + (p x n)^2/Iz + 1/m_p),
            # with Coulomb friction (`body_friction`) along the tangent, and the prop takes exactly -J
            # (`push`), moved in place inside the roll with the floor slowing it (`drift`). The old form
            # took the car's own into-prop speed and left the prop where it stood until the step ended, so
            # every resolution of the step shoved it again: a 1.2 kg box left at about 2.6x the car's speed
            # while the car kept 0.18 of its own -- energy created. Now one impact is one impulse and the
            # momentum is conserved; with k <= 1 and the lever used the same way in u and in the turn
            # (`contact_lever`), no contact gains energy.
            lam = getattr(sp, "contact_lever", 1.0) if getattr(sp, "contact_at_point", True) else 0.0
            t_ = torch.stack([-n[:, 1], n[:, 0]], 1)
            pxt = (p_c[:, 0] * t_[:, 1] - p_c[:, 1] * t_[:, 0]) * lam
            inv_p = torch.where(mov, 1.0 / m_p.clamp_min(1e-6), torch.zeros_like(m_p))
            dvx, dvy = vwx - v_p[:, 0], vwy - v_p[:, 1]
            u_n = dvx * n[:, 0] + dvy * n[:, 1] + r_now * pxn
            u_t = dvx * t_[:, 0] + dvy * t_[:, 1] + r_now * pxt
            Jn = (1 + k) * bleed * (-u_n).clamp_min(0.0) / (1.0 / m_car + pxn * pxn / Iz + inv_p)
            Jt = -bleed * u_t / (1.0 / m_car + pxt * pxt / Iz + inv_p)
            mu = float(getattr(sp, "body_friction", 0.3))
            Jt = torch.maximum(torch.minimum(Jt, mu * Jn), -mu * Jn)
            Px, Py = Jn * n[:, 0] + Jt * t_[:, 0], Jn * n[:, 1] + Jt * t_[:, 1]
            vwx2 = torch.where(mov, vwx + Px / m_car, vwx2)
            vwy2 = torch.where(mov, vwy + Py / m_car, vwy2)
            r2 = torch.where(mov, r_now + (pxn * Jn + pxt * Jt) / Iz, r2)
            into = into | mov
            # the prop's half of it: -P, and its share of the overlap (the car takes `share` below)
            ep.push(self.eid, prop_slot, -torch.stack([Px, Py], 1) * inv_p[:, None],
                    -n * (pen * (1.0 - share) * bleed * mov.to(pen.dtype))[:, None])
        vwx = torch.where(into, vwx2, vwx); vwy = torch.where(into, vwy2, vwy)
        st = state.clone()
        st[:, dyn.IR] = torch.where(into, r2, r_now)
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

    def _wall_contact_speed(self, state: torch.Tensor) -> torch.Tensor:
        """(B,) speed INTO the wall or prop at the contact point, for a car standing in one at `state`
        (0 elsewhere): the quantity `_resolve_wall_contact` records as `wall_impact`, by the same rules
        -- the deeper of wall and prop owns the normal, the contact point is the deepest corner(s)
        along it -- for the crash a terminate step reports and never resolves. Pure: reads `state`."""
        clr = self._footprint_clearance(state)
        pen = (-clr).clamp_min(0.0)
        touching = clr <= 0.0                              # `_post_roll`'s own test for the reported hit
        n = self.track.edt_gradient(state[:, :2], self.tid)
        if getattr(self.track, "has_props", False):
            pen_p, n_p = Simulator._prop_contact(self, state, fn=prop_contacts)[:2]
            deeper = pen_p > pen
            n = torch.where(deeper[:, None], n_p.to(n.dtype), n)
            touching = touching | (pen_p > 0)
        yaw = state[:, dyn.IYAW]
        c, s = torch.cos(yaw), torch.sin(yaw)
        vwx = state[:, dyn.IVX] * c - state[:, dyn.IVY] * s
        vwy = state[:, dyn.IVX] * s + state[:, dyn.IVY] * c
        if getattr(self.cfg.sim, "contact_at_point", True):
            r_now = state[:, dyn.IR]
            cw = torch.einsum("bij,kj->bki", torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2),
                              self.corners)
            proj = (cw * n[:, None, :]).sum(-1)
            w = torch.softmax(-(proj - proj.min(1, keepdim=True).values) / 0.01, 1)
            p_c = (w[..., None] * cw).sum(1)
            vpn = (vwx - r_now * p_c[:, 1]) * n[:, 0] + (vwy + r_now * p_c[:, 0]) * n[:, 1]
        else:
            vpn = vwx * n[:, 0] + vwy * n[:, 1]
        return torch.where(touching, (-vpn).clamp_min(0.0), torch.zeros_like(vpn))

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
