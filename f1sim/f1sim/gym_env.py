"""Gymnasium-style vectorized environment for the LiDAR-only e2e policy.

Observation (what the real car can provide, nothing else):
    scan        (k, n_beams)   last k scans, ranges / range_max, no-return -> 1.0
    speed       (1,)           VESC measured speed / v_max
    prev_action (2,)           previous normalized action
Privileged info for an asymmetric critic is in `info["priv"]` (B, 8): vx, vy, yaw_rate,
lateral offset, heading error to centerline, s / L, curvature ahead proxy (progress rate), wall clearance.

Action (B, 2) in [-1, 1]:  steer = a0 * s_max,  speed = (a1 + 1) / 2 * v_max_policy.

Reward: progress along the centerline [m] (=> minimum lap time), collision penalty,
small steering-rate penalty. Episodes end on collision (terminated) or after max_steps /
`laps` completed laps (truncated). Auto-reset: envs that end are reset in the same step and
their first observation of the new episode is returned (standard vector-env semantics).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Optional

import math

import numpy as np
import torch

from .mpc import ACT_DIM as PLAN_DIM, PlanSpec, PlanTracker, encode as plan_encode, decode as plan_decode
from .params import Config
from .sim import Simulator, StepResult
from .track import Track

REWARD_COMPONENT_KEYS = ("progress", "collision", "collision_speed", "steer_rate", "proximity",
                         "plan_clearance", "wrong_way", "lap", "alive", "car_contact", "overtake",
                         "lap_time", "car_proximity", "sideslip")

#: The nearest-opponent columns of privileged() are stored as metres (or m/s) divided by this, to
#: keep them O(1) for the critic. Anything comparing them against a real distance must multiply.
PRIV_OPP_DIST_SCALE = 5.0


@dataclass
class EnvConfig:
    scan_stack: int = 3
    scan_subsample: int = 1          # 1 = all 1080 beams; 4 -> 270 beams
    action_history: int = 2          # previous normalized actions in the observation
    speed_cap: float = 10.0          # [m/s] current commanded-speed cap (curriculum); observed as speed_cap / v_max_policy
    obs_imu: bool = True             # add step-mean IMU (gyro xyz, accel xyz) and VESC roll/pitch estimate
    imu_gyro_scale: float = 5.0      # [rad/s] normalization
    imu_accel_scale: float = 10.0    # [m/s^2]
    v_max_policy: float = 10.0       # top speed the policy may command (raceline profile peaks here)
    max_steps: int = 1600            # 40 s at 40 Hz
    laps: int = 100                  # truncate after this many laps (racing: keep going)
    reward_progress: float = 1.0     # per meter
    reward_collision: float = -10.0
    reward_steer_rate: float = 0.05  # per unit of normalized steer change
    reward_proximity: float = 0.5    # per metre driven, at zero wall gap
    proximity_speed_ref: float = 0.0 # [m/s] >0: the proximity penalty is scaled by (1 + v / ref): fast past a wall costs more than creeping
    reward_wrong_way: float = 0.2    # per step while facing backwards along the lane (progress is signed anyway; this makes it explicit)
    reward_collision_speed: float = 0.0   # extra collision penalty per m/s of speed at impact (a fast crash costs more than a nudge)
    reward_plan_clearance: float = 0.0
    plan_margin: float = 0.15
    safe_dist: float = 0.30          # [m] body-to-wall gap below which the proximity penalty starts
    reward_lap: float = 0.0          # paid once per completed lap, scaled by the lap's average speed
                                     # (= track length / lap time). Progress reward is already a dense
                                     # stand-in for lap time, but it saturates wherever the speed cap
                                     # rather than the corner decides the speed; this term keeps
                                     # rewarding a genuinely faster lap and requires actually finishing one.
    reward_car_contact: float = 0.0   # extra penalty for touching another car, on top of ending the
                                      # episode. Contact and a wall both just end the episode today,
                                      # so nothing tells the policy that a car is a thing to be
                                      # negotiated with rather than another piece of scenery.
    reward_overtake: float = 0.0      # per metre of arc taken out of the opponents in contention.
                                      # Without it a collision penalty teaches avoidance and only
                                      # avoidance: sitting behind is free, so there is no reason to
                                      # ever pass.
    reward_car_proximity: float = 0.0 # per metre driven, at zero body gap to another car, scaled by
                                      # closing speed. The wall has a proximity term; a car had only the
                                      # contact cliff. So the only gradient near another car was the
                                      # overtake term paying for *closing*, all the way down to touching
                                      # -- "get as close as you can" with nothing saying "leave room".
                                      # A forced, dangerous pass is not something the collision penalty
                                      # can be relied on to price: it is sparse and high-variance, and
                                      # by the time it fires the lesson is one crash out of thousands
                                      # of steps.
    car_safe_gap: float = 0.60        # [m] body-to-body gap below which the car proximity penalty starts
    car_len: float = 0.50             # [m] centre-to-centre distance at which bodies touch nose-to-tail
    car_wid: float = 0.30             # [m] ... and side-by-side
    car_body_dist: float = 0.45       # [m] kept for tests and tools that reason about centre distance
    car_prox_speed_ref: float = 2.0   # [m/s] scale by (1 + closing speed / ref): closing fast on a car
                                      # at 0.3 m is worse than sitting beside it
    overtake_range: float = 12.0      # [m] a car further away than this along the lane is not being
                                      # raced, and scoring it is worse than useless: the signed gap
                                      # has to flip somewhere, and the only place it can flip
                                      # harmlessly is out here rather than at the pass. Two cars
                                      # exactly opposite each other on the lap would otherwise trade
                                      # a spurious clamp-sized jump every time they drifted across
                                      # the far side.
    reward_lap_time: float = 0.0      # time reward: paid per sector for beating this track's own
                                      # recent time through that sector. Progress already points at
                                      # lap time -- over a fixed episode it *is* distance/time -- but
                                      # its sensitivity is hopeless: taking 0.1 s off a 13.5 s lap
                                      # moves the return by 0.95 % on a short track and 0.44 % on a
                                      # long one, against a collision worth about -80, and it sees a
                                      # tenth *less* the longer the track, because progress grows
                                      # with length while the value of a tenth does not.
                                      #
                                      # Three things this term has to get right, none of them
                                      # obvious:
                                      #
                                      # 1. It pays a bonus and never a toll. A negative term at the
                                      #    line is hackable for free: crossing costs a fixed toll,
                                      #    while stopping a centimetre short at the end of an episode
                                      #    costs almost no progress reward, so the policy would learn
                                      #    to coast up to the line and wait. One-sided makes
                                      #    finishing always weakly better than not.
                                      #
                                      # 2. It is scored per *sector*, not per lap. Near a track's
                                      #    limit the policy-driven spread in lap time shrinks toward
                                      #    the spread that randomisation and spawn pose produce on
                                      #    their own, so a once-a-lap signal drowns exactly when the
                                      #    remaining tenths are hardest to find. Sectors give ~12x
                                      #    the events with less accumulated noise in each, and the
                                      #    credit lands at the corner that earned it rather than
                                      #    seconds later at the line. It also still pays on tracks
                                      #    where a full lap is rare, which is most obstacle tracks.
                                      #
                                      # 3. It is measured in *headroom halved*, not seconds saved.
                                      #    Flat seconds get the ordering backwards near the limit: a
                                      #    second found while still far off the pace is easy and a
                                      #    hundredth found at the limit is a real result, and flat
                                      #    seconds pay the easy one a hundred times more. The gain is
                                      #    log(h_ref / h_now) where h is the gap to the raceline's
                                      #    time for that sector, so equal *fractions* of the
                                      #    remaining headroom pay equally: 15 s -> 14 s against a
                                      #    10 s limit pays 0.22, and 10.10 s -> 10.09 s pays 0.10 --
                                      #    still nearly half as much for a hundredth of the time.
    reward_time_sectors: int = 12     # sectors per lap. 12 puts a boundary every 3-6 m on these
                                      # tracks: long enough that one bad corner does not smear across
                                      # several sectors, short enough to time a corner on its own.
    reward_time_momentum: float = 0.98   # EMA setting the time to beat, per track *and* sector. The
                                      # raceline ideal is the physical floor but a poor live target:
                                      # it is out of reach for a long time, so a one-sided bonus
                                      # against it would pay nothing and teach nothing. Racing the
                                      # policy's own recent pace keeps about half of sectors in the
                                      # money at every skill level, and the target falls as it gets
                                      # faster.
    reward_time_floor: float = 0.002  # headroom floor as a fraction of the sector's ideal time, so
                                      # the log stays finite where the policy reaches or beats the
                                      # raceline (which is a point-mass optimum, not a hard bound).
                                      # 0.2 % is 0.02 s over a 10 s lap: the floor has to sit below
                                      # the hundredths this term exists to pay for, or it quietly
                                      # switches the reward off exactly at the limit.
    reward_time_max: float = 0.30     # cap on one sector's gain: halving the headroom in a single
                                      # sector is already a large result, and without a cap a sector
                                      # sitting at the floor turns timing noise into a spike.
    reward_sideslip: float = 0.0      # per metre driven, per radian of body sideslip beyond
                                      # sideslip_free. Sliding is only punished today when it ends
                                      # in a wall; the grip limit itself has no price, so on a
                                      # low-friction draw the policy carries the same corner speed
                                      # it uses on a grippy one and finds out at the wall.
    sideslip_free: float = 0.06       # [rad] ~3.5 deg: the drift angle a clean fast corner has anyway
    reward_alive: float = 0.0
    spawn_lateral_std: float = 0.3
    spawn_yaw_std: float = 0.2
    spawn_min_clearance: float = 0.5  # [m] spawn poses closer to a wall are pulled back to the centerline
    spawn_speed_max: float = 3.0     # random initial speed in [0, this]
    resample_track_on_reset: bool = True   # multi-track sets: pick a random track for each new episode
    scan_stride: int = 1             # frames between stacked scans (3 x stride 3 = 225 ms of history: velocity cues)
    hist_len: int = 0                # >0: proprioceptive history rows (speed, imu, roll/pitch, action) in the observation
    hist_stride: int = 2             # steps between rows (20 x 2 = the last second): implicit identification of grip / lag
    action_mode: str = "direct"      # "direct": (steer, speed); "plan": short local trajectory tracked by an MPC (f1sim.mpc)
    compile_tracker: bool = True
    # races: M cars per track instance, visible to each other's LiDAR, car-car contact = collision
    race_size: int = 1
    opponent: str = "policy"         # "policy": every car is driven by the caller (self-play);
                                     # "teacher": cars 1..M-1 follow the raceline teacher at a random speed scale
    opp_speed_range: tuple = (0.6, 1.0)   # teacher opponents: speed-profile scale per race per reset
    mixed_teacher_frac: float = 0.5   # opponent == "mixed": the share of races whose other cars are
                                      # teacher-driven (the rest are self-play). Measured, each alone
                                      # forgets the other: 6.6M steps of self-play took contact
                                      # avoidance against teachers from 8 to 10.5 per 600 steps while
                                      # passes rose, and the teacher stage had plateaued. Opponent
                                      # *behaviour* is a diversity axis like track shape is; a batch
                                      # with one kind of opponent is a training set with one map.
    spawn_gap: tuple = (2.5, 6.0)    # [m] along the lane between cars of a race at spawn
    selfplay_front_cap: bool = True   # cap the front car of a self-play race (training). The viewer
                                      # turns it off: what is being watched there is the policy
                                      # against an equal, and a silently slowed opponent reads as a
                                      # cap bug ("why is the other car's cap different?")
    selfplay_pace_ref: float = 5.0    # [m/s] the front car of a self-play race is capped at
                                      # opp_speed_range x this, so the car behind has a pass to make
    opp_follow_gap: float = 1.5       # [m] a teacher opponent this close behind another car -- plus its
                                      # braking distance at the closing speed, see follow_cap -- holds
                                      # its speed to that car's. The raceline teacher is blind to other
                                      # cars: measured, 10 of 37 contacts in a race were the teacher
                                      # driving into the learner it had just been passed by. A pass
                                      # that ends with the passed car ramming you is not a lesson
                                      # about passing, and no opponent that can see drives like that.
    opp_follow_ratio: float = 0.9     # fraction of the car-ahead's speed the follower settles to
    opp_follow_decel: float = 3.0     # [m/s^2] what the tracker actually delivers when asked to brake
                                      # from a plan: measured 3.1 against its nominal 5.0 bound, because
                                      # the reference is walked from the current speed and the speed
                                      # weight in the tracking cost is soft. A fixed 3 m trigger let a
                                      # teacher at 4 m/s run straight into a parked car.


class F1VecEnv:
    def __init__(self, track, cfg: Optional[Config] = None, env_cfg: Optional[EnvConfig] = None,
                 num_envs: int = 1024, device: Optional[str] = None):
        """track: a Track or a list of Tracks (envs are spread over them, re-drawn on reset)."""
        self.cfg = cfg or Config()
        self.cfg.sim.terminate_on_collision = True
        self.ecfg = env_cfg or EnvConfig()
        self.sim = Simulator(track, self.cfg, num_envs, device, race_size=self.ecfg.race_size)
        self.B, self.device = num_envs, self.sim.device
        # what `scan[:, ::subsample]` actually yields, which is a ceiling, not a floor. These agreed
        # while the LiDAR had 1080 beams; the measured count is 1081 (270 deg / 0.25 deg, endpoints
        # included), and 1081 // 4 = 270 against a slice of length 271 -- so every scan_subsample > 1
        # died on a shape mismatch when the scan history was written.
        self.n_beams = len(range(0, self.cfg.lidar.n_beams, self.ecfg.scan_subsample))
        self.range_max = self.cfg.lidar.range_max
        self.s_max = self.cfg.vehicle.s_max
        e = self.ecfg
        self.M = e.race_size
        ar = torch.arange(self.B, device=self.device)
        self.race, self.slot = ar // self.M, ar % self.M
        self.learner = torch.ones(self.B, dtype=torch.bool, device=self.device)
        if self.M > 1 and e.opponent == "teacher":
            self.learner[self.slot > 0] = False
        # Which cars the *policy* drives this episode. Static in "teacher" and "policy" mode; in
        # "mixed" mode it changes at every race reset, so the PPO buffers (whose width is fixed by
        # `learner`) hold every car and the loss weights each sample by this mask instead.
        self.on_policy = self.learner.clone()
        self.learner_ids = torch.nonzero(self.learner).flatten()
        self.teacher = None                                    # set_teacher() for opponent == "teacher"
        self.opp_scale = torch.ones(self.B, device=self.device)
        # signed arc to each opponent last step (+ ahead of me, - behind), and whether it is usable.
        # A flag rather than "0 means unset": 0 is a perfectly ordinary gap -- it is the pass itself.
        self.gap_prev = torch.zeros(self.B, max(1, self.M - 1), device=self.device)
        self.gap_valid = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        T_, S_ = self.sim.track.T, int(e.reward_time_sectors)
        self.ideal_lap = torch.zeros(T_, device=self.device)                 # per track, seconds
        self.sector_lim = torch.zeros(T_, S_, device=self.device)            # raceline time per sector
        self.sector_ref = torch.zeros(T_, S_, device=self.device)            # time to beat, 0 = unseen
        self.sector_idx = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.sector_t0 = torch.zeros(self.B, device=self.device)             # episode time at sector entry
        self.sector_valid = torch.zeros(self.B, dtype=torch.bool, device=self.device)  # entered from a boundary
        self.s_prev = torch.zeros(self.B, device=self.device)
        self.hist_len = (e.scan_stack - 1) * e.scan_stride + 1
        self.scan_hist = torch.ones(self.B, self.hist_len, self.n_beams, device=self.device)
        self.act_dim = PLAN_DIM if e.action_mode == "plan" else 2
        self.prev_action = torch.zeros(self.B, self.act_dim, device=self.device)
        self.act_hist = torch.zeros(self.B, self.ecfg.action_history, self.act_dim, device=self.device)
        self.tracker = None; self.prev_steer_norm = torch.zeros(self.B, device=self.device); self.last_cmd = torch.zeros(self.B, 2, device=self.device)
        self.row_dim = 1 + 6 + 2 + self.act_dim
        self.hist = torch.zeros(self.B, (e.hist_len - 1) * e.hist_stride + 1, self.row_dim, device=self.device) if e.hist_len > 0 else None
        self._last_feat = torch.zeros(self.B, 9, device=self.device)
        self._math = self._step_math
        if self.device.type == "cuda" and self.cfg.sim.compile_mode == "reduce-overhead":
            compiled = torch.compile(self._step_math, dynamic=False, mode="reduce-overhead")
            def _math(*args, _c=compiled):
                try:
                    return _c(*args)
                except Exception:                                  # inductor/Triton failure at first use: stay eager
                    self._math = self._step_math
                    return self._step_math(*args)
            self._math = _math
        self.tracker_delay = None
        if e.action_mode == "plan":
            self.tracker = PlanTracker(self.B, self.device, self.cfg.vehicle.lf + self.cfg.vehicle.lr, self.cfg.vehicle.s_max,
                                       e.v_max_policy, compile_solver=e.compile_tracker)
            self._calibrate_tracker(torch.arange(self.B, device=self.device))
        self.speed_cap = torch.full((self.B,), float(self.ecfg.speed_cap), device=self.device)
        # Self-play with identical cars produces almost no passing: the front car is exactly as fast
        # as the one behind, so races settle into a constant-gap procession and the overtake and
        # contact terms read ~0 -- measured, overtake fell from 0.016/step against teachers to
        # 0.0006 the moment the opponent became the policy. So in self-play the *front* car of each
        # race drives under a speed cap scaled by opp_speed_range, the way teacher opponents were
        # slowed: the car behind gets a pass to make, the car in front gets a slower car to defend
        # with, and both are the same network learning both sides of the situation.
        self.cap_base = float(self.ecfg.speed_cap)
        self.cap_scale = torch.ones(self.B, device=self.device)
        self.teacher_race = torch.zeros(self.B, dtype=torch.bool, device=self.device)   # mixed: this race's others are teachers
        self.ep_step = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.lap_start_step = torch.zeros(self.B, dtype=torch.long, device=self.device)   # step of the last finish-line crossing
        self.prev_lap = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.ep_return = torch.zeros(self.B, device=self.device)
        self.ep_progress = torch.zeros(self.B, device=self.device)
        self.last_result: Optional[StepResult] = None
        self._empty_long = torch.zeros(0, dtype=torch.long, device=self.device); self._empty_float = torch.zeros(0, device=self.device)
        self._no_plan = torch.zeros(self.B, 1, 4, device=self.device)
        self.single_observation_space, self.single_action_space = self._spaces()

    def _spaces(self):
        try:
            import gymnasium as gym
            spaces = {
                "scan": gym.spaces.Box(0.0, 1.0, (self.ecfg.scan_stack, self.n_beams), np.float32),
                "speed": gym.spaces.Box(-1.0, 2.0, (1,), np.float32),
                "prev_action": gym.spaces.Box(-1.0, 1.0, (self.act_dim * self.ecfg.action_history,), np.float32),
                **({"hist": gym.spaces.Box(-5.0, 5.0, (self.ecfg.hist_len * self.row_dim,), np.float32)} if self.ecfg.hist_len > 0 else {}),
                "speed_cap": gym.spaces.Box(0.0, 1.0, (1,), np.float32)}
            if self.ecfg.obs_imu and self.cfg.imu.enabled:
                spaces["imu"] = gym.spaces.Box(-5.0, 5.0, (6,), np.float32)
                spaces["imu_att"] = gym.spaces.Box(-2.0, 2.0, (2,), np.float32)
            obs = gym.spaces.Dict(spaces)
            act = gym.spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
            return obs, act
        except ImportError:
            return None, None

    def _calibrate_tracker(self, ids: torch.Tensor):
        """The tracker is *calibrated* on the real car the way the odometry is (measured once): in sim it
        knows each env's command delay + half the servo lag (+-20 ms residual), the servo offset
        (+-0.01 rad) and gain (+-4 %) and the VESC speed gain (+-3 %); grip stays unknown by decision."""
        P = self.sim.P; n = ids.numel(); g = self.sim.gen
        u = lambda a: (torch.rand(n, device=self.device, generator=g) - 0.5) * 2 * a
        if self.tracker_delay is None:
            self.tracker_delay = torch.zeros(self.B, device=self.device)
            self.tracker_cal = torch.zeros(self.B, 3, device=self.device)          # steer bias, steer gain, speed gain
        self.tracker_delay[ids] = (P["cmd_delay"][ids] + 0.5 * P["servo_tau"][ids] + u(0.02)).clamp(0.0, 0.2)
        self.tracker_cal[ids, 0] = P["steer_bias"][ids] + u(0.01)
        self.tracker_cal[ids, 1] = P["steer_gain"][ids] * (1 + u(0.04))
        self.tracker_cal[ids, 2] = P["speed_gain"][ids] * (1 + u(0.03))

    def set_teacher(self, teacher):
        """Raceline teacher that drives the opponent cars (opponent == "teacher")."""
        self.teacher = teacher

    # ------------------------------------------------------------------ helpers
    def _norm_scan(self, scan: torch.Tensor) -> torch.Tensor:
        s = scan[:, :: self.ecfg.scan_subsample]
        s = torch.where(torch.isfinite(s), s, torch.full_like(s, self.range_max))
        return (s / self.range_max).clamp(0.0, 1.0)

    def _obs(self, r: StepResult) -> Dict[str, torch.Tensor]:
        speed = (r.odom[:, 3] / self.ecfg.v_max_policy)[:, None]
        obs = {"scan": self.scan_hist[:, ::self.ecfg.scan_stride].clone(), "speed": speed, "prev_action": self.act_hist.reshape(self.B, -1).clone(),
               "speed_cap": (self.speed_cap / self.ecfg.v_max_policy)[:, None]}
        if self.ecfg.obs_imu and r.imu is not None and r.imu.shape[1] > 0:
            m = r.imu.mean(1)
            obs["imu"] = torch.cat([m[:, :3] / self.ecfg.imu_gyro_scale, m[:, 3:] / self.ecfg.imu_accel_scale], 1)
            obs["imu_att"] = r.imu_att[:, :2] / 0.35          # VESC roll/pitch estimate (yaw drifts: excluded)
        if self.hist is not None:
            self._last_feat = torch.cat([speed, obs.get("imu", torch.zeros(self.B, 6, device=self.device)), obs.get("imu_att", torch.zeros(self.B, 2, device=self.device))], 1)
            obs["hist"] = self.hist[:, ::self.ecfg.hist_stride].reshape(self.B, -1)
        return obs

    def _priv(self, r: StepResult) -> torch.Tensor:
        st = r.state
        xy_c, yaw_c = self.sim.track.pose_at_s(r.s, self.sim.tid)
        herr = torch.remainder(st[:, 2] - yaw_c + np.pi, 2 * np.pi) - np.pi
        pv = torch.stack([st[:, 3], st[:, 4], st[:, 5], r.lateral, herr, r.s / self.sim.track.length[self.sim.tid],
                          r.progress / self.sim.control_dt, r.wall_dist], 1)
        if self.M > 1:
            # Nearest opponent, four columns, every one of them metres (or m/s) divided by
            # PRIV_OPP_DIST_SCALE: [ahead offset, side offset, longitudinal speed difference,
            # distance]. `dv` is other.vx - ego.vx, a difference of two body-frame longitudinal
            # speeds -- it is NOT the line-of-sight closing speed; car_proximity_penalty computes
            # that one separately. Anything comparing these columns against a distance has to
            # multiply by PRIV_OPP_DIST_SCALE first.
            o = self.sim.other_idx                             # (B,C)
            d = st[o][:, :, :2] - st[:, None, :2]
            dist = d.norm(dim=2); j = dist.argmin(1); ar = torch.arange(self.B, device=self.device)
            dn = d[ar, j]; c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
            dx, dy = dn[:, 0] * c + dn[:, 1] * sn, -dn[:, 0] * sn + dn[:, 1] * c
            dv = st[o[ar, j], 3] - st[:, 3]
            s_ = PRIV_OPP_DIST_SCALE
            pv = torch.cat([pv, torch.stack([dx / s_, dy / s_, dv / s_, dist[ar, j] / s_], 1)], 1)
        return pv

    def _reset_envs(self, ids: torch.Tensor):
        if ids.numel() == 0:
            return
        e = self.ecfg
        gen = self.sim.gen
        n = ids.numel()
        if self.M == 1:
            if e.resample_track_on_reset and self.sim.track.T > 1:
                self.sim.tid[ids] = torch.randint(self.sim.track.T, (n,), device=self.device, generator=gen)
            s = None
        else:
            # a race resets as a whole (new track, cars staggered along the lane) when all its cars are in
            # ids; a single car (crashed) respawns behind its race mates on the race's current track
            in_ids = torch.zeros(self.B, dtype=torch.bool, device=self.device); in_ids[ids] = True
            full = in_ids.view(-1, self.M).all(1)                             # (G,)
            race = self.race[ids]; slot = self.slot[ids]; is_full = full[race]
            if e.resample_track_on_reset and self.sim.track.T > 1:
                new_tid = torch.randint(self.sim.track.T, (full.shape[0],), device=self.device, generator=gen)
                self.sim.tid[ids] = torch.where(is_full, new_tid[race], self.sim.tid[ids])
            L = self.sim.track.length[self.sim.tid[ids]]
            gap = e.spawn_gap[0] + (e.spawn_gap[1] - e.spawn_gap[0]) * torch.rand(n, device=self.device, generator=gen)
            base = torch.rand(full.shape[0], device=self.device, generator=gen)   # leader position, fraction of a lap
            # Slot 0 is the learner when the opponents are teachers, and the learner belongs at the
            # *back* of the grid: an overtake is a car behind catching and passing a car ahead, and a
            # learner spawned as the leader meets that situation only if a faster teacher passes it
            # first -- the opposite lesson (being overtaken pays negative), and at 0.6x the teacher
            # never even arrives. Self-play (every car a learner) keeps the leader-first stagger.
            if e.opponent == "mixed":
                teach_race = torch.rand(full.shape[0], device=self.device, generator=gen) < e.mixed_teacher_frac
                self.teacher_race[ids] = torch.where(is_full, teach_race[race], self.teacher_race[ids])
                self.on_policy[ids] = ~(self.teacher_race[ids] & (slot > 0))
            teacher_driven = self.teacher_race[ids] if e.opponent == "mixed" else torch.full_like(slot, e.opponent == "teacher", dtype=torch.bool)
            behind = torch.where(teacher_driven, (self.M - 1 - slot).float(), slot.float())
            s_full = base[race] * L - behind * gap                            # front car at base, the rest behind
            other = self.sim.other_idx[ids, 0]
            s_part = self.sim.s[other] - gap                                  # behind the next car of the race
            s = torch.remainder(torch.where(is_full, s_full, s_part), L)
            scale = e.opp_speed_range[0] + (e.opp_speed_range[1] - e.opp_speed_range[0]) * torch.rand(full.shape[0], device=self.device, generator=gen)
            self.opp_scale[ids] = torch.where(is_full, scale[race], self.opp_scale[ids])
            if e.opponent in ("policy", "mixed") and e.selfplay_front_cap:    # heterogeneous self-play
                # scaled against the pace the policy actually drives (selfplay_pace_ref), not the
                # curriculum cap: at cap 9.0 the policy runs corner-limited at ~4.5 m/s, so 0.5-1.0x
                # of 9.0 bound almost nothing and the races stayed processions (contact and
                # proximity terms ~0, same as with identical cars). Teacher opponents were scaled
                # against their raceline profile (~4.8 m/s), which is the analogue.
                front_cap = (self.opp_scale[ids] * e.selfplay_pace_ref).clamp(max=self.cap_base)
                self.cap_scale[ids] = torch.where((slot == 0) & ~teacher_driven, front_cap / max(self.cap_base, 1e-6),
                                                  torch.ones_like(self.opp_scale[ids]))
                self.speed_cap[ids] = self.cap_base * self.cap_scale[ids]
        poses = self.sim.sample_spawn(n, e.spawn_lateral_std, e.spawn_yaw_std, s=s, tid=self.sim.tid[ids], min_clearance=e.spawn_min_clearance)
        speed = torch.rand(n, device=self.device, generator=gen) * e.spawn_speed_max
        self.sim.reset(ids, poses, speed)
        self.sim.odom.state[ids, 3] = speed
        self.prev_action[ids] = 0.0
        if self.act_dim == 2:
            self.prev_action[ids, 1] = speed / e.v_max_policy * 2 - 1
        else:
            self.prev_action[ids, -2:] = (speed / e.v_max_policy * 2 - 1)[:, None]
            self.tracker.reset(ids); self._calibrate_tracker(ids)
        self.prev_steer_norm[ids] = 0.0; self.last_cmd[ids] = 0.0; self.last_cmd[ids, 1] = speed
        self.gap_prev[ids] = 0.0; self.gap_valid[ids] = False       # no gain scored on the first step
        self.act_hist[ids] = self.prev_action[ids][:, None, :]
        self.ep_step[ids] = 0; self.ep_return[ids] = 0.0; self.ep_progress[ids] = 0.0
        self.lap_start_step[ids] = 0; self.prev_lap[ids] = 0
        if self.ecfg.reward_lap_time > 0:
            # a car spawns mid-sector, so its first boundary is not a sector it drove: start the clock
            # but withhold the reward until it has crossed one boundary and entered the next cleanly
            s_new = self.sim.s[ids]
            L_ = self.sim.track.length[self.sim.tid[ids]].clamp_min(1e-6)
            self.sector_idx[ids] = (s_new / L_ * self.sector_lim.shape[1]).long().clamp_(0, self.sector_lim.shape[1] - 1)
            self.sector_t0[ids] = 0.0; self.sector_valid[ids] = False; self.s_prev[ids] = s_new
        # fill the scan history with a fresh scan from the new pose. Scanned for the whole batch on the
        # compiled path (fixed shapes -> one CUDA graph); a per-reset partial batch would run eager kernels
        cars = self.sim._car_boxes(self.sim.state) if self.M > 1 else None
        scan, scan_true, scan_type = self.sim.lidar.scan(self.sim.state[:, :3], None, self.sim.P, motion_distortion=False, tid=self.sim.tid, cars=cars, compiled=True)
        self.scan_hist[ids] = self._norm_scan(scan[ids])[:, None, :]
        if self.hist is not None:
            feat = torch.zeros(ids.numel(), 9, device=self.device); feat[:, 0] = speed / e.v_max_policy
            self.hist[ids] = torch.cat([feat, torch.zeros(ids.numel(), self.act_dim, device=self.device)], 1)[:, None, :]

        return scan, scan_true, scan_type

    def _reset_result(self, r: StepResult, ids: torch.Tensor, scans: tuple[torch.Tensor, ...]) -> StepResult:
        """Refresh reset rows without advancing any car or mutating transition info."""
        current = replace(r, **{f.name: value.clone() for f in fields(r)
                               if isinstance(value := getattr(r, f.name), torch.Tensor)})
        current.state[ids] = self.sim.state[ids]
        current.odom[ids] = self.sim.odom.state[ids]
        current.s[ids] = self.sim.s[ids]
        _, lateral, _ = self.sim.track.project(self.sim.state[:, :2], self.sim.tid)
        current.lateral[ids] = lateral[ids]
        current.wall_dist[ids] = self.sim._footprint_clearance(self.sim.state)[ids]
        for target, scan in zip((current.scan, current.scan_true, current.scan_type), scans):
            target[ids] = scan[ids]
        for target in (current.attitude, current.imu, current.imu_att, current.collision,
                       current.progress, current.lap, current.car_collision):
            if target is not None:
                target[ids] = 0
        return current

    def _opponent_actions(self, action: torch.Tensor) -> torch.Tensor:
        if self.M == 1 or self.ecfg.opponent not in ("teacher", "mixed"):
            return action
        if self.teacher is None:
            raise RuntimeError("opponent == 'teacher' needs env.set_teacher(RacelineTeacher)")
        follow, v_cap = self.follow_cap(self.sim.state)
        if self.act_dim == 2:
            cmd = self.teacher(self.sim.state, self.sim.P, self.sim.tid)
            v = torch.where(follow, torch.minimum(cmd[:, 1] * self.opp_scale, v_cap), cmd[:, 1] * self.opp_scale)
            an = self.teacher_action_to_normalized(torch.stack([cmd[:, 0], v], 1))
        else:
            an = self.teacher.plan_action(self.sim.state, self.sim.P, self.sim.tid, self.ecfg.v_max_policy, self.tracker.spec)
            an = an.clone(); an[:, -2:] = ((an[:, -2:] + 1) * self.opp_scale[:, None] - 1).clamp(-1, 1)   # speed scale
            cap_n = (v_cap / self.ecfg.v_max_policy * 2 - 1)[:, None]
            an[:, -2:] = torch.where(follow[:, None], torch.minimum(an[:, -2:], cap_n), an[:, -2:])
        return torch.where(self.on_policy[:, None], action, an)

    def follow_cap(self, state: torch.Tensor):
        """(is there a car within opp_follow_gap ahead of each car, the speed to hold behind it)."""
        e = self.ecfg
        d = self.signed_gaps(self.sim.s, self.sim.tid)                                  # (B, M-1)
        ahead = torch.where(d > 0, d, torch.full_like(d, 1e9))
        gap, j = ahead.min(1)
        v_ahead = state[self.sim.other_idx.gather(1, j[:, None])[:, 0], 3]
        closing = (state[:, 3] - v_ahead).clamp_min(0.0)
        trigger = e.opp_follow_gap + closing * closing / (2.0 * e.opp_follow_decel)     # room to brake in
        return gap < trigger, (v_ahead * e.opp_follow_ratio).clamp_min(0.5)

    def car_contact_charge(self, s: torch.Tensor, tid: torch.Tensor, car_hit: torch.Tensor) -> torch.Tensor:
        """The contact penalty falls on the car behind. The one in front was driven into; charging it
        would teach that completing a pass is dangerous, which is the opposite of the lesson."""
        d = self.signed_gaps(s, tid)
        nearest = d.abs().argmin(1, keepdim=True)
        dn = d.gather(1, nearest)[:, 0]
        # exempt only a car that is clearly *behind* me (more than a car length). At |gap| below a car
        # length the two are alongside -- the contact is the pass itself, and the sign of a near-zero
        # arc gap says nothing about who moved into whom; measured, 16 of 43 contacts were "from
        # behind" by sign and nearly all of them were the learner cutting across during a pass.
        responsible = dn > -0.6
        return car_hit * responsible.to(car_hit.dtype)

    # ------------------------------------------------------------------ API
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.sim.gen.manual_seed(seed); torch.manual_seed(seed)
        self._reset_envs(torch.arange(self.B, device=self.device))
        # one zero-action step so that odom/scan come from the simulator path
        r = self.sim.step(torch.stack([torch.zeros(self.B, device=self.device), self.sim.state[:, 3]], 1))
        self.scan_hist = torch.roll(self.scan_hist, 1, 1); self.scan_hist[:, 0] = self._norm_scan(r.scan)
        self.last_result = r
        obs = self._obs(r)
        if self.hist is not None:                                  # history starts from the first real observation (as the car's builder does)
            self.hist[:] = torch.cat([self._last_feat, torch.zeros(self.B, self.act_dim, device=self.device)], 1)[:, None, :]
            obs = self._obs(r)
        return obs, {"priv": self._priv(r)}

    def step(self, action: torch.Tensor):
        a = self._opponent_actions(action.to(self.device).clamp(-1.0, 1.0))
        if self.act_dim == 2:
            v_cmd = torch.minimum((a[:, 1] + 1) * 0.5 * self.ecfg.v_max_policy, self.speed_cap)
            cmd = torch.stack([a[:, 0] * self.s_max, v_cmd], 1)
        else:                                                  # plan -> (steer, speed) through the tracker
            lr = self.last_result
            v_meas = lr.odom[:, 3] if lr is not None else self.sim.state[:, 3]
            yaw_rate = lr.imu[:, :, 2].mean(1) if (lr is not None and lr.imu is not None and lr.imu.shape[1] > 0) else None
            raw = self.tracker(a, v_meas, self.speed_cap, yaw_rate, delay=self.tracker_delay)
            self.last_cmd_raw = raw                                # what the tracker asked for (before calibration)
            cal = self.tracker_cal
            cmd = torch.stack([((raw[:, 0] - cal[:, 0]) / cal[:, 1]).clamp(-self.s_max, self.s_max), raw[:, 1] / cal[:, 2]], 1)
        steer_norm = cmd[:, 0] / self.s_max
        r = self.sim.step(cmd)
        self.last_cmd = cmd
        e = self.ecfg
        self.ep_step += 1
        plan_ref = self.tracker.last_ref if self.tracker is not None and e.reward_plan_clearance > 0 else self._no_plan
        car_hit = (r.car_collision.float() if r.car_collision is not None
                   else torch.zeros_like(self.ep_return))
        out = self._math(r.scan, r.wall_dist, r.s, r.state, r.progress, r.collision, r.lap, a, steer_norm, self.prev_steer_norm,
                         self.scan_hist, self.act_hist, self.ep_step, self.sim.tid, self.ep_return, self.ep_progress, self.prev_lap,
                         plan_ref, self.lap_start_step, car_hit, self.gap_prev, self.gap_valid)
        (self.scan_hist, steer_rate, reward, reward_components, self.act_hist, terminated, truncated,
         crossed, done, self.ep_return, self.ep_progress, flags, self.gap_prev,
         self.gap_valid) = (t.clone() for t in out)
        self.prev_steer_norm = steer_norm; self.prev_action = a
        if self.hist is not None:
            self.hist = torch.cat([torch.cat([self._last_feat, a], 1)[:, None, :], self.hist[:, :-1]], 1)
        obs = self._obs(r)
        any_done, any_lap = flags.tolist()                          # the one host sync of the step
        # true lap times: time between consecutive finish-line crossings (a spawn mid-track does not count)
        if any_lap:
            lap_ids = torch.nonzero(crossed & (self.prev_lap > 0)).flatten()
            lap_times = (self.ep_step[lap_ids] - self.lap_start_step[lap_ids]).float() * self.sim.control_dt
            # same physical bound as the lap bonus, so a wobble on the line cannot be reported as the
            # best lap of the run (it was, repeatedly: 0.05 s)
            real = lap_times >= self.sim.track.length[self.sim.tid[lap_ids]] / e.v_max_policy
            lap_ids, lap_times = lap_ids[real], lap_times[real]
            self.lap_start_step[crossed] = self.ep_step[crossed]
        else:
            lap_ids = self._empty_long; lap_times = self._empty_float
        if e.reward_lap_time > 0:
            self._sector_time_reward(r.s, reward, reward_components)
        self.prev_lap = r.lap.clone()
        info = {"priv": self._priv(r), "progress": r.progress, "lap": r.lap, "wall_dist": r.wall_dist,
                # a snapshot, like track_id below: _reset_envs re-draws the roles in place further
                # down this same step, and the trainer reads this after step() has returned
                "on_policy": self.on_policy.clone(),
                "scan_true": r.scan_true, "track_id": self.sim.tid.clone(), "lap_times": lap_times, "lap_ids": lap_ids,
                "learner": self.learner,
                "reward_components": dict(zip(REWARD_COMPONENT_KEYS, reward_components.unbind(1)))}
        if r.car_collision is not None:
            info["car_collision"] = r.car_collision
        if self.tracker is not None:
            info["plan"] = self.tracker.last_ref                 # (B, N+1, 4) body-frame x, y, heading, speed
        if any_done:
            ids = torch.nonzero(done).flatten()
            info["final"] = {"ids": ids, "return": self.ep_return[ids].clone(), "progress": self.ep_progress[ids].clone(),
                             "steps": self.ep_step[ids].clone(), "collided": terminated[ids].clone(),
                             "lap_time": self.ep_step[ids].float() * self.sim.control_dt}
            # final observation of the ended episodes (before auto-reset)
            info["final_obs"] = {k: v[ids].clone() for k, v in obs.items()}
            info["final_priv"] = self.privileged(r)[ids].clone()
            scans = self._reset_envs(ids)
            r = self._reset_result(r, ids, scans)
            obs = self._obs(r)
        self.last_result = r
        return obs, reward, terminated, truncated, info

    def _step_math(self, scan, wall_dist, s, state, progress, collision, lap, a, steer_norm, prev_steer_norm, scan_hist, act_hist, ep_step, tid,
                   ep_return, ep_progress, prev_lap, plan_ref, lap_start_step, car_hit, gap_prev, gap_valid):
        """Reward, histories and episode flags as pure tensor math (compiled into one CUDA graph when
        the sim runs in reduce-overhead mode: next to a training job every small kernel waits its turn)."""
        e = self.ecfg
        scan_hist = torch.cat([self._norm_scan(scan)[:, None, :], scan_hist[:, :-1]], 1)
        steer_rate = (steer_norm - prev_steer_norm).abs()
        proximity = (e.safe_dist - wall_dist).clamp(min=0.0) / e.safe_dist if e.reward_proximity > 0 else torch.zeros_like(wall_dist)
        if e.proximity_speed_ref > 0:
            proximity = proximity * (1.0 + state[:, 3].abs() / e.proximity_speed_ref)
        wrong_way = torch.zeros_like(wall_dist)
        if e.reward_wrong_way > 0 and self.sim.track.cl is not None:
            _, yaw_c = self.sim.track.pose_at_s(s, tid)
            wrong_way = (torch.cos(state[:, 2] - yaw_c) < 0.0).float()      # more than 90 deg off the lane direction
        crash = collision.float()
        plan_penalty = torch.zeros_like(wall_dist)
        if e.reward_plan_clearance > 0 and plan_ref.shape[1] > 1:
            cosine, sine = torch.cos(state[:, 2])[:, None], torch.sin(state[:, 2])[:, None]
            plan_x = state[:, 0:1] + plan_ref[:, :, 0] * cosine - plan_ref[:, :, 1] * sine
            plan_y = state[:, 1:2] + plan_ref[:, :, 0] * sine + plan_ref[:, :, 1] * cosine
            clearance = self.sim.track.sample_edt(torch.stack([plan_x, plan_y], -1), tid[:, None]).min(1).values
            clearance = clearance - 0.5 * self.cfg.vehicle.width
            plan_penalty = (e.plan_margin - clearance).clamp(min=0.0) / e.plan_margin
        # per metre *driven*, not per metre of centerline progress: with progress the penalty vanishes
        # for a car that stops next to a wall (progress -> 0) and is charged to one that reverses away
        # from it, so "park against the boundary" was the cheapest way to satisfy it.
        travelled = state[:, 3].abs() * self.sim.control_dt
        lap_bonus = torch.zeros_like(progress)
        if e.reward_lap > 0:
            # only a lap between two finish-line crossings has a defined time; the first crossing
            # after a mid-track spawn does not (prev_lap == 0), matching the lap-time bookkeeping
            full_lap = (lap > prev_lap) & (prev_lap > 0)
            lap_dt = (ep_step - lap_start_step).to(progress.dtype) * self.sim.control_dt
            L_lap = self.sim.track.length[tid]
            avg_speed = L_lap / lap_dt.clamp_min(1e-3)
            # A crossing is not a lap unless the time since the last one could physically hold one.
            # The lap counter is symmetric: it counts down when the car slips back over the line and
            # up again when it returns, so a wobble on the start line reads as a lap driven in two
            # control steps. Those show up as 0.05 s "best laps" in the metrics, and here they would
            # be paid an average speed of hundreds of m/s.
            lap_bonus = torch.where(full_lap & (lap_dt >= L_lap / e.v_max_policy),
                                    e.reward_lap * avg_speed, lap_bonus)
        # Overtaking: arc length taken out of the field this step, averaged over the opponents.
        # Positive while closing or passing, negative while being passed. Rewarding the *change*
        # rather than the position keeps it dense -- a pass is one step in a long approach, and a
        # sparse bonus at the moment of passing would be nearly unlearnable.
        #
        # The gap is a *signed* arc wrapped to (-L/2, L/2], not the forward arc to the other car.
        # The forward arc jumps from ~0 to ~L at the instant a pass completes, and the reward reads
        # that jump as a full lap lost: measured, the step that completes a pass scored -2.00 (the
        # clamp) where it should have scored +0.25, and a whole pass paid +5.75 instead of +8.00.
        # The single most informative step of the manoeuvre carried its largest penalty.
        #
        # Averaged over every opponent rather than read off one slot: other_idx is a fixed roster,
        # not "the car ahead", so with three cars the old form tracked whichever car happened to sit
        # in slot 0. Summed over the field there is no target to switch between, and passing anyone
        # pays.
        gap_gain = torch.zeros_like(progress)
        if e.reward_overtake > 0 and self.sim.other_idx is not None:
            gap_gain, _ = self.overtake_gain(s, tid, gap_prev, gap_valid)
        car_prox = self.car_proximity(state) if (e.reward_car_proximity > 0 and self.sim.other_idx is not None) \
            else torch.zeros_like(progress)
        reward_components = torch.stack([
            e.reward_progress * progress,
            e.reward_collision * crash,
            -e.reward_collision_speed * crash * state[:, 3].abs(),
            -e.reward_steer_rate * steer_rate,
            -e.reward_proximity * proximity * travelled,
            -e.reward_plan_clearance * plan_penalty * self.sim.control_dt,
            -e.reward_wrong_way * wrong_way,
            lap_bonus,
            torch.full_like(progress, e.reward_alive),
            -e.reward_car_contact * (self.car_contact_charge(s, tid, car_hit) if self.sim.other_idx is not None else car_hit),
            e.reward_overtake * gap_gain,
            torch.zeros_like(progress),          # lap_time: filled in on the crossing, in step()
            -e.reward_car_proximity * car_prox * travelled,
            -e.reward_sideslip * (torch.atan2(state[:, 4].abs(), state[:, 3].abs().clamp_min(0.5)) - e.sideslip_free).clamp_min(0.0) * travelled,
        ], 1)
        reward = reward_components.sum(1)
        act_hist = torch.cat([a[:, None, :], act_hist[:, :-1]], 1)
        terminated = collision.clone()
        truncated = (~terminated) & ((ep_step >= e.max_steps) | (lap >= e.laps))
        if self.M > 1:                                         # the leader's time limit ends the whole race
            truncated = truncated | (truncated & (self.slot == 0)).view(-1, self.M)[:, 0].repeat_interleave(self.M)
            truncated = truncated & ~terminated
        crossed = lap > prev_lap
        done = terminated | truncated
        flags = torch.stack([done.any(), crossed.any()])
        gap_now = torch.zeros_like(gap_prev)
        if self.sim.other_idx is not None:
            gap_now = self.signed_gaps(s, tid)
        return (scan_hist, steer_rate, reward, reward_components, act_hist, terminated, truncated, crossed,
                done, ep_return + reward, ep_progress + progress, flags, gap_now,
                torch.ones_like(gap_valid))

    def signed_gaps(self, s: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """Arc to each opponent, signed and wrapped to (-L/2, L/2]: + is ahead of me, - is behind.

        Signed, because the forward arc jumps from ~0 to ~L the instant a pass completes and any
        reward reading its change sees a whole lap lost at the one moment that mattered.
        """
        L_ = self.sim.track.length[tid][:, None]
        return (s[self.sim.other_idx] - s[:, None] + L_ / 2) % L_ - L_ / 2

    def car_proximity(self, state: torch.Tensor) -> torch.Tensor:
        """(B,) closeness to the most threatening opponent: 0 outside car_safe_gap of its body, 1 at
        contact, scaled up by the speed the gap is closing at. Euclidean, not arc: the overtake term
        reads arc so a pass can go through zero arc gap, and this reads the body distance so the pass
        has to go *around* the other car rather than through it."""
        e = self.ecfg
        o = self.sim.other_idx                                             # (B,C)
        d = state[o][:, :, :2] - state[:, None, :2]                        # (B,C,2) me -> them
        dist = d.norm(dim=2).clamp_min(1e-6)
        c, sn = torch.cos(state[:, 2]), torch.sin(state[:, 2])
        vme = torch.stack([state[:, 3] * c - state[:, 4] * sn, state[:, 3] * sn + state[:, 4] * c], 1)
        co, so = torch.cos(state[o][:, :, 2]), torch.sin(state[o][:, :, 2])
        vth = torch.stack([state[o][:, :, 3] * co - state[o][:, :, 4] * so,
                           state[o][:, :, 3] * so + state[o][:, :, 4] * co], 2)
        closing = -((vth - vme[:, None, :]) * d).sum(2) / dist             # >0 when the gap is shrinking
        # Body-to-body clearance from an oriented box, not centre distance minus a constant. Cars are
        # 0.5 long and 0.3 wide: alongside, the bodies touch at 0.3 m between centres, so a
        # centre-distance term with a 0.45 m "body" was already saturated at full penalty with 0.3 m
        # of daylight between the cars, and could not tell a rub from a clean pass. Measured on the
        # race leg: 100 % of the learner's contacts were side contacts, and 4M steps of training
        # against that saturated term moved the contact count by nothing.
        lon = (d[..., 0] * c[:, None] + d[..., 1] * sn[:, None]).abs()
        lat = (-d[..., 0] * sn[:, None] + d[..., 1] * c[:, None]).abs()
        gap = torch.sqrt((lon - e.car_len).clamp_min(0.0) ** 2 + (lat - e.car_wid).clamp_min(0.0) ** 2)
        closeness = (1.0 - gap / e.car_safe_gap).clamp(0.0, 1.0)
        scale = 1.0 + closing.clamp_min(0.0) / e.car_prox_speed_ref
        return (closeness * scale).max(1).values

    def overtake_gain(self, s: torch.Tensor, tid: torch.Tensor, gap_prev: torch.Tensor,
                      gap_valid: torch.Tensor):
        """(arc taken out of the field this step, the new gaps). Averaged over every opponent rather
        than read off one slot -- other_idx is a fixed roster, not "the car ahead"."""
        d_now = self.signed_gaps(s, tid)
        near = (d_now.abs() < self.ecfg.overtake_range) & (gap_prev.abs() < self.ecfg.overtake_range)
        gain = torch.where(gap_valid[:, None] & near, (gap_prev - d_now).clamp(-2.0, 2.0),
                           torch.zeros_like(d_now)).mean(1)
        return gain, d_now

    PRIV_PARAMS = ("mu", "mu_f_scale", "cmd_delay", "servo_tau", "motor_tau", "roll_per_g", "steer_bias", "speed_gain")

    @property
    def priv_mu_index(self) -> int:
        """Column of the true friction in privileged(): after the 8 dynamic-state values and, in a
        race, the 4 nearest-opponent values."""
        return 8 + (4 if self.M > 1 else 0)

    def privileged(self, r: StepResult) -> torch.Tensor:
        """Critic-only vector: true dynamic state, track-relative pose, wall clearance, randomized params."""
        pv = self._priv(r)
        params = torch.stack([self.sim.P[k] for k in self.PRIV_PARAMS], 1)
        return torch.cat([pv, params, (self.speed_cap / self.ecfg.v_max_policy)[:, None]], 1)

    def set_speed_cap(self, v: float):
        self.cap_base = float(min(v, self.ecfg.v_max_policy))
        self.speed_cap = self.cap_base * self.cap_scale

    # convenience for the teacher / IL
    @property
    def state(self) -> torch.Tensor:
        return self.sim.state

    def teacher_action_to_normalized(self, cmd: torch.Tensor) -> torch.Tensor:
        """(steer [rad], speed [m/s]) -> policy action in [-1, 1] (direct action space)."""
        return torch.stack([cmd[:, 0] / self.s_max, cmd[:, 1] / self.ecfg.v_max_policy * 2 - 1], 1).clamp(-1, 1)

    def teacher_label(self, teacher) -> torch.Tensor:
        """The teacher's action in this env's action space: (steer, speed) or its local plan."""
        if self.act_dim == 2:
            return self.teacher_action_to_normalized(teacher(self.sim.state, self.sim.P, self.sim.tid))
        return teacher.plan_action(self.sim.state, self.sim.P, self.sim.tid, self.ecfg.v_max_policy, self.tracker.spec)

    def set_ideal_lap(self, racelines) -> None:
        """Fastest plausible time per track and per sector, from the raceline speed profile: sum(ds / v).

        This is what the time reward measures headroom against, so that a hundredth found near a
        sector's limit outweighs a tenth found while still far off it. The raceline is built by
        moving each resampled centerline point along its own normal, so raceline point j sits at
        centerline fraction j/N and its sector follows from the index alone -- no projection, and
        none of the fold-back ambiguity a projection would have to survive.

        A point-mass optimum is not a hard bound: a real lap can beat it in places. reward_time_floor
        keeps the headroom positive where that happens rather than pretending it cannot.
        """
        S = self.sector_lim.shape[1]
        for i, rl in enumerate(racelines[:self.sim.track.T]):
            ds = np.linalg.norm(np.roll(rl.xy, -1, 0) - rl.xy, axis=1)
            dt = ds / np.maximum(rl.v, 0.3)
            sec = np.minimum((np.arange(len(dt)) * S) // len(dt), S - 1)
            self.ideal_lap[i] = float(dt.sum())
            self.sector_lim[i] = torch.as_tensor(np.bincount(sec, weights=dt, minlength=S)[:S],
                                                 dtype=self.sector_lim.dtype, device=self.device)

    def _sector_time_reward(self, s: torch.Tensor, reward: torch.Tensor,
                            reward_components: torch.Tensor) -> None:
        """Pay for beating this track's own recent time through the sector just completed.

        Fully dense over the batch: a sector boundary is crossed several times per second by a
        thousand envs, so anything that had to ask the host which of them fired would cost a CUDA
        sync per step. Nothing here reads a value back.

        The crossing instant is interpolated from arc position rather than rounded to the control
        step. A tenth spread over twelve sectors is under a hundredth each, well below the 25 ms
        step, so integer step counts would quantise away the very signal this term exists to give.
        """
        e = self.ecfg
        S = self.sector_lim.shape[1]
        dt = self.sim.control_dt
        L = self.sim.track.length[self.sim.tid].clamp_min(1e-6)
        sec = (s / L * S).long().clamp(0, S - 1)
        moved = sec != self.sector_idx
        # only a sector left through its own exit boundary was actually driven end to end: a skip
        # (two boundaries in one step) or a slide backwards has no time worth scoring
        adv = ((sec - self.sector_idx) % S) == 1
        ds = (s - self.s_prev) % L
        past = (s - sec.to(s.dtype) * (L / S)) % L
        t_now = self.ep_step.to(s.dtype) * dt - (past / ds.clamp_min(1e-6)).clamp(0.0, 1.0) * dt
        t_sec = t_now - self.sector_t0

        flat = self.sim.tid * S + self.sector_idx              # the sector being left
        lim = self.sector_lim.view(-1)[flat]
        ref = self.sector_ref.view(-1)[flat]
        h_min = (e.reward_time_floor * lim).clamp_min(1e-3)
        # a sector cannot be driven faster than its arc length at the speed limit, so anything quicker
        # is a wobble across the boundary and not a sector at all -- and a wobble is cheap to repeat,
        # which would make the cap on the gain a per-oscillation payout rather than a safety limit.
        # 0.7 of the bound because the racing line cuts inside the centerline the sectors are cut on.
        t_min = 0.7 * (L / S) / e.v_max_policy
        scored = adv & self.sector_valid & (t_sec >= t_min)
        gain = torch.where(scored & (ref > 0),
                           torch.log((ref - lim).clamp_min(h_min) / (t_sec - lim).clamp_min(h_min)),
                           torch.zeros_like(t_sec)).clamp(0.0, e.reward_time_max)
        reward += e.reward_lap_time * gain
        reward_components[:, REWARD_COMPONENT_KEYS.index("lap_time")] = e.reward_lap_time * gain

        # EMA per (track, sector). Many envs finish the same sector on the same step, so their times
        # are averaged and the decay applied once -- stepping it per env would move a busy sector
        # much further than a quiet one for no reason but how many cars happened to be on it.
        w = scored.to(t_sec.dtype)
        flat_ref = self.sector_ref.view(-1)
        total = torch.zeros_like(flat_ref).index_add_(0, flat, t_sec * w)
        count = torch.zeros_like(flat_ref).index_add_(0, flat, w)
        m = e.reward_time_momentum
        mean = total / count.clamp_min(1.0)
        flat_lim = self.sector_lim.view(-1)
        floor = flat_lim + (e.reward_time_floor * flat_lim).clamp_min(1e-3)
        # a sector's first clean time has no reference yet: seed it rather than decaying from zero
        upd = torch.where(flat_ref > 0, m * flat_ref + (1 - m) * mean, mean).clamp_min(floor)
        self.sector_ref = torch.where(count > 0, upd, flat_ref).view_as(self.sector_ref)

        self.sector_t0 = torch.where(moved, t_now, self.sector_t0)
        self.sector_valid |= moved
        self.sector_idx = sec
        self.s_prev = s.clone()

    def plan_world(self, i: int, predicted: bool = False):
        """Env i's current plan (or the tracker's predicted motion) as world (K,3) x, y, speed for viewers."""
        if self.tracker is None or self.tracker.last_ref is None:
            return None
        src = self.tracker.last_pred if predicted else self.tracker.last_ref
        if src is None:
            return None
        ref = src[i].cpu().numpy()
        x, y, yaw = self.sim.state[i, :3].tolist()
        c, s_ = math.cos(yaw), math.sin(yaw)
        return np.stack([x + ref[:, 0] * c - ref[:, 1] * s_, y + ref[:, 0] * s_ + ref[:, 1] * c, ref[:, 3]], 1)
