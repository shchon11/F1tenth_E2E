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

from dataclasses import dataclass
from typing import Dict, Optional

import math

import numpy as np
import torch

from .mpc import ACT_DIM as PLAN_DIM, PlanSpec, PlanTracker, encode as plan_encode, decode as plan_decode
from .params import Config
from .sim import Simulator, StepResult
from .track import Track


@dataclass
class EnvConfig:
    scan_stack: int = 3
    scan_subsample: int = 1          # 1 = all 1080 beams; 4 -> 270 beams
    action_history: int = 2          # previous normalized actions in the observation
    speed_cap: float = 8.0           # [m/s] current commanded-speed cap (curriculum); observed as speed_cap / v_max_policy
    obs_imu: bool = True             # add step-mean IMU (gyro xyz, accel xyz) and VESC roll/pitch estimate
    imu_gyro_scale: float = 5.0      # [rad/s] normalization
    imu_accel_scale: float = 10.0    # [m/s^2]
    v_max_policy: float = 8.0        # top speed the policy may command
    max_steps: int = 1600            # 40 s at 40 Hz
    laps: int = 100                  # truncate after this many laps (racing: keep going)
    reward_progress: float = 1.0     # per meter
    reward_collision: float = -10.0
    reward_steer_rate: float = 0.05  # per unit of normalized steer change
    reward_proximity: float = 0.1    # per step at zero wall gap, linear in (safe_dist - gap)/safe_dist; 0 = off
    proximity_speed_ref: float = 0.0 # [m/s] >0: the proximity penalty is scaled by (1 + v / ref): fast past a wall costs more than creeping
    reward_wrong_way: float = 0.2    # per step while facing backwards along the lane (progress is signed anyway; this makes it explicit)
    reward_collision_speed: float = 0.0   # extra collision penalty per m/s of speed at impact (a fast crash costs more than a nudge)
    safe_dist: float = 0.30          # [m] body-to-wall gap below which the proximity penalty starts
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
    # races: M cars per track instance, visible to each other's LiDAR, car-car contact = collision
    race_size: int = 1
    opponent: str = "policy"         # "policy": every car is driven by the caller (self-play);
                                     # "teacher": cars 1..M-1 follow the raceline teacher at a random speed scale
    opp_speed_range: tuple = (0.6, 1.0)   # teacher opponents: speed-profile scale per race per reset
    spawn_gap: tuple = (2.5, 6.0)    # [m] along the lane between cars of a race at spawn


class F1VecEnv:
    def __init__(self, track, cfg: Optional[Config] = None, env_cfg: Optional[EnvConfig] = None,
                 num_envs: int = 1024, device: Optional[str] = None):
        """track: a Track or a list of Tracks (envs are spread over them, re-drawn on reset)."""
        self.cfg = cfg or Config()
        self.cfg.sim.terminate_on_collision = True
        self.ecfg = env_cfg or EnvConfig()
        self.sim = Simulator(track, self.cfg, num_envs, device, race_size=self.ecfg.race_size)
        self.B, self.device = num_envs, self.sim.device
        self.n_beams = self.cfg.lidar.n_beams // self.ecfg.scan_subsample
        self.range_max = self.cfg.lidar.range_max
        self.s_max = self.cfg.vehicle.s_max
        e = self.ecfg
        self.M = e.race_size
        ar = torch.arange(self.B, device=self.device)
        self.race, self.slot = ar // self.M, ar % self.M
        self.learner = torch.ones(self.B, dtype=torch.bool, device=self.device)
        if self.M > 1 and e.opponent == "teacher":
            self.learner[self.slot > 0] = False
        self.learner_ids = torch.nonzero(self.learner).flatten()
        self.teacher = None                                    # set_teacher() for opponent == "teacher"
        self.opp_scale = torch.ones(self.B, device=self.device)
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
            self.tracker = PlanTracker(self.B, self.device, self.cfg.vehicle.lf + self.cfg.vehicle.lr, self.cfg.vehicle.s_max, e.v_max_policy)
            self._calibrate_tracker(torch.arange(self.B, device=self.device))
        self.speed_cap = torch.full((self.B,), float(self.ecfg.speed_cap), device=self.device)
        self.ep_step = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.lap_start_step = torch.zeros(self.B, dtype=torch.long, device=self.device)   # step of the last finish-line crossing
        self.prev_lap = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.ep_return = torch.zeros(self.B, device=self.device)
        self.ep_progress = torch.zeros(self.B, device=self.device)
        self.last_result: Optional[StepResult] = None
        self._empty_long = torch.zeros(0, dtype=torch.long, device=self.device); self._empty_float = torch.zeros(0, device=self.device)
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
        if self.M > 1:                                         # nearest opponent: body-frame offset, closing speed, distance
            o = self.sim.other_idx                             # (B,C)
            d = st[o][:, :, :2] - st[:, None, :2]
            dist = d.norm(dim=2); j = dist.argmin(1); ar = torch.arange(self.B, device=self.device)
            dn = d[ar, j]; c, sn = torch.cos(st[:, 2]), torch.sin(st[:, 2])
            dx, dy = dn[:, 0] * c + dn[:, 1] * sn, -dn[:, 0] * sn + dn[:, 1] * c
            dv = st[o[ar, j], 3] - st[:, 3]
            pv = torch.cat([pv, torch.stack([dx / 5.0, dy / 5.0, dv / 5.0, dist[ar, j] / 5.0], 1)], 1)
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
            s_full = base[race] * L - slot.float() * gap                      # leader at base, others behind
            other = self.sim.other_idx[ids, 0]
            s_part = self.sim.s[other] - gap                                  # behind the next car of the race
            s = torch.remainder(torch.where(is_full, s_full, s_part), L)
            scale = e.opp_speed_range[0] + (e.opp_speed_range[1] - e.opp_speed_range[0]) * torch.rand(full.shape[0], device=self.device, generator=gen)
            self.opp_scale[ids] = torch.where(is_full, scale[race], self.opp_scale[ids])
        poses = self.sim.sample_spawn(n, e.spawn_lateral_std, e.spawn_yaw_std, s=s, tid=self.sim.tid[ids], min_clearance=e.spawn_min_clearance)
        speed = torch.rand(n, device=self.device, generator=gen) * e.spawn_speed_max
        self.sim.reset(ids, poses, speed)
        self.prev_action[ids] = 0.0
        if self.act_dim == 2:
            self.prev_action[ids, 1] = speed / e.v_max_policy * 2 - 1
        else:
            self.prev_action[ids, -2:] = (speed / e.v_max_policy * 2 - 1)[:, None]
            self.tracker.reset(ids); self._calibrate_tracker(ids)
        self.prev_steer_norm[ids] = 0.0; self.last_cmd[ids] = 0.0; self.last_cmd[ids, 1] = speed
        self.act_hist[ids] = self.prev_action[ids][:, None, :]
        self.ep_step[ids] = 0; self.ep_return[ids] = 0.0; self.ep_progress[ids] = 0.0
        self.lap_start_step[ids] = 0; self.prev_lap[ids] = 0
        # fill the scan history with a fresh scan from the new pose. Scanned for the whole batch on the
        # compiled path (fixed shapes -> one CUDA graph); a per-reset partial batch would run eager kernels
        scan, _, _ = self.sim.lidar.scan(self.sim.state[:, :3], None, self.sim.P, motion_distortion=False, tid=self.sim.tid, compiled=True)
        self.scan_hist[ids] = self._norm_scan(scan[ids])[:, None, :]
        if self.hist is not None:
            feat = torch.zeros(ids.numel(), 9, device=self.device); feat[:, 0] = speed / e.v_max_policy
            self.hist[ids] = torch.cat([feat, torch.zeros(ids.numel(), self.act_dim, device=self.device)], 1)[:, None, :]

    def _opponent_actions(self, action: torch.Tensor) -> torch.Tensor:
        if self.M == 1 or self.ecfg.opponent != "teacher":
            return action
        if self.teacher is None:
            raise RuntimeError("opponent == 'teacher' needs env.set_teacher(RacelineTeacher)")
        if self.act_dim == 2:
            cmd = self.teacher(self.sim.state, self.sim.P, self.sim.tid)
            cmd = torch.stack([cmd[:, 0], cmd[:, 1] * self.opp_scale], 1)
            an = self.teacher_action_to_normalized(cmd)
        else:
            an = self.teacher.plan_action(self.sim.state, self.sim.P, self.sim.tid, self.ecfg.v_max_policy, self.tracker.spec)
            an = an.clone(); an[:, -2:] = ((an[:, -2:] + 1) * self.opp_scale[:, None] - 1).clamp(-1, 1)   # speed scale
        return torch.where(self.learner[:, None], action, an)

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
        out = self._math(r.scan, r.wall_dist, r.s, r.state, r.progress, r.collision, r.lap, a, steer_norm, self.prev_steer_norm,
                         self.scan_hist, self.act_hist, self.ep_step, self.sim.tid, self.ep_return, self.ep_progress, self.prev_lap)
        self.scan_hist, steer_rate, reward, self.act_hist, terminated, truncated, crossed, done, self.ep_return, self.ep_progress, flags = (t.clone() for t in out)
        self.prev_steer_norm = steer_norm; self.prev_action = a
        if self.hist is not None:
            self.hist = torch.cat([torch.cat([self._last_feat, a], 1)[:, None, :], self.hist[:, :-1]], 1)
        obs = self._obs(r)
        any_done, any_lap = flags.tolist()                          # the one host sync of the step
        # true lap times: time between consecutive finish-line crossings (a spawn mid-track does not count)
        if any_lap:
            lap_ids = torch.nonzero(crossed & (self.prev_lap > 0)).flatten()
            lap_times = (self.ep_step[lap_ids] - self.lap_start_step[lap_ids]).float() * self.sim.control_dt
            self.lap_start_step[crossed] = self.ep_step[crossed]
        else:
            lap_ids = self._empty_long; lap_times = self._empty_float
        self.prev_lap = r.lap
        info = {"priv": self._priv(r), "progress": r.progress, "lap": r.lap, "wall_dist": r.wall_dist,
                "scan_true": r.scan_true, "track_id": self.sim.tid.clone(), "lap_times": lap_times, "lap_ids": lap_ids,
                "learner": self.learner}
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
            self._reset_envs(ids)
            obs = self._obs(r)
            fresh = {"scan": self.scan_hist[ids][:, ::e.scan_stride], "speed": (self.sim.state[ids, 3] / e.v_max_policy)[:, None],
                     "prev_action": self.act_hist[ids].reshape(ids.numel(), -1), "speed_cap": (self.speed_cap[ids] / e.v_max_policy)[:, None]}
            if self.hist is not None:
                fresh["hist"] = self.hist[ids][:, ::e.hist_stride].reshape(ids.numel(), -1)
            for k in obs:
                obs[k][ids] = fresh[k] if k in fresh else 0.0
            if self.hist is not None:                              # the next history row of a fresh car starts from its fresh features
                self._last_feat[ids] = 0.0; self._last_feat[ids, 0] = fresh["speed"][:, 0]
        self.last_result = r
        return obs, reward, terminated, truncated, info

    def _step_math(self, scan, wall_dist, s, state, progress, collision, lap, a, steer_norm, prev_steer_norm, scan_hist, act_hist, ep_step, tid,
                   ep_return, ep_progress, prev_lap):
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
        reward = (e.reward_progress * progress + e.reward_collision * crash - e.reward_collision_speed * crash * state[:, 3].abs()
                  - e.reward_steer_rate * steer_rate - e.reward_proximity * proximity - e.reward_wrong_way * wrong_way + e.reward_alive)
        act_hist = torch.cat([a[:, None, :], act_hist[:, :-1]], 1)
        terminated = collision.clone()
        truncated = (~terminated) & ((ep_step >= e.max_steps) | (lap >= e.laps))
        if self.M > 1:                                         # the leader's time limit ends the whole race
            truncated = truncated | (truncated & (self.slot == 0)).view(-1, self.M)[:, 0].repeat_interleave(self.M)
            truncated = truncated & ~terminated
        crossed = lap > prev_lap
        done = terminated | truncated
        flags = torch.stack([done.any(), (crossed & (prev_lap > 0)).any()])
        return scan_hist, steer_rate, reward, act_hist, terminated, truncated, crossed, done, ep_return + reward, ep_progress + progress, flags

    PRIV_PARAMS = ("mu", "mu_f_scale", "cmd_delay", "servo_tau", "motor_tau", "roll_per_g", "steer_bias", "speed_gain")

    def privileged(self, r: StepResult) -> torch.Tensor:
        """Critic-only vector: true dynamic state, track-relative pose, wall clearance, randomized params."""
        pv = self._priv(r)
        params = torch.stack([self.sim.P[k] for k in self.PRIV_PARAMS], 1)
        return torch.cat([pv, params, (self.speed_cap / self.ecfg.v_max_policy)[:, None]], 1)

    def set_speed_cap(self, v: float):
        self.speed_cap.fill_(float(min(v, self.ecfg.v_max_policy)))

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
