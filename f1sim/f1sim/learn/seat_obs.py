"""Per-seat observation builders: a league checkpoint drives its car off its OWN observation contract.

The opponent pool (`learn.opponent_pool`) used to hand every entry the env's own observation, which is
right exactly while the league and the learner share one contract -- and refused the rest. That
stopped being true with the v2 redesign (docs/research/architecture-redesign-2026-10-13.md 6장 1단계,
"상대 좌석별 ObsBuilder"): a v2 learner's env emits contract v2 (the newest frame and the 210-wide ego
vector) while every league seat of the boot camp is a v1 checkpoint (six frames, the 366-wide proprio
vector with attitude, accelerometer and speed cap, linear plans); and the learner's own v2 snapshots
join that league as it grows. Without a builder per contract neither safest nor the league could be
raced by a v2 learner -- the redesign's stated price of `opponent_pool._check_compatible`.

A builder keeps, for every row of the batch, the histories the env WOULD keep if it were built on that
contract: the scan stack, the action history and the proprio ring (v1), or the newest frames and the
per-step ring (v2). The env feeds it at the points where it updates its own (`F1VecEnv` calls
`reset_actions` / `fill` from `_reset_envs_core` / `_fill_scans`, `begin` from `reset`, `advance` and
`refresh` from `step`), with the same arithmetic in the same order -- `F1VecEnv._norm_scan`, `obs.norm_*`,
`obs.assemble_ego_v2` -- so a seat's observation is the one the same checkpoint is handed in an env
built on its own contract, bit for bit (`tests/test_league_v2.py`: a v1 league seat in a v2 env and the
same seat in a v1 env, same state, same commands). What a builder reads is the race's own sensors --
the scan, the odometry speed, the IMU of this step, the cap in force -- never another car's state.

Every row is built, whoever drives it: the pool acts over the whole batch width and selects (no
compaction, `OpponentPool.act`), and the rows a v1 seat does not drive cost a few cheap tensor ops.
Measured cost at the boot camp's width is in `OpponentPool`'s notes; a builder exists only while some
entry reads it.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .obs import (ATT_SCALE, ObsSpec, ObsSpecV2, assemble_ego_v2, norm_att, norm_gyro_z, norm_imu,
                  norm_speed)


def seat_key(spec) -> tuple:
    """A hashable name for a contract: entries that read the same contract share one builder (every
    v1 league seat of the boot camp has the same one, so B x 6 x 1081 is kept once, not per seat)."""
    return (type(spec).__name__,) + tuple(sorted(vars(spec).items()))


def buildable(spec, env) -> None:
    """Refuse a contract this env's sensors cannot produce honestly, with the reason.

    The builder reads the race's own LiDAR, odometry and IMU, so the beam count and the range scale
    have to be the env's (a resampled scan would be a different observation, `obs.resample_ranges`'
    rule), and the IMU has to be on (both contracts read the gyro; v1 also the accelerometer and the
    VESC attitude). The privileged opponent block is an oracle no seat may read."""
    name = type(spec).__name__
    if int(spec.n_beams) != int(env.n_beams):
        raise ValueError(f"its {name} reads {spec.n_beams} beams and this env's LiDAR gives {env.n_beams}: a seat "
                         f"is shown the race's own scan, not a resampled one")
    if abs(float(spec.range_max) - float(env.range_max)) > 1e-9:
        raise ValueError(f"its {name} scales ranges by {spec.range_max:g} m and this env's LiDAR by "
                         f"{env.range_max:g} m")
    if not (env.ecfg.obs_imu and env.cfg.imu.enabled):
        raise ValueError(f"its {name} reads the IMU and this env has it off (EnvConfig.obs_imu / cfg.imu.enabled)")
    if isinstance(spec, ObsSpec):
        if str(getattr(spec, "opp_token", "off") or "off") not in ("off", ""):
            raise ValueError("it reads the privileged opponent block, an oracle no seat may be handed")
        if abs(float(spec.att_scale) - ATT_SCALE) > 1e-12:
            raise ValueError(f"its roll / pitch are scaled by {spec.att_scale:g}; the simulator emits them for "
                             f"{ATT_SCALE:g} (gym_env._obs)")


class SeatObsV1:
    """The v1 observation (`ObsSpec`) of every row, as `F1VecEnv._obs` builds it for a v1 env.

    State mirrors the v1 env's own, initialised as its constructor does: `scan_hist` of ones (the
    scan stack), `act_hist` (the issued actions, `prev_action`), `hist` (rows of [speed, imu 6,
    attitude 2, issued action], every step) and `_last_feat` -- the features the next action is decided
    from, which the next history row pairs with that action."""

    version = 1

    def __init__(self, spec: ObsSpec, env):
        buildable(spec, env)
        self.spec, self.env = spec, env
        B, dev, s = env.B, env.device, spec
        self.scan_hist = torch.ones(B, (s.scan_stack - 1) * s.scan_stride + 1, s.n_beams, device=dev)
        self.act_hist = torch.zeros(B, s.action_history, s.act_dim, device=dev)
        self.hist = (torch.zeros(B, (s.hist_len - 1) * s.hist_stride + 1, s.row_dim, device=dev)
                     if s.hist_len > 0 else None)
        self._last_feat = torch.zeros(B, 9, device=dev)
        #: the observation of the latest step (a dict of `obs.PROPRIO_KEYS` + "scan"), None before one
        self.obs: Optional[Dict[str, torch.Tensor]] = None

    # -- the env's hooks, in the order it calls them
    def reset_actions(self, ids: torch.Tensor, prev_action: torch.Tensor) -> None:
        """`_reset_envs_core`: a fresh episode's action history is its "hold this speed" action."""
        self.act_hist[ids] = prev_action[ids][:, None, :]

    def fill(self, ids: torch.Tensor, scan: torch.Tensor, speed: torch.Tensor) -> None:
        """`_fill_scans`: every frame of a reset row is the fresh scan from its new pose, and its proprio
        history is the spawn speed alone (no IMU sample exists yet) with zero actions."""
        s = self.spec
        self.scan_hist[ids] = self.env._norm_scan(scan[ids])[:, None, :]
        if self.hist is not None:
            n = ids.numel()
            feat = torch.zeros(n, 9, device=self.scan_hist.device)
            feat[:, 0] = speed / s.v_max
            self.hist[ids] = torch.cat([feat, torch.zeros(n, s.act_dim, device=feat.device)], 1)[:, None, :]

    def begin(self, r) -> None:
        """`reset`, after its settling step: the newest frame rolled in, and the proprio history started
        from the first real observation (as the car's `ObsBuilder` does)."""
        self.scan_hist = torch.roll(self.scan_hist, 1, 1)
        self.scan_hist[:, 0] = self.env._norm_scan(r.scan)
        self.observe(r)
        if self.hist is not None:
            B = self.scan_hist.shape[0]
            self.hist[:] = torch.cat([self._last_feat, torch.zeros(B, self.spec.act_dim, device=self.hist.device)],
                                     1)[:, None, :]
            self.observe(r)

    def advance(self, r, a: torch.Tensor) -> None:
        """`step`: `_step_math`'s scan and action histories, then the proprio row of the features `a`
        was decided from and `a` itself (each row's own action -- a league seat's are its own plans)."""
        self.scan_hist = torch.cat([self.env._norm_scan(r.scan)[:, None, :], self.scan_hist[:, :-1]], 1)
        self.act_hist = torch.cat([a[:, None, :], self.act_hist[:, :-1]], 1)
        if self.hist is not None:
            self.hist = torch.cat([torch.cat([self._last_feat, a], 1)[:, None, :], self.hist[:, :-1]], 1)
        self.observe(r)

    def refresh(self, r) -> None:
        """`step`, after the rows that ended were reset: their first observation."""
        self.observe(r)

    def observe(self, r) -> None:
        """`F1VecEnv._obs` (v1) with this contract's own normalisers. `opp_token` never: a seat is a
        deployable checkpoint (`opponent_pool._opponent_obs`'s rule)."""
        s, env = self.spec, self.env
        B = self.scan_hist.shape[0]
        speed = norm_speed(r.odom[:, 3], s.v_max)[:, None]
        obs = {"scan": self.scan_hist[:, ::s.scan_stride].clone(), "speed": speed,
               "prev_action": self.act_hist.reshape(B, -1).clone(),
               "speed_cap": norm_speed(env.speed_cap, s.v_max)[:, None]}
        if r.imu is not None and r.imu.shape[1] > 0:
            obs["imu"] = norm_imu(r.imu.mean(1), s.gyro_scale, s.accel_scale)
            obs["imu_att"] = norm_att(r.imu_att[:, :2], ATT_SCALE)
        if self.hist is not None:
            dev = speed.device
            self._last_feat = torch.cat([speed, obs.get("imu", torch.zeros(B, 6, device=dev)),
                                         obs.get("imu_att", torch.zeros(B, 2, device=dev))], 1)
            obs["hist"] = self.hist[:, ::s.hist_stride].reshape(B, -1)
        self.obs = obs


class SeatObsV2:
    """The v2 observation (`ObsSpecV2`) of every row, as `F1VecEnv._obs_v2` builds it: the newest
    frames, and the per-step ring of [speed, gyro z, issued action] filled from an episode's first
    observation (zero actions) -- `ObsBuilderV2`'s rule, applied by `observe` to the rows `fill` marked.
    For a v2 entry whose contract is not the env's: in a v1 env, or a v2 env with other frames,
    history or issued-action columns (`ObsSpecV2.ego_actions`)."""

    version = 2

    def __init__(self, spec: ObsSpecV2, env):
        spec = spec.validate()
        buildable(spec, env)
        self.spec, self.env = spec, env
        B, dev, s = env.B, env.device, spec
        self.scan_hist = torch.ones(B, (s.n_frames - 1) * s.frame_stride + 1, s.n_beams, device=dev)
        self.hist2 = torch.zeros(B, s.ring_len, s.row_dim, device=dev)
        self._last_feat2 = torch.zeros(B, 2, device=dev)
        self._fresh = torch.ones(B, dtype=torch.bool, device=dev)
        self.obs: Optional[Dict[str, torch.Tensor]] = None

    def reset_actions(self, ids: torch.Tensor, prev_action: torch.Tensor) -> None:
        """Nothing: a v2 history starts from zero actions, not from a "hold" action."""

    def fill(self, ids: torch.Tensor, scan: torch.Tensor, speed: torch.Tensor) -> None:
        self.scan_hist[ids] = self.env._norm_scan(scan[ids])[:, None, :]
        self._fresh[ids] = True

    def begin(self, r) -> None:
        self.scan_hist = torch.roll(self.scan_hist, 1, 1)
        self.scan_hist[:, 0] = self.env._norm_scan(r.scan)
        self.observe(r)

    def advance(self, r, a: torch.Tensor) -> None:
        self.scan_hist = torch.cat([self.env._norm_scan(r.scan)[:, None, :], self.scan_hist[:, :-1]], 1)
        self.hist2 = torch.cat([torch.cat([self._last_feat2, a], 1)[:, None, :], self.hist2[:, :-1]], 1)
        self.observe(r)

    def refresh(self, r) -> None:
        self.observe(r)

    def observe(self, r) -> None:
        s = self.spec
        if r.imu is None or r.imu.shape[1] == 0:
            raise RuntimeError("a v2 seat reads the gyro's z axis and this step delivered no IMU sample")
        B = self.scan_hist.shape[0]
        feat = torch.cat([norm_speed(r.odom[:, 3], s.v_max)[:, None],
                          norm_gyro_z(r.imu.mean(1)[:, 2:3], s.gyro_scale)], 1)
        fill = torch.cat([feat, torch.zeros(B, s.act_dim, device=feat.device, dtype=feat.dtype)], 1)
        self.hist2 = torch.where(self._fresh[:, None, None], fill[:, None, :].expand_as(self.hist2), self.hist2)
        self._fresh = torch.zeros_like(self._fresh)
        self._last_feat2 = feat
        self.obs = {"scan": self.scan_hist[:, ::s.frame_stride].clone(),
                    "ego": assemble_ego_v2(feat, self.hist2, s.hist_len, s.hist_stride, s.ego_actions)}


def make_seat(spec, env):
    """The builder for `spec` (`ObsSpec` -> v1, `ObsSpecV2` -> v2), already started if the env has
    been reset (`adopt`)."""
    seat = (SeatObsV2 if isinstance(spec, ObsSpecV2) else SeatObsV1)(spec, env)
    adopt(seat, env)
    return seat


def adopt(seat, env) -> None:
    """Start a builder made after the env's first reset -- a league seat swapped mid-run for a
    checkpoint of a contract nothing read until then -- from the env's latest step, every row as a
    fresh episode starts: all frames the newest scan, the action history the env's last actions, the
    ring started from this observation. Those rows' histories begin at the swap, as at a respawn;
    the seat's own GRU state starts there too (`OpponentPool.replace`). Before a reset: nothing to
    adopt, the reset starts it."""
    r = getattr(env, "last_result", None)
    if r is None:
        return
    ids = torch.arange(env.B, device=env.device)
    seat.reset_actions(ids, env.prev_action)
    seat.fill(ids, r.scan, r.odom[:, 3])
    seat.begin(r)
