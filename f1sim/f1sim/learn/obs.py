"""Observation encoding shared by training (gym env) and deployment (ROS policy node).

scan    (B, k, N)  last k scans, range / range_max, no return -> 1.0
proprio (B, P)     [speed / v_max, prev actions (2 * h), speed_cap / v_max, imu (6), imu roll/pitch (2)]
Both sides must call the same functions; a mismatch here is a sim-to-real gap by construction.

`ScanAugment` adds the optional extra scan channels (`SCAN_CHANNELS`) on the policy side of that
boundary: they are functions of the scan the env already emits, so the env, the observation spec
recorded in a checkpoint's `extra["spec"]` and every consumer of that spec are untouched by them.
Which channels a checkpoint wants is recorded in its model `meta["scan_channels"]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROPRIO_KEYS = ("speed", "prev_action", "speed_cap", "imu", "imu_att", "hist")   # "hist" only when the spec asks for it

#: Extra scan channels, in the order they are appended to the channel axis. The order is fixed here
#: and not by the caller's spelling: it is the order the first convolution's input columns are laid
#: out in, so a checkpoint written with ("edges", "memory") and loaded as ("memory", "edges") would
#: read two channels that mean the wrong thing while every shape still matched.
SCAN_CHANNELS = ("memory", "edges", "floor")

#: Control period the decay is quoted in [s]. The policy runs at the LiDAR's 40 Hz (`params.py`
#: `control_rate`), on the car and in the simulator alike.
CONTROL_DT = 0.025


def att_index_spec(spec: "ObsSpec") -> dict:
    """Where the floor channel's inputs live in the flattened proprio vector, and their normalisers.

    Recorded into a checkpoint's `meta["scan_channels"]["floor"]["proprio"]` and read back by
    `ScanAugment`, never re-derived from whatever `ObsSpec` a reader happens to hold. The proprio
    layout is a contract between the observation encoding and everything that reads a column out of
    it; a checkpoint whose channel was built against one layout must not silently read another.
    `proprio_dim` travels with the indices so a mismatch is an error at the first forward rather
    than a plausible number. (The same construction `learn.aligned` uses on `feat/motion-memory`.)

    The layout is `flatten_obs`' concatenation of `PROPRIO_KEYS`: speed, prev actions, speed cap,
    imu (gyro xyz then accel xyz), imu roll/pitch, and the optional history. `ObsBuilder.build`
    assembles the same order on the car, so the channel is built from three things the ROS node
    already reads -- VESC wheel speed, the IMU's gyro and its accelerometer.
    """
    i_imu = 1 + spec.act_dim * spec.action_history + 1
    return {"proprio_dim": int(spec.proprio_dim), "speed": 0, "gyro": int(i_imu),
            "accel": int(i_imu + 3), "roll": int(i_imu + 6), "pitch": int(i_imu + 7),
            "v_max": float(spec.v_max), "gyro_scale": float(spec.gyro_scale),
            "accel_scale": float(spec.accel_scale), "att_scale": float(spec.att_scale),
            "range_max": float(spec.range_max)}


def floor_inputs(proprio: torch.Tensor, idx: dict):
    """`(speed (B,), gyro (B,3), accel (B,3), vesc roll/pitch (B,2))` out of the proprio vector.

    The observation carries these divided by their normalisers; the geometry needs metres, radians
    and seconds, so they are multiplied back here -- in one place, so the simulator and the car
    cannot disagree about which column is which or what it is divided by.
    """
    if proprio.dim() != 2:
        raise ValueError(f"proprio must be (B, P), got {tuple(proprio.shape)}")
    want = int(idx["proprio_dim"])
    if proprio.shape[1] != want:
        raise ValueError(
            f"the floor channel was built for a {want}-wide proprio vector and this one is "
            f"{proprio.shape[1]} wide. Its columns are read by index, so a different layout would "
            f"hand the attitude tracker a prev-action where it expects a gyro. Rebuild the channel "
            f"for this observation, or feed the observation it was built for.")
    g, a = int(idx["gyro"]), int(idx["accel"])
    return (proprio[:, int(idx["speed"])] * float(idx["v_max"]),
            proprio[:, g:g + 3] * float(idx["gyro_scale"]),
            proprio[:, a:a + 3] * float(idx["accel_scale"]),
            torch.stack([proprio[:, int(idx["roll"])], proprio[:, int(idx["pitch"])]], 1)
            * float(idx["att_scale"]))


def scan_edges(scan_now: torch.Tensor) -> torch.Tensor:
    """|r[i] - r[i-1]| along the beam axis, first beam 0. (B, N) -> (B, N).

    The crack between two boxes in a row is two range discontinuities a few beams apart; the gap
    the lane actually leaves is one. Convolutions over the raw ranges can form this, but the stem's
    first layer is a stride-2 7-tap convolution, so a two-beam crack lands inside one tap and comes
    out as a mild dip. As its own channel the discontinuity is at full beam resolution before any
    stride touches it. Pure arithmetic on the newest frame: no state, no extra sensor, no cost
    worth measuring (0.02 ms of the 25 ms step, measured in the budget table).
    """
    d = (scan_now[:, 1:] - scan_now[:, :-1]).abs()
    return F.pad(d, (1, 0))


def decayed_occupancy(prev: torch.Tensor, scan_now: torch.Tensor, decay: float) -> torch.Tensor:
    """The closest return seen at each bearing recently, relaxing back toward "free".

    Stored in the same units as the scan (0 = touching the sensor, 1 = no return), so the channel
    is comparable with the frames beside it. A new return pulls the channel down immediately
    (`minimum`); with nothing to see, the memory relaxes toward 1.0 by `decay` per control step,
    which is `exp(-dt / tau)` for a time constant `tau`.

    Bearings are the car's own, uncompensated for its motion: this is a *policy-side* channel built
    from the scan alone, with no pose, no odometry integration and no map -- the two things a
    LiDAR-only policy is allowed to use on the car. A box that leaves the 270 deg window therefore
    leaves a trace at the bearing it left by, fading over `tau`, rather than a correctly
    transformed position. That is the cheap half of the memory; the GRU is the half that can learn
    what the trace means.
    """
    relaxed = 1.0 - (1.0 - prev) * decay
    return torch.minimum(scan_now, relaxed)


class ScanAugment:
    """The extra scan channels for one inference path, with the occupancy memory's state.

    `__call__` takes the stacked scan the env (or `ObsBuilder`) produced, (B, k, N), and returns
    (B, k + len(channels), N): the frames unchanged, then one row per enabled channel in
    `SCAN_CHANNELS` order. The model knows how many extra rows to expect and splits them off before
    it forms temporal deltas, so the frames keep meaning frames.

    The occupancy memory and the floor channel's attitude tracker are episode state and are cleared
    exactly where the recurrent hidden state is -- see `learn.memory.PolicyRuntime`.
    """

    def __init__(self, channels: Sequence[str], n_beams: int, batch: int, device="cpu",
                 tau_s: float = 2.0, dt: float = CONTROL_DT, dtype=torch.float32,
                 floor: Optional[dict] = None):
        unknown = [c for c in channels if c not in SCAN_CHANNELS]
        if unknown:
            raise ValueError(f"unknown scan channel(s) {unknown}; known: {list(SCAN_CHANNELS)}")
        if not channels:
            raise ValueError("ScanAugment with no channels: build None instead, so a path that "
                             "does nothing is visibly doing nothing")
        if not tau_s > 0:
            raise ValueError(f"scan memory tau {tau_s} s must be positive")
        self.channels = tuple(c for c in SCAN_CHANNELS if c in channels)
        self.n_beams, self.batch = int(n_beams), int(batch)
        self.device, self.dtype = torch.device(device), dtype
        self.tau_s, self.dt = float(tau_s), float(dt)
        self.decay = float(np.exp(-self.dt / self.tau_s))
        self.mem = None
        if "memory" in self.channels:
            self.mem = torch.ones(self.batch, self.n_beams, device=self.device, dtype=self.dtype)
        #: The floor channel's geometry, its proprio column map and its attitude tracker. Built only
        #: when the channel is on, so nothing about an unflagged run changes.
        self.floor_cfg = None
        self.floor_idx = None
        self.floor_spec = None
        self.att = None
        self.angles = None
        if "floor" in self.channels:
            from . import floor as _floor
            if not floor or "proprio" not in floor:
                raise ValueError(
                    "the floor channel needs the `floor` block `obs.att_index_spec` builds: it "
                    "reads speed, the gyro and the accelerometer out of the proprio vector by "
                    "index, and guessing the layout is how a channel silently tracks a "
                    "prev-action. Pass meta['scan_channels']['floor'].")
            self.floor_cfg = dict(floor)
            self.floor_idx = dict(floor["proprio"])
            self.floor_spec = _floor.FloorSpec(**(floor.get("spec") or {})).validate()
            self.att_source = str(floor.get("att_source", "ego"))
            if self.att_source not in ("tracker", "vesc", "ego"):
                raise ValueError(f"floor att_source must be 'tracker', 'ego' or 'vesc', got "
                                 f"{self.att_source!r}")
            self.att = _floor.AttitudeTracker(self.batch, device=self.device, dt=self.dt,
                                              dtype=self.dtype)
            #: The ego-state path. Built whether or not it is the source, because it is also what
            #: the front-end reads (`learn/frontend.py`) and one estimator per inference path is
            #: cheaper than two that have to be kept in step.
            self.ego = _floor.EgoStateAttitude(self.batch, device=self.device, dt=self.dt,
                                               dtype=self.dtype)
            self.angles = _floor.beam_angles(self.n_beams, float(floor.get("fov", 1.5 * np.pi)),
                                             device=self.device, dtype=self.dtype)

    def reset(self, done=None) -> None:
        """Clear the episode state: the occupancy memory and the attitude tracker's integrator."""
        if self.att is not None:
            self.att.reset(done)
            self.ego.reset(done)
        if self.mem is None:
            return
        if done is None:
            self.mem.fill_(1.0)
            return
        d = done if torch.is_tensor(done) else torch.as_tensor(done, device=self.mem.device)
        d = d.to(self.mem.device)
        keep = (~d).to(self.mem.dtype) if d.dtype == torch.bool else (1.0 - d.to(self.mem.dtype)).clamp(0.0, 1.0)
        if keep.dim() != 1 or keep.shape[0] != self.mem.shape[0]:
            raise ValueError(f"episode-boundary mask must be ({self.mem.shape[0]},), got {tuple(keep.shape)}")
        self.mem = self.mem * keep[:, None] + (1.0 - keep[:, None])

    def _floor_row(self, now: torch.Tensor, proprio: Optional[torch.Tensor],
                   advance: bool, index=None) -> torch.Tensor:
        """The floor likelihood for this step. Advances the attitude tracker unless `advance` is
        False (a terminal observation, which is scored but never acted on)."""
        from . import floor as _floor
        if proprio is None:
            raise ValueError(
                "the floor channel needs this step's proprio vector: its attitude comes from the "
                "gyro and the accelerometer the observation already carries, and substituting "
                "zeros would hand the geometry a level scan plane on a braking car. Pass "
                "`proprio` -- `PolicyRuntime.observe(scan, proprio)` does.")
        speed, gyro, accel, vesc = floor_inputs(proprio.to(now.dtype), self.floor_idx)
        if self.att_source == "ego":
            # Quasi-static, so there is nothing to un-advance for a terminal observation: the
            # estimator's own filters move, but the value it returns is a function of this step.
            if advance:
                att = self.ego.update(speed, gyro[:, 2])
            else:
                saved = (self.ego.v_lp.clone(), self.ego.w_lp.clone(), self.ego.ax.clone(),
                         self.ego.att.clone(), self.ego.rate.clone(), self.ego.started.clone())
                att = self.ego.update(speed, gyro[:, 2])
                if index is not None:
                    att = att[index]
                (self.ego.v_lp, self.ego.w_lp, self.ego.ax, self.ego.att, self.ego.rate,
                 self.ego.started) = saved
            return _floor.floor_likelihood_norm(
                now, att[:, 0], att[:, 1], float(self.floor_idx["range_max"]), self.floor_spec,
                angles=self.angles,
                range_eps=float(self.floor_cfg.get("range_eps", 0.02)))
        if advance:
            att, ok = self.att.update(gyro, accel, speed), self.att.seen_rest
        else:
            att, ok = self.att.peek(gyro, accel, speed, index)
        if self.att_source == "vesc":
            #: The VESC quaternion instead of the tracker. Kept because it is what the node reads
            #: today and the research note has to be able to run the arm both ways; it is NOT the
            #: default, because measured it is wrong by 8 degrees while driving.
            att, ok = vesc, None
        return _floor.floor_likelihood_norm(
            now, att[:, 0], att[:, 1], float(self.floor_idx["range_max"]), self.floor_spec,
            angles=self.angles, range_eps=float(self.floor_cfg.get("range_eps", 0.02)), att_ok=ok)

    def _channels(self, scan: torch.Tensor, mem: Optional[torch.Tensor],
                  proprio: Optional[torch.Tensor] = None, advance: bool = True, index=None):
        """(stacked channels, the occupancy memory this step, or None)."""
        now = scan[:, 0].detach()
        extra, new_mem = [], None
        for name in self.channels:
            if name == "memory":
                new_mem = decayed_occupancy(mem.to(now.dtype), now, self.decay)
                extra.append(new_mem)
            elif name == "floor":
                extra.append(self._floor_row(now, proprio, advance, index))
            else:                                       # "edges"
                extra.append(scan_edges(now))
        return torch.cat([scan, torch.stack(extra, 1).to(scan.dtype)], 1), new_mem

    def _check(self, scan: torch.Tensor, rows: int):
        if scan.dim() != 3 or scan.shape[2] != self.n_beams:
            raise ValueError(f"scan must be (B, k, {self.n_beams}), got {tuple(scan.shape)}")
        if scan.shape[0] != rows:
            raise ValueError(f"scan batch {scan.shape[0]} is not the {rows} rows this call is for; "
                             f"build one augmenter per inference path")

    def __call__(self, scan: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The augmented scan for this control step; advances the occupancy memory.

        `proprio` is required when the floor channel is on and ignored otherwise, so a caller that
        does not use that channel is unchanged.
        """
        self._check(scan, self.batch)
        out, mem = self._channels(scan, self.mem, proprio)
        if mem is not None:
            self.mem = mem
        return out

    def preview(self, scan: torch.Tensor, index=None, proprio=None) -> torch.Tensor:
        """The augmented scan this augmenter WOULD produce, without advancing its memory.

        For a terminal observation: it is scored (the truncation bootstrap reads its value) but
        never acted on, and advancing the memory for it would leave the next real step carrying a
        step that the next episode did not take. `index` selects the rows of this augmenter's batch
        the observation belongs to.
        """
        mem = self.mem
        if mem is not None and index is not None:
            mem = mem[index]
        self._check(scan, self.batch if index is None else int(len(index)))
        return self._channels(scan, mem, proprio, advance=False, index=index)[0]


@dataclass
class ObsSpec:
    n_beams: int = 1080
    scan_stack: int = 3
    scan_stride: int = 1          # control steps between stacked scans
    action_history: int = 2
    act_dim: int = 2              # 2 = (steer, speed); 6 = local plan (f1sim.mpc), tracked by the MPC on both sides
    hist_len: int = 0             # >0: history of (speed, imu, roll/pitch, action) rows, hist_len rows hist_stride steps apart
    hist_stride: int = 2          # (1 s of history = 20 rows x 2 steps at 40 Hz): the actor can infer grip / lag from its own responses
    range_max: float = 10.0
    v_max: float = 10.0           # = EnvConfig.v_max_policy
    gyro_scale: float = 5.0
    accel_scale: float = 10.0
    att_scale: float = 0.35

    @property
    def row_dim(self) -> int:
        return 1 + 6 + 2 + self.act_dim

    @property
    def proprio_dim(self) -> int:
        return 1 + self.act_dim * self.action_history + 1 + 6 + 2 + self.hist_len * self.row_dim


def flatten_obs(obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """gym env obs dict -> (scan (B,k,N), proprio (B,P)) in the canonical order."""
    return obs["scan"], torch.cat([obs[k] for k in PROPRIO_KEYS if k in obs], 1)


class ObsBuilder:
    """Deployment-side builder: feed raw sensor values each control step, get the same tensors."""

    def __init__(self, spec: ObsSpec, device="cpu"):
        self.spec, self.device = spec, torch.device(device)
        self.reset()

    def reset(self):
        s = self.spec
        self.scan_hist = torch.ones(1, (s.scan_stack - 1) * s.scan_stride + 1, s.n_beams, device=self.device)
        self.act_hist = torch.zeros(1, s.action_history, s.act_dim, device=self.device)
        self.hist = torch.zeros(1, (s.hist_len - 1) * s.hist_stride + 1, s.row_dim, device=self.device) if s.hist_len > 0 else None
        self._pending = None                                       # features of the last build, completed by push_action
        self._first = True

    def push_action(self, action_norm):
        a = torch.as_tensor(action_norm, dtype=torch.float32, device=self.device).reshape(1, self.spec.act_dim)
        self.act_hist = torch.cat([a[:, None, :], self.act_hist[:, :-1]], 1)
        if self.hist is not None and self._pending is not None:    # history row = what the car felt + what it was told
            row = torch.cat([self._pending, a], 1)
            self.hist = torch.cat([row[:, None, :], self.hist[:, :-1]], 1)

    def build(self, ranges, speed: float, imu_mean, imu_att, speed_cap: float):
        """ranges (N,) meters with inf/nan for no return; speed [m/s] from VESC; imu_mean (6,) gyro xyz, accel xyz;
        imu_att (2,) roll, pitch [rad] from the VESC attitude estimate; speed_cap [m/s]."""
        s = self.spec
        r = torch.as_tensor(np.asarray(ranges, dtype=np.float32), device=self.device)
        r = torch.where(torch.isfinite(r), r, torch.full_like(r, s.range_max))
        scan = (r / s.range_max).clamp(0.0, 1.0)[None]
        first = self._first
        if first:
            self.scan_hist[:] = scan[:, None, :]
        else:
            self.scan_hist = torch.roll(self.scan_hist, 1, 1); self.scan_hist[:, 0] = scan
        imu = torch.as_tensor(np.asarray(imu_mean, dtype=np.float32), device=self.device).reshape(1, 6)
        imu = torch.cat([imu[:, :3] / s.gyro_scale, imu[:, 3:] / s.accel_scale], 1)
        att = torch.as_tensor(np.asarray(imu_att, dtype=np.float32), device=self.device).reshape(1, 2) / s.att_scale
        feat = torch.cat([torch.tensor([[speed / s.v_max]], device=self.device), imu, att], 1)     # (1, 9)
        parts = [feat[:, :1], self.act_hist.reshape(1, -1), torch.tensor([[speed_cap / s.v_max]], device=self.device), imu, att]
        if self.hist is not None:
            if first:
                self.hist[:] = torch.cat([feat, torch.zeros(1, s.act_dim, device=self.device)], 1)[:, None, :]
            parts.append(self.hist[:, ::s.hist_stride].reshape(1, -1))
        self._pending = feat; self._first = False
        proprio = torch.cat(parts, 1)
        return self.scan_hist[:, ::s.scan_stride].clone(), proprio
