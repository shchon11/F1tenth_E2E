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
#:
#: `aligned` is appended LAST for the same reason the channels are appended after the frames: every
#: column an existing checkpoint had keeps its index, so `("memory", "edges")` means the same two
#: columns before and after this channel existed, and a warm start into `("memory", "edges",
#: "aligned")` is a copy plus one zeroed column.
SCAN_CHANNELS = ("memory", "edges", "aligned", "aligned_prev", "aligned_valid")

#: The rows the aligned warp contributes, in `SCAN_CHANNELS` order. Asking for any of them builds
#: one `learn.aligned.AlignedScan`; asking for `aligned` alone is the addendum's one-channel
#: ablation, and all three are its research-clean set (the fourth channel it names, the current
#: range, is already the newest frame of the stack).
ALIGNED_CHANNELS = ("aligned", "aligned_prev", "aligned_valid")

#: The SI quantities the `aligned` channel warps with, in the order `motion_from_proprio` returns
#: them: forward speed [m/s], yaw rate [rad/s], body roll [rad], body pitch [rad]. Every one of them
#: is measured on the car -- wheel speed from the VESC, the other three from the IMU -- which is
#: what makes this channel deployable at all.
MOTION_KEYS = ("speed", "yaw_rate", "roll", "pitch")

#: Control period the decay is quoted in [s]. The policy runs at the LiDAR's 40 Hz (`params.py`
#: `control_rate`), on the car and in the simulator alike.
CONTROL_DT = 0.025


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


def motion_index_spec(spec: "ObsSpec") -> dict:
    """Where the four `MOTION_KEYS` live in the flattened proprio vector, and their normalisers.

    Recorded into a checkpoint's `meta["scan_channels"]["aligned"]["proprio"]` block and read back
    by `ScanAugment`, rather than re-derived from whatever `ObsSpec` the reader happens to hold.
    The reason is the one this module already gives for fixing the channel order: the proprio layout
    is a contract between the observation encoding and everything that reads a column out of it, and
    a checkpoint whose channel was built against one layout must not silently read a different one.
    `proprio_dim` travels with the indices so the mismatch is an error at the first forward instead
    of a plausible number.

    The layout is `flatten_obs`' concatenation of `PROPRIO_KEYS`: speed, prev actions, speed cap,
    imu (gyro xyz then accel xyz), imu roll/pitch, and the optional history. `ObsBuilder.build`
    assembles the same order on the car.
    """
    i_imu = 1 + spec.act_dim * spec.action_history + 1
    return {"proprio_dim": int(spec.proprio_dim), "speed": 0, "yaw_rate": int(i_imu + 2),
            "roll": int(i_imu + 6), "pitch": int(i_imu + 7), "v_max": float(spec.v_max),
            "gyro_scale": float(spec.gyro_scale), "att_scale": float(spec.att_scale),
            "range_max": float(spec.range_max)}


def motion_from_proprio(proprio: torch.Tensor, idx: dict) -> torch.Tensor:
    """(B, 4) speed [m/s], yaw rate [rad/s], roll [rad], pitch [rad] out of the proprio vector.

    The observation carries these divided by their normalisers; the warp is geometry and needs
    metres, radians and seconds, so they are multiplied back here -- in one place, so the simulator
    and the car cannot disagree about which column is which or what it is divided by.
    """
    if proprio.dim() != 2:
        raise ValueError(f"proprio must be (B, P), got {tuple(proprio.shape)}")
    want = int(idx["proprio_dim"])
    if proprio.shape[1] != want:
        raise ValueError(
            f"the aligned channel was built for a {want}-wide proprio vector and this one is "
            f"{proprio.shape[1]} wide. Its columns are read by index, so a different layout would "
            f"hand the warp a prev-action where it expects a yaw rate. Rebuild the channel for this "
            f"observation, or feed the observation it was built for.")
    return torch.stack([proprio[:, int(idx["speed"])] * float(idx["v_max"]),
                        proprio[:, int(idx["yaw_rate"])] * float(idx["gyro_scale"]),
                        proprio[:, int(idx["roll"])] * float(idx["att_scale"]),
                        proprio[:, int(idx["pitch"])] * float(idx["att_scale"])], 1)


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
    """The extra scan channels for one inference path, with the state the stateful ones carry.

    `__call__` takes the stacked scan the env (or `ObsBuilder`) produced, (B, k, N), and returns
    (B, k + len(channels), N): the frames unchanged, then one row per enabled channel in
    `SCAN_CHANNELS` order. The model knows how many extra rows to expect and splits them off before
    it forms temporal deltas, so the frames keep meaning frames.

    Two of the three channels carry episode state -- the occupancy memory and, when it is on, the
    `aligned` channel's ring of scans and motion increments. Both are cleared exactly where the
    recurrent hidden state is (`learn.memory.PolicyRuntime`), because they are the same claim about
    what this car has seen.

    `aligned` additionally needs the car's own measured motion, so `__call__` takes the proprio
    vector beside the scan. It is optional and ignored unless that channel is on, which is what
    keeps every existing call site -- and the byte-identical `("memory", "edges")` arms -- unchanged.
    """

    def __init__(self, channels: Sequence[str], n_beams: int, batch: int, device="cpu",
                 tau_s: float = 2.0, dt: float = CONTROL_DT, dtype=torch.float32,
                 aligned: Optional[dict] = None):
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
        self.aligned_cfg = None
        self.aligned = None
        if any(c in ALIGNED_CHANNELS for c in self.channels):
            from .aligned import AlignedScan
            if not aligned or "proprio" not in aligned:
                raise ValueError(
                    "the 'aligned' channel needs its spec, including the `proprio` index block "
                    "`obs.motion_index_spec` builds: its warp reads speed, yaw rate, roll and pitch "
                    "out of the proprio vector by index, and guessing the layout is how a channel "
                    "silently warps with a prev-action. Pass meta['scan_channels']['aligned'].")
            self.aligned_cfg = dict(aligned)
            self.aligned = AlignedScan(self.n_beams, self.batch,
                                       {k: v for k, v in aligned.items() if k != "proprio"},
                                       device=self.device, dt=self.dt,
                                       range_max=float(aligned["proprio"]["range_max"]),
                                       dtype=self.dtype)

    def reset(self, done=None) -> None:
        """Clear the stateful channels: every row (`done=None`) or the rows whose episode ended."""
        if self.aligned is not None:
            self.aligned.reset(done)
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

    def _motion(self, proprio: Optional[torch.Tensor]) -> torch.Tensor:
        if proprio is None:
            raise ValueError(
                "the 'aligned' channel is on and this call passed no proprio vector. The warp runs "
                "on the car's own measured speed, yaw rate and roll/pitch, which live in the "
                "proprio the policy is already being given -- pass it: aug(scan, proprio).")
        return motion_from_proprio(proprio.to(self.dtype), self.aligned_cfg["proprio"])

    def _channels(self, scan: torch.Tensor, mem: Optional[torch.Tensor],
                  proprio: Optional[torch.Tensor] = None, advance: bool = True,
                  index=None):
        """(stacked channels, the occupancy memory this step, or None).

        The aligned rows are computed ONCE however many of them are enabled: the warp is the cost
        and asking for the mask beside the residual must not pay for it twice -- nor advance the
        ring buffers twice, which would warp against the wrong scan.
        """
        now = scan[:, 0].detach()
        extra, new_mem, rows = [], None, None
        for name in self.channels:
            if name == "memory":
                new_mem = decayed_occupancy(mem.to(now.dtype), now, self.decay)
                extra.append(new_mem)
            elif name == "edges":
                extra.append(scan_edges(now))
            else:                                       # one of ALIGNED_CHANNELS
                if rows is None:
                    motion = self._motion(proprio)
                    rows = (self.aligned(now, motion) if advance
                            else self.aligned.preview(now, motion, index))
                extra.append(rows[name])
        return torch.cat([scan, torch.stack(extra, 1).to(scan.dtype)], 1), new_mem

    def _check(self, scan: torch.Tensor, rows: int):
        if scan.dim() != 3 or scan.shape[2] != self.n_beams:
            raise ValueError(f"scan must be (B, k, {self.n_beams}), got {tuple(scan.shape)}")
        if scan.shape[0] != rows:
            raise ValueError(f"scan batch {scan.shape[0]} is not the {rows} rows this call is for; "
                             f"build one augmenter per inference path")

    def __call__(self, scan: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The augmented scan for this control step; advances every stateful channel."""
        self._check(scan, self.batch)
        out, mem = self._channels(scan, self.mem, proprio)
        if mem is not None:
            self.mem = mem
        return out

    def preview(self, scan: torch.Tensor, index=None,
                proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The augmented scan this augmenter WOULD produce, without advancing any of its state.

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
