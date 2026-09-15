"""Actor-critic for the LiDAR-only policy. Actor sees (scan stack, proprio); the critic additionally
sees the privileged vector (asymmetric actor-critic). Actor is TensorRT-friendly (conv1d + MLP)."""
from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .future import FutureHead, future_spec
from .memory import GRUMemory, Hidden, memory_spec
from .motion import MotionDvHead, OppMaskHead, motion_spec


class ResBlock1d(nn.Module):
    """Pre-activation-free residual block with GroupNorm. Zero padding on purpose: the LiDAR spans
    270 deg, so the two ends of the beam array are 90 deg apart behind the car, not neighbours --
    circular padding would glue together two directions the car cannot see through."""

    def __init__(self, cin: int, cout: int, stride: int = 1, k: int = 5, dilation: int = 1, groups: int = 8):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=pad, dilation=dilation)
        self.n1 = nn.GroupNorm(min(groups, cout), cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=dilation, dilation=dilation)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.skip = None if (stride == 1 and cin == cout) else nn.Conv1d(cin, cout, 1, stride=stride)

    def forward(self, x):
        y = F.gelu(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return F.gelu(y + (x if self.skip is None else self.skip(x)))


class ScanStem(nn.Module):
    """1D CNN over the beam axis. 1080 beams -> 256 features.

    scan_stem="plain" is the original 5-conv stack. Its receptive field before the single flatten
    Linear is ~79 beams (~20 deg), so anything that needs to relate a gap at -60 deg to a wall at
    +40 deg has to be done by that one Linear over absolute beam indices -- which is exactly the
    kind of feature that memorizes a track set instead of reading a corridor. Measured failure mode:
    on convoluted maps 69 % of collisions happen with the car already outside the lane, at ~2.2 m/s
    (a bit over half the speed cap), i.e. it takes a wrong branch rather than losing grip.

    scan_stem="resnet" targets that: GroupNorm + residual blocks with a dilated final stage (whole-
    scan receptive field), an explicit beam-angle channel, an explicit min-pool channel (the nearest
    return in a window is what decides a collision and is what strided convolutions blur away), and
    global min/mean pooled features alongside the flatten.
    """

    def __init__(self, n_stack: int, n_beams: int, out: int = 256, scan_deltas: bool = False,
                 temporal_encoder: str = "cnn", scan_stem: str = "plain", extra_channels: int = 0):
        super().__init__()
        if temporal_encoder not in ("cnn", "gru") or (temporal_encoder == "gru" and scan_deltas):
            raise ValueError("temporal_encoder must be cnn or gru; scan deltas apply only to cnn")
        if scan_stem not in ("plain", "resnet"):
            raise ValueError("scan_stem must be plain or resnet")
        #: Extra scan channels (`learn.obs.SCAN_CHANNELS`) arriving on the channel axis after the
        #: frames. They are appended LAST, after the angle ramp and the min-pool, so every input
        #: column the frozen original had keeps its index and a warm start is a copy plus zeros --
        #: see `load_for_memory`. They are also excluded from the temporal deltas: a difference
        #: between a decayed occupancy map and a raw frame is not a temporal delta of anything.
        self.extra_channels = int(extra_channels)
        if self.extra_channels < 0:
            raise ValueError(f"extra_channels must be >= 0, got {extra_channels}")
        if self.extra_channels and temporal_encoder == "gru":
            raise ValueError("extra scan channels and the per-frame gru temporal encoder do not "
                             "combine: that encoder feeds the stem one frame at a time and an "
                             "extra channel is not a frame")
        self.scan_deltas, self.temporal_encoder, self.scan_stem = scan_deltas, temporal_encoder, scan_stem
        self.register_buffer("beam_angle", torch.linspace(-1.0, 1.0, n_beams)[None, None, :], persistent=False)
        if scan_stem == "resnet":
            per_frame = temporal_encoder == "gru"
            cin = (1 if per_frame else (2 * n_stack - 1 if scan_deltas else n_stack)) + 2 + self.extra_channels   # + angle + min-pool + extras
            self.trunk = nn.Sequential(
                nn.Conv1d(cin, 48, 7, stride=2, padding=3), nn.GroupNorm(8, 48), nn.GELU(),
                ResBlock1d(48, 64, stride=2),
                ResBlock1d(64, 96, stride=2),
                ResBlock1d(96, 128, stride=2),
                ResBlock1d(128, 128, stride=2, dilation=2))                                 # whole-scan context
            self.neck = nn.Sequential(nn.Conv1d(128, 48, 1), nn.GroupNorm(8, 48), nn.GELU())
            with torch.no_grad():
                probe = self.trunk(torch.zeros(1, cin, n_beams))
                flat = self.neck(probe).numel() + 2 * probe.shape[1] + self.SECTORS   # + global min/mean + raw sectors
            if per_frame:
                self.frame_fc = nn.Sequential(nn.Linear(flat, 128), nn.GELU())
                self.gru = nn.GRU(128, 160, batch_first=True)
                self.fc = nn.Sequential(nn.Linear(160, out), nn.GELU())
            else:
                self.fc = nn.Sequential(nn.Linear(flat, out), nn.GELU())
            return
        if temporal_encoder == "gru":
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.GELU(),
                nn.Conv1d(16, 24, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv1d(24, 32, 5, stride=2, padding=2), nn.GELU(),
                nn.AdaptiveAvgPool1d(24))
            self.frame_fc = nn.Sequential(nn.Linear(32 * 24, 96), nn.GELU())
            self.gru = nn.GRU(96, 128, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(128, out), nn.GELU())
            return
        channels = (2 * n_stack - 1 if scan_deltas else n_stack) + self.extra_channels
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(64, 32, 3, stride=2, padding=1), nn.GELU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, channels, n_beams)).numel()
        self.fc = nn.Sequential(nn.Linear(n, out), nn.GELU())

    def split_channels(self, scan: torch.Tensor):
        """(frames, extra channels or None). The extras are the trailing rows of the channel axis."""
        if not self.extra_channels:
            return scan, None
        k = scan.shape[1] - self.extra_channels
        if k < 1:
            raise ValueError(f"scan has {scan.shape[1]} channels, too few for "
                             f"{self.extra_channels} extra channel(s) plus at least one frame")
        return scan[:, :k], scan[:, k:]

    def scan_features(self, scan: torch.Tensor) -> torch.Tensor:
        if not self.scan_deltas:
            return scan
        return torch.cat([scan, scan[:, :-1] - scan[:, 1:]], 1)

    SECTORS = 36                                          # 270 deg / 36 = 7.5 deg per sector

    def _augment(self, x: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Append the beam-angle ramp, the windowed nearest return and any extra channels."""
        near = -F.max_pool1d(-x[:, :1], kernel_size=9, stride=1, padding=4)
        parts = [x, self.beam_angle.expand(x.shape[0], -1, -1), near]
        if extra is not None:
            parts.append(extra)
        return torch.cat(parts, 1)

    def _sector_profile(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest return per angular sector of the newest scan, straight from the raw input.

        GroupNorm standardises each group over the whole beam axis, so after the trunk a corridor
        1 m wide and one 3 m wide look alike -- exactly the absolute scale a speed decision needs.
        This bypass carries it to the head un-normalised, and doubles as the classical gap-follower
        descriptor (min range per sector) that the conv stack would otherwise have to rediscover."""
        return -F.adaptive_max_pool1d(-x[:, :1], self.SECTORS).flatten(1)

    def _resnet_features(self, x: torch.Tensor, extra: Optional[torch.Tensor] = None,
                         beams: bool = False):
        rows = self._augment(x, extra)
        h = self.trunk(rows)
        pooled = torch.cat([-F.adaptive_max_pool1d(-h, 1).flatten(1), h.mean(2)], 1)
        out = torch.cat([self.neck(h).flatten(1), pooled, self._sector_profile(x)], 1)
        return (out, h, rows) if beams else out

    #: Channels the per-beam auxiliary head reads: the trunk's map and the stem's own input rows.
    #: Only the resnet stem exposes them; the plain stem's five-conv stack has no whole-scan context
    #: to upsample and `--aux-floor` refuses it rather than training a head on a 20-degree window.
    def beam_channels(self) -> tuple:
        if self.scan_stem != "resnet" or self.temporal_encoder == "gru":
            raise ValueError(
                f"the per-beam floor head needs the resnet stem's feature map and this stem is "
                f"{self.scan_stem!r}/{self.temporal_encoder!r}. The plain stack's receptive field "
                f"is ~79 beams, so a head on it would be shown one 20-degree window at a time and "
                f"could not see that an arc of returns lies on one line -- which is the whole "
                f"signal. Train the head on a resnet-stem checkpoint.")
        cin = self.trunk[0].in_channels
        return int(self.trunk[-1].conv2.out_channels if hasattr(self.trunk[-1], "conv2")
                   else self.neck[0].in_channels), int(cin)

    def forward_beams(self, scan):
        """`(pooled features, trunk map (B, C, L), stem input rows (B, C_in, N))`.

        `forward` is this without the last two, and is written as a call into the same helper, so
        the two cannot compute different things. The arithmetic and its order are unchanged, which
        is what keeps a run without the head bit-identical.
        """
        scan, extra = self.split_channels(scan)
        out, h, rows = self._resnet_features(self.scan_features(scan), extra, beams=True)
        return self.fc(out), h, rows

    def forward(self, scan):
        scan, extra = self.split_channels(scan)
        if self.scan_stem == "resnet":
            if self.temporal_encoder == "gru":
                batch, steps, beams = scan.shape
                f = self._resnet_features(scan.reshape(batch * steps, 1, beams))
                f = self.frame_fc(f).reshape(batch, steps, -1).flip(1)     # oldest -> newest
                return self.fc(self.gru(f)[1][-1])
            return self.fc(self._resnet_features(self.scan_features(scan), extra))
        if self.temporal_encoder == "gru":
            batch, steps, beams = scan.shape
            frame_features = self.conv(scan.reshape(batch * steps, 1, beams)).flatten(1)
            frame_features = self.frame_fc(frame_features).reshape(batch, steps, -1).flip(1)
            _, hidden = self.gru(frame_features)
            return self.fc(hidden[-1])
        feats = self.scan_features(scan)
        if extra is not None:
            feats = torch.cat([feats, extra], 1)
        return self.fc(self.conv(feats).flatten(1))


class Actor(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, hidden: int = 256, log_std_init: float = -0.7,
                 act_dim: int = 2, scan_deltas: bool = False, temporal_encoder: str = "cnn",
                 scan_stem: str = "plain", cond_dim: int = 0, memory: Optional[dict] = None,
                 extra_scan_channels: int = 0, future: Optional[dict] = None):
        super().__init__()
        #: Conditioning enters as an additive term on the first MLP layer's *preactivation*, through
        #: one bias-free projection initialised to zero. Two properties follow, and the experiment
        #: needs both:
        #:
        #:   * forward parity -- `cond(c)` is exactly zero at init, so a checkpoint loaded into a
        #:     conditional actor produces bit-identical actions whatever is in `c`. Widening the
        #:     first layer instead would leave new columns at fresh init, which is *not* the
        #:     baseline however the conditioning is zeroed.
        #:   * gradient flow -- `dL/dW = delta_pre . c^T` is nonzero as soon as the preactivation has
        #:     a gradient, so the path trains from the first update. A two-layer conditioning MLP
        #:     with its input layer zeroed would starve the second layer instead.
        #:
        #: `cond_dim = 0` keeps the module absent entirely: same parameters, same state dict, same
        #: behaviour as before this existed.
        self.cond_dim = int(cond_dim)
        self.stem = ScanStem(n_stack, n_beams, scan_deltas=scan_deltas, temporal_encoder=temporal_encoder,
                             scan_stem=scan_stem, extra_channels=int(extra_scan_channels))
        pw = 64 if proprio_dim <= 32 else 128                     # a proprio history (hundreds of inputs) gets a wider embedding
        self.pro = nn.Sequential(nn.Linear(proprio_dim, pw), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(256 + pw, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.cond = None
        if self.cond_dim > 0:
            self.cond = nn.Linear(self.cond_dim, hidden, bias=False)
            nn.init.zeros_(self.cond.weight)
        #: Recurrent memory over the same per-step embedding the MLP sees, entering through the
        #: identical zero-initialised-projection construction as `cond` above and for the identical
        #: two reasons (forward parity with the checkpoint it was warm-started from, and a gradient
        #: that is nonzero from the first update). The six-frame stack is still the input; this
        #: extends the window past it rather than replacing it.
        #:
        #: `memory = None` keeps the module absent entirely: same parameters, same state dict, same
        #: behaviour as before this existed.
        self.memory = None
        if memory:
            spec = memory_spec(**memory)
            self.memory = GRUMemory(256 + pw, spec["hidden_size"], hidden, spec["layers"])
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        nn.init.zeros_(self.mu.bias); self.mu.weight.data.mul_(0.1)
        # Grip head: predicts the car's friction (privileged, mu - 1) from the same features the
        # action comes from. The actor never sees mu, and measured, it drives every car at about the
        # same corner speed and slides on the low-grip draws (sideslip p90 7.8 deg). Whether the IMU
        # history *can* tell grip apart is what the auxiliary loss forces the trunk to find out --
        # and if it can, the action head sits on top of a representation that already knows.
        # A small MLP fed by the trunk *and* the proprio embedding directly. A linear head on the
        # trunk alone read R^2 0.07 on frozen features: the action trunk does not encode grip, and a
        # linear probe on the raw proprio does no better (0.05) -- grip lives in a nonlinear
        # combination of how the car answered its commands, which a linear read-out cannot express
        # and so cannot push the trunk toward either.
        self.grip = nn.Sequential(nn.Linear(hidden + pw, 128), nn.GELU(), nn.Linear(128, 1))
        # Opponent-motion head: (ahead offset, side offset, closing speed) of the nearest car, from
        # the scan stack. A dynamic object is not a labelled input on the real car -- LiDAR returns
        # are all it gets -- so "moving or not" has to be read from how the returns shift between
        # frames. Probed on frozen features the trunk half-knows it already (R^2 0.4 on closing
        # speed); this makes it a target so the representation is asked to know it rather than
        # allowed to.
        self.opp = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 3))
        #: Future head: the nearest opponent's relative state and the ego's own motion, K control
        #: steps ahead, from the state that carries the past -- the GRU's hidden state when there is
        #: one, the trunk features when there is not. See `f1sim.learn.future` for what it predicts
        #: and why it reads the recurrent state rather than the action features.
        #:
        #: Built LAST and only when asked for. Both halves matter: absent, the module list, the
        #: state dict and the RNG draw are what they were before this existed (which is what
        #: `tests/data/ppo_loss_oracle.json` pins); present, every weight above it was still drawn
        #: from the same generator in the same order, so adding the head to a fresh model does not
        #: move the rest of it.
        self.hidden_width = int(hidden)
        #: The motion branch and its two train-time heads (`learn.motion`). Absent unless asked for,
        #: and attached LAST by `ActorCritic` for the same reason the future head is: every weight
        #: above it is then drawn from the same generator in the same order as in the arm without it.
        self.motion_spec: Optional[dict] = None
        self.motion_rows: Tuple[int, ...] = ()
        self.opp_mask = None
        self.dv = None
        self.future = None
        self.future_spec: Optional[dict] = None
        if future:
            self.attach_future(future)
        #: Per-beam floor/solid head (`learn/floor_head.py`). Built by `ActorCritic` AFTER both
        #: networks and after the future head, for the reason that one gives: every module here
        #: draws from the ambient generator, so a head built in the middle would give the
        #: `--aux-floor 0` arm and the `--aux-floor 1` arm differently-initialised weights above it.
        #: Absent, the module list, the state dict and the RNG draw are what they were before this
        #: existed -- which is what `tests/data/ppo_loss_oracle.json` pins.
        self.floor = None
        self.floor_spec: Optional[dict] = None

    def attach_motion(self, motion: dict, rows: Sequence[int], n_beams: int,
                      heads: Sequence[str] = ()) -> None:
        """Build the motion GRU over the scan's aligned rows, and any train-time head on it.

        `rows` are the indices of the aligned channels within the stem's EXTRA channel block, so the
        encoder reads the same columns the checkpoint recorded rather than whichever rows happen to
        be trailing. `heads` is any of ("mask", "dv"); each is train-time only and is never called
        by `forward` or `step`, which is what keeps it out of the exported graph.
        """
        if self.memory is None:
            raise ValueError(
                "motion memory needs the main recurrent memory: h_dyn is carried inside the same "
                "hidden tensor, and the split that separates them is the main GRU's. Run "
                "--memory gru, which is the arm this is an addition to anyway.")
        spec = motion_spec(**motion)
        rows = tuple(int(r) for r in rows)
        if not rows:
            raise ValueError("the motion encoder was given no aligned rows to read")
        self.memory.attach_motion(len(rows), spec)
        self.motion_spec, self.motion_rows = dict(spec), rows
        unknown = [h for h in heads if h not in ("mask", "dv")]
        if unknown:
            raise ValueError(f"unknown motion head(s) {unknown}; known: mask, dv")
        if "mask" in heads:
            self.opp_mask = OppMaskHead(spec["channels"], int(n_beams),
                                        self.memory.motion.encoder.stride, spec["width"])
        if "dv" in heads:
            self.dv = MotionDvHead(self.memory.motion.hidden_size, spec["width"])

    def attach_future(self, future: dict) -> None:
        """Build the future head. Separate from `__init__` so `ActorCritic` can call it LAST.

        The order matters for a controlled experiment, not for correctness: every module here draws
        from the ambient generator, so building the head between the actor and the critic would give
        the `--aux-future 0` arm and the `--aux-future 1.0` arm differently-initialised critic GRUs
        from the same `--seed`. Built after both, the head is the only thing the flag adds.
        """
        spec = future_spec(**future)
        # Which tensor the head reads follows from what this actor HAS, and it is the same tensor
        # `future_input` returns -- including the case the addendum adds, where a motion branch
        # exists and every auxiliary reads `h_dyn` alone rather than the main state.
        want = ("motion" if self.has_motion else
                ("memory" if self.memory is not None else "trunk"))
        if spec["source"] is not None and spec["source"] != want:
            raise ValueError(
                f"future head source {spec['source']!r} does not match this actor, which reads the "
                f"{want} state. A head trained on one cannot be rebuilt on the other -- its input "
                f"is a different tensor of a different width.")
        spec["source"] = want
        in_dim = (self.memory.motion.hidden_size if self.has_motion else
                  (self.memory.hidden_size if self.memory is not None else self.hidden_width))
        self.future = FutureHead(in_dim, spec["width"])
        #: The RESOLVED spec, so `ActorCritic` records what was built rather than what was asked for.
        self.future_spec = dict(spec)

    def attach_floor_head(self, spec: dict) -> None:
        """Build the per-beam floor head. Separate from `__init__` so `ActorCritic` can call it
        LAST; see `attach_future` for why the order is part of the experiment."""
        from .floor_head import FloorHead, floor_head_spec
        cfg = floor_head_spec(**spec)
        c_ctx, c_in = self.stem.beam_channels()
        self.floor = FloorHead(c_ctx, c_in, cfg["width"], cfg["kernel"])
        self.floor_spec = dict(cfg)

    @property
    def has_floor_head(self) -> bool:
        return self.floor is not None

    def _require_cond(self, c, batch):
        """A conditional actor is never run on an implied zero.

        Defaulting a missing `c` to zeros would make the A0 arm and a caller that simply forgot the
        argument indistinguishable -- and an A1 checkpoint evaluated without its input would look
        like a quietly worse policy rather than a misuse.
        """
        if self.cond_dim == 0:
            if c is not None:
                raise ValueError("this actor is unconditional (cond_dim=0) but a condition was passed")
            return None
        if c is None:
            raise ValueError(
                f"this actor is conditional (cond_dim={self.cond_dim}) and requires an explicit "
                f"condition; it will not substitute zeros. Pass the A0 arm's zeros deliberately.")
        if c.dim() != 2 or c.shape[0] != batch or c.shape[1] != self.cond_dim:
            raise ValueError(f"condition must be ({batch}, {self.cond_dim}), got {tuple(c.shape)}")
        return c

    @property
    def has_memory(self) -> bool:
        return self.memory is not None

    @property
    def has_future(self) -> bool:
        return self.future is not None

    @property
    def has_motion(self) -> bool:
        return self.memory is not None and self.memory.motion is not None

    def motion_input(self, scan):
        """The aligned rows the motion encoder reads, sliced out of the observation.

        By recorded index within the stem's extra-channel block, never by "the last few rows": a
        checkpoint whose encoder was built over (aligned, aligned_prev, aligned_valid) must keep
        reading those three, in that order, whatever else a later run enables beside them.
        """
        if not self.has_motion:
            return None
        extra = self.stem.split_channels(scan)[1]
        if extra is None:
            raise ValueError("this actor carries a motion branch but the scan has no extra "
                             "channels; the aligned rows are where the branch's input comes from")
        idx = torch.as_tensor(self.motion_rows, device=scan.device)
        return extra.index_select(1, idx)

    def recurrent_state(self, feat, h_next):
        """The WHOLE recurrent state after this step -- `[h_main | h_dyn]` when there is a motion
        branch -- which is what `learn.probe_hidden` regresses from.

        Deliberately not the same tensor as `future_input` once the motion branch exists: the
        contract's addendum puts every auxiliary on `h_dyn` only, so that the main representation
        cannot absorb the auxiliary loss through the ego-dynamics shortcut, while the question the
        probe asks -- what does the policy's state carry -- is about all of it. The two are one
        function apart (`future_input` returns a slice of what this returns), and
        `tests/test_future_head.py` pins them to each other.
        """
        if self.memory is None:
            return feat
        return None if h_next is None else h_next[-1]

    def future_input(self, feat, h_next):
        """The tensor the future head reads: the recurrent state AFTER this step (`h_next[-1]`, the
        last GRU layer) when there is memory, the trunk features when there is not.

        Public and used by `learn.probe_hidden` as well as by `future_from`, so that the state the
        probe regresses from cannot drift away from the state the loss trains -- the whole claim
        rests on them being one tensor. `h_next` is None when the recurrence was deliberately
        switched off (`use_memory=False`, the KL reference's path), and then there is no state.
        """
        if self.memory is None:
            return feat
        if h_next is None:
            return None
        if self.has_motion:
            # h_dyn only. The auxiliary losses must not reach the main GRU: ego dynamics are the
            # cheap way to drive any of them down, and a main state that takes that route is the
            # failure this whole split exists to avoid.
            return h_next[-1][:, self.memory.hidden_size:]
        return h_next[-1]

    def future_from(self, feat, h_next):
        """The future head's prediction, or None if this actor carries no head (or no state)."""
        if self.future is None:
            return None
        x = self.future_input(feat, h_next)
        return None if x is None else self.future(x)

    def probe_state(self, scan, proprio, c=None, h=None):
        """(deterministic action, the recurrent state, next hidden) from one forward.

        One call, so a probe cannot accidentally read a different tensor -- a differently-timed
        hidden state, or the trunk features of a recurrent actor -- from the one the policy carried.
        What it returns is `recurrent_state`: the whole state, including `h_dyn` where there is one,
        because "what does the policy's state carry" is a question about all of it.
        """
        feat, _p, h_next, _enc, _fl = self._parts(scan, proprio, c, h)
        return torch.tanh(self.mu(feat)), self.recurrent_state(feat, h_next), h_next

    def initial_hidden(self, batch: int, device=None, dtype=None):
        return None if self.memory is None else self.memory.initial(batch, device, dtype)

    def motion_aux(self, enc, h_next):
        """(per-beam mask logits or None, current-Dv prediction or None) from the motion branch.

        Train-time only and called by `step_all`, never by `forward` / `step`, so neither head is in
        the graph `learn.export` traces. Both read the motion branch and nothing else: the mask
        head reads the encoder's features, the Dv head reads `h_dyn`.
        """
        if not self.has_motion or h_next is None:
            return None, None
        mask = None if self.opp_mask is None or enc is None else self.opp_mask(enc)
        dv = None if self.dv is None else self.dv(h_next[-1][:, self.memory.hidden_size:])
        return mask, dv

    def _feedforward_only(self, who: str):
        """Refuse the memoryless entry points on a recurrent actor.

        A recurrent policy run from a zero hidden state every step is a different, worse policy
        that looks exactly like the right one: same shapes, same magnitudes, plausible driving. So
        the old call is an error rather than a silent downgrade, and every caller that has to
        support memory is the one that says so, by using `step` / `step_all` / `step_dist`.
        """
        if self.memory is not None:
            raise RuntimeError(
                f"Actor.{who}() is a feedforward entry point and this actor carries memory. Use "
                f"step{'_' + who if who in ('all', 'dist') else ''}(), which takes and returns the "
                f"hidden state, or ActorCritic.act()/evaluate(), which carry it for you.")

    def embed(self, scan, proprio):
        """(trunk embedding, proprio embedding) for one control step.

        Split out of `_parts` so a truncated-BPTT update can run the whole (steps x envs) block
        through the convolutional stem in one call and then walk the recurrence over the steps --
        the stem is where the cost is and it has no recurrence to respect.
        """
        p = self.pro(proprio)
        return torch.cat([self.stem(scan), p], 1), p

    def head(self, x, c=None, h=None, use_memory: bool = True, rows=None):
        """(actor features, next hidden, motion encoder features or None) from a per-step embedding.

        `rows` are the aligned scan rows the motion branch reads (`motion_input`), and are required
        exactly when that branch exists.
        """
        mem = self.memory if use_memory else None
        if self.cond is None and mem is None:
            return self.mlp(x), None, None           # legacy path, byte-for-byte what it always was
        # Conditional / recurrent: the same three layers, with the extra terms added to the first
        # preactivation. Written out rather than sliced, because `self.mlp[1:]` builds a new
        # Sequential on every forward and this runs once per env step.
        lin0, act0, lin1, act1 = self.mlp[0], self.mlp[1], self.mlp[2], self.mlp[3]
        pre = lin0(x)
        if self.cond is not None:
            pre = pre + self.cond(c.to(x.dtype))
        h_next, enc = None, None
        if mem is not None:
            delta, h_next, enc = mem.step(x, h, rows)
            pre = pre + delta
        return act1(lin1(act0(pre))), h_next, enc

    def embed_floor(self, scan, proprio):
        """`(embedding, proprio embedding, per-beam floor logits)` from ONE stem pass.

        The head reads the stem's own feature map and its input rows, so running it needs the same
        forward the action needs -- not a second one. `forward_beams` is `forward` plus two tensors
        it already had.
        """
        f, ctx, rows = self.stem.forward_beams(scan)
        p = self.pro(proprio)
        return torch.cat([f, p], 1), p, self.floor(ctx, rows)

    def _parts(self, scan, proprio, c=None, h=None, use_memory: bool = True, floor: bool = False):
        c = self._require_cond(c, proprio.shape[0])
        if floor and self.floor is not None:
            x, p, fl = self.embed_floor(scan, proprio)
        else:
            x, p = self.embed(scan, proprio)
            fl = None
        feat, h_next, enc = self.head(x, c, h, use_memory, self.motion_input(scan))
        return feat, p, h_next, enc, fl

    # ---------------------------------------------------------- feedforward entry points
    def features(self, scan, proprio, c=None):
        self._feedforward_only("features")
        return self._parts(scan, proprio, c)[0]

    def forward(self, scan, proprio, c=None):
        self._feedforward_only("forward")
        return torch.tanh(self.mu(self._parts(scan, proprio, c)[0]))

    def forward_all(self, scan, proprio, c=None):
        """(action mean, grip prediction, opponent-motion prediction) from one pass through the trunk."""
        self._feedforward_only("all")
        return self.step_all(scan, proprio, c)[:3]

    def dist(self, scan, proprio, c=None):
        self._feedforward_only("dist")
        return self.step_dist(scan, proprio, c)[0]

    # ---------------------------------------------------------- memory-aware entry points
    #: All three work on a feedforward actor too and return `None` for the next hidden state, so a
    #: converted call site is written once and does not branch on the checkpoint.
    def step(self, scan, proprio, c=None, h=None, use_memory: bool = True):
        feat, _p, h_next, _enc, _fl = self._parts(scan, proprio, c, h, use_memory)
        return torch.tanh(self.mu(feat)), h_next

    def step_all(self, scan, proprio, c=None, h=None, use_memory: bool = True,
                 floor: bool = False):
        """(action mean, grip, opponent motion, future or None, motion aux, per-beam floor logits
        or None, next hidden).

        Every auxiliary prediction is appended BEFORE the hidden state, so `[:3]` -- which is what
        `forward_all` and the warm-start parity test take -- still means (action, grip, opponent),
        and `[3]` still means the future head. `motion aux` is the pair (per-beam mask logits,
        current Dv), both None unless that head exists. `floor` is False unless a caller asks, so
        the extra tensors `forward_beams` returns are not even formed on the rollout path.
        """
        feat, p, h_next, enc, fl = self._parts(scan, proprio, c, h, use_memory, floor)
        return (torch.tanh(self.mu(feat)), self.grip(torch.cat([feat, p], 1))[:, 0],
                self.opp(feat), self.future_from(feat, h_next),
                self.motion_aux(enc, h_next), fl, h_next)

    def step_dist(self, scan, proprio, c=None, h=None, use_memory: bool = True):
        mu, h_next = self.step(scan, proprio, c, h, use_memory)
        mu = mu.float()
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu)), h_next

    def feedforward_dist(self, scan, proprio, c=None):
        """The distribution this actor's feedforward part alone produces: the GRU is not run and
        its projection is not applied.

        For the KL leash's reference actor only. The reference is the frozen original the run was
        warm-started from, whose memory projection is exactly zero, so this is both what it means
        and what it would compute anyway -- without paying for a recurrence whose output is zero
        and whose hidden state nobody keeps. `ppo` checks the zero before relying on it.
        """
        return self.step_dist(scan, proprio, c, None, use_memory=False)[0]


class Critic(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, priv_dim: int, hidden: int = 256,
                 scan_deltas: bool = False, temporal_encoder: str = "cnn", scan_stem: str = "plain",
                 priv_adapter: Optional[str] = None, memory: Optional[dict] = None,
                 extra_scan_channels: int = 0):
        super().__init__()
        self.stem = ScanStem(n_stack, n_beams, scan_deltas=scan_deltas, temporal_encoder=temporal_encoder,
                             scan_stem=scan_stem, extra_channels=int(extra_scan_channels))
        self.pro = nn.Sequential(nn.Linear(proprio_dim + priv_dim, 128), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(256 + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        #: The critic gets its OWN memory rather than sharing the actor's stem or hidden state.
        #: Two reasons, both already the shape of this codebase: the critic is asymmetric (it reads
        #: the privileged vector, which the actor must never see) and it already carries a separate
        #: stem, so sharing the recurrence would be the one place a value gradient reached the
        #: actor's trunk; and the actor's forward is the thing the Jetson budget measures, so a
        #: critic that is never exported keeps its own state and costs the car nothing. The price
        #: is a second hidden state to carry, which training carries and no deployment path does.
        self.memory = None
        self.motion_rows: Tuple[int, ...] = ()
        if memory:
            spec = memory_spec(**memory)
            self.memory = GRUMemory(256 + 128, spec["hidden_size"], hidden, spec["layers"])
        #: Declared, named reshaping of the privileged vector, applied here so that every value path
        #: -- rollout, minibatch, truncation bootstrap, final bootstrap -- is covered by construction
        #: rather than at four call sites. `None` is strict legacy: nothing is adapted and a width
        #: mismatch raises from the linear layer exactly as it did before.
        #:
        #: The one adapter that exists maps a solo env's 17-wide privileged vector to the 21 a critic
        #: trained in a race expects, by inserting four zeros where the nearest-opponent block sits.
        #: Every trained weight is preserved and the zeros say the true thing: no opponent. Raw
        #: privileged storage and the `mu` / aux label indices are untouched by this.
        self.priv_adapter = priv_adapter
        self._adapt = None
        if priv_adapter:
            from .conditioning import get_priv_adapter
            self._adapt = get_priv_adapter(priv_adapter)

    @property
    def has_memory(self) -> bool:
        return self.memory is not None

    def initial_hidden(self, batch: int, device=None, dtype=None):
        return None if self.memory is None else self.memory.initial(batch, device, dtype)

    def embed(self, scan, proprio, priv):
        """The per-step embedding, privileged adapter applied. See `Actor.embed`."""
        if self._adapt is not None:
            priv = self._adapt(priv)
        return torch.cat([self.stem(scan), self.pro(torch.cat([proprio, priv], 1))], 1)

    def head(self, x, h=None, use_memory: bool = True, rows=None):
        """(value, next hidden) from a per-step embedding."""
        mem = self.memory if use_memory else None
        if mem is None:
            return self.mlp(x).squeeze(1), None       # legacy path, byte-for-byte what it always was
        lin0, act0, lin1, act1, lin2 = self.mlp[0], self.mlp[1], self.mlp[2], self.mlp[3], self.mlp[4]
        delta, h_next, _enc = mem.step(x, h, rows)
        return lin2(act1(lin1(act0(lin0(x) + delta)))).squeeze(1), h_next

    @property
    def has_motion(self) -> bool:
        return self.memory is not None and self.memory.motion is not None

    def motion_input(self, scan):
        """The aligned rows this critic's motion branch reads. See `Actor.motion_input`."""
        if not self.has_motion:
            return None
        extra = self.stem.split_channels(scan)[1]
        if extra is None:
            raise ValueError("this critic carries a motion branch but the scan has no extra channels")
        return extra.index_select(1, torch.as_tensor(self.motion_rows, device=scan.device))

    def attach_motion(self, motion: dict, rows: Sequence[int]) -> None:
        """The critic's own motion branch, on the same terms the actor's is built on."""
        if self.memory is None:
            raise ValueError("the critic's motion memory needs its main memory (--memory-critic own)")
        spec = motion_spec(**motion)
        self.motion_rows = tuple(int(r) for r in rows)
        self.memory.attach_motion(len(self.motion_rows), spec)

    def forward(self, scan, proprio, priv):
        if self.memory is not None:
            raise RuntimeError("Critic.forward() is the feedforward entry point and this critic "
                               "carries memory; use step(), which takes and returns the hidden state.")
        return self.head(self.embed(scan, proprio, priv))[0]

    def step(self, scan, proprio, priv, h=None, use_memory: bool = True):
        return self.head(self.embed(scan, proprio, priv), h, use_memory, self.motion_input(scan))


def name_seed(seed: int, name: str) -> int:
    """A stable per-module seed from `(run seed, module path)`.

    `hash()` is salted per process for strings, so two runs of the same command would disagree; a
    digest is stable across processes, machines and Python versions, which is what "reproducible from
    a seed" has to mean.
    """
    import hashlib
    d = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (int(seed) ^ int.from_bytes(d, "big")) % (2 ** 31 - 1)


def reinit_fresh_by_name(model, fresh, seed: int) -> list:
    """Re-initialise every module whose parameters are all new, from its own NAME. Returns the list.

    Why this exists (`docs/research/motion-memory-2026-09-14.md`): two arms that differ only in how
    many scan channels they enable do not differ only in that. The wider first convolution has more
    parameters, so it draws more numbers from the ambient generator, so every module built after it
    -- including the GRUs a warm start leaves fresh -- gets different weights from the same `--seed`.
    Measured on the phase-1 arms, `actor.memory.gru.weight_ih_l0` differed by up to 0.176 between two
    arms meant to differ by one flag, which is a second difference nobody asked for in a comparison
    built to isolate one.

    Seeding each fresh module from `(seed, its own qualified name)` removes it: a module of the same
    shape and the same name is initialised identically whatever was built before it. Modules are
    re-initialised through their own `reset_parameters`, so each keeps the distribution PyTorch gives
    it rather than one invented here.

    Only modules ALL of whose direct parameters are fresh are touched -- a module holding a single
    copied weight is left exactly as the checkpoint wrote it.
    """
    done = []
    fresh = set(fresh)
    for name, mod in model.named_modules():
        own = [f"{name}.{p}" if name else p for p, _ in mod.named_parameters(recurse=False)]
        if not own or not all(o in fresh for o in own):
            continue
        if not hasattr(mod, "reset_parameters"):
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(name_seed(seed, name))
            mod.reset_parameters()
        done.append(name)
    return done


def _float_pair(mot):
    """The motion aux pair in float32, `(None, None)` passed through unchanged.

    Cast here rather than at the four call sites for the same reason every other head is: under
    `--amp` these come out of an autocast region in bf16, and a loss computed in bf16 against a
    float32 label is a silently different loss.
    """
    a, b = mot if mot is not None else (None, None)
    return (None if a is None else a.float(), None if b is None else b.float())


def name_seed(seed: int, name: str) -> int:
    """A stable per-module seed from `(run seed, module path)`.

    Copied from branch `feat/motion-memory` (`learn/model.py`, worker 15) rather than rewritten: the
    problem and the fix are identical here, and two implementations of the same discipline is how
    two branches stop being comparable.

    `hash()` is salted per process for strings, so two runs of the same command would disagree; a
    digest is stable across processes, machines and Python versions, which is what "reproducible
    from a seed" has to mean.
    """
    import hashlib
    d = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (int(seed) ^ int.from_bytes(d, "big")) % (2 ** 31 - 1)


def reinit_fresh_by_name(model, fresh, seed: int) -> list:
    """Re-initialise every module whose parameters are all new, from its own NAME. Returns the list.

    Also from `feat/motion-memory`, and needed here for the same reason and a second one. Theirs:
    two arms that differ only in how many scan channels they enable do not differ only in that --
    the wider first convolution has more parameters, so it draws more numbers from the ambient
    generator, so every module built after it (the GRUs a warm start leaves fresh) gets different
    weights from the same `--seed`; measured, `actor.memory.gru.weight_ih_l0` differed by up to
    0.176 between two arms meant to differ by one flag. Ours: the four arms of this branch differ by
    the WIDTH OF THE PROPRIO VECTOR (0 / 6 / 10 / 26 extra columns), which moves `actor.pro.0` and
    `critic.pro.0` by exactly the same mechanism. Without this, "A3 beats A0" could be four
    different GRU initialisations.

    Seeding each fresh module from `(seed, its own qualified name)` removes it: a module of the same
    shape and the same name is initialised identically whatever was built before it. Modules are
    re-initialised through their own `reset_parameters`, so each keeps the distribution PyTorch
    gives it rather than one invented here.

    Only modules ALL of whose direct parameters are fresh are touched -- a module holding a single
    copied weight is left exactly as the checkpoint wrote it.
    """
    done = []
    fresh = set(fresh)
    for name, mod in model.named_modules():
        own = [f"{name}.{p}" if name else p for p, _ in mod.named_parameters(recurse=False)]
        if not own or not all(o in fresh for o in own):
            continue
        if not hasattr(mod, "reset_parameters"):
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(name_seed(seed, name))
            mod.reset_parameters()
        done.append(name)
    return done


def scan_channel_spec(scan_channels: Optional[dict]) -> dict:
    """The validated `meta["scan_channels"]` block, or `{}` when no channel is enabled.

    The names are checked against `obs.SCAN_CHANNELS` and re-ordered into that canonical order:
    the order is the layout of the first convolution's trailing input columns, so a checkpoint
    written as ("edges", "memory") and rebuilt as ("memory", "edges") would read two channels that
    mean the wrong thing while every shape still matched.
    """
    if not scan_channels:
        return {}
    from .obs import SCAN_CHANNELS
    names = list((scan_channels or {}).get("channels") or ())
    unknown = [n for n in names if n not in SCAN_CHANNELS]
    if unknown:
        raise ValueError(f"unknown scan channel(s) {unknown}; known: {list(SCAN_CHANNELS)}")
    if len(set(names)) != len(names):
        raise ValueError(f"scan channels repeat: {names}")
    if not names:
        return {}
    tau = float(scan_channels.get("memory_tau_s", 2.0))
    if not tau > 0:
        raise ValueError(f"scan memory tau {tau} s must be positive")
    out = {"channels": [n for n in SCAN_CHANNELS if n in names], "memory_tau_s": tau}
    from .obs import ALIGNED_CHANNELS
    if any(n in ALIGNED_CHANNELS for n in names):
        # The `aligned` block travels with the checkpoint because the channel is not a pure function
        # of the scan: it warps with the car's measured motion, read out of the proprio vector by
        # index, and gated by thresholds that were measured rather than chosen. A checkpoint that
        # did not record them could be rebuilt with a different gate and would look the same.
        from .aligned import aligned_spec
        from .obs import MOTION_KEYS
        cfg = dict(scan_channels.get("aligned") or {})
        pro = dict(cfg.pop("proprio", {}) or {})
        if not pro:
            raise ValueError(
                "the 'aligned' scan channel needs its `proprio` index block "
                "(`learn.obs.motion_index_spec(spec)`): its warp reads "
                + ", ".join(MOTION_KEYS) + " out of the proprio vector by index.")
        missing = [k for k in ("proprio_dim", "speed", "yaw_rate", "roll", "pitch", "v_max",
                               "gyro_scale", "att_scale", "range_max") if k not in pro]
        if missing:
            raise ValueError(f"the aligned channel's proprio block is missing {missing}; build it "
                             f"with learn.obs.motion_index_spec(spec) rather than by hand")
        out["aligned"] = {**aligned_spec(**cfg), "proprio": pro}
    elif scan_channels.get("aligned"):  # noqa: SIM114 - the message is the point
        raise ValueError("scan_channels carries an 'aligned' block but not the 'aligned' channel: "
                         "one of the two is a typo, and guessing which would either build a channel "
                         "nobody asked for or drop a gate somebody measured.")
    # The `floor` block belongs to the GEOMETRIC floor channel and to the `fe_*` channels alike:
    # both read the gyro / accelerometer / ego columns out of the proprio vector by index, and the
    # front-end's own checkpoint path lives in the same block. `obs.ScanAugment` requires it for
    # either (`if "floor" in self.channels or self.fe_channels`), and this function used to reject
    # exactly that combination -- `--scan-channels fe_floor,fe_range` with no `floor` raised "a
    # `floor` block was given but the floor channel is not enabled" and killed the run at load.
    fe_names = [n for n in names if n.startswith("fe_")]
    if "floor" in names or fe_names:
        # Recorded so a checkpoint carries the channel it was trained with rather than whatever the
        # reader's defaults happen to be: which proprio columns it reads, the geometry and tolerance
        # band it was built with, and which attitude it uses.
        from .floor import FloorSpec
        blk = dict(scan_channels.get("floor") or {})
        if not blk.get("proprio"):
            raise ValueError(
                "the floor and `fe_*` channels need the `floor.proprio` block "
                "(`obs.att_index_spec`): they read the gyro and accelerometer out of the proprio "
                "vector by index, and a checkpoint that did not record the layout cannot be "
                "rebuilt against it")
        if fe_names and not (blk.get("frontend") or {}).get("path"):
            raise ValueError(
                f"channel(s) {fe_names} are a trained front-end's outputs, so the block needs "
                f"`floor.frontend.path`; there is nothing to output without it")
        blk["spec"] = FloorSpec(**(blk.get("spec") or {})).validate().to_meta()
        blk.setdefault("att_source", "tracker")
        blk.setdefault("fov", 1.5 * math.pi)
        blk.setdefault("range_eps", 0.02)
        out["floor"] = blk
    elif scan_channels.get("floor"):
        raise ValueError("a `floor` block was given but no channel that reads it is enabled: "
                         "enable `floor`, or an `fe_*` channel, or drop the block")
    return out


class ActorCritic(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, priv_dim: int, act_dim: int = 2,
                 scan_deltas: bool = False, temporal_encoder: str = "cnn", scan_stem: str = "plain",
                 cond_dim: int = 0, cond: Optional[dict] = None, priv_adapter: Optional[str] = None,
                 memory: Optional[dict] = None, scan_channels: Optional[dict] = None,
                 future_head: Optional[dict] = None, motion: Optional[dict] = None,
                 motion_heads: Optional[Sequence[str]] = None,
                 floor_head: Optional[dict] = None, opp_token: Optional[str] = None):
        super().__init__()
        mem = memory_spec(**memory) if memory else None
        chan = scan_channel_spec(scan_channels)
        extra = len(chan.get("channels", ()))
        # The head's input is the recurrent state when there is one, so which it is follows from
        # `memory` rather than being a second thing the caller can get wrong. A checkpoint that
        # recorded one source cannot be rebuilt with the other: `Actor` raises instead.
        fut = future_spec(**future_head) if future_head else None
        mot = motion_spec(**motion) if motion else None
        mot_rows = ()
        if mot:
            if not chan:
                raise ValueError("motion memory reads the observation's aligned rows and this model "
                                 "has no scan channels at all; enable them with --scan-channels")
            names = list(chan["channels"])
            missing = [r for r in mot["rows"] if r not in names]
            if missing:
                raise ValueError(f"motion memory was asked to read {missing}, which this model's "
                                 f"scan channels {names} do not include")
            # Indices within the stem's EXTRA channel block, which is how `Actor.motion_input`
            # slices them: recorded, so a later arm that enables another channel beside these does
            # not silently shift which rows the encoder reads.
            mot_rows = tuple(names.index(r) for r in mot["rows"])
        self.actor = Actor(n_stack, n_beams, proprio_dim, act_dim=act_dim, scan_deltas=scan_deltas,
                           temporal_encoder=temporal_encoder, scan_stem=scan_stem, cond_dim=cond_dim,
                           memory=mem, extra_scan_channels=extra)
        self.critic = Critic(n_stack, n_beams, proprio_dim, priv_dim, scan_deltas=scan_deltas,
                             temporal_encoder=temporal_encoder, scan_stem=scan_stem,
                             priv_adapter=priv_adapter,
                             memory=(mem if mem and mem["critic"] == "own" else None),
                             extra_scan_channels=extra)
        if mot:
            # After both halves and before the future head: the motion branch changes the width of
            # the recurrent state, which is what the future head's first layer reads.
            self.actor.attach_motion(mot, mot_rows, n_beams, motion_heads or ())
            if mot["critic"] == "own" and self.critic.memory is not None:
                self.critic.attach_motion(mot, mot_rows)
        if fut:
            # LAST, so that `--aux-future 1.0` adds a head and changes nothing else: every weight
            # above it was drawn from the same generator in the same order as in the arm without it.
            self.actor.attach_future(fut)
        #: And the per-beam floor head after THAT, for the same reason one level down: with this
        #: order, `--aux-floor` on top of an `--aux-future` arm leaves that arm's every weight where
        #: it was, and `--aux-floor` alone leaves the plain arm's.
        fl_head = None
        if floor_head:
            from .floor_head import floor_head_spec
            fl_head = floor_head_spec(**floor_head)
            self.actor.attach_floor_head(fl_head)
        self.meta = dict(n_stack=n_stack, n_beams=n_beams, proprio_dim=proprio_dim, priv_dim=priv_dim,
                         act_dim=act_dim, scan_deltas=scan_deltas, temporal_encoder=temporal_encoder,
                         scan_stem=scan_stem)
        # Recorded only when used, so a legacy checkpoint's meta is byte-for-byte what it was.
        if cond_dim:
            self.meta["cond_dim"] = int(cond_dim)
            self.meta["cond"] = dict(cond or {})
        if priv_adapter:
            self.meta["priv_adapter"] = priv_adapter
        if mem:
            self.meta["memory"] = dict(mem)
        if chan:
            self.meta["scan_channels"] = dict(chan)
        if mot:
            self.meta["motion"] = dict(self.actor.motion_spec)
            if motion_heads:
                self.meta["motion_heads"] = [h for h in ("mask", "dv") if h in motion_heads]
        if fut:
            self.meta["future_head"] = dict(self.actor.future_spec)
        if fl_head:
            self.meta["floor_head"] = dict(self.actor.floor_spec)
        # Declarative only: the block is part of `proprio_dim` above, so nothing here builds a
        # module. What it buys is that the *loader* can refuse the checkpoint (`load_checkpoint`'s
        # `allow_oracle`), which is the one place every consumer -- exporter, ROS node, viewer,
        # benchmark -- goes through. `extra["spec"]` records the same fact for a reader; `meta` is
        # what travels with the weights.
        if opp_token and str(opp_token) != "off":
            from ..opp_token import opp_token_dim, validate_opp_token
            mode = validate_opp_token(opp_token)
            self.meta["opp_token"] = mode
            if proprio_dim <= opp_token_dim(mode):
                raise ValueError(f"proprio_dim {proprio_dim} cannot hold the opponent token block "
                                 f"({opp_token_dim(mode)} columns) and an observation as well")

    @property
    def has_memory(self) -> bool:
        return self.actor.has_memory or self.critic.has_memory

    @property
    def has_future(self) -> bool:
        return self.actor.has_future

    @property
    def has_motion(self) -> bool:
        return self.actor.has_motion

    @property
    def has_floor_head(self) -> bool:
        return self.actor.has_floor_head

    def initial_hidden(self, batch: int, device=None, dtype=None) -> Optional[Hidden]:
        """An all-zero `Hidden` for `batch` rows, or None for a feedforward checkpoint.

        `Hidden()` (both fields None) means the same thing to every method here -- the GRU builds
        its own zeros -- so a caller may start from either.
        """
        if not self.has_memory:
            return None
        return Hidden(self.actor.initial_hidden(batch, device, dtype),
                      self.critic.initial_hidden(batch, device, dtype))

    @staticmethod
    def _split(h: Optional[Hidden]):
        if h is None:
            return None, None
        if not isinstance(h, Hidden):
            raise TypeError(f"hidden state must be a learn.memory.Hidden (or None), got {type(h).__name__}")
        return h.actor, h.critic

    @torch.no_grad()
    def act(self, scan, proprio, deterministic=False, c=None, h=None):
        """(action, log prob, next hidden). `h_next` is None for a feedforward checkpoint."""
        ha, hc = self._split(h)
        d, ha = self.actor.step_dist(scan, proprio, c, ha)
        a = d.mean if deterministic else d.sample()
        return a.clamp(-1, 1), d.log_prob(a).sum(1), (None if ha is None and hc is None else Hidden(ha, hc))

    def evaluate(self, scan, proprio, priv, actions, c=None, h=None):
        ha, hc = self._split(h)
        d, ha = self.actor.step_dist(scan, proprio, c, ha)
        v, hc = self.critic.step(scan, proprio, priv, hc)
        return (d.log_prob(actions).sum(1), d.entropy().sum(1), v, d,
                (None if ha is None and hc is None else Hidden(ha, hc)))

    def evaluate_aux(self, scan, proprio, priv, actions, c=None, h=None, floor: bool = False):
        """evaluate() plus the auxiliary predictions, from the same trunk pass.

        Returns (log prob, entropy, value, distribution, grip, opponent motion, future, motion,
        per-beam floor logits, hidden). `future`, `motion` and the floor logits are None -- `motion`
        is the pair (None, None) -- for a checkpoint that carries no such head, which is every
        checkpoint written before each existed; the floor logits are also None unless `floor` asks
        for them, so that head costs nothing where it is not scored.
        """
        ha, hc = self._split(h)
        mu, grip, opp, fut, mot, fl, ha = self.actor.step_all(scan, proprio, c, ha, floor=floor)
        mu = mu.float()
        d = torch.distributions.Normal(mu, self.actor.log_std.exp().expand_as(mu))
        v, hc = self.critic.step(scan, proprio, priv, hc)
        return (d.log_prob(actions).sum(1), d.entropy().sum(1), v, d, grip.float(), opp.float(),
                None if fut is None else fut.float(), _float_pair(mot),
                None if fl is None else fl.float(),
                (None if ha is None and hc is None else Hidden(ha, hc)))

    def evaluate_sequence(self, scan, proprio, priv, actions, c=None, h=None, keep=None,
                          floor: bool = False):
        """`evaluate_aux` over a (steps, envs) block, with the recurrence walked step by step.

        Shapes: `scan` (T, m, C, N), `proprio` (T, m, P), `priv` (T, m, V), `actions` (T, m, A),
        `c` (T, m, D) or None, `keep` (T, m) with 0 where the hidden state must NOT be carried into
        that step (the env ended its episode on the step before). `h` is the state at the start of
        the chunk. Returns the same tuple `evaluate_aux` does, flattened to (T * m, ...) in
        row-major (step, env) order -- the order `(T, B)` buffers flatten to -- plus the state at
        the end of the chunk.

        The convolutional stem runs once for the whole block; only the GRU and the two small MLP
        halves walk the steps. That is what makes truncated BPTT cost about what the feedforward
        update costs.
        """
        T, m = scan.shape[0], scan.shape[1]
        flat = lambda t: t.reshape(T * m, *t.shape[2:])
        ha, hc = self._split(h)
        fl = None
        if floor and self.actor.floor is not None:
            xa, pa, fl = self.actor.embed_floor(flat(scan), flat(proprio))
        else:
            xa, pa = self.actor.embed(flat(scan), flat(proprio))
        xv = self.critic.embed(flat(scan), flat(proprio), flat(priv))
        xa = xa.view(T, m, -1); xv = xv.view(T, m, -1)
        cs = None if c is None else c
        if self.actor.cond_dim and cs is None:
            raise ValueError(f"this actor is conditional (cond_dim={self.actor.cond_dim}) and "
                             f"evaluate_sequence was given no condition; pass the stored (T, m, D) "
                             f"block the actions were sampled under")
        rows_a = self.actor.motion_input(flat(scan))
        rows_c = self.critic.motion_input(flat(scan))
        view = lambda r: None if r is None else r.view(T, m, *r.shape[1:])
        rows_a, rows_c = view(rows_a), view(rows_c)
        feats, values, states, encs = [], [], [], []
        for t in range(T):
            if keep is not None:
                k = keep[t].to(xa.dtype)[None, :, None]
                if ha is not None:
                    ha = ha * k
                if hc is not None:
                    hc = hc * k
            f, ha, enc = self.actor.head(xa[t], None if cs is None else cs[t], ha,
                                         rows=None if rows_a is None else rows_a[t])
            v, hc = self.critic.head(xv[t], hc, rows=None if rows_c is None else rows_c[t])
            feats.append(f); values.append(v)
            if enc is not None:
                encs.append(enc)
            if ha is not None:
                # the state AFTER step t, which is what the future head reads and what
                # `probe_hidden` regresses from -- collected here so the head sees, step for step,
                # the tensor the rollout carried.
                states.append(ha[-1])
        feat = torch.cat(feats, 0)                       # (T * m, hidden), row-major (step, env)
        val = torch.cat(values, 0)
        mu = torch.tanh(self.actor.mu(feat)).float()
        d = torch.distributions.Normal(mu, self.actor.log_std.exp().expand_as(mu))
        grip = self.actor.grip(torch.cat([feat, pa], 1))[:, 0]
        opp = self.actor.opp(feat)
        h_seq = None if not states else torch.cat(states, 0)[None]
        fut = self.actor.future_from(feat, h_seq)
        mot = self.actor.motion_aux(None if not encs else torch.cat(encs, 0), h_seq)
        acts = flat(actions)
        return (d.log_prob(acts).sum(1), d.entropy().sum(1), val, d, grip.float(), opp.float(),
                None if fut is None else fut.float(), _float_pair(mot),
                None if fl is None else fl.float(),
                (None if ha is None and hc is None else Hidden(ha, hc)))


def save_checkpoint(path, model: ActorCritic, extra: Optional[dict] = None):
    torch.save({"state_dict": model.state_dict(), "meta": model.meta, "extra": extra or {}}, path)


def controller_arm_of(ck: dict) -> str:
    """The controller arm a checkpoint was trained under. `legacy` for everything written before.

    Read from `extra`, not `meta`: `meta` is splatted into `ActorCritic(**meta)`, so a key added
    there becomes a constructor argument.
    """
    exp = (ck.get("extra") or {}).get("experiment") or {}
    return str(((exp.get("controller") or {}).get("arm")) or "legacy")


def _refuse_controller(ck: dict, path, allow_controller: bool) -> str:
    """A policy trained against a non-legacy controller does not run on the legacy one.

    Its actor learned to emit plans for a tracker whose speed profile and acceleration bounds were
    friction-limited; replayed through the untouched `mpc.solve` those plans mean something else.
    The `estimated` arm additionally needs its frozen estimator present to reproduce what it saw. So
    the default is refusal, and a caller that can supply the controller opts in explicitly -- the
    same posture `allow_conditional` takes, and for the same reason.
    """
    arm = controller_arm_of(ck)
    if arm != "legacy" and not allow_controller:
        exp = (ck.get("extra") or {}).get("experiment") or {}
        c = exp.get("controller") or {}
        raise ValueError(
            f"{os.path.basename(str(path))} was trained with controller arm '{arm}'"
            f"{' (estimator: ' + str(c.get('estimator_path')) + ')' if c.get('estimator_path') else ''}. "
            f"Running it on the legacy tracker would evaluate its plans under a controller it never "
            f"saw. Pass allow_controller=True from a caller that installs the matching runtime.")
    return arm


def oracle_inputs_of(ck: dict) -> str:
    """The privileged observation block a checkpoint was trained with, or "off".

    Read from `meta` (where `ActorCritic` records it) and, failing that, from `extra["spec"]`
    (where the trainer records the whole `ObsSpec`). Both, because the two are written by different
    code paths and a checkpoint that has only one of them is still an oracle.
    """
    meta = ck.get("meta") or {}
    mode = meta.get("opp_token") or ((ck.get("extra") or {}).get("spec") or {}).get("opp_token")
    return str(mode or "off")


def _refuse_oracle(ck: dict, path, allow_oracle: bool) -> str:
    """A policy trained on privileged opponent tokens cannot be deployed, exported or benchmarked
    as a LiDAR-only policy.

    `f1sim.opp_token` is the simulator's ground truth about the other cars -- exact relative
    position, exact velocity, and the opponent's own intended trajectory up to 0.75 s ahead. The
    real car has none of it and there is no degraded substitute: a policy that was trained to
    believe those columns and is then fed zeros is a policy driving on a lie. So the default is
    refusal, and a caller that runs the simulator (which can produce the block) opts in -- the same
    posture `allow_conditional` and `allow_controller` take, for the same reason.
    """
    mode = oracle_inputs_of(ck)
    if mode != "off" and not allow_oracle:
        raise ValueError(
            f"{os.path.basename(str(path))} was trained with privileged opponent tokens "
            f"(opp_token='{mode}'). They are an ORACLE produced by the simulator: the car cannot "
            f"build them, so this checkpoint is not deployable and must not be exported. It can "
            f"only be run inside a simulator that supplies the same block -- pass allow_oracle=True "
            f"from such a caller. See docs/research/oracle-planner-2026-09-15.md.")
    return mode


def load_checkpoint(path, device="cpu", override: Optional[dict] = None,
                    allow_conditional: bool = False, strict_names: bool = False,
                    priv_adapter: Optional[str] = None,
                    allow_controller: bool = False,
                    allow_oracle: bool = False) -> Tuple[ActorCritic, dict]:
    """override: meta fields to change (e.g. priv_dim for a multi-car critic, scan_stack); tensors whose
    shape no longer matches are left at their fresh initialization and listed in extra["skipped"].

    `priv_adapter` names a privileged-vector adapter to build the critic with. It is separate from
    `override` on purpose: overriding `priv_dim` alone widens the critic and loads the checkpoint's
    weights into it, but leaves no adapter, so the env's narrower vector reaches it unmapped. Passing
    `None` (the default) leaves whatever the checkpoint recorded.

    `allow_controller` gates a checkpoint trained against a non-legacy plan controller -- see
    `_refuse_controller`. `allow_oracle` gates one trained on privileged opponent tokens -- see
    `_refuse_oracle`; it is off by default, so the exporter and the ROS node refuse such a
    checkpoint without having to know it exists.

    `allow_conditional` gates checkpoints that need an input the caller may not be able to produce.
    A conditional actor requires an explicit `c` at every forward, and a lab-oracle arm's `c` is the
    true friction -- privileged, and unavailable on the car. The viewer, `export`, `watch`,
    `evaluate` and the ROS node all call this without the flag, so they refuse such a checkpoint by
    default rather than running a policy whose input they would have to invent.

    `strict_names` refuses a load that would leave any tensor at fresh initialization. The tolerant
    default is what lets an existing run widen a critic or add a head; a controlled experiment wants
    the opposite, because "the migration silently reinitialised something" and "the arms differ" look
    identical in the results.
    """
    ck = torch.load(path, map_location=device)
    meta = dict(ck["meta"]); meta.update(override or {})
    if meta.pop("residual_plan", False):
        raise ValueError("experimental residual-plan checkpoints are not supported")
    _refuse_controller(ck, path, allow_controller)
    _refuse_oracle(ck, path, allow_oracle)
    if priv_adapter is not None:
        meta["priv_adapter"] = priv_adapter
    from .conditioning import CondSpec
    cond_spec = CondSpec.from_meta(meta.get("cond"))    # raises on an unsupported or inconsistent spec
    cond_meta = cond_spec.to_meta()
    if cond_spec.dim != int(meta.get("cond_dim", 0)):
        raise ValueError(f"checkpoint cond_dim {meta.get('cond_dim', 0)} disagrees with its "
                         f"conditioning metadata dim {cond_spec.dim}")
    if not allow_conditional and (int(meta.get("cond_dim", 0)) or cond_meta.get("lab_oracle")):
        raise ValueError(
            f"{os.path.basename(str(path))} is a conditional checkpoint "
            f"(cond_dim={meta.get('cond_dim', 0)}, source={cond_meta.get('source', '?')}, "
            f"lab_oracle={bool(cond_meta.get('lab_oracle'))}). It cannot be run without an explicit "
            f"conditioning input, and a lab-oracle arm's input is privileged and does not exist on "
            f"the car. Pass allow_conditional=True only from a caller that supplies it.")
    import inspect
    known = set(inspect.signature(ActorCritic.__init__).parameters) - {"self"}
    unknown = sorted(k for k in meta if k not in known)
    if unknown:
        hint = ""
        if "opp_token" in unknown:
            hint = (" 'opp_token' marks a privileged-opponent (oracle) checkpoint from the oracle-planner "
                    "experiment: it needs the simulator's true opponent state as an input and cannot be "
                    "driven by the console, the exporter or the ROS node; the A0 control arm (opp_token off) "
                    "has no such key and loads normally.")
        raise ValueError(
            f"{os.path.basename(str(path))} was saved by a newer or different branch: its meta has "
            f"field(s) this tree's ActorCritic does not know {unknown}.{hint}")
    m = ActorCritic(**meta).to(device)
    sd = m.state_dict(); skipped = []
    for k_, v in ck["state_dict"].items():
        if k_ in sd and sd[k_].shape == v.shape:
            sd[k_] = v
        else:
            skipped.append(k_)
    if strict_names:
        fresh = [k_ for k_ in sd if k_ not in ck["state_dict"]]
        if skipped or fresh:
            raise ValueError(
                f"strict load failed: {len(skipped)} checkpoint tensor(s) unused {sorted(skipped)[:6]}, "
                f"{len(fresh)} model tensor(s) left at fresh init {sorted(fresh)[:6]}. Migrate by "
                f"name deliberately instead of relying on shape-matching.")
    m.load_state_dict(sd)
    extra = dict(ck.get("extra", {})); extra["skipped"] = skipped
    return m, extra


def load_for_conditioning(path, device, cond_dim: int, cond_meta: dict,
                          priv_adapter: Optional[str] = None,
                          override: Optional[dict] = None,
                          allow_controller: bool = False) -> Tuple[ActorCritic, dict, list]:
    """Load an unconditional checkpoint into a conditional actor, by name, preserving every weight.

    The only tensor that may be fresh is the conditioning projection, which is zero anyway. Anything
    else left fresh means the architectures do not line up and the experiment would be comparing two
    different initialisations rather than two conditioning sources -- so it raises.

    `allow_controller` is the same gate as in `load_checkpoint`: this is the other loader path into
    a training job, so leaving it open would let a controller-trained checkpoint in through the side.
    """
    ck = torch.load(path, map_location=device)
    meta = dict(ck["meta"]); meta.update(override or {})
    if meta.pop("residual_plan", False):
        raise ValueError("experimental residual-plan checkpoints are not supported")
    _refuse_controller(ck, path, allow_controller)
    if int(meta.get("cond_dim", 0)):
        raise ValueError("expected an unconditional checkpoint to migrate from")
    from .conditioning import CondSpec
    spec = CondSpec.from_meta(dict(cond_meta))          # raises on an unsupported or inconsistent spec
    if spec.dim != int(cond_dim):
        raise ValueError(f"cond_dim {cond_dim} disagrees with the spec's dim {spec.dim}")
    meta.update(cond_dim=int(cond_dim), cond=spec.to_meta(), priv_adapter=priv_adapter)
    m = ActorCritic(**meta).to(device)
    sd = m.state_dict()
    allowed_fresh = {k for k in sd if k.startswith("actor.cond.")}
    unused = [k for k in ck["state_dict"] if k not in sd or sd[k].shape != ck["state_dict"][k].shape]
    fresh = [k for k in sd if k not in ck["state_dict"]]
    if unused or set(fresh) - allowed_fresh:
        raise ValueError(
            f"conditional migration is not clean: unused checkpoint tensors {sorted(unused)[:6]}, "
            f"unexpected fresh tensors {sorted(set(fresh) - allowed_fresh)[:6]}. Every legacy weight "
            f"must transfer unchanged; only actor.cond.* may be new.")
    for k, v in ck["state_dict"].items():
        sd[k] = v
    m.load_state_dict(sd)
    extra = dict(ck.get("extra", {})); extra["skipped"] = []
    return m, extra, sorted(fresh)


def _proprio_growth(name: str, src: torch.Tensor, dst: torch.Tensor, p_old: int, k: int):
    """Where the `k` new proprio columns sit in a first-layer weight, or None if this is not one.

    The actor's proprio MLP takes the proprio vector alone, so the new columns are the trailing
    ones. The critic's takes `cat([proprio, priv])`, so they are INSERTED at `p_old` and every
    privileged column shifts right. Getting that wrong is silent: the shapes match either way, and
    the critic would simply read the wrong number for every privileged input it has.

    Returns `(before, after)`: how many of the destination's columns come from the source's head and
    from its tail, with `k` zeros between them.
    """
    if src.dim() != 2 or dst.dim() != 2 or src.shape[0] != dst.shape[0]:
        return None
    if dst.shape[1] != src.shape[1] + k:
        return None
    if name.startswith("actor."):
        if src.shape[1] != p_old:
            return None
        return (p_old, 0)
    if name.startswith("critic."):
        if src.shape[1] < p_old:
            return None
        return (p_old, src.shape[1] - p_old)
    return None


def grow_proprio_moment(name: str, saved: torch.Tensor, target: torch.Tensor,
                        p_old: int, k: int) -> Optional[torch.Tensor]:
    """A saved optimiser moment for a proprio input layer, widened to the token block's layout.

    Adam's `exp_avg` and `exp_avg_sq` are element-wise, so they move with the weight: the block's
    own columns get 0, which is what Adam holds for a coefficient that has not had a gradient yet,
    and every original column keeps its moment. Without this the two widened layers' moments would
    be *dropped* in the oracle arms and *restored* in the control, which is a second difference
    between arms that are supposed to differ by their input width alone.

    Returns None when this tensor is not one of the two, or does not line up.
    """
    if saved.shape == target.shape:
        return saved
    split = _proprio_growth(name, saved, target, p_old, k)
    if split is None:
        return None
    before, after = split
    out = torch.zeros_like(target)
    out[:, :before] = saved[:, :before].to(out.dtype)
    if after:
        out[:, before + k:] = saved[:, before:].to(out.dtype)
    return out


def load_for_memory(path, device, memory: Optional[dict] = None,
                    scan_channels: Optional[dict] = None,
                    priv_adapter: Optional[str] = None, override: Optional[dict] = None,
                    allow_controller: bool = False,
                    allow_conditional: bool = False,
                    future_head: Optional[dict] = None,
                    motion: Optional[dict] = None,
                    motion_heads: Optional[Sequence[str]] = None,
                    floor_head: Optional[dict] = None,
                    opp_token: Optional[str] = None,
                    init_seed: Optional[int] = None) -> Tuple[ActorCritic, dict, list]:
    """Load a feedforward checkpoint into a recurrent actor-critic, by name, preserving every weight.

    Warm start, not re-initialisation. The memory is an addition to the original network, so at
    step 0 the result *is* the original:

    * every tensor the checkpoint holds is copied under its own name -- nothing is shape-matched
      into the wrong slot and nothing is left at fresh init except what is listed below;
    * the GRU's output projection is zero (`GRUMemory.__init__`), so the term it adds to the first
      MLP preactivation is exactly 0.0 and the actor's action, the aux heads and the critic's value
      are bit-identical to the original's for any input and any hidden state;
    * when extra scan channels are enabled the first convolution gains input columns. They are
      appended after every column the original had, and the new ones are zeroed, so that forward is
      bit-identical too -- an extra channel starts as an input the network ignores and learns to
      use, exactly like the memory.

    `tests/test_memory_model.py::test_warm_start_is_bit_identical` is the check, run against the
    real frozen baseline when it is present and against a small stand-in otherwise.

    `future_head` adds the auxiliary future head (`learn.future`) on the same terms: its output
    layer is zero, so it changes no action, no value and no other aux head, and `actor.future.*` is
    the third family of tensors allowed to be new. It reads the recurrent state when `memory` is
    given and the trunk features otherwise, so which it is follows from this call rather than being
    a second thing to keep in step.

    `floor_head` adds the per-beam floor head (`learn.floor_head`) on the identical terms, and it
    is the fourth family of tensors allowed to be new. Its output convolution is zero, so at step 0
    it predicts 0.5 for every beam and changes nothing else; it is never exported and the actor's
    forward never calls it.

    `opp_token` adds the privileged opponent block (`f1sim.opp_token`) to the observation, which
    widens the proprio vector. The two first linear layers that read it gain input columns, and the
    new ones are **zero**, so the warm start is bit-identical in the same sense the scan channels
    are: the block starts as an input the network ignores and can learn to use. The actor's new
    columns are appended; the critic's are inserted before its privileged columns, because its
    proprio MLP reads `cat([proprio, priv])` -- see `_proprio_growth`. This is an ORACLE and the
    checkpoint records it, so every loader that is not a simulator refuses to open the result.

    `init_seed` re-initialises every module a warm start leaves entirely fresh from its own name
    (`reinit_fresh_by_name`), then re-zeroes the projections. It is what makes arms that differ by
    an input width comparable: without it the wider first layer consumes different draws from the
    ambient generator and every module built after it differs too. `None` is the previous
    behaviour exactly.

    `memory` may be None: extra scan channels alone are a legitimate arm, and they need the same
    by-name transfer and the same zeroed new columns. At least one of the four must be asked for,
    or this is `load_checkpoint` with extra steps.

    Returns (model, extra, fresh tensor names). `allow_controller` and `allow_conditional` are the
    same gates `load_checkpoint` documents; this is another loader path into a training job, so
    leaving them open would let a refused checkpoint in through the side.
    """
    ck = torch.load(path, map_location=device)
    meta = dict(ck["meta"]); meta.update(override or {})
    if meta.pop("residual_plan", False):
        raise ValueError("experimental residual-plan checkpoints are not supported")
    _refuse_controller(ck, path, allow_controller)
    if meta.get("future_head"):
        # Same argument as memory, plus a concrete trap: the head's input width is the GRU's hidden
        # size or the trunk's, so adding memory underneath a trained head silently changes what its
        # first layer reads. Resuming keeps both.
        raise ValueError(f"{os.path.basename(str(path))} already carries a future head "
                         f"({meta['future_head']}); warm-starting one from it would re-initialise a "
                         f"path that is already trained, and adding memory underneath it would "
                         f"change the width of its input. Resume it with load_checkpoint instead.")
    if meta.get("motion") and motion:
        raise ValueError(f"{os.path.basename(str(path))} already carries a motion branch "
                         f"({meta['motion']}); warm-starting one from it would re-initialise a path "
                         f"that is already trained. Resume it with load_checkpoint instead.")
    if meta.get("floor_head") and floor_head:
        raise ValueError(f"{os.path.basename(str(path))} already carries a floor head "
                         f"({meta['floor_head']}); warm-starting one from it would re-initialise a "
                         f"path that is already trained. Resume it with load_checkpoint instead.")
    if meta.get("memory"):
        raise ValueError(f"{os.path.basename(str(path))} already carries memory "
                         f"({meta['memory']}); warm-starting memory from it would be a second "
                         f"initialisation of a path that is already trained. Resume it with "
                         f"load_checkpoint instead.")
    if not allow_conditional and int(meta.get("cond_dim", 0)):
        raise ValueError(f"{os.path.basename(str(path))} is a conditional checkpoint and needs an "
                         f"explicit conditioning input at every forward; pass allow_conditional=True "
                         f"only from a caller that supplies it.")
    if priv_adapter is not None:
        meta["priv_adapter"] = priv_adapter
    chan = scan_channel_spec(scan_channels)
    if memory:
        meta["memory"] = memory_spec(**memory)
    if chan:
        meta["scan_channels"] = chan
    if future_head:
        meta["future_head"] = future_spec(**future_head)
    if motion:
        meta["motion"] = motion_spec(**motion)
        if motion_heads:
            meta["motion_heads"] = [h for h in ("mask", "dv") if h in motion_heads]
    if floor_head:
        from .floor_head import floor_head_spec as _fhs
        if str(meta.get("scan_stem")) != "resnet":
            raise ValueError(
                f"the per-beam floor head needs the resnet scan stem and "
                f"{os.path.basename(str(path))} has {meta.get('scan_stem')!r}. See "
                f"`ScanStem.beam_channels`: the plain stack has no whole-scan context to read.")
        meta["floor_head"] = _fhs(**floor_head)
    token = "off" if opp_token is None else str(opp_token)
    if token != "off":
        from ..opp_token import opp_token_dim, validate_opp_token
        token = validate_opp_token(token)
        if oracle_inputs_of(ck) != "off":
            raise ValueError(f"{os.path.basename(str(path))} already carries opponent tokens "
                             f"({oracle_inputs_of(ck)}); warm-starting them from it would zero "
                             f"input columns that are already trained. Resume it with "
                             f"load_checkpoint(allow_oracle=True) instead.")
        meta["opp_token"] = token
    if not (memory or chan or future_head or motion or floor_head) and token == "off":
        raise ValueError("load_for_memory with neither memory, a scan channel, a future head, a "
                         "motion branch, a floor head nor an opponent token would be "
                         "load_checkpoint with extra steps; call that instead.")
    p_old = int(ck["meta"].get("proprio_dim", 0))
    k_tok = opp_token_dim(token) if token != "off" else 0
    if k_tok and int(meta.get("proprio_dim", 0)) != p_old + k_tok:
        raise ValueError(f"opp_token {token!r} adds {k_tok} proprio columns to the checkpoint's "
                         f"{p_old}, i.e. {p_old + k_tok}, but this run's observation spec says "
                         f"{meta.get('proprio_dim')}. The block is appended after every existing "
                         f"key, so anything else means the two sides disagree about the layout.")
    m = ActorCritic(**meta).to(device)
    sd = m.state_dict()
    src = ck["state_dict"]

    #: The only tensors allowed to be new. Everything else must arrive from the checkpoint.
    #: `.memory.` covers the motion branch too -- it lives inside `GRUMemory` -- and the train-time
    #: heads are named separately because they hang off the actor.
    allowed_fresh = {k for k in sd if ".memory." in k or k.startswith("actor.future.")
                     or k.startswith("actor.opp_mask.") or k.startswith("actor.dv.")
                     or k.startswith("actor.floor.")}
    #: name -> (column the zeros are inserted at, how many). Everything left of it keeps its index
    #: and everything right of it is shifted, which is what makes the forward bit-identical.
    grown = {}
    widened = {}                                    # name -> (columns before the block, columns after)
    unused, mismatched = [], []
    for k, v in src.items():
        if k not in sd:
            unused.append(k); continue
        if sd[k].shape == v.shape:
            continue
        # An input-channel extension of the stem's first convolution is one legal reshape: same
        # rank, same everything but the input-channel axis, and only growth.
        if (chan and k.endswith(".weight") and v.dim() == 3 and sd[k].dim() == 3
                and sd[k].shape[0] == v.shape[0] and sd[k].shape[2] == v.shape[2]
                and sd[k].shape[1] == v.shape[1] + len(chan["channels"])):
            grown[k] = (v.shape[1], len(chan["channels"]))
            continue
        # The other is the proprio embedding gaining the privileged opponent block. The actor's
        # input is the proprio vector, so the columns go on the end; the critic's is
        # cat([proprio, priv]), so they go in at the old proprio width -- `_proprio_growth`.
        split = _proprio_growth(k, v, sd[k], p_old, k_tok) if k_tok else None
        if split is not None:
            widened[k] = split
        else:
            mismatched.append((k, tuple(v.shape), tuple(sd[k].shape)))
    fresh = [k for k in sd if k not in src]
    if unused or mismatched or set(fresh) - allowed_fresh:
        hint = ""
        if any(k.startswith("critic.pro.") for k, _c, _m in mismatched):
            hint = (" The critic's privileged input changed width, which is what happens when the "
                    "env's race size differs from the one the checkpoint was trained in: match "
                    "--race-size, or map the columns with --critic-priv-adapter. Silently "
                    "re-initialising that layer would warm-start a critic that has forgotten how "
                    "to value an opponent.")
        raise ValueError(
            f"memory warm start is not clean: unused checkpoint tensors {sorted(unused)[:6]}, "
            f"shape mismatches {mismatched[:4]}, unexpected fresh tensors "
            f"{sorted(set(fresh) - allowed_fresh)[:6]}. Every original weight must transfer "
            f"unchanged; only the memory modules, the future head (and the zeroed new scan-channel "
            f"columns) may be new.{hint}")
    n_conv = sum(1 for k in grown if src[k].dim() == 3)
    if chan and n_conv != 2:
        raise ValueError(f"expected the actor's and the critic's first convolution to grow by "
                         f"{len(chan['channels'])} input channel(s); {n_conv} did: {sorted(grown)}")
    if k_tok and sorted(widened) != ["actor.pro.0.weight", "critic.pro.0.weight"]:
        raise ValueError(f"expected exactly the actor's and the critic's proprio input layer to "
                         f"gain {k_tok} column(s); {sorted(widened)} did. A proprio MLP whose width "
                         f"did not change is one that is not reading the block, and one that "
                         f"changed elsewhere is a re-layout rather than an append.")
    with torch.no_grad():
        for k, v in src.items():
            if k in grown:
                at, n = grown[k]
                w = torch.zeros_like(sd[k])
                v = v.to(sd[k].dtype)
                w[:, :at] = v[:, :at]                        # originals keep their index ...
                w[:, at + n:] = v[:, at:]                    # ... and the new columns are 0
                sd[k] = w
            elif k in widened:
                before, after = widened[k]
                w = torch.zeros_like(sd[k])
                w[:, :before] = v[:, :before].to(sd[k].dtype)
                if after:
                    w[:, before + k_tok:] = v[:, before:].to(sd[k].dtype)
                sd[k] = w                                    # the block's own columns stay 0
            else:
                sd[k] = v
    m.load_state_dict(sd)
    if init_seed is not None:
        # Before the zero checks below, because a re-initialised projection is not zero any more and
        # the zeros are re-applied straight after. See `reinit_fresh_by_name`.
        renamed = reinit_fresh_by_name(m, fresh, init_seed)
        with torch.no_grad():
            for mod in (m.actor, m.critic):
                mem_ = getattr(mod, "memory", None)
                if mem_ is not None:
                    mem_.out.weight.zero_()
                    if mem_.motion is not None:
                        mem_.motion.out.weight.zero_()
            for head in (m.actor.future, m.actor.opp_mask, m.actor.dv):
                if head is not None:
                    head.net[2].weight.zero_(); head.net[2].bias.zero_()
        #: `and fresh`: an arm whose only addition is the opponent block has NO fresh tensor at all
        #: (the block is zeroed columns of an existing layer), and `--name-seed-fresh` on it is a
        #: no-op rather than a mistake. Without the guard the control arm of the oracle experiment
        #: could not be launched with the same flags as the arms it controls for.
        if not renamed and fresh:
            raise RuntimeError(
                f"init_seed={init_seed} was given and no fresh module was re-initialised from its "
                f"name, although {len(fresh)} tensor(s) are fresh. The fresh names no longer match "
                f"the modules, which would leave the arms differently initialised while claiming "
                f"they are not.")
    # The claim the whole warm start rests on, checked rather than assumed. RuntimeError, not
    # assert: `python -O` strips asserts and this is what makes the result the original at step 0.
    for who, mod in (("actor", m.actor), ("critic", m.critic)):
        if mod.memory is not None and float(mod.memory.out.weight.detach().abs().max()) != 0.0:
            raise RuntimeError(f"the {who}'s memory projection is not zero at init; a warm start "
                               f"from it would not reproduce the original")
    for who, mod in (("actor", m.actor), ("critic", m.critic)):
        mm = getattr(mod, "memory", None)
        if mm is not None and mm.motion is not None and float(mm.motion.out.weight.detach().abs().max()) != 0.0:
            raise RuntimeError(f"the {who}'s motion projection is not zero at init; a warm start "
                               f"from it would not reproduce the original")
    if m.actor.opp_mask is not None:
        out = m.actor.opp_mask.net[2]
        if float(out.weight.detach().abs().max()) != 0.0 or float(out.bias.detach().abs().max()) != 0.0:
            raise RuntimeError("the beam-mask head's output layer is not zero at init")
    if m.actor.dv is not None:
        out = m.actor.dv.net[2]
        if float(out.weight.detach().abs().max()) != 0.0 or float(out.bias.detach().abs().max()) != 0.0:
            raise RuntimeError("the current-Dv head's output layer is not zero at init")
    if m.actor.future is not None:
        out = m.actor.future.net[2]
        if float(out.weight.detach().abs().max()) != 0.0 or float(out.bias.detach().abs().max()) != 0.0:
            raise RuntimeError("the future head's output layer is not zero at init; the head would "
                               "start by asserting a future nobody trained it to predict")
    extra = dict(ck.get("extra", {})); extra["skipped"] = []
    return m, extra, sorted(fresh)
