"""Actor-critic for the LiDAR-only policy. Actor sees (scan stack, proprio); the critic additionally
sees the privileged vector (asymmetric actor-critic). Actor is TensorRT-friendly (conv1d + MLP)."""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import GRUMemory, Hidden, memory_spec


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

    def _resnet_features(self, x: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.trunk(self._augment(x, extra))
        pooled = torch.cat([-F.adaptive_max_pool1d(-h, 1).flatten(1), h.mean(2)], 1)
        return torch.cat([self.neck(h).flatten(1), pooled, self._sector_profile(x)], 1)

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
                 extra_scan_channels: int = 0):
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

    def initial_hidden(self, batch: int, device=None, dtype=None):
        return None if self.memory is None else self.memory.initial(batch, device, dtype)

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

    def head(self, x, c=None, h=None, use_memory: bool = True):
        """(actor features, next hidden) from a per-step embedding."""
        mem = self.memory if use_memory else None
        if self.cond is None and mem is None:
            return self.mlp(x), None                 # legacy path, byte-for-byte what it always was
        # Conditional / recurrent: the same three layers, with the extra terms added to the first
        # preactivation. Written out rather than sliced, because `self.mlp[1:]` builds a new
        # Sequential on every forward and this runs once per env step.
        lin0, act0, lin1, act1 = self.mlp[0], self.mlp[1], self.mlp[2], self.mlp[3]
        pre = lin0(x)
        if self.cond is not None:
            pre = pre + self.cond(c.to(x.dtype))
        h_next = None
        if mem is not None:
            delta, h_next = mem.step(x, h)
            pre = pre + delta
        return act1(lin1(act0(pre))), h_next

    def _parts(self, scan, proprio, c=None, h=None, use_memory: bool = True):
        c = self._require_cond(c, proprio.shape[0])
        x, p = self.embed(scan, proprio)
        feat, h_next = self.head(x, c, h, use_memory)
        return feat, p, h_next

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
        feat, _p, h_next = self._parts(scan, proprio, c, h, use_memory)
        return torch.tanh(self.mu(feat)), h_next

    def step_all(self, scan, proprio, c=None, h=None, use_memory: bool = True):
        feat, p, h_next = self._parts(scan, proprio, c, h, use_memory)
        return (torch.tanh(self.mu(feat)), self.grip(torch.cat([feat, p], 1))[:, 0],
                self.opp(feat), h_next)

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

    def head(self, x, h=None, use_memory: bool = True):
        """(value, next hidden) from a per-step embedding."""
        mem = self.memory if use_memory else None
        if mem is None:
            return self.mlp(x).squeeze(1), None       # legacy path, byte-for-byte what it always was
        lin0, act0, lin1, act1, lin2 = self.mlp[0], self.mlp[1], self.mlp[2], self.mlp[3], self.mlp[4]
        delta, h_next = mem.step(x, h)
        return lin2(act1(lin1(act0(lin0(x) + delta)))).squeeze(1), h_next

    def forward(self, scan, proprio, priv):
        if self.memory is not None:
            raise RuntimeError("Critic.forward() is the feedforward entry point and this critic "
                               "carries memory; use step(), which takes and returns the hidden state.")
        return self.head(self.embed(scan, proprio, priv))[0]

    def step(self, scan, proprio, priv, h=None, use_memory: bool = True):
        return self.head(self.embed(scan, proprio, priv), h, use_memory)


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
    return {"channels": [n for n in SCAN_CHANNELS if n in names], "memory_tau_s": tau}


class ActorCritic(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, priv_dim: int, act_dim: int = 2,
                 scan_deltas: bool = False, temporal_encoder: str = "cnn", scan_stem: str = "plain",
                 cond_dim: int = 0, cond: Optional[dict] = None, priv_adapter: Optional[str] = None,
                 memory: Optional[dict] = None, scan_channels: Optional[dict] = None):
        super().__init__()
        mem = memory_spec(**memory) if memory else None
        chan = scan_channel_spec(scan_channels)
        extra = len(chan.get("channels", ()))
        self.actor = Actor(n_stack, n_beams, proprio_dim, act_dim=act_dim, scan_deltas=scan_deltas,
                           temporal_encoder=temporal_encoder, scan_stem=scan_stem, cond_dim=cond_dim,
                           memory=mem, extra_scan_channels=extra)
        self.critic = Critic(n_stack, n_beams, proprio_dim, priv_dim, scan_deltas=scan_deltas,
                             temporal_encoder=temporal_encoder, scan_stem=scan_stem,
                             priv_adapter=priv_adapter,
                             memory=(mem if mem and mem["critic"] == "own" else None),
                             extra_scan_channels=extra)
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

    @property
    def has_memory(self) -> bool:
        return self.actor.has_memory or self.critic.has_memory

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

    def evaluate_aux(self, scan, proprio, priv, actions, c=None, h=None):
        """evaluate() plus the grip prediction, from the same trunk pass."""
        ha, hc = self._split(h)
        mu, grip, opp, ha = self.actor.step_all(scan, proprio, c, ha)
        mu = mu.float()
        d = torch.distributions.Normal(mu, self.actor.log_std.exp().expand_as(mu))
        v, hc = self.critic.step(scan, proprio, priv, hc)
        return (d.log_prob(actions).sum(1), d.entropy().sum(1), v, d, grip.float(), opp.float(),
                (None if ha is None and hc is None else Hidden(ha, hc)))

    def evaluate_sequence(self, scan, proprio, priv, actions, c=None, h=None, keep=None):
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
        xa, pa = self.actor.embed(flat(scan), flat(proprio))
        xv = self.critic.embed(flat(scan), flat(proprio), flat(priv))
        xa = xa.view(T, m, -1); xv = xv.view(T, m, -1)
        cs = None if c is None else c
        if self.actor.cond_dim and cs is None:
            raise ValueError(f"this actor is conditional (cond_dim={self.actor.cond_dim}) and "
                             f"evaluate_sequence was given no condition; pass the stored (T, m, D) "
                             f"block the actions were sampled under")
        feats, values = [], []
        for t in range(T):
            if keep is not None:
                k = keep[t].to(xa.dtype)[None, :, None]
                if ha is not None:
                    ha = ha * k
                if hc is not None:
                    hc = hc * k
            f, ha = self.actor.head(xa[t], None if cs is None else cs[t], ha)
            v, hc = self.critic.head(xv[t], hc)
            feats.append(f); values.append(v)
        feat = torch.cat(feats, 0)                       # (T * m, hidden), row-major (step, env)
        val = torch.cat(values, 0)
        mu = torch.tanh(self.actor.mu(feat)).float()
        d = torch.distributions.Normal(mu, self.actor.log_std.exp().expand_as(mu))
        grip = self.actor.grip(torch.cat([feat, pa], 1))[:, 0]
        opp = self.actor.opp(feat)
        acts = flat(actions)
        return (d.log_prob(acts).sum(1), d.entropy().sum(1), val, d, grip.float(), opp.float(),
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


def load_checkpoint(path, device="cpu", override: Optional[dict] = None,
                    allow_conditional: bool = False, strict_names: bool = False,
                    priv_adapter: Optional[str] = None,
                    allow_controller: bool = False) -> Tuple[ActorCritic, dict]:
    """override: meta fields to change (e.g. priv_dim for a multi-car critic, scan_stack); tensors whose
    shape no longer matches are left at their fresh initialization and listed in extra["skipped"].

    `priv_adapter` names a privileged-vector adapter to build the critic with. It is separate from
    `override` on purpose: overriding `priv_dim` alone widens the critic and loads the checkpoint's
    weights into it, but leaves no adapter, so the env's narrower vector reaches it unmapped. Passing
    `None` (the default) leaves whatever the checkpoint recorded.

    `allow_controller` gates a checkpoint trained against a non-legacy plan controller -- see
    `_refuse_controller`.

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


def load_for_memory(path, device, memory: Optional[dict] = None,
                    scan_channels: Optional[dict] = None,
                    priv_adapter: Optional[str] = None, override: Optional[dict] = None,
                    allow_controller: bool = False,
                    allow_conditional: bool = False) -> Tuple[ActorCritic, dict, list]:
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

    `memory` may be None: extra scan channels alone are a legitimate arm, and they need the same
    by-name transfer and the same zeroed new columns. At least one of the two must be asked for,
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
    if not memory and not chan:
        raise ValueError("load_for_memory with neither memory nor a scan channel would be "
                         "load_checkpoint with extra steps; call that instead.")
    m = ActorCritic(**meta).to(device)
    sd = m.state_dict()
    src = ck["state_dict"]

    #: The only tensors allowed to be new. Everything else must arrive from the checkpoint.
    allowed_fresh = {k for k in sd if ".memory." in k}
    grown = {}                                      # name -> (checkpoint columns, model columns)
    unused, mismatched = [], []
    for k, v in src.items():
        if k not in sd:
            unused.append(k); continue
        if sd[k].shape == v.shape:
            continue
        # An input-channel extension of the stem's first convolution is the one legal reshape: same
        # rank, same everything but the input-channel axis, and only growth.
        if (chan and k.endswith(".weight") and v.dim() == 3 and sd[k].dim() == 3
                and sd[k].shape[0] == v.shape[0] and sd[k].shape[2] == v.shape[2]
                and sd[k].shape[1] == v.shape[1] + len(chan["channels"])):
            grown[k] = (v.shape[1], sd[k].shape[1])
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
            f"unchanged; only the memory modules (and the zeroed new scan-channel columns) may be "
            f"new.{hint}")
    if chan and len(grown) != 2:
        raise ValueError(f"expected the actor's and the critic's first convolution to grow by "
                         f"{len(chan['channels'])} input channel(s); {len(grown)} did: {sorted(grown)}")
    with torch.no_grad():
        for k, v in src.items():
            if k in grown:
                w = torch.zeros_like(sd[k])
                w[:, :grown[k][0]] = v.to(sd[k].dtype)       # originals keep their columns; new ones are 0
                sd[k] = w
            else:
                sd[k] = v
    m.load_state_dict(sd)
    # The claim the whole warm start rests on, checked rather than assumed. RuntimeError, not
    # assert: `python -O` strips asserts and this is what makes the result the original at step 0.
    for who, mod in (("actor", m.actor), ("critic", m.critic)):
        if mod.memory is not None and float(mod.memory.out.weight.detach().abs().max()) != 0.0:
            raise RuntimeError(f"the {who}'s memory projection is not zero at init; a warm start "
                               f"from it would not reproduce the original")
    extra = dict(ck.get("extra", {})); extra["skipped"] = []
    return m, extra, sorted(fresh)
