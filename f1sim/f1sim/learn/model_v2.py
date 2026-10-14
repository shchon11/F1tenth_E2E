"""The v2 actor-critic: every block answers "why", or it is not here.

docs/research/architecture-redesign-2026-10-13.md (section 4 "새 구조", corrected by its appendix 2
"관문 결과와 설계 정정") with the WP4 plan output of docs/research/output-format-2026-10-14.md:

    scan (B, K, N) ─► B1 [frames | 9-tap min-pool of frame 0] ─► B2 ResNet stem, no bypass ─► s (256)
    ego  (B, 210)  ─► B3 Linear ─► GELU ─────────────────────────────────────────────────────► p (128)
    dial c (B, 1)  ───────────────────────────────────────────────────────────────────────────► c
                      B4  e = GELU(Linear([s | p | c] -> 256))
                      B5  y, h' = GRU(LayerNorm(e), h)      (128, chrono update gate)
                      B6  [e | y] (384) -> 256 -> 8, tanh   (WP4 plan; last layer x 0.1)
                      B8  [e | y] -> 3 opponent slots        (train only, `learn.aux_v2`)
    critic (train only, B9 as corrected): own 1-frame stem + [ego | c | privileged v2] -> MLP ->
            one value (for value normalisation) + a "car contact within 1 s" logit. No GRU, no actor
            features.

Why each choice, with the measurement that decided it (the redesign's ledger, section 2):

* B1/B2 -- the newest frame (K = 1..3: one and six frames read the opponent equally, EV 0.41 vs 0.40;
  the perception gate decides 2-3) and the min-pool (23.5 % of the first convolution's variance).
  No angle ramp (zero variance across samples) and no global-pool / 36-sector bypass: 0.14 % of the
  fc variance, the neck alone reads sector distance at R^2 0.81-0.95, and the bypass was what broke
  the ONNX export. 794 k parameters; G1's trainable copy of exactly this stem is what made occluded
  recall work (R^2 0.609).
* B3 -- the ego vector of contract v2 (`learn.obs.ObsSpecV2`, 210 columns, none duplicated: now,
  a_{t-2}, and 20 history rows that start with a_{t-1}'s step; 42 without the issued actions,
  `ego_actions` False -- the copycat study's arm A, docs/research/copycat-2026-10-14.md).
* B4 -- the dial is an ordinary input column: v1's zero-initialised additive projection existed
  only to keep a warm start bit-identical, and there is nothing to warm-start here. Not LayerNorm'd
  on its own (a one-wide LayerNorm is a constant, which would kill the dial -- a judge's catch).
* B5 -- LayerNorm on the GRU input and a chrono update-gate bias (unit timescales log-uniform in
  2..120 steps, i.e. 50 ms .. 3 s, the occlusion durations M1 asks for), output CONCATENATED with e.
  v1's zero-init projection was not the main cause of the memory's failure but slowed learning
  2-3x (K40: 864 vs 301 iterations), and concatenation gives the head the memory's output from the
  first update. What actually makes the memory learn is the dense B8 target (see `aux_v2`).
* B6 -- 384 -> 256 -> 8 with tanh, last layer x 0.1 (starts near the zero plan, as v1's head did),
  state-independent per-dimension log std: kappa_1, kappa_2 at -2.07 (matches linear's executed
  steering noise, 9.8 vs 7.5 deg at sigma 0.165), the rest at -1.8. Weak dimensions are not given a
  smaller sigma: the score-function noise grows as eps / sigma (output-format 5).
* B9 -- one value stream (G4: splitting streams bought nothing; value normalisation lifted the
  0.5 s foresight R from 0.085 to 0.17) plus the contact logit (to 0.21 / AUC 0.82), with the
  ground-truth geometry the information-gap gate measured (+0.049 AUC). No critic GRU (state wiped:
  correlation 0.689 vs 0.680) and no actor features (no gain on car risk, harmful with label leak).

Parameter counts, per block, are pinned against the redesign's table by `tests/test_model_v2.py`.

v1 (`learn.model`) is untouched: a v2 checkpoint says so in `meta["arch"]`, and `load_checkpoint`
hands only those here.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_v2 import AUX_HEADS, BeamCarHead, NearestCarHead, OpponentSlotHead, aux_spec
from .memory import Hidden
from .model import ResBlock1d
from .obs import MINPOOL_TAPS, ObsSpecV2

#: The contract's ego width (`learn.obs.ObsSpecV2`, 210): the default every v2 module is built at.
EGO_DIM = ObsSpecV2().ego_dim

#: `meta["arch"]` of every v2 checkpoint. A v1 checkpoint has no "arch" key at all.
ARCH = "v2"

#: B6's per-dimension initial log std: the first two curvature columns, then everything else.
LOG_STD_KAPPA12 = -2.07
LOG_STD_REST = -1.8

#: B5's chrono range [control steps]: 2 steps (50 ms) to 120 steps (3 s, requirement M1's horizon).
CHRONO_T_MIN = 2.0
CHRONO_T_MAX = 120.0


def chrono_init(gru: nn.GRU, t_min: float = CHRONO_T_MIN, t_max: float = CHRONO_T_MAX) -> torch.Tensor:
    """Set the update gate so unit i keeps its state for T_i steps, T_i log-uniform in [t_min, t_max].

    PyTorch's GRU is h' = (1 - z) n + z h with gates ordered (r, z, n), so z is the KEEP gate; a
    unit that keeps a fraction 1 - 1/T per step has z = 1 - 1/T, i.e. a bias of logit(z) = log(T - 1).
    It goes on `bias_ih`'s z slice and `bias_hh`'s z slice is zeroed, so the sum is exactly that.
    Returns the T drawn. (Tallec & Ollivier's chrono initialisation, as the G1 / percep prototypes.)"""
    H = gru.hidden_size
    with torch.no_grad():
        T = torch.exp(torch.empty(H).uniform_(math.log(t_min), math.log(t_max)))
        for layer in range(gru.num_layers):
            getattr(gru, f"bias_ih_l{layer}")[H:2 * H] = torch.log(T - 1.0)
            getattr(gru, f"bias_hh_l{layer}")[H:2 * H] = 0.0
    return T


def log_std_init(speed_mode: str = "wp4") -> torch.Tensor:
    """(act_dim,) B6's initial log std: the first two curvature columns -2.07, the rest -1.8."""
    from ..mpc import plan_layout
    lay = plan_layout(speed_mode)
    out = torch.full((lay.act_dim,), LOG_STD_REST)
    out[lay.path.start:lay.path.start + 2] = LOG_STD_KAPPA12
    return out


class StemV2(nn.Module):
    """B1 + B2: [newest K frames | 9-tap min-pool of frame 0] -> ResNet trunk -> neck 48 -> fc -> 256.

    The trunk, neck and their padding are v1's resnet stem exactly (`learn.model.ScanStem`) -- zero
    padding on purpose, the 270-degree scan's two ends are not neighbours -- so a v1 stem can seed
    it (`init_stem_from_v1`). The min-pool is computed here, inside the graph, from the frame the
    car feeds; the channel order is frames first, the min-pool last."""

    def __init__(self, n_frames: int = 1, n_beams: int = 1081, out: int = 256):
        super().__init__()
        self.n_frames, self.n_beams = int(n_frames), int(n_beams)
        cin = self.n_frames + 1
        self.trunk = nn.Sequential(
            nn.Conv1d(cin, 48, 7, stride=2, padding=3), nn.GroupNorm(8, 48), nn.GELU(),
            ResBlock1d(48, 64, stride=2),
            ResBlock1d(64, 96, stride=2),
            ResBlock1d(96, 128, stride=2),
            ResBlock1d(128, 128, stride=2, dilation=2))                      # whole-scan context
        self.neck = nn.Sequential(nn.Conv1d(128, 48, 1), nn.GroupNorm(8, 48), nn.GELU())
        with torch.no_grad():
            self.cells = int(self.neck(self.trunk(torch.zeros(1, cin, self.n_beams))).shape[-1])
        self.neck_channels = 48
        self.fc = nn.Sequential(nn.Linear(self.neck_channels * self.cells, out), nn.GELU())

    def rows(self, scan: torch.Tensor) -> torch.Tensor:
        if scan.dim() != 3 or scan.shape[1] != self.n_frames or scan.shape[2] != self.n_beams:
            raise ValueError(f"scan must be (B, {self.n_frames}, {self.n_beams}), got {tuple(scan.shape)}")
        pad = MINPOOL_TAPS // 2
        near = -F.max_pool1d(-scan[:, :1], kernel_size=MINPOOL_TAPS, stride=1, padding=pad)
        return torch.cat([scan, near], 1)

    def forward_map(self, scan: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(features (B, 256), neck map (B, 48, cells)): the map is what `fc` reads, and what the
        per-beam auxiliary head reads, from one pass."""
        nk = self.neck(self.trunk(self.rows(scan)))
        return self.fc(nk.flatten(1)), nk

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        return self.fc(self.neck(self.trunk(self.rows(scan))).flatten(1))


def init_stem_from_v1(stem: StemV2, v1_stem: dict, v1_meta: dict) -> list:
    """Seed a v2 stem from a v1 resnet stem's state dict (keys relative to the stem). Returns the
    keys written.

    The G1 / perception-gate rule: the v1 first convolution reads [frames 0..n-1, angle ramp,
    min-pool]; v2 frames 0..K-2 keep their own weights, frame K-1 takes the SUM of frames K-1..n-1,
    the ramp is dropped and the min-pool weight moves to v2's last channel -- so on a static scene
    (every frame equal) the first layer computes what v1 computed with the ramp switched off, for
    any K. Trunk and neck are copied; `fc` keeps the neck's columns (they come first in v1's
    flatten; the global min/mean and sector columns it drops held 0.14 % of the fc variance)."""
    if str(v1_meta.get("scan_stem")) != "resnet" or str(v1_meta.get("temporal_encoder", "cnn")) != "cnn":
        raise ValueError("a v2 stem can only be seeded from a v1 resnet / cnn stem")
    if v1_meta.get("scan_deltas"):
        raise ValueError("a stem trained on scan deltas has no per-frame first-layer weights to sum")
    if (v1_meta.get("scan_channels") or {}).get("channels"):
        raise ValueError("the v1 stem reads extra scan channels; v2 has no place for them")
    n_old, k = int(v1_meta["n_stack"]), stem.n_frames
    if int(v1_meta["n_beams"]) != stem.n_beams:
        raise ValueError(f"v1 stem is for {v1_meta['n_beams']} beams, this one for {stem.n_beams}")
    if n_old < k:
        raise ValueError(f"the v1 stem reads {n_old} frames, fewer than this stem's {k}")
    own = stem.state_dict()
    W = v1_stem["trunk.0.weight"]
    if W.shape[1] != n_old + 2:
        raise ValueError(f"v1 first convolution has {W.shape[1]} input channels, expected {n_old} + 2")
    written = []
    with torch.no_grad():
        for key in own:
            if key == "trunk.0.weight":
                chans = [W[:, j] for j in range(k - 1)] + [W[:, k - 1:n_old].sum(1), W[:, n_old + 1]]
                own[key].copy_(torch.stack(chans, 1))
            elif key == "fc.0.weight":
                own[key].copy_(v1_stem[key][:, :own[key].shape[1]])
            else:
                if v1_stem[key].shape != own[key].shape:
                    raise ValueError(f"{key}: v1 {tuple(v1_stem[key].shape)} vs v2 {tuple(own[key].shape)}")
                own[key].copy_(v1_stem[key])
            written.append(key)
    stem.load_state_dict(own)
    return written


class ActorV2(nn.Module):
    """B1-B6. `step` / `forward` is the deployed function: (scan, ego, c, h) -> (action, h')."""

    def __init__(self, n_frames: int = 1, n_beams: int = 1081, ego_dim: int = EGO_DIM, act_dim: int = 8,
                 speed_mode: str = "wp4", hidden: int = 128, width: int = 256, ego_width: int = 128):
        super().__init__()
        self.act_dim, self.hidden_size, self.width = int(act_dim), int(hidden), int(width)
        self.stem = StemV2(n_frames, n_beams, width)                                  # B1 + B2
        self.ego = nn.Sequential(nn.Linear(ego_dim, ego_width), nn.GELU())              # B3
        self.fuse = nn.Sequential(nn.Linear(width + ego_width + 1, width), nn.GELU())    # B4
        self.ln = nn.LayerNorm(width)                                                   # B5
        self.gru = nn.GRU(width, self.hidden_size, batch_first=True)
        chrono_init(self.gru)
        self.head = nn.Sequential(nn.Linear(width + self.hidden_size, width), nn.GELU())  # B6
        self.mu = nn.Linear(width, self.act_dim)
        with torch.no_grad():
            self.mu.weight.mul_(0.1); self.mu.bias.zero_()
        init = log_std_init(speed_mode)
        if init.numel() != self.act_dim:
            raise ValueError(f"speed mode {speed_mode!r} has {init.numel()} action columns, not {act_dim}")
        self.log_std = nn.Parameter(init)
        #: train-time heads, attached by `ActorCriticV2` after both networks (`aux_v2`)
        self.b8 = None
        self.beam_car = None
        self.near_car = None

    @property
    def feature_dim(self) -> int:
        return self.width + self.hidden_size

    @property
    def has_memory(self) -> bool:
        """Always: B5 is part of the deployed function. Read by the inference paths that carry a
        hidden state for whatever actor they are handed (`learn.watch.MemoryActorRunner`)."""
        return True

    def _check_c(self, c, batch: int) -> torch.Tensor:
        """The dial is required: a v2 actor is never run on an implied number (v1's rule)."""
        if c is None:
            raise ValueError("the v2 actor reads the grip dial as an input column and has no default: "
                             "pass c = (dial - 1.0) / 0.25 (conditioning.mu_to_c), shape (B, 1)")
        if c.dim() != 2 or c.shape[0] != batch or c.shape[1] != 1:
            raise ValueError(f"c must be ({batch}, 1), got {tuple(c.shape)}")
        return c

    def initial_hidden(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        ref = self.mu.weight
        return torch.zeros(1, int(batch), self.hidden_size, device=device or ref.device, dtype=dtype or ref.dtype)

    # -------------------------------------------------------------- the per-step pieces
    def embed(self, scan, ego, c, beams: bool = False):
        """e (B, 256) = B4(stem(scan), B3(ego), c); with `beams`, also the neck map (B, 48, cells)."""
        c = self._check_c(c, ego.shape[0])
        if beams:
            s, nk = self.stem.forward_map(scan)
        else:
            s, nk = self.stem(scan), None
        e = self.fuse(torch.cat([s, self.ego(ego), c.to(s.dtype)], 1))
        return (e, nk) if beams else e

    def recur(self, e: torch.Tensor, h: Optional[torch.Tensor]):
        """One B5 step: (y (B, hidden), h' (1, B, hidden))."""
        if h is None:
            h = self.initial_hidden(e.shape[0], e.device, e.dtype)
        y, h_next = self.gru(self.ln(e)[:, None, :], h.contiguous())
        return y[:, 0], h_next

    def features(self, e: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """`[e | y]` -- what the action head and every auxiliary head read."""
        return torch.cat([e, y], -1)

    def action_mean(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mu(self.head(z)))

    # -------------------------------------------------------------- entry points
    def step(self, scan, ego, c, h=None):
        """(deterministic action (B, act_dim) in [-1, 1], next hidden (1, B, hidden))."""
        e = self.embed(scan, ego, c)
        y, h_next = self.recur(e, h)
        return self.action_mean(self.features(e, y)), h_next

    def forward(self, scan, ego, c, h=None):
        return self.step(scan, ego, c, h)

    def dist_of(self, mu: torch.Tensor) -> torch.distributions.Normal:
        mu = mu.float()
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def step_dist(self, scan, ego, c, h=None):
        mu, h_next = self.step(scan, ego, c, h)
        return self.dist_of(mu), h_next

    def step_all(self, scan, ego, c, h=None, beams: bool = False) -> dict:
        """Everything one step produces, for training: {"mu", "z", "h", and each attached head's
        output -- "b8" (B, 3, 4), "near_car" (B, 5), "beam_car" (B, N) when `beams`}."""
        if beams and self.beam_car is not None:
            e, nk = self.embed(scan, ego, c, beams=True)
        else:
            e, nk = self.embed(scan, ego, c), None
        y, h_next = self.recur(e, h)
        z = self.features(e, y)
        out = {"mu": self.action_mean(z), "z": z, "h": h_next}
        out.update(self.aux(z, nk))
        return out

    def aux(self, z: torch.Tensor, neck: Optional[torch.Tensor] = None) -> dict:
        """The attached heads' outputs on features `z` (and the neck map, for the per-beam head)."""
        out = {}
        if self.b8 is not None:
            out["b8"] = self.b8(z)
        if self.near_car is not None:
            out["near_car"] = self.near_car(z)
        if self.beam_car is not None and neck is not None:
            out["beam_car"] = self.beam_car(neck)
        return out

    def _walk(self, e: torch.Tensor, h: torch.Tensor, keep: Optional[torch.Tensor]):
        """B5 over a (T, B, W) block: (y (T, B, hidden), h_T). One cuDNN call when no episode ends
        inside the block; otherwise step by step, zeroing the state INTO a step whose `keep` is 0."""
        x = self.ln(e)
        if keep is None:
            y, h = self.gru(x.transpose(0, 1).contiguous(), h.contiguous())
            return y.transpose(0, 1), h
        ys = []
        for t in range(x.shape[0]):
            h = h * keep[t].to(h.dtype)[None, :, None]
            y_t, h = self.gru(x[t][:, None, :], h.contiguous())
            ys.append(y_t[:, 0])
        return torch.stack(ys, 0), h

    def sequence(self, scan, ego, c, h0=None, keep=None, burn_in: int = 0, beams: bool = False) -> dict:
        """A (T, B) block for truncated BPTT: the hook the training unit builds its losses on.

        `scan` (T, B, K, N), `ego` (T, B, E), `c` (T, B, 1); `keep` (T, B), 0 where the state must not
        be carried into that step (its episode began there). The first `burn_in` steps only warm the
        state -- no graph, no outputs -- which is the redesign's burn-in 40 + chunk 80. The stem runs
        once for the whole block. Returns {"z" (T-b, B, 384), "mu" (T-b, B, act_dim), "h" (final),
        each head's output with the same leading (T-b, B), "neck" when `beams`}."""
        T, B = scan.shape[0], scan.shape[1]
        flat = lambda t: t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])
        h = h0 if h0 is not None else self.initial_hidden(B, scan.device, scan.dtype)
        b = int(burn_in)
        if b:
            with torch.no_grad():
                eb = self.embed(flat(scan[:b]), flat(ego[:b]), flat(c[:b])).view(b, B, -1)
                _y, h = self._walk(eb, h, None if keep is None else keep[:b])
            h = h.detach()
        n = T - b
        want_map = beams and self.beam_car is not None
        if want_map:
            e, nk = self.embed(flat(scan[b:]), flat(ego[b:]), flat(c[b:]), beams=True)
        else:
            e, nk = self.embed(flat(scan[b:]), flat(ego[b:]), flat(c[b:])), None
        e = e.view(n, B, -1)
        y, h = self._walk(e, h, None if keep is None else keep[b:])
        z = self.features(e, y)
        out = {"z": z, "mu": self.action_mean(z), "h": h}
        heads = self.aux(z, None)
        if want_map:
            heads["beam_car"] = self.beam_car(nk).view(n, B, -1)
            out["neck"] = nk
        out.update(heads)
        return out


class CriticV2(nn.Module):
    """B9 as corrected: own 1-frame stem + [ego | c | privileged v2] -> MLP -> (value, contact logit).

    The scan it reads is frame 0 of whatever the actor is fed (a K-frame observation's newest frame),
    so the two can share one stored observation. The value is in the normalised units the training
    unit's value normaliser defines; this module only produces it."""

    def __init__(self, n_beams: int = 1081, ego_dim: int = EGO_DIM, priv_dim: int = 34, width: int = 256,
                 branch: int = 128):
        super().__init__()
        self.ego_dim, self.priv_dim = int(ego_dim), int(priv_dim)
        self.stem = StemV2(1, n_beams, width)
        self.inp = nn.Sequential(nn.Linear(self.ego_dim + 1 + self.priv_dim, branch), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(width + branch, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU())
        self.value = nn.Linear(width, 1)
        self.contact = nn.Linear(width, 1)
        with torch.no_grad():
            self.value.weight.mul_(0.1); self.value.bias.zero_()

    def forward(self, scan, ego, c, priv):
        """(value (B,), "car contact within 1 s" logit (B,))."""
        if priv.shape[-1] != self.priv_dim:
            raise ValueError(f"privileged vector is {priv.shape[-1]} wide, this critic reads {self.priv_dim} "
                             f"(F1VecEnv.privileged_v2) -- a v1 privileged() vector is not it")
        if c is None:
            raise ValueError("the critic reads the same dial column the actor does; pass c")
        z = self.mlp(torch.cat([self.stem(scan[:, :1]), self.inp(torch.cat([ego, c.to(ego.dtype), priv], 1))], 1))
        return self.value(z)[:, 0], self.contact(z)[:, 0]


class ActorCriticV2(nn.Module):
    """The v2 pair. Every constructor argument is recorded in `meta` and nothing else is, so
    `ActorCriticV2(**meta)` rebuilds it (`load_v2`)."""

    def __init__(self, arch: str = ARCH, n_beams: int = 1081, n_frames: int = 1, ego_dim: int = EGO_DIM,
                 act_dim: int = 8, speed_mode: str = "wp4", priv_dim: int = 34, hidden: int = 128,
                 cond_dim: int = 1, cond: Optional[dict] = None, obs: Optional[dict] = None,
                 aux: Optional[dict] = None):
        super().__init__()
        from .conditioning import CondSpec, spec_for
        if arch != ARCH:
            raise ValueError(f"ActorCriticV2 builds arch {ARCH!r}, got {arch!r}")
        cs = CondSpec.from_meta(cond) if cond else spec_for("dial")
        if cs.dim != 1 or int(cond_dim) != 1 or cs.source != "dial":
            raise ValueError(f"the v2 dial is one deployable input column (source 'dial'), got {cs}")
        # `from_dict`: a checkpoint written before `ego_actions` existed is read as the contract it was
        spec = ObsSpecV2.from_dict(obs).validate() if obs else ObsSpecV2(n_beams=n_beams, n_frames=n_frames,
                                                                        act_dim=act_dim, speed_mode=speed_mode)
        if (spec.n_beams, spec.n_frames, spec.ego_dim, spec.act_dim, spec.speed_mode) != (
                int(n_beams), int(n_frames), int(ego_dim), int(act_dim), str(speed_mode)):
            raise ValueError(f"the observation contract {spec} does not match the network "
                             f"(beams {n_beams}, frames {n_frames}, ego {ego_dim}, act {act_dim} {speed_mode})")
        ax = aux_spec(**(aux or {}))
        self.actor = ActorV2(n_frames, n_beams, ego_dim, act_dim, speed_mode, hidden)
        self.critic = CriticV2(n_beams, ego_dim, priv_dim)
        # The train-time heads LAST, in a fixed order: every weight above was drawn from the same
        # generator in the same order whatever the flags, so an arm with a head differs from one
        # without it by that head alone (v1's rule for every head it ever added).
        if ax["b8"]:
            self.actor.b8 = OpponentSlotHead(self.actor.feature_dim)
        if ax["beam_car"]:
            st = self.actor.stem
            self.actor.beam_car = BeamCarHead(st.neck_channels, st.cells, st.n_beams)
        if ax["near_car"]:
            self.actor.near_car = NearestCarHead(self.actor.feature_dim)
        self.meta = dict(arch=ARCH, n_beams=int(n_beams), n_frames=int(n_frames), ego_dim=int(ego_dim),
                         act_dim=int(act_dim), speed_mode=str(speed_mode), priv_dim=int(priv_dim),
                         hidden=int(hidden), cond_dim=1, cond=cs.to_meta(), obs=dict(vars(spec)), aux=ax)

    @property
    def obs_spec(self) -> ObsSpecV2:
        return ObsSpecV2.from_dict(self.meta["obs"])

    @property
    def has_memory(self) -> bool:
        return True

    def initial_hidden(self, batch: int, device=None, dtype=None) -> Hidden:
        return Hidden(self.actor.initial_hidden(batch, device, dtype), None)

    @staticmethod
    def _actor_h(h):
        if h is None:
            return None
        if isinstance(h, Hidden):
            return h.actor
        raise TypeError(f"hidden state must be a learn.memory.Hidden (or None), got {type(h).__name__}")

    @torch.no_grad()
    def act(self, scan, ego, deterministic: bool = False, c=None, h=None):
        """(action, log prob, next Hidden) -- `learn.memory.policy_fn`'s contract, so every
        inference path that drives a v1 checkpoint drives this one."""
        d, ha = self.actor.step_dist(scan, ego, c, self._actor_h(h))
        a = d.mean if deterministic else d.sample()
        return a.clamp(-1, 1), d.log_prob(a).sum(1), Hidden(ha, None)

    def evaluate(self, scan, ego, priv, actions, c=None, h=None):
        """One step: (log prob, entropy, value, distribution, contact logit, next Hidden)."""
        d, ha = self.actor.step_dist(scan, ego, c, self._actor_h(h))
        v, logit = self.critic(scan, ego, c, priv)
        return d.log_prob(actions).sum(1), d.entropy().sum(1), v, d, logit, Hidden(ha, None)


def block_parameter_counts(model: ActorCriticV2) -> dict:
    """Parameters per block of the redesign's table (section 4), the deployed actor's total, the
    critic's and the train-time heads'. `tests/test_model_v2.py` pins them."""
    n = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
    a = model.actor
    out = {"B1+B2 stem": n(a.stem), "B3 ego": n(a.ego), "B4 fusion": n(a.fuse), "B5 LayerNorm": n(a.ln),
           "B5 GRU": n(a.gru), "B6 head": n(a.head) + n(a.mu), "B6 log_std": a.log_std.numel()}
    out["actor (deployed)"] = sum(out.values())
    out["B8 slots"] = n(a.b8)
    out["beam car head"] = n(a.beam_car)
    out["nearest car head"] = n(a.near_car)
    out["B9 critic"] = n(model.critic)
    return out


def init_stems_from_v1(model: ActorCriticV2, path: str, device="cpu") -> dict:
    """Seed the actor's and the critic's stems from a v1 checkpoint's (redesign 6장 '재사용': the
    stem's trunk, neck and fc can start from safest; G1 and G4 both did, 'warm'). Returns
    {"actor": keys written, "critic": keys written}."""
    ck = torch.load(path, map_location=device, weights_only=True)
    meta = dict(ck["meta"])
    if meta.get("arch") == ARCH:
        raise ValueError(f"{os.path.basename(str(path))} is already a v2 checkpoint")
    sd = ck["state_dict"]
    sub = lambda pre: {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    # v1's critic stem is built from the same meta as its actor stem (`learn.model.Critic`)
    return {"actor": init_stem_from_v1(model.actor.stem, sub("actor.stem."), meta),
            "critic": init_stem_from_v1(model.critic.stem, sub("critic.stem."), meta)}


def load_v2(ck: dict, path, device="cpu", override: Optional[dict] = None,
            strict_names: bool = False) -> Tuple[ActorCriticV2, dict]:
    """`learn.model.load_checkpoint` for a v2 checkpoint (it dispatches here on `meta["arch"]`).

    The same tolerance as v1's loader: a tensor whose shape no longer matches (an `override` that
    adds a head, say) stays at its fresh initialisation and is listed in `extra["skipped"]`;
    `strict_names` refuses that instead."""
    meta = dict(ck["meta"]); meta.update(override or {})
    import inspect
    known = set(inspect.signature(ActorCriticV2.__init__).parameters) - {"self"}
    unknown = sorted(k for k in meta if k not in known)
    if unknown:
        raise ValueError(f"{os.path.basename(str(path))} was saved by a newer or different branch: "
                         f"its v2 meta has field(s) this tree does not know {unknown}")
    m = ActorCriticV2(**meta).to(device)
    sd = m.state_dict(); skipped = []
    for k_, v in ck["state_dict"].items():
        if k_ in sd and sd[k_].shape == v.shape:
            sd[k_] = v
        else:
            skipped.append(k_)
    if strict_names:
        fresh = [k_ for k_ in sd if k_ not in ck["state_dict"]]
        if skipped or fresh:
            raise ValueError(f"strict v2 load failed: {len(skipped)} unused {sorted(skipped)[:6]}, "
                             f"{len(fresh)} fresh {sorted(fresh)[:6]}")
    m.load_state_dict(sd)
    extra = dict(ck.get("extra", {})); extra["skipped"] = skipped
    return m, extra
