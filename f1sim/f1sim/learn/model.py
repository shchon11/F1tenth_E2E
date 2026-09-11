"""Actor-critic for the LiDAR-only policy. Actor sees (scan stack, proprio); the critic additionally
sees the privileged vector (asymmetric actor-critic). Actor is TensorRT-friendly (conv1d + MLP)."""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 temporal_encoder: str = "cnn", scan_stem: str = "plain"):
        super().__init__()
        if temporal_encoder not in ("cnn", "gru") or (temporal_encoder == "gru" and scan_deltas):
            raise ValueError("temporal_encoder must be cnn or gru; scan deltas apply only to cnn")
        if scan_stem not in ("plain", "resnet"):
            raise ValueError("scan_stem must be plain or resnet")
        self.scan_deltas, self.temporal_encoder, self.scan_stem = scan_deltas, temporal_encoder, scan_stem
        self.register_buffer("beam_angle", torch.linspace(-1.0, 1.0, n_beams)[None, None, :], persistent=False)
        if scan_stem == "resnet":
            per_frame = temporal_encoder == "gru"
            cin = (1 if per_frame else (2 * n_stack - 1 if scan_deltas else n_stack)) + 2   # + angle + min-pool
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
        channels = 2 * n_stack - 1 if scan_deltas else n_stack
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(64, 32, 3, stride=2, padding=1), nn.GELU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, channels, n_beams)).numel()
        self.fc = nn.Sequential(nn.Linear(n, out), nn.GELU())

    def scan_features(self, scan: torch.Tensor) -> torch.Tensor:
        if not self.scan_deltas:
            return scan
        return torch.cat([scan, scan[:, :-1] - scan[:, 1:]], 1)

    SECTORS = 36                                          # 270 deg / 36 = 7.5 deg per sector

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Append the beam-angle ramp and the windowed nearest return to the channel axis."""
        near = -F.max_pool1d(-x[:, :1], kernel_size=9, stride=1, padding=4)
        return torch.cat([x, self.beam_angle.expand(x.shape[0], -1, -1), near], 1)

    def _sector_profile(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest return per angular sector of the newest scan, straight from the raw input.

        GroupNorm standardises each group over the whole beam axis, so after the trunk a corridor
        1 m wide and one 3 m wide look alike -- exactly the absolute scale a speed decision needs.
        This bypass carries it to the head un-normalised, and doubles as the classical gap-follower
        descriptor (min range per sector) that the conv stack would otherwise have to rediscover."""
        return -F.adaptive_max_pool1d(-x[:, :1], self.SECTORS).flatten(1)

    def _resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(self._augment(x))
        pooled = torch.cat([-F.adaptive_max_pool1d(-h, 1).flatten(1), h.mean(2)], 1)
        return torch.cat([self.neck(h).flatten(1), pooled, self._sector_profile(x)], 1)

    def forward(self, scan):
        if self.scan_stem == "resnet":
            if self.temporal_encoder == "gru":
                batch, steps, beams = scan.shape
                f = self._resnet_features(scan.reshape(batch * steps, 1, beams))
                f = self.frame_fc(f).reshape(batch, steps, -1).flip(1)     # oldest -> newest
                return self.fc(self.gru(f)[1][-1])
            return self.fc(self._resnet_features(self.scan_features(scan)))
        if self.temporal_encoder == "gru":
            batch, steps, beams = scan.shape
            frame_features = self.conv(scan.reshape(batch * steps, 1, beams)).flatten(1)
            frame_features = self.frame_fc(frame_features).reshape(batch, steps, -1).flip(1)
            _, hidden = self.gru(frame_features)
            return self.fc(hidden[-1])
        return self.fc(self.conv(self.scan_features(scan)).flatten(1))


class Actor(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, hidden: int = 256, log_std_init: float = -0.7,
                 act_dim: int = 2, scan_deltas: bool = False, temporal_encoder: str = "cnn",
                 scan_stem: str = "plain", cond_dim: int = 0):
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
                             scan_stem=scan_stem)
        pw = 64 if proprio_dim <= 32 else 128                     # a proprio history (hundreds of inputs) gets a wider embedding
        self.pro = nn.Sequential(nn.Linear(proprio_dim, pw), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(256 + pw, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.cond = None
        if self.cond_dim > 0:
            self.cond = nn.Linear(self.cond_dim, hidden, bias=False)
            nn.init.zeros_(self.cond.weight)
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

    def _parts(self, scan, proprio, c=None):
        c = self._require_cond(c, proprio.shape[0])
        p = self.pro(proprio)
        x = torch.cat([self.stem(scan), p], 1)
        if self.cond is None:
            return self.mlp(x), p                    # legacy path, byte-for-byte what it always was
        # Conditional: the same three layers, with the projection added to the first preactivation.
        # Written out rather than sliced, because `self.mlp[1:]` builds a new Sequential on every
        # forward and this runs once per env step.
        lin0, act0, lin1, act1 = self.mlp[0], self.mlp[1], self.mlp[2], self.mlp[3]
        pre = lin0(x) + self.cond(c.to(x.dtype))
        return act1(lin1(act0(pre))), p

    def features(self, scan, proprio, c=None):
        return self._parts(scan, proprio, c)[0]

    def forward(self, scan, proprio, c=None):
        return torch.tanh(self.mu(self.features(scan, proprio, c)))

    def forward_all(self, scan, proprio, c=None):
        """(action mean, grip prediction, opponent-motion prediction) from one pass through the trunk."""
        h, p = self._parts(scan, proprio, c)
        return torch.tanh(self.mu(h)), self.grip(torch.cat([h, p], 1))[:, 0], self.opp(h)

    def dist(self, scan, proprio, c=None):
        mu = self(scan, proprio, c).float()
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))


class Critic(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, priv_dim: int, hidden: int = 256,
                 scan_deltas: bool = False, temporal_encoder: str = "cnn", scan_stem: str = "plain",
                 priv_adapter: Optional[str] = None):
        super().__init__()
        self.stem = ScanStem(n_stack, n_beams, scan_deltas=scan_deltas, temporal_encoder=temporal_encoder,
                             scan_stem=scan_stem)
        self.pro = nn.Sequential(nn.Linear(proprio_dim + priv_dim, 128), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(256 + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
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

    def forward(self, scan, proprio, priv):
        if self._adapt is not None:
            priv = self._adapt(priv)
        return self.mlp(torch.cat([self.stem(scan), self.pro(torch.cat([proprio, priv], 1))], 1)).squeeze(1)


class ActorCritic(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, priv_dim: int, act_dim: int = 2,
                 scan_deltas: bool = False, temporal_encoder: str = "cnn", scan_stem: str = "plain",
                 cond_dim: int = 0, cond: Optional[dict] = None, priv_adapter: Optional[str] = None):
        super().__init__()
        self.actor = Actor(n_stack, n_beams, proprio_dim, act_dim=act_dim, scan_deltas=scan_deltas,
                           temporal_encoder=temporal_encoder, scan_stem=scan_stem, cond_dim=cond_dim)
        self.critic = Critic(n_stack, n_beams, proprio_dim, priv_dim, scan_deltas=scan_deltas,
                             temporal_encoder=temporal_encoder, scan_stem=scan_stem,
                             priv_adapter=priv_adapter)
        self.meta = dict(n_stack=n_stack, n_beams=n_beams, proprio_dim=proprio_dim, priv_dim=priv_dim,
                         act_dim=act_dim, scan_deltas=scan_deltas, temporal_encoder=temporal_encoder,
                         scan_stem=scan_stem)
        # Recorded only when used, so a legacy checkpoint's meta is byte-for-byte what it was.
        if cond_dim:
            self.meta["cond_dim"] = int(cond_dim)
            self.meta["cond"] = dict(cond or {})
        if priv_adapter:
            self.meta["priv_adapter"] = priv_adapter

    @torch.no_grad()
    def act(self, scan, proprio, deterministic=False, c=None):
        d = self.actor.dist(scan, proprio, c)
        a = d.mean if deterministic else d.sample()
        return a.clamp(-1, 1), d.log_prob(a).sum(1)

    def evaluate(self, scan, proprio, priv, actions, c=None):
        d = self.actor.dist(scan, proprio, c)
        return d.log_prob(actions).sum(1), d.entropy().sum(1), self.critic(scan, proprio, priv), d

    def evaluate_aux(self, scan, proprio, priv, actions, c=None):
        """evaluate() plus the grip prediction, from the same trunk pass."""
        mu, grip, opp = self.actor.forward_all(scan, proprio, c)
        mu = mu.float()
        d = torch.distributions.Normal(mu, self.actor.log_std.exp().expand_as(mu))
        return d.log_prob(actions).sum(1), d.entropy().sum(1), self.critic(scan, proprio, priv), d, grip.float(), opp.float()


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
