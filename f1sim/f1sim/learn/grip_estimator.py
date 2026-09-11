"""Causal onboard friction estimator: sensor history, ordered quantile net, calibrated lower bound.

Everything here runs from what the car can actually produce. The inputs are the normalized speed,
the IMU, the VESC attitude estimate and the **previously issued 2D command** -- never the simulator
state, never `P["mu"]`, never the map, never the speed cap, and never the actor's 8D plan. The
observation accessor reads three keys and ignores the rest of the dict, so poisoning the other keys
cannot change a prediction; `test_grip_estimator.py` proves that rather than asserting it.

Timing is the part that is easy to get wrong and impossible to see once it is wrong. `history_t`
holds observation *t* and the command issued at *t-1*, and predicts `mu_t` **before** action *t* is
taken. A window never contains command *t* or any later observation. On termination or truncation
the env's history is cleared and its previous command is zeroed, because the observation that comes
back from `env.step` already belongs to the next episode -- attaching the new episode's `mu` to the
old rows would relabel data that was collected under a different friction.

The consumption rule (plan B2) lives here rather than in the runtime so that the closed-loop
controller and the offline evaluation cannot drift apart: losses of grip apply immediately, gains
are smoothed over tau = 0.25 s, and anything cold or non-finite falls back to `mu_min`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------- feature contract
FEATURE_NAMES = ("speed", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z",
                 "roll", "pitch", "prev_steer", "prev_speed")
#: What each column was divided by. Mirrors the env's own observation scaling exactly -- these are
#: not new normalizations, they are the ones `gym_env._obs` already applied, plus s_max/v_max for
#: the issued command, which `push` applies because the tracker emits it in physical units.
FEATURE_SCALES = (10.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 0.35, 0.35, 0.4189, 10.0)

N_FRAMES = 40                 # 1.0 s at 40 Hz, newest first
N_FEATURES = 11
CONTROL_DT = 0.025
MU_MIN, MU_MAX = 0.73423, 1.15379
COLD_FALLBACK_MU = MU_MIN
WARM_FRAMES = 40              # B2: cold until every row is valid
FILTER_TAU = 0.25             # B2: gains smoothed over this; losses immediate

#: BF1 excitation gate, exactly as predeclared in the plan: at least `EXCITATION_MIN_FRAMES` valid
#: frames in which |accel_x| >= 2.5 m/s^2 OR |accel_y| >= 3.0 m/s^2. Both channels, because
#: cornering identifies friction as well as braking does; the plan's gate is the OR of the two.
EXCITATION_MIN_FRAMES = 5
EXCITATION_AX_MS2 = 2.5
EXCITATION_AY_MS2 = 3.0

OBS_KEYS = ("speed", "imu", "imu_att")      # the only keys ever read from an observation dict


@dataclass(frozen=True)
class FeatureSpec:
    n_frames: int = N_FRAMES
    n_features: int = N_FEATURES
    control_dt: float = CONTROL_DT
    names: Tuple[str, ...] = FEATURE_NAMES
    scales: Tuple[float, ...] = FEATURE_SCALES
    mu_min: float = MU_MIN
    mu_max: float = MU_MAX
    warm_frames: int = WARM_FRAMES
    filter_tau: float = FILTER_TAU

    def to_meta(self) -> dict:
        return asdict(self)

    def validate(self) -> "FeatureSpec":
        """Reject a spec this build cannot honour, rather than letting dimensions drift silently.

        Every one of these has a way of going wrong quietly: a scale list that no longer matches the
        column order, a `warm_frames` above `n_frames` (never warm, permanent fallback, and the arm
        silently becomes the control arm), a non-positive scale that inverts a channel.
        """
        if self.n_frames < 1 or self.n_features < 1:
            raise ValueError(f"n_frames/n_features must be >= 1, got {self.n_frames}/{self.n_features}")
        if len(self.names) != self.n_features or len(self.scales) != self.n_features:
            raise ValueError(f"names ({len(self.names)}) and scales ({len(self.scales)}) must both "
                             f"have n_features = {self.n_features} entries")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate feature names: {self.names}")
        for n_, sc in zip(self.names, self.scales):
            if not math.isfinite(sc) or sc <= 0.0:
                raise ValueError(f"scale for {n_!r} must be finite and positive, got {sc}")
        if not (math.isfinite(self.mu_min) and math.isfinite(self.mu_max)) or self.mu_min >= self.mu_max:
            raise ValueError(f"need finite mu_min < mu_max, got {self.mu_min}/{self.mu_max}")
        if not 1 <= self.warm_frames <= self.n_frames:
            raise ValueError(f"warm_frames must be in [1, {self.n_frames}], got {self.warm_frames}")
        if not math.isfinite(self.control_dt) or self.control_dt <= 0.0:
            raise ValueError(f"control_dt must be finite and positive, got {self.control_dt}")
        if not math.isfinite(self.filter_tau) or self.filter_tau <= 0.0:
            raise ValueError(f"filter_tau must be finite and positive, got {self.filter_tau}")
        return self

    @staticmethod
    def from_meta(m: dict) -> "FeatureSpec":
        if not isinstance(m, dict):
            raise ValueError(f"feature spec must be a dict, got {type(m).__name__}")
        unknown = set(m) - set(FeatureSpec.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported feature-spec keys {sorted(unknown)}")
        missing = set(FeatureSpec.__dataclass_fields__) - set(m)
        if missing:
            raise ValueError(f"feature spec is missing {sorted(missing)}; a partial spec would take "
                             f"this build's defaults for fields the checkpoint was not trained with")
        d = dict(m)
        for k in ("names", "scales"):
            d[k] = tuple(d[k])
        return FeatureSpec(**d).validate()


#: The three keys, with the exact widths the contract fixes. Nothing is reshaped or sliced into
#: place: a tensor of the wrong width is a wiring bug, and quietly taking its first columns would
#: produce a history whose column meanings no longer match `FEATURE_NAMES`.
REQUIRED_OBS = (("speed", 1), ("imu", 6), ("imu_att", 2))


def observation_row(obs: Dict[str, Tensor], batch: int, device) -> Tensor:
    """(B,9) sensor half of a row, from the three permitted keys only. Strict.

    An earlier version substituted zeros for a missing key and then marked the row **valid**. That
    is the worst available answer: zero is a perfectly ordinary normalized reading, so an env with
    no IMU would have produced forty valid rows of "stationary, level, no rotation" and trained the
    model on them. A missing or malformed sensor is a wiring failure and is raised here; a caller
    that genuinely has no reading should not push a row at all.
    """
    if not isinstance(obs, dict):
        raise ValueError(f"observation must be a dict, got {type(obs).__name__}")
    parts = []
    for key, width in REQUIRED_OBS:
        v = obs.get(key)
        if v is None:
            raise ValueError(f"observation is missing required sensor {key!r}; expected "
                             f"({batch}, {width}). Sensors are not optional and are never zero-filled.")
        if not torch.is_tensor(v):
            raise ValueError(f"observation[{key!r}] must be a tensor, got {type(v).__name__}")
        if v.dim() != 2 or v.shape[0] != batch or v.shape[1] != width:
            raise ValueError(f"observation[{key!r}] must be exactly ({batch}, {width}), got "
                             f"{tuple(v.shape)}; this row is not reshaped or sliced into place")
        parts.append(v.to(device=device, dtype=torch.float32))
    return torch.cat(parts, 1)


class SensorHistory:
    """Newest-first ring of `(B, n_frames, 11)` rows plus a validity mask.

    `inputs()` returns views into the buffers, which advance on the next `push`. A consumer that
    persists them -- the dataset writer does -- has to clone.
    """

    def __init__(self, batch: int, device, spec: FeatureSpec = FeatureSpec()):
        self.B, self.device, self.spec = int(batch), torch.device(device), spec
        self.features = torch.zeros(self.B, spec.n_frames, spec.n_features, device=self.device)
        self.valid = torch.zeros(self.B, spec.n_frames, dtype=torch.bool, device=self.device)
        self._steer_scale = float(spec.scales[9])
        self._speed_scale = float(spec.scales[10])

    def _row(self, obs, cmd_norm: Tensor) -> Tensor:
        return torch.cat([observation_row(obs, self.B, self.device), cmd_norm], 1)

    def _normalize_cmd(self, issued: Optional[Tensor]) -> Tensor:
        if issued is None:
            return torch.zeros(self.B, 2, device=self.device)
        c = issued.to(device=self.device, dtype=torch.float32).reshape(self.B, 2)
        return torch.stack([c[:, 0] / self._steer_scale, c[:, 1] / self._speed_scale], 1)

    @torch.no_grad()
    def reset(self, ids: Tensor, current_obs: Dict[str, Tensor]) -> None:
        """Clear these envs and seed them with the current observation and a zero command.

        One valid row, not zero: the observation exists, only the command history does not.
        """
        if ids is None or (torch.is_tensor(ids) and ids.numel() == 0):
            return
        ids = ids.to(self.device).reshape(-1)
        self.features[ids] = 0.0
        self.valid[ids] = False
        row = self._row(current_obs, torch.zeros(self.B, 2, device=self.device))
        self.features[ids, 0] = row[ids]
        self.valid[ids, 0] = True

    @torch.no_grad()
    def push(self, current_obs: Dict[str, Tensor], previous_issued_cmd: Optional[Tensor],
             reset_mask: Optional[Tensor] = None) -> None:
        """Insert one row for every env: observation *t* with the command issued at *t-1*.

        `reset_mask` names envs whose previous episode has ended. Their command is zeroed **and
        their whole history is cleared** before the new row is inserted, so neither a command nor an
        old-friction observation ever crosses an episode boundary. Such an env leaves this call with
        `valid_count == 1`.
        """
        cmd = self._normalize_cmd(previous_issued_cmd)
        if reset_mask is not None:
            m = reset_mask.to(self.device).reshape(-1).bool()
            cmd = torch.where(m[:, None], torch.zeros_like(cmd), cmd)
            # Clearing is the point, not just zeroing the command. Zeroing alone left the previous
            # episode's forty rows in place: the env stayed warm and its old-friction window got
            # labelled with the new episode's mu on the very next step. Clear, then insert, so a
            # reset env comes out of this call with exactly one valid row.
            self.features[m] = 0.0
            self.valid[m] = False
        row = self._row(current_obs, cmd)
        self.features = torch.roll(self.features, 1, dims=1)
        self.valid = torch.roll(self.valid, 1, dims=1)
        self.features[:, 0] = row
        self.valid[:, 0] = True

    def inputs(self) -> Tuple[Tensor, Tensor]:
        return self.features, self.valid

    def valid_count(self) -> Tensor:
        return self.valid.sum(1)

    def is_warm(self) -> Tensor:
        return self.valid_count() >= self.spec.warm_frames


def excitation_frames(features: Tensor, valid: Tensor) -> Dict[str, Tensor]:
    """Per-window counts behind the gate: longitudinal, lateral, and their union.

    Columns 4 and 5 are accel x and y normalized by 10, so multiplying by the scale recovers m/s^2.
    The union is the gate; the two parts are reported separately so a dataset that is excited only
    in cornering is visible as such rather than hidden inside a single pass rate.
    """
    ax = (features[..., 4] * FEATURE_SCALES[4]).abs() >= EXCITATION_AX_MS2
    ay = (features[..., 5] * FEATURE_SCALES[5]).abs() >= EXCITATION_AY_MS2
    return {"long": (ax & valid).sum(1), "lat": (ay & valid).sum(1),
            "union": ((ax | ay) & valid).sum(1)}


def excitation_pass(features: Tensor, valid: Tensor,
                    min_frames: int = EXCITATION_MIN_FRAMES) -> Tensor:
    """BF1 gate as predeclared: >= 5 valid frames with |ax| >= 2.5 OR |ay| >= 3.0 m/s^2."""
    return excitation_frames(features, valid)["union"] >= min_frames


# ---------------------------------------------------------------- model
class QuantileGripNet(nn.Module):
    """`(B,40,11)` + `(B,40)` -> ordered `q10 <= q50 <= q90` in physical mu.

    Ordering is structural, not a penalty: the head emits a base and two softplus increments, so no
    weight assignment can produce crossed quantiles. A temporal convolution rather than a flatten
    because the informative part of a window is a short burst -- a brake application, a corner --
    that can sit anywhere in the second, and a positional MLP would have to learn that invariance
    from a few hundred episodes.
    """

    def __init__(self, spec: FeatureSpec = FeatureSpec(), width: int = 32, hidden: int = 64):
        super().__init__()
        self.spec = spec.validate()
        #: Saved with the weights. A checkpoint trained at another width used to reload into the
        #: default one, where it either raised deep inside `load_state_dict` or, worse, matched.
        self.arch = {"width": int(width), "hidden": int(hidden)}
        c = spec.n_features
        self.conv = nn.Sequential(
            nn.Conv1d(c, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(2 * width + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():                       # start at the middle of the range, tight spread
            mid = 0.5 * (spec.mu_min + spec.mu_max)
            self.head[-1].bias.copy_(torch.tensor([mid - 0.05, math.log(math.e ** 0.05 - 1 + 1e-9),
                                                   math.log(math.e ** 0.05 - 1 + 1e-9)]))

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        m = valid.to(features.dtype)[..., None]
        x = (features * m).transpose(1, 2)                       # (B, C, T), invalid rows zeroed
        h = self.conv(x)
        mt = valid.to(h.dtype)[:, None, :]
        n = mt.sum(-1).clamp_min(1.0)
        mean = (h * mt).sum(-1) / n
        big = torch.finfo(h.dtype).min
        mx = torch.where(mt.bool(), h, torch.full_like(h, big)).max(-1).values
        mx = torch.where(valid.any(1)[:, None], mx, torch.zeros_like(mx))
        frac = (valid.to(h.dtype).sum(1, keepdim=True) / valid.shape[1])
        a = self.head(torch.cat([mean, mx, frac], 1))
        q10 = a[:, 0]
        q50 = q10 + nn.functional.softplus(a[:, 1])
        q90 = q50 + nn.functional.softplus(a[:, 2])
        return torch.stack([q10, q50, q90], 1)


def pinball_loss(pred: Tensor, target: Tensor, taus=(0.1, 0.5, 0.9)) -> Tensor:
    t = target.reshape(-1, 1)
    q = torch.tensor(taus, device=pred.device, dtype=pred.dtype)[None, :]
    d = t - pred
    return torch.maximum(q * d, (q - 1.0) * d).mean()


# ---------------------------------------------------------------- frozen inference
class GripEstimator:
    """A frozen net plus the B2 consumption rule, so runtime and offline eval cannot diverge."""

    def __init__(self, net: QuantileGripNet, spec: FeatureSpec, calibration: dict, meta: dict):
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        spec.validate()
        cal = dict(calibration)
        delta = float(cal.get("cal_delta", 0.0))
        if not math.isfinite(delta):
            raise ValueError(f"cal_delta must be finite, got {delta}")
        if delta < 0.0:
            raise ValueError(f"cal_delta must be >= 0: calibration is a downward correction on q10, "
                             f"and a negative value would raise the assumed friction. Got {delta}")
        self.spec, self.calibration, self.meta = spec, cal, dict(meta)
        # names the runtime lane asked for, so a policy checkpoint can refuse a mismatched pair
        self.feature_spec = spec.to_meta()                 # plain dict, serialisable
        self.mu_range = (spec.mu_min, spec.mu_max)
        #: The width/hidden record, surfaced so a consumer embedding this model does not have to
        #: reach into `.net`. It is NOT under `meta`: it describes the weights, not the run.
        self.arch = dict(net.arch)
        #: Set by `load_grip_estimator` from the file's own bytes. Empty for an in-memory model,
        #: never a value copied out of `meta` -- a hash a file asserts about itself is not evidence.
        self.sha = ""
        self.cal_delta = float(self.calibration.get("cal_delta", 0.0))
        self.alpha = 1.0 - math.exp(-spec.control_dt / spec.filter_tau)
        self._filtered: Optional[Tensor] = None

    def _ensure(self, batch: int, device, dtype):
        if self._filtered is None or self._filtered.numel() != batch:
            self._filtered = torch.full((batch,), self.spec.mu_min, device=device, dtype=dtype)

    def reset_filter(self, ids: Optional[Tensor] = None) -> None:
        if self._filtered is None:
            return
        if ids is None:
            self._filtered.fill_(self.spec.mu_min)
        elif ids.numel():
            self._filtered[ids.to(self._filtered.device).reshape(-1)] = self.spec.mu_min

    @torch.no_grad()
    def quantiles(self, features: Tensor, valid: Tensor) -> Tensor:
        return self.net(features, valid)

    # ---- STATELESS. Safe on shuffled batches, offline validation and calibration.
    @torch.no_grad()
    def calibrated_candidate(self, features: Tensor, valid: Tensor) -> Tuple[Tensor, dict]:
        """`(candidate (B,), diagnostics)` with cold/invalid fallback but **no temporal filter**.

        This is the call for anything whose rows are not one env's ordered stream: shuffled
        validation minibatches, calibration over cal episodes, offline metrics. Running those
        through `lower_mu` would thread one EMA across unrelated episodes and report a number that
        no deployment ever produces.
        """
        q = self.quantiles(features, valid)
        warm = valid.sum(1) >= self.spec.warm_frames
        finite = torch.isfinite(q).all(1)
        ok = warm & finite
        cand = (q[:, 0] - self.cal_delta).clamp(self.spec.mu_min, self.spec.mu_max)
        cand = torch.where(ok, cand, torch.full_like(cand, self.spec.mu_min))
        reason = torch.where(~warm, torch.zeros_like(cand, dtype=torch.int8),
                             torch.where(~finite, torch.ones_like(cand, dtype=torch.int8),
                                         torch.full_like(cand, 2, dtype=torch.int8)))
        ex = excitation_frames(features, valid)
        diag = {"q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2], "candidate": cand,
                "warm": warm, "finite": finite,
                "fallback_reason": reason,                 # 0 cold, 1 non-finite, 2 none
                "q_gap": q[:, 1] - q[:, 0],
                "excitation_pass": ex["union"] >= EXCITATION_MIN_FRAMES,
                "excitation_long": ex["long"], "excitation_lat": ex["lat"]}
        return cand, diag

    # ---- STATEFUL. Ordered live streams only: one row per env per control step, in time order.
    @torch.no_grad()
    def lower_mu(self, features: Tensor, valid: Tensor) -> Tuple[Tensor, dict]:
        """`(used_mu (B,), diagnostics)` -- the calibrated candidate through the B2 filter.

        **Carries per-env state.** Row `b` must be the same environment on every consecutive call,
        in time order, and `reset_filter(ids)` must be called on termination/truncation. Do not use
        it on shuffled data; use `calibrated_candidate` there.

        Backend owns this filter (PM assignment); the runtime must not apply a second one.
        `alpha = 1 - exp(-control_dt / tau)`, losses immediate, gains smoothed over tau = 0.25 s.
        """
        cand, diag = self.calibrated_candidate(features, valid)
        B = features.shape[0]
        self._ensure(B, cand.device, cand.dtype)
        ok = diag["warm"] & diag["finite"]
        ema = (1.0 - self.alpha) * self._filtered + self.alpha * cand
        filtered = torch.where(ok, torch.minimum(cand, ema),
                               torch.full_like(cand, self.spec.mu_min))
        self._filtered = filtered
        diag["used_mu"] = filtered
        return filtered, diag


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_grip_estimator(path, net: QuantileGripNet, spec: FeatureSpec, calibration: dict,
                        meta: dict) -> str:
    """Write the checkpoint and return the sha256 of the bytes actually written.

    `meta` may not carry `checkpoint_sha256`: a file cannot hash itself, so any value written there
    would be a placeholder that later reads would trust. The hash is returned instead, and readers
    recompute it.
    """
    meta = dict(meta)
    if "checkpoint_sha256" in meta:
        raise ValueError("meta must not carry 'checkpoint_sha256'; it is computed from the file "
                         "at save and recomputed at load, never asserted")
    torch.save({"state_dict": net.state_dict(), "feature_spec": spec.validate().to_meta(),
                "arch": dict(net.arch), "calibration": dict(calibration), "meta": meta,
                "format": "grip_estimator_v1"}, path)
    return file_sha256(path)


def load_grip_estimator(path, device="cpu") -> GripEstimator:
    ck = torch.load(path, map_location=device)
    if ck.get("format") != "grip_estimator_v1":
        raise ValueError(f"not a grip_estimator_v1 checkpoint: {ck.get('format')!r}")
    spec = FeatureSpec.from_meta(ck["feature_spec"])
    arch = ck.get("arch")
    if not isinstance(arch, dict) or not {"width", "hidden"} <= set(arch):
        raise ValueError("checkpoint has no architecture record; it predates the width/hidden fix "
                         "and would silently load into this build's defaults")
    net = QuantileGripNet(spec, width=int(arch["width"]), hidden=int(arch["hidden"])).to(device)
    net.load_state_dict(ck["state_dict"])                 # strict: any mismatch raises
    est = GripEstimator(net, spec, ck["calibration"], ck["meta"])
    est.sha = file_sha256(path)                           # the file's own bytes, not a claim
    return est
