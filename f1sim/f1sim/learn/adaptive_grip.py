"""Evidence-gated friction inference from causal onboard sensor/command history.

Wheel odometry is NOT ground-truth body velocity. Derived channels are command-response
residuals, not a measured slip ratio. The learned confidence distinguishes informative tire
saturation from ordinary elastic slip; no low-excitation friction regression is imposed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..params import VehicleParams
from .grip_estimator import (FEATURE_NAMES, FEATURE_SCALES, FeatureSpec, excitation_frames,
                             file_sha256)

FORMAT = "adaptive_grip_v2"


def adaptive_feature_spec() -> FeatureSpec:
    return FeatureSpec(mu_min=0.35, mu_max=1.40, warm_frames=8)


@dataclass(frozen=True)
class AdaptiveConsumption:
    nominal_mu: float = VehicleParams().mu
    confidence_threshold: float = 0.65
    fault_mu: float = 0.35
    physical_floor: float = 0.05

    def validate(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError("consumption parameters must be finite")
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must lie in (0, 1)")
        if not 0 < self.physical_floor <= self.fault_mu <= self.nominal_mu:
            raise ValueError("need 0 < physical_floor <= fault_mu <= nominal_mu")
        return self


class AdaptiveGripNet(nn.Module):
    """Small GRU with structural quantile ordering and a saturation-evidence head.

    Public inputs retain the canonical newest-first, 11-channel feature contract. Valid
    rows are compacted into chronological order; padding never becomes sensor evidence.
    """

    def __init__(self, spec: FeatureSpec = None, hidden: int = 64,
                 wheelbase: float = VehicleParams().lf + VehicleParams().lr):
        super().__init__()
        self.spec = (spec or adaptive_feature_spec()).validate()
        if self.spec.names != FEATURE_NAMES or self.spec.scales != FEATURE_SCALES:
            raise ValueError("adaptive model requires the canonical onboard feature contract")
        if hidden < 4 or not math.isfinite(wheelbase) or wheelbase <= 0:
            raise ValueError("invalid adaptive architecture")
        self.arch = {"hidden": int(hidden), "wheelbase": float(wheelbase)}
        self.gru = nn.GRU(self.spec.n_features + 8, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 4))
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([0.3, -1.7, -1.7, -1.0]))

    def derived_features(self, features: Tensor, valid: Tensor) -> Tensor:
        """Causal finite differences and residuals; never an independent body-speed input."""
        x = torch.where(valid[..., None], features, torch.zeros_like(features)).flip(1)
        mask = valid.flip(1)
        previous = torch.cat([x[:, :1], x[:, :-1]], 1)
        paired = mask & torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], 1)
        d = (x - previous) / self.spec.control_dt
        d = torch.where(paired[..., None], d, torch.zeros_like(d))
        wheel_accel = d[..., 0] * FEATURE_SCALES[0]
        lateral_proxy = ((x[..., 0] * FEATURE_SCALES[0]).square()
                         * torch.tan(x[..., 9] * FEATURE_SCALES[9]) / self.arch["wheelbase"])
        derived = torch.stack([
            wheel_accel / 20.0,
            d[..., 3] * FEATURE_SCALES[3] / 20.0,
            d[..., 9] * FEATURE_SCALES[9] / 10.0,
            d[..., 10] * FEATURE_SCALES[10] / 20.0,
            x[..., 10] - x[..., 0],
            (wheel_accel - x[..., 4] * FEATURE_SCALES[4]) / 20.0,
            (lateral_proxy - x[..., 5] * FEATURE_SCALES[5]) / 20.0,
            x[..., 6] - 0.981,
        ], -1)
        result = torch.cat([x, derived.clamp(-10, 10)], -1)
        return torch.where(mask[..., None], result, torch.zeros_like(result))

    def distribution(self, features: Tensor, valid: Tensor):
        if features.ndim != 3 or features.shape[1:] != (self.spec.n_frames, self.spec.n_features):
            raise ValueError("features do not match adaptive checkpoint feature spec")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean (batch, frames) mask")
        x = self.derived_features(features, valid)
        mask = valid.flip(1)
        t = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask)
        order = torch.argsort(t + (~mask).long() * mask.shape[1], dim=1)
        x = x.gather(1, order[..., None].expand(-1, -1, x.shape[-1]))
        sequence, _ = self.gru(x)
        n = valid.sum(1)
        h = sequence.gather(1, (n - 1).clamp_min(0)[:, None, None].expand(-1, 1, sequence.shape[-1]))[:, 0]
        h = torch.where((n > 0)[:, None], h, torch.zeros_like(h))
        raw = self.head(torch.cat([h, n[:, None].to(h.dtype) / self.spec.n_frames], 1))
        q = nn.functional.softplus(raw[:, :3]).cumsum(1)
        return q, raw[:, 3]

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        return self.distribution(features, valid)[0]


class AdaptiveGripEstimator:
    """Nominal until evidence; immediate decreases and smoothed increases after evidence.

    ``calibrated_candidate`` is stateless and safe for shuffled evaluation. ``lower_mu``
    tracks one ordered stream per batch row. Call reset_filter when that stream ends.
    Faults explicitly enter a conservative state instead of trusting nominal friction.
    """

    def __init__(self, net: AdaptiveGripNet, spec: FeatureSpec, calibration: dict,
                 meta: dict, consumption: AdaptiveConsumption = None):
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.spec = spec.validate()
        if net.spec != self.spec:
            raise ValueError("network and estimator feature specs differ")
        self.consumption = (consumption or AdaptiveConsumption()).validate()
        if self.consumption.nominal_mu > spec.mu_max:
            raise ValueError("nominal_mu exceeds declared training support")
        self.calibration, self.meta = dict(calibration), dict(meta)
        self.meta["format"] = FORMAT
        self.meta["consumption"] = asdict(self.consumption)
        self.cal_delta = float(calibration.get("cal_delta", 0.0))
        if not math.isfinite(self.cal_delta) or self.cal_delta < 0:
            raise ValueError("cal_delta must be finite and nonnegative")
        self.feature_spec, self.mu_range = spec.to_meta(), (spec.mu_min, spec.mu_max)
        self.arch, self.sha = {"model_kind": FORMAT, **net.arch}, ""
        self.alpha = 1.0 - math.exp(-spec.control_dt / spec.filter_tau)
        self._filtered: Optional[Tensor] = None
        self._has_evidence: Optional[Tensor] = None
        self._fault_held: Optional[Tensor] = None

    def _ensure(self, batch, device, dtype):
        if (self._filtered is None or self._filtered.numel() != batch
                or self._filtered.device != device or self._filtered.dtype != dtype):
            self._filtered = torch.full((batch,), self.consumption.nominal_mu, device=device, dtype=dtype)
            self._has_evidence = torch.zeros(batch, dtype=torch.bool, device=device)
            self._fault_held = torch.zeros(batch, dtype=torch.bool, device=device)

    def reset_filter(self, ids: Optional[Tensor] = None):
        if self._filtered is not None:
            selection = slice(None) if ids is None else ids.to(self._filtered.device).reshape(-1)
            self._filtered[selection] = self.consumption.nominal_mu
            self._has_evidence[selection] = False
            self._fault_held[selection] = False

    @torch.no_grad()
    def quantiles(self, features: Tensor, valid: Tensor):
        return self.net(features, valid)

    @torch.no_grad()
    def calibrated_candidate(self, features: Tensor, valid: Tensor):
        q, logit = self.net.distribution(features, valid)
        confidence = logit.sigmoid()
        sensor_finite = (torch.isfinite(features) | ~valid[..., None]).all(dim=(1, 2))
        finite = sensor_finite & torch.isfinite(q).all(1) & torch.isfinite(logit) & torch.isfinite(confidence)
        warm = valid.sum(1) >= self.spec.warm_frames
        informative = warm & finite & (confidence >= self.consumption.confidence_threshold)
        raw = (q[:, 0] - self.cal_delta).clamp(self.consumption.physical_floor, self.spec.mu_max)
        cand = torch.where(informative, raw, torch.full_like(raw, self.consumption.nominal_mu))
        cand = torch.where(finite, cand, torch.full_like(cand, self.consumption.fault_mu))
        reason = torch.where(~finite, 1, torch.where(~warm, 0, torch.where(informative, 2, 3))).to(torch.int8)
        ex = excitation_frames(features, valid)
        diag = {"q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2], "q_gap": q[:, 1] - q[:, 0],
                "posterior_mu": q[:, 1], "calibrated_lower_mu": raw,
                "calibration_discount": torch.full_like(raw, self.cal_delta),
                "candidate": cand, "warm": warm, "finite": finite, "fallback_reason": reason,
                "confidence": confidence, "informative": informative, "has_evidence": informative,
                "fault": ~finite, "fault_now": ~finite, "excitation_pass": ex["union"] >= 5,
                "excitation_long": ex["long"], "excitation_lat": ex["lat"]}
        return cand, diag

    @torch.no_grad()
    def lower_mu(self, features: Tensor, valid: Tensor):
        cand, diag = self.calibrated_candidate(features, valid)
        self._ensure(features.shape[0], cand.device, cand.dtype)
        ema = self._filtered + self.alpha * (cand - self._filtered)
        updated = torch.minimum(cand, ema)
        filtered = torch.where(diag["informative"], updated, self._filtered)
        filtered = torch.where(diag["fault"], torch.minimum(filtered, cand), filtered)
        self._filtered = filtered
        self._has_evidence = self._has_evidence | diag["informative"]
        self._fault_held = (self._fault_held & ~diag["informative"]) | diag["fault_now"]
        diag["fault"] = self._fault_held.clone()
        diag["used_mu"], diag["has_evidence"] = filtered, self._has_evidence.clone()
        return filtered, diag


def save_adaptive_grip_estimator(path, net, spec, calibration, meta, consumption=None):
    consumption = (consumption or AdaptiveConsumption()).validate()
    if "checkpoint_sha256" in meta:
        raise ValueError("checkpoint hash must be computed from file bytes")
    if net.spec != spec:
        raise ValueError("network and saved feature specs differ")
    torch.save({"format": FORMAT, "state_dict": net.state_dict(), "feature_spec": spec.validate().to_meta(),
                "arch": dict(net.arch), "calibration": dict(calibration), "meta": dict(meta),
                "consumption": asdict(consumption)}, path)
    return file_sha256(path)


def load_adaptive_grip_estimator(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("format") != FORMAT:
        raise ValueError("not an adaptive_grip_v2 checkpoint")
    spec = FeatureSpec.from_meta(ck["feature_spec"])
    if set(ck["arch"]) != {"hidden", "wheelbase"}:
        raise ValueError("missing or unsupported adaptive architecture metadata")
    if set(ck["consumption"]) != set(AdaptiveConsumption.__dataclass_fields__):
        raise ValueError("missing or unsupported adaptive consumption metadata")
    net = AdaptiveGripNet(spec, **ck["arch"]).to(device)
    net.load_state_dict(ck["state_dict"])
    est = AdaptiveGripEstimator(net, spec, ck["calibration"], ck["meta"], AdaptiveConsumption(**ck["consumption"]))
    est.sha = file_sha256(path)
    return est
