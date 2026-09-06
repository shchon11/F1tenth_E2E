"""Per-environment parameter tensors with domain randomization."""
from __future__ import annotations

from dataclasses import fields
from typing import Dict

import torch

from .params import Config


class ParamSet:
    """Flattened (B,) tensors for every numeric field of vehicle/actuator/lidar/odom params.
    Access as ps.P['mu'] etc. Field names are unique across groups by construction."""

    GROUPS = ("vehicle", "actuator", "lidar", "odom", "imu")

    def __init__(self, cfg: Config, num_envs: int, device, generator: torch.Generator):
        self.cfg = cfg
        self.B = num_envs
        self.device = torch.device(device)
        self.gen = generator
        self.nominal: Dict[str, float] = {}
        self.group_of: Dict[str, str] = {}
        for g in self.GROUPS:
            dc = getattr(cfg, g)
            for f in fields(dc):
                v = getattr(dc, f.name)
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                if f.name in self.nominal:
                    raise KeyError(f"duplicate param name {f.name}")
                self.nominal[f.name] = float(v)
                self.group_of[f.name] = g
        self.P: Dict[str, torch.Tensor] = {k: torch.full((num_envs,), v, device=self.device)
                                           for k, v in self.nominal.items()}

    def resample(self, env_ids: torch.Tensor):
        """Re-draw randomized params for the given envs (reset). Non-randomized stay nominal."""
        n = env_ids.numel()
        if n == 0:
            return
        rc = self.cfg.rand
        for key, (lo, hi) in rc.ranges.items():
            grp, name = key.split(".")
            if name not in self.P:
                continue
            if not rc.enabled:
                self.P[name][env_ids] = self.nominal[name]
                continue
            u = torch.rand(n, device=self.device, generator=self.gen) * (hi - lo) + lo
            if key in rc.scale_fields:
                self.P[name][env_ids] = self.nominal[name] * u
            else:
                self.P[name][env_ids] = u
        # keep 0 <= v_blend_min < v_blend_max etc. -- nothing randomized there currently

    def set_all(self, name: str, value: float):
        self.nominal[name] = float(value)
        self.P[name].fill_(float(value))
