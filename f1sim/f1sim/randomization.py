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
        # one (K, B) buffer; every P[name] is a row view of it, so a reset re-draws all randomized
        # parameters with a single random draw and a single masked write instead of ~150 small kernels
        names = list(self.nominal)
        self.buf = torch.stack([torch.full((num_envs,), self.nominal[k], device=self.device) for k in names])
        self.P: Dict[str, torch.Tensor] = {k: self.buf[i] for i, k in enumerate(names)}
        rc = cfg.rand
        rows, lo, hi, scale, nom = [], [], [], [], []
        for key, (l, h) in rc.ranges.items():
            _, name = key.split(".")
            if name not in self.P:
                continue
            rows.append(names.index(name)); lo.append(l); hi.append(h); scale.append(key in rc.scale_fields); nom.append(self.nominal[name])
        t = lambda x, dt=torch.float32: torch.tensor(x, device=self.device, dtype=dt)
        self.r_rows, self.r_lo, self.r_hi, self.r_scale, self.r_nom = t(rows, torch.long), t(lo), t(hi), t(scale, torch.bool), t(nom)

    def resample(self, env_ids: torch.Tensor):
        """Re-draw randomized params for the given envs (reset). Non-randomized stay nominal."""
        n = env_ids.numel()
        if n == 0:
            return
        K = self.r_rows.numel()
        if K == 0:
            return
        if not self.cfg.rand.enabled:
            vals = self.r_nom[:, None].expand(K, n)
        else:
            u = torch.rand(K, n, device=self.device, generator=self.gen) * (self.r_hi - self.r_lo)[:, None] + self.r_lo[:, None]
            vals = torch.where(self.r_scale[:, None], self.r_nom[:, None] * u, u)
        self.buf[self.r_rows[:, None], env_ids[None, :]] = vals

    def set_all(self, name: str, value: float):
        self.nominal[name] = float(value)
        self.P[name].fill_(float(value))
