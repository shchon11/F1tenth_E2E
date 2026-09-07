"""Actor-critic for the LiDAR-only policy. Actor sees (scan stack, proprio); the critic additionally
sees the privileged vector (asymmetric actor-critic). Actor is TensorRT-friendly (conv1d + MLP)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScanStem(nn.Module):
    """1D CNN over the beam axis. 1080 beams -> 256 features."""

    def __init__(self, n_stack: int, n_beams: int, out: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_stack, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(64, 32, 3, stride=2, padding=1), nn.GELU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, n_stack, n_beams)).numel()
        self.fc = nn.Sequential(nn.Linear(n, out), nn.GELU())

    def forward(self, scan):
        return self.fc(self.conv(scan).flatten(1))


class Actor(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, hidden: int = 256, log_std_init: float = -0.7, act_dim: int = 2):
        super().__init__()
        self.stem = ScanStem(n_stack, n_beams)
        pw = 64 if proprio_dim <= 32 else 128                     # a proprio history (hundreds of inputs) gets a wider embedding
        self.pro = nn.Sequential(nn.Linear(proprio_dim, pw), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(256 + pw, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        nn.init.zeros_(self.mu.bias); self.mu.weight.data.mul_(0.1)

    def forward(self, scan, proprio):
        h = self.mlp(torch.cat([self.stem(scan), self.pro(proprio)], 1))
        return torch.tanh(self.mu(h))

    def dist(self, scan, proprio):
        mu = self(scan, proprio)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))


class Critic(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, priv_dim: int, hidden: int = 256):
        super().__init__()
        self.stem = ScanStem(n_stack, n_beams)
        self.pro = nn.Sequential(nn.Linear(proprio_dim + priv_dim, 128), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(256 + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, scan, proprio, priv):
        return self.mlp(torch.cat([self.stem(scan), self.pro(torch.cat([proprio, priv], 1))], 1)).squeeze(1)


class ActorCritic(nn.Module):
    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, priv_dim: int, act_dim: int = 2):
        super().__init__()
        self.actor = Actor(n_stack, n_beams, proprio_dim, act_dim=act_dim)
        self.critic = Critic(n_stack, n_beams, proprio_dim, priv_dim)
        self.meta = dict(n_stack=n_stack, n_beams=n_beams, proprio_dim=proprio_dim, priv_dim=priv_dim, act_dim=act_dim)

    @torch.no_grad()
    def act(self, scan, proprio, deterministic=False):
        d = self.actor.dist(scan, proprio)
        a = d.mean if deterministic else d.sample()
        return a.clamp(-1, 1), d.log_prob(a).sum(1)

    def evaluate(self, scan, proprio, priv, actions):
        d = self.actor.dist(scan, proprio)
        return d.log_prob(actions).sum(1), d.entropy().sum(1), self.critic(scan, proprio, priv), d


def save_checkpoint(path, model: ActorCritic, extra: Optional[dict] = None):
    torch.save({"state_dict": model.state_dict(), "meta": model.meta, "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu", override: Optional[dict] = None) -> Tuple[ActorCritic, dict]:
    """override: meta fields to change (e.g. priv_dim for a multi-car critic, scan_stack); tensors whose
    shape no longer matches are left at their fresh initialization and listed in extra["skipped"]."""
    ck = torch.load(path, map_location=device)
    meta = dict(ck["meta"]); meta.update(override or {})
    m = ActorCritic(**meta).to(device)
    sd = m.state_dict(); skipped = []
    for k_, v in ck["state_dict"].items():
        if k_ in sd and sd[k_].shape == v.shape:
            sd[k_] = v
        else:
            skipped.append(k_)
    m.load_state_dict(sd)
    extra = dict(ck.get("extra", {})); extra["skipped"] = skipped
    return m, extra
