"""Can the student's observation recover the grip the teacher's label was built from?

The privileged teacher picks its speed profile from each env's true friction (`RacelineTeacher.
grip_bin`, 12 levels between 0.45 and 1.0 of nominal). The student never sees friction, so if the
observation does not determine it, two identical observations carry speed labels up to 1/0.45 ~ 2.2x
apart and the Huber regression can only fit their conditional mean: too fast on a slippery car, too
slow on a grippy one. `--hist-len 20` exists to make grip inferable from how the car answered its own
commands; whether it does has never been measured.

This trains a small probe on exactly what the actor sees and reports how much of the grip variance it
explains, with an ablation of the proprio history block. Read it as: R^2 near 1 means the true-grip
label is fine, R^2 near 0 means `--teacher-grip nominal` removes real label noise.

    python -m f1sim.learn.grip_probe --tracks train --envs 256 --steps 400
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from ..gym_env import EnvConfig
from . import common
from .model import ScanStem
from .obs import flatten_obs


class GripProbe(nn.Module):
    """Same encoder shape as the actor, single scalar output."""

    def __init__(self, n_stack: int, n_beams: int, proprio_dim: int, scan_stem: str = "resnet"):
        super().__init__()
        self.stem = ScanStem(n_stack, n_beams, scan_stem=scan_stem)
        self.pro = nn.Sequential(nn.Linear(proprio_dim, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(256 + 128, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, scan, proprio):
        return self.head(torch.cat([self.stem(scan), self.pro(proprio)], 1)).squeeze(1)


@torch.no_grad()
def collect(env, teacher, steps: int, device):
    """Drive the teacher and record (scan, proprio, true relative grip) for every step."""
    obs, _ = env.reset()
    scans, pros, grips = [], [], []
    mu_nom, muf_nom = teacher.mu_nom, teacher.mu_f_nom
    for _ in range(steps):
        scan, pro = flatten_obs(obs)
        g = ((env.sim.P["mu"] * env.sim.P["mu_f_scale"]) / (mu_nom * muf_nom)).clamp(max=1.0)
        scans.append(scan.half().cpu()); pros.append(pro.half().cpu()); grips.append(g.cpu())
        obs, *_ = env.step(env.teacher_label(teacher))
    return torch.cat(scans), torch.cat(pros), torch.cat(grips)


def fit(scan, pro, grip, device, epochs: int, batch: int, hist_start: int | None, mask_hist: bool,
        n_stack: int, n_beams: int) -> dict:
    """Fit the probe; mask_hist zeroes the proprio history block to ablate it."""
    n = scan.shape[0]
    split = int(0.8 * n)
    idx = torch.randperm(n)
    train_idx, test_idx = idx[:split], idx[split:]
    probe = GripProbe(n_stack, n_beams, pro.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=3e-4)

    def batch_of(sel):
        s = scan[sel].to(device).float(); p = pro[sel].to(device).float()
        if mask_hist and hist_start is not None:
            p = p.clone(); p[:, hist_start:] = 0.0
        return s, p, grip[sel].to(device).float()

    for _ in range(epochs):
        perm = train_idx[torch.randperm(train_idx.numel())]
        for i in range(0, perm.numel() - batch + 1, batch):
            s, p, g = batch_of(perm[i:i + batch])
            loss = nn.functional.mse_loss(probe(s, p), g)
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in range(0, test_idx.numel() - batch + 1, batch):
            s, p, g = batch_of(test_idx[i:i + batch])
            preds.append(probe(s, p).cpu()); trues.append(g.cpu())
    pred = torch.cat(preds).numpy(); true = torch.cat(trues).numpy()
    resid = float(np.sqrt(np.mean((pred - true) ** 2)))
    var = float(true.var())
    return {"r2": 1.0 - resid ** 2 / var if var > 0 else float("nan"),
            "rmse_grip": resid, "grip_std": float(np.sqrt(var)),
            "speed_label_ratio_1sigma": float((1 + resid) / max(1 - resid, 1e-3))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="train")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hist-len", type=int, default=20)
    ap.add_argument("--hist-stride", type=int, default=2)
    ap.add_argument("--scan-stack", type=int, default=6)
    ap.add_argument("--scan-stride", type=int, default=2)
    ap.add_argument("--speed-cap", type=float, default=4.0)
    ap.add_argument("--action-mode", default="plan", choices=["direct", "plan"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    names = common.track_names(a.tracks)
    tracks, rls = common.load_tracks(names, racelines=True)
    env = common.make_env(tracks, a.envs, device,
                          EnvConfig(speed_cap=a.speed_cap, action_mode=a.action_mode, hist_len=a.hist_len,
                                    hist_stride=a.hist_stride, scan_stack=a.scan_stack, scan_stride=a.scan_stride))
    teacher = common.make_teacher(rls, env)
    env.sim.warmup()
    spec = common.obs_spec(env)
    scan, pro, grip = collect(env, teacher, a.steps, device)
    print(f"collected {scan.shape[0]} samples, grip in [{grip.min():.2f}, {grip.max():.2f}]", flush=True)
    # the history block is the tail of the proprio vector (see ObsSpec.proprio_dim)
    hist_start = spec.proprio_dim - spec.hist_len * spec.row_dim if spec.hist_len else None
    full = fit(scan, pro, grip, device, a.epochs, a.batch, hist_start, False, spec.scan_stack, spec.n_beams)
    print(f"with proprio history : R2 {full['r2']:+.3f}  rmse {full['rmse_grip']:.3f} grip "
          f"(grip std {full['grip_std']:.3f}); a 1-sigma error spans a {full['speed_label_ratio_1sigma']:.2f}x speed label")
    if hist_start is not None:
        none = fit(scan, pro, grip, device, a.epochs, a.batch, hist_start, True, spec.scan_stack, spec.n_beams)
        print(f"history masked out   : R2 {none['r2']:+.3f}  rmse {none['rmse_grip']:.3f} grip")
        print(f"=> the 1 s proprio history explains {100 * (full['r2'] - none['r2']):.0f} pp of grip variance")


if __name__ == "__main__":
    main()
