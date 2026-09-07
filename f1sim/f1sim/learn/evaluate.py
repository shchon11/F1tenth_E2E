"""Evaluate a checkpoint (or the teacher) on held-out tracks with fixed seeds; optional robustness sweeps."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..gym_env import EnvConfig
from ..params import Config
from . import common
from .model import load_checkpoint
from .obs import flatten_obs


def evaluate(ckpt: str, tracks, envs: int, steps: int, speed_cap: float, device, seed=123, cfg: Config = None,
             teacher=False, action_mode="direct") -> dict:
    trs, rls = common.load_tracks(tracks, racelines=teacher)
    model = None; _ = {}
    if not teacher:
        model, _ = load_checkpoint(ckpt, device); model.eval()
    mode = "plan" if (model is not None and model.meta.get("act_dim", 2) >= 5) or (teacher and action_mode == "plan") else "direct"
    sp = (_ if teacher else _).get("spec", {}) if not teacher else {}
    env = common.make_env(trs, envs, device, EnvConfig(speed_cap=speed_cap, resample_track_on_reset=True, action_mode=mode,
                                                        scan_stack=sp.get("scan_stack", 3), scan_stride=sp.get("scan_stride", 1),
                                                        hist_len=sp.get("hist_len", 0), hist_stride=sp.get("hist_stride", 2)), cfg=cfg, seed=seed)
    env.sim.warmup()
    if teacher:
        t = common.make_teacher(rls, env)
        pol = lambda o: env.teacher_label(t)
    else:
        pol = lambda o: model.act(*flatten_obs(o), deterministic=True)[0]
    return common.rollout_metrics(env, pol, steps, speed_cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=""); ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--action-mode", default="direct", choices=["direct", "plan"], help="teacher baseline: which action space it is scored through")
    ap.add_argument("--tracks", default="eval", help="'train', 'eval' or comma separated catalog names"); ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2400, help="60 s at 40 Hz"); ap.add_argument("--speed-cap", type=float, default=8.0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--sweep", action="store_true", help="robustness: mu, latency, lidar height")
    ap.add_argument("--per-track", action="store_true")
    a = ap.parse_args()
    tracks = common.track_names(a.tracks)
    if a.per_track:
        res = {t: evaluate(a.ckpt, [t], a.envs, a.steps, a.speed_cap, a.device, teacher=a.teacher, action_mode=a.action_mode) for t in tracks}
    else:
        res = {"nominal": evaluate(a.ckpt, tracks, a.envs, a.steps, a.speed_cap, a.device, teacher=a.teacher, action_mode=a.action_mode)}
    if a.sweep:
        for name, grp, key, vals in (("mu", "vehicle", "mu", [0.7, 0.85, 1.05]), ("cmd_delay", "actuator", "cmd_delay", [0.02, 0.06, 0.1]),
                                     ("lidar_z", "lidar", "mount_z", [0.12, 0.18])):
            for v in vals:
                cfg = Config(); cfg.rand.enabled = False; setattr(getattr(cfg, grp), key, v)
                res[f"{name}={v}"] = evaluate(a.ckpt, tracks, a.envs, a.steps, a.speed_cap, a.device, cfg=cfg, teacher=a.teacher, action_mode=a.action_mode)
    for k, v in res.items():
        print(f"{k:20s} coll {v['collision_rate']:.3f}  progress {v['progress_rate_mps']:.2f} m/s  speed {v['mean_speed']:.2f}  lap {v['lap_time_s']:.1f} s")
    if not a.per_track:
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
