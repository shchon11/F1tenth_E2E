"""Evaluate deterministic policies with rolling or first-attempt trial protocols."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..gym_env import EnvConfig
from ..params import Config
from . import common
from .evaluation_metrics import TrialAccumulator
from .model import load_checkpoint
from .obs import flatten_obs


@torch.no_grad()
def evaluate(ckpt: str, tracks, envs: int, steps: int, speed_cap: float, device, seed=123, cfg: Config = None,
             teacher=False, action_mode="direct", *, protocol="rolling", race_size=1, opponent="policy") -> dict:
    """Keep rolling metrics compatible; trials count only initial learner attempts."""
    if protocol not in ("rolling", "trials") or steps < 1 or envs < 1 or race_size < 1 or envs % race_size:
        raise ValueError("Require a valid protocol, positive steps/envs, and envs divisible by race_size")
    torch.manual_seed(seed)
    np.random.seed(seed)
    trs, rls = common.load_tracks(tracks, racelines=teacher or (race_size > 1 and opponent == "teacher"))
    model = None
    metadata = {}
    if not teacher:
        model, metadata = load_checkpoint(ckpt, device)
        model.eval()
    mode = "plan" if (model is not None and model.meta.get("act_dim", 2) >= 5) or (teacher and action_mode == "plan") else "direct"
    spec = metadata.get("spec", {})
    ecfg = EnvConfig(speed_cap=speed_cap, resample_track_on_reset=True, action_mode=mode,
                     race_size=race_size, opponent=opponent,
                     scan_stack=spec.get("scan_stack", 3), scan_stride=spec.get("scan_stride", 1),
                     hist_len=spec.get("hist_len", 0), hist_stride=spec.get("hist_stride", 2))
    if protocol == "trials":
        ecfg.max_steps = steps
    env = common.make_env(trs, envs, device, ecfg, cfg=cfg, seed=seed, rls=rls)
    env.sim.warmup()
    if teacher:
        teacher_policy = common.make_teacher(rls, env)
        def policy(obs):
            return env.teacher_label(teacher_policy)
    else:
        def policy(obs):
            return model.act(*flatten_obs(obs), deterministic=True)[0]
    if protocol == "rolling":
        result = common.rollout_metrics(env, policy, steps, speed_cap)
        for key, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                result[key] = None
    else:
        obs, _ = env.reset(seed=seed)
        learner = env.learner.clone()
        initial_ids = env.sim.tid[learner].cpu().numpy().copy()
        lengths = env.sim.track.length[env.sim.tid[learner]].cpu().numpy().astype(np.float64)
        if np.any(lengths <= 0):
            raise ValueError("Trial evaluation requires positive centerline lengths")
        trials = TrialAccumulator(lengths, env.sim.control_dt)
        for _ in range(steps):
            obs, _, term, trunc, info = env.step(policy(obs))
            # Privileged transition velocity is captured before any simulator autoreset.
            speed = torch.linalg.vector_norm(info['priv'][learner, :2], dim=1)
            dynamics = info['priv'][learner, :5].cpu().numpy()
            trials.update(info['progress'][learner].cpu().numpy(), speed.cpu().numpy(),
                          term[learner].cpu().numpy(), trunc[learner].cpu().numpy(), yaw_rate=dynamics[:, 2],
                          heading_error=dynamics[:, 4], longitudinal_speed=dynamics[:, 0], lateral_speed=dynamics[:, 1])
            if not trials.active.any():
                break
        trials.finish()
        result = trials.report()
        result['initial_track_ids'] = initial_ids.tolist()
        result['initial_track_lengths_m'] = lengths.tolist()
    config_report = asdict(env.cfg)
    dropout = config_report['lidar']['dropout_value']
    if not math.isfinite(dropout):
        config_report['lidar']['dropout_value'] = str(dropout)
    result['metadata'] = {
        'protocol': protocol, 'tracks': list(tracks), 'seed': seed,
        'seeds': {'numpy': seed, 'torch': seed, 'simulator': seed, 'reset': seed if protocol == 'trials' else None},
        'checkpoint': str(ckpt), 'teacher': teacher, 'deterministic_policy': True,
        'action_mode': mode, 'device': str(device), 'steps': steps, 'step_dt': env.sim.control_dt,
        'time_budget_s': steps * env.sim.control_dt, 'envs': envs,
        'race_size': race_size, 'opponent': opponent, 'learners': int(env.learner.sum()),
        'race_convention': 'respawning traffic; not strict no-respawn competition',
        'spawn_convention': 'slot 0 starts ahead of following opponents; completion does not establish overtaking',
        'trial_success': 'signed cumulative progress >= initial track length; collision takes precedence' if protocol == 'trials' else None,
        'distance_convention': ('ground-truth planar speed integrated over active control steps' if protocol == 'trials'
                                else 'distance not reported; mean_speed averages post-step longitudinal speed including resets'),
        'instability_convention': ('spin entry: heading error crosses beyond 90 deg; high yaw rate: >5 rad/s; '
                                   'large slip: >20 deg while speed >1 m/s' if protocol == 'trials' else None),
        'config': config_report, 'env_config': asdict(env.ecfg),
        'config_nonfinite_convention': 'nonfinite lidar dropout sentinel encoded as a string',
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--action-mode", default="direct", choices=["direct", "plan"])
    ap.add_argument("--tracks", default="eval", help="'train', 'eval', 'eval_obstacles' or comma separated catalog names")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2400, help="60 s at 40 Hz")
    ap.add_argument("--speed-cap", type=float, default=8.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", action="store_true", help="robustness: mu, latency, lidar height")
    ap.add_argument("--per-track", action="store_true")
    ap.add_argument("--protocol", choices=["rolling", "trials"], default="rolling")
    ap.add_argument("--race-size", type=int, default=1)
    ap.add_argument("--opponent", choices=["policy", "teacher"], default="policy")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--output", type=Path, help="write the complete strict JSON report")
    ap.add_argument("--eager", action="store_true", help="disable simulator compilation (CPU smoke tests)")
    a = ap.parse_args()
    if a.steps < 1 or a.envs < 1 or a.race_size < 1 or a.envs % a.race_size:
        ap.error("steps/envs/race-size must be positive and envs divisible by race-size")
    if not a.teacher and not a.ckpt:
        ap.error("provide a checkpoint or --teacher")
    tracks = common.track_names(a.tracks)

    def run(names, config):
        return evaluate(a.ckpt, names, a.envs, a.steps, a.speed_cap, a.device, seed=a.seed, cfg=config,
                        teacher=a.teacher, action_mode=a.action_mode, protocol=a.protocol,
                        race_size=a.race_size, opponent=a.opponent)

    nominal = Config()
    if a.eager:
        nominal.sim.compile = False
    res = {t: run([t], nominal) for t in tracks} if a.per_track else {"nominal": run(tracks, nominal)}
    if a.sweep:
        for name, grp, key, vals in (("mu", "vehicle", "mu", [0.7, 0.85, 1.05]),
                                     ("cmd_delay", "actuator", "cmd_delay", [0.02, 0.06, 0.1]),
                                     ("lidar_z", "lidar", "mount_z", [0.12, 0.18])):
            for value in vals:
                cfg = Config()
                cfg.rand.enabled = False
                cfg.sim.compile = not a.eager
                setattr(getattr(cfg, grp), key, value)
                if a.per_track:
                    for track in tracks:
                        res[f"{name}={value}/{track}"] = run([track], cfg)
                else:
                    res[f"{name}={value}"] = run(tracks, cfg)
    report = json.dumps(res, indent=2, allow_nan=False)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
