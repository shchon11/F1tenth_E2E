"""Shared training utilities: track sets, env construction, run dirs, W&B."""
from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
import torch

from .. import maps
from ..params import Config
from ..gym_env import EnvConfig, F1VecEnv
from ..raceline import Raceline
from ..teacher import RacelineTeacher
from ..track import Track
from .obs import ObsSpec

RUNS_DIR = os.path.join(os.path.expanduser("~"), "f1sim_runs")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")     # None -> the account's default entity (org-scoped keys reject the org itself)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "f1sim-e2e")


# Training set (user-curated, 2026-09): real competition SLAM maps are the bulk, a few scaled F1
# circuits and procedural tracks for variety, every map in both lap directions (~rev).
REAL_TRAIN = ["icra2022", "blackbox2021_1", "blackbox2021_2", "blackbox2021_3", "blackbox2022_1", "blackbox2022_2"]
RT_TRAIN = ["Spielberg", "Oschersleben"]
GEN_TRAIN = ["gen:competition:1000", "gen:competition:1001"]          # 1000 CCW, 1001 mirrored (CW)
TRAIN_TRACKS = ([f"real:{n}{d}" for n in REAL_TRAIN for d in ("", "~rev")]
                + [f"rt:{n}{d}" for n in RT_TRAIN for d in ("", "~rev")]
                + [f"{n}{d}" for n in GEN_TRAIN for d in ("", "~rev")]
                # static box obstacles (CDC 2025 style cardboard boxes) on ~37 % of the tracks
                + [f"real:{n}+obs{i}{d}" for i, n in enumerate(REAL_TRAIN) for d in ("", "~rev")])
# Held out entirely: the Korea 2025 championship map (the target venue style) and one TU Wien race.
EVAL_TRACKS = ["real:korea_2025_iccas", "real:korea_2025_iccas~rev", "real:blackbox2022_3", "real:blackbox2022_3~rev",
               "rt:Monza", "gen:competition:0"]


def track_names(spec: str = "train") -> List[str]:
    """'train' -> TRAIN_TRACKS, 'eval' -> EVAL_TRACKS, otherwise a comma separated catalog list."""
    if spec == "train":
        return list(TRAIN_TRACKS)
    if spec == "eval":
        return list(EVAL_TRACKS)
    return [n.strip() for n in spec.split(",") if n.strip()]


def load_tracks(names, racelines: bool = False):
    tracks = [maps.load(n) for n in names]
    rls = [Raceline.build_cached(t) for t in tracks] if racelines else None
    return tracks, rls


def make_env(tracks, num_envs, device, env_cfg: Optional[EnvConfig] = None, cfg: Optional[Config] = None, seed: int = 0,
             rls=None):
    """rls: racelines (one per track) -- required when env_cfg.opponent == "teacher" (opponents follow them)."""
    cfg = cfg or Config()
    cfg.sim.seed = seed
    env = F1VecEnv(tracks, cfg, env_cfg or EnvConfig(), num_envs=num_envs, device=device)
    if env.M > 1 and env.ecfg.opponent == "teacher":
        if rls is None:
            rls = [Raceline.build_cached(t) for t in tracks]
        env.set_teacher(make_teacher(rls, env))
    return env


def obs_spec(env: F1VecEnv) -> ObsSpec:
    e = env.ecfg
    return ObsSpec(n_beams=env.n_beams, scan_stack=e.scan_stack, scan_stride=e.scan_stride, action_history=e.action_history,
                   range_max=env.range_max, v_max=e.v_max_policy, gyro_scale=e.imu_gyro_scale, accel_scale=e.imu_accel_scale)


def make_teacher(rls, env: F1VecEnv):
    return RacelineTeacher(rls, wheelbase=env.cfg.vehicle.lf + env.cfg.vehicle.lr, device=env.device)


def run_dir(name: str) -> str:
    d = os.path.join(RUNS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def wandb_init(name: str, config: dict, group: Optional[str] = None, mode: Optional[str] = None):
    import wandb
    return wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name, group=group, config=config,
                      mode=mode or os.environ.get("WANDB_MODE", "online"), dir=run_dir(name))


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter(); dt = now - self.t; self.t = now; return dt


@torch.no_grad()
def rollout_metrics(env: F1VecEnv, policy_fn, steps: int, speed_cap: Optional[float] = None) -> dict:
    """Run a policy (obs -> normalized action) for `steps` control steps on all envs.
    collision_rate = collisions / episodes started (running episodes count as started);
    progress_rate  = mean track progress per second over all envs and steps (lap-time proxy)."""
    if speed_cap is not None:
        env.set_speed_cap(speed_cap)
    obs, info = env.reset()
    lm = env.learner; nL = int(lm.sum())                     # races with teacher opponents: learners only
    n_coll = 0; n_ended = 0; prog = 0.0; lap_times = []; speeds = []
    for t in range(steps):
        a = policy_fn(obs)
        obs, rew, term, trunc, info = env.step(a)
        prog += info["progress"][lm].sum().item()
        speeds.append(env.sim.state[lm, 3].mean().item())
        lap_times += info["lap_times"][lm[info["lap_ids"]]].tolist()
        if "final" in info:
            f = info["final"]; m = lm[f["ids"]]
            n_ended += int(m.sum()); n_coll += int(f["collided"][m].sum())
    started = nL + n_ended
    return {"collision_rate": n_coll / started, "collisions_per_min": n_coll / (steps * env.sim.control_dt / 60) / nL * 60 / 60,
            "progress_rate_mps": prog / (nL * steps * env.sim.control_dt), "mean_speed": float(np.mean(speeds)),
            "episodes_started": started, "lap_time_s": float(np.mean(lap_times)) if lap_times else float("nan")}
