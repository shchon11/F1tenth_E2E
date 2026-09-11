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


REAL_TRAIN = ["icra2022", "blackbox2021_1", "blackbox2021_2", "blackbox2021_3", "blackbox2022_1",
              "blackbox2022_2"]
RT_TRAIN = ["Spielberg", "Oschersleben"]

# The procedural mix is chosen from what was measured to be missing, not from how many seeds a
# generator can produce. Against the real venues the old set had three holes:
#
#   fold-back   how close the lap comes to itself while being far away along it: every generator
#               stayed 5.8-7.0 m away, every real venue folds to 2.4-3.3 m behind a single hose.
#               `serpentine` was written for this and lands at 1.3-2.5 m.
#   pinch       local width over the narrowest spot near it: 1.04-1.05 procedurally against 1.35-1.98
#               on the blackbox maps. `+pinch` closes the lane down at a few places on any style.
#   width var   0.07-0.15 against 0.26. `control` (hairpins, chicanes, varying width) was in the
#               code and in no track set at all.
#
# `competition` is cut back rather than grown: its own docstring says more seeds add little new
# geometry, and it was 32 of 112 tracks. `hallway` is cut hardest -- its scans are so self-similar
# (aliasing 0.087 against 0.28-0.58 everywhere else) that far-apart places are indistinguishable to
# a LiDAR-only policy, which is teaching one observation two answers.
# `serpentine` is written and measured but held back from training for now: it delivers the fold-back
# (1.3-2.5 m against 5.8-7.0 for every other generator) and with the loop seam rounded the teacher's
# episode failures dropped tenfold, but it still runs at 18 collisions/km against 0.2-1.4 elsewhere.
# The residual cause is not the generator: two lanes 3 m apart make the centerline projection
# ambiguous, and progress, lap counting and the wrong-way check all read that projection. Windowing
# the search around the previous index fixed the *real* folded venues (korea_2026 went to 0.35
# collisions/km, s no longer jumping) but not a track that folds this often. Fold-back exposure comes
# from those real maps until the projection is solid enough to carry procedural folds too.
GEN_TRAIN = ([f"gen:control:{seed}" for seed in range(1400, 1408)]         # hairpins, chicanes, width
             + [f"gen:competition:{seed}" for seed in range(1000, 1008)]
             + [f"gen:circuit:{seed}" for seed in range(1200, 1204)]
             + [f"gen:hallway:{seed}" for seed in range(1100, 1104)])
GEN_PINCH = ([f"gen:control:{s}+pinch{s}" for s in range(1408, 1414)]
             + [f"gen:competition:{s}+pinch{s}" for s in range(1008, 1012)])
TRAIN_DIRECTIONS = ("", "~rev", "~mir", "~mir~rev")

# Obstacles are half the point of the exercise, so they get their own share of the set rather than
# one variant per real map. `+obs` sits boxes against a lane edge; `+rlobs` puts them ON the racing
# line, which is the case that actually has to be avoided -- the teacher itself goes from 0.22 to
# 1.67 collisions/km on those, and there were none in any track set.
# The venue this car actually raced at, with obstacles. Its clean laps stay held out, and the
# obstacle seeds here are disjoint from the ones in EVAL_OBSTACLE_TRACKS, so the eval asks the
# question worth asking about a known circuit: the layout is familiar, the obstacles are not.
# Every situation axis is covered on it -- boxes at the lane edge, boxes on the racing line drawn
# across sight-distance bands, and the lane closing down -- in both directions and mirrored.
KOREA26 = "real:korea_2026_competition"
KOREA26_TRAIN = ([f"{KOREA26}+obs{s}{d}" for s in (201, 202, 203) for d in ("", "~rev", "~mir")]
                 + [f"{KOREA26}+rlobs{s}{d}" for s in (211, 212, 213, 214) for d in ("", "~rev", "~mir")]
                 + [f"{KOREA26}+pinch{s}{d}" for s in (221, 222) for d in ("", "~rev")])

TRAIN_OBSTACLES = (list(KOREA26_TRAIN)
                   + [f"real:{n}+obs{i}{d}" for i, n in enumerate(REAL_TRAIN) for d in ("", "~rev")]
                   + [f"real:{n}+rlobs{i + 40}{d}" for i, n in enumerate(REAL_TRAIN) for d in ("", "~rev")]
                   + [f"gen:control:{s}+rlobs{s}" for s in range(1414, 1420)]
                   + [f"gen:competition:{s}+rlobs{s}" for s in range(1012, 1016)])

TRAIN_TRACKS = ([f"real:{n}{d}" for n in REAL_TRAIN for d in TRAIN_DIRECTIONS]
                + [f"rt:{n}{d}" for n in RT_TRAIN for d in TRAIN_DIRECTIONS]
                + [f"{n}{d}" for n in GEN_TRAIN for d in ("", "~rev")]
                + list(GEN_PINCH) + list(TRAIN_OBSTACLES))

# Held out entirely. Grouped by the axis each one probes, so a failure says which kind of novelty
# broke it rather than only that something did.
EVAL_TRACKS = [
    "real:korea_2025_iccas", "real:korea_2025_iccas~rev",   # folded real venue, never seen
    "real:blackbox2022_3", "real:blackbox2022_3~rev",       # pinched real venue
    "rt:Monza",                                             # long, fast, smooth
    "gen:competition:0",                                    # the familiar family, unseen seed
    "real:korea_2026_competition", "real:korea_2026_competition~rev",   # folded real venue
    "gen:control:9100",                                     # hairpins and chicanes
    "gen:competition:9200+pinch9200",                       # sudden narrowing
]
EVAL_OBSTACLE_TRACKS = (
    [f"{n}+obs{s}{d}" for n, s in (("real:korea_2025_iccas", 101), ("real:blackbox2022_3", 102),
                                   ("gen:competition:0", 103))
     for d in ("", "~rev")]
    + [f"{n}+rlobs{s}{d}" for n, s in (("real:korea_2025_iccas", 111), ("real:blackbox2022_3", 112),
                                       ("gen:control:9102", 113))
       for d in ("", "~rev")]
    # the competition venue, obstacle seeds it has never trained on
    + [f"{KOREA26}+obs{s}{d}" for s in (901, 902) for d in ("", "~rev")]
    + [f"{KOREA26}+rlobs{s}{d}" for s in (911, 912, 913) for d in ("", "~rev")]
    + [f"{KOREA26}+pinch921", f"{KOREA26}+pinch921~rev"]
)


def track_names(spec: str = "train") -> List[str]:
    """'train' -> TRAIN_TRACKS, 'eval' -> EVAL_TRACKS, otherwise a comma separated catalog list."""
    if spec == "train":
        return list(TRAIN_TRACKS)
    if spec == "eval":
        return list(EVAL_TRACKS)
    if spec == "eval_obstacles":
        return list(EVAL_OBSTACLE_TRACKS)
    if spec == "eval_all":
        return list(EVAL_TRACKS) + list(EVAL_OBSTACLE_TRACKS)
    return [n.strip() for n in spec.split(",") if n.strip()]


# A viewer and a training job share one GPU, and the trainer will happily take all of it: measured,
# a viewer next to a full-rate PPO run got 1-2 fps and 125 ms per sim step regardless of how many
# cars it drew -- the cost was waiting for the device, not computing. Rather than stop training to
# look at it, the viewer raises a flag and the trainer gives back a slice of each update. The flag
# carries a timestamp so a viewer that dies does not throttle training forever.
VIEWER_FLAG = os.path.join(os.path.expanduser("~"), ".cache", "f1sim", "viewer_active")
VIEWER_CONTROL = os.path.join(os.path.expanduser("~"), ".cache", "f1sim", "viewer_control.json")


def viewer_command(**fields) -> None:
    """Ask a running viewer to do something (a launcher panel writes, the viewer polls).

    A file rather than a socket because the panel has to be its own process: Tk and the GL window
    each want to own the main loop, and running both in one process makes the viewer stutter or the
    panel stop repainting.
    """
    import json
    try:
        os.makedirs(os.path.dirname(VIEWER_CONTROL), exist_ok=True)
        with open(VIEWER_CONTROL, "w") as f:
            json.dump(dict(fields, t=time.time()), f)
    except OSError:
        pass


def viewer_poll_command(since: float):
    """(command, mtime) if the control file is newer than `since`, else (None, since)."""
    import json
    try:
        m = os.path.getmtime(VIEWER_CONTROL)
        if m <= since:
            return None, since
        with open(VIEWER_CONTROL) as f:
            return json.load(f), m
    except (OSError, ValueError):
        return None, since


def viewer_heartbeat() -> None:
    """Called by the viewer: 'I am on screen, leave me some GPU'."""
    try:
        os.makedirs(os.path.dirname(VIEWER_FLAG), exist_ok=True)
        with open(VIEWER_FLAG, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def viewer_gone() -> None:
    try:
        os.remove(VIEWER_FLAG)
    except OSError:
        pass


def viewer_active(max_age: float = 5.0) -> bool:
    """True while a viewer has checked in recently."""
    try:
        return (time.time() - os.path.getmtime(VIEWER_FLAG)) < max_age
    except OSError:
        return False


def raceline_clearance(track, rl) -> float:
    """Smallest gap between the raceline and anything solid [m]."""
    from scipy import ndimage
    edt = ndimage.distance_transform_edt(~track.occupancy).astype(np.float32) * track.resolution
    j = np.clip(((rl.xy[:, 0] - track.origin[0]) / track.resolution).astype(int), 0, edt.shape[1] - 1)
    i = np.clip(((rl.xy[:, 1] - track.origin[1]) / track.resolution).astype(int), 0, edt.shape[0] - 1)
    return float(edt[i, j].min())


def load_tracks(names, racelines: bool = False, drop_infeasible: bool = True, half_width: float = 0.281 / 2,
                **raceline_kw):
    """raceline_kw is forwarded to Raceline.build (e.g. margin=0.55 for a line with more slack;
    the on-disk cache key includes it, so different margins do not collide).

    drop_infeasible: leave out any track whose raceline does not physically fit. The raceline
    optimiser searches a corridor anchored to the centerline, so an obstacle covering the centerline
    leaves no corridor and the line comes back running straight through it. DAgger can only teach
    what the teacher shows, so such a track does not make the student worse at obstacles -- it
    teaches it to drive into them. Loud, because a silently shorter track set is its own bug.
    """
    tracks = [maps.load(n) for n in names]
    if not racelines:
        return tracks, None
    rls = [Raceline.build_cached(t, **raceline_kw) for t in tracks]
    if drop_infeasible:
        keep, dropped = [], []
        for n, t, rl in zip(names, tracks, rls):
            c = raceline_clearance(t, rl)
            (keep if c >= half_width + 0.05 else dropped).append((n, t, rl, c))
        if dropped:
            print(f"WARNING: dropping {len(dropped)} track(s) whose raceline does not fit the car:",
                  flush=True)
            for n, _, _, c in dropped:
                print(f"    {n:52s} clearance {c:.3f} m < {half_width + 0.05:.3f}", flush=True)
        tracks = [t for _, t, _, _ in keep]
        rls = [rl for _, _, rl, _ in keep]
    return tracks, rls


def make_env(tracks, num_envs, device, env_cfg: Optional[EnvConfig] = None, cfg: Optional[Config] = None, seed: int = 0,
             rls=None, teacher_grip: str = "true", teacher_recover_time: float = 0.0):
    """rls: racelines (one per track) -- required when env_cfg.opponent == "teacher" (opponents follow them)."""
    cfg = cfg or Config()
    cfg.sim.seed = seed
    env = F1VecEnv(tracks, cfg, env_cfg or EnvConfig(), num_envs=num_envs, device=device)
    if rls is not None and getattr(env.ecfg, "reward_lap_time", 0.0) > 0:
        env.set_ideal_lap(rls)                       # reference for the seconds-saved reward
    if env.M > 1 and env.ecfg.opponent in ("teacher", "mixed"):
        if rls is None:
            rls = [Raceline.build_cached(t) for t in tracks]
        env.set_teacher(make_teacher(rls, env, grip=teacher_grip, recover_time=teacher_recover_time))
    return env


def obs_spec(env: F1VecEnv) -> ObsSpec:
    e = env.ecfg
    return ObsSpec(n_beams=env.n_beams, scan_stack=e.scan_stack, scan_stride=e.scan_stride, action_history=e.action_history,
                   act_dim=env.act_dim, hist_len=e.hist_len, hist_stride=e.hist_stride, range_max=env.range_max, v_max=e.v_max_policy,
                   gyro_scale=e.imu_gyro_scale, accel_scale=e.imu_accel_scale)


def make_teacher(rls, env: F1VecEnv, grip: str = "true", recover_time: float = 0.0):
    """grip: which friction the teacher's speed profile assumes. "true" is privileged -- the label then
    depends on mu, which the student cannot observe, so identical scans get speed labels up to ~2.2x
    apart and the regression learns their conditional mean. "nominal"/"conservative" are constant and
    therefore imitable. Run `python -m f1sim.learn.grip_probe` to measure whether the proprio history recovers mu at all."""
    t = RacelineTeacher(rls, wheelbase=env.cfg.vehicle.lf + env.cfg.vehicle.lr, device=env.device,
                        recover_time=recover_time)
    t.label_grip = grip
    return t


def run_dir(name: str) -> str:
    d = os.path.join(RUNS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def wandb_init(name: str, config: dict, group: Optional[str] = None, mode: Optional[str] = None,
               resume_id: Optional[str] = None):
    """resume_id: append to an existing run instead of starting a new one.

    A long training job is resumed several times -- a new track set, opponents added, a flag changed
    -- and each resume carries the whole network over. Logging each leg as its own run breaks the
    curves into pieces that restart at step 0, so the one thing the charts are for (did this get
    better or worse than before?) has to be reassembled by eye across tabs.
    """
    import wandb
    return wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=name, group=group, config=config,
                      mode=mode or os.environ.get("WANDB_MODE", "online"), dir=run_dir(name),
                      id=resume_id, resume="allow" if resume_id else None)


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter(); dt = now - self.t; self.t = now; return dt


@torch.no_grad()
def rollout_metrics(env: F1VecEnv, policy_fn, steps: int, speed_cap: Optional[float] = None,
                    per_track: bool = False) -> dict:
    """Run a policy (obs -> normalized action) for `steps` control steps on all envs.

    collisions_per_km is the metric to watch: a hazard rate per metre driven, independent of the
    rollout length and comparable across tracks. `collision_rate` (collisions / episodes started,
    with running episodes counted as started) is kept for continuity but is a function of `steps`
    and must not be compared across runs with different rollout lengths.

    per_track: also return the hazard rate for each track id. The set average is dominated by the
    easy tracks -- a 2.1 coll/km average hid a 12 coll/km track in the same set -- so the maximum
    over tracks is what tells you whether anything is actually broken.
    """
    if speed_cap is not None:
        env.set_speed_cap(speed_cap)
    obs, info = env.reset()
    lm = env.learner; nL = int(lm.sum())                     # races with teacher opponents: learners only
    n_coll = 0; n_ended = 0; prog = 0.0; lap_times = []; speeds = []
    T = env.sim.track.T
    coll_by_track = torch.zeros(T, device=env.device); dist_by_track = torch.zeros(T, device=env.device)
    dt = env.sim.control_dt
    for t in range(steps):
        a = policy_fn(obs)
        tid_before = env.sim.tid.clone()                      # the track this transition happened on
        obs, rew, term, trunc, info = env.step(a)
        prog += info["progress"][lm].sum().item()
        speeds.append(env.sim.state[lm, 3].mean().item())
        lap_times += info["lap_times"][lm[info["lap_ids"]]].tolist()
        if per_track:
            step_dist = torch.where(lm, env.sim.state[:, 3].abs() * dt, torch.zeros_like(env.sim.state[:, 3]))
            dist_by_track.index_add_(0, tid_before, step_dist)
            coll_by_track.index_add_(0, tid_before, (term & lm).float())
        if "final" in info:
            f = info["final"]; m = lm[f["ids"]]
            n_ended += int(m.sum()); n_coll += int(f["collided"][m].sum())
    started = nL + n_ended
    distance_m = float(np.mean(speeds)) * nL * steps * dt
    out = {"collision_rate": n_coll / started,
           "collisions_per_min": n_coll / (steps * dt / 60) / nL,
           "collisions_per_km": n_coll * 1000 / distance_m if distance_m > 0 else float("nan"),
           "progress_rate_mps": prog / (nL * steps * dt), "mean_speed": float(np.mean(speeds)),
           "episodes_started": started, "lap_time_s": float(np.mean(lap_times)) if lap_times else float("nan")}
    if per_track:
        d = dist_by_track.cpu().numpy(); c = coll_by_track.cpu().numpy()
        rate = np.where(d > 0, c * 1000 / np.maximum(d, 1e-9), np.nan)
        out["collisions_per_km_by_track"] = rate.tolist()
        out["collisions_per_km_worst"] = float(np.nanmax(rate)) if np.isfinite(rate).any() else float("nan")
        out["collisions_per_km_p90"] = float(np.nanpercentile(rate, 90)) if np.isfinite(rate).any() else float("nan")
        out["worst_track_index"] = int(np.nanargmax(rate)) if np.isfinite(rate).any() else -1
    return out
