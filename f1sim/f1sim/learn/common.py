"""Shared training utilities: track sets, env construction, run dirs, W&B."""
from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np
import torch

from .. import maps
from .. import tracks
from ..params import Config
from ..gym_env import EnvConfig, F1VecEnv
from ..raceline import Raceline
from ..teacher import RacelineTeacher
from ..track import Track
from .obs import ObsSpec

RUNS_DIR = os.path.join(os.path.expanduser("~"), "f1sim_runs")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")     # None -> the account's default entity (org-scoped keys reject the org itself)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "f1sim-e2e")


# ---------------------------------------------------------------- track sets
# These used to be literal lists of loader strings, a hundred and forty-nine of them, and the only
# way to see what the split *was* was to read the list comprehensions that built it. They are now
# generated from `f1sim.tracks`: a registry of base tracks with short ids, and one `SplitRule` per
# track saying which directions and which obstacle seeds that track contributes. The rules and the
# reasoning behind them live in `tracks.py` (torch-free, so the console can read them too); what is
# here is the same strings, in the same order, under the names every caller already uses.
#
# `tests/test_tracks.py` holds the previous literals as a frozen oracle and compares string for
# string. Nothing downstream may change meaning: a checkpoint's manifest, a frozen benchmark suite
# and a W&B config all name their tracks in the loader's grammar.
REAL_TRAIN = [tracks.get(t).legacy.split(":", 1)[1] for t in tracks.REAL_TRAIN_IDS]
RT_TRAIN = [tracks.get(t).legacy.split(":", 1)[1] for t in tracks.RT_TRAIN_IDS]
GEN_TRAIN = [tracks.get(t).legacy for t in tracks.GEN_TRAIN_IDS]
TRAIN_DIRECTIONS = ("", "~rev", "~mir", "~mir~rev")

#: The venue this car actually raced at, and a training venue only -- see `tracks.KOREA26_ID`.
KOREA26 = tracks.get(tracks.KOREA26_ID).legacy
KOREA26_TRAIN = [n for r in tracks.TRAIN_RULES if r.track == tracks.KOREA26_ID for n in r.legacy_names()]

TRAIN_TRACKS = tracks.split_names("train")

#: The floors that have never been seen in any form, which is what makes a generalisation number
#: possible. See `tracks.HELDOUT_BASE_IDS` for the measurements behind the choice.
HELDOUT_BASE_MAPS = tuple(tracks.get(t).legacy for t in tracks.HELDOUT_BASE_IDS)

HELDOUT_TRACKS = tracks.split_names("heldout")
#: Kept as the name every caller already uses. It is the same list, not a copy: `track_names("eval")`
#: and the viewer's "검증" group must not be able to drift away from the held-out definition.
EVAL_TRACKS = HELDOUT_TRACKS

HELDOUT_OBSTACLE_TRACKS = tracks.split_names("heldout_obstacles")
EVAL_OBSTACLE_TRACKS = HELDOUT_OBSTACLE_TRACKS


def base_map(name: str) -> str:
    """Catalog name with every variant suffix stripped: `real:x+rlobs7~mir~rev` -> `real:x`.

    The suffix lists come from `maps` rather than being restated here, so a modifier or obstacle
    family added to the catalog is covered without a second list to keep in sync. The loop runs to
    a fixed point because the two kinds of suffix can be written in either order.
    """
    prev = None
    while prev != name:
        prev = name
        for m in maps.MODIFIERS:
            if name.endswith(m):
                name = name[:-len(m)]
        stripped, kind, _ = maps._split_obstacle_suffix(name)
        if kind is not None:
            name = stripped
    return name


def heldout_leakage(train_names, heldout_names) -> List[str]:
    """Training entries that touch a held-out base map. Empty means the split is clean.

    Comparing base maps rather than full names is the whole point: `real:map16x07+obs5~mir` is a
    different string from `real:map16x07` but the same floor, and a held-out number measured on a
    floor the policy trained on -- in any direction, with any obstacles stamped into it -- is not a
    generalisation number. Returned in the order given so the caller can name the offenders.
    """
    held = {base_map(n) for n in heldout_names}
    return [n for n in train_names if base_map(n) in held]


SPLIT_SPECS = ("train", "eval", "heldout", "eval_obstacles", "heldout_obstacles",
               "eval_all", "heldout_all")


def track_names(spec: str = "train", draws: int = 8, seed: int = 0) -> List[str]:
    """A split name or a comma separated list -> loader names.

    Both grammars are accepted in the list, and the new one may leave the obstacle seed open:

        --tracks 'real/bb22-1@rev#line:*'      one map, `draws` random box placements
        --tracks 'real/bb22-1+rlobs44~rev'     the old spelling of one of them

    `draws` only applies to an entry whose seed is `*`; anything else is one track however large it
    is. The draw comes from `random.Random(seed)` shared across the whole list, so the same
    `--seed` gives the same set and the manifest can record the concrete names.
    """
    spec = (spec or "").strip()
    if spec == "train":
        return list(TRAIN_TRACKS)
    if spec in ("eval", "heldout"):
        return list(HELDOUT_TRACKS)
    if spec in ("eval_obstacles", "heldout_obstacles"):
        return list(HELDOUT_OBSTACLE_TRACKS)
    if spec in ("eval_all", "heldout_all"):
        return list(HELDOUT_TRACKS) + list(HELDOUT_OBSTACLE_TRACKS)
    return tracks.expand_all([n.strip() for n in spec.split(",") if n.strip()], draws, seed)


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


def announce_episode_boundaries(env):
    """Wrap `env.step` / `env.reset` so a registered recurrent policy is cleared where episodes end.

    Only for callers that cannot thread the hidden state through by hand. Training, evaluation and
    the benchmark all do thread it and register nothing, so the listener list is empty and this
    costs one `if` per step. The one caller that cannot is the viewer worker
    (`f1sim/viewer/sim_worker.py`), which builds its env here and its actor callable in
    `learn.watch.actor_runner` -- two places that never meet, in a file this branch does not own.
    So the env side announces and the policy side listens; see `learn.memory`.
    """
    from . import memory as memory_mod
    step, reset = env.step, env.reset

    def step_announcing(action):
        out = step(action)
        if memory_mod._LISTENERS:
            _obs, _rew, term, trunc, _info = out
            memory_mod.broadcast_boundary(term | trunc, batch=int(env.B))
        return out

    def reset_announcing(*a, **kw):
        out = reset(*a, **kw)
        if memory_mod._LISTENERS:
            memory_mod.broadcast_boundary(None, batch=int(env.B))
        return out

    env.step, env.reset = step_announcing, reset_announcing
    return env


def make_env(tracks, num_envs, device, env_cfg: Optional[EnvConfig] = None, cfg: Optional[Config] = None, seed: int = 0,
             rls=None, teacher_grip: str = "true", teacher_recover_time: float = 0.0,
             opponent_pool: bool = True):
    """rls: racelines (one per track) -- required when env_cfg.opponent == "teacher" (opponents follow them).

    opponent_pool: load and install the checkpoint population `env_cfg.opp_pool` names when the mode
    is "pool". On by default because an env in that mode refuses to step without one; a caller that
    wants to install its own (a test with stub drivers) passes False.
    """
    cfg = cfg or Config()
    cfg.sim.seed = seed
    env = F1VecEnv(tracks, cfg, env_cfg or EnvConfig(), num_envs=num_envs, device=device)
    announce_episode_boundaries(env)
    if rls is not None and getattr(env.ecfg, "reward_lap_time", 0.0) > 0:
        env.set_ideal_lap(rls)                       # reference for the seconds-saved reward
    if env.M > 1 and env.teacher_any:
        if rls is None:
            rls = [Raceline.build_cached(t) for t in tracks]
        env.set_teacher(make_teacher(rls, env, grip=teacher_grip, recover_time=teacher_recover_time))
    if opponent_pool and env.ecfg.opponent == "pool":
        from .opponent_pool import attach     # local: that module imports this one for the obs spec
        attach(env, device=env.device)
    return env


def obs_spec(env: F1VecEnv) -> ObsSpec:
    e = env.ecfg
    return ObsSpec(n_beams=env.n_beams, scan_stack=e.scan_stack, scan_stride=e.scan_stride, action_history=e.action_history,
                   act_dim=env.act_dim, hist_len=e.hist_len, hist_stride=e.hist_stride, range_max=env.range_max, v_max=e.v_max_policy,
                   gyro_scale=e.imu_gyro_scale, accel_scale=e.imu_accel_scale, opp_token=env.opp_token)


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
                    per_track: bool = False, controller=None) -> dict:
    """Run a policy (obs -> normalized action) for `steps` control steps on all envs.

    collisions_per_km is the metric to watch: a hazard rate per metre driven, independent of the
    rollout length and comparable across tracks. `collision_rate` (collisions / episodes started,
    with running episodes counted as started) is kept for continuity but is a function of `steps`
    and must not be compared across runs with different rollout lengths.

    per_track: also return the hazard rate for each track id. The set average is dominated by the
    easy tracks -- a 2.1 coll/km average hid a 12 coll/km track in the same set -- so the maximum
    over tracks is what tells you whether anything is actually broken.

    controller: a `grip_runtime.ControllerRuntime`, driven here in the same order `ppo.py` drives
    it -- `begin` once after the reset, `pre_action` before the policy is asked for anything,
    `post_step` immediately after the step. Getting that order wrong is how an arm becomes a legacy
    run wearing another arm's name, so there is one loop that knows it rather than two.
    `policy_fn` may be stateful: if it has a `reset(done=None)` (as `learn.memory.policy_fn` does),
    it is cleared at the seeded reset and at every episode boundary. A recurrent policy scored
    without that carries the last episode's memory into the next one and is not the policy that
    would drive the car.
    """
    if speed_cap is not None:
        env.set_speed_cap(speed_cap)
    # A recurrent policy carries state between calls, so it has to be told where the episodes end.
    # `learn.memory.policy_fn` provides `.reset`; a plain callable has none and nothing happens,
    # which is what every caller that predates memory does.
    policy_reset = getattr(policy_fn, "reset", None)
    obs, info = env.reset()
    if controller is not None:
        controller.begin(obs)
    if callable(policy_reset):
        policy_reset()
    lm = env.learner; nL = int(lm.sum())                     # races with teacher opponents: learners only
    n_coll = 0; n_ended = 0; prog = 0.0; lap_times = []; speeds = []
    T = env.sim.track.T
    coll_by_track = torch.zeros(T, device=env.device); dist_by_track = torch.zeros(T, device=env.device)
    dt = env.sim.control_dt
    for t in range(steps):
        if controller is not None:
            controller.pre_action(obs)
        a = policy_fn(obs)
        tid_before = env.sim.tid.clone()                      # the track this transition happened on
        obs, rew, term, trunc, info = env.step(a)
        if controller is not None:
            controller.post_step(term, trunc)
        if callable(policy_reset):
            policy_reset(term | trunc)
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
