"""Shared training utilities: track sets, env construction, run dirs, W&B."""
from __future__ import annotations

import json
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

#: Where runs and checkpoints are written. `$F1SIM_RUNS` overrides it, so a shared machine,
#: a scratch disk or a second checkout does not have to live in one home directory.
RUNS_DIR = os.path.abspath(os.path.expanduser(
    os.environ.get("F1SIM_RUNS") or os.path.join("~", "f1sim_runs")))
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")     # None -> the account's default entity (org-scoped keys reject the org itself)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "f1sim-e2e")


def validate_policy_observation(saved: dict, env) -> None:
    """Reject shape-compatible but numerically different policy observations."""
    import math
    if not isinstance(saved, dict) or not saved:
        raise ValueError("policy checkpoint has no observation specification")
    cls = ObsSpec
    if int(saved.get("version", 1)) == 2:
        # contract v2 (`learn.obs.ObsSpecV2`): the same field-by-field comparison, on its own fields,
        # with a dict older than one of them read as what it meant then (`ObsSpecV2.from_dict`)
        from .obs import ObsSpecV2 as cls
    fields = {key: value for key, value in saved.items() if key in cls.__dataclass_fields__}
    expected = cls.from_dict(fields) if hasattr(cls, "from_dict") else cls(**fields)
    actual = obs_spec(env)
    if type(actual) is not cls:
        raise ValueError(f"policy observation contract v{int(saved.get('version', 1))}, environment emits "
                         f"{type(actual).__name__}")
    for key in cls.__dataclass_fields__:
        want, got = getattr(expected, key), getattr(actual, key)
        if isinstance(want, float) or isinstance(got, float):
            same = math.isfinite(float(want)) and math.isfinite(float(got)) \
                and math.isclose(float(want), float(got), rel_tol=0., abs_tol=1e-9)
        else:
            same = want == got
        if not same:
            raise ValueError(f"policy observation {key}: checkpoint {want!r}, environment {got!r}")


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
    if "!assets=" in name:
        name = name.rsplit("!assets=", 1)[0]
    name = tracks.parse(name).legacy()
    prev = None
    while prev != name:
        prev = name
        for m in maps.MODIFIERS:
            if name.endswith(m):
                name = name[:-len(m)]
        name, _ = maps._split_bare(name)
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
    track = track.for_planning()
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
             opponent_pool: bool = True, teacher_limits: Optional[dict] = None):
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
        env.set_teacher(make_teacher(rls, env, grip=teacher_grip, recover_time=teacher_recover_time, **(teacher_limits or {})))
    if opponent_pool and env.pool_paths:
        from .opponent_pool import attach     # local: that module imports this one for the obs spec
        attach(env, device=env.device)
    if getattr(env, "external_slot_rows", None):
        env.set_external_drivers(external_drivers(env))
    # the call, so `make_twin` can build the same world again (twin worlds, `learn.twin_progress`)
    env._make_call = dict(tracks=tracks, device=device, env_cfg=env_cfg, cfg=cfg, seed=seed, rls=rls,
                          teacher_grip=teacher_grip, teacher_recover_time=teacher_recover_time,
                          opponent_pool=opponent_pool, teacher_limits=teacher_limits)
    return env


def make_twin(env, num_envs: int):
    """Another env of the same world as `env` (its own `make_env` call again, with `num_envs` rows). It shares the
    configuration objects, so a runtime change to `env.ecfg` (a speed-cap curriculum) reaches it too."""
    call = getattr(env, "_make_call", None)
    if call is None:
        raise RuntimeError("this env was not built by common.make_env: no call to repeat")
    c = dict(call)
    return make_env(c.pop("tracks"), int(num_envs), c.pop("device"), c.pop("env_cfg"), cfg=c.pop("cfg"), **c)


def external_drivers(env) -> dict:
    """grid slot -> a `learn.baselines` driver for every external slot of `env.slots`, bound to the
    simulator's own scanner (its beam count, field of view and range), as `baseline_node` binds to
    the car's `LaserScan`. End2Race's repo is the vendored checkout next to this one unless the slot
    names another."""
    import os as _os
    from . import baselines
    lid = env.cfg.lidar
    out = {}
    for i, sl in enumerate(env.slots, start=1):
        if not sl.external_driven:
            continue
        opts = dict(sl.options or {})
        if sl.driver.external == "end2race":
            opts.setdefault("repo", _os.path.join(_os.path.dirname(__file__), "..", "..", "..",
                                                  "external", "baselines", "End2Race"))
            opts.setdefault("device", "cpu")
        drv = baselines.load(sl.driver.external, sl.weights or "none", **opts)
        drv.bind_scanner(n_beams=int(lid.n_beams), fov=float(lid.fov), range_max=float(lid.range_max))
        out[i] = drv
    return out


def opponent_reading_feeder(model, env):
    """`EnvConfig.opponent_speed_limit` / `.obstacle_speed_limit`: a callable that copies the actor's latest nearest-car
    output into `env.opponent_reading` (call it after each policy step, before `env.step`), or None when both limits are
    off. Both read the policy's own estimate of the car ahead (the obstacle limit leaves that car to the car limit), so a
    checkpoint without that output cannot drive with either on."""
    reading = getattr(env, "opponent_reading", None)
    if reading is None:
        return None
    actor = getattr(model, "actor", None)
    if getattr(actor, "near_car", None) is None:
        raise ValueError("the speed limits for a car ahead and for an obstacle on the plan's path (opponent_speed_limit, "
                         "obstacle_speed_limit) read the policy's nearest-car output; this checkpoint has none (a v2 "
                         "policy trained with --aux-near-car has)")
    actor.keep_reading = True

    def feed():
        r = getattr(actor, "last_reading", None)
        if r is None:
            return
        if r.shape[0] == reading.shape[0]:
            reading.copy_(r)
        else:                                              # the actor ran on the learner rows only
            reading[env.learner] = r.to(reading.dtype)
    return feed


def student_policy(model, env: F1VecEnv, device=None, deterministic: bool = True, dial=None):
    """`memory.policy_fn` for any student: carries the hidden state, and for a conditional actor reads
    the conditioning at every step (it changes at each reset, so it cannot be fixed).

    dial: for a `dial` checkpoint, a callable returning the (B,) friction the policy is told to use.
    None means "set exactly right" -- the env's true friction -- which is the reference a supervisor
    is scored against, not something the car has.

    Simulator-side only. A `true_mu` checkpoint is a lab oracle -- its input is privileged -- which
    is exactly why the deployment loaders refuse it and why this lives next to the env."""
    from . import conditioning as cond_mod
    from .memory import policy_fn, runtime_for
    from .obs import flatten_obs
    feed = opponent_reading_feeder(model, env)
    if not int(model.meta.get("cond_dim", 0)):
        base = policy_fn(model, env.B, device=device, deterministic=deterministic)
        if feed is None:
            return base

        def run_fed(obs):
            action = base(obs)
            feed()
            return action
        run_fed.runtime, run_fed.reset = base.runtime, base.reset
        return run_fed
    spec = cond_mod.CondSpec.from_meta(model.meta.get("cond"))
    rt = runtime_for(model, env.B, device)

    @torch.no_grad()
    def run(obs):
        scan, proprio = flatten_obs(obs)
        if spec.source == "dial":
            c = cond_mod.mu_to_c(env.sim.P["mu"] if dial is None else dial(), spec)
        else:
            c = cond_mod.make_condition(spec.source, spec, env.sim.P["mu"].reshape(-1, 1), 0)
        action, _lp, rt.hidden = model.act(rt.observe(scan, proprio), proprio, deterministic=deterministic, c=c, h=rt.hidden)
        if feed is not None:
            feed()
        return action

    run.runtime = rt
    run.reset = rt.reset
    return run


def obs_spec(env: F1VecEnv):
    """The observation contract this env emits: `ObsSpec` (v1, every env before 2026-10-14) or, for
    `EnvConfig.obs_version == 2`, `ObsSpecV2`."""
    e = env.ecfg
    if getattr(env, "obs_version", 1) == 2:
        from .obs import ObsSpecV2
        return ObsSpecV2(n_beams=env.n_beams, n_frames=e.scan_stack, frame_stride=e.scan_stride,
                         act_dim=env.act_dim, speed_mode=e.speed_mode, hist_len=e.hist_len,
                         hist_stride=e.hist_stride, range_max=env.range_max, v_max=e.v_max_policy,
                         gyro_scale=e.imu_gyro_scale, ego_actions=bool(e.ego_actions)).validate()
    return ObsSpec(n_beams=env.n_beams, scan_stack=e.scan_stack, scan_stride=e.scan_stride, action_history=e.action_history,
                   act_dim=env.act_dim, hist_len=e.hist_len, hist_stride=e.hist_stride, range_max=env.range_max, v_max=e.v_max_policy,
                   gyro_scale=e.imu_gyro_scale, accel_scale=e.imu_accel_scale,
                   opp_token=env.opp_token)


def teacher_limits(a_lat: Optional[float] = None, a_acc: Optional[float] = None,
                   a_brake: Optional[float] = None) -> dict:
    """The speed-profile limits a caller actually set, as keywords. Both `Raceline.build` and
    `RacelineTeacher` take them under these names and both default to 6 / 6 / 3, so an empty dict is
    the teacher every recorded run had."""
    return {k: float(v) for k, v in (("a_lat", a_lat), ("a_acc", a_acc), ("a_brake", a_brake)) if v is not None}


def plan_output_of(extra: dict) -> dict:
    """The plan-output contract a checkpoint was trained under (`mpc.contract_of`, which the car's
    controller node reads too). Read, never assumed: a profile-mode policy driven as "linear" emits a
    grip belief and an end speed that would be read as two speeds."""
    from ..mpc import contract_of
    return contract_of(extra)


def make_teacher(rls, env: F1VecEnv, grip: str = "true", recover_time: float = 0.0, **limits):
    """grip: which friction the teacher's speed profile assumes. "true" is privileged -- the label then
    depends on mu, which the student cannot observe, so identical scans get speed labels up to ~2.2x
    apart and the regression learns their conditional mean. "nominal"/"conservative" are constant and
    therefore imitable. Run `python -m f1sim.learn.grip_probe` to measure whether the proprio history recovers mu at all.
    limits: `teacher_limits(...)` -- a_lat / a_acc / a_brake of the speed profile the teacher drives."""
    t = RacelineTeacher(rls, wheelbase=env.cfg.vehicle.lf + env.cfg.vehicle.lr, device=env.device,
                        recover_time=recover_time, vehicle=env.cfg.vehicle,
                        mu_nominal=env.cfg.vehicle.mu,
                        mu_f_scale_nominal=env.cfg.vehicle.mu_f_scale, **limits)
    t.label_grip = grip
    return t


def run_dir(name: str) -> str:
    d = os.path.join(RUNS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


#: The file both trainers write their curve to, next to the checkpoints.
PROGRESS_FILENAME = "progress.jsonl"


def _jsonable(v):
    """A plain Python number/str/bool/list, or None for anything else.

    numpy scalars and 0-d tensors are what a metric dict is actually full of, and `json.dumps`
    refuses both. NaN and +/-inf are kept (`json.dumps` writes them as `NaN`/`Infinity`): a metric
    that is not defined this update is data -- "no lap was completed" -- and dropping the key
    instead would silently shorten one series against the others.
    """
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return None


class ProgressLog:
    """`<run_dir>/progress.jsonl`: one JSON object per logged step, flushed on every write.

    The console's training dashboard used to read the progress line out of W&B's `output.log` and
    parse it with one regex. Both of those are side effects of somebody else's code: wandb 0.29 no
    longer writes `output.log` at all, a run started with stdout redirected elsewhere leaves nothing
    in its run directory, and every term added to the human line (`gate`, `kl_ref`, `fut`, `floor
    bce`) broke the regex. This file is the trainer's own record, in the run directory, in a format
    that cannot drift: unknown keys are extra series, not a parse failure.

    Never raises. A full disk, a read-only mount or a run directory somebody moved must not take a
    training run with it -- the file is a side output, not a result.
    """

    def __init__(self, run_dir: str, filename: str = PROGRESS_FILENAME):
        self.path = os.path.join(run_dir, filename)
        self._fh = None
        self.error: Optional[str] = None
        try:
            self._fh = open(self.path, "a", buffering=1)
        except OSError as exc:                      # pragma: no cover - disk-level failure
            self.error = str(exc)

    def write(self, record: dict) -> None:
        if self._fh is None:
            return
        try:
            clean = {k: _jsonable(v) for k, v in record.items()}
            self._fh.write(json.dumps({k: v for k, v in clean.items() if v is not None}) + "\n")
            self._fh.flush()
        except (OSError, ValueError, TypeError) as exc:   # pragma: no cover - disk-level failure
            self.error = str(exc)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:                          # pragma: no cover
                pass
            self._fh = None


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
            coll_by_track.index_add_(0, tid_before, (info["contact_onset"] & lm).float())
        # Per step, not per ended episode: under `collision_mode="soft"` nothing terminates, the
        # episodes' `collided` is always 0, and contacts in an episode still running when the
        # evaluation stops would never be counted. Under `terminate` this is the same count.
        n_coll += int((info["contact_onset"] & lm).sum())
        if "final" in info:
            f = info["final"]; m = lm[f["ids"]]
            n_ended += int(m.sum())
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


#: Kinds that may drive the ego car in place of a checkpoint (the console's "내 차", 2026-10-17).
EGO_PLANNER_KINDS = ("raceline", "forzaeth", "forzaeth_pred", "lane_switch")
EGO_EXTERNAL_KINDS = ("tinylidarnet", "end2race", "ftg", "disparity")


class PlannerEgoDriver:
    """A planner driving the ego rows on its OWN path follower (steer, speed). Used through
    `F1VecEnv.set_ego_driver`, which calls `command_env` every step.

    With a `stack` (`modular_stack.ModularStack`, the default from `build_ego_driver`) it drives on
    what the car can know: its pose from a particle filter on its odometry and scan, the other cars
    and the crates from detection and tracking on that scan, the nominal actuator model -- and holds
    its trailing gap behind the nearest moving track. Without one (`estimated=False`) it is handed
    the simulator's state, the opponents' true state and this episode's actuator parameters: an
    ORACLE, an upper bound to label as such, never "the baseline" (the user, 2026-10-18: "모듈형은
    왜 상대 위치랑 자기 위치를 gt로 써?")."""

    needs_speed = False

    #: The local waypoint array handed to the MAP controller: this far behind the car's projection on
    #: the reference line and this far ahead -- past the longest L1 the tuning allows plus the
    #: furthest speed lookahead, race_stack's own window being about as long.
    PATH_BEHIND_M, PATH_AHEAD_M = 1.0, 10.0

    def __init__(self, teacher, kind: str, stack=None, ref=None, tracker=None):
        self.teacher, self.kind = teacher, kind
        self.stack = stack
        self.rows = None
        #: `ref` the reference line teacher (the planner's `base`, or the teacher itself), `tracker` the
        #: path follower: a `map_controller.MAPController`, or None for the reference teacher's own
        #: pure pursuit (what every planner baseline drove on before 2026-10-19)
        self.ref = ref if ref is not None else teacher
        self.tracker = tracker
        #: the venue's speed knob (`EnvConfig.plan_speed_scale`, the same for every driver): the profile MAP
        #: is handed, x this
        self.speed_scale = 1.0

    def command_env(self, env, rows):
        import torch
        if self.tracker is not None:
            st = self.stack.update(env) if self.stack is not None else env.sim.state
            P = self.stack.nominal_P(env) if self.stack is not None else env.sim.P
            cmd = self._map_command(env, rows, st, P)
            if self.stack is not None:
                follow, v_cap = self.stack.follow_cap(env, st)
            elif env.M > 1:
                follow, v_cap = env.follow_cap(env.sim.state)
            else:
                return cmd
            cmd[:, 1] = torch.where(follow[rows], torch.minimum(cmd[:, 1], v_cap[rows]), cmd[:, 1])
            return cmd
        if self.stack is not None:
            st = self.stack.update(env)
            cmd = self.teacher(st, self.stack.nominal_P(env), env.sim.tid)[rows].clone()
            follow, v_cap = self.stack.follow_cap(env, st)
            cmd[:, 1] = torch.where(follow[rows], torch.minimum(cmd[:, 1], v_cap[rows]), cmd[:, 1])
            return cmd
        cmd = self.teacher(env.sim.state, env.sim.P, env.sim.tid)[rows].clone()
        if env.M > 1:                                  # alone on the track there is no car to trail
            follow, v_cap = env.follow_cap(env.sim.state)
            cmd[:, 1] = torch.where(follow[rows], torch.minimum(cmd[:, 1], v_cap[rows]), cmd[:, 1])
        return cmd

    def _map_command(self, env, rows, st, P):
        """(R, 2) (steer, speed) from the MAP controller on this planner's path. The path is what
        race_stack's state machine publishes as local waypoints: the reference line displaced by the
        planner's lateral offsets (`path_offsets`, the evasion spline while passing), with the line's
        speed profile at this car's grip (x the planner's own speed scale) and its curvature."""
        import torch
        from ..spliner_teacher import TRAILING
        ref, planner, mp = self.ref, self.teacher, self.tracker
        dev = st.device
        B = st.shape[0]
        tid = env.sim.tid
        if planner is not ref:
            _d, mode, scale = planner.decide(st, tid)            # what `planner(...)` does first
        else:
            mode, scale = None, torch.ones(B, device=dev)
        s_r, td = st[rows], tid[rows]
        R = rows.numel()
        idx, _ = ref.project(s_r[:, :2], td)
        ds = ref.ds[td]
        ds0 = float(ds.mean())
        ks = torch.arange(-int(round(self.PATH_BEHIND_M / ds0)), int(round(self.PATH_AHEAD_M / ds0)) + 1, device=dev)
        j = torch.remainder(idx[:, None] + ks[None, :], ref.N)                           # (R, n)
        n = ks.numel()
        d = None
        if planner is not ref:
            d = planner.path_offsets(rows, ks.to(s_r.dtype) * ds0, s_r[:, 3])
        if d is None:
            d = torch.zeros(R, n, device=dev, dtype=s_r.dtype)
        d = ref.clamp_offset(d, td[:, None].expand(-1, n), j)
        tn = ref.tan[td[:, None], j]
        path_xy = ref.xy[td[:, None], j] + d[..., None] * torch.stack([-tn[..., 1], tn[..., 0]], -1)
        gb = ref.grip_bin(P, B, dev)[rows]
        path_v = ref.speed_at(td[:, None].expand(-1, n), j, gb[:, None].expand(-1, n))
        path_v = path_v * scale[rows].to(path_v.dtype)[:, None] * float(self.speed_scale)
        kappa = ref.kappa[td[:, None], j]
        t0, p0 = ref.tan[td, idx], ref.xy[td, idx]
        d_glob = t0[:, 0] * (s_r[:, 1] - p0[:, 1]) - t0[:, 1] * (s_r[:, 0] - p0[:, 0])  # frenet d, left +
        lr = env.last_result
        if lr is not None and lr.imu is not None and lr.imu.shape[1] > 0:
            mp.push_imu(lr.imu[rows, :, 3])                     # accelerometer x: longitudinal
        trailing = gap = vs = visible = None
        if mode is not None and self.stack is not None and self.kind in ("forzaeth", "forzaeth_pred"):
            trailing, gap, vs, visible = self._trailing_opponent(rows, mode, idx, td)
        steer, speed = mp(s_r[:, :2], s_r[:, 2], s_r[:, 3], path_xy, path_v, kappa, ds, d_glob, gb,
                          trailing=trailing, opp_gap=gap, opp_vs=vs, opp_visible=visible, ego_vs=s_r[:, 3],
                          yaw_rate=s_r[:, 5])
        return torch.stack([steer, speed / P["speed_gain"][rows]], 1)

    def _trailing_opponent(self, rows, mode, idx, td):
        """race_stack's `obstacle_cb` on the stack's tracks: the obstacle whose rear is nearest ahead in
        s (a dynamic one before any static one, `prioritize_dyn`), its centre gap, speed and whether
        this scan saw it; `trailing` where the planner is TRAILING and there is one."""
        import torch
        from ..spliner_teacher import TRAILING
        st_, ref = self.stack, self.ref
        L = ref.length[td][:, None]
        s_e = (idx.to(st_.obs_s.dtype) * ref.ds[td])[:, None]
        ok = st_.obs_ok[rows]
        s_c = st_.obs_s[rows]
        rear = torch.remainder(s_c - st_.obs_ext[rows][..., 0] - s_e, L)
        dyn_ = ok & (st_.obs_vs[rows].abs() >= st_.per.tp.static_speed)
        prefer = dyn_ if self.tracker.p.prioritize_dyn else ok
        use = torch.where(prefer.any(1, keepdim=True), prefer, ok)
        j = torch.where(use, rear, torch.full_like(rear, float("inf"))).argmin(1)
        ar = torch.arange(rows.numel(), device=rows.device)
        any_ = ok.any(1)
        gap = torch.remainder(s_c[ar, j] - s_e[:, 0], L[:, 0])
        visible = st_.per.misses[ar, j] == 0
        trailing = (mode[rows] == TRAILING) & any_
        return trailing, gap, st_.obs_vs[rows][ar, j], visible

    def reset(self, mask) -> None:
        if self.tracker is not None:
            import torch
            self.tracker.reset(torch.as_tensor(mask, device=self.tracker.curr_steer.device))
        fn = getattr(self.teacher, "reset_rows", None)
        if fn is not None and self.rows is not None:
            import torch
            ids = self.rows[torch.as_tensor(mask, device=self.rows.device)]
            if ids.numel():
                fn(ids)

    def describe(self) -> dict:
        return {"kind": self.kind, "driver": type(self.teacher).__name__,
                "tracker": ("MAP (ForzaETH race_stack)" if self.tracker is not None else "pure pursuit (reference teacher)"),
                "perception": ("particle filter + scan detection/tracking, nominal actuators" if self.stack is not None
                               else "ORACLE: ground-truth pose, opponents and actuator parameters")}


def build_map_tracker(env, ref, params: Optional[dict] = None):
    """A `map_controller.MAPController` for the ego rows of `env`, its steering table built (once, then
    cached on disk) from the NOMINAL car through this simulator's own actuators and tyres, one table per
    grip level of `ref`'s speed profiles (the planner reads the same level for its speeds)."""
    import torch as _torch
    from .. import map_controller as _map
    sim = env.sim
    nominal = sim.params.nominal
    P_row = {k: _torch.tensor([float(v)]) for k, v in nominal.items()}
    mus = [float(ref.mu_nom * g) for g in ref.grip_levels]
    lut = _map.lut_for(P_row, mus, wheel=bool(sim.wheel_model), dt=float(sim.dt), s_max=float(nominal["s_max"]),
                       device=sim.state.device)
    rows = int((env.slot == 0).sum())
    return _map.MAPController(lut, _map.MAPParams.from_dict(params), rows, sim.state.device, float(nominal["s_max"]))


def build_ego_driver(env, kind: str, weights: str = "", options: Optional[dict] = None, rls=None,
                     limits: Optional[dict] = None, estimated: bool = True, stack_params: Optional[dict] = None):
    """The ego driver the console's "내 차" names: a `PlannerEgoDriver` over our raceline teacher (wrapped
    in the published planner's class for forzaeth / forzaeth_pred / lane_switch, as the opponents
    are), or a `learn.baselines` driver bound to the simulator's scanner. `rls` is the line a planner
    drives (with `limits` its own profile); `options` are the baseline's own (`baselines.load`)."""
    import os as _os
    if kind in EGO_PLANNER_KINDS:
        if not rls:
            raise ValueError(f"ego '{kind}' drives a raceline, and this session has none")
        # `stack_params` carries the planner's own knobs as well as the stack's: "teacher" -- the
        # reference follower's (`RacelineTeacher`: its speed profile's a_lat / a_acc / a_brake and
        # v_max_profile, the pure-pursuit lookahead, the off-line slowdown), "planner" -- the
        # spliner's / lane switcher's. They are what a team tunes, and what a fair comparison tunes.
        sp = dict(stack_params or {})
        # "grip": which friction the planner's speed profile and MAP's steering table are read at -- "true" (each
        # car's own, a simulator oracle), or "nominal" (one floor for every car, as a team sets it at a venue; the
        # fair comparison since 2026-10-20: nobody is told the floor)
        ref = make_teacher(rls, env, grip=str(sp.get("grip", "true")), **dict(limits or {}, **sp.get("teacher", {})))
        planner = ref
        if kind != "raceline":
            from ..opponent_slots import build_teacher, kind_of
            pk = dict(sp.get("planner", {}))
            if kind in ("forzaeth", "forzaeth_pred"):
                # The ego baseline runs upstream's spline planner, not the simplified one our traffic
                # runs (`SplinerTeacher.upstream`), with upstream's own prediction time for `forzaeth`.
                from ..spliner_teacher import UPSTREAM_PRED_TIME
                pk.setdefault("upstream", True)
                if kind == "forzaeth":
                    pk.setdefault("fixed_pred_time", UPSTREAM_PRED_TIME)
            planner = build_teacher(kind_of(kind), ref, env, **pk)
        stack = None
        if estimated:
            import torch as _torch
            from ..modular_stack import ModularStack
            from ..localization import PFParams
            from ..perception import DetectParams, TrackParams
            stack = ModularStack(env, _torch.nonzero(env.slot == 0).flatten(), ref,
                                 pf=PFParams(**sp.get("pf", {})), det=DetectParams(**sp.get("det", {})),
                                 trk=TrackParams(**sp.get("trk", {})))
            if planner is not ref:
                planner.observer = stack
        # The path follower. MAP, ForzaETH's own (`map_controller`), since 2026-10-19 (the user: "waypoint
        # 추종하는거 pure pursuit이여? 에바야 레이싱에 무슨.. 그 MAP 제어기 ... 그걸로 해봐"); "pp" keeps the
        # reference teacher's pure pursuit, which is what these baselines drove on before.
        tracker = None
        if sp.get("tracker", "map") == "map":
            tracker = build_map_tracker(env, ref, sp.get("map", {}))
        elif sp.get("tracker") != "pp":
            raise ValueError(f"tracker must be 'map' or 'pp', not {sp.get('tracker')!r}")
        drv = PlannerEgoDriver(planner, kind, stack, ref=ref, tracker=tracker)
        drv.speed_scale = float(sp.get("speed_scale", 1.0))
        return drv
    if kind in EGO_EXTERNAL_KINDS:
        from . import baselines
        opts = dict(options or {})
        if kind == "end2race":
            opts.setdefault("repo", _os.path.join(_os.path.dirname(__file__), "..", "..", "..",
                                                  "external", "baselines", "End2Race"))
            opts.setdefault("device", "cpu")
        drv = baselines.load(kind, weights or "none", **opts)
        lid = env.cfg.lidar
        drv.bind_scanner(n_beams=int(lid.n_beams), fov=float(lid.fov), range_max=float(lid.range_max))
        return drv
    raise ValueError(f"unknown ego driver {kind!r}: policy, {', '.join(EGO_PLANNER_KINDS + EGO_EXTERNAL_KINDS)}")
