"""Checkpoint and runtime adapter: everything between a roster entry and a driveable cell.

Core owns the geometry, the recorder, the loop, the CLI and the report. This module owns the part
that has to agree with how a checkpoint was *trained*: loading it under the right guards, building
an environment that emits exactly the observation it expects, and installing the controller runtime
it was trained against.

The contracts here are ported from the SGR / 48-cell evaluators, which an independent review already
put through three rounds of defect-finding. **Ported, not imported**: everything reaches the
simulator through `f1sim.*` so the benchmark stays a portable repository package, and nothing under
`work/learning-next/...` is on the import path. Those helpers are frozen and pinned to their own
experiments; a shared import would make this run depend on their freeze.

Four things this exists to get right, each of which has burned someone already:

* **The observation is the checkpoint's, not the default.** `extra["spec"]` carries the beam count,
  the scan stack, the history depth and the normalisation scales. An env built on defaults loads
  fine and then feeds the actor a differently-shaped world.
* **Friction is declared, constant, and the only thing randomised.** `cfg.rand.enabled = False` with
  `cfg.vehicle.mu` pinned means the episode's friction is the number the suite asked for, and the
  test proves the plant agrees rather than trusting the config.
* **Order: graphs, then controller, then routing.** `prepare_graph_runtime` captures `mpc.solve` and
  runs real warm-up steps; a controller hook installed before it is silently overwritten and the
  cell becomes a legacy run wearing another arm's name. Nothing may step between the install and the
  seeded reset.
* **A cross-arm reference is declared, never inferred.** The frozen legacy original evaluated under
  the estimated controller is a deliberate reference; the same mismatch arising by accident is a
  defect. The entry says which it is.

**External systems.** A roster entry may carry `kind: "tinylidarnet" | "end2race"` and `weights:`
instead of an f1sim checkpoint. Those are the published baselines (`f1sim.learn.baselines`): they
emit a steering angle and a speed and nothing else, so they run in `direct` action mode with **no
plan tracker and no controller arm at all** -- `arm` must be the string `"none"`, and an external
entry that names any real arm is refused rather than quietly given one. Their preprocessing is the
driver's, which is the same object `f1sim_ros.baseline_node` drives on the car; the parity test
between the two is therefore about the plumbing, not about two transcriptions of a paper.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

#: Fields of a saved ObsSpec and where each one has to be applied for the env to emit it.
#: An unknown field is refused rather than ignored: silently dropping a normalisation scale is how
#: an actor ends up reading a differently-scaled world and scoring badly for no visible reason.
SPEC_ROUTING = {
    "n_beams": "cfg.lidar.n_beams", "range_max": "cfg.lidar.range_max",
    "scan_stack": "EnvConfig.scan_stack", "scan_stride": "EnvConfig.scan_stride",
    "hist_len": "EnvConfig.hist_len", "hist_stride": "EnvConfig.hist_stride",
    "action_history": "EnvConfig.action_history", "v_max": "EnvConfig.v_max_policy",
    "gyro_scale": "EnvConfig.imu_gyro_scale", "accel_scale": "EnvConfig.imu_accel_scale",
    "act_dim": "checked against the model, not applied to the env",
    "att_scale": "fixed contract constant, asserted equal",
    # The privileged opponent block (`gym_env.OPP_TOKEN_MODES`). Listed so that a checkpoint
    # carrying it reaches a refusal with a reason rather than the generic "cannot honour" -- and
    # NOT routed to anything: a benchmark env never builds an oracle input, because a number
    # produced with the other cars' true future in the observation is not a benchmark result.
    "opp_token": "REFUSED: a simulator oracle is never a benchmark observation",
    "opp_future_model": "records which prediction the oracle block used; not an env setting here",
}


#: Every stochastic sensor field, by (config group, field). `sensor_noise=False` has to zero all of
#: them or the label is false: gaussian range noise is only part of it, and beam dropout, IMU bias
#: random-walk and the speed-dependent vibration model are all draws too. Two earlier versions of
#: this were wrong in instructive ways -- the first set `cfg.lidar.noise`, which does not exist, so
#: `Config` being a plain dataclass tree silently accepted an unread attribute; the second zeroed
#: only the gaussian terms and still reported "noise off".
#:
#: Bias OFFSETS are excluded deliberately: `gyro_bias_*` and `accel_bias_*` default to 0.0 and are
#: fixed offsets, not draws. `vib_onset_v` is a threshold, not an amplitude, and zeroing it would
#: turn vibration ON everywhere.
NOISE_FIELDS = (
    ("lidar", "noise_std"), ("lidar", "noise_std_rel"),
    ("lidar", "dropout_prob"), ("lidar", "floor_dropout"), ("lidar", "duct_graze_dropout"),
    ("imu", "gyro_noise"), ("imu", "accel_noise"), ("imu", "gyro_bias_walk"),
    ("imu", "vib_accel"), ("imu", "vib_gyro"), ("imu", "vib_broadband"),
    ("imu", "vib_accel_floor"), ("imu", "vib_gyro_floor"),
    ("odom", "speed_noise_std"), ("odom", "yaw_rate_noise_std"),
)


class AdapterError(RuntimeError):
    """A refusal this adapter is certain about. Never raised to mean 'probably fine'."""


# ------------------------------------------------------------------ loading
def _arm_base(arm: str) -> str:
    """The tracker arm underneath a possibly-composed name: `"fixed_low+tcs"` -> `"fixed_low"`.

    `"none"` -- an external published baseline -- returns itself: it names the absence of a plan
    tracker, so there is no base arm to find and `split_arm` would (correctly) refuse it.
    """
    from f1sim.learn.grip_runtime import split_arm
    a = str(arm or "legacy")
    if a == EXTERNAL_ARM:
        return EXTERNAL_ARM
    return split_arm(a)[0]


def load_actor(entry: Dict[str, Any], device):
    """One roster entry -> `(model, extra)`, strictly, with the arm declared rather than guessed.

    `entry` needs `path`; `arm` is the runtime arm the benchmark will install, and
    `cross_runtime` marks the deliberate case where they differ.

    Strict loading is the point: the tolerant default lets a checkpoint migrate silently, and
    "the migration reinitialised a head" and "the arms differ" produce the same-looking results.
    """
    if is_external(entry):
        return load_external(entry, device)

    from f1sim.learn.model import controller_arm_of, load_checkpoint
    import torch

    path = entry["path"]
    if not os.path.exists(path):
        raise AdapterError(f"checkpoint does not exist: {path}")
    trained_arm = controller_arm_of(torch.load(path, map_location="cpu", weights_only=False))
    eval_arm = str(entry.get("arm") or "legacy")
    cross = bool(entry.get("cross_runtime"))

    if trained_arm != "legacy" and trained_arm != eval_arm:
        raise AdapterError(
            f"{os.path.basename(path)} was trained under {trained_arm!r} but the benchmark would "
            f"install {eval_arm!r}. A policy is only comparable under the controller it trained "
            f"against; declare a different arm or a different entry.")
    if trained_arm == "legacy" and eval_arm != "legacy" and not cross:
        raise AdapterError(
            f"{os.path.basename(path)} is legacy-trained and would run under {eval_arm!r}. That is "
            f"a cross-runtime reference and has to say so: set cross_runtime=true on the entry.")

    model, extra = load_checkpoint(path, device, allow_controller=(trained_arm != "legacy"),
                                   strict_names=True)
    model.eval()
    if not (extra or {}).get("spec"):
        raise AdapterError(f"{os.path.basename(path)} carries no extra['spec']; the observation "
                           f"layout it was trained with is unknown and cannot be rebuilt")
    return model, extra


#: The arm an external baseline runs under. Not a `grip_runtime` arm and deliberately not spelled
#: `legacy` either: `legacy` means "our plan tracker with nothing installed on it", and these models
#: never reach a plan tracker at all.
EXTERNAL_ARM = "none"


def is_external(entry: Dict[str, Any]) -> bool:
    return bool((entry or {}).get("kind"))


def external_spec(driver, *, v_max: float = 10.0) -> Dict[str, Any]:
    """The ObsSpec an external baseline's cell is built on.

    Minimal on purpose. These models read one scan (and, for End2Race, the measured speed), so the
    stack is 1 deep, there is no action history worth keeping and no proprio history at all: every
    extra channel is simulator work that nothing reads. What is NOT free to choose:

    * `n_beams` and `range_max` are **this car's scanner**, not the driver's. Root's decision
      (2026-09-15): the zero-shot rows feed End2Race our 270 deg scan mapped onto the 270 of its 360
      bearings that a 270 deg scanner covers, with the other 90 filled -- they do not rebuild the
      simulator's LiDAR as a 360 deg one. `driver.bind_scanner` is what performs that mapping, and
      it is the same call `baseline_node` makes from a `LaserScan` header.
    * `v_max` decides the action encoding. `gym_env.step` maps a normalised action to
      `(a + 1) / 2 * v_max_policy`, so `v_max` has to be at least as large as any speed the model
      commands or the encoding would saturate before the suite's own cap did. 10 m/s covers both
      baselines' output ranges (TinyLidarNet tops out at 8 m/s by construction).
    """
    from f1sim.params import Config

    lid = Config().lidar
    return {"n_beams": int(lid.n_beams), "scan_stack": 1, "scan_stride": 1, "action_history": 1,
            "act_dim": 2, "hist_len": 0, "hist_stride": 2, "range_max": float(lid.range_max),
            "v_max": float(v_max), "gyro_scale": 5.0, "accel_scale": 10.0, "att_scale": 0.35}


def load_external(entry: Dict[str, Any], device=None):
    """One external roster entry -> `(driver, extra)`, in the shape the rest of this module expects.

    `device` is accepted and ignored for ONNX (CPU execution provider, single thread -- see
    `baselines/backends.py`) and passed through for torch. The simulator still runs wherever the
    caller put it; a 220 k CNN and an 11 M GRU at batch 8 are noise next to one simulator step, and
    a CPU session is deterministic across batch widths, which is what makes the node/adapter parity
    claim exact rather than approximate.
    """
    from f1sim.learn import baselines

    arm = str(entry.get("arm") or EXTERNAL_ARM)
    if arm != EXTERNAL_ARM:
        raise AdapterError(
            f"external system {entry.get('system_id') or entry.get('kind')} declares arm {arm!r}. "
            f"These models publish a steering angle and a speed directly; there is no plan for a "
            f"tracker to follow and no solver for an arm to wrap, so the only honest declaration is "
            f"arm={EXTERNAL_ARM!r}.")
    weights = entry.get("weights") or entry.get("path")
    if not weights:
        raise AdapterError("an external entry needs `weights` (or `path`)")
    opts = dict(entry.get("options") or {})
    if str(entry["kind"]).lower() == "end2race":
        opts.setdefault("device", str(device or "cpu"))
    driver = baselines.load(entry["kind"], weights, **opts)
    spec = external_spec(driver, v_max=float(opts.get("v_max", 10.0)))
    driver.bind_scanner(n_beams=spec["n_beams"], fov=_lidar_fov(), range_max=spec["range_max"])
    extra = {"spec": spec, "cap": None, "external": driver.describe(),
             "experiment": {"controller": {"arm": EXTERNAL_ARM}}}
    return driver, extra


def _lidar_fov() -> float:
    from f1sim.params import Config
    return float(Config().lidar.fov)


def external_policy(driver, spec: Dict[str, Any], s_max: float) -> Callable:
    """`obs -> normalised action`, wrapping the same driver `baseline_node` runs.

    Three conversions, and each is the inverse of something the environment does:

    * `obs["scan"][:, 0]` is `range / range_max` clamped to [0, 1] (`gym_env._norm_scan`), so metres
      come back by multiplying. Index 0 is the newest frame (`_step_math` prepends).
    * `obs["speed"]` is `odom speed / v_max_policy` (`gym_env._obs`), the same VESC-modelled wheel
      speed `/odom` carries on the car -- not a privileged ground-truth speed.
    * the command goes back through `gym_env.step`'s `direct` mapping: `steer = a0 * s_max` and
      `speed = min((a1 + 1) / 2 * v_max, speed_cap)`, so `a0 = steer / s_max` and
      `a1 = 2 * speed / v_max - 1`. `step` clamps the action to [-1, 1] itself, which is exactly the
      plant refusing a steering angle past its stop and refusing to drive backwards -- the same two
      clips `baseline_node` applies before publishing, for the same reason.

    The returned callable carries `.reset(done=None)`, so `runner.run_cell` clears the driver's
    memory at the cell's seeded reset and per row at every episode boundary, exactly as it does for
    one of our own recurrent checkpoints.
    """
    import numpy as np
    import torch

    range_max = float(spec["range_max"])
    v_max = float(spec["v_max"])
    s_max = float(s_max)

    def policy(obs):
        scan = obs["scan"]
        ranges = (scan[:, 0] * range_max).detach().cpu().numpy().astype(np.float32)
        speed = None
        if driver.needs_speed:
            speed = (obs["speed"][:, 0] * v_max).detach().cpu().numpy().astype(np.float32)
        cmd = driver.command(driver.adapt(ranges), speed)
        a = np.stack([cmd[:, 0] / s_max, 2.0 * cmd[:, 1] / v_max - 1.0], 1)
        return torch.as_tensor(a, dtype=scan.dtype, device=scan.device)

    policy.reset = driver.reset
    policy.driver = driver
    return policy


def policy_for(model, batch: int = None) -> Callable:
    """`obs -> action`, deterministic, with no conditioning and no privileged input invented.

    The benchmark scores what the policy does from its own sensors. If a checkpoint needs a
    conditioning vector, `load_checkpoint` has already refused it; nothing here fabricates one.

    A recurrent checkpoint carries its hidden state (and any extra scan channel) between calls, so
    the returned callable is stateful and has `.reset(done=None)`. `runner.run_cell` clears it at
    the cell's single seeded reset and at every episode boundary within the cell -- i.e. per trial,
    which is the unit the suite scores. The width is taken from the first observation unless
    `batch` says otherwise, and a cell of a different width rebuilds (and so clears) the state,
    because a hidden state is per row and cannot be carried from one cell's rows to another's.

    For a feedforward checkpoint this is `model.act` and `.reset` does nothing, so the scores of
    every system already on the roster are untouched.
    """
    from f1sim.learn.baselines import BaselineDriver
    from f1sim.learn.memory import policy_fn

    if isinstance(model, BaselineDriver):
        raise AdapterError(
            "an external baseline needs `external_policy(driver, spec, s_max)`: its action encoding "
            "depends on the cell's own v_max and the vehicle's steering limit, which a bare "
            "`policy_for(model)` does not know.")
    return policy_fn(model, batch, device=next(model.parameters()).device, deterministic=True)


# ------------------------------------------------------------------ environment
def eval_config(true_mu: float, spec: dict, *, sensor_noise: bool, compile_sim: bool = False):
    """Nominal everything, randomisation OFF, friction pinned to the declared value.

    Randomisation off is what makes `true_mu` true: with it on, `vehicle.mu` is a scale applied to a
    draw and the episode's friction is not the number the suite asked for. Both compile switches are
    pinned rather than inherited so the backend does not depend on that day's defaults.
    """
    from f1sim.params import Config
    cfg = Config()
    cfg.rand.enabled = False
    cfg.vehicle.mu = float(true_mu)
    cfg.sim.compile = bool(compile_sim)
    cfg.sim.compile_mode = "default" if compile_sim else "none"
    cfg.lidar.n_beams = int(spec["n_beams"])
    cfg.lidar.range_max = float(spec["range_max"])
    if not sensor_noise:
        for grp, field in NOISE_FIELDS:
            setattr(getattr(cfg, grp), field, 0.0)
    return cfg


def assert_env_matches_spec(env, spec: dict, allow_oracle: bool = False) -> dict:
    """What the env WILL emit, read back after construction, against what the actor expects.

    Read back rather than assumed: the EnvConfig fields are an intention and `common.obs_spec(env)`
    is what the built object actually produces.
    """
    from f1sim.learn import common
    from f1sim.learn.obs import ObsSpec

    unsupported = sorted(set(spec) - set(SPEC_ROUTING))
    if unsupported:
        raise AdapterError(f"checkpoint records observation scalars this adapter cannot honour: "
                           f"{unsupported}")
    if spec.get("opp_token") and not allow_oracle:
        raise AdapterError(
            f"checkpoint was trained with the privileged opponent block "
            f"(opp_token={spec['opp_token']!r}): the other cars' true relative position, velocity "
            f"and future, read out of the simulator. A benchmark env does not build one, and a "
            f"score obtained with it would not be comparable with any row in the suite. A caller "
            f"that is deliberately measuring an oracle arm -- and that will label what it gets as "
            f"one -- passes allow_oracle=True; `benchmark run` never does.")
    want = ObsSpec(**{k: v for k, v in spec.items() if k in ObsSpec.__dataclass_fields__})
    got = common.obs_spec(env)
    diffs = {}
    for f in ObsSpec.__dataclass_fields__:
        a, b = getattr(want, f), getattr(got, f)
        same = (abs(float(a) - float(b)) <= 1e-9 if isinstance(a, float) or isinstance(b, float)
                else a == b)
        if not same:
            diffs[f] = [a, b]
    if diffs:
        raise AdapterError(f"built env does not match the actor's observation spec "
                           f"(checkpoint -> env): {diffs}")
    return {"checked": sorted(ObsSpec.__dataclass_fields__), "proprio_dim": int(got.proprio_dim)}


@dataclass
class PreparedCell:
    """One driveable cell, plus what it actually turned out to be.

    `controller` is None for the legacy arm -- there is no runtime to install and a no-op object
    would let a caller believe one ran.
    """
    env: Any
    controller: Optional[Any]
    graph_holder: Any
    protocol: Dict[str, Any] = field(default_factory=dict)
    _closed: bool = False

    def begin(self, obs):
        if self.controller is not None:
            self.controller.begin(obs)

    def pre_action(self, obs):
        if self.controller is not None:
            self.controller.pre_action(obs)

    def post_step(self, terminated, truncated):
        if self.controller is not None:
            self.controller.post_step(terminated, truncated)

    def close(self):
        """Uninstall in the reverse order of installation, and only once.

        Every stage runs even if an earlier one raises. The first version marked itself closed and
        then released in sequence, so a throwing `controller.release()` left the graph runtime
        installed and the env referenced -- and `_closed` was already True, so the retry that would
        have cleaned up returned immediately. Failures are collected and the first is re-raised
        after everything has been attempted.
        """
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            if self.controller is not None:
                try:
                    self.controller.release()
                except Exception as exc:                      # noqa: BLE001 - reported below
                    errors.append(exc)
                finally:
                    self.controller = None
        finally:
            holder, self.graph_holder = self.graph_holder, None
            rel = getattr(holder, "release", None)
            if callable(rel):
                try:
                    rel()
                except Exception as exc:                      # noqa: BLE001 - reported below
                    errors.append(exc)
            self.env = None
        if errors:
            raise errors[0]


def prepare_cell(entry: Dict[str, Any], extra: Dict[str, Any], cell: Dict[str, Any],
                 suite: Dict[str, Any], device, *, tracks_override=None, racelines=None,
                 spawn_s_m: Optional[float] = None, allow_oracle: bool = False) -> PreparedCell:
    """Build the env this checkpoint expects, capture graphs, install the arm. Nothing steps after.

    Order is the contract, not a preference:

    1. build the env on the checkpoint's own ObsSpec and the cell's declared friction;
    2. `warmup()` then `prepare_graph_runtime` -- real steps, which advance the simulator;
    3. install the controller runtime, which must not step;
    4. the caller performs the single seeded reset and then `begin(obs)`.

    Reversing 2 and 3 silently overwrites the arm's solver hook (`grip_runtime` assigns
    `tracker._solver` during capture), and the cell becomes a legacy run reported under another
    arm's name.
    """
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    from f1sim.learn.evaluate import budget_steps
    from f1sim.learn.graph_runtime import prepare_graph_runtime

    spec = dict(extra["spec"])
    external = is_external(entry)
    arm = str(entry.get("arm") or (EXTERNAL_ARM if external else "legacy"))
    if external and arm != EXTERNAL_ARM:
        raise AdapterError(f"external system declares arm {arm!r}; only {EXTERNAL_ARM!r} is possible "
                           f"for a model that publishes a command instead of a plan")
    if not external and arm == EXTERNAL_ARM:
        raise AdapterError(f"arm {EXTERNAL_ARM!r} names the absence of a plan tracker, which only an "
                           f"external baseline can declare; this entry has no `kind`")
    # `envs` in a cell is the number of LEARNERS. A race of M cars needs learners x M simulator
    # envs, because `race_size` slots every car into the same batch: passing 8 for a race of 2 gives
    # 4 scored cars, not 8, and the row's denominator would be half what the suite declared.
    learners = int(cell.get("envs", suite.get("envs", 8)))
    speed_cap = float(suite.get("speed_cap") or extra.get("cap") or 9.0)
    sensor_noise = bool(suite.get("sensor_noise", True))
    race_size = int(cell.get("race_size", suite.get("race_size", 1)))
    envs = learners * race_size
    # The opponent must not be the candidate. `EnvConfig.opponent` defaults to "policy" (self-play),
    # which would drive the other car with the checkpoint under test and make the opponent change
    # whenever the candidate does -- the one thing an overtake row cannot allow. Races therefore
    # declare the teacher path and its speed range explicitly.
    opponent = str(suite.get("opponent", "teacher")) if race_size > 1 else "policy"
    opp_speed_range = tuple(suite.get("opp_speed_range", (0.6, 0.8)))
    # Scripted opponent behaviour (f1sim.opponent_events). Part of the scenario, so it arrives with
    # the suite dict and is recorded in `protocol`; a run whose opponent brakes and one whose
    # opponent does not are different cells, and a row has to be able to say which it was.
    from f1sim.opponent_events import parse_events
    opp_events = parse_events(suite.get("opp_events") or ())
    opp_event_rate = float(suite.get("opp_event_rate", 0.0) or 0.0)
    if opp_events and not (race_size > 1 and opponent in ("teacher", "mixed")):
        raise AdapterError(
            f"opponent events {list(opp_events)} were declared for a cell with race_size "
            f"{race_size} and opponent {opponent!r}: the events script the teacher-driven cars of a "
            f"race, and there are none here, so the cell would silently be the unflagged one.")
    if opp_events and opp_event_rate <= 0.0:
        raise AdapterError(
            f"opponent events {list(opp_events)} at rate {opp_event_rate}: at a non-positive rate "
            f"nothing ever fires and the cell is silently the unflagged one.")

    cfg = eval_config(float(cell["true_mu"]), spec, sensor_noise=sensor_noise)
    # `tracks_override` carries already-built `Track` objects when core has placed an obstacle on
    # one -- re-loading it by name would throw the placement away and score a clean track. Only a
    # list of names goes through `load_tracks`; built tracks are used as they are.
    if tracks_override:
        if all(isinstance(t, str) for t in tracks_override):
            trs, rls = common.load_tracks(list(tracks_override), racelines=racelines is not False,
                                          drop_infeasible=False)
        else:
            trs = list(tracks_override)
            rls = racelines if isinstance(racelines, (list, tuple)) else None
    else:
        trs, rls = common.load_tracks([cell["map"]], racelines=racelines is not False,
                                      drop_infeasible=False)
    step_dt = 1.0 / cfg.sim.control_rate
    steps = int(budget_steps(trs, speed_cap, step_dt, float(suite.get("budget_laps", 3)), 24000))

    # `direct` for an external baseline: (steer, speed) straight into the plant, no PlanTracker
    # built at all. `plan` for one of ours.
    ecfg = EnvConfig(speed_cap=speed_cap, resample_track_on_reset=True,
                     action_mode=("direct" if external else "plan"),
                     race_size=race_size, max_steps=steps,
                     opponent=opponent, opp_speed_range=opp_speed_range,
                     opp_events=opp_events, opp_event_rate=opp_event_rate,
                     selfplay_front_cap=False,
                     scan_stack=int(spec["scan_stack"]), scan_stride=int(spec["scan_stride"]),
                     hist_len=int(spec["hist_len"]), hist_stride=int(spec["hist_stride"]),
                     action_history=int(spec["action_history"]),
                     imu_gyro_scale=float(spec["gyro_scale"]),
                     imu_accel_scale=float(spec["accel_scale"]),
                     v_max_policy=float(spec["v_max"]),
                     # Only ever non-empty under `allow_oracle`, and then the caller has already
                     # accepted that what it is about to measure is not a suite row.
                     opp_token=(str(spec.get("opp_token") or "") if allow_oracle else ""),
                     # Whatever the checkpoint recorded; the env's own default otherwise, so a
                     # cell never silently measures a prediction nothing in the tree still uses.
                     **({"opp_future_model": str(spec["opp_future_model"])}
                        if spec.get("opp_future_model") else {}),
                     compile_tracker=bool(suite.get("compile_tracker", False)))
    if race_size > 1 and opponent == "teacher" and rls is None:
        raise AdapterError("a teacher-opponent race needs racelines; pass racelines= or names")
    env = common.make_env(trs, envs, device, ecfg, cfg=cfg, seed=int(cell["seed"]),
                          rls=rls if (race_size > 1 and opponent in ("teacher", "mixed")) else None)
    spec_match = assert_env_matches_spec(env, spec, allow_oracle=allow_oracle)

    if spawn_s_m is not None:
        # The avoidance scenario's fixed start arc.
        #
        # There is no EnvConfig field for this: for a solo cell `_reset_envs` passes `s=None` and
        # the simulator draws a random arc (`gym_env.py:381, 421`). But `sim.sample_spawn` already
        # takes `s=`, so the existing interface is supplied rather than bypassed -- the wrapper
        # fills in the arc the caller left unset and changes nothing else. Applied BEFORE the seeded
        # reset so the stacked scan and the action history are built from that pose; teleporting
        # after a reset would leave the observation describing the spawn point instead.
        _sample = env.sim.sample_spawn

        def _spawn_at(n, *a, s=None, **kw):
            import torch
            if s is None:
                s = torch.full((n,), float(spawn_s_m), device=env.sim.device)
            return _sample(n, *a, s=s, **kw)

        env.sim.sample_spawn = _spawn_at

    env.sim.warmup()
    holder = prepare_graph_runtime(env, log=lambda _t: None)

    controller = None
    if arm not in ("legacy", EXTERNAL_ARM):
        from f1sim.learn import grip_runtime as grip_rt
        t0, phase0 = float(env.sim.t), int(getattr(env.sim, "_imu_phase", 0))
        try:
            controller = grip_rt.ControllerRuntime(
                env, arm,
                estimator_path=(entry.get("estimator_path")
                                if _arm_base(arm) == "estimated" else None),
                device=device)
            controller.install(graph_rt=holder)
            if (float(env.sim.t), int(getattr(env.sim, "_imu_phase", 0))) != (t0, phase0):
                raise AdapterError("installing the controller stepped the simulator; cells would "
                                   "start from different sensor phases")
        except BaseException:
            # `prepare_graph_runtime` already installed the graph runtime, and this function owns it
            # until a PreparedCell exists to hand it to. Raising here without releasing leaves it
            # hooked into a simulator nobody holds a reference to, and the caller has no handle to
            # clean up with. Cleanup failures are suppressed so the ORIGINAL exception is what the
            # caller sees -- a release error while handling a refusal is not the interesting one.
            for release in (getattr(controller, "release", None),
                            getattr(holder, "release", None)):
                if callable(release):
                    try:
                        release()
                    except Exception:
                        pass
            raise

    protocol = {
        "arm": arm, "trained_arm": str((extra.get("experiment") or {}).get("controller", {})
                                       .get("arm") or "legacy"),
        "external": extra.get("external"),
        "action_mode": str(ecfg.action_mode),
        # Where it ran. CPU and CUDA do not produce bit-identical float arithmetic, and a rollout is
        # chaotic enough for that to change an outcome, so two rows are only comparable when this
        # agrees -- the same reason `source_digest` is on every row.
        "device": str(getattr(env.sim, "device", device)),
        "cross_runtime": bool(entry.get("cross_runtime")),
        "map": cell["map"], "true_mu": float(cell["true_mu"]), "seed": int(cell["seed"]),
        "learners": learners, "envs": envs, "race_size": race_size,
        "opponent": opponent, "opp_speed_range": list(opp_speed_range),
        "opp_events": list(opp_events), "opp_event_rate": opp_event_rate,
        "speed_cap": speed_cap,
        "sensor_noise": sensor_noise,
        "sensor_noise_fields": {f"{g}_{f}": float(getattr(getattr(cfg, g), f))
                                for g, f in NOISE_FIELDS},
        "budget_steps": steps,
        "randomisation_enabled": bool(cfg.rand.enabled),
        "cfg_vehicle_mu": float(cfg.vehicle.mu),
        "plant_mu": float(env.sim.P["mu"].min()), "plant_mu_max": float(env.sim.P["mu"].max()),
        "sim_compile": bool(cfg.sim.compile), "compile_tracker": bool(ecfg.compile_tracker),
        # Present and true only for a deliberate oracle measurement; a row carrying it is not a
        # suite row and can never be pooled with one.
        "oracle_opp_token": str(ecfg.opp_token or "") or None,
        "spec": spec, "spec_match": spec_match,
        "spawn_s_m": spawn_s_m,
        "graphed": bool(getattr(controller, "graphed", False)) if controller else False,
        "estimator_path": (entry.get("estimator_path")
                           if _arm_base(arm) == "estimated" else None),
        "wheel_model": bool(getattr(env.sim, "wheel_model", False)),
    }
    return PreparedCell(env=env, controller=controller, graph_holder=holder, protocol=protocol)
