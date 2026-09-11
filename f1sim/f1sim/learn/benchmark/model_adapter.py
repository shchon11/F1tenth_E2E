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
def load_actor(entry: Dict[str, Any], device):
    """One roster entry -> `(model, extra)`, strictly, with the arm declared rather than guessed.

    `entry` needs `path`; `arm` is the runtime arm the benchmark will install, and
    `cross_runtime` marks the deliberate case where they differ.

    Strict loading is the point: the tolerant default lets a checkpoint migrate silently, and
    "the migration reinitialised a head" and "the arms differ" produce the same-looking results.
    """
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


def policy_for(model) -> Callable:
    """`obs -> action`, deterministic, with no conditioning and no privileged input invented.

    The benchmark scores what the policy does from its own sensors. If a checkpoint needs a
    conditioning vector, `load_checkpoint` has already refused it; nothing here fabricates one.
    """
    import torch
    from f1sim.learn.obs import flatten_obs

    def policy(obs):
        scan, proprio = flatten_obs(obs)
        with torch.no_grad():
            return model.act(scan, proprio, deterministic=True)[0]

    return policy


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


def assert_env_matches_spec(env, spec: dict) -> dict:
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
                 spawn_s_m: Optional[float] = None) -> PreparedCell:
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
    arm = str(entry.get("arm") or "legacy")
    # `envs` in a cell is the number of LEARNERS. A race of M cars needs learners x M simulator
    # envs, because `race_size` slots every car into the same batch: passing 8 for a race of 2 gives
    # 4 scored cars, not 8, and the row's denominator would be half what the suite declared.
    learners = int(cell.get("envs", suite.get("envs", 8)))
    speed_cap = float(suite.get("speed_cap", extra.get("cap", 9.0)))
    sensor_noise = bool(suite.get("sensor_noise", True))
    race_size = int(cell.get("race_size", suite.get("race_size", 1)))
    envs = learners * race_size
    # The opponent must not be the candidate. `EnvConfig.opponent` defaults to "policy" (self-play),
    # which would drive the other car with the checkpoint under test and make the opponent change
    # whenever the candidate does -- the one thing an overtake row cannot allow. Races therefore
    # declare the teacher path and its speed range explicitly.
    opponent = str(suite.get("opponent", "teacher")) if race_size > 1 else "policy"
    opp_speed_range = tuple(suite.get("opp_speed_range", (0.6, 0.8)))

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

    ecfg = EnvConfig(speed_cap=speed_cap, resample_track_on_reset=True, action_mode="plan",
                     race_size=race_size, max_steps=steps,
                     opponent=opponent, opp_speed_range=opp_speed_range,
                     selfplay_front_cap=False,
                     scan_stack=int(spec["scan_stack"]), scan_stride=int(spec["scan_stride"]),
                     hist_len=int(spec["hist_len"]), hist_stride=int(spec["hist_stride"]),
                     action_history=int(spec["action_history"]),
                     imu_gyro_scale=float(spec["gyro_scale"]),
                     imu_accel_scale=float(spec["accel_scale"]),
                     v_max_policy=float(spec["v_max"]),
                     compile_tracker=bool(suite.get("compile_tracker", False)))
    if race_size > 1 and opponent == "teacher" and rls is None:
        raise AdapterError("a teacher-opponent race needs racelines; pass racelines= or names")
    env = common.make_env(trs, envs, device, ecfg, cfg=cfg, seed=int(cell["seed"]),
                          rls=rls if (race_size > 1 and opponent in ("teacher", "mixed")) else None)
    spec_match = assert_env_matches_spec(env, spec)

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
    if arm != "legacy":
        from f1sim.learn import grip_runtime as grip_rt
        t0, phase0 = float(env.sim.t), int(getattr(env.sim, "_imu_phase", 0))
        try:
            controller = grip_rt.ControllerRuntime(
                env, arm,
                estimator_path=(entry.get("estimator_path") if arm == "estimated" else None),
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
        "cross_runtime": bool(entry.get("cross_runtime")),
        "map": cell["map"], "true_mu": float(cell["true_mu"]), "seed": int(cell["seed"]),
        "learners": learners, "envs": envs, "race_size": race_size,
        "opponent": opponent, "opp_speed_range": list(opp_speed_range),
        "speed_cap": speed_cap,
        "sensor_noise": sensor_noise,
        "sensor_noise_fields": {f"{g}_{f}": float(getattr(getattr(cfg, g), f))
                                for g, f in NOISE_FIELDS},
        "budget_steps": steps,
        "randomisation_enabled": bool(cfg.rand.enabled),
        "cfg_vehicle_mu": float(cfg.vehicle.mu),
        "plant_mu": float(env.sim.P["mu"].min()), "plant_mu_max": float(env.sim.P["mu"].max()),
        "sim_compile": bool(cfg.sim.compile), "compile_tracker": bool(ecfg.compile_tracker),
        "spec": spec, "spec_match": spec_match,
        "spawn_s_m": spawn_s_m,
        "graphed": bool(getattr(controller, "graphed", False)) if controller else False,
        "estimator_path": (entry.get("estimator_path") if arm == "estimated" else None),
    }
    return PreparedCell(env=env, controller=controller, graph_holder=holder, protocol=protocol)
