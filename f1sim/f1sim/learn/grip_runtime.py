"""Controller runtime for the experiment arms. Opt-in; `legacy` changes nothing.

    legacy      nothing installed -- `mpc.solve` exactly as it was
    fixed_low   mu = 0.73423 everywhere (the control arm)
    oracle      mu = the episode's true friction, per env (lab only)
    estimated   mu = the frozen estimator's filtered lower quantile, from causal sensors only

and, composable on top of any of them and on each other (`tcs`, `fixed_low+clearance`,
`fixed_low+clearance+tcs`, ...):

    +clearance  a local occupancy built from the current LiDAR frame alone, and the policy's plan
                bent and slowed until every point of it keeps a stated body-edge margin from
                anything the scan saw. See `clearance.py`. Orthogonal to the four above: those
                decide what friction the tracker plans under, this decides what geometry it is
                handed. It sits on `tracker._plan_hook` while they sit on `tracker._solver`, so
                the two cannot overwrite each other and the installation order cannot change the
                command.
    +tcs        the car's own `TractionGuard` shaping the speed command between the controller and
                the VESC, fed from the simulated ERPM odometry and IMU. See `traction_arm.py`.
                Orthogonal to the four above: those decide what friction the *tracker* plans under,
                this decides what happens when a wheel lets go anyway.

The suffixes are written in the order the car meets them -- plan, tracker, wheels -- so the
deployment composite reads `fixed_low+clearance+tcs`.

The actor is unchanged in all of them: proprio 366 in, 8D plan out. Only the controller between the
plan and the wheels differs, which is what makes the comparison about knowing the friction.

Three things here are timing, not modelling, and all three are easy to get silently wrong:

1. **The issued command must be read before auto-reset overwrites it.** `gym_env._reset_envs`
   assigns `last_cmd[ids] = 0` and then a spawn speed (`gym_env.py:431`), and it runs *inside*
   `env.step()` before it returns. Reading `env.last_cmd` afterwards therefore yields the post-reset
   value for exactly the envs whose episode just ended -- the rows where a wrong feature is most
   damaging. `IssuedCommandSpy` snapshots it on entry to `_reset_envs`.
2. **Inference is pre-action.** `history_t` holds observation *t* and issued command *t-1*, and the
   friction it predicts is handed to the MPC *before* action *t* is taken. Never command *t*.
3. **A finished env's history is cleared, not carried.** The observation `step()` returns is already
   the post-reset one for those ids, so their old rows belong to a different episode -- and, since
   `randomization.py` writes `P` in place, to a different friction.

The estimator itself (`grip_estimator.py`), including the predeclared warmup/fallback/EMA, is
backend's. **This module never smooths twice**: it calls `lower_mu` and uses what comes back.
"""
from __future__ import annotations

from collections import deque
from typing import NamedTuple, Optional

import torch

from . import grip_control as gc

#: The plan-tracker arms. `legacy` is the untouched path and installs nothing.
BASE_ARMS = ("legacy", "fixed_low", "oracle", "estimated")

#: The composable layers, in the order the car meets them: the plan reaches the tracker, the
#: tracker's command reaches the wheels. A name lists the tracker base first (omitted when it is
#: `legacy`, which installs nothing) and then the layers it wears, in this order.
LAYERS = ("clearance", "tcs")


def _compose(base: str, layers: tuple) -> str:
    parts = ([] if base == "legacy" else [base]) + list(layers)
    return "+".join(parts) or "legacy"


#: Every arm, spelled out. Two layers are *composable*: `clearance` adjusts the plan before the
#: tracker decodes it and `tcs` shapes the speed command between the tracker and the VESC, so
#: neither changes what the tracker's friction is set from and either can be worn on top of any of
#: the four bases and of each other. `tcs` on its own is the legacy tracker plus the guard;
#: `fixed_low+tcs` is the deployment default today. Spelling them out rather than parsing on the fly
#: is what lets `benchmark/roster.py` keep validating `controller_arm in ARMS` unchanged, and what
#: makes `--controller` reject `tcs+fixed_low` or `legacy+clearance` instead of quietly accepting a
#: second name for something that already has one.
ARMS = tuple(_compose(base, layers)
             for base in BASE_ARMS
             for layers in ((), ("clearance",), ("tcs",), ("clearance", "tcs")))


class ArmParts(NamedTuple):
    """`split_arm`'s answer. A tuple, so `[0]` is still the tracker base and `[1]` still `tcs`."""
    base: str
    tcs: bool
    clearance: bool


def split_arm(arm: str) -> ArmParts:
    """`"fixed_low+clearance"` -> `("fixed_low", False, True)`; `"tcs"` -> `("legacy", True, False)`."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    parts = arm.split("+")
    base = "legacy" if parts[0] in LAYERS else parts[0]
    worn = parts if parts[0] in LAYERS else parts[1:]
    return ArmParts(base, "tcs" in worn, "clearance" in worn)


#: B3's predeclared flag thresholds. Fixed before any evaluation; a flag never retunes anything.
DEGENERACY_FALLBACK_FRAC = 0.50      # warm-only fallback fraction
DEGENERACY_AT_FLOOR_FRAC = 0.95      # warm mu_used within FLOOR_EPS of mu_min
DEGENERACY_UPDATES = 5               # consecutive updates before the flag fires
OVERESTIMATE_FRAC = 0.20             # warm steps with used_mu > true mu + OVER_EPS
OVERESTIMATE_UPDATES = 5
OVERESTIMATE_MIN_WARM_STEPS = 10_000
FLOOR_EPS = 0.02
OVER_EPS = 0.02

#: `diag["fallback_reason"]` from the estimator: 0 cold, 1 non-finite, 2 none. So a fallback is
#: `!= FALLBACK_NONE` -- not `!= 0`, which would read every cold step as a normal one and invert the
#: degeneracy flag exactly where it matters.
FALLBACK_NONE = 2


def arm_to_grip_mode(arm: str) -> str:
    """The `GripSpec.mode` each arm runs. `fixed_low` is the experiment's name for `fixed`."""
    return {"fixed_low": "fixed", "oracle": "oracle", "estimated": "estimated"}[arm]


class IssuedCommandSpy:
    """Snapshots `env.last_cmd` on entry to `_reset_envs`, before auto-reset overwrites it.

    Non-invasive by design: it wraps the bound method and restores it on `release()`, rather than
    asking `gym_env.py` for a hook it does not otherwise need. (`cmd_shaper`, which the `tcs` arm
    uses, is a hook -- a command shaper has to run *between* two statements inside `step`, which is
    not something a wrapper can reach.) The snapshot is of the whole batch,
    not just the resetting ids: between the command being issued and the reset running, no other env
    has been touched, so the clone is the correct pre-reset command for every env.
    """

    def __init__(self, env):
        self.env = env
        self._orig = None
        self._snapshot: Optional[torch.Tensor] = None
        self._fired = False

    def install(self) -> "IssuedCommandSpy":
        if self._orig is not None:
            raise RuntimeError("already installed")
        self._orig = self.env._reset_envs

        def wrapped(ids):
            if ids.numel():                      # a no-op reset must not count as a boundary
                self._snapshot = self.env.last_cmd.clone()
                self._fired = True
            return self._orig(ids)

        self.env._reset_envs = wrapped
        return self

    def release(self) -> None:
        if self._orig is not None:
            self.env._reset_envs = self._orig
            self._orig = None

    def take(self) -> torch.Tensor:
        """The command actually issued this step, for every env. Call once per step, after `step()`.

        Falls back to the live `last_cmd` when no reset ran, which is then untouched and correct.
        """
        cmd = self._snapshot if self._fired else self.env.last_cmd
        self._snapshot, self._fired = None, False
        return cmd.clone()

    def discard(self) -> None:
        """Drop a pending snapshot. `env.reset()` resets every env, which fires this spy; without
        dropping it, the first `post_step` of the run would hand back that priming snapshot instead
        of the command the first real step issued."""
        self._snapshot, self._fired = None, False


#: Accumulator slots. Everything is summed on device and read back once per update: a `float(t)`
#: per field per step is a host synchronise, and at ~15 of them per step that is the dominant cost
#: of the whole controller.
_SLOTS = ("env_steps", "warm", "fallback", "warm_fallback", "q_gap", "excitation",
          "mu_used", "mu_used_sq", "at_floor", "a_brk", "a_brk_n", "over", "covered", "truth")
_IX = {k: i for i, k in enumerate(_SLOTS)}


class Accumulator:
    """B3's per-update sums, in **env-transitions** (envs x steps), never rollout steps.

    The unit matters: a rollout is 32 steps, so a threshold of 10 000 expressed in rollout steps
    would sit 300 updates away and the flag would never fire. One update at 256 envs x 32 steps is
    8192 env-transitions, so the 10 000 minimum is met across the five-update evidence window, not
    within a single update -- which is why `FlagTracker` sums the window rather than testing one.
    """

    def __init__(self, device):
        self.v = torch.zeros(len(_SLOTS), device=device, dtype=torch.float64)
        self.lo = torch.full((), float("inf"), device=device, dtype=torch.float64)
        self.hi = torch.full((), float("-inf"), device=device, dtype=torch.float64)
        self.steps = 0

    def add(self, **kw) -> None:
        for k, val in kw.items():
            self.v[_IX[k]] += val

    def add_range(self, t: torch.Tensor) -> None:
        self.lo = torch.minimum(self.lo, t.min().double())
        self.hi = torch.maximum(self.hi, t.max().double())

    def read(self) -> dict:
        """The one host synchronise of the update."""
        vals = self.v.tolist()
        d = {k: vals[i] for k, i in _IX.items()}
        d["a_brk_lo"] = float(self.lo)
        d["a_brk_hi"] = float(self.hi)
        d["steps"] = self.steps
        return d


#: Keys that only mean anything for the estimated arm. For `fixed_low`/`oracle` there is no
#: estimator, so emitting them as 0.0 would report "no excitation, never warm, zero quantile gap"
#: about a quantity that does not exist -- the same shape of false negative as a vacuously-passing
#: finiteness check. They are omitted instead.
_ESTIMATOR_ONLY = ("fallback_frac", "warm_fallback_frac", "warm_frac", "q_gap_mean",
                   "excitation_pass_frac", "mu_used_at_floor_frac", "warm_env_transitions")


def metrics_dict(r: dict, prefix="controller/", estimator: bool = True) -> dict:
    """Fractions from the raw sums. Each denominator is the population the numerator was counted on."""
    n = max(r["env_steps"], 1.0)
    w = max(r["warm"], 1.0)
    t = max(r["truth"], 1.0)
    nb = max(r["a_brk_n"], 1.0)
    mean = r["mu_used"] / n
    var = r["mu_used_sq"] / n - mean * mean
    d = {
        f"{prefix}fallback_frac": r["fallback"] / n,
        f"{prefix}warm_fallback_frac": r["warm_fallback"] / w,
        f"{prefix}warm_frac": r["warm"] / n,
        # conditional mean over warm transitions only -- sum over warm, divided by the warm count,
        # not by a fraction
        f"{prefix}q_gap_mean": r["q_gap"] / w,
        f"{prefix}excitation_pass_frac": r["excitation"] / n,
        f"{prefix}mu_used_mean": mean,
        f"{prefix}mu_used_std": max(var, 0.0) ** 0.5,
        f"{prefix}mu_used_at_floor_frac": r["at_floor"] / w,
        # the limit the solver actually imposed (minimum over the planned path), not the
        # straight-line budget -- on any curved plan those differ
        f"{prefix}a_brake_realised_mean": r["a_brk"] / nb,
        f"{prefix}a_brake_realised_spread": (r["a_brk_hi"] - r["a_brk_lo"]) if r["a_brk_n"] else 0.0,
        f"{prefix}warm_env_transitions": r["warm"],
    }
    if r["truth"]:                                 # only when a pre-action truth clone was supplied
        d[f"{prefix}overestimate_frac"] = r["over"] / t
        d[f"{prefix}coverage_q10_q90"] = r["covered"] / t
    if not estimator:
        for k in _ESTIMATOR_ONLY:
            d.pop(f"{prefix}{k}", None)
    return d


class FlagTracker:
    """B3's two predeclared flags. Counts consecutive updates; never changes anything.

    A flag is evidence for a separately versioned iteration, not a trigger to retune a running job.
    """

    def __init__(self):
        self.degeneracy_run = 0
        self.overestimate_run = 0
        self.window: deque = deque(maxlen=max(DEGENERACY_UPDATES, OVERESTIMATE_UPDATES))
        self.fired: list[str] = []

    def update(self, r: dict) -> list[str]:
        """`r` is one update's raw sums. Rates are per-update; the sample-size gate is per-window."""
        self.window.append(r)
        new = []
        w = r["warm"]
        degen = w > 0 and ((r["warm_fallback"] / w > DEGENERACY_FALLBACK_FRAC)
                           or (r["at_floor"] / w > DEGENERACY_AT_FLOOR_FRAC))
        self.degeneracy_run = self.degeneracy_run + 1 if degen else 0
        if self.degeneracy_run == DEGENERACY_UPDATES:
            new.append("degeneracy")
        over_rate = r["truth"] > 0 and r["over"] / r["truth"] > OVERESTIMATE_FRAC
        self.overestimate_run = self.overestimate_run + 1 if over_rate else 0
        # The >=10000 minimum is evidence across the five-update window, not inside one update:
        # a single update carries at most envs x horizon transitions.
        window_warm = sum(x["truth"] for x in list(self.window)[-OVERESTIMATE_UPDATES:])
        if (self.overestimate_run >= OVERESTIMATE_UPDATES
                and window_warm >= OVERESTIMATE_MIN_WARM_STEPS
                and "overestimate" not in self.fired):
            new.append("overestimate")
        self.fired.extend(new)
        return new


class ControllerRuntime:
    """Owns the arm: the sensor history, the frozen estimator, and the MPC's live friction.

    Usage, once per step, in this order -- the order is the contract:

        rt.begin(obs)                              # once, after env.reset()
        for t in ...:
            rt.pre_action(obs)                     # push history, infer, update the MPC
            action = policy(obs)
            obs, r, term, trunc, info = env.step(action)
            rt.post_step(term, trunc)              # capture the issued command, mark resets
    """

    def __init__(self, env, arm: str, estimator_path: Optional[str] = None,
                 gspec: Optional[gc.GripSpec] = None, device=None,
                 traction_params=None, clearance_spec=None):
        base, tcs, clear = split_arm(arm)
        if base == "estimated" and not estimator_path:
            raise ValueError("the estimated arm needs --estimator PATH; it has no default")
        if base != "estimated" and estimator_path:
            raise ValueError(f"--estimator is only meaningful for the estimated arm, not {arm!r}")
        if traction_params is not None and not tcs:
            raise ValueError(f"traction parameters are only meaningful for a +tcs arm, not {arm!r}")
        if clearance_spec is not None and not clear:
            raise ValueError(f"a clearance spec is only meaningful for a +clearance arm, not {arm!r}")
        self.env, self.arm = env, arm
        #: The tracker arm underneath, and which composable layers ride on top. Every branch below
        #: keys off `base`, so `fixed_low+clearance` is `fixed_low` plus a plan shaper and nothing
        #: else about the tracker changes.
        self.base, self.tcs, self.clear = base, tcs, clear
        self.traction = None
        self.traction_params = traction_params
        self.clearance = None
        self.clearance_spec = clearance_spec
        self.device = torch.device(device or env.device)
        self.B = int(env.B)
        self.estimator_path = estimator_path
        self.estimator = None
        self.net = None                            # backend's parameters live on GripEstimator.net
        self.history = None
        self.grip = None
        self.spy = None
        self.graphed = False
        self._reset_mask = None
        self._prev_cmd = None
        self._installed = False
        self.acc = Accumulator(self.device)
        self.flags = FlagTracker()
        self._pending_truth = None
        mode = "legacy" if base == "legacy" else arm_to_grip_mode(base)
        self.gspec = (gspec or gc.GripSpec(mode=mode)).validate()
        if self.gspec.mode != mode:
            raise ValueError(f"arm {arm!r} needs GripSpec(mode={mode!r}), got {self.gspec.mode!r}")

    # -- lifecycle ---------------------------------------------------------------
    def install(self, graph_rt=None, adopt: bool = True) -> "ControllerRuntime":
        """Install the MPC hook and the command spy. **After** `prepare_graph_runtime`.

        `prepare_graph_runtime` captures `mpc.solve` and assigns `tracker._solver`, so a hook
        installed before it is silently overwritten and the run would be a `legacy` run wearing
        another arm's name.

        `graph_rt` is that runtime. When it captured the plan solver, its `GraphedCallable` holds the
        eleven real arguments recorded from actual steps, and those are exactly what this solver
        needs to capture its own graph. Without them the controller would install an **eager** solver
        on top of a run that had just paid to capture a graphed one -- correct, and several times
        slower for every step of a 1,048,576-step job.
        """
        if self._installed:
            raise RuntimeError("already installed")
        if self.clear:
            # Independent of the tracker's friction, and installed before or after the grip solver
            # indifferently: this binds `tracker._plan_hook`, that binds `tracker._solver`. It has
            # to come before the first step, not before the graph capture, because the plan hook is
            # not part of anything captured -- it produces the action the captured solver is fed.
            self.clearance = self._build_clearance().install()
        if self.tcs:
            # Independent of the tracker: the guard sits on the speed command, not on the plan, so
            # it works in `--action-mode direct` as well and needs nothing captured first.
            from .traction_arm import TractionArm
            self.traction = TractionArm(self.env, self.traction_params, device=self.device).install()
        if self.base == "legacy":
            self._installed = True                 # no tracker friction to install, by definition
            return self
        tracker = self.env.tracker
        if tracker is None:
            raise RuntimeError("the controller arms need the plan tracker (--action-mode plan)")
        if self.base == "estimated":
            self._load_estimator()
        self.grip = gc.GripMPC(tracker, self.gspec, self.B, self.device,
                               tracker.wb, tracker.s_max, tracker.v_max)
        example = None
        mpc_graph = getattr(graph_rt, "mpc", None) if graph_rt is not None else None
        if mpc_graph is not None and self.device.type == "cuda":
            example = list(mpc_graph._static)      # the eleven args, recorded from real steps
        self.grip.install(graph=example is not None, example_args=example, adopt=adopt)
        self.graphed = example is not None
        self.spy = IssuedCommandSpy(self.env).install()
        self._installed = True
        return self

    def _build_clearance(self):
        """The `clearance` layer, built from this environment's own sensor geometry.

        Nominal, not per-env: `mount_x` and the beam bearings come from the configuration, not from
        the randomized `sim.P`. The car does not know its own mounting error either, and an arm that
        read the draw would be using a number that does not exist onboard -- the same rule the
        estimated arm keeps about friction.
        """
        from . import clearance as cl
        tracker = self.env.tracker
        if tracker is None:
            raise RuntimeError("the clearance arm needs the plan tracker (--action-mode plan)")
        lidar = self.env.cfg.lidar
        sub = max(1, int(getattr(self.env.ecfg, "scan_subsample", 1)))
        angles = cl.beam_angles(lidar.n_beams, lidar.fov, device=self.device)[::sub]
        if angles.numel() != int(self.env.n_beams):
            raise RuntimeError(f"{angles.numel()} bearings for the {self.env.n_beams} beams this "
                               f"env reports: the grid would be built from bearings the returns do "
                               f"not have")
        from . import floor as fl
        # Nominal mounting again, for the same reason the bearings are: the floor gate's geometry is
        # what the car believes about its sensor, not the draw the simulator made.
        fspec = fl.FloorSpec(mount_x=float(lidar.mount_x), mount_y=float(lidar.mount_y),
                             mount_z=float(lidar.mount_z))
        return cl.ClearanceArm(tracker, self.clearance_spec or cl.ClearanceSpec(), self.B,
                               self.device, float(self.env.ecfg.v_max_policy), angles,
                               float(self.env.range_max), float(lidar.mount_x),
                               float(lidar.mount_y), fspec=fspec,
                               dt=float(self.env.sim.control_dt))

    def adopt(self) -> None:
        """Transfer the grip solver graph to the calling thread (viewer: the sim thread). See
        `GripMPC.install(adopt=False)`. No-op for the legacy arm or an eager solver."""
        if self.grip is not None:
            self.grip.adopt()

    def _load_estimator(self) -> None:
        """Load the student, prove it is frozen, and keep the module that actually holds the weights.

        Backend's `GripEstimator` is a wrapper: the parameters live on `.net`. Reading
        `estimator.parameters()` therefore finds nothing, which would make both the freeze check and
        the checkpoint embed silently vacuous -- a passing check over an empty list.
        """
        from .grip_estimator import SensorHistory, load_grip_estimator
        self.estimator = load_grip_estimator(self.estimator_path, self.device)
        net = getattr(self.estimator, "net", None)
        if net is None or not list(net.parameters()):
            raise RuntimeError(
                "could not find the estimator's parameters on `.net`; refusing to continue rather "
                "than record an empty freeze check and embed an empty student into the checkpoint")
        self.net = net
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)                # never reachable by an optimizer
        if any(p.requires_grad for p in net.parameters()):
            raise RuntimeError("the estimator still has trainable parameters after freezing")
        self.history = SensorHistory(self.B, self.device, self.estimator.spec)

    def release(self) -> None:
        if self.grip is not None:
            self.grip.release()
        if self.spy is not None:
            self.spy.release()
        if self.traction is not None:
            self.traction.release()
        if self.clearance is not None:
            self.clearance.release()
        self.grip = self.spy = self.traction = self.clearance = None
        self._installed = False

    def begin(self, obs: dict) -> None:
        """Called once after `env.reset()`. Every env starts cold, with a zero previous command.

        The history is deliberately *not* written here. `env.reset()`'s observation is the same
        object the first `pre_action` receives, so seeding a row now and pushing it again there
        would put one real observation into two rows -- two frames of a 40-frame window, reaching
        `warm` a step early on a duplicate. Instead the first `pre_action` pushes it once, with the
        reset mask set for every env, which clears and inserts exactly one row.
        """
        self._prev_cmd = torch.zeros(self.B, 2, device=self.device)
        self._reset_mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        if self.spy is not None:
            # `env.reset()` resets every env, which fired the spy. That snapshot is the pre-reset
            # command of whatever ran before this run, not of a step this run took.
            self.spy.discard()

    # -- per step ----------------------------------------------------------------
    @torch.no_grad()
    def pre_action(self, obs: dict) -> Optional[torch.Tensor]:
        """Advance the history, infer the friction, and hand it to the MPC. Before the action."""
        if self.clearance is not None:
            # This step's own LiDAR frame, before the action it will shape is asked for. The arm
            # holds one frame and nothing carried across a boundary, so nothing here needs
            # clearing; its floor gate's attitude tracker does, and `post_step` does that.
            self.clearance.update_scan(obs["scan"])
            if self.clearance.cspec.floor_gate:
                # The gate's attitude, from the same IMU columns the policy's own observation
                # carries -- read out of the observation rather than off `sim.P`, so this is code
                # the car can run. `imu` is the mean of this step's samples, already normalised.
                from .obs import floor_inputs
                idx = self._floor_idx()
                _speed, gyro, accel, _vesc = floor_inputs(
                    torch.cat([obs[k] for k in ("speed", "prev_action", "speed_cap", "imu",
                                                "imu_att")], 1), idx)
                self.clearance.update_attitude(gyro, accel, obs["speed"][:, 0] * idx["v_max"])
        if self.traction is not None:
            # A finished episode is a sensor gap: a new car, on a new surface, possibly at a
            # different speed. Carrying a latched release across it would release the brake of a
            # car that no longer exists. Done here rather than in `post_step` because `_reset_mask`
            # is only written there, and because this runs before the command it would shape.
            self.traction.reset(self._reset_mask)
        if self.base == "legacy":
            return None
        if self.base == "fixed_low":
            # Constant friction: nothing to update, but it is still one of the two main PPO arms and
            # its used friction and realised brake bound belong in the same log as the other's.
            self._record_constant(self.grip.mu)
            return None
        if self.base == "oracle":
            mu = self._truth()
            self.grip.update(mu)
            self._record_constant(mu)
            return mu
        # estimated
        done = self._reset_mask
        # `push(reset_mask)` clears features and valid for those ids before inserting, so a just-reset
        # env comes back with valid_count == 1 and cold. Calling `history.reset` as well would clear
        # it twice and drop the row this step just inserted -- backend's contract is one or the
        # other, not both. The filter state is still mine to reset: backend holds it, I say when an
        # episode ended.
        self.history.push(obs, self._prev_cmd, done)
        ids = torch.nonzero(done).flatten()
        if ids.numel():
            self.estimator.reset_filter(ids)
        features, valid = self.history.inputs()
        used_mu, diag = self.estimator.lower_mu(features, valid)
        self.grip.update(used_mu)
        self._record(used_mu, diag)
        return used_mu

    def _floor_idx(self) -> dict:
        """The proprio column map of the observation this env emits, built once.

        Deliberately the *short* proprio (no history block): `pre_action` assembles the prefix it
        needs from `obs` rather than taking the flattened vector, because the flattened width
        depends on `hist_len` and the columns the gate reads are all in the prefix.
        """
        if getattr(self, "_fidx", None) is None:
            from .obs import ObsSpec, att_index_spec
            e = self.env.ecfg
            sp = ObsSpec(n_beams=int(self.env.n_beams), act_dim=int(self.env.act_dim),
                         action_history=int(e.action_history), hist_len=0,
                         range_max=float(self.env.range_max), v_max=float(e.v_max_policy),
                         gyro_scale=float(e.imu_gyro_scale), accel_scale=float(e.imu_accel_scale))
            self._fidx = att_index_spec(sp)
        return self._fidx

    def post_step(self, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        """Capture the issued command before auto-reset can overwrite it, and mark the boundaries.

        The realised control bound is read here rather than in `pre_action` because the solver has
        now run: at `pre_action` time `last_bounds` still holds the previous step's limits.
        """
        done = terminated | truncated
        if self.clearance is not None:
            # The floor gate's attitude integrator is episode state: a new car on a new floor does
            # not inherit the last one's tilt. Its at-rest reference is NOT cleared -- that is the
            # sensor's mounting, which the same car keeps across episodes.
            self.clearance.reset(done)
        if self.base == "legacy":
            self._reset_mask = done
            return
        self._record_realised_bounds()
        if self.base == "estimated":
            self._prev_cmd = self.spy.take()
        self._reset_mask = done

    # -- truth, only where it is allowed -----------------------------------------
    def _truth(self) -> torch.Tensor:
        """The episode's true friction, cloned. **Only the oracle arm may call this.**

        `randomization.py` writes `P` in place and its entries are views, so an un-cloned read is a
        live alias that a later reset rewrites.
        """
        if self.base != "oracle":
            raise RuntimeError(f"the {self.arm!r} arm must not read P; truth arrives through "
                               f"observe_truth(), supplied by the caller")
        return self.env.sim.P["mu"].detach().clone().reshape(-1)

    def observe_truth(self, mu_true: Optional[torch.Tensor]) -> None:
        """Supply a **pre-action clone** of the true friction, for metrics only.

        Kept outside inference on purpose. The estimated arm must run on a car with no access to
        `P` or the privileged vector, so reaching into `sim.P` from inside `pre_action` -- even only
        to log -- would make the onboard path depend on something that does not exist onboard. The
        caller owns that read; if it never calls this, coverage and the overestimate flag are simply
        absent and inference is unaffected.
        """
        self._pending_truth = None if mu_true is None else mu_true.detach().reshape(-1)

    # -- metrics -----------------------------------------------------------------
    def _record_constant(self, mu: torch.Tensor) -> None:
        """fixed_low and oracle: friction is not inferred, so only the used value is logged."""
        self.acc.add(mu_used=mu.sum(), mu_used_sq=(mu * mu).sum())

    def _record(self, used_mu: torch.Tensor, diag: dict) -> None:
        """B3's per-update logging, in env-transitions, accumulated on device.

        Nothing is converted to a Python float here. Each `float(tensor)` is a host synchronise, and
        fifteen of them per step was costing more than the inference being measured.
        """
        a = self.acc
        warm = diag["warm"].to(used_mu.dtype)
        fallback = (diag["fallback_reason"] != FALLBACK_NONE).to(used_mu.dtype)
        at_floor = ((used_mu - self.gspec.mu_fixed).abs() <= FLOOR_EPS).to(used_mu.dtype)
        a.add(warm=warm.sum(), fallback=fallback.sum(),
              warm_fallback=(fallback * warm).sum(), q_gap=(diag["q_gap"] * warm).sum(),
              excitation=diag["excitation_pass"].to(used_mu.dtype).sum(),
              mu_used=used_mu.sum(), mu_used_sq=(used_mu * used_mu).sum(),
              at_floor=(at_floor * warm).sum())
        truth = self._pending_truth
        if truth is not None:
            # Supplied by the caller as a pre-action clone. Never read from `P` here: the estimated
            # arm's inference path must not touch anything a car does not have.
            over = (used_mu > truth + OVER_EPS).to(used_mu.dtype)
            covered = ((diag["q10"] <= truth) & (truth <= diag["q90"])).to(used_mu.dtype)
            a.add(truth=warm.sum(), over=(over * warm).sum(), covered=(covered * warm).sum())
        self._pending_truth = None

    def _record_realised_bounds(self) -> None:
        """The brake limit the solver actually imposed this step, read after it ran.

        `GripMPC.last_bounds` is written inside `solve_grip` (and inside the graph, since it is
        closed over), so this is the minimum over the planned path -- the number the iLQR was really
        clamped to. The straight-line budget at rho = 0 is a different, larger number on any curved
        plan, and logging that instead would overstate the limit the car was driving under.
        """
        a_brk = self.grip.last_bounds[:, 0]
        self.acc.steps += 1
        self.acc.add(a_brk=a_brk.sum(), a_brk_n=float(self.B), env_steps=float(self.B))
        self.acc.add_range(a_brk)

    def collect_metrics(self) -> tuple[dict, list[str]]:
        """Drain the update's sums and evaluate B3's flags. One host synchronise, once per update."""
        tcs_d = self.traction.metrics() if self.traction is not None else {}
        if self.clearance is not None:
            tcs_d.update(self.clearance.metrics())
        if self.acc.steps == 0:
            return tcs_d, []
        raw = self.acc.read()
        self.acc = Accumulator(self.device)
        fired = self.flags.update(raw)
        d = metrics_dict(raw, estimator=self.base == "estimated")
        d["controller/flag_degeneracy_run"] = self.flags.degeneracy_run
        d["controller/flag_overestimate_run"] = self.flags.overestimate_run
        d.update(tcs_d)
        return d, fired

    # -- provenance --------------------------------------------------------------
    def checkpoint_meta(self) -> dict:
        """What a policy checkpoint must carry so a consumer can refuse a mismatched pair."""
        meta = {"arm": self.arm, "grip_spec": self.gspec.to_meta(),
                "estimator_path": self.estimator_path,
                "flags_fired": list(self.flags.fired)}
        if self.traction is not None:
            # Every threshold the guard ran under. A `tcs` policy learned to drive with a specific
            # release authority and a specific lock gate; a consumer that installs different ones is
            # not running the controller this checkpoint was trained against.
            meta["traction"] = self.traction.meta()
        if self.clearance is not None:
            # The margin, the grid and the beam geometry the plan was judged against. A result
            # produced at 0.20 m through a 0.06 m grid is not the same system as one at 0.10 m.
            meta["clearance"] = self.clearance.meta()
        if self.estimator is not None:
            e = self.estimator
            meta["estimator"] = {
                "sha256": getattr(e, "sha", None),          # content hash of the estimator checkpoint
                "feature_spec": _as_plain(getattr(e, "feature_spec", None)),
                "mu_range": _as_plain(getattr(e, "mu_range", None)),
                "calibration": _as_plain(getattr(e, "calibration", None)),
                "meta": _as_plain(getattr(e, "meta", None)),
            }
            # Embed the student itself. It is small, and an embedded copy means a result can be
            # reproduced from the policy checkpoint alone -- an external path can be moved, replaced
            # or retrained under the same filename, and `sha256` would then only tell you that it
            # no longer matches, not what the run actually used.
            net = getattr(self, "net", None)
            if net is None or not list(net.parameters()):
                raise RuntimeError("no estimator parameters to embed; refusing to write a "
                                   "checkpoint that claims an estimated arm but carries no student")
            meta["estimator_state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            # `GripEstimator.arch`, not `meta`. `meta` is the run record -- train seed, split report,
            # dataset -- and deliberately says nothing about the weights, so a `.get("architecture")`
            # on it returns nothing and would embed an empty architecture next to a real state dict:
            # weights that cannot be rebuilt, with no error at the point that matters. Backend pins
            # `arch` with a test, so this breaks loudly if it ever moves.
            arch = getattr(e, "arch", None)
            if not arch:
                raise RuntimeError("the estimator reports no architecture; refusing to embed a "
                                   "state dict that could not be reconstructed from it")
            meta["estimator_architecture"] = _as_plain(arch)
        return meta


def _as_plain(v):
    """Dataclasses and tensors reduced to something `torch.save` and a human can both read."""
    import dataclasses
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _as_plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {k: _as_plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_as_plain(x) for x in v]
    return str(v)
