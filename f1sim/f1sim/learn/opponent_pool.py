"""A population of checkpoints driving the other cars of a race (`--opponent pool`).

Training has offered the policy one opponent: either a copy of itself (self-play) or the raceline
teacher, which drives one line at one speed scale and never looks at the learner. Measured, that
is a training set with one map. Scripted behaviour widens it (`f1sim.opponent_events`) but every
scripted car is still the same car underneath -- the same tracker on the same line, differing in
what it has been told to do this second.

A *checkpoint* is a different car. It takes its own line through a corner because its own network
decided to, it defends its position because being passed costs it reward, it brakes at its own
braking point and it makes its own mistakes. A pool of them is opponent diversity of the kind that
cannot be written down as a rule, and the population is a flag rather than a mode: `--opp-pool
a.pt,b.pt,self,teacher` is four kinds of opponent drawn per race, of which one is the learner's own
current weights and one is the teacher.

Three properties this module is built around:

* **Loaded once.** The checkpoints are read at construction, put in `eval()` with gradients off,
  and never touched again. A pool is not a second training job.
* **Acted in batch.** One forward per entry per step over the whole env width -- no Python loop
  over cars, and no `nonzero` compaction either: selecting the rows an entry drives would be a
  device-to-host sync every step, and a recurrent entry's hidden state has to advance on the rows
  it owns anyway. The cost is therefore honest and flat: a pool of K checkpoints is K extra
  policy forwards per step (the sim step dominates both).
* **Its state is cleared where episodes end.** `gym_env._reset_envs` calls `reset(ids)`; a hidden
  state carried across a respawn is a policy remembering a track its car is no longer on.

Which car each entry drives is the env's business (`gym_env.OPP_DRIVER_*`, drawn per race from
`sim.gen`); this module only answers "what does entry j command, given the observation the policy
itself just saw" -- or, for an entry of another observation contract, the observation the same
checkpoint would see in an env of its own contract (`learn.seat_obs`).

**Each entry in its own contract** (2026-10-14, the redesign's 6장 1단계). An entry whose observation
contract is the env's reads the env's observation, as every entry always did. Any other one -- a v1
league seat (six frames, the 366-wide proprio vector) racing a v2 learner, or a v2 learner's snapshot
joining that league -- reads a per-seat builder (`learn.seat_obs`) that keeps that contract's
histories for every row, fed by the env at the points it updates its own; one builder per contract,
shared by every entry that reads it. And each entry plans in its own format (`entry_formats`): the env
reads its rows in that format (`F1VecEnv._set_entry_formats`, the per-row tracker). A v1 seat in a v2
env commands, bit for bit, what it commands in a v1 env from the same state (`tests/test_league_v2.py`).
A v1 entry in a v1 env whose observation does not fit is still refused, as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import os

import torch

from ..gym_env import OPP_DRIVER_POOL, POOL_SELF, POOL_TEACHER
from .memory import PolicyRuntime, runtime_for
from .model import controller_arm_of, load_checkpoint
from .obs import flatten_obs


def _recorded_arm(path: str) -> str:
    """The controller arm a checkpoint file records, without building the model.

    `weights_only=True` because this runs before any compatibility check: reading the header of a
    file that may turn out to be the wrong architecture entirely should not execute anything in it.
    """
    try:
        return controller_arm_of(torch.load(path, map_location="cpu", mmap=True, weights_only=True))
    except Exception:
        return "legacy"                # the loader below says what is actually wrong with it


@dataclass
class PoolEntry:
    """One loaded checkpoint, its inference runtime, and where it came from."""
    path: str
    model: object
    runtime: PolicyRuntime
    hidden: object = None
    meta: Optional[dict] = None
    #: The plan-controller arm this checkpoint records (`learn.model.controller_arm_of`). Reported
    #: rather than installed: the env has one plan tracker shared by every car, so a pool entry
    #: cannot run its own. What it buys is a refusal instead of a silent mismatch -- a policy whose
    #: plans were fitted to a friction-limited tracker means something else on the untouched one.
    arm: str = "legacy"
    #: A `learn.watch.DialSource` for a conditional ("dial") entry, or None for an unconditional
    #: one. `.c(batch)` is its (batch, cond_dim) input. An opponent has no operator turning a dial for it, so it
    #: is given one and holds it: the cautious end of the training range, which is what a car you
    #: are racing against should be doing rather than something the learner cannot anticipate.
    cond: object = None
    #: The plan format its rows are read in: its own contract's speed mode (`_check_contract`).
    speed_mode: str = "linear"
    #: `learn.seat_obs.seat_key` of the builder this entry reads, or None: the env's own observation.
    seat_key: Optional[tuple] = None
    #: The observation contract it was trained on (`ObsSpec` / `ObsSpecV2`), when it reads a builder.
    seat_spec: object = None

    @property
    def memory_kind(self) -> str:
        """`memory`, `channels` or `feedforward` -- what carrying its state across a step means."""
        if self.runtime.memory:
            return "memory"
        return "channels" if self.runtime.channels else "feedforward"

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def _spec_of(env):
    """The observation spec a POOL ENTRY has to fit, which is not always the learner's.

    Privileged opponent tokens (`f1sim.opp_token`) are the learner's oracle and are stripped from
    the observation the pool acts on (`F1VecEnv._opponent_obs`), so an entry is checked against the
    proprio vector without them: a deployable LiDAR-only checkpoint is exactly what belongs in the
    population, and requiring it to be an oracle too would make the whole race privileged.
    """
    from dataclasses import replace

    from .common import obs_spec                       # local: common imports the env, we import common
    spec = obs_spec(env)
    # contract v2 has no oracle block to strip (`ObsSpecV2` has no such field)
    return replace(spec, opp_token="off") if getattr(spec, "opp_token", "off") != "off" else spec


def _check_contract(path: str, extra: Optional[dict], env) -> str:
    """Refuse a checkpoint whose plan-output contract (`mpc.contract_of`) this env cannot execute;
    return the plan format its rows are to be read in.

    The shape check cannot catch it: "envelope" and "linear" are both N_KNOTS + 2 wide, and the
    speed command or tracker weights do not change the width at all. An entry trained under another
    contract would drive with its speed dimensions read as something they never meant.

    The format alone may differ. One tracker reads the `mpc.ROW_FORMATS` side by side, a format per
    row (`PlanTracker.row_format`), so a linear league seat in a wp4 learner's race, or the learner's
    wp4 snapshot in a linear league, is read in its own format (`F1VecEnv._set_entry_formats`) -- the
    output-format design's per-entry contract (6, '리그'). What the env's single tracker cannot honour
    per row -- the speed command, the tracker's weights and bounds, a format outside ROW_FORMATS --
    still has to be the env's. The default an entry is compared with is the format the env reads the
    league's rows in: `EnvConfig.opponent_speed_mode` when set, else the env's own.
    """
    from ..mpc import ROW_FORMATS, contract_of
    e = env.ecfg
    have = contract_of(extra)
    mode = getattr(e, "opponent_speed_mode", None) or e.speed_mode
    want = {"speed_mode": str(mode), "speed_command": str(e.speed_command)}
    for k in ("tracker_speed_weight", "tracker_a_max"):
        if getattr(e, k, None) is not None:
            want[k] = float(getattr(e, k))
    for k in ("tracker_profile_from_prediction", "tracker_speed_ff"):
        if getattr(e, k, False):
            want[k] = True
    if have == want:
        return want["speed_mode"]
    rest = lambda c: {k: v for k, v in c.items() if k != "speed_mode"}
    own = have["speed_mode"]
    if (rest(have) == rest(want) and getattr(e, "action_mode", "plan") == "plan"
            and own in ROW_FORMATS and str(e.speed_mode) in ROW_FORMATS and want["speed_mode"] in ROW_FORMATS):
        return own
    raise ValueError(
        f"opponent checkpoint {os.path.basename(str(path))} was trained under plan-output "
        f"contract {have} but this env runs {want}. One tracker reads the plan formats "
        f"{ROW_FORMATS} side by side, a format per car; everything else about the contract is the "
        f"env's own, so this entry's speed dimensions would mean something else: race it only in an "
        f"env with its own contract.")


def _observation_of(path: str, model, extra: Optional[dict], env, spec):
    """How entry `path` is shown the race: (None, None) -- the env's own observation, its contract
    being the env's -- or (key, its spec) of the per-seat builder (`learn.seat_obs`) that keeps its own
    contract's histories. Refuses what neither can drive."""
    from .obs import ObsSpecV2, spec_from_dict
    from .seat_obs import buildable, seat_key
    meta = dict(model.meta)
    name = os.path.basename(str(path))
    env_v2 = getattr(env, "obs_version", 1) == 2
    if meta.get("arch") == "v2":
        own = ObsSpecV2.from_dict(meta["obs"]).validate()
        if env_v2 and own == spec:
            return None, None
    elif not env_v2:
        # A v1 entry in a v1 env reads the env's own observation, exactly as every pool entry did;
        # one that does not fit it is refused as before (`_check_compatible`).
        _check_compatible(path, meta, env, spec)
        return None, None
    else:
        own = spec_from_dict((extra or {}).get("spec"))
        if not (extra or {}).get("spec"):
            raise ValueError(f"opponent checkpoint {name} records no observation spec (extra['spec']); a v1 entry "
                             f"in a contract-v2 env is shown its own observation, which it has to name")
        have = {"n_beams": int(meta.get("n_beams", -1)), "n_stack": int(meta.get("n_stack", -1)),
                "proprio_dim": int(meta.get("proprio_dim", -1)), "act_dim": int(meta.get("act_dim", -1))}
        said = {"n_beams": own.n_beams, "n_stack": own.scan_stack, "proprio_dim": own.proprio_dim,
                "act_dim": own.act_dim}
        if have != said:
            raise ValueError(f"opponent checkpoint {name}: its network {have} is not the observation its spec "
                             f"describes {said}")
    if int(own.act_dim) != int(env.act_dim):
        raise ValueError(f"opponent checkpoint {name} emits {own.act_dim} action dimensions and this env's "
                         f"cars take {env.act_dim}")
    try:
        buildable(own, env)
    except ValueError as exc:
        raise ValueError(f"opponent checkpoint {name} cannot be shown its own observation here: {exc}") from exc
    return seat_key(own), own


def _check_compatible(path: str, meta: dict, env, spec) -> None:
    """Refuse a v1 checkpoint whose observation or action space is not this v1 env's.

    Loudly, and without an `override`: overriding the shapes is how a warm start widens a critic,
    and doing it here would silently re-initialise part of the *opponent's* actor and put a
    half-random driver in the other car. What a pool entry has to be is a policy that can already
    drive this observation. (Across contracts -- a v1 entry in a v2 env, a v2 entry anywhere -- the
    entry is shown its own observation instead, `_observation_of`.)
    """
    want = {"n_beams": env.n_beams, "n_stack": env.ecfg.scan_stack,
            "proprio_dim": spec.proprio_dim, "act_dim": env.act_dim}
    bad = {k: (int(meta.get(k, -1)), int(v)) for k, v in want.items() if int(meta.get(k, -1)) != int(v)}
    if bad:
        detail = ", ".join(f"{k}: checkpoint {a} vs env {b}" for k, (a, b) in bad.items())
        raise ValueError(
            f"--opp-pool entry {os.path.basename(path)} does not fit this env ({detail}). A pool "
            f"opponent drives the same car in the same race off the same observation the learner "
            f"sees, so its observation and action space have to be this env's: run the pool "
            f"entries under the same --scan-stack / --scan-stride / --hist-len / --action-mode as "
            f"the learner, or leave them out of the pool.")


class OpponentPool:
    """The loaded population. `act(obs, driver)` commands every pool-driven car of the batch.

    `seats` holds one per-seat observation builder (`learn.seat_obs`) per contract an entry reads
    that is not the env's own; the env feeds them (`F1VecEnv._seat_builders`). Measured at the boot
    camp's width (256 rows, CPU under a load of ~7 on 4 cores): the v1 builder's step (scan stack, action
    and proprio rings, the observation) 2.7 ms against 347 ms for one v1 entry's forward on one thread
    (0.8 %), 3.4 against 178 ms on four (1.9 %) -- the forwards, as before, are the cost."""

    def __init__(self, entries: Sequence[PoolEntry], batch: int, act_dim: int, device, seats: Optional[dict] = None):
        self.entries: List[PoolEntry] = list(entries)
        self.B, self.act_dim, self.device = int(batch), int(act_dim), torch.device(device)
        #: seat_key -> builder, for the contracts some entry reads (`_prune` drops the rest)
        self.seats: dict = dict(seats or {})
        for ent in self.entries:
            ent.model.eval()
            for prm in ent.model.parameters():
                prm.requires_grad_(False)
            ent.runtime.ensure(self.B, self.device)
            ent.hidden = None
        self._prune()

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def seat_builders(self) -> tuple:
        """The builders the env feeds at every reset and step (`F1VecEnv._seat_builders`)."""
        return self._builders

    @property
    def entry_formats(self) -> List[str]:
        """Each checkpoint entry's plan format, in driver order: what the env reads its rows in."""
        return [e.speed_mode for e in self.entries]

    def _prune(self) -> None:
        used = {e.seat_key for e in self.entries if e.seat_key is not None}
        self.seats = {k: v for k, v in self.seats.items() if k in used}
        self._builders = tuple(self.seats.values())

    @staticmethod
    def _load_entry(path: str, env, device, spec, want_arm: str = "legacy") -> PoolEntry:
        """One checkpoint entry, checked: its controller arm, its observation (`_observation_of`), its
        plan-output contract (`_check_contract`). The builder its contract needs is the caller's."""
        recorded = _recorded_arm(path)
        if want_arm != recorded:
            raise ValueError(
                f"opponent checkpoint {os.path.basename(str(path))} records controller arm "
                f"'{recorded}' but the slot says '{want_arm}'. Its actor emits plans for the "
                f"tracker it was trained against; replayed through another one they mean "
                f"something else. Set the slot's controller to '{recorded}'.")
        # No `allow_oracle`, and no slot field that could ask for one: an entry trained on
        # privileged tokens cannot drive off the observation the pool is handed, and an
        # opponent that needs an oracle is not an opponent a deployable policy would ever meet.
        # `allow_conditional`: a dial checkpoint is a policy like any other once something
        # supplies its dial, and this does (below). Refusing it made the best policies this
        # project produces the only ones that could not be raced against.
        model, extra = load_checkpoint(path, device, allow_controller=(want_arm != "legacy"),
                                       allow_conditional=True)
        key, own = _observation_of(path, model, extra, env, spec)
        mode = _check_contract(path, extra, env)
        cond = None
        if int((model.meta.get("cond") or {}).get("dim", 0) or 0):
            from .watch import dial_for
            cond = dial_for(model, device)          # the cautious end of the training range
        return PoolEntry(path=str(path), model=model, runtime=runtime_for(model, env.B, device), meta=extra,
                         arm=recorded, cond=cond, speed_mode=mode, seat_key=key, seat_spec=own)

    def _seat_for(self, ent: PoolEntry, env) -> None:
        """The builder `ent` reads, made (and started from the env's latest step) if no entry read
        that contract until now."""
        if ent.seat_key is not None and ent.seat_key not in self.seats:
            from .seat_obs import make_seat
            self.seats[ent.seat_key] = make_seat(ent.seat_spec, env)

    @classmethod
    def load(cls, paths: Sequence[str], env, device=None,
             arms: Optional[Dict[str, str]] = None) -> "OpponentPool":
        """Load the checkpoint entries of an `opp_pool` (the `self` / `teacher` entries are the env's).

        `arms` maps a path to the controller arm the caller says that checkpoint was trained under
        -- a slot table carries one per row. Naming it is what opts the load in: a checkpoint whose
        recorded arm is not `legacy` is refused by default, and refused again here when the caller's
        answer is not the file's, because "running it on the wrong tracker" and "the arms differ" are
        the same mistake at two different moments.
        """
        device = torch.device(device or env.device)
        spec = _spec_of(env)
        arms = dict(arms or {})
        entries = [cls._load_entry(path, env, device, spec, str(arms.get(str(path), "legacy"))) for path in paths]
        pool = cls(entries, env.B, env.act_dim, device)
        for ent in entries:
            pool._seat_for(ent, env)
        pool._prune()
        return pool

    def replace(self, j: int, path: str, env) -> None:
        """Swap entry `j` for the checkpoint at `path` (a growing league: the learner's own snapshot
        takes an old snapshot's place). The cars entry `j` drives keep their driver index and are
        driven by the new weights from the next step; their carried state starts fresh. Same checks
        as `load` -- a snapshot of the learner passes them by construction. The new entry may read
        another observation contract and plan in another format than the one it replaces (a v2
        snapshot in a v1 seat): the env is told (`set_opponent_pool`), which re-reads both."""
        if not 0 <= int(j) < len(self.entries):
            raise IndexError(f"pool entry {j} of {len(self.entries)}")
        old = self.entries[int(j)]
        new = self._load_entry(path, env, self.device, _spec_of(env), old.arm)
        new.model.eval()
        for prm in new.model.parameters():
            prm.requires_grad_(False)
        new.runtime.ensure(self.B, self.device)
        new.hidden = None
        self.entries[int(j)] = new
        self._seat_for(new, env)
        self._prune()
        if getattr(env, "pool", None) is self:
            env.set_opponent_pool(self)

    @torch.no_grad()
    def act(self, obs: Optional[Dict[str, torch.Tensor]], driver: torch.Tensor) -> torch.Tensor:
        """(B, act_dim) normalized action, valid on the rows `driver` assigns to a pool entry.

        Deterministic: the population is an environment, and a sampled opponent would make the same
        race a different race on a re-run of the same seed. An entry of the env's own contract acts
        on `obs`, the observation the policy itself just saw; any other on its builder's observation
        of the same step (`learn.seat_obs`).
        """
        if obs is None:
            raise RuntimeError("the opponent pool was asked for an action before the env produced "
                               "an observation: call env.reset() first")
        scan, proprio = flatten_obs(obs)
        out = torch.zeros(scan.shape[0], self.act_dim, device=scan.device, dtype=scan.dtype)
        flat = {None: (scan, proprio)}
        for j, ent in enumerate(self.entries):
            if ent.seat_key not in flat:
                seat_obs = self.seats[ent.seat_key].obs
                if seat_obs is None:
                    raise RuntimeError(f"pool entry {ent.name}'s observation builder has not seen a step: "
                                       f"the env was not reset since it was attached")
                flat[ent.seat_key] = flatten_obs(seat_obs)
            s_, p_ = flat[ent.seat_key]
            c = None if ent.cond is None else ent.cond.c(s_.shape[0])
            action, _logp, ent.hidden = ent.model.act(ent.runtime.observe(s_, p_), p_,
                                                      deterministic=True, c=c, h=ent.hidden)
            out = torch.where((driver == OPP_DRIVER_POOL + j)[:, None], action.to(out.dtype), out)
        return out

    def reset(self, ids: torch.Tensor) -> None:
        """Clear the runtime state of the rows whose episode just ended."""
        if ids.numel() == 0 or not any(e.runtime.stateful for e in self.entries):
            return
        done = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        done[ids] = True
        for ent in self.entries:
            if not ent.runtime.stateful:
                continue
            ent.runtime.reset(done)
            if ent.hidden is not None:
                ent.hidden = ent.hidden.reset(done)

    def describe(self) -> str:
        if not self.entries:
            return "no checkpoint entries"

        def tag(e):
            # an entry in the env's own contract is described as it always was
            if e.seat_key is None:
                return ""
            return f" [own observation v{2 if hasattr(e.seat_spec, 'n_frames') else 1}, {e.speed_mode} plans]"
        return ", ".join(f"{OPP_DRIVER_POOL + j}:{e.name}{tag(e)}" for j, e in enumerate(self.entries))


def attach(env, device=None, verbose: bool = True):
    """Build and install the pool an env's `opp_pool` names. No-op outside `opponent == "pool"`.

    One function so that training, the census and any later caller wire the population the same
    way -- the env refuses to run a pool mode without one, which is the point.
    """
    if env.ecfg.opponent not in ("pool", "slots") or not env.pool_paths:
        return None
    arms = {}
    if getattr(env, "slots", None) is not None:
        arms = {str(sl.checkpoint): str(sl.controller) for sl in env.slots if sl.checkpoint}
    pool = OpponentPool.load(env.pool_paths, env, device=device, arms=arms)
    env.set_opponent_pool(pool)
    if verbose:
        special = [n for n in env.pool_names if n in (POOL_SELF, POOL_TEACHER)]  # noqa: E501
        print(f"opponent pool: {len(env.pool_names)} entr{'y' if len(env.pool_names) == 1 else 'ies'}"
              f" -- {pool.describe() or 'none'}"
              + (f" + {', '.join(special)}" if special else ""), flush=True)
    return pool
