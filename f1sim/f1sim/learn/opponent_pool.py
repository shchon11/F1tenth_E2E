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
itself just saw".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import os

import torch

from ..gym_env import OPP_DRIVER_POOL, POOL_SELF, POOL_TEACHER
from .memory import PolicyRuntime, runtime_for
from .model import load_checkpoint
from .obs import flatten_obs


@dataclass
class PoolEntry:
    """One loaded checkpoint, its inference runtime, and where it came from."""
    path: str
    model: object
    runtime: PolicyRuntime
    hidden: object = None
    meta: Optional[dict] = None

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
    return replace(spec, opp_token="off") if spec.opp_token != "off" else spec


def _check_compatible(path: str, meta: dict, env, spec) -> None:
    """Refuse a checkpoint whose observation or action space is not this env's.

    Loudly, and without an `override`: overriding the shapes is how a warm start widens a critic,
    and doing it here would silently re-initialise part of the *opponent's* actor and put a
    half-random driver in the other car. What a pool entry has to be is a policy that can already
    drive this observation.
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
    """The loaded population. `act(obs, driver)` commands every pool-driven car of the batch."""

    def __init__(self, entries: Sequence[PoolEntry], batch: int, act_dim: int, device):
        self.entries: List[PoolEntry] = list(entries)
        self.B, self.act_dim, self.device = int(batch), int(act_dim), torch.device(device)
        for ent in self.entries:
            ent.model.eval()
            for prm in ent.model.parameters():
                prm.requires_grad_(False)
            ent.runtime.ensure(self.B, self.device)
            ent.hidden = None

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, paths: Sequence[str], env, device=None) -> "OpponentPool":
        """Load the checkpoint entries of an `opp_pool` (the `self` / `teacher` entries are the env's)."""
        device = torch.device(device or env.device)
        spec = _spec_of(env)
        entries = []
        for path in paths:
            # No `allow_oracle`: an entry trained on privileged tokens cannot drive off the
            # observation the pool is handed, and an opponent that needs an oracle is not an
            # opponent a deployable policy would ever meet.
            model, extra = load_checkpoint(path, device)
            _check_compatible(path, dict(model.meta), env, spec)
            entries.append(PoolEntry(path=str(path), model=model,
                                     runtime=runtime_for(model, env.B, device), meta=extra))
        return cls(entries, env.B, env.act_dim, device)

    @torch.no_grad()
    def act(self, obs: Optional[Dict[str, torch.Tensor]], driver: torch.Tensor) -> torch.Tensor:
        """(B, act_dim) normalized action, valid on the rows `driver` assigns to a pool entry.

        Deterministic: the population is an environment, and a sampled opponent would make the same
        race a different race on a re-run of the same seed.
        """
        if obs is None:
            raise RuntimeError("the opponent pool was asked for an action before the env produced "
                               "an observation: call env.reset() first")
        scan, proprio = flatten_obs(obs)
        out = torch.zeros(scan.shape[0], self.act_dim, device=scan.device, dtype=scan.dtype)
        for j, ent in enumerate(self.entries):
            action, _logp, ent.hidden = ent.model.act(ent.runtime.observe(scan), proprio,
                                                      deterministic=True, h=ent.hidden)
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
        return ", ".join(f"{OPP_DRIVER_POOL + j}:{e.name}" for j, e in enumerate(self.entries))


def attach(env, device=None, verbose: bool = True):
    """Build and install the pool an env's `opp_pool` names. No-op outside `opponent == "pool"`.

    One function so that training, the census and any later caller wire the population the same
    way -- the env refuses to run a pool mode without one, which is the point.
    """
    if env.ecfg.opponent != "pool":
        return None
    pool = OpponentPool.load(env.pool_paths, env, device=device)
    env.set_opponent_pool(pool)
    if verbose:
        special = [n for n in env.pool_names if n in (POOL_SELF, POOL_TEACHER)]
        print(f"opponent pool: {len(env.pool_names)} entr{'y' if len(env.pool_names) == 1 else 'ies'}"
              f" -- {pool.describe() or 'none'}"
              + (f" + {', '.join(special)}" if special else ""), flush=True)
    return pool
