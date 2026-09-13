"""Recurrent memory for the driving policy: the state, the module, and the per-path runtime.

Why this exists (`docs/research/memory-policy-2026-09-13.md`): the actor's observation is six LiDAR
frames -- 150 ms at 40 Hz -- plus a 20-row proprio history. A box that has left the scan window is
gone from the observation entirely, and a row of boxes with cracks between them reads as a row of
openings. Three finetunes with more obstacles, looser leashes and a plan-clearance penalty all
scored the same on the held-out proxies, and 11 of 16 crashes on the user's held-out scene were on
one row of boxes. What is left is the observation and the architecture.

Two pieces live here:

* `GRUMemory` -- a GRU over the per-step trunk embedding whose output enters the first MLP layer's
  *preactivation* through a bias-free projection initialised to zero. That is the same construction
  `Actor.cond` uses, and for the same two reasons: at step 0 the actor's output is bit-identical to
  the feedforward original it was warm-started from (nothing the original could do is lost), and
  the gradient `dL/dW = delta_pre . y^T` is nonzero from the first update, so the path trains
  immediately rather than starving.

* `Hidden`, and the runtime that carries it. A recurrent policy is only correct if every caller
  threads its state through *and* clears it at episode boundaries; a hidden state carried across a
  reset is a policy remembering a track it is no longer on. `PolicyRuntime` is the one object that
  owns (hidden state, scan-channel memory) for an inference path, so a path is converted by using
  it rather than by remembering to do two things.

GRU rather than LSTM: one state tensor instead of two -- which is what every inference path,
the ROS node's callback state and the viewer's static capture tensor have to carry -- about a
quarter fewer parameters at the same width, and nothing the extra gate is needed for at the
sequence lengths this trains on (truncated BPTT over a 32-step horizon, 0.8 s at 40 Hz). The repo
already uses `nn.GRU` for `ScanStem`'s optional temporal encoder, so the export path is the one
that has been exercised.
"""
from __future__ import annotations

import weakref
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn as nn

#: Supported memory kinds. Recorded in `meta["memory"]["kind"]`.
MEMORY_KINDS = ("gru",)

#: Default width. 128 over the 384-wide trunk embedding: inside the contract's 256 ceiling, and
#: measured at 1.07x the frozen actor's CPU forward time against a 1.5x budget
#: (`python -m f1sim.learn.budget`).
DEFAULT_HIDDEN = 128


def memory_spec(kind: str = "gru", hidden_size: int = DEFAULT_HIDDEN, layers: int = 1,
                critic: str = "own") -> dict:
    """The `meta["memory"]` block, validated. Written by the trainer, read by every loader."""
    kind = str(kind)
    if kind not in MEMORY_KINDS:
        raise ValueError(f"memory kind {kind!r} is not one of {MEMORY_KINDS}")
    hidden_size, layers = int(hidden_size), int(layers)
    if not 0 < hidden_size <= 256:
        raise ValueError(f"memory hidden_size {hidden_size} must be in (0, 256]: the contract's "
                         f"Jetson budget caps it at 256")
    if layers < 1:
        raise ValueError(f"memory layers {layers} must be >= 1")
    if critic not in ("own", "none"):
        raise ValueError(f"memory critic must be 'own' or 'none', got {critic!r}")
    return {"kind": kind, "hidden_size": hidden_size, "layers": layers, "critic": critic}


class GRUMemory(nn.Module):
    """GRU over a per-step embedding, added to a preactivation through a zero-initialised projection.

    `step` advances one control step. `h` is `(layers, batch, hidden_size)`, the layout `nn.GRU`
    uses, so nothing has to be transposed on the way in or out.
    """

    def __init__(self, in_dim: int, hidden_size: int, out_dim: int, layers: int = 1):
        super().__init__()
        self.hidden_size, self.layers = int(hidden_size), int(layers)
        self.gru = nn.GRU(int(in_dim), self.hidden_size, num_layers=self.layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, int(out_dim), bias=False)
        nn.init.zeros_(self.out.weight)

    def initial(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        ref = self.out.weight
        return torch.zeros(self.layers, int(batch), self.hidden_size,
                           device=device or ref.device, dtype=dtype or ref.dtype)

    def step(self, x: torch.Tensor, h: Optional[torch.Tensor]):
        """(delta on the preactivation, next hidden). `x` is (batch, in_dim)."""
        if h is None:
            h = self.initial(x.shape[0], x.device, x.dtype)
        y, h_next = self.gru(x.unsqueeze(1), h)
        return self.out(y[:, 0]), h_next


class Hidden(NamedTuple):
    """The recurrent state of one `ActorCritic`, as it travels through the inference paths.

    `critic` is None on every deployment path -- the critic does not run on the car -- and is
    carried only by training. A NamedTuple rather than a tensor so that adding the critic's state
    did not change any caller's unpacking, and so that `h.actor is None` is the honest answer for a
    feedforward checkpoint instead of a zero tensor that looks like memory.
    """
    actor: Optional[torch.Tensor] = None
    critic: Optional[torch.Tensor] = None

    @property
    def empty(self) -> bool:
        return self.actor is None and self.critic is None

    def detach(self) -> "Hidden":
        return Hidden(*(None if t is None else t.detach() for t in self))

    def clone(self) -> "Hidden":
        return Hidden(*(None if t is None else t.clone() for t in self))

    def to(self, *args, **kw) -> "Hidden":
        return Hidden(*(None if t is None else t.to(*args, **kw) for t in self))

    def reset(self, done) -> "Hidden":
        """Zero the columns of every carried state whose episode ended. `done` is (batch,)."""
        return Hidden(*(None if t is None else reset_hidden(t, done) for t in self))

    def select(self, index) -> "Hidden":
        return Hidden(*(None if t is None else t.index_select(1, index) for t in self))


def reset_hidden(h: Optional[torch.Tensor], done) -> Optional[torch.Tensor]:
    """`h` with the batch columns named by `done` set to zero. Out of place, `None` in `None` out.

    `done` is anything broadcastable to (batch,) -- a bool mask, a float mask, or an index tensor is
    NOT accepted, because "indices" and "a mask" are indistinguishable at a glance and one of the
    two silently resets the wrong rows.
    """
    if h is None:
        return None
    if not torch.is_tensor(done):
        done = torch.as_tensor(done, device=h.device)
    if done.dtype == torch.bool:
        keep = (~done).to(h.dtype)
    else:
        keep = (1.0 - done.to(h.dtype)).clamp_(0.0, 1.0)
    if keep.dim() != 1 or keep.shape[0] != h.shape[1]:
        raise ValueError(f"episode-boundary mask must be ({h.shape[1]},), got {tuple(keep.shape)}")
    return h * keep[None, :, None]


# ------------------------------------------------------------------ per-path runtime
class PolicyRuntime:
    """Everything a (possibly) recurrent, (possibly) scan-augmented policy carries between steps.

    One object per inference path. `reset()` with no argument clears everything (a fresh episode on
    every row); `reset(done)` clears the rows whose episode just ended. Both the hidden state and
    the decayed scan-occupancy channel are cleared together, because they are the same claim --
    "what I have seen so far" -- and clearing one without the other leaves the policy half in the
    episode that ended.

    The width is taken from the first observation rather than declared, and a change of width
    rebuilds the state from scratch. Callers that drive one env of a fixed size never notice; the
    benchmark, which walks cells whose `envs` differ, gets the only correct behaviour available --
    state per row cannot be carried from 8 rows to 16 -- instead of an exception in the middle of a
    suite or, worse, a broadcast.
    """

    def __init__(self, memory: bool = False, channels=(), n_beams: int = 0, tau_s: float = 2.0,
                 aligned: Optional[dict] = None):
        self.memory, self.channels = bool(memory), tuple(channels)
        self.n_beams, self.tau_s = int(n_beams), float(tau_s)
        #: The `aligned` channel's spec, including the proprio index block it warps with. Carried
        #: rather than rebuilt so that a runtime is a faithful copy of what the checkpoint recorded.
        self.aligned = dict(aligned) if aligned else None
        self.hidden: Optional[Hidden] = Hidden() if self.memory else None
        self.scan = None
        self.batch: Optional[int] = None
        self._device = None

    @property
    def stateful(self) -> bool:
        return self.memory or bool(self.channels)

    def ensure(self, batch: int, device=None) -> None:
        """Size the state to `batch` rows, rebuilding (and so clearing) it if the width changed.

        The device is compared against the tensor that actually holds state, not against the
        argument: `torch.device("cuda")` and `torch.device("cuda:0")` are unequal objects for the
        same device, and rebuilding on that difference would clear the memory every step.
        """
        if not self.stateful:
            return
        held = None if self.scan is None or self.scan.mem is None else self.scan.mem.device
        if self.batch == int(batch) and (device is None or held is None or held == device):
            return
        from .obs import ScanAugment
        self.batch, self._device = int(batch), device
        self.hidden = Hidden() if self.memory else None
        self.scan = (ScanAugment(self.channels, self.n_beams, self.batch, device=device or "cpu",
                                 tau_s=self.tau_s, aligned=self.aligned)
                     if self.channels else None)

    def reset(self, done=None) -> None:
        if self.hidden is not None:
            self.hidden = Hidden() if done is None else self.hidden.reset(done)
        if self.scan is not None:
            self.scan.reset(done)

    def observe(self, scan: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The scan the policy actually sees: the stacked frames plus any enabled extra channels.

        `proprio` is needed only by the `aligned` channel, which warps with the car's own measured
        motion; every other channel ignores it and a path that enables none is unchanged. It is not
        defaulted to zeros inside the augmenter: a warp told the car is standing still would return
        a residual made of the ego's motion, which is exactly the thing this channel exists to
        remove, and it would look like a working channel.
        """
        self.ensure(scan.shape[0], scan.device)
        return scan if self.scan is None else self.scan(scan, proprio)


def runtime_for(model, batch: Optional[int] = None, device=None) -> PolicyRuntime:
    """The runtime a checkpoint needs -- inert (`stateful` False) for a legacy checkpoint.

    `batch` is optional: the width is taken from the first observation if it is not given.
    """
    meta = getattr(model, "meta", {}) or {}
    chan = meta.get("scan_channels") or {}
    rt = PolicyRuntime(memory=bool(meta.get("memory")), channels=chan.get("channels") or (),
                       n_beams=int(meta.get("n_beams", 0)),
                       tau_s=float(chan.get("memory_tau_s", 2.0)),
                       aligned=chan.get("aligned"))
    if batch is not None:
        rt.ensure(int(batch), device or next(model.parameters()).device)
    return rt


def policy_fn(model, batch: Optional[int] = None, device=None, deterministic: bool = True,
              cond: Optional[torch.Tensor] = None) -> Callable:
    """`obs -> action`, carrying the hidden state and the scan channels across calls.

    The returned callable has `.runtime` and `.reset(done=None)`. Every inference path that scores
    or drives a checkpoint uses this rather than calling `model.act` itself, so "does this path
    carry the memory?" has one answer instead of one per call site.
    """
    from .obs import flatten_obs
    rt = runtime_for(model, batch, device)

    @torch.no_grad()
    def run(obs):
        scan, proprio = flatten_obs(obs)
        action, _logp, rt.hidden = model.act(rt.observe(scan, proprio), proprio,
                                             deterministic=deterministic, c=cond, h=rt.hidden)
        return action

    run.runtime = rt
    run.reset = rt.reset
    return run


# ------------------------------------------------------------------ episode-boundary broadcast
#: Objects that asked to be cleared whenever an env they are driving ends an episode. A listener is
#: anything with `reset(done=None)` and a `batch` (its width, or None until it has seen an
#: observation).
#:
#: Why a registry at all: the viewer worker builds its env (`common.make_env`) and its actor
#: callable (`watch.actor_runner`) in two places that never meet, inside `f1sim/viewer/**`, which
#: this branch does not own. So the env side announces boundaries (`common.announce_episode_
#: boundaries`) and the policy side listens. Weak references, so a session that is torn down takes
#: its listener with it rather than leaking one per map change.
#:
#: Normally empty: training, evaluation and the benchmark all thread the state through by hand and
#: register nothing, so nothing is broadcast to them and the announcement costs one `if` per step.
_LISTENERS: "weakref.WeakSet" = weakref.WeakSet()


def add_boundary_listener(rt) -> Callable[[], None]:
    """Register `rt` for episode-boundary resets. Returns the function that unregisters it."""
    _LISTENERS.add(rt)

    def remove():
        _LISTENERS.discard(rt)
    return remove


def broadcast_boundary(done=None, batch: Optional[int] = None) -> None:
    """Tell every registered listener that these rows started a new episode.

    `batch` is the announcing env's width: a listener built for a different width is not this env's
    and is skipped, so two sessions in one process cannot reset each other's state.
    """
    if not _LISTENERS:
        return
    for rt in list(_LISTENERS):
        if batch is not None and not _runtime_matches(rt, batch):
            continue
        rt.reset(done)


def _runtime_matches(rt, batch: int) -> bool:
    """A listener that has not seen an observation yet matches anything: it has no width to compare
    against, and clearing an empty state is a no-op."""
    b = getattr(rt, "batch", None)
    return b is None or int(b) == int(batch)


def describe(meta: dict) -> str:
    """One line for a log or a checkpoint header."""
    mem = (meta or {}).get("memory")
    ch = ((meta or {}).get("scan_channels") or {}).get("channels") or ()
    if not mem and not ch:
        return "feedforward"
    parts = []
    if mem:
        parts.append(f"{mem.get('kind', '?')} h={mem.get('hidden_size')} x{mem.get('layers', 1)} "
                     f"(critic: {mem.get('critic', 'none')})")
    if ch:
        parts.append("scan channels: " + ",".join(ch))
    al = ((meta or {}).get("scan_channels") or {}).get("aligned")
    if al:
        from .aligned import describe as describe_aligned
        parts.append(describe_aligned({k: v for k, v in al.items() if k != "proprio"}))
    return " | ".join(parts)
