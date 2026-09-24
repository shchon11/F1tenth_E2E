"""DAgger: distill a privileged teacher into the student.

Iteration 0 collects with the teacher driving; later iterations let the student drive (with
probability 1 - beta) while the teacher labels every visited state. All data is aggregated;
the student is trained with a Huber loss on the normalized action.
Scans are stored once per step (fp16, CPU) and stacks are rebuilt at sampling time.

Two things this loop can now do that it could not:

* **The teacher may be `f1sim.interactive_teacher.InteractiveTeacher`** (`--teacher interactive`),
  which searches the plan space against the opponents' predicted motion instead of following the
  raceline. The rest of the loop is unchanged: the label is still `env.teacher_label(teacher)`.
  This is the whole point -- a student can only be taught what its teacher does, and the raceline
  teacher does not race.
* **The student may be recurrent** (`--memory gru`), and then the update is truncated BPTT over
  contiguous chunks of the buffer rather than a uniform draw of single steps. A recurrent policy
  trained from a zero hidden state at every sample is a different policy from the one that drives,
  and the GRU would learn a bias term instead of a memory; the chunks are what make the two the
  same network. The buffer is already (steps, envs), so a chunk is a slice of it and nothing had
  to be stored twice -- except the decayed occupancy channel, which is recursive in the scans and
  so is recorded as the policy saw it rather than replayed from the middle of an episode.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..gym_env import EnvConfig, PRIV_OPP_DIST_SCALE, OPP_FUTURE_MODELS
from ..opp_token import OPP_TOKEN_MODES, validate_opp_token
from ..mpc import ACT_DIM, N_KNOTS
from ..interactive_teacher import DEFAULT_OFFSETS, DEFAULT_SPEEDS, InteractiveTeacher, TeacherCost
from .. import tracks as track_catalog
from ..params import Config
from . import common
from . import conditioning as cond_mod

#: How close an opponent has to be for the auxiliary head to be scored on it [m]; `ppo.AUX_OPP_RANGE_M`.
AUX_OPP_RANGE_M = 6.0
from . import opponent_config as opp_cfg
from .memory import policy_fn as memory_policy_fn, memory_spec, runtime_for
from .model import ActorCritic, load_checkpoint, load_for_memory, save_checkpoint, scan_channel_spec
from .obs import PROPRIO_KEYS, SCAN_CHANNELS, scan_edges, flatten_obs


class StepBuffer:
    """Per-step storage; a training sample (t, b) rebuilds its scan stack from t, t-1, t-2 of env b."""

    def __init__(self, k: int, stride: int = 1, channels=()):
        self.k, self.stride = k, stride
        #: Extra scan channels this buffer has to be able to reproduce, in `SCAN_CHANNELS` order.
        #: `edges` is arithmetic on the stored frame and is recomputed at sampling time; `memory` is
        #: a decayed recursion over every scan since the episode began, so replaying it from the
        #: start of a chunk would hand the network a channel the policy never saw. It is recorded.
        self.channels = tuple(c for c in SCAN_CHANNELS if c in channels)
        self.keep_mem = "memory" in self.channels
        self.scan, self.pro, self.lab, self.newep, self.gap, self.mem = [], [], [], [], [], []
        self.valid = []
        #: What the student was conditioned on, and the two auxiliary targets. Empty lists for a run
        #: that asks for none, which is how `extras_at` answers None without a branch per field.
        self.cond, self.mu, self.opp = [], [], []

    def add(self, scan_now, proprio, label, new_episode, gap=None, mem=None, valid=None,
            cond=None, mu=None, opp=None):
        valid = (torch.ones(label.shape[0], dtype=torch.bool) if valid is None
                 else valid.detach().to(device="cpu", dtype=torch.bool).clone())
        if valid.shape != label.shape[:1]:
            raise ValueError("label validity must have one entry per environment")
        self.valid.append(valid)
        self.scan.append(scan_now.to(torch.float16).cpu()); self.pro.append(proprio.to(torch.float16).cpu())   # a 1 s proprio history is 322 floats/sample
        self.lab.append(label.cpu()); self.newep.append(new_episode.cpu())
        self.gap.append((torch.zeros(label.shape[0]) if gap is None else gap.detach().float().cpu()))
        if self.keep_mem:
            if mem is None:
                raise ValueError("this buffer records the decayed occupancy channel and was handed none")
            self.mem.append(mem.to(torch.float16).cpu())
        for store, value in ((self.cond, cond), (self.mu, mu), (self.opp, opp)):
            if value is not None:
                store.append(value.detach().float().cpu())

    def finalize(self):
        self.S = torch.stack(self.scan); self.P = torch.stack(self.pro); self.L = torch.stack(self.lab); self.N = torch.stack(self.newep)
        self.G = torch.stack(self.gap)                     # |student - teacher| at collection time
        self.V = torch.stack(self.valid)
        self.valid_indices = self.V.reshape(-1).nonzero(as_tuple=True)[0]
        self.valid_count = self.valid_indices.numel()
        self.MEM = torch.stack(self.mem) if self.keep_mem else None
        self.C = torch.stack(self.cond) if self.cond else None      # (T,B,D) the dial the student drove on
        self.M = torch.stack(self.mu) if self.mu else None          # (T,B) normalised friction
        self.O = torch.stack(self.opp) if self.opp else None        # (T,B,4) nearest opponent + in-range flag
        self.T, self.B = self.S.shape[:2]
        self.scan, self.pro, self.lab, self.newep, self.gap, self.mem = [], [], [], [], [], []
        self.valid = []
        self.cond, self.mu, self.opp = [], [], []
        return self

    def extras_at(self, t, b, device):
        """(conditioning, friction target, opponent target) for the rows `samples_at` returns."""
        return (None if self.C is None else self.C[t, b].to(device),
                None if self.M is None else self.M[t, b].to(device),
                None if self.O is None else self.O[t, b].to(device))

    def __len__(self):
        return self.T * self.B

    def samples_at(self, t, b, device):
        self.last_index = (t, b)                # the rows just drawn, for `extras_at`
        n = t.numel()
        stack = []
        cur_t = t.clone(); blocked = torch.zeros(n, dtype=torch.bool)
        for _ in range(self.k):
            stack.append(self.S[cur_t, b])
            for _ in range(self.stride):
                blocked = blocked | self.N[cur_t, b] | (cur_t == 0)
                cur_t = torch.where(blocked, cur_t, cur_t - 1)
        scan = torch.stack(stack, 1).to(device, torch.float32)
        if self.channels:
            # appended in SCAN_CHANNELS order, which is the order the first convolution's input
            # columns are laid out in -- see learn.obs.SCAN_CHANNELS
            extra = []
            now = scan[:, 0]
            for name in self.channels:
                extra.append(self.MEM[t, b].to(device, torch.float32) if name == "memory" else scan_edges(now))
            scan = torch.cat([scan, torch.stack(extra, 1)], 1)
        return scan, self.P[t, b].to(device).float(), self.L[t, b].to(device)

    def sample_sequences(self, m: int, length: int, device):
        """m contiguous runs of `length` steps, each from one env: scan (L,m,k,beams), proprio (L,m,P),
        label (L,m,A), keep (L,m) -- 0 where an episode *starts*, so the recurrence is not carried into it.

        A recurrent student can only learn to hold something for as long as one training sample
        lasts. Friction shows itself in a corner and is needed in the next one, seconds later; an
        i.i.d. (t, b) draw never contains both."""
        length = min(length, self.T)
        t0 = torch.randint(self.T - length + 1, (m,)); b = torch.randint(self.B, (m,))
        t = (t0[None] + torch.arange(length)[:, None]).reshape(-1); bb = b[None].expand(length, m).reshape(-1)
        scan, pro, lab = self.samples_at(t, bb, device)
        keep = (~self.N[t, bb]).to(device).view(length, m)
        return (scan.view(length, m, *scan.shape[1:]), pro.view(length, m, -1), lab.view(length, m, -1), keep)

    def sample(self, n, device, hard_frac: float = 0.0, power: float = 1.0):
        """hard_frac of the batch is drawn in proportion to how far the student was from the teacher
        when the state was collected.

        Measured motivation: on the teacher's own trajectory the student reproduces its action to
        within 2 %, but in the second before a collision the error is 4-10x that. A uniform loss over
        the buffer therefore optimises the on-line majority and leaves the recovery states -- the
        ones that actually end episodes -- effectively unweighted."""
        if not self.valid_count:
            raise ValueError("cannot sample a buffer with no valid teacher labels")
        n_hard = int(n * hard_frac)
        idx = self.valid_indices[torch.randint(self.valid_count, (n - n_hard,))]
        if n_hard:
            # Index first: invalid gaps may themselves be non-finite.
            w = self.G.reshape(-1)[self.valid_indices].double().clamp_min(0.0) ** power
            if float(w.sum()) <= 0:
                picked = torch.randint(self.valid_count, (n_hard,))
            else:
                picked = torch.multinomial(w, n_hard, replacement=True)
            idx = torch.cat([idx, self.valid_indices[picked]])
        t, b = idx // self.B, idx % self.B
        return self.samples_at(t, b, device)

    def sample_chunks(self, n: int, length: int, device, *, return_valid: bool = False):
        """`n` contiguous chunks of `length` steps: (scan (L,n,C,N), proprio (L,n,P), label (L,n,A),
        keep (L,n)).

        `keep` is 0 at a step that begins a new episode, i.e. exactly where the recurrent state must
        NOT be carried in -- the same convention `ActorCritic.evaluate_sequence` takes, so the two
        recurrent update paths in this codebase agree on what an episode boundary does.

        Row-major (step, env): `samples_at` is called once on the flattened index pair, so the scan
        stacks of a whole chunk-batch are rebuilt in one gather rather than L of them.
        """
        length = max(1, min(int(length), self.T))
        t0 = torch.randint(self.T - length + 1, (n,))
        b = torch.randint(self.B, (n,))
        ts = (t0[None, :] + torch.arange(length)[:, None])          # (L, n)
        bs = b[None, :].expand(length, n)
        scan, pro, lab = self.samples_at(ts.reshape(-1), bs.reshape(-1), device)
        keep = (~self.N[ts, bs]).to(device)
        shape = lambda x: x.view(length, n, *x.shape[1:])
        result = (shape(scan), shape(pro), shape(lab), keep)
        # Validity is supervision-only; it must never reset recurrence or remove frames.
        return result + (self.V[ts, bs].to(device),) if return_valid else result


#: `--speed-loss asym` constants, declared here and recorded in every checkpoint the run writes.
#: The loss is in **m/s**, on `e = v_student - v_teacher`, and it is zero at the teacher's speed:
#:
#:     e > 0 (too fast)   W_OVER  * e^2
#:     e < 0 (too slow)   W_UNDER * |e|^UNDER_POW
#:
#: Why asymmetric at all. With a mu-dependent label (`--teacher-grip true`) two identical
#: observations can carry speed labels 2.2x apart, and a symmetric regression can only fit their
#: conditional MEAN -- which is too fast on the low-grip draws, exactly where being too fast ends
#: the episode. An asymmetric loss moves the optimum of that same ambiguity to a low quantile
#: instead: a student that cannot resolve mu from its history backs off to the safe side, and a
#: student that CAN resolve it still reaches the limit at zero loss. The asymmetry is therefore not
#: a safety margin bolted on; it is what makes "I do not know the grip" and "the grip is low" have
#: the same answer, which is the only honest thing a student in that position can do.
#:
#: The ratio, not the absolute size, is the design: the gradient at |e| is 2*W_OVER*|e| above and
#: W_UNDER below, so at the 0.5 m/s error these labels actually carry it is 16x steeper to be fast
#: than to be slow.
W_OVER, W_UNDER, UNDER_POW = 4.0, 0.25, 1.0

#: Both halves of the plan action, so the split cannot drift from `mpc.ACT_DIM`.
SPEED_COLS = slice(N_KNOTS, ACT_DIM)
KNOT_COLS = slice(0, N_KNOTS)


def plan_loss(mu: torch.Tensor, lab: torch.Tensor, speed_loss: str, v_max: float,
              parts: bool = False):
    """Huber on the whole plan (`symmetric`), or Huber on the knots plus an asymmetric speed term.

    `symmetric` is `F.smooth_l1_loss(mu, lab, beta=0.1)` and nothing else, so `--speed-loss
    symmetric` is byte-identical to every DAgger run before this flag existed.

    `asym` keeps the curvature knots on that same loss and replaces only the two speed targets. The
    two halves are recombined in the proportion the single mean had them -- 6 knots to 2 speeds --
    so the knot term keeps its old magnitude and only the speed term is a new thing.
    """
    if speed_loss == "symmetric":
        total = F.smooth_l1_loss(mu, lab, beta=0.1)
        return (total, {"knot": total, "speed": total}) if parts else total
    knot = F.smooth_l1_loss(mu[..., KNOT_COLS], lab[..., KNOT_COLS], beta=0.1)
    # normalized plan speed a -> m/s is (a + 1)/2 * v_max, so an error in a is (v_max / 2) times
    # the error in m/s. The loss is defined in m/s because that is the unit the asymmetry is about.
    e = (mu[..., SPEED_COLS] - lab[..., SPEED_COLS]) * (0.5 * v_max)
    over = e.clamp_min(0.0)
    under = (-e).clamp_min(0.0)
    speed = (W_OVER * over ** 2 + W_UNDER * under ** UNDER_POW).mean()
    n_k, n_s = N_KNOTS, ACT_DIM - N_KNOTS
    total = (n_k * knot + n_s * speed) / ACT_DIM
    return (total, {"knot": knot, "speed": speed}) if parts else total


def actor_sequence(actor, scan, proprio, keep, cond=None) -> torch.Tensor:
    """(L*n, act_dim) deterministic actions over a chunk, the recurrence walked step by step.

    The stem runs once for the whole block and only the GRU and the small MLP walk the steps, which
    is what makes truncated BPTT here cost about what the flat update costs. Same construction as
    `ActorCritic.evaluate_sequence`; separate because DAgger trains the actor alone and has no
    value, no distribution and no privileged vector to carry.
    """
    L, m = scan.shape[0], scan.shape[1]
    flat = lambda t: t.reshape(L * m, *t.shape[2:])
    x, _p = actor.embed(flat(scan), flat(proprio))
    x = x.view(L, m, -1)
    #: The aligned rows a motion branch reads, sliced once for the whole block and walked with it,
    #: exactly as `ActorCritic.evaluate_sequence` does. `None` for every actor without one, which is
    #: every actor this function saw before `--motion-memory` existed.
    rows = actor.motion_input(flat(scan))
    if rows is not None:
        rows = rows.view(L, m, *rows.shape[1:])
    h, feats = None, []
    for t in range(L):
        if h is not None:
            h = h * keep[t].to(x.dtype)[None, :, None]
        f, h, _enc = actor.head(x[t], None if cond is None else cond[t], h,
                                rows=None if rows is None else rows[t])
        feats.append(f)
    return torch.tanh(actor.mu(torch.cat(feats, 0)))


@contextlib.contextmanager
def rng_island(env):
    """Run a block without letting it move any random stream the rest of the loop draws from.

    The per-iteration evaluation is a diagnostic: it adds nothing to the buffer and touches no
    parameter. But `rollout_metrics` calls `env.reset()` and then drives the env for `eval_steps`,
    and every one of those steps spends randomness -- autoresets resample tracks and spawns,
    opponent events fire, friction is redrawn -- out of `env.sim.gen` and the global generators,
    which is exactly where the NEXT iteration's `collect()` continues from. So the length of a
    measurement silently decided the training data, and two arms that differed only in how often
    they were measured were not comparable runs.

    Saving and restoring the streams makes the diagnostic free of consequence: collection sees the
    identical draw whether the eval before it ran 800 steps, 200, or none at all. That is what lets
    `--eval-steps` and `--eval-every` be tuned for cost without touching the experiment.
    """
    saved = {"cpu": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "sim": env.sim.gen.get_state(),
             "py": random.getstate(),
             "np": np.random.get_state()}
    try:
        yield
    finally:
        torch.set_rng_state(saved["cpu"])
        if saved["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda"])
        env.sim.gen.set_state(saved["sim"])
        random.setstate(saved["py"])
        np.random.set_state(saved["np"])


def collection_steps(steps: int, solo_fraction: float):
    """Split a fixed learner-step budget; both cohorts have one learner per race."""
    if steps < 1 or not math.isfinite(solo_fraction) or not 0 <= solo_fraction < 1:
        raise ValueError("--steps must be positive and --solo-fraction must be finite in [0, 1)")
    solo = int(math.floor(steps * solo_fraction + 0.5))
    if solo_fraction and (solo < 1 or solo >= steps):
        raise ValueError("--steps is too small to collect both solo and traffic at this --solo-fraction")
    return solo, steps - solo


def bare_track_names(names):
    """Remove authored and procedural obstacles, retaining base map and travel direction."""
    result = []
    for name in names:
        sc = track_catalog.parse(name)
        if sc.raw:
            raise ValueError(f"cannot derive a guaranteed empty track from {name!r}; use --solo-tracks")
        bare = replace(sc.with_options(obstacle="bare"), asset="", scale=1.0).legacy()
        if bare not in result:
            result.append(bare)
    if not result:
        raise ValueError("solo collection needs at least one track")
    return result


def friction_target(env) -> torch.Tensor:
    """The env's friction on the conditioning module's own scale: one number, one meaning."""
    return (env.sim.P["mu"].reshape(-1) - cond_mod.MU_OFFSET) / cond_mod.MU_SCALE


def opponent_target(env) -> torch.Tensor:
    """(B,4): the nearest opponent's (ahead, side, closing speed) on the scale the actor's head
    predicts, and 1 where one is close enough to be scored on.

    A dynamic car is not a labelled input -- the student sees LiDAR returns and nothing else -- so
    "there is a car there and it is moving like this" has to be read out of how the returns shift
    between frames. The teacher cannot demonstrate that, but the simulator knows it, which makes it
    a dense supervised target where the action is not one.
    """
    priv = env.privileged(env.last_result)
    if env.M <= 1 or priv.shape[1] < 12:
        return torch.zeros(priv.shape[0], 4, device=priv.device)
    o = priv[:, 8:11] / torch.tensor([3.0, 1.0, 2.0], device=priv.device)
    near = (priv[:, 11] * PRIV_OPP_DIST_SCALE < AUX_OPP_RANGE_M).to(priv.dtype)
    return torch.cat([o, near[:, None]], 1)


def collect(env, model, teacher, steps, beta, device, buf: StepBuffer, noise=0.0, need_gap=False,
            cond_fn=None, dial=None):
    """Roll the env for `steps`, labelling every state with the teacher and storing it.

    The student is driven through its `PolicyRuntime`, so a recurrent or scan-augmented checkpoint
    drives here exactly as it drives in evaluation: the hidden state and the decayed occupancy are
    carried across steps and cleared at episode boundaries. For a feedforward student with no extra
    channels the runtime is inert and this is the loop it always was.
    """
    obs, info = env.reset()
    # Only the primary learner is labelled. Opponent rows are driven by their configured
    # scripted/policy population and are not the trajectory this DAgger intervention visited.
    ids = torch.arange(0, env.B, env.M, device=device)
    rt = runtime_for(model, env.B, device)
    new_ep = torch.ones(env.B, dtype=torch.bool, device=device)
    h = model.initial_hidden(env.B, device)                # None for a feedforward student
    # In a race only the learner's rows are the student's own experience: the opponents are driven
    # by the env (`_opponent_actions`) from a teacher, a pool checkpoint or a scripted behaviour, so
    # their observations carry actions the student did not produce. The policy is still *asked* for
    # every row -- the env overrides the ones it drives itself -- but only these are stored.
    lid = env.learner_ids
    solo = int(lid.numel()) == int(env.B)
    keep = (lambda x: x) if solo else (lambda x: x[lid])
    with torch.no_grad():
        for t in range(steps):
            scan, pro = flatten_obs(obs)
            #: proprio too: the `aligned` channel warps with the car's own measured motion and
            #: refuses a call that does not carry it. Every other channel ignores it.
            seen = rt.observe(scan, pro)                # advances the occupancy channel, always
            mem = None if rt.scan is None or rt.scan.mem is None else rt.scan.mem.clone()
            if dial is not None:
                dial.redraw(new_ep)
            label = env.teacher_label(teacher, mu=None if dial is None else dial.value())
            valid = getattr(teacher, "last_label_valid", None)
            c = None if cond_fn is None else cond_fn()
            mu_t, opp_t = friction_target(env), opponent_target(env)
            student = None
            if beta < 1.0 or need_gap:
                student, _lp, rt.hidden = model.act(seen, pro, deterministic=True, h=rt.hidden)
            gap = None if student is None else (student - label).abs().mean(1)
            buf.add(scan[ids, 0], pro[ids], label[ids], new_ep[ids],
                    None if gap is None else gap[ids], None if mem is None else mem[ids],
                    valid=None if valid is None else valid[ids],
                    cond=None if c is None else c[ids], mu=mu_t[ids], opp=opp_t[ids])
            if beta >= 1.0:
                a = label                               # iteration 0 drives the teacher
            else:
                a_student = student
                if noise > 0:
                    a_student = (a_student + noise * torch.randn_like(a_student)).clamp(-1, 1)
                use_t = torch.rand(env.B, device=device) < beta
                a = torch.where(use_t[:, None], label, a_student)
            obs, rew, term, trunc, info = env.step(a)
            new_ep = torch.zeros(env.B, dtype=torch.bool, device=device)
            if "final" in info:
                new_ep[info["final"]["ids"]] = True
            rt.reset(new_ep)
    return buf


def imitation_loss(mu, lab, quantile_dim: int = -1, tau: float = 0.5):
    """Huber on the normalized command; optionally a pinball loss on one dimension.

    The envelope mode's grip belief is the one output whose errors are not symmetric: too low costs
    lap time, too high costs the car. Its label also cannot be known until the car has cornered hard
    enough to feel the floor, and what a symmetric loss learns for "cannot know" is the mean of the
    randomisation range -- optimistic for half the floors. tau < 0.5 makes "do not know" mean "assume
    the slippery end", and the belief then has to be *earned* upwards by evidence."""
    if quantile_dim < 0 or tau == 0.5:
        return F.smooth_l1_loss(mu, lab, beta=0.1)
    keep = [j for j in range(mu.shape[1]) if j != quantile_dim]
    e = lab[:, quantile_dim] - mu[:, quantile_dim]
    pin = torch.maximum(tau * e, (tau - 1.0) * e).mean()
    return (F.smooth_l1_loss(mu[:, keep], lab[:, keep], beta=0.1) * len(keep) + pin) / mu.shape[1]


def sequence_means(model, scan, pro, keep, burn: int = 0, cond=None):
    """(action means, friction predictions, opponent predictions) over a (L, m) block with the recurrence walked step by step
    from a zero state. The stem runs once for the whole block (as `ActorCritic.evaluate_sequence`
    does); the first `burn` steps only warm the state up and are left out."""
    L, m = scan.shape[:2]
    x, p = model.actor.embed(scan.reshape(L * m, *scan.shape[2:]), pro.reshape(L * m, -1))
    x = x.view(L, m, -1); p = p.view(L, m, -1)
    ha = model.actor.initial_hidden(m, x.device, x.dtype)
    out, grip, opp = [], [], []
    for t in range(L):
        if ha is not None:
            ha = ha * keep[t].to(x.dtype)[None, :, None]
        f, ha = model.actor.head(x[t], None if cond is None else cond[t], ha)
        if t >= burn:
            out.append(torch.tanh(model.actor.mu(f))); grip.append(model.actor.grip(torch.cat([f, p[t]], 1))[:, 0])
            opp.append(model.actor.opp(f))
    return torch.cat(out, 0), torch.cat(grip, 0), torch.cat(opp, 0)


def train_epochs(model, bufs, epochs, batch, device, opt, log, hard_frac: float = 0.0, hard_power: float = 1.0,
                 log_every: int = 25, chunk: int = 0, speed_loss: str = "symmetric",
                 v_max: float = 10.0, quantile_dim: int = -1, tau: float = 0.5,
                 aux_grip: float = 0.0, aux_opp: float = 0.0):
    n_stored = sum(len(b) for b in bufs)
    n_total = sum(b.valid_count for b in bufs)
    log({"dagger/valid_labels": n_total, "dagger/invalid_labels": n_stored - n_total})
    if not n_total:
        log({"dagger/skipped_no_valid_labels": 1, "dagger/optimizer_updates": 0})
        return 0.0  # Explicitly skipped, not an observed zero training loss.
    steps = max(1, int(epochs * n_total / batch))
    buffer_weights = torch.tensor([b.valid_count for b in bufs], dtype=torch.float)
    recurrent = model.actor.has_memory
    losses, knots, speeds = [], [], []
    for i in range(steps):
        b = bufs[torch.multinomial(buffer_weights, 1).item()]
        if recurrent:
            L = max(1, int(chunk))
            scan, pro, lab, keep, valid = b.sample_chunks(max(1, batch // L), L, device,
                                                        return_valid=True)
            if not valid.any():
                continue
            c, fric, opp_t = b.extras_at(*b.last_index, device)
            mu = actor_sequence(model.actor, scan, pro, keep,
                                cond=None if c is None else c.view(scan.shape[0], scan.shape[1], -1))
            grip = opp = None
            # Process every frame, then select targets before loss arithmetic (NaN invalid
            # labels must not contaminate either loss or gradients).
            valid = valid.reshape(-1)
            mu = mu[valid]
            lab = lab.reshape(-1, lab.shape[-1])[valid]
        else:
            scan, pro, lab = b.sample(batch, device, hard_frac, hard_power)
            c, fric, opp_t = b.extras_at(*b.last_index, device)
            mu, grip, opp = model.actor.forward_all(scan, pro, c)
        loss, part = plan_loss(mu, lab, speed_loss, v_max, parts=True)
        if aux_grip > 0 and grip is not None and fric is not None:
            # The friction the labels were built from, asked of the same features the action comes
            # from. Imitation alone averages it away (the "imitation gap").
            loss = loss + aux_grip * F.smooth_l1_loss(grip, fric, beta=0.25)
        if aux_opp > 0 and opp is not None and opp_t is not None:
            # Where the nearest car is and how fast it is closing. The teacher never looks at an
            # opponent, so this is a target rather than a label, scored only where one is in range.
            near = opp_t[:, 3:4]
            loss = loss + aux_opp * ((((opp - opp_t[:, :3]) ** 2).mean(1, keepdim=True) * near).sum()
                                     / near.sum().clamp_min(1.0))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0); opt.step()
        losses.append(loss.item()); knots.append(part["knot"].item()); speeds.append(part["speed"].item())
        if i % log_every == 0:
            # the two halves separately, because the whole point of `asym` is that they are not the
            # same quantity and a single number would hide which one moved
            log({"dagger/loss": float(np.mean(losses[-log_every:])),
                 "dagger/knot_loss": float(np.mean(knots[-log_every:])),
                 "dagger/speed_loss": float(np.mean(speeds[-log_every:])),
                 "dagger/train_step": i, "dagger/lr": opt.param_groups[0]["lr"]})
    log({"dagger/optimizer_updates": len(losses),
         "dagger/skipped_empty_batches": steps - len(losses)})
    return float(np.mean(losses[-500:])) if losses else 0.0


def build_teacher(kind: str, rls, env, a):
    """The teacher whose action becomes the label."""
    limits = getattr(a, "_teacher_limits", None) or {}
    base = common.make_teacher(rls, env, grip=a.teacher_grip, recover_time=a.teacher_recover_time, **limits)
    base.speed_scale = a.teacher_speed
    if kind == "raceline":
        return base, f"raceline teacher (grip {a.teacher_grip}, speed x{a.teacher_speed:g}, limits {limits or 'default'})"
    if kind == "rollout":
        from ..rollout_teacher import RolloutTeacher
        rt = RolloutTeacher(base, env, horizon_s=a.rollout_horizon, every=a.rollout_every,
                            graphs=a.rollout_graphs and not a.eager)
        where = "each env's layout line" if rt.layout is not None else "the raceline"
        return rt, (f"rollout teacher: {rt.K} candidates (offsets from {where} x speed scales), "
                    f"{rt.H} steps simulated with the opponents' own planners, a decision every {rt.every} steps")
    w = [float(x) for x in a.teacher_cost.split(",")] if a.teacher_cost else None
    cost = TeacherCost(*w) if w else TeacherCost()
    it = InteractiveTeacher(base, env,
                            offsets=[float(x) for x in a.teacher_offsets.split(",")],
                            speeds=[float(x) for x in a.teacher_speeds.split(",")],
                            horizon_s=a.teacher_horizon, cost=cost,
                            cand_iters=a.teacher_cand_iters,
                            future_model=a.opp_future_model)
    return it, (f"interactive teacher: {it.n_candidates} candidates "
                f"({len(it.offsets)} offsets x {len(it.speeds)} speeds + braking/evasion), {it.horizon_s:g} s horizon, "
                f"futures '{a.opp_future_model}', cost {cost}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"dagger_{time.strftime('%m%d_%H%M')}")
    ap.add_argument("--envs", type=int, default=1024); ap.add_argument("--tracks", default="train", help="'train', 'eval' or comma separated catalog names")
    ap.add_argument("--iters", type=int, default=8); ap.add_argument("--steps", type=int, default=250, help="learner steps per iteration (x envs/race-size labels), split across solo/traffic when enabled")
    ap.add_argument("--solo-fraction", type=float, default=0.0,
                    help="fraction of learner-labelled samples collected on genuinely empty tracks; [0,1)")
    ap.add_argument("--solo-tracks", default="",
                    help="optional solo map set; defaults to selected traffic bases; always removes obstacles")
    ap.add_argument("--epochs", type=float, default=3.0); ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--beta0", type=float, default=0.6)
    ap.add_argument("--speed-cap", type=float, default=8.0); ap.add_argument("--device", default="cuda")
    ap.add_argument("--action-mode", default="direct", choices=["direct", "plan"], help="plan: the student outputs a local trajectory (f1sim.mpc)")
    ap.add_argument("--teacher-speed", type=float, default=1.0, help="scale on the teacher's speed profile (0.9: fewer teacher crashes through the plan tracker)")
    ap.add_argument("--hist-len", type=int, default=0, help="proprio history rows in the observation")
    ap.add_argument("--scan-stack", type=int, default=3); ap.add_argument("--scan-stride", type=int, default=1)
    ap.add_argument("--scan-deltas", action="store_true", help="append temporal scan differences before the CNN")
    ap.add_argument("--temporal-encoder", choices=["cnn", "gru"], default="cnn")
    ap.add_argument("--scan-stem", choices=["plain", "resnet"], default="resnet",
                    help="scan encoder: 'resnet' adds GroupNorm/residual blocks, a whole-scan receptive field, "
                         "a beam-angle channel and a raw per-sector nearest-return bypass")
    ap.add_argument("--raceline-margin", type=float, default=None,
                    help="[m] free space the teacher's raceline keeps from the boundary (default 0.40). A "
                         "minimum-curvature line has no robustness budget: the teacher tracks it with ground-truth "
                         "pose and latency compensation, the student has neither")
    ap.add_argument("--speed-mode", choices=["linear", "envelope", "knots"], default="linear",
                    help="what the plan's speed dimensions mean (f1sim.mpc.SPEED_MODES). 'linear': two speeds, linear "
                         "between (every existing checkpoint). 'envelope': a grip belief a_hat and an end speed; the "
                         "profile follows the plan's own curvature. 'knots': a speed at every curvature knot")
    ap.add_argument("--grip-quantile", type=float, default=0.5,
                    help="envelope only: pinball-loss quantile for the grip-belief dimension. 0.5 is the plain Huber "
                         "loss; below it, a student that cannot tell the floor yet assumes the slippery end")
    ap.add_argument("--cond", choices=["none", "true_mu", "dial"], default="none",
                    help="condition the student on friction (learn/conditioning.py). 'true_mu' is a LAB ORACLE: the "
                         "input is privileged and does not exist on the car. It answers one question -- does a "
                         "student that is *told* the floor drive it? 'dial' is the deployable form: the student is "
                         "told the floor's friction minus a per-episode margin and the teacher drives for that same "
                         "number, so the input is a command an operator or a supervisor sets")
    ap.add_argument("--dial-margin", type=float, default=0.30, help="dial: largest margin below the true friction")
    ap.add_argument("--dial-exact", type=float, default=0.30, help="dial: share of episodes with no margin at all")
    ap.add_argument("--aux-grip", type=float, default=0.0,
                    help="weight of an auxiliary loss asking the actor's grip head for the env's friction, from the "
                         "same features the action is read from")
    ap.add_argument("--aux-opp", type=float, default=0.0,
                    help="weight of an auxiliary loss asking the actor's opponent head where the nearest car is and "
                         "how fast it is closing. Needs --race-size > 1. The teacher never looks at an opponent, so "
                         "it cannot demonstrate a pass or a yield -- but 'a car is there and it moves like this' is "
                         "read out of how the LiDAR returns shift, and the simulator knows the answer")
    ap.add_argument("--seq-len", type=int, default=0,
                    help="train on contiguous runs of this many steps per env (needs --memory gru to matter). 0 = "
                         "i.i.d. samples, which never contain both the corner that showed the grip and the next one")
    ap.add_argument("--seq-burn", type=int, default=0, help="leading steps of each run that only warm the state up")
    ap.add_argument("--raceline-objective", choices=["min_curvature", "min_time"], default=None,
                    help="line the teacher follows: 'min_time' refines the minimum-curvature line by descending "
                         "the lap time of the speed profile at the --teacher-a-* limits (default: min_curvature)")
    ap.add_argument("--teacher-a-lat", type=float, default=None, help="[m/s^2] lateral limit of the teacher's speed profile (default 6.0)")
    ap.add_argument("--teacher-a-acc", type=float, default=None, help="[m/s^2] drive limit of the profile (default 6.0)")
    ap.add_argument("--teacher-a-brake", type=float, default=None, help="[m/s^2] braking limit of the profile (default 3.0)")
    ap.add_argument("--teacher-grip", choices=["true", "nominal", "conservative"], default="true",
                    help="grip the teacher's speed profile assumes. 'true' is privileged, so identical scans get "
                         "speed labels up to 2.2x apart and the Huber loss can only fit their mean")
    ap.add_argument("--teacher-recover-time", type=float, default=0.0,
                    help="[s] >0: cap the teacher's commanded speed at what can still be steered back onto the lane "
                         "(v <= a_lat * t / heading_error). 0 keeps the old behaviour, where a car facing "
                         "backwards on the line is told to carry full racing speed")
    # ---- which teacher, and how it searches
    ap.add_argument("--rollout-horizon", type=float, default=1.5, metavar="S",
                    help="[s] --teacher rollout: how long each candidate is simulated in the shadow env")
    ap.add_argument("--rollout-graphs", type=int, default=1, choices=[0, 1],
                    help="--teacher rollout: run the shadow env under the CUDA-graph runtime (1) rather than "
                         "torch.compile. Measured 7.1 against 10.5 s a decision on the training floor, with "
                         "the shadow tracking the real env as closely (1.3 mm median after 10 steps)")
    ap.add_argument("--rollout-every", type=int, default=10, metavar="STEPS",
                    help="--teacher rollout: steps between decisions; the chosen candidate policy drives between")
    # ---- the floor and what a touch is, as in learn.ppo (same EnvConfig fields, same defaults)
    ap.add_argument("--procedural-obstacles", type=float, default=0.0, metavar="FRAC",
                    help="share of env resets that get a freshly drawn obstacle layout (learn.ppo)")
    ap.add_argument("--procedural-density", type=float, default=1.0, metavar="PER10M")
    ap.add_argument("--procedural-max-props", type=int, default=0, metavar="N")
    ap.add_argument("--procedural-raceline-margin", type=float, default=0.25, metavar="M")
    ap.add_argument("--procedural-raceline-corridor", choices=["on", "off"], default="on")
    ap.add_argument("--collision-mode", choices=["terminate", "soft"], default="terminate")
    ap.add_argument("--movable-obstacles", action="store_true", help="needs --collision-mode soft (learn.ppo)")
    ap.add_argument("--spawn-runway", type=float, default=0.0, metavar="M")
    ap.add_argument("--kind-mix-assign", choices=["partition", "redraw"], default="partition")
    ap.add_argument("--teacher", dest="teacher_kind", default="raceline", choices=["raceline", "interactive", "rollout"],
                    help="raceline: pure pursuit on the precomputed line, blind to the other cars (every DAgger "
                         "run before this one). interactive: f1sim.interactive_teacher, which scores a family of "
                         "plans against the opponents' predicted motion and labels with the argmin -- the only one "
                         "of the two that can demonstrate a pass")
    ap.add_argument("--teacher-horizon", type=float, default=1.0, metavar="S",
                    help="[s] how far the interactive teacher rolls each candidate out and how far ahead it reads "
                         "the opponents")
    ap.add_argument("--teacher-cand-iters", type=int, default=2, metavar="N",
                    help="Gauss-Newton iterations per candidate plan (the reference plan uses 6; the fit starts "
                         "from a pure-pursuit blend that is already close)")
    ap.add_argument("--teacher-offsets", default=",".join(f"{x:g}" for x in DEFAULT_OFFSETS),
                    metavar="A,B,C", help="[m left of the raceline] the candidate family's lateral offsets")
    ap.add_argument("--teacher-speeds", default=",".join(f"{x:g}" for x in DEFAULT_SPEEDS),
                    metavar="A,B,C", help="multipliers on the plan's speed targets")
    ap.add_argument("--teacher-cost", default="", metavar="PROG,WALL,OPP,CLEAR,SMOOTH",
                    help="the five cost weights (f1sim.interactive_teacher.TeacherCost); empty = its defaults")
    # ---- the student's inputs and architecture
    ap.add_argument("--opp-token", default="off", choices=OPP_TOKEN_MODES,
                    help="privileged opponent block in the observation. An ORACLE: the other cars' true relative "
                         "position ('pos'), velocity ('posvel') and future ('future'), from the simulator. "
                         "Refused by the exporter and by the ROS node; a checkpoint trained with it is a "
                         "measurement, not a policy")
    ap.add_argument("--opp-future-model", default=EnvConfig.opp_future_model, choices=list(OPP_FUTURE_MODELS),
                    help="which prediction the 'future' columns and the interactive teacher read "
                         "(f1sim.gym_env.OPP_FUTURE_MODELS)")
    ap.add_argument("--memory", default="off", choices=("off", "gru"),
                    help="gru: a recurrent student, trained by truncated BPTT over --chunk-length steps")
    ap.add_argument("--memory-hidden", type=int, default=128)
    ap.add_argument("--scan-channels", default="", metavar="A,B",
                    help=f"extra scan channels ({','.join(SCAN_CHANNELS)}), as in ppo.py")
    ap.add_argument("--scan-memory-tau", type=float, default=2.0, metavar="S")
    ap.add_argument("--speed-loss", default="symmetric", choices=["symmetric", "asym"],
                    help="loss on the plan's two SPEED targets (the curvature knots keep the Huber "
                         "either way). 'symmetric' is the loss every DAgger run before this flag "
                         "used and is byte-identical to it. 'asym' is quadratic above the teacher's "
                         "speed and linear below it, in m/s: with a mu-dependent label a symmetric "
                         "regression fits the conditional MEAN of speed labels that are 2.2x apart, "
                         "which is too fast on exactly the low-grip draws where too fast ends the "
                         "episode. The asymmetry moves that optimum to a low quantile, so a student "
                         "that cannot resolve grip backs off and one that can still reaches the "
                         f"limit at zero loss. Constants: w_over {W_OVER:g}, w_under {W_UNDER:g}, "
                         f"under power {UNDER_POW:g}")
    ap.add_argument("--chunk-length", type=int, default=16, metavar="N",
                    help="truncated-BPTT length for a recurrent student. The buffer is (steps, envs), so a chunk "
                         "is a slice of it; a recurrent policy trained one isolated step at a time is not the "
                         "policy that drives")
    ap.add_argument("--init", default="", metavar="CKPT",
                    help="warm start the student from this checkpoint. Memory, extra scan channels and the "
                         "privileged opponent block are added by name with zeroed new columns, so the student "
                         "starts as the checkpoint and learns to use what was added")
    ap.add_argument("--hard-frac", type=float, default=0.0,
                    help="fraction of each training batch drawn in proportion to the student-teacher gap at "
                         "collection time. 0 = uniform (previous behaviour). Measured motivation: the student "
                         "matches the teacher to within 2 %% on the teacher's own trajectory but is 4-10x further "
                         "off in the second before a collision, and a uniform loss barely sees those states")
    ap.add_argument("--hard-power", type=float, default=1.0, help="exponent on the gap when weighting")
    ap.add_argument("--keep-iters", type=int, default=5, help="aggregate the data of at most this many recent iterations (host RAM)")
    ap.add_argument("--start-iter", type=int, default=0,
                    help="index of the first iteration to run. For RESUMING a killed run from its last "
                         "checkpoint: beta, the exploration noise and the checkpoint names all key off the "
                         "true iteration index, so restarting a run that died after iteration 6 needs "
                         "--start-iter 7, not --iters 1 (which is iteration 0, where beta is 1.0 and the "
                         "TEACHER drives). What it cannot restore is the aggregated buffer: DAgger trains on "
                         "the last --keep-iters collections and those live in host RAM only, so a resumed "
                         "iteration trains on its own data alone. Declare that wherever the run is reported.")
    ap.add_argument("--log-every", type=int, default=25, help="training steps between W&B loss rows (was 200)")
    ap.add_argument("--eval-steps", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate every Nth iteration (the last one always). 1 = every "
                         "iteration, the old behaviour. The evaluation is a diagnostic and "
                         "cannot reach the training data (see rng_island), so this trades "
                         "diagnostic resolution for wall-clock and nothing else.")
    ap.add_argument("--wandb", default="online")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eager", action="store_true", help="disable simulator / tracker compilation (CPU smoke tests)")
    opp_cfg.add_arguments(ap)
    a = ap.parse_args()
    opp_cfg.validate(a)
    if a.keep_iters < 1:
        raise SystemExit("--keep-iters must be positive")
    token = validate_opp_token(a.opp_token)
    try:
        solo_steps, traffic_steps = collection_steps(a.steps, a.solo_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if a.solo_tracks and not solo_steps:
        raise SystemExit("--solo-tracks needs a positive --solo-fraction")
    if solo_steps and (a.race_size < 2 or a.teacher_kind != "interactive" or token != "off"):
        raise SystemExit("--solo-fraction needs --race-size >= 2, --teacher interactive and --opp-token off "
                         "so empty and traffic observations share the same actor contract")
    a.scan_channels = [c.strip() for c in str(a.scan_channels).split(",") if c.strip()]
    unknown = [c for c in a.scan_channels if c not in SCAN_CHANNELS]
    if unknown:
        raise SystemExit(f"--scan-channels {unknown}: known channels are {', '.join(SCAN_CHANNELS)}")
    if a.teacher_kind == "rollout" and a.action_mode != "plan":
        raise SystemExit("--teacher rollout needs --action-mode plan: its candidates are plans.")
    if a.teacher_kind == "interactive":
        if a.action_mode != "plan":
            raise SystemExit("--teacher interactive needs --action-mode plan: its candidates ARE plans, and in "
                             "the direct action space it would fall back to the raceline teacher silently.")
        if a.race_size < 2:
            raise SystemExit(f"--teacher interactive with --race-size {a.race_size}: with no other car its "
                             f"opponent term is identically zero and the run is a raceline run wearing "
                             f"another name.")
    if a.memory != "off" and a.hard_frac > 0:
        raise SystemExit("--hard-frac with --memory gru: the recurrent update draws contiguous chunks, not "
                         "single steps, so a per-step weighting has no sample to apply to. Drop one.")
    if a.envs % a.race_size:
        raise SystemExit(f"--envs {a.envs} is not a multiple of --race-size {a.race_size}.")
    if token != "off" and a.race_size < 2:
        raise SystemExit(f"--opp-token {a.opp_token} with --race-size {a.race_size}: the block describes the "
                         f"other cars of a race and there are none.")
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    names = common.track_names(a.tracks, seed=a.seed)
    need_rl = opp_cfg.needs_racelines(a) or True          # the teacher itself always needs one
    print(f"loading {len(names)} tracks + racelines ...", flush=True)
    # The line is optimised for the profile the teacher drives, as in learn.ppo. A merge (75fea37)
    # dropped these lines and left the flags parsed and ignored: every DAgger run since labelled
    # with the min-curvature line at the default 6 / 6 / 3 limits whatever it was asked for.
    limits = common.teacher_limits(a.teacher_a_lat, a.teacher_a_acc, a.teacher_a_brake)
    rl_kw = dict(limits)
    if a.raceline_margin is not None:
        rl_kw["margin"] = a.raceline_margin
    if a.raceline_objective is not None:
        rl_kw["objective"] = a.raceline_objective
    a._teacher_limits = limits
    tracks, rls = common.load_tracks(names, racelines=need_rl, **rl_kw)
    cfg = Config()
    if a.eager:
        cfg.sim.compile = False
    env = common.make_env(tracks, a.envs, device,
                          EnvConfig(speed_cap=a.speed_cap, action_mode=a.action_mode, hist_len=a.hist_len,
                                    scan_stack=a.scan_stack, scan_stride=a.scan_stride,
                                    opp_token=token, opp_future_model=a.opp_future_model,
                                    compile_tracker=not a.eager,
                                    **opp_cfg.env_kwargs(a),
                                    procedural_obstacles=a.procedural_obstacles,
                                    procedural_density=a.procedural_density,
                                    procedural_max_props=a.procedural_max_props,
                                    procedural_raceline_margin=a.procedural_raceline_margin,
                                    procedural_raceline_corridor=a.procedural_raceline_corridor,
                                    spawn_runway=a.spawn_runway,
                                    collision_mode=a.collision_mode,
                                    # nothing terminates under soft, so the batch would reset in
                                    # lockstep forever (learn.ppo does the same)
                                    stagger_first_episode=a.collision_mode == "soft",
                                    kind_mix_assign=a.kind_mix_assign,
                                    movable_obstacles=a.movable_obstacles),
                          cfg=cfg, seed=a.seed, rls=rls,
                          teacher_grip=a.teacher_grip, teacher_recover_time=a.teacher_recover_time,
                          teacher_limits=limits)
    print(f"opponents: {opp_cfg.describe(a)}", flush=True)
    teacher, teacher_desc = build_teacher(a.teacher_kind, rls, env, a)
    print(f"teacher: {teacher_desc}", flush=True)
    spec = common.obs_spec(env)
    solo_env = solo_teacher = None
    solo_names = []
    if solo_steps:
        solo_names = bare_track_names(common.track_names(a.solo_tracks, seed=a.seed) if a.solo_tracks else names)
        solo_tracks, solo_rls = common.load_tracks(solo_names, racelines=True, **rl_kw)
        if not solo_tracks or any(t.props for t in solo_tracks):
            raise SystemExit("solo tracks must be nonempty and contain no placed obstacles")
        # A separate simulator with M=1: no hidden/distant cars or invisible obstacle geometry.
        solo_env = common.make_env(solo_tracks, env.B // env.M, device,
            EnvConfig(speed_cap=a.speed_cap, action_mode=a.action_mode, hist_len=a.hist_len,
                      scan_stack=a.scan_stack, scan_stride=a.scan_stride, opp_token="off",
                      race_size=1, compile_tracker=not a.eager),
            cfg=replace(cfg, sim=replace(cfg.sim)), seed=a.seed + 1, rls=solo_rls,
            teacher_grip=a.teacher_grip, teacher_recover_time=a.teacher_recover_time)
        if common.obs_spec(solo_env) != spec:
            raise SystemExit("solo and traffic observation contracts differ")
        solo_teacher, _solo_desc = build_teacher("raceline", solo_rls, solo_env, a)
    mix = {"requested_solo_fraction": a.solo_fraction, "solo_fraction": solo_steps / a.steps,
           "solo_steps": solo_steps, "traffic_steps": traffic_steps,
           "learners_per_step": env.B // env.M, "solo_tracks": solo_names,
           "traffic_tracks": names,
           "traffic_loaded_tracks": [t.name for t in tracks],
           "solo_loaded_tracks": [t.name for t in solo_tracks] if solo_steps else [],
           "solo_teacher": "raceline" if solo_steps else None,
           "traffic_teacher": a.teacher_kind, "opponents": opp_cfg.describe(a),
           "opponent_slots": opp_cfg.slots_config(a)}
    print(f"collection: solo {solo_steps} / traffic {traffic_steps} steps x {env.B // env.M} learner(s) "
          f"(solo {mix['solo_fraction']:.1%}); opponent rows excluded", flush=True)
    priv_dim = env.privileged(env.reset()[1] and env.last_result).shape[1]
    chan = scan_channel_spec({"channels": a.scan_channels, "memory_tau_s": a.scan_memory_tau}) if a.scan_channels else None
    mem_spec = memory_spec(hidden_size=a.memory_hidden) if a.memory != "off" else None
    if a.init:
        # RESUME vs WARM START. `load_for_memory` adds a GRU / extra channels / an opponent token to
        # a checkpoint that has none, and it refuses -- correctly -- to touch one that already has
        # them, because that would re-initialise a trained path. A resumed arm (--start-iter > 0)
        # hands it exactly such a checkpoint, so the request has to go to the plain loader instead.
        prior = {}
        try:
            prior = (torch.load(a.init, map_location="cpu", weights_only=False) or {}).get("meta") or {}
        except Exception:
            prior = {}
        have_mem, have_chan = bool(prior.get("memory")), tuple(prior.get("scan_channels") or ())
        have_tok = prior.get("opp_token") or "off"
        resuming = have_mem and bool(mem_spec)
        if resuming and (tuple(chan or ()) != have_chan or (token != "off" and token != have_tok)):
            raise SystemExit(
                f"{os.path.basename(a.init)} already carries memory (so this is a resume) but the flags ask for "
                f"channels {tuple(chan or ())}/token {token} against its {have_chan}/{have_tok}. Adding a path to "
                f"a trained checkpoint and continuing its schedule are different operations; do one or the other.")
        if resuming:
            model, _extra = load_checkpoint(a.init, device, override={"priv_dim": priv_dim, "act_dim": env.act_dim})
            print(f"resume from {os.path.basename(a.init)} (iteration {_extra.get('iter') if isinstance(_extra, dict) else '?'} "
                  f"of {_extra.get('iters') if isinstance(_extra, dict) else '?'}); continuing at --start-iter {a.start_iter}",
                  flush=True)
        elif mem_spec or chan or token != "off":
            model, _extra, fresh = load_for_memory(a.init, device, memory=mem_spec, scan_channels=chan,
                                                   opp_token=(token if token != "off" else None),
                                                   override={"n_stack": spec.scan_stack,
                                                             "n_beams": spec.n_beams,
                                                             "priv_dim": priv_dim,
                                                             "act_dim": env.act_dim})
            print(f"warm start from {os.path.basename(a.init)}: {len(fresh)} fresh tensor(s)", flush=True)
        else:
            model, _extra = load_checkpoint(a.init, device, override={"priv_dim": priv_dim, "act_dim": env.act_dim})
        model = model.to(device)
    else:
        # The condition input as ec51507 built it; the merge 75fea37 dropped it, and a fresh student
        # under --cond was built unconditional and died at its first training batch.
        model = ActorCritic(spec.scan_stack, spec.n_beams, spec.proprio_dim, priv_dim, act_dim=env.act_dim,
                            scan_deltas=a.scan_deltas, temporal_encoder=a.temporal_encoder,
                            scan_stem=a.scan_stem, memory=mem_spec, scan_channels=chan,
                            **({} if a.cond == "none" else {"cond_dim": 1, "cond": cond_mod.spec_for(a.cond).to_meta()})
                            ).to(device)
    if int(model.meta["proprio_dim"]) != spec.proprio_dim:
        raise SystemExit(f"the student's proprio width is {model.meta['proprio_dim']} and this env produces "
                         f"{spec.proprio_dim}: --opp-token / --hist-len do not match the checkpoint.")
    cspec = None if a.cond == "none" else cond_mod.spec_for(a.cond)
    dial = cond_mod.DialDraw(env, a.dial_margin, a.dial_exact) if a.cond == "dial" else None
    cond_fn = None if cspec is None else (
        lambda: cond_mod.mu_to_c(dial.value() if dial is not None else env.sim.P["mu"], cspec))
    #: The action dimension whose error is not symmetric: too little grip costs lap time, too much
    #: costs the car. `--grip-quantile` below 0.5 makes "cannot tell yet" mean the slippery end.
    quantile_dim = N_KNOTS if a.speed_mode == "envelope" else -1
    opt = torch.optim.Adam(model.actor.parameters(), lr=a.lr)
    run = common.wandb_init(a.name, vars(a) | {"phase": "dagger", "tracks": names,
                                               "teacher_desc": teacher_desc, "collection_mix": mix,
                                               "opp_slots": opp_cfg.slots_config(a)}, group="dagger", mode=a.wandb)
    out = common.run_dir(a.name)
    # The run's own curve, next to the students. Written whatever `--wandb` is set to: the console's
    # dashboard reads this file, and a DAgger run's `iter` line was never parsed by anything.
    progress_log = common.ProgressLog(out)
    log = lambda d: run.log(d)
    env.sim.warmup()
    if solo_env is not None:
        solo_env.sim.warmup()
    buffer_iters = []
    bufs = []
    teacher_metrics = None
    m = None; m_iter = -1; t_teach = 0.0
    t0 = time.time()
    for it in range(a.start_iter, a.start_iter + a.iters):
        beta = 1.0 if it == 0 else a.beta0 * (0.5 ** (it - 1))
        tm = common.Timer()
        current = []
        counts = {}
        for cohort, active_env, active_teacher, n_steps in (
                ("traffic", env, teacher, traffic_steps),
                ("solo", solo_env, solo_teacher, solo_steps)):
            counts[cohort] = 0
            if not n_steps:
                continue
            buf = collect(active_env, model, active_teacher, n_steps, beta, device,
                          StepBuffer(spec.scan_stack, spec.scan_stride, a.scan_channels),
                          noise=0.05 if it else 0.0, need_gap=a.hard_frac > 0,
                          cond_fn=cond_fn, dial=dial).finalize()
            counts[cohort] = len(buf)
            current.append(buf)
        buffer_iters.append(current)
        buffer_iters = buffer_iters[-a.keep_iters:]
        bufs = [b for iteration in buffer_iters for b in iteration]
        t_col = tm.lap()        # host RAM: keep the last few iterations (14 GB laptop)
        loss = train_epochs(model, bufs, a.epochs, a.batch, device, opt, log, a.hard_frac, a.hard_power,
                            a.log_every, chunk=a.chunk_length, speed_loss=a.speed_loss,
                            v_max=env.ecfg.v_max_policy, quantile_dim=quantile_dim, tau=a.grip_quantile,
                            aux_grip=a.aux_grip, aux_opp=a.aux_opp); t_tr = tm.lap()
        # Diagnostics only, and fenced off from the training stream. `eval_every` skips the
        # measurement, never the collection or the training: 75 % of a calibrated iteration was
        # this rollout (873 s of 1165), and eight iterations of it is two hours of measuring a
        # student that gets scored properly after the arm finishes anyway.
        do_eval = (it % max(1, a.eval_every) == 0) or (it == a.start_iter + a.iters - 1)
        if do_eval:
            with rng_island(env):
                # `student_policy`, not `memory.policy_fn`: it is the same runtime for an unconditional
                # student and passes the dial (set to the true friction) to a conditional one, which
                # refuses to run without it. The merge 75fea37 swapped them.
                m = common.rollout_metrics(env, common.student_policy(model, env, device, deterministic=True),
                                           a.eval_steps, a.speed_cap, per_track=True)
            m_iter = it
        t_ev = tm.lap()
        if teacher_metrics is None:
            with rng_island(env):
                teacher_metrics = common.rollout_metrics(env, lambda o: env.teacher_label(teacher), a.eval_steps,
                                                         a.speed_cap, per_track=True)
        t_teach = tm.lap()
        scalar = lambda d: {k: v for k, v in d.items() if not isinstance(v, list)}      # per-track arrays stay out of W&B
        log({"dagger/iter": it, "dagger/beta": beta, "dagger/samples": sum(len(b) for b in bufs), "dagger/final_loss": loss,
             "dagger/solo_samples": counts["solo"], "dagger/traffic_samples": counts["traffic"],
             **({f"student/{k}": v for k, v in scalar(m).items()} if do_eval else {}),
             **{f"teacher/{k}": v for k, v in scalar(teacher_metrics).items()},
             "time/collect_s": t_col, "time/train_s": t_tr, "time/eval_s": t_ev,
             # separately, because it is paid once for the whole run and folding it into eval_s is
             # what made iteration 0 look like a 1165 s iteration when it is nothing of the kind
             "time/teacher_eval_s": t_teach, "time/elapsed_min": (time.time() - t0) / 60})
        # `m` is None until the first evaluation, and --eval-every can skip one, so this cannot
        # dereference it unconditionally the way it could when every iteration measured.
        worst = names[m["worst_track_index"]] if (m and m.get("worst_track_index", -1) >= 0) else "n/a"
        # The stable schema the console plots, then every other scalar the iteration measured. Same
        # shape as PPO's record and told apart by "kind", so one reader serves both.
        record = {"kind": "dagger", "iter": it, "total": a.start_iter + a.iters, "beta": beta,
                  "samples": sum(len(b) for b in bufs), "loss": loss,
                  "solo_samples": counts["solo"], "traffic_samples": counts["traffic"],
                  "collection_mix": mix,
                  "teacher_coll_per_km": teacher_metrics["collisions_per_km"],
                  "teacher_prog_mps": teacher_metrics["progress_rate_mps"],
                  "teacher_lap_s": teacher_metrics["lap_time_s"],
                  "wall_s": time.time() - t0, "worst_track": worst,
                  "collect_s": t_col, "train_s": t_tr, "eval_s": t_ev, "teacher_eval_s": t_teach}
        # Student rows ONLY on an iteration that measured. --eval-every can skip the evaluation, and
        # writing the previous iteration's numbers under this iteration's index would draw the
        # console a flat segment the run never produced. `metrics_from_iter` says which one it was.
        if do_eval:
            record.update({"student_coll_per_km": m["collisions_per_km"],
                           "student_prog_mps": m["progress_rate_mps"],
                           "student_lap_s": m["lap_time_s"]})
            record.update({f"student/{k}": v for k, v in scalar(m).items()})
        record["metrics_from_iter"] = m_iter
        record.update({f"teacher/{k}": v for k, v in scalar(teacher_metrics).items()})
        progress_log.write(record)
        stud = (f"student {m['collisions_per_km']:.1f} coll/km (worst {m['collisions_per_km_worst']:.1f} on {worst}) "
                f"prog {m['progress_rate_mps']:.2f} m/s lap {m['lap_time_s']:.1f} s"
                if do_eval else f"student not measured this iter (last at {m_iter})")
        print(f"iter {it}: beta {beta:.2f} samples {sum(len(b) for b in bufs)} loss {loss:.4f} | {stud} | "
              f"teacher {teacher_metrics['collisions_per_km']:.1f} coll/km (worst {teacher_metrics['collisions_per_km_worst']:.1f}) "
              f"lap {teacher_metrics['lap_time_s']:.1f} s | {t_col:.0f}+{t_tr:.0f}+{t_ev:.0f}"
              f"{f'+{t_teach:.0f} teach' if t_teach > 1 else ''} s", flush=True)
        meta = {"spec": spec.__dict__, "phase": "dagger", "run": a.name, "iter": it, "iters": a.start_iter + a.iters,
                "start_iter": a.start_iter,          # >0 means this checkpoint came from a resume
                "resumed_from": a.init if a.start_iter else None,
                "cap": a.speed_cap,          # the viewer defaults to the cap the policy was trained at
                "samples": sum(len(b) for b in bufs), "metrics": m, "metrics_from_iter": m_iter,
                "teacher": teacher_metrics,
                "action_mode": a.action_mode, "teacher_kind": a.teacher_kind, "teacher_desc": teacher_desc,
                "collection_mix": mix, "solo_samples": counts["solo"], "traffic_samples": counts["traffic"],
                "opp_token": token, "opp_future_model": a.opp_future_model,
                # declared before training and carried by every checkpoint, so an arm's loss is
                # never something that has to be reconstructed from a shell history
                "speed_loss": a.speed_loss,
                "speed_loss_constants": ({"w_over": W_OVER, "w_under": W_UNDER, "under_pow": UNDER_POW}
                                         if a.speed_loss == "asym" else None),
                "teacher_grip": a.teacher_grip, "teacher_speed": a.teacher_speed}
        save_checkpoint(os.path.join(out, f"student_it{it}.pt"), model, meta)
        save_checkpoint(os.path.join(out, "student_latest.pt"), model, meta)
    progress_log.close()
    run.finish()


if __name__ == "__main__":
    main()
