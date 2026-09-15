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
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..gym_env import EnvConfig, OPP_FUTURE_MODELS, OPP_TOKEN_MODES, opp_token_dim, opp_token_mode
from ..mpc import ACT_DIM, N_KNOTS
from ..interactive_teacher import DEFAULT_OFFSETS, DEFAULT_SPEEDS, InteractiveTeacher, TeacherCost
from ..params import Config
from . import common
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

    def add(self, scan_now, proprio, label, new_episode, gap=None, mem=None):
        self.scan.append(scan_now.to(torch.float16).cpu()); self.pro.append(proprio.to(torch.float16).cpu())   # a 1 s proprio history is 322 floats/sample
        self.lab.append(label.cpu()); self.newep.append(new_episode.cpu())
        self.gap.append((torch.zeros(label.shape[0]) if gap is None else gap.detach().float().cpu()))
        if self.keep_mem:
            if mem is None:
                raise ValueError("this buffer records the decayed occupancy channel and was handed none")
            self.mem.append(mem.to(torch.float16).cpu())

    def finalize(self):
        self.S = torch.stack(self.scan); self.P = torch.stack(self.pro); self.L = torch.stack(self.lab); self.N = torch.stack(self.newep)
        self.G = torch.stack(self.gap)                     # |student - teacher| at collection time
        self.M = torch.stack(self.mem) if self.keep_mem else None
        self.T, self.B = self.S.shape[:2]
        self.scan, self.pro, self.lab, self.newep, self.gap, self.mem = [], [], [], [], [], []
        return self

    def __len__(self):
        return self.T * self.B

    def samples_at(self, t, b, device):
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
                extra.append(self.M[t, b].to(device, torch.float32) if name == "memory" else scan_edges(now))
            scan = torch.cat([scan, torch.stack(extra, 1)], 1)
        return scan, self.P[t, b].to(device).float(), self.L[t, b].to(device)

    def sample(self, n, device, hard_frac: float = 0.0, power: float = 1.0):
        """hard_frac of the batch is drawn in proportion to how far the student was from the teacher
        when the state was collected.

        Measured motivation: on the teacher's own trajectory the student reproduces its action to
        within 2 %, but in the second before a collision the error is 4-10x that. A uniform loss over
        the buffer therefore optimises the on-line majority and leaves the recovery states -- the
        ones that actually end episodes -- effectively unweighted."""
        n_hard = int(n * hard_frac)
        t = torch.randint(self.T, (n - n_hard,)); b = torch.randint(self.B, (n - n_hard,))
        if n_hard:
            w = self.G.reshape(-1).double().clamp_min(0.0) ** power
            if float(w.sum()) <= 0:
                th = torch.randint(self.T, (n_hard,)); bh = torch.randint(self.B, (n_hard,))
            else:
                idx = torch.multinomial(w, n_hard, replacement=True)
                th, bh = idx // self.B, idx % self.B
            t = torch.cat([t, th]); b = torch.cat([b, bh])
        return self.samples_at(t, b, device)

    def sample_chunks(self, n: int, length: int, device):
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
        return shape(scan), shape(pro), shape(lab), keep


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


def actor_sequence(actor, scan, proprio, keep) -> torch.Tensor:
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
        f, h, _enc = actor.head(x[t], None, h, rows=None if rows is None else rows[t])
        feats.append(f)
    return torch.tanh(actor.mu(torch.cat(feats, 0)))


def collect(env, model, teacher, steps, beta, device, buf: StepBuffer, noise=0.0, need_gap=False):
    """Roll the env for `steps`, labelling every state with the teacher and storing it.

    The student is driven through its `PolicyRuntime`, so a recurrent or scan-augmented checkpoint
    drives here exactly as it drives in evaluation: the hidden state and the decayed occupancy are
    carried across steps and cleared at episode boundaries. For a feedforward student with no extra
    channels the runtime is inert and this is the loop it always was.
    """
    obs, info = env.reset()
    rt = runtime_for(model, env.B, device)
    new_ep = torch.ones(env.B, dtype=torch.bool, device=device)
    with torch.no_grad():
        for t in range(steps):
            scan, pro = flatten_obs(obs)
            #: proprio too: the `aligned` channel warps with the car's own measured motion and
            #: refuses a call that does not carry it. Every other channel ignores it.
            seen = rt.observe(scan, pro)                # advances the occupancy channel, always
            mem = None if rt.scan is None or rt.scan.mem is None else rt.scan.mem.clone()
            label = env.teacher_label(teacher)
            student = None
            if beta < 1.0 or need_gap:
                student, _lp, rt.hidden = model.act(seen, pro, deterministic=True, h=rt.hidden)
            gap = None if student is None else (student - label).abs().mean(1)
            buf.add(scan[:, 0], pro, label, new_ep, gap, mem)
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


def train_epochs(model, bufs, epochs, batch, device, opt, log, hard_frac: float = 0.0, hard_power: float = 1.0,
                 log_every: int = 25, chunk: int = 0, speed_loss: str = "symmetric",
                 v_max: float = 10.0):
    n_total = sum(len(b) for b in bufs)
    steps = max(1, int(epochs * n_total / batch))
    buffer_weights = torch.tensor([len(b) for b in bufs], dtype=torch.float)
    recurrent = model.actor.has_memory
    losses, knots, speeds = [], [], []
    for i in range(steps):
        b = bufs[torch.multinomial(buffer_weights, 1).item()]
        if recurrent:
            L = max(1, int(chunk))
            scan, pro, lab, keep = b.sample_chunks(max(1, batch // L), L, device)
            mu = actor_sequence(model.actor, scan, pro, keep)
            lab = lab.reshape(-1, lab.shape[-1])
        else:
            scan, pro, lab = b.sample(batch, device, hard_frac, hard_power)
            mu = model.actor(scan, pro)
        loss, part = plan_loss(mu, lab, speed_loss, v_max, parts=True)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0); opt.step()
        losses.append(loss.item()); knots.append(part["knot"].item()); speeds.append(part["speed"].item())
        if i % log_every == 0:
            # the two halves separately, because the whole point of `asym` is that they are not the
            # same quantity and a single number would hide which one moved
            log({"dagger/loss": float(np.mean(losses[-log_every:])),
                 "dagger/knot_loss": float(np.mean(knots[-log_every:])),
                 "dagger/speed_loss": float(np.mean(speeds[-log_every:])),
                 "dagger/train_step": i, "dagger/lr": opt.param_groups[0]["lr"]})
    return float(np.mean(losses[-500:]))


def build_teacher(kind: str, rls, env, a):
    """The teacher whose action becomes the label."""
    base = common.make_teacher(rls, env, grip=a.teacher_grip, recover_time=a.teacher_recover_time)
    base.speed_scale = a.teacher_speed
    if kind == "raceline":
        return base, f"raceline teacher (grip {a.teacher_grip}, speed x{a.teacher_speed:g})"
    w = [float(x) for x in a.teacher_cost.split(",")] if a.teacher_cost else None
    cost = TeacherCost(*w) if w else TeacherCost()
    it = InteractiveTeacher(base, env,
                            offsets=[float(x) for x in a.teacher_offsets.split(",")],
                            speeds=[float(x) for x in a.teacher_speeds.split(",")],
                            horizon_s=a.teacher_horizon, cost=cost,
                            cand_iters=a.teacher_cand_iters,
                            future_model=a.opp_future_model)
    return it, (f"interactive teacher: {it.n_candidates} candidates "
                f"({len(it.offsets)} offsets x {len(it.speeds)} speeds), {it.horizon_s:g} s horizon, "
                f"futures '{a.opp_future_model}', cost {cost}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"dagger_{time.strftime('%m%d_%H%M')}")
    ap.add_argument("--envs", type=int, default=1024); ap.add_argument("--tracks", default="train", help="'train', 'eval' or comma separated catalog names")
    ap.add_argument("--iters", type=int, default=8); ap.add_argument("--steps", type=int, default=250, help="env steps per iteration (x envs samples)")
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
    ap.add_argument("--teacher-grip", choices=["true", "nominal", "conservative"], default="true",
                    help="grip the teacher's speed profile assumes. 'true' is privileged, so identical scans get "
                         "speed labels up to 2.2x apart and the Huber loss can only fit their mean")
    ap.add_argument("--teacher-recover-time", type=float, default=0.0,
                    help="[s] >0: cap the teacher's commanded speed at what can still be steered back onto the lane "
                         "(v <= a_lat * t / heading_error). 0 keeps the old behaviour, where a car facing "
                         "backwards on the line is told to carry full racing speed")
    # ---- which teacher, and how it searches
    ap.add_argument("--teacher", dest="teacher_kind", default="raceline", choices=["raceline", "interactive"],
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
    ap.add_argument("--opp-token", default="off", choices=[m for m in OPP_TOKEN_MODES if m != ""],
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
    ap.add_argument("--log-every", type=int, default=25, help="training steps between W&B loss rows (was 200)")
    ap.add_argument("--eval-steps", type=int, default=800); ap.add_argument("--wandb", default="online")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eager", action="store_true", help="disable simulator / tracker compilation (CPU smoke tests)")
    opp_cfg.add_arguments(ap)
    a = ap.parse_args()
    opp_cfg.validate(a)
    token = opp_token_mode(a.opp_token)
    a.scan_channels = [c.strip() for c in str(a.scan_channels).split(",") if c.strip()]
    unknown = [c for c in a.scan_channels if c not in SCAN_CHANNELS]
    if unknown:
        raise SystemExit(f"--scan-channels {unknown}: known channels are {', '.join(SCAN_CHANNELS)}")
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
    if token and a.race_size < 2:
        raise SystemExit(f"--opp-token {a.opp_token} with --race-size {a.race_size}: the block describes the "
                         f"other cars of a race and there are none.")
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    names = common.track_names(a.tracks, seed=a.seed)
    need_rl = opp_cfg.needs_racelines(a) or True          # the teacher itself always needs one
    print(f"loading {len(names)} tracks + racelines ...", flush=True)
    tracks, rls = common.load_tracks(names, racelines=need_rl,
                                     **({} if a.raceline_margin is None else {"margin": a.raceline_margin}))
    cfg = Config()
    if a.eager:
        cfg.sim.compile = False
    env = common.make_env(tracks, a.envs, device,
                          EnvConfig(speed_cap=a.speed_cap, action_mode=a.action_mode, hist_len=a.hist_len,
                                    scan_stack=a.scan_stack, scan_stride=a.scan_stride,
                                    opp_token=token, opp_future_model=a.opp_future_model,
                                    compile_tracker=not a.eager,
                                    **opp_cfg.env_kwargs(a)),
                          cfg=cfg, seed=a.seed, rls=rls,
                          teacher_grip=a.teacher_grip, teacher_recover_time=a.teacher_recover_time)
    print(f"opponents: {opp_cfg.describe(a)}", flush=True)
    teacher, teacher_desc = build_teacher(a.teacher_kind, rls, env, a)
    print(f"teacher: {teacher_desc}", flush=True)
    spec = common.obs_spec(env)
    priv_dim = env.privileged(env.reset()[1] and env.last_result).shape[1]
    chan = scan_channel_spec({"channels": a.scan_channels, "memory_tau_s": a.scan_memory_tau}) if a.scan_channels else None
    mem_spec = memory_spec(hidden_size=a.memory_hidden) if a.memory != "off" else None
    if a.init:
        if mem_spec or chan or token:
            model, _extra, fresh = load_for_memory(a.init, device, memory=mem_spec, scan_channels=chan,
                                                   opp_token_dim=opp_token_dim(token),
                                                   override={"n_stack": spec.scan_stack,
                                                             "n_beams": spec.n_beams,
                                                             "priv_dim": priv_dim,
                                                             "act_dim": env.act_dim})
            print(f"warm start from {os.path.basename(a.init)}: {len(fresh)} fresh tensor(s)", flush=True)
        else:
            model, _extra = load_checkpoint(a.init, device, override={"priv_dim": priv_dim, "act_dim": env.act_dim})
        model = model.to(device)
    else:
        model = ActorCritic(spec.scan_stack, spec.n_beams, spec.proprio_dim, priv_dim, act_dim=env.act_dim,
                            scan_deltas=a.scan_deltas, temporal_encoder=a.temporal_encoder,
                            scan_stem=a.scan_stem, memory=mem_spec, scan_channels=chan).to(device)
    if int(model.meta["proprio_dim"]) != spec.proprio_dim:
        raise SystemExit(f"the student's proprio width is {model.meta['proprio_dim']} and this env produces "
                         f"{spec.proprio_dim}: --opp-token / --hist-len do not match the checkpoint.")
    opt = torch.optim.Adam(model.actor.parameters(), lr=a.lr)
    run = common.wandb_init(a.name, vars(a) | {"phase": "dagger", "tracks": names,
                                               "teacher_desc": teacher_desc}, group="dagger", mode=a.wandb)
    out = common.run_dir(a.name)
    log = lambda d: run.log(d)
    env.sim.warmup()
    bufs = []
    teacher_metrics = None
    t0 = time.time()
    for it in range(a.iters):
        beta = 1.0 if it == 0 else a.beta0 * (0.5 ** (it - 1))
        tm = common.Timer()
        buf = collect(env, model, teacher, a.steps, beta, device,
                      StepBuffer(spec.scan_stack, spec.scan_stride, a.scan_channels),
                      noise=0.05 if it else 0.0, need_gap=a.hard_frac > 0).finalize()
        bufs.append(buf); bufs = bufs[-a.keep_iters:]; t_col = tm.lap()        # host RAM: keep the last few iterations (14 GB laptop)
        loss = train_epochs(model, bufs, a.epochs, a.batch, device, opt, log, a.hard_frac, a.hard_power,
                            a.log_every, chunk=a.chunk_length, speed_loss=a.speed_loss,
                            v_max=env.ecfg.v_max_policy); t_tr = tm.lap()
        m = common.rollout_metrics(env, memory_policy_fn(model, env.B, device=device, deterministic=True), a.eval_steps, a.speed_cap,
                                   per_track=True); t_ev = tm.lap()
        if teacher_metrics is None:
            teacher_metrics = common.rollout_metrics(env, lambda o: env.teacher_label(teacher), a.eval_steps, a.speed_cap,
                                                     per_track=True)
        scalar = lambda d: {k: v for k, v in d.items() if not isinstance(v, list)}      # per-track arrays stay out of W&B
        log({"dagger/iter": it, "dagger/beta": beta, "dagger/samples": sum(len(b) for b in bufs), "dagger/final_loss": loss,
             **{f"student/{k}": v for k, v in scalar(m).items()}, **{f"teacher/{k}": v for k, v in scalar(teacher_metrics).items()},
             "time/collect_s": t_col, "time/train_s": t_tr, "time/eval_s": t_ev, "time/elapsed_min": (time.time() - t0) / 60})
        worst = names[m["worst_track_index"]] if m.get("worst_track_index", -1) >= 0 else "n/a"
        print(f"iter {it}: beta {beta:.2f} samples {sum(len(b) for b in bufs)} loss {loss:.4f} | student {m['collisions_per_km']:.1f} coll/km "
              f"(worst {m['collisions_per_km_worst']:.1f} on {worst}) prog {m['progress_rate_mps']:.2f} m/s lap {m['lap_time_s']:.1f} s | "
              f"teacher {teacher_metrics['collisions_per_km']:.1f} coll/km (worst {teacher_metrics['collisions_per_km_worst']:.1f}) "
              f"lap {teacher_metrics['lap_time_s']:.1f} s | {t_col:.0f}+{t_tr:.0f}+{t_ev:.0f} s", flush=True)
        meta = {"spec": spec.__dict__, "phase": "dagger", "run": a.name, "iter": it, "iters": a.iters,
                "cap": a.speed_cap,          # the viewer defaults to the cap the policy was trained at
                "samples": sum(len(b) for b in bufs), "metrics": m, "teacher": teacher_metrics,
                "action_mode": a.action_mode, "teacher_kind": a.teacher_kind, "teacher_desc": teacher_desc,
                "opp_token": token, "opp_future_model": a.opp_future_model,
                # declared before training and carried by every checkpoint, so an arm's loss is
                # never something that has to be reconstructed from a shell history
                "speed_loss": a.speed_loss,
                "speed_loss_constants": ({"w_over": W_OVER, "w_under": W_UNDER, "under_pow": UNDER_POW}
                                         if a.speed_loss == "asym" else None),
                "teacher_grip": a.teacher_grip, "teacher_speed": a.teacher_speed}
        save_checkpoint(os.path.join(out, f"student_it{it}.pt"), model, meta)
        save_checkpoint(os.path.join(out, "student_latest.pt"), model, meta)
    run.finish()


if __name__ == "__main__":
    main()
