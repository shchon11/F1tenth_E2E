"""DAgger: distill the privileged raceline teacher into the LiDAR-only student.

Iteration 0 collects with the teacher driving; later iterations let the student drive (with
probability 1 - beta) while the teacher labels every visited state. All data is aggregated;
the student is trained with a Huber loss on the normalized (steer, speed) command.
Scans are stored once per step (fp16, CPU) and stacks are rebuilt at sampling time.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..gym_env import EnvConfig
from ..mpc import N_KNOTS
from . import common
from . import conditioning as cond_mod
from .model import ActorCritic, save_checkpoint
from .obs import PROPRIO_KEYS, flatten_obs


class StepBuffer:
    """Per-step storage; a training sample (t, b) rebuilds its scan stack from t, t-1, t-2 of env b."""

    def __init__(self, k: int, stride: int = 1):
        self.k, self.stride = k, stride
        self.scan, self.pro, self.lab, self.newep, self.gap, self.cond, self.mu = [], [], [], [], [], [], []

    def add(self, scan_now, proprio, label, new_episode, gap=None, cond=None, mu=None):
        self.scan.append(scan_now.to(torch.float16).cpu()); self.pro.append(proprio.to(torch.float16).cpu())   # a 1 s proprio history is 322 floats/sample
        self.lab.append(label.cpu()); self.newep.append(new_episode.cpu())
        self.gap.append((torch.zeros(label.shape[0]) if gap is None else gap.detach().float().cpu()))
        if cond is not None:
            self.cond.append(cond.detach().float().cpu())
        if mu is not None:
            self.mu.append(mu.detach().float().cpu())

    def finalize(self):
        self.S = torch.stack(self.scan); self.P = torch.stack(self.pro); self.L = torch.stack(self.lab); self.N = torch.stack(self.newep)
        self.G = torch.stack(self.gap)                     # |student - teacher| at collection time
        self.C = torch.stack(self.cond) if self.cond else None      # (T,B,D) what the student was conditioned on
        self.M = torch.stack(self.mu) if self.mu else None          # (T,B) normalised friction, the auxiliary target
        self.T, self.B = self.S.shape[:2]
        self.scan, self.pro, self.lab, self.newep, self.gap, self.cond, self.mu = [], [], [], [], [], [], []
        return self

    def extras_at(self, t, b, device):
        """(conditioning, friction target) for the same rows `samples_at` returns; None where not stored."""
        return (None if self.C is None else self.C[t, b].to(device), None if self.M is None else self.M[t, b].to(device))

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
        self.last_index = (t, b)                           # for `extras_at`: the rows just drawn
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


def friction_target(env) -> torch.Tensor:
    """The env's friction on the conditioning module's own scale, (mu - 1) / 0.25: one number, one meaning."""
    return (env.sim.P["mu"].reshape(-1) - cond_mod.MU_OFFSET) / cond_mod.MU_SCALE


def collect(env, model, teacher, steps, beta, device, buf: StepBuffer, noise=0.0, need_gap=False, cond_fn=None,
            dial: "cond_mod.DialDraw | None" = None):
    obs, info = env.reset()
    new_ep = torch.ones(env.B, dtype=torch.bool, device=device)
    h = model.initial_hidden(env.B, device)                # None for a feedforward student
    with torch.no_grad():
        for t in range(steps):
            scan, pro = flatten_obs(obs)
            if dial is not None:
                dial.redraw(new_ep)
            label = env.teacher_label(teacher, mu=None if dial is None else dial.value())
            if h is not None:
                h = h.reset(new_ep)
            c = None if cond_fn is None else cond_fn()
            mu_t = friction_target(env)
            if beta >= 1.0:
                # iteration 0 drives the teacher; query the student anyway when the gap is needed
                gap = None
                if need_gap:
                    a_s, _, h = model.act(scan, pro, deterministic=True, c=c, h=h)
                    gap = (a_s - label).abs().mean(1)
                buf.add(scan[:, 0], pro, label, new_ep, gap, c, mu_t)
                a = label
            else:
                a_student, _, h = model.act(scan, pro, deterministic=True, c=c, h=h)
                buf.add(scan[:, 0], pro, label, new_ep, (a_student - label).abs().mean(1), c, mu_t)
                if noise > 0:
                    a_student = (a_student + noise * torch.randn_like(a_student)).clamp(-1, 1)
                use_t = torch.rand(env.B, device=device) < beta
                a = torch.where(use_t[:, None], label, a_student)
            obs, rew, term, trunc, info = env.step(a)
            new_ep = torch.zeros(env.B, dtype=torch.bool, device=device)
            if "final" in info:
                new_ep[info["final"]["ids"]] = True
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
    """(action means, friction predictions) over a (L, m) block with the recurrence walked step by step
    from a zero state. The stem runs once for the whole block (as `ActorCritic.evaluate_sequence`
    does); the first `burn` steps only warm the state up and are left out."""
    L, m = scan.shape[:2]
    x, p = model.actor.embed(scan.reshape(L * m, *scan.shape[2:]), pro.reshape(L * m, -1))
    x = x.view(L, m, -1); p = p.view(L, m, -1)
    ha = model.actor.initial_hidden(m, x.device, x.dtype)
    out, grip = [], []
    for t in range(L):
        if ha is not None:
            ha = ha * keep[t].to(x.dtype)[None, :, None]
        f, ha = model.actor.head(x[t], None if cond is None else cond[t], ha)
        if t >= burn:
            out.append(torch.tanh(model.actor.mu(f))); grip.append(model.actor.grip(torch.cat([f, p[t]], 1))[:, 0])
    return torch.cat(out, 0), torch.cat(grip, 0)


def train_epochs(model, bufs, epochs, batch, device, opt, log, hard_frac: float = 0.0, hard_power: float = 1.0,
                 log_every: int = 25, seq_len: int = 0, seq_burn: int = 0, quantile_dim: int = -1, tau: float = 0.5,
                 aux_grip: float = 0.0):
    n_total = sum(len(b) for b in bufs)
    steps = max(1, int(epochs * n_total / batch))
    buffer_weights = torch.tensor([len(b) for b in bufs], dtype=torch.float)
    losses = []
    for i in range(steps):
        b = bufs[torch.multinomial(buffer_weights, 1).item()]
        if seq_len > 0:
            # the same number of labelled rows per step as the i.i.d. path, in runs of seq_len
            L = min(seq_len, b.T); burn = min(seq_burn, L - 1)
            m = max(1, batch // (L - burn))
            scan, pro, lab, keep = b.sample_sequences(m, L, device)
            c, fric = b.extras_at(*b.last_index, device)
            c = None if c is None else c.view(L, m, -1)
            mu, grip = sequence_means(model, scan, pro, keep, burn, c)
            lab = lab[burn:].reshape(-1, lab.shape[-1])
            fric = None if fric is None else fric.view(L, m)[burn:].reshape(-1)
        else:
            scan, pro, lab = b.sample(batch, device, hard_frac, hard_power)
            c, fric = b.extras_at(*b.last_index, device)
            mu, grip, _ = model.actor.forward_all(scan, pro, c)
        loss = imitation_loss(mu, lab, quantile_dim, tau)
        if aux_grip > 0:
            # The friction the labels were built from, asked of the same features the action comes
            # from. Imitation alone averages it away (the "imitation gap"); this is the dense signal
            # that only falls when the representation actually holds it.
            aux = F.smooth_l1_loss(grip, fric, beta=0.25)
            loss = loss + aux_grip * aux
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if i % log_every == 0:
            log({"dagger/loss": float(np.mean(losses[-log_every:])), "dagger/train_step": i, "dagger/lr": opt.param_groups[0]["lr"],
                 **({"dagger/aux_grip": float(aux.detach())} if aux_grip > 0 else {})})
    return float(np.mean(losses[-500:]))


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
    ap.add_argument("--memory", choices=["none", "gru"], default="none", help="recurrent student (learn/memory.py)")
    ap.add_argument("--memory-hidden", type=int, default=128)
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
    ap.add_argument("--hard-frac", type=float, default=0.0,
                    help="fraction of each training batch drawn in proportion to the student-teacher gap at "
                         "collection time. 0 = uniform (previous behaviour). Measured motivation: the student "
                         "matches the teacher to within 2 %% on the teacher's own trajectory but is 4-10x further "
                         "off in the second before a collision, and a uniform loss barely sees those states")
    ap.add_argument("--hard-power", type=float, default=1.0, help="exponent on the gap when weighting")
    ap.add_argument("--keep-iters", type=int, default=5, help="aggregate the data of at most this many recent iterations (host RAM)")
    ap.add_argument("--log-every", type=int, default=25, help="training steps between W&B loss rows (was 200)")
    ap.add_argument("--eval-steps", type=int, default=800); ap.add_argument("--wandb", default="online")
    a = ap.parse_args()
    device = torch.device(a.device)
    names = common.track_names(a.tracks)
    print(f"loading {len(names)} tracks + racelines ...", flush=True)
    limits = common.teacher_limits(a.teacher_a_lat, a.teacher_a_acc, a.teacher_a_brake)
    rl_kw = dict(limits)                                   # the line is optimised for the profile the teacher drives
    if a.raceline_margin is not None:
        rl_kw["margin"] = a.raceline_margin
    if a.raceline_objective is not None:
        rl_kw["objective"] = a.raceline_objective
    tracks, rls = common.load_tracks(names, racelines=True, **rl_kw)
    if a.speed_mode != "linear" and a.action_mode != "plan":
        ap.error("--speed-mode is a property of the plan action space; add --action-mode plan")
    if a.seq_len > 0 and a.hard_frac > 0:
        ap.error("--hard-frac re-weights single samples and --seq-len draws whole runs; pick one")
    plan_a_brake = limits.get("a_brake", 3.0)             # the envelope's backward pass brakes like the teacher's profile
    env = common.make_env(tracks, a.envs, device, EnvConfig(speed_cap=a.speed_cap, action_mode=a.action_mode, hist_len=a.hist_len,
                                                             scan_stack=a.scan_stack, scan_stride=a.scan_stride,
                                                             speed_mode=a.speed_mode, plan_a_brake=plan_a_brake))
    teacher = common.make_teacher(rls, env, grip=a.teacher_grip, recover_time=a.teacher_recover_time, **limits); teacher.speed_scale = a.teacher_speed
    spec = common.obs_spec(env)
    priv_dim = env.privileged(env.reset()[1] and env.last_result).shape[1]
    model = ActorCritic(spec.scan_stack, spec.n_beams, spec.proprio_dim, priv_dim, act_dim=env.act_dim,
                        scan_deltas=a.scan_deltas, temporal_encoder=a.temporal_encoder,
                        scan_stem=a.scan_stem,
                        memory=({"kind": "gru", "hidden_size": a.memory_hidden, "critic": "none"} if a.memory == "gru" else None),
                        **({} if a.cond == "none" else {"cond_dim": 1, "cond": cond_mod.spec_for(a.cond).to_meta()})
                        ).to(device)
    cspec = None if a.cond == "none" else cond_mod.spec_for(a.cond)
    dial = cond_mod.DialDraw(env, a.dial_margin, a.dial_exact) if a.cond == "dial" else None
    cond_fn = None if cspec is None else (lambda: cond_mod.mu_to_c(dial.value() if dial is not None else env.sim.P["mu"], cspec))

    quantile_dim = N_KNOTS if a.speed_mode == "envelope" else -1
    opt = torch.optim.Adam(model.actor.parameters(), lr=a.lr)
    run = common.wandb_init(a.name, vars(a) | {"phase": "dagger", "tracks": names}, group="dagger", mode=a.wandb)
    out = common.run_dir(a.name)
    log = lambda d: run.log(d)
    env.sim.warmup()
    bufs = []
    teacher_metrics = None
    t0 = time.time()
    for it in range(a.iters):
        beta = 1.0 if it == 0 else a.beta0 * (0.5 ** (it - 1))
        tm = common.Timer()
        buf = collect(env, model, teacher, a.steps, beta, device, StepBuffer(spec.scan_stack, spec.scan_stride),
                      noise=0.05 if it else 0.0, need_gap=a.hard_frac > 0, cond_fn=cond_fn, dial=dial).finalize()
        bufs.append(buf); bufs = bufs[-a.keep_iters:]; t_col = tm.lap()        # host RAM: keep the last few iterations (14 GB laptop)
        loss = train_epochs(model, bufs, a.epochs, a.batch, device, opt, log, a.hard_frac, a.hard_power, a.log_every,
                            seq_len=a.seq_len, seq_burn=a.seq_burn, quantile_dim=quantile_dim, tau=a.grip_quantile,
                            aux_grip=a.aux_grip); t_tr = tm.lap()
        m = common.rollout_metrics(env, common.student_policy(model, env, device), a.eval_steps, a.speed_cap, per_track=True); t_ev = tm.lap()
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
                "samples": sum(len(b) for b in bufs), "metrics": m, "teacher": teacher_metrics, "action_mode": a.action_mode,
                "speed_mode": a.speed_mode, "plan_a_brake": plan_a_brake, "teacher_limits": limits,
                "raceline_objective": a.raceline_objective or "min_curvature"}
        save_checkpoint(os.path.join(out, f"student_it{it}.pt"), model, meta)
        save_checkpoint(os.path.join(out, "student_latest.pt"), model, meta)
    run.finish()


if __name__ == "__main__":
    main()
