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
from . import common
from .model import ActorCritic, save_checkpoint
from .obs import PROPRIO_KEYS, flatten_obs


class StepBuffer:
    """Per-step storage; a training sample (t, b) rebuilds its scan stack from t, t-1, t-2 of env b."""

    def __init__(self, k: int, stride: int = 1):
        self.k, self.stride = k, stride
        self.scan, self.pro, self.lab, self.newep, self.gap = [], [], [], [], []

    def add(self, scan_now, proprio, label, new_episode, gap=None):
        self.scan.append(scan_now.to(torch.float16).cpu()); self.pro.append(proprio.to(torch.float16).cpu())   # a 1 s proprio history is 322 floats/sample
        self.lab.append(label.cpu()); self.newep.append(new_episode.cpu())
        self.gap.append((torch.zeros(label.shape[0]) if gap is None else gap.detach().float().cpu()))

    def finalize(self):
        self.S = torch.stack(self.scan); self.P = torch.stack(self.pro); self.L = torch.stack(self.lab); self.N = torch.stack(self.newep)
        self.G = torch.stack(self.gap)                     # |student - teacher| at collection time
        self.T, self.B = self.S.shape[:2]
        self.scan, self.pro, self.lab, self.newep, self.gap = [], [], [], [], []
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


def collect(env, model, teacher, steps, beta, device, buf: StepBuffer, noise=0.0, need_gap=False):
    obs, info = env.reset()
    new_ep = torch.ones(env.B, dtype=torch.bool, device=device)
    with torch.no_grad():
        for t in range(steps):
            scan, pro = flatten_obs(obs)
            label = env.teacher_label(teacher)
            if beta >= 1.0:
                # iteration 0 drives the teacher; query the student anyway when the gap is needed
                gap = (model.act(scan, pro, deterministic=True)[0] - label).abs().mean(1) if need_gap else None
                buf.add(scan[:, 0], pro, label, new_ep, gap)
                a = label
            else:
                a_student, _ = model.act(scan, pro, deterministic=True)
                buf.add(scan[:, 0], pro, label, new_ep, (a_student - label).abs().mean(1))
                if noise > 0:
                    a_student = (a_student + noise * torch.randn_like(a_student)).clamp(-1, 1)
                use_t = torch.rand(env.B, device=device) < beta
                a = torch.where(use_t[:, None], label, a_student)
            obs, rew, term, trunc, info = env.step(a)
            new_ep = torch.zeros(env.B, dtype=torch.bool, device=device)
            if "final" in info:
                new_ep[info["final"]["ids"]] = True
    return buf


def train_epochs(model, bufs, epochs, batch, device, opt, log, hard_frac: float = 0.0, hard_power: float = 1.0,
                 log_every: int = 25):
    n_total = sum(len(b) for b in bufs)
    steps = max(1, int(epochs * n_total / batch))
    buffer_weights = torch.tensor([len(b) for b in bufs], dtype=torch.float)
    losses = []
    for i in range(steps):
        b = bufs[torch.multinomial(buffer_weights, 1).item()]
        scan, pro, lab = b.sample(batch, device, hard_frac, hard_power)
        mu = model.actor(scan, pro)
        loss = F.smooth_l1_loss(mu, lab, beta=0.1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if i % log_every == 0:
            log({"dagger/loss": float(np.mean(losses[-log_every:])), "dagger/train_step": i, "dagger/lr": opt.param_groups[0]["lr"]})
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
    tracks, rls = common.load_tracks(names, racelines=True,
                                     **({} if a.raceline_margin is None else {"margin": a.raceline_margin}))
    env = common.make_env(tracks, a.envs, device, EnvConfig(speed_cap=a.speed_cap, action_mode=a.action_mode, hist_len=a.hist_len,
                                                             scan_stack=a.scan_stack, scan_stride=a.scan_stride))
    teacher = common.make_teacher(rls, env, grip=a.teacher_grip, recover_time=a.teacher_recover_time); teacher.speed_scale = a.teacher_speed
    spec = common.obs_spec(env)
    priv_dim = env.privileged(env.reset()[1] and env.last_result).shape[1]
    model = ActorCritic(spec.scan_stack, spec.n_beams, spec.proprio_dim, priv_dim, act_dim=env.act_dim,
                        scan_deltas=a.scan_deltas, temporal_encoder=a.temporal_encoder,
                        scan_stem=a.scan_stem).to(device)
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
                      noise=0.05 if it else 0.0, need_gap=a.hard_frac > 0).finalize()
        bufs.append(buf); bufs = bufs[-a.keep_iters:]; t_col = tm.lap()        # host RAM: keep the last few iterations (14 GB laptop)
        loss = train_epochs(model, bufs, a.epochs, a.batch, device, opt, log, a.hard_frac, a.hard_power, a.log_every); t_tr = tm.lap()
        m = common.rollout_metrics(env, lambda o: model.act(*flatten_obs(o), deterministic=True)[0], a.eval_steps, a.speed_cap,
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
                "samples": sum(len(b) for b in bufs), "metrics": m, "teacher": teacher_metrics, "action_mode": a.action_mode}
        save_checkpoint(os.path.join(out, f"student_it{it}.pt"), model, meta)
        save_checkpoint(os.path.join(out, "student_latest.pt"), model, meta)
    run.finish()


if __name__ == "__main__":
    main()
