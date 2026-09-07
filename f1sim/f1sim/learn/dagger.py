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

    def __init__(self, k: int):
        self.k = k
        self.scan, self.pro, self.lab, self.newep = [], [], [], []

    def add(self, scan_now, proprio, label, new_episode):
        self.scan.append(scan_now.to(torch.float16).cpu()); self.pro.append(proprio.to(torch.float16).cpu())   # a 1 s proprio history is 322 floats/sample
        self.lab.append(label.cpu()); self.newep.append(new_episode.cpu())

    def finalize(self):
        self.S = torch.stack(self.scan); self.P = torch.stack(self.pro); self.L = torch.stack(self.lab); self.N = torch.stack(self.newep)
        self.T, self.B = self.S.shape[:2]
        self.scan, self.pro, self.lab, self.newep = [], [], [], []
        return self

    def __len__(self):
        return self.T * self.B

    def sample(self, n, device):
        t = torch.randint(self.T, (n,)); b = torch.randint(self.B, (n,))
        stack = []
        cur_t = t.clone(); blocked = torch.zeros(n, dtype=torch.bool)
        for j in range(self.k):
            stack.append(self.S[cur_t, b])
            # step back one frame unless that crosses an episode boundary
            blocked = blocked | self.N[cur_t, b] | (cur_t == 0)
            cur_t = torch.where(blocked, cur_t, cur_t - 1)
        scan = torch.stack(stack, 1).to(device, torch.float32)
        return scan, self.P[t, b].to(device).float(), self.L[t, b].to(device)


def collect(env, model, teacher, steps, beta, device, buf: StepBuffer, noise=0.0):
    obs, info = env.reset()
    new_ep = torch.ones(env.B, dtype=torch.bool, device=device)
    with torch.no_grad():
        for t in range(steps):
            scan, pro = flatten_obs(obs)
            label = env.teacher_label(teacher)
            buf.add(scan[:, 0], pro, label, new_ep)
            if beta >= 1.0:
                a = label
            else:
                a_student, _ = model.act(scan, pro, deterministic=True)
                if noise > 0:
                    a_student = (a_student + noise * torch.randn_like(a_student)).clamp(-1, 1)
                use_t = torch.rand(env.B, device=device) < beta
                a = torch.where(use_t[:, None], label, a_student)
            obs, rew, term, trunc, info = env.step(a)
            new_ep = torch.zeros(env.B, dtype=torch.bool, device=device)
            if "final" in info:
                new_ep[info["final"]["ids"]] = True
    return buf


def train_epochs(model, bufs, epochs, batch, device, opt, log):
    n_total = sum(len(b) for b in bufs)
    steps = max(1, int(epochs * n_total / batch))
    weights = torch.tensor([len(b) for b in bufs], dtype=torch.float)
    losses = []
    for i in range(steps):
        b = bufs[torch.multinomial(weights, 1).item()]
        scan, pro, lab = b.sample(batch, device)
        mu = model.actor(scan, pro)
        loss = F.smooth_l1_loss(mu, lab, beta=0.1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if i % 200 == 0:
            log({"dagger/loss": float(np.mean(losses[-200:]))})
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
    ap.add_argument("--keep-iters", type=int, default=5, help="aggregate the data of at most this many recent iterations (host RAM)")
    ap.add_argument("--eval-steps", type=int, default=800); ap.add_argument("--wandb", default="online")
    a = ap.parse_args()
    device = torch.device(a.device)
    names = common.track_names(a.tracks)
    print(f"loading {len(names)} tracks + racelines ...", flush=True)
    tracks, rls = common.load_tracks(names, racelines=True)
    env = common.make_env(tracks, a.envs, device, EnvConfig(speed_cap=a.speed_cap, action_mode=a.action_mode, hist_len=a.hist_len))
    teacher = common.make_teacher(rls, env); teacher.speed_scale = a.teacher_speed
    spec = common.obs_spec(env)
    priv_dim = env.privileged(env.reset()[1] and env.last_result).shape[1]
    model = ActorCritic(spec.scan_stack, spec.n_beams, spec.proprio_dim, priv_dim, act_dim=env.act_dim).to(device)
    opt = torch.optim.Adam(model.actor.parameters(), lr=a.lr)
    run = common.wandb_init(a.name, vars(a) | {"phase": "dagger", "tracks": names}, group="dagger", mode=a.wandb)
    out = common.run_dir(a.name)
    log = lambda d: run.log(d)
    env.sim.warmup()
    bufs = []
    t0 = time.time()
    for it in range(a.iters):
        beta = 1.0 if it == 0 else a.beta0 * (0.5 ** (it - 1))
        tm = common.Timer()
        buf = collect(env, model, teacher, a.steps, beta, device, StepBuffer(spec.scan_stack), noise=0.05 if it else 0.0).finalize()
        bufs.append(buf); bufs = bufs[-a.keep_iters:]; t_col = tm.lap()        # host RAM: keep the last few iterations (14 GB laptop)
        loss = train_epochs(model, bufs, a.epochs, a.batch, device, opt, log); t_tr = tm.lap()
        m = common.rollout_metrics(env, lambda o: model.act(*flatten_obs(o), deterministic=True)[0], a.eval_steps, a.speed_cap); t_ev = tm.lap()
        tm_ = common.rollout_metrics(env, lambda o: env.teacher_label(teacher), a.eval_steps, a.speed_cap) if it == 0 else tm_
        log({"dagger/iter": it, "dagger/beta": beta, "dagger/samples": sum(len(b) for b in bufs), "dagger/final_loss": loss,
             **{f"student/{k}": v for k, v in m.items()}, **{f"teacher/{k}": v for k, v in tm_.items()},
             "time/collect_s": t_col, "time/train_s": t_tr, "time/eval_s": t_ev, "time/elapsed_min": (time.time() - t0) / 60})
        print(f"iter {it}: beta {beta:.2f} samples {sum(len(b) for b in bufs)} loss {loss:.4f} | student coll {m['collision_rate']:.2f} "
              f"prog {m['progress_rate_mps']:.2f} m/s lap {m['lap_time_s']:.1f} s | teacher coll {tm_['collision_rate']:.2f} prog {tm_['progress_rate_mps']:.2f} m/s "
              f"lap {tm_['lap_time_s']:.1f} s | {t_col:.0f}+{t_tr:.0f}+{t_ev:.0f} s", flush=True)
        meta = {"spec": spec.__dict__, "phase": "dagger", "run": a.name, "iter": it, "iters": a.iters,
                "samples": sum(len(b) for b in bufs), "metrics": m, "teacher": tm_, "action_mode": a.action_mode}
        save_checkpoint(os.path.join(out, f"student_it{it}.pt"), model, meta)
        save_checkpoint(os.path.join(out, "student_latest.pt"), model, meta)
    run.finish()


if __name__ == "__main__":
    main()
