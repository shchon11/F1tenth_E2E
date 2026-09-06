"""PPO fine-tuning with an asymmetric critic, initialized from the DAgger student.

- KL(pi_IL || pi) regularizer, decayed over the first `kl_decay` env steps, keeps the policy near
  the imitation policy while the critic warms up.
- Speed-cap curriculum: the commanded speed cap ramps from `cap0` to `cap1` over `cap_steps`
  env steps (the cap is part of the observation, so the policy stays consistent).
- Time-limit truncations bootstrap with the current value estimate.
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch

from ..gym_env import EnvConfig
from . import common
from .model import ActorCritic, load_checkpoint, save_checkpoint
from .obs import flatten_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=f"ppo_{time.strftime('%m%d_%H%M')}")
    ap.add_argument("--init", default="", help="DAgger checkpoint to start from")
    ap.add_argument("--envs", type=int, default=2048); ap.add_argument("--tracks", default="train", help="'train', 'eval' or comma separated catalog names")
    ap.add_argument("--horizon", type=int, default=32); ap.add_argument("--total", type=float, default=100e6)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lr-end", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99); ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--vf", type=float, default=0.5); ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--kl-coef", type=float, default=0.3); ap.add_argument("--kl-decay", type=float, default=15e6)
    ap.add_argument("--cap0", type=float, default=4.0); ap.add_argument("--cap1", type=float, default=8.0); ap.add_argument("--cap-steps", type=float, default=40e6)
    ap.add_argument("--critic-warmup", type=int, default=10, help="updates with the actor frozen")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--wandb", default="online")
    ap.add_argument("--save-every", type=int, default=25); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for network forward/backward (~2x faster)")
    ap.add_argument("--collision-penalty", type=float, default=10.0); ap.add_argument("--steer-penalty", type=float, default=0.05)
    ap.add_argument("--proximity-penalty", type=float, default=0.1, help="per-step penalty at zero wall gap (0 = off)")
    ap.add_argument("--safe-dist", type=float, default=0.30, help="[m] body-to-wall gap where the proximity penalty starts")
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--scan-stack", type=int, default=3); ap.add_argument("--scan-stride", type=int, default=1, help="control steps between stacked scans")
    ap.add_argument("--race-size", type=int, default=1, help="cars per track instance (>1: opponents in the LiDAR, car-car collisions)")
    ap.add_argument("--opponent", default="policy", choices=["policy", "teacher"], help="who drives cars 1..M-1: the policy (self-play) or the raceline teacher")
    ap.add_argument("--opp-speed", type=float, nargs=2, default=(0.6, 1.0), help="teacher opponents: speed scale range per race")
    a = ap.parse_args()
    device = torch.device(a.device); torch.manual_seed(a.seed)
    names = common.track_names(a.tracks)
    need_rl = a.race_size > 1 and a.opponent == "teacher"
    print(f"loading {len(names)} tracks{' + racelines' if need_rl else ''} ...", flush=True)
    tracks, rls = common.load_tracks(names, racelines=need_rl)
    env = common.make_env(tracks, a.envs, device, EnvConfig(speed_cap=a.cap0, reward_collision=-abs(a.collision_penalty),
                                                              reward_steer_rate=a.steer_penalty, reward_proximity=a.proximity_penalty,
                                                              safe_dist=a.safe_dist, max_steps=int(a.episode_s * 40),
                                                              scan_stack=a.scan_stack, scan_stride=a.scan_stride,
                                                              race_size=a.race_size, opponent=a.opponent,
                                                              opp_speed_range=tuple(a.opp_speed)), seed=a.seed, rls=rls)
    spec = common.obs_spec(env)
    obs, info = env.reset(seed=a.seed)
    priv = env.privileged(env.last_result); priv_dim = priv.shape[1]
    lid = env.learner_ids                                   # races with teacher opponents: only the learners' data is used
    if a.init:
        model, extra = load_checkpoint(a.init, device, override={"n_stack": spec.scan_stack, "n_beams": spec.n_beams,
                                                                  "proprio_dim": spec.proprio_dim, "priv_dim": priv_dim})
        print("init from", a.init, extra.get("metrics"), "| re-initialized:", extra.get("skipped") or "nothing")
    else:
        model = ActorCritic(spec.scan_stack, spec.n_beams, spec.proprio_dim, priv_dim).to(device)
    ref = copy.deepcopy(model.actor).eval()
    for p_ in ref.parameters(): p_.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    run = common.wandb_init(a.name, vars(a) | {"phase": "ppo", "tracks": names}, group="ppo", mode=a.wandb)
    out = common.run_dir(a.name)
    env.sim.warmup()

    T, B = a.horizon, int(lid.numel())
    k, N, P = spec.scan_stack, spec.n_beams, spec.proprio_dim
    buf_scan = torch.zeros(T, B, k, N, device=device, dtype=torch.float16)
    buf_pro = torch.zeros(T, B, P, device=device); buf_priv = torch.zeros(T, B, priv_dim, device=device)
    buf_act = torch.zeros(T, B, 2, device=device); buf_logp = torch.zeros(T, B, device=device)
    buf_rew = torch.zeros(T, B, device=device); buf_done = torch.zeros(T, B, device=device); buf_trunc = torch.zeros(T, B, device=device)
    buf_val = torch.zeros(T + 1, B, device=device)

    steps_done = 0; update = 0; t_start = time.time()
    ep_stats = {"return": [], "progress": [], "collided": [], "lap_time": [], "steps": []}
    n_updates = int(a.total // (T * B))
    while steps_done < a.total:
        frac = steps_done / a.total
        cap = a.cap0 + (a.cap1 - a.cap0) * min(1.0, steps_done / a.cap_steps); env.set_speed_cap(cap)
        kl_coef = a.kl_coef * max(0.0, 1.0 - steps_done / a.kl_decay)
        lr = a.lr + (a.lr_end - a.lr) * frac
        for g in opt.param_groups: g["lr"] = lr
        tm = common.Timer()
        # ---------------- rollout
        model.eval()
        ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and device.type == "cuda")
        with torch.no_grad():
            for t in range(T):
                scan, pro = flatten_obs(obs)
                with ac:
                    act, logp = model.act(scan, pro)
                    val = model.critic(scan[lid], pro[lid], priv[lid]).float()
                act, logp = act.float(), logp.float()
                buf_scan[t] = scan[lid].half(); buf_pro[t] = pro[lid]; buf_priv[t] = priv[lid]; buf_act[t] = act[lid]; buf_logp[t] = logp[lid]; buf_val[t] = val
                obs, rew, term, trunc, info = env.step(act)
                priv = env.privileged(env.last_result)
                buf_rew[t] = rew[lid]; buf_done[t] = term[lid].float(); buf_trunc[t] = trunc[lid].float()
                ep_stats["lap_time"] += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                if "final" in info:
                    f = info["final"]; m = env.learner[f["ids"]]
                    ep_stats["return"] += f["return"][m].tolist(); ep_stats["progress"] += f["progress"][m].tolist()
                    ep_stats["collided"] += f["collided"][m].float().tolist(); ep_stats["steps"] += f["steps"][m].tolist()
            scan, pro = flatten_obs(obs)
            with ac:
                buf_val[T] = model.critic(scan[lid], pro[lid], priv[lid]).float()
            # GAE; truncation bootstraps with the state's own value (approximation)
            adv = torch.zeros(T, B, device=device); last = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                nonterm = 1.0 - buf_done[t]
                next_v = torch.where(buf_trunc[t] > 0, buf_val[t], buf_val[t + 1])
                delta = buf_rew[t] + a.gamma * next_v * nonterm - buf_val[t]
                last = delta + a.gamma * a.lam * nonterm * (1.0 - buf_trunc[t]) * last
                adv[t] = last
            ret = adv + buf_val[:T]
        t_roll = tm.lap()
        # ---------------- update
        model.train()
        n = T * B
        f_scan = buf_scan.reshape(n, k, N); f_pro = buf_pro.reshape(n, P); f_priv = buf_priv.reshape(n, priv_dim)
        f_act = buf_act.reshape(n, 2); f_logp = buf_logp.reshape(n); f_adv = adv.reshape(n); f_ret = ret.reshape(n); f_val = buf_val[:T].reshape(n)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        stats = {"pg": [], "vf": [], "ent": [], "kl_ref": [], "approx_kl": [], "clipfrac": []}
        freeze_actor = update < a.critic_warmup
        for ep in range(a.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, a.minibatch):
                idx = perm[i:i + a.minibatch]
                scan = f_scan[idx].float(); pro = f_pro[idx]
                with ac:
                    logp, ent, val, d = model.evaluate(scan, pro, f_priv[idx], f_act[idx])
                logp, ent, val = logp.float(), ent.float(), val.float()
                ratio = (logp - f_logp[idx]).exp()
                pg = -torch.min(ratio * f_adv[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * f_adv[idx]).mean()
                v_clipped = f_val[idx] + (val - f_val[idx]).clamp(-a.clip, a.clip)
                vf = 0.5 * torch.max((val - f_ret[idx]) ** 2, (v_clipped - f_ret[idx]) ** 2).mean()
                with torch.no_grad(), ac:
                    d_ref = ref.dist(scan, pro)
                d_ref = torch.distributions.Normal(d_ref.mean.float(), d_ref.stddev.float())
                d = torch.distributions.Normal(d.mean.float(), d.stddev.float())
                kl_ref = torch.distributions.kl_divergence(d_ref, d).sum(1).mean()
                loss = a.vf * vf + (0.0 if freeze_actor else 1.0) * (pg - a.ent * ent.mean() + kl_coef * kl_ref)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad); opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.item()); stats["vf"].append(vf.item()); stats["ent"].append(ent.mean().item())
                    stats["kl_ref"].append(kl_ref.item()); stats["approx_kl"].append(((ratio - 1) - (logp - f_logp[idx])).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
        t_upd = tm.lap()
        steps_done += n; update += 1
        # ---------------- logging
        if update % 2 == 0 or update == 1:
            n_ep = len(ep_stats["return"])
            log = {"steps": steps_done, "update": update, "speed_cap": cap, "kl_coef": kl_coef, "lr": lr,
                   "rollout/reward_per_step": buf_rew.mean().item(), "rollout/mean_speed": env.sim.state[:, 3].mean().item(),
                   "rollout/value_mean": buf_val[:T].mean().item(), "rollout/adv_std": adv.std().item(),
                   "loss/pg": np.mean(stats["pg"]), "loss/vf": np.mean(stats["vf"]), "loss/entropy": np.mean(stats["ent"]),
                   "loss/kl_ref": np.mean(stats["kl_ref"]), "loss/approx_kl": np.mean(stats["approx_kl"]), "loss/clipfrac": np.mean(stats["clipfrac"]),
                   "policy/log_std_steer": model.actor.log_std[0].item(), "policy/log_std_speed": model.actor.log_std[1].item(),
                   "time/rollout_s": t_roll, "time/update_s": t_upd, "time/env_steps_per_s": n / (t_roll + t_upd),
                   "time/elapsed_min": (time.time() - t_start) / 60}
            if n_ep:
                log.update({"episode/return": np.mean(ep_stats["return"]), "episode/progress_m": np.mean(ep_stats["progress"]),
                            "episode/collision_rate": np.mean(ep_stats["collided"]), "episode/len_steps": np.mean(ep_stats["steps"]),
                            "episode/count": n_ep})
                if ep_stats["lap_time"]:
                    log["episode/lap_time_s"] = np.mean(ep_stats["lap_time"])
                ep_stats = {kk: [] for kk in ep_stats}
            run.log(log, step=steps_done)
            print(f"upd {update}/{n_updates} steps {steps_done/1e6:.1f}M cap {cap:.1f} | rew/step {log['rollout/reward_per_step']:.3f} "
                  f"coll {log.get('episode/collision_rate', float('nan')):.2f} prog {log.get('episode/progress_m', float('nan')):.0f} m "
                  f"lap {log.get('episode/lap_time_s', float('nan')):.1f} s | kl_ref {log['loss/kl_ref']:.3f} | {log['time/env_steps_per_s']:.0f} steps/s", flush=True)
        if update % a.save_every == 0:
            save_checkpoint(os.path.join(out, f"ppo_u{update}.pt"), model, {"spec": spec.__dict__, "steps": steps_done, "cap": cap})
            save_checkpoint(os.path.join(out, "ppo_latest.pt"), model, {"spec": spec.__dict__, "steps": steps_done, "cap": cap})
    save_checkpoint(os.path.join(out, "ppo_final.pt"), model, {"spec": spec.__dict__, "steps": steps_done, "cap": cap})
    run.finish()


if __name__ == "__main__":
    main()
