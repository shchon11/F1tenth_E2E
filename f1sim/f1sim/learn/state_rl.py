"""Privileged-state RL (rl-audit 2026-10-29): learn to race from the simulator's TRUE state, from scratch.

Why, in one line: every PPO run so far learned from a LiDAR scan through a plan that MAP mostly ignores, with 25 ms
decisions whose effect (2 cm) drowns in the return's noise -- the policies only traded speed for hits on one front
(J ~0.10). The fundamental change: give the learner what the outcome depends on, and let it act on the car directly.

* Observation (no scan): the v2 privileged vector (ego dynamics, offset and heading to the centerline, 3 nearest
  opponents with relative position / velocity, the nearest car's closing speed and time to contact, the 3 nearest props
  ahead, the race time left) + the track AHEAD: the centerline at fixed arc distances, in the car's frame, with the half
  width there + the steering angle + the last action. Running mean / std normalised.
* Action: the car's own command, steer and speed (`EnvConfig.action_mode "direct"`), held REPEAT control steps.
* Reward per decision: progress + steer-rate + wrong-way - PRICE x charged hard (>= 1.5 m/s) onsets (walls, props, cars):
  the objective every table reports.
* PPO, MLP actor-critic, many envs (no LiDAR to cast for the learner: --beams small).

Stages (--stage): solo (no props, no cars) -> props (props on the racing line, movable) -> race (+ planner opponents).
This policy is never deployed: the LiDAR policy learns it afterwards (teacher-student), so no localization on the car.
usage: python -m f1sim.learn.state_rl --name NAME [--stage solo] [--envs 4096] [--updates 600] [--init CKPT]"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .. import Config, maps
from ..gym_env import REWARD_COMPONENT_KEYS, EnvConfig
from . import common

#: arc distances ahead [m] where the centerline is shown to the policy
AHEAD = (1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0)
KOREA = "real:korea_2026_competition"


def build_env(a, device):
    tracks = [KOREA] + ([f"{KOREA}+warp{k}" for k in range(int(a.warps))] if a.warps else [])
    trs = [maps.load(n) for n in tracks]
    from ..raceline import Raceline
    rls = [Raceline.build_cached(t) for t in trs] if a.stage == "race" else None
    cfg = Config()
    cfg.sim.compile = False
    cfg.lidar.n_beams = int(a.beams)
    cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0)
    cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
    e = dict(action_mode="direct", max_steps=int(a.episode_s * 40), speed_cap=10.0, v_max_policy=10.0,
             collision_mode="soft", contact_severity_min=1.5, stagger_first_episode=True,
             reward_progress=1.0, reward_collision=-1.0, reward_car_contact=1.0, reward_steer_rate=0.02,
             reward_wrong_way=0.4, reward_proximity=0.0, reward_lap=0.0, reward_lap_time=0.0,
             spawn_runway=3.0, priv_props=True, priv_time=True, compile_tracker=False)
    if a.stage in ("props", "race"):
        e.update(procedural_obstacles=1.0, procedural_density=1.5, procedural_max_props=18,
                 procedural_raceline_corridor="off", movable_obstacles=True)
    if a.stage == "race":
        slot = {"kind_mix": ["forzaeth", "forzaeth_pred", "lane_switch", "interactive"], "speed_scale": [0.7, 1.05],
                "present": 0.5, "spawn": "random"}
        from .opponent_config import parse_slots_json
        e.update(race_size=2, opponent="teacher", opp_slots=parse_slots_json(json.dumps([slot])))
    env = common.make_env(trs, int(a.envs), device, EnvConfig(**e), cfg=cfg, seed=int(a.seed), rls=rls)
    return env


class StateObs:
    """The observation, from the env's state after a step."""

    def __init__(self, env):
        self.env = env
        self.ahead = torch.tensor(AHEAD, device=env.device)
        self.dim = None

    def __call__(self, last_action):
        env = self.env
        lid = env.learner_ids
        pv = env.privileged_v2()[lid]
        res = env.last_result
        st = res.state[lid]
        tid = env.sim.tid[lid]
        s = res.s[lid]
        tr = env.sim.track
        n = self.ahead.numel()
        s_k = s[:, None] + self.ahead[None, :]
        xy_k, yaw_k = tr.pose_at_s(s_k.reshape(-1), tid.repeat_interleave(n))
        xy_k = xy_k.reshape(-1, n, 2); yaw_k = yaw_k.reshape(-1, n)
        d = xy_k - st[:, None, :2]
        c, si = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
        dx, dy = d[..., 0] * c + d[..., 1] * si, -d[..., 0] * si + d[..., 1] * c
        dyaw = torch.remainder(yaw_k - st[:, 2:3] + math.pi, 2 * math.pi) - math.pi
        w = tr.sample_edt(xy_k, tid[:, None].expand(-1, n))
        ahead = torch.stack([dx / 10.0, dy / 5.0, dyaw / math.pi, w / 2.0], 2).reshape(st.shape[0], -1)
        steer = (st[:, 6] / float(env.s_max))[:, None]
        out = torch.cat([pv, ahead, steer, last_action], 1).float()
        self.dim = out.shape[1]
        return out


class RunningNorm:
    def __init__(self, dim, device, clip=8.0):
        self.mean = torch.zeros(dim, device=device); self.var = torch.ones(dim, device=device); self.n = 1e-4
        self.clip = clip

    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        bm, bv, bn = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean; tot = self.n + bn
        self.mean = self.mean + d * bn / tot
        self.var = (self.var * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-self.clip, self.clip)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "n": self.n}

    def load_state_dict(self, d):
        self.mean, self.var, self.n = d["mean"], d["var"], float(d["n"])


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=2, log_std=-0.5, hidden=256):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim, hidden)
        self.v = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))
        nn.init.zeros_(self.pi[-1].bias); self.pi[-1].weight.data.mul_(0.01)

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())


def train(a):
    device = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    env = build_env(a, device)
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    obs_fn = StateObs(env)
    env.reset(seed=int(a.seed))
    lid = env.learner_ids
    B = int(lid.numel())
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    last_a = torch.zeros(B, 2, device=device)
    o = obs_fn(last_a)
    D = obs_fn.dim
    model = ActorCritic(D, log_std=a.log_std).to(device)
    norm = RunningNorm(D, device)
    ret_rms = RunningNorm(1, device)
    upd0 = 0
    if a.init:
        ck = torch.load(a.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); norm.load_state_dict(ck["norm"]); ret_rms.load_state_dict(ck["ret_rms"])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    w_col, w_car = abs(env.ecfg.reward_collision), abs(env.ecfg.reward_car_contact)
    ci = {k: REWARD_COMPONENT_KEYS.index(k) for k in REWARD_COMPONENT_KEYS}
    out = os.path.expanduser(f"~/f1sim_runs/{a.name}"); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    log = open(os.path.join(out, "progress.jsonl"), "a")
    T, R = int(a.horizon), int(a.repeat)
    touching = torch.zeros(B, dtype=torch.bool, device=device)
    print(f"[state_rl] {a.name}: 단계 {a.stage}, 환경 {env.B} (학습 행 {B}), 관측 {D}차원, 결정 {40 // R} Hz, "
          f"롤아웃 {T}결정 = {T * R} 스텝, 가격 {a.price:g}", flush=True)
    t0 = time.time()
    for u in range(upd0 + 1, upd0 + int(a.updates) + 1):
        frac = (u - upd0 - 1) / max(1, int(a.updates))
        for g in opt.param_groups:
            g["lr"] = a.lr + (a.lr_end - a.lr) * frac
        # the hit price ramps in (--price-hold, --price-ramp): from scratch, a full price at once teaches the car to stand
        # still (measured 10-29: 2.0 -> 0.2 m/s in 10 updates); progress first, then the price, then the full objective
        k = (u - upd0 - 1 - int(a.price_hold)) / max(1, int(a.price_ramp))
        price = a.price_start + (a.price - a.price_start) * min(1.0, max(0.0, k))
        bo = torch.zeros(T, B, D, device=device); ba = torch.zeros(T, B, 2, device=device)
        blp = torch.zeros(T, B, device=device); bv = torch.zeros(T + 1, B, device=device)
        br = torch.zeros(T, B, device=device); bd = torch.zeros(T, B, device=device)
        prog = hits_w = hits_c = 0.0; laps = []
        tr0 = time.time()
        with torch.no_grad():
            for t in range(T):
                norm.update(o)
                on = norm(o)
                dist = model.dist(on)
                act = dist.sample()
                bo[t] = on; ba[t] = act; blp[t] = dist.log_prob(act).sum(1)
                bv[t] = model.v(on).squeeze(1) * (ret_rms.var.sqrt() + 1e-6)
                rew = torch.zeros(B, device=device); done = torch.zeros(B, dtype=torch.bool, device=device)
                act_full[lid] = act.clamp(-1, 1)
                for k in range(R):
                    _, r_, te, trc, info = env.step(act_full)
                    comp = info["reward_components"]
                    col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
                    tch = car < 0
                    onset = tch & ~touching
                    touching = tch
                    onw = (-col / max(w_col, 1e-9)).clamp_min(0)
                    dense = r_[lid].float() - col - car
                    step_r = dense - price * (onw + onset.float())
                    alive = (~done).float()
                    rew += step_r * alive
                    prog += float((info["progress"][lid].float().clamp_min(0) * alive).sum())
                    hits_w += float((onw * alive).sum()); hits_c += float((onset.float() * alive).sum())
                    d_ = (te | trc)[lid]
                    touching = touching & ~d_
                    done |= d_
                    if info["lap_times"].numel():
                        laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
                br[t] = rew; bd[t] = done.float()
                last_a = torch.where(done[:, None], torch.zeros_like(act), act.clamp(-1, 1))
                o = obs_fn(last_a)
            bv[T] = model.v(norm(o)).squeeze(1) * (ret_rms.var.sqrt() + 1e-6)
            adv = torch.zeros(T, B, device=device); last = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                nd = 1.0 - bd[t]
                delta = br[t] + a.gamma * bv[t + 1] * nd - bv[t]
                last = delta + a.gamma * a.lam * nd * last
                adv[t] = last
            ret = adv + bv[:T]
            ret_rms.update(ret.reshape(-1, 1))
            ret_n = ret / (ret_rms.var.sqrt() + 1e-6)
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        t_roll = time.time() - tr0
        N = T * B
        fo, fa, flp, fadv, fret = bo.reshape(N, D), ba.reshape(N, 2), blp.reshape(N), adv_n.reshape(N), ret_n.reshape(N)
        mb = N // int(a.minibatches)
        kls, cfs = [], []
        for ep in range(int(a.epochs)):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, mb):
                j = perm[i:i + mb]
                dist = model.dist(fo[j])
                lp = dist.log_prob(fa[j]).sum(1)
                ratio = (lp - flp[j]).exp()
                pg = -torch.min(ratio * fadv[j], ratio.clamp(1 - a.clip, 1 + a.clip) * fadv[j]).mean()
                vf = 0.5 * (model.v(fo[j]).squeeze(1) - fret[j]).pow(2).mean()
                loss = pg + a.vf * vf
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    kls.append(float(((ratio - 1) - (lp - flp[j])).mean())); cfs.append(float(((ratio - 1).abs() > a.clip).float().mean()))
        km = prog / 1000.0
        secs = T * R * B * env.sim.control_dt
        stats = {"update": u, "progress_ms": prog / secs, "hits_wall_prop_per_km": hits_w / max(km, 1e-3),
                 "hits_car_per_km": hits_c / max(km, 1e-3), "J_per_step": float(br.sum() / (T * R * B)),
                 "laps": len(laps), "lap_mean_s": float(np.mean(laps)) if laps else None,
                 "lap_best_s": float(np.min(laps)) if laps else None, "std": model.log_std.exp().tolist(),
                 "kl": float(np.mean(kls)), "clipfrac": float(np.mean(cfs)), "t_roll": t_roll,
                 "t_upd": time.time() - tr0 - t_roll, "min": (time.time() - t0) / 60, "price": price,
                 "J80_per_step": float((prog - 80.0 * (hits_w + hits_c)) / (T * R * B))}
        log.write(json.dumps(stats) + "\n"); log.flush()
        if u % 5 == 0 or u == upd0 + 1:
            lp_ = f"{stats['lap_mean_s']:.2f}" if laps else "-"
            print(f"[{time.strftime('%H:%M')}] 업데이트 {u} | 진행 {stats['progress_ms']:.2f} m/s | 강한 충돌 벽+상자 "
                  f"{stats['hits_wall_prop_per_km']:.2f} 차 {stats['hits_car_per_km']:.2f} /km | 랩 {len(laps)}개 평균 {lp_} s | "
                  f"J(가격 {price:.0f}) {stats['J_per_step']:.4f} J80 {stats['J80_per_step']:.4f} | 표준편차 {[round(x, 3) for x in stats['std']]} | KL {stats['kl']:.4f} | "
                  f"{t_roll:.1f}+{stats['t_upd']:.1f} s", flush=True)
        if u % int(a.save_every) == 0 or u == upd0 + int(a.updates):
            torch.save({"model": model.state_dict(), "norm": norm.state_dict(), "ret_rms": ret_rms.state_dict(),
                        "obs_dim": D, "args": vars(a), "update": u, "ahead": AHEAD},
                       os.path.join(out, f"state_u{u}.pt"))
    print(f"[state_rl] {a.name} 끝", flush=True)


@torch.no_grad()
def evaluate(a):
    """The checkpoint's MEAN action (no exploration noise) in its stage's world: lap times, progress, hard hits per km."""
    device = torch.device(a.device)
    ck = torch.load(a.eval, map_location=device, weights_only=False)
    env = build_env(a, device)
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    obs_fn = StateObs(env)
    env.reset(seed=int(a.seed))
    lid = env.learner_ids; B = int(lid.numel())
    last_a = torch.zeros(B, 2, device=device)
    o = obs_fn(last_a)
    model = ActorCritic(o.shape[1]).to(device); model.load_state_dict(ck["model"])
    norm = RunningNorm(o.shape[1], device); norm.load_state_dict(ck["norm"])
    w_col = abs(env.ecfg.reward_collision)
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    touching = torch.zeros(B, dtype=torch.bool, device=device)
    prog = hw = hc = 0.0; laps = []; steps = 0
    R = int(a.repeat)
    for t in range(int(a.eval_steps) // R):
        act = model.pi(norm(o)).clamp(-1, 1)
        act_full[lid] = act
        for k in range(R):
            _, r_, te, trc, info = env.step(act_full)
            comp = info["reward_components"]
            col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
            tch = car < 0; onset = tch & ~touching; d_ = (te | trc)[lid]; touching = tch & ~d_
            hw += float((-col / max(w_col, 1e-9)).clamp_min(0).sum()); hc += float(onset.float().sum())
            prog += float(info["progress"][lid].float().clamp_min(0).sum()); steps += 1
            if info["lap_times"].numel():
                laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
        o = obs_fn(act)
    km = prog / 1000.0
    secs = steps * B * env.sim.control_dt
    q = np.percentile(laps, [10, 50, 90]) if laps else [float("nan")] * 3
    print(f"[평가] {os.path.basename(a.eval)} 단계 {a.stage}, 평균 행동, {B}대 x {steps * env.sim.control_dt:.0f} s = {km:.1f} km | "
          f"진행 {prog / secs:.2f} m/s | 강한 충돌 벽+상자 {hw / max(km, 1e-3):.2f}, 차 {hc / max(km, 1e-3):.2f} /km | "
          f"랩 {len(laps)}개, 중앙값 {q[1]:.2f} s (p10 {q[0]:.2f}, p90 {q[2]:.2f}) | J80/스텝 {(prog - 80 * (hw + hc)) / (steps * B):.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="", help="evaluate this checkpoint's mean action instead of training")
    ap.add_argument("--eval-steps", type=int, default=2400)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stage", default="solo", choices=("solo", "props", "race"))
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--warps", type=int, default=16)
    ap.add_argument("--beams", type=int, default=36)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--log-std", type=float, default=-0.5)
    ap.add_argument("--price", type=float, default=80.0)
    ap.add_argument("--price-start", type=float, default=0.0)
    ap.add_argument("--price-hold", type=int, default=60, help="updates at --price-start before the ramp")
    ap.add_argument("--price-ramp", type=int, default=200, help="updates of the linear ramp to --price")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--init", default="")
    ap.add_argument("--seed", type=int, default=5001)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.eval:
        evaluate(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
