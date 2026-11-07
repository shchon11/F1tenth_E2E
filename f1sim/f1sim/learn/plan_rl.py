"""PPO for the redesigned planner (11-07, every choice agreed with the user step by step; see plan_steps, plan_reward,
plan_policy). A new, plain trainer -- none of state_rl's switches (spawn behind boxes, level replay, stall price,
exploration floors, correlated noise, KL leashes): the world, the plan format, the reward, and PPO.

* World: `configs/plan_world_korea25.json` -- Korea 2025 and 16 warped variants, the four cases (empty, opponent,
  boxes, boxes + opponent) about 25 % each, box layouts drawn per episode (1..all), half of them removed at 10-30 s,
  ForzaETH opponents at 0.4-1.2x, 40 s episodes.
* Every 0.1 s each learner car draws a plan and drives its first 0.1 s exactly (`GymEnv.enable_ideal`); contacts are the
  simulator's own.
* PPO: gamma 0.995 per decision (a ~20 s horizon, about three laps), lambda 0.95, 4 epochs x 8 minibatches, clip 0.2,
  lr 3e-4 -> 1e-4, exploration std 0.6 at the start and learned.

usage: python -m f1sim.learn.plan_rl --name NAME [--updates 2000] [--envs 1024] [--seed 1]"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from .. import plan_steps as PS
from .. import dynamics as dyn
from .. import map_controller as M
from . import state_rl as SR
from .plan_policy import Actor, Critic, PlanObs, plan_world, tanh_logp, PT_IDX, Actor2, PlanObs2, FORMAT2
from .plan_reward import PlanReward

WORLD = os.path.join(os.path.dirname(__file__), "configs", "plan_world_korea25.json")


def build(world: str, envs: int, seed: int, device, overrides=None):
    a = argparse.Namespace(**json.load(open(world)))
    a.envs, a.seed, a.device = envs, seed, str(device)
    for k, v in (overrides or {}).items():
        setattr(a, k, v)
    env = SR.build_env(a, device)
    env.enable_ideal()
    from .graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    env.reset(seed=seed)
    lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), device)
    return env, lim, a


class Driver:
    """drives the learner rows one decision: plan from the action, the 4 control steps, the reward"""

    def __init__(self, env, lim, pr: PlanReward, obs: PlanObs):
        self.env, self.lim, self.pr, self.obs = env, lim, pr, obs
        self.lid = env.learner_ids
        self.act0 = torch.zeros(env.B, env.act_dim, device=env.device)
        from ..lidar import HIT_CAR
        self._hit_car = HIT_CAR

    def decision(self, a: torch.Tensor, p=None):
        """drive one decision: `p` a plan already made (v2: from the policy's speed estimate and last command), else
        made from `a` at the car's true state (v1)"""
        env, lid = self.env, self.lid
        st = env.sim.state[lid].clone()
        if p is None:
            p = PS.roll(a, st[:, dyn.IVX], st[:, dyn.ISTEER], self.lim)
        x, y, psi = plan_world(p, st)
        pc = self.pr.plan_cost(x, y, psi, p["v"])
        idx = PT_IDX.to(a.device)
        pts = torch.stack([x[:, idx], y[:, idx]], -1)
        env.sim.ideal_vcap[lid] = 1e3
        restarted = torch.zeros(lid.numel(), dtype=torch.bool, device=a.device)
        for j in range(PS.SUB):
            cmd_d = torch.where(restarted, torch.zeros_like(p["cmd_d"][:, j]), p["cmd_d"][:, j])
            cmd_v = torch.where(restarted, torch.zeros_like(p["cmd_v"][:, j]), p["cmd_v"][:, j])
            env.ideal_cmd[lid, 0] = cmd_d; env.ideal_cmd[lid, 1] = cmd_v
            _, _, te, trc, _ = env.step(self.act0)
            restarted |= (te | trc)[lid]
            r = env.last_result
            self.pr.control_step((r.scan_type[lid] == self._hit_car).any(1))     # a beam ended on a car
        out = self.pr.decision(pts, pc["price"], restarted)
        out["plan_hit_price"] = out.pop("plan_hit")
        out["plan_hit"] = pc["hit"].float()
        self.obs.push_scan()
        self.obs.set_plan(pts, p["pv"])
        if hasattr(self.obs, "set_plan2"):
            self.obs.set_plan2(p)
        if bool(restarted.any()):
            rows = torch.nonzero(restarted)[:, 0]
            self.obs.reset_rows(rows)
        out["restarted"] = restarted
        return out, p, (x, y, psi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--world", default=WORLD)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=64, help="decisions per rollout (0.1 s each)")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=1e-4)
    ap.add_argument("--std0", type=float, default=0.6)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--format", default="v2", choices=("v1", "v2"),
                    help="v2 (11-08): raw sensors only, the plan starts from the policy's own speed estimate")
    ap.add_argument("--vest-coef", type=float, default=1.0, help="v2: weight of the speed estimate's loss [per (m/s)^2]")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", default=os.path.expanduser("~/f1sim_runs"))
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    run = os.path.join(args.runs, args.name)
    if os.path.exists(os.path.join(run, "progress.jsonl")):
        raise SystemExit(f"{run} already has a run: pick an unused name")
    os.makedirs(run, exist_ok=True)
    env, lim, world = build(args.world, args.envs, args.seed, dev)
    lid = env.learner_ids; Bl = int(lid.numel())
    pr = PlanReward(env, args.gamma, dev)
    v2 = args.format == "v2"
    obs = (PlanObs2 if v2 else PlanObs)(env, lim, pr.goal)
    drv = Driver(env, lim, pr, obs)
    priv_dim = int(obs.priv().shape[1])
    actor = (Actor2 if v2 else Actor)(obs.beams, std0=args.std0).to(dev)
    critic = Critic(obs.beams, priv_dim, state_dim=int(obs.state().shape[1])).to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-5)
    json.dump({"args": vars(args), "world": vars(world), "priv_dim": priv_dim, "beams": obs.beams},
              open(os.path.join(run, "args.json"), "w"), indent=1, default=str)
    print(f"[plan_rl] {args.name}: 환경 {env.B} (학습 차 {Bl}), 결정 0.1 s, 롤아웃 {args.horizon}결정, LiDAR {obs.beams}빔 x {3}장, "
          f"critic 추가 입력 {priv_dim}", flush=True)
    T = args.horizon
    log = open(os.path.join(run, "progress.jsonl"), "a")
    t0 = time.time()
    for upd in range(1, args.updates + 1):
        frac = (upd - 1) / max(1, args.updates - 1)
        for gp in opt.param_groups:
            gp["lr"] = args.lr + (args.lr_end - args.lr) * frac
        S, X, PV, U, LP, V, R, D, VT = [], [], [], [], [], [], [], [], []
        vest_err = []
        stats = {k: torch.zeros(Bl, device=dev) for k in ("lap", "shaping", "contact", "plan_hit_cost", "plan_hit")}
        speed_sum, stop_sum = torch.zeros(Bl, device=dev), torch.zeros(Bl, device=dev)
        c0 = pr.n_contacts.clone()
        tr0 = time.time()
        actor.eval(); critic.eval()
        for t in range(T):
            with torch.no_grad():
                sc, s = obs.scans.clone(), obs.state()
                pv_ = obs.priv()
                actor.norm.update(s); critic.norm.update(torch.cat([s, pv_], 1))
                if v2:
                    mean, log_std, v_hat = actor(sc, s)
                else:
                    mean, log_std = actor(sc, s)
                u = mean + log_std.exp() * torch.randn_like(mean)
                lp = tanh_logp(u, mean, log_std)
                v = critic(sc, s, pv_)
                v_true = env.sim.state[lid, dyn.IVX].float().clone()
                p = None
                if v2:
                    # the plan starts from what the policy thinks its speed is and what it last commanded
                    p = PS.roll(torch.tanh(u), v_hat.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
                    vest_err.append((v_hat - v_true).abs())
                out, p, _ = drv.decision(torch.tanh(u), p)
            S.append(s); X.append(sc); PV.append(pv_); U.append(u); LP.append(lp); V.append(v); VT.append(v_true)
            R.append(out["reward"]); D.append(out["restarted"].float())
            for k in ("lap", "shaping", "contact"):
                stats[k] += out[k]
            stats["plan_hit_cost"] += out["plan_hit_price"]
            stats["plan_hit"] += out["plan_hit"]
            sp = env.sim.state[lid, dyn.IVX].abs()
            speed_sum += sp; stop_sum += (sp < 0.3).float()
        with torch.no_grad():
            v_last = critic(obs.scans, obs.state(), obs.priv())
        t_roll = time.time() - tr0
        # GAE; a restart (episode end) cuts the trace and bootstraps from the value before it
        V_ = torch.stack(V); R_ = torch.stack(R); D_ = torch.stack(D)
        adv = torch.zeros_like(R_); gae = torch.zeros(Bl, device=dev)
        for t in reversed(range(T)):
            v_next = v_last if t == T - 1 else V_[t + 1]
            v_next = torch.where(D_[t] > 0, V_[t], v_next)
            delta = R_[t] + args.gamma * v_next - V_[t]
            gae = delta + args.gamma * args.lam * (1.0 - D_[t]) * gae
            adv[t] = gae
        ret = adv + V_
        bS, bX, bPV, bU, bVT = torch.cat(S), torch.cat(X), torch.cat(PV), torch.cat(U), torch.cat(VT)
        bLP, bA, bR, bV = torch.stack(LP).reshape(-1), adv.reshape(-1), ret.reshape(-1), V_.reshape(-1)
        N = bS.shape[0]
        actor.train(); critic.train()
        kls, cfs, vls, els = [], [], [], []
        tu0 = time.time()
        for ep in range(args.epochs):
            perm = torch.randperm(N, device=dev)
            for mb in perm.chunk(args.minibatches):
                if v2:
                    mean, log_std, v_hat = actor(bX[mb], bS[mb])
                    el = (v_hat - bVT[mb]).pow(2).mean()
                else:
                    mean, log_std = actor(bX[mb], bS[mb])
                    el = torch.zeros((), device=dev)
                lp = tanh_logp(bU[mb], mean, log_std)
                ratio = (lp - bLP[mb]).exp()
                a_ = bA[mb]; a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pl = -torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_).mean()
                vl = 0.5 * (critic(bX[mb], bS[mb], bPV[mb]) - bR[mb]).pow(2).mean()
                loss = pl + args.vf_coef * vl + args.vest_coef * el
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad)
                opt.step()
                with torch.no_grad():
                    kls.append(float((bLP[mb] - lp).mean())); cfs.append(float(((ratio - 1).abs() > args.clip).float().mean()))
                    vls.append(float(vl)); els.append(float(el))
        t_upd = time.time() - tu0
        secs = T * PS.DT
        mins = secs * Bl / 60.0
        dc = (pr.n_contacts - c0).sum(0)
        row = {"update": upd, "min": (time.time() - t0) / 60, "t_roll": t_roll, "t_upd": t_upd,
               "laps_per_min": float(stats["lap"].sum()) / mins, "speed": float(speed_sum.mean()) / T,
               "stopped": float(stop_sum.mean()) / T, "contacts_wall_per_min": float(dc[0]) / mins,
               "contacts_prop_per_min": float(dc[1]) / mins, "contacts_car_per_min": float(dc[2]) / mins,
               "plan_hit_share": float(stats["plan_hit"].mean()) / T, "reward": float(R_.mean()),
               "shaping": float(stats["shaping"].mean()) / T, "contact_cost": float(stats["contact"].mean()) / T,
               "kl": sum(kls) / len(kls), "clipfrac": sum(cfs) / len(cfs), "vloss": sum(vls) / len(vls),
               "vest_mae": float(torch.stack(vest_err).mean()) if vest_err else 0.0,
               "std": [round(float(x), 3) for x in actor.log_std.detach().exp()]}
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"[{time.strftime('%H:%M')}] 업데이트 {upd} | 바퀴 {row['laps_per_min']:.2f}/분 | 속도 {row['speed']:.2f} m/s | 멈춤 "
              f"{row['stopped'] * 100:.0f} % | 접촉 벽 {row['contacts_wall_per_min']:.2f} 상자 {row['contacts_prop_per_min']:.2f} 차 "
              f"{row['contacts_car_per_min']:.2f} /분 | 박는 plan {row['plan_hit_share'] * 100:.0f} % | 보상 {row['reward']:.4f} | "
              f"KL {row['kl']:.4f} 잘림 {row['clipfrac'] * 100:.0f} % | 속도 짐작 오차 {row['vest_mae']:.2f} m/s | 표준편차 {sum(row['std']) / len(row['std']):.3f} | "
              f"{t_roll:.1f}+{t_upd:.1f} s", flush=True)
        if upd % args.save_every == 0 or upd == args.updates:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "args": vars(args), "world": vars(world),
                        "priv_dim": priv_dim, "beams": obs.beams, "update": upd, "format": FORMAT2 if v2 else "plan_steps"},
                       os.path.join(run, f"plan_u{upd}.pt"))


if __name__ == "__main__":
    main()
