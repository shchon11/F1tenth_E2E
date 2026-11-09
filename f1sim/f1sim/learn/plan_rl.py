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
* `--hold-p` (11-09, agreed with the user): that share of new plans is driven 0.2-0.4 s (`--hold-len`) without a new
  one, from where the car is (`plan_steps.shift_plan`). Re-planning every 0.1 s, only a plan's first 0.1 s was ever
  driven, and its later points were a rough shape (0.3 / 0.5 s on, 12 / 44 cm off where the car went) that MAP and
  the real car's controller follow. The policy cannot tell which plan will be held; the critic is told how many
  decisions the plan in hand still drives; held decisions are left out of the policy loss and the advantage runs
  through them undiscounted by lambda. Drive time (evaluation, console, car) re-plans every 0.1 s as before.

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
from .plan_policy import PlanObs3, FORMAT3, STATE_DIM3, plan3_models
from .. import plan_steps3 as P3
from .plan_reward import PlanReward

WORLD = os.path.join(os.path.dirname(__file__), "configs", "plan_world_korea25.json")


def build(world: str, envs: int, seed: int, device, overrides=None, fmt: str = "v2"):
    a = argparse.Namespace(**json.load(open(world)))
    a.envs, a.seed, a.device = envs, seed, str(device)
    for k, v in (overrides or {}).items():
        setattr(a, k, v)
    env = SR.build_env(a, device)
    if fmt == "v3":
        # the plan knows the control delay (11-09): the training's execution goes through the same delay, lags and table;
        # set before the graphs are captured (the roll reads it then)
        _lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), device)
        env.enable_ideal(lag=True, turn=plan3_models(env, _lim, device)[1])
    else:
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

    def decision(self, a: torch.Tensor, p=None, charge=None):
        """drive one decision: `p` a plan already made (v2: from the policy's speed estimate and last command), else
        made from `a` at the car's true state (v1). `charge` (B_l,) bool: rows whose plan is new this decision -- the
        plan-hit cost is charged once, when a plan is drawn (a held plan's later decisions pay none)"""
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
        price, hit = pc["price"], pc["hit"].float()
        if charge is not None:
            price, hit = torch.where(charge, price, torch.zeros_like(price)), hit * charge.float()
        out = self.pr.decision(pts, price, restarted)
        out["plan_hit_price"] = out.pop("plan_hit")
        out["plan_hit"] = hit
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
    ap.add_argument("--format", default="v2", choices=("v1", "v2", "v3"),
                    help="v2 (11-08): raw sensors only, the plan starts from the policy's own speed estimate")
    ap.add_argument("--vest-coef", type=float, default=1.0, help="v2: weight of the speed estimate's loss [per (m/s)^2]")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--init", default="", help="v3: start the actor from this v2 checkpoint (its 36 inputs + the servo angle at 0), "
                    "the critic's scan encoder too")
    ap.add_argument("--actor-freeze", type=int, default=0, help="updates in which only the critic learns (a started actor)")
    ap.add_argument("--plan-hit-share", type=float, default=0.1, help="the plan-hit cost's share of a contact (0 = off)")
    ap.add_argument("--plan-hit-s", type=float, default=1.0, help="the plan-hit cost looks this far into the plan [s]")
    ap.add_argument("--hold-p", type=float, default=0.0,
                    help="v2/v3: the share of new plans driven longer than 0.1 s without a new one (11-09, the user: 0.2)")
    ap.add_argument("--hold-len", default="2,4", help="a held plan drives this many decisions, drawn uniformly (2,4 = 0.2-0.4 s)")
    ap.add_argument("--hold-clear", type=float, default=0.0,
                    help="hold only when no other car and no box is within this many metres; one that comes inside it ends a "
                         "hold at once (11-09, the user: racing needs the reaction -- holds near them teach timid plans). 0 = anywhere")
    ap.add_argument("--init-critic", action="store_true", help="with --init: the critic too, when its inputs match (a run continued)")
    ap.add_argument("--tail-consistency", type=float, default=0.0,
                    help="v2: only the plan's first 0.1 s is the RL action (its noise and log-probability); the other 18 outputs are "
                         "taught, with this weight, to say what the NEXT plan says for the same moments (11-09, the user chose (가): "
                         "the previous plan's 0.1-1.0 s pulled toward the new plan's 0-0.9 s, never the other way). The output layer "
                         "alone learns it: nothing that drives is touched. 0 = off (all 20 outputs are actions)")
    ap.add_argument("--memory", type=int, default=0,
                    help="v2: a GRU of this size in the actor (11-09, the user: \"A1\" -- adapt by remembering); 0 = none")
    ap.add_argument("--mem-chunk", type=int, default=32, help="the memory's update replays sequences of this many decisions")
    ap.add_argument("--speed-std-floor", type=float, default=0.0,
                    help="v2: the exploration noise's floor on the first step's speed output (11-09: it had shrunk to 0.17)")
    ap.add_argument("--removal-p", type=float, default=-1.0,
                    help="the share of races whose obstacles are taken away mid-episode (-1 = the world's `removal_p`; 0 = none). "
                         "11-09: plan_rl never ran it, so no plan policy had seen the competition's removal")
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
    env, lim, world = build(args.world, args.envs, args.seed, dev, fmt=args.format)
    lid = env.learner_ids; Bl = int(lid.numel())
    pr = PlanReward(env, args.gamma, dev)
    pr.plan_hit_share = float(args.plan_hit_share)
    pr.plan_hit_n = None if args.plan_hit_s >= PS.N_STEP * PS.DT else int(round(args.plan_hit_s / PS.DT * PS.SUB))
    v3 = args.format == "v3"
    v2 = args.format == "v2" or v3                     # v3 is v2 with the delay-aware plan and the servo angle
    obs = (PlanObs3 if v3 else PlanObs2 if v2 else PlanObs)(env, lim, pr.goal)
    act3, turn3 = plan3_models(env, lim, dev) if v3 else (None, None)
    drv = Driver(env, lim, pr, obs)
    hold = args.hold_p > 0
    if hold and not v2:
        raise SystemExit("--hold-p needs --format v2 or v3")
    h_lo, h_hi = (int(x_) for x_ in args.hold_len.split(","))
    hold_left = torch.zeros(Bl, dtype=torch.long, device=dev)      # decisions a held plan still drives after this one
    hold_k = torch.zeros(Bl, dtype=torch.long, device=dev)         # whole steps of the held plan already driven
    p_held = None
    others = env.sim.other_idx[lid] if env.M > 1 else None

    def near_threat() -> torch.Tensor:
        """(Bl,) another car of the race or a live box within `--hold-clear` metres of the car"""
        st_ = env.sim.state[lid, :2]
        near = torch.zeros(Bl, dtype=torch.bool, device=dev)
        if others is not None:
            d = (env.sim.state[others, :2] - st_[:, None, :]).norm(dim=-1)
            if getattr(env.sim, "absent_any", False):
                d = torch.where(env.sim.absent[others], torch.full_like(d, float("inf")), d)
            near |= d.min(1).values < args.hold_clear
        if env.procedural is not None:
            near |= env.procedural.clearance(st_, env.sim.eid[lid]) < args.hold_clear
        return near
    # the critic is told how many decisions the plan in hand still drives (the policy is not: it cannot know which plan
    # will be held, so every plan's later steps have to be ones the car can drive)
    priv_h = (lambda: torch.cat([obs.priv(), hold_left[:, None].float() / max(1, h_hi - 1)], 1)) if hold else obs.priv
    priv_dim = int(priv_h().shape[1])
    # the competition's sessions: the obstacles stay where they are for a while, then go (`state_rl.PropRemoval`, per race,
    # at a time drawn in removal_s after the episode's start)
    rm_p = float(getattr(world, "removal_p", 0.0)) if args.removal_p < 0 else float(args.removal_p)
    rm_s = tuple(getattr(world, "removal_s", (10.0, 30.0)))
    remover = SR.PropRemoval(env, rm_p, float(rm_s[0]), float(rm_s[1]), seed=args.seed) if rm_p > 0 else None
    if args.memory and (not v2 or v3):
        raise SystemExit("--memory needs --format v2")
    actor = (Actor2(obs.beams, std0=args.std0, state_dim=STATE_DIM3) if v3 else
             Actor2(obs.beams, std0=args.std0, memory=args.memory) if v2 else Actor(obs.beams, std0=args.std0)).to(dev)
    tailc = args.tail_consistency > 0
    if tailc and (not v2 or v3 or args.hold_p > 0):
        raise SystemExit("--tail-consistency needs --format v2 and no holding")
    rl_mask = torch.zeros(PS.ACT_DIM, dtype=torch.bool, device=dev)
    rl_mask[0] = True; rl_mask[PS.N_STEP] = True                   # the first step's steering and speed
    rl_idx = torch.nonzero(rl_mask)[:, 0]
    if args.speed_std_floor > 0:
        actor.log_std_floor[PS.N_STEP] = math.log(args.speed_std_floor)        # the first step's speed: the one driven
    if args.memory:
        obs.mem_h = torch.zeros(Bl, args.memory, device=dev)
    critic = Critic(obs.beams, priv_dim, state_dim=int(obs.state().shape[1])).to(dev)
    if args.init:
        # 11-09: every v3 run from scratch learned to stand still (the lagging car crashed while learning to drive); v3's
        # commands take v2's step sizes, so a v2 policy's outputs mean the same commands -- start from one, RL on from there
        ck0 = torch.load(args.init, map_location=dev, weights_only=False)
        sd = dict(ck0["actor"])
        w = sd["trunk.0.weight"]
        sd["trunk.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], actor.trunk[0].in_features - w.shape[1], device=w.device)], 1)
        for k_, fill in (("norm.mean", 0.0), ("norm.var", 1.0)):
            v_ = sd[k_]
            sd[k_] = torch.cat([v_, torch.full((actor.norm.mean.numel() - v_.numel(),), fill, device=v_.device)])
        for k_, v_ in actor.state_dict().items():
            if k_ not in sd:                                    # a memory actor from one without: its GRU fresh, output zero
                sd[k_] = v_
        actor.load_state_dict(sd)
        enc = {k_[4:]: v_ for k_, v_ in ck0["critic"].items() if k_.startswith("enc.")}
        critic.enc.load_state_dict(enc)
        if args.init_critic:
            c_sd = ck0["critic"]
            if all(k_ in c_sd and c_sd[k_].shape == v_.shape for k_, v_ in critic.state_dict().items()):
                critic.load_state_dict(c_sd)
                print("[plan_rl] critic too (its inputs match)", flush=True)
            else:
                print("[plan_rl] critic NOT taken: its inputs differ", flush=True)
        print(f"[plan_rl] actor from {args.init} (update {ck0.get('update')}), critic's scan encoder too", flush=True)
    opt = torch.optim.Adam([{"params": list(actor.parameters())}, {"params": list(critic.parameters())}], lr=args.lr, eps=1e-5)
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
        if upd <= args.actor_freeze:
            opt.param_groups[0]["lr"] = 0.0                    # the critic catches up with a started actor first
        S, X, PV, U, LP, V, R, D, VT, FM, HM = [], [], [], [], [], [], [], [], [], [], []
        V0, D0, ST0, PTS, PVS = [], [], [], [], []
        n_removed = 0
        near_sum = torch.zeros(Bl, device=dev)
        vest_err = []
        stats = {k: torch.zeros(Bl, device=dev) for k in ("lap", "shaping", "contact", "plan_hit_cost", "plan_hit")}
        speed_sum, stop_sum = torch.zeros(Bl, device=dev), torch.zeros(Bl, device=dev)
        c0 = pr.n_contacts.clone()
        tr0 = time.time()
        actor.eval(); critic.eval()
        for t in range(T):
            with torch.no_grad():
                sc, s = obs.scans.clone(), obs.state()
                pv_ = priv_h()
                if hold and args.hold_clear > 0:
                    # another car or a box within reach: no hold, and one in progress ends now (the car re-plans as it will
                    # on the road)
                    near = near_threat()
                    hold_left = torch.where(near, torch.zeros_like(hold_left), hold_left)
                    near_sum += near.float()
                newp = hold_left == 0                              # rows that draw a plan now (the others drive a held one)
                actor.norm.update(s); critic.norm.update(torch.cat([s, pv_], 1))
                if args.memory:
                    HM.append(obs.mem_h.clone())                  # the state this decision starts from
                    mean, log_std, v_hat = actor(sc, s, obs.mem_h)
                    obs.mem_h = actor.h_out.detach()               # a restart below clears its rows (`obs.reset_rows`)
                elif v2:
                    mean, log_std, v_hat = actor(sc, s)
                else:
                    mean, log_std = actor(sc, s)
                if tailc:
                    u = mean + log_std.exp() * torch.randn_like(mean) * rl_mask      # the tail: its mean, no noise
                    lp = tanh_logp(u[:, rl_idx], mean[:, rl_idx], log_std[:, rl_idx])
                else:
                    u = mean + log_std.exp() * torch.randn_like(mean)
                    lp = tanh_logp(u, mean, log_std)
                v = critic(sc, s, pv_)
                v_true = env.sim.state[lid, dyn.IVX].float().clone()
                p = None
                if v3:
                    # the plan knows the control delay: from the speed estimate, the servo angle the last plan left and the
                    # command still in effect
                    p = P3.roll(torch.tanh(u), v_hat.clamp(-PS.V_REV, PS.V_FWD), obs.servo,
                                torch.stack([obs.cmd_d, obs.cmd_v], 1), lim, act3, turn3)
                    vest_err.append((v_hat - v_true).abs())
                elif v2:
                    # the plan starts from what the policy thinks its speed is and what it last commanded
                    p = PS.roll(torch.tanh(u), v_hat.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
                    vest_err.append((v_hat - v_true).abs())
                    if tailc:
                        V0.append(v_hat.clamp(-PS.V_REV, PS.V_FWD).clone()); D0.append(obs.cmd_d.clone())
                        ST0.append(env.sim.state[lid].clone())
                if hold:
                    # a new plan is kept; rows inside a held plan drive its next 0.1 s, re-expressed from where they are
                    sel = lambda A, B_: torch.where(newp.view(-1, *([1] * (A.dim() - 1))), A, B_)
                    p_held = p if p_held is None else {k_: sel(p[k_], p_held[k_]) if p[k_].dim() and p[k_].shape[0] == Bl
                                                       else p[k_] for k_ in p}
                    hold_k = torch.where(newp, torch.zeros_like(hold_k), hold_k + 1)
                    p = PS.shift_plan(p_held, hold_k)
                    more = torch.randint(h_lo - 1, h_hi, (Bl,), device=dev)
                    start = newp & (torch.rand(Bl, device=dev) < args.hold_p)
                    if args.hold_clear > 0:
                        start &= ~near
                    hold_left = torch.where(start, more, torch.where(newp, torch.zeros_like(hold_left), hold_left - 1))
                out, p, (xw_, yw_, _) = drv.decision(torch.tanh(u), p, charge=newp if hold else None)
                if tailc:
                    ii_ = PT_IDX.to(dev)
                    PTS.append(torch.stack([xw_[:, ii_], yw_[:, ii_]], -1)); PVS.append(p["pv"].clone())
                if remover is not None:
                    was = remover.removed.clone()
                    remover.step()
                    n_removed += int((remover.removed & ~was).sum())
                if hold:
                    hold_left = torch.where(out["restarted"], torch.zeros_like(hold_left), hold_left)
            S.append(s); X.append(sc); PV.append(pv_); U.append(u); LP.append(lp); V.append(v); VT.append(v_true)
            FM.append(newp.float())
            R.append(out["reward"]); D.append(out["restarted"].float())
            for k in ("lap", "shaping", "contact"):
                stats[k] += out[k]
            stats["plan_hit_cost"] += out["plan_hit_price"]
            stats["plan_hit"] += out["plan_hit"]
            sp = env.sim.state[lid, dyn.IVX].abs()
            speed_sum += sp; stop_sum += (sp < 0.3).float()
        with torch.no_grad():
            v_last = critic(obs.scans, obs.state(), priv_h())
        t_roll = time.time() - tr0
        # GAE; a restart (episode end) cuts the trace and bootstraps from the value before it
        V_ = torch.stack(V); R_ = torch.stack(R); D_ = torch.stack(D); FM_ = torch.stack(FM)
        adv = torch.zeros_like(R_); gae = torch.zeros(Bl, device=dev)
        for t in reversed(range(T)):
            v_next = v_last if t == T - 1 else V_[t + 1]
            v_next = torch.where(D_[t] > 0, V_[t], v_next)
            delta = R_[t] + args.gamma * v_next - V_[t]
            # inside a held plan the trace runs on undiscounted by lambda: a plan's advantage is all it drove, then
            # lambda from the next plan on
            lam_t = args.lam if t == T - 1 else torch.where(FM_[t + 1] > 0, torch.full_like(gae, args.lam), torch.ones_like(gae))
            gae = delta + args.gamma * lam_t * (1.0 - D_[t]) * gae
            adv[t] = gae
        ret = adv + V_
        bS, bX, bPV, bU, bVT = torch.cat(S), torch.cat(X), torch.cat(PV), torch.cat(U), torch.cat(VT)
        bLP, bA, bR, bV = torch.stack(LP).reshape(-1), adv.reshape(-1), ret.reshape(-1), V_.reshape(-1)
        bFM = FM_.reshape(-1)                                  # the policy learns from the decisions it made
        N = bS.shape[0]
        if args.memory:
            # whole sequences of `mem_chunk` decisions per car, replayed from the state each began with; a restart inside
            # one clears the state for the decisions after it
            Lc = args.mem_chunk; nC = T // Lc
            bHM, bD = torch.cat(HM), D_.reshape(-1)
            tt = torch.arange(nC * Lc, device=dev).view(nC, Lc)
            seq_idx = (tt[:, :, None] * Bl + torch.arange(Bl, device=dev)[None, None, :]).permute(1, 0, 2).reshape(Lc, nC * Bl)

        def batches():
            if args.memory:
                for sq in torch.randperm(seq_idx.shape[1], device=dev).chunk(args.minibatches):
                    yield seq_idx[:, sq]
            else:
                for mb_ in torch.randperm(N, device=dev).chunk(args.minibatches):
                    yield mb_

        def actor_on(sel):
            if not args.memory:
                f = actor.features(bX[sel], bS[sel])
                return (*actor.heads(f), sel, f)
            mb_ = sel.reshape(-1)
            f = actor.features(bX[mb_], bS[mb_]).view(sel.shape[0], sel.shape[1], -1)
            h = bHM[sel[0]]
            outs = []
            for i in range(sel.shape[0]):
                if i > 0:
                    h = h * (1.0 - bD[sel[i - 1]])[:, None]
                fi, h = actor.remember(f[i], h)
                outs.append(fi)
            fc = torch.cat(outs, 0)
            return (*actor.heads(fc), mb_, fc)

        actor.train(); critic.train()
        kls, cfs, vls, els, cls_, ce5 = [], [], [], [], [], []
        if tailc:
            bV0, bD0, bST0 = torch.cat(V0), torch.cat(D0), torch.cat(ST0)
            bPTS, bPVS = torch.cat(PTS), torch.cat(PVS)
            NB = bV0.shape[0]
            # a decision has a next plan to learn from unless it is the rollout's last or its car restarted in it
            c_ok = torch.ones(NB, dtype=torch.bool, device=dev); c_ok[-Bl:] = False
            c_ok &= D_.reshape(-1) <= 0
        tu0 = time.time()
        for ep in range(args.epochs):
            for sel in batches():
                if v2:
                    mean, log_std, v_hat, mb, f_heads = actor_on(sel)
                    el = (v_hat - bVT[mb]).pow(2).mean()
                else:
                    mb = sel
                    mean, log_std = actor(bX[mb], bS[mb])
                    el = torch.zeros((), device=dev)
                if tailc:
                    lp = tanh_logp(bU[mb][:, rl_idx], mean[:, rl_idx], log_std[:, rl_idx])
                else:
                    lp = tanh_logp(bU[mb], mean, log_std)
                ratio = (lp - bLP[mb]).exp()
                m_ = bFM[mb]; n_ = m_.sum().clamp_min(1.0)
                a_ = bA[mb]; mu_ = (a_ * m_).sum() / n_
                a_ = (a_ - mu_) / (((a_ - mu_).pow(2) * m_).sum() / n_).sqrt().add(1e-8)
                pl = -(torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_) * m_).sum() / n_
                vl = 0.5 * (critic(bX[mb], bS[mb], bPV[mb]) - bR[mb]).pow(2).mean()
                loss = pl + args.vf_coef * vl + args.vest_coef * el
                if tailc:
                    # (가): this decision's plan, its first step as driven, its tail from the output layer on features it
                    # does not move, laid where the car was -- against the next decision's plan for the same moments
                    ok_ = c_ok[mb]
                    if bool(ok_.any()):
                        mb_c = mb[ok_]
                        tail_mean = actor.pi(f_heads[ok_].detach())
                        a_c = torch.where(rl_mask[None, :], torch.tanh(bU[mb_c]), torch.tanh(tail_mean))
                        pc_ = PS.roll(a_c, bV0[mb_c], bD0[mb_c], lim)
                        xw_c, yw_c, _ = plan_world(pc_, bST0[mb_c])
                        ii_ = PT_IDX.to(dev)
                        pts_c = torch.stack([xw_c[:, ii_], yw_c[:, ii_]], -1)
                        tgt, tgv = bPTS[mb_c + Bl], bPVS[mb_c + Bl]
                        lpos = nn.functional.smooth_l1_loss(pts_c[:, 1:], tgt[:, :-1], reduction="none").sum(-1)
                        lv = nn.functional.smooth_l1_loss(pc_["pv"][:, 1:], tgv[:, :-1], reduction="none")
                        lc = (lpos + lv).mean()
                        loss = loss + args.tail_consistency * lc
                        cls_.append(float(lc.detach()))
                        ce5.append(float((pts_c[:, 5] - tgt[:, 4]).norm(dim=-1).median()))
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad)
                opt.step()
                with torch.no_grad():
                    kls.append(float(((bLP[mb] - lp) * m_).sum() / n_))
                    cfs.append(float((((ratio - 1).abs() > args.clip).float() * m_).sum() / n_))
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
               "vest_mae": float(torch.stack(vest_err).mean()) if vest_err else 0.0, "held_share": 1.0 - float(FM_.mean()), "removed": n_removed,
               "tail_loss": (sum(cls_) / len(cls_)) if cls_ else None, "tail_next_05": (sum(ce5) / len(ce5)) if ce5 else None, "near_share": float(near_sum.mean()) / T,
               "std": [round(float(x), 3) for x in actor.log_std.detach().exp()]}
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"[{time.strftime('%H:%M')}] 업데이트 {upd} | 바퀴 {row['laps_per_min']:.2f}/분 | 속도 {row['speed']:.2f} m/s | 멈춤 "
              f"{row['stopped'] * 100:.0f} % | 접촉 벽 {row['contacts_wall_per_min']:.2f} 상자 {row['contacts_prop_per_min']:.2f} 차 "
              f"{row['contacts_car_per_min']:.2f} /분 | 박는 plan {row['plan_hit_share'] * 100:.0f} % | 보상 {row['reward']:.4f} | "
              f"KL {row['kl']:.4f} 잘림 {row['clipfrac'] * 100:.0f} % | 속도 짐작 오차 {row['vest_mae']:.2f} m/s | 표준편차 {sum(row['std']) / len(row['std']):.3f} | "
              + (f"붙잡은 결정 {row['held_share'] * 100:.0f} % | " if hold else "")
              + (f"가까이 차·상자 {row['near_share'] * 100:.0f} % | " if hold and args.hold_clear > 0 else "")
              + (f"상자 치움 {n_removed}판 | " if remover is not None else "")
              + (f"뒷부분 손실 {row['tail_loss']:.3f}, 다음 plan과 0.5 s 차이 {row['tail_next_05'] * 100:.0f} cm | " if row.get("tail_loss") is not None else "")
              + f"{t_roll:.1f}+{t_upd:.1f} s", flush=True)
        if upd % args.save_every == 0 or upd == args.updates:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "args": vars(args), "world": vars(world),
                        "priv_dim": priv_dim, "beams": obs.beams, "update": upd, "format": FORMAT3 if v3 else FORMAT2 if v2 else "plan_steps"},
                       os.path.join(run, f"plan_u{upd}.pt"))


if __name__ == "__main__":
    main()
