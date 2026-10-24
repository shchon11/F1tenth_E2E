"""Research (2026-10-24): WHERE does a fine-tuned policy differ from its start? Shadow-run a second checkpoint on the
first one's own states and compare the first plan speed (v1, what MAP drives) step by step, by situation.

The shield hinge at coef 20 (v2PHsh20_s2424) cut hard car contacts at equal race speed but also slowed SOLO laps (8.89 ->
9.12 s): either it learnt to slow near a car in our path AND everywhere (a global shift), or only near cars. Here the
start policy (A) drives the finals cells; B gets the same observations (and dial) with its own hidden state, reset
together, and does not drive. Per learner step: v1 of A and of B [m/s], our speed, and the opponent's relative state
(ego frame: dx, |dy|, our speed advantage, lateral rate toward us) and the nearest live prop ahead. Reported: mean
v1(B) - v1(A) by situation -- nothing near / a prop ahead (0 < x < 6 m, |y| < 0.45; no car within 3 m) / a car within
3 m outside the rules / inside the in-lane rule (wholly ahead, |dy| < 0.35) / inside pass-slow (ahead, |dy| < 0.8,
faster by > 1 m/s). v1 is the plan's first speed column, (a + 1) / 2 x v_max: what the hinge acts on.
usage: policy_diff.py CKPT_A CKPT_B OUT.json [SEEDS=81] [KINDS=solo,forzaeth,lane_switch]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, mpc
from f1sim.learn import common
from f1sim.learn import evaluate as ev
from f1sim.learn.model import load_checkpoint

CKA, CKB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "81").split(",")]
KINDS = (sys.argv[5] if len(sys.argv) > 5 else "solo,forzaeth,lane_switch").split(",")
TRACK = "real:korea_2026_competition"
CUR = {"env": None}
ACC = {}
I_V1 = mpc.plan_layout("wp4").speed.start
_orig_sp = common.student_policy


def student_policy(model, env, device=None, deterministic=True, dial=None):
    pa = _orig_sp(model, env, device, deterministic=deterministic, dial=dial)
    mb, _ = load_checkpoint(CKB, device, allow_oracle=True, allow_conditional=True)
    mb.eval()
    pb = _orig_sp(mb, env, device, deterministic=deterministic, dial=dial)
    vmax = float(env.cfg.v_max_policy) if hasattr(env, "cfg") and hasattr(env.cfg, "v_max_policy") else None

    def run(obs):
        a = pa(obs)
        b = pb(obs)
        record(env, a, b)
        return a

    def reset(done=None):
        pa.reset(done)
        pb.reset(done)

    run.reset = reset
    return run


def record(env, a, b):
    lr = env.learner
    if a.shape[0] != lr.shape[0]:
        return
    vmax = float(env.tracker.v_max)
    va = (a[:, I_V1].float() + 1) / 2 * vmax
    vb = (b[:, I_V1].float() + 1) / 2 * vmax
    st = env.sim.state
    vme = torch.hypot(st[:, dyn.IVX], st[:, dyn.IVY])
    lab = torch.zeros_like(va, dtype=torch.long)                       # 0 nothing near
    ep = env.sim.track.env_props
    if ep is not None:
        poses, _n, _d, zlo, zhi = ep.slots(env.sim.eid)
        live = zhi > zlo
        c, s_ = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
        px, py = poses[..., 0] - st[:, dyn.IX][:, None], poses[..., 1] - st[:, dyn.IY][:, None]
        xl, yl = c * px + s_ * py, -s_ * px + c * py
        prop = (live & (xl > 0) & (xl < 6.0) & (yl.abs() < 0.45)).any(1)
        lab = torch.where(prop, torch.ones_like(lab), lab)
    if env.sim.M > 1:
        dx, dy, _dyaw, dvx, dvy, dist, present = env._opp_rel(st)
        dx, dy, dvx, dvy, dist = dx[:, 0], dy[:, 0], dvx[:, 0], dvy[:, 0], dist[:, 0]
        o = env.sim.other_idx[:, 0]
        vop = torch.hypot(st[o, dyn.IVX], st[o, dyn.IVY])
        dv = vme - vop
        close = dist < 3.0
        inlane = (dx - 0.5 > 0.1) & (dy.abs() < 0.35) & (dx < 6.0)
        passslow = (dx > 0) & (dy.abs() < 0.8) & (dv > 1.0)
        lab = torch.where(close, torch.full_like(lab, 2), lab)
        lab = torch.where(passslow, torch.full_like(lab, 4), lab)
        lab = torch.where(inlane, torch.full_like(lab, 3), lab)
    d = (vb - va)[lr]; lab = lab[lr]
    k = ACC.setdefault(CUR["kind"], {x: torch.zeros(5, device=d.device) for x in ("n", "sum", "sq", "va")})
    k["n"].index_add_(0, lab, torch.ones_like(d)); k["sum"].index_add_(0, lab, d); k["sq"].index_add_(0, lab, d * d)
    k["va"].index_add_(0, lab, va[lr])


common.student_policy = student_policy
for seed in SEEDS:
    for kind in KINDS:
        CUR["kind"] = kind
        out = os.path.join(os.path.dirname(OUT) or ".", f"_pd_{os.path.basename(OUT)}_{seed}_{kind}.json")
        race = [] if kind == "solo" else ["--race-size", "2", "--opp-slots", json.dumps([{"kind": kind, "speed_scale": [0.7, 1.0]}])]
        sys.argv = ["evaluate", CKA, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", "off", *race,
                    "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        k = ACC[kind]
        n, s, q, va = (k[x].cpu() for x in ("n", "sum", "sq", "va"))
        names = ["nothing near", "prop ahead", "car within 3 m, outside the rules", "in-lane (wholly ahead, |dy|<0.35)",
                 "pass-slow region"]
        print(f"seed {seed} {kind}: " + " | ".join(f"{names[i]}: n {int(n[i])} v1(A) {float(va[i] / max(n[i], 1)):.2f} "
              f"dv1 {float(s[i] / max(n[i], 1)):+.3f}" for i in range(5) if n[i] > 0), flush=True)
res = {kind: {x: [float(v) for v in ACC[kind][x].cpu()] for x in ("n", "sum", "sq", "va")} for kind in ACC}
json.dump({"a": CKA, "b": CKB, "seeds": SEEDS, "labels": ["nothing", "prop_ahead", "car_close_other", "inlane", "passslow"],
           "acc": res}, open(OUT, "w"))
