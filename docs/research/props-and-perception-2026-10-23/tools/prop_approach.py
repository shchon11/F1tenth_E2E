"""Research (2026-10-23): how our policies meet a prop on their path -- when they start to steer around it or brake, and
how often it ends in a hard hit. The v2 entry hits props 2.3x as often as v1 (races: 1.34 vs 0.59 /km), with or without
an opponent near, so this runs SOLO with the races' props (procedural obstacles 1.0, density 1.5, max 10, corridor
off; rep floor, dial 1.0489, the checkpoint's follower), 32 cars x 60 s per seed.

An APPROACH is a run of steps on which a live prop's centre lies ahead of the car (0 < x < 6 m in the car frame) and inside
its corridor (|y| < 0.45 m: it would be hit driving straight on). Per approach: the time to reach it (x / v) when it
entered; the time to reach it when the car first turned away (the prop's |y| growing by > 0.10 m from its entry value) and
when it first commanded a brake (plan v1 < speed - 0.5 m/s); and the outcome -- a hard (>= 1.5 m/s) prop onset during
the approach or within 0.25 s after, a softer one, or none.
RACE mode (4th argument = opponent kinds, as the finals' race cells): the same with one opponent; also counted -- whether the
prop had been moved from where it was first seen (pushed by a car), and the opponents' own prop onsets (the planners'
drivers cannot see props: `ProceduralObstacles.clearance`).
usage: prop_approach.py CKPT OUT.json [SEEDS=81,82] [KINDS]"""
import json
import math
import os
import statistics as st
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, mpc
from f1sim.learn import evaluate as ev
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "81,82").split(",")]
KINDS = sys.argv[4].split(",") if len(sys.argv) > 4 else [None]
TRACK = os.environ.get("ANAT_TRACK", "real:korea_2026_competition")
X_AHEAD, Y_CORR, HARD = 6.0, 0.45, 1.5
CUR = {"env": None, "in_step": False}
APPROACHES = []
STATE = {}


def observe(env, sim, r, action, v_plan):
    lr = env.learner
    B = lr.shape[0]
    ep = sim.track.env_props
    if ep is None:
        return
    poses, _n, _d, zlo, zhi = ep.slots(sim.eid)                    # (B, S, 3) ...
    live = zhi > zlo
    # where each prop was first seen this episode (rows of a race share their env's props)
    fresh_rows = sim.steps <= 1
    first = STATE.get("first")
    if first is None or first.shape != poses[..., :2].shape:
        first = STATE["first"] = poses[..., :2].clone()
    first = torch.where(fresh_rows[:, None, None], poses[..., :2], first)
    STATE["first"] = first
    disp_all = (poses[..., :2] - first).norm(dim=-1)
    st_ = r.state
    x, y, yaw = st_[:, dyn.IX], st_[:, dyn.IY], st_[:, dyn.IYAW]
    v = torch.hypot(st_[:, dyn.IVX], st_[:, dyn.IVY])
    c, s_ = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    dx, dy = poses[..., 0] - x[:, None], poses[..., 1] - y[:, None]
    xl, yl = c * dx + s_ * dy, -s_ * dx + c * dy
    cand = live & (xl > 0) & (xl < X_AHEAD) & (yl.abs() < Y_CORR)
    big = torch.full_like(xl, float("inf"))
    xs = torch.where(cand, xl, big)
    xmin, j = xs.min(1)
    has = torch.isfinite(xmin) & lr
    ylj = yl.gather(1, j[:, None])[:, 0]
    dispj = disp_all.gather(1, j[:, None])[:, 0]
    spec, vmax = env.tracker.spec, float(env.tracker.v_max)
    cap = torch.full((B,), 9.0, device=action.device)
    if spec.speed_mode == "wp4":
        _, _, vk = mpc.decode_wp4(action.float(), v_plan.float(), vmax, cap, spec); v1 = vk[:, 0]
    else:
        _, _, v1, _ = mpc.decode(action.float(), v_plan.float(), vmax, cap, spec)
    on, imp = r.contact_onsets, r.contact_impacts
    prop_on = on[:, 1] if on is not None else torch.zeros_like(has)
    prop_imp = imp[:, 1] if imp is not None else torch.zeros_like(v)
    fresh = sim.steps <= 1
    opp = ~lr
    STATE["opp_prop_onsets"] = STATE.get("opp_prop_onsets", 0) + int((prop_on & opp).sum())
    STATE["opp_prop_hard"] = STATE.get("opp_prop_hard", 0) + int((prop_on & opp & (prop_imp >= HARD)).sum())
    STATE["opp_dist"] = STATE.get("opp_dist", 0.0) + float((v * opp.float()).sum()) * float(sim.control_dt)
    H = dict(has=has.cpu(), xmin=xmin.cpu(), yl=ylj.cpu(), v=v.cpu(), v1=v1.cpu(), on=prop_on.cpu(), imp=prop_imp.cpu(),
             fresh=fresh.cpu(), lr=lr.cpu(), disp=dispj.cpu())
    t = STATE.setdefault("t", 0)
    open_ = STATE.setdefault("open", {})
    for b in range(B):
        if not bool(H["lr"][b]):
            continue
        a = open_.get(b)
        if bool(H["fresh"][b]) and a is not None:
            open_.pop(b); a = None                                    # a reset ends it unscored
        if a is not None and bool(H["on"][b]):
            a["hits"].append(float(H["imp"][b]))
        if bool(H["has"][b]):
            vb = max(float(H["v"][b]), 0.3)
            ttr = float(H["xmin"][b]) / vb
            if a is None:
                a = open_[b] = {"seed": STATE["seed"], "opp": STATE.get("kind"), "disp0": float(H["disp"][b]),
                                "t0": t, "ttr0": ttr, "x0": float(H["xmin"][b]), "y0": float(H["yl"][b]),
                                "v0": float(H["v"][b]), "turn_ttr": None, "brake_ttr": None, "hits": [], "steps": 0, "tail": 0}
            a["steps"] += 1; a["tail"] = 0
            if a["turn_ttr"] is None and abs(float(H["yl"][b])) > abs(a["y0"]) + 0.10:
                a["turn_ttr"] = ttr
            if a["brake_ttr"] is None and float(H["v1"][b]) < float(H["v"][b]) - 0.5:
                a["brake_ttr"] = ttr
        elif a is not None:
            a["tail"] += 1
            if a["tail"] > 10:                                        # 0.25 s past it: close
                APPROACHES.append(open_.pop(b))
    STATE["t"] = t + 1


_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step


def env_step(self, action):
    CUR.update(env=self, action=action, in_step=True, v_plan=self._tracker_plan_speed().clone())
    try:
        return _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False


def sim_step(self, *a, **kw):
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self:
        observe(env, self, r, CUR["action"], CUR["v_plan"])
    return r


gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
km = 0.0
OPP = {"onsets": 0, "hard": 0, "km": 0.0}
for seed in SEEDS:
  for kind in KINDS:
    STATE.clear(); STATE["seed"] = seed; STATE["kind"] = kind
    out = OUT.replace(".json", f"_{'solo' if kind is None else kind}_s{seed}.json")
    race = [] if kind is None else ["--race-size", "2", "--opp-slots", json.dumps([{"kind": kind, "speed_scale": [0.7, 1.0]}])]
    sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                *(race or ["--race-size", "1"]), "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                "--procedural-max-props", "10", "--procedural-raceline-corridor", "off",
                "--output", out, "--dial-value", "1.0489",
                "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
    ev.main()
    d = json.load(open(out))["nominal"]
    cs = d.get("contact_severity") or {}
    km += float(cs.get("learner_distance_m", 0.0)) / 1000.0
    OPP["onsets"] += STATE.get("opp_prop_onsets", 0); OPP["hard"] += STATE.get("opp_prop_hard", 0)
    OPP["km"] += STATE.get("opp_dist", 0.0) / 1000.0
    print(f"seed {seed} {kind or 'solo'}: {len(APPROACHES)} approaches so far, hard props in the meter "
          f"{int(sum(((cs.get('prop') or {}).get('fine_hist') or [])[15:]))}, {km:.1f} km", flush=True)

A = APPROACHES
hard = [a for a in A if a["hits"] and max(a["hits"]) >= HARD]
soft = [a for a in A if a["hits"] and max(a["hits"]) < HARD]
clear = [a for a in A if not a["hits"]]
q = lambda xs: (round(st.median(xs), 3) if xs else None)
res = {"ckpt": CK, "km": round(km, 2), "approaches": len(A), "per_km": round(len(A) / max(km, 1e-6), 2),
       "hard": len(hard), "soft": len(soft), "clear": len(clear), "p_hard": round(len(hard) / max(len(A), 1), 3),
       "hard_per_km": round(len(hard) / max(km, 1e-6), 2),
       "opponents_prop_onsets_per_km": round(OPP["onsets"] / max(OPP["km"], 1e-6), 2) if OPP["km"] else None,
       "opponents_prop_hard_per_km": round(OPP["hard"] / max(OPP["km"], 1e-6), 2) if OPP["km"] else None}
for name, grp in (("hard", hard), ("soft", soft), ("clear", clear)):
    res[name + "_stats"] = {
        "entry_ttr_s": q([a["ttr0"] for a in grp]), "entry_speed": q([a["v0"] for a in grp]),
        "entry_abs_y": q([abs(a["y0"]) for a in grp]),
        "displaced_share": round(sum(a["disp0"] > 0.2 for a in grp) / max(len(grp), 1), 3),
        "turned_share": round(sum(a["turn_ttr"] is not None for a in grp) / max(len(grp), 1), 3),
        "turn_ttr_s": q([a["turn_ttr"] for a in grp if a["turn_ttr"] is not None]),
        "braked_share": round(sum(a["brake_ttr"] is not None for a in grp) / max(len(grp), 1), 3),
        "brake_ttr_s": q([a["brake_ttr"] for a in grp if a["brake_ttr"] is not None])}
print(json.dumps(res, indent=1), flush=True)
json.dump({"summary": res, "approaches": A}, open(OUT, "w"))
