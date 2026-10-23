"""Research (2026-10-24): does our policy REACT when a car cuts in? A randomized test (the 10-07 cut-in RCT, rebuilt;
the scratchpad copy died with the 10-17 reboot).

Race of 2 against the raceline teacher at 0.6-0.8 of its line's speed (slower than us), no props. When the opponent is
0-3 m ahead along the track and 0.3-1.2 m beside us, idle, and the pair has not been triggered in the last 3 s, a coin:
  CUT   the opponent's lateral offset ramps over 0.5 s to our lateral position and holds 1.5 s (an injected `shift`
        event of `f1sim.opponent_events`: it closes the door);
  KEEP  nothing happens (the same trigger, the control arm).
Over the next 1.5 s: how much we slow (speed at the trigger minus the lowest speed), how far we move away from the
opponent sideways, whether a car contact (any, and >= 1.5 m/s) starts, and how far the opponent actually moved toward
us (the manipulation check). The REACTION is CUT minus KEEP, with bootstrap 95 % intervals. The state probe says v2
knows the opponent's sideways speed (R^2 0.75); this says whether it acts on it.
usage: cutin_rct.py CKPT OUT.json [SEEDS=31,32] [ENVS=64] [STEPS=4000]"""
import json
import math
import os
import random
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env
from f1sim.learn import evaluate as ev
from f1sim.opponent_events import EVENT_ID
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "31,32").split(",")]
ENVS = int(sys.argv[4]) if len(sys.argv) > 4 else 64
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
TRACK = os.environ.get("ANAT_TRACK", "real:korea_2026_competition")
GAP, LAT, WIN, COOL = (0.0, 3.0), (0.3, 1.2), 60, 120          # m, m, steps (1.5 s), steps (3 s)
HOLD_S, RAMP_S, MAX_OFF = 1.5, 0.5, 0.9
CUR = {"env": None, "in_step": False}
TRIALS = []
ST = {}


def frenet(sim, xy):
    s, d, _ = sim.track.project(xy, sim.tid, prev_idx=sim.cl_idx)
    return s, d


def pre_step(env):
    """Before the env steps: open trials (and inject the CUT arm's event)."""
    sim, evs = env.sim, env.events
    if evs is None or sim.M < 2:
        return
    B = sim.state.shape[0]
    lr = env.learner
    st = sim.state
    s, d = frenet(sim, st[:, :2])
    L = sim.track.length[sim.tid]
    o = sim.other_idx[:, 0]
    gap = torch.remainder(s[o] - s + L / 2, L) - L / 2                 # arc gap, + = the opponent ahead
    dlat = d[o] - d
    v = torch.hypot(st[:, dyn.IVX], st[:, dyn.IVY])
    if "last" not in ST:
        ST["last"] = torch.full((B,), -10 ** 6, dtype=torch.long, device=st.device)
        ST["t"] = 0
    t = ST["t"]
    idle = evs.kind[o] == 0
    trig = (lr & (gap > GAP[0]) & (gap < GAP[1]) & (dlat.abs() > LAT[0]) & (dlat.abs() < LAT[1]) & idle & (v > 1.0)
            & (t - ST["last"] > COOL) & (sim.steps > 40))
    for i in trig.nonzero().flatten().tolist():
        j = int(o[i])
        arm = "CUT" if random.random() < 0.5 else "KEEP"
        if arm == "CUT":
            off = max(-MAX_OFF, min(MAX_OFF, float(d[i] - d[j])))
            evs.kind[j] = EVENT_ID["shift"]; evs.t[j] = 0.0; evs.dur[j] = HOLD_S + 2 * RAMP_S
            evs.p0[j] = off; evs.p1[j] = RAMP_S
        ST["last"][i] = t
        TRIALS.append({"seed": ST["seed"], "arm": arm, "row": i, "opp": j, "t0": t, "v0": float(v[i]), "gap0": float(gap[i]),
                       "dlat0": float(dlat[i]), "d_me0": float(d[i]), "d_opp0": float(d[j]), "v_min": float(v[i]),
                       "contact": False, "contact_hard": False, "open": True, "steps": 0, "ep": int(sim.steps[i])})


def post_step(env, r, entry):
    sim = env.sim
    st = r.state
    s, d = frenet(sim, st[:, :2])
    v = torch.hypot(st[:, dyn.IVX], st[:, dyn.IVY])
    car = r.car_collision if r.car_collision is not None else torch.zeros_like(env.learner)
    imp = r.car_impact if r.car_impact is not None else torch.zeros_like(v)
    prev = ST.get("prev_car")
    if prev is None:
        prev = torch.zeros_like(car)
    onset = car & ~(prev & ~(entry == 0))
    ST["prev_car"] = car.clone()
    fresh = sim.steps <= 1
    for tr in TRIALS:
        if not tr["open"]:
            continue
        i, j = tr["row"], tr["opp"]
        if bool(fresh[i]):
            tr["open"] = False; tr["reset"] = True
            continue
        tr["steps"] += 1
        tr["v_min"] = min(tr["v_min"], float(v[i]))
        if bool(onset[i]):
            tr["contact"] = True
            if float(imp[i]) >= 1.5:
                tr["contact_hard"] = True
        if tr["steps"] >= WIN:
            side = 1.0 if tr["dlat0"] >= 0 else -1.0                       # the opponent's side of us
            tr["me_away"] = -(float(d[i]) - tr["d_me0"]) * side              # + = we moved away from it
            tr["opp_toward"] = -(float(d[j]) - tr["d_opp0"]) * side          # + = it moved toward us
            tr["slowing"] = tr["v0"] - tr["v_min"]
            tr["open"] = False
    ST["t"] += 1


_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step


def env_step(self, action):
    CUR.update(env=self, in_step=True)
    if self.sim.M > 1:
        pre_step(self)
    try:
        return _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False


def sim_step(self, *a, **kw):
    entry = self.steps.clone()
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self and self.M > 1:
        post_step(env, r, entry)
    return r


gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
for seed in SEEDS:
    ST.clear(); ST["seed"] = seed
    random.seed(1000 + seed)
    out = OUT.replace(".json", f"_s{seed}_eval.json")
    slot = {"kind": "raceline", "speed_scale": [0.6, 0.8], "events": ["shift"], "event_rate": 1e-6}   # ~never on its own: the test injects
    sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", str(ENVS), "--steps", str(STEPS),
                "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--spawn-runway", "3.0",
                "--race-size", "2", "--opp-slots", json.dumps([slot]), "--output", out, "--dial-value", "1.0489",
                "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
    n0 = len(TRIALS)
    ev.main()
    done = [t for t in TRIALS[n0:] if "slowing" in t]
    print(f"seed {seed}: {len(TRIALS) - n0} triggers, {len(done)} complete "
          f"(CUT {sum(t['arm'] == 'CUT' for t in done)}, KEEP {sum(t['arm'] == 'KEEP' for t in done)})", flush=True)

T = [t for t in TRIALS if "slowing" in t]
rng = random.Random(7)


def diff_ci(key, as_float=float):
    cut = [as_float(t[key]) for t in T if t["arm"] == "CUT"]; keep = [as_float(t[key]) for t in T if t["arm"] == "KEEP"]
    if not cut or not keep:
        return None
    m = lambda xs: sum(xs) / len(xs)
    point = m(cut) - m(keep)
    boots = []
    for _ in range(2000):
        c = [cut[rng.randrange(len(cut))] for _ in cut]; k = [keep[rng.randrange(len(keep))] for _ in keep]
        boots.append(m(c) - m(k))
    boots.sort()
    return {"cut": round(m(cut), 4), "keep": round(m(keep), 4), "diff": round(point, 4),
            "ci95": [round(boots[50], 4), round(boots[1949], 4)], "n_cut": len(cut), "n_keep": len(keep)}


res = {"ckpt": CK, "trials": len(T), "opp_toward_m": diff_ci("opp_toward"), "slowing_mps": diff_ci("slowing"),
       "me_away_m": diff_ci("me_away"), "contact_rate": diff_ci("contact", lambda x: float(bool(x))),
       "hard_contact_rate": diff_ci("contact_hard", lambda x: float(bool(x)))}
print(json.dumps(res, indent=1), flush=True)
json.dump({"summary": res, "trials": T}, open(OUT, "w"))
