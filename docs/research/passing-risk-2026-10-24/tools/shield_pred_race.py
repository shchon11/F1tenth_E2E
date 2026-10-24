"""Research (2026-10-24): a PREDICTIVE oracle speed shield -- does anticipating an opponent's LATERAL move remove the hard
car contacts the in-lane shield cannot see?

The in-lane oracle shield (shield_race.py: cap our speed for a car wholly ahead within |y| < 0.35) cut hard car contacts by
23 % (1.05 -> 0.81 /km) at no speed cost. In the anatomy of the entry's hard car contacts (anat_v2entry, 31 events) the
opponent was inside |dy| < 0.35 only 23 % of the time 0.5 s before; in 84 % it moved toward us (0.3 m over that 0.5 s).
So most hard hits are LATERAL CONVERGENCE while we pass. Here an opponent counts as in our path when, in Frenet
coordinates, it is ahead of us (body gap > 0.1 m) within 6 m and its lateral offset to us, extrapolated at the current
lateral rate (constant velocity), comes under W within H seconds -- unless we would already be a car length past it then.
The cap is the same as the in-lane shield's: v_safe = v_opp + sqrt(2 a_b max(0, gap - g_min - (v_me - v_opp)+ tau)).
H = 0 is the in-lane shield in Frenet coordinates (the comparison that isolates the prediction).
PASS-SLOW variant (DV > 0, SHIELD_ALONGSIDE=1): the cap allows DV m/s over the opponent's speed --
v_safe = v_opp + DV + sqrt(2 a_b max(0, gap - g_min - (v_me - v_opp - DV)+ tau)) -- and holds while the opponent's centre is
ahead of ours (alongside too: with DV > 0 we can never be held beside it). From the close-situation data of the entry
(probe_v2entry, 84k samples): hard contacts come from passing faster than ~1 m/s within ~0.8 m laterally (2.5-7.4 % within
0.5 s there, <= 0.4 % beyond 0.8 m, 0 % under 1 m/s); W 0.8 / DV 1.0 holds 88 % of the pre-contact samples.
SHIELD_MEASURE_ONLY=1: count, do not cap. True states (oracle). Finals race cells (seeds, 4 kinds, props on the line;
RACE_CORRIDOR=on keeps them off the raceline, where the opponents do not push them into our path).
usage: shield_pred_race.py CKPT OUT_DIR SEEDS [H=1.0] [A_B=3.0] [W=0.35] [TAU=0.2] [GMIN=0.2] [DV=0.0]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc
from f1sim.learn import evaluate as ev

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
H = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
A_B = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
W = float(sys.argv[6]) if len(sys.argv) > 6 else 0.35
TAU = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2
GMIN = float(sys.argv[8]) if len(sys.argv) > 8 else 0.2
DV = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
ALONGSIDE = bool(os.environ.get("SHIELD_ALONGSIDE"))
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
TRACK = "real:korea_2026_competition"
CAR_LEN = 0.5
MEASURE = bool(os.environ.get("SHIELD_MEASURE_ONLY"))
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None}
STAT = {"steps": 0, "active": 0, "cut_sum": 0.0, "pred_only": 0}

_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step


def frenet(sim):
    st = sim.state
    tr, tid = sim.track, sim.tid
    s, d, idx = tr.project(st[:, :2], tid, prev_idx=sim.cl_idx)
    t = tr.cl_tangent[tid, idx]                                             # (B, 2) unit tangent
    c, sn = torch.cos(st[:, dyn.IYAW]), torch.sin(st[:, dyn.IYAW])
    vxw = st[:, dyn.IVX] * c - st[:, dyn.IVY] * sn
    vyw = st[:, dyn.IVX] * sn + st[:, dyn.IVY] * c
    sd = vxw * t[:, 0] + vyw * t[:, 1]                                      # ds/dt (curvature factor ignored)
    dd = -vxw * t[:, 1] + vyw * t[:, 0]                                     # dd/dt, left +
    return s, d, sd, dd, tr.length[tid]


def v_safe(env):
    sim = env.sim
    o = sim.other_idx[:, 0]
    s, d, sd, dd, L = frenet(sim)
    ds = s[o] - s
    ds = (ds + L / 2) % L - L / 2                                           # signed arc distance to the opponent
    gap = ds - CAR_LEN
    dlat, dlat_rate = d[o] - d, dd[o] - dd
    v_me, v_opp = sd, sd[o]
    room = (gap - GMIN - (v_me - v_opp - DV).clamp_min(0.0) * TAU).clamp_min(0.0)
    vs = v_opp.clamp_min(0.0) + DV + torch.sqrt(2.0 * A_B * room)
    in_lane = dlat.abs() < W
    # first time the extrapolated lateral offset comes under W (0 if already), inf if not within H
    closing = (dlat * dlat_rate) < 0
    t_in = torch.where(in_lane, torch.zeros_like(dlat),
                       torch.where(closing, (dlat.abs() - W) / dlat_rate.abs().clamp_min(1e-6), torch.full_like(dlat, 1e9)))
    enters = t_in <= H
    past = (ds + (sd[o] - sd) * t_in) < -(CAR_LEN + 0.1)                   # we would be a car length past it by then
    ahead = ((ds > 0.0) if ALONGSIDE else (gap > 0.1)) & (ds < 6.0)
    hit = ahead & enters & ~past
    return torch.where(hit, vs, torch.full_like(vs, 1e9)), hit & ~in_lane


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    out = _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)
    env = CUR["env"]
    if env is not None and env.tracker is self and env.sim.M > 1:
        vs, pred_only = v_safe(env)
        lr = env.learner
        cmd = out[:, 1]
        cut = lr & (cmd > vs)
        new = cmd if MEASURE else torch.where(cut, vs.to(cmd.dtype), cmd)
        if "t_steps" not in STAT:
            for k in ("t_steps", "t_active", "t_cut", "t_pred"):
                STAT[k] = torch.zeros((), device=cmd.device)
        STAT["t_steps"] += lr.sum(); STAT["t_active"] += cut.sum(); STAT["t_pred"] += (cut & pred_only).sum()
        STAT["t_cut"] += torch.where(cut, cmd - vs.to(cmd.dtype), torch.zeros_like(cmd)).sum()
        out = torch.stack([out[:, 0], new], 1)
    return out


def env_step(self, action):
    CUR["env"] = self
    return _orig_env_step(self, action)


mc.MAPPlanTracker.__call__ = map_call
gym_env.F1VecEnv.step = env_step
for seed in SEEDS:
    for kind in KINDS:
        out = os.path.join(OUT, f"race_s{seed}_{kind}.json")
        if os.path.exists(out):
            continue
        slot = {"kind": kind, "speed_scale": [0.7, 1.0]}
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off"),
                    "--opp-slots", json.dumps([slot]),
                    "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        if "t_steps" in STAT:
            STAT["steps"] += int(STAT.pop("t_steps")); STAT["active"] += int(STAT.pop("t_active"))
            STAT["cut_sum"] += float(STAT.pop("t_cut")); STAT["pred_only"] += int(STAT.pop("t_pred"))
        print(f"seed {seed} {kind}: shield active {STAT['active']}/{STAT['steps']} learner steps "
              f"({100 * STAT['active'] / max(STAT['steps'], 1):.2f} %, of them by prediction only {STAT['pred_only']}), "
              f"mean cut {STAT['cut_sum'] / max(STAT['active'], 1):.2f} m/s", flush=True)
json.dump({"h": H, "a_b": A_B, "w": W, "tau": TAU, "gmin": GMIN, "dv": DV, "alongside": ALONGSIDE, "measure_only": MEASURE, **STAT},
          open(os.path.join(OUT, f"shield_stats_{'_'.join(map(str, SEEDS))}.json"), "w"))
