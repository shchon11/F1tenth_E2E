"""Research (2026-10-24): what would a reaction to a slower car ahead buy? An ORACLE speed shield as an upper bound.

The policy knows a hard car contact is coming (state probe AUC 0.90) but keeps accelerating into slower cars until
0.3 s before (car_contact_anatomy); cost and traffic changes in fine-tuning never taught it to brake. Before teaching a
reaction, measure what the reaction is worth: after the follower, the speed command of a learner is capped so it can
still match the speed of a car in its path before the gap closes --
    v_safe = v_opp + sqrt(2 a_b max(0, gap - g_min - max(0, v_me - v_opp) tau))
for the nearest opponent ahead (x > 0 in our frame) and in our path (|y| < W); gap = body gap along our heading,
v_opp its speed along our heading, tau the drive lag. True states (oracle). Nothing else changes: the policy plans as
always and the cap only bites when the car would otherwise run into the car ahead.
Finals race cells (seeds, 4 kinds, props on the line), the Korea entry; reports race speed, passes, hard contacts and
how often / how much the shield cut the command.
usage: shield_race.py CKPT OUT_DIR SEEDS [A_B=3.0] [W=0.45] [TAU=0.2] [GMIN=0.2]"""
import json
import math
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc
from f1sim.learn import evaluate as ev

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
A_B = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
W = float(sys.argv[5]) if len(sys.argv) > 5 else 0.45
TAU = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
GMIN = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
TRACK = "real:korea_2026_competition"
CAR_LEN = 0.5
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None}
STAT = {"steps": 0, "active": 0, "cut_sum": 0.0}

_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step


def v_safe(env):
    st = env.sim.state
    o = env.sim.other_idx[:, 0]
    c, s = torch.cos(st[:, dyn.IYAW]), torch.sin(st[:, dyn.IYAW])
    dxw, dyw = st[o, 0] - st[:, 0], st[o, 1] - st[:, 1]
    dx, dy = c * dxw + s * dyw, -s * dxw + c * dyw
    vme = st[:, dyn.IVX]                                                    # our forward speed
    co, so = torch.cos(st[o, dyn.IYAW]), torch.sin(st[o, dyn.IYAW])
    vox = st[o, dyn.IVX] * co - st[o, dyn.IVY] * so; voy = st[o, dyn.IVX] * so + st[o, dyn.IVY] * co
    v_opp = vox * c + voy * s                                               # its speed along our heading
    gap = (dx - CAR_LEN).clamp_min(0.0)
    room = (gap - GMIN - (vme - v_opp).clamp_min(0.0) * TAU).clamp_min(0.0)
    vs = v_opp.clamp_min(0.0) + torch.sqrt(2.0 * A_B * room)
    # clearly AHEAD: no longitudinal overlap (gap > 0.1 m) -- the first version also capped a car ALONGSIDE (dx < 0.5,
    # gap 0 -> v_safe = v_opp) and kept it stuck beside the opponent mid-pass: car contacts 9.0 -> 15.6 /km (10-24)
    ahead = (gap > 0.1) & (dy.abs() < W) & (dx < 6.0)
    return torch.where(ahead, vs, torch.full_like(vs, 1e9))


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    out = _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)
    env = CUR["env"]
    if env is not None and env.tracker is self and env.sim.M > 1:
        vs = v_safe(env)
        lr = env.learner
        cmd = out[:, 1]
        cut = lr & (cmd > vs)
        # SHIELD_MEASURE_ONLY=1: count where a policy would have been capped, drive it uncapped (does a trained
        # policy still run into the car ahead by itself?)
        new = cmd if os.environ.get("SHIELD_MEASURE_ONLY") else torch.where(cut, vs.to(cmd.dtype), cmd)
        # on the device: no host sync per step (the counters are read once per cell)
        if "t_steps" not in STAT:
            STAT["t_steps"] = torch.zeros((), device=cmd.device); STAT["t_active"] = torch.zeros((), device=cmd.device)
            STAT["t_cut"] = torch.zeros((), device=cmd.device)
        STAT["t_steps"] += lr.sum(); STAT["t_active"] += cut.sum(); STAT["t_cut"] += torch.where(cut, cmd - vs.to(cmd.dtype), torch.zeros_like(cmd)).sum()
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
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", "off", "--opp-slots", json.dumps([slot]),
                    "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        if "t_steps" in STAT:
            STAT["steps"] += int(STAT.pop("t_steps")); STAT["active"] += int(STAT.pop("t_active"))
            STAT["cut_sum"] += float(STAT.pop("t_cut"))
        print(f"seed {seed} {kind}: shield active {STAT['active']}/{STAT['steps']} learner steps "
              f"({100 * STAT['active'] / max(STAT['steps'], 1):.1f} %), mean cut "
              f"{STAT['cut_sum'] / max(STAT['active'], 1):.2f} m/s", flush=True)
json.dump({"a_b": A_B, "w": W, "tau": TAU, "gmin": GMIN, **STAT}, open(os.path.join(OUT, f"shield_stats_{'_'.join(map(str, SEEDS))}.json"), "w"))
