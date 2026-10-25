"""Research (2026-10-25): a car that has (almost) stopped on the racing line -- how does the entry policy deal with it, and
what does the shipped speed limit for a car ahead do there?

Why: on the real venue bags the policy's nearest-car output calls the fixed objects on the track (~0.3 m, some in the
lane) a car, closing at our own speed -- a stopped car. With the speed limit on, the real car will brake for them as for a
stopped car. A raceline car at speed_scale 0.05 (~0.3 m/s) is the simulator's nearest stand-in: race of 2, props off the
line (corridor on), finals' floor. Reports race speed, hard contacts, passes and -- like prop_plan_clearance.py -- how
close the plan passes the slow car's centre when it is 0.5-3.5 m ahead within 0.6 m of our heading (clearance = distance
- 0.30, only when the path gets that far). OSL=1 turns the shipped speed limit on.
usage: slowcar_race.py CKPT OUT_DIR SEEDS [SPEED_SCALE=0.05]"""
import json
import os
import sys

import numpy as np
import torch

from f1sim import gym_env, map_controller as mc, mpc
from f1sim.learn import evaluate as ev

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
TRACK = "real:korea_2026_competition"
DS, N_S = 0.1, 40
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None}
ROWS = []
_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step


def measure(env, action, v_meas):
    if env.sim.M < 2:
        return
    spec, vmax = env.tracker.spec, float(env.tracker.v_max)
    B = action.shape[0]
    k, L, _ = mpc.decode_wp4(action.float(), v_meas.float(), vmax, torch.full((B,), 9.0, device=action.device), spec)
    s = torch.arange(1, N_S + 1, device=k.device, dtype=k.dtype) * DS
    seg = (s[None, :] / (L[:, None] / k.shape[1]).clamp_min(1e-3)).floor().long().clamp(max=k.shape[1] - 1)
    th = torch.cumsum(k.gather(1, seg) * DS, 1)
    px, py = torch.cumsum(torch.cos(th) * DS, 1), torch.cumsum(torch.sin(th) * DS, 1)
    on = s[None, :] <= torch.minimum(L, torch.full_like(L, 4.0))[:, None]
    dx, dy, _a, _b, _c, _d, pr = env._opp_rel(env.sim.state)
    xl, yl, pr = dx[:, 0], dy[:, 0], pr[:, 0]
    sel = env.learner & pr & (xl > 0.5) & (xl < 3.5) & (yl.abs() < 0.6)
    if not bool(sel.any()):
        return
    d = torch.hypot(px - xl[:, None], py - yl[:, None])
    rr = torch.hypot(px, py)
    near = on & (rr <= torch.hypot(xl, yl)[:, None] + 0.5)
    dmin = torch.where(near, d, torch.full_like(d, float("inf"))).min(1).values
    reach = torch.where(on, rr, torch.zeros_like(rr)).max(1).values >= torch.hypot(xl, yl) - 0.15
    clear = torch.where(reach & torch.isfinite(dmin), dmin - 0.30, torch.full_like(dmin, float("nan")))
    b = sel.nonzero().flatten()
    ROWS.append(torch.stack([xl[b], yl[b], env.sim.state[b, 3], clear[b], L[b]], 1).float().cpu())


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    env = CUR["env"]
    if env is not None and env.tracker is self:
        measure(env, action, v_meas)
    return _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)


def env_step(self, action):
    CUR["env"] = self
    return _orig_env_step(self, action)


mc.MAPPlanTracker.__call__ = map_call
gym_env.F1VecEnv.step = env_step
for seed in SEEDS:
    out = os.path.join(OUT, f"race_s{seed}_slowcar.json")
    if os.path.exists(out):
        continue
    ROWS.clear()
    sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                "--procedural-max-props", "10", "--procedural-raceline-corridor", "on",
                "--opp-slots", json.dumps([{"kind": "raceline", "speed_scale": SCALE}]), "--output", out,
                "--dial-value", "1.0489", "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92",
                *(["--opponent-speed-limit"] if os.environ.get("OSL") else [])]
    ev.main()
    r = torch.cat(ROWS).numpy() if ROWS else np.zeros((0, 5), np.float32)
    np.save(os.path.join(OUT, f"rows_s{seed}.npy"), r)
    for lo, hi in ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi); c = r[m, 3]; dec = np.isfinite(c)
        if dec.sum():
            print(f"seed {seed} slow car {lo}-{hi} m ahead: steps {int(m.sum())}, plan reaches it {int(dec.sum())}: clears (> 0.05) "
                  f"{100 * np.mean(c[dec] > 0.05):.0f} %, into (< 0) {100 * np.mean(c[dec] < 0):.0f} %, clearance p50 {np.median(c[dec]):+.2f} m, "
                  f"speed p50 {np.median(r[m, 2]):.2f}", flush=True)
