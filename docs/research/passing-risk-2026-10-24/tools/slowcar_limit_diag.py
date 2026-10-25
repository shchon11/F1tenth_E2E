"""Research (2026-10-25): why does the speed limit for a car ahead, read by the policy's nearest-car output, barely help against a
near-stopped car (x0.05: none 2.51, shipped head 2.19, slow-car-retrained head 1.94, TRUE state 0.94 hard /km) although the
retrained head reads such cars far better offline (recall 63 -> 93 %, closing bias -1.20 -> -0.25 m/s)?

Runs slowcar_race.py's scene with the shipped limit ON (OSL) and, every step, computes the limit two ways with the shipped
function (`opponent_speed_limit.speed_limit`): from the head's reading the env actually used (env.opponent_reading) and
from the TRUE relative state written in the same format (LiDAR frame; closing along the line of sight; v_lat). Each learner
row keeps its last 40 steps (1 s); at every HARD car-contact onset (>= 1.5 m/s) that history is saved. Reports, over the
second before the hits: how often the head saw the car, whether its limit applied, how far it sat above the true limit,
and which of the limit's conditions failed (not seen / not in lane / not wholly ahead / closing read low).
usage: slowcar_limit_diag.py CKPT OUT_DIR SEEDS [SPEED_SCALE=0.05]"""
import json
import os
import sys

import numpy as np
import torch

from f1sim import dynamics as dyn
from f1sim import gym_env
from f1sim.learn import evaluate as ev
from f1sim.opponent_speed_limit import speed_limit, relative
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
TRACK = "real:korea_2026_competition"
K = 40
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None, "in_step": False}
REC = {}
EVENTS = []
_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step


def true_reading(env):
    st = env.sim.state
    # the nominal mount, as the shipped path (`F1VecEnv.step`) reads it: relative() then recovers the true centre-frame state
    mx, my = float(env.cfg.lidar.mount_x), float(env.cfg.lidar.mount_y)
    dx, dy, _dyaw, dvx, dvy, _dist, present = env._opp_rel(st)
    dx, dy, dvx, dvy, present = dx[:, 0], dy[:, 0], dvx[:, 0], dvy[:, 0], present[:, 0]
    lx, ly = dx - mx, dy - my
    closing = -(lx * dvx + ly * dvy) / torch.hypot(lx, ly).clamp_min(1e-6)
    ex = torch.where(present, torch.full_like(dx, 10.0), torch.full_like(dx, -10.0))
    return torch.stack([ex, lx / 5.0, ly / 2.0, closing / 5.0, dvy / 2.0], 1), mx, my


def env_step(self, action):
    CUR.update(env=self, in_step=True)
    try:
        out = _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False
    if self.opponent_reading is not None and self.sim.M > 1:
        tr, mx, my = true_reading(self)
        v = self.sim.state[:, dyn.IVX]
        rh = self.opponent_reading.float()
        lh = speed_limit(rh, v, float(mx), float(my)); lt = speed_limit(tr, v, float(mx), float(my))
        exh, dxh, dyh, clh, _ = relative(rh, float(mx), float(my))
        ext, dxt, dyt, clt, _ = relative(tr, float(mx), float(my))
        row = torch.stack([v, self.last_cmd_raw[:, 1].float(), lh.clamp(max=99.0), lt.clamp(max=99.0), rh[:, 0], dxh, dyh, clh,
                           dxt, dyt, clt], 1)
        rec = REC.get(id(self))
        if rec is None:
            rec = REC[id(self)] = {"buf": torch.zeros(K, self.B, row.shape[1], device=row.device), "n": 0}
        rec["buf"] = torch.roll(rec["buf"], 1, 0); rec["buf"][0] = row; rec["n"] += 1
    return out


PREV = {}


def sim_step(self, *a, **kw):
    entry = self.steps.clone()
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self and self.M > 1 and r.car_collision is not None \
            and r.car_impact is not None:
        prev = PREV.get(id(self))
        fresh = entry == 0
        onset = r.car_collision & ~((prev if prev is not None else torch.zeros_like(r.car_collision)) & ~fresh)
        PREV[id(self)] = r.car_collision.clone()
        hit = onset & env.learner & (r.car_impact >= 1.5)
        rec = REC.get(id(env))
        if bool(hit.any()) and rec is not None and rec["n"] >= K:
            for i in hit.nonzero().flatten().tolist():
                EVENTS.append(dict(LABEL, impact=float(r.car_impact[i]), hist=rec["buf"][:, i].cpu().tolist()))
    return r


gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
LABEL = {}
for seed in SEEDS:
    out = os.path.join(OUT, f"race_s{seed}_slowcar.json")
    if os.path.exists(out):
        continue
    LABEL.clear(); LABEL.update(seed=seed); REC.clear(); PREV.clear()
    sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                "--procedural-max-props", "10", "--procedural-raceline-corridor", "on",
                "--opp-slots", json.dumps([{"kind": "raceline", "speed_scale": SCALE}]), "--output", out,
                "--dial-value", "1.0489", "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92",
                "--opponent-speed-limit"]
    ev.main()
    json.dump(EVENTS, open(os.path.join(OUT, "events.json"), "w"))
    print(f"seed {seed}: {len(EVENTS)} hard-contact histories so far", flush=True)
# summary over the 1 s before each hit (newest first in hist: index 0 = the step of the hit)
H = np.array([e["hist"] for e in EVENTS])            # (E, K, 11): v, v_cmd, lim_head, lim_true, logit, dxh, dyh, clh, dxt, dyt, clt
if len(H):
    for a, b, tag in ((1, 8, "last 0.2 s"), (8, 20, "0.2-0.5 s"), (20, 40, "0.5-1.0 s")):
        h = H[:, a:b]
        v, vc, lh, lt, lg = h[..., 0], h[..., 1], h[..., 2], h[..., 3], h[..., 4]
        need = lt < v                                   # the true limit asks to slow down here
        print(f"{tag}: steps where the TRUE limit is below our speed {100 * need.mean():.0f} %; there the head: saw a car "
              f"{100 * (lg[need] > 0).mean():.0f} %, its limit applied {100 * (lh[need] < 98).mean():.0f} %, below our speed "
              f"{100 * (lh[need] < v[need]).mean():.0f} %, head limit - true limit median {np.median((np.minimum(lh, 20) - lt)[need]):+.2f} m/s | "
              f"speed {np.median(v):.2f}, command {np.median(vc):.2f}")
        seen = need & (lg > 0)
        if seen.any():
            dyh, dyt, dxh, dxt, clh, clt = h[..., 6], h[..., 9], h[..., 5], h[..., 8], h[..., 7], h[..., 10]
            print(f"   where the head saw it: |dy| head {np.median(np.abs(dyh[seen])):.2f} vs true {np.median(np.abs(dyt[seen])):.2f} m "
                  f"(head says outside the 0.35 m lane {100 * (np.abs(dyh[seen]) >= 0.35).mean():.0f} %, truth {100 * (np.abs(dyt[seen]) >= 0.35).mean():.0f} %), "
                  f"dx head {np.median(dxh[seen]):.2f} vs true {np.median(dxt[seen]):.2f}, closing head {np.median(clh[seen]):+.2f} vs true {np.median(clt[seen]):+.2f}")
