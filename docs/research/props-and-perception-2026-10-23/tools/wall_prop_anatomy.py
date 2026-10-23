"""Research (2026-10-23): the anatomy of our HARD wall / prop contacts in races (impact >= 1.5 m/s, the tables' "hard").
The v2 entry's hard-hit surplus over v1 is walls + props (finals: 1.11 vs 0.88 /km), not cars (1.22 vs 1.47).

Each hard wall or prop onset of a learner row is recorded at the simulator step it happens (before any reset), with:
  where on the car it touched (the step's contact point in the car frame: front / side / rear);
  how the car was moving: speed, slip angle, yaw rate, the curvature it ASKED for (plan kappa_1) against the one it
  DROVE (yaw rate / speed) -- understeer shows as asked >> driven;
  the last 1 s: our commanded first speed v1 against our speed (accelerating / braking), the nearest opponent's distance
  (was a car within 1.5 m: an evasion or a pass), the centreline's curvature where it happened (corner / straight).
Protocol = the finals' race cells (as car_contact_anatomy.py).
usage: wall_prop_anatomy.py CKPT OUT_DIR [SEEDS=81,82] [KINDS=forzaeth,forzaeth_pred,lane_switch,interactive]"""
import json
import math
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, mpc
from f1sim.learn import evaluate as ev
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "81,82").split(",")]
KINDS = (sys.argv[4] if len(sys.argv) > 4 else "forzaeth,forzaeth_pred,lane_switch,interactive").split(",")
TRACK = os.environ.get("ANAT_TRACK", "real:korea_2026_competition")
HARD = 1.5
K = 40
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None, "action": None, "in_step": False}
EVENTS, LABEL, RECS = [], {}, {}


def cl_curvature(track, tid, idx, half=5):
    """Signed centreline curvature [1/m] around index idx (heading change over +-half points / arc)."""
    cl = track.cl[tid]                                       # (N, 2)
    N = cl.shape[0]
    a, b, c = cl[(idx - half) % N], cl[idx % N], cl[(idx + half) % N]
    h1 = math.atan2(float(b[1] - a[1]), float(b[0] - a[0])); h2 = math.atan2(float(c[1] - b[1]), float(c[0] - b[0]))
    dh = math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))
    ds = float(torch.hypot(*(c - a)))
    return dh / max(ds, 1e-6)


class Recorder:
    def __init__(self, env):
        sim = env.sim
        B, A = int(sim.state.shape[0]), int(env.act_dim)
        dev = sim.device
        self.st = torch.zeros(K, B, 7, device=dev)           # x y yaw vx vy r steer
        self.act = torch.zeros(K, B, A, device=dev)
        self.vm = torch.zeros(K, B, device=dev)
        self.age = torch.zeros(B, dtype=torch.long, device=dev)

    def observe(self, env, sim, r, entry, action, v_plan):
        fresh = entry == 0
        st = r.state[:, [dyn.IX, dyn.IY, dyn.IYAW, dyn.IVX, dyn.IVY, dyn.IR, dyn.ISTEER]].float()
        self.age = torch.where(fresh, torch.zeros_like(self.age), self.age)
        self.st = torch.roll(self.st, 1, 0); self.st[0] = st
        self.act = torch.roll(self.act, 1, 0); self.act[0] = action.float()
        self.vm = torch.roll(self.vm, 1, 0); self.vm[0] = v_plan.float()
        self.age += 1
        on, imp = r.contact_onsets, r.contact_impacts
        if on is None or imp is None:
            return
        hit = (on[:, 0] & (imp[:, 0] >= HARD)) | (on[:, 1] & (imp[:, 1] >= HARD))
        hit = hit & env.learner
        if not bool(hit.any()):
            return
        spec, vmax = env.tracker.spec, float(env.tracker.v_max)
        _, _, idx = sim.track.project(r.state[:, :2], sim.tid, prev_idx=sim.cl_idx)
        for i in hit.nonzero().flatten().tolist():
            n = int(min(K, int(self.age[i])))
            e = self.st[:n, i].cpu()
            kind = "prop" if bool(on[i, 1] & (imp[i, 1] >= HARD)) and not bool(on[i, 0] & (imp[i, 0] >= HARD)) else "wall"
            impact = float(imp[i, 1] if kind == "prop" else imp[i, 0])
            x, y, yaw, vx, vy, w, dl = (float(q) for q in e[0])
            c, s_ = math.cos(yaw), math.sin(yaw)
            px, py = float(sim.contact_pt[i, 0]) - x, float(sim.contact_pt[i, 1]) - y
            lx, ly = c * px + s_ * py, -s_ * px + c * py
            where = "front" if lx > 0.15 else ("rear" if lx < -0.15 else "side")
            v = math.hypot(vx, vy)
            slip = math.degrees(math.atan2(vy, max(abs(vx), 0.3)))
            a_hist = self.act[:n, i]
            cap = torch.full((n,), 9.0, device=a_hist.device)
            if spec.speed_mode == "wp4":
                k, _, vk = mpc.decode_wp4(a_hist, self.vm[:n, i], vmax, cap, spec)
                v1 = vk[:, 0].cpu(); k1 = k[:, 0].cpu()
            else:
                k, _, v0, _ = mpc.decode(a_hist, self.vm[:n, i], vmax, cap, spec)
                v1 = v0.cpu(); k1 = k[:, 0].cpu()
            v_hist = [math.hypot(float(e[t, 3]), float(e[t, 4])) for t in range(n)]
            h = min(n - 1, 20)
            near = None
            if sim.M > 1:
                j = int(sim.other_idx[i, 0])
                d = [float(torch.hypot(*(self.st[t, j, :2] - self.st[t, i, :2]).cpu())) for t in range(n)]
                near = round(min(d), 2)
            k_drv = w / max(v, 0.5)
            EVENTS.append(dict(LABEL, row=i, kind=kind, impact=round(impact, 3), where=where, v=round(v, 2),
                               slip_deg=round(slip, 1), yaw_rate=round(w, 2), steer_deg=round(math.degrees(dl), 1),
                               k_asked=round(float(k1[0]), 3), k_driven=round(k_drv, 3),
                               k_asked_05=round(float(k1[h]), 3), cl_kappa=round(cl_curvature(sim.track, int(sim.tid[i]), int(idx[i])), 3),
                               plan_v1_now=round(float(v1[0]), 2), plan_v1_05=round(float(v1[h]), 2),
                               v_05=round(v_hist[h], 2), v_hist=[round(q, 2) for q in v_hist],
                               near_car_min_1s=near, hist_steps=n))


_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step


def env_step(self, action):
    CUR.update(env=self, action=action, in_step=True, v_plan=self._tracker_plan_speed().clone())
    try:
        return _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False


def sim_step(self, *a, **kw):
    entry = self.steps.clone()
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self:
        rec = RECS.get(id(env))
        if rec is None:
            rec = RECS[id(env)] = Recorder(env)
        rec.observe(env, self, r, entry, CUR["action"], CUR["v_plan"])
    return r


gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
for seed in SEEDS:
    for kind in KINDS:
        LABEL.clear(); LABEL.update(seed=seed, opp=kind)
        RECS.clear()
        out = os.path.join(OUT, f"race_s{seed}_{kind}.json")
        slot = {"kind": kind, "speed_scale": [0.7, 1.0]}
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", "off", "--opp-slots", json.dumps([slot]),
                    "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        n0 = len(EVENTS)
        ev.main()
        d = json.load(open(out))["nominal"]
        cs = d.get("contact_severity") or {}
        km = float(cs.get("learner_distance_m", 0.0)) / 1000.0
        hw = int(sum(((cs.get("wall_prop") or {}).get("fine_hist") or [])[15:]))
        print(f"seed {seed} {kind}: {len(EVENTS) - n0} wall/prop events recorded, the meter's hard wall_prop onsets {hw}, "
              f"{km:.2f} learner km", flush=True)
        json.dump(EVENTS, open(os.path.join(OUT, "events.json"), "w"))

from collections import Counter
print("by kind/where:", dict(Counter((e["kind"], e["where"]) for e in EVENTS)), flush=True)
