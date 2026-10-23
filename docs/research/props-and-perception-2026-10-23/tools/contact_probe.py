"""Research (2026-10-23, R4): does our policy KNOW a hard car contact is coming, or can it not see it?

Lowering the charged contact threshold 1.5 -> 0.75 m/s cut hard wall/prop hits 30 % in 256 updates and moved car contacts
not at all (car_contact_anatomy + v2PHs_s2424): static obstacles respond to the cost, cars do not. Two readings:
  (a) the policy's state holds what predicts the contact (gap, closing speed, the opponent's sideways motion) and the
      policy acts on it badly -- an incentive / credit-assignment problem;
  (b) its state does not hold it -- a perception / prediction problem, which no cost can fix.
This reads the actor's own state at every step of the finals race protocol (races of 2, 4 kinds, rep floor):
  1. its nearest-car head (trained as an auxiliary on [exists, dx/5, dy/2, closing/5, v_lat/2]) against the truth,
     in close situations (centre distance < 3 m), and in the last 0.5 s before hard contacts;
  2. linear probes for "a hard (>= 1.5 m/s) car-contact onset within the next 0.5 s", fitted on seed-81 cells and
     scored on seed-82 cells (ROC AUC): on the actor's features [e | y] (384), on the TRUE relative state
     (distance, dx, dy, closing, v_lat, own speed, opponent speed), and on both.
If the features probe scores near the truth probe, the information is in the state: reading (a). Far below: (b).
usage: contact_probe.py CKPT OUT_DIR [SEEDS=81,82] [KINDS=forzaeth,forzaeth_pred,lane_switch,interactive]"""
import json
import os
import sys

import numpy as np
import torch

from f1sim import dynamics as dyn
from f1sim import gym_env
from f1sim.learn import evaluate as ev
from f1sim.learn import model_v2
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "81,82").split(",")]
KINDS = (sys.argv[4] if len(sys.argv) > 4 else "forzaeth,forzaeth_pred,lane_switch,interactive").split(",")
TRACK = os.environ.get("ANAT_TRACK", "real:korea_2026_competition")
HARD, CLOSE, AHEAD = 1.5, 3.0, 20                # [m/s], [m] centre distance, label window [steps] (0.5 s)
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None, "in_step": False, "z": None, "near": None}

_orig_features = model_v2.ActorV2.features


def features(self, e, y):
    """The policy is asked BEFORE env.step: keep its latest [e | y] (and the nearest-car head's reading of it)."""
    z = _orig_features(self, e, y)
    if not CUR["in_step"]:
        CUR["z"] = z.detach()
        CUR["near"] = self.near_car(z).detach() if self.near_car is not None else None
    return z


model_v2.ActorV2.features = features

# v1 (`learn.model.Actor`): the trunk features the action head reads (`_mu`); no nearest-car head
from f1sim.learn import model as model_v1
_orig_mu = model_v1.Actor._mu


def mu_v1(self, feat):
    if not CUR["in_step"]:
        CUR["z"] = feat.detach()
        CUR["near"] = None
    return _orig_mu(self, feat)


model_v1.Actor._mu = mu_v1
_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step
REC = {"rows": []}


def env_step(self, action):
    # the observation the action was chosen from was built from the state before this step
    st = self.sim.state.clone()
    z, near = CUR["z"], CUR["near"]
    CUR.update(env=self, in_step=True, onset=None)
    try:
        out = _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False
    lr = self.learner
    B = lr.shape[0]
    nl = int(lr.sum())
    if z is not None and self.M > 1:
        if z.shape[0] == B:
            zl, nl_ = z[lr], near[lr] if near is not None else None
        elif z.shape[0] == nl:
            zl, nl_ = z, near
        else:
            zl = nl_ = None
        if zl is not None:
            dx, dy, _dyaw, dvx, dvy, dist, present = self._opp_rel(st)
            mx, my, _ = self._lidar_pose()
            lx, ly = dx[:, 0] - mx, dy[:, 0] - my
            dl = torch.hypot(lx, ly).clamp_min(1e-6)
            closing = -(lx * dvx[:, 0] + ly * dvy[:, 0]) / dl
            v_me = torch.hypot(st[:, dyn.IVX], st[:, dyn.IVY])
            o = self.sim.other_idx[:, 0]
            v_ot = v_me[o]
            gt = torch.stack([dist[:, 0], lx, ly, closing, dvy[:, 0], v_me, v_ot], 1)[lr]
            REC["rows"].append({"z": zl.half().cpu(), "near": None if nl_ is None else nl_.float().cpu(),
                                "gt": gt.float().cpu(), "row": lr.nonzero().flatten().cpu(), "t": REC.get("t", 0)})
    r = self.last_result
    REC.setdefault("onsets", []).append(CUR["onset"].cpu() if CUR["onset"] is not None else torch.zeros(B, dtype=torch.bool))
    REC.setdefault("fresh", []).append((self.sim.steps <= 1).cpu())
    REC["t"] = REC.get("t", 0) + 1
    return out


def sim_step(self, *a, **kw):
    entry = self.steps.clone()
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self and r.car_collision is not None and r.car_impact is not None:
        prev = CUR.get("prev_car")
        if prev is None or prev.shape != r.car_collision.shape:
            prev = torch.zeros_like(r.car_collision)
        fresh = entry == 0
        onset = r.car_collision & ~(prev & ~fresh) & (r.car_impact >= HARD)
        CUR["prev_car"] = r.car_collision.clone()
        CUR["onset"] = onset
    return r


gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
data = {}
for seed in SEEDS:
    for kind in KINDS:
        REC.clear(); REC["rows"] = []
        CUR.update(prev_car=None, z=None, near=None)
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
        ev.main()
        ons = torch.stack(REC["onsets"])                      # (T, B) hard onsets per env step
        fr = torch.stack(REC["fresh"])
        T = ons.shape[0]
        ep = torch.cumsum(fr.long(), 0)
        Z, NEAR, GT, Y = [], [], [], []
        for rec in REC["rows"]:
            t, rows = rec["t"], rec["row"]
            if t >= T:
                continue
            w = ons[t:t + AHEAD, rows]                           # this step and the next 0.5 s
            same = ep[t:t + AHEAD, rows] == ep[t, rows][None]
            y = (w & same).any(0)
            close = rec["gt"][:, 0] < CLOSE
            if not bool(close.any()):
                continue
            Z.append(rec["z"][close]); GT.append(rec["gt"][close]); Y.append(y[close])
            if rec["near"] is not None:
                NEAR.append(rec["near"][close])
        data[(seed, kind)] = {"z": torch.cat(Z), "gt": torch.cat(GT), "y": torch.cat(Y),
                              "near": torch.cat(NEAR) if NEAR else None}
        print(f"seed {seed} {kind}: {int(data[(seed, kind)]['y'].shape[0])} close steps, "
              f"{int(data[(seed, kind)]['y'].sum())} within 0.5 s of a hard car contact", flush=True)
        torch.save(data, os.path.join(OUT, "probe_data.pt"))

print('collected; analyse with contact_probe_analyze.py', flush=True)
