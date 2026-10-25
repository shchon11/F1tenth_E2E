"""Research (2026-10-24): data to retrain the policy's nearest-car head on its own frozen features.

The deployable shield reads the v2 actor's NearestCarHead ([exists logit, dx/5, dy/2, closing/5, v_lat/2], LiDAR frame).
From it the predictive shield cut hard car contacts 1.00 -> 0.70 /km (props off), from the true state 1.00 -> 0.48: the
head's reading is the gap. It was trained as an auxiliary with weights 10 (position) / 1 (closing) / 0.1 (v_lat), and in
the last 0.5 s before hard contacts it reads closing at R2 0.44 and v_lat at 0.31. The head does not feed the action,
so retraining it on the frozen features changes what the car perceives, not how it drives.
This records, at every 2nd learner step of the finals race cells, the actor's features z = [e | y] and the env's own
target for the head (`nearest_visible_car_v2`, with `opp_visibility` switched on), plus the true opponent state and
whether a hard car contact starts within 0.5 s (for weighting / evaluation).
usage: head_data.py CKPT OUT.pt SEEDS [KINDS=forzaeth,forzaeth_pred,lane_switch,interactive]"""
import dataclasses
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env
from f1sim.learn import evaluate as ev
from f1sim.learn import model_v2
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
KINDS = (sys.argv[4] if len(sys.argv) > 4 else "forzaeth,forzaeth_pred,lane_switch,interactive").split(",")
TRACK = "real:korea_2026_competition"
HARD, AHEAD, EVERY = 1.5, 20, 2
CUR = {"z": None, "t": 0, "in_step": False, "prev_car": None}
REC = {"z": [], "tgt": [], "gt": [], "t": [], "row": [], "onset": [], "fresh": []}

_orig_init = gym_env.F1VecEnv.__init__
_orig_features = model_v2.ActorV2.features
_orig_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step
_orig_reset = gym_env.F1VecEnv.reset


def init(self, track, cfg=None, env_cfg=None, *a, **kw):
    env_cfg = dataclasses.replace(env_cfg if env_cfg is not None else gym_env.EnvConfig(), opp_visibility=True)
    return _orig_init(self, track, cfg, env_cfg, *a, **kw)


def features(self, e, y):
    z = _orig_features(self, e, y)
    CUR["z"] = z.detach()
    return z


def env_step(self, action):
    t = CUR["t"]
    if self.M > 1 and CUR["z"] is not None and t % EVERY == 0:
        lr = self.learner
        z = CUR["z"]
        zl = z[lr] if z.shape[0] == lr.shape[0] else z
        tgt = self.nearest_visible_car_v2()[lr]
        st = self.sim.state
        dx, dy, _dyaw, dvx, dvy, dist, present = self._opp_rel(st)
        mx, my, _ = self._lidar_pose()
        lx, ly = dx[:, 0] - mx, dy[:, 0] - my
        closing = -(lx * dvx[:, 0] + ly * dvy[:, 0]) / torch.hypot(lx, ly).clamp_min(1e-6)
        gt = torch.stack([dist[:, 0], lx, ly, closing, dvy[:, 0], st[:, dyn.IVX]], 1)[lr]
        REC["z"].append(zl.half().cpu()); REC["tgt"].append(tgt.float().cpu()); REC["gt"].append(gt.float().cpu())
        REC["t"].append(t); REC["row"].append(lr.nonzero().flatten().cpu())
    CUR["in_step"] = True
    try:
        out = _orig_step(self, action)
    finally:
        CUR["in_step"] = False
    on = CUR.get("onset_now")
    REC["onset"].append((on if on is not None else torch.zeros(self.B, dtype=torch.bool)).cpu())
    REC["fresh"].append((self.sim.steps <= 1).cpu())
    CUR["t"] = t + 1
    return out


def env_reset(self, *a, **kw):
    """the evaluation's own reset (after the graph warm-up steps): records start here"""
    for k in REC:
        REC[k] = []
    CUR.update(t=0, prev_car=None, onset_now=None)
    return _orig_reset(self, *a, **kw)


def sim_step(self, *a, **kw):
    entry = self.steps.clone()
    r = _orig_sim_step(self, *a, **kw)
    if CUR["in_step"] and r.car_collision is not None and r.car_impact is not None:
        prev = CUR.get("prev_car")
        if prev is None or prev.shape != r.car_collision.shape:
            prev = torch.zeros_like(r.car_collision)
        onset = r.car_collision & ~(prev & ~(entry == 0)) & (r.car_impact >= HARD)
        CUR["prev_car"] = r.car_collision.clone()
        CUR["onset_now"] = onset
    return r


gym_env.F1VecEnv.__init__ = init
model_v2.ActorV2.features = features
gym_env.F1VecEnv.step = env_step
gym_env.F1VecEnv.reset = env_reset
Simulator.step = sim_step
data = {}
for seed in SEEDS:
    for kind in KINDS:
        for k in REC:
            REC[k] = []
        CUR.update(z=None, t=0, prev_car=None, onset_now=None)
        out = os.path.join(os.path.dirname(OUT) or ".", f"_hd_{os.path.basename(OUT)}_{seed}_{kind}.json")
        # HD_SCALE (10-25): the opponent's speed_scale as one number (0.05 = a near-stopped car, the head's blind spot)
        slot = {"kind": kind, "speed_scale": (float(os.environ["HD_SCALE"]) if os.environ.get("HD_SCALE") else [0.7, 1.0])}
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off"),
                    "--opp-slots", json.dumps([slot]), "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        ons = torch.stack(REC["onset"]); fr = torch.stack(REC["fresh"])
        ep = torch.cumsum(fr.long(), 0); T = ons.shape[0]
        Y = []
        for t, rows in zip(REC["t"], REC["row"]):
            w = ons[t:t + AHEAD, rows]; same = ep[t:t + AHEAD, rows] == ep[t, rows][None]
            Y.append((w & same).any(0))
        data[(seed, kind)] = {"z": torch.cat(REC["z"]), "tgt": torch.cat(REC["tgt"]), "gt": torch.cat(REC["gt"]),
                              "y": torch.cat(Y)}
        d = data[(seed, kind)]
        print(f"seed {seed} {kind}: {d['z'].shape[0]} samples, seen {int((d['tgt'][:, 0] > 0).sum())}, "
              f"pre-contact {int(d['y'].sum())}", flush=True)
        torch.save(data, OUT)
print("done", flush=True)
