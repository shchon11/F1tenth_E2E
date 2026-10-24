"""Research (2026-10-25): where do the deployable shield's extra wall hits come from -- caps the true state would not
make (perception false alarms), or the braking itself?

The retrained-head shield (shield_head_race.py SOURCE=head) cut hard car contacts 35-42 % but hard wall hits rose
0.08 -> 0.26 /km (props off) and 0.11 -> 0.20 (props on); the same shield from the true state also showed 0.20 with
props on. Here the head shield drives (caps), and at every learner step the TRUE-state shield's verdict is computed
beside it: caps are BOTH (the truth would cap too), HEAD-ONLY (false alarm) or the truth's alone (MISSED). Each hard
(>= 1.5 m/s) wall or prop onset of a learner car is attributed to a head-only cap in the second before it, else to a
both-cap, else to none. Finals race cells (seeds, 4 kinds); RACE_CORRIDOR=on keeps the props off the line.
Head-only caps are split by what the truth says: no car wholly ahead within 6 m (a phantom), the car in our lane now but
the truth's ceiling above the command (a misjudged gap or closing speed), or the car outside our lane and the truth not
predicting it in within H (a misjudged lateral rate) -- the head's own reason being in lane now or predicted in.
usage: shield_dual_diag.py CKPT OUT.json SEEDS [H=1.0]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc
from f1sim.learn import evaluate as ev
from f1sim.learn import model_v2
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
H = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
A_B, W, TAU, GMIN, CAR_LEN, HARD, WIN = 3.0, 0.35, 0.2, 0.2, 0.5, 1.5, 40
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
TRACK = "real:korea_2026_competition"
CUR = {"env": None, "near": None, "t": 0, "in_step": False}
ROW = {}
CNT = {}

_orig_features = model_v2.ActorV2.features
_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step


def bump(k, v):
    CNT[k] = CNT.get(k, 0) + int(v)


def features(self, e, y):
    z = _orig_features(self, e, y)
    if self.near_car is not None:
        CUR["near"] = self.near_car(z).detach()
    return z


def relative(env, source):
    st = env.sim.state
    mx, my, _ = env._lidar_pose()
    if source == "true":
        dx, dy, _dyaw, dvx, dvy, dist, present = env._opp_rel(st)
        dx, dy, dvx, dvy, present = dx[:, 0], dy[:, 0], dvx[:, 0], dvy[:, 0], present[:, 0]
        lx, ly = dx - mx, dy - my
        return present, dx, dy, -(lx * dvx + ly * dvy) / torch.hypot(lx, ly).clamp_min(1e-6), dvy
    near = CUR["near"]
    B = st.shape[0]
    if near.shape[0] != B:
        full = torch.zeros(B, near.shape[1], device=near.device, dtype=near.dtype); full[env.learner] = near; near = full
    near = near.float()
    return near[:, 0] > 0, near[:, 1] * 5.0 + mx, near[:, 2] * 2.0 + my, near[:, 3] * 5.0, near[:, 4] * 2.0


def v_safe(env, source, parts=False):
    ex, dx, dy, closing, vlat = relative(env, source)
    gap = dx - CAR_LEN
    v_opp = (env.sim.state[:, dyn.IVX] - closing).clamp_min(0.0)
    room = (gap - GMIN - closing.clamp_min(0.0) * TAU).clamp_min(0.0)
    vs = v_opp + torch.sqrt(2.0 * A_B * room)
    in_lane = dy.abs() < W
    t_in = torch.where(in_lane, torch.zeros_like(dy),
                       torch.where((dy * vlat) < 0, (dy.abs() - W) / vlat.abs().clamp_min(1e-6), torch.full_like(dy, 1e9)))
    past = (dx - closing * t_in.clamp(max=H)) < -(CAR_LEN + 0.1)
    hit = ex & (gap > 0.1) & (dx < 6.0) & (t_in <= H) & ~past
    if parts:
        return torch.where(hit, vs, torch.full_like(vs, 1e9)), ex & (gap > 0.1) & (dx < 6.0), in_lane, hit
    return torch.where(hit, vs, torch.full_like(vs, 1e9))


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    out = _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)
    env = CUR["env"]
    if env is not None and env.tracker is self and env.sim.M > 1 and CUR["near"] is not None:
        lr = env.learner
        cmd = out[:, 1]
        vh, h_ahead, h_inlane, h_hit = v_safe(env, "head", parts=True)
        vt, t_ahead, t_inlane, t_hit = v_safe(env, "true", parts=True)
        ch, ct = lr & (cmd > vh), lr & (cmd > vt)
        ho = ch & ~ct
        for k, m in (("ho_phantom_no_car_ahead", ho & ~t_ahead),
                     ("ho_car_in_lane_truth_ceiling_above_cmd", ho & t_ahead & t_inlane),
                     ("ho_car_out_of_lane_head_says_in_lane_now", ho & t_ahead & ~t_inlane & h_inlane),
                     ("ho_car_out_of_lane_head_predicts_entry", ho & t_ahead & ~t_inlane & ~h_inlane),
                     ("both_caps_head_in_lane_now", ch & ct & h_inlane), ("both_caps_head_predicted", ch & ct & ~h_inlane)):
            CNT.setdefault("_t_" + k, torch.zeros((), device=cmd.device))
            CNT["_t_" + k] += m.sum()
        B = cmd.shape[0]
        if "last_ho" not in ROW or ROW["last_ho"].shape[0] != B:
            ROW["last_ho"] = torch.full((B,), -10 ** 6, dtype=torch.long, device=cmd.device)
            ROW["last_both"] = torch.full((B,), -10 ** 6, dtype=torch.long, device=cmd.device)
        t = CUR["t"]
        ROW["last_ho"] = torch.where(ch & ~ct, torch.full_like(ROW["last_ho"], t), ROW["last_ho"])
        ROW["last_both"] = torch.where(ch & ct, torch.full_like(ROW["last_both"], t), ROW["last_both"])
        for k, m in (("steps", lr), ("cap_both", ch & ct), ("cap_head_only", ch & ~ct), ("cap_true_only_missed", ~ch & ct)):
            CNT.setdefault("_t_" + k, torch.zeros((), device=cmd.device))
            CNT["_t_" + k] += m.sum()
        out = torch.stack([out[:, 0], torch.where(ch, vh.to(cmd.dtype), cmd)], 1)
    return out


def sim_step(self, *a, **kw):
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self and r.contact_onsets is not None and "last_ho" in ROW:
        on, imp = r.contact_onsets, r.contact_impacts
        t = CUR["t"]
        rec_ho = (t - ROW["last_ho"]) <= WIN
        rec_both = (t - ROW["last_both"]) <= WIN
        for kind, i in (("wall", 0), ("prop", 1)):
            hard = env.learner & on[:, i] & (imp[:, i] >= HARD)
            for k, m in ((f"hard_{kind}_after_head_only_cap", hard & rec_ho), (f"hard_{kind}_after_both_cap", hard & ~rec_ho & rec_both),
                         (f"hard_{kind}_no_recent_cap", hard & ~rec_ho & ~rec_both)):
                CNT.setdefault("_t_" + k, torch.zeros((), device=on.device))
                CNT["_t_" + k] += m.sum()
    return r


def env_step(self, action):
    CUR["env"] = self
    CUR["in_step"] = True
    try:
        out = _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False
    CUR["t"] += 1
    return out


model_v2.ActorV2.features = features
mc.MAPPlanTracker.__call__ = map_call
gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
for seed in SEEDS:
    for kind in KINDS:
        CUR.update(near=None, t=0); ROW.clear()
        out = os.path.join(os.path.dirname(OUT) or ".", f"_dd_{os.path.basename(OUT)}_{seed}_{kind}.json")
        slot = {"kind": kind, "speed_scale": [0.7, 1.0]}
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off"),
                    "--opp-slots", json.dumps([slot]), "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        for k in [k for k in CNT if k.startswith("_t_")]:
            bump(k[3:], int(CNT.pop(k)))
        print(f"seed {seed} {kind}: " + " | ".join(f"{k} {v}" for k, v in CNT.items()), flush=True)
        json.dump(CNT, open(OUT, "w"), indent=1)
