"""Research (2026-10-23): is the plan an honest forecast of what the car then does?

Under MAP only v_1 is executed (map_authority.py). The later waypoint speeds can still carry information -- the policy
may know it will slow for the corner -- or none. For every plan of a learner at step t, with waypoints at arc s_k
(k = 1..4) and speeds v_k, find the step t' where the car has TRAVELLED s_k from where the plan was made (integral of
|v| dt, same episode) and compare the speed it had there, v(t'), with three forecasts:
  plan     v_k(t)            -- what the plan said
  hold     v(t)              -- "the speed stays as it is"
  first    v_1(t)            -- "the commanded speed stays" (all MAP executes)
If the plan's error is no smaller than the "first" forecast's at k = 2..4, the profile beyond v_1 carries nothing.
Also where the car got to: its distance from the plan's own waypoint k, in the frame of the pose the plan was made from.
Solo, Korea, rep floor, 32 cars x 30 s, the checkpoint's follower and dial (as the finals solo cells).
For a linear (v1) plan the four points are 0.1 .. 0.4 of its 1.5 s length, i.e. the same first 0.6 s.
usage: plan_honesty.py CKPT [OUT.json]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, mpc
from f1sim.learn.evaluate import evaluate
from f1sim.mpc import contract_of
from f1sim.params import Config

CK = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
TRACK = os.environ.get("HON_TRACK", "real:korea_2026_competition")
extra = torch.load(CK, map_location="cpu", weights_only=False)["extra"]
contract = contract_of(extra)
rec = {k: [] for k in ("v", "fresh", "vk", "sk", "p0", "p1", "wx", "wy")}
DT = {"dt": None}
orig = gym_env.F1VecEnv.step
Q_WP4, Q_LIN = (0.25, 0.5, 0.75, 1.0), (0.1, 0.2, 0.3, 0.4)


def step(self, a):
    sp, lr = self.tracker.spec, self.learner
    vm = self._tracker_plan_speed()
    vmax, cap = float(self.tracker.v_max), self.speed_cap.float()
    if sp.speed_mode == "wp4":
        k, L, vk = mpc.decode_wp4(a.float(), vm.float(), vmax, cap, sp)
        x, y, _, s, _ = mpc.wp4_path(k, L, sp, extend=1.0)          # s uniform over [0, L]
        qs = Q_WP4
    else:
        k, L, v0, v1 = mpc.decode(a.float(), vm.float(), vmax, cap, sp)
        x, y, _, s = mpc.path_points(k, L, 49)[:4]                    # s uniform over [0, L]
        qs = Q_LIN
        vk = v0[:, None] + (v1 - v0)[:, None] * torch.tensor(qs, device=a.device)[None]
    n = s.shape[1]
    idx = [min(n - 1, int(round(q * (n - 1)))) for q in qs]
    sk = torch.stack([s[:, i] for i in idx], 1)
    wx, wy = torch.stack([x[:, i] for i in idx], 1), torch.stack([y[:, i] for i in idx], 1)
    p0 = self.sim.state[:, [dyn.IX, dyn.IY, dyn.IYAW]].clone()
    out = orig(self, a)
    r = self.last_result
    DT["dt"] = float(self.sim.control_dt)
    v = torch.hypot(r.state[:, dyn.IVX], r.state[:, dyn.IVY])
    fresh = self.sim.steps <= 1                                       # reset in this step, or its first step
    for key, val in (("v", v), ("fresh", fresh), ("vk", vk), ("sk", sk), ("p0", p0),
                     ("p1", r.state[:, [dyn.IX, dyn.IY, dyn.IYAW]]), ("wx", wx), ("wy", wy)):
        rec[key].append(val[lr].detach().float().cpu() if key != "fresh" else val[lr].cpu())
    return out


gym_env.F1VecEnv.step = step
cfg = Config()
cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0)
cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
kw = dict(dial_value=1.0489)
# HON_FOLLOWER=mpc|map overrides the checkpoint's own follower (map: the Korea round-5 numbers), for the 2x2 of
# training follower x driving follower
FOL = os.environ.get("HON_FOLLOWER") or contract.get("plan_follower", "mpc")
if FOL == "map":
    mp = dict(contract.get("plan_follower_map") or {}) or json.load(open(
        "/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/out/map_k26_round5.json"))
    kw.update(plan_follower="map", plan_follower_map=mp)
else:
    kw.update(plan_follower="mpc")
evaluate(CK, [TRACK], 32, 1200, 9.0, "cuda", seed=123, cfg=cfg, protocol="trials", budget_laps=3.0, **kw)
gym_env.F1VecEnv.step = orig
R = {k: torch.stack(v) for k, v in rec.items()}
V, F = R["v"], R["fresh"]
T, B = V.shape
Dg = torch.cumsum(V * DT["dt"], 0)                     # distance travelled by the end of each step (monotone)
D0 = Dg - V * DT["dt"]                                  # ... by its start: where the plan of step t was made
ep = torch.cumsum(F.long(), 0)
ar = torch.arange(T)[:, None].expand(T, B)
res = {"ckpt": CK, "speed_mode": contract.get("speed_mode"), "follower": FOL,
       "plans": int(T * B), "by_point": {}}
for k in range(4):
    target = D0 + R["sk"][..., k]
    t2 = torch.stack([torch.searchsorted(Dg[:, b].contiguous(), target[:, b].contiguous()) for b in range(B)], 1)
    ok = t2 < T
    t2c = t2.clamp(max=T - 1)
    ok &= (ep.gather(0, t2c) == ep) & (t2c >= ar)
    ve = V.gather(0, t2c)
    e_plan = (ve - R["vk"][..., k]).abs()[ok]
    e_hold = (ve - V).abs()[ok]
    e_first = (ve - R["vk"][..., 0]).abs()[ok]
    # where the car got to, in the frame of the pose the plan was made from
    x0, y0, yaw0 = R["p0"][..., 0], R["p0"][..., 1], R["p0"][..., 2]
    gx = R["p1"][..., 0].gather(0, t2c) - x0
    gy = R["p1"][..., 1].gather(0, t2c) - y0
    c, s_ = torch.cos(yaw0), torch.sin(yaw0)
    px, py = c * gx + s_ * gy, -s_ * gx + c * gy
    e_pos = torch.hypot(px - R["wx"][..., k], py - R["wy"][..., k])[ok]
    e_lat = (py - R["wy"][..., k]).abs()[ok]
    res["by_point"][f"p{k + 1}"] = {
        "n": int(ok.sum()), "mean_arc_m": round(float(R["sk"][..., k][ok].mean()), 3),
        "speed_err_plan": round(float(e_plan.mean()), 3), "speed_err_hold": round(float(e_hold.mean()), 3),
        "speed_err_first": round(float(e_first.mean()), 3),
        "speed_change_mean_abs": round(float((ve - V).abs()[ok].mean()), 3),
        "pos_err_m": round(float(e_pos.mean()), 3), "lat_err_m": round(float(e_lat.mean()), 3)}
print(json.dumps(res, indent=1), flush=True)
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
