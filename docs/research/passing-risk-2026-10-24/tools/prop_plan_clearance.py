"""Research (2026-10-25): in the simulator, how close does the entry policy's PLAN pass a prop ahead? The sim side of
real_bag_plan_vs_objects.py.

On the real venue bags (open loop, the ForzaETH stack driving) our policy's plan ran into a fixed object 1.5-2.5 m ahead in
~40 % of the scans where it reached it, and cleared every one under 1.5 m. Is that how it treats a prop in the simulator
(late avoidance, then fine), or does it see real obstacles worse? Same measurement, closed loop, solo laps with props on the
line (finals' solo cells): every learner step with a live prop whose centre is 0.5-3.5 m ahead and within 0.6 m of our
heading (car frame); the plan decoded as the tracker does and walked every 0.1 m up to min(L, 4 m); clearance = the
smallest distance to the prop's centre - 0.15 (prop half width) - 0.15 (our half width), only when the path gets as far
as the prop. Rows (distance, lateral, speed, clearance, L) -> OUT/rows_s{seed}.npy.
OPEN_LOOP=forzaeth: the real bags' situation in the simulator -- the car is driven by the ForzaETH planner (the console's
"내 차" planner ego on MAP, true state), while our policy plans alongside every step exactly as the ROS node did on the
replays (its plans fill its own plan history; they are decoded and measured, never driven). Closed vs open loop on the
same props says whether the real bags' 43 % "into" is the replay's doing or the real objects'.
usage: prop_plan_clearance.py CKPT OUT_DIR SEEDS"""
import json
import os
import sys

import numpy as np
import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc, mpc
from f1sim.learn import evaluate as ev
if os.environ.get("VEIL"):
    import veil_patch  # noqa: F401  (10-25: mixed pixels at object edges, see veil_patch.py)
from f1sim.learn import model_v2
HEAD = {"out": None}
_orig_features = model_v2.ActorV2.features


def _features(self, e, y):
    z = _orig_features(self, e, y)
    if self.near_car is not None:
        HEAD["out"] = self.near_car(z).detach()          # the policy's nearest-car reading this step (exists logit first)
    return z


model_v2.ActorV2.features = _features

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
TRACK = "real:korea_2026_competition"
DS, N_S = 0.1, 40
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None}
ROWS = []
_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step


def measure(env, action, v_meas):
    sim = env.sim
    ep = sim.track.env_props
    if ep is None:
        return
    spec, vmax = env.tracker.spec, float(env.tracker.v_max)
    B = action.shape[0]
    k, L, _ = mpc.decode_wp4(action.float(), v_meas.float(), vmax, torch.full((B,), 9.0, device=action.device), spec)
    s = torch.arange(1, N_S + 1, device=k.device, dtype=k.dtype) * DS
    seg = (s[None, :] / (L[:, None] / k.shape[1]).clamp_min(1e-3)).floor().long().clamp(max=k.shape[1] - 1)
    kap = k.gather(1, seg)
    th = torch.cumsum(kap * DS, 1)
    px, py = torch.cumsum(torch.cos(th) * DS, 1), torch.cumsum(torch.sin(th) * DS, 1)
    on = s[None, :] <= torch.minimum(L, torch.full_like(L, 4.0))[:, None]                      # (B, N_S)
    poses, _n, _d, zlo, zhi = ep.slots(sim.eid)
    st = sim.state
    c, sn = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
    qx, qy = poses[..., 0] - st[:, dyn.IX][:, None], poses[..., 1] - st[:, dyn.IY][:, None]
    xl, yl = c * qx + sn * qy, -sn * qx + c * qy                                            # (B, S) car frame
    sel = env.learner[:, None] & (zhi > zlo) & (xl > 0.5) & (xl < 3.5) & (yl.abs() < 0.6)
    if not bool(sel.any()):
        return
    d = torch.hypot(px[:, None, :] - xl[:, :, None], py[:, None, :] - yl[:, :, None])         # (B, S, N_S)
    rr = torch.hypot(px, py)                                                                  # (B, N_S)
    near = on[:, None, :] & (rr[:, None, :] <= torch.hypot(xl, yl)[:, :, None] + 0.5)
    dmin = torch.where(near, d, torch.full_like(d, float("inf"))).min(2).values
    reach = (torch.where(on, rr, torch.zeros_like(rr)).max(1).values[:, None] >= torch.hypot(xl, yl) - 0.15)
    clear = torch.where(reach & torch.isfinite(dmin), dmin - 0.30, torch.full_like(dmin, float("nan")))
    b, j = sel.nonzero(as_tuple=True)
    v = st[:, dyn.IVX]
    h = HEAD["out"]
    if h is not None and h.shape[0] != B:
        full = torch.full((B, h.shape[1]), -10.0, device=h.device); full[env.learner] = h; h = full
    logit = h[b, 0].float() if h is not None else torch.full_like(v[b], float("nan"))
    # 10-25: which catalogue shape the prop is (ProceduralObstacles.p_sid -> shapes[sid].style / across), to split by kind
    sid = ep.p_sid[sim.eid][b, j].float() if hasattr(ep, "p_sid") else torch.full_like(v[b], -1.0)
    # where the reading says its car is (LiDAR frame -> car frame), to tell a box read as a car from the real car of a race
    hx = (h[b, 1].float() * 5.0 + float(env.cfg.lidar.mount_x)) if h is not None else torch.full_like(v[b], float("nan"))
    hy = (h[b, 2].float() * 2.0) if h is not None else torch.full_like(v[b], float("nan"))
    ROWS.append(torch.stack([xl[b, j], yl[b, j], v[b], clear[b, j], L[b], logit, sid, hx, hy], 1).float().cpu())


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
if os.environ.get("OPEN_LOOP"):
    from f1sim.learn import common as _common
    _orig_make_env = _common.make_env

    def make_env(*a, **kw):
        env = _orig_make_env(*a, **kw)
        lim = dict(a_lat=7.0, a_acc=6.5, a_brake=4.0)
        _trs, rls = _common.load_tracks([TRACK], racelines=True, objective="min_time", **lim)
        drv = _common.build_ego_driver(env, os.environ["OPEN_LOOP"], rls=rls, limits=lim, estimated=False)
        env.set_ego_driver(drv)
        env._all_external = False          # keep solving the tracker on our policy's plan: the hook measures it
        return env
    _common.make_env = make_env
    ev.common.make_env = make_env
for seed in SEEDS:
    out = os.path.join(OUT, f"solo_s{seed}.json")
    if os.path.exists(out):
        continue
    ROWS.clear()
    sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                # RACE_KIND (10-25): a race of 2 with this opponent kind instead of solo -- does another car in the scene make
                # boxes read as a car (the real bags had one in 11 of 13 runs) and planned around later?
                *(["--race-size", "2", "--opp-slots", json.dumps([{"kind": os.environ["RACE_KIND"], "speed_scale": [0.7, 1.0]}])]
                  if os.environ.get("RACE_KIND") else ["--race-size", "1"]),
                "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                "--procedural-max-props", "10", "--procedural-raceline-corridor", "off", "--output", out,
                "--dial-value", "1.0489", "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
    ev.main()
    r = torch.cat(ROWS).numpy() if ROWS else np.zeros((0, 9), np.float32)
    np.save(os.path.join(OUT, f"rows_s{seed}.npy"), r)
    for lo, hi in ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi); c = r[m, 3]; dec = np.isfinite(c)
        if dec.sum():
            print(f"seed {seed} prop {lo}-{hi} m ahead: steps {int(m.sum())}, plan reaches it {int(dec.sum())}: clears (> 0.05) "
                  f"{100 * np.mean(c[dec] > 0.05):.0f} %, into (< 0) {100 * np.mean(c[dec] < 0):.0f} %, clearance p50 {np.median(c[dec]):+.2f} m, "
                  f"speed p50 {np.median(r[m, 2]):.2f}, head says 'car' {100 * np.nanmean(r[m, 5] > 0):.1f} % (on the box: {100 * np.nanmean((r[m, 5] > 0) & (np.hypot(r[m, 7] - r[m, 0], r[m, 8] - r[m, 1]) < 0.6)):.1f} %)", flush=True)
