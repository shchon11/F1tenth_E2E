"""Research (2026-10-26): the real runs' MOTION in the simulator. Is it how the real car moved that makes our policy plan through
the venue boxes (43 % at 1.5-2.5 m on the replays; 32 % even with fully synthetic scans on the real trajectory; sim 4-6 %)?

The simulator's car is made to drive a venue bag's recorded trajectory: every step its state is set to the localisation pose
(/pf/pose/odom, interpolated), forward speed and yaw rate (/odom), steering atan(L r / v), wheel speed v / r_w, and its command
for the step is that speed and steering (an external driver), so the scan is cast from where the real car was a step later. Sim
boxes (0.30 x 0.30 x 0.30 m cardboard) stand where the real boxes stood (clusters of the bag's fixed objects). Our policy plans
alongside from the simulator's own scans and ego signals (open loop); its plan is measured against the boxes as in
prop_plan_clearance.py. Rows of the 16 envs start at staggered times through the run (60 s each).
TAKEOVER=1 (closed loop from those states): whenever a box is 1.5-2.5 m ahead within 0.4 m of the heading line, the row is
handed to our policy for 1.5 s (no pinning, its own plan driven by the tracker), and the box contacts and the closest approach
in that window are counted; then the replay resumes. Does a plan that crosses a box at 2 m turn into a hit when the policy is
actually driving from there?
usage: trajreplay_sim.py CKPT TRAJ.npz BOXES_JSON OUT_DIR SEEDS"""
import dataclasses
import json
import math
import os
import sys

import numpy as np
import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc, maps, mpc
from f1sim.learn import evaluate as ev
from f1sim.learn import model_v2
from f1sim.track import StaticProp

CK, TRAJ, BOXES, OUT = sys.argv[1:5]
SEEDS = [int(x) for x in sys.argv[5].split(",")]
TRACK = "real:korea_2026_competition"
DS, N_S, NENV, STEPS = 0.1, 40, 16, 2400
os.makedirs(OUT, exist_ok=True)
tz = np.load(TRAJ); pf, od = tz["pf"], tz["odom"]
boxes = np.array(json.loads(BOXES), dtype=np.float64).reshape(-1, 2)
T0, T1 = float(max(pf[0, 0], od[0, 0])), float(min(pf[-1, 0], od[-1, 0]))
span = max(T1 - T0 - STEPS * 0.025, 1.0)
_orig_load = maps.load


def load(name, *a, **kw):
    t = _orig_load(name, *a, **kw)
    if name == TRACK:
        t = dataclasses.replace(t, props=tuple(
            StaticProp("cardboard_box", float(x), float(y), 0.0, seed=i, dims=(("width", 0.3), ("depth", 0.3), ("height", 0.3)))
            for i, (x, y) in enumerate(boxes)))
    return t


maps.load = load
HEAD = {"out": None}
_orig_features = model_v2.ActorV2.features


def _features(self, e, y):
    z = _orig_features(self, e, y)
    if self.near_car is not None:
        HEAD["out"] = self.near_car(z).detach()
    return z


model_v2.ActorV2.features = _features
CUR = {"env": None, "step": 0, "rec": None, "in_step": False}
ROWS = []
TAKEOVER = bool(os.environ.get("TAKEOVER"))
TK = {"left": None, "min_d": None, "hit": None, "hard": None, "events": []}
from f1sim.sim import Simulator
_orig_sim_step = Simulator.step


def sim_step(self, *a, **kw):
    r = _orig_sim_step(self, *a, **kw)
    if TAKEOVER and CUR["in_step"] and TK["left"] is not None and r.contact_impacts is not None:
        act = TK["left"] > 0
        imp = r.contact_impacts[:, 1]
        TK["hit"] |= act & (imp > 0)
        TK["hard"] |= act & (imp >= 1.5)
    return r


Simulator.step = sim_step


def recorded(n_rows, step, device):
    t = T0 + np.arange(n_rows) * (span / max(n_rows - 1, 1)) + step * 0.025
    t = np.minimum(t, T1)
    x, y, yaw = np.interp(t, pf[:, 0], pf[:, 1]), np.interp(t, pf[:, 0], pf[:, 2]), np.interp(t, pf[:, 0], pf[:, 3])
    v, w = np.interp(t, od[:, 0], od[:, 1]), np.interp(t, od[:, 0], od[:, 2])
    return [torch.as_tensor(a, dtype=torch.float32, device=device) for a in (x, y, yaw, v, w)]


class ReplayDriver:
    """the external driver: the recorded speed and the steering of the recorded yaw rate"""
    rows = None
    resets = 0

    def reset(self, hit):                     # the env calls this on a row's reset; the next step's state comes from the record
        ReplayDriver.resets += int(np.asarray(hit).sum())

    def command_env(self, env, rows):
        x, y, yaw, v, w = CUR["rec"]
        v, w = v[rows], w[rows]
        P = env.sim.P
        L = (P["lf"] + P["lr"]).reshape(-1)[rows] if torch.is_tensor(P["lf"]) else float(P["lf"] + P["lr"])
        steer = torch.atan(L * w / v.clamp_min(0.3))
        return torch.stack([steer, v], 1)


def measure(env, action, v_meas):
    B = action.shape[0]
    spec, vmax = env.tracker.spec, float(env.tracker.v_max)
    k, L, _ = mpc.decode_wp4(action.float(), v_meas.float(), vmax, torch.full((B,), 9.0, device=action.device), spec)
    s = torch.arange(1, N_S + 1, device=k.device, dtype=k.dtype) * DS
    seg = (s[None, :] / (L[:, None] / k.shape[1]).clamp_min(1e-3)).floor().long().clamp(max=k.shape[1] - 1)
    th = torch.cumsum(k.gather(1, seg) * DS, 1)
    px, py = torch.cumsum(torch.cos(th) * DS, 1), torch.cumsum(torch.sin(th) * DS, 1)
    on = s[None, :] <= torch.minimum(L, torch.full_like(L, 4.0))[:, None]
    st = env.sim.state
    bx = torch.as_tensor(boxes[:, 0], dtype=torch.float32, device=st.device)[None, :]
    by = torch.as_tensor(boxes[:, 1], dtype=torch.float32, device=st.device)[None, :]
    c, sn = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
    qx, qy = bx - st[:, dyn.IX][:, None], by - st[:, dyn.IY][:, None]
    xl, yl = c * qx + sn * qy, -sn * qx + c * qy
    sel = (xl > 0.5) & (xl < 3.5) & (yl.abs() < 0.6)
    if not bool(sel.any()):
        return
    d = torch.hypot(px[:, None, :] - xl[:, :, None], py[:, None, :] - yl[:, :, None])
    rr = torch.hypot(px, py)
    near = on[:, None, :] & (rr[:, None, :] <= torch.hypot(xl, yl)[:, :, None] + 0.5)
    dmin = torch.where(near, d, torch.full_like(d, float("inf"))).min(2).values
    reach = torch.where(on, rr, torch.zeros_like(rr)).max(1).values[:, None] >= torch.hypot(xl, yl) - 0.15
    clear = torch.where(reach & torch.isfinite(dmin), dmin - 0.30, torch.full_like(dmin, float("nan")))
    b, j = sel.nonzero(as_tuple=True)
    h = HEAD["out"]
    logit = h[b, 0].float() if h is not None and h.shape[0] == B else torch.full_like(xl[b, j], float("nan"))
    ROWS.append(torch.stack([xl[b, j], yl[b, j], st[b, dyn.IVX], clear[b, j], L[b], logit, st[b, dyn.IR]], 1).float().cpu())


_orig_map_call = mc.MAPPlanTracker.__call__
_orig_step = gym_env.F1VecEnv.step


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    env = CUR["env"]
    if env is not None and env.tracker is self:
        measure(env, action, v_meas)
    return _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)


def env_step(self, action):
    if CUR["env"] is not self:
        CUR.update(env=self, step=0)
        self.set_ego_driver(ReplayDriver())
        self._all_external = False
    rec = recorded(self.B, CUR["step"], self.sim.state.device)
    CUR["rec"] = rec
    x, y, yaw, v, w = rec
    st = self.sim.state
    P = self.sim.P
    dev = st.device
    if TK["left"] is None:
        TK["left"] = torch.zeros(self.B, dtype=torch.long, device=dev)
        TK["min_d"] = torch.full((self.B,), 9.0, device=dev)
        TK["hit"] = torch.zeros(self.B, dtype=torch.bool, device=dev); TK["hard"] = torch.zeros_like(TK["hit"])
    pin = TK["left"] == 0
    Lw = (P["lf"] + P["lr"]).reshape(-1) if torch.is_tensor(P["lf"]) else float(P["lf"] + P["lr"])
    rw = P["r_w"].reshape(-1) if torch.is_tensor(P["r_w"]) else float(P["r_w"])
    sel = lambda new, old: torch.where(pin, new, old)
    st[:, dyn.IX], st[:, dyn.IY], st[:, dyn.IYAW] = sel(x, st[:, dyn.IX]), sel(y, st[:, dyn.IY]), sel(yaw, st[:, dyn.IYAW])
    st[:, dyn.IVX], st[:, dyn.IVY], st[:, dyn.IR] = sel(v, st[:, dyn.IVX]), sel(torch.zeros_like(v), st[:, dyn.IVY]), sel(w, st[:, dyn.IR])
    st[:, dyn.ISTEER] = sel(torch.atan(Lw * w / v.clamp_min(0.3)), st[:, dyn.ISTEER])
    st[:, dyn.IOMEGA] = sel(v / rw, st[:, dyn.IOMEGA])
    if TAKEOVER:
        # a box 1.5-2.5 m ahead within 0.4 m of the heading line on a pinned row: hand that row to the policy for 1.5 s
        bx = torch.as_tensor(boxes[:, 0], dtype=torch.float32, device=dev)[None, :]
        by = torch.as_tensor(boxes[:, 1], dtype=torch.float32, device=dev)[None, :]
        c, sn = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
        qx, qy = bx - st[:, dyn.IX][:, None], by - st[:, dyn.IY][:, None]
        xl, yl = c * qx + sn * qy, -sn * qx + c * qy
        start = pin & ((xl > 1.5) & (xl < 2.5) & (yl.abs() < 0.4)).any(1) & (v > 1.0) & (TK.get("cool", torch.zeros_like(TK["left"])) == 0)
        if bool(start.any()):
            TK["left"] = torch.where(start, torch.full_like(TK["left"], 60), TK["left"])
            TK["min_d"] = torch.where(start, torch.full_like(TK["min_d"], 9.0), TK["min_d"])
            TK["hit"] = TK["hit"] & ~start; TK["hard"] = TK["hard"] & ~start
        d_box = torch.hypot(qx, qy).min(1).values
        active = TK["left"] > 0
        TK["min_d"] = torch.where(active, torch.minimum(TK["min_d"], d_box), TK["min_d"])
        # the policy drives the rows in a takeover: the replay driver's command goes to the other rows only
        self.external_driven = ~active
        self.external_slot_rows[0] = (~active).nonzero().flatten()
        self._all_external = False
    CUR["in_step"] = True
    try:
        out = _orig_step(self, action)
    finally:
        CUR["in_step"] = False
    if TAKEOVER:
        ending = TK["left"] == 1
        if bool(ending.any()):
            for i in ending.nonzero().flatten().tolist():
                TK["events"].append((float(TK["min_d"][i]), bool(TK["hit"][i]), bool(TK["hard"][i])))
        TK["left"] = (TK["left"] - 1).clamp_min(0)
        cool = TK.get("cool", torch.zeros_like(TK["left"]))
        TK["cool"] = torch.where(ending, torch.full_like(cool, 80), (cool - 1).clamp_min(0))   # 2 s before the next takeover
    CUR["step"] += 1
    return out


mc.MAPPlanTracker.__call__ = map_call
gym_env.F1VecEnv.step = env_step
for seed in SEEDS:
    out = os.path.join(OUT, f"replay_s{seed}.json")
    if os.path.exists(out):
        continue
    ROWS.clear(); CUR.update(env=None, step=0)
    sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", str(NENV), "--steps", str(STEPS),
                "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                "--seed", str(seed), "--collision-mode", "soft", "--race-size", "1", "--output", out,
                "--dial-value", "1.0489", "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
    ev.main()
    r = torch.cat(ROWS).numpy() if ROWS else np.zeros((0, 7), np.float32)
    np.save(os.path.join(OUT, f"rows_s{seed}.npy"), r)
    m = (r[:, 0] >= 1.5) & (r[:, 0] < 2.5) & np.isfinite(r[:, 3]) & (r[:, 2] > 1.0)
    print(f"seed {seed}: box 1.5-2.5 m ahead, plan reaches it {int(m.sum())}: crosses {100 * np.mean(r[m, 3] < 0):.0f} %, clearance p50 "
          f"{np.median(r[m, 3]) if m.any() else float('nan'):+.2f} m, read as a car {100 * np.nanmean(r[m, 5] > 0):.0f} %, speed p50 "
          f"{np.median(r[m, 2]) if m.any() else 0:.2f}, |yaw rate| p50 {np.median(np.abs(r[m, 6])) if m.any() else 0:.2f}, "
          f"row resets {ReplayDriver.resets}", flush=True)
    if TAKEOVER and TK["events"]:
        e = np.array(TK["events"], dtype=np.float64)
        print(f"  TAKEOVERS {len(e)}: box contact {100 * e[:, 1].mean():.0f} %, hard {100 * e[:, 2].mean():.0f} %, closest approach to the box "
              f"centre p10/p50 {np.percentile(e[:, 0], 10):.2f}/{np.median(e[:, 0]):.2f} m", flush=True)
        np.save(os.path.join(OUT, f"takeovers_s{seed}.npy"), e)
    TK.update(left=None, events=[]); TK.pop("cool", None)
