"""Research (2026-10-25): a speed limit for an obstacle on the planned path -- the upper bound from the TRUE prop positions.

With the props on the raceline the opponents push them about (33 contacts /km) and a pushed prop enters our path ~0.56 s
before we reach it; v2 then brakes but rarely steers round (8 %), and hard prop hits are ~40 % of all hard hits in those
races (0.85-0.93 /km, fresh seeds 91-94) -- the speed limit for a car ahead does nothing for them. Here, at every step,
the policy's own plan (wp4: four curvatures over the plan length, extended at its last curvature out to `reach`) is
swept as a corridor of half-width W; a live prop whose centre lies inside it is an obstacle at that arc length, and the
speed command is held to the speed from which the car can still stop short of it:
    limit = sqrt(2 a_b max(0, s - front - radius - g_min - v tau)).
True prop positions (oracle): what a perfect obstacle reading would buy. SHIELD_MEASURE_ONLY=1: count, do not cap.
OSL=1 also switches on the speed limit for a car ahead (the shipped one, EnvConfig.opponent_speed_limit) -- needs a
checkpoint with a nearest-car output. Finals race cells (seeds, 4 kinds, props on the line unless RACE_CORRIDOR=on); SOLO=1
runs the props-on-the-line solo cells instead (32 cars x 60 s per seed: the cost where nothing pushes a prop).
First read (10-25, seeds 81-82 + part of 83-84, car limit on in both): prop hard hits 1.45 -> 0.40 /km but race speed
4.46 -> 3.09 m/s -- the limit held on ~10 % of steps (2.3 m/s cut): the plan passes near props it steers round, and the
extension past the plan runs into props the policy would still avoid. Narrower variants:
  REACH=0: the plan's own length only (no extension);
  MOVED_ONLY=1: only props displaced > 0.2 m from where they were at the episode's start (pushed by a car; 78 % of the
  hard-hit approaches in races were such props, props-and-perception-2026-10-23).
usage: prop_limit_race.py CKPT OUT_DIR SEEDS [A_B=3.0] [W=0.40] [TAU=0.2] [GMIN=0.1] [REACH=6.0]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc, mpc
from f1sim.learn import evaluate as ev

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
A_B = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
W = float(sys.argv[5]) if len(sys.argv) > 5 else 0.40
TAU = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
GMIN = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
REACH = float(sys.argv[8]) if len(sys.argv) > 8 else 6.0
PLAN_ONLY = REACH <= 0
REACH = 6.0 if PLAN_ONLY else REACH
MOVED_ONLY = bool(os.environ.get("MOVED_ONLY"))
FIRST = {}
FRONT, RADIUS, DS = 0.45, 0.15, 0.1          # [m] reference point -> front bumper; a prop's centre -> its face; path step
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
TRACK = "real:korea_2026_competition"
MEASURE = bool(os.environ.get("SHIELD_MEASURE_ONLY"))
SOLO = bool(os.environ.get("SOLO"))
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None}
STAT = {"steps": 0, "active": 0, "cut_sum": 0.0}
_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step
N_S = int(REACH / DS)


def path_points(k, L):
    """(B, N_S, 2) points of the plan's path in the car frame: curvature k[:, i] over the i-th quarter of L, then k[:, -1]"""
    s = torch.arange(1, N_S + 1, device=k.device, dtype=k.dtype) * DS                       # (N_S,)
    seg = (s[None, :] / (L[:, None] / k.shape[1]).clamp_min(1e-3)).floor().long().clamp(max=k.shape[1] - 1)
    kap = k.gather(1, seg)                                                                  # (B, N_S)
    th = torch.cumsum(kap * DS, 1)
    x = torch.cumsum(torch.cos(th) * DS, 1)
    y = torch.cumsum(torch.sin(th) * DS, 1)
    return torch.stack([x, y], -1), s


def prop_limit(env, action, v_meas):
    sim = env.sim
    ep = sim.track.env_props
    B = action.shape[0]
    inf = torch.full((B,), float("inf"), device=action.device)
    if ep is None:
        return inf
    spec, vmax = env.tracker.spec, float(env.tracker.v_max)
    k, L, _ = mpc.decode_wp4(action.float(), v_meas.float(), vmax, torch.full((B,), 9.0, device=action.device), spec)
    pts, s = path_points(k, L)                                                              # (B, N_S, 2)
    poses, _n, _d, zlo, zhi = ep.slots(sim.eid)
    live = zhi > zlo                                                                        # (B, S)
    if MOVED_ONLY:
        first = FIRST.get("xy")
        fresh = sim.steps <= 1
        if first is None or first.shape != poses[..., :2].shape:
            first = poses[..., :2].clone()
        first = torch.where(fresh[:, None, None], poses[..., :2], first)
        FIRST["xy"] = first
        live = live & ((poses[..., :2] - first).norm(dim=-1) > 0.2)
    st = sim.state
    c, sn = torch.cos(st[:, dyn.IYAW])[:, None], torch.sin(st[:, dyn.IYAW])[:, None]
    px, py = poses[..., 0] - st[:, dyn.IX][:, None], poses[..., 1] - st[:, dyn.IY][:, None]
    xl, yl = c * px + sn * py, -sn * px + c * py                                            # (B, S) car frame
    d2 = (pts[:, None, :, 0] - xl[:, :, None]) ** 2 + (pts[:, None, :, 1] - yl[:, :, None]) ** 2   # (B, S, N_S)
    dmin, j = d2.min(2)
    inside = live & (dmin.sqrt() < W) & (xl > 0)
    if PLAN_ONLY:
        inside = inside & (s[j] <= L[:, None])
    s_hit = torch.where(inside, s[j], torch.full_like(dmin, float("inf"))).min(1).values     # nearest such prop
    v = st[:, dyn.IVX]
    room = (s_hit - FRONT - RADIUS - GMIN - v.clamp_min(0) * TAU).clamp_min(0.0)
    return torch.where(torch.isfinite(s_hit), torch.sqrt(2.0 * A_B * room), inf)


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    out = _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)
    env = CUR["env"]
    if env is not None and env.tracker is self:
        lim = prop_limit(env, action, v_meas)
        lr = env.learner
        cmd = out[:, 1]
        cut = lr & (cmd > lim)
        new = cmd if MEASURE else torch.where(cut, lim.to(cmd.dtype), cmd)
        if "t_steps" not in STAT:
            for k_ in ("t_steps", "t_active", "t_cut"):
                STAT[k_] = torch.zeros((), device=cmd.device)
        STAT["t_steps"] += lr.sum(); STAT["t_active"] += cut.sum()
        STAT["t_cut"] += torch.where(cut, cmd - lim.to(cmd.dtype), torch.zeros_like(cmd)).sum()
        out = torch.stack([out[:, 0], new], 1)
    return out


def env_step(self, action):
    CUR["env"] = self
    return _orig_env_step(self, action)


mc.MAPPlanTracker.__call__ = map_call
gym_env.F1VecEnv.step = env_step
osl = ["--opponent-speed-limit"] if os.environ.get("OSL") else []
for seed in SEEDS:
    for kind in (("solo",) if SOLO else KINDS):
        out = os.path.join(OUT, f"{'solo' if SOLO else 'race'}_s{seed}_{kind}.json")
        if os.path.exists(out):
            continue
        race = (["--race-size", "1"] if SOLO else
                ["--race-size", "2", "--opp-slots", json.dumps([{"kind": kind, "speed_scale": [0.7, 1.0]}])] + osl)
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off"),
                    *race, "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        if "t_steps" in STAT:
            STAT["steps"] += int(STAT.pop("t_steps")); STAT["active"] += int(STAT.pop("t_active"))
            STAT["cut_sum"] += float(STAT.pop("t_cut"))
        print(f"seed {seed} {kind}: prop limit active {STAT['active']}/{STAT['steps']} learner steps "
              f"({100 * STAT['active'] / max(STAT['steps'], 1):.2f} %), mean cut {STAT['cut_sum'] / max(STAT['active'], 1):.2f} m/s",
              flush=True)
json.dump({"a_b": A_B, "w": W, "tau": TAU, "gmin": GMIN, "reach": REACH, "plan_only": PLAN_ONLY, "moved_only": MOVED_ONLY,
           "measure_only": MEASURE, "solo": SOLO,
           "osl": bool(osl), **STAT}, open(os.path.join(OUT, f"stats_{'_'.join(map(str, SEEDS))}.json"), "w"))
