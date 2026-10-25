"""Research (2026-10-25): a speed limit from the LiDAR along the policy's own plan -- the deployable form of the prop limit.

The prop limit from TRUE prop positions, restricted to the plan's own length and a 0.30 m centre-to-centre corridor
(prop_limit_race.py REACH=0), cut hard prop hits 1.29 -> 0.58 /km and all hard hits 2.07 -> 1.45 at -5 % race speed (props
on the line, car limit on, seeds 81-84). Here the obstacle is whatever the current scan returns: each return, put in the
car frame through the LiDAR's nominal mount, is an obstacle point; the plan's path (wp4: four curvatures over the plan
length, `prop_limit_race.path_points`) is walked in 0.1 m steps, and at the first point within THR of a return -- the car's
reference point that close to a return -- the speed is held to the speed from which it can still stop there:
    limit = sqrt(2 a_b max(0, s - g_min - v tau)).
Returns from a car are left to the speed limit for a car ahead: every return within 0.45 m of where the policy's own
nearest-car output puts the nearest car (env.opponent_reading, with the car limit on) is ignored. No truth anywhere: the
same arithmetic runs on the car (the controller node has the scan and the plan). SHIELD_MEASURE_ONLY=1: count, do not
cap. OSL=1: the car limit on too. SOLO=1: solo cells (props on the line unless RACE_CORRIDOR=on; NOPROPS=1: no props).
First read (10-25, props-on races, car limit on, 11 cells): the limit held on 18 % of steps (2.4 m/s cut), race 4.46 -> 3.77,
hard car 0.55 -> 0.85, prop 1.46 -> 0.68: far more than the oracle (6 %). Suspects: floor returns (the simulated LiDAR
returns HIT_GROUND where a pitched beam meets the floor) and car returns the 0.45 m exclusion misses. TYPE_FILTER=1 drops
both by the simulator's own beam class (privileged: what a perfect floor / car rejection would leave). Read: no better
(17 %, race 3.83): the trigger was the WALLS beside the car -- the path's first points (from 0.1 m) sit within THR of the
side wall whenever the body runs close to it, and room = s - ... came out ~0, a stop command. Fixed: only path points past
the front bumper (s > FRONT 0.45 m) count, and the gap is s - FRONT - g_min (a return beside the body is not ahead of it).
Still 14-17 % of steps (2.4-3.0 m/s cut) on the raw scan: the plan's path is a sketch that runs within a few cm of the walls
on this 2 m wide track, though the car does not. MAP_FILTER=1 drops every return within 0.15 m of the static map (ForzaETH's
detection step: the map from the track, the pose TRUE here -- a particle filter is within ~5 cm on this car), so only what
is not wall (props, cars, floor) can hold the speed; with TYPE_FILTER=1 as well, floor and car returns go too.
CLUSTER_FILTER=1 (10-26, MAP-FREE -- what our map-less stack can run): keep only returns that belong to a small isolated
cluster of the scan -- a run of >= 3 valid beams with no range jump over 0.15 m between neighbours, end to end under 0.6 m,
standing at least 0.3 m in front of the beams beyond both of its ends. A wall is long; a box or a pole is not.
usage: scan_limit_race.py CKPT OUT_DIR SEEDS [THR=0.19] [A_B=3.0] [TAU=0.2] [GMIN=0.1]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc, mpc
from f1sim.learn import evaluate as ev
from f1sim.opponent_speed_limit import relative

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.19
A_B = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
TAU = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
GMIN = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
CAR_EXCLUDE, DS, REACH, FRONT = 0.45, 0.1, 6.0, 0.45
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
TRACK = "real:korea_2026_competition"
MEASURE = bool(os.environ.get("SHIELD_MEASURE_ONLY"))
SOLO = bool(os.environ.get("SOLO"))
NOPROPS = bool(os.environ.get("NOPROPS"))
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None}
STAT = {"steps": 0, "active": 0, "cut_sum": 0.0}
_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step
N_S = int(REACH / DS)


def path_points(k, L):
    s = torch.arange(1, N_S + 1, device=k.device, dtype=k.dtype) * DS
    seg = (s[None, :] / (L[:, None] / k.shape[1]).clamp_min(1e-3)).floor().long().clamp(max=k.shape[1] - 1)
    kap = k.gather(1, seg)
    th = torch.cumsum(kap * DS, 1)
    return torch.stack([torch.cumsum(torch.cos(th) * DS, 1), torch.cumsum(torch.sin(th) * DS, 1)], -1), s


def cluster_keep(r, ok, px, py, brk_jump=0.15, max_ext=0.6, stand=0.3, min_beams=3):
    """(B, n) the valid returns that belong to a small isolated cluster (see CLUSTER_FILTER)"""
    B, n = r.shape
    dev = r.device
    rf = torch.where(ok, r, torch.full_like(r, 30.0))                       # no return = far behind
    jump = (rf[:, 1:] - rf[:, :-1]).abs() > brk_jump
    brk = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), jump | ~ok[:, 1:] | ~ok[:, :-1]], 1)
    seg = torch.cumsum(brk.long(), 1)                                       # segment id of every beam
    gid = (seg + torch.arange(B, device=dev)[:, None] * (n + 2)).flatten()
    idx = torch.arange(n, device=dev)[None].expand(B, n).flatten()
    first = torch.full((B * (n + 2),), n, dtype=torch.long, device=dev).scatter_reduce(0, gid, idx, "amin", include_self=True)
    last = torch.full((B * (n + 2),), -1, dtype=torch.long, device=dev).scatter_reduce(0, gid, idx, "amax", include_self=True)
    f, l = first[gid].view(B, n), last[gid].view(B, n)
    bi = torch.arange(B, device=dev)[:, None].expand(B, n)
    ext = torch.hypot(px[bi, l] - px[bi, f], py[bi, l] - py[bi, f])
    r_left, r_right = rf[bi, (f - 1).clamp_min(0)], rf[bi, (l + 1).clamp(max=n - 1)]
    standing = (f > 0) & (l < n - 1) & (r_left > rf[bi, f] + stand) & (r_right > rf[bi, l] + stand)
    return ok & ((l - f + 1) >= min_beams) & (ext < max_ext) & standing


def scan_limit(env, action, v_meas):
    sim = env.sim
    lr = env.last_result
    B = action.shape[0]
    inf = torch.full((B,), float("inf"), device=action.device)
    if lr is None or lr.scan is None:
        return inf
    lid = env.cfg.lidar
    mx, my = float(lid.mount_x), float(lid.mount_y)
    r = lr.scan.float()
    n = r.shape[1]
    ang = torch.linspace(-0.5 * float(lid.fov), 0.5 * float(lid.fov), n, device=r.device)
    ok = torch.isfinite(r) & (r > 0.05) & (r < float(lid.range_max))
    if os.environ.get("TYPE_FILTER") and getattr(lr, "scan_type", None) is not None:
        from f1sim.lidar import HIT_CAR, HIT_GROUND
        ok = ok & (lr.scan_type != HIT_GROUND) & (lr.scan_type != HIT_CAR)
    rr = torch.where(ok, r, torch.zeros_like(r))
    px, py = mx + rr * torch.cos(ang)[None], my + rr * torch.sin(ang)[None]           # (B, n) car frame
    if os.environ.get("MAP_FILTER"):
        st0 = sim.state
        c0, s0 = torch.cos(st0[:, dyn.IYAW])[:, None], torch.sin(st0[:, dyn.IYAW])[:, None]
        wx, wy = st0[:, dyn.IX][:, None] + c0 * px - s0 * py, st0[:, dyn.IY][:, None] + s0 * px + c0 * py
        edt = sim.track.sample_edt(torch.stack([wx, wy], -1), sim.tid[:, None].expand(B, n))
        ok = ok & (edt > 0.15)
    if os.environ.get("CLUSTER_FILTER"):
        ok = cluster_keep(r, ok, px, py)
    reading = getattr(env, "opponent_reading", None)
    if reading is not None:
        ex, cx, cy, _cl, _vl = relative(reading, mx, my)
        near_car = ex[:, None] & (torch.hypot(px - cx[:, None], py - cy[:, None]) < CAR_EXCLUDE)
        ok = ok & ~near_car
    spec, vmax = env.tracker.spec, float(env.tracker.v_max)
    k, L, _ = mpc.decode_wp4(action.float(), v_meas.float(), vmax, torch.full((B,), 9.0, device=action.device), spec)
    pts, s = path_points(k, L)                                                       # (B, N_S, 2)
    d2 = (pts[:, :, None, 0] - px[:, None, :]) ** 2 + (pts[:, :, None, 1] - py[:, None, :]) ** 2   # (B, N_S, n)
    d2 = torch.where(ok[:, None, :], d2, torch.full_like(d2, 1e9))
    near = d2.min(2).values.sqrt() < THR                                             # (B, N_S)
    near = near & (s[None, :] <= L[:, None]) & (s[None, :] > FRONT)                  # the plan's own length, ahead of the bumper
    s_hit = torch.where(near, s[None, :].expand_as(near), torch.full_like(near, float("inf"), dtype=s.dtype)).min(1).values
    v = sim.state[:, dyn.IVX]
    room = (s_hit - FRONT - GMIN - v.clamp_min(0) * TAU).clamp_min(0.0)
    return torch.where(torch.isfinite(s_hit), torch.sqrt(2.0 * A_B * room), inf)


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    out = _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)
    env = CUR["env"]
    if env is not None and env.tracker is self:
        lim = scan_limit(env, action, v_meas)
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
props = [] if NOPROPS else ["--procedural-obstacles", "1.0", "--procedural-density", "1.5", "--procedural-max-props", "10",
                            "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off")]
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
                    *props, *race, "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        if "t_steps" in STAT:
            STAT["steps"] += int(STAT.pop("t_steps")); STAT["active"] += int(STAT.pop("t_active"))
            STAT["cut_sum"] += float(STAT.pop("t_cut"))
        print(f"seed {seed} {kind}: scan limit active {STAT['active']}/{STAT['steps']} learner steps "
              f"({100 * STAT['active'] / max(STAT['steps'], 1):.2f} %), mean cut {STAT['cut_sum'] / max(STAT['active'], 1):.2f} m/s",
              flush=True)
json.dump({"thr": THR, "a_b": A_B, "tau": TAU, "gmin": GMIN, "measure_only": MEASURE, "solo": SOLO, "noprops": NOPROPS,
           "osl": bool(osl), **STAT}, open(os.path.join(OUT, f"stats_{'_'.join(map(str, SEEDS))}.json"), "w"))
