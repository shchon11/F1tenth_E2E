"""Research (2026-10-25): the predictive speed shield from a MODULAR perception -- LiDAR detection + Kalman tracking
(perception.Perception, ForzaETH's detect + tracking) on the particle filter's pose (localization.ParticleFilter), the
stack the modular baselines race with (modular_stack.ModularStack). No ground truth: deployable.

The ego-frame predictive shield from the TRUE opponent state cut hard contacts 1.22 -> 0.62 /km (props off) and
2.18 -> 1.81 (props on) at -2 % race speed; from the policy's own nearest-car head only 1.22 -> 1.09 and 2.18 -> 2.15:
the head reads the opponent's lateral rate in the last 0.5 s before a hard contact at R2 0.1-0.2 (retrained: no
better). A tracker estimates (s, d, vs, vd) of each obstacle in the raceline's Frenet frame by filtering positions over
time. Here the shield reads, per learner car, its moving tracks (|vs| >= the tracker's static speed) against its own
estimated station, lateral offset and rates (pose from the filter, speed from odometry), in that frame:
  SOURCE=trk: the tracks and the estimated ego (what the car knows);
  SOURCE=true: the true opponent and ego states, the same frame and formula (the like-for-like oracle).
A track counts when it is wholly ahead (gap > 0.1 m, within 6 m) and its lateral offset to us is under W now or, at
the current lateral rate, within H s -- unless we would be a car length past it by then. Ceiling:
v_safe = v_opp + sqrt(2 a_b max(0, gap - g_min - (v_me - v_opp)+ tau)). The stack advances after each step on that
step's odometry and scan, so a command reads the perception of the scan before it. Finals race cells; RACE_CORRIDOR=on
keeps the props off the line. SHIELD_MEASURE_ONLY=1: count, do not cap.
usage: shield_trk_race.py CKPT OUT_DIR SEEDS SOURCE [H=1.0] [A_B=3.0] [W=0.35] [TAU=0.2] [GMIN=0.2]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc
from f1sim.learn import common
from f1sim.learn import evaluate as ev
from f1sim.modular_stack import ModularStack

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
SOURCE = sys.argv[4]
H = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
A_B = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
W = float(sys.argv[7]) if len(sys.argv) > 7 else 0.35
TAU = float(sys.argv[8]) if len(sys.argv) > 8 else 0.2
GMIN = float(sys.argv[9]) if len(sys.argv) > 9 else 0.2
assert SOURCE in ("trk", "true")
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
TRACK = "real:korea_2026_competition"
CAR_LEN = 0.5
MEASURE = bool(os.environ.get("SHIELD_MEASURE_ONLY"))
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None, "rls": None}
STACKS = {}
STAT = {"steps": 0, "active": 0, "cut_sum": 0.0}

_orig_make_env = common.make_env
_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step


def make_env(*a, **kw):
    CUR["rls"] = kw.get("rls")
    return _orig_make_env(*a, **kw)


def stack_of(env):
    st = STACKS.get(id(env))
    if st is None:
        ref = common.make_teacher(CUR["rls"], env, grip="nominal")
        rows = env.learner.nonzero().flatten()
        st = STACKS[id(env)] = ModularStack(env, rows, ref)
    return st


def ego_frenet(stack, xy, yaw, v, tid):
    """(s, d, ds/dt, dd/dt) of cars at xy / yaw / speed v in the raceline frame of `stack.base`"""
    per, b = stack.per, stack.base
    s, d, idx = per.frenet(xy, tid)
    t = b.tan[tid, idx]
    c, sn = torch.cos(yaw), torch.sin(yaw)
    sd = v * (c * t[:, 0] + sn * t[:, 1])
    dd = v * (-c * t[:, 1] + sn * t[:, 0])
    return s, d, sd, dd


def v_safe(env):
    stack = stack_of(env)
    sim = env.sim
    B = sim.state.shape[0]
    tid = sim.tid
    L = stack.base.length[tid][:, None]
    if SOURCE == "trk":
        est = stack.est_state
        s_e, d_e, sd_e, dd_e = ego_frenet(stack, est[:, :2], est[:, dyn.IYAW], est[:, dyn.IVX], tid)
        ok = stack.obs_ok & (stack.obs_vs.abs() >= stack.per.tp.static_speed)
        s_o, d_o, vs_o, vd_o = stack.obs_s, stack.obs_d, stack.obs_vs, stack.obs_vd
        v_me = est[:, dyn.IVX]
    else:
        st = sim.state
        v = torch.hypot(st[:, dyn.IVX], st[:, dyn.IVY])
        yaw = st[:, dyn.IYAW] + torch.atan2(st[:, dyn.IVY], st[:, dyn.IVX].clamp_min(1e-3))     # course over ground
        s_a, d_a, sd_a, dd_a = ego_frenet(stack, st[:, :2], yaw, v, tid)
        o = sim.other_idx                                                                     # (B, C)
        s_e, d_e, sd_e, dd_e = s_a, d_a, sd_a, dd_a
        s_o, d_o, vs_o, vd_o = s_a[o], d_a[o], sd_a[o], dd_a[o]
        ok = torch.ones_like(s_o, dtype=torch.bool)
        if getattr(sim, "absent_any", False):
            ok = ok & ~sim.absent[o]
        v_me = st[:, dyn.IVX]
    ds = torch.remainder(s_o - s_e[:, None] + 0.5 * L, L) - 0.5 * L
    gap = ds - CAR_LEN
    dlat, rate = d_o - d_e[:, None], vd_o - dd_e[:, None]
    v_opp = vs_o.clamp_min(0.0)
    room = (gap - GMIN - (v_me[:, None] - v_opp).clamp_min(0.0) * TAU).clamp_min(0.0)
    vs = v_opp + torch.sqrt(2.0 * A_B * room)
    in_lane = dlat.abs() < W
    t_in = torch.where(in_lane, torch.zeros_like(dlat),
                       torch.where((dlat * rate) < 0, (dlat.abs() - W) / rate.abs().clamp_min(1e-6), torch.full_like(dlat, 1e9)))
    past = (ds + (vs_o - sd_e[:, None]) * t_in.clamp(max=H)) < -(CAR_LEN + 0.1)
    hit = ok & (gap > 0.1) & (ds < 6.0) & (t_in <= H) & ~past
    return torch.where(hit, vs, torch.full_like(vs, 1e9)).min(1).values


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    out = _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)
    env = CUR["env"]
    if env is not None and env.tracker is self and env.sim.M > 1 and id(env) in STACKS:
        vs = v_safe(env)
        lr = env.learner
        cmd = out[:, 1]
        cut = lr & (cmd > vs)
        new = cmd if MEASURE else torch.where(cut, vs.to(cmd.dtype), cmd)
        if "t_steps" not in STAT:
            for k in ("t_steps", "t_active", "t_cut"):
                STAT[k] = torch.zeros((), device=cmd.device)
        STAT["t_steps"] += lr.sum(); STAT["t_active"] += cut.sum()
        STAT["t_cut"] += torch.where(cut, cmd - vs.to(cmd.dtype), torch.zeros_like(cmd)).sum()
        out = torch.stack([out[:, 0], new], 1)
    return out


def env_step(self, action):
    CUR["env"] = self
    out = _orig_env_step(self, action)
    if self.sim.M > 1 and CUR["rls"] is not None:
        stack_of(self).update(self)            # this step's odometry + scan -> the next command's perception
    return out


common.make_env = make_env
ev.common.make_env = make_env
mc.MAPPlanTracker.__call__ = map_call
gym_env.F1VecEnv.step = env_step
for seed in SEEDS:
    for kind in KINDS:
        out = os.path.join(OUT, f"race_s{seed}_{kind}.json")
        if os.path.exists(out):
            continue
        STACKS.clear()
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
        if "t_steps" in STAT:
            STAT["steps"] += int(STAT.pop("t_steps")); STAT["active"] += int(STAT.pop("t_active"))
            STAT["cut_sum"] += float(STAT.pop("t_cut"))
        print(f"seed {seed} {kind}: shield active {STAT['active']}/{STAT['steps']} learner steps "
              f"({100 * STAT['active'] / max(STAT['steps'], 1):.2f} %), mean cut {STAT['cut_sum'] / max(STAT['active'], 1):.2f} m/s",
              flush=True)
json.dump({"source": SOURCE, "h": H, "a_b": A_B, "w": W, "tau": TAU, "gmin": GMIN, "measure_only": MEASURE,
           "corridor": os.environ.get("RACE_CORRIDOR", "off"), **STAT},
          open(os.path.join(OUT, f"shield_stats_{'_'.join(map(str, SEEDS))}.json"), "w"))
