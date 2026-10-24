"""Research (2026-10-24): can the policy's OWN perception drive the predictive speed shield -- a deployable safety layer?

The predictive oracle shield (shield_pred_race.py: cap our speed for a car wholly ahead that is in our lane now or drifts
into it within 1 s) cut hard car contacts 1.05 -> 0.59 /km (props on the line) and 1.00 -> 0.34 (props off), from TRUE
states. The v2 actor carries a nearest-visible-car head (aux_v2.NearestCarHead: [exists logit, dx/5, dy/2, closing/5,
v_lat/2] in the LiDAR frame, trained as an auxiliary); for a car ahead it reads dx R2 0.78, dy 0.92, closing 0.72,
v_lat 0.86 (probe_v2entry). Here the same shield runs in the ego frame from one of two sources:
  SOURCE=true: the opponent's true relative state (env._opp_rel), any car present -- the oracle in this frame;
  SOURCE=head: the head's reading of the policy's own features at this step -- what the car itself perceives.
Ceiling: gap = dx - car length, v_opp = v_me - closing, v_safe = v_opp + sqrt(2 a_b max(0, gap - g_min - closing+ tau));
a car counts when it exists, is wholly ahead (gap > 0.1 m, dx < 6 m) and |dy| < W now or within H s at the current
lateral rate -- unless we would be a car length past it by then. Finals race cells (seeds, 4 kinds; props on the line,
RACE_CORRIDOR=on keeps them off). SHIELD_MEASURE_ONLY=1: count, do not cap.
SOLO=1 (SOURCE=head): the selection's solo protocol instead (64 cars, seed 123, 3 laps, no props, nobody else on the
track) with the shield live -- a deployed shield runs whether or not a car is there, so any reading of a car where there
is none (walls?) costs lap time; the stats say how often it capped.
SHIELD_DEADBAND=x [m/s]: cap only to v_safe + x (a command within x of the ceiling passes) -- 30 % of the head's caps are
false alarms (shield_dual_diag.py), split over a phantom car (15 %), a misjudged gap / closing speed (35 %), a misjudged
lateral offset (13 %) and lateral rate (37 %), and they precede 8 of 12 hard wall hits; small reading errors flip a
command just over the ceiling.
usage: shield_head_race.py CKPT OUT_DIR SEEDS SOURCE [H=1.0] [A_B=3.0] [W=0.35] [TAU=0.2] [GMIN=0.2]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, map_controller as mc
from f1sim.learn import evaluate as ev
from f1sim.learn import model_v2

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
SOURCE = sys.argv[4]
H = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
A_B = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
W = float(sys.argv[7]) if len(sys.argv) > 7 else 0.35
TAU = float(sys.argv[8]) if len(sys.argv) > 8 else 0.2
GMIN = float(sys.argv[9]) if len(sys.argv) > 9 else 0.2
assert SOURCE in ("true", "head")
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
TRACK = "real:korea_2026_competition"
CAR_LEN = 0.5
MEASURE = bool(os.environ.get("SHIELD_MEASURE_ONLY"))
DEADBAND = float(os.environ.get("SHIELD_DEADBAND", "0") or 0)
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None, "near": None}
STAT = {"steps": 0, "active": 0, "cut_sum": 0.0}

_orig_features = model_v2.ActorV2.features
_orig_map_call = mc.MAPPlanTracker.__call__
_orig_env_step = gym_env.F1VecEnv.step


def features(self, e, y):
    z = _orig_features(self, e, y)
    if self.near_car is not None:
        CUR["near"] = self.near_car(z).detach()          # the policy asks before env.step: this step's reading
    return z


def relative(env):
    """(exists, dx, dy [CG, ego frame], closing, v_lat) of the nearest opponent, from SOURCE"""
    st = env.sim.state
    mx, my, _ = env._lidar_pose()
    if SOURCE == "true":
        dx, dy, _dyaw, dvx, dvy, dist, present = env._opp_rel(st)
        dx, dy, dvx, dvy, present = dx[:, 0], dy[:, 0], dvx[:, 0], dvy[:, 0], present[:, 0]
        lx, ly = dx - mx, dy - my
        closing = -(lx * dvx + ly * dvy) / torch.hypot(lx, ly).clamp_min(1e-6)
        return present, dx, dy, closing, dvy
    near = CUR["near"]
    B = st.shape[0]
    if near is None:
        z = torch.zeros(B, device=st.device)
        return z.bool(), z, z, z, z
    if near.shape[0] != B:                                    # the actor ran on the learner rows only
        full = torch.zeros(B, near.shape[1], device=near.device, dtype=near.dtype)
        full[env.learner] = near
        near = full
    near = near.float()
    return near[:, 0] > 0, near[:, 1] * 5.0 + mx, near[:, 2] * 2.0 + my, near[:, 3] * 5.0, near[:, 4] * 2.0


def v_safe(env):
    ex, dx, dy, closing, vlat = relative(env)
    v_me = env.sim.state[:, dyn.IVX]
    gap = dx - CAR_LEN
    v_opp = (v_me - closing).clamp_min(0.0)
    room = (gap - GMIN - closing.clamp_min(0.0) * TAU).clamp_min(0.0)
    vs = v_opp + torch.sqrt(2.0 * A_B * room)
    in_lane = dy.abs() < W
    t_in = torch.where(in_lane, torch.zeros_like(dy),
                       torch.where((dy * vlat) < 0, (dy.abs() - W) / vlat.abs().clamp_min(1e-6), torch.full_like(dy, 1e9)))
    past = (dx - closing * t_in.clamp(max=H)) < -(CAR_LEN + 0.1)
    hit = ex & (gap > 0.1) & (dx < 6.0) & (t_in <= H) & ~past
    return torch.where(hit, vs, torch.full_like(vs, 1e9))


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    out = _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)
    env = CUR["env"]
    if env is not None and env.tracker is self and (env.sim.M > 1 or SOURCE == "head"):
        vs = v_safe(env) + DEADBAND
        lr = env.learner
        cmd = out[:, 1]
        cut = lr & (cmd > vs)
        new = cmd if MEASURE else torch.where(cut, vs.to(cmd.dtype), cmd)
        if "t_steps" not in STAT:
            for k in ("t_steps", "t_active", "t_cut"):
                STAT[k] = torch.zeros((), device=cmd.device)
        STAT["t_steps"] += lr.sum(); STAT["t_active"] += cut.sum()
        if SOURCE == "head":                                      # is the head read at all, and does it see a car
            nr = CUR["near"]
            if "t_read" not in STAT:
                STAT["t_read"] = torch.zeros((), device=cmd.device); STAT["t_seen"] = torch.zeros((), device=cmd.device)
            if nr is not None:
                STAT["t_read"] += float(nr.shape[0]) * 0 + nr.shape[0]
                STAT["t_seen"] += (nr[:, 0] > 0).sum()
        STAT["t_cut"] += torch.where(cut, cmd - vs.to(cmd.dtype), torch.zeros_like(cmd)).sum()
        out = torch.stack([out[:, 0], new], 1)
    return out


def env_step(self, action):
    CUR["env"] = self
    return _orig_env_step(self, action)


model_v2.ActorV2.features = features
mc.MAPPlanTracker.__call__ = map_call
gym_env.F1VecEnv.step = env_step
if os.environ.get("SOLO"):
    from f1sim.params import Config
    cfg = Config()
    cfg.rand.ranges["vehicle.mu"] = (1.0, 1.0); cfg.rand.ranges["vehicle.mu_f_scale"] = (0.92, 0.92)
    r = ev.evaluate(CK, [TRACK], 64, 2400, 8.0, "cuda", seed=123, cfg=cfg, protocol="trials", budget_laps=3.0,
                    dial_value=1.0489)
    json.dump(r, open(os.path.join(OUT, "solo.json"), "w"), default=str)
    STAT["steps"] = int(STAT.get("t_steps", 0)); STAT["active"] = int(STAT.get("t_active", 0))
    STAT["cut_sum"] = float(STAT.get("t_cut", 0.0))
    STAT["head_rows_read"] = int(STAT.get("t_read", 0)); STAT["head_rows_seeing_a_car"] = int(STAT.get("t_seen", 0))
    for k in ("t_steps", "t_active", "t_cut", "t_read", "t_seen"):
        STAT.pop(k, None)
    print(f"solo: lap {r.get('lap_time_s')} s, completions {r.get('completions')}, shield active {STAT['active']}/{STAT['steps']}"
          f" ({100 * STAT['active'] / max(STAT['steps'], 1):.3f} %), mean cut {STAT['cut_sum'] / max(STAT['active'], 1):.2f}; head read on "
          f"{STAT['head_rows_read']} rows, a car seen on {STAT['head_rows_seeing_a_car']}", flush=True)
    json.dump({"source": SOURCE, "h": H, "solo": True, **STAT}, open(os.path.join(OUT, "shield_stats_solo.json"), "w"))
    sys.exit(0)
for seed in SEEDS:
    for kind in KINDS:
        out = os.path.join(OUT, f"race_s{seed}_{kind}.json")
        if os.path.exists(out):
            continue
        CUR["near"] = None
        slot = {"kind": kind, "speed_scale": [0.7, 1.0]}
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off"),
                    "--opp-slots", json.dumps([slot]),
                    "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        if "t_steps" in STAT:
            STAT["steps"] += int(STAT.pop("t_steps")); STAT["active"] += int(STAT.pop("t_active"))
            STAT["cut_sum"] += float(STAT.pop("t_cut"))
        if "t_read" in STAT:                                   # the head counters (a race fails at the json dump without this)
            STAT["head_rows_read"] = STAT.get("head_rows_read", 0) + int(STAT.pop("t_read"))
            STAT["head_rows_seeing_a_car"] = STAT.get("head_rows_seeing_a_car", 0) + int(STAT.pop("t_seen"))
        print(f"seed {seed} {kind}: shield active {STAT['active']}/{STAT['steps']} learner steps "
              f"({100 * STAT['active'] / max(STAT['steps'], 1):.2f} %), mean cut {STAT['cut_sum'] / max(STAT['active'], 1):.2f} m/s",
              flush=True)
json.dump({"source": SOURCE, "h": H, "a_b": A_B, "w": W, "tau": TAU, "gmin": GMIN, "measure_only": MEASURE,
           "deadband": DEADBAND, "corridor": os.environ.get("RACE_CORRIDOR", "off"), **STAT},
          open(os.path.join(OUT, f"shield_stats_{'_'.join(map(str, SEEDS))}.json"), "w"))
