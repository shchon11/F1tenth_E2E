"""Research (2026-10-25): would a LATERAL reaction stop the car contacts the speed limit leaves? Upper bound, true states.

With the speed limit for a car ahead on, 74 % of the hard car contacts left have our own car moving toward the other in
their last 0.5 s, mostly beside it -- where a speed limit cannot act (held to its speed beside it, a car stays there:
section 1). The rule here: while another car overlaps us lengthwise (|dx| < OVERLAP, centre to centre) and sits within
NEAR m sideways, a steering command that turns TOWARD it is scaled by KEEP (0 = no turning toward it at all, 1 = off).
The command turning away, or any command with no car beside, passes untouched. True relative state (oracle). Runs with the
speed limit for a car ahead on (OSL=1, the shipped evaluate flag). Finals race cells; RACE_CORRIDOR=on keeps props off.
GUARD_SOURCE=head: where the other car is comes from the policy's own nearest-car output instead (env.opponent_reading,
which the speed limit already fills every step) -- the deployable form. Read: worse (total 1.00 -> 1.24 /km): that output
reports cars the LiDAR sees, and a car beside us is often not seen. GUARD_SOURCE=beside: a dedicated 'car beside us' reading
(beside_head_train.py, BESIDE_HEAD=path) on the policy's features, trained on the true state of any visibility.
With an estimated source, every activation is also sorted by the TRUE state: a car really beside us on that side, a live
prop beside us on that side (the reading took a prop for a car), on the other side, or nothing -- where do the extra prop
hits of the estimated guards (props off the line: 0.17 -> 0.37-0.41 /km, true-state guard 0.20) come from?
Read 10-25 (beside reading, 81-82): 70 % of its activations have a car really beside on that side, ~26 % a car near but
just outside the window, a prop almost never -- so the extra prop hits are not props taken for cars. BESIDE_LOGIT: the
reading's existence threshold (validation: logit > 2 is right 86 % instead of 75 %, recall 62 % instead of 80 %).
usage: lateral_guard_race.py CKPT OUT_DIR SEEDS [KEEP=0.5] [NEAR=0.6] [OVERLAP=0.7]"""
import json
import os
import sys

import torch

from f1sim import gym_env, map_controller as mc
from f1sim.learn import evaluate as ev

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
KEEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
NEAR = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
OVERLAP = float(sys.argv[6]) if len(sys.argv) > 6 else 0.7
KINDS = ("forzaeth", "forzaeth_pred", "lane_switch", "interactive")
TRACK = "real:korea_2026_competition"
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None}
STAT = {"steps": 0, "active": 0, "steer_cut": 0.0}
_orig_map_call = mc.MAPPlanTracker.__call__
BESIDE = {"net": None, "out": None}
if os.environ.get("GUARD_SOURCE") == "beside":
    import torch.nn as nn
    from f1sim.learn import model_v2
    _bh = torch.load(os.environ["BESIDE_HEAD"], map_location="cpu")
    BESIDE["net"] = nn.Sequential(nn.Linear(_bh["in_dim"], _bh["hidden"]), nn.GELU(), nn.Linear(_bh["hidden"], 2))
    BESIDE["net"].load_state_dict(_bh["state_dict"]); BESIDE["net"].eval()
    _orig_features = model_v2.ActorV2.features

    def _features(self, e, y):
        z = _orig_features(self, e, y)
        net = BESIDE["net"]
        if net is not None:
            if next(net.parameters()).device != z.device:
                net.to(z.device)
            BESIDE["out"] = net(z.float()).detach()
        return z
    model_v2.ActorV2.features = _features
_orig_env_step = gym_env.F1VecEnv.step


def map_call(self, action, v_meas, speed_cap, yaw_rate=None, delay=None):
    out = _orig_map_call(self, action, v_meas, speed_cap, yaw_rate, delay)
    env = CUR["env"]
    if env is not None and env.tracker is self and env.sim.M > 1:
        if os.environ.get("GUARD_SOURCE") == "beside" and BESIDE["out"] is not None:
            o = BESIDE["out"]
            if o.shape[0] != env.sim.state.shape[0]:
                full = torch.zeros(env.sim.state.shape[0], 2, device=o.device); full[:, 0] = -10.0
                full[env.learner] = o; o = full
            present, dy = o[:, 0] > float(os.environ.get("BESIDE_LOGIT", "0")), o[:, 1]
            dx = torch.zeros_like(dy)                      # the reading already means 'overlapping lengthwise'
        elif os.environ.get("GUARD_SOURCE") == "head" and getattr(env, "opponent_reading", None) is not None:
            from f1sim.opponent_speed_limit import relative
            lid = env.cfg.lidar
            present, dx, dy, _cl, _vl = relative(env.opponent_reading, float(lid.mount_x), float(lid.mount_y))
        else:
            dx, dy, _dyaw, _dvx, _dvy, _dist, present = env._opp_rel(env.sim.state)
            dx, dy, present = dx[:, 0], dy[:, 0], present[:, 0]
        beside = present & (dx.abs() < OVERLAP) & (dy.abs() < NEAR)
        steer = out[:, 0]
        toward = beside & (torch.sign(steer) == torch.sign(dy)) & (steer.abs() > 1e-4)
        lr = env.learner
        cut = lr & toward
        new = torch.where(cut, steer * KEEP, steer)
        if "t_steps" not in STAT:
            for k in ("t_steps", "t_active", "t_cut", "t_true", "t_prop_same", "t_prop_other", "t_none"):
                STAT[k] = torch.zeros((), device=steer.device)
        STAT["t_steps"] += lr.sum(); STAT["t_active"] += cut.sum(); STAT["t_cut"] += (steer - new).abs().sum()
        if os.environ.get("GUARD_SOURCE") in ("head", "beside"):
            tdx, tdy, _a, _b, _c, _d, tpr = env._opp_rel(env.sim.state)
            tdx, tdy, tpr = tdx[:, 0], tdy[:, 0], tpr[:, 0]
            side = torch.sign(dy)
            true_b = tpr & (tdx.abs() < OVERLAP) & (tdy.abs() < NEAR) & (torch.sign(tdy) == side)
            ep = env.sim.track.env_props
            if ep is not None:
                poses, _n, _d2, zlo, zhi = ep.slots(env.sim.eid)
                st_ = env.sim.state
                c_, s_ = torch.cos(st_[:, 2])[:, None], torch.sin(st_[:, 2])[:, None]
                qx, qy = poses[..., 0] - st_[:, 0][:, None], poses[..., 1] - st_[:, 1][:, None]
                xl, yl = c_ * qx + s_ * qy, -s_ * qx + c_ * qy
                near_p = (zhi > zlo) & (xl.abs() < OVERLAP) & (yl.abs() < NEAR + 0.2)
                p_same = (near_p & (torch.sign(yl) == side[:, None])).any(1)
                p_other = (near_p & (torch.sign(yl) != side[:, None])).any(1)
            else:
                p_same = p_other = torch.zeros_like(cut)
            f = cut & ~true_b
            STAT["t_true"] += (cut & true_b).sum(); STAT["t_prop_same"] += (f & p_same).sum()
            STAT["t_prop_other"] += (f & ~p_same & p_other).sum(); STAT["t_none"] += (f & ~p_same & ~p_other).sum()
        out = torch.stack([new, out[:, 1]], 1)
    return out


def env_step(self, action):
    CUR["env"] = self
    return _orig_env_step(self, action)


mc.MAPPlanTracker.__call__ = map_call
gym_env.F1VecEnv.step = env_step
for seed in SEEDS:
    for kind in KINDS:
        out = os.path.join(OUT, f"race_s{seed}_{kind}.json")
        if os.path.exists(out):
            continue
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "2400",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off"),
                    "--opp-slots", json.dumps([{"kind": kind, "speed_scale": [0.7, 1.0]}]), "--output", out,
                    "--dial-value", "1.0489", "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92",
                    *(["--opponent-speed-limit"] if os.environ.get("OSL") else [])]
        ev.main()
        if "t_steps" in STAT:
            STAT["steps"] += int(STAT.pop("t_steps")); STAT["active"] += int(STAT.pop("t_active"))
            STAT["steer_cut"] += float(STAT.pop("t_cut"))
            for k in ("true", "prop_same", "prop_other", "none"):
                STAT[k] = STAT.get(k, 0) + int(STAT.pop("t_" + k))
            if os.environ.get("GUARD_SOURCE") in ("head", "beside"):
                a = max(STAT["active"], 1)
                print(f"  activations by the true state: a car really beside on that side {100 * STAT['true'] / a:.1f} %, "
                      f"a prop beside on that side {100 * STAT['prop_same'] / a:.1f} %, a prop on the other side "
                      f"{100 * STAT['prop_other'] / a:.1f} %, nothing {100 * STAT['none'] / a:.1f} %", flush=True)
        print(f"seed {seed} {kind}: lateral guard active {STAT['active']}/{STAT['steps']} learner steps "
              f"({100 * STAT['active'] / max(STAT['steps'], 1):.2f} %), mean steering removed "
              f"{STAT['steer_cut'] / max(STAT['active'], 1):.3f} rad", flush=True)
json.dump({"keep": KEEP, "near": NEAR, "overlap": OVERLAP, "osl": bool(os.environ.get("OSL")),
           "source": os.environ.get("GUARD_SOURCE", "true"), **STAT},
          open(os.path.join(OUT, f"stats_{'_'.join(map(str, SEEDS))}.json"), "w"))
