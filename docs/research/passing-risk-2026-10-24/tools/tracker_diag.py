"""Research (2026-10-25): how good is the modular perception's picture of the race car ahead?

The predictive shield from the modular stack's tracks (shield_trk_race.py: ABD detection + constant-velocity Kalman
tracks in the raceline frame, on the particle filter's pose) made races WORSE (props off, seeds 83-84: hard contacts
1.22 -> 1.64 /km, caps on 3.6 % of steps vs 2.1 % from the true state), while the same shield from the true state cut
them to 0.62 and from the policy's retrained nearest-car head to 0.96. The modular baselines race on this perception,
so its quality also bears on how fairly they are compared. Per learner step (no cap; the policy drives) this records:
  * localisation: the particle filter's position error;
  * the true opponent, when it is ahead within 6 m: whether a confirmed track lies within 0.7 m of it, and that track's
    errors in s, d, vs (along-track speed) and vd (lateral rate) against the truth in the same raceline frame;
  * confirmed MOVING tracks (|vs| >= 0.5) farther than 0.7 m from the opponent (ghosts / props read as movers), and
    how many of those sit within 0.6 m laterally of our raceline offset ahead (the ones a shield would brake for).
usage: tracker_diag.py CKPT OUT.json SEEDS [KINDS=forzaeth,forzaeth_pred,lane_switch,interactive]"""
import json
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env
from f1sim.learn import common
from f1sim.learn import evaluate as ev
from f1sim.modular_stack import ModularStack

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
KINDS = (sys.argv[4] if len(sys.argv) > 4 else "forzaeth,forzaeth_pred,lane_switch,interactive").split(",")
TRACK = "real:korea_2026_competition"
CUR = {"rls": None}
STACKS = {}
ACC = {}
_orig_make_env = common.make_env
_orig_env_step = gym_env.F1VecEnv.step


def make_env(*a, **kw):
    CUR["rls"] = kw.get("rls")
    return _orig_make_env(*a, **kw)


def add(key, t):
    """accumulate count, sum and sum of squares (and abs) of a 1-D tensor, on the device"""
    t = t.float()
    a = ACC.setdefault(key, torch.zeros(4, device=t.device))
    a += torch.stack([torch.tensor(float(t.numel()), device=t.device), t.sum(), (t * t).sum(), t.abs().sum()])


def frenet_state(stack, xy, yaw, v, tid):
    per, b = stack.per, stack.base
    s, d, idx = per.frenet(xy, tid)
    t = b.tan[tid, idx]
    c, sn = torch.cos(yaw), torch.sin(yaw)
    return s, d, v * (c * t[:, 0] + sn * t[:, 1]), v * (-c * t[:, 1] + sn * t[:, 0])


def observe(env, stack):
    sim = env.sim
    st = sim.state
    rows = stack.rows
    tid = sim.tid
    add("loc_err_m", stack.localisation_error(env))
    v = torch.hypot(st[:, dyn.IVX], st[:, dyn.IVY])
    yaw = st[:, dyn.IYAW] + torch.atan2(st[:, dyn.IVY], st[:, dyn.IVX].clamp_min(1e-3))
    s, d, sd, dd = frenet_state(stack, st[:, :2], yaw, v, tid)
    o = sim.other_idx[:, 0]
    L = stack.base.length[tid]
    ds_true = torch.remainder(s[o] - s + 0.5 * L, L) - 0.5 * L
    ahead = (ds_true > 0) & (ds_true < 6.0)
    ok = stack.obs_ok
    moving = ok & (stack.obs_vs.abs() >= stack.per.tp.static_speed)
    # distance of each track to the true opponent, in the raceline frame (s wrapped)
    dts = torch.remainder(stack.obs_s - s[o][:, None] + 0.5 * L[:, None], L[:, None]) - 0.5 * L[:, None]
    dist = torch.hypot(dts, stack.obs_d - d[o][:, None])
    dist = torch.where(ok, dist, torch.full_like(dist, 1e9))
    near, j = dist.min(1)
    match = near < 0.7
    r_ah = torch.zeros_like(ahead); r_ah[rows] = True
    sel = r_ah & ahead
    add("opp_ahead_steps", sel[sel])
    add("opp_ahead_tracked", (sel & match)[sel])
    m = sel & match
    ar = torch.arange(st.shape[0], device=st.device)
    if bool(m.any()):
        e_s = stack.obs_s[ar, j] - s[o]
        add("err_s", (torch.remainder(e_s + 0.5 * L, L) - 0.5 * L)[m])
        add("err_d", (stack.obs_d[ar, j] - d[o])[m])
        add("err_vs", (stack.obs_vs[ar, j] - sd[o])[m])
        add("err_vd", (stack.obs_vd[ar, j] - dd[o])[m])
        add("true_vd_abs", dd[o][m])
    ghost = moving & (dist >= 0.7)
    gs = torch.remainder(stack.obs_s - s[:, None] + 0.5 * L[:, None], L[:, None]) - 0.5 * L[:, None]
    in_path = ghost & (gs > 0) & (gs < 6.0) & ((stack.obs_d - d[:, None]).abs() < 0.6)
    add("ghost_moving_tracks_per_step", ghost.sum(1).float()[rows])
    add("ghost_in_path_per_step", in_path.sum(1).float()[rows])


def env_step(self, action):
    out = _orig_env_step(self, action)
    if self.sim.M > 1 and CUR["rls"] is not None:
        st = STACKS.get(id(self))
        if st is None:
            st = STACKS[id(self)] = ModularStack(self, self.learner.nonzero().flatten(), common.make_teacher(CUR["rls"], self, grip="nominal"))
        st.update(self)
        observe(self, st)
    return out


common.make_env = make_env
gym_env.F1VecEnv.step = env_step
for seed in SEEDS:
    for kind in KINDS:
        STACKS.clear()
        out = os.path.join(os.path.dirname(OUT) or ".", f"_td_{os.path.basename(OUT)}_{seed}_{kind}.json")
        slot = {"kind": kind, "speed_scale": [0.7, 1.0]}
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", "32", "--steps", "1200",
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("RACE_CORRIDOR", "off"),
                    "--opp-slots", json.dumps([slot]), "--output", out, "--dial-value", "1.0489",
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        ev.main()
        res = {}
        for k, a in ACC.items():
            n, s1, s2, sa = (float(x) for x in a.cpu())
            res[k] = {"n": n, "mean": s1 / max(n, 1), "rms": (s2 / max(n, 1)) ** 0.5, "mean_abs": sa / max(n, 1)}
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"seed {seed} {kind}: " + " | ".join(f"{k} {v['mean']:+.3f} (rms {v['rms']:.3f}, n {int(v['n'])})" for k, v in res.items()), flush=True)
