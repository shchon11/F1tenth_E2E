"""Research (2026-10-26): which of the obstacle limit's holds were needed?

The obstacle limit (f1sim/obstacle_speed_limit.py) costs 4-5 % race speed for its -24..-39 % hard contacts. It holds the speed
for the first kept cluster on the plan's whole path (up to the plan's length), but the plan is honest only to ~2 m
(docs/research/plan-authority-honesty-2026-10-23): a box 3 m ahead on today's plan may be one the policy steers round anyway.
MEASURE-ONLY: the env runs with the limit on, but the limit is patched to record what it would do and return +inf, so the car
drives as without it. Every onset of a would-hold (the limit under the command the follower asked for) is followed for 1.5 s:
how close did the car's footprint come to the return that triggered it (world point), and did a hard (>= 1.5 m/s) prop contact
follow near it?
  hit      -- a hard prop contact in the window with the triggering point within 0.4 m of the footprint;
  graze    -- a softer prop contact (< 1.5 m/s) there, no hard one (the user: grazes are fine, hard hits are not);
  close    -- the footprint came within 0.10 m of the point without a prop contact there;
  near     -- within 0.10-0.30 m;
  clear    -- never within 0.30 m: the hold was not needed for this obstacle.
By distance along the plan to the hit point (s_hit) and by speed. Also the RECALL side: the share of all hard prop contacts that
had a would-hold onset in the 1.5 s before them.
Same cells as the finals race cells with props on the line; OSL=1 also switches the car limit on (active).
usage: hold_anatomy.py CKPT OUT_DIR [SEEDS=81,82] [KINDS=forzaeth,forzaeth_pred,lane_switch,interactive]
env ANAT_DEVICE, ANAT_ENVS, ANAT_STEPS for smoke tests."""
import json
import math
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env
from f1sim import obstacle_speed_limit as osl
from f1sim.learn import evaluate as ev
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "81,82").split(",")]
KINDS = (sys.argv[4] if len(sys.argv) > 4 else "forzaeth,forzaeth_pred,lane_switch,interactive").split(",")
TRACK = os.environ.get("ANAT_TRACK", "real:korea_2026_competition")
HARD, W = 1.5, 60                         # [m/s], window [steps] = 1.5 s at 40 Hz
os.makedirs(OUT, exist_ok=True)
EVENTS, CONTACTS, LABEL = [], [], {}
CUR = {"env": None, "in_step": False}
REAL = osl.speed_limit


def limit_and_point(ranges, angles, k, L, v_me, mount_x=0.297, mount_y=0.0, car_reading=None, range_max=30.0, p=osl.DEFAULT):
    """`osl.speed_limit`, plus the car-frame point of the return that set it (nan where none)."""
    r = ranges.float()
    ang = angles.float()
    if ang.dim() == 1:
        ang = ang[None].expand_as(r)
    ok = torch.isfinite(r) & (r > 0.05) & (r < range_max - 1e-3)
    valid = ok.clone()
    rr = torch.where(ok, r, torch.zeros_like(r))
    px, py = mount_x + rr * torch.cos(ang), mount_y + rr * torch.sin(ang)
    ok = osl.cluster_keep(r, ok, px, py, p)
    # each beam's segment ends (as cluster_keep segments the scan), to say whether a kept cluster was cut out by no-return beams
    B, n = r.shape
    rf = torch.where(valid, r, torch.full_like(r, 30.0))
    brk = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=r.device),
                     ((rf[:, 1:] - rf[:, :-1]).abs() > p.break_jump) | ~valid[:, 1:] | ~valid[:, :-1]], 1)
    seg = torch.cumsum(brk.long(), 1)
    gid = (seg + torch.arange(B, device=r.device)[:, None] * (n + 2)).flatten()
    idx = torch.arange(n, device=r.device)[None].expand(B, n).flatten()
    f_ = torch.full((B * (n + 2),), n, dtype=torch.long, device=r.device).scatter_reduce(0, gid, idx, "amin", include_self=True)[gid].view(B, n)
    l_ = torch.full((B * (n + 2),), -1, dtype=torch.long, device=r.device).scatter_reduce(0, gid, idx, "amax", include_self=True)[gid].view(B, n)
    if car_reading is not None:
        from f1sim.opponent_speed_limit import relative
        ex, cx, cy, _cl, _vl = relative(car_reading.to(r.device), mount_x, mount_y)
        ok = ok & ~(ex[:, None] & (torch.hypot(px - cx[:, None], py - cy[:, None]) < p.car_exclude))
    pts, s = osl.path_points(k.float(), L.float(), p)
    d2 = (pts[:, :, None, 0] - px[:, None, :]) ** 2 + (pts[:, :, None, 1] - py[:, None, :]) ** 2
    d2 = torch.where(ok[:, None, :], d2, torch.full_like(d2, 1e9))
    dmin, bi = d2.min(2)
    near = (dmin.sqrt() < p.corridor) & (s[None, :] <= L.float()[:, None]) & (s[None, :] > p.front)
    sh = torch.where(near, s[None, :].expand_as(near), torch.full_like(near, float("inf"), dtype=s.dtype))
    s_hit, si = sh.min(1)
    room = (s_hit - p.front - p.gap_min - v_me.float().clamp_min(0.0) * p.lag).clamp_min(0.0)
    lim = torch.where(torch.isfinite(s_hit), torch.sqrt(2.0 * p.brake * room), torch.full_like(s_hit, float("inf")))
    b = bi.gather(1, si[:, None]).squeeze(1)
    rows = torch.arange(b.shape[0], device=b.device)
    hx = torch.where(torch.isfinite(s_hit), px[rows, b], torch.full_like(s_hit, float("nan")))
    hy = torch.where(torch.isfinite(s_hit), py[rows, b], torch.full_like(s_hit, float("nan")))
    fb, lb = f_[rows, b], l_[rows, b]
    edge_inv = ~valid[rows, (fb - 1).clamp_min(0)] | ~valid[rows, (lb + 1).clamp(max=n - 1)]
    CUR["edge_invalid"] = edge_inv
    return lim, s_hit, hx, hy


def measuring_limit(*a, **kw):
    """the env's call: record the would-be limit and its point, drive as if there were none"""
    lim, s_hit, hx, hy = limit_and_point(*a, **kw)
    CUR.update(lim=lim, s_hit=s_hit, hx=hx, hy=hy)
    return torch.full_like(lim, float("inf"))


class Tracker:
    def __init__(self, env):
        sim = env.sim
        B = int(sim.state.shape[0])
        dev = sim.device
        self.prev = torch.zeros(B, dtype=torch.bool, device=dev)
        self.open = []                                      # events being followed
        self.recent = [[] for _ in range(B)]                # (step, s_hit) of the last onsets per row
        self.active = [None] * B                            # the event whose hold is still on, per row
        self.dt = float(sim.control_dt)
        self.k = 0
        vp = env.cfg.vehicle
        self.hl, self.hw = float(vp.length) / 2, float(vp.width) / 2

    def foot_dist(self, st, wx, wy):
        """distance from the world point to the car's rectangle footprint"""
        c, s_ = math.cos(st[2]), math.sin(st[2])
        dx, dy = wx - st[0], wy - st[1]
        lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
        ox, oy = max(abs(lx) - self.hl, 0.0), max(abs(ly) - self.hw, 0.0)
        return math.hypot(ox, oy)

    def observe(self, env, sim, r, entry):
        self.k += 1
        lim = CUR.get("lim")
        fresh = entry == 0
        cmd = env.last_cmd_raw[:, 1].float()
        if lim is None:
            return
        would = env.learner & (lim.to(cmd.dtype) < cmd)
        onset = would & ~(self.prev & ~fresh)
        self.prev = would.clone()
        st_all = r.state[:, [dyn.IX, dyn.IY, dyn.IYAW]].float()
        on, imp = getattr(r, "contact_onsets", None), getattr(r, "contact_impacts", None)
        hard = (on[:, 1] & env.learner & (imp[:, 1].float() >= HARD)) if (on is not None and imp is not None) else None
        st_cpu = st_all.cpu()
        hard_rows = [] if hard is None else hard.nonzero().flatten().tolist()
        anyp = (on[:, 1] & env.learner) if on is not None else None
        soft_rows = {} if anyp is None else {i: float(imp[i, 1]) for i in anyp.nonzero().flatten().tolist()}
        # the holds still on: how long, and how much speed they would cut
        would_l = would.tolist()
        cut = (cmd - lim.to(cmd.dtype)).clamp_min(0.0).tolist()
        for i, e in enumerate(self.active):
            if e is None:
                continue
            if would_l[i] and not bool(fresh[i]):
                e["dur"] += 1
                e["cut_sum"] += cut[i] * self.dt
            else:
                self.active[i] = None
        for i in hard_rows:
            # recall: did this row have a would-hold onset in the last 1.5 s?
            had = [s for (k0, s) in self.recent[i] if self.k - k0 <= W]
            CONTACTS.append(dict(LABEL, row=i, impact=round(float(imp[i, 1]), 3), held_before=bool(had),
                                 s_hit_before=(round(min(had), 2) if had else None)))
        # follow the open events
        still = []
        for e in self.open:
            i = e["row"]
            if bool(fresh[i]):
                e["reset"] = True
                EVENTS.append(e)
                continue
            d = self.foot_dist(st_cpu[i].tolist(), e["wx"], e["wy"])
            e["min_d"] = min(e["min_d"], d)
            if i in hard_rows and d < 0.4:
                e["hit"] = True
            if i in soft_rows and d < 0.4:
                e["touch"] = max(e.get("touch", 0.0), soft_rows[i])
            e["age"] += 1
            if e["age"] >= W:
                EVENTS.append(e)
            else:
                still.append(e)
        self.open = still
        if bool(onset.any()):
            hx, hy, s_hit = CUR["hx"], CUR["hy"], CUR["s_hit"]
            po = env.procedural
            for i in onset.nonzero().flatten().tolist():
                x, y, yaw = st_cpu[i].tolist()
                c, s_ = math.cos(yaw), math.sin(yaw)
                lx, ly = float(hx[i]), float(hy[i])
                if not (math.isfinite(lx) and math.isfinite(ly)):
                    continue
                wx, wy = x + c * lx - s_ * ly, y + s_ * lx + c * ly
                v = float(torch.hypot(r.state[i, dyn.IVX], r.state[i, dyn.IVY]))
                # what the return was (the simulator's truth): the other car, a prop, a wall, or something else
                trig = "other"
                if sim.M > 1:
                    j = int(sim.other_idx[i, 0])
                    if self.foot_dist(st_cpu[j].tolist(), wx, wy) < 0.15:
                        trig = "car"
                if trig == "other" and po is not None:
                    live = po.p_zhi[i] > po.p_zlo[i]
                    dp = torch.hypot(po.p_poses[i, :, 0] - wx, po.p_poses[i, :, 1] - wy)
                    dp = torch.where(live, dp, torch.full_like(dp, float("inf")))
                    if float(dp.min()) < 0.40:
                        trig = "prop"
                if trig == "other":
                    pt = torch.tensor([[wx, wy]], dtype=torch.float32, device=sim.device)
                    if float(sim.track.sample_edt(pt, sim.tid[i:i + 1])[0]) < 0.15:
                        trig = "wall"
                self.recent[i] = [(k0, s) for (k0, s) in self.recent[i] if self.k - k0 <= W] + [(self.k, float(s_hit[i]))]
                e = dict(LABEL, row=i, s_hit=round(float(s_hit[i]), 2), v=round(v, 2), trigger=trig,
                         edge_invalid=bool(CUR["edge_invalid"][i]),
                         cmd=round(float(cmd[i]), 2), lim=round(float(lim[i]), 2), wx=wx, wy=wy,
                         min_d=self.foot_dist([x, y, yaw], wx, wy), hit=False, age=0, reset=False,
                         dur=1, cut_sum=max(float(cmd[i]) - float(lim[i]), 0.0) * self.dt)
                self.open.append(e)
                self.active[i] = e

    def flush(self):
        EVENTS.extend(self.open)
        self.open = []


TRK = {}
_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step


def env_step(self, action):
    CUR.update(env=self, in_step=True, lim=None)
    try:
        return _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False


def sim_step(self, *a, **kw):
    entry = self.steps.clone()
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self:
        t = TRK.get(id(env))
        if t is None:
            t = TRK[id(env)] = Tracker(env)
        t.observe(env, self, r, entry)
    return r


gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
osl.speed_limit = measuring_limit             # the env imports it at call time: `from .obstacle_speed_limit import speed_limit`
for seed in SEEDS:
    for kind in KINDS:
        LABEL.clear(); LABEL.update(seed=seed, kind=kind)
        TRK.clear()
        out = os.path.join(OUT, f"race_s{seed}_{kind}.json")
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", os.environ.get("ANAT_ENVS", "32"),
                    "--steps", os.environ.get("ANAT_STEPS", "2400"), "--budget-laps", "0", "--teacher-a-lat", "7.0",
                    "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0", "--raceline-objective", "min_time",
                    "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", os.environ.get("ANAT_DEVICE", "cuda:0"),
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", "off",
                    "--opp-slots", json.dumps([{"kind": kind, "speed_scale": [0.7, 1.0]}]),
                    "--output", out, "--dial-value", "1.0489", "--obstacle-speed-limit",
                    *(["--opponent-speed-limit"] if os.environ.get("OSL") else []),
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        n0, c0 = len(EVENTS), len(CONTACTS)
        ev.main()
        for t in TRK.values():
            t.flush()
        dd = json.load(open(out))["nominal"]
        cs = dd.get("contact_severity") or {}
        km = float(cs.get("learner_distance_m", 0.0)) / 1000.0
        print(f"seed {seed} {kind}: {len(EVENTS) - n0} would-hold onsets, {len(CONTACTS) - c0} hard prop contacts, {km:.2f} km, "
              f"race {float(dd['mean_speed']):.2f} m/s", flush=True)
        json.dump({"events": EVENTS, "contacts": CONTACTS}, open(os.path.join(OUT, "holds.json"), "w"))


def cls(e):
    if e["hit"]:
        return "hit"
    if e.get("touch", 0.0) > 0.0:
        return "graze"
    return "close" if e["min_d"] < 0.10 else "near" if e["min_d"] < 0.30 else "clear"


full = [e for e in EVENTS if not e["reset"]]
print(f"{len(full)} would-hold onsets followed 1.5 s ({len(EVENTS) - len(full)} cut short by a reset)")
tot0 = max(sum(e["cut_sum"] for e in full), 1e-9)
print("triggered by (truth): " + " ".join(f"{t} {sum(1 for e in full if e['trigger'] == t)} ({100 * sum(e['cut_sum'] for e in full if e['trigger'] == t) / tot0:.0f} % of the cut)"
                                           for t in ("prop", "car", "wall", "other")))
for t in ("prop", "car", "wall", "other"):
    sub = [e for e in full if e["trigger"] == t]
    if sub:
        print(f"  {t}: cut out by a no-return beam at an end in {100 * sum(e['edge_invalid'] for e in sub) / len(sub):.0f} % of {len(sub)}")
full_all = full
full = [e for e in full_all if e["trigger"] == "prop"]
print("--- PROP-triggered holds only:")
for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 9.0)):
    sub = [e for e in full if lo <= e["s_hit"] < hi]
    if not sub:
        continue
    n = len(sub)
    cnt = {c: sum(1 for e in sub if cls(e) == c) for c in ("hit", "graze", "close", "near", "clear")}
    print(f"  s_hit {lo:.0f}-{hi:.0f} m: n {n:4d} | " + " ".join(f"{c} {100 * v / n:4.1f} %" for c, v in cnt.items())
          + f" | mean cut {sum(e['cmd'] - e['lim'] for e in sub) / n:.2f} m/s | held {sum(e['dur'] for e in sub) / n * 0.025:.2f} s"
          + f" | share of all cut {100 * sum(e['cut_sum'] for e in sub) / max(sum(e['cut_sum'] for e in full), 1e-9):.0f} %")
tot = max(sum(e["cut_sum"] for e in full), 1e-9)
print("share of all cut by outcome: " + " ".join(f"{c} {100 * sum(e['cut_sum'] for e in full if cls(e) == c) / tot:.0f} %"
                                                 for c in ("hit", "graze", "close", "near", "clear")))
for lo, hi in ((1, 2), (2, 4), (4, 8), (8, 10 ** 6)):
    sub = [e for e in full if lo <= e["dur"] < hi]
    if sub:
        print(f"  holds lasting {lo}-{hi} steps: n {len(sub)} | clear {100 * sum(cls(e) == 'clear' for e in sub) / len(sub):.0f} % | "
              f"hit {100 * sum(cls(e) == 'hit' for e in sub) / len(sub):.0f} % | graze {100 * sum(cls(e) == 'graze' for e in sub) / len(sub):.0f} % | close {100 * sum(cls(e) == 'close' for e in sub) / len(sub):.0f} %")
if CONTACTS:
    print(f"recall: {sum(c['held_before'] for c in CONTACTS)}/{len(CONTACTS)} hard prop contacts had a would-hold onset in the 1.5 s before")
