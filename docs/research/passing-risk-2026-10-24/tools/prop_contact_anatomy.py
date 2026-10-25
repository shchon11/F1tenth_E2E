"""Research (2026-10-26): what the HARD prop contacts left with the obstacle speed limit on are.

With both speed limits on (the car limit with the slow-car head + latch, and the obstacle limit), props on the racing line,
the finals race cells still have 0.6-0.8 hard (>= 1.5 m/s) prop contacts per km -- half of the hard contacts left. The
obstacle limit holds the speed before the first SMALL ISOLATED scan cluster on the plan's own path; a prop it does not act on
is one of:
  pushed       -- the prop moved >= 0.10 m in the last second (an opponent or we shoved it): it came into the path late;
  beside       -- never within the limit's corridor of the plan's path (from the bumper to the plan's length) in the last
                  second: a side swipe, a corner, or a path too short to reach it;
  unseen       -- on the path, but no return of the scans the policy planned from fell on it (occluded, out of view);
  not_kept     -- on the path and seen, never a kept cluster: part of a row grown from a wall, or next to another piece
                  (the procedural patterns are rows from a wall), or too few beams;
  held         -- the limit held the command at least once in the last second and the car still hit at >= 1.5 m/s;
  kept_free    -- kept, on the path, never held: the command was already under the limit.
Classified in that order. Each hard prop onset of a learner row (the simulator's per-kind onsets, as the contact-severity
meter files them) is recorded with the hit prop (the live slot nearest the car), its pattern (`procedural_obstacles.KINDS`),
its displacement, its neighbours within 0.6 m, its distance to the wall, and per step of the last second: whether it was
on the plan's path, seen, kept, and whether the limit held, with the command and the speed.

Protocol = the finals' race cells with props on the line (tools/research/obsft_eval_4070.sh), both limits switchable.
usage: prop_contact_anatomy.py CKPT OUT_DIR [SEEDS=81,82] [KINDS=forzaeth,forzaeth_pred,lane_switch,interactive]
env OSL=1 / OBSL=1: the car / obstacle limit on; ANAT_DEVICE, ANAT_ENVS, ANAT_STEPS, ANAT_HARD for smoke tests."""
import json
import math
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, mpc
from f1sim.learn import evaluate as ev
from f1sim.obstacle_speed_limit import DEFAULT as LIM, cluster_keep, path_points
from f1sim.procedural_obstacles import KINDS as PATTERNS
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "81,82").split(",")]
KINDS = (sys.argv[4] if len(sys.argv) > 4 else "forzaeth,forzaeth_pred,lane_switch,interactive").split(",")
TRACK = os.environ.get("ANAT_TRACK", "real:korea_2026_competition")
HARD = float(os.environ.get("ANAT_HARD", "1.5"))
K = 40                                   # 1 s at 40 Hz
PUSHED = 0.10                            # [m] moved in the last second
os.makedirs(OUT, exist_ok=True)

CUR = {"env": None, "action": None, "in_step": False}
EVENTS = []
LABEL = {}


class Recorder:
    def __init__(self, env):
        sim = env.sim
        B, A = int(sim.state.shape[0]), int(env.act_dim)
        dev = sim.device
        n = int(sim.lidar.angles.shape[0])
        C = int(env.procedural.p_poses.shape[1])
        self.st = torch.zeros(K, B, 6, device=dev)              # x y yaw vx vy r
        self.act = torch.zeros(K, B, A, device=dev)
        self.vm = torch.zeros(K, B, device=dev)
        self.scan = torch.zeros(K, B, n, device=dev)            # the scan the policy planned from
        self.pose = torch.zeros(K, B, C, 2, device=dev)         # the props, after the step
        self.held_obs = torch.zeros(K, B, dtype=torch.bool, device=dev)
        self.held_car = torch.zeros(K, B, dtype=torch.bool, device=dev)
        self.cmd = torch.zeros(K, B, device=dev)
        self.age = torch.zeros(B, dtype=torch.long, device=dev)

    def push(self, buf, val):
        buf.copy_(torch.roll(buf, 1, 0))
        buf[0] = val

    def observe(self, env, sim, r, entry, action, v_plan, scan):
        fresh = entry == 0
        self.age = torch.where(fresh, torch.zeros_like(self.age), self.age)
        st = r.state[:, [dyn.IX, dyn.IY, dyn.IYAW, dyn.IVX, dyn.IVY, dyn.IR]].float()
        self.push(self.st, st)
        self.push(self.act, action.float())
        self.push(self.vm, v_plan.float())
        self.push(self.scan, scan.float())
        self.push(self.pose, env.procedural.p_poses[:, :, :2].float())
        self.push(self.held_obs, env.obstacle_limited)
        self.push(self.held_car, env.opponent_limited)
        self.push(self.cmd, env.last_cmd_raw[:, 1].float())
        self.age += 1
        on, imp = getattr(r, "contact_onsets", None), getattr(r, "contact_impacts", None)
        if on is None or imp is None:
            return
        hit = on[:, 1] & env.learner & (imp[:, 1].float() >= HARD)
        if not bool(hit.any()):
            return
        po = env.procedural
        ang = sim.lidar.angles.float()
        mx, my = float(env.cfg.lidar.mount_x), float(env.cfg.lidar.mount_y)
        rmax = float(env.cfg.lidar.range_max)
        spec, vmax = env.tracker.spec, float(env.tracker.v_max)
        for i in hit.nonzero().flatten().tolist():
            nh = int(min(K, int(self.age[i])))
            live = po.p_zhi[i] > po.p_zlo[i]
            if not bool(live.any()) or nh < 2:
                continue
            ex, ey = self.st[0, i, 0], self.st[0, i, 1]
            d = torch.hypot(self.pose[0, i, :, 0] - ex, self.pose[0, i, :, 1] - ey)
            d = torch.where(live, d, torch.full_like(d, float("inf")))
            s = int(d.argmin())
            r_in = float(po.p_d[i, s].min().clamp(max=1.0))               # inscribed radius (face planes)
            centre = self.pose[:nh, i, s]                                 # newest first
            moved = float(torch.hypot(*(centre[0] - centre[nh - 1])))
            others = torch.hypot(self.pose[0, i, :, 0] - centre[0, 0], self.pose[0, i, :, 1] - centre[0, 1])
            others = torch.where(live, others, torch.full_like(others, float("inf")))
            others[s] = float("inf")
            neigh = int((others < 0.6).sum())
            wall = float(sim.track.sample_edt(centre[0][None], sim.tid[i:i + 1])[0])
            pat = PATTERNS[int(po.last_kind[i, int(po.slot_pattern[i, s])])]
            # per step of the last second: the prop in the car frame, on the plan's path, seen, kept
            yaw = self.st[:nh, i, 2]
            c, s_ = torch.cos(yaw), torch.sin(yaw)
            wx, wy = centre[:, 0] - self.st[:nh, i, 0], centre[:, 1] - self.st[:nh, i, 1]
            rx, ry = c * wx + s_ * wy, -s_ * wx + c * wy                  # (nh,)
            a_h = self.act[:nh, i]
            cap = torch.full((nh,), float(env.speed_cap[i]), device=a_h.device)
            dec = mpc.decode_wp4 if spec.speed_mode == "wp4" else mpc.decode
            k_h, L_h = dec(a_h, self.vm[:nh, i], vmax, cap, spec)[:2]
            pts, sarc = path_points(k_h.float(), L_h.float())           # (nh, N, 2)
            dp = torch.hypot(pts[..., 0] - rx[:, None], pts[..., 1] - ry[:, None]) - r_in
            valid = (sarc[None, :] <= L_h.float()[:, None]) & (sarc[None, :] > LIM.front)
            on_path = ((dp < LIM.corridor) & valid).any(1)
            rr = self.scan[:nh, i]
            ok = torch.isfinite(rr) & (rr > 0.05) & (rr < rmax - 1e-3)
            r0 = torch.where(ok, rr, torch.zeros_like(rr))
            px, py = mx + r0 * torch.cos(ang), my + r0 * torch.sin(ang)
            keep = cluster_keep(rr, ok, px, py)
            near = torch.hypot(px - rx[:, None], py - ry[:, None]) < (r_in + 0.12)
            seen = (near & ok).any(1)
            kept = (near & keep).any(1)
            held = self.held_obs[:nh, i]
            if moved >= PUSHED:
                cls = "pushed"
            elif not bool(on_path.any()):
                cls = "beside"
            elif not bool((seen & on_path).any()):
                cls = "unseen"
            elif not bool((kept & on_path).any()):
                cls = "not_kept"
            elif bool(held.any()):
                cls = "held"
            else:
                cls = "kept_free"
            v_h = torch.hypot(self.st[:nh, i, 3], self.st[:nh, i, 4])
            EVENTS.append(dict(LABEL, row=i, impact=round(float(imp[i, 1]), 3), cls=cls, pattern=pat,
                               sid=int(po.p_sid[i, s]), mass=round(float(po.p_mass[i, s]), 2), r_in=round(r_in, 3),
                               moved=round(moved, 3), neighbours=neigh, wall_dist=round(wall, 3),
                               rel_now=[round(float(rx[0]), 3), round(float(ry[0]), 3)],
                               rel_hist=[[round(float(a), 2), round(float(b), 2)] for a, b in zip(rx.tolist(), ry.tolist())],
                               on_path=[int(x) for x in on_path.tolist()], seen=[int(x) for x in seen.tolist()],
                               kept=[int(x) for x in kept.tolist()], held_obs=[int(x) for x in held.tolist()],
                               held_car=[int(x) for x in self.held_car[:nh, i].tolist()],
                               cmd=[round(float(x), 2) for x in self.cmd[:nh, i].tolist()],
                               v=[round(float(x), 2) for x in v_h.tolist()],
                               plan_len=[round(float(x), 2) for x in L_h.tolist()], hist_steps=nh))


RECS = {}
_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step


def env_step(self, action):
    lr = self.last_result
    scan = lr.scan if (lr is not None and lr.scan is not None) else None
    CUR.update(env=self, action=action, in_step=True, v_plan=self._tracker_plan_speed().clone(),
               scan=None if scan is None else scan.clone())
    try:
        return _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False


def sim_step(self, *a, **kw):
    entry = self.steps.clone()
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self and env.procedural is not None and CUR.get("scan") is not None:
        rec = RECS.get(id(env))
        if rec is None:
            rec = RECS[id(env)] = Recorder(env)
        rec.observe(env, self, r, entry, CUR["action"], CUR["v_plan"], CUR["scan"])
    return r


gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
for seed in SEEDS:
    for kind in KINDS:
        LABEL.clear(); LABEL.update(seed=seed, kind=kind)
        RECS.clear()
        out = os.path.join(OUT, f"race_s{seed}_{kind}.json")
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", os.environ.get("ANAT_ENVS", "32"),
                    "--steps", os.environ.get("ANAT_STEPS", "2400"), "--budget-laps", "0", "--teacher-a-lat", "7.0",
                    "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0", "--raceline-objective", "min_time",
                    "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", os.environ.get("ANAT_DEVICE", "cuda:0"),
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("ANAT_CORRIDOR", "off"),
                    "--opp-slots", json.dumps([{"kind": kind, "speed_scale": [0.7, 1.0]}]),
                    "--output", out, "--dial-value", "1.0489",
                    *(["--opponent-speed-limit"] if os.environ.get("OSL") else []),
                    *(["--obstacle-speed-limit"] if os.environ.get("OBSL") else []),
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        n0 = len(EVENTS)
        ev.main()
        dd = json.load(open(out))["nominal"]
        cs = dd.get("contact_severity") or {}
        km = float(cs.get("learner_distance_m", 0.0)) / 1000.0
        hard_prop = int(sum(((cs.get("prop") or {}).get("fine_hist") or [])[int(round(HARD / 0.1)):]))
        print(f"seed {seed} {kind}: {len(EVENTS) - n0} events recorded, the meter's hard prop onsets {hard_prop}, "
              f"{km:.2f} learner km, race {float(dd['mean_speed']):.2f} m/s", flush=True)
        json.dump(EVENTS, open(os.path.join(OUT, "events.json"), "w"))

from collections import Counter
cnt = Counter(e["cls"] for e in EVENTS)
print("classes:", dict(cnt), flush=True)
print("patterns:", dict(Counter(e["pattern"] for e in EVENTS)), flush=True)
for c in cnt:
    sub = [e for e in EVENTS if e["cls"] == c]
    mean = lambda k: sum(e[k] for e in sub) / len(sub)
    print(f"  {c:10s} n {len(sub):3d} | impact {mean('impact'):.2f} | moved {mean('moved'):.2f} m | neighbours "
          f"{mean('neighbours'):.1f} | wall {mean('wall_dist'):.2f} m | patterns {dict(Counter(e['pattern'] for e in sub))}",
          flush=True)
