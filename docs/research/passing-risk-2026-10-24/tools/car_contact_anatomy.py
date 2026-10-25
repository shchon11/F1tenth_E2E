"""Research (2026-10-23): the anatomy of our policy's HARD car contacts in races (closing speed >= 1.5 m/s, the tables'
"hard"). On the representative floor most hard hits are car contacts (v1 entry finals: 1.47 of 2.35 /km) and nobody has
looked at one: which car hit which, from where, who moved, and what our plan said in the second before.

Each hard car-contact onset of a learner row is recorded at the simulator step it happens (before any reset) with:
  geometry at the onset, in the ego frame: the partner's position (dx ahead, dy left), heading difference, closing speed;
  the last 1 s (40 steps) of both cars: speeds, Frenet lateral offsets (who moved toward whom), the gap;
  our plan over that second: the commanded first speed v1 (what MAP drives) against the measured speed.
Classes: "turned" (|heading difference| >= 45 deg: a spin, a T-bone), else by the partner's place at the onset --
"ego_hits_rear" (partner ahead, dx > 0.30 m), "hit_from_behind" (dx < -0.30), "side" (abreast).
Lateral attribution (side / rear, last 0.5 s): the Frenet lateral distance each car covered TOWARD the other.

Protocol = the finals' race cells (tools/baselines/final_eval2.ours_race): race of 2, one opponent kind per cell with
speed_scale U(0.7, 1.0), soft contacts, props, rep floor, dial 1.0489, the checkpoint's own follower (its contract).
usage: car_contact_anatomy.py CKPT OUT_DIR [SEEDS=81,82] [KINDS=forzaeth,forzaeth_pred,lane_switch,interactive]
env OSL=1 / OBSL=1: the speed limit for a car ahead / for an obstacle on the plan's path on (10-26: each event then also carries the
last second of what the limits did -- held or not, the policy's nearest-car reading, the command after the limits)."""
import json
import math
import os
import sys

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, mpc
from f1sim.learn import evaluate as ev
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "81,82").split(",")]
KINDS = (sys.argv[4] if len(sys.argv) > 4 else "forzaeth,forzaeth_pred,lane_switch,interactive").split(",")
TRACK = os.environ.get("ANAT_TRACK", "real:korea_2026_competition")
HARD = float(os.environ.get("ANAT_HARD", "1.5"))   # smoke tests lower it
K = 40                                   # 1 s of history at 40 Hz
os.makedirs(OUT, exist_ok=True)

CUR = {"env": None, "action": None, "in_step": False}
EVENTS = []
LABEL = {}


class Recorder:
    def __init__(self, env):
        sim = env.sim
        B, A = int(sim.state.shape[0]), int(env.act_dim)
        dev = sim.device
        self.st = torch.zeros(K, B, 6, device=dev)        # x y yaw vx vy r
        self.lat = torch.zeros(K, B, device=dev)          # Frenet lateral offset
        self.act = torch.zeros(K, B, A, device=dev)
        self.vm = torch.zeros(K, B, device=dev)           # the speed the plan was decoded at
        self.age = torch.zeros(B, dtype=torch.long, device=dev)
        self.prev_car = torch.zeros(B, dtype=torch.bool, device=dev)
        self.held_car = torch.zeros(K, B, dtype=torch.bool, device=dev)   # the car limit held the command
        self.held_obs = torch.zeros(K, B, dtype=torch.bool, device=dev)   # the obstacle limit held the command
        self.reading = torch.zeros(K, B, 5, device=dev)                   # the policy's nearest-car reading (raw)
        self.cmd = torch.zeros(K, B, device=dev)                          # the speed command after the limits

    def observe(self, env, sim, r, entry, action, v_plan):
        fresh = entry == 0
        st = r.state[:, [dyn.IX, dyn.IY, dyn.IYAW, dyn.IVX, dyn.IVY, dyn.IR]].float()
        _, lat, _ = sim.track.project(r.state[:, :2], sim.tid, prev_idx=sim.cl_idx)
        self.age = torch.where(fresh, torch.zeros_like(self.age), self.age)
        self.st = torch.roll(self.st, 1, 0); self.st[0] = st
        self.lat = torch.roll(self.lat, 1, 0); self.lat[0] = lat.float()
        self.act = torch.roll(self.act, 1, 0); self.act[0] = action.float()
        self.vm = torch.roll(self.vm, 1, 0); self.vm[0] = v_plan.float()
        self.held_car = torch.roll(self.held_car, 1, 0); self.held_car[0] = env.opponent_limited
        self.held_obs = torch.roll(self.held_obs, 1, 0); self.held_obs[0] = env.obstacle_limited
        rd = getattr(env, "opponent_reading", None)
        self.reading = torch.roll(self.reading, 1, 0)
        if rd is not None:
            self.reading[0] = rd.float()
        self.cmd = torch.roll(self.cmd, 1, 0); self.cmd[0] = env.last_cmd_raw[:, 1].float()
        self.age += 1
        if r.car_collision is None or r.car_impact is None:
            return
        onset = r.car_collision & ~(self.prev_car & ~fresh)
        self.prev_car = r.car_collision.clone()
        hit = onset & env.learner & (r.car_impact >= HARD)
        if not bool(hit.any()):
            return
        spec, vmax = env.tracker.spec, float(env.tracker.v_max)
        for i in hit.nonzero().flatten().tolist():
            j = int(sim.other_idx[i, 0])
            n = int(min(K, int(self.age[i]), int(self.age[j])))
            e, o = self.st[:n, i].cpu(), self.st[:n, j].cpu()                 # newest first
            le, lo = self.lat[:n, i].cpu(), self.lat[:n, j].cpu()
            c, s_ = math.cos(float(e[0, 2])), math.sin(float(e[0, 2]))
            wx, wy = float(o[0, 0] - e[0, 0]), float(o[0, 1] - e[0, 1])
            dx, dy = c * wx + s_ * wy, -s_ * wx + c * wy
            dpsi = math.degrees(math.atan2(math.sin(float(o[0, 2] - e[0, 2])), math.cos(float(o[0, 2] - e[0, 2]))))
            speed = lambda t: float(torch.hypot(t[0], t[1]))
            ve, vo = speed(e[0, 3:5]), speed(o[0, 3:5])
            if abs(dpsi) >= 45:
                cls = "turned"
            elif dx > 0.30:
                cls = "ego_hits_rear"
            elif dx < -0.30:
                cls = "hit_from_behind"
            else:
                cls = "side"
            h = min(n - 1, 20)                                               # 0.5 s back
            side = 1.0 if float(lo[0] - le[0]) >= 0 else -1.0                # partner to the ego's left (+) in Frenet
            ego_toward = float(le[0] - le[h]) * side                         # ego's lateral move toward the partner
            opp_toward = float(lo[h] - lo[0]) * side                         # partner's lateral move toward the ego
            gap = [float(torch.hypot(*(self.st[t, j, :2] - self.st[t, i, :2]).cpu())) for t in range(n)]
            a_hist = self.act[:n, i]
            if spec.speed_mode == "wp4":
                _, _, vk = mpc.decode_wp4(a_hist, self.vm[:n, i], vmax, torch.full((n,), 9.0, device=a_hist.device), spec)
                v1 = vk[:, 0].cpu()
                v_end = vk[:, -1].cpu()
            else:
                _, _, v0, v1_ = mpc.decode(a_hist, self.vm[:n, i], vmax, torch.full((n,), 9.0, device=a_hist.device), spec)
                v1, v_end = v0.cpu(), v1_.cpu()
            ve_hist = [speed(e[t, 3:5]) for t in range(n)]
            vo_hist = [speed(o[t, 3:5]) for t in range(n)]
            EVENTS.append(dict(LABEL, row=i, impact=round(float(r.car_impact[i]), 3), cls=cls,
                               dx=round(dx, 3), dy=round(dy, 3), dpsi=round(dpsi, 1), v_ego=round(ve, 2), v_opp=round(vo, 2),
                               ego_toward=round(ego_toward, 3), opp_toward=round(opp_toward, 3),
                               v_ego_hist=[round(x, 2) for x in ve_hist], v_opp_hist=[round(x, 2) for x in vo_hist],
                               gap_hist=[round(x, 2) for x in gap],
                               plan_v1_hist=[round(float(x), 2) for x in v1], plan_vend_hist=[round(float(x), 2) for x in v_end],
                               yaw_rate_ego=round(float(e[0, 5]), 2), yaw_rate_opp=round(float(o[0, 5]), 2),
                               lat_ego=round(float(le[0]), 3), lat_opp=round(float(lo[0]), 3), hist_steps=n,
                               car_held_hist=[int(x) for x in self.held_car[:n, i].tolist()],
                               obs_held_hist=[int(x) for x in self.held_obs[:n, i].tolist()],
                               reading_hist=[[round(float(v), 3) for v in row] for row in self.reading[:n, i].tolist()],
                               cmd_hist=[round(float(x), 2) for x in self.cmd[:n, i].tolist()]))


RECS = {}
_orig_env_step = gym_env.F1VecEnv.step
_orig_sim_step = Simulator.step


def env_step(self, action):
    CUR.update(env=self, action=action, in_step=True, v_plan=self._tracker_plan_speed().clone())
    try:
        return _orig_env_step(self, action)
    finally:
        CUR["in_step"] = False


def sim_step(self, *a, **kw):
    entry = self.steps.clone()
    r = _orig_sim_step(self, *a, **kw)
    env = CUR["env"]
    if CUR["in_step"] and env is not None and env.sim is self and self.M > 1:
        rec = RECS.get(id(env))
        if rec is None:
            rec = RECS[id(env)] = Recorder(env)
        rec.observe(env, self, r, entry, CUR["action"], CUR["v_plan"])
    return r


gym_env.F1VecEnv.step = env_step
Simulator.step = sim_step
for seed in SEEDS:
    for kind in KINDS:
        LABEL.clear(); LABEL.update(seed=seed, kind=kind)
        RECS.clear()
        out = os.path.join(OUT, f"race_s{seed}_{kind}.json")
        # ANAT_SCALE: the opponent's speed_scale (one number, e.g. 0.05 = a near-stopped car; 10-25), ANAT_CORRIDOR: the
        # props' raceline corridor ("on" keeps them off the line)
        slot = {"kind": kind, "speed_scale": (float(os.environ["ANAT_SCALE"]) if os.environ.get("ANAT_SCALE") else [0.7, 1.0])}
        sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", os.environ.get("ANAT_ENVS", "32"), "--steps", os.environ.get("ANAT_STEPS", "2400"),
                    "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                    "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", os.environ.get("ANAT_DEVICE", "cuda:0"),
                    "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                    "--race-size", "2", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                    "--procedural-max-props", "10", "--procedural-raceline-corridor", os.environ.get("ANAT_CORRIDOR", "off"),
                    "--opp-slots", json.dumps([slot]),
                    "--output", out, "--dial-value", "1.0489",
                    *(["--opponent-speed-limit"] if os.environ.get("OSL") else []),
                    *(["--obstacle-speed-limit"] if os.environ.get("OBSL") else []),
                    "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
        n0 = len(EVENTS)
        ev.main()
        d = json.load(open(out))["nominal"]
        cs = d.get("contact_severity") or {}
        km = float(cs.get("learner_distance_m", 0.0)) / 1000.0
        hard_car = int(sum(((cs.get("car") or {}).get("fine_hist") or [])[15:]))
        print(f"seed {seed} {kind}: {len(EVENTS) - n0} events recorded, the meter's hard car onsets {hard_car}, "
              f"{km:.2f} learner km, race {float(d['mean_speed']):.2f} m/s", flush=True)
        json.dump(EVENTS, open(os.path.join(OUT, "events.json"), "w"))

from collections import Counter
cnt = Counter(e["cls"] for e in EVENTS)
print("classes:", dict(cnt), flush=True)
for c in cnt:
    sub = [e for e in EVENTS if e["cls"] == c]
    mean = lambda k: sum(e[k] for e in sub) / len(sub)
    print(f"  {c:16s} n {len(sub):3d} | impact {mean('impact'):.2f} | v_ego {mean('v_ego'):.2f} v_opp {mean('v_opp'):.2f} | "
          f"ego toward {mean('ego_toward'):+.3f} m, opp toward {mean('opp_toward'):+.3f} m (last 0.5 s)", flush=True)
