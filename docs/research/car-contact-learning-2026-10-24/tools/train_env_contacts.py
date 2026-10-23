"""Research (2026-10-23, R5): what the car contacts are like in the TRAINING traffic.

Charging contacts from 0.75 m/s did not move our car contacts in 256 updates, while the policy's state predicts them at
AUC 0.90 (contact_probe). One reason the cost could fail to teach: the training traffic is not the evaluation's. The
recipe races 4 cars (3 opponents), half of them league policies, the scripted ones with timed events (brake, stop,
shift, weave at 0.5 per 10 s) and reactive behaviour (defend 0.3, line 0.2, oblivious 0.1). A contact an opponent
causes with an event the policy cannot foresee carries no lesson; if those are most of the contacts the cost sees, it is
mostly noise to PPO.

The entry policy driven in that traffic (the recipe's own slot table, league of the v2PH_s2421 run, props as in
training, Korea, rep floor, dial 1.0489, MAP): every learner car-contact onset >= 0.75 m/s, with the touching car (the
nearest of its race at the onset), whether that car is a league policy or scripted, its timed event and reactive action
now and in the last 1 s, the geometry class and whether we were accelerating 0.5 s before.
usage: train_env_contacts.py CKPT OUT_DIR [SEEDS=81,82] [ENVS=64]"""
import json
import math
import os
import sys
from collections import Counter

import torch

from f1sim import dynamics as dyn
from f1sim import gym_env, mpc
from f1sim.learn import evaluate as ev
from f1sim.opponent_events import EVENT_ID, REACTIVE_BIT
from f1sim.sim import Simulator

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "81,82").split(",")]
ENVS = int(sys.argv[4]) if len(sys.argv) > 4 else 64
TRACK = os.environ.get("ANAT_TRACK", "real:korea_2026_competition")
MIN_IMP, HARD, K = 0.75, 1.5, 40
RR, C = "/home/shchon11/f1sim_runs", "/home/shchon11/F1tenth_E2E/checkpoints"
LEAGUE = [f"{RR}/v2PH_s2421_seats/seat0.pt", f"{RR}/v2PH_s2421_seats/seat1.pt", f"{C}/iccas_constrained_v2R.pt",
          f"{C}/iccas_constrained_safe_s976.pt", f"{C}/iccas_league_c_s949.pt", f"{RR}/ffR_lin/ppo_final.pt",
          f"{RR}/spec_korea_c2new_s947/ppo_final.pt", f"{RR}/spec_korea_contact2_s942/ppo_final.pt",
          f"{RR}/spec_korea_contact_s915/ppo_u768.pt"]
SLOT = {"kind_mix": ["forzaeth", "forzaeth_pred", "lane_switch", "interactive"], "speed_scale": [0.7, 1.05],
        "ckpt_mix": LEAGUE, "ckpt_p": 0.5, "present": 0.5, "spawn": "random",
        "reactive": {"defend": 0.3, "line": 0.2, "oblivious": 0.1}, "events": ["brake", "stop", "shift", "weave"],
        "event_rate": 0.5}
ID2EV = {v: k for k, v in EVENT_ID.items()}
os.makedirs(OUT, exist_ok=True)
CUR = {"env": None, "in_step": False}
EVENTS, LABEL, RECS = [], {}, {}


class Recorder:
    def __init__(self, env):
        B, dev = int(env.sim.state.shape[0]), env.sim.device
        self.ev = torch.zeros(K, B, dtype=torch.long, device=dev)
        self.re = torch.zeros(K, B, dtype=torch.long, device=dev)
        self.v = torch.zeros(K, B, device=dev)
        self.v1 = torch.zeros(K, B, device=dev)
        self.prev = torch.zeros(B, dtype=torch.bool, device=dev)
        self.km = 0.0

    def observe(self, env, sim, r, entry, action, v_plan):
        fresh = entry == 0
        lr = env.learner
        evs = env.events
        kind = evs.kind if evs is not None else torch.zeros_like(self.prev, dtype=torch.long)
        react = evs.react_now if evs is not None else torch.zeros_like(kind)
        v = torch.hypot(r.state[:, dyn.IVX], r.state[:, dyn.IVY])
        spec, vmax = env.tracker.spec, float(env.tracker.v_max)
        cap = torch.full_like(v, 9.0)
        if spec.speed_mode == "wp4":
            _, _, vk = mpc.decode_wp4(action.float(), v_plan.float(), vmax, cap, spec); v1 = vk[:, 0]
        else:
            _, _, v1, _ = mpc.decode(action.float(), v_plan.float(), vmax, cap, spec)
        for buf, val in (("ev", kind), ("re", react), ("v", v), ("v1", v1)):
            t = getattr(self, buf); t = torch.roll(t, 1, 0); t[0] = val; setattr(self, buf, t)
        self.km += float((v * lr.float()).sum()) * float(sim.control_dt) / 1000.0
        if r.car_collision is None or r.car_impact is None:
            return
        onset = r.car_collision & ~(self.prev & ~fresh)
        self.prev = r.car_collision.clone()
        hit = onset & lr & (r.car_impact >= MIN_IMP)
        if not bool(hit.any()):
            return
        st = r.state
        for i in hit.nonzero().flatten().tolist():
            others = sim.other_idx[i]
            d = (st[others, :2] - st[i, :2]).norm(dim=1)
            j = int(others[int(d.argmin())])
            c, s_ = math.cos(float(st[i, 2])), math.sin(float(st[i, 2]))
            wx, wy = float(st[j, 0] - st[i, 0]), float(st[j, 1] - st[i, 1])
            dx = c * wx + s_ * wy
            dpsi = math.degrees(math.atan2(math.sin(float(st[j, 2] - st[i, 2])), math.cos(float(st[j, 2] - st[i, 2]))))
            cls = "turned" if abs(dpsi) >= 45 else ("ego_hits_rear" if dx > 0.30 else ("hit_from_behind" if dx < -0.30 else "side"))
            ev_now, ev_1s = int(self.ev[0, j]), sorted({int(x) for x in self.ev[:, j].tolist()} - {0})
            re_1s = int(self.re[:, j].max())
            EVENTS.append(dict(LABEL, impact=round(float(r.car_impact[i]), 3), hard=bool(float(r.car_impact[i]) >= HARD),
                               cls=cls, partner_league=bool(env.pool_driven[j]),
                               partner_event_now=ID2EV.get(ev_now, str(ev_now)) if ev_now else None,
                               partner_events_1s=[ID2EV.get(x, str(x)) for x in ev_1s],
                               partner_defend_1s=bool(re_1s & REACTIVE_BIT["defend"]),
                               partner_line_1s=bool(re_1s & REACTIVE_BIT["line"]),
                               partner_oblivious=bool(re_1s & REACTIVE_BIT["oblivious"]),
                               ego_accel_05=bool(float(self.v1[min(20, K - 1), i]) > float(self.v[min(20, K - 1), i]) + 0.3),
                               v_ego=round(float(self.v[0, i]), 2), v_opp=round(float(self.v[0, j]), 2)))


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
KM = 0.0
for seed in SEEDS:
    LABEL.clear(); LABEL.update(seed=seed)
    RECS.clear()
    out = os.path.join(OUT, f"train_traffic_s{seed}.json")
    sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", TRACK, "--envs", str(ENVS), "--steps", "2400",
                "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                "--race-size", "4", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                "--procedural-max-props", "18", "--procedural-raceline-corridor", "off",
                "--opp-slots", json.dumps([SLOT, SLOT, SLOT]),
                "--output", out, "--dial-value", "1.0489",
                "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92"]
    n0 = len(EVENTS)
    ev.main()
    km = sum(r.km for r in RECS.values())
    KM += km
    sub = EVENTS[n0:]
    print(f"seed {seed}: {len(sub)} car contacts >= {MIN_IMP} m/s ({sum(e['hard'] for e in sub)} hard) in {km:.1f} learner km", flush=True)
    json.dump(EVENTS, open(os.path.join(OUT, "events.json"), "w"))

E = EVENTS
frac = lambda xs: round(sum(xs) / max(len(xs), 1), 3)
res = {"ckpt": CK, "km": round(KM, 2), "contacts_ge_075": len(E), "per_km_ge_075": round(len(E) / max(KM, 1e-6), 2),
       "hard": sum(e["hard"] for e in E), "hard_per_km": round(sum(e["hard"] for e in E) / max(KM, 1e-6), 2)}
for name, grp in (("all", E), ("hard", [e for e in E if e["hard"]])):
    res[name] = {"n": len(grp), "partner_league": frac([e["partner_league"] for e in grp]),
                 "partner_event_in_last_1s": frac([bool(e["partner_events_1s"]) for e in grp]),
                 "partner_event_now": dict(Counter(e["partner_event_now"] for e in grp if e["partner_event_now"])),
                 "partner_defend_1s": frac([e["partner_defend_1s"] for e in grp]),
                 "partner_line_1s": frac([e["partner_line_1s"] for e in grp]),
                 "partner_oblivious": frac([e["partner_oblivious"] for e in grp]),
                 "scripted_no_event_no_react": frac([(not e["partner_league"]) and not e["partner_events_1s"] and not
                                                     (e["partner_defend_1s"] or e["partner_line_1s"]) for e in grp]),
                 "ego_accel_05": frac([e["ego_accel_05"] for e in grp]),
                 "classes": dict(Counter(e["cls"] for e in grp))}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
