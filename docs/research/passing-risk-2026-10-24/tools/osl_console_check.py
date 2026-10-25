"""Console path (SimWorker), 10-25: the speed limit for a car ahead with the latched lane check, on a near-stopped car.
Race of 2 on Korea 2026, the other car a raceline teacher at x0.05 (props off the line), 30 s x 4 races per setting:
the console's checkbox (SessionConfig.opponent_speed_limit) off and on, for CKPT. Prints the worker's own log lines about
the limit, how often it held the command, the learner's car contacts (onsets) and progress.
usage: osl_console_check.py CKPT"""
import sys

import torch

from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P

ckpt = sys.argv[1]


class _Null:
    def send(self, *a, **k):
        pass

    def put(self, *a, **k):
        pass

    def poll(self, *a):
        return False


logs = []
w = sim_worker.SimWorker(_Null(), _Null())
w.say = lambda kind, **k: logs.append(k.get("text", ""))
w.stage = lambda *a, **k: None
w._hold_parked_ok = True
for on in (False, True):
    logs.clear()
    cfg = P.SessionConfig(run=ckpt, map_name="real:korea_2026_competition", races=4, cars_per_race=2,
                          opponent_slots=[{"kind": "raceline", "speed_scale": 0.05}], opponent_speed_limit=on)
    s = w.build_session(cfg, 0)
    try:
        for g in (s.get("fastpath"), s.get("controller"), s.get("teacher_graph")):
            if g is not None and hasattr(g, "adopt"):
                g.adopt()
        env = s["env"]
        onsets = held = steps = 0
        prev = torch.zeros_like(env.learner)
        for _ in range(1200):
            w._step_once(s)
            c = env.sim.car_collided if hasattr(env.sim, "car_collided") else env.sim.collided
            now = c & env.learner
            onsets += int((now & ~prev).sum()); prev = now
            held += int((env.opponent_limited & env.learner).sum()); steps += int(env.learner.sum())
        prog = env.ep_progress[env.learner]
        print(f"limit {'ON ' if on else 'OFF'}: held {100 * held / max(steps, 1):.2f} % of steps, contact onsets {onsets}, mean progress "
              f"{float(prog.mean()):.1f} m in 30 s | log: {[t for t in logs if '감속' in t][:3]}", flush=True)
    finally:
        w._release_session(s)
