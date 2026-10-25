"""Console path (SimWorker), 10-26: the obstacle speed limit checkbox (SessionConfig.obstacle_speed_limit) with the car limit, a race of 2
on Korea 2026 with props on the line, 30 s x 4 races, off vs on. Prints the worker's log lines about the limits and how often the
obstacle limit held the command. usage: obs_console_check.py CKPT [both|obs_only]  (obs_only: the car limit off, 10-26 --
the obstacle limit must still take the policy's nearest-car reading)"""
import sys

from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P

ckpt = sys.argv[1]
car_on = (sys.argv[2] if len(sys.argv) > 2 else "both") != "obs_only"


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
                          opponent_slots=[{"kind": "forzaeth", "speed_scale": [0.7, 1.0]}], opponent_speed_limit=car_on,
                          obstacle_speed_limit=on)
    s = w.build_session(cfg, 0)
    try:
        for g in (s.get("fastpath"), s.get("controller"), s.get("teacher_graph")):
            if g is not None and hasattr(g, "adopt"):
                g.adopt()
        env = s["env"]
        held = steps = 0
        for _ in range(1200):
            w._step_once(s)
            held += int((env.obstacle_limited & env.learner).sum()); steps += int(env.learner.sum())
        prog = env.ep_progress[env.learner]
        rd = getattr(env, "opponent_reading", None)
        fed = None if rd is None else bool((rd[env.learner, 0] != -10.0).all())
        print(f"obstacle limit {'ON ' if on else 'OFF'}: held {100 * held / max(steps, 1):.2f} % of steps, mean progress "
              f"{float(prog.mean()):.1f} m in 30 s | car reading fed: {fed} | log: {[t[:60] for t in logs if '감속' in t][:4]}", flush=True)
    finally:
        w._release_session(s)
