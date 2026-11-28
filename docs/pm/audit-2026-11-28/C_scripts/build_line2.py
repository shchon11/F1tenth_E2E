"""Audit C: the min-time raceline on the exam track (korea26_scan+duct+smooth -> its smooth parent), for the modular
raceline+MAP baseline only (our policy never reads it). Limits = the modular 'raceline solo' tuning on the
neighbouring venue map korea_2026_competition (baseline_settings.json: limits [9.0, 7.5, 6.0])."""
import sys, time
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import common
from f1sim.raceline import Raceline
t0 = time.time()
trs, _ = common.load_tracks(["real:korea26_scan+duct+smooth"], racelines=False)
lim = common.teacher_limits(7.0, 6.5, 4.0)
rl = Raceline.build_cached(trs[0], objective="min_time", **lim)
print("done", len(rl.xy), f"{time.time() - t0:.0f} s", flush=True)
