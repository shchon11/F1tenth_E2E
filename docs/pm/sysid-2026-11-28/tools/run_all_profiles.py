"""bag_profile on every bag (no --car: each bag on its own) -> per-bag outputs in the scratch area + one json row per
bag for the reproducibility table. Usage: run_all_profiles.py [keys...]"""
import json, os, sys, time, traceback
from common import WORK, bag_dir, all_keys
from f1sim.bag_profile import report

keys = sys.argv[1:] or all_keys()
root = os.path.join(WORK, "profiles")
os.makedirs(root, exist_ok=True)
for key in keys:
    out = os.path.join(root, key)
    if os.path.exists(os.path.join(out, "bag_profile.json")):
        print(key, "exists", flush=True); continue
    t = time.time()
    try:
        report.run(bag_dir(key), out=out, name=key, log=lambda *a: print("  ", *a, flush=True))
        print(key, f"ok {time.time() - t:.0f}s", flush=True)
    except Exception as e:
        print(key, "FAILED", e, flush=True)
        traceback.print_exc()
print("ALLDONE", flush=True)
