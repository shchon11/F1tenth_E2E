"""Compare arms on the (seed, kind) cells every arm has: race_<arm>_s<seed>_<kind>.json. Totals per arm: km, speed, hard
contacts per km by kind, all contacts per km, passes per minute. usage: cmp_cells.py OUT_DIR ARM..."""
import glob
import json
import os
import re
import sys

D = sys.argv[1]
arms = sys.argv[2:]
cells = {}
for arm in arms:
    for f in glob.glob(os.path.join(D, f"race_{arm}_s*_*.json")):
        if f.endswith(".bins.json"):
            continue
        m = re.match(rf"race_{re.escape(arm)}_s(\d+)_(.+)\.json$", os.path.basename(f))
        if m:
            cells.setdefault(arm, {})[(m.group(1), m.group(2))] = f
common = set.intersection(*[set(c) for c in cells.values()]) if len(cells) == len(arms) else set()
print(f"{len(common)} common cells: {sorted(common)}")
for arm in arms:
    km = mins = 0.0; passes = 0; sp = []
    hard = {"wall": 0, "car": 0, "prop": 0}; allc = 0
    for c in sorted(common):
        n = json.load(open(cells[arm][c]))["nominal"]
        cs = n.get("contact_severity") or {}
        km += float(cs.get("learner_distance_m", 0)) / 1000
        for k in hard:
            fh = (cs.get(k) or {}).get("fine_hist") or []
            hard[k] += sum(fh[15:]); allc += sum(fh)
        tr = n.get("traffic") or {}
        mins += float(tr.get("learner_minutes") or 0); passes += int(tr.get("passes") or 0)
        sp.append(float(n["mean_speed"]))
    tot = sum(hard.values())
    print(f"{arm:8s} km {km:5.1f} | speed {sum(sp) / len(sp):.2f} | hard {tot} = {tot / km:.2f}/km (wall {hard['wall']}, car {hard['car']}, "
          f"prop {hard['prop']}) | all {allc / km:.2f}/km | passes/min {passes / max(mins, 1e-6):.2f}")
