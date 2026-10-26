"""Aggregate verify_guard.sh / verify_alat.sh outputs: per arm (name prefix), over its seeds/kinds: lap, mean speed, top
speed (p99 of the car's speed), command at the profile's 9+ m/s stretch, contacts per km (hard >= 1.5 m/s and all) by kind,
passes per minute in races. usage: agg_verify.py OUT_DIR PREFIX..."""
import glob
import json
import os
import sys

import numpy as np

D = sys.argv[1]
for pre in sys.argv[2:]:
    files = sorted(f for f in glob.glob(os.path.join(D, pre + "*.json")) if not f.endswith(".bins.json"))
    if not files:
        print(f"{pre}: no files"); continue
    km = 0.0; hard = {"wall": 0, "car": 0, "prop": 0}; allc = {"wall": 0, "car": 0, "prop": 0}
    laps, sp, mins, passes = [], [], 0.0, 0
    top, cmd9 = [], []
    for f in files:
        n = json.load(open(f))["nominal"]
        cs = n.get("contact_severity") or {}
        km += float(cs.get("learner_distance_m", 0)) / 1000
        for k in hard:
            fh = (cs.get(k) or {}).get("fine_hist") or []
            hard[k] += sum(fh[15:]); allc[k] += sum(fh)
        if n.get("lap_time_s") == n.get("lap_time_s") and n.get("lap_time_s"):
            laps.append(float(n["lap_time_s"]))
        sp.append(float(n["mean_speed"]))
        tr = n.get("traffic") or {}
        mins += float(tr.get("learner_minutes") or 0); passes += int(tr.get("passes") or 0)
        npy = f.replace(".json", ".bins.npy")
        if os.path.exists(npy):
            R = np.load(npy)
            top.append(np.percentile(R[:, 7], 99))
            m = R[:, 0] >= 9.0
            if m.any():
                cmd9.append(R[m, 6].mean())
    lap = f"{np.mean(laps):.2f} s" if laps else "-"
    pz = f" | passes/min {passes / mins:.2f}" if mins > 0 else ""
    print(f"{pre:24s} n {len(files)} | km {km:5.1f} | lap {lap} | speed {np.mean(sp):.2f} | top p99 {np.mean(top) if top else float('nan'):.2f} "
          f"| cmd@line9+ {np.mean(cmd9) if cmd9 else float('nan'):.2f} | hard/km wall {hard['wall'] / km:.2f} car {hard['car'] / km:.2f} "
          f"prop {hard['prop'] / km:.2f} | all/km wall {allc['wall'] / km:.2f} car {allc['car'] / km:.2f} prop {allc['prop'] / km:.2f}{pz}")
