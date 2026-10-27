"""Summarise eval_ckpt.sh outputs: name=OUT_DIR/<name> ... -> solo lap, speed by raceline-profile band, straight command
(profile >= 9 m/s), corner |v r| p90 (profile < 6), solo hard contacts; races (4 kinds x seeds 81-82): km, speed, hard /km by kind."""
import glob
import json
import os
import sys

import numpy as np

print("이름 | 혼자 랩 s | 직선 명령 (주행선 9+) | 직선 속도 | 코너 |v r| p90 | 혼자 강한/km || 경주 km | 경주 속도 | 강한/km (벽/상자/차)")
for arg in sys.argv[1:]:
    name, d = arg.split("=", 1)
    out = [name]
    sj = os.path.join(d, "solo.json")
    if os.path.exists(sj):
        n = json.load(open(sj))["nominal"]
        lap = n.get("lap_time_s") or n.get("mean_lap_time_s") or n.get("lap_time")
        R = np.load(os.path.join(d, "solo.bins.npy"))
        st = R[:, 0] >= 9.0
        co = R[:, 0] < 6.0
        cs = n.get("contact_severity") or {}
        km = float(cs.get("learner_distance_m", 0)) / 1000
        hard = sum(sum(((cs.get(k) or {}).get("fine_hist") or [])[15:]) for k in ("wall", "car", "prop"))
        out += [f"{lap:.2f}" if isinstance(lap, (int, float)) else str(lap), f"{R[st, 6].mean():.2f}", f"{R[st, 7].mean():.2f}",
                f"{np.percentile(np.abs(R[co, 10]), 90):.2f}" if R.shape[1] > 10 else "-", f"{hard / max(km, 1e-6):.2f}"]
    else:
        out += ["-"] * 5
    km = 0.0; sp = []; hard = {"wall": 0, "prop": 0, "car": 0}
    for f in sorted(glob.glob(os.path.join(d, "race_s*_*.json"))):
        n = json.load(open(f))["nominal"]
        cs = n.get("contact_severity") or {}
        km += float(cs.get("learner_distance_m", 0)) / 1000
        for k in hard:
            hard[k] += sum(((cs.get(k) or {}).get("fine_hist") or [])[15:])
        sp.append(float(n["mean_speed"]))
    if sp:
        tot = sum(hard.values())
        out += [f"{km:.1f}", f"{np.mean(sp):.2f}", f"{tot / km:.2f} ({hard['wall'] / km:.2f}/{hard['prop'] / km:.2f}/{hard['car'] / km:.2f})"]
    print(" | ".join(out))
