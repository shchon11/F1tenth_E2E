"""Aggregate finals race cells (10-26): per row, hard (>= 1.5 m/s) contacts per km by kind, race speed, passes per minute, and
-- where the file has it (evaluate since 5535993) -- the share of learner steps each speed limit held the command.
Rows are compared only on the cells present in every row of a group (same seeds and kinds).
usage: agg_obs.py OUT_DIR "label=dir[,dir...]" ["label=dir"...]   (dirs relative to OUT_DIR)"""
import glob
import json
import os
import sys

OUT = sys.argv[1]
rows = []
for arg in sys.argv[2:]:
    label, dirs = arg.split("=", 1)
    files = {}
    for d in dirs.split(","):
        for f in glob.glob(os.path.join(OUT, d, "race_s*.json")):
            files.setdefault(os.path.basename(f), f)
    rows.append((label, files))
common = set.intersection(*[set(f) for _, f in rows]) if rows else set()


def agg(files):
    t = {"car": 0, "wall": 0, "prop": 0}
    km = mins = 0.0
    passes = 0
    sp = []
    held = {"car_share": [], "obstacle_share": []}
    for f in files:
        d = json.load(open(f))["nominal"]
        cs = d.get("contact_severity") or {}
        tr = d.get("traffic") or {}
        for k in t:
            t[k] += sum(((cs.get(k) or {}).get("fine_hist") or [])[15:])
        km += float(cs.get("learner_distance_m", 0)) / 1000
        sp.append(float(d["mean_speed"]))
        mins += float(tr.get("learner_minutes") or 0)
        passes += int(tr.get("passes") or 0)
        sl = d.get("speed_limits") or {}
        for k in held:
            if sl.get(k) is not None:
                held[k].append(float(sl[k]))
    return t, km, sum(sp) / max(len(sp), 1), passes / max(mins, 1e-6), {k: (sum(v) / len(v) if v else None) for k, v in held.items()}


print(f"{len(common)} common cells")
for label, files in rows:
    t, km, sp, ps, held = agg([files[n] for n in sorted(common)])
    tot = t["car"] + t["wall"] + t["prop"]
    h = " | ".join(f"{k.split('_')[0]} held {100 * v:.1f} %" for k, v in held.items() if v is not None)
    print(f"{label:34s} km {km:5.1f} | speed {sp:.2f} | hard car {t['car'] / km:.2f} prop {t['prop'] / km:.2f} "
          f"wall {t['wall'] / km:.2f} | total {tot / km:.2f} +- {tot ** 0.5 / km:.2f} | passes {ps:.2f}" + (f" | {h}" if h else ""))
