"""Aggregate oracle-shield race cells against the entry's no-shield and knob cells (seeds 81-84, 16 cells each).
usage: shield_compare.py OUT_ROOT"""
import glob
import json
import sys

R = sys.argv[1]


def agg(files):
    hc = hw = allc = 0; km = 0.0; sp = []; mins = 0.0; passes = 0
    for f in files:
        d = json.load(open(f))["nominal"]; cs = d.get("contact_severity") or {}; tr = d.get("traffic") or {}
        hc += sum(((cs.get("car") or {}).get("fine_hist") or [])[15:]); hw += sum(((cs.get("wall_prop") or {}).get("fine_hist") or [])[15:])
        allc += sum((cs.get("car") or {}).get("fine_hist") or [])
        km += float(cs.get("learner_distance_m", 0)) / 1000; sp.append(float(d["mean_speed"]))
        mins += float(tr.get("learner_minutes") or 0); passes += int(tr.get("passes") or 0)
    return len(files), km, hc, hw, allc, sum(sp) / max(len(sp), 1), passes / max(mins, 1e-6)


rows = [("no shield", glob.glob(f"{R}/anat_v2entry/race_s*.json") + glob.glob(f"{R}/xseeds/v2entry_k100/race_s*.json")),
        ("in-lane, ego frame", glob.glob(f"{R}/shield2_v2entry_a3_w035/race_s*.json"))]
for d in sorted(glob.glob(f"{R}/shieldpred_entry_*/")):
    rows.append((d.rstrip("/").split("shieldpred_entry_")[-1], glob.glob(f"{d}race_s*.json")))
rows += [(f"knob {k}", glob.glob(f"{R}/knob/v2entry_k{k}/race_s*.json")) for k in ("090", "085", "080")]
out = []
for tag, files in rows:
    n, km, hc, hw, allc, sp, ps = agg(files)
    if not n:
        continue
    out.append(dict(tag=tag, cells=n, km=km, race=sp, hard_car=hc / km, hard_car_se=hc ** 0.5 / km, hard_wall_prop=hw / km, all_car=allc / km, passes=ps))
    print(f"{tag:24s} cells {n:2d} km {km:5.1f} | race {sp:.2f} | hard car {hc/km:.2f} +- {hc**0.5/km:.2f} | hard wall+prop {hw/km:.2f} | all car {allc/km:.1f} | passes {ps:.2f}/min")
for f in sorted(glob.glob(f"{R}/shieldpred_entry_*/shield_stats_*.json")):
    s = json.load(open(f))
    print(f.split("/")[-2], f.split("_")[-1][:-5], f"active {100 * s['active'] / max(s['steps'], 1):.2f} % (prediction only {100 * s.get('pred_only', 0) / max(s['steps'], 1):.2f} %), mean cut {s['cut_sum'] / max(s['active'], 1):.2f} m/s")
json.dump(out, open(f"{R}/shield_compare.json", "w"), indent=1)
