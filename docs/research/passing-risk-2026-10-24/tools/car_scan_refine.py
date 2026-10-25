"""Research (2026-10-26), offline on car_contact_anatomy.py SCANS=1 output: does the scan cluster at the policy's nearest-car reading
say where the car is better than the reading does, in the last 0.5 s before a hard car contact?

The car limit's lane test reads the reading's lateral offset dy; at close range it drifts outward (true |dy| 0.30-0.36 m at the hit,
read 0.5-0.7), so the limit lets go of a car we are about to clip. Refinement: take the scan segment (the cluster rule's
segmentation: range jumps over 0.15 m and missed returns split it) with a point nearest the reading's position (within 0.8 m), and
estimate the car's centre from its points:
  mid   -- lateral: the middle of the segment's lateral extent; longitudinal: its nearest point + half a car length
  cent  -- the segment's centroid
Compared with the TRUE centre (the partner's state point in the ego frame), per step of the last 0.5 s.
usage: car_scan_refine.py ANAT_OUT_DIR"""
import json
import math
import os
import sys

import torch

D = sys.argv[1]
E = json.load(open(os.path.join(D, "events.json")))
S = torch.load(os.path.join(D, "scans.pt"))
MX, N = 0.297, 1081
ANG = torch.linspace(-2.356194490192345, 2.356194490192345, N)
HALF_LEN, LANE, EXIT = 0.25, 0.35, 0.5


def segments(r):
    ok = torch.isfinite(r) & (r > 0.05) & (r < 9.99)
    rf = torch.where(ok, r, torch.full_like(r, 30.0))
    brk = torch.cat([torch.ones(1, dtype=torch.bool), ((rf[1:] - rf[:-1]).abs() > 0.15) | ~ok[1:] | ~ok[:-1]])
    return torch.cumsum(brk.long(), 0), ok


rows = []
for rec in S:
    e = E[rec["event"]]
    scans, rel, rd = rec["scans"].float(), rec["rel"], rec["reading"]
    for t in range(scans.shape[0]):
        if rd[t, 0] <= 0:
            continue
        cx, cy = float(rd[t, 1]) * 5 + MX, float(rd[t, 2]) * 2
        r = scans[t]
        seg, ok = segments(r)
        px, py = MX + r * torch.cos(ANG), r * torch.sin(ANG)
        d = torch.where(ok, torch.hypot(px - cx, py - cy), torch.full_like(r, 1e9))
        i = int(d.argmin())
        if float(d[i]) > 0.8:
            continue
        m = (seg == seg[i]) & ok
        xs, ys = px[m], py[m]
        mid = (float(xs.min()) + HALF_LEN, 0.5 * (float(ys.min()) + float(ys.max())))
        cent = (float(xs.mean()), float(ys.mean()))
        rows.append(dict(cls=e["cls"], t=t, true=(float(rel[t, 0]), float(rel[t, 1])), read=(cx, cy), mid=mid, cent=cent,
                         n=int(m.sum())))
print(f"{len(S)} events, {len(rows)} (event, step) pairs with a reading and a segment at it")
for cls in ("ego_hits_rear", "side", "hit_from_behind", "turned", "all"):
    sub = [x for x in rows if (cls == "all" or x["cls"] == cls) and x["t"] < 12]
    if not sub:
        continue
    err = lambda k: sum(abs(x[k][1] - x["true"][1]) for x in sub) / len(sub)
    errx = lambda k: sum(abs(x[k][0] - x["true"][0]) for x in sub) / len(sub)
    lane_true = [abs(x["true"][1]) < LANE for x in sub]
    agree = lambda k: sum((abs(x[k][1]) < LANE) == lt for x, lt in zip(sub, lane_true)) / len(sub)
    missed = lambda k: sum(1 for x, lt in zip(sub, lane_true) if lt and abs(x[k][1]) >= EXIT) / max(sum(lane_true), 1)
    print(f"{cls:16s} n {len(sub):4d} (last 0.3 s) | |dy| error: reading {err('read'):.3f} mid {err('mid'):.3f} cent {err('cent'):.3f} | "
          f"|dx| error: reading {errx('read'):.3f} mid {errx('mid'):.3f} | in-lane agreement: reading {agree('read'):.2f} "
          f"mid {agree('mid'):.2f} | truly in lane but read beyond {EXIT}: reading {missed('read'):.2f} mid {missed('mid'):.2f}")


# ---- the car limit's decision along each event's last 0.5 s: with the reading as it was, and with the scan-refined one
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine_proto import refine
from f1sim.opponent_speed_limit import speed_limit
res = {"read": [], "refined": []}
for rec in S:
    e = E[rec["event"]]
    m = rec["scans"].shape[0]
    ref, found = refine(rec["reading"], rec["scans"].float(), ANG, range_max=10.0)
    v = torch.tensor(e["v_ego_hist"][:m], dtype=torch.float32)
    for name, rd in (("read", rec["reading"].float()), ("refined", ref)):
        held = torch.zeros(1, dtype=torch.bool)
        bind = []
        for t in range(m - 1, -1, -1):                    # oldest first, with the latch carried
            lim = speed_limit(rd[t:t + 1], v[t:t + 1], held=held)
            held = torch.isfinite(lim)
            bind.append(bool(lim[0] < v[t] - 0.1))
        first = next((m - 1 - k for k, b in enumerate(bind) if b), None)   # steps before the hit of the first binding step
        res[name].append(dict(cls=e["cls"], any=any(bind), lead=first))
for cls in ("ego_hits_rear", "side", "hit_from_behind", "turned"):
    for name in ("read", "refined"):
        sub = [x for x in res[name] if x["cls"] == cls]
        if sub:
            leads = [x["lead"] for x in sub if x["lead"] is not None]
            print(f"{cls:16s} {name:8s}: the limit would bind in the last 0.5 s for {sum(x['any'] for x in sub)}/{len(sub)} | "
                  f"first bind {sum(leads) / len(leads) * 0.025 if leads else 0:.2f} s before the hit (mean)")
