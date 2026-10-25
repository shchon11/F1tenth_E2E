"""Research (2026-10-26), offline on car_contact_anatomy.py SCANS=1 output (with the plans): would a car limit that asks "is the
car on the path my PLAN is about to drive?" bind earlier, before the rear-end clips, than the shipped one that asks "is the car
within 0.35 m of my current heading line?"

In those clips the true sideways closing is 0.6-0.9 m/s in the last 0.35 s, mostly our own move, and the reading's v_lat reads ~0
(corr ~0), so the shipped test only sees the car in its lane at the end (0.2-0.3 s before the hit).
Plan-path test, per step: decode the plan (as the tracker does), walk its path (obstacle_speed_limit.path_points); the car is on
it when the path passes within LANE (0.35 m) of the car's centre -- read by the policy's reading (deployable) or the truth (the
ceiling); the gap is the arc length to that point minus a car length; the limit is the shipped formula with that gap.
usage: car_plan_lane.py ANAT_OUT_DIR"""
import json
import os
import sys

import torch

from f1sim import mpc
from f1sim.obstacle_speed_limit import path_points
from f1sim.opponent_speed_limit import DEFAULT as P, speed_limit

D = sys.argv[1]
E = json.load(open(os.path.join(D, "events.json")))
S = torch.load(os.path.join(D, "scans.pt"))
CK = "/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
spec = mpc.contract_spec(mpc.contract_of(ck.get("extra")))
from f1sim.learn.obs import spec_from_dict
v_max = float(spec_from_dict(dict((ck.get("extra") or {}).get("spec") or {})).v_max)
MX, LANE = 0.297, 0.35


def plan_lane_limit(k, L, cx, cy, v_me, closing, held_lane=LANE):
    """the shipped formula, with the gap along the plan's path to the point nearest the car, when the path passes within
    `held_lane` of the car's centre; +inf otherwise"""
    pts, s = path_points(k[None].float(), L[None].float())
    pts, s = pts[0], s
    valid = s <= float(L)
    d = torch.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    d = torch.where(valid, d, torch.full_like(d, 1e9))
    j = int(d.argmin())
    if float(d[j]) >= held_lane:
        return float("inf")
    gap = float(s[j]) - P.car_length
    if gap <= 0.1:
        return float("inf")
    v_opp = max(v_me - closing, 0.0)
    room = max(gap - P.gap_min - max(closing, 0.0) * P.lag, 0.0)
    return v_opp + (2.0 * P.brake * room) ** 0.5


out = {"heading (shipped)": [], "plan path, reading": [], "plan path, truth": []}
for rec in S:
    e = E[rec["event"]]
    if "act" not in rec:
        sys.exit("this anatomy output has no plans (run car_contact_anatomy.py SCANS=1 after 10-26 06:55)")
    m = rec["scans"].shape[0]
    rd, rel = rec["reading"].float(), rec["rel"]
    act, vp = rec["act"].float(), rec["v_plan"].float()
    cap = torch.full((m,), rec["cap"])
    k_all, L_all = mpc.decode_wp4(act, vp, v_max, cap, spec)[:2] if spec.speed_mode == "wp4" else mpc.decode(act, vp, v_max, cap, spec)[:2]
    v = rec["v_ego"]
    for name in out:
        held = torch.zeros(1, dtype=torch.bool)
        held_p = False
        bind = []
        for t in range(m - 1, -1, -1):                    # oldest first
            ex = bool(rd[t, 0] > 0)
            cx, cy = float(rd[t, 1]) * 5 + MX, float(rd[t, 2]) * 2
            closing = float(rd[t, 3]) * 5
            if name == "heading (shipped)":
                lim = speed_limit(rd[t:t + 1], torch.tensor([v[t]]), held=held)
                held = torch.isfinite(lim)
                lim = float(lim[0])
            else:
                if name == "plan path, truth":
                    ex, cx, cy = True, float(rel[t, 0]), float(rel[t, 1])
                lim = plan_lane_limit(k_all[t], L_all[t], cx, cy, v[t], closing, P.lane_exit if held_p else LANE) if ex else float("inf")
                held_p = lim < float("inf")
            bind.append(lim < v[t] - 0.1)
        first = next((m - 1 - kk for kk, b in enumerate(bind) if b), None)
        out[name].append(dict(cls=e["cls"], any=any(bind), lead=first))
for cls in ("ego_hits_rear", "side", "hit_from_behind", "turned"):
    for name, res in out.items():
        sub = [x for x in res if x["cls"] == cls]
        if sub:
            leads = [x["lead"] for x in sub if x["lead"] is not None]
            early = sum(1 for x in sub if x["lead"] is not None and x["lead"] >= 16)      # bound 0.4 s or more before
            print(f"{cls:16s} {name:20s}: binds in the last 0.5 s for {sum(x['any'] for x in sub):2d}/{len(sub)}, "
                  f">= 0.4 s before the hit {early:2d} | first bind {sum(leads) / len(leads) * 0.025 if leads else 0:.2f} s before (mean)")
