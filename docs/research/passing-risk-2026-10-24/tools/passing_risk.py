"""Research (2026-10-24): where in a pass does a hard car contact come from? Offline, from the close-situation data of a
policy's race cells (contact_probe.py's probe_data.pt: every learner step with the opponent within 3 m, the TRUE relative
state [distance, dx, dy (from the LiDAR mount), closing speed, lateral relative speed, v_me, v_opp] and whether a hard
(>= 1.5 m/s) car-contact onset follows within 0.5 s).
Reports: (1) the geometry of the pre-contact samples (wholly ahead / overlapping, lateral offset, lateral rate toward us,
speed advantage); (2) P(hard contact within 0.5 s) over lateral offset x speed advantage while passing (opponent ahead, we
faster by > 0.3 m/s); (3) how much of the pre-contact samples, and of the other close samples, a shield rule would cap:
in-lane (|dy| < 0.35, opponent wholly ahead) with a constant-velocity lateral prediction over H, and pass-slow
(|dy| < W, closing faster than DV).
usage: passing_risk.py PROBE_DATA.pt OUT.json"""
import json
import sys

import torch

d = torch.load(sys.argv[1])
GT = torch.cat([v["gt"] for v in d.values()]); Y = torch.cat([v["y"] for v in d.values()]).bool()
dist, lx, ly, closing, dvy, vme, vop = GT.T
MX, CAR_LEN = 0.27, 0.5                           # LiDAR mount ahead of the reference point; body length
dx = lx + MX; gap = dx - CAR_LEN; dv = vme - vop; a = ly.abs()
toward = dvy * torch.sign(-ly)                     # lateral rate of the opponent toward us (+)
out = {"samples": int(Y.numel()), "pre_contact": int(Y.sum())}
g = {}
for name, m in (("wholly_ahead", gap > 0.1), ("overlapping", (gap <= 0.1) & (gap > -1.0)), ("behind", gap <= -1.0)):
    mm = m & Y
    g[name] = {"share": float(mm.float().sum() / Y.float().sum()), "abs_dy": float(a[mm].mean()) if mm.any() else None,
               "toward_mps": float(toward[mm].mean()) if mm.any() else None, "dv": float(dv[mm].mean()) if mm.any() else None}
out["pre_contact_geometry"] = g
P = (dx > 0) & (dv > 0.3)
dyb = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.2), (1.2, 3.0)]
dvb = [(0.3, 1.0), (1.0, 2.0), (2.0, 9.0)]
tab = []
for lo, hi in dyb:
    for vl, vh in dvb:
        m = P & (a >= lo) & (a < hi) & (dv >= vl) & (dv < vh)
        tab.append({"dy": [lo, hi], "dv": [vl, vh], "n": int(m.sum()), "p_hard_pct": 100 * float((m & Y).sum()) / max(int(m.sum()), 1)})
out["passing_risk"] = tab
out["passing_time_share_by_dy"] = {f"{lo}-{hi}": float((P & (a >= lo) & (a < hi)).float().sum() / P.float().sum()) for lo, hi in dyb}
rules = []
for H in (0.0, 0.25, 0.5, 0.75, 1.0):
    W = 0.35
    t_in = torch.where(a < W, torch.zeros_like(ly), torch.where((ly * dvy) < 0, (a - W) / dvy.abs().clamp_min(1e-6), torch.full_like(ly, 1e9)))
    flag = (gap > 0.1) & (dx < 6) & (t_in <= H)
    room = (gap - 0.2 - dv.clamp_min(0) * 0.2).clamp_min(0)
    bind = flag & (vme > vop.clamp_min(0) + torch.sqrt(6.0 * room))
    rules.append({"rule": f"in-lane, CV lateral prediction H {H}", "binds_pre_contact": float(bind[Y].float().mean()), "binds_other_close": float(bind[~Y].float().mean())})
for W, DV in ((0.6, 1.0), (0.8, 1.0), (0.8, 0.5), (1.0, 1.0)):
    region = (dx > 0) & (a < W) & (dv > DV)
    rules.append({"rule": f"pass-slow W {W} DV {DV}", "binds_pre_contact": float(region[Y].float().mean()), "binds_other_close": float(region[~Y].float().mean())})
out["rules"] = rules
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out["pre_contact_geometry"], indent=1)); [print(r) for r in rules]
