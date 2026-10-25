"""Research (2026-10-25): what would the speed limit for a car ahead do on the real venue runs -- the shipped rule on the shipped
readings (`_nearcar`, no latch) against the new one (`_nearcar_slow` readings, latched lane check), scan by scan with the car's
own speed (/odom)? Reports how often the limit sits below the car's speed and by how much, split by what the reading was on
(a moving object, a fixed object, nothing found) on the well-localised scans (real_bag_head_check.py's classes).
usage: real_bag_limit_replay.py OBJS.npz READ_OLD.npz READ_NEW.npz"""
import math
import sys

import numpy as np
import torch

from f1sim.opponent_speed_limit import speed_limit

OBJS, RO, RN = sys.argv[1:4]
zo = np.load(OBJS); obj, scan, od = zo["obj"], zo["scan"], zo["odom"]
cls = np.load(OBJS.replace(".npz", "_cls.npy"))
ts = scan[:, 0]
by_scan = {}
for n, r in enumerate(obj):
    by_scan.setdefault(int(r[0]), []).append(n)


def run(path, latch):
    z = np.load(path); st, rd = z["stamp"], z["reading"]
    v = np.interp(st, od[:, 0], od[:, 1])
    held = False; lims = np.full(len(st), np.inf)
    for i in range(len(st)):
        h = torch.tensor([held]) if latch else None
        lim = float(speed_limit(torch.as_tensor(rd[i:i + 1], dtype=torch.float32), torch.tensor([float(v[i])]), held=h)[0])
        held = math.isfinite(lim)
        lims[i] = lim
    slow = np.isfinite(lims) & (lims < v)
    # what the reading sat on, where the scan is well localised
    k = np.searchsorted(ts, st).clip(1, len(ts) - 1)
    k = np.where(np.abs(ts[k - 1] - st) < np.abs(ts[k] - st), k - 1, k)
    m = np.abs(ts[k] - st) < 0.02
    kind = np.full(len(st), -1)
    for i in np.where(slow & m)[0]:
        hx, hy = rd[i, 1] * 5, rd[i, 2] * 2
        best = None
        for n in by_scan.get(int(k[i]), []):
            d = math.hypot(obj[n, 2] - hx, obj[n, 3] - hy)
            if d < 0.6 and (best is None or d < best[0]):
                best = (d, n)
        kind[i] = 0 if best is None else (1 if cls[best[1]] == 1 else 2 if cls[best[1]] == 2 else 3)
    km = slow & m
    cut = (v - lims)[slow]
    return (100 * slow.mean(), float(np.mean(cut)) if slow.any() else 0.0, int(km.sum()),
            [100 * np.mean(kind[km] == c) if km.any() else 0 for c in (1, 2, 3, 0)])


for tag, path, latch in (("shipped (_nearcar, no latch)", RO, False), ("new (_nearcar_slow + latch)", RN, True)):
    p, cut, n, kinds = run(path, latch)
    print(f"{tag:30s}: limit below the car's speed {p:.1f} % of scans, mean cut {cut:.2f} m/s | on well-localised scans ({n}): "
          f"moving object {kinds[0]:.0f} %, fixed object {kinds[1]:.0f} %, other object {kinds[2]:.0f} %, nothing {kinds[3]:.0f} %")
