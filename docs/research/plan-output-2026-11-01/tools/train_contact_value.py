"""(11-28) Does the critic see a car contact coming, in TRAINING? From the live-tap logs of a run (ace_side/live_tap_log.py's
"t<TAB>k<TAB>json" lines, the 8 tapped cars, every decision): every car-contact onset (hit[2] rising) of a tapped car, the
critic's value V over the 3 s before and 0.5 s after (same update), split by kind (the other car ahead / alongside / behind
in our frame) and by ordinary vs replayed episode. Also how often the situation occurs: 1v1 decisions closing > 1 m/s on a car
0.5-5 m ahead. usage: train_contact_value.py LOG.tsv [...]"""
import collections
import json
import sys

import numpy as np

seq = collections.defaultdict(dict)
for fn in sys.argv[1:]:
    for ln in open(fn):
        try:
            _t, k, b = ln.split("\t", 2); nw = json.loads(b)["now"]
        except Exception:
            continue
        seq[int(k)][(int(nw.get("upd", 0)), int(nw.get("k", 0)))] = nw
tot = opp_dec = close_dec = 0; rows = []
for k, d in seq.items():
    keys = sorted(d); fr = [d[x] for x in keys]
    for i, x in enumerate(fr):
        tot += 1
        o, c = x.get("opp"), x.get("car")
        if o is None or c is None:
            continue
        opp_dec += 1
        dx = (o[0] - c[0]) * np.cos(c[2]) + (o[1] - c[1]) * np.sin(c[2]); dy = -(o[0] - c[0]) * np.sin(c[2]) + (o[1] - c[1]) * np.cos(c[2])
        if 0.5 < dx < 5 and abs(dy) < 1.0 and c[3] - o[3] > 1.0:
            close_dec += 1
        h = x.get("hit") or [0, 0, 0]
        if h[2] > 0 and (i == 0 or (fr[i - 1].get("hit") or [0, 0, 0])[2] == 0):
            kind = "side" if abs(dx) <= 0.3 else ("rear" if dx > 0.3 else "behind")
            def V(j):
                if 0 <= j < len(fr) and fr[j].get("upd") == x.get("upd") and fr[j].get("value") is not None and abs(keys[j][1] - keys[i][1]) == abs(j - i):
                    return float(fr[j]["value"])
                return np.nan
            rows.append((kind, bool(x.get("replay")), [V(i + o_) for o_ in (-30, -20, -10, -5, -2, -1, 0, 1, 3, 5)], c[3] - o[3]))
print(f"decisions {tot}, with a car {opp_dec} ({100 * opp_dec / max(1, tot):.0f} %), closing > 1 m/s on a car 0.5-5 m ahead {close_dec} "
      f"({100 * close_dec / max(1, opp_dec):.1f} % of those) | car contact onsets {len(rows)}")
lab = "-3/-2/-1/-0.5/-0.2/-0.1/0/+0.1/+0.3/+0.5 s"
for kind in ("rear", "side", "behind"):
    for rp in (False, True):
        r = [v for v in rows if v[0] == kind and v[1] == rp]
        if not r:
            continue
        vv = np.array([v[2] for v in r], float)
        ref = vv[:, 0:1]
        print(f"  {kind:<6} {'replay' if rp else 'ordinary':<8} n {len(r):3d} | V {lab}: {np.round(np.nanmedian(vv, 0), 2).tolist()} | "
              f"V - V(-3 s): {np.round(np.nanmedian(vv - ref, 0), 2).tolist()} | closing {np.median([v[3] for v in r]):.2f} m/s")
