import numpy as np
from common import WORK, PLACE
from f1sim.bag_profile import measure as MS
for place, keys in PLACE.items():
    for k in keys:
        Z = np.load(f"{WORK}/grid_{k}.npz"); S = {n: Z[n] for n in Z.files}
        la = MS.lateral_limit(S)
        mv = S["v_wheel"] > 1
        print(f"{place:14s} {k} vmax {S['v_wheel'].max():4.1f} moving {mv.sum()/100:5.0f}s ay_held {la['value']:5.2f} (top10 med {np.median(la['top10']) if la['top10'] else float('nan'):4.1f}) at v {la['v_at']:4.1f} slides {len(la['slides_t'])} {la['slides_t'][:4]}")
