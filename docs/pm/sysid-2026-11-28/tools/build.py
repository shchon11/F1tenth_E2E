"""Every bag -> 100 Hz grid + replay windows + calibration record (scratch cache). bag_profile's own loader."""
import json
import pickle
import sys
import time

import numpy as np

from common import WORK, SPLIT, FALLBACK_SERVO, bag_dir, car_of, all_keys
from f1sim.bag_profile import signals as SG, windows as WN

out = {}
for key in all_keys():
    t = time.time()
    try:
        bag, roles, missing, c, S = SG.load(bag_dir(key), fallback_servo=FALLBACK_SERVO[car_of(key)])
    except Exception as e:
        print(key, "skip:", e, flush=True)
        continue
    W = WN.windows_from_grid(S, key, c.pose_ok)
    S.pop("_raw", None)
    np.savez_compressed(f"{WORK}/grid_{key}.npz", **{k: v for k, v in S.items() if isinstance(v, np.ndarray)})
    out[key] = dict(W=W, calib=c.__dict__, missing=missing, roles=roles)
    print(f"{key} windows {len(W['r']):4d} impact-excluded {W['n_impact_excluded']:3d} pose_ok {c.pose_ok} "
          f"{time.time() - t:.1f}s", flush=True)
pickle.dump(out, open(f"{WORK}/windows.pkl", "wb"))
for car, sp in SPLIT.items():
    for name, keys in sp.items():
        print(car, name, sum(len(out[k]["W"]["r"]) for k in keys if k in out))
