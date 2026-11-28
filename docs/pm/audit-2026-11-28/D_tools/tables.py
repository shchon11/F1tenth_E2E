"""Error tables from preds.pkl: per split x model x horizon, median [p90] over windows; subsets by lateral load."""
import json
import pickle

import numpy as np

import models as M
from windows import WARM

Wa = pickle.load(open("windows.pkl", "rb"))
preds = pickle.load(open("preds.pkl", "rb"))
MODELS = ["P", "K0", "K1", "K2", "S0", "S1"]
OUT = {}


def summarize(W, pr, idx):
    e = M.errors(W, pr, idx)
    return {h: {k: (float(np.nanmedian(v)), float(np.nanpercentile(v, 90)), float(np.sqrt(np.nanmean(v ** 2)))) for k, v in d.items()}
            for h, d in e.items()}


for split, W in Wa.items():
    n = len(W["r"])
    ay1 = np.abs(W["ay"][:, WARM:WARM + 101]).max(1)
    subsets = {"all": np.arange(n), "ay<4": np.flatnonzero(ay1 < 4), "ay4-7": np.flatnonzero((ay1 >= 4) & (ay1 < 7)),
               "ay>=7": np.flatnonzero(ay1 >= 7)}
    OUT[split] = {}
    for sname, idx in subsets.items():
        if len(idx) < 10:
            continue
        OUT[split][sname] = {"n": int(len(idx))}
        for m in MODELS:
            pr = {k: v[idx] for k, v in preds[split][m].items()}
            OUT[split][sname][m] = summarize(W, pr, idx)
json.dump(OUT, open("tables.json", "w"), indent=1)

for split in OUT:
    for sname, d in OUT[split].items():
        print(f"\n=== {split} / {sname}  n={d['n']}")
        for h in (50, 100, 200):
            print(f"  h={h/100:.1f}s   " + "  ".join(f"{m}: yaw {d[m][h]['yaw'][0]:.3f} hd {d[m][h]['head'][0]:.3f} v {d[m][h]['v'][0]:.3f} ay {d[m][h]['ay'][0]:.2f} pos {d[m][h].get('pos', (np.nan,))[0]:.2f}" for m in MODELS))
