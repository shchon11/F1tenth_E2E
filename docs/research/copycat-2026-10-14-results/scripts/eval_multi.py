"""Closed-loop solo trials of several checkpoints on one track, one process (the compile is paid once).
usage: python eval_multi.py OUT.json TRACK CKPT [CKPT ...]"""
import json
import sys
import time

from f1sim.learn import common
from f1sim.learn.evaluate import evaluate
from f1sim.params import Config

out, track, ckpts = sys.argv[1], sys.argv[2], sys.argv[3:]
res = {}
for ck in ckpts:
    t0 = time.time()
    r = evaluate(ck, [track], 128, 2400, 8.0, "cuda", seed=123, cfg=Config(), protocol="trials", budget_laps=3.0)
    r["wall_s"] = time.time() - t0
    res[ck] = r
    keep = {k: r.get(k) for k in ("completions", "collisions", "completion_rate", "distance_m", "collisions_per_km",
                                  "mean_speed", "lap_time_s", "spin_events", "wall_s")}
    print(ck, json.dumps(keep), flush=True)
json.dump(res, open(out, "w"), indent=1, default=str)
