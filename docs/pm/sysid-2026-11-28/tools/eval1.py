"""Task 1 step 3: every candidate on a split (val for selection; final only once). Usage: eval1.py comp val [models...]"""
import json, pickle, sys, time
import numpy as np, torch
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN
import evalcore as E

torch.set_num_threads(2)
car, split = sys.argv[1], sys.argv[2]
want = sys.argv[3:] or None
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT[car][split] if k in D and len(D[k]["W"]["r"])])
R = json.load(open(f"{WORK}/fit1_{car}.json"))
spd = R["speed"]["p"]
K2 = dict(tau=0.10, k=1.15, b=-0.03, kus=0.02, tau_v=0.1, tau_m=0.15, a_a=4, a_b=4, v_s=10)   # audit fitted.json
cands = {
    "지속": lambda: M.persistence(W),
    "(a) 선형 자전거+지연, 감사 값": lambda: M.lin_rollout(W, K2),
    "(a) 선형 자전거+지연, 재맞춤": lambda: M.lin_rollout(W, R["lin"]["p"]),
    "(b) 단일트랙+포화 타이어 (C 자유)": lambda: M.st_rollout(W, R["st"]["p"]),
    "(b') C 1.3 고정": lambda: M.st_rollout(W, R["st_C13"]["p"]) if "st_C13" in R else None,
    "(b'') C 1.0 고정 (tanh형)": lambda: M.st_rollout(W, R["st_C10"]["p"]) if "st_C10" in R else None,
    "(d) (b)+종방향 하중 이동": lambda: M.st_lt_rollout(W, R["st_lt"]["p"]) if "st_lt" in R else None,
    "시뮬 명목": lambda: M.sim_rollout(W),
    "시뮬+조향 이득·트림 맞춤 (감사 S1)": lambda: M.sim_rollout(W, over={"steer_gain": 0.9, "steer_bias": -0.02}),
}
extra = {}
try:
    extra = json.load(open(f"{WORK}/fit_sim_{car}.json"))
except FileNotFoundError:
    pass
if "over" in extra:
    cands["(c1) 시뮬 구조 재맞춤 (h 0.074)"] = lambda: M.sim_rollout(W, over=extra["over"])
for fn_, tag in ((f"{WORK}/fit_sim2_{car}.json", "(c2) 시뮬 구조, 종방향 하중이동 끔 (h 0)"),
                 (f"{WORK}/fit_sim3_{car}.json", "(c2*) (c2) + 마찰 앞뒤 공통·하한값")):
    try:
        ov = json.load(open(fn_))["over"]
        cands[tag] = (lambda ov=ov: M.sim_rollout(W, over=ov))
    except FileNotFoundError:
        pass
preds = {}
t0 = time.time()
for nm, fn in cands.items():
    if want and not any(w in nm for w in want):
        continue
    pr = fn()
    if pr is None:
        continue
    preds[nm] = pr
    print(nm, f"{time.time() - t0:.0f}s", flush=True)
T = E.table(W, preds)
json.dump(T, open(f"{WORK}/eval_{car}_{split}.json", "w"), indent=1, ensure_ascii=False)
print(E.md(T, f"{car} / {split}"))
