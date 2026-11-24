"""(11-24) --dist-world 학습에서 모든 process가 같은 collective를 같은 순서로 부르나 -- race32a가 u646에서 멈춘 일의 확인.

race32a 인자 그대로(경주 세계, 복사본, 장면, 예측 머리, near miss, noise probe) plan_rl.main을 한 process로 돌리되 차 수와
update 수, 롤아웃 길이만 작게. torch.distributed의 all_reduce / broadcast를 감싸 부를 때마다 (update, 종류, 원소 수, dtype)를
기록해 json으로 남김. 롤아웃 길이 배수 K(--horizon-auto)는 잡음 크기가 유한해야 움직이는데 짧은 시험에서는 그러지 않으므로,
K를 update마다 주어진 순서대로 강제(horizon_k 대신)해 K가 바뀌는 update를 지나게 함.
인자: args.json world.json init.pt rank envs updates port K순서(예 2,2,4,4,2) out.json
      비교: cmp r0.json r1.json
"""
import argparse
import json
import math
import os
import sys
import time

if sys.argv[1] == "cmp":
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    n = min(len(a), len(b))
    bad = next((i for i in range(n) if a[i] != b[i]), None)
    ups = sorted(set(x[0] for x in a))
    print(f"[dist] rank 0: collective {len(a)}개, rank 1: {len(b)}개, update {ups[0] if ups else '-'}..{ups[-1] if ups else '-'}")
    if bad is None and len(a) == len(b):
        per = {u: sum(1 for x in a if x[0] == u) for u in ups}
        print(f"[dist] 두 rank의 collective 순서가 처음부터 끝까지 같음 (종류, 원소 수, dtype). update별 개수: {per}")
    else:
        i = bad if bad is not None else n
        print(f"[dist] {i}번째에서 갈림: rank0 {a[i] if i < len(a) else '-'} / rank1 {b[i] if i < len(b) else '-'}")
    raise SystemExit(0)

import torch
import torch.distributed as D

ARGS_JSON, WORLD, INIT = sys.argv[1], sys.argv[2], sys.argv[3]
RANK, ENVS, UPD, PORT = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
SCHED = [int(x) for x in sys.argv[8].split(",")]
OUT = sys.argv[9]

from f1sim.learn import plan_rl as PR

saved = json.load(open(ARGS_JSON))["args"]
name = os.environ.get("DIST_CHECK_NAME", f"racefast_dist_{time.strftime('%m%d_%H%M')}")
off = int(saved["update_offset"])
over = dict(name=name, world=WORLD, init=INIT, envs=ENVS, dist_world=2, dist_rank=RANK, dist_init=f"tcp://127.0.0.1:{PORT}",
            horizon=int(os.environ.get("DIST_CHECK_HORIZON", "16")), updates=off + UPD, save_every=10 ** 6, live_tap=-1,
            live_taps=0)
_orig_pa = argparse.ArgumentParser.parse_args


def _pa(self, argv=None, namespace=None):
    a = _orig_pa(self, ["--name", name])
    for k, v in saved.items():
        if hasattr(a, k):
            setattr(a, k, v)
    for k, v in over.items():
        setattr(a, k, v)
    return a


argparse.ArgumentParser.parse_args = _pa

LOG = []
CUR = {"upd": off}


def dump():
    json.dump(LOG, open(OUT, "w"))


def wrap(fn, kind):
    def w(t, *a, **kw):
        LOG.append([CUR["upd"], kind, int(t.numel()), str(t.dtype)])
        if len(LOG) % 50 == 0:
            dump()
        return fn(t, *a, **kw)
    return w


D.all_reduce = wrap(D.all_reduce, "all_reduce")
D.broadcast = wrap(D.broadcast, "broadcast")


class _Math:
    """`math` for plan_rl, but the noise scale always counts as measured: the horizon gate is then always passed"""

    def __getattr__(self, n):
        return getattr(math, n)

    @staticmethod
    def isfinite(x):
        return True


PR.math = _Math()
_calls = [0]


def _hk(k, k_low, *a, **kw):
    i = _calls[0]; _calls[0] += 1
    CUR["upd"] = off + 1 + i
    K = SCHED[min(i, len(SCHED) - 1)]
    print(f"[dist] rank {RANK}: update {off + 1 + i}: K = {K} (horizon {over['horizon'] * K}, minibatches {saved['minibatches'] * K})",
          flush=True)
    return K, 0


PR.horizon_k = _hk
try:
    PR.main()
finally:
    dump()
    print(f"[dist] rank {RANK}: collective {len(LOG)}개 기록 -> {OUT}", flush=True)
