"""(11-25) plan_rl --arch pts: what a full-size update costs on this GPU, measured on random inputs at the training's shapes
(no env): the rollout's forward per decision (the policy + its free-space head at Bl rows, no gradient) and one PPO minibatch
(a sequence group of the update: L decisions x S cars, the tokens built again in recomputed pieces of 128 decisions, block 1,
the GRU unrolled over L, block 2, the heads, the policy loss's backward) and the free-space head's step on the same minibatch.
usage: python pts_cost.py [BL 304] [L 256] [S 10] [OUT.json]   (S = cars per minibatch: 304 cars / 32 minibatches ~ 10)"""
import json
import math
import sys
import time

import torch
from torch.utils.checkpoint import checkpoint

from f1sim.learn import pts_policy as T
from f1sim.learn.plan_policy import STATE_DIM2

BL = int(sys.argv[1]) if len(sys.argv) > 1 else 304
L = int(sys.argv[2]) if len(sys.argv) > 2 else 256
S = int(sys.argv[3]) if len(sys.argv) > 3 else 10
dev = torch.device("cuda")
torch.manual_seed(0)
BEAMS, CH = 1081, 128
ang = torch.linspace(-math.radians(135), math.radians(135), BEAMS)
actor = T.PtsActor(ang, (0.12555, 0.0, 0.0), 10.0, memory=128, mem_value=True).to(dev)
with torch.no_grad():
    actor.pi.lin.weight.normal_(0, 0.01); actor.fs_in.weight.normal_(0, 0.01)


def rand_in(n):
    fr = torch.rand(n, 5, BEAMS, device=dev) * 1.4                     # ~30 % of the beams without a return (> 1)
    return T.PtsIn(fr.clamp(max=1.0), torch.randn(n, 5, 3, device=dev) * 0.1, torch.ones(n, 5, dtype=torch.bool, device=dev),
                   torch.randn(n, 10, 3, device=dev), torch.rand(n, 10, device=dev) * 5, torch.ones(n, dtype=torch.bool, device=dev),
                   torch.randn(n, 20, device=dev) * 0.3)


def timed(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


# the rollout: one decision for BL cars
P = rand_in(BL); s = torch.randn(BL, STATE_DIM2, device=dev); h = torch.zeros(BL, 128, device=dev)
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    t_roll = timed(lambda: actor(P, s, h), n=20, warm=3)
mem_roll = torch.cuda.max_memory_allocated() / 2 ** 20

# one minibatch of the update
n = L * S
Pm = rand_in(n); sm = torch.randn(n, STATE_DIM2, device=dev); fs = torch.randn(n, 10, 2, device=dev) * 0.3
u = torch.randn(n, 20, device=dev)


def pieces(m):
    return [slice(i, min(m, i + CH)) for i in range(0, m, CH)]


def mb_step():
    outs = [checkpoint(lambda *a: (lambda q1, ctx, sn: (q1, sn) + tuple(ctx))(*actor.stage1(T.PtsIn(*a[:7]), a[7])),
                       *[x[sl] for x in Pm], sm[sl], use_reentrant=False) for sl in pieces(n)]
    cat = [torch.cat([o[j] for o in outs], 0) for j in range(len(outs[0]))]
    q1, sn, ctx = cat[0], cat[1], tuple(cat[2:])
    f = q1[:, -1].reshape(L, S, -1)
    hh = torch.zeros(S, 128, device=dev); hs = []
    for i in range(L):
        hh = actor.gru(f[i], hh); hs.append(hh)
    hc = torch.cat(hs, 0)
    fc = torch.cat([checkpoint(lambda q, a1, a2, a3, a4, a5, hx, fx, ux: actor.trunk(q, (a1, a2, a3, a4, a5), hx, fx, ux),
                               q1[sl], *[c[sl] for c in ctx], hc[sl], fs[sl], Pm.u_anc[sl], use_reentrant=False) for sl in pieces(n)], 0)
    mean, ls, vh = actor.heads(fc)
    loss = ((mean - u) ** 2).mean() + vh.pow(2).mean() + actor.vmem_read(actor.memory_alone(f, torch.zeros(S, 128, device=dev),
                                                                                       torch.zeros(L, S, device=dev))).pow(2).mean()
    loss.backward()
    actor.zero_grad(set_to_none=True)
    return sn


opt = torch.optim.Adam(actor.free_parameters(), lr=1e-4)


def free_step():
    sn = actor.norm(sm)
    run = lambda P_, s_: tuple(torch.cat([o[j] for o in [checkpoint(lambda *a: actor.free(actor.geom, T.PtsIn(*a[:7]), a[7]),
                                                                             *[x[sl] for x in P_], s_[sl], use_reentrant=False)
                                                                  for sl in pieces(n)]], 0) for j in range(2))
    T.free_head_step(actor, opt, Pm, sn, torch.rand(n, 10, device=dev), torch.ones(n, 10, dtype=torch.bool, device=dev), run=run)


torch.cuda.reset_peak_memory_stats()
t_mb = timed(mb_step, n=3, warm=1)
mem_mb = torch.cuda.max_memory_allocated() / 2 ** 20
torch.cuda.reset_peak_memory_stats()
t_free = timed(free_step, n=3, warm=1)
mem_free = torch.cuda.max_memory_allocated() / 2 ** 20
res = dict(gpu=torch.cuda.get_device_name(0), bl=BL, L=L, cars_per_mb=S, decisions_per_mb=n, s_roll_decision=t_roll,
           s_minibatch=t_mb, s_free_step=t_free, mb_roll_peak=mem_roll, mb_update_peak=mem_mb, mb_free_peak=mem_free)
print(json.dumps(res, indent=1))
print(f"[비용] {res['gpu']}: rollout 결정 하나(차 {BL}대, policy + 빈 공간 머리) {t_roll * 1000:.1f} ms (최고 {mem_roll:.0f} MB); "
      f"minibatch 하나({n} 결정 = {L} x {S}대) 앞뒤 {t_mb:.2f} s (최고 {mem_mb:.0f} MB), 빈 공간 머리 한 걸음 {t_free:.2f} s "
      f"(최고 {mem_free:.0f} MB)", flush=True)
if len(sys.argv) > 4:
    json.dump(res, open(sys.argv[4], "w"), indent=1)
