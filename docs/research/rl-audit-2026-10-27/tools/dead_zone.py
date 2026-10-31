"""경주 정책의 '죽은 구간' 측정 (2026-11-01): 정책 평균(자르기 전)이 행동 범위 밖으로 2σ 넘게 나가 있으면, 탐색 잡음이 있어도 실행되는
행동은 늘 범위 끝 그대로 — 그 차원의 표본들이 실행에서 다르지 않으니 PPO 이득이 그 차원에 대해 아무것도 말해 주지 못한다(기울기 0).
속도가 그렇게 최대에 붙은 채 느린 차 / 상자 앞에 있으면 정책은 브레이크를 배울 길이 없다.

학습 세계(state_rl.build_env, 체크포인트의 인자, 상대·상자 비율은 덮어쓸 수 있음)에서 혼합 정책을 평균 행동으로 몰고, 경주 정책이 몬 결정마다
속도·조향 평균과 표준편차를 기록: 죽은 비율(속도 μ > 1 + 2σ 또는 μ < −1 − ext − 2σ; 조향 |μ| > 1 + 2σ)을 상황별로 —
경주 정책이 몬 결정 전체 / 상대가 0~10 m 앞 / 상자가 10 m 안 / 강한 충돌(벽·상자·차) 직전 1 s.
사용: python dead_zone.py CKPT [--envs 512 --steps 2400 --opp-present 0.5 --prop-fraction 0.8 --opp-slot JSON]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--envs", type=int, default=512)
ap.add_argument("--steps", type=int, default=2400)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--opp-present", type=float, default=0.5)
ap.add_argument("--prop-fraction", type=float, default=0.8)
ap.add_argument("--opp-slot", default="")
a0 = ap.parse_args()
dev = torch.device("cuda")
ck = torch.load(a0.ckpt, map_location=dev, weights_only=False)
ar = dict(ck["args"])
ar.update(envs=int(a0.envs), seed=int(a0.seed), opp_present=float(a0.opp_present), prop_fraction=float(a0.prop_fraction),
          opp_slot=a0.opp_slot, stage="race", episode_s=float(a0.steps) / 40 + 1, device="cuda")
a = SimpleNamespace(**ar)
env = state_rl.build_env(a, dev)
from f1sim.learn.graph_runtime import prepare_graph_runtime  # noqa: E402
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=int(a.seed))
lid = env.learner_ids; B = int(lid.numel())
tp = state_rl.TeacherPolicy(env, ck, dev)
ext = tp.ext
R = int(ar.get("repeat", 2))
W = 40 // R                                    # decisions in 1 s
act = torch.zeros(B, 2, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
w_col = abs(env.ecfg.reward_collision)
touching = torch.zeros(B, dtype=torch.bool, device=dev)
hist = torch.zeros(W, B, 3, device=dev)        # last 1 s: gated, dead speed, dead steer
acc = {k: np.zeros(4) for k in ("all", "follow", "prop", "prehit")}   # n, dead speed, dead steer, saturated speed (any)
mu_follow = []
with torch.no_grad():
    for t in range(int(a0.steps) // R):
        tp.o = tp.sobs(act)
        dist = tp.net.dist(tp.nrm(tp.o))
        mu, sd = dist.mean, dist.stddev
        a_pol = state_rl.clamp_act(mu, ext)
        act = tp.mix(a_pol, act) if tp.mix is not None else a_pol
        gate = tp.mix.gate if tp.mix is not None else torch.ones(B, dtype=torch.bool, device=dev)
        dead_v = (mu[:, 1] > 1 + 2 * sd[:, 1]) | (mu[:, 1] < -1 - ext - 2 * sd[:, 1])
        dead_s = mu[:, 0].abs() > 1 + 2 * sd[:, 0]
        sat_v = (mu[:, 1] > 1) | (mu[:, 1] < -1 - ext)
        hist = torch.roll(hist, -1, 0); hist[-1] = torch.stack([gate.float(), dead_v.float(), dead_s.float()], 1)
        res = env.last_result
        follow = torch.zeros(B, dtype=torch.bool, device=dev)
        if env.M > 1:
            g = env.signed_gaps(res.s, env.sim.tid)[lid][:, 0]
            follow = (g > 0) & (g < 10)
        cb = state_rl.corridor_block(env, lid, res.s[lid], env.sim.tid[lid]).reshape(B, len(state_rl.AHEAD), state_rl.CORRIDOR_BINS)
        propn = cb[:, :tp.mix.n_seg if tp.mix is not None else 8].flatten(1).any(1)
        for k, m in (("all", gate), ("follow", gate & follow), ("prop", gate & propn)):
            acc[k] += [float(m.sum()), float((m & dead_v).sum()), float((m & dead_s).sum()), float((m & sat_v).sum())]
        if t > W:
            mf = (gate & follow)
            if bool(mf.any()):
                mu_follow.append(mu[mf, 1].cpu().numpy())
        act_full[lid] = act
        hit = torch.zeros(B, dtype=torch.bool, device=dev)
        for k in range(R):
            _, r_, te, trc, info = env.step(act_full)
            comp = info["reward_components"]
            col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
            tch = car < 0; onset = tch & ~touching; d_ = (te | trc)[lid]; touching = tch & ~d_
            hit |= ((-col / max(w_col, 1e-9)) > 0) | onset
        if bool(hit.any()):
            h = hist[:, hit]                          # (W, n, 3): the second before each hit, the decisions the race policy drove
            g_ = h[..., 0] > 0
            acc["prehit"] += [float(g_.sum()), float((g_ & (h[..., 1] > 0)).sum()), float((g_ & (h[..., 2] > 0)).sum()), 0.0]
name = os.path.basename(os.path.dirname(a0.ckpt)) + "/" + os.path.basename(a0.ckpt)
lab = {"all": "경주 정책이 몬 결정 전체", "follow": "상대 0~10 m 앞", "prop": "상자 10 m 안", "prehit": "강한 충돌 직전 1 s"}
for k in ("all", "follow", "prop", "prehit"):
    n, dv, ds, sv = acc[k]
    extra = f", 속도 범위 밖(아무 만큼) {sv / max(n, 1) * 100:.0f} %" if k != "prehit" else ""
    print(f"[죽은 구간] {name} | {lab[k]}: 결정 {int(n)}개, 속도 죽음 {dv / max(n, 1) * 100:.1f} %, 조향 죽음 {ds / max(n, 1) * 100:.1f} %{extra}",
          flush=True)
if mu_follow:
    x = np.concatenate(mu_follow)
    print(f"[죽은 구간] {name} | 상대 0~10 m 앞 속도 평균 분위 5/25/50/75/95 % = "
          + " / ".join(f"{q:+.2f}" for q in np.percentile(x, [5, 25, 50, 75, 95])), flush=True)
