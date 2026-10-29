"""내 차 중심 점유 격자 그림 확인 (2026-10-30, `state_rl.bev_grid`).

경주 세계(상대 + 상자)에서 기존 선생(DRIVER, 예: srl21)으로 차를 몰면서 몇 순간을 골라, 왼쪽에 위에서 본 실제 장면(트랙, 상자의 참 모양,
상대 차, 내 차와 격자 범위 사각형), 오른쪽에 그 순간의 격자 5채널(벽 / 상자 / 차 / 차의 상대 속도 x, y)을 내 차 좌표계 그대로 그림.
격자에서 앞(+x)은 위쪽, 왼쪽(+y)은 왼쪽. 그림에서 벽·상자·차가 같은 자리에 같은 모양으로 보이면 렌더링이 맞음.
사용: python bev_check.py DRIVER_CKPT --out PNG
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl  # noqa: E402
from f1sim.learn.state_rl import (BEV_H, BEV_RES, BEV_W, BEV_X0, ActorCritic, RunningNorm, StateObs,  # noqa: E402
                                  bev_grid)

ap = argparse.ArgumentParser(); ap.add_argument("driver"); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=400)
a0 = ap.parse_args()
dev = torch.device("cuda")
ck = torch.load(a0.driver, map_location=dev, weights_only=False)
a = argparse.Namespace(stage="race", warps=0, beams=108, envs=16, episode_s=40.0, seed=7, prop_density=1.5, prop_max=18,
                       opp_present=1.0, opp_spawn="ahead")
env = state_rl.build_env(a, dev)
from f1sim.learn.graph_runtime import prepare_graph_runtime  # noqa: E402
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=7)
lid = env.learner_ids; B = int(lid.numel())
sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)), corridor=bool(ck["args"].get("corridor_obs", 0)),
                car_channel=bool(ck["args"].get("car_channel", 0)))
last = torch.zeros(B, 2, device=dev)
o = sobs(last)
net = ActorCritic(o.shape[1], bev=ck.get("bev")).to(dev); net.load_state_dict(ck["model"], strict=False); net.eval()
nrm = RunningNorm(o.shape[1], dev); nrm.load_state_dict(ck["norm"])
act = torch.zeros(env.B, env.act_dim, device=dev)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
shots = []
with torch.no_grad():
    for t in range(int(a0.steps)):
        if t % 2 == 0:
            last = net.pi(nrm(sobs(last))).clamp(-1, 1); act[lid] = last
        env.step(act)
        if t % 2 == 1 and t > 40 and len(shots) < 4 and t % 60 == 41:
            g = bev_grid(env, lid)                                     # (B, 5, H, W)
            st = env.last_result.state
            # the row with the most car + prop cells in its grid
            score = g[:, 1].sum((1, 2)) + 2 * g[:, 2].sum((1, 2))
            i = int(score.argmax())
            shots.append((t, i, g[i].cpu().numpy(), st.cpu().numpy(), int(lid[i])))
track = env.sim.track.tracks[0]
fig, axs = plt.subplots(len(shots), 6, figsize=(24, 5.2 * len(shots)), dpi=80)
names = ["벽", "상자", "차", "차 상대속도 x(/5)", "차 상대속도 y(/5)"]
po = env.procedural
for r, (t, i, g, st, row) in enumerate(shots):
    ax = axs[r, 0]
    occ = track.occupancy; rs = track.resolution; ox, oy = track.origin
    ax.imshow(occ, origin="lower", cmap="Greys", extent=(ox, ox + occ.shape[1] * rs, oy, oy + occ.shape[0] * rs), vmin=0, vmax=1.5)
    for c in (po.p_sid[row] >= 0).nonzero().flatten().tolist():
        n_, d_ = po.p_n[row, c].cpu().numpy(), po.p_d[row, c].cpu().numpy()
        ok = np.isfinite(d_); n_, d_ = n_[ok], d_[ok]
        oo = np.argsort(np.arctan2(n_[:, 1], n_[:, 0])); n_, d_ = n_[oo], d_[oo]
        cs = [np.linalg.solve(np.array([n_[j], n_[(j + 1) % len(n_)]]), np.array([d_[j], d_[(j + 1) % len(n_)]]))
              for j in range(len(n_)) if abs(np.linalg.det(np.array([n_[j], n_[(j + 1) % len(n_)]]))) > 1e-9]
        if len(cs) >= 3:
            cs = np.array(cs); px, py, pyaw = po.p_poses[row, c].cpu().numpy()
            cc, ss = math.cos(pyaw), math.sin(pyaw)
            Wp = cs @ np.array([[cc, ss], [-ss, cc]]) + np.array([px, py])
            ax.fill(np.r_[Wp[:, 0], Wp[:1, 0]], np.r_[Wp[:, 1], Wp[:1, 1]], color="darkorange")
    x, y, yaw = st[row, 0], st[row, 1], st[row, 2]
    for q in env.sim.other_idx[row].tolist():
        ax.plot(st[q, 0], st[q, 1], "s", color="purple", ms=8)
        ax.arrow(st[q, 0], st[q, 1], 0.5 * math.cos(st[q, 2]), 0.5 * math.sin(st[q, 2]), width=0.05, color="purple")
    ax.arrow(x, y, 0.6 * math.cos(yaw), 0.6 * math.sin(yaw), width=0.07, color="blue")
    cx = [BEV_X0, BEV_X0 + BEV_H * BEV_RES, BEV_X0 + BEV_H * BEV_RES, BEV_X0, BEV_X0]
    cy = [-BEV_W * BEV_RES / 2, -BEV_W * BEV_RES / 2, BEV_W * BEV_RES / 2, BEV_W * BEV_RES / 2, -BEV_W * BEV_RES / 2]
    wx = [x + u * math.cos(yaw) - v * math.sin(yaw) for u, v in zip(cx, cy)]
    wy = [y + u * math.sin(yaw) + v * math.cos(yaw) for u, v in zip(cx, cy)]
    ax.plot(wx, wy, "-", color="cyan", lw=1.5)
    ax.set_xlim(x - 9, x + 9); ax.set_ylim(y - 9, y + 9); ax.set_aspect("equal")
    ax.set_title(f"t {t / 40:.1f}s 차 {row}: 위에서 본 실제(파랑 = 나, 보라 = 상대, 하늘색 = 격자 범위)", fontsize=8)
    ext = (BEV_W * BEV_RES / 2, -BEV_W * BEV_RES / 2, BEV_X0, BEV_X0 + BEV_H * BEV_RES)   # left (+y) on the left
    for k in range(5):
        ax = axs[r, k + 1]
        cm, vmin, vmax = ("Greys", 0, 1) if k < 3 else ("coolwarm", -1, 1)
        ax.imshow(g[k][:, ::-1], origin="lower", cmap=cm, vmin=vmin, vmax=vmax, extent=ext, aspect="equal")   # column 0 = +y
        ax.plot([0], [0], "b^", ms=8)
        ax.set_xlabel("왼쪽 +y (m)"); ax.set_ylabel("앞 +x (m)")
        ax.set_title(names[k] + f" (합 {g[k].sum():.1f})", fontsize=9)
fig.tight_layout(); fig.savefig(a0.out); plt.close(fig)
print("saved", a0.out)
