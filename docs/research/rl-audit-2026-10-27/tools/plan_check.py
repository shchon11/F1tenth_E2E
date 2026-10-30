"""계획기 그림 확인 + 기하만의 시험 (2026-10-30, `learn.path_plan.PathPlanner`).

경주 세계(상대 + 상자, 한국)에서 학습 차를 단순 추종기로 몬다: 조향 = 계획된 길 위 앞 (1.0 + 0.3 v) m 점으로 순수 추종,
속도 = min(주행선 속도 프로파일, 막힌 곳까지 3 m/s² 로 멈출 수 있는 속도). 학습된 것이 하나도 없으므로 상자·차 충돌이
적으면 계획기의 길이 맞다는 뜻(강화학습이 이 길을 빠르게 달리게 하는 것은 그다음).
그림: 몇 순간 -- 트랙, 상자의 참 모양, 상대 차(보라)와 그 속도, 후보 길(회색 = 막힘 없음, 빨강 = 막힘), 고른 길(자홍), 내 차.
수치: 강한 충돌(1.5 m/s 이상)/km 벽·상자·차, 진행 속도, 랩.
사용: python plan_check.py --out DIR [--envs 32 --steps 2400 --opp 1]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import common, state_rl  # noqa: E402
from f1sim.learn.path_plan import PathPlanner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--envs", type=int, default=32)
ap.add_argument("--steps", type=int, default=2400)
ap.add_argument("--opp", type=float, default=1.0, help="share of races with an opponent")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--images", type=int, default=6)
ap.add_argument("--stage", default="race", choices=("solo", "props", "race"))
a0 = ap.parse_args()
os.makedirs(a0.out, exist_ok=True)
dev = torch.device("cuda")
a = argparse.Namespace(stage=a0.stage, warps=0, beams=36, envs=int(a0.envs), episode_s=40.0, seed=int(a0.seed),
                       prop_density=1.5, prop_max=18, opp_present=float(a0.opp), opp_spawn="random")
env = state_rl.build_env(a, dev, extra=dict(stagger_first_episode=False))
from f1sim.learn.graph_runtime import prepare_graph_runtime  # noqa: E402
prepare_graph_runtime(env, log=lambda _t: None)
env.sim.measure_impacts = True
_, rls = common.load_tracks([state_rl.KOREA], racelines=True, **dict(common.teacher_limits(7.0, 6.5, 4.0), objective="min_time"))
env.reset(seed=int(a0.seed))
lid = env.learner_ids; B = int(lid.numel())
pl = PathPlanner(env, lid, rls)
act = torch.zeros(env.B, env.act_dim, device=dev)
WB = 0.33
prog = 0.0; hits = {"벽": 0, "상자": 0, "차": 0}; touching = torch.zeros(B, dtype=torch.bool, device=dev)
laps = []; shots = []
for t in range(int(a0.steps)):
    if t % 2 == 0:
        pl.update()
        res = env.last_result; st = res.state[lid]; tid = env.sim.tid[lid]; s_car = res.s[lid]
        v = st[:, 3].abs()
        Ld = 1.0 + 0.3 * v
        tgt, _ = pl.path_xy((s_car + Ld)[:, None], tid)
        tgt = tgt[:, 0]
        dx, dy = tgt[:, 0] - st[:, 0], tgt[:, 1] - st[:, 1]
        c, s_ = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        ly = -dx * s_ + dy * c
        delta = torch.atan(2 * WB * ly / (Ld ** 2))
        v_rl = pl.rl_speed(s_car + 1.0, tid)
        v_stop = torch.sqrt(2 * 3.0 * (pl.free - 1.0).clamp_min(0.0))
        v_cmd = torch.minimum(v_rl, torch.where(pl.free < 15.0, v_stop, torch.full_like(v_stop, 99.0)))
        act[lid, 0] = (delta / float(env.s_max)).clamp(-1, 1)
        act[lid, 1] = (v_cmd / float(env.ecfg.v_max_policy) * 2 - 1).clamp(-1, 1)
        if len(shots) < int(a0.images) and t > 60 and t % 90 == 0:
            pts, bad, best = pl.last_candidates
            po = env.procedural
            n_props = (po.p_sid[lid] >= 0).sum(1) if po is not None else torch.zeros(B, device=dev)
            # the row with a blocked candidate and the most props / a car near
            score = bad.any(-1).float().sum(1) + 0.1 * n_props.float()
            i = int(score.argmax())
            shots.append((t, i, int(lid[i]), pts[i].cpu().numpy(), bad[i].cpu().numpy(), int(best[i]), st.cpu().numpy(),
                          env.last_result.state.cpu().numpy(), float(pl.free[i])))
    _, r_, te, trc, info = env.step(act)
    res = env.last_result
    on, imp = res.contact_onsets[lid], res.contact_impacts[lid]
    hits["벽"] += int((on[:, 0] & (imp[:, 0] >= 1.5)).sum()); hits["상자"] += int((on[:, 1] & (imp[:, 1] >= 1.5)).sum())
    if res.car_collision is not None:
        cc = res.car_collision[lid]; ci = res.car_impact[lid] if res.car_impact is not None else torch.zeros(B, device=dev)
        hits["차"] += int((cc & ~touching & (ci >= 1.5)).sum()); touching = cc
    prog += float(info["progress"][lid].float().clamp_min(0).sum())
    if info["lap_times"].numel():
        laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
km = prog / 1000.0; secs = int(a0.steps) * B * float(env.sim.control_dt)
print(f"[계획기 + 순수 추종] {km:.1f} km, 진행 {prog / secs:.2f} m/s, 랩 중앙 {np.median(laps) if laps else float('nan'):.2f} s ({len(laps)}개) | "
      f"강한 충돌/km 벽 {hits['벽'] / km:.2f} 상자 {hits['상자'] / km:.2f} 차 {hits['차'] / km:.2f}", flush=True)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
track = env.sim.track.tracks[0]
po = env.procedural
ncol = 3; nrow = int(math.ceil(len(shots) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(7 * ncol, 7 * nrow), dpi=80)
axs = np.atleast_2d(axs)
for k, (t, i, row, pts, bad, best, st, st_all, free) in enumerate(shots):
    ax = axs[k // ncol, k % ncol]
    occ = track.occupancy; rs = track.resolution; ox, oy = track.origin
    ax.imshow(occ, origin="lower", cmap="Greys", extent=(ox, ox + occ.shape[1] * rs, oy, oy + occ.shape[0] * rs), vmin=0, vmax=1.5)
    for c in ((po.p_sid[row] >= 0).nonzero().flatten().tolist() if po is not None else []):
        n_, d_ = po.p_n[row, c].cpu().numpy(), po.p_d[row, c].cpu().numpy()
        ok = np.isfinite(d_); n_, d_ = n_[ok], d_[ok]
        oo = np.argsort(np.arctan2(n_[:, 1], n_[:, 0])); n_, d_ = n_[oo], d_[oo]
        cs = [np.linalg.solve(np.array([n_[j], n_[(j + 1) % len(n_)]]), np.array([d_[j], d_[(j + 1) % len(n_)]]))
              for j in range(len(n_)) if abs(np.linalg.det(np.array([n_[j], n_[(j + 1) % len(n_)]]))) > 1e-9]
        if len(cs) >= 3:
            cs = np.array(cs); px, py, pyaw = po.p_poses[row, c].cpu().numpy()
            cc_, ss_ = math.cos(pyaw), math.sin(pyaw)
            W = cs @ np.array([[cc_, ss_], [-ss_, cc_]]) + np.array([px, py])
            ax.fill(np.r_[W[:, 0], W[:1, 0]], np.r_[W[:, 1], W[:1, 1]], color="darkorange")
    for cnd in range(pts.shape[0]):
        col = "red" if bad[cnd].any() else "silver"
        ax.plot(pts[cnd, :, 0], pts[cnd, :, 1], "-", color=col, lw=0.6, alpha=0.6)
    ax.plot(pts[best, :, 0], pts[best, :, 1], "-", color="magenta", lw=2.2)
    x, y, yaw = st[i, 0], st[i, 1], st[i, 2]
    ax.arrow(x, y, 0.6 * math.cos(yaw), 0.6 * math.sin(yaw), width=0.07, color="blue")
    if env.sim.other_idx is not None:
        for q in env.sim.other_idx[row].tolist():
            if env.sim.absent_any and bool(env.sim.absent[q]):
                continue
            qx, qy, qyaw, qv = st_all[q, 0], st_all[q, 1], st_all[q, 2], st_all[q, 3]
            ax.plot(qx, qy, "s", color="purple", ms=9)
            ax.arrow(qx, qy, 0.4 * qv * math.cos(qyaw), 0.4 * qv * math.sin(qyaw), width=0.04, color="purple")
    ax.set_xlim(x - 8, x + 8); ax.set_ylim(y - 8, y + 8); ax.set_aspect("equal")
    ax.set_title(f"t {t / 40:.1f}s 차 {row}: 자홍 = 고른 길, 회색 = 후보(막힘 없음), 빨강 = 후보(막힘) | 막힌 곳까지 {free:.1f} m", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(a0.out, "plan.png")); plt.close(fig)
print("saved", os.path.join(a0.out, "plan.png"))
