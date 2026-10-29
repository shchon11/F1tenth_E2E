"""라이다 격자 그림 확인 (2026-10-30, `learn.lidar_grid`): 학생이 자기 라이다 4장 + 자기 주행거리계로 만든 격자와 선생의 참 격자를 나란히.

경주 세계(상대 + 상자, 라이다 1081빔 4장 x 4스텝 = 100 ms 간격)에서 기존 선생(DRIVER)으로 몰며 몇 순간: 열 1 = 참 격자(벽+상자 회색, 차 빨강),
열 2 = 가장 새 장의 빈 칸(초록) / 맞은 칸(검정), 열 3 = 가장 오래된 장(0.3 s 전, 주행거리계로 지금 차 기준에 옮김)의 맞은 칸,
열 4 = 새 장 맞은 칸(검정)과 옛 장 맞은 칸(빨강) 겹침 -- 보정이 맞으면 벽·상자는 겹치고 움직이는 차만 어긋남.
수치: 참 격자의 막힌 칸 중 새 장이 맞은 칸으로 본 비율(보이는 면만 맞으므로 1보다 작음), 새 장이 맞은 칸 중 참으로 막힌 칸(정밀도).
사용: python lidar_grid_check.py DRIVER_CKPT --out PNG
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl  # noqa: E402
from f1sim.learn.lidar_grid import OdomHist, lidar_grid  # noqa: E402
from f1sim.learn.state_rl import BEV_H, BEV_RES, BEV_W, BEV_X0, ActorCritic, RunningNorm, StateObs, bev_grid  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("driver"); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=400)
a0 = ap.parse_args()
dev = torch.device("cuda")
ck = torch.load(a0.driver, map_location=dev, weights_only=False)
a = argparse.Namespace(stage="race", warps=0, beams=1081, envs=16, episode_s=40.0, seed=7, prop_density=1.5, prop_max=18,
                       opp_present=1.0, opp_spawn="ahead")
env = state_rl.build_env(a, dev, extra=dict(obs_version=1, scan_stack=4, scan_stride=4))
from f1sim.learn.graph_runtime import prepare_graph_runtime  # noqa: E402
prepare_graph_runtime(env, log=lambda _t: None)
obs, _ = env.reset(seed=7)
lid = env.learner_ids; B = int(lid.numel())
sobs = StateObs(env, scan=bool(ck["args"].get("scan_obs", 0)), corridor=bool(ck["args"].get("corridor_obs", 0)),
                car_channel=bool(ck["args"].get("car_channel", 0)), bev=bool(ck["args"].get("bev", 0)),
                t_beams=int(ck["args"].get("beams", 36)))
last = torch.zeros(B, 2, device=dev)
o = sobs(last)
net = ActorCritic(o.shape[1], bev=ck.get("bev")).to(dev); net.load_state_dict(ck["model"], strict=False); net.eval()
nrm = RunningNorm(o.shape[1], dev); nrm.load_state_dict(ck["norm"])
oh = OdomHist(env, lid, 4, 4); oh.update()
ang = env.sim.lidar.angles
mount = (float(env.cfg.lidar.mount_x), float(env.cfg.lidar.mount_y), float(env.cfg.lidar.mount_yaw))
act = torch.zeros(env.B, env.act_dim, device=dev)
shots = []; rec = []; prec = []
with torch.no_grad():
    for t in range(int(a0.steps)):
        if t % 2 == 0:
            last = net.pi(nrm(sobs(last))).clamp(-1, 1); act[lid] = last
        obs, *_ = env.step(act)
        oh.update()
        if t > 40 and t % 20 == 1:
            tg = bev_grid(env, lid)                                       # (B, 5, H, W)
            lg = lidar_grid(obs["scan"][lid].float(), oh.rel(), ang, mount, float(env.cfg.lidar.range_max))
            occ = (tg[:, 0] + tg[:, 1] + tg[:, 2]).clamp(max=1.0) > 0.4
            hit = lg[:, 1] > 0.4
            if int(hit.sum()):
                prec.append(float((hit & occ).sum() / hit.sum()))
                rec.append(float((hit & occ).sum() / occ.sum().clamp_min(1)))
            if len(shots) < 4 and t % 80 == 41:
                i = int((tg[:, 2].sum((1, 2)) + 0.3 * tg[:, 1].sum((1, 2))).argmax())
                shots.append((t, tg[i].cpu().numpy(), lg[i].cpu().numpy()))
print(f"[라이다 격자] 새 장이 맞은 칸 중 참으로 막힌 칸(정밀도) {np.mean(prec) * 100:.0f} %, 참 막힌 칸 중 맞은 칸(보이는 면) {np.mean(rec) * 100:.0f} %", flush=True)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
ext = (BEV_W * BEV_RES / 2, -BEV_W * BEV_RES / 2, BEV_X0, BEV_X0 + BEV_H * BEV_RES)
fig, axs = plt.subplots(len(shots), 4, figsize=(14, 5.6 * len(shots)), dpi=80)
for r, (t, tg, lg) in enumerate(shots):
    F = lg.shape[0] // 2
    true_rgb = np.ones((BEV_H, BEV_W, 3))
    so = np.clip(tg[0] + tg[1], 0, 1); true_rgb -= so[..., None] * 0.6
    true_rgb[..., 1] -= tg[2] * 0.9; true_rgb[..., 2] -= tg[2] * 0.9
    new_rgb = np.ones((BEV_H, BEV_W, 3)) * 0.85
    new_rgb[lg[0] > 0.4] = (0.75, 1.0, 0.75); new_rgb[lg[1] > 0.3] = (0, 0, 0)
    old_rgb = np.ones((BEV_H, BEV_W, 3)) * 0.85
    old_rgb[lg[2 * (F - 1)] > 0.4] = (0.75, 1.0, 0.75); old_rgb[lg[2 * (F - 1) + 1] > 0.3] = (0, 0, 0)
    ov = np.ones((BEV_H, BEV_W, 3))
    ov[lg[2 * (F - 1) + 1] > 0.3] = (1.0, 0.2, 0.2); ov[lg[1] > 0.3] = (0, 0, 0)
    both = (lg[1] > 0.3) & (lg[2 * (F - 1) + 1] > 0.3); ov[both] = (0.45, 0.45, 0.45)
    for k, (img, ttl) in enumerate(((true_rgb, "참 격자 (회색 벽+상자, 빨강 차)"), (new_rgb, "라이다 새 장: 초록 빈 칸, 검정 맞은 칸"),
                                    (old_rgb, "라이다 0.3 s 전 장 (주행거리계로 옮김)"), (ov, "겹침: 검정 새 / 빨강 옛 / 회색 둘 다"))):
        ax = axs[r, k]
        ax.imshow(img[:, ::-1], origin="lower", extent=ext, aspect="equal")
        ax.plot([0], [0], "b^", ms=8); ax.set_title(f"t {t / 40:.1f}s | " + ttl, fontsize=8)
        ax.set_xlabel("왼쪽 +y (m)"); ax.set_ylabel("앞 +x (m)")
fig.tight_layout(); fig.savefig(a0.out); plt.close(fig)
print("saved", a0.out)
