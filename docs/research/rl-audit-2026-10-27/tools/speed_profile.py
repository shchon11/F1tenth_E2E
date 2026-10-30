"""빈 트랙 속도 프로파일 비교 (2026-10-31): 경주 학습이 빈 트랙 랩을 어디서 느리게 만드는지 그림으로.

선생 체크포인트 여러 개를 같은 빈 한국 트랙(상대·상자 없음, 같은 출발)에서 평균 행동으로 2바퀴 몰고, 트랙 위치(중앙선 s)마다 속도를 평균해
겹쳐 그림(위), 첫 정책 대비 속도 차이(가운데), 트랙 지도에 위치 표시(아래 왼쪽)와 구간별 잃은 시간(아래 오른쪽).
사용: python speed_profile.py --out PNG NAME=CKPT [NAME=CKPT ...]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl  # noqa: E402
from f1sim.learn.state_rl import ActorCritic, RunningNorm, StateObs, clamp_act  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("pol", nargs="+")
ap.add_argument("--steps", type=int, default=1400)
a0 = ap.parse_args()
dev = torch.device("cuda")
RES = 0.25
prof = {}
track = None; L = None; cl = None
for item in a0.pol:
    name, path = item.split("=", 1)
    ck = torch.load(path, map_location=dev, weights_only=False)
    ar = ck["args"]
    a = argparse.Namespace(stage="solo", warps=0, beams=int(ar.get("beams", 36)), envs=16, episode_s=40.0, seed=5,
                           reverse_ext=float(ar.get("reverse_ext", 0.0) or 0.0))
    env = state_rl.build_env(a, dev, extra=dict(stagger_first_episode=False))
    from f1sim.learn.graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    env.reset(seed=5)
    lid = env.learner_ids; B = int(lid.numel())
    sobs = StateObs(env, scan=bool(ar.get("scan_obs", 0)), corridor=bool(ar.get("corridor_obs", 0)),
                    car_channel=bool(ar.get("car_channel", 0)), bev=bool(ar.get("bev", 0)), t_beams=int(ar.get("beams", 36)))
    last = torch.zeros(B, 2, device=dev); o = sobs(last)
    net = ActorCritic(o.shape[1], bev=ck.get("bev")).to(dev); net.load_state_dict(ck["model"], strict=False); net.eval()
    nrm = RunningNorm(o.shape[1], dev); nrm.load_state_dict(ck["norm"])
    act = torch.zeros(env.B, env.act_dim, device=dev)
    L = float(env.sim.track.length[0]); n = int(L / RES) + 1
    acc = np.zeros(n); cnt = np.zeros(n); laps = []
    with torch.no_grad():
        for t in range(int(a0.steps)):
            if t % 2 == 0:
                last = clamp_act(net.pi(nrm(sobs(last))), ar.get("reverse_ext", 0.0)); act[lid] = last
            _, _, _, _, info = env.step(act)
            if t > 120:
                s = env.last_result.s[lid].cpu().numpy(); v = env.sim.state[lid, 3].cpu().numpy()
                i = np.clip((s / RES).astype(int), 0, n - 1)
                np.add.at(acc, i, v); np.add.at(cnt, i, 1)
            if info["lap_times"].numel():
                laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
    prof[name] = (acc / np.maximum(cnt, 1), float(np.median(laps)) if laps else float("nan"))
    track = env.sim.track.tracks[0]
    cl = env.sim.track.cl[0].cpu().numpy(); cls = env.sim.track.cl_s[0].cpu().numpy()
    print(f"[속도 프로파일] {name}: 랩 중앙 {prof[name][1]:.2f} s", flush=True)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
names = list(prof)
s_ax = np.arange(len(prof[names[0]][0])) * RES
fig = plt.figure(figsize=(18, 13), dpi=80)
ax1 = fig.add_subplot(3, 1, 1)
for nm in names:
    ax1.plot(s_ax, prof[nm][0], lw=1.3, label=f"{nm} (랩 {prof[nm][1]:.2f} s)")
ax1.set_ylabel("속도 m/s"); ax1.legend(fontsize=9); ax1.set_title("빈 트랙: 트랙 위치(중앙선 s)마다 평균 속도", fontsize=10)
ax2 = fig.add_subplot(3, 1, 2)
ref = prof[names[0]][0]
for nm in names[1:]:
    ax2.plot(s_ax, prof[nm][0] - ref, lw=1.3, label=f"{nm} - {names[0]}")
ax2.axhline(0, color="k", lw=0.5); ax2.set_ylabel("속도 차이 m/s"); ax2.set_xlabel("중앙선 s (m)"); ax2.legend(fontsize=9)
ax3 = fig.add_subplot(3, 2, 5)
occ = track.occupancy; rs = track.resolution; ox, oy = track.origin
ax3.imshow(occ, origin="lower", cmap="Greys", extent=(ox, ox + occ.shape[1] * rs, oy, oy + occ.shape[0] * rs), vmin=0, vmax=1.5)
for k in range(0, int(L), 5):
    j = int(np.argmin(np.abs(cls - k)))
    ax3.plot(cl[j, 0], cl[j, 1], "o", color="red", ms=3); ax3.text(cl[j, 0], cl[j, 1], str(k), fontsize=7, color="red")
ax3.set_aspect("equal"); ax3.set_title("중앙선 s 위치(m)", fontsize=9)
ax4 = fig.add_subplot(3, 2, 6)
seg = 5.0
for nm in names[1:]:
    lost = []
    for k in np.arange(0, L, seg):
        m = (s_ax >= k) & (s_ax < k + seg)
        t_ref = np.sum(RES / np.maximum(ref[m], 0.3)); t_nm = np.sum(RES / np.maximum(prof[nm][0][m], 0.3))
        lost.append(t_nm - t_ref)
    ax4.bar(np.arange(0, L, seg) + (names.index(nm) - 1) * 1.2, lost, width=1.2, label=nm)
ax4.axhline(0, color="k", lw=0.5); ax4.set_xlabel("구간 시작 s (m, 5 m씩)"); ax4.set_ylabel(f"{names[0]}보다 더 걸린 시간 s"); ax4.legend(fontsize=8)
fig.tight_layout(); fig.savefig(a0.out); plt.close(fig)
print("saved", a0.out)
