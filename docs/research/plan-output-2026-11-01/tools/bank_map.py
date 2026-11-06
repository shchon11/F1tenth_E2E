"""obstacle 배치 묶음(은행)의 상자 분포를 맵 위에 (11-06 14:45, 사용자: "그냥 맵에서 장애물 분포 뽑으면 되잖아"). 학습과 콘솔 평가 모두 이 묶음에서
배치를 꺼냄(학습: 배치의 1개~전부 무작위 + 절반은 10~30 s에 치움, 평가: 배치 전부). 왼쪽: 원래 트랙의 모든 배치(400개) 상자 중심 밀도,
가운데: 무작위 배치 8개 예시(각 색이 한 배치), 오른쪽: 배치당 상자 수 분포(학습에서 실제로 깔리는 수 vs 평가). 인자: 묶음.pt 트랙이름 맵id 출력png"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import maps, tracks
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

bank, name, mp, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
b = torch.load(bank, map_location="cpu", weights_only=False)["per_track"][name]
sid, pose = b["sid"].numpy(), b["pose"].numpy()
live = sid >= 0
tr = maps.load(tracks.resolve(mp))
occ = tr.occupancy; res = tr.resolution; H, W = occ.shape
ext = [tr.origin[0], tr.origin[0] + W * res, tr.origin[1], tr.origin[1] + H * res]
xy = pose[live][:, :2]
fig, axs = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.8]))
ax = axs[0]
ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.5)
hb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize=60, mincnt=1, cmap="inferno_r", alpha=0.85)
plt.colorbar(hb, ax=ax, fraction=0.035, label="상자 수 (400개 배치 합)")
ax.set_title(f"{name}: 배치 {sid.shape[0]}개의 상자 중심 {len(xy)}개", fontsize=11); ax.set_aspect("equal")
ax = axs[1]
ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.5)
rng = np.random.default_rng(0)
for i, k in enumerate(rng.choice(sid.shape[0], 8, replace=False)):
    p = pose[k][live[k]]
    ax.scatter(p[:, 0], p[:, 1], s=28, color=plt.cm.tab10(i), edgecolor="k", lw=0.4, label=f"배치 {k} ({len(p)}개)")
ax.legend(fontsize=7, loc="upper left", ncol=2); ax.set_aspect("equal")
ax.set_title("무작위 배치 8개 (평가는 이런 배치를 상자 전부로)", fontsize=11)
ax = axs[2]
n = live.sum(1)
ks = np.arange(1, n.max() + 1)
train = np.zeros(len(ks))
for nn in n:                                      # training: 1 .. nn uniformly
    train[:nn] += 1.0 / nn
train /= len(n)
ax.bar(ks - 0.2, train * 100, width=0.4, label="학습 (상자 있는 episode, 1개~전부 무작위)")
ev = np.array([np.mean(n == k) for k in ks]) * 100
ax.bar(ks + 0.2, ev, width=0.4, label="평가 (배치 전부)")
ax.set_xlabel("배치에 깔린 상자 수"); ax.set_ylabel("%"); ax.legend(fontsize=8)
ax.set_title(f"배치당 상자 수: 학습 평균 {np.sum(ks * train):.1f}개, 평가 평균 {n.mean():.1f}개\n(학습은 obstacle episode 50 %, 그중 절반은 10~30 s에 치움)", fontsize=10)
plt.tight_layout(); plt.savefig(out, dpi=80)
print(out, "| 배치", sid.shape[0], "| 배치당 상자 평균", round(float(n.mean()), 1), "| 최소", int(n.min()), "최대", int(n.max()))
