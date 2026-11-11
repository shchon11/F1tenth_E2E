"""세게 닿은 자리가 트랙의 몇 군데에 몰려 있나 (11-11: hit_scenes의 세게 벽에 닿은 33건 중 9건이 다섯 변형 트랙의 같은 자리(8.8, 0.5)).
hit_scenes.py가 쓴 건별 기록 JSON을 읽어, 트랙마다 지도 위에 세게 닿은 점을 찍음: 벽 = 검은 별, 상자 = 갈색 점, 상대 차 = 빨간 점.
세계는 지도만 쓰려고 작게 만듦(환경 34개).
인자: 기록JSON 출력png"""
import os, sys, json
os.environ["F1SIM_NO_STAGGER"] = "1"
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

torch.set_grad_enabled(False)
rows = [r for r in json.load(open(sys.argv[1])) if r["hard"]]
out_png = sys.argv[2]
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, 34, 4242, dev, fmt="v2")
trk = env.sim.track
_edt = trk.edt.float().cpu().numpy(); _off = trk.t_off.cpu().numpy(); _H = trk.t_H.cpu().numpy().astype(int); _W = trk.t_W.cpu().numpy().astype(int)
_res = trk.t_res.cpu().numpy(); _org = trk.t_origin.cpu().numpy()
MAPC = ListedColormap(["#666666", "white"])
n_trk = len(_H)
fig, axs = plt.subplots(3, 6, figsize=(30, 15))
for ax in axs.flat:
    ax.set_axis_off()
for k, ax in zip(range(n_trk), axs.flat):
    ax.set_axis_on()
    g = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k])
    ext = [_org[k][0], _org[k][0] + _W[k] * _res[k], _org[k][1], _org[k][1] + _H[k] * _res[k]]
    ax.imshow(g > 0.0, cmap=MAPC, vmin=0, vmax=1, origin="lower", extent=ext, zorder=0)
    R = [r for r in rows if r["tid"] == k]
    for kind, mk, col, s_, z in ((2, "o", "tab:red", 40, 3), (1, "o", "saddlebrown", 40, 3), (0, "*", "black", 260, 4)):
        P = np.array([[r["x"], r["y"]] for r in R if r["kind"] == kind]).reshape(-1, 2)
        ax.scatter(P[:, 0], P[:, 1], marker=mk, c=col, s=s_, edgecolors="gold" if kind == 0 else "none", linewidths=0.8, zorder=z)
    n = [sum(r["kind"] == q for r in R) for q in range(3)]
    ax.set_title(("원래 트랙" if k == 0 else f"변형 {k}") + f" · 벽 {n[0]} · 상자 {n[1]} · 상대 차 {n[2]}", fontsize=13)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"세게 닿은 자리 {len(rows)}건 -- 검은 별 = 벽, 갈색 점 = 상자, 빨간 점 = 상대 차 (학습 세계, 학습 조건)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(out_png, dpi=60); plt.close(fig)
print(f"그림 {out_png}", flush=True)
