"""solo lap에서 어디서 느려졌나 (11-04 12:20): one-world를 빠른 solo policy에서 시작했는데 u50 solo lap 6.50 -> 6.71 s.
checkpoint 여러 개를 빈 트랙에서 혼자 달려(콘솔 세션 그대로, 출발 첫 랩 버림) 트랙 위치(arc) 별 속도와 centerline에서 가로 위치를 겹쳐 그림.
아래 칸은 첫 checkpoint 대비 속도 차를 트랙 위에 색으로. 인자: 맵id 초 출력png 체크포인트들..."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
_fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


mp, secs, out, runs = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4:]
NB = 400
res = []
for run in runs:
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=0, races=1, cars_per_race=1, device="cuda", compile=False,
                                        collision_mode="soft", procedural=False, episode_s=max(60.0, secs + 5),
                                        randomize=False), 0)
    env = s["env"]
    tid = env.sim.tid[:1]
    L = float(env.sim.track.length[env.sim.tid[0]])
    rows = []
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        st = env.sim.state
        sa, lat, _ = env.sim.track.project(st[:1, :2], tid)
        rows.append((float(sa[0]), float(lat[0]), float(st[0, 3]), float(st[0, 0]), float(st[0, 1])))
    a = np.array(rows)
    a = a[int(8.0 / 0.025):]                                   # 출발과 첫 랩 버림
    b = np.clip((a[:, 0] / L * NB).astype(int), 0, NB - 1)
    v = np.array([a[b == i, 2].mean() if (b == i).any() else np.nan for i in range(NB)])
    off = np.array([a[b == i, 1].mean() if (b == i).any() else np.nan for i in range(NB)])
    xy = np.array([a[b == i, 3:5].mean(0) if (b == i).any() else [np.nan, np.nan] for i in range(NB)])
    tl = float(np.nansum(L / NB / np.maximum(v, 0.3)))
    name = f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run).replace('state_', '').replace('.pt', '')}"
    res.append((name, v, off, xy, tl))
    print(f"{name}: 구간 속도로 잰 랩 {tl:.2f} s, 평균 속도 {np.nanmean(v):.2f} m/s", flush=True)
    del env, s, w
    torch.cuda.empty_cache()
x = (np.arange(NB) + 0.5) / NB * L
fig, ax = plt.subplots(3, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [1, 1, 1.6]})
for name, v, off, xy, tl in res:
    ax[0].plot(x, v, lw=1.2, label=f"{name} ({tl:.2f} s)")
    ax[1].plot(x, off, lw=1.2, label=name)
ax[0].set_ylabel("속도 [m/s]"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_ylabel("centerline에서 가로 [m]"); ax[1].set_xlabel("트랙 위치 [m]"); ax[1].grid(alpha=.3)
v0, xy0 = res[0][1], res[0][3]
for name, v, off, xy, tl in res[1:]:
    sc = ax[2].scatter(xy0[:, 0], xy0[:, 1], c=v - v0, cmap="coolwarm_r", vmin=-1.5, vmax=1.5, s=14)
    ax[2].plot(xy[:, 0], xy[:, 1], "k-", lw=.5)
for i in range(0, NB, 40):
    ax[2].annotate(f"{x[i]:.0f}", xy0[i], fontsize=8)
ax[2].set_aspect("equal"); ax[2].set_title(f"속도 차 (마지막 - 첫 checkpoint, 빨강 = 느려짐), 숫자 = 트랙 위치 [m]")
plt.colorbar(sc, ax=ax[2], shrink=.6)
plt.tight_layout(); plt.savefig(out, dpi=90)
print(out, flush=True)
