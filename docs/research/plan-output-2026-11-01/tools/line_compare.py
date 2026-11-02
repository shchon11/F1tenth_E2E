"""깨끗한 맵에서 혼자 정책과 상자 정책의 주행선·속도 비교 (11-03): 상자 정책은 깨끗한 맵에서 7.18 s(혼자 6.50 s)인데, 학습 세계 빈 길에서
속도 명령을 +1.5 m/s 올려도 실제 진행은 거의 그대로 -> 속도 명령이 아니라 주행선(경로 모양)이 속도를 묶는지. 콘솔 워커로 각 정책 30 s,
트랙 위치(s)별 가로 위치·속도·실행 속도 명령·첫 곡률, 맵 위 주행선 그림. 인자: 맵id 그림머리 체크포인트...(이름=경로)"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


mp, head = sys.argv[1], sys.argv[2]
runs = [x.split("=", 1) for x in sys.argv[3:]]
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
res = {}
for name, path in runs:
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    s = w.build_session(P.SessionConfig(run=path, map_name=mp, seed=0, races=1, cars_per_race=1, device="cuda", compile=False,
                                        collision_mode="soft", procedural=False, episode_s=3600.0, randomize=False), 0)
    env = s["env"]
    rows = []
    for k in range(int(30.0 / 0.025)):
        w._step_once(s)
        if k < 200:
            continue
        r = env.last_result
        rows.append((float(r.s[0]), float(r.lateral[0]), float(env.sim.state[0, 3]), float(env.sim.cmd[0, 1]), float(env.sim.state[0, 0]),
                     float(env.sim.state[0, 1]), float(env.sim.cmd[0, 0])))
    A = np.array(rows)
    Lt = float(env.sim.track.length[env.sim.tid[0]])
    A[:, 0] = np.mod(A[:, 0], Lt)
    res[name] = A
    print(f"{name}: 평균 속도 {A[:, 2].mean():.2f} m/s, 최고 {A[:, 2].max():.2f}, 속도 명령 평균 {A[:, 3].mean():.2f}, 조향 크기 평균 {np.abs(A[:, 6]).mean():.3f} rad", flush=True)
    occ_info = (env.sim.track, Lt)
    del w, s, env
    torch.cuda.empty_cache()
# per 0.5 m of track
names = list(res)
Lt = occ_info[1]
bins = np.arange(0, Lt + 0.5, 0.5)
fig, axs = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
cols = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
prof = {}
for c, n in zip(cols, names):
    A = res[n]
    i = np.digitize(A[:, 0], bins)
    lat = np.array([A[i == j, 1].mean() if np.any(i == j) else np.nan for j in range(1, len(bins))])
    v = np.array([A[i == j, 2].mean() if np.any(i == j) else np.nan for j in range(1, len(bins))])
    vc = np.array([A[i == j, 3].mean() if np.any(i == j) else np.nan for j in range(1, len(bins))])
    prof[n] = (lat, v, vc)
    axs[0].plot(bins[:-1], lat, color=c, label=n); axs[1].plot(bins[:-1], v, color=c, label=n); axs[2].plot(bins[:-1], vc, color=c, label=n)
axs[0].set_ylabel("가로 위치 m (중심선 기준, 왼쪽 +)"); axs[1].set_ylabel("실제 속도 m/s"); axs[2].set_ylabel("MAP 속도 명령 m/s")
axs[2].set_xlabel("트랙 위치 s [m]")
for ax in axs:
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("깨끗한 맵: 트랙 위치별 주행선과 속도")
fig.tight_layout(); fig.savefig(head + "_profile.png", dpi=60); plt.close(fig)
if len(names) >= 2:
    a, b = names[0], names[1]
    dl = np.abs(prof[a][0] - prof[b][0]); dv = prof[a][1] - prof[b][1]
    print(f"{a} 대 {b}: 가로 위치 차이 중앙 {np.nanmedian(dl):.2f} m, p90 {np.nanpercentile(dl, 90):.2f} m | 속도 차이 평균 {np.nanmean(dv):+.2f} m/s, "
          f"가장 큰 곳 {np.nanmax(dv):+.2f} m/s (s = {bins[np.nanargmax(dv)]:.1f} m)")
print("그림", head + "_profile.png")
