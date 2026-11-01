"""콘솔 워커(SimWorker)로 체크포인트를 혼자 달리게 하고 그림 한 장: 맵 위 주행 경로(속도 색) + 0.5 s마다 정책이 낸 궤적(검은 선),
아래에 랩별 속도-거리 곡선 (11-01). 인자: 체크포인트 맵id 초 png"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
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


run, mp, secs, png = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
s = w.build_session(P.SessionConfig(run=run, map_name=mp, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=False, episode_s=max(60.0, secs + 5), randomize=False), 0)
env = s["env"]
from f1sim import maps, tracks
tr = maps.load(tracks.resolve(mp))                      # the map as loaded (the sim keeps only its tensors)
from matplotlib import font_manager
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
xs, ys, vs, laps_i, plans, hits = [], [], [], [], [], []
lap0 = int(env.last_result.lap[0]); prev = False
for k in range(int(secs / 0.025)):
    w._step_once(s)
    st = env.sim.state[0]
    xs.append(float(st[0])); ys.append(float(st[1])); vs.append(float(st[3])); laps_i.append(int(env.last_result.lap[0]))
    c = bool(env.last_result.collision[0])
    if c and not prev:
        hits.append((xs[-1], ys[-1]))
    prev = c
    if k % 20 == 0:
        fr = w._snapshot(s)
        if fr.get("plan") is not None:
            plans.append(np.asarray(fr["plan"], dtype=float)[:, :2])
xs, ys, vs, laps_i = map(np.asarray, (xs, ys, vs, laps_i))
occ = tr.occupancy; res = tr.resolution; H, W = occ.shape
fig = plt.figure(figsize=(13, 13))
ax = fig.add_axes([0.03, 0.30, 0.94, 0.68])
ax.imshow(~occ, cmap="gray", origin="lower", extent=[tr.origin[0], tr.origin[0] + W * res, tr.origin[1], tr.origin[1] + H * res], alpha=0.6)
sc = ax.scatter(xs, ys, c=vs, s=2, cmap="turbo", vmin=0, vmax=10)
for p in plans:
    ax.plot(p[:, 0], p[:, 1], "k-", lw=0.8)
for h in hits:
    ax.plot(*h, "rx", ms=12, mew=3)
pad = 1.0
ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.min() - pad, ys.max() + pad); ax.set_aspect("equal")
plt.colorbar(sc, ax=ax, fraction=0.03, label="속도 m/s")
ax.set_title(f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)} @ {mp}: {secs:.0f} s, 벽 충돌 시작 {len(hits)}회")
bx = fig.add_axes([0.06, 0.04, 0.9, 0.22])
d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
for L in np.unique(laps_i):
    m = laps_i == L
    if m.sum() > 10:
        bx.plot(d[m] - d[m][0], vs[m], lw=1, label=f"랩 {L}")
bx.set_xlabel("랩 안 거리 m"); bx.set_ylabel("속도 m/s"); bx.legend(fontsize=7, ncol=8); bx.grid(alpha=0.3)
fig.savefig(png, dpi=70)
print("그림", png, "| 궤적 수", len(plans))
