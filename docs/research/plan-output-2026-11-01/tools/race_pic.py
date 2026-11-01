"""1:1 경주 장면 그림 (11-02): 콘솔 워커로 내 차(0)와 상대(1)를 달리게 하고, 두 차의 경로(시간 색)와 속도-시간, 앞뒤 간격-시간,
랩 카운터가 바뀐 지점(×)을 그린다. 인자: 체크포인트 맵id 초 상대종류 속도배율 png"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim import maps, tracks


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


def main():
    run, mp, secs, kind, sc, png = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], float(sys.argv[5]), sys.argv[6]
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    s = w.build_session(P.SessionConfig(run=run, map_name=mp, races=1, cars_per_race=2, device="cuda", compile=False,
                                        collision_mode="soft", procedural=False, episode_s=max(60.0, secs + 5),
                                        randomize=False, opponent_slots=[{"kind": kind, "speed_scale": [sc, sc]}]), 0)
    env = s["env"]
    from matplotlib import font_manager
    for f in ("Pretendard", "NanumGothic", "Noto Sans CJK KR"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f; break
    tr = maps.load(tracks.resolve(mp))
    X, V, G, LAP = [], [], [], []
    r = env.last_result
    gap = float(r.s[0] - r.s[1])
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        r = env.last_result
        st = env.sim.state
        gap += float(r.progress[0] - r.progress[1])
        X.append([float(st[0, 0]), float(st[0, 1]), float(st[1, 0]), float(st[1, 1])])
        V.append([float(st[0, 3]), float(st[1, 3])]); G.append(gap); LAP.append(int(r.lap[0]))
    X, V, G, LAP = map(np.asarray, (X, V, G, LAP))
    t = np.arange(len(G)) * 0.025
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_axes([0.04, 0.38, 0.92, 0.6])
    occ = tr.occupancy; res = tr.resolution; H, W_ = occ.shape
    ax.imshow(~occ, cmap="gray", origin="lower", extent=[tr.origin[0], tr.origin[0] + W_ * res, tr.origin[1], tr.origin[1] + H * res], alpha=0.5)
    ax.scatter(X[:, 2], X[:, 3], c="orange", s=1, alpha=0.4, label="상대")
    sc_ = ax.scatter(X[:, 0], X[:, 1], c=t, s=2, cmap="viridis", label="내 차 (색 = 시각)")
    ch = np.nonzero(np.diff(LAP))[0]
    ax.plot(X[ch + 1, 0], X[ch + 1, 1], "rx", ms=10, mew=2, label="랩 카운터 바뀜")
    pad = 1.0
    ax.set_xlim(X[:, ::2].min() - pad, X[:, ::2].max() + pad); ax.set_ylim(X[:, 1::2].min() - pad, X[:, 1::2].max() + pad)
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8); plt.colorbar(sc_, ax=ax, fraction=0.03, label="s")
    ax.set_title(f"{os.path.basename(os.path.dirname(run))} @ {mp} vs {kind} x{sc}")
    b1 = fig.add_axes([0.06, 0.20, 0.9, 0.13]); b1.plot(t, V[:, 0], label="내 차"); b1.plot(t, V[:, 1], label="상대")
    b1.set_ylabel("속도 m/s"); b1.legend(fontsize=8); b1.grid(alpha=0.3)
    b2 = fig.add_axes([0.06, 0.04, 0.9, 0.13]); b2.plot(t, G); b2.axhline(0, color="k", lw=0.6)
    b2.set_ylabel("간격 m (+ 앞섬)"); b2.set_xlabel("시각 s"); b2.grid(alpha=0.3)
    for c in ch:
        b2.axvline(t[c], color="r", lw=0.6, alpha=0.5)
    fig.savefig(png, dpi=65)
    print("그림", png, "| 랩 카운터 변화", len(ch), "| 간격 부호 바뀜", int((np.diff(np.sign(G)) != 0).sum()))


if __name__ == "__main__":
    main()
