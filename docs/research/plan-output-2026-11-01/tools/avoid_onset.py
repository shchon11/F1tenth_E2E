"""라인 위 상자를 언제부터 피하나 (11-05 16:15, 사용자: "애초에 장애물이 저 멀리 있어서 한참 전부터 피할 수 있는데"). 콘솔 세션(학습 obstacle,
layout 시드 몇 개)에서 checkpoint가 달리며, 결정마다 앞 12 m 안의 raceline 위 상자(상자 중심이 raceline에서 옆으로 0.5 m 안)마다 (상자까지 트랙 거리,
차와 상자의 옆 간격 = |차 가로 위치 - 상자 가로 위치|)를 모음. 거리 구간별 평균 옆 간격: 일찍 피하면 먼 구간부터 간격이 벌어지고, 늦게 피하면
1~2 m에서야 벌어지거나 끝까지 작음. 실제로 달린 경로라 제어기와 무관. 인자: 맵id 초 시드들(쉼표) 체크포인트들..."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.learn import state_rl as SR


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


mp, secs, seeds, runs = sys.argv[1], float(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")], sys.argv[4:]
BINS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 12)]
for run in runs:
    rows = []
    for seed in seeds:
        w = sim_worker.SimWorker(_N(), _N())
        w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
        s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                            collision_mode="soft", procedural=True, episode_s=max(60.0, secs + 5),
                                            randomize=False), 0)
        env = s["env"]
        pl = SR._line_planner(env)
        tid = env.sim.tid[:1]
        L = float(env.sim.track.length[env.sim.tid[0]])
        po = env.procedural
        for k in range(int(secs / 0.025)):
            w._step_once(s)
            if k % 2 or k * 0.025 < 2.0:
                continue
            with torch.no_grad():
                sc, lc, _ = env.sim.track.project(env.sim.state[:1, :2], tid)
                live = po.p_sid[0] >= 0
                if not bool(live.any()):
                    continue
                pp = po.p_poses[0][live][:, :2]
                sp, lp, _ = env.sim.track.project(pp, tid.expand(pp.shape[0]))
                rl = pl.rl_offset(sp, tid.expand(pp.shape[0]))
                ds = torch.remainder(sp - sc[0], L)
                on = ((lp - rl).abs() < 0.5) & (ds < 12.0)
                for d_, gap in zip(ds[on].tolist(), (lc[0] - lp[on]).abs().tolist()):
                    rows.append((d_, gap))
        del s, env, w
        torch.cuda.empty_cache()
    a = np.array(rows) if rows else np.zeros((0, 2))
    name = f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)}"
    parts = []
    for lo, hi in BINS:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi) if len(a) else np.zeros(0, bool)
        parts.append(f"{lo}~{hi} m: {a[m, 1].mean():.2f} m" if m.sum() >= 5 else f"{lo}~{hi} m: -")
    print(f"{name}: raceline 위 상자까지 거리별 평균 옆 간격 | " + " | ".join(parts) + f" | 표본 {len(a)}", flush=True)
