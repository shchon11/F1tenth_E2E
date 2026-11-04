"""느린 상대를 멀리서 못 따라잡나 (11-04 04:00, 사용자: "상대가 앞에 계속 달리고 있을 때 정책이 꽤 멀리 있는데에도 라인을 너무 이상하게 잡아가면서
상대가 충분히 느린데에도 너무 못따라잡고 불안하게 주행하는 양상이 꽤 보였었음"). 가설: base policy를 섞는 checkpoint는 상대가 앞 10 m 안이면
(장애물 없어도) race policy가 넘겨받는데, race policy는 빈 트랙 주행을 학습한 적 없음(빈 구간은 늘 base policy). 콘솔 race(console_races.py와
같은 세션, obstacle 없음)를 달리며 결정마다: 상대까지 arc 간격(+ = 상대가 앞), gating(1 = race policy), 속도 / 그 자리 min-time raceline 속도,
raceline에서 가로 거리. 간격 구간별로 묶음. 인자: 체크포인트 맵id 판당초 판수 상대종류 상대속도배율 시드"""
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


run, mp, secs, n_race, kind, sc, seed = (sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
                                         float(sys.argv[6]), int(sys.argv[7]))
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=2, device="cuda", compile=False,
                                    collision_mode="soft", procedural=False, episode_s=3600.0, randomize=False,
                                    opponent_slots=[{"kind": kind, "speed_scale": [sc, sc]}]), 0)
env = s["env"]; runner = s["act_fn"]
pl = SR._line_planner(env)
tid = env.sim.tid[:1]
rows = []
for R in range(n_race):
    if R > 0:
        s["obs"], _ = env.reset()
    gap = None
    for k in range(int(secs / 0.025)):
        w._step_once(s)
        r = env.last_result
        if gap is None:
            Lt = float(env.sim.track.length[env.sim.tid[0]])
            gap = float(r.s[1] - r.s[0]); gap = (gap + 0.5 * Lt) % Lt - 0.5 * Lt
        else:
            gap += float(r.progress[1] - r.progress[0])
        if k % 2 or k * 0.025 < 3.0:
            continue
        st = env.sim.state
        s0 = r.s[:1]
        v_rl = float(pl.rl_speed(s0, tid)[0])
        _, lat, _ = env.sim.track.project(st[:1, :2], tid)
        off = abs(float(lat[0] - pl.rl_offset(s0, tid)[0]))
        g = runner.gate
        rows.append((gap, int(bool(g[0])) if g is not None else -1, float(st[0, 3]), v_rl, off))
a = np.array(rows)
print(f"== {os.path.basename(os.path.dirname(run))}/{os.path.basename(run)} vs {kind} x{sc}: {n_race}판 x {secs:.0f} s, 결정 {len(a)}개 (출발 3 s 뒤부터)", flush=True)
print("  상대 간격(+ 앞)   비율   race policy 운전   속도/raceline 속도   raceline에서 가로 거리(중앙)", flush=True)
for lo, hi in ((-50, -1), (-1, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 50)):
    m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    if m.sum() < 5:
        continue
    gate = a[m, 1]
    sp = lambda q: f"{np.mean(a[q, 2] / np.maximum(a[q, 3], 0.5)):.2f} / {np.median(a[q, 4]):.2f} m ({int(q.sum())})" if q.sum() >= 5 else "-"
    print(f"  {lo:+4d} ~ {hi:+4d} m  {m.mean() * 100:5.1f} %   {np.mean(gate == 1) * 100 if (gate >= 0).any() else float('nan'):5.1f} %"
          f"            {np.mean(a[m, 2] / np.maximum(a[m, 3], 0.5)):.2f}               {np.median(a[m, 4]):.2f} m"
          f"   | base policy 운전 {sp(m & (a[:, 1] == 0))}, race policy 운전 {sp(m & (a[:, 1] == 1))}", flush=True)
