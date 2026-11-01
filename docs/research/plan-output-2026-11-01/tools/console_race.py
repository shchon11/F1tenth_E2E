"""콘솔 워커(SimWorker)로 1:1 경주 (11-02): 내 차(0번) 랩, 벽·상자 충돌 시작, 차 접촉 시작(강한 것: 닫히는 속도 >= 1.5 m/s),
추월 / 추월당함, 끝 간격. 인자: 체크포인트 맵id 초 상대종류 상대속도배율 [학습장애물 0/1]"""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, kind, sc = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], float(sys.argv[5])
proc = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, races=1, cars_per_race=2, device="cuda", compile=False,
                                    collision_mode="soft", procedural=proc, episode_s=max(60.0, secs + 5), randomize=False,
                                    opponent_slots=[{"kind": kind, "speed_scale": [sc, sc]}]), 0)
env = s["env"]
r = env.last_result
gap = float(r.s[0] - r.s[1])                       # >0: I am ahead along the lap (start offset, wrapped below)
L = float(env.sim.track.length) if hasattr(env.sim.track, "length") else None
lap0 = int(r.lap[0]); t_last = 0.0; laps = []
wall = car = car_hard = passes = passed = 0
pw = pc = False
t0 = time.time()
for k in range(int(secs / 0.025)):
    w._step_once(s)
    r = env.last_result
    gap_prev = gap
    gap += float(r.progress[0] - r.progress[1])
    if gap_prev < 0 <= gap:
        passes += 1
    elif gap_prev >= 0 > gap:
        passed += 1
    lp = int(r.lap[0])
    if lp != lap0:
        tt = (k + 1) * 0.025
        if lap0 >= 1 or laps or t_last > 0:
            laps.append(tt - t_last)
        t_last, lap0 = tt, lp
    wc = bool(r.wall_collision[0]) if r.wall_collision is not None else bool(r.collision[0])
    cc = bool(r.car_collision[0]) if r.car_collision is not None else False
    wall += int(wc and not pw); pw = wc
    if cc and not pc:
        car += 1
        imp = getattr(r, "car_impact", None)
        car_hard += int(imp is not None and float(imp[0]) >= 1.5)
    pc = cc
print(f"{secs:.0f} s 경주 vs {kind} x{sc} ({'학습 장애물' if proc else '장애물 없음'}, {time.time() - t0:.0f} s 걸림): "
      f"랩 {[round(x, 2) for x in laps]} | 벽·상자 충돌 시작 {wall} | 차 접촉 시작 {car} (강한 것 {car_hard}) | "
      f"추월 {passes} / 추월당함 {passed} | 끝 간격 {gap:+.1f} m")
