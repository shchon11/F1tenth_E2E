"""콘솔 워커(SimWorker)로 체크포인트를 혼자 몇 바퀴: 랩 시간, 벽 충돌 시작 수, 궤적 그림 데이터 유무 (11-01)"""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


run, mp, secs, png = sys.argv[1], sys.argv[2], float(sys.argv[3]), (sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else "")
proc = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False          # 1: the training's obstacles (procedural)
w = sim_worker.SimWorker(_N(), _N()); logs = []
w.say = lambda k, **f: logs.append(f.get("text") or ""); w.stage = lambda *a, **k: None; w._hold_parked_ok = True
w._overlay = dict(getattr(w, "_overlay", {}) or {}, plan=True)
s = w.build_session(P.SessionConfig(run=run, map_name=mp, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=proc, episode_s=max(60.0, secs + 5), randomize=False), 0)
env = s["env"]
print("세션:", s["info_line"], "| 방식", s["mode"], "| 결정", env.ecfg.decision_every, "| 계획", env.ecfg.speed_mode, env.ecfg.plan_follower)
lap0 = int(env.last_result.lap[0]); t_last = 0.0; laps = []; hits = 0; prev_col = False; vmax = 0.0; vs = []
N = int(secs / 0.025); t0 = time.time()
for k in range(N):
    w._step_once(s)
    r = env.last_result
    lp = int(r.lap[0])
    if lp != lap0:
        t = (k + 1) * 0.025
        if lap0 >= 1 or laps or t_last > 0:
            laps.append(t - t_last)
        t_last, lap0 = t, lp
    c = bool(r.collision[0]); hits += int(c and not prev_col); prev_col = c
    v = float(env.sim.state[0, 3]); vmax = max(vmax, v); vs.append(v)
fr = w._snapshot(s)
print(f"{secs:.0f} s 주행 ({time.time() - t0:.0f} s 걸림): 랩 {[round(x, 2) for x in laps]} | 평균 속도 {np.mean(vs):.2f} m/s, 최고 {vmax:.1f} | 벽 충돌 시작 {hits}회 | "
      f"궤적 그림 점 {None if fr.get('plan') is None else len(fr['plan'])}")
if png:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from f1sim.viewer.console.overlays import PolicyInputPanel
    from f1sim.viewer.console.session import _plan_in_body_frame
    pnl = PolicyInputPanel(); pnl.resize(520, 560)
    pnl.set_data(fr.get("scan"), None, _plan_in_body_frame(fr), 9.0, None, drv=None); pnl.span = 8.0
    pnl.grab().save(png); print("그림", png)
