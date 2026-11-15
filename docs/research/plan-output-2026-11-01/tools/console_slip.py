"""(11-16, 사용자: "콘솔로 보면 존나 미끌미끌하게 가") 콘솔 워커로 혼자 달리며 차가 얼마나 미끄러지나: 미끄럼각 |beta| = |atan2(vy, vx)|
(차체 방향과 실제 진행 방향 차이), 옆가속 / 공칭 타이어 한계, 조향 명령의 좌우 뒤집힘(흔들림). 인자: checkpoint 지도 초 [시드]
F1SIM_PLAN_MODE = map / ideal / track, F1SIM_MAP_OLD=1 = 예전 MAP 설정, F1SIM_LAPS_RANDOMIZE=1 = 차 물리 무작위"""
import os, sys, math
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim import dynamics as dyn


class _N:
    def send(self, *a): pass
    def poll(self, *a): return False


run, mp, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
w = sim_worker.SimWorker(_N(), _N()); w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=False, episode_s=max(60.0, secs + 5),
                                    randomize=os.environ.get("F1SIM_LAPS_RANDOMIZE", "0") == "1",
                                    plan_speed_scale=float(os.environ.get("F1SIM_SLIP_SPEED_SCALE", "1.0"))), 0)
env = s["env"]
beta, alat, steer, v, along = [], [], [], [], []
vx_prev = None
for k in range(int(secs / 0.025)):
    w._step_once(s)
    st = env.sim.state[0]
    vx, vy, r = float(st[dyn.IVX]), float(st[dyn.IVY]), float(st[dyn.IR])
    ax_ = 0.0 if vx_prev is None else (vx - vx_prev) / 0.025
    vx_prev = vx
    if k > 80 and vx > 2.0:
        beta.append(math.degrees(math.atan2(vy, vx))); alat.append(abs(math.hypot(vx, vy) * r)); v.append(vx); along.append(ax_)
        steer.append(float(env.last_cmd_raw[0, 0]) if getattr(env, "last_cmd_raw", None) is not None else float(st[dyn.ISTEER]))
b = np.abs(np.array(beta)); a = np.array(alat); sd = np.array(steer); al = np.convolve(np.array(along), np.ones(8) / 8, mode="same")
slide = b > 10
if slide.any():
    print(f"  10도 넘게 미끄러진 순간 {slide.mean() * 100:.0f} %: 그중 감속 중(-2 m/s^2 아래) {np.mean(al[slide] < -2) * 100:.0f} %, "
          f"가속 중(+2 위) {np.mean(al[slide] > 2) * 100:.0f} %, 그 사이 {np.mean(np.abs(al[slide]) <= 2) * 100:.0f} % | "
          f"전체 시간 중 감속 {np.mean(al < -2) * 100:.0f} %, 가속 {np.mean(al > 2) * 100:.0f} %")
flip = np.mean(np.sign(np.diff(sd))[1:] * np.sign(np.diff(sd))[:-1] < 0) * 100      # steering-rate sign changes per step
print(f"{os.environ.get('F1SIM_PLAN_MODE', 'map')}{' (예전 MAP)' if os.environ.get('F1SIM_MAP_OLD') == '1' else ''}: 미끄럼각 |beta| 중앙 {np.median(b):.1f}도, "
      f"90 % {np.percentile(b, 90):.1f}도, 5도 넘는 시간 {np.mean(b > 5) * 100:.0f} % | 옆가속 90 % {np.percentile(a, 90):.1f} m/s^2 "
      f"(공칭 한계 8.2) | 조향 변화 방향이 걸음마다 뒤집힘 {flip:.0f} % | 평균 속도 {np.mean(v):.2f} m/s")
