"""콘솔 경로(물리 차 + 제어기)에서 차가 흔들리나 + 콘솔 배속 (11-10, 사용자: "콘솔은 뭐 렉이 더 걸리냐? ... 콘솔로 직접 보는거하고 갈수록 차가
존나 불안하게 흔들리면서 가냐"). 콘솔 워커(SimWorker.build_session, 콘솔과 같은 길)로 혼자 빈 트랙을 checkpoint마다 같은 시드로 30 s 달리며
걸음마다 차의 실제 조향각과 요 각속도를 모음:
  조향 흔들림 = 0.1 s마다 실제 조향각 변화의 평균 크기 [deg], 지그재그 비율 = 0.1 s 변화의 부호가 바로 뒤집힌 비율(둘 다 0.5 deg 넘게),
  요 각속도 떨림 = 요 각속도에서 1 s 평균을 뺀 나머지의 표준편차 [deg/s], 배속 = 시뮬 시간 / 실제 걸린 시간(그림 그리기 없이)
인자: checkpoint1,checkpoint2,... [초, 기본 30]   (F1SIM_PLAN_MODE = track(기본, 새 제어기) / map / ideal)"""
import os, sys, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim import dynamics as dyn


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


cks = sys.argv[1].split(",")
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
mode = os.environ.get("F1SIM_PLAN_MODE", "track")
for run in cks:
    w = sim_worker.SimWorker(_N(), _N())
    w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
    s = w.build_session(P.SessionConfig(run=run, map_name="real/iccas25", seed=0, races=1, cars_per_race=1, device="cuda",
                                        compile=False, collision_mode="soft", procedural=False, episode_s=120.0, randomize=False), 0)
    env = s["env"]
    for _ in range(80):                                            # 2 s to get going
        w._step_once(s)
    n = int(secs / 0.025)
    SD, YR = [], []
    torch.cuda.synchronize(); t0 = time.time()
    for k in range(n):
        w._step_once(s)
        SD.append(float(env.sim.state[0, dyn.ISTEER])); YR.append(float(env.sim.state[0, dyn.IR]))
    torch.cuda.synchronize(); dt = time.time() - t0
    sd = np.degrees(np.array(SD)); yr = np.degrees(np.array(YR))
    s10 = sd[::4]                                                  # every 0.1 s
    dd = np.diff(s10)
    jit = float(np.mean(np.abs(dd)))
    zz = float(np.mean((dd[1:] * dd[:-1] < 0) & (np.abs(dd[1:]) > 0.5) & (np.abs(dd[:-1]) > 0.5)))
    yrs = np.convolve(yr, np.ones(40) / 40, "same")
    wob = float(np.std((yr - yrs)[20:-20]))
    name = f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run).replace('plan_', '').replace('.pt', '')}"
    print(f"[{mode}] {name}: 조향 흔들림 {jit:.2f} deg/0.1 s | 지그재그 {zz * 100:.0f} % | 요 각속도 떨림 {wob:.1f} deg/s | "
          f"배속 {n * 0.025 / dt:.2f}x (걸음당 {dt / n * 1000:.1f} ms)", flush=True)
    del w, s, env
    torch.cuda.empty_cache()
