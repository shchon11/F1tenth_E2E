"""MPC가 조향 끝(24 deg)에 붙어 있는 구간에서 무슨 일이 있나 (11-10, 첫 판: 콘솔 경로 u300 혼자 빈 트랙, 3.7-4.6 s와 10.3-11.7 s에
24 deg로 1초 넘게 붙음 -- plan의 운동학 조향은 6 deg 안팎). 콘솔 워커(console_wobble과 같은 길)로 몰며 걸음마다: 보낸 조향 명령,
실제 조향각, 지금 시각 plan 자리에 대한 옆 오차(+ = 차가 plan 왼쪽)·방향 오차, 요 각속도(차 / plan), 속도(차 / plan), MPC가 예측한
0.5 s 뒤 옆 오차. 인자: checkpoint [초, 기본 12] [시작, 끝: 표로 찍을 구간, 기본 3 5]"""
import os, sys, math
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


run = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
t_a = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
t_b = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name="real/iccas25", seed=0, races=1, cars_per_race=1, device="cuda",
                                    compile=False, collision_mode="soft", procedural=False, episode_s=120.0, randomize=False), 0)
env = s["env"]
pr = s["act_fn"]
f = pr.follower
for _ in range(80):
    w._step_once(s)
rows = []
for k in range(int(secs / 0.025)):
    w._step_once(s)
    st = env.sim.state[0].float().cpu()
    kf = (f.k.float() - 1.0).to(f.X.device)                      # the step just commanded
    xr, yr, ps, vr, rr = [a[0, 0].item() for a in f._ref(kf[:, None])]
    ex, ey = st[0].item() - xr, st[1].item() - yr
    e_lat = -math.sin(ps) * ex + math.cos(ps) * ey
    e_head = (st[2].item() - ps + math.pi) % (2 * math.pi) - math.pi
    cmd = f.hist[0, 0].item() if hasattr(f, "hist") else float("nan")
    rows.append((2.0 + (k + 1) * 0.025, math.degrees(cmd), math.degrees(st[dyn.ISTEER].item()), e_lat * 100,
                 math.degrees(e_head), math.degrees(st[dyn.IR].item()), math.degrees(rr), st[dyn.IVX].item(), vr,
                 math.degrees(st[dyn.IVY].item() / max(0.3, st[dyn.IVX].item()))))
R = np.array(rows)
sat = np.abs(R[:, 1]) > 23.5
print(f"조향 명령이 끝(23.5 deg 넘게)에 붙은 걸음 {sat.mean() * 100:.0f} % | 옆 오차 중앙 {np.median(np.abs(R[:, 3])):.1f} cm, "
      f"끝에 붙었을 때 {np.median(np.abs(R[sat, 3])) if sat.any() else float('nan'):.1f} cm", flush=True)
print("   t  | 명령  실제 | 옆 오차 cm | 방향 오차 | 요 각속도 차 / plan | 속도 차 / plan | 옆미끄럼각", flush=True)
for r in R:
    if t_a <= r[0] <= t_b and abs((r[0] * 40) % 4) < 1e-6:
        print(f"{r[0]:5.2f} | {r[1]:5.1f} {r[2]:5.1f} | {r[3]:+7.1f} | {r[4]:+6.1f} | {r[5]:+6.0f} / {r[6]:+6.0f} | "
              f"{r[7]:4.2f} / {r[8]:4.2f} | {r[9]:+5.1f}", flush=True)
