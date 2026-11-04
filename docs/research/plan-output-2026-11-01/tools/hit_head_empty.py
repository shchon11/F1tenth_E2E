"""빈 트랙에서 critic의 hit head가 위험을 얼마나 높게 보나 (11-04 19:25). one-world는 --hit-shaping 1.0(Phi = -80 x P(hard hit within K),
solo policy 학습은 0): λ 0.95 advantage에서는 P를 낮추는 행동(= 느리게)이 매 결정 80 x ΔP를 받음. P가 obstacle·차 episode에서 '빠름 = 위험'으로 배워
빈 트랙에도 크게 나오면 빈 트랙에서도 감속에 reward가 붙음. 체크포인트를 빈 트랙에서 혼자 달리며 결정마다 P(hit)와 속도. 기각(미리 정함): 빈 트랙
평균 P < 0.005이고 속도 위 1/3과 아래 1/3의 P 차이 < 0.005면 이 원인 아님. 인자: 체크포인트(critic 쪽) 맵 초 [운전 체크포인트]"""
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


crit, mp, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
drv = sys.argv[4] if len(sys.argv) > 4 else crit
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=drv, map_name=mp, seed=0, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=False, episode_s=max(60.0, secs + 5), randomize=False), 0)
env = s["env"]
ck = torch.load(crit, map_location="cuda", weights_only=False)
ar = ck["args"]
sobs = SR.StateObs(env, scan=bool(ar.get("scan_obs", 0)), corridor=bool(ar.get("corridor_obs", 0)),
                   corridor_cars=bool(ar.get("corridor_cars", 0)), car_channel=bool(ar.get("car_channel", 0)),
                   car_st=bool(ar.get("car_st", 0)), scan_nocars=bool(ar.get("scan_nocars", 0)),
                   corridor_car_st=bool(ar.get("corridor_car_st", 0)), blank_opp=bool(ar.get("blank_opp", 0)),
                   car_st_line=bool(ar.get("car_st_line", 0)), vis_actor=bool(ar.get("vis_actor", 0)),
                   car_slots=bool(ar.get("car_slots", 0)), t_beams=int(ar.get("beams", 108)))
A = SR.act_dim_of(ar)
o = sobs(torch.zeros(1, A, device="cuda"))
net = SR.ActorCritic(o.shape[1], act_dim=A, hidden=int(ar.get("hidden", 256) or 256), residual=SR.residual_of(ar)).cuda()
net.load_state_dict(ck["model"], strict=False); net.eval()
nrm = SR.RunningNorm(o.shape[1], "cuda"); nrm.load_state_dict(ck["norm"])
rows = []
for k in range(int(secs / 0.025)):
    w._step_once(s)
    if k % 2 or k * 0.025 < 3.0:
        continue
    with torch.no_grad():
        o = sobs(torch.zeros(1, A, device="cuda"))
        v, hl = net.value_and_hit(nrm(o))
    rows.append((float(torch.sigmoid(hl)[0]), float(env.sim.state[0, 3]), float(v[0])))
a = np.array(rows)
lo, hi = np.quantile(a[:, 1], [1 / 3, 2 / 3])
print(f"critic {os.path.basename(os.path.dirname(crit))}/{os.path.basename(crit)}, 운전 {os.path.basename(os.path.dirname(drv))}/{os.path.basename(drv)}: "
      f"결정 {len(a)}개 | 빈 트랙 P(hit) 평균 {a[:, 0].mean():.4f}, 90 % {np.quantile(a[:, 0], 0.9):.4f}, 최대 {a[:, 0].max():.4f} | "
      f"속도 아래 1/3 P {a[a[:, 1] <= lo, 0].mean():.4f}, 위 1/3 P {a[a[:, 1] >= hi, 0].mean():.4f} | "
      f"80 x P 평균 {80 * a[:, 0].mean():.2f} (결정당 progress 약 {0.05 * a[:, 1].mean():.2f} m)", flush=True)
