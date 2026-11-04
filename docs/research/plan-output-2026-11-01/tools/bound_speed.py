"""policy 출력 범위 처리가 최고 속도를 깎나 (11-04 19:15). 빠른 solo policy에서 시작한 네트워크 하나(srp25ows)는 빈 트랙 lap이
6.50 -> 6.71 -> 6.90 s(u50, u100), 라인은 같고 빠른 구간만 0.3~0.5 m/s 낮아짐(speed_profile.py). solo policy는 --soft-bound 0 / --bound-loss 0으로
학습했는데 one-world는 둘 다 켬: soft bound는 [hi - m, hi](속도 m 0.2) 안에서 이미 눌러서 평균 hi는 hi - 0.048로 실행되고, bound loss는 raw 평균을
hi 이하로 당김 -> 평균으로는 최고 속도를 영영 못 냄. 학습 없이 solo policy를 세 가지로 빈 트랙 60 s: (a) 그대로(clamp), (b) soft bound를 씌움
(one-world 시작점), (c) soft bound + raw 평균을 [lo, hi]로 자름(bound loss가 다 당긴 상태). 결정마다 속도 4칸의 raw 평균이 hi - m / hi를 넘는 비율도.
기각(미리 정함): (c)가 (a)보다 0.1 s 넘게 느리지 않으면 이 원인은 아님. 인자: 체크포인트 맵 초 방식(a|b|c)"""
import os, sys, time
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


run, mp, secs, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
if mode == "c":
    _sb = SR.soft_bound
    SR.soft_bound = lambda z, lo, hi, m: _sb(torch.maximum(torch.minimum(z, hi), lo), lo, hi, m)
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=0, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=False, episode_s=max(60.0, secs + 5), randomize=False), 0)
env = s["env"]; runner = s["act_fn"].tp
net = runner.net
A = int(net.log_std.numel())
if mode in ("b", "c"):
    net.set_mean_bounds(*SR.mean_bounds(runner.ar, A))
hi = net.mean_hi[SR.PLAN_KAPPA_DIMS:] if mode != "a" else torch.ones(A - SR.PLAN_KAPPA_DIMS, device=net.mean_hi.device)
raw = []
lap0 = int(env.last_result.lap[0]); t_last = 0.0; laps = []; vmax = 0.0; vs = []
for k in range(int(secs / 0.025)):
    w._step_once(s)
    if net.raw_mean is not None and k % 2 == 0:
        raw.append(net.raw_mean[0, SR.PLAN_KAPPA_DIMS:].detach().float().cpu().numpy())
    r = env.last_result
    lp = int(r.lap[0])
    if lp != lap0:
        t = (k + 1) * 0.025
        if lap0 >= 1 or laps or t_last > 0:
            laps.append(t - t_last)
        t_last, lap0 = t, lp
    v = float(env.sim.state[0, 3]); vmax = max(vmax, v); vs.append(v)
raw = np.array(raw)
print(f"방식 {mode}: 랩 {[round(x, 2) for x in laps]} | 평균 속도 {np.mean(vs):.2f} m/s, 최고 {vmax:.1f} | 속도 raw 평균 > 0.8 비율 "
      f"{np.mean(raw > 0.8) * 100:.0f} %, > 1.0 비율 {np.mean(raw > 1.0) * 100:.0f} %, 최대 {raw.max():.2f}", flush=True)
