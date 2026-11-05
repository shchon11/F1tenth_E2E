"""policy가 obstacle 입력에 반응하나 (11-05 11:55, 사용자: 콘솔에서 "뭐 피하려는 움직임조차 없었음"). 콘솔 세션(학습 obstacle, layout 시드 몇 개)에서
checkpoint가 직접 달리며, 주변에 obstacle이 있는 결정(CalmGate near)마다 평균 action을 두 번 계산: 원래 입력 / 보이는 obstacle·차 입력(StateObs.vis0
이후 칸)만 빈 값(0)으로. 두 action의 차이(조향 칸 4개, 속도 칸 4개 각각 평균 절댓값)가 0에 가까우면 그 checkpoint는 보이는 obstacle 입력을 무시하는
것. scan(LiDAR)은 그대로 두므로 scan으로 반응하는 몫은 여기 안 잡힘. 인자: 맵id 초 시드들(쉼표) 체크포인트들..."""
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
for run in runs:
    dk, ds, n_near, n_all = [], [], 0, 0
    for seed in seeds:
        w = sim_worker.SimWorker(_N(), _N())
        w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
        s = w.build_session(P.SessionConfig(run=run, map_name=mp, seed=seed, races=1, cars_per_race=1, device="cuda", compile=False,
                                            collision_mode="soft", procedural=True, episode_s=max(60.0, secs + 5),
                                            randomize=False), 0)
        env = s["env"]; tp = s["act_fn"].tp
        gate = SR.CalmGate(env, 10.0, 1, visible=True)   # hold 1: near now (hold 0 never opens the gate)
        v0 = tp.sobs.vis0
        K = SR.PLAN_KAPPA_DIMS
        for k in range(int(secs / 0.025)):
            w._step_once(s)
            if k % 2:
                continue
            near = bool(gate.step()[0])
            n_all += 1
            if not near:
                continue
            n_near += 1
            with torch.no_grad():
                o = tp.o
                oz = o.clone(); oz[:, v0:] = 0.0
                a1 = tp.policy(o)[0]; a0 = tp.policy(oz)[0]
            d = (a1 - a0).abs().cpu().numpy()
            dk.append(d[:K].mean()); ds.append(d[K:].mean())
        del s, env, w
        torch.cuda.empty_cache()
    name = f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)}"
    print(f"{name}: obstacle 근처 결정 {n_near}/{n_all} | 보이는 obstacle 입력을 비웠을 때 action 변화(평균 절댓값, action 범위 ±1): "
          f"조향 칸 {np.mean(dk):.3f}, 속도 칸 {np.mean(ds):.3f} | 조향 칸 변화가 0.05 넘는 결정 {np.mean(np.array(dk) > 0.05) * 100:.0f} %", flush=True)
