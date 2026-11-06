"""콘솔 세션과 학습 env에서 policy 입력이 다른가 (11-06 14:40). 학습 env를 콘솔처럼 맞춰도(상자 전부, 원래 트랙, 상대 없음, 차 1대) 2 s 넘게
서 있는 시간 6-8 %, 콘솔 50 % 안팎 -> 입력 칸 단위로: 각 env에서 policy가 직접 몰며 결정마다 입력을 모아, 학습 정규화(평균·표준편차)로 바꾼
값의 칸별 평균을 비교, 차이가 큰 칸과 그 칸이 무엇인지(블록) 출력. actor가 못 보는 칸(actor_keep 0)은 뺌. 인자: 체크포인트 맵id 초"""
import os, sys, argparse
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.viewer import sim_worker
from f1sim.viewer.console import protocol as P
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime


class _N:
    def send(self, *a):
        pass

    def poll(self, *a):
        return False


path, mp, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
# console
w = sim_worker.SimWorker(_N(), _N())
w.say = lambda k, **f: None; w.stage = lambda *a, **k: None; w._hold_parked_ok = True
s = w.build_session(P.SessionConfig(run=path, map_name=mp, seed=777, races=1, cars_per_race=1, device="cuda", compile=False,
                                    collision_mode="soft", procedural=True, episode_s=3600.0, randomize=False), 0)
tp_c = s["act_fn"].tp
oc = []
for k in range(int(secs / 0.025)):
    w._step_once(s)
    if k % 2 == 0 and k > 40:
        oc.append(tp_c.nrm(tp_c.o)[0].detach().cpu().numpy())
oc = np.array(oc)
keep = tp_c.net.actor_keep.detach().cpu().numpy() > 0.5
# training env (eval-like, one car)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = 32, 5, "cuda"
a.prop_count, a.prop_fraction, a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop, a.stage = "fixed", 1.0, 0.0, 0, 0.0, 0.0, "props"
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=5)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel())
act = torch.zeros(B, tp.A, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
ot = []
for k in range(int(secs / 0.05)):
    act = tp(act); act_full[lid] = act
    for _ in range(int(a.repeat)):
        env.step(act_full)
    if k > 20:
        ot.append(tp.nrm(tp.o).detach().cpu().numpy())
ot = np.concatenate(ot, 0)
so = tp_c.sobs
n_scan0 = so.scan0
blocks = [("자기 상태·트랙 앞", 0, n_scan0)]
nb = env.sim.lidar.angles.numel()
blocks += [("LiDAR 전체 거리", n_scan0, n_scan0 + nb), ("LiDAR 상자 점", n_scan0 + nb, n_scan0 + 2 * nb), ("LiDAR 차 점", n_scan0 + 2 * nb, n_scan0 + 3 * nb)]
blocks += [("corridor (전체, critic 전용 포함)", n_scan0 + 3 * nb, so.vis0 or oc.shape[1]), ("보이는 블록", so.vis0 or oc.shape[1], oc.shape[1])]
mc, mt = oc.mean(0), ot.mean(0)
sc, st_ = oc.std(0), ot.std(0)
d = np.abs(mc - mt)
print(f"콘솔 결정 {len(oc)}개, 학습 env 결정 {len(ot)}개, 입력 {oc.shape[1]}칸 (actor가 보는 칸 {int(keep.sum())})")
for name, lo, hi in blocks:
    m = np.zeros_like(keep); m[lo:hi] = True; m &= keep
    if m.sum() == 0:
        print(f"  {name}: actor가 보는 칸 없음"); continue
    print(f"  {name} ({int(m.sum())}칸): 평균 차이 평균 {d[m].mean():.2f}, 최대 {d[m].max():.2f} (칸 {int(np.argmax(np.where(m, d, -1)))})"
          f" | 콘솔 표준편차 평균 {sc[m].mean():.2f}, 학습 {st_[m].mean():.2f}")
top = np.argsort(-np.where(keep, d, -1))[:12]
print("  차이 큰 칸:", ", ".join(f"{i}: 콘솔 {mc[i]:+.2f} 학습 {mt[i]:+.2f}" for i in top))
