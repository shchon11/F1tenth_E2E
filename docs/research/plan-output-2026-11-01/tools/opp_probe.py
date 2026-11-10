"""policy가 앞 상대 차의 자리·상대 속도를 아나 (11-10, 상대 차 접촉 장면: 느린 앞차 옆을 0.2-0.4 m로 지나가다 닿고, 닿기 0.5 s 전 plan이
상대 차 자리를 그대로 지나감 -> 가설: 상대 차가 얼마나 느리고 어디로 가는지 모름). 학습 세계(plan_rl.build, 학습 조건)를 policy로 몰며,
상대 차가 가까이(6 m 안, 차 기준 앞 -1 m부터) 있는 결정마다 정답 = critic이 받는 상대 차 칸(가장 가까운 상대: 앞 dx, 옆 dy, 상대 속도
dvx, dvy, 차 기준)과, policy 안의 세 가지를 모음: 지금 LiDAR·상태로 뽑은 특징(기억 더하기 전) / 기억 상태 / 둘을 더한 특징(운전이
읽는 것). 앞 70 %로 선형 회귀(능선)를 맞추고 뒤 30 %에서 설명력(R²)을 잼 -- 1이면 그 값을 다 앎, 0이면 모름.
인자: checkpoint [환경 수, 기본 512] [초, 기본 40]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim.gym_env import PRIV_V2_EGO_KEYS, PRIV_V2_OPP_KEYS
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path = sys.argv[1]
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 512
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, n_env, 31, dev, fmt="v2")
lid = env.learner_ids
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
o0 = len(PRIV_V2_EGO_KEYS); KO = len(PRIV_V2_OPP_KEYS)
ki = {k: o0 + PRIV_V2_OPP_KEYS.index(k) for k in ("present", "dx", "dy", "dist", "dvx", "dvy")}
F0, HM, FC, Y = [], [], [], []
for kd in range(int(secs / PS.DT)):
    pv = env.privileged_v2()[lid].float()                       # the nearest opponent first, units /5
    f0 = actor.features(obs.scans, obs.state())
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    h = obs.mem_h
    fc = f0 + actor.mem_out(h) if getattr(actor, "memory", 0) else f0
    near = (pv[:, ki["present"]] > 0.5) & (pv[:, ki["dist"]] * 5.0 < 6.0) & (pv[:, ki["dx"]] * 5.0 > -1.0)
    if bool(near.any()):
        F0.append(f0[near]); FC.append(fc[near]); HM.append(h[near] if h is not None else f0[near][:, :0])
        Y.append(torch.stack([pv[near, ki[k]] * 5.0 for k in ("dx", "dy", "dvx", "dvy")], 1))
    drv.decision(torch.tanh(mean), p)
F0, FC, HM, Y = torch.cat(F0), torch.cat(FC), torch.cat(HM), torch.cat(Y)
N = Y.shape[0]
g = torch.Generator(device="cpu"); g.manual_seed(0)
perm = torch.randperm(N, generator=g).to(dev)
ntr = int(0.7 * N); tr, te = perm[:ntr], perm[ntr:]


def r2(X):
    if X.shape[1] == 0:
        return [float("nan")] * Y.shape[1]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    Xn = torch.cat([(X - mu) / sd, torch.ones(N, 1, device=dev)], 1)
    A = Xn[tr]; lam = 1.0
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=dev), A.T @ Y[tr])
    P = Xn[te] @ W
    res = ((P - Y[te]) ** 2).sum(0); tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
    return (1.0 - res / tot).tolist()


name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
out = {"지금 특징": r2(F0), "기억 상태": r2(HM), "운전이 읽는 특징": r2(FC)}
cols = ("앞 dx", "옆 dy", "상대 속도(앞) dvx", "상대 속도(옆) dvy")
print(f"{name}: 상대 차가 6 m 안 앞에 있는 결정 {N}번, 설명력 R² (1 = 다 앎, 0 = 모름)", flush=True)
for k, v in out.items():
    print(f"  {k}: " + " | ".join(f"{c} {x:.2f}" for c, x in zip(cols, v)), flush=True)
print(f"  (참고: 상대 속도(앞) 표준편차 {float(Y[:, 2].std()):.2f} m/s, 옆 {float(Y[:, 3].std()):.2f} m/s)", flush=True)
