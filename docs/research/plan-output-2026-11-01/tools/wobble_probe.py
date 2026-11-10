"""차가 흔들리며 가나 (11-10, 사용자: "학습 환경하고 제어기 물린거하고 콘솔로 직접 보는거하고 갈수록 차가 존나 불안하게 흔들리면서 가냐").
학습 조건(plan대로 정확히 실행), 원래 트랙 빈 트랙(상자·상대 없음), 차 N대 x S초를 checkpoint마다 같은 시드로 몰며, 결정마다 실제로 내린
조향 명령(plan의 첫 0.1 s)과 요 각속도를 모음:
  조향 흔들림 = 결정 사이 조향 명령 변화의 평균 크기 [deg]
  지그재그 비율 = 조향이 꺾이는 방향이 바로 뒤집힌 결정의 비율(이번 변화와 바로 앞 변화의 부호가 반대, 둘 다 0.5 deg 넘게)
  요 각속도 떨림 = 요 각속도에서 1 s 평균을 뺀 나머지의 표준편차 [deg/s]
그림: checkpoint마다 차 한 대의 조향 명령 10 s(같은 출발).
인자: 출력png checkpoint1,checkpoint2,... [차 수, 기본 64] [초, 기본 30]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

torch.set_grad_enabled(False)
out, cks = sys.argv[1], sys.argv[2].split(",")
n_env = int(sys.argv[3]) if len(sys.argv) > 3 else 64
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, n_env, 5, dev, overrides={"prop_fraction": 0.0, "opp_present": 0.0, "warps": 0,
                                                             "removal_p": 0.0}, fmt="v2")
lid = env.learner_ids
traces = {}
for ck_path in cks:
    env.reset(seed=5)
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    actor = load_actor2(ck, dev)
    pr = PlanReward(env, 0.995, dev)
    obs = PlanObs2(env, lim, pr.goal)
    drv = PR.Driver(env, lim, pr, obs)
    D, W = [], []
    for kd in range(int(secs / PS.DT)):
        _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
        D.append(p["cmd_d"][:, 0].float().cpu())
        drv.decision(torch.tanh(mean), p)
        W.append(env.sim.state[lid, dyn.IR].float().cpu())
    D = np.degrees(torch.stack(D).numpy()); W = np.degrees(torch.stack(W).numpy())     # (T, B)
    D, W = D[20:], W[20:]                                          # the start's first 2 s left out
    dd = np.diff(D, axis=0)
    jit = float(np.mean(np.abs(dd)))
    zz = float(np.mean((dd[1:] * dd[:-1] < 0) & (np.abs(dd[1:]) > 0.5) & (np.abs(dd[:-1]) > 0.5)))
    k = np.ones(10) / 10
    Ws = np.stack([np.convolve(W[:, b], k, "same") for b in range(W.shape[1])], 1)
    wob = float(np.std((W - Ws)[5:-5]))
    name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path).replace('plan_', '').replace('.pt', '')}"
    print(f"{name}: 조향 흔들림 {jit:.2f} deg/결정 | 지그재그 {zz * 100:.0f} % | 요 각속도 떨림 {wob:.1f} deg/s", flush=True)
    traces[name] = D[:100, 0]
fig, ax = plt.subplots(figsize=(14, 5))
for i, (n, tr) in enumerate(traces.items()):
    ax.plot(np.arange(len(tr)) * PS.DT + 2.0, tr + 0 * i, lw=1.3, label=n)
ax.set_xlabel("시간 [s]"); ax.set_ylabel("조향 명령 [deg]"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title("학습 조건, 빈 트랙, 같은 출발의 차 한 대 -- checkpoint마다 조향 명령 (톱니처럼 오르내리면 흔들림)", fontsize=11)
plt.tight_layout(); plt.savefig(out, dpi=80)
print(out, flush=True)
