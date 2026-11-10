"""지그재그 비용(plan_rl --smooth-reg)이 어디서 나오나 (11-10: 무게 0.005와 0.02 모두 비용 1.2-1.3에서 안 움직임 -> 가설: U턴 입구·출구의
큰 정상 전환이 비용을 다 차지하고 작은 떨림은 몫이 작음). 학습 세계(plan_rl.build, 학습 조건, 상자·상대 포함)를 policy로 몰며 결정마다
첫 단계 조향 행동 평균 m = tanh(mean[0])과 속도를 모아, 연속한 결정 사이 변화 |dm|의 크기 구간마다 (횟수 비율, 비용 몫)을 셈 -- 비용 =
학습과 같은 Huber(v^2 ds/L dm / 4). 두 가지: 학습처럼 탐색 잡음을 섞어 몰 때 / 평균으로만 몰 때.
인자: checkpoint [차 수, 기본 128] [초, 기본 30]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path = sys.argv[1]
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 128
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, n_env, 17, dev, fmt="v2")
lid = env.learner_ids
ck = torch.load(ck_path, map_location=dev, weights_only=False)
floor = float(ck["args"].get("speed_std_floor") or 0.0)
ds = lim.sv_max * PS.DT
bins = [0.0, 0.1, 0.3, 0.6, 1.0, 2.01]
for noisy in (True, False):
    env.reset(seed=17)
    actor = load_actor2(ck, dev)
    if floor > 0:
        actor.log_std_floor[PS.N_STEP] = float(np.log(floor))
    pr = PlanReward(env, 0.995, dev)
    obs = PlanObs2(env, lim, pr.goal)
    drv = PR.Driver(env, lim, pr, obs)
    rl = torch.zeros(PS.ACT_DIM, device=dev); rl[0] = 1.0; rl[PS.N_STEP] = 1.0
    M, V, RS = [], [], []
    for kd in range(int(secs / PS.DT)):
        _, mean, log_std, v_hat, p = plan_v2(actor, obs, lim)
        M.append(torch.tanh(mean[:, 0]).cpu()); V.append(env.sim.state[lid, 3].float().cpu())
        if noisy:
            u = mean + log_std.exp() * torch.randn_like(mean) * rl
            p = PS.roll(torch.tanh(u), v_hat.detach().clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
            out, _, _ = drv.decision(torch.tanh(u), p)
        else:
            out, _, _ = drv.decision(torch.tanh(mean), p)
        RS.append(out["restarted"].cpu())
    M, V, RS = torch.stack(M).numpy(), torch.stack(V).numpy(), torch.stack(RS).numpy()
    dm = M[1:] - M[:-1]
    ok = ~RS[:-1]                                                   # no pair across a restart
    j = V[1:] ** 2 * ds / lim.wheelbase * dm / 4.0
    hub = np.where(np.abs(j) < 1.0, 0.5 * j * j, np.abs(j) - 0.5)
    a, h = np.abs(dm)[ok], hub[ok]
    tot = h.sum()
    parts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (a >= lo) & (a < hi)
        parts.append(f"|dm| {lo:g}-{hi:g}: 횟수 {m.mean() * 100:.0f} %, 비용 몫 {h[m].sum() / tot * 100:.0f} %")
    print(f"{'잡음 섞음(학습처럼)' if noisy else '평균만'}: 비용 평균 {h.mean():.3f} | " + " | ".join(parts), flush=True)
