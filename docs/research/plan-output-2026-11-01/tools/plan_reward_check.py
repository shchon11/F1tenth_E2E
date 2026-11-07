"""재설계 3단계 확인 (11-07): 보상(plan_reward.py)이 약속대로인가. 지금 있는 policy(MAP으로 달림)로 차 N대를 빈 트랙에서 달리며 0.1 s마다
보상을 계산: gamma 1이면 중간 점수 + 한 바퀴 점수의 합이 (바퀴 수 + 끝 Phi - 처음 Phi)와 정확히 같아야 하고(정산), 남은 거리가 결승선 말고는
튀지 않아야 함(0.1 s 사이 변화가 그 사이 달린 거리보다 크게 줄면 튐). 인자: 체크포인트 차수 초"""
import os, sys, math, argparse, time
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import PlanReward

torch.set_grad_enabled(False)
path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, 81, "cuda"
a.prop_fraction, a.removal_p, a.opp_present, a.spawn_behind_prop, a.stage, a.episode_s = 0.0, 0.0, 0.0, 0.0, "props", 3600.0
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=81)
t0 = time.time()
pr = PlanReward(env, 1.0, dev)
print(f"남은 거리 지도: 트랙 {pr.goal.lap.numel()}개, {time.time() - t0:.1f} s, 한 바퀴 최단 거리 {[round(float(x), 1) for x in pr.goal.lap[:4]]} ...", flush=True)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel()); R = int(a.repeat)
last = torch.zeros(B, tp.A, device=dev); act = torch.zeros(env.B, env.act_dim, device=dev)
phi0 = pr.phi.clone()
tot = torch.zeros(B, device=dev); laps = torch.zeros(B, device=dev); jumps = 0; nstep = 0
dpos = torch.zeros(B, device=dev); pos_prev = env.sim.state[lid, :2].clone()
for k in range(int(secs / env.sim.control_dt)):
    if k % R == 0:
        last = tp(last); act[lid] = last
    env.step(act)
    pr.control_step()
    dpos += (env.sim.state[lid, :2] - pos_prev).norm(dim=-1); pos_prev = env.sim.state[lid, :2].clone()
    if (k + 1) % 4 == 0:
        dprev = pr.d_prev.clone()
        out = pr.decision()
        tot += out["lap"] + out["shaping"]; laps += out["lap"]
        # a jump: the distance left fell by more than the car drove (the line's own jump aside)
        fell = dprev - pr.d_prev
        jumps += int(((fell > dpos + 0.15) & (out["lap"] == 0)).sum()); nstep += B
        dpos.zero_()
err = (tot - (laps + pr.phi - phi0)).abs()
print(f"차 {B}대 x {secs:.0f} s: 바퀴 {float(laps.sum()):.0f}개 | 점수 합 - (바퀴 + 끝 Phi - 처음 Phi) 최대 {float(err.max()):.2e} | "
      f"결승선 말고 남은 거리가 달린 거리보다 15 cm 넘게 준 결정 {jumps}/{nstep}", flush=True)
