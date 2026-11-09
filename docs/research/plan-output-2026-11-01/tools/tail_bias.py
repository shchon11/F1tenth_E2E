"""plan 뒷부분이 실제 갈 길이 되어 가나 (11-09, plan25m의 학습 때 붙잡기 효과 확인). 원래 트랙, 빈 트랙, 학습 조건(plan대로 정확히,
0.1 s마다 새 plan -- 평가와 같음)으로 차 N대를 몰며, 결정마다 plan이 말한 0.1..1.0 s 뒤의 속도·자리를 그 시각에 차가 실제로 있던 속도·자리와
비교. 붙잡기가 효과가 있으면 뒷부분의 속도 차이(지금까지 0.5 s에서 plan이 약 1 m/s 느림)와 자리 차이가 줄어야 함.
인자: checkpoint [차수] [초]   (여러 checkpoint를 차례로: 쉼표로)"""
import os, sys, argparse, json
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import PlanReward
from f1sim.learn.plan_policy import plan_world, PT_IDX, PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M

torch.set_grad_enabled(False)
cks = sys.argv[1].split(",")
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 64
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD))); a.envs, a.seed, a.device = n_env, 7, "cuda"
a.warps, a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0, 0.0, 0.0, 0.0, secs + 5.0
env = SR.build_env(a, dev)
env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
for ck_path in cks:
    env.reset(seed=7)
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    actor = load_actor2(ck, dev)
    pr = PlanReward(env, 0.995, dev)
    obs = PlanObs2(env, lim, pr.goal)
    PV, PP, VV, XY = [], [], [], []
    n = int(secs / PS.DT)
    for kd in range(n):
        st = env.sim.state[lid].clone()
        _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
        x, y, psi = plan_world(p, st)
        idx = PT_IDX.to(dev)
        PV.append(p["pv"].cpu()); PP.append(torch.stack([x[:, idx], y[:, idx]], -1).cpu())
        env.sim.ideal_vcap[lid] = 1e3
        for j in range(PS.SUB):
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
            env.step(torch.zeros(env.B, env.act_dim, device=dev))
        VV.append(env.sim.state[lid, dyn.IVX].cpu()); XY.append(env.sim.state[lid, :2].cpu())
        obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"]); obs.set_plan2(p)
    PV, PP, VV, XY = torch.stack(PV, 0), torch.stack(PP, 0), torch.stack(VV, 0), torch.stack(XY, 0)   # (T, B, 10[, 2])
    skip = 30
    res = []
    for k in (1, 3, 5, 10):
        t0 = torch.arange(skip, n - k)
        dv = PV[t0, :, k - 1] - VV[t0 + k - 1]                 # plan's speed at k*0.1 s - the car's then
        dp = (PP[t0, :, k - 1] - XY[t0 + k - 1]).norm(dim=-1)
        res.append((k, float(dv.median()), float(dv.abs().median()), float(dp.median())))
    name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
    print(f"{name}: plan - 실제 (0.1 s마다 새 plan) | " + " | ".join(
        f"{k * 0.1:.1f} s: 속도 {m:+.2f} (|.| {ma:.2f}) m/s, 자리 {d * 100:.1f} cm" for k, m, ma, d in res), flush=True)
