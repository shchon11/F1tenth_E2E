"""plan과 실제 위치 차이 쪼개기 (11-08 01:40, 사용자: "학습조건에서 plan과 실제 차이가 더 크다고?"): plan_validate와 같은 세계·시드로
차 N대를 몰며, 결정 k의 plan이 0.1·0.3·0.5 s 뒤 있겠다던 곳과 그 시각 실제 위치의 차이를 진행 방향(앞서면 +)과 옆으로 나눔. 그리고 plan이
0.1·0.3·0.5 s 뒤 낼 거라던 속도와 그 시각 실제 속도, 결정 k+1의 plan이 같은 시각에 대해 말하는 속도(다음 plan이 말을 바꾸나).
인자: checkpoint 조건(ideal|map) 차수 초"""
import os, sys, math, argparse, json
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_policy import PlanObs, load_actor, plan_world, plan_segments, PT_IDX
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M

torch.set_grad_enabled(False)
ck_path, mode, n_env, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD))); a.envs, a.seed, a.device = n_env, 4242, "cuda"
a.episode_s = secs + 5.0
if mode == "map":
    a.plan_format, a.repeat = "tw", 4
torch.manual_seed(4242)
env = SR.build_env(a, dev)
if mode == "ideal":
    env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=4242)
lid = env.learner_ids
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False); actor = load_actor(ck, dev); obs = PlanObs(env, lim, None)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
P, PSI, PV, X, V = [], [], [], [], []
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    mean, _ = actor(obs.scans, obs.state())
    p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
    x, y, psi = plan_world(p, st)
    P.append(torch.stack([x[:, ::PS.SUB], y[:, ::PS.SUB]], -1)); PSI.append(psi[:, ::PS.SUB]); PV.append(p["v"][:, ::PS.SUB])
    X.append(st[:, :3].clone()); V.append(st[:, dyn.IVX].clone())
    if mode == "map":
        k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(act0)
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
P = torch.stack(P); PSI = torch.stack(PSI); PV = torch.stack(PV); X = torch.stack(X); V = torch.stack(V)
print(f"[{mode}] {os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}: 차 {int(lid.numel())}대 x {secs:.0f} s, 평균 속도 {float(V.abs().mean()):.2f} m/s", flush=True)
for h in (0.1, 0.3, 0.5):
    k = int(round(h / PS.DT))
    d = X[k:, :, :2] - P[:-k, :, k]
    ps = PSI[:-k, :, k]
    along = d[..., 0] * torch.cos(ps) + d[..., 1] * torch.sin(ps)          # + = the car got further than the plan said
    side = -d[..., 0] * torch.sin(ps) + d[..., 1] * torch.cos(ps)
    dv = V[k:] - PV[:-k, :, k]                                            # actual - planned speed at that time
    # what the next plan says about the same time (h - 0.1 s on it) against what this plan said
    dv_next = PV[1:-k + 1 if k > 1 else None, :, k - 1] - PV[:-k, :, k] if k > 1 else None
    s = f"  {h:.1f} s 뒤: 진행 방향 차이 중앙 {float(along.median()) * 100:+.1f} cm (실제가 앞서면 +), 옆 |중앙| {float(side.abs().median()) * 100:.1f} cm | 실제 속도 - plan 속도 중앙 {float(dv.median()):+.2f} m/s"
    if dv_next is not None:
        s += f" | 다음 plan이 같은 시각에 말하는 속도 - 이 plan 중앙 {float(dv_next.median()):+.2f} m/s"
    print(s, flush=True)
# the plan's own speed profile: how much it accelerates over its second
print(f"  plan 속도: 0.1 s {float(PV[:, :, 1].mean()):.2f}, 0.5 s {float(PV[:, :, 5].mean()):.2f}, 1.0 s {float(PV[:, :, 10].mean()):.2f} m/s (지금 {float(V.mean()):.2f})", flush=True)
