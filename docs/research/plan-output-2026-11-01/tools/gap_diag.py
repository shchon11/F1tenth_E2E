"""학습 조건과 실차 조건의 바퀴 시간 차이가 어디서 나나 (11-08 00:45): 실차 조건(물리 + MAP)으로 학습된 planner를 몰며, 결정마다
plan이 0.1 s 뒤 말한 속도·꺾기(곡률)와 0.1 s 뒤 실제 속도·꺾기를 비교. 속도를 못 따라가나(가속·감속), 꺾기를 못 따라가나(핸들·타이어),
경로 옆으로 벌어지나. 인자: checkpoint 차수 초"""
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
ck_path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = n_env, 5, "cuda"
a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0.0, 0.0, 0.0, 3600.0
a.plan_format, a.repeat = "tw", 4
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=5)
lid = env.learner_ids
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor(ck, dev); obs = PlanObs(env, lim, None)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
rows = []
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    mean, _ = actor(obs.scans, obs.state())
    p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
    x, y, psi = plan_world(p, st)
    k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    for j in range(PS.SUB):
        env.step(act0)
    s1 = env.sim.state[lid]
    v_plan = p["v"][:, PS.SUB]; v_act = s1[:, dyn.IVX]
    kap_plan = torch.tan(p["delta"][:, PS.SUB]) / lim.wheelbase
    kap_act = s1[:, dyn.IR] / s1[:, dyn.IVX].abs().clamp_min(0.3)
    dx = s1[:, 0] - x[:, PS.SUB]; dy = s1[:, 1] - y[:, PS.SUB]
    side = (-dx * torch.sin(psi[:, PS.SUB]) + dy * torch.cos(psi[:, PS.SUB]))
    along = (dx * torch.cos(psi[:, PS.SUB]) + dy * torch.sin(psi[:, PS.SUB]))
    acc_plan = (p["v"][:, PS.SUB] - st[:, dyn.IVX]) / PS.DT
    rows.append(torch.stack([v_plan, v_act, kap_plan, kap_act, side, along, acc_plan, st[:, dyn.IVX]], 1))
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
R = torch.cat(rows)
v_plan, v_act, kp, ka, side, along, accp, v0 = R.T
print(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)} 실차 조건, 결정 {R.shape[0]}개", flush=True)
for name, m in (("가속하라는 결정(plan 가속 > 2 m/s^2)", accp > 2), ("유지(|가속| < 1)", accp.abs() < 1), ("감속하라는 결정(< -2)", accp < -2)):
    if m.any():
        print(f"  {name} {int(m.sum())}개: 0.1 s 뒤 plan 속도 - 실제 속도 중앙 {float((v_plan - v_act)[m].median()):+.3f} m/s, 진행 방향 위치 차이 중앙 {float(along[m].median()) * 100:+.1f} cm", flush=True)
for name, m in (("직진(|plan 곡률| < 0.1)", kp.abs() < 0.1), ("꺾기(0.1~0.4)", (kp.abs() >= 0.1) & (kp.abs() < 0.4)), ("급하게(> 0.4)", kp.abs() >= 0.4)):
    if m.any():
        ratio = (ka[m] / kp[m]).median()
        print(f"  {name} {int(m.sum())}개: 실제 곡률 / plan 곡률 중앙 {float(ratio):.2f}, 옆 위치 차이 |중앙| {float(side[m].abs().median()) * 100:.1f} cm (바깥쪽이면 +: {float((side[m] * -torch.sign(kp[m])).median()) * 100:+.1f} cm)", flush=True)
print(f"  plan 속도 분포: 0.1 s 뒤 plan 속도 중앙 {float(v_plan.median()):.2f}, 실제 {float(v_act.median()):.2f} m/s", flush=True)
