"""박은 뒤 왜 못 빠져나오나 (11-08 13:20): u600 배치 6에서 실차 조건(물리 + MAP)만 상자에 박고 36 s 멈춤(학습 조건은 0번 닿음, u500
배치 1은 6 s 뒤 후진해 빠져나옴). 상자 배치 세계(상대 없음, 고정 시드)에서 차 N대를 몰며 결정마다 실제 속도와 plan의 첫 0.5 s 속도를
기록하고, 2 s 넘게 |속도| < 0.3 m/s인 구간(멈춤, 출발 3 s 제외) 동안 plan이 무엇을 말했나 셈: 앞으로(> +0.3 m/s), 뒤로(< -0.3 m/s),
제자리. plan이 뒤로 가자는데 차가 안 움직이면 추종(MAP) 쪽 문제, plan이 계속 앞(상자 쪽)으로 가자면 policy 쪽 문제.
인자: checkpoint 조건(ideal|map) 차수 초"""
import os, sys, argparse, json
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_policy import PlanObs, load_actor, plan_world, plan_segments, PT_IDX, PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M

torch.set_grad_enabled(False)
ck_path, mode, n_env, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
SEED = 4242
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = n_env, SEED, "cuda"
a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 1.0, 0.0, 0.0, secs + 5.0
if mode == "map":
    a.plan_format, a.repeat = "tw", 4
torch.manual_seed(SEED)
env = SR.build_env(a, dev)
if mode == "ideal":
    env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=SEED)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
V2 = ck.get("format") == "plan_steps2"
actor = (load_actor2 if V2 else load_actor)(ck, dev)
obs = (PlanObs2 if V2 else PlanObs)(env, lim, None)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
VX, PV, S0, VH = [], [], [], []
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    if V2:
        _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    else:
        mean, _ = actor(obs.scans, obs.state())
        p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
    VX.append(st[:, dyn.IVX]); PV.append(p["pv"][:, :5].mean(1)); VH.append(v_hat if V2 else st[:, dyn.IVX])
    x, y, psi = plan_world(p, st)
    k_, v_ = plan_segments(p, lim); S0.append(v_[:, 0])
    if mode == "map":
        kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(act0)
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
    if V2:
        obs.set_plan2(p)
VX = torch.stack(VX, 1).cpu(); PV = torch.stack(PV, 1).cpu()     # (B, K)
S0 = torch.stack(S0, 1).cpu(); VH = torch.stack(VH, 1).cpu()     # the first 0.1 s segment's speed (what MAP commands), the speed estimate
skip = int(3.0 / PS.DT); need = int(2.0 / PS.DT)
runs, fw, bw, zr, n_dec, longest = 0, 0, 0, 0, 0, 0
seg_b, seg_f, vh_all = 0, 0, []
for b in range(B):
    slow = (VX[b].abs() < 0.3).tolist(); lo = None
    for t, s in enumerate(slow + [False]):
        if s and lo is None and t >= skip:
            lo = t
        elif not s and lo is not None:
            if t - lo >= need:
                pv = PV[b, lo:t]
                runs += 1; n_dec += t - lo; longest = max(longest, t - lo)
                s0 = S0[b, lo:t]; seg_b += int((s0 < -0.05).sum()); seg_f += int((s0 > 0.05).sum()); vh_all.append(VH[b, lo:t])
                fw += int((pv > 0.3).sum()); bw += int((pv < -0.3).sum()); zr += int(((pv >= -0.3) & (pv <= 0.3)).sum())
            lo = None
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
tot = max(n_dec, 1)
print(f"[{mode}] {name}: 상자 배치 차 {B}대 x {secs:.0f} s | 2 s 넘는 멈춤 {runs}번, 합 {n_dec * PS.DT:.0f} s (가장 긴 것 {longest * PS.DT:.0f} s) | "
      f"멈춘 동안 plan 첫 0.5 s 속도: 앞으로 {fw / tot * 100:.0f} %, 제자리 {zr / tot * 100:.0f} %, 뒤로 {bw / tot * 100:.0f} %", flush=True)
vh = torch.cat(vh_all) if vh_all else torch.tensor([float("nan")])
print(f"    멈춘 동안 MAP에 가는 첫 0.1 s 구간 속도: 뒤로 {seg_b / tot * 100:.0f} %, 앞으로 {seg_f / tot * 100:.0f} %, 0 {100 - (seg_b + seg_f) / tot * 100:.0f} % | "
      f"policy의 속도 짐작(실제 |속도| < 0.3): 중앙 {float(vh.median()):+.2f} m/s, +0.3 넘음 {float((vh > 0.3).float().mean()) * 100:.0f} %, "
      f"-0.3 아래 {float((vh < -0.3).float().mean()) * 100:.0f} %", flush=True)
