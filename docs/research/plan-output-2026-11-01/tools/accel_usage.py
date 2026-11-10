"""가속·감속 한계를 쓰고 있나 (11-09, 사용자: "가속 한계까지는 아직 못쓰고있다고 봐여하나?"). 원래 트랙, 빈 트랙, 상대 없음에서 학습
조건(plan대로 정확히)으로 차 N대를 몰며 결정(0.1 s)마다 plan 첫 단계의 속도를 봄. plan의 속도 범위는 [v - a_brake 0.1, v + a_acc 0.1]
(모터·브레이크, plan_steps.step_targets)이고 그 위를 지금 고른 회전이 타이어로 버티는 속도(vcap)가 다시 막음. 결정마다:
  * 가속 쪽: 0.1 s 뒤 속도 - 지금 속도 / (a_acc 0.1)  -> 모터 한계 대비 몇 %를 썼나, 막은 것이 모터인지 회전(vcap)인지
  * 감속 쪽: 같은 식으로 a_brake 대비
  * 바퀴 위치(결승선까지 벽 사이 최단 거리)별로 모아 어디서 덜 쓰는지
인자: checkpoint 차수 초"""
import os, sys, argparse, json
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
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
ck_path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD))); a.envs, a.seed, a.device = n_env, 7, "cuda"
a.warps, a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0, 0.0, 0.0, 0.0, secs + 5.0
env = SR.build_env(a, dev)
env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=7)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
tid = env.sim.tid[lid]
Lap = float(pr.goal.lap[tid][0])
rows = []
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    act, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    x, y, psi = plan_world(p, st)
    # the first step's speed range, as step_targets builds it (from the policy's own speed estimate and last steering)
    v0 = v_hat.clamp(-PS.V_REV, PS.V_FWD)
    v_lo = (v0 - lim.a_brake * PS.DT).clamp(min=-PS.V_REV); v_hi = (v0 + lim.a_acc * PS.DT).clamp(max=PS.V_FWD)
    d1 = p["delta"][:, PS.SUB]
    vcap = lim.speed_for(PS.steer_to_kappa(d1, lim))
    v1 = p["v"][:, PS.SUB]
    a_sp = torch.tanh(mean)[:, PS.N_STEP] if mean.shape[1] > PS.N_STEP else torch.full_like(v0, float("nan"))
    env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(torch.zeros(env.B, env.act_dim, device=dev))
    d_rem = pr.goal.remaining(env.sim.state[lid][:, None, :2], tid[:, None])[:, 0]
    rows.append(torch.stack([st[:, dyn.IVX], v0, v1, v_lo, v_hi, vcap, a_sp, d_rem, st[:, dyn.ISTEER].abs()], 1).cpu().numpy())
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"]); obs.set_plan2(p)
R = np.stack(rows, 1)[:, 30:]                     # (B, T, 9), the first 3 s aside
v, v0, v1, vlo, vhi, vcap, asp, drem, steer = [R[..., i].reshape(-1) for i in range(9)]
s = Lap - drem
dv = v1 - v0
acc = dv > 0.02; brk = dv < -0.02
use_a = dv / (lim.a_acc * PS.DT); use_b = -dv / (lim.a_brake * PS.DT)
capped = vcap < vhi - 1e-3                         # the turn chosen holds the speed below what the motor could reach
print(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)} 학습 조건, 원래 트랙 빈 트랙 차 {B}대 x {secs:.0f} s | "
      f"plan 한계: 가속 {lim.a_acc:.1f}, 감속 {lim.a_brake:.1f} m/s^2, 최고 {PS.V_FWD:.0f} m/s", flush=True)
print(f"결정 {len(v)}개: 속도를 올림 {acc.mean() * 100:.0f} %, 내림 {brk.mean() * 100:.0f} %, 유지 {(~acc & ~brk).mean() * 100:.0f} %")
print(f"  올릴 때 모터 한계 대비 사용: 중앙 {np.median(use_a[acc]) * 100:.0f} %, 90 % 넘게 쓴 결정 {(use_a[acc] > 0.9).mean() * 100:.0f} % "
      f"| 그중 회전(타이어)이 속도를 막은 결정 {capped[acc].mean() * 100:.0f} %")
m = acc & ~capped
print(f"  회전이 막지 않았는데(모터만 한계) 올릴 때: 사용 중앙 {np.median(use_a[m]) * 100:.0f} %, 90 % 넘게 {(use_a[m] > 0.9).mean() * 100:.0f} %, "
      f"속도 출력(-1..1) 중앙 {np.median(asp[m]):+.2f}")
print(f"  내릴 때 브레이크 한계 대비: 중앙 {np.median(use_b[brk]) * 100:.0f} %, 90 % 넘게 {(use_b[brk] > 0.9).mean() * 100:.0f} %")
print("바퀴 위치별 (2 m 칸): 속도 중앙 | 올리는 결정 비율, 그때 모터 한계 사용 중앙, 회전이 막은 비율 | 내리는 결정 비율, 브레이크 사용 중앙 | 조향 중앙")
for lo in np.arange(0, Lap, 2.0):
    k = (s >= lo) & (s < lo + 2.0)
    if k.sum() < 50:
        continue
    ka, kb = k & acc, k & brk
    print(f"  {lo:5.1f}-{lo + 2:5.1f} m: {np.median(v[k]):4.1f} m/s | 올림 {ka.mean() / k.mean() * 100:3.0f} %, "
          f"사용 {np.median(use_a[ka]) * 100 if ka.any() else float('nan'):3.0f} %, 회전이 막음 {capped[ka].mean() * 100 if ka.any() else float('nan'):3.0f} % | "
          f"내림 {kb.mean() / k.mean() * 100:3.0f} %, 사용 {np.median(use_b[kb]) * 100 if kb.any() else float('nan'):3.0f} % | 조향 {np.median(steer[k]):.2f}")
