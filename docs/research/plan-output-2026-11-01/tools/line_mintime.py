"""같은 라인에서 한계를 다 쓰면 몇 초인가 (11-09, 사용자: "가속 한계까지는 아직 못쓰고있다고 봐여하나?"). 학습 조건(plan대로 정확히),
원래 트랙 빈 트랙에서 policy로 몰아 차마다 3바퀴째 한 바퀴의 라인(위치, 곡률 = 회전율 / 속도)을 잡고, 그 라인 위에서
  * 실제로 걸린 시간
  * 같은 라인, plan 모델의 한계를 다 쓰는 최단 시간: 속도 <= 타이어가 그 곡률을 버티는 속도(plan_steps.Limits.speed_for), <= 10 m/s,
    가속 <= a_acc, 감속 <= a_brake (앞으로·뒤로 한 번씩)
  * 같은 라인, 실차 가속 한계(a_max x min(1, v_switch / v), 11-09 실측 반영 값)로
를 비교. 라인을 바꾸지 않으니 '앞뒤 가속을 덜 써서 잃는 시간'만 남음.
인자: checkpoint 차수"""
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
ck_path, n_env = sys.argv[1], int(sys.argv[2])
secs = 40.0
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD))); a.envs, a.seed, a.device = n_env, 7, "cuda"
a.warps, a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0, 0.0, 0.0, 0.0, secs + 5.0
env = SR.build_env(a, dev)
env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=7)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
nom = env.tracker.nominal
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
tid = env.sim.tid[lid]
Lap = float(pr.goal.lap[tid][0])
X, Y, V, Rr, Dd = [], [], [], [], []
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    x, y, psi = plan_world(p, st)
    env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(torch.zeros(env.B, env.act_dim, device=dev))
        s_ = env.sim.state[lid]
        X.append(s_[:, 0].clone()); Y.append(s_[:, 1].clone()); V.append(s_[:, dyn.IVX].clone()); Rr.append(s_[:, dyn.IR].clone())
        Dd.append(pr.goal.remaining(s_[:, None, :2], tid[:, None])[:, 0])
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"]); obs.set_plan2(p)
X, Y, V, Rr, Dd = (torch.stack(z, 1).cpu().numpy() for z in (X, Y, V, Rr, Dd))
dt = float(env.sim.control_dt)
a_acc, a_brk = float(lim.a_acc), float(lim.a_brake)
a_max, v_sw = float(nom["a_max"]), float(nom["v_switch"])
out = []
for b in range(B):
    d = Dd[b]
    # lap boundaries: the remaining distance jumps up by about a lap at the finish line
    jumps = np.nonzero(np.diff(d) > 0.5 * Lap)[0]
    if len(jumps) < 3:
        continue
    i0, i1 = jumps[1] + 1, jumps[2] + 1                 # the second full lap
    x, y, v, r = X[b, i0:i1 + 1], Y[b, i0:i1 + 1], V[b, i0:i1 + 1], Rr[b, i0:i1 + 1]
    ds = np.hypot(np.diff(x), np.diff(y))
    t_drv = (i1 - i0) * dt
    kap = r / np.maximum(np.abs(v), 0.3)
    vlat = lim.speed_for(torch.tensor(kap, device=dev, dtype=torch.float32)).cpu().numpy()
    vcap = np.minimum(np.minimum(vlat, PS.V_FWD), 1e3)

    def mintime(acc_fn):
        n = len(vcap)
        w = vcap.copy()
        w[0] = min(w[0], v[0])                           # the lap starts at the speed the car crossed the line
        for i in range(n - 1):                           # forward: motor
            w[i + 1] = min(w[i + 1], np.sqrt(w[i] ** 2 + 2 * acc_fn(w[i]) * ds[i]))
        for i in range(n - 2, -1, -1):                   # backward: brakes
            w[i] = min(w[i], np.sqrt(w[i + 1] ** 2 + 2 * a_brk * ds[i]))
        vm = 0.5 * (w[1:] + w[:-1])
        return float(np.sum(ds / np.maximum(vm, 0.05))), w

    t_plan, w_plan = mintime(lambda u: a_acc)
    t_car, w_car = mintime(lambda u: a_max * min(1.0, v_sw / max(u, 1e-3)))
    out.append((t_drv, t_plan, t_car, float(ds.sum()), float(np.max(v)), float(np.max(w_plan)), float(np.max(w_car))))
O = np.array(out)
print(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)} 학습 조건, 원래 트랙 빈 트랙, 차 {len(O)}대의 둘째 바퀴 | "
      f"plan 한계: 가속 {a_acc:.1f}, 감속 {a_brk:.1f}, 최고 {PS.V_FWD:.0f} m/s; 실차 가속 {a_max:.1f} x min(1, {v_sw:.0f}/v)", flush=True)
print(f"  라인 길이 중앙 {np.median(O[:, 3]):.2f} m | 실제 바퀴 시간 중앙 {np.median(O[:, 0]):.2f} s (최고 속도 {np.median(O[:, 4]):.2f} m/s)")
print(f"  같은 라인, plan 모델 한계를 다 쓰면 {np.median(O[:, 1]):.2f} s (최고 {np.median(O[:, 5]):.2f} m/s) -> {np.median(O[:, 0] - O[:, 1]):.2f} s 빠름")
print(f"  같은 라인, 실차 가속 한계로 {np.median(O[:, 2]):.2f} s (최고 {np.median(O[:, 6]):.2f} m/s) -> {np.median(O[:, 0] - O[:, 2]):.2f} s 빠름")
