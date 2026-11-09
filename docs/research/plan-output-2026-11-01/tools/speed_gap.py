"""어디서 느린가 (11-09, 사용자: "근데 느리긴 하다"): 원래 트랙(변형 없음), 빈 트랙, 상대 없음에서 차 N대를 몰아(조건: ideal = 학습
조건 / track = 물리 + 새 제어기) 0.025 s마다 속도, 옆 가속(속도 x 회전율), 진행 위치(결승선까지 벽 사이 최단 거리)를 기록하고, 바퀴를 0.5 m
칸으로 나눠 칸마다 머문 시간(한 바퀴 기준), 중앙 속도, 옆 가속, 그 속도에서 타이어가 버티는 옆 가속(MAP 조향표, 기준 마찰 = plan이 쓰는 한계)
대비 비율을 저장. 두 조건 비교는 speed_gap_cmp.py.
인자: checkpoint 조건(ideal|track) 차수 초 출력json"""
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
ck_path, mode, n_env, secs, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD))); a.envs, a.seed, a.device = n_env, 7, "cuda"
a.warps, a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0, 0.0, 0.0, 0.0, secs + 5.0
env = SR.build_env(a, dev)
if mode == "ideal":
    env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=7)
lid = env.learner_ids; B = int(lid.numel())
follower = None
if mode == "track":
    from f1sim.plan_follow import PlanFollower, to_car
    follower = PlanFollower(B, dev, s_max=float(env.s_max), lut=env.tracker.map.lut,
                            grip=M.GRIP_LEVELS.index(1.0), wheelbase=float(env.tracker.nominal["lf"]) + float(env.tracker.nominal["lr"]))
    env.ext_mask[lid] = True; env._ext_ids.update(int(i) for i in lid.tolist())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
tid = env.sim.tid[lid]
Lap = float(pr.goal.lap[tid][0])
Dd, V, R = [], [], []
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    x, y, psi = plan_world(p, st)
    if follower is not None:
        follower.set_plan(slice(None), x, y, psi, p["v"], p["delta"])
    env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        if follower is not None:
            env.ext_cmd[lid] = to_car(env, lid, follower.command(env.sim.state[lid]))
        env.step(act0)
        s_ = env.sim.state[lid]
        Dd.append(pr.goal.remaining(s_[:, None, :2], tid[:, None])[:, 0])
        V.append(s_[:, dyn.IVX].clone()); R.append(s_[:, dyn.IR].clone())
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"]); obs.set_plan2(p)
Dd = torch.stack(Dd, 1).cpu().numpy(); V = torch.stack(V, 1).cpu().numpy(); R = torch.stack(R, 1).cpu().numpy()
dt = float(env.sim.control_dt)
skip = int(3.0 / dt)                                        # the standing start aside
Dd, V, R = Dd[:, skip:], V[:, skip:], R[:, skip:]
s = Lap - Dd                                                # progress along the lap [m]
ok = np.isfinite(s)
# laps driven per car: the unwrapped progress
laps = []
for b in range(V.shape[0]):
    sb = s[b][ok[b]]
    jump = np.diff(sb)
    jump = np.where(jump < -0.5 * Lap, jump + Lap, np.where(jump > 0.5 * Lap, jump - Lap, jump))
    laps.append(max(1e-6, jump.sum() / Lap))
laps = np.array(laps)
nb = int(np.ceil(Lap / 0.5))
edges = np.linspace(0, Lap, nb + 1)
bi = np.clip(np.digitize(np.where(ok, s, 0.0), edges) - 1, 0, nb - 1)
alat = np.abs(V * R)
vg = np.abs(V).clip(max=float(lim.v_grid[-1]))
alim = (lim.kappa_at(torch.from_numpy(vg).float().to(dev)).cpu().numpy()) * vg * vg
rows = []
for k in range(nb):
    m = ok & (bi == k)
    t_bin = float(m.sum() * dt / laps.sum())                # all cars' time in the bin / all their laps
    rows.append({"s0": float(edges[k]), "t": t_bin, "v": float(np.median(V[m])) if m.any() else float("nan"),
                 "alat": float(np.median(alat[m])) if m.any() else float("nan"),
                 "alim": float(np.median(alim[m])) if m.any() else float("nan")})
lap_t = sum(r_["t"] for r_ in rows)
json.dump({"ck": ck_path, "mode": mode, "lap": Lap, "lap_time": lap_t, "laps_per_car": float(laps.mean()),
           "laps": [float(x) for x in laps], "secs": float(V.shape[1] * dt), "bins": rows},
          open(out, "w"), indent=0)
vmax = float(np.nanmax([r_["v"] for r_ in rows]))
util = np.array([r_["alat"] / r_["alim"] for r_ in rows if r_["alim"] > 0])
print(f"[{mode}] {os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}: 바퀴 {Lap:.1f} m, 칸 시간 합 {lap_t:.2f} s, "
      f"차당 {laps.mean():.1f}바퀴(차별 바퀴 시간 중앙 {np.median(V.shape[1] * dt / laps):.2f} s, 가장 느린 차 {np.max(V.shape[1] * dt / laps):.2f} s) | 칸 중앙 속도 최고 {vmax:.2f} m/s (plan 상한 {PS.V_FWD:.0f}) | 옆 가속 / 타이어 한계: 칸 중앙 "
      f"{np.median(util) * 100:.0f} %, 상위 10 % {np.quantile(util, 0.9) * 100:.0f} %", flush=True)
