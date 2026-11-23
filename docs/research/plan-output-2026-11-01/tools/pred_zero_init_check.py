"""(11-24, 경주 단계) 상대 차 예측 머리(plan_rl --pred-opp)를 붙인 fork가 update 0에서 부모와 똑같이 달리나.

fork를 만드는 길(plan_rl.init_actor: trunk 첫 층에 예측 입력 열을 0으로 붙이고, 예측 머리의 스캔 encoder는 policy 것을 복사) 그대로
부모 checkpoint에서 fork actor를 만들고, 경주 세계(상대 차 = 복사본, 배율 여러 가지 -> 예측 값이 실제로 움직이는 상황)에서 부모가 모는
대로 달리며 결정마다 같은 입력(스캔 3장, 상태, 기억)을 두 actor에 넣어 출력(plan 평균 20개, 표준편차, 속도 짐작, 새 기억 상태)을 비교.
인자: checkpoint [경주 수 8] [결정 수 60]
"""
import os, sys, json
os.environ["F1SIM_NO_STAGGER"] = "1"
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
torch.set_grad_enabled(False)

ck_path = sys.argv[1]
G = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
dev = torch.device("cpu")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of, load_actor2, Actor2, PlanObs2, plan_world, PT_IDX
os.environ["F1SIM_LIDAR_MOUNT_X"] = str(lidar_mount_of(ck))
os.environ["F1SIM_PLAN_LAT_MARGIN"] = str((ck.get("args") or {}).get("plan_lat_margin", 0.95))
from f1sim.learn import plan_rl as PR
from f1sim import plan_steps as PS

base = load_actor2(ck, dev)
ca = ck["args"]
fork = Actor2(int(ck["beams"]), std0=float(ca.get("std0", 0.6)), memory=int(ca.get("memory", 0)), mem_front=bool(ca.get("mem_front")),
              mem_value="v_mem.weight" in ck["actor"], pred_opp=True).to(dev)
added = PR.init_actor(fork, ck["actor"])
fork.eval()
assert added and fork.pred is not None
W = os.path.join(os.path.dirname(PR.WORLD), "plan_world_korea26scan_race.json")
env, lim, a = PR.build(W, 2 * G, 3, dev, overrides={"opp_present": 1.0, "align_episodes": True})
lid = env.learner_ids
env.ep_limit[:] = 10 ** 6
obs = PlanObs2(env, lim, None)
so = PR.SelfOpponents(env, lim, 1.0, seed=4, max_pool=1, p_newest=1.0, pace=(0.6, 0.95, 0.2))
so.add(base, int(ck.get("update") or 0))
act0 = torch.zeros(env.B, env.act_dim, device=dev)
hb = torch.zeros(lid.numel(), base.memory, device=dev); hf = hb.clone()
dm = dls = dv = dh = 0.0
pv_rng = []
for t in range(N):
    sc, s = obs.scans.clone(), obs.state()
    mb, lsb, vb = base(sc, s, hb); hb1 = base.h_out
    mf, lsf, vf = fork(sc, s, hf); hf1 = fork.h_out
    dm = max(dm, float((mb - mf).abs().max())); dls = max(dls, float((lsb - lsf).abs().max()))
    dv = max(dv, float((vb - vf).abs().max())); dh = max(dh, float((hb1 - hf1).abs().max()))
    pv_rng.append(torch.sigmoid(fork.pred_out[:, -1]))
    hb, hf = hb1, hf1
    p = PS.roll(torch.tanh(mb), vb.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
    st = env.sim.state[lid].clone()
    x, y, _ = plan_world(p, st)
    so.plan()
    env.sim.ideal_vcap[lid] = 1e3
    rs = torch.zeros(lid.numel(), dtype=torch.bool)
    for j in range(PS.SUB):
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        so.command(j)
        _, _, te, trc, _ = env.step(act0)
        so.stepped(te | trc); rs |= (te | trc)[lid]
    so.after()
    ii = PT_IDX
    obs.push_scan(); obs.set_plan(torch.stack([x[:, ii], y[:, ii]], -1), p["pv"]); obs.set_plan2(p)
    if bool(rs.any()):
        r_ = torch.nonzero(rs)[:, 0]; obs.reset_rows(r_); hb[r_] = 0.0; hf[r_] = 0.0
pv = torch.stack(pv_rng)
print(f"[예측 머리 0 초기화 확인] {os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)} -> fork (예측 머리 붙임), "
      f"경주 {G}개 x {N} 결정, 상대 차 = 복사본(배율 0.6-0.95 / 제 페이스)")
print(f"  두 actor의 최대 차이: plan 평균 {dm:.3g} | 표준편차 {dls:.3g} | 속도 짐작 {dv:.3g} m/s | 새 기억 상태 {dh:.3g}")
print(f"  그동안 예측 머리의 '보임' 확률: 최소 {float(pv.min()):.3f} 최대 {float(pv.max()):.3f} (움직였는데도 plan은 그대로)")
print(json.dumps({"mean": dm, "log_std": dls, "v_hat": dv, "mem": dh, "pvis_min": float(pv.min()), "pvis_max": float(pv.max())}))
