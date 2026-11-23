"""(11-24, 경주 단계) 느리게 한 복사본 상대 차가 정말 그 배율의 바퀴 시간으로 도나 -- plan_rl --self-opp-pace 의 측정.

사용자와 정한 설계: 복사본이 모는 경주의 ~80 %는 우리 페이스의 0.6-0.95배, ~20 %는 제 페이스. 방식(SelfOpponents의 PACE_KEYS 설명):
배율 f인 복사본은 '제 페이스라면 지금 속도'(참 속도 / f)에서 plan을 그리고 plan의 속도를 f배로 달림 -> 같은 길을 f배 빠르기로.
여기서는 그 방식 그대로(SelfOpponents 자체)로 복사본만 혼자 돌게 하고(우리 차는 트랙 밖에 세워 둠, 상자 없음) 배율마다 바퀴 시간을 잼.
기준: 배율 1(제 페이스)의 바퀴 시간 / 그 배율의 바퀴 시간 = 실제 페이스 비율. 벽 접촉도 같이 셈(느리게 한 차가 이상하게 달리지 않나).
인자: checkpoint [배율들 1,0.95,0.9,0.8,0.7,0.6] [배율당 경주 수 2] [초 40] [장치 cpu]
"""
import os, sys, json, time
os.environ["F1SIM_NO_STAGGER"] = "1"
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(False)

ck_path = sys.argv[1]
facs = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,0.95,0.9,0.8,0.7,0.6").split(",")]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 2
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
dev = torch.device(sys.argv[5] if len(sys.argv) > 5 else "cpu")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
from f1sim.learn.plan_policy import lidar_mount_of, load_actor2
os.environ["F1SIM_LIDAR_MOUNT_X"] = str(lidar_mount_of(ck))
os.environ["F1SIM_PLAN_LAT_MARGIN"] = str((ck.get("args") or {}).get("plan_lat_margin", 0.95))
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_reward import GoalField
from f1sim import plan_steps as PS

W = os.path.join(os.path.dirname(PR.WORLD), "plan_world_korea26scan_race.json")
G = len(facs) * per
env, lim, a = PR.build(W, 2 * G, 11, dev, overrides={"warps": 0, "prop_fraction": 0.0, "opp_present": 1.0,
                                                       "spawn_half_lap": 0.0, "align_episodes": True})
lid = env.learner_ids
env.ep_limit[:] = 10 ** 6
env.sim.park(lid)                                         # our car off the map: the copy drives alone
actor = load_actor2(ck, dev)
so = PR.SelfOpponents(env, lim, 1.0, seed=5, max_pool=1, p_newest=1.0, pace=(0.6, 0.95, 0.0))
so.add(actor, int(ck.get("update") or 0))
fac = torch.tensor([f for f in facs for _ in range(per)], device=dev)
so.pace[:] = fac
goal = GoalField(env.sim.track, dev)
rows = so.rows
tid = env.sim.tid[rows]
L = goal.lap[tid]
d_prev = goal.remaining(env.sim.state[rows, :2].float(), tid)
cross = [[] for _ in range(G)]
c0 = env.sim.contact_counts[rows, 0:3].clone()
v_sum = torch.zeros(G, device=dev)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
T = int(round(secs / PS.DT))
t0 = time.time()
for t in range(T):
    so.plan()
    env.sim.ideal_vcap[lid] = 1e3
    env.ideal_cmd[lid] = 0.0
    for j in range(PS.SUB):
        so.command(j)
        _, _, te, trc, _ = env.step(act0)
        so.stepped(te | trc)
    so.after()
    d = goal.remaining(env.sim.state[rows, :2].float(), tid)
    d = torch.where(torch.isfinite(d), d, d_prev)
    up = (d - d_prev) > 0.5 * L                           # the line crossed forwards
    for g in torch.nonzero(up).flatten().tolist():
        cross[g].append(t + 1)
    d_prev = d
    v_sum += env.sim.state[rows, 3].float()
    if (t + 1) % 100 == 0:
        print(f"  {(t + 1) * PS.DT:.0f} s / {secs:.0f} s ({time.time() - t0:.0f} s)", flush=True)
walls = (env.sim.contact_counts[rows, 0:3] - c0)[:, 0].float()
lap = []
for g in range(G):
    c = cross[g]
    lap.append(float(np.mean(np.diff(c))) * PS.DT if len(c) >= 2 else float("nan"))
lap = np.array(lap)
ref = np.nanmean(lap[np.array([abs(f - 1.0) < 1e-4 for f in fac.tolist()])])
print(f"\n[복사본 속도 배율 측정] {os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)} · Korea 2026 (bag 스캔) 원래 지도, "
      f"상자 없음, 혼자, {secs:.0f} s, 배율당 {per}판 (plan대로 정확히 실행)")
print(f"{'배율':>6} | {'바퀴 시간 s':>10} | {'실제 페이스 비율':>14} | {'평균 속도 m/s':>12} | {'벽 접촉/분':>9} | 잰 바퀴")
out = {}
for f in facs:
    m = np.array([abs(x - f) < 1e-4 for x in fac.tolist()])
    lt = np.nanmean(lap[m]); n_l = int(sum(max(0, len(cross[g]) - 1) for g in np.nonzero(m)[0]))
    vs = float(v_sum[torch.as_tensor(m, device=dev)].mean()) / T
    wm = float(walls[torch.as_tensor(m, device=dev)].sum()) / (secs / 60.0) / max(int(m.sum()), 1)
    print(f"{f:6.2f} | {lt:10.2f} | {ref / lt:14.3f} | {vs:12.2f} | {wm:9.2f} | {n_l}")
    out[f"{f:g}"] = {"lap_s": float(lt), "pace_ratio": float(ref / lt), "speed": vs, "walls_per_min": wm, "laps": n_l}
print(json.dumps(out))
