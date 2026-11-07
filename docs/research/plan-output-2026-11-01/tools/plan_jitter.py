"""시간별 waypoint plan이 결정마다 얼마나 흔들리나 (11-07 11:40, 사용자: "왜이리 planner가 불안정하게 뽑히냐"). 학습 인자로 env를 만들고(상자 없음,
상대 없음, 차 N대) 평균 행동으로 30 s 달리며, 결정(0.05 s)마다 policy의 plan을 세계 좌표로 저장. 바로 앞 결정의 plan과 비교해 같은 미래 시각
(지금부터 0.1·0.3·0.5·0.7·0.9 s 뒤)의 위치가 얼마나 옮겨 갔는지, 그리고 그 시각에 차가 실제로 있던 곳과 plan의 거리를 셈. 인자: 체크포인트 차수"""
import os, sys, math, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim import plan_tw

torch.set_grad_enabled(False)
path, n_env = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, 81, "cuda"
a.prop_fraction, a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop, a.stage = 0.0, 0.0, 0, 0.0, 0.0, "props"
a.episode_s = 3600.0
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=81)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel()); R = int(a.repeat); dt = env.sim.control_dt
last = torch.zeros(B, tp.A, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
H = (0.1, 0.3, 0.5, 0.7, 0.9) if a.plan_format == "tw" else (0.1, 0.2, 0.3, 0.4, 0.5)
plans, poses = [], []                                     # per decision: world (B, len(H), 2); car (B, 2)
for k in range(int(30.0 / dt)):
    if k % R == 0:
        last = tp(last); act_full[lid] = last
        kk, vv = plan_tw.decode(last, a.plan_format)
        dn = plan_tw.dense(kk, vv)                        # (B, n, 5) body frame, t in col 4
        st = env.sim.state[lid, :3]
        c, s_ = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        pts = []
        for h in H:
            j = int(round(h / dn[0, -1, 4].item() * (dn.shape[1] - 1)))
            bx, by = dn[:, j, 0], dn[:, j, 1]
            pts.append(torch.stack([st[:, 0] + bx * c - by * s_, st[:, 1] + bx * s_ + by * c], 1))
        plans.append(torch.stack(pts, 1)); poses.append(st[:, :2].clone())
    env.step(act_full)
P = torch.stack(plans, 0); X = torch.stack(poses, 0)       # (T, B, H, 2), (T, B, 2)
step_s = R * dt
print(f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: 차 {B}대 x 30 s, 빈 트랙, 평균 행동, 결정 {P.shape[0]}개", flush=True)
for i, h in enumerate(H):
    # the plan of decision k for time t_k + h vs the plan of decision k+1 for the same instant (its h - step_s)
    j_prev = i
    hh = h - step_s
    # nearest horizon of the next plan to that instant: interpolate by recomputing would be exact; use the same h (0.05 s later) as an upper bound
    d_next = (P[1:, :, i] - P[:-1, :, i]).norm(dim=-1)            # same horizon, 0.05 s later
    k_ahead = int(round(h / step_s))
    real = (P[:-k_ahead, :, i] - X[k_ahead:]).norm(dim=-1) if k_ahead < P.shape[0] else torch.zeros(1)
    # what a plan moves by in 0.05 s just from the car moving on: |v| x 0.05
    print(f"  {h:.1f} s 앞 점: 다음 결정에서 옮겨 간 거리 중앙 {float(d_next.median()) * 100:.1f} cm (p90 {float(d_next.quantile(0.9)) * 100:.1f}) | "
          f"그 시각 실제 차 위치와 거리 중앙 {float(real.median()) * 100:.1f} cm (p90 {float(real.quantile(0.9)) * 100:.1f})", flush=True)
v = env.sim.state[lid, 3].abs().mean()
print(f"  (같은 점이라도 0.05 s 뒤 plan은 차가 그만큼 나아간 자리에서 다시 재므로, 차 속도 ~{float(v):.1f} m/s면 그대로여도 ~{float(v) * step_s * 100:.0f} cm 옮겨 감)", flush=True)
