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
# how often consecutive plans swing to the other side: the 0.3 s point's sideways offset (car frame) flips sign by > 10 cm
plans, poses, yaws, vels, pv, k1s, v1s, side3 = [], [], [], [], [], [], [], []         # per decision: world (B, len(H), 2); car (B, 2), yaw, speed; plan speed (B, len(H))
for k in range(int(30.0 / dt)):
    if k % R == 0:
        last = tp(last); act_full[lid] = last
        st = env.sim.state[lid, :3].clone()             # the pose the plan is anchored at (the tracker's, before the step)
        env.step(act_full)
        if getattr(env, "_tw_carry", None) is not None:  # a carried plan: the segments the step drove
            kk, vv = env._tw_carry.segments(); kk, vv = kk[lid], vv[lid]
        else:
            kk, vv = plan_tw.decode(last, a.plan_format)
        dn = plan_tw.dense(kk, vv)                        # (B, n, 5) body frame, t in col 4
        c, s_ = torch.cos(st[:, 2]), torch.sin(st[:, 2])
        pts = []
        for h in H:
            j = int(round(h / dn[0, -1, 4].item() * (dn.shape[1] - 1)))
            bx, by = dn[:, j, 0], dn[:, j, 1]
            pts.append(torch.stack([st[:, 0] + bx * c - by * s_, st[:, 1] + bx * s_ + by * c], 1))
        plans.append(torch.stack(pts, 1)); poses.append(st[:, :2].clone())
        yaws.append(st[:, 2].clone()); vels.append(env.sim.state[lid, 3].clone())
        pv.append(torch.stack([dn[:, int(round(h / dn[0, -1, 4].item() * (dn.shape[1] - 1))), 3] for h in H], 1))
        k1s.append(kk[:, 0].clone()); v1s.append(vv[:, 0].clone())
        j3 = int(round(0.3 / dn[0, -1, 4].item() * (dn.shape[1] - 1)))
        side3.append(dn[:, j3, 1].clone())               # the 0.3 s point's sideways offset in the car's frame
        continue
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
# split the gap to where the car really was into along its heading (speed) and across it (path shape)
Y = torch.stack(yaws, 0); Vc = torch.stack(vels, 0); PV = torch.stack(pv, 0)
for i, h in enumerate(H):
    k_ahead = int(round(h / step_s))
    d = P[:-k_ahead, :, i] - X[k_ahead:]
    yy = Y[k_ahead:]
    along = d[..., 0] * torch.cos(yy) + d[..., 1] * torch.sin(yy)
    cross = -d[..., 0] * torch.sin(yy) + d[..., 1] * torch.cos(yy)
    dv = PV[:-k_ahead, :, i] - Vc[k_ahead:]
    # sideways gap signed by the way the car really turned over [t, t + h]: + = the plan bends further into the turn
    turn = torch.sign(torch.atan2(torch.sin(Y[k_ahead:] - Y[:-k_ahead]), torch.cos(Y[k_ahead:] - Y[:-k_ahead])))
    big = (Y[k_ahead:] - Y[:-k_ahead]).abs() > 0.05
    inside = (cross * turn)[big]
    print(f"  {h:.1f} s: 차가 실제로 돈 방향 기준 plan의 옆 차이 중앙 {float(inside.median()) * 100:+.0f} cm (+ = plan이 더 안쪽으로 꺾임), "
          f"더 안쪽인 비율 {float((inside > 0).float().mean()) * 100:.0f} %", flush=True)
    print(f"  {h:.1f} s: 진행 방향 차이(plan이 앞서면 +) 중앙 {float(along.median()) * 100:+.0f} cm, |중앙| {float(along.abs().median()) * 100:.0f} cm | "
          f"옆 방향 |중앙| {float(cross.abs().median()) * 100:.0f} cm | plan 속도 - 실제 속도 중앙 {float(dv.median()):+.2f} m/s (|중앙| {float(dv.abs().median()):.2f})", flush=True)
# the first segment's curvature vs the curvature the car really drove over the next decision, and the lateral accel they mean
K1 = torch.stack(k1s, 0)[:-1]; V1 = torch.stack(v1s, 0)[:-1]
dyaw = torch.atan2(torch.sin(Y[1:] - Y[:-1]), torch.cos(Y[1:] - Y[:-1]))
vm = 0.5 * (Vc[1:] + Vc[:-1])
kd = dyaw / (vm.abs() * step_s).clamp(min=1e-3)
fast = vm.abs() > 3.0
for name_, x_ in (("plan 첫 구간 곡률 x 실제 속도^2", (K1 * vm ** 2).abs()), ("실제로 돈 곡률 x 속도^2", (kd * vm ** 2).abs())):
    q = x_[fast]
    print(f"  3 m/s 넘을 때 {name_} [m/s^2]: 중앙 {float(q.median()):.1f}, p90 {float(q.quantile(0.9)):.1f}, p99 {float(q.quantile(0.99)):.1f}", flush=True)
r = (K1[fast] * kd[fast] > 0)
print(f"  같은 방향으로 꺾는 비율 {float(r.float().mean()) * 100:.0f} %, plan 곡률이 실제의 2배 넘는 비율 {float(((K1.abs() > 2 * kd.abs()) & fast).float().sum() / fast.float().sum()) * 100:.0f} %", flush=True)
S3 = torch.stack(side3, 0)
fl = (S3[1:] * S3[:-1] < 0) & (S3[1:].abs() > 0.1) & (S3[:-1].abs() > 0.1)
print(f"  0.3 s 점이 바로 전 plan과 반대쪽(둘 다 10 cm 넘게)으로 뒤집힌 결정: {float(fl.float().mean()) * 100:.1f} % "
      f"(1 s에 {float(fl.float().mean()) / step_s:.2f}번)", flush=True)
v = env.sim.state[lid, 3].abs().mean()
print(f"  (같은 점이라도 0.05 s 뒤 plan은 차가 그만큼 나아간 자리에서 다시 재므로, 차 속도 ~{float(v):.1f} m/s면 그대로여도 ~{float(v) * step_s * 100:.0f} cm 옮겨 감)", flush=True)
