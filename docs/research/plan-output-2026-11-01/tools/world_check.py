"""plan 학습 세계가 대회 방식(배치가 에피소드 동안 고정 -> 치움, 도중에 새로 생기지 않음)과 맞나 (11-09, 사용자: 기억으로 적응 "A1",
기억이 쓸모 있으려면 세계가 그래야 함). plan_rl과 같은 세계(plan_rl.build, 학습 조건 실행)를 policy로 몰며, 학습 차마다 0.1 s마다:
상자가 있나(그 차 세계의 배치에 산 상자), 상대 차가 있나, 에피소드가 새로 시작했나. 세는 것:
  * 에피소드 시작 때 상자 있음 / 상대 있음 비율, 에피소드 길이
  * 에피소드 도중에 상자가 새로 생김 / 사라짐, 상대가 생김 / 사라짐 (그 차의 재시작 없이)
인자: checkpoint 차수 초"""
import os, sys, argparse, json
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import plan_world, PT_IDX, PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
dev = torch.device("cuda")
env, lim, world = PR.build(PR.WORLD, n_env, 5, dev, fmt="v2")
lid = env.learner_ids; Bl = int(lid.numel())
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
proc = env.procedural
others = env.sim.other_idx[lid] if env.M > 1 else None
H, O, RS, EID, EPS, OEP = [], [], [], [], [], []
for kd in range(int(secs / PS.DT)):
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    out, _, _ = drv.decision(torch.tanh(mean), p)
    has_p = (proc.p_sid[env.sim.eid[lid]] >= 0).any(-1) if proc is not None else torch.zeros(Bl, dtype=torch.bool, device=dev)
    absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
    has_o = (~absent[others]).any(-1) if others is not None else torch.zeros(Bl, dtype=torch.bool, device=dev)
    H.append(has_p.cpu()); O.append(has_o.cpu()); RS.append(out["restarted"].cpu())
    EID.append(env.sim.eid[lid].cpu()); EPS.append(env.ep_step[lid].cpu())
    OEP.append(env.ep_step[others[:, 0]].cpu() if others is not None else env.ep_step[lid].cpu())
H, O, RS = (torch.stack(z, 1).numpy() for z in (H, O, RS))         # (Bl, T)
EID, EPS, OEP = (torch.stack(z, 1).numpy() for z in (EID, EPS, OEP))
# a few appearance events, to see the mechanism
shown = 0
for b in range(Bl):
    for t in range(1, H.shape[1]):
        if (not H[b, t - 1]) and H[b, t] and not RS[b, t] and shown < 6:
            print(f"  상자 생김: 차 {b} 결정 {t}: eid {EID[b, t - 1]}->{EID[b, t]}, 그 차 ep_step {EPS[b, t - 1]}->{EPS[b, t]}, "
                  f"상대 ep_step {OEP[b, t - 1]}->{OEP[b, t]}, 상대 있음 {O[b, t - 1]}->{O[b, t]}")
            shown += 1
ep_len, start_p, start_o = [], [], []
add_p = rem_p = add_o = rem_o = 0
n_dec = 0
for b in range(Bl):
    # the decision a restart happened in already ends in the new episode (its state is after the reset)
    starts = [0] + [t for t in np.nonzero(RS[b])[0] if t > 0]
    ends = starts[1:] + [H.shape[1]]
    for s0, s1 in zip(starts, ends):
        if s1 - s0 < 2:
            continue
        hp, ho = H[b, s0:s1], O[b, s0:s1]
        start_p.append(hp[0]); start_o.append(ho[0]); ep_len.append((s1 - s0) * PS.DT)
        add_p += int(np.sum(~hp[:-1] & hp[1:])); rem_p += int(np.sum(hp[:-1] & ~hp[1:]))
        add_o += int(np.sum(~ho[:-1] & ho[1:])); rem_o += int(np.sum(ho[:-1] & ~ho[1:]))
        n_dec += s1 - s0
mins = n_dec * PS.DT / 60
print(f"학습 세계 (plan_rl.build), 학습 차 {Bl}대 x {secs:.0f} s, 에피소드 조각 {len(ep_len)}개 (길이 중앙 {np.median(ep_len):.1f} s)", flush=True)
print(f"  에피소드 시작 때 상자 있음 {np.mean(start_p) * 100:.0f} %, 상대 있음 {np.mean(start_o) * 100:.0f} %")
print(f"  에피소드 도중(재시작 없이): 상자 새로 생김 {add_p}번 ({add_p / mins:.2f}/분), 상자 사라짐 {rem_p}번 ({rem_p / mins:.2f}/분), "
      f"상대 생김 {add_o}번 ({add_o / mins:.2f}/분), 상대 사라짐 {rem_o}번 ({rem_o / mins:.2f}/분)")
