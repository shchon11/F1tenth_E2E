"""전체 reset 뒤 첫 episode와 자동 reset 뒤 episode의 멈춤 비교 (11-06 15:00). 같은 잣대(출발 3 s 이후 0.3 m/s 아래)로 학습 env를 콘솔처럼
몰면 55-66 % 멈추는데, 40 s episode가 계속 돌아가는 측정(stuck_share)은 6-8 % -> episode가 env.reset()(콘솔·평가가 매 배치 쓰는 전체 reset)으로
시작하느냐, 시간 끝에 env 안에서 자동 reset(학습이 쓰는 경로)으로 시작하느냐의 차이인지. episode 40 s, 시작 엇갈림 없음, 80 s: [3, 40) s = 전체
reset 뒤, [43, 80) s = 자동 reset 뒤. 인자: 체크포인트 차수"""
import os, sys, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime

path, n_env = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, 21, "cuda"
a.prop_count, a.prop_fraction, a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop, a.stage = "fixed", 1.0, 0.0, 0, 0.0, 0.0, "props"
a.episode_s = 40.0
env = SR.build_env(a, dev)
env.ecfg.stagger_first_episode = False
prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=21)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel())
last = torch.zeros(B, tp.A, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
stop = [torch.zeros(B, device=dev), torch.zeros(B, device=dev)]; dt = env.sim.control_dt
for k in range(int(80.0 / dt)):
    if k % int(a.repeat) == 0:
        last = tp(last)
        act_full[lid] = last
    _, _, te, trc, _ = env.step(act_full)
    d_ = (te | trc)[lid]
    last = torch.where(d_[:, None], torch.zeros_like(last), last)
    t = k * dt
    if 3.0 <= t < 40.0:
        stop[0] += (env.sim.state[lid, 3].abs() < 0.3).float() * dt
    elif 43.0 <= t < 80.0:
        stop[1] += (env.sim.state[lid, 3].abs() < 0.3).float() * dt
for i, lab in enumerate(("전체 reset(env.reset) 뒤 첫 episode", "자동 reset 뒤 episode")):
    s = stop[i].cpu().numpy()
    print(f"{lab}: 멈춰 있던 시간 {s.sum() / (B * 37) * 100:.0f} %, 5 s 넘게 멈춘 차 {int((s >= 5).sum())}/{B}", flush=True)
