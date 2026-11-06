"""콘솔 평가와 똑같은 잣대를 학습 env에 (11-06 14:55). 콘솔 배치 평가는 시드 777·123·456 모두 50~58 % 서 있고, 학습 env는 평가처럼 맞춰도 2 s 넘게 서
있는 시간 6~8 % -> 잣대 차이인지 env 차이인지. 학습 인자로 env를 만들되 콘솔처럼(상자 전부·안 치움·원래 트랙·상대 없음·차 1대·상자 뒤 출발
없음·episode 길게) 하고, 차 N대를 40 s 동안 콘솔 실행기와 같이(2 step마다 평균 행동, 사이엔 유지) 몰며 콘솔 평가와 같은 잣대로 셈: 출발 3 s
이후 0.3 m/s 아래 시간, 5 s 넘게 멈춘 차. 인자: 체크포인트 차수 [noise]"""
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
a.episode_s = 3600.0
if len(sys.argv) > 4:
    a.prop_count = sys.argv[4]                # "uniform": 1 .. all of each layout's pieces, as the training world draws them
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=21)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel())
last = torch.zeros(B, tp.A, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
stop = torch.zeros(B, device=dev); dt = env.sim.control_dt
for k in range(int(40.0 / dt)):
    if k % int(a.repeat) == 0:
        if len(sys.argv) > 3 and sys.argv[3] == "noise":          # the training's exploration noise
            tp.o = tp.sobs(last)
            d_ = tp.net.dist(tp.nrm(tp.o))
            last = SR.clamp_any(d_.mean + d_.stddev * torch.randn_like(d_.mean), tp.ar)
        else:
            last = tp(last)
        act_full[lid] = last
    env.step(act_full)
    if k * dt >= 3.0:
        stop += (env.sim.state[lid, 3].abs() < 0.3).float() * dt
s = stop.cpu().numpy()
print(f"학습 env를 콘솔처럼, 차 {B}대 x 40 s, {'탐색 흔들림 섞음' if len(sys.argv) > 3 and sys.argv[3] == 'noise' else '평균 행동'}, 상자 수 {a.prop_count}: 멈춰 있던 시간 {s.sum():.0f} s / {B * 37:.0f} s ({s.sum() / (B * 37) * 100:.0f} %), "
      f"5 s 넘게 멈춘 차 {int((s >= 5).sum())}/{B}", flush=True)
