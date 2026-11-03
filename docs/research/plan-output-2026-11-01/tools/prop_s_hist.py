"""학습 세계에서 라인을 막는 상자가 트랙 어디에 놓이나 (11-03): 콘솔 30배치에서 못 피하는 상자는 두 구간에만 몰림 -- s 38~39 m(시작선
앞 빠른 구간: 배치 5 #1, 배치 0 #14, 상자 하나 자리 5)와 s 2.5~3.8 m(시작선 직후 머리핀 입구: 배치 11, 20, 15). 같은 구간의 라인 막는
상자는 모두 박고 다른 구간은 거의 다 피함. 가설: 학습 세계(뒤틀린 트랙 16개 + 원본, 같은 생성기)가 그 구간에 라인 막는 상자를 거의 안 놓아
정책이 거기서 반응을 배운 적이 없음. 측정: 학습과 같은 인자로 환경을 만들어 리셋을 거듭하며, 상자마다 자기 트랙에서 s(랩 비율), 가로 위치,
그 자리 최소시간 레이싱 라인 가로 위치 -> |상자 - 라인| < 상자 반크기 + 0.155(차 반폭)이면 라인을 막음. 랩을 2 % 칸으로 나눠 셈.
인자: 체크포인트(학습 인자) 환경수 리셋수 [--prop-line 비율: 라인 위 상자를 켠 학습 세계]"""
import sys, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR

pck, n_env, n_rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
line = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
dev = torch.device("cuda")
ck = torch.load(pck, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device, a.prop_line = n_env, 5, "cuda", line
env = SR.build_env(a, dev)
proc, tr = env.procedural, env.sim.track
pl = SR._line_planner(env)
NB = 50
h_all, h_on = np.zeros(NB), np.zeros(NB)
L0 = None
for rep in range(n_rep):
    env.reset()
    tid_e = env.sim.tid
    sid, pose, dd = proc.p_sid, proc.p_poses, proc.p_d
    ext = torch.where(torch.isfinite(dd), dd, torch.zeros_like(dd)).amax(-1)     # (B, C)
    B, C = sid.shape
    live = sid >= 0
    xy = pose[..., :2].reshape(-1, 2).float()
    tid = tid_e[:, None].expand(B, C).reshape(-1)
    s, lat, _ = tr.project(xy, tid)
    rl = pl.rl_offset(s[:, None], tid[:, None]).reshape(-1)
    L = tr.length[tid]
    frac = (s / L).reshape(B, C)
    on = live & ((lat - rl).abs().reshape(B, C) < ext + 0.155)
    b = (frac * NB).long().clamp(0, NB - 1)
    h_all += np.bincount(b[live].cpu().numpy(), minlength=NB)
    h_on += np.bincount(b[on].cpu().numpy(), minlength=NB)
    if L0 is None:
        L0 = float(tr.length[0])
print(f"== 라인 위 상자 비율 {line}: {n_env} 환경 x {n_rep} 리셋, 트랙 0 길이 {L0:.1f} m, 상자 {int(h_all.sum())}개 중 라인 막음 {int(h_on.sum())}개", flush=True)
print("  랩 칸(2 %)  s 범위(트랙 0 기준)  상자   라인 막음", flush=True)
for i in range(NB):
    print(f"  {i * 2:3d}~{i * 2 + 2:3d} %   {i * L0 / NB:5.1f}~{(i + 1) * L0 / NB:5.1f} m  {int(h_all[i]):6d}  {int(h_on[i]):6d}  {'#' * int(60 * h_on[i] / max(h_on.max(), 1))}", flush=True)
