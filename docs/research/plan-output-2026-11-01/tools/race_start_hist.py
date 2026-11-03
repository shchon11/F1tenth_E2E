"""학습 세계 경주는 어떻게 시작하나 (11-03): 상자 단계가 풀린 선생 gwt를 그대로 상대와 붙이면(콘솔 경주, 시드 777) forzaeth 0.6배 20판
모두 추월하지만 차 접촉 53(세게 32) -- 장면은 출발 1~2 s 안 첫 머리핀에서 바로 앞 느린 차 옆을 비집다 부딪힘. 콘솔 경주는 두 차가
출발선(s ~0) 근처에 2 m쯤 떨어져 멈춘 채 시작하고 바로 머리핀. 상자 때처럼 학습 세계에 이 장면이 드문지: 상대 단계와 같은 인자로
환경을 만들어 리셋을 거듭하며, 경주마다 내 차의 랩 위치(2 % 칸), 상대와의 앞뒤 간격(호 길이, + = 상대가 앞), 두 차의 출발 속도.
인자: 체크포인트(상대 단계 학습 인자) 환경수 리셋수"""
import sys, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR

pck, n_env, n_rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
ck = torch.load(pck, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, 7, "cuda"
env = SR.build_env(a, dev)
tr = env.sim.track
lid = env.learner_ids
NB = 25
h_s = np.zeros(NB)
gaps, v_me, v_op = [], [], []
for rep in range(n_rep):
    env.reset()
    st = env.sim.state
    tid = env.sim.tid
    s_all, _, _ = tr.project(st[:, :2].float(), tid)
    L = tr.length[tid]
    o = env.sim.other_idx[lid][:, 0]
    s_me, s_op = s_all[lid], s_all[o]
    Lm = L[lid]
    gap = torch.remainder(s_op - s_me + Lm / 2, Lm) - Lm / 2
    frac = s_me / Lm
    h_s += np.bincount((frac * NB).long().clamp(0, NB - 1).cpu().numpy(), minlength=NB)
    gaps += gap.cpu().tolist(); v_me += st[lid, 3].cpu().tolist(); v_op += st[o, 3].cpu().tolist()
gaps, v_me, v_op = np.array(gaps), np.array(v_me), np.array(v_op)
print(f"== {n_env} 환경 x {n_rep} 리셋 = 경주 {len(gaps)}개 (트랙 0 길이 {float(tr.length[0]):.1f} m)", flush=True)
print("  내 차 출발 랩 위치(4 % 칸): " + " ".join(f"{int(x)}" for x in h_s), flush=True)
for lo, hi in ((-50, -5), (-5, -1), (-1, 1), (1, 5), (5, 50)):
    m = (gaps >= lo) & (gaps < hi)
    print(f"  상대 간격 {lo:+d}~{hi:+d} m: {m.mean() * 100:5.1f} %", flush=True)
print(f"  출발 속도 내 차 중앙 {np.median(v_me):.2f} (0.3 m/s 미만 {np.mean(v_me < 0.3) * 100:.0f} %), 상대 중앙 {np.median(v_op):.2f}", flush=True)
near = np.abs(gaps) < 5
print(f"  5 m 안에서 함께 출발: {near.mean() * 100:.0f} %, 그중 둘 다 1 m/s 미만: {np.mean(near & (v_me < 1) & (v_op < 1)) * 100:.0f} %", flush=True)
