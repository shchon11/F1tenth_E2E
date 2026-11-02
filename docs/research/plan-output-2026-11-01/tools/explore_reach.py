"""탐색 잡음이 차를 옆으로 얼마나 보내나 (11-03): 상자 정책(--vis-actor, 잡음 상관 0)은 주행선 위 상자에 선을 바꾸지 않고, 상자 입력
가중치가 200회에 0.14~0.16밖에 안 자람(상관 0.9였던 srp25h2의 통로는 0.30). 가설: 결정마다 독립인 잡음은 정책이 바로 되돌리므로
상자를 비켜 갈 만큼(0.3 m 넘게) 옆으로 가는 일이 거의 없어, 비켜 가는 쪽의 이득이 표본에 안 잡힘 -- 늦추기(속도 잡음은 쌓임)만 배움.
학습과 같은 세계(체크포인트 인자, 같은 시드로 리셋해 같은 출발·같은 배치), 같은 정책을 (가) 평균 행동 (나) 학습 잡음 상관 0
(다) 상관 0.9로 4 s씩 몰아, 같은 시각의 가로 위치·속도 차이(잡음 - 평균)의 분포. 상자 정책이 모는(게이트 켜짐) 차만 따로.
잡음은 따로 둔 난수 생성기로(환경 난수 순서 그대로). 인자: 체크포인트 환경수 시드"""
import os, sys, math, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime

path, n_env, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, seed, "cuda"
env = SR.build_env(a, dev)
prepare_graph_runtime(env, log=lambda _t: None)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel()); R = int(a.repeat)
std = torch.maximum(tp.net.log_std.exp(), tp.net.std_floor).detach()
print(f"== {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: 행동 잡음 표준편차 {[round(float(x), 3) for x in std]}", flush=True)
SECS = 4.0
T = int(SECS / env.sim.control_dt) // R


def run(corr):
    env.reset(seed=seed)
    if tp.mix is not None:
        tp.mix.left.zero_(); tp.mix.gate.fill_(True)
    g = torch.Generator(device=dev); g.manual_seed(12345)
    last = torch.zeros(B, tp.A, device=dev); eps = torch.randn(B, tp.A, device=dev, generator=g)
    af = torch.zeros(env.B, env.act_dim, device=dev)
    lat, v, gate = [], [], []
    with torch.no_grad():
        for t in range(T):
            tp.o = tp.sobs(last)
            mu = tp.net.pi(tp.nrm(tp.o))
            if corr is None:
                act = mu
            else:
                eps = corr * eps + math.sqrt(1 - corr * corr) * torch.randn(B, tp.A, device=dev, generator=g)
                act = mu + std * eps
            exe = SR.clamp_any(act, tp.ar)
            if tp.mix is not None:
                exe = tp.mix(exe, last)
            last = exe
            af[lid] = exe
            for _ in range(R):
                env.step(af)
            r = env.last_result
            lat.append(r.lateral[lid].float().clone()); v.append(env.sim.state[lid, 3].float().clone())
            gate.append(tp.mix.gate.clone() if tp.mix is not None else torch.ones(B, dtype=torch.bool, device=dev))
    return torch.stack(lat), torch.stack(v), torch.stack(gate)


L0, V0, G0 = run(None)
L0b, _, _ = run(None)
print(f"  평균 행동 두 번(결정론 확인): 가로 차이 최대 {float((L0 - L0b).abs().max()):.4f} m", flush=True)
dt = env.sim.control_dt * R
for corr in (None, 0.0, 0.9):
    L1, V1, _ = run(corr) if corr is not None else (L0b, run(None)[1], None)
    dl = (L1 - L0).abs(); dv = (V1 - V0)
    line = []
    for ts in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        k = min(int(ts / dt) - 1, T - 1)
        m = G0[max(0, k - int(1.0 / dt)):k + 1].all(0)          # the race policy drove the whole last second (no-noise run)
        d_all = dl[k]; d_g = dl[k][m] if m.any() else torch.zeros(1, device=dev)
        line.append(f"{ts:.2f}s: 가로 |차| 중앙 {float(d_all.median()):.3f} p90 {float(d_all.quantile(0.9)):.3f} "
                    f"(상자 정책 {int(m.sum())}대: 중앙 {float(d_g.median()):.3f} p90 {float(d_g.quantile(0.9)):.3f}, 0.3 m 넘음 {float((d_g > 0.3).float().mean()) * 100:.0f} %) "
                    f"속도 차 표준편차 {float(dv[k].std()):.2f} m/s")
    print(f"  {'평균 행동 다시(기준: 잡음 없이도 생기는 차이)' if corr is None else f'잡음 상관 {corr}'}:", flush=True)
    for l in line:
        print("    " + l, flush=True)
