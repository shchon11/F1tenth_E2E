"""선생 네트워크 용량 판별 (11-03, 사용자: "학습 방법과 세팅에서 문제가 없다면 모델이 학습을 담기에 너무 부족하다고 생각할 수도 있지").
상자 정책(srp25h2_props u200)은 깨끗한 맵에서도 7.18 s(혼자 정책 6.50 s) -- 빈 길 운전이 상자 단계 동안 무너짐. 용량 탓이면 같은
구조(306 -> 256 -> 256 -> 8, tanh)가 "빈 길 = 혼자 정책, 상자 근처 = 상자 정책"인 합성 행동을 담지 못해야 한다.
학습 세계(체크포인트 args 그대로)에서 합성 정책으로 몰며(탐색 잡음을 조금 섞어 상태를 넓힘) 상태와 합성 행동을 모으고, 새 네트워크를
지도학습으로 맞춘다: 폭 256(지금과 같음) vs 768. 따로 떼어 둔 상태에서 빈 길 / 상자 근처별로 실행 속도 명령(첫 경로점, 제동 한계 적용)과
첫 곡률의 오차. 256이 두 쪽 다 작게 맞추면 용량은 충분 -> 원인은 강화학습 신호 쪽.
인자: 상자 체크포인트 혼자 체크포인트 환경수 초 시드"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim import mpc

p_path, s_path, n_env, secs, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
dev = torch.device("cuda")
ck = torch.load(p_path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, seed, "cuda"
a.base, a.base_reach, a.base_hold = s_path, 10.0, 10                     # the composite: SoloMix as the deployed teacher
ck_mix = dict(ck); ck_mix["args"] = dict(ck["args"], base=s_path, base_reach=10.0, base_hold=10)
env = SR.build_env(a, dev)
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=seed)
lid = env.learner_ids; B = int(lid.numel())
remover = SR.PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=seed)
tp = SR.TeacherPolicy(env, ck_mix, dev)
inner = env.tracker._inner if hasattr(env.tracker, "_inner") else env.tracker
spec = inner.spec
X, Y, G, V = [], [], [], []
act = torch.zeros(B, tp.A, device=dev); eps = torch.zeros_like(act)
act_full = torch.zeros(env.B, env.act_dim, device=dev)
R = int(a.repeat)
with torch.no_grad():
    for t in range(int(secs / env.sim.control_dt) // R):
        target = tp(act)                                                 # the composite's mean action (gated mixture)
        if t % 4 == 0:
            X.append(tp.nrm(tp.o).cpu()); Y.append(target.cpu()); G.append(tp.mix.gate.cpu()); V.append(env.sim.state[lid, 3].float().cpu())
        eps = 0.9 * eps + math.sqrt(1 - 0.81) * torch.randn_like(eps)
        act = (target + 0.5 * torch.tensor([0.05] * 4 + [0.2] * 4, device=dev) * eps).clamp(-1, 1)   # half the training noise
        act_full[lid] = act
        for _ in range(R):
            env.step(act_full)
            remover.step()
X = torch.cat(X); Y = torch.cat(Y); G = torch.cat(G); V = torch.cat(V)
n = X.shape[0]
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
te, tr = perm[: n // 10], perm[n // 10:]
print(f"== 상태 {n}개 (빈 길 {float((~G).float().mean()) * 100:.0f} %, 상자 근처 {float(G.float().mean()) * 100:.0f} %), 시험용 {te.numel()}", flush=True)
cap = torch.full((te.numel(),), 10.0, device=dev)


def v1k1(y, v):
    k, _, vv = mpc.decode_wp4(y.to(dev), v.to(dev), 10.0, cap, spec)
    return vv[:, 0].cpu(), k[:, 0].cpu()


vt, kt = v1k1(Y[te], V[te])
for h in (256, 768):
    torch.manual_seed(1)
    net = SR.mlp(X.shape[1], 8, h).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xd, Yd = X[tr].to(dev), Y[tr].to(dev)
    for it in range(20000):
        j = torch.randint(0, Xd.shape[0], (4096,), device=dev)
        loss = ((net(Xd[j]) - Yd[j]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 12000:
            for g_ in opt.param_groups:
                g_["lr"] = 3e-4
    with torch.no_grad():
        yp = net(X[te].to(dev)).clamp(-1, 1)
        tr_loss = float(((net(Xd[:20000]) - Yd[:20000]) ** 2).mean())
    vp, kp = v1k1(yp.cpu(), V[te])
    for name, m in (("빈 길(혼자 정책 몫)", ~G[te]), ("상자 근처(상자 정책 몫)", G[te])):
        if m.sum() == 0:
            continue
        ev = (vp[m] - vt[m]).abs(); ek = (kp[m] - kt[m]).abs()
        print(f"  폭 {h}: {name}: 실행 속도 명령 오차 중앙 {ev.median():.3f}, p90 {ev.quantile(0.9):.3f} m/s, 평균 치우침 {(vp[m] - vt[m]).mean():+.3f} | "
              f"첫 곡률 오차 중앙 {ek.median():.3f} 1/m | 학습 오차(행동 단위 MSE) {tr_loss:.5f}", flush=True)
