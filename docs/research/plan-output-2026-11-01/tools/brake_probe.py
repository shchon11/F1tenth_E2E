"""직선 끝에서 일찍·약하게 밟는 이유 확인 (11-09, 사용자 "ㅇㅋㅇㅋ"; accel_usage.py에서 긴 직선 끝 10-12 m에서 가속 57 %, 12-16 m에서
브레이크 48-51 %). 학습 조건(plan대로 정확히), 원래 트랙 빈 트랙에서 policy로 몰며, 직선 끝 구간(바퀴 위치 9.5-14.5 m)의 결정마다:
1) 같은 시작(속도 짐작, 직전 조향)에서 plan 묶음을 그림: 속도 = 처음 k단계 최대 가속 뒤 최대 제동(k = 0..10, 10 = 1 s 내내 가속),
   조향 = policy 자신의 조향, 그것을 +-0.2/+-0.4 옮긴 것, 매끈한 무작위 조향 40개.
   '안전' = 1 s plan이 벽에 안 닿고(plan 충돌 감점 없음) 그 끝에서 최대 제동 + 일정 조향 9가지 중 하나로 1 s 더 가도 안 닿음.
   policy가 밟기 시작한 단계(plan 속도가 처음 줄어드는 단계)와, 안전한 plan이 있는 가장 늦은 k를 비교. 1 s 뒤 진행(결승선까지 벽 사이
   최단 거리가 줄어든 양)도 policy plan과 가장 많이 가는 안전 plan을 비교.
2) 탐색 잡음: 그 결정들의 첫 단계 속도 출력 -- 잡음 없이(평가) / 학습 때처럼 잡음을 넣어 64번 뽑은 평균.
인자: checkpoint 차수 초"""
import os, sys, argparse, json
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import PlanReward
from f1sim.learn.plan_policy import plan_world, PT_IDX, PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M

torch.set_grad_enabled(False)
ck_path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
S_LO, S_HI = float(os.environ.get("S_LO", 9.5)), float(os.environ.get("S_HI", 14.5))
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD))); a.envs, a.seed, a.device = n_env, 7, "cuda"
a.warps, a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0, 0.0, 0.0, 0.0, secs + 5.0
env = SR.build_env(a, dev)
env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=7)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
actor = load_actor2(ck, dev)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
tid = env.sim.tid[lid]
Lap = float(pr.goal.lap[tid][0])
C4 = pr.corners.float()
track = env.sim.track
N, SUB = PS.N_STEP, PS.SUB


def hit_any(x, y, psi, t_rows):
    c, s = torch.cos(psi)[..., None], torch.sin(psi)[..., None]
    cx = x[..., None] + C4[:, 0] * c - C4[:, 1] * s
    cy = y[..., None] + C4[:, 0] * s + C4[:, 1] * c
    pts = torch.stack([cx, cy], -1)
    h = (track.sample_edt(pts, t_rows[:, None, None].expand_as(cx)) <= 0.0).any(-1)
    h[:, 0] = False
    return h.any(1)


# ---- drive, collecting the window's decisions
D = {k: [] for k in ("st", "v0", "d0", "act", "mean", "logstd", "s")}
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    u, mean, log_std, v_hat, p = plan_v2(actor, obs, lim)
    x, y, psi = plan_world(p, st)
    d_now = pr.goal.remaining(st[:, None, :2], tid[:, None])[:, 0]
    s_now = Lap - d_now
    if kd >= 30:
        m = (s_now >= S_LO) & (s_now < S_HI)
        if m.any():
            D["st"].append(st[m]); D["v0"].append(v_hat.clamp(-PS.V_REV, PS.V_FWD)[m]); D["d0"].append(obs.cmd_d[m])
            D["act"].append(torch.tanh(mean)[m]); D["mean"].append(mean[m])
            D["logstd"].append(log_std.expand_as(mean)[m]); D["s"].append(s_now[m])
    env.sim.ideal_vcap[lid] = 1e3
    for j in range(SUB):
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(torch.zeros(env.B, env.act_dim, device=dev))
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"]); obs.set_plan2(p)
D = {k: torch.cat(v) for k, v in D.items()}
nd = D["st"].shape[0]
print(f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}: 직선 끝({S_LO}-{S_HI} m) 결정 {nd}개", flush=True)
t1 = tid[0].repeat(1)

# ---- 1) the plan family
g = torch.Generator(device=dev); g.manual_seed(3)
KS = N + 1                                   # k = 0..10
res = {"k_pol": [], "k_safe": [], "dprog": [], "pol_safe": [], "pol_hit": [], "best_k": []}
turn_c = torch.linspace(-1, 1, 9, device=dev)
for i0 in range(0, nd, 8):
    sl = slice(i0, min(nd, i0 + 8)); nb = sl.stop - sl.start
    act = D["act"][sl]                                                   # (nb, 20)
    # steering candidates: own, own shifted, 40 smooth random
    own = act[:, :N]
    cand = [own, (own + 0.2).clamp(-1, 1), (own - 0.2).clamp(-1, 1), (own + 0.4).clamp(-1, 1), (own - 0.4).clamp(-1, 1)]
    base = torch.rand(nb, 40, 1, device=dev, generator=g) * 2 - 1
    rnd = (base + 0.25 * torch.randn(nb, 40, N, device=dev, generator=g).cumsum(-1) / 2).clamp(-1, 1)
    T = torch.cat([c[:, None] for c in cand] + [rnd], 1)                 # (nb, ST, N)
    ST = T.shape[1]
    ks = torch.arange(KS, device=dev)
    spd = torch.where(torch.arange(N, device=dev)[None, :] < ks[:, None], 1.0, -1.0)   # (KS, N)
    # every (decision, k, steering) plus the policy's own plan at k index -1
    A_t = T[:, None].expand(nb, KS, ST, N)
    A_s = spd[None, :, None].expand(nb, KS, ST, N)
    A = torch.cat([A_t, A_s], -1).reshape(-1, 2 * N)
    A = torch.cat([A, act], 0)                                           # last nb rows = the policy's own plans
    rep = lambda t_: torch.cat([t_.repeat_interleave(KS * ST, 0), t_], 0)
    v0, d0, st0 = rep(D["v0"][sl]), rep(D["d0"][sl]), rep(D["st"][sl])
    p = PS.roll(A, v0, d0, lim)
    x, y, psi = plan_world(p, st0)
    tr = t1.expand(x.shape[0])
    h1 = hit_any(x, y, psi, tr)
    # continuation: from the plan's end, full brake + 9 constant turns, 1 s
    stE = st0.clone(); stE[:, 0] = x[:, -1]; stE[:, 1] = y[:, -1]; stE[:, dyn.IYAW] = psi[:, -1]
    n_all = x.shape[0]
    Ac = torch.cat([turn_c[:, None].expand(9, N), torch.full((9, N), -1.0, device=dev)], 1)
    Ac = Ac[None].expand(n_all, 9, 2 * N).reshape(-1, 2 * N)
    pc = PS.roll(Ac, p["v"][:, -1].repeat_interleave(9), p["delta"][:, -1].repeat_interleave(9), lim)
    xc, yc, psic = plan_world(pc, stE.repeat_interleave(9, 0))
    h2 = hit_any(xc, yc, psic, tr.repeat_interleave(9)).view(n_all, 9).all(1)   # every continuation hits
    safe = ~h1 & ~h2
    prog = -pr.goal.remaining(torch.stack([x[:, -1], y[:, -1]], -1)[:, None], tr[:, None])[:, 0]
    nf = nb * KS * ST
    safe_f = safe[:nf].view(nb, KS, ST); prog_f = prog[:nf].view(nb, KS, ST)
    pol_safe, pol_prog, pol_hit = safe[nf:], prog[nf:], h1[nf:]
    # policy's brake onset: the first step whose end speed is below the step before (from v0)
    pv = torch.cat([D["v0"][sl][:, None], p["pv"][nf:]], 1)
    dec = (pv[:, 1:] < pv[:, :-1] - 0.02)
    k_pol = torch.where(dec.any(1), dec.float().argmax(1), torch.full((nb,), N, device=dev))
    any_safe_k = safe_f.any(2)                                           # (nb, KS)
    k_safe = torch.where(any_safe_k.any(1), (any_safe_k * ks[None]).amax(1), torch.full((nb,), -1, device=dev))
    best = torch.where(safe_f, prog_f, torch.full_like(prog_f, -1e9)).view(nb, -1)
    bp, bi = best.max(1)
    res["k_pol"].append(k_pol); res["k_safe"].append(k_safe); res["dprog"].append(bp - pol_prog)
    res["pol_safe"].append(pol_safe); res["pol_hit"].append(pol_hit); res["best_k"].append(bi // ST)
R = {k: torch.cat(v).float().cpu().numpy() for k, v in res.items()}
s = D["s"].cpu().numpy()
ok = R["k_safe"] >= 0
print(f"policy plan: 1 s 안에 벽에 닿음(plan 충돌 감점) {R['pol_hit'].mean() * 100:.1f} %, 안전(2 s 기준) {R['pol_safe'].mean() * 100:.0f} %", flush=True)
print(f"안전한 plan이 하나라도 있는 결정 {ok.mean() * 100:.0f} %")
for lo, hi in ((S_LO, 11.0), (11.0, 12.5), (12.5, S_HI)):
    m = ok & (s >= lo) & (s < hi)
    if m.sum() < 5:
        continue
    dk = R["k_safe"][m] - R["k_pol"][m]
    print(f"  {lo:.1f}-{hi:.1f} m (결정 {m.sum()}): policy가 밟기 시작하는 단계 중앙 {np.median(R['k_pol'][m]):.0f} (x 0.1 s), "
          f"안전하게 가장 늦게 밟을 수 있는 단계 중앙 {np.median(R['k_safe'][m]):.0f} -> 더 늦출 수 있는 여유 중앙 {np.median(dk) * 100:.0f} ms, "
          f"0.2 s 이상 여유 {np.mean(dk >= 2) * 100:.0f} % | 1 s 뒤 진행: 가장 많이 가는 안전 plan이 policy보다 중앙 {np.median(R['dprog'][m]) * 100:+.0f} cm "
          f"(그 plan의 k 중앙 {np.median(R['best_k'][m]):.0f})", flush=True)

# ---- 2) exploration noise on the first speed output
mu, ls = D["mean"][:, N], D["logstd"][:, N]
eps = torch.randn(mu.shape[0], 64, device=dev, generator=g)
noisy = torch.tanh(mu[:, None] + ls.exp()[:, None] * eps)
det = torch.tanh(mu)
for lo, hi in ((S_LO, 11.0), (11.0, 12.5), (12.5, S_HI)):
    m = torch.from_numpy((s >= lo) & (s < hi)).to(dev)
    if int(m.sum()) < 5:
        continue
    print(f"  첫 단계 속도 출력(-1 최대 제동 .. +1 최대 가속) {lo:.1f}-{hi:.1f} m: 잡음 없이 중앙 {float(det[m].median()):+.2f}, "
          f"학습 때처럼 잡음 넣은 평균의 중앙 {float(noisy[m].mean(1).median()):+.2f}, 잡음 표본이 +0.9 넘는 비율 {float((noisy[m] > 0.9).float().mean()) * 100:.0f} % "
          f"/ -0.9 아래 {float((noisy[m] < -0.9).float().mean()) * 100:.0f} % (표준편차 {float(ls.exp()[0]):.3f})", flush=True)
