"""상자 앞에 멈춘 차는 학습 world에서 어떻게 풀려나나 (11-06 15:45, 사용자: "보상문제는 없으려나 왜자꾸 쳐박고 가만히있을까"). 학습 인자 그대로
(상자 수 1개~전부, 절반은 10~30 s에 치움, 상대 차, 상자 뒤 출발) 학습 때처럼 탐색 흔들림으로 몰고, '상자 앞에 멈춤'(속도 0.3 m/s 아래 1 s 넘게,
차 앞쪽 1.2 m 안에 상자)마다 어떻게 끝났는지 셈: (가) 상자가 치워져서, (나) 스스로 빠져나감(1 m/s 넘게 0.5 s), (다) episode 끝까지 그대로.
멈춰 있는 동안 policy가 후진을 고를 확률(첫 속도 값이 -1 아래로 뽑힐 확률)과 그 평균값도. 인자: 체크포인트 차수 초"""
import os, sys, math, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime

path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
dev = torch.device("cuda")
torch.set_grad_enabled(False)
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, 31, "cuda"
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=31)
remover = SR.PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=31)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel()); po = env.procedural
act = torch.zeros(B, tp.A, device=dev); eps = torch.zeros_like(act); act_full = torch.zeros(env.B, env.act_dim, device=dev)
R = int(a.repeat); dt = env.sim.control_dt; dv = SR.PLAN_KAPPA_DIMS
slow_t = torch.zeros(B, device=dev); fast_t = torch.zeros(B, device=dev)
inspell = torch.zeros(B, dtype=torch.bool, device=dev); spell_t = torch.zeros(B, device=dev)
ends = {"치워짐": [], "빠져나감": [], "episode 끝": []}
p_rev, mu_v = [], []
t_box = 0.0; t_stuck = 0.0


def box_ahead():
    st = env.sim.state[lid]
    live = po.p_sid[lid] >= 0
    d = po.p_poses[lid][..., :2] - st[:, None, :2]
    c, s_ = torch.cos(st[:, 2])[:, None], torch.sin(st[:, 2])[:, None]
    x, y = d[..., 0] * c + d[..., 1] * s_, -d[..., 0] * s_ + d[..., 1] * c
    return (live & (x > -0.1) & (x.hypot(y) < 1.2)).any(1), live.any(1)


for t in range(int(secs / dt) // R):
    tp.o = tp.sobs(act)
    d_ = tp.net.dist(tp.nrm(tp.o))
    al = float(a.noise_corr)
    eps = al * eps + math.sqrt(1.0 - al * al) * torch.randn_like(eps)
    act = SR.clamp_any(d_.mean + d_.stddev * eps, tp.ar)
    if bool(inspell.any()):
        z = (-1.0 - d_.mean[inspell, dv]) / d_.stddev[inspell, dv]
        p_rev.append((0.5 * (1 + torch.erf(z / math.sqrt(2)))).cpu().numpy()); mu_v.append(d_.mean[inspell, dv].cpu().numpy())
    act_full[lid] = act
    for _ in range(R):
        _, _, te, trc, _ = env.step(act_full)
        remover.step()
        done = (te | trc)[lid]
        ahead, anyb = box_ahead()
        v = env.sim.state[lid, 3].abs()
        slow_t = torch.where(v < 0.3, slow_t + dt, torch.zeros_like(slow_t))
        fast_t = torch.where(v > 1.0, fast_t + dt, torch.zeros_like(fast_t))
        t_box += float(anyb.sum()) * dt; t_stuck += float(inspell.sum()) * dt
        start = ~inspell & ahead & (slow_t > 1.0)
        inspell = inspell | start; spell_t = torch.where(start, slow_t, spell_t)
        spell_t = torch.where(inspell & ~start, spell_t + dt, spell_t)
        for lab, m in (("episode 끝", inspell & done), ("치워짐", inspell & ~done & ~anyb), ("빠져나감", inspell & ~done & anyb & (fast_t > 0.5))):
            if bool(m.any()):
                ends[lab] += spell_t[m].cpu().tolist()
                inspell = inspell & ~m
        if bool(done.any()):
            slow_t = torch.where(done, torch.zeros_like(slow_t), slow_t); fast_t = torch.where(done, torch.zeros_like(fast_t), fast_t)
n = sum(len(v) for v in ends.values())
name = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
print(f"{name}, 학습 world 그대로 차 {B}대 x {secs:.0f} s (상자 깔린 시간 {t_box:.0f} s, 그중 상자 앞에 멈춰 있던 시간 {t_stuck / max(t_box, 1e-9) * 100:.0f} %): "
      f"상자 앞에 멈춤 {n}번", flush=True)
for lab, v in ends.items():
    print(f"  {lab}: {len(v)}번 ({len(v) / max(n, 1) * 100:.0f} %), 멈춰 있던 시간 중앙값 {np.median(v) if v else 0:.1f} s", flush=True)
if p_rev:
    p = np.concatenate(p_rev); m = np.concatenate(mu_v)
    print(f"  멈춰 있는 동안 결정 {len(p)}개: 후진을 고를 확률 평균 {p.mean() * 100:.1f} % (10 % 넘는 결정 {np.mean(p > 0.1) * 100:.0f} %), "
          f"첫 속도 값 평균 {m.mean():.2f} (-1 = 정지, 그 아래 = 후진, 최저 -1.5)", flush=True)
