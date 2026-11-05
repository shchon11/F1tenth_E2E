"""학습 world에서 상자에 걸려 서 있는 시간 (11-05 22:40). 장면: 고정 상자(--prop-movable 0, 16:42부터)에 앞이 닿으면 후진 action이
없고(v_min 0) 멈춘 차는 MAP로 돌지 못해 그 episode 끝까지 갇힘(srp26owp u550 layout 2: 35 s, srp25owp u350 layout 6: 25 s 동안 6번).
체크포인트를 그 학습과 똑같은 world(args 그대로)에서 학습 때처럼 탐색 noise로 몰고, 결정마다 '상자 옆에 서 있음'(속도 < 0.3 m/s, 가장
가까운 상자 중심 0.8 m 안)을 셈. 미리 정한 판정: obstacle이 깔린 시간 중 그 몫이 10 % 이상이면 갇힘이 학습을 크게 갉아먹는 것 -> 갇히면
episode를 끝내도록 고침. 인자: 체크포인트 대수 초 시드"""
import os, sys, math, argparse
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

path, n_env, secs, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, seed, "cuda"
env = SR.build_env(a, dev)
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=seed)
lid = env.learner_ids; B = int(lid.numel())
remover = SR.PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=seed) \
    if float(getattr(a, "removal_p", 0) or 0) > 0 else None
tp = SR.TeacherPolicy(env, ck, dev)
act = torch.zeros(B, tp.A, device=dev); eps = torch.zeros_like(act)
act_full = torch.zeros(env.B, env.act_dim, device=dev)
i_hp = CC.index("hard_prop")
cf_prev = env.sim.contact_frame()[lid].float()
R = int(a.repeat); dt = env.sim.control_dt
po = env.procedural
t_props = t_stuck = t_all = 0.0
run = torch.zeros(B, device=dev)                  # how long each car has stood beside a prop [s]
t_long = 0.0                                      # time inside stands longer than 2 s
hits = hits_stuck = 0
for t in range(int(secs / dt) // R):
    tp.o = tp.sobs(act)
    dist = tp.net.dist(tp.nrm(tp.o))
    al = float(a.noise_corr)
    eps = al * eps + math.sqrt(1.0 - al * al) * torch.randn_like(eps)
    act = SR.clamp_any(dist.mean + dist.stddev * eps, tp.ar)
    act_full[lid] = act
    for _ in range(R):
        env.step(act_full)
        if remover is not None:
            remover.step()
        st = env.sim.state[lid]
        live = po.p_sid[lid] >= 0
        has = live.any(1)
        dp = torch.where(live, (po.p_poses[lid][..., :2] - st[:, None, :2]).norm(dim=-1),
                         torch.full(live.shape, 99.0, device=dev)).amin(1)
        stuck = has & (st[:, 3].abs() < 0.3) & (dp < 0.8)
        run = torch.where(stuck, run + dt, torch.zeros_like(run))
        t_all += B * dt
        t_props += float(has.sum()) * dt
        t_stuck += float(stuck.sum()) * dt
        t_long += float((run > 2.0).sum()) * dt
        cf = env.sim.contact_frame()[lid].float()
        h = (cf[:, i_hp] - cf_prev[:, i_hp]) > 0
        cf_prev = cf
        hits += int(h.sum()); hits_stuck += int((h & (st[:, 3].abs() < 0.5)).sum())
name = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
print(f"{name}: obstacle 깔린 시간 {t_props / t_all * 100:.0f} % | 그중 상자 옆에 서 있음 {t_stuck / max(t_props, 1e-9) * 100:.1f} % "
      f"(2 s 넘게 이어진 것 {t_long / max(t_props, 1e-9) * 100:.1f} %) | 상자에 세게 부딪힘 {hits}번, 그중 거의 선 채(0.5 m/s 미만) {hits_stuck}번",
      flush=True)
