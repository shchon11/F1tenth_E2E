"""상자 뒤 출발 episode에서 처음 2 s 안에 그 상자에 세게 부딪히나 (11-06 12:10). 상자 world 이어서 학습(srp25box)이 시작점보다 나빠짐
(배치 8개 obstacle 세게 1 -> 2~7번) -> 의심: --spawn-behind-prop가 앞쪽 여유 0.15-1.15 m로 너무 가까워 피할 수 없는 충돌이 벌점만 줌.
체크포인트를 그 학습 world(args 그대로, spawn_behind_prop 1.0)에서 학습 때처럼 탐색 noise로 몰고, 상자 뒤에서 출발한 차마다 출발할 때 가장 가까운
상자 중심까지 거리와 처음 2 s 안 세게 obstacle 부딪힘 여부를 거리 구간별로. 인자: 체크포인트 대수 시드"""
import os, sys, math, argparse
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

path, n_env, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, seed, "cuda"
a.prop_fraction, a.removal_p, a.spawn_behind_prop = 1.0, 0.0, 1.0
env = SR.build_env(a, dev)
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=seed)
lid = env.learner_ids; B = int(lid.numel()); po = env.procedural
st = env.sim.state[lid]
live = po.p_sid[lid] >= 0
d0 = torch.where(live, (po.p_poses[lid][..., :2] - st[:, None, :2]).norm(dim=-1), torch.full(live.shape, 99.0, device=dev)).amin(1)
tp = SR.TeacherPolicy(env, ck, dev)
act = torch.zeros(B, tp.A, device=dev); eps = torch.zeros_like(act)
act_full = torch.zeros(env.B, env.act_dim, device=dev)
i_hp = CC.index("hard_prop")
cf0 = env.sim.contact_frame()[lid, i_hp].float().clone()
R = int(a.repeat)
for t in range(int(2.0 / env.sim.control_dt) // R):
    tp.o = tp.sobs(act)
    dist = tp.net.dist(tp.nrm(tp.o))
    eps = torch.randn_like(eps)
    act = SR.clamp_any(dist.mean + dist.stddev * eps, tp.ar)
    act_full[lid] = act
    for _ in range(R):
        env.step(act_full)
hit = (env.sim.contact_frame()[lid, i_hp].float() - cf0) > 0
name = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
print(f"{name}: 차 {B}대 중 상자 뒤 출발 {getattr(env, 'behind_prop_spawns', 0)}대", flush=True)
for lo, hi in ((0.6, 0.9), (0.9, 1.2), (1.2, 1.7), (1.7, 99)):
    m = (d0 >= lo) & (d0 < hi)
    if int(m.sum()):
        print(f"  출발 때 가장 가까운 상자 중심까지 {lo}~{hi if hi < 99 else '먼'} m: {int(m.sum())}대, 처음 2 s 안에 세게 부딪힘 {int((hit & m).sum())}대 "
              f"({float((hit & m).sum()) / float(m.sum()) * 100:.0f} %)", flush=True)
