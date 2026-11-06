"""상자 바로 뒤에서 출발한 차가 상자를 지나가나 (11-06 21:35, 사용자: "가만히 왔다갔다하는건 고질병이네 왜그러지?"). 영상의 멈춤은 거의 다
'상자에 세게 부딪힘 -> 0.5~0.7 m 후진 -> 그 자리에 섬'. 학습은 episode의 30 %를 상자 중심 0.65~1.6 m 뒤(차선 방향 +-0.1 rad, 0~1 m/s)에서
출발시킴 -> 그 출발 상태 자체를 policy가 풀 줄 아는지, 부딪힌 뒤 상태만 못 푸는지 가름. 학습 인자로 env를 만들되 상자 뒤 출발 100 %, 상자 전부,
안 치움, 상대 없음, 차 1대; 평균 행동(또는 F1SIM_POLICY_NOISE 등 env로 학습 흔들림)으로 10 s 달리고, 출발 때 가장 가까운 상자 중심 거리별로
'지나감'(트랙 따라 2.5 m 넘게 전진) / '서 있음'(0.5 m 미만 전진) / 세게 부딪힘 몫. 인자: 체크포인트 차수"""
import os, sys, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

torch.set_grad_enabled(False)
path, n_env = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, 41, "cuda"
a.prop_count, a.prop_fraction, a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop, a.stage = "fixed", 1.0, 0.0, 0, 0.0, 1.0, "props"
a.episode_s = 3600.0
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=41)
lid = env.learner_ids; B = int(lid.numel()); po = env.procedural; tid = env.sim.tid[lid]
st = env.sim.state[lid]
live = po.p_sid[lid] >= 0
d0 = torch.where(live, (po.p_poses[lid][..., :2] - st[:, None, :2]).norm(dim=-1), torch.full(live.shape, 99.0, device=dev)).amin(1)
s0, _, _ = env.sim.track.project(st[:, :2], tid)
L = env.sim.track.length[tid]
tp = SR.TeacherPolicy(env, ck, dev)
last = torch.zeros(B, tp.A, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
i_hp = CC.index("hard_prop"); cf0 = env.sim.contact_frame()[lid, i_hp].float().clone()
prog = torch.zeros(B, device=dev); s_prev = s0.clone(); dt = env.sim.control_dt
cf_prev = cf0.clone(); hit_v = []                  # the car's speed just before each hard prop onset
for k in range(int(10.0 / dt)):
    if k % int(a.repeat) == 0:
        last = tp(last)
        act_full[lid] = last
    v_pre = env.sim.state[lid, 3].abs().clone()
    env.step(act_full)
    cf_k = env.sim.contact_frame()[lid, i_hp].float()
    new_hit = (cf_k - cf_prev) > 0
    if bool(new_hit.any()):
        hit_v += v_pre[new_hit].cpu().tolist()
    cf_prev = cf_k
    s_now, _, _ = env.sim.track.project(env.sim.state[lid, :2], tid)
    ds = torch.remainder(s_now - s_prev + 0.5 * L, L) - 0.5 * L
    prog += ds; s_prev = s_now
hit = (env.sim.contact_frame()[lid, i_hp].float() - cf0) > 0
nb = int(getattr(env, "behind_prop_spawns", 0))
mode = "평균 행동" if not os.environ.get("F1SIM_POLICY_NOISE") else "학습 흔들림"
print(f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} {mode}: 차 {B}대 중 상자 뒤 출발 {nb}대, 10 s 동안", flush=True)
for lo, hi in ((0.0, 0.8), (0.8, 1.1), (1.1, 1.4), (1.4, 99.0)):
    m = (d0 >= lo) & (d0 < hi)
    n = int(m.sum())
    if n:
        print(f"  출발 때 가장 가까운 상자 중심 {lo:.1f}~{hi if hi < 99 else '먼'} m: {n}대 | 지나감 {float(((prog > 2.5) & m).sum()) / n * 100:.0f} % | "
              f"서 있음 {float(((prog < 0.5) & m).sum()) / n * 100:.0f} % | 세게 부딪힘 {float((hit & m).sum()) / n * 100:.0f} %", flush=True)
if hit_v:
    hv = np.array(hit_v)
    print(f"  세게 부딪힌 {len(hv)}번의 직전 속도: 1.0~1.5 m/s {np.mean(hv < 1.5) * 100:.0f} %, 1.5~2.5 m/s {np.mean((hv >= 1.5) & (hv < 2.5)) * 100:.0f} %, "
          f"2.5 m/s 넘게 {np.mean(hv >= 2.5) * 100:.0f} % (중앙값 {np.median(hv):.2f} m/s)", flush=True)
