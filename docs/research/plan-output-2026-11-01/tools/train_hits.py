"""학습 world에서 세게 부딪히는 건 어떤 상황인가 (11-07 01:40). 시간별 waypoint 처음부터 학습(srp25tw2)이 충돌 벌점이 오르자 서고 느려짐
(u150·u200 평가에서 빈 트랙도 거의 서 있음), 학습 중 세게 부딪힘이 km당 33~46으로 8개 형식 처음부터 학습(9~19)보다 3~4배. 학습 인자
그대로의 world에서 학습 때 흔들림(느릴 때 하한, 이어지는 흔들림)으로 몰며, 세게 부딪힐 때마다 직전 속도(후진 / 1.5 m/s 아래 전진 /
그 위)와 무엇에 부딪혔는지(벽 / 상자), 상자 뒤에서 출발한 episode인지를 셈. 인자: 체크포인트 차수 초"""
import os, sys, math, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

torch.set_grad_enabled(False)
path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, 71, "cuda"
env = SR.build_env(a, dev); prepare_graph_runtime(env, log=lambda _t: None); env.reset(seed=71)
remover = SR.PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=71)
se = getattr(a, "slow_explore", None)
os.environ["F1SIM_POLICY_NOISE"] = "1"
os.environ["F1SIM_POLICY_NOISE_CORR"] = str(float(a.noise_corr))
if se:
    os.environ["F1SIM_SLOW_EXPLORE"] = ",".join(str(float(x)) for x in se)
if len(sys.argv) > 4:
    os.environ["F1SIM_CORR_SLOW"] = sys.argv[4]          # hold the noise only below this speed (the fix under test)
tp = SR.TeacherPolicy(env, ck, dev)
lid = env.learner_ids; B = int(lid.numel()); R = int(a.repeat); dt = env.sim.control_dt
iw, ip = CC.index("hard_wall"), CC.index("hard_prop")
cf_prev = env.sim.contact_frame()[lid].float()
last = torch.zeros(B, tp.A, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
cnt = {}
dist_m = 0.0; t_stop = 0.0
for k in range(int(secs / dt)):
    if k % R == 0:
        last = tp(last); act_full[lid] = last
    v_pre = env.sim.state[lid, 3].clone()
    _, _, te, trc, info = env.step(act_full)
    remover.step()
    dist_m += float(info["progress"][lid].clamp_min(0).sum())
    t_stop += float((env.sim.state[lid, 3].abs() < 0.3).float().sum()) * dt
    cf = env.sim.contact_frame()[lid].float()
    for col, what in ((iw, "벽"), (ip, "상자")):
        h = (cf[:, col] - cf_prev[:, col]) > 0
        if bool(h.any()):
            vv = v_pre[h]
            for lab, m in (("후진 중", vv < -0.1), ("느린 전진(<1.5)", (vv >= -0.1) & (vv < 1.5)), ("빠른 전진", vv >= 1.5)):
                cnt[(what, lab)] = cnt.get((what, lab), 0) + int(m.sum())
    cf_prev = cf
    d_ = (te | trc)[lid]
    last = torch.where(d_[:, None], torch.zeros_like(last), last)
km = dist_m / 1000
tot = sum(cnt.values())
print(f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} 학습 world, 학습 흔들림{' (이어지는 흔들림은 ' + sys.argv[4] + ' m/s 아래에서만)' if len(sys.argv) > 4 else ''}, 차 {B}대 x {secs:.0f} s: 앞으로 간 거리 {km:.2f} km, "
      f"세게 부딪힘 {tot}번 (km당 {tot / max(km, 1e-9):.1f}), 0.3 m/s 아래로 서 있던 시간 {t_stop / (B * secs) * 100:.0f} %", flush=True)
for (what, lab), n in sorted(cnt.items(), key=lambda kv: -kv[1]):
    print(f"  {what}·{lab}: {n}번 ({n / max(tot, 1) * 100:.0f} %)", flush=True)
