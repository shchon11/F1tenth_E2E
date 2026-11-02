"""상자 단계(--vis-actor) 충돌 진단 (11-03): Korea 2025 srp25h3는 업데이트 85에도 세게 13/km(Korea 2026 srp26h3는 2.5/km).
가설: 좁은 머리핀 트랙에서 상자가 모퉁이를 돌아야 보이는데(학생과 같은 시야), 보인 뒤 피할 시간이 모자람. 학습과 같은 세계에서 평균 행동으로
몰며, 세게 부딪힐 때마다: 벽/상자, 속도, 부딪힌 상자가 처음 보인 뒤 몇 초 만인지(안 보인 채 부딪힘 = -1), 그 순간 거리. 상자 충돌을
"보인 지 0.5 s 안", "0.5~1.5 s", "1.5 s 넘게", "안 보인 채"로 나눔. 인자: 체크포인트 환경수 초 시드"""
import os, sys, math, argparse
import numpy as np
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
remover = SR.PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=seed)
tp = SR.TeacherPolicy(env, ck, dev)
po = env.procedural
C = po.C
first_vis = torch.full((B, C), -1.0, device=dev)
act = torch.zeros(B, tp.A, device=dev); act_full = torch.zeros(env.B, env.act_dim, device=dev)
iw, ip = CC.index("hard_wall"), CC.index("hard_prop")
cf_prev = env.sim.contact_frame()[lid].float()
R = int(a.repeat); t_now = 0.0; prog = 0.0
rows = []
ep_prev = env.ep_step[lid].clone()
with torch.no_grad():
    for t in range(int(secs / env.sim.control_dt) // R):
        act = tp(act); act_full[lid] = act
        for _ in range(R):
            _, r_, te, trc, info = env.step(act_full)
            remover.step()
            t_now += env.sim.control_dt
            prog += float(info["progress"][lid].float().clamp_min(0).sum())
            ep = env.ep_step[lid]
            new = ep < ep_prev; ep_prev = ep.clone()
            first_vis[new] = -1.0
            vis = SR.visible_props(env, lid)
            if vis is not None:
                first_vis = torch.where(vis & (first_vis < 0), torch.full_like(first_vis, t_now), first_vis)
            cf = env.sim.contact_frame()[lid].float()
            hw = (cf[:, iw] - cf_prev[:, iw]) > 0; hp = (cf[:, ip] - cf_prev[:, ip]) > 0
            cf_prev = cf
            if bool((hw | hp).any()):
                st = env.sim.state[lid]
                for i in torch.nonzero(hw | hp).flatten().tolist():
                    v = float(st[i, 3])
                    if bool(hp[i]):
                        live = po.p_sid[lid[i]] >= 0
                        d = torch.where(live, (po.p_poses[lid[i], :, :2] - st[i, None, :2]).norm(dim=-1), torch.full((C,), 99.0, device=dev))
                        c = int(d.argmin())
                        fv = float(first_vis[i, c])
                        rows.append(("상자", v, (t_now - fv) if fv >= 0 else -1.0))
                    else:
                        rows.append(("벽", v, float("nan")))
km = prog / 1000.0
P_ = [r for r in rows if r[0] == "상자"]; W_ = [r for r in rows if r[0] == "벽"]
print(f"== {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} 평균 행동, {B}대 x {secs:.0f} s = {km:.1f} km | 세게 벽 {len(W_) / km:.2f}/km, 상자 {len(P_) / km:.2f}/km")
if P_:
    dt = np.array([r[2] for r in P_]); v = np.array([r[1] for r in P_])
    for name, m in (("안 보인 채", dt < 0), ("보인 지 0.5 s 안", (dt >= 0) & (dt < 0.5)), ("0.5~1.5 s", (dt >= 0.5) & (dt < 1.5)), ("1.5 s 넘게", dt >= 1.5)):
        print(f"  상자 충돌 {name}: {m.sum()}건 ({m.mean() * 100:.0f} %), 충돌 속도 중앙 {np.median(v[m]) if m.any() else float('nan'):.2f} m/s")
if W_:
    print(f"  벽 충돌 속도 중앙 {np.median([r[1] for r in W_]):.2f} m/s")
