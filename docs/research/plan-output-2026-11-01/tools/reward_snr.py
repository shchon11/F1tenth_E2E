"""빈 트랙에서 속도 차이가 학습 reward에 얼마나 드러나나 (11-04 22:55, 사용자: "보상설계가 잘못된거 아냐 그럼..?"). 학습 env(네 경우 world 설정에서
obstacle·상대 차만 뺀 빈 트랙)에서 checkpoint마다 같은 탐색 noise(--noise-from 체크포인트의 std)로 달려, 10 s 창(200결정)마다 학습 reward 합
(progress - steer-rate 비용 - 가격 x 벽 hard 시작)을 모음. 두 policy의 창 평균 차이를 창 사이 표준편차와 비교(효과 크기 d), 그리고 PPO가 한 update
(1024 환경 x 64결정 중 빈 트랙 25 %)에서 볼 수 있는 표본으로 그 차이가 몇 표준오차인지. 인자: 맵id 환경수 결정수 noise체크포인트 체크포인트들..."""
import os, sys, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR

mp_args, n_env, n_dec, noise_ck = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
runs = sys.argv[5:]
dev = torch.device("cuda")
ckn = torch.load(noise_ck, map_location=dev, weights_only=False)
std = torch.maximum(ckn["model"]["log_std"].exp(), ckn["model"].get("std_floor", torch.zeros_like(ckn["model"]["log_std"])))
ar = dict(json_ar := __import__("json").load(open(mp_args)))
ar.update(envs=n_env, prop_fraction=0.0, opp_present=0.0, removal_p=0.0, level_replay=0.0)
a = argparse.Namespace(**ar)
res = {}
for run in runs:
    torch.manual_seed(0)
    env = SR.build_env(a, dev)
    from f1sim.learn.graph_runtime import prepare_graph_runtime
    prepare_graph_runtime(env, log=lambda _t: None)
    env.reset(seed=4321)
    ck = torch.load(run, map_location=dev, weights_only=False)
    tp = SR.TeacherPolicy(env, ck, dev)
    lid = env.learner_ids; B = int(lid.numel()); A = tp.A; R = int(a.repeat)
    act_full = torch.zeros(env.B, env.act_dim, device=dev)
    last = torch.zeros(B, A, device=dev)
    w_col = abs(env.ecfg.reward_collision)
    rew = torch.zeros(n_dec, B, device=dev)
    with torch.no_grad():
        for t in range(n_dec):
            mu = tp(last)
            exe = SR.clamp_any(mu + std.to(dev) * torch.randn_like(mu), tp.ar)
            act_full[lid] = exe
            for k in range(R):
                _, r_, te, trc, info = env.step(act_full)
                comp = info["reward_components"]
                col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
                onw = (-col / max(w_col, 1e-9)).clamp_min(0)
                rew[t] += r_[lid].float() - col - car - float(a.price) * onw
            last = exe
    W = 200
    r = rew[: (n_dec // W) * W].reshape(n_dec // W, W, B).sum(1).flatten().cpu().numpy()
    per_dec = rew.flatten().cpu().numpy()
    res[run] = (r, per_dec)
    print(f"{os.path.basename(os.path.dirname(run))}/{os.path.basename(run)}: 10 s 창 {r.size}개, reward 합 평균 {r.mean():.2f} ± {r.std():.2f} (표준편차) | "
          f"결정당 평균 {per_dec.mean():.4f}, 표준편차 {per_dec.std():.4f}", flush=True)
    del env, tp
    torch.cuda.empty_cache()
if len(runs) >= 2:
    (r1, d1), (r2, d2) = res[runs[0]], res[runs[1]]
    diff = r1.mean() - r2.mean(); sd = np.sqrt(0.5 * (r1.var() + r2.var()))
    n_upd = 1024 // 2 * 64 * 0.25                   # learner rows x decisions x empty share: the decisions of one update
    se_dec = np.sqrt(0.5 * (d1.var() + d2.var()) * 2 / n_upd)
    print(f"차이(첫 - 둘째): 10 s 창 {diff:.2f} = 창 표준편차의 {diff / sd:.2f}배 | 결정당 {d1.mean() - d2.mean():.4f}, "
          f"한 update의 빈 트랙 결정 {int(n_upd)}개로 보면 {(d1.mean() - d2.mean()) / se_dec:.1f} 표준오차", flush=True)
