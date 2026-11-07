"""재설계 2단계 확인 (11-07): plan대로 정확히 실행(`env.enable_ideal`)하면 차가 plan의 자리에 있나, 접촉은 시뮬 그대로 처리되나.
학습 인자로 env(상대·상자 없음, 차 N대)를 만들고, 0.1 s마다 지금 상태에서 무작위 출력(크기 SCALE)으로 plan(plan_steps)을 만들어
그 4개 제어 step의 명령을 그대로 넘김. 접촉이 없던 결정에서 0.025·0.05·0.1 s 뒤 차 위치·방향·속도와 plan의 차이,
접촉이 난 결정에서 접촉 전후 속도(부딪힌 속도별)를 셈. 인자: 체크포인트(인자만) 차수 초 [SCALE]"""
import os, sys, math, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

torch.set_grad_enabled(False)
path, n_env, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
dev = torch.device("cuda")
ck = torch.load(path, map_location="cpu", weights_only=False)
a = argparse.Namespace(**ck["args"]); a.envs, a.seed, a.device = n_env, 5, "cuda"
a.prop_fraction, a.removal_p, a.warps, a.opp_present, a.spawn_behind_prop, a.stage = 0.0, 0.0, 0, 0.0, 0.0, "props"
a.episode_s = 3600.0
env = SR.build_env(a, dev)
env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=5)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car")]
g = torch.Generator(device=dev); g.manual_seed(1)
err = {1: [], 2: [], 4: []}; yerr, verr = [], []
hits = []                                           # (impact speed, speed before, speed right after, speed at the decision's end)
n_dec = int(secs / PS.DT)
for kd in range(n_dec):
    st = env.sim.state[lid].clone()
    v0, d0 = st[:, dyn.IVX], st[:, dyn.ISTEER]
    act = (torch.rand(B, PS.ACT_DIM, device=dev, generator=g) * 2 - 1) * scale
    act[:, PS.N_STEP:] += 0.3                      # lean forward so the cars keep moving
    p = PS.roll(act, v0, d0, lim)
    env.sim.ideal_vcap[lid] = 1e3
    cf0 = env.sim.contact_frame()[lid].float()[:, i_on].sum(1).clone()
    c, s_ = torch.cos(st[:, dyn.IYAW]), torch.sin(st[:, dyn.IYAW])
    touched = torch.zeros(B, dtype=torch.bool, device=dev)
    for j in range(PS.SUB):
        env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]
        env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        v_before = env.sim.state[lid, dyn.IVX].clone()
        _, _, te, trc, _ = env.step(act0)
        cf = env.sim.contact_frame()[lid].float()[:, i_on].sum(1)
        new = (cf > cf0) & ~touched
        if bool(new.any()):
            imp = env.sim.contact_frame()[lid].float()
            for i in torch.nonzero(new)[:, 0].tolist():
                hits.append([float(env.last_result.wall_impact[lid[i]]) if env.last_result.wall_impact is not None else float("nan"),
                             float(v_before[i]), float(env.sim.state[lid[i], dyn.IVX])])
        touched |= (cf > cf0) | env.sim.kind_prev[lid].any(1)          # a new contact, or one still going on
        if (j + 1) in err:
            px = st[:, 0] + p["x"][:, j + 1] * c - p["y"][:, j + 1] * s_
            py = st[:, 1] + p["x"][:, j + 1] * s_ + p["y"][:, j + 1] * c
            ok = ~touched & ~(te | trc)[lid]
            d = torch.hypot(env.sim.state[lid, 0] - px, env.sim.state[lid, 1] - py)
            err[j + 1].append(d[ok])
            if j + 1 == 4:
                dy = torch.remainder(env.sim.state[lid, dyn.IYAW] - (st[:, dyn.IYAW] + p["psi"][:, j + 1]) + math.pi, 2 * math.pi) - math.pi
                yerr.append(dy[ok].abs()); verr.append((env.sim.state[lid, dyn.IVX] - p["v"][:, j + 1])[ok].abs())
    for h in hits[-int(touched.sum()):] if bool(touched.any()) else []:
        pass
print(f"차 {B}대 x {secs:.0f} s, 결정 {n_dec}개, 무작위 출력 크기 {scale}", flush=True)
for j, L_ in err.items():
    e = torch.cat(L_) * 100
    print(f"  접촉 없던 결정, {j * 0.025:.3f} s 뒤 위치 차이: 중앙 {float(e.median()):.3f} cm, p99 {float(e.quantile(0.99)):.3f} cm, 최대 {float(e.max()):.3f} cm (n {e.numel()})", flush=True)
y_ = torch.cat(yerr); v_ = torch.cat(verr)
print(f"  0.1 s 뒤 방향 차이 최대 {float(y_.max()) * 180 / math.pi:.3f}°, 속도 차이 최대 {float(v_.max()):.4f} m/s", flush=True)
if hits:
    H = np.array(hits)
    for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 10.0)):
        m = (H[:, 0] >= lo) & (H[:, 0] < hi)
        if m.any():
            print(f"  접촉 (부딪힌 속도 {lo}~{hi} m/s) {int(m.sum())}번: 직전 속도 중앙 {np.median(H[m, 1]):.2f} -> 직후 {np.median(H[m, 2]):.2f} m/s", flush=True)
