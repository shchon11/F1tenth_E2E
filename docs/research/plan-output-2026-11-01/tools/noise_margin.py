"""상자 정책이 상자가 없어도 10 % 느린 원인 시험 (11-03, 사용자: "GT가 들어가는데 학습이 제대로 안 된다는 건 치명적 결함").
가설: PPO는 잡음 섞인 운전의 점수를 최적화한다. 탐색 잡음 하한(--std-floor 0.05 0.2, 상관 --noise-corr 0.9 = 약 0.5 s씩 이어짐)
때문에 상자 빽빽한 세계에서 빨리 가면 잡음에 박히므로, 잡음 없는 평균 행동도 느려진다.
판별(학습 없이): 학습과 같은 세계(체크포인트 args 그대로: 상자 비율·밀도·치우기·가격·문턱)에서 상자 정책(혼자 섞기 없음)의 평균 행동에
속도 4칸(경로점 속도)만 +d를 더한 변형들을 (가) 잡음 없이 (나) 학습 때와 같은 잡음(같은 표준편차·상관)으로 몬다. 가설이 맞으면:
(가)에서는 d > 0가 J(진행 - 80 x 값 매긴 충돌)를 올리고, (나)에서는 d > 0가 J를 내린다.
인자: 체크포인트 환경수 초 시드 [d 목록, 쉼표]"""
import os, sys, math, argparse
import numpy as np
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim import mpc
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

path, n_env, secs, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
DS = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.0, 0.1, 0.2]
#: 'all' = +d on every decision; 'clear' = +d only where no live prop is in the corridor within 10 m (빈 길만)
MODE = sys.argv[6] if len(sys.argv) > 6 else "all"
#: optional override of the exploration noise's correlation (the run's own --noise-corr otherwise)
CORR = float(sys.argv[7]) if len(sys.argv) > 7 else None
dev = torch.device("cuda")
ck = torch.load(path, map_location=dev, weights_only=False)
a = argparse.Namespace(**ck["args"])
a.envs, a.seed, a.device = n_env, seed, "cuda"
lay = mpc._LAYOUTS["wp4"]
i_hard = [CC.index(k) for k in ("hard_wall", "hard_prop")]


def run(d, noise):
    env = SR.build_env(a, dev)
    prepare_graph_runtime(env, log=lambda _t: None)
    env.reset(seed=seed)
    lid = env.learner_ids; B = int(lid.numel())
    remover = SR.PropRemoval(env, float(a.removal_p), float(a.removal_s[0]), float(a.removal_s[1]), seed=seed) \
        if float(getattr(a, "removal_p", 0) or 0) > 0 else None
    tp = SR.TeacherPolicy(env, ck, dev)
    w_col = abs(env.ecfg.reward_collision)
    act = torch.zeros(B, tp.A, device=dev); eps = torch.zeros_like(act)
    act_full = torch.zeros(env.B, env.act_dim, device=dev)
    R = int(a.repeat); al = float(a.noise_corr) if CORR is None else CORR
    prog = 0.0; hits = 0.0; hard1 = 0.0; steps = 0; laps = []
    # split by what is ahead: a live prop in the corridor within 10 m (SoloMix's own test) = 상자 근처, else 빈 길
    n_seg = sum(1 for x in SR.AHEAD if x <= 10.0 + 1e-6)
    pc = torch.zeros(2, device=dev); hc = torch.zeros(2, device=dev); tc = torch.zeros(2, device=dev)
    npr = []
    cf0 = env.sim.contact_frame()[lid].float()
    with torch.no_grad():
        for t in range(int(secs / env.sim.control_dt) // R):
            tp.o = tp.sobs(act)
            dist = tp.net.dist(tp.nrm(tp.o))
            m = dist.mean.clone()
            dd = torch.full((B, 1), d, device=dev)
            if MODE == "clear":
                cb0 = SR.corridor_block(env, lid, env.last_result.s[lid], env.sim.tid[lid]).reshape(B, len(SR.AHEAD), SR.CORRIDOR_BINS)
                dd = dd * (~cb0[:, :n_seg].flatten(1).any(1)).float()[:, None]
            if t == 0 and d == DS[0] and noise:
                print("  잡음 표준편차(행동 단위, 곡률 4 + 속도 4):", [round(float(x), 3) for x in dist.stddev[0]], flush=True)
            m[:, lay.speed] = m[:, lay.speed] + dd
            if noise:
                eps = al * eps + math.sqrt(1.0 - al * al) * torch.randn_like(eps)
                act = SR.clamp_any(m + dist.stddev * eps, tp.ar)
            else:
                act = SR.clamp_any(m, tp.ar)
            act_full[lid] = act
            for _ in range(R):
                _, r_, te, trc, info = env.step(act_full)
                if remover is not None:
                    remover.step()
                comp = info["reward_components"]
                h_ = (-comp["collision"][lid].float() / max(w_col, 1e-9)).clamp_min(0)
                p_ = info["progress"][lid].float().clamp_min(0)
                hits += float(h_.sum()); prog += float(p_.sum()); steps += 1
                cb = SR.corridor_block(env, lid, env.last_result.s[lid], env.sim.tid[lid]).reshape(B, len(SR.AHEAD), SR.CORRIDOR_BINS)
                near = cb[:, :n_seg].flatten(1).any(1).long()
                pc.index_add_(0, near, p_); hc.index_add_(0, near, h_); tc.index_add_(0, near, torch.ones_like(p_))
                if t == 0:
                    po = env.procedural
                    if po is not None:
                        npr = (po.p_sid[lid] >= 0).sum(1).cpu().numpy()
                if info["lap_times"].numel():
                    laps += info["lap_times"][env.learner[info["lap_ids"]]].tolist()
    cf = env.sim.contact_frame()[lid].float()
    hard1 = float((cf[:, i_hard] - cf0[:, i_hard]).sum())
    km = prog / 1000.0
    T = steps * B * env.sim.control_dt
    J = (prog - 80.0 * hits) / (steps * B)
    lp = np.median(laps) if laps else float("nan")
    print(f"  d {d:+.2f} ({d * 5:+.1f} m/s) {'잡음 있음' if noise else '잡음 없음'}: 진행 {prog / T:.2f} m/s | 값 매긴 충돌(>= 1.5) {hits / km:.2f}/km, "
          f"콘솔 세게(>= 1.0) 벽+상자 {hard1 / km:.2f}/km | 랩 중앙 {lp:.2f} s ({len(laps)}개) | J80 {J:.4f}/스텝", flush=True)
    for i, nm in ((0, "빈 길"), (1, "상자 근처")):
        tt = float(tc[i]) * env.sim.control_dt
        if tt > 0:
            print(f"      {nm}: 시간 {float(tc[i]) / float(tc.sum()) * 100:.0f} % | 진행 {float(pc[i]) / tt:.2f} m/s | 값 매긴 충돌 {float(hc[i]) / max(float(pc[i]) / 1000, 1e-3):.2f}/km | "
                  f"J80 {(float(pc[i]) - 80 * float(hc[i])) / float(tc[i]):.4f}/스텝", flush=True)
    if d == DS[0] and not noise and len(npr):
        print(f"      첫 배치 상자 수(학습 세계): 0개 {np.mean(npr == 0) * 100:.0f} %, 중앙 {np.median(npr):.0f}, p90 {np.percentile(npr, 90):.0f}, 최대 {npr.max()}", flush=True)
    del env, tp
    torch.cuda.empty_cache()
    return J


print(f"== {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} | 학습 세계 {n_env}대 x {secs:.0f} s | 잡음 상관 {a.noise_corr if CORR is None else CORR} | 속도 올림 {'빈 길에서만' if MODE == 'clear' else '모든 결정'}")
for noise in (False, True):
    for d in DS:
        run(d, noise)
