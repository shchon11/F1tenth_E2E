"""재설계 7단계 시험 (11-07): 정해진 운전 규칙(scripted_planner.Sampler)을 두 조건에서 몰아 봄 -- ideal: 학습처럼 plan대로 정확히,
map: 실차처럼 전체 물리 + MAP(시간별 waypoint 추종 경로, 0.1 s마다 새 plan, 사이에는 env가 세계 좌표로 붙잡음). 같은 규칙·같은
출발로 바퀴 시간, 접촉, 멈춘 시간, plan 품질(0.3 s 점이 바로 전 plan과 반대쪽으로 넘어간 횟수, plan이 0.1·0.3·0.5 s 뒤 있겠다던 곳과
실제 위치 차이)을 비교. 학습된 planner도 같은 잣대로: K 자리에 plan_rl checkpoint(.pt)를 주면 그 policy(평균 출력)가 몬다.
인자: 조건(ideal|map) 차수 초 상자(0|1) [K 또는 checkpoint]"""
import os, sys, math, argparse, json, time
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(__file__))
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import PlanReward
from f1sim.learn.plan_policy import plan_world, plan_segments, PT_IDX
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M
from scripted_planner import Sampler

torch.set_grad_enabled(False)
mode, n_env, secs, props = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1"
CKPT = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5].endswith(".pt") else None
K = int(sys.argv[5]) if len(sys.argv) > 5 and CKPT is None else 64
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = n_env, 5, "cuda"
a.prop_fraction = 1.0 if props else 0.0
a.removal_p, a.opp_present, a.episode_s = 0.0, 0.0, 3600.0
if mode == "map":
    a.plan_format, a.repeat = "tw", 4                     # MAP follows the plan as a time-waypoint plan, a decision per 0.1 s
env = SR.build_env(a, dev)
if mode == "ideal":
    env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=5)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
pr = PlanReward(env, 1.0, dev)
if CKPT is None:
    pol = Sampler(env, lim, pr, K, dev)
else:
    from f1sim.learn.plan_policy import PlanObs, load_actor
    _ck = torch.load(CKPT, map_location=dev, weights_only=False)
    _actor = load_actor(_ck, dev)
    _obs = PlanObs(env, lim, None)

    def pol(st):
        mean, _ = _actor(_obs.scans, _obs.state())
        return torch.tanh(mean)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
plans, poses = [], []
laps = torch.zeros(B, device=dev); lap_t = []; t_last = torch.full((B,), float("nan"))
stop = torch.zeros(B, device=dev)
t0 = time.time()
n_dec = int(secs / PS.DT)
for kd in range(n_dec):
    st = env.sim.state[lid].clone()
    act = pol(st)
    p = PS.roll(act, st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
    x, y, psi = plan_world(p, st)
    plans.append(torch.stack([x, y], -1)); poses.append(st[:, :3].clone())
    if mode == "map":
        k_, v_ = plan_segments(p, lim)
        kk[lid] = k_; vv[lid] = v_
        env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(act0)
        pr.control_step()
    idx = PT_IDX.to(dev)
    out = pr.decision(torch.stack([x[:, idx], y[:, idx]], -1))
    if CKPT is not None:
        _obs.push_scan(); _obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
    laps += out["lap"]
    for i in torch.nonzero(out["lap"] > 0)[:, 0].tolist():
        tnow = (kd + 1) * PS.DT
        if not math.isnan(float(t_last[i])):
            lap_t.append(tnow - float(t_last[i]))
        t_last[i] = tnow
    stop += (env.sim.state[lid, dyn.IVX].abs() < 0.3).float() * PS.DT
W = torch.stack(plans); X = torch.stack(poses)                 # (T, B, n, 2), (T, B, 3)
tt = p["t"]
def at(Wk, t):
    j = int(round(t / float(tt[1] - tt[0])))
    return Wk[..., j, :]
step = PS.DT
a_ = at(W[:-1], 0.3 + step) - X[1:, :, :2]; b_ = at(W[1:], 0.3) - X[1:, :, :2]
yy = X[1:, :, 2]
sa = -a_[..., 0] * torch.sin(yy) + a_[..., 1] * torch.cos(yy); sb = -b_[..., 0] * torch.sin(yy) + b_[..., 1] * torch.cos(yy)
flip = ((sa * sb < 0) & (sa.abs() > 0.1) & (sb.abs() > 0.1)).float().mean() / step
moving = X[1:, :, :2].sub(X[:-1, :, :2]).norm(dim=-1) > 0.03
hon = []
for h in (0.1, 0.3, 0.5):
    k_ = int(round(h / step))
    d = (at(W[:-k_], h) - X[k_:, :, :2]).norm(dim=-1)
    m = moving[k_ - 1:] if moving.shape[0] >= d.shape[0] else moving
    hon.append(f"{h:.1f} s {float(d.median()) * 100:.1f} cm (p90 {float(d.quantile(0.9)) * 100:.1f})")
lt = torch.tensor(lap_t) if lap_t else torch.tensor([float("nan")])
mins = secs * B / 60
who = (os.path.basename(os.path.dirname(CKPT)) + "/" + os.path.basename(CKPT)) if CKPT else f"정해진 규칙(K {K})"
print(f"[{mode}] {who} {'상자 배치' if props else '빈 트랙'}, 차 {B}대 x {secs:.0f} s ({time.time() - t0:.0f} s) | 바퀴 {float(laps.sum()):.0f}, "
      f"한 바퀴 중앙 {float(lt.median()):.2f} s | 접촉 벽 {float(pr.n_contacts[:, 0].sum()) / mins:.1f} 상자 {float(pr.n_contacts[:, 1].sum()) / mins:.1f} /분 | "
      f"멈춘 시간 {float(stop.mean()) / secs * 100:.0f} % | 0.3 s 점 뒤집힘 {float(flip):.2f}/s | plan과 실제 위치 차이 중앙: {', '.join(hon)}", flush=True)
