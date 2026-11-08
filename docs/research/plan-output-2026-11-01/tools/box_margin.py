"""상자 옆을 얼마나 바짝 지나가나 (11-08 12:00): 실차 조건(물리 + MAP)에서만 같은 상자에 바퀴마다 박고 학습 조건에서는 0번이었음(u500
배치 1, u300 배치 6) -> 가설: 학습 때 plan대로 정확히 움직이니 policy가 상자에 여유 없이 붙는 라인을 배웠고, MAP 추종 오차(0.3 s 12 cm,
0.5 s 22 cm)가 그 여유를 먹음. 확인: 학습 세계의 상자 배치(상대 없음, 고정 시드)에서 차 N대를 몰며 0.025 s마다 차 외곽(모서리 4 +
변 가운데 4)과 가장 가까운 상자 사이 틈을 재고, 틈이 0.6 m 아래로 들어갔다 나온 구간 하나를 '한 번 지나감'으로 그 구간의 가장 좁은 틈을
셈. 학습 조건에서 지나감 대부분이 추종 오차보다 좁으면 가설이 맞음, 넓으면 틀림.
인자: checkpoint 조건(ideal|map) 차수 초"""
import os, sys, argparse, json
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_policy import PlanObs, load_actor, plan_world, plan_segments, PT_IDX, PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M

torch.set_grad_enabled(False)
ck_path, mode, n_env, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
SEED = 4242
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = n_env, SEED, "cuda"
a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 1.0, 0.0, 0.0, secs + 5.0
if mode == "map":
    a.plan_format, a.repeat = "tw", 4
torch.manual_seed(SEED)
env = SR.build_env(a, dev)
if mode == "ideal":
    env.enable_ideal()
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=SEED)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
V2 = ck.get("format") == "plan_steps2"
actor = (load_actor2 if V2 else load_actor)(ck, dev)
obs = (PlanObs2 if V2 else PlanObs)(env, lim, None)
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
c4 = env.sim.corners.float()
pts = torch.cat([c4, 0.5 * (c4 + c4.roll(1, 0))], 0)            # (8, 2) the footprint's corners and edge middles about the CoG
eid = env.sim.eid[lid][:, None].expand(B, pts.shape[0]).reshape(-1)
gaps = []
for kd in range(int(secs / PS.DT)):
    st = env.sim.state[lid].clone()
    if V2:
        _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    else:
        mean, _ = actor(obs.scans, obs.state())
        p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
    x, y, psi = plan_world(p, st)
    if mode == "map":
        k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(act0)
        s = env.sim.state[lid]
        c, sn = torch.cos(s[:, 2:3]), torch.sin(s[:, 2:3])
        wx = s[:, 0:1] + pts[:, 0] * c - pts[:, 1] * sn; wy = s[:, 1:2] + pts[:, 0] * sn + pts[:, 1] * c
        g = env.procedural.clearance(torch.stack([wx, wy], -1).reshape(-1, 2), eid).view(B, -1).min(1).values
        gaps.append(g)
    idx = PT_IDX.to(dev)
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
    if V2:
        obs.set_plan2(p)
G = torch.stack(gaps, 1).cpu()                                   # (B, T)
skip = int(3.0 / env.sim.control_dt)
G = G[:, skip:]
passes = []
for b in range(B):
    near = (G[b] < 0.6).tolist(); lo = None
    for t, n in enumerate(near + [False]):
        if n and lo is None:
            lo = t
        elif not n and lo is not None:
            passes.append(float(G[b, lo:t].min())); lo = None
P = torch.tensor(passes) if passes else torch.tensor([float("nan")])
edges = [-1e9, 0.0, 0.10, 0.20, 0.30, 0.6]
names = ["닿음", "0-10 cm", "10-20 cm", "20-30 cm", "30-60 cm"]
share = [float(((P >= lo) & (P < hi)).float().mean()) * 100 for lo, hi in zip(edges[:-1], edges[1:])]
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
print(f"[{mode}] {name}: 상자 배치 차 {B}대 x {secs:.0f} s, 상자 옆 지나감 {len(passes)}번 | 가장 좁은 틈 중앙 {float(P.median()) * 100:.0f} cm | "
      + ", ".join(f"{n_} {s_:.0f} %" for n_, s_ in zip(names, share)), flush=True)
