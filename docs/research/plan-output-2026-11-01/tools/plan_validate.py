"""planner validation, batched (11-08, the user: "학습때 planner 평가해서 보상주는것처럼, validation도 평가 환경에다가 보상값만 쭈르륵
뽑으면 빠른거 아냐?"): the training world (`plan_rl` WORLD: the four cases, box layouts, opponents) with a FIXED seed -- the same
cases, layouts, starts and opponents for every checkpoint -- many cars at once, the policy's mean, for `secs` seconds, under
one condition:
  ideal  the training's: the car drives the plan exactly, contacts the simulator's
  map    the real car's: full physics, MAP follows the plan (the console's way)
Per case (empty / opponent / boxes / boxes + opponent): laps a minute, lap time (median of whole laps), contacts a minute by kind
and the hard ones (>= 1 m/s), stopped share, the reward's parts a decision, plan quality (0.3 s point flips a second, plan vs
where the car was 0.1 / 0.3 / 0.5 s later). One line per case, prefixed by `tag`.
args: checkpoint condition(ideal|map) envs secs [tag]"""
import os, sys, math, argparse, json, time
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_reward import PlanReward
from f1sim.learn.plan_policy import PlanObs, load_actor, plan_world, plan_segments, PT_IDX, PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn
from f1sim import map_controller as M
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
from f1sim.lidar import HIT_CAR

torch.set_grad_enabled(False)
ck_path, mode, n_env, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
tag = sys.argv[5] if len(sys.argv) > 5 else ""
SEED = 4242
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = n_env, SEED, "cuda"
a.episode_s = secs + 5.0                                   # one episode per car: no respawn inside the measurement
if mode == "map":
    a.plan_format, a.repeat = "tw", 4
torch.manual_seed(SEED)
env = SR.build_env(a, dev)
if mode == "ideal":
    env.enable_ideal()
if mode == "map":
    # the controller's delay handling under test (11-08): F1SIM_MAP_DELAY [s] (MAP's pose prediction), F1SIM_SPEED_LEAD [s]
    if os.environ.get("F1SIM_MAP_DELAY"):
        env.tracker.map.p.delay_comp = float(os.environ["F1SIM_MAP_DELAY"])
    if os.environ.get("F1SIM_SPEED_LEAD"):
        env.tracker.tw_speed_lead = float(os.environ["F1SIM_SPEED_LEAD"])
    tag = tag + f"(지연 {env.tracker.map.p.delay_comp:.3f} s, 속도 앞당김 {env.tracker.tw_speed_lead:.2f} s) "
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=SEED)
lid = env.learner_ids; B = int(lid.numel())
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
V2 = ck.get("format") == "plan_steps2"
actor = (load_actor2 if V2 else load_actor)(ck, dev)
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
obs = (PlanObs2 if V2 else PlanObs)(env, lim, pr.goal)
vest = []
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
# the case of each learner row: boxes in its world? an opponent in its race?
has_props = torch.zeros(B, dtype=torch.bool, device=dev)
proc = getattr(env, "procedural", None)
if proc is not None:
    sid = proc.p_sid[lid] if proc.p_sid.shape[0] == env.B else proc.p_sid[env.sim.eid[lid]]
    has_props = (sid >= 0).any(-1)
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
others = env.sim.other_idx[lid] if env.M > 1 else None
has_opp = torch.zeros(B, dtype=torch.bool, device=dev) if others is None else (~absent[others]).any(-1)
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car")]
i_hd = [CC.index(k) for k in ("hard_wall", "hard_prop", "hard_car")]
cf0 = env.sim.contact_frame()[lid].float()
t0 = time.time()
n_dec = int(secs / PS.DT)
plans, poses = [], []
parts = {k: torch.zeros(B, device=dev) for k in ("lap", "shaping", "contact", "plan_hit")}
stop = torch.zeros(B, device=dev); lap_t0 = torch.full((B,), float("nan"), device=dev); lap_times = [[] for _ in range(B)]
for kd in range(n_dec):
    st = env.sim.state[lid].clone()
    if V2:
        _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
        vest.append((v_hat - st[:, dyn.IVX]).abs())
    else:
        mean, _ = actor(obs.scans, obs.state())
        p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
    x, y, psi = plan_world(p, st)
    pc = pr.plan_cost(x, y, psi, p["v"])
    plans.append(torch.stack([x[:, ::PS.SUB], y[:, ::PS.SUB]], -1)); poses.append(st[:, :3].clone())
    if mode == "map":
        k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, j]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, j]
        env.step(act0)
        r = env.last_result
        pr.control_step((r.scan_type[lid] == HIT_CAR).any(1))
    idx = PT_IDX.to(dev)
    out = pr.decision(torch.stack([x[:, idx], y[:, idx]], -1), pc["price"])
    for k in ("lap", "shaping", "contact"):
        parts[k] += out[k]
    parts["plan_hit"] += out["plan_hit"]
    obs.push_scan(); obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
    if V2:
        obs.set_plan2(p)
    tnow = (kd + 1) * PS.DT
    crossed = out["lap"] > 0
    if bool(crossed.any()):
        for i in torch.nonzero(crossed)[:, 0].tolist():
            if not math.isnan(float(lap_t0[i])):
                lap_times[i].append(tnow - float(lap_t0[i]))
            lap_t0[i] = tnow
    stop += (env.sim.state[lid, dyn.IVX].abs() < 0.3).float() * PS.DT
cf = env.sim.contact_frame()[lid].float() - cf0
on, hd = cf[:, i_on], cf[:, i_hd]
W = torch.stack(plans); X = torch.stack(poses)            # (T, B, 11, 2) plan points every 0.1 s (index 0 = the car), (T, B, 3)
def flips(m):
    a_ = W[:-1, :, 4] - X[1:, :, :2]; b_ = W[1:, :, 3] - X[1:, :, :2]   # the 0.4 s point of plan k = the 0.3 s point of plan k+1
    yy = X[1:, :, 2]
    sa = -a_[..., 0] * torch.sin(yy) + a_[..., 1] * torch.cos(yy); sb = -b_[..., 0] * torch.sin(yy) + b_[..., 1] * torch.cos(yy)
    f = ((sa * sb < 0) & (sa.abs() > 0.1) & (sb.abs() > 0.1)).float()
    return float(f[:, m].mean()) / PS.DT
def honest(m, h):
    k = int(round(h / PS.DT))
    d = (W[:-k, :, k] - X[k:, :, :2]).norm(dim=-1)[:, m]
    return float(d.median()) * 100
cases = (("빈 트랙", ~has_props & ~has_opp), ("상대 차", ~has_props & has_opp), ("상자", has_props & ~has_opp), ("상자+상대", has_props & has_opp))
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
_ve = f", 속도 짐작 오차 중앙 {float(torch.cat(vest).median()):.2f} m/s" if vest else ""
print(f"{tag}[{mode}] {name}: 차 {B}대 x {secs:.0f} s, 시드 {SEED} ({time.time() - t0:.0f} s){_ve}", flush=True)
for cname, m in cases:
    n = int(m.sum())
    if n == 0:
        continue
    mins = secs * n / 60
    lt = [t for i in torch.nonzero(m)[:, 0].tolist() for t in lap_times[i]]
    lt_ = torch.tensor(lt) if lt else torch.tensor([float("nan")])
    print(f"{tag}  {cname} {n}대 | 바퀴 {float(parts['lap'][m].sum()) / mins:.2f}/분, 한 바퀴 중앙 {float(lt_.median()):.2f} s | 접촉 /분 벽 "
          f"{float(on[m, 0].sum()) / mins:.2f}(세게 {float(hd[m, 0].sum()) / mins:.2f}) 상자 {float(on[m, 1].sum()) / mins:.2f}(세게 "
          f"{float(hd[m, 1].sum()) / mins:.2f}) 차 {float(on[m, 2].sum()) / mins:.2f}(세게 {float(hd[m, 2].sum()) / mins:.2f}) | 멈춤 "
          f"{float(stop[m].mean()) / secs * 100:.0f} % | 보상/결정 {float((parts['lap'] + parts['shaping'] + parts['contact'] + parts['plan_hit'])[m].mean()) / n_dec:.4f} "
          f"(접촉 {float(parts['contact'][m].mean()) / n_dec:.4f}, plan 충돌 {float(parts['plan_hit'][m].mean()) / n_dec:.4f}) | 뒤집힘 {flips(m):.2f}/s, "
          f"plan과 실제 0.1/0.3/0.5 s {honest(m, 0.1):.1f}/{honest(m, 0.3):.1f}/{honest(m, 0.5):.1f} cm", flush=True)
