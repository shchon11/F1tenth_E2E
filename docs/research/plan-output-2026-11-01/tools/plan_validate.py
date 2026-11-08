"""planner validation, batched (11-08, the user: "학습때 planner 평가해서 보상주는것처럼, validation도 평가 환경에다가 보상값만 쭈르륵
뽑으면 빠른거 아냐?"): the training world (`plan_rl` WORLD: the four cases, box layouts, opponents) with a FIXED seed -- the same
cases, layouts, starts and opponents for every checkpoint -- many cars at once, the policy's mean, for `secs` seconds, under
one condition:
  ideal  the training's: the car drives the plan exactly, contacts the simulator's
  map    the real car's: full physics, MAP follows the plan (the console's way)
  track  full physics, `plan_follow.PlanFollower` follows the plan in time (11-09, the user: "얼추 여기에 맞는 제어기 좀
         만들어놔"); gains from F1SIM_TRACK_GAINS="k_v,k_lat,k_head"
F1SIM_HOLD=N (track / ideal): the policy plans every N decisions and the controller follows each plan N x 0.1 s -- prints
the tracking error of a whole plan (where the plan said the car would be at 0.1 ... N x 0.1 s vs where it was).
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
HOLD = int(os.environ.get("F1SIM_HOLD", "1"))
follower = None
if mode == "track":
    from f1sim.plan_follow import PlanFollower
    _g = [float(v) for v in os.environ.get("F1SIM_TRACK_GAINS", "2.0,1.0,1.0").split(",")]
    follower = PlanFollower(B, dev, k_v=_g[0], k_lat=_g[1], k_head=_g[2], s_max=float(env.s_max))
    env.ext_mask[lid] = True; env._ext_ids.update(int(i) for i in lid.tolist())
    tag = tag + f"(추종 이득 속도 {_g[0]:g} 옆 {_g[1]:g} 방향 {_g[2]:g}) "
hold_err = {h: [] for h in range(1, PS.N_STEP + 1)}
kd0, cur_pts, kd0_prev = 0, None, -9
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
V2 = ck.get("format") == "plan_steps2"
actor = (load_actor2 if V2 else load_actor)(ck, dev)
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
obs = (PlanObs2 if V2 else PlanObs)(env, lim, pr.goal)
vest, vest_s, v01, vnow, v01_act = [], [], [], [], []
act0 = torch.zeros(env.B, env.act_dim, device=dev)
kk = torch.zeros(env.B, PS.N_STEP, device=dev); vv = torch.zeros(env.B, PS.N_STEP, device=dev)
# the case of each learner row: boxes in its world? an opponent in its race?
has_props = torch.zeros(B, dtype=torch.bool, device=dev)
proc = getattr(env, "procedural", None)
if proc is not None:
    # the props of a car's world sit at its world index (`Sim.eid`: both cars of a race share one), not at its row
    sid = proc.p_sid[env.sim.eid[lid]]
    has_props = (sid >= 0).any(-1)
props_ever = has_props.clone()                             # boxes in its world at any moment
absent = env.sim.absent if getattr(env.sim, "absent_any", False) else torch.zeros(env.B, dtype=torch.bool, device=dev)
others = env.sim.other_idx[lid] if env.M > 1 else None
has_opp = torch.zeros(B, dtype=torch.bool, device=dev) if others is None else (~absent[others]).any(-1)
opp_ever = has_opp.clone()                                 # an opponent present at any moment (a seat can be refilled)
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
    if kd > 0 and len(v01_act) < len(v01) and kd - 1 == kd0_prev:
        v01_act.append(st[:, dyn.IVX])
    if cur_pts is not None and 1 <= kd - kd0 <= PS.N_STEP:
        hold_err[kd - kd0].append((st[:, :2] - cur_pts[:, kd - kd0 - 1]).norm(dim=-1))
    new = kd % HOLD == 0
    if new:
        kd0 = kd; kd0_prev = kd
        if V2:
            _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
            vest.append((v_hat - st[:, dyn.IVX]).abs()); vest_s.append(v_hat - st[:, dyn.IVX])
            v01.append(p["v"][:, PS.SUB]); vnow.append(st[:, dyn.IVX])
        else:
            mean, _ = actor(obs.scans, obs.state())
            p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
        x, y, psi = plan_world(p, st)
        pc = pr.plan_cost(x, y, psi, p["v"])
        cur_pts = torch.stack([x[:, PT_IDX.to(dev)], y[:, PT_IDX.to(dev)]], -1)
        if follower is not None:
            follower.set_plan(slice(None), x, y, psi, p["v"], p["delta"])
    plans.append(torch.stack([x[:, ::PS.SUB], y[:, ::PS.SUB]], -1)); poses.append(st[:, :3].clone())
    if mode == "map":
        k_, v_ = plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            jj = min((kd - kd0) * PS.SUB + j, p["cmd_d"].shape[1] - 1)
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, jj]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, jj]
        if follower is not None:
            env.ext_cmd[lid] = follower.command(env.sim.state[lid])
        env.step(act0)
        r = env.last_result
        if others is not None and getattr(env.sim, "absent_any", False):
            opp_ever |= (~env.sim.absent[others]).any(-1)
        if proc is not None:
            props_ever |= (proc.p_sid[env.sim.eid[lid]] >= 0).any(-1)
        pr.control_step((r.scan_type[lid] == HIT_CAR).any(1))
    idx = PT_IDX.to(dev)
    out = pr.decision(torch.stack([x[:, idx], y[:, idx]], -1), pc["price"])
    for k in ("lap", "shaping", "contact"):
        parts[k] += out[k]
    parts["plan_hit"] += out["plan_hit"]
    obs.push_scan()
    if new:
        obs.set_plan(torch.stack([x[:, idx], y[:, idx]], -1), p["pv"])
        if V2:
            obs.set_plan2(p)
    elif V2:
        # a held plan: the next plan starts from the steering / speed the held one has reached by now (its 'last command')
        _i = min((kd - kd0 + 1) * PS.SUB, p["delta"].shape[1] - 1)
        obs.cmd_d, obs.cmd_v = p["delta"][:, _i], p["v"][:, _i]
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
# 11-09: a seat or a layout can be filled mid-run (a race that resets whole redraws them), so the 'empty' case collected car
# and box contacts -- a car is put in a case by what it met at any moment, not by its start
_late_o, _late_p = int((opp_ever & ~has_opp).sum()), int((props_ever & ~has_props).sum())
if _late_o or _late_p:
    print(f"{tag}  (도중에 상대 차가 나타난 차 {_late_o}대, 상자가 나타난 차 {_late_p}대: 칸은 40 s 동안 한 번이라도 있었는지로 나눔)", flush=True)
has_opp, has_props = opp_ever, props_ever
cases = (("빈 트랙", ~has_props & ~has_opp), ("상대 차", ~has_props & has_opp), ("상자", has_props & ~has_opp), ("상자+상대", has_props & has_opp))
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
_ve = (f", 속도 짐작 오차 중앙 {float(torch.cat(vest).median()):.2f} m/s (짐작 - 실제 중앙 {float(torch.cat(vest_s).median()):+.2f}, "
       f"평균 {float(torch.cat(vest_s).mean()):+.2f}; 0.1 s 뒤 plan 속도 - 실제 중앙 "
       f"{float((torch.stack(v01[:len(v01_act)]) - torch.stack(v01_act)).median()):+.2f} m/s)") if vest and v01_act else ""
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
if HOLD > 1:
    hs = [h for h in hold_err if hold_err[h]]
    print(f"{tag}  plan 하나를 {HOLD * PS.DT:.1f} s 동안 따라감: plan이 말한 자리와 실제 위치 차이 중앙 "
          + ", ".join(f"{h * PS.DT:.1f} s {float(torch.cat(hold_err[h]).median()) * 100:.1f}" for h in hs) + " cm", flush=True)
