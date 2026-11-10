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
from f1sim.learn.plan_policy import PlanObs3, load_actor3, plan_v3, plan_segments3, plan3_models
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
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:          # the tyre limit the checkpoint was trained with (11-10)
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
V3 = ck.get("format") == "plan_steps3"
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
act3, turn3 = plan3_models(env, lim, dev) if V3 else (None, None)
if mode == "ideal":
    env.enable_ideal(lag=V3, turn=turn3)                       # v3: the delay-aware execution (before the graphs)
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
    from f1sim.plan_follow import PlanFollower, to_car
    _g = [float(v) for v in os.environ.get("F1SIM_TRACK_GAINS", "2.0,1.0,1.0").split(",")]
    follower = PlanFollower(B, dev, k_v=_g[0], k_lat=_g[1], k_head=_g[2], s_max=float(env.s_max),
                            lut=None if os.environ.get("F1SIM_TRACK_NOLUT") == "1" else env.tracker.map.lut,
                            ff_window=float(os.environ.get("F1SIM_TRACK_FFWIN", "0")),
                            slow_tau=float(os.environ.get("F1SIM_TRACK_SLOWTAU", "0")),
                            k_yaw=float(os.environ.get("F1SIM_TRACK_KYAW", "0")),
                            grip=M.GRIP_LEVELS.index(1.0), wheelbase=float(env.tracker.nominal["lf"]) + float(env.tracker.nominal["lr"]))
    if os.environ.get("F1SIM_TRACK_MPC") == "1":
        from f1sim.plan_mpc import PlanMPC
        _w = [float(v) for v in os.environ.get("F1SIM_MPC_W", "400,100,2500,0").split(",")]
        follower = PlanMPC(B, os.environ.get("F1SIM_MPC_DEV", "cuda"), env.tracker.nominal, s_max=float(env.s_max), k_v=_g[0],
                           horizon=int(os.environ.get("F1SIM_MPC_H", "20")), sub=int(os.environ.get("F1SIM_MPC_SUB", "4")),
                           every=int(os.environ.get("F1SIM_MPC_EVERY", "1")), lut=env.tracker.map.lut, grip=M.GRIP_LEVELS.index(1.0),
                           **dict(zip(("w_lat", "w_head", "w_rate", "w_yaw"), _w)))
        tag = tag + f"(MPC 무게 옆 {_w[0]:g} 방향 {_w[1]:g} 조향 변화 {_w[2]:g} 요 {_w[3]:g}, {follower.H} 걸음) "
    env.ext_mask[lid] = True; env._ext_ids.update(int(i) for i in lid.tolist())
    tag = tag + f"(추종 이득 속도 {_g[0]:g} 옆 {_g[1]:g} 방향 {_g[2]:g}" + (f", 표 입력 {follower.ff_window:g} s 평균" if follower.ff_window > 0 else "") + ") "
hold_err = {h: [] for h in range(1, PS.N_STEP + 1)}
kd0, cur_pts, kd0_prev = 0, None, -9
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
V2 = ck.get("format") == "plan_steps2" or V3
actor = (load_actor3 if V3 else load_actor2 if V2 else load_actor)(ck, dev)
pr = PlanReward(env, float(ck["args"].get("gamma", 0.995)), dev)
obs = (PlanObs3 if V3 else PlanObs2 if V2 else PlanObs)(env, lim, pr.goal)
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
SDs, YRs = [], []                                          # the car's steering angle / yaw rate every control step (11-10)
LIMR, TBS = [], []                                         # the plan's 0.1 s point: share of the tyre limit, the table's steering
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
        if V3:
            _, mean, _, v_hat, p = plan_v3(actor, obs, lim, act3, turn3)
            vest.append(
                (v_hat - st[:, dyn.IVX]).abs()); vest_s.append(v_hat - st[:, dyn.IVX])
            v01.append(p["v"][:, PS.SUB]); vnow.append(st[:, dyn.IVX])
        elif V2:
            _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
            vest.append((v_hat - st[:, dyn.IVX]).abs()); vest_s.append(v_hat - st[:, dyn.IVX])
            v01.append(p["v"][:, PS.SUB]); vnow.append(st[:, dyn.IVX])
        else:
            mean, _ = actor(obs.scans, obs.state())
            p = PS.roll(torch.tanh(mean), st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
        x, y, psi = plan_world(p, st)
        pc = pr.plan_cost(x, y, psi, p["v"])
        if V2 and not V3:
            # 11-10: how close to the tyres' limit the plan drives (`plan_steps.Limits.kappa_at`: the table's highest
            # steady-state lateral acceleration, i.e. at full lock) and the steering the car needs for it there (MAP's table)
            _v1, _d1 = p["v"][:, PS.SUB], p["delta"][:, PS.SUB]
            LIMR.append((torch.tan(_d1).abs() / lim.wheelbase / lim.kappa_at(_v1).clamp_min(1e-6)).float())
            _acc = _v1 * _v1 * torch.tan(_d1) / lim.wheelbase
            _tb = env.tracker.map.lut.lookup(_acc, _v1.abs().clamp_min(0.5), torch.full_like(_v1, M.GRIP_LEVELS.index(1.0), dtype=torch.long), interp="linear")
            TBS.append(torch.rad2deg(torch.nan_to_num(_tb, nan=float(env.s_max)).abs()).float())
        cur_pts = torch.stack([x[:, PT_IDX.to(dev)], y[:, PT_IDX.to(dev)]], -1)
        if follower is not None:
            # v3: the plan's poses in time too, its servo angle as the kinematic steering of the same turn (11-09)
            follower.set_plan(slice(None), x, y, psi, p["v"], turn3.eff(p["delta"], p["v"]) if V3 else p["delta"])
    plans.append(torch.stack([x[:, ::PS.SUB], y[:, ::PS.SUB]], -1)); poses.append(st[:, :3].clone())
    if mode == "map":
        k_, v_ = plan_segments3(p, lim, turn3) if V3 else plan_segments(p, lim); kk[lid] = k_; vv[lid] = v_; env.ext_plan_kv = (kk, vv)
    else:
        env.sim.ideal_vcap[lid] = 1e3
    for j in range(PS.SUB):
        if mode == "ideal":
            jj = min((kd - kd0) * PS.SUB + j, p["cmd_d"].shape[1] - 1)
            env.ideal_cmd[lid, 0] = p["cmd_d"][:, jj]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, jj]
        if follower is not None:
            env.ext_cmd[lid] = to_car(env, lid, follower.command(env.sim.state[lid]))
        env.step(act0)
        r = env.last_result
        if others is not None and getattr(env.sim, "absent_any", False):
            opp_ever |= (~env.sim.absent[others]).any(-1)
        if proc is not None:
            props_ever |= (proc.p_sid[env.sim.eid[lid]] >= 0).any(-1)
        pr.control_step((r.scan_type[lid] == HIT_CAR).any(1))
        SDs.append(env.sim.state[lid, dyn.ISTEER].float()); YRs.append(env.sim.state[lid, dyn.IR].float())
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
        if V3:
            obs.servo, obs.cmd_d, obs.cmd_v = p["delta"][:, _i], p["cmd_d"][:, _i - 1], p["cmd_v"][:, _i - 1]
        else:
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
# 11-10 (the user: "결국 planner 출력 자체가 흔들리게 나온다는거야?"): does the car itself wobble -- its steering angle's change every
# 0.1 s, the share of those changes that flip sign at once (both > 0.5 deg), and the yaw rate's oscillation (minus its 1 s mean);
# the same measures as console_wobble.py, over all cars and the empty-track ones, the first 2 s left out. With F1SIM_HOLD=10 the
# controller follows each plan a whole second -- the controller on its own, without the policy re-planning
_SD = torch.rad2deg(torch.stack(SDs))[80:]; _YR = torch.rad2deg(torch.stack(YRs))[80:]
def _wob(msk):
    s10 = _SD[::PS.SUB][:, msk]; dd = s10[1:] - s10[:-1]
    zz = ((dd[1:] * dd[:-1] < 0) & (dd[1:].abs() > 0.5) & (dd[:-1].abs() > 0.5)).float().mean()
    yr = _YR[:, msk].T[:, None]
    yrs = torch.nn.functional.avg_pool1d(yr, 40, 1, padding=20, count_include_pad=False)[:, 0, :yr.shape[-1]]
    return float(dd.abs().mean()), float(zz) * 100, float((yr[:, 0] - yrs)[:, 20:-20].std())
_wa, _we = _wob(torch.ones_like(cases[0][1])), _wob(cases[0][1])
if LIMR:
    _lr = torch.cat(LIMR); _tb = torch.cat(TBS)
    print(f"{tag}  plan의 0.1 s 점이 타이어 한계(조향 끝에서 버티는 횡가속도)의 80/90/95 % 넘는 결정: "
          f"{float((_lr > 0.8).float().mean()) * 100:.0f} / {float((_lr > 0.9).float().mean()) * 100:.0f} / "
          f"{float((_lr > 0.95).float().mean()) * 100:.0f} % | 그 횡가속도에 차가 필요한 조향(정상 상태 표) 18/20/22 deg 넘음: "
          f"{float((_tb > 18).float().mean()) * 100:.0f} / {float((_tb > 20).float().mean()) * 100:.0f} / "
          f"{float((_tb > 22).float().mean()) * 100:.0f} % (조향 끝 {math.degrees(float(env.s_max)):.0f} deg)", flush=True)
print(f"{tag}  차의 실제 조향: 0.1 s마다 바뀌는 크기 전체 {_wa[0]:.2f} deg · 방향 바로 뒤집힘 {_wa[1]:.0f} % · 요 각속도 떨림 {_wa[2]:.1f} deg/s"
      f" | 빈 트랙 {_we[0]:.2f} deg · {_we[1]:.0f} % · {_we[2]:.1f} deg/s", flush=True)
