"""Do plans that know the control delay come true? (11-09, `plan_steps3`, the user: "저만큼 제어딜레이 가정해서 출력하게하면
되잖아"). No policy: every 1.0 s each car draws a random smooth command plan (steering a fraction of the grip, a speed
command 2-7 m/s) from its TRUE state (speed, servo angle, the command in effect), the car is sent that plan's 40
commands as they are -- no feedback -- and where it is every 0.1 s is compared with where the plan said. Windows with any
contact, and plans that would touch a wall or a box within their 1 s (a car already against one), are left out. Conditions:
  ideal    the training's execution (`Sim.ideal_lag`): the nominal command path, kinematic poses -- should be ~0
  nominal  the full physics on the nominal car (no randomised parameter): what tyres and wheel slip add
  random   the full physics with the training world's randomised car (motor_tau 0.1-0.3, a_max x0.6-1.2, ...)
For comparison, the same 1 s test with the v2 planner: the time follower 12-14 cm at 0.5-1 s, MAP 15 / 46 cm.
args: condition(ideal|nominal|random) envs plans"""
import os, sys, argparse, json
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_policy import plan_world
from f1sim.learn.plan_rl import WORLD
from f1sim import plan_steps as PS
from f1sim import plan_steps3 as P3
from f1sim import dynamics as dyn
from f1sim import map_controller as M
from f1sim.plan_follow import to_car
from f1sim.learn.plan_reward import PlanReward

torch.set_grad_enabled(False)
mode, n_env, n_plans = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
a = argparse.Namespace(**json.load(open(WORLD)))
a.envs, a.seed, a.device = n_env, 11, "cuda"
a.warps, a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0, 0.0, 0.0, 0.0, n_plans * 1.0 + 10.0
torch.manual_seed(11)
env = SR.build_env(a, dev)
TURN = (P3.Turning(env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), float(env.tracker.nominal["lf"]) + float(env.tracker.nominal["lr"]), dev)
        if os.environ.get("F1SIM_TURN", "lut") == "lut" else None)
if mode == "ideal":
    env.enable_ideal(lag=True, turn=TURN)                     # before the graphs are captured: the roll reads it then
prepare_graph_runtime(env, log=lambda _t: None)
env.reset(seed=11)
lid = env.learner_ids; B = int(lid.numel())
if mode == "nominal":
    nom = env.tracker.nominal
    for k_, v_ in env.sim.P.items():
        if k_ in nom and torch.is_tensor(v_) and v_.shape[:1] == (env.B,):
            try:
                v_.fill_(float(nom[k_]))
            except (TypeError, ValueError):
                pass
    for k_ in ("steer_gain", "speed_gain"):
        if k_ in env.sim.P:
            env.sim.P[k_].fill_(1.0)
    for k_ in ("steer_bias", "motor_delay", "motor_ramp"):
        if k_ in env.sim.P:
            env.sim.P[k_].fill_(0.0)
    if "cmd_delay" in env.sim.P:
        env.sim.P["cmd_delay"].fill_(P3.Actuators().delay)
    _A = P3.Actuators()
    for k_, v_ in (("servo_tau", _A.servo_tau), ("motor_tau", _A.motor_tau), ("a_max", _A.a_max), ("a_brake", _A.a_brake),
                   ("v_switch", _A.v_switch), ("sv_max", _A.sv_max)):
        if k_ in env.sim.P:
            env.sim.P[k_].fill_(v_)
lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
act = P3.Actuators.nominal(env.tracker.nominal)
if mode != "ideal":
    env.ext_mask[lid] = True; env._ext_ids.update(int(i) for i in lid.tolist())
act0 = torch.zeros(env.B, env.act_dim, device=dev)
pr = PlanReward(env, 1.0, dev)
c_prev = torch.zeros(B, 2, device=dev)
g = torch.Generator(device=dev); g.manual_seed(5)
err = {h: [] for h in range(1, PS.N_STEP + 1)}
along = {h: [] for h in err}; across = {h: [] for h in err}; dv = {h: [] for h in err}; dpsi = {h: [] for h in err}
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC
i_on = [CC.index(k) for k in ("onsets_wall", "onsets_prop", "onsets_car")]
# a short warm-up: drive the first command plan without measuring (cars start standing)
for n in range(n_plans + 1):
    st = env.sim.state[lid].clone()
    # a driver that keeps the cars on the track: K random smooth command plans per car, the one that ends furthest
    # along (walls-only distance to the line) without touching a wall or a box
    K = 24
    base_t = torch.rand(B, K, 1, device=dev, generator=g) * 2.0 - 1.0
    a_turn = (base_t + 0.2 * torch.randn(B, K, PS.N_STEP, device=dev, generator=g)).clamp(-1, 1)
    v_tgt = 2.0 + 5.0 * torch.rand(B, K, 1, device=dev, generator=g)
    a_speed = ((v_tgt - (-PS.V_REV)) / (PS.V_FWD + PS.V_REV) * 2 - 1).expand(B, K, PS.N_STEP)
    rep = lambda t_: t_.repeat_interleave(K, 0)
    pk = P3.roll(torch.cat([a_turn, a_speed], 2).reshape(B * K, -1), rep(st[:, dyn.IVX]), rep(st[:, dyn.ISTEER]), rep(c_prev), lim, act, TURN)
    xk, yk, psik = plan_world(pk, rep(st))
    tid_k = rep(env.sim.tid[lid])
    idx10 = torch.arange(1, PS.N_STEP + 1, device=dev) * PS.SUB
    rem = pr.goal.remaining(torch.stack([xk[:, idx10], yk[:, idx10]], -1), tid_k[:, None].expand(-1, PS.N_STEP))
    d_end = rem[:, -1]
    hit_k = ~torch.isfinite(rem).all(1)                       # a point in a wall (the empty track: no boxes)
    score = torch.where(torch.isfinite(d_end), -d_end, torch.full_like(d_end, -1e6)) - 1e3 * hit_k.float()
    best = score.view(B, K).argmax(1)
    sel = torch.arange(B, device=dev) * K + best
    p = {k_: v_[sel] if torch.is_tensor(v_) and v_.shape[0] == B * K else v_ for k_, v_ in pk.items()}
    x, y, psi = plan_world(p, st)
    pts = torch.stack([x[:, PS.SUB::PS.SUB], y[:, PS.SUB::PS.SUB]], -1)     # (B, 10, 2)
    free = ~pr.plan_cost(x, y, psi, p["v"])["hit"].bool()             # the plan stays off walls and boxes for its 1 s
    cf0 = env.sim.contact_frame()[lid][:, i_on].float().sum(1)
    pos, sts = [], []
    for k in range(PS.N_STEP):
        for j in range(PS.SUB):
            jj = k * PS.SUB + j
            if mode == "ideal":
                env.ideal_cmd[lid, 0] = p["cmd_d"][:, jj]; env.ideal_cmd[lid, 1] = p["cmd_v"][:, jj]
            else:
                env.ext_cmd[lid] = to_car(env, lid, torch.stack([p["cmd_d"][:, jj], p["cmd_v"][:, jj]], 1))
            env.step(act0)
        pos.append(env.sim.state[lid, :2].clone()); sts.append(env.sim.state[lid].clone())
    c_prev = torch.stack([p["cmd_d"][:, -1], p["cmd_v"][:, -1]], 1)
    clean = ((env.sim.contact_frame()[lid][:, i_on].float().sum(1) - cf0) == 0) & free
    # a car that respawned inside the window (the first episode is staggered) jumped: no 0.1 s step of more than 1.5 m
    steps_ = torch.stack([st[:, :2]] + pos, 1)
    clean &= (steps_[:, 1:] - steps_[:, :-1]).norm(dim=-1).amax(1) < 1.5
    if n == 0:
        continue
    for h in range(1, PS.N_STEP + 1):
        e = (pos[h - 1] - pts[:, h - 1]).norm(dim=-1)
        err[h].append(e[clean])
        ph = psi[:, h * PS.SUB]; d_ = pos[h - 1] - pts[:, h - 1]
        along[h].append((d_[:, 0] * torch.cos(ph) + d_[:, 1] * torch.sin(ph))[clean])
        across[h].append((-d_[:, 0] * torch.sin(ph) + d_[:, 1] * torch.cos(ph))[clean])
        dv[h].append((sts[h - 1][:, dyn.IVX] - p["v"][:, h * PS.SUB])[clean])
        dpsi[h].append((torch.remainder(sts[h - 1][:, dyn.IYAW] - ph + 3.14159, 6.28318) - 3.14159)[clean])
n_ok = sum(int(t.numel()) for t in err[1])
print(f"[{mode}] 명령 plan {n_plans}개 x 차 {B}대, 접촉 없는 plan {n_ok}개 | plan이 말한 자리와 실제 위치 차이 중앙 (p90): "
      + ", ".join(f"{h * PS.DT:.1f} s {float(torch.cat(err[h]).median()) * 100:.1f} ({float(torch.cat(err[h]).quantile(0.9)) * 100:.1f})"
                  for h in (1, 3, 5, 7, 10)) + " cm", flush=True)
_m = lambda d_, h: float(torch.cat(d_[h]).median())
print(f"    실제 - plan (중앙): 진행 방향 " + ", ".join(f"{h * PS.DT:.1f} s {_m(along, h) * 100:+.1f}" for h in (3, 5, 10)) + " cm | 옆(왼쪽 +) "
      + ", ".join(f"{h * PS.DT:.1f} s {_m(across, h) * 100:+.1f}" for h in (3, 5, 10)) + " cm | 옆 |중앙| "
      + ", ".join(f"{float(torch.cat(across[h]).abs().median()) * 100:.1f}" for h in (3, 5, 10)) + " cm | 속도 "
      + ", ".join(f"{_m(dv, h):+.2f}" for h in (3, 5, 10)) + " m/s | 방향 |중앙| "
      + ", ".join(f"{float(torch.cat(dpsi[h]).abs().median()) * 57.3:.1f}" for h in (3, 5, 10)) + " deg", flush=True)
