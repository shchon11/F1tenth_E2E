"""운전하는 쪽이 기억을 읽나 (11-11, 사용자: "으엥 왜저럴까" -- 바퀴마다 기억을 지워도 바퀴 시간이 같고 접촉은 기억을 둘 때 더 많음).
학습 세계(plan_rl.build), 학습 조건, 혼자, 상자 계속(치우지 않음), 통째 에피소드. 결정마다 같은 입력으로 policy를 두 번: 기억 그대로(실제로 모는
plan) / 기억을 0으로 지운 plan(반사실, 몰지는 않음). 두 plan의 차이: 0.5 s·1.0 s 점의 옆 자리(차 기준) [cm], 0.5 s 점의 속도 [m/s].
결정을 나눔: 트랙을 따라 앞 R m 안 가장 가까운 상자가 벽에 가려 안 보임 / 보임 / 없음 x 첫 바퀴(에피소드 6 s 전) / 다음 바퀴들(9 s 뒤).
안 보이는 상자 앞 다음 바퀴에서 기억이 plan을 그 상자에서 멀어지는 쪽으로 옮기면(옆 차이를 상자 쪽 + / 반대쪽 -로 잼) 기억을 쓰는 것.
인자: checkpoint [환경 수, 기본 256] [초, 기본 60]"""
import os, sys
os.environ["F1SIM_NO_STAGGER"] = "1"      # whole episodes: no staggered first-episode restarts (11-10)
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import PlanObs2, load_actor2, plan_v2
from f1sim.learn.plan_reward import PlanReward
from f1sim import plan_steps as PS

torch.set_grad_enabled(False)
ck_path = sys.argv[1]
n_env = int(sys.argv[2]) if len(sys.argv) > 2 else 256
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
dev = torch.device("cuda")
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if (ck.get("args") or {}).get("plan_lat_margin") is not None:
    os.environ["F1SIM_PLAN_LAT_MARGIN"] = str(ck["args"]["plan_lat_margin"])
env, lim, world = PR.build(PR.WORLD, n_env, 13, dev, fmt="v2", overrides={
    "prop_fraction": 1.0, "opp_present": 0.0, "removal_p": 0.0, "episode_s": secs + 5.0})
lid = env.learner_ids; Bl = int(lid.numel())
actor = load_actor2(ck, dev)
R = float(ck["args"].get("box_range") or 12.0)
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
tid = env.sim.tid[lid]
NS = 24
tt = torch.linspace(0.05, 0.92, NS, device=dev)
dt_ctrl = float(env.sim.control_dt)
I5, I10 = PS.SUB * 5, PS.SUB * 10                                     # the dense plan's 0.5 s and 1.0 s points
keys = [f"{a}_{b}" for a in ("hid", "vis", "none") for b in ("first", "later")]
acc = {k: torch.zeros(6, device=dev) for k in keys}                   # n, |dy5|, |dy10|, |dv5|, toward-box dy5, toward-box dy10
for kd in range(int(secs / PS.DT)):
    tb = PR.track_boxes(env, pr.goal, lid, 3, R).view(Bl, 3, 4)
    st = env.sim.state[lid].float()
    present = tb[:, 0, 0] > 0.5
    dx, dy = tb[:, 0, 1] * 5.0, tb[:, 0, 2] * 5.0
    c, s_ = torch.cos(st[:, 2]), torch.sin(st[:, 2])
    bx, by = st[:, 0] + dx * c - dy * s_, st[:, 1] + dx * s_ + dy * c
    px = st[:, 0:1] + (bx - st[:, 0])[:, None] * tt; py = st[:, 1:2] + (by - st[:, 1])[:, None] * tt
    rem = pr.goal.remaining(torch.stack([px, py], -1).reshape(-1, 2), tid.repeat_interleave(NS)).view(Bl, NS)
    hidden = present & torch.isnan(rem).any(1)
    visible = present & ~hidden
    h_in = obs.mem_h.clone() if getattr(obs, "mem_h", None) is not None else None
    # the counterfactual first (the real call below steps the memory): the same inputs, the memory wiped
    if h_in is not None:
        mean0, _, v0 = actor(obs.scans, obs.state(), torch.zeros_like(h_in))
        p0 = PS.roll(torch.tanh(mean0), v0.clamp(-PS.V_REV, PS.V_FWD), obs.cmd_d, lim)
    _, mean, _, v_hat, p = plan_v2(actor, obs, lim)
    if h_in is not None:
        dy5 = p["y"][:, I5] - p0["y"][:, I5]; dy10 = p["y"][:, I10] - p0["y"][:, I10]
        dv5 = p["v"][:, I5] - p0["v"][:, I5]
        side = torch.sign(dy)                                         # + = the box is on the car's left
        t_ep = env.ep_step[lid].float() * dt_ctrl
        first, later = t_ep < 6.0, t_ep >= 9.0
        for k, m in (("hid_first", hidden & first), ("hid_later", hidden & later), ("vis_first", visible & first),
                     ("vis_later", visible & later), ("none_first", ~present & first), ("none_later", ~present & later)):
            mf = m.float()
            acc[k] += torch.stack([mf.sum(), (dy5.abs() * mf).sum(), (dy10.abs() * mf).sum(), (dv5.abs() * mf).sum(),
                                   (dy5 * side * mf).sum(), (dy10 * side * mf).sum()])
    drv.decision(torch.tanh(mean), p)
name = f"{os.path.basename(os.path.dirname(ck_path))}/{os.path.basename(ck_path)}"
print(f"{name}: 학습 조건, 혼자, 상자 계속, 학습 차 {Bl}대 x {secs:.0f} s -- 기억 그대로 plan과 기억을 지운 plan의 차이(같은 입력)", flush=True)
names = {"hid": "벽에 가려 안 보이는 상자 앞", "vis": "보이는 상자 앞", "none": f"앞 {R:g} m에 상자 없음"}
for a in ("hid", "vis", "none"):
    for b, bn in (("first", "첫 바퀴"), ("later", "다음 바퀴들")):
        v = acc[f"{a}_{b}"]; n = max(float(v[0]), 1.0)
        print(f"  {names[a]}, {bn} ({int(v[0])}번): 옆 자리 차이 0.5 s {float(v[1]) / n * 100:.1f} cm, 1.0 s {float(v[2]) / n * 100:.1f} cm | "
              f"속도 차이 0.5 s {float(v[3]) / n:.2f} m/s | 상자 쪽(+)/반대쪽(-)으로 옮김 0.5 s {float(v[4]) / n * 100:+.1f} cm, "
              f"1.0 s {float(v[5]) / n * 100:+.1f} cm", flush=True)
