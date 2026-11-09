"""Checks for plan holding in training (11-09, `plan_steps.shift_plan`, `plan_rl --hold-p`): (1) a plan moved k steps on
and placed at its own pose there lies on the original plan; (2) in the training's execution a plan driven 0.4 s through
`Driver.decision` (shifted every 0.1 s, from where the car is) puts the car where the plan said at 0.1 / 0.2 / 0.3 / 0.4
s -- the point of holding is that a held plan's later steps are really driven; (3) two short updates of plan_rl with
holding run and log.
args: runs_dir (scratch)"""
import os, sys
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import plan_world, PlanObs2
from f1sim.learn.plan_reward import PlanReward
from f1sim import plan_steps as PS
from f1sim import dynamics as dyn

torch.set_grad_enabled(False)
dev = torch.device("cuda")
env, lim, _ = PR.build(PR.WORLD, 32, 7, dev, overrides={"prop_fraction": 0.0, "opp_present": 0.0}, fmt="v2")
lid = env.learner_ids; B = int(lid.numel())
pr = PlanReward(env, 0.995, dev)
obs = PlanObs2(env, lim, pr.goal)
drv = PR.Driver(env, lim, pr, obs)
# let the cars get going: a few gentle plans
a0 = torch.zeros(B, PS.ACT_DIM, device=dev); a0[:, PS.N_STEP:] = 0.6
for _ in range(15):
    st = env.sim.state[lid]
    drv.decision(a0, PS.roll(a0, st[:, dyn.IVX], st[:, dyn.ISTEER], lim))
# (1) geometry
st = env.sim.state[lid].clone()
g = torch.Generator(device=dev); g.manual_seed(3)
a = (torch.rand(B, PS.ACT_DIM, device=dev, generator=g) * 2 - 1) * 0.8
p = PS.roll(a, st[:, dyn.IVX], st[:, dyn.ISTEER], lim)
xw, yw, pw = plan_world(p, st)
M = p["x"].shape[1]
for k in (1, 2, 3, 4):
    pk = PS.shift_plan(p, torch.full((B,), k, device=dev))
    i0 = k * PS.SUB
    sk = st.clone(); sk[:, 0], sk[:, 1], sk[:, dyn.IYAW] = xw[:, i0], yw[:, i0], pw[:, i0]
    xk, yk, _ = plan_world(pk, sk)
    n = M - i0
    e = torch.hypot(xk[:, :n] - xw[:, i0:], yk[:, :n] - yw[:, i0:]).max()
    print(f"(1) {k * 0.1:.1f} s 앞으로 옮긴 plan과 원래 plan의 위치 차이 최대 {float(e) * 100:.3f} cm, "
          f"명령 차이 {float((pk['cmd_d'][:, :n - 1] - p['cmd_d'][:, i0:]).abs().max()):.2e} rad", flush=True)
# (2) drive it 0.4 s, shifted every 0.1 s from where the car is
pts0 = torch.stack([xw[:, PS.SUB::PS.SUB], yw[:, PS.SUB::PS.SUB]], -1)
q = p
errs = []
alive = torch.ones(B, dtype=torch.bool, device=dev)
for k in range(4):
    q = PS.shift_plan(p, torch.full((B,), k, device=dev))
    out, _, _ = drv.decision(a, q, charge=torch.full((B,), k == 0, device=dev))
    alive &= ~out["restarted"]
    errs.append(float((env.sim.state[lid, :2] - pts0[:, k]).norm(dim=-1)[alive].median()) * 100)
print("(2) 한 plan을 0.4 s 몰았을 때 plan이 말한 자리와 실제 위치 차이 중앙 0.1/0.2/0.3/0.4 s: "
      + " / ".join(f"{e_:.2f}" for e_ in errs) + " cm", flush=True)
del env, drv, obs, pr
torch.cuda.empty_cache()
# (3) the trainer with holding
torch.set_grad_enabled(True)
PR.main(["--name", "hold_smoke", "--envs", "64", "--updates", "2", "--horizon", "16", "--hold-p", "0.5",
         "--runs", sys.argv[1]])
