"""Does the delay-aware training execution count one wall hit many times? (11-09 08:20, plan25d/e: v3 cars ran into
walls 5-40x as often as v2 early and learned to stand still.) Same straight commands (steering 0, speed 1.5 m/s) for 4 s
to every car, the empty track: the training execution of v2 (`enable_ideal()`: a contact caps the speed until the next
command) and of v3 (`enable_ideal(lag=True)`: the speed loop keeps pushing). Per car that touched a wall: how many wall
contacts were counted, and the contact cost `PlanReward` would charge. args: envs"""
import sys, argparse, json
import torch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim.learn import state_rl as SR
from f1sim.learn.graph_runtime import prepare_graph_runtime
from f1sim.learn.plan_rl import WORLD
from f1sim.learn.plan_reward import PlanReward
from f1sim.learn.plan_policy import plan3_models
from f1sim import plan_steps as PS
from f1sim import map_controller as M
from f1sim.sim import CONTACT_FRAME_COLUMNS as CC

torch.set_grad_enabled(False)
n_env = int(sys.argv[1])
dev = torch.device("cuda")
for lag in (False, True):
    a = argparse.Namespace(**json.load(open(WORLD)))
    a.envs, a.seed, a.device = n_env, 3, "cuda"
    a.warps, a.prop_fraction, a.removal_p, a.opp_present, a.episode_s = 0, 0.0, 0.0, 0.0, 30.0
    env = SR.build_env(a, dev)
    lim = PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)
    env.enable_ideal(lag=lag, turn=plan3_models(env, lim, dev)[1] if lag else None)
    prepare_graph_runtime(env, log=lambda _t: None)
    env.reset(seed=3)
    lid = env.learner_ids; B = int(lid.numel())
    pr = PlanReward(env, 0.995, dev)
    act0 = torch.zeros(env.B, env.act_dim, device=dev)
    i_w = CC.index("onsets_wall")
    cf0 = env.sim.contact_frame()[lid][:, i_w].float()
    cost = torch.zeros(B, device=dev)
    for k in range(160):                       # 4 s
        env.ideal_cmd[lid, 0] = 0.0; env.ideal_cmd[lid, 1] = 1.5
        if k % 4 == 0:
            env.sim.ideal_vcap[lid] = 1e3          # a new command every 0.1 s, as the training sends
        env.step(act0)
        pr.control_step()
        if k % 4 == 3:
            cost += -pr.decision(torch.zeros(B, PS.N_STEP, 2, device=dev))["contact"]
    n = env.sim.contact_frame()[lid][:, i_w].float() - cf0
    hit = n > 0
    print(f"[{'v3 지연 실행' if lag else 'v2 실행'}] 차 {B}대 중 벽에 닿은 차 {int(hit.sum())}대 | 닿은 차 한 대당 벽 접촉 수 중앙 "
          f"{float(n[hit].median()) if hit.any() else 0:.0f} (최대 {float(n.max()):.0f}) | 접촉 감점 합 중앙 {float(cost[hit].median()) if hit.any() else 0:.3f}", flush=True)
