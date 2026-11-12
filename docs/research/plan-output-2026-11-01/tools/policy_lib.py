"""policy 안을 보는 분석 도구들이 같이 쓰는 것 (mem_link.py, box_react.py): 차의 한계(plan 펼치기)와 기록된 입력에서 plan 다시 계산."""
import os, json, argparse
import torch

dev = torch.device("cpu")


def limits_for(ck):
    """the car's limits (plan roll-out) from a two-car env of the real map, built as the exam builds it"""
    from f1sim.learn.plan_policy import lidar_mount_of
    mx = lidar_mount_of(ck)
    if mx > 0:
        os.environ["F1SIM_LIDAR_MOUNT_X"] = str(mx)
    else:
        os.environ.pop("F1SIM_LIDAR_MOUNT_X", None)
    from f1sim.learn import state_rl as SR
    from f1sim.learn.plan_rl import WORLD
    from f1sim import plan_steps as PS
    from f1sim import map_controller as M
    a = argparse.Namespace(**json.load(open(WORLD)))
    a.envs, a.seed, a.device = 2, 0, "cpu"
    a.warps, a.track_scales = 0, None
    a.prop_fraction, a.opp_present = 0.0, 0.0
    env = SR.build_env(a, dev)
    return PS.Limits.from_nominal(env.tracker.nominal, env.tracker.map.lut, M.GRIP_LEVELS.index(1.0), dev)


def plan_of(actor, lim, f, h0, h1=None):
    """the plan of a `mem_front` memory policy from its inputs f (scan encoding 256 + normalised state) and memory h0 (or the
    new memory h1 given): (lat at 1.0 s [m], speed at 1.0 s [m/s], mean (n, 20), lat at 0.5 s [m]). Rolled out from a
    straight wheel: for comparing two plans of the same moment only."""
    from f1sim import plan_steps as PS
    if h1 is None:
        h1 = actor.gru(f, h0)
    g = actor.trunk(torch.cat([f, h1], 1))
    mean, _, v_hat = actor.heads(g)
    p = PS.roll(torch.tanh(mean), v_hat.clamp(-PS.V_REV, PS.V_FWD), torch.zeros_like(v_hat), lim)
    return p["pts"][:, -1, 1], p["pv"][:, -1], mean, p["pts"][:, 4, 1]
