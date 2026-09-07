"""The deployment-side tracker (batch 1, CPU, eager) must reproduce the training-side tracker (batched,
CUDA-graph compiled) step by step: same plan history in -> same (steer, speed) out."""
import torch
from f1sim import Config
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.mpc import PlanTracker


def test_cpu_tracker_matches_env_tracker():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks, rls = common.load_tracks(["gen:competition:0"], racelines=True)
    cfg = Config(); cfg.rand.enabled = False
    env = common.make_env(tracks, 4, dev, EnvConfig(action_mode="plan", speed_cap=5.0), cfg=cfg, seed=3)
    teacher = common.make_teacher(rls, env); env.reset(seed=3)
    node = PlanTracker(1, "cpu", env.cfg.vehicle.lf + env.cfg.vehicle.lr, env.cfg.vehicle.s_max, env.ecfg.v_max_policy)
    i = 1; worst = (0.0, 0.0)
    for t in range(60):
        a = env.teacher_label(teacher)
        lr = env.last_result
        v_meas = lr.odom[i, 3].item(); yaw_rate = lr.imu[i, :, 2].mean().item()
        cmd_node = node(a[i:i + 1].cpu(), torch.tensor([v_meas]), torch.tensor([5.0]), torch.tensor([yaw_rate]),
                        delay=env.tracker_delay[i:i + 1].cpu())[0]
        env.step(a)
        cmd_env = env.last_cmd_raw[i].cpu()                      # before the env's per-car calibration
        worst = (max(worst[0], abs(float(cmd_node[0] - cmd_env[0]))), max(worst[1], abs(float(cmd_node[1] - cmd_env[1]))))
    assert worst[0] < 2e-3 and worst[1] < 0.05, worst        # rad, m/s
