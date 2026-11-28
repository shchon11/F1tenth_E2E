"""Audit C wrapper: runs the prior developer's exam_record.py UNMODIFIED (runpy), optionally with a harsher physics
randomisation (C_STRESS=1) patched into common.make_env, and prints the per-car physics draws actually used.
Args are exam_record.py's: ckpt out.npz mode n_env count secs rm_t
C_STRESS=1 ranges (multiplicative for scale fields, absolute otherwise):
  vehicle.mu 0.80-1.05 (x nominal; training/exam default pins 1.0), actuator.cmd_delay 0.04-0.08 s (default 0.02-0.05),
  actuator.servo_tau 0.04-0.09 s (default 0.025-0.07), lidar.noise_std 0.008-0.015 m (default 0.005-0.011)."""
import os, sys, runpy
TOOLS = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools"
sys.path.insert(0, TOOLS)
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
import torch
import f1sim.learn.common as C

STRESS = os.environ.get("C_STRESS", "") == "1"
STRESS_RANGES = {"vehicle.mu": (0.80, 1.05), "actuator.cmd_delay": (0.04, 0.08),
                 "actuator.servo_tau": (0.04, 0.09), "lidar.noise_std": (0.008, 0.015)}
_orig = C.make_env


def make_env(*a, **k):
    cfg = k.get("cfg")
    if STRESS and cfg is not None:
        cfg.rand.ranges.update(STRESS_RANGES)
        print(f"[C] STRESS ranges {STRESS_RANGES}", flush=True)
    env = _orig(*a, **k)
    try:
        P = env.sim.P
        for key in ("mu", "mu_f_scale", "cmd_delay", "servo_tau", "noise_std", "steer_bias", "a_max"):
            if key in P and torch.is_tensor(P[key]):
                x = P[key].float()
                print(f"[C] physics {key}: min {float(x.min()):.4f} mean {float(x.mean()):.4f} max {float(x.max()):.4f}", flush=True)
        print(f"[C] rand.enabled={cfg.rand.enabled if cfg is not None else '?'}", flush=True)
    except Exception as e:                                            # noqa
        print(f"[C] physics print failed: {e}", flush=True)
    return env


C.make_env = make_env
sys.argv = [os.path.join(TOOLS, "exam_record.py")] + sys.argv[1:]
runpy.run_path(os.path.join(TOOLS, "exam_record.py"), run_name="__main__")
