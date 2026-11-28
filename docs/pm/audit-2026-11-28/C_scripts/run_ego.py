"""Audit C: a BASELINE drives the ego car on exactly the exam sessions of exam_record.py (same world, layouts, seed, physics
draws, removal, recording). exam_record.py runs unmodified in mode "map"; the env's ego rows get `set_ego_driver` (the
console's "내 차" path), whose (steer, speed) replaces the command after the tracker -- the policy's plan is computed but
never reaches the car. C_EGO = ftg | disparity | raceline (modular: min-time line + particle-filter localisation +
scan detection + MAP, `common.build_ego_driver(estimated=True)`). Settings: baseline_settings.json tuned for the
neighbouring venue map korea_2026_competition (none exist for korea26_scan). C_STRESS as run_exam.py.
Args: ckpt(any korea26scan plan checkpoint, only for the exam's world) out.npz n_env count secs rm_t"""
import os, sys, runpy
TOOLS = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools"
sys.path.insert(0, TOOLS)
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
import torch
import f1sim.learn.common as C
from f1sim.learn import state_rl as SR
from f1sim.learn import baseline_settings as BS

KIND = os.environ["C_EGO"]
SOLO = os.environ.get("C_EGO_RACE", "") != "1"
STRESS = os.environ.get("C_STRESS", "") == "1"
STRESS_RANGES = {"vehicle.mu": (0.80, 1.05), "actuator.cmd_delay": (0.04, 0.08),
                 "actuator.servo_tau": (0.04, 0.09), "lidar.noise_std": (0.008, 0.015)}
_mk = C.make_env


def make_env(*a, **k):
    cfg = k.get("cfg")
    if STRESS and cfg is not None:
        cfg.rand.ranges.update(STRESS_RANGES)
        print(f"[C] STRESS ranges {STRESS_RANGES}", flush=True)
    env = _mk(*a, **k)
    P = env.sim.P
    for key in ("mu", "cmd_delay", "servo_tau", "noise_std"):
        x = P[key].float()
        print(f"[C] physics {key}: min {float(x.min()):.4f} mean {float(x.mean()):.4f} max {float(x.max()):.4f}", flush=True)
    return env


C.make_env = make_env
_be = SR.build_env


def build_env(a, device, extra=None):
    env = _be(a, device, extra=extra)
    tuned = BS.lookup("korea_2026_competition", KIND, solo=SOLO)
    print(f"[C] ego {KIND} tuned setting (korea_2026_competition, solo={SOLO}): {tuned.get('source') if tuned else None}", flush=True)
    if KIND in C.EGO_PLANNER_KINDS:
        from f1sim.raceline import Raceline
        lim = C.teacher_limits(*[float(x) for x in tuned["limits"]])
        trs, _ = C.load_tracks([str(a.tracks).split(",")[0]], racelines=False)
        rls = [Raceline.build_cached(trs[0], cache_only=True, objective="min_time", **lim)]
        sp = dict(tuned.get("stack_params") or {}, grip="nominal")
        drv = C.build_ego_driver(env, KIND, "", None, rls=rls, limits=lim, stack_params=sp)
    else:
        drv = C.build_ego_driver(env, KIND, str(tuned.get("weights") or ""), dict(tuned.get("options") or {}))
    env.set_ego_driver(drv)
    print(f"[C] ego driver {type(drv).__name__} on rows {int(env.external_slot_rows[0].numel())}", flush=True)
    return env


SR.build_env = build_env
sys.argv = [os.path.join(TOOLS, "exam_record.py"), sys.argv[1], sys.argv[2], "map"] + sys.argv[3:]
runpy.run_path(os.path.join(TOOLS, "exam_record.py"), run_name="__main__")
