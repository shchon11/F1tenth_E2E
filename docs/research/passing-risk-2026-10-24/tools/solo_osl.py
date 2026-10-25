"""Research (2026-10-25): the speed limit for a car ahead in SOLO laps (props on the line): alone on the track it must never bite.
Counts the learner steps where the env held the speed command (`env.opponent_limited`) and reports the finals' solo numbers.
LATCH=0.5 adds the latched lane check (latch_patch.py). usage: solo_osl.py CKPT OUT_DIR SEEDS"""
import os
import sys

import torch

from f1sim import gym_env
from f1sim.learn import evaluate as ev
if os.environ.get("LATCH"):
    import latch_patch  # noqa: F401

CK, OUT = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
os.makedirs(OUT, exist_ok=True)
STAT = {"steps": 0, "held": 0}
_orig = gym_env.F1VecEnv.step


def step(self, action):
    out = _orig(self, action)
    if getattr(self, "opponent_limited", None) is not None:
        STAT["steps"] += int(self.learner.sum()); STAT["held"] += int((self.opponent_limited & self.learner).sum())
    return out


gym_env.F1VecEnv.step = step
for seed in SEEDS:
    out = os.path.join(OUT, f"solo_s{seed}.json")
    if os.path.exists(out):
        continue
    sys.argv = ["evaluate", CK, "--action-mode", "plan", "--tracks", "real:korea_2026_competition", "--envs", "32", "--steps", "2400",
                "--budget-laps", "0", "--teacher-a-lat", "7.0", "--teacher-a-acc", "6.5", "--teacher-a-brake", "4.0",
                "--raceline-objective", "min_time", "--dial-offset", "0.0", "--speed-cap", "9.0", "--device", "cuda:0",
                "--seed", str(seed), "--graphs", "--collision-mode", "soft", "--movable-obstacles", "--spawn-runway", "3.0",
                "--race-size", "1", "--procedural-obstacles", "1.0", "--procedural-density", "1.5",
                "--procedural-max-props", "10", "--procedural-raceline-corridor", "off", "--output", out,
                "--dial-value", "1.0489", "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92",
                "--opponent-speed-limit"]
    ev.main()
    print(f"seed {seed}: speed held by the limit on {STAT['held']} of {STAT['steps']} learner steps "
          f"({100 * STAT['held'] / max(STAT['steps'], 1):.3f} %)", flush=True)
