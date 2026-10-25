"""How often the speed limits held the evaluated cars' speed command (2026-10-26).

The speed limit for a car ahead (`opponent_speed_limit`) and for an obstacle on the plan's path (`obstacle_speed_limit`)
cut the learner's speed command; the env marks the rows they cut (`F1VecEnv.opponent_limited`, `.obstacle_limited`) for
the command `sim.step` is about to drive. This meter reads those marks on every `sim.step`, on the learner rows, and
reports the share of the learner exposure each limit (and either) held -- a policy that expects a limit should run into
it less often than one that does not.
"""
from __future__ import annotations

import torch


class SpeedLimitMeter:
    def __init__(self, env):
        self.env = env
        dev = env.sim.state.device
        self.steps = torch.zeros((), device=dev)
        self.car = torch.zeros((), device=dev)
        self.obstacle = torch.zeros((), device=dev)
        self.either = torch.zeros((), device=dev)
        self._orig_step = None

    def __enter__(self):
        sim = self.env.sim
        self._orig_step = sim.step

        def step_spy(*a, **kw):
            self._observe()
            return self._orig_step(*a, **kw)

        sim.step = step_spy
        return self

    def __exit__(self, *exc):
        if self._orig_step is not None:
            self.env.sim.step = self._orig_step
            self._orig_step = None
        return False

    def _observe(self) -> None:
        env = self.env
        rows = env.learner
        car = rows & getattr(env, "opponent_limited", torch.zeros_like(rows))
        obs = rows & getattr(env, "obstacle_limited", torch.zeros_like(rows))
        self.steps += rows.sum()
        self.car += car.sum()
        self.obstacle += obs.sum()
        self.either += (car | obs).sum()

    def report(self) -> dict:
        n = float(self.steps)
        share = (lambda x: float(x) / n if n else None)
        ecfg = self.env.ecfg
        return {"speed_limits": {"car_on": bool(getattr(ecfg, "opponent_speed_limit", False)),
                                 "obstacle_on": bool(getattr(ecfg, "obstacle_speed_limit", False)),
                                 "car_share": share(self.car), "obstacle_share": share(self.obstacle),
                                 "either_share": share(self.either), "learner_steps": int(n)}}
