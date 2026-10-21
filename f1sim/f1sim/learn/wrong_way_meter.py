"""Wrong-way driving of the evaluated cars: facing backwards along the lane while driving forwards.

2026-10-21: in the console a car bounced off a hose head-on drove the wrong way at 3-5 m/s for the rest of
the run (+5 -> -15 m of progress). The trials protocol counts time facing backwards
(`evaluation_metrics.TrialAccumulator`); the race protocol counted nothing. This meter reads every
`sim.step` before the env's auto-reset, on the learner rows, and reports for both protocols:

* ``seconds`` / ``share``: time spent driving the wrong way, and its share of the exposure;
* ``events`` / ``per_km``: onsets of it, and onsets per km driven.

"Wrong way" is cos(heading error to the lane) < 0 with the body-frame forward speed above `MIN_SPEED`. A car
backing out of a wedge (`PlanSpec.wedge_s`) faces the right way and moves backwards: not counted. A car
turned round at a standstill is not counted until it drives off that way.
"""
from __future__ import annotations

import math

import torch

#: [m/s] forward speed above which facing backwards counts as driving the wrong way
MIN_SPEED = 1.0


class WrongWayMeter:
    def __init__(self, env):
        self.env = env
        B = env.sim.state.shape[0]
        dev = env.sim.state.device
        self.was = torch.zeros(B, dtype=torch.bool, device=dev)
        self.seconds = torch.zeros((), device=dev)
        self.exposure = torch.zeros((), device=dev)
        self.metres = torch.zeros((), device=dev)
        self.events = torch.zeros((), dtype=torch.long, device=dev)
        self._orig_step = None

    def __enter__(self):
        sim = self.env.sim
        self._orig_step = sim.step

        def step_spy(*a, **kw):
            self.was &= sim.steps != 0                   # a row that (re)starts begins facing the right way
            r = self._orig_step(*a, **kw)
            self._observe(r)
            return r

        sim.step = step_spy
        return self

    def __exit__(self, *exc):
        if self._orig_step is not None:
            self.env.sim.step = self._orig_step
            self._orig_step = None
        return False

    def _observe(self, r) -> None:
        env, sim = self.env, self.env.sim
        st = r.state
        _, yaw_c = sim.track.pose_at_s(r.s, sim.tid)
        herr = torch.remainder(st[:, 2] - yaw_c + math.pi, 2 * math.pi) - math.pi
        rows = env.learner
        wrong = rows & (torch.cos(herr) < 0) & (st[:, 3] > MIN_SPEED)
        dt = float(sim.control_dt)
        self.events += (wrong & ~self.was).sum()
        self.seconds += wrong.sum() * dt
        self.exposure += rows.sum() * dt
        self.metres += (torch.linalg.vector_norm(st[:, 3:5], dim=1) * rows).sum() * dt
        self.was = wrong

    def report(self) -> dict:
        exposure, km = float(self.exposure), float(self.metres) / 1000.0
        seconds, events = float(self.seconds), int(self.events)
        return {"wrong_way": {"seconds": seconds, "share": seconds / exposure if exposure else None,
                              "events": events, "per_km": events / km if km else None, "km": km,
                              "min_speed_mps": MIN_SPEED}}
