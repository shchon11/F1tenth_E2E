"""Scripted, non-candidate reference drivers.

The raceline teacher is blind to obstacles and to other cars (`gym_env.py:194`), so driving it shows
that an obstacle blocks the line and nothing more. It cannot show that a scenario is *dynamically
feasible* -- that some driver can clear the obstacle, or complete a pass, inside the plant's real
limits. These experts close that gap.

Both follow a reference line with pure pursuit and emit a real plan through `mpc.encode`, so the
action is the same object a policy produces and goes through the same tracker. The reference line is
the track centreline displaced laterally: toward the corridor the geometry proof verified, or around
the car being passed. Displacing the *line* is what makes this work -- an earlier version biased the
curvature knots directly, which makes the car turn continuously rather than move over, and it drove
into the wall every time.

Deterministic, checkpoint-free, and derived from the scenario's own geometry. Feasibility
references, never leaderboard entries.
"""
from __future__ import annotations

import torch

from f1sim import mpc

#: Pure-pursuit lookahead: L = clamp(gain * v, lo, hi).
LOOKAHEAD_GAIN, LOOKAHEAD_MIN, LOOKAHEAD_MAX = 0.9, 1.2, 4.0
#: Lateral acceleration the reference driver is willing to use when picking a speed.
A_LAT_REF = 5.0


def _signed_to(s, target: float, length: float):
    return (s - target + length / 2.0) % length - length / 2.0


class _PurePursuit:
    """Follows `centreline + lateral_offset(s)` and emits a plan action."""

    def __init__(self, env, *, v_ref: float = 4.0):
        self.env = env
        self.v_ref = float(v_ref)
        t = env.sim.track
        self.cl = t.cl                                   # (T, N, 2)
        self.tangent = t.cl_tangent                      # (T, N, 2)
        self.n_pts = self.cl.shape[1]
        self.length = env.sim.track.length               # (T,)

    # -- subclasses say how far to move off the line, per env -----------------------------------
    def offset(self, s_ahead: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(s_ahead)

    def speed_scale(self) -> torch.Tensor:
        return torch.ones(self.env.B, device=self.env.device)

    def __call__(self, obs=None, k=None, env=None) -> torch.Tensor:
        e = self.env
        tid, idx = e.sim.tid, e.sim.cl_idx
        state = e.sim.state
        v = state[:, 3].abs()
        Ld = (LOOKAHEAD_GAIN * v).clamp(LOOKAHEAD_MIN, LOOKAHEAD_MAX)

        L = self.length[tid]
        step = (L / self.n_pts).clamp_min(1e-6)
        ahead = (idx + (Ld / step).round().long()) % self.n_pts
        s_ahead = e.sim.s + Ld

        p = self.cl[tid, ahead]                          # (B,2) point on the line
        tg = self.tangent[tid, ahead]
        nrm = torch.stack([-tg[:, 1], tg[:, 0]], 1)
        target = p + nrm * self.offset(s_ahead)[:, None]

        d = target - state[:, :2]
        yaw = state[:, 2]
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        x_b = d[:, 0] * c + d[:, 1] * s_                 # body frame
        y_b = -d[:, 0] * s_ + d[:, 1] * c
        dist2 = (x_b ** 2 + y_b ** 2).clamp_min(1e-3)
        kappa = (2.0 * y_b / dist2).clamp(-mpc.PlanSpec().kappa_max, mpc.PlanSpec().kappa_max)

        spec = e.tracker.spec
        v_curve = torch.sqrt(A_LAT_REF / kappa.abs().clamp_min(0.05))
        v_tgt = torch.minimum(torch.full_like(v_curve, self.v_ref), v_curve) * self.speed_scale()
        v_tgt = v_tgt.clamp(0.5, float(e.ecfg.v_max_policy))

        knots = kappa[:, None].expand(-1, mpc.N_KNOTS).contiguous()
        return mpc.encode(knots, v_tgt, v_tgt, float(e.ecfg.v_max_policy), spec).clamp(-1.0, 1.0)


class AvoidanceExpert(_PurePursuit):
    """Steps into the corridor the geometry proof verified, then returns to the line.

    `free_side` and `corridor_offset_m` come from the proven placement, so the manoeuvre is derived
    from the scenario rather than fitted to any outcome.
    """

    def __init__(self, env, *, s_obs_m: float, free_side: int, corridor_offset_m: float = 0.45,
                 window_m: float = 12.0, v_ref: float = 2.0, approach_v_ref: float = 1.6):
        super().__init__(env, v_ref=v_ref)
        self.s_obs, self.side = float(s_obs_m), int(free_side)
        self.amp, self.window = float(corridor_offset_m), float(window_m)
        # A wider window starts the displacement earlier, and a lower approach speed gives the
        # tracker the time to achieve it. Measured: at 3 m/s into a 0.6 m corridor the car arrives
        # still crossing the line and clips the box.
        self.approach_v_ref = float(approach_v_ref)

    def _window_weight(self, s_ref):
        L = self.length[self.env.sim.tid]
        ds = (s_ref - self.s_obs + L / 2) % L - L / 2
        # raised cosine: on and off smoothly, so the tracker is never asked for a step change
        w = (1.0 - (ds.abs() / self.window).clamp(0.0, 1.0))
        return 0.5 * (1.0 - torch.cos(torch.pi * w))

    def offset(self, s_ahead):
        return self.side * self.amp * self._window_weight(s_ahead)

    def speed_scale(self):
        w = self._window_weight(self.env.sim.s)
        return (1.0 - w) + w * (self.approach_v_ref / max(self.v_ref, 1e-6))


class PassExpert(_PurePursuit):
    """Pulls off the line to go around the car ahead, then comes back once clearly through."""

    def __init__(self, env, *, engage_m: float = 7.0, offset_m: float = 0.40,
                 release_m: float = 1.2, side: int = 1, v_ref: float = 5.0):
        super().__init__(env, v_ref=v_ref)
        self.engage, self.amp = float(engage_m), float(offset_m)
        self.release, self.side = float(release_m), int(side)
        self._w = torch.zeros(env.B, device=env.device)

    def _weight(self):
        e = self.env
        if e.sim.other_idx is None:
            return torch.zeros(e.B, device=e.device)
        g = e.signed_gaps(e.sim.s, e.sim.tid)[:, 0]      # + = opponent ahead of me
        engaging = ((g > -self.release) & (g < self.engage)).float()
        # hysteresis: hold the line out until genuinely through, so the pass is not abandoned
        self._w = torch.maximum(engaging, self._w * (g > -self.release).float())
        return self._w

    def offset(self, s_ahead):
        return self.side * self.amp * self._weight()

    def speed_scale(self):
        return 1.0 + 0.25 * self._weight()
