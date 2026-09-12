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
        #: Slowest speed this driver will ever command. 0.5 m/s keeps a lone car rolling, which is
        #: what the avoidance and pass references want. A driver that has to hold station behind a
        #: STOPPED car needs to be able to reach zero, and sets this to 0.
        self.v_floor = 0.5

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
        v_tgt = v_tgt.clamp(self.v_floor, float(e.ecfg.v_max_policy))

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


class TrafficExpert:
    """Reference driver for the T (traffic) family: come through traffic clean, pass where it fits.

    Built on the **raceline teacher**, not on the centreline pure-pursuit the other two experts use,
    and measured that is the whole difference. The pure-pursuit reference follows the centreline with
    a curvature speed limit; on the held-out floors it drove into the track 0/4 on four of the five
    maps before an opponent was involved at all. The teacher drives an optimised line with a speed
    profile derived from the actual friction, and completes 16/16 clean laps on both real floors.
    A scenario check is only evidence if the driver can drive; "the reference crashed" says nothing
    about the scenario.

    What this adds to the teacher is the two things a teacher does not have -- it is blind to other
    cars apart from the follow-gap slowdown, which is the whole reason opponents are boring:

    * **going round**, by planning through a lateral offset. `RacelineTeacher.plan_action` already
      takes one and clamps it per raceline point against the track's own distance field, which is
      the same mechanism a scripted `shift` event uses; so a 0.45 m move through a 1.4 m section
      becomes as much of one as fits, and never a wall.
    * **holding station**, by scaling the plan's commanded speeds toward the car ahead's. Where the
      clamp falls below the width a car needs to get past another, this driver does not go: it sits
      behind. That is the normal case on a 0.70 m half-lane and it has to be survivable, because
      the alternative -- attempting the pass anyway -- is what took the O family off those floors.

    Deterministic, checkpoint-free, and derived from the scenario's own geometry. A feasibility
    reference, never a leaderboard entry.
    """

    def __init__(self, env, *, engage_frac: float = 0.12, engage_max_m: float = 8.0,
                 engage_min_m: float = 2.5, desired_offset_m: float = 0.45,
                 min_pass_offset_m: float = 0.32, follow_gap_m: float = 1.8,
                 release_m: float = 1.2, side: int = 1, margin_m: float = 0.10,
                 follow_decel: float = 3.0):
        from f1sim.opponent_events import raceline_offset_limit
        if env.teacher is None:
            raise ValueError("TrafficExpert drives the raceline teacher; this env has none. A T "
                             "cell always has one -- it is what drives the opponents.")
        self.env = env
        self.teacher = env.teacher
        self.side = int(side)
        self.desired = float(desired_offset_m)
        self.min_pass = float(min_pass_offset_m)
        self.follow_gap = float(follow_gap_m)
        self.release = float(release_m)
        self.follow_decel = float(follow_decel)
        # Engage over a distance proportional to the lap, so the manoeuvre is the same fraction of a
        # 33 m hairpin and a 68 m circuit. The fixed 7 m window of `PassExpert` is a fifth of the
        # first and a tenth of the second, which is why it spent most of a `map16x07` lap holding an
        # offset line rather than briefly going round a car.
        self.engage = (env.sim.track.length * float(engage_frac)).clamp(float(engage_min_m),
                                                                       float(engage_max_m))
        # The clamp the teacher applies. The env only builds this when scripted events are on, so a
        # no-event T cell would otherwise have an unclamped teacher and this driver could ask for an
        # offset the lane does not have.
        self.limit = raceline_offset_limit(self.teacher, env.sim.track,
                                           0.5 * float(env.cfg.vehicle.width), float(margin_m))
        if self.teacher.offset_limit is None:
            self.teacher.offset_limit = self.limit
        self._w = torch.zeros(env.B, device=env.device)

    # -- what the traffic ahead is doing ---------------------------------------------------------
    def _ahead(self):
        """(arc to the nearest car ahead, its speed). The arc is +1e9 where nothing is ahead."""
        e = self.env
        if e.sim.other_idx is None:
            return (torch.full((e.B,), 1e9, device=e.device),
                    torch.zeros(e.B, device=e.device))
        d = e.signed_gaps(e.sim.s, e.sim.tid)                      # (B, M-1), + = ahead of me
        ahead = torch.where(d > 0, d, torch.full_like(d, 1e9))
        gap, j = ahead.min(1)
        v = e.sim.state[e.sim.other_idx.gather(1, j[:, None])[:, 0], 3]
        return gap, v

    def _room(self, idx):
        """Achievable |offset| at each car's own raceline point, capped at what it wants."""
        lim = self.limit[self.env.sim.tid, idx]
        return torch.minimum(lim, torch.full_like(lim, self.desired))

    # -- the manoeuvre ----------------------------------------------------------------------------
    def __call__(self, obs=None, k=None, env=None) -> torch.Tensor:
        e = self.env
        idx, _ = self.teacher.project(e.sim.state[:, :2], e.sim.tid)
        room = self._room(idx)
        fits = room >= self.min_pass
        gap, v_other = self._ahead()

        # Hysteresis, as in `PassExpert`: once committed, hold the line out until genuinely through.
        # The gate is the room the LANE has, so a section that cannot take the offset never engages
        # and the driver falls through to holding station instead.
        engaging = ((gap < self.engage[e.sim.tid]) & fits).float()
        self._w = torch.maximum(engaging, self._w * (gap < 1e8).float() * fits.float())
        offset = self.side * room * self._w

        an = self.teacher.plan_action(e.sim.state, e.sim.P, e.sim.tid, e.ecfg.v_max_policy,
                                      e.tracker.spec, offset=offset)

        # Hold station behind a car this driver is not going round. The trigger carries the room to
        # brake at the current closing speed, the shape the env's own `follow_cap` uses: a fixed
        # distance is a rear-end waiting for a fast approach. Zero inside the body gap, because a
        # car a `stop` event has parked is a wall that happens to be a car.
        mine = e.sim.state[:, 3]
        closing = (mine - v_other).clamp_min(0.0)
        trigger = self.follow_gap + closing * closing / (2.0 * self.follow_decel)
        holding = (gap < trigger) & (self._w < 0.5)
        v_hold = torch.where(gap < self.follow_gap * 0.55, torch.zeros_like(v_other), 0.9 * v_other)
        scale = torch.where(holding,
                            (v_hold / max(float(e.ecfg.v_max_policy), 1e-6)).clamp(0.0, 1.0),
                            torch.ones_like(v_other))
        # The plan's two speed entries are normalized to [-1, 1]; scaling them the way
        # `gym_env._opponent_actions` scales an event's speed keeps one convention for "drive this
        # plan slower" rather than inventing a second.
        an = an.clone()
        target = ((an[:, -2:] + 1) * 0.5 * float(e.ecfg.v_max_policy))
        target = torch.where(holding[:, None],
                             torch.minimum(target, (scale * float(e.ecfg.v_max_policy))[:, None]),
                             target)
        an[:, -2:] = (target / max(float(e.ecfg.v_max_policy), 1e-6) * 2 - 1).clamp(-1.0, 1.0)
        return an.clamp(-1.0, 1.0)
