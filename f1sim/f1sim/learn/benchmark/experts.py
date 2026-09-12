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

from .geom import contact_extent_m

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
                 engage_min_m: float = 2.5, desired_offset_m: float = 0.75,
                 min_pass_offset_m: float = 0.70, follow_gap_m: float = 2.2,
                 pass_speed_edge: float = 0.80, follow_gain: float = 1.0,
                 release_m: float = 1.2, side: int = 1, margin_m: float = 0.10,
                 follow_decel: float = 3.0, abreast_frac: float = 0.7,
                 speed_scale: float = 0.85, pass_speed_scale: float = 0.85,
                 lookahead_span_m: float = 3.0, offset_step_m: float = 0.05,
                 max_offset_m: float = 0.80):
        from f1sim.opponent_events import raceline_offset_limit
        if env.teacher is None:
            raise ValueError("TrafficExpert drives the raceline teacher; this env has none. A T "
                             "cell always has one -- it is what drives the opponents.")
        self.env = env
        self.teacher = env.teacher
        self.side = int(side)
        self.desired = float(desired_offset_m)
        self.min_pass = float(min_pass_offset_m)
        self.margin = float(margin_m)
        self.follow_gap = float(follow_gap_m)
        # Only go round a car this driver can actually get past: one doing less than this fraction
        # of its own current speed. Without it the driver committed to a manoeuvre it could not
        # finish and sat alongside for six seconds in a 2.2 m lane -- 4/4 car contacts on
        # gen:control:9100 with no wall involved. Two cars abreast leave 0.29 m of daylight at the
        # offset this driver uses; holding that for six seconds while both are steering is not a
        # pass, it is a rub waiting to happen, and no reference driver should be scripted to try it.
        self.pass_speed_edge = float(pass_speed_edge)
        #: [1/s] how hard the gap-keeping law pulls back toward `follow_gap_m`.
        self.follow_gain = float(follow_gain)
        self.release = float(release_m)
        #: Arc at which a manoeuvre is released: the worst-case nose-to-tail contact extent plus
        #: `release_m`. Inside it the two bodies can still touch, whichever side of the other each
        #: car's centre is on.
        self.release_hold = contact_extent_m(float(env.cfg.vehicle.length)) + float(release_m)
        self.follow_decel = float(follow_decel)
        # How much of the commanded offset has to have been ACHIEVED before this driver is allowed
        # to close on the car ahead. Measured: without it the driver commits to the pass, keeps its
        # speed while the tracker is still bringing it across, and arrives at the other car's
        # bumper still on the line -- 4/4 contacts on korea_2025_iccas and gen:competition:0, no
        # wall involved. A commanded offset is not a position.
        self.abreast_frac = float(abreast_frac)
        # Two fractions of the teacher's own speed profile: one always, one extra while off line.
        #
        # A reference driver is not a lap-time attempt. The profile is computed for the raceline at
        # the limit, and every OPPONENT in a T cell runs it scaled to 0.5-0.95 (`opp_speed_range`),
        # so the only car that would drive it flat out is this one. Measured, flat out is not
        # survivable: on gen:control:9100 at mu 0.73423 this driver went off the track 4/4 in a
        # scenario where it never pulled out to pass at all, while the opponents on the same line
        # at 0.8-0.95x completed the stint.
        #
        # A displaced line is a different radius, so a manoeuvre costs a little more again;
        # `AvoidanceExpert` carries the same term for the same reason -- "at 3 m/s into a 0.6 m
        # corridor the car arrives still crossing the line and clips the box".
        #
        # 0.85 leaves 28 % of margin in lateral acceleration (v^2 scaling) and still passes a car
        # doing 0.5-0.7x of the same profile, which is the scenario a pass is meant to be available
        # in. It does not reliably pass one at 0.8-0.95x, and it should not: that cell is carried by
        # pace and by staying clean, not by passes.
        self.speed_scale = float(speed_scale)
        self.pass_speed_scale = float(pass_speed_scale)
        # Engage over a distance proportional to the lap, so the manoeuvre is the same fraction of a
        # 33 m hairpin and a 68 m circuit. The fixed 7 m window of `PassExpert` is a fifth of the
        # first and a tenth of the second, which is why it spent most of a `map16x07` lap holding an
        # offset line rather than briefly going round a car.
        self.engage = (env.sim.track.length * float(engage_frac)).clamp(float(engage_min_m),
                                                                       float(engage_max_m))
        # The clamp the teacher applies. The env only builds this when scripted events are on, so a
        # no-event T cell would otherwise have an unclamped teacher and this driver could ask for an
        # offset the lane does not have.
        # How far this driver may go, measured two ways `raceline_offset_limit` does not.
        #
        # That helper is the right bound for a scripted `shift`: symmetric, evaluated at the car's
        # own point, and never more than 0.35 m of offset ramped over a second. For a 0.60 m pass it
        # is wrong twice over, and both showed up as crashes:
        #
        #  * it is OMNIDIRECTIONAL. The distance field is the free space in every direction, and the
        #    raceline is not centred in the lane, so on the roomy side it understates the space by
        #    the width of the tight side. Measured on these five maps, the per-side room at the same
        #    points is a median 0.60-0.80 m against the symmetric bound's 0.28-0.64 m.
        #  * it is evaluated WHERE THE CAR IS. The plan reaches several metres ahead and the car
        #    drives it, so on a map whose width changes quickly the offset is validated at a wide
        #    point and then carries the car into a narrow one. On gen:control:9100 -- hairpins and
        #    chicanes -- that was 3/4 collisions at full speed and 4/4 at 0.9x, the failures growing
        #    as the driver slowed and therefore spent LONGER off the line.
        #
        # So: sweep the actual side, in `offset_step_m` increments, and keep the largest displacement
        # whose footprint centre still has half a car plus a margin of clear space -- then take the
        # minimum of that over the next `lookahead_span_m` of arc, which is about what the plan
        # covers. Conservative in the direction that matters and honest about the direction it does
        # not need to be.
        self.limit = self._forward_min(
            self._side_limit(float(margin_m), float(offset_step_m), float(max_offset_m)),
            float(lookahead_span_m))
        #: The SYMMETRIC bound, which is what the teacher's own clamp is for: the opponents' shift
        #: events go through the same object and must keep the conservative both-ways version.
        self.symmetric_limit = raceline_offset_limit(
            self.teacher, env.sim.track, 0.5 * float(env.cfg.vehicle.width), float(margin_m))
        if self.teacher.offset_limit is None:
            # The env only builds this when scripted events are on, and the opponents need it.
            self.teacher.offset_limit = self.symmetric_limit
        self._w = torch.zeros(env.B, device=env.device)

    # -- what the traffic ahead is doing ---------------------------------------------------------
    def _ahead(self):
        """(arc to the nearest car ahead, its speed, arc to the nearest car either way).

        The third value is what a manoeuvre is released on. The first is +1e9 where nothing is
        ahead -- and the moment a pass goes through, that is exactly what happens, which is why the
        release cannot be keyed on it.
        """
        e = self.env
        if e.sim.other_idx is None:
            big = torch.full((e.B,), 1e9, device=e.device)
            return big, torch.zeros(e.B, device=e.device), big
        d = e.signed_gaps(e.sim.s, e.sim.tid)                      # (B, M-1), + = ahead of me
        ahead = torch.where(d > 0, d, torch.full_like(d, 1e9))
        gap, j = ahead.min(1)
        v = e.sim.state[e.sim.other_idx.gather(1, j[:, None])[:, 0], 3]
        return gap, v, d.abs().min(1).values

    def _side_limit(self, margin: float, step: float, max_offset: float):
        """(T, N) largest displacement toward `self.side` each raceline point tolerates.

        Swept rather than solved: the distance field gives clearance at a point, so the question
        "how far can I go this way" is answered by walking out along the normal until the clearance
        stops covering the car's half-width plus the margin. Monotone by construction -- once a step
        fails, no larger one is admitted -- so a pocket of space beyond a pinch is never claimed.
        """
        t = self.env.sim.track
        xy = self.teacher.xy.to(t.device)                                 # (T, N, 2)
        tan = self.teacher.tan.to(t.device)
        nrm = torch.stack([-tan[..., 1], tan[..., 0]], -1) * float(self.side)
        T_, N, _ = xy.shape
        tid = torch.arange(T_, device=t.device)[:, None].expand(T_, N)
        need = 0.5 * float(self.env.cfg.vehicle.width) + margin
        ok = torch.ones(T_, N, dtype=torch.bool, device=t.device)
        lim = torch.zeros(T_, N, device=t.device)
        o = step
        while o <= max_offset + 1e-9:
            ok = ok & (t.sample_edt(xy + nrm * o, tid) >= need)
            lim = torch.where(ok, torch.full_like(lim, o), lim)
            o += step
        return lim

    def _forward_min(self, lim, span_m: float):
        """(T, N) the smallest bound within `span_m` of arc ahead of each raceline point.

        Built once. The window is per track, because the same number of raceline points is a
        different distance on a 33 m lap and a 444 m one.
        """
        T_, N = lim.shape
        ds = (self.env.sim.track.length.to(lim.device) / N).clamp_min(1e-6)      # (T,)
        K = torch.ceil(torch.as_tensor(float(span_m), device=lim.device) / ds).long()
        out = lim.clone()
        for k in range(1, int(K.max().item()) + 1):
            out = torch.where((K >= k)[:, None], torch.minimum(out, torch.roll(lim, -k, dims=1)),
                              out)
        return out

    def _lateral(self, idx):
        """(B,) signed metres left of the raceline each car actually IS, at its own raceline point.

        The same signed cross product `RacelineTeacher` uses for its own lateral error, so "have I
        got across yet" is measured the way the line itself is defined rather than estimated.
        """
        e = self.env
        xy = e.sim.state[:, :2]
        p0 = self.teacher.xy[e.sim.tid, idx]
        t0 = self.teacher.tan[e.sim.tid, idx]
        return t0[:, 0] * (xy[:, 1] - p0[:, 1]) - t0[:, 1] * (xy[:, 0] - p0[:, 0])

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
        gap, v_other, near = self._ahead()

        # Commit, then hold the line out until GENUINELY through -- and "through" is a body length,
        # not a sign change.
        #
        # The obvious release is "nothing ahead of me any more", and it is wrong in the one place
        # that matters: the instant the ego's arc passes the opponent's, `gap` jumps from ~0 to
        # +1e9, the offset collapses, and the car cuts back across a nose it is still level with.
        # Traced on gen:control:9100 -- offset dropped at an arc gap of -0.29 m, contact on the same
        # step. So the release is keyed on the nearest car in EITHER direction, and the hold is a
        # car length plus a margin.
        #
        # `fits` gates committing but not continuing: a manoeuvre already begun is finished, and the
        # magnitude follows the lane through `room` anyway, so a narrowing shrinks the offset rather
        # than abandoning the car beside another one.
        slower = v_other < self.pass_speed_edge * e.sim.state[:, 3].clamp_min(0.5)
        engaging = ((gap < self.engage[e.sim.tid]) & fits & slower).float()
        self._w = torch.maximum(engaging, self._w * (near < self.release_hold).float())
        offset = self.side * room * self._w

        # `plan_action` clamps the offset against `teacher.offset_limit`, and that object is shared
        # with the opponents, whose scripted `shift` events need the SYMMETRIC bound in both
        # directions. This driver has measured the room on the side it is actually using, which is
        # larger, so the two clamps would fight and the symmetric one would win. Swap for the call
        # and restore: single-threaded, and the opponents' own actions are computed later, inside
        # `env.step`. try/finally so a raise cannot leave the opponents with a one-sided bound.
        was = self.teacher.offset_limit
        self.teacher.offset_limit = self.limit
        try:
            an = self.teacher.plan_action(e.sim.state, e.sim.P, e.sim.tid, e.ecfg.v_max_policy,
                                          e.tracker.spec, offset=offset)
        finally:
            self.teacher.offset_limit = was
        # Off the line, off the profile's speed -- ramped with the same weight the offset is, so
        # there is no step change when the manoeuvre starts.
        off_line = self.speed_scale * (1.0 - (1.0 - self.pass_speed_scale) * self._w)

        # Hold station behind a car this driver is not going round, AND behind one it has committed
        # to but is not yet beside. The trigger carries the room to brake at the current closing
        # speed, the shape the env's own `follow_cap` uses: a fixed distance is a rear-end waiting
        # for a fast approach. Zero inside the body gap, because a car a `stop` event has parked is
        # a wall that happens to be a car.
        mine = e.sim.state[:, 3]
        closing = (mine - v_other).clamp_min(0.0)
        trigger = self.follow_gap + closing * closing / (2.0 * self.follow_decel)
        abreast = self._lateral(idx) * self.side >= self.abreast_frac * offset.abs()
        holding = (gap < trigger) & ((self._w < 0.5) | ~abreast)
        # Gap keeping, not a fixed fraction of the car ahead.
        #
        # "0.9 x the other car's speed whenever it is close" is bang-bang, and it fails the way
        # bang-bang always does: an opponent on a raceline profile slows hard into every corner, and
        # a follower commanded 0.9 x the speed it had a moment ago is still closing when it gets
        # there. Measured on gen:control:9100 that was 13 s per trial spent inside 3 m and 3/4 car
        # contacts with no wall involved. This is the standard law instead -- match the car ahead at
        # the desired gap, slower when nearer, faster when further -- which has an equilibrium
        # rather than a limit cycle. Hard zero inside a metre: at that range the arc gap is about one
        # car length and the next thing that happens is contact.
        v_hold = (v_other + self.follow_gain * (gap - self.follow_gap)).clamp_min(0.0)
        v_hold = torch.where(gap < 1.0, torch.zeros_like(v_hold), v_hold)
        scale = torch.where(holding,
                            (v_hold / max(float(e.ecfg.v_max_policy), 1e-6)).clamp(0.0, 1.0),
                            torch.ones_like(v_other))
        # The plan's two speed entries are normalized to [-1, 1]; scaling them the way
        # `gym_env._opponent_actions` scales an event's speed keeps one convention for "drive this
        # plan slower" rather than inventing a second.
        an = an.clone()
        target = ((an[:, -2:] + 1) * 0.5 * float(e.ecfg.v_max_policy)) * off_line[:, None]
        target = torch.where(holding[:, None],
                             torch.minimum(target, (scale * float(e.ecfg.v_max_policy))[:, None]),
                             target)
        an[:, -2:] = (target / max(float(e.ecfg.v_max_policy), 1e-6) * 2 - 1).clamp(-1.0, 1.0)
        return an.clamp(-1.0, 1.0)
