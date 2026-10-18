"""The ForzaETH `spliner` overtaking planner, as a batched opponent for this simulator.

Why this exists: every opponent this project could put on the track was either the raceline
teacher -- which has never heard of the other cars, and whose only answer to one is the env's
follow cap -- or a checkpoint of our own. Neither is a *baseline*. "Our policy overtakes" only
means something against a driver somebody else designed, published and won with.

What is reproduced, and what is not
-----------------------------------
This is the **driving behaviour** of the ForzaETH race stack's `spliner` local planner
(`github.com/ForzaETH/race_stack`, arXiv:2403.11784), not its ROS graph. Localisation, opponent
detection and the EKF are the simulator's own job and it does them exactly; reproducing their
*error* would make the baseline worse than the one the authors measured, not more faithful. The
MAP controller underneath is likewise not reimplemented -- `RacelineTeacher` already tracks a line
with this simulator's own tyre model, which is a better tracker of our cars than a LUT fitted to
theirs. What is reproduced is the part that is actually the contribution: **where the line goes
when there is a car in the way**.

The constants below are the upstream defaults, named so a reader can check them against the
source rather than trust this docstring:

* seven spline control points at arc-length offsets ``-4.0, -3.0, -1.5, 0 (apex), +2.0, +3.0,
  +4.0`` m around the opponent, every one on the reference line except the apex;
* the apex is displaced sideways off the opponent by ``EVASION_DIST`` (0.65 m, centre to centre --
  comfortably more than the 0.31 m two half-widths need);
* a side is usable if the evasion point has ``SPLINE_BOUND_MINDIST`` (0.2 m) of room beyond the
  car's own half-width; the side with more room wins;
* the planner will not switch sides while the car is more than ``SIDE_SWITCH_D`` (0.25 m) off its
  own reference line, which is what stops it weaving across a car it is already beside;
* only opponents within ``LOOKAHEAD`` (10 m) and ``OBS_TRAJ_THRESH`` (0.3 m) of the reference line
  count -- a car off the line is not in the way;
* distances scale with speed by ``clip(1 + v / v_max, 1.0, 1.5)``;
* a pass toward the inside of a corner takes ``INSIDE_SPEED_SCALE`` (0.9) of the plan's speed.

Both of the stack's published local planners are here, because they differ in exactly one number:
`SplinerTeacher` draws its spline around where the other car *is*, and
`PredictiveSplinerTeacher` around where it will be in ``FIXED_PRED_TIME`` seconds. Two baselines,
one implementation, so that a fix to the spline cannot land in one of them and not the other.

The state machine is the stack's -- **RACING** on the global line, **TRAILING** behind a car that
cannot be passed, **OVERTAKING** on the evasion spline -- with one deliberate omission: TRAILING
here puts the line back on the raceline and does *not* command a speed. This env already holds a
gap behind the car ahead for every teacher-driven car (`F1VecEnv.follow_cap`, applied after the
teacher in `_opponent_actions`), and two controllers both deciding how hard to brake would fight.

Everything runs on the whole batch at once, because that is how the env asks: one `decide()` over
every row, and the plan itself comes from `RacelineTeacher.plan_action(offset=...)`, so the line
this returns is that teacher's line through a displaced reference and not a second implementation
of one.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .opponent_planner import FrenetOpponentPlanner
from .teacher import RacelineTeacher

#: Arc-length offsets of the spline's control points from the opponent's apex, in metres.
#: Upstream `pre_apex_0/1/2` and `post_apex_0/1/2`, with the apex between them.
CONTROL_S = (-4.0, -3.0, -1.5, 0.0, 2.0, 3.0, 4.0)
APEX_I = 3

#: Lateral clearance held from the opponent at the apex [m], centre to centre.
EVASION_DIST = 0.65

#: Room the evasion point needs beyond the car's own half-width before a side is usable, which is
#: also the margin the resulting line is held off the boundary by [m].
SPLINE_BOUND_MINDIST = 0.2

#: How far ahead an opponent is looked for [m], and how close to the reference line it has to be
#: before it counts as being in the way [m].
LOOKAHEAD = 10.0
OBS_TRAJ_THRESH = 0.3

#: The planner will not change which side it passes on while the car is further than this off the
#: centre of its own reference line [m].
SIDE_SWITCH_D = 0.25

#: Multiplier on the plan's speed when the pass goes toward the inside of a corner.
INSIDE_SPEED_SCALE = 0.9

#: Constant-time opponent prediction: the opponent is advanced `FIXED_PRED_TIME * v_opp` along its
#: own arc before the spline is drawn [s]. This one number is the whole difference between the
#: stack's two published local planners -- `spliner` evades where the other car *is*, `predictive
#: spliner` where it will be -- so it is a parameter with two named values rather than two
#: implementations that would drift apart (see `PredictiveSplinerTeacher`).
FIXED_PRED_TIME = 0.5

#: State machine codes. Exposed so a caller can log or plot which state each row was in.
RACING, TRAILING, OVERTAKING = 0, 1, 2

#: Upstream's extra stretch of the control points when the pass is on the outside of the corner
#: (`do_spline`: `si = s_apex + dst * 1.75`, a "TODO make parameter" there).
OUTSIDE_STRETCH = 1.75


def _natural_cubic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n-1, 4) coefficients of the natural cubic spline through (x, y), lowest power first.

    Segment i is `y = c[i,0] + c[i,1] t + c[i,2] t^2 + c[i,3] t^3` with `t = X - x[i]`. This is
    `scipy.interpolate.CubicSpline(..., bc_type="natural")`, which is what the upstream planner
    fits its control points with; written out because seven knots do not justify the dependency,
    and because it runs once at import.
    """
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n))
    r = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0                                   # natural: zero second derivative
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        r[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, r)
    out = np.zeros((n - 1, 4))
    out[:, 0] = y[:-1]
    out[:, 2] = c[:-1]
    out[:, 1] = (y[1:] - y[:-1]) / h - h * (2 * c[:-1] + c[1:]) / 3
    out[:, 3] = (c[1:] - c[:-1]) / (3 * h)
    return out


#: The spline's *shape*: its value along the control arc when the apex is displaced by one metre
#: and every other control point sits on the line. A cubic spline is linear in its control values,
#: so the curve for any apex displacement is this shape times that displacement -- which is why
#: the seven-point fit is done once at import rather than once per car per step.
#:
#: It peaks at 1.01 just past the apex and dips to -0.12 about 2 m before it: a cubic through six
#: zeros and a one leans slightly the *other* way on the approach. That is the curve those seven
#: control points define, not an artefact of computing it this way, and it is left alone -- at a
#: 0.65 m evasion it is an 8 cm set-up lean, which is what a driver does anyway. It does mean the
#: commanded offset and the side the planner committed to can disagree in sign 2 m out, so which
#: side a pass is on is read from `last_state`/`_side` and never from the sign of one offset.
_SHAPE = _natural_cubic(np.asarray(CONTROL_S, dtype=np.float64),
                        np.eye(len(CONTROL_S))[APEX_I])


class SplinerTeacher(FrenetOpponentPlanner):
    """ForzaETH `spliner`, batched. Drop-in wherever this env drives a car with a teacher.

    `base` supplies the reference line, the grip-aware speed profile, the latency compensation and
    the plan fit; `FrenetOpponentPlanner` supplies the Frenet view of the race and the wiring.
    This class decides one thing: the lateral offset that line is followed at.
    """

    #: Tensors carried from one call to the next, updated in place. A captured teacher reads them as
    #: inputs and writes them back after every replay (`viewer.graph_fastpath._TeacherCapture`).
    GRAPH_STATE = ("_side",)

    STATE_NAMES = ("racing", "trailing", "overtaking")

    def __init__(self, base: RacelineTeacher, env=None, *,
                 evasion_dist: float = EVASION_DIST,
                 bound_mindist: float = SPLINE_BOUND_MINDIST,
                 lookahead: float = LOOKAHEAD,
                 obs_traj_thresh: float = OBS_TRAJ_THRESH,
                 side_switch_d: float = SIDE_SWITCH_D,
                 inside_speed_scale: float = INSIDE_SPEED_SCALE,
                 fixed_pred_time: float = 0.0,
                 upstream: bool = False,
                 outside_stretch: float = OUTSIDE_STRETCH):
        super().__init__(base, env=None)          # attach last: the fields below are its inputs
        #: True: `decide` is upstream's `do_spline` (race_stack @ 6a2900a, see `_decide_upstream`) --
        #: what an ego baseline runs. False: the simplified spliner the opponents in our races have
        #: always run, kept so that the traffic our policies were trained and scored against does
        #: not change under them.
        self.upstream = bool(upstream)
        self.outside_stretch = float(outside_stretch)
        self.evasion_dist = float(evasion_dist)
        self.bound_mindist = float(bound_mindist)
        self.lookahead = float(lookahead)
        self.obs_traj_thresh = float(obs_traj_thresh)
        self.side_switch_d = float(side_switch_d)
        self.inside_speed_scale = float(inside_speed_scale)
        self.fixed_pred_time = float(fixed_pred_time)
        self.shape = torch.tensor(_SHAPE, dtype=torch.float32, device=self.device)   # (6, 4)
        self.knots = torch.tensor(CONTROL_S, dtype=torch.float32, device=self.device)
        #: Which side the pass in progress is committed to, per row: +1 left, -1 right, 0 none.
        #: Held across steps because the side-switch rule is about *changing* it, not about
        #: picking it in the first place.
        self._side: Optional[torch.Tensor] = None
        #: The last `decide`'s spline per row, (apex offset, apex station ahead of the car, stretch),
        #: or None when there was no race to plan against -- what `path_offsets` draws.
        self.last_spline: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        if env is not None:
            self.attach(env)

    # ------------------------------------------------------------------ the planner
    @torch.no_grad()
    def decide(self, state: torch.Tensor, tid: Optional[torch.Tensor] = None
               ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.upstream:
            return self._decide_upstream(state, tid)
        return self._decide_simplified(state, tid)

    @torch.no_grad()
    def _decide_upstream(self, state: torch.Tensor, tid: Optional[torch.Tensor] = None
                         ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Upstream's `SplinePlanner.do_spline` (ForzaETH race_stack @ 6a2900a, `planner/local_planners/
        spline_planner/spline_planner.py`), batched -- read from the source, line for line where it
        decides anything:

        * obstacles within 10 m are moved `fixed_pred_time` ahead at their tracked (vs, vd)
          (`_predict_obs_movement`, mode "constant": upstream's default 0.15 s);
        * only obstacles with |d_center| < `obs_traj_thresh` and within `lookahead` ahead count
          (`_obs_filtering`), the closest of them is the one splined around;
        * the apex sits at the obstacle's longitudinal centre (s_start + s_end) / 2, displaced
          `evasion_dist` beyond its LEFT or RIGHT EDGE -- not its centre (`_more_space`): a side is
          usable when the gap between that edge and the track boundary exceeds evasion_dist +
          spline_bound_mindist; with both or neither usable, the apex nearer the raceline wins; an
          apex on the far side of the raceline is pulled back to it;
        * the control points stretch by clip(1 + v / v_max, 1, 1.5), and by 1.75 more when the pass is
          on the outside of the corner; the offset is clipped to [0, d_apex];
        * no side switch while more than 0.25 m off the raceline (`_check_ot_side_possible`), and no
          evasion at all where the spline comes within `spline_bound_mindist` of the boundary -- both
          leave the car on the raceline behind the obstacle (TRAILING; the env's or the stack's
          follow cap holds the gap, upstream's state machine does it there);
        * an inside pass drives the plan's speed x 0.9.

        One addition, ours: an obstacle stays the one splined around until the car is `4 m x stretch`
        past it (`CONTROL_S[0]`), because this runs stateless per step and upstream's state machine is
        what finishes a pass there.
        """
        B = state.shape[0]
        dev, dt = state.device, state.dtype
        zero = torch.zeros(B, device=dev, dtype=dt)
        one = torch.ones(B, device=dev, dtype=dt)
        mode = torch.full((B,), RACING, device=dev, dtype=torch.long)
        Bf = self._opp_state(state).shape[0]
        if self._side is None or self._side.shape != (Bf,) or self._side.device != dev:
            self._side = torch.zeros(Bf, device=dev, dtype=dt)
        tid_b = torch.zeros(B, dtype=torch.long, device=dev) if tid is None else tid
        obs = self._obstacles(state, tid_b)
        if obs is None:
            self.last_state, self.last_offset, self.last_spline = mode, zero, None
            return None, mode, one
        stretch = self._stretch(state)
        base = self.base
        # prediction (`_predict_obs_movement`, constant): only obstacles within 10 m are moved
        near = obs["ok"] & (obs["gap"] < 10.0)
        gap = torch.where(near, obs["gap"] + self.fixed_pred_time * obs["vs"], obs["gap"])
        dc = torch.where(near, obs["d"] + self.fixed_pred_time * obs["vd"], obs["d"])
        rear = CONTROL_S[0] * stretch
        cand = obs["ok"] & (dc.abs() < self.obs_traj_thresh) & (gap < self.lookahead) & (gap > rear[:, None])
        rank = torch.where(cand, gap, torch.full_like(gap, float("inf")))
        j = rank.argmin(1)
        ar = torch.arange(B, device=dev)
        engaged = torch.isfinite(rank[ar, j])
        g, d0 = gap[ar, j], dc[ar, j]
        back, front = obs["back"][ar, j], obs["front"][ar, j]
        gap_apex = g + 0.5 * (front - back)                       # (s_start + s_end) / 2
        d_left = d0 + obs["left"][ar, j]
        d_right = d0 - obs["right"][ar, j]
        ds = base.ds[tid_b]
        idx_e = base.project(state[:, :2], tid_b)[0]
        idx_apex = torch.remainder(idx_e + (gap_apex / ds).round().long(), base.N)
        lane_l, lane_r = self._lane_bounds()
        lb, rb = lane_l[tid_b, idx_apex], lane_r[tid_b, idx_apex]
        left_gap, right_gap = (lb - d_left).abs(), (rb + d_right).abs()
        min_space = self.evasion_dist + self.bound_mindist
        cand_l, cand_r = d_left + self.evasion_dist, d_right - self.evasion_dist
        only_r = (right_gap > min_space) & (left_gap < min_space)
        only_l = (left_gap > min_space) & (right_gap < min_space)
        go_left = torch.where(only_r, torch.zeros_like(only_r),
                              torch.where(only_l, torch.ones_like(only_l), cand_l.abs() <= cand_r.abs()))
        side = torch.where(go_left, one, -one)
        d_apex = torch.where(go_left, cand_l.clamp_min(0.0), cand_r.clamp_max(0.0))
        # the outside of the corner at the apex: the sign of the summed curvature over the next 2 m
        n_k = max(2, int(round(2.0 / float(base.ds.min()))))
        kk = torch.remainder(idx_apex[:, None] + torch.arange(n_k, device=dev)[None], base.N)
        outside = torch.where(base.kappa[tid_b[:, None], kk].sum(1) < 0, one, -one)   # + left
        span = stretch * torch.where(outside == side, torch.full_like(one, self.outside_stretch), one)
        # side lock and the track-bound check -> no evasion (TRAILING)
        held = self._r(self._side)
        ego_d = self._lateral(state[:, :2], tid_b)[0]
        blocked_switch = (ego_d.abs() > self.side_switch_d) & (held != 0) & (side != held)
        rs = torch.linspace(CONTROL_S[0], CONTROL_S[-1], 33, device=dev, dtype=dt)
        lo, hi = torch.minimum(d_apex, zero), torch.maximum(d_apex, zero)
        d_s = (d_apex[:, None] * self._shape_at(rs).to(dt)[None]).clamp(min=lo[:, None]).clamp(max=hi[:, None])
        idx_s = torch.remainder(idx_e[:, None] + ((gap_apex[:, None] + rs[None] * span[:, None]) / ds[:, None]).round().long(),
                                base.N)
        bound = torch.where(side[:, None] > 0, lane_l[tid_b[:, None], idx_s], lane_r[tid_b[:, None], idx_s])
        too_close = ((d_s.abs() > 0.1) & (d_s.abs() > bound - self.bound_mindist)).any(1)
        danger = blocked_switch | too_close
        passing = engaged & ~danger
        trailing = engaged & danger
        r0 = ((0.0 - gap_apex) / span).to(dt)
        d_cmd = (d_apex * self._shape_at(r0)).clamp(min=lo).clamp(max=hi)
        d_out = torch.where(passing, d_cmd, zero)
        mode = torch.where(passing, torch.full_like(mode, OVERTAKING),
                           torch.where(trailing, torch.full_like(mode, TRAILING), mode))
        self._put(self._side, torch.where(passing, side, held))
        scale = torch.where(passing & (outside != side), one * self.inside_speed_scale, one)
        self.last_state, self.last_offset = mode, d_out
        self.last_spline = (d_apex, gap_apex, span)
        return d_out, mode, scale

    def _obstacles(self, state: torch.Tensor, tid: torch.Tensor):
        """Upstream's `ObstacleArray` relative to each car: the observer's tracks (the modular
        stack), or -- the oracle -- the other cars and the procedural props from the simulator, with
        their true footprints. (B, K) tensors: gap to the centre (+ ahead, on the raceline's own
        arc), d centre, vs, vd, half extents back / front / left / right, ok. None: nothing to see."""
        if self.observer is not None:
            return self.observer.obstacle_list(self, state, tid)
        env, base = self.env, self.base
        if env is None:
            return None
        B = state.shape[0]
        dev = state.device
        idx_e = base.project(state[:, :2], tid)[0]
        L = base.length[tid]
        parts = []

        def frenet(xy, tid_k):
            idx = base.project(xy.reshape(-1, 2), tid_k.reshape(-1))[0].view(xy.shape[:-1])
            nrm = self._normal(tid_k, idx)
            d = ((xy - base.xy[tid_k, idx]) * nrm).sum(-1)
            g = torch.remainder((idx - idx_e[:, None]).to(state.dtype) * base.ds[tid][:, None] + 0.5 * L[:, None],
                                L[:, None]) - 0.5 * L[:, None]
            return g, d

        if int(getattr(env, "M", 1)) >= 2 and env.sim.other_idx is not None \
                and self._opp_state(state).shape[0] == env.sim.state.shape[0]:
            other = self._r(env.sim.other_idx)
            whole = self._opp_state(state)
            C = other.shape[1]
            tid_c = tid[:, None].expand(B, C)
            g, d = frenet(whole[other, :2], tid_c)
            half_l = 0.5 * float(env.cfg.vehicle.length)
            half_w = 0.5 * float(env.cfg.vehicle.width)
            full = torch.ones_like(g)
            parts.append({"gap": g, "d": d, "vs": whole[other, 3], "vd": torch.zeros_like(g), "back": full * half_l,
                          "front": full * half_l, "left": full * half_w, "right": full * half_w,
                          "ok": torch.ones_like(g, dtype=torch.bool)})
        if self.props is not None:
            pr = self.props
            rows = self._env_rows(B, dev)
            sid = pr.p_sid[rows]
            alive = sid >= 0
            P = sid.shape[1]
            tid_p = tid[:, None].expand(B, P)
            g, d = frenet(pr.p_poses[rows, :, :2], tid_p)
            if not hasattr(self, "_sh_along"):
                self._sh_along = torch.tensor([s_.along for s_ in pr.shapes], dtype=torch.float32, device=dev)
            sc = sid.clamp_min(0)
            half_a = 0.5 * pr.sh_across[sc].to(g.dtype)
            half_l = 0.5 * self._sh_along[sc].to(g.dtype)
            parts.append({"gap": g, "d": d, "vs": torch.zeros_like(g), "vd": torch.zeros_like(g), "back": half_l,
                          "front": half_l, "left": half_a, "right": half_a, "ok": alive})
        if not parts:
            return None
        return {k: torch.cat([p_[k] for p_ in parts], 1) for k in parts[0]}

    def _lane_bounds(self):
        """(T, N) metres from each reference point to the static map, left and right: upstream's
        global waypoints' d_left / d_right (the track boundaries the spliner checks against)."""
        held = getattr(self, "_lanes", None)
        if held is not None:
            return held
        base, tr = self.base, self.track
        xy, tan = base.xy, base.tan
        Tn, N, _ = xy.shape
        tid = torch.arange(Tn, device=xy.device)[:, None].expand(Tn, N)
        out = []
        for sign in (1.0, -1.0):
            nrm = torch.stack([-tan[..., 1], tan[..., 0]], -1) * sign
            ok = torch.ones(Tn, N, dtype=torch.bool, device=xy.device)
            w = torch.zeros(Tn, N, device=xy.device)
            for k in range(1, 61):
                o = 0.05 * k
                ok = ok & (tr.sample_edt(xy + nrm * o, tid) > 0.03)
                w = torch.where(ok, torch.full_like(w, o), w)
            out.append(w)
        self._lanes = (out[0], out[1])
        return self._lanes

    @torch.no_grad()
    def _decide_simplified(self, state: torch.Tensor, tid: Optional[torch.Tensor] = None
                           ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """(offset [m left of the reference line], state code, speed scale) per row.

        The offset is the spline evaluated at the car's own station: the control points are what
        upstream fits, and what the car needs each step is that fit's value here. It is `None`,
        not zeros, when there is no race to plan against -- see `_merge_offset`.
        """
        B = state.shape[0]
        dev, dt = state.device, state.dtype
        zero = torch.zeros(B, device=dev, dtype=dt)
        one = torch.ones(B, device=dev, dtype=dt)
        mode = torch.full((B,), RACING, device=dev, dtype=torch.long)
        Bf = self._opp_state(state).shape[0]          # memory is whole-batch sized (`row_planning`)
        if self._side is None or self._side.shape != (Bf,) or self._side.device != dev:
            self._side = torch.zeros(Bf, device=dev, dtype=dt)

        # Distances stretch with speed, and the rear bound is the spline's own arc: a manoeuvre is
        # not over when the other car's gap goes negative, it is over when the line has rejoined.
        stretch = self._stretch(state)
        seen = self._opponents(state, tid, CONTROL_S[0] * stretch)
        if seen is None:
            # No opponent structure at all -- a solo env, or a caller this planner cannot match to
            # the simulator's rows. `None` rather than zeros, so the reference teacher takes the
            # path it takes when nobody asked it for an offset (see `_merge_offset`).
            self._put(self._side, torch.zeros_like(self._r(self._side)))
            self.last_state, self.last_offset, self.last_spline = mode, zero, None
            return None, mode, one
        gap, d_opp, idx_opp, tid_b, ego_d = seen

        # A crate on the line is an obstacle too, and upstream's detector says so: it reports
        # obstacles, not cars. Whichever is nearer ahead is the thing to plan around, and a prop is
        # a car that will not move -- so it enters with zero speed and the prediction below leaves
        # it where it is.
        b_gap, b_d = self._blockage_ahead(state, tid_b, CONTROL_S[-1] * stretch,
                                          CONTROL_S[-1] * self.STRETCH_MAX)
        take_prop = b_gap < gap.clamp_min(0.0)
        gap = torch.where(take_prop, b_gap, gap)
        d_opp = torch.where(take_prop, b_d, d_opp)
        idx_opp = torch.where(take_prop,
                              (self.base.project(state[:, :2], tid_b)[0]
                               + (b_gap.nan_to_num(posinf=0.0) / self.base.ds[tid_b]).round().long()
                               ) % self.base.N, idx_opp)

        # Constant-time prediction: where the opponent will be by the time we are there. Advancing
        # the apex along the arc is upstream's whole prediction, and it is what aims the spline at
        # the gap the car is moving into rather than at the one it is leaving.
        v_opp = torch.where(take_prop, torch.zeros_like(gap), self._opponent_speed(state))
        s_apex = gap + self.fixed_pred_time * v_opp

        # In the way at all: near the line we are driving, and inside the spline's own arc. The
        # rear bound is what keeps the post-apex control points -- the return to the line -- being
        # driven, instead of the offset snapping to zero the instant the pass completes.
        span = CONTROL_S[-1] * stretch
        engaged = ((d_opp.abs() < self.obs_traj_thresh)
                   & (s_apex < self.lookahead) & (s_apex > -span))

        # Which side has room, asked of the simulator's own distance field at the point the
        # evasion line would actually pass through. Better information than a lane half-width: it
        # is the clearance on the side the car really goes.
        clear_l, clear_r = self._side_clearance(idx_opp, tid_b, d_opp, stretch)
        need = self.half_width + self.bound_mindist
        side = torch.where(clear_l >= clear_r, one, -one)

        # Do not change sides while committed: off the centre of its own line by more than
        # `side_switch_d` the car is already beside something, and weaving is how contact happens.
        held = self._r(self._side)
        committed = (held != 0) & (ego_d.abs() > self.side_switch_d)
        side = torch.where(committed, held, side)
        room = torch.where(side > 0, clear_l, clear_r)

        # The apex, and the spline through it evaluated where we are. A cubic spline is linear in
        # its control values, so the curve is the fixed seven-knot shape times this displacement.
        d_apex = d_opp + side * self.evasion_dist * stretch
        d_cmd = d_apex * self._shape_at((-s_apex / stretch).to(dt))

        # State machine. A pass there is no room for is trailing, not a pass attempted anyway --
        # and trailing here means "back on the line"; the env's follow cap holds the gap.
        passing = engaged & (room >= need)
        trailing = engaged & ~passing
        mode = torch.where(passing, torch.full_like(mode, OVERTAKING),
                           torch.where(trailing, torch.full_like(mode, TRAILING), mode))
        d_out = torch.where(passing, d_cmd, zero)
        # In place, not rebound: the side lock is this planner's memory between steps, and a CUDA
        # graph can only carry state it is allowed to write back (`GRAPH_STATE`).
        self._put(self._side, torch.where(passing, side, torch.zeros_like(side)))

        # An inside pass gives a little speed back: the inside of a corner is the side the
        # reference line is already turning toward, and it is the shorter, tighter way past.
        inside = passing & (side * self._corner_sign(idx_opp, tid_b) > 0)
        scale = torch.where(inside, one * self.inside_speed_scale, one)
        self.last_state, self.last_offset = mode, d_out
        self.last_spline = (d_apex, s_apex, stretch)
        return d_out, mode, scale

    def path_offsets(self, rows: torch.Tensor, s: torch.Tensor, speed: torch.Tensor) -> Optional[torch.Tensor]:
        """The evasion spline itself along `s` for the rows that are passing, the line elsewhere.
        At `s = 0` this is the offset `decide` commanded: the curve the car is driving through."""
        if self.last_spline is None or self.last_state is None:
            return super().path_offsets(rows, s, speed)
        d_apex, s_apex, stretch = (t[rows].to(s.dtype) for t in self.last_spline)
        r = (s[None, :] - s_apex[:, None]) / stretch[:, None]
        d = d_apex[:, None] * self._shape_at(r.reshape(-1)).reshape(r.shape)
        if self.upstream:                                    # upstream clips the spline to [0, d_apex]
            z = torch.zeros_like(d_apex)
            d = d.clamp(min=torch.minimum(d_apex, z)[:, None]).clamp(max=torch.maximum(d_apex, z)[:, None])
        passing = (self.last_state[rows] == OVERTAKING)[:, None]
        return torch.where(passing, d, torch.zeros_like(d))

    # ------------------------------------------------------------------ pieces
    def reset_rows(self, rows: torch.Tensor) -> None:
        """A new race on this row is not the pass this one was committed to."""
        if self._side is not None:
            self._side[rows] = 0.0

    def _shape_at(self, r: torch.Tensor) -> torch.Tensor:
        """The spline's shape at arc position `r` relative to the apex, zero outside its span."""
        i = torch.bucketize(r.to(self.knots.dtype), self.knots[1:-1]).clamp(0, self.shape.shape[0] - 1)
        c = self.shape[i]                                              # (B, 4)
        t = (r.to(c.dtype) - self.knots[i])
        val = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))
        inside = (r >= float(CONTROL_S[0])) & (r <= float(CONTROL_S[-1]))
        return torch.where(inside, val.to(r.dtype), torch.zeros_like(r))

    def _side_clearance(self, idx: torch.Tensor, tid: torch.Tensor, d_opp: torch.Tensor,
                        stretch: torch.Tensor):
        """Free space at the left and right evasion points of the opponent's station [m each]."""
        step = self.evasion_dist * stretch
        return (self._clearance_at(idx, tid, d_opp + step).to(d_opp.dtype),
                self._clearance_at(idx, tid, d_opp - step).to(d_opp.dtype))

    def _corner_sign(self, idx: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """+1 where the reference line curves left, -1 right, 0 straight. `RacelineTeacher.kappa`
        is signed left-positive, the same sign convention as the lateral offset."""
        return torch.sign(self.base.kappa[tid, idx])

class PredictiveSplinerTeacher(SplinerTeacher):
    """ForzaETH `predictive spliner`: the same planner, aimed where the other car will be.

    The published pair differ in one thing, and this class is that thing: the apex is placed at
    the opponent's position `FIXED_PRED_TIME` seconds from now rather than at the one it occupies.
    Upstream reaches that position through a learned Gaussian-process model of the opponent's
    speed around the lap; here the opponent's own speed is known exactly, so the constant-time
    step `ds = fixed_pred_time * v_opp` is that model's output without its estimation error --
    which makes this the *optimistic* end of what the published planner does, and it is reported
    as such rather than as a like-for-like port.

    Kept as a subclass with one default changed, rather than a second module, so that a fix to the
    spline, the side choice or the state machine cannot land in one baseline and not the other.
    """

    def __init__(self, base: RacelineTeacher, env=None, *,
                 fixed_pred_time: float = FIXED_PRED_TIME, **kw):
        super().__init__(base, env=env, fixed_pred_time=fixed_pred_time, **kw)

#: Upstream's spline_planner predicts every obstacle within 10 m by this much at its tracked speed
#: (`fixed_pred_time`, `_predict_obs_movement` mode "constant"): the ego spliner's default.
UPSTREAM_PRED_TIME = 0.15
