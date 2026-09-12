"""Per-cell measurement loop.

Split deliberately in two. `CellRecorder` holds all the bookkeeping -- first-attempt masking, the
static-mu guard, pass detection, avoidance windows -- and is driven by plain arrays, so every rule it
enforces is testable on CPU without a simulator. `run_cell` is the thin part that owns a real env and
feeds it.

Two invariants the loop exists to protect:

  * measurements are read **before** the auto-reset, because the simulator rewrites its buffers a few
    lines later and a post-reset read silently measures the next episode;
  * a trial contributes **once**. After its first terminal outcome it is inactive, and nothing it
    does afterwards reaches any counter.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .geom import assert_rear_box_within_bound, thresholds
from .overtake import PassDetector, TrafficTrace, outcome as pass_outcome
from .tally import Tally


class MuChangedError(RuntimeError):
    """Friction moved inside an episode. The run is void, not merely worse."""


@dataclass
class CellRecorder:
    """One cell: `n` trials measured to their first terminal outcome.

    `expected_n` is carried into the Tally so a short cell reports N/A rather than a rank.
    """
    n: int
    suite: str
    track_length_m: float
    vehicle_length: float
    vehicle_width: float
    hold_steps: int
    s_obs_m: float | None = None
    obstacle_window_m: float = 4.0
    frozen: bool = True
    #: T only: how many opponents each learner is racing, and the two arc windows the traffic trace
    #: uses. `n_opponents` sizes the per-pair detectors; 0 leaves the traffic path inert.
    n_opponents: int = 0
    contention_range_m: float = 12.0
    attack_range_m: float = 3.0

    active: list = field(default_factory=list)
    outcome: list = field(default_factory=list)
    detectors: list = field(default_factory=list)
    encountered: list = field(default_factory=list)
    cleared: list = field(default_factory=list)
    completed: list = field(default_factory=list)
    #: |lateral offset| accumulated per active step, and the step count it was accumulated over.
    #: Emitted as a mean; a trial with no samples reports N/A, never a measured 0.
    cross_track_sum: list = field(default_factory=list)
    cross_track_sq: list = field(default_factory=list)
    cross_track_n: list = field(default_factory=list)
    progress_m: list = field(default_factory=list)
    distance_m: list = field(default_factory=list)
    elapsed_s: list = field(default_factory=list)
    #: T only, per trial: one `PassDetector(repeat=True)` per opponent, one `TrafficTrace`, and
    #: whether an opponent event was running while this learner was inside the contention window.
    traffic: list = field(default_factory=list)
    pair_detectors: list = field(default_factory=list)
    event_in_window: list = field(default_factory=list)
    event_in_window_s: list = field(default_factory=list)
    event_seen_s: list = field(default_factory=list)
    contended: list = field(default_factory=list)
    _mu0: list = field(default_factory=list)

    def __post_init__(self):
        overlap, clear = thresholds(self.vehicle_length)
        self.active = [True] * self.n
        self.outcome = [None] * self.n
        self.encountered = [False] * self.n
        self.cleared = [False] * self.n
        self.completed = [False] * self.n
        self.progress_m = [0.0] * self.n
        self.distance_m = [0.0] * self.n
        self.cross_track_sum = [0.0] * self.n
        self.cross_track_sq = [0.0] * self.n
        self.cross_track_n = [0] * self.n
        self.elapsed_s = [0.0] * self.n
        self.detectors = [PassDetector(length=self.track_length_m, overlap=overlap, clear=clear,
                                       hold_steps=self.hold_steps) for _ in range(self.n)]
        if self.suite == "T":
            self.traffic = [TrafficTrace(contention_range_m=self.contention_range_m,
                                         attack_range_m=self.attack_range_m)
                            for _ in range(self.n)]
            # One detector per (learner, opponent) pair. `keep_history=False`: a traffic stint runs
            # to its full budget -- thousands of steps -- against up to two opponents, and the trace
            # was only ever read for a flag the detector now carries itself.
            self.pair_detectors = [
                [PassDetector(length=self.track_length_m, overlap=overlap, clear=clear,
                              hold_steps=self.hold_steps, repeat=True, keep_history=False)
                 for _ in range(self.n_opponents)] for _ in range(self.n)]
            self.event_in_window = [False] * self.n
            self.event_in_window_s = [0.0] * self.n
            self.event_seen_s = [0.0] * self.n
            self.contended = [False] * self.n

    # -- friction ------------------------------------------------------------------------------
    def bind_mu(self, mu_all) -> None:
        """Bind EVERY car of the cell, opponents included.

        Binding only the measured rows leaves an opponent free to change friction mid-episode
        unnoticed, which is the same violation wearing a different slot number.
        """
        self._mu0 = [float(x) for x in mu_all]

    def check_mu(self, mu_all, fresh) -> None:
        """Static per episode, across every car. Only a car that just reset may show a new value.

        `fresh[i]` means env i entered this step with `sim.steps == 0`, i.e. the gym layer reset it
        after the previous step. That is the only correct discriminator from inside the call: the
        trace reads before the auto-reset, so a car reset a moment ago still shows a clean +1 step
        advance while its friction was legitimately redrawn in between.
        """
        for i, (m, is_fresh) in enumerate(zip(mu_all, fresh)):
            if is_fresh:
                self._mu0[i] = float(m)
            elif abs(float(m) - self._mu0[i]) > 1e-9:
                raise MuChangedError(
                    f"car {i}: mu moved {self._mu0[i]} -> {float(m)} inside an episode; friction "
                    f"is fixed for the whole episode")

    # -- overtaking seeding ----------------------------------------------------------------------
    def seed_gaps(self, gap) -> None:
        """Seed each detector at the real initial gap, BEFORE the first transition.

        Seeding inside the first `update` would consume that transition's terminal flags, so a car
        that crashed on step one would be recorded as merely never having armed.
        """
        for i, g in enumerate(gap):
            self.detectors[i].start(float(g))

    def seed_pair_gaps(self, gaps) -> None:
        """T: seed every (learner, opponent) pair. `gaps[i][j]` is learner i's arc to opponent j."""
        for i, row in enumerate(gaps):
            for j, g in enumerate(row):
                self.pair_detectors[i][j].start(float(g))

    # -- one transition, read pre-reset ----------------------------------------------------------
    def update(self, *, progress, speed, dt, collision, truncated, s=None, gap=None,
               car_contact=None, opponent_reset=None, opponent_incident=None,
               budget_exhausted=False, lateral=None, pair_gaps=None, pair_progress=None,
               pair_reset=None, opp_event_active=None):
        """Precedence is fixed and deliberate: a collision on the same step as a success is a
        failure. A car that touches something on the exit stride did not clear the obstacle, and a
        car that crashes as the hold completes did not complete a clean pass.

        The T arguments are per (learner, opponent): `pair_gaps[i][j]` the signed arc, positive when
        opponent j is ahead of learner i; `pair_progress[i][j]` that opponent's wrapped arc advance
        this step; `pair_reset[i][j]` whether it entered this step just respawned.
        `opp_event_active[i]` is whether any of learner i's opponents is running a scripted event.
        """
        for i in range(self.n):
            if not self.active[i]:
                continue
            self.progress_m[i] += float(progress[i])
            self.distance_m[i] += abs(float(speed[i])) * dt
            self.elapsed_s[i] += dt
            if lateral is not None:
                lat = float(lateral[i])
                self.cross_track_sum[i] += abs(lat)
                self.cross_track_sq[i] += lat * lat
                self.cross_track_n[i] += 1

            hit = bool(collision[i])
            contact = bool(car_contact[i]) if car_contact is not None else False

            if self.suite == "T":
                self._update_traffic(i, dt=dt, progress=float(progress[i]),
                                     pair_gaps=pair_gaps, pair_progress=pair_progress,
                                     pair_reset=pair_reset, contact=contact, hit=hit,
                                     opp_event_active=opp_event_active)

            # Detectors advance every step, with terminal flags applied, even the first.
            if self.suite == "O" and gap is not None:
                # Same-step precedence. An opponent that crashed on THIS transition must invalidate
                # before the hold can complete on it; keying only on the next step's entry freshness
                # is one frame late and lets a pass succeed on the tick the opponent died.
                opp_out = bool(opponent_reset[i]) if opponent_reset is not None else False
                if opponent_incident is not None:
                    opp_out = opp_out or bool(opponent_incident[i])
                self.detectors[i].update(
                    float(gap[i]), contact=contact, opponent_reset=opp_out, terminated=hit)

            if hit:
                self._finish(i, False, self._collision_reason(i, contact))
                continue

            if self.suite == "T":
                # No task event ends a traffic trial early. A pass is a counter, not a finish line,
                # and there is nothing to "complete": the stint runs to the budget, and the only
                # ways out are the wall and the other car, both handled above. Reaching the end is
                # the success, and it is the one an unpassable floor still allows.
                if bool(truncated[i]) or budget_exhausted:
                    self._finish(i, True, None)
                continue

            if self.suite == "O":
                d = self.detectors[i]
                if d.succeeded:
                    self._finish(i, True, None)
                    continue
                if d.done:                      # INVALID: respawn, contact, termination
                    self._finish(i, False, d.reason or "no_pass")
                    continue

            if self.suite == "A" and s is not None and self.s_obs_m is not None:
                ds = _signed_to(float(s[i]), self.s_obs_m, self.track_length_m)
                if abs(ds) <= self.obstacle_window_m:
                    self.encountered[i] = True
                elif self.encountered[i] and ds > self.obstacle_window_m:
                    self._finish(i, True, None)     # past the window, having touched nothing
                    continue

            if self.suite == "S" and self.progress_m[i] >= self.track_length_m:
                # Inclusive one-lap rule, matching TrialAccumulator: reaching the lap distance
                # within the budget is a completion, not a timeout. 51 m of a 50 m lap completes.
                self._finish(i, True, None)
                continue

            if bool(truncated[i]) or budget_exhausted:
                self._finish(i, False, self._timeout_reason(i))

    def _update_traffic(self, i: int, *, dt, progress, pair_gaps, pair_progress, pair_reset,
                        contact, hit, opp_event_active) -> None:
        """One traffic transition for learner i: the pair detectors, then the continuous trace."""
        gaps = list(pair_gaps[i]) if pair_gaps is not None else []
        opp_prog = list(pair_progress[i]) if pair_progress is not None else []
        resets = list(pair_reset[i]) if pair_reset is not None else [False] * len(gaps)
        for j, det in enumerate(self.pair_detectors[i]):
            if j >= len(gaps):
                break
            det.update(float(gaps[j]), contact=contact,
                       opponent_reset=bool(resets[j]), terminated=hit)
        tr = self.traffic[i]
        tr.update(dt=dt, ego_progress=progress, opponent_progress=opp_prog, gaps=gaps)
        in_window = any(abs(float(g)) <= self.contention_range_m for g in gaps)
        if in_window:
            self.contended[i] = True
        if opp_event_active is not None and bool(opp_event_active[i]):
            self.event_seen_s[i] += dt
            if in_window:
                # The measurement the scenario stands on: an event that fires while the learner is
                # half a lap away is a schedule entry, not a thing the learner had to react to.
                self.event_in_window[i] = True
                self.event_in_window_s[i] += dt

    def _collision_reason(self, i: int, car_contact: bool = False) -> str:
        if car_contact:
            return "contact"                     # another car, from the actual flag
        if self.suite == "A":
            return "hit" if self.encountered[i] else "approach_collision"
        return "collision"

    def _timeout_reason(self, i: int) -> str:
        if self.suite == "A":
            return "approach_timeout"
        if self.suite == "O":
            return pass_outcome(self.detectors[i])["reason"] or "no_pass"
        return "timeout"

    #: Set by `run_cell`. A/O finish on their own task events, which the accumulator knows nothing
    #: about -- without this it keeps integrating exposure after a trial has already been scored.
    on_finish = None

    def _finish(self, i: int, success: bool, reason: str | None) -> None:
        self.active[i] = False
        if self.on_finish is not None:
            self.on_finish(i)
        self.cleared[i] = success and self.suite == "A"
        self.completed[i] = success and self.suite == "S"
        self.outcome[i] = {"success": success, "reason": reason}

    def finalize(self) -> dict:
        """Trials still running at the budget's end are timeouts, not silent drops."""
        for i in range(self.n):
            if self.active[i]:
                self._finish(i, False, self._timeout_reason(i))
        t = Tally(expected_n=self.n, frozen=self.frozen)
        for o in self.outcome:
            t.record(o["success"], o["reason"])
        out = {"suite": self.suite, "n": self.n, "tally": t.as_dict(),
               "progress_m": list(self.progress_m),
               # travelled distance, not |net progress|: a car that oscillates covers ground, and
               # collisions/km divided by net progress overstates the rate badly.
               "distance_m": list(self.distance_m),
               "elapsed_s": list(self.elapsed_s), "completed": list(self.completed),
               "lap_time_s": [e if c else None for e, c in zip(self.elapsed_s, self.completed)],
               "outcomes": list(self.outcome)}
        # Both statistics, each computed as its name says. The previous field was a mean of
        # |offset| rendered as an RMS; relabelling one as the other is how a number stops meaning
        # what it claims.
        # Emitted here, not bolted on by `run_cell` afterwards: a row that `finalize` produced
        # should be a complete row. Adding it downstream meant a producer-side test built a dict the
        # validator then refused for a missing array that the real pipeline happens to fill in.
        out["route_progress_fraction"] = route_progress_fraction(
            self.progress_m, [self.track_length_m] * self.n)
        out["cross_track_abs_mean_m"] = [
            (t / n) if n else {"value": None, "reason": "no samples"}
            for t, n in zip(self.cross_track_sum, self.cross_track_n)]
        out["cross_track_rms_m"] = [
            (q / n) ** 0.5 if n else {"value": None, "reason": "no samples"}
            for q, n in zip(self.cross_track_sq, self.cross_track_n)]
        out["cross_track_samples"] = list(self.cross_track_n)
        if self.suite == "A":
            out["encountered"] = int(sum(self.encountered))
            out["conditional_cleared"] = t.conditional_rate(int(sum(self.encountered)))
        if self.suite == "O":
            out["hold_interruptions"] = [d.interruptions for d in self.detectors]
        if self.suite == "T":
            out.update(self._traffic_out())
        return out

    def _clean_among_contended(self) -> dict:
        """Clean runs among the trials that actually met traffic. N/A, never 0, when none did."""
        from .tally import na
        idx = [i for i in range(self.n) if self.contended[i]]
        if not idx:
            return na("no trial came within the contention range of an opponent")
        return {"value": sum(1 for i in idx if (self.outcome[i] or {}).get("success")) / len(idx),
                "reason": None}

    def _traffic_out(self) -> dict:
        """The T block: per-trial traffic arrays plus the cell-level counts derived from them.

        Everything here is per trial and in the trial's own order, so the report can pool it the
        same way it pools every other array -- by summing numerators and denominators rather than
        averaging per-trial rates, which would weight a trial that crashed at 2 s like one that ran
        the whole stint.
        """
        pairs = self.pair_detectors
        passes = [sum(d.passes for d in row) for row in pairs]
        lost = [sum(d.repasses for d in row) for row in pairs]
        reseeds = [sum(d.reseeds for d in row) for row in pairs]
        traces = [tr.as_dict() for tr in self.traffic]
        reasons = [(o or {}).get("reason") for o in self.outcome]
        out = {
            "n_opponents": self.n_opponents,
            "contention_range_m": self.contention_range_m,
            "attack_range_m": self.attack_range_m,
            "passes": passes,
            "leads_lost": lost,
            "opponent_respawns": reseeds,
            "contention_s": [t["contention_s"] for t in traces],
            "following_s": [t["following_s"] for t in traces],
            "attack_s": [t["attack_s"] for t in traces],
            "defending_s": [t["defending_s"] for t in traces],
            "opponent_progress_m": [t["opponent_progress_m"] for t in traces],
            "pace_ratio": [t["pace_ratio"] for t in traces],
            "closest_arc_gap_m": [
                t["closest_arc_gap_m"] if t["closest_arc_gap_m"] is not None
                else {"value": None, "reason": "no opponent sampled"} for t in traces],
            # Cell-level, all derived from the arrays above so the two cannot disagree.
            "contended": int(sum(self.contended)),
            "car_contacts": sum(1 for r in reasons if r == "contact"),
            "wall_collisions": sum(1 for r in reasons if r == "collision"),
            # NOT `Tally.conditional_rate`: that helper enforces successes <= encountered, which is
            # right for avoidance -- you cannot clear an obstacle you never reached -- and wrong
            # here, because a stint whose opponent crashed on lap one is a clean run that met no
            # traffic. Computed over the contended trials directly instead.
            "clean_conditional": self._clean_among_contended(),
            # The event scenario's own evidence. `trials` is the count whose contention window an
            # event actually landed in; `seconds` splits it from event time the learner was too far
            # away to care about. Both are 0 on a cell with no events, which is the truth there.
            "event_in_window_trials": int(sum(self.event_in_window)),
            "event_in_window_s": list(self.event_in_window_s),
            "event_seen_s": list(self.event_seen_s),
        }
        return out


def _signed_to(s: float, target: float, length: float) -> float:
    """Signed arc from `target` to `s`, wrapped. Positive once past it."""
    return (s - target + length / 2.0) % length - length / 2.0


def route_progress_fraction(progress_m, lengths) -> list:
    """Signed progress over track length. Reversing subtracts; required before cross-track compare."""
    return [float(p) / float(L) for p, L in zip(progress_m, lengths)]


def guard_rear_box(sim) -> float:
    """Assert the live rear-box draw stays inside the bound the thresholds were derived from."""
    return assert_rear_box_within_bound(sim.car_rear[:, 0])


# ---------------------------------------------------------------- the live loop

class _PreResetTrace:
    """Wraps `sim.step` so every measurement is read before the auto-reset rewrites the buffers.

    The gym layer resets *after* `sim.step` returns, and the simulator reuses its tensors, so a
    measurement taken after the call belongs to the next episode. Everything captured here is
    cloned on the spot for the same reason.
    """

    def __init__(self, env):
        self.env, self._orig, self.pending = env, env.sim.step, None

    def __enter__(self):
        def wrapped(*a, **kw):
            entry_steps = self.env.sim.steps.clone()
            entry_mu = self.env.sim.P["mu"].clone()
            r = self._orig(*a, **kw)
            exit_mu = self.env.sim.P["mu"].clone()
            if not bool((entry_mu == exit_mu).all()):
                raise MuChangedError("mu changed across a single sim.step call")
            lr = getattr(self.env, "last_result", None)
            cc = getattr(r, "car_collision", None)
            if cc is None and lr is not None:
                cc = getattr(lr, "car_collision", None)
            self.pending = {"entry_steps": entry_steps,
                            "steps_now": self.env.sim.steps.clone(),
                            "mu": exit_mu,
                            "speed": self.env.sim.state[:, 3].clone(),
                            "state": self.env.sim.state.clone(),
                            "heading_err": self._heading_error(),
                            # THIS step's IMU, from the result just returned -- not
                            # `env.last_result`, which still holds the previous step's samples. The
                            # stale read lagged by a frame and could miss the peak on the terminal
                            # step, which is exactly the step a spin or a crash happens on.
                            "yaw_rate": _yaw_rate_of(r, lr),
                            "lateral": (getattr(r, "lateral", None).clone()
                                        if getattr(r, "lateral", None) is not None else None),
                            "wall_dist": (lr.wall_dist.clone()
                                          if lr is not None and hasattr(lr, "wall_dist") else None),
                            # the actual car-to-car flag, so contact is labelled from evidence
                            # rather than inferred from "it was a race, so it was probably a car"
                            "car_contact": None if cc is None else cc.clone(),
                            "s": self.env.sim.s.clone()}
            return r
        self.env.sim.step = wrapped
        return self

    def __exit__(self, *exc):
        self.env.sim.step = self._orig
        return False

    def _heading_error(self):
        """Heading minus lane direction, from the tangent at the car's own centreline index.

        The previous version read `track.cl_yaw`, which does not exist -- `TrackTensors` carries
        `cl_tangent` -- and a catch-all turned that into zeros, so wrong-way and spin exposure were
        silently always zero. No catch-all here: if the track API changes, this must fail loudly
        rather than quietly report clean driving.
        """
        import torch
        env = self.env
        tg = env.sim.track.cl_tangent[env.sim.tid, env.sim.cl_idx]      # (B, 2)
        yaw_c = torch.atan2(tg[:, 1], tg[:, 0])
        err = env.sim.state[:, 2] - yaw_c
        return (err + torch.pi) % (2 * torch.pi) - torch.pi             # wrapped to (-pi, pi]

    def fresh(self):
        """True where the env ENTERED this step just reset (sim.py:339 zeroes steps on reset).

        Read at entry, not across the call: the trace runs before the gym auto-reset, so the reset
        that matters for this step happened after the previous one returned.
        """
        return (self.pending["entry_steps"] == 0).tolist()


def run_cell(env, policy, *, suite: str, n_steps: int, s_obs_m=None, hold_steps: int = 40,
             vehicle_length: float = 0.58, vehicle_width: float = 0.31, frozen: bool = True,
             controller=None, seed: int | None = None, obs_spec: dict | None = None,
             contention_range_m: float = 12.0, attack_range_m: float = 3.0) -> dict:
    """Drive one cell to completion and return its record.

    `policy(obs) -> action` is any callable. The loop never inspects what produced it, which is what
    lets a scripted non-candidate policy exercise the whole path without scoring a roster system.

    `controller` is the adapter's `PreparedCell` (or None for a scripted run). Its lifecycle is
    explicit and load-bearing: one seeded reset here, then `begin(obs)`, then `pre_action(obs)`
    before every policy call and `post_step(term, trunc)` after every env step. Skipping
    `pre_action` would leave the estimator's history un-fed, so the arm would run on a stale
    friction belief while still being reported under its name.
    """
    import torch

    length = float(env.sim.track.length[env.sim.tid].max())
    on_policy = env.on_policy if env.M > 1 else torch.ones(env.B, dtype=torch.bool)
    rows = torch.nonzero(on_policy).flatten().tolist()
    n = len(rows)
    n_opponents = (int(env.M) - 1) if (env.M > 1 and env.sim.other_idx is not None) else 0
    rec = CellRecorder(n=n, suite=suite, track_length_m=length, vehicle_length=vehicle_length,
                       vehicle_width=vehicle_width, hold_steps=hold_steps, s_obs_m=s_obs_m,
                       frozen=frozen, n_opponents=n_opponents,
                       contention_range_m=contention_range_m, attack_range_m=attack_range_m)
    if suite == "T" and n_opponents == 0:
        raise ValueError("a T cell with no opponent measures nothing: the traffic family needs "
                         "race_size > 1 and a non-candidate opponent")
    guard_rear_box(env.sim)

    # Exposure counters come from the project's own validated implementation rather than a second
    # one written here. It is fed the same pre-reset transitions.
    acc = None
    try:
        import numpy as _np
        from f1sim.learn.evaluation_metrics import TrialAccumulator
        acc = TrialAccumulator(_np.full(n, length, dtype=float), float(env.sim.control_dt),
                               time_budget_s=n_steps * float(env.sim.control_dt),
                               speed_cap=float(env.ecfg.speed_cap))
        # An avoidance clear or a held pass ends the measured task, but the accumulator only knows
        # about laps and collisions. Retiring its row here stops it integrating exposure for a
        # trial that has already been scored.
        rec.on_finish = lambda i: acc.active.__setitem__(i, False)
    except Exception:
        acc = None

    obs = _reset_obs(env, seed)                     # the single FINAL seeded reset
    # Captured HERE: after the seeded reset, before `begin` and before any action. `begin` pushes a
    # history row, so a fingerprint taken after it records part of the run rather than its start.
    from .fingerprint import start_fingerprint
    fp = start_fingerprint(env, obs, obs_spec=obs_spec)
    if controller is not None:
        controller.begin(obs)
    rec.bind_mu(env.sim.P["mu"].tolist())          # every car, opponents included
    dt = float(env.sim.control_dt)
    prev_s = env.sim.s[on_policy].clone()

    if suite == "O" and env.sim.other_idx is not None:
        g0 = env.signed_gaps(env.sim.s, env.sim.tid)[on_policy][:, 0]
        rec.seed_gaps(g0.tolist())                  # before the first transition
    prev_opp_s = None
    if suite == "T":
        # Every opponent of every learner, seeded before the first transition for the same reason
        # the O family seeds one: doing it inside the first `update` would eat that transition's
        # terminal flags.
        rec.seed_pair_gaps(env.signed_gaps(env.sim.s, env.sim.tid)[on_policy].tolist())
        prev_opp_s = env.sim.s[env.sim.other_idx[on_policy]].clone()      # (n, M-1)

    with _PreResetTrace(env) as trace:
        for k in range(n_steps):
            if controller is not None:
                controller.pre_action(obs)
            action = policy(obs)
            obs, reward, term, trunc, info = _step(env, action)
            if controller is not None:
                controller.post_step(term, trunc)
            pend = trace.pending
            fresh = trace.fresh()
            rec.check_mu(pend["mu"].tolist(), fresh)   # every car

            s_now = pend["s"][on_policy]
            progress = _wrap_delta(s_now - prev_s, length).tolist()
            prev_s = s_now.clone()

            gap = opp_reset = opp_incident = None
            pair_gaps = pair_progress = pair_reset = opp_event = None
            car_contact = pend["car_contact"]
            car_contact = ([bool(car_contact[i]) for i in rows] if car_contact is not None
                           else [False] * n)
            if env.M > 1 and env.sim.other_idx is not None:
                gap = env.signed_gaps(pend["s"], env.sim.tid)[on_policy][:, 0].tolist()
                opp_idx = env.sim.other_idx[on_policy][:, 0].tolist()
                opp_reset = [fresh[j] for j in opp_idx]
                # this step's own terminal flags for the paired opponent, read pre-reset
                opp_incident = [bool(term[j]) or bool(trunc[j]) for j in opp_idx]
            if suite == "T":
                all_idx = env.sim.other_idx[on_policy]                    # (n, M-1)
                pair_gaps = env.signed_gaps(pend["s"], env.sim.tid)[on_policy].tolist()
                opp_s = pend["s"][all_idx]
                # Wrapped, exactly like the learner's own progress: an opponent crossing the line
                # would otherwise show one lap of negative arc in a single step and turn the pace
                # ratio into nonsense at the one place a race is usually decided.
                pair_progress = _wrap_delta(opp_s - prev_opp_s, length).tolist()
                prev_opp_s = opp_s.clone()
                pair_reset = [[fresh[j] for j in row] for row in all_idx.tolist()]
                ev = info.get("opp_event") if isinstance(info, dict) else None
                if ev is not None:
                    # An event is "running" for a learner when ANY of its own opponents is in one.
                    # Read per pair rather than per env: with two opponents the second one's brake
                    # is as much a thing to react to as the first one's.
                    ids = ev["id"][all_idx]
                    opp_event = (ids != 0).any(dim=1).tolist()

            if acc is not None:
                import numpy as _np
                # Re-assert the exposure mask from the TASK's activity before every update. The
                # accumulator retires a row at its own first lap -- 12.5 s of a 50 m lap -- while an
                # overtake task runs to its own budget, so exposure stopped a third of the way in
                # and slip/spin were under-counted for every O cell. `on_finish` can only stop a row
                # early; it cannot revive one the accumulator already retired. This keeps the two
                # masks identical each step without touching the frozen implementation.
                acc.active[:] = _np.asarray(rec.active, dtype=bool)
                st = pend["state"][on_policy]
                yaw_rate = pend["yaw_rate"]
                acc.update(_np.asarray(progress, dtype=float),
                           _np.asarray(pend["speed"][on_policy].tolist(), dtype=float),
                           _np.asarray(term[on_policy].tolist(), dtype=bool),
                           _np.asarray(trunc[on_policy].tolist(), dtype=bool),
                           yaw_rate=(None if yaw_rate is None
                                     else _np.asarray(yaw_rate[on_policy].tolist(), dtype=float)),
                           heading_error=_np.asarray(pend["heading_err"][on_policy].tolist(),
                                                     dtype=float),
                           longitudinal_speed=_np.asarray(st[:, 3].tolist(), dtype=float),
                           lateral_speed=_np.asarray(st[:, 4].tolist(), dtype=float))
            lateral = pend.get("lateral")
            rec.update(lateral=None if lateral is None else lateral[on_policy].tolist(),
                       progress=progress, speed=pend["speed"][on_policy].tolist(), dt=dt,
                       collision=term[on_policy].tolist(), truncated=trunc[on_policy].tolist(),
                       s=s_now.tolist(), gap=gap, car_contact=car_contact,
                       opponent_reset=opp_reset, opponent_incident=opp_incident,
                       pair_gaps=pair_gaps, pair_progress=pair_progress, pair_reset=pair_reset,
                       opp_event_active=opp_event,
                       # inclusive budget: the last step of the budget may still complete
                       budget_exhausted=(k == n_steps - 1))
            if not any(rec.active):
                break

    out = rec.finalize()
    if acc is not None:
        out["spin_events"] = int(acc.spin_events)
        out["large_slip_seconds"] = float(acc.large_slip_seconds)
        out["wrong_way_seconds"] = float(acc.wrong_way_seconds)
        out["max_abs_yaw_rate"] = float(acc.max_abs_yaw_rate)
        out["time_budget_s"] = acc.time_budget_s
        out["speed_cap"] = acc.speed_cap
    out["track_length_m"] = length
    out["start_fingerprint"] = fp
    return out


def _yaw_rate_of(result, fallback):
    """Mean yaw rate from a StepResult's IMU samples, preferring the current step's."""
    for src in (result, fallback):
        imu = getattr(src, "imu", None) if src is not None else None
        if imu is not None and imu.shape[1] > 0:
            return imu[:, :, 2].mean(1).clone()
    return None


def _reset_obs(env, seed=None):
    """The single FINAL seeded reset, after graphs, controller and routing are installed.

    `F1VecEnv.reset(seed=...)` reseeds both `sim.gen` and the global RNG (gym_env.py:510). That
    matters because graph warm-up and controller construction happen first and consume the global
    RNG by different amounts per arm -- an unseeded reset here would give each system a different
    physical start, and the paired comparison the whole benchmark rests on would be between
    different initial conditions.
    """
    r = env.reset(seed=seed) if seed is not None else env.reset()
    return r[0] if isinstance(r, tuple) else r


def _step(env, action):
    r = env.step(action)
    if len(r) == 5:
        return r
    obs, reward, done, info = r          # older gym tuple
    return obs, reward, done, done, info


def _wrap_delta(d, length):
    return (d + length / 2.0) % length - length / 2.0
