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
from .overtake import PassDetector, outcome as pass_outcome
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

    # -- one transition, read pre-reset ----------------------------------------------------------
    def update(self, *, progress, speed, dt, collision, truncated, s=None, gap=None,
               car_contact=None, opponent_reset=None, opponent_incident=None,
               budget_exhausted=False, lateral=None):
        """Precedence is fixed and deliberate: a collision on the same step as a success is a
        failure. A car that touches something on the exit stride did not clear the obstacle, and a
        car that crashes as the hold completes did not complete a clean pass."""
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
             controller=None, seed: int | None = None, obs_spec: dict | None = None) -> dict:
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
    rec = CellRecorder(n=n, suite=suite, track_length_m=length, vehicle_length=vehicle_length,
                       vehicle_width=vehicle_width, hold_steps=hold_steps, s_obs_m=s_obs_m,
                       frozen=frozen)
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
            car_contact = pend["car_contact"]
            car_contact = ([bool(car_contact[i]) for i in rows] if car_contact is not None
                           else [False] * n)
            if env.M > 1 and env.sim.other_idx is not None:
                gap = env.signed_gaps(pend["s"], env.sim.tid)[on_policy][:, 0].tolist()
                opp_idx = env.sim.other_idx[on_policy][:, 0].tolist()
                opp_reset = [fresh[j] for j in opp_idx]
                # this step's own terminal flags for the paired opponent, read pre-reset
                opp_incident = [bool(term[j]) or bool(trunc[j]) for j in opp_idx]

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
