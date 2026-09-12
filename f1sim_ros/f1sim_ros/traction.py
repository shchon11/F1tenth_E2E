"""Traction guard for the real car: wheel-vs-body acceleration, brake lock and launch spin.

ROS-free and dependency-free on purpose (`math` only). Inside `policy_node` it is fed once per
`/odom` sample (50 Hz on this car) and shapes once per published command (the 40 Hz scan rate), and
the same class is replayed over the recorded bags offline by `scripts/replay_traction.py` -- one
implementation, two callers, so what is validated is what drives.

Where it is validated
---------------------
On the car, by replay over the 22 recordings under `real_data/`, where VESC-ERPM wheel speed
decelerates at -40 ... -143 m/s^2 while the IMU body decel stays inside -3 ... -16 m/s^2. See
`docs/ros2.md` and `evidence/wheelslip_bags.py`.

When this module was written the simulator could not produce either failure -- `dynamics.py` carried
no wheel rotation state and `odom.py` reported ground speed, so the simulated wheel speed WAS the
body speed and this residual was identically zero. Since 2026-09-13 it can (`vehicle.wheel_model`),
and this same class runs inside the simulator loop as the `tcs` controller arm
(`f1sim/f1sim/learn/traction_arm.py`). One implementation, three callers: the car, the bag replay,
and the simulator. See `docs/research/wheel-model-2026-09-13.md`.

The measurement
---------------
* wheel speed -- `/odom` `twist.twist.linear.x`, which on this car is `vesc_to_odom`'s ERPM-derived
  wheel speed, not ground speed. Under lock or spin it is simply wrong about the vehicle, which is
  exactly what makes it a slip sensor.
* body acceleration -- `/sensors/imu/raw` `linear_acceleration.x`, in m/s^2. This car publishes
  that field in **g**; converting it is the caller's job (`policy_node` already detects and scales,
  `calib/bagread.py` multiplies by G on read).

The wheel speed is differentiated over a *window*, never sample to sample: 7.2 % of the `/odom`
steps in these recordings are shorter than 15 ms, down to 0.29 ms, where one ERPM quantum reads as
110 m/s^2 of wheel acceleration the car never saw. See `TractionParams.min_diff_dt`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Physical ceiling on this car's *body* longitudinal acceleration, m/s^2. mu ~ 1.05 on the
#: competition floor times g; `evidence/wheelslip_bags.py` uses the same number. Every body
#: acceleration is clamped to +/- this before it is compared with the wheel, which is what makes
#: the detector survive the recordings' IMU shock spikes: the raw accelerometer reaches -101 m/s^2
#: at 20260826-173704 t=46.46 (an impact, not a deceleration), and an unclamped residual would read
#: that as "the body is losing speed faster than the wheel" and miss the hardest lock in the set.
A_BODY_MAX = 1.05 * 9.80665

OK, LOCK, SPIN = "ok", "lock", "spin"


@dataclass(frozen=True)
class TractionParams:
    """Detector and shaper constants. Defaults are justified against the 22 real bags in REPORT.md.

    Accelerations are m/s^2, speeds m/s, times s, currents A.
    """

    #: A rolling wheel cannot decelerate faster than the body, and the body cannot exceed
    #: `a_body_max` = 10.3 m/s^2 on this floor, so any wheel deceleration past this is a wheel that
    #: has stopped rolling -- no IMU needed. This absolute gate is required *alongside* the residual
    #: below, and it is what keeps the guard off the accelerometer's bad days: in five places across
    #: the set the raw IMU x reads +14 to +20 m/s^2 (above the friction bound, so not a body
    #: acceleration at all) through an ordinary 10-17 m/s^2 wheel-speed dip, and the residual alone
    #: called those locks. 18.0 = 1.75 * mu * g, and every labelled run in the must-catch set
    #: (|a_wheel| >= 30) clears it with 12 m/s^2 to spare.
    lock_accel: float = 18.0
    #: The same absolute gate for spin: a gripping wheel cannot gain speed faster than the body.
    #: 14.0 sits above mu*g (10.3) and under the +14.5 ... +27.6 m/s^2 the labelled spin runs reach.
    spin_accel: float = 14.0
    #: Enter `lock` when (body accel - wheel accel) exceeds this. 18.0 is below the smallest
    #: residual any labelled lock run with |a_wheel| >= 30 can produce -- a_wheel <= -30 against a
    #: body accel clamped at >= -10.3 gives >= 19.7 -- and above the quiet-driving floor: over every
    #: bag, the same windowed wheel accel while moving and away from a labelled run peaks at
    #: 20.4 m/s^2 in the worst bag and p99 <= 12.8, and that noise sits on top of a body accel of
    #: the same sign, so the residual stays far lower.
    lock_rate: float = 18.0
    #: Enter `spin` when (wheel accel - body accel) exceeds this. Matches the labelling rule in
    #: `evidence/wheelslip_bags.py` (a_wheel - a_body > 8 with a_wheel > 12), tightened because the
    #: body term here is clamped and filtered rather than zero-phase.
    spin_rate: float = 12.0
    #: Leave a state when the residual falls below `clear_frac` of the entry threshold *and* the
    #: slip has closed. Hysteresis, so one sample of noise cannot chatter the actions.
    clear_frac: float = 0.45
    #: Wheel/body speed disagreement, m/s, that keeps a state latched after the acceleration
    #: transient is over. A locked wheel sits near zero while the body still moves; releasing the
    #: brake the instant the wheel accel settles would re-lock it.
    slip_hold: float = 0.30
    #: Consecutive samples the entry condition must hold before a state is entered. Lock is 1: the
    #: hardest labelled locks are a single sample of the ERPM channel collapsing and waiting for a
    #: second one gives the brake another 20 ms to stay locked. Spin is 2, matching the `min_len=2`
    #: of the labelling rule, because a launch spin starts where the body speed is near zero -- the
    #: one place the ERPM channel throws isolated one-sample spikes -- and capping the command is
    #: not as time-critical as releasing a brake.
    lock_persist: int = 1
    spin_persist: int = 2
    #: Lock needs the body to actually have been moving: below this the wheel and the body are
    #: both near zero, the release has nothing to release towards, and the residual is
    #: quantisation. Gated on the guard's own body-speed estimate -- never on the raw wheel speed,
    #: because at standstill the ERPM channel throws single-sample spikes worth up to 106 m/s^2 of
    #: apparent wheel acceleration -- and on its recent *peak* rather than its current value, over
    #: `v_ref_hold`: the body speed a lock has to be judged against is the one the car had when the
    #: wheel let go, and by the time the detector has 20 ms of evidence the estimate is already
    #: coming down.
    v_lock_min: float = 1.00
    v_ref_hold: float = 0.20
    #: Spin needs drive torque. `/sensors/core` `current_motor`; None (topic absent, as in
    #: 20260725-152213) disables the check rather than blocking detection.
    spin_current_min: float = 5.0
    #: Longest a single state may stay latched. The body speed it is holding out for is pure IMU
    #: integration while the wheel is untrustworthy, and that drifts.
    max_hold: float = 0.40
    #: Shortest a state stays latched once entered, so a one-sample detection still produces a
    #: usable action instead of a single-cycle blip.
    min_hold: float = 0.06

    #: Low-pass corners, Hz. `wheel_fc` is deliberately high: the labelled lock runs are 40-240 ms
    #: long, and a 5 Hz corner would attenuate a two-sample event below the threshold.
    wheel_fc: float = 20.0
    imu_fc: float = 8.0
    #: Body-speed estimator: time constant, s, with which the estimate is pulled back onto the
    #: wheel speed while no event is active. The pull is additionally slew-limited to
    #: `a_body_max`, so the estimate can never move faster than the car physically can.
    v_body_tau: float = 0.15

    #: Bound on the body acceleration used in every comparison (see `A_BODY_MAX`).
    a_body_max: float = A_BODY_MAX
    #: Shortest interval, s, the wheel-speed derivative may be taken over. `/odom` is nominally
    #: 50 Hz but its stamps jitter: in the recordings consecutive samples land 1-3 ms apart
    #: (20260827-111616 t=5.541/5.542, 20260714-223707 t=2.878/2.881), where a per-sample
    #: difference of one ERPM quantum reads as 38-110 m/s^2 of wheel acceleration that the car
    #: never saw. Differentiating over a window instead of a sample is what rejects those.
    min_diff_dt: float = 0.015
    #: Longest interval the derivative may span: beyond this the two samples describe different
    #: manoeuvres and the derivative is not computed at all.
    max_diff_dt: float = 0.08
    #: A step larger than this is a gap, not a sample interval: re-seed instead of integrating
    #: across it.
    max_step_dt: float = 0.25

    #: While a lock is latched the body-speed estimate is decayed by at least this fraction of
    #: `a_body_max`, whatever the accelerometer says. The ERPM channel is the *driven* (rear) axle;
    #: with it sliding the rear tyres are at the sliding-friction limit while the free front axle
    #: still rolls, so the body decel is a substantial fraction of mu*g. Without this the estimate
    #: grows on IMU shock ringing -- at 20260826-173704 t=109.1 the raw accelerometer reads +15 to
    #: +20 m/s^2 through a wheel-speed collapse -- and the release target grows with it.
    lock_decay_frac: float = 0.40
    #: Lock action. The commanded speed is released towards `release_frac * body_speed` at
    #: `release_rate`, and while the lock is latched it may not be *reduced* faster than
    #: `brake_rate`. 0.9 leaves a little slip, which is where the tyre makes its peak force.
    release_frac: float = 0.90
    release_rate: float = 12.0
    brake_rate: float = 6.0
    #: Hard ceiling on the guard's authority: however wrong the body-speed estimate is, the lock
    #: release may not raise the commanded speed more than this above what the policy asked for.
    #: The two sensors this guard has cannot tell "the wheel is sliding at 7 m/s of body speed"
    #: from "the car has stopped and the estimate has not caught up" -- both look like a wheel at
    #: zero and an integrated body speed of 7 -- and one bag shows the difference costing 7.6 m/s
    #: of commanded speed without this cap (20260827-115713 t=40.96). 2.0 m/s still unloads the
    #: brake, which is the whole point, without handing the guard the throttle.
    release_max: float = 2.0
    #: Spin action. The command is capped at `body_speed + spin_margin` while spinning; when the
    #: state clears the cap is held for `spin_ramp` seconds more, growing at `a_body_max` (the
    #: fastest the body could legitimately be gaining speed), then dropped.
    spin_margin: float = 0.50
    spin_ramp: float = 0.40

    def validate(self) -> "TractionParams":
        """Refuse a parameter set that cannot mean anything. Returns self so it can be chained."""
        pos = ("lock_rate", "spin_rate", "lock_accel", "spin_accel", "slip_hold", "v_lock_min",
               "max_hold", "wheel_fc", "lock_decay_frac", "v_ref_hold", "release_max",
               "imu_fc", "v_body_tau", "a_body_max", "min_diff_dt", "max_diff_dt", "max_step_dt",
               "release_rate", "brake_rate", "spin_margin", "spin_ramp")
        for n in pos:
            v = getattr(self, n)
            if not (isinstance(v, float) and math.isfinite(v) and v > 0.0):
                raise ValueError(f"TractionParams.{n} must be a positive finite float, got {v!r}")
        for n in ("min_hold", "spin_current_min"):
            v = getattr(self, n)
            if not (isinstance(v, float) and math.isfinite(v) and v >= 0.0):
                raise ValueError(f"TractionParams.{n} must be a non-negative finite float, got {v!r}")
        for n in ("lock_persist", "spin_persist"):
            v = getattr(self, n)
            if not (isinstance(v, int) and v >= 1):
                raise ValueError(f"TractionParams.{n} must be an int >= 1, got {v!r}")
        if not 0.0 < self.clear_frac <= 1.0:
            raise ValueError(f"TractionParams.clear_frac must be in (0, 1], got {self.clear_frac}")
        if not 0.0 < self.lock_decay_frac <= 1.0:
            raise ValueError(f"TractionParams.lock_decay_frac must be in (0, 1], got {self.lock_decay_frac}")
        if not 0.0 <= self.release_frac <= 1.0:
            raise ValueError(f"TractionParams.release_frac must be in [0, 1], got {self.release_frac}")
        if self.min_diff_dt >= self.max_diff_dt:
            raise ValueError(f"min_diff_dt {self.min_diff_dt} must be under max_diff_dt {self.max_diff_dt}")
        if self.max_step_dt < self.max_diff_dt:
            raise ValueError(f"max_step_dt {self.max_step_dt} must be at least max_diff_dt {self.max_diff_dt}")
        if self.min_hold > self.max_hold:
            raise ValueError(f"min_hold {self.min_hold} must not exceed max_hold {self.max_hold}")
        return self


@dataclass(frozen=True)
class TractionState:
    """What the guard believes, after one `update`. Cheap to build, safe to log or store."""

    state: str = OK                 #: OK / LOCK / SPIN
    t: float = 0.0                  #: the timestamp handed to `update`
    dt: float = 0.0                 #: step used, 0.0 when the step was a gap or the first sample
    wheel_speed: float = 0.0        #: as measured
    wheel_accel: float = 0.0        #: windowed, filtered derivative of the wheel speed
    body_accel: float = 0.0         #: filtered IMU longitudinal acceleration, clamped
    body_speed: float = 0.0         #: the guard's plausible body speed
    slip: float = 0.0               #: wheel_speed - body_speed
    residual: float = 0.0           #: body_accel - wheel_accel (positive = the wheel is falling behind)
    changed: bool = False           #: this update entered or left a state
    detecting: bool = False         #: the derivative was usable this step
    locks: int = 0                  #: lock events entered since the last reset
    spins: int = 0                  #: spin events entered since the last reset

    @property
    def active(self) -> bool:
        return self.state != OK


class TractionGuard:
    """Detect wheel lock / spin from (wheel speed, body acceleration) and shape the speed command.

    Usage, once per control cycle::

        st = guard.update(t, wheel_speed, imu_ax, motor_current)
        cmd = guard.shape(cmd, accel_hint)

    `shape` uses the step `update` just took, so it must be called after `update` in the same cycle;
    calling it without a preceding `update` returns the command unchanged. `reset()` restores the
    exact constructed state -- the guard carries no history across it -- and a time step larger than
    `max_step_dt` re-seeds the estimators rather than integrating across the gap.
    """

    def __init__(self, params: TractionParams | None = None):
        self.p = (params or TractionParams()).validate()
        self.reset()

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Forget everything. After this the guard is indistinguishable from a fresh one."""
        self._t = None                  # last accepted timestamp
        self._hist: list[tuple[float, float]] = []   # (t, wheel_speed), newest last
        self._aw = 0.0                  # filtered wheel acceleration
        self._ab = 0.0                  # filtered, clamped body acceleration
        self._ab_seen = False
        self._v_body = 0.0
        self._state = OK
        self._since = 0.0               # seconds in the current state
        self._n_lock = 0                # consecutive samples over the lock / spin entry condition
        self._n_spin = 0
        self._v_ref: list[tuple[float, float]] = []   # (t, body speed) over the last v_ref_hold
        self._dt = 0.0
        self._locks = 0
        self._spins = 0
        self._last = TractionState()
        # shaper
        self._cmd_prev = None           # last speed this guard emitted
        self._t_shape = None            # `t` of the update the last `shape` call was paired with
        self._cap = None                # active spin cap
        self._cap_left = 0.0            # seconds of post-spin cap ramp remaining

    @property
    def state(self) -> TractionState:
        """The last state, without advancing anything."""
        return self._last

    # ---------------------------------------------------------------- detector

    def update(self, t, wheel_speed, imu_ax, motor_current=None) -> TractionState:
        """Feed one sample. `imu_ax` is body longitudinal acceleration in **m/s^2** (not g); None or
        a non-finite value holds the previous value. `motor_current` is `/sensors/core`
        `current_motor` in A, or None when the topic does not exist."""
        p = self.p
        t = float(t)
        if not math.isfinite(t) or not math.isfinite(float(wheel_speed)):
            return self._last                       # nothing usable; the previous belief stands
        v = float(wheel_speed)

        if imu_ax is not None and math.isfinite(float(imu_ax)):
            a_raw = max(-p.a_body_max, min(p.a_body_max, float(imu_ax)))
            if not self._ab_seen:
                self._ab, self._ab_seen = a_raw, True
            else:
                self._ab = _lp(self._ab, a_raw, self._dt or 0.02, p.imu_fc)

        if self._t is None or t - self._t > p.max_step_dt:
            return self._seed(t, v)                 # first sample, or a gap: re-seed, do not detect
        if t <= self._t:
            # A repeated or out-of-order timestamp is a sample to drop, not a reason to forget
            # everything: re-seeding here would clear a latched release mid-lock. `/odom` stamps in
            # these bags are not guaranteed monotonic (`calib/bagread.py` sorts each topic for
            # exactly that reason) and two records can share a nanosecond.
            return self._last

        dt = t - self._t
        self._t, self._dt = t, dt
        self._hist.append((t, v))
        while len(self._hist) > 2 and t - self._hist[1][0] >= p.max_diff_dt:
            self._hist.pop(0)                       # keep one sample older than the window

        detecting = False
        for (t0, v0) in reversed(self._hist[:-1]):  # newest first: narrowest usable window wins
            span = t - t0
            if p.min_diff_dt <= span <= p.max_diff_dt:
                self._aw = _lp(self._aw, (v - v0) / span, dt, p.wheel_fc)
                detecting = True
                break

        # Plausible body speed: integrate the (clamped) body acceleration, and while no event is
        # active pull it back onto the wheel speed -- the two agree under grip. The pull is itself
        # slew-limited, so no wheel-speed glitch can move this estimate faster than the car can go.
        if self._state == LOCK:
            # A sliding tyre is at the friction limit, so the body IS losing speed -- and the IMU is
            # the one signal that cannot be trusted during the event (raw spikes to +/-100 m/s^2 in
            # these bags). Falling back on the bound keeps the release target shrinking instead of
            # growing while the accelerometer rings.
            self._v_body += min(self._ab, -p.lock_decay_frac * p.a_body_max) * dt
        else:
            self._v_body += self._ab * dt
        if self._state == OK:
            pull = (v - self._v_body) * min(1.0, dt / p.v_body_tau)
            lim = p.a_body_max * dt
            self._v_body += max(-lim, min(lim, pull))

        self._v_ref.append((t, self._v_body))
        while len(self._v_ref) > 1 and t - self._v_ref[0][0] > p.v_ref_hold:
            self._v_ref.pop(0)

        residual = self._ab - self._aw              # > 0: the wheel is losing speed faster
        slip = v - self._v_body
        prev = self._state
        self._since += dt
        if detecting:
            self._advance(residual, slip, motor_current)
        if self._state != prev:
            self._since = 0.0
            if self._state == LOCK:
                self._locks += 1
            elif self._state == SPIN:
                self._spins += 1
        if self._state == OK and prev != OK:
            # The wheel is trustworthy again; stop holding out for an integrated body speed.
            self._v_body += max(-p.a_body_max * dt, min(p.a_body_max * dt, v - self._v_body))

        self._last = TractionState(
            state=self._state, t=t, dt=dt, wheel_speed=v, wheel_accel=self._aw,
            body_accel=self._ab, body_speed=self._v_body, slip=slip, residual=residual,
            changed=self._state != prev, detecting=detecting, locks=self._locks, spins=self._spins)
        return self._last

    def _seed(self, t, v):
        """Start (or restart) the estimators on this sample without claiming a detection."""
        self._t, self._dt = t, 0.0
        self._hist = [(t, v)]
        self._aw = 0.0
        self._v_body = v
        self._state, self._since = OK, 0.0
        self._n_lock = self._n_spin = 0
        self._v_ref = [(t, v)]
        self._cmd_prev, self._t_shape, self._cap, self._cap_left = None, None, None, 0.0
        self._last = TractionState(state=OK, t=t, wheel_speed=v, body_accel=self._ab,
                                   body_speed=v, locks=self._locks, spins=self._spins)
        return self._last

    def _advance(self, residual, slip, motor_current):
        """The state machine. Enter on the acceleration residual, hold until the slip closes."""
        p = self.p
        if self._state == OK:
            lock_now = (residual >= p.lock_rate and self._aw <= -p.lock_accel
                        and max(v for _, v in self._v_ref) >= p.v_lock_min)
            spin_now = (-residual >= p.spin_rate and self._aw >= p.spin_accel
                        and (motor_current is None or float(motor_current) >= p.spin_current_min))
            self._n_lock = self._n_lock + 1 if lock_now else 0
            self._n_spin = self._n_spin + 1 if spin_now else 0
            if self._n_lock >= p.lock_persist:
                self._state = LOCK
            elif self._n_spin >= p.spin_persist:
                self._state = SPIN
            return
        self._n_lock = self._n_spin = 0
        if self._since < p.min_hold:
            return                                  # too soon to leave
        if self._since >= p.max_hold:
            self._state = OK
            return
        if self._state == LOCK:
            # Still locking, or the wheel is still running well below the body.
            if residual < p.clear_frac * p.lock_rate and slip > -p.slip_hold:
                self._state = OK
        else:
            if -residual < p.clear_frac * p.spin_rate and slip < p.slip_hold:
                self._state = OK

    # ---------------------------------------------------------------- shaper

    def shape(self, cmd_speed, cmd_accel_hint=None) -> float:
        """Shape one speed command, m/s. Uses the state from the last `update`.

        * `lock` -- release towards `release_frac * body_speed` at `release_rate`, and refuse to cut
          the command faster than `brake_rate`. The guard never commands *less* than it was asked
          for, so it cannot make the car faster than the policy wanted, only less braked.
        * `spin` -- cap at `body_speed + spin_margin`, then hold the cap for `spin_ramp` seconds
          after the state clears, growing it at `a_body_max`.

        Every rate here is per second of *command* time, measured between consecutive `shape` calls
        off the timestamps `update` was given -- not the update step. The two differ on this car:
        `/odom` arrives at 50 Hz and the command goes out at the 40 Hz scan rate, and using the
        20 ms update step for a 25 ms command interval would quietly run every slew limit 20 % slow.

        `cmd_accel_hint` (m/s^2, the tracker's intended acceleration, or None) is accepted for
        symmetry with the tracker's own interface and currently unused: the lock action already
        refuses to lower the command and the spin cap already refuses to raise it, so a hint about
        which way the command was heading changes neither.
        """
        p = self.p
        if not math.isfinite(float(cmd_speed)):
            return cmd_speed
        cmd = float(cmd_speed)
        if self._t is None:
            self._cmd_prev = cmd
            return cmd
        dt = max(0.0, self._t - (self._t if self._t_shape is None else self._t_shape))
        self._t_shape = self._t
        prev = cmd if self._cmd_prev is None else self._cmd_prev
        out = cmd

        if self._state == LOCK:
            target = p.release_frac * self._v_body
            if target > prev:
                floor = min(target, prev + p.release_rate * dt)
            else:
                floor = max(target, prev - p.brake_rate * dt)
            # The authority cap goes on the output, not on the target: `brake_rate` alone would let
            # a command the policy has already cut to zero coast down from 8 m/s at 6 m/s^2 and sit
            # 7 m/s above it for the length of the hold.
            out = min(max(cmd, floor), cmd + p.release_max)
            self._cap, self._cap_left = None, 0.0
        elif self._state == SPIN:
            # Sign-symmetric, and never across zero: capping a forward command at a body speed that
            # has gone negative would command reverse, which is not what "cap" means.
            self._cap = max(0.0, self._v_body) + p.spin_margin
            self._cap_left = p.spin_ramp
            out = _cap(cmd, self._cap)
        elif self._cap is not None:
            # Ramping the cap back out after a spin: it grows at the fastest rate the body could
            # legitimately be gaining speed, and is dropped entirely after `spin_ramp`.
            self._cap_left -= dt
            if self._cap_left <= 0.0:
                self._cap, self._cap_left = None, 0.0
            else:
                self._cap += p.a_body_max * dt
                out = _cap(cmd, self._cap)
        self._cmd_prev = out
        return out


def _cap(cmd, cap):
    """Limit |cmd| to `cap` without changing its sign. `cap` is non-negative."""
    return min(cmd, cap) if cmd >= 0.0 else max(cmd, -cap)


def _lp(y, x, dt, fc):
    """One step of a causal first-order low-pass: the same pole `evidence/wheelslip_bags.py` uses
    forward and backward, run forward only because a controller cannot see the future."""
    a = math.exp(-2.0 * math.pi * fc * dt)
    return a * y + (1.0 - a) * x


__all__ = ["TractionGuard", "TractionParams", "TractionState", "A_BODY_MAX", "OK", "LOCK", "SPIN"]
