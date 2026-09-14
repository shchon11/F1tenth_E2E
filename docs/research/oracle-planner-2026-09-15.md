# The oracle planner: hand it the other car, and see whether it drives better (2026-09-15)

Branch `feat/oracle-planner`, base main `0b78111`, independent of `feat/motion-memory`. Everything
here is simulation.

## The question, and why it is the other end of the last one

[motion-memory-2026-09-14](motion-memory-2026-09-14.md) asked whether the policy's representation
carries the nearest opponent's present relative state and answered *barely*: velocity at R² 0.09 …
0.28 and position at 0.11 … 0.27, while the ego's own speed reads 0.79 … 0.94 on the same rows with
the same read-out. [future-head-2026-09-14](future-head-2026-09-14.md) had already failed to get the
opponent's *future* out of the same state. Both notes end in the same place: PPO gives the recurrent
state no reason to build frame-to-frame correspondence on a few car-shaped beams.

There are two ways to read that, and they call for opposite work:

1. **The planner wants the opponent's motion and cannot get it.** Then perception is the
   bottleneck, and an encoder that delivers it is worth building.
2. **The planner would not use it if it had it.** Then no encoder can help, because the thing that
   would consume its output does not consume ground truth either.

Nothing measured so far separates them, because every experiment has been on the perception side.
This note tests the other end, which is the cheaper test by a wide margin: **give the planner the
opponent's true state, for free, from the simulator, and see whether racing improves.**

That makes this an *oracle* and not a policy. Nothing in this branch is deployable, and the loaders
say so rather than trusting anybody to remember.

## The block

`--opp-token off|pos|posvel|future` (`f1sim/opp_token.py`, `docs/training.md`). A proprio block for
the nearest two opponents, in the ego's body frame, appended after every existing proprio key:

| mode | per car | columns | proprio width |
|---|---|---|---|
| `off` | — | — | 366 |
| `pos` | 3 | Δx, Δy, present | 372 |
| `posvel` | 5 | + Δv_x, Δv_y | 376 |
| `future` | 13 | + (f_x, f_y) at +0.10 / +0.25 / +0.50 / +0.75 s | 392 |

`present` is 1 only where that slot holds a car inside `overtake_range` (12 m), and every other
column of the slot is multiplied by it — an absent, out-of-range or not-yet-spawned car contributes
zeros rather than a stale position. Positions and velocities are divided by `PRIV_OPP_DIST_SCALE`,
the normaliser `future_labels` and the critic's privileged opponent columns already use. Presence is
placed last in every mode so each mode's columns are a strict prefix of the next's.

### What the `future` columns actually are

A batched simulator cannot peek ahead, so the label is not ground truth. It is **where the
opponent's own controller intends to be**: every car of a race — teacher-driven or a pool checkpoint
— is driven through the same plan tracker, so there is one object to read, and for a teacher-driven
car it already carries that car's scheduled event (brake / stop / shift / defend / yield / line),
its speed scale and its follow cap, because all of those act on the action before the tracker sees
it.

The tracker produces two trajectories, and which one is the label was decided by measurement rather
than by preference:

* `ref` — `PlanTracker.last_ref`, the reference the tracker is chasing (the raceline target, or a
  pool policy's emitted plan, walked at the speed the car can actually have);
* `pred` — `PlanTracker.last_pred`, the **iLQR forward rollout**: where the tracker's own kinematic
  model says the car will be under the commands it just chose. Its clock starts at the
  latency-compensated pose, so it is read with that car's calibrated command delay subtracted.

The block's clock is offset by one control step: the plan's t = 0 is the instant the tracker was
called, and the observation the block goes into describes the state one `control_dt` later. Getting
that wrong is a systematic along-track bias of about 0.1 m at 4 m/s, in the same direction at every
horizon — the kind of error a pooled R² hides and an RMSE does not.

### The label, validated

`work/oracle-planner/work/label_validation.py`: the training distribution rolled out (three tracks,
race size 3, all seven opponent behaviours at 1.0 per 10 s, procedural obstacles, the frozen
original driving the learner), comparing the label at t against where that same car actually was at
t + h, in the ego's body frame at t, on rows where the token says the car is present and no car of
the race reset in between.

TABLE_LABEL_VALIDATION

`pred` is better than `ref` at every horizon and is what the token reports. `linear` is the fair
baseline for what a `posvel` policy can do for itself: position plus h × the opponent's velocity in
the ego frame at t, which the block's relative velocity and the proprio vector's own speed together
give it. From 0.25 s out the plan beats it by a factor of two to three, which is what makes `future`
a different arm rather than a re-parametrisation of `posvel`.

Read the RMSE, not the pooled R². `r2_xy` divides by the variance of the realised *position*, which
is mostly the variance of how far away the cars are, so anything that knows the current position
reads +0.99 — the same trap `motion-memory-2026-09-14.md` documents for its distance split.
`r2_disp`, which divides by the variance of the displacement over the horizon, is beside it.

## It is an oracle, and five things refuse it

RULES_REFUSAL

## The reward the audit demanded

REWARD_SECTION

## The arms

ARMS_SECTION

## Results

RESULTS_SECTION

## The verdict

VERDICT_SECTION

## What this licenses, and what it does not

LICENSE_SECTION
