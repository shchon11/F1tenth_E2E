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

Two rollout seeds, 24 envs x 400 steps each, ~16 000 scored rows per horizon per seed. RMSE in
metres, seed 4242 / seed 909:

| horizon | label (`pred`) | the reference (`ref`) | `linear` | `frozen` | R² | R²(disp) |
|---|---|---|---|---|---|---|
| 0.10 s | **0.029 / 0.032** | 0.049 / 0.049 | 0.019 / 0.019 | 0.405 / 0.376 | +1.000 | +0.994 / +0.992 |
| 0.25 s | **0.066 / 0.065** | 0.131 / 0.128 | 0.121 / 0.118 | 1.023 / 0.952 | +1.000 | +0.995 / +0.995 |
| 0.50 s | **0.193 / 0.181** | 0.292 / 0.284 | 0.471 / 0.455 | 2.092 / 1.963 | +0.999 | +0.989 / +0.990 |
| 0.75 s | **0.415 / 0.382** | 0.512 / 0.485 | 1.023 / 0.977 | 3.216 / 3.046 | +0.995 / +0.996 | +0.977 / +0.979 |

The stand-in plan a respawned car gets covers 0.43-0.52 % of rows.

Three readings, in order of how much they matter:

1. **`pred` beats `ref` at every horizon**, by 1.3x to 1.7x, on both seeds. That is what made it the
   label. Both are "its own plan"; one of them is measurably closer to where the car went.
2. **`linear` is better at 0.10 s and loses from 0.25 s on** — 0.019 m against 0.029 at a tenth of a
   second, 0.471 against 0.193 at half a second, 1.023 against 0.415 at three quarters. At the
   shortest horizon a straight line through the current velocity *is* the right answer and the plan
   is carrying the tracker's own model error; past that, the plan knows about the corner and the
   event and the follow cap, and the straight line does not. So A3 is told something A2 cannot work
   out, and the horizon where that becomes true is between 0.10 and 0.25 s.
3. **The two seeds agree to a few per cent on every cell**, which is what makes the first two
   readings safe to make at all.

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

Five, and they are independent on purpose. A checkpoint whose observation contains something the
car cannot produce is not a checkpoint with a caveat; it is a checkpoint that must not reach a
vehicle, and one guard with one reader is how such a thing reaches one anyway.

1. **`meta["opp_token"]`** — `ActorCritic` records the mode, so the fact travels with the weights.
2. **`extra["spec"]["opp_token"]`** — the trainer records the whole `ObsSpec`, so a checkpoint
   written by a path that did not set `meta` still says what it needs.
3. **`load_checkpoint(..., allow_oracle=False)`** is the default, and reads both. `learn/export.py`,
   `f1sim_ros/policy_node.py`, `learn/watch.py`, `learn/opponent_pool.py` and the viewer's worker
   all call it without the flag, so all of them refuse without having to know this feature exists.
   `learn/ppo.py`, `learn/evaluate.py` and `learn/benchmark/model_adapter.py` opt in, because each
   of them builds the simulator that produces the block.
4. **`ObsBuilder`** — the deployment-side observation builder — refuses the spec on its own. It is
   the thing that would have to build the block on the car, and it cannot.
5. **The benchmark refuses a solo cell.** An arm with the block has no observation at all without
   another car, and `build_cell` says so rather than feeding zeros to a policy trained to believe
   them. That is why the four arms are compared on family T and why
   `benchmark run --only` exists.

`learn/export.py` and `policy_node.py` deliberately have **no flag** to override it.

One more, less obvious: **a pool opponent is handed the observation without the block**
(`F1VecEnv._opponent_obs`). A `--opp-pool` entry is a deployable LiDAR-only checkpoint driving the
other car, which is the whole point of the population; giving it the oracle would make every car in
the race privileged, and the experiment would no longer be the one its name claims.

## The reward the audit demanded

The [reward audit](reward-audit-2026-09-13.md) left the traffic terms in a specific state: the only
thing that prices being near another car is `car_proximity`, charged **per metre driven** and only
inside `car_safe_gap`, and the only thing that prices touching one is a collision penalty that fires
on 0.2 % of steps after it is too late. The [traffic attribution](failure-attribution-2026-09-13.md)
§5 says what that misses: contacts are side-by-side, closing at +1.5 m/s, and half the wall
collisions happen while alongside or just behind the car being passed. A learner closing at 3 m/s
from two metres away — 0.7 s from a contact — pays nothing, because at two metres the gap is outside
the band.

Two terms, applied identically to every arm, both new `REWARD_COMPONENT_KEYS` entries, both default
off:

* **`--ttc-penalty λ --ttc-safe T`**: `−λ·max(0, (T − TTC)/T)` per step, with TTC the body-to-body
  gap over the line-of-sight closing speed to the most threatening car. Per step and not per metre
  driven, unlike the two proximity terms: what is priced is the time left, and standing still beside
  a car that is closing on you is exactly as close to a contact as driving past it is. A pair that is
  not closing has no time to contact and pays nothing.
* **`--overtake-sustained d T bonus`**: paid **once**, when the learner has led an opponent by ≥ d m
  along the lane, continuously, for ≥ T s — not on the crossing instant, which `--overtake-bonus`
  already pays for and which measured +0.03/s at coefficient 1.0, i.e. 1 % of the objective.

The second one had a trap in it that the first smoke found, and it is worth writing down because
every "pay for the outcome, not the approach" term has the same shape of trap. `--spawn-order
random` starts one race in three with the learner **ahead**, so "led by 1.5 m for 1.0 s" was true
from step one and the term paid for a grid position; a crashed opponent respawning behind the field
is the same bonus arriving mid-race. The payment is now blocked from the start of every race and
armed only by a step in which the gap is valid and this car is *not* leading — the only evidence
available that the lead, when it comes, was taken rather than given.

### What they are worth

Sized before the arms ran, by measuring both at coefficient 1.0 on the checkpoint every arm starts
from (`work/oracle-planner/work/reward_audit.py`, the 2026-09-13 method), so the coefficient is
arithmetic rather than a preference.

Measured at coefficient 1.0 on the frozen original, 5250 learner-steps, 24 episodes ended, progress
+4.156 /s on the same rows:

| term | at coefficient 1.0 | steps active | target share | coefficient | as trained |
|---|---|---|---|---|---|
| `ttc` | −3.4555 /s | 19.0 % | 6 % | **0.072** | −0.249 /s |
| `overtake_hold` | +0.0610 /s | 0.2 % | 8 % | **5.5** | +0.335 /s |

The two targets are not independent, and the constraint between them is the design:

> a **dense** cost on the approach must not exceed the **sparse** payment for the completed pass, or
> the optimal policy is to not approach.

TTC is charged on ~19 % of steps; the lead is paid once per pass. A penalty larger than the bonus
would make hanging back worth more per second than passing — the opposite of what these terms were
added for, and it would make all four arms measure the same uninteresting thing. 6 % puts TTC above
`car_proximity` (3.5 %) and `car_contact` (3.7 %), well below the collision penalty (20 %) — an
imminent contact priced below an actual one — and below the bonus. Both are far from the 0.8 % the
old overtake bonus measured at coefficient 1.0, which is the number the audit called effectively
absent.

The `overtake_hold` figure rests on about eight events in 5250 learner-steps, so its coefficient is
good to a factor rather than to a decimal.

and re-measured on a trained arm afterwards, because a term's share of the return is a property of
the policy as much as of the coefficient:

TABLE_REWARD_AUDIT

## The arms

Four arms, one difference each, from the recipe of
`work/learning-next/future-20260914/launch.sh` minus `--aux-future`, plus the two reward terms
above: memory GRU 128 with `memory,edges`, race size 3, `--opponent pool` (teacher, self, the frozen
original, A701, mem_u8) with the seven reactive behaviours at 1.0 per 10 s, per-reset procedural
obstacles, `--kl-coef 0.0`, `--plan-clearance-penalty 4.0 @ 0.25 m`, `--car-safe-gap 0.45`,
`--overtake-bonus 5.0`, warm-started from the frozen original, `--seed 701`, 1000 updates,
checkpoints every 100.

| arm | `--opp-token` | proprio | what the planner is told |
|---|---|---|---|
| **A0** | `off` | 366 | nothing; LiDAR, as today |
| **A1** | `pos` | 372 | where the nearest two cars are |
| **A2** | `posvel` | 376 | + how fast they are moving |
| **A3** | `future` | 392 | + where their own controllers intend to be, four horizons |

63 environments (21 races), not the source recipe's 258: the contract fixes the budget in *updates*,
and at 258 envs one update is 8256 env-steps against 2016, so four arms would be about 44 h of a
card two other workers are queued on. 63 x 3 is the width `feat/motion-memory`'s phase 2 used for
its own four-arm comparison, which makes the two branches' proxy numbers commensurable.

### Making "one difference" true

An arm that differs by an input width does not differ only by an input width, and three separate
things had to be fixed before this comparison meant what it says. All three are under
`--name-seed-fresh`, and the check that they worked is that **A0 and A3 log identical rollouts for
their first two updates** — which is what "the block's input columns are zero at init" has to mean,
and which no amount of reading the code proves.

1. **The modules a warm start leaves fresh.** A wider first layer has more parameters, so it draws
   more numbers from the ambient generator, so the GRU built after it differs. Each fresh module is
   initialised from `(--seed, its own qualified name)` instead
   (`learn.model.reinit_fresh_by_name`, copied from `feat/motion-memory`, where it was measured at
   up to 0.176 of difference on `actor.memory.gru.weight_ih_l0`).
2. **The exploration stream.** The generator is *left* in a different state for the same reason, so
   the arms' first sampled action differed and the rollouts diverged for a reason unrelated to the
   input. It is re-seeded once the model is built. The simulator's own generator was never
   affected — it is a separate `torch.Generator` seeded by `env.reset(seed=...)`.
3. **The optimiser.** Adam's moments for the two widened proprio layers were *dropped* in A1-A3 and
   *restored* in A0. They are element-wise, so they travel across the same column insert the weights
   do (`learn.model.grow_proprio_moment`): the block's own columns start at 0, which is what Adam
   holds for a coefficient that has not had a gradient yet.

### How the arms are read

`work/oracle-planner/work/decide.md`, written before any arm ran and not edited since, fixes the
resolution of every row and the rule that turns the table into a verdict. In short: an arm counts as
beating the control only on **car contacts per learner-minute, passes per learner-minute or family-T
completions**, by more than that row's resolution, with the two evaluation seeds agreeing on the
sign. Pace and wall collisions alone do not count — both are things a policy can buy by racing
differently rather than by using the opponent's state.

## Results

RESULTS_TABLE

### Which cell the numbers land in

`work/oracle-planner/work/decide.md` carries two things, deliberately separate. The **frozen rule**,
written before the arms ran, decides the one binary question the contract asks. The **reading
guide** appended afterwards (the user's, via root) says what the *shape* of the four arms means once
that is settled:

RESULTS_CELLS

### Does the planner read the block at all?

The all-equal cell cannot tell its own two causes apart, and they call for opposite next steps. So
each oracle arm is scored once more with its own block **zeroed** (`--opp-token-ablate`), width
kept. A large drop means the planner used the information and it did not pay; no drop means it never
read it, and a flat table then says nothing about whether the information is useful.

RESULTS_ABLATION

### The training curve, which is not a score

RESULTS_TRAIN

## The verdict

VERDICT_SECTION

## What this licenses, and what it does not

LICENSE_LICENSED

LICENSE_NOT

## What to do with it

LICENSE_NEXT
