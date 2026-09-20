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

Measured on trained A3 (`work/oracle-planner/work/audit_a3.json`), 63 envs x 800 steps, under the
reward the arms actually trained (`--overtake-bonus 0`):

| term | /s | steps active | share of progress |
|---|---|---|---|
| progress | +3.919 | 100 % | 100 % |
| **collision** | −0.810 | 0.2 % | **20.7 %** |
| **plan clearance** | −0.802 | 28.9 % | **20.5 %** |
| lap | +0.365 | 0.0 % | 9.3 % |
| steer rate | −0.272 | 99.9 % | 6.9 % |
| lap time | +0.241 | 1.1 % | 6.1 % |
| car contact | −0.190 | 0.1 % | 4.9 % |
| **sustained lead** | +0.170 | 0.1 % | **4.3 %** |
| **ttc** | −0.115 | 9.4 % | **2.9 %** |
| collision speed | −0.107 | 0.2 % | 2.7 % |
| car proximity | −0.061 | 5.6 % | 1.6 % |
| wall proximity | −0.042 | 11.6 % | 1.1 % |
| sideslip / wrong way / alive | −0.023 | — | 0.6 % |
| overtake (removed) | +0.000 | 0 % | 0 % |

Both new terms came in at **about half** their sized share — 2.9 % against 6 %, 4.3 % against 8 % —
and neither is the ~1 % the 2026-09-13 audit called effectively absent. They were sized on the
frozen original, which spends 19 % of its steps inside the TTC band against trained A3's 9.4 %: a
penalty successfully avoided reads small, which is the term working rather than failing.

Grouped, which is what matters for the result:

| group | per second | share of progress |
|---|---|---|
| **opponent** (ttc, sustained lead, car contact, car proximity) | 0.537 | **13.7 %** |
| **geometric safety** (plan clearance, collision, collision speed, wall proximity) | 1.760 | **44.9 %** |

The objective is **more than three times as much about staying off walls as about the other cars**,
and `plan_clearance` alone is larger than every opponent term combined.

A note on the audit's convention, because it is easy to misread: its `/s` figures are at the
**trained coefficients**, not at unit coefficient — the script builds the `EnvConfig` with them and
reads the reward as paid. The 2026-09-13 audit's +0.03 /s for the dense overtake term was at
coefficient 1.0; the same term measured here at unit coefficient is +0.765 /s, 25x that, which is
the configuration (race size 3, opponent pool, random spawn) and not the coefficient.

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
things had to be fixed before this comparison meant what it says. All three are under `--name-seed-fresh`.

What that buys, stated exactly, because the loose version is tempting and wrong:

* **The block cannot change the action at initialisation, bit-for-bit.** Feed an arm two completely
  different token blocks and `torch.equal` holds on the action
  (`tests/test_opp_token.py::test_the_new_proprio_columns_are_zero_and_cannot_move_the_action`).
  That is exact, because `0 * x = 0` exactly for finite `x`.
* **The arms' training trajectories are NOT identical, and cannot be.** A wider proprio layer is a
  wider GEMM and a GEMM reassociates its sum, so the *original* columns come out ~1e-9 different --
  the block contributes mathematically nothing and numerically a different rounding. Measured at
  warm start: |Δaction| ≤ 1e-7 on a [−1, 1] action, |Δvalue| ≤ 1e-6. PPO is chaotic, so 1e-9 is
  enough. A0 and A1 agree on **every logged rollout metric at update 1**, their *losses* already
  differ there, and by update 2 the rollouts have diverged.

So the arms are one experiment with one difference; they are not one trajectory. That is the usual
situation for any two seeds of one recipe, and it is why the comparison is read against a measured
resolution (`work/decide.md`) rather than against trajectories matching.

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
clean rate**, by more than that row's resolution, with the two evaluation seeds agreeing on the
sign. Pace and wall collisions alone do not count — both are things a policy can buy by racing
differently rather than by using the opponent's state.

## Results

1000 updates each, warm-started from the frozen original, `--seed 701`, evaluated at u1000 on two
proxy seeds and all 80 suite-T cells. Full table: `work/oracle-planner/work/arm_table.md`.

| row | res | A0 `off` | A1 `pos` | A2 `posvel` | A3 `future` |
|---|---|---|---|---|---|
| collisions / km ↓ | 2.2 | 10.8 | 9.9 | 6.8 | 8.0 |
| wall collisions / learner-min ↓ | 0.11 | 0.92 | 0.97 | 0.33 | 0.59 |
| car contacts / learner-min ↓ | 0.45 | 1.56 | 1.42 | 1.34 | 1.34 |
| passes held / learner-min ↑ | 0.25 | 1.87 [1.79/1.96] | 2.31 [2.30/2.31] | 2.22 [2.16/2.29] | 2.11 [2.11/2.10] |
| pace vs the opponents ↑ | 0.073 | 1.263 | 1.313 | 1.286 | 1.304 |
| sustained leads / learner-min ↑ | 0.61 | 2.43 ±0.26 | 2.90 ±0.30 | 2.63 ±0.27 | 2.85 ±0.27 |
| suite v2.1 family T clean rate ↑ | 0.056 | 328/640 (51.2 %) | 261/640 (40.8 %) | 349/640 (54.5 %) | 356/640 (55.6 %) |

> **What the family-T row measures.** A family-T "success" is a **clean run** — a trial that met traffic and finished with no wall collision and no car contact — and **not a completed lap**: no system completes a lap in family T, and T has no timeout outcome, so a slow car reads clean (worker 18, 2026-09-16).

**No row of this table attributes anything to the tokens**, for the reason the next two sections
give. Reported because the contract asks for it, and because the spread across it is itself the
evidence for how large single-run variance is here: A1 is the best arm on passes and the worst of
all four on the held-out suite, 10.4 pp below the *control* against a 5.6 pp bar. One training run
cannot be both "position helps most" and "position hurts most".

### Which cell the numbers land in

`work/oracle-planner/work/decide.md` carries two things, deliberately separate. The **frozen rule**,
written before the arms ran, decides the one binary question the contract asks. The **reading
guide** appended afterwards (the user's, via root) says what the *shape* of the four arms means once
that is settled:

**(b) — the planner / objective does not use the information.** Reached not by the arms being
numerically equal, which they are not, but by the ablation below, which is the stronger route: the
arms that differ do not differ *because of the tokens*.

Not the other cells. `A1 < A2` and `A2 < A3` cannot be claimed, because the differences are not
caused by the treatment. `A1 ≈ A3` is true numerically and says nothing about geometry versus
future. The privileged-shortcut cell does not hold — no arm shows training up with held-out down;
A3 has the best training reward *and* the best family-T clean rate. The joint interaction+geometry cell does
not hold — A3 has the fewest car contacts and is not worst on walls.

`posvel ≈ future >> pos`, the outcome the user hoped for, is **not** what happened: A2 ≈ A3, but A1
is worse than the control, and none of it is attributable. That outcome would have required the
tokens to be consumed.

### Does the planner read the block at all?

The all-equal cell cannot tell its own two causes apart, and they call for opposite next steps. So
each oracle arm is scored once more with its own block **zeroed** (`--opp-token-ablate`), width
kept. A large drop means the planner used the information and it did not pay; no drop means it never
read it, and a flat table then says nothing about whether the information is useful.

| row | res | A1 `pos` | A2 `posvel` | A3 `future` |
|---|---|---|---|---|
| car contacts / learner-min | 0.45 | 1.35 → 1.27 (−0.08) | 1.21 → 1.38 (+0.18) | 1.20 → 1.21 (+0.01) |
| passes held / learner-min | 0.25 | 2.30 → 2.29 (−0.01) | 2.16 → 2.06 (−0.09) | 2.11 → 2.36 (+0.25) |
| pace vs the opponents | 0.073 | 1.270 → 1.285 | 1.275 → 1.297 | 1.287 → 1.245 |
| collisions / km | 2.2 | 10.0 → 9.2 | 6.3 → 7.3 | 6.9 → 7.3 |

Every shift is inside its row's resolution and several are *improvements* — noise, not a policy
losing something it depended on. A3's passes row is the sharpest: +0.25 with the block removed, at
the resolution, in the wrong direction for "the future columns are load-bearing".

**Zero is an in-distribution input**, so this is a fair question rather than an off-manifold probe.
Presence gating makes an absent slot exactly zero, i.e. the encoding for "no car within 12 m", and
that occurs on 1.0 % of steps in a three-car race and 4.4 % in a two-car one, with the second slot
absent on 6.6 % and 100 % respectively (`work/oracle-planner/work/token_zero_fraction.json`). A
policy that used the block would drive differently when told the road is clear. These do not.

The limitation worth naming: this answers *does it respond to this input at all*, not *does it use
the information correctly*. A stronger probe would feed a plausible-but-wrong block — a shuffled or
delayed opponent. The weaker question suffices here only because the answer is no response at all.

### The training curve, which is not a score

Median (`collisions / km`) and mean (the rest) over each arm's final 100 updates. In-distribution,
and here for one row of the reading guide only.

| row | A0 `off` | A1 `pos` | A2 `posvel` | A3 `future` |
|---|---|---|---|---|
| reward / step ↑ | 0.0716 | 0.0650 | 0.0735 | 0.0796 |
| collisions / km ↓ | 16.6 | 26.3 | 19.7 | 19.1 |
| episode progress [m] ↑ | 58 | 45 | 54 | 52 |
| reward: ttc / step | −0.00155 | −0.00207 | −0.00159 | −0.00115 |
| reward: sustained lead / step | +0.00343 | +0.00412 | +0.00524 | +0.00310 |
| reward: overtake / step | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| loss: aux opponent MSE ↓ | 0.0498 | 0.0441 | 0.0569 | 0.0462 |

Two things. **No privileged shortcut**: A3 has the best training reward and the best family T, so
the "training up, held-out down" cell does not hold. And **the `aux_opp` asymmetry named in advance
did not materialise** — `work/decide.md` predicted the auxiliary opponent head would collapse in the
oracle arms, since its target is an input column there; the MSE is 0.044-0.057 across all four, with
no ordering. Worth recording as a prediction that did not come true.

## The verdict

VERDICT_SECTION

## What this licenses, and what it does not

**Licensed.**

* *The `future` token's label is a good description of where the opponent actually went.* RMSE
  0.029 / 0.066 / 0.193 / 0.415 m at 0.10 / 0.25 / 0.50 / 0.75 s, two rollout seeds agreeing to a few
  per cent, ~16 000 scored rows per horizon per seed, presence-conditioned and masked across resets.
* *The tracker's iLQR rollout is a better label than the reference it chases*, at every horizon on
  both seeds, by 1.3x to 1.7x. Both were defensible as "its own plan"; one was measured.
* *A `posvel` policy cannot derive the `future` columns for itself beyond about 0.2 s.* A straight
  line through the opponent's velocity is BETTER than its plan at 0.10 s (0.019 m vs 0.029) and 2x to
  2.5x worse from 0.50 s on, because the plan knows about the corner, the scheduled event and the
  follow cap. The horizon at which A3 is told something new is between 0.10 and 0.25 s.
* *The four arms differ by their input width and nothing else.* Fresh modules seeded from their own
  names, the ambient generator re-seeded after the model is built, and the widened layers' Adam
  moments carried across the same column insert as the weights. The block's contribution to the
  action at init is **exactly** zero (bit-exact, tested); the arms' *trajectories* diverge from the
  first update regardless, because a wider GEMM reassociates its sum at ~1e-9 and PPO is chaotic.
  Not fixable, and the reason the comparison is read against a measured resolution.
* *`--opp-token off` is the run it was.* No module, no RNG draw, no proprio column; the frozen loss
  oracle (`tests/data/ppo_loss_oracle.json`) is bit-identical.
* *An oracle checkpoint cannot reach a car.* Six independent refusals, two of which are tested
  against a checkpoint carrying only half the metadata.
* *The two new reward terms are not the 1 % the audit called absent.* Sized by measurement to 6 %
  and 8 % of progress on the checkpoint every arm starts from.

* *The planner does not consume the opponent's state it is handed, at any of three levels of
  detail.* Zeroing an arm's own block moves nothing beyond the proxy's own resolution, on a block
  whose zero value is in-distribution (1.0-4.4 % of steps outright, 6.6-100 % for the second slot).
* *The objective the arms optimised is more than three times as much about geometry as about the
  other cars* — 44.9 % of progress against 13.7 %, with `plan_clearance` alone larger than every
  opponent term combined.
* *The two reward terms this branch added are not negligible* — 2.9 % and 4.3 % of progress, about
  half their sized targets, against the ~1 % the 2026-09-13 audit called effectively absent.

**Not licensed.**

* **Anything about the real car.** Nothing in this branch is deployable, and that is the point
  rather than a limitation.
* **Any claim that `--ttc-penalty` or `--overtake-sustained` is the right reward.** They were sized
  to be *felt*, which is the bar the audit set; whether they help is a different experiment, and
  every arm here carries both, so this branch cannot separate them from the tokens.
* **Any claim about the 0.75 s horizon specifically.** The four horizons enter as one block; no arm
  isolates them, and the label's own error grows fastest there.
* **Any statement about `--aux-opp`.** Its target is an input column in three of the four arms, so
  the term does not mean the same thing across them. `work/decide.md` names the asymmetry in
  advance, and the direction of the bias -- against finding a token effect -- is stated there.

* **Any attribution of any arm-table difference to the tokens.** Each arm is ONE training run and
  the ablation says the tokens are unused, so every between-arm difference is training-run variance
  until a second seed bounds it. That includes A2's five-times-resolution wall-collision advantage
  and A1's 10.4 pp family-T deficit, which are the two largest numbers in the table.
* **Any claim that opponent state is useless to a planner.** What is shown is that *this* planner,
  under *this* objective, does not read it. An objective that priced traffic comparably to geometry
  might.
* **Any claim about which horizon matters.** The four `future` horizons enter as one block and no
  arm isolates them; with the block unread, they are untested rather than tested and rejected.
* **Anything from the ablation about correct USE of the information.** It shows no response to the
  input at all; it does not test whether a policy that responded would respond correctly.

## What to do with it

1. **Do not build a motion encoder yet.** That was the decision this branch existed to inform, and
   the answer is that the consumer does not exist: a planner that ignores ground truth will ignore
   an estimate of it. `motion-memory-2026-09-14` and `future-head-2026-09-14` asked whether the
   representation *can* carry the opponent; this asks whether the planner *would use it*, and the
   two together say the perception work is premature.
2. **Make the objective about the other cars, then re-run one arm.** The opponent terms are 13.7 %
   of progress against geometry's 44.9 %. The cheapest decisive experiment is A2 (`posvel`, the
   arm that needs no future predictor) under a reward where those two groups are comparable, with
   the same ablation as the test of whether it is read. One arm, ~3.5 h.
3. **Two seeds per arm, or no arm-to-arm attribution.** This branch's single-seed design cannot
   separate a treatment effect from a training run, and the spread it produced is large — the same
   arm best on one instrument and worst on another. Any successor comparing arms needs a within-arm
   spread first; the ablation is what saved this one from over-claiming, and it should be standard
   rather than an afterthought.
4. **Keep the ablation, and strengthen it.** Zeroing answers "is this read at all". A shuffled or
   time-shifted opponent would answer "is it read *correctly*", which is the question once something
   does read it.
