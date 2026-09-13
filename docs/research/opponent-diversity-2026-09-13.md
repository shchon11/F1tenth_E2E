# Opponents that behave like opponents (2026-09-13)

Follow-up to [the failure attribution](failure-attribution-2026-09-13.md) §5, which found that
traffic races end on walls during a pass and on side-by-side contacts closing at +1.5 m/s, and that
rear-ends essentially do not happen. The user's reading, which the data supports: the policy has
never seen an opponent that takes a different line, defends, yields, contests the same gap, or
simply does not see it.

One column of that table has to be withdrawn and re-measured before anything else, because it is
quoted as part of the diagnosis. §5 reports **0** contacts within 1 s of an opponent event; the
analysis script's event detector probed for a key the env does not use and returned "no event" on
every step, so the column was structurally zero rather than measured. Re-measured here with the
detector fixed (section 5 of this note; REPORT.md carries the defect): 4 of the frozen original's 10 contacts
fall within that window, against a 51 % base rate — so contacts are *under*-represented around
events rather than absent from them, and §5's conclusion holds on a measurement instead of on an
artefact. Everything else in §5 is untouched.

This note is what the training distribution contained, what it contains now, and how much of that
the learner actually experiences — in seconds.

## 1. What one opponent was

Before this branch, `--race-size 2` offered exactly one situation. `--opponent teacher` is the
raceline teacher at 0.6–1.0× of its own speed profile, blind to everything except the follow-gap
slowdown that makes it brake politely for a car ahead. `--opponent policy` is a copy of the learner.
`--opponent mixed` draws one or the other per race. And in every case **the learner spawned at the
back**, because an overtake is a car behind catching a car ahead.

The scripted behaviours built the day before (`brake`, `stop`, `shift`, `weave`) fire on a Poisson
timer and never look at the learner. They were accepted on "37 events in 32 races" — that they
fire. With the detector repaired, the contacts that follow one within a second are fewer than
chance would give (section 5), which is the same statement from the other side: an event that fires
while the learner is 40 m away is not a lesson.

## 2. What is there now

Three axes, all off by default, all measurable.

**A population, not an opponent** (`--opponent pool --opp-pool a.pt,b.pt,self,teacher`). One entry
drawn per race from the simulator's own generator. An entry is a checkpoint, `self` (the learner's
current weights — that race is self-play) or `teacher`. A checkpoint is the only opponent that
cannot be written down as a rule: it takes its own line because its own network decided to, defends
its position because being passed costs it reward, and makes its own mistakes. Loaded once, in
`eval()`, gradients off, one extra full-width forward per entry per step.

**Reactive behaviour** — four new names in the same `--opp-events` list, each a disposition drawn
per teacher-driven car per race at its own probability, each reading the learner's relative position
every step rather than a timer:

* `defend` — a learner inside `overtake_range` behind moves the opponent's line toward the side that
  learner is coming down, at full amplitude inside 3 m and ramping to nothing at 12 m.
* `yield` — a learner alongside moves the opponent away from it.
* `line` — an out-in or in-out offset sweep through each corner, mode and amplitude drawn per
  corner, from corner tables built once from the raceline's own smoothed curvature.
* `oblivious` — the follow-gap slowdown is switched off. This is the car that rams, and the only
  member of the population that produces "the car I just passed drove into me" on purpose rather
  than as a teacher bug.

The three offset behaviours share one output: summed, capped at 0.45 m, rate-limited to 0.6 m/s, and
then clamped per raceline point against the track's own distance field — the same budget the
scripted offsets use, so a reactive offset can no more reach a wall than a `shift` can. `oblivious`
produces no offset; it removes the cap rather than raising a command through it, so the invariant
"an event never lifts an opponent over its follow-gap cap" is untouched.

**A grid** (`--spawn-order behind|ahead|alongside|random`). `behind` is every race trained so far.
`ahead` is the only way the learner is the car being overtaken, and it needs `--opp-speed` above 1.0
to be one — which is now allowed and validated rather than clamped. `alongside` puts the cars
abreast, which is where §5 says the contacts happen. `--race-size 3` works with all of it.

## 3. Does the learner experience it? The census

`python -m f1sim.learn.opponent_census` runs N races with a given configuration and reports, per
situation, how many **learner-seconds** were spent in it. It takes the same flags as the trainer out
of one shared parser group (`learn/opponent_config.py`), so what it measures is a configuration
`ppo` can build, and it reads the same `gym_env.learner_view()` the reactive behaviours react to, so
"alongside" is one definition rather than two that share a name.

Both arms below are the frozen original driving, 32 races × 1200 steps (30 s) on the three smoke
tracks, seed 4401, CUDA. "Learner-seconds" are seconds driven by a car the policy drives whose
transitions a PPO update would use.

* **before** = today's opponent: `--race-size 2 --opponent mixed --mixed-teacher-frac 0.5
  --opp-speed 0.5 1.0` (recipe A's own setting), `--spawn-order behind`.
* **after** = `--opponent pool --opp-pool teacher,teacher,self,A701 --opp-events
  defend,yield,line,oblivious --opp-defend-prob 0.8 --opp-yield-prob 0.6 --opp-line-prob 0.8
  --opp-oblivious-prob 0.4 --spawn-order random --opp-speed 0.6 1.2`, where `A701` is
  `cl_origrecipe_legacy_s701/ppo_final.pt`, the best system on suite v1. `teacher` is listed twice
  because the draw is uniform over entries, so repetition is how a population is weighted: half the
  races get a scripted car carrying the reactive behaviours, a quarter are self-play, a quarter face
  a stronger trained policy.
* **after, three cars** = the same with `--race-size 3` (21 races × 3 cars), which is the only
  configuration in which "two opponents in range" can happen at all.

The first version of the after arm was **four distinct pool entries at probabilities
0.5 / 0.35 / 0.5 / 0.3**, and the census rejected it: defended against 1.3 % of learner-seconds seen
by 4 of 64 cars, yielded to 0.1 % by 2, oblivious 0.5 % by 4, own-line 2.5 % by 6. Present, but too
thin to be a training distribution — one teacher entry in four means the reactive behaviours are
available in a quarter of races before their own probability is applied. It is kept in
`out/config.sh` as `ARM_B0`, because "the census rejected this configuration" is the evidence that
the census is load-bearing rather than decorative.

### Situation census: learner-seconds, and how many of the learner's cars ever saw it

| situation | before: s | % | cars | after: s | % | cars | after, 3 cars: s | % | cars |
|---|---|---|---|---|---|---|---|---|---|
| behind a slower car within overtake_range | 287 | 19.3 % | 49/64 | 224 | 18.5 % | 39/64 | 262 | 28.2 % | 28/63 |
| alongside (bodies overlapping along the lane) | 47 | 3.2 % | 51/64 | 48 | 4.0 % | 42/64 | 37 | 3.9 % | 24/63 |
| being overtaken: a car behind inside the attack window, closing | 53 | 3.6 % | 26/64 | 21 | 1.7 % | 14/64 | 33 | 3.5 % | 17/63 |
| defended against: a car ahead is blocking my side, this step | 0 | 0.0 % | 0/64 | 111 | 9.1 % | 14/64 | 153 | 16.4 % | 11/63 |
| yielded to: a car alongside is moving away | 0 | 0.0 % | 0/64 | 14 | 1.2 % | 14/64 | 9 | 1.0 % | 10/63 |
| an oblivious car closing from behind | 0 | 0.0 % | 0/64 | 5 | 0.4 % | 5/64 | 18 | 1.9 % | 8/63 |
| two opponents within overtake_range at once | 0 | 0.0 % | 0/64 | 0 | 0.0 % | 0/64 | 488 | 52.5 % | 31/63 |
| *(context)* an opponent in range driving its own corner line | 0 | 0.0 % | 0/64 | 90 | 7.4 % | 16/64 | 183 | 19.7 % | 13/63 |
| *(context)* any opponent within overtake_range | 946 | 63.7 % | 56/64 | 816 | 67.2 % | 45/64 | 720 | 77.5 % | 31/63 |
| *(context)* a car ahead inside the attack window | 172 | 11.6 % | 52/64 | 128 | 10.5 % | 38/64 | 152 | 16.4 % | 28/63 |
| *(context)* nearest opponent in range is teacher-driven | 251 | 16.9 % | 26/64 | 260 | 21.4 % | 20/64 | 280 | 30.1 % | 15/63 |
| *(context)* nearest opponent in range is a pool checkpoint | 0 | 0.0 % | 0/64 | 189 | 15.6 % | 12/64 | 51 | 5.5 % | 4/63 |
| *(context)* nearest opponent in range is the policy itself (self-play) | 695 | 46.8 % | 48/64 | 367 | 30.3 % | 26/64 | 389 | 41.9 % | 15/63 |

* **before** — `race 2 | teacher / self-play per race | speed x0.5-1 | spawn behind gap 2.5-6 m | opponent events off`
  * 32 races x 1200 steps, 64 learner cars, 1486 learner-seconds, 96 episodes ended, seed 4401, 165 s wall on CUDA
  * grids drawn (cars that saw each): behind 56, ahead 0, alongside 0
* **after** — `race 2 | pool teacher,teacher,self,cl_origrecipe_legacy_s701/ppo_final.pt | speed x0.6-1.2 | spawn random gap 2.5-6 m | reactive defend p=0.8, yield p=0.6, line p=0.8, oblivious p=0.4`
  * 32 races x 1200 steps, 64 learner cars, 1213 learner-seconds, 108 episodes ended, seed 4401, 101 s wall on CUDA
  * grids drawn (cars that saw each): behind 29, ahead 23, alongside 21; pool loaded: 2:ppo_final.pt
* **after, 3 cars** — `race 3 | pool teacher,teacher,self,cl_origrecipe_legacy_s701/ppo_final.pt | speed x0.6-1.2 | spawn random gap 2.5-6 m | reactive defend p=0.8, yield p=0.6, line p=0.8, oblivious p=0.4`
  * 21 races x 1200 steps, 63 learner cars, 930 learner-seconds, 102 episodes ended, seed 4401, 209 s wall on CUDA
  * grids drawn (cars that saw each): behind 16, ahead 8, alongside 10; pool loaded: 2:ppo_final.pt

### Terminations in the same rollouts

| arm | learner car contacts | learner walls | learner coll/km | learner km | opponent car contacts | opponent walls | opponent walls/km |
|---|---|---|---|---|---|---|---|
| before | 38 | 24 | 9.74 | 6.36 | 12 | 0 | 0.00 |
| after | 27 | 33 | 10.21 | 5.87 | 19 | 2 | 0.72 |
| after, 3 cars | 21 | 25 | 10.18 | 4.52 | 23 | 3 | 0.81 |

### Reading

**Four of the seven situations did not exist.** `defended against`, `yielded to`, `an oblivious car
closing from behind` and `two opponents in range` are exactly 0.0 s in the before arm — not small,
zero — and so is "an opponent driving its own corner line". The learner has had 138.8 M steps of
training and has never once been blocked, moved away from, rammed by a car that did not brake, or
surrounded. The grid row says the same thing from the other end: 56 of 56 cars that met an opponent
started behind it.

**They exist now, and they are not incidental.** Being defended against is 9.1 % of learner-seconds
in the two-car arm and 16.4 % with three cars; an opponent driving its own line, 7.4 % and 19.7 %;
two opponents in range, 52.5 % of the three-car arm. Every grid is drawn (21–29 of the cars saw each
in the two-car arm). The population is real: the nearest car in range was a pool checkpoint for
15.6 % of learner-seconds and the teacher for 21.4 %, against 0 % and 16.9 % before.

**`yield` and `oblivious` are the thin ones**, at 1.2 % and 0.4 % of the two-car arm. Both need a
geometry the learner has to arrive at rather than start in — `yield` needs the learner *alongside* a
yielding car, and `alongside` is only 4 % of the rollout to begin with; `oblivious` needs the
learner *ahead* of an oblivious car and being caught, which is one grid in three times one pool
entry in two times p=0.4. They are experienced, by 5 and 14 of 64 cars, and the three-car arm lifts
`oblivious` to 1.9 % and 8 cars. A configuration that wants more of either can have it: measured
separately at `--spawn-order ahead --opp-oblivious-prob 1.0 --opp-yield-prob 1.0 --opp-speed 1.0
1.3` on `gen:competition:0`, "an oblivious car closing from behind" is 11.8 % of learner-seconds
over 6 of 16 cars and "being overtaken" 7.1 % over 4 — so the mechanism is not starved, the mixed
configuration dilutes it. Which is the number a census is for.

**`being overtaken` went down, 3.6 % → 1.7 %,** and that is worth naming rather than hiding. The
before arm produces it already, from a source that has nothing to do with opponents being diverse:
half of its races are self-play, whose *front* car is speed-capped (`selfplay_front_cap`) so the car
behind has a pass to make — and that capped front car is itself a learner row, being overtaken. Arm
B has a quarter as many self-play races, and replaces that source with a real one (a 1.2× teacher
plus the `ahead` grid) which lands at 1.7 %. The three-car arm is back at 3.5 %. So the honest
statement is that this axis was already covered, by accident, and is now covered on purpose; what
was missing is the other four.

**The denominators differ on purpose.** "Learner-seconds" counts cars the policy drives *and* whose
transitions an update would use. The before arm has 50 % self-play races, so more of its 64 rows are
on-policy at any moment (1486 s against 1213 s over the same 1200 steps). That is why every row is
also reported as a share, and why "how many cars ever saw it" is there: it is the column that cannot
be moved by a denominator.

**Nobody drove into a wall to make this happen.** The opponents' own wall rate is 0.72 / km in the
two-car arm and 0.81 / km with three, against 0.00 before — the price of a 1.2× speed scale and a
reactive offset, and two orders of magnitude below the learner's own 10 / km. The learner's total
collision rate is unchanged (9.74 → 10.21 / km), but its *composition* moves the way the situation
census predicts: 38 car contacts and 24 walls before, 27 and 33 after. Fewer contacts because the
opponents now sometimes move out of the way; more walls because the learner is being squeezed in
places it was never squeezed before, which is §2's margin problem seen from the side and is the
thing this distribution exists to train against.

## 4. The dense lateral signal: root's suspicion, confirmed

Analysis only. No reward was changed on this branch.

Recipe A charges `--car-proximity-penalty 0.8` per metre driven at zero body gap, ramping in below
`--car-safe-gap 0.9` m and scaled by closing speed. Root suspected it is nearly always zero. It is:

| quantity | arm | n | zero | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| `car_proximity` reward per learner step | before | 59428 | 93.4 % | 0.0000 | 0.0000 | 0.1034 | 0.2647 |
| `car_proximity` reward per learner step | after | 48534 | 93.1 % | 0.0000 | 0.0000 | 0.0972 | 0.2616 |
| `car_proximity` reward per learner step | after, 3 cars | 37194 | 93.0 % | 0.0000 | 0.0000 | 0.0923 | 0.2808 |
| &nbsp;&nbsp;... restricted to the steps spent alongside | before | 1897 | 0.4 % | 0.0727 | 0.1373 | 0.2120 | 0.2630 |
| &nbsp;&nbsp;... restricted to the steps spent alongside | after | 1938 | 0.0 % | 0.0613 | 0.1209 | 0.2010 | 0.2616 |
| &nbsp;&nbsp;... restricted to the steps spent alongside | after, 3 cars | 1469 | 0.2 % | 0.0597 | 0.1186 | 0.1995 | 0.2808 |
| the unit-less closeness the reward multiplies | before | 59428 | 93.5 % | 0.0000 | 0.0000 | 1.1192 | 2.1457 |
| the unit-less closeness the reward multiplies | after | 48534 | 93.2 % | 0.0000 | 0.0000 | 1.0075 | 2.2333 |
| the unit-less closeness the reward multiplies | after, 3 cars | 37194 | 93.0 % | 0.0000 | 0.0000 | 0.9586 | 2.7867 |
| body-to-body gap to the nearest car [m] | before | 59428 | 0.0 % | 4.8754 | 20.8134 | 35.2292 | 47.3861 |
| body-to-body gap to the nearest car [m] | after | 48534 | 0.0 % | 5.7300 | 16.2501 | 34.2342 | 46.8042 |
| body-to-body gap to the nearest car [m] | after, 3 cars | 37194 | 0.0 % | 3.8281 | 13.5541 | 28.5366 | 42.5619 |
| lateral body-to-body gap while alongside [m] | before | 1897 | 3.0 % | 0.2643 | 0.5046 | 0.8083 | 0.8992 |
| lateral body-to-body gap while alongside [m] | after | 1938 | 3.8 % | 0.2508 | 0.4760 | 0.7787 | 0.8981 |
| lateral body-to-body gap while alongside [m] | after, 3 cars | 1469 | 1.6 % | 0.3195 | 0.5858 | 0.8439 | 0.8994 |

Coefficients: `--car-proximity-penalty 0.8`, `--car-safe-gap 0.9` m, bodies touching at 0.5 m nose-to-tail and 0.3 m side by side, closing-speed reference 2 m/s.

**The term is zero on 93 % of learner steps, in every arm.** Adding opponents that behave like
opponents did not change that — 93.4 % before, 93.1 % after, 93.0 % with three cars — because what
decides it is not how the opponent behaves but how much of the time any car is within 0.9 m of the
learner's body at all, and the answer is 6–7 % of the time. The median body gap to the nearest car
is 4.6–5.7 m.

**On the steps that matter it is not zero at all.** Restricted to the steps the learner spends
alongside, the term is nonzero on essentially every one (0.0–0.4 % zero) at a median of 0.06–0.07
and a p99 of 0.20 per step. Over the ~1.2 s a pass takes, that is about 3 units of reward against a
collision worth −10 and a per-lap progress reward of order 30 — so where it fires, it is a real
term, not a rounding error.

**And inside that window it grades the gap properly.** The lateral body gap while alongside has a
median of 0.25–0.32 m and a p90 of 0.48–0.59 m, which against the 0.9 m `car_safe_gap` puts the term
at 0.72, 0.47 and 0.14 of full strength at the p50, p90 and p99 of the gaps the learner actually
passes at. It is not saturated and it is not in the noise: over the range a pass happens in, it is a
term that can tell a rub from a clean pass.

So the finding is narrow and specific: **the term is well-shaped where it applies and applies to
6–7 % of steps.** Two things follow, neither of which this branch acts on:

* **Opponent diversity does not fix the duty cycle.** The share of steps on which the term is
  nonzero went 6.6 % → 6.9 % → 7.0 % across the three arms, and the alongside share 3.2 % → 4.0 % →
  3.9 %. What decides it is how much of a race is spent within 0.9 m of another body, and a richer
  opponent population barely moves that. If the duty cycle is the problem, it has to be fixed in the
  reward, not in the distribution.
* **Whether a 6 % duty cycle is a problem is a question about the optimiser, not about the term.**
  Charged at 0.8 it pays about 3 units over a 1.2 s pass, against a −10 collision that fires once
  per several hundred passes — so per *pass* the dense term is already the larger signal. The case
  for widening the gap or reading only the lateral component is that it would grade the approach as
  well as the overlap; the case against is that the number above says the term is doing its job
  where it fires. That is a reward decision and it belongs to root; what was missing was the
  measurement, and it is above.

## 5. Smoke: does a recipe-A finetune still run, and does the attribution move?

Not a result about learning — 262 144 steps is 0.19 % of the frozen original's 138.8 M and 25 % of
the 1 M-step legs that produced A701, and neither arm is a checkpoint anybody should score. What a
smoke is for is that the recipe still runs under the new distribution, at what speed, and that the
failure *composition* is not obviously broken.

Both arms are recipe A verbatim (`docs/research/recipe-restore-2026-09-12.md`) warm-started from the
frozen original, `--envs 64 --total 262144`, seed 801, on
`real:blackbox2022_1,gen:control:1400,real:korea_2026_competition+rlobs211`, with
`--sim-backend graphs` (a 1.28 s graph capture instead of minutes of `compile`, which is what keeps
each arm inside the contract's 20 minutes; the same choice in both arms).

| arm | updates | steps | coll/km (median, 2nd half) | progress m/episode | lap s | steps/s | final kl_ref |
|---|---|---|---|---|---|---|---|
| A — today's opponent | 128/128 | 139.0M | 12.50 | 80 | 11.10 | 507 | 0.104 |
| B — population + reactive + random grid | 128/128 | 139.0M | 9.75 | 80 | 10.20 | 496 | 0.103 |

The two `coll/km` and `lap s` columns are **not a comparison**: each arm is measured inside its own
distribution, and arm B's distribution has faster opponents, blocking opponents and a grid that
sometimes starts the learner abreast. A lower number there means the arm's own rollout was cleaner,
not that its policy is better, and the only fair comparison of the two checkpoints is the
attribution below (same opponent, same maps, same seed for both). What the columns do say is that
both arms trained: 128 of 128 updates, the KL against the reference checkpoint sitting at ~0.10
with the penalty coefficient at its recipe value of 0.05, no divergence, and ~500 steps/s.

### Provenance of every checkpoint named here

| role | path | sha256 (first 16) |
|---|---|---|
| frozen original — the init of both arms and of every census | `~/f1sim_runs/_baselines/frozen_original_48cc698f.pt` | `48cc698f8c51feb5` |
| A701 — the pool's checkpoint entry (best system on suite v1) | `~/f1sim_runs/cl_origrecipe_legacy_s701/ppo_final.pt` | `29e82233853868be` |
| smoke A final | `~/f1sim_runs/smoke_oppdiv_A_s801/ppo_final.pt` | `0b1574d304aec928` |
| smoke B final | `~/f1sim_runs/smoke_oppdiv_B_s801/ppo_final.pt` | `533b9a63b6fd6d31` |

Neither smoke checkpoint should be scored or put in a roster: 262 144 steps is 0.19 % of the frozen
original's 138.8 M.

Then `traffic_attribution.py` on each final checkpoint, plus the frozen original as the reference
row: 2-car races on the three held-out traffic maps (`real:map16x07`, `gen:control:9100`,
`real:korea_2025_iccas`), teacher opponent at 0.6–0.8× with `brake,stop,shift` at 3 per 10 s, 16
learner trials per map. **The event detector in that script was broken** and is fixed here — see
REPORT.md; the `within 1 s of an event` column is therefore a measurement for the first time, and
every other column stays comparable with the published §5 table.

| checkpoint | terminations /48 | car contacts | alongside | ahead (rear-end) | behind | within 1 s of an opponent event | (base rate) | plan through the opponent | closing at contact | wall collisions | of which during a pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen original (published §5) | 42 | 10 | 9 | 1 | 0 | 0 (0 %) | not recorded | 1 | +1.88 m/s | 32 | 15 |
| frozen original (re-run, detector fixed) | 42 | 10 | 9 | 1 | 0 | 4 (40 %) | 51 % | 1 | +1.88 m/s | 32 | 15 |
| smoke A — today's opponent | 43 | 9 | 7 | 1 | 1 | 3 (33 %) | 50 % | 0 | +1.31 m/s | 34 | 12 |
| smoke B — population + reactive + grid | 39 | 15 | 10 | 5 | 0 | 5 (33 %) | 53 % | 2 | +1.56 m/s | 24 | 8 |

### Reading, with the sample size in front

48 learner trials per checkpoint, 9–15 contacts each. Differences of two or three contacts are
noise; what is worth reading is a direction that all three of the census, the terminations and this
table agree on, and one corrected number.

**The corrected number.** The published §5 row says 0 contacts within 1 s of an opponent event. With
the detector fixed the same checkpoint on the same seed gives **4 of 10**, and the base rate — the
share of the learner's own active steps on which its opponent was inside that window anyway — is
**51 %**. So contacts are *under*-represented around events (40 % against 51 %), not absent from
them. The conclusion §5 drew, "the scripted behaviours address a failure mode the data does not
contain", now rests on a measurement instead of on a detector that never fired. Every other column
of the re-run reproduces the published row exactly (42 / 10 / 9 / 1 / 1 / 32 / 15), which is what
makes it the same measurement.

**The direction all three agree on.** Arm B's checkpoint ends fewer trials (39 against 43 and 42),
hits fewer walls (24 against 34 and 32) and fewer of them during a pass (8 against 12 and 15), and
makes more car contacts (15 against 9 and 10). The census predicted exactly that trade from the
other side: in arm B's distribution the learner's wall count rose and its contact count fell,
because opponents that move take some of the squeeze away and put the learner in contact geometries
instead. Here it is the mirror image — a policy trained for 262 144 steps in that distribution
takes the wall-during-a-pass failure down by a third and pays for it in contacts. On 48 trials that
is a hint about the distribution, not a result about the policy, and it is the hint the branch was
built to produce.

**One thing not to read into it.** Arm B's 5 rear-end contacts against arm A's 1 is 4 events. It is
the direction `oblivious` and `defend` would produce if they generalised, and it is also what four
coin flips look like.

## 6. Limits

* **Neither smoke checkpoint is a system.** 262 144 steps is 0.19 % of the frozen original's
  138.8 M and a quarter of the 1 M-step legs behind A701. Nothing here is a benchmark number and
  neither arm belongs in a roster.
* **The census measures a distribution, not learning.** It says the learner is in these situations
  for this many seconds. Whether training on them moves the frozen v2.1 traffic family is the next
  experiment, and this branch does not run it.
* **`--opp-speed` above 1.0 has a ceiling nobody has found yet.** 1.2× costs 0.72 opponent walls per
  km and 1.3× costs none on `gen:competition:0`, but the teacher's profile plans at the grip limit
  and the number is per track. The census reports opponent wall contacts for exactly this reason;
  raise the scale and watch that row.
* **A pool entry must share the learner's observation space.** A checkpoint trained at a different
  `--scan-stack` / `--hist-len` / `--action-mode` is refused rather than adapted, which is the right
  answer but does mean the population can only hold checkpoints from the same family.
* **A pool of K checkpoints costs K full-width policy forwards per step.** Measured: arm B with one
  checkpoint runs at ~490 steps/s against arm A's ~505 at 64 envs, so one entry is ~3 %. Four
  entries would be four forwards; the pool is not free and does not pretend to be.
