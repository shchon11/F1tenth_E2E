# A faster teacher, and what the plan's two speed numbers cannot say (2026-09-19)

Branch `feat/mintime-teacher`. Everything here is opt-in: the default raceline, its cache keys, the
default teacher and the `linear` plan action space are what they were.

User: "DAgger 학습은 어때? teacher 상태는 괜찮아?" → "두개 맵은 빼고 … 마찰 한계까지 쓰는 최소시간
라인으로" → "속도 두개 띡 밷는 지금 구조로는 안될 것 같아서".

Evaluation set throughout: the held-out split **minus `rt:Monza` and `real:map16x07`** (the user's
call, after the teacher's own failures concentrated there) — nine scenarios, passed as `--tracks`;
the split definition and the frozen suites are untouched. `evaluate --teacher --action-mode plan
--protocol trials --envs 128 --budget-laps 3 --speed-cap 8`, default randomisation, seed 123, so
1152 first attempts per row. `--budget-laps 3` because at the default 2 the budget is two laps *at
the cap* and a teacher averaging 4 m/s barely fits one: 905 of 1536 trials read as timeouts on the
first pass, which is an artefact of the budget and not a failure.

## 1. The line: descend the lap time itself

`raceline.min_time_raceline` starts from the minimum-curvature line and descends
`sum(ds / speed_profile)` — the same point-mass, friction-ellipse profile the teacher drives, at
the same limits. Offsets are a periodic cubic B-spline along the current normals, boxed by the lane
under each coefficient's support; outer iterations re-parametrise and re-measure the lane as the
minimum-curvature solve does. Three things it took to make it work, each measured:

* **Central differences.** A 1 cm bump on a 0.5 m knot moves the local curvature by ~0.08 1/m, so a
  forward difference measures the bump's second-order cost — positive in every coordinate, 4 s/m —
  and L-BFGS-B finds no descent at all.
* **The line's own resolution.** Optimised on a 434-point copy of an 800-point line the descent
  gained 1.6 % there and came back **19 % slower** on the 800 points the teacher drives: linear
  resampling carries curvature noise, and the descent learns to cancel it.
* **Coarse to fine (2.0 → 1.0 → 0.5 m knots).** On the fine basis alone the same budget buys 1.5 %;
  with the schedule 13.6 % (`real:korea_2025_iccas`, 46.2 m → 39.3 m). Moving a corner takes a
  basis function as long as the corner.

It never returns a slower line than it was given (same clearance and turn-radius repair, kept only
if still quicker at full resolution). Point-mass lap, summed over the nine maps:

| line | limits a_lat / a_acc / a_brake | Σ lap [s] | vs today |
| --- | --- | ---: | ---: |
| min curvature (today) | 6 / 6 / 3 | 119.6 | |
| min curvature | 9 / 7 / 5 | 97.9 | −18 % |
| **min time** | 6 / 6 / 3 | 109.3 | −9 % |
| min time | 7 / 6.5 / 4 | 101.2 | −15 % |
| min time | 9 / 7 / 5 | 89.5 | −25 % |

Minimum clearance and peak curvature are unchanged (0.40–0.43 m, ≤ 1.08 1/m). ~50 s per track on
one core; cached.

## 2. Driven: the friction limit is not a usable teacher

| teacher | completed | coll/km | Σ median one-lap time [s] | large-slip time |
| --- | ---: | ---: | ---: | ---: |
| min curvature 6 / 6 / 3 (today) | 99.1 % | 0.14 | 140.8 | 0.00 % |
| **min time 6 / 6 / 3** | **99.4 %** | **0.11** | 129.1 (−8.3 %) | 0.01 % |
| **min time 7 / 6.5 / 4** | 98.6 % | 0.25 | **122.4 (−13.0 %)** | 0.25 % |
| min time 7.5 / 7 / 4.5 | 96.8 % | 0.58 | 119.8 | 0.71 % |
| min time 8 / 7 / 3 | 95.1 % | 0.88 | 119.6 | 0.07 % |
| min time 8 / 7 / 4.5 | 93.8 % | 1.12 | 118.0 | 0.90 % |
| min time 9 / 7 / 4 | 87.8 % | 2.29 | 116.0 | 0.81 % |
| min time 9 / 7 / 5 | 83.9 % | 3.09 | 115.6 | 1.63 % |
| min curvature 9 / 7 / 5 | 75.1 % | 4.71 | 124.4 | 0.58 % |
| min time 9.5 / 7 / 5 | 77.5 % | 4.49 | 114.4 | 1.82 % |

(One-lap time is from a random start at spawn speed, not a flying lap; it ranks rows, it is not a
lap record.)

* **The line is free.** At today's limits the min-time line is 8 % quicker and no less safe.
* **`7 / 6.5 / 4` is the knee.** Past it each further 2–3 % of pace multiplies collisions by 2–4.
  A min-time line at 7 is *quicker* than a min-curvature line at 9 (122.4 vs 124.4 s) at a
  twentieth of the collision rate: the safe pace is in the line, not in the limits.
* **Lateral, not braking.** 9/7/4 → 9/7/5 moves collisions 2.29 → 3.09; 8/7/3 → 8/7/4.5 moves them
  0.88 → 1.12; the step from 8 to 9 lateral at matched braking doubles to triples them.
* **No single randomised parameter is to blame** (9/7/5, the three worst maps, 768 first episodes,
  41.8 % crashed, median crash speed 3.7 m/s — in corners, not at top speed). Crash rate by tercile:
  `mu` 51 / 43 / 32 %, `cmd_delay` 34 / 38 / 53 %, `servo_tau` 36 / 38 / 52 %, `motor_tau`
  51 / 45 / 30 %, `a_brake` 44 / 40 / 41 %. Even the kindest tercile of each crashes ~30 %.
* **And the premise was a different floor.** The 9.2–11.5 m/s² in `teacher.py`'s comment is the
  pre-competition bags; the *target* floor sustains **8.51** over 200 ms
  (`real_data_calibration.md` §2.2; the simulator 8.89). A profile at 9–9.5 is above what the real
  car holds there either. `grip_control.RHO_MAX = 0.85` × 9.47 ≈ 8.0 says the same thing from the
  controller side.

The teacher for the experiments below is **min time, 7 / 6.5 / 4**.

## 3. What two speed numbers cannot say

The plan's speed is `v0` (0.15 s ahead) and `v1` (plan end), linear in arc length between. A
straight line has its minimum at an end, so an apex *inside* the plan cannot be expressed. Measured
against the teacher's own profile, over the 0.6 s of plan the tracker consumes, from every fourth
point of every line (1800 windows; positive = the plan asks for more speed than the profile allows):

| speed parameterisation | numbers | over-request p50 / p90 / p99 [m/s] | windows > 0.5 m/s over |
| --- | ---: | --- | ---: |
| linear `v0 → v1` — today's teacher | 2 | 0.49 / 1.08 / 1.84 | 49 % |
| linear — min time 7 / 6.5 / 4 | 2 | 0.63 / 1.62 / 2.25 | 60 % |
| speed at each curvature knot (`knots`) — 7 / 6.5 / 4 | 6 | 0.04 / 0.15 / 0.30 | 0.5 % |
| `a_hat`, `v_end` → curvature envelope (`envelope`) — 7 / 6.5 / 4 | 2 | 0.08 / 0.27 / 1.37 | 2.4 % |

An apex slower than both label points sits inside the consumed part in 39 % of windows for today's
teacher and 52 % for the fast one: the faster the teacher, the worse `linear` fits it. The plan is
re-issued at 40 Hz and `v0` is always right, so closed loop hides part of this — the teacher driven
through `linear` and through `knots` scores the same (below). What it cannot hide is that the plan is
not a truthful profile, which is the gap `fixed_low` fills from outside (110 → 136 / 144 solo,
16 → 43 / 48 at low μ, `controller-arms-2026-09-12.md`).

## 4. Speed modes (`mpc.SPEED_MODES`, `--speed-mode`)

* `envelope` — same 8 action dimensions; the last two become `a_hat`, the lateral acceleration the
  policy believes it can use, and `v_end`. `v(s) = sqrt(a_hat / |kappa(s)|)` on the plan's own
  curvature, braked backwards from `v_end` on the friction ellipse (`a_hat` is the whole budget),
  ramped forwards from the measured speed. No friction estimator and no μ formula: the one piece of
  physics is `v² κ ≤ a`, and how slippery it is is a learned output. The teacher's label for it is
  its own per-environment budget `a_lat · grip` — dense, and unobservable except through how the car
  answered, which is the point. `--grip-quantile τ < 0.5` puts a pinball loss on that dimension, so
  "cannot tell yet" means the slippery end and the belief is earned upwards (the `reactive` arm
  started optimistic and cut on slip, and lost avoidance 43 → 33 / 64: an obstacle is met head-on
  before anything slips).
* `knots` — a speed at each of the six curvature knots; 12 action dimensions.
* `mpc.decode` refuses a non-`linear` plan. The grip arms, the clearance arm and the viewers read
  plans through it and would otherwise produce a plausible wrong plan; they do not run on the new
  modes yet.

The teacher driven through each mode (min time 7 / 6.5 / 4, same protocol):

| mode | completed | coll/km | Σ one-lap [s] |
| --- | ---: | ---: | ---: |
| linear | 98.6 % | 0.25 | 122.4 |
| knots | 98.7 % | 0.23 | 122.4 |
| envelope | 99.2 % | 0.14 | 129.1 |

`envelope` is safer and 5 % slower. Rolling A/B on three maps (192 envs × 30 s): linear 0.27 coll/km
at 9.00 s; envelope 0.16 at 9.42 s; envelope without the forward pass (`PlanSpec.envelope_forward
= False`) 0.39 at 9.10 s. So ~3.6 of the 4.7 % is the ellipse-limited drive out of corners — pace
that `linear` takes by letting the tracker pull 6 m/s² while still turning.

## 5. Students: DAgger only, one seed each

Same teacher (min time 7 / 6.5 / 4, `--teacher-grip true`), same 44 training scenarios (the six
real maps both ways, 12 `gen:control`, 12 `gen:competition`, 4 `gen:circuit`, 4 `gen:hallway`; no
`rt:` maps, no obstacle variants), 512 envs × 320 steps × 8 iterations, `--keep-iters 3` (host RAM:
two concurrent runs at 5.5–6 GB each on a 15 GB machine), `--batch 2160`, `--hist-len 20`. Recurrent
arms: GRU 128, `--seq-len 120 --seq-burn 12` (3 s runs from a zero state). `envelope` arms:
`--grip-quantile 0.3` unless stated, forward pass on. ~27–35 min each. Scored on the nine held-out
scenarios, 128 first attempts each. **One seed per arm; no PPO.** `real:blackbox2022_3` is the map
the user ruled unfair on 2026-09-13 (dead-end side branches; every system scores near zero); it is
reported and also left out.

| student | all nine | without `blackbox2022_3` | coll/km | Σ one-lap [s] | `map12x16` both ways |
| --- | ---: | ---: | ---: | ---: | ---: |
| teacher | 98.6 % | 98.5 % | 0.32 | 80.2 | 96.5 % |
| linear, feedforward (today's structure) | 74.2 % | 90.6 % | 2.13 | 83.1 | 80.1 % |
| linear, GRU + sequences | 68.8 % | 81.2 % | 4.44 | 83.8 | 55.9 % |
| **linear, feedforward, teacher at 0.90× (pace-matched control)** | 83.1 % | **95.2 %** | **1.08** | 90.0 | 86.7 % |
| envelope τ 0.5, feedforward | 76.3 % | 86.8 % | 2.97 | 88.4 | 64.8 % |
| **envelope τ 0.3, feedforward** | **86.2 %** | **94.5 %** | 1.21 | 92.6 | 87.9 % |
| **envelope τ 0.3, GRU + sequences** | 82.0 % | **94.6 %** | **1.16** | 92.5 | **90.2 %** |
| knots, GRU + sequences | 54.1 % | 68.0 % | 8.35 | 79.6 | 46.5 % |

(coll/km and Σ one-lap are without `blackbox2022_3`.)

* **Pace margin is what buys safety, and the existing structure buys it as cheaply.** The cautious
  envelope students halve the feedforward linear student's collisions per km (1.2 vs 2.1) for 11 %
  of lap time — and a linear student whose teacher is simply slowed to 0.90× does the same for 8 %:
  95.2 % completed, 1.08 coll/km, 90.0 s, against 94.5 % / 1.21 / 92.6 s. On the narrow `map12x16`
  the three are 86.7 / 87.9 / 90.2 %, inside what one seed can resolve. **Pace-matched, the envelope
  speed head shows no advantage at the DAgger stage.**
* **The envelope's gain was the quantile, not the parameterisation.** With a plain Huber loss
  (τ 0.5) the envelope student is no better than linear — 86.8 % against 90.6 %, and slower. What
  τ 0.3 adds is a scalar that slows the car where it is lateral-limited and nowhere else; the
  control says that placing the caution there is worth no more than spreading it over the lap.
* **§3's representational error is real and does not show up closed loop.** The plan is re-issued
  at 40 Hz and its first speed is always right, which is why the teacher scores the same through
  `linear` and `knots` (§4), and evidently why the student does not pay for it either.
* **Memory changes nothing for the envelope student** (94.5 % feedforward, 94.6 % recurrent) —
  consistent with the next section: there is no grip inference for a memory to hold.
* **Six free speed numbers are worse than two.** `knots` fits the teacher's profile best on paper
  (§3) and drives worst: nothing ties its speeds to the corner it has planned, so every one of them
  is an independent regression error.
* **Sequence training as built here hurts the linear student** (81 % vs 91 %). Two things differ and
  both count against it: 20 runs per gradient step instead of 2160 independent samples, and a
  recurrence trained over 3 s from a zero state but driven for a whole episode — a hidden state the
  training never saw.

### It is safe because it is careful, not because it knows the floor

Randomisation off, friction pinned, three held-out maps, 192 envs × 30 s:

| | mean speed at μ 0.734 / 0.944 / 1.154 | ratio high / low | coll/km |
| --- | --- | ---: | --- |
| teacher (privileged grip) | 4.25 / 4.72 / 4.96 | **1.165** | 0.08 / 0.07 / 0.07 |
| envelope, GRU | 3.86 / 3.96 / 4.00 | 1.036 | 0.22 / 0.31 / 0.22 |
| linear, GRU | 4.38 / 4.60 / 4.72 | 1.078 | 3.81 / 3.51 / 3.27 |
| knots, GRU | 4.48 / 4.76 / 4.84 | 1.079 | 8.83 / 3.50 / 1.94 |

The envelope student's emitted `a_hat`, by seconds since the episode began, is 4.1 in the first
second and **4.8 from then on — on every floor** (the teacher's label is 4.2 / 5.4 / 6.0). It has
learned one cautious number, which the pinball loss at τ = 0.3 makes the best answer for a student
that cannot tell the floors apart. That is the designed failure direction — slow on a grippy floor
rather than fast on a slippery one, where `knots` crashes 8.8 times per km — but it is not grip
inference, and the 11 % of pace is what not knowing costs. Implicit grip learning did not emerge from
27 minutes of DAgger on 3 s runs; this agrees with the grip head's R² 0.07 on the old trunk
(`model.py`) rather than overturning it.

## 6. Open

* **What is left of the case for `envelope`.** Not safety or pace at this stage (§5). What it still
  offers is a channel: one interpretable number that a grip belief would move, that can be logged on
  the car and clamped per venue. That is worth something only if grip inference can be made to
  work, so that experiment comes first, and it can be run on either speed head.
* **`--teacher-speed` is the cheapest lever found today**: 0.90× takes the current structure from
  90.6 % to 95.2 % for 8 % of lap time. A sweep (0.85–1.0) would place the students on the same
  pace-against-collisions curve §2 draws for the teacher.
* **Grip inference has not emerged.** Candidates, cheapest first: store the collection-time hidden
  state and start training runs from it (the mismatch above); longer runs; the auxiliary grip loss
  DAgger does not yet apply; the frozen estimator's filtered lower quantile as an *input* through
  the existing zero-initialised conditioning path (a number for "how slippery it has felt", with
  what to do about it still learned). The probe in §5 — mean speed at pinned μ, target ratio 1.165 —
  is the test for all of them.
* **PPO on the envelope student.** Every finetune under a hidden speed clamp regressed
  (`recipe-restore-2026-09-12.md`). Here the envelope is the action's meaning rather than a clamp
  behind it; whether that is enough of a difference is an experiment, not an argument.
* **Runtime layers.** The grip arms, `+clearance` and the viewers read a plan through `mpc.decode`,
  which refuses the new modes. `+clearance` is the one that matters (it edits the plan's speeds).
* **One seed per arm.** The 94.5 / 94.6 agreement between the two envelope arms is the only
  replication here.

Artefacts: `~/f1sim_runs/sm_{lin_ff,lin_ff_s90,lin_gru,env_ff,env_ff_q50,env_gru,knots_gru}/`; per-cell JSON
and the scripts that produced every table are in
`~/f1sim_runs/_eval/mintime-teacher-2026-09-19/` (not in the repository).

## 7. Grip inference (2026-09-20): it is not there to be inferred, it is worth having, and a dial delivers it

User: "진짜 미해결 문제는 노면 추론 … 레퍼런스를 함 찾아봐바" → "함 잘 해봐바". The literature names the
symptom of §5 — a teacher acting on information the student cannot see is averaged away by imitation
(the *imitation gap*: Weihs et al. 2021, Shenfeld et al. 2023) — and the standard remedy is to regress
the privileged quantity from history (UP-OSI, Yu et al. 2017; RMA, Kumar et al. 2021). Before building
that, two questions, both measured.

### Can friction be read from what the car senses? Not at a pace it survives.

A driver that does not know μ (the teacher on the nominal profile), full randomisation, 1024 envs,
~3100 six-second episodes so there are thousands of distinct μ draws, scored on held-out *envs*:

| probe | features | R² of μ after 3 s |
| --- | --- | ---: |
| GRU, read out every step, pace ×0.9 (2.8 % of episodes crash) | speed, gyro z, accel x/y, issued steer and speed | **+0.012** |
| same, pace ×1.0 (8.2 % crash) | same | +0.018 |
| same probe, same data, target `speed_gain` instead | same | +0.480 |
| yaw-gain droop (high-load gain ÷ low-load gain within one car, so steering calibration cancels), 10 s episodes | gyro z, speed, issued steer | +0.004 / +0.014 |

The probe works — it reads `speed_gain` at 0.48 with train ≈ test, so this is absence of signal, not
overfitting (a first attempt with 384 envs *did* overfit, to −0.03…−0.6, and is discarded). It predicts
μ ≈ 0.945 on floors that are truly 0.79 and truly 1.10 alike.

Why, with randomisation off and friction pinned (same driver, corners only, |v·r| > 3 m/s²):

| pace | μ | sideslip p50 / p90 | yaw gain r / r_kinematic | IMU a_y − v·r, rms |
| --- | --- | --- | ---: | ---: |
| ×0.9 | 0.734 / 0.944 / 1.154 | 1.5° / 4.4°, 0.9° / 2.2°, 0.7° / 2.8° | 0.81 / 0.88 / 0.91 | ~3.0 m/s² |
| ×1.0 | 0.734 | 2.4° / **14.6°** (64 of 96 cars crash) | 0.76 | 3.5 m/s² |

At a safe pace friction moves the yaw gain by 13 % across the *whole* range — the size of the
randomised steering calibration it is confounded with — and the accelerometer carries 3 m/s² rms of
vibration (fitted to the car's own IMU) against a 4.5–6 m/s² signal. The car finds out when it slides,
and at that pace most of them crash. "저속에서는 잘 모르다가 … 털리는 느낌": measured. It is also why
`estimated` bought no completion over `fixed_low` (`2026-09-11-controller-arm-evaluation.md`).

### Is it worth knowing? Yes.

`--cond true_mu` (the existing zero-initialised conditioning path; a lab oracle), linear, feedforward,
otherwise the §5 recipe. Randomisation off, friction pinned, three held-out maps:

| student | mean speed at μ 0.734 / 0.944 / 1.154 | ratio | coll/km | lap [s] |
| --- | --- | ---: | --- | --- |
| teacher | 4.25 / 4.72 / 4.96 | 1.165 | 0.08 / 0.07 / 0.07 | 9.88 / 8.89 / 8.42 |
| linear, not told | 4.25 / 4.45 / 4.55 | 1.070 | **0.82** / 0.19 / 0.19 | 9.96 / 9.30 / **8.95** |
| linear, told μ | 4.11 / 4.55 / 4.94 | **1.202** | **0.42** / 0.31 / 0.49 | 10.16 / 9.10 / **8.35** |

Told the floor, the student uses all of it: half the collisions on the slippery floor, 6.7 % quicker on
the grippy one. Averaged over the randomised held-out protocol the two ends cancel (90.8 % vs 90.6 %
completed, 82.6 vs 83.1 s), which is why §5's aggregate never showed friction as a problem.

### So give it, do not infer it: the conditioning input as a grip dial

The same checkpoint, the floor fixed, the number it is *told* swept:

| floor μ | dial 0.65 | 0.734 | 0.84 | 0.944 | 1.05 | 1.154 | 1.25 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.734 — lap [s] / coll per km | 10.47 / 0.44 | **10.11 / 0.38** | 9.83 / 0.86 | 9.75 / 1.23 | 9.84 / 3.92 | 10.03 / 8.48 | 10.03 / 13.3 |
| 0.944 | 10.21 / 0.43 | 9.78 / 0.38 | 9.26 / 0.08 | **9.01 / 0.35** | 8.81 / 0.74 | 8.73 / 1.27 | 8.68 / 2.39 |
| 1.154 | 10.12 / 0.60 | 9.47 / 0.38 | 9.07 / 0.24 | 8.65 / 0.54 | 8.45 / 0.59 | **8.24 / 0.50** | 8.23 / 0.66 |

Pace follows the dial monotonically on every floor (19 % of lap time on the grippy one). Set low it is
slow and safe; set high on a slippery floor it is fast for two notches and then only dangerous — past
the true μ the lap stops improving while collisions go 1 → 4 → 8 → 13 per km. That asymmetry is the
whole operating rule: **start low, raise it while laps are clean, back off at the first slide** —
Wischnewski et al. 2019, and what a driver does. A competition is one floor: the dial is set in
practice, and nothing has to be inferred at 40 Hz from an IMU that cannot show it.

### The controller already has a dial. It is a different one, and the two do not stack.

User: "그립다이얼?? 제어기랑은 상관없나". `grip_control` puts a friction number into the *tracker*
(`fixed_low`: 0.734; `oracle`: the truth): a curvature speed cap and friction-derived accel bounds. Same
protocol as above (randomisation off, friction pinned, three held-out maps, ~30 km per cell, one seed):

| floor μ | not told, `legacy` | not told, `fixed_low` | not told, `oracle` controller | **told μ, `legacy`** | told μ, `oracle` controller |
| --- | --- | --- | --- | --- | --- |
| 0.734 — lap [s] / coll per km | 9.96 / 0.82 | 10.36 / 0.66 | 10.36 / 0.66 | **10.16 / 0.42** | 10.52 / 0.89 |
| 0.944 | 9.30 / 0.19 | 9.82 / 0.38 | 9.36 / 0.24 | **9.10 / 0.31** | 9.25 / 0.66 |
| 1.154 | 8.95 / 0.19 | 9.50 / 0.12 | 8.95 / 0.11 | **8.35 / 0.49** | 8.42 / 0.68 |

* **The controller's dial is a cap.** Set exactly right it makes the uninformed student a little safer
  on the slippery floor for 4 % of lap time, and on the grippy floor it buys no pace at all (8.95 =
  8.95): it cannot ask for speed the policy never asked for. `fixed_low` costs 6 % there.
* **The policy's dial moves the plan, both ways**: safer than the controller's on the slippery floor
  (0.42 vs 0.66, and quicker), and 6.7 % quicker on the grippy one, which the controller cannot give.
* **Both at once is worse than the policy's alone on all three floors** (0.89 / 0.66 / 0.68 against
  0.42 / 0.31 / 0.49, and slower). The student was trained on the legacy tracker; the grip arm changes
  the accel bounds its plans are executed with, and a policy that already slows for the floor has
  nothing left for the cap to catch. Counts are small (a dozen to two dozen events per cell), but the
  direction is the same three times out of three. So: the number goes into the policy; the controller's
  dial stays what it has been — the retraining-free fix for checkpoints that were never told.

Next: (1) a deployable conditioning source — trained on μ minus a random margin, so the number means
"at least this much grip" and an under-set dial is in distribution — since `true_mu` is refused by
every deployment loader by design; (2) the supervisor: a slide detector on gyro / wheel-speed events
(which *are* observable: 14.6° of sideslip) and a lap-by-lap raise, scored against `fixed_low` and
the oracle; (3) PPO on the dial-conditioned student.

## 8. The dial made deployable, and PPO rebuilt around it (2026-09-20)

User: "제어기는 신경쓰지 말자 … 정책 다이얼에 신경쓰자" → "DAgger 학습 돌려놓고 PPO 싹다 재설계 들어가고
마무리하자". The tracker stays `legacy` throughout: it follows the plan and nothing else.

### `--cond dial`

`true_mu` is refused by every deployment loader, by design. `dial` is the same input with a different
contract: per episode the student is told `mu - margin` (margin 0 in 30 % of episodes, else
U(0, 0.30)), **and the teacher is asked to drive for that same number**
(`env.teacher_label(teacher, mu=...)`). Labelling with the true friction instead would teach "the floor
is somewhere above the dial, go faster than it says"; labelled this way the input is a command. One
seed, the §5 recipe otherwise:

| linear, feedforward student | held-out completed (no `blackbox2022_3`) | coll/km | Σ one-lap [s] |
| --- | ---: | ---: | ---: |
| not told | 90.6 % | 2.13 | 83.1 |
| told μ (`true_mu`) | 90.8 % | 2.08 | 82.6 |
| **`dial`, set exactly** | **92.6 %** | **1.61** | 82.8 |

and it obeys: on a μ 0.944 floor the lap goes 10.49 → 9.88 → 9.39 → **9.06** s as the dial goes
0.65 → 0.73 → 0.84 → 0.94 at 0.17–0.29 collisions/km, then 8.87 / 8.78 s at 0.79 / 1.86 over-set.

### Raising it from the car: the signal exists in the population and is noisy per car

Sensors only (gyro z, measured speed, issued steer). Step-level slide detectors are weak — AUROC
0.60–0.69 against true tyre slip > 10° — but one statistic has a clean dose-response in the dial:
the **lateral shortfall** in corners, `|v·r_kinematic| − |v·r|` (understeer: the car is asked to turn
and does not). Dial raised 0.05 every 36 s on the same car, 960 cars, full randomisation:

| dial − μ | −0.20 | −0.10 | 0 | +0.10 | +0.20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| shortfall [m/s²] | 0.27 | 0.30 | 0.36 | 0.45 | 0.57 |
| true tyre slip in corners | 4.5° | 5.2° | 5.9° | 6.5° | 7.2° |
| collisions / km | 0.82 | 0.50 | 0.72 | 1.13 | 1.57 |

Per car it is about one sigma per notch: the difference from a car's own low-dial baseline has an
interquartile range of 0.25 m/s² against a 0.2 m/s² shift from −0.1 to +0.1. (A *ratio* to the
baseline is useless — the statistic's level is track- and calibration-specific and is ≤ 0 for a
quarter of the cars.) An offline stop rule at +0.12 m/s² settles at a median of −0.05 with p10 / p90
−0.10 / +0.20. A first closed-loop supervisor settled 27 % of cars 0.26 *below* the floor, because it
read a crash as proof: this student crashes ~0.7 / km for reasons that have nothing to do with
friction. The revised rule — two consecutive notches over the line to settle, a crash costs a notch
and the climb continues — is the run `sup_v2` (result at the end of this note).

### PPO, rebuilt around the dial

The PPO core (stored log-probabilities, truncation bootstrap, KL leash, recurrent replay) is
untouched; what it optimises and from where is new:

| | before | now |
| --- | --- | --- |
| initial policy | unconditional DAgger student, min-curvature 6 / 6 / 3 teacher | `dial` student of the min-time 7 / 6.5 / 4 teacher |
| friction | unobserved; one compromise speed for every floor (§7) | an input, `mu − margin`, redrawn per episode |
| what keeps the input meaningful under RL | — | `reward_grip_budget`: per second, per (m/s²)² of `|v·r|` beyond 0.95 · dial · μ_f · g. Return alone would teach the policy that the floor has at least the dial's grip and to use more |
| KL reference | the unconditional baseline (conditioning projection required to be zero) | the dial student itself, fed the same dial: the leash holds obedience |
| critic | privileged state | privileged state **+ the dial** — the return depends on how much grip was allowed |
| lap-time reference, teacher opponents | min-curvature line, 6 / 6 / 3 | `--raceline-objective min_time --teacher-a-*` |
| exploration at start | the checkpoint's `log_std` (a DAgger checkpoint arrives at std 0.50 and crashes through its first updates) | reset to the plan-space default when `--init` is a DAgger checkpoint |
| tracker | — | `legacy`, and the grip arms stay off: a cap stacked on a told policy was worse on all three floors (§7) |
| scoring | dial-less | `evaluate --dial-offset`: exactly right, and 0.15 under |

The first run is solo (`ppo_dial_s901`, 8.4 M steps from `dg_dial_s901`); opponents and procedural
obstacles are the next stage, from its checkpoint. Launcher:
`~/f1sim_runs/_eval/mintime-teacher-2026-09-19/scripts/run_dial_pipeline.sh`.

Three tests fail on this machine with and without this branch (`git stash` checked): the PPO loss
bit-identity oracle (`approx_kl` differs in the 9th digit from a value frozen on another machine),
`test_truncated_bptt_learns_a_task_the_feedforward_actor_cannot`, and
`test_multicar.py::test_race_lidar_and_contact`.

### `sup_v2`: the supervisor, closed loop

The `dial` student, 720 cars that keep their floor across crashes, full randomisation, six maps
(the three of §7 plus `gen:competition:0` and two reversed), the dial started at 0.55 for every car
and moved 0.05 per 24 s notch by the revised rule, sixteen notches. Steady-state notches:

| dial | mean speed [m/s] | collisions / km | dial − μ |
| --- | ---: | ---: | --- |
| left at `fixed_low`'s 0.734 | 3.84 | 1.2–1.3 | −0.21 median |
| **supervisor**, last three notches | **4.05 (+6 %)** | **1.2–1.4** | median **−0.06**, p10 −0.37, p90 +0.29 |
| set exactly right (not available on a car) | 4.19 (+9 %) | 2.7 | 0 |

By floor the final dial is 0.70 / 0.85 / 1.00 on floors that are 0.80 / 0.93 / 1.07: it tracks the
floor from below, with about 0.08 in hand. It recovers two thirds of the pace between the fixed dial
and the exact one **at the fixed dial's collision rate** — and the exact dial is not the target it
looks like: this student at its teacher's full pace for the floor crashes twice as often, which is §5's
pace-margin result again. What is not good is the spread per car (a third end more than 0.10 over,
38 % more than 0.15 under): one understeer statistic, one notch, is one sigma. Longer notches, a
second statistic, or lap time as a second vote are the obvious next things; so is a student that
crashes less for unrelated reasons, which is what PPO is for.

## 9. The full run: the student is the result; PPO has not yet earned its place

`dg_dial_s901` (the whole `train` split, 149 scenarios, `--scan-stack 6`, 12 iterations, min-time
7 / 6.5 / 4 teacher, `--cond dial`), then PPO from it, 8.4 M steps solo, twice: with
`--grip-budget-penalty 2.0` and with 0. Held-out, 896 first attempts (no `blackbox2022_3`), one seed each.

| | dial | completed | coll/km | Σ one-lap [s] | `map12x16` |
| --- | --- | ---: | ---: | ---: | ---: |
| **DAgger student** | exact | 93.4 % | 1.48 | 82.3 | 86.7 % |
| **DAgger student** | **−0.15** | **97.3 %** | **0.55** | 86.7 | **94.9 %** |
| PPO, grip budget 2.0 | exact | 92.9 % | 1.54 | 81.7 | 83.2 % |
| PPO, grip budget 2.0 | −0.15 | 96.9 % | 0.69 | 85.5 | 91.8 % |
| PPO, no budget | exact | 91.1 % | 1.96 | 80.8 | 76.6 % |
| PPO, no budget | −0.15 | 95.5 % | 1.00 | 84.4 | 86.7 % |

* **The best row is the DAgger student with its dial 0.15 under the floor**: 97.3 % completed and
  0.55 collisions/km on maps it has never seen, 95 % on the narrow one. The pipeline that produced
  it — min-time line, 7 / 6.5 / 4 limits, the dial with teacher labels built for the dial — is the
  day's result.
* **PPO moved along the pace-safety curve, not off it.** Both runs are 1–2 % quicker and no safer
  than the student they started from; without the budget it is the quickest and the least safe.
  With this reward (lap-time bonus 2, lap bonus 5, collision 10) 8.4 M solo steps buy pace. Making
  PPO buy safety instead is a reward-balance question that is still open, not something this run
  answered.
* **The trial did not replicate, and that is the honest reading of both.** On the 44-scenario trial
  (`ppo_dial_pilot*`, from the earlier student) PPO without the budget went 92.6 → 94.6 % and beat
  the budget run (92.7 %); at full scale the order is reversed and neither beats its student. The
  differences between PPO arms (±2 %) are the size of what one seed moves. Whether the grip budget
  helps is **undecided**; what is decided is that the dial survives RL either way — the KL leash to
  the obedient student is enough at these lengths (dial sweeps in `pilot/`).

Results: `~/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/` and `pilot/`. Checkpoints:
`~/f1sim_runs/{dg_dial_s901,ppo_dial_s901,ppo_dial_s901_nobudget}/`.

## 10. Reward audit, and "drive the same map until you know it" (2026-09-20)

User: "보상설계 한번 싹 점검해봐. 랩타임이 확 안빨라지는것도 좀 의심스러워 … 반복해서 달리다 보면 맵을 외우잖아."

### What the policy is paid for (final recipe's weights, deterministic, four maps, per second of driving)

| term | DAgger student | PPO from it |
| --- | ---: | ---: |
| progress | +4.52 | +4.53 |
| lap bonus (5 × average speed, per lap) | +1.55 | +1.56 |
| lap-time (sector) bonus | +0.33 | +0.32 |
| plan clearance, steer rate, proximity, sideslip | −0.18 | −0.13 |
| **collision (+ impact speed)** | **−0.035** | −0.036 |

Collisions are 0.7 % of what the policy earns. In its own currency — the −10, the impact term and the
future it forfeits, which γ = 0.99 at 40 Hz caps at 100 steps — a crash costs about what 4–5 s of
driving pays. A race charges a DNF or a reset for it. Every other large term pays for pace, three
times over. So the reward was never short of a reason to go faster; what held the lap time still was
the run, not the reward: 8.4 M steps (the frozen original had 139 M), a KL leash whose decay horizon
(40 M) is five times the run so it never let go, a student whose pace *is* the teacher's (7 / 6.5 / 4,
~80 % of the car), and 70 % of training episodes told by their dial to go slower.

### One map, 6.3 M steps (~45 min), from the generalist `dg_dial_s901`

`real:map12x16` as the stand-in for a competition track, randomisation on, dial set exactly,
256 cars × 60 s, true line-to-line laps:

| policy | median lap [s] | best 10 % | collisions / km | a crash every |
| --- | ---: | ---: | ---: | ---: |
| teacher (privileged; point-mass ideal of its line 7.38 s) | 8.57 | 7.97 | 0.34 | 89 laps |
| generalist student | 8.48 | 7.75 | 1.18 | 26 laps |
| generalist PPO | 8.43 | 7.76 | 1.32 | 23 laps |
| **specialised, the recipe as it was** (leash 0.05 held, γ 0.99, collision 10, lap-time 2) | **8.05** | 7.43 | **0.41** | **74 laps** |
| — same, dial 0.15 under | 8.45 | 7.75 | 0.24 | 125 laps |
| **specialised, revised** (leash released over 4 M, γ 0.997, collision 40, lap-time 6, steer 0.02, lr ×2) | **6.97** | 6.47 | 1.99 | 15 laps |
| — same, dial 0.30 under | 7.10 | 6.55 | 1.27 | 24 laps |

* **Knowing the map is worth both things at once**: with nothing else changed, 5 % quicker *and*
  three times fewer collisions than the generalist, past the privileged teacher's pace at its
  collision rate. "Make the map in practice, train on it for an hour" works.
* **The lap time was there to be had.** Letting go of the leash and paying for sectors takes another
  13 % — under the point-mass ideal of the teacher's own limits, so this is pace the teacher does not
  have — and was still falling at the end of the run (7.5 → 7.4 → 7.3 s in training).
* **It paid for it in collisions, and it spent the dial.** Five times the collision rate despite the
  collision term at 4×; and a policy trained with its dial always exact and no leash has almost
  stopped listening to it (0.30 under moves the lap 1.9 %), where the leashed one still obeys
  (0.15 under: +5 % lap, collisions 0.41 → 0.24). Five things changed at once in `revised`, so which
  of them bought the pace is not separated here.

Running next (`spec_map12_fastsafe_{budget,nobudget}`): the revised recipe with the dial margin kept
in training and the collision term at 60, with and without the grip budget — the one setting the
budget was designed for, a released leash.

### Fast *and* safe: the recipe that gets both

The revised recipe's problem was not its pace but what it spent for it (§10): five things changed at
once, among them a dial that was always exact, and the result stopped listening to the dial. Keeping
the dial margin in training and paying 60 for a collision — and this time with the grip budget,
which is what it was designed for, a released leash — gives both. Same map, same protocol:

| policy | median lap [s] | best 10 % | collisions / km | a crash every |
| --- | ---: | ---: | ---: | ---: |
| teacher (privileged) | 8.57 | 7.97 | 0.34 | 89 laps |
| generalist student | 8.48 | 7.75 | 1.18 | 26 laps |
| specialised, recipe as it was | 8.05 | 7.43 | 0.41 | 74 laps |
| specialised, revised (leash off, dial always exact) | **6.97** | 6.47 | 1.99 | 15 laps |
| **specialised, fast+safe** (collision 60, dial margin kept, grip budget) | **7.50** | 6.90 | 0.56 | 54 laps |
| — same, dial 0.15 under | 7.62 | 7.03 | **0.29** | **105 laps** |

Against the generalist it started from: **10 % quicker and four times fewer collisions**. Against the
privileged teacher: 11 % quicker at a lower crash rate. And the dial works again — 0.15 under costs
1.6 % of lap and halves the collisions, where `revised` barely responded to it. The pieces that
matter are separable now: the pace came from releasing the leash and paying for sectors; the safety
came from the collision term and from keeping the dial a live input; the budget is what let the
second survive the first.

### And the grip budget decides it

The same fast+safe recipe with the budget off, which is the arm the full-scale runs of §9 could not
separate. Released leash, collision 60, dial margin kept, everything else identical:

| | median lap [s] | collisions / km | a crash every |
| --- | ---: | ---: | ---: |
| **grip budget on** | 7.50 | **0.56** | **54 laps** |
| — dial 0.15 under | 7.62 | **0.29** | **105 laps** |
| grip budget off | 7.15 | 2.16 | 14 laps |
| — dial 0.15 under | 7.22 | 1.57 | 19 laps |
| — dial 0.30 under | 7.28 | 1.20 | 25 laps |

**Four times the collision rate for 4.7 % of lap time**, and a dial that barely answers: 0.30 under
buys 44 % fewer collisions where the budget arm gets 48 % from half that. So the budget does earn its
place — but only where it was designed to, with the leash released. Under the KL leash of §5 and §9 it
does nothing, because the leash is already holding the policy to a student that obeys the dial;
released, the leash is gone and the budget is the only thing left that prices the dial. That is why
the trial and the full run disagreed, and neither was wrong.

**Recipe of record** (`spec_map12_fastsafe_budget`, dial 0.15 under): per-venue specialisation from
the generalist dial student, `--lr 2e-4 --lr-end 5e-5 --kl-coef 0.05 --kl-decay 4e6 --gamma 0.997
--collision-penalty 60 --steer-penalty 0.02 --lap-time-bonus 6 --dial-margin 0.30 --dial-exact 0.30
--grip-budget-penalty 2.0`, 6.3 M steps on one map, ~45 min.
