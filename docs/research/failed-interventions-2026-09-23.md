# Four interventions that did not work, and what the measurements said afterwards — 2026-09-23

Branch `main`, base `f48ec29`. No repository code was changed for any of this; the four runs are
recipes under `~/f1sim_runs/_eval/mintime-teacher-2026-09-19/scripts/`, and the numbers come from
`f1sim.learn.evaluate` and from the diagnostic scripts in that directory's `diag/`.

Runs: `spec_korea_contact_s916` … `s919`, all continued from the same checkpoint,
`spec_korea_contact_s915/ppo_u768.pt` — the best policy measured so far (5.32 static contacts per
km pooled over two evaluation seeds and 52 km). Evaluation protocol: `scripts/eval_ckpts.sh`
(seed 77) and `scripts/eval_ckpts_seed78.sh` (seed 78), six scenarios each, pooled by
`eval_center/table_pooled.py`.

Companion to [`console-hygiene-2026-09-23.md`](console-hygiene-2026-09-23.md), which covers the
same day's console and test work.

## Summary

* **Four interventions, four failures, and the same shape in each one.** Every one began from a
  measurement that was correct. Every one then made an inference from that measurement to a change,
  and that inference was never itself tested before a GPU-hour was spent on it. Three of the four
  cost about six hours of 4070 time between them.
* **s916 — curvature knots rescaled to the grip left at the current speed** (`--plan-kappa-mode
  feasible`). Premise: at 5 m/s a 21-step sweep of one knot collapses into two outcomes, so the
  policy has almost no plan to choose from. True. Result: 5.32 → 7.26 contacts/km. The post-hoc
  signature says the wider action space was not used to slow down or to steer away.
* **s917 and s918 — exploration narrowed from 0.15 to 0.07.** Worse in both curvature modes (9.27
  and 10.51 /km). s918 also dropped the KL leash that s915 had, so it is **not a clean control** for
  exploration width; it tests two changes at once.
* **s919 — the price of a contact raised 2.5x** (flat 8 → 20, per-m/s 8 → 16). Result: 5.81 to 6.47
  /km across three checkpoints, none better than the baseline. The decisive number is not the rate:
  it is that the **median impact speed went up, 1.65 → 2.25 m/s**, under a per-m/s penalty that had
  just been doubled.
* **The error bars are 1.6x wider than Poisson.** Contacts cluster, so counting them as independent
  events understates the spread. Measured overdispersion φ = 2.69 across 70 evaluation cells. At
  52 km a rate near 5–6.5 /km carries ±1.0–1.1 /km at 95 %; at 26 km — one seed — it is ±1.5–1.6,
  which is wider than every effect reported here. **One seed cannot decide any of these.**
* **`collisions_per_km` is walls *and* obstacles.** So is `traffic["wall_collisions"]`, despite its
  name. Neither field separates the two, so nothing in this note should be read as "obstacle
  avoidance improved or got worse" — only "the car touched static geometry more or less often".

## 1. How the four were judged

### 1.1 Contacts cluster, so Poisson error bars are too narrow

The pooled table divides total contacts by total distance and reads the result as a rate. The
question is how much of the difference between two such rates is noise. Treating contacts as
independent events gives a Poisson interval, and that interval is measurably too tight.

Fitting each (checkpoint, scenario) pair its own rate and letting only the two evaluation seeds
vary — so genuine scenario structure is not counted as noise — the residual chi-square is 94.2 on
35 degrees of freedom:

```
phi = 94.2 / 35 = 2.69        error bars scale by sqrt(phi) = 1.64
```

Which turns into:

| | Poisson | with overdispersion |
|---|---:|---:|
| 5.32 /km over 52 km (two seeds) | ±0.63 | **±1.03** |
| 6.47 /km over 52 km (two seeds) | ±0.69 | **±1.13** |
| 5.32 /km over 26 km (one seed) | ±0.89 | **±1.45** |
| 6.47 /km over 26 km (one seed) | ±0.98 | **±1.60** |

The practical consequence is in the second column. s919u256 at 5.81 against the baseline's 5.32 is
a difference of 0.49 with a ±1.03 bar on each: on its own, that comparison decides nothing. What
carries the s919 verdict is not that one number, it is that all three of its checkpoints moved the
same way, and that the post-hoc measurement in §4 has no error bar problem at all.

It is also why the baseline's own scatter is the right yardstick. Across its five obstacle
scenarios s915u768 scores 5.86 on seed 77 and 4.80 on seed 78 — a swing of 1.06, larger than the
s919u256 effect. Earlier in the week s912's final scored 9.40 and 5.95 on the same six scenarios at
two seeds. Ranking checkpoints from a single seed is ranking the seed.

### 1.2 The baseline was re-scored in the same sweep

Commit `f48ec29` landed at 08:01, after the baseline's existing numbers had been written (04:53 and
06:03). Its default path (`--plan-kappa-mode absolute`) should be byte-identical — `gym_env.py:796`
does not even construct a `PlanSpec` when the mode is `absolute` and the speed mode is `linear` —
but "should be" is exactly what had cost an evaluation earlier that morning. So s915u768 was scored
again alongside s919, in the same sweep, by the same code.

It came back 275 contacts over 52 km: the same 5.32, to the digit. That confirms two things at
once — `f48ec29` changed nothing on the default path, and the evaluation protocol reproduces.

## 2. s916 — a curvature knot as a share of the grip

### 2.1 The premise, which was right

`diag/action_authority.py` sweeps one curvature knot across [-1, 1] in 21 steps from the same pose
and drives 0.5 s, and reports the lateral metres that result. Under `absolute`, the knots span
±1.6 1/m at any speed, while 8 m/s² of grip at 5 m/s is 0.32 1/m. The sweep collapses:

```
1 m/s  -0.17 -0.09 -0.01 +0.03 +0.08 +0.12    (at a = -1, -0.6, -0.2, .2, .6, 1)
5 m/s  -0.34 -0.33 -0.33 +0.37 +0.57 +0.53
7 m/s  -0.15 -0.16 -0.15 +0.47 +0.52 +0.49
```

At 7 m/s the action is left-or-right and nothing in between. `diag/saturation.py` agreed from the
other side: 64 % of steps had at least one knot at |a| > 0.95, and 53 % of what the first knot asked
for was over the grip limit. `feasible` rescales the knots to ±min(κ_max, a_lat/v²), and the same
sweep then gives 5 m/s a genuine ladder: −0.36 / −0.14 / +0.15 / +0.37 / +0.50.

### 2.2 The result

| | contacts/km (2 seeds, 53 km) | per seed | lap, obstacles | lap, empty | car contacts/min |
|---|---:|---:|---:|---:|---:|
| s915u768 | **5.32** | 5.86 / 4.80 | 8.42 s | 7.98 s | 11.3 |
| s916final | 7.26 | 6.83 / 7.67 | 8.57 s | 8.18 s | 15.1 |

Worse on both seeds, and slower — 8.18 s on an empty track against 7.98. 6.29 M steps did not
recover it; `kl_ref` reached 22, which is consistent with a policy having to relearn what its
actions mean, though that reading was never proved.

### 2.3 What it did with the room it was given

Measured after the fact by the diagnostics session, on approach to obstacles, binned by distance:

* **It does not slow down.** Through the 1–3 m band s916 holds 5.42–5.65 m/s where the baseline
  goes 5.46 → 4.61.
* **It aims at the thing more often, not less.** Inside the last 0.5–1 m, 25.2 % of its steps are
  pointed at the obstacle against the baseline's 6.5 %.
* **Its plan is inside the object at the end.** Plan clearance over the final 0.5 m is −0.04 m
  against the baseline's +0.08 m.

So the extra resolution was real and went somewhere else. The premise "it cannot steer finely
enough" was true; the inference "therefore finer steering will make it avoid things" did not
follow, because nothing in the measurement showed the policy *trying* to place itself just outside
an obstacle and missing by a knot's worth of resolution.

## 3. s917 and s918 — narrower exploration

The policy's exploration standard deviation had barely moved in 15 M steps (s906 0.153/0.148/0.169,
s916 0.116/0.128/0.170) and `--ent` is 0, so there is no entropy bonus holding it open. Against
that, the counterfactual probe (`diag/does_it_look.py`, scan with the obstacles erased) put the
action difference caused by an obstacle within 3 m at 0.105 — smaller than the 0.15 exploration
noise the action is drawn with. The inference: the signal is being drowned, so narrow the noise.

`--init-log-std -2.66` (σ ≈ 0.07) on both curvature modes:

| | contacts/km (2 seeds) | per seed | lap, obstacles | lap, empty | car contacts/min |
|---|---:|---:|---:|---:|---:|
| s915u768 | **5.32** | 5.86 / 4.80 | 8.42 s | 7.98 s | 11.3 |
| s917final (`feasible`, σ 0.07) | 9.27 | 9.57 / 8.97 | 8.68 s | 8.19 s | 13.1 |
| s918final (`absolute`, σ 0.07) | 10.51 | 10.07 / 10.94 | 8.31 s | 7.86 s | 18.9 |

Both far outside the ±1.1 bar, on both seeds, in both curvature modes.

**s918 is not a clean control.** It was meant to isolate exploration width by keeping s915's recipe
and changing only `--init-log-std`, and it did not: it also ran with `--kl-coef 0.0`, dropping the
imitation-KL leash that s915 had at 0.05. Two changes, one run. Whatever s918 shows is a property
of "narrow exploration *and* no leash", and the exploration half of it cannot be extracted. If the
question is worth re-asking, the run to do it with is s915's recipe with `--kl-coef 0.05` kept and
`--init-log-std` as the only edit.

Where s918's contacts are is worth recording, because it is not uniform. Pooled over both seeds:

| | solo + obstacles | with an opponent |
|---|---:|---:|
| s915u768 | 2.56 /km | 6.74 /km |
| s918final | 7.18 /km | 12.22 /km |

Both roughly tripled, and the absolute damage is concentrated in traffic — which is also where its
car-contact rate went, 11.3 → 18.9 a minute. Note that the diagnostics session reports a separate
solo measurement of s918 that points the other way; the two are not the same protocol (this table
is the 32-car evaluation `solo_obst` scenario) and that discrepancy is unresolved at the time of
writing, so nothing here should be read as settled about s918 driving alone.

## 4. s919 — the price of a contact

### 4.1 The premise

`diag/saturation.py` on s915u768 gives the reward breakdown per step: progress +0.131, lap +0.048,
lap-time +0.040, plan clearance −0.018, collision −0.005 and −0.008. A contact fires on 0.1 % of
steps and costs −8 flat plus −8 per m/s at a median impact speed of 1.65 m/s, so about 21 points,
against driving's ~0.20 a step, ~8 points a second. One contact ≈ 2.4 s of driving; on an 8.4 s lap,
a quarter of a lap.

The inference: the policy is taking a trade the reward offers it, so change the price. `s919` is
s915's recipe with `--collision-penalty 8 → 20` and `--collision-speed-penalty 8.0 → 16.0` and
nothing else — verified by diffing the `args.json` the run writes against s915's, where the only
other differences are the run name, seed, init checkpoint and step budget. 2.10 M steps (512
updates), 56 minutes.

A caveat that applies to any short continuation of this recipe: `lr` (5e-5 → 2e-5) and the KL leash
(0.05 → 0) both decay over `--total`, so over 2.10 M steps they fall about three times faster than
they did over s915's 6.29 M. That is not the variable under test and it applies equally to s918.

### 4.2 The result

| | contacts/km (2 seeds, ~52 km) | per seed | lap, obstacles | lap, empty | car contacts/min |
|---|---:|---:|---:|---:|---:|
| s915u768 | **5.32** | 5.86 / 4.80 | 8.42 s | 7.98 s | 11.3 |
| s919u256 | 5.81 | 5.78 / 5.83 | 8.32 s | 7.90 s | 10.7 |
| s919u384 | 6.47 | 7.58 / 5.37 | 8.21 s | 7.85 s | 15.9 |
| s919final | 6.47 | 5.64 / 7.28 | 8.54 s | 8.17 s | 11.2 |

Three checkpoints, none better than the baseline, and two of them (u256, u384) *faster* than it on
an empty track. Charging a touch 2.5x more did not buy fewer touches and did not even slow the car.

Inside the training environment the same thing: over the last half of the run s919 averaged 8.3
contacts/km against 7.5 for s915's matching 512-update stretch, at 8.48 s a lap against 8.57 —
quicker and touching slightly more, on a reward that had just made touching much more expensive.

### 4.3 The measurement that settles it

`diag/saturation.py` on `s919final`, 32 cars × 1200 steps, solo, scored with the *baseline's*
reward definition so the terms are comparable. Both sides of the comparison come from that same
harness, which sets the grip dial the way the evaluation does (`dial = mu + 0.30`, the same
construction as `evaluate.py:324` under `--dial-offset 0.30`) and which draws obstacles at the
*training* budget of 18 prop slots rather than the evaluation's 10. The absolute impact speeds are
therefore not directly comparable to the evaluation protocol; the 1.65 → 2.25 difference between
two policies measured the same way is.

```
  collision        -0.0052   active   0.1 %   when active  -8.000
  collision_speed  -0.0117   active   0.1 %   when active -17.978
```

`collision_speed` when active is −17.978 at a coefficient of 8, so the **median impact speed is
2.25 m/s**. The baseline's is −13.2 at the same coefficient: **1.65 m/s**.

The per-m/s coefficient had been doubled, from 8 to 16, and the policy hit things 36 % *faster*.
Not more gently, not less often — harder. Under its own training reward that makes a typical
contact 20 + 16 × 2.25 ≈ 56 points where the baseline's was 8 + 8 × 1.65 ≈ 21.

This is what breaks the premise. If contacts were a trade the policy chooses, raising the price
should move *something* in its favour, and the cheapest concession available — arriving slower, so
the speed-proportional half of the penalty is smaller — is exactly the one it did not make. A
policy that cannot buy a lower impact speed for 16 points per m/s is not choosing the impact.

The rest of the same measurement says why it cannot. Saturation is 57.9 % of steps (baseline 64 %),
the first knot asks a median 9.4 m/s² of lateral acceleration with **56 % of requests over the
8 m/s² grip limit** (baseline 53 %), and commanded curvature is 0.42 1/m against 0.24 executed
(baseline 0.44 against 0.24). More than half of what this policy orders, the tyres cannot deliver.
Raising the price of an outcome the actor cannot avoid lowers the value function and leaves the
policy where it was.

## 5. What the metric counts

Both fields that look like they separate obstacles from walls do not.

`collisions_per_km` (`evaluation_metrics.py`) counts the environment's collision flag, which is
walls and props together and excludes car-to-car contact — the latter is reported separately as
`traffic["car_contacts_per_learner_min"]`.

`traffic["wall_collisions"]` is named for walls but is defined in `benchmark/overtake.py:504` as

```python
touching = bool(hit[i]) and not bool(contact[i])
```

— everything the car touched that was not another car. Walls and obstacles are in the same bucket.
An earlier single-run breakdown (`diag/why_hit.py`, 23 contacts) split as 13 obstacles and 10 walls,
so the mixture is roughly even and neither term can be ignored.

The consequence for this note: s919 raised the price of *both*, and the evaluation metric measures
*both*. "Static contacts per km" is the honest name and is the one used above. No claim here
distinguishes hitting a crate from brushing a wall.

Scenario structure also matters when reading any pooled number: with an opponent in the race the
rate is about three times the solo rate (s915u768: 2.56 /km solo against 6.74 /km in traffic), so a
pooled rate is dominated by the traffic scenarios and a change that only affects solo driving would
barely show.

## 6. What the four have in common

In all four cases the measurement was sound and the step after it was not.

| run | measurement (correct) | inference (untested) | outcome |
|---|---|---|---|
| s916 | the action collapses to two outcomes at speed | so give it resolution and it will steer around things | 7.26 /km, and it aims at obstacles *more* |
| s917 | the obstacle's effect on the action (0.105) is below the exploration noise (0.15) | so narrow the noise and the signal will act | 9.27 /km |
| s918 | same | same, and also drop the leash | 10.51 /km, and not a clean control |
| s919 | a contact costs 2.4 s of driving | so the policy is choosing contacts; reprice them | 5.81–6.47 /km, and impact speed *rose* |

The inferences share a form: they assume the policy is failing at something it is trying to do —
steer finely, hear a signal, weigh a cost — and that relieving that constraint will change the
behaviour. None of them checked whether the policy was trying. The s919 post-hoc is the clearest
counter-example, because the concession it declined (arrive slower) needed no extra steering
resolution, no louder signal and no new skill; it needed only to value 36 points, and it did not
take it.

The contrast worth keeping is the fifth candidate of the day, an extension of the reward probe.
That one was **refuted before any training started**, from data already on disk, and cost no GPU
time — against roughly six hours for s916, s917 and s918 and another hour for s919. The difference
was not that the fifth idea was better. It was that its inference was stated in a form that
existing measurements could contradict, and then checked against them first.

The rule this suggests, for the next candidate: before spending a GPU-hour, write down what the
policy would have to be doing for the change to help, and ask whether anything already measured
says it is doing that. Three of the four above would not have survived that question.
