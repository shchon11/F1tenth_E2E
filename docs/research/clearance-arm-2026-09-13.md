# A clearance arm: keeping the executed plan off what the LiDAR can see

**2026-09-13.** A runtime controller layer that adjusts the policy's plan — bending it, and where it
cannot bend, slowing it — until every point of it keeps a stated body-edge margin from the local
occupancy built out of the current LiDAR frame. No map, no pose, no training. Measured before and
after with `crash_attribution.py` on the eight held-out proxy tracks, under `legacy` and under
`fixed_low`. That script and its results live in the worker directory beside this checkout
(`work/clearance-arm/`), not in the repository: it reads checkpoints and writes JSON that is not
part of the simulator.

Module: [`f1sim/learn/clearance.py`](../../f1sim/f1sim/learn/clearance.py). Arm names:
`clearance`, `fixed_low+clearance`, and the same two with `+tcs`.

**Headline.** On the frozen original over 256 held-out trials, with the policy untouched:
`legacy` → `clearance` is **223 → 203 collisions and 79 → 110 completions** for 3.1 % of mean speed;
`fixed_low` → `fixed_low+clearance` is **182 → 169 and 105 → 130** for 2.5 %. The share of
collisions whose plan had passed *through* an occupied cell falls **26 % → 11 %** and **24 % → 8 %**,
while `fixed_low` alone moves it 26 % → 24 % — the improvement is geometric, and a friction clamp
does not produce it. What moved in the plan-margin distribution is its **lower tail**: p25 goes from
−0.140 m, a plan a full body radius inside an obstacle, to −0.028 m. The **median** barely moves,
0.001 → 0.018 m. The arm removed the class of collision where the plan was aimed into something the
scan could see; the survivors still have almost no margin, and 169 of 256 trials still end in one.
These tracks are hard.

**In traffic**, on the same three clean held-out maps with a teacher-driven opponent and scripted
brake / stop / shift events, 48 learner trials per arm: **car contacts 29 → 15 → 11** and **wall
collisions 22 → 22 → 17** (`legacy` → `fixed_low` → `fixed_low+clearance`), completions **21 → 26 →
30 of 48**, and **passes 37 → 40 → 42** with no lead ever lost. The arm does **not** buy that by
backing off: the mean following gap changes by −0.06 m when the layer is added to `fixed_low`
(3.95 → 3.89 m) and the mean gap inside the attacking window is flat across all three arms
(1.77 / 1.76 / 1.80 m).

## The evidence it answers

Root measured this on 2026-09-13 (`evidence_attr3_frozen.json`, frozen original
`48cc698f…`, 32 trials on each of eight held-out tracks):

| fact | number |
| --- | --- |
| privileged raceline teacher, same iLQR tracker | 30–32 of 32 trials completed, 3.3–4.5 m/s |
| the policies, same tracks | 7–24 of 32 |
| **the policy's own last plan at its collisions**, minimum body-edge clearance | **median 0.00–0.06 m** |
| collisions whose plan margin was under 0.10 m | 85 % |
| collisions whose plan passed *through* occupied cells | 49 % |
| the car's own clearance 25 ms before impact | ~0.2 m |
| speed at impact on the narrow real floors, against µ 0.93 | 2.6–2.9 m/s |

Read together: the tracks are feasible, the tracker can follow safe plans, and the failure is not
grip. **The policy plans with no margin.** The grip clamp (`fixed_low`) was the one thing that moved
held-out numbers on 2026-09-12 and it is a runtime layer that needs no training. This is the same
idea applied to geometry.

## What the arm is

Each control step, from the 1081 returns and nothing else:

1. **A local occupancy.** Each return that came back becomes a point in the car's own frame — the
   scanner sits 0.297 m ahead of `base_link`, which is one and a half of the margin being defended,
   so the offset is taken out — and is binned into a 0.06 m grid covering −0.24 to 4.44 m ahead and
   ±2.04 m across. A bearing with **no** return marks nothing: an unobserved bearing is unknown, not
   a wall at 10 m.
2. **A distance field.** The exact clipped Euclidean transform, through Felzenszwalb's separable
   decomposition of the squared distance, written out as a fixed number of shifted minima rather
   than the usual lower-envelope scan. That matters twice: it is exact (a chamfer approximation
   would misread a diagonal by up to 4 %, and the whole quantity here is 0.2 m), and it has no
   data-dependent control flow, so it is as CUDA-graph-safe as the rest of the path. The field
   saturates at 0.60 m, which is above everything the arm reasons about and bounds the transform at
   10 offsets per pass.
3. **A bend.** Thirteen candidate plans, each the policy's own knots plus one curvature offset
   parameterised by the lateral displacement it produces at the evaluation horizon, and each scored
   on the path `mpc.path_points` actually produces for it — not on a linearisation. The offset is
   **tapered to zero one knot past the horizon**, so the tail curvature is untouched and
   `fixed_low`'s speed envelope over the tail is not moved along with it. The score is described
   below; it is where the two interesting mistakes are.
4. **A cap.** Where no bend reaches the margin, the arm takes away the fraction of the available
   speed range that the margin is short by: `v_stop` at no clearance, rising linearly to the
   caller's own cap at the margin. Anchoring the top on that cap rather than on a constant is what
   makes the rule exactly non-binding at the margin — a constant below the cap would slow a car that
   was keeping the margin perfectly well. The cost is that the speed action scales with the
   environment's cap: at the 9 m/s cap these runs use, a plan at half the margin is still allowed
   4.8 m/s, which is above the pace these policies hold, so the cap bites only where the margin is
   nearly gone. By the evidence, that is where the collisions are. The result is then put through a
   backward braking pass so the car may still be fast now if it can shed the speed by the time it
   arrives. A plan carries two speed numbers and its profile is linear between them, so the
   envelope has to be fitted with a line: the arm computes the two lines that matter — the one that
   keeps the near target as high as the envelope allows, and the policy's own profile scaled down
   until it fits — and takes the faster. Neither can exceed what the policy asked for, and taking
   only the first crushes the far target to nothing whenever the envelope is flat and low, which is
   a hard brake the geometry never asked for.

### The one design decision everything else follows from

It sits on `PlanTracker._plan_hook`, which runs **before the action is decoded**, and not on the
solver.

`tracker.last_ref` is built inside `mpc.solve` *from the action*. It is what the iLQR follows and it
is what `crash_attribution.py` measures. An arm that changes the action is therefore measured, and
executed, as the plan it produced — which is what makes the before/after table below a statement
about the same object the evidence above is about.

It also makes the composition order-free. `fixed_low` binds `tracker._solver`; this binds
`tracker._plan_hook`. Neither can overwrite the other, and the friction envelope is computed on the
adjusted geometry whichever went on first. `tests/test_clearance.py` and
`tests/test_policy_node_clearance.py` both assert that, in the simulator and on the node.

### Three things it will not do

* **Raise a speed**, or straighten a plan the policy bent. Every change is one-sided.
* **Change anything when the margin is already met.** The bend is applied in normalized curvature
  units so that adding zero is exact, and each speed is replaced through a `where` rather than a
  round trip through `decode`/`encode`, so an unadjusted plan comes back **bit-identical**.
* **Read a map.** `track.edt` is what the *evaluation* measures with. It is not on the runtime path
  and cannot be: the car does not have one.

### How a candidate is scored, and two ways of getting it wrong

The score is the **mean, over the evaluation window, of the candidate's body-edge clearance
saturated at the margin** — how much of the arc ahead the plan keeps clear, and how clear — minus a
small penalty on the deviation. A candidate that keeps the margin everywhere scores exactly
`margin`, which nothing can beat, so "meets the margin" is still identified exactly and the penalty
then picks the smallest bend that is enough. Both of the obvious alternatives were tried first and
both are wrong:

**Pointwise clearance is wrong** because the distance field is unsigned: a plan point a metre past a
wall reads as a metre of free space, so a candidate that drives *through* the wall and out the far
side outscores one that only grazes it. Everything from the plan's first **contact** is therefore
void — the car stops at the first thing it meets.

**A running minimum is wrong** for the opposite reason. It is pinned by the tightest point the plan
has already passed, and on a narrow floor that is routinely the window's own first sample, which no
candidate can move — every plan leaves (0,0) heading straight ahead. Scored that way every candidate
ties, the deviation penalty picks zero, and the arm does nothing on exactly the tracks it exists
for. Only an actual contact makes what comes after it meaningless; a merely tight point does not.

The window starts at 0.50 m for the same reason: `base_link` is the rear axle and the car's body
reaches about 0.48 m ahead of it, so a plan point closer than that is inside the car's own
footprint. Its clearance is a fact about where the car already is — on these tracks the car drives
safely at 0.15–0.20 m from a wall all the time — and counting it would cap the speed for a wall the
car is already alongside.

## Cost

Measured under the deployment budget's own protocol (`python3 -m f1sim.learn.budget --clearance`):
CPU, single thread, batch 1, fp32, the fastest of five blocks of 200 iterations after warm-up.

| part | ms | of a 25 ms control step |
| --- | ---: | ---: |
| occupancy + distance field (5 304 cells, 20 transform passes) | 0.48 | 1.9 % |
| candidate search + speed cap (13 candidates × 25 samples) | 0.77 | 3.1 % |
| **total** | **1.25** | **5.0 %** |

A proxy on this machine, like the memory-policy table it is measured beside, and not a Jetson
number. It is tensor work on whatever device the session is on, so in training and evaluation it is
batched across environments and costs a fraction of this per car.

<!-- RESULTS -->

### Per track — collisions / completions of 32

| track | `legacy` | `clearance` | `fixed_low` | `fixed_low+clearance` |
| --- | --- | --- | --- | --- |
| `scene:scene_0912_2355` | 32 / 2 | 29 / 11 | 30 / 6 | 23 / 12 |
| `scene:scene_0912_2355+hard1` | 32 / 0 | 30 / 4 | 29 / 6 | 28 / 7 |
| `real:map16x07+hard2` | 31 / 5 | 28 / 10 | 30 / 6 | 27 / 9 |
| `real:map12x16+hard3` | 32 / 4 | 32 / 1 | 32 / 0 | 32 / 3 |
| `gen:control:9100+hard4` | 18 / 22 | 19 / 24 | 13 / 21 | 14 / 22 |
| `real:korea_2025_iccas+hard5` | 28 / 10 | 27 / 20 | 23 / 21 | 19 / 26 |
| `real:map16x07` | 25 / 19 | 21 / 20 | 16 / 21 | 12 / 26 |
| `real:map12x16` | 25 / 17 | 17 / 20 | 9 / 24 | 14 / 25 |

### Per track — the plan's margin at the collisions it did have [m], median

| track | `legacy` | `clearance` | `fixed_low` | `fixed_low+clearance` |
| --- | --- | --- | --- | --- |
| `scene:scene_0912_2355` | 0.085 | 0.410 | 0.160 | 0.410 |
| `scene:scene_0912_2355+hard1` | 0.014 | 0.235 | 0.160 | 0.110 |
| `real:map16x07+hard2` | 0.040 | 0.040 | 0.025 | 0.010 |
| `real:map12x16+hard3` | -0.140 | -0.040 | -0.115 | -0.028 |
| `gen:control:9100+hard4` | 0.018 | 0.040 | 0.040 | 0.042 |
| `real:korea_2025_iccas+hard5` | 0.001 | -0.040 | -0.028 | -0.028 |
| `real:map16x07` | 0.010 | 0.060 | 0.060 | 0.050 |
| `real:map12x16` | -0.090 | 0.010 | -0.040 | 0.014 |

### Per track — pace: mean speed [m/s] over first attempts, and median lap time [s] of the trials that completed

| track | `legacy` speed | `legacy` lap (n) | `clearance` speed | `clearance` lap (n) | `fixed_low` speed | `fixed_low` lap (n) | `fixed_low+clearance` speed | `fixed_low+clearance` lap (n) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `scene:scene_0912_2355` | 4.37 | 14.48 (2) | 4.18 | 15.55 (11) | 3.72 | 18.65 (6) | 3.65 | 17.80 (12) |
| `scene:scene_0912_2355+hard1` | 4.39 | — (0) | 4.15 | 15.73 (4) | 3.67 | 19.77 (6) | 3.57 | 18.23 (7) |
| `real:map16x07+hard2` | 3.81 | 7.90 (5) | 3.73 | 8.23 (10) | 3.29 | 10.43 (6) | 3.25 | 9.63 (9) |
| `real:map12x16+hard3` | 3.87 | 8.48 (4) | 3.80 | 8.50 (1) | 3.22 | — (0) | 3.19 | 10.43 (3) |
| `gen:control:9100+hard4` | 4.92 | 10.83 (22) | 4.72 | 11.15 (24) | 4.18 | 12.58 (21) | 4.08 | 12.48 (22) |
| `real:korea_2025_iccas+hard5` | 4.74 | 8.40 (10) | 4.58 | 8.70 (20) | 4.03 | 9.85 (21) | 3.87 | 9.88 (26) |
| `real:map16x07` | 3.90 | 7.90 (19) | 3.83 | 7.88 (20) | 3.37 | 9.20 (21) | 3.29 | 9.10 (26) |
| `real:map12x16` | 4.01 | 7.75 (17) | 3.93 | 7.78 (20) | 3.44 | 9.13 (24) | 3.37 | 9.28 (25) |

### Pooled over the eight tracks

| arm | collisions / 256 | completed | plan margin median | p25 | share < 0.10 m | share through occupied | mean speed m/s | median lap s (n) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 223 / 256 | 79 | 0.001 | -0.140 | 85% | 26% | 4.25 | 8.94 (79) |
| `clearance` | 203 / 256 | 110 | 0.010 | -0.040 | 78% | 11% | 4.12 | 9.81 (110) |
| `fixed_low` | 182 / 256 | 105 | 0.010 | -0.090 | 74% | 24% | 3.62 | 11.20 (105) |
| `fixed_low+clearance` | 169 / 256 | 130 | 0.018 | -0.028 | 71% | 8% | 3.53 | 11.22 (130) |

### What the arm did (mean over the eight tracks)

| metric | clearance | fixed_low | fixed_low+clearance |
| --- | --- | --- | --- |
| `controller/a_brake_realised_mean` | — | 1.808 | 1.900 |
| `controller/a_brake_realised_spread` | — | 1.287 | 1.295 |
| `controller/clearance_bent_frac` | 0.191 | — | 0.195 |
| `controller/clearance_plan_margin_after` | 0.164 | — | 0.166 |
| `controller/clearance_plan_margin_before` | 0.142 | — | 0.143 |
| `controller/clearance_shift_mean` | 0.043 | — | 0.043 |
| `controller/clearance_slowed_frac` | 0.134 | — | 0.119 |
| `controller/clearance_speed_cut_mean` | 0.805 | — | 0.690 |
| `controller/flag_degeneracy_run` | — | 0.000 | 0.000 |
| `controller/flag_overestimate_run` | — | 0.000 | 0.000 |
| `controller/mu_used_mean` | — | 0.734 | 0.734 |
| `controller/mu_used_std` | — | 0.000 | 0.000 |

<!-- /RESULTS -->

## In traffic

The table above is a solo measurement, and it is not the interesting half for a racing car. The arm
sees opponents: other cars are traced into the LiDAR as porous meshes (`sim._car_boxes`,
`lidar.HIT_CAR`), so an opponent's returns land in the same local occupancy a wall's do — sparser,
because a porous hit can be dropped, but there. The same three clean held-out maps, `--race-size 2
--opponent teacher --opp-events brake,stop,shift --opp-event-rate 3.0 --envs 32 --protocol trials
--budget-laps 3 --seed 4401`, run through `evaluate.evaluate` itself:

### all three

| arm | completed | collided | car / wall | passes | lost | contention | following | attacking | pace | follow med m | follow p10 m | attack <1 m | speed m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 7 / 16 | 9 | 6 / 10 | 13 | 0 | 80% | 51% | 26% | 1.86 | 2.99 | 1.21 | 16% | 4.55 |
| `fixed_low` | 5 / 16 | 11 | 10 / 7 | 10 | 0 | 86% | 66% | 26% | 1.55 | 3.64 | 1.22 | 16% | 3.89 |
| `fixed_low+clearance` | 9 / 16 | 7 | 6 / 4 | 12 | 0 | 80% | 57% | 24% | 1.83 | 3.38 | 1.18 | 18% | 3.95 |

### map16x07

| arm | completed | collided | car / wall | passes | lost | contention | following | attacking | pace | follow med m | follow p10 m | attack <1 m | speed m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 4 / 16 | 12 | 18 / 9 | 4 | 0 | 96% | 85% | 42% | 1.49 | 3.02 | 1.22 | 14% | 3.67 |
| `fixed_low` | 7 / 16 | 9 | 8 / 7 | 8 | 0 | 98% | 75% | 27% | 1.33 | 3.84 | 1.24 | 15% | 3.36 |
| `fixed_low+clearance` | 7 / 16 | 9 | 9 / 8 | 10 | 0 | 98% | 69% | 26% | 1.43 | 3.59 | 1.34 | 16% | 3.18 |

### control:9100

| arm | completed | collided | car / wall | passes | lost | contention | following | attacking | pace | follow med m | follow p10 m | attack <1 m | speed m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 10 / 16 | 6 | 3 / 8 | 18 | 0 | 80% | 45% | 22% | 1.78 | 3.02 | 0.92 | 23% | 4.97 |
| `fixed_low` | 11 / 16 | 5 | 2 / 4 | 15 | 0 | 79% | 38% | 12% | 1.52 | 4.00 | 1.41 | 22% | 4.14 |
| `fixed_low+clearance` | 13 / 16 | 3 | 1 / 2 | 15 | 0 | 76% | 38% | 14% | 1.64 | 3.64 | 1.36 | 17% | 4.05 |

### iccas25

| arm | completed | collided | car / wall | passes | lost | contention | following | attacking | pace | follow med m | follow p10 m | attack <1 m | speed m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 7 / 16 | 9 | 8 / 5 | 15 | 0 | 90% | 50% | 26% | 1.48 | 2.94 | 0.82 | 23% | 4.71 |
| `fixed_low` | 8 / 16 | 8 | 5 / 11 | 17 | 0 | 90% | 51% | 19% | 1.58 | 3.68 | 1.19 | 22% | 4.01 |
| `fixed_low+clearance` | 10 / 16 | 6 | 1 / 7 | 17 | 0 | 87% | 47% | 17% | 1.53 | 3.74 | 1.22 | 20% | 3.92 |

### The three maps pooled — 48 learner trials per arm

| arm | completed / 48 | collided | car contacts | wall collisions | passes | leads lost | follow mean m | attack mean m | mean speed m/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 21 | 27 | 29 | 22 | 37 | 0 | 3.04 | 1.77 | 4.58 |
| `fixed_low` | 26 | 22 | 15 | 22 | 40 | 0 | 3.95 | 1.76 | 3.89 |
| `fixed_low+clearance` | 30 | 18 | 11 | 17 | 42 | 0 | 3.89 | 1.80 | 3.80 |

`completed` / `collided` are first attempts and sum to the trial count; `car contacts` and `wall
collisions` are `TrafficMeter`'s count over the whole rollout, including after an auto-reset, so
they split what was hit rather than partitioning the trials. The `follow` columns are an addition:
the arc gap to the nearest car ahead at every step the learner is following one, recorded by
wrapping `TrafficTrace.update` inside the measuring process only.

### What it does to following, and to a pass

**The contacts fall and the passes do not.** Over 48 learner trials, car contacts go 29 → 15 → 11
and wall collisions 22 → 22 → 17 — the friction clamp does nothing at all to the walls, and the
geometry layer takes a fifth of them. Passes go 37 → 40 → 42, and `leads_lost` is 0 for every arm.

**And it does not buy that by backing off.** The +0.9 m of following distance between `legacy` and
`fixed_low` is the friction clamp driving a slower car. Adding this layer on top moves it −0.06 m,
and inside the attacking window — where a pass is actually on — the mean gap is flat at
1.77 / 1.76 / 1.80 m. The car is not sitting further back; it is arriving with a plan that no longer
points at the other car.

That is what the design predicts, and it is worth stating because a naive margin layer would do the
opposite. The arm judges only the arc the tracker's own reference covers — 0.6 s, about 2.4 m at
this pace — and acts only where a plan point comes within 0.34 m of a return. A car three to four
metres ahead is outside the window entirely, so following it costs nothing; the arm engages at the
moment the plan turns into the opponent, which is the moment the contact would have happened.

**Sample size.** The `all three` row is one 32-env run over the three maps — 16 learner trials — and
at that size the arms disagree with the 48-trial pooling (`fixed_low` reads 5/16 there against
`legacy`'s 7/16, and 26/48 against 21/48 per map). The per-map runs are the same flags on each map
alone; they are what the paragraphs above are read from.

## How to read the pace

`mean speed` is over every active step of every first attempt, so it is not distorted by which
trials finished; it is the honest pace number and it falls 3.1 % (`legacy` base) and 2.5 %
(`fixed_low` base).

The pooled **lap time** rises — 8.94 → 9.81 s on the `legacy` base — and that is composition, not
the car going slower. The completed sets are not the same trials: 31 more finish, and the tracks
that gain most of them are the slow ones (`scene:scene_0912_2355` goes 2 → 11 completions at ~15.5 s
a lap). On the two tracks whose completion count barely moves it is 7.90 → 7.88 s and 7.75 → 7.78 s.
On the `fixed_low` base, where 25 more trials complete, the pooled median moves 11.20 → 11.22 s. The
per-track table above is there so this can be read rather than taken on trust; a properly paired
comparison over the trials both arms completed would need per-trial identity, which this script does
not record.

## Limitations, stated

* **The backward pass is optimistic under `fixed_low`.** It assumes 3.3 m/s², the deceleration the
  whole command chain was measured to deliver at the bottom of the friction range. `fixed_low`
  clamps the solver tighter than that — 2.97 m/s² on a straight plan at µ 0.734, and less in a
  corner; the composite run above logged a realised bound of **1.90 m/s²** on average
  (`controller/a_brake_realised_mean`), so the gap is real and measured rather than hypothetical. It is left optimistic on purpose: reading the other layer's bound would make the
  installation order matter, which is the one property that makes the two composable. The cap is a
  target re-issued at 40 Hz and tightening as the obstacle nears, not a stopping guarantee.
* **Quantisation.** Binning a return to the cell containing it puts it within half a cell of where it
  is (≤ 0.03 m per axis, unbiased), and sampling a field defined on cell centres slightly
  *under*-reads the clearance along a medial axis, where the interpolation cuts a corner off a
  concave function. Both are small against a 0.20 m margin and both err safe.
* **Unknown is free.** Outside the grid, beyond the range, behind the 270° window and along a dropped
  beam, the arm assumes space. That is the only assumption a LiDAR-only car can make without braking
  for its own blind spot every step, and it means the arm cannot defend a margin against something
  it never saw.
* **One lateral parameter.** The bend is a single curvature offset, so it can move the plan over but
  not reshape it. A gap that needs an S is out of reach, and what happens there is the speed cap.
* **Traffic is 48 learner trials per arm on three maps.** Enough to see a factor-of-two change in
  car contacts; not enough to separate a 40-from-42 change in passes from noise. The pass column
  supports "nothing was given back", not "passing improved".
* **An opponent is a porous target.** A car's returns can be dropped beam by beam, so its footprint
  in the grid is sparser and more ragged than a wall's — the arm defends the same margin against a
  noisier estimate of where the other car is.
* **Not trained under.** Deliberately: the grip clamp's lesson was that a policy trained under a
  clamp learns to lean on it. Train legacy, deploy clamped.
