# Motion-grounded memory: can the policy's state see the other car move? (2026-09-14)

Branch `feat/motion-memory`, base `0b78111`. Everything here is simulation except the noise floors
measured on the real recordings, which are said to be real where they appear.

## The question, and why it is narrower than last time

[future-head-2026-09-14](future-head-2026-09-14.md) asked whether the actor's GRU carries the nearest
opponent's state **half a second ahead** and answered no (R² 0.0–0.36, spread ±0.1–0.4, the ordering
of the arms flipping between rollout seeds). That result is compatible with two very different
stories: the state is a poor extrapolator, or the state never knew where the other car was at all.

So this work asks the narrower question first, and the answer to it decides whether the harder one
is even well posed:

> From the representation the policy actually has, can a linear read-out produce the nearest
> opponent's **present** relative state — Δx, Δy, Δv_x, Δv_y in the ego frame?

The two halves have different answers by construction, and that is the point.

* **Δx, Δy is *object observability*.** A car a few metres away is a handful of short beams. Where
  it is does not need any memory; a read-out that cannot recover it says the probe or the
  representation is broken rather than anything about motion.
* **Δv_x, Δv_y is *temporal motion observability*, and it is the headline.** A single range image
  contains no velocity. Recovering it requires relating two instants — the frame stack, the
  recurrent state, or an explicitly aligned residual — so it is the column where "does this
  architecture build frame-to-frame correspondence?" actually gets tested.

The hypothesis under test is the user's reading of the future-head result: PPO gives the GRU no
reason to build that correspondence. Ego motion and static apparent motion dominate the raw scan,
the opponent is a few beams, and remembering ego dynamics is the easy way to reduce every loss the
policy has. If that is right, the remedy is inductive bias and not capacity — so the GRU is not
widened anywhere in this work.

## E1 — the scoreboard

`python -m f1sim.learn.probe_hidden --current`. Ridge read-out from the policy's own state, held-out
**cars** (25 % of the env columns, never rows inside one trajectory), 8 draws of which cars × 2
independent rollout seeds. R² and MAE, presence-conditioned (rows with no car inside
`overtake_range` are excluded — 12.9 % of otherwise-valid rows — because there is no relative
position to a car that is not there), and split near / mid / far so a handful of very close
opponents cannot carry a pooled number. The full table, with the distance split and the per-seed
rows, is `work/motion-memory/work/e1/e1.md`; its definition is fixed and each arm adds a column.

The arms are a ladder of how much temporal information the representation is allowed to have, and
nothing else changes between them — same rollout conditions, same seeds, same split protocol:

| arm | what it is |
|---|---|
| `memory_off` | the frozen original, feedforward, fed the **newest frame in every slot** of the stack (`--stack-mode repeat`). Same weights, same proprio; the 150 ms of apparent motion the stack carries is the one thing removed |
| `frame_stack` | the frozen original, feedforward, the real 6-frame stack, probed at its trunk features (`--probe-state trunk`) |
| `gru_warm` | the same checkpoint warm-started into GRU 128 + `memory,edges`. Its projection is zero, so the policy driving is still bit-identical to the original's: what the probe reads is a **random** recurrent feature map over the original's embedding |
| `gru_trained` | `cl_mem_s701` u8 — the same architecture with 8 PPO updates behind it |

### The headline

Median R² over the 16 draws, with the interquartile range. (The mean and its spread are in the full
table; they agree except where one draw held out cars that barely moved and divided by a variance
near zero — one draw in sixteen did, reading −3.5 on `ego_yaw_rate` against +0.7 in the other
fifteen, which is a fact about that draw and not about the representation.)

| | `memory_off` | `frame_stack` | `gru_warm` | `gru_trained` |
|---|---|---|---|---|
| **Δv_x** R² | **−0.059** [−0.13, +0.06] | **+0.117** [−0.09, +0.14] | **+0.185** [+0.16, +0.23] | **+0.112** [+0.06, +0.22] |
| **Δv_y** R² | −0.007 [−0.16, +0.08] | −0.025 [−0.19, +0.08] | +0.132 [+0.09, +0.20] | +0.178 [+0.08, +0.21] |
| Δx R² | +0.111 [−0.00, +0.22] | +0.115 [−0.03, +0.18] | +0.235 [+0.14, +0.33] | +0.241 [+0.18, +0.36] |
| Δy R² | +0.070 [+0.01, +0.18] | +0.085 [+0.03, +0.15] | +0.190 [+0.08, +0.27] | +0.202 [+0.03, +0.27] |
| Δv_x MAE [m/s] | 1.875 | 1.891 | 1.743 | 1.828 |
| Δv_y MAE [m/s] | 1.637 | 1.621 | 1.561 | 1.554 |
| Δx MAE [m] | 3.092 | 3.059 | 2.924 | 2.840 |
| Δy MAE [m] | 2.069 | 2.031 | 2.004 | 2.018 |
| ego speed R² | +0.828 ±0.034 | +0.790 ±0.117 | +0.898 ±0.024 | +0.918 ±0.014 |
| ego yaw rate R² | +0.789 ±0.020 | +0.457 ±1.032 | +0.807 ±0.030 | +0.838 ±0.022 |

Four things, in order of how much the evidence supports them.

1. **The read-out works, and the control says so.** The ego's own speed and yaw rate come back at
   R² 0.79–0.92 with a spread of ±0.02–0.03 on every arm. That is what makes an 0.1 on the opponent
   columns worth believing rather than an artefact of the fit. (`frame_stack`'s ±1.03 on the yaw
   rate is the one bad draw named above, not a property of that arm: its median is +0.711.)
2. **Δv is not in a representation with no temporal information, and it is not much in any of
   them.** `memory_off` reads −0.06 and −0.01 on the two velocity columns: at chance, as it must be,
   because nothing in that network relates two instants. Every other arm is between 0.0 and +0.19.
   Against a target whose MAE is 1.5–1.9 m/s on cars closing at up to a few m/s, none of these is a
   representation anything could plan through.
3. **The recurrence is what separates the arms, and training on top of it does not.** Both GRU arms
   read +0.11 … +0.19 on both velocity columns against −0.06 … +0.12 for the two feedforward ones,
   and the two GRU arms are **not separable from each other**: `gru_warm` is ahead on Δv_x (+0.185
   vs +0.112) and behind on Δv_y (+0.132 vs +0.178), inside each other's interquartile range in both
   cases. `gru_warm` is a **random** GRU — its projection is zero, so the policy it drives with is
   the frozen original's — so what that column measures is a longer window and nothing else. Eight
   PPO updates on top of it buy nothing measurable here, which is the experiment's premise showing
   up in the data: PPO is not shaping the state toward this.
4. **Position is barely better than velocity, which is the surprise.** Δx reads +0.11 with no
   temporal information at all and +0.24 with a GRU — for a quantity one scan determines, at a MAE
   of 2.8–3.1 m in a 12 m window. The representation is a *driving* representation: it encodes the
   corridor, and where in it a handful of car-shaped beams sit is not something it is asked to keep.
   That reframes the future-head result: the opponent's future is not extractable partly because the
   opponent's **present** is barely there either.

### By distance

Read MAE in the distance split, and read R² there with its denominator in mind: within a bin R²
divides by that bin's own variance, and a read-out fitted across all distances has most of its
predicted variation *between* the bins, so a strongly negative within-bin R² beside a reasonable
pooled one says the read-out is largely recovering how far away the car is rather than how it is
moving. That is exactly what the table shows, and it is worth knowing.

Δv_x MAE [m/s], mean over the 16 draws:

| bin | `memory_off` | `frame_stack` | `gru_warm` | `gru_trained` |
|---|---|---|---|---|
| near (< 2 m) | 1.414 ±0.324 | 1.473 ±0.206 | 1.328 ±0.186 | 1.268 ±0.309 |
| mid (2–5 m) | 1.741 ±0.253 | 1.786 ±0.302 | 1.670 ±0.231 | 1.895 ±0.272 |
| far (> 5 m) | 2.207 ±0.316 | 2.149 ±0.300 | 2.085 ±0.304 | 1.824 ±0.315 |

Every cell is inside its neighbours' spread. The honest summary of this table is that **the ladder is
visible on pooled Δv R² and it does not survive being cut three ways** — 500–1200 held-out rows per
bin is not enough to separate arms that differ by 0.1 of an R². It is reported because the addendum
asks for it and because a read-out that only worked at two metres would have shown up here.

## E2 — an ego-motion-aligned residual

### What it is

Not `r_t − r_{t−k}`. The scan from k control steps ago becomes a point cloud in the old sensor's own
frame, is carried into the current one, and is **re-rasterised onto the current angular bins**:

    R_t(θ) = r_t(θ) − r̃_{t−k}(θ),   r̃_{t−k} = rasterise( T_{t←t−k} P_{t−k} )

with, in order: the 3-D lift by the old roll/pitch using the simulator's own beam geometry
(`Ry(pitch) Rx(roll)` on the level bearing, so a beam that was tilted into the floor is placed where
it hit); the planar carry by k composed constant-(v, ω) arcs built from the trapezoidal average of
the two endpoint measurements; the expression in the current sensor plane by the inverse of the
current tilt, dropping any point whose out-of-plane offset exceeds `z_tol`; and a nearest-bin
scatter keeping the closest point, which is the rule the sensor itself applies.

**Only the attitude's *change* enters.** Both tilts are taken relative to their own midpoint, which
preserves the difference exactly and removes whatever bias they share. That is not a nicety: the
VESC attitude estimate is biased and drifts — in simulation its |roll, pitch| rms over a rollout is
about 11 deg, and five of the thirteen competition recordings swing past 40 deg (one to 178) with
the quaternion unit-norm throughout. Fed the absolute value, the warp integrates the ego's arcs in a
frame tilted by that bias and the out-of-plane test then discards points the current scan can see.
Measured on one clean recording, everything else fixed:

| what the warp is told about the attitude | beams with a prediction | static-scene survivors |
|---|---|---|
| the absolute estimate | 70.3 % | 2.89 % |
| **only how it changed** | **82.1 %** | **2.67 %** |
| nothing (told the car is level) | 91.3 % | 3.06 % |

It is also the more honest frame: the ego motion is measured in the car's own axes, so the plane the
arcs are integrated in is the car's own mean attitude over the interval and not a level plane nobody
measured. Dropping the attitude altogether buys the most coverage and the worst false-positive rate,
which is the out-of-plane test doing real work.

**Everything the warp uses is measured on the car** — VESC wheel speed, IMU gyro z, IMU roll/pitch.
No pose, no map, no odometry beyond those k steps. `f1sim_ros/policy_node.py` already reads all
three and now hands them to the channel.

Three details are load-bearing rather than decorative:

* **a no-return warps as a lower bound, not a point.** "Nothing out to `range_max`, that way" is
  still a bound after the car has moved, so only something appearing *closer* than it counts.
  Treated as a point at `range_max`, every far beam of every straight would report the ego's own
  0.9 m of travel as motion — a systematic false positive on the part of the scan a policy reads to
  plan. Measured: the ungated |R| p99.9 on the real bags falls from 9.10 m to 5.81 m when the bound
  is honoured, and again to 6.12 m / p99 1.375 m once beams with no *current* return are also
  treated as "the sensor said nothing" rather than as a disappearance.
* **the valid mask is a channel.** A bin no warped point reached, or one whose current beam did not
  return, is *unknown*: the residual there is exactly 0 and `aligned_valid` says which zeros are
  which. "I cannot tell" and "nothing moved" must not be the same number.
* **a bearing tolerance, stated.** The comparison is against the warped scan's envelope over
  ±`tol_beams`, because the warp has a real bearing uncertainty: over k = 4 steps at 9 m/s the ego
  moves 0.9 m, so a 1 % speed error is 9 mm, a 0.05 rad/s yaw-rate error is about one beam, and a
  1 deg error in the measured tilt — the plant's floor wobble exactly — is four beams at 0.25 deg
  spacing. Against an oblique wall four beams is metres of range, which is why the ungated residual's
  p99 is metres while its median is centimetres. `tol_beams = 0` is the literal one-bin difference
  and the floor table reports both.

The channel emits three rows, appended after every column the original network had and
zero-initialised in the first convolution: `aligned` (the soft-thresholded residual), `aligned_prev`
(the warped previous range) and `aligned_valid` (the mask). The one-channel variant (`aligned`
alone) is available as the addendum's ablation.

### Does the warp work? A scene whose answer is known

`tests/test_aligned_scan.py` drives the channel through an analytic world — a circular wall, with an
optional disc in it, ranges computed in closed form with the simulator's beam geometry:

* pure translation and a hard arc through a **static** world give a residual of **exactly 0.0**, not
  approximately, at every beam that had a prediction;
* a disc moving across the field gives a residual only at its own bearings (< 60 deg of span),
  negative where it arrived and positive where it left, while a **world-fixed** disc at the same
  place gives exactly 0;
* told the wrong speed, the channel reports the whole static world as moving.

### The noise floor, on the real recordings

`python -m f1sim.learn.aligned_floor bags`. Every recording is replayed through the same channel from
the same three sensors the car will use: `/odom` wheel speed, `/sensors/imu/raw` gyro z, and
roll/pitch from that message's orientation quaternion via the very function
`policy_node.attitude_from_orientation` uses.

**Ten of the 22 recordings are usable, and the twelve that are not are a finding of their own.** The
dataset README warns that some raw gyros are unusable, so the check is made per bag on the data
rather than from a folder-name tag: seven are refused for a gyro-z |p99| of 97–245 rad/s against the
3–4 a 1/10 car turns at, and **five competition recordings are refused for an attitude |p99| of
40–112 deg**. That second group is not a gyro problem: in `20260826-193532` the VESC's own
orientation quaternion is unit-norm throughout and still swings the extracted roll to 178 deg, with
3.5 % of samples past 20 deg in runs up to 86 samples (~1 s), against 0.17 % in runs of ≤ 10 for a
clean bag. **The policy node feeds that same quaternion into the observation's `imu_att` columns
today**, so this is a deployment risk for the existing policy and not only for this channel. It is
recorded here and not fixed here.

Ungated (τ = 0), over the ten usable recordings, by lag — the sensitivity the addendum asks for,
with **k = 4 declared a priori and not chosen from these numbers**:

| k | bins the warp could predict | \|R\| p50 | p90 | p99 | share over 0.05 m | over 0.10 m |
|---|---|---|---|---|---|---|
| 2 | 93.4 % | 0.000 | 0.020 | 0.875 | 4.6 % | 2.9 % |
| **4** | **87.0 %** | **0.000** | **0.040** | **1.760** | **7.8 %** | **4.5 %** |
| 8 | 71.9 % | 0.015 | 0.075 | 2.255 | 20.0 % | 7.6 % |

The trade is legible: a longer lag gives a moving car more displacement to show, and costs coverage
and accuracy because more of the world has rotated out of the window, more points fail the
out-of-plane test, and the composed ego motion has further to extrapolate.

**τ, fixed before any training.** σ_static is read off the *literal* residual (`--tol-beams 0`, no
envelope, so the distribution has no atom at zero) at k = 4 over the same ten recordings:
**σ_static = 0.025 m** by the p68 of |R|, and 0.022 m by the median-based estimate — the two agree,
and they are the same numbers before and after the attitude change above, so nothing about τ was
chosen after seeing a result.
The contract's rule is 2–3 σ, and the declared value is

> **τ = 3 σ_static = 0.075 m**, with `sign(R)·max(|R| − τ, 0)`.

Soft rather than hard, so a real residual keeps its size (minus τ) instead of arriving as a step
function at the threshold. In the declared configuration — k = 4, τ = 0.075 m, the two-frame
consistency test at ±8 beams — the real-bag floor is:

| | value |
|---|---|
| bins with a prediction | 87.0 % |
| over τ | 5.4 % of those |
| surviving the consistency test | **3.8 %** |
| the survivors' \|R\| | p50 0.235 m, p99 6.15 m |
| their run lengths along the beam axis | p50 5 beams, p90 27 |

That is about 30 flagged beams in a 1081-beam scan, in runs whose median is 5 beams. An opponent at
3–5 m covers 30–80 contiguous beams, so the signal and the floor differ in extent as well as in
magnitude — which is what a convolutional stem is in a position to use. And it is an **upper** bound
on the floor: these recordings have people beside the track and, in some, another car, so an unknown
part of those 3.9 % is real motion that the channel is right to report.

### The noise floor, in simulation

`python -m f1sim.learn.aligned_floor sim`. Solo races — no other car anywhere — with the procedural
obstacle layouts on and the attitude randomisation on, driven by the frozen original so the ego
motion is a policy's and not a fixture's. **Every residual there is error by construction**, which is
the thing the real recordings cannot say. The simulated attitude estimate is harsher than the usable
recordings' — |roll, pitch| rms 11.0 deg against 6.2 — because the plant models the VESC estimate's
drift, which is exactly the defect the five refused recordings exhibit.

Ungated, literal residual (`--tol-beams 0`), by lag:

| k | bins predicted | σ (p68 of \|R\|) | \|R\| p50 | p90 | p99 |
|---|---|---|---|---|---|
| 2 | 80.5 % | 0.030 m | 0.020 | 0.065 | 0.635 |
| **4** | **68.0 %** | **0.045 m** | **0.030** | **0.100** | **1.245** |
| 8 | 48.2 % | 0.075 m | 0.050 | 0.170 | 2.320 |

σ in simulation is about twice the real recordings' 0.025 m, so the declared **τ = 0.075 m is 3 σ on
the bags and 1.7 σ here** — which is the right direction for a threshold that was fixed on the data
the car will actually see, and the reason the sim floor below is the more pessimistic of the two.

The declared configuration (k = 4, τ = 0.075 m, consistency ±8 beams), with the same channel told
the car is level as the control:

| | attitude change used | attitude ignored |
|---|---|---|
| bins predicted | 72.5 % | 91.7 % |
| over τ | 7.9 % | 8.8 % |
| **surviving the consistency test** | **4.9 %** | 6.8 % |
| the survivors' \|R\| | p50 0.055 m, p99 **5.34 m** | p50 0.085 m, p99 **6.83 m** |
| run lengths | p50 2, p90 8 beams | p50 2, p90 8 |
| share of flagged beams in runs ≥ 20 | **20 %** | 23 % |

Two readings worth separating:

* **the tilt handling trades coverage for the tail.** Ignoring the attitude predicts a fifth more
  bins and pays for it in the p99 of the residual (2.38 m vs 1.15 m ungated, 6.83 m vs 5.34 m among
  the survivors) and in a false-positive rate a third higher. The out-of-plane test is discarding
  points the two scan planes genuinely do not share, which is what it is for.
* **the survivors look different from an object.** In simulation, where every one of them is error,
  they come in runs of median 2 and p90 8 beams, and only 20 % of flagged beams sit in a run of 20 or
  more. On the real recordings the same configuration gives runs of median 5 and p90 27, with 57 % in
  runs of ≥ 20 — and those recordings contain people beside the track and, in some, another car. The
  difference in *extent*, not only in magnitude, is what a convolutional stem is in a position to
  use, and it is also the strongest evidence available here that a good part of the 3.8 % on the bags
  is real motion the channel is right to report.

### The floor beside something to find

A false-positive rate on a scene where the right answer is zero everywhere is half a number. The
other half is the same channel, the same tracks, the same policy, the same threshold, with two
teacher opponents in the race (`work/e2/signal.sh`):

| | solo (the floor) | race size 3 |
|---|---|---|
| bins with a prediction | 72.4 % | 74.5 % |
| over τ | 7.8 % | 8.3 % |
| surviving the consistency test | 5.00 % | 5.49 % |
| flagged beams per scan | 10.0 | 10.9 |
| **median \|residual\| of the survivors** | **0.050 m** | **0.270 m** |
| their p68 / p90 | 0.145 / 1.015 m | 0.730 / 2.430 m |
| run lengths | p50 2, p90 8 | p50 2, p90 9 |

**The discrimination is in magnitude, not in count.** Putting two cars in the scene barely changes
how many beams the channel flags (5.00 % → 5.49 %) or how they are shaped, but it moves the median
survivor from 5 cm to 27 cm and the p90 from 1.0 m to 2.4 m. That is the right shape for a signal
fed to a convolutional stem: the floor is small-magnitude noise scattered in short runs, and a car is
a *large* residual in the same places a small one could have been. It also says the consistency test
and τ are not what separate signal from floor — the value is.

A corollary worth naming: **counting flagged beams is the wrong summary of this channel**, and the
3.8 % / 4.9 % floors above should be read with that in mind. They bound how often the channel says
*something*; they do not bound how often it says something a policy would act on.

**The warp is doing the work it claims.** On a clean recording, with everything else fixed, the share
of beams surviving the gate moves as:

| what the warp is told | survivors |
|---|---|
| the measured speed and yaw rate | **2.0 %** |
| speed scaled ×0.95 | 1.9 % |
| speed scaled ×1.10 | 2.6 % |
| yaw rate ×0.8 / ×1.2 | 4.1 % / 3.6 % |
| the yaw rate zeroed | 26.2 % |
| the yaw rate sign flipped | 40.2 % |
| the speed zeroed | 45.2 % |

Removing the ego compensation multiplies the false-positive rate by 13–22×. The gyro is already
correctly scaled (1.0 is the optimum) and the ERPM speed reads about 5 % high, which is consistent
with `speed_gain` in `docs/real_data_calibration.md`; neither is corrected here — a 5 % speed error
is 9 mm over the warp's 0.9 m of travel and the measurement says so.

### The smoke arms: PPO health

The same smoke the future-head worker ran, one flag apart: 262144 env steps, 63 envs (not 64 — the
trainer refuses an env count that is not a multiple of `--race-size 3`), three tracks, seed 701,
warm-started from `frozen_original_48cc698f`, the opponent-diversity flags of
`work/learning-next/future-20260914/launch.sh`. **Not a performance result**: 131 updates is about a
thirtieth of the shortest finetune that has ever moved a held-out number. What it is evidence for is
that the new inputs and the new losses do not destabilise the update.

Loss columns are means over the window; **the episode columns and the gradient norm are medians**,
because a single update's collisions/km is a ratio over the handful of episodes that ended in it —
reading them as means made one arm look like it had collapsed to 131 collisions/km when its
per-quarter medians never left 15–26. (`work/e2/health.md`.)

| first 20 / last 20 | A: `memory,edges` | B: + aligned rows | E3-a: + motion GRU + mask | E3-b: + current Δv |
|---|---|---|---|---|
| `kl_ref` | 0.061 / 0.125 | 0.052 / 0.132 | 0.057 / 0.131 | 0.062 / 0.121 |
| `clipfrac` | 0.0003 / 0.0002 | 0.0005 / 0.0001 | 0.0006 / 0.0000 | 0.0006 / 0.0001 |
| `grad_norm` (median) | 12.1 / 16.4 | 12.5 / 20.7 | 13.7 / 16.3 | 12.5 / **22.9** |
| `loss/vf` | 3.55 / 3.91 | 3.65 / 4.21 | 3.49 / 3.34 | 3.80 / **4.37** |
| collisions / km (median) | 36.9 / 16.1 | 32.7 / 11.2 | 32.8 / 11.0 | 38.7 / 17.9 |
| progress [m] (median) | 27.2 / 61.9 | 30.6 / 83.9 | 30.5 / 91.2 | 25.9 / 52.2 |
| median env steps/s | 343 | 340 | 310 | 306 |
| wall clock | 16.5 min | 13.8 min | 15.0 min | 14.9 min |

Every arm ran all 131 updates with no non-finite loss and no non-finite gradient norm — with
`--memory` the trainer raises on either, so a finished run *is* that evidence — and every arm
recovers from the warm start the same way. Two things are worth taking:

* **the motion branch costs about 10 % of training throughput** (343 → 306 env steps/s) and the
  aligned channel costs nothing measurable (343 → 340). The channel's cost is in the per-step
  inference budget, not in the rollout; the branch's is in both.
* **E3-b ends with the largest gradient norm and value loss of the four** (22.9 and 4.37 against
  16.4 and 3.91 for the control). That is what two extra loss terms look like and it is not
  instability — the clip fraction and the approximate KL are identical across the arms — but it is
  the only column where the arms are visibly ordered, and the ordering is by how many terms the loss
  has rather than by anything about representations.

Nothing else here distinguishes the arms, which is what 131 updates on one seed was said in advance
to be unable to do.

### What the channel did to the probe

The two smoke checkpoints go back through the E1 table as two more columns, same protocol, same
seeds, same splits (`work/e1/e1.md`). Median R² over the 16 draws, with the earlier arms for scale:

| median R² | `gru_warm` (random GRU) | `gru_trained` (8 upd) | **`e2_raw`** (131 upd) | **`e2_aligned`** (131 upd) |
|---|---|---|---|---|
| **Δv_x** | +0.185 | +0.112 | **+0.188** | **+0.230** |
| **Δv_y** | +0.132 | +0.178 | **+0.139** | **+0.150** |
| Δx | +0.235 | +0.241 | +0.217 | +0.152 |
| Δy | +0.190 | +0.202 | +0.185 | +0.271 |

**Alignment alone did not move Δv.** The gap is +0.042 on Δv_x and +0.011 on Δv_y — inside the ±0.07
that separates `gru_warm` from `gru_trained`, two arms this table already calls indistinguishable.
On the mean rather than the median it is +0.005. The threshold was written down before these columns
existed (`work/e3/decide.md`), so this is the rule firing and not a line drawn afterwards.

Two things belong beside that, and neither of them rescues it:

* **The channel arrives as zero-initialised input columns.** At update 1 the network ignores them by
  construction — that is what makes the warm start bit-identical — and it has 131 updates to learn
  to use three new rows. A null here is strong evidence about the *budget* and weak evidence about
  the idea. It is also precisely the argument for E3: a loss that asks a branch for the opponent's
  motion does not wait for PPO to discover that an input is worth reading.
* **`e2_raw` lands on top of `gru_warm`** (+0.188 against +0.185). A hundred and thirty-one updates
  of PPO left Δv_x exactly where an untrained recurrence already had it — the same finding the E1
  ladder gave, now with training rather than by construction, and the premise the whole contract is
  testing.

### The second table: the racing proxy

The addendum asks for this separately from the representation, so that *"Δv R² up but racing flat"*
is a reportable outcome rather than an unasked question. `evaluate --protocol rolling` in traffic on
the smoke's own three tracks, one fixed seed, 2400 steps × 96 cars, teacher opponents with all seven
behaviours — and note that `--per-track` is one full evaluation **per track**, so each arm is three
runs pooled by the learner-minutes behind them (`work/e2/proxy.md`).

| | A: `memory,edges` | B: + aligned rows | E3-b: + motion GRU, mask, Δv |
|---|---|---|---|
| collisions / km ↓ | **16.6** | 19.5 | 17.1 |
| wall collisions ↓ | **116** | 186 | 145 |
| car contacts / learner-min ↓ | **3.15** | 3.33 | **3.15** |
| passes / learner-min ↑ | 2.48 | 2.70 | **2.73** |
| mean speed [m/s] ↑ | 4.77 | **4.86** | 4.81 |
| lap time [s] ↓ | 14.5 | **14.0** | 14.3 |
| pace vs the opponents ↑ | 1.361 | **1.377** | 1.336 |
| share of time in contention | 73.9 % | 72.7 % | 74.5 % |

**The two halves of this table disagree, and so does it with the training log.** Arm B is the faster
policy — quicker laps, higher mean speed, more passes held — and it crashes more, mostly into walls
(186 against 116). That is the ordinary speed-for-safety trade and not a statement about
representations. And during training the ordering was the other way round: arm B ended the smoke at
18.5 collisions/km against A's 23.6.

E3-b sits between the two E2 arms on nearly every row, which is what a third draw from the same
distribution looks like.

The honest reading is that **at 262144 env steps the racing numbers are noise**, which is what the
smoke was said in advance not to be able to answer. The table is here because the addendum asks for
it to exist before anyone is tempted to infer racing from a probe, and what it shows is exactly why:
one seed, one protocol, a third arm landing in between, and a direction that flips between the
training log and the evaluation.

E3-a was not proxied. The three rows above already establish what this table can say at this budget,
and a fourth half-hour evaluation would have bought a fourth row of the same; its own smoke's
collisions/km — 11.0 median over the last twenty updates, the best of the four arms — is the evidence
that the motion branch does not break driving.

### A confound in the smoke arms, measured rather than assumed

The two E2 smoke arms differ by one flag, and by one thing nobody asked for: **their fresh GRUs are
differently initialised**. Adding three input columns widens the scan stem's first convolution, which
consumes a different number of draws from the ambient generator, so every module built after it —
including the actor's and the critic's GRU, the only tensors a warm start leaves fresh — gets
different random numbers from the same `--seed`. Checked directly rather than reasoned about: with
`--seed 701` the two arms' `actor.memory.gru.weight_ih_l0` differ by up to 0.176, while every weight
copied from the frozen original is bit-identical.

It does not touch the probe — each arm is probed as itself, and the E1 table's `gru_warm` column
already says what a *random* recurrence is worth on these targets — and at 131 updates it is far
below the noise on any of the health metrics. It does mean the smoke cannot be read as "one flag,
everything else held". The same applies to the memory-policy and future-head smokes' channel arms,
which were built the same way.

The fix, if an arm ever has to carry a result: seed a warm start's fresh tensors from their own
NAMES rather than from the ambient generator, so that a module of the same shape is initialised the
same way whatever was built before it. Not done here — changing the initialisation discipline
between one arm and the next would replace this confound with a worse one.

## Budget

`python -m f1sim.learn.budget`, the same proxy the memory work is held to: CPU, one thread, batch 1,
fp32, the fastest of five blocks of 200 iterations, measured with nothing else on the machine. The
rule is the actor's **forward** within 1.5× the frozen original's and its parameters within 2×; the
step ratio (forward plus the channels the car builds once per scan) is reported beside it.

| variant | actor forward [ms] | channels [ms] | step [ms] | **forward ratio** | step ratio | actor params | **ratio** |
|---|---|---|---|---|---|---|---|
| frozen original | 1.820 | 0.000 | 1.820 | **1.00×** | 1.00× | 1 168 164 | **1.00×** |
| + GRU 128 | 2.011 | 0.000 | 2.011 | 1.10× | 1.10× | 1 398 308 | 1.20× |
| + `memory,edges` | 2.062 | 0.028 | 2.091 | 1.13× | 1.15× | 1 398 980 | 1.20× |
| **E2** + the three aligned rows | 2.025 | 0.658 | 2.683 | **1.11×** | 1.47× | 1 399 988 | **1.20×** |
| **E3** + the motion branch | 2.351 | 0.669 | 3.020 | **1.29×** | 1.66× | 1 447 500 | **1.24×** |

Both rules pass for both experiments, with room. In absolute terms the whole network part of a
control step is **3.0 ms of the 25 ms the car has** even with the motion branch, and the three
aligned rows are 0.66 ms of that.

Two honest qualifications:

* **the channel is operator-launch bound at batch 1.** Its arithmetic is about 2 M multiply-accumulates
  — the warp is one 3×3 matrix applied to 1081 beams, two scatters and a handful of pools — so
  0.66 ms is this desktop's Python and dispatch overhead rather than work that will shrink on a
  faster core or grow on a slower one in proportion. If the Jetson proves tight the remedy is fusing
  the channel, not shrinking it, and that is a measurement nobody here has made.
* **the step ratio, not the forward ratio, is what the channel moves** (1.15× → 1.47×). The rule is
  stated on the forward and the forward barely moves, which is true and slightly flattering; the
  number a deployment should plan against is the 3.0 ms.

In training the picture is the other way round: the aligned channel costs nothing measurable
(343 → 340 env steps/s at 63 envs) and the motion branch costs about 10 % (343 → 306).

## E3 — a second state, and the auxiliaries that are allowed to shape it

### Why the state is split rather than asked for more

The E1 table says the recurrent state carries the ego's own motion at R² 0.9 and the opponent's
relative velocity at 0.1–0.2. Handing that same GRU an extra input and an extra loss would not
change the incentive that produced the asymmetry: whatever the loss, ego dynamics are the cheapest
way to reduce it, and the state would go on spending itself on them. So the recurrence is split.

```
current LiDAR ------ scan stem --------------> main GRU (128) ---+
aligned rows ------- motion encoder --------> motion GRU (h_dyn <= 64) ---+---> plan head
                          |                        |
                          +-- per-beam mask        +-- current Dv, and the 0.5 s future head
```

Both enter the plan head's first preactivation through their own bias-free, zero-initialised
projection, which is the construction the conditioning and the memory already use and buys the same
two things: at step 0 the actor is bit-identical to the checkpoint it was warm-started from, and the
new path's gradient is nonzero from the first update.

`h_dyn` is carried **inside the same hidden tensor** as the main state, `[h_main | h_dyn]`. That is
not a detail: the hidden state is written into the rollout buffers, masked by the truncation
boundary, carried across the ROS node's scan callbacks, copied into the viewer's static CUDA-graph
buffer and exported as ONNX's `hidden` input. Every one of those is written against a width and none
of them has to learn that there are now two states.

### The auxiliaries attach to the motion branch and to nothing else

This is the addendum's requirement and it is the property the whole split exists for, so it is
tested as a fact about the autograd graph rather than asserted in a comment: back-propagate from the
two auxiliary losses and from the future head, and the set of parameters that received a gradient
must be exactly the motion encoder, the motion GRU and the heads themselves — not the scan stem, not
the main GRU, not the MLP, not the action head, not the critic.

| stage | flag | loss | label |
|---|---|---|---|
| E3-a | `--aux-opp-mask` | weighted BCE on a per-beam "is this beam on another car" logit read from the motion **encoder**'s features | the LiDAR's own `scan_type == HIT_CAR`, which the simulator has classified since cars were cast as meshes — privileged, train-time, and already there |
| E3-b | `--aux-motion` | MSE on the nearest opponent's **current** relative velocity, read from `h_dyn` | the k = 0 row of the same privileged snapshot the future head uses, presence-masked |
| E3-c | `--aux-future` | the existing 0.5 s head, now reading `h_dyn` | unchanged |

Two details that would otherwise be invisible. The mask's positive weight is the reciprocal of the
batch's own positive rate, **clamped at 50**: a car covers a few percent of the beams when it is
there at all and none when it is not, so an unclamped weight would let one frame with a single
car-hit beam dominate an update; the loss reports the rate it saw and the weight it used, plus recall
and precision, because a BCE that falls while the head answers "no car" everywhere is the failure
this label invites. And the mask logits are **train-time only**: neither head is called by the
actor's forward, so the traced ONNX graph cannot contain them, while the motion branch itself is
exported because it is part of the policy.

## E4 — scope only

The question E4 asks is whether this representation can be built **without privileged labels**. The
candidate is a per-beam, self-supervised target with a label on every step:

    y_t(θ) = L_{t+k}(θ) − static_render(pose_{t+k}, att_{t+k})(θ)

— the scan k steps from now, minus what the static world alone would have returned from the same
pose and the same roll/pitch with the other cars removed. What is left is everything that moved by
itself, at full beam resolution, and unlike the beam mask it needs no `scan_type` and no privileged
vector: it is a difference of two range images the simulator can produce for any scene.

What it needs at train time is **one extra LiDAR cast per env step** — the same `Lidar.scan` the step
already runs, with `cars=None`, at the pose and attitude the step just produced, kept in a k-deep
ring so that step t's label is available at step t + k. `work/e4/cost.py` measures exactly that and
nothing else, in the configuration a run would use:

| 63 cars, race size 3, compiled backend, CUDA | env steps/s |
|---|---|
| the simulator as it is | 382 |
| + one static LiDAR cast per step | 375 |

**1.02× the simulator's time**, 3.06 ms per batched step. That is the whole cost, and it is the
number that makes E4 worth putting in front of the other two: a label on **every beam of every step**
for two percent of the rollout, against the auxiliary future head's ~20 % of steps after its
boundary masking, and against a beam mask that needs the simulator's privileged `scan_type`.

It is not free of assumptions: on the car the same label would need a map and a pose, so it is a
training-time target either way. What it buys over the E3 auxiliaries is scale — every beam of every
step is labelled, against the ~20 % of steps the future head can label after its boundary masking —
and independence from the simulator's privileged state, which is what makes it the honest answer to
"could this have been learned from data a real car could collect". Not implemented.
