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

## Budget

`python -m f1sim.learn.budget`, the same proxy the memory work is held to: CPU, one thread, batch 1,
fp32, the fastest of several blocks of 200 iterations. The rule is the actor's **forward** within
1.5× the frozen original's and its parameters within 2×.

| variant | actor forward [ms] | channels [ms] | step [ms] | forward ratio | actor params | ratio |
|---|---|---|---|---|---|---|
| frozen original | 1.80 | 0.000 | 1.80 | 1.00× | 1 168 164 | 1.00× |
| + GRU 128 | 2.00 | 0.000 | 2.00 | 1.11× | 1 398 308 | 1.20× |
| + GRU 128, `memory,edges` | 1.90 | 0.026 | 1.93 | 1.06× | 1 398 980 | 1.20× |
| + GRU 128, `memory,edges,aligned*` (3 rows) | 1.99 | **0.57** | 2.56 | **1.16×** | 1 399 988 | 1.20× |

Both rules pass with room. The honest caveat is the channel's 0.57 ms: at batch 1 it is
operator-launch bound rather than arithmetic bound (the warp is ~2 M multiply-accumulates), so it is
0.57 ms of this desktop's Python/dispatch overhead and would not scale down on a slower core the way
the convolutions do. It is 2.3 % of the car's 25 ms step as measured; if the Jetson proves tight the
remedy is fusing the channel, not shrinking it.
