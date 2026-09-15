# Published F1TENTH baselines on this car, and a fair comparison (2026-09-15)

Two published end-to-end F1TENTH policies, run as ROS 2 nodes on this project's own link and scored
on its held-out suites; then the same two architectures retrained on the same expert demonstrations
as ours, so that the only thing left between them is the network.

Branch `feat/baselines`, base `0b78111`. Everything below is simulation.

## What was vendored

Read-only, under `external/baselines/`, never staged.

| repo | commit | date | licence |
| --- | --- | --- | --- |
| [CSL-KU/TinyLidarNet](https://github.com/CSL-KU/TinyLidarNet) | `516800231b935673e034156e6c8bfee49045a52c` | 2025-05-15 | **no licence file in the repository.** Used for evaluation and citation only, and noted here because that is what the absence of one means |
| [michigan-traffic-lab/End2Race](https://github.com/michigan-traffic-lab/End2Race) | `c563dd07a8cefc978f88a09def82153469b6879c` | 2025-11-12 | MIT (`LICENSE`) |

## The preprocessing is theirs, quoted, not re-derived

CONTRACT.md: *"Each model's preprocessing is reproduced from its own repo code, not re-derived"*.
Both papers under-specify the two things that decide the numbers — which beams go in, and what the
second output means — so every constant below is a quotation with its file and line, and the tests
**execute the vendored source** rather than restating it:

* `zarrar/tiny_lidarnet.py` is imported with `tensorflow`, `numba` and their `BasePlanner` stubbed,
  so their real `plan()` runs on numpy in a venv that has no TensorFlow. A recording stub stands in
  for the interpreter, and the tensor their code feeds the network is compared with ours **bit for
  bit**, at all three published beam counts.
* `End2Race/eval_singleagent.py` is a script that imports gym, f110_gym, imageio and their lattice
  planner, so it cannot be imported here. Its preprocessing lines are extracted **by content** and
  executed; the test asserts the exact source text is still present, so an upstream change breaks
  the test instead of being silently skipped.

### TinyLidarNet (IROS 2024)

| | |
| --- | --- |
| weights | `Models/f1_tenth_model.h5`, Keras `Sequential`, **220 686 params** — the contract's "220 k" |
| network | `train.py:170-181`: Conv1D(24,10,s4) · Conv1D(36,8,s4) · Conv1D(48,4,s2) · Conv1D(64,3) · Conv1D(64,3) · Flatten · 100 · 50 · 10 · 2, relu throughout, **tanh** on the output |
| input | `inference.py:33` / `train.py:86` / `zarrar/tiny_lidarnet.py:59`: `ranges[::skip_n]`, `skip_n` ∈ {1, 2, 4} for the paper's three sizes. **No normalisation at all**: the network eats metres |
| clip | `zarrar/tiny_lidarnet.py:63`: `scans[scans>10] = 10` |
| steer out | `inference.py:120`: the first output **is** the steering angle in radians |
| speed out | `zarrar/tiny_lidarnet.py:77-79`: `linear_map(out, 0,1, 1,8)` — their simulator. `inference.py:126`: `linear_map(out, 0,1, -0.5,7.0)` — their car. **The two disagree**, so the mapping is a declared parameter and is recorded on every row; the reference rows use the simulator one, because that is the protocol their published simulation numbers come from |
| rate | `inference.py:26`: 40 Hz — the same rate this project's control loop runs at |

The 1081-beam variant needs **no resampling on this car**: 1081 beams over 270° is exactly this
scanner (`params.LidarParams:173-174`) and exactly what their benchmark assumed
(`Benchmark/params/simulator_params.yaml:6-8`). Their simulator config rounds 270° to 4.7 rad, 0.26 %
narrow; that rounding is theirs and is not reproduced, because copying it would resample a real 270°
scan onto a window it does not have.

**One departure, declared.** `zarrar/tiny_lidarnet.py:46-47` adds `N(0, 0.5)` metres of noise to
every beam inside `plan()`. That is a sensor model, not preprocessing: this project's suites declare
their own noise policy per cell (`suite.sensor_noise`, `model_adapter.NOISE_FIELDS`), and applying a
second one inside the driver would make the cell's declared policy false. Not applied;
`describe()["upstream_noise_applied"]` says so and it is on every row.

### End2Race (arXiv 2509.16894)

| | |
| --- | --- |
| weights | `pretrained/end2race.pth`, **11 301 482 params** — not a small network |
| network | `model.py:5-100`: per-beam learnable `k` → pressure token → concat a 60-wide speed embedding → GRU(420→1680) → 1680·420·2. Their `model.py` is **imported**, never transcribed |
| token | `model.py:81`: `(-1/(1+e^{-k·r}) + 1)·2` — 1.0 at 0 m, ~0.01 at 10 m. `k` is fitted per beam: 0.241–1.335 against an initialisation of 0.529 |
| input scan | their fork's `Simulator(num_beams=1440, fov=6.28)` (`f1tenth_gym/.../base_classes.py:459`) — a **360°, 1440-beam** scan, `max_range=30.0` (`laser_models.py:362`) — of which `eval_singleagent.py:104-107` takes 360, 1° apart |
| speed in | `eval_singleagent.py:118,126`: the **previous** step's measured `linear_vels_x`, seeded at `initial_speed * 0.9` |
| outputs | `:121-125`: steer clipped to ±0.52 rad; **speed not clipped at all** |
| rate | `:38,136`: `timestep=0.01` with one `env.step` per inference — **100 Hz** |

Two inconsistencies inside their own repository, both recorded rather than resolved silently:

1. **100 Hz evaluation, 10 Hz training data.** `demonstration.py:213-216` records the training CSVs
   at `sample_interval = 0.1`. A GRU's recurrence is a function of the rate it is run at, so the
   published weights were fitted on 10 Hz transitions and evaluated on 100 Hz ones. This project's
   loop is 40 Hz — between the two. Root's instruction was to measure it rather than argue about it,
   so the pretrained model is scored **both** at our 40 Hz and at a 100 Hz-equivalent ticking (the
   GRU stepped 2.5 times per control step on the held scan, with a fractional accumulator so the
   long-run rate is exactly 100 Hz; the plant stays at 40 Hz).
2. **The speed input.** Training feeds the previous step's *commanded* speed (`train.py:87`, the
   `desired_speed` label); evaluation feeds the *measured* one (`eval_singleagent.py:126`). The
   evaluation convention is what their driver runs here.

## The 360° problem, and what root decided

End2Race reads a full circle. A Hokuyo UST-10LX spans 270°, so **90 of its 360 features have no
measurement behind them on this car** — and a 1-D index map would be worse than useless, because
squeezing 270° of returns into 360° of indices puts every obstacle at a bearing it was not measured
at while looking perfectly healthy downstream.

Root's decision (2026-09-15), implemented exactly:

* **zero-shot rows**: map the 270° scan by *bearing* onto the corresponding 270 of their 360
  bearings, and fill the other 90 with their no-return value — 30 m, what their own ray tracer
  returns for a beam that hits nothing (`laser_models.py:143-144`). Measured, not asserted: the
  mapping fills exactly 90 of the 360, and the test pins that number.
* **a second variant**, because their code contains one other convention for "a beam carrying no
  information": `eval_singleagent.py:114` writes `0.0` into beams it masks out, which the pressure
  token reads as 1.0 — *a wall right here*, the opposite extreme. Scored as its own system.
* **the fair retrained arm** keeps their token, GRU and loss and reads **270 evenly spaced beams**
  (1.0037° apart) of this car's own 270° window, so each learned per-beam `k` still indexes a
  bearing. That is one line of their `model.py` (`num_features = 360` → `270`), rewritten only after
  checking it occurs exactly once, and recorded on the checkpoint.

Nothing about the simulator's LiDAR is changed for any of this: every row is driven by this car's
1081-beam 270° 10 m scanner, which is what makes the fair table a fair table.

## How the weights are executed, and the numbers that say it is the same network

TinyLidarNet's weights are Keras and **TensorFlow is not in the project venv** — installing it there
is not neutral, because it pins protobuf and numpy ranges and that venv is shared with other
workers' running training jobs. So CONTRACT.md's second branch is the one that runs: converted once
with tf2onnx in an isolated venv (`work/baselines/scripts/tln_to_onnx.py`, TF 2.17.1, tf2onnx 1.16.1,
opset 13, **batch dimension left dynamic**), executed with onnxruntime 1.23.2 on the CPU execution
provider, single-threaded.

| check | result |
| --- | --- |
| TensorFlow 2.17.1 vs onnxruntime, **100 real scans from the bags** | max \|Δsteer\| **3.5e-7** rad, max \|Δspeed\| **3.0e-6** m/s |
| onnxruntime batch 1 vs batch 100 | **bit-identical (0.0)** |
| the PyTorch port used for retraining, vs the published Keras weights, same 100 scans | max \|Δ\| **3.6e-7** |

Single-threading is a determinism choice rather than a speed one: the intra-op thread pool changes
reduction order, and the node/adapter parity claim is checked to 1e-5 — close enough to that noise
to be worth removing.

The 100 scans are real: 50 each from two recordings on different floors and different days
(`01_competition_0826-0827/…_v8.7_128s_racepace`, `02_pre-competition/…_v4.5_86s`), evenly spaced
through each, stored as uint16 millimetres because that is what `urg_node` publishes.

The PyTorch port exists because the DAgger loop has to put the student back in the car after every
iteration and a TensorFlow round trip each time is three moving parts where one will do. Getting it
to 3.6e-7 found the one thing that would have made it a different network: Keras `Flatten` on a
channels-last `(B, L, C)` produces index `l·C + c`, PyTorch's native flatten produces `c·L + l`, and
the first dense layer's 1792 weights are laid out for the first. Flattened the other way it scored
0.99 of a ±1 output away — a plausible-looking driver that was reading a transposed world.

## One implementation, two call sites

`f1sim.learn.baselines` holds each model once. `f1sim_ros.baseline_node` drives one car in real time
from `/scan`; `learn.benchmark.model_adapter`'s `external` roster kind drives a batch of them inside
the simulator. Both reach the network through the same `BaselineDriver.command`, so the parity test
between them is about the message plumbing — the miss sentinel, the bearing map, the metres ↔
normalised round trip, the action encoding, the two output clips — and not about two transcriptions
of a paper.

| node ↔ batched adapter, 100 recorded scans | max \|Δsteer\| | max \|Δspeed\| |
| --- | --- | --- |
| TinyLidarNet | 2.1e-7 rad | 1.8e-6 m/s |
| End2Race | 1.5e-7 rad | 1.1e-6 m/s |

The residual is the float32 `range/range_max` round trip the environment's observation does. A
deliberately bent scan moves it by more than 1e-3, so the test is capable of failing.

**It earned its keep before it ever passed.** The first version disagreed by 4.5e-4 rad and 8e-3 m/s
on End2Race: a Hokuyo is usable to 10 m but still emits returns out to ~30 m and the bags contain
them, while the simulator saturates its scan at `cfg.lidar.range_max` before the observation is
built. The node was handing End2Race a 14 m return where the adapter handed it 10, and the GRU
amplified one beam's 4.4 m disagreement into 8 mm/s of commanded speed. `adapt()` now clamps at the
source scanner's declared range first — beyond it there is no measurement to preserve.

## The real-time budget

`python -m f1sim.learn.budget --baselines …`, the protocol the project already quotes: CPU, 1 thread,
batch 1, fp32, the fastest of 5 blocks of 200 iterations. For the baselines this times the **whole**
deployment path — the scan-window mapping and then beam selection, clipping or pressure token,
forward, and output mapping, with any recurrent state carried in and out.

| | params | ms/step | of the 25 ms step |
| --- | ---: | ---: | ---: |
| frozen original, actor forward only (no plan tracker) | 1 168 164 | 1.936 | 7.7 % |
| **TinyLidarNet-L** (1081 beams) | 220 686 | **0.083** | 0.3 % |
| TinyLidarNet-M (541 beams) | ~114 k | 0.047 | 0.2 % |
| **End2Race** | 11 301 482 | **1.853** | 7.4 % |

Two things this table does *not* say. Ours is only the actor: the iLQR plan tracker and any
controller arm are on top, and they are exactly the runtime the fair comparison exists to price. And
this is a desktop CPU, not the Jetson — the ratio is the transferable part, the absolute number is
not.

## Before the table: is End2Race's failure an integration mistake?

The zero-shot End2Race rows read 0 successes, and a number like that has to be separated from a
wiring error before it is reported. Two conventions were checked first and agree between the two
simulators, so neither can be it: beam 0 is the **rightmost** beam in both
(`f1sim/lidar.py:141` against `f1tenth_gym/.../laser_models.py:165-175`), and a positive steering
angle is a **left** turn in both (`psi_dot = v/lwb · tan(delta)`, `dynamic_models.py:120`; measured
on ours as +1.01 rad of yaw over 20 steps of +0.8 normalised steer).

Then the same cell was driven five ways (`work/baselines/scripts/e2r_probe.py`, `gen:control:9100`,
µ = 0.944, 8 cars, a **57 m** lap, 600 steps):

| | per-car progress [m] | survived | mean commanded speed |
| --- | --- | ---: | ---: |
| **A** our 270° scanner, unseen 90° filled at 30 m (the reference row) | 27 38 15 34 34 28 0 29 | 0/8 | 5.88 m/s |
| **B** our 270° scanner, unseen 90° filled at 0 m (their masking value) | **53 50 54 58 52** 6 1 **54** | 0/8 | **3.65 m/s** |
| **C** *their own* scanner: 1440 beams, 360°, 30 m — nothing filled | 26 7 16 34 12 6 0 29 | 0/8 | 5.35 m/s |
| **D** C with sensor noise off, as their gym has it | 26 7 16 34 12 28 0 29 | 0/8 | 5.32 m/s |
| **E** C at their 100 Hz recurrence against our 40 Hz plant | 26 8 16 34 12 6 0 29 | 0/8 | 5.44 m/s |

**C is the answer to the question.** Handing End2Race the exact scanner its repository builds —
1440 beams over a full circle out to 30 m, with no substituted bearings anywhere — does not help;
it is slightly *worse* than the filled 270° version. So the missing quarter of the scan is not what
is wrong, and neither is our sensor noise (D) nor the recurrence rate (E, which moves nothing).

What does move is **B**: told that the bearings behind it are a wall rather than open space, the
model slows from 5.9 m/s to 3.7 m/s and five of eight cars get within a few metres of completing a
57 m lap. The network is not broken and the integration is not wrong — it is commanding a speed
calibrated for the tracks it learned on. Its demonstrations came from a lattice planner on
f1tenth_racetracks circuits (Austin, Hockenheim, MoscowRaceway, Nürburgring), which are several
hundred metres round; six metres per second into a corner of a 57 m lap is the sim-to-sim gap
CONTRACT.md asks to be stated plainly, and this is it, measured.

## Table 1 — the published weights on held-out suite v2, zero-shot

**64 cells, 512 trials a system. CPU, `arm: none`, `direct` action mode, one `source_digest`.**
Every row here was measured today under the same code on the same device; see "Why these rows are
re-measured" below for why the published numbers of our own checkpoints are not reused.

| system | | S/384 | S−bb3/336 | low µ | mid µ | high µ | A/96 | O/32 | coll/km |
|---|---|---|---|---|---|---|---|---|---|
| A701 `@fixed_low` | ours, current best | **275** | 260 | 72/128 | 96/128 | 107/128 | **57** | **25** | **4.12** |
| frozen original `@fixed_low` | ours, the deployment arm | 242 | 237 | 59/128 | 89/128 | 94/128 | 48 | 24 | 5.84 |
| A701 `@legacy` | ours, **policy only** | 217 | 211 | 22/128 | 81/128 | 114/128 | 30 | 17 | 7.03 |
| frozen original `@legacy` | ours, the reference | 196 | 194 | 22/128 | 73/128 | 101/128 | 29 | 17 | 8.47 |
| **End2Race**, rear filled 0 m | published, zero-shot | 53 | 53 | 15/128 | 16/128 | 22/128 | 32 | 3 | 12.94 |
| **TinyLidarNet-L**, speed map 1-8 | published, zero-shot | 47 | 45 | 11/128 | 18/128 | 18/128 | **1** | 9 | 18.02 |
| TinyLidarNet-L, speed map -0.5-7.0 | published, zero-shot | 31 | 31 | 13/128 | 9/128 | 9/128 | 0 | 0 | 18.90 |
| **End2Race**, rear filled 30 m | published, zero-shot | 3 | 3 | 0/128 | 0/128 | 3/128 | **0** | 3 | 40.13 |
| End2Race, 100 Hz recurrence | published, zero-shot | 1 | 1 | 0/128 | 0/128 | 1/128 | 0 | 0 | 45.73 |

Rendered and validated by the benchmark's own reporter, which checks every pin, every cell against
the frozen grid, and the paired start of every row before it will render:
`work/baselines/out/leaderboard-v2.md`, **9 systems, 576 cells**, one source digest, `device: cpu`.

**Their 100 Hz evaluation rate is not what is holding End2Race back.** CONTRACT.md and root both
asked for the rate inconsistency to be measured rather than argued about: their evaluation steps the
GRU every 10 ms (`eval_singleagent.py:38`) while the CSVs it learned from were sampled at 10 Hz
(`demonstration.py:213`), and this project's loop is 40 Hz between the two. Ticking the GRU 2.5
times per control step on the held scan — exactly 100 Hz in the long run, with the plant still at
40 Hz — gives **1/384 against 3/384, and 45.7 collisions per km against 40.1**: marginally worse,
and nowhere near the eighteen-fold swing the fill convention produces. Here the one-cell probe and
the 64-cell row agree; that they did *not* agree for TinyLidarNet's speed mapping is exactly why
this one was run at suite scale too.

**A701 `@legacy` is the "policy only" row CONTRACT.md asks for**, and it is worth reading beside
A701 `@fixed_low`: the same weights, with the plan tracker untouched instead of clamped, lose
**58 solo completions, 27 avoidance clears and 8 passes**. That is the size of the runtime layer,
measured on the same cells as the baselines — and it is larger than the entire gap between the two
published baselines.

**When TinyLidarNet finishes, it finishes fastest.** Its mean lap time over its own completions is
**10.79 s** against the frozen original's 16.21 s and A701 `@fixed_low`'s 18.00 s. It is not a slow
cautious driver that runs out of budget; it is a fast one that crashes. The overtaking block says
the same thing from the other side: 9 passes held with only **4 contacts**, fewer than the frozen
original's 11 at the same arm.

Per map, which is where the two failures stop looking alike:

| map | lap | frozen `@legacy` | A701 `@fixed_low` | TinyLidarNet-L | End2Race |
|---|---|---|---|---|---|
| `real:korea_2025_iccas` | 43 m | 26/48 | 43/48 | **37/48** | 0/48 |
| `real:map12x16` | 36 m | 37/80 | 53/80 | 1/80 | 0/80 |
| `real:map16x07` | 33 m | 29/80 | 51/80 | 0/80 | 0/80 |
| `gen:control:9100` | 57 m | 55/112 | 88/112 | 10/112 | 3/112 |
| `gen:competition:0` | 68 m | 33/48 | 33/48 | 5/48 | 0/48 |
| `gen:competition:9200+pinch9200` | 60 m | 41/48 | 46/48 | 2/48 | 0/48 |
| `real:blackbox2022_3` | 106 m | 2/48 | 15/48 | 2/48 | 0/48 |
| `rt:Monza` | 446 m | 19/48 | 28/48 | 0/48 | **3/48** |

**TinyLidarNet does not fail uniformly — it fails by track.** 37 of 48 on a held-out 43 m floor is
a real transfer result for a 220 k network trained on real-car bags from one 2023 competition, and
0/80 on the two *tighter* real floors (33 and 36 m) is a different thing from the 0/48 on the 446 m
Monza, where it covers 218 m a trial and never finishes. Its **1/96 on avoidance** is the number to
sit with: its training set contained no obstacles at all.

**The single biggest effect on End2Race is not its architecture — it is which constant fills the 90
bearings this car cannot see.** Both values are theirs: 30 m is what their ray tracer returns for a
beam that hits nothing (`laser_models.py:143-144`), 0.0 is what their evaluation writes into beams
it masks out (`eval_singleagent.py:114`). Swapping one for the other moves the row from **3/384 to
53/384 solo, from 0/96 to 32/96 on avoidance, and from 40.1 to 12.9 collisions per km** — an
eighteen-fold change in completions and thirty-two avoidance clears out of nothing, on identical
weights and identical cells. The reason is in the probe above: told the space behind it is a wall
rather than open road, the network commands 3.7 m/s instead of 5.9, and on a 33-68 m lap that is the
difference between finishing and not. Root asked for the second variant to be run; on the evidence it
is the more informative of the two, and a paper reporting only the no-return convention would have
reported a number dominated by a substitution rather than by the model.

**End2Race's three successes under the 30 m fill are all on Monza**, the one long wide circuit in the suite and the only
map resembling the f1tenth_racetracks circuits its lattice-planner demonstrations came from. Its
mean progress is 5 m of a 34–68 m lap on the small maps and 124 m of Monza's 446. The diagnosis in
the probe above — a speed calibrated for several-hundred-metre circuits — is what the per-map
breakdown says too.

### Why these rows are re-measured rather than quoted

`frozen_original@legacy` has a published suite v2 row from 2026-09-12. Same weights, same sha, same
frozen suite, same arm — and it does not reproduce:

| | S/384 | low | mid | high | A/96 | O/32 | coll/km |
|---|---|---|---|---|---|---|---|
| published 2026-09-12, CUDA | 228 | 27 | 91 | 110 | 21 | 23 | 6.85 |
| here 2026-09-15, CPU | 196 | 22 | 73 | 101 | **29** | 17 | 8.47 |

Between the two the simulator gained the recalibrated attitude model, the hard-obstacle patterns and
the opponent-event machinery, and the device changed. 32 fewer solo completions and 8 *more*
avoidance clears is not noise and it is not a regression; it is a different measurement, which is
precisely what `source_digest` and `effective.device` exist to make visible. Putting a baseline row
next to a number from another digest would have produced a difference that belongs to the simulator
and reported it as a difference between policies.

### The output mapping does NOT explain TinyLidarNet's tight-floor collapse — a correction

An earlier version of this note claimed it did, on the strength of `scripts/tln_probe.py`. **That
claim was wrong, and the probe was the reason.**

The hypothesis was reasonable: `zarrar/tiny_lidarnet.py:77-79` maps the second output to 1-8 m/s, so
**output 0 is still 1 m/s** and the network has to go negative to ask for less than walking pace,
while their real-car node maps the same weights to -0.5-7.0 (`inference.py:126`) and reaches a stop
at output 1/15. The probe appeared to confirm it: on `real:map16x07` the car mapping took a cell
from 0/8 to 4/8 and halved the commanded speed from 3.01 to 1.59 m/s.

Scored properly as its own system on the whole suite, it does not hold:

| | S/384 | S−bb3 | low µ | mid µ | high µ | A/96 | O/32 | coll/km |
|---|---|---|---|---|---|---|---|---|
| speed map 1-8 (their simulator) | **47** | 45 | 11 | 18 | 18 | **1** | **9** | 18.02 |
| speed map -0.5-7.0 (their car) | 31 | 31 | 13 | 9 | 9 | 0 | 0 | 18.90 |

The car mapping is **worse overall** — it keeps 31 of the 37 completions on `real:korea_2025_iccas`
and scores **zero on every other map**, including the 0/80 on `real:map16x07` it was supposed to
rescue.

**Why the probe was wrong: it gave the model twice the suite's time budget.** The suite derives the
budget from the track and the speed cap — 444 steps for that 33.3 m lap — and `tln_probe.py` ran
900. A mapping that halves the commanded speed needs roughly twice as long to get round, so under a
doubled budget it looks like a fix and under the real one it does not. Read back at the suite's own
444 steps, the same cell shows exactly that: the car mapping's cars reach a consistent 19-21 m of
the 33.3 m lap and time out, where the simulator mapping's spread is 2-32 m. It travels *further on
average* and still completes nothing.

So the honest statement is the narrow one: the two mappings change how fast it drives and where it
gets to, and neither turns the tight floors into completions. What causes the tight-floor failure is
not established here. Both rows are reported because both are theirs and the difference is real;
neither is a fix.

## Table 2 — suite v2.1, the traffic family (in progress)

The 80-cell **T** family adds other cars on five held-out floors: `slow`, `pace` and `pair`
opponents and an `event` scenario where they brake, stop and change line. TinyLidarNet's row is the
first complete one.

| | S/384 | A/96 | O/32 | T/640 | T:slow | T:pace | T:event | T:pair | passes held | car contacts |
|---|---|---|---|---|---|---|---|---|---|---|
| **TinyLidarNet-L** | 47 | 1 | 9 | **78/640** | 24/160 | 19/160 | 19/160 | 16/160 | **165** | 144 |

Every one of the 640 trials met traffic inside the contention window, so the family measured what it
is for. **165 held passes from a network that has never seen another car** is the headline, and the
144 car contacts beside it is the price: it passes by driving through the space rather than around
it. The four scenarios separate the way you would expect if the opponent is being treated as scenery
— best against a `slow` car (24/160), worst when there are two of them (16/160).

### The two suite runs reproduce each other exactly — four systems, 256 cells

v2.1 contains v2's 64 cells unchanged, and every system here is scored on both, in two independent
runs hours apart under two different suite freezes. Across all **256 shared cells**:

| system | what it is | cells | success Δ | per-trial outcome Δ | physical start Δ | max \|Δ progress\| | max \|Δ lap\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TinyLidarNet-L | ONNX CNN, direct action | 64 | 0 | 0 | 0 | 0.00e+00 | 0.00e+00 |
| End2Race | torch **GRU**, direct action | 64 | 0 | 0 | 0 | 0.00e+00 | 0.00e+00 |
| frozen original `@legacy` | ours, plan + iLQR | 64 | 0 | 0 | 0 | 0.00e+00 | 0.00e+00 |
| A701 `@fixed_low` | ours, plan + iLQR + clamp | 64 | 0 | 0 | 0 | 0.00e+00 | 0.00e+00 |

Not "agree to within a tolerance" — **bit-identical**, down to every trial's distance along the
route and every completed lap time.

The spread matters more than the zeroes. This is not one lucky system: it is a 220 k-parameter CNN
through an ONNX runtime, an 11 M-parameter **recurrent** network through torch carrying hidden state
across every step of every trial, and two plan-space PPO policies under two different controller
arms. The recurrent one is the case that could plausibly have drifted — a GRU accumulates state, so
a single differing float in the first step of a 444-step trial has 443 steps to grow — and it did
not, in 64 cells.

That is the determinism the whole paired design rests on, demonstrated rather than assumed across
the full variety of systems in these tables. It is also why the earlier `frozen_original@legacy`
disagreement with its *2026-09-12* row cannot be waved away as run-to-run noise: under one digest on
one device, in these tables, there is no run-to-run noise to appeal to.

(Only the 64 shared cells can be compared this way; the 80-cell traffic family exists in v2.1 alone.
The rows above are counted from the two independent cell files on disk, not from a summary.)

## The fair comparison: what is held fixed, and what is deliberately not

The zero-shot rows above measure four differences at once — expert, data, track set, sensor — and
none of them is the architecture. End2Race is an imitation framework and so is ours, so the
comparison the paper needs is the one where everything except the network is the same.

**Held fixed**

| | |
| --- | --- |
| the expert | worker 17's `InteractiveTeacher` (or `RacelineTeacher`, declared, until that branch merges) |
| the demonstrations | one collection loop, in the environment worker 17's D3 collects in: the training track set, race size 3, teacher opponents at 0.6–1.15×, all seven scripted and reactive behaviours at 1.0 per 10 s, procedural obstacles from the track list |
| the label | the teacher's **tracked command** — its plan through the same iLQR tracker the car runs, read back as `env.last_cmd_raw` |
| the DAgger schedule | 8 iterations × 250 steps, β₀ 0.6 halving, 3 epochs over the aggregate, last 4 iterations kept, seed 701, cap 9.0 m/s |
| the sensor | this car's 1081-beam 270° 10 m scan, for every architecture |
| the evaluation | suite v2, v2.1 family T, and the traffic proxy at two seeds — the same cells, the same seeds |

**Deliberately not held fixed: the loss and the optimiser.** Each architecture trains with its own
repository's, at its defaults — `Adam(5e-5)` and Keras `huber` at batch 64 for TinyLidarNet
(`train.py:58-61,187`); `Adam(1e-3)`, MSE weighted `steer + 0.05·speed`, batch 16 sequences,
`ReduceLROnPlateau(0.5, patience 10)`, gradient clip 1.0, `mask_prob 0.1`, `hidden_scale 4` for
End2Race (`train.py:28-30,132-135,183-185`). "Their architecture under our recipe" would be a third
system that is neither theirs nor ours. `distill.HYPERPARAMETERS` records which side every number
came from and is written into every checkpoint.

**Iteration 0 is literally the same data for every architecture.** The teacher drives at β = 1, so
at one seed the scans and the labels are bit-identical whichever network is being trained — a test
asserts it. Later iterations are each student's own on-policy states, which is what DAgger is and
what ours does too.

### The four declared deviations

1. **End2Race reads 270 beams, not 360.** Root's decision: one per degree over this car's own 270°
   window, so each learned per-beam `k` still indexes a bearing. It is one line of their `model.py`
   (`num_features = 360` → `270`), rewritten only after checking that line occurs exactly once, and
   the diff is recorded on the checkpoint.
2. **End2Race's speed input is the previous step's *measured* speed.** Their evaluation feeds that
   (`eval_singleagent.py:126`) and their training feeds the previous step's *commanded* speed
   instead (`train.py:87`). The two disagree in their own repository; training on the one their
   driver never sees would fit a network to an input that does not exist at test time.
3. **TinyLidarNet's speed scaling uses this data's own min and max**, which is their rule
   (`train.py:135`, `min_speed` hard-coded 0) applied to our labels — not their published constants
   (1, 8) or (−0.5, 7.0), which are what *their* data's range was replaced by. Carried as a running
   maximum, because a DAgger aggregate grows and drops old iterations while their dataset is fixed.
4. **TinyLidarNet trains in PyTorch**, because the DAgger loop has to put the student back in the
   car after every iteration. It is their architecture layer for layer, and it is checked against
   the published Keras weights before it is trained on anything: **max |Δ| 3.6e-7** on the 100 real
   scans.

### The label, and why it is the *raw* tracker output

`last_cmd_raw` and not `last_cmd`. `last_cmd` is the same command with **this** car's randomised
servo offset and speed gain divided out, so that the plant delivers what the tracker intended
(`gym_env.py:1063`). Two cars looking at the same scan therefore have different `last_cmd` and the
same `last_cmd_raw`, and a LiDAR-only network cannot see which car it is in — a `last_cmd` label
asks it to predict an unobservable per-vehicle constant, and it can only fit the mean. It is also
what this project's own direct-mode labels have always been (`gym_env.teacher_label` at
`act_dim == 2` applies no calibration). The consequence is stated rather than hidden: a
direct-output policy commands the intended angle and the actuator delivers it through its own gain,
while a plan policy has that gain cancelled by the tracker. That is one of the runtime layers this
comparison is about, not an accident of the label.

### What this table can and cannot settle, with the tree as it is today

`feat/interactive-teacher` has not merged, so two of the four rows the contract asks for are not the
rows it asks for, and saying which is which matters more than the numbers:

| contract's row | here |
| --- | --- |
| End2Race architecture, our teacher's demonstrations | **as asked**, with the teacher named on the checkpoint |
| TinyLidarNet architecture, same demonstrations | **as asked**, same |
| ours: worker 17's D3 LiDAR-only student | **not available** — it is on the unmerged branch and had not started training. A701 stands in |
| "ours, policy only" (legacy arm, no clamp) | A701 `@legacy` — the same weights with the plan tracker untouched |

So the comparison this table *does* settle cleanly is **architecture against architecture**:
TinyLidarNet's 1-D CNN and End2Race's pressure-token GRU, on bit-identical iteration-0
demonstrations from one expert, through one sensor, under one evaluation, each with its own loss.
The comparison against *ours* is weaker than the contract intends, because A701 is a PPO policy and
not a student distilled from the same expert — it shares the evaluation and the sensor but not the
data. It is labelled that way in every table and it is not called a controlled comparison.

What closes the gap, in order: (1) `feat/interactive-teacher` merges, (2) re-run the collection with
`--teacher interactive` — one flag, nothing else changes — and (3) put worker 17's own D3 student in
the "ours" row.

**(3) is now an *identical-data* comparison and not merely an identical-protocol one.**
`DemoBuffer` carries the teacher's **plan** action beside its tracked command, so a plan-space
student can be trained from the very same buffer instead of from a re-collection at the same seed.
Nothing in `distill.py` reads it; it costs 8 fp32 against 1081 fp16, about 1.5% of the buffer, and
it was added the moment no collection was in flight to be disturbed. Two things about it are worth
stating because they are the ways it could go wrong quietly: `finalize()` **refuses** a plan label
present on some steps and absent on others rather than stacking a ragged list — that would shift
`P` against `S`/`V`/`L` by however many steps were missed, and nothing downstream would notice —
and the npz key is optional on load, so the iteration-0 dumps already written stay readable. Both
are asserted. `distill.py` is deliberately outside `RUNTIME_MODULES`, so this change cannot alter
`source_digest` or split a scoring protocol.

### The TinyLidarNet arm is trained (2026-09-16 01:37)

`tinylidarnet_raceline_dry_s701` completed all 8 DAgger iterations on CPU at `nice 19`, one thread.

| iter | β | aggregate | loss | elapsed |
| --- | --- | --- | --- | --- |
| 0 | 1.000 | 6 000 | 0.00753 | 48.7 min |
| 1 | 0.600 | 12 000 | 0.00419 | 200.5 min |
| 2 | 0.300 | 18 000 | 0.00322 | 224.8 min |
| 3 | 0.150 | 24 000 | 0.00267 | 248.6 min |
| 4 | 0.075 | 24 000 | 0.00212 | 271.9 min |
| 5 | 0.037 | 24 000 | 0.00189 | 295.0 min |
| 6 | 0.019 | 24 000 | 0.00193 | 319.6 min |
| 7 | 0.009 | 24 000 | **0.00183** | 347.3 min |

Three things in that table are worth reading rather than skipping. The aggregate stops growing at
24 000 because `keep_iters 4` drops the oldest iteration's 6 000 samples — the sliding window is
worker 17's setting, not a limit hit by accident. The loss flattens from iteration 5 and ticks *up*
at 6, which is what DAgger looks like once the student's own states stop being new: the last three
iterations are fitting essentially one distribution. And **iteration 1 took 152 minutes against
~24 for every later one** — that iteration spans the 20:41–22:42 full stop, when every job of mine
was SIGSTOPped by the user's order. It is wall-clock, not compute, and it changes nothing about the
weights.

The checkpoint carries what the row is allowed to claim: teacher `raceline` with the note *"blind to
the other cars, so a student distilled from it can at best learn 'pass the car you see'"*, 264
tracks, 72 cars, 24 learners, race size 3, seed 701, cap 9.0, and `matches:
work/interactive-teacher/work/d3_dagger.sh`. It is pinned immutably at iteration 7 —
`tinylidarnet_raceline_dry_s701@none`, sha `35ace2554b6b`, the `_it7.pt` file rather than the
rewritten `_final.pt`.

It loads through the same `BaselineDriver` the ROS node uses: backend `torch`, 220 686 parameters,
`speed_map: fitted [0.00, 7.61]` — deviation 3 above, their `train.py:135` rule applied to our
labels. On the 100 recorded bag scans it asks for −0.253…+0.468 rad and up to 4.40 m/s, i.e. it
behaves like a driver and not like a saturated network.

**The retrained arm's parity is ≤1e-5, not bit-identical, and that is a backend property.** The
published TinyLidarNet runs through ONNX on a single-threaded CPU execution provider and is
bit-identical whatever the batch width. A DAgger student has to go back into the car after every
iteration, so it runs through torch (deviation 4), and torch selects different convolution kernels
per batch width. Measured on this checkpoint, one row's command against the same row inside batches
of 1, 2, 4, 8 and 16:

| width | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| max \|Δ\| vs width 8 | 7.2e-7 | 4.8e-7 | 0 | 0 | 1.5e-8 |

Two consequences, and only the first is a caveat. The node↔adapter parity bar for a torch-backed
system is the contract's **≤1e-5** branch rather than its "bit-identical" one — comfortably met, at
7e-7, but it should be claimed as what it is. It does **not** weaken the exact-reproducibility
result: the suite runs a cell at one fixed batch width, and at a fixed width this backend is
bit-identical run to run (verified, max \|Δ\| exactly 0.0). Reproducing a row means re-running the
same cells, not re-running them at a different width.

**It has not been scored yet, so there is no row here.** Scoring it needs one benchmark lane and all
three are committed to the v2.1 traffic rows; the End2Race arm has not been started at all. Both
commands are recorded in `work/baselines/STATUS.md` under "Blocked, needs root". Until those run,
the honest statement of this section is that the pipeline is demonstrated end to end for one of the
two architectures and the comparison itself is unmeasured.

<!-- D3 TABLE: filled when the runs finish -->

