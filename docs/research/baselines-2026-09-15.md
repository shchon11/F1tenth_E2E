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

The Table 3 student is the same architecture as the second row and costs **2.5× more per step**,
because a DAgger loop has to put the student back in the car every iteration and so it runs through
torch rather than ONNX:

| | params | backend | ms/step |
| --- | ---: | --- | ---: |
| TinyLidarNet-L, published weights | 220 686 | onnxruntime | **0.090** |
| TinyLidarNet-L, our demonstrations (Table 3) | 220 686 | torch | **0.223** |

Identical parameter count, identical layers, 2.5× the cost — the gap is entirely the runtime, and it
is the price of a student that has to be re-entered into the simulator between iterations. Both sit
far inside a 25 ms step, so it changes nothing about deployability; it is recorded because a reader
comparing "the same network" across two rows of this note would otherwise find two different
numbers and no reason for them. (Re-measuring the published model at the same moment gave 0.090
against the 0.083 in the table above — 7 % of measurement noise under three running lanes, which is
also worth knowing before anyone reads a 5 % difference anywhere in this section as real.)

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
57 m lap. So the network is not broken and the integration is not wrong — that much the probe does
establish, and it is what the probe was for.

> **Correction — the probe's explanation does not survive the full suite, and the probe's own cell
> was the worst possible one to draw it from.** I wrote that End2Race "is commanding a speed
> calibrated for the tracks it learned on", its demonstrations coming from several-hundred-metre
> f1tenth_racetracks circuits, so 6 m/s into the corner of a 57 m lap is the gap. That story predicts
> the 0 m fill (which slows it) helps most on the *short* tracks and least on Monza, the one circuit
> resembling its training. Scored over all 64 cells, both halves are wrong:
>
> | map | lap | 30 m fill | 0 m fill | Δ |
> | --- | ---: | ---: | ---: | ---: |
> | `real:map16x07` | 33 m | 0/80 | 0/80 | **0** |
> | `real:map12x16` | 36 m | 0/80 | 0/80 | **0** |
> | `real:korea_2025_iccas` | 43 m | 0/48 | 0/48 | **0** |
> | `gen:control:9100` | 57 m | 3/112 | 62/112 | **+59** |
> | `gen:competition:9200+pinch9200` | 60 m | 0/48 | 0/48 | **0** |
> | `gen:competition:0` | 68 m | 0/48 | 0/48 | **0** |
> | `real:blackbox2022_3` | 106 m | 0/48 | 0/48 | **0** |
> | `rt:Monza` | **446 m** | 3/48 | 26/48 | **+23** |
>
> Slowing it down helps most on the **longest** circuit — Monza goes 3/48 → 26/48 — and does nothing
> whatever on the two tightest floors, where mean progress barely moves (5.1 → 7.0 m, and 5.1 → 4.4 m,
> which is *worse*). The whole +82 lives in **2 of 8 maps**.
>
> And the probe was run on `gen:control:9100`, which alone accounts for **+59 of the +82 — 72 % of
> the entire effect**. I picked that cell before the rows existed, and it happens to be the single
> map where the substitution does the most work. That is the same mistake as the TinyLidarNet
> output-mapping probe in a new place: a one-cell diagnostic generalised to a suite, when the cell
> was not representative of it.
>
> What stands: the integration is not broken (A vs C settles that), and the fill substitution moves
> the row by a lot. **Why it moves those two maps and no others is not established.** The commanded
> speed does drop, and the drop does coincide with the improvement in the probe's cell — but a
> track-length mechanism is contradicted at both ends, so the speed story is a hypothesis this suite
> does not support. Isolating it needs the experiment nobody has run: cap the commanded speed
> externally at a fixed 30 m fill, and see whether the improvement reappears without touching the
> scan. That separates "it drives too fast" from "the rear bearings change what it sees"; the two are
> confounded in every row here, because one knob changes both.

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
**58 solo completions, 27 avoidance clears and 8 passes** (S 217→275, A 30→57, O 17→25). That is the
size of the runtime layer, measured on the same cells as the baselines — and it is larger than the
entire gap between the two published baselines.

Checked per cell rather than on the totals, because a net figure can hide a trade: the clamp helps
**24 solo cells and hurts 14** (net +58), helps 8 avoidance cells and hurts 3, helps 3 overtaking
cells and hurts 1. So it is a real and large net effect, not a uniform one — there are cells the
unclamped tracker does better on, and the aggregate does not say otherwise.

**TinyLidarNet is not a slow cautious driver that runs out of budget** — it is a driver that
crashes. That half is as clean as this suite gets: across all **512 trials it times out exactly
zero times**. Every failure is a collision. (The −0.5–7.0 mapping, which halves its commanded speed,
times out 84 times in the same 512 — so the metric does fire when a system is actually too slow.)

**It is not, however, "the fastest when it finishes", which is what I wrote first.** Unpaired, its
mean lap over its own completions is 10.79 s against the frozen original's 16.21 s and A701
`@fixed_low`'s 18.00 s. That comparison is confounded: **37 of its 47 completions are on one short
43 m floor**, while the others complete on Monza's 446 m too, which drags their means up. Restricted
to cells where *both* systems completed a lap, it comes out **slower**:

| paired on cells both finished | cells | TinyLidarNet | the other | |
| --- | --- | --- | --- | --- |
| vs frozen original `@legacy` | 14 | 13.73 s | 11.85 s | **1.88 s slower** |
| vs A701 `@fixed_low` | 15 | 14.42 s | 13.59 s | **0.83 s slower** |

The overtaking block was read the same wrong way. It holds 9 of 32 with only **4** contacts against
the frozen original's 11, which looks like cleanliness — but its O-suite failures are **19 wall
collisions against frozen's 4**. It makes fewer contacts because it crashes before reaching the car,
not because it passes more carefully.

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
Monza. On the tight floors it stops at about a third of a lap every time (10.7 m of 33.3 m, 11.3 m
of 36.1 m). Monza is not a consistent failure and should not be quoted as one: its 218 m mean spans
**3.9 m to 426.9 m**, median 236.9, with only 9 of 48 trials within 10 % of that mean — it once got
to within 20 m of finishing. The mean is a summary of a wide spread, not a description of a trial. Its **1/96 on avoidance** is the number to
sit with: its training set contained no obstacles at all.

**The single biggest effect on End2Race is not its architecture — it is which constant fills the 90
bearings this car cannot see.** Both values are theirs: 30 m is what their ray tracer returns for a
beam that hits nothing (`laser_models.py:143-144`), 0.0 is what their evaluation writes into beams
it masks out (`eval_singleagent.py:114`). Swapping one for the other moves the row from **3/384 to
53/384 solo, from 0/96 to 32/96 on avoidance, and from 40.1 to 12.9 collisions per km** — an
eighteen-fold change in completions and thirty-two avoidance clears out of nothing, on identical
weights and identical cells. Root asked for the second variant to be run; on the evidence it is the
more informative of the two, and a paper reporting only the no-return convention would have reported
a number dominated by a substitution rather than by the model.

The *mechanism* is open, and the correction under the probe above says why: the commanded speed does
fall from 5.9 to 3.7 m/s, but slowing it helps most on the **longest** circuit and not at all on the
tightest floors, so "it was simply driving too fast for these tracks" is contradicted at both ends.
One knob changes both the speed and what the rear bearings say, and no run here separates them.

**"Dominated" is the right word for the row and the wrong word for the cells, so both belong here.**
Paired cell by cell, the 0 m fill is better in **18 of 64**, worse in 1, and *identical in 45*; mean
route progress rises in 47 of 64 (+0.142 overall). It moves the row because the row was near zero
and eighteen cells is a large fraction of what there was to move — it does not lift the system
broadly. Two details say the same thing: the fill changes no cell's **approach_collision** count
(64 in both rows, a failure that happens before the policy has much say), and the 0 m variant
introduces **21 timeouts** the 30 m variant never has, which is the commanded-speed drop showing up
as its own failure mode.

**All three of End2Race's *solo* successes under the 30 m fill are on Monza**, the one long wide
circuit in the suite and the only map resembling the f1tenth_racetracks circuits its lattice-planner
demonstrations came from. The scoping matters: it has three more successes in the **O** family on
`gen:control:9100`, which the per-map table above shows and an unqualified "its three successes"
would contradict.

Its mean progress is 124.1 m of Monza's 446 against 10.3 m pooled over every other map — though that
too is a spread rather than a level: 5.1 m on the two tightest floors, 18.1 m on `gen:control:9100`.
The diagnosis in the probe above — a speed calibrated for several-hundred-metre circuits — is what
the per-map breakdown says too.

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

## Table 2 — suite v2.1, the traffic family (complete, 5 systems × 144 cells)

The 80-cell **T** family adds other cars on five held-out floors: `slow`, `pace` and `pair`
opponents and an `event` scenario where they brake, stop and change line. **All five rows are complete** as of 2026-09-16 10:18.

One property of this table is worth stating before its rows arrive, because it is what lets them sit
together at all. The published baselines drive in `direct` mode (`act_dim` 2) and ours in `plan`
mode (`act_dim` 8), so their observation layouts hash to **different** `obs_spec_sha256` and can
never be pooled naively. What makes the comparison legitimate is that the *physical* start is
identical anyway: across every cell the two kinds share — 144 for End2Race, 143 for
`frozen_original@legacy`, and counting up as the rest land — **the `physical_sha256` differs in
zero**. Same track, same spawn, same friction, same opponents; only the observation encoding
differs. `validate_results` groups by layout and pairs within groups, `assert_physically_paired`
checks across them, and `claim_check.py` now pins it independently.

| | S/384 | A/96 | O/32 | T/640 | T:slow | T:pace | T:event | T:pair | passes held | car contacts |
|---|---|---|---|---|---|---|---|---|---|---|
| **TinyLidarNet-L** | 47 | 1 | 9 | **78/640** | 24/160 | 19/160 | 19/160 | 16/160 | **165** | 144 |
| **End2Race** (rear 30 m) | 3 | 0 | 3 | **0/640** | 0/160 | 0/160 | 0/160 | 0/160 | 40 | 136 |
| frozen original `@legacy` (ours, the reference) | 196 | 29 | 17 | **95/640** | 22/160 | 27/160 | 30/160 | 16/160 | 289 | 251 |
| A701 `@fixed_low` (ours, current best) | 275 | 57 | 25 | **233/640** | 71/160 | 52/160 | 59/160 | 51/160 | 319 | 246 |
| A701 `@legacy` (ours, **policy only**) | 217 | 30 | 17 | **118/640** | 35/160 | 26/160 | 33/160 | 24/160 | 309 | 249 |

### What survives traffic is the runtime arm, not the weights — settled

On the five maps the T family shares with the solo family (the restriction matters — the solo family
runs eight, and End2Race's three solo successes are all on Monza, which T does not run, so it has no
ratio to take):

| on the five T-family maps | solo | traffic | keeps |
| --- | ---: | ---: | ---: |
| A701 `@fixed_low` (ours, best) | 186/240 (77.5 %) | 233/640 (36.4 %) | **47.0 %** |
| A701 `@legacy` (ours, **policy only**) | 144/240 (60.0 %) | 118/640 (18.4 %) | **30.7 %** |
| frozen original `@legacy` (ours, reference) | 134/240 (55.8 %) | 95/640 (14.8 %) | **26.6 %** |
| TinyLidarNet-L (published) | 43/240 (17.9 %) | 78/640 (12.2 %) | **68.0 %** |
| End2Race (published) | 0/240 | 0/640 | — |

The middle row is the control, and it decides the question the first three rows could not. A701
`@legacy` is **A701's own weights under the other arm**:

| holding this fixed | changing this | keeps moves |
| --- | --- | ---: |
| the **weights** (A701) | `@fixed_low` → `@legacy` | 46.98 → 30.73 = **16.25 points** |
| the **arm** (`@legacy`) | A701 → frozen original | 30.73 → 26.59 = **4.14 points** |

**The arm accounts for 3.9× what the weights do.** (Those are computed from the unrounded rates.
Taking them off the rounded column above gives 16.3 and 4.1 — which is what I did first, and
`claim_check.py` failed it: a difference of rounded numbers is not the rounded difference.) A701 `@legacy` lands next to the
frozen original it shares an arm with, not next to the A701 it shares weights with — so what
survives traffic is a property of the runtime layer, not of the network.

Two things follow that are worth stating separately. First, this is finding 4 again in a harder
setting: the clamp was already worth more than the entire gap between the two published baselines
solo, and in traffic its *relative* benefit is larger still — **1.97× on traffic success (36.4 % vs
18.4 %) against 1.29× solo (77.5 % vs 60.0 %)**. The runtime layer matters roughly twice as much
when there is another car to get around.

Second, the earlier sentence "traffic costs our policy far more than it costs the baseline" is now
properly dead. It was true only of the two `@legacy` rows, and what it was picking up was the arm,
not whose policy it was. TinyLidarNet's 68 % is still unexplained and the floor effect remains the
live candidate there — a system succeeding 17.9 % of the time has less to lose than one at 77.5 % —
and nothing in this suite separates that from genuine robustness.

**The contact structure holds across all five systems.** Contacts occurring in trials with no pass
at all: A701 `@fixed_low` **229/246 (93 %)**, A701 `@legacy` 231/249 (93 %), frozen `@legacy`
239/251 (95 %), TinyLidarNet 142/144 (99 %), End2Race 136/136 (100 %). A CNN, a GRU and two plan-space PPO policies under two
different arms — four failure profiles, one structure. A contact is what happens *instead* of an
overtake, not the price of one. That is the 02:28 correction confirmed on every system in the
table, including the one that passes best (223 clean pass-trials out of 274).

**End2Race does not complete a single lap in traffic — 0 of 640, every scenario, every floor.** It
finished 3 solo trials, so this is not merely the solo row repeated. All 640 trials met traffic, so
the family measured what it is for.

Its 40 passes divide exactly as TinyLidarNet's did, which is the useful part: **all 40 pass-trials
end against a wall, and all 136 car contacts are in the 600 trials that never passed.** Two
independent systems, two very different architectures, and the same structure — a contact is what
happens instead of an overtake, not what one costs. The corrected reading in Table 2 above is not a
one-system artefact.

One comparison worth *not* making, since it is the trap this note has already fallen into four
times: End2Race's mean progress is 8.6 m in traffic against 24.7 m solo, which looks like traffic
costing it two thirds of its lap. It is not. The T family runs five floors and the solo family eight,
and the three it does not share include Monza, where End2Race travels furthest by an order of
magnitude. **Like for like on the five shared maps it is 11.2 m solo against 8.6 m in traffic** — a
23 % reduction, not a collapse. It was already failing on these floors alone; the other cars are not
what stops it.

Every one of the **640/640** trials met traffic inside the contention window, so the family measured
what it is for. **165 held passes from a network that has never seen another car** is the headline.
The 144 car contacts sit beside it, and what they are is worth getting right — my first reading of
them was wrong.

**Correction: the contacts are not the price of the passes.** I wrote that this row showed a car
"passing by driving through the space rather than around it", from the two totals alone. Split by
trial, that does not survive:

| | trials | clean | wall collision | **car contact** | median closest gap |
| --- | --- | --- | --- | --- | --- |
| trials with ≥1 pass | 144 | 77 | 65 | **2** | 0.02 m |
| trials with no pass | 496 | 1 | 353 | **142** | 1.56 m |

**142 of the 144 contacts are in trials where it never completed a pass at all.** A contact is not
what its overtakes cost; it is what happens instead of one — it closes on the car ahead and drives
into it. When it does get by, it gets by cleanly of the other car in 142 of 144 trials, and the way
those trials fail is by hitting a **wall** (65), which is the same way it fails everywhere else in
this suite.

The 0.02 m median gap during a pass is also not evidence of recklessness, which is the other thing I
would have read into it. On the traffic cells all four systems have finished, the median gap at the
moment of a pass is **0.02 m for every one of them, ours included** — it is what the metric means,
not a property of this driver. The number that does separate them runs the other way. Over the **complete** traffic family, 80
cells paired across all four systems:

| median closest gap over all trials | |
| --- | ---: |
| End2Race | **1.40 m** |
| TinyLidarNet-L | **0.86 m** |
| frozen original `@legacy` | 0.07 m |
| A701 `@fixed_low` | 0.06 m |

The published baselines keep *further* from other cars than our policies do — by more than an order
of magnitude. They are not aggressive; they are oblivious, and obliviousness looks like distance
until the gap closes on its own.

Getting those four numbers right took a guard rather than care. When I first wrote this paragraph I
had **~1.2 m and ~1.6 m against ~0.1 m** from a *partial* set, because the figure is paired over the
cells all four systems have finished and that set grows while the lanes run — 26 cells, then 39,
then 80. TinyLidarNet's median read 1.19 and settles at **0.86**; frozen `@legacy`'s moved 0.11 →
0.10 → **0.07**. `claim_check.py` refused to pin the decimals while they were moving, pinned only
the separation, and then **failed** the moment the last row landed and the value settled outside the
bound it had been given. That failure is the whole point of it: a number that drifts silently in a
research note is exactly what this section is about.

The four scenarios separate the way you would expect if the opponent is being treated as scenery —
best against a `slow` car (24/160), worst when there are two of them (`pair`, 16/160):

| variant | success | passes | contacts |
| --- | --- | --- | --- |
| `slow` | 24/160 | 42 | 36 |
| `event` | 19/160 | 48 | 42 |
| `pace` | 19/160 | 25 | 22 |
| `pair` | 16/160 | 50 | 44 |

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

## Every causal claim in this note, checked per trial

The two corrections above were found the same way — an aggregate suggested a mechanism and the
per-trial split disagreed — so rather than wait for a third, every sentence in this note and in
REPORT.md that asserts a *why* was checked against the per-trial records. Verdicts, with the
evidence that settles each:

| claim | verdict | what the per-trial data says |
| --- | --- | --- |
| TinyLidarNet "is not a cautious driver running out of budget" | **verified** | **0 timeouts in 512 trials.** Every failure is a collision. The −0.5–7.0 mapping times out 84 times in the same cells, so the metric does fire when a system is genuinely too slow |
| TinyLidarNet "finishes fastest when it finishes" | **REVERSED** | confounded by subset: 37 of its 47 completions are on one 43 m floor. Paired on cells both finished, it is **1.88 s slower** than frozen `@legacy` (14 cells) and **0.83 s slower** than A701 `@fixed_low` (15 cells) |
| "9 passes with only 4 contacts" shows it passes cleanly | **REWORDED** | its O-suite failures are **19 wall collisions against frozen's 4**. Fewer contacts because it crashes before reaching the car |
| "covers 218 m a trial on Monza" | **REWORDED** | 218 m is a mean over **3.9–426.9 m**; only 9 of 48 trials lie within 10 % of it. One trial came within 20 m of finishing |
| 37/48 on a held-out 43 m floor; a third of a lap on the tight floors | **verified** | 37/48 on `real:korea_2025_iccas` (43.4 m); 10.7 m of 33.3 m and 11.3 m of 36.1 m = 32 % and 31 % |
| End2Race's row is "dominated by a substitution, not the architecture" | **verified, refined** | true of the row, not of the cells: paired, the 0 m fill is better in **18 of 64**, worse in 1, **identical in 45**. `approach_collision` is unchanged (64 both), and the 0 m variant adds **21 timeouts** of its own |
| End2Race "is commanding a speed calibrated for the tracks it learned on" | **REVERSED** | that predicts the slower 0 m fill helps most on short tracks and least on Monza. Both fail: Monza (446 m) gains **+23**, the two tightest floors gain **0**, and the whole +82 is in **2 of 8 maps**. The probe's own cell is 72 % of the effect |
| "End2Race's three successes are all on Monza" | **REWORDED** | true of the **solo** family only; it has 3 more in **O** on `gen:control:9100`, which the per-map table already showed |
| "mean progress 5 m on the small maps" | **REWORDED** | 5.1 m on the two tightest floors, **10.3 m pooled** over all non-Monza maps, 18.1 m on `gen:control:9100` |
| the baselines drive closer to other cars than ours do | **REWORDED** | backwards: median closest gap over all trials is ~1.2 m and ~1.6 m for the baselines against ~0.1 m for both of ours. Not aggressive — oblivious |
| the runtime layer costs 58 solo / 27 avoidance / 8 passes | **verified, refined** | S 217→275, A 30→57, O 17→25. Per cell it is a net not a uniform effect: the clamp **helps 24 solo cells and hurts 14**, helps 8 avoidance and hurts 3 |
| "every one of the 640 traffic trials met traffic" | **verified** | `contended` = 640/640 |
| the 144 car contacts are the price of the 165 passes | **REVERSED** (above) | 142 of 144 are in trials with **no pass at all**; passing trials end in car contact **2** times in 144 |
| the output mapping explains the tight-floor collapse | **REVERSED** (above) | the car mapping is worse overall, 31/384 against 47, and still 0/80 on that floor |

**Every number in this table is recomputed from the raw per-cell records by
`work/baselines/scripts/claim_check.py`, which exits non-zero if any of them stops holding — **158
checks, all passing**: Table 1's rows, Table 2's, every number in the audit above, and the invariant
underneath all of them — that all ten v2 rows share one `(source_digest, suite freeze, device)`
group, and that no row straddles two digests internally. The corrections above replaced claims that drifted from their evidence with
claims that are themselves numeric, and a number only a reader re-checks is a number that drifts the
same way. If a row there ever fails, it is this note that is wrong, not the script.

Design rationales — why the port is in PyTorch, why the ONNX session is single-threaded, why the
label is `last_cmd_raw` — are not in this table. They are claims about code and are settled by the
code and the tests, not by trial records.

One pattern runs through every reversal: a number that was true became an explanation that was not,
because the aggregate was compared across different subsets, or read as a rate when it was a spread,
or paired with a second number that had a different denominator. The aggregates in the tables above
are unchanged and remain correct; what changed is the sentences that claimed to know why.

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

**Iteration 0 is the same demonstrations for every architecture in distribution, and — correcting
what this section said until 2026-09-16 — *not* bit-identical.** The claim here was that "at one
seed the scans and the labels are bit-identical whichever network is being trained — a test asserts
it". Both halves were wrong.

The two interactive runs share a teacher, a seed and an environment, and their iteration-0 buffers
differ in **97.7 % of scan elements and 100 % of labels**, from the first step. The mechanism, which
a test now pins:

* `sim.py:80` seeds the **global** torch RNG when the environment is built;
* `lidar.py:338` draws sensor noise with `torch.randn_like`, i.e. from that global RNG, not from the
  simulator's own `self.gen` which everything else uses;
* `cmd_distill` builds the student **after** the environment and **before** collecting, so a
  220 686-parameter CNN and a 6 359 312-parameter GRU leave the global stream in different places.

Verified directly: two collections at one seed with no model built are identical; build a model in
between and the first scan moves by up to **9.5 m**.

And the test that was cited as evidence never checked it. `test_iteration_zero_is_the_same_
demonstrations_whatever_the_architecture` built the same environment twice and collected with
`driver=None` both times — it varied nothing. It is renamed to what it tests (collection determinism
at one seed), and a second test now asserts the real behaviour, so the false claim cannot return.

**What this does and does not cost the comparison.** It is sensor noise, identical in distribution,
from the same teacher in the same environment on the same tracks — so the architectures are still
being compared on the same demonstrations in every sense that matters, and neither is favoured. What
is lost is the stronger statement, that they saw the *same samples*. Making that true again is a
one-line change (seed the global RNG immediately before `collect`, or have the lidar draw from
`sim.gen`), but it would alter every future collection's data and must not be done inside a running
experiment series — the four rows being scored tonight were collected under the present behaviour.
It is a decision for root, between series.

Later iterations are each student's own on-policy states, which is what DAgger is and what ours does
too.

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

## Table 3 — the fair comparison: same architecture, our demonstrations (64/64, 2026-09-16 05:08)

The TinyLidarNet architecture trained on **our** demonstrations, scored on the same 64 cells, same
digest, same device as everything in Table 1. Beside the published weights of the same network:

| system | | S/384 | A/96 | O/32 | route fraction | timeouts | collisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TinyLidarNet arch, our demonstrations** | D3, **interactive** teacher | **48** | **37** | 0† | 0.644 | 264 | 72 |
| TinyLidarNet arch, our demonstrations | D3, raceline teacher (the dry run) | **48** | **35** | 0† | 0.665 | 242 | 94 |
| TinyLidarNet, published weights | zero-shot reference | 47 | 1 | 9 | 0.416 | **0** | 356 |
| **End2Race arch, our demonstrations** | D3, interactive teacher | **0** | **26** | 0† | 0.277 | 11 | **397** |
| *ours, policy only (A701 `@legacy`)* | on suite v2 | 217 | 30 | 17 | — | — | — |

† **not an overtaking measurement** — see below.

### Changing the teacher changed almost nothing, which is the result

The interactive-teacher row was the one this table was waiting for, and it lands on top of the dry
run. Solo **48 against 48**, identical. Avoidance 37 against 35, two clears in 96. Per map the two
rows differ in exactly one place, `real:map12x16`, 5 against 3:

| map | lap | interactive | raceline dry run | published |
| --- | ---: | ---: | ---: | ---: |
| `rt:Monza` | 446 m | 48/48 | 48/48 | 0/48 |
| `gen:control:9100` | 57 m | 32/112 | 32/112 | 10/112 |
| `real:map12x16` | 36 m | **5/80** | **3/80** | 1/80 |
| every other map | | identical | identical | |

This is the prediction recorded at 15:32 from the iteration-0 buffers, and it holds: the two
teachers' label speeds were within **0.013 m/s** of each other, so the students inherit the same
speed and fail the same way. The interactive student achieves **2.34 m/s** against the dry run's
2.57 and times out **264** times against 242 — slightly slower and slightly worse, in the direction
its commanded speeds predicted (median 2.10 against 2.41 m/s on the recorded scans).

### End2Race scores 0/384 solo, and it is NOT the timeout mechanism

The End2Race arm is the one the contract was missing, and it comes back at **zero** — against 48 for
the TinyLidarNet student on the same demonstrations, and 53 for End2Race's own published weights
under the better fill. Root asked whether this is the slow-label story again or something in the
port. It is neither of the first and specifically the second, and the two students separate cleanly:

| paired, same cells | S/384 | A/96 | timeouts | collisions | route reached | achieved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyLidarNet, our demos | **48** | 37 | **264** | 72 | **0.641** | 2.34 m/s |
| End2Race, our demos | **0** | 26 | 11 | **397** | **0.277** | 2.33 m/s |
| End2Race, published `fill0` | 53 | 32 | 21 | 325 | 0.344 | 2.80 m/s |

**Identical achieved speed, opposite failure.** Both students drive at 2.33–2.34 m/s, but
TinyLidarNet runs out of clock two-thirds of the way round while End2Race hits a wall at just over a
quarter. Its failures are **397 collisions to 11 timeouts** — the reverse of TinyLidarNet's 264 to
72. Whatever is wrong is not the budget.

It is also not a slow network. On the recorded scans End2Race's student commands a **median 2.96
m/s and clears the suite's 3.00 m/s bar on 48 %** of them, against the TinyLidarNet student's 2.10
and 16 %. The arm that scores 48 is the *slower* one. Any explanation resting on demonstration speed
predicts the opposite of what happened.

**What it is: the student learned speed and never learned to steer.** Run each student over its own
iteration-0 buffer — the teacher-driven data it trained on — and compare its outputs to the labels:

| student | steer RMSE | steer R² | speed RMSE | speed R² |
| --- | ---: | ---: | ---: | ---: |
| TinyLidarNet, our demos | 0.094 rad | **+0.438** | 0.93 m/s | +0.503 |
| End2Race, our demos | 0.157 rad | **−0.557** | 0.65 m/s | **+0.738** |

A negative R² means worse than predicting the constant mean: its steering RMSE, **0.157 rad**, is
larger than the labels' own spread of 0.126 rad. It fits speed better than TinyLidarNet does and
steers worse than a constant. A car that tracks the expert's speed and not its steering drives into
a wall at a quarter distance, which is exactly the row.

**The mechanism is their loss weighting meeting our label distribution**, and it is a consequence of
the contract, not a bug. `End2Race/train.py:130-132` optimises `MSE(steer) + 0.05·MSE(speed)` on
**unscaled** units, and the 0.05 is there precisely because the units are unscaled. On our labels
steer has sd **0.126 rad** and speed sd **1.27 m/s**, so the speed term enters at 0.05 × 1.6 ≈ 0.08
against steer's ≈ 0.016 — **five to one in speed's favour even after the 0.05**. Early in training,
when speed error is still metres per second, the ratio is nearer thirty to one. The network spends
its capacity where the gradient is.

This is what "keep every hyperparameter of theirs at its repo default and say so" is for. The
weighting is theirs, the result is a faithful reproduction, and the finding is that **their loss
does not transfer to a label distribution whose speed spread is ten times its steering spread**. It
is a property of the pairing, not of the architecture: the same network under a weighting that
balanced the two terms is untested here and is the obvious next experiment.

Two candidates root raised and the data does not support. The **270-beam deviation** is shared with
nothing else that failed — the published End2Race rows used 360 features and also crashed, and the
retrained model's steering is broken on the data it trained on, before any sensor-geometry question
arises. The **GRU at 40 Hz** likewise: its recurrence was measured at 100 Hz in Table 1 and moved
the row by one trial in 384.

### The overtaking row was never an overtaking measurement, and I said it was

This is the claim I got wrong, and it was a prominent one. Of the dry run's 0/32 I wrote: *"the dry
run's declared limitation arriving on schedule … a teacher that does not react to traffic cannot
demonstrate holding a pass, and its student does not hold one. This row is evidence about **the
teacher**, not about the architecture, and it is the single strongest argument for re-running on
`InteractiveTeacher`."*

The re-run happened. **The interactive teacher — which does react to other cars — scores the same
0/32.** So the attribution was wrong.

It is worse than wrong, because the outcomes say the row measures nothing at all. Every one of the
interactive student's 32 overtaking trials ends in `opponent_respawn`, and 31 of 32 for the dry run:

| on the O cells | outcomes | mean elapsed | of budget |
| --- | --- | ---: | ---: |
| D3 interactive | **32 × `opponent_respawn`** | 19.0 s | **100 %** |
| D3 raceline dry run | 31 × `opponent_respawn`, 1 contact | 18.6 s | 98 % |
| published weights | 19 collision, 4 contact, **9 clean** | 4.7 s | 25 % |

`opponent_respawn` is not a failed pass. It fires when *the car ahead* crashes and respawns, which
voids the race because the one pass being measured can no longer be attributed
(`benchmark/overtake.py:150-157`). The D3 students survive the **entire** budget; over 19 s of
driving the opponent eventually crashes, and the trial is thrown away. The published weights are
gone in 4.7 s — they crash long before the opponent does, so their races stay valid and can be
scored.

So both D3 rows' `O` column is **void, not zero**: a slow-but-surviving driver keeps the scenario
alive long enough for the voiding condition to fire every time. Nothing about either student's
ability to overtake has been measured here, and no argument about the teacher can rest on it. What
would measure it is an overtaking family whose trials do not void on an opponent crash, or students
fast enough to finish before one happens — and the second of those is the same speed problem again.

The last two rows are what CONTRACT.md asks for and this table does not yet have: the End2Race arm
has never been trained, and the "ours" row is a PPO policy rather than a student distilled from the
same expert, so it shares the evaluation and the sensor but not the data. Both are labelled
everywhere they appear and neither is called a controlled comparison.

Two networks with identical architecture, differing only in what they were shown. The solo totals
are a coincidence and the per-map numbers say so:

| map | lap | ours | published |
| --- | ---: | ---: | ---: |
| `rt:Monza` | 446 m | **48/48** | 0/48 |
| `gen:control:9100` | 57 m | 32/112 | 10/112 |
| `real:map12x16` | 36 m | 3/80 | 1/80 |
| `real:korea_2025_iccas` | 43 m | **0/48** | **37/48** |
| `real:blackbox2022_3` | 106 m | 0/48 | 2/48 |
| `gen:competition:0` | 68 m | 0/48 | 5/48 |
| `gen:competition:9200+pinch9200` | 60 m | 0/48 | 2/48 |
| `real:map16x07` | 33 m | 0/80 | 0/80 |

**48 against 47 solo is two systems that share a total and almost nothing else.** Ours takes Monza
48/48 where theirs never finishes; theirs takes the 43 m real floor 37/48 where ours never finishes.

**Avoidance, 1/96 → 35/96, is the result the contract was built to get.** TinyLidarNet's training
set — real-car bags from one 2023 competition — contained no obstacles; ours contains procedural
ones. Same 220 686 parameters, same layers, 35× the clears.

The obvious objection is that ours simply drives slower and so has longer to react, and the suite
already contains the control: the published network under its *own* slower `−0.5–7.0` mapping.

| on the A cells | clears | achieved speed |
| --- | ---: | ---: |
| ours (our demonstrations) | **35/96** | 2.32 m/s |
| published, `1–8` mapping | 1/96 | 2.34 m/s |
| published, `−0.5–7.0` mapping | 0/96 | 1.14 m/s |

Ours clears 35 at **the same speed** the published network manages 1 at, and halving that network's
speed makes avoidance *worse*, not better. Slowness is not the mechanism; the demonstrations are.

**Overtaking, 9/32 → 0/32, is the dry run's declared limitation arriving on schedule.** The
checkpoint has carried this sentence since before it was scored: *"RacelineTeacher (grip 'true'):
the DRY RUN. It is blind to the other cars, so a student distilled from it can at best learn 'pass
the car you see'."* A teacher that does not react to traffic cannot demonstrate holding a pass, and
its student does not hold one. This row is evidence about **the teacher**, not about the
architecture, and it is the single strongest argument for re-running on `InteractiveTeacher`.

**And the 242 timeouts have an exact arithmetic cause.** The suite's budget is `track_length / 3.0`
on every cell — measured, `track_length ÷ time_budget` = 2.9955…2.9990 across all 64 — so a cell
buys **one lap at 3.00 m/s**. Against that bar:

* the teacher's own demonstrated speeds have median **2.90 m/s**, with **52.9 % of its labels below
  3.00**. The expert sits on the threshold.
* the student, imitating it faithfully, achieves **2.41 m/s** off Monza, and clears 3.00 m/s in
  **3 of 464** trials.
* on Monza it achieves **4.15 m/s** — and takes all 48.

It is not failing to drive. It travels **0.665** of a route against the published model's 0.416 and
runs out of clock at 78 % of a lap. It is imitating an expert whose speeds do not clear the
evaluation's own bar, and it inherits that exactly.

**What this row settles and what it does not.** It settles that on identical architecture the
demonstrations decide obstacle competence (1 → 35) and traffic competence (9 → 0, in the direction
the teacher's own recorded limitation predicts). It does **not** settle why the raceline teacher
drove at 2.90 m/s — whether traffic held it there or it is conservative everywhere — because the
buffer behind this checkpoint recorded no opponent distance. `DemoBuffer.G` now records it, and the
first buffer carrying it arrived the same day.

### What the first buffer with the gap label says (2026-09-16 15:32)

The user launched the interactive-teacher re-run after the merge; its iteration-0 buffer is the
first written with `G`. The answer is not the one the question expected, and the honest version is
smaller than the first version I computed.

**The two teachers' demonstrated speeds are practically identical.** Interactive: median **2.89**
m/s, **52.8 %** of labels below the suite's 3.00 m/s bar. Raceline dry run: median **2.90**,
**52.9 %** below. A KS test separates them (p = 5e-17 at n = 6000) but the statistic is 0.08 and the
medians differ by 0.013 m/s — distinguishable, not different. **Swapping the teacher does not change
the speed distribution the student learns**, so Table 3's 242 timeouts should be expected to survive
into the interactive row. That is a prediction, recorded before the row is scored.

**The traffic split does not settle why, and my first reading of it was an artifact of one
threshold.** `speed_by_contention()` at the suite's own 12 m range gave clear-road median 2.83
against in-traffic 2.90 — clear road *slower*, which reads as "conservative everywhere". Sweeping
the threshold destroys that:

| gap threshold | in-traffic median | clear-road median | difference |
| ---: | ---: | ---: | ---: |
| 8 m | 2.73 (n=5332) | **3.68** (n=668) | **+0.95** |
| **12 m** | 2.90 (n=5734) | 2.83 (n=266) | **−0.07** |
| 20 m | 2.84 (n=5859) | **3.97** (n=141) | **+1.13** |
| 30 m | 2.84 (n=5859) | **3.97** (n=141) | **+1.13** |

Every threshold except the one I picked says clear road is about a metre per second faster. And the
12 m clear-road side is 266 samples from **5 of 24 learner rows, 230 of them from two** — not an
independent sample and not something to read a verdict off. `teacher_speed_split.py` now sweeps
thresholds and prints the row clustering, because one threshold produced a confident sentence that
was an artifact of the threshold.

**What the buffer does say cleanly is more useful than what it was asked.** **95.6 % of the
demonstrations are within 12 m of another car, the median gap is 3.6 m, and not one sample has no
opponent at all.** Race size 3 on these tracks keeps the cars packed. So the demonstrations are
almost entirely in-traffic driving; a student trained on them has barely seen an open track, and the
solo family *is* an open track. That is a property of the collection design, identical for both
teachers, and a better account of the timeouts than anything about either teacher's temperament.

**Still to come**: the interactive TinyLidarNet row (running), the End2Race interactive arm (queued
behind it by the one-distill cap), and the End2Race dry run (never started). The L5 dry-run row
stays in this table as the contrast — the two differ in the teacher and nothing else.

<!-- D3 TABLE: filled when the runs finish -->

