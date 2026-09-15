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

<!-- TABLES: filled when the suite finishes -->

