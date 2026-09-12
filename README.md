# F1TENTH_E2E

An end-to-end driving policy for 1/10-scale autonomous racing, trained in a batched simulator
written for it. The policy sees LiDAR and its own proprioception — no map, no localisation, no
camera — and emits a short local plan that an iLQR tracker turns into steering and speed. Around it:
a driving console with three pages (drive / train / environment editor), a ROS 2 link, a track
registry with a scenario grammar, and a frozen held-out benchmark.

**Status: simulation results only. No learned policy has driven the real car yet.** Several sensor and
actuator parameters *are* fitted to 22 recordings from the physical car
([calibration](docs/real_data_calibration.md)) and a real-car ROS 2 policy node exists
([ROS 2](docs/ros2.md)), but sim-to-real transfer is untested, every number below was measured in
simulation, and no trained checkpoints ship with this repository.

## See it

Two 25-second clips, each a single rollout rather than a benchmark; the GIFs are 5-second excerpts, so
click either for the full 1920 × 1080 MP4. Maps, checkpoints, per-episode friction, renderer and
hashes: [video provenance](docs/media/PROVENANCE-video.md).

| Simulator | Trained policy |
| --- | --- |
| [![simulator demo preview](docs/media/f1tenth-simulator-demo.gif)](docs/media/f1tenth-simulator-demo.mp4) | [![trained policy demo preview](docs/media/f1tenth-learning-demo.gif)](docs/media/f1tenth-learning-demo.mp4) |
| The **scripted raceline teacher** — the privileged reference driver, **not a learned policy** — threading the duct hoses of `real:korea_2026_competition` at µ 0.921, orange LiDAR returns along the hose it is passing. | A finished run of `cl_main_fixed_low_s501` played back on the `fixed_low` arm, its emitted plan drawn green ahead of the car beside the blue raceline, with the run's own 128-update training history inset. **The car hits a wall at 5.125 s and the episode resets**, marked on screen. |

[![the f1sim driving console](docs/media/f1tenth-visualizer-overview.png)](docs/media/f1tenth-visualizer-overview.png)

<sub>The console on `real:korea_2026_competition`, driven by `cl_origrecipe_legacy_s701` under the
`legacy` arm. Software-rendered on CPU, so its `sim 배속` and `렌더 fps` are llvmpipe figures, **not**
performance numbers ([provenance](docs/media/PROVENANCE-visualizer.md)); captured 2026-09-12, before
the map card became the three groups below, so its left column is the older loader-name list.</sub>

## Run it in five minutes

CPU only, no GPU and no ROS.

```bash
git clone --recurse-submodules https://github.com/shchon11/F1tenth_E2E.git
cd F1tenth_E2E
python3 -m venv .venv && source .venv/bin/activate
pip install -e f1sim
python3 -m f1sim.viewer.console       # same as python3 -m f1sim.learn.watch
```

Pick a map in the left column and press **시작**. That is the whole first run: the defaults are
정방향 / 없음 / 무작위, so **choosing a map is enough to start**
([viewer design](docs/viewer_design.md#the-map-card-2026-09-12)). The console is the one thing that
needs PyQt5, which is in no extra on purpose — provide it in your environment yourself.

As a library, from inside `f1sim/` — the repository root holds a folder that shadows the installed
package ([getting started](docs/getting_started.md#a-directory-shadowing-trap)):

```python
import torch; from f1sim import Track, Config, Simulator
cfg = Config(); cfg.sim.compile = False      # torch.compile is a CUDA path; skip it on CPU
sim = Simulator(Track.generate_random(seed=0), cfg, num_envs=4, device="cpu")
r = sim.step(torch.zeros(4, 2))              # (steer [rad], target speed [m/s]) per env
print(r.scan.shape, r.odom.shape, r.state.shape)     # (4, 1081) (4, 5) (4, 7)
```

### The three pages

| page | what is on it |
| --- | --- |
| **주행** ([design](docs/viewer_design.md)) | a map card listing **base maps** in three groups (학습 / 검증 / 내 환경) over a search box for the whole catalogue, a scenario row under it (방향, 장애물, 시드), a controller-arm picker (*고급 설정 → 플랜 제어기*), a surface-friction control (*구성 → 노면 마찰 μ*, 랜덤 or 고정, appliable mid-session), and the policy I/O strip that puts what the policy was commanded beside what the simulator actually did |
| **학습** ([training](docs/training.md#watching-a-run)) | recipe presets (`원본 레이스 레시피` = the original policy's own conditions, `SGR`, `R10`, custom) that show the exact `python -m f1sim.learn.ppo …` they will run, start it detached, and chart it live from the trainer's own `upd k/N …` log lines; a 지금 돌고 있는 학습 card per running job; *주행 화면에서 보기* to hand a checkpoint to the drive page |
| **환경** ([editor](docs/environment_editor.md)) | paint duct hoses and tall walls or draw them as editable vector paths, place props, import GLB / OBJ / STL meshes as obstacles, save as `scene:<name>` — a catalogue entry the simulator, the trainer and the drive page all load. **랜덤 트랙 생성** builds closed tracks from a recipe of features (straights, chicanes, slaloms, corners, hairpins) at a chosen size and seed, as editable scenes; `python -m f1sim.trackgen` does batches |

## Train, evaluate, benchmark

### Data and action contracts

The policy is distilled from a privileged raceline teacher (DAgger), then refined by PPO with an
asymmetric critic. Observation: **6 stacked 1081-beam scans plus a 366-number proprioceptive vector**
(speed, IMU, VESC roll/pitch, previous normalised actions, speed cap, 20 rows of history) at the
`scan_stack=6, hist_len=20` defaults. Action: **8 numbers** — 6 curvature knots along the next stretch
of travel plus start and end speed targets — tracked by an iLQR on a kinematic bicycle, 12 × 50 ms
([`mpc.py`](f1sim/f1sim/mpc.py), [architecture](docs/architecture.md#plan-tracking)); integrated pose
is excluded on purpose, so the same tracker runs on the car.

### The recipe, and the rule

**The recipe that works is A** — the original race recipe: `--tracks train` (149 variants),
`--race-size 2`, mixed opponents, the aux grip/opponent heads, Adam moments restored, lr 5e-5 → 2e-5,
KL 0.05. **Train under `legacy`, deploy under `fixed_low`**: every finetune trained *under* the grip
clamp lost low-friction completion, avoidance and overtaking while gaining about 0.2 s of lap time —
across three map sets, optimizer states and learning rates — while both policies trained under the
untouched tracker held the reference. PPO learns plans that lean on the clamp it trains against. Full
argv and the three-way test: [recipe-restore](docs/research/recipe-restore-2026-09-12.md).

```bash
python3 -m f1sim.learn.ppo --tracks train ...            # the curated split, as the recipes were measured
python3 -m f1sim.learn.evaluate --tracks heldout ...     # first attempts on maps nothing trained on
python3 -m f1sim.learn.benchmark plan                    # the frozen suite's matrix; loads no checkpoint
```

`ppo` defaults to `--envs 2048` and `--total 100e6`, so running it bare starts a multi-day job on a
large GPU; [training](docs/training.md) gives bounded recipes that finish.

### The suites

| suite | what it asks | scenarios |
| --- | --- | --- |
| **v1** | was the training distribution fitted at all | 34 cells / 272 trials, 3 reused development maps ([benchmark.md](docs/benchmark.md#the-suite-v1-in-distribution)) |
| **v2** | does it hold up on geometry it has never seen | 64 cells / 512 trials, 8 maps outside every training variant, freeze `89805514…`. It changes the maps and *nothing else* — friction levels, seeds, envs, race size, speed cap, lap budget, sensor noise and opponent are identical to v1, so a v1 and a v2 row differ in the scenario and nothing that could explain the difference away ([benchmark.md](docs/benchmark.md#held-out-suite-v2)) |
| **v2.1** | and does it hold up in traffic | v2 byte-for-byte plus an 80-cell traffic family T — slow / pace / event / pair opponents on 5 held-out maps — for 144 cells / 1152 trials. Branch `feat/overtake-suite`, **pending merge** |

**Adoption decisions use v2.**

### The current headline table — suite v2, 2026-09-13

Four complete systems; every figure from
[benchmark-v2-first-2026-09-13.md](docs/research/benchmark-v2-first-2026-09-13.md), raw cells beside it.

| system | solo /384 | solo less bb22-3 /336 | low µ /128 | avoidance /96 | overtaking /32 | coll/km ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_original@legacy` | 228 | 227 | 27 | 21 | 23 | 6.85 |
| `frozen_original@estimated` | 265 | 258 | 71 | 43 | 22 | 5.04 |
| **`cl_origrecipe_legacy_s701@fixed_low`** (A, seed 701) | **287** | **274** | **77** | 55 | **25** | **3.74** |
| `cl_oppdiv_control_s801@fixed_low` (A recipe, seed 801) | 285 | 277 | 75 | **60** | **25** | **3.74** |

**The A recipe generalises**: it beats the frozen original on every aggregate column under either
arm, and its second seed lands within noise of the first. Per map it is not uniform — on
`gen:competition:0` A701 takes 33/48 against the estimated arm's 36/48 — and `real:blackbox2022_3`
scores 1–13 / 48 for *every* system, the user having ruled it an unfair map with dead-end side
branches no training map has, so the second column, which drops it, is the headline solo number while
the map stays in the freeze. On the target floor `real:map16x07`: 39/80 legacy → 62/80 estimated →
64/80 A701 → 66/80 s801. A **memorisation probe** the same evening found trained and novel obstacle
seeds on one map score alike (frozen 21 vs 17 / 64, A701 25 vs 26): what the policy fails on is novel
map *structure*, not novel placement. Lap time is absent on purpose — it is over each system's own
completions, so a system that fails the hard cells looks fast.

### Controller arms

Four arms; the actor's inputs and outputs are identical across all four, so an arm comparison is only
clean with the policy frozen.

| arm | friction the tracker assumes | on the car? |
| --- | --- | --- |
| `legacy` | none — no explicit limit. The training default. | yes |
| `fixed_low` | one conservative constant, µ 0.73423 | yes — **the deployment default** |
| `estimated` | inferred from causal onboard signals by a frozen quantile model | yes, with an estimator |
| `oracle` | the environment's true friction | no: privileged |

On suite v1 with the policy frozen, `fixed_low` scored S 136/144, low-µ 43/48, A 43/64, O 42/64 at
5.58 collisions/km — the safest arm on every stability column, for 0.22 s per lap (~2 %) against
`estimated` and needing no estimator. A `reactive` slip-triggered arm was built and **rejected**: it
delivered the pace but dropped avoidance to 33/64, because obstacle contacts happen head-on before
lateral slip exists, so its detector never fires
([controller-arms](docs/research/controller-arms-2026-09-12.md)). With the policy untouched, the clamp also
lifts low-µ completion on the 64-trial matched grid from 34/64 under `legacy`
([probe](docs/research/2026-09-11-legacy-recipe-probe.md), 25 + 9) to 56/64
([arm evaluation](docs/research/2026-09-11-controller-arm-evaluation.md)).

## Tracks and scenarios

A **track** is a base map; a **scenario** is a track plus what you do to it. Those used to be one
glued-together string; there are two names now and `f1sim.tracks` converts freely. **Nothing was
renamed on disk**: every checkpoint manifest and frozen suite still says
`real:blackbox2022_1+rlobs44~rev` and still means what it always did.

```
<track id> [ @<direction> ] [ #<obstacle>:<seed> ]
  direction  rev 역방향 · mir 거울 · mir+rev
  obstacle   edge 가장자리 · line 주행선 위 · pinch 좁아짐 · props 입체 (modelled solids) · hard 극단
real/bb22-1@rev#line:44      reversed, boxes on the racing line, placement seed 44
real/bb22-1#hard:*           extreme patterns, placement seed drawn at run time
```

The registry offers only the families a given track's loader can carry. Seed `*` means "anywhere": the
console draws one at session start and shows the concrete scenario in its facts strip so a placement
worth keeping can be pinned, while training expands it into `--obstacle-draws N` rasterised variants
from `--seed`. Grammar, the 59-entry catalogue and the conversion table: [tracks.md](docs/tracks.md).

The three groups answer one question — **has the policy seen this floor?** 학습 is the maps a policy
trains on (53 base tracks, 149 variants); 검증 is maps never used for training in any form (8 base
tracks, and **the only generalisation number there is**); 내 환경 is whatever you built in the editor.
`real/korea26`, the venue this car actually raced at, is a *training* venue — 25 obstacle variants of
that floor are trained on, so a clean lap there says a known circuit was memorised. Splits are
generated from one `SplitRule` per track and compared string-for-string against a frozen oracle in
`tests/test_tracks.py`; `heldout_leakage` is a test, not a convention.

**Hard obstacle patterns** (`+hard<seed>`, [`hard_obstacles.py`](f1sim/f1sim/hard_obstacles.py)) are
copied from scenes the user built by hand, because a single box with a metre of room is not what the
policy fails at: gates of boxes across the lane, diagonal barriers, two-sided chicanes, apex blocks,
wedged clusters, scatters of small 0.10–0.22 m objects mid-lane. Each leaves **55–75 % of the lane
open** — what the hand-built scenes leave — and is *proved* passable, by eroding free space by the
car's half-width plus margin and requiring the lane to still connect through.

**Opponent behaviour events** ([training.md](docs/training.md#opponent-behaviour-events)) give
teacher-driven cars scripted behaviour over a speed scale: `brake`, `stop`, `shift` (a lane change or
blocking line), `weave`. Three properties are enforced, not hoped for: unflagged, nothing is drawn and
a run is bit-identical to one from before the feature existed; events only ever *slow* a car, applied
before the follow-gap cap; and the lateral offset is clamped against the track's distance field, so a
0.35 m lane change through a 1.6 m section becomes as much of one as fits. A 32-race demo gave 37
events, **0 opponent wall contacts**, learner contacts 64 → 74 (`work/opponent-events/REPORT.md`).

## ROS 2

Three ways in ([ros2.md](docs/ros2.md)); Humble, built as in [getting started](docs/getting_started.md#ros-2-workspace).

```bash
python -m f1sim.viewer.console                                            # ROS2 연동 on, then 시작
ros2 launch f1sim_ros pure_pursuit.launch.py                              # an external controller + rviz
ros2 launch f1sim_ros f1tenth_stack_sim.launch.py map:=gen:competition:3  # the real car's stack, unmodified
ros2 launch f1sim_ros sim.launch.py                                       # standalone bridge, one env, real time
```

**The console on ROS 2.** *고급 설정 → ROS2 연동* puts the console's session on the ROS graph:
`센서 토픽 발행` publishes car 0's sensors and the whole scene (`/scan`, `/odom`, `/ego_racecar/odom`,
`/sensors/imu`, `/map`, raceline and marker arrays) for a localiser, a planner under test or rviz, and
`센서 발행 + /drive 로 외부 제어` also hands car 0 to whatever publishes `/drive`. Names, frames, stamps
and QoS match the standalone bridge, where `vesc_sim_node.py` speaks the interfaces of `vesc_driver`
and `urg_node` so only the hardware drivers are replaced — one node runs against both, and against the
real car. The distinction that matters throughout: `/odom` is the drifting dead-reckoned estimate the
car actually has, `/ego_racecar/odom` is simulation-only ground truth, and a planner consuming the
latter will not transfer.

**The real-car policy node.** `policy_node.py` subscribes to `/scan`, `/odom` and `/sensors/imu` and
publishes `/drive`, building its observation with the same [`learn/obs.py`](f1sim/f1sim/learn/obs.py)
used in training, so it runs unchanged against real hardware — though nothing here has been tested on a
physical vehicle. It installs the grip-aware limit by default (`controller:=fixed_low`, µ 0.73423);
`estimated` and `reactive` are simulator research arms, refused here.

**Traction guard — pending merge** (branch `feat/real-car-tcs`). The simulator cannot lock or spin a
wheel: `dynamics.py` has no wheel rotation state and the VESC loop closes on the true body speed, so
wheel speed *is* body speed and a slip detector's residual is identically zero. The recordings show
both failures clearly — −40 to −143 m/s² of wheel deceleration under brake lock against −3 to −16 of
body deceleration, plus launch spin — so the guard sits on the car between policy and VESC, validated
by replaying all 22 bags: **20 of the 22 must-catch runs** caught, **no** firing anywhere the car is
stationary or cruising, active 0.9 % of moving time (`work/real-car-tcs/REPORT.md`; both misses are
timestamp artefacts in the label signal).

## Simulator fidelity

A policy trained here should not discover an advantage that does not exist on the car. What each
parameter is fitted to: [calibration](docs/real_data_calibration.md); weaknesses: [audit](docs/simulator_audit.md).

| area | modelled |
| --- | --- |
| Vehicle | single-track body, Pacejka tyres, friction circle on the driven axle, longitudinal load transfer, front/rear grip asymmetry, drag and rolling resistance |
| Suspension | sprung mass with roll and asymmetric squat/dive, so the LiDAR's scan plane moves with the body |
| Actuators | servo lag, rate limit, trim bias and gain error; VESC speed loop with motor and traction limits; command latency |
| Odometry | VESC dead reckoning following `vesc_to_odom`'s formula with calibration residuals, so it drifts — and deliberately not offered to the policy |
| LiDAR | 1081 beams over 270°, cast in **3D** against a layered map — beams pass over a low duct hose, strike the floor under dive, see clutter beyond the track; motion distortion across the sweep; range noise and dropouts |
| IMU | VESC 6-axis unit on its own 50 Hz clock against the 40 Hz control step; gravity leakage through roll/pitch, CoG lever arm, speed-dependent vibration, bias and random walk, quantisation, and a model of the VESC's own attitude estimate |
| Randomisation | a *configured set* of [named ranges](docs/architecture.md#what-is-randomised) (49 entries when written) re-sampled per environment on reset — **not** every numeric parameter; `vehicle.mu_r_scale`, for one, is fixed |

**Calibrated 2026-09-13: the attitude model.** The accelerometer path for body roll had failed twice;
the gyro path settled it — integrate the roll/pitch rate, band-pass out bias, regress against the same
filtered specific force after subtracting the yaw-rate leakage through sensor misalignment, and run it
on simulation first, where the answer is known, to calibrate the estimator's own attenuation. Result:
**roll was 3× exaggerated.** Nominal roll is now **1.7 deg/g** (was 5.7), squat **1.7** and dive
**0.46** (was 5.2 for both), plus a **1° rms** floor/tyre wobble the old model had at zero and IMU
misalignment ±4° ([§6.1a](docs/real_data_calibration.md)) — shifting the observation distribution,
since existing checkpoints learned against much larger tilts.

**Not calibrated: wheel slip.** With no wheel rotation state the simulator cannot produce the lock-up
or launch spin the recordings show, so neither can be trained against or evaluated in sim. Branch
`feat/wheel-dynamics` is adding rear-axle rotation, ERPM odometry and an in-sim `tcs` arm — **in
progress**. Also open ([§6.2](docs/real_data_calibration.md)): duct-hose diameter, the
target-period steering gain, latency separated from `k_us`, motion distortion, floor dropout.

## Repository map

| path | contents |
| --- | --- |
| [`f1sim/`](f1sim/) | the simulator package (`pip install -e f1sim`), ROS-independent, `COLCON_IGNORE`d |
| `f1sim/f1sim/` | `sim.py`, `dynamics.py`, `lidar.py`, `imu.py`, `odom.py`, `actuators.py`, `randomization.py`, `track.py`, `maps.py`, `tracks.py`, `mpc.py`, `raceline.py`, `teacher.py`, `gym_env.py`, `params.py`, `hard_obstacles.py`, `opponent_events.py`, `scene.py`, `trackgen.py`, `props.py` |
| `f1sim/f1sim/learn/` · `f1sim/f1sim/viewer/` | observation encoding, model, DAgger, PPO, evaluation, export, the grip arms (`grip_*.py`), `benchmark/`, `leaderboard.py`, `watch.py`; and `console/` (the three-page PyQt5 console and the editor), the moderngl renderer (`native.py`, `gl_scene.py`), `ros_link.py`, `sim_worker.py` |
| `f1sim/f1sim/calib/` · [`f1sim/tests/`](f1sim/tests/) · [`f1sim/scripts/`](f1sim/scripts/) | the bag-reading and fitting tools behind [calibration](docs/real_data_calibration.md); the test suite; viewer demo, map and car-model generation, galleries, throughput benchmark, bag-map extraction |
| [`f1sim_ros/`](f1sim_ros/) | ROS 2 bridge: launch files, `vesc_sim`, `policy_node`, `pure_pursuit`, `teleop`, config, sample maps |
| [`external/`](external/) · [`docs/`](docs/) | pinned third-party submodules (see [below](#requirements-testing-third-party)); documentation, figures and [research notes](docs/research/) |

**[docs/README.md](docs/README.md) is the documentation index**: the guides
([getting started](docs/getting_started.md), [architecture](docs/architecture.md),
[training](docs/training.md), [tracks](docs/tracks.md), [environment editor](docs/environment_editor.md),
[benchmark](docs/benchmark.md), [leaderboard](docs/leaderboard/README.md), [ROS 2](docs/ros2.md),
[viewer design](docs/viewer_design.md), [로컬 실행 안내](docs/local_pc.md)), the engineering notes
([calibration](docs/real_data_calibration.md), [audit](docs/simulator_audit.md),
[algorithm assessment](docs/algorithm_assessment.md)) and the figures.

## Research notes

Dated reports with protocol, data, and what each does *not* establish. Newest first.

| note | what it found |
| --- | --- |
| [Consolidated run `cl_hard_events_s701`](docs/research/run-recipe-2026-09-13.md) | 2026-09-13, **training now**: recipe A plus opponent events, hard obstacles, the new attitude model and the user's scenes, over 190 tracks. No result yet. |
| [First held-out scoring, suite v2](docs/research/benchmark-v2-first-2026-09-13.md) | the table above: A generalises on unseen geometry, at a second seed, with no obstacle-position memorisation |
| [Controller arms on suite v1](docs/research/controller-arms-2026-09-12.md) | `fixed_low` is the deployment default; the `reactive` arm is not adoptable |
| [Recipe restore](docs/research/recipe-restore-2026-09-12.md) | training *under* the grip clamp is the cause of the regression; train legacy, deploy clamped |
| [A702 replication](docs/research/a702-replication-2026-09-12.md) | the A recipe reproduces at a second seed; all six predeclared retention gates pass |
| [Opponent diversity, stage 1](docs/research/opponent-diversity-stage1-2026-09-12.md) | widening opponent speed and spawn gap together **fails** three predeclared gates; nothing promoted |
| [Static-grip retention](docs/research/static-grip-retention-2026-09-12.md) | four recipes × two seeds: **no eligible recipe** (best +1.875 pp against a required ≥ 2 pp), seeds disagree in sign |
| [Checkpoint benchmark v1, three-system subset](docs/research/benchmark-v1-2026-09-12.md) | the published v1 rows; a time-bounded subset chosen before any score was judged |
| [Checkpoint benchmark v1, R10](docs/research/benchmark-v1-r10-2026-09-12.md) | training on reversed directions does not recover what the five-map solo recipe loses |
| [Controller-arm evaluation](docs/research/2026-09-11-controller-arm-evaluation.md) | matched policies, 72 cells / 1152 trials: **no robust method-level gain** for `estimated`; seeds disagree on the sign |
| [Legacy-recipe probe](docs/research/2026-09-11-legacy-recipe-probe.md) | one seed: does not separate the recipe from a controller × learning interaction, in either direction |

## What is not established

- **On-car performance of a learned policy.** Nothing here has driven the physical car — separate from
  parameter calibration, several parameters of which *are* fitted to its recordings.
- **Generalisation beyond suite v2's eight maps**, two of them the pre-competition floors: v2 is the
  only unseen-venue evidence there is, and v1 is in-distribution by construction.
- **Overtaking on the real floors.** v2 measures it on one generated map — a scripted reference driver
  completed 0/4 passes on each real floor, and those cells were dropped rather than admitted
  undemonstrated ([why](docs/benchmark.md#the-paired-families-are-smaller-than-the-solo-family-deliberately)).
  Also not established: **throughput** (measure with [`scripts/benchmark.py`](f1sim/scripts/benchmark.py))
  and **friction-estimation accuracy** — a low-µ column is completion at the minimum declared friction.

## Requirements, testing, third-party

Python ≥ 3.10 with `numpy`, `torch`, `scipy`, `scikit-image`, `pyyaml`, `pillow`, `matplotlib`,
`websockets`; extras `[gym]`, `[learn]`, `[viewer]`, `[dev]` in
[`f1sim/pyproject.toml`](f1sim/pyproject.toml). **PyQt5 is in no extra** and the console needs it.
CUDA is recommended for training and real-time multi-car simulation, not required; ROS 2 Humble only
for `f1sim_ros/`. Most of the test suite runs on CPU, and so do the benchmark's `plan`, `geometry`,
`gate` and `feasibility` stages ([benchmark.md](docs/benchmark.md#checking-it-without-a-gpu)):

```bash
cd f1sim && python3 -m pytest tests -q
```

`external/` holds submodules pinned to the commits this work was developed against — `f1tenth_system`,
`f1tenth_gym`, `f1tenth_racetracks`, `f1tenth_maps`, `f1tenth-racing-stack-ICRA22`, `diagnostics` and
Korean F1TENTH team repositories used as map sources, upstreams in [`.gitmodules`](.gitmodules). They
are not vendored copies and each keeps its own licence; `git submodule update --init --recursive` if
the clone omitted them, and some must be hidden from `colcon`
([getting started](docs/getting_started.md#ros-2-workspace)). This repository carries no licence file.
