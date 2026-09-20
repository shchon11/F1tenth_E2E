<div align="center">

<img src="f1sim/f1sim/assets/branding/f1sim-128.png" width="96" alt="f1sim">

# F1TENTH&nbsp;E2E

### An end-to-end driving policy for 1/10-scale autonomous racing — and the batched simulator written to train it

LiDAR and proprioception in. A short local plan out. **No map, no localisation, no camera.**

<p>
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-batched%20sim-EE4C2C?logo=pytorch&logoColor=white">
<img alt="ROS 2 Humble" src="https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white">
<img alt="status" src="https://img.shields.io/badge/results-simulation%20only-orange">
<a href="docs/README.md"><img alt="docs" src="https://img.shields.io/badge/docs-index-informational"></a>
<a href="docs/research/"><img alt="research notes" src="https://img.shields.io/badge/research-34%20notes-blueviolet"></a>
</p>

<a href="docs/media/f1tenth-visualizer-overview.png">
  <img src="docs/media/f1tenth-visualizer-overview.png" width="92%" alt="The f1sim driving console">
</a>

<sub><b>The console.</b> Drive, train and build environments in one window. A real session, not a mock-up:
the shipped <code>dial_student_s901</code> driving <code>gen:competition:2</code>, photographed by
<a href="f1sim/scripts/capture_console_media.py"><code>capture_console_media.py</code></a>. Software-rendered
on CPU here, so its <code>sim 배속</code> and <code>렌더 fps</code> are llvmpipe figures, not performance numbers
(<a href="docs/media/PROVENANCE-visualizer.md">provenance</a>).</sub>

</div>

---

> [!IMPORTANT]
> **Every number in this repository was measured in simulation.** No learned policy has driven the physical car.
> Several sensor and actuator parameters *are* fitted to 22 recordings from it
> ([calibration](docs/real_data_calibration.md)) and a real-car ROS 2 policy node exists ([ROS 2](docs/ros2.md)),
> but sim-to-real transfer is untested. Two trained checkpoints ship in [`checkpoints/`](checkpoints/);
> everything older was deleted.

## See it move

Two 25-second clips, each a single rollout rather than a benchmark. The GIFs are 5-second excerpts —
click either for the full 1920 × 1080 MP4. Maps, checkpoints, per-episode friction, renderer and hashes:
[video provenance](docs/media/PROVENANCE-video.md).

<table>
<tr>
<td width="50%" align="center">
  <a href="docs/media/f1tenth-simulator-demo.mp4"><img src="docs/media/f1tenth-simulator-demo.gif" alt="simulator demo"></a>
  <br><b>The simulator</b>
</td>
<td width="50%" align="center">
  <a href="docs/media/f1tenth-learning-demo.mp4"><img src="docs/media/f1tenth-learning-demo.gif" alt="trained policy demo"></a>
  <br><b>A trained policy</b>
</td>
</tr>
<tr>
<td valign="top"><sub>The <b>scripted raceline teacher</b> — the privileged reference driver, <b>not a learned
policy</b> — threading the duct hoses of <code>real:korea_2026_competition</code> at µ 0.921, orange LiDAR
returns along the hose it is passing.</sub></td>
<td valign="top"><sub>A finished run played back with its emitted plan drawn green ahead of the car beside the
blue raceline, and its own training history inset. <b>The car hits a wall at 5.125 s and the episode
resets</b>, marked on screen.</sub></td>
</tr>
</table>

## Quick start

You need **Python 3.10+** and a Linux machine. A GPU is optional — training is much faster with one,
everything else runs fine without.

```bash
git clone --recurse-submodules https://github.com/shchon11/F1tenth_E2E.git
cd F1tenth_E2E
./setup.sh              # makes .venv, picks the right torch for your hardware, checks it runs
source .venv/bin/activate
f1sim-console
```

`setup.sh` looks at what the machine actually has — it reads your NVIDIA driver version and installs
the CUDA build that driver can load, or a ROCm build, or the CPU build if there is no GPU. It touches
nothing outside the checkout. Re-run it any time; it upgrades in place.

<table>
<tr><td><code>./setup.sh --cpu</code></td><td>CPU torch even where there is a GPU</td></tr>
<tr><td><code>./setup.sh --no-viewer</code></td><td>skip the GUI (a headless training box)</td></tr>
<tr><td><code>./setup.sh --desktop</code></td><td>also put <b>f1sim Console</b> in the application menu</td></tr>
<tr><td><code>./setup.sh --venv ~/envs/f1</code></td><td>put the virtualenv somewhere else</td></tr>
</table>

**Your first run:** pick a policy under **① 정책 런**, a map under **② 맵**, press **시작**. The line
above the button always says which of the two is still missing. Everything else has a working default
(정방향 / 없음 / 무작위), so those two choices are the whole first run. Press <kbd>H</kbd> for the keys.

### Where it keeps things

Nothing is written outside your home directory, and every location has an environment variable:

| what | default | override |
| --- | --- | --- |
| runs and checkpoints | `~/f1sim_runs` | `$F1SIM_RUNS` |
| maps you import | `~/.f1sim/maps` | `$F1SIM_MAPS` |
| environments you draw | `~/f1sim_scenes` | `$F1SIM_SCENES` |
| console settings | `~/.f1sim/console.json` | `$F1SIM_CONSOLE_PREFS` |

Point `$F1SIM_RUNS` at a scratch disk and nothing else has to change.

### If something does not work

| | |
| --- | --- |
| `No module named venv` | `sudo apt install python3-venv` (Debian/Ubuntu) |
| the console opens but no map appears | the submodules did not clone: `git submodule update --init --recursive` |
| `torch.cuda.is_available()` is False | your driver is older than the wheel. `./setup.sh --cuda cu121`, or `./setup.sh --cpu` |
| recording is greyed out | `sudo apt install ffmpeg` — everything else works without it |
| the GUI uses the GPU you are training on | it picks the emptiest card; pin it with the 연산 장치 control under 고급 설정 |

### Install it as an application

```bash
packaging/install-desktop-entry.sh --venv .venv    # adds it to the application menu, per user
packaging/build-appimage.sh                        # a portable single file, dist/f1sim-Console-*.AppImage
```

The AppImage carries the window, not torch: a CUDA build is gigabytes and has to match the driver of
whatever machine runs it. The simulation runs in a separate process under your own Python — set
`F1SIM_WORKER_PYTHON=/path/to/python` if it cannot find one with torch installed.

<details>
<summary><b>Use it as a library</b> — four lines to a stepping simulator</summary>

Run this from inside `f1sim/`; the repository root holds a folder that shadows the installed package
([why](docs/getting_started.md#a-directory-shadowing-trap)).

```python
import torch
from f1sim import Track, Config, Simulator

cfg = Config()
cfg.sim.compile = False                      # torch.compile is a CUDA path; skip it on CPU
track = Track.generate_random(seed=0)        # or Track.from_ros_map("map.yaml")
sim = Simulator(track, cfg, num_envs=4, device="cpu")

r = sim.step(torch.zeros(4, 2))              # action = (steer [rad], target speed [m/s]) per env
print(r.scan.shape, r.odom.shape, r.state.shape)     # (4, 1081) (4, 5) (4, 8)
sim.reset(torch.nonzero(r.collision).flatten())      # a partial reset re-samples the randomised parameters
```
</details>

<details>
<summary><b>Bring your own map</b> — SLAM toolbox output, straight in</summary>

A ROS `map_server` pair (`map.yaml` + `map.pgm`/`.png`) is a track. The centerline is traced and cached,
so it arrives with a raceline, a teacher and a lap-time reference.

```bash
python3 -m f1sim.slam_map ~/maps/venue.yaml --preview venue.png   # measure and look before training
python3 -m f1sim.learn.evaluate CKPT --tracks ~/maps/venue.yaml   # use it by path
python3 -m f1sim.slam_map ~/maps/venue.yaml --install venue       # or catalogue it as user:venue
```

The environment editor imports the same pair (**SLAM 맵 불러오기**), which is where to fix a map the
tracing got wrong. See [tracks](docs/tracks.md) and the [editor](docs/environment_editor.md).
</details>

## What is in the box

<table>
<tr><td width="33%" valign="top">

### 🏎️ Batched simulator
A 3D LiDAR cast against a layered map, a dynamic single-track vehicle with Pacejka tyres and a
rotating rear axle, VESC actuators with latency, an IMU on its own clock, and 51 randomised
parameter ranges — thousands of environments on one GPU.

[architecture](docs/architecture.md) · [audit](docs/simulator_audit.md)

</td><td width="33%" valign="top">

### 🖥️ One console for everything
**Drive**, **train** and **build environments** without leaving the window: live policy internals,
a recorder, a friction dial, obstacle difficulty, and a launcher that charts a running job from its
own log.

[viewer design](docs/viewer_design.md) · [editor](docs/environment_editor.md)

</td><td width="33%" valign="top">

### 📊 Frozen benchmarks
Pinned scenario suites with sealed weights, so a number is comparable across months. Held-out maps,
traffic families, per-surface retention, and research notes that say what each result does *not*
establish.

[benchmark](docs/benchmark.md) · [leaderboard](docs/leaderboard/README.md)

</td></tr>
</table>

## How the policy is built

```
LiDAR ×6 frames + proprioception ──▶  CNN + MLP (+ GRU)  ──▶  6 curvature knots + 2 speeds
                                                                        │
                                              iLQR tracker, 12 × 50 ms ─┴─▶  steer, speed
```

**Observation:** 6 stacked 1081-beam scans and a 366-number proprioceptive vector — speed, IMU, VESC
roll/pitch, previous normalised actions, the speed cap and 20 rows of history.
**Action:** 8 numbers — 6 curvature knots along the next stretch of travel plus start and end speed —
tracked by an iLQR on a kinematic bicycle. Integrated pose is excluded on purpose, so the same tracker
runs on the car ([`mpc.py`](f1sim/f1sim/mpc.py), [architecture](docs/architecture.md#plan-tracking)).

**Training** distils a privileged raceline teacher into the LiDAR-only student (DAgger), then refines
it with PPO against an asymmetric critic.

```bash
python3 -m f1sim.learn.dagger   --tracks train ...       # imitate the teacher
python3 -m f1sim.learn.ppo      --tracks train ...       # then race
python3 -m f1sim.learn.evaluate --tracks heldout ...     # first attempts on maps nothing trained on
python3 -m f1sim.learn.benchmark plan                    # the frozen suite's matrix; loads no checkpoint
```

`ppo` defaults to `--envs 2048` and `--total 100e6`, so running it bare starts a multi-day job on a
large GPU. [training](docs/training.md) gives bounded recipes that finish.

### The grip dial

Friction cannot be read from the car's sensors at a pace it survives — a GRU probe reads µ at
**R² 0.01–0.02** on held-out environments while reading the speed-scale calibration at 0.48 from the same
data. A policy that is *told* the friction uses all of it, so the number is **supplied, not inferred**:
`--cond dial` trains the student on `µ − margin` and builds the teacher's labels for that same number,
which makes the input a command an operator sets rather than a guess.

| on `real:map12x16`, 256 cars × 60 s | median lap | a crash every |
| --- | ---: | ---: |
| privileged teacher | 8.57 s | 89 laps |
| generalist student | 8.48 s | 26 laps |
| **specialised on this map, dial 0.15 under** | **7.62 s** | **105 laps** |

Forty-five minutes of PPO on one map takes the generalist **10 % quicker and four times safer**, past the
privileged teacher on both — which is what a practice day before a race buys.
[the research note](docs/research/mintime-teacher-speed-head-2026-09-19.md) ·
[checkpoints](checkpoints/)

### The three pages

| page | what is on it |
| --- | --- |
| **주행** ([design](docs/viewer_design.md)) | a map card listing base maps in three groups (학습 / 검증 / 내 환경) over a search box for the whole catalogue; a scenario row (방향, 장애물 난이도, 시드); surface friction and the **grip dial**, both appliable mid-session; a recorder; and the policy I/O strip that puts what the policy was commanded beside what the simulator actually did |
| **학습** ([training](docs/training.md#watching-a-run)) | recipe presets that show the exact `python -m f1sim.learn.ppo …` they will run, start it detached, and chart it live from the trainer's own log lines; a card per running job; *주행 화면에서 보기* hands a checkpoint to the drive page |
| **환경** ([editor](docs/environment_editor.md)) | paint duct hoses and tall walls or draw them as editable vector paths, place props, import GLB / OBJ / STL meshes, **import a SLAM map**, save as `scene:<name>` — a catalogue entry the simulator, the trainer and the drive page all load. **랜덤 트랙 생성** builds closed tracks from a recipe of features |

## Results

<details open>
<summary><b>Held-out suite v2 — four systems, 2026-09-13</b></summary>

Every figure from [benchmark-v2-first-2026-09-13.md](docs/research/benchmark-v2-first-2026-09-13.md),
raw cells beside it. Suite v2 changes the maps and *nothing else* against v1 — friction levels, seeds,
envs, race size, speed cap, lap budget, sensor noise and opponent are identical — so a v1 and a v2 row
differ in the scenario and nothing that could explain the difference away.

| system | solo /384 | solo less bb22-3 /336 | low µ /128 | avoidance /96 | overtaking /32 | coll/km ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_original@legacy` | 228 | 227 | 27 | 21 | 23 | 6.85 |
| `frozen_original@estimated` | 265 | 258 | 71 | 43 | 22 | 5.04 |
| **`cl_origrecipe_legacy_s701`** (A, seed 701) | **287** | **274** | **77** | 55 | **25** | **3.74** |
| `cl_oppdiv_control_s801` (A recipe, seed 801) | 285 | 277 | 75 | **60** | **25** | **3.74** |

**The A recipe generalises**: it beats the frozen original on every aggregate column, and its second
seed lands within noise of the first. Per map it is not uniform, and `real:blackbox2022_3` scores
1–13 / 48 for *every* system — the user ruled it an unfair map with dead-end side branches no training
map has, so the second column is the headline solo number while the map stays in the freeze. A
**memorisation probe** found trained and novel obstacle seeds on one map score alike: what the policy
fails on is novel map *structure*, not novel placement. Lap time is absent on purpose — it is over each
system's own completions, so a system that fails the hard cells looks fast.

</details>

<details>
<summary><b>The three suites, and which one decides</b></summary>

| suite | what it asks | scenarios |
| --- | --- | --- |
| **v1** | was the training distribution fitted at all | 34 cells / 272 trials, 3 reused development maps |
| **v2** | does it hold up on geometry it has never seen | 64 cells / 512 trials, 8 maps outside every training variant, freeze `89805514…` |
| **v2.1** | and does it hold up in traffic | v2 byte-for-byte plus an 80-cell traffic family — slow / pace / event / pair opponents on 5 held-out maps |

**Adoption decisions use v2.** ([benchmark.md](docs/benchmark.md))

</details>

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
from `--seed`. Grammar, the catalogue and the conversion table: [tracks.md](docs/tracks.md).

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
ros2 launch f1sim_ros graph_sim.launch.py checkpoint:=...                 # the graph on the simulator
ros2 launch f1sim_ros graph_console.launch.py checkpoint:=...             # the graph on a console session
ros2 launch f1sim_ros graph_car.launch.py checkpoint:=...                 # the graph on the real car
ros2 launch f1sim_ros f1tenth_stack_sim.launch.py map:=gen:competition:3  # the real car's stack, unmodified
ros2 launch f1sim_ros sim.launch.py                                       # standalone bridge, one env, real time
ros2 run f1sim_ros system_check                                           # is everything there?
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

**The graph is the system boundary.** `policy_node` turns the sensor topics into a plan
(`/f1sim/plan`: eight normalized floats, the checkpoint that produced them, and the stamp of the
scan they came from) with the same [`learn/obs.py`](f1sim/f1sim/learn/obs.py) the training side
uses; `controller_node` turns the plan into `/drive` with the iLQR tracker, the runtime arm
(`fixed_low` by default, µ 0.73423; `fixed_low+clearance` adds the geometry layer that reads
`/scan` and nothing else; `estimated` and `reactive` are simulator research arms, refused here) and
the traction guard. One `config/graph.yaml` carries every parameter and the three launch files
differ only in where the sensors come from, so a simulator run and a car run are the same run.
Nothing here has been tested on a physical vehicle.

The split is bit-exact against the monolithic node it replaced: 1920 commands over four arms, the
guard on and off, a simulator bag and a real car bag, worst difference 0.000e+00 on steering and
speed. A checkpoint that emits steering and speed directly, and a published-baseline node, bypass
the controller and publish `/drive` themselves. `ros2 run f1sim_ros eval` scores one benchmark cell
through the graph with the benchmark's own metric code; suites are still scored batched.

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
| Randomisation | a *configured set* of [named ranges](docs/architecture.md#what-is-randomised) (51 in `RandomizationConfig.ranges` today) re-sampled per environment on reset — **not** every numeric parameter; `vehicle.mu_r_scale`, for one, has no entry at all |

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
| [`f1sim_ros/`](f1sim_ros/) | the ROS 2 graph: `policy_node`, `controller_node`, `eval`, `system_check`, the `vesc_sim` and standalone bridges, `pure_pursuit`, `teleop`, launch files, config, sample maps |
| [`f1sim_interfaces/`](f1sim_interfaces/) | `Plan` and `PolicyState`: the two messages on the boundary between the policy and the controller |
| [`external/`](external/) · [`docs/`](docs/) | pinned third-party submodules (see [below](#requirements-testing-third-party)); documentation, figures and [research notes](docs/research/) |
| [`setup.sh`](setup.sh) · [`packaging/`](packaging/) | one-command install (detects CUDA / ROCm / CPU); the desktop entry, its per-user installer and the AppImage build |

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

[`./setup.sh`](setup.sh) installs all of this and is the supported way in. What it installs: Python
≥ 3.10 with `numpy`, `torch`, `scipy`, `scikit-image`, `pyyaml`, `pillow`, `matplotlib`,
`websockets`; extras `[gym]`, `[learn]`, `[viewer]`, `[dev]` in
[`f1sim/pyproject.toml`](f1sim/pyproject.toml), plus PyQt5, which is in no extra because a headless
training box does not want it. CUDA is recommended for training and real-time multi-car simulation,
**not required** — the console, the tests and the CPU benchmark stages all run without a GPU. ROS 2
Humble is needed only for `f1sim_ros/`, and `ffmpeg` only to record video.

Nothing in the package assumes a particular machine: the GPU is chosen by free memory rather than by
index (`auto` in the 연산 장치 control), and every directory it writes to has an environment variable
([above](#where-it-keeps-things)). Most of the test suite runs on CPU, and so do the benchmark's
`plan`, `geometry`, `gate` and `feasibility` stages
([benchmark.md](docs/benchmark.md#checking-it-without-a-gpu)):

```bash
cd f1sim && python3 -m pytest tests -q
```

`external/` holds submodules pinned to the commits this work was developed against — `f1tenth_system`,
`f1tenth_gym`, `f1tenth_racetracks`, `f1tenth_maps`, `f1tenth-racing-stack-ICRA22`, `diagnostics` and
Korean F1TENTH team repositories used as map sources, upstreams in [`.gitmodules`](.gitmodules). They
are not vendored copies and each keeps its own licence; `git submodule update --init --recursive` if
the clone omitted them, and some must be hidden from `colcon`
([getting started](docs/getting_started.md#ros-2-workspace)). This repository carries no licence file.
