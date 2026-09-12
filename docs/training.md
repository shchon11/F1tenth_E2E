# Training

The policy is trained inside the vectorised environment, first by imitating a privileged teacher and
then by reinforcement learning. Everything here needs the `gym` extra. CUDA is recommended but not
required — `--device cpu` works for both loops and for the estimator, at smoke-test sizes; see
[Getting started](getting_started.md#training).

- [Environment](#environment)
- [Observation and action](#observation-and-action)
- [DAgger](#dagger)
- [PPO](#ppo)
- [Evaluation](#evaluation)
- [Watching a run](#watching-a-run)
- [Export](#export)
- [Experimental: friction-aware control](#experimental-friction-aware-control)

Commands are written from the repository root. Runs and checkpoints go to `~/f1sim_runs/<name>/`.
Pass `--help` to any module for the full flag list; the flags shown below are the ones that change
the experiment rather than the logging.

## Environment

```python
import torch
from f1sim.params import Config
from f1sim.gym_env import F1VecEnv, EnvConfig
from f1sim.learn import common

cfg = Config()
tracks, _ = common.load_tracks(["gen:competition:3"])
env = F1VecEnv(tracks, cfg, EnvConfig(action_mode="plan", scan_stack=6, hist_len=20),
               num_envs=1024, device="cuda")
obs, info = env.reset(seed=0)
action = torch.zeros(env.B, env.act_dim)           # normalised to [-1, 1]
obs, rew, term, trunc, info = env.step(action)
```

Environments auto-reset: an episode that ends is restarted within the same step. Three separate keys
carry what the ended episode left behind (`gym_env.py`, the `any_done` branch):

| key | contents |
| --- | --- |
| `info["final"]` | episode **statistics** for the ended ids: `ids`, `return`, `progress`, `steps`, `collided`, `lap_time` |
| `info["final_obs"]` | the **terminal observation**, before the reset overwrote it — this is what a learner bootstraps from |
| `info["final_priv"]` | the terminal privileged vector |

`info["priv"]` carries the privileged vector for an asymmetric critic. **It is the vector of the
transition that just happened, taken before auto-reset**, while the returned `obs` is already the
*new* episode's first observation for any environment that finished. At an episode boundary the two
therefore describe different states, and pairing them as "the same next state" is wrong. Use
`info["final_obs"]` / `info["final_priv"]` for the terminal state, and `info["priv"]` alongside the
observation from *before* the step for the transition being credited.

Environment actions are normalised to `[-1, 1]`; the `Simulator` underneath takes physical
`(steer [rad], speed [m/s])`.

Track sets are named: `common.TRAIN_TRACKS`, `common.EVAL_TRACKS`. The `--tracks` flag accepts
`train`, `eval`, `eval_obstacles` or a comma-separated list of catalogue names.

## Observation and action

The observation contract and the 8-number plan action are described in the
[README](../README.md#data-and-action-contracts). Two details matter when configuring a run:

- **Scan stacking.** `scan_stack` scans `scan_stride` control steps apart. At 40 Hz, six scans at
  stride 2 span 250 ms. Stacking is what lets a feedforward network see motion.
- **Proprioceptive history.** `hist_len` rows of speed, IMU, roll/pitch and action, `hist_stride`
  steps apart. It is included so that the car's own grip and lag are *inferable* from the signals it
  has — the design intent, not a demonstrated capability. Whether a trained actor uses the history
  that way has not been established; `learn/grip_probe.py` exists to ask that question. The critic
  is given those parameters directly, so only the actor would have to infer anything.

The reward is centerline progress in metres, minus a collision cost, minus a small steering-rate
penalty, minus a proximity term that ramps once the body-to-wall gap closes below a threshold, minus
a penalty for facing backwards along the lane. Progress is signed, so driving the wrong way already
pays negative reward; the explicit term makes it unambiguous.

## DAgger

Distils the privileged teacher into a policy that sees only what the car sees. The teacher drives
first; control is handed to the student while the teacher keeps labelling, and the aggregated buffer
is replayed.

A bounded starting point — raise `--envs` only as far as GPU memory allows:

```bash
python3 -m f1sim.learn.dagger --name dagger_v1 --action-mode plan \
  --envs 256 --iters 4 --steps 250
```

Relevant flags: `--iters`, `--steps`, `--beta0` (initial teacher share), `--hard-frac` (over-sample
hard states), `--scan-stack`, `--scan-stride`, `--scan-stem`, `--hist-len`, `--teacher-grip`.

## PPO

Reinforcement learning from the distilled student, with an asymmetric critic that sees privileged
state and a KL regulariser back to the imitation policy that decays over training.

A bounded starting point. `--total` is in environment steps and is the run's length; without it the
default is long enough to run for days:

```bash
python3 -m f1sim.learn.ppo --name ppo_v1 \
  --init ~/f1sim_runs/dagger_v1/student_latest.pt \
  --action-mode plan --scan-stack 6 --hist-len 20 \
  --envs 256 --horizon 32 --total 1_000_000 --sim-backend graphs
```

`--envs` is the main memory lever. For scale: 256 environments over five maps at `--cap 9` peaked at
about 5.1 GB **device-wide** (that reading includes other processes on the card) on an 8 GB
RTX 4060 Ti. Treat it as one data point, not a budget — memory scales with `--envs`, the number of
distinct tracks held on the GPU, and `--scan-stack`. Measure before committing to a size.

Relevant flags: `--envs`, `--horizon`, `--epochs`, `--minibatch`, `--total`, `--amp`, `--cap0` /
`--cap1` / `--cap-steps` (speed-cap curriculum), `--kl-coef` / `--kl-decay`, `--critic-warmup`,
`--fresh-opt`, `--tracks`, `--sim-backend`.

`--sim-backend` selects the simulator runtime: `compile` (default), `eager`, or `graphs`. `graphs`
captures explicit CUDA graphs rather than compiling; on the machine used here that capture took
1.3 s against minutes for `compile`, at 256 environments over five maps. The gap is a startup cost
only and depends on the machine and the configuration.

Multi-car fine-tuning: `--race-size M` with `--opponent teacher` puts the learner in a field of
teacher-driven cars (only car 0's transitions train), and `--opponent policy` makes every car the
learner for self-play.

### Opponent behaviour events

A teacher opponent drives the raceline at a fixed speed scale and is blind to other cars apart from
the follow-gap slowdown, so every opponent the policy meets poses the same problem: a slightly
slower car holding the racing line. `--opp-events` gives the **teacher-driven** cars of a race
scripted behaviour on top of that — a car that brakes, a car that has stopped, a car that moves
across the lane:

```bash
python3 -m f1sim.learn.ppo --race-size 2 --opponent teacher \
    --opp-events brake,stop,shift --opp-event-rate 1.0
```

| event   | what the opponent does | drawn from |
|---------|------------------------|------------|
| `brake` | commands `k` × its profile speed for `d` seconds, then resumes | `--opp-brake-scale` (k), `--opp-brake-time` (d) |
| `stop`  | commands 0 for `d` seconds — the stalled car | `--opp-stop-time` |
| `shift` | tracks the raceline offset by `o` m: ramp in, hold, ramp out — a lane change / blocking line | `--opp-shift-offset` (\|o\|, sign drawn separately), `--opp-shift-hold`, `--opp-shift-ramp` |
| `weave` | sinusoidal lateral offset | `--opp-weave-amp`, `--opp-weave-period`, `--opp-weave-time` |

`--opp-event-rate` is the expected number of events per opponent per 10 s of driving. Events never
overlap, so the realised rate is a little below the flag (at rate 1.0 with ~1.7 s events, about
0.85). `--opp-event-margin` is the body-to-wall gap kept when an event moves a car off the line.

Three properties the flags rely on:

* **Off is off.** Without `--opp-events` nothing is stepped and nothing is drawn from the
  simulator's generator, so an unflagged run is bit-identical to the same run before the feature
  existed (`f1sim/tests/test_opponent_events.py`).
* **The events only ever slow a car down**, and the multiplier is applied *before* the
  `opp_follow_gap` cap, so an opponent already braking for the car ahead never accelerates because
  an event told it to.
* **The lateral offset cannot reach a wall.** It is clamped per raceline point against the track's
  own distance field (free space − car half-width − `--opp-event-margin`), so a 0.35 m lane change
  through a 1.6 m section becomes as much of one as fits.

Events are teacher-only: with `--opponent mixed` they apply to the teacher races and never to a car
the policy is driving, and `--opp-events` with `--race-size 1` or `--opponent policy` is refused
rather than silently ignored. Each step, `info["opp_event"]` reports `id` (0 = none, otherwise the
1-based index into `brake, stop, shift, weave`), `time_left` in seconds and the `offset` in metres
each car is holding, for a viewer or a logger.

**Evaluating against them.** Training against a behaviour and never measuring it is how a recipe
gets adopted on a number that does not cover the thing it changed. Two places measure it now:

* `python3 -m f1sim.learn.evaluate --race-size 2 --opponent teacher --opp-events brake,stop,shift
  --opp-event-rate 3` takes the same flags and reports the traffic metrics — passes, pace against
  the opponent, engagement time, contacts — under `traffic` in every result, `--per-track`
  included. Quick check, no roster, no freeze, not a benchmark score.
* The benchmark's **T (traffic) family**, frozen as suite v2.1, is the scored version: brake / stop
  / shift at 3.0 per opponent per 10 s is one of its four scenarios, on five held-out maps at two
  friction levels. See [the benchmark's traffic family](benchmark.md#the-t-traffic-family). The
  scenario's event rate is chosen so that an event actually lands while the learner is in
  contention, which is measured rather than assumed.

## Evaluation

```bash
CKPT="$HOME/f1sim_runs/ppo_v1/ppo_final.pt"
python3 -m f1sim.learn.evaluate "$CKPT" --per-track --protocol trials --envs 64 --speed-cap 4
python3 -m f1sim.learn.evaluate --teacher --action-mode plan --per-track --protocol trials

# in traffic: a slower car, a car at pace, or one that brakes and stops and changes lane
python3 -m f1sim.learn.evaluate "$CKPT" --per-track --envs 64 \
    --race-size 2 --opponent teacher --opp-speed-range 0.5 0.7
python3 -m f1sim.learn.evaluate "$CKPT" --per-track --envs 64 \
    --race-size 2 --opponent teacher --opp-events brake,stop,shift --opp-event-rate 3
```

Any run with an opponent adds a `traffic` block to each result: `passes` and
`passes_per_learner_min`, `pace_vs_opponent` (the learners' arc over the opponents' arc over the
same steps), `contention_fraction` / `following_fraction` / `attacking_fraction`, `car_contacts`,
`wall_collisions`, and — when events are on — `opp_event_in_window_fraction`, the share of event
time the learner was actually close enough to have to react to. `--contention-range` and
`--attack-range` set the two windows; the defaults are the env's own `overtake_range` (12 m) and
the benchmark's tight 3 m. A solo run reports `"traffic": null` with the reason, never zeros.

The definitions are the benchmark's own — `learn.benchmark.overtake.TrafficMeter` is the same
`TrafficTrace` and the same pass detector the frozen T cells use — so a quick check and a scored
cell are measuring one thing. What differs is the protocol: `evaluate` runs auto-resetting envs
where there is no such thing as a trial, so these are rates over the rollout rather than per-trial
outcomes, and they are not comparable with a T table.

`--protocol trials` runs independent fixed-length attempts; `rolling` runs continuously with
auto-reset. `--budget-laps` sets the time budget as laps of the track at the speed cap, which makes
the budget comparable across tracks of different length. `--tracks eval` uses the held-out set.

Report first-attempt outcomes from a frozen checkpoint on the held-out set. Numbers from a run's own
training tracks are not evidence of generalisation.

To compare several finished checkpoints against each other rather than measure one, use the
[checkpoint benchmark](benchmark.md): a frozen scenario suite with pinned weights, covering driving,
stability, per-surface retention, obstacle avoidance and overtaking.

```bash
python3 -m f1sim.learn.benchmark plan        # matrix and trial counts; loads no checkpoint
```

## Watching a run

**With no arguments, this opens the PyQt5 driving console** — one window to pick a policy and a map
and watch it drive:

```bash
python3 -m f1sim.learn.watch          # console; same as python3 -m f1sim.viewer.console
```

**Plan controller.** *고급 설정 > 플랜 제어기* selects the arm the session installs: `legacy` is the
untouched MPC; `estimated` and `fixed_low` apply the grip-aware curvature speed limit and
mu-dependent acceleration/brake budgets (`learn/grip_control.py`, the `@estimated` rows of the
[checkpoint benchmark](benchmark.md)). `estimated` needs a frozen estimator `.pt`; the field is
pre-filled from `$F1SIM_GRIP_ESTIMATOR` or `~/f1sim_runs/_estimators/estimator_seed401.pt` when one
exists. One car per race only, as in training. A checkpoint trained under a non-legacy arm opens only
under that arm; a legacy-trained checkpoint opens under any — the "train legacy, deploy clamped"
configuration in [recipe-restore](research/recipe-restore-2026-09-12.md). The session facts carry
`controller` / `estimator` so the header says which was used.

**Surface friction.** *구성 > 노면 마찰 μ* is `랜덤` (per car, per reset, from the training range when
randomisation is on) or `고정`: one value pinned for every car and re-applied after each reset. The
`적용` button pushes a new fixed value into a running session immediately (all cars, mid-episode);
switching back to `랜덤` lets the next reset draw again. The panel's `시뮬 참값 μ` always shows what
the watched car actually has.

**Training from the console.** The header's *학습* page launches and watches PPO runs without
leaving the window. The recipe form starts from presets (`원본 레이스 레시피` = the original policy's
own conditions, `SGR`, `R10`, or custom), shows the exact `python -m f1sim.learn.ppo …` it will run,
and starts it as a detached process (its own session: closing the console does not stop training).
Records live in `~/f1sim_runs/_console_jobs/`, the job's output in `<run>/console-train.log`. The
monitor parses the trainer's own `upd k/N …` lines -- from that log, or from a run's W&B
`output.log`, so runs started from a shell are watchable too -- into progress, ETA, and six curves
(reward/step, collisions/km, progress, lap time, KL to the original, throughput), lists the run's
checkpoints, and *주행 화면에서 보기* makes one the driving page's next start. *중지* sends SIGINT
(the last periodic checkpoint is what remains) and SIGTERM on a second press after 12 s.

PyQt5 is **not** in the `[viewer]` extra (which is `moderngl`, `glfw`, `trimesh`) and nothing here
installs it; if it is missing the console says so and points at `--legacy-launcher`, the older Tk
picker. Importing `f1sim.learn.watch` for a headless run does not import Qt, so the recording paths
work on a machine with no Qt at all.

The headless and explicit-argument paths are unchanged:

```bash
python3 -m f1sim.learn.watch --run ~/f1sim_runs/ppo_v1 --map gen:competition:2 --cars 128
python3 -m f1sim.learn.watch --run ~/f1sim_runs/ppo_v1/ppo_final.pt --cars 256 --record out.mp4
```

These run a **separate** simulator on the latest checkpoint of a training run and reload it as it
changes, so rendering is not inside the training loop. It is not free: on one machine the two
processes still compete for CPU, GPU and memory. Keep `--cars` modest while a run you care about is
training, and expect the training job to slow down if you do not. The display shows the focus car's LiDAR points coloured by saliency —
which beams the action depends on most — along with hidden-unit and scan-feature heat grids, action
mean and standard deviation, and the critic's value estimate.

`--episodes N` records full episodes, ranks the agents (clean runs first, then progress) and replays
the best with a chase camera; `--highlights DIR` writes one video per episode headlessly.

## Export

```bash
CKPT="$HOME/f1sim_runs/ppo_v1/ppo_final.pt"
python3 -m f1sim.learn.export "$CKPT" --trt
```

Writes ONNX, optionally with a TensorRT engine, for the Jetson. The ROS policy node consumes the
same observation encoding as training ([`learn/obs.py`](../f1sim/f1sim/learn/obs.py)), so the
deployed input is constructed by the same code path — see [ROS 2](ros2.md).

## Experimental: friction-aware control

**Opt-in, unfinished, and off by default.** The default arm is `legacy`, the original behaviour.

`--controller` selects how the tracker treats tyre friction:

| arm | friction used by the tracker |
| --- | --- |
| `legacy` | none — no explicit friction limit is applied. **The default.** |
| `fixed_low` | one conservative constant, identical in every environment |
| `oracle` | the environment's true friction, per environment. Simulation only: it reads privileged state. |

`oracle` is a **privileged reference for this controller** — what perfect friction knowledge buys
*given this speed-envelope and bound derivation*. It is not a mathematical upper bound on achievable
performance: a different controller could use the same knowledge better, and nothing here proves the
derivation optimal.
| `estimated` | an estimate from causal onboard signals only |

`legacy` and `fixed_low` are different things: the first applies no explicit limit at all, the second
applies a constant one. `oracle` cannot run on a car.

The `estimated` arm works from:

- a causal history of 40 frames × 11 features — measured speed, IMU, roll/pitch, and the previously
  **issued** steering and speed commands, all of which exist on the real car;
- a small quantile model producing a lower, median and upper estimate, with a calibrated lower
  quantile and an explicit fallback while the history is still filling;
- the resulting friction adjusts the tracker's curvature-limited speed envelope and its acceleration
  and braking bounds.

The actor's inputs and outputs are identical across all four arms. That isolates the controller
**only in a frozen-actor evaluation**, where one fixed set of weights is run under each arm. In the
matched PPO runs the actor *trains* under its arm, so the trained weights differ and the result is a
comparison of two (controller, policy) pairs — not of two controllers holding the policy constant.
Identical architecture does not make the trained policies interchangeable either.

A matched set of PPO runs comparing `fixed_low` and `estimated` — three seeds each — has completed,
and so has its paired evaluation: 72 cells, 1152 trials, reported in
[Does estimating friction help the plan tracker?](research/2026-09-11-controller-arm-evaluation.md).
The short answer is **no robust method-level completion gain** — the seeds disagree on the sign — with
a consistent but small time advantage among shared successes, and a low-friction regression affecting
both arms whose **cause is not established** — a one-seed `legacy`-recipe probe
([follow-up](research/2026-09-11-legacy-recipe-probe.md)) did not separate the recipe from a
controller × learning interaction.

That evaluation is separate from the frozen-actor grid across the four arms, which holds one policy
fixed and varies only the controller; the two answer different questions and neither substitutes for
the other.

The follow-up asked whether a **training-recipe** change recovers the low-friction loss, screening
the two existing options — restoring the auxiliary friction-prediction objective (`aux`, whose head
is supervised with true µ; note the training critic separately receives privileged state and
parameters including true µ in every recipe, while the **actor** inputs exclude it at training and
deployment alike) and raising the initial KL coefficient
from 0.05 to 0.20 (`anchor`) — alone and together, two seeds each, all under the `estimated` arm.
**No recipe was eligible**: the best reached +1.875 pp mean low-friction gain against a required
≥ 2 pp. The criteria were fixed before the final selection and before the outcomes were inspected,
and the threshold was not changed after the fact. For every recipe the two training seeds also
disagree in sign at low friction, so these runs give insufficient evidence of a robust method-level
benefit. These are engineering eligibility gates, not a significance or power analysis. Protocol,
per-trial data and the independent eligibility audit:
[Can a training-recipe change recover low-friction completion?](research/static-grip-retention-2026-09-12.md).

Note that `aux` is not only an objective change: during the first 10 updates PPO's policy gradient
and KL term are off while the aux term is active, so it also changes *when* the early policy features
start moving.

### Loading a non-legacy checkpoint

`load_checkpoint` and `load_for_conditioning` **refuse** a checkpoint recorded under a non-`legacy`
arm unless the caller passes `allow_controller=True`. The default consumers — `watch`, `export` and
the ROS policy node — do not pass it, because running such a policy through the untouched tracker
would evaluate plans under a controller they were never shaped for.

The practical limitation today: **research controller checkpoints are not runnable through the
standard tools.** Only a consumer that installs the matching controller runtime should pass the flag.
The arm, the controller configuration and the estimator's identity travel inside each checkpoint, so
a loader can tell what it is being handed.

Modules: `learn/grip_control.py`, `learn/grip_estimator.py`, `learn/grip_runtime.py`, each with its
own test module.
