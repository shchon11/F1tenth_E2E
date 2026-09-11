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

## Evaluation

```bash
CKPT="$HOME/f1sim_runs/ppo_v1/ppo_final.pt"
python3 -m f1sim.learn.evaluate "$CKPT" --per-track --protocol trials --envs 64 --speed-cap 4
python3 -m f1sim.learn.evaluate --teacher --action-mode plan --per-track --protocol trials
```

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
