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

Track sets are named: `common.TRAIN_TRACKS`, `common.EVAL_TRACKS`. The `--tracks` flag accepts a
split name (`train`, `heldout` / `eval`, `heldout_obstacles`, `heldout_all`) or a comma-separated
list in either grammar — `real/bb22-1@rev#line:44` or `real:blackbox2022_1+rlobs44~rev`. See
[Tracks](tracks.md) for the ids, the scenario grammar and what the three splits are.

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
hard states), `--scan-stack`, `--scan-stride`, `--scan-stem`, `--hist-len`, `--teacher-grip`. The
race, opponent, behaviour and spawn flags are the shared group (`--race-size`, `--opponent`,
`--opp-events`, ... — see [Opponents](#opponents-the-situations-not-the-events)), so a DAgger run and
a PPO run describe the same traffic with the same words.

### `--teacher interactive` — a teacher that races the other car

`RacelineTeacher` follows a line computed once from the occupancy grid and has never heard of the
other cars; in a race it is kept off them only by the follow-gap cap, which can slow it and nothing
else. A student distilled from it can learn *hold the line, and brake for what is in front of you
now*, and no amount of DAgger will get a pass out of it.

`--teacher interactive` (`f1sim.interactive_teacher`) labels with a **best response** instead. It
keeps the raceline teacher as its reference — the line, the grip-aware speed profile, the latency
compensation, the plan fit — and searches a small family around it:

* candidates in the same 8-D plan space the student outputs: `--teacher-offsets` (metres left of
  the raceline, default `-0.6,-0.3,0,0.3,0.6`) × `--teacher-speeds` (multipliers on the plan's two
  speed targets, default `1,0.8,0.55,0.3`), each generated by `RacelineTeacher.plan_action` itself;
* each rolled out to `--teacher-horizon` seconds (default 1.0) by `mpc.reference`, i.e. by the
  geometry the plan tracker walks, so the trajectory scored is the trajectory that would be driven;
* each scored by `C_progress + w_wall C_wall + w_opp C_opp_future + w_clear C_clear +
  w_smooth C_smooth` (`--teacher-cost`, five comma-separated weights);
* argmin is the label.

`C_opp_future` is the point: it is matched **sample for sample in time** against where each opponent
is predicted to be, not against where it is now. The three situations that matter are all invisible
at t = 0 — a car drifting right is straight ahead *now*, a car braking is at a comfortable distance
*now*, a car about to take the gap is leaving it open *now*.

Needs `--action-mode plan` (its candidates are plans) and `--race-size > 1` (with no other car its
opponent term is identically zero), and both are refused rather than silently downgraded.

### `--memory gru` in DAgger

A recurrent student is trained by **truncated BPTT over contiguous chunks** of the buffer
(`--chunk-length`, default 16) rather than a uniform draw of single steps. A recurrent policy
trained from a zero hidden state at every sample is a different policy from the one that drives —
`Actor.forward` refuses a recurrent actor for exactly that reason. `--hard-frac` and `--memory gru`
do not combine: the chunk update has no single step to weight.

`--scan-channels` works here as it does in PPO. The decayed occupancy channel is *recorded* during
collection rather than replayed at sampling time: it is a recursion over every scan since the
episode began and cannot be reconstructed from the middle of a chunk.

### `--opp-token` — the privileged opponent block (an ORACLE)

A block appended to the proprio vector, after every column the observation already had, carrying the
nearest two cars' state from the simulator:

| `--opp-token` | columns per car | what they are |
|---|---|---|
| `off` | 0 | the observation this env has always produced |
| `pos` | 3 | Δx, Δy in the ego frame, and a presence flag |
| `posvel` | 5 | + Δv_x, Δv_y |
| `future` | 13 | + where that car will be at +0.1 / 0.25 / 0.5 / 0.75 s |

Everything is on `PRIV_OPP_DIST_SCALE`, presence-gated (every column of an absent car is exactly 0,
the flag included), and zero-initialised on a warm start, so a checkpoint widened into it starts as
the checkpoint it came from and learns to use the columns. `--opp-future-model` says which prediction
the `future` columns carry (`plan`, the default, or `constv`).

**It is not deployable and it is not a benchmark input.** No sensor on the car produces it, so
`learn.export`, `f1sim_ros.policy_node`, `learn.obs.ObsBuilder` and the benchmark's model adapter
all refuse a checkpoint that declares one. It exists to answer a question — *does the planner use
the opponent's motion when it is handed it?* — and a checkpoint trained with it is a measurement.

How well the `future` columns predict the realised future is measured, not assumed:

```bash
python3 -m f1sim.learn.opp_future_check --tracks gen:control:1400 --steps 600 --device cpu --eager
```

reports MAE and the body-frame displacement R² per horizon, for `plan` and for the constant-velocity
floor, split by teacher-driven and policy-driven cars.

```bash
# the whole chain: interactive teacher, oracle input, recurrent student
python3 -m f1sim.learn.dagger --name dagger_it --action-mode plan --envs 258 \
  --race-size 3 --opponent teacher --opp-speed 0.6 1.15 \
  --opp-events brake,stop,shift,defend,yield,line,oblivious --opp-event-rate 1.0 \
  --opp-defend-prob 0.3 --opp-yield-prob 0.2 --opp-line-prob 0.3 --opp-oblivious-prob 0.1 \
  --teacher interactive --opp-token future \
  --memory gru --memory-hidden 128 --scan-channels memory,edges --chunk-length 16
```

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
`--fresh-opt`, `--tracks`, `--obstacle-draws`, `--sim-backend`, `--memory` /
`--memory-hidden` / `--memory-critic`, `--scan-channels` / `--scan-memory-tau`,
`--aux-grip` / `--aux-opp` / `--aux-future` (auxiliary heads), `--metrics-jsonl`.

`--tracks` takes a split name or a list of scenarios ([Tracks](tracks.md)):

```bash
--tracks train                                       # the curated 149-variant training split
--tracks 'real/icra22,real/icra22@rev,rt/spielberg'  # three scenarios, new grammar
--tracks 'real:icra2022,rt:Spielberg~rev'            # the loader's grammar, still accepted
--tracks 'real/bb22-1#line:*' --obstacle-draws 8     # one map, eight random box placements
```

`--obstacle-draws N` (default 8) is how many rasterised variants an open seed (`#<kind>:*`) becomes.
The draw comes from `--seed`, so the same seed gives the same maps, and the concrete loader names go
into the run's W&B config. An entry with a fixed seed, or no obstacles at all, is one track however
large `N` is.

`--sim-backend` selects the simulator runtime: `compile` (default), `eager`, or `graphs`. `graphs`
captures explicit CUDA graphs rather than compiling; on the machine used here that capture took
1.3 s against minutes for `compile`, at 256 environments over five maps. The gap is a startup cost
only and depends on the machine and the configuration.

Multi-car fine-tuning: `--race-size M` with `--opponent teacher` puts the learner in a field of
teacher-driven cars (only car 0's transitions train), and `--opponent policy` makes every car the
learner for self-play.

### Policy memory (`--memory gru`)

Off by default. An unflagged run is byte-for-byte the run it was: the same network, the same
rollout, the same minibatching, the same loss (`tests/test_ppo_memory.py` holds it to a loss oracle
frozen from the code before this existed).

**Why.** The actor's observation is six LiDAR frames — 150 ms at 40 Hz — and a 20-row proprio
history. A box that has left the 270° scan window is gone from the observation entirely. Three
finetunes with more obstacles, a looser leash, a higher learning rate and a plan-clearance penalty
all scored the same on the held-out proxies, and 11 of 16 crashes on the user's held-out scene were
on one row of boxes. What was left to change is the observation and the architecture.

**What it is.** A GRU over the actor's per-step trunk embedding (the scan stem's 256 features
concatenated with the proprio embedding), whose output enters the first MLP layer's *preactivation*
through a bias-free projection initialised to zero — the same construction `--cond` uses. Two
properties follow and the warm start needs both: at step 0 the actor's output is bit-identical to
the checkpoint it was warm-started from, so nothing the original could do is lost; and the
projection's gradient is nonzero from the first update, so the path trains immediately.

GRU rather than LSTM: one state tensor rather than two — which is what the ROS node's callback
state, the viewer's static capture tensor and every scoring path have to carry — about a quarter
fewer parameters at the same width, and nothing to gain from the extra gate over a 32-step
truncated-BPTT horizon (0.8 s at 40 Hz).

The critic gets its **own** GRU by default (`--memory-critic own`; `none` leaves it feedforward).
It already carries its own stem and reads the privileged vector the actor must never see, so
sharing the recurrence would be the one place a value gradient reached the actor's trunk — and the
actor's forward is what the deployment budget measures, so a critic that is never exported costs
the car nothing.

```bash
python3 -m f1sim.learn.ppo --name ppo_mem --init "$FROZEN_ORIGINAL" \
  --memory gru --memory-hidden 128 --scan-channels memory,edges \
  --action-mode plan --scan-stack 6 --hist-len 20 --envs 256 --horizon 32 --total 4_194_304
```

**Warm start.** `--init` plus `--memory gru` goes through `model.load_for_memory`, not
`load_checkpoint`: every tensor the checkpoint holds is copied *by name*, and the only tensors
allowed to be new are the memory modules (whose output projections are zero), the future head
(whose output layer is zero) and, when extra scan channels are on, the zeroed new input columns of
the stem's first convolution. Anything else left fresh raises rather than training a
half-initialised network. Adam's moments are re-keyed by parameter name and the new tensors start
fresh, so the optimiser state carries over too.

**Resuming is not warm-starting.** Whatever the `--init` checkpoint already records — its memory,
its scan channels, its future head — it keeps, and only the pieces it does not have are added
(`ppo.warm_start_additions`). This matters because the flag that *builds* a piece is also the flag
that keeps training it: `--aux-future 1.0` is a coefficient, so leg two of a run passes the same
command line as leg one with `--init` moved, and must not be sent down the warm-start path for
something that is already trained. The run prints which architecture it resumed.

**How the update works.** The rollout stores each env's hidden state at the start of every horizon
chunk; the update replays whole env chunks through the network (truncated BPTT over the 32-step
horizon) and masks the hidden state back to zero at every episode boundary, exactly where the
rollout reset it. Minibatches are therefore whole env chunks: `--minibatch` stays a sample count,
and a recurrent minibatch is that count divided by `--horizon` envs. The KL leash's reference stays
the **feedforward** original — the frozen copy is evaluated with the recurrence switched off, and
the run refuses to start if that copy's projection is not still exactly zero.

**Where the hidden state goes afterwards.** Every inference path carries it: `rollout_metrics`,
`evaluate.py`, the benchmark adapter (cleared per trial), the viewer worker's actor runner (a
static tensor updated in place) and the ROS policy node (across scan callbacks, cleared on
`/f1sim/reset` and when the scan stream restarts). A recurrent checkpoint refuses the feedforward
entry points (`Actor.forward`, `.dist`, `.forward_all`) rather than silently running from zeros
every step, so an unconverted caller is an error and not a quietly worse policy.

### Extra scan channels (`--scan-channels`)

Off by default, and independent of `--memory`. Both are pure arithmetic on the scan the env already
emits, added on the policy side of the observation boundary, so the env, a checkpoint's recorded
`extra["spec"]` and every consumer of that spec are untouched. Each enabled channel is one more row
on the scan's channel axis, appended after every column the original had and zero-initialised, so a
warm start stays bit-identical.

| channel | what it is | why |
|---|---|---|
| `memory` | the closest return seen at each bearing recently, relaxing back toward "no return" with time constant `--scan-memory-tau` (default 2 s) | explicit cheap memory the GRU does not have to learn. Bearings are the car's own and are **not** motion-compensated: a LiDAR-only policy has no pose, so a box that leaves the window leaves a fading trace at the bearing it left by, not a transformed position |
| `edges` | `abs(r[i] - r[i-1])` per beam | the crack between two boxes in a row is two range discontinuities a few beams apart; the gap the lane actually leaves is one. The stem's first layer is a stride-2 7-tap convolution, so a two-beam crack lands inside one tap — as its own channel it survives at full beam resolution |
| `aligned` | the signed residual `r_t - warp(r_{t-k})`, soft-thresholded | see below |
| `aligned_prev` | the warped previous range itself | |
| `aligned_valid` | 1 where the warp had a prediction AND the current beam returned | |

Cost, measured with the rest of the budget below: 0.03 ms of a 25 ms control step for `memory` +
`edges`; 0.57 ms for the three aligned rows together (they share one warp).

#### The ego-motion-aligned residual (`aligned`, `aligned_prev`, `aligned_valid`)

**Why.** Between two LiDAR frames almost everything that moves is the *ego*: a corridor sweeping past
at 9 m/s, against which another car is a handful of beams whose apparent motion is the sum of its own
and the ego's. Nothing in PPO's objective rewards separating the two, and `probe_hidden --current`
measures what that costs — the recurrent state carries the ego's own motion (R² 0.9) and the
opponent's relative velocity hardly at all. This channel subtracts the ego's motion from the
observation *before* the network sees it, so what still changes is what moved by itself. It is
inductive bias rather than capacity: the GRU is not widened.

**What it is.** Not `r_t − r_{t−k}`. The scan from k control steps ago becomes a point cloud in the
old sensor's own frame (the simulator's beam geometry, `Ry(pitch) Rx(roll)` on the level bearing),
is carried forward by k composed constant-(v, ω) arcs, and is **re-rasterised onto the current
angular bins**; the residual is the current range minus that. Everything the warp uses is measured
on the car — VESC wheel speed, IMU gyro z, IMU roll/pitch — so the channel is deployable, and
`f1sim_ros/policy_node.py` feeds it from the three sensors it already reads.

Three details are not decoration:

* **a no-return warps as a lower bound, not a point.** "Nothing out to `range_max`, that way" stays
  a bound after the car moves, so only something appearing *closer* than it counts. Treated as a
  point, every far beam of every straight would report the ego's own 0.9 m of travel as motion.
* **only the attitude's CHANGE is used**, both tilts taken relative to their own midpoint. The VESC
  attitude estimate is biased and drifts — 11° rms in simulation, past 40° in five of the thirteen
  competition recordings — and the absolute value tilts the frame the ego's arcs are integrated in.
  Measured on a clean recording: coverage 70 % → 82 % of beams, false positives 2.89 % → 2.67 %.
* **`aligned_valid` is a real channel.** A bin no warped point reached, or one where the current
  beam did not return, is *unknown*; the residual there is exactly 0 and the mask says so. "I cannot
  tell" and "nothing moved" must not be the same number.
* **the threshold is `sign(R)·max(|R| − τ, 0)`, and τ was fixed before training** from the
  static-world floor on the real recordings: σ_static = 0.025 m, τ = 3σ = **0.075 m**
  (`python -m f1sim.learn.aligned_floor bags --aligned-k 4 --tau 0 --tol-beams 0`). Soft rather than
  hard, so a real residual keeps its size instead of arriving as a step function. On top of it, a
  two-frame consistency test: a residual survives only if the previous step had one of the same sign
  above τ within `--aligned-consist-beams` of the same bearing.

```bash
python -m f1sim.learn.ppo ... --memory gru --memory-hidden 128 \
  --scan-channels memory,edges,aligned,aligned_prev,aligned_valid --aligned-k 4
python -m f1sim.learn.aligned_floor sim    # the floor with attitude randomisation on
python -m f1sim.learn.aligned_floor bags   # and on the real recordings
```

`--aligned-k` is declared a priori at 4 (100 ms); `docs/research/motion-memory-2026-09-14.md` reports
the sensitivity over {2, 4, 8} rather than picking one from a result.

#### A separate motion state (`--motion-memory`, `--aux-opp-mask`, `--aux-motion`)

Off by default: no module, no meta, no RNG draw, and `tests/test_ppo_memory.py` holds the loss to the
same frozen oracle. On, the recurrence is **split**:

```
current LiDAR ----- scan stem ------------> main GRU  ---+
aligned rows ------ motion encoder ------> motion GRU ---+--> plan head
                         |                     |
                         +-- beam mask         +-- current Δv, and --aux-future
```

`h_dyn` (≤ 64, `--motion-hidden`) is carried *inside the same hidden tensor* as the main state, so
every path that already carries one — the rollout buffers, `reset_hidden`, the ROS node's callback
state, the viewer's CUDA-graph buffer, the ONNX `hidden` input — carries this too, unchanged.

The two auxiliaries attach to `h_dyn` and the motion encoder **only**, never to the main hidden
state, and `--aux-future` moves to `h_dyn` as well when the branch exists. The reason is mechanical:
ego dynamics are the cheap way to drive any of these losses down, so a main representation that is
allowed to absorb them will.
`tests/test_motion_memory.py::test_the_auxiliaries_reach_the_motion_branch_and_nothing_else` is that
claim as a statement about the autograd graph.

| flag | loss | label |
|---|---|---|
| `--aux-opp-mask COEF` | weighted BCE on a per-beam "is this beam on another car" logit, read from the motion **encoder**'s features | the LiDAR's own `scan_type == HIT_CAR` — privileged, train-time, already cast |
| `--aux-motion COEF` | MSE on the nearest opponent's **current** relative velocity, read from `h_dyn` | the k = 0 row of the same privileged snapshot the future head uses, presence-masked |

Staging is strict and in this order: **E3-a** mask, **E3-b** + Δv, **E3-c** + `--aux-future`. The
mask logits are train-time only: they are never fed to the planner and never exported (the heads are
not called by `Actor.forward` / `.step`, so the traced graph cannot contain them — checked).

### Predicting the near future (`--aux-future`)

Off by default, and off the head is not built at all: same parameters, same state dict, same loss
(`tests/test_ppo_memory.py` holds `--aux-future 0` to the same frozen oracle `--memory off` is held
to, and `tests/test_future_head.py` checks that a checkpoint which *carries* a trained head still
scores a batch identically while the coefficient is 0).

**Why.** `--memory gru` gives the actor a hidden state; nothing so far asks it to *represent*
anything. PPO rewards driving, the grip head asks for a friction, and the opponent head asks where
the nearest car is **now** — which six stacked LiDAR frames already nearly determine. A recurrent
state that is never asked for something the present observation does not contain is free to settle
into a smoothed copy of it. The claim the work is aiming at is "a latent race state you plan
through", and that needs (a) a loss that forces the state to carry the near future and (b) a
measurement that says whether it does. This is (a); `python -m f1sim.learn.probe_hidden` is (b).

**What it predicts.** From the actor's **recurrent state** after the step (`h_t`; the trunk features
when `--memory off`), K = 20 control steps — 0.5 s at 40 Hz — ahead:

| target | what it is | scale |
|---|---|---|
| `opp_lon`, `opp_lat` | the nearest opponent's position in the ego body frame at t + K | `PRIV_OPP_DIST_SCALE` (5 m), the scale `privileged()` already puts the present offsets on |
| `opp_vlon`, `opp_vlat` | that opponent's velocity **relative to the ego**, rotated into the ego frame at t + K | the same 5 |
| `ego_speed`, `ego_yaw_rate` | the ego's own longitudinal speed and yaw rate at t + K | `v_max_policy` (10 m/s), `imu_gyro_scale` (5 rad/s) |
| `opp_present` | 1 where the nearest opponent is inside `overtake_range` (12 m) at t + K — a **logit**, trained with cross-entropy | — |

The labels are privileged (`F1VecEnv.future_labels`) and nothing the policy sees is built from them.
`opp_vlon` is deliberately *not* `privileged()[10]`: that column is `other.vx - ego.vx`, a difference
of two body-frame longitudinal speeds taken in two different frames, which is a serviceable
present-tense cue and a poor prediction target. Here both velocities go to the world frame and the
difference is rotated into one frame, so `opp_lon + dt * opp_vlon` is, to first order, where the car
will be. "Nearest" is recomputed at t + K, exactly as `privileged()` recomputes it at t — so in a
three-car field the target can change which car it refers to, and that is a floor on the achievable
MSE rather than something the head can learn away. The probe scores the identical target, which is
what makes the two numbers comparable.

**Why the recurrent state and not the trunk features.** Because the probe measures `h_t`, and the
thing being trained and the thing being measured have to be the same tensor or the evidence is about
something else. The head's input is `h_t` (the GRU's last layer after the step), never the action
features, so the gradient reaches the recurrence directly. With `--memory off` there is no such
tensor and it reads the trunk features instead, which keeps the feedforward ablation available.

**Zero init.** The head's output layer — weight *and* bias — starts at exactly zero, the same
construction `--cond` and the GRU projection use. The output feeds nothing else, so forward parity
is free; what the zero buys is that the output layer trains from the first update while everything
below it is starved for exactly one, so a warm start is bit-identical and the path still starts
moving immediately.

**Masking — say what is dropped.** A label exists for step `t` only if the state at `t + K` is
available and belongs to the same situation:

* the **last K steps of each `--horizon` chunk** carry no label. The label for step t is the state
  at t + K, and the chunk does not reach that far; the next chunk's states are not available during
  this update (the update runs between rollouts) and no second rollout and no extra simulator step
  is paid for them, so those steps are dropped rather than approximated. The chunk's own final state
  — the one the value bootstrap already visits — is recorded as the (T+1)-th label row, so `t = T−K`
  is the last labelled step: at the default `--horizon 32` and K = 20 that is **13 of 32 steps,
  41 %**. A longer `--horizon` raises it; the trainer prints the fraction at startup.
* a label is **never read across an episode boundary**, and in a race the boundary is any car of that
  race resetting, not just this one. An opponent that crashes is respawned behind the field in place
  without ending the learner's episode, so its pose half a second later is not the continuation of
  the motion the head was asked to extrapolate.
* where **no opponent is inside `overtake_range` at t + K**, the four opponent columns are masked
  (and the stored label is zeroed). `opp_present` always trains — it is the column that says whether
  the others mean anything, and a presence logit trained only on frames with a car in them could
  never say "no car".

A teacher-driven car's transitions carry no weight here either: the same `on_policy` weight the rest
of the PPO loss uses multiplies the mask.

**Logged.** `loss/aux_future_mse` is the scalar the coefficient multiplies (the mean of the six
mean-squared errors plus the presence cross-entropy) and `loss/aux_future/<target>` is each
component, plus `labelled_frac` (share of the minibatch that carried a label at all) and
`present_frac`. A total that falls because one easy column collapsed is not a head that learned the
opponent, and only the split says which happened. `--metrics-jsonl PATH` appends every logged dict
to a file, which is how a `--wandb disabled` run leaves a curve behind.

```bash
python3 -m f1sim.learn.ppo --name ppo_future --init "$FROZEN_ORIGINAL" \
  --memory gru --memory-hidden 128 --scan-channels memory,edges \
  --race-size 3 --opponent pool --opp-pool "$POOL" \
  --aux-grip 1.0 --aux-opp 1.0 --aux-future 1.0 \
  --action-mode plan --scan-stack 6 --hist-len 20 --envs 63 --horizon 32 --total 4_194_304
```

`--aux-future-k` (default 20) changes the lookahead and `--aux-future-width` (default 128) the
head's one hidden layer. A `k` at or past `--horizon` leaves not one labelled step and the run
refuses to start.

**Not deployed.** The head is training-only. `export.py` exports `(scan, proprio, hidden) ->
(action, hidden_next)` and nothing reaches the head from there, which
`tests/test_future_head.py::test_export_does_not_carry_the_future_head` asserts against the exported
ONNX graph. It costs the car nothing and it is not counted in the budget below.

**Measuring what it did.** [`probe_hidden`](#probing-the-hidden-state-probe_hidden) below.

### Probing the hidden state (`probe_hidden`)

That the auxiliary loss falls is evidence the *head* learned something. Whether the *state* carries
it — legibly enough for anything else to use — is a separate question, and it is the one the word
"belief" rests on. `probe_hidden` rolls a checkpoint out with traffic and opponent events on, freezes
the hidden states it produced, and fits a **linear** ridge read-out from `h_t` to the same privileged
targets at `t + k`, with a held-out split. Linear on purpose: a nonlinear probe measures the probe.

**`--current` is the narrower question, and the one to ask first.** It scores the nearest opponent's
*present* relative state — Δx, Δy, Δv_x, Δv_y in the ego frame — with R² **and** MAE,
presence-conditioned (rows with no car in range are excluded and the excluded share is reported) and
split near / mid / far so a few close cars cannot carry a pooled number. Δx and Δy measure *object
observability* and are readable from one scan; **Δv is the headline**, because a single range image
contains no velocity, so recovering it is exactly the test of whether anything in the network relates
two instants. Two flags make a ladder of representations comparable on it:

* `--stack-mode repeat` puts the newest frame in every slot of the stack — same weights, same
  proprio, 150 ms of temporal information removed and nothing else. The memory-off floor.
* `--probe-state trunk` probes a feedforward checkpoint as it is, reading the trunk features the
  action comes from, instead of warm-starting it into a GRU. The frame-stack row.

With a motion branch the probe reads the **whole** recurrent state, `[h_main | h_dyn]`, while the
auxiliary heads read only the `h_dyn` slice of it — one function apart, by design, and pinned by a
test.

The table this produces is fixed once and each arm adds a column:
`work/motion-memory/work/e1/table.py`, and `docs/research/motion-memory-2026-09-14.md` reports it.

```bash
python3 -m f1sim.learn.probe_hidden \
  frozen="$FROZEN_ORIGINAL" no_head=ppo_a_final.pt with_head=ppo_b_final.pt \
  --tracks real:blackbox2022_1 --steps 400 --envs 48 \
  --scan-channels memory,edges --memory-hidden 128 \
  --k 0,20 --out probe.json --md probe.md
```

* Each checkpoint gets its **own** rollout from the same seed: a different policy visits different
  situations, and replaying one policy's states through another's trajectory would measure neither.
* A checkpoint with **no memory** (the frozen original is feedforward) is warm-started into the GRU
  named by `--memory-hidden` / `--scan-channels`, so it has a state to probe. That is not a distortion
  of the baseline: the projection is zero, so the policy it drives with is bit-identical to the
  original's, and what the probe then reads is a *random recurrent feature map* fed by the original's
  own embedding — the honest "before any of this trained" row.
* `--k 0` is the second baseline: what the state knows about the **present**, which six LiDAR frames
  nearly determine anyway. A `k = 20` R² near the `k = 0` one is the interesting result; a `k = 20`
  R² near zero while `k = 0` is high says the state carries the present and not the future.
* The split holds out whole **env columns** — whole cars — never rows inside one trajectory. Two
  consecutive steps of one car are the same situation 25 ms apart, and a random row split reports how
  well the read-out interpolates inside a trajectory it has already seen, which is near 1 for almost
  any feature map. The ridge penalty is chosen on a slice of the training columns and never on the
  held-out ones.
* The opponent targets are scored only on rows where a car is inside `overtake_range` at `t + k` —
  the same rows the auxiliary loss weights. A negative R² means the read-out does worse on held-out
  cars than predicting their mean.

The table for the two smoke arms is in
[the research note](research/future-head-2026-09-14.md).

### The deployment budget

The car runs the policy at the LiDAR's 40 Hz, so one control step has 25 ms for scan preprocessing,
the policy forward, the iLQR tracker and publishing. The Jetson is not on the training machine, so
the rule this work is held to is a **ratio on this machine, measured the same way for both
networks** — a proxy, and quoted as one:

> CPU, single thread, batch 1, fp32, the fastest of several blocks of 200 iterations after
> warm-up (`torch.utils.benchmark`; the table records how many). The new actor's forward must
> stay within **1.5×** the frozen original's and its parameter count within **2×**.

The auxiliary heads are not in that measurement and are not meant to be: `--aux-grip`, `--aux-opp`
and `--aux-future` are training-only, the exported graph does not contain them, and the car never
runs them.

The fastest block, not one block: on a machine that is also training, a single 200-iteration block
measures the contention rather than the work.

```bash
python3 -m f1sim.learn.budget                      # the table, against the frozen original
python3 -m f1sim.learn.budget --hidden 256 --variants baseline,gru   # what the ceiling costs
```

Measured on this machine: the 128-wide default is 1.07× the frozen actor's forward and 1.20× its
parameters; the 256 ceiling is 1.14× and 1.48×. Both fit, and 128 is the default because it leaves
headroom for the Jetson to be slower than this CPU in ways a ratio on one machine cannot see.

`tests/test_memory_model.py::test_jetson_budget_ratio` asserts the rule. The measured table is in
[the research note](research/memory-policy-2026-09-13.md).

### Opponents: the situations, not the events

A race puts another car on the track. What the policy learns from it is not the car, it is the
*situations* the car puts it in — and until 2026-09-13 there was one. `--opponent teacher` gave a
raceline teacher at 0.6–1.0× that never looks at the learner; `--opponent policy` gave a copy of the
learner; the learner always spawned at the **back**. So the training distribution contained "a
slightly slower car ahead of me on the racing line", and nothing else. The measured failures in
traffic are the other situations: side-by-side contacts closing at +1.5 m/s and walls hit while
alongside or just behind the car being passed, with no contact ever following a scripted event
([failure attribution §5](research/failure-attribution-2026-09-13.md)).

Three flags widen it, and one tool says whether they worked.

#### `--opp-pool` — a population instead of an opponent

```bash
python3 -m f1sim.learn.ppo --race-size 2 --opponent pool \
    --opp-pool teacher,self,~/f1sim_runs/cl_origrecipe_legacy_s701/ppo_final.pt
```

One entry is drawn per race from the simulator's own generator, so a seed reproduces which opponent
the learner met in which race. An entry is a **checkpoint path**, `self` (the learner's own current
weights — that race is self-play) or `teacher` (the raceline teacher, carrying whatever
`--opp-events` are configured). Empty is off, and `--opponent pool` with an empty pool is refused
rather than silently run as something else.

A checkpoint is the one opponent that cannot be written down as a rule: it takes its own line
through a corner because its own network decided to, it defends its position because being passed
costs it reward, and it makes its own mistakes. Properties:

* Loaded once, `eval()`, gradients off — a pool is not a second training job. It costs one extra
  policy forward per entry per step over the whole env width (no `nonzero` compaction: selecting
  rows would be a device sync every step, and a recurrent entry has to advance anyway).
* An entry must already fit this env's observation and action space (`--scan-stack`,
  `--scan-stride`, `--hist-len`, `--action-mode`); a mismatch is refused by name rather than
  reshaped, because reshaping would put a half-reinitialised driver in the other car.
* `--opp-speed` can only *slow* a pool car, through its speed cap. Being overtaken by one is a
  question of the grid, not of a cap.
* A pool without `self` can never put a policy-driven car in an opponent slot, so the PPO buffers
  stay narrow there instead of being half masked out of every update.

#### Reactive behaviour — `defend`, `yield`, `line`, `oblivious`

The four names below join `brake, stop, shift, weave` in the same `--opp-events` list, and are a
different kind of thing: not timed events a car falls into, but **dispositions** drawn per
teacher-driven car per race, each with its own probability, each reading the learner's position
relative to that car every step.

| behaviour | what the opponent does | flags |
|---|---|---|
| `defend` | while a learner is within `--opp-defend-range` (0 = the env's 12 m `overtake_range`) **behind**, moves its line toward the side that learner is coming down — full amplitude inside `--opp-defend-full`, ramping to nothing at the far edge | `--opp-defend-prob`, `--opp-defend-offset`, `--opp-defend-range`, `--opp-defend-full` |
| `yield` | with a learner **alongside**, moves away from it | `--opp-yield-prob`, `--opp-yield-offset` |
| `line` | drives an out-in or in-out offset sweep through each corner, mode and amplitude drawn **per corner** — so the opponent is simply not on the line the learner's model of it assumes | `--opp-line-prob`, `--opp-line-offset`, `--opp-corner-kappa/-min-arc/-smooth` |
| `oblivious` | the `opp_follow_gap` slowdown is switched off: it does not brake for a car ahead or beside it. **This is the car that rams** | `--opp-oblivious-prob` |

"Alongside" is `--opp-alongside-lon` (0.8 m of body-frame longitudinal offset; the bodies are 0.58 m
long) and `--opp-alongside-lat`. The same definition is what `learn.opponent_census` counts, from
the same code (`gym_env.learner_view`) — a behaviour that reacts to one "alongside" and a census
that reports another is exactly the failure the census exists to rule out.

The three offset behaviours share one output, summed, then capped at `--opp-react-max` (0.45 m) and
rate-limited to `--opp-react-slew` (0.6 m/s), and then clamped by the lane's own free space — the
same budget the scripted offsets use, so a reactive offset can no more reach a wall than a `shift`
can. `oblivious` produces no offset; it removes the follow cap rather than raising a command through
it, so the invariant "an event never lifts an opponent over its follow-gap cap" is untouched.

Each is off at probability 0, and `--opp-events defend` without `--opp-defend-prob` is refused: a
named behaviour at probability 0 is the unflagged run wearing another run's name.

#### `--spawn-order` — where the learner starts

`behind` (the default, and every race trained before this), `ahead`, `alongside`, or `random` drawn
per race. It applies to a race whose other cars are not the learner itself; in a self-play race
every car is the learner, so only `alongside` is a different grid there.

* `ahead` is the only way the learner is ever the car **being overtaken**, and it needs a faster
  opponent to be one: `--opp-speed` above 1.0 is allowed and means a teacher driving above its own
  raceline profile. Measured, 1.0–1.3× costs no opponent wall collisions on these tracks; the
  profile already plans at the grip limit, so much above that is a car leaving the road, and the
  census reports opponent wall contacts so the ceiling is measured rather than assumed.
* `alongside` puts the cars abreast, which is where the contacts measured in traffic actually
  happen. The lateral separation is what the track's own distance field has room for at each car's
  spawn point, sized against the **rotated** footprint (the spawn yaw jitter is cut to
  `spawn_alongside_yaw` for an abreast row, since cars on a grid are lined up with the track); where
  two bodies do not fit, that race spawns staggered instead. Two cars spawned in contact terminate
  on step 1, so this is decided per race, not per car.
* `--race-size 3` works with all of it. The spawn arc is cumulative — each car sits its own
  `--spawn-gap` behind the car in front of it — rather than `rank × gap`, which scrambled a grid of
  three whenever the draws differed.

#### Measuring it: `python -m f1sim.learn.opponent_census`

Training against a behaviour and never measuring it is how a recipe gets adopted on a number that
does not cover the thing it changed. The events work before this one was accepted on "it fires":
37 events in 32 races. That is the wrong bar — an event that fires while the learner is 40 m away is
not a lesson. The census runs N races with a given configuration and reports, per situation, **how
many learner-seconds were spent in it**:

```bash
python3 -m f1sim.learn.opponent_census "$CKPT" --races 32 --steps 1200 --device cuda \
    --tracks 'real:blackbox2022_1,gen:control:1400' \
    --race-size 2 --opponent pool --opp-pool teacher,self,A.pt \
    --opp-events defend,yield,line,oblivious \
    --opp-defend-prob 0.5 --opp-yield-prob 0.35 --opp-line-prob 0.5 --opp-oblivious-prob 0.3 \
    --spawn-order random --opp-speed 0.6 1.2 --out census.json
```

It takes **the same flags as the trainer**, out of one shared parser group
(`learn/opponent_config.py`), so the configuration it measures is a configuration `ppo` can build.
The seven rows are behind a slower car, alongside, being overtaken, defended against, yielded to, an
oblivious car closing from behind, and two opponents in range; context rows below them name which
kind of opponent was nearest and which grids were drawn. Each row carries seconds, the share of
learner-seconds, and how many of the learner's cars ever saw it — a situation that happens a lot in
one race and never in the other thirty-one is not a distribution either.

"Learner-seconds" are seconds driven by a car the policy drives *and* whose transitions the update
would use, which is slot 0 in `teacher` mode and every self-play car in `mixed` and pool-with-`self`.

The same rollouts carry a second report, which is analysis and not a reward change: the distribution
of the `car_proximity` reward per step and of the body-to-body lateral gap while alongside. Recipe A
charges `--car-proximity-penalty 0.8` below `--car-safe-gap 0.9`, and the question is whether that
term produces any signal in the situations that matter. See
[the research note](research/opponent-diversity-2026-09-13.md) for the measured table.

#### Timed events (`brake`, `stop`, `shift`, `weave`)

Unchanged, and still what `--opp-event-rate` drives: the expected number of events per teacher
opponent per 10 s. Events never overlap, so the realised rate is a little below the flag (at rate
1.0 with ~1.7 s events, about 0.85).

| event   | what the opponent does | drawn from |
|---------|------------------------|------------|
| `brake` | commands `k` × its profile speed for `d` seconds, then resumes | `--opp-brake-scale` (k), `--opp-brake-time` (d) |
| `stop`  | commands 0 for `d` seconds — the stalled car | `--opp-stop-time` |
| `shift` | tracks the raceline offset by `o` m: ramp in, hold, ramp out — a lane change / blocking line | `--opp-shift-offset` (\|o\|, sign drawn separately), `--opp-shift-hold`, `--opp-shift-ramp` |
| `weave` | sinusoidal lateral offset | `--opp-weave-amp`, `--opp-weave-period`, `--opp-weave-time` |

`--opp-event-margin` is the body-to-wall gap kept when any offset — scripted or reactive — moves a
car off the line.

#### The properties all of it rests on

* **Off is off.** With no `--opp-events`, no `--opp-pool` and `--spawn-order behind`, nothing here
  is stepped and nothing is drawn from the simulator's generator: a rollout is bit-identical to the
  same rollout on the code before any of it existed, for `--opponent policy`, `teacher` and `mixed`
  alike, and so is a rollout with the timed events on (`tests/test_opponent_events.py`,
  `tests/test_opponent_diversity.py`). Every checkpoint, benchmark number and report on this branch
  was measured on that path.
* **Behaviour is teacher-only.** A pool checkpoint drives itself; the scripted and reactive layers
  act on teacher-driven cars, never on a car the policy is driving, so no PPO transition ever
  carries an action the policy did not produce. `--opp-events` with no teacher-driven car anywhere
  (solo, `--opponent policy`, or a pool with no `teacher` entry) is refused.
* **No offset can reach a wall.** Scripted plus reactive are summed and clamped per raceline point
  against the track's distance field (free space − car half-width − `--opp-event-margin`).
* **Nobody spawns in contact.** Measured over 512–528 spawns per grid per track at
  `--race-size 2` and `3`: zero. With `--procedural-obstacles` on, the grid is tested against the
  drawn layout as well as against the lane — a crate standing where an abreast grid wants to be
  makes that race start staggered, because the spawn's prop rejection replaces a blocked pose with a
  *centerline* one and two cars pulled onto one line are two cars in contact.
* **Each step, `info["opp_event"]`** reports `id` (0 = none, otherwise the 1-based index into
  `brake, stop, shift, weave, defend, yield, line, oblivious`), `time_left`, the `offset` in metres,
  and two bitmasks — `react` (which reactive behaviours are acting right now) and `disposition`
  (which this car was given for the race) — for a viewer, a logger or the census.

**Evaluating against them.** `python3 -m f1sim.learn.evaluate --race-size 2 --opponent teacher
--opp-events brake,stop,shift --opp-event-rate 3` reports the traffic metrics (passes, pace against
the opponent, engagement time, contacts) under `traffic` in every result. The benchmark's **T
(traffic) family**, frozen as suite v2.1, is the scored version: brake / stop / shift at 3.0 per
opponent per 10 s is one of its four scenarios, on five held-out maps at two friction levels — see
[the benchmark's traffic family](benchmark.md#the-t-traffic-family).

### Obstacle layouts redrawn at every reset

Every training track's obstacle layout is fixed. `+rlobs`, `+obs`, `+pinch` and `+hard<seed>` are
rasterised into the occupancy grid once, at load, and shared by every environment driving that map
for the whole run. A layout that never changes can be learned, and the progress reward pays for
learning it: speed through a *known* layout is worth exactly what speed through a *seen* one is
worth. On an unseen layout the same speed is a collision, and that is the mechanism the held-out proxy
shows: right after a warm start the policy drives slightly slower on unseen maps and completes more
of them; as training goes on its speed returns to the original's and its completions fall back
(`docs/research/procedural-obstacles-2026-09-13.md`).

`--procedural-obstacles` draws a new layout for each environment at every reset:

```bash
python3 -m f1sim.learn.ppo --procedural-obstacles 1.0 --procedural-density 1.0
```

| flag | what it does |
|---|---|
| `--procedural-obstacles FRAC` | share of resets that get a layout. `0` (the default) is off, and off is bit-identical to a run before the feature existed |
| `--procedural-density PER10M` | patterns per 10 m of lap (default 1.0) |
| `--procedural-max-props N` | prop slots per environment; `0` sizes it from the density and the longest lap. Every slot costs the beam tracer one pass per step, so this is the cost dial |
| `--procedural-raceline-margin M` | free space kept either side of the raceline, beyond the car's half-width (default 0.25 m) |

It composes with the opponent work above: the layout is drawn before the grid is laid out, one
layout per race (the cars of a race see each other, so they have to see the same crates), and
`--spawn-order alongside` consults it — see the last-but-one property in the previous section.

The obstacles are the same six patterns `#hard:*` draws — gate, diagonal, chicane, apex, cluster,
scatter — at the same sizes, including the small objects. They are placed as **props**
(`f1sim.props`: boxes, crates, a drum, a post), which the LiDAR and the collision test handle
analytically, so nothing is rasterised and a redraw is a batched write of a few dozen numbers per
environment.

**Passable by construction.** `#hard:*` proves a layout passable after the fact: it erodes the free
space by 0.25 m and checks that the lane still connects across the pattern, redrawing when it does
not. That loop cannot be batched onto a GPU. Here the gap is guaranteed before anything is placed —
every piece of a row sits inside a band of width `span` measured from one wall, `span` is at most
`width − g` with `g ≥ max(1.2 m, 0.55 × width)`, and `width` is the narrowest the lane gets anywhere
the pattern reaches. Measured over 1000 draws on the catalogue maps, the realised free space is
**≥ 1.20 m in 100 % of layouts** (worst 1.288 m, median 2.453 m).

**The teacher opponents are never routed through one.** The raceline teacher is pure pursuit on a
line built from the occupancy grid; props are not in the grid, so it cannot see one and will not
steer round one. When a teacher is installed, the gap is additionally required to contain the band
the raceline occupies, widened by the car's half-width and `--procedural-raceline-margin`. The cost
is worth stating plainly: **with opponents, the gap is always where the racing line is.** The layout
still moves at every reset — where the patterns are, which they are, which wall is blocked, what
size the pieces are — but a policy that could already find the racing line would find the gap. With
`--race-size 1` there is no teacher and no corridor, and the gap can be anywhere across the lane.

**What does not see these props.** Everything that reads the occupancy grid or its distance field:

* `--proximity-penalty` (wall gap from the EDT) and `--plan-clearance-penalty` (plan points against
  the EDT) do not price a prop; a car alongside a crate is charged as if the lane were empty;
* `StepResult.wall_dist`, and so the `wall_dist` column of the privileged vector, is the distance to
  the nearest grid wall only;
* the raceline and the teacher's speed profile were built before the layout existed.

The collision test, the LiDAR, and the spawn rejection *do* see them. That is the set that makes the
layout something the policy has to look at rather than something it can be told about.

**Races share one layout.** The cars of a race drive the same track and see each other; giving them
different crates would have one collide with a box another cannot see. A race redraws when it resets
as a whole, and a single car respawning behind its mates keeps the layout the race is running —
exactly as it keeps the race's track.

**Cost.** Measured on an RTX 4060 Ti at `--envs 256` on `blackbox2022_1` (149.6 m, 15 patterns,
52 prop slots), `gen:control:1400` and `scene:scene_0912_2344` — see the research note for the
table. The dominant term is the beam merge against the prop prisms, which is why both
`prop_math.ray_prisms_hits` and `prism_contacts` are `torch.compile`d on CUDA: eagerly the merge is
a kernel launch per elementwise op per slot (84.4 ms a call at 52 slots), fused it is 3.34 ms. The
fused result is **not** bit-for-bit: which beams hit is identical (zero disagreements over 276 736
beams), but reassociated float32 moves a range by up to 1.9 um. This also applies to the `+props`
catalogue path, which now goes through the same compiled primitives.

**The console does not draw them.** The environment page renders a track's `props`, and these
belong to the environment rather than to the map, so a procedural layout is invisible there; nothing
under `f1sim/viewer/` was touched. To *see* one of these layouts, draw the same patterns into a map
with `#hard:<seed>` and open that.

**Evaluation is untouched.** `f1sim.learn.evaluate` and the frozen benchmark suites have no
procedural option and never draw one; `EnvConfig.procedural_obstacles` defaults to 0 and every
evaluation path takes the default. A layout redrawn per reset is not a thing a frozen suite can
contain, and adding the flag there would make the score depend on it.

## Evaluation

```bash
CKPT="$HOME/f1sim_runs/ppo_v1/ppo_final.pt"
python3 -m f1sim.learn.evaluate "$CKPT" --per-track --protocol trials --envs 64 --speed-cap 4
python3 -m f1sim.learn.evaluate --teacher --action-mode plan --per-track --protocol trials

# the interactive teacher's own ceiling, against reactive opponents (`--teacher-kind`)
python3 -m f1sim.learn.evaluate --teacher --teacher-kind interactive --action-mode plan \
    --race-size 3 --opponent teacher --opp-speed-range 0.6 1.15 \
    --opp-events brake,stop,shift,defend,yield,line,oblivious --opp-event-rate 1.0 \
    --opp-defend-prob 0.3 --opp-yield-prob 0.2 --opp-line-prob 0.3 --opp-oblivious-prob 0.1

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
the budget comparable across tracks of different length. `--tracks heldout` (`eval` is the same
list) uses the held-out set. `--per-track` keys are the **loader** names, because they are data that
older reports are already keyed by.

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
| `estimated` | an estimate from causal onboard signals only |

Any of them may additionally carry either composable layer, written as a suffix: `+clearance` (the
plan geometry) and `+tcs` (the traction guard). Both are described below, and both can be worn at
once — the suffixes are written in the order the car meets them, so the fullest composite reads
`fixed_low+clearance+tcs`.

`oracle` is a **privileged reference for this controller** — what perfect friction knowledge buys
*given this speed-envelope and bound derivation*. It is not a mathematical upper bound on achievable
performance: a different controller could use the same knowledge better, and nothing here proves the
derivation optimal.

`legacy` and `fixed_low` are different things: the first applies no explicit limit at all, the second
applies a constant one. `oracle` cannot run on a car.

### `+clearance` — the plan kept off what the LiDAR can see

`clearance` is a **composable** arm and a geometry clamp, exactly as `fixed_low` is a friction
clamp: a runtime layer that needs no training. Each control step it turns the current LiDAR frame —
and nothing else — into a coarse occupancy grid in the car's own frame, builds a distance field on
it, and adjusts the policy's plan until every point of it keeps a stated **body-edge** margin
(default 0.20 m, i.e. 0.34 m from a plan point to the nearest return) from anything the scan saw.
Where the corridor allows it the plan is bent away from the nearer side; where it does not, the
plan's speed targets come down so the car arrives slow.

```
python3 -m f1sim.learn.evaluate CKPT --controller clearance --action-mode plan ...
python3 -m f1sim.learn.evaluate CKPT --controller fixed_low+clearance ...
```

**Why it exists.** Measured on the eight held-out proxy tracks on 2026-09-13: the privileged
raceline teacher completes 30–32 of 32 trials through the same iLQR tracker the policy uses, while
the policies complete 7–24. At the policies' collisions the **policy's own last plan** had a median
minimum body-edge clearance of 0.00–0.06 m; 85 % of collisions had a plan margin under 0.10 m and
49 % of the plans passed *through* occupied cells, while the car still had ~0.2 m of clearance 25 ms
before impact. The tracks are feasible and the tracker can follow safe plans; the policy plans with
no margin. See [the research note](research/clearance-arm-2026-09-13.md).

**Where it sits, and why that is the whole design.** On `PlanTracker._plan_hook`, which runs
*before* the action is decoded — not on the solver. `tracker.last_ref` is built inside `mpc.solve`
from the action, so it is both what the iLQR follows and what the attribution script measures; an
arm that changes the action is therefore measured, and executed, as the plan it produced. It also
makes composition order-free: `fixed_low` binds `tracker._solver` and this binds
`tracker._plan_hook`, so neither can overwrite the other and the friction envelope is computed on
the adjusted geometry whichever was installed first.

**What it may see.** The current scan and nothing else. No map, no pose, no `track.edt` — the
privileged distance field is what the *evaluation* measures with and is not on the runtime path. A
bearing with no return contributes nothing and everything outside the grid reads as free: the arm
acts on what the sensor saw rather than braking for the 90° behind the window. That is what lets
`f1sim_ros/policy_node.py` run this identical code off `/scan`.

**What it may do**, and only these two, both one-sided:

- **bend** — one curvature offset added to the knots inside the tracker's own horizon and tapered to
  zero beyond it, so the tail curvature (and with it `fixed_low`'s speed envelope over the tail) is
  untouched. Thirteen candidate offsets are scored on the mean, over the window, of their body-edge
  clearance saturated at the margin — with everything from a candidate's first *contact* counted as
  void, because the distance field is unsigned and a plan a metre past a wall would otherwise read
  as a metre of free space. The smallest bend that meets the margin wins.
- **slow** — the plan's two speed targets, through a backward braking pass at 3.3 m/s², which is
  what the command chain was measured to deliver at the bottom of the friction range rather than the
  5.0 m/s² the tracker is allowed to command. Under `fixed_low` the solver is clamped tighter still
  (2.97 m/s² on a straight plan at µ 0.734, and less in a corner), so under the composite the pass
  is optimistic about how late it may start slowing. It is left optimistic deliberately: reading the
  other layer's bound would make the installation order matter, which is the one property that makes
  the two composable. The cap is a target re-issued at 40 Hz and tightening as the obstacle nears,
  and what is actually commanded is the tracker's to bound — a planning approximation, like the grip
  envelope, and not a stopping guarantee.

It never raises a speed and never straightens a plan the policy bent, and a plan that already keeps
the margin is returned **bit-identical** — "the arm did nothing" is a fact about the action, not an
approximation of one.

| knob | default | meaning |
| --- | --- | --- |
| `margin` | 0.20 m | body-edge clearance defended; 0.34 m centre-to-return with `body_radius` 0.14 m |
| `cell` | 0.06 m | occupancy cell; a return is binned to the cell containing it, so the position error is ≤ 0.03 m per axis and unbiased |
| `d_clip` | 0.60 m | the distance field saturates here, which bounds the transform at 10 offsets per pass |
| `max_shift` | 0.60 m | lateral authority at the evaluation horizon — an adjustment, not a replan |
| `horizon_s` | 0.60 s | the far end of the arc the arm is responsible for: the tracker's own `N·dt`. Beyond it the plan is replanned before it is ever executed |
| `s_min` | 0.50 m | and the near end. `base_link` is the rear axle and the nose is ~0.48 m ahead of it, so a plan point closer than this is inside the car's own footprint — its clearance is a fact about where the car already is, which no bend can move and no speed can change |
| `a_brake` | 3.3 m/s² | the deceleration the backward pass assumes, measured through the whole command chain at low grip |

**Cost.** The grid is 5 304 cells; the distance field is Felzenszwalb's separable decomposition of
the *exact* squared transform, written as a fixed number of shifted minima — no chamfer
approximation, no data-dependent control flow, so it is as graph-safe as the rest of the path.
Measured under the deployment budget's own protocol (CPU, single thread, batch 1, fp32, the fastest
of several blocks): **1.25 ms of the 25 ms control step** — 0.48 ms for the occupancy and its
distance field, 0.77 ms for the candidate search and the cap.

```bash
python3 -m f1sim.learn.budget --clearance          # the number above, on this machine
```

`controller/clearance_*` metrics (the fraction of steps bent and slowed, the mean shift and speed
cut, and the plan margin before and after) are logged alongside the tracker arm's.

**Not trained under.** Training with the arm in the loop is deliberately not done — the grip clamp's
lesson was that a policy trained under a clamp learns to lean on it (faster laps, worse avoidance),
so the recipe is train legacy, deploy clamped. Modules: `learn/clearance.py`,
`tests/test_clearance.py`, `tests/test_policy_node_clearance.py`.

### `+tcs` — the car's traction guard, inside the loop

`tcs` is a **composable** arm: it does not change what the tracker plans under, it shapes the speed
command between the controller and the VESC. So it is written as a suffix and can be worn on top of
any of the four above — `--controller tcs` is the legacy tracker plus the guard, and
**`--controller fixed_low+tcs` is the deployment default**, because `fixed_low` is the control arm
the benchmark roster runs and the guard is what the car ships.

What runs is `f1sim_ros/f1sim_ros/traction.py` — the same class, unmodified, that
[`docs/ros2.md`'s traction guard](ros2.md) describes and that `scripts/replay_traction.py` validated
over the 22 real recordings. It is fed the simulated sensors and nothing else: the ERPM-quantised,
jitter-stamped wheel speed from `StepResult.odom` / `odom_t`, the last IMU sample's longitudinal
acceleration, and the emulated `/sensors/core` motor current. Its thresholds are the ones it uses on
the car, and they travel inside the checkpoint (`experiment.controller.traction.params`) so a
consumer can refuse a mismatched pair.

```
python3 -m f1sim.learn.ppo --controller fixed_low+tcs --action-mode plan ...
python3 -m f1sim.learn.evaluate CKPT --controller fixed_low+tcs --wheel-model on ...
```

**It needs `vehicle.wheel_model`.** With the switch off the simulated wheel speed *is* the body
speed, the residual the detector keys on is identically zero, and the arm would be a `legacy` run
wearing another arm's name — so it refuses to construct rather than run silently inert.

**Cost.** The guard is host Python over scalars: one `update` + `shape` pair per car per control
step, plus one device→host transfer to fetch its four inputs and one back to return the shaped
speed. Nothing about it is inside `sim._roll`, so it is outside the CUDA-graph fastpath by
construction; the two transfers are a synchronise per step, which on the `graphs` backend is the
cost that matters rather than the Python. Measured numbers are in
[the wheel-model note](research/wheel-model-2026-09-13.md).

`controller/tcs_*` metrics (active fraction, lock and spin counts, the largest release and cap) are
logged alongside the tracker arm's.

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

Modules: `learn/grip_control.py`, `learn/grip_estimator.py`, `learn/grip_runtime.py`,
`learn/clearance.py`, each with its own test module.
