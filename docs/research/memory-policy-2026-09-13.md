# A policy with memory, inside the Jetson budget (2026-09-13)

Worker note for branch `feat/memory-policy`, base `ae1f4df`. What this is: the observation and the
architecture changed, warm-started from the frozen original, with the deployment budget stated and
measured. What this is **not**: a result. The training here is a smoke — a proof that the pipeline
runs end to end — and the numbers in the last section should be read as "it trains", not as "it is
better".

## Why

`docs/research/free-run-selection-2026-09-13.md` and `run-recipe-free-2026-09-13.md`: three
finetunes of the frozen original, with more obstacles, opponent behaviour events, a looser KL leash,
a higher learning rate and a plan-clearance penalty, all score the same on the held-out obstacle and
traffic proxies — flat within noise from the first checkpoints. A crash map on the user's held-out
scene put 11 of 16 crashes on one row of boxes, entered at 4–7 m/s.

The observation is six LiDAR frames (150 ms at 40 Hz) and a 20-row proprio history. Two things
follow directly:

* a box that has left the 270° scan window is **gone**: there is no state in which it survives;
* a row of boxes with cracks between them presents, beam by beam, as a row of openings. The stem's
  first layer is a stride-2 7-tap convolution, so a two-beam crack is inside one tap.

So the lever left is the observation and the architecture. Two changes, both off by default, both
warm-started so that step 0 is bit-identical to the policy they extend.

## What was built

### 1. A recurrent actor (`--memory gru`)

A GRU over the actor's per-step trunk embedding — the scan stem's 256 features concatenated with the
128-wide proprio embedding, 384 in — with hidden size 128. Its output enters the first MLP layer's
**preactivation** through a bias-free projection initialised to zero, which is the construction
`Actor.cond` already uses, and for the same two reasons:

* **forward parity**: the term added is exactly `0.0`, so a checkpoint loaded into a memory actor
  produces bit-identical actions, aux-head outputs and values for any input and any hidden state.
  Nothing the original could do is lost at the start.
* **gradient flow**: `dL/dW = delta_pre · y^T` is nonzero as soon as the preactivation has a
  gradient, so the path trains from the first update rather than starving.

`n_stack` is untouched: the six-frame stack remains the input and the memory extends the window past
it.

**What the GRU is fed.** The contract says "over the per-step stem embedding"; what it is given is
the per-step *trunk* embedding — the stem's 256 features **concatenated with the proprio
embedding**, which is exactly the vector the first MLP layer sees. Deliberately: a memory of what
was seen is only useful together with how fast the car was going and what it had been commanded
when it saw it, and that is what the proprio half carries. Feeding the stem's output alone would
make the recurrence blind to the car's own state at the moment it is remembering.

**GRU, not LSTM.** One state tensor rather than two — and that tensor has to be carried by the ROS
node's callback state, a static CUDA-graph buffer in the viewer worker, the benchmark adapter and
two scoring paths, so halving what travels is worth more here than it usually is. It is also about a
quarter fewer parameters at the same width, and there is nothing the extra gate is needed for over
the 32-step truncated-BPTT horizon (0.8 s at 40 Hz) this trains on. `ScanStem`'s optional temporal
encoder is already an `nn.GRU`, so the export path is the exercised one.

**The critic gets its own.** It already carries its own stem and it reads the privileged vector the
actor must never see; sharing the recurrence would be the one place a value gradient reached the
actor's trunk. And the actor's forward is what the deployment budget measures, so a critic that is
never exported costs the car nothing. `--memory-critic none` leaves it feedforward.

### 2. Two zero-cost scan channels (`--scan-channels`)

Functions of the scan the env already emits, computed on the **policy** side of the observation
boundary (`learn/obs.py`), so the simulator, a checkpoint's recorded `extra["spec"]` and every
consumer of that spec are untouched. Each is one more row on the scan's channel axis, appended after
every column the original had and zero-initialised in the first convolution, so a warm start stays
bit-identical.

* **`memory`** — the closest return seen at each bearing recently, relaxing back toward "no return"
  with a 2 s time constant (`--scan-memory-tau`). Explicit cheap memory the GRU does not have to
  learn. Bearings are the car's own and are deliberately **not** motion-compensated: a LiDAR-only
  policy has no pose and no odometry it is allowed to use, so a box that leaves the window leaves a
  fading trace at the bearing it left by, not a correctly transformed position. That trace is still
  a signal ("something was there, recently, that way"); the GRU is the half that can learn what it
  means.
* **`edges`** — `|r[i] − r[i−1]|` per beam, so the crack between two boxes reads as two
  discontinuities a few beams apart while the gap the lane leaves reads as one, at full beam
  resolution before any stride touches it.

Both are episode-scoped in the same way the hidden state is: the occupancy memory is cleared exactly
where the hidden state is cleared, because they are the same claim about what the car has seen.

### 3. Recurrent PPO (`ppo.py`)

* The rollout stores each env's hidden state at the start of every horizon chunk, carries the state
  across steps, and clears the rows whose episode ended (`term | trunc`) — after the terminal
  observation's value has been read, so the truncation bootstrap belongs to the episode that ended.
* The update replays whole env chunks (truncated BPTT over the 32-step horizon) and masks the hidden
  state back to zero at every boundary inside the chunk, exactly where the rollout reset it.
  Minibatches are therefore env chunks: `--minibatch` stays a sample count and the chunk count is
  that divided by `--horizon`.
* The convolutional stem runs once for the whole (steps × envs) block; only the GRU and the two
  small MLP halves walk the steps. That is what keeps a recurrent update near the cost of a
  feedforward one.
* The hidden state is kept in float32 whatever `--amp` does. It is carried for a whole episode, so
  accumulating it in bf16 would compound every step's rounding into the one tensor the policy reads
  back; the GRU's own arithmetic is still autocast, because that is one step deep.
* The KL leash's reference stays the **feedforward** original: the frozen copy is evaluated with the
  recurrence switched off (`Actor.feedforward_dist`), and the run refuses to start if that copy's
  memory projection is not still exactly zero — the same guard the conditioning arms already had.
* `--memory off` is byte-identical. `tests/test_ppo_memory.py` holds the extracted loss to
  `tests/data/ppo_loss_oracle.json`, recorded from the inlined update as of `ae1f4df` before any of
  this existed, and asserts equality with `==`, not `approx`.

### 4. The hidden state in every inference path

| path | how |
|---|---|
| `common.rollout_metrics` | a stateful `policy_fn`; cleared at the seeded reset and per row at every boundary |
| `evaluate.py` (`--per-track`, both protocols) | the same `policy_fn`; the trials protocol clears at its own reset too |
| benchmark `model_adapter.policy_for` + `runner.run_cell` | cleared at the cell's single seeded reset (a cell drives one trial per row) and per row at every boundary |
| viewer worker (`learn/watch.actor_runner`) | `MemoryActorRunner`: the hidden state is ONE tensor, updated in place, so `torch.compile(mode="reduce-overhead")`'s graph buffers cannot clobber it; falls back to eager and sets `fell_back_to_eager` (which the worker already reports as `actor_fell_back`) |
| `viewer/graph_fastpath.graph_actor_step` | an explicit CUDA graph of one actor step with the hidden state as a **static input buffer** every replay copies into; built by the caller during a session build via `learn/graph_runtime.prepare_actor_graph`, never lazily mid-step, because a failed capture is fatal to the process and only the build is prepared for that |
| `f1sim_ros/policy_node.py` | carried across `/scan` callbacks; cleared on a scan-stream restart and on a `std_msgs/Empty` message on the `/f1sim/reset` topic, which `bridge_node` and `vesc_sim_node` now publish after they reset |

The feedforward entry points (`Actor.forward`, `.dist`, `.forward_all`, `Critic.forward`) **refuse**
a recurrent module rather than running it from zeros. A recurrent policy restarted every step looks
exactly like a working one — same shapes, same magnitudes, plausible driving — so an unconverted
caller has to be an error.

The viewer worker's env and its actor callable are built in two places that never meet, inside
`f1sim/viewer/**`, which this branch does not own. So `common.make_env` announces episode boundaries
to a weakly-held listener registry (`learn/memory`) and the actor runner listens. The registry is
empty for training, evaluation and the benchmark, which all thread the state through by hand.

## The budget

The car runs the policy at the LiDAR's 40 Hz (`params.py` `control_rate`), so one control step has
**25 ms** for scan preprocessing + policy forward + the iLQR tracker + publishing. The Jetson is not
on this machine. The rule this work is held to is therefore a ratio measured the same way for both
networks on this machine, and it is a **proxy** — the absolute milliseconds below are this desktop's
CPU, not the car's:

> CPU, single thread, batch 1, fp32, the fastest of several blocks of 200 iterations after
> warm-up (`torch.utils.benchmark`; the table records how many). Forward within **1.5×** the
> frozen original's; parameters within **2×**.

The contract says 200 iterations. What is reported is the fastest of several such blocks: one
block is one sample, and on a machine that is also running a training job it measures the
contention rather than the work — under load, one block put the same network anywhere from 1.11×
to 1.76×. The minimum is the stable estimate, and both networks are measured the same way, which
is what a ratio needs.

Baseline `~/f1sim_runs/_baselines/frozen_original_48cc698f.pt`: actor 1 168 164 parameters, critic
1 086 353, total 2 254 517. (CONTRACT.md's "2.25 M actor" is the whole actor-critic; the actor alone
is 1.17 M. Both ratios are reported, and both have to pass.)

| variant | actor forward [ms] | extra channels [ms] | step [ms] | forward ratio | step ratio | actor params | ratio | total params | ratio |
|---|---|---|---|---|---|---|---|---|---|
| frozen original (feedforward) | 1.841 | 0.000 | 1.841 | **1.00×** | 1.00× | 1,168,164 | **1.00×** | 2,254,517 | 1.00× |
| + GRU 128 | 1.972 | 0.000 | 1.972 | **1.07×** | 1.07× | 1,398,308 | **1.20×** | 2,714,805 | 1.20× |
| + GRU 128, scan `memory` | 2.030 | 0.019 | 2.049 | **1.10×** | 1.11× | 1,398,644 | **1.20×** | 2,715,477 | 1.20× |
| + GRU 128, scan `edges` | 2.042 | 0.017 | 2.059 | **1.11×** | 1.12× | 1,398,644 | **1.20×** | 2,715,477 | 1.20× |
| + GRU 128, both channels | 2.030 | 0.027 | 2.057 | **1.10×** | 1.12× | 1,398,980 | **1.20×** | 2,716,149 | 1.20× |

Measured as: CPU, 1 thread, batch 1, fp32, the fastest of 7 block(s) of 200 iterations after warm-up (torch.utils.benchmark). Limits: forward **1.5×**, parameters **2.0×**. Everything here is inside both, with room — the whole network part of a control step is 2.1 ms of the 25 ms the step has, and the two extra channels are 0.03 ms of that.

The contract's ceiling is a hidden size of 256, and that fits too — measured the same way,
`--memory-hidden 256` is **1.14×** the frozen actor's forward (1.17× for the whole step with both
channels) and **1.48×** its parameters, against the 1.5× / 2× limits. 128 is the default because it
leaves that headroom for the Jetson to be slower than this CPU in ways a ratio on one machine
cannot see; `python -m f1sim.learn.budget --hidden 256` re-measures it before anything is trained.

`python -m f1sim.learn.budget` reproduces it; `tests/test_memory_model.py::test_jetson_budget_ratio`
asserts the rule.

## The smoke

**A proof the pipeline runs end to end, not a result.** 128 updates is about a thirtieth of the
run this feeds, on three tracks, on 64 environments. Nothing below distinguishes the arms and
nothing below is meant to.

The recipe is `cl_free_s701`'s, verbatim
(`work/learning-next/free-20260913/launch.sh`), with `--envs 64 --total 262144` and the contract's
three tracks, opponent events on (`brake,stop,shift,weave` at 1.0 per 10 s), warm-started from
`frozen_original_48cc698f.pt`. The three arms differ in **nothing but** `--memory` /
`--scan-channels`, and each was launched only after `nvidia-smi` showed the shared GPU under the
contract's 4 GB bar. Wall clock: 14 / 13 / 14 minutes, all inside the 20-minute limit.

One of the three tracks, `real:map16x07+hard1`, is dropped before training by the raceline builder
(clearance 0.050 m against the 0.191 m the car needs) because the smoke races two cars and teacher
opponents need a raceline. It is dropped identically for all three arms, and the proxy below —
which runs solo and needs no raceline — scores it anyway, so for this smoke it is an unseen track.

### It trains

| arm | updates | kl_ref first / last / max | coll/km, updates 1-32 | coll/km, updates 97-128 | progress m, 97-128 | median steps/s | non-finite loss or crash |
|---|---|---|---|---|---|---|---|
| feedforward (control) | 128/128 | 0.002 / 0.557 / 0.691 | 53 (32–72) | 17 (9–38) | 52 (27–78) | 446 | none |
| --memory gru | 128/128 | 0.001 / 0.405 / 0.453 | 49 (31–72) | 24 (11–35) | 42 (28–62) | 433 | none |
| --memory gru + scan channels | 128/128 | 0.002 / 0.441 / 0.528 | 54 (31–109) | 15 (8–27) | 58 (37–81) | 420 | none |


Read that as: every arm ran all 128 updates to completion; **no non-finite loss and no non-finite
gradient norm** (with `--memory`, `ppo` raises on either rather than letting `clip_grad_norm_`
propagate a NaN, so a finished run *is* that evidence); `kl_ref` moves from ~0.001 at the first
update to 0.4–0.6, which is the leash doing its job on a policy that is changing; and the collision
rate falls by a factor of two to three over the run on every arm. The per-update columns are
medians with the interquartile range, over a window of updates, because a single update's
collisions/km is a ratio over the handful of episodes that ended in it.

Two things worth taking from the table beyond "it runs":

* **Update 1 is identical across the arms** — `rew/step 0.052`, the same collision count, the same
  progress. That is the bit-identical warm start showing up in the training log: on the first
  rollout the recurrent actor with both extra channels *is* the frozen original. They diverge from
  update 2, when the auxiliary heads' gradient (which `--critic-warmup` does not freeze) starts
  moving the trunk.
* **Truncated BPTT costs about 3 % of training throughput** — 446 → 433 median env steps/s for the
  GRU, 420 with the channels on as well. The stem runs once for the whole (steps × envs) block and
  only the GRU walks the steps, which is what keeps it that cheap.

### The held-out proxy

First-attempt trials (`evaluate --protocol trials --per-track`), solo, 48 environments per track, a
3-lap-equivalent budget, the speed cap the smoke trained at, one fixed seed for every arm, on the
same three tracks:

| checkpoint | `1400+hard2` | `map16x07+hard1` | `scene_0912_2344` | all | coll/km |
|---|---|---|---|---|---|
| frozen original (what every arm started from) | 31/48 (9) | 0/48 (117) | 13/48 (14) | **44/144** | 20.9 |
| smoke: feedforward | 41/48 (3) | 0/48 (122) | 32/48 (4) | **73/144** | 11.4 |
| smoke: `--memory gru` | 38/48 (5) | 0/48 (122) | 28/48 (6) | **66/144** | 13.8 |
| smoke: `--memory gru` + channels | 35/48 (6) | 0/48 (134) | 29/48 (6) | **64/144** | 14.3 |

Cells are completions / trials, with collisions per km in brackets.


Read the columns, not the totals, and read `map16x07+hard1` as a hazard rate only: it is the track
whose raceline does not fit the car, nobody completes a trial on it, and the smoke did not train on
it.

* Every trained arm is well clear of the frozen original it started from, on the two tracks it
  trained on. That is 128 updates of fitting two tracks, which is what it looks like — not
  generalisation.
* **The arms are not distinguishable.** 73 vs 66 of 144 completions is about 0.8 of one standard
  error on a binomial of that size (≈ 8.5 trials); the collisions/km spread is the same story. With
  one seed, one 128-update run and three tracks, it could not have been otherwise. What this
  answers is "does a recurrent, scan-augmented policy train, score and drive through the same
  pipeline as the feedforward one" — yes, at rc=0, on every path.
* `--memory gru` was scored here through `evaluate.py --per-track` with the hidden state carried and
  cleared per trial, so this table is also the end-to-end evidence for that inference path.

## What is not here

* No claim that memory helps. That needs the full run launched from the merged tree and the
  held-out suite, not a 128-update smoke on three tracks.
* No answer to which scan channel helps. The smoke measured it and the arms are inside one standard
  error of each other; a winner read off that would be invented.
* No motion compensation in the occupancy channel, by design (see above).
* No change to the reward, and no change to `n_stack`.
* The viewer's actor is not captured as a CUDA graph by default — the capability is built and
  tested, but a failed capture is fatal to the process and the worker only handles that during a
  session build, in a file this branch does not own. See REPORT.md's deviations.

## What to do with it next

The run this is for is the one the recipe already describes, with `--memory gru` added and the
checkpoint selection unchanged:

```bash
python -m f1sim.learn.ppo <the cl_free_s701 flags> --memory gru --memory-hidden 128 --name cl_mem_s701
```

If the extra channels are to be arms of their own, `--scan-channels memory` and
`--scan-channels memory,edges` are the two to run beside it; each starts bit-identical to the
original, so the three are comparable from step 0. The budget leaves room for a wider GRU
(`--memory-hidden 256` is the contract's ceiling) if 128 turns out to be the binding constraint,
and `python -m f1sim.learn.budget --hidden 256` measures what that costs before anything is
trained.
