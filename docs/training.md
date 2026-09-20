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
hard states), `--scan-stack`, `--scan-stride`, `--scan-stem`, `--hist-len`, `--teacher-grip`.

**The teacher's line and limits** (opt-in; the defaults are the teacher every recorded run had, and
the default raceline cache keys are unchanged). `--raceline-objective min_time` refines the
minimum-curvature line by descending the lap time of the speed profile itself
(`raceline.min_time_raceline`); `--teacher-a-lat` / `--teacher-a-acc` / `--teacher-a-brake` set the
profile's limits (default 6 / 6 / 3), and the line is optimised for the same limits the teacher then
drives. `evaluate --teacher` takes the same flags. Measured on nine held-out maps, the line alone is
8 % quicker at the default limits with no more collisions, and `7 / 6.5 / 4` is the knee of the
pace-against-collisions curve; at the friction limit (9 and up) the teacher crashes in a quarter of
its trials and is not a usable label source. See
[the research note](research/mintime-teacher-speed-head-2026-09-19.md).

**What the plan's speed dimensions mean** (`--speed-mode`, opt-in, plan action space only;
`mpc.SPEED_MODES`). `linear` is the two speeds every existing checkpoint emits. `envelope` replaces
them with a grip belief `a_hat` and an end speed, and the profile follows the plan's own curvature —
`v(s) = sqrt(a_hat / |kappa(s)|)`, braked backwards from the end speed; `--grip-quantile` below 0.5
puts a pinball loss on `a_hat`, so a student that cannot tell the floor yet assumes the slippery
end. `knots` emits a speed at every curvature knot (12 action dimensions). The mode is recorded in
the checkpoint and `evaluate` reads it from there. `mpc.decode` — which the grip arms, the clearance
arm and the viewers read a plan through — refuses a non-`linear` plan rather than misreading it, so
those layers do not yet run on the new modes.

**A grip dial** (`--cond dial`, with `--dial-margin` / `--dial-exact`). The student is told the
floor's friction minus a per-episode margin and the teacher drives for that same number, so the
input is a command — "use this much grip" — that an operator or a supervisor sets at run time.
`--cond true_mu` is the lab-oracle form of the same input and `--aux-grip` asks the actor's grip
head for the friction; see [the research note](research/mintime-teacher-speed-head-2026-09-19.md) §7
for why the number is supplied rather than inferred.

**A recurrent student.** `--memory gru --seq-len N` trains on contiguous runs of `N` steps per
environment with the recurrence walked from a zero state (`--seq-burn` leading steps only warm it
up). Without `--seq-len` the buffer is sampled i.i.d. and the memory is never trained.

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

**Which card.** Two NVIDIA GPUs here: CUDA index 0 is the RTX 4070 SUPER and index 1 the RTX 5060
Laptop (`nvidia-smi` numbers them the other way round; the third adapter `lspci` lists is an AMD
iGPU with no ROCm torch, so it drives the display and nothing else). Training takes index 0 and
nothing else does, so a run's throughput does not depend on whether someone is watching a replay;
evaluation, lap measurement and the console go on index 1 (`CUDA_VISIBLE_DEVICES=1`).

**The dial recipe (2026-09-20).** Friction cannot be read from the car's sensors at a pace it
survives, but a policy that is told it uses all of it, so the friction is an *input the operator
sets* — a grip dial — and PPO is built around keeping it one:

```bash
python3 -m f1sim.learn.ppo --name ppo_dial --init ~/f1sim_runs/dg_dial/student_latest.pt \
  --cond dial --fresh-opt --grip-budget-penalty 2.0 \
  --action-mode plan --scan-stack 6 --hist-len 20 --tracks train \
  --raceline-objective min_time --teacher-a-lat 7 --teacher-a-acc 6.5 --teacher-a-brake 4 \
  --lap-time-bonus 2 --kl-coef 0.05 --kl-decay 4e7 --cap0 9 --cap1 9 --sim-backend graphs --amp
```

* `--init` is a DAgger student trained with `--cond dial` (the teacher drives for the dial's
  number, so the student arrives obeying it). It is loaded as it is — nothing is migrated — and the
  KL leash holds the run to that obedient policy. A DAgger checkpoint's exploration std is reset to
  the plan-space default (imitation never trains it; it arrives at 0.50).
* Each episode's dial is the floor's friction minus a margin (`--dial-margin`, `--dial-exact`).
* `--grip-budget-penalty` charges for lateral acceleration beyond what the dial allows
  (`EnvConfig.reward_grip_budget`). Without it the dial is only a hint under RL — the floor has at
  least that much grip, so return alone teaches the policy to go faster than it was told.
* The critic is told the dial (appended to the privileged vector), and the lap-time reference is the
  min-time line at the teacher's limits.
* `evaluate --dial-offset -0.15` scores a checkpoint with its dial set on the safe side.

Relevant flags: `--envs`, `--horizon`, `--epochs`, `--minibatch`, `--total`, `--amp`, `--cap0` /
`--cap1` / `--cap-steps` (speed-cap curriculum), `--kl-coef` / `--kl-decay`, `--critic-warmup`,
`--fresh-opt`, `--tracks`, `--obstacle-draws`, `--sim-backend`, `--memory` /
`--memory-hidden` / `--memory-critic`, `--scan-channels` / `--scan-memory-tau`.

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
allowed to be new are the memory modules (whose output projections are zero) and, when extra scan
channels are on, the zeroed new input columns of the stem's first convolution. Anything else left
fresh raises rather than training a half-initialised network. Adam's moments are re-keyed by
parameter name and the new tensors start fresh, so the optimiser state carries over too.

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

Cost, measured with the rest of the budget below: 0.03 ms of a 25 ms control step for both.

### The deployment budget

The car runs the policy at the LiDAR's 40 Hz, so one control step has 25 ms for scan preprocessing,
the policy forward, the iLQR tracker and publishing. The Jetson is not on the training machine, so
the rule this work is held to is a **ratio on this machine, measured the same way for both
networks** — a proxy, and quoted as one:

> CPU, single thread, batch 1, fp32, the fastest of several blocks of 200 iterations after
> warm-up (`torch.utils.benchmark`; the table records how many). The new actor's forward must
> stay within **1.5×** the frozen original's and its parameter count within **2×**.

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

## The plan controller

The tracker is the tracker: it follows the plan the policy emitted, and nothing is layered on it.

It used to offer friction-clamp *arms* -- `fixed_low`, `estimated`, `oracle`, plus the composable
`+clearance` and `+tcs` -- selected with `--controller`. They existed because a policy that had never
been told the floor's friction drove every floor at one compromise speed, and clamping the tracker
was the retraining-free way to make that safe: on suite v1 it took the frozen original from 110 to
136 solo completions and from 16 to 43 at low mu.

`--cond dial` tells the policy instead (see above), which makes the clamp redundant and then
harmful: measured on three pinned frictions, a dial policy with `fixed_low` -- or even a
perfectly-set `oracle` -- on top was **worse than the dial alone on every one of them**. The policy
already slows for the floor it was told about, and the clamp can only remove what it does not do
([the research note](research/mintime-teacher-speed-head-2026-09-19.md) section 7). So there is no
`--controller` / `--estimator` on `ppo` or `evaluate` and no selector in the console; a session
records `controller: "legacy"`, the single spelling for "nothing installed".

`learn/grip_runtime.py`, `grip_control.py`, `grip_estimator.py`, `clearance.py` and
`traction_arm.py` remain, and so do the `controller=` / `estimator=` keyword arguments of
`evaluate.evaluate`: the [checkpoint benchmark](benchmark.md) records which arm each historical
system was scored under, and re-reading those results needs the code that produced them.
