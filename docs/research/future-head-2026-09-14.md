# An auxiliary head that predicts the opponent 0.5 s ahead, and a probe that says whether the hidden state knows it (2026-09-14)

Branch `feat/future-head`, base `88a4358`. Everything here is simulation.

## The question

`--memory gru` ([memory-policy-2026-09-13](memory-policy-2026-09-13.md)) gave the actor a hidden
state, and the opponent work ([opponent-diversity-2026-09-13](opponent-diversity-2026-09-13.md))
gave it cars to race. Nothing asks the hidden state to *represent* those cars. PPO rewards driving;
the grip head asks for a friction; the opponent head asks where the nearest car is **now**, which
six stacked LiDAR frames already nearly determine. A recurrent state that is never asked for
something the present observation does not contain is free to settle into a smoothed copy of it.

The paper claim the user wants is "a latent race state you plan through". Two things are needed for
it: a loss that forces the state to carry the near future, and a measurement that says whether it
does. This note is about building both and running them once.

**It is not a performance result.** The two arms below are 262144 environment steps each — 1/32 of
the run root has queued (`work/learning-next/future-20260914/launch.sh`, 8.4 M steps) and about 3 %
of the optimiser steps the head would get there. Nothing here says whether `--aux-future` makes the
policy race better; the benchmark says that, and it has not been run.

## What was built

`--aux-future COEF` (default 0, and off the head is not built at all). From the actor's **recurrent
state** after the step — the GRU's hidden state, `h_t`, not the action features; the trunk features
when `--memory off` — it predicts, K = 20 control steps (0.5 s at 40 Hz) ahead:

| target | what | scale |
|---|---|---|
| `opp_lon`, `opp_lat` | nearest opponent's position in the ego body frame at t + K | `PRIV_OPP_DIST_SCALE` = 5 m |
| `opp_vlon`, `opp_vlat` | that opponent's velocity relative to the ego, in the ego frame at t + K | 5 |
| `ego_speed`, `ego_yaw_rate` | the ego's own longitudinal speed and yaw rate at t + K | `v_max_policy` = 10 m/s, `imu_gyro_scale` = 5 rad/s |
| `opp_present` | logit: is a car inside `overtake_range` (12 m) at t + K | — |

Three choices worth naming, because each of them could have been made the other way:

* **The head reads the recurrent state, not the trunk.** The probe measures `h_t`; if the loss
  trained a different tensor the evidence would be about something else. `Actor.future_input` is the
  single function both take it from, and a test pins them to each other.
* **The label is the privileged state at t + K, recomputed the way `_priv` recomputes it at t** —
  the nearest opponent **at t + K**, not the car that was nearest at t. That keeps the label a pure
  function of one instant and makes the head's target and the probe's target identical by
  construction. The cost is that in a three-car field the target can change which car it refers to;
  that is a floor on the achievable MSE, not something the head can learn away. Tracking identity
  would remove the floor and would need the opponent's index carried through the rollout buffer.
* **`opp_vlon` is the true ego-frame relative velocity, not `privileged()[10]`.** That column is
  `other.vx - ego.vx`, a difference of two body-frame longitudinal speeds taken in two different
  frames. It is a serviceable present-tense cue and a poor prediction target: `opp_lon + dt *
  opp_vlon` is only "where the car will be" if both velocities live in one frame.

### Masking — what is dropped, measured

* **The last K steps of each `--horizon` chunk.** The label for step t is the state at t + K and the
  chunk does not reach that far. The chunk's own final state — the one the value bootstrap already
  visits — is stored as the (T+1)-th label row, so `t = T − K` is the last labelled step: **13 of 32**
  at `--horizon 32`, 41 %. The next chunk's states are not available during this update and no
  second rollout and no extra simulator step is paid for them, so those steps are dropped rather
  than approximated.
* **Any window containing an episode boundary**, where "boundary" means *any car of that race*
  resetting, not just the ego's. An opponent that crashes is respawned behind the field in place
  without ending the learner's episode, so its pose half a second later is not the continuation of
  the motion the head was asked to extrapolate. This is stricter than the ego's own boundary and it
  costs: measured on arm B, the labelled fraction was **0.20–0.25**, against the 0.41 ceiling. Half
  the otherwise-labelled steps are dropped because somebody in the race reset. Relaxing the rule to
  the ego's own boundary would roughly double the data and turn an opponent respawn into target
  noise; it is a lever, and this note picks label quality.
* **The four opponent columns where no car is inside `overtake_range` at t + K.** `opp_present`
  always trains: it is the column that says whether the others mean anything.

### The probe

`python -m f1sim.learn.probe_hidden` rolls a checkpoint out with traffic and events on, freezes the
hidden states, and fits a **linear** ridge read-out from `h_t` to the same targets at t + k. Linear
on purpose: a nonlinear probe measures the probe. Whole env columns — whole cars — are held out,
never rows inside a trajectory, and the R² is averaged over eight draws of which cars those are.
That last part turned out to matter more than anything else here (below).

A feedforward checkpoint is warm-started into the GRU so it has a state to probe. That is not a
distortion: the projection is zero, so the policy it drives with is bit-identical to the original's,
and what the probe reads is a *random recurrent feature map* fed by the original's own embedding —
the honest "before any of this trained" row.

## The two-arm smoke

Identical except for one flag, both warm-started from `frozen_original_48cc698f`, both
`--memory gru --memory-hidden 128 --scan-channels memory,edges`, `--race-size 3 --opponent pool`
with the opponent-diversity flags of `work/learning-next/long-20260913/launch.sh`, three tracks
(`real:blackbox2022_1`, `gen:control:1400`, `real:korea_2026_competition+rlobs211`), seed 701,
`--total 262144`. `--envs 63`, not 64: the trainer refuses an env count that is not a multiple of
`--race-size`. 131 logged updates, ~13.6 min each on the RTX 4060 Ti at ~345 env steps/s.

### PPO health — the term costs the objective nothing visible

| metric | A (`--aux-future 0`) first 20 | A last 20 | B (`--aux-future 1.0`) first 20 | B last 20 |
| --- | --- | --- | --- | --- |
| `kl_ref` | 0.036 | 0.127 | 0.046 | 0.135 |
| `clipfrac` | 0.0004 | 0.0000 | 0.0004 | 0.0000 |
| `approx_kl` | 0.0004 | 0.0002 | 0.0004 | 0.0002 |
| `grad_norm` | 12.18 | 17.62 | 12.82 | 19.59 |
| `loss/vf` | 3.64 | 3.69 | 3.48 | 3.39 |
| `aux_grip` | 0.0168 | 0.0115 | 0.0157 | 0.0116 |
| `aux_opp` | 0.0613 | 0.0472 | 0.0633 | 0.0395 |
| collisions/km | 117.0 | 21.9 | 119.5 | 20.3 |
| progress [m] | 27.4 | 74.2 | 26.6 | 79.9 |
| reward/step | 0.086 | 0.085 | 0.083 | 0.100 |

Both arms recover from the warm start the same way and end in the same place. The KL leash ends
marginally longer in B (0.135 vs 0.127) and the gradient norm marginally larger (19.6 vs 17.6),
which is what one more term in the loss looks like. On one seed and 131 updates none of these
differences mean anything; what they rule out is the failure worth ruling out — that adding the head
destabilises the update.

### The auxiliary loss

`loss/aux_future_mse` is the mean of the six squared errors plus the presence cross-entropy.

| updates | total | `opp_lon` | `opp_lat` | `opp_vlon` | `opp_vlat` | `ego_speed` | `ego_yaw_rate` | presence BCE (base-rate floor) | labelled | present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1–26 | 0.931 | 0.632 | 0.276 | 0.250 | 0.275 | 0.1620 | 0.039 | 0.659 (0.101) | 0.202 | 0.972 |
| 27–52 | 0.767 | 0.776 | 0.451 | 0.326 | 0.323 | 0.0139 | 0.034 | 0.446 (0.236) | 0.208 | 0.917 |
| 53–78 | 0.615 | 0.799 | 0.396 | 0.272 | 0.337 | 0.0079 | 0.035 | 0.308 (0.251) | 0.226 | 0.919 |
| 79–104 | 0.595 | 0.768 | 0.366 | 0.268 | 0.352 | 0.0078 | 0.037 | 0.296 (0.267) | 0.245 | 0.914 |
| 105–130 | 0.685 | 0.745 | 0.467 | 0.363 | 0.330 | 0.0089 | 0.040 | 0.359 (0.329) | 0.204 | 0.885 |

**It falls: 0.93 → 0.60 over the first hundred updates**, with a noisy last band. But a raw MSE
against a target distribution that is itself moving — the policy goes from crashing after 27 m to
completing 80 m — is not a learning curve, so the same numbers divided by each target's variance
under the same mask (`1 − mse / var`, the head's own explained variance):

| updates | `opp_lon` | `opp_lat` | `opp_vlon` | `opp_vlat` | `ego_speed` | `ego_yaw_rate` |
| --- | --- | --- | --- | --- | --- | --- |
| 1–26 | −0.044 | −0.194 | −0.543 | −0.073 | −23.1 | −0.109 |
| 27–52 | −0.049 | −0.078 | −0.132 | −0.053 | −1.00 | +0.020 |
| 53–78 | −0.057 | −0.055 | −0.063 | −0.036 | +0.154 | +0.020 |
| 79–104 | −0.039 | −0.012 | −0.031 | −0.033 | +0.219 | +0.043 |
| 105–130 | −0.049 | −0.047 | −0.050 | −0.054 | +0.149 | +0.036 |

Read honestly: the fall is carried by **the presence logit learning the base rate** (its BCE tracks
its own base-rate floor from update ~50 on — it has learned "there is usually a car", and nothing
more) and by **`ego_speed` learning its mean and then about a fifth of its variance**. The four
opponent columns sit at about −0.05 throughout: after 131 updates the head does no better on them
than predicting their mean.

That is a step-budget statement, not a verdict. 131 updates × 6 minibatch steps = **786 Adam steps**
at the finetune's `--lr 5e-5`, from an output layer initialised at exactly zero. The trained head
bears it out: at the end its output layer's weights have an rms of **0.0067** and a maximum of
0.0199. The head is a few percent into its own training curve. The run root has queued
(`work/learning-next/future-20260914/launch.sh`, 8.4 M steps at `--envs 258`) gives it about 27000
such steps — 1016 updates of 27 minibatches.

Also worth knowing before reading anything into the presence column: in this configuration a car is
inside `overtake_range` **89–97 %** of the time, so the label is nearly constant and carries little
information. It would carry more with a bigger field or looser spawn gaps.

## The probe

400 steps × 32 learner cars, three tracks, `--race-size 3 --opponent teacher` with all seven
opponent behaviours on, 25 % of the cars held out, averaged over 8 draws of which cars (± is the
spread across draws). Run twice on independent rollouts (`--seed 123` and `--seed 321`).

### k = 20 steps (0.5 s ahead)

| seed | checkpoint | `opp_lon` | `opp_lat` | `opp_vlon` | `opp_vlat` | `ego_speed` | `ego_yaw_rate` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | frozen, warm-started | +0.154 ±0.147 | +0.131 ±0.195 | +0.138 ±0.147 | +0.084 ±0.207 | **+0.837 ±0.045** | **+0.655 ±0.058** |
| 123 | arm A, no head | +0.058 ±0.153 | +0.091 ±0.406 | +0.104 ±0.203 | +0.029 ±0.207 | **+0.844 ±0.013** | **+0.688 ±0.050** |
| 123 | arm B, with head | +0.013 ±0.139 | +0.054 ±0.107 | +0.101 ±0.158 | −0.071 ±0.192 | **+0.822 ±0.032** | **+0.665 ±0.059** |
| 321 | frozen, warm-started | +0.195 ±0.104 | +0.245 ±0.121 | +0.130 ±0.084 | +0.107 ±0.097 | **+0.854 ±0.022** | **+0.623 ±0.039** |
| 321 | arm A, no head | +0.311 ±0.134 | +0.294 ±0.122 | +0.244 ±0.077 | +0.239 ±0.110 | **+0.872 ±0.019** | **+0.661 ±0.035** |
| 321 | arm B, with head | +0.126 ±0.103 | +0.262 ±0.131 | +0.143 ±0.114 | +0.124 ±0.129 | **+0.844 ±0.023** | **+0.689 ±0.035** |

### k = 0 (the present, the control)

| seed | checkpoint | `opp_lon` | `opp_lat` | `opp_vlon` | `opp_vlat` | `ego_speed` | `ego_yaw_rate` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | frozen, warm-started | +0.215 ±0.156 | +0.104 ±0.180 | +0.202 ±0.134 | +0.065 ±0.195 | +0.880 ±0.019 | +0.812 ±0.036 |
| 123 | arm A, no head | +0.164 ±0.145 | +0.110 ±0.338 | +0.156 ±0.177 | +0.071 ±0.186 | +0.915 ±0.009 | +0.815 ±0.034 |
| 123 | arm B, with head | +0.090 ±0.142 | +0.036 ±0.124 | +0.132 ±0.125 | −0.054 ±0.165 | +0.902 ±0.014 | +0.807 ±0.030 |
| 321 | frozen, warm-started | +0.221 ±0.104 | +0.219 ±0.077 | +0.172 ±0.056 | +0.170 ±0.077 | +0.916 ±0.011 | +0.803 ±0.022 |
| 321 | arm A, no head | +0.361 ±0.107 | +0.280 ±0.115 | +0.284 ±0.056 | +0.218 ±0.113 | +0.927 ±0.007 | +0.836 ±0.013 |
| 321 | arm B, with head | +0.134 ±0.084 | +0.218 ±0.113 | +0.204 ±0.092 | +0.082 ±0.102 | +0.915 ±0.012 | +0.814 ±0.022 |

`opp_present` is omitted from both tables: it is constant 89–97 % of the time, its variance across
held-out cars is tiny, and its R² is accordingly between −0.35 and +0.04 with a spread up to ±0.86.
It measures nothing in this configuration.

Three things this says, in order of confidence:

1. **The state carries the ego's own near future, and the probe can see it.** `ego_speed` reads
   +0.82 … +0.87 half a second ahead and `ego_yaw_rate` +0.62 … +0.69, with a spread of ±0.02–0.06
   across held-out draws — tight, reproduced on both rollout seeds, and only slightly below the
   k = 0 values. Half a second of the ego's own motion is nearly determined by its state, which is
   both unsurprising and the control that says the probe, the label scaling and the alignment are
   working.
2. **It does not carry the opponent's, and the head did not change that here.** Every opponent
   column is between 0.0 and +0.36 with a spread of ±0.08 to ±0.41, and the **ordering of the arms
   flips between rollout seeds** — arm A is the worst on seed 123 and the best on seed 321. On this
   evidence no arm is distinguishable from any other, and the honest summary is a null result, not a
   small effect.
3. **It does not carry where the opponent is *now* either.** The k = 0 column is the same 0.1–0.36.
   The existing `--aux-opp` head predicts the present opponent from the *trunk features*, not from
   the GRU state, so this is not a contradiction — but it does say plainly that after a 262144-step
   smoke the recurrent state is not an opponent tracker at all, and that a future head with 786
   Adam steps behind it was never going to make it one.

## Is it the step budget, or is the information not there?

A third measurement, offline and cheap, separates them. Take each checkpoint's frozen hidden states
from the probe's own rollout (`--save-states`), train the real `FutureHead` on them — a stationary
distribution, no PPO, any learning rate — and score it on **cars it never trained on**, the same
whole-column split the probe uses. `work/future-head/smoke/head_convergence.py`.

| states from | best held-out total (at step) | init | `opp_lon` | `opp_lat` | `opp_vlon` | `opp_vlat` | `ego_speed` | `ego_yaw_rate` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen, warm-started | 0.524 (800) | 1.153 | +0.250 | +0.282 | +0.153 | +0.090 | +0.263 | +0.519 |
| arm A, no head | 0.647 (800) | 1.192 | +0.214 | +0.254 | +0.165 | +0.142 | +0.200 | +0.633 |
| arm B, with head | 0.519 (2000) | 1.054 | +0.146 | +0.065 | −0.136 | +0.060 | +0.681 | +0.542 |

And what happens if it is allowed to keep going (arm B's states, 8000 Adam steps):

| lr | train total @8000 | held-out total @8000 | held-out `opp_lon` | `opp_lat` | `opp_vlon` | `opp_vlat` |
| --- | --- | --- | --- | --- | --- | --- |
| 5e-5 | 0.352 | 0.663 | −0.063 | −0.232 | −0.707 | −0.050 |
| 2e-4 | 0.159 | 0.879 | −0.320 | −0.596 | −1.128 | −0.272 |
| 1e-3 | **0.029** | **1.617** | −0.656 | −0.849 | −2.029 | −0.614 |

The head has no trouble driving the training loss to 0.03 — the plumbing, the labels and the
gradient all work, and the 17 k-parameter head can fit 8.7 k correlated rows exactly. Every one of
those gains is memorisation: the held-out loss rises monotonically while the training loss falls,
and the held-out explained variance on the opponent columns goes *negative*. With early stopping the
nonlinear head reaches +0.15 … +0.28 on the opponent columns — essentially what the **linear** probe
already read. A 2-layer nonlinear read-out buys nothing over a linear one on cars it has not seen.

So the answer to the question is: **at this point in training the opponent's near future is not in
the 128-dimensional hidden state in any form that generalises across cars** — not linearly, not
nonlinearly. The step budget explains why the head did not learn it during the smoke; it does not
explain the probe, which is a statement about the state as it stands and is independent of how many
updates the head had.

## What this does and does not license

**Licensed.** The head exists, is off by default and byte-identically so; it trains without
destabilising PPO; the labels and their masks are tested against hand-computed geometry and against
a constant-velocity opponent; the probe is a working, reproducible measurement with an error bar,
and it reads +0.84 on a target the state genuinely carries, which is what makes its ~0.1 on the
opponent targets worth believing.

**Not licensed.** Any sentence of the form "the policy plans through a latent race state". The
probe says the opposite today, for every arm including the one with the head. The most that can be
said now is: *the ego's own near-future motion is linearly decodable from the hidden state (R² 0.82
at 0.5 s); the nearest opponent's is not (R² 0.0–0.36, spread ±0.1–0.4, no arm separable).*

## What to do with it

1. **Run the real thing.** `work/learning-next/future-20260914/launch.sh` is 8.4 M steps and gives
   the head ~27000 Adam steps against the 786 it has had. Re-run `probe_hidden` on its final
   checkpoint against the same frozen baseline. That is the experiment; this was the instrument.
2. **Give the probe more independent cars before trusting a difference.** Eight held-out cars on
   three tracks is not enough: the spread across draws is larger than every between-arm difference
   measured. More envs and more tracks, not more steps — the steps inside one car's trajectory are
   correlated and buy little.
3. **Two levers that are one line each if the head stalls at the full budget.** The boundary rule
   could be relaxed from "any car of the race" to "the ego" (≈2× the labels, at the cost of turning
   an opponent respawn into target noise), and the target could track the opponent that was nearest
   at *t* instead of recomputing at *t + K* (removes the identity-switch floor, at the cost of
   carrying an index through the rollout buffer).
4. **Do not read the presence logit** until the field is large enough or the spawn gaps loose enough
   for it to vary. At 89–97 % present it is a constant.

## Reproducing

```bash
cd work/future-head
./smoke/launch.sh a          # --aux-future 0
./smoke/launch.sh b          # --aux-future 1.0
python3 smoke/summarise.py smoke/logs/fh_smoke_a.jsonl smoke/logs/fh_smoke_b.jsonl
./smoke/probe.sh             # writes smoke/probe/probe.{json,md} and smoke/probe/states/*.pt
./env.sh python smoke/head_convergence.py --states smoke/probe/states/arm_b_with_head.pt --k 20
```

Checkpoints: `~/f1sim_runs/fh_smoke_a/ppo_final.pt`, `~/f1sim_runs/fh_smoke_b/ppo_final.pt`.
Tests: `f1sim/tests/test_future_head.py` (25) and the extended
`test_memory_model.py::test_warm_start_is_bit_identical`.
