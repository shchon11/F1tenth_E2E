# Can a training-recipe change recover low-friction completion?

**2026-09-12.** Eight PPO runs — four recipes × two training seeds — evaluated against a frozen
reference on fixed scenarios with a static friction coefficient per episode. Per-trial data:
[`static-grip-retention-2026-09-12-trials.json`](static-grip-retention-2026-09-12-trials.json).
Independent eligibility audit:
[`static-grip-retention-2026-09-12-audit.json`](static-grip-retention-2026-09-12-audit.json).

**Headline: no eligible recipe.** The best of the three, `aux_anchor`, reached a **+1.875 pp** mean
low-friction gain against a required **≥ 2 pp**. It missed. The threshold was not changed, and no
recipe was selected. Alongside that, the per-seed results **disagree in sign** for every recipe, so
these runs do not provide evidence of a robust method-level benefit.

## What was varied

The policy's inputs, outputs and the friction estimator were held fixed. Two existing training
options were separated and tested against a plain control:

| recipe | aux coefficient | initial KL coefficient |
| --- | ---: | ---: |
| `base` | 0 | 0.05 |
| `aux` — restore the auxiliary friction-prediction objective | 1 | 0.05 |
| `anchor` — stay closer to the original policy | 0 | 0.20 |
| `aux_anchor` — both | 1 | 0.20 |

The `aux` head is supervised with the true friction coefficient. That is not the only place true µ
enters training: PPO is an asymmetric actor-critic, and the **critic** receives a privileged vector —
true dynamic state, track-relative pose, wall clearance and randomised parameters — which includes
true µ (`GymEnv.PRIV_PARAMS`), in every recipe here including `base`. What excludes true µ is the
**actor**, whose inputs are the same at training and deployment and never contain it. So `aux`
changes what the auxiliary head is supervised on, not whether privileged information exists in
training at all.

For the first 10 updates PPO's policy gradient and KL term are off while
the aux term is active, so `aux` changes not only the objective but *when* the early policy features
begin to move. Calling that the same warmup across recipes would be inaccurate.

## The eight runs

All start from the same frozen policy, the same estimator (seed 401,
`a4e6fe02e27e764f…`), the same five training maps, single car, 9 m/s speed cap, and the same budget:
128 updates / 1 048 576 environment steps each, **8 388 608 steps total**.

| run | aux | KL | steps | min | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| `base:701` | 0 | 0.05 | 1 048 576 | 17.94 | [`lhnfc8j4`](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/lhnfc8j4) |
| `aux:701` | 1 | 0.05 | 1 048 576 | 17.91 | [`xycdk24q`](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/xycdk24q) |
| `anchor:701` | 0 | 0.20 | 1 048 576 | 18.13 | [`pfcylek4`](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/pfcylek4) |
| `aux_anchor:701` | 1 | 0.20 | 1 048 576 | 18.32 | [`yxnh9b2s`](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/yxnh9b2s) |
| `base:702` | 0 | 0.05 | 1 048 576 | 18.26 | [`anqdhqya`](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/anqdhqya) |
| `aux:702` | 1 | 0.05 | 1 048 576 | 17.97 | [`mogwqjzy`](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/mogwqjzy) |
| `anchor:702` | 0 | 0.20 | 1 048 576 | 17.92 | [`empewl32`](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/empewl32) |
| `aux_anchor:702` | 1 | 0.20 | 1 048 576 | 18.30 | [`v2ehh0pk`](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/v2ehh0pk) |

Reward and collision rate from the *training* logs are not used as results. Everything below comes
from the separate controlled evaluation.

## Evaluation protocol

| | |
| --- | --- |
| grid (table below) | 9 checkpoints × 5 maps × 3 frictions × 2 eval seeds × 8 envs = **270 cells, 2 160 trials** |
| per checkpoint | 5 × 3 × 2 = **30 cells**, × 8 envs = **240 trials** |
| maps | `real:icra2022`, `real:blackbox2021_2`, `real:blackbox2022_1`, `rt:Spielberg`, `gen:control:1400` |
| true µ | 0.73423 (low), 0.94401 (mid), 1.15379 (high) |
| friction | **static per episode** — set per cell, constant for the whole episode, never re-drawn mid-run |
| randomisation | **off** during evaluation |
| controller | `estimated` arm for every checkpoint |
| budget | 3.0 laps at the 9.0 m/s cap; **no trial timed out** |

**Two evaluation stages, kept distinct.** The *screen* ran the 8 trained checkpoints (8 × 30 = **240
cells**). A *preflight* stage ran 4 further policies — `frozen_original` and three earlier
`estimated` policies — on the identical grid (4 × 30 = **120 cells**). The results table below is the
9-checkpoint set: the 8 screen checkpoints plus `frozen_original`, **270 cells**. The three earlier
policies account for the remaining **90 cells**; they are published for reference and are not part of
the recipe decision. All **360 cells** across both stages appear in the per-trial data.

**Matched starts.** 30 scenarios were checked for identical initial conditions across **360 cell
records — 12 system records (8 screen + 4 preflight) × 30 scenarios** — with zero invalid or missing
fingerprints and zero mismatches within or across files. Each start is recorded as three separate
digests — physical state, actor input, and calibration — never one combined hash, so a disagreement
names the layer it happened in. Every published cell carries its own three digests.

## Results

Completion out of 240 trials per checkpoint. Low friction is 80 trials; mid+high is 160.

| checkpoint | overall | low µ | mid+high |
| --- | --- | --- | --- |
| `frozen_original` | 152 / 240 — 63.33 % | 39 / 80 — 48.75 % | 113 / 160 — 70.63 % |
| `base:701` | 159 / 240 — 66.25 % | 41 / 80 — 51.25 % | 118 / 160 — 73.75 % |
| `base:702` | 170 / 240 — 70.83 % | 49 / 80 — 61.25 % | 121 / 160 — 75.63 % |
| `aux:701` | 168 / 240 — 70.00 % | 45 / 80 — 56.25 % | 123 / 160 — 76.88 % |
| `aux:702` | 167 / 240 — 69.58 % | 45 / 80 — 56.25 % | 122 / 160 — 76.25 % |
| `anchor:701` | 171 / 240 — 71.25 % | 47 / 80 — 58.75 % | 124 / 160 — 77.50 % |
| `anchor:702` | 160 / 240 — 66.67 % | 42 / 80 — 52.50 % | 118 / 160 — 73.75 % |
| `aux_anchor:701` | 167 / 240 — 69.58 % | 46 / 80 — 57.50 % | 121 / 160 — 75.63 % |
| `aux_anchor:702` | 170 / 240 — 70.83 % | 47 / 80 — 58.75 % | 123 / 160 — 76.88 % |

Every trained checkpoint completes more often than the frozen original. That is the one comparison
here with a consistent sign — and it is a comparison against a policy that never saw this training
recipe at all, not evidence for any one recipe.

## Eligibility

Seven checks, fixed before the final screen selection and before the outcomes were inspected. A
recipe is eligible only if it passes all seven.

| recipe | mean low gain vs `base` | per-seed low deltas | failed checks | eligible |
| --- | ---: | --- | --- | --- |
| `aux` | 0.000 pp | +5.00 / −5.00 pp | `low_gain_vs_base_ge_2pp`, `each_seed_low_loss_le_2_5pp` | **no** |
| `anchor` | −0.625 pp | +7.50 / −8.75 pp | `low_gain_vs_base_ge_2pp`, `each_seed_low_loss_le_2_5pp` | **no** |
| `aux_anchor` | **+1.875 pp** | +6.25 / −2.50 pp | `low_gain_vs_base_ge_2pp` | **no** |

`aux_anchor` passed six of seven and missed only the headline gain, by 0.125 pp. That is a miss. It
is recorded as a miss, the 2 pp threshold was not revisited after seeing the result, no recipe was
proposed, and no selection lock was written.

The other five checks — mid+high retention, low and overall retention against the frozen original,
and pace against both references — passed for all three recipes.

## Seed dependence

The per-seed columns above are worth reading alongside the means. For every recipe the two training
seeds **disagree in sign** at low friction: `anchor` is +7.50 pp on one seed and −8.75 pp on the
other, `aux` is +5.00 / −5.00, and `aux_anchor` is +6.25 / −2.50. A mean taken over two values of
opposite sign does not show a consistent direction.

The control varies too. `base:701` and `base:702` differ only in training seed and complete
**41 / 80** and **49 / 80** at low friction, an 8-trial difference.

These are raw completion counts per checkpoint, not estimates of the variance of a paired difference,
and this screen applies **engineering eligibility gates — it is not a significance or power
analysis**. Two seeds per recipe are not a basis for quantifying run-to-run variability, so no
statement about detectable effect size is made here. What the data supports is narrower and
sufficient for the decision: **these runs provide insufficient evidence of a robust method-level
benefit**, because no recipe met the required gain and no recipe's seeds agreed on the direction.

## Pace

Pace bounds how much a completion gain may be paid for in lap time. The gate is **one-sided: a
worsening of at most 3 %**. It does not cap going faster. Positive numbers below mean *slower* on
commonly completed trials.

| recipe | mean pace vs `base` | mean pace vs `frozen_original` |
| --- | ---: | ---: |
| `aux` | +0.52 % slower | −1.31 % faster |
| `anchor` | +0.95 % slower | −0.96 % faster |
| `aux_anchor` | +1.32 % slower | −0.56 % faster |

All three are **slightly slower than `base`** — +0.5 to +1.3 % — and all three stay inside the
allowed 3 % slowdown. So the pace check is passed, but it should not be read as showing the recipes
drove no slower: against `base` they all did, by a margin the gate permits. Against the frozen
original all three are faster.

Pace is computed only over trials **both** sides completed — 145–159 of 240 depending on the pair,
with the per-pair counts in the audit JSON. Comparing mean lap time across different sets of
completed trials would let a checkpoint look faster by failing the hard ones.

## Scope — what this does not establish

- **No statistical significance or power claim.** These are engineering eligibility thresholds, and
  the audit labels itself the same way. Nothing here estimates the variance of a paired difference or
  what effect size this design could detect; two seeds per recipe are not a basis for either.
- **No new-venue generalisation.** All five maps are training maps. The four DEV and four final-test
  maps were not driven, and were not opened before this decision.
- **No on-car claim.** Nothing here was measured on a physical vehicle.
- **The +1.875 pp is not a near-pass to build on.** It is one recipe's mean across two seeds that
  disagree by 8.75 pp.
- **The frozen-original comparison is not a recipe result.** It compares against a policy trained
  under a different setup entirely.

## Reproduction inputs

Published beside this note:

| file | contents |
| --- | --- |
| [`static-grip-retention-2026-09-12-trials.json`](static-grip-retention-2026-09-12-trials.json) | per-trial outcomes for all 360 evaluated cells — completion, collision, timeout, elapsed time, progress and track length for every trial, with each cell's three start digests |
| [`static-grip-retention-2026-09-12-audit.json`](static-grip-retention-2026-09-12-audit.json) | the independent eligibility audit, verbatim apart from local paths reduced to filenames |

Every completion figure in this note was recomputed from the published per-trial arrays alone and
matched the audit exactly across all 27 checkpoint × band combinations.

**The full per-step traces are not in this repository, and the published files are not a truncated
stand-in for them.** They are 0.62 GiB (preflight) and 1.24 GiB (screen) of JSONL. The per-trial
outcomes above are sufficient to recompute every number in this note; the traces are needed only to
re-derive per-step controller diagnostics. Both trace files are identified by sha256 and row count
in the `provenance` block of the trials JSON, alongside the sha256 of the compact evaluated-grid
JSON each was reduced from.

## Status of what follows

A directional-coverage experiment (R10) has an **approved design** — the same five layouts driven in
reverse, against the matched `base:701`/`base:702` controls. It is **now running** — seed 701 started
2026-09-12 06:19 KST, seed 702 queued — and **no outcome is claimed or implied here**. Reverse
direction on known layouts would in any case be evidence about directional coverage, not about
unseen venues.

The checkpoint benchmark leaderboard is a separate development instrument covering driving,
stability, per-surface retention, obstacle avoidance and overtaking. Its measured table is not part
of this note.
