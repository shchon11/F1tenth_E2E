# Does estimating friction help the plan tracker?

**2026-09-11.** Matched-policy evaluation of two controller arms, `fixed_low` and `estimated`, on
fixed diagnostic scenarios. W&B run [`pqz5wuae`](https://wandb.ai). Outcome-only data:
[`controller-arm-eval-outcomes.json`](controller-arm-eval-outcomes.json).

**Headline: no robust method-level completion gain.** `estimated` completes more trials in
aggregate, but the three training seeds do not agree on the direction, and the seed spread is larger
than the gap. A separate, consistent result does appear in *time* among trials both arms complete.
Both arms also come out below the frozen actor at the lowest friction — an observation whose cause
this experiment does not establish.

## Protocol

The plan tracker derives its speed envelope and acceleration bounds from an assumed tyre friction.
`fixed_low` assumes one conservative constant (µ 0.73423); `estimated` infers µ from a causal sensor
and issued-command history through a frozen quantile model. Both arms start from the same frozen
actor and train under their own arm for 128 updates / 1 048 576 environment steps, three seeds each.

| | |
| --- | --- |
| grid | 6 checkpoints × 2 maps × 3 frictions × 2 eval seeds × 16 trials = **72 cells, 1152 trials** |
| maps | `gen:control:9100`, `real:korea_2026_competition` |
| true µ | 0.73423, 0.94401, 1.15379 |
| randomisation | **off** during evaluation — µ is set per cell, not drawn |
| budget | 3.0 laps at the speed cap, 9.0 m/s; no trial timed out |
| backend | CUDA graphs, `compile_tracker` off |

**Matched starts and frozen source.** Every arm meets identical initial states, verified against the
frozen reference (`paired_initial_states_match_frozen_reference: true`), and both arms run the same
grip spec as that reference. Runtime source was frozen before the runs and unchanged throughout
(`source_frozen: true`). Independently audited: `pm-final-eval-audit.json`.

## Results

| checkpoint | completed / 1152·⅙ | rate | collisions |
| --- | --- | --- | --- |
| `fixed_low` seed 501 | 161 / 192 | 83.85 % | 31 |
| `fixed_low` seed 502 | 153 / 192 | 79.69 % | 39 |
| `fixed_low` seed 503 | 172 / 192 | 89.58 % | 20 |
| `estimated` seed 501 | 180 / 192 | 93.75 % | 12 |
| `estimated` seed 502 | 167 / 192 | 86.98 % | 25 |
| `estimated` seed 503 | 159 / 192 | 82.81 % | 33 |
| **`fixed_low` total** | **486 / 576** | **84.375 %** | 90 |
| **`estimated` total** | **506 / 576** | **87.847 %** | 70 |

Paired by training seed:

| seed | completion difference | only `fixed_low` | only `estimated` | shared successes | `estimated` time advantage |
| --- | --- | --- | --- | --- | --- |
| 501 | **+9.90 pp** | 0 | 19 | 161 | 0.123 s |
| 502 | **+7.29 pp** | 2 | 16 | 151 | 0.146 s |
| 503 | **−6.77 pp** | 14 | 1 | 158 | 0.225 s |

**Why this is not a method-level completion gain.** Seed 503 reverses the sign, and the ~10–11 pp
spread across seeds of the same arm is wider than the 3.5 pp aggregate difference. Three training
seeds cannot separate a method effect from seed variance at this size, and pooling the 576 trials per
arm does not help: the trials within an arm share three training runs, so they are not 576
independent samples of the method.

**The time result is consistent.** On trials both arms complete — the only fair comparison, since
completion sets differ — `estimated` is faster in **all three** seeds, by 0.123 / 0.146 / 0.225 s.
Modest, but it does not change sign, including in the seed whose completion rate went the other way.

## The low-friction observation

Split by true µ. Each frozen reference is the **same frozen actor run under that arm's controller**,
so the two reference columns are not identical and are labelled separately. Fine-tuned counts are
pooled over the three training seeds, which is why their denominator is 192 rather than 64.

| true µ | frozen actor under `fixed_low` | fine-tuned `fixed_low` | frozen actor under `estimated` | fine-tuned `estimated` |
| --- | --- | --- | --- | --- |
| **0.73423** | **56 / 64** (87.5 %) | **131 / 192** (68.2 %) | **56 / 64** (87.5 %) | **137 / 192** (71.4 %) |
| 0.94401 | 58 / 64 (90.6 %) | 169 / 192 (88.0 %) | 59 / 64 (92.2 %) | 178 / 192 (92.7 %) |
| 1.15379 | 63 / 64 (98.4 %) | 186 / 192 (96.9 %) | 63 / 64 (98.4 %) | 191 / 192 (99.5 %) |

At the two higher frictions the fine-tuned policies are close to the frozen actor. **At the lowest
friction both arms fall well short of it** — 68.2 % and 71.4 % against 87.5 %.

**What this does and does not say.** Deterioration at low µ is *observed after fine-tuning under
these controllers*. That is the whole claim. It does **not** establish a cause: both arms share the
new controller design as well as the new training setup, so a controller × learning interaction
remains entirely possible, and nothing here isolates map narrowing, the optimiser reset, the dropped
auxiliary losses or the task change as responsible. An earlier draft of this note argued that because
both arms drop, the cause must lie in the recipe rather than the controller. That inference is
invalid — the arms have the controller redesign in common too — and it has been removed.

A one-seed `legacy`-recipe probe has since completed and is reported separately —
[Does the fine-tuning recipe alone explain the low-friction drop?](2026-09-11-legacy-recipe-probe.md).
It does not support a recipe-only attribution either. Read that note rather than inferring a cause
from the table above.

## Scope — what this does not establish

- **Diagnostic maps, not unseen maps.** `real:korea_2026_competition` appears in the original actor's
  recorded training exposure (25 variants in the retained ancestor-leg configs of W&B run
  `qxkniuk2`), so no generalisation-to-unseen-venue claim is made. The historical track list differs
  by leg: three retained ancestor-leg snapshots record 149 entries including 25 Korea forms, the
  live remote config records 125 without Korea and belongs to a later leg that crashed without
  producing a checkpoint, and the checkpoint's own leg kept no config — its size is bounded at ≥147
  by its curriculum-gate output, its composition unrecorded. The per-leg audit lives with the
  experiment records outside this repository.
- **No on-car result.** Nothing here ran on a physical vehicle.
- **Three training seeds per arm.** Aggregate counts do not add independent training repetitions.
- **Cause is not established.** The step from the frozen actor to these six runs changed several
  things at once — map coverage, racing to solo, two auxiliary losses, the optimiser reset — *and*
  introduced the controller arms. The low-µ deterioration is an observation about the outcome of that
  combined step, not evidence for any one component.

## Reproduction

Cell identities, checkpoint hashes and per-trial outcomes are in
[`controller-arm-eval-outcomes.json`](controller-arm-eval-outcomes.json) (72 cells, 1152 trials).
The full record including traces is `final-eval-grid.jsonl` in the runner directory, outside this
repository.
