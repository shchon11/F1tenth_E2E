# Does the fine-tuning recipe alone explain the low-friction drop?

**2026-09-11.** Follow-up to
[the controller-arm evaluation](2026-09-11-controller-arm-evaluation.md). One control run, one seed.

**Answer: it does not settle it, in either direction.**

## Why the probe exists

The matched evaluation found both arms below the frozen actor at the lowest friction. Two families
of explanation fit that: the fine-tuning recipe (map coverage narrowed, racing → solo, two auxiliary
losses dropped, optimiser reset) or an interaction between the controller redesign and learning. Both
arms share *both* changes, so the 72-cell grid cannot separate them.

This probe fine-tunes under `--controller legacy` — the untouched tracker — with the same recipe, so
that the recipe varies alone.

## Result

| | completed / 192 starts |
| --- | --- |
| frozen actor under `legacy` | **161 / 192** |
| `legacy`-recipe probe (one seed) | **159 / 192** |

Same 192 starts; source and references validated. The aggregate difference is two trials.

Split by scenario, the two low-friction cells move in **opposite directions** and largely cancel:

| low-µ scenario | frozen `legacy` | probe |
| --- | --- | --- |
| `real:korea_2026_competition` | 25 | **19** |
| `gen:control:9100` | 9 | **13** |

## What this supports

**Not a blanket recipe-only attribution.** A one-seed control with a two-trial aggregate difference,
whose low-friction cells cancel, is not evidence that the recipe alone causes the drop seen in the
matched arms — and it is not evidence against it either. The matched evaluation's three seeds spanned
~10–11 pp on the same scenarios; a single run carries no way to tell a real effect from that spread.

The honest reading of both notes together: **low-friction deterioration after fine-tuning is
observed; its cause is open.** The controller × learning interaction remains live, and so does the
recipe. Neither has been isolated.

## Scope

One seed, one arm, the same fixed diagnostic scenarios and matched starts as the primary grid. The
primary 72-cell result is unchanged by this probe and was not re-run.
