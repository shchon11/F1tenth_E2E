# A702: replicating the original-recipe legacy run (2026-09-12)

A second seed of the recorded A recipe, trained and scored to answer one question: does the recipe
reproduce, or was A701 a single lucky seed? Nothing here promotes anything, and no pilot arm was
run.

Data in this directory, all local copies: `a702-replication-2026-09-12-raw-cells.jsonl` (the 102
scored cells), `-roster.json` (the three systems, loadable by `f1sim.learn.benchmark`),
`-judgment.json` (the reconstruction and the gate verdicts), `-leaderboard.md` (the canonical
`f1sim.learn.benchmark report` tables rendered from those cells), `-paired-pace.json` (lap time
over common completions) and `-manifest.json` (every hash below).

## What was trained

| | |
|---|---|
| run | `cl_origrecipe_legacy_s702`, seed 702 |
| argv | the recorded A701 invocation, differing in `--name` and `--seed` and nothing else |
| init | `frozen_original_48cc698f.pt` (`48cc698f…`), Adam moments restored, no `--fresh-opt` |
| budget | 128 updates x 8192 nominal rollout slots = 1,048,576; horizon 32, 256 envs, `learner_ids` 256 |
| result | returncode 0, 82.42 minutes, final checkpoint `0d904012dd29e2de…` |
| source | 77 files frozen before the run (`42e146c5233792fe…`), zero drift measured afterwards |

Friction stays constant within every episode. Training randomization occurs only at reset;
evaluation uses each cell's fixed friction value.

`learner_ids` is 256 because mixed races keep every car in the buffer; the teacher-driven
transitions are stored and then masked out of the loss. The 8192 figure is therefore nominal
rollout slots, not an equal number of on-policy actor samples.

## What was scored

Benchmark v1, suite freeze `6f710412bacf5a09…`, 34 cells and 272 trials per system, frozen
estimator `a4e6fe02e27e…`, `estimated` runtime for all three.

* `frozen_original@estimated` — rescored here. Its published rows could not be merged: they carry a
  different benchmark `__main__` digest, and the reporter refuses a table spanning two source
  digests.
* `cl_origrecipe_legacy_s702@estimated` — freshly scored.
* `cl_origrecipe_legacy_s701@estimated` — existing rows reused, after every reuse condition was
  checked: identical runtime and benchmark source digests to this source, suite freeze and version,
  pinned checkpoint `29e822338538…` and estimator, 34/34 cells with S 144 / A 64 / O 64 trials and
  48 low-friction trials, and identical physical, calibration and observation-spec start
  fingerprints to both freshly scored systems in all 34 cells.

## Counts

Successes out of trials, counted from the raw outcome arrays.

| system | solo | low mu | avoidance | overtaking | collisions/km | large-slip s/km | km |
|---|---|---|---|---|---|---|---|
| `frozen_original@estimated` (reference) | 134/144 | 41/48 | 41/64 | 42/64 | 6.067 | 3.420 | 9.066 |
| `cl_origrecipe_legacy_s701@estimated` | 134/144 | 41/48 | 44/64 | 43/64 | 5.596 | 3.333 | 9.114 |
| `cl_origrecipe_legacy_s702@estimated` | 136/144 | 43/48 | 43/64 | 45/64 | 5.254 | 2.966 | 9.136 |

Collisions and large-slip seconds are the pooled 272-trial figures over every suite, per travelled
kilometre. The canonical category tables (driving, stability, per-friction surface, avoidance,
overtaking) are rendered by `f1sim.learn.benchmark report` from the same rows.

## Predeclared retention screen

Fixed before these scores existed: against the original reference, solo and low-friction successes
may lose at most 2 trials, avoidance and overtaking at most 2 each, and collisions/km and
large-slip s/km may each rise by at most 10 % (candidate / reference <= 1.10; a zero reference
would require a zero candidate, and an unavailable or nonfinite value fails).

| gate | reference | A702 | delta / ratio | verdict |
|---|---|---|---|---|
| solo successes | 134/144 | 136/144 | +2 | pass |
| low-mu successes | 41/48 | 43/48 | +2 | pass |
| avoidance successes | 41/64 | 43/64 | +2 | pass |
| overtaking successes | 42/64 | 45/64 | +3 | pass |
| collisions/km | 6.067 | 5.254 | 0.866 | pass |
| large-slip s/km | 3.420 | 2.966 | 0.867 | pass |

All six pass, and A702 is at or above the reference on every one of them rather than merely within
tolerance.

## What this does and does not establish

* The recipe reproduces at a second seed on this suite: A702 lands beside A701 on every category
  and slightly better on the pooled safety measures. That is a replication **check**, not a method
  proof — two seeds do not make a significance claim, and the screen is an engineering gate.
* Scope is three reused development maps. `gen:control:1400` is an SGR training map;
  `korea_2026_competition` and `gen:control:9100` are existing diagnostic maps. No unseen-map or
  generalisation claim attaches to any number here.
* Nothing is promoted. The opponent-diversity pilot arms `cl_oppdiv_control_s801` and
  `cl_oppdiv_range_s801` were prepared and **not run**; their launch is a separate decision.
* Retention reproduced; both A seeds have slightly slower common-completion laps. On the 134 solo
  trials **both** the reference and the candidate completed -- the only fair lap-time comparison --
  A701 is +0.103358 s (+0.9486 %) and A702 +0.157276 s (+1.4435 %) against the reference's
  10.8955 s mean (`a702-replication-2026-09-12-paired-pace.json`). The safety improvements above
  coexist with slightly slower paired laps; this does not establish a speed change as their cause.
  Lap time stays secondary here and is conditional on common completions -- the leaderboard's own
  lap-time column is over each system's own completions and so spans different trial sets.
