# Opponent diversity, stage 1: the joint range expansion does not advance (2026-09-12)

One paired pilot, declared in advance, measured and stopped. Widening both opponent-diversity
ranges at once — opponent speed multiplier (0.5, 1.0) -> (0.35, 1.10) and spawn gap (2.5, 6.0) ->
(1.5, 9.0) m — **fails the predeclared advancement gates**. Nothing is promoted, no arm is retrained and
no threshold moved.

Data in this directory, all local copies: `-raw-cells.jsonl` (the 170 scored cells),
`-roster.json` (five systems, loadable by `f1sim.learn.benchmark`), `-leaderboard.md` (the
canonical report tables), `-judgment.json` (the independent audit and gate verdicts),
`-paired-pace.json` (lap time over common completions), `-manifest.json` (every hash).

## The pair

Both arms start from the same A701 checkpoint (`29e822338538…`, Adam moments restored, no
`--fresh-opt`), run 32 updates x 8192 nominal rollout slots = 262,144, and differ in exactly three
CLI options: `--name` and the two ranges, `--opp-speed` and `--spawn-gap`.

Both arms train on the same 149 train map variants in two-car mixed races with a teacher fraction
of 0.5, under the legacy controller during training and the frozen estimated runtime during
evaluation, with every other A-recipe reward, auxiliary loss and learning-rate schedule unchanged.

| | control (`cl_oppdiv_control_s801`) | range (`cl_oppdiv_range_s801`) |
|---|---|---|
| `--opp-speed` | 0.5 1.0 | 0.35 1.10 |
| `--spawn-gap` | 2.5 6.0 | 1.5 9.0 |
| final checkpoint | `3fd9063e634021d8…` | `1207e17f45665ab7…` |
| returncode / updates | 0 / 32 | 0 / 32 |
| tensors changed vs init | 125/125 (actor 67/67), L2 1.9462 | 125/125 (actor 67/67), L2 1.9490 |
| finite model and Adam tensors | yes | yes |

`--opp-speed` is a dimensionless scale, not a speed interval: in self-play races it sets the front
car's cap (scale x 5.0 m/s, so 2.50–5.00 vs 1.75–5.50 m/s), and in teacher races it multiplies the
teacher's plan speed before the common 9 m/s cap. `--spawn-gap` is the arc along the lane between
the cars of a race, drawn per reset. Friction is constant within an episode in both training and
evaluation: mu is drawn once at each reset. The source was frozen before the first job (77 files,
`42e146c5233792fe…`) and measured unchanged after each one.

## Counts

Benchmark v1, suite freeze `6f710412bacf5a09…`, 34 cells and 272 trials per system, frozen
estimator `a4e6fe02e27e…`, `estimated` runtime throughout. All 170 rows carry one runtime and one
benchmark source digest, matching this worktree; all five systems started every cell from the same
state — every start-fingerprint field, the actor input included.

| system | solo | low mu | avoidance | overtaking | collisions/km | large-slip s/km |
|---|---|---|---|---|---|---|
| `frozen_original@estimated` | 134/144 | 41/48 | 41/64 | 42/64 | 6.06691 | 3.41953 |
| `cl_origrecipe_legacy_s701@estimated` | 134/144 | 41/48 | 44/64 | 43/64 | 5.59598 | 3.33290 |
| `cl_origrecipe_legacy_s702@estimated` | 136/144 | 43/48 | 43/64 | 45/64 | 5.25390 | 2.96627 |
| `cl_oppdiv_control_s801@estimated` | 136/144 | 43/48 | **50/64** | 42/64 | 4.80247 | 2.56223 |
| `cl_oppdiv_range_s801@estimated` | 136/144 | 43/48 | **47/64** | 44/64 | 4.86703 | 3.12842 |

## The predeclared gates

Fixed before these scores existed. All of them had to pass to call the pilot promising.

| gate | required | measured | verdict |
|---|---|---|---|
| solo vs control | loses <= 2/144 | 0 | pass |
| low mu vs control | loses <= 1/48 | 0 | pass |
| avoidance vs control | no loss | **-3** | **fail** |
| overtaking vs control | no loss | +2 | pass |
| avoidance + overtaking vs control | gains >= 4/128 | **-1** | **fail** |
| collisions/km vs control | ratio <= 1.10 | 1.01344 | pass |
| large-slip s/km vs control | ratio <= 1.10 | **1.22098** | **fail** |
| low mu vs original | loses <= 2/48 | +2 | pass |
| collisions/km vs original | ratio <= 1.10 | 0.80223 | pass |
| large-slip s/km vs original | ratio <= 1.10 | 0.91487 | pass |

Three gates fail, so **the joint range expansion is not a promising pilot**. The treatment gives up
3 avoidance trials for 2 overtaking trials — a combined loss where a gain of at least 4 was
required — and spends 22 % more time past the slip threshold per kilometre than its matched
control.

Lap time is secondary and conditional on common completions: over the 136 solo trials both pilot
arms completed, the range arm is 0.01618 s faster (-0.1457 %) — a difference this small on one
paired seed carries no weight, and neither arm's pace is what the gates turn on.

## What this does and does not establish

* This is one paired seed on three reused development maps (`gen:control:1400` is an SGR training
  map; `korea_2026_competition` and `gen:control:9100` are diagnostic maps). It is a bounded
  engineering screen, not a significance claim, and no generalisation claim attaches to it.
* It tests the two axes **together**. Nothing here separates a speed-range effect from a spawn-gap
  effect, and nothing here says a narrower widening, or either axis alone, would fail.
* Training-log collision trends are not evaluation results; only the scored rows above are.
* The control arm is the useful surprise: against the original reference it is the best of the five
  on collisions/km (4.80) and slip s/km (2.56) and on avoidance (50/64). That is an observation
  about the matched control, not a promotion of it.
* No promotion, no additional seed, no retraining, no threshold change. The decision on what to do
  next is the user's.
