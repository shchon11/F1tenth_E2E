# Checkpoint benchmark v1 — representative subset

Report generated 2026-09-11T21:16:55Z — that is when this report was rendered, not when the runs were
measured. **3 systems, 2 weight sets, 102 cells, 816 trials.**

This is a time-bounded representative subset, chosen **for time and checkpoint availability, before
any score was judged** — not by performance ranking. The initial planned roster had 17 systems; the
frozen suite defines 34 cells per system. Nothing here is a full-benchmark result, and the subset is
not a best-of selection.

Of the 14 systems not in this table, **13 were never started**. The fourteenth,
`cl_main_estimated_s502@estimated`, was started and **excluded**: a cancelled partial at 1/34 cells,
rc −15, shed by the VRAM guard when `nvidia-smi` failed mid-run. It is preserved on disk and is not a
result.

## Files

| file | contents |
| --- | --- |
| [`benchmark-v1-2026-09-12-raw-cells.jsonl`](benchmark-v1-2026-09-12-raw-cells.jsonl) | the 102 rows as produced, unmodified |
| [`benchmark-v1-2026-09-12-derived-cells.jsonl`](benchmark-v1-2026-09-12-derived-cells.jsonl) | the same rows plus `n_envs` — **derived**, and what the table was rendered from |
| [`benchmark-v1-2026-09-12-results.json`](benchmark-v1-2026-09-12-results.json) | both of the above as one JSON payload, carrying both hashes |
| [`benchmark-v1-2026-09-12-manifest.json`](benchmark-v1-2026-09-12-manifest.json) | hashes, per-system return codes, source drift, exclusions |
| [`benchmark-v1-2026-09-12-roster.json`](benchmark-v1-2026-09-12-roster.json) | the three systems as a loadable roster — same IDs, checkpoint hashes, arms and `cross_runtime` as the roster actually scored, with ordinary relative paths |

### Regenerate this table yourself

`report` reads metadata only — it never opens a checkpoint — so the table regenerates from the
published rows with **no weights of any kind**:

```bash
python3 -c "
from importlib import resources
open('suite-v1.json','w').write(
    resources.files('f1sim.learn.benchmark').joinpath('suite-v1.example.json').read_text())
"

python3 -m f1sim.learn.benchmark report \
    --suite suite-v1.json \
    --roster benchmark-v1-2026-09-12-roster.json \
    --results benchmark-v1-2026-09-12-derived-cells.jsonl \
    --out leaderboard.md
```

Verified in an empty directory containing no checkpoints and no estimator: exit 0, `3 systems, 102
cells`, and every row identical to the table below. The packaged suite carries freeze
`6f710412bacf5a09`, which is the freeze these rows were scored against — `suite.load()` refuses a
mismatch, so a wrong suite cannot quietly produce a different table.

Use the **derived** file here: the raw one is what the reporter refuses for the missing `n_envs`,
which is the whole reason the derived extract exists.

To *score* fresh systems rather than re-render these, put the real checkpoint and estimator files at
the roster's relative paths. `plan` and `run` re-hash them against the recorded sha256 and refuse a
mismatch, so the paths being relative changes nothing about what is pinned.

Method, metric definitions and units: [Benchmark](../benchmark.md).

Two reminders when quoting the table: the "vs midpoint" columns render a **difference of fractions,
not percentage points** (multiply by 100 for pp), and avoidance `cleared/pre-validated` divides by
**all pre-validated spawned trials including approach failures**, not by obstacles encountered.

## Provenance

| | |
| --- | --- |
| suite freeze | `6f710412bacf5a09` |
| raw rows | 102, `398816e485000ac9` — unchanged |
| derived extract | `cb3b51060360be69` — raw **plus one key** |
| reporter | `python -m f1sim.learn.benchmark report`, exit 0 |

**Repair, stated plainly.** The runner in force at scoring time did not write a top-level `n_envs`,
and the reporter requires it. The published extract adds **only** that key, taken from the frozen
cell declaration (`Cell.envs` = 8), after independently checking on every row that `result.n`
already equalled the declared value — 102/102 repaired, 0 refused. No other field was added,
changed or removed, and the raw file is retained unmodified alongside it.

`__main__.py` hashed `55fd3b878381` at scoring time, identical to `SOURCE-FREEZE.md`, so these rows
were produced under the declared freeze. It has since been edited so future runs emit `n_envs`
(`-> 34bd0fb7730d`, with `test_run_path.py -> 459f98982d41`). That edit cannot retroactively add the
key to rows already written, which is why the derived extract exists.

`cl_main_estimated_s502` is excluded: a cancelled partial at 1/34 cells, shed by the VRAM guard when
nvidia-smi failed mid-run. `cl_main_estimated_s501` completed 34/34, but its exit code was never
collected — its supervisor was terminated before it could reap the child.

## Results

Below is `leaderboard-3systems.md` as the reporter wrote it, byte for byte.

<!-- BEGIN reporter output (verbatim) -->

# Checkpoint benchmark v1

Suite freeze `6f710412bacf5a09` · 3 systems · 102 cells

**Scope.** gen:control:1400 is an SGR training map; korea_2026_competition and gen:control:9100 are existing diagnostic maps. No unseen-map or generalisation claim attaches to any result.

**Friction levels.** 0.73423 low (MU_MIN) · 0.94401 mid (range midpoint) · 1.15379 high (MU_MAX) Nominal vehicle mu is 1.0489; the mid level is the *range midpoint*, not that nominal.

**No composite score.** Categories are reported separately with their own units and directions; rows are not ranked across suites.

## Driving

| system | runtime | completion ↑ | progress mean ↑ | progress p10 ↑ | progress p90 ↑ | lap time s ↓ (own completions only) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cl_main_estimated_s501@estimated` | estimated | 0.938 | 0.969 | 1.000 | 1.004 | 10.741 | 144 |
| `frozen_original@estimated` | estimated | 0.931 | 0.965 | 1.000 | 1.003 | 10.896 | 144 |
| `frozen_original@legacy` | legacy | 0.764 | 0.902 | 0.504 | 1.003 | 9.530 | 144 |

## Stability

| system | runtime | collisions/km ↓ | distance km | spins ↓ | large-slip s/km ↓ | max yaw rate rad/s (diagnostic) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cl_main_estimated_s501@estimated` | estimated | 6.955 | 9.058 | 0 | 3.991 | 4.112 | 272 |
| `frozen_original@estimated` | estimated | 6.067 | 9.066 | 0 | 3.420 | 4.123 | 272 |
| `frozen_original@legacy` | legacy | 12.069 | 8.286 | 0 | 7.163 | 5.126 | 272 |

## Surface

| system | runtime | completion ↑ | progress mean ↑ | completion vs midpoint ↑ | progress vs midpoint ↑ | large-slip s/km ↓ | centreline offset RMS m | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cl_main_estimated_s501@estimated` | estimated · mu=0.73423 | 0.854 | 0.942 | -0.125 | -0.040 | 4.946 | 0.368 | 48 |
| `cl_main_estimated_s501@estimated` | estimated · mu=0.94401 | 0.979 | 0.982 | 0.000 | 0.000 | 3.299 | 0.379 | 48 |
| `cl_main_estimated_s501@estimated` | estimated · mu=1.15379 | 0.979 | 0.983 | 0.000 | 0.001 | 1.691 | 0.389 | 48 |
| `frozen_original@estimated` | estimated · mu=0.73423 | 0.854 | 0.941 | -0.104 | -0.029 | 4.817 | 0.373 | 48 |
| `frozen_original@estimated` | estimated · mu=0.94401 | 0.958 | 0.971 | 0.000 | 0.000 | 1.952 | 0.387 | 48 |
| `frozen_original@estimated` | estimated · mu=1.15379 | 0.979 | 0.982 | 0.021 | 0.012 | 1.188 | 0.396 | 48 |
| `frozen_original@legacy` | legacy · mu=0.73423 | 0.333 | 0.744 | -0.646 | -0.237 | 12.695 | 0.388 | 48 |
| `frozen_original@legacy` | legacy · mu=0.94401 | 0.979 | 0.981 | 0.000 | 0.000 | 6.616 | 0.379 | 48 |
| `frozen_original@legacy` | legacy · mu=1.15379 | 0.979 | 0.982 | 0.000 | 0.001 | 2.862 | 0.386 | 48 |

## Avoidance

| system | runtime | cleared/pre-validated ↑ | encountered | approach failures ↓ | n |
| --- | --- | --- | --- | --- | --- |
| `cl_main_estimated_s501@estimated` | estimated | 0.594 | 64 | 0 | 64 |
| `frozen_original@estimated` | estimated | 0.641 | 64 | 0 | 64 |
| `frozen_original@legacy` | legacy | 0.281 | 64 | 0 | 64 |

## Overtaking

| system | runtime | passes held/race ↑ | hold interruptions | contact ↓ | n |
| --- | --- | --- | --- | --- | --- |
| `cl_main_estimated_s501@estimated` | estimated | 0.562 | 0 | 20 | 64 |
| `frozen_original@estimated` | estimated | 0.656 | 1 | 15 | 64 |
| `frozen_original@legacy` | legacy | 0.688 | 1 | 13 | 64 |

<!-- END reporter output -->

## Reading these numbers

**There is no universal winner, and none is claimed.**

`cl_main_estimated_s501` against `frozen_original@estimated`: slightly higher solo completion
(135/144 vs 134/144), identical low-friction completion (41/48 both), and **lower** avoidance
(38/64 vs 41/64) and overtaking (36/64 vs 42/64). Training helped solo driving marginally and cost
interaction performance.

`frozen_original@estimated` against `frozen_original@legacy`: better solo (134 vs 110 of 144),
much better low-friction (41 vs 16 of 48) and avoidance (41 vs 18 of 64), but **worse** overtaking
(42 vs 44 of 64). The estimated runtime is not a strict improvement.

Two column notes, since both are easy to misread. `cleared/pre-validated` divides by **all**
pre-validated spawned trials including those that failed on approach and never reached the
obstacle; `encountered` is a separate column. The surface `vs midpoint` columns are a **difference
of fractions**, not percentage points — `-0.125` is twelve and a half points of completion, not
one eighth of a point.

## Limits

Single machine, single GPU, one estimator pin (`a4e6fe02…`) across every estimated-arm system.
`gen:control:1400` is an SGR training map, so no unseen-map or generalisation claim attaches to any
row. Raw rows and the full manifest sit beside this file.
