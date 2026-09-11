# Checkpoint benchmark

A development leaderboard for comparing trained checkpoints on fixed scenarios: driving, stability,
per-surface retention, obstacle avoidance and overtaking. It ships inside the installed package, so
it runs from a normal `pip install -e f1sim` with no extra path setup.

```bash
python3 -m f1sim.learn.benchmark --help
```

A measured run of this suite is published as a research note:
[Checkpoint benchmark v1 — three-system subset](research/benchmark-v1-2026-09-12.md) (3 systems, 102 cells, 816 trials). It is a subset chosen for time, not the 17-system benchmark.

**What this is not.** All three maps are reused development tracks, so no result here is evidence of
generalisation to an unseen venue, and nothing here is an on-car claim. Scope is restated in every
generated report.

## The pipeline

| stage | loads a checkpoint? | needs a GPU? | what it does |
| --- | --- | --- | --- |
| `plan` | no | no | matrix, trial counts, cost projection, roster **file-pin** verification |
| `geometry` | no | no | proves obstacle placements, optionally times a smoke run, then **freezes** the suite |
| `gate` | no | **no — runs on CPU** | proves the opponent does not react to the candidate's controller |
| `feasibility` | no | no | scripted-expert check that each scenario is achievable at all |
| `run` | **yes** | yes, with `--lease` | scores one pinned system against the frozen suite |
| `report` | no | no | validates raw results and renders the leaderboard |

`plan` and `geometry` never load an evaluated checkpoint. That ordering is deliberate: scenario
choice cannot be influenced by how a candidate happens to score on it. `run` refuses to score against
a suite that is not frozen.

## The suite

`plan` prints the matrix without touching a checkpoint:

```console
$ python3 -m f1sim.learn.benchmark plan
suite v1  freeze (not frozen)
cells 34   trials/system 272   S=144  A=64  O=64
projected cost: unavailable (pass --map-lengths for the pinned geometry file)
```

Three scenario families, resolved from the suite definition:

| family | maps | µ | seeds | cells | trials |
| --- | --- | --- | --- | ---: | ---: |
| **S** solo | `gen:control:1400`, `real:korea_2026_competition`, `gen:control:9100` | 0.73423, 0.94401, 1.15379 | 4401, 4402 | 18 | 144 |
| **A** avoidance | `gen:control:1400`, `gen:control:9100` | 0.73423, 0.94401 | 4401, 4402 | 8 | 64 |
| **O** overtaking | `gen:control:1400`, `gen:control:9100` | 0.73423, 0.94401 | 4401, 4402 | 8 | 64 |

**34 cells, 272 trials per system.** S runs 8 environments per cell; O runs 8 learners against
`race_size` 2, so 16 cars. A cell id is `suite:map:mu:seed`, e.g.
`S:gen:control:1400:0.73423:4401`.

Friction is **static per episode**: µ is set per cell and constant for the whole run. Only a reset
may resample it. High µ appears in S only — A and O use low and mid, where the interesting failures
are.

Protocol fields that change outcomes live in the suite and are hashed with it: `speed_cap` 9.0 m/s,
`budget_laps` 3.0, `sensor_noise` on, opponent `teacher` at `opp_speed_range` (0.6, 0.8),
`hold_seconds` 1.0, `obstacle_window_m` 4.0. A benchmark whose speed cap drifted silently is not the
same benchmark, so these are part of the freeze rather than command-line options.

### Freezing

```bash
python3 -m f1sim.learn.benchmark geometry --suite suite-v1.json --measure --freeze
```

This proves each obstacle placement actually blocks the racing line while leaving a traversable
corridor, and writes the suite with a `freeze_sha256`. **`--measure` is what times a teacher-driven
cell per map**; without it no timing is taken and the suite's `calibration.measured` is `false`.
`suite.load()`
recomputes that hash and **refuses a file edited after freezing**. Placement geometry and the
measured calibration are recorded in the suite; the calibration is explicitly excluded from the
freeze hash, because re-measuring a rate is a fact about the machine rather than about the scenario.

Run `geometry` before any system is scored, and do not re-run it to "fix" a scenario after seeing
results.

## The roster

The roster is a JSON object with a `systems` list pinning exactly which weights are being compared —
**not a bare array**. Every entry is verified before anything is scored.

```json
{
  "systems": [
    {
      "system_id": "sgr_aux_anchor_s701",
      "path": "/abs/path/cl_sgr_aux_anchor_s701/ppo_final.pt",
      "checkpoint_sha256": "…64 hex…",
      "controller_arm": "estimated",
      "estimator_path": "/abs/path/estimator_seed401.pt",
      "estimator_sha256": "a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf",
      "note": "SGR screen final, seed 701"
    },
    {
      "system_id": "frozen_original_estimated",
      "path": "/abs/path/baseline/ppo_latest_frozen.pt",
      "checkpoint_sha256": "…64 hex…",
      "controller_arm": "estimated",
      "estimator_path": "/abs/path/estimator_seed401.pt",
      "estimator_sha256": "a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf",
      "cross_runtime": true,
      "note": "legacy-trained weights evaluated under the estimated arm — declared deliberately"
    }
  ]
}
```

| field | meaning |
| --- | --- |
| `system_id` | unique label; duplicates are refused |
| `path` | the real file. Not a symlink, and no `latest` path segment |
| `checkpoint_sha256` | re-hashed at verification; a mismatch refuses |
| `controller_arm` | `legacy`, `fixed_low` or `estimated` |
| `estimator_path` / `estimator_sha256` | required **iff** the arm is `estimated`; both, not one |
| `cross_runtime` | declares a checkpoint evaluated under an arm it was not trained under |
| `note` | free text, carried into the report |

### What verification refuses

- **Moving pointers.** `ppo_latest.pt`, `latest.pt` and `last.pt` are rewritten in place by training.
  A roster may not point at one — pin an immutable copy. Note this is a check on the *basename*, so a
  legitimately frozen file such as `ppo_latest_frozen.pt` is fine.
- **Symlinks**, and any unresolved `latest` directory segment.
- **A drifted file** — the sha is recomputed, not trusted.
- **A bare estimator sha with no resolvable file.** A sha the runtime cannot load is not a pin, and
  it must fail here rather than hours later on the GPU.
- **Arm/estimator disagreement** in either direction: an `estimated` entry without an estimator, or a
  non-`estimated` entry carrying one.
- **An arm that contradicts the checkpoint's own record** — see below.
- **Duplicate identity** — two entries with the same (weights, arm, estimator) triple, even under
  different `system_id`s.

### Cross-runtime entries

A checkpoint records the arm it trained under, and `cross_runtime` covers exactly one case:

| checkpoint recorded as | declared arm | result |
| --- | --- | --- |
| `legacy` | `legacy` | fine |
| `legacy` | `estimated` / `fixed_low` | allowed **only** with `"cross_runtime": true` |
| `estimated` | `estimated` | fine |
| `estimated` | anything else | **refused — `cross_runtime` does not override this** |

So `cross_runtime` is not a general escape hatch. A non-legacy checkpoint may only run under its own
arm, because a policy trained against one controller is not comparable under another; only
legacy-trained weights may be lifted into a non-legacy runtime, and then only when declared, so it
cannot arise from a typo. That case is how the frozen original enters a leaderboard whose other
systems trained under `estimated`.

The same weights may appear twice under different arms; those are two systems, and the report keeps
them distinct by runtime.

Verify a roster without scoring anything:

```bash
python3 -m f1sim.learn.benchmark plan --suite suite-v1.json --roster roster.json \
    --map-lengths map-geometry-pins.json
```

This prints each system with its short shas and arm, the unique-weight count, total trials and a
projected GPU-hour figure. The projection is a measured per-cell rate scaled by raceline length with
an inferred race multiplier — rough in both directions, and no substitute for the geometry smoke.

**`plan` checks file pins only** — paths, symlinks, moving pointers, shas, and arm/estimator
consistency *within the entry*. It does **not** open the checkpoint to compare the declared arm
against the arm the checkpoint recorded. That check runs in `run`, and it runs even without
`--lease`, so this is the cheapest way to catch a mis-declared arm before booking a GPU:

```bash
python3 -m f1sim.learn.benchmark run --suite suite-v1.json --roster roster.json \
    --system <id> --estimator /abs/path/estimator_seed401.pt          # no --lease
```

It verifies the pin, checks the recorded arm, reports how many cells are ready, and then refuses to
score (exit `3`).

## Independence of the opponent

In the O family the opponent must not react to the candidate's *controller*. That is not obvious: the
teacher's plans and the candidate's are pushed through one tracker, and the teacher opponents
legitimately react to the candidate's arc position and speed, so simply running both and comparing
proves nothing.

```bash
python3 -m f1sim.learn.benchmark gate --estimator /abs/path/estimator_seed401.pt --steps 60
```

The gate restores identical state and probes it under each arm, then checks two things:

- **routed** — opponent commands must be *identical* across arms. Any difference means the opponent
  moved with the candidate.
- **unrouted control** — a deliberately perturbed run that must *differ*. Without it a gate that
  compares nothing would pass, which is a vacuous pass, and the CLI reports that case separately.

It also requires both **cold and warm** estimator history to be covered: `--steps` must exceed the
estimator's warm-frame count or warm coverage cannot be claimed. Exit codes: `1` opponent moved,
`2` control did not diverge (vacuous), `3` cold+warm not covered.

`--estimator` is required — the `estimated` arm has no default. There is **no device option**: the
gate builds its environments on CPU, so it can be run before any GPU is booked.

## Scoring

```bash
python3 -m f1sim.learn.benchmark run \
    --suite suite-v1.json \
    --roster roster.json \
    --system sgr_aux_anchor_s701 \
    --estimator /abs/path/estimator_seed401.pt \
    --device cuda --lease \
    --out results/
```

Without `--lease`, `run` verifies everything and then refuses (exit `3`), printing what it would have
scored.

**`--lease` is a caller acknowledgement, not a lock.** It takes no OS or GPU lock and reserves
nothing; it is you asserting that exclusive access was arranged. Nothing stops a second process from
starting on the same card, so the coordination is yours to do.

Before any cell runs, `run` requires all of: a frozen suite, a verified pin whose recorded arm
matches, proven placements, and a passing independence gate. Each is reported as itself rather than
surfacing as a confusing failure later.

Results are written per cell as they complete. Re-running resumes: existing cells are matched on full
identity — suite freeze, protocol, checkpoint sha, estimator sha and arm — not on cell id alone, and
a duplicate cell id or a corrupt line is refused rather than silently accepted.

**Concurrency.** Budget with *resident* VRAM, not with torch's own counters: a scoring process also
pays a fixed CUDA context and cuBLAS/cuDNN workspace cost that `max_memory_allocated()` cannot see,
and that cost is per process. Two processes is a reasonable default on a single 8 GiB card; verify on
yours before assuming it.

## Reading the report

`run` writes one file per system, `results/<system_id>.cells.jsonl`. **`report --results` takes a
single file, not a directory** — either one `.jsonl` of cell records, or a `.json` payload with a
`cells` list. So concatenate first:

```bash
cat results/*.cells.jsonl > benchmark-cells.jsonl

python3 -m f1sim.learn.benchmark report --suite suite-v1.json --roster roster.json \
    --results benchmark-cells.jsonl --out docs/benchmarks/leaderboard.md
```

Write the combined file to a **separate name**, not into `results/` — a `.cells.jsonl` inside that
directory would be picked up by the next glob and double-count every row.

A payload carrying only a precomputed `summary` is refused: everything rendered has to be derived
from raw per-cell records, or nothing ties the table to a measurement.

Validation happens **before** aggregation: rows are checked against the declared cells and roster.
Comparisons also require a single observation spec — if the rows span more than one,
`assert_single_obs_spec` **refuses** and names the split rather than reporting groups side by side.
Systems built on different input layouts therefore need their own roster, their own results file and
their own report; they are separate benchmarks, not two rows of one table.

### Metrics

| category | columns | units |
| --- | --- | --- |
| **Driving** | completion ↑, progress mean ↑, progress p10 ↑, progress p90 ↑, lap time ↓ | fraction, fraction of route, seconds |
| **Stability** | collisions/km ↓, distance km, spins ↓, large-slip s/km ↓, max yaw rate | per km, km, count, s/km, rad/s |
| **Surface** | completion ↑, progress mean ↑, completion vs midpoint ↑, progress vs midpoint ↑, large-slip s/km ↓, centreline offset RMS | fraction, fraction, pp, fraction, s/km, metres |
| **Avoidance** | cleared/pre-validated ↑, encountered, approach failures ↓ | fraction, count, count |
| **Overtaking** | passes held/race ↑, hold interruptions, contact ↓ | per race, count, count |

- **Progress** is a *signed* route fraction, so driving backwards does not accumulate credit.
- **Lap time** is over the system's **own completions only**, and is therefore not comparable across
  systems with different completion rates — a system that fails the hard cells can look fast.
- **Completion vs midpoint** compares against µ 0.94401, which is the *midpoint of the tested range*.
  It is not the vehicle's nominal friction — that is 1.0489 — so "mid" should not be read as the
  design point. The generated report repeats this next to the table.
- **"vs midpoint" columns are rendered as a difference of fractions, not percentage points.** A
  value of `-0.05` means five percentage points worse; multiply by 100 to state it in pp.
- **Max yaw rate** is labelled diagnostic: it is a single extreme value, not a rate.
- **Surface** reports per µ, which is the point of the category — a system may hold mid and high while
  losing low.
- **Avoidance** — `cleared/pre-validated` divides by **all pre-validated spawned trials**, which
  includes trials that failed on the approach and never reached the obstacle. It is not "cleared ÷
  obstacles encountered": a system that crashes before arriving is counted as not having cleared it,
  which is the intended reading. `encountered` is reported alongside as its own column, and approach
  collisions and timeouts are broken out as `approach failures`.
- **Overtaking** requires a pass to be *held*, not merely achieved: a pass counts once the candidate
  is clear ahead and stays clear for `hold_seconds`. Clearance comes from the declared vehicle
  footprint rather than a hand-tuned distance.

### N/A is not zero

An unmeasured value renders as `N/A (reason)` and never as `0`. Rendering an absent measurement as
zero would assert "no spins observed" where the truth is "spins were not measured", which is how a
system with missing instrumentation comes to look like the most stable one. A category with no
measurement at all prints `N/A (category not evaluated)`.

## Provenance in every result

Each cell records a **start fingerprint** taken immediately after the final seeded reset, before the
controller initialises and before any action: three separate digests — physical state, actor input,
and calibration — plus tensor counts, `sim_t`, IMU phase, observation-spec hash, total cars and race
size.

Three digests rather than one combined hash, so a disagreement names the layer it happened in: a
physical mismatch is a different starting state, an actor-input mismatch is a different observation,
and a calibration mismatch is a different vehicle. Digests are taken over the original tensor bytes
including each field's name, dtype and shape, so a reshape or a dtype change cannot pass as identical
and a 1e-10 difference is not rounded away.

Comparisons refuse unless the starts match. Every row is validated before grouping or comparison,
including a lone row — a single row is where "paired" is least earned. The recorded cars/race-size
must also match the declared cell, so contradictory batch metadata cannot reach a paired comparison.

Each result additionally carries a **source digest**: the sha256 of every runtime module a
measurement depends on plus the benchmark's own code. A result measured against changed source is
identifiable as such.

## Adding a checkpoint later

The suite is frozen; the roster is not. To add a newly completed checkpoint:

1. Copy the checkpoint to an immutable path — not `ppo_latest.pt`, which training rewrites.
2. Append an entry with its sha, arm, and estimator pin if the arm is `estimated`.
3. `plan --roster` to verify the pin resolves and the arm matches the checkpoint's record.
4. `run` that one `--system`. **Do not re-run existing systems**: they were scored against this same
   frozen suite, and their results stay valid.
5. `report` over the whole results directory.

Adding a system does not invalidate earlier rows, which is the reason the suite freeze and the
per-row source digest exist. What *would* invalidate them is editing the suite — `load()` refuses
that outright — or changing a runtime module, which the source digest makes visible.

If a new checkpoint expects a different observation layout, it is still admissible: the report groups
by observation spec and reports the groups separately rather than pooling them.

## Reproducing an existing table

**Start from the packaged suite, not from `geometry`.** Re-running `geometry` regenerates placements
from the current defaults (`--s-obs 20.0`), which produces a *different* suite with a different
freeze hash — scenarios that were never the ones scored. The v1 suite ships inside the package:

```bash
python3 -c "
from importlib import resources
src = resources.files('f1sim.learn.benchmark').joinpath('suite-v1.example.json')
open('suite-v1.json','w').write(src.read_text())
"
python3 -m f1sim.learn.benchmark plan --suite suite-v1.json     # confirm the freeze hash
```

Then:

```bash
# 1. verify the roster resolves and every pin still matches
python3 -m f1sim.learn.benchmark plan --suite suite-v1.json --roster roster.json

# 2. prove the opponent is independent of the candidate controller (CPU)
python3 -m f1sim.learn.benchmark gate --estimator /abs/path/estimator_seed401.pt --steps 60

# 3. score each system (GPU, one at a time or two concurrent)
python3 -m f1sim.learn.benchmark run --suite suite-v1.json --roster roster.json \
    --system <id> --estimator /abs/path/estimator_seed401.pt --device cuda --lease --out results/

# 4. combine the per-system files, then render
cat results/*.cells.jsonl > benchmark-cells.jsonl
python3 -m f1sim.learn.benchmark report --suite suite-v1.json --roster roster.json \
    --results benchmark-cells.jsonl --out docs/benchmarks/leaderboard.md
```

A reproduction is only meaningful if the **suite freeze hash matches** the published one and step 1
passes unchanged. If the hash differs, the scenarios differ and the numbers are not comparable —
which is precisely why the suite is copied rather than regenerated.

## Checking it without a GPU

```bash
python3 -m f1sim.learn.benchmark feasibility --suite suite-v1.json --envs 4 --device cpu
```

This drives the scenarios with a scripted expert on synthetic metadata to confirm each one is
achievable at all. It loads **no candidate weights and is not a benchmark score** — it answers "is
this scenario possible?", not "how good is this policy?". A scenario the expert cannot complete
should be fixed or dropped before the freeze, never after seeing candidate results.
