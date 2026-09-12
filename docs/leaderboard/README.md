# F1TENTH checkpoint leaderboard

As of `2026-09-12`. Rendered offline from results that were already measured: no new trials, no weights, no GPU.

Each cohort is validated separately and ranked separately. The cohorts were recorded under different benchmark source digests, so their rows are never pooled and their ranks are never shared.

This file is the GitHub-readable view. `index.html` in this directory is the same report with a cohort switch, search, sortable metric columns and per-checkpoint details; it is a self-contained offline page, so download it and open it in a browser -- GitHub does not run it for you.

## Recipe study (`recipe-study`)

Seven systems with identical suite, source digests and paired starts. A recipe replication, R10 and controller-recipe ablations.

- 7 systems, each on the same grid of 34 scenario cells = 272 trials per system
- 238 system-cells and 1904 trial outcomes in total (pooled over systems, not distinct scenarios)
- Suite `v1` freeze `6f710412bacf5a0936b06ba13013cc2ea6e92c269864e1f19df1a9ec3bca60ce`
- Benchmark source digest `30a057e563f0710c1bcbd213d4ad027632c1cffea845dad89ffc726923172b53`
- Solo friction levels 0.73423 · 0.94401 · 1.15379; the low-mu column is completion at the minimum of these (0.73423), which is not a measurement of friction-estimation accuracy
- Reference system for paired pace: `frozen_original@estimated`

Sorted by Solo completion, highest first. Percentages carry the exact numerator/denominator they were derived from. Exactly equal values share a competition rank.

| Rank | Checkpoint | Solo completion ↑ | Low-mu completion ↑ | Avoidance ↑ | Overtaking ↑ | Collisions/km ↓ | Large-slip s/km ↓ |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A recipe · seed 702<br>`cl_origrecipe_legacy_s702@estimated` | 94.4%<br>136/144 | 89.6%<br>43/48 | 67.2%<br>43/64 | 70.3%<br>45/64 | 5.25 | 2.97 |
| 2 | A recipe · seed 701<br>`cl_origrecipe_legacy_s701@estimated` | 93.1%<br>134/144 | 85.4%<br>41/48 | 68.8%<br>44/64 | 67.2%<br>43/64 | 5.60 | 3.33 |
| 2 | Legacy recipe · seed 501<br>`cl_recipe_legacy_s501@estimated` | 93.1%<br>134/144 | 87.5%<br>42/48 | 67.2%<br>43/64 | 70.3%<br>45/64 | 5.52 | 4.10 |
| 2 | Original + surface controller **(reference)**<br>`frozen_original@estimated` | 93.1%<br>134/144 | 85.4%<br>41/48 | 64.1%<br>41/64 | 65.6%<br>42/64 | 6.07 | 3.42 |
| 5 | R10 · seed 701<br>`cl_r10_base_s701@estimated` | 90.3%<br>130/144 | 77.1%<br>37/48 | 64.1%<br>41/64 | 64.1%<br>41/64 | 6.68 | 5.37 |
| 5 | C: restored optimizer · seed 701<br>`cl_sgr_base_restoreopt_s701@estimated` | 90.3%<br>130/144 | 77.1%<br>37/48 | 56.3%<br>36/64 | 57.8%<br>37/64 | 7.70 | 5.21 |
| 7 | R10 · seed 702<br>`cl_r10_base_s702@estimated` | 89.6%<br>129/144 | 75.0%<br>36/48 | 56.3%<br>36/64 | 59.4%<br>38/64 | 7.74 | 5.14 |

<details>
<summary>Checkpoint details, paired pace and provenance</summary>

| Checkpoint | Training | Controller | Paired pace vs reference (n) | Checkpoint sha256 | Estimator sha256 |
| --- | --- | --- | --- | --- | --- |
| `cl_origrecipe_legacy_s702@estimated` | Legacy · 149 track variants | estimated | +0.157 s (n=134) | `0d904012dd29e2de997a5ecf0e82faace8b4f4005862c3d8a2bf29e6af2348d2` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |
| `cl_origrecipe_legacy_s701@estimated` | Legacy · 149 track variants | estimated | +0.103 s (n=134) | `29e82233853868be33773ba0ea65d68268ba259b477828f019f7f106095055b6` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |
| `cl_recipe_legacy_s501@estimated` | Legacy · 5 maps · solo | estimated | +0.041 s (n=133) | `7b57da56edf4834459c83b693c3dc3f3eb4dc8066c9090d707c61dbf3cffe0d4` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |
| `frozen_original@estimated` | Original legacy recipe | estimated | N/A (reference system) | `48cc698f8c51feb53f60dc29fbc3e12eb8d002531a893b0d7bea0fd763dfbaef` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |
| `cl_r10_base_s701@estimated` | Estimated · 5 maps · solo | estimated | -0.193 s (n=130) | `3c5d2633f7361b2ff291873d5c36ae1e1a882c436ed62e5d68ca6bbbc8324131` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |
| `cl_sgr_base_restoreopt_s701@estimated` | Estimated · 5 maps · solo | estimated | -0.182 s (n=130) | `40c605440e7d48f32bd22e4deed1e4d2e335666bc328d744be8f876b2b8c9bc5` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |
| `cl_r10_base_s702@estimated` | Estimated · 5 maps · solo | estimated | -0.205 s (n=129) | `ce8c2a1ad4538af1d0fad365c1b98c6d78dceaeaf3f801b1234d55e2ff26c8fb` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |

Paired pace is the mean solo lap-time difference against the reference over the trials both systems completed. Negative is faster than the reference. It is descriptive with its own n, not a rank metric, and its sample differs for every pair.

Notes as declared in the manifest:

- `cl_origrecipe_legacy_s702@estimated` — Mixed traffic; 149 track variants include routes, directions and obstacle variations. Second seed of the A recipe; learning rate 5e-5 to 2e-5. Candidate, not a deployment promotion.
- `cl_origrecipe_legacy_s701@estimated` — Mixed traffic; 149 track variants include routes, directions and obstacle variations. Adam restored; learning rate 5e-5 to 2e-5. Candidate, not a deployment promotion.
- `cl_recipe_legacy_s501@estimated` — Fresh Adam; learning rate 1e-4.
- `frozen_original@estimated` — Comparison reference; estimator 401 at evaluation.
- `cl_r10_base_s701@estimated` — Reverse-direction data added; learning rate 1e-4.
- `cl_sgr_base_restoreopt_s701@estimated` — Adam restored; learning rate 5e-5 to 2e-5.
- `cl_r10_base_s702@estimated` — Second seed of R10; learning rate 1e-4.

Evidence (raw per-cell records, hashed as read):

- [`../../research/a702-replication-2026-09-12-raw-cells.jsonl`](../research/a702-replication-2026-09-12-raw-cells.jsonl) — 102 cells, sha256 `cff6b0ad4286b350091ca4c08ba2034413400b013f9859e680f703bff7410f7e`
- [`recipe-study-additional-cells.jsonl`](data/recipe-study-additional-cells.jsonl) — 136 cells, sha256 `bfec51dd9eb9ba080c3280d082433550289cc95251af95f0703244837c40d59e`

</details>

## Initial study (`initial-study`)

Three-system initial study. A different benchmark recorder digest prevents a shared ranking with the recipe study. The n_envs-only metadata derivation is documented in the original research note.

- 3 systems, each on the same grid of 34 scenario cells = 272 trials per system
- 102 system-cells and 816 trial outcomes in total (pooled over systems, not distinct scenarios)
- Suite `v1` freeze `6f710412bacf5a0936b06ba13013cc2ea6e92c269864e1f19df1a9ec3bca60ce`
- Benchmark source digest `979f744eb459a77d8bbf2c302b106e63c27f331e042231397a1900ffe5ee9375`
- Solo friction levels 0.73423 · 0.94401 · 1.15379; the low-mu column is completion at the minimum of these (0.73423), which is not a measurement of friction-estimation accuracy
- Reference system for paired pace: `frozen_original@estimated`

Sorted by Solo completion, highest first. Percentages carry the exact numerator/denominator they were derived from. Exactly equal values share a competition rank.

| Rank | Checkpoint | Solo completion ↑ | Low-mu completion ↑ | Avoidance ↑ | Overtaking ↑ | Collisions/km ↓ | Large-slip s/km ↓ |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Estimated retraining · seed 501<br>`cl_main_estimated_s501@estimated` | 93.8%<br>135/144 | 85.4%<br>41/48 | 59.4%<br>38/64 | 56.3%<br>36/64 | 6.95 | 3.99 |
| 2 | Original + surface controller **(reference)**<br>`frozen_original@estimated` | 93.1%<br>134/144 | 85.4%<br>41/48 | 64.1%<br>41/64 | 65.6%<br>42/64 | 6.07 | 3.42 |
| 3 | Original + legacy controller<br>`frozen_original@legacy` | 76.4%<br>110/144 | 33.3%<br>16/48 | 28.1%<br>18/64 | 68.8%<br>44/64 | 12.07 | 7.16 |

<details>
<summary>Checkpoint details, paired pace and provenance</summary>

| Checkpoint | Training | Controller | Paired pace vs reference (n) | Checkpoint sha256 | Estimator sha256 |
| --- | --- | --- | --- | --- | --- |
| `cl_main_estimated_s501@estimated` | Estimated · 5 maps · solo | estimated | -0.152 s (n=133) | `e7f01759708d541136b2165f1a05a9e5a21e63963fd35a732d6eacd75e591d87` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |
| `frozen_original@estimated` | Original legacy recipe | estimated | N/A (reference system) | `48cc698f8c51feb53f60dc29fbc3e12eb8d002531a893b0d7bea0fd763dfbaef` | `a4e6fe02e27e764fe11457abb802deff3e3566e4da10c457c438e518fa7ffbbf` |
| `frozen_original@legacy` | Original legacy recipe | legacy | -1.254 s (n=108) | `48cc698f8c51feb53f60dc29fbc3e12eb8d002531a893b0d7bea0fd763dfbaef` | `none` |

Paired pace is the mean solo lap-time difference against the reference over the trials both systems completed. Negative is faster than the reference. It is descriptive with its own n, not a rank metric, and its sample differs for every pair.

Notes as declared in the manifest:

- `cl_main_estimated_s501@estimated` — Initial estimated-controller retraining experiment.
- `frozen_original@estimated` — Comparison reference; estimator 401 at evaluation.
- `frozen_original@legacy` — Original weights evaluated with the legacy controller.

Evidence (raw per-cell records, hashed as read):

- [`../../research/benchmark-v1-2026-09-12-derived-cells.jsonl`](../research/benchmark-v1-2026-09-12-derived-cells.jsonl) — 102 cells, sha256 `cb3b51060360be6937145c375ba0977122cc61f10bff35a67b1313b2b5400333`

</details>

## Method

1. Each cohort's raw per-cell records are read and pushed through f1sim.learn.benchmark.report.validate_results with the frozen suite object, the declared cell grid, the declared trial counts, the declared system set and the roster. A missing cell, a duplicate, a short denominator, an edited row, an unpinned checkpoint, a mixed source digest or an unpaired start refuses the whole cohort, and nothing partial is written.
2. Metrics come from report.aggregate over the validated cells. Percentages are additionally derived from the validated outcome arrays as exact counts, and the two must agree exactly or the report is refused.
3. Ranks are competition ranks over the exact values, per cohort and per metric: equal exact values share a rank and the next rank skips. Rounded display values are never ranked. A missing measurement is N/A with its reason, carries no rank and sorts last.
4. Paired pace is the mean solo lap-time difference against the cohort reference over the trials both systems completed, reported with its n. It is descriptive and never ranked.
5. There is no composite or overall score: the six metrics have different units and different denominators.

## Scope and limitations

- A measured, representative development subset: three reused development maps, two seeds, eight trials per cell.
- Static per-episode friction. Each episode runs at one fixed friction level; no level changes inside an episode.
- No unseen-map, generalisation or on-car claim attaches to any number here. All maps were already in use during development.
- Low-mu completion is completion at the minimum declared solo friction level. It is not a measurement of friction-estimation accuracy.
- Paired pace is descriptive, not a rank metric: it is conditional on the trials where both systems completed, so its sample differs for every pair.
- Cohorts were recorded under different benchmark source digests. They are never ranked together.

## Regenerating

```sh
cd f1sim && python -m f1sim.learn.leaderboard --manifest ../docs/leaderboard/data/manifest.json --out-dir ../docs/leaderboard
```

Deterministic: the same manifest and the same result files produce byte-identical `index.html`, `README.md` and `leaderboard.json`. Nothing is written unless every cohort validates.
