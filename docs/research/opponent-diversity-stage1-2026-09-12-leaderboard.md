# Checkpoint benchmark v1

Suite freeze `6f710412bacf5a09` · 5 systems · 170 cells

**Scope.** gen:control:1400 is an SGR training map; korea_2026_competition and gen:control:9100 are existing diagnostic maps. No unseen-map or generalisation claim attaches to any result.

**Friction levels.** 0.73423 low (MU_MIN) · 0.94401 mid (range midpoint) · 1.15379 high (MU_MAX) Nominal vehicle mu is 1.0489; the mid level is the *range midpoint*, not that nominal.

**No composite score.** Categories are reported separately with their own units and directions; rows are not ranked across suites.

## Driving

| system | runtime | completion ↑ | progress mean ↑ | progress p10 ↑ | progress p90 ↑ | lap time s ↓ (own completions only) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cl_oppdiv_control_s801@estimated` | estimated | 0.944 | 0.970 | 1.000 | 1.005 | 11.102 | 144 |
| `cl_oppdiv_range_s801@estimated` | estimated | 0.944 | 0.970 | 1.000 | 1.004 | 11.086 | 144 |
| `cl_origrecipe_legacy_s701@estimated` | estimated | 0.931 | 0.965 | 1.000 | 1.004 | 10.999 | 144 |
| `cl_origrecipe_legacy_s702@estimated` | estimated | 0.944 | 0.970 | 1.000 | 1.004 | 11.073 | 144 |
| `frozen_original@estimated` | estimated | 0.931 | 0.965 | 1.000 | 1.003 | 10.896 | 144 |

## Stability

| system | runtime | collisions/km ↓ | distance km | spins ↓ | large-slip s/km ↓ | max yaw rate rad/s (diagnostic) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cl_oppdiv_control_s801@estimated` | estimated | 4.802 | 9.162 | 0 | 2.562 | 3.882 | 272 |
| `cl_oppdiv_range_s801@estimated` | estimated | 4.867 | 9.246 | 0 | 3.128 | 3.869 | 272 |
| `cl_origrecipe_legacy_s701@estimated` | estimated | 5.596 | 9.114 | 0 | 3.333 | 4.151 | 272 |
| `cl_origrecipe_legacy_s702@estimated` | estimated | 5.254 | 9.136 | 0 | 2.966 | 4.262 | 272 |
| `frozen_original@estimated` | estimated | 6.067 | 9.066 | 0 | 3.420 | 4.123 | 272 |

## Surface

| system | runtime | completion ↑ | progress mean ↑ | completion vs midpoint ↑ | progress vs midpoint ↑ | large-slip s/km ↓ | centreline offset RMS m | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cl_oppdiv_control_s801@estimated` | estimated · mu=0.73423 | 0.896 | 0.957 | -0.062 | -0.014 | 3.374 | 0.370 | 48 |
| `cl_oppdiv_control_s801@estimated` | estimated · mu=0.94401 | 0.958 | 0.971 | 0.000 | 0.000 | 1.432 | 0.384 | 48 |
| `cl_oppdiv_control_s801@estimated` | estimated · mu=1.15379 | 0.979 | 0.983 | 0.021 | 0.012 | 0.713 | 0.395 | 48 |
| `cl_oppdiv_range_s801@estimated` | estimated · mu=0.73423 | 0.896 | 0.957 | -0.062 | -0.014 | 4.270 | 0.377 | 48 |
| `cl_oppdiv_range_s801@estimated` | estimated · mu=0.94401 | 0.958 | 0.971 | 0.000 | 0.000 | 2.288 | 0.388 | 48 |
| `cl_oppdiv_range_s801@estimated` | estimated · mu=1.15379 | 0.979 | 0.983 | 0.021 | 0.011 | 1.297 | 0.397 | 48 |
| `cl_origrecipe_legacy_s701@estimated` | estimated · mu=0.73423 | 0.854 | 0.942 | -0.104 | -0.029 | 4.482 | 0.376 | 48 |
| `cl_origrecipe_legacy_s701@estimated` | estimated · mu=0.94401 | 0.958 | 0.971 | 0.000 | 0.000 | 2.247 | 0.388 | 48 |
| `cl_origrecipe_legacy_s701@estimated` | estimated · mu=1.15379 | 0.979 | 0.982 | 0.021 | 0.011 | 1.290 | 0.399 | 48 |
| `cl_origrecipe_legacy_s702@estimated` | estimated · mu=0.73423 | 0.896 | 0.957 | -0.062 | -0.014 | 3.395 | 0.365 | 48 |
| `cl_origrecipe_legacy_s702@estimated` | estimated · mu=0.94401 | 0.958 | 0.971 | 0.000 | 0.000 | 1.584 | 0.385 | 48 |
| `cl_origrecipe_legacy_s702@estimated` | estimated · mu=1.15379 | 0.979 | 0.982 | 0.021 | 0.011 | 1.079 | 0.397 | 48 |
| `frozen_original@estimated` | estimated · mu=0.73423 | 0.854 | 0.941 | -0.104 | -0.029 | 4.817 | 0.373 | 48 |
| `frozen_original@estimated` | estimated · mu=0.94401 | 0.958 | 0.971 | 0.000 | 0.000 | 1.952 | 0.387 | 48 |
| `frozen_original@estimated` | estimated · mu=1.15379 | 0.979 | 0.982 | 0.021 | 0.012 | 1.188 | 0.396 | 48 |

## Avoidance

| system | runtime | cleared/pre-validated ↑ | encountered | approach failures ↓ | n |
| --- | --- | --- | --- | --- | --- |
| `cl_oppdiv_control_s801@estimated` | estimated | 0.781 | 64 | 0 | 64 |
| `cl_oppdiv_range_s801@estimated` | estimated | 0.734 | 64 | 0 | 64 |
| `cl_origrecipe_legacy_s701@estimated` | estimated | 0.688 | 64 | 0 | 64 |
| `cl_origrecipe_legacy_s702@estimated` | estimated | 0.672 | 64 | 0 | 64 |
| `frozen_original@estimated` | estimated | 0.641 | 64 | 0 | 64 |

## Overtaking

| system | runtime | passes held/race ↑ | hold interruptions | contact ↓ | n |
| --- | --- | --- | --- | --- | --- |
| `cl_oppdiv_control_s801@estimated` | estimated | 0.656 | 1 | 10 | 64 |
| `cl_oppdiv_range_s801@estimated` | estimated | 0.688 | 2 | 15 | 64 |
| `cl_origrecipe_legacy_s701@estimated` | estimated | 0.672 | 0 | 10 | 64 |
| `cl_origrecipe_legacy_s702@estimated` | estimated | 0.703 | 2 | 11 | 64 |
| `frozen_original@estimated` | estimated | 0.656 | 1 | 15 | 64 |
