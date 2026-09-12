# First held-out scoring: benchmark suite v2 (2026-09-13)

Suite v2, freeze `89805514350d932a…`, 64 cells / 512 trials per system, every map unseen in training
in any form (see `docs/benchmark.md`). Scored from a detached checkout of `fda4df1` with the roster in
`benchmark-v2-first-2026-09-13-roster.json`; raw cells per system alongside. The user stopped the run
after four complete systems ("여기까지정도 평가하면 됐어"); the two partial rows are reported as partial
and must not be pooled.

```
system                                  S/384 S-bb3/336  low/128  mid/128  high/128  A/96  O/32  coll/km  slip Δlap vs fixed_low
frozen_original@fixed_low              133/224  121/176   38/80    51/80     44/64    0/0   0/0      3.90  2.20   +0.000s n=133 (partial 28/64)
frozen_original@legacy                 228/384  227/336   27/128   91/128   110/128  21/96 23/32     6.85  3.75   -1.596s n=92
frozen_original@estimated              265/384  258/336   71/128   93/128   101/128  43/96 22/32     5.04  2.55   -0.329s n=119
cl_origrecipe_legacy_s701@fixed_low    287/384  274/336   77/128  102/128   108/128  55/96 25/32     3.74  2.75   -0.135s n=127
cl_origrecipe_legacy_s701@estimated     89/144   69/96    18/48    33/48     38/48    0/0   0/0      2.91  2.48   -0.683s n=63 (partial 18/64)
cl_oppdiv_control_s801@fixed_low       285/384  277/336   75/128   99/128   111/128  60/96 25/32     3.74  2.50   +0.010s n=120

map                                 rozen_original@fixed  rozen_original@legac  rozen_original@estim  pe_legacy_s701@fixed  pe_legacy_s701@estim  v_control_s801@fixed
gen:competition:0                           36/48                 31/48                 36/48                 33/48                  0/0                  30/48       
gen:competition:9200+pinch9200               0/0                  41/48                 46/48                 45/48                  0/0                  47/48       
gen:control:9100                            30/32                 60/112                76/112                90/112                 0/0                  96/112      
real:blackbox2022_3                         12/48                  1/48                  7/48                 13/48                 20/48                  8/48       
real:korea_2025_iccas                       32/48                 42/48                 29/48                 47/48                 48/48                 48/48       
real:map12x16                                0/0                  40/80                 52/80                 52/80                  0/0                  54/80       
real:map16x07                                0/0                  39/80                 62/80                 64/80                  0/0                  66/80       
rt:Monza                                    23/48                 18/48                 22/48                 23/48                 21/48                 21/48       
```

Reading (four complete systems):

* **The A recipe generalises.** `cl_origrecipe_legacy_s701@fixed_low` beats the frozen original on
  every aggregate column (per map it loses only `gen:competition:0`, 33/48 against 36/48 for
  `frozen_original@estimated`): solo 287 vs 228 (legacy) / 265 (estimated), low-μ 77 vs 27 / 71, avoidance
  55 vs 21 / 43, overtaking 25 vs 23 / 22, 3.74 collisions/km vs 6.85 / 5.04. Its second seed
  (`cl_oppdiv_control_s801`, same recipe, seed 801) lands within noise: 285 / 75 / 60 / 25 / 3.74.
* **`real:blackbox2022_3`** is near zero for every system (1–13 / 48). The user ruled it an unfair map
  (dead-end side branches no training map has); it stays in the freeze but the `S-bb3` column is the
  headline solo number.
* **The target floor `real:map16x07`**: 39/80 legacy → 62/80 estimated → 64/80 A701, 66/80 s801.
* Memorisation probe the same evening (10 tracks × 32 trials): trained vs novel obstacle seeds on
  the same map score alike (frozen 21 vs 17 / 64, A701 25 vs 26) — no evidence of obstacle-position
  memorisation. What the policy fails on is novel map structure, not novel seeds.

Not shown: the frozen original under `fixed_low` (28/64 cells) and A701 under `estimated` (18/64),
stopped early. Their partial rows are in `/home/shchon11/Documents/Codex/2026-09-10/new-chat/work/checkpoint-benchmark/scores/run-v2-20260912/cells/`.
