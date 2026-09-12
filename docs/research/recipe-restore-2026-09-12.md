# Why every finetune under the grip clamp regressed, and the recipe that does not

**2026-09-12.** Follow-up to the [R10 result](benchmark-v1-r10-2026-09-12.md). Three cheap tests on
frozen suite v1 (freeze `6f710412bacf5a09`, 34 cells / 272 trials per system, all scored under the
`estimated` controller arm with estimator seed 401) to separate three explanations for the
low-friction / avoidance / overtaking regression seen in every retrained checkpoint so far.

| hypothesis | test | result |
| --- | --- | --- |
| training distribution narrowed (5 maps, solo, no obstacles or opponents) | **A**: finetune from the frozen original with the original's own recipe: `--tracks train` (149 variants incl. reversed/mirrored, obstacle and pinch variants), `race_size 2`, mixed opponents, aux grip/opp heads, legacy controller | matches the reference on driving, better on avoidance and collisions |
| optimizer shock (every previous finetune used `--fresh-opt` at lr 1e-4 on a 139 M-step policy) | **C**: the SGR `base` recipe (5 maps, solo, estimated arm) with Adam moments restored and lr 5e-5 to 2e-5 | **same regression as R10 / SGR** |
| the policy learns to exploit the speed clamp when trained under it | **legacy-trained finetune scored under the clamp**: existing `cl_recipe_legacy_s501` (5 maps solo, legacy arm, `--fresh-opt`, lr 1e-4) declared `cross_runtime` | at or above the reference on every category |

**Reading.** The two policies trained under the *legacy* controller (A with the full recipe, s501 with
the narrow one) both hold the reference under the clamp. The four policies trained *under* the clamp
(s501-estimated, R10 701/702, C) all lose low-mu completion, avoidance and overtaking while gaining
about 0.2 s of lap time on commonly completed trials, regardless of map set, optimizer state or
learning rate. That isolates the controller-in-the-loop during training as the cause: PPO learns
plans that lean on the clamp. Map narrowing and optimizer shock are not the primary cause: C removes
the optimizer confound and still regresses; s501-legacy keeps the narrow maps and still holds.

**Decision.** Train under the legacy controller; apply the grip-aware clamp at deployment.
`cl_origrecipe_legacy_s701` (A) is the best system on this suite and the candidate baseline for opponent and
obstacle work. One seed each; a second seed of A is the next confirmation.

## Systems and provenance

| system | weights sha256 | trained under | W&B | notes |
| --- | --- | --- | --- | --- |
| `frozen_original@estimated` | `48cc698f8c51feb53f60dc29fbc3e12eb8d002531a893b0d7bea0fd763dfbaef` | legacy (ppo_race_0910, 138.8 M steps) | `qxkniuk2` | published reference |
| `cl_recipe_legacy_s501@estimated` | `7b57da56edf4834459c83b693c3dc3f3eb4dc8066c9090d707c61dbf3cffe0d4` | legacy, 5 maps solo, fresh Adam, lr 1e-4, 1 M steps | `bjal4ve1` | existing checkpoint, scored here as `cross_runtime` |
| **`cl_origrecipe_legacy_s701@estimated`** (A) | `29e82233853868be33773ba0ea65d68268ba259b477828f019f7f106095055b6` | legacy, original recipe, Adam restored, lr 5e-5 to 2e-5, 1 M steps, 44 min | `ltyzel7v` | `cross_runtime` |
| `cl_sgr_base_restoreopt_s701@estimated` (C) | `40c605440e7d48f32bd22e4deed1e4d2e335666bc328d744be8f876b2b8c9bc5` | estimated, 5 maps solo, Adam restored, lr 5e-5 to 2e-5, 1 M steps, 13 min | `yg6adv7v` | |

Both new runs start from `frozen_original_48cc698f.pt` with the checkpoint's Adam moments restored
(`optimizer state restored`, `re-initialized: nothing` in the logs). Both were scored in one process
each, concurrently, `OMP/MKL=1`, RTX 4060 Ti, rc 0, 34/34 cells, no timeouts. The runtime source
digest is identical to the published rows; the benchmark package digest differs only in
`__main__.py` (the post-scoring `n_envs` fix), which is why the comparison below is computed from
per-cell tallies rather than pooled by `report`. The same method reproduces the published reporter
columns exactly.

A's full argv (C differs in `--tracks`, `--race-size 1`, aux weights 0,
`--critic-priv-adapter absent_opponent_17_to_21`, `--controller estimated --estimator ...`):

```
python -m f1sim.learn.ppo --action-mode plan --horizon 32 --minibatch 1024 --epochs 3 --cap0 9.0 --cap1 9.0 --cap-steps 5000000.0 --cap-gate 4.0 --cap-gate-quantile 0.9 --cap-gate-min-km 1.0 --kl-coef 0.05 --kl-decay 40000000.0 --collision-penalty 10.0 --steer-penalty 0.05 --proximity-penalty 0.5 --safe-dist 0.3 --sideslip-penalty 0.5 --lap-bonus 5.0 --lap-time-bonus 2.0 --collision-speed-penalty 0.5 --critic-warmup 10 --episode-s 40.0 --scan-stack 6 --scan-stride 1 --hist-len 20 --gamma 0.99 --lam 0.95 --clip 0.2 --ent 0.0 --vf 0.5 --max-grad 0.5 --envs 256 --total 1048576.0 --lr 5e-05 --lr-end 2e-05 --init /home/shchon11/f1sim_runs/_baselines/frozen_original_48cc698f.pt --seed 701 --device cuda --wandb online --log-every 1 --wandb-group recipe-restore-2026-09-12 --amp --wandb-new --save-every 10 --tracks train --race-size 2 --opponent mixed --mixed-teacher-frac 0.5 --opp-speed 0.5 1.0 --overtake-bonus 1.0 --car-proximity-penalty 0.8 --car-safe-gap 0.9 --car-contact-penalty 5.0 --aux-grip 1.0 --aux-opp 1.0 --controller legacy --name cl_origrecipe_legacy_s701
```

## Results (successes / trials; S = solo, A = avoidance, O = overtaking)

```
system                                 S/144  low/48  mid/48  high/48   A/64   O/64  coll/km   slip  pairedΔlap
frozen_original@legacy                   110      16      47       47     18     44    12.07   7.16   -1.254s n=108
frozen_original@estimated                134      41      46       47     41     42     6.07   3.42   +0.000s n=134
cl_main_estimated_s501@estimated         135      41      47       47     38     36     6.95   3.99   -0.152s n=133
cl_r10_base_s701@estimated               130      37      46       47     41     41     6.68   5.37   -0.193s n=130
cl_r10_base_s702@estimated               129      36      46       47     36     38     7.74   5.14   -0.205s n=129
cl_origrecipe_legacy_s701@estimated      134      41      46       47     44     43     5.60   3.33   +0.103s n=134
cl_recipe_legacy_s501@estimated          134      42      45       47     43     45     5.52   4.10   +0.041s n=133
cl_sgr_base_restoreopt_s701@estimated     130      37      46       47     36     37     7.70   5.21   -0.182s n=130
```

`pairedΔlap` is the mean lap-time difference against `frozen_original@estimated` over S trials both
systems completed from the same seeded start; negative is faster.

### Per cell: low-mu solo and all avoidance / overtaking cells

| cell (/16) | orig@est | legacy s501 | **A** | C |
| --- | ---: | ---: | ---: | ---: |
| S control:1400 low | 15 | 15 | 15 | 15 |
| S control:9100 low | 13 | 14 | 13 | 12 |
| S korea low | 13 | 13 | 13 | 10 |
| A control:1400 low / mid | 16 / 16 | 15 / 16 | 16 / 16 | 16 / 16 |
| A control:9100 low / mid | 2 / 7 | 4 / 8 | 4 / 8 | 0 / 4 |
| O control:1400 low / mid | 11 / 9 | 12 / 9 | 12 / 8 | 9 / 9 |
| O control:9100 low / mid | 10 / 12 | 11 / 13 | 12 / 11 | 8 / 11 |

## Files

| file | contents |
| --- | --- |
| [`recipe-restore-2026-09-12-cl_origrecipe_legacy_s701-cells.jsonl`](recipe-restore-2026-09-12-cl_origrecipe_legacy_s701-cells.jsonl) | A, 34 rows |
| [`recipe-restore-2026-09-12-cl_sgr_base_restoreopt_s701-cells.jsonl`](recipe-restore-2026-09-12-cl_sgr_base_restoreopt_s701-cells.jsonl) | C, 34 rows |
| [`recipe-restore-2026-09-12-cl_recipe_legacy_s501-cells.jsonl`](recipe-restore-2026-09-12-cl_recipe_legacy_s501-cells.jsonl) | legacy s501 under the clamp, 34 rows |
| [`recipe-restore-2026-09-12-roster.json`](recipe-restore-2026-09-12-roster.json) | all eight systems as scored (absolute pins) |

Scope as always: three reused development maps, static mu per episode, no unseen-map or on-car
claim. `ppo.py` refuses `--controller estimated` for `race_size > 1` ("validated solo only"), so the
arm "original recipe trained under the clamp" could not be run; given the result above it is also no
longer the interesting arm.
