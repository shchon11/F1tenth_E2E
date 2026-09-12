# Run `cl_free_s701` (2026-09-13 04:35): the leash loosened, plan clearance on, held-out checkpoint selection

Why: `cl_hard_events_s701` (recipe A + hard obstacles + events + new attitude model, 1M steps, KL 0.05)
scored the same as A701 and the frozen original on a held-out quick check (104 / 104 / 111 of 256), and
the user's own driving agreed. The A-long run had already shown the recipe plateaus. Diagnosis
(`docs/research/run-recipe-2026-09-13.md`, this session): the finetune is built to stay near the
original — KL leash 0.05, lr 5e-5 — and the reward does not target the failure. A crash map on the
held-out scene 2355 put 11 of 16 crashes on one *gapped* row of boxes; training rows were solid, so the
LiDAR never showed a crack that is not a way through (fixed in `hard_obstacles.py`, commit 11d71ec).

Changes against `cl_hard_events_s701`, nothing else:

| flag | before | now |
|---|---|---|
| `--kl-coef` | 0.05 | **0.01** |
| `--lr` / `--lr-end` | 5e-5 / 2e-5 | **1e-4 / 5e-5** |
| `--total` | 1 048 576 | **4 194 304** (512 updates) |
| `--plan-clearance-penalty` / `--plan-margin` | 0 / 0.15 | **1.0 / 0.25** (plan points nearer than 0.25 m to any occupied cell, boxes included, are charged per second) |
| `--save-every` | 10 | **8** |
| tracks | 204, solid rows | 204, **gapped rows** (seed 702 draws) |

Checkpoint selection: `select_ckpt.sh` scores every saved `ppo_u*.pt` while the run trains, on a fixed
held-out proxy under the legacy arm — obstacles (8 tracks: scene 2355 ± hard, `+hard` variants of
map16x07 / map12x16 / control:9100 / iccas25, clean map16x07 / map12x16; 32 trials each, 3-lap budget)
and traffic (map16x07 / control:9100 / iccas25, race size 2, teacher opponent with brake/stop/shift at
3.0 per 10 s). Best = most completions, ties by fewer collisions/km → `ppo_best.pt` and
`work/learning-next/free-20260913/select/selection.md`. The training log is not the criterion.
