# Consolidated training run `cl_hard_events_s701` (2026-09-13)

One run after all training-relevant preparation was merged (user, 00:40: "준비 다 되면 학습 해").
Main tree at the commit recorded in the run's manifest (after `9d8d35d`, the track-catalog merge).

## Ingredients, and where each came from

| ingredient | state | evidence |
|---|---|---|
| Recipe A (original race recipe, legacy controller, Adam kept, lr 5e-5 → 2e-5, KL 0.05) | unchanged | `recipe-restore-2026-09-12.md`, held-out table `benchmark-v2-first-2026-09-13.md` |
| Opponent behaviour events `brake,stop,shift,weave` at 1.0 / opponent / 10 s | new | `work/opponent-events/REPORT.md`, `docs/training.md` |
| Hard obstacle patterns `+hard` (gate / diagonal / chicane / apex / cluster / scatter, 55–75 % of the lane open, small objects) | new | `f1sim/hard_obstacles.py`, modeled on `~/f1sim_scenes/scene_0912_*` |
| Track registry + random obstacle seeds (`#hard:*`, drawn with `random.Random(701)`) | new | `docs/tracks.md` |
| Attitude model measured from the bags (roll 1.7 deg/g, squat/dive asymmetric, 1° floor wobble, IMU misalignment ±4°) | new, shifts observations | `docs/real_data_calibration.md` §6.1a |
| User scenes 2334 and 2344 (fwd + rev, clean and `#hard`) | new | scene 2355 held out |

Not in the run: the traction guard (real-car only until the simulator has wheel state), per-reset
obstacle re-placement (many seeds per map instead), the traffic benchmark (evaluation only).

## Track set

149 TRAIN tracks + 55 additions (see `tracks.txt` / `hard_specs.txt` in the run's work dir): 12 real
`#hard:*` ×2 draws fwd/rev, korea26 `#hard:*` ×3 draws in three directions, 16 generated `#hard:*`,
scenes 2334/2344 fwd/rev and one `#hard` draw each. Leakage guard against `HELDOUT_TRACKS`: clean.

## Command

See `work/learning-next/run-20260913/launch.sh`. Recipe A flags verbatim plus
`--opp-events brake,stop,shift,weave --opp-event-rate 1.0 --tracks "$(cat tracks.txt)"`, seed 701,
name `cl_hard_events_s701`, W&B group `consolidated-2026-09-13`.

## How it will be judged

Suite v2 (S/A/O) and, once merged, v2.1's traffic family, scored under the new attitude model
alongside re-scored references (`frozen_original@fixed_low`, `cl_origrecipe_legacy_s701@fixed_low`).
Rows scored before the attitude change are not comparable and are not pooled.
