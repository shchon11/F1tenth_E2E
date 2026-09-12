# Documentation

Entry point: the [repository README](../README.md).

## Guides

| page | contents |
| --- | --- |
| [Getting started](getting_started.md) | installation, CPU and GPU paths, the ROS 2 workspace build, troubleshooting |
| [Architecture](architecture.md) | timing, vehicle, sensors, what is randomised, the map catalogue, raceline and teacher, plan tracking, the two viewers |
| [Training](training.md) | DAgger, PPO, opponent behaviour events, evaluation, the console's 학습 page, the controller arms, observation and action detail |
| [Tracks](tracks.md) | the track ids, the scenario grammar (direction / obstacle / seed), what 무작위 시드 does, the catalogue and the three splits |
| [Environment editor](environment_editor.md) | the console's 환경 page: paths and brushes, props, mesh import, the random track generator, the `scene:` prefix, validation |
| [Benchmark](benchmark.md) | the checkpoint benchmark CLI — suites v1 and v2, freezing, roster pinning, the independence gate, scoring, metric definitions, adding a checkpoint later |
| [Leaderboard](leaderboard/README.md) | the rendered v1 leaderboard: six metrics per cohort with exact counts, paired pace, evidence links and limitations. `leaderboard/index.html` is the same report as a self-contained offline page |
| [ROS 2](ros2.md) | the console on ROS 2 (topics out, `/drive` in, rviz), the `f1tenth_system` stack on the simulator, the standalone bridge, teleoperation, the policy node and its plan controller |
| [Viewer design](viewer_design.md) | the console's visual language: references, tokens, the scene, the instruments, the map card |
| [로컬 실행 안내](local_pc.md) | 이 워크스테이션 전용 실행 명령 (Korean) |

## Research notes

Dated experiment reports: protocol, data, and an explicit statement of what each does not establish.
Newest first. Raw per-cell rows, rosters and manifests sit beside each note in
[`research/`](research/).

| page | contents |
| --- | --- |
| [Consolidated run `cl_hard_events_s701`](research/run-recipe-2026-09-13.md) | 2026-09-13: what went into the single training run after the preparation was merged — recipe A plus opponent events, hard obstacles, the new attitude model, the user's scenes — and how it will be judged. No result yet |
| [First held-out scoring, suite v2](research/benchmark-v2-first-2026-09-13.md) | four systems on the frozen held-out suite — 64 cells, 512 trials each; the A recipe generalises at two seeds, plus an obstacle-seed memorisation probe |
| [Controller arms on suite v1](research/controller-arms-2026-09-12.md) | the frozen policy under `legacy` / `estimated` / `fixed_low` / `reactive`: `fixed_low` becomes the deployment default, `reactive` is not adoptable |
| [Recipe restore](research/recipe-restore-2026-09-12.md) | three cheap tests separating map narrowing, optimizer shock and controller-in-the-loop: train under `legacy`, deploy clamped. Recipe A's full argv |
| [A702 replication](research/a702-replication-2026-09-12.md) | a second seed of recipe A, trained and scored against six predeclared retention gates — all pass ([leaderboard](research/a702-replication-2026-09-12-leaderboard.md)) |
| [Opponent diversity, stage 1](research/opponent-diversity-stage1-2026-09-12.md) | one paired pilot widening opponent speed and spawn gap together: three predeclared gates fail, nothing promoted ([leaderboard](research/opponent-diversity-stage1-2026-09-12-leaderboard.md)) |
| [Static-grip retention](research/static-grip-retention-2026-09-12.md) | four training recipes × two seeds against a frozen reference — 30 cells, 240 trials per checkpoint; no eligible recipe, with per-trial data and an independent eligibility audit |
| [Checkpoint benchmark v1, three-system subset](research/benchmark-v1-2026-09-12.md) | three systems on the frozen v1 suite — 102 cells, 816 trials, with raw and derived rows; **not** the 17-system benchmark |
| [Checkpoint benchmark v1, R10](research/benchmark-v1-r10-2026-09-12.md) | the two R10 directional-coverage checkpoints on the same v1 suite: no improvement, and a low-friction regression |
| [Controller-arm evaluation](research/2026-09-11-controller-arm-evaluation.md) | matched-policy `fixed_low` vs `estimated` — 72 cells, 1152 trials, with outcome-only data beside it: no robust method-level completion gain |
| [Legacy-recipe probe](research/2026-09-11-legacy-recipe-probe.md) | one-seed control asking whether the fine-tuning recipe alone explains the low-friction drop. It does not settle it either way |
| [A wheel that can slip](research/wheel-model-2026-09-13.md) | the rear-axle rotation state, the ERPM channel's artefacts and the in-simulator traction arm, scored against the 22 real recordings by the same script that scores the car |

## Engineering notes

Working documents rather than guides: they record how parameters were chosen and where the design is
known to be weak.

| page | contents |
| --- | --- |
| [Real-data calibration](real_data_calibration.md) | what the 22 rosbags fixed and what is still unmeasured, including the 2026-09-13 attitude measurement (§6.1a) and the open list (§6.2) (Korean) |
| [Simulator audit](simulator_audit.md) | fidelity review against recorded vehicle data |
| [Algorithm assessment](algorithm_assessment.md) | critique of the current architecture and the comparisons planned against it |

## Media

[`media/`](media/) holds the figures the README shows, each with the record of how it was made:

| file | provenance |
| --- | --- |
| `f1tenth-simulator-demo.{mp4,gif,-poster.png,-contact-sheet.png}`, `f1tenth-learning-demo.{…}` | [PROVENANCE-video.md](media/PROVENANCE-video.md), [video-manifest.json](media/video-manifest.json), [video-continuity-check.json](media/video-continuity-check.json) |
| `f1tenth-visualizer-{overview,chase,policy-panel,props}.png` | [PROVENANCE-visualizer.md](media/PROVENANCE-visualizer.md), [visualizer-manifest-2026-09-12.json](media/visualizer-manifest-2026-09-12.json), [visualizer-props-manifest-2026-09-12.json](media/visualizer-props-manifest-2026-09-12.json) |
| `f1tenth-environment-editor.png`, `f1tenth-environment-editor-top.png` | [environment-editor-manifest-2026-09-12.json](media/environment-editor-manifest-2026-09-12.json) |
| `f1tenth-trackgen.png`, `f1tenth-trackgen-gallery.png` | [trackgen-manifest-2026-09-13.json](media/trackgen-manifest-2026-09-13.json) |

## Figures

Generated by scripts under [`../f1sim/scripts/`](../f1sim/scripts/) and checked in for reference:
track galleries (`tracks_real.png`, `tracks_rt.png`, `tracks_gen_competition.png`,
`tracks_gen_other.png`, `map_grid.png`, `map_zoo.png`), racelines (`racelines.png`,
`raceline_1.png`), real venues (`real_venues.png`, `real_maps_candidates.png`,
`real_maps_centerlines.png`, `korea_team_maps.png`), multi-car LiDAR (`multicar_lidar.png`), an IMU
trace (`imu_trace.png`), a single lap (`track_1_lap.png`), and the older viewer captures
(`viewer_overview.png`, `viewer_top.png`, `viewer_chase.png`, `viewer_closeup.png`,
`watch_ppo_v2.png`). `tracks_info.json` holds the catalogue metadata behind the galleries.

`promo.gif` / `promo.mp4` are the earlier rendered promo, regenerable with
`../f1sim/scripts/promo_video.py`. The current demonstration clips are the two in `media/` above.
