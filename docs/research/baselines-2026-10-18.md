# External baselines vs our E2E RL policy (2026-10-18)

The user (2026-10-17): compare against published stacks, not our own checkpoints, and run the E2E baselines both with
their published weights and trained in this environment. Every row is the same simulator (fixed contact physics, main
633d3c3), the same race protocol and the same solo protocol. "Hard" = an impact at >= 1.5 m/s normal speed.

- **Race**: 4 opponent planners (ForzaETH spliner, predictive spliner, lane switch, interactive) x seeds 81-84, race of
  2, props on the line, 32 envs x 2400 steps (`tools/baselines/race_system.sh`; ICRA baselines 2 seeds). The harness
  reproduces our earlier numbers exactly (Korea pick, seeds 81-82: 3.94 m/s, 1.26 / 0.69 /km).
- **Solo**: 128 trials, a trial ends at the first contact (`tools/eval_system.py`).
- **Modular** (ForzaETH spliner, predictive spliner, lane switch as the EGO via `evaluate --teacher --teacher-kind`, main
  7a46cf0; raceline-only): ground-truth pose and opponent state; own path follower; line limits tuned 7..10 m/s^2 for the
  fastest setting with >= 124/128 solo (ICCAS 10/8/7, Korea 7/6.5/4; ICRA plan mode), opponents kept on 7/6.5/4
  (`--ego-teacher-limits`, d6e891e). The spliner is our reimplementation of the paper's planner.
- **Reactive**: ForzaETH's Follow-the-Gap ported line for line (parity test against the vendored upstream file) and the
  Disparity Extender from its authors' description (214aa7b); top speed tuned per map.
- **E2E**: TinyLidarNet / End2Race published weights (zero-shot) and the same architectures trained here with their own
  loss and optimiser on our interactive teacher (7/6.5/4 min-time line), the map + 16 warps, props on the line and the
  boot camp's planner opponents (`python -m f1sim.learn.baselines distill ... --env-extra`, 5a44d83 / 245ddf8), 8 DAgger
  iterations.
- Harness and raw results: `~/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/baselines/` (race/, out/,
  report_data.json, baselines_report.html). Survey of public stacks: SURVEY_2026-10-17.md there.


## Korea 2026

| driver | setting | solo lap | solo done | race m/s | hard walls+props /km | hard cars /km | passes /km | race km |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 우리 정책 (E2E RL) | korea26_fixphys_s1201_u384 | 9.92 s | 127/128 | 3.98 | 1.27 | 0.67 | 2.60 | 61 |
| ForzaETH spliner (추월) | 주행선 7/6.5/4, 자체 추종기 | 10.00 s | 124/128 | 2.66 | 10.70 | 1.76 | 1.78 | 41 |
| ForzaETH predictive | 같은 설정 | = raceline |  | 2.54 | 14.81 | 2.13 | 1.46 | 39 |
| Lane switch (UNICORN 계열) | 같은 설정 | = raceline |  | 3.04 | 11.21 | 0.83 | 2.50 | 47 |
| 주행선 추종만 | 장애물·상대 차를 못 봄 | 10.00 s | 124/128 | 2.67 | 11.69 | 7.48 | 0.37 | 41 |
| Follow-the-Gap (ForzaETH) | max_speed 8 | 13.83 s | 126/128 | 3.05 | 1.47 | 0.49 | 2.73 | 47 |
| Disparity Extender | top_speed 6 | 13.57 s | 103/128 | 2.82 | 1.34 | 0.69 | 2.67 | 43 |
| TinyLidarNet · 우리 환경 학습 | 같은 교사·장애물·상대 | 11.67 s | 123/128 | 3.09 | 1.96 | 3.66 | 1.41 | 48 |
| TinyLidarNet · 공개 가중치 | zero-shot | 12.83 s | 88/128 | 2.32 | 9.61 | 1.10 | 1.97 | 36 |
| End2Race · 우리 환경 학습 | 같은 교사·장애물·상대 | 11.87 s | 94/128 | 3.06 | 12.97 | 1.51 | 1.85 | 47 |
| End2Race · 공개 가중치 | zero-shot | - | 0/128 | - | - | - | - | - |

## ICCAS 2025

| driver | setting | solo lap | solo done | race m/s | hard walls+props /km | hard cars /km | passes /km | race km |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 우리 정책 (E2E RL) | iccas_fixphys_s1102_u384 | 8.46 s | 128/128 | 4.22 | 0.88 | 0.82 | 3.78 | 65 |
| ForzaETH spliner (추월) | 빠른 주행선 10/8/7, 자체 추종기 | 8.22 s | 127/128 | 2.63 | 10.00 | 4.21 | 2.03 | 40 |
| ForzaETH predictive | 같은 설정 | = raceline |  | 2.52 | 12.65 | 4.88 | 1.91 | 39 |
| Lane switch (UNICORN 계열) | 같은 설정 | = raceline |  | 2.88 | 17.72 | 2.37 | 1.83 | 44 |
| 주행선 추종만 | 장애물·상대 차를 못 봄 | 8.22 s | 127/128 | 2.94 | 9.55 | 7.29 | 1.88 | 45 |
| Follow-the-Gap (ForzaETH) | max_speed 6 | 15.55 s | 92/128 | 2.33 | 1.45 | 2.09 | 4.13 | 36 |
| Disparity Extender | top_speed 4 | 13.28 s | 98/128 | 2.51 | 2.75 | 1.53 | 3.74 | 38 |
| TinyLidarNet · 우리 환경 학습 | 같은 교사·장애물·상대 | 10.01 s | 125/128 | 2.85 | 3.58 | 2.94 | 2.46 | 44 |
| TinyLidarNet · 공개 가중치 | zero-shot | 9.06 s | 80/128 | 2.73 | 11.01 | 5.40 | 3.30 | 42 |
| End2Race · 우리 환경 학습 | 같은 교사·장애물·상대 | 11.35 s | 110/128 | 2.50 | 5.48 | 1.77 | 3.17 | 38 |
| End2Race · 공개 가중치 | zero-shot | 10.21 s | 2/128 | - | - | - | - | - |

## ICRA 2022

| driver | setting | solo lap | solo done | race m/s | hard walls+props /km | hard cars /km | passes /km | race km |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 우리 정책 (E2E RL) | 일반 정책 → 혼자 달리기 → 경주 384 | 15.31 s | 126/128 | 3.61 | 1.35 | 2.90 | 2.18 | 56 |
| ForzaETH spliner (추월) | 주행선 7/6.5/4, 추종 제어기 | 19.55 s | 110/128 | 2.33 | 4.41 | 0.73 | 1.73 | 18 |
| Lane switch (UNICORN 계열) | 같은 설정 | = raceline |  | 2.62 | 3.33 | 0.69 | 1.74 | 20 |
| 주행선 추종만 | 장애물·상대 차를 못 봄 | 19.55 s | 110/128 | 2.01 | 0.26 | 0.13 | 1.55 | 15 |
| Follow-the-Gap (ForzaETH) | max_speed 6~12 | - | 0/128 | - | - | - | - | - |
| Disparity Extender | top_speed 4~8 | 18.56 s | 3/128 | - | - | - | - | - |
| TinyLidarNet · 우리 환경 학습 | 같은 교사·장애물·상대 | 21.64 s | 22/128 | 1.94 | 0.33 | 0.74 | 0.74 | 15 |
| TinyLidarNet · 공개 가중치 | zero-shot | - | 0/128 | - | - | - | - | - |
| End2Race · 우리 환경 학습 | 같은 교사·장애물·상대 | - | 0/128 | 1.43 | 3.64 | 0.27 | 1.00 | 11 |
| End2Race · 공개 가중치 | zero-shot | - | 0/128 | - | - | - | - | - |

## Reading

- In races (props + an opponent) our policy is the fastest on all three maps (Korea 3.98 m/s vs the best baseline 3.09;
  ICCAS 4.22 vs 2.94; ICRA 3.61 vs 2.62) and has the fewest hard wall/prop hits on Korea and ICCAS. On ICRA the much
  slower line follower (2.0 m/s) hits less.
- Solo time trial: on ICCAS a tuned line follower with ground-truth pose is faster (8.22 s vs 8.46 s); on Korea it is
  equal (10.00 s, 124/128 vs 9.92 s, 127/128); on ICRA it fails (19.5 s, 110/128 vs 15.3 s, 126/128).
- Car contact is our weak axis: Follow-the-Gap on Korea (0.49 vs 0.67 /km) and the planners on ICRA (0.7 vs 2.9) hit the
  other car less, at far lower speed.
- The E2E baselines trained here on the same teacher and world stay well behind (imitation cannot pass its teacher;
  ours adds RL), and published weights barely drive this car.

