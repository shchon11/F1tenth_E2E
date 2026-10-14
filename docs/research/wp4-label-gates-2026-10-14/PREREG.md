# WP4 교사 라벨: 경로 레시피 결정과 라벨 품질 관문 — 사전 등록 (2026-10-14)

> 이 문서는 측정을 돌리기 **전에** 커밋합니다. 판정 규칙, 조건, 셈법은 여기 적힌 대로이며, 결과를 본 뒤 바꾸지 않습니다. 결과는 같은 디렉터리의 `RESULTS.md`와 `results/*.json`에 적습니다.
> 근거 설계: `docs/research/output-format-2026-10-14.md` 5절(교사 라벨), 7절 시험 2~6, 8절.
> 코드: feat/v2 `ba43cda`(교사의 wp4 인코더, guard의 wp4 미리보기, 정지 후보 = a_brake 제동 곡선) 위에 이 문서와 `f1sim/scripts/wp4_label_gates.py`를 더한 커밋.

## 0. 공통 조건

- **교사.** 최소시간 레이싱 라인 교사 7 / 6.5 / 4 (`objective="min_time"`, a_lat 7.0, a_acc 6.5, a_brake 4.0). `label_grip="true"`, speed_scale 1.0. 설계 4절 재생과 같은 교사입니다.
- **env.** plan 모드, speed_cap 9.0 m/s, collision `soft`, compile 끔, `max_steps = steps + 10`, `laps = 10^6`(한 run이 한 에피소드), GPU 0 (RTX 4070). 형식은 `speed_mode` "wp4" 또는 "linear" (추종기 설정은 둘이 같음).
- **라벨.** 따로 적지 않으면 생산 경로 `env.teacher_label(teacher)`, 즉 env의 guard(오프셋 0, 속도 배율 1/0.8/0.55/0.3, 정지, 회피 4개, 설치된 추종기로 미리보기)를 거친 라벨입니다. linear는 여기에 인증(`project_plan_action`)이 붙은 생산 라벨입니다.
- **셈법** (`f1sim/scripts/wp4_label_replay.py`, ego 행 = `env.learner`만):
  - 벽/prop 접촉 = `info["wall_onset"]`의 시작 건수.
  - 차 접촉 = `info["car_collision"]`의 행별 상승 에지. **에피소드 첫 2 s(80 스텝) 안의 것은 세지 않습니다**(스폰 접촉 버그를 main에서 따로 고치는 중). 따로 기록만 합니다.
  - km = ego 행의 Σ max(v, 0)·dt.
  - 라인 밖 정지 비율 = 첫 2 s 이후 ego 스텝 중 |v| < 0.3 m/s 이고 레이싱 라인(교사의 `project`, 가장 가까운 라인 점까지 거리) > 0.15 m 인 비율.
  - κ1 라벨 점프 비율 = 첫 2 s 이후 연속한 두 스텝의 ego 라벨에서 |Δa[0]| > 0.1 인 비율 (linear는 같은 자리, 첫 곡률 knot).
  - 페이스 = ego 평균 속도(km 가중), 보조로 완주 랩 중앙값.
  - 건수 N, 거리 km 의 Poisson 1σ = sqrt(max(N, 1)) / km.
- 실행이 중간에 죽으면 같은 설정 그대로 다시 돌립니다. 그 밖의 재실행, 설정 변경, 시드 추가는 하지 않습니다.
- 등록 전 확인: 두 스크립트의 코드 경로를 icra2022 짧은 실행으로만 시험했습니다(noise-bias 48 상태 × 2 표본: 편향 −0.046, 조향 비 0.92; 재생 스크립트는 60 스텝 이하). ICCAS 관문 수치는 아직 보지 않았습니다.

## A. 경로 레시피 결정 (`RacelineTeacher.wp4_path`)

후보: `refit`(교사의 1.5 s linear 계획의 첫 창을 4점으로 재적합), `direct`(창 위 라인 점에 4곡률을 직접 적합, ridge 0.3), `hybrid`(라인 위는 direct, 오프셋 라인은 refit, |오프셋| 0~10 cm에서 섞음).

- **A1 단독.** 트랙 `real:korea_2025_iccas`, `real:icra2022`, `real:blackbox2021_1` × 시드 0, 1 × 레시피 refit, direct. 1024 env × 1200 스텝(30 s). guard 라벨.
  - guard는 오프셋 0만 넘기므로 hybrid의 라벨은 direct와 비트 단위로 같습니다(`tests/test_wp4_teachers.py`). A1의 hybrid 값은 direct 값을 그대로 씁니다.
- **A2 교통.** ego = `InteractiveTeacher`(DAgger `--teacher interactive` 기본값: 오프셋 ±0.6/±0.3/0, 속도 1/0.8/0.55/0.3, 후보 GN 2회, 1 s), 상대 = 레이싱 라인 교사(`opponent="teacher"`, 속도 배율 0.7~0.9), race_size 2. 트랙 ICCAS, icra2022 × 시드 0, 1 × 레시피 refit, direct, hybrid. 512 행(256 레이스) × 1200 스텝.
- **지표.** 레시피마다 접촉 = A1 벽 + A2 ego(벽 + 2 s 이후 차), 거리 = A1 km + A2 ego km. 율 r = 접촉/km. 페이스 = A1과 A2를 합친 km 가중 평균 속도.
- **판정.** 페이스가 가장 빠른 레시피의 98 % 이상인 레시피 가운데 r이 가장 낮은 것을 고릅니다. 두 레시피의 r 차이가 그 차이의 1σ(sqrt(N1/km1² + N2/km2²)) 안이면 동률로 보고, 동률이면 A2 율이 낮은 쪽, 그것도 1σ 안이면 hybrid > direct > refit 순(설계가 제안한 순서)으로 고릅니다.
- 고른 레시피를 `wp4_path` 기본값으로 커밋한 뒤 B를 돌립니다.

## B. 라벨 품질 관문 (설계 7절, 고른 레시피로)

**시험 2 — 잡음 편향.** `wp4_label_gates.py noise-bias`: ICCAS, 시드 0, wp4 guard 재생 8 env × 1200 스텝, 50 스텝부터 20 스텝마다 → 464 상태. 상태마다 추종기 입력을 `F1VecEnv.step`이 넘기는 그대로 캡처(재현 오차 0.0 확인). 잡음 16 표본, 차원별 σ = exp(log_std), log_std = (−2.07, −2.07, −1.8 × 6).
- 통과: (a) 실행 v_cmd 평균 편향(역방향 패스 있음 − 없음) ≥ −0.10 m/s, (b) 실행 첫 조향의 rms 잡음(라벨 대비)이 linear(같은 상태·입력, linear 교사 라벨, σ = exp(−1.8) 8차원)의 1.1배 이하.

**시험 3 — 교통 라벨.** `wp4_label_replay.py --opponent K --opp-speed 0.7 1.0 --envs 256 --steps 1200`, K ∈ {lane_switch, forzaeth_pred, interactive} × 시드 0, 1 × 형식 wp4, linear. ICCAS. guard 라벨(네이티브 guard, wp4는 wp4 추종기로 채점).
- 통과 (K마다, 두 시드 합산): (a) wp4 차 접촉/km ≤ linear 차 접촉/km + linear의 Poisson 1σ, (b) 라인 밖 정지 비율 ≤ 2 %, (c) κ1 점프 비율 ≤ 1.0 %.

**시험 4 — 장애물과 벽.**
- (a) prop: `--props --ego layout --envs 64 --steps 1200`, ICCAS, 절차형 prop(밀도 1.5, 최대 10, 라인 위 허용, 움직이는 prop, spawn_runway 3 m), 시드 0, 1, wp4. 통과: prop 접촉 0건(벽 거리장으로는 비어 있는데 시작된 접촉을 prop으로 셈). 벽 접촉도 함께 적습니다.
- (b) 벽: icra2022 단독, 시드 0, 1, 1024 env × 1200 스텝. wp4 = A1의 해당 run(고른 레시피; hybrid면 direct run). 비교 = linear, `speed_horizon_s` 0.6, 인증 끔(설계 4절 표의 "linear, 끝 0.6 s, 인증 끔"). 통과: wp4 벽 접촉/km ≤ 비교 쪽, 그리고 wp4 평균 속도 ≥ 비교 쪽의 0.98배. 접촉이 적어도 2 % 넘게 느리면 "판정 불가(페이스)"로 적습니다.

**시험 5 — 저속 권한.** `wp4_label_gates.py authority`: ICCAS, 시드 0, wp4 guard 재생 1024 env, 측정 속도 1~2 m/s인 상태(스폰 0~3 m/s에서 나옴) 중 512개 무작위. 차원마다 ±0.05 섭동.
- 통과: 8차원 모두 창 끝(추종기 예측 마지막 상태) 횡위치 ≥ 0.1 m/unit 또는 창 끝 속도 ≥ 0.1 m/s/unit. 실행 조향 °/unit, v_cmd m/s/unit도 함께 적습니다.

**시험 6 — 잡음 재생.** `wp4_label_replay.py --noise --envs 1024 --steps 1200`, ICCAS, icra2022 × 시드 0, 1 × 형식 wp4, linear(생산). 실행 행동 = clamp(라벨 + σ⊙ε), ε는 스텝마다 독립, σ는 시험 2와 같음(linear는 exp(−1.8) 8차원).
- 통과: wp4 접촉/km(두 트랙·두 시드 합산) ≤ linear의 같은 값. 페이스도 함께 적습니다.

## C. 범위 밖

- 시험 1(코드 정합)은 UNIT 1에서 끝났고, 7~9(오프라인 BC, Orin 지연, 학습 후 관문)는 이후 단위입니다.
- 관문이 실패해도 이 단위 안에서 설정을 바꿔 다시 맞추지 않습니다. 실패와 그 측정값을 그대로 보고합니다.
