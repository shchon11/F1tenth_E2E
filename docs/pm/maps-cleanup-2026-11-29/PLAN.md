# 지도(코스) 전수 조사와 삭제 계획 (2026-11-29)

이번 작업에서는 아무것도 지우거나 고치지 않았다. 읽기와 헤더 확인뿐이다. 지울 때의 영향은 grep과 파일 크기로 쟀다.

## 0. 결론

| 구분 | 개수 | 내용 |
|---|---|---|
| (가) 남김 | 3 | `real/icra22`(ICRA 2022), `real/iccas25`(ICCAS 2025), `real/korea26-scan`(IFAC 2026, 현재 정본) |
| (다) 판단 필요 | 12 | `real/korea26`(같은 대회 중복 판), `real/korea26x12`(×1.2), `real/lab16x07`, `real/lab12x16`, `real/bb21-1..3`, `real/bb22-1..3`, `rt/spielberg`, `rt/oschersleben` |
| (나) 지움 | 레이스트랙 21 + gym 5 + 생성기 전체(`gen/` 5 스타일, 무한 시드) | 실제 F1 서킷 축소판, gym 그림 지도, 절차 생성 지도 |

- **PNU championship 지도는 저장소에 없다.** 파일 이름, 문서, 커밋 어디에도 PNU·부산대가 없다. 웹에는 PNU-HYUNSONG CHAMPIONSHIP(2025-01-22~23, 부산대 기계관 국제회의장) 기사만 있고 트랙 지도는 공개된 곳을 못 찾았다.
- **정본이 저장소 밖에 있다.** IFAC 2026 정본 `korea26_scan`은 `~/.f1sim/maps/korea26_scan/`(112 KB, git 추적 안 됨, 복구 불가)에 있다. 새 시뮬 `library/courses/kr26-hall`, 학습 설정 8개, `compbench` 시험이 모두 이 지도를 쓴다. 무엇을 지우기 전에 저장소 안으로 복사해야 한다.
- **(가) 3개 중 2개는 저장소 밖 서브모듈 안에 있다.** ICRA 2022는 `external/f1tenth-racing-stack-ICRA22`, ICCAS 2025는 `external/korea_teams/KORA_K3`다. 서브모듈을 지우면 이 둘도 같이 사라지므로 파일 3개를 먼저 `f1sim/f1sim/assets/maps/`로 옮겨야 한다.

## 1. 전수 목록

"추적"은 git 상태다. `external/*`는 모두 서브모듈이고 `.gitmodules`에 `ignore = untracked`가 붙어 있다. 내용은 메인 저장소 커밋으로 되돌릴 수 없고, GitHub에서 다시 받아야 한다.

### 1-1. 실측 지도 (등록 id)

| 등록 id | 표시 이름 | 옛 이름(로더) | 파일 | 크기 · 치수 | 추적 |
|---|---|---|---|---|---|
| `real/icra22` | ICRA 2022 | `icra2022` | `external/f1tenth-racing-stack-ICRA22/maps/icra_2.{yaml,png}` | 446×348 px, 0.05 m/px, 22.3×17.4 m, png 10 KB | 서브모듈(저장소 밖) |
| `real/bb21-1,2,3` | Blackbox 2021 #1~3 | `blackbox2021_1..3` | `external/f1tenth_maps/maps/blackbox2021_N.{yaml,png}` | 2000×2000 px, 0.05 m/px(칸 100 m), png 32~63 KB | 서브모듈 |
| `real/bb22-1,2,3` | Blackbox 2022 #1~3 | `blackbox2022_1..3` | `external/f1tenth_maps/maps/blackbox2022_N.{yaml,png}` | 같음, png 44~67 KB | 서브모듈 |
| `real/korea26` | Korea 2026 | `korea_2026_competition` | `f1sim/f1sim/assets/maps/korea_2026_competition.{yaml,pgm}` | 75 KB, 0.05 m/px, 8.4×22.6 m(대회 bag의 /map) | 추적됨 |
| `real/korea26x12` | Korea 2026 ×1.2 | `korea_2026_x1p2` | `assets/maps/korea_2026_x1p2.yaml`(145 B, 위의 pgm을 0.06 m/px로 읽음) | 파일 새로 없음 | 추적됨 |
| `real/lab16x07` | 실측 16×7 m | `map16x07` | `assets/maps/map16x07.{yaml,pgm}` | 43 KB, 15.5×7.0 m | 추적됨 |
| `real/lab12x16` | 실측 12×16 m | `map12x16` | `assets/maps/map12x16.{yaml,pgm}` | 80 KB, 12.3×16.4 m | 추적됨 |
| `real/iccas25` | ICCAS 2025 | `korea_2025_iccas` | `external/korea_teams/KORA_K3/src/kora_k3/maps/real.{yaml,pgm}` | 484×495 px, 0.05 m/px, 24.2×24.8 m, 234 KB | 서브모듈(저장소 밖) |
| `real/korea26-scan` | Korea 2026 (bag 스캔 지도) | `korea26_scan` | `~/.f1sim/maps/korea26_scan/{.yaml,.pgm,map.json}` | 95 KB | **추적 안 됨, 저장소 밖** |

`map17x18`은 bag 이름에만 있고 등록된 지도 파일이 없다(`profiles/pre_car_map17x18.yaml`만 있음).

### 1-2. 생성기·레이스트랙·gym·에디터 계열

| 계열 | 등록 | 실제로 쓰이는 id | 파일 |
|---|---|---|---|
| `rt/*` | 23개(`_RT_KNOWN`) | 학습 `rt/spielberg`, `rt/oschersleben`(각 4방향). 검증 `rt/monza` | `external/f1tenth_racetracks/<이름>/` 23폴더, 합계 7.0 MB |
| `gym/*` | 5개 (berlin, levine, skirk, stata, vegas) | 분할 규칙에 없음 | `external/f1tenth_gym/gym/f110_gym/envs/maps/` 4.2 MB |
| `gen/*` | 스타일 5개(comp, control, circuit, hall, serp) × 임의 시드 | 학습 `control-1400..1419`, `comp-1000..1015`, `circuit-1200..1203`, `hall-1100..1103`. 검증 `comp-0`, `control-9100`, `control-9102`, `comp-9200` | 파일 없음. `Track.generate_random`(`track.py:1049`)이 그때그때 만든다. 맛보기 시드 4개 PNG는 `f1sim_ros/maps/random_0..3` |
| `scene/*` | 에디터 장면 | 없음(`~/f1sim_scenes` 비어 있음) | 해당 없음 |

### 1-3. 변형 접미사: 파일이 아니라 그때그때 계산한다

`+hose<k>`, `+duct`, `+smooth`, `+warp<k>`, `+scale<p>`, `~rev`, `~mir`, `+bare`, `+obs/+rlobs/+pinch/+props/+hard<seed>`는 모두 `maps._load_base`(`maps.py` 290~330행)가 기반 지도를 불러온 뒤 메모리에서 바꾼다. 디스크에 저장된 변형 지도는 없다. 그래서 변형 접미사는 "기반 지도가 남는 한" 계속 동작한다. 예외는 `korea26x12`다. 이것은 접미사가 아니라 `assets/maps/korea_2026_x1p2.yaml` 파일이다.

### 1-4. 등록 없이 디스크에만 있는 지도 (정보용)

| 위치 | 내용 | 크기 |
|---|---|---|
| `external/f1tenth_maps/maps/` | berlin, levine, skirk, vegas, torino*, f1_aut/esp/gbr/mco(실제 서킷), columbia, porto, mtl, train_* 등 약 40종과 `Oschersleben_map` | 15 MB(+원본 svg 4.4 MB) |
| `external/f1tenth-racing-stack-ICRA22/maps/` | `icra.yaml`(74×35 m, 다른 판), levine*, skir, `Spielberg_map` | 32 MB 중 일부 |
| `external/korea_teams/KORA_K3/src/kora_k3/maps/` | f3~f11, f_1, f_2, test1, 6th_floor(팀 연습 SLAM 지도), `f5.posegraph` 18 MB | 6.4 MB(저장소 전체 135 MB) |
| `external/korea_teams/project_The_4th_F1TENTH_Korea_championship/maps/` | 0625~0924(날짜 이름 SLAM 지도 18개, 0.03 m/px, 15~99 m 변) | 31 MB(저장소 전체 153 MB) |
| `external/korea_teams/F1tenthKorea2024`, `f1tenth_tada` | 웹사이트와 일반 코드(지도 아님. `F1tenthKorea2024`는 제주 호텔 안내 그림만 있음) | 175 MB, 0.1 MB |
| `external/f1tenth_gym/.../maps/` | gym 5종(위 `gym/*`와 같음) | 4.2 MB |

## 2. 출처 판정

| 지도 | 판정 | 근거 |
|---|---|---|
| `real/icra22` | (가) | 지도가 들어 있는 저장소 `zzjun725/f1tenth-racing-stack-ICRA22`의 README: "ICRA'22 F1tenth Competition"에 낸 코드. 지도는 팀이 대회 트랙을 SLAM으로 만든 것(`maps.py:45`의 설명도 같음). 같은 폴더의 `icra.yaml`(74×35 m)은 다른 판이고 등록되지 않았다. 등록된 `icra_2`(22×17 m)가 대회 크기인지는 이번에 재지 못했다. **(가)로 남기되 크기 검증 필요.** |
| `real/korea26-scan` | (가) 정본 | 2026-08-27 대회장(`real_data/01_competition_0826-0827_map08x23`)의 가장 빠른 bag `/scan+/odom`으로 만든 지도. `map.json`에 bag의 /map과 차선 IoU 0.95라고 기록됨. 책임자 이름 규칙으로 "IFAC 2026". 현재 학습·시험·새 시뮬이 쓰는 지도. |
| `real/korea26` | (다) 중복 판 | 같은 대회장의 bag `/map`을 그대로 뽑은 것. 정본은 `korea26-scan`. 옛 체크포인트(`checkpoints/korea26_*.pt` 8개), `plan_world_korea_2026_*.json` 3개, 상자 은행 `~/f1sim_runs/prop_banks_*_korea_2026_competition`, 문서 146개가 이 이름을 쓴다. |
| `real/korea26x12` | (다) 임의 축척 | 같은 지도를 1.2배로 읽은 것(`maps.py:64` 주석에 11-01 지시로 만든 것이라고 적힘). 실제 대회 지도 아님. 참조: 코드 3파일, 테스트 1파일. |
| `real/iccas25` | (가) | `maps.py:68`: "4th F1TENTH Korea Championship 2025 (ICCAS) race track, team SLAM map". 파일은 KORA_K3 팀의 `real.yaml`. 웹: 2025 ICCAS F1TENTH Autonomous Racing Competition은 송도 컨벤시아(인천)에서 열렸다. 저장소 `project_The_4th_F1TENTH_Korea_championship`가 있어 4회 한국 대회와 ICCAS의 연결은 이 프로젝트의 기존 판단이다. KORA_K3 파일이 정확히 대회 트랙인지는 문서 근거가 팀 SLAM 지도라는 것뿐이다. 참고: 팀 SLAM 지도는 20~25 % 작게 나온 사례가 있었으나 KORA 지도는 올바른 크기로 기록되어 있다(메모리 11-02). |
| `real/lab16x07`, `real/lab12x16` | (다) 시험 장소 | 대회 전 시험 장소(`real_data/02_pre-competition`, 2026-07)에서 팀이 그린 지도. 대회 지도가 아님. 동시에 일반화 시험(검증 분할 `HELDOUT_BASE_IDS`)과 `compbench` dev·confirm 스위트, 새 시뮬 `library/courses`의 `lab-*`가 쓴다. |
| `real/bb21-1~3`, `bb22-1~3` | (다) | `external/f1tenth_maps`(CPS-TUWien) 소개: "to be used in f1tenth-style races". 커밋 메시지: "add blackbox maps of 2021 ARC class", "add 2022 blackbox maps". TU Wien(Scuderia Segfault)이 연 F1TENTH 방식의 BlackBox 경기 지도다. F1TENTH 공식 대회, RoboRacer Korea, PNU 중 어느 것도 아니다. 학습 분할 6개 중 5개(bb22-3은 검증)가 쓰고 있다. |
| `rt/spielberg`, `rt/oschersleben` | (다) 약한 후보 | `f1tenth_racetracks` 소개: "real race tracks (mainly F1 and DTM) downscaled on 1:10". 그 자체로는 (나). 다만 `f1tenth_maps` README가 `Oschersleben`을 "F1TENTH IROS 2022 Race Resources"로 적었고, ICRA22 팀 저장소에 `Spielberg_map.yaml`이 있다. 이 두 서킷이 F1TENTH 경기에 쓰였을 가능성은 있으나, 저장소 안의 `rt/*` 파일 자체가 그 대회 지도인지는 확인하지 못했다. 학습 분할에 4방향씩 들어 있다. |
| 나머지 `rt/*` 21개 | (나) | 실제 F1·DTM 서킷 축소판. |
| `gym/*` 5개 | (나) | `f1tenth_gym`이 같이 주는 시뮬 예제 지도(Levine=펜실베이니아대 건물, Stata=MIT 지하 등). 대회 지도라는 근거 없음. `vegas`는 출처를 확인하지 못했다. |
| `gen/*` | (나) | 절차 생성기 산출물. |

### 축척이 잘못된 판과 바른 판

- Korea 2025: 팀 SLAM 지도는 20~25 % 작게 나온 사례가 있고, 이 저장소는 바른 KORA 판을 쓴다(`bag_map.py`의 축척 검사는 이후에 만든 도구).
- Korea 2026: `korea26`(bag /map)과 `korea26-scan`은 IoU 0.95로 같은 모양이다. 축척 문제는 보고된 바 없다.
- ICRA 2022: `icra.yaml`과 `icra_2.yaml`이 크기가 다르다(74×35 m와 22×17 m). 어느 쪽이 대회 크기인지 이번에 확인하지 못했다.

## 3. PNU와 저장소에 없는 공개 지도 후보

- PNU: 저장소에 없다. 웹 확인 결과는 위 0절과 같다. 기사 주소만 적는다: https://www.asiae.co.kr/en/article/2025012510460666714 (지도 언급 없음).
- 후보(내려받지 않음):

| 후보 | 주소 | 비고 |
|---|---|---|
| IROS 2022 경기 자료(Oschersleben 지도) | `external/f1tenth_maps/maps/Oschersleben_map.yaml`(이미 디스크에 있음, 미등록) | README 출처: F1TENTH IROS 2022 Race Resources. 가장 확실한 추가 후보 |
| ICRA22 팀 저장소의 `Spielberg_map` | `external/f1tenth-racing-stack-ICRA22/maps/Spielberg_map.yaml`(이미 있음, 미등록) | 어느 대회 트랙인지 불명 |
| 4회 한국 대회 팀 저장소의 날짜 지도 0625~0924 | https://github.com/JihyunKim725/project_The_4th_F1TENTH_Korea_championship (이미 있음) | 한 팀의 연습·대회 중 SLAM 지도. 어느 것이 ICCAS 본선 트랙인지 알 수 없음 |
| ICCAS 2025 공식 안내(4회 한국 대회) | korea-race24f1tenth.org (검색에서 링크만 나옴, 내용은 못 읽음) | 지도 공개 여부 불명 |

## 4. 영향 조사

### 4-1. (나)를 지울 때 깨지는 것 (코드 · 테스트 · 설정)

참조 파일 수(`git grep`, 서브모듈과 `real_data` 제외, 문서는 별도):

| 지도 | 코드 | 테스트 | 비고 |
|---|---|---|---|
| `gen/*` 생성기 | 30개 이상 (`track.py`, `tracks.py`, `maps.py`, `learn/benchmark/*`, `learn/watch.py`, `learn/aligned_floor.py`, `teacher.py`, `minimum_time.py`, `viewer/console/env_editor.py`, `viewer/console/training.py`, `viewer/sim_worker.py` 등) | 96개 | `f1sim_ros` 기본 지도(`vesc_sim_node.py:78`, `launch/f1tenth_stack_sim.launch.py:23`)가 `gen:competition:0`, `bridge_node.py:80`도 `generate_random` |
| `rt/*` | 19개 (`tracks.py`, `maps.py`, `catalog.py`, `window.py`, `sim_worker.py`, `server/server.py`, `perception.py`, `teacher.py`, `learn/baselines/distill.py`, `learn/watch_gui.py`, `scripts/demo_viewer.py`, `scripts/track_gallery.py` 등) | 10개 | 같은 목록에 `rt/spielberg`, `rt/oschersleben`(학습), `rt/monza`(검증) |
| `gym/*` | 6개 | 2개 | `learn/baselines/end2race.py`는 주석만 |
| `bb21/bb22` | 17개 | 16개 | `learn/common.py`, `learn/ppo.py`, `learn/aligned_floor.py`, `learn/probe_hidden.py`, `viewer/console/protocol.py`, `scripts/encoder_bench.py` |
| `lab16x07`/`lab12x16` | 20여 개 | 13개 | `params.py`, `dynamics.py`, `bag_profile/measure.py`, `learn/state_rl.py`, `profiles/pre_car_map*.yaml`, `game/tests/fixtures/catalog.json:39,58`, `library/courses/lab-*.yaml` |
| `korea26x12` | 3개 (`maps.py:66`, `tracks.py:232`, `viewer/console/window.py`) | 1개 | 영향 작음 |
| `icra22` | 11개 | 12개 | (가)이므로 지우지 않음 |

핵심 코드는 `tracks.py`의 `_REAL`(214행), `_RT_KNOWN`(238행), `GEN_*`(251행), 분할 규칙 `REAL_TRAIN_IDS`(829행), `RT_TRAIN_IDS`(831행), `GEN_TRAIN_IDS`(856행), `TRAIN_RULES`(872행), `HELDOUT_RULES`(902행), `HELDOUT_OBSTACLE_RULES`(916행)와 `maps.py`의 `REAL`(42행), `catalog()`다. 가장 많이 걸리는 테스트: `tests/tracks_oracle.py`(142곳), `test_tracks.py`(41), `test_heldout_split.py`(23), `test_bare_obstacles.py`(20), `test_console_training.py`(18), `test_console_map_card.py`(16), `test_map_registry.py`(13), `test_procedural_obstacles.py`, `test_multitrack.py`, `test_raceline_feasibility.py`, `test_training_setup.py`, `test_training_schema.py`, `test_track_modifiers.py`, `test_trackgen.py`, `test_scene.py`, `test_asset_obstacles.py`. 전체 테스트 259개 파일 중 약 100개가 지도 id를 직접 이름으로 쓴다. 학습 콘솔의 `training_schema.json`(8354행부터 지도 목록)과 `learn/benchmark/suite*.json` 3개(`suite-v1/v2/v2.1.example.json`)도 이 이름들을 박아 두었다.

### 4-2. 체크포인트 · 학습 실행

- `checkpoints/manifest.json`의 지도 언급은 `icra2022` 1건과 `map12x16` 1건뿐이다. 나머지 체크포인트는 `iccas_*` 11개와 `korea26_*` 8개로, 파일 이름에 지도가 있다.
- `map12x16_specialist_s903.pt`(lab12x16 전용), `icra2022_map3R2_s1321_u384.pt`(icra22 전용)는 (가)·(다) 쪽이라 지울 이유가 없다.
- 현재 선정본인 `scan31b`, `hs43b`, `race33b`, `rc42a`의 학습 설정은 `plan_world_korea26scan_*.json`이고 지도는 `real:korea26_scan+duct+smooth`(+hose 변형 16개)뿐이다. (나)를 지워도 이 실행들은 그대로 돌아간다.
- `plan_world_korea25.json`, `plan_world_korea26_*.json`은 `korea_2025_iccas`·`korea_2026_competition`을 쓴다. 이 둘은 (가)·(다) 중복 판이라 영향 없다.
- 옛 학습 설정이 `rt/*`, `gen/*`, `bb*`를 쓰는 곳은 `learn/benchmark/suite*.json`과 기본 `--tracks train` 분할(`TRAIN_RULES`)이다. 분할 규칙을 바꾸면 `--tracks train`의 의미가 바뀐다. 이미 학습된 체크포인트의 manifest에는 구체 이름이 기록되어 있어서 기록은 안 깨진다.

### 4-3. 진행 중인 평가 도구

- `f1sim/f1sim/compbench/suites/dev.json`은 `korea26_scan`(+hose23, +hose24), `map12x16`(+smooth, ~rev, +duct+hose3), `map16x07`(+smooth, ~rev)을 쓴다. `confirm.json`도 `korea26_scan`(+hose25), `map12x16`, `map16x07`을 쓴다. 그래서 **lab 두 지도를 (나)로 지우면 compbench dev·confirm이 깨진다.** (나) 기준인 `rt/gen/gym`은 compbench에서 쓰지 않는다.
- 상자 은행 `/data/compbench/banks/`에는 lab 두 지도용이 있다: `labconf_map12x16*`, `labdev_map12x16*`(각 36 KB), `grid_map12x16.npz` 5.0 MB, `grid_map16x07.npz` 2.7 MB, `*_refline.npz`, `empty_map*`. `exam3_k26scan`, `confirm_k26scan`, `tune_k26scan`은 IFAC 2026용이다.
- 지금 돌아가는 작업: `docs/pm/dodge-signal-2026-11-28/scripts/cap_exam.py`(hs43b u900, 학습 지도와 같은 korea26scan 계열로 보임), `train_queue.py`(`~/f1sim_runs/_train_status.json`에 살아 있는 실행은 없음), `dash_server.py`. 이 작업들이 (가) 지도만 쓰는지 실제 인자는 확인하지 못했다. 따라서 실행 중인 시험이 끝나기 전에는 `korea26_scan`, 상자 은행, `~/.f1sim/maps`를 건드리지 말 것.

### 4-4. 문서 · 그림

- 추적되는 지도 그림: `docs/tracks_gen_competition.png` 104 KB, `tracks_gen_other.png` 104 KB, `tracks_rt.png` 372 KB, `tracks_real.png` 184 KB, `korea_team_maps.png` 180 KB, `map_grid.png` 136 KB, `map_zoo.png` 160 KB, `docs/media/f1tenth-trackgen.png` 684 KB, `f1tenth-trackgen-gallery.png` 220 KB, `trackgen-manifest-2026-09-13.json`, `docs/tracks_info.json` 12 KB (합계 약 2.2 MB).
- `docs/tracks.md`는 카탈로그 표(146~240행)에 위 id를 전부 적었다. 갱신 필요. `README.md`, `docs/README.md`, `docs/ros2.md:870`(make_maps.py 안내), `docs/environment_editor.md`, `docs/local_pc.md`, `docs/training.md`, `docs/viewer_design.md`, `docs/benchmark.md:124`(검증 스위트 정의에 lab, iccas, bb22-3, Monza, gen)도 갱신.
- `docs/research`는 과거 연구 기록(참조 수백 건)이다. 고치지 말고 그대로 두는 것을 권한다.

## 5. 삭제 계획

### 5-1. 단계

| 단계 | 할 일 | 위험 | 지금 가능? |
|---|---|---|---|
| ⓪ 보존 | `~/.f1sim/maps/korea26_scan/` 3파일, KORA_K3 `real.{yaml,pgm}`, ICRA22 `icra_2.{yaml,png}`를 `f1sim/f1sim/assets/maps/`로 복사하고 `maps.py`·`tracks.py`의 경로를 저장소 안으로 바꾼다. (복사만, 원본 유지) | 없음 | **지금 가능** |
| ① 새 시뮬 노출 | `library/courses/`에 (가)만. 현재 4개 중 `kr26-hall`, `kr25-iccas`는 유지. `lab-12x16`, `lab-16x07`은 (다) 결정 전까지 숨긴다(파일 이동이 아니라 목록 제외). `icra22`용 `*.course.yaml`을 새로 추가해야 한다(없음). `game/tests/fixtures/catalog.json:39,58`와 `test_m1.py`(kr26/kr25만 단정함)는 확인 | lab 코스 상자 은행 `/data/compbench/banks/labconf_map12x16` 참조가 끊김 | **지금 가능**(새 시뮬은 compbench와 별개) |
| ② 등록 목록 정리 | (나) 제거: `tracks.py`의 `_RT_KNOWN`·`racetrack_dirs`·`gym_*`·`GEN_*`와 `FAMILIES`·`FAMILY_LABEL`, `get()`의 `rt/gym/gen` 분기, 분할 규칙 3종 재작성, `maps.py`의 `rt/gym/gen:` 분기(약 320~400행). 고칠 코드·테스트는 4-1의 목록 | 분할 의미 변경 | 평가 작업이 끝난 뒤 |
| ③ 디스크 삭제 | 아래 5-2 표 | 서브모듈은 GitHub에서 재취득 가능, 추적 안 되는 캐시·은행은 일부 복구 불가 | 일부 지금 가능, 아래 구분 |
| ④ 생성기 폐기 | 아래 5-3 | `env_editor`가 `trackgen`을 쓴다 | 새 생성기가 나온 뒤 |
| ⑤ 문서·그림 | `docs/tracks.md`, README 갱신, 그림 삭제 | 낮음 | ②와 함께 |

### 5-2. 디스크에서 지울 파일 (용량 계산, 지우지 않음)

| 대상 | 용량 | 되돌리기 | 지금 안전? |
|---|---|---|---|
| `external/f1tenth_racetracks` (서브모듈 전체) | 7.0 MB | `git submodule update --init`으로 재취득. 메인 저장소 커밋으로는 못 되돌림 | ⓪ 뒤 가능. 단 `rt/spielberg`·`oschersleben`이 (다) |
| `external/f1tenth_gym` | 9.3 MB | 서브모듈 재취득 | 가능(코드는 주석에서만 언급) |
| `external/f1tenth_maps` | 20 MB | 서브모듈 재취득 | (다) bb21/bb22 결정 뒤. `Oschersleben_map` 후보도 여기 있음 |
| `external/korea_teams` 4개 | 462 MB (KORA_K3 135, 4회 대회 팀 저장소 153, F1tenthKorea2024 175, tada 0.1) | 서브모듈 재취득(원격 저장소가 사라지면 복구 불가) | ⓪ 뒤 가능 |
| `external/f1tenth-racing-stack-ICRA22` | 32 MB | 서브모듈 재취득 | ⓪ 뒤 가능 |
| `assets/maps/` lab 두 지도 | 약 124 KB + yaml 2개 | git 커밋으로 되돌림 | (다) 결정 뒤 |
| `assets/maps/korea_2026_x1p2.yaml` | 145 B | git | (다) 결정 뒤 |
| `docs/tracks_*.png` 등 그림 11개 | 약 2.2 MB | git | 가능 |
| `f1sim_ros/maps/random_*` | 수십 KB | git | 가능 |
| `~/.cache/f1sim/racelines` (지도 이름이 (가) 아닌 것 946개) | 83.6 MB (전체 97 MB 중) | 되돌릴 수 없으나 지도에서 다시 계산됨 | 가능 |
| `~/.cache/f1sim/centerlines` (같은 기준 110개) | 4.3 MB | 같음, 자동 재생성 | 가능 |
| `~/.cache/f1sim/goal_field` 453 MB, `map_lut` 3.2 MB | 해시 이름이라 지도별 구분 불가 | 자동 재생성 | 지도 정리 뒤 통째로 비우는 것을 검토 |
| `/data/compbench/banks` lab 두 지도 은행(`labconf*`, `labdev*`, `grid_*`, `*_refline.npz`, `empty_map*`) | 약 8 MB | **복구 불가**(상자 배치는 도구로 다시 만들 수 있으나 같은 은행이 아님) | (다) 결정과 compbench 종료 뒤 |
| `~/f1sim_runs/prop_banks_*_korea_2026_competition` 4개 | 약 52 MB | 복구 불가 | `korea26`을 지우기로 하면 그때 |

합계: 서브모듈 외부 저장소 약 530 MB(메인 커밋으로는 복구 불가, 재취득 가능), 추적되는 파일 약 2.5 MB(git으로 복구), 캐시·은행 약 150 MB(복구 불가, 대부분 재생성). `~/.cache/f1sim/inductor` 34 GB는 지도와 무관하므로 계산에서 뺐다. 이번 조사로 지도 정리에서 새로 생기는 큰 용량 이득은 외부 저장소 530 MB가 사실상 전부다.

### 5-3. 지도 생성기 코드 처리 방안

| 파일 | 크기 | 처리 | 이유 |
|---|---|---|---|
| `f1sim/f1sim/trackgen.py` | 498행 | 폐기 후보. 단 `viewer/console/env_editor.py:1060~1567`(에디터의 무작위 트랙 기능)과 `maps.py:328`(`gen:recipe:`), `tests/test_env_editor.py:596`, `docs/environment_editor.md`가 쓴다 | 새 생성기가 나올 때까지 보류 |
| `f1sim/f1sim/track.py`의 `generate_random`·`_gen_*`(1049~1400행) | 약 350행 | 같은 방식. 지도 파일 읽기·벽 처리와 한 클래스(`Track`)에 섞여 있어 통째 삭제 말고 메서드만 제거 | ② 이후 |
| `f1sim/scripts/make_maps.py` | 8행 | 삭제 가능. 참조는 `docs/ros2.md:870` 한 곳 | 지금 가능 |
| `f1sim/tests/test_trackgen.py` | 107행 | 생성기와 함께 삭제 | 생성기 폐기 때 |
| `scripts/track_gallery.py`, `docs/media/f1tenth-trackgen*` | | 삭제 | ② 이후 |
| `f1sim_ros` | `bridge_node.py:80`, `vesc_sim_node.py:78`, `f1tenth_stack_sim.launch.py:23`의 기본값이 `gen:competition:0` | 기본값을 `real:korea26_scan` 같은 (가) 지도로 바꿔야 함 | ② 때 |

### 5-4. 실행 순서와 단계별 확인 테스트

테스트는 ROS 플러그인 때문에 자동 로드를 꺼야 돈다(메모리의 `ros-pytest-plugin-conflict`). 파이썬은 `/home/shchon11/F1tenth_E2E/.venv/bin/python`.

1. **⓪ 보존**(지금) 뒤 확인: `f1sim/tests/test_map_registry.py`, `test_tracks.py`, `test_console_map_card.py`와 `game` 쪽 `server/tests/test_m1.py`. 경로만 바뀌고 id는 같아야 한다.
2. **① 새 시뮬 노출**(지금) 뒤 확인: `f1sim/f1sim/server/tests/test_m1.py`와 `game/tests`(fixtures `catalog.json`).
3. **평가가 끝난 시점**(`cap_exam.py`, compbench dev·confirm, 학습 큐가 모두 쉬는 시점)을 PM이 확인한다.
4. **② 등록 목록 정리** 뒤 확인: `test_tracks.py`, `tracks_oracle.py`, `test_heldout_split.py`, `test_multitrack.py`, `test_training_setup.py`, `test_training_schema.py`, `test_console_training.py`, `test_bare_obstacles.py`, `test_track_modifiers.py`, `test_procedural_obstacles.py`, `test_raceline_feasibility.py`. 이어서 전체 `f1sim/tests`와 `compbench/tests`, `server/tests`. 콘솔 맵 목록과 `compbench` dev 스위트를 직접 한 번 돌려서 korea26_scan 점수가 이전과 같은지 본다.
5. **③ 디스크 삭제**: 서브모듈은 `git submodule deinit` 후 제거 순서. 추적 안 되는 캐시는 마지막에. 삭제 직후 `python -c "from f1sim import maps; maps.load('real:korea26_scan+duct+smooth')"`와 `f1sim.tracks.catalog()`가 오류 없이 도는지 확인.
6. **④ 생성기 폐기**(새 생성기 뒤) 뒤 확인: `test_env_editor.py`, `f1sim_ros` 빌드와 기본 launch.

### 5-5. 지금 지워도 안전한 것과 기다릴 것

| 지금 안전 | 기다릴 것 |
|---|---|
| ⓪ 복사(삭제 아님), ① 새 시뮬 목록 정리(`lab-*` 숨김), `f1sim/scripts/make_maps.py`, `docs/tracks_*.png` 등 그림 11개, `f1sim_ros/maps/random_*`, `external/f1tenth_gym`(코드 참조 없음), `~/.cache/f1sim/racelines`·`centerlines` 중 (가) 아닌 이름 | 등록·분할 코드 수정(②)과 관련 테스트(약 100개 파일), compbench 시험이 쓰는 lab 두 지도와 은행, `korea26` 계열(옛 체크포인트·상자 은행), 서브모듈 `korea_teams`·`ICRA22`·`f1tenth_maps`(⓪ 보존 뒤), `trackgen.py`와 에디터 무작위 트랙(새 생성기 뒤), 실행 중 `cap_exam.py`가 끝나기 전의 `korea26_scan`·상자 은행 일체 |

## 6. PM이 정할 일

1. `real/korea26` 보존 여부. 권고: 보존(75 KB, 추적됨). 옛 체크포인트 `korea26_*` 8개를 다시 돌릴 때 필요하다. 정본은 `korea26-scan`.
2. lab 두 지도. 근거상 대회 지도가 아니고, 동시에 compbench dev·confirm과 일반화 검증(`HELDOUT`)의 유일한 "학습에 안 쓴 실측" 지도다. 지우면 일반화 점수를 잴 실측 지도가 없어진다.
3. `bb21/bb22`와 `rt/spielberg`·`oschersleben`. 지우면 현재 기본 학습 분할 `TRAIN_RULES`가 사실상 `korea26`만 남아 `--tracks train`의 정의가 바뀐다. 이번 학습(korea26_scan 전용)은 영향이 없다.
4. `korea26x12`는 (나)로 보내도 영향이 가장 작다. 권고: 지움.
5. `Oschersleben_map`(IROS 2022 경기 자료)을 (가)로 추가할지.
6. ICRA 2022의 두 판 중 어느 쪽이 대회 크기인지 확인.
