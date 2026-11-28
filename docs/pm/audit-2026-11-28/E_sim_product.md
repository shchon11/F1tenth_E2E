# E. 시뮬레이터·콘솔 현황 조사 (2026-11-28)

조사 방법: 코드를 직접 읽고, 콘솔을 실제로 띄우고(오프스크린 Qt + llvmpipe 소프트웨어 GL, CPU 전용), 생성기를 직접 돌리고, 테스트를 돌렸다. 문서는 참고만 했고 코드와 어긋나는 곳은 아래에 적었다. 기존 파일 수정·commit·학습 없음. `~/.f1sim/console.json`은 전후 md5가 같다(`F1SIM_CONSOLE_PREFS`로 임시 파일 사용).
스크립트: `/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/E/` (cap.py, cap_b.py, gen_syn.py, gen_real.py). 그림·원자료: `E_figs/`.

## 0. 한 줄 요약

- 콘솔은 **화면이 둘뿐인** 주행·학습 앱이다(≡ 메뉴 "주행 / 학습"). 환경 편집기·학습 설정 폼·각종 설정 패널은 **코드가 남아 있지만 화면에 안 나온다**(2026-11-02 정리, `window.py:175-180`). 문서(`environment_editor.md`, `viewer_design.md`)는 이 정리 이전 상태를 설명한다.
- 차량·센서·컴퓨팅은 **파이썬 dataclass 상수**(`params.py`)다. 차종은 사실상 1종, UI에서 바꿀 수 있는 차량·센서 항목은 **0개**.
- 노면은 "차 전체에 하나의 μ(랜덤 또는 고정)"뿐이고, **장소별 노면 차이는 시뮬 코어에 없다**.
- 지도 생성기는 6종(trackgen 포함)이 있고 어느 것도 실제 대회 지도 분포를 따르지 못한다. 그 중 레시피형 trackgen은 콘솔에서 **도달 불가**다.
- 시뮬 물리 코어(`sim.py`)는 UI·학습과 깨끗이 분리돼 있다(import 폐포에 viewer/learn 없음). 결합은 `gym_env.py`(5.1k줄)와 `sim_worker.py`(3.2k줄)에 몰려 있다.

---

## 1. 기능 목록 (콘솔 화면별)

범례: **UI** = 화면에서 가능 / **CLI** = 명령행·코드 수정으로만 / **없음**.

### 1-1. 주행 화면 (기본 화면) — 스크린샷 01, 05, 06, 12, 13

| 조작 | 상태 | 근거 |
|---|---|---|
| 정책(체크포인트) 선택: 검색 팝업, ★추천 묶음 + 모든 런(최근 순), 설명·맵 태그 | UI | `modern.py RichPicker`, `window.py:315` (02) |
| 맵 선택(검색 가능 목록, 실측/생성/레이싱트랙 등) | UI | `window.py:318` (03b) |
| 주행 조건 4종: 혼자 달리기 / 학습 장애물 / 1:1 경주 / 경주+장애물 | UI | `COND_ITEMS` `window.py:244` |
| 시작·정지·일시정지·리셋·수동 조작(키보드) | UI | 상단 바, 단축키 Space, Ctrl+R, Ctrl+. |
| 장애물 시드 입력, 다시 뽑기, 직접 편집(놓기·이동·회전·삭제·저장·불러오기), 치우기 | UI | 조건 바 + `layout_editor.py` (09, 10) |
| 경주 조건: 제어기(MAP 등), 상대차 종류(ForzaETH 등), 상대 속도 범위, 대수 | UI | 조건 바 (12) |
| 카메라 2종(전체/추격), 기본 시점 복귀, 속도 배율 슬라이더 | UI | HUD (05, 06) |
| HUD: 속도·조향·랩·시뮬 시간·배속 타일, 제어 추종(요구 vs 실제) 차트, 표시 4개(정책 궤적·주행 자취·LiDAR·디코더) | UI | `_build_hud` |
| 보는 차 바꾸기 `[` `]`, 도움말 F1/H | UI(단축키만) | `window.py:2399-2410` |
| **차량·센서·컴퓨팅 설정** | **없음** | `SessionConfig`(`protocol.py:99-`)에 차량/센서 필드 0개 |
| 노면 마찰 μ(랜덤/고정), 차 물리 무작위화, 신경망 내부 활성화, saliency, 충돌 모드, ROS2 연동, 녹화, 스크린샷, 행동 샘플링 | **화면에 없음(숨김)** — 위젯은 만들어지지만 `self._legacy.setVisible(False)`. 값은 prefs 파일 복원으로만 들어감 | `window.py:179-186, 233` |
| 맵 만들기·편집·지형 칠하기 | 없음(콘솔 안에서) | `self.editor = None` |
| bag → 지도 / 환경 프로필 | 없음 | CLI(§2 R3) |

### 1-2. 학습 화면 — 스크린샷 11

| 조작 | 상태 | 근거 |
|---|---|---|
| 새 학습 요청: 프리셋 불러오기, 이름, 맵(world), 출발점, 끝 update, 8개 노브(탐색 바닥, 상자 비율 등), GPU 선택, 시험 | UI | `plan_train.py`, `train_requests.py KNOBS` |
| 그 밖의 모든 인자 | UI이나 **CLI 플래그를 텍스트 박스에 직접 입력** | 11 하단 "그 밖의 인자" |
| 진행 감시: 바퀴/분·속도·접촉·탐색 크기 곡선, 체크포인트별 시험 표 | UI | 11 |
| 런 목록, 정지 요청, "주행 화면에서 보기" | UI | `watch_requested` |
| 런 두 개 겹쳐 비교 | **콘솔에는 없음**. 웹 대시보드(`dash_server.py`)에만 있음 | `/api/compare` |
| 실행 방식 | 콘솔은 요청 JSON 파일만 쓰고, 별도 PC의 큐(`tools/train_queue.py`)가 실행·상태 회신. 이 큐 코드가 `docs/research/plan-output-2026-11-01/tools/`에 있음 | `train_requests.py:1-17` |

### 1-3. 콘솔에 없는 화면 (코드만 남음)

| 화면 | 코드 | 상태 |
|---|---|---|
| 환경 편집기(페인트·경로·프롭·SLAM 맵 불러오기·trackgen 카드) | `env_editor.py` 2877줄, `editor_viewport.py` 860줄 | `window.py`에서 생성 안 함. 테스트 1곳만 import. 마지막 수정 2026-09-20 |
| 학습 설정 폼(전체 인자 스키마) | `training_setup.py` 1229줄, `training.py`의 TrainingPage, `training_schema.json` | 콘솔에서 안 씀. 테스트만 사용 |

### 1-4. 별도 도구

| 도구 | 내용 | 비고 |
|---|---|---|
| 웹 대시보드 `dash_server.py`(661줄) | 학습 곡선, 체크포인트별 시험표, 런 비교, 그림 모음, 연구 기록, 영상 | 읽기 전용. 절대경로·Tailscale IP가 상수(`dash_server.py:16-20`). 학습을 **시작/조작은 못 함** |
| 구 네이티브 뷰어(glfw) `viewer/native.py`, 구 웹 뷰어 `viewer/server.py`+three.js | 시뮬 프로세스 안에서 그림 | 콘솔 이전 세대 |
| ROS 연동 `viewer/ros_link.py` | 센서 토픽 발행, `/drive`로 차 인수, rviz 시각화 | 숨김 패널에서만 켬 |

---

## 2. 요구(R1~R6) 대비 격차표

| 요구 | 세부 | 현재 | 근거 |
|---|---|---|---|
| **R1 차량** | 외관 | glb 파일 1개 고정. 생성 스크립트는 있으나 UI 선택 없음 | `assets/f1tenth_car.glb`, `scripts/make_car_model.py`, `gl_scene.py:610 load_car` |
| | 규격(질량, 축거, 크기, 타이어, 한계) | 코드 상수. `m=3.74`, `lf/lr`, `width 0.31`, `length 0.58`, `v_max 12`, `a_max 5` 등 | `params.py:20-` VehicleParams |
| | HW 구성(모터/서보/VESC 지연) | 코드 상수 | `params.py:224` ActuatorParams (`servo_tau 0.04`, `cmd_delay 0.03` :256) |
| | 센서 설정 | LiDAR `n_beams 1081, range 10, rate 40, mount_x 0.297`(`params.py:272-287`), IMU `50 Hz`(:409), 오도메트리 모두 dataclass 상수. 설정 가능 단위는 있으나 **정책 관측 규격(`ObsSpec.n_beams`)과 묶여** 바꾸면 체크포인트가 호환 안 됨. LiDAR rate는 제어 주기(40 Hz)와 같은 값이어야 함 | `learn/obs.py`, `docs/simulator_audit.md` IMU 스케줄 |
| | 컴퓨팅(Jetson 범위) | 없음. 40 Hz에서 25 ms 예산을 **CPU 단일 스레드 배율**로 재는 스크립트만 있음 | `learn/budget.py:1-20` (기기 모델·TOPS·메모리 없음) |
| | 데이터 분리 여부 | `Config.from_dict/from_yaml`이 있으나 **호출하는 곳이 없음**(grep). 차종 파일 없음. `SessionConfig`에 차량/센서 필드 없음 | `params.py:743-758` |
| | 차종 수 | **1종 고정**(F1TENTH Traxxas Slash 기반). 무작위화 범위 표는 있으나 이는 "한 차의 오차" | `randomization.py` |
| **R2 환경** | 노면별 거동 | 차 전체에 μ 하나(랜덤 0.70-1.10 또는 고정). 위치별 μ 장은 없음(`mu_map` 등 grep 0건). 바닥 종류·롤링저항 분리 없음 | `params.py:602-` `vehicle.mu`, 숨김 패널 `노면 마찰 μ` |
| | 장소별 차이 | 없음 | `sim.py`, `track.py` |
| | 벽/덕트/프롭 | 있음. 맵 파일(pgm)+덕트 마스크+프롭 | `track.py`, `props.py` |
| **R3 bag→프로필** | bag→지도 | CLI 있음: `python -m f1sim.bag_map`, `scan_mapper.py`, `slam_map.py`, `map_registry.py`(`~/.f1sim/maps/<id>/`에 놓으면 카탈로그에 자동 등록) | `bag_map.py:1-12`, `map_registry.py:1-15` |
| | 노면·차량 한계 추정 | 스크립트 수준. 동역학 재생 `scripts/replay_dynamics.py`(절대경로 하드코딩), 파라미터는 사람이 읽고 `params.py` 주석에 반영(`servo_tau 0.04` 등 11-09 재측정). 환경 프로필이라는 산출물·파일 형식 없음. 온보드 마찰 추정기는 있음 | `calib/`, `learn/grip_estimator.py` |
| | UI | 없음 | — |
| **R4 지도 제작** | 생성기 | 6종(§5). 콘솔에서 고를 수 있는 것은 `gen:competition|hallway|circuit|control` 시드형뿐 | `track.py:1049`, `trackgen.py` |
| | 편집기 | 코드는 있으나 콘솔에 연결 안 됨 | §1-3 |
| | 대회 지도 수집 | SLAM 지도 불러오기·정리·중심선 추출 코드 있음(`slam_map.py`) | |
| **R5 학습** | UI에서 실행 | 요청 파일을 쓰는 방식. 실제 실행은 다른 PC의 큐. 설정은 노브 8개 + 플래그 텍스트 | `plan_train.py`, `train_requests.py` |
| | 감시 | 콘솔: 곡선 4개 + 시험표. 웹 대시보드: 훨씬 풍부 | 11, `dash_server.py` |
| | 비교 | 콘솔 없음, 웹에만 | |
| | 학습 파이프라인 정합 | 학습 스크립트 계열이 6세대 공존(§6). 콘솔 지원은 `plan_rl`만 | |
| **R6 UI 조작성** | 전반 | 주행·학습 2화면 + 장애물 편집만 UI. 지도·차·센서·환경·bag·비교는 UI 밖 | §1 |

---

## 3. 구조 지도

### 3-1. 프로세스·스레드·통신

```
[콘솔 프로세스: PyQt5, torch 미import]            [sim worker 프로세스: torch]            [학습 PC: 별도 머신]
  ConsoleWindow (3.9k줄)                             SimWorker (sim_worker.py 3.2k줄)
  SessionController (session.py)                      - sim 스레드(_sim_loop)
    _Reader QThread ←── 프레임 Pipe(단방향) ────────── - frame-sender 스레드(drop-oldest)
    제어 Pipe(양방향) ──────────────────────────────→ 명령 처리
  multiprocessing spawn Pipe (session.py:376-385)     RosLink(선택, rclpy 스레드)
  PlanTrainPage ── 요청 JSON 파일 ──(ssh/rsync)──→ train_queue.py ──→ plan_rl 프로세스
                 ←─ _train_status.json, progress.jsonl ──
```

- 프로토콜: `viewer/console/protocol.py` (Qt·torch 둘 다 없이 import 가능). 세대(gen) 스탬프, seq 확인응답, drop-oldest 프레임 규칙이 문서화돼 있다. 메시지는 pickle 직렬화된 dict(`control.send(P.message(...))`) — 언어 중립 아님.
- 콘솔은 torch를 import하지 않는다(실측: `import window` 후 `'torch' in sys.modules == False`). AppImage도 같은 구조(torch 미포함, 워커용 파이썬을 따로 지정).

### 3-2. UI 툴킷·렌더링

- **PyQt5**. 3D는 `QOpenGLWidget`(core profile **OpenGL 3.3**, MSAA 4x, vsync) 위에 **자체 렌더러** `gl_scene.py`(1.5k줄, moderngl로 셰이더·메시 관리, `#version 330`). 차는 glb를 trimesh로 읽어 인스턴싱. 차트는 `QWidget` 직접 페인팅(`LineChart`). 스타일은 `theme.py` 토큰 + 33곳의 `setStyleSheet`.
- 구 렌더 경로 2개가 더 있음: glfw 네이티브(`native.py`), three.js 웹(`server.py`+`static/`).

### 3-3. 모듈 의존

| 계층 | 파일 규모 | 의존 |
|---|---|---|
| 물리 코어: `sim.py`(1.9k), `dynamics`, `actuators`, `lidar`, `imu`, `odom`, `params`, `randomization`, `track`(2.0k) | 실측 import 폐포: `actuators, dynamics, imu, lidar, lidar_triton, odom, params, prop_math, prop_triton, randomization, sim, track` | torch + (선택) triton. **viewer·learn import 없음** |
| 환경: `gym_env.py`(5.1k), `scene`, `props`, `procedural_obstacles`, `opponent_*`, teachers, `mpc`, `raceline` | | `gym_env`가 `f1sim.learn.obs`를 import(`gym_env.py:31`), 반대로 `learn/*`는 `gym_env`를 import → **패키지 수준 순환**(지연 import로 봉합: `gym_env.py:2046` 주석) |
| 학습: `learn/` 86파일 44k줄 | | `learn/watch.py`가 viewer를 import(콘솔 런처 겸함) |
| 뷰어·콘솔: `viewer/` 30k줄 | | viewer가 learn을 76곳 import(catalog, checkpoint_metadata, plan_train 등 — 대부분 torch-free 모듈) |
| ROS: `f1sim_ros/` 5.2k줄 | | |

저장 위치: `~/.f1sim/console.json`(콘솔 설정), `~/.f1sim/maps/`(등록 지도), `~/.cache/f1sim/centerlines/`, `~/f1sim_runs/`(런·학습 요청·상태·프롭 은행), `~/f1sim_videos`, `/run/user/1000/f1sim-console/<인스턴스>/`(워커 소켓·로그, 종료 시 삭제 — 실패한 세션의 로그가 사라져 원인 추적이 어려움을 이번 조사에서 겪음).

### 3-4. 큰 파일과 책임 뒤섞임

| 파일 | 줄 | 내용 |
|---|---|---|
| `gym_env.py` | 5111 | `F1VecEnv` 4153줄(메서드 115). `__init__` 497줄, `step` 381줄. 관측 조립·보상·접촉 회계·상대차·이상적 추종기·장면 재생이 한 클래스 |
| `window.py` | 3860 | `ConsoleWindow` 3776줄(메서드 149). 상단 바·조건 바·HUD·**숨김 패널 ~1,100줄**(`_build_left` 386, `_build_record_card` 192, `_build_centre` 315, `_build_right` 200)이 한 클래스. **숨긴 위젯이 곧 데이터 모델**: 세션 설정과 prefs가 위젯 값을 읽는다 (`window.py:177-179` 주석이 직접 인정) |
| `sim_worker.py` | 3215 | `SimWorker` 2427줄. `build_session` 한 함수 655줄. 체크포인트 로딩·정책 드라이버·환경 생성·프롭·상대·오버레이·ROS·녹화 |
| `env_editor.py` | 2877 | 콘솔에 연결 안 됨 |
| `training.py` | 2382 | Job/JobManager/Progress/차트/TrackPicker/TrainingPage 혼재. 콘솔은 `LineChart`만 씀 |
| `plan_rl.py` | 4524 | 학습기 |

경로·머신 하드코딩: `sim_worker.py:305`(`/home/shchon11/f1sim_runs/prop_banks_box2`), `opponent_table.py:110`, `protocol.py:300`, `dash_server.py:16-20`(경로+Tailscale IP), `scripts/replay_dynamics.py`. 저장소 내 `.py` 6개 파일 12곳. `docs/local_pc.md`는 낡음(RTX 4060 Ti, `/home/shchon11/F1tenth`).

---

## 4. 실행 확인

환경: 오프스크린 Qt(`DISPLAY=:0` GLX 컨텍스트만 사용) + llvmpipe 소프트웨어 GL, `CUDA_VISIBLE_DEVICES=""`(GPU 연산 없음). Xvfb·xdotool 없음. 순수 `offscreen`만으로는 `QOpenGLWidget: Failed to create context`가 나서 3D가 안 그려진다(DISPLAY 필요) — 헤드리스 서버에서 콘솔 스모크가 되지 않는다. `capture_console_media.py`는 `xvfb-run`을 안내하지만 이 PC에는 없다.

| 항목 | 측정(소프트웨어 GL, 다른 프로세스가 CPU를 쓰는 중) |
|---|---|
| 모듈 import | 0.1 s |
| 창 표시 | 0.7-1.6 s (경합 시 7 s까지) |
| 맵 목록 도착 | 창 표시 후 **5 s 시점에도 "맵 목록 읽는 중…"**, 팝업이 빈 채로 열림(03) — 목록은 워커가 보내므로 워커 기동이 끝나야 채워짐 |
| 세션 시작(▶ → 주행 중) | 8.2-9.0 s (맵 로드 ~4 s, 지오메트리 ~4 s). 시작 단계별 진행 문구는 잘 나옴 |
| 속도 | 시뮬 배속 0.4-0.5x (CPU 시뮬 + llvmpipe) — **GPU 성능 수치 아님** |
| 렌더 fps | 측정 불가: 오프스크린에서 `frameSwapped` 97.8회/s는 vsync 없는 수치라 의미 없음. 화면 GPU(5060)에서의 fps는 **측정하지 않았다** |
| 오류 처리 | 없는 맵 지정 시 "맵 'x+smooth'을 읽지 못했습니다" + 트레이스백이 즉시 표시됨(좋음) |

스크린샷(`E_figs/`):

| 파일 | 화면 | 관찰 |
|---|---|---|
| `01_idle_start.png` | 시작 직후 | 정보 구조는 깔끔. 맵 "읽는 중…", HUD 타일은 전부 "—". 정책 제목이 길어 잘림 |
| `02_popup_policy.png` | 정책 팝업 | ★추천 묶음. 이름에 "hc41a u800 (세게 닿으면 그 뒤 return 끊김; 빈 트랙 7.10 s…)" 같은 **연구 내부 용어**가 제목 |
| `03_popup_map.png` / `03b_popup_map_loaded.png` | 맵 팝업(로드 전/후) | 로드 전 빈 상자. 로드 후에도 220 px 높이에 3줄만 보임 |
| `04_popup_condition.png` | 조건 팝업 | 4개 항목, 설명 있음 |
| `05_running_overview.png` | 주행 | 3D 전체보기. 선택 후 **정책 칸에 절대경로**(`/home/.../state_latest.pt`)가 그대로 표시됨. 상태 바에 "맵 지오메트리 업로드 29 ms (worker 생성 3949 ms, 22.8 MB)" 같은 개발자 진단 문구가 항상 노출 |
| `06_running_chase.png` | 추격 카메라 | 차 모델(glb), LiDAR 점, 계획 궤적. 시각 품질은 양호 |
| `09_condition_props_idle_or_restart.png` | 학습 장애물 조건 | 조건 바에 시드·다시 뽑기·편집·불러오기·치우기 |
| `10_layout_editor.png` | 장애물 편집 | 오른쪽 카드로 놓기/이동/회전, "이 배치로 달리기". 화면은 탑뷰 정지 상태로 전환. 단축키 설명이 작은 글씨 6줄로 카드 안에 있음 |
| `11_training_page.png` | 학습 | 3열(설정 폼 / 곡선+시험표 / 런 목록+요청). 시험표 헤더가 "한 바퀴·벽·상자·보인 때(지나가기 몇 초 전)·같은 상자 몇 번째 통과"로 **연구 지표 용어**. 하단 상태 바에 **장애물 편집 문구가 남아 있음**(화면 전환 시 상태 문구 미초기화) |
| `12_race_condition.png` | 1:1 경주 | 조건을 바꾸자 상대차 컨트롤(제어기·상대차 종류·속도·대수)이 나타나지만, 이미 달리던 세션은 **재시작되지 않고 차 1대 그대로**(시뮬 시간 7.7 s 이어짐). 조건 표시와 실제 세션이 어긋남 |
| `13_manual_mode.png` | 수동 조작 | 버튼 토글 |

UX 문제 요약: (1) 숨은 기능 — 마찰 μ·녹화·스크린샷·ROS·DR이 화면에서 완전히 사라졌고 단축키도 없음. (2) 설정 변경이 즉시 적용되는지 재시작이 필요한지 일관되지 않음(12; 속도 배율은 즉시, 조건은 재시작 필요). (3) 한·영·약어 혼용: "MAP (실차 방식)", "update", "exam", "RMS", "LiDAR", "ForzaETH 예측 spliner", CLI 플래그 `--horizon-auto-mult`가 화면에 노출. (4) 상태 문구가 화면 간에 새어 나옴. (5) 학습 화면은 8개 노브 외에 플래그 텍스트 입력이라 비전문가가 쓸 수 없음. (6) 카메라 2종뿐, 차량 시점·자유 시점 없음.

---

## 5. 지도 생성기 평가

생성기 6종: `Track._gen_grid`("competition", 기본), `_gen_competition`("control"), `_gen_circuit`, `_gen_hallway`, `_gen_serpentine`(`track.py:1049`), `trackgen.py`(레시피형, 498줄, 커밋 2개, 편집기 전용). 콘솔 지도 목록의 `gen/*`는 앞의 시드형이다. `_gen_grid`의 docstring이 스스로 "narrow parametric family — more seeds add little new geometry"라고 적었다.

조사: trackgen 시드 0-29 / `gen:competition` 시드 0-29(한 장씩 갤러리) / hallway·circuit·control 각 6 / 실제 지도 36개(`real:*` 13 + `rt:*` 23). 측정은 중심선 0.1 m 재표본 + 0.3 m 가우시안 평활 곡률, 폭은 중심선에서의 2×거리장. 갤러리: `gallery_trackgen.png`, `gallery_gen_competition.png`, `gallery_real_reference.png`. 원자료 `gen_eval_syn.json`, `gen_eval_real.json`.

### 5-1. 숫자 비교 (최소 / 중앙 / 최대)

| 집합 | 길이 m | 중앙 폭 m | 최소 폭 m | 곡률 p95 1/m | 직선 비율 | 급커브 수 | 순회전 ° |
|---|---|---|---|---|---|---|---|
| **실측 실내(13)** | 33 / 64 / 168 | 1.42 / 2.32 / 3.42 | 0.59 / 1.63 / 3.16 | 0.53 / 1.14 / 1.46 | 0.15 / 0.26 / 0.44 | 4 / 11 / 25 | 350 / 365 / 388 |
| 그 중 Korea 2026 계열(대회장) | 46-56 | 1.9-2.3 | 1.6-2.0 | ~1.0-1.2 | ~0.2 | 11-13 | — |
| F1 서킷 1:10(23) | 261 / 403 / 554 | 1.34 / 2.15 / 2.53 | — | 0.06 / 0.30 / 0.52 | 0.59 / 0.70 / 0.83 | 0 / 5 / 15 | — |
| **trackgen(30)** | 49 / 62 / 72 | **1.23 / 1.24 / 1.25** | 1.20 | **0.34 / 0.50 / 0.71** | 0.12 / 0.35 / 0.62 | 0 / 4 / 11 | 353 / 359 / 368 |
| **gen:competition(30)** | 55 / 66 / 78 | 1.66 / 1.84 / 2.27 | 1.47 / 1.63 / 2.10 | 1.27 / 1.43 / 1.66 | 0.20 / 0.46 / 0.70 | 8 / 15 / 24 | 361 / 363 / 365 |
| gen:hallway(6) | 36 / 57 / 64 | 1.88 / 2.50 / 3.10 | 0.62 / 1.27 / 1.94 | 0.97 / 1.93 / 2.48 | 0.68 / 0.78 / 0.89 | 4 / 8 / 8 | 398 / 399 / 400 |
| gen:circuit(6) | 52 / 52 / 53 | 2.16 / 2.19 / 2.28 | 1.76-1.86 | 0.25 / 0.26 / 0.29 | 0.10-0.20 | 0 | 358-361 |
| gen:control(6) | 49 / 53 / 57 | 1.65 / 1.76 / 2.02 | 1.46 / 1.46 / 1.75 | 0.82 / 0.99 / 1.13 | 0.04-0.17 | 11-13 | 361-381 |

(길이 33-168 m인 실측 실내 지도 중 BlackBox 계열 5개가 100 m 이상이라 중앙값이 대회장 지도보다 위로 치우친다. "급커브"는 |곡률|>0.5 구간 개수. 폭 최소 0.59/0.63은 덕트 마스크 잡음일 수 있다.)

### 5-2. 실패 유형 분류

| 유형 | 발견 | 증거 |
|---|---|---|
| 자기 교차 | **없음**. 6종 모두 폐곡선을 통과(trackgen은 `_self_clear`로 거름, 시드 30개 전부 clear, 시도 1-4회, 0.2-0.7 s) | gen_eval_syn.json |
| 폭 비현실 | trackgen: 요청 `lane_width=1.6`인데 **자유 폭 1.20 m 고정**(시드 전체에서 1.20-1.25). 실측 대회장 1.9-2.3의 약 60 %. 모든 시드가 같은 폭이라 폭 변화·병목이 없다. 실측에는 0.6-1.7 m 병목이 있음 | 5-1 표 |
| 단조로움 | trackgen: 30개 모두 반시계, 둥근 "콩"형 외곽에 곡률 p95 0.5(회전반경 ≥2 m) — 실측 1.14의 절반. 헤어핀·접힘 구간이 사실상 없음. 레시피에 헤어핀을 켜도 29개 시드가 서로 다른 순서일 뿐 형태 가족은 하나. gen:competition: 격자 블롭 외곽이라 **직각+둥근 모서리의 직사각형 계열 하나**(갤러리 전체가 L/T/십자 모양). 시드가 늘어도 새 형태가 안 나온다(코드 docstring이 인정) | 갤러리 2장 |
| 잡음 | gen:competition: 맵 전체에 **흩뿌려진 점·짧은 벽 조각**(방 안 장애물 잔재; 실제 SLAM 지도의 스프레이와 닮았지만 임의 배치), 중앙 폭 변화가 시드 간 1.7-2.3로 좁음 | 갤러리 |
| 실제 대회와의 차이 | 실측 지도는 (a) 홀을 가로지르는 직선 벽으로 **트랙이 접혀 한 덕트를 사이에 두고 나란히 달리는** 구조, (b) 폭 1.4-3.4 m의 큰 변화, (c) 33-168 m로 길이 분산이 큼, (d) 비정형 외곽. 생성기에서 (a)는 `_gen_serpentine`만 시도한다(카탈로그에는 있으나 `tracks.py:848` 주석상 학습에서 보류, **이번 조사에서 렌더·측정하지 않음**). (b)(c)(d)는 위 6종 어디에도 없음 | 갤러리 3장 |
| 방향 | trackgen·대부분 CCW. gen:competition은 홀수 시드를 뒤집어 CW를 만든다(미러링이지 새 형태 아님) | |
| 시험이 품질을 안 봄 | `test_trackgen.py` 7개 테스트는 "폐곡선·클리어런스·크기·재현성"만 확인. 폭·곡률·다양성 검증 없음 | `tests/test_trackgen.py:36-` |

**판정 근거(사실)**: 기하학적 결함(교차·끊김)은 없고 재현성은 좋다. 그러나 (1) 분포가 실측과 다르다(trackgen은 폭·곡률 모두 눈에 띄게 단조, competition은 직교 형태 한 가족), (2) 가장 현실적인 접힘 구조(serpentine)는 학습에서 보류돼 있고 이번에 검증하지 못했으며, (3) 레시피형 trackgen은 콘솔에서 도달 불가(§1-3)다. "형편없다"는 평가는 기하 오류보다 **분포의 협소함과 실측과의 거리**에 해당한다.

---

## 6. 테스트·건전성

- 테스트: `f1sim/tests/` 파일 257개(`test_*.py` 246개), 56,094줄, 테스트 함수 약 2,583개.
- 실행: 콘솔·지도 계열 29개 파일(`test_console_*`, `test_trackgen`, `test_tracks`, `test_track_lap`, `test_track_modifiers`, `test_map_registry`, `test_env_editor`, `test_editor_viewport`, `test_training_schema/setup`, `test_smooth_walls`, `test_multitrack`)을 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, CPU, 오프스크린으로 실행: **569 통과 / 4 실패 / 8 건너뜀, 752 s**. 느린 것: `test_multitrack`(274 s), `test_track_lap::test_full_lap_on_random_tracks`(222 s). `test_console_launch_smoke`는 xdotool이 없어 건너뜀.
- 실패 4건은 모두 `test_training_schema.py`: 체크인된 `training_schema.json`이 현재 `ppo`/`dagger` 인자 목록과 **어긋남**(예: `shield_brake`). CLI가 바뀌는 동안 UI 스키마가 따라가지 못한다는 사실이 시험으로 드러난 것. (해당 UI는 콘솔에서 쓰이지 않음.)
- 죽은 코드·중복:

| 항목 | 내용 |
|---|---|
| 학습기 6세대 | `learn/ppo.py` 2820줄, `ppo_v2.py` 1105, `plan_rl.py` 4524, `state_rl.py` 2579, `train_v2.py` 738, `sac.py` 554. 콘솔이 다루는 것은 `plan_rl` |
| DAgger 2세대 | `dagger.py` 1258 / `dagger_v2.py` 964 |
| 모델 2세대 | `model.py` 1752(110곳에서 import) / `model_v2.py`(24곳) |
| 플랜 스텝 | `plan_steps.py` 208줄(20곳) / `plan_steps3.py`(5곳, 제어 지연을 반영한 변형 — 대체가 아니라 병존) |
| 교사 | `teacher.py`, `spliner_teacher`, `lane_teacher`, `layout_line`, `interactive_teacher`, `rollout_teacher` 등 |
| 구 뷰어 | `viewer/native.py`(glfw), `viewer/server.py`+`static/three/`(웹) |
| 콘솔 내부 | `env_editor.py`+`editor_viewport.py`+`training_setup.py`+TrainingPage ≈ 5.5k줄은 시험 외 사용 없음. 숨김 위젯 ~1.1k줄. `set_mode`가 존재하지 않는 `mode_buttons`를 참조 |
| 미사용 API | `Config.from_yaml`(호출자 0), `trackgen`의 UI(편집기 전용) |
| 문서 | `environment_editor.md`는 "주행·학습·환경" 3화면을 서술(현재 2화면), `local_pc.md`는 낡음 |

- 패키징: `pyproject.toml`(setuptools, 의존 torch 포함, `f1sim-console` 진입점). `packaging/build-appimage.sh`(141줄) — torch 없는 GUI 단독 AppImage. **`dist/`에 산출물 없음 → 빌드·실행은 이번에 확인하지 않음**. `f1sim_ros`는 별도 ament 패키지. 저장소 pack 687 MiB.
- 작업 트리는 다른 작업자의 수정(M 10여 개)이 있어 위 수치는 2026-11-28 19:3x 시점 기준.

---

## 7. 재사용 판정용 사실 정리 (판정은 PM)

| 모듈 | 사실 |
|---|---|
| 물리 코어 `sim.py`·`dynamics`·`actuators`·`lidar`·`imu`·`odom`·`params` | import 폐포에 viewer/learn/Qt 없음. 배치 B개 환경을 torch로 벡터화(GPU/CPU). 파라미터는 실측 bag 재생으로 반복 보정된 근거 주석이 풍부(`params.py`). 단점: 차량·센서가 상수, 위치별 노면 없음, torch(+triton) 필수, 센서 주기가 제어 주기에 묶임, 관측 규격과 체크포인트에 결합 |
| `track.py`/`tracks.py`/`maps.py`/`map_registry.py` | 지도 = 점유격자(pgm)+덕트 마스크+프롭. 카탈로그 문법(`real/…@rev#line:*`)이 복잡. 지도 데이터 추가는 폴더 규칙(`~/.f1sim/maps/<id>`)으로 코드 수정 없이 가능 |
| `gym_env.py` | 물리 + 관측 + 보상 + 상대차 + 접촉회계 + 이상적 추종기가 한 클래스. 렌더·시각화와 무관하나 UI 서버로 쓰려면 학습 개념(보상·교사)이 따라옴 |
| `sim_worker.py` | 사실상의 "헤드리스 서버". 프로토콜이 이미 분리돼 있음(`protocol.py`). 그러나 한 파일이 체크포인트·정책·환경·프롭·오버레이·녹화·ROS를 모두 안고 있고, 세션 시작이 "정책(체크포인트) 선택"을 전제로 설계됨(정책 없이 수동·baseline 모드는 별도 경로) |
| `protocol.py`+`session.py` | UI↔워커 계약이 문서화·시험됨(세대·ack·drop-oldest). 전송은 multiprocessing Pipe + pickle이라 같은 머신·같은 파이썬이 전제 |
| `gl_scene.py`/`viewport.py` | moderngl 자체 렌더러. Qt·GL3.3에 묶임. 지오메트리(`TrackGeometry`)는 워커가 배열로 만들어 보냄 → 다른 프런트엔드가 같은 배열을 소비할 수 있는 지점 |
| `ros_link.py` | 이미 "다른 프런트엔드" 하나: 센서 토픽 발행·`/drive` 수신·시각화 토픽(rviz) |
| `window.py` | 3.9k줄 단일 클래스, 위젯이 데이터 모델 |
| `plan_train.py`+`train_requests.py` | 파일 기반 요청/상태 계약. 정규식 검증이 있음(명령줄 주입 방지). 큐 구현은 `docs/research/…/tools/`에 있음 |
| `dash_server.py` | 단일 파일 http 서버, 읽기 전용, 절대경로·IP 하드코딩 |
| `bag_map.py`·`scan_mapper.py`·`slam_map.py` | CLI 완결형 bag→지도 |
| trackgen·`_gen_*` | §5 |

**물리 코어를 헤드리스 서버로 떼어 다른 프런트엔드(웹/게임엔진)에 붙일 때의 결합 지점**
1. 전송: `session.py:376` Pipe+pickle → 소켓/gRPC 등으로 교체 필요. 메시지 종류는 `protocol.py`에 이미 열거돼 있음.
2. 프레임 레이아웃: `viewer/console/frames.py`(496줄)와 `sim_worker._snapshot`(237줄)이 상태를 어떻게 내보낼지 정의. 프런트엔드가 이 형식에 맞춰야 함.
3. 세션 구성: `SessionConfig`(체크포인트·맵 문자열·장애물·상대차 슬롯)에 차량/센서/노면 필드가 없음. R1·R2를 넣으려면 이 계약과 `Config` 주입(`sim_worker.build_session`, 655줄)을 같이 바꿔야 함.
4. 정책 결합: 환경이 `learn.obs`의 관측 규격을 import(`gym_env.py:31`). 센서 구성 변경은 정책 호환성으로 번짐.
5. 지도: 지오메트리를 워커가 만들어 보내므로 렌더러 독립이지만, 지도 카탈로그가 `maps`/`tracks` 문법에 묶여 있음.
6. 환경 의존: 하드코딩 경로(`sim_worker.py:305`), `~/f1sim_runs` 규약, 학습 PC 큐.
7. 런타임: torch 필수, 소프트웨어 렌더 조건에서 시뮬 배속 0.4-0.5x(CPU) — 서버 쪽 GPU 사용 여부는 `device` 인자로 결정(`SessionConfig.device`).

---

## 8. 구조상 가장 큰 문제 5개 (사실 기반)

1. **화면과 코드가 어긋남**: 환경 편집기·학습 설정 폼·마찰/녹화/ROS 패널이 코드(~6.6k줄)로만 남고 화면엔 없음. 숨긴 위젯이 설정의 데이터 모델 역할을 한다(`window.py:177-179`).
2. **차량·센서·노면·컴퓨팅이 데이터가 아니라 코드 상수**이고, 센서는 정책 관측 규격과 묶임. 로더(`Config.from_yaml`)는 있으나 호출자 0. 세션 계약에 해당 필드 없음.
3. **`gym_env.py`/`window.py`/`sim_worker.py`가 각각 4-5k줄 단일 클래스**로 책임이 섞임. `gym_env`↔`learn` 패키지 순환.
4. **학습 인프라가 UI 밖·연구 폴더에**: 큐(`train_queue.py`)와 대시보드가 `docs/research/…/tools/`에 있고 절대경로·IP가 박혀 있음. 콘솔 학습 화면은 요청 파일 쓰기 + 8개 노브 + 플래그 텍스트. 스키마 시험 4개 실패가 이 드리프트를 보여줌. 학습기는 6세대 병존.
5. **지도 파이프라인 단절**: bag→지도·환경 추정은 CLI/스크립트, 편집기·trackgen은 콘솔에 없고, 시드형 생성기는 분포가 협소(§5). R2의 "장소별 노면"은 시뮬 코어에 개념 자체가 없음.

## 9. 한계

- 렌더 fps와 GPU 성능은 측정하지 않았다(소프트웨어 GL).
- AppImage는 빌드·실행하지 않았다.
- 학습 화면의 실제 요청 전송은 시험하지 않았다(학습 금지).
- 콘솔 전체(257개 시험 파일)가 아니라 29개 파일 부분 집합만 실행했다.
