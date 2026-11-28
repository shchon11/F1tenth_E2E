# TENTH 구조 설계서 — Godot 4 프런트엔드 + Python 물리 코어

2026-11-28 · 기술 검증 스파이크 결과를 반영한 초안 v0.1
읽은 것: `audit-2026-11-28/E_sim_product.md`(구조·결합 지점 7개), `D_sim_fidelity.md`(물리 출처), `sim-design-2026-11-28/02_direction.md`·`03_ia.md`·`05_components.md`·`tokens.css`, 코드.
새로 만든 것: `game/`(Godot 프로젝트), `f1sim/f1sim/server/`(시뮬 서버 다리), 이 문서와 스크린샷. 기존 파일은 고치지 않았다.

---

## 0. 결론

1. **구조가 성립한다.** Godot 4.7.2(공식 Linux 배포본, Vulkan Forward+, RTX 5060 Laptop)가 기존 `f1sim` 코어를 그대로 쓰는 Python 서버에 TCP로 붙어, Korea 2026 지도·차(glb)·LiDAR 1081점·디자인 토큰 HUD를 그렸다. 시작·일시정지·리셋 명령이 앱에서 서버로 가고, 랩 계시가 화면에 나온다(스크린샷 2장).
2. **에디터 없이 개발할 수 있다.** 장면·스크립트·테마는 모두 텍스트로 만들었다. 명령 한 줄로 “서버+앱 실행 → N초 뒤 PNG → 둘 다 종료”와 “헤드리스 단위 시험”이 된다. 앱 창은 포커스를 가져가지 않는 8×8 px 창으로 화면 구석에 뜨고, 장면은 1920×1200으로 그려진다.
3. **전송은 TCP + MessagePack으로 한다.** 네 방식의 전송 지연 차이는 0.1 ms 수준이다. 앱은 화면 주기(16.7 ms)마다 소켓을 읽으므로 이 차이는 화면에서 보이지 않는다. 그래서 신뢰성, 큰 메시지(코스 2.2 MB), 원격 접속이 되는지로 골랐다. WebSocket은 같은 본문을 나르는 두 번째 문으로 남긴다.
4. **가장 큰 위험은 그리기가 아니라 시뮬 속도다.** CPU 한 코어의 기존 코어는 실시간의 0.53~0.67배로 돈다. 그리기는 여유가 많다(제한을 풀면 706 fps). 실시간을 만들려면 CPU 컴파일의 그래프 분기를 먼저 고치거나 GPU 경로를 써야 한다(§1.3, §7).
5. **채점은 시뮬 시간으로 한다.** 그러면 서버가 실시간보다 느려도 결과가 바뀌지 않는다. 느리면 보는 사람만 기다린다. 첫 마일스톤(1~2일)은 이 원칙 위에서 “예선 형식 → 채점 결과”까지 간다(§6).

---

## 1. 스파이크 결과

### 1.1 무엇을 만들었나

```
game/                                 Godot 프로젝트 (텍스트만)
  project.godot                       자동 로드 App·SimLink, 테마, Forward+, 1920×1200 기준
  scenes/race_slice.tscn              수직 조각: 환경·조명·코스·차·LiDAR·카메라·HUD
  scripts/autoload/app.gd             명령줄 인자, 캡처, 측정 JSON, 입력 주입(--inject)
  scripts/autoload/sim_link.gd        전송 4종(tcp/ws/udp/shm), 세대·지연·왕복 측정
  scripts/net/msgpack.gd              MessagePack 부분집합(해독·부호화)
  scripts/race/*.gd, floor_grid.gdshader   코스 메시, 차, LiDAR 점, 카메라, 바닥 격자
  scripts/ui/{tokens.gd(생성), fmt.gd, keys.gd, digit_readout.gd, race_hud.gd}
  theme/tenth_theme.tres              tokens.css에서 생성한 테마
  assets/{fonts(목업 글꼴 woff2), models/f1tenth_car.glb}
  tests/{run_tests.gd, test_*.gd, fixtures/}   애드온 없는 시험 실행기 + 파이썬이 만든 고정 자료
  tools/{godot.sh, tenth.py, tokens_to_theme.py, make_fixtures.py}
  bench/transport_bench.tscn          전송 비교 장면(차 2대 × 1081점)
f1sim/f1sim/server/                   시뮬 서버 다리 (기존 모듈은 import만)
  wire.py  protocol.py  session.py  transports.py  server.py  bench.py  probe.py  tests/
```

### 1.2 환경과 버전

| 항목 | 값 |
|---|---|
| Godot | **4.7.2-stable official (ed1daf0bf)**, 2026-08-18 공개. `game/.godot_bin/`(git 무시). SHA-512를 `tools/godot.sh`에 고정 |
| 렌더 | **Vulkan 1.4.329, Forward+**, 장치 #0 = NVIDIA GeForce RTX 5060 Laptop GPU(화면 담당 GPU). 4070은 쓰지 않았다 |
| 화면 | Xorg `:0`(GNOME). Xvfb 없음. Xephyr·lavapipe는 있지만 쓰지 않음 |
| 서버 | 저장소 `.venv`(torch 2.11 CPU 경로, `CUDA_VISIBLE_DEVICES=""`), 한 스레드 |
| 메모리 | 서버 최대 663 MB + 앱 최대 960 MB. 합계 3 GB 안 |

### 1.3 측정

**(1) 수직 조각**(Korea 2026, 차 1대, Follow-the-Gap, CPU, TCP, 1920×1200 화면 밖 렌더)

| 항목 | 값 | 메모 |
|---|---|---|
| 서버 기동 | 0.44 s | 지도 0.1 s + 지오메트리 0.29 s. 코스 메시지 2.15 MB(덕트 정점 42,000개) |
| 시뮬 배속 | **0.53~0.67×** | 한 스텝 38~47 ms(물리 25 서브스텝 + LiDAR). 감사 D §6과 같다 |
| 앱 fps | 60(수직 동기 상한), 프레임 p95 16.7 ms | 제한을 풀면 **706 fps**, 1.42 ms(p99 2.08 ms) |
| 앱 CPU | 한 코어의 13 %(60 fps 정상 상태) / 69 %(제한 없음) | 6 s 짧은 실행은 기동·코스 메시 구성을 포함해 평균 28 % |
| 서버 CPU | 한 코어의 100 % | 실시간보다 느려 쉬지 않는다 |
| 상태 프레임 | 5.9 KB/장(차 1대), 해독 0.11~0.22 ms(GDScript), 적용 0.26~0.49 ms | |
| 서버 송신 → 앱 수신 | p50 9~10 ms | 앱이 화면 주기마다 읽어서 생기는 대기가 대부분 |
| 입력 → 상태 반영 | 일시정지 34 ms, 출발 67 ms | `--inject`로 키 동작을 넣고, 바뀐 첫 상태를 받은 때까지. 서버 루프 한 바퀴(25 ms) + 스텝 시간이 대부분. 화면까지는 여기에 1프레임을 더한다 |
| 랩 계시 | Follow-the-Gap 한 바퀴 15.793 s(시뮬 시간), 접촉 0 | 스크린샷 `spike_race_chase.png` |

**(2) 전송 비교**: 합성 서버(`server/bench.py`). 차 2대 × LiDAR 1081점, 40 Hz, 한 장 11.7 KB.

| 전송 | 전송만의 지연 p50 / p95 [ms] (파이썬 수신기) | 왕복 p50 [ms] | 앱(Godot) 수신 프레임 / 15 s | 앱 fps | 앱 CPU | 서버 CPU | 앱 쪽 지연 p50 [ms] |
|---|---|---|---|---|---|---|---|
| **TCP** | 0.41 / 0.47 | 0.36 | 600 / 600 | 60 | 13.4 % | 2.5 % | 16.1 |
| WebSocket | 0.51 / 0.72 | 0.51 | 600 / 600 | 60 | 12.7 % | 2.6 % | 9.9 |
| UDP | 0.38 / 0.45 | 0.28 | 600 / 600 | 60 | 12.8 % | 2.2 % | 14.6 |
| 공유 메모리 | 0.31 / 0.45 | — (한 방향) | 595 / 600 | 60 | 13.3 % | 1.8 % | 7.2 |

- 전송만의 지연에는 파이썬 해독 0.25 ms가 들어 있다. 실제 전송 차이는 0.1~0.2 ms다.
- 앱 쪽 지연(7~16 ms)이 전송마다 달라 보이는 이유는 전송이 아니다. 서버 송신(40 Hz)과 앱 화면 주기(60 Hz)의 위상이 실행마다 달라서다. 왕복이 모두 15.6 ms(한 프레임)인 것도 같은 이유다.
- 앱의 해독(0.36 ms)과 적용(0.7 ms, LiDAR 메시 2개 재구성)은 프레임 예산(16.7 ms)의 6 %다.

**(3) CPU 컴파일 시험**(기존 코드를 고치지 않고 서버에서 `torch.compile(sim._roll_physics)`)

- 정상 상태 15.5 ms/스텝, 즉 **64 step/s = 1.6×**. 실시간이 된다.
- 그러나 첫 컴파일이 캐시 없이 약 90 s, 캐시가 있어도 17~33 s 걸렸다. 서버 루프 안에서는 4분이 지나도 정상 상태에 가지 못해 중단했다.
- 원인은 코드에서 확인했다. `sim.py:1588`의 `if k in self.imu_idx`가 IMU 위상(50 Hz 대 40 Hz라서 4가지)마다 다른 그래프를 굳힌다. 그래서 컴파일이 4번 일어난다.
- 고칠 곳: 위상을 텐서 인자로 넘겨 그래프를 하나로 만든다(§3, 결합 지점 7).

**(4) 개발 루프**

| 명령 | 결과 |
|---|---|
| `python3 game/tools/tenth.py import` | 자산 가져오기. 첫 실행 3.8 s(처음 한 번은 글꼴보다 테마를 먼저 읽어 오류 줄이 나온다. 두 번째부터 0줄), 다시 실행 2.4 s |
| `python3 game/tools/tenth.py test` | GDScript 12개 시험(0.7 s, 모든 스크립트·장면 적재 시험 포함) + 서버 파이썬 30개 시험(8 s). 일부러 넣은 실패와 깨진 스크립트를 잡고 종료 코드 1을 돌려주는지 확인함 |
| `python3 game/tools/tenth.py shot --seconds 12 --out game/out/race.png [--camera top] [--inject race_toggle@5,race_toggle@6] [--uncapped]` | 서버(setsid) → 준비 줄 대기 → 앱(화면 밖) → N초 뒤 PNG + 측정 JSON → 앱이 스스로 끝남 → 서버를 PID로 끝냄. CPU·메모리 기록 |
| `python3 game/tools/tenth.py bench` / `probe` | 전송 4종 비교(앱 기준 / 파이썬 수신기 기준) |
| `python3 game/tools/tenth.py run` | 사람용 보통 창. **이번에는 실행하지 않았다**(사용자 화면을 가로채므로) |

**화면을 가로채지 않는 방법**(시험해서 정했다)

- 프로젝트 거울 `game/.capture_host/`를 둔다. 원본 파일은 심볼릭 링크로 비추고, `override.cfg`에 `display/window/size/no_focus=true` 한 줄만 다르게 둔다.
  - Godot X11은 no_focus 창을 override-redirect로 만든다. 창 관리자가 위치를 바꾸지 않고 포커스도 넘기지 않는다.
  - 시험 결과 활성 창 ID가 실행 전·중·후 그대로였다. no_focus가 없으면 Godot 창이 포커스를 가져갔다.
- 창은 8×8 px로 주 모니터 오른쪽 아래 구석에 둔다. 장면은 `content_scale_mode=viewport`, 1920×1200으로 그리므로 캡처 해상도는 창 크기와 무관하다.
- **최소화는 쓸 수 없다.** 창이 모두 최소화되면 Godot은 그리기와 `_process`를 멈춘다(시험에서 확인).
- `--position`은 Godot이 화면 안으로 끌어온다. 그래서 창을 완전히 화면 밖에 둘 수는 없었다. 남는 것은 최대 8×8 px다.

---

## 2. 프로세스 구성, 책임 경계, 프로토콜

### 2.1 구성

```
 ┌──────────────── 사용자 PC ─────────────────┐        ┌──── 연구실 PC / 서버(원격 노드) ────┐
 │  TENTH 앱 (Godot)                          │  TCP   │  시뮬 서버 (python -m f1sim.server)  │
 │  표시·조작·편집, 화면 상태, 디자인 토큰     │◀──────▶│  세션: Simulator + 드라이버 + 규칙    │
 │  파이썬·torch 없음                         │ 47800  │  자료 서비스: 차·장소·코스·기록 파일  │
 │        ▲ WS(같은 본문, 관전·브라우저용)      │        │  리플레이 기록(.tenthrec)            │
 └────────┼───────────────────────────────────┘        └──────────────┬──────────────────────┘
          │                                                           │ 파일 요청/상태(기존 train_requests)
          │                                            ┌──────────────▼──────────────────────┐
          └──────── (훈련 상태 읽기: 서버 경유) ───────▶│  학습 작업자 (plan_rl 등, GPU 배치)    │
                                                       │  큐 train_queue.py, ~/f1sim_runs     │
                                                       └─────────────────────────────────────┘
 실차(Jetson AGX Orin)·ROS: 기존 f1sim_ros 그대로. bag은 서버의 bag 분석이 읽는다.
```

| 프로세스 | 책임 | 하지 않는 것 |
|---|---|---|
| **TENTH 앱**(Godot) | 화면 7개, 3D 표시, 입력(키보드·게임패드), 편집 UI(차고 시트, 트랙 스튜디오), 화면 상태(고른 탭·카메라), 캡처 | 물리·센서·채점 계산, 파일 형식 해석, torch |
| **시뮬 서버**(Python) | 세션 실행(실시간 맞추기·일시정지·리셋), 드라이버 실행(학습된 정책·기준선·사람 명령), 규칙과 채점, 자료 서비스(차·장소·코스·기록 파일의 유일한 읽기·쓰기 주체, 검증·호환 검사), bag 분석 작업, 리플레이 기록 | 화면 그리기 |
| **학습 작업자** | 배치 학습·시험(기존 `plan_rl`, `train_queue.py`) | UI. 서버와는 요청/상태 파일로만 |
| **원격 노드** | 같은 서버 코드를 다른 PC에서 실행. 앱은 주소만 바꾼다 | — |

원칙:
- **파이썬이 자료의 유일한 주인이다.** 앱은 차·장소·코스 파일을 직접 읽지 않고 서버에 요청한다. 그러면 원격에서도 같고, YAML 해석기를 GDScript로 다시 만들 필요가 없다.
- **앱은 상태를 갖지 않는다**(화면 상태 제외). 같은 세션에 앱 둘(조작 + 관전)이 붙어도 된다.

### 2.2 전송 선택

**TCP + MessagePack, u32 little-endian 길이 접두 틀.** 같은 본문을 WebSocket으로도 내보낸다(관전·브라우저·프록시 뒤 원격용).

| 기준 | TCP | WebSocket | UDP | 공유 메모리 |
|---|---|---|---|---|
| 지연(이 PC) | 0.41 ms | 0.51 ms | 0.38 ms | 0.31 ms |
| 신뢰·순서 | 있음 | 있음 | 없음(명령은 따로 재전송 필요) | 최신 칸 하나 |
| 큰 메시지(코스 2.2 MB) | 그대로 | 그대로 | **못 보냄**(데이터그램 64 KB). 옆에 TCP가 필요 | 칸 크기 제한 |
| 앱 → 서버 명령 | 같은 연결 | 같은 연결 | 같은 소켓(손실 가능) | **없음**(반대 칸을 따로 만들어야) |
| 원격 | SSH 터널·Tailscale 그대로 | HTTP 프록시도 통과, 브라우저 가능 | NAT·방화벽에 약함, LAN에서 IP 조각화 | **같은 PC만** |
| Godot 구현 | `StreamPeerTCP`(내장) | `WebSocketPeer`(내장) | `PacketPeerUDP`(내장) | 공식 API 없음. 파일로 읽으면 매번 다시 열어야 했다(같은 핸들은 stdio 버퍼 때문에 새 값을 못 봄) |

- 무선망에서 TCP는 앞 패킷이 늦으면 뒤 패킷도 기다린다. 이 지연이 화면에 쌓이지 않도록 서버가 앱마다 **최신 상태 한 장만** 들고 있다가 보낸다(drop-oldest). 앱도 한 프레임에 여러 장이 오면 마지막 것만 쓴다.
- 프레임 크기가 작다(차 2대 11.7 KB × 40 Hz = 470 KB/s). 그래서 원격 LAN에서도 대역폭은 문제가 아니다.

### 2.3 메시지 (스키마 v1, `server/protocol.py`)

모든 메시지는 사전이고 `k`(종류)와 `v`(스키마 버전)를 갖는다. 버전이 다르면 앱이 연결을 끊고 사실을 한 문장으로 보인다. 지금은 v1만 있다.

| 방향 | 종류 | 내용 | 신뢰 |
|---|---|---|---|
| 서버→앱 | `hello` | 서버 이름, 제어 주기, 지원 명령, 서버 시각 | 버리지 않음 |
| | `track` | 코스 메시(덕트·벽·바닥: `viewer/geometry.py` 배열 그대로), 중심선, 출발선, 경계, LiDAR(줄 수·시야·장착), 차 치수 | 버리지 않음 |
| | `state` | 40 Hz. `gen, seq, t(시뮬 시각), ts(송신 시각), rt(배속), status, cars[], scans[]` | **버려도 됨**(최신만) |
| | `event` | 랩 완료(시간, 무충돌 여부, 최고 갱신), 접촉 | 버리지 않음 |
| | `ack` / `error` / `pong` | 명령 확인(seq), 오류 문장, 왕복 측정 | 버리지 않음 |
| 앱→서버 | `hello` | 앱 이름, 버전 | |
| | `cmd` | `op: start/pause/resume/stop/reset`, `seq` | |
| | `ping` | 앱 시각. **전송 계층이 즉시 답한다**(시뮬 루프에 묶이지 않게) | |

추가 예정(M1~M2): `session.start{spec}`, `session.result`, `drive{steer, speed}`(사람 조작 50 Hz, 0.5 s 무입력이면 정지 — 기존 콘솔 `manual` 규칙), `data.get/put/validate`(자료 서비스), `job.*`(bag 분석·훈련 상태).

- `state.cars[i]`: `id, x, y, yaw, v, vy, wz, steer, roll, pitch, cmd_steer, cmd_v, lap, s, lap_t, last_lap, best_lap, clean_streak, best_streak, contacts, contact, driver`. 다음에 넣을 것: `plan(k×3)`, 접촉 프레임, 상자 동적 위치.
- `state.scans[i]`: `{car, pose[x,y,yaw](센서 자세), ranges(f4 1081, 0 = 반환 없음), hit(u1)}`.
  - 점 대신 거리 배열을 보낸다. 크기가 절반이고, 센서 계약과 같은 모양이다. 점 계산은 앱이 한다(1081점 0.3 ms).
- 배열은 `{"$nd": "f4", "shape": [...], "data": <bin little-endian>}`로 보낸다. Godot은 `PackedByteArray.to_float32_array()` 한 번으로 읽는다.
- 세대(`gen`)·`seq`·ack·drop-oldest 규칙은 기존 콘솔 `protocol.py`와 같다.

**두 언어의 규격 일치를 시험으로 묶는다.** 파이썬이 만든 실제 프레임(`game/tests/fixtures/state_frame.msgpack`)을 GDScript 시험이 읽는다. 파이썬 시험은 표준 MessagePack 바이트를 고정해서 확인한다. 규격이 바뀌면 `make_fixtures.py`를 다시 돌린다.

### 2.4 리플레이 기록 형식 `.tenthrec`

- **서버가 보낸 메시지를 같은 틀(u32 길이 + MessagePack) 그대로 이어 쓴 파일**이다(`server.py --record`로 구현됨). 첫 두 메시지는 `hello`, `track`이고 그 뒤로 `state`·`event`가 시간순으로 온다.
- 다시 보기 = 앱의 SimLink가 소켓 대신 이 파일을 읽는 것. 같은 화면 코드가 생방송과 다시 보기를 모두 그린다. 되감기는 앞에서부터 색인(`seq` → 파일 위치)을 만들어 한다.
- 크기: 차 1대 5.9 KB × 40 Hz ≈ 14 MB/분. 오래 보관할 것은 zstd로 압축하고(LiDAR 배열이 대부분), 장면(분석 탭)은 구간만 잘라 둔다.
- 에이전트 시험에도 쓴다: `--replay=파일`로 시뮬 없이 같은 화면을 결정적으로 캡처한다(M1에서 추가).

---

## 3. 데이터 모델

### 3.1 공통: 값마다 오차와 출처

모든 물리·센서 값은 사전 하나(**양(quantity)**)로 적는다. 맨 숫자는 검증에서 거부한다.

```yaml
mount_x: {v: 0.126, u: m, err: 0.005, src: measured, ref: "bag:korea26/0827-115713#tf_static", date: 2026-08-27,
          note: "base_link→laser를 무게중심 기준으로 환산"}
```

| 키 | 뜻 |
|---|---|
| `v`, `u` | 값, SI 단위(각도는 화면에서 도, 파일은 단위를 적은 대로) |
| `err` 또는 `lo`/`hi` | 표준오차(±) 또는 구간. 없으면 화면에 —. 가정 값의 무작위 범위는 `range: [lo, hi]` |
| `src` | `measured`(실측 ■) / `estimated`(추정 ◧, 구간 필수) / `spec`(규격 ▣) / `assumed`(가정 □) / `lower_bound`(하한만 ▬) |
| `ref`, `date`, `n` | 근거(bag·방법·문서), 날짜, 근거 bag 수 |
| `note` | 사람 말 한 줄. **파일에 주석을 쓰지 않는다**. 앱이 고쳐 다시 쓸 때 주석이 사라지므로, 설명은 모두 값으로 둔다 |

- 형식: **YAML**(사람·에이전트가 읽기 쉽고 PyYAML이 이미 쓰인다). 서버가 키 순서를 고정해서 쓰므로 git diff가 작다.
- 검증: `server/schema/*.py`(dataclass 또는 pydantic)로 검사한다. 저장할 때마다, 그리고 CI에서 모든 파일을 검사한다.
- 학습 무작위화 범위는 이 구간에서 만든다. 가정 값은 넓게, 측정 값은 `err`의 2배로 잡는다(D 감사 §7.2의 권고).

### 3.2 파일과 저장 위치

| 객체 | 파일 | 위치 | 비고 |
|---|---|---|---|
| 차량 빌드 | `cars/<id>.car.yaml` | 저장소 `tenth_data/` (git) | 섀시·타이어·구동계·조향·센서 목록과 장착 자세·컴퓨팅 프리셋·외관 |
| 장소 | `venues/<id>/venue.yaml` + `map.yaml/pgm` | `tenth_data/` | 노면 구역별 그립, 실측 지도(bag_map 산출), LiDAR 환경(원거리 누락 등) |
| 코스 | `courses/<id>.course.yaml` | `tenth_data/` | 덕트 선·벽·출발선·방향·장애물 배치와 일정. 장소를 참조 |
| 보정 기록 (차 × 장소) | `records/<car>@<venue>/<date>.record.yaml` | `tenth_data/` | bag으로 잰 값 + held-out 오차 |
| 드라이버 | `drivers/<id>.driver.yaml` | `tenth_data/` | 종류, 체크포인트(내용 해시 + 경로), 센서 계약 |
| 규칙 프리셋 | `rules/<id>.rules.yaml` | `tenth_data/` | 대회 형식(시간, 채점, 출발, 접촉 처리) |
| 세션 | `sessions/<id>/session.yaml` + `run.tenthrec` + `result.yaml` | `~/.tenth/` (개인) | 형식 + 조합 + 상대 + 조건 → 결과지·다시 보기 |
| 학습 실행 | `~/f1sim_runs/<run>/` + `run.yaml` | 기존 위치 유지 | 레시피·조합·장비. 기존 `train_requests` 계약을 감싼다 |
| bag | 원본은 `real_data/`·`/data` | 기존 위치 | `tenth_data/bags.yaml`에 id·sha256·경로만 |

예시(초안, 숫자는 `params.py`와 D 감사 값):

```yaml
# tenth_data/cars/real-2.car.yaml
schema: tenth.car/1
id: real-2
name: 실차 2호
based_on: f1tenth-traxxas-slash-4x4
chassis:
  mass:          {v: 3.74, u: kg, src: assumed, ref: "f1tenth_gym 기본값", note: "이 차 ini는 3.518, 저울 기록 없음"}
  wheelbase:     {v: 0.3302, u: m, src: spec, ref: "Traxxas Slash 4x4"}
  cog_to_rear:   {v: 0.17145, u: m, src: assumed, ref: "f1tenth_gym"}
  yaw_inertia:   {v: 0.04712, u: kg*m^2, src: assumed, range: [0.035, 0.06], note: "bag으로 구분 안 됨(11-09)"}
tires:
  front: {B: {v: 2.0, src: estimated, ref: "2026-09-28 대회장 요 이득 맞춤", note: "조향 이득과 맞바뀜, 한계 미검증"},
          C: {v: 1.8, src: estimated}, grip_scale: {v: 0.92, src: assumed}}
steering:
  limit_left:    {v: 0.412, u: rad, src: measured, ref: "bag 서보 창(대회)"}
  limit_right:   {v: -0.254, u: rad, src: measured, ref: "bag 서보 창(대회)"}
  servo_tau:     {v: 0.04, u: s, src: estimated, ref: "11-09 재생 맞춤(지연과 합만)"}
  cmd_delay:     {v: 0.03, u: s, src: estimated, lo: 0.03, hi: 0.06}
drivetrain:
  accel_max:     {v: 5.0, u: m/s^2, src: estimated, ref: "11-09 전류 한계 구간"}
  erpm_per_mps:  {v: 4204, src: measured, ref: "bag ERPM/속도"}
sensors:
  - slot: lidar
    model: hokuyo-ust-10lx
    beams:     {v: 1081, src: spec}
    fov:       {v: 270, u: deg, src: spec}
    rate:      {v: 40, u: Hz, src: spec}
    range_max: {v: 10, u: m, src: assumed, note: "메시지는 30 m까지"}
    mount: {x: {v: 0.126, u: m, src: measured, ref: "tf_static"}, z: {v: 0.110, u: m, src: measured},
            pitch: {v: 0.0, u: deg, src: assumed, range: [-1.15, 1.15]}}
  - slot: imu
    model: vesc-bmi160
    rate: {v: 50, u: Hz, src: measured}
compute: {preset: agx-orin, control_rate: {v: 40, u: Hz, src: spec}, budget_ms: 25}
appearance: {body: open-chassis, number: 2}
```

```yaml
# tenth_data/courses/kr26-final.course.yaml
schema: tenth.course/1
id: kr26-final
venue: kr26-hall              # 노면 값은 장소에서 물려받는다
walls: {from_map: true, duct_diameter: {v: 0.30, u: m, src: measured}}
start: {x: 1.2, y: -0.4, yaw: 1.571}
direction: ccw
obstacles:
  - {id: box1, shape: box, size: [0.35, 0.32, 0.30], pose: [3.1, 2.0, 0.0], schedule: [{from: 0, to: 50}]}
validation: {closed: true, min_width_m: 1.62, baseline_lap: {driver: ftg, laps: 3, contacts: 0}}
```

```yaml
# tenth_data/rules/kr-event-quali.rules.yaml  (프리셋 예: 팀이 겪은 국내 형식)
schema: tenth.rules/1
id: kr-event-quali
name: 예선 형식 · 90초 한 번
duration_s: 90
start: {grid: standing, at: start_line}
scoring: [{key: clean_streak, order: desc}, {key: best_lap, order: asc}]   # 무충돌 연속 바퀴 → 최고 랩
contact: {count: [wall, prop], hard_mps: 0.5}
obstacles: course_schedule      # 코스에 적힌 일정대로(예: 50초에 치움)
time_base: sim                  # 채점은 시뮬 시간
```

같은 틀로 `roboracer-2025-quali`(5분 히트 2회, 최고 랩 + 무충돌 연속)와 `roboracer-2025-h2h`(나란히 30 cm 출발, 20바퀴 선착, 3판 2선승, 과실 차 2 m 뒤 재출발)를 둔다. 팀의 국내 대회는 1대1을 **반대편(반 바퀴 떨어져)에서 출발**했다. 그래서 출발 방식도 규칙 값이다.

### 3.3 `params.py`에서 옮기는 길

1. **역방향 사상표.** `server/compose.py`에 “파일 경로 → `Config` 필드” 표를 하나 둔다. 예: `chassis.mass → vehicle.m`, `sensors[lidar].mount.x → lidar.mount_x`, `venue.surface.default.grip → vehicle.mu`.
   - `compose(car, venue, record, overrides) -> Config`를 만든다.
   - 기록(차 × 장소)이 차와 장소의 값을 덮어쓴다. 우선순위는 세션 조건 > 기록 > 장소 > 차 > `params.py` 기본값.
2. **초기 파일 생성.** `tools/params_to_yaml.py`로 `Config()` 기본값에서 `cars/f1tenth-default.car.yaml`과 `venues/default/venue.yaml`을 만든다.
   - 출처는 주석에서 반자동으로 채운다. `params.py` 주석은 "measured"가 42줄, "bag" 65줄, "gym" 4줄, 가정 5줄이다. 채운 뒤 사람이 검토한다.
3. **동일성 시험.** `compose(기본 파일들) == Config()`를 필드마다 확인한다(무작위화 범위 62개 포함). 이 시험이 통과하기 전에는 아무것도 바꾸지 않는다.
4. **분류.** `params.py` 필드를 넷으로 나눈다.
   - 차: Vehicle 대부분, Actuator, Lidar·Imu·Odom의 사양·장착.
   - 장소: `mu`, `road_tilt*`, 덕트 반사·누락, IMU 진동.
   - 시뮬 설정: `SimParams`, 수치 정규화. 이것은 파일로 내보내지 않는다.
   - 무작위화: 값의 구간에서 만든다.
5. `params.py`는 그대로 남는다. 코드 기본값이자 사상표의 대상이다. 코어(`sim.py`)는 계속 `Config`만 받는다. **코어는 바뀌지 않는다.**

### 3.4 센서 구성과 정책 입력 규격: 드라이버 계약

문제: 학습된 정책은 관측 규격(`ObsSpec`: 줄 수, 쌓기, 정규화)과 LiDAR 장착(`args.lidar_mount_x`)을 체크포인트에 갖고 있다. 센서를 바꾸면 그 정책은 다른 입력을 받는다.
- 지금 콘솔은 줄 수·쌓기·정규화가 다르면 시작을 거부한다(`sim_worker.obs_spec_problems`).
- 배포 쪽 `resample_ranges`는 줄 수를 조용히 보간하고 시야를 확인하지 않는다(`obs.py:102`).
- 기준선은 `ScanContract` + `bind_scanner` + `map_scan`으로 방위각 기준 적응을 이미 한다.

해결: **드라이버마다 센서 계약(요구 사항)을 파일로 두고, 차 빌드와 맞춰 보는 검사기 하나를 모든 곳(세션 시작, 차고, 고르기 창)에서 쓴다.**

```yaml
# tenth_data/drivers/mir-43.driver.yaml
schema: tenth.driver/1
id: mir-43
name: 미르 43
kind: learned                 # learned | baseline | human
checkpoint: {path: "~/f1sim_runs/hs43b/state_u900.pt", sha256: "…"}
requires:
  control_rate_hz: 40
  lidar: {beams: 1081, fov_deg: 270, range_max_m: 10, adapt: none,
          mount_x_m: {trained: 0.12555, dr: [0.10, 0.15]}, pitch_deg: {dr: [-1.15, 1.15]}}
  odom_speed: true
  imu: {rate_hz: 50}
output: {kind: plan, points: 10, follower: map}
trained_on: {car: real-2@<해시>, venues: [kr26-hall], recipe: "…"}
```

검사기 `f1sim/contracts.py`(torch 없음)는 차 빌드 × 드라이버 → 항목마다 ✓ / △ / ✕와 한국어 문장을 돌려준다.

| 판정 | 조건 | 화면 |
|---|---|---|
| ✕ | 필요한 센서 없음, 제어 주기 다름, 학습된 드라이버의 줄 수·시야 다름(`adapt: none`) | “이 드라이버는 줄 수 720으로 훈련했습니다. 지금 차는 1081입니다.” 출발 버튼 비활성 |
| △ | 장착이 훈련 무작위 범위 밖, 기준선의 시야 차이로 안 보이는 비율 > 0(`unseen_fraction`), `range_max` 다름(자름) | 출발 가능, 이유 표시 |
| ✓ | 같거나 훈련 범위 안 | — |

- 차고에서 잠긴 값(줄 수, 초당 스캔 수)을 바꾸면 서버가 모든 드라이버에 검사기를 돌린다. 그리고 “드라이버 4명 중 3명이 다시 훈련해야 합니다”를 돌려준다(디자인 `ConfirmSheet`).
- 학습 쪽은 체크포인트를 저장할 때 이 계약을 같이 써야 한다. 기존 체크포인트는 `extra["spec"]`·`meta`·`args`에서 계약을 만들어 내는 변환기를 쓴다.
- `gym_env.py:31`이 `learn.obs`를 import하는 순환은 `ObsSpec`·정규화 함수를 `contracts`(또는 `f1sim/obs_spec.py`)로 내려서 끊는다(§4 결합 지점 4).

---

## 4. Python 코어 리팩터링

### 4.1 결합 지점 7개

| # | 지점(E 감사 §7) | 처리 | 스파이크에서 한 것 |
|---|---|---|---|
| 1 | 전송: `session.py` Pipe + pickle | `server/wire.py`(MessagePack) + `transports.py`(TCP 주, WS 보조). 콘솔 `protocol.py`의 gen·seq·ack·drop-oldest 의미를 v1에 옮긴다. 옛 콘솔은 그대로 둔다 | 완료(4종 구현, 시험 30개) |
| 2 | 프레임 모양: `frames.py`, `sim_worker._snapshot` | v1 `state`(cars[], scans[])로 문서화하고 고정 자료로 두 언어를 묶는다. `plan`·접촉 프레임·상자 위치·상대 표식을 차례로 넣는다 | 차 자세·속도·조향·랩·LiDAR까지 |
| 3 | 세션 구성: `SessionConfig`(필드 43개, 차·센서 없음) | `session.yaml`(형식 + 조합 + 상대 + 조건)을 서버가 읽는다. `compose()`로 `Config`를 만든다(§3.3) | 코드에서 `Config` 4개 값만 바꿈 |
| 4 | 정책 결합: `gym_env` → `learn.obs` | `ObsSpec`·정규화를 torch 없는 `contracts`로 내린다. 드라이버 계약과 검사기(§3.4). env는 계약 객체를 받는다 | 기준선(FTG)은 `ScanContract`를 그대로 씀 |
| 5 | 지도: `tracks`/`maps` 문법 | 코스·장소 파일이 `maps.load`(그대로)의 이름을 참조한다. 지오메트리 배열(`viewer/geometry.build_track_geometry`)은 **그대로** 보낸다 | 완료: 콘솔과 같은 메시를 Godot이 그림(감는 방향만 뒤집음) |
| 6 | 환경 의존: 하드코딩 경로(6파일 12곳), `~/f1sim_runs` | `f1sim/paths.py` + 작업 공간 설정 하나(환경 변수 또는 `~/.tenth/config.yaml`) | 서버는 경로를 쓰지 않는다 |
| 7 | 런타임: torch 필수, CPU 0.6× | 서버 연산 장치를 세 갈래로 둔다. **cpu-eager**(지금 0.53~0.67×), **cpu-compiled**(IMU 위상 `sim.py:1588`을 텐서 인자로 바꿔 그래프 하나 → 1.6× 예상), **gpu-graph**(콘솔의 CUDA 그래프 경로, 1.0×). 채점은 시뮬 시간이므로 셋 다 같은 결과 | 측정 완료(§1.3) |

### 4.2 거대 파일 분해 순서

1. **`viewer/sim_worker.py`(3.2k줄) — 먼저.** 새 서버가 이 자리를 대신한다.
   - (a) M1에서는 **감싸기 어댑터**로 시작한다. `build_session`(655줄: 체크포인트 → 지도 → 레이스라인 → env → 자아 드라이버 → 관측 검사 → 준비)을 그대로 불러 세션을 만들고, `_snapshot` 사전을 v1 `state`로 옮긴다. 콘솔이 지금 돌리는 학습된 드라이버와 조건이 그날 바로 새 앱에 나온다.
   - (b) 그다음 조각을 하나씩 떼어 낸다. 순서는 `server/drivers/checkpoint.py`(§1 체크포인트 + §6 자아 드라이버 + §8 검사), `drivers/baseline.py`, `drivers/human.py`, `world.py`(§2~5 지도·env·상자·상대), `overlays.py`, `ros_bridge.py`(기존 `ros_link`).
   - 조각마다 “같은 시드에서 어댑터와 새 경로의 `state`가 같다”는 동일성 시험을 둔다.
   - 하드코딩 경로(`sim_worker.py:305` 상자 은행 등)는 떼어 낼 때 `paths.py`로 옮긴다.
2. **`gym_env.py`(5.1k줄) — 학습이 쓰는 파일이라 마지막에, 학습 담당과 맞춰서.**
   - 관측 조립 / 보상 / 접촉 회계 / 상대 슬롯 / 이상적 추종기 / 장면 재생을 구성 요소로 나눈다.
   - 고정 시드 롤아웃이 비트 단위로 같아야 합친다. **돌고 있는 학습이 있을 때는 손대지 않는다.**
3. **`viewer/console/window.py`(3.9k줄) — 고치지 않는다.** 동결했다가 폐기한다.
4. 학습기 6세대 정리는 이 프로젝트 범위 밖이다(훈련소는 `plan_rl` 요청 계약만 감싼다).

### 4.3 그대로 / 감싸기 / 버리기

| 구분 | 대상 |
|---|---|
| **그대로 쓴다** | `sim.py`, `dynamics`, `actuators`, `lidar(+triton)`, `imu`, `odom`, `track`, `maps.load`, `map_registry`, `bag_map`·`scan_mapper`·`slam_map`, `viewer/geometry.py`·`contours.py`(메시 배열), `learn/baselines/*`(FTG, Disparity, 외부 기준선), `contact_severity`, `params.py`(기본값·사상 대상), 감사 `C_scripts/comp_metrics.py`의 채점 논리(서버 `rules.py`로 옮김) |
| **감싼다** | `sim_worker.build_session`·`_snapshot`(M1 어댑터, 이후 분해), `params.Config`(→ `compose`), `plan_train`·`train_requests`(훈련소 요청), `dash_server`의 읽기 API(훈련 곡선·시험표 → 서버 `job.*`), `calib/*`·D 감사 `D_tools/models.py`(bag 프로필 추정), `trackgen.py`(새 생성기의 출발점), `gym_env`(학습된 드라이버의 관측 조립) |
| **버린다**(새 앱이 덮은 뒤) | Qt 콘솔 전체(`window.py`, `modern.py`, `widgets.py`, `viewport.py`, `gl_scene.py`의 렌더 부분), 구 뷰어 `native.py`·`server.py`+`static/three`, 연결 안 된 `env_editor.py`·`editor_viewport.py`·`training_setup.py`·`TrainingPage`(약 5.5k줄), 숨김 위젯 약 1.1k줄. 시드형 생성기 `_gen_*`는 회귀 시험용으로만 남긴다 |

---

## 5. Godot 프로젝트

### 5.1 장면 트리

```
root (Window)
├─ App        (자동 로드) 인자·캡처·측정·입력 주입·창 모드
├─ SimLink    (자동 로드) 서버 연결, 최신 상태, 신호(track/state/event/ack)
├─ Store      (자동 로드, M1) 현재 조합(차·장소·드라이버), 자료 캐시, 서버 자료 요청
├─ Router     (자동 로드, M1) 화면 7개 이동, 주소(/race/setup 등), 숫자 키 1~7
└─ Main (scenes/app.tscn)
   ├─ TopBar / SessionStrip     위 띠(조합 칸 3, 상태) · 주행 중 세션 띠
   ├─ ScreenHost                화면 하나만 붙임: hub · garage · venue · studio · race(setup/live/result) · camp · lab
   │   └─ race_live.tscn        Viewport3D(코스·차·LiDAR·궤적·카메라) + HUD 판들(Plate)
   ├─ HintBar                   그 화면에서 되는 키만(Keys 표에서)
   ├─ Toasts · ConfirmSheet
   └─ DiagDrawer                진단 서랍(연결·지연·fps·서버 로그)
```

- 화면은 `scenes/screens/*.tscn` + 같은 이름의 스크립트로 둔다. 컴포넌트(05 문서의 이름 그대로: `Plate`, `SheetRow`, `SourceBadge`, `DigitReadout` …)는 `scripts/ui/components/`에 둔다. 스크립트로 만들고, 장면은 얇게 둔다.
- 3D 화면 공용 노드: `TrackView`(서버 메시), `CarView`(glb + 외관), `LidarView`(점), `PlanRibbon`, `RaceCamera`(추격·위에서·차 시점·자유). 스파이크 코드가 출발점이다.

### 5.2 텍스트 기반 개발 규약 (에이전트용)

1. **에디터를 열지 않는다.** `.tscn`에는 노드 이름·형·스크립트·몇 개 속성만 둔다. 복잡한 UI는 스크립트에서 만든다(스파이크 HUD처럼). `uid=`는 손으로 쓰지 않는다. 경로만 써도 된다. Godot이 만든 `.uid`·`.import` 파일은 저장소에 둔다.
2. **버전은 하나.** `tools/godot.sh`가 4.7.2-stable 바이너리를 SHA-512로 고정해 내려받는다. 올릴 때는 이 파일만 바꾸고 전체 시험·캡처를 다시 돈다.
3. **새로 받은 저장소**에서는 `tenth.py import`를 먼저 한다. 첫 실행의 테마 오류 줄은 무시하고, 두 번째부터 0줄이어야 한다.
4. **시험** = `tenth.py test`.
   - 시험 파일은 `game/tests/test_*.gd`(`TestCase` 상속, `test_*` 함수)다. 애드온은 없다.
   - 모든 스크립트·장면 적재 시험이 문법·형 오류를 잡는다. `godot --check-only`는 자동 로드를 모르고, 실패해도 종료 코드 0이라 쓰지 않는다(확인함).
   - 서버 규격이 바뀌면 고정 자료를 다시 만든다.
5. **캡처로 눈으로 확인한다.** 화면을 바꾼 작업은 `tenth.py shot …` PNG를 직접 열어 보고 보고서에 붙인다. 숫자만으로 끝내지 않는다.
   - 입력 경로는 `--inject=동작@초`로 시험한다.
   - 결정적 캡처가 필요하면 `--replay=…tenthrec`(M1)로 시뮬 없이 그린다.
6. **화면을 가로채지 않는다.** 에이전트는 `shot`·`bench`·`test`만 쓴다(포커스 없는 8×8 창). `run`은 사람용이다.
7. **프로세스는 PID로만 끝낸다.** `tenth.py`가 setsid로 띄우고 SIGTERM → 6 s 뒤 SIGKILL 한다. `pkill`은 쓰지 않는다.
8. **GDScript는 형을 적는다**(`var x := …`, 인자·반환형). 경고를 오류로 보는 설정은 M1에서 켠다.

### 5.3 디자인 토큰 → Godot 테마

- `tools/tokens_to_theme.py`가 `tokens.css`의 `:root`를 읽어 두 파일을 만든다. 손으로 고치지 않는다.
  - `scripts/ui/tokens.gd`(`class_name Tokens`): 의미 색 52개(`Tokens.C_ACCENT` …), 간격·높이 23개, 모션 시간, 글자 단계 표. 원색 `--p-*`는 `PALETTE`에만 둔다.
  - `theme/tenth_theme.tres`: 기본 글꼴(Plex Sans KR), 글자 단계마다 **형 변형**(`ReadoutXL`·`ReadoutL`·…·`Caps`·`Micro`·`Data`, Label 기반). 판 StyleBox(`Plate`·`Strip`·`HintBar`·`KeyCap`·`ChipAccent`).
- **글꼴 섞기**: CSS의 `font-family` 순서(Saira → Plex KR)를 `FontVariation.fallbacks`로 옮겼다. 숫자·라틴은 Saira, 한글은 Plex KR로 떨어지는 것을 시험으로 확인했다.
- **CSS로 그린 것 중 StyleBox로 안 되는 것**(14° 사선 딱지, 깎은 모서리, 모서리 눈금, 점선 테두리, 배경 흐림)은 `_draw()`에서 다각형으로 그리는 작은 컴포넌트로 만든다. 배경 흐림 8 px는 화면 텍스처 셰이더로 하되, 원격·저사양에서는 끈다(05 문서 §8.4).
- **바뀌는 숫자는 `DigitReadout`**으로 그린다. Saira 숫자 폭이 달라서 글자마다 고정폭 칸에 그린다. 초당 10번만 갱신한다(스파이크에 구현됨).
- 목업 글꼴은 한글 2,350자 부분집합이다. 제품에는 원본 전체(OFL)를 넣는다.

### 5.4 3D 자산·입력

- **차 모델**: 기존 `f1tenth_car.glb`(ROS 좌표: 뒤 차축 원점, z 위)를 Godot 가져오기 그대로 쓴다.
  - 부모 노드에서 x축 −90° 회전하고, 무게중심 기준으로 `lr`만큼 뒤로 민다.
  - 앞바퀴 노드(`wheel_fl/fr`)를 조향각으로 돌린다. 정점 색을 바탕색으로 쓰도록 재질을 고친다.
  - 외관(색·번호·뒤 범퍼 라임 띠 = 학습된 드라이버)은 실행 중 재질 덮어쓰기로 한다.
  - 새 차종은 `scripts/make_car_model.py`가 glb를 만들고 차 빌드의 `appearance`가 가리킨다.
- **지도는 Godot 자산이 아니다.** 서버가 메시 배열을 보낸다(Korea 2026: 2.15 MB, 1회). 지도의 주인은 파이썬 하나다.
- **좌표**: 시뮬 (x, y, z) → Godot (x, z, −y), yaw = `rotation.y`. 서버 메시는 OpenGL 감는 방향이라 삼각형 순서를 뒤집는다(`Geo.mesh_from`).
- **입력 맵은 코드 표**(`scripts/ui/keys.gd`)다. `project.godot`의 `[input]` 직렬화는 에이전트가 고치기 어려워서 쓰지 않는다. 동작 이름 → 키 목록 → 키 안내 글자를 한 표에 두고, HintBar도 이 표를 읽는다. 게임패드(사람 드라이버)는 M1에서 같은 표에 축 동작을 더한다.

---

## 6. 새 트랙 생성기

### 6.1 요구 사양

**목표 분포**: 실측 실내 13개(E 감사 §5-1)에 맞춘다. 대회장 계열(Korea 2026)을 기본 묶음으로 둔다.

| 특징 | 실측 실내 (최소 / 중앙 / 최대) | 대회장 계열 | 합격 기준(초안) |
|---|---|---|---|
| 길이 | 33 / 64 / 168 m | 46~56 m | 고른 묶음 분포와 Wasserstein 거리 ≤ 실측 묶음 안 거리의 1.5배 |
| 중앙 폭 | 1.42 / 2.32 / 3.42 m | 1.9~2.3 m | 같음 |
| 최소 폭(병목) | 0.59 / 1.63 / 3.16 m | 1.6~2.0 m | 규칙 하한(RoboRacer 1.20 m) 이상 |
| 곡률 p95 | 0.53 / 1.14 / 1.46 1/m | 1.0~1.2 | 분포 거리 기준 |
| 직선 비율 | 0.15 / 0.26 / 0.44 | ~0.2 | 분포 거리 기준 |
| 급커브 수(\|κ\|>0.5) | 4 / 11 / 25 | 11~13 | 분포 거리 기준 |

- 지금 생성기가 못 하는 것: 폭 변화·병목, 덕트 하나를 사이에 두고 나란히 달리는 **접힘**, 비정형 외곽, 길이 분산. trackgen은 폭이 1.20 m로 고정이고, 곡률 p95가 0.5로 실측의 절반이다.
- **스타일 조절**(트랙 스튜디오 슬라이더):
  - 길이, 폭 범위, 병목 수
  - 헤어핀 수(회전 ≥ 150°, 반경 0.6~1.2 m)
  - 시케인 수(반대 방향 연속 커브 2~3개)
  - 접힘 구간 수(평행 주행 ≥ 4 m, 사이 덕트 하나)
  - 방향(CW/CCW), 장소 바닥 크기(예: 12×16 m 방 안), 무작위 시드
- **출력**:
  - 덕트 중심선 폴리라인(지름 0.30 m)과 코스 파일(§3.2). 점유 격자는 서버가 만든다.
  - 중심선과 폭 프로필.
  - “실제 대회장과 닮은 정도” 점수와 항목별 분포 위치.
- **검증**(저장 전 깃발):
  - 닫힘, 자기 교차 없음, 덕트 사이 여유 ≥ 지름 + 0.1 m, 방 안에 들어감.
  - 최소 폭 ≥ 설정값.
  - **기준 드라이버(Follow-the-Gap) 3바퀴를 양방향으로 무접촉 완주**(서버 세션을 헤드리스로 돌림).
  - 실패하면 막힌 곳 좌표를 돌려준다.
- **성능**: 후보 하나 < 1 s(검증 주행 제외). 후보 6개를 동시에 보여 준다.

### 6.2 접근 비교

| | A. 조각 문법 + 분포 맞춤 거절 표본 | B. 방 분할 골격 | C. 실측 지도에서 배운 곡률 연쇄 |
|---|---|---|---|
| 방법 | 직선·일정 반경 커브·헤어핀·시케인·접힘 조각을 거북이 그래픽으로 잇는다. 닫힘은 마지막 몇 조각의 길이·각을 최적화해 맞춘다. 폭은 조각마다 뽑는다. 실측 분포에서 먼 후보는 버린다 | 방 바닥을 칸으로 나눠(BSP·미로) 통로 골격을 만든다. 골격을 부드럽게 해 중심선을 얻고 양옆에 덕트를 놓는다 | 실측 13개의 중심선을 곡률 구간으로 잘라 구간 사이 전이 확률을 배운다. 새 연쇄를 뽑고 닫힘을 최적화한다. `+warp`(이미 있음)로 변형을 더한다 |
| 스타일 조절 | **직접**(조각 수가 곧 슬라이더) | 간접(골격 모양에 달림) | 약함(학습된 분포를 따름) |
| 실측 닮음 | 거절 표본으로 맞춤 | 접힘·비정형 외곽이 자연스럽다 | **구성상 높다** |
| 새로운 모양 | 높음 | 중간 | 낮음(13개에 묶임) |
| 구현 | 기존 `trackgen.py`(레시피·자기 간섭 검사, 498줄) 확장 | 새로 작성 | 새로 작성 + 자료 적음 |
| 위험 | 닫힘·방 안 채우기가 어렵다 | 헤어핀 수 같은 요구를 맞추기 어렵다 | 과적합, 다양성 부족 |

**권장: A를 주로, B의 “접힘”을 A의 조각 하나로, C의 실측 통계를 합격 기준으로 쓴다.**
- 출발점은 `trackgen.py`다(이미 레시피 JSON, 자기 간섭 검사, 시드 재현이 있다).
- 첫 단계는 폭 변화와 접힘 조각을 더하고, 실측 분포 거리로 거절하는 것이다.

---

## 7. 일정

마일스톤은 “사용자가 화면에서 할 수 있게 되는 것”으로 정의한다. 작업 단위의 검수 기준에는 모두 **`tenth.py test` 통과 + 해당 화면 캡처 PNG를 작업자가 직접 열어 본 기록**이 들어간다(아래 표에는 추가 기준만 적는다).

### M1 (1~2일) — 레이스 화면에서 고른 드라이버로 예선 형식을 달리고 대회 채점 결과를 본다

| 단위 | 내용 | 병렬 | 선행 | 검수 기준 |
|---|---|---|---|---|
| M1-A | 서버 `rules.py` + 규칙 프리셋 2개(`kr-event-quali` 90 s, `roboracer-2025-quali` 5분). 시뮬 시간 기준 채점(무충돌 연속 → 최고 랩), 결과 사전. `comp_metrics.py` 논리를 옮김 | ○ | — | 합성 랩 기록으로 순위·동점 처리 시험. 결과 JSON 고정 자료 |
| M1-B | 서버 드라이버 목록: FTG, Disparity, 사람(`drive` 메시지), 학습된 드라이버 = `sim_worker.build_session` 감싸기 어댑터 | ○ | — | 드라이버마다 `shot` 1장. FTG 3바퀴 무접촉. 어댑터가 첫날 안 되면 학습된 드라이버는 M2로 넘기고 그 사실을 적는다 |
| M1-C | 프로토콜 `session.start/result`, `list.drivers/formats`, `--replay` 재생 | △ | A·B 인터페이스(초안으로 먼저 고정) | 파이썬 시험 + GDScript 고정 자료 시험 |
| M1-D | 앱 틀: TopBar·HintBar·Router·Store, 레이스 구성 화면(형식·드라이버 고르기, 호환 표식 자리) | ○ | — | 고정 자료로 캡처(서버 없이) |
| M1-E | 주행 HUD: 세션 판(남은 시간·채점 두 항목), 랩 타이밍 판, 바퀴 기록, 세션 띠 | △ | C의 필드 | 90 s 세션 캡처 3장(출발·중간·끝) |
| M1-F | 결과 화면: 결과지(형식별 채점), 바퀴 표, 접촉 목록, “다시 달리기” | △ | A의 결과 스키마 | `tenth.py shot --flow quali --driver ftg` 한 줄로 구성 → 주행 → 결과 PNG 3장 + 결과 JSON |
| M1-G | 서버 연산 장치 선택(`--device cpu|cuda`), 시작 시 배속 표시 | ○ | — | 5060에서 1.0× 이상 또는 CPU 배속을 HUD에 정직하게 표시 |

첫날: A·B·D·G를 병렬로 한다. 둘째 날: C·E·F와 합쳐 끝에서 끝까지 확인한다.

### M2 (3~4일) — 차고에서 차 값을 출처와 함께 보고, 센서 장착을 바꿔 새 차로 저장하고, 그 차로 달린다
- 단위:
  - `params_to_yaml.py` + `compose` + 동일성 시험
  - `contracts.py` 검사기 + 기존 체크포인트 계약 변환기
  - 서버 `data.*` API(목록·읽기·검증·저장)
  - 차고 화면(시트 행·출처 배지·RowEditor·3D 차 + 시야 겹침·호환 목록·다른 이름으로 저장)
- 병렬: 서버 셋과 화면 하나.
- 검수: 잠긴 값을 바꾸면 “N명 중 M명” 확인 창이 뜬다. 저장한 차로 M1 흐름이 돈다.

### M3 (1주) — bag 하나를 넣으면 지도와 장소·차 값이 나오고, 그 장소로 달린다
- 단위:
  - `bag_profile` 작업(재현되는 값부터 자동: 지연, 트림, 제동 한계, 서보 창, ERPM 이득, 자이로 단위. 마찰은 하한만. D 감사 §7.3)
  - `bag_map` 연결
  - 마법사 화면 5단계
  - 장소 화면(노면 구역 칠하기)
- 선행: M2의 자료 서비스.
- 검수:
  - 대회장 bag 6개로 만든 프로필이 held-out bag에서 “선형 자전거 + 지연” 기준선보다 나쁘지 않거나, 나쁘면 그렇다고 표시한다.
  - 사전 bag과 대회 bag의 차 설정이 섞이면 경고한다.

### M4 (1주) — 트랙 스튜디오에서 코스를 생성·편집·검증해 저장하고, 상자 일정을 짠다
- 단위: 생성기(§6, 접근 A), 닮은 정도 점수, FTG 완주 검증 작업, 편집 뷰포트(덕트 선 끌기·상자 놓기 — 옛 `layout_editor` 조작을 옮김), 일정 띠.
- 검수: 후보 30개의 분포 거리 표 + 갤러리 PNG. 검증 실패 지점 표시.

### M5 (1주) — 훈련소에서 레시피로 훈련을 걸고, 곡선을 보고, 저장 지점을 달려 보고, 드라이버로 올린다
- 단위: `job.*`(기존 `train_requests`·큐·`dash_server` 읽기 감싸기), 레시피 파일, 훈련소 화면(장비 일정·곡선·성적표), 올리기 → 드라이버 계약 파일 작성.
- 검수: 실제 큐에 요청 파일이 써지고(학습은 사용자 허락 뒤), 기존 런의 곡선·시험표가 화면에 나온다.

### M6 (1~2주) — 본선 1대1을 달리고, 다시 보기·랩 비교·사고 장면을 분석한다
- 단위: 상대 드라이버(상대 슬롯 감싸기), 1대1 규칙(나란히 / 반대편 출발, 선착 바퀴, 과실 재출발), `.tenthrec` 색인·되감기, 분석 탭 3개.
- 검수: 같은 세션의 생방송과 다시 보기 캡처가 같은 시각에서 일치한다.

### M7 (1주) — 다른 PC·서버의 시뮬에 붙고, 설치본으로 받고, 옛 콘솔을 끈다
- 단위: 원격 접속(서버는 기본 127.0.0.1, 원격은 Tailscale/SSH 터널 + 토큰), Godot 내보내기(Linux, 내보내기 템플릿 고정) + 서버 환경 설치 스크립트, 옛 콘솔 기능 대조표 통과 후 폐기.
- 검수: 연구실 PC의 앱에서 이 PC의 서버로 주행하고 왕복 지연을 표시한다. 대조표 항목이 모두 ✓다.

---

## 8. 위험과 완화

| 위험 | 내용(근거) | 완화 |
|---|---|---|
| **시뮬 속도** | CPU 한 코어 0.53~0.67×. CPU 컴파일은 1.6×지만 IMU 위상 분기 때문에 그래프가 4개 생긴다(첫 컴파일 수 분) | 채점은 시뮬 시간. `sim.py:1588` 위상을 텐서로(코어 쪽 작은 수정, 학습 동일성 시험과 함께). 5060 GPU 경로. 배속을 HUD에 항상 표시 |
| 컴파일 캐시 | 환경 변수(스레드 수)가 다르면 캐시를 못 쓴다(서버에서 215 s 기동을 겪음) | 서버 기동 환경을 고정한다. 캐시를 미리 데우는 명령을 둔다 |
| 앱 쪽 비용 증가 | 차 4대 이상·상자·궤적이 늘면 GDScript 해독·점 생성이 늘어난다(지금 차 2대 1.1 ms/프레임) | 프레임당 최신 한 장만 해독. 점 생성은 셰이더(거리 배열을 텍스처로)로 옮길 수 있다. 필요하면 해독만 GDExtension(C++)으로 |
| 큰 코스 메시지 | Korea 2026 2.15 MB, 덕트 정점 42k. 앱의 메시 구성은 GDScript 반복문(시간 미측정) | 정점을 서버에서 Godot 좌표·순서로 보내 앱의 반복을 없앤다. 압축, 코스 해시로 캐시 |
| 원격 | 무인증 TCP. 무선망에서 순서 대기 | 기본 127.0.0.1. 원격은 Tailscale/SSH 터널 + 토큰. 최신 한 장 규칙. 관전은 WS |
| 화면 GPU 경합 | 5060이 화면 + 때때로 학습 rank-1을 맡는다(메모: 브라우저 페이지가 SM 72 %를 먹은 일) | 앱은 60 fps 상한, 멈춤·대기 화면에서는 저전력 모드, `nvidia-smi pmon`으로 확인하는 진단 서랍 |
| 패키징 | Godot 내보내기 템플릿(버전 고정 필요), 서버는 torch 포함 수 GB | 앱과 서버를 따로 배포한다. 앱은 작은 실행 파일, 서버는 기존 venv. 같은 PC면 앱이 서버를 띄운다 |
| Godot 버전 | 4.x 마이너 업데이트에서 `.tscn`·API가 바뀔 수 있음 | `godot.sh` SHA 고정. 올리기는 별도 작업(전체 시험 + 캡처 비교) |
| **에이전트 개발 한계** | 에디터 없이 레이아웃을 코드로 쓰면 길다. 보기 좋은지 판단은 캡처를 보는 눈에 달렸다. `--check-only`를 쓸 수 없다. GDScript 형 오류는 실행할 때 난다. 셰이더 디버깅이 어렵다 | 컴포넌트를 05 문서 이름대로 작게 나누고 견본 화면(상태 모음)을 M1에 만든다. 적재 시험 + 형 표기 + 경고를 오류로. 캡처를 항상 보고서에 붙인다. 디자인 검수는 사람이 마일스톤마다 한다 |
| 화면 가로채기 | 캡처 창이 8×8 px로 구석에 남는다. Godot이 위치를 화면 안으로 끌어온다 | 포커스는 안 가져간다(확인). Xvfb가 생기면 가상 화면으로 옮긴다 |
| 글꼴 | 목업은 부분집합(2,350자) | 제품은 OFL 원본 전체, 예약 이름 조항 확인 |
| 두 UI 병행 | 옛 콘솔 동결 중 서버 변경이 콘솔을 깨뜨릴 수 있음 | 새 서버는 새 패키지에만 둔다. 공유 모듈(`geometry`, `baselines`)은 그대로 쓰고 바꾸지 않는다. 콘솔 시험을 CI에 남긴다 |

---

## 9. PM이 결정할 것

1. **자료 파일 형식**: YAML + 값마다 `{v, u, err, src, ref}`. 파이썬 서버만 읽고 쓴다. 승인 여부.
2. **출처 다섯 가지**: 실측·추정·**규격**·가정 + **하한만**. 디자인 문서의 ‘규격’ 제안을 확정할지.
3. **M1의 학습된 드라이버 경로**:
   - 옛 `sim_worker`를 감싸면 빠르고 지금 콘솔의 모든 드라이버가 나온다. 대신 옛 결합이 남는다.
   - 새 세션에 정책만 옮기면 깨끗하지만 1~2일을 넘길 위험이 있다.
   - 권장은 감싸기로 시작하고 M2부터 떼어 내는 것.
4. **대화형 서버의 연산 장치**: CPU(지금 0.6×, 컴파일을 고치면 1.6× 예상) 또는 5060 GPU(화면과 공유). 그리고 “채점은 시뮬 시간” 원칙.
5. **기본 규칙 프리셋**: RoboRacer 2025(5분 히트×2, 나란히 출발) 또는 팀이 겪은 국내 형식(상자 일정, 1대1 반대편 출발). 둘 다 둘 경우 무엇을 먼저 보일지.
6. **자료 저장 위치**: 저장소 `tenth_data/`(공유·이력) 또는 사용자 폴더. 세션·리플레이는 `~/.tenth/` 권장.
7. **원격 보안**: Tailscale/SSH 터널만 허용할지, 토큰 인증을 더할지.
8. **코어 수정 허용 범위**: CPU 실시간을 위한 `sim.py` IMU 위상 수정처럼 학습 경로에도 닿는 변경을 이 프로젝트에서 할지, 학습 담당과 묶어 할지.
9. **트랙 생성기 접근**: A(조각 문법) + 실측 분포 합격 기준으로 갈지.

---

## 부록: 스크린샷과 재현

- `spike_race_chase.png`: 추격 카메라, 시작 48 s 뒤. 2번째 바퀴, 직전·세션 최고 15.793 s(보라), 무충돌 연속 1, 접촉 0, 배속 0.53×(호박색 경고).
- `spike_race_top.png`: 위에서 본 정사영. Korea 2026 덕트 배치 전체와 차·LiDAR 점.
- 재현:
  ```
  python3 game/tools/tenth.py import
  python3 game/tools/tenth.py test
  python3 game/tools/tenth.py shot --seconds 48 --out game/out/race_chase_lap.png
  python3 game/tools/tenth.py shot --seconds 24 --camera top --out game/out/race_top.png
  python3 game/tools/tenth.py bench --transports tcp,ws,udp,shm --seconds 15   # 앱 기준
  python3 game/tools/tenth.py probe --transports tcp,ws,udp,shm --seconds 10   # 전송만
  ```
- 측정 원자료(실행 때마다 새로 쓰임, git 무시): `game/out/*_metrics.json`, `bench_summary.json`, `probe_summary.json`, `server.log`, `godot.log`.
- 이번에 하지 않은 것:
  - 사람용 보통 창 실행(`tenth.py run`)과 실제 키보드 입력. 입력 경로는 `--inject`로만 시험했다.
  - 원격 PC 접속, Godot 내보내기(설치본), 게임패드.
  - 앱의 코스 메시 구성 시간 측정.
  - CPU 컴파일 서버의 정상 상태 운전. 벤치 스크립트에서만 1.6×를 쟀다.
