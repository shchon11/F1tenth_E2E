# TENTH 앱 (설치본)

이 PC와 연구실 PC(10.100.0.2)에 설치돼 있다. 옛 콘솔(f1sim Console AppImage와 그 아이콘)은 그대로 있다.

## 실행

- 바탕화면이나 앱 메뉴의 **TENTH** 아이콘을 누른다. 명령으로는 `~/Applications/TENTH/tenth-launch`.
- 실행 순서
  1. 작은 진행 창이 뜨고 시뮬 서버가 올라온다(10~20 s).
  2. 서버가 준비되면 앱 창이 열린다.
  3. 앱 창을 닫으면 서버도 끝난다.
- 이미 열려 있으면 다시 누를 때 "이미 실행 중"이라는 알림만 뜬다.
- 서버가 뜨지 못하면 오류 창에 원인 한 줄과 서버 로그 경로가 나온다.
- GPU를 쓰지 못해 CPU로 돌게 되면 알림이 뜬다. CPU에서는 실시간의 약 0.4배로 돈다.
- 문제를 찾을 때 쓰는 명령(창이 뜨지 않음)
  - `tenth-launch check`: 서버만 띄웠다가 끝낸다. 출력에서 어느 코드 사본과 torch를 쓰는지 볼 수 있다.
  - `tenth-launch flow --out DIR`: 메인 → 레이스 준비 → 주행(60 s) → 결과를 자동으로 밟는다. 화면마다 PNG를 남기고 `DIR/flow_summary.json`에 배속·메모리·결과를 적는다.
  - `flow`의 앱 창은 포커스를 받지 않는 8×8 창이라 화면을 가로채지 않는다. 다만 화면 구석에 점 하나가 보인다.

## 갱신

```
packaging/tenth/update.sh            # 둘 다(기본)
packaging/tenth/update.sh local      # 이 PC만
packaging/tenth/update.sh lab        # 연구실 PC만
packaging/tenth/update.sh --worktree # 커밋 안 한 변경까지 넣음(시험용)
```

- **빌드 대상**: 커밋된 HEAD로 빌드한다(`build.sh`). 커밋하지 않은 변경은 빌드에 들어가지 않고, 빌드할 때 그 목록을 경고로 보여 준다.
- **Godot 내보내기 템플릿**
  - 4.7.2 공식 릴리스 파일이고, SHA-512를 `build.sh`에 고정해 두었다.
  - 처음 한 번 받아 `game/.godot_bin/xdg_data/`(git 무시)에 리눅스 x86_64 두 개만 푼다.
- **실행 중이면 멈춤**: 앱이 열려 있으면 그 PC의 설치는 바꾸지 않고 멈춘다.
- **연구실 PC로 보내는 것**: 바뀐 파일만 rsync로 보낸다. 끊겨도 다시 돌리면 이어받는다.
  - 설치본(약 95 MB)
  - 서버 자료(약 20 MB): `lab_manifest.py` 목록 = 드라이버 체크포인트, 상자 배치, 지도, 중심선·주행선·MAP 표 캐시

## 설치 구성

| | 이 PC | 연구실 PC |
|---|---|---|
| 설치본 | `~/Applications/TENTH/` | 같음 |
| 서버 파이썬 | 저장소 `.venv` (torch 2.11) | `~/F1tenth/.venv` (torch 2.10) |
| 서버 GPU | `gpu: "1"` = 5060 (번호가 거꾸로) | `gpu: "0"` = 4060 Ti |
| 체크포인트·캐시 | 보통 홈(`~/f1sim_runs`, `~/.cache/f1sim`) | `~/F1tenth/tenth/` (서버의 HOME) |
| 결과·리플레이 | `/data/tenth/` | `~/F1tenth/tenth/data/` |
| 서버 메모리 상한 | 3G (앱 3G, 시작 전 가용 3 GB 필요) | 10G |

설치본 폴더에 들어 있는 것:
- 앱: `tenth.x86_64` + `tenth.pck`
- 실행기: `tenth-launch`
- 같은 커밋의 서버 코드 사본: `server/f1sim`
- 자료 사본: `library/`

**저장소를 직접 가리키지 않고 사본을 두는 이유**
- 이 PC의 작업 트리는 다른 작업자가 계속 고친다. 저장소를 가리키면 아이콘을 누를 때마다 다른 코드가 돈다.
- 그러면 고치다 만 파일에 앱이 깨질 수 있다.
- 앱(pck)과 서버의 통신 규격이 서로 다른 커밋에서 올 수도 있다.

**편집 설치를 끄는 이유**
- 서버는 `tenth_boot`를 거쳐 뜬다. 이것이 venv의 f1sim 편집 설치(`pip -e`)를 끈다.
- 편집 설치가 켜져 있으면, 사본에 없는 모듈을 체크아웃에서 조용히 가져온다.
  - 이 PC에서는 작업 트리에서 가져온다.
  - 연구실 PC에서는 옛 `~/F1tenth/F1tenth_E2E`에서 가져온다.

**torch와 venv는 사본으로 두지 않고 그 PC의 것을 쓴다.** 크기가 수 GB이고, GPU 드라이버에 맞아야 하기 때문이다.

**연구실 PC는 서버 HOME을 바꿔서 돌린다**
- 서버를 HOME=`~/F1tenth/tenth`로 돌린다.
- 옛 코드는 `~/.cache/f1sim`, `~/f1sim_runs`, `~/.f1sim/maps`를 경로에 박아 두고 쓴다. HOME을 바꾸면 이 경로들이 모두 새 폴더 안을 가리킨다.
- 그래서 옛 콘솔이 쓰는 폴더를 건드리지 않는다.
- 상자 배치의 절대 경로(`/data/compbench/banks`)는 설치할 때 `~/F1tenth/tenth/banks`로 바꾼다.

## 설정 파일

`~/.config/tenth/config.yaml` 한 곳에 둔다. 다시 설치해도 고친 값은 유지된다.

| 키 | 뜻 |
|---|---|
| `python` | 시뮬 서버를 돌릴 파이썬(torch가 있는 venv) |
| `device` | `cuda` 또는 `cpu` |
| `gpu` | 서버가 쓸 GPU(`CUDA_VISIBLE_DEVICES` 값) |
| `app_gpu_index` | 앱(Godot)이 그릴 GPU. 비우면 Godot이 고른다 |
| `library` | 코스·규칙·드라이버·물리 자료 위치 |
| `data` | 결과·리플레이 저장 위치 |
| `external` | 팀 지도 폴더 |
| `server_home` | 서버의 HOME. 체크포인트는 `<홈>/f1sim_runs`, 캐시는 `<홈>/.cache/f1sim`에서 찾는다 |
| `server_memory_max` / `app_memory_max` | systemd-run 메모리 상한 |
| `min_free_gb` | 시작 전에 필요한 가용 메모리 |
| `port` | 서버 포트. 쓰이고 있으면 다음 빈 번호를 쓴다 |

## 로그

`~/Applications/TENTH/logs/`에 쌓인다.
- `launcher.log`: 실행기 기록
- `server-<시각>.log`, `app-<시각>.log`: 최근 20개씩 남긴다
- `flow-*/`: 자동 확인 결과

## 확인 결과 (2026-11-29, commit 0a043a1)

설치본 실행기 `flow`로 메인 → 준비 → 주행 → 결과를 끝까지 밟았다.
- 조건: IFAC 2026, scan31b u475, 실차 방식, 60 s, 상자 3개(30 s에 치움)
- PNG 5장. Pretendard 글꼴과 3D가 정상으로 그려졌다.
  - 이 PC: `~/Applications/TENTH/logs/flow-local-final/`
  - 연구실 PC: 그 PC의 `~/Applications/TENTH/logs/flow-lab-final/`. 이 PC로 가져온 사본은 `~/Applications/TENTH/logs/lab/`에 있다.
- 앱 화면 갱신은 두 PC 모두 60 fps다.

| | 배속(주행 중 중앙값) | 서버 메모리 | 세션 준비 |
|---|---|---|---|
| 이 PC (5060) | 0.89~0.91× (4회) | 2.7 GB | 7~19 s |
| 연구실 PC (4060 Ti) | 0.91~0.93× (3회, 결과 평균 0.94~0.96×) | 2.2 GB | 7~14 s |
| 연구실 PC, ICCAS 2025 + hs43b | 0.94× | **4.0 GB** | 13 s |

## 알려진 문제

- **이 PC에서 ICCAS 2025 코스가 꺼진다**
  - 원인: 이 코스는 서버 메모리가 약 4 GB 든다. 이 PC의 상한은 3G라서 서버가 강제 종료된다(오류 창이 뜸).
  - 해결: 메모리 여유가 있을 때 `server_memory_max: 5G`로 올린다. 연구실 PC는 상한이 10G라 문제없다.
- **독립 검산(`flow`의 verify)이 설치본에서 돌지 않는다**
  - 원인: 커밋되지 않은 `f1sim/f1sim/compbench`가 필요하다.
  - 주행과 채점에는 영향이 없다. `rules.py`가 같은 규칙으로 대신 계산한다.
  - `compbench`가 커밋되면 다음 갱신부터 돈다.
- **Godot 릴리스 템플릿은 `--main-pack`과 `--path`를 막아 둔다**
  - 그래서 `flow`는 `run/capture_host/`에 실행 파일의 하드 링크를 만들고, 그 옆의 `override.cfg`로 8×8 창을 띄운다.
- **배속이 1.0×에 조금 못 미친다(0.9×)**
  - 채점은 시뮬 시간 기준이라 결과에는 영향이 없다.
  - 화면 오른쪽 위에 배속이 늘 표시된다.
