# 해외 대회 트랙 지도 조사 (2026-11-29)

방법: 공식 사이트(f1tenth.github.io/race.html, 대회별 *-race.roboracer.ai), github.com/f1tenth 조직 저장소 전수, AutoDRIVE 조직, GitHub 저장소 검색(topic f1tenth/roboracer + 이름·설명 검색, 약 1,560개 저장소 목록) 후 그중 50 MB 미만 약 1,100개의 파일 경로를 jsDelivr 파일 목록으로 대회 키워드(icra, iros, cdc, itsc, cps, atlanta, vienna 등) 검색, 의심 저장소는 raw 파일과 README 확인. 저장소 clone 없음, 시뮬 등록 없음, git commit 없음. 지도는 `/data/tenth/map_candidates/intl/`에 파일 단위로만 저장.

## 1. 대회 연표와 지도 공개 여부

연표 출처: 공식 목록 https://f1tenth.github.io/race.html (링크·날짜), 장소·회차는 확인된 것만 적음. 2024년 회차는 서베이 논문(arXiv 2506.15899) 표 IV, 그 밖에는 각 대회 사이트.

| 대회 (표기) | 날짜 · 장소 | 지도 공개 여부 |
|---|---|---|
| Embedded Systems Week 2016 (pittsburgh2016) | 2016-10-02~07, Pittsburgh | 미확인 (공식 목록에 페이지만) |
| Cyber Physical Systems Week 2018 (porto2018) | 2018-04-10~13, Porto | 미확인 |
| Embedded Systems Week 2018 (torino2018) | 2018-09-30~10-05, Torino | 미확인 |
| CPS-IoT Week 2019 (montreal2019) | 2019-04-15~18, Montreal | 미확인 |
| Columbia 2019 (columbia2019) | 2019-10-13~18, New York | 미확인 |
| IFAC 2020 (ifac2020) / IROS 2020 | 2020-07-12~17 Berlin(목록 표기) / 가상 | 미확인. IROS 2020 팀 저장소 다수는 f1tenth_gym 기본 맵만 사용 |
| IROS 2021 | 가상 대회 | 공식 f1tenth/iros_2021_logs는 주행 로그(jsonl)만, 지도 없음 |
| ICRA 2022 (icra2022, icra2022-race) | 미확인 | 공식 f1tenth/icra_2022_logs `resources/maps`에 있으나 전부 F1 서킷 축소판(Oschersleben, Silverstone, Sochi 등, 제외 대상). 실제 트랙 비슷한 지도는 팀 저장소에만(아래 B) |
| German Open 2022 (germany-race2022) | 미확인 | 못 찾음 |
| IV 2023, CPS 2023, ICRA 2023(London, ForzaETH 논문 기준), IROS 2023(Detroit, CMU 페이지 기준) | 미확인 | CPS 2023만 팀 저장소에서 발견(B). 나머지 못 찾음 |
| ICRA 2024 (Yokohama, 15th), CPS-IoT Week 2024 (Hong Kong, 17th), IV 2024 (Jeju, 18th), SM 2024 (Ontario, 19th), ITSC 2024 (Edmonton, 20th), IROS 2024 (Abu Dhabi, 21st), CDC 2024 (Milan, 22nd), BU 2024, Korea 2024 | 2024 | 못 찾음. 사이트 *.f1tenth.org는 이 PC에서 DNS 실패/인증서 불일치로 열리지 않음(미확인) |
| ICRA 2025 (24th, Atlanta, 2025-05-19~23) | | 주최 측 배포 여부 미확인(사이트 접속 불가). 팀 저장소에 지도 있음: 근거 A |
| CDC / IROS 2025 | | 못 찾음 |
| ICRA 2026 (27th, Vienna, 2026-06-01~05) | | **주최 측 공개**: icra2026-race 저장소 `track/` Classic·Master |
| IV 2026 (28th, Detroit, 2026-06-22~25) | | 주최 측 "Tentative Track Layout" Google Drive 폴더 링크만(내용 미확인). 팀 저장소 사본 1개(B) |
| IROS 2026 (Pittsburgh, 2026-09-27~30, 회차 표기는 페이지마다 달라 미확인) | | **주최 측 공개**: iros2026-race 저장소 `assets/maps` |
| IFAC 2026 (Busan) | | 한국 조사자 담당 |
| RoboRacer China 2026 | | 팀(TianRacer, 2위 기재) 저장소에 트랙 지도 있음(B). 주최 측 페이지는 못 찾음 |

AutoDRIVE RoboRacer Sim Racing League(ICRA 2025·2026, CDC-TF 2025, IROS 2024·2026, CDC 2024) 트랙은 `AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Racetracks`에 3D 모델(.fbx/.skp)과 렌더 PNG(`Library/SRL 20xx ... Track.png`)로만 있고 점유격자(yaml/pgm)는 없음. 실제 대회 지도가 아니라 가상 트랙(디지털 트윈)이라 받지 않음.

## 2. 찾은 지도 (근거 등급순)

크기 = 이미지 가로×세로(괄호는 트랙 범위). 폭은 free 영역 골격에서 잰 중앙값이며 RobeSafe README 수치가 있으면 병기. 중심선 길이 추정(골격)은 분기를 포함해 과대할 수 있음.

| 등급 | 대회 | 지도 | 성격 | 출처 · 라이선스 | 해상도 · 크기 · 폭 | 크기 타당성 | 저장 폴더 |
|---|---|---|---|---|---|---|---|
| A | ICRA 2026 Vienna | `icra2026_classic` | 주최 측 배포 | f1tenth/icra2026_race `track/` (커밋 d2f41555), 사이트 race_resources에도 링크 · 라이선스 미확정(GitHub "Other") | 0.019, 19.0x18.0 m, 폭 1.4~3.5 m(중앙 1.6), 길이 약 100+ m | 타당(해상도 0.019는 비표준이나 폭이 정상). 점선 벽 = 열린 경계 | `ICRA-2026_f1tenth-official/` |
| A | ICRA 2026 Vienna | `icra2026_master` | 주최 측 배포 | 위와 같음 | 0.019, 38.0x18.0 m, 폭 1.0~3.8 m | 타당. 중앙에 0.7 m 다리(격자에는 없음), 5갈래 교차 | 위와 같음 |
| A | IROS 2026 Pittsburgh | `IROS2026` | 주최 측 배포(바닥도면 설계, 벽 윤곽선만) | f1tenth/iros2026_race `assets/maps` (커밋 a9cdd791), 사이트 race_resources 링크. RobeSafe 사본과 바이트 동일 | 0.05, 51.8x31.6 m(트랙 약 46x28), 폭 약 2.2 m(1.9~2.9) | 타당. 윤곽선 지도라 바깥도 free, 채워서 써야 함. 길이 추정 약 200 m로 흔한 범위 상단 | `IROS-2026_f1tenth-official/` |
| A | ICRA 2025 Atlanta | `icra2025_original` (AuRoRus `Atlanta_2025`의 PNG 변환) | 대회 지도(AuRoRus가 import) | RobeSafe-UAH/rbsf-racing `tracks/icra2025` (커밋 fd2bb742), 원본 Noce99/AuRoRus `maps/Atlanta_2025`(커밋 5175016, tiff라 미수령) · Apache-2.0 | 0.05, 18.3x25.7 m, 폭 0.9~2.4 m(중앙 1.3), 길이 약 100 m | 타당(좁은 편) | `ICRA-2025_RobeSafe-AuRoRus-Atlanta/` |
| A | ICRA 2025 Atlanta | `icra_race_2` | 팀 SLAM | Shreyas0812/ICRA_Roboracer `maps/` (README "ICRA 2025에 쓴 노드") · 라이선스 없음 | 0.05, 38.6x23.2 m(미지 여백 포함), 폭 약 1.3 m | 위 지도와 같은 트랙(축척 동일하게 겹쳐 IoU 0.81) | `ICRA-2025_Shreyas0812-SLAM/` |
| B | IV 2026 Detroit | `iv2026_original` (IV_2026_SIM) | 주최 측 시뮬 지도 사본 | RobeSafe-UAH/rbsf-racing `tracks/iv2026` (커밋 2d0d9851) · Apache-2.0 | 0.0625, 97.6x75.7 m(트랙 약 31x34), 폭 1.0~1.9 m | 타당. 1픽셀 벽선이라 대각선으로 free가 샘(RobeSafe가 처리법 기록), 점 15개 잡음 | `IV-2026_RobeSafe/` |
| B | CPS-IoT Week 2023 | `cps_2023` | 팀 SLAM | TUM-AVS/RoboRacer-Evaluation `particle_filter/maps`, vikasdotvivek/f1tenth 사본 · 라이선스 없음 | 0.1, 28.7x31.9 m, 폭 약 1.4 m | 타당. 대회 장소·날짜는 저장소 글로 확인 못함(파일명만) | `CPS-IoT-Week-2023_TUM-AVS/` |
| B | ICRA 2026 (Classic 배치) | `ICRA_1` | 팀 SLAM 스캔(콘 포함) | JacobBau04/RoboRacer `mppi_bringup/maps` · MIT | 0.05, 18.4x18.9 m, 폭 약 1.4 m | 타당. 대회장에서 만든 것인지 복제 트랙인지 미확인 | `ICRA-2026_JacobBau04-classic-scan/` |
| B | RoboRacer China 2026 | `point_lio_competition_track` | 팀(TianRacer) 지도, 2025 트랙 모델에서 | npu-ius-lab/Roboracer_China_2026 `simulation_ws/.../roboracer_gazebo/config` · 모델 GPL-3.0 | 0.05, 30.8x20.2 m(트랙 약 23x14), 폭 약 1.4 m | 타당. ICRA 2026 Classic과 같은 계열 배치(점선 경계, 음표 섬). 공식 대회명은 저장소 표기 그대로, 주최 측 확인 못함 | `RoboRacer-China-2026_npu-ius-lab/` |
| B | ICRA 2022 | `icra_2`, `icra` | 팀 지도 | zzjun725/f1tenth-racing-stack-ICRA22 `maps/` (README "ICRA'22 대회에 쓴 코드"), 복사본 K-Jyotiraditya/F1Tenth 등 · 라이선스 없음 | 0.05, 22.3x17.4 m, 폭 약 1.7 m(`icra_2`). `icra`는 74x34 m 윤곽선 그림으로 축척 의심 | `icra_2` 타당, `icra` 의심(표시) | `ICRA-2022_zzjun725/` |
| B | ICRA (연도 미상) | `icra`, `icra_ext` | 팀 지도 | HiPeRT/unimore_roboracer `src/maps/maps/icra` | 0.05, 35.7x16.5 m, 폭 약 1.7 m | 타당. 모양이 위 ICRA 2022 `icra_2`와 비슷(IoU 0.63)이라 같은 트랙일 가능성. `icra_ext`는 바깥을 free로 채운 변형 | `ICRA-unknown-year_HiPeRT-Modena/` |

각 폴더에 `source.yaml`(위 항목) 저장. 정본 후보 권고: ICRA 2026은 주최 측 `icra2026_classic/master`, IROS 2026은 `IROS2026`(윤곽선을 채워 쓸 것), ICRA 2025는 `icra_race_2`(SLAM, 실제 센서가 본 모양) + `icra2025_original`(깨끗한 격자) 쌍. 같은 대회의 복수 판은 RobeSafe-UAH/rbsf-racing의 `docs/MAP_CATALOG.md`와 `scripts/tracks/drawn_track.py`가 윤곽선 지도 처리법을 이미 갖고 있음(참고만, 받지 않음).

갤러리: `/home/shchon11/F1tenth_E2E/docs/pm/maps-search-2026-11-29/INTL_gallery.png` (13장, 대회명·크기·등급 표기). 열어 본 결과 깨진 파일 없음, 건물 복도형 없음. 주의: `ICRA-2022_zzjun725/icra`(윤곽선, 축척 의심), `icra_ext`(바깥 free)는 그대로 쓰기 부적합.

## 3. 찾았지만 받지 않은 것

| 대상 | 이유 |
|---|---|
| Noce99/AuRoRus `maps/Atlanta_2025`, `Vienna_2026_Quali`, `Vienna_2026_Final_Ground`, `Vienna_2026_Final_Bridge` (map.tiff + info.json, 0.05 m, Quali 655x565 px = 32.7x28.3 m) | tiff는 받지 말라는 형식 제약. **Vienna 2026 예선(Quali) 지도는 주최 측 저장소에 없고 이 저장소에만 있음**, 변환 허락 필요 |
| f1tenth/f1tenth_racetracks 25개, f1tenth/icra_2022_logs `resources/maps`, f1tenth_gym 기본 맵(berlin, vegas 등) | 실제 F1 서킷 축소판·시뮬 기본 맵(제외 대상) |
| DCAM1/F1Tenth_Simulator, mini0909-web/2025-F1TENTH-Racing-Simulator의 `icra2025_map.pgm` | Atlanta와 같은 파일(원점이 Shreyas0812와 동일)인데 해상도를 0.07로 잘못 적어 1.4배 큼. README는 "iccas2025"라고 해 한국 대회와 혼동 |
| F1Tenth-UCF/cavrel_racer `icra2025.png`(0.0625, 27x10 m 윤곽선) | 모양이 Atlanta와 다르고 근거 없음(C) |
| vaithak/f1tenth-icra-race `maps/race*`(README: ICRA 2025 5위) | 파일명이 대회가 아니라 race 번호, Atlanta 트랙과 IoU 0.4로 다름(대회장 연습 공간으로 추정, C). 받았다가 삭제 |
| architnh/f1tenth_icra2022 `maps/0424_clean` 등 | L자 방 실내 지도(연습 공간), 대회 트랙 아님(C). 받았다가 삭제 |
| f1tenth-class/race-2, ImAlko·li51-AMX·ronchuxia의 `race2/race3`, navidmdn/f1tenth_rl `race1~4` | 수업·가상 레이스 지도, 대회명 근거 없음(C) |
| eungyun-im/Roboracer, SIGRobotics-UIUC/roboracer의 `icra26_track`, `iros26_track`, fixtures | 공식 지도의 가공본(파일명만 확인), 원본이 이미 A |
| IFAC 2026 `ifac_roboracer`(hyunyoung1018, xg6144 등), woongeyaaa320 `competition_8x22`, columncat `pnu_champ` | 한국 대회, 다른 조사자 담당 |
| TUM-AVS `cps_2023_old` | `cps_2023`과 같은 크기의 이전 판 |

## 4. 지도가 공개되지 않은 대회와 요청처

2016~2024년 모든 대회(IROS 2023 Detroit, ICRA 2023 London, ICRA 2024 Yokohama, IROS 2024 Abu Dhabi, CDC 2024 Milan, ITSC 2024, CPS-IoT Week 2024, German Open 2022 등), ICRA 2025 Atlanta의 주최 측 파일, CDC/IROS 2025. 공식 *.f1tenth.org 사이트가 열리지 않아 "비공개"가 아니라 "미확인"이다.

요청처 후보: 주최 측 contact@roboracer.ai 와 RoboRacer Slack; 각 대회 사이트를 만든 저장소(icra2026_race 등 f1tenth 조직 저장소처럼 `track/` 폴더를 두는 방식이 2026년부터 보임); 상위권 팀 ForzaETH(ICRA 2023 우승, 2025 3위), UniBo Motorsport·UNICORN(ICRA 2025 1·2위), UPenn BitShifters, Scuderia Segfault(TU Wien, ICRA 2024 4위·2025 참가), TUM AVS. IV 2026은 주최 측 Drive 폴더(사이트 race_resources 링크)에 원본이 있을 가능성이 큼.

## 5. 한계

- GitHub 인증 없는 API 한도(시간당 60회)를 소진해 50 MB 이상 저장소 약 180개(예: Brojoon-10/ICRA2026_HJ, hee4040/ssupath-f1tenth-race-stack, uci-f1tenth/UCI_race_stack, Tejendra00/F1tenth_ICRA-24, jetag/IROS_f1tenth)는 파일 목록을 못 봄. npu-ius-lab/Roboracer_China_2026은 수동 확인.
- 검색은 파일 경로의 대회 키워드 일치 기반이라 이름이 일반적인(`map1`, `track`) 지도는 놓칠 수 있음. 대회 장소·회차는 확인한 것만 기재.
- 라이선스는 저장소 선언 기준이며 LICENSE 본문은 읽지 않음(주최 측 저장소는 GitHub이 "Other"로 표시).
