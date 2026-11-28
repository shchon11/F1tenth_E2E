# 한국(대회) 트랙 지도 조사 (2026-11-29)

방법: 웹 검색, 공식 사이트(roboracer.or.kr, korea-race*.f1tenth.org), GitHub 저장소 목록·raw 파일. 저장소 통째 clone 없음. 시뮬 등록 없음, git commit 없음. 일본·대만·중국 지역전은 한국 밖이라 검색했으나 지도를 찾지 못함(미확인).

## 1. 대회 연표와 지도 공개 여부 (표기는 공식 그대로)

| 대회 | 날짜 · 장소 | 주최 측 지도 배포 | 지도 공개(찾은 것) |
|---|---|---|---|
| 1st F1TENTH Korea Championship (KSMTE 2022 Autumn Conference) | 2022-12-12~13, 제주 | 규정: 트랙은 사전 공개, 전용 매핑 시간 없음 | 못 찾음 |
| 2nd F1TENTH Korea Championship (ICCAS 2023) | 2023-10-17~19, 여수 | 미확인 | 못 찾음. KwonYeonghun/F1TENTH-2023-Archive는 발표·사진·영상만 있음(지도 없음) |
| 3rd F1TENTH Korea Championship (ICCAS 2024) | 2024-10-29~11-01, 제주신화월드 | 배포 안 함. 규정: 트랙 약 25x10 m, 공기관(직경 50 cm) 경계, 팀별 매핑 3~5분 | 못 찾음. HMCL-UNIST/F1tenthKorea2024는 웹사이트뿐 |
| 2025 F1TENTH PNU-HYUNSONG Championship | 2025-01-22~23, 부산대 기계관 | 미확인 | **있음**: columncat/F1Tenth-Racer `maps/pnu_champ` |
| 4th F1TENTH Korea Championship (ICCAS 2025) | 2025-11-04~07, 송도 ConvensiA | 미확인 | KORA_K3 `real`(이미 등록). 우리 팀 비공개 저장소 nov4(축척 작음, 이 PC에 없음) |
| 2025 F1TENTH Global Innovation Festa G-NEX Championship | 2025-11-30~12-01, 창원컨벤션센터 | 미확인 | 못 찾음 |
| 2026 2nd F1TENTH PNU-HYUNSONG Championship (roboracer.or.kr 표기) | 2026-01-25~27, BEXCO | 미확인 | 못 찾음 |
| 5th RoboRacer Korea Championship at IFAC 2026 | 2026-08-24~27, BEXCO 부산 | 사이트에 지도 없음 | **있음**: Sangbum-K/IFAC_2026, tkddn647-ship-it/2027_F1tenth_test `ifac_roboracer`(동일 파일) 외 |
| 2026 RoboRacer G-NEX Championship | 2026-11-01~02, 창원 CECO(기사: 인제대 연계) | 미확인 | 못 찾음 |

주의: "2026 PNU-HYUNSONG"은 기사에 BEXCO 2026-01-25~27(16팀)로 나오고 공식 이름은 사이트마다 다르다(roboracer.or.kr: "2026 2nd F1TENTH PNU-HYUNSONG Championship", pnu26.roboracer.or.kr: "2026 RoboRacer PNU-HYUNSONG CHAMPIONSHIP").

## 2. 찾은 지도 (근거 등급: A 명시 근거 복수, B 단일 출처 대회명, C 추정)

| 대회 | 지도 | 출처 | 해상도 · 크기 · 폭 | 축척 신뢰도 | 등급 | 저장 |
|---|---|---|---|---|---|---|
| IFAC 2026 | `ifac_roboracer` | Sangbum-K/IFAC_2026 `maps/`, tkddn647-ship-it/2027_F1tenth_test `maps/` (두 저장소 파일이 바이트 동일, md5 b17aeedc...) | 0.025, 22.8x9.8 m, 면적 99 m2, 폭 ~1.9 m | 높음: 우리 대회 bag /map과 IoU 0.95(메모 11-02), 면적 99 vs 98 m2 | A | `/data/tenth/map_candidates/korea/IFAC-2026_Sangbum-K/`, `IFAC-2026_tkddn647/` |
| IFAC 2026 | 로컬 bag 스캔 `korea26_scan`(+ bag /map) | `~/.f1sim/maps/korea26_scan/`, `real_data/01_competition_*` | 0.05, 9.8x24.5 m | 실측 1.00 확인 | A (기존 등록) | 복사 안 함 |
| IFAC 2026 | `my_track0824_realfinal` | Sangbum-K/IFAC_2026 `maps/` | 0.05, 19.9x20.7 m | 낮음(팀 SLAM, 8-24 연습일. 모양이 최종 레이스 지도와 다름) | B | 위와 같음 |
| PNU-HYUNSONG 2025 | `pnu_champ` | columncat/F1Tenth-Racer `maps/pnu_champ.{yaml,pgm}` (부산대 Jeoktoma 팀, README에 1위 기재) | 0.05, 11.0x18.6 m, 면적 137 m2, 폭 ~1.8 m | 미확인(같은 대회 다른 팀 지도 없음. 팀 SLAM이면 20~25 % 작을 수 있음) | B | `.../PNU-HYUNSONG-2025_columncat-jeoktoma/` |
| ICCAS 2025 | `real` (KORA_K3) | `external/korea_teams/KORA_K3/src/kora_k3/maps/real.*` (커밋 2025-11-04 race_map_update_again) | 0.05, 24.2x24.8 m(홀 전체) | 사용자 확인("실제가 좀 컸어") | A(대회 당일 커밋 + 사용자 확인) | 이미 등록 |
| 미상 | `roboracer_0817` | tkddn647 저장소 | 0.05, 21.3x8.0 m | - | C | 같은 폴더(참고용) |

IFAC 2026 `ifac_roboracer`가 주최 배포본인지 팀 제작인지는 파일만으로 못 가렸다. 두 팀이 같은 파일을 쓰고 `.pbstream`·centerline이 함께 있어 팀 공유 또는 배포본일 가능성이 있다(추정). 정본 후보: IFAC 2026은 해상도 0.025인 `ifac_roboracer`(또는 기존 `korea26_scan`)로 교차 확인, PNU 2025는 `pnu_champ` 하나뿐.

## 3. 로컬 `external/korea_teams` 전수

| 폴더 | 지도 | 판정 |
|---|---|---|
| KORA_K3 | `real`(ICCAS 2025, 등록됨) | A |
| KORA_K3 | f3~f11, f_1, f_2, test1, 6th_floor (0.05, 팀 연습 SLAM), `columbia`(path_planning 예제) | C 연습·예제 |
| project_The_4th_F1TENTH_Korea_championship | `maps/` 18개: 0625, 0626-2/3/4, 07734, 077111114, 0807, 08072, 08073, 0810, 08102, 08103, 0811, 0901, 0902, 0909_01, 0911, 0924. 날짜가 2025-06~09로 ICCAS 2025(11월) 이전 | C 연습 SLAM. 대회 당일 지도 아님. (0625는 99x99 m로 거의 빈 지도) |
| 같은 폴더 | `MAP_ws/.../maps`의 berlin, f, porto, overtake_map, hangar | 대회 지도 아님(시뮬 예제) |
| F1tenthKorea2024 | 웹사이트 소스. 지도 없음 | - |
| f1tenth_tada | 실습 코드. 지도 없음 | - |

등록 안 된 채 로컬에 있던 대회 지도: 없음. 4th 저장소 지도는 모두 연습일 것으로 추정(C).

## 4. 아직 못 구한 것과 얻을 곳
- PNU-HYUNSONG 2026, G-NEX 2025/2026, ICCAS 2023/2024, KSMTE 2022: 지도 공개 없음. 연락처(공개 페이지 기재): RoboRacer Korea roboracer.korea@gmail.com, f1tenth.korea@gmail.com (https://www.roboracer.or.kr/race/events/2025_3/ ), PNU RISE 미래모빌리티UIC 051-510-7116 (https://pnu26.roboracer.or.kr/ ), IFAC 2026 UNIST ryujs@unist.ac.kr (https://2026ifac-roboracer.com ). 출전팀 저장소에서 `maps/`를 찾는 것이 더 빠를 수 있다.
- magp(sssamsssong/magp)의 `src/race_stack/maps/`에 08-08~08-24(IFAC 직전·당일 추정), 11월 지도 약 70개가 있고 README에 `bexco_track` 예가 있다. 어느 것이 대회 트랙인지 명시가 없어 C로 두고 받지 않았다. 0824-*, comp26_B가 후보.
- 저장소 `columncat/F1Tenth-Racer`와 비슷한 팀 저장소 검색은 GitHub 검색 API 한도(분당 10건, 시간당 60건)에 막혀 일부만 했다. 2024 제주 대회 팀 저장소는 찾지 못함.
- 일본·대만·중국 지역전: 지도 못 찾음(대회 자체 미확인). NTU-Deepspeed/Korea_2025(대만 NTU 팀의 ICCAS 2025 참가)는 저장소가 비어 있음.
- 빈 저장소(지도 없음): NTU-Deepspeed/Korea_2025, kingminkyo/F1TENTH_Project, ujin99·2-ujin2/ICCAS2025_F1Tenth.
- 우리 팀 비공개 저장소 antelcore/2025ICCAS_F1Tenth의 nov4(ICCAS 2025 당일)는 이번에 접근하지 않았다(축척 작음 메모).

## 5. 갤러리
`/home/shchon11/F1tenth_E2E/docs/pm/maps-search-2026-11-29/korea_gallery.png` (12개: 로컬 6 + 신규 6; 지도마다 대회명, 크기, 면적, 폭, 등급). 직접 열어 확인: 모양이 정상이다. 주의점은 (1) KORA `real`은 홀 전체가 free라 면적 587 m2로 트랙 면적이 아님, (2) `ifac_roboracer`는 `korea26_scan`을 90도 돌린 모양과 일치, (3) 0625는 거의 빈 지도, (4) `0807`은 레이 부채꼴 잔상이 큰 불완전 SLAM.
