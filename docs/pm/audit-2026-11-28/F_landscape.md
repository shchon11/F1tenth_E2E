# F. 외부 동향 조사 (2024~2026): RoboRacer 학습 기반 주행과 시뮬레이터

작성일 2026-11-28. 웹 검색과 논문 초록/본문(arXiv HTML) 확인 기준. 확인하지 못한 항목은 "미확인"으로 표기했습니다. 검색은 일반 검색어만 사용했습니다.

## 1. 대회

### 1.1 규정 (가장 최근에 확인한 원문 = ICRA 2025, 24회, 2025-02-03판)
출처: [roboracer_rules (GitHub)](https://github.com/f1tenth/roboracer_rules). 2026판 원문은 미확인(규정은 대회마다 바뀐다고 명시됨).

| 항목 | 내용 |
|---|---|
| 예선(Time Trial) | 5분 2개 히트. 점수 = 최고 랩타임 + 무충돌 연속 랩 수(동점은 랩 많은 팀). 충돌 후 트랙 복구에 걸린 시간이 곧 벌칙(타이머 안 멈춤). 장애물(배리어) 5 cm 이하 이동은 무벌칙 |
| 장애물 | 정적/동적 장애물 최대 35x32x30 cm(LiDAR가 보는 골판지 등). 멈추지 않고 통과해야 통과. 실패 팀은 1대1에서 추월 금지(추월하면 실격, 상대 충돌 시 제외) |
| 본선(Head-to-Head) | 2대 동시 주행, 더블 일리미네이션(예선 시드) 후 Final Four 단판. 한 세트 3판 2선승, 한 판은 20바퀴 먼저 도는 차가 승 |
| 출발 | 예선 출발선에서 30 cm 간격 나란히. 1판 상위 시드가 좌우 선택, 2판 교대, 3판 동전 |
| 차량 충돌 | 가벼운 접촉 무벌칙. 과실 차가 사고 지점에서 2 m 뒤에서 재출발(맞은 차가 멈췄다면 3 m). 고의/회피 시도 없는 충돌은 경고, 3회 실격 |
| 비고 | 사용자 팀이 쓰는 "무충돌 연속 랩 + 최고 랩" 예선은 이 규정의 Time Trial 점수와 일치. 출발 방식은 "나란히 30 cm"로, 팀 메모의 "반대편 출발"과 다름(대회마다 다를 수 있어 한국 대회 규정 확인 필요) |

### 1.2 2025~2026 대회와 결과

| 대회 | 형식/특징 | 결과·접근 | 링크 |
|---|---|---|---|
| ICRA 2025 Atlanta (24회) | 위 규정 | 2위 UNIST UNICORN Racing. 1위 팀 미확인. 접근 상세 미확인(포스터/영상 링크만 있음) | [UNICORN 글](https://hmcl-unist.github.io/unicorn-racing/posts/2025-icra-roboracer-2nd-place-en/) |
| ICRA 2026 Vienna (27회, 약 30~37팀) | 시간주행, Classic Cup(1대1 더블 일리미네이션), Master Cup(다리/경사 포함 새 트랙, 매핑·학습 시간 매우 짧음) | Classic 1위 VAUL 2(캐나다), 2위 Brake Check Buddies, 3위 tron racing. Master Cup 1위 UNICORN_Racing(한국), 2위 UBM-Atlas, 3위 UBM-Tom. LAMARRacing(본)은 "Best Performance Overall", 시간주행 전체 최고 속도, 1대1 4위 | [TU Wien 결과](https://www.tuwien.at/inf/scuderia-segfault/news-detail/27th-roboracer-autonomous-racing-competition/), [UNIST 보도](https://www.kpinews.kr/newsView/1065574828996021), [Lamarr](https://lamarr-institute.org/?p=37542) |
| ICRA 2026 Sim League (AutoDRIVE) | 컨테이너 제출, 예선 = 연습 트랙 10랩 무충돌, 본선 = 미공개 트랙 시간 경쟁(충돌당 +10 s로 추정) | 1위 MonacoF1 110.27 s | [대회 페이지](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-icra-2026) |
| 30회 VTC2026-Fall (Boston, 9/6~9) | 예선 시간주행 시드, 본선 1대1 토너먼트 | 결과 미확인 | [공지](https://vtsociety.org/post/announcement/30th-roboracer-autonomous-racing-competition-vtc2026-fall) |
| ICRA 2027 Seoul (5/24~28, COEX) | 대회 목록에 RoboRacer 공지 미확인. "RoboRacer Korea" 규정도 미확인 | 미확인 | [ICRA 2027](https://2027.ieee-icra.org/?p=61) |

핵심 관찰
- 상위권 스택은 모듈형이다. UNIST는 3D LiDAR(유일)로 3D 위치추정+경로계획+제어를 조합, 학습 사용 여부는 보도에 없음(미확인). LAMARRacing은 F1 주행 데이터로 학습한 초기 궤적으로 최소시간 궤적 최적화를 가속(학습은 최적화의 초기값으로 쓰임) ([arXiv 2603.07126, 2026](https://arxiv.org/abs/2603.07126)).
- E2E/순수 RL 정책이 ICRA 상위권에 오른 사례는 찾지 못함(미확인). TinyLidarNet(원시 LiDAR 1D-CNN, 모방학습 계열로 추정, 확인 안 됨)이 F1TENTH Grand Prix 3위였지만 2024년 IROS 논문 기준이며, 저자도 "E2E는 실제 대회에서 성능 문제로 인기 없다"고 서술 ([arXiv 2410.07447, 2024](https://arxiv.org/abs/2410.07447)).
- ForzaETH race stack(ICRA 2023 1위 등)은 공개 모듈형 스택이다. 2025~26 학습 후속은 못 찾음 ([GitHub](https://github.com/ForzaETH/race_stack), [arXiv 2403.11784](https://arxiv.org/abs/2403.11784)). 2026 다른 상위 사례 AROLA(모듈 ROS 2 구조)는 [arXiv 2602.02730](https://arxiv.org/abs/2602.02730).

## 2. 학습 기반 레이싱: 실차 검증 사례

| # | 논문 (연도) | 관측 | 출력 | 알고리즘 | 경험량 | sim2real | 실차 성능 |
|---|---|---|---|---|---|---|---|
| 1 | Czechmanowski 외, [On learning racing policies with RL, IROS 2025](https://arxiv.org/abs/2504.02420) | 상태(속도, 요레이트, Frenet 오차, 앞 트랙 곡률/폭). LiDAR 아님 | 조향 직접 + 가속도(적분해 속도지령) | PPO, 병렬 400 env | 약 1.2억 스텝 | MF6.1 타이어, 1차 액추에이터 모델(제거 시 최고랩 5.56→6.43 s), 36분 주행 데이터로 식별, 마찰만 2% 무작위화(전 파라미터 무작위화는 오히려 악화) | 17 m L트랙: RL 최고 5.561 s vs MPC 5.639 s vs 인간 전문가 5.847 s (제로샷) |
| 2 | Ghignone 외 RLPP, [ICRA 2025](https://arxiv.org/abs/2501.17311) | 상태+앞 기준선/경계 20점 | Pure Pursuit 위에 잔차(조향, 속도), 실차 스케일 0.55 | SAC | 200만 스텝, 약 4시간(RTX 3090) | sim-실차 랩타임 격차 2.12% (PP 4.21%, 순수 RL TC-Driver 17.51%) | 최소랩 13.37 s vs PP 14.28 s, 그러나 MAP 12.60 s, MPC 12.75 s보다 느림(모듈형 최상위는 못 넘음) |
| 3 | Trumpp 외 α-RPO, [arXiv 2603.12960, 2026](https://arxiv.org/abs/2603.12960) | 센서 | 잔차 정책(기반 정책 영향을 점차 감쇠, 최종은 독립 신경망) | PPO 기반, 기반 정책이 배포 시 없는 센서(특권) 사용 가능 | 미확인 | RoboRacer 제로샷 | 기준선 대비 개선(수치 미확인) |
| 4 | Steiner 외, [Accelerating Real-World Overtaking, arXiv 2510.26040, 2025](https://arxiv.org/abs/2510.26040) | 속도, 조향, 10점 평균 필터 LiDAR | 선속도 0~2 m/s + 조향 직접 | TD3 | 14만 스텝(약 12만에서 포화), Follow-the-Gap 4대 상대, 트랙 24개 | 최대속도 2→1.5 m/s로만 낮춰 제로샷(지연 모델링 미흡) | 추월 성공 87% vs 경주만 학습 56% vs FTG 44% (상대 0.75 m/s로 느림). 단독 랩타임은 FTG보다 느림 |
| 5 | Xu 외, [Vision-based policy learning for high-speed racing, IEEE RAM 2026](https://ramagazine.ieee.org/2026/04/17/vision-based-policy-learning-for-high-speed-autonomous-racing-a-two-phase-learning-paradigm/) | 학생 = 깊이 카메라만, 교사 = 트랙 특권정보 | 미확인 | RL 교사 -> 증류 학생 | 미확인 | 1/10 실차 검증 | 정량 수치 미확인(초록만 확인) |
| 6 | Dikici 외, [On-track sysID in under a minute, RA-L 2025](https://arxiv.org/abs/2411.17508) | 30초 주행 데이터 | Pacejka 타이어 파라미터 | 잔차 MLP(58 파라미터)+최소제곱 | 학습 약 3초 | 실차 NUC | 랩 8.10 s vs 8.16 s, 경질->연질 타이어 전환 약 1초 적응(시스템 식별은 학습 정책이 아니라 시뮬 정확도 도구) |

참고(실차 미확인 또는 시뮬 위주): End2Race(모방학습, 1대1, 시뮬 위주로 보임, [arXiv 2509.16894](https://arxiv.org/abs/2509.16894)), 2023 end-to-end LiDAR RL ([arXiv 2309.00296](https://arxiv.org/abs/2309.00296), 2023이라 범위 밖), Context-aware MBRL(시뮬만, [arXiv 2510.11501](https://arxiv.org/abs/2510.11501)), Sivashangaran 외 깊이 기반 RL ([arXiv 2604.09499](https://arxiv.org/abs/2604.09499), 축소 하드웨어, 세부 미확인), Cihlar 외 RL 추월+상대 자세추정([arXiv 2603.27207](https://arxiv.org/abs/2603.27207)). UPenn xLab, TUM의 F1TENTH RL 계열은 이번 조사에서 직접 확인하지 못함(미확인). TUM의 A2RL 실차 우승 논문은 모듈형 ([arXiv 2602.08571](https://arxiv.org/abs/2602.08571)).

관찰: 실차에서 통한 사례는 거의 모두 (a) 상태 기반 관측(원시 LiDAR 아님) 또는 (b) 고전 제어기 위의 잔차, (c) 강한 액추에이터/타이어 모델이 있는 시뮬이다. 원시 LiDAR -> 궤적 출력 E2E가 모듈형을 이긴 실차 사례는 못 찾았다.

## 3. 상대 차가 있는 레이싱 학습

| 사례 | 무엇이 통했나 | 경험량/보상 | 링크 |
|---|---|---|---|
| UZH 리그 학습, 다중 쿼드로터 (2026) | 자기 체크포인트 이력(75%) + 고정 리그 20개 정책(단독 정책 4 + IPPO 16). 상대 수/순서 불변 Perceiver 어텐션 인코더 필수(없으면 단독 정책보다 나쁨). 리그가 단독 대비 충돌 -50%(크래시율 78%->23%). 인간 챔피언 대비 1대1 완주율 100% vs 53% | 약 2억 스텝, RTX 4090 1장 약 27시간. 보상 = 게이트 진행 + 순위 + 상대 근접 항 + 충돌 종료(상호 충돌 10%는 비종료) | [arXiv 2605.22748](https://arxiv.org/abs/2605.22748) |
| UPenn, 경쟁 멀티에이전트 (2025) | IPPO, 희소 승리 보상(게이트 통과 선두 +10/후미 +5, 랩 완주 선두 +50)만으로 저수준 제어 학습, 밀집 진행 보상 단독 정책보다 실기 전이가 좋음 | 표본 수 미보고. 실기는 경기당 3판이라 통계 약함 | [arXiv 2512.11781](https://arxiv.org/abs/2512.11781) |
| Sony GT Sophy 계열 | 비전 기반 GT7 에이전트: QR-SAC + GRU 액터, 비대칭 크리틱(상대 7+7대 격자와 트랙 에지는 크리틱만 봄; 대칭 변형은 거의 1위 못 함), 0/1/2/3/4/7/12/19대 상대 혼합 시나리오, 상대 출력 +-25% 무작위, 보상은 진행 + 충돌/추월 항 | 롤아웃 워커 20개(PS4), 20000 에폭, 버퍼 500만. 자동 보상 설계(LLM+VLM) 후속도 있음 | [arXiv 2504.09021](https://arxiv.org/abs/2504.09021), [arXiv 2511.02094](https://arxiv.org/abs/2511.02094) |
| F1TENTH 1대1 RL | 추월 보상 없이 상대 4대(느린 FTG) 앞에서 진행 보상 + 근접/충돌 페널티만으로 87% 추월(실차) | 14만 스텝 | [arXiv 2510.26040](https://arxiv.org/abs/2510.26040) |
| 모델기반 형태 | 상대 행동을 맥락(context)으로 두는 MBRL이 분포 밖 상대에 더 일반화(시뮬) | - | [arXiv 2510.11501](https://arxiv.org/abs/2510.11501) |
| 게임이론/MPC 계열 | 위상적 간격 선택 + 가속 MPC, 밀집 병목 추월 성공 81%+ (F1TENTH 실험, 실차 여부 미확인) | 학습 없음 | [arXiv 2603.09188](https://arxiv.org/abs/2603.09188) |

1대1 학습의 핵심 교훈
1. 단일 상대/자기 자신과만 학습하면 과적합한다(IPPO 단독은 자기 상대로 완주율 약 60%, 진짜 이질적 상대에서 실패). 이력 + 고정 리그 혼합이 필요.
2. 상대 관측은 순서/수 불변 인코더로(어텐션). 없으면 단독 정책보다 나쁨.
3. 보상은 단순하게: 승리/순위 희소 보상(UPenn) 또는 진행 + 소수 페널티(UZH, GT Sophy). 정교한 추월 보상은 성능을 해친다는 보고가 있음(Steiner).
4. 크리틱만 특권정보(상대 상태, 트랙 전체)를 보는 비대칭 구조가 GT Sophy 후속에서 명확히 이득.
5. 상대 다양성(수 0~19대, 성능 +-25%)과 충돌 일부 비종료 처리(UZH 10%)로 접촉 학습 신호를 확보.
6. 경험량은 단독 주행 상태 기반 RL이 약 1억 스텝, 1대1/리그는 약 2억 스텝 + GPU 1장 하루 규모가 기준점.

## 4. 인접 분야에서 가져올 것

| 아이디어 | 출처 | 1/10 레이싱 적용 가능성 |
|---|---|---|
| 단순 보상(경로 진행 x 위반 시 곱셈 감점/종료) + 큰 미니배치 PPO 스케일(CARLA 3억, nuPlan 5억 샘플, 8-GPU 1노드) | [CaRL, CoRL 2025](https://arxiv.org/abs/2504.17838), [코드](https://github.com/autonomousvision/CaRL) | 높음: 보상 항 누적 대신 단순화, 배치 크기와 보상 설계의 상호작용 |
| 궤적 anchor + 절단 diffusion + RL 파인튜닝(GRPO류) | [DiffusionDriveV2, 2025](https://arxiv.org/abs/2512.07745), [DriveAnchor, 2026](https://arxiv.org/abs/2606.00519) | 중간: 속도 프로파일 포함 anchor 어휘 + RL로 모드 붕괴 방지. 레이싱 적용 사례는 미확인 |
| 특권 교사 -> 센서 학생, 감쇠 잔차(교사의 영향을 점차 0으로) | [α-RPO 2026](https://arxiv.org/abs/2603.12960), [RAM 2026](https://ramagazine.ieee.org/2026/04/17/vision-based-policy-learning-for-high-speed-autonomous-racing-a-two-phase-learning-paradigm/) | 높음: 처음부터 PPO 대신 부트스트랩 후 독립 정책 |
| 비대칭 액터-크리틱 | GT Sophy 비전 에이전트(위) | 높음, 구현 비용 낮음 |
| 액추에이터/타이어 식별 + 마찰만 약하게 무작위화 | 사례 1, 6, [Wheeled Lab](https://arxiv.org/abs/2502.07380)(드리프트 마찰 중심값 0.4 실측, 0.2~0.8 넓은 범위는 실패) | 높음: 넓은 무작위화는 오히려 느려짐 |
| 대규모 병렬 GPU 시뮬 + 소형 온보드 학생 | Isaac Lab(Wheeled Lab 1024 env), UZH 144 env x 4기 | 높음 |
| 다리 로봇 RMA류 적응 모듈 | 이번 조사에서 1/10 레이싱 적용 사례 미확인 | 미확인 |

## 5. 시뮬레이터와 도구

| 도구 | 물리 | 병렬 | 센서 | UI/편집 | ROS 2 | 라이선스 | 링크 |
|---|---|---|---|---|---|---|---|
| f1tenth_gym (원본 후속) | 단일 트랙 모델 + Pacejka | CPU 위주 | 2D LiDAR | 최소 | 예(브리지) | 미확인 | [GitHub](https://github.com/f1tenth/f1tenth_gym) |
| f1tenth_gym_jax | 운동/동역학 단일 트랙, 저속 분기 | jit/vmap, 다차 동시 스텝, 결정적 PRNG | LiDAR 항목 문서에서 미확인 | 트랙 유틸, 지도 캐시 | 미확인 | 미확인 | [문서](https://f1tenth-gym-jax.readthedocs.io/en/latest/) |
| ForzaETH race stack 시뮬 | 스택 포함 sim | 단일 | 스택과 연동 | 없음 | 예 | 오픈소스 | [GitHub](https://github.com/ForzaETH/race_stack) |
| AutoDRIVE (Unity) | 차량 + 디지털 트윈 | 단일 인스턴스 위주 | LiDAR, 카메라 등 | Unity 환경, explore/practice/compete 빌드 | 예(Devkit) | 미확인 | [ICRA26 sim league](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-icra-2026), [논문](https://arxiv.org/abs/2307.13272) |
| Wheeled Lab (Isaac Lab) | PhysX 기반 소형 차(HOUND, MuSHR, F1TENTH) | 1024 env x 1 GPU(RTX 3080), 맵 생성기 2000+ 에이전트 | 깊이/카메라/고도맵, 센서 시뮬 | Isaac Lab | 미확인 | 논문 CC BY-NC-SA, 코드 미확인 | [CoRL 2025](https://arxiv.org/abs/2502.07380) |
| Assetto Corsa Gym | 상용 레이싱 게임 물리(차 178, 트랙 19) | 단일 | 텔레메트리 | 게임 자체 | 예 | 데이터 CC-BY-4.0, 게임은 상용 | [NeurIPS 2024](https://arxiv.org/abs/2407.16680) |
| Flightmare/Agilicious (드론) | 다기체 + 다운워시 모델 | 144 env x 4기 | 상태 | 없음 | - | 미확인 | [arXiv 2605.22748](https://arxiv.org/abs/2605.22748) |
| MuJoCo MJX/Brax, CARLA | 이번 조사에서 레이싱 관련 최신 사례 직접 확인 못 함 | - | - | - | - | - | 미확인 |

on-track 시스템 식별 최근 연구: [Dikici 2025](https://arxiv.org/abs/2411.17508)(30초 데이터), [Oeltjen 2025 슬립/마찰 온라인 추정, 1/10 실차](https://arxiv.org/abs/2509.15423), [Wu 2026 비전 기반 마찰 사전분포, 시뮬만](https://arxiv.org/abs/2603.09399).

"차량·센서·환경을 UI로 구성하는 게임 같은 시뮬"에 가장 가까운 선례 Top 3
1. AutoDRIVE Simulator: Unity 기반, 디지털 트윈, 빌드 3종, ROS 2 Devkit, 컨테이너 제출 (대회 운영 실적).
2. Isaac Lab + Wheeled Lab: GPU 병렬 + 센서 시뮬 + 소형 차 자산(UI 편집은 Isaac Sim 편집기 의존).
3. Assetto Corsa Gym: 차/트랙 선택 UI와 대규모 인간 데이터(단, 상용 게임이라 센서/차 구성 자유도는 낮음).

## 6. 시사점

### 6.1 "원시 LiDAR -> 궤적 E2E를 PPO 단독, 처음부터"가 주류 성공 사례와 갈리는 지점
| 주류 성공 사례 | 팀 접근 |
|---|---|
| 상위 대회 팀은 모듈형, 학습은 초기값/보조 (LAMARRacing, UNIST로 보이나 후자는 미확인) | 학습이 전체를 대체 |
| 실차 RL은 상태 관측 + 직접 조향 또는 고전 제어기 위 잔차 | 원시 LiDAR 입력, 궤적 출력 |
| 시뮬 충실도(액추에이터/타이어 식별)에 투자 (사례 1: 모델 제거 시 0.9 s 손해) | 팀 시뮬의 식별 수준은 이 조사에서 검증하지 않음(내부 점검 필요) |
| 1대1은 리그 + 어텐션 인코더 + 비대칭 크리틱 | PPO 단독, 상대 정체(특히 상대 있을 때) |
| 교사/특권정보로 부트스트랩 후 감쇠 | 처음부터 |
| 단순 보상, 큰 배치, 1~2억 스텝 + 병렬 | 현재 실험 규모/반복 방식은 내부 점검 사항(이 조사 범위 밖) |

### 6.2 성공 사례의 공통 요소 (7개)
1. 시뮬 동역학을 실차 데이터로 식별(액추에이터 지연/타이어)하고, 무작위화는 마찰 등 핵심 몇 개에 좁게.
2. 관측을 압축/구조화(상태, 앞 경계점, 10점 LiDAR 요약 등). 대용량 원시 입력은 증류 학생 단계로.
3. 액션 공간 설계: 가속도 적분, 고전 제어기 위 잔차, 감쇠로 독립 정책화(직접 속도 지령은 실차에서 과격).
4. 1억 스텝 이상의 병렬 대규모 학습(400 env, 144 env, 8-GPU) 또는 4시간 안의 효율 SAC(잔차).
5. 단순한 보상(진행 + 소수 페널티 또는 희소 승리), 보상 항 누적 회피.
6. 1대1은 리그/이력 + 순서 불변 상대 인코더 + 비대칭 크리틱 + 상대 다양성.
7. 고전 기준선(MPC, MAP, FTG)과 같은 트랙, 같은 조건 실차 비교로 평가. 사례들도 MAP/MPC를 못 이기는 경우를 솔직히 보고(RLPP).

### 6.3 한계
- 대회 상위 팀의 세부 스택은 대부분 비공개(UNICORN, VAUL 2 등 미확인).
- 검색 도구가 초록 위주여서 일부 수치(Xu 외, Sivashangaran 외, α-RPO)는 본문 미확인.
- ICRA 2027/RoboRacer Korea 공지는 찾지 못함. 한국 대회 규정은 2024년 Korea Race 사이트(korea-race24f1tenth.org)가 있다는 사실만 확인, 내용 미확인.
