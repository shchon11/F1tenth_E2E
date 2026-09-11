# 이 PC에서 실행하기

작업 경로: `/home/shchon11/F1tenth`, 소스: `F1tenth_E2E`.
RTX 4060 Ti 8GB, Python 3.10, PyTorch 2.10.0+cu130, ROS 2 Humble 환경이다.
`.venv`는 기존 시스템/사용자 PyTorch를 공유하고, 추가 패키지는 가상환경에 설치했다.

## 터미널 활성화

bash와 zsh 모두 새 터미널에서 다음을 실행한다.

```bash
source /home/shchon11/F1tenth/activate.sh
```

ROS overlay, Python 경로, W&B entity/project, CPU 스레드 기본값을 설정한다.
W&B 인증은 `~/.netrc`에 저장되어 있으며 토큰은 저장소에 포함하지 않는다.
실행 이름은 기존 실험과 겹치지 않게 지정한다. 체크포인트는 `~/f1sim_runs/<name>/`에 저장된다.

## 시뮬레이터 보기

```bash
python F1tenth_E2E/f1sim/scripts/demo_viewer.py --map gen:competition:0 --cars 2
```

직접 운전하려면 `--manual`을 추가한다. W/S, A/D, Space 제동, R 초기화.
실측 대회 맵은 `--map real:korea_2025_iccas`로 선택한다.
처음 사용하는 맵의 raceline 생성과 새로운 배치 크기의 CUDA 커널 컴파일에는 수 분이 걸릴 수 있다.

## 학습 시작점

아래는 8GB GPU를 위한 보수적인 시작 설정이다. 전체 32개 학습 트랙의 장시간 수렴을 검증한 설정은 아니다.
동시에 여러 학습을 실행하지 말고, 메모리가 부족하면 `--envs`와 배치 크기를 절반으로 줄인다.
먼저 DAgger를 완료한 뒤 그 체크포인트로 PPO를 실행한다.

```bash
python -m f1sim.learn.dagger \
  --name dagger_4060ti_01 --tracks train --action-mode plan --hist-len 20 \
  --scan-stack 6 --scan-stride 2 --scan-deltas \
  --envs 1024 --batch 1024 --iters 8 --steps 250 --epochs 6 \
  --speed-cap 4 --teacher-speed 0.95 --wandb online

python -m f1sim.learn.ppo \
  --name ppo_4060ti_01 --init ~/f1sim_runs/dagger_4060ti_01/student_latest.pt \
  --tracks train --action-mode plan --hist-len 20 \
  --envs 256 --horizon 64 --minibatch 256 --amp \
  --total 10000000 --cap0 4 --cap1 4 --gamma 0.995 \
  --collision-penalty 150 --collision-speed-penalty 20 \
  --proximity-penalty 0.5 --proximity-speed-ref 4 \
  --plan-clearance-penalty 2 --plan-margin 0.15 --wandb online
```

PPO는 환경에 보내는 명령을 범위 제한하되, 학습에는 제한 전 행동과 그 행동의 확률을 저장하도록 수정했다.
기존 체크포인트의 형상과 추론 API는 유지된다.

## 평가와 학습 관찰

```bash
python -m f1sim.learn.evaluate \
  ~/f1sim_runs/ppo_4060ti_01/ppo_final.pt \
  --tracks eval --per-track --envs 64 --steps 2400 --speed-cap 4

python -m f1sim.learn.watch \
  --run ~/f1sim_runs/ppo_4060ti_01 --map gen:competition:0 --cars 4
```

인자 없이 `python -m f1sim.learn.watch`를 실행하면 launcher GUI가 열린다. 기본 balanced compile은 이 PC에서 처음 약 17초, 캐시 후 약 15초에 열렸고 2대 기준 sim 0.99~1.00x, 렌더 29.8fps였다. compile을 끄면 약 6초에 열리지만 sim은 약 0.19x로 느려진다. 학습과 동시에 볼 때는 차량 4대 정도를 권장한다.

DAgger/PPO 학습 그래프의 `collision_rate`는 분모가 다르다. 동일 조건의 별도 평가로 비교한다.
세팅 검증용 `setup_4060ti_*` 모델은 짧은 실행 결과이며 레이싱 성능을 갖춘 모델이 아니다.

## ROS 2 스택

```bash
ros2 launch f1sim_ros f1tenth_stack_sim.launch.py map:=gen:competition:0
```

다른 활성화된 터미널에서:

```bash
ros2 topic pub -r 20 /drive ackermann_msgs/msg/AckermannDriveStamped \
  '{drive: {steering_angle: 0.0, speed: 0.7}}'
```

화면 없이 실행하려면 launch 명령에 `viewer:=false`를 추가한다.
재빌드:

```bash
bash F1tenth_E2E/external/setup_colcon_ignore.sh
colcon build --symlink-install --base-paths F1tenth_E2E \
  --packages-up-to f1sim_ros f1tenth_stack --parallel-workers 3
```

## 의존성 재설치

기존 CUDA PyTorch를 공유하는 이 PC의 환경을 재구성할 때:

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python --no-deps -e 'F1tenth_E2E/f1sim[learn,viewer,dev,gym]'
uv pip install --python .venv/bin/python websockets wandb moderngl glfw trimesh gymnasium scikit-image
uv pip install --python .venv/bin/python --no-deps matplotlib==3.10.8
printf 'import mpl_toolkits\n' > .venv/lib/python3.10/site-packages/00-f1sim-matplotlib.pth
git -C F1tenth_E2E submodule update --init --recursive
```

완전히 새로운 PC에서는 PyTorch/CUDA를 먼저 설치하고 기본 의존성 전체도 설치해야 한다.
이 PC의 Ubuntu Matplotlib namespace가 가상환경의 Matplotlib을 가리지 않도록 `.pth` 파일을 설정했다.

## 설치 검증 기록

- 학습/평가 카탈로그 38개 항목을 모두 로드했다.
- CUDA 시뮬레이터에서 유한한 차량 상태와 1080빔 스캔을 확인했다.
- DAgger 짧은 실행에서 체크포인트 저장과 W&B 업로드를 확인했다.
- 해당 DAgger 체크포인트로 PPO 256개 환경, horizon 16, 4회 업데이트(16,384 step), bf16 학습 및 최종 체크포인트 저장을 확인했다. 한 개 절차 생성 트랙에서 컴파일 이후 약 8,700~8,800 step/s였다. 전체 트랙 성능 측정은 아니다.
- OpenGL 오프스크린 이미지 5개와 GLFW 실제 창 20프레임을 렌더링했다.
- ROS 패키지 7개 빌드 후 `/drive` 0.7 m/s 명령으로 `/odom` 속도 약 0.65 m/s와 `/scan` 수신을 확인했다.
- PPO 확률 일관성 CPU/CUDA·FP32/bf16 회귀 테스트 및 기존 학습 테스트: 6개 통과.
- 수정한 Python 파일의 오류 중심 Ruff 검사, compileall 및 diff 공백 검사를 통과했다.

환경 버전과 스크린샷은 `/home/shchon11/F1tenth/setup-evidence/`에 있다.
ROS 빌드에는 기존 외부 패키지의 헤더 설치 경고가 있으나 빌드와 실제 토픽 동작은 통과했다.

W&B 실행: [DAgger 세팅 검증](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/ol7mgc9h), [PPO 세팅 검증](https://wandb.ai/shchon11-hanyang-university/f1sim-e2e/runs/961pa6yk).
