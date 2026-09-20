from pathlib import Path
import hashlib,json,os,time,math
from datetime import datetime,timezone
root=Path('/home/shchon11/Documents/Codex/2026-09-17/new-chat');out=root/'outputs'
run=Path('/home/shchon11/f1sim_runs/dagger_mixed_minimumtime_s18501')
man=json.loads((run/'dagger-v4-launch-manifest.json').read_text());pid=man['pid']
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
p=run/'progress.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
ckpts=sorted(run.glob('student_it*.pt'))
smoke=root/'work/dagger-v4/gpu-mixed-smoke/progress.jsonl';sr=[json.loads(x) for x in smoke.read_text().splitlines() if x.strip()]
source=json.loads((run/'source-manifest.json').read_text())['source_sha256']
plans=json.loads((run/'raceline-validation.json').read_text())
status={'checked_utc':datetime.now(timezone.utc).isoformat(),'pid':pid,'alive':alive,'run_dir':str(run),'source_sha256':source,'init_checkpoint_sha256':man['checkpoint']['sha256'],'configured_iterations':15,'latest_full_iteration':rows[-1] if rows else None,'checkpoints':[str(x) for x in ckpts],'gpu_smoke':sr[-1],'settings_file':str(out/'dagger-v4-settings.json'),'plans':plans,'focused_tests_passed':277,'extra_cold_map_suite':{'completed':False,'passed_before_interrupt':51,'elapsed_s':1067.77,'reason':'SLSQP cold-map runtime; no full-catalogue claim'}}
def finite_json(value):
 if isinstance(value, float) and not math.isfinite(value): return None
 if isinstance(value, dict): return {k:finite_json(v) for k,v in value.items()}
 if isinstance(value, list): return [finite_json(v) for v in value]
 return value
(out/'dagger-v4-status.json').write_text(json.dumps(finite_json(status),indent=2,ensure_ascii=False,allow_nan=False))
phase=f"본 실행 {rows[-1]['iter'] + 1}/15회 보고서와 checkpoint 확인." if rows and ckpts else '본 실행은 진행 중이며, 첫 전체 반복 보고서와 checkpoint는 아직 생성되지 않았다.'
text=f'''# DAgger 실행 및 레이싱 라인 수정 — 2026-09-18

**백그라운드 DAgger 실행 중**: `dagger_mixed_minimumtime_s18501`, PID `{pid}`. {phase} 별도의 짧은 실제 CUDA 실행에서는 혼합 수집·학습 업데이트·checkpoint 저장이 모두 통과했다. 원본 D3는 변경하지 않았다.

[패널에서 불러올 설정](dagger-v4-settings.json) · [실행 상태와 검증 수치](dagger-v4-status.json) · [경로·장애물 그림](minimum-time-obstacles-v4.png)

학습 패널의 **설정 불러오기**로 JSON을 열면 같은 실험을 확인할 수 있다. 이미 시작한 실행에 같은 이름으로 새 작업을 중복 실행하지 않는다. 실행 디렉터리는 `{run}`이며, 로그는 `dagger-v4.log`, 진행 수치는 `progress.jsonl`, checkpoint는 `student_it*.pt`에 저장된다. 세션 종료 후에도 프로세스는 별도 process group에서 계속 실행된다.

## 이번 DAgger 구성

- 원본 D3 `ppo_u500.pt`의 actor를 초기값으로 사용한 별도 DAgger 학습. GRU128·6개 scan·20개 history 관측 구조를 유지한다.
- **빈 트랙 30% / traffic 70%**: 차량 1대·장애물 없는 별도 환경에서 solo 라벨을 수집한다. 상대차를 화면에서만 숨기지 않는다.
- 48대 / race3 = 학습 차량16대. 반복마다 solo4,800개 + traffic11,200개 = **16,000개 학습 차량 라벨**. 상대차 행은 학습 데이터에서 제외한다.
- 15회 반복, 명목 신규 라벨240,000개, 최근5회 replay, 2epochs, batch64, recurrent chunk8, lr0.0001, seed18501. 속도 상한10m/s. 긴 Torch compilation을 끈 eager CUDA 실행이다.
- TRAIN의 ICRA22·control1400·control1401만 사용한다. Traffic은 기본3개 + 에셋 배치3개 변형, solo는 각 기본 맵의 완전히 빈 버전이다. 배치 시드441/442/443을 고정하고 초기 상태·마찰·상대차 행동을 reset마다 샘플링한다. 전체 학습 맵/held-out 평가를 수행했다는 뜻은 아니다.

| 상대차 | 종류 | 기준 속도 배율 | 방어 | 양보 | 코너 라인 | 상대 무시 |
|---|---|---:|---:|---:|---:|---:|
| 1 | raceline teacher | 0.55–0.90 | 0.45 | 0.10 | 0.35 | 0.10 |
| 2 | interactive teacher | 0.80–1.00 | 0.20 | 0.40 | 0.35 | 0.05 |

성향은 reset 때 각각 확률적으로 배정되므로 동시에 활성화될 수 있다. 두 상대의 출발 위치는 무작위다. 1번에는 급제동·정지·라인 변경을0.6회/10초의 설정률로, 2번에는 급제동·라인 변경을0.4회/10초로 둔다. 표는 설정된 분포이며 실현 빈도 측정값은 아니다.

학생의 정답 teacher는 solo에서 raceline, traffic에서 interactive다. Interactive는20개 경로/속도 후보를 평가한다. 관측에 나타나지 않는 임의의 teacher 성격으로 같은 상황의 정답을 바꾸는 방식은 아니다.

첫 본 학습 반복은16,000개 라벨(4,800 solo /11,200 traffic), loss0.03175로 완료됐고 모든 저장 tensor가 유한했다. 첫 평가의 student 충돌은53.34회/km, teacher는16.01회/km였다. 어려운 혼합 조건의 초기 결과이며, 개선이나 채택 성공으로 해석하지 않는다. 학습은 계속 진행한다.

## 레이싱 라인과 장애물 수정

경로 offset과 속도를 함께 변수로 두고 **시간 적분 자체를 최소화하는 NLP**로 교체했다. 전·후륜 마찰원, 하중 이동, 구동 출력·제동·저항, 조향각·조향속도와 차체 공간을 제약으로 둔다. 기존의 고정18회 보정·1m 이동 제한·숨은90% 횡그립 제한을 제거했다. 비균일 샘플의 곡률과 주기 경계 양쪽 미분도 바로잡았다.

수렴 성공, KKT 잔차, 촘촘한 제약 검사, 실제 반환 경로·속도의 독립 검사를 통과해야 저장한다. 수렴하지 않은 초기 경로를 최적화 결과로 조용히 대체하지 않는다. 진단값은 CSV 캐시에 함께 저장된다. **준정상 차량 모델의 국소 해**이며 전역 최소시간 증명이나 완전한 타이어·휠 과도동역학 최적해는 아니다. 모델 선택의 기준은 [TUM의 최소시간 문제 정식화](https://github.com/TUMFTM/global_racetrajectory_optimization/blob/master/opt_mintime_traj/Readme.md)를 참고했고, 이 코드가 그 full dynamic solver를 재현했다는 뜻은 아니다.

별도 형상으로만 저장되던 에셋을 계획용 점유 격자에 투영하고, 막힌 초기 경로 구간을 우회시킨다. 실제 차체의 전·중·후 영역을 검사한다. 장애물 모서리에서는 단일 normal-ray 폭이 과대평가되므로 차체 전체가 들어갈 공간으로 여유를 산출한다. `margin=0.4`는 상한이며 모든 곳의40cm 보장이 아니다. 실제3개 장애물 편집 맵에서는 계산된 여유가 약3.7–21.4cm였다.

- ICRA22: 모델 예상11.57초, KKT2.14e-5, 반환 경로 물리 제약 위반은 부동소수점 오차 수준.
- 실제 편집 맵 `scene_0915_2026`: 장애물3개를 포함해 모델 예상6.98초, KKT8.40e-6. 이전에는 teacher 계획에서 빠졌던 장애물을 피하는 경로가 생성됐다.
- 수치는 모델 계획값이며 실제 주행 기록이 아니다. 전체 카탈로그의 최소시간 해나 주행 안정성을 검증했다고 주장하지 않는다.

장애물 UI는 **없음 / 기본 / 랜덤 낮음·중간·높음**으로 정리했다. 랜덤 모드는 기존 에셋의 종류와 크기를 시드로 뽑는다. 실행 중 물체가 계속 크기를 바꾸는 것은 아니다. 높음은 기존 극단 배치의 패턴과 위치를 사용하며, 다른 실루엣의 정량적 난이도 동일성까지 검증한 것은 아니다. 기존 저장 시나리오는 재현성을 위해 유지한다.

## 검증과 변경 범위

최종 집중 검사 **277개 통과**. 실제 CUDA 혼합 학습에서도20라벨(traffic14+solo6), 유한 loss와 checkpoint 저장을 확인했다. 본 실행의 최초 시작에서 per-car grip 설정을 interactive 후보 batch에 복제하지 않던 오류를 발견해 수정했고, 실패 로그는 보존했다. 빠른 GN 경로 맞춤이 새 경로에서 진동하던 문제는 목적값이 줄어드는 backtracking으로 수정했다. 기존2회/6회 맞춤 정확도 검사의 기준을 완화하지 않고 통과했다.

확대한 cold-map 검사는51개 통과 후 SLSQP 계산이 길어17분47초에 중단했다. **다른 맵의 최초 최적화 시간은 남은 제한**이다. 이번 학습에 쓰는6개 변형은 모두 별도로 계산·검증·캐시 완료했다.

주요 변경 파일: `minimum_time.py`, `raceline.py`, `planning_seed.py`, `track.py`, `teacher.py`, `interactive_teacher.py`, `asset_obstacles.py`, `maps.py`, `tracks.py`, `hard_obstacles.py`, `learn/dagger.py`, `learn/common.py`, `learn/benchmark/suite.py`, 주행·학습 패널과 schema/checkpoint metadata, 관련 회귀 검사. 별도 입체 UI와 제한된 옛 경로 보정 루프를 제거했고, 기존 에셋 치수·파서·패널 형식을 재사용했다. 새 application dependency는 추가하지 않았다.

소스 snapshot SHA256: `{source}`. 학습은 고정 snapshot을 사용하며 원본 D3 SHA256은 `{man['checkpoint']['sha256']}`다. 이번 학습 결과의 성능 개선이나 기본 모델 승격은 아직 판단하지 않았다.
'''
(out/'dagger-v4-status.md').write_text(text)
print({'alive':alive,'pid':pid,'full_iteration_reports':len(rows),'checkpoints':len(ckpts),'source':source})
