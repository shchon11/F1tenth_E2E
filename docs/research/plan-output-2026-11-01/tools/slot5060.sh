#!/bin/bash
# 5060 무거운 작업 자리 2개 (11-09: 평가·분석·영상 셋이 겹쳐 14 GB 메모리가 바닥났고 Claude Code가 대기 작업을 죽임 -> 한 번에 둘까지).
# 자리 파일 둘(out/.slot5060.0, .1) 중 빈 것을 flock으로 잡고 명령을 돌린 뒤 놓음(잡은 건 이 셸이고 명령에는 넘기지 않음). 둘 다 차 있거나
# 남은 메모리가 3.5 GB 아래면(작업 하나 약 3 GB; 콘솔을 켜 두면 더 적음) 10 s마다 다시 봄.
# 11-10: 학습 조건 평가(plan_validate 환경 1024)는 5060 GPU 메모리를 6.3 GB(8 GB 중) 써서 영상과 같이 돌면 죽음(u50 평가가 출력 없이 끝남)
# -> 그 명령은 두 자리를 다 잡고 혼자 돔.
# 쓰는 법: CUDA_VISIBLE_DEVICES=1 bash slot5060.sh timeout 1500 python3 ...
L=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/.slot5060
ALL=0
case " $* " in *plan_validate.py*" 1024 "*) ALL=1 ;; esac
if [ $ALL = 1 ]; then
  while [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -lt 3500000 ]; do sleep 10; done
  exec 8>$L.0; flock 8
  exec 9>$L.1; flock 9
  "$@" 8>&- 9>&-; rc=$?
  exit $rc
fi
while true; do
  if [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge 3500000 ]; then
    for i in 0 1; do
      exec 9>$L.$i
      if flock -n 9; then
        "$@" 9>&-; rc=$?
        exec 9>&-
        exit $rc
      fi
      exec 9>&-
    done
  fi
  sleep 10
done
