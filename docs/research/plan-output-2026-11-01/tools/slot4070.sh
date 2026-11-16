#!/bin/bash
# 4070 평가 자리 2개 (11-16): 5060에 학습(kr26d, 장면 재생으로 GPU 4.9 GB)이 있으면 그 옆에 평가가 들어갈 GPU 메모리가 없어(plan_validate
# CUDA out of memory, 결과 줄 없이 끝남) 평가를 4070으로 옮김. 4070 학습(4.6 GB) + 평가 둘(~3.4 GB씩)은 12 GB 안 -- 자리 하나로는 두 run의
# 평가가 밀려 kr26d u1150 시험이 '최신만 평가'로 건너뛰어짐(11-16 10:23) -> 둘. 4070 남은 메모리 4 GB, 컴퓨터 메모리 3.5 GB 이상일 때만 시작.
# gpu_guard3.sh는 tools/*.py를 다른 작업으로 치지 않으므로 4070 학습을 멈추지 않음. 쓰는 법: CUDA_VISIBLE_DEVICES=0 bash slot4070.sh timeout 1500 python3 ...
L=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/.slot4070
U=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep 4070 | cut -d, -f1)
# 11-16 19:55: 4000 -> 2600 MB -- kr26e3 (horizon 384) holds 8.6 GB of the 4070, nvidia-smi free 3.3 GB; an exam (256 envs) runs in
# < 2.2 GB (the kr26i exams ran on the 5060 beside its training with 2.2 GB free); a 4000 MB gate held them forever. Training memory is at its steady peak (horizon-auto at its max K).
gpu_free() { nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | grep "$U" | awk -F', ' '{print $2}'; }
while true; do
  if [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge 3500000 ] && [ "$(gpu_free)" -ge 2600 ]; then
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
  sleep 15
done
