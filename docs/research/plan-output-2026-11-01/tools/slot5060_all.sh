#!/bin/bash
# 5060 자리 둘을 다 잡고 혼자 도는 판(slot5060.sh의 1024 환경 갈래와 같은 순서 .0 -> .1): 환경 1024개짜리 시험 기록(exam_record.py)처럼
# GPU 메모리를 6 GB 넘게 쓰는 작업용. 쓰는 법: CUDA_VISIBLE_DEVICES=1 bash slot5060_all.sh timeout 3000 python3 ...
L=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/.slot5060
while [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -lt 3500000 ]; do sleep 10; done
exec 8>$L.0; flock 8
exec 9>$L.1; flock 9
"$@" 8>&- 9>&-; rc=$?
exit $rc
