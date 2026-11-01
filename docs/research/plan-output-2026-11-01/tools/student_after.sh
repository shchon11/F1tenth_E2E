#!/bin/bash
# Korea 2025 학생이 끝나면 ×1.2 학생 (호스트 메모리: 학생 하나가 5.3~6 GB, 둘을 같이 돌리면 사용 가능 메모리 0.9 GB, 스왑 7 GB)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
until [ -f $D/sst_srp25h2_props.done ]; do sleep 30; done
bash $T/gpu_guard3.sh "state_[s]tudent --name sst_srp26xh2_props" $D/sst_srp26xh2_props.done > $D/gpu_guard_sst26x.log 2>&1 &
bash $T/student_propsmix.sh srp26xh2 0 9102
