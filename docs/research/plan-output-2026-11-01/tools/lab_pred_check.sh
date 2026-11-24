#!/bin/bash
# (11-24) 상대 차 예측을 policy가 쓰나 (pred_check.py): 랩 PC에서 checkpoint 하나의 exam4를 F1SIM_EXAM_PRED_PROBE=1 또는
# F1SIM_EXAM_PRED_ORACLE=1로 (exam_record.py) -- lab_race_exam_loop.sh가 exam4를 띄우는 것과 같은 환경 변수·인자, 10 GB 상한 scope,
# /usr/bin/time -v로 최대 RSS를 로그에 남김. 도구는 ~/F1tenth/exam_lab/tools_pred (돌고 있는 루프의 tools_race와 따로), 출력은 out_pred.
# 인자: RUN U MODE(probe|oracle) [환경 수 256] [초 90]. 띄우고 바로 돌아옴(기록 이름을 찍음); 끝나면 로그 끝에 "rc N".
RUN=$1; UU=$2; MODE=$3; NE=${4:-256}; SECS=${5:-90}
H=${LAB_H:-shchon11@10.100.0.2}; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
LC=/home/shchon11/F1tenth/console_main/f1sim; LT=/home/shchon11/F1tenth/exam_lab/tools_pred; LX=/home/shchon11/F1tenth/exam_lab/out_pred
case $MODE in probe) FLAG=F1SIM_EXAM_PRED_PROBE=1;; oracle) FLAG=F1SIM_EXAM_PRED_ORACLE=1;; *) echo "MODE: probe|oracle"; exit 1;; esac
XS=""; SUF=""
if [ "$SECS" != 90 ]; then XS=" $SECS"; SUF="_s$SECS"; fi
B=exam4_${RUN}_u${UU}_ideal_race4_n${NE}_${MODE}${SUF}
ENVS="PYTHONPATH=$LC WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555 CUDA_VISIBLE_DEVICES=0 $FLAG"
$SSH $H "mkdir -p $LX && cd $LC && setsid nohup bash -c 'systemd-run --user --scope -p MemoryMax=10G --quiet /usr/bin/time -v env $ENVS timeout 5400 ~/F1tenth/.venv/bin/python3 -u -W ignore $LT/exam_record.py ~/f1sim_runs/$RUN/plan_u$UU.pt $LX/$B.npz ideal $NE race4$XS > $LX/$B.log 2>&1; echo rc \$? >> $LX/$B.log' > /dev/null 2>&1 < /dev/null &"
echo $B
