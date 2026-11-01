#!/bin/bash
# 4070 지킴이 (11-01): 사용자의 다른 작업(콘솔 등)이 4070을 쓰는 동안 4070의 내 학습(이름 머리 $1)을 멈추고(SIGSTOP),
# 4070이 비면 이어 감(SIGCONT). 연쇄가 끝나면($1 chain done) 같이 끝남
TAG=$1
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
U=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | grep 4070 | cut -d, -f1)
st=run
while [ ! -f $D/chain_${TAG}.done ]; do
  mine=$(pgrep -f "state_[r]l --name ${TAG}_" | tr '\n' ' ')
  other=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$U" | awk -F', ' '{print $2}'); do
    case " $mine " in *" $p "*) ;; *) other=1;; esac
  done
  if [ $other = 1 ] && [ $st = run ] && [ -n "${mine// /}" ]; then kill -STOP $mine; st=stop; echo "$(date +%H:%M:%S) 4070에 다른 작업 -> 멈춤 $mine"; fi
  if [ $other = 0 ] && [ $st = stop ]; then [ -n "${mine// /}" ] && kill -CONT $mine; st=run; echo "$(date +%H:%M:%S) 4070 비었음 -> 재개 $mine"; fi
  sleep 5
done
