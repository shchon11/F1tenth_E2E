#!/bin/bash
# (11-17) 상대 차 시험(oppexam_run.sh)을 학습의 100 update마다: checkpoint가 생기면 5060에서 돌림. 인자: 이름 처음 끝
R=/home/shchon11/f1sim_runs; T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
N=$1; U=$2; U1=$3
while [ $U -le $U1 ]; do
  until [ -e $R/$N/plan_u$U.pt ]; do sleep 60; done
  sleep 30
  bash $T/oppexam_run.sh $N/plan_u$U
  U=$(( U + 100 ))
done
