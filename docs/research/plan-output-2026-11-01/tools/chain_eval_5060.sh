#!/bin/bash
# 5060 평가 (11-02 06:00): 같은 30배치(시드 777, 40 s)로 옛 ×1.2 선생·학생 기준선, 그리고 상자 접촉 대가 선생 두 개(+ 혼자 60 s)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
bash $T/layouts_run.sh tch26x $R/srp26xh2_propsmix/state_u200.pt real/korea26x12 40 30 777 > $D/layouts_tch26x.log 2>&1
bash $T/layouts_run.sh sst26x $R/sst_srp26xh2_props/student_i80.pt real/korea26x12 40 30 777 > $D/layouts_sst26x.log 2>&1
for X in "srp25h2 real/iccas25 tch25t" "srp26xh2 real/korea26x12 tch26xt"; do
  set -- $X
  until [ -f $D/$1_propst.done ]; do sleep 30; done
  C=$R/$1_propstmix/state_u100.pt
  cd /home/shchon11/F1tenth_E2E/f1sim
  echo "$3 혼자: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C $2 60 2>&1 | grep '주행\|Error')" > $D/solo_$3.log
  bash $T/layouts_run.sh $3 $C $2 40 30 777 > $D/layouts_$3.log 2>&1
done
echo done > $D/chain_eval_5060.done
