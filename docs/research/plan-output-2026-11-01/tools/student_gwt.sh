#!/bin/bash
# Korea 2025 라이다 학생을 새 상자 선생 gwt u100에서 다시 증류 (11-03 19:05): 콘솔 ★ 학생 sst25h2는 11-02 선생(srp25h2_propsmix)에서 배움.
# gwt = 랩 위치 없음 + 라인 위 상자 + 독립 잡음 + 상자 미는 시간 값 -- 30배치 7.75 s, 세게 4, 멈춤 0. 학생 세계도 같은 라인 위 상자(--prop-line 0.3).
# 나머지는 chain_props3.sh의 학생과 같음(DAgger 80회, 라이다 4장, 격자) -- 단 버퍼 30만 -> 15만(11 GB가 상대 단계 학습과 같이 못 돎).
# 기각(미리 정함, 같은 30배치 시드 777에서 sst25h2와 나란히): 세게(벽+상자)가 sst25h2보다 많거나 같음, 1 s 넘게 멈춘 배치 하나라도,
# 랩 중앙이 sst25h2보다 0.3 s 넘게 느림, 혼자 랩이 6.6 s보다 느림.
GPU=$1; S=$2; MC=real/iccas25
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_student --name sst25gwt_props --teacher $R/srp25h3gwt_props/state_u100.pt \
  --stage props --envs 768 --iters 80 --steps 256 --buffer 150000 --buffer-cpu 1 --epochs 3 --frames 4 --stride 4 --danger-weight 4 \
  --prop-fraction 0.8 --prop-count uniform --prop-line 0.3 --removal-p 0.5 --removal-s 10 30 --unwedge 0 --grid 1 --seed ${S}9 \
  > $D/sst25gwt_props.log 2>&1 || { echo fail > $D/student_gwt.done; exit 1; }
C=$(ls -t $R/sst25gwt_props/student_i*.pt | head -1)
L=$D/student_gwt.log
echo "== 새 학생 $(basename $C) (선생 gwt u100)" > $L
echo "  혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $C $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/one_crate.py $MC 30x34 lineref=$R/srp25h2_propsmix/state_u200.pt sst25gwt=$C 2>&1 | grep -A 12 "== sst25gwt" >> $L
for x in "sst25gwt $C" "sst25h2 $R/pick_sst25h2/state_latest.pt"; do
  set -- $x
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $2 $MC 40 30 777 $D/layouts/s_$1 2>&1 | grep "배치\|합계\|Error" > $D/layouts_s_$1.log
  echo "  $1 30배치: $(tail -n 1 $D/layouts_s_$1.log)" >> $L
done
echo done > $D/student_gwt.done
