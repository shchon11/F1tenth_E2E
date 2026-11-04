#!/bin/bash
# 네트워크 하나, world 하나 + PPO 속도 신호 고침 (11-04 20:20, 사용자: "policy를 나눠서 하는건 아니지..?" -> 네트워크 하나, "ppo 문제가 더 클 것 같은데").
# 전수조사: 빈 트랙만이면 6.50 -> 6.55 s(audit A u100), 네 경우 world + solo 학습 설정이면 6.61 -> 6.75 s(audit B), srp25ows 6.71 -> 6.90 s.
# 섞인 world에서 빈 트랙 속도 신호가 묻힘: (1) hit-shaping 항이 step당 0.8 -> 1.9로 progress(약 0.13 m)의 10배 넘게 섞이고 빈 트랙에서도 감속
# 쪽(hit head가 빈 트랙 P(hit) 0.01~0.016, 실제 0), (2) advantage를 batch 전체로 정규화해 가격 80 충돌이 크기를 정함 -> 빈 트랙의 속도 advantage가
# 0에 가깝게 눌림, (3) level replay가 빈 트랙 혼자를 25 -> 약 13 %로 줄임. 이제 hit-shaping 0, --adv-norm-case 1(네 경우별 정규화), level replay 0.
# 기각(미리 정함, 시드 777): solo lap u50 > 6.58 s 또는 u100 > 6.62 s (2025) -> 속도 신호 고침으로는 부족. 이후 obstacle·race 기각선은 그대로
# (30 layout hard 47 / 26, race 0.6배 hard 차 <= 8 + 추월 >= 15/20, 1.0배 hard 차 <= 3).
# one_world_f2.sh (11-04 21:20) = one_world_f.sh + 8번째 인자 EXTRA(추가 state_rl 인자, 예: "--hidden 512"). 첫 사용: srp25owd = 512칸 supervised
# 네트워크 하나(capacity_test.py, 빈 트랙 6.50 s, 앞 8 layout hard 14)에서 같은 PPO. 기각(미리 정함): u50 solo lap > 6.58 s면 시작점 문제 아님.
# 인자: init체크포인트 이름 GPU 환경수 시드 업데이트 지도(25|26) [EXTRA]
INIT=$1; NAME=$2; GPU=$3; ENVS=$4; S=$5; U=$6; MAPK=$7; PAR=0; EXTRA=$8
if [ "$MAPK" = 26 ]; then M=real:korea_2026_competition; MC=real/korea26; else M=real:korea_2025_iccas; MC=real/iccas25; fi
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/one_${NAME}.log
echo "== ${NAME}: 네트워크 하나 + 속도 신호 고침(hit-shaping 0, 경우별 advantage 정규화, level replay 0), init $(basename $(dirname $INIT))/$(basename $INIT)" > $L
evalc() {
  local UU=$1; local C=$R/${NAME}/state_u${UU}.pt
  CUDA_VISIBLE_DEVICES=$GPU timeout 7200 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/o_${NAME}_u${UU} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_o_${NAME}_u${UU}.log
  echo "  u${UU} obstacle 30 layout: $(tail -n 1 $D/layouts_o_${NAME}_u${UU}.log)" >> $L
  for x in "forzaeth 0.6" "forzaeth 1.0" "forzaeth 1.2"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 5400 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${NAME}_u${UU}_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU $1 x$2 /" >> $L
  done
  for x in "forzaeth 0.6" "forzaeth 1.0"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 7200 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/obs_${NAME}_u${UU}_$1_x$2 1 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU obstacle+$1 x$2 /" >> $L
  done
  CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/follow_probe.py $C $MC 30 12 forzaeth 0.8 777 > $D/follow_probe_${NAME}_u${UU}_x0.8.log 2>&1
  echo "  u${UU} 따라가기 0.8배: $(grep '+3 ~   +6\|-1 ~   +1' $D/follow_probe_${NAME}_u${UU}_x0.8.log | tr -s ' ' | tr '\n' ';')" >> $L
  echo done > $D/one_${NAME}_u${UU}.done
}
( for UU in $(seq 50 50 $U); do
    until [ -e $R/${NAME}/state_u${UU}.pt ] || [ -e $D/one_${NAME}.done ]; do sleep 20; done
    [ -e $R/${NAME}/state_u${UU}.pt ] || break
    echo "  u${UU} solo: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${NAME}/state_u${UU}.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
    echo "  u${UU} obstacle 앞 8 layout: $(CUDA_VISIBLE_DEVICES=$GPU timeout 1800 python3 $T/console_layouts.py $R/${NAME}/state_u${UU}.pt $MC 40 8 777 $D/layouts/q_${NAME}_u${UU} 2>&1 | grep '합계' | cut -c1-200)" >> $L
    if [ "$PAR" = 1 ] && [ $UU = $((U / 2)) ]; then evalc $UU; fi
  done ) &
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 0 --level-replay 0 --adv-norm-case 1 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.5 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40 --prop-bank $R/prop_banks"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $COMMON --updates $U \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed $S \
  --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  --init $INIT $EXTRA > $D/${NAME}.log 2>&1 || { echo fail > $D/one_${NAME}.done; exit 1; }
wait
evalc $U
echo done > $D/one_${NAME}.done
