#!/bin/bash
# (11-16 02:10) 사용자: 콘솔에서 "존나 미끌미끌", plan 타이어 한계 0.95 -> 0.85는 "ㄴ" -> 제어기(MAP) 쪽에서: 설정별로 벽·상자 충돌, 속도와
# 미끄럼각을 같이 잼 (exam1, 128 세션 x 30 s, 차 물리는 학습 세계처럼 무작위·마찰 고정). 4070 (kr26a 학습은 그동안 지킴이가 멈춤)
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/mapslip; mkdir -p $O
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
CK=/home/shchon11/f1sim_runs/avg26s/plan_u900.pt
cd /home/shchon11/F1tenth_E2E/f1sim
run() {   # name, override json (or OLD)
  N=$O/$1.npz; [ -e $N ] && return
  if [ "$2" = OLD ]; then F1SIM_MAP_OLD=1 CUDA_VISIBLE_DEVICES=0 timeout 1200 python3 -u -W ignore $T/exam_record.py $CK $N map 256 uniform 30 50 > ${N%.npz}.log 2>&1
  else F1SIM_MAP_OVERRIDE="$2" CUDA_VISIBLE_DEVICES=0 timeout 1200 python3 -u -W ignore $T/exam_record.py $CK $N map 256 uniform 30 50 > ${N%.npz}.log 2>&1; fi
  echo "$(date +%T) $1 끝"
}
run now '{}'
run old OLD
run gain100 '{"v_steer_gain_max": 1.0}'
run gain115 '{"v_steer_gain_max": 1.15}'
run rate01 '{"steer_rate_limit": 0.1}'
run rate005 '{"steer_rate_limit": 0.05}'
run gain115_rate01 '{"v_steer_gain_max": 1.15, "steer_rate_limit": 0.1}'
run gain100_l1old '{"v_steer_gain_max": 1.0, "m_l1": 0.47, "q_l1": 0.0, "t_clip_min": 0.35}'
python3 - <<'PY'
import glob, numpy as np, os
O = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/mapslip"
print(f"{'설정':<16}{'벽 세게':>8}{'상자 세게':>9}{'차 세게':>8}{'평균 속도':>9}{'미끄럼 90%':>11}{'10도 넘음':>10}")
for f in sorted(glob.glob(O + "/*.npz"), key=os.path.getmtime):
    z = np.load(f); c = list(z["cols"]); n = z["cnt"][-1]; st = z["st"][:-1]; vy = z["vy"]
    vx = st[:, :, 3]; m = vx > 2.0
    b = np.degrees(np.abs(np.arctan2(vy[m], vx[m])))
    print(f"{os.path.basename(f)[:-4]:<16}{n[:, c.index('hard_wall')].mean():8.2f}{n[:, c.index('hard_prop')].mean():9.2f}{n[:, c.index('hard_car')].mean():8.2f}"
          f"{vx.mean():9.2f}{np.percentile(b, 90):10.1f}도{np.mean(b > 10) * 100:9.0f} %")
PY
