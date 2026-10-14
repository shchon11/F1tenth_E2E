# sourced: the environment every v2cc job runs in (the v2smoke_b settings, on wt_cc)
SP=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/d98b5699-f98e-4674-9a12-ab5a57bf35f1/scratchpad
WT=$SP/wt_cc
D=$SP/v2cc
cd $WT
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=$WT/f1sim:$WT/f1sim_ros:${PYTHONPATH:-}
export TORCHINDUCTOR_COMPILE_THREADS=2 WANDB_MODE=disabled
SLOT='{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.7,1.05],"present":0.5,"spawn":"random","reactive":{"defend":0.3,"line":0.2,"oblivious":0.1},"events":["brake","stop","shift","weave"],"event_rate":0.5}'
OPP="--race-size 4 --kind-mix-assign partition --opp-slots [$SLOT,$SLOT,$SLOT] --opponent-speed-mode linear"
WORLD="--collision-mode soft --movable-obstacles --spawn-runway 3.0 --procedural-obstacles 0.5 --procedural-density 1.5 --procedural-max-props 18"
TEACH="--raceline-objective min_curvature --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0"
INITSTEM="--v2-init-stem $WT/checkpoints/iccas_bootcamp_safest.pt"
HELD='real:korea_2025_iccas,real:korea_2025_iccas~rev,real:blackbox2022_3,real:blackbox2022_3~rev,gen:competition:0,gen:control:9100,gen:competition:9200+pinch9200,real:map12x16,real:map12x16~rev'
