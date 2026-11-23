#!/bin/bash
# (11-23, the user: "그냥 새로 해", then the training method agreed "ㅇㅋㅇㅋ": a hard hit does not end the race -- the car gets
# itself out -- and the scenes that teach not ramming are made common) the bag-scan world's training bank (2,000 layouts a
# track, judge from the smoothed centerline) -> merge the two PCs' parts -> check -> launch scan31a from scratch, 40 s
# episodes: half the box draws keep the boxes on the car's own line first, the ones it sees late (< 4 m of its line, behind a
# corner) first of all; boxes taken away at 10-30 s in half the box races; distance shaping only until the car drives
# (u100-150), then lap time and contact costs only; exam3 on the lab PC every 25 updates
RT=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; D=$RT/out; P=/home/shchon11/f1sim_runs/_bank_parts
BK=/home/shchon11/f1sim_runs/prop_banks_box2_korea26scan
H=shchon11@10.100.0.2; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
until grep -q "^== 끝" $D/bank_korea26scan_smooth_local.log && $SSH $H 'grep -q "^== 끝" ~/F1tenth/exam_lab/out/bank_korea26scan_smooth_lab.log'; do sleep 30; done
echo "$(date +%T) bank 두 조각 끝"
scp -q $H:f1sim_runs/_bank_parts/korea26scan_smooth_lab.pt $P/ || { echo "연구실 조각 받기 실패"; exit 1; }
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
cd $RT/tools
python3 -W ignore - <<'PY' || { echo "bank 확인 실패"; exit 1; }
import os, shutil, sys
sys.path.insert(0, ".")
import layout_bank_par as LP
import torch
P = "/home/shchon11/f1sim_runs/_bank_parts"; BK = "/home/shchon11/f1sim_runs/prop_banks_box2_korea26scan"
new = f"{P}/korea26scan_smooth.pt"
per = LP.merge([f"{P}/korea26scan_smooth_local.pt", f"{P}/korea26scan_smooth_lab.pt"], new)
names = ["real:korea26_scan+duct"] + [f"real:korea26_scan+duct+hose{k}" for k in range(16)]
assert sorted(per) == sorted(names), sorted(per)
for n, v in per.items():
    s = v["sid"].numpy()
    assert s.shape[0] == 2000, (n, s.shape)
    assert set(s[s >= 0].tolist()) <= set(range(32, 40)), n
for f in os.listdir(BK):
    if f.endswith(".pt"):
        shutil.move(f"{BK}/{f}", f"{P}/old/{f}")
shutil.copy(new, f"{BK}/korea26scan.pt")
print("bank 17 트랙 x 2000 배치 -> " + BK)
PY
tar -cf - -C /home/shchon11/f1sim_runs prop_banks_box2_korea26scan | $SSH $H 'tar -x -C ~/f1sim_runs'
cd $RT
EVAL=lab3 bash tools/launch_run.sh scan31a /home/shchon11/F1tenth_E2E/f1sim/f1sim/learn/configs/plan_world_korea26scan_qual.json 1500 "" 0 \
  --mem-keep-s 30 --mem-keep-log --scene-replay 0.25 --scene-keep-unsolved --scene-episode-s 13 --scene-from 100 --removal-p 0.5 \
  --online-boxes 0.5 --online-late 4 --shaping-fade 100,150 --horizon 64 --minibatches 8 --horizon-auto 4 --horizon-auto-min 2 \
  --horizon-auto-mult 2 --horizon-auto-down --mem-chunk-rollout --mem-prev 80
echo "$(date +%T) scan31a 띄움"
for k in $(seq 1 40); do
  sleep 30
  ps -eo args | grep -q "[p]lan_rl --name scan31a" || { echo "$(date +%T) scan31a 프로세스 없음"; tail -20 $D/scan31a.log; exit 1; }
  # update 3: the own-line / late-box draws start at update 2 (the line is measured in update 1)
  grep -q "업데이트 3 " $D/scan31a.log 2>/dev/null && { echo "$(date +%T) scan31a update 3"; grep -m3 "업데이트 [123] " $D/scan31a.log | cut -c1-200; exit 0; }
done
echo "$(date +%T) 20분 안에 첫 update 없음"; tail -5 $D/scan31a.log
