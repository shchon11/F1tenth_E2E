#!/bin/bash
# (11-15 23:40) 다시 돌리면 최신 코드·checkpoint로 갱신. lab PC(tcar, WireGuard 10.100.0.2)에서 콘솔을 돌릴 수 있게: 최신 코드(f1sim) + 주요 checkpoint + 상자 배치 + 캐시
H=shchon11@10.100.0.2; SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
RUNS=/home/shchon11/f1sim_runs; L=/home/shchon11/F1tenth_E2E
PICKS="pick_avg26s_u900 watch_kr26b watch_kr26a watch_plan26t watch_plan25r pick_korea26_v2_limits pick_plan25b_u1200 pick_srp25tw3_u800 pick_sst26h3 pick_sst25gwt pick_iccas_s1102_u384"
echo "$(date +%T) 시작"
$SSH $H 'mkdir -p ~/F1tenth/console_main ~/f1sim_runs ~/.cache/f1sim/racelines ~/.cache/f1sim/map_lut ~/.cache/f1sim/centerlines'
# 1) code: the f1sim package at our HEAD (= GitHub main), compressed; external/ -> the lab checkout's (Korea 2025 map etc.)
( cd $L && git archive HEAD f1sim | gzip -1 | $SSH $H 'tar -xz -C ~/F1tenth/console_main && ln -sfn ~/F1tenth/F1tenth_E2E/external ~/F1tenth/console_main/external' ) && echo "$(date +%T) 코드 끝" &
# 2) checkpoints (symlinks resolved), banks, caches: parallel streams (one stream through the tunnel crawls)
for p in $PICKS prop_banks_box2 prop_banks_box2_korea_2026_competition; do
  ( tar -chf - -C $RUNS $p | $SSH $H 'tar -x -C ~/f1sim_runs' && echo "$(date +%T) $p 끝" ) &
done
( cd ~/.cache/f1sim && tar -czf - map_lut centerlines $(cd racelines && ls | grep -E "^korea_2025_iccas_[0-9a-f]+\.csv$|^korea_2026_competition_[0-9a-f]+\.csv$" | sed 's#^#racelines/#') | $SSH $H 'tar -xz -C ~/.cache/f1sim' && echo "$(date +%T) 캐시 끝" ) &
wait
# 3) the console list: only what was sent, in our order
python3 - "$PICKS" <<'PY' | $SSH $H 'cat > ~/f1sim_runs/console_picks.json'
import json, sys
keep = sys.argv[1].split()
d = json.load(open("/home/shchon11/f1sim_runs/console_picks.json"))
d["picks"] = [e for e in d["picks"] if e["run"] in keep]
print(json.dumps(d, ensure_ascii=False, indent=1))
PY
# 4) the launcher
$SSH $H 'cat > ~/F1tenth/console_main/run_console.sh' <<'SH'
#!/bin/bash
# 최신 코드(GitHub main = 연구 PC와 같음)로 콘솔. checkpoint는 ~/f1sim_runs (연구 PC에서 보낸 주요 것), 기존 ~/F1tenth/F1tenth_E2E는 그대로
source /home/shchon11/F1tenth/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth/console_main/f1sim F1SIM_RUNS=/home/shchon11/f1sim_runs F1SIM_RACELINE_CACHE_ONLY=1
export DISPLAY=${DISPLAY:-:0}
cd /home/shchon11/F1tenth/console_main/f1sim && exec python3 -m f1sim.viewer.console "$@"
SH
$SSH $H 'chmod +x ~/F1tenth/console_main/run_console.sh; du -sh ~/f1sim_runs ~/F1tenth/console_main/f1sim'
echo "$(date +%T) 전송 끝"
