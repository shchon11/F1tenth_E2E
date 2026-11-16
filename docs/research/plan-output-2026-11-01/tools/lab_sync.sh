#!/bin/bash
# (11-15 23:40, 11-16 00:05 고침) 다시 돌리면 최신 코드·checkpoint로 갱신. lab PC(tcar, WireGuard 10.100.0.2)에서 콘솔을 돌릴 수 있게:
# 최신 코드(f1sim) + 콘솔 주요 checkpoint + 상자 배치 + 캐시 -> ~/F1tenth/console_main/run_console.sh
# 터널 한 줄은 0.2 MB/s라 여러 줄로 보내되, 한꺼번에 12줄 넘게 열면 lab의 sshd가 연결을 끊음(MaxStartups) -> 5줄씩, 실패하면 3번까지 다시
H=shchon11@10.100.0.2; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
RUNS=/home/shchon11/f1sim_runs; L=/home/shchon11/F1tenth_E2E
PICKS="pick_avg26s_u900 pick_kr26a_u550 watch_kr26e watch_kr26f watch_kr26d watch_kr26c watch_kr26b watch_kr26a watch_plan26t watch_plan25r pick_korea26_v2_limits pick_plan25b_u1200 pick_srp25tw3_u800 pick_sst26h3 pick_sst25gwt pick_iccas_s1102_u384"
export H SSH RUNS L
echo "$(date +%T) 시작"
$SSH $H 'mkdir -p ~/F1tenth/console_main ~/f1sim_runs ~/.cache/f1sim/racelines ~/.cache/f1sim/map_lut ~/.cache/f1sim/centerlines'
send_one() {                                   # one item: "code" | "cache" | a folder under ~/f1sim_runs
  for try in 1 2 3; do
    case "$1" in
      code)  (cd $L && git archive HEAD f1sim | gzip -1 | $SSH $H 'tar -xz -C ~/F1tenth/console_main && ln -sfn ~/F1tenth/F1tenth_E2E/external ~/F1tenth/console_main/external') ;;
      cache) (cd ~/.cache/f1sim && tar -czf - map_lut centerlines $(cd racelines && ls | grep -E "^korea_2025_iccas_[0-9a-f]+\.csv$|^korea_2026_competition_[0-9a-f]+\.csv$" | sed 's#^#racelines/#') | $SSH $H 'tar -xz -C ~/.cache/f1sim') ;;
      *)     (tar -chf - -C $RUNS "$1" | $SSH $H 'tar -x -C ~/f1sim_runs') ;;
    esac && { echo "$(date +%T) $1 끝"; return 0; }
    echo "$(date +%T) $1 실패 ($try번째) -> 다시"; sleep 3
  done
  echo "$(date +%T) $1 끝내 실패"; return 1
}
export -f send_one
printf "%s\n" code cache $PICKS prop_banks_box2 prop_banks_box2_korea_2026_competition | xargs -P 5 -I{} bash -c 'send_one {}'
python3 - "$PICKS" <<'PY' | $SSH $H 'cat > ~/f1sim_runs/console_picks.json'
import json, sys
keep = sys.argv[1].split()
d = json.load(open("/home/shchon11/f1sim_runs/console_picks.json"))
d["picks"] = [e for e in d["picks"] if e["run"] in keep]
print(json.dumps(d, ensure_ascii=False, indent=1))
PY
$SSH $H 'cat > ~/F1tenth/console_main/run_console.sh' <<'SH'
#!/bin/bash
# 최신 코드(GitHub main = 연구 PC와 같음)로 콘솔. checkpoint는 ~/f1sim_runs (연구 PC에서 보낸 주요 것), 기존 ~/F1tenth/F1tenth_E2E는 그대로
source /home/shchon11/F1tenth/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth/console_main/f1sim F1SIM_RUNS=/home/shchon11/f1sim_runs F1SIM_RACELINE_CACHE_ONLY=1
export DISPLAY=${DISPLAY:-:0}
cd /home/shchon11/F1tenth/console_main/f1sim && exec python3 -m f1sim.viewer.console "$@"
SH
$SSH $H 'chmod +x ~/F1tenth/console_main/run_console.sh; du -sh ~/f1sim_runs ~/F1tenth/console_main/f1sim; grep -c _tw_cache ~/F1tenth/console_main/f1sim/f1sim/map_controller.py'
echo "$(date +%T) 전송 끝"
