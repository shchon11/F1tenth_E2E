#!/bin/bash
# (11-24, 사용자: "콘솔에서 왜 안보이냐 랩 pc") 콘솔 목록(console_picks.json)이 이 PC에서 바뀌면 1분 안에 연구실 PC로 보냄.
# 전에는 lab_sync.sh가 보낼 때만 갔고, 그 뒤에 띄운 학습(race32c)이 연구실 콘솔에 안 나왔음.
# 연구실에 checkpoint 폴더가 있는 것만 보냄 (없는 줄은 콘솔이 못 엶). 하나만 돌게 함 (이미 돌면 끝).
H=${LAB_H:-shchon11@10.100.0.2}; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
P=/home/shchon11/f1sim_runs/console_picks.json; L=/tmp/lab_picks_sync.lock
exec 9>$L; flock -n 9 || exit 0
last=""
while true; do
  m=$(stat -c %Y $P 2>/dev/null)
  if [ "$m" != "$last" ]; then
    have=$($SSH $H 'ls ~/f1sim_runs' 2>/dev/null | tr '\n' ' ')
    # (11-27) a pick whose folder is only here (pick_hc41a_u800: copied here, never sent) goes over first -- before, such picks
    # were silently left out and the lab console never showed hc41a
    for r in $(python3 -c "import json; print(' '.join(e['run'] for e in json.load(open('$P'))['picks']))"); do
      case " $have " in *" $r "*) continue ;; esac
      [ -d /home/shchon11/f1sim_runs/$r ] && tar -chf - -C /home/shchon11/f1sim_runs $r | $SSH $H 'tar -x -C ~/f1sim_runs' && have="$have $r" && echo "$(date +%T) $r 폴더 보냄"
    done
    if [ -n "$have" ] && python3 - "$have" <<'PY' | $SSH $H 'cat > ~/f1sim_runs/console_picks.json.new && mv ~/f1sim_runs/console_picks.json.new ~/f1sim_runs/console_picks.json'
import json, sys
have = set(sys.argv[1].split())
d = json.load(open("/home/shchon11/f1sim_runs/console_picks.json"))
d["picks"] = [e for e in d["picks"] if e["run"] in have]
print(json.dumps(d, ensure_ascii=False, indent=1))
PY
    then last=$m; echo "$(date +%T) 보냄"; fi
  fi
  sleep 60
done
