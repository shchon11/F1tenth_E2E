"""run별 추이 표 (11-06, 사용자: "추이같은거 좀 제대로 표로 좀 써. 잘 되고있는지 모르겠어 넘 난잡해"). one_<run>.log의 u<N> 줄을 읽어
run 하나에 표 하나: 행 = checkpoint(첫 행 = 시작점), 열 = 빈 트랙 한 바퀴(중간값) / 앞 8 layout 세게 obstacle / 앞 8 layout 접촉(가볍게 포함) /
부딪힌 layout 수, 30 layout과 경주 결과가 있으면 그 checkpoint 행 아래에 따로. 인자: run=시작점lap,시작8세게,시작8접촉,시작8배치 ..."""
import re, sys
import numpy as np

D = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out"


def table(run, start):
    rows = {}
    extra = []
    for l in open(f"{D}/one_{run}.log", errors="replace"):
        m = re.search(r"u(\d+) solo: .*랩 \[([^\]]*)\]", l)
        if m and m.group(2).strip():
            rows.setdefault(int(m.group(1)), {})["lap"] = float(np.median([float(v) for v in m.group(2).split(",")]))
        m = re.search(r"u(\d+) obstacle 앞 8 layout: .*접촉 난 배치 (\d+)/8 \| 벽 \d+\(세게 (\d+)\) 장애물 (\d+)\(세게 (\d+)\)", l)
        if m:
            r = rows.setdefault(int(m.group(1)), {})
            r["hard"], r["touch"], r["bad"], r["whard"] = int(m.group(5)), int(m.group(4)), int(m.group(2)), int(m.group(3))
        m = re.search(r"u(\d+) obstacle 30 layout: .*접촉 난 배치 (\d+)/30 \| 벽 \d+\(세게 (\d+)\) 장애물 \d+\(세게 (\d+)\)", l)
        if m:
            extra.append(f"u{m.group(1)} 30 layout: obstacle 세게 {m.group(4)}번, 벽 세게 {m.group(3)}번, 부딪힌 layout {m.group(2)}/30")
        m = re.search(r"u(\d+) forzaeth x([\d.]+) .*추월 (\d+) / .*차 접촉 \d+ \(세게 (\d+)\)", l)
        if m:
            extra.append(f"u{m.group(1)} 상대 {m.group(2)}배: 추월 {m.group(3)}/20판, 차와 세게 {m.group(4)}번")
    lap0, h0, t0, b0 = start
    out = ["| checkpoint | 빈 트랙 한 바퀴 | obstacle 세게 (앞 8 layout) | 접촉 전부 | 부딪힌 layout |", "|---|---|---|---|---|",
           f"| 시작점 | {lap0} s | {h0} | {t0} | {b0}/8 |"]
    for u in sorted(rows):
        r = rows[u]
        lap = f"{r['lap']:.2f} s" if "lap" in r else "-"
        h = f"{r['hard']}" + (f" (+벽 {r['whard']})" if r.get("whard") else "") if "hard" in r else "-"
        out.append(f"| u{u} | {lap} | {h} | {r.get('touch', '-')} | {str(r['bad']) + '/8' if 'bad' in r else '-'} |")
    return "\n".join(out) + ("\n\n" + "\n".join("- " + e for e in extra) if extra else "")


for arg in sys.argv[1:]:
    run, st = arg.split("=")
    print(f"**{run}**\n\n{table(run, st.split(','))}\n")
